# lora/trainer.py — Lightweight RWKV-7 G1 fine-tuner
#
# Trains only a small set of adapter parameters on top of frozen base weights.
# No deepspeed, no lightning, no CUDA toolkit — just PyTorch.
#
# Strategy: inject tiny LoRA matrices (A @ B) in parallel with the frozen
# receptance/key/value/output projections in each attention block.
# All base weights are frozen; only the adapter matrices are trained.
#
# Memory strategy: frozen weights live in CPU RAM (_cpu_z). The custom
# _FrozenLinear / _AdapterLinear autograd Functions move each weight to GPU
# for the matmul, then return it to CPU — it is never stored in the autograd
# backward graph, so VRAM usage is dominated by small activations only.

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── LoRA linear layer ─────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen Linear weight with a trainable low-rank delta."""

    def __init__(self, weight: torch.Tensor, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        delta = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scale
        return base + delta

    def merged_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_B @ self.lora_A) * self.scale

    def adapter_state(self) -> dict:
        return {"lora_A": self.lora_A.data, "lora_B": self.lora_B.data}


# ── Memory-efficient matmul helpers ───────────────────────────────────────────
#
# Standard autograd saves every weight tensor used in a matmul for backward
# (to compute grad_input = grad_output @ W.T). For a 32-layer model this means
# ~10 GB of VRAM just for saved weights. These custom Functions keep W on CPU
# and re-fetch it during backward, so VRAM only holds small activations.

class _FrozenLinear(torch.autograd.Function):
    """y = x @ W_cpu  —  W_cpu stays on CPU; never stored in GPU autograd graph."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, W_cpu: torch.Tensor) -> torch.Tensor:
        W = W_cpu.to(x.device, dtype=torch.float32)
        y = x.float() @ W
        ctx.save_for_backward(x, W_cpu)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, W_cpu = ctx.saved_tensors
        W = W_cpu.to(grad_y.device, dtype=torch.float32)
        grad_x = grad_y.float() @ W.t()
        return grad_x, None  # no grad for W_cpu


class _AdapterLinear(torch.autograd.Function):
    """y = x @ (W_cpu + lora_B @ lora_A * scale)  —  W_cpu stays on CPU."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, W_cpu: torch.Tensor,
                lora_B: torch.Tensor, lora_A: torch.Tensor,
                scale: float) -> torch.Tensor:
        dev = x.device
        W = W_cpu.to(dev, dtype=torch.float32) + (lora_B.float() @ lora_A.float()) * scale
        y = x.float() @ W
        ctx.save_for_backward(x, W_cpu, lora_B, lora_A)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, W_cpu, lora_B, lora_A = ctx.saved_tensors
        dev = grad_y.device
        scale = ctx.scale
        lB = lora_B.float()
        lA = lora_A.float()
        W = W_cpu.to(dev, dtype=torch.float32) + (lB @ lA) * scale
        grad_y_f = grad_y.float()
        # Gradient for x
        grad_x = grad_y_f @ W.t()
        # Gradients for lora_B and lora_A
        # y = x @ (W_frozen + lB @ lA * scale)
        # dL/d(lB) = (x.T @ grad_y) @ lA.T * scale   shape: [in, out] @ [out, r] = [in, r]
        # dL/d(lA) = lB.T @ (x.T @ grad_y) * scale   shape: [r, in] @ [in, out] = [r, out]
        grad_W = x.float().t() @ grad_y_f          # [in, out]
        grad_lB = (grad_W @ lA.t()) * scale         # [in, r]
        grad_lA = (lB.t() @ grad_W) * scale         # [r, out]
        return grad_x, None, grad_lB, grad_lA, None


# ── Trainer ───────────────────────────────────────────────────────────────────

class RWKVLoRATrainer:
    """
    Attaches LoRA adapters to a loaded RWKV-7 G1 model and fine-tunes them.

    Parameters
    ----------
    backend     : rwkv_backend instance (has .model and .tokenizer)
    r           : LoRA rank
    alpha       : LoRA alpha (scaling = alpha / r)
    lr          : learning rate
    epochs      : number of passes over training data
    device      : torch device string
    """

    TARGET_LAYERS = ("receptance", "key", "value", "output")

    def __init__(
        self,
        backend,
        r: int = 16,
        alpha: int = 32,
        lr: float = 1e-4,
        epochs: int = 1,
        device: str = "cuda",
        dropout: float = 0.05,
        max_seq_len: int = 64,
    ):
        self.backend = backend
        self.r = r
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self._adapters: dict[str, LoRALinear] = {}
        self._scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ── Adapter injection ─────────────────────────────────────────────────────

    def _get_weight_dict(self):
        """
        Return a flat {name: tensor} dict from the model.
        backend       = RWKVBackend instance
        backend.model = RWKV_x070 instance  (RWKV-7 uses .z, older RWKV uses .w)
        """
        rwkv_model = getattr(self.backend, "model", None)
        if rwkv_model is not None and hasattr(rwkv_model, "z"):
            return rwkv_model.z  # RWKV_x070 (RWKV-7) stores weights in .z
        if rwkv_model is not None and hasattr(rwkv_model, "w"):
            return rwkv_model.w  # older RWKV class stores weights in .w
        raise RuntimeError(
            f"Cannot access model weights: "
            f"backend={type(self.backend).__name__} "
            f"backend.model={type(rwkv_model).__name__ if rwkv_model else 'None'}"
        )

    def _inject_adapters(self):
        """Build LoRALinear adapters for target attention projection weights.

        RWKV-7 G1 .pth keys look like:
            blocks.N.att.receptance.weight  (2560, 2560)
            blocks.N.att.key.weight
            blocks.N.att.value.weight
            blocks.N.att.output.weight

        Frozen weights remain in _cpu_z (CPU RAM). Only the trainable lora_A
        and lora_B matrices are moved to self.device to avoid VRAM overhead.

        Also caches a CPU fp32 copy of all model weights once so that
        _forward_for_training avoids repeated CUDA→CPU copies per step.
        """
        weight_dict = self._get_weight_dict()
        self._adapters = {}

        for key, tensor in weight_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
                continue
            parts = key.split(".")
            if (
                len(parts) == 5
                and parts[0] == "blocks"
                and parts[2] == "att"
                and parts[3] in self.TARGET_LAYERS
                and parts[4] == "weight"
            ):
                name = ".".join(parts[:4])
                # Create adapter with weight on CPU — only lora_A/lora_B go to device
                lora = LoRALinear(
                    tensor.float().cpu(),
                    r=self.r,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                lora.lora_A = nn.Parameter(lora.lora_A.to(self.device))
                lora.lora_B = nn.Parameter(lora.lora_B.to(self.device))
                self._adapters[name] = lora
                log.debug("Injected LoRA adapter: %s", key)

        log.info("Injected %d LoRA adapters", len(self._adapters))

        # Cache all model tensors on CPU (fp32) once
        log.info("Caching model weights on CPU for training...")
        self._cpu_z: dict = {}
        for key, tensor in weight_dict.items():
            if isinstance(tensor, torch.Tensor):
                self._cpu_z[key] = tensor.float().cpu()
        log.info("Weight cache ready (%d tensors)", len(self._cpu_z))

    def _mm(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """
        Compute x @ W[key], keeping W on CPU for the autograd backward pass.
        Applies the LoRA delta if an adapter exists for this weight.
        """
        name = key[:-7] if key.endswith(".weight") else key
        lora = self._adapters.get(name)
        if lora is not None:
            return _AdapterLinear.apply(x, self._cpu_z[key], lora.lora_B, lora.lora_A, lora.scale)
        return _FrozenLinear.apply(x, self._cpu_z[key])

    # ── Forward pass (gradient-enabled, memory-efficient) ────────────────────

    def _forward_for_training(self, token_ids: List[int]) -> torch.Tensor:
        """
        Gradient-enabled forward pass for RWKV-7 G1.

        Mirrors forward_seq from rwkv.model but without torch.no_grad(), so
        that gradients flow through the LoRA-merged attention weight matrices.

        Frozen weights are accessed via _mm() which uses _FrozenLinear /
        _AdapterLinear to prevent them from being stored in the autograd graph.
        This keeps VRAM usage low (only small activations accumulate).

        Returns logits of shape (T, vocab_size).
        """
        model = self.backend.model
        H, N = model.n_head, model.head_size
        dev = self.device

        # Initialise RNN state tensors — fp32 for numerical accuracy (outside autocast)
        n_embd = model.n_embd
        n_layer = model.n_layer
        state_tmix_prev = [
            torch.zeros(n_embd, dtype=torch.float32, device=dev)
            for _ in range(n_layer)
        ]
        state_tmix_wkv = [
            torch.zeros(H, N, N, dtype=torch.float32, device=dev)
            for _ in range(n_layer)
        ]
        state_cmix_prev = [
            torch.zeros(n_embd, dtype=torch.float32, device=dev)
            for _ in range(n_layer)
        ]

        c = self._cpu_z

        with torch.autocast(device_type="cuda", enabled=(dev == "cuda")):
            # Embedding lookup: index into CPU tensor, move result to device
            x = c['emb.weight'][token_ids].to(dev)  # (T, n_embd), fp32
            T = x.shape[0]

            v_first = torch.zeros_like(x)

            for i in range(n_layer):
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                bbb = f'blocks.{i}.'

                # ── Time-mix (attention) ─────────────────────────────────────────
                xx_ln = F.layer_norm(x, (n_embd,),
                                     weight=c[bbb+'ln1.weight'].to(dev),
                                     bias=c[bbb+'ln1.bias'].to(dev))

                sx = torch.cat((state_tmix_prev[i].unsqueeze(0), xx_ln[:-1, :]))
                shift = sx - xx_ln

                xr = xx_ln + shift * c[att+'x_r'].squeeze().to(dev)
                xw = xx_ln + shift * c[att+'x_w'].squeeze().to(dev)
                xk = xx_ln + shift * c[att+'x_k'].squeeze().to(dev)
                xv = xx_ln + shift * c[att+'x_v'].squeeze().to(dev)
                xa = xx_ln + shift * c[att+'x_a'].squeeze().to(dev)
                xg = xx_ln + shift * c[att+'x_g'].squeeze().to(dev)

                # All 2D weight matmuls go through _mm() to keep W on CPU for backward
                r = self._mm(xr, att+'receptance.weight')
                w = torch.tanh(self._mm(xw, att+'w1'))
                w = self._mm(w, att+'w2')
                k = self._mm(xk, att+'key.weight')
                v = self._mm(xv, att+'value.weight')
                a = torch.sigmoid(c[att+'a0'].to(dev) + self._mm(self._mm(xa, att+'a1'), att+'a2'))
                g = torch.sigmoid(self._mm(xg, att+'g1'))
                g = self._mm(g, att+'g2')

                k_k = c[att+'k_k'].squeeze().to(dev)
                k_a = c[att+'k_a'].squeeze().to(dev)
                r_k = c[att+'r_k'].to(dev)

                kk = F.normalize((k * k_k).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
                k  = k * (1 + (a - 1) * k_a)

                if i == 0:
                    v_first = v
                else:
                    v = v + (v_first - v) * torch.sigmoid(
                        c[att+'v0'].to(dev) + self._mm(self._mm(xv, att+'v1'), att+'v2')
                    )

                decay_w = torch.exp(-0.606531 * torch.sigmoid(c[att+'w0'].to(dev) + w))

                state = state_tmix_wkv[i]
                xx_out = torch.zeros(T, H * N, dtype=torch.float32, device=dev)
                for t in range(T):
                    r_, w_, k_, v_, kk_, a_ = r[t], decay_w[t], k[t], v[t], kk[t], a[t]
                    vk = v_.float().view(H, N, 1) @ k_.float().view(H, 1, N)
                    ab = (-kk_).float().view(H, N, 1) @ (kk_ * a_).float().view(H, 1, N)
                    s_prev = state.detach()
                    state = s_prev * w_.float().view(H, 1, N) + s_prev @ ab + vk
                    xx_out[t] = (state @ r_.float().view(H, N, 1)).view(H * N)

                state_tmix_wkv[i] = state.detach()
                state_tmix_prev[i] = xx_ln[-1, :].float().detach()

                xx_out = F.group_norm(xx_out.view(T, H * N), num_groups=H,
                                      weight=c[att+'ln_x.weight'].to(dev),
                                      bias=c[att+'ln_x.bias'].to(dev),
                                      eps=64e-5).view(T, H * N)
                rk_term = ((r * k * r_k).view(T, H, N).sum(dim=-1, keepdim=True) * v.view(T, H, N)).view(T, H * N)
                xx_out = xx_out + rk_term
                att_out = self._mm(xx_out * g, att+'output.weight')
                x = x + att_out

                # ── Channel-mix (FFN) ────────────────────────────────────────────
                xx_ln2 = F.layer_norm(x, (n_embd,),
                                      weight=c[bbb+'ln2.weight'].to(dev),
                                      bias=c[bbb+'ln2.bias'].to(dev))

                sx2 = torch.cat((state_cmix_prev[i].unsqueeze(0), xx_ln2[:-1, :]))
                k_ffn = xx_ln2 + (sx2 - xx_ln2) * c[ffn+'x_k'].squeeze().to(dev)
                k_ffn = torch.relu(self._mm(k_ffn, ffn+'key.weight')) ** 2
                ffn_out = self._mm(k_ffn, ffn+'value.weight')
                state_cmix_prev[i] = xx_ln2[-1, :].float().detach()
                x = x + ffn_out

            x = F.layer_norm(x, (n_embd,), weight=c['ln_out.weight'].to(dev), bias=c['ln_out.bias'].to(dev))
            logits = self._mm(x, 'head.weight')
        return logits  # (T, vocab_size)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(self, token_ids: List[int]) -> torch.Tensor:
        if len(token_ids) < 2:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        logits = self._forward_for_training(token_ids)  # (T, vocab)
        # Predict next token: input[:-1] → target[1:]
        shift_logits = logits[:-1]
        shift_labels = torch.tensor(token_ids[1:], dtype=torch.long, device=self.device)
        return F.cross_entropy(shift_logits, shift_labels)

    # ── Tokenise training segments ───────────────────────────────────────────

    def _tokenize_segments(self, segments: List[dict]) -> List[List[int]]:
        from rwkv.utils import PIPELINE
        pipeline = PIPELINE(self.backend.model, "rwkv_vocab_v20230424")
        result = []
        for seg in segments:
            text = seg.get("text", "")
            if not text.strip():
                continue
            ids = pipeline.encode(text)
            if len(ids) >= 4:
                result.append(ids)
        return result

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, segments: List[dict], progress_callback=None) -> dict:
        """
        Fine-tune on segments (list of {"text": str}).
        progress_callback(epoch, step, total_steps, loss) called after each step.
        Returns {"loss": float, "steps": int, "elapsed": float}.
        """
        self._inject_adapters()
        if not self._adapters:
            return {"loss": 0.0, "steps": 0, "elapsed": 0.0, "error": "No adapters injected"}

        params = []
        for lora in self._adapters.values():
            params.extend([lora.lora_A, lora.lora_B])

        optimizer = torch.optim.Adam(params, lr=self.lr)

        log.info("Tokenising %d segments...", len(segments))
        token_seqs = self._tokenize_segments(segments)
        if not token_seqs:
            return {"loss": 0.0, "steps": 0, "elapsed": 0.0, "error": "No valid segments"}
        log.info("Training on %d sequences for %d epoch(s)", len(token_seqs), self.epochs)

        total_loss = 0.0
        steps = 0
        min_loss = float("inf")
        loss_curve: List[float] = []  # sampled every N steps for DB storage
        t0 = time.time()

        for epoch in range(self.epochs):
            import random
            random.shuffle(token_seqs)
            epoch_loss = 0.0
            total_seq = len(token_seqs)
            # Sample ~50 points across all steps for the curve
            curve_interval = max(1, (total_seq * self.epochs) // 50)
            for step_i, ids in enumerate(token_seqs):
                ids = ids[:self.max_seq_len]
                optimizer.zero_grad()
                loss = self._compute_loss(ids)
                if loss.requires_grad:
                    if self._scaler:
                        self._scaler.scale(loss).backward()
                        self._scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        self._scaler.step(optimizer)
                        self._scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimizer.step()
                loss_val = loss.item()
                epoch_loss += loss_val
                if loss_val < min_loss:
                    min_loss = loss_val
                steps += 1
                if steps % curve_interval == 0:
                    loss_curve.append(round(loss_val, 4))
                if progress_callback:
                    progress_callback(epoch + 1, step_i + 1, total_seq, loss_val)
            avg = epoch_loss / max(len(token_seqs), 1)
            log.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, self.epochs, avg)
            total_loss += epoch_loss

        elapsed = time.time() - t0
        avg_loss = total_loss / max(steps, 1)
        if min_loss == float("inf"):
            min_loss = avg_loss
        log.info("Training done in %.1fs — avg loss: %.4f over %d steps", elapsed, avg_loss, steps)

        # Free CPU weight cache to release RAM
        self._cpu_z = {}

        return {
            "loss": avg_loss,
            "min_loss": min_loss,
            "steps": steps,
            "elapsed": elapsed,
            "loss_curve": loss_curve,
        }

    # ── Save / load adapter ───────────────────────────────────────────────────

    def save_adapter(self, path: str):
        """Save adapter matrices to a .pt file."""
        state = {name: lora.adapter_state() for name, lora in self._adapters.items()}
        import os as _os
        _os.makedirs(_os.path.dirname(path) if _os.path.dirname(path) else ".", exist_ok=True)
        meta = {"r": self.r, "alpha": self.alpha, "version": 1}
        torch.save({"meta": meta, "adapters": state}, path)
        log.info("Adapter saved to %s (%d layers)", path, len(self._adapters))

    @staticmethod
    def load_adapter(backend, path: str) -> bool:
        """
        Load and apply a saved adapter to a backend's weight dict.
        Returns True on success.
        """
        if not Path(path).exists():
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            meta = checkpoint.get("meta", {})
            r = meta.get("r", 16)
            alpha = meta.get("alpha", 32)
            scale = alpha / r
            model = backend.model
            z = model.z if hasattr(model, "z") else model.w
            applied = 0
            for name, state in checkpoint.get("adapters", {}).items():
                key = name + ".weight"
                if key not in z:
                    continue
                lora_A = state["lora_A"].to(z[key].device)
                lora_B = state["lora_B"].to(z[key].device)
                z[key] = (z[key].float() + (lora_B @ lora_A) * scale).to(z[key].dtype)
                applied += 1
            log.info("Applied %d LoRA adapter layers from %s", applied, path)
            return applied > 0
        except Exception as exc:
            log.error("Failed to load adapter: %s", exc)
            return False


# ── CLI entry point (called as subprocess by pipeline.py) ────────────────────

if __name__ == "__main__":
    import os
    import sys
    import json
    import argparse
    import logging

    # JIT must be off before rwkv.model is imported
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ.setdefault("RWKV_V7_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Flint LoRA trainer")
    parser.add_argument("--model",       required=True,  help="Path to .pth model (without extension)")
    parser.add_argument("--strategy",    default="cuda fp16")
    parser.add_argument("--data",        required=True,  help="Path to JSONL training file")
    parser.add_argument("--adapter_out", required=True,  help="Where to save the adapter .pt file")
    parser.add_argument("--r",           type=int,   default=16)
    parser.add_argument("--alpha",       type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--adapter_in",  default="",     help="Existing adapter to continue from")
    args = parser.parse_args()

    from rwkv.model import RWKV_x070 as RWKV
    from rwkv.utils import PIPELINE

    log.info("Loading model: %s (%s)", args.model, args.strategy)
    model = RWKV(model=args.model, strategy=args.strategy)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    # Load training data
    segments = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    log.info("Loaded %d segments from %s", len(segments), args.data)

    # Simple backend shim so RWKVLoRATrainer can access model.w
    class _Backend:
        def __init__(self, m, p):
            self.model = m
            self.pipeline = p

    backend = _Backend(model, pipeline)
    device = "cuda" if "cuda" in args.strategy else "cpu"

    trainer = RWKVLoRATrainer(
        backend=backend,
        r=args.r,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
        dropout=0.05,
    )

    # Load existing adapter if provided
    if args.adapter_in and os.path.exists(args.adapter_in):
        log.info("Loading existing adapter: %s", args.adapter_in)
        RWKVLoRATrainer.load_adapter(backend, args.adapter_in)

    result = trainer.train(segments)
    if "error" in result:
        log.error("Training error: %s", result["error"])
        sys.exit(1)

    trainer.save_adapter(args.adapter_out)
    log.info("Done — loss=%.4f steps=%d elapsed=%.1fs", result["loss"], result["steps"], result["elapsed"])
    sys.exit(0)
