# lora/trainer.py — Lightweight RWKV-7 G1 fine-tuner
#
# Trains only a small set of adapter parameters on top of frozen base weights.
# No deepspeed, no lightning, no CUDA toolkit — just PyTorch.
#
# Strategy: inject tiny LoRA matrices (A @ B) in parallel with the frozen
# receptance/key/value/output projections in each attention block.
# All base weights are frozen; only the adapter matrices are trained.

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


# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

class _Tokenizer:
    def __init__(self, backend):
        self._backend = backend

    def encode(self, text: str) -> List[int]:
        w = self._backend.w
        if hasattr(w, "tokenizer"):
            return w.tokenizer.encode(text)
        # RWKV world tokenizer lives on the model object
        from rwkv.utils import PIPELINE
        pipeline = PIPELINE(self._backend.model, "rwkv_vocab_v20230424")
        return pipeline.encode(text)


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
    ):
        self.backend = backend
        self.r = r
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.dropout = dropout
        self._adapters: dict[str, LoRALinear] = {}

    # ── Adapter injection ─────────────────────────────────────────────────────

    def _inject_adapters(self):
        """Replace target Linear layers with LoRALinear wrappers."""
        model_weights = self.backend.model.w  # OrderedDict of tensors
        self._adapters = {}

        # Scan weight keys for attention projection matrices
        for key, tensor in model_weights.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            parts = key.split(".")
            # Match: blocks.N.att.receptance.weight etc.
            if (
                len(parts) >= 4
                and parts[0] == "blocks"
                and parts[2] == "att"
                and parts[3] in self.TARGET_LAYERS
                and tensor.dim() == 2
            ):
                lora = LoRALinear(
                    tensor.to(self.device),
                    r=self.r,
                    alpha=self.alpha,
                    dropout=self.dropout,
                ).to(self.device)
                self._adapters[key.replace(".weight", "")] = lora
                log.debug("Injected LoRA adapter: %s", key)

        log.info("Injected %d LoRA adapters", len(self._adapters))

    # ── Forward pass (manual, stateless) ────────────────────────────────────

    def _forward_with_adapters(self, token_ids: List[int]) -> torch.Tensor:
        """
        Run a stateless forward pass through the RWKV pip package model,
        patching the weight dict with merged LoRA weights for the loss step.
        """
        w = self.backend.model.w
        # Temporarily merge adapter weights into the weight dict
        original = {}
        for name, lora in self._adapters.items():
            key = name + ".weight"
            original[key] = w[key]
            w[key] = lora.merged_weight()

        try:
            tokens = torch.tensor(token_ids, dtype=torch.long).to(self.device)
            # Use RWKV model's forward — returns logits for each token
            logits, _ = self.backend.model.forward(tokens, None)
        finally:
            # Restore original weights
            for key, orig in original.items():
                w[key] = orig

        return logits  # shape: (T, vocab)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(self, token_ids: List[int]) -> torch.Tensor:
        if len(token_ids) < 2:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        logits = self._forward_with_adapters(token_ids)  # (T, vocab)
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

    def train(self, segments: List[dict]) -> dict:
        """
        Fine-tune on segments (list of {"text": str}).
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
        t0 = time.time()

        for epoch in range(self.epochs):
            import random
            random.shuffle(token_seqs)
            epoch_loss = 0.0
            for ids in token_seqs:
                optimizer.zero_grad()
                loss = self._compute_loss(ids)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            avg = epoch_loss / max(len(token_seqs), 1)
            log.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, self.epochs, avg)
            total_loss += epoch_loss

        elapsed = time.time() - t0
        avg_loss = total_loss / max(steps, 1)
        log.info("Training done in %.1fs — avg loss: %.4f over %d steps", elapsed, avg_loss, steps)
        return {"loss": avg_loss, "steps": steps, "elapsed": elapsed}

    # ── Save / load adapter ───────────────────────────────────────────────────

    def save_adapter(self, path: str):
        """Save adapter matrices to a .pt file."""
        state = {name: lora.adapter_state() for name, lora in self._adapters.items()}
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
            w = backend.model.w
            applied = 0
            for name, state in checkpoint.get("adapters", {}).items():
                key = name + ".weight"
                if key not in w:
                    continue
                lora_A = state["lora_A"].to(w[key].device)
                lora_B = state["lora_B"].to(w[key].device)
                w[key] = w[key] + (lora_B @ lora_A) * scale
                applied += 1
            log.info("Applied %d LoRA adapter layers from %s", applied, path)
            return applied > 0
        except Exception as exc:
            log.error("Failed to load adapter: %s", exc)
            return False
