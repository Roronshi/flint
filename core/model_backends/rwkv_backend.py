from __future__ import annotations

import os
import threading
import time
import torch
from typing import Callable, Optional

import config
from .base_backend import BaseModelBackend

# RWKV-7 G1 models require RWKV_V7_ON=1 to activate the RWKV_x070 codepath
# in the rwkv pip package.  Set before importing rwkv.model.
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ["RWKV_JIT_ON"] = "0"  # JIT off — required for in-process LoRA training (.w access)
os.environ.setdefault("RWKV_CUDA_ON", "0")  # pure-Python fallback; safe on all hardware

try:
    from rwkv.model import RWKV_x070 as RWKV  # type: ignore  # RWKV-7 G1 class
    from rwkv.utils import PIPELINE  # type: ignore
except Exception:  # pragma: no cover
    RWKV = None
    PIPELINE = None


class RWKVBackend(BaseModelBackend):
    backend_kind = "rwkv"
    supports_reasoning_mode = True
    supports_persistent_state = True
    supports_adapter_training = True

    def __init__(self) -> None:
        self.model = None
        self.pipeline = None
        self.state = None
        self._stop_event = threading.Event()

    def load(self, path: str, strategy: str | None = None, vocab_path: str | None = None) -> None:
        if RWKV is None or PIPELINE is None:
            raise RuntimeError("rwkv package is not installed")
        # RWKV_x070 appends '.pth' internally — strip it if already present
        model_path = path[:-4] if path.endswith(".pth") else path
        self._model_path   = path
        self._model_strategy = strategy
        self._vocab_path   = vocab_path
        self.model = RWKV(model=model_path, strategy=strategy or "cuda fp16")
        self.pipeline = PIPELINE(self.model, vocab_path)
        # RWKV_x070 derives n_head/head_size internally from the r_k weight shape.

    def offload_to_cpu(self) -> None:
        """Move model weights to CPU and free GPU memory. Used before LoRA subprocess."""
        import gc
        import torch
        if self.model is None:
            return
        try:
            # Move all tensors in model.w to CPU
            if hasattr(self.model, "w"):
                for k in self.model.w:
                    if hasattr(self.model.w[k], "to"):
                        self.model.w[k] = self.model.w[k].cpu()
            # Also move state to CPU if present
            if self.state is not None:
                self.state = [s.cpu() if hasattr(s, "cpu") else s for s in self.state]
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    def reload_to_gpu(self) -> None:
        """Reload model from disk back onto GPU after LoRA subprocess completes."""
        if not hasattr(self, "_model_path") or self._model_path is None:
            return
        self.load(self._model_path, self._model_strategy, self._vocab_path)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.85,
        top_k: int = 0,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, int, float]:
        if self.model is None or self.pipeline is None:
            raise RuntimeError("RWKV backend not loaded")
        self._stop_event.clear()
        # Encode stop sequences once — prevents model from writing the user's next turn
        stop_ids_list = [self.pipeline.encode(s) for s in config.STOP_SEQUENCES]
        prompt_tokens = self.pipeline.encode(prompt)
        logits = None
        while prompt_tokens:
            chunk = prompt_tokens[:256]
            prompt_tokens = prompt_tokens[256:]
            logits, self.state = self.model.forward(chunk, self.state)
        if logits is None:
            return "", 0, 0.0
        generated: list[int] = []
        occurrence: dict[int, int] = {}
        out_last = 0
        out_str = ""
        t0 = time.perf_counter()
        for _ in range(max_tokens):
            if self._stop_event.is_set():
                break
            for uid, cnt in occurrence.items():
                logits[uid] -= config.ALPHA_PRESENCE + cnt * config.ALPHA_FREQUENCY
            logits[0] = -float("inf")
            token = self.pipeline.sample_logits(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
            )
            generated.append(token)
            occurrence[token] = occurrence.get(token, 0) + 1
            decoded = self.pipeline.decode(generated[out_last:])
            if "\ufffd" not in decoded and decoded:
                if stream_callback:
                    stream_callback(decoded)
                out_str += decoded
                out_last = len(generated)
            # Stop sequence detection
            stop_hit = False
            for stop_ids in stop_ids_list:
                n = len(stop_ids)
                if len(generated) >= n and generated[-n:] == stop_ids:
                    stop_str = self.pipeline.decode(stop_ids)
                    if out_str.endswith(stop_str):
                        out_str = out_str[:-len(stop_str)]
                    stop_hit = True
                    break
            if stop_hit:
                break
            logits, self.state = self.model.forward([token], self.state)
        if out_last < len(generated):
            tail = self.pipeline.decode(generated[out_last:])
            if tail and "\ufffd" not in tail:
                if stream_callback:
                    stream_callback(tail)
                out_str += tail
        elapsed = time.perf_counter() - t0
        return out_str.strip(), len(generated), elapsed

    def encode_context(self, text: str) -> None:
        if self.model is None or self.pipeline is None:
            return
        tokens = self.pipeline.encode(text)
        while tokens:
            chunk = tokens[:256]
            tokens = tokens[256:]
            _, self.state = self.model.forward(chunk, self.state)

    def stop_generation(self) -> None:
        self._stop_event.set()

    def reset_state(self) -> None:
        self.state = None

    def save_state(self, path: str) -> None:
        if self.state is None:
            return
        tmp = path + ".tmp"
        torch.save(self.state, tmp)
        os.replace(tmp, path)

    def load_state(self, path: str) -> bool:
        if os.path.exists(path):
            self.state = torch.load(path, map_location="cpu", weights_only=False)
            return True
        self.state = None
        return False

    def load_lora(self, path: str) -> bool:
        if not os.path.exists(path) or self.model is None:
            return False
        # RWKV-PEFT is a training-scripts repo and does not expose a Python API
        # for runtime adapter loading.  LoRA fine-tuning runs as a nightly
        # subprocess (lora/pipeline.py) and produces updated base weights;
        # runtime hot-loading is not currently supported.
        import logging
        logging.getLogger(__name__).info(
            "load_lora: runtime LoRA application not supported — "
            "adapter is applied during nightly training, not at runtime."
        )
        return False
