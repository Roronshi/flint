# core/model.py — model wrapper with backend abstraction and persistent state

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional

import config

# Matches turn-separator artifacts that slip through without a leading newline,
# e.g. "How are you? User:" — stripped before returning to the caller.
_TRAILING_TURN_RE = re.compile(
    r'\s*\b(?:User|Human|Assistant)\s*:\s*$',
    re.IGNORECASE,
)
from core.model_backends import DummyBackend, ONNXBackend, RWKVBackend

log = logging.getLogger(__name__)


class GenerationResult:
    def __init__(self, text: str, tokens: int, elapsed: float):
        self.text = text
        self.tokens = tokens
        self.elapsed = elapsed

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.elapsed if self.elapsed > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.tokens} tokens in {self.elapsed:.2f}s ({self.tokens_per_second:.1f} tok/s)"


_BACKENDS = {
    ".pth": RWKVBackend,
    ".onnx": ONNXBackend,
}


class CompanionModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.MODEL_PATH
        self.backend = None
        self.backend_kind = "dummy"
        self.load_error: Optional[str] = None
        self._ensure_dirs()
        self._load_backend()

    @property
    def dummy(self) -> bool:
        return self.backend_kind == "dummy"

    @property
    def state(self):
        return getattr(self.backend, "state", None)

    @state.setter
    def state(self, value):
        if self.backend is not None:
            setattr(self.backend, "state", value)

    def _ensure_dirs(self):
        Path(config.STATE_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.LORA_DIR).mkdir(parents=True, exist_ok=True)

    def _load_backend(self) -> None:
        model_path = str(self.model_path)
        if model_path.lower() == "dummy":
            self.backend = DummyBackend()
            self.backend_kind = "dummy"
            log.warning("MODEL_PATH is set to 'dummy'; using dummy companion model.")
            return
        suffix = Path(model_path).suffix.lower()
        backend_cls = _BACKENDS.get(suffix)
        if backend_cls is None:
            self.backend = DummyBackend()
            self.backend_kind = "dummy"
            self.load_error = f"Unsupported model format: {suffix or '<none>'}"
            log.warning(self.load_error)
            return
        if not os.path.exists(model_path):
            self.backend = DummyBackend()
            self.backend_kind = "dummy"
            self.load_error = f"Model file not found at {model_path}"
            log.warning("%s; using dummy companion model.", self.load_error)
            return
        try:
            backend = backend_cls()
            backend.load(model_path, strategy=config.MODEL_STRATEGY, vocab_path=config.VOCAB_PATH)
            self.backend = backend
            self.backend_kind = backend.backend_kind
            log.info("Loaded %s backend from %s", self.backend_kind, model_path)
        except Exception as exc:
            self.backend = DummyBackend()
            self.backend_kind = "dummy"
            self.load_error = str(exc)
            log.warning("Model load failed (%s) — falling back to dummy backend.", exc)

    def reload(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or self.model_path
        self.load_error = None
        self._load_backend()

    def load_state(self, path: str | None = None) -> bool:
        path = path or config.STATE_FILE
        ok = self.backend.load_state(path)
        if ok:
            log.info("State loaded: %s", path)
        else:
            log.info("No saved state — starting fresh.")
        return ok

    def save_state(self, path: str | None = None):
        self.backend.save_state(path or config.STATE_FILE)

    def reset_state(self):
        self.backend.reset_state()
        log.info("State reset.")

    def prime_system_prompt(self):
        log.info("Priming system prompt...")
        self.encode_context(config.SYSTEM_PROMPT + "\n\n")
        log.info("System prompt primed.")

    def checkpoint_state(self, label: str) -> str:
        path = os.path.join(config.STATE_DIR, f"{config.USER_NAME}_checkpoint_{label}.pt")
        self.save_state(path)
        log.info("Checkpoint saved: %s", path)
        return path

    def stop_generation(self):
        self.backend.stop_generation()

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        text, tokens, elapsed = self.backend.generate(
            prompt,
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            top_p=top_p if top_p is not None else config.TOP_P,
            top_k=top_k if top_k is not None else config.TOP_K,
            stream_callback=stream_callback,
        )
        # Strip leading role-format artifact (RWKV G1 sometimes generates
        # "Answer: " or "Response: " immediately after "Assistant:")
        text = re.sub(r'^(?:Answer|Response)\s*:\s*', '', text, flags=re.IGNORECASE)
        # Strip any trailing turn-separator the stop-sequence check missed
        # (e.g. "How are you? User:" where no newline preceded the marker).
        text = _TRAILING_TURN_RE.sub('', text).rstrip()
        # Also strip the user's name if it appears as a turn marker
        user_marker_re = re.compile(
            r'\s*\b' + re.escape(config.USER_NAME) + r'\s*:\s*$',
            re.IGNORECASE,
        )
        text = user_marker_re.sub('', text).rstrip()
        return GenerationResult(text, tokens, elapsed)

    def generate_stateless(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        state_before = copy.deepcopy(self.state)
        try:
            return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
        finally:
            self.state = state_before

    def encode_context(self, text: str):
        self.backend.encode_context(text)

    def load_lora(self, path: str | None = None) -> bool:
        path = path or config.LORA_ADAPTER
        ok = self.backend.load_lora(path)
        if ok:
            log.info("LoRA applied: %s", path)
        elif self.dummy:
            log.info("Dummy model active — skipping LoRA application.")
        else:
            log.info("No LoRA adapter found or backend does not support LoRA.")
        return ok
