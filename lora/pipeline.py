# lora/pipeline.py — LoRA training pipeline with replay buffer

import os
import json
import random
import shutil
import tempfile
import threading
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import config
from core.session import ConversationDB

if TYPE_CHECKING:
    from core.model_backends.rwkv_backend import RWKVBackend

log = logging.getLogger(__name__)


class LoRAPipeline:
    """
    Manages LoRA fine-tuning of the RWKV model.

    Flow:
    1. Fetch new (unprocessed) sessions from SQLite
    2. Mix in old sessions (replay buffer, 30%)
    3. Convert to training JSONL
    4. Run RWKV-PEFT training subprocess
    5. Back up previous adapter, save new one, log the run

    Note on RWKV-PEFT CLI args:
    Arguments in _run_peft_training() match the RWKV-PEFT repo as of early 2025.
    If training fails with 'unrecognized arguments', verify with:
        python RWKV-PEFT/train.py --help
    """

    # Class-level lock prevents concurrent training runs
    _training_lock = threading.Lock()

    def __init__(self, db: ConversationDB, backend=None, companion_id: str | None = None):
        self.db = db
        self._backend = backend
        self._progress_callback = None
        self._companion_id = companion_id
        Path(config.LORA_DIR).mkdir(parents=True, exist_ok=True)

    def should_run(self) -> tuple[bool, str]:
        """Check whether there is enough new data for a training run."""
        new_sessions = self.db.get_unprocessed_sessions()
        n = len(new_sessions)
        if n < config.LORA_MIN_CONVOS:
            return False, f"Only {n} new sessions (minimum: {config.LORA_MIN_CONVOS})"
        return True, f"{n} new sessions ready"

    def build_training_data(self) -> tuple[list, list]:
        """
        Build training batch with replay buffer.
        Returns (new_session_ids, list_of_segment_dicts).
        """
        new_session_ids = self.db.get_unprocessed_sessions()

        n_replay     = max(1, int(len(new_session_ids) * config.REPLAY_RATIO))
        old_sessions = self.db.get_random_old_sessions(n_replay)

        all_ids = new_session_ids + old_sessions
        random.shuffle(all_ids)

        training_texts = []
        for sid in all_ids:
            text = self.db.get_session_as_training_text(sid)
            if text and len(text.strip()) > 50:
                training_texts.extend(self._split_into_segments(text))

        # Mix in dream (synthetic inner monologue) segments
        dream_ratio = getattr(config, "DREAM_RATIO", 0.25)
        n_dreams    = max(1, int(len(training_texts) * dream_ratio))
        if self._companion_id:
            dream_texts = self.db.get_dream_texts(self._companion_id, limit=n_dreams * 2)
            random.shuffle(dream_texts)
            for text in dream_texts[:n_dreams]:
                if text and len(text.strip()) > 20:
                    training_texts.append({"text": text.strip()})

        random.shuffle(training_texts)

        log.info(
            f"Training data: {len(new_session_ids)} new + "
            f"{len(old_sessions)} replay + {min(n_dreams, len(dream_texts) if self._companion_id else 0)} dreams"
            f" = {len(training_texts)} segments"
        )
        return new_session_ids, training_texts

    def _split_into_segments(self, text: str, max_len: int = None) -> list:
        """
        Split long text into segments at turn boundaries.
        max_len is in tokens (approximated via word count x 0.7).
        """
        max_len   = max_len or config.MAX_SEQ_LEN
        max_words = int(max_len * 0.7)

        lines     = text.split("\n")
        segments  = []
        current: list[str] = []
        current_words = 0

        for line in lines:
            lw = len(line.split())
            if current_words + lw > max_words and current:
                segments.append({"text": "\n".join(current)})
                current       = current[-2:] if len(current) >= 2 else []
                current_words = sum(len(l.split()) for l in current)
            current.append(line)
            current_words += lw

        if current:
            segments.append({"text": "\n".join(current)})

        return segments

    def write_jsonl(self, segments: list, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in segments:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def run(self, dry_run: bool = False) -> bool:
        """
        Run the full pipeline. Returns False immediately if already running.
        dry_run=True validates data without training.
        """
        if not LoRAPipeline._training_lock.acquire(blocking=False):
            log.warning("LoRA training already in progress — skipping.")
            return False
        try:
            return self._run_locked(dry_run)
        finally:
            LoRAPipeline._training_lock.release()

    def _run_locked(self, dry_run: bool) -> bool:
        ok, msg = self.should_run()
        if not ok:
            log.info(f"LoRA training skipped: {msg}")
            return False

        log.info(f"Starting LoRA training: {msg}")
        new_session_ids, segments = self.build_training_data()

        if not segments:
            log.warning("No training segments generated.")
            return False

        if dry_run:
            log.info(f"Dry run: {len(segments)} segments ready. Training not run.")
            return True

        run_id = self.db.begin_training_run(self._companion_id)
        success, train_result = self._run_peft_training(segments, backend=self._backend)
        self.db.update_training_run(
            run_id=run_id,
            steps=train_result.get("steps", 0),
            avg_loss=train_result.get("loss", 0.0),
            min_loss=train_result.get("min_loss", 0.0),
            loss_curve=train_result.get("loss_curve", []),
            success=success,
        )

        self.db.log_lora_run(
            sessions_used=new_session_ids,
            adapter_path=config.LORA_ADAPTER,
            success=success,
            notes=f"{len(segments)} segments"
        )

        if success:
            log.info(f"LoRA training complete. Adapter: {config.LORA_ADAPTER}")
        else:
            log.error("LoRA training failed — check output above.")

        return success

    def _run_peft_training(self, segments: list, backend=None) -> tuple[bool, dict]:
        """
        Fine-tune in-process using RWKVLoRATrainer.
        JIT is disabled globally (RWKV_JIT_ON=0) so model.w is accessible.
        Progress reported via self._progress_callback if set.
        Chat is blocked externally via state.training_active flag.
        Returns (success, result_dict).
        """
        from lora.trainer import RWKVLoRATrainer

        adapter_path = config.LORA_ADAPTER
        backup_path  = adapter_path + ".bak"
        _empty = {"steps": 0, "loss": 0.0, "min_loss": 0.0, "loss_curve": []}

        if backend is None:
            log.error("No backend provided — cannot train in-process.")
            return False, _empty

        if os.path.exists(adapter_path):
            try:
                shutil.copy2(adapter_path, backup_path)
            except Exception as e:
                log.warning(f"Adapter backup failed (continuing): {e}")

        # Free VRAM so the training forward pass has room alongside the adapter weights
        if hasattr(backend, "offload_to_cpu"):
            log.info("Offloading inference model to CPU to free VRAM for training...")
            backend.offload_to_cpu()

        try:
            trainer = RWKVLoRATrainer(
                backend=backend,
                r=config.LORA_R,
                alpha=config.LORA_ALPHA,
                lr=config.LORA_LR,
                epochs=getattr(config, "LORA_EPOCHS", 1),
                device="cuda" if "cuda" in config.MODEL_STRATEGY else "cpu",
                dropout=0.05,
            )
            result = trainer.train(
                segments,
                progress_callback=self._progress_callback,
            )
            if "error" in result:
                log.error("Trainer error: %s", result["error"])
                return False, _empty
            trainer.save_adapter(adapter_path)
            log.info(
                "LoRA training complete — loss: %.4f steps: %d elapsed: %.1fs",
                result["loss"], result["steps"], result["elapsed"],
            )
            return True, result
        except Exception as exc:
            log.error("LoRA training failed: %s", exc, exc_info=True)
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, adapter_path)
                log.info("Restored previous adapter after training error.")
            return False, _empty
        finally:
            # Reload inference model back to GPU regardless of success/failure
            if hasattr(backend, "reload_to_gpu"):
                log.info("Reloading inference model to GPU...")
                backend.reload_to_gpu()
