# lora/scheduler.py — Nightly automatic LoRA training

import json
import logging
import os
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import config
from core.session import ConversationDB
from lora.pipeline import LoRAPipeline
from services.backup_service import BackupService

log = logging.getLogger(__name__)

# Persist last-run date so a server restart during the 03:00 window doesn't
# trigger a duplicate training run.
_LAST_RUN_FILE = os.path.join(config.DATA_DIR, "lora_last_run.json")


class LoRAScheduler:
    """
    Runs LoRA training automatically at the configured time (default 03:00).
    Runs as a daemon thread so the chat is never blocked.
    """

    def __init__(self, db: ConversationDB, companion_id: str | None = None):
        self.db           = db
        self.companion_id = companion_id
        self.pipeline     = LoRAPipeline(db, backend=None, companion_id=companion_id)  # backend set later via set_backend()
        self._backup      = BackupService()
        self._thread: threading.Thread = None
        self._stop_event = threading.Event()
        self.last_run: Optional[datetime] = self._load_last_run()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_last_run(self) -> Optional[datetime]:
        """Load last run timestamp from disk to survive server restarts."""
        try:
            with open(_LAST_RUN_FILE) as f:
                ts = json.load(f).get("last_run")
                if ts:
                    return datetime.fromisoformat(ts)
        except (FileNotFoundError, Exception):
            pass
        return None

    def _save_last_run(self, dt: datetime):
        try:
            Path(_LAST_RUN_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(_LAST_RUN_FILE, "w") as f:
                json.dump({"last_run": dt.isoformat()}, f)
        except Exception as e:
            log.warning(f"Could not persist last_run: {e}")

    # ── Control ────────────────────────────────────────────────────────────────

    def set_backend(self, backend):
        """Attach the live model backend so the trainer can access model weights."""
        self.pipeline._backend = backend

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="lora-scheduler"
        )
        self._thread.start()
        log.info(f"LoRA scheduler started (runs daily at {config.LORA_SCHEDULE})")

    def stop(self):
        self._stop_event.set()

    def run_now(self) -> threading.Thread:
        """Trigger training immediately (e.g. via /lora now)."""
        log.info("Manual LoRA training triggered.")
        t = threading.Thread(
            target=self._run_training, daemon=True, name="lora-manual"
        )
        t.start()
        return t

    # ── Loop ───────────────────────────────────────────────────────────────────

    def _loop(self):
        target_hour, target_min = map(int, config.LORA_SCHEDULE.split(":"))

        while not self._stop_event.is_set():
            now = datetime.now()

            already_ran_today = (
                self.last_run is not None
                and self.last_run.date() >= date.today()
            )

            if (
                now.hour == target_hour
                and now.minute == target_min
                and not already_ran_today
            ):
                self._run_training()

            self._stop_event.wait(30)

    def _run_training(self):
        log.info("Starting LoRA training run...")
        try:
            success       = self.pipeline.run()
            self.last_run = datetime.now()
            self._save_last_run(self.last_run)
            log.info(f"LoRA training {'complete' if success else 'failed'}.")
            if success:
                try:
                    result = self._backup.run_backup(self.companion_id)
                    log.info("Post-training backup: %s", result["files"])
                except Exception as be:
                    log.warning("Backup after training failed: %s", be)
        except Exception as e:
            log.error(f"Uncaught error in LoRA scheduler: {e}", exc_info=True)

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> str:
        ok, msg  = self.pipeline.should_run()
        last_str = self.last_run.strftime("%Y-%m-%d %H:%M") if self.last_run else "Never"
        return (
            f"Next run: {config.LORA_SCHEDULE} | "
            f"Last run: {last_str} | "
            f"Data: {msg}"
        )
