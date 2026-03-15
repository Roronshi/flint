# services/backup_service.py — Automatic backup of relation-critical files

from __future__ import annotations

import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import config

log = logging.getLogger(__name__)

_BACKUP_DIR = Path(config.DATA_DIR) / "backups"
_MAX_BACKUPS = 7


class BackupService:
    """
    Backs up the three files that constitute the entire Flint relationship:
      - RWKV state  (.pt)
      - LoRA adapter (.pth)
      - SQLite DB   (.db)

    Called automatically after each successful LoRA training run,
    and available as a manual trigger via API.
    """

    def run_backup(self, companion_id: str | None = None) -> Dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = _BACKUP_DIR / ts
        dest.mkdir(parents=True, exist_ok=True)

        backed_up: List[str] = []

        for src_path in (config.STATE_FILE, config.LORA_ADAPTER, config.CONVERSATIONS_DB):
            src = Path(src_path)
            if src.exists():
                shutil.copy2(src, dest / src.name)
                backed_up.append(src.name)
            else:
                log.debug("Backup: %s not found, skipping.", src)

        self._prune()

        log.info("Backup complete → %s (%s)", dest, ", ".join(backed_up) or "no files")
        return {"path": str(dest), "timestamp": ts, "files": backed_up}

    def list_backups(self) -> List[Dict[str, Any]]:
        if not _BACKUP_DIR.exists():
            return []
        entries = []
        for d in sorted(_BACKUP_DIR.iterdir(), reverse=True):
            if d.is_dir():
                files = [f.name for f in d.iterdir()]
                entries.append({"timestamp": d.name, "path": str(d), "files": files})
        return entries

    def make_zip(self, timestamp: str) -> Path | None:
        """Create a zip archive of the backup for download. Returns zip path."""
        dest = _BACKUP_DIR / timestamp
        if not dest.exists():
            return None
        zip_path = _BACKUP_DIR / f"{timestamp}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dest.iterdir():
                zf.write(f, arcname=f.name)
        return zip_path

    def _prune(self):
        """Keep only the most recent _MAX_BACKUPS backups."""
        if not _BACKUP_DIR.exists():
            return
        dirs = sorted(
            [d for d in _BACKUP_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        for old in dirs[_MAX_BACKUPS:]:
            shutil.rmtree(old, ignore_errors=True)
            log.debug("Pruned old backup: %s", old)
