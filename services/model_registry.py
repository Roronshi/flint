from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from core.session import ConversationDB


class ModelRegistryService:
    SUPPORTED_EXTENSIONS = {
        ".pth": {
            "engine_type": "rwkv_pth",
            "family": "rwkv",
            "supports_adapter_training": True,
            "supports_persistent_state": True,
            "supports_reasoning_mode": True,
        },
        ".onnx": {
            "engine_type": "rwkv_onnx",
            "family": "rwkv",
            "supports_adapter_training": False,
            "supports_persistent_state": True,
            "supports_reasoning_mode": True,
        },
    }

    def __init__(self, db: ConversationDB):
        self.db = db
        self.models_dir = Path(config.BASE_DIR) / "models"

    def scan_models(self) -> List[Dict[str, Any]]:
        self.models_dir.mkdir(exist_ok=True)
        active_name = Path(config.MODEL_PATH).name
        entries: List[Dict[str, Any]] = []

        for ext, meta in self.SUPPORTED_EXTENSIONS.items():
            for file_path in sorted(self.models_dir.glob(f"*{ext}")):
                stat = file_path.stat()
                model_id = self._model_id_from_path(file_path)
                manifest = {
                    "path": str(file_path),
                    "filename": file_path.name,
                    "size_mb": round(stat.st_size / 1_048_576, 1),
                    "engine_type": meta["engine_type"],
                    "family": meta["family"],
                    "format": ext.lstrip("."),
                    "active": file_path.name == active_name,
                    "official": file_path.name.lower().startswith("rwkv7-g1") or "rwkv7-g1" in file_path.name.lower(),
                    "readiness": "ready",
                    "supports_persistent_state": meta["supports_persistent_state"],
                    "supports_adapter_training": meta["supports_adapter_training"],
                    "supports_reasoning_mode": meta["supports_reasoning_mode"],
                }
                self.db.upsert_model(
                    model_id=model_id,
                    engine_type=meta["engine_type"],
                    family=meta["family"],
                    name=file_path.stem,
                    version=None,
                    manifest_json=json.dumps(manifest),
                )
                self.db.upsert_model_installation(
                    model_id=model_id,
                    install_path=str(file_path),
                    backend=config.MODEL_STRATEGY,
                    is_default=file_path.name == active_name,
                    verification_status="ok",
                )
                entries.append({"id": model_id, **manifest})

        # Always ensure the configured MODEL_PATH has a DB record so new_session()
        # can satisfy its FK constraint even when the file isn't downloaded yet.
        configured_id = self._model_id_from_path(Path(config.MODEL_PATH))
        if not any(e["id"] == configured_id for e in entries):
            dummy_path = config.MODEL_PATH
            dummy_name = Path(dummy_path).stem or "dummy"
            manifest = {
                "path": dummy_path,
                "filename": Path(dummy_path).name or "dummy",
                "size_mb": 0.0,
                "engine_type": "dummy",
                "family": "rwkv",
                "format": "dummy",
                "active": True,
                "official": False,
                "readiness": "dummy",
                "supports_persistent_state": False,
                "supports_adapter_training": False,
                "supports_reasoning_mode": False,
            }
            self.db.upsert_model(
                model_id=configured_id,
                engine_type="dummy",
                family="rwkv",
                name=dummy_name,
                version=None,
                manifest_json=json.dumps(manifest),
            )
            self.db.upsert_model_installation(
                model_id=configured_id,
                install_path=dummy_path,
                backend=config.MODEL_STRATEGY,
                is_default=not entries,
                verification_status="unverified",
            )
            entries.append({"id": configured_id, **manifest})
        return entries

    def active_model_id(self) -> str:
        return self._model_id_from_path(Path(config.MODEL_PATH))

    def active_model_info(self) -> Optional[Dict[str, Any]]:
        for entry in self.scan_models():
            if entry["id"] == self.active_model_id():
                return entry
        return None

    @staticmethod
    def _model_id_from_path(file_path: Path) -> str:
        stem = file_path.stem.lower().replace(" ", "-")
        return f"rwkv::{stem or 'dummy'}"
