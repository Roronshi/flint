# web/server.py — FastAPI backend with WebSocket streaming and Flint background services

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.app_state import AppState
from core.model import CompanionModel, GenerationResult
from core.session import ConversationDB, Session
from lora.scheduler import LoRAScheduler
from services.chat_service import ChatService
from services.model_registry import ModelRegistryService
from services.reflection_service import ReflectionService
from services.scheduler_service import SchedulerService
from services.state_service import StateService
from services.idle_reasoning import IdleReasoningService
from services.model_presets import G1_PRESETS, recommend_preset
from services.training_presets import TRAINING_PRESETS, default_training_preset

log = logging.getLogger(__name__)

_VERSION_FILE = Path(__file__).parent.parent / "VERSION"


def _read_version() -> str:
    try:
        return _VERSION_FILE.read_text().strip()
    except Exception:
        return "0.0.0"


APP_VERSION = _read_version()
state = AppState()
chat_service: ChatService | None = None
model_registry: ModelRegistryService | None = None
reflection_service: ReflectionService | None = None
background_scheduler: SchedulerService | None = None
state_service: StateService | None = None
idle_reasoning_service: IdleReasoningService | None = None


def _detect_vram_gb() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory / (1024**3))
    except Exception:
        return None
    return None


def _current_training_profile() -> dict:
    cfg_method = getattr(config, "TRAINING_METHOD", "").strip() if hasattr(config, "TRAINING_METHOD") else ""
    cfg_preset = getattr(config, "TRAINING_PRESET", "").strip() if hasattr(config, "TRAINING_PRESET") else ""
    if cfg_preset:
        for item in TRAINING_PRESETS:
            if item["id"] == cfg_preset:
                payload = dict(item)
                if cfg_method:
                    payload["method"] = cfg_method
                return payload
    item = dict(default_training_preset())
    if cfg_method:
        item["method"] = cfg_method
    return item


def _is_official_g1_model(filename: str | None) -> bool:
    if not filename:
        return False
    name = filename.lower()
    return name.startswith("rwkv7-g1") or "rwkv7-g1" in name



def _format_job_health(jobs: dict, key: str) -> dict:
    job = jobs.get(key, {}) if jobs else {}
    return {
        "last_success": job.get("last_success"),
        "last_error": job.get("last_error"),
        "run_count": job.get("run_count", 0),
        "last_duration_ms": job.get("last_duration_ms"),
    }


def _latest_snapshot_meta() -> dict:
    if not state_service or not state.companion_id:
        return {}
    snap = state_service.latest_runtime_snapshot(state.companion_id, state.active_model_id)
    if not snap:
        return {}
    return {
        "latest_snapshot_id": snap.get("id"),
        "latest_snapshot_path": snap.get("snapshot_path"),
        "latest_snapshot_created_at": snap.get("created_at"),
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_service, model_registry, reflection_service, background_scheduler, state_service, idle_reasoning_service

    log.info("Initialising Flint backend...")
    loop = asyncio.get_running_loop()

    state.db = ConversationDB()
    model_registry = ModelRegistryService(state.db)
    available_models = model_registry.scan_models()
    state.companion_id = state.db.get_or_create_default_companion()
    state.active_model_id = model_registry.active_model_id()

    state.model = await loop.run_in_executor(None, CompanionModel)
    await loop.run_in_executor(None, state.model.load_lora)
    has_state = await loop.run_in_executor(None, state.model.load_state)
    if not has_state:
        await loop.run_in_executor(None, state.model.prime_system_prompt)

    state.active_session = Session(
        state.db,
        companion_id=state.companion_id,
        model_id=state.active_model_id,
    )
    state.scheduler = LoRAScheduler(state.db)
    state.scheduler.start()

    chat_service = ChatService(state)
    reflection_service = ReflectionService(state.db)
    background_scheduler = SchedulerService(state.db)

    # State service for runtime snapshots and adapter version management
    state_service = StateService(state.db, state.model)
    idle_reasoning_service = IdleReasoningService(state.db, state.model, reflection_service)
    background_scheduler.register_job(
        "conversation_ingest_job",
        120,
        lambda: reflection_service.ingest_conversation_blocks(state.companion_id, state.active_model_id),
    )
    background_scheduler.register_job(
        "recent_summary_job",
        900,
        lambda: reflection_service.summarize_recent_blocks(state.companion_id, state.active_model_id),
    )
    background_scheduler.register_job(
        "period_synthesis_job",
        21600,
        lambda: reflection_service.synthesize_recent_period(state.companion_id, state.active_model_id),
    )
    background_scheduler.register_job(
        "semantic_memory_refresh_job",
        86400,
        lambda: reflection_service.refresh_semantic_memory(state.companion_id, state.active_model_id),
    )

    # Periodically save a runtime snapshot.  This runs every 12 hours by default
    background_scheduler.register_job(
        "runtime_snapshot_job",
        43200,
        lambda: state_service.save_runtime_snapshot(state.companion_id, state.active_model_id) if state_service else 0,
    )

    # Schedule regular reflection generation and gating jobs.  These jobs
    # build reflection candidates from recent summaries, gate them
    # against the active initiative profile, and render any approved
    # outreach items.  The interval of one hour is a reasonable
    # default; more aggressive profiles can trigger manual runs via
    # /api/reflect/run.
    def reflection_cycle_job() -> int:
        profile = state.db.get_active_initiative_profile(state.companion_id)
        if not profile:
            return 0
        created = reflection_service.generate_reflections(state.companion_id, state.active_model_id, profile)
        if created:
            reflection_service.gate_reflections(state.companion_id, profile)
            reflection_service.render_pending_outreach(state.companion_id)
        return created

    background_scheduler.register_job(
        "reflection_cycle_job",
        3600,
        reflection_cycle_job,
    )

    def idle_reasoning_job() -> int:
        profile = state.db.get_active_initiative_profile(state.companion_id)
        if not profile:
            return 0
        freq = int(profile.get("reflection_frequency_minutes", 180))
        status = background_scheduler.status().get("idle_reasoning_job", {}) if background_scheduler else {}
        if status.get("last_success"):
            from datetime import datetime as _dt
            try:
                last_success = _dt.fromisoformat(status["last_success"])
                if (_dt.now() - last_success).total_seconds() < freq * 60:
                    return 0
            except Exception:
                pass
        result = idle_reasoning_service.run(state.companion_id, state.active_model_id, profile)
        return int(result.get("created", 0))

    background_scheduler.register_job(
        "idle_reasoning_job",
        300,
        idle_reasoning_job,
    )
    background_scheduler.start()
    state.startup_done = True

    # First opportunistic reflection after startup if enough summaries exist.
    profile = state.db.get_active_initiative_profile(state.companion_id)
    reflection_service.generate_reflections(state.companion_id, state.active_model_id, profile)
    reflection_service.gate_reflections(state.companion_id, profile)
    reflection_service.render_pending_outreach(state.companion_id)

    log.info("Backend ready on http://%s:%s", config.HOST, config.PORT)
    log.info("Active model: %s (%d discovered)", state.active_model_id, len(available_models))

    yield

    log.info("Shutting down...")
    if state.model:
        state.model.stop_generation()
    if state_service and state.companion_id:
        try:
            await loop.run_in_executor(
                None,
                lambda: state_service.save_runtime_snapshot(
                    state.companion_id, state.active_model_id, notes="shutdown"
                ),
            )
        except Exception as exc:
            # Last-resort fallback — at minimum save to the legacy STATE_FILE.
            log.warning("StateService shutdown save failed (%s) — falling back.", exc)
            try:
                await loop.run_in_executor(None, state.model.save_state)
            except Exception:
                pass
    elif state.model:
        await loop.run_in_executor(None, state.model.save_state)
    if state.active_session:
        state.active_session.end()
    if background_scheduler:
        background_scheduler.stop()
    if state.scheduler:
        state.scheduler.stop()
    log.info("Shutdown complete.")


app = FastAPI(
    title="Flint",
    version=APP_VERSION,
    description="RWKV-first companion with persistent state, reflections and automatic background learning.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: simple chat transcript parser
#
def _parse_chat_transcript(text: str) -> list[dict]:
    """
    Parse a chat transcript into a list of message dicts with keys
    ``role`` and ``content``.  The parser handles a few simple formats:

    1. JSON object with a ``messages`` array where each item has a
       ``role`` and ``content`` key.  Common for ChatGPT exports.
    2. JSON array of objects with ``role``/``content`` fields.
    3. Plain text transcript where each line starts with ``User:`` or
       ``Assistant:`` (case-insensitive).  Lines without a prefix are
       appended to the previous message.
    4. Plain text with no explicit prefixes; the parser alternates roles,
       starting with ``user``.  Each non-empty line becomes a message.

    Unknown roles are normalised to ``user`` or ``assistant``.

    Returns a list of message dicts.  If no messages can be parsed,
    returns an empty list.
    """
    text = text.strip()
    if not text:
        return []
    # Try JSON formats
    try:
        data = json.loads(text)
        msgs: list[dict] = []
        if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
            for item in data["messages"]:
                role = str(item.get("role", "user")).lower()
                content = str(item.get("content", "")).strip()
                if content:
                    msgs.append({"role": "assistant" if role in {"assistant", "system"} else "user", "content": content})
            return msgs
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    role = str(item.get("role", "user")).lower()
                    content = str(item.get("content", "")).strip()
                    if content:
                        msgs.append({"role": "assistant" if role in {"assistant", "system"} else "user", "content": content})
                else:
                    content = str(item).strip()
                    if content:
                        # Alternate roles for plain strings in array
                        role = "assistant" if len(msgs) % 2 == 1 else "user"
                        msgs.append({"role": role, "content": content})
            return msgs
    except Exception:
        pass  # fall back to text parsing
    # Plain text parsing
    lines = text.splitlines()
    messages: list[dict] = []
    current_role: str | None = None
    current_content: list[str] = []
    def flush_current():
        nonlocal current_role, current_content
        if current_role and current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
        current_content = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Detect prefix
        prefix_sep = s.find(":")
        if prefix_sep > 0:
            prefix = s[:prefix_sep].lower().strip()
            rest = s[prefix_sep+1:].lstrip()
            if prefix in {"user", "human"}:
                flush_current()
                current_role = "user"
                current_content = [rest]
                continue
            if prefix in {"assistant", "bot", "companion", "ai"}:
                flush_current()
                current_role = "assistant"
                current_content = [rest]
                continue
        # Otherwise append to current
        if current_role is None:
            # No role determined yet; alternate starting with user
            current_role = "user" if not messages or messages[-1]["role"] == "assistant" else "assistant"
            current_content = [s]
        else:
            current_content.append(s)
    flush_current()
    return messages




@app.get("/health", tags=["meta"])
async def health():
    if not state.startup_done:
        return JSONResponse(status_code=503, content={"status": "loading", "ready": False})
    return {"status": "ok", "ready": True}


@app.get("/api/info", tags=["meta"])
async def info():
    active = model_registry.active_model_info() if model_registry else None
    training = _current_training_profile()
    recommended = recommend_preset(_detect_vram_gb())
    entries = model_registry.scan_models() if model_registry else []
    active_filename = active["filename"] if active else None
    needs_model_upload = not active or (state.model.backend_kind == "dummy")
    info_payload = {
        "name": "flint",
        "version": APP_VERSION,
        "active_model": active,
        "strategy": config.MODEL_STRATEGY,
        "bot_name": config.BOT_NAME,
        "user_name": config.USER_NAME,
        "max_tokens": config.MAX_TOKENS,
        "context": config.CONTEXT_SIZE,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "model": active_filename,
        "backend_kind": state.model.backend_kind if state.model else "dummy",
        "model_load_error": state.model.load_error if state.model else None,
        "training_method": training["method"],
        "training_preset": training["id"],
        "training_label": training["label"],
        "training_note": training["note"],
        "training_mode": training["mode"],
        "recommended_training_preset": default_training_preset()["id"],
        "recommended_model_preset": recommended["id"],
        "recommended_model_label": recommended["label"],
        "recommended_model_best_for": recommended["best_for"],
        "models_count": len(entries),
        "needs_model_upload": needs_model_upload,
        "g1_onboarding": needs_model_upload,
        "active_model_official_g1": _is_official_g1_model(active_filename),
    }
    if state.model and state.model.backend_kind == "onnx":
        backend = getattr(state.model, "backend", None)
        info_payload["onnx_graph_signature"] = backend._graph_signature() if backend else "unavailable"
        info_payload["onnx_graph_ready"] = bool(getattr(backend, "graph_ready", False))
        info_payload["onnx_token_input"] = getattr(backend, "token_input_name", None)
        info_payload["onnx_logits_output"] = getattr(backend, "logits_output_name", None)
    return info_payload


@app.get("/api/models", tags=["meta"])
async def list_models():
    entries = model_registry.scan_models() if model_registry else []
    return {"models": entries, "active": state.active_model_id}


@app.post("/api/models/rescan", tags=["meta"])
async def rescan_models():
    entries = model_registry.scan_models() if model_registry else []
    return {"models": entries, "active": state.active_model_id}


@app.get("/api/model_presets", tags=["meta"])
async def list_model_presets():
    recommended = recommend_preset(_detect_vram_gb())
    return {"presets": G1_PRESETS, "recommended": recommended}


@app.get("/api/training_presets", tags=["meta"])
async def list_training_presets():
    return {"presets": TRAINING_PRESETS, "recommended": default_training_preset()}


@app.post("/api/models/upload", tags=["meta"])
async def upload_model(file: UploadFile = File(...), activate: bool = True):
    if not model_registry:
        return {"ok": False, "message": "Backend not ready"}
    raw_name = (file.filename or "").strip()
    if not raw_name:
        return {"ok": False, "message": "Missing filename"}
    suffix = Path(raw_name).suffix.lower()
    if suffix not in model_registry.SUPPORTED_EXTENSIONS:
        return {"ok": False, "message": "Unsupported model format. Use .pth or .onnx"}
    dest_dir = Path(config.BASE_DIR) / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(raw_name).name
    final_path = dest_dir / safe_name
    tmp_path = final_path.with_suffix(final_path.suffix + ".uploading")
    bytes_written = 0
    limit = config.MAX_UPLOAD_MODEL_BYTES
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)
                if limit is not None and bytes_written > limit:
                    tmp_path.unlink(missing_ok=True)
                    limit_mb = limit // 1024 // 1024
                    return JSONResponse(
                        status_code=413,
                        content={"ok": False, "message": f"File too large (max {limit_mb} MB)"},
                    )
        if bytes_written == 0:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return {"ok": False, "message": "Uploaded model file was empty"}
        tmp_path.replace(final_path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        log.error("model upload failed: %s", exc, exc_info=True)
        return {"ok": False, "message": "Model upload failed during file write"}
    entries = model_registry.scan_models()
    target = next((m for m in entries if m["filename"] == safe_name), None)
    if activate and target:
        loop = asyncio.get_running_loop()
        acquired = await loop.run_in_executor(None, lambda: state.generation_lock.acquire(timeout=10))
        if acquired:
            try:
                result = await _swap_model(target, loop)
                result["uploaded"] = target
                return result
            finally:
                state.generation_lock.release()
        return {"ok": False, "message": "Model uploaded, but activation is blocked because generation is currently in progress", "uploaded": target}
    return {"ok": True, "uploaded": target, "models": entries}


@app.get("/api/status", tags=["session"])
async def get_status():
    payload = chat_service.status_payload() if chat_service else {}
    jobs = background_scheduler.status() if background_scheduler else {}
    payload.update(
        {
            "lora_active": Path(config.LORA_ADAPTER).exists(),
            "lora_status": state.scheduler.status() if state.scheduler else "—",
            "background_jobs": jobs,
            "snapshot_status": _latest_snapshot_meta(),
            "reflection_status": _format_job_health(jobs, "reflection_cycle_job"),
            "idle_reasoning_status": _format_job_health(jobs, "idle_reasoning_job"),
            "summary_status": _format_job_health(jobs, "recent_summary_job"),
            "snapshot_job_status": _format_job_health(jobs, "runtime_snapshot_job"),
            "lora_last_run": state.scheduler.last_run.isoformat() if state.scheduler and state.scheduler.last_run else None,
            "system_ready": bool(state.startup_done),
        }
    )
    return payload


@app.get("/api/history", tags=["session"])
async def get_history(limit: int = 60):
    if not state.db:
        return {"messages": []}
    limit = min(limit, config.MAX_API_LIMIT)
    rows = state.db.get_recent_messages(limit=limit)
    return {
        "messages": [
            {
                "timestamp": r["timestamp"],
                "role": r["role"],
                "content": r["content"],
                "session_id": r["session_id"],
            }
            for r in rows
        ]
    }


@app.get("/api/search", tags=["session"])
async def search_history(q: str, limit: int = 10):
    if not state.db or not q.strip():
        return {"results": []}
    limit = min(limit, config.MAX_API_LIMIT)
    rows = state.db.search(q, limit=limit)
    return {
        "results": [
            {
                "timestamp": r["timestamp"],
                "role": r["role"],
                "content": r["content"],
                "session_id": r["session_id"],
            }
            for r in rows
        ]
    }


@app.get("/api/outreach", tags=["session"])
async def get_outreach(limit: int = 5):
    if not state.db or not state.companion_id:
        return {"items": []}
    rows = state.db.get_visible_outreach(state.companion_id, limit=limit)
    for row in rows:
        if not row["delivered_at"]:
            state.db.mark_outreach_delivered(row["id"])
    return {"items": rows}


# ──────────────────────────────────────────────────────────────────────────────
# Initiative profile management
# -----------------------------------------------------------------------------

@app.get("/api/initiative_profiles", tags=["session"])
async def list_initiative_profiles():
    """
    Return all initiative profiles for the current companion.

    Each profile contains its thresholds, frequency and whether it is active. If
    none have been created yet, an empty list is returned.
    """
    if not state.db or not state.companion_id:
        return {"profiles": []}
    rows = state.db.get_initiative_profiles(state.companion_id)
    return {"profiles": rows}


@app.post("/api/initiative_profiles", tags=["session"])
async def upsert_initiative_profile(profile: dict):
    """
    Create or update an initiative profile for the current companion.  The
    request body must include:

    - profile_name: str
    - reflection_frequency_minutes: int
    - outreach_max_per_day: int
    - minimum_priority_threshold: float
    - minimum_groundedness_threshold: float
    - minimum_novelty_threshold: float
    - active: bool (optional, default false)

    If active is true, this profile will become the active one for the
    companion, and any previous profiles will be deactivated.  Returns the
    updated list of profiles.
    """
    if not state.db or not state.companion_id:
        return {"ok": False, "message": "Backend not ready"}
    required = [
        "profile_name",
        "reflection_frequency_minutes",
        "outreach_max_per_day",
        "minimum_priority_threshold",
        "minimum_groundedness_threshold",
        "minimum_novelty_threshold",
    ]
    for key in required:
        if key not in profile:
            return {"ok": False, "message": f"Missing field: {key}"}
    try:
        active_flag = 1 if profile.get("active") else 0
        state.db.upsert_initiative_profile(
            companion_id=state.companion_id,
            profile_name=str(profile["profile_name"]),
            reflection_frequency_minutes=int(profile["reflection_frequency_minutes"]),
            outreach_max_per_day=int(profile["outreach_max_per_day"]),
            minimum_priority_threshold=float(profile["minimum_priority_threshold"]),
            minimum_groundedness_threshold=float(profile["minimum_groundedness_threshold"]),
            minimum_novelty_threshold=float(profile["minimum_novelty_threshold"]),
            active=active_flag,
        )
    except Exception as exc:
        log.error("Failed to save initiative profile: %s", exc, exc_info=True)
        return {"ok": False, "message": "Failed to save profile — see server log for details"}
    # Return updated list
    rows = state.db.get_initiative_profiles(state.companion_id)
    return {"ok": True, "profiles": rows}


@app.post("/api/outreach/{candidate_id}/dismiss", tags=["session"])
async def dismiss_outreach(candidate_id: str):
    """Dismiss an outreach candidate so it no longer appears in the UI."""
    if not state.db:
        return {"ok": False, "message": "Backend not ready"}
    state.db.dismiss_outreach(candidate_id)
    return {"ok": True}


@app.post("/api/reflect/run", tags=["session"])
async def run_reflection_cycle():
    if not state.db or not reflection_service:
        return {"ok": False, "message": "Backend not ready"}
    profile = state.db.get_active_initiative_profile(state.companion_id)
    created = reflection_service.generate_reflections(state.companion_id, state.active_model_id, profile)
    gated = reflection_service.gate_reflections(state.companion_id, profile)
    visible = reflection_service.render_pending_outreach(state.companion_id)
    return {"ok": True, "created": created, "gated": gated, "visible": visible}


@app.post("/api/reset", tags=["session"])
async def reset_state():
    loop = asyncio.get_running_loop()
    acquired = await loop.run_in_executor(None, lambda: state.generation_lock.acquire(timeout=5))
    if not acquired:
        return {"ok": False, "message": "Generation in progress — try again shortly"}
    try:
        session = chat_service.reset_conversation()
        return {"ok": True, "message": "State reset, new session started", "session_id": session.session_id}
    finally:
        state.generation_lock.release()


# ──────────────────────────────────────────────────────────────────────────────
# Model activation
# ----------------------------------------------------------------------------

@app.post("/api/models/activate", tags=["meta"])
async def activate_model(payload: dict):
    """
    Activate a different model by its id or filename.  The request body must
    include either ``model_id`` or ``filename``.  The backend will update
    the configuration, reload the model and restart the active session.  If
    the model cannot be loaded, an error is returned.  Returns the updated
    active model info on success.
    """
    if not state.startup_done or not model_registry:
        return {"ok": False, "message": "Backend not ready"}
    model_id = payload.get("model_id")
    filename = payload.get("filename")
    # Discover available models
    entries = model_registry.scan_models()
    target = None
    if model_id:
        target = next((e for e in entries if e["id"] == model_id), None)
    elif filename:
        target = next((e for e in entries if e["filename"] == filename), None)
    if not target:
        return {"ok": False, "message": "Model not found"}
    # If already active, no-op
    if target["id"] == state.active_model_id:
        return {"ok": True, "active": target}
    # Acquire generation lock before mutating model state.
    # This prevents a racing WS handler from generating with the old model
    # while session metadata has already been updated to point at the new one.
    loop = asyncio.get_running_loop()
    acquired = await loop.run_in_executor(None, lambda: state.generation_lock.acquire(timeout=10))
    if not acquired:
        return {"ok": False, "message": "Generation in progress — try again in a moment"}
    try:
        return await _swap_model(target, loop)
    finally:
        state.generation_lock.release()


async def _swap_model(target: dict, loop) -> dict:
    """Execute the model swap with generation_lock already held."""
    try:
        # Update global config
        config.MODEL_PATH = target["path"]
        state.active_model_id = target["id"]
        # Stop current generation
        if state.model:
            state.model.stop_generation()
        # Create a new model instance and load LoRA/state
        new_model = await loop.run_in_executor(None, CompanionModel)
        await loop.run_in_executor(None, new_model.load_lora)
        # Attempt to load previous state snapshot for this companion and model
        # (if available) to preserve conversation continuity
        if state_service:
            latest = state_service.latest_runtime_snapshot(state.companion_id, state.active_model_id)
        else:
            latest = None
        if latest:
            try:
                await loop.run_in_executor(None, new_model.load_state, latest["snapshot_path"])
            except Exception:
                await loop.run_in_executor(None, new_model.prime_system_prompt)
        else:
            await loop.run_in_executor(None, new_model.prime_system_prompt)
        state.model = new_model
        # Restart chat service session with new model
        if chat_service and state.active_session:
            state.active_session.end()
        state.active_session = Session(state.db, companion_id=state.companion_id, model_id=state.active_model_id)
        try:
            state.db.upsert_model_installation(
                model_id=state.active_model_id,
                install_path=target["path"],
                backend=config.MODEL_STRATEGY,
                is_default=1,
                verification_status="ok",
            )
        except Exception:
            pass
        return {"ok": True, "active": target}
    except Exception as exc:
        log.error("activate_model failed: %s", exc, exc_info=True)
        return {"ok": False, "message": "Model activation failed — see server log for details"}


# ──────────────────────────────────────────────────────────────────────────────
# Chat log import
# ----------------------------------------------------------------------------

@app.post("/api/import_chat", tags=["session"])
async def import_chat(file: UploadFile = File(...)):
    """
    Import a chat transcript from a file.  The uploaded file is parsed
    into user and assistant messages using a simple heuristic parser.
    A new session is created for the current companion and active model,
    and all messages are inserted with the ``imported`` flag set.  The
    parser accepts JSON export formats or plain text.  Returns the new
    session id and number of messages imported.
    """
    if not state.db or not state.companion_id:
        return {"ok": False, "message": "Backend not ready"}
    # ── Size guard ────────────────────────────────────────────────────────────
    try:
        content = await file.read(config.MAX_UPLOAD_CHAT_BYTES + 1)
    except Exception as exc:
        log.error("import_chat: file read error: %s", exc, exc_info=True)
        return {"ok": False, "message": "Could not read uploaded file"}
    if len(content) > config.MAX_UPLOAD_CHAT_BYTES:
        return JSONResponse(
            status_code=413,
            content={"ok": False, "message": f"File too large (max {config.MAX_UPLOAD_CHAT_BYTES // 1024 // 1024} MB)"},
        )
    text = content.decode("utf-8", errors="ignore")
    messages = _parse_chat_transcript(text)
    if not messages:
        return {"ok": False, "message": "No messages detected in file"}
    # Create a new import session
    session_id = state.db.new_session(state.companion_id, state.active_model_id)
    # Mark session as import source
    try:
        with state.db._conn() as conn:
            conn.execute(
                "UPDATE sessions SET source = ?, import_batch_id = ? WHERE id = ?",
                ("import", str(uuid.uuid4())[:8], session_id),
            )
    except Exception:
        pass
    # Insert messages
    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        try:
            message_id = state.db.add_message(
                session_id=session_id,
                role=role,
                content=content,
                companion_id=state.companion_id,
                model_id=state.active_model_id,
                turn_index=None,
            )
            # Mark as imported
            with state.db._conn() as conn:
                conn.execute("UPDATE messages SET imported = 1 WHERE id = ?", (message_id,))
        except Exception:
            continue
    try:
        state.db.end_session(session_id, lora_version="imported")
    except Exception:
        pass
    # After import, ingest and summarize conversation blocks so that
    # reflections/idle reasoning can operate on this data soon
    try:
        reflection_service.ingest_conversation_blocks(state.companion_id, state.active_model_id)
        reflection_service.summarize_recent_blocks(state.companion_id, state.active_model_id)
    except Exception:
        pass
    return {"ok": True, "session_id": session_id, "messages": len(messages)}


# ──────────────────────────────────────────────────────────────────────────────
# Runtime state export & import
# ----------------------------------------------------------------------------

@app.get("/api/state/snapshots", tags=["session"])
async def list_runtime_snapshots(limit: int = 10):
    """
    Return the most recent runtime state snapshots for the current
    companion and active model.  The `limit` parameter bounds the
    number of results.  Each snapshot includes its id, path and
    timestamp.  If no snapshots exist, an empty list is returned.
    """
    if not state_service or not state.companion_id:
        return {"snapshots": []}
    limit = min(limit, config.MAX_API_LIMIT)
    rows = state.db.get_runtime_state_snapshots(state.companion_id, state.active_model_id, limit)
    return {"snapshots": rows or []}


@app.get("/api/state/export", tags=["session"])
async def export_runtime_state():
    """
    Download the latest runtime state snapshot for the current
    companion and active model.  If none exists, a 404 response is
    returned.  The file is served directly to the client.  Note that
    restoring the snapshot on another instance will require using the
    `/api/state/import` endpoint.
    """
    if not state_service or not state.companion_id:
        return JSONResponse(status_code=404, content={"message": "Snapshot service unavailable"})
    snap = state_service.latest_runtime_snapshot(state.companion_id, state.active_model_id)
    if not snap:
        return JSONResponse(status_code=404, content={"message": "No snapshots found"})
    path = snap.get("snapshot_path")
    if not path or not Path(path).exists():
        return JSONResponse(status_code=404, content={"message": "Snapshot file not found"})
    return FileResponse(path, filename=Path(path).name, media_type="application/octet-stream")


@app.post("/api/state/import", tags=["session"])
async def import_runtime_state(file: UploadFile = File(...)):
    """
    Restore a runtime snapshot from an uploaded file.  The snapshot
    file is saved into the state directory and registered in the
    database.  If a model is attached, its state will be loaded from
    the file, replacing the current runtime state.  Returns the
    snapshot id.  Any errors result in a failure message.
    """
    if not state_service or not state.companion_id:
        return {"ok": False, "message": "Snapshot service unavailable"}
    # ── Size guard ────────────────────────────────────────────────────────────
    try:
        content = await file.read(config.MAX_UPLOAD_SNAPSHOT_BYTES + 1)
    except Exception as exc:
        log.error("import_runtime_state: file read error: %s", exc, exc_info=True)
        return {"ok": False, "message": "Could not read uploaded file"}
    if len(content) > config.MAX_UPLOAD_SNAPSHOT_BYTES:
        return JSONResponse(
            status_code=413,
            content={"ok": False, "message": f"File too large (max {config.MAX_UPLOAD_SNAPSHOT_BYTES // 1024 // 1024} MB)"},
        )
    # ── Path traversal guard ──────────────────────────────────────────────────
    # Strip any directory components from the client-supplied filename so a
    # malicious value like "../../web/static/index.html" cannot escape the
    # states directory.
    raw_name = file.filename or f"imported-{datetime.now().strftime('%Y%m%dT%H%M%S')}.bin"
    safe_name = Path(raw_name).name  # strip all directory parts
    # Resolve the destination and verify it stays within states_dir.
    dest_path = (state_service.states_dir / safe_name).resolve()
    if not str(dest_path).startswith(str(state_service.states_dir.resolve())):
        return {"ok": False, "message": "Invalid filename"}
    try:
        with open(dest_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        log.error("import_runtime_state: could not write file: %s", exc, exc_info=True)
        return {"ok": False, "message": "Could not save snapshot file — see server log for details"}
    # Register snapshot and load state if a model exists
    try:
        snapshot_id = state.db.add_runtime_state_snapshot(
            companion_id=state.companion_id,
            model_id=state.active_model_id,
            snapshot_path=str(dest_path),
            notes="imported via API",
        )
        # Attempt to load state into the model
        loop = asyncio.get_running_loop()
        if state.model:
            try:
                await loop.run_in_executor(None, state.model.load_state, str(dest_path))
            except Exception:
                # ignore load errors — snapshot is still registered
                pass
        return {"ok": True, "snapshot_id": snapshot_id}
    except Exception as exc:
        log.error("import_runtime_state: DB registration failed: %s", exc, exc_info=True)
        return {"ok": False, "message": "Snapshot imported but registration failed — see server log for details"}



@app.post("/api/save", tags=["session"])
async def save_runtime_state():
    if not state_service or not state.companion_id:
        return {"ok": False, "message": "Snapshot service unavailable"}
    try:
        snapshot_id = state_service.save_runtime_snapshot(state.companion_id, state.active_model_id, notes="manual save")
        return {"ok": True, "message": "State saved", "snapshot_id": snapshot_id}
    except Exception as exc:
        log.error("Manual save failed: %s", exc, exc_info=True)
        return {"ok": False, "message": "Save failed — see server log for details"}


@app.get("/api/export", tags=["session"])
async def export_history(fmt: str = "json", limit: int = 2000):
    """
    Export conversation history as a download.
    fmt=json (default) → structured JSON
    fmt=txt            → human-readable plain text with session dividers
    Uses StreamingResponse so no temporary files are written to disk.
    """
    from fastapi.responses import StreamingResponse as _StreamingResponse
    limit = min(limit, 5000)  # generous cap for exports
    rows = state.db.get_recent_messages(limit=limit) if state.db else []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "txt":
        lines = [f"Flint — Export {ts}", "=" * 50, ""]
        prev_session = None
        for r in rows:
            if r["session_id"] != prev_session:
                if prev_session is not None:
                    lines.append("")
                lines.append(f"── Session {r['session_id']} ─────────────────")
                prev_session = r["session_id"]
            prefix = config.USER_NAME if r["role"] == "user" else config.BOT_NAME
            ts_short = r["timestamp"][:16].replace("T", " ")
            lines.append(f"[{ts_short}] {prefix}: {r['content']}")
        content = "\n".join(lines) + "\n"
        return _StreamingResponse(
            iter([content]),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="flint_export_{ts}.txt"'},
        )
    else:
        payload = {
            "exported_at": datetime.now().isoformat(),
            "bot_name": config.BOT_NAME,
            "user_name": config.USER_NAME,
            "messages": [dict(r) for r in rows],
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return _StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="flint_export_{ts}.json"'},
        )


@app.get("/api/lora/status", tags=["session"])
async def lora_status():
    pipeline = state.scheduler.pipeline if state.scheduler else None
    can_run, reason = pipeline.should_run() if pipeline else (False, "scheduler unavailable")
    return {
        "lora_active": Path(config.LORA_ADAPTER).exists(),
        "can_run": can_run,
        "reason": reason,
        "last_run": state.scheduler.last_run.isoformat() if state.scheduler and state.scheduler.last_run else None,
        "in_progress": bool(pipeline and getattr(pipeline.__class__, "_training_lock").locked()),
    }


@app.post("/api/lora/run", tags=["session"])
async def run_lora_now():
    if not state.scheduler:
        return {"ok": False, "message": "LoRA scheduler unavailable"}
    ok, msg = state.scheduler.pipeline.should_run()
    if not ok:
        return {"ok": False, "message": msg}
    state.scheduler.run_now()
    return {"ok": True, "message": "LoRA training started"}


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue
            if msg_type == "stop" and state.model:
                state.model.stop_generation()
                await websocket.send_text(json.dumps({"type": "stopped"}))
                continue
            if msg_type != "message":
                continue
            if not state.startup_done:
                await websocket.send_text(json.dumps({"type": "error", "message": "Model is still loading — please wait."}))
                continue
            user_input = msg.get("content", "").strip()
            if not user_input:
                continue
            # Silently truncate oversized inputs — prevents prompt-stuffing / OOM.
            if len(user_input) > config.MAX_INPUT_CHARS:
                user_input = user_input[:config.MAX_INPUT_CHARS]
            if not state.generation_lock.acquire(blocking=False):
                await websocket.send_text(json.dumps({"type": "busy", "message": "Already generating — please wait."}))
                continue
            try:
                await _generate_and_stream(websocket, user_input)
            finally:
                state.generation_lock.release()
    except WebSocketDisconnect:
        log.debug("WebSocket disconnected — aborting generation if active.")
        if state.model:
            state.model.stop_generation()
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, state.model.save_state)
    except Exception as exc:  # pragma: no cover - defensive
        log.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "An internal server error occurred"}))
        except Exception:
            pass


async def _generate_and_stream(websocket: WebSocket, user_input: str):
    # Build prompt using the last few turns of history so the model has
    # conversational context even though RWKV state carries long-term memory.
    # G1 World models are trained on "User: ...\n\nAssistant: ..." format.
    history_turns = []
    if state.db:
        rows = state.db.get_recent_messages(limit=6)
        for r in rows:
            role_label = config.USER_NAME if r["role"] == "user" else config.BOT_NAME
            history_turns.append(f"{role_label}: {r['content']}")
    history_block = "\n\n".join(history_turns)
    if history_block:
        prompt = f"{history_block}\n\n{config.USER_NAME}: {user_input}\n\n{config.BOT_NAME}:"
    else:
        prompt = f"{config.USER_NAME}: {user_input}\n\n{config.BOT_NAME}:"
    await websocket.send_text(json.dumps({"type": "start", "timestamp": datetime.now().isoformat()}))

    token_queue: queue.Queue = queue.Queue()
    result_holder: list[GenerationResult] = []

    def generate_in_thread():
        def on_token(token: str):
            token_queue.put(("token", token))
        try:
            result = state.model.generate(prompt=prompt, stream_callback=on_token)
            result_holder.append(result)
        except Exception as exc:
            token_queue.put(("error", str(exc)))
        finally:
            token_queue.put(("done", None))

    thread = threading.Thread(target=generate_in_thread, daemon=True)
    thread.start()
    loop = asyncio.get_running_loop()

    while True:
        try:
            event_type, value = await loop.run_in_executor(None, lambda: token_queue.get(timeout=60))
        except queue.Empty:
            await websocket.send_text(json.dumps({"type": "error", "message": "Generation timed out."}))
            break

        if event_type == "token":
            await websocket.send_text(json.dumps({"type": "token", "content": value}))
        elif event_type == "error":
            await websocket.send_text(json.dumps({"type": "error", "message": value}))
            break
        elif event_type == "done":
            result = result_holder[0] if result_holder else None
            if result and result.text:
                chat_service.register_turn(user_input, result)
                if state.active_session and state.active_session.turn_count % config.AUTOSAVE_TURNS == 0:
                    if state_service:
                        await loop.run_in_executor(
                            None,
                            lambda: state_service.save_runtime_snapshot(
                                state.companion_id, state.active_model_id, notes="autosave"
                            ),
                        )
                    else:
                        await loop.run_in_executor(None, state.model.save_state)
                # After a completed turn, opportunistically refresh small pieces.
                if reflection_service and state.db:
                    reflection_service.ingest_conversation_blocks(state.companion_id, state.active_model_id)
                    reflection_service.summarize_recent_blocks(state.companion_id, state.active_model_id, limit=2)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "done",
                        "turn_count": state.active_session.turn_count if state.active_session else 0,
                        "tokens": result.tokens if result else 0,
                        "elapsed": round(result.elapsed, 2) if result else 0,
                        "tokens_per_second": round(result.tokens_per_second, 1) if result else 0,
                        "session_id": state.active_session.session_id if state.active_session else None,
                    }
                )
            )
            break


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


if __name__ == "__main__":
    uvicorn.run(
        "web.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )
