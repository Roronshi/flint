# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Flint is a local-only personal AI companion built on RWKV (a recurrent model architecture). Unlike transformer-based chatbots, RWKV carries genuine RNN state between sessions — state is saved to disk at session end and restored on startup. Nightly LoRA fine-tuning (03:00 default) gradually shapes the model's character from accumulated conversations. Everything runs on consumer hardware with no cloud dependency.

**Three-layer memory architecture:**
| Layer | What it is | Timescale |
|---|---|---|
| RWKV State | Active relationship memory, saved as `.pt` file | Per session |
| LoRA Adapter | Character formation via nightly fine-tuning | Weeks/months |
| SQLite log | Exact conversation history | Forever |

## Running the project

```bash
bash install.sh          # First-time setup
bash start.sh            # Start backend + open http://localhost:8000
bash start.sh --fg       # Run in foreground (logs to stdout)
bash stop.sh             # Stop background process
python main.py           # Terminal UI instead of web UI
uvicorn web.server:app --host 0.0.0.0 --port 8000  # Backend only
```

## Tests

```bash
pytest tests/                      # All tests
pytest tests/test_api_surface.py   # Specific file
pytest -v                          # Verbose
```

## Configuration

- `config.py` — defaults, **do not edit**
- `config_local.py` — local overrides, `.gitignored`. Copy `config.example.py` to create it.
- Key settings to override: `MODEL_PATH`, `MODEL_STRATEGY` (`"cuda fp16"` or `"cpu fp32"`), `USER_NAME`, `BOT_NAME`, `SYSTEM_PROMPT`
- `config.py` loads `config_local.py` via `from config_local import *` at the bottom, then re-derives `STATE_FILE` and resolves `SYSTEM_PROMPT` placeholders — this ordering matters if you add new derived values.

## Architecture

### Request flow
```
index.html (WebSocket) → web/server.py (FastAPI)
    → services/chat_service.py (session lifecycle)
    → core/model.py (CompanionModel)
    → core/model_backends/rwkv_backend.py (RWKV-7 G1)
```

### Key modules

**`core/model.py`** — `CompanionModel`: wraps any backend, owns `load_state()`/`save_state()`/`generate()`. The backend is selected at startup based on file extension (`.pth` → RWKV, `.onnx` → ONNX).

**`core/session.py`** — `ConversationDB` (SQLite) and `Session`. The DB schema has: companions, models, sessions, messages, summaries, reflections, outreach tables. `get_session_as_training_text()` formats a session for LoRA training.

**`core/app_state.py`** — `AppState` dataclass: the single shared runtime object holding `db`, `model`, `scheduler`, `active_session`, `training_active`, `training_progress`, and performance metrics. Passed around via FastAPI `app.state`.

**`web/server.py`** — FastAPI app (~2000 lines). Lifespan startup initializes model, DB, and scheduler. WebSocket `/ws` is the main chat endpoint with token streaming. REST endpoints cover `/api/status`, `/api/chat/history`, `/api/models/*`, `/api/state/*`, `/api/search`, `/api/lora/*`, `/api/upload/*`.

**`lora/trainer.py`** — `LoRALinear` (frozen base weights + trainable low-rank delta) and `RWKVLoRATrainer`. Targets receptance, key, value, output projections. Pure PyTorch, no deepspeed dependency for in-process training.

**`lora/pipeline.py`** — `LoRAPipeline`: builds training data (new sessions + replay buffer at `REPLAY_RATIO`), checks `LORA_MIN_CONVOS` threshold before running.

**`lora/scheduler.py`** — `LoRAScheduler`: daemon thread, runs training at `LORA_SCHEDULE` time, persists last-run timestamp to `data/lora_last_run.json`.

**`services/`** — Thin service layer over core: `chat_service.py` (session lifecycle), `model_registry.py` (model discovery/metadata), `reflection_service.py` (auto-generated conversation summaries), `idle_reasoning.py` (proactive question generation), `scheduler_service.py` (bridges scheduler to API).

**`core/model_backends/dummy_backend.py`** — Returns empty/stub responses. Used in tests to avoid loading real model weights.

### State file location
`data/states/{USER_NAME}_state.pt` — derived after `config_local.py` overrides, so changing `USER_NAME` changes the state file path.

### LoRA adapter
`data/lora_adapters/current_adapter.pth` — loaded at startup if present, applied on top of base model weights.
