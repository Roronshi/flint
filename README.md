# Flint

**A local AI companion that grows with you over time through persistent state and nightly fine-tuning. Early, incomplete, and entirely yours.**

Flint is an attempt to build something that doesn't exist yet: a personal AI companion that lives on your machine, develops over time, and remains entirely yours.

Most AI assistants reset with every conversation and recall everything with perfect fidelity. Flint aims for something different — something closer to how human memory actually works by help of using RWKV.

Flint won't remember every detail of what you said three months ago but if you have a tendency to talk about specific themes or perspectives it'll gradually adapt to being more in tune with you on these matters.

What Flint aims to do instead is to gradually become shaped by the sum of your shared history: the things you return to, the way you phrase things, the texture of how you relate. Less a searchable database, more a relationship that accumulates.

The underlying mechanism is RWKV — a recurrent model that carries genuine internal state rather than processing a context window from scratch each time. That state is saved to disk at the end of every session and restored at the start of the next.

Alongside that, Flint runs nightly LoRA fine-tuning on recent conversations mixed with older ones, slowly bending the model's character over weeks and months. The goal isn't a smarter assistant. It's a companion with a developing persona.

Critically, all of this is designed to run on a personal computer. RWKV models are small enough that a modern consumer GPU can run inference comfortably, and the same is true of the nightly fine-tuning — LoRA training on a modest dataset fits within the kind of hardware people actually own. No cloud compute, no rented GPU, no subscription. The whole system — inference, memory, training — runs on the machine on your desk.

This is early and incomplete. The architecture is in place — chat, state persistence, nightly training, conversation import, web UI — but real-world validation against a live model is still ahead. Nothing here is a promise. It's a direction.

```bash
curl -fsSL https://raw.githubusercontent.com/Roronshi/flint/main/install.sh | bash
```

---

## How it works

RWKV is an RNN with transformer-level performance but constant memory usage — unlike transformers, it pays no growing cost for longer conversations. This makes it uniquely suited for long-term companion use.

The project consists of three layers operating on different timescales:

| Layer | What it is | Timescale |
|---|---|---|
| **RWKV State** | Active relationship memory — what is alive right now | Per session |
| **LoRA Adapter** | Character formation — how the model fundamentally meets you | Weeks, months |
| **SQLite log** | Exact conversation history, searchable, yours | Forever |

State is saved to disk at the end of every session. Next time you open the chat, the model continues exactly where it left off — not through text reconstruction but through a genuinely stored mental state. LoRA runs automatically every night (03:00) on the day's conversations mixed with older ones (replay buffer), gradually bending the base weights toward how you communicate.

---

## Quick start

### 1. Clone
```bash
git clone https://github.com/Roronshi/flint
cd flint
```

### 2. Install
```bash
bash install.sh
```

The script installs all Python dependencies, clones RWKV-PEFT, creates data directories, and guides you through the model download.

### 3. Get a model

There are now two normal ways to do this:

1. Run `bash install.sh` and let Flint guide you toward an official **RWKV-7 G1** size.
2. Or upload your own `.pth` or `.onnx` model directly from the web UI after startup.

**Recommended official G1 sizes:**

| Size | Best for |
|---|---|
| 0.1B | testing / very limited hardware |
| 0.4B | lightweight laptops and CPUs |
| 1.5B | balanced local use |
| 2.9B | faster GPU-backed local use |
| 7.2B | strongest local quality on high-end hardware |

Custom models are allowed, but they are not the default recommendation.

### 4. Configure
```bash
cp config.example.py config_local.py
# Open config_local.py and set MODEL_PATH, USER_NAME, BOT_NAME
```

### 5. Start
```bash
# Web UI (recommended)
uvicorn web.server:app --host 0.0.0.0 --port 8000

# Terminal UI
python main.py
```

Open `http://localhost:8000` in your browser.

---

## Web UI

Clean, dark, fast. Tokens stream in real time. Sidebar with session statistics, full-text search across conversation history, and a manual LoRA trigger.

Accessible from mobile via [Tailscale](https://tailscale.com):
```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000
# Open on mobile: http://:8000
```

**Sidebar buttons:**

| Button | Function |
|---|---|
| ◈ Save state | Save manually (also happens automatically every 5 turns) |
| ⟳ Run LoRA now | Trigger training immediately without waiting until 03:00 |
| ◌ Reset state | Start the relationship from scratch (log is preserved) |
| Search field | Full-text search across the entire conversation history |

---

## Terminal UI
```bash
python main.py
```

**Commands:**
```
/status       — state, LoRA version, session statistics
/save         — save state manually
/reset        — reset state
/search <q>   — search conversation history
/lora now     — run LoRA training immediately
/lora status  — show scheduler status
/quit         — exit and save
```

---

## Importing existing conversations

The aim for the system is to allow for the import of your ChatGPT, Claude or other LLM history into Flint which gets fed this data as a starting point. The more data, the stronger the LoRA starting point.
```bash
# ChatGPT (Settings → Export data → conversations.json)
python tools/parser.py --source chatgpt --file ~/Downloads/conversations.json

# Claude (Settings → Export)
python tools/parser.py --source claude --file ~/Downloads/conversations.json

# Preview without saving
python tools/parser.py --source claude --file export.json --dry-run
```

---

## LoRA — automatic character formation

LoRA runs automatically every night at 03:00 if there are enough new conversations (default: at least 3). You won't notice — the bot is just a little more itself the next morning.

**How it works:**

1. Fetches sessions not yet trained on
2. Mixes in 30% older sessions (replay buffer — prevents new training from erasing old character)
3. Runs RWKV-PEFT training (~15–30 min on 12GB VRAM)
4. Saves updated adapter, loaded at next session start

**Configuration in `config_local.py`:**
```python
LORA_R          = 16      # Rank — higher = more dramatic change
LORA_MIN_CONVOS = 3       # Don't train on fewer sessions than this
REPLAY_RATIO    = 0.3     # 30% old conversations in each training batch
LORA_SCHEDULE   = "03:00" # Nightly training time
```

---

## Decentralized setup

All data is stored locally. Sync `data/states/` and `data/lora_adapters/` with e.g. Syncthing to have the same companion across multiple devices. The base model only needs to exist on the server.
```
flint/
├── models/
│   └── rwkv-7b.pth               # Base weights (~14GB)
├── data/
│   ├── states/{name}_state.pt    # Active relationship memory (~200MB)
│   ├── lora_adapters/current.pth # Personal adapter (~100–200MB)
│   └── conversations.db          # SQLite log
└── config_local.py
```

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11 |
| RAM | 16GB | 32GB+ |
| VRAM | 8GB (Q8) | 12GB+ (fp16) |
| Disk | 20GB | 50GB+ |
| OS | Linux / macOS | Linux |

---

## Project structure
```
flint/
├── core/
│   ├── model.py        # RWKV wrapper, state load/save, generation
│   └── session.py      # SQLite log, session management
├── lora/
│   ├── pipeline.py     # LoRA training with replay buffer
│   └── scheduler.py    # Nightly automatic training
├── interface/
│   └── terminal.py     # Terminal chat UI
├── web/
│   ├── server.py       # FastAPI + WebSocket backend
│   └── static/
│       └── index.html  # Complete web UI (single file)
├── tools/
│   └── parser.py       # Conversation import (ChatGPT / Claude)
├── models/             # Place .pth files here
├── data/               # State, adapters, SQLite — created automatically
├── config.py           # Default configuration
├── config.example.py   # Template for your configuration
├── main.py             # Terminal UI entry point
├── install.sh          # Installer script
└── requirements.txt
```

---

## License

MIT

## Beta candidate checklist

See `BETA_CANDIDATE_CHECKLIST.md` for the concrete definition of what remains before Flint should be called a true beta candidate.

