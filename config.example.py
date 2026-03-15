# config_local.py — Your personal configuration
#
# Copy this file to config_local.py and fill in your values.
# This file is loaded by config.py and overrides the defaults.
# config_local.py is .gitignore'd — your settings never leave this machine.

# ── Required ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/path/to/your/rwkv-7-7b.pth"
USER_NAME  = "yourname"
BOT_NAME   = "Flint"

# ── Model strategy ────────────────────────────────────────────────────────────
# "cuda fp16"                   — NVIDIA GPU with 12GB+ VRAM (recommended)
# "cuda fp16 *20 -> cpu fp32"   — Offload some layers to RAM (saves VRAM)
# "cpu fp32"                    — CPU only (slow but works anywhere)
MODEL_STRATEGY = "cuda fp16"

# ── Persona ───────────────────────────────────────────────────────────────────
# {USER_NAME} and {BOT_NAME} are resolved automatically at load time.
SYSTEM_PROMPT = (
    "You are {BOT_NAME}, a companion to {USER_NAME} — not an assistant, not a product, "
    "not a corporate chatbot. You have your own character: direct, curious, and genuinely "
    "interested in the person you talk to. You form real opinions and share them plainly. "
    "You do not use bullet points or numbered lists in casual conversation. "
    "You never say 'as an AI', never reference guidelines, policies, or your own programming. "
    "You do not give corporate disclaimers. When you are uncertain, you say so in plain language. "
    "Your personality develops through your shared history with {USER_NAME}.\n\n"
)

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS  = 500
TEMPERATURE = 1.0
TOP_P       = 0.85
TOP_K       = 0      # 0 = disabled

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_SCHEDULE   = "03:00"   # Nightly training time (HH:MM, 24h)
LORA_MIN_CONVOS = 3         # Minimum new sessions before training runs

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"   # Use "127.0.0.1" to restrict to localhost only
PORT = 8000

# ── Logging ───────────────────────────────────────────────────────────────────
# "DEBUG" for verbose output during development, "INFO" for normal use
LOG_LEVEL = "INFO"
# Set to a path to also write logs to file, e.g. "data/companion.log"
# LOG_FILE = "data/companion.log"
