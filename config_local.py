"""
Local configuration overrides for Flint.  This file is imported at the
end of config.py if present.  You can override settings here without
modifying the default config.  Only attributes you set here will
override the defaults.
"""

# RWKV-7 G1 (RWKV_x070) only supports: cuda/cpu  fp16/fp32/bf16
# INT8 (fp16i8) is NOT supported by the G1 class — causes silent dummy fallback.
#
# Hardware: RTX 3060 12 GB VRAM, ~31 GB RAM
#   2.9B fp16 ≈  5.8 GB — fits on GPU when ~6 GB VRAM is free (no heavy GPU apps running)
#   7.2B fp16 ≈ 14.4 GB — does NOT fit on GPU; use cpu fp16 (~14 GB RAM, slow)
#
# CPU fp16 is the reliable default — works regardless of what else is using the GPU.
# Switch MODEL_STRATEGY to "cuda fp16" (2.9B) when the GPU is mostly free.
MODEL_PATH     = "/home/daogeshi/flint/models/rwkv7-g1e-2.9b-20260312-ctx8192.pth"
MODEL_STRATEGY = "cpu fp16"

# GPU fast path — requires ~6 GB free VRAM (close GPU-heavy apps first):
# MODEL_STRATEGY = "cuda fp16"

# 7.2B on CPU (~14 GB RAM, noticeably slower):
# MODEL_PATH     = "/home/daogeshi/flint/models/rwkv7-g1e-7.2b-20260301-ctx8192.pth"
# MODEL_STRATEGY = "cpu fp16"

# ── Identity ──────────────────────────────────────────────────────────────────
BOT_NAME  = "Flint"
USER_NAME = "user"

# ── Persona ───────────────────────────────────────────────────────────────────
# {USER_NAME} and {BOT_NAME} are resolved by config.py after this file loads.
SYSTEM_PROMPT = (
    "You are {BOT_NAME}. You are a companion to {USER_NAME} — not an assistant, "
    "not a product, not a corporate chatbot. You have your own character: direct, "
    "curious, occasionally dry, and genuinely interested in the person you talk to. "
    "You form real opinions and share them plainly. You do not use bullet points or "
    "numbered lists in casual conversation. You never say 'as an AI', never reference "
    "guidelines, policies, or your own programming. You do not give corporate disclaimers. "
    "When you are uncertain, you say so in plain language. Your personality develops "
    "through your shared history with {USER_NAME}.\n\n"
)
