#!/usr/bin/env bash
# install.sh — Flint installer
#
# Usage:
#   bash install.sh          — interactive first-time install
#   bash install.sh --update — re-run to update deps (skips config prompts)

set -euo pipefail

# When piped through curl, BASH_SOURCE[0] is empty or unset.
# Detect this and require the user to clone first.
if [[ -z "${BASH_SOURCE[0]:-}" ]] || [[ "${BASH_SOURCE[0]}" == "bash" ]]; then
    echo ""
    echo "Flint — installer"
    echo "────────────────────────────────────────────"
    echo ""
    echo "  Please clone the repository before running install.sh:"
    echo ""
    echo "    git clone https://github.com/Roronshi/flint"
    echo "    cd flint"
    echo "    bash install.sh"
    echo ""
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
UPDATE_MODE=false
[[ "${1:-}" == "--update" ]] && UPDATE_MODE=true

# ── Colours ───────────────────────────────────────────────────────────────────
RESET=$'\033[0m'; BOLD=$'\033[1m'; DIM=$'\033[2m'
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; CYAN=$'\033[36m'

ok()   { echo "${GREEN}✓${RESET}  $*"; }
warn() { echo "${YELLOW}⚠${RESET}  $*"; }
err()  { echo "${RED}✗${RESET}  $*" >&2; }
info() { echo "${DIM}  $*${RESET}"; }
hdr()  { echo ""; echo "${BOLD}$*${RESET}"; echo "${DIM}────────────────────────────────────────────${RESET}"; }

echo ""
echo "${BOLD}${CYAN}Flint${RESET} — installer"
echo "${DIM}────────────────────────────────────────────${RESET}"

# ── 1. Python 3.10+ ───────────────────────────────────────────────────────────
hdr "Python"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        maj=$("$candidate" -c "import sys; print(sys.version_info.major)")
        mino=$("$candidate" -c "import sys; print(sys.version_info.minor)")
        if (( maj >= 3 && mino >= 10 )); then
            ver=$("$candidate" -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
            PYTHON="$candidate"
            ok "Python $ver  ($(command -v $candidate))"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "Python 3.10+ not found."
    info "Arch:   sudo pacman -S python"
    info "Ubuntu: sudo apt install python3.12"
    exit 1
fi

# ── 2. GPU detection ──────────────────────────────────────────────────────────
hdr "Hardware"

CUDA_AVAILABLE=false
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
VRAM_GB=0
RECOMMENDED_MODEL="1.5B"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
    VRAM_GB=$(( VRAM_MB / 1024 ))

    CUDA_VERSION=""
    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || true)
    fi
    if [[ -z "$CUDA_VERSION" ]]; then
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
    fi

    if [[ -n "$CUDA_VERSION" ]]; then
        CUDA_MAJOR="${CUDA_VERSION%%.*}"
        case "$CUDA_MAJOR" in
            12) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
            11) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
             *) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        esac
        CUDA_AVAILABLE=true
        ok "GPU: $GPU_NAME  (${VRAM_GB} GB VRAM, CUDA $CUDA_VERSION)"

        if   (( VRAM_GB >= 14 )); then RECOMMENDED_MODEL="7.2B"
        elif (( VRAM_GB >=  8 )); then RECOMMENDED_MODEL="2.9B"
        elif (( VRAM_GB >=  4 )); then RECOMMENDED_MODEL="1.5B"
        else                           RECOMMENDED_MODEL="0.1B"
        fi
    else
        warn "NVIDIA GPU found but CUDA toolkit not detected — falling back to CPU"
    fi
else
    warn "No NVIDIA GPU detected — using CPU (slower inference)"
fi

info "Recommended model size for this hardware: ${BOLD}${RECOMMENDED_MODEL}${RESET}"

# ── 3. Virtual environment ────────────────────────────────────────────────────
hdr "Virtual environment"

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Created .venv"
else
    ok ".venv exists — will update packages"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"
"$PIP" install --quiet --upgrade pip setuptools wheel

# ── 4. Python packages ────────────────────────────────────────────────────────
hdr "Python packages"

info "Core packages…"
"$PIP" install --quiet \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    python-multipart \
    numpy \
    "huggingface-hub[cli]" \
    rwkv \
    onnxruntime
ok "fastapi, uvicorn, rwkv, onnxruntime, huggingface-hub"

info "PyTorch ($(basename $TORCH_INDEX))…"
"$PIP" install --quiet torch --index-url "$TORCH_INDEX"
ok "torch"

# ── 5. RWKV-PEFT (LoRA training — optional) ──────────────────────────────────
hdr "RWKV-PEFT  (LoRA training)"

if command -v git &>/dev/null; then
    if [[ ! -d "$SCRIPT_DIR/RWKV-PEFT" ]]; then
        info "Cloning…"
        git clone --quiet https://github.com/Joluck/RWKV-PEFT "$SCRIPT_DIR/RWKV-PEFT"
        "$PIP" install --quiet -e "$SCRIPT_DIR/RWKV-PEFT/"
        ok "RWKV-PEFT installed  (Joluck/RWKV-PEFT — RWKV7 LoRA support)"
    else
        ok "RWKV-PEFT already present"
    fi
else
    warn "git not found — RWKV-PEFT skipped (LoRA training will be unavailable)"
fi

# ── 6. Runtime directories ────────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/data/states" \
         "$SCRIPT_DIR/data/lora_adapters" \
         "$SCRIPT_DIR/models"

# ── 7. Configuration ──────────────────────────────────────────────────────────
hdr "Configuration"

if [[ -f "$SCRIPT_DIR/config_local.py" ]]; then
    ok "config_local.py already exists — leaving untouched"
elif [[ "$UPDATE_MODE" == true ]]; then
    cp "$SCRIPT_DIR/config.example.py" "$SCRIPT_DIR/config_local.py"
    ok "config_local.py created from example (update mode)"
else
    echo ""
    echo "  Two quick questions to personalise Flint."
    echo ""

    read -r -p "  ${BOLD}Your name${RESET}  (e.g. alex):    " USER_NAME_INPUT
    USER_NAME_INPUT="${USER_NAME_INPUT:-user}"
    USER_NAME_INPUT="${USER_NAME_INPUT// /_}"

    read -r -p "  ${BOLD}Companion name${RESET}  (e.g. lyra):  " BOT_NAME_INPUT
    BOT_NAME_INPUT="${BOT_NAME_INPUT:-companion}"
    BOT_NAME_INPUT="${BOT_NAME_INPUT// /_}"

    STRATEGY="cpu fp32"
    [[ "$CUDA_AVAILABLE" == true ]] && STRATEGY="cuda fp16"

    cat > "$SCRIPT_DIR/config_local.py" << CONF
# config_local.py — generated by install.sh
# Edit freely. This file is .gitignore'd and stays on your machine.

MODEL_PATH     = "$SCRIPT_DIR/models/rwkv-model.pth"
MODEL_STRATEGY = "$STRATEGY"
USER_NAME      = "$USER_NAME_INPUT"
BOT_NAME       = "$BOT_NAME_INPUT"

SYSTEM_PROMPT = (
    "You are {BOT_NAME}, a personal companion for {USER_NAME}. "
    "You are curious, thoughtful, and direct. You remember our previous conversations "
    "through your internal state. You develop gradually and are shaped by our shared history."
)
CONF
    ok "config_local.py created  (${USER_NAME_INPUT} / ${BOT_NAME_INPUT})"
fi

# ── 8. Model ──────────────────────────────────────────────────────────────────
hdr "Model"

EXISTING_MODEL=$(ls "$SCRIPT_DIR/models/"*.pth "$SCRIPT_DIR/models/"*.onnx 2>/dev/null | head -1 || true)

if [[ -n "$EXISTING_MODEL" ]]; then
    ok "Model found: $(basename "$EXISTING_MODEL")"
    sed -i "s|^MODEL_PATH.*=.*|MODEL_PATH = \"$EXISTING_MODEL\"|" \
        "$SCRIPT_DIR/config_local.py" 2>/dev/null || true
else
    echo ""
    echo "  No model found in models/. RWKV‑7 G1 is available in several sizes."
    echo ""
    printf "  %s  %-8s  %-10s  %s\n" "#" "Size" "Approx" "Best for"
    echo "  ${DIM}────────────────────────────────────────────────────${RESET}"
    # Present G1 sizes.  These approximate disk sizes reflect the official G1 releases.
    printf "  %s  %-8s  %-10s  %s\n" "0"  "0.1B"    "~0.4 GB"  "testing / very limited hardware"
    printf "  %s  %-8s  %-10s  %s\n" "1"  "0.4B"    "~0.9 GB"  "lightweight CPUs and laptops"
    printf "  %s  %-8s  %-10s  %s\n" "2"  "1.5B"    "~3.0 GB"  "balanced performance"
    printf "  %s  %-8s  %-10s  %s\n" "3"  "2.9B"    "~6.0 GB"  "8–12 GB VRAM GPUs"
    printf "  %s  %-8s  %-10s  %s\n" "4"  "7.2B"    "~14.0 GB" "16 GB VRAM (best quality)"
    printf "  %s  %-8s  %-10s  %s\n" "5"  "skip"    ""          "I will add a model manually"
    echo ""

    # Select default choice based on recommended model size computed earlier.
    case "$RECOMMENDED_MODEL" in
        7.2B) DEFAULT_CHOICE=4 ;;
        2.9B) DEFAULT_CHOICE=3 ;;
        1.5B) DEFAULT_CHOICE=2 ;;
        0.4B) DEFAULT_CHOICE=1 ;;
        *)    DEFAULT_CHOICE=0 ;;
    esac

    read -r -p "  Choose [0-5, default ${DEFAULT_CHOICE}]: " MODEL_CHOICE
    MODEL_CHOICE="${MODEL_CHOICE:-$DEFAULT_CHOICE}"

    # Map choices to size strings used for filename filtering
    declare -A MODEL_SIZES=(
        [0]="0.1b"
        [1]="0.4b"
        [2]="1.5b"
        [3]="2.9b"
        [4]="7.2b"
    )

    if [[ "$MODEL_CHOICE" == "5" ]]; then
        warn "Skipping model download."
        info "Place a .pth or .onnx file in models/ and set MODEL_PATH in config_local.py"
        info "Official G1 models: https://huggingface.co/BlinkDL/rwkv7-g1"
    elif [[ -n "${MODEL_SIZES[$MODEL_CHOICE]:-}" ]]; then
        CHOSEN_SIZE="${MODEL_SIZES[$MODEL_CHOICE]}"
        echo ""
        info "Downloading RWKV-7 G1 ${CHOSEN_SIZE} from Hugging Face…"
        info "This may take a while. The file lands in models/."
        echo ""
        # Download only the file(s) matching the chosen size string.
        "$PYTHON_VENV" "$SCRIPT_DIR/scripts/download_models.py" \
            --repo BlinkDL/rwkv7-g1 \
            --dest "$SCRIPT_DIR/models" \
            --pattern .pth .onnx \
            --size "$CHOSEN_SIZE" || true
        # After download, pick the first file matching the chosen size string.
        MODEL_DEST="$(ls "$SCRIPT_DIR"/models/*${CHOSEN_SIZE}*.pth 2>/dev/null | head -1 || true)"
        if [[ -n "$MODEL_DEST" ]]; then
            ok "Selected model: $(basename "$MODEL_DEST")"
            sed -i "s|^MODEL_PATH.*=.*|MODEL_PATH = \"$MODEL_DEST\"|" "$SCRIPT_DIR/config_local.py" 2>/dev/null || true
            ok "config_local.py updated with model path"
        else
            warn "No downloaded file matched size ${CHOSEN_SIZE}. Check models/ manually and set MODEL_PATH in config_local.py"
        fi
    else
        warn "Invalid choice — skipping. Set MODEL_PATH in config_local.py when ready."
    fi
fi

# ── 9. Sanity check ───────────────────────────────────────────────────────────
hdr "Sanity check"
"$PYTHON_VENV" -m py_compile \
    "$SCRIPT_DIR/main.py" \
    "$SCRIPT_DIR/web/server.py" \
    "$SCRIPT_DIR/config.py"
ok "Python syntax OK"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "${DIM}────────────────────────────────────────────${RESET}"
echo "${GREEN}${BOLD}Done!${RESET}"
echo ""
echo "  ${BOLD}Start:${RESET}   ${CYAN}./start.sh${RESET}"
echo "  ${BOLD}Stop:${RESET}    ${CYAN}./stop.sh${RESET}"
echo "  ${BOLD}Update:${RESET}  ${CYAN}./install.sh --update${RESET}"
echo ""
echo "  ${DIM}After first start, Flint will show onboarding if no real model is active.${RESET}"
echo "  ${DIM}You can upload a model from the web UI, import prior chats, and keep using official RWKV-7 G1 presets as the normal path.${RESET}"
echo "  ${DIM}Edit config_local.py to change names, persona or generation settings.${RESET}"
echo ""
