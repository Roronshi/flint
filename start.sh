#!/usr/bin/env bash
# start.sh — Start the Flint backend and open the UI in a browser.
#
# Usage:
#   ./start.sh          — start in background, open browser
#   ./start.sh --fg     — start in foreground (logs to stdout)
#   ./start.sh --port N — use a different port (default 8000)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="$SCRIPT_DIR/data"
PID_FILE="$DATA_DIR/flint.pid"
LOG_FILE="$DATA_DIR/flint.log"
PORT=8000
FOREGROUND=false
RESTART=false

# ── Colours ───────────────────────────────────────────────────────────────────
RESET=$'\033[0m'; BOLD=$'\033[1m'; DIM=$'\033[2m'
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; CYAN=$'\033[36m'

ok()   { echo "${GREEN}✓${RESET}  $*"; }
warn() { echo "${YELLOW}⚠${RESET}  $*"; }
err()  { echo "${RED}✗${RESET}  $*" >&2; }
info() { echo "${DIM}  $*${RESET}"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
_open_browser() {
    local target="$1"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$target" &>/dev/null &
    elif command -v open &>/dev/null; then
        open "$target"
    elif command -v wslview &>/dev/null; then
        wslview "$target"
    fi
}


# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fg|--foreground) FOREGROUND=true ;;
        --restart)         RESTART=true ;;
        --port) PORT="${2:-8000}"; shift ;;
        --port=*) PORT="${1#*=}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

URL="http://localhost:${PORT}"

echo ""
echo "${BOLD}${CYAN}Flint${RESET}"
echo "${DIM}────────────────────────────────────────────${RESET}"

# ── Preflight checks ──────────────────────────────────────────────────────────

if [[ ! -d "$VENV_DIR" ]]; then
    err "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/config_local.py" ]]; then
    err "config_local.py not found. Run ./install.sh first."
    exit 1
fi

mkdir -p "$DATA_DIR"

# ── Helper: find PID owning port ─────────────────────────────────────────────
_pid_on_port() {
    local pid
    pid=$(ss -tlnp "sport = :$PORT" 2>/dev/null \
          | grep -oP 'pid=\K[0-9]+' | head -1)
    if [[ -z "$pid" ]]; then
        pid=$(lsof -ti "tcp:$PORT" 2>/dev/null | head -1)
    fi
    echo "$pid"
}

# ── Already running? ──────────────────────────────────────────────────────────
# Check both PID file and the actual port — whichever tells the truth
RUNNING_PID=""
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        RUNNING_PID="$OLD_PID"
    else
        rm -f "$PID_FILE"
    fi
fi
if [[ -z "$RUNNING_PID" ]]; then
    PORT_PID=$(_pid_on_port)
    [[ -n "$PORT_PID" ]] && RUNNING_PID="$PORT_PID"
fi

_kill_and_wait() {
    local pid="$1"
    info "Stopping old process (PID $pid)…"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
}

if [[ -n "$RUNNING_PID" ]]; then
    if [[ "$RESTART" == true ]]; then
        info "Restarting Flint (stopping PID $RUNNING_PID)…"
        _kill_and_wait "$RUNNING_PID"
    else
        warn "Flint is already running (PID $RUNNING_PID)"
        info "Use './start.sh --restart' to stop it and start fresh."
        echo ""
        echo "  ${BOLD}UI:${RESET}   ${CYAN}$URL${RESET}"
        echo "  ${BOLD}Stop:${RESET} ${CYAN}./stop.sh${RESET}"
        echo ""
        _open_browser "$URL" 2>/dev/null || true
        exit 0
    fi
fi

# ── Foreground mode ───────────────────────────────────────────────────────────
if [[ "$FOREGROUND" == true ]]; then
    echo ""
    ok "Starting in foreground on $URL"
    info "Press Ctrl-C to stop."
    echo ""
    cd "$SCRIPT_DIR"
    exec "$VENV_DIR/bin/uvicorn" web.server:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info
fi

# ── Background mode ───────────────────────────────────────────────────────────
info "Starting Flint backend…"

cd "$SCRIPT_DIR"
nohup "$VENV_DIR/bin/uvicorn" web.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    >> "$LOG_FILE" 2>&1 &

FLINT_PID=$!
echo "$FLINT_PID" > "$PID_FILE"

# ── Wait for the server to become ready (max 60 s) ───────────────────────────
echo ""
printf "  Waiting for backend"
READY=false
for i in $(seq 1 60); do
    if curl -sf "$URL/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    # Check if the process died already
    if ! kill -0 "$FLINT_PID" 2>/dev/null; then
        echo ""
        err "Backend process exited unexpectedly."
        info "Check the log: tail -40 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    printf "."
    sleep 1
done
echo ""

if [[ "$READY" == false ]]; then
    warn "Backend did not respond within 60 s (model may still be loading)."
    info "Check: tail -f $LOG_FILE"
fi

ok "Flint is running (PID $FLINT_PID)"
echo ""
echo "  ${BOLD}UI:${RESET}   ${CYAN}$URL${RESET}"
echo "  ${BOLD}Logs:${RESET} ${DIM}tail -f $LOG_FILE${RESET}"
echo "  ${BOLD}Stop:${RESET} ${CYAN}./stop.sh${RESET}"
echo ""

# ── Open browser ─────────────────────────────────────────────────────────────
if [[ "${FLINT_NO_BROWSER:-}" != "1" ]]; then
    _open_browser "$URL"
fi
