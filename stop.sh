#!/usr/bin/env bash
# stop.sh — Stop the Flint backend gracefully.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/data/flint.pid"
PORT="${FLINT_PORT:-8000}"

RESET=$'\033[0m'; BOLD=$'\033[1m'; DIM=$'\033[2m'
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'

ok()   { echo "${GREEN}✓${RESET}  $*"; }
warn() { echo "${YELLOW}⚠${RESET}  $*"; }
err()  { echo "${RED}✗${RESET}  $*" >&2; }

echo ""

# ── Helper: find PID owning port ─────────────────────────────────────────────
_pid_on_port() {
    # Try ss first (iproute2), fall back to lsof
    local pid
    pid=$(ss -tlnp "sport = :$PORT" 2>/dev/null \
          | grep -oP 'pid=\K[0-9]+' | head -1)
    if [[ -z "$pid" ]]; then
        pid=$(lsof -ti "tcp:$PORT" 2>/dev/null | head -1)
    fi
    echo "$pid"
}

# ── Helper: kill a PID and wait ───────────────────────────────────────────────
_kill_and_wait() {
    local pid="$1" label="$2"
    echo "  Stopping $label (PID $pid)…"
    kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "Process did not exit cleanly — sending SIGKILL."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi
}

# ── Collect PIDs to kill ──────────────────────────────────────────────────────
PIDS=()

# From PID file
if [[ -f "$PID_FILE" ]]; then
    FILE_PID=$(cat "$PID_FILE")
    if kill -0 "$FILE_PID" 2>/dev/null; then
        PIDS+=("$FILE_PID")
    fi
    rm -f "$PID_FILE"
fi

# Whatever is actually holding the port (catches stale-PID situations)
PORT_PID=$(_pid_on_port)
if [[ -n "$PORT_PID" ]] && [[ "$PORT_PID" != "${PIDS[0]:-}" ]]; then
    PIDS+=("$PORT_PID")
fi

# ── Nothing to kill ───────────────────────────────────────────────────────────
if [[ ${#PIDS[@]} -eq 0 ]]; then
    ok "Flint is not running."
    echo ""
    exit 0
fi

# ── Kill everything found ─────────────────────────────────────────────────────
for pid in "${PIDS[@]}"; do
    _kill_and_wait "$pid" "Flint"
done

# ── Final confirmation ────────────────────────────────────────────────────────
LEFTOVER=$(_pid_on_port)
if [[ -n "$LEFTOVER" ]]; then
    err "Port $PORT is still occupied by PID $LEFTOVER — something else may be running on it."
    echo ""
    exit 1
fi

ok "Flint stopped. Port $PORT is free."
echo ""
