#!/usr/bin/env bash
# Launch prism + a wayland client in one shot. Designed for TTY runs where
# the text console is frozen once prism grabs DRM master, so you can't type
# the client launch command after the fact.
#
# Usage:
#   scripts/tty-test.sh [seconds]
#
# Env knobs:
#   OUTPUT             prism's "run [output]" arg (e.g. DP-4). Default: first connected.
#   DEPTH              "8" or "10". Default: 10 (matches prism's default).
#   CLIENT_BIN         override client binary (default: target/release/prism-shmtest).
#   PRISM_CRUMBS       breadcrumb file (default: ./prism.crumbs).
#   PRISM_WATCHDOG_SECS (default: seconds + 5).
#   PRISM_MAX_FRAMES   (default: seconds * 60 + 30).
#
# After it exits, look at:
#   - ./prism.log         compositor stdout/stderr
#   - ./prism-client.log  client stdout/stderr
#   - ${PRISM_CRUMBS}     fsync'd per-frame breadcrumbs

set -euo pipefail

SECS=${1:-8}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLIENT_BIN=${CLIENT_BIN:-"$ROOT/target/release/prism-shmtest"}
PRISM_BIN="$ROOT/target/release/prism"
PRISM_LOG="$ROOT/prism.log"
CLIENT_LOG="$ROOT/prism-client.log"
SCRIPT_LOG="$ROOT/tty-test.log"

# Mirror everything we print to a log file so we can inspect what happened
# after prism's TTY grab freezes the console.
: > "$SCRIPT_LOG"
exec > >(tee -a "$SCRIPT_LOG") 2>&1
say() { echo "[tty-test $(date +%H:%M:%S.%3N)] $*"; }
say "script start (pid=$$, args=[$*])"
# Ensure every set -e abort is recoverable on stdout (logged with line number).
trap 'rc=$?; say "ERR at line $LINENO (rc=$rc)"; exit $rc' ERR

export PRISM_CRUMBS=${PRISM_CRUMBS:-"$ROOT/prism.crumbs"}
export PRISM_WATCHDOG_SECS=${PRISM_WATCHDOG_SECS:-$((SECS + 5))}
# Wall-clock shutdown trigger — prism flips `running=false` after this many
# seconds. Replaces the old PRISM_MAX_FRAMES cap which broke for multi-output
# (the frame counter scales with output count, so 510 frames at 5 × 60Hz
# fires in ~1.7s, not 8s).
export PRISM_MAX_RUNTIME_SECS=${PRISM_MAX_RUNTIME_SECS:-$SECS}
# Capture smithay's and wayland-server's logs too — we need them to see
# protocol errors that cause client disconnects but don't surface to our
# default `prism=info` filter.
export RUST_LOG=${RUST_LOG:-"prism=info,smithay=info"}
# Full backtraces on panic — the breadcrumb hook captures the message but
# the backtrace via stderr (→ PRISM_LOG) tells us where it happened.
export RUST_BACKTRACE=${RUST_BACKTRACE:-full}

# Build everything we need.
say " building prism + prism-shmtest"
cargo build --release -p prism -p prism-shmtest

# Compose `prism run` args. Both output + depth must be passed together
# because prism uses positional args; if you only want depth set, also set
# OUTPUT to the connector you want.
PRISM_ARGS=(run)
if [[ -n "${OUTPUT:-}" ]]; then
    PRISM_ARGS+=("$OUTPUT")
    PRISM_ARGS+=("${DEPTH:-10}")
elif [[ -n "${DEPTH:-}" ]]; then
    say " WARNING: DEPTH set without OUTPUT — ignored (prism CLI is positional)"
fi

RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
say " XDG_RUNTIME_DIR=$RUNTIME_DIR"

# Launch prism. It grabs the TTY's DRM master so the console freezes —
# everything from here on is invisible until prism exits.
say " launching: prism ${PRISM_ARGS[*]} (seconds=$SECS, watchdog=${PRISM_WATCHDOG_SECS}s)"
: > "$PRISM_LOG"
: > "$CLIENT_LOG"
: > "$PRISM_CRUMBS"

"$PRISM_BIN" "${PRISM_ARGS[@]}" >"$PRISM_LOG" 2>&1 &
PRISM_PID=$!
say " prism pid=$PRISM_PID"

# Cleanup hook in case we're interrupted.
cleanup() {
    if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
        kill -TERM "$CLIENT_PID" 2>/dev/null || true
    fi
    if kill -0 "$PRISM_PID" 2>/dev/null; then
        kill -TERM "$PRISM_PID" 2>/dev/null || true
        sleep 0.5
        kill -KILL "$PRISM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait for prism's wayland socket name to appear in its log. Prism logs
# `WAYLAND_DISPLAY=wayland-N` once the listening socket is bound. We can't
# diff XDG_RUNTIME_DIR for "new" sockets because smithay re-uses the same
# name across runs (the file persists, only the lockfile gates ownership).
SOCKET=""
for _ in $(seq 1 80); do  # up to ~8s
    sleep 0.1
    if ! kill -0 "$PRISM_PID" 2>/dev/null; then
        say " prism exited before socket appeared; see $PRISM_LOG"
        exit 1
    fi
    # `|| true` because grep returns 1 when prism hasn't logged the line
    # yet — pipefail+errexit would otherwise abort the script mid-poll.
    SOCKET=$(grep -oE 'WAYLAND_DISPLAY=[a-zA-Z0-9_.-]+' "$PRISM_LOG" 2>/dev/null \
        | head -n1 | cut -d= -f2 || true)
    [[ -n "$SOCKET" ]] && break
done

if [[ -z "$SOCKET" ]]; then
    say " no WAYLAND_DISPLAY line in $PRISM_LOG within 8s"
    exit 1
fi
say " prism socket: $SOCKET (file: $RUNTIME_DIR/$SOCKET)"
if [[ ! -S "$RUNTIME_DIR/$SOCKET" ]]; then
    say " WARNING: $RUNTIME_DIR/$SOCKET does not exist as a socket — client may fail to connect"
fi

# Launch the client. It runs in the background; once the SECS window ends
# (or prism's watchdog fires), prism exits and we clean up below.
say " launching client: $CLIENT_BIN ($SECS seconds)"
say "   WAYLAND_DISPLAY=$SOCKET XDG_RUNTIME_DIR=$RUNTIME_DIR"
WAYLAND_DISPLAY="$SOCKET" XDG_RUNTIME_DIR="$RUNTIME_DIR" \
    "$CLIENT_BIN" "$SECS" >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!
say " client pid=$CLIENT_PID"
sleep 0.5
if kill -0 "$CLIENT_PID" 2>/dev/null; then
    say " client is alive after 500ms"
else
    say " client exited immediately! head of client log:"
    head -n 10 "$CLIENT_LOG" 2>&1 | sed 's/^/   /'
fi

# Wait for prism to finish (either via PRISM_MAX_RUNTIME_SECS or watchdog).
# `wait ... || PRISM_RC=$?` keeps the real exit code while preventing the
# ERR trap from firing on prism's nonzero exit (which fires even under
# `set +e`).
say " wait for prism pid=$PRISM_PID"
PRISM_RC=0
wait "$PRISM_PID" || PRISM_RC=$?
say " prism exited (rc=$PRISM_RC)"

# Make sure the client is done too.
if kill -0 "$CLIENT_PID" 2>/dev/null; then
    sleep 0.3
    kill -TERM "$CLIENT_PID" 2>/dev/null || true
fi
wait "$CLIENT_PID" 2>/dev/null || true

echo
say " summary"
echo "  compositor log: $PRISM_LOG"
echo "  client log:     $CLIENT_LOG"
echo "  breadcrumbs:    $PRISM_CRUMBS"
echo
echo "── tail of breadcrumbs ──"
tail -n 20 "$PRISM_CRUMBS" 2>/dev/null || true
echo
echo "── tail of client log ──"
tail -n 10 "$CLIENT_LOG" 2>/dev/null || true
