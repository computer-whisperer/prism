#!/usr/bin/env bash
# Launch prism + wlr_layer_shell clients (a wallpaper + a bar) + a window in
# one shot, to verify layer-shell rendering. A sibling of scripts/tty-test.sh,
# specialized for the layer-shell Z-order + color path:
#
#   - swaybg  → a Background-layer wallpaper. Should sit BEHIND windows.
#   - waybar  → a Top-layer bar.            Should sit ABOVE windows.
#   - a window (terminal) in between, so the Z-order is visible at a glance.
#
# What to look for on the monitors (especially the HDR output):
#   - wallpaper behind the window, bar above it (correct layer Z-order);
#   - the wallpaper's solid color + the bar render at sane SDR luminance —
#     NOT blown out to peak / wrong gamma. Layer surfaces go through the same
#     color-managed decode→encode path as windows, so a sRGB-grey wallpaper
#     should look mid-grey on a PQ panel, not white.
#
# Usage:
#   scripts/layer-test.sh [seconds]
#
# Env knobs:
#   OUTPUT             prism's "run [output]" arg (e.g. DP-4). Default: first connected.
#   DEPTH              "8" or "10". Default: 10. (Must be set together with OUTPUT —
#                       prism's CLI is positional; see tty-test.sh.)
#   WALLPAPER          image path for swaybg. Default: a solid color (below).
#   WALLPAPER_COLOR    swaybg solid color (rrggbb, no '#') when no image. Default: 808080
#                       (sRGB mid-grey — the clearest "is the gamma right?" check).
#   BAR_BIN / BAR_ARGS bar client + args. Default: waybar (uses your waybar config).
#   WINDOW_BIN/ARGS    window client. Default: first of alacritty / foot, else
#                       the in-tree prism-shmtest.
#   NO_WALLPAPER / NO_BAR / NO_WINDOW   if set (any value), skip that client.
#   PRISM_CRUMBS       breadcrumb file (default: ./prism.crumbs).
#   PRISM_WATCHDOG_SECS (default: seconds + 5).
#
# After it exits, look at:
#   - ./prism.log               compositor stdout/stderr
#   - ./prism-layer-bg.log      swaybg stdout/stderr
#   - ./prism-layer-bar.log     bar stdout/stderr
#   - ./prism-layer-win.log     window stdout/stderr
#   - ${PRISM_CRUMBS}           fsync'd per-frame breadcrumbs

set -euo pipefail

SECS=${1:-10}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRISM_BIN="$ROOT/target/release/prism"
PRISM_LOG="$ROOT/prism.log"
BG_LOG="$ROOT/prism-layer-bg.log"
BAR_LOG="$ROOT/prism-layer-bar.log"
WIN_LOG="$ROOT/prism-layer-win.log"
SCRIPT_LOG="$ROOT/layer-test.log"

# Mirror everything we print to a log file so we can inspect what happened
# after prism's TTY grab freezes the console.
: > "$SCRIPT_LOG"
exec > >(tee -a "$SCRIPT_LOG") 2>&1
say() { echo "[layer-test $(date +%H:%M:%S.%3N)] $*"; }
say "script start (pid=$$, args=[$*])"
trap 'rc=$?; say "ERR at line $LINENO (rc=$rc)"; exit $rc' ERR

export PRISM_CRUMBS=${PRISM_CRUMBS:-"$ROOT/prism.crumbs"}
export PRISM_WATCHDOG_SECS=${PRISM_WATCHDOG_SECS:-$((SECS + 5))}
export PRISM_MAX_RUNTIME_SECS=${PRISM_MAX_RUNTIME_SECS:-$SECS}
# prism_protocols=info surfaces the "layer_shell: surface created + mapped"
# lines; smithay=info catches protocol errors / client disconnects.
export RUST_LOG=${RUST_LOG:-"prism=info,prism_protocols=info,smithay=info"}
export RUST_BACKTRACE=${RUST_BACKTRACE:-full}

# Resolve the window client: prefer a real terminal so the Z-order is obvious,
# fall back to the in-tree shm test client (always present).
WINDOW_BIN=${WINDOW_BIN:-"$(command -v alacritty || command -v foot || echo "$ROOT/target/release/prism-shmtest")"}

# Build prism (+ prism-shmtest if it's our window fallback).
say " building prism"
BUILD_PKGS=(-p prism)
if [[ "$WINDOW_BIN" == *"prism-shmtest"* ]]; then
    BUILD_PKGS+=(-p prism-shmtest)
fi
cargo build --release "${BUILD_PKGS[@]}"

# Compose `prism run` args (positional output + depth, like tty-test.sh).
PRISM_ARGS=(run)
if [[ -n "${OUTPUT:-}" ]]; then
    PRISM_ARGS+=("$OUTPUT" "${DEPTH:-10}")
elif [[ -n "${DEPTH:-}" ]]; then
    say " WARNING: DEPTH set without OUTPUT — ignored (prism CLI is positional)"
fi

RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
say " XDG_RUNTIME_DIR=$RUNTIME_DIR"

# Launch prism. It grabs the TTY's DRM master so the console freezes —
# everything from here on is invisible until prism exits.
say " launching: prism ${PRISM_ARGS[*]} (seconds=$SECS, watchdog=${PRISM_WATCHDOG_SECS}s)"
: > "$PRISM_LOG"
: > "$BG_LOG"
: > "$BAR_LOG"
: > "$WIN_LOG"
: > "$PRISM_CRUMBS"

"$PRISM_BIN" "${PRISM_ARGS[@]}" >"$PRISM_LOG" 2>&1 &
PRISM_PID=$!
say " prism pid=$PRISM_PID"

# Track every client pid for cleanup.
CLIENT_PIDS=()
cleanup() {
    for pid in "${CLIENT_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
    done
    if kill -0 "$PRISM_PID" 2>/dev/null; then
        kill -TERM "$PRISM_PID" 2>/dev/null || true
        sleep 0.5
        kill -KILL "$PRISM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait for prism's wayland socket name to appear in its log (same approach as
# tty-test.sh — smithay re-uses the socket name across runs, so we can't diff).
SOCKET=""
for _ in $(seq 1 80); do  # up to ~8s
    sleep 0.1
    if ! kill -0 "$PRISM_PID" 2>/dev/null; then
        say " prism exited before socket appeared; see $PRISM_LOG"
        exit 1
    fi
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
    say " WARNING: $RUNTIME_DIR/$SOCKET does not exist as a socket — clients may fail to connect"
fi

# launch_client <label> <logfile> <bin> [args...]
# Backgrounds a client in prism's wayland env, records its pid, and reports
# whether it survived 500ms (a client that dies instantly usually logged why).
launch_client() {
    local label="$1" log="$2"
    shift 2
    if [[ ! -x "$1" && -z "$(command -v "$1" 2>/dev/null)" ]]; then
        say " $label: '$1' not found — skipping"
        return
    fi
    say " launching $label: $*"
    WAYLAND_DISPLAY="$SOCKET" XDG_RUNTIME_DIR="$RUNTIME_DIR" \
        "$@" >"$log" 2>&1 &
    local pid=$!
    CLIENT_PIDS+=("$pid")
    say "   $label pid=$pid"
    sleep 0.5
    if kill -0 "$pid" 2>/dev/null; then
        say "   $label alive after 500ms"
    else
        say "   $label exited immediately! head of $log:"
        head -n 10 "$log" 2>&1 | sed 's/^/     /'
    fi
}

# Wallpaper (Background layer). Solid color by default — the clearest check
# that a known sRGB value lands at the right luminance on a PQ panel.
if [[ -z "${NO_WALLPAPER:-}" ]]; then
    if [[ -n "${WALLPAPER:-}" ]]; then
        launch_client wallpaper "$BG_LOG" swaybg -i "$WALLPAPER" -m fill
    else
        launch_client wallpaper "$BG_LOG" swaybg -c "${WALLPAPER_COLOR:-808080}"
    fi
fi

# Bar (Top layer).
if [[ -z "${NO_BAR:-}" ]]; then
    # shellcheck disable=SC2206
    bar_args=(${BAR_ARGS:-})
    launch_client bar "$BAR_LOG" "${BAR_BIN:-waybar}" "${bar_args[@]}"
fi

# A normal window, so the layer Z-order is visible (wallpaper behind it, bar
# above it). prism-shmtest takes a seconds arg; real terminals don't, so only
# pass the seconds fallback when we ended up on shmtest.
if [[ -z "${NO_WINDOW:-}" ]]; then
    # shellcheck disable=SC2206
    win_args=(${WINDOW_ARGS:-})
    if [[ ${#win_args[@]} -eq 0 && "$WINDOW_BIN" == *"prism-shmtest"* ]]; then
        win_args=("$SECS")
    fi
    launch_client window "$WIN_LOG" "$WINDOW_BIN" "${win_args[@]}"
fi

say " wait for prism pid=$PRISM_PID"
PRISM_RC=0
wait "$PRISM_PID" || PRISM_RC=$?
say " prism exited (rc=$PRISM_RC)"

# Make sure the clients are gone too.
for pid in "${CLIENT_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
done
for pid in "${CLIENT_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
done

echo
say " summary"
echo "  compositor log: $PRISM_LOG"
echo "  wallpaper log:  $BG_LOG"
echo "  bar log:        $BAR_LOG"
echo "  window log:     $WIN_LOG"
echo "  breadcrumbs:    $PRISM_CRUMBS"
echo
echo "── prism layer-shell lines ──"
grep -aiE 'layer_shell|layer.surface|map_layer' "$PRISM_LOG" 2>/dev/null \
    | sed -E 's/\x1b\[[0-9;]*m//g' | tail -n 30 || true
echo
echo "── any prism errors / protocol rejects ──"
grep -aiE 'error|panic|reject|invalid|disconnect' "$PRISM_LOG" 2>/dev/null \
    | sed -E 's/\x1b\[[0-9;]*m//g' | tail -n 20 || true
