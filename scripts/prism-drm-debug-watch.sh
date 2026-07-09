#!/bin/bash
# prism-drm-debug-watch — arm kernel-side diagnostics briefly when prism
# quarantines a GPU (docs/device-loss-recovery.md). Two captures per episode:
#
# 1. drm.debug=0xf (core|driver|kms|prime; deliberately NOT vblank 0x20,
#    which would spam per-refresh on the healthy GPU's live outputs) —
#    logs the ioctl sequence around the rejected recovery submissions.
#    2026-07-08 episode result: the failing AMDGPU_CS returns -22 with NO
#    driver debug output at any of these levels — the EINVAL path is
#    print-free. Hence capture 2.
#
# 2. ftrace function_graph rooted at amdgpu_cs_ioctl with funcgraph-retval,
#    pid-filtered to the prism compositor, streamed to
#    /var/log/prism-cs-trace-<stamp>.log. The innermost traced function
#    returning 0xffffffea (-22) names the kernel path that rejects the CS.
#    (amdgpu_cs_pass1/pass2 are inlined and invisible; amdgpu_vm_ready and
#    the other non-inlined callees do appear.)
#
# Runs as root (a systemd system service — see prism-drm-debug-watch.service).
# Watches the user journal for the quarantine line emitted by
# PrismState::note_gpu_lost; on match, arms both captures for DURATION
# seconds, then restores everything. Cooldown bounds the volume if
# quarantines repeat. Recovery retries land at +2s/+4s/... so the window
# captures attempt >= 2.

set -u

DEBUG_VAL=0xf
DURATION=60
COOLDOWN=600
DEBUG_PARAM=/sys/module/drm/parameters/debug
TRACEFS=/sys/kernel/tracing
TRACE_BUF_KB=8192

arm_cs_ftrace() {
    local stamp outfile pid tid
    stamp=$(date +%Y%m%d-%H%M%S)
    outfile=/var/log/prism-cs-trace-${stamp}.log

    # The compositor, not the gpu-probe child (also argv[0] "prism").
    pid=$(pgrep -f -o 'prism run --session' || true)

    echo 0 >"$TRACEFS/tracing_on"
    echo nop >"$TRACEFS/current_tracer"
    echo >"$TRACEFS/trace"
    echo "$TRACE_BUF_KB" >"$TRACEFS/buffer_size_kb"
    echo amdgpu_cs_ioctl >"$TRACEFS/set_graph_function"

    # Trace only prism's threads; follow threads it spawns mid-window.
    echo >"$TRACEFS/set_ftrace_pid"
    if [[ -n "$pid" && -d /proc/$pid/task ]]; then
        for tid in /proc/"$pid"/task/*; do
            echo "${tid##*/}" >>"$TRACEFS/set_ftrace_pid"
        done
        echo 1 >"$TRACEFS/options/function-fork" 2>/dev/null || true
    else
        echo "prism pid not found; tracing amdgpu_cs_ioctl unfiltered"
    fi

    echo function_graph >"$TRACEFS/current_tracer"
    # Tracer-specific option: only exists while function_graph is active.
    echo 1 >"$TRACEFS/options/funcgraph-retval" 2>/dev/null ||
        echo "funcgraph-retval unsupported; trace will lack return values"

    # Drain the ring to disk so bursts can't overwrite themselves.
    cat "$TRACEFS/trace_pipe" >"$outfile" &
    local drain=$!
    echo 1 >"$TRACEFS/tracing_on"
    echo "cs ftrace armed for ${DURATION}s -> ${outfile} (pid=${pid:-none})"

    (
        sleep "$DURATION"
        echo 0 >"$TRACEFS/tracing_on"
        sleep 1 # let the drain catch up on the tail
        kill "$drain" 2>/dev/null
        # An open trace_pipe reader makes tracer/filter writes EBUSY; the
        # drain is the parent shell's child, so poll instead of `wait`.
        while kill -0 "$drain" 2>/dev/null; do sleep 0.2; done
        echo nop >"$TRACEFS/current_tracer"
        echo >"$TRACEFS/set_graph_function"
        echo >"$TRACEFS/set_ftrace_pid"
        echo 1408 >"$TRACEFS/buffer_size_kb"
        echo "cs ftrace disarmed; $(wc -l <"$outfile") lines in ${outfile}"
    ) &
}

last=0
# -n 0: tail only (never match a quarantine line from a past session).
journalctl -f -n 0 -o cat _SYSTEMD_USER_UNIT=prism.service |
    while IFS= read -r line; do
        case "$line" in
        *"quarantining GPU for recovery"*)
            now=$(date +%s)
            if ((now - last >= COOLDOWN)); then
                last=$now
                echo "quarantine detected; drm.debug=${DEBUG_VAL} for ${DURATION}s"
                echo "$DEBUG_VAL" >"$DEBUG_PARAM"
                (
                    sleep "$DURATION"
                    echo 0 >"$DEBUG_PARAM"
                    echo "drm.debug restored to 0"
                ) &
                arm_cs_ftrace
            fi
            ;;
        esac
    done
