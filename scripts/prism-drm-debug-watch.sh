#!/bin/bash
# prism-drm-debug-watch — crank drm.debug briefly when prism quarantines a
# GPU, so the kernel logs WHY it rejects the recovery attempts' command
# submissions (docs/device-loss-recovery.md; the observed -22 EINVALs from
# amdgpu_cs are silent at the default log level).
#
# Runs as root (a systemd system service — see prism-drm-debug-watch.service).
# Watches the user journal for the quarantine line emitted by
# PrismState::note_gpu_lost; on match, enables drm.debug categories
# core|driver|kms|prime (0xf — deliberately NOT vblank 0x20, which would
# spam per-refresh on the healthy GPU's live outputs) for DURATION seconds,
# then restores 0. Cooldown bounds the volume if quarantines repeat.
# Recovery retries land at +2s/+4s/... so the window captures attempt >= 2.

set -u

DEBUG_VAL=0xf
DURATION=60
COOLDOWN=600
DEBUG_PARAM=/sys/module/drm/parameters/debug

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
            fi
            ;;
        esac
    done
