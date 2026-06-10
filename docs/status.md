# Status

What's built, what's verified, and what's left. Updated as work lands. For *how*
the pieces fit together see [architecture.md](architecture.md); for the backlog see
[deferred-work.md](deferred-work.md).

Verification legend: ✅ = verified on hardware (or user-verified in a live
session); ☐ = built, compiles, reviewed — but not yet exercised on hardware.
A ☐ is a claim about code, not about behavior.

## Snapshot — 2026-06-09

prism is a daily-drivable niri-style scrolling-tile compositor on its own
Vulkan/KMS stack: multi-output multi-GPU bringup, HDR output signaling with
per-display 3D-LUT calibration, and now full window management — scrolling +
floating layout, open/close/resize animations, rounded-corner decorations with
shadows, overview mode, window rules, config hot-reload — plus session
integration (systemd session units, Xwayland via xwayland-satellite, drag and
drop), screen capture (wlr-screencopy), and DRM leasing for VR (SteamVR +
Index). Protocol behavior is gated by a WLCS conformance harness.

**Most recent work**: a 2026-06-09 six-agent deep-dive review filed issues
#5–#32; 21 are fixed (GPU-resource lifetime via a deferred-destroy queue,
cross-GPU mirror synchronization, nonblocking IPC, live window transactions,
animation-snapshot coordinate fixes, null-buffer unmap, live window-rule
recompute, decoration fade/crop polish). Before that: overview mode, DnD (icon +
drop-target activation + a smithay fix for data-device-less drop targets), DRM
lease with runtime headset hotplug, window decorations, window rules, config
hot-reload, the focus-model rework, wlr-screencopy, ext-workspace +
foreign-toplevel, the prism-tune GUI, and the SDR drive-domain / measured-gamut
color reforms. **All 06-09 review fixes are runtime-unverified.**

## What's built

### Display, scanout, color

| Area | State |
|---|---|
| Vulkan instance/device/queue, per-GPU, DRM-node matched | ✅ |
| DRM enumeration, multi-card open, per-connector CRTC/mode pick | ✅ |
| Multi-output + multi-GPU bringup, vblank-driven double-buffered scanout | ✅ — verified on 7 outputs across both cards |
| VRR / Freesync at bringup; EDID-keyed per-output config | ✅ |
| Renderer: decode → fp16 BT.2020 intermediate → synthesized per-output encode | ✅ |
| RGB decode (8/10-bit), YUV decode (NV12/P010), premultiplied-alpha handling | ✅ |
| Per-surface primary conversion (sRGB/Display-P3/BT.2020 → BT.2020) | ✅ |
| Encode transfers: sRGB, PQ, linear; KMS HDR signaling (PQ metadata, 10-bit, fp16 scanout); survives DPMS + VT handoff | ✅ |
| HDR video end-to-end (Firefox P010, native + cross-GPU); scRGB HDR swapchains | ✅ |
| Per-output calibration: 3D LUT (EDID-keyed, live IPC reload), CTM, response curves; measured-gamut hue-preserving bake; SDR drive-domain LUTs | ✅ tooling / ☐ gamut + SDR reforms pending a full recalibration |
| Render intents (perceptual/relative/absolute) + reference-white anchoring + per-output brightness | ☐ |
| Cross-GPU mirror: dmabuf replication + copy path, both directions fence-synchronized | ✅ mirror / ☐ sync fixes (33cb4d1, 0cacc63) |
| Damage tracking: content tokens, per-output region tracker, decode scissor, occlusion culling, zero-damage frame skip, partial shm upload, dmabuf import cache | ✅ |
| GPU resource lifetime: deferred-destroy retire queue drained by fence serials (every `queue_submit2` must `note_submit` first) | ☐ |
| DRM lease (`wp_drm_lease_device_v1`) + runtime VR-headset connector hotplug | ✅ — SteamVR + Valve Index, plug/unplug live |

### Window management

| Area | State |
|---|---|
| Scrolling layout (niri port): columns, tabbed columns, workspaces per monitor, floating layer, fullscreen/maximize | ✅ |
| Unmapped-window stage: initial configure sized from the layout, window rules resolved pre-map | ☐ |
| Window rules: resolution, hot-reload recompute, title/app-id-change recompute, at-startup matcher | ☐ |
| Window open/close animations | ✅ (coord + z-order fixes 1b099f7 ☐) |
| Resize animation (size tween + snapshot crossfade); workspace-switch animation | ☐ |
| Decorations: rounded-corner focus ring/border (SDF), clip-to-geometry, drop shadows | ✅ |
| Decoration polish: tile fade covers decorations, workspace-band crop covers SDF elements (b7a27ab) | ☐ |
| Overview mode: render + toggle, pointer interaction, keyboard focus, hot corner | ✅ |
| Overview: touchpad gestures, spatial-movement grabs + DnD layout feed | ☐ |
| Window transactions: column co-resizes atomic (300 ms deadline, commit blockers) | ☐ |
| Focus model: keyboard focus derived from layout per-frame; focus-follows-mouse with niri's guards | ☐ |
| Interactive move/resize | ✅ |
| xdg popups: render, grabs, unconstrain; xdg-dialog modal floating | ✅ |
| Config hot-reload (file watcher + live re-apply; `outputs`/`debug` sections restart-only) | ☐ |

### Input

| Area | State |
|---|---|
| libinput via udev, keyboard + pointer, device hotplug, hit-testing | ✅ |
| Action dispatcher at niri parity (~150 actions); mouse/wheel/touchpad-scroll binds; bind cooldown + repeat | ☐ |
| `input { touchpad/mouse/… }` libinput device settings applied (device-add + reload) | ☐ |
| Pointer constraints + relative pointer (locked-cursor games; hint clamped to output) | ✅ |
| Cursor: client surfaces + `wp_cursor_shape`, themed, hardware plane, auto-hide options | ✅ |
| VT switching (Ctrl+Alt+Fn) + display/HDR restore on resume | ✅ |

### Protocols, session, capture

| Area | State |
|---|---|
| Core + xdg-shell + the optional set: viewporter, presentation, fractional-scale, content-type, xdg-activation, xdg-decoration, single-pixel-buffer, idle-notify/-inhibit, output-power-management, drm-syncobj | ✅ |
| `wlr_layer_shell`: 4 layers color-managed, exclusive zones, keyboard interactivity | ✅ — waybar, swaybg, fuzzel |
| Layer-shell popups (`get_popup`) | ☐ |
| `ext-session-lock-v1`: niri-port state machine (lock confirmed only after every powered output presents a locked frame), lock-only render path, lock-screen focus + pointer gating, `allow-when-locked` binds | ☐ — needs a swaylock pass |
| `wp_color_management_v1`: output + surface paths, feedback, `surface_exists`, version-gated events | ✅ output path / ☐ 06-09 fixes |
| Drag and drop: full data-device, drag-icon rendering, drop-target activation, smithay fork fix for data-device-less targets | ✅ — Firefox tab tear-off |
| Clipboard managers: wlr-data-control + ext-data-control (primary selection included) | ☐ — needs a cliphist/wl-paste --watch run |
| Xwayland: on-demand xwayland-satellite | ✅ |
| Screen capture: wlr-screencopy, SHM + dmabuf, async from the render loop | ✅ grim / ☐ recording perf (see deferred-work) |
| ext-workspace-v1 + wlr-foreign-toplevel (+ ext list) for status bars | ☐ — needs a waybar run |
| `wp_alpha_modifier_v1` | ☐ |
| IPC: nonblocking per-connection socket; version/outputs/focused-output/workspaces/windows/output actions, LUT push over memfd | ✅ / ☐ nonblocking rework (aad4323) + introspection requests; EventStream not implemented |
| Session: `prism-session` (systemd --user units, real user bus), spawn-at-startup + env, child signal-mask reset | ✅ |
| WLCS protocol-conformance harness (curated subset, 6-entry expected-failures allowlist) | ✅ |
| Packaging: PKGBUILD + .SRCINFO (AUR push pending release tag) | ☐ |

## What's not done

Details and triggers in [deferred-work.md](deferred-work.md). Feature-sized gaps
are tracked as GitHub issues. The notable ones:

- **IME / text-input** (#26) — none.
- **IPC EventStream** (#28) — one-shot Workspaces/Windows/Outputs work;
  the long-lived event-stream form (status-bar push updates) doesn't exist.
- **Insert hint + tab indicator** (#29) — layout computes them; rendering is stubbed.
- **Layer rules** — the `layer-rule` config section parses but nothing computes
  `ResolvedLayerRules`; the section is inert (no issue filed yet).
- **Tone mapping** — decode (over-bright input) and encode (above-peak) both
  hard-clip. No EETF/Reinhard/Hable operator yet.
- **Touch input** — no `wl_touch` (keyboard + pointer are wired).
- **Desktop output hotplug** — desktop outputs are opened at startup; only
  DRM-lease (VR) connectors hot-plug at runtime. CRTC rebinding isn't done either.
- **Recording performance** — wlr-screencopy is correct but wf-recorder lags;
  diagnosis in [screen-capture.md](screen-capture.md). PipeWire/portal capture
  not started.
- **Encode-side mixed-primaries blending**, **subpixel FIR + dither fragments**,
  **HLG/BT.1886 decode**, **scanout bandwidth opts** (DCC import,
  `FB_DAMAGE_CLIPS`, test commits), **linux-dmabuf v4**, **tablet** — unchanged,
  see deferred-work.

## Subcommands

```
prism                 headless smoke suite: Vulkan probe + DRM enum + tracer self-tests
prism scanout [output]              TTY: clear an output to green (vkCmdClearColorImage), hold 5s
prism gradient [output] [8|10]      TTY: render a gradient through decode→encode, hold 5s
                                      depth defaults to 10 (XR30 + max_bpc=10); pass 8 for XR24
prism run [output] [8|10] [--session]   TTY: integrated mode — wayland server + scanout + vblank
                                      render on every connected output across every opened card.
                                      --session imports the environment into systemd/D-Bus
                                      (portals, keyring). Ctrl-C to exit.
                                      Env: CARDS (comma-sep DRM paths; default both cards),
                                           PRISM_MAX_RUNTIME_SECS (wall-clock self-shutdown),
                                           PRISM_WATCHDOG_SECS (hard-kill), PRISM_CRUMBS (breadcrumb file),
                                           PRISM_VK_VALIDATION (Vulkan validation layers)
prism wayland         wayland server on wayland-N; logs surface lifecycle, no rendering
```

For a real session, launch via `resources/prism-session` (systemd --user units;
wired up as a wayland-session by the package) rather than `prism run` directly.

`prism-tune` (separate binary — color calibration + inspection, talks to a running compositor):

```
prism-tune gui                 damascene GUI: IPC control panel, BT.2020 frame inspector,
                               3D gamut point cloud, effective-LUT lattice inspector
prism-tune characterize        measure per-channel response (gain/gamma) with the colorimeter
prism-tune calibrate           derive a correction
prism-tune calibrate-lut3d     generate a 3D LUT (.lut + .csv) and push it live over IPC
prism-tune rebake-lut3d        re-derive a .lut offline from saved measurements
prism-tune validate-lut3d      software validation of the color pipeline against a LUT
prism-tune msg <version|outputs|focused-output|output ...>   IPC query/command to the compositor
```

The headless suite (`prism` with no args) runs three tracer self-tests:
GBM→Vulkan→clear→CPU readback (dmabuf import infra), the dmabuf-handler import path
(wayland-side dmabuf flow), and a decode→encode gradient (sRGB OETF anchor points).

## Test helpers

- `scripts/tty-test.sh [seconds]` — launches `prism run` + the in-tree
  `prism-shmtest` client, polls for the socket, runs, dumps log tails. Env:
  `OUTPUT`, `DEPTH`, `RUST_LOG`.
- `scripts/firefox-test.sh` — `prism run` + Firefox in a TTY with
  `prism_protocols=debug` and Firefox `MOZ_LOG` for the HDR-video path. `NO_HDR=1`
  disables Firefox Wayland HDR (isolates the HDR path from plain SDR rendering).
- `scripts/layer-test.sh [seconds]` — `prism run` + `swaybg` (Background
  wallpaper) + `waybar` (Top bar) + a window, to verify `wlr_layer_shell`
  Z-order and color management. Env: `OUTPUT`/`DEPTH`, `WALLPAPER`(_COLOR),
  `BAR_BIN`/`WINDOW_BIN`, `NO_WALLPAPER`/`NO_BAR`/`NO_WINDOW`.
- `crates/prism-wlcs/conformance/run.sh /path/to/wlcs` — protocol-conformance gate.
