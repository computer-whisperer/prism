# Status

What's built, what's verified, and what's left. Updated as work lands. For *how*
the pieces fit together see [architecture.md](architecture.md); for the backlog see
[deferred-work.md](deferred-work.md).

## Snapshot — 2026-05-24

prism is a working multi-output, multi-GPU compositor. It brings up every connected
output across both AMD GPUs with vblank-driven double-buffered scanout, composites
real clients (alacritty, mpv, Firefox, nautilus), takes keyboard + pointer input,
decodes YUV video (NV12/P010), drives **HDR output signaling** (config-driven PQ),
and plays **HDR video end-to-end** — native and cross-GPU-mirrored. Per-display
color calibration runs through a 3D-LUT encode path driven by `prism-tune`. Protocol
behavior is gated by a WLCS conformance harness.

**Most recent work** (the last push): YUV video decode (NV12/P010) and Firefox HDR
video end-to-end; per-surface primary conversion into BT.2020 (named sRGB /
Display-P3 / BT.2020); `wp_color_management_output_v1.get_output`; the 3D-LUT encode
calibration path; compositing fixes (`wp_viewport` source crop, subsurface z-order,
geometry-less window anchoring); real pointer hit-testing + focus re-evaluation; and
the WLCS conformance harness.

## What's built

| Area | State |
|---|---|
| Vulkan instance/device/queue, per-GPU, DRM-node matched | ✅ |
| DRM enumeration, multi-card open, per-connector CRTC/mode pick | ✅ |
| Multi-output + multi-GPU bringup, vblank-driven double-buffered scanout | ✅ — verified on 7 outputs across both cards |
| `wl_output` + xdg-output advertisement, per-output element mapping (`enter`/`leave`) | ✅ |
| VRR / Freesync enabled at bringup when the connector supports it | ✅ |
| EDID parsing (`EdidInfo`) → EDID-keyed `output "Make Model Serial"` config + per-output defaults | ✅ |
| Wayland protocols: compositor, xdg-shell, xdg-decoration, shm, single-pixel-buffer, linux-dmabuf, wl_seat, data-device, viewporter, presentation, fractional-scale, content-type, xdg-activation, drm-syncobj (advertise), wlr-layer-shell, ext-idle-notify, idle-inhibit, wlr-output-power-management | ✅ |
| dmabuf import → VkImage, modifier-negotiated; per-GPU replication; cross-GPU mirror | ✅ |
| shm upload, per-GPU | ✅ |
| Input: libinput via udev, keyboard + pointer capabilities, real dispatch; pointer hit-testing + focus tracking | ✅ |
| `wlr_layer_shell`: 4 layers color-managed in z-order, anchors/margins/exclusive-zones via `LayerMap`, exclusive-zone work-area, keyboard interactivity (Exclusive/OnDemand/None) | ✅ — user-verified (waybar, swaybg, fuzzel) |
| Idle / display sleep: `ext-idle-notify-v1` + `zwp_idle_inhibit` (swayidle), `zwlr_output_power_management` DPMS + `PowerOff/OnMonitors` actions; HDR/10-bit survives DPMS cycle | ✅ — user-verified (swayidle, wlopm, mpv inhibit) |
| Cursor: client `set_cursor` (shm surface cursors) + `wp_cursor_shape` named shapes, themed from `cursor {}` config, hardware cursor plane, scale-matched per output | ✅ — user-verified |
| Renderer: decode → fp16 BT.2020 intermediate → synthesized per-output encode | ✅ |
| RGB decode (8/10-bit, RGBA/BGRA order), YUV decode (NV12/P010) | ✅ |
| Per-surface primary conversion (sRGB/Display-P3/BT.2020 → BT.2020) | ✅ |
| Encode transfers: sRGB, PQ, linear | ✅ |
| Per-output calibration: 3D LUT (KDL file + EDID-keyed + live IPC reload), CTM, per-channel response gain/gamma | ✅ |
| KMS HDR signaling — `hdr` config block → `HDR_OUTPUT_METADATA` + `Colorspace=BT2020_RGB` + `max_bpc=10` + fp16 scanout + PQ encode; re-pushed across VT handoff; cleared on shutdown | ✅ |
| `wp_color_management_v1` output capability (`get_output`) | ✅ |
| HDR video end-to-end (Firefox P010, native + cross-GPU) | ✅ — user-verified |
| Color calibration tooling (`prism-tune`) | ✅ |
| WLCS protocol-conformance harness | ✅ — curated subset, 38 pass / 6 expected-fail |

## What's not done

Details and triggers in [deferred-work.md](deferred-work.md). The notable gaps:

- **Tone mapping** — decode (over-bright input) and encode (above-peak) both
  hard-clip. No EETF/Reinhard/Hable operator yet.
- **Touch input** — no `wl_touch` (keyboard + pointer are wired).
- **Output/connector runtime hotplug** — outputs are opened at startup; we don't
  react to connector add/remove (input-device hotplug *does* work, via udev). CRTC
  rebinding for already-occupied CRTCs isn't done either.
- **Encode-side mixed-primaries blending** — a BT.2020 HDR window and a BT.709 SDR
  window sharing one output blend with wrong chromaticity for the SDR portion (the
  conversion folds into a calibrated output's LUT but isn't general).
- **Subpixel FIR + dither encode fragments** — in the `EncodeFragment` enum but
  emitting either returns `MissingFeature` (needs multi-sample synthesis).
- **Remaining decode transfer functions** — HLG, BT.1886, parametric gamma are stubs
  (sRGB, linear, PQ are done).
- **Scanout bandwidth optimizations** — DCC multi-plane import, `FB_DAMAGE_CLIPS`,
  atomic test commits, per-output `LINEAR` policy hint.
- **A few protocols** — linux-dmabuf v4 modifier-aware feedback; idle-inhibit,
  relative-pointer, pointer-constraints, tablet.

## Subcommands

```
prism                 headless smoke suite: Vulkan probe + DRM enum + tracer self-tests
prism scanout [output]              TTY: clear an output to green (vkCmdClearColorImage), hold 5s
prism gradient [output] [8|10]      TTY: render a gradient through decode→encode, hold 5s
                                      depth defaults to 10 (XR30 + max_bpc=10); pass 8 for XR24
prism run [output] [8|10]           TTY: integrated mode — wayland server + scanout + vblank
                                      render on every connected output across every opened card.
                                      Ctrl-C to exit.
                                      Env: CARDS (comma-sep DRM paths; default both cards),
                                           PRISM_MAX_RUNTIME_SECS (wall-clock self-shutdown),
                                           PRISM_WATCHDOG_SECS (hard-kill), PRISM_CRUMBS (breadcrumb file),
                                           PRISM_VK_VALIDATION (Vulkan validation layers)
prism wayland         wayland server on wayland-N; logs surface lifecycle, no rendering
```

`prism-tune` (separate binary — color calibration, talks to a running compositor):

```
prism-tune characterize        measure per-channel response (gain/gamma) with the colorimeter
prism-tune calibrate           derive a correction
prism-tune calibrate-lut3d     generate a 3D LUT (.lut + .csv) and push it live over IPC
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
