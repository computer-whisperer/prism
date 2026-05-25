# prism

Vulkan-native, HDR-native Wayland compositor.

Color management is first-class from day 1: every render element carries its source
color description; every output its target. The renderer composites in a BT.2020
absolute-nits linear fp16 intermediate; the per-output encode pass calibrates,
tone-maps, and encodes for scanout (sRGB or PQ). Direct scanout is an optimization
on top of that path, not a separate path.

## Status

A working multi-output, multi-GPU compositor. It brings up every connected output
across both AMD GPUs with vblank-driven double-buffered scanout, takes keyboard and
pointer input, composites real clients (alacritty, mpv, Firefox, nautilus), decodes
YUV video (NV12/P010), drives config-driven **HDR output signaling** (PQ), and plays
**HDR video end-to-end** — native and cross-GPU-mirrored. Per-display color
calibration (3D LUT, CTM, response curve) is config-driven and EDID-keyed, with a
live-reload loop through `prism-tune`. Protocol behavior is gated by a WLCS
conformance harness.

Notable gaps: tone mapping (decode/encode still hard-clip), touch input,
output/connector runtime hotplug, and the subpixel-FIR + dither encode fragments.
See [docs/status.md](docs/status.md) for the full picture and
[docs/deferred-work.md](docs/deferred-work.md) for the backlog.

## Documentation

- [docs/status.md](docs/status.md) — what's built and verified, and the subcommand reference.
- [docs/architecture.md](docs/architecture.md) — crate layering, runtime object graph, renderer/protocol/DRM decisions.
- [docs/color-management.md](docs/color-management.md) — the color/HDR design, amdgpu KMS quirks, calibration data flow.
- [docs/reuse-map.md](docs/reuse-map.md) — what we depend on from smithay/niri vs. what we own.
- [docs/deferred-work.md](docs/deferred-work.md) — the backlog, grouped by subsystem.

## Workspace layout

| Crate | Role |
|---|---|
| `prism-frame` | Renderer-independent frame description (`Element`, `FrameDescription`, `ColorDescription`, color matrices, opaque handles) |
| `prism-renderer` | Vulkan renderer (`ash`). Per-GPU instance. Decode → composite → encode pipeline. |
| `prism-drm` | KMS frontend — `DrmCompositor`-equivalent + per-output scanout buffer pool + modifier negotiation |
| `prism-layout` | Window layout + per-surface state (ported from niri) |
| `prism-input` | Input routing + pointer hit-testing (libinput → seat → focused surface) |
| `prism-animation` | Spring/curve animation engine + frame clock (ported from niri) |
| `prism-config` | KDL config schema (ported from niri-config) |
| `prism-protocols` | Wayland protocol wiring (smithay handlers), the `PrismState` machine, buffer import |
| `prism-ipc` | IPC types for talking to a running compositor |
| `prism-tune` | Closed-loop color calibration tool (IPC client + colorimeter + patch surface) |
| `prism-wlcs` | WLCS conformance-test integration shim (cdylib) |
| `prism-shmtest` | Minimal xdg-shell shm client for end-to-end TTY testing |
| `prism` | The compositor binary |

The load-bearing boundary is `prism-frame`: layout/input/protocol code describes
frames without ever seeing a Vulkan type, which keeps renderer concerns from
leaking outward (and lets the protocol stack run headless for conformance tests).
The renderer is one concrete `ash`-backed type per GPU — no `Renderer` trait
polymorphism. See [docs/architecture.md](docs/architecture.md).

## Building

```sh
cargo check
```

Needs Vulkan dev headers, libinput, libseat, libdrm, libgbm, wayland-server, and
`glslangValidator` (the build shells out to it to compile shaders). On Arch:

```sh
sudo pacman -S vulkan-headers vulkan-validation-layers libinput libseat libdrm \
               wayland mesa glslang
```

## Running

`prism run` drives every connected output on the TTY; `scripts/tty-test.sh` launches
it alongside the in-tree `prism-shmtest` client. The full subcommand reference
(including `prism-tune` for calibration) is in [docs/status.md](docs/status.md).

## License

GPL-3.0-or-later (matching the niri / smithay ecosystem we depend on).
