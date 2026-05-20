# prism

Vulkan-native, HDR-native Wayland compositor.

Color management is first-class from day 1: every render element carries its source color description; every output its target. The renderer composites in a BT.2020 absolute-nits linear fp16 intermediate; per-output postprocess tone-maps and encodes for scanout (sRGB or PQ). Direct scanout is an optimization on top of that path, not a separate path.

## Status

Pre-tracer skeleton. Architectural decisions are in [`../docs/`](../docs/):

- [phase-2-backend-notes.md](../docs/phase-2-backend-notes.md) — color-management architecture (the brain dump)
- [phase-2-reuse-map.md](../docs/phase-2-reuse-map.md) — smithay/niri reuse strategy (cut line, dependencies)

## Workspace layout

| Crate | Role |
|---|---|
| `prism-frame` | Renderer-independent frame description (`Element`, `FrameDescription`, `ColorDescription`, opaque handles) |
| `prism-renderer` | Vulkan renderer (ash). Per-GPU instance. |
| `prism-drm` | KMS frontend — DrmCompositor-equivalent + scanout buffer pool |
| `prism-layout` | Window layout (ported from niri) |
| `prism-input` | Input routing + bindings (smithay libinput backend) |
| `prism-config` | KDL config schema |
| `prism-protocols` | Wayland protocol wiring (smithay handlers) |
| `prism` | Binary |

The layering boundary at `prism-frame` keeps renderer concerns from leaking into layout/input/protocol code. The renderer is one concrete type (`ash`-backed); there is no `Renderer` trait polymorphism axis to maintain.

## Building

```sh
cargo check
```

Needs Vulkan dev headers + libinput + libseat + libdrm + libgbm + wayland-server in the system. On Arch:

```sh
sudo pacman -S vulkan-headers vulkan-validation-layers libinput libseat libdrm wayland mesa
```

## License

GPL-3.0-or-later (matching the niri / smithay ecosystem we depend on).
