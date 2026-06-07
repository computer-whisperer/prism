# Reuse map — smithay & niri

The dependency strategy: what code prism depends on, what it copies, and what it
replaces. The companion to [color-management.md](color-management.md) (the
architecture we wanted) and [architecture.md](architecture.md) (what got built).

## TL;DR

prism is a Vulkan-native, HDR-native compositor — but only the **renderer** and the
**renderer-coupled parts of the KMS frontend** are new code. Everything else flows
from crates.io smithay as a dependency, plus copy-and-modify of select niri modules
(layout, input, animation, frame clock, config).

**No fork maintenance.** crates.io smithay is the version we use; if it doesn't do
what we need at the protocol/utils/drm level, we depend on it and work around. If it
*can't* do what we need at the renderer level — that's exactly the layer we're
replacing.

The pinned smithay rev is `85f83ab6` (upstream master). We carry one local
patch on top of it — a Cargo `[patch]` onto the `prism/dnd-no-data-device-cancel`
branch of the computer-whisperer/smithay fork — fixing a DnD bug where a drop on
a client with no `wl_data_device` reports `dnd_drop_performed` instead of
`cancelled`. Drop the patch when it lands upstream.

## The cut line, in one sentence

We replace `smithay::backend::renderer::*` and
`smithay::backend::drm::compositor::DrmCompositor`, and we copy several niri
modules. Everything else is a dependency.

## Smithay piece-by-piece

| Module | Action | Notes |
|---|---|---|
| `smithay::utils` (Logical/Physical Point, Size, Rectangle, Scale, Transform, Buffer) | **Depend** | Pure data types, renderer-independent. The layout port already uses them. |
| `smithay::wayland::compositor` | **Depend** | Surface state, roles, commit hooks, `SurfaceData`/`UserDataMap`. Big, well-tested. |
| `smithay::wayland::shell::xdg` | **Depend** | xdg-shell (toplevels, popups, decoration). |
| `smithay::wayland::shell::wlr_layer` | **Depend** | layer-shell (panels, backgrounds, lockscreen). |
| `smithay::wayland::dmabuf` | **Depend** | Client dmabuf import + format negotiation. We hook `ImportDma` into our renderer. |
| `smithay::wayland::shm` | **Depend** | Shared-memory buffer import. |
| `smithay::wayland::seat` | **Depend** | Input focus, keyboard/pointer/touch protocol state. |
| `smithay::wayland::output` | **Depend** | wl_output advertisement, mode/scale/transform. |
| `smithay::wayland::presentation` | **Depend** | wp_presentation feedback. **Wired.** |
| `smithay::wayland::viewporter` | **Depend** | Surface src/dst rects. **Wired** (source crop honored). |
| `smithay::wayland::content_type` | **Depend** | wp_content_type — tone-map policy hint. **Wired.** |
| `smithay::wayland::fractional_scale` | **Depend** | wp_fractional_scale. **Wired.** |
| `smithay::wayland::xdg_activation` | **Depend** | **Wired.** |
| `smithay::wayland::idle_inhibit` / `xdg_decoration` / `xdg_foreign` / IME / pointer-constraints / tablet | **Depend** | Misc protocols, adopt as needed. |
| `smithay::wayland::security_context` | **Depend** | Sandboxing. Later. |
| `smithay::backend::drm::{DrmDevice, DrmSurface, DrmNode, NodeType}` | **Depend** | Ergonomic drm-rs wrapper: device open, connector/CRTC enum, atomic commit accumulation. |
| `smithay::backend::drm::compositor::DrmCompositor` | **Replace** | Generic over `R: Renderer + Bind<Dmabuf>` — bound to smithay's renderer trait. Our `prism-drm` is the equivalent that takes our Vulkan scanout output and submits to KMS. |
| `smithay::backend::allocator::{Dmabuf, Format, Modifier, Fourcc}` | **Depend** | Pure data types; the dmabuf is our import primitive into Vulkan. |
| `smithay::backend::allocator::gbm` | **Depend** | GBM allocator for scanout BOs + dmabuf import. |
| `smithay::backend::renderer::*` (Renderer/Frame/Texture traits; Gles/MultiRenderer; OutputDamageTracker; all element types) | **Replace** | The layer we intentionally rewrite: `prism-renderer`. Vulkan-native, HDR-native, one renderer per GPU, color-management first-class. |
| `smithay::backend::libinput` | **Depend** | Thin libinput wrapper. |
| `smithay::backend::session::{LibSeatSession, Session}` | **Depend** | seatd/logind, VT switching, device acquisition. |
| `smithay::backend::input` (event types) | **Depend** | Input event abstractions. |
| `smithay::desktop` (Space, Window, LayerSurface) | **Use selectively** | We don't use `Space` (layout owns placement, like niri), but we *do* use `Window`/`PopupManager`/surface-tree helpers (`Window::surface_under`, `under_from_surface_tree`) in pointer focus. |
| `smithay::backend::winit` / `x11` | **Don't use** | TTY/KMS only. |
| `smithay::reexports::*` | **Depend transitively** | drm-rs, wayland-server, wayland-protocols, calloop. |

### What `Replace` means concretely

1. **Renderer** (`prism-renderer`): own `ash`-based Vulkan renderer per GPU. Owns
   texture pool, the synthesized decode/encode shaders, the fp16 BT.2020
   absolute-nits intermediate, per-output color descriptions, the encode pass.
2. **Element / FrameDescription** (`prism-frame`): concrete data structures
   replacing smithay's `RenderElement` trait family. No `R: Renderer` generic.
3. **DrmCompositor-equivalent** (`prism-drm`): per-output frontend owning the
   scanout BO pool, running the renderer's frame, building the atomic commit
   (composing our color/HDR properties + scanout-plane assignment + modifier
   negotiation). Composite-everything-to-primary for now; direct scanout + overlay
   planes are post-MVP.

### What `Depend` means concretely

We hook smithay's traits **without implementing its `Renderer`/`Frame`/`Texture`
traits.** Implement `DmabufHandler::dmabuf_imported` and pass the dmabuf to our
renderer; implement `CompositorHandler::commit` and extract surface state into our
per-surface cache; own a `DrmDevice` per GPU and build atomic commits in `prism-drm`;
standard `LibSeatSession` + `LibinputInputBackend` integration. None of these
require exposing our renderer through smithay's renderer trait.

## Niri module-by-module

| Niri module | Action | Landed as |
|---|---|---|
| `niri/src/layout/` | **Copy + modify** | `prism-layout` — scrollable tiling, columns, workspaces, floating. Output side produces our `Element`/`FrameDescription`. |
| `niri/src/animation/` | **Copy + light modify** | `prism-animation` — spring/curve engine + the frame `Clock`. |
| `niri/src/input/` | **Copy + modify** | `prism-input` — routing, bindings, pointer focus (`contents_under` ported from `Niri::contents_under`). |
| `niri/src/frame_clock.rs` | **Copy** | in `prism-drm` — VRR-aware per-output pacing. Load-bearing (Freesync is a hard requirement). |
| `niri/src/config/` | **Copy + modify** | `prism-config` — KDL schema; we add color/HDR/calibration keys, drop what we don't carry. |
| `niri/src/window.rs` / `layer.rs` | **Copy + modify** | folded into `prism-layout` — per-surface wrappers; `Mapped::window_geometry()` diverges from niri (see [architecture.md](architecture.md)). |
| `niri/src/render_helpers/` | **Reference, don't copy** | The `NiriRenderer` trait + `OutputRenderElements` exist to bridge niri to smithay's `Renderer` — superfluous when we own the renderer. The shader *math* (PQ, primary matrices) ported into the synthesized chain. |
| `niri/src/backend/tty.rs` | **Reference, don't copy** | A roadmap of every KMS integration point (connect/disconnect, VT resume, mode change, EDID, gamma/color props) — but it uses smithay primitives we replace. Re-implemented against our renderer in `prism-drm`. |
| `niri/src/ipc/` | **Copy later → done** | `prism-ipc` (types) + `prism-tune msg` (client). Used by the calibration tool. |
| `niri/src/protocols/` | **Reference** | Adopt selectively. |
| `niri/src/niri.rs` (main state) | **Reference, don't copy** | Our `PrismState` (in `prism-protocols`) has a different shape (per-GPU renderers, our layout output type). |

### Why copy + modify niri's layout rather than depend on it

niri's layout is its UX heart, tightly coupled to its config/animations/input/surface
lifecycle, and not exposed as a library. Copying gives us freedom to evolve it
(color-aware, multi-monitor-aware), no transitive dependency on niri's renderer
integration, and the ability to delete what we don't use. Cost: no automatic
upstream improvements — acceptable, niri's layout is mature.

## New crates with no niri ancestor

- **`prism-tune`** — closed-loop color calibration (colorimeter + patch surface +
  IPC client). Replaces the phase-1 spyder scripts. See
  [color-management.md](color-management.md).
- **`prism-wlcs`** — WLCS conformance harness (cdylib). No niri shortcut here: niri
  doesn't use WLCS (it has `niri-visual-tests` + in-`src` tests), so this was ported
  from smithay's `wlcs_anvil`. See `crates/prism-wlcs/conformance/`.
- **`prism-shmtest`** — minimal xdg-shell shm client for TTY testing.

## Open questions tied to this strategy

1. **`DrmCompositor` reusable subset.** We replace it (generic over `R: Renderer`).
   Worth a look whether the non-renderer-bound logic (plane assignment, atomic
   commit assembly) could be lifted without the trait dependency. So far we
   reimplement.
2. **Color management protocol.** smithay doesn't implement `wp_color_management_v1`;
   we implement it as a smithay-style handler in `prism-protocols` (the
   `get_output` capability path is live). Contribute upstream later if it matures;
   our timeline shouldn't depend on an upstream PR.
3. **screencopy / screenshot.** Requires reading back our intermediate/scanout.
   Architecturally fine, adds Vulkan readback paths. Post-MVP.
4. **wayland-protocols version skew.** smithay pins specific staging-protocol
   versions; if a color protocol lands in a different `wayland-protocols` than
   smithay's pin we juggle versions. Manageable; flag if it bites.

## Updating this doc

Update when:
- A "Depend" entry turns out not to fit and we copy+modify instead.
- A "Replace" entry turns out to have a reusable sub-component we should depend on.
- A new niri module becomes relevant to port.
