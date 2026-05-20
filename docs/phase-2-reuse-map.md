# Phase-2 reuse map — smithay & niri

The dependency-strategy companion to [phase-2-backend-notes.md](phase-2-backend-notes.md). That doc covers *what color-management architecture we want*; this doc covers *what code we depend on, what we copy, and what we replace* to get there.

## TL;DR

We are building a Vulkan-native, HDR-native compositor — but only the **renderer** and the **renderer-coupled parts of the KMS frontend** are new code. Everything else flows from crates.io smithay as a dependency, plus copy-and-modify of select niri modules (layout, input, animation, frame clock, config).

**No fork maintenance.** No upstream PR cycles as a dependency path. crates.io smithay is the version we use; if it doesn't do what we need at the protocol/utils/drm level, we depend on it anyway and work around. If it *can't* do what we need at the renderer level, that's the case we're already replacing.

## The cut line, in one sentence

We replace `smithay::backend::renderer::*` and `smithay::backend::drm::compositor::DrmCompositor`, and we copy several niri modules. Everything else is a dependency.

## Smithay piece-by-piece

| Module | Action | Notes |
|---|---|---|
| `smithay::utils` (Logical, Physical, Point, Size, Rectangle, Scale, Transform, Buffer) | **Depend** | Pure data types. Renderer-independent. Niri's layout module already uses these — keeping them simplifies the layout port. |
| `smithay::wayland::compositor` | **Depend** | Surface state, role assignment, commit hooks, `SurfaceData`/`UserDataMap` machinery. Renderer-independent. Big code, well-tested, no reason to reinvent. |
| `smithay::wayland::shell::xdg` | **Depend** | xdg-shell (toplevels, popups, decoration negotiation). Protocol boilerplate. |
| `smithay::wayland::shell::wlr_layer` | **Depend** | layer-shell (panels, backgrounds, lockscreen). |
| `smithay::wayland::dmabuf` | **Depend** | Client buffer import via dmabuf. Format negotiation + protocol. We hook the `ImportDma` side into our renderer ourselves. |
| `smithay::wayland::shm` | **Depend** | Shared-memory buffer import. Less common path on modern clients; still needed. |
| `smithay::wayland::seat` | **Depend** | Input focus, keyboard/pointer/touch protocol state. |
| `smithay::wayland::output` | **Depend** | wl_output advertisement, mode/scale/transform. |
| `smithay::wayland::presentation` | **Depend** | wp_presentation feedback. Frame-timing signalling to clients. |
| `smithay::wayland::viewporter` | **Depend** | Surface src/dst rect transformations. Maps directly to our `Element` src/geometry fields. |
| `smithay::wayland::content_type` | **Depend** | wp_content_type — clients hint "video"/"game"/"photo". Useful for tone-map policy. |
| `smithay::wayland::fractional_scale` | **Depend** | wp_fractional_scale. |
| `smithay::wayland::idle_inhibit` / `idle_notify` | **Depend** | Idle management. |
| `smithay::wayland::xdg_activation` / `xdg_decoration` / `xdg_foreign` / ... | **Depend** | Misc xdg protocols. |
| `smithay::wayland::input_method` / `text_input` / `virtual_keyboard` | **Depend** | IME protocols. |
| `smithay::wayland::pointer_constraints` / `pointer_gestures` / `relative_pointer` | **Depend** | Pointer protocols (relevant for games, drawing apps). |
| `smithay::wayland::tablet_manager` | **Depend** | Tablet/stylus input. |
| `smithay::wayland::screencopy` (if available) / external screencopy | **Depend** (or build later) | Screen capture for OBS-style clients. Note: requires renderer-level support for reading back the scanout intermediate. |
| `smithay::wayland::security_context` | **Depend** | Sandboxing. |
| `smithay::backend::drm::{DrmDevice, DrmSurface, DrmNode, NodeType}` | **Depend** | Wraps drm-rs ergonomically: device open, connector/CRTC enumeration, mode handling, atomic commit accumulation. Renderer-independent. |
| `smithay::backend::drm::compositor::DrmCompositor<A, F, U, G, R>` | **Replace** | Generic over `R: Renderer + Bind<Dmabuf>` — bound to smithay's renderer trait. Plane assignment, direct-scanout decisions, atomic commit construction live here. We implement our equivalent that takes our Vulkan renderer's scanout output and submits to KMS. |
| `smithay::backend::allocator::{Dmabuf, Format, Modifier, Fourcc}` | **Depend** | Pure data types. The Dmabuf is our import-from-client primitive into Vulkan textures. |
| `smithay::backend::allocator::gbm` | **Depend** | GBM allocator for scanout buffers + dmabuf import. We hand the dmabufs to our renderer via `VK_EXT_image_drm_format_modifier`. |
| `smithay::backend::renderer::*` (Renderer, Frame, Texture traits; GlesRenderer, MultiRenderer; OutputDamageTracker; RenderElement and all element types) | **Replace** | This is the layer we are intentionally rewriting. Vulkan-native, HDR-native, single-renderer-per-GPU, color-management first-class. See [phase-2-backend-notes.md](phase-2-backend-notes.md). |
| `smithay::backend::libinput` | **Depend** | Tiny wrapper around `libinput`. |
| `smithay::backend::session::{LibSeatSession, Session}` | **Depend** | seatd/logind session management. VT switching, device acquisition. |
| `smithay::backend::input` (event types: PointerEvent, KeyboardEvent, ...) | **Depend** | Input event abstractions consumed by libinput backend. |
| `smithay::desktop` (Space, Window, LayerSurface) | **Don't use** | Niri doesn't use these heavily either; its layout module owns window placement. We follow suit — our layout produces `FrameDescription` directly. The smithay desktop types are oriented around a stacking-WM model. |
| `smithay::backend::winit` / `smithay::backend::x11` | **Don't use** | We're TTY/KMS only. No winit backend, no nested X11 backend. (Maybe revisit winit-backend for development convenience later; not MVP.) |
| `smithay::reexports::*` | **Depend transitively** | drm-rs, wayland-server, wayland-protocols, calloop, etc. via smithay. |

### What `Replace` actually means concretely

The replaced layer is roughly:

1. **Renderer**: own `ash`-based Vulkan renderer per GPU. Owns texture pool, shader programs, intermediate (BT.2020 absolute-nits linear fp16), output color descriptions, the postprocess pass.
2. **Element / FrameDescription**: concrete data structures (see [phase-2-element-design.md](phase-2-element-design.md) once written) that replace smithay's `RenderElement` trait family. No `R: Renderer` generic; one renderer.
3. **Damage tracker**: our renderer owns per-output damage tracking, keyed on `ElementId` for cross-frame correlation. Replaces `OutputDamageTracker`.
4. **DrmCompositor-equivalent**: per-output frontend that owns the scanout buffer pool, runs the renderer's frame, submits the atomic commit (composing properties from our color/HDR layer + the scanout plane assignment). For MVP, no plane optimization — composite-everything-to-primary and scan that out. Direct scanout + overlay planes are post-MVP optimizations.

### What `Depend` actually means concretely

We hook into smithay's traits where it exposes them, **without implementing its `Renderer`/`Frame`/`Texture` traits.** Concretely:

- For `wayland::dmabuf`: implement `DmabufHandler::dmabuf_imported(&mut self, ..., dmabuf: Dmabuf)` and pass the dmabuf to our renderer for Vulkan import. Track the resulting `TextureHandle` in surface user data.
- For `wayland::compositor`: implement `CompositorHandler`; on commit, extract surface state (buffer, src/dst rects, opaque region, damage, color description from wp_color_management_v1 once we add it), update our per-surface renderer-owned cache.
- For `backend::drm`: own a `DrmDevice` per GPU; on connector hotplug create a `DrmSurface`; our DrmCompositor-equivalent owns the surface and builds atomic commits.
- For `backend::session`: standard `LibSeatSession` integration — VT switch handlers, device pause/resume.
- For `backend::libinput`: standard `LibinputInputBackend` integration — events feed into our input router (ported from niri).

None of these require us to expose our renderer through smithay's renderer trait.

## Niri module-by-module

| Niri module | Action | Notes |
|---|---|---|
| `niri/src/layout/` | **Copy + modify** | Scrollable tiling, columns, workspaces, focus stacks. Heart of the UX. Currently produces `Vec<RenderElement<R: NiriRenderer>>`. We modify the output side to produce our `Vec<Element>` (or `FrameDescription` directly). Algorithms are renderer-independent. |
| `niri/src/animation/` | **Copy + light modify** | Spring/curve animation engine. Drives layout transitions. Outputs progress values consumed by layout; renderer-agnostic. |
| `niri/src/input/` | **Copy + modify** | Input routing, bindings, keybinds, mouse-button-to-action mapping. Smithay input events in, layout actions out. Some niri-specific actions ("focus column right", "consume window into column") map directly; others need adjustment. |
| `niri/src/frame_clock.rs` | **Copy** | VRR-aware per-output frame pacing. **Especially load-bearing for us** since Freesync is a hard requirement and most compositors get this wrong. Niri already runs per-output independent render loops with vblank-driven scheduling that adapts to VRR. We want this from day 1. |
| `niri/src/config/` (via `niri-config` crate) | **Copy + modify** | KDL config schema. We add color/HDR/calibration keys (much of which we already prototyped on the fork — `ColorConfig`, `HdrConfig`, `HdrMode`). We remove things we don't carry (xwayland config, etc., depending on scope). |
| `niri/src/window.rs` / `niri/src/layer.rs` | **Copy + modify** | Per-surface wrappers around smithay's `WlSurface` that carry niri-specific state (which workspace, fullscreen state, requested geometry, etc.). Renderer-coupled where they produce render elements — modify to produce our Elements. |
| `niri/src/render_helpers/` | **Reference, don't copy** | Niri's renderer-trait extensions (`NiriRenderer`), custom shaders (`hdr_pq.frag` etc.), `OutputRenderElements` enum. Most of this exists to bridge between niri's needs and smithay's `Renderer` trait — superfluous when we own the renderer. The shader code (PQ encoding, primary conversion matrices) ports as-is to Vulkan GLSL. |
| `niri/src/backend/tty.rs` | **Reference, don't copy** | This is the smithay-DrmCompositor + MultiRenderer integration. Useful as a *roadmap* of every KMS integration point (connector connect/disconnect, VT resume, mode change, EDID parsing, gamma props, ColorProps, HdrProps) but the integration itself uses smithay primitives we're replacing. We re-implement against our renderer. |
| `niri/src/backend/winit.rs` / `niri/src/backend/headless.rs` | **Don't use** | Backend variants for development/testing. Not MVP. |
| `niri/src/ipc/` (varlink `niri msg`) | **Copy later** | Useful for scripting + status bars. Not load-bearing for MVP. |
| `niri/src/protocols/` | **Reference** | Niri's custom protocols (output management, etc.). Adopt selectively. |
| `niri/src/utils/` | **Copy selectively** | Misc helpers. Take what's useful, leave the rest. |
| `niri/src/niri.rs` (main state struct) | **Reference, don't copy** | The compositor state machine that wires everything together. Our version exists but has a different shape (per-GPU renderers, different layout output type). The structure is informative. |

### Why copy + modify niri's layout rather than depend on it

niri's layout module is the project's UX heart. It's tightly coupled to the rest of niri (configuration, animations, input bindings, surface lifecycle). Trying to depend on a published niri crate for layout would require niri to expose the layout module as a library — which it doesn't, and which would create a coupling we don't want.

Copying gives us:
- Freedom to evolve the layout for our needs (different surface paradigms, color-aware layout decisions, multi-monitor-aware tiling, etc.) without coordinating with upstream.
- No transitive dependencies on niri's renderer integration we don't want.
- The ability to delete code we don't use.

The cost: we don't get upstream improvements automatically. Acceptable — niri's layout is mature and we don't expect a flood of upstream changes to chase.

## What we own vs. depend on, by LOC estimate

Order-of-magnitude estimates, for scoping intuition:

| Code | LOC | Source |
|---|---|---|
| Our Vulkan renderer (ash + shaders + intermediate + postprocess) | 5–8k | New |
| Our DrmCompositor-equivalent | 1–2k | New |
| Element + FrameDescription + color description types | <500 | New |
| Niri layout (ported) | ~8–10k | Copy from niri |
| Niri input + animation + frame_clock (ported) | ~3–5k | Copy from niri |
| Niri config schema (ported) | ~2k | Copy from niri |
| Glue / wiring / main state | ~2–3k | New |
| smithay (depended) | ~50k | Dependency |
| wayland-rs + drm-rs + ash + calloop (transitive) | ~100k+ | Dependency |

Our code: ~25k LOC. Of which ~15k is copied from niri. Of the ~10k we write fresh, the bulk is the renderer.

That's a tractable scope — comparable to a focused single-person project, achievable in 3-4 months of focused work to MVP, longer to daily-driver polish.

## Implications for the project structure

```
compositor/
├── crates/
│   ├── compositor-frame/      Element, FrameDescription, ColorDescription, opaque handles. No renderer deps.
│   ├── compositor-renderer/   The Vulkan renderer (ash). Consumes FrameDescription.
│   ├── compositor-drm/        Our DrmCompositor-equivalent. Depends on smithay::backend::drm + compositor-renderer.
│   ├── compositor-layout/     Ported niri layout. Produces FrameDescription. No renderer deps.
│   ├── compositor-input/      Ported niri input. Depends on smithay::backend::input/libinput.
│   ├── compositor-config/     Ported niri config. KDL schema.
│   ├── compositor-protocols/  Wayland protocol wiring (smithay handlers, dispatching to layout/input/renderer).
│   └── compositor/            The binary. Wires everything together.
```

Roughly. The crate split isn't load-bearing; the layering is. The data-structure boundary at `compositor-frame` is what keeps layout/input from leaking renderer concerns.

## Open questions tied to this strategy

1. **`smithay::backend::drm::compositor` reusability.** We've decided to replace it because it's generic over `R: Renderer + Bind<Dmabuf>`. Worth checking: is there a non-renderer-bound subset (plane assignment logic, atomic commit assembly) we could lift into our own DrmCompositor without taking the Renderer trait dependency? If yes, that's free code. If no, we reimplement.
2. **Color management protocol implementation.** Smithay doesn't implement `wp_color_management_v1`. We can either (a) implement it as a smithay-style handler in our codebase, contributing nothing upstream, or (b) implement it as a contribution to smithay that we depend on. Per the upstream-first principle: option (a) for now; if mature enough to contribute later, do so. But our timeline shouldn't depend on smithay accepting a PR.
3. **screencopy / screenshot.** Requires reading back from our intermediate or scanout buffer. Architecturally fine but adds Vulkan readback paths. Defer to post-MVP.
4. **wayland-protocols version skew.** Smithay tracks specific versions of staging protocols. If `wp_color_management_v1` lands in a different `wayland-protocols` version than smithay's pinned one, we may need version juggling. Manageable; flag if it bites.
5. **Niri layout port granularity.** Whole `layout/` directory, or selectively (just `Layout` + `Workspace` + `Column` + `Tile`)? Probably whole-directory to start; trim afterward.

## Updating this doc

This is a strategy doc — update it when:
- A "Depend" entry turns out to not actually fit and we have to copy+modify instead.
- A "Replace" entry turns out to have a reusable sub-component we should depend on.
- A new niri module becomes relevant to port.
- The LOC estimates turn out wildly off — recalibrate so the planning intuition stays useful.
