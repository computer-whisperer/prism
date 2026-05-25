# Architecture

How prism is put together: the crate layering, the runtime object graph, and
the renderer / protocol / DRM decisions made during implementation. This is the
"how it's built and why" reference — read it before changing the renderer, the
output bringup path, or the protocol wiring.

Companion docs:
- [color-management.md](color-management.md) — the color/HDR design and the math the renderer implements.
- [reuse-map.md](reuse-map.md) — what we depend on from smithay/niri vs. what we own.
- [status.md](status.md) — what's built and verified right now.
- [deferred-work.md](deferred-work.md) — the backlog.

## Crate layering

| Crate | Role |
|---|---|
| `prism-frame` | Renderer-independent frame description (`Element`, `FrameDescription`, `ColorDescription`, color matrices, opaque handles) |
| `prism-renderer` | Vulkan renderer (`ash`). Per-GPU instance. Decode → composite → encode pipeline. |
| `prism-drm` | KMS frontend — `DrmCompositor`-equivalent + per-output scanout buffer pool + modifier negotiation |
| `prism-layout` | Window layout + per-surface state (ported from niri's `layout/` + `window/` + `layer/` + `cursor.rs`) |
| `prism-input` | Input routing: libinput → seat → focused surface, pointer hit-testing |
| `prism-animation` | Spring/curve animation engine + the frame `Clock` (ported from niri) |
| `prism-config` | KDL config schema (ported from niri-config) |
| `prism-protocols` | Wayland protocol wiring (smithay handlers), the `PrismState` machine, buffer import |
| `prism-ipc` | IPC types/helpers for talking to the running compositor (used by `prism-tune`) |
| `prism-tune` | Closed-loop color calibration tool (IPC client + colorimeter + patch surface) |
| `prism-wlcs` | WLCS conformance-test integration shim (cdylib) |
| `prism-shmtest` | Tiny xdg-shell shm client for end-to-end TTY testing |
| `prism` | The compositor binary |

**The load-bearing boundary is `prism-frame`.** Layout, input, and protocol code
describe frames in terms of `Element` / `FrameDescription` / `ColorDescription`
and opaque handles — they never see a Vulkan type. This keeps renderer concerns
from leaking outward, and (as a side benefit) lets the protocol/state machine run
with zero GPUs, which is what the WLCS headless harness relies on.

The renderer is **one concrete type** (`ash`-backed), one instance per physical
GPU. There is no `Renderer` trait polymorphism axis to maintain — a deliberate
departure from smithay, whose `R: Renderer` generic is the thing we're replacing
(see [reuse-map.md](reuse-map.md)).

## Runtime object graph

`prism run` opens every connected card, brings up every connected connector
across all GPUs, and pumps independent vblank-driven scanout on each.

```
SeatSession           1 per process. libseat-backed; holds the DRM-master grant per opened card.
 └─ DrmCardContext    1 per /dev/dri/cardN we drive. Owns DrmDevice + GBM + DrmDeviceNotifier.
                       Picks connector + CRTC + mode for each output.
  ─ GpuContext        1 per physical Vulkan device. Matched to a card by drm_render dev-id.
                       Arc<prism_renderer::Device>.
   ─ OutputContext    1 per active connector. Owns DrmSurface, double-buffered scanout BOs,
                       a per-output Renderer (pipelines bake in the per-output EncodeConfig).
                       References its card + its GPU.
```

`PrismState` carries the maps:

```rust
session: SeatSession,
cards:   HashMap<DrmDevId, DrmCardContext>,
gpus:    HashMap<DrmDevId, Arc<prism_renderer::Device>>,
outputs: HashMap<OutputId, OutputContext>,
dmabuf_textures: HashMap<ObjectId, HashMap<DrmDevId, Arc<ImportedImage>>>,
                  // ^ cross-GPU import: every registered GPU gets its own VkImage,
                  //   so any output's render path can sample without GPU↔GPU copies.
```

- **`OutputConfig`** (in `prism-drm`) is the static-per-output bundle: scanout
  depth + Vulkan format + intermediate format + `EncodeConfig`, plus the color
  fields — HDR signaling, CTM, per-channel response curve, panel-peak-nits, SDR
  reference nits, and the loaded 3D LUT. These are resolved from KDL config
  (EDID-keyed) at bringup.
- **`OutputState`** (in `prism-frame`) is the per-frame output state.

### Multi-GPU buffer replication

Client buffers are imported on **every registered GPU** at commit time so any
output's render path can sample locally with no GPU↔GPU copy:

- **dmabuf**: `dmabuf_imported` imports the same buffer once per GPU.
- **shm**: `SurfaceTexture::Shm { by_gpu }` mirrors that — bytes are read once and
  uploaded into one staged `VkImage` per GPU on each commit.
- **Cross-GPU mirror**: when a surface allocated on GPU A must show on an output
  driven by GPU B, `GpuTex::Mirror` transports the buffer. For YUV it carries an
  optional `MirrorChroma` so both planes cross the GPU boundary (see
  [color-management.md](color-management.md)).

Per-GPU narrowing (upload only to the GPUs actually hosting a surface, derived
from its output assignment) is a deferred optimization — see
[deferred-work.md](deferred-work.md).

### Output assignment and enter/leave

`layout_outputs()` assigns logical positions by stacking outputs horizontally at
`y=0` in sorted-connector-name order (non-overlapping geometry). Each surface
carries a `SurfacePlacementSlot { logical_pos, current_output }`. After a
new-buffer commit, `process_surface_buffer` computes the surface center and
`output_containing(point)` finds the wl_output whose geometry contains it; on
transition we dispatch `wl_surface.enter` / `.leave` on the right smithay
`Output`s. `present_for_crtc` filters toplevels to those whose `current_output`
matches the output being presented.

## Renderer pipeline

Two color transforms, compositing in linear light in between:

```
per-element decode → fp16 BT.2020 absolute-nits linear intermediate → per-output encode → scanout
```

The intermediate is `R16G16B16A16_SFLOAT`, BT.2020 primaries, absolute nits
(1.0 = 1 nit). This is non-negotiable for correct alpha blending and per-client
color handling; the rationale is in [color-management.md](color-management.md).
Direct scanout (when a client's description matches the output's) is an
optimization *on top of* this path, not a separate path.

Decisions baked into the renderer:

- **Per-element draws**, not a full-screen composite. A 4-vertex triangle-strip
  quad per element, dynamic-rendering color attachment, premultiplied-alpha blend.
  Each element runs the decode shader with its own source color params.
- **The encode pass is a single full-screen triangle.** Reads the intermediate as
  `SHADER_READ_ONLY_OPTIMAL`, writes the scanout image as
  `COLOR_ATTACHMENT_OPTIMAL`, ends in `GENERAL` for KMS handoff. (`PRESENT_SRC_KHR`
  is not valid without `VK_KHR_swapchain`; for KMS scanout the correct final
  layout is `GENERAL`.)
- **The per-output encode shader is synthesized, not statically compiled.** SPIR-V
  is emitted at `Renderer::new` time from an `EncodeConfig` (an ordered list of
  `EncodeFragment`s) via `rspirv::dr::Builder` (see
  `crates/prism-renderer/src/encode_synth/`). Each fragment threads a `vec3` color
  through one `fragment::emit_*` block. A single fixed `EncodePushSynth` struct
  (128 bytes — Vulkan's minimum push-constant limit) is shared across all
  configurations; fragments use whichever slots they need. Rationale: per-display
  effects (3D LUT calibration, response correction, future subpixel FIR + dither)
  chain into **one** fragment shader and one dispatch — no ping-pong post-process
  passes, and the optimizer can fuse the math. The vertex shader (full-screen tri)
  has no per-output variation, so it stays statically compiled from GLSL.
- **`EncodeFragment` variants** (chain order): `Lut3d`, `CalibrationMatrix`,
  `PerChannelResponseGainGamma`, `OutputTransferSrgb` / `OutputTransferPq` /
  `OutputTransferLinear`, plus not-yet-implemented `SubpixelFir3Horizontal` and
  `InterleavedGradientNoiseDither` (which need multi-sample synthesis — emitting
  them currently returns `MissingFeature`). The SDR/PQ/linear defaults are
  `[Lut3d, OutputTransfer*]`: the 3D LUT is the calibration path, identity unless
  a calibration is loaded. See [color-management.md](color-management.md).
- **Decode shader** handles RGB and two-plane YUV (NV12 8-bit, P010 10-bit) via
  manual plane sampling (binding 0 = luma, binding 1 = chroma) — not
  `VkSamplerYcbcrConversion`, which doesn't fit a push-constant-driven decode. It
  applies YUV→RGB (BT.709/BT.2020 range expansion), the source transfer function,
  and the surface-primaries→BT.2020 matrix (`DecodePush::decode_matrix`).
- **Push constants for per-draw state** (mat4 + scalars). No specialization
  constants (would multiply pipeline count for marginal perf).
- **Vulkan 1.3 dynamic-rendering** (`cmd_begin_rendering` + `RenderingAttachmentInfo`,
  no `VkRenderPass`/`VkFramebuffer`) and **synchronization2** for all barriers.
- **Persistent command buffers + fences + push descriptors.** Per-frame CPU cost
  in the steady-state present path is ~60 µs/output (down from ~6 ms when the
  tracer used `queue_wait_idle` per frame). Slots are fence-gated and reused.
- **Shader build**: `build.rs` shells out to `glslangValidator` for the GLSL
  shaders; SPIR-V bytes land in `$OUT_DIR` and are `include_bytes!`'d. The build
  fails fast if glslang isn't installed.

## Wayland / protocol

- The state machine is **decoupled from the renderer**: buffer import is lazy and
  keyed off output placement, commits succeed with zero GPUs, and input has no GPU
  touchpoints. (This is what lets `prism wayland` and the WLCS harness run
  headless.)
- Per-client data: `PrismClient { compositor: CompositorClientState }`.
- **`PrismState` carries `Arc<prism_renderer::Device>`** so dmabuf imports validate
  against real Vulkan format/modifier support, not a hardcoded list.
- **xdg-shell initial configure** is sent on first commit (checking
  `XdgToplevelSurfaceData::initial_configure_sent` inside `CompositorHandler::commit`),
  not on `new_toplevel` — gives the client a chance to set `title`/`app_id` first.
- **dmabuf import** validates by plane formats: RGB via `vk_format_for`, YUV via
  `yuv_kind_for` + the per-plane formats. The import-time guard **must** accept
  everything `build_advertised_dmabuf_formats` advertises — a `create_immed`
  rejection is a fatal `invalid_wl_buffer` protocol error that crashes the client
  (this is what crashed Firefox before NV12 acceptance landed). Advertised formats
  are modifier-negotiated against the GPU: 8-bit RGBA/BGRA, 10-bit, fp16, and
  NV12 + P010 YUV.
- **wp_viewport source crop** is honored: `src_rect_uv` is computed from
  smithay's `SurfaceView.src` (was previously hardcoded to the full texture, which
  garbled Firefox's tiled WebRender compositor).
- **Subsurface render order**: smithay's `with_surface_tree_downward` emits
  top-to-bottom (front-to-back), but prism's renderer draws the element vec
  front-first with src-over (earlier = behind), so the walk's emissions are
  collected and appended reversed.
- **`Mapped::window_geometry()`** anchors a window with no explicit xdg geometry
  at the main-surface origin `(0,0)` sized to the root surface's `dst`, rather than
  at smithay's subsurface-inclusive bbox. (A deliberate divergence from niri,
  user-approved; without it a subsurface at negative coords drags the whole
  toplevel.)
- **Input dispatch**: libinput runs over udev (`Libinput::new_with_udev` +
  `udev_assign_seat`); `prism_input::dispatch::process_input_event` (ported from
  niri's `State::process_input_event`) routes events, and `on_device_added` flips
  the seat's keyboard/pointer capabilities per the device's libinput capabilities.
  (No `wl_touch` yet — see [deferred-work.md](deferred-work.md).)
- **Pointer focus**: `PrismState::contents_under` (in
  `prism-protocols/src/pointer_focus.rs`, ported from niri) does z-ordered
  recursive hit-testing (layers + windows, descending popups/subsurfaces) and
  returns the deepest surface + true global origin. `refresh_pointer_focus`
  re-evaluates after surface/layout changes and re-sends `pointer.motion` so
  smithay emits enter/leave.

## DRM / GBM

- **GEM handles are per-fd.** `GbmDevice` and `DrmDevice` must share the same
  `DeviceFd` (`Arc<OwnedFd>`) or `addfb2` returns ENOENT. Use
  `GbmDevice::from_device_fd(drm.device_fd().device_fd())`.
- **Double-buffering is required, not optional.** Single-buffered rendering at
  60 Hz wedges the kernel on amdgpu+RADV: the 3D engine writing the actively
  scanned-out BO while the display engine reads it hits implicit sync that locks
  the system. Two BOs; `back_index` tracks which to render into; `mark_vblank()`
  toggles it after each vblank event.
- **Both notifiers MUST be drained.** `OutputContext::new` returns
  `(ctx, OutputNotifiers { drm, session })` and the caller must insert both into
  the calloop loop. Without the **DRM** notifier, page-flip event allocations
  cascade to ENOMEM at 60 Hz. Without the **libseat** notifier, logind can't
  request a VT switch (we never ack the pause) — Ctrl+Alt+Fn hangs, SIGINT is
  blocked, and a tight loop holding DRM master + unack-able libseat means a full
  reboot. **Rule: any time smithay hands back a `*Notifier`, assume it must be
  drained — never bind it to `_`.**
- **Shutdown order: drop `PrismState` BEFORE the event loop.** The `LibSeatSession`
  (DRM-master holder) lives in the notifier inside the loop; `OutputContext::Drop`
  → `DrmSurface::clear()` needs DRM master, so the state (which owns the surfaces)
  must drop first. Documented at the `run_integrated` call site.
- **Scanout modifier negotiation**: `pick_scanout_modifiers` negotiates tiled
  modifiers (fp16 tiled works on both DCN1/Vega 20 and DCN3/Navi 21) with a LINEAR
  fallback. Multi-plane (DCC) modifiers are filtered out — the importer is
  single-plane. `ScanoutDepth::{Bpc8, Bpc10, Fp16}` selects GBM format + Vulkan
  format together; `max bpc` is set on the connector when exposed.
- **HDR signaling** is config-driven (`prism-drm/src/hdr.rs`): an output whose
  config carries a `color.hdr` block gets `HDR_OUTPUT_METADATA` (the kernel
  `struct hdr_output_metadata` blob, built by `build_hdr_metadata_blob`) +
  `Colorspace = BT2020_RGB` + `max_bpc = 10` set at bringup, the scanout BO
  allocated fp16, and the encode chain switched to PQ. `HdrProps::set_hdr` is
  re-pushed after a VT handoff; `HdrProps::clear` resets metadata→0 /
  Colorspace→Default on shutdown so the next session doesn't inherit stale PQ
  signaling. `EdidInfo::read` parses EDID at bringup to key per-output config and
  seed defaults.
- **Hardware target**: two AMD cards driven concurrently. `card0` = Vega 20
  (Radeon VII, primary 226:0 / render 226:129) drives the Samsung array;
  `card1` = Navi 21 (RX 6900 XT, primary 226:1 / render 226:128) drives the
  central panel. Vulkan device is matched to a card by `drm_primary`/`drm_render`
  dev-id.

## Testing

- `prism-shmtest` + `scripts/tty-test.sh` exercise the full TTY path (compositor +
  client drawing cycling colors).
- `scripts/firefox-test.sh` runs prism + Firefox in a TTY with protocol/Mozilla
  debug logging for the HDR-video path (`NO_HDR=1` toggles Firefox Wayland HDR).
- `prism-wlcs` is a protocol-conformance harness (WLCS); `crates/prism-wlcs/conformance/`
  holds the curated test allowlist + `run.sh`. WLCS tests protocol behavior, not
  pixels — it catches wrong pointer/surface coordinates and subsurface bookkeeping,
  but not renderer-side pixel offsets. See [status.md](status.md) for current pass
  counts.
