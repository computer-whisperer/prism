# Screen capture: screenshots and screen recording

> **Status.** `wlr-screencopy` complete (`grim` HW-verified). Done: the sRGB
> capture primitive (renderer); **both** the dmabuf (zero-copy, whole-output) and
> SHM (whole-output + region) paths, **both asynchronous and serviced from the
> render loop** right after the output's `present()`. That (a) fixes the dmabuf
> GPU-ordering caveat by sequencing the capture submit between frames on the
> shared queue, (b) throttles `copy_with_damage` to actual damage, and (c) removes
> the explicit main-thread `queue_wait_idle` the SHM path used to do per frame.
> **Recording is still laggy, though** — that drain was not the dominant cost. The
> remaining per-frame main-thread cost is diagnosed but **not yet fixed**; see
> [Recording perf](#recording-perf-still-laggy--diagnosed-not-fixed) below. The
> dmabuf path and the async-SHM rework are otherwise **runtime-unverified** since
> the rework. Next: **in-process** PipeWire + `org.gnome.Mutter.ScreenCast` D-Bus
> (reusing `capture_into_dmabuf`), then `ext-image-copy-capture-v1`. Target
> end-state (agreed): both capture protocols, niri-style in-process recording,
> color-correct HDR capture via the sRGB encode pass. niri is the reference for
> the protocol/PipeWire/D-Bus glue; the renderer coupling is prism-specific
> (custom Vulkan pipeline, not smithay's `GlesRenderer`).

How prism lets external clients capture output contents — `grim`-style
screenshots and OBS/portal-style screen recording — and how that couples to
prism's two-pass HDR render pipeline. [renderer.rs](../crates/prism-renderer/src/renderer.rs)
is the *what* of the pipeline (decode-per-element → fp32 BT.2020 intermediate →
encode-per-output → scanout). This doc is the capture layer beside it: how a
capture request turns the live intermediate into pixels a client can consume,
and how color is handled when the output is HDR.

## The two layers: capture protocol vs. recording transport

"Standard screenshot/record protocol" is two things, and conflating them
mis-scopes the work:

- **Screenshots** — tools like `grim`, `grimblast`, `wf-recorder`, `hyprshot`
  talk a capture protocol *directly*: `wlr-screencopy` (de-facto standard,
  wlroots-origin) or the freedesktop successor `ext-image-copy-capture-v1` +
  `ext-image-capture-source-v1`.
- **Recording / sharing** — OBS, browser screenshare, Zoom almost never talk
  those protocols directly. They go through **xdg-desktop-portal → PipeWire**.
  The portal's *backend* is what speaks a capture protocol or a private
  compositor API. `xdg-desktop-portal-wlr` consumes `wlr-screencopy`; niri
  instead runs PipeWire in-process and exposes `org.gnome.Mutter.ScreenCast`
  over D-Bus, paired with `xdg-desktop-portal-gnome`.

We are building both layers (the agreed target is the niri-style in-process
PipeWire path, not portal-wlr offloading). But they share one renderer
primitive — see below — so recording is mostly protocol/transport plumbing on
top of the same capture core.

## Does Wayland already solve capture color management? Not yet — direction is set

This was checked explicitly before committing to an sRGB encode pass, because
"the protocol handles it" would change the design. Findings (May 2026):

- **`wlr-screencopy`** carries **no** color information — only a pixel *format*
  (`Xrgb8888`). It is a frozen wlroots protocol; it will never carry color. A
  captured HDR frame copied verbatim is PQ/BT.2020-encoded and looks wrong in
  any sRGB viewer.
- **`ext-image-copy-capture-v1`** (current staging release, v1) **also has no
  color management.** Frame metadata is only `transform`, `damage`,
  `presentation_time`. Verified against the live protocol definition.
- **The ecosystem's chosen answer is "tag, don't convert."** The groundwork
  merged: `wp_image_description_reference_v1` landed in wayland-protocols
  explicitly *"as a prerequisite for adding color management to
  `ext-image-copy-capture-v1`."* The plan is that the compositor hands over
  pixels in the output's **native** color space tagged with a color-management
  *image description*, and the **client** converts. A versioned capture event
  (the `ready2` / image-description-reference work, 64-bit image-description IDs)
  is in progress but **not in any released protocol version**.
- niri hasn't solved this either: its screencopy renders to the capture buffer
  with no color conversion, so on an HDR output niri also emits raw HDR pixels.

So there is **no protocol-level color answer available today**, but the future
shape is known. This yields two paths, which prism's capture primitive must
both accommodate:

| Path | Pixels handed out | Works with | Status |
|---|---|---|---|
| **A. Compositor converts** (sRGB encode) | tone-mapped sRGB, `Xrgb8888` | *everything* — wlr-screencopy, today's grim/OBS, CM-unaware clients | only thing that works now |
| **B. Compositor tags, client converts** | native (HDR/PQ) + image description | `ext-image-copy-capture` + CM extension | future, not shipping |

**Decision:** Path A (sRGB encode) is the mandatory baseline — it is the only
thing that works with `wlr-screencopy` and every client shipping today. But the
capture color step is a *parameter* of the primitive, not hardcoded, so Path B
(native + attach image description) can be added when the protocol exists.
prism is unusually well-positioned for Path B because it already speaks
`wp_color_management_v1` ([color-negotiation.md](color-negotiation.md)) — it
already has the image-description machinery to tag a buffer; niri does not.

## The load-bearing abstraction: one capture primitive

Every frontend — screenshot protocol, PipeWire screencast, debug dump — needs
the same thing: *render an output's current composite into a caller-supplied
destination buffer, with a chosen capture color profile.* Build that once
against prism's Vulkan renderer; every frontend sits on top.

The source of truth is the per-output **fp32 BT.2020 absolute-nits
intermediate** ([intermediate.rs](../crates/prism-renderer/src/intermediate.rs)),
which already persists between frames and is already `TRANSFER_SRC`-capable (the
window-close snapshot path uses it). Capturing from the intermediate — *before*
the panel-specific encode — is what makes a correct sRGB capture possible: the
intermediate is panel-independent real-light, so the capture encode is a clean
colorimetric transform rather than an attempt to invert a specific panel's
calibrated PQ output.

### Capture encode ≠ panel encode

The live per-output encode chain is **panel realization**:
`[Lut3d, OutputTransfer{Pq,Srgb}]` — it applies the *measured panel
correction* (3D LUT) and the *panel's* transfer function. That output is wrong
for a screenshot in two ways: it bakes in this panel's measured quirks, and on
HDR outputs it is PQ-encoded.

The **capture** encode is a different chain entirely — a colorimetric sRGB
target with no panel correction:

```
[CalibrationMatrix(BT.2020 → BT.709), OutputTransferSrgb]   (no Lut3d)
```

- `CalibrationMatrix` is set to the **BT.2020 → BT.709 primaries matrix**
  (`prism_frame::color::bt2020_to_srgb_matrix`), converting the intermediate's
  BT.2020 primaries to sRGB/BT.709 primaries (still in absolute nits).
- `OutputTransferSrgb` does `srgb_oetf(clamp(in / sdr_white_nits, 0, 1))`, with
  `sdr_white_nits` = the output's reference-white level
  (`effective_sdr_reference_nits()`, ~80 SDR / ~203 HDR). Diffuse white →
  1.0; the sRGB OETF encodes for an sRGB viewer.
- **No `Lut3d`** — the panel LUT corrects for *this physical panel*; a
  screenshot must not carry that. The capture pipeline declares no binding 1.

This reuses the existing encode synthesizer
([encode_synth](../crates/prism-renderer/src/encode_synth/mod.rs)) and push-
constant layout unchanged; the capture profile is just a different `EncodeConfig`
+ `EncodePush`. No new shader machinery.

### First-cut tone-mapping = highlight clip (acknowledged)

`clamp(in / sdr_white_nits, 0, 1)` hard-clips anything above reference white and
clips out-of-BT.709-gamut colors (negative after the matrix) at 0. That is a
crude SDR rendition of HDR content — highlights blow out rather than roll off.
It is correct-*looking* (neutral, viewer-safe) and is the right phase-1 cut; a
later refinement is a proper tone curve + hue-preserving gamut roll-off (the same
problem the panel LUT solves downstream, applied here for the sRGB target).

### Recording perf (still laggy) — diagnosed, not fixed

**TODO (deferred).** Continuous recording (`wf-recorder`) still makes the system
laggy even after both paths were made async. The async rework removed the
explicit `queue_wait_idle`, but that was not the dominant cost. The residual
per-frame work lands on the compositor's **main (calloop) thread**, which is also
the input thread — hence system-wide lag. Diagnosed by code-reasoning (not
measured; per the project's perf approach), three costs:

1. **Per-frame `vkAllocateMemory`.** `HostReadback::new` allocates a fresh ~33 MB
   (4K) host-coherent buffer every SHM capture, on the main thread
   (`submit_one_capture`). Large per-frame host allocations stall on many drivers.
   *Fix:* pool the readback buffers (reuse across captures; the `AsyncSlot` fence
   already serializes them).
2. **Single shared `AsyncSlot` → main-thread fence wait.** `slot.begin()` waits on
   the *previous* capture's completion fence. With one slot, capture N+1's
   recording blocks the main thread on capture N's GPU work. At 4K we now run
   **two** full-res encode passes per frame (panel realization for scanout **+**
   the sRGB capture pass) plus the readback; when that exceeds the frame budget
   this wait stalls input, self-reinforcingly. *Fix:* round-robin N async slots so
   `begin()` rarely waits.
3. **Per-frame ~33 MB CPU `memcpy`** (`copy_readback_to_shm`, in
   `complete_screencopy`, also main thread). SHM-only. *Fix:* import the client
   `wl_shm` pool via `VK_EXT_external_memory_host` and have the GPU write it
   directly (zero-copy), or offload the copy to a worker thread.

The **dmabuf** path avoids #1 and #3 (zero-copy); only #2 applies to it, so the
cheapest win — and the confirming experiment — is multiple async slots + a
whole-output dmabuf recording test. Which path `wf-recorder` actually took
(SHM vs dmabuf) was not determinable from its log; an instrumented run (capture
path + per-frame main-thread timing) would pin the dominant cost before investing
in the SHM-specific fixes (#1/#3).

### Destination-agnostic by design

The primitive records *render intermediate → dst image* into a command buffer;
the destination varies per frontend:

- **Debug dump / SHM clients** — render into an owned offscreen
  `R8G8B8A8_UNORM` image, then copy to a host-visible buffer and `memcpy` into
  the client's SHM pool (or a `.ppm`/`.png` for the debug path). This is exactly
  the [`EncodeDiagnoseProbe`](../crates/prism-renderer/src/diagnose.rs) shape at
  full output size.
- **dmabuf clients** — import the client's dmabuf as a Vulkan image (prism
  already imports client dmabufs) and render the capture encode *directly into
  it*; no readback.
- **PipeWire screencast** — render into a GBM-allocated dmabuf from the
  PipeWire buffer pool; queue after the GPU sync point.

## Where capture hooks into the frame

The intermediate holds the last composited frame (persistent, left in
`SHADER_READ_ONLY_OPTIMAL` after each encode), so a capture can run any time
after a frame and reproduce that frame. Two timing modes mirror the protocols:

- **Immediate** (screenshot / `copy` without damage) — capture the current
  intermediate on demand, out of band, via a one-shot submit (the phase-1 and
  `grim` path). Simple; one extra encode pass + readback.
- **With-damage** (efficient recording / `copy_with_damage`) — queue the
  capture onto the next output redraw in
  [`render_output_now`](../crates/prism/src/main.rs) /
  [`present`](../crates/prism-drm/src/output_ctx.rs), riding the frame's command
  buffer and reusing the output `DamageTracker`. Needed so a 60fps screencast
  doesn't force full re-encodes.

Phase 1 uses the immediate path only.

## Frontends (build order)

1. **Renderer capture primitive + sRGB profile** *(phase 1, in progress)* —
   `Renderer::capture_srgb()` producing RGBA8 from the live intermediate, plus a
   debug dump to verify HDR→sRGB end-to-end before any protocol churn. De-risks
   the only prism-specific part.
2. **`wlr-screencopy`** *(done; dmabuf path runtime-unverified)* — hand-rolled
   `zwlr_screencopy_v1` (`crates/prism-protocols/src/screencopy.rs`), in the
   `output_power.rs` style (impls on `PrismState`, no delegate macro). Two buffer
   paths on the phase-1 primitive: **SHM** (synchronous readback; whole-output +
   region, cropped on the CPU copy) and **dmabuf** (async, zero-copy:
   `Renderer::capture_into_dmabuf` renders into the imported client buffer and
   returns a sync_fd; `ready` fires from a calloop source; whole-output only).
   `copy_with_damage` is serviced like `copy` (full-frame). The async dmabuf path
   has an *ordering caveat* — it samples the intermediate from a separate submit,
   relying on same-queue (radv) serialization; the robust fix is render-loop
   integration (next phase). See the `capture_into_dmabuf` doc comment.
3. **In-process PipeWire + `org.gnome.Mutter.ScreenCast`** — port niri's
   `screencasting/` (`pw_utils.rs`) and `dbus/mutter_screen_cast.rs`. Net-new
   deps: `pipewire`, `zbus`. The PipeWire mainloop folds into prism's existing
   `calloop` loop (niri's pattern ports directly — prism is calloop 0.14).
   Unlocks OBS / browser screenshare / Zoom via `xdg-desktop-portal-gnome`.
4. **`ext-image-copy-capture-v1`** — second frontend on the same primitive, and
   the point where Path B (native pixels + `wp_image_description_reference_v1`)
   becomes worth wiring.

## Architectural decisions (this round)

- **Capture from the intermediate, not the scanout.** The intermediate is
  panel-independent real-light; the scanout is panel-encoded (PQ on HDR). A
  correct sRGB capture is a forward colorimetric encode from the intermediate,
  not an inverse of a calibrated panel.
- **Capture has its own encode chain.** `[CalibrationMatrix(2020→709),
  OutputTransferSrgb]`, no panel LUT. Reuses the existing synthesizer; adds no
  shader infrastructure.
- **The color step is a parameter (Path A now, Path B later).** sRGB conversion
  is the default; the primitive is structured so a future native-pixels +
  image-description path slots in without rework.
- **One destination-agnostic primitive.** Render-into-dst is shared; readback
  (SHM/PNG) vs. render-into-dmabuf (zero-copy) vs. PipeWire-pool is the only
  per-frontend difference.
- **niri is the port reference for protocol/PipeWire/D-Bus only.** Its
  buffer-fill (`render_to_dmabuf`/`render_to_shm` over smithay's `GlesRenderer`/
  `ExportMem`) does **not** port — prism's Vulkan renderer replaces it.

## Open questions

1. **Capture extent / scaling.** Capture at the output's physical pixel size
   (current intermediate extent) and let the client scale, or offer scaled
   capture? Phase 1 = physical size.
2. **Cursor.** Composite the cursor into captures (embedded), omit it, or expose
   both (`ext-image-capture-source` / Mutter cursor modes both model this)?
   The cursor is currently a hardware plane, not in the intermediate — embedding
   it means software-compositing it into the capture encode.
3. **Tone-mapping curve.** When to replace the highlight-clip first cut with a
   real HDR→SDR tone curve + gamut roll-off for Path A captures.
4. **Multi-GPU.** Capture on the output's own GPU (the intermediate lives
   there); confirm no cross-GPU copy is needed for the common case.
5. **Damage timing.** Exact hook for `copy_with_damage` into `present` so a
   queued capture rides the frame command buffer without an extra submit.
