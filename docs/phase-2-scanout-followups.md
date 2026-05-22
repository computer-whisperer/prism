# Scanout pipeline — deferred follow-ups

Tracked here so they don't get lost. Each item is something we deliberately
*did not* do as part of the modifier-negotiation work, with the reasoning
for deferral and what triggering it would cost. Order is rough priority
descending.

## 1. Multi-plane (DCC-compressed) modifier import

**Status:** filtered out today. `pick_scanout_modifiers` drops any modifier
with `plane_count > 1` because `ImportedImage::import` only handles a single
plane / single dmabuf fd / single Vulkan memory allocation.

**Why we'd want it:** AMD DCC (Delta Color Compression) modifiers carry an
auxiliary metadata plane describing per-tile compression. On desktop
content (solid backgrounds, repeated UI chrome) DCC can drop effective
scanout fetch by another ~2× on top of plain tiling. Expected to matter most
on the 4K HDR OLED at 240Hz where every bandwidth saving compounds.

**What it costs:** rework `ImportedImage::import` to:

- Iterate `dmabuf.planes`, allocating one Vulkan `DeviceMemory` per plane
  via `ImportMemoryFdInfoKHR` (each plane fd is a separate import).
- Bind each memory to the image with the right `VkBindImageMemoryInfo` +
  `VkBindImagePlaneMemoryInfo` (plane aspect mask).
- Pass per-plane offset/stride/pitch via `VkSubresourceLayout` array in
  `ImageDrmFormatModifierExplicitCreateInfoEXT::plane_layouts` (already
  array-shaped — we just only ever populate one entry).
- The renderer code paths that bind the image as a color attachment don't
  need to change — Vulkan handles the per-plane fetch internally.

**Trigger:** when we see scanout bandwidth pressure that the plain-tiled
modifier doesn't solve, OR when the 240Hz OLED arrives. Should also drop
`pick_scanout_modifiers`'s `plane_count == 1` filter at that time.

## 2. Per-plane damage clips (`FB_DAMAGE_CLIPS`)

**Status:** smithay's atomic backend already plumbs `FB_DAMAGE_CLIPS` from
`PlaneState.damage_clips`; we always pass `None`, so the kernel/display
engine treats every frame as full-screen damage.

**Why we'd want it:** with a tiled scanout buffer, telling the display
engine "only the rectangle (200, 100, 800, 600) changed" lets it skip the
fetch for unchanged tiles. Roughly: a typical text-cursor blink redraw
goes from 4K full-fetch (~33 MB at 10bpc) to a few tile bursts (~64 KB).
Compounding bandwidth win on top of (1).

**What it costs:**

- Track damaged rectangles per output. We have output-level "needs
  redraw" today; this is a finer-grained version. Producer is wayland
  surface damage rolled up through the layout's window-to-output
  mapping.
- Convert tracked damage to KMS-coordinate rectangles, build a property
  blob (`drm_mode_rect[]`), attach to `PlaneState.damage_clips`.
- Reset to `None` (full-screen damage) on any mode-changing event:
  scale change, mode swap, HDR enable/disable, plane reconfiguration.

**Trigger:** when we start optimizing for power on the laptop scenario
(if/when prism runs on a laptop), or when measured idle-desktop scanout
bandwidth is meaningfully high.

## 3. Atomic test commits before mode-changing operations

**Status:** today we call `surface.commit()` / `surface.page_flip()`
directly. If the kernel rejects (HDR property set during incompatible
state, modifier mismatch after hotplug, etc.) we discover it at commit
time when the previous state is already partially gone.

**Why we'd want it:** smithay exposes `DrmSurface::test_state` which runs
the same atomic ioctl with `TEST_ONLY`. It tells us "yes, the kernel
would accept this" without actually applying it. Cheap. Lets us:

- Sanity-check HDR enable before flipping connector properties + scanout
  format together.
- Probe mode-set after hotplug before tearing the active mode down.
- Probe VRR enable/disable on connectors that report `RequiresModeset`.

**What it costs:** thin wrapper in `output_ctx.rs` that builds the same
`PlaneState` we'd commit, calls `test_state` first, returns
typed errors so the caller can recover instead of leaving the surface in
a broken state. ~80-100 LOC plus selective call-site adoption.

**Trigger:** next time we hit a "the commit looked right but the kernel
rejected it after we changed the active mode" failure mode. Likely
during HDR enable/disable on a running session.

## 4. Per-output explicit `LINEAR` policy hint

**Status:** `pick_scanout_modifiers` always appends `LINEAR` as the
fallback. That's the safe default. But on certain configurations (older
GPUs with broken tiled fp16, headless / virtual outputs, looking-glass
capture pipelines) we may want to *force* `LINEAR` for a specific
output.

**Why we'd want it:** Vega 20 + fp16 + tiled is the case we built this for
and it should be fine. But e.g. Navi 21 has had specific kernel
regressions on tiled fp16 in the past; if we hit one, we want a per-
output config knob to opt that output back into `LINEAR` without
disabling the negotiation globally.

**What it costs:** add `force_linear_scanout: bool` to
`prism_config::OutputConfig.color`, plumb through `OutputConfig`,
short-circuit `pick_scanout_modifiers` when set. ~20 LOC.

**Trigger:** if we see a kernel/driver regression on a specific GPU
that our modifier negotiation picks wrong. Cheap insurance to add
proactively, but not blocking anything today.

## 5. Renderer-side intermediate-buffer modifier negotiation

**Status:** the renderer's intermediate buffer (the fp16 linear-light
working surface between decode and encode passes) is still allocated
with implicit tiling — Vulkan picks whatever it likes since it's never
exposed as a dmabuf. Fine for now.

**Why we'd want to change it:** if we ever import client surfaces
*directly* as overlay-plane buffers (skipping the intermediate composite
for fullscreen surfaces), the renderer needs to negotiate the import
modifier on those buffers too. Not interesting until per-plane color
management is real on the hardware we target — see top-level
phase-2-progress notes on why HDR currently rules out plane assignment.

**Trigger:** if/when Linux kernel exposes per-plane CRTC color
properties usefully on Vega 20 + Navi 21 (currently DCN2+ only via
`AMD_FMT_MOD_CRTC_*` properties). Not on our roadmap.
