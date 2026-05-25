# Deferred work

The backlog: things we deliberately *did not* do, grouped by subsystem, each with
why it's deferred and roughly what triggering it would cost. This is the place to
look before starting something — it may already be scoped here.

Ordering within each section is rough priority. When an item lands, delete it (and
update [status.md](status.md)).

## Color / HDR

KMS HDR signaling, EDID parsing, per-output calibration (3D LUT, CTM, response
curve), and per-output color config are **done** — see [status.md](status.md). What
remains:

### Tone mapping

Both decode (over-bright input) and encode (above-display-peak) currently
hard-clip. Real tone mapping (BT.2390 EETF reference; Reinhard/Hable simpler) goes
in the encode chain between calibration and the OETF, as per-output config. Wait
until we have mixed SDR+HDR content to tune against. The cases (HDR→SDR, SDR→HDR,
HDR→HDR peak mismatch, cross-gamut) are inventoried in
[color-management.md](color-management.md).

### Automatic HDR enablement from EDID

EDID is parsed and used to match per-output config blocks, but HDR is turned on by
an explicit `color.hdr` config block — we don't auto-enable HDR from the EDID's
advertised HDR static metadata. Low priority; explicit config is arguably the safer
default anyway. Trigger: wanting plug-and-play HDR on a fresh panel without config.

### Remaining decode transfer functions

The decode shader implements sRGB, linear, and PQ. HLG, BT.1886, and
custom-parametric gamma are TODO stubs — add as content needs them.

### Encode-side primary conversion / per-output working space

The decode side converts each surface's primaries into the BT.2020 working space
(`DecodePush::decode_matrix`, from real chromaticities). The **encode** side does
not separately convert intermediate (BT.2020) → output primaries — for a calibrated
output that conversion folds into the 3D LUT, and for the identity case sRGB-decode
→ sRGB-encode round-trips cleanly. This is correct for single-primary content and
for Spyder calibration (BT.2020 patches on a BT.2020 target).

It becomes wrong when **mixed-primaries content shares one output** (an HDR BT.2020
window beside an SDR BT.709 window): the SDR window's saturated colors get BT.2020
coordinates and read out-of-gamut. Fixing means either an explicit
`IntermediatePrimariesToOutput` encode fragment or baking the conversion into every
output's LUT. Recommend keeping the intermediate at BT.2020 universally. **Trigger:**
mixed-primaries content on one output looking visibly wrong.

## Renderer

### Subpixel FIR + dither encode fragments

`SubpixelFir3Horizontal` (per-channel 3-tap horizontal FIR for non-stripe subpixel
layouts, e.g. QD-OLED triangular) and `InterleavedGradientNoiseDither` (hide 8-bit
banding without a noise texture) are in the `EncodeFragment` enum but emitting
either returns `MissingFeature`. The FIR needs multi-sample synthesis: detect FIR
in the chain, sample the intermediate at 3 positions, run the pre-FIR pipeline on
each, weight-sum (kernel weights are already in the `fir_kernel_r/g/b` push slots),
then run the post-FIR pipeline — loop-unrolled at synthesis time. Dither is
simpler: `(interleaved_gradient_noise(gl_FragCoord.xy) - 0.5) * dither_strength /
255.0` added before write. **Trigger:** when we drive the QD-OLED (FIR) or an 8-bit
panel shows visible banding (dither).

### Renderer-side intermediate-buffer modifier negotiation

The fp16 intermediate is allocated with implicit tiling (Vulkan's choice) since
it's never exposed as a dmabuf — fine today. If we ever import client surfaces
*directly* as overlay-plane buffers (skipping the composite for fullscreen
surfaces), the renderer must negotiate the import modifier on those too. **Trigger:**
per-plane CRTC color properties becoming usable on Vega 20 / Navi 21 (currently
DCN2+ only). Not on the roadmap.

## Scanout / KMS

### Multi-plane (DCC-compressed) modifier import

`pick_scanout_modifiers` and `build_advertised_dmabuf_formats` both drop any
modifier with `plane_count > 1`, because `ImportedImage::import` handles a single
plane / fd / Vulkan allocation. AMD DCC modifiers carry an auxiliary metadata plane;
on desktop content DCC can cut scanout fetch ~2× on top of plain tiling — matters
most on the 4K@240 OLED. Cost: rework `import` to iterate `dmabuf.planes`, import
one `DeviceMemory` per plane (`ImportMemoryFdInfoKHR`), bind with
`VkBindImagePlaneMemoryInfo`, and populate per-plane `plane_layouts` (the field is
already array-shaped, we just fill one entry). The color-attachment paths don't
change. **Trigger:** scanout bandwidth pressure plain tiling doesn't solve, or the
240 Hz OLED arriving. (Note: the two-plane *YUV* import path already does per-plane
imports — DCC is the single-image-with-metadata-plane variant.)

### Per-plane damage clips (`FB_DAMAGE_CLIPS`)

smithay's atomic backend plumbs `FB_DAMAGE_CLIPS` from `PlaneState.damage_clips`;
we always pass `None`, so every frame is treated as full-screen damage. Telling the
display engine "only rect (x,y,w,h) changed" lets it skip fetch for unchanged tiles
(a cursor-blink redraw drops from a 4K full-fetch to a few tile bursts). Cost: track
per-output damaged rects (finer than today's output-level "needs redraw"; producer
is wayland surface damage rolled up through the layout's window→output mapping),
convert to KMS-coordinate `drm_mode_rect[]`, attach to `PlaneState.damage_clips`,
and reset to `None` on any mode-changing event. **Trigger:** laptop power
optimization, or measured idle-desktop scanout bandwidth being meaningfully high.

### Atomic test commits before mode-changing operations

Today we call `surface.commit()` / `page_flip()` directly; a kernel rejection
(incompatible HDR property, post-hotplug modifier mismatch) surfaces at commit time
when the previous state is already partially torn down. `DrmSurface::test_state`
runs the same ioctl with `TEST_ONLY` — cheap, tells us "the kernel would accept
this" first. Lets us sanity-check HDR enable, probe modeset after hotplug, and probe
VRR enable on `RequiresModeset` connectors. Cost: ~80–100 LOC wrapper in
`output_ctx.rs` plus selective call-site adoption. **Trigger:** the next "looked
right but the kernel rejected it after we changed the mode" failure.

### Per-output explicit `LINEAR` policy hint

`pick_scanout_modifiers` always appends `LINEAR` as the safe fallback but otherwise
prefers tiled. On a configuration with broken tiled fp16 (a GPU/kernel regression,
a virtual/looking-glass output) we'd want to *force* `LINEAR` for one output. Cost:
`force_linear_scanout: bool` on `prism_config::OutputConfig.color`, short-circuit
the picker. ~20 LOC. Cheap insurance; not blocking.

### Output / connector runtime hotplug

Outputs are opened at startup (`DrmEvent` handling covers `VBlank`/`Error` only).
We don't react to runtime connector add/remove. (Input-device hotplug *does* work —
libinput runs over udev and `on_device_added` flips seat capabilities.) **Trigger:**
plugging/unplugging a display on a running session.

### CRTC assignment with rebinding

We pick a non-colliding CRTC per connected connector and fail cleanly if all
compatible CRTCs are occupied by another session. We do **not** atomically un-bind a
CRTC held by another connector in the same commit. Proper rebinding comes with the
real multi-output config work.

### Shutdown quirks

- **`surface.clear()` EINVAL** on one-shot subcommand exit (`prism gradient` /
  `prism scanout`): smithay's `clear_state` releases with EINVAL after the display
  was correctly held. Only affects the end-of-run handoff back to the desktop.
- **Flaky multi-output shutdown**: `OutputContext::Drop → DrmSurface::clear()`
  normally ~100 ms, but occasionally one output's clear takes 400+ ms and the next
  output's `clear()` hangs indefinitely (watchdog SIGKILL at 13 s). Per-output Drop
  breadcrumbs are wired so the next repro has fsync'd evidence. Likely a kernel-side
  atomic-commit ordering issue with overlapping in-flight clears.

## Input

### Touch

No `wl_touch` — keyboard and pointer are wired (libinput → seat → focused surface),
touch is not. (WLCS touch tests are skipped accordingly.) Add when a touch device
or touch-driven client is in play.

## Wayland / protocol

### linux-dmabuf-v1 v4 (modifier-aware feedback)

We use v3 with a negotiated format list. v4 lets us advertise per-display *preferred*
modifiers and is the path that closes off direct-scanout with tiled modifiers.
Bundle with direct-scanout / overlay-plane work.

### Full layer-shell rendering

The layer-shell calibration/full-output pattern (Spyder's: a surface sized to the
output, rendered on top of the workspace) works. The full version — status bars,
notification daemons, wallpapers with proper anchoring/exclusive-zones/layers — is a
follow-up. (Noted at `crates/prism-protocols/src/layer_shell.rs`.)

### Remaining optional protocols

`wp_presentation`, `wp_viewporter`, `wp_fractional_scale_manager_v1`,
`wp_content_type_v1`, `xdg_activation`, `xdg-decoration`, and single-pixel-buffer are
wired. Still unwired (all graceful-degrade): `zwp_idle_inhibit` (prevent lock during
playback), `relative_pointer` / `pointer_constraints` (games, drawing apps),
`tablet_manager`. Add as specific clients need them.
