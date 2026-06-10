# Deferred work

The backlog: things we deliberately *did not* do, grouped by subsystem, each with
why it's deferred and roughly what triggering it would cost. This is the place to
look before starting something — it may already be scoped here.

Feature-sized gaps from the 2026-06-09 deep-dive review are also tracked as
GitHub issues (#25–#32); items below cite their issue number where one exists.

Ordering within each section is rough priority. When an item lands, delete it (and
update [status.md](status.md)).

## Color / HDR

KMS HDR signaling, EDID parsing, per-output calibration (3D LUT, CTM, response
curve), the measured-gamut bake, render intents + reference-white anchoring, and
SDR drive-domain LUTs are **done** — see [status.md](status.md). What remains:

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

### Gradient decorations

Focus ring / border render solid colors; the config's gradient variants
(linear gradients with color-space interpolation, niri-style) are parsed but
collapse to the corresponding flat color. Needs gradient parameters plumbed to the SDF
fragment — push-constant space is full, so this wants a small UBO (or SSBO) per
frame. **Trigger:** wanting niri-parity visuals; bundle with the tab-indicator
render work (#29) which has the same plumbing shape.

### Remaining damage-tracking increments

Damage tracking landed through the decode path (content tokens, per-output
region tracker, decode-pass scissor, occlusion culling, zero-damage frame skip).
Remaining increments, in order of value: **encode-pass scissor with buffer-age**
(the encode currently re-runs full-screen every presented frame), **per-rect
scissor** (today damage collapses to one bounding rect per output), and KMS
`FB_DAMAGE_CLIPS` (see Scanout below). Direct scanout is mostly a non-goal in
HDR mode — the encode pass has to run anyway on calibrated outputs.

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

## Layout / window management

### Insert hint + tab indicator rendering (#29)

The layout computes both (insert-hint geometry during interactive move, tab
indicator state per tabbed column) but the render side is stubbed — interactive
move gives no drop-location feedback and tabbed columns are visually
indistinguishable. The element vocabulary they need (RoundedBox + gradients)
mostly exists; see Gradient decorations above for the shared plumbing.

### Layer rules are inert (no issue yet)

`prism-layout` has `ResolvedLayerRules` types and the config parses the
`layer-rule` section, but nothing ever computes rules for a layer surface — the
whole section is a no-op (same shape as the window-rule gap fixed in f3b707f).
Port the resolution call into the layer-surface map path.

### Resize-snapshot anchor displacement

The resize crossfade captures the old frame at the window's *current-frame*
position rather than the previous-frame one, so a simultaneous move+resize can
show the old content one frame of motion off (noted in 037f2c7's commit
message). One-frame visual nit; revisit if it's visible in practice.

### Overview leftovers

Per-card wallpaper backdrops, key-repeat for overview binds, and touch input in
the overview are deferred (the pointer/gesture/keyboard paths are in). Touch is
gated on touch support generally — see Input.

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
we always pass `None`, so the display engine treats every flip as full-screen
damage. The per-output damage tracker now exists renderer-side — remaining cost:
convert its rects to KMS-coordinate `drm_mode_rect[]`, attach to
`PlaneState.damage_clips`, reset to `None` on mode-changing events. **Trigger:**
laptop power optimization, or measured idle-desktop scanout bandwidth being
meaningfully high.

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

### Desktop output / connector runtime hotplug

Desktop outputs are opened at startup; we don't react to runtime connector
add/remove for them. The udev/connector-probe machinery now exists for
**DRM-lease** connectors (VR headsets plug/unplug live, 5a949df) — extending it
to desktop outputs means driving full output bringup/teardown (CRTC pick,
swapchain, wl_output lifecycle, layout reflow) from the hotplug event instead of
startup. **Trigger:** plugging/unplugging a monitor on a running session.

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

### Bind gating: allow-when-locked / allow-inhibiting

Both bind flags parse but aren't enforced: `allow-when-locked` is meaningless
until session lock exists (#25), and `allow-inhibiting` until
keyboard-shortcuts-inhibit is wired. Land each alongside its protocol.

### Cursor: hardware-plane constraints

Cursors render only via the hardware cursor plane (`update_output_cursors`). Client
surface cursors are read from **shm** and uploaded to the plane BO; consequences:
a client cursor larger than the BO can't be shown (logged, falls back to the theme
cursor), dmabuf-backed cursor surfaces fall back too, and the sprite goes raw to
scanout (bypasses color management — same as the themed cursor). Animated cursors
show the committed frame, not timed multi-frame animation. A software-composite
cursor path (through the color-managed render pass) would lift all of these but
costs a full-frame redraw on cursor move; add if a real client needs it.

### Cursor: app-driven hide-on-keystroke (Firefox)

Under niri, typing in Firefox's URL bar hides the cursor; under prism it doesn't
(alacritty hides under neither — so it's app-driven, not a global setting). Apps
hide the cursor via core `wl_pointer.set_cursor` with a *null* surface (⇒
`CursorImageStatus::Hidden`) — there is no separate hide protocol, and prism's
`cursor_image(Hidden)` path *should* disable the plane (`resolve None →
hide_all_cursors`). Since the I-beam (set_cursor with a real surface) *does* work
on prism, `cursor_image` is reached; so either Firefox isn't sending null on
keystroke on this setup, or the `Hidden` path has a bug a code-read didn't catch.
Not yet chased down — the `debug!(kind, "client set cursor image")` in
`cursor_image` will show what the app actually sends (`RUST_LOG=…=debug`). The
`cursor { hide-when-typing }` option hides compositor-side regardless, so it's the
practical workaround.

## Screen capture

wlr-screencopy (SHM + dmabuf, async, serviced from the render loop) is done and
grim-verified — see [screen-capture.md](screen-capture.md) for the subsystem doc.

### Recording performance

wf-recorder recording is laggy; async SHM did *not* fix it. Diagnosis (three
stacked costs: per-frame 33 MB allocation, single-`AsyncSlot` fence stall, 33 MB
memcpy) and the fix ladder (cheapest: multi-slot + steer recorders to the dmabuf
path) live in [screen-capture.md](screen-capture.md). Deliberately deferred.

### PipeWire / portal capture, then ext-image-copy-capture

The planned path for OBS/portal screen sharing: in-process PipeWire stream +
the Mutter-style D-Bus portal interface, reusing the shared
`capture_into_dmabuf` primitive; `ext-image-copy-capture-v1` after that.
Not started.

## Wayland / protocol

### Session lock (#25)

No `ext-session-lock-v1` — there is no way to lock the screen. The biggest
user-facing protocol gap. Also gates `allow-when-locked` bind enforcement and a
real `Quit` confirmation flow.

### Text input / IME (#26)

No `zwp_text_input` / `input_method` — no CJK or on-screen-keyboard input.

### Data control (#27)

No wlr-data-control / ext-data-control — clipboard managers (cliphist et al.)
silently see nothing (`selection.rs:50`).

### IPC introspection (#28)

`Workspaces`, `Windows`, and `EventStream` IPC requests return "not
implemented", and `LogicalOutput` hardcodes `x:0, y:0` — so scripting parity
with niri's `niri msg` doesn't exist yet and multi-monitor region capture
(slurp-style) mis-targets. The event-stream design should follow niri's
(initial-state replay + deltas).

### Quit confirmation

`Quit { skip_confirmation: false }` just logs and quits — the confirmation
dialog needs compositor-drawn overlay UI that doesn't exist yet (same bucket as
an on-screen-display layer generally).

### linux-dmabuf-v1 v4 (modifier-aware feedback)

We use v3 with a negotiated format list. v4 lets us advertise per-display *preferred*
modifiers and is the path that closes off direct-scanout with tiled modifiers.
Bundle with direct-scanout / overlay-plane work.

### Layer-shell remaining gaps

Layer **popups** landed (20bfc36). Remaining: layer **shadows**; exclusive
keyboard grab is scoped to the *focused* output (an exclusive surface on a
non-focused monitor waits until that monitor is focused); and layer rules are
inert (see Layout above).

### Foreign-toplevel: parent relationship

`wlr-foreign-toplevel-management`'s `parent` event is never sent (dialogs don't
group under their parent in taskbar-style clients). Small; needs the xdg parent
chain exposed to the protocol module.

### Remaining optional protocols

Still unwired (all graceful-degrade): `tablet_manager` (drawing tablets),
`keyboard-shortcuts-inhibit` (VMs/remote-desktop grabbing all keys; also gates
`allow-inhibiting`), `security-context` (sandboxed-client tagging). Add as
specific clients need them.

Idle-inhibit honors an inhibitor while its surface is *alive*, not gated on
visibility (the protocol's "ignore invisible inhibitors" note) — a backgrounded
inhibitor still blocks idle. Refine if it bites. (Firefox didn't request inhibition in
testing; mpv does — that's a Firefox-side behavior, not prism.)

### Xwayland follow-ups

xwayland-satellite integration is in (on-demand spawn). Remaining: the optional
game-oriented protocols satellite can use when present (pointer warp,
fractional-scale-v2 game hints), picking up satellite path changes on config
reload, and clearing a stale X11 lock file left by a crashed satellite (blocks
the display number until removed by hand).

## Tracked code debt

#30 catalogs the load-bearing `TODO`s in the tree (places where a comment is
standing in for required behavior, as the transaction system's was) — work
through it opportunistically when touching the surrounding code.

## Release / tooling

- **AUR push** — PKGBUILD + .SRCINFO are committed and makepkg-validated;
  publishing waits on a `v0.1.0` tag, the repo going public, and `updpkgsums`.
- **smithay fork → upstream** — the DnD fix (cancel drops on data-device-less
  clients) lives on our fork branch; upstream PR still to be filed. Until it
  merges we track the fork, which adds a rebase cost to smithay bumps.
- **prism-tune calibration GUI** — the GUI has the control panel + inspectors;
  driving an actual calibration run (characterize/calibrate-lut3d) from the GUI
  is deferred, CLI remains the path.
