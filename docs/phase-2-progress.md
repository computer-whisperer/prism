# Phase 2 — implementation progress

Tracks what's built, what's verified, and what's deferred. Updated as tasks land.

## Status snapshot — 2026-05-20

Multi-output + multi-GPU foundation (59.1–59.3) complete. `prism run` opens every connected card, brings up all connected connectors across both GPUs as independent `OutputContext`s, and pumps vblank-driven double-buffered scanout on each at its native rate. End-to-end verified on TTY against `scripts/tty-test.sh` + `prism-shmtest`: **7 outputs (5 on card0/Vega20, 2 on card1/Navi21) all rendered the client's cycling colors at 60 Hz**, ~1200 frames in 8s, shutdown clean in ~1.1s.

Renderer CPU cost in the steady-state present path: ~60 µs/frame per output. dmabuf and shm client buffers are both imported/uploaded on every registered GPU at commit time, so any output's render path can sample without GPU↔GPU copies.

Remaining within #59: `wl_output` advertisement (59.4) — needed before real clients (mpv, browsers) run through prism — and per-output element mapping (59.5). After #59 the per-output config layer is real and EDID parsing / HDR signaling / fp16 scanout / per-display calibration plug in directly.

## What's built

| # | Task | Status | Verification |
|---|------|--------|--------------|
| 43 | prism-frame API types | ✅ | compiles, used by all downstream crates |
| 44 | Vulkan instance + device + queue | ✅ | smoke test selects Vega 20 by DRM-node match |
| 45 | DRM enumeration (read-only) | ✅ | matches hardware memory (5 conn on card0, 2 on card1) |
| 46 | Wayland server (compositor + xdg-shell + shm) | ✅ | alacritty connects, gets xdg-toplevel initial-configure |
| 47 | linux-dmabuf-v1 import → VkImage | ✅ | self-test imports a synthesized smithay::Dmabuf |
| 48 | Decode/encode shader pipeline | ✅ | sRGB-OETF gradient passes anchor-point check; TTY visual pending |
| 50 | GBM ↔ Vulkan dmabuf round-trip | ✅ | tracer clears magenta, CPU mapping reads back B=ff G=00 R=ff |
| 51 | Atomic solid-color scanout commit | ✅ | TTY-verified green frame on DP-4 (Vega 20) |
| 49a | Integrated event loop (timer-driven) | ✅ | TTY-verified; `prism run [output]` brings up wayland + scanout in one process |
| 49b | Vblank-driven render pacing | ✅ | TTY-verified at 60Hz double-buffered (kernel-wedge at single-buffered drove the BO doubling). `mark_vblank()` toggles `back_index` per vblank event from `DrmDeviceNotifier` |
| 49c | Client compositing (shm + dmabuf) | ✅ | TTY-verified via `scripts/tty-test.sh` → `prism-shmtest` drawing cycling colors @ 60Hz. Per-surface `SurfaceTexSlot` in surface `data_map`; dmabuf imports cached on `dmabuf_imported`; shm uploads via `ShmTexture` on commit |
| 55 | Renderer refactor (persistent CB + fences + push descriptors) | ✅ | per-frame CPU dropped from ~6 ms (queue_wait_idle) to ~60 µs (fence-gated slot reuse) |
| 59 | Multi-output + multi-GPU foundation | 🟡 | 59.1/2/3 done (structural split, multi-output on one card, multi-GPU + per-GPU client buffer replication). 59.4 (wl_output) + 59.5 (per-output element mapping) remaining. |

## Task #59 — multi-output + multi-GPU foundation

The tracer MVP hardcoded one card (`/dev/dri/card0`), one Vulkan device (Vega 20 by drm-node match), one connector (first connected or `[output]` arg). The whole `phase-2-backend-notes.md` color-management design assumes per-output state as a first-class concept, and `multi-gpu.md` documents the cross-GPU buffer-import patterns we'll need on the 5+1 hardware (5 outputs on Vega 20 + central panel on Navi 21).

#59 is the structural work to make all the per-output / per-GPU plumbing real. EDID parsing, HDR signaling, fp16 scanout, per-display calibration ingest all sit on top of this — they have nothing to plug into until the per-output config is a thing.

### End-state architecture (what we're building to)

```
SeatSession           1 per process. libseat-backed; holds DRM-master grant per opened card.
 └─ DrmCardContext    1 per /dev/dri/cardN we drive. Owns DrmDevice + GBM + DrmDeviceNotifier.
                       Picks connector + CRTC + mode for each output.
  ─ GpuContext        1 per physical Vulkan device. Matched to a card via drm_render dev-id.
                       Currently `Arc<prism_renderer::Device>` — may grow per-GPU upload queue etc.
   ─ OutputContext    1 per active connector. Owns DrmSurface, double-buffered scanout BOs,
                       per-output Renderer (because pipelines bake in per-output EncodeConfig).
                       References its card + its GPU.
```

`PrismState` carries:

```rust
session: SeatSession,
cards:   HashMap<DrmDevId, DrmCardContext>,
gpus:    HashMap<DrmDevId, Arc<prism_renderer::Device>>,
outputs: HashMap<OutputId, OutputContext>,
dmabuf_textures: HashMap<ObjectId, HashMap<DrmDevId, Arc<ImportedImage>>>,
                  // ^ cross-GPU import: every registered GPU gets its own VkImage
                  //   so any output's render path can sample without GPU↔GPU copies
```

`OutputConfig` (new, in prism-drm) is the static-per-output bundle: depth + format + intermediate_format + EncodeConfig today; will grow to include target color description, calibration matrix, tone-map curve choice, panel peak luminance (from EDID).

`OutputState` (already exists in prism-frame) is the per-frame state and stays as-is.

### Sub-steps

- **59.1 ✅ — structural split, no enumeration change.** Layered types defined: `SeatSession::new()` → `(Self, notifier)`; `DrmCardContext::open(&mut session, path)` → `(Self, drm_notifier)`; `OutputContext::new(&mut card, device, pick, &config)`. `PrismState` carries the maps (`cards`, `gpus`, `outputs`) even though they each hold one entry. Vblank handler routes by CRTC handle. No behavior change.
- **59.2 ✅ — multi-output on a single card.** `pick_all_connected(drm)` enumerates every connected connector with non-colliding CRTC assignment; one `OutputContext` per connector sharing the card's `DrmCardContext` + one `Arc<Device>`. Vblank routing disambiguates by CRTC. Element list (toplevels) shared across outputs. Verified on the 5 Samsungs on Vega 20.
- **59.3 ✅ — multi-GPU.** Both cards opened; per-card `DrmCardContext`; per-GPU `Arc<Device>` keyed by `DrmDevId`. `dmabuf_textures` is per-GPU: `dmabuf_imported` imports the same buffer on every registered GPU so any output's render path can sample locally. **shm is also per-GPU**: `SurfaceTexture::Shm { by_gpu }` mirrors dmabuf — bytes are read once and uploaded into one staged `VkImage` per registered GPU on each commit. Verified: 7 outputs across both cards all rendered the client's surface concurrently.
- **59.4 — `wl_output` advertisement.** One wl_output global per `OutputContext`; modes, transform, scale, geometry advertised to clients. Required before any real client (mpv, browsers) will run through prism.
- **59.5 — per-output element mapping.** First trivial rule: a surface "lives on" the output whose geometry contains its center; only that output samples it. Real placement / move-between-outputs is layout-level work for later. This is also when per-GPU shm upload can be narrowed from "every GPU" to "only the GPUs whose outputs host the surface."

### What this enables next

Once 59.x lands, the remaining HDR work plugs into per-output config naturally:

- **fp16 scanout**: `ScanoutDepth::Fp16` → `AB48` / `R16G16B16A16_SFLOAT`, allocated per-output. Today's #100 note in this doc confirms fp16 scanout (`Xbgr16161616f`) works on both DCN1 (Vega 20) and DCN3 (Navi 21).
- **PQ encode**: per-output `EncodeConfig` already exists; add `EncodeFragment::OutputTransferPq` to the chain on HDR outputs. The PQ math is in encode_synth, just unselected today.
- **KMS HDR_OUTPUT_METADATA + Colorspace**: set per-output at bringup based on the output's target color description in `OutputConfig`. niri's `build_hdr_metadata_blob` knows the byte layout; port.
- **EDID parsing**: populates `OutputConfig` defaults (peak luminance, native primaries, supported EOTFs) before user-config overrides.
- **Per-display calibration**: per-output 3×3 + 1D/3D LUT plumbing fits in `OutputConfig` + the encode synth's existing `CalibrationMatrix` fragment.

## Architectural decisions made during implementation

These were taken either silently in code or in design discussion; calling them out so the next session doesn't re-derive.

### Renderer

- **Two-pass pipeline**: per-element decode → fp32 BT.2020 absolute-nits intermediate → encode to per-output color space. Confirmed in #48.
- **Per-output encode shader is synthesized**, not statically compiled. SPIR-V is emitted at `Renderer::new` time from an `EncodeConfig` (ordered list of `EncodeFragment` variants) via `rspirv::dr::Builder`. Each fragment (`CalibrationMatrix`, `OutputTransferSrgb`, `OutputTransferPq`, `OutputTransferLinear`, planned `SubpixelFir3Horizontal`, `InterleavedGradientNoiseDither`) maps to a `fragment::emit_*` function that threads a vec3 color through. Single fixed `EncodePushSynth` struct (128 bytes, at Vulkan's minimum push-constant limit) is shared across all configurations; fragments use whichever slots they need. The vertex shader stays statically compiled (full-screen tri has no per-output variation). Rationale: per-display effects (FIR for QD-OLED, calibration LUT, dither for 8-bit panels) chain into one fragment shader rather than running as separate post-process passes — single dispatch, no ping-pong buffers, optimizer can fuse the math.
- **Intermediate format**: `R16G16B16A16_SFLOAT`. Sufficient for PQ peak (10000 fits in fp16 range). Banding risk at low values when mixing SDR+HDR; revisit if visible.
- **Per-element draws** (not full-screen-tri composite). 4-vertex triangle-strip quad per element, dynamic-rendering color attachment, premultiplied-alpha blend.
- **Encode pass** is a single full-screen triangle. Reads intermediate as `SHADER_READ_ONLY_OPTIMAL`, writes scanout as `COLOR_ATTACHMENT_OPTIMAL`, ends in `GENERAL` for KMS handoff.
- **Push constants for per-draw state.** mat4 + scalar params. No specialization constants (would multiply pipeline count for marginal perf).
- **Shader compilation**: `build.rs` shells out to `glslangValidator`. GLSL files in `crates/prism-renderer/shaders/`, SPIR-V bytes in `$OUT_DIR`, `include_bytes!` from there. Build fails fast if glslang isn't installed.
- **dynamic-rendering** is enabled (Vulkan 1.3 feature); no `VkRenderPass` / `VkFramebuffer` objects, just `cmd_begin_rendering` + `RenderingAttachmentInfo`.
- **synchronization2** for all barriers; `PipelineStageFlags2::CLEAR` for clear, `COLOR_ATTACHMENT_OUTPUT` ↔ `FRAGMENT_SHADER` for the pass-handoff barrier.
- **`PRESENT_SRC_KHR` is not valid** without `VK_KHR_swapchain`. For KMS scanout the correct final layout is `GENERAL`.

### Wayland / protocol

- Per-client data: `PrismClient { compositor: CompositorClientState }`. No security context yet.
- **`PrismState` carries `Arc<prism_renderer::Device>`** so dmabuf imports validate against real Vulkan support, not a hardcoded format list.
- **xdg-shell initial configure** is sent on first commit (from inside `CompositorHandler::commit` checking `XdgToplevelSurfaceData::initial_configure_sent`), not on `new_toplevel` — gives the client a chance to set `title` / `app_id` first.
- linux-dmabuf-v1 uses `create_global` (v3), not `create_global_with_default_feedback` (v4). Modifier-aware feedback comes later when we negotiate beyond LINEAR.
- Currently advertised dmabuf formats: **XRGB8888 + ARGB8888 with `DRM_FORMAT_MOD_LINEAR` only**. Both map to `vk::Format::B8G8R8A8_UNORM`.

### DRM / GBM

- **GEM handles are per-fd.** `GbmDevice` and `DrmDevice` must share the same `DeviceFd` (`Arc<OwnedFd>`), or `addfb2` returns ENOENT. Use `GbmDevice::from_device_fd(drm.device_fd().device_fd())`.
- libseat: backed by smithay's `LibSeatSession`. Acquires master only when on a foreground VT; falls back to ENOSYS off-TTY (which is correct behavior — we exit cleanly).
- Hardware target: both AMD cards driven concurrently. `/dev/dri/card0` is Vega 20 (primary 226:0, render 226:129), drives the 5 Samsungs. `/dev/dri/card1` is Navi 21 (primary 226:1, render 226:128), drives the central panel + ancillary HMD. Vulkan device matched to card by `Device::physical.drm_primary`.
- **Shutdown order rule:** drop `PrismState` BEFORE the event loop. The real `Rc<LibSeatSessionImpl>` (DRM master holder) lives in `LibSeatSessionNotifier` inside the calloop event loop; `OutputContext::Drop::surface.clear()` needs DRM master, so state (which owns the surfaces) must drop first. Documented at the call site in `run_integrated`.

## Deferred ("do this later") items

Listed alongside the task that introduced them so we don't lose track.

### Color management

- **Tone mapping**. Both `decode.frag` (over-bright input clamp) and `encode.frag` (above-display-peak clamp) currently hard-clip. Need BT.2390 EETF or similar. Wait until we have mixed SDR+HDR content to tune against. (#48)
- **HDR scanout BO + KMS signaling**. We've built PQ encode in the shader (`OutputTransfer::Pq`) but never invoke it — scanout is XRGB8888 LINEAR + sRGB-encoded. To exercise the PQ path needs:
  - GBM allocate as `A2R10G10B10` or `RGBA16_SFLOAT`
  - KMS connector properties: `HDR_OUTPUT_METADATA`, `Colorspace=BT.2020_RGB`, `max_bpc=10`
  - Reference: phase 1's `niri/src/backend/tty.rs::build_hdr_metadata_blob` knows the byte layout
  - Belongs in its own task (not in #48).
- **Calibration ingest**. `EncodePush::cal_matrix` is identity. We have Spyder-derived 3×3 corrections from phase 1; need to plumb them in (config file? IPC?). Belongs to a later "real per-display config" task.
- **Per-element decode transfer functions**. `decode.frag` only implements Linear + sRGB. PQ EOTF is written but not anchor-tested (no PQ-source content to feed it). HLG, BT.1886, and Gamma are TODO stubs.
- **3D LUT support**. For now only 3×3 + transfer. Non-linear corrections need a 3D LUT path; out of scope until calibration produces one.

### Renderer

- **Dynamic descriptor pool sizing**. Decode pool is hardcoded to 64 max sets, encode to 8. Will need real sizing once we know element-counts-per-frame from real workloads. (#48)
- **Multi-element compositing in the same frame.** The pipeline supports it (it loops elements and binds per-draw descriptor sets) but only exercised with 1 element in tests. Need a multi-element test once we have something to composite.
- **Render target reuse**. `render_frame` creates a new `VkImageView` per call and destroys it; should cache per scanout image. Minor perf, not correctness. (#48)
- **Per-frame command-buffer reuse**. Each frame allocates and frees a new command buffer. Same — minor.
- **Real frame pacing.** `render_frame` is synchronous: `queue_wait_idle` at the end. `prism run` currently uses a 60Hz calloop `Timer` instead of vblank events — that's #49b. Tearing is possible because we're single-buffered and the timer doesn't sync to scanout.
- **Double-buffering: NOW REQUIRED, in place.** Originally documented as a #49b nice-to-have for tearing; turned out to be necessary for correctness. Single-buffered rendering at 60Hz wedged the kernel on amdgpu+RADV — the 3D engine writing into the actively-scanned-out BO while the display engine reads it hits implicit synchronization that fully locks the system. Now: two BOs, `back_index` tracks which to render into, `mark_vblank()` toggles it after each vblank event (the just-flipped BO is now front, the other is now safe to write). Confirmed via breadcrumb log: 1Hz worked single-buffered (low contention); 60Hz did not; needs re-test at 60Hz double-buffered.
- **`DrmDeviceNotifier` + `LibSeatSessionNotifier` MUST both be polled.** First hotfix (37ec3f9) added DRM notifier drain — kernel page-flip event allocations were cascading to ENOMEM at 60Hz. Second hotfix (this) added libseat notifier drain — without it, logind can't request a VT switch (we never ack the "pause"), Ctrl+Alt+Fn hangs, AND SIGINT delivery is blocked because the desktop session is stuck waiting on us. The second variant was more catastrophic than the first: a tight loop holding DRM master + libseat unack-able = no escape, full system reboot. `OutputContext::new` now returns `(ctx, OutputNotifiers { drm, session })`; the caller MUST insert both into the calloop loop. **Pattern to internalize:** any time smithay returns a `*Notifier` alongside a resource, ASSUME it must be drained, do not bind it to `_`. The one-shot subcommands (`prism scanout`/`prism gradient`) hold the session for ≤5s so are bounded, but they're documented at the call site as "diagnostic only — VT switch may hang during the hold."
- **`SubpixelFir3Horizontal` fragment** is in the `EncodeFragment` enum but `synthesize_fragment_shader` errors `MissingFeature` if it's in the chain. Multi-sample synthesis means the synthesizer needs to: (1) detect FIR in the chain, (2) generate code that samples the intermediate at 3 positions, (3) run the pre-FIR pipeline (e.g. cal matrix) on each sample, (4) weight-sum, (5) run the post-FIR pipeline. Loop unrolling at synthesis time. The kernel weights come from the `fir_kernel_r/g/b` push-constant slots which already exist.
- **`InterleavedGradientNoiseDither` fragment** — same story, in the enum but not implemented. Math: `dither = (interleaved_gradient_noise(gl_FragCoord.xy) - 0.5) * push.dither_strength / 255.0`, added to the encoded result before writing.
- **`OutputConfig` color-extension.** Today's `OutputConfig` (in prism-drm) carries depth + vk_format + intermediate_format + EncodeConfig — sufficient for SDR sRGB outputs. The target color description / panel peak luminance / calibration matrix fields will land alongside EDID parsing and the per-display config layer.

### Wayland / protocol

- **wl_output advertisement.** Not implemented yet. Without it, real clients (mpv, browsers) probe-fail and bail to the host compositor. Blocks "actually run a client through prism."
- **wl_seat / input.** Not implemented. Same blocker class as wl_output — many clients want to know they have an input device.
- **linux-dmabuf-v1 v4 feedback (modifier-aware).** Current v3 advertises a hardcoded format list. v4 lets us advertise per-display preferred modifiers; the negotiation closes off direct-scanout with tiled modifiers (which we don't support yet, but will need to).
- **Vulkan format-modifier capability query.** Right now we hardcode "we can import XRGB8888 + ARGB8888 LINEAR." Real querying via `VK_EXT_image_drm_format_modifier::vkGetPhysicalDeviceFormatProperties2` would advertise everything the GPU actually supports.
- **Multi-planar dmabuf import**. `ImportedImage::import` rejects multi-plane sources. Needed for NV12 / P010 (video clients) but not for the tracer MVP.

### DRM / KMS

- **Real CRTC assignment**. We avoid CRTCs bound to *other* connectors by another session (would require atomically un-binding them in the same commit, which we don't do). If all compatible CRTCs are occupied, we fail with a clean error rather than the cryptic kernel reject. Proper multi-output CRTC assignment with rebinding comes with the multi-output config work. (#51)
- **HDR_OUTPUT_METADATA stickiness**. Confirmed 2026-05-20 on DP-4: phase-1 niri set this property; it persists across master handoffs and across prism invocations. Result: panel interprets our sRGB-encoded bytes as PQ, anything above byte ~100 clamps to peak nits — gradient saturates at ~25% screen brightness. Need to explicitly set `HDR_OUTPUT_METADATA = 0`, `Colorspace = Default` on the connector in our atomic commit for SDR content. Will need to be done correctly anyway when we add HDR scanout (other direction of the same toggle).
- **`surface.clear()` returns EINVAL.** Smithay's clear_state call at the end of `prism gradient` / `prism scanout` releases with `Invalid argument`. The display has already been correctly held for 5s; only affects the brief end-of-run handoff back to the desktop session. Investigate when we wire real frame pacing.
- **VRR / Freesync wiring**. The DrmSurface API supports it (`use_vrr`, `vrr_supported`); we just don't enable it. Was a phase 2 requirement.
- **Hot-plug handling proper.** `LibSeatSessionNotifier` and `DrmDeviceNotifier` are now drained (required for VT-switch + page-flip event accounting), but we don't yet react to hot-plug add/remove of devices or connectors. Multi-card discovery (#59.3) landed open-everything-at-startup; runtime hotplug is a follow-on.
- **10-bit scanout availability is per-display.** `ScanoutDepth::{Bpc8,Bpc10}` selects the GBM format and Vulkan format together. The `max bpc` connector property is set accordingly (or silently skipped if the connector doesn't expose it). Per-output negotiation (which displays support 10-bit, what HDR transfer functions, etc.) belongs in the future config layer; for now `prism gradient [output] [8|10]` lets us drive each path manually.

## Subcommands today

```
prism             headless smoke suite: Vulkan probe + DRM enum + tracer paths
prism scanout [output]              TTY: clear an output to green via vkCmdClearColorImage, hold 5s
prism gradient [output] [8|10]      TTY: render gradient through decode→encode pipeline, hold 5s
                                      depth defaults to 10 (XR30 + max_bpc=10); pass 8 for XR24 path
prism run [output] [8|10]           TTY: integrated mode — wayland server + scanout + vblank-driven render
                                      on every connected output across every opened card. Ctrl-C to exit.
                                      Env: CARDS (comma-separated DRM paths; default both card0+card1),
                                           PRISM_MAX_RUNTIME_SECS (wall-clock self-shutdown),
                                           PRISM_WATCHDOG_SECS (hard-kill), PRISM_CRUMBS (breadcrumb file)
prism wayland     wayland server on wayland-N; logs surface lifecycle, no rendering
```

`scripts/tty-test.sh [seconds]` launches `prism run` plus the in-tree `prism-shmtest` client against it, polls for the new wayland socket, lets things run for `seconds`, and dumps log tails. Env knobs: `OUTPUT`, `DEPTH`, `RUST_LOG`.

The headless suite runs three tracer self-tests on every invocation:
- GBM → Vulkan → clear → CPU readback (verifies dmabuf import infrastructure)
- dmabuf-handler import path (verifies the wayland-side dmabuf protocol flow)
- Render pipeline gradient (verifies decode→encode end-to-end against sRGB OETF anchor points)
