# Async render rework — cross-GPU mirror + per-GPU concurrency

Status: **design / not started.** Phase 1 detailed below; Phase 2 sketched.

## Motivation

The per-frame render profiler (`prism-tune profiling`) surfaced this on a
Twitch stream displayed on DP-6 (a Vega 20 output), with the Navi 21 as the
primary GPU — i.e. the cross-GPU **mirror** path:

```
span      p50      p95   (µs)
walk       13       20
damage      2        3
lower       6        8
submit    103      171
decode   7632     7648
deband   6271     6688
encode    323      521     <- 4K, full LUT, yet ~20x cheaper than decode
```

### Diagnosis

It is not pixel count or LUT cost — it is **memory locality**.

- `encode` reads the per-output fp32 intermediate and writes the scanout,
  **both in the target GPU's local VRAM (OPTIMAL-tiled)**. ~175 MB local
  traffic / ~400 GB/s ≈ 0.4 ms. Matches.
- `decode` and `deband` each read the **source surface**, which on this
  output is the cross-GPU mirror: a **LINEAR, host-visible (GTT) image**
  (`cross_gpu.rs` `pick_exportable_memory` prefers `HOST_VISIBLE`, because
  peer GPUs cannot read each other's VRAM). The target imports that dmabuf
  and **samples it directly** (`surface_tex.rs:164`, `GpuTex::view()` →
  `target.view()`). Every texel fetch crosses PCIe from an untiled buffer.
- The two passes each scan the mirror once: `deband`'s first downsample blit
  reads it (`pipeline/deband.rs:168`, source → device-local fp16 scratch),
  and `decode`'s binding 0 reads it again per output fragment
  (`shaders/decode.frag:283`). So the same buffer is dragged across PCIe
  **twice per frame**, untiled, in two separate passes.

The minimum unavoidable cost is **one** streaming PCIe transfer of that
buffer. Today we pay two scattered, cache-hostile ones, inline on the
graphics queue, on the render critical path.

## Hardware reality (probed 2026-06-30, this machine)

Cards: Navi 21 (RX 6900 XT) + Vega 20 (Radeon VII), RADV / Mesa 26.1.3.

**Queue families (both GPUs, default):**

| Family | Count | Flags |
|---|---|---|
| 0 | 1 | GRAPHICS + COMPUTE + TRANSFER (the GFX pipe we use today) |
| 1 | **4** | COMPUTE + TRANSFER (async-compute engines / ACEs) |
| 2,3 | 1 ea | VIDEO_DECODE / ENCODE (Navi only) |
| last | 1 | SPARSE only |

- **No dedicated SDMA/transfer-only family by default.** It appears *only*
  under `RADV_PERFTEST=transfer_queue` (experimental, historically buggy —
  do not build on it).
- **The lever is family 1 — the 4 ACE queues.** Always present, stable, run
  concurrently with the GFX engine, carry both TRANSFER (so `vkCmdCopyImage`
  runs there) and COMPUTE. This is the transfer-overlap engine *and* the
  future substrate for intra-GPU multi-output overlap.

**PCIe P2P is not viable here** (so GTT staging stays):
1. BARs are **256 MB** against 16 GB VRAM — Resizable BAR off; no flat
   peer-VRAM window.
2. Cross-architecture cards on plain PCIe — no XGMI (`use_xgmi_p2p`
   irrelevant), no `pcie_p2p` amdgpu param.
3. RADV exposes no peer-VRAM-as-importable-dmabuf path anyway.

## Current pipeline concurrency (for reference)

Single-threaded but **not** synchronous:
- 1 calloop thread; vblank → mark output `Queued` → `redraw_queued_outputs`
  renders queued outputs **serially** (`prism/src/main.rs:2433`).
- 1 `Device` per GPU, **1 graphics queue each** (`device.rs:319`); 1
  `Renderer` per output.
- CPU/GPU overlap via `FRAMES_IN_FLIGHT = 2` slots; the slot fence wait is on
  the frame from 2 frames ago, so it only blocks if a GPU falls behind
  (`renderer.rs:787`).
- Page-flip is non-blocking atomic with `IN_FENCE_FD` (`output_ctx.rs`).

The mirror handshake is already fully GPU-side (sync_file fds, never a CPU
fence wait): `prepare_mirror_waits` (`state.rs:4718`) submits the home→GTT
copy via `copy_batch_async` on the **home graphics queue** (`cross_gpu.rs:512`),
exports a copy-done semaphore, the target render waits it.

## Requirement that frames the rework

Daily driver: **6 monitors, 2 on Navi, 4 on Vega, all VRR**, compositing and
flipping at independent rates. VRR removes the fixed-vblank stagger that lets
the serial loop survive today. Hard requirements:
1. All output operations able to run asynchronously (overlap mirror transfers
   with decode/encode).
2. Multiple composites progress in parallel.

Physical bound to keep honest: across the two GPUs this is *true* parallelism;
*within* one GPU the single GFX engine time-slices graphics passes — intra-GPU
"parallelism" means transfer/compute (ACE) overlap + tight GFX packing, not
two parallel graphics pipes.

---

## Phase 1 — per-device ACE queue + target-local mirror copy

Goal: collapse the two untiled GTT scans into **one streaming GTT→local-VRAM
copy on an ACE queue**, so decode/deband sample local tiled memory and the
copy overlaps GFX work. Additive — does not touch the threading model.
Expected: cross-GPU output ~14 ms → ~3–4 ms (measure with the profiler).

### 1.1 Device gains an ACE queue

`crates/prism-renderer/src/device.rs` (~:317-340): today one
`DeviceQueueCreateInfo` for `graphics_queue_family`. Add a second queue from
**family 1** (COMPUTE+TRANSFER):
- Probe for a queue family with `COMPUTE | TRANSFER` that is *not* the
  graphics family; store its index as `transfer_queue_family`. Fall back to
  the graphics queue if absent (single-family GPUs, llvmpipe).
- Request one queue from it; store `transfer_queue: vk::Queue`.
- Keep the deferred-destroy serial machinery; the ACE queue submits bump the
  same monotonic serial.

### 1.2 Target-local image on `GpuTex::Mirror`

`crates/prism-protocols/src/surface_tex.rs:134`. The `Mirror` variant today
holds `scratch` (home, LINEAR/GTT) + `target` (that scratch imported on this
GPU, LINEAR/GTT — what decode samples). Add:

```
target_local: [Arc<LocalImage>; 2],   // OPTIMAL device-local, double-buffered
target_local_idx: Cell<usize>,        // flips each copy
```

- Format = the import's format; tiling OPTIMAL; usage
  `SAMPLED | TRANSFER_DST`; **`SharingMode::CONCURRENT`** across
  {graphics, transfer} families (avoids per-frame ownership-transfer
  barriers; an every-frame sampled image makes the concurrent cost
  negligible vs. the barrier dance).
- Same for `MirrorChroma` (half-res chroma plane).
- `GpuTex::view()` / `yuv()` return `target_local[idx].view()` instead of
  `target.view()`. **`target` (the GTT import) is now read only by the ACE
  copy, never by the GFX render.**

### 1.3 Target-side copy (ACE queue)

New per-GPU copier stage (extend `MirrorCopier`, or a sibling
`TargetMirrorCopier`) that records, on the **transfer queue**:

```
vkCmdCopyImage(target /*LINEAR,GTT*/  →  target_local[idx] /*OPTIMAL,VRAM*/)
```

LINEAR→OPTIMAL detiling is handled by the driver; the transfer/compute family
carries TRANSFER capability. Validate this runs on family 1 (spec allows
image copy on a transfer-capable queue; confirm radv accepts it).

### 1.4 Sequencing (the careful part)

Insert one ACE stage between the existing home copy and the target render.
All dependencies stay GPU-side semaphores.

```
home GFX:  client_src ──copy──▶ scratch (GTT)        signals  copy_done
target ACE: target (GTT) ──copy──▶ target_local[idx]  waits copy_done,
                                                       signals local_done
target GFX: decode/deband sample target_local[idx]    waits local_done
```

The overwrite races shift accordingly:
- **GTT scratch** is now read only by the **ACE copy**. So the next home→GTT
  copy must wait the target's **ACE-copy-done** fence, not the render-done
  fence. Re-point `render_done_dup` / `note_mirror_render_done`
  (`state.rs:4818,4853`) to the ACE copy's completion.
- **`target_local`** is written by the ACE copy, read by the GFX render.
  Double-buffer (`[_; 2]`, flip per copy) so ACE-copy(N+1) writing buffer B
  cannot tear render(N) reading buffer A. The GFX render waits `local_done`
  for the buffer it samples.
- `prepare_mirror_waits` returns the **`local_done`** semaphores (instead of
  `copy_done`) to hand to `present()` → `render_frame`'s wait list.

### 1.5 Skip when unchanged

The ACE copy only needs to run when the mirror was re-copied this commit (new
client content). If the surface is static, `target_local` still holds the
last frame — skip the copy and reuse the same `target_local_idx`. Ties into
the existing per-commit `acquire_waited` tracking (`state.rs:4777`). Without
this, a static mirrored window re-copies every frame for nothing.

### 1.5b Status — GFX-first sub-step DONE (build/clippy/tests clean, runtime-unverified)

The locality half of Phase 1 is implemented with the copy on the **graphics**
queue (recorded at the start of `render_frame`'s cb), deferring the ACE-queue
move and double-buffering. Provably correct via the *existing* render-done gate
(the copy now lives inside the render cb, so `note_mirror_render_done` already
covers the GTT-scratch overwrite race). `target_local` is **single-buffered**,
i.e. exactly the same write-then-read-per-frame profile as the persistent
intermediate — no new hazard class; double-buffering (next-next step) makes it
strictly safer + enables cross-frame pipelining. Remaining for full Phase 1:
move the copy to the ACE `transfer_queue` (separate submit + `local_done` sem,
the gate re-point in 1.4) for GFX/ACE overlap, then double-buffer. Also TODO: a
dedicated profile span for the copy (currently recorded before t0, so it lands
in total frame time but not in any per-span bucket).

### 1.6 Touch list

- `device.rs` — ACE queue family probe + queue.
- `cross_gpu.rs` — target-copy record method on the transfer queue + its
  completion fence/semaphore; re-point the overwrite-gate fence.
- `surface_tex.rs` — `target_local` (+chroma), `view()`/`yuv()` redirect,
  allocation on import/refresh.
- `state.rs` `prepare_mirror_waits` / `note_mirror_render_done` — drive the
  ACE copy, return `local_done`, re-point render-done.
- No shader changes (decode/deband bindings unchanged; they just sample a
  local image now).

### 1.7 Validation

- **Correctness:** mirrored window (e.g. Twitch on the Vega output) renders
  identically; no tearing under rapid scroll (the producer-sync race class
  from `firefox-scroll-blue-bleed` must stay closed — the new `local_done`
  chain must still transitively wait the client's write fence).
- **Perf:** `prism-tune profiling` before/after on the Vega output — expect
  `decode`+`deband` to collapse (local sampling) and a new bounded
  ACE-copy cost that overlaps GFX.
- **PCIe floor:** `sudo lspci -vv | grep LnkSta` for the negotiated link
  speed; the streaming copy is PCIe-bound and this sets its floor.
- YUV mirror (chroma plane) path exercised separately from RGB.

### 1.8 Risks

- ~~Image copy rejected on the ACE/compute queue by radv~~ **RESOLVED**
  (2026-06-30): `examples/ace_copy_probe.rs` runs the exact op — LINEAR
  host-visible → OPTIMAL device-local `vkCmdCopyImage` on a family-1 queue,
  then back — on every physical device. **PASS on both Navi 21 and Vega 20**,
  validation-clean, round-trip pixels correct. Lesson baked into the probe:
  `HOST_READ`/`HOST_WRITE` access masks are illegal in barriers on a
  compute/transfer queue (no host stage in its `ALL_COMMANDS` expansion) — the
  real path doesn't touch host memory on the ACE queue, but barrier access
  masks there must stay within transfer/compute stages.
- CONCURRENT sharing perf regression vs. EXCLUSIVE+ownership-transfer →
  measure; switch to explicit ownership transfer if it shows.
- Double-buffer + skip interaction: a skipped copy must not advance the
  buffer index or the render samples a stale half.

---

## Phase 2 — per-GPU render threads (full design)

Status: **designed 2026-07-01, not started.** This section is the
implementation contract: decisions, ownership split, message protocol,
staged increments, and the validation plan. Written to be executable by a
fresh session without the design discussion.

### 2.0 Goal and evidence

Required by the 6-output VRR target (2 Navi + 4 Vega, independent flip
cadences): Navi ∥ Vega true parallelism, and a struggling GPU must never
stall the other GPU or input.

What actually blocks the single main thread today (all inside
`render_one_queued` → `render_output_now` → `OutputContext::present`):

1. **The FRAMES_IN_FLIGHT slot-fence wait** (`renderer.rs`, `FrameSlot::fence`)
   — blocks whenever a GPU falls 2 frames behind. This is the killer: a Vega
   output over budget stalls rendering for *every* output.
2. **CPU record + `vkQueueSubmit`** — the `submit` span, ~100–200 µs/output
   today, up to ms with big scenes. Serial across all 6 outputs.
3. **Atomic commit / page-flip ioctls.**

The scene walk + lower is *cheap* (walk 13–20 µs, lower ~6 µs per the
profiler) — that informs the split: scene production can stay on main;
execution must move.

### 2.1 Design decisions

**D1 — one render thread per `Device` (per GPU/card), not per output.**
`VkQueue` requires external synchronization; all outputs on a GPU share its
queues, and the single GFX engine time-slices graphics work anyway — a
thread per output on one GPU buys nothing and forces queue locking on the
hot path. One thread per GPU serializes that GPU's submits for free and
makes the two GPUs truly parallel. Cards and GPUs are 1:1 here
(`DrmDevId` keys both `cards` and `gpus`).

**D2 — scheduling stays on main; execution moves to the render threads.**
Main keeps the redraw state machine, frame-callback/feedback dispatch, and
the layout walk; the render thread runs damage-compute → record → submit →
flip. Frames flow main → render thread as a self-contained `LoweredFrame`;
vblanks and present outcomes flow back as messages.

*Rejected: full scene-snapshot render threads* (main pushes scene state,
threads render at their own cadence — the game-engine model). Animations
are sampled by the layout at a per-frame target time, so main would have to
produce a per-output-per-frame snapshot anyway; it collapses back into D2
with more machinery. *Rejected: moving the redraw state machine to the
render threads.* Frame callbacks, presentation feedback, and animation
clocks are all main-side; splitting the state machine across threads buys
~nothing (main's per-frame work is µs) and costs niri-parity — the
`redraw.rs` semantics survive almost unchanged under D2.

**D3 — queue submits get a lock instead of strict thread-ownership.**
The pure "only the owning thread touches the queue" model breaks on the
mirror path: lowering a *target*-GPU frame on main submits the home→GTT
copy on the **home** GPU's queue (`prepare_mirror_waits` →
`copy_batch_async`), and shm uploads submit from the commit path on main.
Rather than marshal every submit to the owning thread, `Device` grows an
internal per-queue `Mutex` and all submit/wait-idle sites go through
locked helpers. Contention is negligible (submits are µs and rare from
main). The render thread remains the *primary* submitter; the lock makes
the exceptions correct.

**D4 — no Wayland object ever crosses a thread.** All wayland-server /
smithay object interaction (frame callbacks, presentation feedback,
screencopy completion, syncobj release trackers) stays on main.
`LoweredFrame` carries only Vulkan handles + `Arc`s + `OwnedFd`s + plain
data. This sidesteps every wayland-rs thread-affinity question.

**D5 — `DrmCardContext` (fd, gbm, master/session, leases) stays on main;
`OutputContext` moves to the render thread.** VT switch, DRM lease
(SteamVR), and udev handling keep their current shape. `DrmSurface` holds
its own internal handle to the device fd (`output_ctx.rs:196`), and
page-flip ioctls are thread-safe — so the render thread can flip while
main owns the card. DRM events (vblank) keep dispatching on main's
`DrmDeviceNotifier` and are **forwarded** to the owning render thread as
messages (the kernel timestamp travels in the message, so `FrameClock`
precision is unaffected; the forward hop is µs).

**D6 — render threads are plain `std::mpsc` loops, not calloop.** With
scheduling (and estimated-vblank timers) on main, the render thread needs
exactly one wait point: blocking `recv()` on its command channel. No
timers, no sub-sources. The backchannel to main is a `calloop::channel` so
it wakes the main loop.

### 2.2 Ownership after the split

| Main thread (calloop, `PrismState`) | Render thread (one per GPU) |
|---|---|
| Wayland dispatch, input, layout, config, IPC | — |
| `cards: DrmCardContext` (fd, gbm, master, leases) | — |
| DRM event dispatch (forwards vblanks) | `mark_vblank` bookkeeping |
| `gpus: Arc<Device>` (shared; locked queues) | primary submitter |
| redraw state machine (`OutputRedrawState`) | — |
| estimated-vblank timers | — |
| scene walk + lower → `LoweredFrame` | damage compute, record, submit, flip |
| `SurfaceTexSlot` materialization (commit path) | — |
| mirror prep (`prepare_mirror_waits`, gates) | ACE copy submit (increment C) |
| frame callbacks, feedback, screencopy completion | — |
| `OutputShadow` (identity, geometry, config, redraw) | **`OutputContext`** (DrmSurface, swapchain, `Renderer`, `DamageTracker`, `FrameClock`, `CursorPlane`) |

`PrismState` stays `!Send` (`Rc<RefCell<Config>>` etc.) — it never moves.
The refactor extracts `outputs: HashMap<OutputId, OutputContext>` out of it;
main-side readers of `state.outputs` (IPC info queries, cursor logic,
config, power) are re-pointed at a new lightweight `OutputShadow` map that
mirrors identity/geometry/config/color/redraw-state. Inventory the readers
with `grep -n 'state.outputs\|\.outputs\.' crates/prism*/src` during B2 —
each is either shadow data (keep on main) or execution state (move).

### 2.3 Message protocol

Main → render thread (`std::mpsc::Sender<RenderCmd>`, one per GPU):

- `SubmitFrame { output: OutputId, frame: LoweredFrame }`
- `Vblank { output: OutputId, crtc, presentation_time }` — forwarded DRM
  event; runs `mark_vblank` (clears `frame_pending`, flips `back_index`,
  feeds `frame_clock.presented()`).
- `CursorUpdate { output, pos, visible, sprite: Option<SpritePixels> }` —
  sprite pixels only on change (≤256 KB); position rides `SubmitFrame`
  when a frame is going anyway.
- `CreateOutput { … } / DestroyOutput { … } / Reconfigure { mode, vrr, color, hdr }`
  — `OutputContext` is *created on* the render thread (gbm BO allocation
  and `DrmSurface` creation happen there; see Send-audit in A-increments).
- `SessionPause / SessionActivate` (ack via backchannel), `Shutdown`.

Render thread → main (`calloop::channel::Sender<RenderEvent>`, shared):

- `PresentOutcome { output, outcome }` where `outcome` mirrors today's
  `PresentOutcome` plus payloads: `Presented { present_sync_dup: OwnedFd,
  next_present_estimate }`, `SkippedNoDamage { next_present_estimate }`,
  `FlipPending`, `FlipFailed { err }`. The fd dup feeds
  `note_mirror_render_done` and `register_release_after_submit` on main.
  `next_present_estimate` (from the thread-owned `FrameClock`) lets main
  arm estimated-vblank timers without owning the clock.
- `PauseAck`, `Fault { output, err }`.

There is deliberately **no vblank round-trip**: the DRM event lands on main
(D5), main runs its state-machine/feedback/frame-callback logic directly
off the kernel event (it carries everything needed — crtc, timestamp,
sequence) and forwards `Vblank` to the thread purely for `mark_vblank`
bookkeeping. Command-channel FIFO makes the ack unnecessary: `Vblank` is
sent before any subsequent `SubmitFrame`, so the thread always clears
`frame_pending` / flips `back_index` before the next present arrives.

**Ordering argument (the mirror gate).** All hazards that today rely on
serial execution reduce to per-channel FIFO plus one dispatch-order rule:

- `PresentOutcome(N)` is sent at submit time, milliseconds before frame
  N's flip retires — so it is on main's backchannel long before the vblank
  that triggers lowering N+1. The one rule to uphold: main drains the
  backchannel (a calloop source, dispatched) **before**
  `redraw_queued_outputs` runs (which is already the loop shape: dispatch,
  then drain queued redraws). Then the mirror-gate fd from N is stored
  (`note_mirror_render_done`) before `prepare_mirror_waits` runs for N+1 —
  no gap where a home→GTT copy could overwrite scratch a still-in-flight
  render reads.
- `Vblank(N)` precedes `SubmitFrame(N+1)` on the command channel — the
  thread never sees a present while it still thinks a flip is pending.

**Pre-existing audit (A3) — RESOLVED (2026-07-01), no fix needed:** the
gate (`MirrorCopier::render_done`) has replace semantics, and that is
*sound*: every render that samples a target GPU's scratch is submitted on
that GPU's single graphics queue, so by Vulkan's implicit queue ordering
(§7.2) the latest present fd transitively proves all earlier renders
complete. Preserved by Phase 2 (one thread per queue keeps submission
order) and by increment C **provided** all of a target's scratch-reading
ACE copies stay on one queue (they do — one `transfer_queue` per Device).
Reasoning recorded at `cross_gpu.rs` `note_render_done`.

### 2.4 `LoweredFrame` (the boundary type)

Everything `OutputContext::present` + `render_frame` consume, made
self-contained and `Send` (add a `static_assertions`-style
`fn _assert_send<T: Send>()` check):

- `elements: Vec<ElementDraw>` (raw `vk::ImageView`s + `DecodePush` — plain
  data) **plus** `keepalive: Vec<Arc<dyn Any + Send + Sync>>` holding strong
  refs to every GPU object the views/handles point into (`GpuTex` images,
  `Arc<LocalImage>`, `Arc<SnapshotTexture>`, LUTs if per-frame). Today the
  borrow of `&mut PrismState` keeps them alive; across a channel the frame
  must own them. Destruction stays safe via the existing deferred-destroy
  (`device.retire` gated on submit serials).
- damage-tracker element states / encode metadata (whatever
  `DamageTracker::compute` consumes — it moves with the tracker to the
  thread; the frame carries the per-element commit counters/geometry it
  diffs).
- `encode_push`, `force_full_repaint`, profile flag.
- `wait_semaphores: Vec<vk::Semaphore>` (imported on the target device by
  main during lowering) + ownership so the thread retires them post-submit
  (replaces the call-site `destroy_render_wait_semaphores`).
- `local_copies: Vec<LocalMirrorCopy>` (Phase 1) — in increment C these
  become the ACE-copy job the thread submits before the render.
- `snapshots: Vec<SnapshotCopy>`, screencopy jobs (dmabuf targets +
  completion tokens; the *tokens* are opaque ids — completion fires on main
  when the outcome message returns them).
- cursor plane state for this flip (pos/visible; sprite if dirty).

Presentation-feedback and syncobj-release *harvesting* stays in lowering on
main (it walks Wayland surface trees); the harvested smithay objects stay
in main-side `OutputRedrawState.pending_feedback` exactly as today and fire
on the DRM vblank event. They never enter `LoweredFrame`.

### 2.5 Redraw state machine changes (`redraw.rs`)

One new state. Today `render_one_queued` learns the outcome synchronously;
now it's a message, so `Queued` transitions to:

- `AwaitingOutcome { redraw_needed: bool }` — frame lowered + sent, outcome
  not yet back. `queue_redraw()` during it sets `redraw_needed`.
- On `PresentOutcome`: `Presented → WaitingForVBlank { redraw_needed }`;
  `SkippedNoDamage → WaitingForEstimatedVBlank*` (arm timer from
  `next_present_estimate`); `FlipPending/FlipFailed → Queued` (retry, as
  today).
- `WaitingForVBlank` is cleared by the DRM vblank event on main, exactly as
  today (`on_vblank`'s main-side half) — no thread involvement.

Everything else (`Idle/Queued/WaitingForVBlank/WaitingForEstimatedVBlank*`,
`queue_redraw` entry point, frame-callback sequence) is unchanged.
`redraw_queued_outputs` becomes non-blocking: it lowers + sends every
queued output back-to-back — outputs on different GPUs then render in
true parallel; outputs on one GPU queue in that thread's channel (the
physically required serialization).

### 2.6 Session, hotplug, teardown

- **VT switch:** `PauseSession` → send `SessionPause` to each thread, wait
  for `PauseAck` (bounded wait + log on timeout), then `card.drm.pause()`.
  Resume: `drm.activate()`, `SessionActivate`, re-queue all outputs.
- **Hotplug/config:** connector add → `CreateOutput` (thread builds
  `OutputContext`); remove → `DestroyOutput` + drop the shadow; mode/VRR/
  color changes → `Reconfigure`.
- **Teardown order** (today `main.rs:2196-2230`, needs DRM master):
  `Shutdown` to each thread → thread drops its `OutputContext`s (clearing
  scanout state) → join threads → then drop cards/session. Bounded join +
  abort path if a thread is wedged on a fence.
- **amdgpu atomic-commit ENOMEM ceiling (Vega, `main.rs:2466`):** per-CRTC
  staggering is preserved naturally — one thread per card serializes that
  card's commits.

### 2.7 Cross-cutting audits

- **Send-audit** (compiler-enforced; resolve as hit): `DrmSurface`,
  `gbm::BufferObject`/`GbmDevice` (if `!Send`, create BOs on the render
  thread and guard the shared `GbmDevice` with a `Mutex` — allocation is
  cold-path), `CursorPlane` (mapped BO), `Renderer` internals. Fallbacks
  in order: restructure so the type never crosses; `Mutex`-wrap;
  `unsafe impl Send` with a written justification.
- **`Device` internals:** submit-serial counter → `AtomicU64`; retire list
  → `Mutex`; queue helpers per D3 (`submit_gfx`, `submit_transfer`,
  `wait_idle` — audit *every* `queue_submit`/`queue_wait_idle` call site:
  renderer.rs, cross_gpu.rs, upload.rs, capture.rs, diagnose.rs,
  lut3d.rs).
- **Profiling ring:** `prism-tune` IPC (main) reads per-`Renderer` stats
  that now live on the thread → put the 256-frame ring behind
  `Arc<Mutex<…>>`, main keeps a clone per output (registered at
  `CreateOutput`).
- **Deadlock rule:** a render thread never blocks on main (only channel
  recv + GPU fences); main never blocks on a render thread except
  `PauseAck`/`Shutdown` joins, both with timeouts.
- **Backpressure:** none needed — the state machine caps in-flight frames
  at one per output by construction.

### 2.8 Staged increments (each commit-sized, buildable, tested)

**A — groundwork (single-threaded, zero behavior change):**
1. **A1** `Device` queue locks + submit helpers; route all submit/wait-idle
   sites. Serial → `AtomicU64`, retire list → `Mutex`.
2. **A2** `LoweredFrame` type + **split `render_output_now`** into
   `lower_output_frame(&mut PrismState, id) -> Option<LoweredFrame>` /
   `execute_frame(&mut OutputContext, LoweredFrame) -> PresentOutcome` /
   `process_present_outcome(&mut PrismState, id, outcome)` — still called
   back-to-back synchronously. Assert `LoweredFrame: Send`. This is the
   big refactor and it's fully testable single-threaded.
3. **A3** mirror-gate accumulate audit/fix (see 2.3).
4. **A4** split `on_vblank` into thread-side `mark_vblank` bookkeeping vs
   main-side state machine + callbacks, callable separately.

**B — the split:**
1. **B1** `RenderThread` (spawn per GPU, mpsc loop, backchannel), move
   `OutputContext` ownership in, `OutputShadow` on main, re-point
   `state.outputs` readers. The flip to async happens here.
2. **B2** state machine `AwaitingOutcome` + outcome-driven transitions;
   vblank forwarding; estimated-vblank from `next_present_estimate`.
3. **B3** cursor/screencopy/syncobj-release/feedback flows per 2.4;
   profiling-ring sharing.
4. **B4** session pause/activate, hotplug, config `Reconfigure`, teardown
   ordering.

**C — Phase 1 remainder, on the new structure:** move the GTT→local mirror
copy from the GFX cb to the ACE `transfer_queue` — now naturally a
render-thread-side submit before the render (separate cb on the thread's
transfer pool, `local_done` semaphore into the render's wait list; the
scratch-overwrite gate re-points to ACE-copy-done per §1.4). Then
double-buffer `target_local` (skip-copy must not flip the index, §1.8).
Add a profile span for the copy. *Deliberately re-sequenced after B*: on
the thread structure the ACE submit needs no cross-thread choreography.
**Delete the interim path**: the `local_copies` parameter threaded through
`render_frame`/`present` (added by the GFX-first sub-step e6d9a77) comes
back out — the copy op list rides `LoweredFrame` into the thread's ACE
submit instead. Don't leave both mechanisms alive.

**Do not start B until A is committed and the compositor has been
daily-driven on A** — A carries all the refactor risk with none of the
concurrency, so regressions surface attributably.

### 2.9 Validation plan (for live testing later)

Parity checklist (all currently-working features, on the real 6-output rig):
- 6-output bringup, independent VRR rates (video on one output must not
  change another's flip cadence — verify with `prism-tune profiling` on
  two outputs simultaneously).
- Cursor on every output, cross-output motion, auto-hide.
- VT switch away/back; session lock (swaylock); output power off/on.
- Hotplug: DP unplug/replug; DRM lease (SteamVR headset).
- Mirror windows on both GPUs' outputs (Twitch-on-DP-6 case), rapid-scroll
  tear check (firefox-scroll-blue-bleed class).
- grim screenshot + wf-recorder on outputs of both GPUs; xwayland;
  open/close/resize animations; config hot-reload of `tune` sections.

Perf acceptance:
- The Phase-1 metric: mirror-output decode+deband collapse (§1.7).
- The Phase-2 metric: with a Vega output deliberately over budget (e.g.
  4K high-refresh + heavy content), Navi outputs' frame times and input
  latency stay flat. Before/after with the profiler; the `submit` span on
  main disappears entirely (it lives on the threads now).

### 2.10 Risks

- Widest blast radius is B1's `state.outputs` re-pointing — mechanical but
  everywhere. Mitigate: `OutputShadow` first as a pure refactor (A-side if
  convenient), thread move after.
- Fence-wedged render thread at teardown/VT-switch → bounded waits +
  abort logging, never an unbounded join on the main thread.
- Message-protocol drift vs. reality (e.g. a forgotten `state.outputs`
  reader) — the compiler finds movers; grep + the parity checklist find
  readers.
- Frame-callback throttling regressions under estimated-vblank when a
  thread reports `SkippedNoDamage` — watch idle CPU wakeups after B2.

## Open items

- ~~Confirm radv accepts `vkCmdCopyImage` on the family-1 queue~~ **DONE** —
  see 1.8 (`examples/ace_copy_probe.rs`, PASS both GPUs).
- Negotiated PCIe link speed (1.7) → copy floor. Needs
  `sudo lspci -vv | grep LnkSta`; non-blocking.
- Whether the 4 ACE queues should later host compute-shader decode/encode
  for intra-GPU multi-output overlap (post-Phase-2; the D1 thread already
  owns the ACE queues to build on).
- Cursor-only atomic commits (reposition without a full render pass) —
  cheap on the render thread once B lands; latency win for idle scenes.
