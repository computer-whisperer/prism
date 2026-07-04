# VkDevice-loss recovery

Status: R1 IN PROGRESS · R2 TODO · R3 TODO · R4 TODO

## 1. Motivation

Two observed failure classes permanently poison a VkDevice today:

- **Silent first-CS EINVAL after long idle** (2026-07-03 09:12, 2026-07-04
  09:44): the first command submission on the Navi device after ~7-9h of
  DPMS-off idle is rejected by the kernel with `-22`, dmesg silent. Root
  cause unknown (tracked separately; per-fd/ctx state suspected — a fresh
  `Device::new` opens a new render-node fd + amdgpu context, so recovery is
  plausibly the *correct* fix, not just a mask).
- **Real GPU resets** (2026-06-27: Vega SMU hang, `-133` ECANCELED + kernel
  SMU/ring logs).

RADV marks the VkDevice lost after one rejected submission (sticky). Every
subsequent call returns `VK_ERROR_DEVICE_LOST`. prism has **zero** handling:
errors funnel to log-and-continue arms (`perform_redraw` Err arm
`main.rs:2534`; warn sites in `state.rs` mirror/shm/materialize paths), so
the session half-works forever — outputs on the lost GPU stay dark, new
windows never materialize, journal spams — until the user restarts.

Goal: on device loss, quarantine the GPU, tear down its Vulkan state,
recreate the device, and resume — clients never notice.

## 2. Design decisions

### D1. Recovery unit = one VkDevice (GPU); the instance survives

The shared `VkInstance` is unaffected by device loss. Recovery calls the
existing `Device::new(instance.clone(), Some(drm_dev_id))` bringup path —
same physical-device selection by DRM major/minor, **fresh render-node fd +
fresh amdgpu context**. Nothing client-visible changes: dmabuf feedback
advertises DRM node ids (`build_output_feedback`), not device handles — no
feedback resend, no renegotiation.

### D2. Per-surface GPU textures: generation-validated lazy purge

There is no global registry of surfaces holding a `SurfaceTexSlot`
(windows, layers, subsurfaces, DnD icons, lock surfaces, …). An eager
"walk all surfaces and drop `by_gpu[X]`" teardown is unverifiable — one
missed entry hands a dead-device `vk::ImageView` to the *new* renderer's
command buffer (cross-device handle = UB/EINVAL again). So staleness is
made structurally impossible at read time instead:

- `PrismState.gpu_generation: HashMap<DrmDevId, u64>` — bumped when a
  device is replaced.
- Every `by_gpu` entry records, at insert, the generation of **every device
  it uses**: Native/Shm → 1 dep; Mirror → 2 deps (consumer + home).
  Enforced by construction: `by_gpu` becomes private with an insert method
  that requires deps (compiler-checked; no bare inserts).
- `SurfaceTexture::purge_stale(&mut self, gens)` drops mismatched entries.
  Called at the read/mutate choke points (all hold the slot Mutex):
  the render walk's per-surface texture lookup, `ensure_surface_textures`,
  and `materialize_surface_on_gpu`.
- A purged entry reads as absent → the existing lazy materialization
  (commit path + render-demand safety net) rebuilds it on the new device,
  typically the same frame. Dmabuf content is intact in client BOs; a fresh
  `ShmTexture` already forces a full re-upload from the live `wl_buffer`;
  solid color is trivial.

The old `Arc<Device>` fades: stale entries hold it until purged; every
device-child struct holds `Arc<Device>`, so `Device::drop` (which drains
the deferred-destroy queue unconditionally and destroys the device) runs
strictly last. No wait-for-stragglers step is needed before rebuilding —
`gpus[X]` is replaced immediately. Log `Arc::strong_count` at replace time
for observability.

### D3. Eager teardown only for enumerable state

| State | Where | Action on recovery |
|---|---|---|
| `gpus[X]` | state.rs:486 | replace with new `Device` + bump generation |
| `mirror_copiers[X]` | state.rs:457 | drop + rebuild (single-GPU variant of `build_mirror_copiers`) |
| `OutputContext.renderer` + 2× `ScanoutBuffer.image` for outputs with `gpu_id == X` | output_ctx.rs | rebuild renderer from retained config; re-import GBM BOs (BOs + DRM FBs survive — screen keeps last frame); reload LUT; `mark_for_resume`-style flip reset |
| `dmabuf_sources[*].native_imports[X]` memos | state.rs:451 | eager clear (map is enumerable) |
| `screencopy_pending` / `screencopy_inflight` on X's outputs | state.rs:430/434 | fail to client (existing `fail_pending_screencopy` + an inflight fail-drain) |
| Layout snapshot ghosts (close/resize/interactive-move) on X | prism-layout | purge by device id (`SnapshotTexture` exposes its device); animations degrade to no-crossfade — snapshot-alloc failure is already a tolerated path |
| Cursor planes | output_ctx | **unaffected** (GBM + DRM FB, no Vulkan) |

MirrorCopier::drop waits its fence with `u64::MAX`; on a lost device RADV
returns (with error, ignored) rather than blocking — acceptable. Renderer
teardown uses `wait_device_idle` with ignored results throughout.

### D4. Detection + quarantine

- `RendererError::is_device_lost()` in prism-renderer; an
  anyhow-chain-walking helper in prism-protocols (`state.rs` wraps errors
  in `anyhow::Context`).
- Detection sites: `perform_redraw` Err arm (output → `gpu_id`); the
  state.rs warn sites (mirror copy submit, shm upload, dmabuf materialize).
  All call `PrismState::note_gpu_error(gpu, &err)` → inserts into
  `lost_gpus: HashSet<DrmDevId>`, warns **once**, sets a recovery-pending
  flag.
- Quarantine gates while `gpu_is_lost(X)`: `perform_redraw` parks X's
  outputs (like powered-off); `consumer_gpus_for_surface` filters X (stops
  materialize/mirror/upload attempts and the log spam);
  `prepare_mirror_waits`/`refresh_dmabuf_mirrors` skip X as home or target.
- **Session-lock interaction**: a lost output can't render the lock but
  keeps scanning out its last (desktop!) frame. If the session is locked or
  locks while X is lost, `power_off()` X's outputs (CRTC clear → blank →
  safe to `note_lock_render(true)`, matching the powered-off lock path).

### D5. Rebuild reuses bringup code

- Refactor the renderer+scanout construction inside `OutputContext::new`
  (output_ctx.rs:199) into a helper shared with a new
  `OutputContext::rebuild_render_state(card, device)`.
- Extract the per-output LUT load/upload block (main.rs:1415-1495,
  identity-LUT fallback included) into a function recovery can rerun.
- Orchestrator in `crates/prism/src/recovery.rs`, driven from the main
  dispatch loop's post-dispatch block (main.rs:2053ff).
- `Device::new` failure (GPU genuinely dead, e.g. SMU hang persisting):
  retry with backoff (1s → 30s cap) on a calloop timer; outputs stay
  parked; other GPUs unaffected. Re-attempt also on config reload.
- After rebuild: bump generation, queue redraws on X's outputs. Lazy
  re-materialization repopulates content.

### D6. Sequencing: before the Phase 2 B thread split

Built single-threaded first: teardown/rebuild is far easier to verify
without cross-thread in-flight frames, it protects the A-phase daily-drive
gate now, and it touches the same `OutputContext` seam B1 will move. Under
B, recovery becomes "kill and respawn that GPU's render thread" — increment
B4 (session/hotplug/teardown) absorbs the orchestrator; the
quarantine/generation/purge machinery carries over unchanged.

## 3. In-flight hazards inventory

- Frame slots: `Renderer::drop` → `wait_device_idle` (ignored) + destroy —
  safe on lost device.
- Pending page-flip with an IN_FENCE from the lost device: kernel signals
  all dma_fences on reset (and the EINVAL case loses the device while
  idle), so flips complete; flip-state reset mirrors the session-resume
  semantics (`mark_for_resume`) — careful not to double-flip (EBUSY) if a
  flip is genuinely pending; same risk profile as VT-switch resume.
- A5 keepalive Arcs (`FrameSubmission._keepalive`) hold dead-device objects
  until the frame bookkeeping drops them — they retire into the doomed
  device's deferred queue, drained by `Device::drop`. Safe.
- Deferred-destroy serials never advance on a lost device — irrelevant;
  the drain at `Device::drop` is unconditional.

## 4. Increments

- **R1 — detect + quarantine.** `is_device_lost`, chain helper,
  `lost_gpus`/`gpu_generation`/pending-recovery fields, `note_gpu_error`,
  all quarantine gates incl. lock/power-off interaction, main-loop hook
  (stub orchestrator that logs). Deliverable: a lost GPU parks cleanly —
  frozen outputs, zero spam, healthy GPUs unaffected.
- **R2 — teardown.** Generation deps on `by_gpu` (private map + insert
  API), `purge_stale` at the three choke points, eager clears (memos,
  screencopy, snapshots via device-id purge), single-GPU copier rebuild
  plumbing.
- **R3 — rebuild.** `OutputContext::rebuild_render_state`, LUT-reload
  extraction, `recovery.rs` orchestrator (replace device, bump generation,
  rebuild copier + outputs, reset flip state, queue redraws), retry
  w/ backoff.
- **R4 — injection + validation.** `prism-tune debug recover-gpu <id>`
  IPC command forcing recovery on a healthy device. Validation checklist
  below.

## 5. Validation (R4, live)

With windows on both GPUs, mirrored (cross-GPU) content visible, a video
playing, one animation mid-flight, screencopy loop (`wf-recorder`) running:

1. `prism-tune debug recover-gpu <navi>` → all outputs keep last frame,
   then resume within ~1s; window content intact (dmabuf) or refreshed
   (shm); no client dies; no validation errors; journal shows one
   quarantine + one recovery line.
2. Same on the Vega (4 outputs, mirror consumers).
3. Recover during active screencopy → capture fails gracefully, next
   capture succeeds.
4. Recovery while session locked → lost outputs blank (not desktop),
   lock confirms, unlock renders.
5. Back-to-back recovery of both GPUs.
6. The real thing: overnight DPMS-off idle → morning wake recovers
   automatically (the original bug).

## 6. Non-goals

- Root cause of the wake EINVAL (separate effort: DPMS-on probe submit +
  `drm.debug` capture — recovery must not end that investigation).
- VkInstance loss, GPU hot-unplug (DRM-level, different machinery).
- Preserving shm content the client rewrote during the outage (torn/stale
  until next commit — acceptable).
