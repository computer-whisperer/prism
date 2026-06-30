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

## Phase 2 — per-GPU render threads (sketch, gated on Phase 1)

Required by the 6-output VRR target: navi ∥ vega true parallelism, and a
struggling secondary GPU's slot-fence wait (`renderer.rs:787`) must not stall
the primary GPU or input.

- **Granularity: one render thread per `Device`** (not per output —
  `VkQueue` needs external sync; outputs on one GPU share queues, so one
  owning thread serializes submits for free).
- **Split:** main thread keeps Wayland/smithay/input/layout (inherently
  single-threaded), produces a self-contained `LoweredFrame` per output and
  sends it over a channel; the GPU thread does record → submit → atomic
  flip and runs that GPU's per-output VRR scheduling + DRM vblank/flip
  events; presentation feedback flows back.
- **Enabler:** cross-GPU sync is already fd-based (sync_file / dmabuf),
  thread- and process-agnostic — the home/target handshake works unchanged
  across threads.
- **Hardest part:** GPU-resource lifetime across the boundary. Prefer the
  GPU thread owning *all* of it — main ships client fds + geometry, the GPU
  thread imports/uploads and runs the deferred-destroy gate locally.
- **Constraint:** preserve per-CRTC commit staggering (independent threads
  satisfy it naturally as long as flips aren't batched — the amdgpu
  atomic-commit ENOMEM ceiling on Vega 20, `main.rs:2466`).

Do Phase 2 only if, after Phase 1, measurement still shows cross-GPU
contamination (most likely if a Vega output is on a high-refresh VRR panel
whose budget the GPU can't meet).

## Open items

- ~~Confirm radv accepts `vkCmdCopyImage` on the family-1 queue~~ **DONE** —
  see 1.8 (`examples/ace_copy_probe.rs`, PASS both GPUs).
- Negotiated PCIe link speed (1.7) → copy floor. Needs
  `sudo lspci -vv | grep LnkSta`; non-blocking.
- Whether the 4 ACE queues should later host compute-shader decode/encode for
  intra-GPU multi-output overlap (Phase 2+ consideration).
