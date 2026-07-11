//! Per-GPU render thread — owner of the output execution state
//! (`docs/async-render-rework.md` §2.1 D1/D2).
//!
//! One thread per `Device` (per GPU/card). The thread owns every
//! [`OutputContext`] on its GPU; the main thread keeps only the
//! [`crate::OutputShadow`]s and talks to the thread over an mpsc command
//! channel.
//!
//! **Increment B1 shape — synchronous bridge.** Every operation, the
//! per-frame present included, is a closure shipped to the thread with the
//! caller blocking on a reply channel ([`RenderThread::with_outputs`]).
//! That makes B1 semantically identical to the pre-split code (main is
//! serialized against the render exactly as when it ran the code inline)
//! while moving *ownership* wholesale — the whole point of the increment:
//! all the refactor risk, none of the concurrency. Increment B2 replaces
//! the hot path (submit + vblank + outcome) with fire-and-forget messages
//! and a calloop backchannel, at which point the blocking bridge remains
//! only for cold paths (IPC calibration, power, recovery, session fixups).
//!
//! Rules inherited from the design (§2.3, §2.7):
//! - No Wayland object ever crosses this channel (D4) — closures capture
//!   Vulkan handles, `Arc`s, fds, plain data.
//! - The thread never blocks on main: its only waits are `recv()` on the
//!   command channel and GPU fences inside `present`.
//! - Main may block on the thread only through the bounded
//!   [`RenderThread::with_outputs`] bridge (`BRIDGE_TIMEOUT`); a timeout
//!   surfaces as an error, never a deadlock.

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use prism_renderer::{vk, DrmDevId, EncodePush, FrameProfile, GpuKeepalive, LoweredFrame};
use smithay::utils::{Logical, Size};

use crate::breadcrumb::breadcrumb;
use crate::{OutputContext, PresentOutcome};

/// The GPU-side half of a lowered frame — everything
/// [`OutputContext::present`] consumes, made self-contained and `Send`
/// (`docs/async-render-rework.md` §2.4). Built by the main thread's
/// lowering pass, executed on the owning GPU's render thread. No Wayland
/// objects; the main-side half is `FrameBookkeeping` in `main.rs`.
///
/// Lifetime-bearing: `_keepalive` owns strong refs to every GPU object the
/// frame's raw `vk::ImageView`s / mirror copy ops point into — the drawn
/// surfaces' textures (collected per-surface in the walk) and the layout's
/// snapshot ghosts (`Layout::snapshot_keepalives`) — so a surface dying or
/// an animation ending while the frame is in flight can't stamp the
/// texture's deferred-destroy retirement before the frame's submit.
pub struct FrameSubmission {
    /// RAII only — never read. See above.
    pub _keepalive: Vec<GpuKeepalive>,
    pub frame: LoweredFrame,
    pub view_size: Size<f64, Logical>,
    pub encode_push: EncodePush,
    /// Imported cross-GPU copy-done + dmabuf-acquire semaphores the render
    /// submit waits on; retired via the deferred-destroy queue post-present.
    pub render_waits: Vec<vk::Semaphore>,
    pub local_copies: Vec<prism_renderer::LocalMirrorCopy>,
    pub snapshot_copies: Vec<prism_renderer::SnapshotCopy>,
    pub force_full_decode: bool,
    /// Per-frame phase profile with the lowering-side CPU spans already
    /// filled; `present` fills damage/submit and rings the record.
    pub profile: Option<FrameProfile>,
}

// Compile-time assertion: `FrameSubmission` crosses the render-thread
// channel, so it must stay `Send`. (A trivially-false concrete `where`
// bound is a compile error, which is exactly the check we want.)
fn _assert_frame_submission_send()
where
    FrameSubmission: Send,
{
}

/// Everything the main thread needs back from a present: the outcome plus
/// a snapshot of the thread-owned `FrameClock`'s estimates, so main can
/// arm estimated-vblank timers and stamp feedback target times without
/// owning the clock (§2.3 `next_present_estimate`).
pub struct SubmitReply {
    pub outcome: Result<PresentOutcome>,
    pub clock: FrameClockSnapshot,
}

/// See [`SubmitReply`]. Mirrored into `OutputShadow` on every reply.
#[derive(Clone, Copy)]
pub struct FrameClockSnapshot {
    pub next_presentation_time: Duration,
    pub refresh_interval: Option<Duration>,
}

/// The output map a render thread owns, keyed by `OutputId` (connector
/// name) exactly like the main-side shadow map.
pub type ThreadOutputs = HashMap<String, OutputContext>;

enum RenderCmd {
    /// Run a closure against the thread's outputs. The closure carries its
    /// own typed reply channel when the caller wants a result.
    Run(Box<dyn FnOnce(&mut ThreadOutputs) + Send>),
    /// Drop every output (instrumented per-output) and exit the loop.
    Shutdown,
}

/// Ceiling on a blocking bridge call. Present slot-fence waits bound the
/// thread's longest single operation; a device-lost fence wait returns
/// with an error rather than hanging, so in practice replies arrive in
/// milliseconds. A timeout means the thread is wedged (or the GPU hung
/// without device-loss) — the caller gets an error and the compositor
/// stays alive.
const BRIDGE_TIMEOUT: Duration = Duration::from_secs(30);

/// Handle to one GPU's render thread. Owned by `PrismState`; dropped (and
/// joined) at teardown after an explicit [`Self::shutdown_join`].
pub struct RenderThread {
    pub gpu: DrmDevId,
    cmd: mpsc::Sender<RenderCmd>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl RenderThread {
    /// Spawn the render thread for `gpu`. Outputs are attached afterwards
    /// via [`Self::attach_output`].
    pub fn spawn(gpu: DrmDevId) -> Result<Self> {
        let (cmd, rx) = mpsc::channel::<RenderCmd>();
        let join = std::thread::Builder::new()
            .name(format!("prism-render-{}:{}", gpu.major, gpu.minor))
            .spawn(move || render_thread_loop(gpu, rx))
            .map_err(|e| anyhow!("spawn render thread for {gpu:?}: {e}"))?;
        tracing::info!(?gpu, "render thread spawned");
        Ok(Self {
            gpu,
            cmd,
            join: Some(join),
        })
    }

    /// Move an output's execution context onto the thread. Fire-and-forget
    /// (bringup/attach paths have nothing to do with the result).
    pub fn attach_output(&self, id: String, ctx: OutputContext) -> Result<()> {
        let boxed = Box::new(ctx);
        self.run(move |outputs| {
            if outputs.insert(id.clone(), *boxed).is_some() {
                tracing::warn!(output = %id, "render thread: attach replaced an existing context");
            }
        })
    }

    /// Ship a closure to the thread without waiting for it (command-channel
    /// FIFO still orders it against every other command).
    pub fn run(&self, f: impl FnOnce(&mut ThreadOutputs) + Send + 'static) -> Result<()> {
        self.cmd
            .send(RenderCmd::Run(Box::new(f)))
            .map_err(|_| anyhow!("render thread {:?} is gone", self.gpu))
    }

    /// Run a closure against the thread's outputs and block for its result
    /// (bounded by [`BRIDGE_TIMEOUT`]).
    pub fn with_outputs<R, F>(&self, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut ThreadOutputs) -> R + Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel::<R>(1);
        self.run(move |outputs| {
            // Receiver gone (bridge timed out) ⇒ nowhere to report to.
            let _ = tx.send(f(outputs));
        })?;
        rx.recv_timeout(BRIDGE_TIMEOUT).map_err(|e| {
            anyhow!(
                "render thread {:?} bridge: {e} (thread wedged or panicked)",
                self.gpu
            )
        })
    }

    /// Present one lowered frame on `output_id` and return the outcome +
    /// clock snapshot. Blocking (B1); B2 turns this into a message with the
    /// outcome delivered over the backchannel.
    pub fn submit_frame(&self, output_id: &str, mut sub: FrameSubmission) -> Result<SubmitReply> {
        let id = output_id.to_owned();
        self.with_outputs(move |outputs| {
            let Some(ctx) = outputs.get_mut(&id) else {
                return SubmitReply {
                    outcome: Err(anyhow!("no output bound to id {id}")),
                    clock: FrameClockSnapshot {
                        next_presentation_time: Duration::ZERO,
                        refresh_interval: None,
                    },
                };
            };
            let outcome = ctx.present(
                &sub.frame,
                sub.view_size,
                &sub.encode_push,
                &sub.render_waits,
                &sub.local_copies,
                &sub.snapshot_copies,
                sub.force_full_decode,
                sub.profile.take(),
            );
            // Keepalives + imported wait semaphores die here, on the thread,
            // after the submit that references them — exactly the lifetime
            // the deferred-destroy serials account for. (`render_waits` are
            // retired by the caller via destroy_render_wait_semaphores.)
            let clock = FrameClockSnapshot {
                next_presentation_time: ctx.frame_clock.next_presentation_time(),
                refresh_interval: ctx.frame_clock.refresh_interval(),
            };
            SubmitReply { outcome, clock }
        })
    }

    /// Forward a DRM vblank for `output_id` (§2.3: no round-trip — FIFO
    /// orders it before any later submit). Runs `mark_vblank` bookkeeping.
    pub fn vblank(&self, output_id: &str, presentation_time: Duration) -> Result<()> {
        let id = output_id.to_owned();
        self.run(move |outputs| {
            if let Some(ctx) = outputs.get_mut(&id) {
                ctx.mark_vblank(presentation_time);
            }
        })
    }

    /// Shut the thread down: drop every output (instrumented), then join.
    /// Bounded — a wedged thread is abandoned (detached) with a loud log
    /// rather than hanging teardown.
    pub fn shutdown_join(&mut self, timeout: Duration) {
        if self.cmd.send(RenderCmd::Shutdown).is_err() {
            // Thread already gone (panic); fall through to join.
        }
        let Some(join) = self.join.take() else { return };
        let deadline = std::time::Instant::now() + timeout;
        while !join.is_finished() {
            if std::time::Instant::now() >= deadline {
                tracing::error!(
                    gpu = ?self.gpu,
                    "render thread failed to shut down within {timeout:?}; abandoning it"
                );
                breadcrumb(&format!(
                    "shutdown: render thread {:?} WEDGED, abandoned",
                    self.gpu
                ));
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        if join.join().is_err() {
            tracing::error!(gpu = ?self.gpu, "render thread panicked");
        }
    }
}

impl Drop for RenderThread {
    fn drop(&mut self) {
        // Normal teardown calls shutdown_join explicitly; this covers
        // early-unwind paths so contexts still drop on the owning thread.
        self.shutdown_join(Duration::from_secs(5));
    }
}

fn render_thread_loop(gpu: DrmDevId, rx: mpsc::Receiver<RenderCmd>) {
    let mut outputs: ThreadOutputs = HashMap::new();
    while let Ok(cmd) = rx.recv() {
        match cmd {
            RenderCmd::Run(f) => f(&mut outputs),
            RenderCmd::Shutdown => break,
        }
    }
    // Shutdown (or every sender dropped): release scanout one output at a
    // time so a hang is attributable — mirrors the pre-split instrumented
    // teardown in main.rs.
    for (id, ctx) in outputs.drain() {
        let t = std::time::Instant::now();
        let crtc = ctx.crtc;
        drop(ctx);
        breadcrumb(&format!(
            "shutdown: dropped output {id} (crtc {crtc:?}) in {}ms",
            t.elapsed().as_millis()
        ));
    }
    tracing::info!(?gpu, "render thread exited");
}
