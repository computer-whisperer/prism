//! `wp_linux_drm_syncobj_v1` — explicit GPU fence synchronization for
//! dmabuf surfaces.
//!
//! Modern Mesa (and Firefox via Mesa) opt out of implicit dmabuf
//! synchronization by attaching this protocol's per-surface acquire
//! and release timeline points. With implicit sync, the kernel
//! auto-waits when a Vulkan submit reads a dmabuf the client is
//! still producing; with explicit sync, the kernel doesn't, and
//! it's on us to:
//!
//!   1. **Wait on the client's acquire point** before sampling the
//!      committed buffer in our render pass.
//!   2. **Signal the client's release point** once every render that
//!      sampled the buffer has finished on the GPU. Until that
//!      signal, the client must not write to the buffer; signaling
//!      too early causes torn frames.
//!
//! ## Acquire: commit-deferral via smithay's blocker
//!
//! Smithay's `DrmSyncPoint::generate_blocker()` produces a
//! [`Blocker`] backed by an eventfd that fires when the syncobj
//! point signals (kernel `syncobj_eventfd` ioctl). We register the
//! source with calloop and `add_blocker(surface, blocker)`; smithay
//! holds the commit dispatch until the eventfd fires, after which
//! the buffer is safely readable. No Vulkan-side wait needed — the
//! delayed commit means our render path only ever sees ready
//! buffers.
//!
//! ## Release: Arc-Drop signaling
//!
//! Each commit's release point is wrapped in a [`CommitReleaseTracker`]
//! whose `Drop` impl signals the point. The tracker lives in the
//! surface's [`SurfaceReleaseSlot`] from commit time until the next
//! commit replaces it. Every in-flight render that sampled the
//! buffer clones the tracker's `Arc` into its render-completion
//! callback, so the inner tracker — and thus the signal — survives
//! until both the slot has released its reference and every render
//! callback has fired. The last drop signals.
//!
//! This handles the multi-GPU / multi-output case naturally: if a
//! surface is sampled by two outputs on two different GPUs in the
//! same frame, two callbacks each hold a clone; whichever fires
//! last triggers the signal.
//!
//! Edge case: a stable surface re-rendered across many frames with
//! no new commit keeps the slot's `Arc` reference alive, so release
//! is never signaled — correct, since we're still reading the
//! buffer. Smithay's destruction hook signals release on surface
//! destroy as a backstop.
//!
//! ## Multi-GPU import device
//!
//! Smithay's `DrmSyncobjState::new` takes a single `DrmDeviceFd` as
//! the syncobj import device. We pick the primary GPU's card fd —
//! the same GPU we advertise via dmabuf feedback's `main_device`,
//! so clients open the same kernel namespace for their syncobjs.
//! Both AMD GPUs use the amdgpu driver, so syncobjs imported on the
//! primary card are usable as-is by the kernel scheduler regardless
//! of which GPU's Vulkan queue ends up reading.

use std::sync::Arc;

use smithay::backend::drm::DrmDeviceFd;
use smithay::reexports::calloop::{generic::Generic, Interest, LoopHandle, Mode, PostAction};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::DisplayHandle;
use smithay::reexports::wayland_server::Resource;
use smithay::wayland::compositor::{
    add_blocker, add_pre_commit_hook, with_states, BufferAssignment, SurfaceAttributes,
};
use smithay::wayland::dmabuf::get_dmabuf;
use smithay::wayland::drm_syncobj::{
    supports_syncobj_eventfd, DrmSyncPoint, DrmSyncobjCachedState, DrmSyncobjState,
};

use crate::state::PrismState;

/// Per-commit release-point holder. The point is signaled exactly
/// once when this is dropped — last reference fires. Wrapped in an
/// `Arc` so renders can clone it into their per-submit completion
/// callback; the wayland surface holds another clone for as long as
/// the commit's buffer is the active one.
#[derive(Debug)]
pub struct CommitReleaseTracker {
    point: DrmSyncPoint,
}

impl CommitReleaseTracker {
    pub fn new(point: DrmSyncPoint) -> Arc<Self> {
        Arc::new(Self { point })
    }
}

impl Drop for CommitReleaseTracker {
    fn drop(&mut self) {
        if let Err(e) = self.point.signal() {
            // Already-signaled is the common harmless case (smithay's
            // destruction_hook may have raced us). Other errors mean
            // the kernel rejected the ioctl — log so we know.
            tracing::error!(error = ?e, "drm_syncobj: failed to signal release point");
        }
    }
}

/// Per-surface slot in `SurfaceData::data_map` holding the active
/// commit's release tracker. Replaced on every commit that carries
/// a new release point; the old tracker drops once any in-flight
/// renders release their `Arc` clones.
#[derive(Default)]
pub struct SurfaceReleaseSlot {
    inner: std::sync::Mutex<Option<Arc<CommitReleaseTracker>>>,
}

impl SurfaceReleaseSlot {
    /// Install a new active tracker, returning the previous one for
    /// inspection (callers normally let it drop here, which kicks
    /// off the "wait for in-flight renders, then signal" path).
    pub fn replace(
        &self,
        new: Option<Arc<CommitReleaseTracker>>,
    ) -> Option<Arc<CommitReleaseTracker>> {
        std::mem::replace(&mut *self.inner.lock().unwrap(), new)
    }

    /// Clone the active tracker, if any. Returned `Arc` should be
    /// moved into the render-completion callback so the slot's
    /// drop-on-replace doesn't fire until in-flight reads finish.
    pub fn current(&self) -> Option<Arc<CommitReleaseTracker>> {
        self.inner.lock().unwrap().clone()
    }
}

/// Try to bring up the `wp_linux_drm_syncobj_manager_v1` global
/// using `device_fd` as the syncobj import device. Returns `None`
/// (and logs at INFO) when the kernel doesn't expose the
/// `syncobj_eventfd` ioctl — we can't generate eventfd-backed
/// blockers without it, so advertising the protocol would let
/// clients import timelines we can't actually wait on.
pub fn try_init(dh: &DisplayHandle, device_fd: DrmDeviceFd) -> Option<DrmSyncobjState> {
    if !supports_syncobj_eventfd(&device_fd) {
        tracing::info!(
            "drm_syncobj: kernel lacks syncobj_eventfd support — protocol not advertised"
        );
        return None;
    }
    tracing::info!("drm_syncobj: advertising wp_linux_drm_syncobj_manager_v1");
    Some(DrmSyncobjState::new::<PrismState>(dh, device_fd))
}

/// Install a pre-commit hook on a surface that adds smithay's
/// syncobj acquire blocker when the client has set an acquire
/// point. Called once per surface from [`PrismState::new_surface`];
/// the hook itself fires on every commit but is a fast no-op for
/// non-syncobj commits.
///
/// The hook reads `state.loop_handle` at fire time rather than
/// capturing one — `add_pre_commit_hook` requires `Send + Sync`
/// closures, and `LoopHandle`'s `Rc` internals aren't `Sync`. If
/// insertion of the eventfd source ever fails we log and skip;
/// the commit proceeds without a blocker. The kernel scheduler
/// still honors the dma_fence the syncobj is tracking, so we
/// don't render torn frames — we just lose the commit-deferral
/// granularity (subsequent damage processing may stall a tick).
pub fn install_pre_commit_blocker(surface: &WlSurface) {
    add_pre_commit_hook::<PrismState, _>(surface, |state, _dh, surface| {
        // No-op when drm_syncobj is disabled or loop_handle hasn't
        // been stashed yet (window between PrismState::new and
        // main.rs's set_loop_handle — clients can't have surfaces
        // before the socket is bound, but guard).
        if state.drm_syncobj_state.is_none() {
            return;
        }
        let Some(loop_handle) = state.loop_handle.clone() else {
            return;
        };

        // Pull the pending acquire point + the pending buffer
        // assignment from the surface's double-buffered state.
        // Acquire-without-buffer / buffer-without-acquire cases
        // are validated separately by smithay's own commit_hook
        // (it posts the protocol error) — we only care about the
        // "real commit" path where both are present.
        let (acquire_point, has_dmabuf) = with_states(surface, |states| {
            let acquire = states
                .cached_state
                .get::<DrmSyncobjCachedState>()
                .pending()
                .acquire_point
                .clone();
            let dmabuf = states
                .cached_state
                .get::<SurfaceAttributes>()
                .pending()
                .buffer
                .as_ref()
                .and_then(|b| match b {
                    BufferAssignment::NewBuffer(buf) => get_dmabuf(buf).cloned().ok(),
                    _ => None,
                });
            (acquire, dmabuf.is_some())
        });

        let (Some(acquire), true) = (acquire_point, has_dmabuf) else {
            return;
        };

        match acquire.generate_blocker() {
            Ok((blocker, source)) => {
                let Some(client) = surface.client() else {
                    return;
                };
                let res = loop_handle.insert_source(source, move |_, _, state| {
                    // Re-arm dispatch on the gated client. Smithay's
                    // CompositorClientState::blocker_cleared walks
                    // the client's blockers and resumes commits whose
                    // blockers are all released.
                    let dh = state.display_handle.clone();
                    use smithay::wayland::compositor::CompositorHandler;
                    state
                        .client_compositor_state(&client)
                        .blocker_cleared(state, &dh);
                    Ok(())
                });
                if res.is_ok() {
                    add_blocker(surface, blocker);
                } else {
                    tracing::warn!(
                        "drm_syncobj: insert_source failed for acquire blocker — \
                         commit proceeding without explicit wait"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = ?e,
                    "drm_syncobj: generate_blocker failed — commit proceeding without explicit wait"
                );
            }
        }
    });
}

/// Read the committed release point from a surface's cached state
/// and build a fresh [`CommitReleaseTracker`] for it. Called from
/// our compositor commit handler after smithay has already moved
/// pending→current. Returns `None` if no release point is attached
/// (non-syncobj surface, or first commit before any set_release_point).
pub fn build_tracker_for_current_commit(surface: &WlSurface) -> Option<Arc<CommitReleaseTracker>> {
    with_states(surface, |states| {
        states
            .cached_state
            .get::<DrmSyncobjCachedState>()
            .current()
            .release_point
            .clone()
            .map(CommitReleaseTracker::new)
    })
}

/// Apply a freshly-built tracker (from
/// [`build_tracker_for_current_commit`]) to a surface, replacing
/// whatever was active. The previous tracker drops here — if no
/// in-flight render holds a clone, that fires its release point
/// immediately; otherwise the last clone drop does.
pub fn install_tracker(surface: &WlSurface, tracker: Option<Arc<CommitReleaseTracker>>) {
    with_states(surface, |states| {
        states
            .data_map
            .insert_if_missing_threadsafe(SurfaceReleaseSlot::default);
        let slot = states.data_map.get::<SurfaceReleaseSlot>().unwrap();
        let _previous = slot.replace(tracker);
    });
}

/// Clone the active release tracker for a surface, if any. Used by
/// the render path: every surface we render this frame contributes
/// one clone into the per-submit completion callback (see
/// [`register_release_after_submit`]).
pub fn tracker_for_render(surface: &WlSurface) -> Option<Arc<CommitReleaseTracker>> {
    with_states(surface, |states| {
        states
            .data_map
            .get::<SurfaceReleaseSlot>()
            .and_then(|slot| slot.current())
    })
}

/// Take ownership of a per-submit sync_file FD and the set of
/// release trackers cloned from surfaces sampled during that
/// submit. Register a one-shot calloop source on the FD; when it
/// becomes readable (Vulkan submit done), the callback drops the
/// trackers. Whichever clone is the last to drop signals the
/// matching release point.
///
/// On failure to register (kernel out of fds, etc.), the trackers
/// are dropped immediately — this is incorrect under multi-output
/// sampling (we'd signal release before the OTHER output's GPU
/// finishes) but only happens under resource exhaustion.
pub fn register_release_after_submit(
    loop_handle: &LoopHandle<'static, PrismState>,
    sync_fd: std::os::fd::OwnedFd,
    trackers: Vec<Arc<CommitReleaseTracker>>,
) {
    if trackers.is_empty() {
        // No syncobj-managed surfaces in this frame — drop the
        // sync_fd, nothing to wait on for release purposes.
        return;
    }
    // Generic<OwnedFd, Level> would re-fire forever; Level on a
    // one-shot signal we want to remove ourselves is the correct
    // pattern (Mode::OneShot would un-poll after the first wake
    // but the source still needs explicit removal).
    let source = Generic::new(sync_fd, Interest::READ, Mode::OneShot);
    let mut trackers_holder = Some(trackers);
    let res = loop_handle.insert_source(source, move |_, _, _state| {
        // Dropping the Vec drops every Arc clone we collected.
        // For each surface where this was the last reference, the
        // tracker's Drop signals the release point. Other surfaces
        // are still being sampled elsewhere; their trackers
        // survive until those callbacks fire.
        drop(trackers_holder.take());
        Ok(PostAction::Remove)
    });
    if let Err(e) = res {
        tracing::warn!(
            error = ?e,
            "drm_syncobj: insert_source for release sync_fd failed — \
             release points will fire from the slot drop instead, which \
             may race in-flight GPU reads on other outputs"
        );
        // trackers_holder still owns the trackers; it drops here.
    }
}
