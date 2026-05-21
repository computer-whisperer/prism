//! `PrismState` — the smithay handler-trait carrier.
//!
//! Smithay's protocol dispatch model: one application-owned struct
//! (`PrismState` here) implements every protocol's `*Handler` trait that the
//! compositor wants to participate in, and `delegate_*!` macros wire the
//! protocol message dispatch into those traits.
//!
//! Scope of this scaffolding (task #46):
//!   - wl_compositor (surface lifecycle, basic commits)
//!   - xdg-shell (toplevel windows, configure / map / unmap)
//!   - wl_shm (software-rendered clients)
//!
//! Not yet wired (will come incrementally):
//!   - linux-dmabuf (hardware-rendered clients)
//!   - wl_seat / input
//!   - wl_output (display advertisement)
//!   - presentation-time, viewporter, fractional-scale, …
//!
//! On commit we currently only log; rendering hooks in once #47 (texture
//! import) and #48 (shader pipeline) are wired up.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use prism_renderer::{DrmDevId, vk};
use smithay::backend::allocator::Format as DrmFormat;
use smithay::backend::allocator::dmabuf::Dmabuf as SmithayDmabuf;
use smithay::delegate_compositor;
use smithay::delegate_dmabuf;
use smithay::delegate_output;
use smithay::delegate_seat;
use smithay::delegate_shm;
use smithay::delegate_xdg_shell;
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Scale, Subpixel};
use prism_frame::{DrmFourcc, DrmModifier};
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::backend::{ClientData, ObjectId};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Display, DisplayHandle, Resource};
use smithay::utils::{Serial, Transform};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    BufferAssignment, CompositorClientState, CompositorHandler, CompositorState,
    SurfaceAttributes, get_role, with_states,
};
use smithay::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState, with_buffer_contents};

use crate::client::PrismClient;
use crate::surface_tex::{SurfacePlacementSlot, SurfaceTexSlot, SurfaceTexture};

/// Stable per-output id. Today we key by the connector name (e.g. `"DP-4"`,
/// `"HDMI-A-1"`). amdgpu's connector names are globally unique across cards
/// on this hardware, so this is sufficient as a primary key. If we ever
/// support a backend that reuses connector names per device, switch to
/// `(DrmDevId, connector::Handle)`.
pub type OutputId = String;

/// Field declaration order is load-bearing within this struct (outputs
/// before cards before gpus, since outputs hold strong references to
/// both). But it does NOT solve the libseat/DRM-master lifetime issue:
/// `SeatSession` here is just `Weak<LibSeatSessionImpl>` — the actual
/// `Rc<LibSeatSessionImpl>` (the thing that holds DRM master) lives in
/// `LibSeatSessionNotifier`, which the caller stashed in the calloop
/// event loop. Same shape for `DrmDevice` (Arc held by `DrmDeviceNotifier`).
/// Master is released when those notifiers drop with the event loop.
///
/// **Shutdown rule:** drop `PrismState` BEFORE the event loop in any
/// caller that wants `OutputContext::Drop::surface.clear()` to succeed.
/// See `run_integrated` in `prism/src/main.rs` for the canonical
/// teardown sequence.
pub struct PrismState {
    pub display_handle: DisplayHandle,
    pub compositor: CompositorState,
    pub xdg_shell: XdgShellState,
    pub shm: ShmState,
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: DmabufGlobal,
    /// wl_output + xdg-output-unstable-v1 manager. Holds the global IDs
    /// for the xdg-output manager; per-output `Output` instances live in
    /// `wl_outputs` and carry their own wl_output global IDs.
    pub output_manager: OutputManagerState,
    /// wl_seat state for all advertised seats. We only have one ("seat0")
    /// today and it's advertised with zero capabilities — enough to make
    /// clients (mpv, browsers) that require *some* seat to be present
    /// connect successfully. Real input dispatch (keyboard/pointer)
    /// lands when we wire up libinput.
    pub seat_state: SeatState<PrismState>,
    /// The single seat we advertise. Kept around so we can flip
    /// capabilities on later (when we add keyboard/pointer support).
    pub seat: Seat<PrismState>,

    /// Per-output smithay `Output`, keyed by the same `OutputId`
    /// (connector name) as `outputs`. Populated by [`advertise_output`];
    /// logical positions assigned by [`layout_outputs`]. Drops before
    /// `outputs` so wl_output globals are destroyed while the
    /// `DisplayHandle` is still alive.
    pub wl_outputs: HashMap<OutputId, Output>,

    // ── Client buffer textures ─────────────────────────────────────────────
    // Reference Vulkan devices (via Arc); drop before `gpus` so we don't
    // double-free or hit "device destroyed while images outstanding" paths.
    /// Per-GPU Vulkan import of every dmabuf-backed `wl_buffer`. Outer key
    /// is the wl_buffer object id; inner key is the GPU's `DrmDevId`.
    /// Populated in `dmabuf_imported` (imports on every registered GPU so
    /// any output's render path can sample the buffer without GPU→GPU
    /// copies); dropped in `buffer_destroyed`. Multi-GPU support (#59.3)
    /// makes the inner map non-trivial; today there's typically one entry.
    pub dmabuf_textures:
        HashMap<ObjectId, HashMap<DrmDevId, Arc<prism_renderer::ImportedImage>>>,

    // ── DRM stack — declaration order = drop order = outer to inner ────────
    /// Active outputs across all cards. Each `OutputContext` Drop calls
    /// `surface.clear()`, which needs DRM master. Must drop before
    /// `session` releases libseat (else EACCES on clear).
    pub outputs: HashMap<OutputId, prism_drm::OutputContext>,
    /// Open DRM cards, keyed by their primary-node major/minor. One per
    /// `/dev/dri/cardN` we're driving. Drops after outputs so their
    /// `DrmDevice` is still valid during surface teardown.
    pub cards: HashMap<DrmDevId, prism_drm::DrmCardContext>,
    /// Vulkan devices, keyed by the matching card's primary-node
    /// major/minor (i.e. `Device::physical.drm_primary`). One per physical
    /// GPU we're rendering on. Drops after the renderers that depend on
    /// them (held inside outputs/scanout buffers).
    pub gpus: HashMap<DrmDevId, Arc<prism_renderer::Device>>,
    /// libseat grant (one per process). Holds DRM master across all cards
    /// when the session is active. Dropped LAST so master is still held
    /// while DRM devices and surfaces are torn down. `None` for headless
    /// usage (`prism wayland`).
    pub session: Option<prism_drm::SeatSession>,
}

impl PrismState {
    /// Build a `PrismState`. Three usage shapes:
    ///
    /// - **integrated** (`prism run`): `session: Some(SeatSession)`,
    ///   `gpus: {one or more GPUs}`. Outputs attached after via
    ///   [`attach_card`] + [`attach_output`].
    /// - **wayland-only** (`prism wayland`): `session: None`,
    ///   `gpus: {one GPU}` for dmabuf import validation. No scanout.
    /// - **truly headless** (tracer self-tests): `session: None`,
    ///   `gpus: {}`. dmabuf imports rejected.
    pub fn new(
        display: &Display<PrismState>,
        session: Option<prism_drm::SeatSession>,
        gpus: HashMap<DrmDevId, Arc<prism_renderer::Device>>,
    ) -> Self {
        let dh = display.handle();
        let compositor = CompositorState::new::<PrismState>(&dh);
        let xdg_shell = XdgShellState::new::<PrismState>(&dh);
        // Empty extra-formats list: ARGB8888 and XRGB8888 are mandatory and
        // smithay advertises them implicitly.
        let shm = ShmState::new::<PrismState>(&dh, []);

        // Hardcoded minimal dmabuf format set for now: XRGB8888 / ARGB8888
        // with LINEAR modifier. Both map to vk::Format::B8G8R8A8_UNORM. Tiled
        // modifiers will be added once we query the Vulkan device for
        // VK_EXT_image_drm_format_modifier support.
        let supported_formats = [
            DrmFormat {
                code: DrmFourcc::Xrgb8888,
                modifier: DrmModifier::Linear,
            },
            DrmFormat {
                code: DrmFourcc::Argb8888,
                modifier: DrmModifier::Linear,
            },
        ];
        let mut dmabuf_state = DmabufState::new();
        let dmabuf_global =
            dmabuf_state.create_global::<PrismState>(&dh, supported_formats.iter().copied());

        // wl_output v4 + xdg-output-unstable-v1. Bundling both is the
        // standard smithay pattern; modern clients (mpv, browsers,
        // Firefox) probe xdg_output to get logical-pixel geometry that
        // accounts for fractional scaling.
        let output_manager = OutputManagerState::new_with_xdg_output::<PrismState>(&dh);

        // wl_seat advertised with zero capabilities. Many clients (mpv
        // in particular) refuse to start without *some* wl_seat global;
        // advertising one with no keyboard/pointer/touch makes them
        // connect cleanly. Real input dispatch lands when libinput
        // wiring does.
        let mut seat_state = SeatState::<PrismState>::new();
        let seat = seat_state.new_wl_seat(&dh, "seat0");

        Self {
            display_handle: dh,
            compositor,
            xdg_shell,
            shm,
            dmabuf_state,
            dmabuf_global,
            output_manager,
            seat_state,
            seat,
            session,
            cards: HashMap::new(),
            gpus,
            outputs: HashMap::new(),
            wl_outputs: HashMap::new(),
            dmabuf_textures: HashMap::new(),
        }
    }

    /// Insert an opened card into the state. Returns the previous entry for
    /// that DrmDevId if there was one (shouldn't happen in normal use).
    pub fn attach_card(
        &mut self,
        card: prism_drm::DrmCardContext,
    ) -> Option<prism_drm::DrmCardContext> {
        self.cards.insert(card.drm_dev_id, card)
    }

    /// Insert a built output. Returns the previous entry for that
    /// OutputId if there was one (shouldn't happen in normal use).
    pub fn attach_output(
        &mut self,
        output: prism_drm::OutputContext,
    ) -> Option<prism_drm::OutputContext> {
        let id: OutputId = output.connector_name.clone();
        self.outputs.insert(id, output)
    }

    /// Locate the output bound to a particular CRTC (e.g. for routing a
    /// vblank event from `DrmDeviceNotifier`).
    pub fn output_for_crtc(
        &mut self,
        crtc: smithay::reexports::drm::control::crtc::Handle,
    ) -> Option<&mut prism_drm::OutputContext> {
        self.outputs.values_mut().find(|o| o.crtc == crtc)
    }

    /// Build a smithay `Output` mirroring the given `OutputContext` and
    /// announce it as a wl_output global. Sets mode + scale + transform
    /// from `ctx`; logical position is **not** assigned here — call
    /// [`layout_outputs`] after every output is advertised.
    ///
    /// EDID-derived `PhysicalProperties` (mm size, make/model/serial)
    /// aren't available yet; we advertise placeholder values that won't
    /// confuse clients but won't drive DPI-aware scaling correctly
    /// either. Refine when EDID parsing lands.
    pub fn advertise_output(&mut self, ctx: &prism_drm::OutputContext) -> &Output {
        let mode = OutputMode {
            size: (ctx.extent.width as i32, ctx.extent.height as i32).into(),
            // smithay::output::Mode::refresh is in milli-Hz.
            refresh: (ctx.mode.vrefresh() as i32) * 1000,
        };
        let output = Output::new(
            ctx.connector_name.clone(),
            PhysicalProperties {
                // (0, 0) = "unknown" per wl_output; clients fall back to
                // DPI-agnostic scaling. EDID parsing fills this in later.
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: "prism".to_string(),
                model: ctx.connector_name.clone(),
                serial_number: String::new(),
            },
        );
        // Create the wl_output global. We drop the returned GlobalId
        // because the Output itself carries it for the lifetime of the
        // Output value (smithay destroys the global when the Output
        // drops).
        let _global = output.create_global::<PrismState>(&self.display_handle);
        output.add_mode(mode);
        output.set_preferred(mode);
        output.change_current_state(
            Some(mode),
            Some(Transform::Normal),
            Some(Scale::Integer(1)),
            // location assigned by layout_outputs once all outputs known
            None,
        );
        self.wl_outputs.insert(ctx.connector_name.clone(), output);
        // unwrap: just inserted under this key
        self.wl_outputs.get(&ctx.connector_name).unwrap()
    }

    /// First `OutputId` whose advertised geometry (current_location +
    /// current_mode.size) contains the given logical point, or `None`
    /// if the point lies in no output's region. Iteration order is
    /// HashMap-random; for non-overlapping layouts (today's horizontal
    /// stack) that's fine. With overlapping outputs, becomes a "topmost
    /// contains" rule once we have z-order.
    pub fn output_containing(&self, point: (i32, i32)) -> Option<OutputId> {
        for (id, output) in &self.wl_outputs {
            let loc = output.current_location();
            let Some(mode) = output.current_mode() else {
                continue;
            };
            let x0 = loc.x;
            let y0 = loc.y;
            let x1 = x0.saturating_add(mode.size.w);
            let y1 = y0.saturating_add(mode.size.h);
            if point.0 >= x0 && point.0 < x1 && point.1 >= y0 && point.1 < y1 {
                return Some(id.clone());
            }
        }
        None
    }

    /// Assign logical positions to every advertised output by stacking
    /// them horizontally at `y = 0` in sorted-connector-name order.
    /// Idempotent: safe to call repeatedly as outputs are
    /// added/removed. Real layout (config-driven placement, per-output
    /// scale, rotation) lands with the config layer.
    pub fn layout_outputs(&mut self) {
        let mut names: Vec<OutputId> = self.wl_outputs.keys().cloned().collect();
        names.sort();
        let mut x: i32 = 0;
        for name in names {
            let output = self
                .wl_outputs
                .get(&name)
                .expect("name from wl_outputs.keys()");
            let width = output.current_mode().map(|m| m.size.w).unwrap_or(0);
            output.change_current_state(None, None, None, Some((x, 0).into()));
            tracing::info!(
                connector = %name,
                logical_x = x,
                width,
                "wl_output positioned"
            );
            x = x.saturating_add(width);
        }
    }
}

// ─── wl_output / xdg-output ─────────────────────────────────────────────────

impl OutputHandler for PrismState {
    fn output_bound(
        &mut self,
        output: smithay::output::Output,
        _wl_output: smithay::reexports::wayland_server::protocol::wl_output::WlOutput,
    ) {
        // Logged at info so the integration test can confirm clients
        // see our wl_output advertisements.
        tracing::info!(connector = %output.name(), "client bound wl_output");
    }
}

delegate_output!(PrismState);

// ─── wl_seat ────────────────────────────────────────────────────────────────

impl SeatHandler for PrismState {
    // WlSurface is the focus carrier — smithay provides KeyboardTarget /
    // PointerTarget / TouchTarget impls for it. No input dispatch yet,
    // so these are mostly placeholders; the seat is advertised with
    // zero capabilities and clients can bind but won't receive events.
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }
    // focus_changed / cursor_image / led_state_changed default to no-ops.
}

delegate_seat!(PrismState);

// ─── wl_compositor ──────────────────────────────────────────────────────────

impl CompositorHandler for PrismState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client
            .get_data::<PrismClient>()
            .expect("client missing PrismClient")
            .compositor
    }

    fn commit(&mut self, surface: &WlSurface) {
        let role = get_role(surface);
        tracing::debug!(?role, "wl_surface commit");

        // Process any newly attached buffer: import (dmabuf) or upload (shm)
        // into a SurfaceTexture and stash on the surface's data_map. Done
        // BEFORE the configure dance so the texture is ready by the time
        // the next vblank fires.
        process_surface_buffer(self, surface);

        // For xdg-shell toplevels, send an initial configure on first commit so
        // the client knows it can start drawing. Skipped here once already
        // configured.
        if let Some("xdg_toplevel") = role {
            let needs_initial_configure = with_states(surface, |states| {
                states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .map(|d| {
                        let attrs = d.lock().unwrap();
                        !attrs.initial_configure_sent
                    })
                    .unwrap_or(false)
            });
            if needs_initial_configure {
                if let Some(toplevel) = self
                    .xdg_shell
                    .toplevel_surfaces()
                    .iter()
                    .find(|t| t.wl_surface() == surface)
                    .cloned()
                {
                    toplevel.send_configure();
                    tracing::info!("sent initial configure to xdg_toplevel");
                }
            }
        }
    }
}

impl BufferHandler for PrismState {
    fn buffer_destroyed(&mut self, buffer: &WlBuffer) {
        // Drop our dmabuf import if this buffer was a dmabuf. shm buffers
        // aren't in the map, so this is a no-op for them.
        self.dmabuf_textures.remove(&buffer.id());
    }
}

delegate_compositor!(PrismState);

// ─── wl_shm ─────────────────────────────────────────────────────────────────

impl ShmHandler for PrismState {
    fn shm_state(&self) -> &ShmState {
        &self.shm
    }
}

delegate_shm!(PrismState);

// ─── xdg-shell ──────────────────────────────────────────────────────────────

impl XdgShellHandler for PrismState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        tracing::info!(
            surface_id = ?surface.wl_surface().id(),
            "new xdg_toplevel"
        );
        // Initial configure is sent on first commit via the CompositorHandler
        // hook above (so the client has a chance to set title / app_id first).
    }

    fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {
        tracing::info!("new xdg_popup (not yet handled)");
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: WlSeat, _serial: Serial) {
        // No popup grab handling yet — no input plumbing.
    }

    fn reposition_request(
        &mut self,
        _surface: PopupSurface,
        _positioner: PositionerState,
        _token: u32,
    ) {
    }
}

delegate_xdg_shell!(PrismState);

// ─── linux-dmabuf ───────────────────────────────────────────────────────────

impl DmabufHandler for PrismState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: SmithayDmabuf,
        notifier: ImportNotifier,
    ) {
        // Import the client's dmabuf into a Vulkan image **on every registered
        // GPU**. Any output's render path will sample from the import that
        // matches its GPU's `DrmDevId`. If we have one GPU, this is one
        // import; with multi-GPU (#59.3) it's one per GPU. If any GPU fails
        // to import, we still accept the buffer iff at least one succeeded
        // (the remaining outputs can sample it; the failing GPU will skip
        // surfaces using this buffer until a copy path exists).
        if self.gpus.is_empty() {
            tracing::warn!("dmabuf import: no GPUs registered, rejecting");
            notifier.failed();
            return;
        }

        let w = smithay::backend::allocator::Buffer::size(&dmabuf).w;
        let h = smithay::backend::allocator::Buffer::size(&dmabuf).h;
        let fmt = smithay::backend::allocator::Buffer::format(&dmabuf);

        let mut imports: HashMap<DrmDevId, Arc<prism_renderer::ImportedImage>> = HashMap::new();
        for (&gpu_id, device) in &self.gpus {
            match import_dmabuf(device, &dmabuf) {
                Ok(image) => {
                    imports.insert(gpu_id, Arc::new(image));
                }
                Err(e) => {
                    tracing::warn!(
                        gpu = ?gpu_id,
                        "dmabuf import failed on this GPU: {e:#}"
                    );
                }
            }
        }

        if imports.is_empty() {
            tracing::warn!("dmabuf import rejected: failed on every GPU");
            notifier.failed();
            return;
        }

        match notifier.successful::<PrismState>() {
            Ok(buffer) => {
                let id = buffer.id();
                tracing::info!(
                    ?w,
                    ?h,
                    ?fmt,
                    gpus = imports.len(),
                    "imported client dmabuf as VkImage (cached on {} GPU(s))",
                    imports.len()
                );
                self.dmabuf_textures.insert(id, imports);
            }
            Err(_) => {
                tracing::warn!("dmabuf successful() failed — client may be dead");
            }
        }
    }
}

delegate_dmabuf!(PrismState);

/// Import a client-provided dmabuf as a sampled `VkImage`. Returned image
/// is owned by the caller; the dmabuf fds are dup'd by Vulkan during import
/// so it's safe for the caller's `SmithayDmabuf` to be dropped afterward.
fn import_dmabuf(
    device: &Arc<prism_renderer::Device>,
    src: &SmithayDmabuf,
) -> Result<prism_renderer::ImportedImage> {
    let dmabuf = prism_frame::Dmabuf::from_smithay(src).context("Dmabuf::from_smithay")?;
    let vk_format = vk_format_for(dmabuf.format).with_context(|| {
        format!("no Vulkan format mapping for {:?}", dmabuf.format)
    })?;
    let image = prism_renderer::ImportedImage::import(
        device.clone(),
        &dmabuf,
        vk_format,
        vk::ImageUsageFlags::SAMPLED,
    )
    .context("ImportedImage::import (SAMPLED)")?;
    Ok(image)
}

/// Pull the most-recently-attached buffer out of a surface's pending state,
/// build (or refresh) its `SurfaceTexture`, stash on the surface's data_map.
/// Sends `wl_buffer.release` for shm (we've copied the bytes out) but holds
/// dmabuf-backed buffers — releasing them would let the client overwrite
/// pixels we're still sampling.
///
/// Called from the `CompositorHandler::commit` hook. Smithay aggregates the
/// buffer into `SurfaceAttributes::buffer` and expects us to clear it once
/// we've processed it (otherwise it gets re-handed-back next commit).
fn process_surface_buffer(state: &mut PrismState, surface: &WlSurface) {
    // No GPUs registered → headless mode → accept commit, no texture work.
    if state.gpus.is_empty() {
        return;
    }

    // Take the new buffer assignment (if any) and act on it inside
    // with_states (which holds the SurfaceData lock). We do the texture
    // build under the lock, then compute the surface→output assignment
    // from the resulting extent and dispatch enter/leave.
    //
    // Returns (old_output, new_output) so we can dispatch the wl_surface
    // events OUTSIDE the with_states callback — calling Output::enter/leave
    // re-enters smithay's surface bookkeeping and we don't want to nest
    // that under our own SurfaceData lock.
    let transition: Option<(Option<OutputId>, Option<OutputId>)> = with_states(surface, |states| {
        let mut attrs = states.cached_state.get::<SurfaceAttributes>();
        let current = attrs.current();
        // `take()` so we don't keep re-processing the same buffer across
        // every following commit (a damage-only commit re-runs commit but
        // shouldn't re-upload).
        let Some(assignment) = current.buffer.take() else {
            return None;
        };

        states
            .data_map
            .insert_if_missing_threadsafe(SurfaceTexSlot::default);
        states
            .data_map
            .insert_if_missing_threadsafe(SurfacePlacementSlot::default);
        let slot = states
            .data_map
            .get::<SurfaceTexSlot>()
            .expect("just inserted SurfaceTexSlot");
        let placement_slot = states
            .data_map
            .get::<SurfacePlacementSlot>()
            .expect("just inserted SurfacePlacementSlot");

        match assignment {
            BufferAssignment::Removed => {
                *slot.0.lock().unwrap() = None;
                // Surface is unmapping; leave its current output (if any).
                let mut placement = placement_slot.0.lock().unwrap();
                let old = placement.current_output.take();
                old.map(|o| (Some(o), None))
            }
            BufferAssignment::NewBuffer(buffer) => {
                if let Err(e) = build_surface_texture(state, &buffer, slot) {
                    tracing::warn!("surface buffer import failed: {e:#}");
                    return None;
                }
                // Compute output assignment from the freshly-built
                // texture's extent and the surface's logical position.
                let extent = slot
                    .0
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|t| t.extent())
                    .unwrap_or_default();
                let mut placement = placement_slot.0.lock().unwrap();
                let center = (
                    placement.logical_pos.0 + (extent.width as i32 / 2),
                    placement.logical_pos.1 + (extent.height as i32 / 2),
                );
                let new_output = state.output_containing(center);
                if new_output == placement.current_output {
                    None
                } else {
                    let old_output = placement.current_output.take();
                    placement.current_output = new_output.clone();
                    Some((old_output, new_output))
                }
            }
        }
    });

    // Dispatch enter/leave outside the SurfaceData lock.
    if let Some((old, new)) = transition {
        if let Some(old_id) = old.as_ref() {
            if let Some(output) = state.wl_outputs.get(old_id) {
                output.leave(surface);
                tracing::debug!(
                    surface_id = ?surface.id(),
                    connector = %old_id,
                    "wl_surface.leave dispatched"
                );
            }
        }
        if let Some(new_id) = new.as_ref() {
            if let Some(output) = state.wl_outputs.get(new_id) {
                output.enter(surface);
                tracing::info!(
                    surface_id = ?surface.id(),
                    connector = %new_id,
                    "wl_surface.enter dispatched"
                );
            }
        }
    }
}

fn build_surface_texture(
    state: &PrismState,
    buffer: &WlBuffer,
    slot: &SurfaceTexSlot,
) -> Result<()> {
    // dmabuf path: clone the per-GPU import map directly into the slot —
    // any output's render path can pick its GPU's view at sample time.
    if let Some(per_gpu) = state.dmabuf_textures.get(&buffer.id()) {
        if per_gpu.is_empty() {
            anyhow::bail!("dmabuf buffer has no imports on any GPU");
        }
        *slot.0.lock().unwrap() = Some(SurfaceTexture::Dmabuf {
            by_gpu: per_gpu.clone(),
        });
        // Don't release: client will overwrite the BO we're still sampling.
        return Ok(());
    }

    // shm path: read bytes once, upload to every registered GPU.
    if state.gpus.is_empty() {
        anyhow::bail!("shm upload requires at least one registered GPU");
    }
    let upload_result = with_buffer_contents(buffer, |ptr, len, data| {
        upload_shm_buffer(&state.gpus, slot, ptr, len, data)
    })
    .context("with_buffer_contents")?;
    upload_result?;
    // Bytes have been copied into our staging buffers — safe to let the
    // client reuse this wl_buffer.
    buffer.release();
    Ok(())
}

fn upload_shm_buffer(
    gpus: &HashMap<DrmDevId, Arc<prism_renderer::Device>>,
    slot: &SurfaceTexSlot,
    ptr: *const u8,
    len: usize,
    data: smithay::wayland::shm::BufferData,
) -> Result<()> {
    let vk_format = vk_format_for_shm(data.format).with_context(|| {
        format!("no Vulkan format mapping for wl_shm::{:?}", data.format)
    })?;
    if data.width <= 0 || data.height <= 0 || data.stride <= 0 || data.offset < 0 {
        anyhow::bail!(
            "invalid shm buffer geometry: {}x{} stride={} offset={}",
            data.width,
            data.height,
            data.stride,
            data.offset
        );
    }
    let extent = vk::Extent2D {
        width: data.width as u32,
        height: data.height as u32,
    };
    let offset = data.offset as usize;
    let stride = data.stride as usize;
    let needed = stride * (data.height as usize);
    if offset.saturating_add(needed) > len {
        anyhow::bail!(
            "shm buffer too small: offset={} need={} pool_len={}",
            offset,
            needed,
            len
        );
    }

    // SAFETY: smithay holds the pool mapping for the duration of the
    // with_buffer_contents callback; ptr+offset..+needed is in-bounds per
    // the check above. We immediately copy out into per-GPU staging buffers.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.add(offset), needed) };

    let mut guard = slot.0.lock().unwrap();
    // Reuse existing per-GPU ShmTextures iff extent + format still match
    // AND the registered GPU set hasn't changed. The GPU-set check is
    // cheap insurance for hotplug; at runtime today the set is constant.
    let needs_new = match &*guard {
        Some(SurfaceTexture::Shm { by_gpu }) => {
            by_gpu.len() != gpus.len()
                || by_gpu
                    .iter()
                    .any(|(id, t)| !gpus.contains_key(id)
                        || t.extent() != extent
                        || t.format() != vk_format)
        }
        _ => true,
    };
    if needs_new {
        let mut by_gpu = HashMap::with_capacity(gpus.len());
        for (&gpu_id, device) in gpus {
            let texture = prism_renderer::ShmTexture::new(device.clone(), extent, vk_format)
                .with_context(|| {
                    format!("ShmTexture::new on gpu {}:{}", gpu_id.major, gpu_id.minor)
                })?;
            by_gpu.insert(gpu_id, texture);
        }
        *guard = Some(SurfaceTexture::Shm { by_gpu });
    }
    let Some(SurfaceTexture::Shm { by_gpu }) = guard.as_mut() else {
        unreachable!("just ensured Some(Shm)");
    };
    for (gpu_id, texture) in by_gpu.iter_mut() {
        texture.upload_bytes(bytes, stride).with_context(|| {
            format!(
                "ShmTexture::upload_bytes on gpu {}:{}",
                gpu_id.major, gpu_id.minor
            )
        })?;
    }
    Ok(())
}

fn vk_format_for_shm(fmt: wl_shm::Format) -> Option<vk::Format> {
    Some(match fmt {
        // wl_shm formats are byte-order in memory the same way DRM fourcc
        // is: Argb8888 == B,G,R,A bytes == vk::Format::B8G8R8A8_UNORM.
        wl_shm::Format::Argb8888 | wl_shm::Format::Xrgb8888 => vk::Format::B8G8R8A8_UNORM,
        _ => return None,
    })
}

/// Map a DRM fourcc to the Vulkan format we'd sample it as. Single-planar
/// 32-bit packed formats only for now.
fn vk_format_for(fourcc: DrmFourcc) -> Option<vk::Format> {
    Some(match fourcc {
        // DRM is little-endian-byte-order, so XRGB8888 in memory is B,G,R,X.
        // Vulkan's B8G8R8A8 reads exactly that byte order.
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
        _ => return None,
    })
}

// ─── Per-client data helper ─────────────────────────────────────────────────

/// Build the per-client data smithay attaches to each new client.
pub fn new_client_data() -> Arc<dyn ClientData> {
    Arc::new(PrismClient::default())
}

/// Convenience: create a fresh `Display<PrismState>`. Wrapped so callers
/// don't need a direct `wayland_server` dependency.
pub fn new_display() -> Result<Display<PrismState>> {
    Display::<PrismState>::new().context("wayland_server::Display::new")
}
