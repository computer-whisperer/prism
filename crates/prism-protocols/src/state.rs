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

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result};
use prism_animation::Clock;
use prism_config::Config;
use prism_layout::cursor::{CursorManager, CursorTextureCache, RenderCursor};
use prism_layout::layout::{ActivateWindow, AddWindowTarget, Layout};
use prism_layout::window::{Mapped, ResolvedWindowRules};
use prism_renderer::{DrmDevId, vk};
use smithay::backend::allocator::Format as DrmFormat;
use smithay::backend::allocator::dmabuf::Dmabuf as SmithayDmabuf;
use smithay::delegate_compositor;
use smithay::delegate_content_type;
use smithay::delegate_dmabuf;
use smithay::delegate_drm_syncobj;
use smithay::delegate_fractional_scale;
use smithay::delegate_output;
use smithay::delegate_presentation;
use smithay::delegate_seat;
use smithay::delegate_shm;
use smithay::delegate_single_pixel_buffer;
use smithay::delegate_viewporter;
use smithay::delegate_xdg_activation;
use smithay::delegate_xdg_decoration;
use smithay::delegate_xdg_shell;
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Scale, Subpixel};
use smithay::reexports::calloop::LoopHandle;
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
    add_pre_commit_hook, CompositorClientState, CompositorHandler, CompositorState, get_parent,
    get_role, with_states,
};
use smithay::backend::renderer::utils::{on_commit_buffer_handler, RendererSurfaceStateUserData};
use smithay::desktop::Window;
use smithay::wayland::dmabuf::{
    DmabufFeedback, DmabufFeedbackBuilder, DmabufGlobal, DmabufHandler, DmabufState,
    ImportNotifier, SurfaceDmabufFeedbackState,
};
use smithay::wayland::content_type::ContentTypeState;
use smithay::wayland::drm_syncobj::{DrmSyncobjHandler, DrmSyncobjState};
use smithay::wayland::fractional_scale::{
    FractionalScaleHandler, FractionalScaleManagerState,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::presentation::PresentationState;
use smithay::wayland::selection::data_device::{DataDeviceState, set_data_device_focus};
use smithay::wayland::selection::primary_selection::{
    PrimarySelectionState, set_primary_focus,
};
use smithay::wayland::single_pixel_buffer::SinglePixelBufferState;
use smithay::wayland::viewporter::ViewporterState;
use smithay::wayland::xdg_activation::{
    XdgActivationHandler, XdgActivationState, XdgActivationToken, XdgActivationTokenData,
};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState, with_buffer_contents};

use crate::client::PrismClient;
use crate::input_state::{KeyboardFocus, PointerVisibility};
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
    /// Compositor config (config file + transient IPC overrides). Shared
    /// `Rc<RefCell>` to mirror niri's pattern — the layout, input
    /// dispatch, and IPC handlers all need read access.
    pub config: Rc<RefCell<Config>>,

    /// Animation/event clock. Lazy: caches the monotonic time across a
    /// single event-loop turn and is cleared once per turn. The layout
    /// and every animation subsystem reads from this so a single
    /// `gettime` syscall amortizes across the whole frame.
    pub clock: Clock,

    /// Scrollable tiling layout (workspaces × monitors × tiles). The
    /// generic parameter is `Mapped`, the production `LayoutElement`
    /// impl that wraps an `XdgToplevel` (in the future also XWayland
    /// surfaces). Input dispatch routes activations and resizes here;
    /// the render path reads tile geometry from here.
    pub layout: Layout<Mapped>,

    pub display_handle: DisplayHandle,
    pub compositor: CompositorState,
    pub xdg_shell: XdgShellState,
    /// zxdg-decoration-manager-v1 — lets us negotiate SSD with
    /// clients that support it. We advertise the global; the
    /// [`XdgDecorationHandler`] decides per-toplevel whether to push
    /// `ServerSide` mode (when `config.prefer_no_csd` is set) or
    /// leave it client-controlled.
    pub xdg_decoration: XdgDecorationState,
    pub shm: ShmState,
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: DmabufGlobal,
    /// Format set we advertise as the "main" tranche for *every* dmabuf
    /// feedback object (global default + per-output). Built once at
    /// startup; today this is the broad render-friendly set. Per-output
    /// preference tranches are prepended on top via
    /// [`build_output_feedback`].
    pub dmabuf_main_formats: Vec<DrmFormat>,
    /// Per-output `DmabufFeedback`, keyed by `OutputId`. Built lazily
    /// in [`attach_output`] from the output's `scanout_formats` plus
    /// `dmabuf_main_formats`. Returned from
    /// [`DmabufHandler::new_surface_feedback`] and dispatched again
    /// (via `SurfaceDmabufFeedbackState::set_feedback`) when a surface
    /// changes its bound output. `None`-keyed surfaces (not yet mapped
    /// to an output) fall back to the global default.
    pub output_feedback: HashMap<OutputId, DmabufFeedback>,
    /// wl_output + xdg-output-unstable-v1 manager. Holds the global IDs
    /// for the xdg-output manager; per-output `Output` instances live in
    /// `wl_outputs` and carry their own wl_output global IDs.
    pub output_manager: OutputManagerState,
    /// wl_seat state for all advertised seats. We only have one
    /// ("seat0") today, advertised with keyboard + pointer capabilities
    /// at startup so GDK clients (Firefox/GTK) construct a usable
    /// GdkSeat before they query it — see the bind-site comment in
    /// `PrismState::new`.
    pub seat_state: SeatState<PrismState>,
    /// The single seat we advertise. Kept around so libinput's
    /// device-added handler can attach per-device state and so we can
    /// retract capabilities if the last device of a kind unplugs.
    pub seat: Seat<PrismState>,
    /// `wl_data_device_manager` (v3) — clipboard + DnD. GTK4 ≥ 4.22
    /// hard-requires this; before we advertised it, every GTK app
    /// silently refused to use the wayland display and either fell
    /// back to X11 (then failed) or crashed in obscure ways
    /// (Firefox: child-process abort, Nautilus: clean exit). All
    /// handler trait impls and the delegate live in
    /// [`crate::selection`].
    pub data_device_state: DataDeviceState,
    /// `wp_primary_selection_device_manager_v1` — middle-click paste.
    /// Universally expected on Linux desktops; advertised to all
    /// clients (no per-client filter — see TODO in
    /// [`crate::selection`]).
    pub primary_selection_state: PrimarySelectionState,
    /// Active drag-and-drop cursor icon, set while a DnD grab is
    /// in flight. Currently *stored only* — the render path doesn't
    /// draw it yet, so drags work functionally but without a visual
    /// preview. See [`crate::selection`] for the TODO.
    pub dnd_icon: Option<crate::selection::DndIcon>,
    /// wp_viewporter global. mpv (with `--vo=gpu --gpu-context=wayland`)
    /// hard-requires this to attach destination/source rects to each
    /// surface. We accept the protocol but currently *ignore* the
    /// destination rect at present time — surfaces still render
    /// full-screen on the output they belong to. Honoring the viewport
    /// state lands when we add per-surface dst-rect positioning.
    pub viewporter: ViewporterState,
    /// wp_presentation. Tells clients which clock we use for timestamps
    /// (CLOCK_MONOTONIC) and lets them register per-frame feedback
    /// callbacks that we fire with actual present time + refresh
    /// interval + vblank sequence. Big quality win for video clients
    /// (mpv): without it they fall back to wl_callback.frame timestamp
    /// guesses and end up dropping frames pessimistically.
    pub presentation: PresentationState,
    /// `wp_color_management_v1` identity counter + global handle.
    /// Dispatch for the manager + creator + description + surface
    /// extension interfaces lives in [`crate::color_management`]; this
    /// struct just owns the state the dispatch code references.
    pub color_management: crate::color_management::ColorManagementState,
    /// `wp_fractional_scale_v1`. Smithay handles the protocol; we
    /// own a handle so we can call
    /// [`smithay::wayland::fractional_scale::with_fractional_scale`]
    /// to push `preferred_scale` events when a surface's output
    /// changes. Today we don't drive fractional scale per-surface
    /// (all outputs are advertised at integer scale 1 or 2), so the
    /// state is essentially advertise-only — it kills the GDK-side
    /// "fractional scale not advertised" fallback noise and gives us
    /// the hook to drive real fractional scaling later.
    pub fractional_scale: FractionalScaleManagerState,
    /// `wp_single_pixel_buffer_v1`. Smithay materializes the
    /// `wl_buffer`; clients use it for solid-color fills (window
    /// backgrounds, GTK rects). NOTE: the surface-texture importer
    /// doesn't yet recognize this buffer type — surfaces that attach
    /// one will fail import (logged as a warning) but won't crash.
    /// Wire-up to the renderer is a follow-up.
    pub single_pixel_buffer: SinglePixelBufferState,
    /// `wp_content_type_v1`. Smithay tracks the per-surface content
    /// type hint (game/photo/video). The render path doesn't act on
    /// it yet — when VRR cadence / frame pacing land, they should
    /// read this via
    /// [`smithay::wayland::content_type::ContentTypeSurfaceCachedState`].
    pub content_type: ContentTypeState,
    /// `xdg_activation_v1`. Real handler in [`crate::state`] — see
    /// [`XdgActivationHandler`] impl below; it validates tokens
    /// against the seat's last keyboard/pointer enter serial and
    /// calls [`Layout::activate_window`] on success.
    pub xdg_activation: XdgActivationState,
    /// `wp_linux_drm_syncobj_v1` state, or `None` when the kernel
    /// lacks `syncobj_eventfd` support (we can't generate
    /// eventfd-backed blockers without it, so we don't advertise
    /// the global). Initialized via [`Self::init_drm_syncobj`] after
    /// the primary card is attached — at `PrismState::new` time we
    /// don't yet have a `DrmDeviceFd`. See [`crate::drm_syncobj`].
    pub drm_syncobj_state: Option<DrmSyncobjState>,
    /// DrmDevId of the GPU we advertise as the dmabuf main_device.
    /// Kept around so [`Self::init_drm_syncobj`] can look up the
    /// matching card after [`Self::attach_card`] populates `cards`.
    pub primary_gpu_id: Option<DrmDevId>,
    /// calloop handle used by the drm_syncobj acquire-blocker
    /// pre-commit hook and the per-render release-signal source.
    /// `None` until [`Self::set_loop_handle`] is called from
    /// `main.rs` after the event loop is constructed; the hook
    /// guards on `.as_ref()` so commits before set_loop_handle
    /// just skip the explicit-sync work (no client surfaces can
    /// possibly exist before the wayland socket is bound, which
    /// happens after the event loop is built, so this is a
    /// theoretical window).
    pub loop_handle: Option<LoopHandle<'static, PrismState>>,
    /// `wlr_layer_shell_unstable_v1` server state. MVP — see
    /// [`crate::layer_shell`] for the deliberate scope gaps.
    pub layer_shell_state:
        smithay::wayland::shell::wlr_layer::WlrLayerShellState,
    /// Per-output layer-shell surfaces. Render order within a vec
    /// is creation order; cross-layer Z is enforced at render time
    /// (Background, Bottom, normal-layout, Top, Overlay).
    pub layer_surfaces: HashMap<OutputId, Vec<crate::layer_shell::LayerEntry>>,

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

    /// Per-output redraw state machine + the `wl_callback.frame` /
    /// `wp_presentation_feedback` stashed at submit time, waiting on
    /// the next vblank to fire with the kernel-reported presentation
    /// timestamp. Keyed by `OutputId` to match `outputs` / `wl_outputs`.
    /// See [`crate::redraw`] for the state-machine shape.
    pub output_redraw: HashMap<OutputId, crate::redraw::OutputRedrawState>,

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

    // ── Input state ────────────────────────────────────────────────────────
    /// Where the next keyboard event should land. Defaults to
    /// `Layout { surface: None }` until a window maps.
    pub keyboard_focus: KeyboardFocus,
    /// Cursor visibility tri-state — `Visible` normally, `Hidden`
    /// during auto-hide grace, `Disabled` after touch input. See
    /// [`PointerVisibility`].
    pub pointer_visibility: PointerVisibility,
    /// Keycodes whose press was swallowed by a compositor binding;
    /// release events for these are filtered out so the focused
    /// client never sees a dangling release. Keyed by raw keycode
    /// (same `Keycode` type smithay's `KeyboardKeyEvent::key_code`
    /// returns).
    pub suppressed_keys:
        std::collections::HashSet<smithay::input::keyboard::Keycode>,
    /// libinput devices currently plugged in. Used to recompute seat
    /// capabilities and apply per-device settings on (re)load.
    pub libinput_devices: std::collections::HashSet<input::Device>,
    /// Whether monitors are powered on. Stub: always true until we
    /// wire DPMS / idle-blank. Input dispatch checks this to decide
    /// whether to forward activity or just wake the screens.
    pub monitors_active: bool,
    /// Set by a quit binding (or other shutdown trigger) — the main
    /// loop reads this between dispatches and falls out of the loop.
    /// Avoids dragging `LoopSignal` or `Arc<AtomicBool>` into every
    /// caller of input dispatch.
    pub should_stop: bool,
    /// Current cursor position in global logical coordinates.
    /// Updated by pointer motion / motion-absolute handlers; sampled
    /// by hit-test and (later) cursor-plane setup. Starts at (0, 0).
    pub pointer_pos: smithay::utils::Point<f64, smithay::utils::Logical>,

    /// XCursor theme + sprite source. Resolves [`CursorImageStatus`]
    /// (Hidden / Named / client-Surface) into a renderable sprite
    /// every frame. Initialized in [`Self::new`] with a config-derived
    /// theme name + size.
    pub cursor_manager: CursorManager,
    /// Decoded sprite cache feeding the cursor-plane uploader. Keys
    /// by (icon, scale); values are the per-frame ARGB8888 pixels +
    /// dimensions. Populated lazily on first need.
    pub cursor_texture_cache: CursorTextureCache,
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
    /// Build a `PrismState`.
    ///
    /// `primary_gpu` is the GPU advertised to clients via
    /// `linux-dmabuf-v1 v4`'s default [`DmabufFeedback`] as the
    /// "main_device" — i.e. the render node EGL/Vulkan clients should
    /// open. Pick the one whose outputs you expect to host the most
    /// surfaces (Navi 21 on this hardware: 5 ancillary panels on
    /// Vega 20 vs central + VR + OLED on Navi 21). `None` falls back
    /// to dmabuf v3 (no feedback): clients can still send dmabufs but
    /// have to guess which device to render on, which lands many of
    /// them in software fallback.
    pub fn new(
        display: &Display<PrismState>,
        config: Config,
        session: Option<prism_drm::SeatSession>,
        gpus: HashMap<DrmDevId, Arc<prism_renderer::Device>>,
        primary_gpu: Option<DrmDevId>,
    ) -> Self {
        let dh = display.handle();
        let config = Rc::new(RefCell::new(config));
        let clock = Clock::default();
        let layout = Layout::<Mapped>::new(clock.clone(), &config.borrow());

        let compositor = CompositorState::new::<PrismState>(&dh);
        let xdg_shell = XdgShellState::new::<PrismState>(&dh);
        let xdg_decoration = XdgDecorationState::new::<PrismState>(&dh);
        // Empty extra-formats list: ARGB8888 and XRGB8888 are mandatory and
        // smithay advertises them implicitly.
        // Advertise fp16 shm formats alongside the mandatory
        // XRGB8888/ARGB8888 (smithay adds those implicitly). fp16 is
        // what HDR-aware clients (Spyder calibration patches, future
        // color-managed UI work) need to write PQ-encoded values at
        // useful precision — 8-bit PQ has visible banding from ~30
        // nits up. Advertised unconditionally so clients can probe;
        // ARGB clients are unaffected.
        let shm = ShmState::new::<PrismState>(
            &dh,
            [
                smithay::reexports::wayland_server::protocol::wl_shm::Format::Xbgr16161616f,
                smithay::reexports::wayland_server::protocol::wl_shm::Format::Abgr16161616f,
            ],
        );

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
        // dmabuf v4 + DmabufFeedback when we know the primary GPU's
        // render node. Without that we'd fall back to v3 (no feedback),
        // and clients like mpv that probe the dmabuf-feedback's
        // main_device to pick a render node land in software EGL.
        let dmabuf_global = match primary_gpu.and_then(|id| {
            gpus.get(&id).map(|dev| (id, dev))
        }) {
            Some((id, device)) => {
                // Prefer the render node for client rendering; fall
                // back to the primary node if a render node isn't
                // exposed (shouldn't happen on amdgpu but be defensive).
                let node = device
                    .physical
                    .drm_render
                    .or(device.physical.drm_primary)
                    .unwrap_or(id);
                let main_device = libc::makedev(node.major as u32, node.minor as u32);
                let feedback = DmabufFeedbackBuilder::new(
                    main_device,
                    supported_formats.iter().copied(),
                )
                .build()
                .expect("DmabufFeedbackBuilder::build");
                tracing::info!(
                    "dmabuf v4 advertised with main_device {}:{} ({} formats)",
                    node.major,
                    node.minor,
                    supported_formats.len()
                );
                dmabuf_state.create_global_with_default_feedback::<PrismState>(&dh, &feedback)
            }
            None => {
                tracing::warn!(
                    "no primary GPU registered; falling back to dmabuf v3 — clients may end up in software EGL"
                );
                dmabuf_state.create_global::<PrismState>(&dh, supported_formats.iter().copied())
            }
        };

        // wl_output v4 + xdg-output-unstable-v1. Bundling both is the
        // standard smithay pattern; modern clients (mpv, browsers,
        // Firefox) probe xdg_output to get logical-pixel geometry that
        // accounts for fractional scaling.
        let output_manager = OutputManagerState::new_with_xdg_output::<PrismState>(&dh);

        // wl_seat advertised with keyboard + pointer capabilities up
        // front. Adding the capabilities at bind time (vs. waiting for
        // libinput to enumerate) matters because:
        //   - libseat/libinput device enumeration on this hardware lags
        //     several seconds behind socket-ready, and clients that
        //     bind wl_seat in that window see capabilities=0.
        //   - GDK (Firefox, GTK apps) refuses to construct a GdkSeat
        //     from a capability-less wl_seat: gdk_seat_get_keyboard
        //     starts returning NULL, internal assertions fire, and the
        //     client crashes within ~700ms of connecting.
        // The xkb keymap is loaded from the config that's already on
        // disk. The libinput dispatcher's on_device_added guards with
        // `get_keyboard().is_none()` so real device discovery becomes a
        // no-op for capability bookkeeping (it still attaches per-device
        // settings).
        let mut seat_state = SeatState::<PrismState>::new();
        let mut seat = seat_state.new_wl_seat(&dh, "seat0");
        {
            let cfg = config.borrow();
            let kb = &cfg.input.keyboard;
            if let Err(e) = seat.add_keyboard(
                kb.xkb.to_xkb_config(),
                i32::from(kb.repeat_delay),
                i32::from(kb.repeat_rate),
            ) {
                tracing::warn!("seat: failed to add keyboard at startup: {e:?}");
            }
        }
        seat.add_pointer();

        // wp_viewporter — hard-required by mpv's wayland-egl path so it
        // can set destination rects on its video surface. Smithay
        // handles all the protocol bookkeeping; we just advertise.
        let viewporter = ViewporterState::new::<PrismState>(&dh);

        // wp_presentation_time, advertising CLOCK_MONOTONIC. mpv (and
        // any client doing precise A/V sync) needs this for proper
        // pacing — otherwise it estimates display time from
        // wl_callback.frame timestamps and ends up dropping frames
        // pessimistically.
        let presentation =
            PresentationState::new::<PrismState>(&dh, libc::CLOCK_MONOTONIC as u32);

        // wp_color_management_v1 global. See module doc for scope —
        // accepts parametric image descriptions, surfaces them via
        // `SurfaceColorSlot`, render path consumption is Step 4.
        let color_management = crate::color_management::ColorManagementState::new(&dh);

        // wlr_layer_shell global. MVP scope — see crate::layer_shell.
        let layer_shell_state =
            smithay::wayland::shell::wlr_layer::WlrLayerShellState::new::<PrismState>(&dh);

        // Modern clients (Firefox, GTK4, recent toolkits) probe these
        // globals at startup and either fall back loudly or take
        // degraded paths when missing. We advertise them now so the
        // protocol surface is complete; per-protocol render/scheduling
        // wiring follows incrementally (see field docs on PrismState).
        let fractional_scale = FractionalScaleManagerState::new::<PrismState>(&dh);
        let single_pixel_buffer = SinglePixelBufferState::new::<PrismState>(&dh);
        let content_type = ContentTypeState::new::<PrismState>(&dh);
        let xdg_activation = XdgActivationState::new::<PrismState>(&dh);

        // wl_data_device_manager + wp_primary_selection_device_manager_v1.
        // GTK4 ≥ 4.22 hard-requires the former; without it every GTK
        // client refuses the wayland display. See crate::selection.
        let data_device_state = DataDeviceState::new::<PrismState>(&dh);
        let primary_selection_state = PrimarySelectionState::new::<PrismState>(&dh);

        Self {
            config,
            clock,
            layout,
            display_handle: dh,
            compositor,
            xdg_shell,
            xdg_decoration,
            shm,
            dmabuf_state,
            dmabuf_global,
            dmabuf_main_formats: supported_formats.to_vec(),
            output_feedback: HashMap::new(),
            output_manager,
            seat_state,
            seat,
            data_device_state,
            primary_selection_state,
            dnd_icon: None,
            viewporter,
            presentation,
            color_management,
            fractional_scale,
            single_pixel_buffer,
            content_type,
            xdg_activation,
            drm_syncobj_state: None,
            primary_gpu_id: primary_gpu,
            loop_handle: None,
            layer_shell_state,
            layer_surfaces: HashMap::new(),
            session,
            cards: HashMap::new(),
            gpus,
            outputs: HashMap::new(),
            wl_outputs: HashMap::new(),
            dmabuf_textures: HashMap::new(),
            output_redraw: HashMap::new(),
            keyboard_focus: KeyboardFocus::default(),
            pointer_visibility: PointerVisibility::default(),
            suppressed_keys: std::collections::HashSet::new(),
            libinput_devices: std::collections::HashSet::new(),
            monitors_active: true,
            should_stop: false,
            pointer_pos: smithay::utils::Point::from((0.0, 0.0)),
            cursor_manager: CursorManager::new("default", 24),
            cursor_texture_cache: CursorTextureCache::default(),
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

    /// Stash the calloop loop handle on the state. Needed by the
    /// `drm_syncobj` pre-commit hook (registers eventfd sources for
    /// acquire blockers) and the per-render release-signal source.
    /// Must be called after the event loop is built and before the
    /// dispatch loop starts servicing clients — `main.rs` does this
    /// once at startup.
    pub fn set_loop_handle(&mut self, handle: LoopHandle<'static, PrismState>) {
        self.loop_handle = Some(handle);
    }

    /// Bring up the `wp_linux_drm_syncobj_manager_v1` global using
    /// the primary GPU's card fd as the syncobj import device.
    /// No-op when:
    ///   - no primary GPU was registered at construction time
    ///   - the primary GPU's card isn't yet attached
    ///   - the kernel lacks `syncobj_eventfd` support
    ///
    /// Call from `main.rs` after the `attach_card` loop completes.
    pub fn init_drm_syncobj(&mut self) {
        let Some(primary) = self.primary_gpu_id else {
            tracing::info!("drm_syncobj: no primary GPU set, skipping");
            return;
        };
        let Some(card) = self.cards.get(&primary) else {
            tracing::warn!(
                gpu = ?primary,
                "drm_syncobj: primary GPU card not attached, skipping"
            );
            return;
        };
        let device_fd = card.drm.device_fd().clone();
        self.drm_syncobj_state =
            crate::drm_syncobj::try_init(&self.display_handle, device_fd);
    }

    /// Insert a built output. Returns the previous entry for that
    /// OutputId if there was one (shouldn't happen in normal use).
    ///
    /// Also builds and caches the per-output `DmabufFeedback` so the
    /// wayland-side `wp_linux_dmabuf_v1.get_surface_feedback` path can
    /// advertise direct-scanout-friendly formats to clients whose
    /// surfaces land on this output.
    pub fn attach_output(
        &mut self,
        mut output: prism_drm::OutputContext,
    ) -> Option<prism_drm::OutputContext> {
        let id: OutputId = output.connector_name.clone();
        // Seed the cursor plane (if any) with the default sprite so a
        // subsequent show won't flash. set_visible(false) keeps it off
        // until update_output_cursors flips it.
        if let Some(cursor_plane) = output.cursor.as_mut() {
            if let Err(e) =
                upload_default_cursor(&self.cursor_manager, &self.cursor_texture_cache, cursor_plane)
            {
                tracing::warn!(
                    "cursor seed failed on {}: {e:#} — cursor will not appear on this output",
                    output.connector_name
                );
            }
        }
        // Build the per-output dmabuf feedback before moving `output`.
        // Skipped (and logged) if the output's GPU isn't registered
        // (shouldn't happen — `gpus` is populated before bringup) or
        // if feedback build fails (e.g. shm shortage). Either way the
        // client gets the global default feedback as a fallback.
        if let Some(feedback) = build_output_feedback(
            &output,
            &self.gpus,
            &self.dmabuf_main_formats,
        ) {
            self.output_feedback.insert(id.clone(), feedback);
        }
        // Same step for wp_color_management_v1: derive the output's
        // preferred image description from HDR config + EDID. Used
        // by `wp_color_management_surface_feedback_v1` so clients
        // know "this surface, on this output, should be PQ BT.2020
        // mastered to X nits" (HDR) or sRGB+gamma22 (SDR).
        let preferred = crate::color_management::build_output_preferred(
            &output,
            &self.color_management,
        );
        tracing::info!(
            connector = %output.connector_name,
            identity = preferred.identity,
            tf = ?preferred.tf,
            "color-mgmt: output preferred description registered"
        );
        self.color_management
            .set_output_preferred(id.clone(), preferred);
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
    /// Per-output `scale` is taken from the KDL config (`output "NAME"
    /// { scale 1.5 }`); integer values become `Scale::Integer`, anything
    /// else becomes `Scale::Fractional`. Range 0.1..10 (validated at
    /// parse time by `FloatOrInt<0,10>`).
    ///
    /// `transform` is currently advertised as `Normal` regardless of the
    /// config — the render path does not yet rotate scanout buffers, so
    /// advertising a non-Normal transform would make clients render
    /// pre-rotated buffers that we'd then scan out un-rotated. A warning
    /// is logged when the config asks for one. Render-side rotation lands
    /// with its own task.
    ///
    /// `PhysicalProperties` are populated from the parsed EDID
    /// (`OutputContext.edid`): make/model/serial drive both
    /// `wl_output` advertisement and config matching by "make model
    /// serial"; physical mm size lets DPI-aware clients pick correct
    /// font sizes. Fields the panel didn't advertise fall back to the
    /// strings smithay treats as "unknown" ("Unknown" / empty / 0×0).
    pub fn advertise_output(&mut self, ctx: &prism_drm::OutputContext) -> &Output {
        let mode = OutputMode {
            size: (ctx.extent.width as i32, ctx.extent.height as i32).into(),
            // smithay::output::Mode::refresh is in milli-Hz.
            refresh: (ctx.mode.vrefresh() as i32) * 1000,
        };
        // OutputContext carries EDID directly — use it so EDID-keyed
        // `output "Make Model Serial"` blocks resolve here. Without this
        // the resolution falls back to defaults (scale=1, no rotation)
        // for any EDID-keyed config.
        let scale_xfrm_name = prism_config::output::OutputName {
            connector: ctx.connector_name.clone(),
            make: ctx.edid.make.clone(),
            model: ctx.edid.model.clone(),
            serial: ctx.edid.serial.clone(),
        };
        let (scale, transform) = self.resolve_output_scale_transform(&scale_xfrm_name);
        let size_mm = ctx
            .edid
            .size_mm
            .map(|(w, h)| (w as i32, h as i32).into())
            .unwrap_or_else(|| (0, 0).into());
        let make = ctx.edid.make.clone().unwrap_or_else(|| "Unknown".to_owned());
        let model = ctx
            .edid
            .model
            .clone()
            .unwrap_or_else(|| ctx.connector_name.clone());
        let serial_number = ctx.edid.serial.clone().unwrap_or_default();
        let output = Output::new(
            ctx.connector_name.clone(),
            PhysicalProperties {
                size: size_mm,
                subpixel: Subpixel::Unknown,
                make,
                model,
                serial_number,
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
            Some(transform),
            Some(scale),
            // location assigned by layout_outputs once all outputs known
            None,
        );
        tracing::info!(
            connector = %ctx.connector_name,
            mode_w = mode.size.w,
            mode_h = mode.size.h,
            scale = scale.fractional_scale(),
            "wl_output advertised"
        );
        // Attach the OutputName user data the layout uses to track
        // outputs across disconnects (workspaces remember which
        // output they originated on by name). Populated from EDID so
        // `OutputName::matches` can now match either the kernel
        // connector name OR a `"Make Model Serial"` config target.
        output
            .user_data()
            .insert_if_missing(|| prism_config::OutputName {
                connector: ctx.connector_name.clone(),
                make: ctx.edid.make.clone(),
                model: ctx.edid.model.clone(),
                serial: ctx.edid.serial.clone(),
            });
        // Inform the layout. This creates a Monitor entry, splices in any
        // workspaces that named this output via `original_output`, and
        // (if this is the first output) hosts the no-output workspace
        // backlog. `None` layout_config = use defaults; per-output config
        // lookup arrives once we wire `config.outputs` indexing.
        self.layout.add_output(output.clone(), None);
        self.wl_outputs.insert(ctx.connector_name.clone(), output);
        // unwrap: just inserted under this key
        self.wl_outputs.get(&ctx.connector_name).unwrap()
    }

    /// First `OutputId` whose advertised geometry (current_location +
    /// current_mode.size, scaled to logical units) contains the given
    /// logical point, or `None` if the point lies in no output's region.
    /// Iteration order is HashMap-random; for non-overlapping layouts
    /// (today's horizontal stack) that's fine. With overlapping outputs,
    /// becomes a "topmost contains" rule once we have z-order.
    pub fn output_containing(&self, point: (i32, i32)) -> Option<OutputId> {
        for (id, output) in &self.wl_outputs {
            let loc = output.current_location();
            let Some((lw, lh)) = output_logical_size(output) else {
                continue;
            };
            let x0 = loc.x;
            let y0 = loc.y;
            let x1 = x0.saturating_add(lw);
            let y1 = y0.saturating_add(lh);
            if point.0 >= x0 && point.0 < x1 && point.1 >= y0 && point.1 < y1 {
                return Some(id.clone());
            }
        }
        None
    }

    /// The smithay `Output` whose connector is the layout's currently
    /// active monitor (i.e. the one carrying the focus ring). `None` if
    /// the layout has no active output.
    pub fn active_output(&self) -> Option<Output> {
        self.layout.active_output().cloned()
    }

    /// Output to the left of the active one — i.e. the nearest other
    /// output whose center is to the left of the active output's center
    /// and whose vertical extent overlaps the active output's vertical
    /// extent. Ported from niri's `output_left_of` in `niri.rs:3465`.
    /// `None` if no such neighbor exists.
    pub fn output_left(&self) -> Option<Output> {
        let cur = self.active_output()?;
        self.neighbor_in_direction(&cur, Direction::Left)
    }

    /// Output to the right of the active one. See [`output_left`].
    pub fn output_right(&self) -> Option<Output> {
        let cur = self.active_output()?;
        self.neighbor_in_direction(&cur, Direction::Right)
    }

    /// Output above the active one. See [`output_left`].
    pub fn output_up(&self) -> Option<Output> {
        let cur = self.active_output()?;
        self.neighbor_in_direction(&cur, Direction::Up)
    }

    /// Output below the active one. See [`output_left`].
    pub fn output_down(&self) -> Option<Output> {
        let cur = self.active_output()?;
        self.neighbor_in_direction(&cur, Direction::Down)
    }

    /// Previous output in sorted-connector-name order, wrapping at the
    /// front (so calling Previous from the leftmost returns the rightmost).
    /// `None` if there's only one output.
    pub fn output_previous(&self) -> Option<Output> {
        let cur = self.active_output()?;
        self.cyclic_neighbor(&cur, /* forward */ false)
    }

    /// Next output in sorted-connector-name order, wrapping at the end.
    /// `None` if there's only one output.
    pub fn output_next(&self) -> Option<Output> {
        let cur = self.active_output()?;
        self.cyclic_neighbor(&cur, /* forward */ true)
    }

    fn neighbor_in_direction(&self, current: &Output, dir: Direction) -> Option<Output> {
        // Build (output, logical_rect) for everyone. Skip outputs we
        // can't measure (no mode).
        let mut all: Vec<(&Output, i32, i32, i32, i32)> = Vec::new();
        for o in self.wl_outputs.values() {
            let loc = o.current_location();
            let (lw, lh) = output_logical_size(o)?;
            all.push((o, loc.x, loc.y, lw, lh));
        }
        let cur = all.iter().find(|(o, ..)| *o == current).copied()?;
        let cur_cx = cur.1 + cur.3 / 2;
        let cur_cy = cur.2 + cur.4 / 2;

        // "Extended" rect mirroring niri: same height (for left/right)
        // or same width (for up/down), stretched to the screen-edge so
        // we pick up any output that overlaps the relevant axis stripe.
        all.iter()
            .filter(|(o, ..)| *o != current)
            .filter_map(|&(o, x, y, w, h)| {
                let cx = x + w / 2;
                let cy = y + h / 2;
                match dir {
                    Direction::Left => (cx < cur_cx
                        && overlaps_y(cur.2, cur.4, y, h))
                        .then(|| (o, cur_cx - cx)),
                    Direction::Right => (cx > cur_cx
                        && overlaps_y(cur.2, cur.4, y, h))
                        .then(|| (o, cx - cur_cx)),
                    Direction::Up => (cy < cur_cy
                        && overlaps_x(cur.1, cur.3, x, w))
                        .then(|| (o, cur_cy - cy)),
                    Direction::Down => (cy > cur_cy
                        && overlaps_x(cur.1, cur.3, x, w))
                        .then(|| (o, cy - cur_cy)),
                }
            })
            .min_by_key(|(_, d)| *d)
            .map(|(o, _)| o.clone())
    }

    fn cyclic_neighbor(&self, current: &Output, forward: bool) -> Option<Output> {
        let mut sorted: Vec<&Output> = self.wl_outputs.values().collect();
        sorted.sort_by(|a, b| {
            let an = a
                .user_data()
                .get::<prism_config::OutputName>()
                .map(|n| n.connector.clone())
                .unwrap_or_default();
            let bn = b
                .user_data()
                .get::<prism_config::OutputName>()
                .map(|n| n.connector.clone())
                .unwrap_or_default();
            an.cmp(&bn)
        });
        if sorted.len() < 2 {
            return None;
        }
        let i = sorted.iter().position(|o| *o == current)?;
        let next = if forward {
            (i + 1) % sorted.len()
        } else {
            (i + sorted.len() - 1) % sorted.len()
        };
        Some(sorted[next].clone())
    }

    /// Look up the config-specified scale + transform for a connector.
    /// Falls back to `(Scale::Integer(1), Transform::Normal)` when there's
    /// no matching `output "..."` block. Transform != Normal logs a
    /// warning and is downgraded to Normal — see [`advertise_output`].
    fn resolve_output_scale_transform(
        &self,
        output_name: &prism_config::output::OutputName,
    ) -> (Scale, Transform) {
        let cfg = self.config.borrow();
        let Some(out) = find_output_cfg(output_name, &cfg.outputs.0) else {
            return (Scale::Integer(1), Transform::Normal);
        };
        let scale = match out.scale {
            None => Scale::Integer(1),
            Some(s) => {
                let v = s.0;
                if v == v.trunc() && v >= 1.0 {
                    Scale::Integer(v as i32)
                } else {
                    Scale::Fractional(v)
                }
            }
        };
        let cfg_transform = out.transform;
        if !matches!(cfg_transform, prism_ipc::Transform::Normal) {
            tracing::warn!(
                connector = %output_name.connector,
                transform = ?cfg_transform,
                "output `transform` configured but render path does not yet rotate; \
                 advertising Normal — config ignored"
            );
        }
        (scale, Transform::Normal)
    }

    /// Assign logical positions to every advertised output. Outputs with
    /// an explicit `position x=… y=…` in the KDL config get that exact
    /// location; unpositioned outputs are stacked horizontally at `y=0`
    /// starting from the right edge of the rightmost positioned output
    /// (or `x=0` if none are positioned), in sorted-connector-name order
    /// for stable assignment across runs.
    ///
    /// Logs (warns) if any pair of advertised outputs overlap, but does
    /// not refuse to set them — overlapping outputs may legitimately be
    /// used for mirroring or other intentional cases. The user is the
    /// authority.
    ///
    /// Idempotent: safe to call repeatedly as outputs are added/removed.
    pub fn layout_outputs(&mut self) {
        // Snapshot config so we don't hold a borrow while we mutate
        // outputs via change_current_state (which doesn't touch
        // self.config, but cleaner this way).
        let positions: HashMap<OutputId, Option<prism_config::output::Position>> = {
            let cfg = self.config.borrow();
            // Iterate values (smithay Outputs) so we can pull EDID
            // make/model/serial out of physical_properties and build an
            // OutputName for the matcher — purely connector-keyed lookup
            // misses EDID-keyed `output "Make Model Serial"` blocks.
            self.wl_outputs
                .iter()
                .map(|(name, output)| {
                    let output_name = output_name_from_smithay(name, output);
                    let pos = find_output_cfg(&output_name, &cfg.outputs.0)
                        .and_then(|o| o.position);
                    (name.clone(), pos)
                })
                .collect()
        };

        // First pass: positioned outputs go where the user asked. Track the
        // rightmost edge so the fallback stack picks up from there.
        let mut rightmost: i32 = 0;
        let mut positioned: Vec<OutputId> = Vec::new();
        for (name, pos) in &positions {
            if let Some(p) = pos {
                let output = self.wl_outputs.get(name).expect("from positions iter");
                output.change_current_state(None, None, None, Some((p.x, p.y).into()));
                if let Some((lw, _lh)) = output_logical_size(output) {
                    rightmost = rightmost.max(p.x.saturating_add(lw));
                }
                tracing::info!(
                    connector = %name,
                    logical_x = p.x,
                    logical_y = p.y,
                    "wl_output positioned (from config)"
                );
                positioned.push(name.clone());
            }
        }

        // Second pass: stack remaining outputs to the right of the
        // positioned region, in sorted-connector-name order.
        let mut remaining: Vec<OutputId> = positions
            .keys()
            .filter(|n| !positioned.contains(n))
            .cloned()
            .collect();
        remaining.sort();
        let mut x = rightmost;
        for name in remaining {
            let output = self.wl_outputs.get(&name).expect("from positions iter");
            let (lw, _) = output_logical_size(output).unwrap_or((0, 0));
            output.change_current_state(None, None, None, Some((x, 0).into()));
            tracing::info!(
                connector = %name,
                logical_x = x,
                width = lw,
                "wl_output positioned (auto-stack)"
            );
            x = x.saturating_add(lw);
        }

        // Overlap detection. Quadratic in outputs (6 today, fine).
        let rects: Vec<(OutputId, i32, i32, i32, i32)> = self
            .wl_outputs
            .iter()
            .filter_map(|(name, out)| {
                let loc = out.current_location();
                let (lw, lh) = output_logical_size(out)?;
                Some((name.clone(), loc.x, loc.y, lw, lh))
            })
            .collect();
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                let (a, ax, ay, aw, ah) = &rects[i];
                let (b, bx, by, bw, bh) = &rects[j];
                let overlap_x = *ax < bx.saturating_add(*bw) && *bx < ax.saturating_add(*aw);
                let overlap_y = *ay < by.saturating_add(*bh) && *by < ay.saturating_add(*ah);
                if overlap_x && overlap_y {
                    tracing::warn!(
                        a = %a, b = %b,
                        "wl_output regions overlap; cursor routing + window placement may behave \
                         oddly. Check `output \"…\" {{ position x=… y=… }}` blocks in the config."
                    );
                }
            }
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
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&WlSurface>) {
        // Hand clipboard + primary selection ownership to whatever
        // client owns the keyboard focus. Without this, data offers
        // are never dispatched and paste targets see an empty
        // clipboard. The lookup-then-call pattern matches niri's.
        let dh = &self.display_handle;
        let client = focused.and_then(|s| dh.get_client(s.id()).ok());
        set_data_device_focus(dh, seat, client.clone());
        set_primary_focus(dh, seat, client);
    }
    // cursor_image / led_state_changed default to no-ops.
}

delegate_seat!(PrismState);

// ─── wp_viewporter ──────────────────────────────────────────────────────────

// No handler trait required — smithay stores per-surface viewport
// state in SurfaceData::cached_state; we'd read it via with_states +
// ViewportCachedState if/when we honor it in the render path.
delegate_viewporter!(PrismState);

// ─── wp_presentation_time ───────────────────────────────────────────────────

delegate_presentation!(PrismState);

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

    fn new_surface(&mut self, surface: &WlSurface) {
        // Install the drm_syncobj acquire-blocker pre-commit hook.
        // The hook itself is a fast no-op for non-syncobj surfaces
        // (it checks pending acquire_point + pending dmabuf and
        // returns early when either is absent), so installing on
        // every surface is fine. The hook also self-guards on
        // drm_syncobj being enabled + loop_handle being set,
        // reading both from `state` at fire time.
        crate::drm_syncobj::install_pre_commit_blocker(surface);
    }

    fn commit(&mut self, surface: &WlSurface) {
        let role = get_role(surface);
        tracing::debug!(?role, "wl_surface commit");

        // drm_syncobj release tracking: if this commit carries a
        // release point, wrap it in a CommitReleaseTracker and
        // install on the surface (replacing any previous one). The
        // old tracker's Arc drops here; if no in-flight render holds
        // a clone, its Drop signals the old release point
        // immediately, otherwise the last clone drop does. Surfaces
        // not using drm_syncobj produce None — no-op.
        let new_tracker = crate::drm_syncobj::build_tracker_for_current_commit(surface);
        crate::drm_syncobj::install_tracker(surface, new_tracker);

        // Populate smithay's `RendererSurfaceState` for every surface in
        // the tree under `surface`. This is what computes the
        // `SurfaceView` (offset / src / dst) the render walk reads in
        // `Mapped::render_normal`, and what tracks buffer dimensions /
        // scale / transform / viewport on our behalf. Niri calls this
        // at the top of its commit handler for the same reason.
        //
        // CAUTION: this consumes the `BufferAssignment` out of
        // `SurfaceAttributes::current`, so `process_surface_buffer`
        // below must read the buffer from `RendererSurfaceState`
        // instead of `cached_state` (already updated to do so).
        on_commit_buffer_handler::<PrismState>(surface);

        // Apply any pending wp_color_management_surface_v1 state
        // (set/unset_image_description is double-buffered per the
        // spec). Cheap no-op for surfaces without an image
        // description attachment. Done before the buffer-import path
        // so future Step-4 work that picks a decode shader per
        // surface can read the committed description.
        with_states(surface, |states| {
            crate::color_management::SurfaceColorSlot::commit_pending(states);
        });

        // Process the buffer: import (dmabuf) or upload (shm) into our
        // Vulkan-side SurfaceTexture and stash it on the surface's
        // data_map for the render path. Reads the buffer from the
        // `RendererSurfaceState` populated above.
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
            } else if self.layout.find_window_and_output(surface).is_none() {
                // Already configured but not yet in the layout. We map
                // on the first commit that successfully attached a
                // buffer. The signal we read is the SurfaceTexSlot:
                // process_surface_buffer (called above) consumes any
                // BufferAssignment::NewBuffer out of cached_state and
                // populates this slot with a SurfaceTexture on success.
                // So if the slot is now Some, the client has produced
                // its first renderable frame and is ready to be mapped.
                //
                // Niri does this via an explicit `unmapped_windows`
                // HashMap that tracks the pre-buffer state; we don't
                // need that since we can read map readiness off the
                // texture slot directly.
                let has_texture = with_states(surface, |states| {
                    states
                        .data_map
                        .get::<SurfaceTexSlot>()
                        .map(|s| s.0.lock().unwrap().is_some())
                        .unwrap_or(false)
                });
                if has_texture {
                    if let Some(toplevel) = self
                        .xdg_shell
                        .toplevel_surfaces()
                        .iter()
                        .find(|t| t.wl_surface() == surface)
                        .cloned()
                    {
                        let window = Window::new_wayland_window(toplevel);
                        // Update the window's cached bbox from the
                        // committed surface tree. Without this,
                        // `Window::geometry()` returns an empty rect
                        // (the bbox is initialised to zero), so
                        // `tile.size = window.geometry().size` is
                        // (0,0) — and `Column::width()` (which is
                        // `max(tile.size.w)`) hands the layout a
                        // zero-width column. Every column then sits
                        // at x = sum of zeros + gaps, producing the
                        // "stacked tiles, 16-px-offset-per-window"
                        // visual.
                        window.on_commit();
                        // Pre-commit hook is a no-op for now; niri
                        // uses it for dmabuf-readiness blockers + the
                        // post-commit transaction queue. We don't
                        // have those subsystems yet but Mapped::new
                        // requires a HookId so register a no-op hook
                        // to get one. The hook does fire on every
                        // commit, so keep it cheap.
                        let hook = add_pre_commit_hook::<PrismState, _>(
                            surface,
                            |_state, _dh, _surface| {},
                        );
                        let (mapped, default_column_width) = {
                            let config = self.config.borrow();
                            let m = Mapped::new(
                                window,
                                ResolvedWindowRules::default(),
                                hook,
                                &config,
                            );
                            // Without an explicit per-window-rule width,
                            // fall back to the configured default. niri
                            // resolves this via
                            // `ws.resolve_default_width(rules.default_width, false)`
                            // which collapses to `options.layout.default_column_width`
                            // when no rule overrides. Skipping this is
                            // what makes new windows arrive at width 0
                            // — `resolve_scrolling_width` then falls back
                            // to `Fixed(window.size().w)` which is 0 for
                            // a just-mapped surface.
                            let w = config.layout.default_column_width;
                            (m, w)
                        };
                        let id = mapped.id().clone();
                        // Place the new window on the output that
                        // currently hosts the pointer (rather than
                        // always falling back to the layout's active
                        // monitor, which today is just whichever
                        // output got added first — DP-4 in the
                        // current hardware-test setup). niri uses
                        // its `focus_follows_mouse` infra plus the
                        // last-active monitor to make this choice;
                        // we approximate by reading
                        // `state.pointer_pos` directly. When focus
                        // tracking lands the `Auto` path will be
                        // sufficient on its own.
                        let pointer_output_id = self.output_containing((
                            self.pointer_pos.x as i32,
                            self.pointer_pos.y as i32,
                        ));
                        let pointer_output = pointer_output_id
                            .as_ref()
                            .and_then(|id| self.wl_outputs.get(id))
                            .cloned();
                        let target = match pointer_output.as_ref() {
                            Some(out) => AddWindowTarget::Output(out),
                            None => AddWindowTarget::Auto,
                        };
                        let output = self.layout.add_window(
                            mapped,
                            target,
                            default_column_width,
                            None,
                            false,
                            false,
                            ActivateWindow::Smart,
                        );
                        // Make the new window's monitor the active
                        // one so its tile's focus ring renders with
                        // active-color, not inactive-color. Without
                        // this `active_monitor_idx` stays pinned to
                        // monitor 0 (DP-4 in connector-name sort
                        // order) and only DP-4 windows ever look
                        // focused. niri does this from its input
                        // handlers via `layout.focus_output(&output)`;
                        // for the MVP we do it at add_window time.
                        let output_for_focus = output.cloned();
                        let output_name = output_for_focus.as_ref().map(|o| o.name());
                        if let Some(out) = output_for_focus {
                            self.layout.focus_output(&out);
                        }
                        tracing::info!(
                            ?id,
                            output = ?output_name,
                            "mapped xdg_toplevel into layout"
                        );
                    }
                }
            } else if let Some((mapped, _)) =
                self.layout.find_window_and_output(surface)
            {
                // Re-commit on an already-mapped window.
                //
                // First refresh the smithay Window's cached bbox from
                // the newly-committed surface tree. Without this,
                // `Window::geometry()` returns the bbox at the time
                // of the *previous* commit (or empty on the first),
                // so all the downstream size readers — including
                // `Tile::tile_size()` / `Column::width()` — see
                // stale dimensions. Mirrors niri/src/handlers/compositor.rs:90.
                let window = mapped.window.clone();
                window.on_commit();

                // Then forward through to the layout so it can update
                // its per-tile/per-column size record (ColumnData /
                // TileData) from the now-fresh window geometry.
                //
                // Mirrors niri/src/handlers/compositor.rs:346:
                //   self.niri.layout.update_window(&window, serial);
                //
                // We don't yet thread the ack_configure serial through
                // (would let the layout match commits to specific
                // configures for animation purposes); `None` is the
                // "just resync from the current window geometry" path.
                self.layout.update_window(&window, None);
            }
        }

        // Surface→output assignment + wl_surface.enter/leave. Runs after
        // both process_surface_buffer (in case the new buffer is what
        // produced a layout-visible window) and the optional add_window
        // above, so the layout has the authoritative answer by the time we
        // ask. Also re-runs on every commit, which handles the layout
        // moving a window between outputs.
        dispatch_surface_output_from_layout(self, surface);

        // Damage-driven redraw scheduling: a commit that lands renderable
        // pixels (new buffer, geometry change, popup attach…) needs the
        // output(s) it sits on to repaint. Mark them Queued; the next
        // pass through `redraw_queued_outputs` (driven from the main
        // loop after dispatch) will render them. Idle outputs that
        // nobody committed to stay Idle — no GPU work, no page-flips.
        queue_redraw_for_surface(self, surface);
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

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        // Pop the window out of the layout so the columns behind it
        // can fall into the freed slot. Without this the layout keeps
        // a tile for a window whose surface is gone — invisible (no
        // texture) but still occupying a column, which manifests as
        // "I closed the middle window and the third one didn't slide
        // back."
        //
        // Mirrors niri's `Layout::remove_window` call from its
        // unmap path. `Transaction::new()` is an empty transaction
        // (we don't yet thread the cross-window commit-atomicity
        // transaction system that niri uses to keep resize neighbours
        // in sync; a fresh transaction is the "don't coordinate with
        // anyone" default).
        let wl_surface = surface.wl_surface();
        let lookup = self
            .layout
            .find_window_and_output(wl_surface)
            .map(|(mapped, out)| (mapped.window.clone(), out.cloned()));
        if let Some((window, output)) = lookup {
            self.layout.remove_window(
                &window,
                prism_layout::utils::transaction::Transaction::new(),
            );
            tracing::info!(
                surface_id = ?wl_surface.id(),
                "removed destroyed xdg_toplevel from layout"
            );
            // After removing the last window on an output, the screen
            // hangs on the previous frame until something else triggers
            // a redraw (vblank doesn't fire on its own — render is
            // damage-driven). Queue a redraw on the affected output (or
            // every output as a fallback if we couldn't determine which
            // one) so the now-empty workspace repaints once.
            match output {
                Some(out) => {
                    if let Some(name) = self
                        .wl_outputs
                        .iter()
                        .find_map(|(id, o)| (o == &out).then_some(id.clone()))
                    {
                        self.output_redraw.entry(name).or_default().queue_redraw();
                    }
                }
                None => {
                    let ids: Vec<_> = self.outputs.keys().cloned().collect();
                    for id in ids {
                        self.output_redraw.entry(id).or_default().queue_redraw();
                    }
                }
            }
        }
    }
}

delegate_xdg_shell!(PrismState);

// ─── xdg-decoration v1 ──────────────────────────────────────────────────────
//
// Clients that opt into negotiation get told "server-side" iff
// `config.prefer_no_csd` is set. Clients that don't engage with this
// protocol keep drawing their own decorations regardless (mpv is one
// such example); the focus ring will be occluded for those.
//
// niri's deeper rule (in window-rule:draw-border-with-background) can
// override per-window — not ported yet, simple all-or-nothing here.

impl XdgDecorationHandler for PrismState {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode;
        if self.config.borrow().prefer_no_csd {
            toplevel.with_pending_state(|s| {
                s.decoration_mode = Some(Mode::ServerSide);
            });
            toplevel.send_configure();
        }
    }

    fn request_mode(
        &mut self,
        toplevel: ToplevelSurface,
        mode: smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode,
    ) {
        use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode;
        let chosen = if self.config.borrow().prefer_no_csd {
            // Even if the client asks for CSD, prefer SSD when the
            // compositor is configured that way. Client can override
            // later via another request_mode.
            Mode::ServerSide
        } else {
            mode
        };
        toplevel.with_pending_state(|s| {
            s.decoration_mode = Some(chosen);
        });
        toplevel.send_configure();
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|s| {
            s.decoration_mode = None;
        });
        toplevel.send_configure();
    }
}

delegate_xdg_decoration!(PrismState);

// ─── linux-dmabuf ───────────────────────────────────────────────────────────

impl DmabufHandler for PrismState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    /// Override the default feedback for a freshly-requested
    /// `wp_linux_dmabuf_feedback_v1` if the surface is already mapped
    /// to a known output. Per-output feedback advertises the
    /// direct-scanout-friendly format+modifier we negotiated at output
    /// bringup, so a client allocating against it can produce buffers
    /// the display engine can fetch without going through our
    /// recomposite path.
    ///
    /// Returning `None` lets smithay fall back to the global default
    /// (the broad set we register on the dmabuf global), which is
    /// correct for unmapped surfaces and as a safety net if the
    /// surface's output isn't (yet) in `output_feedback`.
    fn new_surface_feedback(
        &mut self,
        surface: &WlSurface,
        _global: &DmabufGlobal,
    ) -> Option<DmabufFeedback> {
        let current_output = with_states(surface, |states| {
            states
                .data_map
                .get::<SurfacePlacementSlot>()
                .and_then(|slot| slot.0.lock().unwrap().current_output.clone())
        });
        let id = current_output?;
        self.output_feedback.get(&id).cloned()
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

// ─── wlr_layer_shell ────────────────────────────────────────────────────────
// Handler impl lives in `crate::layer_shell`; this just hooks the
// smithay delegate macro so dispatch routes here. The macro generates
// GlobalDispatch + Dispatch impls for the manager + per-surface
// interfaces.

smithay::delegate_layer_shell!(PrismState);

// ─── fractional_scale / single_pixel_buffer / content_type ──────────────────
// Advertise-only today. The handlers are no-op shims so smithay's
// delegate macros can find the trait impl; protocol bookkeeping is
// entirely on smithay's side. See field docs on PrismState for the
// follow-up wiring each needs.

impl FractionalScaleHandler for PrismState {}
delegate_fractional_scale!(PrismState);
delegate_single_pixel_buffer!(PrismState);
delegate_content_type!(PrismState);

// ─── linux_drm_syncobj_v1 ───────────────────────────────────────────────────
// Real handler lives in [`crate::drm_syncobj`] — release tracking,
// pre-commit blocker installation, calloop wiring. This impl just
// hands smithay the state slot (or None when the kernel doesn't
// support `syncobj_eventfd` and we couldn't bring up the global).

impl DrmSyncobjHandler for PrismState {
    fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
        self.drm_syncobj_state.as_mut()
    }
}
delegate_drm_syncobj!(PrismState);

// ─── xdg_activation_v1 ──────────────────────────────────────────────────────
// Activation tokens carry a seat + serial that the requesting client
// captured from a recent input event. We accept the token iff that
// serial is no older than the seat's last keyboard- or pointer-enter
// — same rule niri uses (handlers/mod.rs:773). Tokens without a serial
// (Discord/Telegram tray icons, libnotify-via-notify-osd, …) are
// accepted as "urgency-only" and surface as a focus-ring change
// rather than a focus steal. Mirrors niri's compromise; without it
// those clients can't bring their windows forward at all.
//
// Pre-libinput edge case: `seat.get_keyboard()` returns `None` before
// any libinput device fires; we reject serial-bearing tokens in that
// window since we have no last-enter to compare against.

impl XdgActivationHandler for PrismState {
    fn activation_state(&mut self) -> &mut XdgActivationState {
        &mut self.xdg_activation
    }

    fn token_created(
        &mut self,
        _token: XdgActivationToken,
        data: XdgActivationTokenData,
    ) -> bool {
        let Some((serial, seat)) = data.serial else {
            // No serial — urgency-only path. Always accept; the
            // window manager treats this as a hint, not a focus
            // grant.
            return true;
        };
        let Some(seat) = Seat::<PrismState>::from_resource(&seat) else {
            return false;
        };
        // Compare against both keyboard and pointer last_enter, since
        // a layer-shell surface with no keyboard interactivity may
        // still produce a valid token via pointer focus alone.
        if let Some(kb) = seat.get_keyboard() {
            if kb.last_enter().is_some_and(|le| serial.is_no_older_than(&le)) {
                return true;
            }
        }
        if let Some(ptr) = seat.get_pointer() {
            if ptr.last_enter().is_some_and(|le| serial.is_no_older_than(&le)) {
                return true;
            }
        }
        false
    }

    fn request_activation(
        &mut self,
        token: XdgActivationToken,
        token_data: XdgActivationTokenData,
        surface: WlSurface,
    ) {
        // 10s window — matches niri's `XDG_ACTIVATION_TOKEN_TIMEOUT`
        // and the typical "user just clicked" interval. Older tokens
        // are stale (the user has moved on); silently drop.
        const TOKEN_TIMEOUT_SECS: u64 = 10;
        if token_data.timestamp.elapsed().as_secs() < TOKEN_TIMEOUT_SECS {
            if let Some((mapped, _)) = self.layout.find_window_and_output(&surface) {
                let window = mapped.window.clone();
                self.layout.activate_window(&window);
                // Queue a redraw on every output — `activate_window`
                // may have moved focus across monitors, and we don't
                // know which ones need to repaint until the layout
                // settles.
                let ids: Vec<_> = self.outputs.keys().cloned().collect();
                for id in ids {
                    self.output_redraw.entry(id).or_default().queue_redraw();
                }
            }
            // Surface not yet mapped: today we drop the token. niri
            // queues it on the unmapped-window record so the window
            // gets activated on its first commit; we don't have that
            // bookkeeping yet (no `unmapped_windows` map), so the
            // common case where the just-spawned client's window
            // arrives milliseconds after the activation token won't
            // pre-focus. Track as a follow-up if it bites.
        }
        self.xdg_activation.remove_token(&token);
    }
}
delegate_xdg_activation!(PrismState);

/// Build the per-output `DmabufFeedback` published to clients whose
/// surfaces map onto this output.
///
/// Shape:
///   - **main_device** = the output's GPU render node (falling back to
///     the primary node if no render node is exposed). Tells clients
///     "render here for the cheapest path to this output."
///   - **preference tranche** = the output's `scanout_formats`
///     (direct-scanout-compatible fourcc + modifier list, ordered with
///     the preferred modifier first and LINEAR last). target_device
///     equals main_device — a buffer allocated on the rendering GPU
///     with one of these formats can be scanned out without an
///     intermediate copy through our compositor.
///   - **main tranche** = the broad render-friendly fallback set
///     (`dmabuf_main_formats`). Used by clients that need a wider
///     format range than scanout supports.
///
/// Returns `None` (caller falls back to the global default) if the
/// output's GPU isn't registered or if the feedback builder errored.
/// Both are unexpected in steady state but we don't want them to
/// hard-fail output bringup.
fn build_output_feedback(
    ctx: &prism_drm::OutputContext,
    gpus: &HashMap<DrmDevId, Arc<prism_renderer::Device>>,
    main_formats: &[DrmFormat],
) -> Option<DmabufFeedback> {
    let device = gpus.get(&ctx.gpu_id)?;
    let node = device.physical.drm_render.or(device.physical.drm_primary)?;
    let main_device = libc::makedev(node.major as u32, node.minor as u32);
    let scanout_formats: Vec<DrmFormat> = ctx
        .scanout_formats
        .iter()
        .copied()
        .map(|(code, modifier)| DrmFormat { code, modifier })
        .collect();
    let mut builder = DmabufFeedbackBuilder::new(main_device, main_formats.iter().copied());
    if !scanout_formats.is_empty() {
        builder = builder.add_preference_tranche(main_device, None, scanout_formats);
    }
    match builder.build() {
        Ok(fb) => {
            tracing::info!(
                connector = %ctx.connector_name,
                main_device = format!("{}:{}", node.major, node.minor),
                scanout_n = ctx.scanout_formats.len(),
                "per-output dmabuf feedback built"
            );
            Some(fb)
        }
        Err(e) => {
            tracing::warn!(
                connector = %ctx.connector_name,
                "build_output_feedback: {e:#}; falling back to global default"
            );
            None
        }
    }
}

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
    // Sampled dmabuf imports start in UNDEFINED layout but the render path
    // binds them as SHADER_READ_ONLY_OPTIMAL. Run the one-shot transition
    // here so the first frame's sample is legal — without this radv hangs
    // the queue on the first cmd_draw that touches the descriptor.
    image
        .transition_for_sampling()
        .context("ImportedImage::transition_for_sampling")?;
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

    // Take the new buffer assignment (if any) and (re-)build the texture
    // under the SurfaceData lock. Output assignment + wl_surface.enter/leave
    // dispatch happens separately (after this returns) in
    // dispatch_surface_output_from_layout — that source of truth is the
    // layout, not the buffer's logical_pos.
    with_states(surface, |states| {
        // `on_commit_buffer_handler` (called before us) took the
        // BufferAssignment out of cached_state and stashed it in
        // RendererSurfaceState. We read it back from there. The
        // "previously imported" handle we keep in SurfaceTexSlot
        // tells us whether to re-import or skip.
        let renderer_state = states.data_map.get::<RendererSurfaceStateUserData>();
        let current_buffer = renderer_state
            .and_then(|s| s.lock().unwrap().buffer().cloned());

        states
            .data_map
            .insert_if_missing_threadsafe(SurfaceTexSlot::default);
        let slot = states
            .data_map
            .get::<SurfaceTexSlot>()
            .expect("just inserted SurfaceTexSlot");

        match current_buffer {
            None => {
                // No buffer currently attached — either initial (never had
                // one) or just unmapped (BufferAssignment::Removed arrived
                // this commit and on_commit_buffer_handler cleared the
                // state). Drop our texture; smithay's `InnerBuffer::Drop`
                // already fired `wl_buffer.release` when it cleared its
                // own state — don't release again here.
                let mut guard = slot.0.lock().unwrap();
                let _ = guard.take();
            }
            Some(buffer) => {
                // `buffer` derefs to &WlBuffer. Check whether it's the
                // same WlBuffer we already have a SurfaceTexture for —
                // skip the import on damage-only commits where the
                // client reused the buffer. For shm and for any new
                // dmabuf, (re-)import.
                let wl_buffer: &WlBuffer = &buffer;
                let is_same_dmabuf = matches!(
                    &*slot.0.lock().unwrap(),
                    Some(SurfaceTexture::Dmabuf { buffer: existing, .. }) if existing == wl_buffer
                );
                if !is_same_dmabuf {
                    if let Err(e) = build_surface_texture(state, wl_buffer, slot) {
                        tracing::warn!("surface buffer import failed: {e:#}");
                    }
                }
            }
        }
    });
}

/// Recompute which output the surface lives on (using the layout as the
/// source of truth) and dispatch `wl_surface.enter`/`.leave` on the
/// transition, if any. Updates `SurfacePlacement.current_output`.
///
/// Called from the commit handler after `process_surface_buffer` AND after
/// the optional `add_window` — that order matters, since on the first
/// commit for a fresh toplevel the layout doesn't know the surface until
/// `add_window` returns. Prior code keyed off the surface's `logical_pos`,
/// which defaults to `(0, 0)` and ends up dispatching enter on whichever
/// output happens to contain the origin instead of the one the layout
/// actually placed the window on. Per-frame this also re-syncs us if the
/// layout moved the window to a different monitor.
fn dispatch_surface_output_from_layout(state: &mut PrismState, surface: &WlSurface) {
    // Resolve the surface's current output via the layout. If the surface
    // isn't a layout-tracked window (e.g., a layer surface, once those land)
    // we silently skip — the layer-surface path will do its own dispatch.
    let new_output: Option<String> = state
        .layout
        .find_window_and_output(surface)
        .and_then(|(_, out)| out.map(|o| o.name()));

    // Read + update the placement slot under the SurfaceData lock; return
    // the transition (if any) so we can dispatch enter/leave outside the
    // lock — Output::enter/leave re-enters smithay's surface bookkeeping
    // and we don't want to nest that.
    let transition: Option<(Option<String>, Option<String>)> = with_states(surface, |states| {
        states
            .data_map
            .insert_if_missing_threadsafe(SurfacePlacementSlot::default);
        let placement_slot = states
            .data_map
            .get::<SurfacePlacementSlot>()
            .expect("just inserted SurfacePlacementSlot");
        let mut placement = placement_slot.0.lock().unwrap();
        if placement.current_output == new_output {
            None
        } else {
            let old = placement.current_output.take();
            placement.current_output = new_output.clone();
            Some((old, new_output))
        }
    });

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
            // If this surface already has a wp_linux_dmabuf_feedback_v1
            // bound (because the client called get_surface_feedback at
            // some earlier point), push the new output's feedback so
            // the client can re-allocate against scanout-friendly
            // formats for the new output. Surfaces without a feedback
            // object pick up the per-output feedback on their next
            // `get_surface_feedback` via DmabufHandler::new_surface_feedback.
            if let Some(feedback) = state.output_feedback.get(new_id) {
                with_states(surface, |states| {
                    if let Some(sfs) = SurfaceDmabufFeedbackState::from_states(states) {
                        sfs.set_feedback(feedback);
                    }
                });
            }
            // Same shape for wp_color_management_v1 surface_feedback:
            // push preferred_changed2 with the new output's preferred
            // image description identity. Skipped if the client
            // never bound a feedback object (the slot is missing)
            // or if the identity matches what we last sent.
            if let Some(preferred) = state.color_management.output_preferred(new_id) {
                with_states(surface, |states| {
                    crate::color_management::SurfaceColorFeedbackSlot::notify_preferred_changed(
                        states,
                        preferred.identity,
                    );
                });
            }
        }
    }
}

/// Mark the output(s) hosting `surface` as needing a redraw. Called from
/// the commit handler so a wayland commit drives a render between
/// vblanks instead of having to wait for the next one to schedule us.
/// Outputs nobody committed to stay Idle, so they don't burn vblanks
/// on a no-op page-flip.
///
/// Subsurfaces commit *independently* of their parent in the default
/// `desync` mode (and that's what GTK4 / Firefox / Mesa use). The
/// layout only knows about the toplevel's root wl_surface, so we walk
/// up the parent chain to find the root before querying the layout.
/// Without this, every subsurface commit silently skips redraw
/// queueing — animations on subsurface-backed content (Firefox web
/// content, GTK4 decorations) freeze until something else nudges the
/// output (e.g. cursor motion).
fn queue_redraw_for_surface(state: &mut PrismState, surface: &WlSurface) {
    // Walk up the parent chain to the root of the surface tree. For a
    // toplevel root surface this is a single None-check.
    let mut root = surface.clone();
    while let Some(parent) = get_parent(&root) {
        root = parent;
    }

    // Resolve the root surface to an output via the layout (xdg
    // toplevels) then layer-shell tracking. Surfaces with no layout
    // binding yet (initial commit before add_window) silently skip —
    // the subsequent add_window path queues the redraw itself.
    let output_name = state
        .layout
        .find_window_and_output(&root)
        .and_then(|(_, out)| out.map(|o| o.name()))
        .or_else(|| layer_surface_output(state, &root));

    let Some(output_id) = output_name else {
        return;
    };

    state
        .output_redraw
        .entry(output_id)
        .or_default()
        .queue_redraw();
}

/// Search layer-shell tracking for a surface, returning the OutputId
/// it's bound to. Used by queue_redraw_for_surface (and by
/// dispatch_surface_output_from_layout's enter/leave logic) so a
/// layer-surface commit drives a redraw on the right output.
fn layer_surface_output(state: &PrismState, surface: &WlSurface) -> Option<OutputId> {
    for (id, list) in &state.layer_surfaces {
        for entry in list {
            if entry.surface.wl_surface() == surface {
                return Some(id.clone());
            }
        }
    }
    None
}

/// Upload the current default cursor sprite (frame 0) into a given
/// cursor plane. Used at output attach + on icon changes.
///
/// Phase A: hardcoded to the Named/Default cursor at scale 1. Client
/// surfaces and animation lands later.
fn upload_default_cursor(
    cursor_manager: &CursorManager,
    cache: &CursorTextureCache,
    cursor_plane: &mut prism_drm::CursorPlane,
) -> Result<()> {
    let render = cursor_manager.get_render_cursor(1);
    let (icon, scale, xcursor) = match render {
        RenderCursor::Named { icon, scale, cursor } => (icon, scale, cursor),
        RenderCursor::Hidden | RenderCursor::Surface { .. } => {
            return Ok(());
        }
    };
    let frame = cache.get(icon, scale, &xcursor, 0);
    cursor_plane
        .upload_sprite(&frame.pixels_rgba, frame.width, frame.height)
        .context("CursorPlane::upload_sprite")?;
    Ok(())
}

/// Walk every output, update its cursor plane to show the cursor on
/// the output containing the pointer (and hide on the rest), and
/// queue redraws on outputs whose state changed.
///
/// Called from the input pointer-motion path. Returns the hotspot
/// offset of the current sprite (the cursor *position* on screen is
/// the pointer position minus this hotspot).
///
/// Phase A: a single output ever shows the cursor at a time. Cursor
/// position is computed CRTC-local (pointer global - output origin).
/// The cursor only updates at vblank cadence — Phase B will add
/// sub-vblank cursor-only commits.
pub fn update_output_cursors(state: &mut PrismState) {
    // Resolve the current sprite. If hidden / unsupported, just hide
    // everywhere.
    let render = state.cursor_manager.get_render_cursor(1);
    let (hotspot, owning_output) = match &render {
        RenderCursor::Hidden | RenderCursor::Surface { .. } => {
            // Surface-backed cursor isn't supported yet; treat as
            // hidden for hardware cursor purposes.
            hide_all_cursors(state);
            return;
        }
        RenderCursor::Named { icon, scale, cursor } => {
            let frame = state.cursor_texture_cache.get(*icon, *scale, cursor, 0);
            // xcursor hotspot lives on the original Image, not on
            // CursorImageFrame — fish it back out via xcursor.frame(0).
            let (_idx, image) = cursor.frame(0);
            let hot = (image.xhot as i32, image.yhot as i32);
            // Pick the output the pointer is in.
            let owner = state
                .output_containing((state.pointer_pos.x as i32, state.pointer_pos.y as i32));
            let _ = frame; // sprite already seeded at attach_output
            (hot, owner)
        }
    };

    // Apply visibility + position to each output.
    let pointer_pos = state.pointer_pos;
    for (id, output_ctx) in state.outputs.iter_mut() {
        let Some(cursor) = output_ctx.cursor.as_mut() else {
            continue;
        };
        let wl_output = match state.wl_outputs.get(id) {
            Some(o) => o,
            None => {
                cursor.set_visible(false);
                continue;
            }
        };
        let is_owner = owning_output.as_ref().map_or(false, |o| o == id);
        let was_visible = cursor.visible();
        let prev_pos = cursor.position();

        if is_owner {
            // pointer_pos and origin are both in logical coords; the
            // delta is the logical offset within the output (0..logical_w).
            // The DRM cursor plane wants physical CRTC pixels, so
            // multiply by the output's fractional scale before placing.
            // Hotspot is in cursor-sprite pixels (physical, since the
            // sprite is uploaded at native size into the cursor BO) and
            // subtracts from the physical position as-is.
            //
            // TODO: pick a per-output cursor sprite scale to match —
            // today we always request scale=1 from CursorManager so on
            // a scale=2 monitor the cursor is half its natural size.
            let origin = wl_output.current_location();
            let scale = wl_output.current_scale().fractional_scale().max(0.01);
            let lx = pointer_pos.x - origin.x as f64;
            let ly = pointer_pos.y - origin.y as f64;
            let x = (lx * scale).round() as i32 - hotspot.0;
            let y = (ly * scale).round() as i32 - hotspot.1;
            cursor.set_position(x, y);
            cursor.set_visible(true);
        } else {
            cursor.set_visible(false);
        }

        let changed = was_visible != cursor.visible() || prev_pos != cursor.position();
        if changed {
            state
                .output_redraw
                .entry(id.clone())
                .or_default()
                .queue_redraw();
        }
    }
}

fn hide_all_cursors(state: &mut PrismState) {
    for (id, output_ctx) in state.outputs.iter_mut() {
        if let Some(cursor) = output_ctx.cursor.as_mut() {
            if cursor.visible() {
                cursor.set_visible(false);
                state
                    .output_redraw
                    .entry(id.clone())
                    .or_default()
                    .queue_redraw();
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
    //
    // We do NOT call `old.release()` when swapping — smithay's
    // `RendererSurfaceState` wraps every `WlBuffer` in an
    // `Arc<InnerBuffer>` whose `Drop` already invokes
    // `wl_buffer.release()` (and signals the syncobj release point if
    // one was attached). Doing it ourselves additionally yields a
    // double-release that crashes GTK4 clients on their next frame
    // (cairo's `CAIRO_REFERENCE_COUNT_HAS_REFERENCE` assertion fires
    // when the second release callback hits a freed cairo_surface).
    if let Some(per_gpu) = state.dmabuf_textures.get(&buffer.id()) {
        if per_gpu.is_empty() {
            anyhow::bail!("dmabuf buffer has no imports on any GPU");
        }
        let mut guard = slot.0.lock().unwrap();
        let _previous = guard.replace(SurfaceTexture::Dmabuf {
            by_gpu: per_gpu.clone(),
            buffer: buffer.clone(),
        });
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
    // Bytes are copied; smithay's `InnerBuffer::Drop` will send
    // `wl_buffer.release` when this buffer is replaced on the next
    // commit (or when the surface state resets). Calling it here
    // ourselves was a double-release bug — see comment in the dmabuf
    // branch above.
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
        // fp16. wl_shm `Abgr16161616f` is R,G,B,A half-floats in
        // memory order — that's Vulkan's R16G16B16A16_SFLOAT. `Xbgr`
        // is the alpha-undefined variant; Vulkan has no Xbgr float
        // format, so we sample as R16G16B16A16_SFLOAT and the alpha
        // channel is whatever the client wrote.
        wl_shm::Format::Xbgr16161616f | wl_shm::Format::Abgr16161616f => {
            vk::Format::R16G16B16A16_SFLOAT
        }
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

// ─── Per-output config helpers ──────────────────────────────────────────────

/// Cardinal directions used by `PrismState::output_left/right/up/down`.
/// Kept private; the public API exposes one method per direction.
#[derive(Clone, Copy, Debug)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

/// Do two 1-D intervals overlap? `a_start..a_start+a_len` ∩
/// `b_start..b_start+b_len` non-empty. Used by `neighbor_in_direction`
/// to require y-overlap for left/right neighbors (and x-overlap for
/// up/down).
fn overlaps_x(ax: i32, aw: i32, bx: i32, bw: i32) -> bool {
    ax < bx.saturating_add(bw) && bx < ax.saturating_add(aw)
}

fn overlaps_y(ay: i32, ah: i32, by: i32, bh: i32) -> bool {
    ay < by.saturating_add(bh) && by < ay.saturating_add(ah)
}

/// Logical (post-scale) size of an `Output` in logical pixels. `None` if no
/// current mode is set. Mirrors `Mode.size.to_logical(scale)` but spelled
/// out so we don't depend on a particular smithay overload.
pub(crate) fn output_logical_size(output: &Output) -> Option<(i32, i32)> {
    let mode = output.current_mode()?;
    let scale = output.current_scale().fractional_scale().max(0.01);
    let w = ((mode.size.w as f64) / scale).round() as i32;
    let h = ((mode.size.h as f64) / scale).round() as i32;
    Some((w, h))
}

/// Find the `output "..."` config block for a kernel connector name (e.g.
/// `DisplayPort-4`), the short alias (`DP-4`), OR the EDID
/// `<Make> <Model> <Serial>` triple. The unified matcher lives in
/// [`prism_config::output::block_matches_output`].
///
/// `OutputName` carries the connector + EDID fields — callers in
/// state.rs build it from either an [`OutputContext`] (which holds
/// EDID directly) or a smithay [`Output`] (where the physical_properties
/// were populated from EDID at advertise time).
pub(crate) fn find_output_cfg<'a>(
    output_name: &prism_config::output::OutputName,
    outputs_cfg: &'a [prism_config::output::Output],
) -> Option<&'a prism_config::output::Output> {
    outputs_cfg
        .iter()
        .find(|o| prism_config::output::block_matches_output(&o.name, output_name))
}

/// Build an [`OutputName`] from a smithay output's physical properties.
/// Treats the "Unknown" / empty sentinels the same way
/// [`OutputName::from_ipc_output`] does — those fields drop to `None`
/// so the EDID-matcher doesn't fire on partial-EDID outputs (which
/// can't uniquely identify a physical unit anyway).
pub(crate) fn output_name_from_smithay(
    connector_name: &str,
    output: &Output,
) -> prism_config::output::OutputName {
    let phys = output.physical_properties();
    prism_config::output::OutputName {
        connector: connector_name.to_string(),
        make: (phys.make != "Unknown").then(|| phys.make.clone()),
        model: (phys.model != "Unknown").then(|| phys.model.clone()),
        serial: (!phys.serial_number.is_empty()).then(|| phys.serial_number.clone()),
    }
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
