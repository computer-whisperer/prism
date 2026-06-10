//! `PrismState` — the smithay handler-trait carrier.
//!
//! Smithay's protocol dispatch model: one application-owned struct
//! (`PrismState` here) implements every protocol's `*Handler` trait that the
//! compositor wants to participate in, and a single `delegate_dispatch2!`
//! invocation (below) blanket-implements `Dispatch`/`GlobalDispatch` for
//! `PrismState` by deferring to the per-user-data `Dispatch2` /
//! `GlobalDispatch2` impls (smithay's own for its built-in protocols, ours
//! for the hand-rolled ones).
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
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use prism_animation::Clock;
use prism_config::{Config, PresetSize, WorkspaceReference};
use prism_frame::{DrmFourcc, DrmModifier};
use prism_layout::cursor::{CursorManager, CursorTextureCache, RenderCursor};
use prism_layout::layout::{
    ActivateWindow, AddWindowTarget, Layout, LayoutElement as _, WorkspaceId,
};
use prism_layout::utils::{output_matches_name, update_tiled_state};
use prism_layout::window::{
    InitialConfigureState, Mapped, ResolvedWindowRules, Unmapped, WindowRef,
};
use prism_renderer::{vk, DrmDevId};
use smithay::backend::allocator::dmabuf::Dmabuf as SmithayDmabuf;
use smithay::backend::allocator::Format as DrmFormat;
use smithay::backend::renderer::utils::{
    on_commit_buffer_handler, CommitCounter, RendererSurfaceStateUserData,
};
use smithay::desktop::{
    find_popup_root_surface, get_popup_toplevel_coords, PopupKeyboardGrab, PopupKind, PopupManager,
    PopupPointerGrab, PopupUngrabStrategy, Window,
};
use smithay::input::pointer::{CursorIcon, CursorImageStatus, Focus, PointerHandle};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Scale, Subpixel};
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::{LoopHandle, RegistrationToken};
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_positioner::ConstraintAdjustment;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::backend::{ClientData, ObjectId};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::{Display, DisplayHandle, Resource};
use smithay::utils::{IsAlive, Logical, Rectangle, Serial, Transform};
use smithay::wayland::alpha_modifier::AlphaModifierState;
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    add_blocker, add_pre_commit_hook, get_parent, get_role, with_states,
    with_surface_tree_downward, CompositorClientState, CompositorHandler, CompositorState,
    TraversalAction,
};
use smithay::wayland::content_type::ContentTypeState;
use smithay::wayland::cursor_shape::CursorShapeManagerState;
use smithay::wayland::dmabuf::{
    DmabufFeedback, DmabufFeedbackBuilder, DmabufGlobal, DmabufHandler, DmabufState,
    ImportNotifier, SurfaceDmabufFeedbackState,
};
use smithay::wayland::drm_syncobj::{DrmSyncobjHandler, DrmSyncobjState};
use smithay::wayland::fractional_scale::{FractionalScaleHandler, FractionalScaleManagerState};
use smithay::wayland::idle_inhibit::{IdleInhibitHandler, IdleInhibitManagerState};
use smithay::wayland::idle_notify::{IdleNotifierHandler, IdleNotifierState};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::pointer_constraints::{
    with_pointer_constraint, PointerConstraintsHandler, PointerConstraintsState,
};
use smithay::wayland::presentation::PresentationState;
use smithay::wayland::relative_pointer::RelativePointerManagerState;
use smithay::wayland::selection::data_device::{set_data_device_focus, DataDeviceState};
use smithay::wayland::selection::ext_data_control::DataControlState as ExtDataControlState;
use smithay::wayland::selection::primary_selection::{set_primary_focus, PrimarySelectionState};
use smithay::wayland::selection::wlr_data_control::DataControlState as WlrDataControlState;
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::shell::xdg::dialog::{XdgDialogHandler, XdgDialogState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{with_buffer_contents, ShmHandler, ShmState};
use smithay::wayland::single_pixel_buffer::{self, SinglePixelBufferState};
use smithay::wayland::viewporter::ViewporterState;
use smithay::wayland::xdg_activation::{
    XdgActivationHandler, XdgActivationState, XdgActivationToken, XdgActivationTokenData,
};

use crate::client::PrismClient;
use crate::input_state::{KeyboardFocus, PointerVisibility};
use crate::surface_tex::{
    GpuTex, MirrorChroma, SurfacePlacementSlot, SurfaceTexSlot, SurfaceTexture, TexSource,
};

// The one dispatch wiring for everything: `Dispatch`/`GlobalDispatch` for
// `PrismState` defer to `Dispatch2`/`GlobalDispatch2` on the user-data type.
// This replaced the per-protocol `delegate_*!` macro zoo (smithay 0d14cd65).
smithay::delegate_dispatch2!(PrismState);

/// Stable per-output id. Today we key by the connector name (e.g. `"DP-4"`,
/// `"HDMI-A-1"`). amdgpu's connector names are globally unique across cards
/// on this hardware, so this is sufficient as a primary key. If we ever
/// support a backend that reuses connector names per device, switch to
/// `(DrmDevId, connector::Handle)`.
pub type OutputId = String;

/// A client dmabuf `wl_buffer`'s GPU-agnostic source (dup'd fds) plus its
/// memoized per-GPU native imports. Stored in [`PrismState::dmabuf_sources`]
/// keyed by `wl_buffer` id and dropped when the buffer is destroyed.
///
/// `native_imports` lets a client that re-commits the same `wl_buffer` (a
/// pool/swapchain — the well-behaved pattern most toolkits use) reuse its
/// `VkImage` instead of rebuilding it on every swap back to that buffer. It's
/// the per-buffer analogue of smithay's renderer dmabuf cache
/// (`HashMap<WeakDmabuf, Texture>`): same lifetime (entry dropped with the
/// buffer), so it doesn't keep buffers alive. Interior `Mutex` because the
/// import runs on the per-commit / render-demand path, which holds
/// `&PrismState`; uncontended (main thread), so the lock is ~free. Mirror
/// (cross-GPU) imports are not cached here — their scratch+target already
/// survive buffer swaps via the `SurfaceTexture` carry.
pub struct DmabufSourceEntry {
    pub source: Arc<prism_frame::Dmabuf>,
    native_imports: Mutex<HashMap<DrmDevId, Arc<prism_renderer::ImportedImage>>>,
}

impl DmabufSourceEntry {
    fn new(source: prism_frame::Dmabuf) -> Self {
        Self {
            source: Arc::new(source),
            native_imports: Mutex::new(HashMap::new()),
        }
    }
}

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

    /// Hook for the `load-config-file` action: asks the config file
    /// watcher for an immediate (re)load, optionally from an explicit
    /// path. `None` until `main` wires the watcher up — or forever, when
    /// the watcher failed to start. A boxed closure rather than the
    /// watcher handle itself so prism-protocols stays free of the
    /// watcher type (deliberate divergence from niri, which stores the
    /// watcher on its `State`).
    pub config_load_request: Option<Box<dyn Fn(Option<String>)>>,

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
    /// xdg_popup bookkeeping (menus, dropdowns, tooltips). Popups are
    /// not layout windows and not subsurfaces — they're a separate
    /// surface tree parented to a toplevel (or another popup) and
    /// positioned by an xdg_positioner. `PopupManager` owns the parent
    /// chain and the per-popup positioner geometry; the render path
    /// reads it back via `PopupManager::popups_for_surface` (an
    /// associated fn keyed on the parent surface, so the layout walk
    /// doesn't need a handle to this field).
    pub popups: PopupManager,
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
    /// `zwlr_data_control_manager_v1` — the wlr clipboard-manager
    /// protocol (cliphist, `wl-paste --watch`, clipman). Constructed
    /// with the primary-selection state so managers can watch
    /// middle-click paste too. Advertised to all clients (prism has
    /// no security-context sandboxing to filter on, unlike niri).
    pub wlr_data_control_state: WlrDataControlState,
    /// `ext_data_control_manager_v1` — the standardized successor to
    /// wlr-data-control; newer tools bind this one first.
    pub ext_data_control_state: ExtDataControlState,
    /// Active drag-and-drop cursor icon, set while a DnD grab is
    /// in flight. The render walk draws it on the output under the
    /// pointer, offset by the accumulated `wl_surface.offset` deltas
    /// (see the commit handler's DnD-icon block).
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
    /// ext-foreign-toplevel-list + wlr-foreign-toplevel-management window
    /// lists (taskbars / status bars). Kept in sync with the layout by
    /// [`crate::foreign_toplevel::refresh`], called once per dispatch cycle.
    pub foreign_toplevel_state: crate::foreign_toplevel::ForeignToplevelManagerState,
    /// ext-workspace-v1 workspace observation/control (status bars). Kept in
    /// sync with the layout by [`crate::ext_workspace::refresh`], called once
    /// per dispatch cycle.
    pub ext_workspace_state: crate::ext_workspace::ExtWorkspaceManagerState,
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
    /// `xdg_wm_dialog_v1` (xdg-dialog) state. Carries the per-toplevel
    /// dialog/modal hint, which we fold into the open-floating decision
    /// (see [`ResolvedWindowRules::compute_open_floating`]).
    pub xdg_dialog: XdgDialogState,
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
    /// Sender side of the transaction-completion channel. When a
    /// multi-window [`Transaction`] completes (all participants
    /// committed, or the 300 ms deadline fired), it sends each blocked
    /// client here; the receiver (inserted in [`Self::set_loop_handle`])
    /// calls `CompositorClientState::blocker_cleared` so smithay applies
    /// the queued commits. niri drains an mpsc in its per-dispatch
    /// refresh instead; a calloop channel needs no drain point. `None`
    /// until set_loop_handle — the pre-commit hook then skips blocker
    /// wiring (commits just apply unsynchronized).
    ///
    /// [`Transaction`]: prism_layout::utils::transaction::Transaction
    pub transaction_notify_tx: Option<calloop::channel::Sender<Client>>,
    /// Whether the session is still in its startup phase — the input to
    /// window-rule `match at-startup=true/false`. niri semantics: true
    /// for the first 60 seconds, flipped by a timer registered in
    /// [`Self::set_loop_handle`], which also force-recomputes every
    /// mapped window's rules so at-startup-only matches drop off.
    pub is_at_startup: bool,
    /// Toplevels that exist but aren't in the layout yet, keyed by
    /// root wl_surface. A window lives here from `new_toplevel` until
    /// its first buffer commit maps it into the layout. The initial
    /// configure is computed against this record (window rules,
    /// default column width, target output/workspace — see
    /// [`Self::send_initial_configure`]) so clients draw their first
    /// buffer at the size the layout intends, instead of mapping at
    /// their own natural size and getting resized into the column a
    /// commit later (which left the view-offset fit computed against
    /// the wrong width — the "new window opens partially scrolled"
    /// bug). Mirrors niri's `unmapped_windows`.
    pub unmapped_windows: HashMap<WlSurface, Unmapped>,
    /// xwayland-satellite integration: the bound X11 sockets and their
    /// on-demand spawn watch. `None` when disabled by config, when the
    /// installed satellite is too old, or before [`crate::xwayland`] setup
    /// runs. See [`crate::xwayland::satellite::setup`].
    pub satellite: Option<crate::xwayland::satellite::Satellite>,
    /// `wlr_layer_shell_unstable_v1` server state. MVP — see
    /// [`crate::layer_shell`] for the deliberate scope gaps.
    pub layer_shell_state: smithay::wayland::shell::wlr_layer::WlrLayerShellState,
    /// `ext-idle-notify-v1` state — feeds idle/resume notifications to
    /// clients like swayidle. Created lazily in [`Self::set_loop_handle`]
    /// because [`IdleNotifierState::new`] needs the calloop `LoopHandle`
    /// (its idle timers live on the event loop). `None` until then; no
    /// global exists before it, so the handler getter is never hit early.
    pub idle_notifier: Option<IdleNotifierState<PrismState>>,
    /// `zwp_idle_inhibit_manager_v1` state — held to keep the global
    /// alive. Inhibiting surfaces are tracked in [`Self::idle_inhibitors`];
    /// [`Self::refresh_idle_inhibit`] folds them into the notifier.
    pub idle_inhibit_manager: IdleInhibitManagerState,
    /// Surfaces that currently hold an idle inhibitor (e.g. fullscreen
    /// video). Non-empty ⇒ idle is inhibited. Dead surfaces are pruned on
    /// refresh.
    pub idle_inhibitors: std::collections::HashSet<WlSurface>,
    /// Live `zwlr_output_power_v1` objects (wlopm & co.), each paired with
    /// the `OutputId` it controls, so DPMS changes from any source can be
    /// broadcast back as `mode` events. Dead objects are pruned on
    /// broadcast. See [`crate::output_power`].
    pub output_power_objects: Vec<(
        OutputId,
        smithay::reexports::wayland_protocols_wlr::output_power_management::v1::server::zwlr_output_power_v1::ZwlrOutputPowerV1,
    )>,
    /// `ext-session-lock-v1` global state (screen locking). The actual
    /// lock machinery lives in [`Self::lock_state`]; see
    /// [`crate::session_lock`].
    pub session_lock_state: smithay::wayland::session_lock::SessionLockManagerState,
    /// The session-lock state machine. Render + input paths consult
    /// [`PrismState::is_locked`]; the render path must show ONLY
    /// [`Self::lock_surfaces`] + the locked backdrop while it's true.
    pub lock_state: crate::session_lock::LockState,
    /// Per-output lock surface (keyed like `wl_outputs`). Populated by
    /// `get_lock_surface`, cleared on unlock. An output without an
    /// entry renders the bare locked backdrop.
    pub lock_surfaces: HashMap<OutputId, smithay::wayland::session_lock::LockSurface>,
    /// What the last presented frame on each output showed. The lock is
    /// confirmed to the client only once every powered output is
    /// `Locked` here — see [`PrismState::note_lock_render`].
    pub lock_render_state: HashMap<OutputId, crate::session_lock::LockRenderState>,
    /// Queued screencopy captures (dmabuf or SHM) awaiting their output's next
    /// frame. Drained by [`PrismState::submit_pending_screencopy`] from the
    /// render loop right after `present()`. See [`crate::screencopy`].
    pub screencopy_pending: Vec<crate::screencopy::PendingScreencopy>,
    /// In-flight async screencopy captures: the GPU is rendering into each
    /// entry's target; a calloop sync_fd source fires `ready` (memcpying first
    /// for SHM) and drops the target once done. See [`crate::screencopy`].
    pub screencopy_inflight: Vec<crate::screencopy::ScreencopyInflight>,
    /// Per-output smithay `Output`, keyed by the same `OutputId`
    /// (connector name) as `outputs`. Populated by [`advertise_output`];
    /// logical positions assigned by [`layout_outputs`]. Drops before
    /// `outputs` so wl_output globals are destroyed while the
    /// `DisplayHandle` is still alive.
    pub wl_outputs: HashMap<OutputId, Output>,

    // ── Client buffer textures ─────────────────────────────────────────────
    // Reference Vulkan devices (via Arc); drop before `gpus` so we don't
    // double-free or hit "device destroyed while images outstanding" paths.
    /// GPU-agnostic source description of every accepted dmabuf-backed
    /// `wl_buffer`, keyed by wl_buffer object id, plus its memoized per-GPU
    /// native imports. Holds the dup'd fds so we can import the buffer
    /// *lazily* on whichever GPU(s) actually display the surface
    /// (`ensure_surface_textures`), rather than eagerly on every registered
    /// GPU. Populated in `dmabuf_imported`; dropped in `buffer_destroyed`.
    pub dmabuf_sources: HashMap<ObjectId, Arc<DmabufSourceEntry>>,

    /// Per-GPU command infrastructure for the cross-GPU mirror copy
    /// (`GpuTex::Mirror`). One reusable copier per registered GPU, used
    /// when a surface is displayed on an output whose GPU can't natively
    /// import the client buffer. Empty/unused in single-GPU configs.
    pub mirror_copiers: HashMap<DrmDevId, prism_renderer::MirrorCopier>,

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
    /// Per-card DRM lease state (`wp_drm_lease_device_v1`): VR headset
    /// connectors reserved at bringup + the leases handed out. Keyed
    /// like `cards`. Declared before `cards` so active leases revoke
    /// first on teardown (not strictly required — each `DrmLease`
    /// holds its own fd clone — but it keeps revoke-before-close
    /// ordering obvious).
    pub drm_lease: HashMap<DrmDevId, crate::drm_lease::CardLeaseState>,
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
    /// Whether the libseat session currently holds the seat (i.e. we're on
    /// the foreground VT). Flipped false on `PauseSession` and true on
    /// `ActivateSession` by the session-notifier callback in `main`. While
    /// false we hold no DRM master, so rendering is suppressed — committing a
    /// page-flip would fail with `EACCES`. Always true in headless /
    /// wayland-only modes, which never receive session events.
    pub session_active: bool,

    // ── Input state ────────────────────────────────────────────────────────
    /// The *effective* keyboard focus — the surface smithay's keyboard is
    /// pointed at right now. Computed by [`Self::update_keyboard_focus`],
    /// which arbitrates layer-shell focus against the layout. Defaults to
    /// `Layout { surface: None }` until a window maps.
    pub keyboard_focus: KeyboardFocus,
    /// The layer surface the user clicked while it advertised `OnDemand`
    /// keyboard interactivity, if any. The focus arbiter keeps the keyboard
    /// here until the surface unmaps, stops being `OnDemand`, or focus moves
    /// elsewhere. `None` for the common case (no on-demand layer focused).
    pub on_demand_layer_focus: Option<WlSurface>,
    /// Cursor visibility tri-state — `Visible` normally, `Hidden`
    /// during auto-hide grace, `Disabled` after touch input. See
    /// [`PointerVisibility`]. Consulted by [`update_output_cursors`] to
    /// auto-hide the cursor (`cursor { hide-when-typing / hide-after-inactive-ms }`).
    pub pointer_visibility: PointerVisibility,
    /// Pending `cursor { hide-after-inactive-ms }` timer, (re)armed on each
    /// pointer activity by [`Self::note_pointer_activity`]. `None` when no
    /// timer is pending (option unset, or it already fired).
    pub pointer_inactivity_timer: Option<RegistrationToken>,
    /// Keycodes whose press was swallowed by a compositor binding;
    /// release events for these are filtered out so the focused
    /// client never sees a dangling release. Keyed by raw keycode
    /// (same `Keycode` type smithay's `KeyboardKeyEvent::key_code`
    /// returns).
    pub suppressed_keys: std::collections::HashSet<smithay::input::keyboard::Keycode>,
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

    /// Wheel-tick accumulators (the v120 → discrete-tick conversion
    /// niri's wheel trackers do). Shared by the overview's hardcoded
    /// scroll bindings and the configured `WheelScroll*` bind dispatch
    /// — sharing keeps accumulation seamless when the overview opens
    /// or closes mid-scroll, matching niri's single tracker pair.
    pub vertical_wheel_tracker: prism_layout::input::ScrollTracker,
    pub horizontal_wheel_tracker: prism_layout::input::ScrollTracker,
    /// Finger-scroll accumulators for the configured `TouchpadScroll*`
    /// binds. Separate from the wheel trackers because finger deltas
    /// are in pixels (tick = 10), not v120 units.
    pub vertical_finger_scroll_tracker: prism_layout::input::ScrollTracker,
    pub horizontal_finger_scroll_tracker: prism_layout::input::ScrollTracker,
    /// Mouse button codes whose press was consumed by a configured
    /// `Mouse*` bind; the matching release is swallowed so the focused
    /// client never sees a dangling release (the pointer analogue of
    /// `suppressed_keys`).
    pub suppressed_buttons: std::collections::HashSet<u32>,
    /// Per-bind cooldown deadlines (`cooldown-ms`): a bind found here
    /// with a deadline still in the future does not fire. Instant-based
    /// instead of niri's timer-token map — same semantics, no event
    /// loop entanglement.
    pub bind_cooldown_until: std::collections::HashMap<prism_config::Key, std::time::Instant>,
    /// Active key-repeat timer for a held repeating bind (`repeat`,
    /// default true): re-fires the bind's action at the keyboard's
    /// repeat rate until any key release cancels it. Mirrors niri's
    /// `bind_repeat_timer` (including its known limitation: any
    /// release stops the repeat, not just the bound key's).
    pub bind_repeat_timer: Option<RegistrationToken>,
    /// Last overview wheel workspace switch — niri puts a 50ms
    /// cooldown on its synthesized workspace-switch bind so one flick
    /// doesn't skip several workspaces.
    pub overview_wheel_last_switch: Option<std::time::Instant>,
    /// Accumulated (dx, dy) of an undecided 3-finger touchpad swipe.
    /// `Some` from the swipe-begin until the 16px decision threshold
    /// (GNOME Shell's), at which point the dominant axis picks the
    /// workspace-switch (vertical) or view-offset (horizontal)
    /// gesture and this resets to `None`.
    pub gesture_swipe_3f_cumulative: Option<(f64, f64)>,
    /// Begin/update/end edge detection for two-finger (axis
    /// `Finger`-source) scrolling in the overview, which drives the
    /// workspace-switch / view-offset gestures continuously — libinput
    /// only reports swipe gestures for 3+ fingers.
    pub overview_scroll_swipe_gesture: prism_layout::input::ScrollSwipeGesture,
    /// Whether the pointer was inside a hot corner on the last motion
    /// event — the trigger is edge-sensitive (fires once on entry, not
    /// continuously while parked in the corner).
    pub pointer_inside_hot_corner: bool,

    /// The surface (and its global origin) last reported under the pointer,
    /// as resolved by [`PrismState::contents_under`]. Tracked so that focus
    /// can be re-evaluated after surface/layout changes (window moved,
    /// resized, restacked, subsurface committed) without a pointer-motion
    /// event: [`refresh_pointer_focus`] recomputes the contents and only
    /// re-delivers enter/leave/motion when this differs.
    ///
    /// [`refresh_pointer_focus`]: prism_input equivalent — kept in sync by
    /// both the pointer-motion handlers and the post-dispatch refresh.
    pub pointer_contents: Option<(
        WlSurface,
        smithay::utils::Point<f64, smithay::utils::Logical>,
    )>,

    /// XCursor theme + sprite source. Resolves [`CursorImageStatus`]
    /// (Hidden / Named / client-Surface) into a renderable sprite
    /// every frame. Initialized in [`Self::new`] with a config-derived
    /// theme name + size.
    pub cursor_manager: CursorManager,
    /// Decoded sprite cache feeding the cursor-plane uploader. Keys
    /// by (icon, scale); values are the per-frame ARGB8888 pixels +
    /// dimensions. Populated lazily on first need.
    pub cursor_texture_cache: CursorTextureCache,
    /// Set when the cursor *sprite* changes (client `set_cursor`,
    /// `wp_cursor_shape`, or a commit on the cursor surface) so
    /// [`update_output_cursors`] re-uploads it to the hardware cursor
    /// plane. Pointer *motion* alone only repositions (cheap), it doesn't
    /// re-upload.
    pub cursor_dirty: bool,
    /// Which output's cursor plane currently holds the uploaded sprite, so
    /// we re-upload when the pointer crosses to a different output (whose
    /// plane may hold a stale / differently-scaled sprite). `None` until
    /// the first upload (and while the cursor is hidden).
    pub cursor_uploaded_to: Option<OutputId>,
    /// Hotspot of the currently-uploaded sprite, in physical sprite pixels.
    /// Cached so pointer-motion frames (which don't re-resolve the sprite)
    /// can position the plane without re-reading a client cursor buffer.
    pub cursor_hotspot: (i32, i32),
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
    ///
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
                // RGBA byte order (R8G8B8A8) — the natural shm format for many
                // GL/GLES clients; mandatory ARGB8888/XRGB8888 are always
                // advertised by wl_shm core. Keep this list in sync with
                // vk_format_for_shm.
                smithay::reexports::wayland_server::protocol::wl_shm::Format::Xbgr8888,
                smithay::reexports::wayland_server::protocol::wl_shm::Format::Abgr8888,
                smithay::reexports::wayland_server::protocol::wl_shm::Format::Xbgr16161616f,
                smithay::reexports::wayland_server::protocol::wl_shm::Format::Abgr16161616f,
            ],
        );

        // Dmabuf format/modifier set, queried from the primary GPU's
        // Vulkan driver: the 8-bit BGRA formats plus HDR-capable 10-bit
        // and fp16, each paired with the tiled modifiers the driver
        // reports as SAMPLED-capable (plus LINEAR). Advertising the real
        // modifier set is what stops HDR clients (Firefox, mpv) from
        // allocating implementation-defined layouts we can't import —
        // see build_advertised_dmabuf_formats. Falls back to LINEAR
        // 8-bit when no primary GPU is registered.
        let supported_formats: Vec<DrmFormat> = match primary_gpu.and_then(|id| gpus.get(&id)) {
            Some(device) => build_advertised_dmabuf_formats(device),
            None => vec![
                DrmFormat {
                    code: DrmFourcc::Xrgb8888,
                    modifier: DrmModifier::Linear,
                },
                DrmFormat {
                    code: DrmFourcc::Argb8888,
                    modifier: DrmModifier::Linear,
                },
            ],
        };
        let mut dmabuf_state = DmabufState::new();
        // dmabuf v4 + DmabufFeedback when we know the primary GPU's
        // render node. Without that we'd fall back to v3 (no feedback),
        // and clients like mpv that probe the dmabuf-feedback's
        // main_device to pick a render node land in software EGL.
        let dmabuf_global = match primary_gpu.and_then(|id| gpus.get(&id).map(|dev| (id, dev))) {
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
                let feedback =
                    DmabufFeedbackBuilder::new(main_device, supported_formats.iter().copied())
                        .build()
                        .expect("DmabufFeedbackBuilder::build");
                tracing::info!(
                    "dmabuf v4 advertised with main_device {}:{} ({} format/modifier pairs)",
                    node.major,
                    node.minor,
                    supported_formats.len()
                );
                // Log the distinct fourccs so HDR client support is
                // verifiable at a glance (10-bit / fp16 present?).
                let mut codes: Vec<DrmFourcc> = supported_formats.iter().map(|f| f.code).collect();
                codes.dedup();
                tracing::info!(?codes, "dmabuf advertised fourccs");
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
        let presentation = PresentationState::new::<PrismState>(&dh, libc::CLOCK_MONOTONIC as u32);

        // wp_color_management_v1 global. See module doc for scope —
        // accepts parametric image descriptions, surfaces them via
        // `SurfaceColorSlot`, and the render path decodes per-surface
        // from them (description_to_params → decode shader push).
        let color_management = crate::color_management::ColorManagementState::new(&dh);

        // wlr_layer_shell global. MVP scope — see crate::layer_shell.
        let layer_shell_state =
            smithay::wayland::shell::wlr_layer::WlrLayerShellState::new::<PrismState>(&dh);

        // Idle: zwp_idle_inhibit_manager_v1 now (no loop handle needed);
        // ext-idle-notify-v1 is created in set_loop_handle (it needs the
        // event loop for its timers).
        let idle_inhibit_manager = IdleInhibitManagerState::new::<PrismState>(&dh);

        // ext-session-lock-v1 — screen locking (swaylock). All clients
        // may bind (no security-context filtering yet, matching every
        // other prism global). See crate::session_lock.
        let session_lock_state = smithay::wayland::session_lock::SessionLockManagerState::new::<
            PrismState,
            _,
        >(&dh, |_| true);

        // zwlr_output_power_management_v1 — DPMS control for wlopm/swayidle.
        // Hand-rolled (smithay has none); see crate::output_power. The global
        // is kept alive by the display; nothing else to store.
        crate::output_power::create_output_power_global(&dh);

        // wlr-screencopy — screenshots / capture (grim, wf-recorder, portal).
        // Hand-rolled; see crate::screencopy. Stateless (per-frame data lives in
        // the frame resource), so nothing to store here.
        crate::screencopy::create_screencopy_global(&dh);

        // Window lists (ext-foreign-toplevel-list + wlr-foreign-toplevel-
        // management) and ext-workspace-v1 — taskbar / status-bar support.
        // Hand-rolled niri ports; kept in sync with the layout by the
        // per-dispatch-cycle refresh() calls in the main loop.
        let foreign_toplevel_state = crate::foreign_toplevel::ForeignToplevelManagerState::new(&dh);
        let ext_workspace_state = crate::ext_workspace::ExtWorkspaceManagerState::new(&dh);

        // wp_cursor_shape_v1 — clients request a named cursor shape (text,
        // pointer, grab, …) instead of providing a buffer. smithay routes
        // each request through `SeatHandler::cursor_image(Named(icon))`, so
        // it reuses the themed-cursor render path. Global kept alive by the
        // display.
        let _cursor_shape = CursorShapeManagerState::new::<PrismState>(&dh);

        // Cursor theme + base size from the `cursor { … }` config block (read
        // now, before `config` is moved into the struct below).
        let cursor_manager = {
            let cfg = config.borrow();
            CursorManager::new(&cfg.cursor.xcursor_theme, cfg.cursor.xcursor_size)
        };

        // Modern clients (Firefox, GTK4, recent toolkits) probe these
        // globals at startup and either fall back loudly or take
        // degraded paths when missing. We advertise them now so the
        // protocol surface is complete; per-protocol render/scheduling
        // wiring follows incrementally (see field docs on PrismState).
        let fractional_scale = FractionalScaleManagerState::new::<PrismState>(&dh);
        let single_pixel_buffer = SinglePixelBufferState::new::<PrismState>(&dh);
        let content_type = ContentTypeState::new::<PrismState>(&dh);
        let xdg_activation = XdgActivationState::new::<PrismState>(&dh);
        let xdg_dialog = XdgDialogState::new::<PrismState>(&dh);

        // zwp_relative_pointer_manager_v1 — lets clients (games, 3D/CAD apps,
        // and X11 apps via xwayland-satellite) read unaccelerated relative
        // motion deltas alongside absolute pointer motion. Emitted from the
        // pointer-motion handler; global kept alive by the display.
        let _relative_pointer = RelativePointerManagerState::new::<PrismState>(&dh);

        // zwp_pointer_constraints_v1 — lets a surface lock the pointer in place
        // (FPS mouselook) or confine it to a region (drawing apps). Activation
        // and enforcement live in the pointer-motion handler; smithay
        // auto-deactivates a constraint when pointer focus leaves the surface.
        // Global kept alive by the display.
        let _pointer_constraints = PointerConstraintsState::new::<PrismState>(&dh);

        // zwp_pointer_gestures_v1 — forwards touchpad swipe/pinch/hold
        // gestures the compositor didn't consume (3/4-finger swipes drive
        // workspace switching and the overview) to the focused client.
        // Emission happens in prism-input's gesture handlers via the seat
        // pointer's gesture_* methods. Global kept alive by the display.
        let _pointer_gestures =
            smithay::wayland::pointer_gestures::PointerGesturesState::new::<PrismState>(&dh);

        // wl_data_device_manager + wp_primary_selection_device_manager_v1.
        // GTK4 ≥ 4.22 hard-requires the former; without it every GTK
        // client refuses the wayland display. See crate::selection.
        let data_device_state = DataDeviceState::new::<PrismState>(&dh);
        let primary_selection_state = PrimarySelectionState::new::<PrismState>(&dh);

        // Clipboard-manager protocols (wlr + ext variants), built on top
        // of the selection providers above. Passing the primary-selection
        // state lets managers watch middle-click paste as well.
        let wlr_data_control_state =
            WlrDataControlState::new::<PrismState, _>(&dh, Some(&primary_selection_state), |_| {
                true
            });
        let ext_data_control_state =
            ExtDataControlState::new::<PrismState, _>(&dh, Some(&primary_selection_state), |_| {
                true
            });

        // wp_alpha_modifier_v1 — a per-surface opacity multiplier, committed
        // as double-buffered surface state (smithay caches it in
        // `AlphaModifierSurfaceCachedState`). Consumed by the render walk
        // (`push_surface_tree_elements`), which folds the multiplier into the
        // element's `alpha` fade. Global kept alive by the display.
        let _alpha_modifier = AlphaModifierState::new::<PrismState>(&dh);

        Self {
            config,
            config_load_request: None,
            clock,
            layout,
            display_handle: dh,
            compositor,
            xdg_shell,
            popups: PopupManager::default(),
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
            wlr_data_control_state,
            ext_data_control_state,
            dnd_icon: None,
            viewporter,
            presentation,
            color_management,
            foreign_toplevel_state,
            ext_workspace_state,
            fractional_scale,
            single_pixel_buffer,
            content_type,
            xdg_activation,
            xdg_dialog,
            drm_syncobj_state: None,
            primary_gpu_id: primary_gpu,
            loop_handle: None,
            transaction_notify_tx: None,
            is_at_startup: true,
            unmapped_windows: HashMap::new(),
            satellite: None,
            layer_shell_state,
            session,
            session_active: true,
            drm_lease: HashMap::new(),
            cards: HashMap::new(),
            mirror_copiers: build_mirror_copiers(&gpus),
            gpus,
            outputs: HashMap::new(),
            wl_outputs: HashMap::new(),
            dmabuf_sources: HashMap::new(),
            output_redraw: HashMap::new(),
            keyboard_focus: KeyboardFocus::default(),
            on_demand_layer_focus: None,
            idle_notifier: None,
            idle_inhibit_manager,
            idle_inhibitors: std::collections::HashSet::new(),
            output_power_objects: Vec::new(),
            session_lock_state,
            lock_state: crate::session_lock::LockState::default(),
            lock_surfaces: HashMap::new(),
            lock_render_state: HashMap::new(),
            screencopy_pending: Vec::new(),
            screencopy_inflight: Vec::new(),
            pointer_visibility: PointerVisibility::default(),
            pointer_inactivity_timer: None,
            suppressed_keys: std::collections::HashSet::new(),
            libinput_devices: std::collections::HashSet::new(),
            monitors_active: true,
            should_stop: false,
            pointer_pos: smithay::utils::Point::from((0.0, 0.0)),
            // 120 = one wheel notch in v120 units; 10 = niri's
            // pixels-per-tick for finger scrolls.
            vertical_wheel_tracker: prism_layout::input::ScrollTracker::new(120),
            horizontal_wheel_tracker: prism_layout::input::ScrollTracker::new(120),
            vertical_finger_scroll_tracker: prism_layout::input::ScrollTracker::new(10),
            horizontal_finger_scroll_tracker: prism_layout::input::ScrollTracker::new(10),
            suppressed_buttons: std::collections::HashSet::new(),
            bind_cooldown_until: std::collections::HashMap::new(),
            bind_repeat_timer: None,
            overview_wheel_last_switch: None,
            gesture_swipe_3f_cumulative: None,
            overview_scroll_swipe_gesture: prism_layout::input::ScrollSwipeGesture::new(),
            pointer_inside_hot_corner: false,
            pointer_contents: None,
            cursor_manager,
            cursor_texture_cache: CursorTextureCache::default(),
            // Upload on the first update_output_cursors (correct scale for
            // whatever output the pointer starts on).
            cursor_dirty: true,
            cursor_uploaded_to: None,
            cursor_hotspot: (0, 0),
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
        // ext-idle-notify-v1 keeps its idle timers on the event loop, so it
        // can only be built once we have the loop handle. Build it here so
        // the global is advertised before clients connect (set_loop_handle
        // runs before the wayland socket is inserted).
        self.idle_notifier = Some(IdleNotifierState::new(&self.display_handle, handle.clone()));

        // Transaction-completion notifications: when a multi-window
        // resize transaction completes, each blocked client arrives on
        // this channel and gets its queued commits re-evaluated. See
        // `transaction_notify_tx` and the mapped-toplevel pre-commit
        // hook in `map_new_window`.
        let (tx, rx) = calloop::channel::channel();
        match handle.insert_source(rx, |event, _, state: &mut PrismState| {
            if let calloop::channel::Event::Msg(client) = event {
                let dh = state.display_handle.clone();
                use smithay::wayland::compositor::CompositorHandler;
                state
                    .client_compositor_state(&client)
                    .blocker_cleared(state, &dh);
            }
        }) {
            Ok(_token) => self.transaction_notify_tx = Some(tx),
            Err(e) => {
                // Leave the sender unset: the pre-commit hook then never
                // adds transaction blockers, so commits apply
                // unsynchronized instead of hanging until the deadline.
                tracing::warn!("transaction notify channel insert failed: {e}");
            }
        }

        // End of the startup phase (niri parity: 60 s). Window-rule
        // `match at-startup=true` applies to windows opened before this
        // fires; on the flip, every mapped window re-resolves so
        // at-startup-only matches drop their rules.
        let timer = Timer::from_duration(std::time::Duration::from_secs(60));
        if let Err(e) = handle.insert_source(timer, |_, _, state: &mut PrismState| {
            state.is_at_startup = false;
            if state.recompute_window_rules() {
                queue_redraw_all(state);
            }
            TimeoutAction::Drop
        }) {
            tracing::warn!("startup-phase timer insert failed: {e} (at-startup stays true)");
        }

        self.loop_handle = Some(handle);
    }

    /// Reset the idle timers on every seat — call on any user input so an
    /// idle client (swayidle) is told the user is active again. No-op until
    /// the notifier is built (pre-loop) or if no seat input has occurred.
    pub fn notify_idle_activity(&mut self) {
        let seat = self.seat.clone();
        if let Some(notifier) = self.idle_notifier.as_mut() {
            notifier.notify_activity(&seat);
        }
    }

    /// Recompute whether idle is inhibited (any live inhibitor surface ⇒
    /// inhibited) and push it to the notifier. Prunes dead inhibitors.
    ///
    /// Note: this honors an inhibitor as long as its surface is alive, not
    /// only while it is *visible* — a backgrounded inhibitor still blocks
    /// idle. Visibility-gating (per the protocol's "ignore invisible
    /// inhibitors" note) is a possible refinement.
    pub fn refresh_idle_inhibit(&mut self) {
        self.idle_inhibitors.retain(|s| s.alive());
        let inhibited = !self.idle_inhibitors.is_empty();
        tracing::debug!(
            inhibited,
            inhibitors = self.idle_inhibitors.len(),
            "idle-inhibit refreshed"
        );
        if let Some(notifier) = self.idle_notifier.as_mut() {
            notifier.set_is_inhibited(inhibited);
        }
    }

    /// Note pointer activity (motion / button / axis): reveal the cursor if
    /// it was auto-hidden, and (re)arm the hide-after-inactivity timer.
    pub fn note_pointer_activity(&mut self) {
        if !self.pointer_visibility.is_visible() {
            self.pointer_visibility = PointerVisibility::Visible;
            update_output_cursors(self);
        }
        self.arm_pointer_inactivity_timer();
    }

    /// Hide the cursor because the user is typing, if `cursor {
    /// hide-when-typing }` is set. Called on key press; the cursor reappears
    /// on the next pointer activity. No-op if the option is off or the
    /// cursor is already hidden.
    pub fn hide_pointer_for_typing(&mut self) {
        if !self.config.borrow().cursor.hide_when_typing {
            return;
        }
        if self.pointer_visibility.is_visible() {
            self.pointer_visibility = PointerVisibility::Hidden;
            update_output_cursors(self);
        }
    }

    /// (Re)arm the `cursor { hide-after-inactive-ms }` timer, cancelling any
    /// pending one first. No-op (just cancels) when the option is unset or
    /// no event loop is bound yet.
    fn arm_pointer_inactivity_timer(&mut self) {
        if let (Some(tok), Some(lh)) = (
            self.pointer_inactivity_timer.take(),
            self.loop_handle.as_ref(),
        ) {
            lh.remove(tok);
        }
        let Some(ms) = self.config.borrow().cursor.hide_after_inactive_ms else {
            return;
        };
        let Some(lh) = self.loop_handle.clone() else {
            return;
        };
        self.pointer_inactivity_timer = lh
            .insert_source(
                Timer::from_duration(std::time::Duration::from_millis(ms as u64)),
                |_instant, _, state| {
                    state.pointer_inactivity_timer = None;
                    if state.pointer_visibility.is_visible() {
                        state.pointer_visibility = PointerVisibility::Hidden;
                        update_output_cursors(state);
                    }
                    TimeoutAction::Drop
                },
            )
            .ok();
    }

    /// Drive one output's DPMS power state (see
    /// [`prism_drm::OutputContext::power_off`]). On power-on, queues a
    /// redraw so the next render pass re-establishes the mode. Notifies any
    /// bound `zwlr_output_power_v1` clients of the change. No-op for an
    /// unknown output. Used by the output-power protocol and the
    /// `PowerOffMonitors` / `PowerOnMonitors` actions.
    pub fn set_monitor_powered(&mut self, output_id: &OutputId, on: bool) {
        let Some(ctx) = self.outputs.get_mut(output_id) else {
            return;
        };
        if on {
            ctx.power_on();
        } else {
            if let Err(e) = ctx.power_off() {
                tracing::warn!(connector = %output_id, "DPMS power_off failed: {e:#}");
                return;
            }
            // A zero-damage skip may have left an estimated-vblank timer armed.
            // Drop the redraw state to Idle so its now-stale fire is a no-op
            // (`on_estimated_vblank`'s guard early-returns) instead of waking
            // clients on a powered-off output. The calloop source self-reaps on
            // that single fire.
            self.output_redraw
                .entry(output_id.clone())
                .or_default()
                .redraw = crate::redraw::RedrawState::Idle;
            self.broadcast_output_power_mode(output_id);
            return;
        }
        // Power-on: re-render to re-establish the mode, then notify clients.
        self.output_redraw
            .entry(output_id.clone())
            .or_default()
            .queue_redraw();
        self.broadcast_output_power_mode(output_id);
    }

    /// Drive every output's DPMS power state. Used by the `PowerOffMonitors`
    /// / `PowerOnMonitors` IPC + bind actions (swayidle via `prism msg`, or
    /// a keybind).
    pub fn set_all_monitors_powered(&mut self, on: bool) {
        let ids: Vec<OutputId> = self.outputs.keys().cloned().collect();
        for id in ids {
            self.set_monitor_powered(&id, on);
        }
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
        self.drm_syncobj_state = crate::drm_syncobj::try_init(&self.display_handle, device_fd);
    }

    /// Bring up the `wp_drm_lease_device_v1` globals (one per attached
    /// card) and advertise the non-desktop (VR headset) connectors found
    /// at bringup. Call after cards are attached.
    pub fn init_drm_lease(
        &mut self,
        non_desktop_by_card: HashMap<DrmDevId, Vec<prism_drm::NonDesktopConnector>>,
    ) {
        crate::drm_lease::init(self, non_desktop_by_card);
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
        output: prism_drm::OutputContext,
    ) -> Option<prism_drm::OutputContext> {
        let id: OutputId = output.connector_name.clone();
        // The cursor plane is created hidden; `update_output_cursors`
        // uploads the correct, scale-matched sprite before it makes the
        // plane visible, so no seed upload is needed here.
        // Build the per-output dmabuf feedback before moving `output`.
        // Skipped (and logged) if the output's GPU isn't registered
        // (shouldn't happen — `gpus` is populated before bringup) or
        // if feedback build fails (e.g. shm shortage). Either way the
        // client gets the global default feedback as a fallback.
        if let Some(feedback) =
            build_output_feedback(&output, &self.gpus, &self.dmabuf_main_formats)
        {
            self.output_feedback.insert(id.clone(), feedback);
        }
        // Same step for wp_color_management_v1: derive the output's
        // preferred image description from HDR config + EDID. Used
        // by `wp_color_management_surface_feedback_v1` so clients
        // know "this surface, on this output, should be PQ BT.2020
        // mastered to X nits" (HDR) or sRGB+gamma22 (SDR).
        let preferred =
            crate::color_management::build_output_preferred(&output, &self.color_management);
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
        let name = prism_config::output::OutputName {
            connector: ctx.connector_name.clone(),
            make: ctx.edid.make.clone(),
            model: ctx.edid.model.clone(),
            serial: ctx.edid.serial.clone(),
        };
        let size_mm = ctx
            .edid
            .size_mm
            .map(|(w, h)| (w as i32, h as i32))
            .unwrap_or((0, 0));
        self.advertise_output_from_parts(name, mode, size_mm)
    }

    /// DRM-independent core of [`advertise_output`]: build the smithay
    /// `Output` from already-extracted parts, create its `wl_output`
    /// global, apply mode/scale/transform, attach the `OutputName`
    /// user-data, inform the layout, and stash it in `wl_outputs`.
    ///
    /// `name` drives both scale/transform resolution (the KDL `output
    /// "…"` block lookup) and the `wl_output` make/model/serial
    /// advertisement; its `connector` is the `wl_outputs` map key.
    /// `size_mm` is the physical panel size in millimetres (`(0, 0)` ⇒
    /// unknown). Scale comes from config; transform is forced to `Normal`
    /// until the render path can rotate scanout. Logical position is
    /// **not** assigned here — call [`layout_outputs`] once every output
    /// has been advertised.
    ///
    /// Used by [`advertise_output`] (DRM path, parts pulled from an
    /// `OutputContext`) and by the WLCS test harness (synthetic parts,
    /// no `OutputContext` / no scanout behind the output).
    pub fn advertise_output_from_parts(
        &mut self,
        name: prism_config::output::OutputName,
        mode: OutputMode,
        size_mm: (i32, i32),
    ) -> &Output {
        let (scale, transform) = self.resolve_output_scale_transform(&name);
        let connector = name.connector.clone();
        let make = name.make.clone().unwrap_or_else(|| "Unknown".to_owned());
        let model = name.model.clone().unwrap_or_else(|| connector.clone());
        let serial_number = name.serial.clone().unwrap_or_default();
        let output = Output::new(
            connector.clone(),
            PhysicalProperties {
                size: size_mm.into(),
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
            connector = %connector,
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
        output.user_data().insert_if_missing(|| name);
        // Inform the layout. This creates a Monitor entry, splices in any
        // workspaces that named this output via `original_output`, and
        // (if this is the first output) hosts the no-output workspace
        // backlog. `None` layout_config = use defaults; per-output config
        // lookup arrives once we wire `config.outputs` indexing.
        self.layout.add_output(output.clone(), None);
        self.wl_outputs.insert(connector.clone(), output);
        // unwrap: just inserted under this key
        self.wl_outputs.get(&connector).unwrap()
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
                    Direction::Left => {
                        (cx < cur_cx && overlaps_y(cur.2, cur.4, y, h)).then(|| (o, cur_cx - cx))
                    }
                    Direction::Right => {
                        (cx > cur_cx && overlaps_y(cur.2, cur.4, y, h)).then(|| (o, cx - cur_cx))
                    }
                    Direction::Up => {
                        (cy < cur_cy && overlaps_x(cur.1, cur.3, x, w)).then(|| (o, cur_cy - cy))
                    }
                    Direction::Down => {
                        (cy > cur_cy && overlaps_x(cur.1, cur.3, x, w)).then(|| (o, cy - cur_cy))
                    }
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

    /// First output matching `target` by name — connector ("DP-2") or
    /// make/model/serial, case-insensitive. Niri's `output_by_name_match`;
    /// backs the `focus-monitor "name"` family of binds.
    pub fn output_by_name_match(&self, target: &str) -> Option<Output> {
        self.wl_outputs
            .values()
            .find(|o| output_matches_name(o, target))
            .cloned()
    }

    /// Resolve a `WorkspaceReference` from a bind or IPC request to
    /// `(owning output, workspace index on that output)`. `None` output
    /// means "the active output" — always the case for plain `Index`
    /// references, which are 1-based in config and clamp to 0-based here.
    /// Name/Id references fail (`None`) if no such workspace exists.
    /// Ported from niri's `find_output_and_workspace_index`.
    pub fn find_output_and_workspace_index(
        &self,
        reference: WorkspaceReference,
    ) -> Option<(Option<Output>, usize)> {
        let (index, workspace) = match reference {
            WorkspaceReference::Index(index) => {
                return Some((None, usize::from(index.saturating_sub(1))));
            }
            WorkspaceReference::Name(name) => self.layout.find_workspace_by_name(&name)?,
            WorkspaceReference::Id(id) => self
                .layout
                .find_workspace_by_id(WorkspaceId::specific(id))?,
        };
        Some((workspace.current_output().cloned(), index))
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
                if v < 0.1 {
                    // The config type's range floor is 0 (FloatOrInt<0, 10>),
                    // so `scale 0` parses — but a zero/near-zero scale makes
                    // every logical-size division degenerate. Refuse it here.
                    tracing::warn!(
                        connector = %output_name.connector,
                        scale = v,
                        "output scale below 0.1 is degenerate; using 1"
                    );
                    Scale::Integer(1)
                } else if v == v.trunc() && v >= 1.0 {
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
                    let pos =
                        find_output_cfg(&output_name, &cfg.outputs.0).and_then(|o| o.position);
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

    /// Drive the layout's fullscreen state for the window backing a
    /// client toplevel. Shared by the `set_fullscreen` xdg request
    /// handlers; mirrors the keybind path in `prism-input::actions`.
    fn set_window_fullscreen(&mut self, surface: &ToplevelSurface, fullscreen: bool) {
        let window = self
            .layout
            .find_window_and_output(surface.wl_surface())
            .map(|(mapped, _)| mapped.window.clone());
        if let Some(w) = window {
            self.layout.set_fullscreen(&w, fullscreen);
            queue_redraw_for_surface(self, surface.wl_surface());
        }
    }

    /// Drive the layout's maximized state for the window backing a
    /// client toplevel. Shared by the `set_maximized` xdg request
    /// handlers.
    fn set_window_maximized(&mut self, surface: &ToplevelSurface, maximize: bool) {
        let window = self
            .layout
            .find_window_and_output(surface.wl_surface())
            .map(|(mapped, _)| mapped.window.clone());
        if let Some(w) = window {
            self.layout.set_maximized(&w, maximize);
            queue_redraw_for_surface(self, surface.wl_surface());
        }
    }

    /// Resolve window rules + target placement for an unmapped toplevel
    /// and send its initial configure, sized by the layout (default
    /// column width × working-area height, or the floating / fullscreen
    /// / maximized variants). The resolved state is recorded on the
    /// `Unmapped` record and consumed by [`Self::map_unmapped_toplevel`].
    /// Sending a concrete size here is what makes clients draw their
    /// first buffer at the width the column will actually have — mapping
    /// at the client's own natural size left the view-offset fit
    /// computed against the wrong width (the "new window opens partially
    /// scrolled" bug). Ported from niri's `State::send_initial_configure`.
    fn send_initial_configure(&mut self, toplevel: &ToplevelSurface) {
        // The output hosting the pointer — fallback target monitor. niri
        // uses its focus-follows-mouse infra plus the last-active
        // monitor here; prism approximates with the pointer position
        // until focus tracking drives `active_monitor_idx` on its own
        // (same approximation the map path used before this port).
        // Computed up front: it's a `&self` method call, which can't
        // overlap the `&mut` borrow of the unmapped record below.
        let pointer_output = self
            .output_containing((self.pointer_pos.x as i32, self.pointer_pos.y as i32))
            .as_ref()
            .and_then(|id| self.wl_outputs.get(id))
            .cloned();

        let Some(unmapped) = self.unmapped_windows.get_mut(toplevel.wl_surface()) else {
            tracing::error!(
                "window must be present in unmapped_windows in send_initial_configure()"
            );
            return;
        };

        let config = self.config.borrow();
        let rules = ResolvedWindowRules::compute(
            &config.window_rules,
            WindowRef::Unmapped(unmapped),
            self.is_at_startup,
        );

        let Unmapped { window, state, .. } = unmapped;

        let InitialConfigureState::NotConfigured {
            wants_fullscreen,
            wants_maximized,
        } = state
        else {
            tracing::error!("window must not be already configured in send_initial_configure()");
            return;
        };

        // Pick the target monitor. First, check if we had a workspace
        // set in the window rules.
        let mon = rules
            .open_on_workspace
            .as_deref()
            .and_then(|name| self.layout.monitor_for_workspace(name));

        // If not, check if we had an output set in the window rules.
        let mon = mon.or_else(|| {
            rules
                .open_on_output
                .as_deref()
                .and_then(|name| {
                    self.wl_outputs
                        .values()
                        .find(|output| output_matches_name(output, name))
                })
                .and_then(|o| self.layout.monitor_for_output(o))
        });

        // If not, check if the window requested one for fullscreen.
        let mon = mon.or_else(|| {
            wants_fullscreen
                .as_ref()
                .and_then(|x| x.as_ref())
                // The monitor might not exist if the output was disconnected.
                .and_then(|o| self.layout.monitor_for_output(o))
        });

        // If not, check if this is a dialog with a parent, to place it
        // next to the parent.
        let mon = mon.map(|mon| (mon, false)).or_else(|| {
            toplevel
                .parent()
                .and_then(|parent| self.layout.find_window_and_output(&parent))
                .and_then(|(_win, output)| output)
                .and_then(|o| self.layout.monitor_for_output(o))
                .map(|mon| (mon, true))
        });

        // If not, use the pointer's output, then the active monitor.
        let mon = mon.or_else(|| {
            pointer_output
                .as_ref()
                .and_then(|o| self.layout.monitor_for_output(o))
                .map(|mon| (mon, false))
        });
        let mon = mon.or_else(|| self.layout.active_monitor_ref().map(|mon| (mon, false)));

        // If we're following the parent, don't set the target output, so
        // that when the window is mapped, it fetches the possibly changed
        // parent's output again, and shows up there.
        let output = mon
            .filter(|(_, parent)| !parent)
            .map(|(mon, _)| mon.output().clone());
        let mon = mon.map(|(mon, _)| mon);

        let mut width = None;
        let mut floating_width = None;
        let mut height = None;
        let mut floating_height = None;
        let is_full_width = rules.open_maximized.unwrap_or(false);
        let is_floating = rules.compute_open_floating(toplevel);

        // Tell the surface the preferred size and bounds for its likely
        // output.
        let ws = rules
            .open_on_workspace
            .as_deref()
            .and_then(|name| mon.map(|mon| mon.find_named_workspace(name)))
            .unwrap_or_else(|| {
                mon.map(|mon| mon.active_workspace_ref())
                    .or_else(|| self.layout.active_workspace())
            });

        let mut is_pending_maximized = false;
        if let Some(ws) = ws {
            // Set a fullscreen and maximized state based on the window's
            // request and the window rules.
            is_pending_maximized = (*wants_maximized && rules.open_maximized_to_edges.is_none())
                || rules.open_maximized_to_edges == Some(true);

            if (wants_fullscreen.is_some() && rules.open_fullscreen.is_none())
                || rules.open_fullscreen == Some(true)
            {
                toplevel.with_pending_state(|state| {
                    state.states.set(xdg_toplevel::State::Fullscreen);
                });
            } else if is_pending_maximized {
                toplevel.with_pending_state(|state| {
                    state.states.set(xdg_toplevel::State::Maximized);
                });
            }

            width = ws.resolve_default_width(rules.default_width, false);
            floating_width = ws.resolve_default_width(rules.default_width, true);
            height = ws.resolve_default_height(rules.default_height, false);
            floating_height = ws.resolve_default_height(rules.default_height, true);

            let configure_width = if is_floating {
                floating_width
            } else if is_full_width {
                Some(PresetSize::Proportion(1.))
            } else {
                width
            };
            let configure_height = if is_floating { floating_height } else { height };
            ws.configure_new_window(
                window,
                configure_width,
                configure_height,
                is_floating,
                &rules,
            );
        }

        // Set the tiled state for the initial configure.
        update_tiled_state(toplevel, config.prefer_no_csd, rules.tiled_state);

        // Record the resolved settings; the map path consumes them.
        *state = InitialConfigureState::Configured {
            rules,
            width,
            height,
            floating_width,
            floating_height,
            is_full_width,
            output,
            workspace_name: ws.and_then(|w| w.name().cloned()),
            is_pending_maximized,
        };

        tracing::info!(
            surface_id = ?toplevel.wl_surface().id(),
            "sent initial configure to xdg_toplevel"
        );
        toplevel.send_configure();
    }

    /// Send the initial configure from an idle callback, so the client
    /// can supply more info (title, app_id, min/max size) after the
    /// initial commit — window rules and the open-floating heuristics
    /// match on those. Ported from niri's `queue_initial_configure`.
    fn queue_initial_configure(&mut self, toplevel: ToplevelSurface) {
        let Some(loop_handle) = self.loop_handle.clone() else {
            // No event loop handle yet (theoretical — the wayland socket
            // binds after `set_loop_handle`); configure inline.
            self.send_initial_configure(&toplevel);
            return;
        };
        loop_handle.insert_idle(move |state| {
            if !toplevel.alive() {
                return;
            }
            if let Some(unmapped) = state.unmapped_windows.get(toplevel.wl_surface()) {
                if unmapped.needs_initial_configure() {
                    state.send_initial_configure(&toplevel);
                }
            }
        });
    }

    /// Map a toplevel out of `unmapped_windows` into the layout,
    /// consuming the rules / size / placement resolved at
    /// initial-configure time. Ported from niri's map path
    /// (handlers/compositor.rs).
    fn map_unmapped_toplevel(&mut self, surface: &WlSurface) {
        let Some(unmapped) = self.unmapped_windows.remove(surface) else {
            return;
        };
        let Unmapped {
            window,
            state,
            activation_token_data,
        } = unmapped;

        // Refresh the window's cached bbox from the committed surface
        // tree. Without this, `Window::geometry()` returns an empty rect
        // (the bbox is initialised to zero), so `tile.size =
        // window.geometry().size` is (0,0) — and `Column::width()` hands
        // the layout a zero-width column.
        window.on_commit();

        let toplevel = window.toplevel().expect("no x11 support").clone();

        let (rules, width, height, is_full_width, output, workspace_id, is_pending_maximized) =
            if let InitialConfigureState::Configured {
                rules,
                width,
                height,
                floating_width: _,
                floating_height: _,
                is_full_width,
                output,
                workspace_name,
                is_pending_maximized,
            } = state
            {
                // Check that the output is still connected.
                let output = output.filter(|o| self.layout.monitor_for_output(o).is_some());

                // Check that the workspace still exists.
                let workspace_id = workspace_name
                    .as_deref()
                    .and_then(|n| self.layout.find_workspace_by_name(n))
                    .map(|(_, ws)| ws.id());

                (
                    rules,
                    width,
                    height,
                    is_full_width,
                    output,
                    workspace_id,
                    is_pending_maximized,
                )
            } else {
                // Can happen when a surface unmaps by attaching a null
                // buffer while there are in-flight pending configures.
                tracing::debug!("window mapped without proper initial configure");
                (
                    ResolvedWindowRules::default(),
                    None,
                    None,
                    false,
                    None,
                    None,
                    false,
                )
            };

        // The GTK about dialog sets min/max size after the initial
        // configure but before mapping, so we need to compute
        // open_floating at the last possible moment, that is here.
        let is_floating = rules.compute_open_floating(&toplevel);

        // Figure out if we should activate the window.
        let activate = rules.open_focused.map(|focus| {
            if focus {
                ActivateWindow::Yes
            } else {
                ActivateWindow::No
            }
        });
        let activate = activate.unwrap_or_else(|| {
            // Check the token timestamp again in case the window took a
            // while between requesting activation and mapping.
            let token = activation_token_data
                .filter(|token| token.timestamp.elapsed().as_secs() < TOKEN_TIMEOUT_SECS);
            if token.is_some() {
                ActivateWindow::Yes
            } else {
                ActivateWindow::Smart
            }
        });

        // Open dialogs next to their parent window. Only consider the
        // parent if we configured the window for the same output:
        // normally when we're following the parent, the configured
        // output is None; if it's set, it came from a window rule or a
        // fullscreen request.
        let parent = toplevel
            .parent()
            .and_then(|parent| self.layout.find_window_and_output(&parent))
            .filter(|(_, parent_output)| {
                parent_output.is_none() || output.is_none() || output.as_ref() == *parent_output
            })
            .map(|(mapped, _)| mapped.window.clone());

        // Pre-commit hook: drains the transaction queue and
        // drives the resize animation's snapshot capture.
        // niri also uses it for dmabuf-readiness blockers;
        // prism instead waits on the client's implicit write
        // fence at render time (see prepare_dmabuf_acquire_waits).
        //
        // It fires before `on_commit_buffer_handler`
        // applies the new buffer, so the window still
        // reports its OLD size here — exactly when the
        // resize animation needs to snapshot the
        // pre-resize size. If this commit acks a configure
        // we flagged to animate (`request_size(animate=true)`
        // → `animate_next_configure` → `animate_serials`),
        // `should_animate_commit` matches the acked serial
        // and we store the snapshot; `Tile::update_window`
        // later consumes it (`take_animation_snapshot`) to
        // seed `ResizeAnimation.size_from`. Mirrors niri's
        // `add_mapped_toplevel_pre_commit_hook`.
        let hook = add_pre_commit_hook::<PrismState, _>(surface, |state, _dh, surface| {
            // The serial the client acked with this
            // commit (set by ack_configure, processed
            // before this hook). `None` before any ack.
            let acked = with_states(surface, |states| {
                states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .and_then(|d| d.lock().unwrap().last_acked.as_ref().map(|c| c.serial))
            });
            let Some(serial) = acked else {
                return;
            };
            // None until the window is mapped into the
            // layout — initial pre-map commits no-op.
            let Some((mapped, _)) = state.layout.find_window_and_output_mut(surface) else {
                return;
            };

            // Drain this commit's transaction (niri parity). A
            // transactional configure (multi-window column resize)
            // queued a clone of the column's Transaction against the
            // configure serial; this commit fulfills our window's part.
            // If other windows in the transaction haven't committed yet,
            // block THIS commit on the transaction so all windows
            // resize in lockstep — the blocker releases when the last
            // participant commits (its drop completes the transaction
            // and notifies us via transaction_notify_tx) or when the
            // 300 ms deadline fires.
            if let Some(transaction) = mapped.take_pending_transaction(serial) {
                // Already completed = it ran past the deadline.
                let disable = state.config.borrow().debug.disable_transactions;
                if !transaction.is_completed() && !disable {
                    if let Some(loop_handle) = state.loop_handle.as_ref() {
                        transaction.register_deadline_timer(loop_handle);
                    }
                    // Last holder: dropping it below completes the
                    // transaction and wakes the other windows' blocked
                    // commits — no blocker needed on our own surface.
                    if !transaction.is_last() {
                        if let (Some(tx), Some(client)) =
                            (state.transaction_notify_tx.as_ref(), surface.client())
                        {
                            transaction.add_notification(tx.clone(), client.clone());
                            add_blocker(surface, transaction.blocker());
                        }
                    }
                }
                // Transaction dropped here. niri keeps it alive until
                // the commit's dmabuf is ready; prism's producer-sync
                // happens at render time, so there's nothing to wait on.
            }

            if mapped.should_animate_commit(serial) {
                mapped.store_animation_snapshot();
            }
        });

        let mapped = {
            let config = self.config.borrow();
            Mapped::new(window, rules, hook, &config)
        };
        let id = mapped.id();
        // The layout keys windows by their smithay `Window`
        // (its `LayoutElement::Id`), distinct from the
        // `MappedId` above. Clone the handle before `mapped` is
        // moved into `add_window` so we can start the open
        // animation afterwards.
        let window = mapped.window.clone();

        let target = if let Some(p) = &parent {
            AddWindowTarget::NextTo(p)
        } else if let Some(ws_id) = workspace_id {
            AddWindowTarget::Workspace(ws_id)
        } else if let Some(output) = &output {
            AddWindowTarget::Output(output)
        } else {
            AddWindowTarget::Auto
        };
        let output = self
            .layout
            .add_window(
                mapped,
                target,
                width,
                height,
                is_full_width,
                is_floating,
                activate,
            )
            .cloned();

        // The window state cannot contain Fullscreen and Maximized at
        // once. Therefore, if the window ended up fullscreen, then we
        // only know that it is also maximized from the
        // is_pending_maximized variable. Tell the layout about it here
        // so that unfullscreening the window makes it maximized.
        if is_pending_maximized {
            let pending_fullscreen = self
                .layout
                .find_window_and_output(surface)
                .map(|(m, _)| m.pending_sizing_mode().is_fullscreen())
                .unwrap_or(false);
            if pending_fullscreen {
                self.layout.set_maximized(&window, true);
            }
        }

        // Make the new window's monitor the active
        // one so its tile's focus ring renders with
        // active-color, not inactive-color. Without
        // this `active_monitor_idx` stays pinned to
        // monitor 0 (DP-4 in connector-name sort
        // order) and only DP-4 windows ever look
        // focused. niri does this from its input
        // handlers via `layout.focus_output(&output)`;
        // for the MVP we do it at add_window time.
        let output_name = output.as_ref().map(|o| o.name());
        if let Some(out) = output {
            self.layout.focus_output(&out);
        }
        // Kick off the open (zoom + fade-in) animation now that
        // the window is in the layout — niri does the same right
        // after adding (handlers/compositor.rs).
        self.layout.start_open_animation_for_window(&window);
        tracing::info!(
            ?id,
            output = ?output_name,
            "mapped xdg_toplevel into layout"
        );
    }

    /// Apply a freshly parsed config (file watcher / future IPC reload).
    /// Mirrors niri's `State::reload_config` (niri.rs:1421), minus the
    /// pieces prism doesn't have yet (layer rules, custom animation
    /// shaders, libinput device settings, xwayland-satellite restart).
    ///
    /// The caller has already stripped startup-only sections (environment,
    /// spawn-at-startup). Sections that only take effect at startup are
    /// detected and logged as needing a restart rather than silently
    /// half-applied: `output` blocks (mode/position/scale/HDR/LUT decisions
    /// are baked into DRM + renderer state at bringup) and `debug`.
    pub fn reload_config(&mut self, config: Config) {
        tracing::info!("applying reloaded config");

        // Named workspaces removed from the config lose their name (they
        // become regular workspaces, cleaned up when emptied).
        let removed_workspaces: Vec<String> = self
            .config
            .borrow()
            .workspaces
            .iter()
            .filter(|ws| !config.workspaces.iter().any(|w| w.name == ws.name))
            .map(|ws| ws.name.0.clone())
            .collect();
        for name in removed_workspaces {
            self.layout.unname_workspace(&name);
        }

        // Layout options: gaps, struts, default widths, focus ring, border,
        // shadow, animations — propagates down monitors → workspaces →
        // tiles.
        self.layout.update_config(&config);
        for ws_config in &config.workspaces {
            self.layout.ensure_named_workspace(ws_config);
        }

        self.clock
            .set_rate(1.0 / config.animations.slowdown.max(0.001));
        self.clock.set_complete_instantly(config.animations.off);

        // Diff the sections that need explicit re-application, then swap
        // the shared config in place. Everything that reads the config live
        // per event (binds, mod-key, focus-follows-mouse, keyboard
        // shortcuts) picks the swap up automatically.
        let mut reload_xkb = None;
        let mut reload_repeat = None;
        let window_rules_changed;
        {
            let mut old_config = self.config.borrow_mut();

            if config.cursor.xcursor_theme != old_config.cursor.xcursor_theme
                || config.cursor.xcursor_size != old_config.cursor.xcursor_size
            {
                self.cursor_manager
                    .reload(&config.cursor.xcursor_theme, config.cursor.xcursor_size);
                self.cursor_texture_cache.clear();
            }

            let kb = &config.input.keyboard;
            let old_kb = &old_config.input.keyboard;
            if kb.xkb != old_kb.xkb {
                // Pre-existing gap shared with the startup path: `xkb.file`
                // is parsed but not honoured (no set_xkb_file machinery).
                reload_xkb = Some(kb.xkb.clone());
            }
            if kb.repeat_rate != old_kb.repeat_rate || kb.repeat_delay != old_kb.repeat_delay {
                reload_repeat = Some((i32::from(kb.repeat_rate), i32::from(kb.repeat_delay)));
            }

            window_rules_changed = config.window_rules != old_config.window_rules;

            if config.outputs != old_config.outputs {
                tracing::warn!(
                    "`output` configuration changed: not hot-reloaded (modes, positions, \
                     scale, HDR signaling and LUTs are resolved at bringup) — restart prism \
                     to apply"
                );
            }
            if config.debug != old_config.debug {
                tracing::warn!("`debug` configuration changed: not hot-reloaded — restart prism");
            }

            *old_config = config;
        }

        if let Some(xkb) = reload_xkb {
            if let Some(keyboard) = self.seat.get_keyboard() {
                // Changing the keymap resets modifier state; carry num lock
                // over (niri does the same — it's latched state the user
                // doesn't expect a config reload to clear).
                let num_lock = keyboard.modifier_state().num_lock;
                if let Err(err) = keyboard.set_xkb_config(self, xkb.to_xkb_config()) {
                    tracing::warn!("error updating xkb config: {err:?}");
                } else {
                    let mut mods = keyboard.modifier_state();
                    if mods.num_lock != num_lock {
                        mods.num_lock = num_lock;
                        keyboard.set_modifier_state(mods);
                    }
                }
            }
        }
        if let Some((rate, delay)) = reload_repeat {
            if let Some(keyboard) = self.seat.get_keyboard() {
                keyboard.change_repeat_info(rate, delay);
            }
        }

        if window_rules_changed {
            self.recompute_window_rules();
        }

        // Decoration/rule changes re-damage via element content tokens; a
        // full redraw pass picks them all up on the next frame.
        queue_redraw_all(self);
    }

    /// Force-recompute every window's resolved rules — mapped windows
    /// through the layout (re-configuring those whose rules changed,
    /// since sizing rules affect tile geometry), and configured-but-
    /// unmapped windows in place (their recorded rules are consumed at
    /// map time). Returns whether any mapped window changed. Callers:
    /// config hot-reload (rules section edited) and the startup-phase
    /// flip (`at-startup` matches drop off). Mirrors niri's
    /// `recompute_window_rules`.
    pub fn recompute_window_rules(&mut self) -> bool {
        let is_at_startup = self.is_at_startup;
        let config = Rc::clone(&self.config);
        let config = config.borrow();

        for unmapped in self.unmapped_windows.values_mut() {
            let new_rules = ResolvedWindowRules::compute(
                &config.window_rules,
                WindowRef::Unmapped(unmapped),
                is_at_startup,
            );
            if let InitialConfigureState::Configured { rules, .. } = &mut unmapped.state {
                *rules = new_rules;
            }
        }

        let mut changed_windows = Vec::new();
        self.layout.with_windows_mut(|mapped, _output| {
            if mapped.recompute_window_rules(&config.window_rules, is_at_startup) {
                changed_windows.push(mapped.window.clone());
            }
        });
        drop(config);
        let any_changed = !changed_windows.is_empty();
        for win in changed_windows {
            self.layout.update_window(&win, None);
        }
        any_changed
    }

    /// Re-resolve window rules after a title or app-id change — rules
    /// matching on title (e.g. "Save As" dialogs) must re-apply when the
    /// title arrives or changes after mapping. Port of niri's
    /// `update_window_rules` (handlers/xdg_shell.rs): an unmapped-but-
    /// configured window updates its recorded rules in place (consumed at
    /// map time); a mapped window recomputes, and on change re-runs
    /// `update_window` + queues a redraw on its output.
    pub fn update_window_rules(&mut self, toplevel: &ToplevelSurface) {
        let is_at_startup = self.is_at_startup;
        let config = Rc::clone(&self.config);
        let config = config.borrow();
        let window_rules = &config.window_rules;

        if let Some(unmapped) = self.unmapped_windows.get_mut(toplevel.wl_surface()) {
            let new_rules = ResolvedWindowRules::compute(
                window_rules,
                WindowRef::Unmapped(unmapped),
                is_at_startup,
            );
            if let InitialConfigureState::Configured { rules, .. } = &mut unmapped.state {
                *rules = new_rules;
            }
        } else if let Some((mapped, output)) = self
            .layout
            .find_window_and_output_mut(toplevel.wl_surface())
        {
            if mapped.recompute_window_rules(window_rules, is_at_startup) {
                let output = output.cloned();
                let window = mapped.window.clone();
                drop(config);
                self.layout.update_window(&window, None);

                if let Some(out) = output {
                    if let Some(name) = self
                        .wl_outputs
                        .iter()
                        .find_map(|(id, o)| (o == &out).then_some(id.clone()))
                    {
                        self.output_redraw.entry(name).or_default().queue_redraw();
                    }
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
        wl_output: smithay::reexports::wayland_server::protocol::wl_output::WlOutput,
    ) {
        // A late-bound wl_output needs retroactive enter events on the
        // foreign-toplevel handles (toplevel output_enter) and ext-workspace
        // groups (group output_enter) the same client already holds.
        crate::foreign_toplevel::on_output_bound(self, &output, &wl_output);
        crate::ext_workspace::on_output_bound(self, &output, &wl_output);

        // Logged at info so the integration test can confirm clients
        // see our wl_output advertisements.
        tracing::info!(connector = %output.name(), "client bound wl_output");
    }
}

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

    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        // The focused client set its cursor — via `wl_pointer.set_cursor`
        // (a Surface, or Hidden when it passes a null surface) or
        // `wp_cursor_shape` (a Named icon, which smithay funnels through
        // here). Stash it and re-resolve the sprite now, so the cursor
        // changes even with the pointer stationary (hovering a link / text).
        // The `kind` log is a breadcrumb for "does app X hide on keystroke?"
        // — apps hide by passing a null surface (⇒ Hidden), there is no
        // separate hide protocol.
        let kind = match &image {
            CursorImageStatus::Hidden => "hidden",
            CursorImageStatus::Named(_) => "named",
            CursorImageStatus::Surface(_) => "surface",
        };
        tracing::debug!(kind, "client set cursor image");
        self.cursor_manager.set_cursor_image(image);
        self.cursor_dirty = true;
        update_output_cursors(self);
    }
    // led_state_changed defaults to a no-op.
}

// wp_cursor_shape attaches shape devices to both pointers and tablet tools,
// so its delegate requires TabletSeatHandler even though we have no tablet
// support yet. The default `tablet_tool_image` (a no-op) is all we need; a
// tool's cursor will wire through here if/when tablet support lands.
impl smithay::wayland::tablet_manager::TabletSeatHandler for PrismState {}

// ─── zwp_relative_pointer / zwp_pointer_constraints ──────────────────────────

// Relative pointer needs no handler trait — the manager just gates the global,
// and the per-motion deltas are pushed from the input layer via
// `PointerHandle::relative_motion`.

impl PointerConstraintsHandler for PrismState {
    fn new_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {
        // A constraint can only activate while the pointer is focused on its
        // surface and inside any requested region. `pointer_contents` is kept
        // current by the motion handlers and the post-dispatch focus refresh,
        // so a constraint created while the pointer is already inside (the
        // normal case for click-to-lock games) activates immediately.
        self.maybe_activate_pointer_constraint();
    }

    fn remove_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {
        // smithay added this hook (df73da71) so the compositor can apply a
        // pending cursor-position hint at the moment the lock/confinement is
        // torn down — that's what anvil now does, having moved hint handling
        // out of `cursor_position_hint`. prism keeps the *eager* model
        // instead: `cursor_position_hint` repositions the pointer the instant
        // the client commits the hint (load-bearing for the captured-game
        // escape fix, 8f204d6 — see the clamp comment below), so by the time
        // the constraint is removed the pointer is already where it should be.
        // Nothing to defer here.
    }

    fn cursor_position_hint(
        &mut self,
        surface: &WlSurface,
        pointer: &PointerHandle<Self>,
        location: smithay::utils::Point<f64, smithay::utils::Logical>,
    ) {
        // The client hints where the cursor should reappear once a lock is
        // released. Only honor it while the constraint is actually active and
        // the hint surface is the one under the pointer (we need its origin).
        let is_active =
            with_pointer_constraint(surface, pointer, |c| c.is_some_and(|c| c.is_active()));
        if !is_active {
            return;
        }
        let Some((under, origin)) = self.pointer_contents.clone() else {
            return;
        };
        if &under != surface {
            return;
        }
        // `location` is surface-local; `origin` is the surface origin in global
        // logical space. prism tracks the pointer position itself, so move it
        // directly — the next motion event syncs smithay's internal location.
        let mut target = origin + location;
        // Clamp into the surface's output, with the half-open correction:
        // integer output sizes are exclusive on the right/bottom edge. A locked
        // game (e.g. Proton/Wine) continuously hints the cursor toward the
        // window border; an unclamped hint at the right/bottom edge lands one
        // logical pixel *outside* the output, so the next focus refresh finds no
        // surface there, sends `wl_surface.leave`, and smithay deactivates the
        // lock — the cursor then escapes onto the neighbouring monitor. Pinning
        // to `output_loc + size - 1` keeps the hotspot on the locked output (the
        // cursor image may still overflow the edge), which is what niri does in
        // its `cursor_position_hint` (handlers/mod.rs: "i32 sizes are exclusive,
        // but f64 sizes are inclusive").
        let pp = self.pointer_pos;
        if let Some(out_id) = self.output_containing((pp.x as i32, pp.y as i32)) {
            if let Some(out) = self.wl_outputs.get(&out_id) {
                if let Some((w, h)) = output_logical_size(out) {
                    let loc = out.current_location();
                    target.x = target.x.clamp(loc.x as f64, (loc.x + w - 1) as f64);
                    target.y = target.y.clamp(loc.y as f64, (loc.y + h - 1) as f64);
                }
            }
        }
        self.pointer_pos = target;
        update_output_cursors(self);
    }
}

// ─── ext-idle-notify-v1 / zwp_idle_inhibit ───────────────────────────────────

impl IdleNotifierHandler for PrismState {
    fn idle_notifier_state(&mut self) -> &mut IdleNotifierState<Self> {
        // Built in set_loop_handle, before the wayland socket is inserted,
        // so the global (and thus this getter) can't be reached earlier.
        self.idle_notifier
            .as_mut()
            .expect("idle notifier built in set_loop_handle before clients connect")
    }
}

impl IdleInhibitHandler for PrismState {
    fn inhibit(&mut self, surface: WlSurface) {
        tracing::debug!(surface = ?surface.id(), "idle-inhibit: client created an inhibitor");
        self.idle_inhibitors.insert(surface);
        self.refresh_idle_inhibit();
    }

    fn uninhibit(&mut self, surface: WlSurface) {
        tracing::debug!(surface = ?surface.id(), "idle-inhibit: client removed an inhibitor");
        self.idle_inhibitors.remove(&surface);
        self.refresh_idle_inhibit();
    }
}

// ─── wp_viewporter ──────────────────────────────────────────────────────────

// No handler trait required — smithay stores per-surface viewport
// state in SurfaceData::cached_state; we'd read it via with_states +
// ViewportCachedState if/when we honor it in the render path.

// ─── wp_presentation_time ───────────────────────────────────────────────────

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
        // `RendererSurfaceState` populated above. (Subsurface descendants are
        // refreshed after output resolution below — only the ones whose content
        // actually advanced this commit.)
        process_surface_buffer(self, surface);

        // Layer-shell surfaces re-arrange their output's LayerMap on commit
        // (so anchor / size / margin / exclusive-zone changes take effect)
        // and get their initial configure here. Gated on the layer-surface
        // role so subsurface commits of a layer don't re-trigger it.
        if let Some("zwlr_layer_surface_v1") = role {
            self.layer_shell_commit(surface);
        }

        // Session-lock surfaces: while WaitingForSurfaces, a newly
        // mapped lock surface may complete the per-output set and
        // advance the lock; afterwards a commit repaints that output's
        // lock screen.
        if let Some("ext_session_lock_surface_v1") = role {
            self.session_lock_surface_commit(surface);
        }

        // If this commit is on the current cursor surface (the client updated
        // its cursor buffer — e.g. an animated cursor's next frame), re-upload
        // the sprite. `on_commit_buffer_handler` above already refreshed its
        // RendererSurfaceState, so update_output_cursors reads the new buffer.
        let is_cursor_surface = matches!(
            self.cursor_manager.cursor_image(),
            CursorImageStatus::Surface(s) if s == surface
        );
        if is_cursor_surface {
            self.cursor_dirty = true;
            update_output_cursors(self);
        }

        // DnD icon surface (or one of its subsurfaces). The icon isn't a
        // layout window or layer surface, so the placement-driven
        // `queue_redraw_for_surface` at the bottom can't resolve it to an
        // output — queue the repaint here, on the output under the pointer
        // (the only place the render walk draws it). A commit on the icon
        // surface itself may also carry a wl_surface.offset delta that
        // moves the icon relative to the cursor; accumulate it. Mirrors
        // niri handlers/compositor.rs:438.
        {
            let mut icon_root = surface.clone();
            while let Some(parent) = get_parent(&icon_root) {
                icon_root = parent;
            }
            if matches!(&self.dnd_icon, Some(icon) if icon.surface == icon_root) {
                let icon = self.dnd_icon.as_mut().unwrap();
                if surface == &icon.surface {
                    with_states(surface, |states| {
                        let buffer_delta = states
                            .cached_state
                            .get::<smithay::wayland::compositor::SurfaceAttributes>()
                            .current()
                            .buffer_delta
                            .take()
                            .unwrap_or_default();
                        icon.offset += buffer_delta;
                    });
                }
                if let Some(id) =
                    self.output_containing((self.pointer_pos.x as i32, self.pointer_pos.y as i32))
                {
                    self.output_redraw.entry(id).or_default().queue_redraw();
                }
            }
        }

        // xdg-shell toplevel lifecycle. A toplevel lives in
        // `unmapped_windows` from `new_toplevel` until its first buffer
        // commit. The first (buffer-less) commit triggers the initial
        // configure — sized by the layout and recorded on the unmapped
        // record (see `send_initial_configure`) — so the client draws
        // its first buffer at the size the layout intends. The first
        // commit that attaches a buffer maps the window into the layout
        // using that recorded state. Mirrors niri's
        // handlers/compositor.rs commit flow.
        if let Some("xdg_toplevel") = role {
            if self.unmapped_windows.contains_key(surface) {
                // Map readiness is read off the SurfaceTexSlot:
                // process_surface_buffer (called above) consumes any
                // BufferAssignment::NewBuffer out of cached_state and
                // populates this slot with a SurfaceTexture on success.
                // So if the slot is now Some, the client has produced
                // its first renderable frame and is ready to be mapped.
                // (niri checks the attached buffer directly.)
                let has_texture = with_states(surface, |states| {
                    states
                        .data_map
                        .get::<SurfaceTexSlot>()
                        .map(|s| s.0.lock().unwrap().is_some())
                        .unwrap_or(false)
                });
                if has_texture {
                    // The toplevel got mapped.
                    self.map_unmapped_toplevel(surface);
                } else {
                    // The toplevel remains unmapped: send the initial
                    // configure if it hasn't been sent yet. Deferred to
                    // an idle callback so the client can supply more
                    // info (title, app_id, min/max size) after this
                    // commit — window rules and the open-floating
                    // heuristics match on those.
                    let unmapped = &self.unmapped_windows[surface];
                    if unmapped.needs_initial_configure() {
                        let toplevel = unmapped.toplevel().clone();
                        self.queue_initial_configure(toplevel);
                    }
                }
            } else if let Some((mapped, output)) = self.layout.find_window_and_output(surface) {
                let window = mapped.window.clone();
                let output = output.cloned();

                // Null-buffer unmap. process_surface_buffer (above)
                // drops the SurfaceTexSlot on BufferAssignment::Removed,
                // so an empty slot on a mapped root surface means this
                // commit unmapped the toplevel (test client:
                // wleird-unmap; real clients unmap to remap later).
                // Previously only toplevel_destroyed removed windows, so
                // this left a permanent invisible tile holding a column.
                let has_texture = with_states(surface, |states| {
                    states
                        .data_map
                        .get::<SurfaceTexSlot>()
                        .map(|s| s.0.lock().unwrap().is_some())
                        .unwrap_or(false)
                });
                if !has_texture {
                    // The toplevel got unmapped. Mirrors niri
                    // handlers/compositor.rs ("toplevel got unmapped"):
                    // start the close animation BEFORE on_commit — the
                    // snapshot must record the pre-unmap geometry, not
                    // the refreshed (now bufferless) bbox.
                    let transaction = prism_layout::utils::transaction::Transaction::new();
                    self.layout.store_unmap_snapshot(&window);
                    self.layout
                        .start_close_animation_for_window(&window, transaction.blocker());
                    window.on_commit();
                    self.layout.remove_window(&window, transaction.clone());
                    // If neighbours in a co-resize hold clones, cap how
                    // long they can wait on this one (niri does the same).
                    if !transaction.is_last() {
                        if let Some(loop_handle) = self.loop_handle.as_ref() {
                            transaction.register_deadline_timer(loop_handle);
                        }
                    }

                    // Back to the pre-map stage: a remap re-runs window
                    // rules + the initial commit-configure sequence
                    // afresh, exactly like a brand-new toplevel.
                    self.unmapped_windows
                        .insert(surface.clone(), Unmapped::new(window));
                    tracing::info!(
                        surface_id = ?surface.id(),
                        "xdg_toplevel unmapped (null buffer); back to pre-map stage"
                    );

                    // Render is damage-driven; repaint the vacated
                    // region (same fallback policy as toplevel_destroyed).
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
                    return;
                }

                // Re-commit on an already-mapped window.
                //
                // First refresh the smithay Window's cached bbox from
                // the newly-committed surface tree. Without this,
                // `Window::geometry()` returns the bbox at the time
                // of the *previous* commit (or empty on the first),
                // so all the downstream size readers — including
                // `Tile::tile_size()` / `Column::width()` — see
                // stale dimensions. Mirrors niri/src/handlers/compositor.rs:90.
                window.on_commit();

                // Then forward through to the layout so it can update
                // its per-tile/per-column size record (ColumnData /
                // TileData) from the now-fresh window geometry.
                //
                // Mirrors niri/src/handlers/compositor.rs:346:
                //   self.niri.layout.update_window(&window, serial);
                //
                // The acked configure serial is threaded through so the
                // layout can match this commit to the configure it
                // responded to. `Mapped::on_commit(serial)` retires
                // acked maximized / windowed-fullscreen / interactive-
                // resize state, and (via the pre-commit snapshot stored
                // above) lets `Tile::update_window` start the resize
                // animation. `None` would be the bare "resync geometry"
                // path with no animation.
                let acked = with_states(surface, |states| {
                    states
                        .data_map
                        .get::<XdgToplevelSurfaceData>()
                        .and_then(|d| d.lock().unwrap().last_acked.as_ref().map(|c| c.serial))
                });
                self.layout.update_window(&window, acked);
            }
        }

        // xdg_popup: advance the popup tree's committed state (re-resolves
        // the positioner geometry against the latest parent geometry) and
        // send the initial configure on the first commit so the client can
        // attach a buffer and map. Mirrors the toplevel initial-configure
        // dance above; `PopupManager::commit` is the popup analogue of the
        // layout's `update_window`.
        if let Some("xdg_popup") = role {
            self.popups.commit(surface);
            if let Some(PopupKind::Xdg(ref xdg)) = self.popups.find_popup(surface) {
                if !xdg.is_initial_configure_sent() {
                    // PopupSurface::send_configure only errors if the
                    // positioner violated the protocol's constraint-
                    // adjustment rules; with an untouched positioner it
                    // can't, so surface the error loudly if it ever fires.
                    if let Err(err) = xdg.send_configure() {
                        tracing::warn!(?err, "failed to send initial xdg_popup configure");
                    }
                }
            }
        }

        // Surface→output assignment + wl_surface.enter/leave. Runs after
        // both process_surface_buffer (in case the new buffer is what
        // produced a layout-visible window) and the optional add_window
        // above, so the layout has the authoritative answer by the time we
        // ask. Also re-runs on every commit, which handles the layout
        // moving a window between outputs.
        dispatch_surface_output_from_layout(self, surface);

        // Materialize this surface's texture on the GPU(s) that display it
        // (consumer set from the placement resolved just above), and do the
        // per-commit refresh (mirror copies, shm re-uploads). Deferred to
        // here so the consumer-GPU set is known; a window on a single
        // monitor only ever imports on that monitor's GPU.
        ensure_surface_textures(self, surface);
        // Record this surface's refreshed version so the gated descendant pass
        // (here or on a future ancestor commit) doesn't redundantly refresh it.
        mark_texture_refreshed(surface);

        // Refresh subsurface descendants whose content advanced on this commit.
        // `on_commit_buffer_handler` applies *synchronized* subsurfaces' cached
        // buffers (advancing their commit counter) on the parent's commit, so
        // their GPU import must refresh here or the render walk samples a stale
        // buffer (the bug behind partially-updating Firefox widgets). But doing
        // it for the *whole* tree every commit re-armed work for unchanged
        // children — each `ensure_surface_textures` re-clears `acquire_waited`
        // (re-importing the client's implicit fence next render) and repeats a
        // layout lookup. So gate per child on its content version actually
        // advancing. The commit counter is the robust signal: a buffer-identity
        // check would miss same-`wl_buffer` content changes (in-place dmabuf
        // damage, shm reuse) and leave those children stale.
        for s in surface_tree_surfaces(surface) {
            if &s != surface {
                refresh_subsurface_texture_if_changed(self, &s);
            }
        }

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
        // Drop the dmabuf source if this buffer was a dmabuf. The per-GPU
        // materializations live on surfaces' SurfaceTexSlots and are
        // replaced when the surface commits a different buffer; shm buffers
        // were never in this map, so this is a no-op for them.
        self.dmabuf_sources.remove(&buffer.id());
    }
}

// ─── wl_shm ─────────────────────────────────────────────────────────────────

impl ShmHandler for PrismState {
    fn shm_state(&self) -> &ShmState {
        &self.shm
    }
}

// ─── xdg-shell ──────────────────────────────────────────────────────────────

impl PrismState {
    /// Reposition `popup`'s pending geometry so it stays within its parent's
    /// on-screen working area, honoring the client's positioner
    /// constraint_adjustment (flip / slide / resize). Mirrors niri's
    /// `unconstrain_popup`, dispatching on the popup's root: layout windows
    /// here, layer-shell surfaces (popups bound via
    /// `zwlr_layer_surface_v1.get_popup`) in
    /// [`Self::unconstrain_layer_popup`].
    pub(crate) fn unconstrain_popup(&self, popup: &PopupKind) {
        let Ok(root) = find_popup_root_surface(popup) else {
            return;
        };
        let Some(window) = self
            .layout
            .find_window_and_output(&root)
            .map(|(mapped, _)| mapped.window.clone())
        else {
            // Not a window popup — the root may be a layer surface (a bar's
            // tray popover). No-op if it's neither.
            self.unconstrain_layer_popup(popup, &root);
            return;
        };

        // `popup_target_rect` is relative to the parent window's geometry;
        // shift it into the popup's own coordinate space (the positioner
        // anchors against the parent, and the popup's own toplevel-coords
        // offset must be subtracted to compare in the same frame).
        let mut target = self.layout.popup_target_rect(&window);
        target.loc -= get_popup_toplevel_coords(popup).to_f64();

        let PopupKind::Xdg(popup) = popup else {
            return;
        };
        popup.with_pending_state(|state| {
            state.geometry = unconstrain_with_padding(state.positioner, target);
        });
    }

    /// Resolve a (sub)surface to its root shell surface: subsurfaces walk
    /// up the parent chain; popups follow the xdg_popup parent chain to
    /// their toplevel (or layer) surface. Mirrors niri's
    /// `find_root_shell_surface` (niri.rs:6125) — the layout only knows
    /// toplevel root surfaces, so resolve before querying it.
    pub fn find_root_shell_surface(&self, surface: &WlSurface) -> WlSurface {
        let mut root = surface.clone();
        while let Some(parent) = get_parent(&root) {
            root = parent;
        }
        if let Some(popup) = self.popups.find_popup(&root) {
            return find_popup_root_surface(&popup).unwrap_or(root);
        }
        root
    }
}

/// Unconstrain `positioner` against `target`, preferring an 8px inset (nicer
/// looking) and falling back to the full target if the padded fit fails.
/// Ported from niri's `unconstrain_with_padding`.
fn unconstrain_with_padding(
    positioner: PositionerState,
    target: Rectangle<f64, Logical>,
) -> Rectangle<i32, Logical> {
    const PADDING: f64 = 8.;

    let mut padded = target;
    if PADDING * 2. < padded.size.w {
        padded.loc.x += PADDING;
        padded.size.w -= PADDING * 2.;
    }
    if PADDING * 2. < padded.size.h {
        padded.loc.y += PADDING;
        padded.size.h -= PADDING * 2.;
    }

    // Too small to pad — unconstrain against the raw target.
    if padded == target {
        return positioner.get_unconstrained_geometry(target.to_i32_round());
    }

    // Try the padded target without allowing a resize (resizing to fit the
    // inset would defeat the cosmetic padding).
    let mut no_resize = positioner;
    no_resize
        .constraint_adjustment
        .remove(ConstraintAdjustment::ResizeX);
    no_resize
        .constraint_adjustment
        .remove(ConstraintAdjustment::ResizeY);
    let geo = no_resize.get_unconstrained_geometry(padded.to_i32_round());
    if padded.contains_rect(geo.to_f64()) {
        return geo;
    }

    // Padded fit failed; fall back to the full target.
    positioner.get_unconstrained_geometry(target.to_i32_round())
}

impl XdgShellHandler for PrismState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        tracing::info!(
            surface_id = ?surface.wl_surface().id(),
            "new xdg_toplevel"
        );
        // Track the window as unmapped until its first buffer commit.
        // The initial configure is sent from the commit handler (so the
        // client has a chance to set title / app_id first); the rules +
        // size resolved there are stored on this record and consumed
        // when the window maps. Mirrors niri's `new_toplevel`.
        let wl_surface = surface.wl_surface().clone();
        let window = Window::new_wayland_window(surface);
        let existing = self
            .unmapped_windows
            .insert(wl_surface, Unmapped::new(window));
        if existing.is_some() {
            tracing::error!("new_toplevel got called for an existing window");
        }
    }

    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        // Set the popup's pending geometry, unconstrained against the
        // parent's on-screen working area so it stays visible (flip/slide/
        // resize per the client's positioner constraint_adjustment). This is
        // sent with the initial configure on first commit. smithay has
        // already stashed the positioner in the popup's pending state by the
        // time this fires, so `unconstrain_popup` reads it back (we don't
        // need the `_positioner` arg).
        self.unconstrain_popup(&PopupKind::Xdg(surface.clone()));
        if let Err(err) = self.popups.track_popup(PopupKind::Xdg(surface)) {
            tracing::warn!(?err, "failed to track new xdg_popup");
        }
    }

    fn grab(&mut self, surface: PopupSurface, seat: WlSeat, serial: Serial) {
        // A client requests an explicit popup grab for a menu: while the
        // grab is active, pointer + keyboard events route through the popup
        // tree, and a press outside the grabbing client dismisses it.
        //
        // Note: many clients (e.g. Firefox/GTK) manage their menus WITHOUT a
        // grab, driving dismissal off the pointer-leave they receive on the
        // parent toplevel; this path only runs for clients that do request a
        // grab. It's still needed for those that do.
        let seat: Seat<Self> = Seat::from_resource(&seat).expect("seat from this display");
        let kind = PopupKind::Xdg(surface);
        let Ok(root) = find_popup_root_surface(&kind) else {
            return;
        };

        // Locked session: only the lock client's own popups may grab —
        // a session window's menu must not pull keyboard focus off the
        // lock screen (niri handlers/xdg_shell.rs:283).
        if self.is_locked() && Some(&root) != self.lock_surface_focus().as_ref() {
            tracing::trace!("ignoring popup grab: session is locked");
            let _ = smithay::desktop::PopupManager::dismiss_popup(&root, &kind);
            return;
        }

        let mut grab = match self.popups.grab_popup(root, kind, &seat, serial) {
            Ok(grab) => grab,
            Err(err) => {
                tracing::warn!(?err, "failed to start popup grab");
                return;
            }
        };

        // Hand the keyboard to the grab. If some unrelated grab already
        // holds the keyboard (and it isn't this grab's chain), bail and
        // dismiss rather than stomping it — mirrors anvil's guard.
        if let Some(keyboard) = seat.get_keyboard() {
            if keyboard.is_grabbed()
                && !(keyboard.has_grab(serial)
                    || keyboard.has_grab(grab.previous_serial().unwrap_or(serial)))
            {
                grab.ungrab(PopupUngrabStrategy::All);
                return;
            }
            keyboard.set_focus(self, grab.current_grab(), serial);
            keyboard.set_grab(self, PopupKeyboardGrab::new(&grab), serial);
        }

        // Same for the pointer. `Focus::Keep` leaves the current pointer
        // focus in place; the grab's own motion handler takes over routing.
        if let Some(pointer) = seat.get_pointer() {
            if pointer.is_grabbed()
                && !(pointer.has_grab(serial)
                    || pointer.has_grab(grab.previous_serial().unwrap_or_else(|| grab.serial())))
            {
                grab.ungrab(PopupUngrabStrategy::All);
                return;
            }
            pointer.set_grab(self, PopupPointerGrab::new(&grab), serial, Focus::Keep);
        }
    }

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        // xdg_popup.reposition: recompute geometry from the new positioner,
        // unconstrain it against the parent's working area, then echo the
        // token back via the repositioned event so the client can correlate.
        // Used by menus that re-anchor (e.g. a submenu that would overflow).
        // Store the new positioner so `unconstrain_popup` adjusts against it.
        // Redraw is queued from the subsequent commit.
        surface.with_pending_state(|state| {
            state.geometry = positioner.get_geometry();
            state.positioner = positioner;
        });
        self.unconstrain_popup(&PopupKind::Xdg(surface.clone()));
        surface.send_repositioned(token);
    }

    // ─── Client-initiated state requests ────────────────────────────────
    //
    // These are the requests behind a client's own titlebar / window
    // buttons (mpv's fullscreen button, a GTK app's maximize button,
    // etc.) — `xdg_toplevel.set_fullscreen` and friends. Without these
    // overrides smithay's default no-op runs and the button does nothing;
    // only the compositor keybinds (which drive the layout directly via
    // `actions.rs`) would work. We resolve the toplevel's surface back to
    // its layout window and call the same directional layout API the
    // keybinds use. `set_*` (not `toggle_*`) because these requests are
    // directional, not toggles. `set_fullscreen` / `set_maximized` push a
    // fresh configure to the client themselves, so we only owe a redraw.

    fn fullscreen_request(&mut self, surface: ToplevelSurface, output: Option<WlOutput>) {
        // A pre-map request (mpv --fullscreen, games): record it on the
        // unmapped record so the initial configure carries the
        // fullscreen state (and targets the requested output). Only the
        // not-yet-configured state is handled; a request landing between
        // initial configure and map is rare enough to skip (niri
        // re-configures in that window — port if it bites).
        if let Some(unmapped) = self.unmapped_windows.get_mut(surface.wl_surface()) {
            if let InitialConfigureState::NotConfigured {
                wants_fullscreen, ..
            } = &mut unmapped.state
            {
                *wants_fullscreen = Some(output.as_ref().and_then(Output::from_resource));
            }
            return;
        }
        // `output` for mapped windows: a client may request fullscreen
        // on a specific monitor. We fullscreen on the window's current
        // output, matching the keybind behaviour; honouring a
        // cross-output target would mean moving the window first and
        // isn't wired up yet.
        self.set_window_fullscreen(&surface, true);
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        if let Some(unmapped) = self.unmapped_windows.get_mut(surface.wl_surface()) {
            if let InitialConfigureState::NotConfigured {
                wants_fullscreen, ..
            } = &mut unmapped.state
            {
                *wants_fullscreen = None;
            }
            return;
        }
        self.set_window_fullscreen(&surface, false);
    }

    fn maximize_request(&mut self, surface: ToplevelSurface) {
        if let Some(unmapped) = self.unmapped_windows.get_mut(surface.wl_surface()) {
            if let InitialConfigureState::NotConfigured {
                wants_maximized, ..
            } = &mut unmapped.state
            {
                *wants_maximized = true;
            }
            return;
        }
        self.set_window_maximized(&surface, true);
    }

    fn unmaximize_request(&mut self, surface: ToplevelSurface) {
        if let Some(unmapped) = self.unmapped_windows.get_mut(surface.wl_surface()) {
            if let InitialConfigureState::NotConfigured {
                wants_maximized, ..
            } = &mut unmapped.state
            {
                *wants_maximized = false;
            }
            return;
        }
        self.set_window_maximized(&surface, false);
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        // The toplevel may die before ever mapping (client crashed, or
        // closed its window before attaching a buffer) — drop the
        // unmapped record and we're done.
        if self.unmapped_windows.remove(surface.wl_surface()).is_some() {
            return;
        }

        // Pop the window out of the layout so the columns behind it
        // can fall into the freed slot. Without this the layout keeps
        // a tile for a window whose surface is gone — invisible (no
        // texture) but still occupying a column, which manifests as
        // "I closed the middle window and the third one didn't slide
        // back."
        //
        // Mirrors niri's `Layout::remove_window` call from its
        // unmap path. A fresh `Transaction::new()` is the "don't
        // coordinate with anyone" default — the surface is already
        // destroyed, so there is no commit to gate (unlike the
        // null-buffer unmap path in the commit handler, which
        // registers its transaction's deadline for co-resize
        // neighbours).
        let wl_surface = surface.wl_surface();
        let lookup = self
            .layout
            .find_window_and_output(wl_surface)
            .map(|(mapped, out)| (mapped.window.clone(), out.cloned()));
        if let Some((window, output)) = lookup {
            // Start the close (shrink + fade) animation before removing the
            // window: record the tile's geometry, then spin up a ClosingWindow
            // that replays the window's last frame (captured from the
            // intermediate at the next render). Mirrors niri's unmap path.
            // An already-completed blocker = no cross-window transaction gating.
            self.layout.store_unmap_snapshot(&window);
            self.layout.start_close_animation_for_window(
                &window,
                prism_layout::utils::transaction::TransactionBlocker::completed(),
            );
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

    fn app_id_changed(&mut self, surface: ToplevelSurface) {
        self.update_window_rules(&surface);
    }

    fn title_changed(&mut self, surface: ToplevelSurface) {
        self.update_window_rules(&surface);
    }
}

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
        // Don't import to any GPU here. A wl_buffer is created
        // surface-agnostically (create_immed), so we don't yet know which
        // output — hence which GPU(s) — will display it. Instead keep a
        // GPU-agnostic source description (dup'd fds); the surface that
        // ends up showing this buffer drives a lazy per-GPU import in
        // `ensure_surface_textures`. GPUs that can't read the modifier get
        // a cross-GPU mirror there rather than blank.
        if self.gpus.is_empty() {
            tracing::warn!("dmabuf import: no GPUs registered, rejecting");
            notifier.failed();
            return;
        }

        let src = match prism_frame::Dmabuf::from_smithay(&dmabuf) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("dmabuf rejected: Dmabuf::from_smithay failed: {e:#}");
                notifier.failed();
                return;
            }
        };

        // Validate up front: at least one GPU must be able to import this
        // (format, modifier). Otherwise the buffer is unusable — reject so the
        // client falls back instead of rendering blank forever.
        //
        // CRITICAL: this must accept everything `build_advertised_dmabuf_formats`
        // advertised, or a client using `create_immed` (Firefox does) gets a
        // fatal `invalid_wl_buffer` protocol error and crashes. YUV (NV12/P010)
        // is multi-planar — it has no single `vk_format_for` mapping — so we
        // validate its luma+chroma plane formats; `import_dmabuf` then does the
        // real two-plane import via `import_yuv`. Without this, prism advertised
        // NV12 but rejected every NV12 buffer here → Firefox crash / green.
        let modifier = u64::from(src.modifier);
        let importable = if let Some(kind) = yuv_kind_for(src.format) {
            let (luma_fmt, chroma_fmt) = kind.plane_formats();
            self.gpus.values().any(|d| {
                gpu_supports_dmabuf(d, luma_fmt, modifier)
                    && gpu_supports_dmabuf(d, chroma_fmt, modifier)
            })
        } else {
            let Some(vk_format) = vk_format_for(src.format) else {
                tracing::warn!(fmt = ?src.format, "dmabuf rejected: no Vulkan format mapping");
                notifier.failed();
                return;
            };
            self.gpus
                .values()
                .any(|d| gpu_supports_dmabuf(d, vk_format, modifier))
        };
        if !importable {
            tracing::warn!(
                fmt = ?src.format,
                modifier = format!("{modifier:#x}"),
                "dmabuf rejected: no GPU supports this format+modifier for SAMPLED"
            );
            notifier.failed();
            return;
        }

        match notifier.successful::<PrismState>() {
            Ok(buffer) => {
                // `trace`, not `info`: this fires once per client dmabuf buffer
                // *creation* (create_immed), which a tiling client like Firefox
                // does hundreds of times per second — at `info` the synchronous
                // formatting + write was a measurable smoothness drag. Format /
                // modifier negotiation is already visible at startup
                // ("dmabuf advertised fourccs" / "candidate format").
                tracing::trace!(
                    w = src.width,
                    h = src.height,
                    fmt = ?src.format,
                    modifier = format!("{modifier:#x}"),
                    "accepted client dmabuf source (lazy per-GPU import)"
                );
                self.dmabuf_sources
                    .insert(buffer.id(), Arc::new(DmabufSourceEntry::new(src)));
            }
            Err(_) => {
                tracing::warn!("dmabuf successful() failed — client may be dead");
            }
        }
    }
}

// ─── wlr_layer_shell ────────────────────────────────────────────────────────
// Handler impl lives in `crate::layer_shell`; this just hooks the
// smithay delegate macro so dispatch routes here. The macro generates
// GlobalDispatch + Dispatch impls for the manager + per-surface
// interfaces.

// ─── fractional_scale / single_pixel_buffer / content_type ──────────────────
// Advertise-only today. The handlers are no-op shims so smithay's
// delegate macros can find the trait impl; protocol bookkeeping is
// entirely on smithay's side. See field docs on PrismState for the
// follow-up wiring each needs.

impl FractionalScaleHandler for PrismState {}

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

/// 10s activation-token window — matches niri's
/// `XDG_ACTIVATION_TOKEN_TIMEOUT` and the typical "user just clicked"
/// interval. Older tokens are stale (the user has moved on); silently
/// drop. Checked both at `request_activation` and again at map time
/// (`map_unmapped_toplevel`), in case the window took a while between
/// requesting activation and mapping.
const TOKEN_TIMEOUT_SECS: u64 = 10;

impl XdgActivationHandler for PrismState {
    fn activation_state(&mut self) -> &mut XdgActivationState {
        &mut self.xdg_activation
    }

    fn token_created(&mut self, _token: XdgActivationToken, data: XdgActivationTokenData) -> bool {
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
            if kb
                .last_enter()
                .is_some_and(|le| serial.is_no_older_than(&le))
            {
                return true;
            }
        }
        if let Some(ptr) = seat.get_pointer() {
            if ptr
                .last_enter()
                .is_some_and(|le| serial.is_no_older_than(&le))
            {
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
            } else if let Some(unmapped) = self.unmapped_windows.get_mut(&surface) {
                // Surface not yet mapped: queue the token on the
                // unmapped-window record. The map path re-checks the
                // timestamp and activates the window on its first
                // commit — covers the common case where the
                // just-spawned client's window arrives milliseconds
                // after the activation token. Mirrors niri
                // (handlers/mod.rs).
                unmapped.activation_token_data = Some(token_data);
            }
        }
        self.xdg_activation.remove_token(&token);
    }
}

// xdg-dialog: we only need the global advertised and the hint tracked (smithay
// stores it on the toplevel role). The dialog/modal hint is consumed at window
// open time in `compute_open_floating`, so the `dialog_hint_changed` default
// no-op is all we need here.
impl XdgDialogHandler for PrismState {}

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
/// Build one reusable [`MirrorCopier`] per registered GPU, for the
/// cross-GPU mirror path. A copier that fails to construct is simply
/// omitted (logged) — the mirror path for that GPU then can't run, but
/// native imports and single-GPU configs are unaffected.
fn build_mirror_copiers(
    gpus: &HashMap<DrmDevId, Arc<prism_renderer::Device>>,
) -> HashMap<DrmDevId, prism_renderer::MirrorCopier> {
    let mut copiers = HashMap::new();
    for (&gpu_id, device) in gpus {
        match prism_renderer::MirrorCopier::new(device.clone()) {
            Ok(c) => {
                copiers.insert(gpu_id, c);
            }
            Err(e) => tracing::warn!(gpu = ?gpu_id, "mirror copier init failed: {e:#}"),
        }
    }
    copiers
}

/// Whether `device`'s driver can import a single-plane SAMPLED image with
/// this `(vk_format, modifier)`. This is the guard that keeps a bad buffer
/// off the GPU.
///
/// The case that matters most is `modifier == DRM_FORMAT_MOD_INVALID`
/// (u64::MAX): a client that allocated without an explicit modifier
/// (legacy GBM). On modern AMD the real layout is tiled, but
/// `ImageDrmFormatModifierExplicitCreateInfoEXT` would take u64::MAX at
/// face value, build a garbage-tiled image, and page-fault the GPU on
/// first sample — a *hard* recovery that wedges the card. Invalid is never
/// in the driver's reported modifier list, so this check rejects it. It's
/// also how we decide native-vs-mirror: a GPU that returns false here gets
/// a cross-GPU mirror instead of a native import.
fn gpu_supports_dmabuf(
    device: &prism_renderer::Device,
    vk_format: vk::Format,
    modifier: u64,
) -> bool {
    device
        .supported_drm_format_modifiers(vk_format)
        .iter()
        .any(|m| {
            m.modifier == modifier
                && m.plane_count == 1
                && m.tiling_features
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        })
}

/// Import a client dmabuf as a zero-copy `VkImage` on `device`. Caller must
/// have already confirmed the modifier is supported (via
/// [`gpu_supports_dmabuf`]).
///
/// `for_sampling`: when true, transition the image to
/// `SHADER_READ_ONLY_OPTIMAL` so the render path can sample it (needed for
/// native consumer textures and the mirror's LINEAR target). When false the
/// image is left in `UNDEFINED` — used for a mirror's `home_src`, which is
/// only ever a copy *source*: the async copy transitions it to
/// `TRANSFER_SRC` itself, so the extra blocking transition submit is skipped
/// (it would stall the event loop on every commit of a non-pooling client).
fn import_dmabuf(
    device: &Arc<prism_renderer::Device>,
    dmabuf: &prism_frame::Dmabuf,
    for_sampling: bool,
) -> Result<prism_renderer::ImportedImage> {
    let image = if let Some(kind) = yuv_kind_for(dmabuf.format) {
        // Two-plane YUV video (NV12/P010): import luma + chroma as separate
        // single-plane images; the decode shader recombines them.
        prism_renderer::ImportedImage::import_yuv(
            device.clone(),
            dmabuf,
            kind,
            vk::ImageUsageFlags::SAMPLED,
        )
        .context("ImportedImage::import_yuv (SAMPLED)")?
    } else {
        let vk_format = vk_format_for(dmabuf.format)
            .with_context(|| format!("no Vulkan format mapping for {:?}", dmabuf.format))?;
        prism_renderer::ImportedImage::import(
            device.clone(),
            dmabuf,
            vk_format,
            vk::ImageUsageFlags::SAMPLED,
        )
        .context("ImportedImage::import (SAMPLED)")?
    };
    // Sampled dmabuf imports start in UNDEFINED layout but the render path
    // binds them as SHADER_READ_ONLY_OPTIMAL. Run the one-shot transition
    // here so the first frame's sample is legal — without this radv hangs
    // the queue on the first cmd_draw that touches the descriptor.
    if for_sampling {
        image
            .transition_for_sampling()
            .context("ImportedImage::transition_for_sampling")?;
    }
    Ok(image)
}

/// Collect `surface` and every subsurface beneath it (the full committed
/// surface tree) into a flat list. Used by the commit handler to refresh
/// textures across the whole tree: smithay's `on_commit_buffer_handler` applies
/// synchronized subsurfaces' buffers on the *parent's* commit, so the per-commit
/// texture refresh must reach those children too, not just the committed
/// surface. Read-only walk (no state borrow); ordering is irrelevant since each
/// surface is processed independently.
fn surface_tree_surfaces(surface: &WlSurface) -> Vec<WlSurface> {
    let mut out = Vec::new();
    with_surface_tree_downward(
        surface,
        (),
        |_, _, _| TraversalAction::DoChildren(()),
        |s, _, _| out.push(s.clone()),
        |_, _, _| true,
    );
    out
}

/// Per-surface record of the `RendererSurfaceState` commit counter at the last
/// texture refresh, so the commit handler can skip re-refreshing a subsurface
/// descendant whose content didn't advance. Stored in the surface's `data_map`
/// (independent of the `SurfaceTexture`, which is rebuilt on buffer swap).
#[derive(Default)]
struct LastTexRefreshCommit(Mutex<Option<CommitCounter>>);

/// Re-import/upload `surface`'s texture iff its content version advanced since
/// the last refresh. For subsurface descendants on a parent commit: smithay
/// applies synchronized children's buffers here (advancing their commit
/// counter), so a changed child must refresh in lockstep — but an *unchanged*
/// child must be left alone, or every parent commit re-arms its acquire-fence
/// wait and repeats a consumer-GPU layout lookup for nothing.
///
/// Gating on the commit counter (vs the imported buffer's identity) is what
/// makes this safe across every buffer-reuse pattern: a client that re-commits
/// the same `wl_buffer` with new pixels (in-place dmabuf damage, shm reuse)
/// advances the counter but keeps the buffer id, so an identity check would
/// wrongly skip it and leave the child stale.
fn refresh_subsurface_texture_if_changed(state: &mut PrismState, surface: &WlSurface) {
    // Decide under one lock: the current content version (None ⇒ no buffer
    // committed yet, nothing to refresh), and whether it differs from the last
    // version we refreshed. `Some(cur)` ⇒ refresh, then record `cur`.
    let to_refresh = with_states(surface, |states| -> Option<CommitCounter> {
        let cur = {
            let data = states.data_map.get::<RendererSurfaceStateUserData>()?;
            let guard = data.lock().unwrap();
            guard.view()?; // no view ⇒ no committed buffer
            guard.current_commit()
        };
        states
            .data_map
            .insert_if_missing_threadsafe(LastTexRefreshCommit::default);
        let last = states.data_map.get::<LastTexRefreshCommit>().unwrap();
        (*last.0.lock().unwrap() != Some(cur)).then_some(cur)
    });
    if to_refresh.is_none() {
        return;
    }

    process_surface_buffer(state, surface);
    ensure_surface_textures(state, surface);
    // Record the version we just refreshed at — after the work, so a failed
    // import re-attempts next commit rather than being marked done.
    mark_texture_refreshed(surface);
}

/// Record that `surface`'s texture is current as of its committed content
/// version, so [`refresh_subsurface_texture_if_changed`] skips it until it
/// advances again. Called both after a surface's own unconditional refresh in
/// `commit()` (so a desync subsurface that already refreshed itself isn't
/// re-refreshed by its parent's gated descendant pass) and after a gated child
/// refresh. No-op for a surface with no committed buffer.
fn mark_texture_refreshed(surface: &WlSurface) {
    with_states(surface, |states| {
        let cur = {
            let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() else {
                return;
            };
            let guard = data.lock().unwrap();
            if guard.view().is_none() {
                return;
            }
            guard.current_commit()
        };
        states
            .data_map
            .insert_if_missing_threadsafe(LastTexRefreshCommit::default);
        *states
            .data_map
            .get::<LastTexRefreshCommit>()
            .unwrap()
            .0
            .lock()
            .unwrap() = Some(cur);
    });
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

    // Set (or refresh) the surface's GPU-agnostic texture SOURCE under the
    // SurfaceData lock. No GPU work happens here — the per-GPU
    // materialization is deferred to `ensure_surface_textures`, which runs
    // after `dispatch_surface_output_from_layout` has resolved which
    // output (hence GPU) displays the surface. That source of truth is the
    // layout, not the buffer's logical_pos.
    with_states(surface, |states| {
        // `on_commit_buffer_handler` (called before us) took the
        // BufferAssignment out of cached_state and stashed it in
        // RendererSurfaceState. We read it back from there.
        let renderer_state = states.data_map.get::<RendererSurfaceStateUserData>();
        let current_buffer = renderer_state.and_then(|s| s.lock().unwrap().buffer().cloned());

        states
            .data_map
            .insert_if_missing_threadsafe(SurfaceTexSlot::default);
        let slot = states
            .data_map
            .get::<SurfaceTexSlot>()
            .expect("just inserted SurfaceTexSlot");

        let Some(buffer) = current_buffer else {
            // No buffer attached — initial commit, or unmapped this commit
            // (BufferAssignment::Removed). Drop our texture; smithay's
            // `InnerBuffer::Drop` already fired `wl_buffer.release` when it
            // cleared its own state — don't release again here.
            slot.0.lock().unwrap().take();
            return;
        };
        let wl_buffer: &WlBuffer = &buffer;

        // Same dmabuf re-committed (damage-only): the zero-copy import is
        // still valid and `ensure_surface_textures` will refresh any
        // mirror. Leave the existing materializations untouched.
        let same_dmabuf = matches!(
            &*slot.0.lock().unwrap(),
            Some(SurfaceTexture { source: TexSource::Dmabuf { buffer: existing, .. }, .. })
                if existing == wl_buffer
        );
        if same_dmabuf {
            return;
        }

        // New/changed buffer: (re)build the source, carrying over the
        // materializations that survive a buffer swap.
        //   - shm: keep everything if geometry matches (re-uploaded each
        //     commit anyway), so double-buffered clients don't recreate
        //     ShmTextures every frame.
        //   - dmabuf: keep only Mirror entries when extent+format match —
        //     their scratch + target import depend only on extent/format and
        //     are expensive to rebuild, so a churning client that
        //     reallocates its dmabuf every frame reuses them (ensure only
        //     re-imports home_src + re-copies). Native entries reference the
        //     old buffer and are dropped (re-imported for the new buffer).
        match build_tex_source(state, wl_buffer) {
            Ok(source) => {
                let mut guard = slot.0.lock().unwrap();
                let (carried, carried_commit): (HashMap<DrmDevId, GpuTex>, Option<CommitCounter>) =
                    match (guard.take(), &source) {
                        (Some(old), TexSource::Shm { extent, format, .. })
                            if matches!(
                                &old.source,
                                TexSource::Shm { extent: oe, format: of, .. }
                                    if oe == extent && of == format
                            ) =>
                        {
                            // shm geometry matches: keep the per-GPU ShmTextures
                            // (and their initialized state) and the damage cursor,
                            // so the next upload only touches what changed.
                            (old.by_gpu, old.shm_upload_commit)
                        }
                        (Some(old), TexSource::Dmabuf { dmabuf, format, .. })
                            if matches!(
                                &old.source,
                                TexSource::Dmabuf { dmabuf: od, format: of, .. }
                                    if od.width == dmabuf.width
                                        && od.height == dmabuf.height
                                        && of == format
                            ) =>
                        {
                            (
                                old.by_gpu
                                    .into_iter()
                                    .filter(|(_, gt)| matches!(gt, GpuTex::Mirror { .. }))
                                    .collect(),
                                None,
                            )
                        }
                        _ => (HashMap::new(), None),
                    };
                *guard = Some(SurfaceTexture {
                    source,
                    by_gpu: carried,
                    shm_upload_commit: carried_commit,
                    // Cleared per-commit in ensure_surface_textures (covers
                    // same-buffer damage re-commits too); start empty here.
                    acquire_waited: std::collections::HashSet::new(),
                });
            }
            Err(e) => tracing::warn!("surface buffer source build failed: {e:#}"),
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
    // Layer-shell surfaces (and their subsurface trees) bind to an output via
    // the LayerMap, and arrange() already sends their wl_surface.enter. Skip
    // the layout-driven placement dispatch for them so we don't clobber that
    // assignment or fire a spurious wl_surface.leave.
    let mut probe = surface.clone();
    while let Some(p) = get_parent(&probe) {
        probe = p;
    }
    if matches!(get_role(&probe), Some("zwlr_layer_surface_v1")) {
        return;
    }

    // Resolve the surface's current output via the layout. Non-window surfaces
    // (handled above for layer shell) resolve to None and silently skip.
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
/// Queue redraw on every output. Coarser than ideal, but conservative —
/// used by protocol request paths (foreign-toplevel activate, ext-workspace
/// activate/assign) whose layout effects can span outputs. Same shape as
/// `queue_redraw_all` in prism-input.
pub(crate) fn queue_redraw_all(state: &mut PrismState) {
    let ids: Vec<_> = state.outputs.keys().cloned().collect();
    for id in ids {
        state.output_redraw.entry(id).or_default().queue_redraw();
    }
}

pub(crate) fn queue_redraw_for_surface(state: &mut PrismState, surface: &WlSurface) {
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
    //
    // Popups are not subsurfaces, so the `get_parent` walk above stops at
    // the popup's own surface rather than its toplevel — that resolves to
    // neither a layout window nor a layer. Follow the xdg_popup parent
    // chain to the real root surface (toplevel or layer) and resolve that,
    // so a popup commit repaints the output its parent sits on.
    let resolve = |state: &PrismState, s: &WlSurface| {
        state
            .layout
            .find_window_and_output(s)
            .and_then(|(_, out)| out.map(|o| o.name()))
            .or_else(|| state.layer_surface_output_id(s))
    };
    let output_name = resolve(state, &root).or_else(|| {
        let popup = state.popups.find_popup(&root)?;
        let popup_root = find_popup_root_surface(&popup).ok()?;
        resolve(state, &popup_root)
    });

    let Some(output_id) = output_name else {
        return;
    };

    state
        .output_redraw
        .entry(output_id)
        .or_default()
        .queue_redraw();
}

/// A cursor sprite resolved for upload to the hardware cursor plane:
/// tightly-packed RGBA8888 pixels (what [`prism_drm::CursorPlane::upload_sprite`]
/// wants) plus dimensions and the hotspot in *sprite* (physical) pixels.
struct CursorSprite {
    pixels_rgba: Vec<u8>,
    width: u32,
    height: u32,
    hotspot: (i32, i32),
}

/// Resolve the current cursor into an uploadable sprite at `scale`, or
/// `None` if the cursor is hidden.
///
/// - `Named` (theme / `wp_cursor_shape`): frame 0 of the XCursor at the
///   owning output's integer scale.
/// - `Surface` (client `set_cursor`): the client's committed shm buffer,
///   swizzled to RGBA8888, hotspot scaled by the buffer scale. A non-shm
///   (dmabuf) or unreadable cursor buffer falls back to the default theme
///   cursor so the pointer never silently vanishes.
fn resolve_cursor_sprite(state: &PrismState, scale: i32) -> Option<CursorSprite> {
    let named_sprite =
        |icon: CursorIcon, scale: i32, cursor: &Rc<prism_layout::cursor::XCursor>| {
            let frame = state.cursor_texture_cache.get(icon, scale, cursor, 0);
            // The xcursor hotspot lives on the original Image (physical px at
            // this scale), not on the decoded frame.
            let (_idx, image) = cursor.frame(0);
            CursorSprite {
                pixels_rgba: (*frame.pixels_rgba).clone(),
                width: frame.width,
                height: frame.height,
                hotspot: (image.xhot as i32, image.yhot as i32),
            }
        };

    match state.cursor_manager.get_render_cursor(scale) {
        RenderCursor::Hidden => None,
        RenderCursor::Named {
            icon,
            scale,
            cursor,
        } => Some(named_sprite(icon, scale, &cursor)),
        RenderCursor::Surface { hotspot, surface } => {
            read_shm_cursor_sprite(&surface, (hotspot.x, hotspot.y)).or_else(|| {
                let cursor = state.cursor_manager.get_default_cursor(scale);
                Some(named_sprite(CursorIcon::Default, scale, &cursor))
            })
        }
    }
}

/// Read a client cursor surface's committed shm buffer into a tightly-packed
/// RGBA8888 sprite. `None` for a non-shm (dmabuf) buffer, an unsupported
/// pixel format, or a missing buffer — the caller falls back to a theme
/// cursor. `hotspot_logical` is the protocol hotspot (surface-local); we
/// scale it to physical sprite pixels by the surface's buffer scale.
fn read_shm_cursor_sprite(
    surface: &WlSurface,
    hotspot_logical: (i32, i32),
) -> Option<CursorSprite> {
    let (buffer, buffer_scale) = with_states(surface, |states| {
        let s = states.data_map.get::<RendererSurfaceStateUserData>()?;
        let guard = s.lock().unwrap();
        Some((guard.buffer().cloned()?, guard.buffer_scale().max(1)))
    })?;

    let sprite = with_buffer_contents(&buffer, |ptr, len, data| {
        if data.width <= 0 || data.height <= 0 || data.stride <= 0 || data.offset < 0 {
            return None;
        }
        let (w, h, stride, offset) = (
            data.width as usize,
            data.height as usize,
            data.stride as usize,
            data.offset as usize,
        );
        if offset.saturating_add(stride * h) > len {
            return None;
        }
        // Swizzle each wl_shm format into RGBA8888 (R,G,B,A). Argb/Xrgb are
        // B,G,R,A in memory (little-endian); Abgr/Xbgr are already R,G,B,A.
        let swap_rb = match data.format {
            wl_shm::Format::Argb8888 | wl_shm::Format::Xrgb8888 => true,
            wl_shm::Format::Abgr8888 | wl_shm::Format::Xbgr8888 => false,
            _ => return None,
        };
        let opaque = matches!(
            data.format,
            wl_shm::Format::Xrgb8888 | wl_shm::Format::Xbgr8888
        );
        // SAFETY: smithay holds the pool mapping for this callback; the
        // offset+len bounds were checked above.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            let row = offset + y * stride;
            for x in 0..w {
                let i = row + x * 4;
                let o = (y * w + x) * 4;
                let (b0, b1, b2) = (bytes[i], bytes[i + 1], bytes[i + 2]);
                if swap_rb {
                    out[o] = b2;
                    out[o + 1] = b1;
                    out[o + 2] = b0;
                } else {
                    out[o] = b0;
                    out[o + 1] = b1;
                    out[o + 2] = b2;
                }
                out[o + 3] = if opaque { 255 } else { bytes[i + 3] };
            }
        }
        Some(CursorSprite {
            pixels_rgba: out,
            width: w as u32,
            height: h as u32,
            hotspot: (
                hotspot_logical.0 * buffer_scale,
                hotspot_logical.1 * buffer_scale,
            ),
        })
    })
    .ok()??;
    Some(sprite)
}

/// Show the cursor on the output containing the pointer (hidden on the
/// rest), re-uploading the sprite to that output's hardware cursor plane
/// when it changed ([`PrismState::cursor_dirty`]) or the pointer crossed to
/// a new output. Pointer motion within one output only repositions.
///
/// Theme cursors are loaded at the owning output's integer scale, so the
/// cursor is correctly sized on HiDPI monitors. Only one output shows the
/// cursor at a time. Cursor-only commits at sub-vblank cadence are still a
/// future refinement (today it rides the next redraw).
pub fn update_output_cursors(state: &mut PrismState) {
    state.cursor_manager.check_cursor_image_surface_alive();

    // Auto-hidden (typing / inactivity) ⇒ hide the plane, but leave the
    // uploaded sprite intact so revealing it again (on pointer motion) is a
    // cheap reposition rather than a re-upload.
    if !state.pointer_visibility.is_visible() {
        hide_all_cursors(state);
        return;
    }

    // Owner = the output the pointer is in. Off all outputs ⇒ hide.
    let Some(owner_id) =
        state.output_containing((state.pointer_pos.x as i32, state.pointer_pos.y as i32))
    else {
        hide_all_cursors(state);
        return;
    };
    let owner_scale = state
        .wl_outputs
        .get(&owner_id)
        .map(|o| o.current_scale().fractional_scale())
        .unwrap_or(1.0)
        .round()
        .max(1.0) as i32;

    // Re-resolve + upload only when the sprite content changed
    // (`cursor_dirty`) or the pointer crossed onto an output whose plane
    // holds a stale sprite. Plain motion within one output skips this — it
    // just repositions using the cached hotspot, so we never re-read a
    // client cursor buffer per motion event.
    let need_upload = state.cursor_dirty || state.cursor_uploaded_to.as_ref() != Some(&owner_id);
    if need_upload {
        let Some(sprite) = resolve_cursor_sprite(state, owner_scale) else {
            // Hidden cursor.
            hide_all_cursors(state);
            state.cursor_uploaded_to = None;
            state.cursor_dirty = false;
            return;
        };
        state.cursor_hotspot = sprite.hotspot;
        if let Some(plane) = state
            .outputs
            .get_mut(&owner_id)
            .and_then(|o| o.cursor.as_mut())
        {
            match plane.upload_sprite(&sprite.pixels_rgba, sprite.width, sprite.height) {
                Ok(()) => {
                    state.cursor_uploaded_to = Some(owner_id.clone());
                    state.cursor_dirty = false;
                }
                // Leave dirty set so the next pass retries (e.g. a client
                // cursor larger than the BO, or no plane yet).
                Err(e) => tracing::warn!(connector = %owner_id, "cursor upload failed: {e:#}"),
            }
        }
    }

    let hotspot = state.cursor_hotspot;
    let pointer_pos = state.pointer_pos;
    for (id, output_ctx) in state.outputs.iter_mut() {
        let Some(cursor) = output_ctx.cursor.as_mut() else {
            continue;
        };
        let Some(wl_output) = state.wl_outputs.get(id) else {
            cursor.set_visible(false);
            continue;
        };
        let was_visible = cursor.visible();
        let prev_pos = cursor.position();

        if *id == owner_id {
            // pointer_pos and origin are logical; the DRM cursor plane wants
            // physical CRTC pixels, so scale the in-output offset by the
            // output's fractional scale. The hotspot is already in sprite
            // (physical) pixels and subtracts as-is.
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
            // The cursor is a hardware plane committed in `present`'s page-flip,
            // not a render element — so a move / visibility change produces no
            // element damage. Force the present past the zero-damage skip, or the
            // cursor would freeze on screen until something else damaged. (`cursor`
            // is no longer borrowed here, so the &mut self call is fine.)
            output_ctx.force_next_present();
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
        // Hide the plane if it was visible; `changed` ends the cursor borrow
        // before the `force_next_present`/`queue_redraw` calls below.
        let changed = match output_ctx.cursor.as_mut() {
            Some(c) if c.visible() => {
                c.set_visible(false);
                true
            }
            _ => false,
        };
        if changed {
            // Hiding the cursor plane is a page-flip change with no element
            // damage — force the present so the hide actually reaches the screen.
            output_ctx.force_next_present();
            state
                .output_redraw
                .entry(id.clone())
                .or_default()
                .queue_redraw();
        }
    }
}

/// Build the GPU-agnostic [`TexSource`] for `buffer`: a dmabuf source
/// (looked up in `dmabuf_sources`) or an shm source (geometry read from
/// the buffer). No GPU work.
fn build_tex_source(state: &PrismState, buffer: &WlBuffer) -> Result<TexSource> {
    if let Some(entry) = state.dmabuf_sources.get(&buffer.id()) {
        let dmabuf = &entry.source;
        // For YUV (NV12/P010), `format` is the luma-plane format (R8/R16) — a
        // proxy used only by the per-GPU native-vs-mirror gate in
        // `materialize_dmabuf_for_gpu`. The real two-plane import keys off
        // `yuv_kind_for(dmabuf.format)` in `import_dmabuf`. vk_format_for has no
        // single mapping for multi-planar YUV, so don't route it through there.
        let format = match yuv_kind_for(dmabuf.format) {
            Some(kind) => kind.plane_formats().0,
            None => vk_format_for(dmabuf.format)
                .with_context(|| format!("no Vulkan format mapping for {:?}", dmabuf.format))?,
        };
        // YUV is opaque; otherwise the fourcc's A-vs-X variant decides.
        let has_alpha = yuv_kind_for(dmabuf.format).is_none() && fourcc_has_alpha(dmabuf.format);
        return Ok(TexSource::Dmabuf {
            dmabuf: dmabuf.clone(),
            format,
            buffer: buffer.clone(),
            has_alpha,
        });
    }
    // wp_single_pixel_buffer: a 1x1 solid color (swaybg `-c`, solid
    // backgrounds). Not dmabuf or shm — there's no texture to upload; carry
    // the premultiplied sRGB RGBA and lower it to a color-managed SolidColorEl
    // in the render walk.
    if let Ok(spb) = single_pixel_buffer::get_single_pixel_buffer(buffer) {
        return Ok(TexSource::SolidColor {
            rgba: spb.rgba8888(),
            buffer: buffer.clone(),
        });
    }
    // shm: read geometry (uploads happen lazily per consuming GPU).
    let (extent, format, has_alpha) = with_buffer_contents(buffer, |_ptr, _len, data| {
        let format = vk_format_for_shm(data.format)
            .with_context(|| format!("no Vulkan format mapping for wl_shm::{:?}", data.format))?;
        if data.width <= 0 || data.height <= 0 {
            anyhow::bail!("invalid shm geometry: {}x{}", data.width, data.height);
        }
        Ok((
            vk::Extent2D {
                width: data.width as u32,
                height: data.height as u32,
            },
            format,
            shm_format_has_alpha(data.format),
        ))
    })
    .context("with_buffer_contents (shm geometry)")??;
    Ok(TexSource::Shm {
        extent,
        format,
        buffer: buffer.clone(),
        has_alpha,
    })
}

/// The set of GPUs that currently need this surface's pixels, derived from
/// placement: the GPU driving the output the surface's *root* toplevel is
/// on. Subsurfaces inherit their toplevel's output/GPU. Single-output
/// today (the layout assigns each window one monitor), but the return type
/// is a set so spanning windows extend it with no change to materialization.
/// Empty for surfaces the layout doesn't place yet (pre-map, layer shell) —
/// the render-demand safety net covers those.
fn consumer_gpus_for_surface(state: &PrismState, surface: &WlSurface) -> Vec<DrmDevId> {
    let mut root = surface.clone();
    while let Some(parent) = get_parent(&root) {
        root = parent;
    }
    let resolve = |state: &PrismState, s: &WlSurface| {
        state
            .layout
            .find_window_and_output(s)
            .and_then(|(_, out)| out.map(|o| o.name()))
            // Layer-shell surfaces aren't layout windows; resolve their output
            // from the LayerMap so their textures materialize on the hosting GPU.
            .or_else(|| state.layer_surface_output_id(s))
    };
    // Popups aren't subsurfaces, so the parent walk above stops at the popup
    // surface, which resolves to neither a window nor a layer — leaving the
    // consumer-GPU set empty and `ensure_surface_textures` unable to refresh
    // the popup's texture on re-commit (so menu hover/press frames never
    // update). Follow the xdg_popup parent chain to the real root and resolve
    // that, mirroring `queue_redraw_for_surface`.
    let output_name = resolve(state, &root)
        .or_else(|| {
            let popup = state.popups.find_popup(&root)?;
            let popup_root = find_popup_root_surface(&popup).ok()?;
            resolve(state, &popup_root)
        })
        // The DnD icon isn't a layout window or layer surface — it rides
        // the pointer, so its pixels are consumed by the GPU driving the
        // output under the pointer. Without this, same-buffer shm damage
        // on an animated icon never re-uploads (refresh_shm_uploads only
        // writes to consumer GPUs).
        .or_else(|| {
            let icon = state.dnd_icon.as_ref()?;
            if icon.surface != root {
                return None;
            }
            state.output_containing((state.pointer_pos.x as i32, state.pointer_pos.y as i32))
        });
    let Some(name) = output_name else {
        return Vec::new();
    };
    state
        .outputs
        .get(&name)
        .map(|o| o.gpu_id)
        .into_iter()
        .collect()
}

/// Materialize a surface's texture on each GPU that displays it, and do the
/// per-commit refresh work (mirror copies, shm re-uploads). Runs from the
/// commit handler *after* `dispatch_surface_output_from_layout`, so the
/// consumer-GPU set is known. All GPU work is on `&PrismState` (devices and
/// copiers are shared via `Arc` / `&self`), so this takes a shared borrow.
///
/// Materializations are kept warm: a GPU's texture is built once and reused
/// across commits, dropped only when the surface's buffer is replaced
/// (`process_surface_buffer` rebuilds the source with an empty per-GPU map)
/// or destroyed.
fn ensure_surface_textures(state: &PrismState, surface: &WlSurface) {
    if state.gpus.is_empty() {
        return;
    }
    let consumer_gpus = consumer_gpus_for_surface(state, surface);

    with_states(surface, |states| {
        let Some(slot) = states.data_map.get::<SurfaceTexSlot>() else {
            return;
        };
        let mut guard = slot.0.lock().unwrap();
        let Some(tex) = guard.as_mut() else {
            return;
        };
        let extent = tex.extent();
        let shm_last_commit = tex.shm_upload_commit;

        match &tex.source {
            TexSource::Dmabuf { .. } => {
                // Refresh existing mirrors first — the client rewrote the BO
                // this commit (new buffer, or damage on a reused one). The
                // scratch + target import are reused; we only re-import
                // home_src if the client buffer changed, then re-copy + make
                // the new pixels visible on the target GPU. Bounded: only
                // cross-GPU surfaces have mirror entries.
                refresh_dmabuf_mirrors(state, tex, extent);
                // Materialize for any consumer GPU we don't yet have.
                for &g in &consumer_gpus {
                    if tex.by_gpu.contains_key(&g) {
                        continue;
                    }
                    if let Err(e) = materialize_dmabuf_for_gpu(state, tex, g) {
                        tracing::warn!(gpu = ?g, "dmabuf materialize failed: {e:#}");
                    }
                }
                // The client wrote this buffer this commit — on every GPU, the
                // next render that samples it must wait on its implicit write
                // fence once (GPUs re-add themselves via
                // mark_dmabuf_acquire_waited after a confirmed render submit).
                tex.acquire_waited.clear();
            }
            TexSource::Shm { .. } => {
                // Upload only the regions the client damaged this commit
                // (buffer coords map 1:1 onto the per-GPU image). damage_since
                // returns the whole buffer when the commit is unknown/too old,
                // and a freshly created ShmTexture forces a full upload — so a
                // new image is always fully written either way.
                let mut regions: Vec<vk::Rect2D> = Vec::new();
                let mut current_commit = shm_last_commit;
                if let Some(rss) = states.data_map.get::<RendererSurfaceStateUserData>() {
                    let rss = rss.lock().unwrap();
                    current_commit = Some(rss.damage().current_commit());
                    for rect in rss.damage_since(shm_last_commit).iter() {
                        if let Some(r) = clamp_buffer_rect_to_extent(rect, extent) {
                            regions.push(r);
                        }
                    }
                }
                match refresh_shm_uploads(state, tex, &consumer_gpus, &regions) {
                    Ok(()) => tex.shm_upload_commit = current_commit,
                    Err(e) => tracing::warn!("shm upload failed: {e:#}"),
                }
            }
            // No texture to materialize — the render walk reads the color
            // directly and emits a SolidColorEl.
            TexSource::SolidColor { .. } => {}
        }
    });
}

/// Render-demand safety net: materialize `surface` on `gpu` because the
/// render walk found it being drawn on an output whose GPU has no texture
/// for it yet (a (surface, GPU) pair the commit-time, placement-driven
/// `ensure_surface_textures` didn't cover — spanning windows, surfaces that
/// committed before their toplevel was placed, layer surfaces). Called by
/// the integrator *after* the surface-tree walk, never inside it (GPU work +
/// `with_states` would re-enter the surface lock and deadlock). A no-op
/// if the texture already exists on `gpu` by the time we run.
pub fn materialize_surface_on_gpu(state: &PrismState, surface: &WlSurface, gpu: DrmDevId) {
    if state.gpus.is_empty() {
        return;
    }
    with_states(surface, |states| {
        let Some(slot) = states.data_map.get::<SurfaceTexSlot>() else {
            return;
        };
        let mut guard = slot.0.lock().unwrap();
        let Some(tex) = guard.as_mut() else {
            return;
        };
        if tex.by_gpu.contains_key(&gpu) {
            return;
        }
        let result = match &tex.source {
            TexSource::Dmabuf { .. } => materialize_dmabuf_for_gpu(state, tex, gpu),
            // Fresh texture on this GPU (guarded by the contains_key check
            // above) → full upload via the uninitialized rule; damage ignored.
            TexSource::Shm { .. } => refresh_shm_uploads(state, tex, &[gpu], &[]),
            // Solid color: no per-GPU texture; the render walk lowers it.
            TexSource::SolidColor { .. } => Ok(()),
        };
        match result {
            Ok(()) => {
                tracing::debug!(gpu = ?gpu, surface = ?surface.id(), "demand-materialized surface texture")
            }
            Err(e) => tracing::warn!(gpu = ?gpu, "demand materialize failed: {e:#}"),
        }
    });
}

/// Render-time cross-GPU mirror sync. For each surface in `surfaces` that
/// has a mirror on `target_gpu`, submit its home→scratch copy
/// asynchronously on the home GPU (batched per home, one submit each) and
/// import the resulting `sync_file` as a wait semaphore on the target GPU.
/// The caller adds the returned semaphores to the target output's render
/// submit (so the render waits for the copies GPU-side) and **must** pass
/// them back to [`destroy_render_wait_semaphores`] afterwards.
///
/// Each copy submit in turn waits (GPU-side, on the home queue) on:
/// - the client's implicit write fence for each copied dmabuf, so the copy
///   never reads a buffer the client's GPU is still writing — the mirror
///   analog of [`prepare_dmabuf_acquire_waits`], with the same per-buffer
///   skip via `acquire_waited` (keyed by the *home* GPU; marked right after
///   the copy submit is queued, which — unlike a present — is confirmed
///   synchronously);
/// - the target GPU's latest render-done fence
///   ([`prism_renderer::MirrorCopier::render_done_dup`]), so overwriting
///   the scratch can't tear a still-in-flight read of the previous frame.
///
/// This is what keeps the cross-GPU path off the event loop: the copy is
/// non-blocking and every dependency is a GPU semaphore, not a CPU fence
/// wait. Returns empty for outputs with no mirrored surfaces.
pub fn prepare_mirror_waits(
    state: &PrismState,
    surfaces: &[WlSurface],
    target_gpu: DrmDevId,
) -> Vec<vk::Semaphore> {
    use std::os::fd::AsFd;

    if surfaces.is_empty() {
        return Vec::new();
    }
    /// Per-home-GPU batch: the copy ops plus the producer fences the copy
    /// submit must wait on (and which surfaces they came from, for the
    /// post-submit `acquire_waited` marking).
    #[derive(Default)]
    struct HomeBatch {
        ops: Vec<prism_renderer::MirrorCopyOp>,
        producer_fences: Vec<std::os::fd::OwnedFd>,
        fenced_surfaces: Vec<WlSurface>,
    }
    // Gather copy ops grouped by home GPU (collect the vk::Image handles and
    // export the write fences under each surface's lock; submit after
    // releasing it).
    let mut by_home: HashMap<DrmDevId, HomeBatch> = HashMap::new();
    for surface in surfaces {
        with_states(surface, |states| {
            let Some(slot) = states.data_map.get::<SurfaceTexSlot>() else {
                return;
            };
            let guard = slot.0.lock().unwrap();
            let Some(tex) = guard.as_ref() else { return };
            if let Some(GpuTex::Mirror {
                home,
                home_src,
                scratch,
                chroma,
                ..
            }) = tex.by_gpu.get(&target_gpu)
            {
                let batch = by_home.entry(*home).or_default();
                batch.ops.push(prism_renderer::MirrorCopyOp {
                    src: home_src.image(),
                    dst: scratch.image(),
                    extent: scratch.extent(),
                });
                // YUV mirror: copy the chroma plane too. The home_src is a
                // two-plane YUV import, so its chroma image is the source.
                if let Some(chroma) = chroma {
                    if let Some(chroma_src) = home_src.chroma_image() {
                        batch.ops.push(prism_renderer::MirrorCopyOp {
                            src: chroma_src,
                            dst: chroma.scratch.image(),
                            extent: chroma.scratch.extent(),
                        });
                    }
                }
                // Producer fence for this buffer, unless a previous copy on
                // this home GPU already waited it (one wait per buffer per
                // GPU; the set is cleared on every new commit). Plane 0
                // carries the implicit fence, chroma planes share the BO.
                if !tex.acquire_waited.contains(home) {
                    if let TexSource::Dmabuf { dmabuf, .. } = &tex.source {
                        if let Some(fd) = dmabuf
                            .planes
                            .first()
                            .and_then(|p| crate::dmabuf_sync::export_read_fence(p.fd.as_fd()).ok())
                        {
                            batch.producer_fences.push(fd);
                            batch.fenced_surfaces.push(surface.clone());
                        }
                    }
                }
            }
        });
    }

    let mut waits = Vec::new();
    let Some(target_copier) = state.mirror_copiers.get(&target_gpu) else {
        return waits;
    };
    for (home, batch) in by_home {
        let Some(home_copier) = state.mirror_copiers.get(&home) else {
            continue;
        };
        // Import the copy submit's waits on the home device: one semaphore
        // per producer fence, plus the target's render-done fence. Surfaces
        // whose fence fails to import are left out of the post-submit
        // marking so the next frame retries the wait.
        let mut copy_waits = Vec::new();
        let mut fenced_surfaces = Vec::new();
        for (fd, surface) in batch.producer_fences.into_iter().zip(batch.fenced_surfaces) {
            match home_copier.import_wait_semaphore(fd) {
                Ok(sem) => {
                    copy_waits.push(sem);
                    fenced_surfaces.push(surface);
                }
                Err(e) => {
                    tracing::debug!(?home, "mirror producer-fence import failed: {e:#}")
                }
            }
        }
        if let Some(fd) = target_copier.render_done_dup() {
            match home_copier.import_wait_semaphore(fd) {
                Ok(sem) => copy_waits.push(sem),
                Err(e) => tracing::debug!(?home, "mirror render-done import failed: {e:#}"),
            }
        }
        match home_copier.copy_batch_async(&batch.ops, &copy_waits) {
            Ok(fd) => {
                // The copy submit carrying the producer waits is queued —
                // these buffers don't need another wait on the home GPU.
                mark_dmabuf_acquire_waited(&fenced_surfaces, home);
                match target_copier.import_wait_semaphore(fd) {
                    Ok(sem) => waits.push(sem),
                    Err(e) => tracing::warn!(?home, "mirror wait import failed: {e:#}"),
                }
            }
            Err(e) => tracing::warn!(?home, "mirror copy submit failed: {e:#}"),
        }
        // Queued or failed, the imported copy-wait semaphores are done with;
        // the home device's deferred-destroy queue frees them once its
        // serials prove the submit (if any) completed.
        for sem in copy_waits {
            home_copier.destroy_imported_semaphore(sem);
        }
    }
    waits
}

/// Record the present-completion `sync_file` of a confirmed `Presented`
/// outcome on `target_gpu` whose render sampled mirror scratches. The next
/// home→scratch copy for that GPU waits a dup of it (see
/// [`prepare_mirror_waits`]), closing the render→overwrite race on the
/// shared LINEAR scratch. Call only when the present really queued a render
/// submit — after FlipPending / SkippedNoDamage there is no new fence and
/// the previously stored one must stay in place.
pub fn note_mirror_render_done(
    state: &PrismState,
    target_gpu: DrmDevId,
    render_done: &std::os::fd::OwnedFd,
) {
    let Some(copier) = state.mirror_copiers.get(&target_gpu) else {
        return;
    };
    match render_done.try_clone() {
        Ok(fd) => copier.note_render_done(fd),
        Err(e) => tracing::warn!(?target_gpu, "dup of present sync fd failed: {e}"),
    }
}

/// Destroy the render-wait semaphores returned by [`prepare_mirror_waits`] /
/// [`prepare_dmabuf_acquire_waits`] (both import SYNC_FDs the same way), after
/// the render submit that waited on them has been queued.
pub fn destroy_render_wait_semaphores(
    state: &PrismState,
    target_gpu: DrmDevId,
    sems: Vec<vk::Semaphore>,
) {
    if let Some(copier) = state.mirror_copiers.get(&target_gpu) {
        for sem in sems {
            copier.destroy_imported_semaphore(sem);
        }
    }
}

/// For each native-dmabuf surface drawn on `target_gpu` this frame, import the
/// client's implicit write fence as a wait semaphore on `target_gpu`, so the
/// render submit doesn't sample a buffer the client's GPU is still writing.
///
/// This is the Vulkan analog of the implicit sync a GL/EGL compositor gets for
/// free from Mesa: we export the dmabuf's read-sync fence
/// ([`dmabuf_sync::export_read_fence`]) and import it as a binary semaphore
/// (same path as the cross-GPU mirror waits). Surfaces with no exportable fence
/// (already-signalled, or kernel without `EXPORT_SYNC_FILE`) are skipped — that
/// just degrades to the prior unsynchronized behavior for those.
///
/// Destroy the returned semaphores with [`destroy_render_wait_semaphores`]
/// after the present submit.
pub fn prepare_dmabuf_acquire_waits(
    state: &PrismState,
    surfaces: &[WlSurface],
    target_gpu: DrmDevId,
) -> Vec<vk::Semaphore> {
    use std::os::fd::AsFd;

    if surfaces.is_empty() {
        return Vec::new();
    }
    let Some(copier) = state.mirror_copiers.get(&target_gpu) else {
        return Vec::new();
    };

    let mut waits = Vec::new();
    for surface in surfaces {
        // Export the producer write fence under the surface lock (we need the
        // dmabuf plane fd); import it as a semaphore after releasing the lock.
        // Deliberately does NOT mark the wait as done here: `present()` can
        // still bail (FlipPending / no damage) without queueing any GPU work,
        // and the retry must re-export. The caller marks via
        // `mark_dmabuf_acquire_waited` once the render submit is confirmed.
        let fence_fd = with_states(surface, |states| {
            let slot = states.data_map.get::<SurfaceTexSlot>()?;
            let mut guard = slot.0.lock().unwrap();
            let tex = guard.as_mut()?;
            let TexSource::Dmabuf { dmabuf, .. } = &tex.source else {
                return None;
            };
            let plane = dmabuf.planes.first()?;
            crate::dmabuf_sync::export_read_fence(plane.fd.as_fd()).ok()
        });
        let Some(fence_fd) = fence_fd else { continue };
        match copier.import_wait_semaphore(fence_fd) {
            Ok(sem) => waits.push(sem),
            Err(e) => tracing::debug!(?target_gpu, "dmabuf acquire-fence import failed: {e:#}"),
        }
    }
    // Quiet unless a drawn native surface had no exportable write fence — the
    // expected case is one fence per surface (verified: Mesa attaches them).
    // A shortfall means the client/driver attached no implicit fence for some
    // buffer, leaving that sample unsynchronized.
    if waits.len() != surfaces.len() {
        tracing::debug!(
            ?target_gpu,
            surfaces = surfaces.len(),
            imported = waits.len(),
            "dmabuf acquire: some surfaces had no exportable write fence"
        );
    }
    waits
}

/// Record that a GPU submit on `target_gpu` carrying the acquire waits for
/// `surfaces` was actually queued: those surfaces' current buffers don't need
/// another producer-fence wait on this GPU. Two callers, same rule — only a
/// *confirmed* submit counts:
/// - the render path, on a confirmed `Presented` outcome (after FlipPending /
///   SkippedNoDamage the waits went unused and the next attempt must
///   re-export — see `acquire_waited`);
/// - [`prepare_mirror_waits`], right after a successful home-GPU copy submit
///   (queued synchronously, so no Presented-style confirmation is needed).
pub fn mark_dmabuf_acquire_waited(surfaces: &[WlSurface], target_gpu: DrmDevId) {
    for surface in surfaces {
        with_states(surface, |states| {
            if let Some(slot) = states.data_map.get::<SurfaceTexSlot>() {
                if let Some(tex) = slot.0.lock().unwrap().as_mut() {
                    tex.acquire_waited.insert(target_gpu);
                }
            }
        });
    }
}

/// Materialize a dmabuf-backed surface on GPU `g`: a zero-copy native
/// import if `g`'s driver supports the modifier, else a cross-GPU mirror
/// (home import + LINEAR exportable scratch copied and re-imported on `g`).
fn materialize_dmabuf_for_gpu(
    state: &PrismState,
    tex: &mut SurfaceTexture,
    g: DrmDevId,
) -> Result<()> {
    // Pull source essentials out so we don't hold a borrow on tex.source
    // while mutating tex.by_gpu below.
    let (dmabuf, format, buffer_id) = match &tex.source {
        TexSource::Dmabuf {
            dmabuf,
            format,
            buffer,
            ..
        } => (dmabuf.clone(), *format, buffer.id()),
        TexSource::Shm { .. } | TexSource::SolidColor { .. } => return Ok(()),
    };
    let device_g = state
        .gpus
        .get(&g)
        .context("target gpu not registered")?
        .clone();
    let modifier = u64::from(dmabuf.modifier);

    if gpu_supports_dmabuf(&device_g, format, modifier) {
        // Reuse a memoized import for this (buffer, GPU) when present: a
        // pool-reusing client re-commits the same wl_buffer, and rebuilding the
        // VkImage each swap-back is wasted driver work. The cache lives in the
        // buffer's `dmabuf_sources` entry, so it's dropped when the buffer is
        // destroyed (no dangling import). Mirrors smithay's renderer dmabuf
        // cache. The entry should exist for any live dmabuf (inserted in
        // `dmabuf_imported`); fall back to an uncached import if it's somehow
        // gone rather than failing.
        let img = match state.dmabuf_sources.get(&buffer_id) {
            Some(entry) => {
                let mut cache = entry.native_imports.lock().unwrap();
                match cache.get(&g) {
                    Some(img) => img.clone(),
                    None => {
                        let img = Arc::new(
                            import_dmabuf(&device_g, &dmabuf, true)
                                .context("native import on consumer GPU")?,
                        );
                        cache.insert(g, img.clone());
                        img
                    }
                }
            }
            None => Arc::new(
                import_dmabuf(&device_g, &dmabuf, true).context("native import on consumer GPU")?,
            ),
        };
        tex.by_gpu.insert(g, GpuTex::Native(img));
        // Freshly attached (incl. via the render-demand path, which doesn't go
        // through the per-commit ensure): its first sample must wait on the
        // client's write fence. (Cleared even on a cache hit — the client may
        // have rewritten this buffer's pixels since we last sampled it.)
        tex.acquire_waited.clear();
        return Ok(());
    }

    // Cross-GPU mirror. Find/establish a home GPU that can read the buffer.
    // home_src is a copy *source* only → imported untransitioned (the async
    // copy moves it to TRANSFER_SRC itself).
    let (home_id, home_src) = ensure_home_import(state, tex, format, &dmabuf)?;
    let device_home = state.gpus.get(&home_id).context("home gpu gone")?.clone();
    let extent = vk::Extent2D {
        width: dmabuf.width,
        height: dmabuf.height,
    };
    // The scratch is LINEAR (modifier 0) — a safe, universally-defined
    // layout (unlike DRM_FORMAT_MOD_INVALID, which faulted the GPU).
    const DRM_FORMAT_MOD_LINEAR: u64 = 0;

    // Build one LINEAR scratch on the home GPU plus its sampleable import on
    // the consumer GPU, for a single plane. No copy happens here: the scratch
    // is filled at render time by the async copy (prepare_mirror_waits),
    // GPU-synchronized against the target's render submit; this import just
    // sets up the sampleable image, gated behind that copy's semaphore on its
    // first sample. We confirm the consumer can sample LINEAR for the format
    // up front so an unsupported pairing fails cleanly rather than at import,
    // and import with the known vk::Format directly (the scratch is always
    // single-plane, so no fourcc round-trip is needed).
    let make_plane = |plane_extent: vk::Extent2D,
                      vk_fmt: vk::Format,
                      fourcc: DrmFourcc|
     -> Result<(
        prism_renderer::ExportableImage,
        Arc<prism_renderer::ImportedImage>,
    )> {
        if !gpu_supports_dmabuf(&device_g, vk_fmt, DRM_FORMAT_MOD_LINEAR) {
            anyhow::bail!("consumer GPU can't sample LINEAR for {vk_fmt:?}; no mirror");
        }
        let scratch =
            prism_renderer::ExportableImage::new(device_home.clone(), plane_extent, vk_fmt, fourcc)
                .context("ExportableImage::new (mirror scratch)")?;
        let target = prism_renderer::ImportedImage::import(
            device_g.clone(),
            scratch.exported_dmabuf(),
            vk_fmt,
            vk::ImageUsageFlags::SAMPLED,
        )
        .context("import mirror scratch on consumer GPU")?;
        target
            .transition_for_sampling()
            .context("transition mirror scratch for sampling")?;
        Ok((scratch, Arc::new(target)))
    };

    let yuv = yuv_kind_for(dmabuf.format);
    // Luma plane (or the whole RGB image). `format` is already the luma
    // vk::Format for YUV (set in build_tex_source) and the RGB format
    // otherwise; the fourcc is descriptive only since we import by vk::Format.
    let luma_fourcc = match yuv {
        Some(prism_renderer::YuvKind::Nv12) => DrmFourcc::R8,
        Some(prism_renderer::YuvKind::P010) => DrmFourcc::R16,
        None => dmabuf.format,
    };
    let (scratch, target) = make_plane(extent, format, luma_fourcc)?;

    // Chroma plane for YUV: interleaved Cb/Cr at half res in both axes
    // (4:2:0), recombined with luma by the consumer's decode shader.
    let chroma = match yuv {
        Some(kind) => {
            let (_, chroma_fmt) = kind.plane_formats();
            let chroma_fourcc = match kind {
                prism_renderer::YuvKind::Nv12 => DrmFourcc::Gr88,
                prism_renderer::YuvKind::P010 => DrmFourcc::Gr1616,
            };
            let chroma_extent = vk::Extent2D {
                width: extent.width.div_ceil(2),
                height: extent.height.div_ceil(2),
            };
            let (scratch, target) = make_plane(chroma_extent, chroma_fmt, chroma_fourcc)?;
            Some(MirrorChroma {
                scratch,
                target,
                kind,
            })
        }
        None => None,
    };

    tracing::debug!(
        target = ?g,
        home = ?home_id,
        yuv = ?yuv,
        "built cross-GPU mirror for surface ({}x{})",
        extent.width,
        extent.height
    );
    tex.by_gpu.insert(
        g,
        GpuTex::Mirror {
            home: home_id,
            home_src,
            home_src_buffer: buffer_id,
            scratch,
            target,
            chroma,
        },
    );
    Ok(())
}

/// Find a home GPU for a mirror (one whose driver can import the client
/// buffer, to serve as the copy source) and import the buffer there. Reuses
/// an existing native import of this surface on some GPU if one is present
/// (e.g. a spanning window with both a native and a mirrored consumer);
/// otherwise imports on the first capable GPU. Does NOT insert into
/// `tex.by_gpu` — the import is owned by the mirror's `home_src`.
fn ensure_home_import(
    state: &PrismState,
    tex: &SurfaceTexture,
    format: vk::Format,
    dmabuf: &Arc<prism_frame::Dmabuf>,
) -> Result<(DrmDevId, Arc<prism_renderer::ImportedImage>)> {
    if let Some((&gid, img)) = tex.by_gpu.iter().find_map(|(id, gt)| match gt {
        GpuTex::Native(img) => Some((id, img.clone())),
        _ => None,
    }) {
        return Ok((gid, img));
    }
    let modifier = u64::from(dmabuf.modifier);
    let (&home_id, device) = state
        .gpus
        .iter()
        .find(|(_, d)| gpu_supports_dmabuf(d, format, modifier))
        .context("no GPU can import this dmabuf to serve as a mirror home")?;
    let img = Arc::new(import_dmabuf(device, dmabuf, false).context("home import for mirror")?);
    Ok((home_id, img))
}

/// Per-commit refresh of a dmabuf surface's existing cross-GPU mirrors:
/// re-import `home_src` (the copy source) when the client swapped to a new
/// buffer. **No GPU copy or sync happens here** — that's deferred to render
/// time ([`prepare_mirror_waits`]), where it's submitted asynchronously and
/// synchronized against the target's render via a semaphore, so the commit
/// path never blocks the event loop. The scratch + target import are reused
/// across buffer swaps. No-op for surfaces with no mirror entries.
fn refresh_dmabuf_mirrors(state: &PrismState, tex: &mut SurfaceTexture, _extent: vk::Extent2D) {
    let (dmabuf, buffer_id) = match &tex.source {
        TexSource::Dmabuf { dmabuf, buffer, .. } => (dmabuf.clone(), buffer.id()),
        TexSource::Shm { .. } | TexSource::SolidColor { .. } => return,
    };

    for gt in tex.by_gpu.values_mut() {
        let GpuTex::Mirror {
            home,
            home_src,
            home_src_buffer,
            ..
        } = gt
        else {
            continue;
        };
        if *home_src_buffer == buffer_id {
            continue; // same buffer — home_src still valid (damage re-copied at render)
        }
        if let Some(home_dev) = state.gpus.get(home) {
            match import_dmabuf(home_dev, &dmabuf, false) {
                Ok(img) => {
                    *home_src = Arc::new(img);
                    *home_src_buffer = buffer_id.clone();
                }
                Err(e) => tracing::warn!(home = ?home, "mirror home_src re-import failed: {e:#}"),
            }
        }
    }
}

/// Upload the current shm bytes to each consuming GPU, creating or reusing
/// a per-GPU [`ShmTexture`]. Runs each commit for shm-backed surfaces.
///
/// `regions` are the damaged image rects to upload (clamped buffer coords);
/// a newly created `ShmTexture` ignores them and uploads its full extent. The
/// same `regions` go to every consumer GPU — they share the source pixels.
fn refresh_shm_uploads(
    state: &PrismState,
    tex: &mut SurfaceTexture,
    consumer_gpus: &[DrmDevId],
    regions: &[vk::Rect2D],
) -> Result<()> {
    let (extent, format, buffer) = match &tex.source {
        TexSource::Shm {
            extent,
            format,
            buffer,
            ..
        } => (*extent, *format, buffer.clone()),
        TexSource::Dmabuf { .. } | TexSource::SolidColor { .. } => return Ok(()),
    };
    with_buffer_contents(&buffer, |ptr, len, data| {
        if data.width <= 0 || data.height <= 0 || data.stride <= 0 || data.offset < 0 {
            anyhow::bail!(
                "invalid shm buffer geometry: {}x{} stride={} offset={}",
                data.width,
                data.height,
                data.stride,
                data.offset
            );
        }
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
        // SAFETY: smithay holds the pool mapping for the duration of this
        // callback; ptr+offset..+needed is in-bounds per the check above.
        let bytes = unsafe { std::slice::from_raw_parts(ptr.add(offset), needed) };

        for &g in consumer_gpus {
            let Some(device) = state.gpus.get(&g) else {
                continue;
            };
            let reuse = matches!(
                tex.by_gpu.get(&g),
                Some(GpuTex::Shm(t)) if t.extent() == extent && t.format() == format
            );
            if !reuse {
                let t = prism_renderer::ShmTexture::new(device.clone(), extent, format)
                    .with_context(|| format!("ShmTexture::new on gpu {}:{}", g.major, g.minor))?;
                tex.by_gpu.insert(g, GpuTex::Shm(t));
            }
            if let Some(GpuTex::Shm(t)) = tex.by_gpu.get_mut(&g) {
                t.upload_bytes(bytes, stride, regions).with_context(|| {
                    format!("ShmTexture::upload_bytes on gpu {}:{}", g.major, g.minor)
                })?;
            }
        }
        Ok(())
    })
    .context("with_buffer_contents (shm upload)")?
}

/// Clamp a buffer-space damage rect to the texture extent and convert to a
/// `vk::Rect2D`. shm buffer coords map 1:1 onto the uploaded image, so this is
/// just a bounds clamp; returns `None` if the rect is empty after clamping.
fn clamp_buffer_rect_to_extent(
    rect: &Rectangle<i32, smithay::utils::Buffer>,
    extent: vk::Extent2D,
) -> Option<vk::Rect2D> {
    let x0 = rect.loc.x.max(0);
    let y0 = rect.loc.y.max(0);
    let x1 = (rect.loc.x + rect.size.w).min(extent.width as i32);
    let y1 = (rect.loc.y + rect.size.h).min(extent.height as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: y0 },
        extent: vk::Extent2D {
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        },
    })
}

/// Whether a `wl_shm` format carries meaningful alpha (`A`-format) vs an
/// undefined `X` byte. `vk_format_for_shm` maps both to the same alpha-bearing
/// `vk::Format`, so this is the only place the distinction survives — the
/// decode shader needs it to force opaque alpha for `X` buffers (otherwise a
/// client that leaves the `X` byte at 0 renders invisible) and to treat `A`
/// buffers as premultiplied.
fn shm_format_has_alpha(fmt: wl_shm::Format) -> bool {
    matches!(
        fmt,
        wl_shm::Format::Argb8888 | wl_shm::Format::Abgr8888 | wl_shm::Format::Abgr16161616f
    )
}

/// Whether a DRM fourcc carries meaningful alpha (`A`-format) vs an undefined
/// `X` byte. See [`shm_format_has_alpha`]; YUV is handled by the caller (always
/// opaque). `vk_format_for` conflates `Xrgb`/`Argb`, so this is the alpha
/// source of truth for dmabuf sources.
fn fourcc_has_alpha(fourcc: DrmFourcc) -> bool {
    matches!(
        fourcc,
        DrmFourcc::Argb8888
            | DrmFourcc::Abgr8888
            | DrmFourcc::Argb2101010
            | DrmFourcc::Abgr2101010
            | DrmFourcc::Abgr16161616f
    )
}

fn vk_format_for_shm(fmt: wl_shm::Format) -> Option<vk::Format> {
    Some(match fmt {
        // wl_shm formats are byte-order in memory the same way DRM fourcc
        // is: Argb8888 == B,G,R,A bytes == vk::Format::B8G8R8A8_UNORM.
        wl_shm::Format::Argb8888 | wl_shm::Format::Xrgb8888 => vk::Format::B8G8R8A8_UNORM,
        // RGBA byte order (R,G,B,A) == vk::Format::R8G8B8A8_UNORM.
        wl_shm::Format::Abgr8888 | wl_shm::Format::Xbgr8888 => vk::Format::R8G8B8A8_UNORM,
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

/// Two-plane YUV video fourccs prism imports via [`ImportedImage::import_yuv`]
/// (luma + chroma as separate single-plane images). `None` ⇒ not YUV; the
/// caller falls back to the packed-RGB [`vk_format_for`] path.
fn yuv_kind_for(fourcc: DrmFourcc) -> Option<prism_renderer::YuvKind> {
    match fourcc {
        DrmFourcc::Nv12 => Some(prism_renderer::YuvKind::Nv12),
        DrmFourcc::P010 => Some(prism_renderer::YuvKind::P010),
        _ => None,
    }
}

/// Map a DRM fourcc to the Vulkan format we'd sample it as. Single-planar
/// 32-bit packed formats only for now.
fn vk_format_for(fourcc: DrmFourcc) -> Option<vk::Format> {
    Some(match fourcc {
        // DRM is little-endian-byte-order, so XRGB8888 in memory is B,G,R,X.
        // Vulkan's B8G8R8A8 reads exactly that byte order.
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
        // RGBA byte order: DRM ABGR8888 is R,G,B,A in memory (LE u32
        // [A:24][B:16][G:8][R:0]), matching Vulkan R8G8B8A8. The natural
        // format for many GL/GLES/Vulkan clients (Mesa's EGL default), so
        // accepting it avoids a hard reject on buffers we can sample fine.
        DrmFourcc::Xbgr8888 | DrmFourcc::Abgr8888 => vk::Format::R8G8B8A8_UNORM,
        // 10-bit: DRM AB30/XB30 pack [A:30][B:20][G:10][R:0] in a LE u32,
        // which is exactly Vulkan's A2B10G10R10_UNORM_PACK32. HDR10 clients
        // (Firefox with HDR on, mpv PQ passthrough) allocate these.
        DrmFourcc::Xbgr2101010 | DrmFourcc::Abgr2101010 => vk::Format::A2B10G10R10_UNORM_PACK32,
        // 10-bit, BGRA component order: DRM AR30/XR30 pack [A:30][R:20][G:10]
        // [B:0], which is Vulkan's A2R10G10B10_UNORM_PACK32. The less common
        // 10-bit variant (HDR10 clients usually pick AB30 above), accepted
        // for the same reason as ABGR8888.
        DrmFourcc::Xrgb2101010 | DrmFourcc::Argb2101010 => vk::Format::A2R10G10B10_UNORM_PACK32,
        // fp16: DRM ABGR16161616F is R,G,B,A 16-bit floats in memory order,
        // matching Vulkan R16G16B16A16_SFLOAT. scRGB / fp16 HDR clients use
        // this; values can exceed 1.0.
        DrmFourcc::Xbgr16161616f | DrmFourcc::Abgr16161616f => vk::Format::R16G16B16A16_SFLOAT,
        _ => return None,
    })
}

/// DRM fourccs prism can import as a sampled texture, in rough
/// preference order (8-bit first, then HDR-capable 10-bit + fp16).
/// Each MUST have a [`vk_format_for`] mapping. The 10-bit and fp16
/// entries are what HDR clients allocate; advertising them with the
/// real tiled modifiers the GPU supports keeps Mesa from falling
/// back to an implementation-defined layout (`modifier=Invalid`),
/// which we can't import without a GPU page fault — see
/// [`build_advertised_dmabuf_formats`] and the import-time guard in
/// [`import_dmabuf`].
const DMABUF_CANDIDATE_FOURCCS: &[DrmFourcc] = &[
    DrmFourcc::Xrgb8888,
    DrmFourcc::Argb8888,
    DrmFourcc::Xbgr8888,
    DrmFourcc::Abgr8888,
    DrmFourcc::Xbgr2101010,
    DrmFourcc::Abgr2101010,
    DrmFourcc::Xrgb2101010,
    DrmFourcc::Argb2101010,
    DrmFourcc::Xbgr16161616f,
    DrmFourcc::Abgr16161616f,
];

/// Build the dmabuf format/modifier set to advertise, by intersecting
/// [`DMABUF_CANDIDATE_FOURCCS`] with the modifiers `device` actually
/// supports for single-plane SAMPLED import. Every client buffer is
/// composited as a texture, so SAMPLED (not COLOR_ATTACHMENT, which is
/// the scanout side's filter) is the right capability bit.
///
/// An empty result (driver advertises no modifiers for any candidate)
/// falls back to LINEAR 8-bit, which every driver supports — keeps
/// basic clients working even on a minimal driver.
/// Single-plane, SAMPLED-capable DRM modifiers for `vk_format`. Used to
/// advertise the per-plane importable modifier set for YUV formats.
fn single_plane_sampled_modifiers(
    device: &prism_renderer::Device,
    vk_format: vk::Format,
) -> Vec<DrmModifier> {
    device
        .supported_drm_format_modifiers(vk_format)
        .into_iter()
        .filter(|m| {
            m.plane_count == 1
                && m.tiling_features
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        })
        .map(|m| DrmModifier::from(m.modifier))
        .collect()
}

fn build_advertised_dmabuf_formats(device: &prism_renderer::Device) -> Vec<DrmFormat> {
    let mut out = Vec::new();
    for &fourcc in DMABUF_CANDIDATE_FOURCCS {
        let Some(vk_format) = vk_format_for(fourcc) else {
            continue;
        };
        let all = device.supported_drm_format_modifiers(vk_format);
        let mut accepted = 0usize;
        let mut has_tiled = false;
        for m in &all {
            // Our importer is single-plane only; multi-plane modifiers
            // (DCC/CCS metadata planes) need separate memory imports.
            if m.plane_count != 1 {
                continue;
            }
            if !m
                .tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
            {
                continue;
            }
            let modifier = DrmModifier::from(m.modifier);
            if modifier != DrmModifier::Linear {
                has_tiled = true;
            }
            out.push(DrmFormat {
                code: fourcc,
                modifier,
            });
            accepted += 1;
        }
        // Per-format breakdown: a client (Firefox HDR) that wants to
        // *render* into a format needs a tiled modifier — if we only
        // offer LINEAR for an HDR format, Mesa won't use it as a render
        // target and falls back to 8-bit. `has_tiled=false` on a 10-bit
        // or fp16 row is the thing to look at.
        tracing::info!(
            ?fourcc,
            queried = all.len(),
            accepted,
            has_tiled,
            "dmabuf candidate format"
        );
    }

    // Two-plane YUV video formats. We import each plane as its own
    // single-plane image (see import_yuv), so advertise the modifiers
    // supported (single-plane, SAMPLED) by *both* the luma and chroma plane
    // formats — the intersection is what we can actually import.
    //
    // NV12 (8-bit SDR) verified end-to-end native + cross-GPU; P010 (10-bit
    // HDR) is what Firefox's Wayland HDR path decodes to. Both planes import
    // as their own single-plane image, so advertise the modifiers supported
    // by both: R8/R8G8 for NV12, R16/R16G16 for P010.
    for &(fourcc, luma_fmt, chroma_fmt) in &[
        (
            DrmFourcc::Nv12,
            vk::Format::R8_UNORM,
            vk::Format::R8G8_UNORM,
        ),
        (
            DrmFourcc::P010,
            vk::Format::R16_UNORM,
            vk::Format::R16G16_UNORM,
        ),
    ] {
        let luma = single_plane_sampled_modifiers(device, luma_fmt);
        let chroma = single_plane_sampled_modifiers(device, chroma_fmt);
        let mut accepted = 0usize;
        let mut has_tiled = false;
        for m in &luma {
            if !chroma.contains(m) {
                continue;
            }
            if *m != DrmModifier::Linear {
                has_tiled = true;
            }
            out.push(DrmFormat {
                code: fourcc,
                modifier: *m,
            });
            accepted += 1;
        }
        tracing::info!(?fourcc, accepted, has_tiled, "dmabuf YUV candidate format");
    }

    if out.is_empty() {
        out.extend([
            DrmFormat {
                code: DrmFourcc::Xrgb8888,
                modifier: DrmModifier::Linear,
            },
            DrmFormat {
                code: DrmFourcc::Argb8888,
                modifier: DrmModifier::Linear,
            },
        ]);
    }
    out
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
