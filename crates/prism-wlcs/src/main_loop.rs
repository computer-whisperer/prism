//! The prism compositor running under WLCS.
//!
//! Unlike smithay's `wlcs_anvil` (which drives `AnvilState` + a smithay
//! `Space` + `DummyRenderer`), this builds the real [`PrismState`] headless,
//! mirroring prism's `wayland` bring-up mode with two differences:
//!   - no AF_UNIX listening socket — WLCS injects client fds directly;
//!   - a synthetic virtual output and a timer-driven frame-callback pump,
//!     standing in for the DRM scanout + vblank that don't exist here.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use smithay::output::Mode as OutputMode;
use smithay::reexports::calloop::channel::{Channel, Event as ChannelEvent};
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::EventLoop;
use smithay::reexports::wayland_server::Client;
use smithay::utils::{Logical, Point};

use prism_protocols::PrismState;

use crate::WlcsEvent;

/// 800x600@60 at the origin — matches anvil's WLCS output so tests that
/// assume that geometry behave the same against prism.
const OUTPUT_W: i32 = 800;
const OUTPUT_H: i32 = 600;

/// Client bookkeeping the compositor thread keeps alongside `PrismState`:
/// WLCS refers to clients by the raw fd it minted, so we map that to the
/// smithay `Client` to resolve `PositionWindow` targets.
type Clients = Rc<RefCell<HashMap<i32, Client>>>;

pub fn run(channel: Channel<WlcsEvent>) {
    // A Vulkan device (default-picked: lavapipe in CI, the primary card
    // locally). dmabuf conformance needs it; shm-only protocol tests do
    // not, so failure here is non-fatal — run with no GPU and let any
    // dmabuf tests fail rather than aborting the whole suite.
    let mut gpus = HashMap::new();
    let mut primary = None;
    match make_gpu() {
        Ok(device) => match device.physical.drm_primary.or(device.physical.drm_render) {
            Some(key) => {
                primary = Some(key);
                gpus.insert(key, device);
            }
            None => tracing::warn!("WLCS: Vulkan device has no DRM node id; running without GPU"),
        },
        Err(e) => {
            tracing::warn!("WLCS: no Vulkan device ({e:#}); shm-only, dmabuf tests will fail")
        }
    }

    let display = prism_protocols::new_display().expect("WLCS: new_display");
    let mut state = PrismState::new(
        &display,
        prism_config::Config::default(),
        None,
        gpus,
        primary,
    );

    // Settle animations instantly. prism's animation `Clock` is normally
    // driven forward each frame from the kernel vblank time; this headless
    // harness has no vblank, so without help, in-flight tile move/open
    // animations stay pinned at their starting offset (their `value()`
    // returns `from` while the clock sits at the animation's start time).
    // That displaces both rendering and pointer hit-testing from a window's
    // resting position. WLCS asserts on resting-state geometry, not on
    // animation, so collapse every animation to its target immediately.
    state.clock.set_complete_instantly(true);

    let mut event_loop: EventLoop<'static, PrismState> =
        EventLoop::try_new().expect("WLCS: EventLoop::try_new");
    // Stash the loop handle before any client surfaces appear (drm_syncobj
    // pre-commit hook self-guards when no card is attached, as in `wayland`
    // mode).
    state.set_loop_handle(event_loop.handle());

    let running = Rc::new(Cell::new(true));
    let clients: Clients = Rc::new(RefCell::new(HashMap::new()));

    // WLCS command channel.
    {
        let running = running.clone();
        let clients = clients.clone();
        event_loop
            .handle()
            .insert_source(channel, move |event, &mut (), state| match event {
                ChannelEvent::Msg(evt) => handle_event(evt, state, &clients, &running),
                ChannelEvent::Closed => running.set(false),
            })
            .expect("WLCS: insert command channel");
    }

    // Client-request dispatch (moves `display` into the loop). No listening
    // socket — clients arrive only via WlcsEvent::NewClient.
    prism_protocols::server::insert_display_source(&event_loop.handle(), display)
        .expect("WLCS: insert display source");

    // The virtual output: a real wl_output + layout monitor with nothing
    // behind it. `advertise_output_from_parts` leaves the logical position
    // unassigned, so follow with `layout_outputs`.
    let name = prism_config::output::OutputName {
        connector: "WLCS-1".to_string(),
        make: None,
        model: None,
        serial: None,
    };
    let mode = OutputMode {
        size: (OUTPUT_W, OUTPUT_H).into(),
        refresh: 60_000,
    };
    state.advertise_output_from_parts(name, mode, (0, 0));
    state.layout_outputs();

    // Frame-callback pump. Clients that gate their next frame on
    // wl_surface.frame would stall without vblank, so fire callbacks on a
    // ~60Hz timer. Unlike production we fire immediately (no
    // presentation-time accuracy needed) and discard presentation-feedback
    // + drm_syncobj release trackers.
    event_loop
        .handle()
        .insert_source(Timer::immediate(), move |_, _, state| {
            send_frame_callbacks(state);
            TimeoutAction::ToDuration(Duration::from_millis(16))
        })
        .expect("WLCS: insert frame timer");

    while running.get() {
        if event_loop
            .dispatch(Some(Duration::from_millis(16)), &mut state)
            .is_err()
        {
            running.set(false);
        } else {
            // Mirror prism's production post-dispatch step (main.rs:1525+):
            // drain destructors, advance + prune animations (so move/open
            // animations settle and stop displacing tile geometry), flush
            // pending configures to clients, then clear the cached monotonic
            // time so the next tick re-samples.
            state.color_management.drain_pending_info_done();
            state.layout.advance_animations();
            state.layout.refresh(true);
            // Re-evaluate pointer focus: a surface may have moved, resized,
            // or restacked under a stationary pointer this cycle.
            prism_input::pointer::refresh_pointer_focus(&mut state);
            let _ = state.display_handle.flush_clients();
            state.clock.clear();
        }
    }
}

fn make_gpu() -> anyhow::Result<Arc<prism_renderer::Device>> {
    let instance = prism_renderer::Instance::new()?;
    Ok(prism_renderer::Device::new(instance, None)?)
}

/// Harvest pending `wl_surface.frame` callbacks across mapped toplevels
/// (descending into subsurfaces) and fire them with a real monotonic
/// timestamp.
fn send_frame_callbacks(state: &mut PrismState) {
    let time_ms = monotonic_ms();
    let roots: Vec<_> = state
        .xdg_shell
        .toplevel_surfaces()
        .iter()
        .map(|t| t.wl_surface().clone())
        .collect();

    let mut frame_cbs = Vec::new();
    let mut presentation_cbs = Vec::new();
    let mut release_trackers = Vec::new();
    for root in &roots {
        prism_protocols::redraw::harvest_surface_feedback(
            root,
            &mut frame_cbs,
            &mut presentation_cbs,
            &mut release_trackers,
        );
    }
    for cb in frame_cbs {
        cb.done(time_ms);
    }
}

fn handle_event(event: WlcsEvent, state: &mut PrismState, clients: &Clients, running: &Cell<bool>) {
    match event {
        WlcsEvent::Exit => running.set(false),
        WlcsEvent::NewClient { stream, client_id } => {
            match state
                .display_handle
                .insert_client(stream, prism_protocols::state::new_client_data())
            {
                Ok(client) => {
                    clients.borrow_mut().insert(client_id, client);
                }
                Err(e) => tracing::warn!("WLCS: insert_client failed: {e}"),
            }
        }
        WlcsEvent::PositionWindow {
            client_id,
            surface_id,
            location,
        } => position_window(state, clients, client_id, surface_id, location),
        WlcsEvent::PointerMoveAbsolute { location, .. } => {
            let time = monotonic_ms();
            prism_input::pointer::wlcs_pointer_motion_absolute(state, location, time);
        }
        WlcsEvent::PointerMoveRelative { delta, .. } => {
            let time = monotonic_ms();
            prism_input::pointer::wlcs_pointer_motion_relative(state, delta, time);
        }
        WlcsEvent::PointerButtonDown { button_id, .. } => {
            let time = monotonic_ms();
            prism_input::pointer::wlcs_pointer_button(state, button_id as u32, true, time);
        }
        WlcsEvent::PointerButtonUp { button_id, .. } => {
            let time = monotonic_ms();
            prism_input::pointer::wlcs_pointer_button(state, button_id as u32, false, time);
        }
        // Device add/remove is bookkeeping only — prism has no per-device
        // pointer state to track. Touch isn't wired yet (prism doesn't
        // advertise wl_touch); WLCS touch tests skip/fail until it is.
        WlcsEvent::NewPointer { .. }
        | WlcsEvent::PointerRemoved { .. }
        | WlcsEvent::NewTouch { .. }
        | WlcsEvent::TouchDown { .. }
        | WlcsEvent::TouchMove { .. }
        | WlcsEvent::TouchUp { .. }
        | WlcsEvent::TouchRemoved { .. } => {}
    }
}

/// Milliseconds from a fixed process start — the timestamp source for
/// `wl_callback.done` and synthetic input events. prism's `state.clock` is a
/// manually-advanced animation clock that nothing drives in the headless
/// harness, so it reports a frozen time and fails WLCS's monotonic-timestamp
/// checks. A real monotonic source matches how production sources
/// frame-callback time from the kernel vblank clock rather than the animation
/// clock; the base is arbitrary, which is all `wl_callback` requires.
fn monotonic_ms() -> u32 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u32
}

/// Resolve a WLCS (client, surface) pair to a mapped toplevel and place it
/// floating at `location`. WLCS uses absolute global coordinates; with our
/// single virtual output at the origin, the floating `SetFixed` coordinate
/// (which is relative to the output working area) coincides with global —
/// revisit if we ever advertise multiple / offset outputs.
fn position_window(
    state: &mut PrismState,
    clients: &Clients,
    client_id: i32,
    surface_id: u32,
    location: Point<i32, Logical>,
) {
    use smithay::reexports::wayland_server::Resource;

    let Some(client) = clients.borrow().get(&client_id).cloned() else {
        return;
    };

    // Match anvil: scan mapped toplevels for the one owned by this client
    // whose wl_surface carries the given protocol id.
    let toplevels: Vec<_> = state
        .xdg_shell
        .toplevel_surfaces()
        .iter()
        .map(|t| t.wl_surface().clone())
        .collect();
    let surface = toplevels.into_iter().find(|s| {
        let same_client = state.display_handle.get_client(s.id()).ok().as_ref() == Some(&client);
        same_client && s.id().protocol_id() == surface_id
    });
    let Some(surface) = surface else {
        return;
    };

    let Some(window) = state
        .layout
        .find_window_and_output(&surface)
        .map(|(m, _)| m.window.clone())
    else {
        return;
    };

    state.layout.set_window_floating(Some(&window), true);
    state.layout.move_floating_window(
        Some(&window),
        prism_ipc::PositionChange::SetFixed(location.x as f64),
        prism_ipc::PositionChange::SetFixed(location.y as f64),
        false,
    );
}
