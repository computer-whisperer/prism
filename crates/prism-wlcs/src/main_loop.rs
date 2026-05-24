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
use std::sync::Arc;
use std::time::Duration;

use smithay::output::Mode as OutputMode;
use smithay::reexports::calloop::channel::{Channel, Event as ChannelEvent};
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::EventLoop;
use smithay::reexports::wayland_server::Client;

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
            // Deferred destructor events queued during dispatch, then flush.
            state.color_management.drain_pending_info_done();
            let _ = state.display_handle.flush_clients();
        }
    }
}

fn make_gpu() -> anyhow::Result<Arc<prism_renderer::Device>> {
    let instance = prism_renderer::Instance::new()?;
    Ok(prism_renderer::Device::new(instance, None)?)
}

/// Harvest pending `wl_surface.frame` callbacks across mapped toplevels
/// (descending into subsurfaces) and fire them with the current clock.
fn send_frame_callbacks(state: &mut PrismState) {
    let time_ms = state.clock.now().as_millis() as u32;
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
        } => {
            // TODO(task 6): resolve (client_id, surface_id) -> prism window,
            // mark it floating and set its floating position to `location`
            // (prism-layout FloatingSpace). Stubbed until the input +
            // positioning work lands.
            let _ = (client_id, surface_id, location);
            tracing::debug!(
                client_id,
                surface_id,
                ?location,
                "WLCS: PositionWindow (TODO)"
            );
        }
        // TODO(task 6): synthesize pointer/touch into state.seat, computing
        // the surface-under via prism-input. No-ops for now so the harness
        // builds and runs the protocol-only tests.
        WlcsEvent::NewPointer { .. }
        | WlcsEvent::PointerMoveAbsolute { .. }
        | WlcsEvent::PointerMoveRelative { .. }
        | WlcsEvent::PointerButtonDown { .. }
        | WlcsEvent::PointerButtonUp { .. }
        | WlcsEvent::PointerRemoved { .. }
        | WlcsEvent::NewTouch { .. }
        | WlcsEvent::TouchDown { .. }
        | WlcsEvent::TouchMove { .. }
        | WlcsEvent::TouchUp { .. }
        | WlcsEvent::TouchRemoved { .. } => {}
    }
}
