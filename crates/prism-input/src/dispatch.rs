//! Top-level input dispatch: libinput → PrismState bookkeeping →
//! smithay seat → focused surface.
//!
//! This is the port of niri's `State::process_input_event` (niri
//! input/mod.rs, line 115). Coverage today:
//!   - DeviceAdded / DeviceRemoved — capability bookkeeping on the seat
//!   - Keyboard — focus dispatch + config-driven binds with cooldown
//!     and key-repeat (`handle_bind` / `start_key_repeat`), plus the
//!     hardcoded VT-switch / emergency-quit / overview escape hatches.
//!     Accessibility and the screenshot/MRU/exit-confirm UIs are
//!     deferred.
//!   - Pointer motion/button/axis — handled in `super::pointer`,
//!     including `Mouse*` / `WheelScroll*` / `TouchpadScroll*` binds.
//!
//! Variants we ignore (logged at trace level so a stray tablet/touch
//! event doesn't silently disappear): tablet, touch, gestures,
//! switches. These come back when the matching subsystems land.

use prism_config::{Bind, Binds, ModKey, Modifiers, Trigger};
use prism_protocols::PrismState;
use smithay::backend::input::{Device, DeviceCapability, InputEvent, KeyState, KeyboardKeyEvent};
use smithay::input::keyboard::{FilterResult, ModifiersState};
use smithay::utils::SERIAL_COUNTER;

use crate::actions;
use crate::backend_ext::PrismInputBackend;

/// Dispatch a single input event.
///
/// Mirrors niri's `process_input_event` top-level shape but covers
/// only the event variants prism has subsystems for. Unhandled
/// variants are traced and dropped — they'll grow real handlers as
/// the corresponding state lands on `PrismState`.
pub fn process_input_event<I: PrismInputBackend + 'static>(
    state: &mut PrismState,
    event: InputEvent<I>,
) where
    I::Device: 'static,
{
    use InputEvent::*;
    // Any real user input resets the idle timers (ext-idle-notify-v1), so
    // swayidle & friends see the user as active. Device hotplug is not
    // activity.
    if !matches!(event, DeviceAdded { .. } | DeviceRemoved { .. }) {
        state.notify_idle_activity();
    }
    // Pointer activity reveals an auto-hidden cursor and (re)arms the
    // hide-after-inactivity timer (`cursor { hide-after-inactive-ms }`).
    // Keyboard hides it (`hide-when-typing`) — handled in `on_keyboard`,
    // which knows press vs release.
    if matches!(
        event,
        PointerMotion { .. }
            | PointerMotionAbsolute { .. }
            | PointerButton { .. }
            | PointerAxis { .. }
    ) {
        state.note_pointer_activity();
    }
    match event {
        DeviceAdded { device } => on_device_added(state, device),
        DeviceRemoved { device } => on_device_removed(state, device),
        Keyboard { event } => on_keyboard::<I>(state, event),
        PointerMotion { event } => super::pointer::on_pointer_motion::<I>(state, event),
        PointerMotionAbsolute { event } => {
            super::pointer::on_pointer_motion_absolute::<I>(state, event)
        }
        PointerButton { event } => super::pointer::on_pointer_button::<I>(state, event),
        PointerAxis { event } => super::pointer::on_pointer_axis::<I>(state, event),
        GestureSwipeBegin { event } => super::gestures::on_gesture_swipe_begin::<I>(state, event),
        GestureSwipeUpdate { event } => super::gestures::on_gesture_swipe_update::<I>(state, event),
        GestureSwipeEnd { event } => super::gestures::on_gesture_swipe_end::<I>(state, event),
        GesturePinchBegin { event } => super::gestures::on_gesture_pinch_begin::<I>(state, event),
        GesturePinchUpdate { event } => super::gestures::on_gesture_pinch_update::<I>(state, event),
        GesturePinchEnd { event } => super::gestures::on_gesture_pinch_end::<I>(state, event),
        GestureHoldBegin { event } => super::gestures::on_gesture_hold_begin::<I>(state, event),
        GestureHoldEnd { event } => super::gestures::on_gesture_hold_end::<I>(state, event),
        // Everything below is awaiting its subsystem. Each line
        // points at the niri source so the port is mechanical when
        // we get there.
        TabletToolAxis { .. }      // niri input/mod.rs:3555
        | TabletToolTip { .. }     // 3622
        | TabletToolProximity { .. } // 3730
        | TabletToolButton { .. }  // 3780
        | TouchDown { .. }         // 4111
        | TouchMotion { .. }
        | TouchUp { .. }
        | TouchCancel { .. }
        | TouchFrame { .. }
        | SwitchToggle { .. }      // 4320
        | Special(_) => {
            tracing::trace!("input: event variant not handled yet");
        }
    }
}

fn on_device_added(state: &mut PrismState, device: impl Device) {
    // Today: log capabilities + flip seat capabilities so wl_seat
    // tells clients we have keyboard/pointer. Once tablets/touch
    // land, mirror niri input/mod.rs:243.
    let has_kb = device.has_capability(DeviceCapability::Keyboard);
    let has_ptr = device.has_capability(DeviceCapability::Pointer);
    let has_touch = device.has_capability(DeviceCapability::Touch);
    tracing::info!("input: device added (keyboard={has_kb} pointer={has_ptr} touch={has_touch})");

    if has_kb && state.seat.get_keyboard().is_none() {
        let cfg = state.config.borrow();
        let kb = &cfg.input.keyboard;
        let result = state.seat.add_keyboard(
            kb.xkb.to_xkb_config(),
            i32::from(kb.repeat_delay),
            i32::from(kb.repeat_rate),
        );
        drop(cfg);
        match result {
            Ok(_) => tracing::info!("seat: keyboard added"),
            Err(e) => tracing::warn!("seat: failed to add keyboard: {e:?}"),
        }
    }
    if has_ptr && state.seat.get_pointer().is_none() {
        state.seat.add_pointer();
        tracing::info!("seat: pointer added");
    }
    // touch / tablet capabilities deferred (no handlers yet).
}

fn on_device_removed(state: &mut PrismState, device: impl Device) {
    let had_kb = device.has_capability(DeviceCapability::Keyboard);
    let had_ptr = device.has_capability(DeviceCapability::Pointer);
    tracing::info!(
        "input: device removed (kb={had_kb} ptr={} touch={})",
        had_ptr,
        device.has_capability(DeviceCapability::Touch),
    );

    // Re-evaluate seat capabilities: when the last device of a kind
    // unplugs, stop advertising it on wl_seat (niri input/mod.rs:255).
    // `libinput_devices` is maintained by the libinput source pre-pass in
    // main.rs and has already had this device removed; non-libinput
    // backends (WLCS) don't populate it, but there the seat lives for the
    // harness's lifetime anyway.
    use smithay::reexports::input as libinput;
    if had_kb
        && state.seat.get_keyboard().is_some()
        && !state
            .libinput_devices
            .iter()
            .any(|d| d.has_capability(libinput::DeviceCapability::Keyboard))
    {
        state.seat.remove_keyboard();
        tracing::info!("seat: last keyboard unplugged — keyboard capability removed");
    }
    if had_ptr
        && state.seat.get_pointer().is_some()
        && !state
            .libinput_devices
            .iter()
            .any(|d| d.has_capability(libinput::DeviceCapability::Pointer))
    {
        state.seat.remove_pointer();
        tracing::info!("seat: last pointer unplugged — pointer capability removed");
    }
}

fn on_keyboard<I: PrismInputBackend>(state: &mut PrismState, event: I::KeyboardKeyEvent) {
    let Some(keyboard) = state.seat.get_keyboard() else {
        return;
    };

    let serial = SERIAL_COUNTER.next_serial();
    let time = smithay::backend::input::Event::time_msec(&event);
    let pressed = event.state() == KeyState::Pressed;

    // Any key release stops bind key-repeat. Niri's behavior, with the
    // same known imperfection (releasing a *different* key than the
    // repeating one also stops the repeat — good enough).
    if !pressed {
        if let Some(token) = state.bind_repeat_timer.take() {
            if let Some(handle) = state.loop_handle.as_ref() {
                handle.remove(token);
            }
        }
    }

    let key_code = event.key_code();
    // "Mod" in user binds maps to Super on TTY (defaults match niri's
    // mod-key resolution). When the config overrides input.mod_key we
    // honour that.
    let mod_key = state.config.borrow().input.mod_key.unwrap_or(ModKey::Super);

    // Snapshot the binds — the filter closure borrows state, but
    // matching against the config requires also borrowing config; do
    // it once up front. Cloning a Bind is cheap (small struct +
    // Action; Action::Spawn carries a Vec<String> but only one per
    // bind, on press).
    let snapshot = {
        let cfg = state.config.borrow();
        cfg.binds.0.clone()
    };

    let bind = keyboard.input::<Option<Bind>, _>(
        state,
        key_code,
        event.state(),
        serial,
        time,
        |this, mods, keysym| {
            // Release: if we suppressed the press, suppress the release
            // too so the client doesn't see a dangling release.
            if !pressed {
                if this.suppressed_keys.remove(&key_code) {
                    return FilterResult::Intercept(None);
                }
                return FilterResult::Forward;
            }

            let raw = keysym.raw_latin_sym_or_raw_current_sym();
            let Some(raw) = raw else {
                return FilterResult::Forward;
            };

            // Hard-coded escape hatches — handled before the user's
            // bind table so they're available no matter what (broken
            // config that fell back to defaults, missing `quit` bind,
            // anything else that would normally leave the user
            // trapped on the prism VT with no working keybindings).
            //
            // Ctrl+Alt+Fn → VT switch via libseat. xkbcommon's TTY
            // layout maps this combo to the contiguous
            // `XF86_Switch_VT_1..XF86_Switch_VT_12` keysym range
            // (0x1008FE01..0x1008FE0C). Routed before the configurable
            // bind lookup so a user bind on `F1` (rare) can't shadow
            // it. We never want a config to be able to disable VT
            // switch — it's the OS-level escape hatch.
            //
            // Detected from the *modified* sym, not `raw`: the VT keysyms
            // live at the non-standard Ctrl+Alt shift level, and `raw`
            // (`raw_latin_sym_or_raw_current_sym`) reads level 0 — so it
            // reports plain `F1..F12` here and would never match. smithay
            // documents this exact pitfall on that method.
            const VT_KEYSYM_BASE: u32 = 0x1008_FE01;
            const VT_KEYSYM_LAST: u32 = 0x1008_FE0C;
            let raw_u32 = raw.raw();
            let modified_u32 = keysym.modified_sym().raw();
            if (VT_KEYSYM_BASE..=VT_KEYSYM_LAST).contains(&modified_u32) {
                let vt = (modified_u32 - VT_KEYSYM_BASE + 1) as i32;
                match this.session.as_ref() {
                    Some(session) => match session.change_vt(vt) {
                        Ok(()) => tracing::info!("VT switch to {vt} requested"),
                        Err(e) => tracing::warn!("VT switch to {vt} failed: {e:#}"),
                    },
                    None => tracing::warn!(
                        "VT switch to {vt} requested but no SeatSession bound \
                         (wayland-only / headless mode)"
                    ),
                }
                // The release will land on the other VT, so a suppression
                // entry for this key would dangle and intercept the release
                // of the *next* plain press of the same F-key (stuck key for
                // one cycle). Clear the whole set instead, mirroring
                // `A::ChangeVt` — a dangling release forwarded to a client
                // that never saw the press is harmless.
                this.suppressed_keys.clear();
                return FilterResult::Intercept(None);
            }

            // Ctrl+Alt+Backspace → emergency quit. Niri/X.org
            // convention. Hardcoded so a typo'd KDL config or a
            // missing `quit` bind can never leave the user with no
            // way out short of ssh + pkill. `KEY_BackSpace` keysym
            // is 0xFF08.
            const KEY_BACKSPACE: u32 = 0xFF08;
            if mods.ctrl && mods.alt && raw_u32 == KEY_BACKSPACE {
                tracing::warn!("Ctrl+Alt+Backspace pressed — emergency quit");
                this.should_stop = true;
                this.suppressed_keys.insert(key_code);
                return FilterResult::Intercept(None);
            }

            match find_bind(&snapshot, mod_key, Trigger::Keysym(raw), *mods) {
                Some(bind) => {
                    this.suppressed_keys.insert(key_code);
                    FilterResult::Intercept(Some(bind))
                }
                None => {
                    // No configured bind: while the overview owns the
                    // keyboard, unmodified Esc/Return/arrows drive it
                    // (niri's `hardcoded_overview_bind`). Unmatched keys
                    // still Forward, but land on the overview's None
                    // focus — no client surface receives them.
                    if this.keyboard_focus.is_overview() {
                        if let Some(bind) = hardcoded_overview_bind(raw, *mods) {
                            this.suppressed_keys.insert(key_code);
                            return FilterResult::Intercept(Some(bind));
                        }
                    }
                    FilterResult::Forward
                }
            }
        },
    );

    if let Some(Some(bind)) = bind {
        handle_bind(state, bind.clone());
        start_key_repeat(state, bind);
    }

    // hide-when-typing: a key *press* hides the cursor (reappears on the
    // next pointer activity). Press-only so releasing a modifier held during
    // a drag doesn't hide it mid-gesture.
    if pressed {
        state.hide_pointer_for_typing();
    }
}

/// Hardcoded keys while the overview owns the keyboard (niri
/// input/mod.rs `hardcoded_overview_bind`): unmodified Esc/Return
/// close it, arrows navigate columns / windows / workspaces. Only
/// consulted after the user's bind table found no match, so a
/// configured bind still wins.
fn hardcoded_overview_bind(
    raw: smithay::input::keyboard::Keysym,
    mods: ModifiersState,
) -> Option<Bind> {
    use prism_config::{Action, Key};
    use smithay::input::keyboard::Keysym;

    if !modifiers_from_state(mods).is_empty() {
        return None;
    }

    let mut repeat = true;
    let action = match raw {
        Keysym::Escape | Keysym::Return => {
            repeat = false;
            Action::ToggleOverview
        }
        Keysym::Left => Action::FocusColumnLeft,
        Keysym::Right => Action::FocusColumnRight,
        Keysym::Up => Action::FocusWindowOrWorkspaceUp,
        Keysym::Down => Action::FocusWindowOrWorkspaceDown,
        _ => return None,
    };

    Some(Bind {
        key: Key {
            trigger: Trigger::Keysym(raw),
            modifiers: Modifiers::empty(),
        },
        action,
        repeat,
        cooldown: None,
        allow_when_locked: false,
        allow_inhibiting: false,
        hotkey_overlay_title: None,
    })
}

/// Run a matched bind's action, honoring its `cooldown-ms`. Ported
/// from niri's `handle_bind` (input/mod.rs:643), with an Instant map
/// instead of niri's timer-token map — same semantics (the bind can't
/// fire again until the cooldown elapses), no event-loop entanglement.
///
/// While the session is locked, only binds marked `allow-when-locked`
/// (plus the always-safe action set below) fire — every dispatch path
/// funnels through here (initial press, key-repeat, mouse/wheel/scroll
/// binds), so this is the single gate. `allow-inhibiting` remains
/// parse-only pending keyboard-shortcuts-inhibit support.
pub(crate) fn handle_bind(state: &mut PrismState, bind: Bind) {
    if state.is_locked() && !(bind.allow_when_locked || allowed_when_locked(&bind.action)) {
        return;
    }
    let Some(cooldown) = bind.cooldown else {
        actions::handle_action(state, bind.action);
        return;
    };

    let now = std::time::Instant::now();
    if state
        .bind_cooldown_until
        .get(&bind.key)
        .is_some_and(|&until| now < until)
    {
        return;
    }
    state.bind_cooldown_until.insert(bind.key, now + cooldown);
    actions::handle_action(state, bind.action);
}

/// Actions that fire on a locked session even without
/// `allow-when-locked` — escape hatches and hardware toggles that can't
/// leak session content (niri input/mod.rs:4635).
fn allowed_when_locked(action: &prism_config::Action) -> bool {
    use prism_config::Action;
    matches!(
        action,
        Action::Quit(_)
            | Action::ChangeVt(_)
            | Action::Suspend
            | Action::PowerOffMonitors
            | Action::PowerOnMonitors
            | Action::SwitchLayout(_)
            | Action::ToggleKeyboardShortcutsInhibit
    )
}

/// Arm the key-repeat timer for a held repeating bind (`repeat`,
/// default true): after the keyboard's repeat delay, re-fire the
/// bind's action at the repeat rate until any key release cancels the
/// timer (see `on_keyboard`). Port of niri's `start_key_repeat`.
fn start_key_repeat(state: &mut PrismState, bind: Bind) {
    use smithay::reexports::calloop::timer::{TimeoutAction, Timer};

    if !bind.repeat {
        return;
    }
    let Some(handle) = state.loop_handle.clone() else {
        // Headless / WLCS harness: no repeat, first fire already done.
        return;
    };

    // Stop the previous bind's repeat, if any.
    if let Some(token) = state.bind_repeat_timer.take() {
        handle.remove(token);
    }

    let (repeat_delay, repeat_rate) = {
        let cfg = state.config.borrow();
        (
            cfg.input.keyboard.repeat_delay,
            cfg.input.keyboard.repeat_rate,
        )
    };
    if repeat_rate == 0 {
        return;
    }
    let repeat_duration = std::time::Duration::from_secs_f64(1. / f64::from(repeat_rate));

    let timer = Timer::from_duration(std::time::Duration::from_millis(u64::from(repeat_delay)));
    match handle.insert_source(timer, move |_, _, state| {
        handle_bind(state, bind.clone());
        TimeoutAction::ToDuration(repeat_duration)
    }) {
        Ok(token) => state.bind_repeat_timer = Some(token),
        Err(e) => tracing::warn!("failed to arm bind key-repeat timer: {e}"),
    }
}

/// Whether any bind exists on one of `triggers` for exactly this
/// modifier combination. The scroll dispatch uses this to decide
/// consume-vs-passthrough for the *whole* axis event — even sub-tick
/// amounts accumulate silently rather than reaching clients when a
/// bind exists for the held modifiers.
///
/// Per-event recompute of niri's precomputed `mods_with_wheel_binds` /
/// `mods_with_finger_scroll_binds` sets (input/mod.rs:5012
/// `mods_with_binds`) — the bind list is small and scroll events are
/// infrequent, and recomputing can't go stale across config reloads.
pub(crate) fn binds_have_trigger_for_mods(
    binds: &[Bind],
    mod_key: ModKey,
    triggers: &[Trigger],
    modifiers: Modifiers,
) -> bool {
    binds.iter().any(|bind| {
        if !triggers.contains(&bind.key.trigger) {
            return false;
        }
        let mut mods = bind.key.modifiers;
        if mods.contains(Modifiers::COMPOSITOR) {
            mods.remove(Modifiers::COMPOSITOR);
            mods.insert(mod_key.to_modifiers());
        }
        mods == modifiers
    })
}

/// Convert smithay's `ModifiersState` (bool fields) into the
/// `bitflags` `Modifiers` set the bind table uses.
pub(crate) fn modifiers_from_state(mods: ModifiersState) -> Modifiers {
    let mut m = Modifiers::empty();
    if mods.ctrl {
        m |= Modifiers::CTRL;
    }
    if mods.shift {
        m |= Modifiers::SHIFT;
    }
    if mods.alt {
        m |= Modifiers::ALT;
    }
    if mods.logo {
        m |= Modifiers::SUPER;
    }
    if mods.iso_level3_shift {
        m |= Modifiers::ISO_LEVEL3_SHIFT;
    }
    if mods.iso_level5_shift {
        m |= Modifiers::ISO_LEVEL5_SHIFT;
    }
    m
}

/// Walk the config's bind list looking for one matching the current
/// `(trigger, modifiers)`. The `COMPOSITOR` bit acts as an alias for
/// `mod_key` so `Mod+Q` and (e.g.) `Super+Q` both match. Ported from
/// niri input/mod.rs:4489 (`find_configured_bind`).
pub(crate) fn find_bind(
    binds: &[Bind],
    mod_key: ModKey,
    trigger: Trigger,
    mods: ModifiersState,
) -> Option<Bind> {
    let mut modifiers = modifiers_from_state(mods);
    let mod_down = modifiers.contains(mod_key.to_modifiers());
    if mod_down {
        modifiers |= Modifiers::COMPOSITOR;
    }

    for bind in binds {
        if bind.key.trigger != trigger {
            continue;
        }
        let mut bind_modifiers = bind.key.modifiers;
        if bind_modifiers.contains(Modifiers::COMPOSITOR) {
            bind_modifiers |= mod_key.to_modifiers();
        } else if bind_modifiers.contains(mod_key.to_modifiers()) {
            bind_modifiers |= Modifiers::COMPOSITOR;
        }
        if bind_modifiers == modifiers {
            return Some(bind.clone());
        }
    }
    None
}

// Silence unused-import warning for the IntoIter shape — we just use
// Binds::0 above. Kept on the imports for symmetry with niri.
#[allow(dead_code)]
fn _binds_is_iterable(b: &Binds) {
    let _ = b.0.iter();
}
