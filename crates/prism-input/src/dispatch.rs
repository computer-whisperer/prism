//! Top-level input dispatch: libinput → PrismState bookkeeping →
//! smithay seat → focused surface.
//!
//! This is the MVP port of niri's `State::process_input_event` (niri
//! input/mod.rs, line 115). Coverage today:
//!   - DeviceAdded / DeviceRemoved — capability bookkeeping on the seat
//!   - Keyboard — focus dispatch + a single hardcoded quit binding
//!     (Super+Escape). Per-key repeat, full config-driven binds,
//!     accessibility, screenshot/MRU/exit-confirm UIs are deferred.
//!   - Pointer motion/button/axis — handled in `super::pointer`.
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
        // Everything below is awaiting its subsystem. Each line
        // points at the niri source so the port is mechanical when
        // we get there.
        TabletToolAxis { .. }      // niri input/mod.rs:3555
        | TabletToolTip { .. }     // 3622
        | TabletToolProximity { .. } // 3730
        | TabletToolButton { .. }  // 3780
        | GestureSwipeBegin { .. } // 3793
        | GestureSwipeUpdate { .. }
        | GestureSwipeEnd { .. }
        | GesturePinchBegin { .. } // 3994
        | GesturePinchUpdate { .. }
        | GesturePinchEnd { .. }
        | GestureHoldBegin { .. }  // 4048
        | GestureHoldEnd { .. }
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

fn on_device_removed(_state: &mut PrismState, device: impl Device) {
    // Today: log only. A real impl would re-evaluate seat
    // capabilities after the last device of a given kind unplugs.
    // See niri input/mod.rs:255.
    tracing::info!(
        "input: device removed (kb={} ptr={} touch={})",
        device.has_capability(DeviceCapability::Keyboard),
        device.has_capability(DeviceCapability::Pointer),
        device.has_capability(DeviceCapability::Touch),
    );
}

fn on_keyboard<I: PrismInputBackend>(state: &mut PrismState, event: I::KeyboardKeyEvent) {
    let Some(keyboard) = state.seat.get_keyboard() else {
        return;
    };

    let serial = SERIAL_COUNTER.next_serial();
    let time = smithay::backend::input::Event::time_msec(&event);
    let pressed = event.state() == KeyState::Pressed;
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
            const VT_KEYSYM_BASE: u32 = 0x1008_FE01;
            const VT_KEYSYM_LAST: u32 = 0x1008_FE0C;
            let raw_u32 = raw.raw();
            if (VT_KEYSYM_BASE..=VT_KEYSYM_LAST).contains(&raw_u32) {
                let vt = (raw_u32 - VT_KEYSYM_BASE + 1) as i32;
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
                this.suppressed_keys.insert(key_code);
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
                None => FilterResult::Forward,
            }
        },
    );

    if let Some(Some(bind)) = bind {
        // TODO: cooldown enforcement, key-repeat timer for bind.repeat,
        // allow_when_locked once we have a lock state, etc.
        actions::handle_action(state, bind.action);
    }

    // hide-when-typing: a key *press* hides the cursor (reappears on the
    // next pointer activity). Press-only so releasing a modifier held during
    // a drag doesn't hide it mid-gesture.
    if pressed {
        state.hide_pointer_for_typing();
    }
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
fn find_bind(
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
