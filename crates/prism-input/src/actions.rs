//! Action dispatcher — runs a config-bound `Action` against
//! `PrismState`.
//!
//! Ported subset of niri's `State::handle_bind` /
//! `State::do_action` (input/mod.rs lines 643+). What's here today
//! covers the bind set we expect to use immediately; everything else
//! logs an "unhandled action" line so we can grow incrementally.
//!
//! Action vocabulary is defined in [`prism_config::Action`]. The full
//! niri vocabulary is ~150 variants; we implement only the
//! direct-user-facing ones first. Actions that need subsystems we
//! don't have yet (screenshot UI, MRU, lock screen, hotkey overlay,
//! IPC, animations) are stubbed.

use std::process::Command;

use prism_config::Action;
use prism_layout::layout::ActivateWindow;
use prism_protocols::PrismState;
use smithay::output::Output;

pub fn handle_action(state: &mut PrismState, action: Action) {
    use Action as A;
    match action {
        A::Quit(_skip_confirmation) => {
            tracing::info!("action: Quit");
            state.should_stop = true;
        }
        A::Spawn(args) => spawn(args),
        A::SpawnSh(cmd) => spawn(vec!["sh".to_string(), "-c".to_string(), cmd]),
        A::CloseWindow => {
            // Use the layout's view of the focused window rather than
            // `state.keyboard_focus`. `keyboard_focus` updates on click
            // but the visible focus ring tracks the layout's
            // `active_window` per active monitor — when those drift
            // apart (multi-monitor without focus-follows-mouse), the
            // ring shows one window but the keyboard sees another.
            // Closing what the user can see is focused is the
            // least-surprising choice.
            let toplevel = state.layout.focus().map(|m| m.toplevel().clone());
            match toplevel {
                Some(t) => {
                    t.send_close();
                    queue_redraw_active_output(state);
                }
                None => tracing::debug!("action: CloseWindow with no focused window"),
            }
        }
        // Focus navigation
        A::FocusColumnLeft => {
            state.layout.focus_left();
            queue_redraw_active_output(state);
        }
        A::FocusColumnRight => {
            state.layout.focus_right();
            queue_redraw_active_output(state);
        }
        A::FocusWindowUp => {
            state.layout.focus_up();
            queue_redraw_active_output(state);
        }
        A::FocusWindowDown => {
            state.layout.focus_down();
            queue_redraw_active_output(state);
        }
        // Move column
        A::MoveColumnLeft => {
            state.layout.move_left();
            queue_redraw_active_output(state);
        }
        A::MoveColumnRight => {
            state.layout.move_right();
            queue_redraw_active_output(state);
        }
        A::MoveWindowUp => {
            state.layout.move_up();
            queue_redraw_active_output(state);
        }
        A::MoveWindowDown => {
            state.layout.move_down();
            queue_redraw_active_output(state);
        }
        // Monitor navigation. Directional lookup walks every wl_output's
        // (location, logical_size) and picks the nearest neighbor whose
        // perpendicular extent overlaps the active output's. Mirrors
        // niri's `output_left_of`/`output_right_of`/etc.
        A::FocusMonitorLeft => {
            if let Some(out) = state.output_left() {
                state.layout.focus_output(&out);
                queue_redraw_active_output(state);
            }
        }
        A::FocusMonitorRight => {
            if let Some(out) = state.output_right() {
                state.layout.focus_output(&out);
                queue_redraw_active_output(state);
            }
        }
        A::FocusMonitorUp => {
            if let Some(out) = state.output_up() {
                state.layout.focus_output(&out);
                queue_redraw_active_output(state);
            }
        }
        A::FocusMonitorDown => {
            if let Some(out) = state.output_down() {
                state.layout.focus_output(&out);
                queue_redraw_active_output(state);
            }
        }
        A::FocusMonitorPrevious => {
            if let Some(out) = state.output_previous() {
                state.layout.focus_output(&out);
                queue_redraw_active_output(state);
            }
        }
        A::FocusMonitorNext => {
            if let Some(out) = state.output_next() {
                state.layout.focus_output(&out);
                queue_redraw_active_output(state);
            }
        }
        // Move the active column to a neighboring monitor (the whole
        // tile stack moves together). `activate=true` means focus
        // follows the column to the new monitor — the most common
        // expectation; matches niri's default.
        A::MoveColumnToMonitorLeft => move_active_column_to(state, state.output_left()),
        A::MoveColumnToMonitorRight => move_active_column_to(state, state.output_right()),
        A::MoveColumnToMonitorUp => move_active_column_to(state, state.output_up()),
        A::MoveColumnToMonitorDown => move_active_column_to(state, state.output_down()),
        A::MoveColumnToMonitorPrevious => move_active_column_to(state, state.output_previous()),
        A::MoveColumnToMonitorNext => move_active_column_to(state, state.output_next()),
        // Move a single window (the active one) — leaves the rest of
        // the column behind. Uses `Layout::move_to_output(None, ...)`
        // which picks "the active window" when window=None.
        A::MoveWindowToMonitorLeft => move_active_window_to(state, state.output_left()),
        A::MoveWindowToMonitorRight => move_active_window_to(state, state.output_right()),
        A::MoveWindowToMonitorUp => move_active_window_to(state, state.output_up()),
        A::MoveWindowToMonitorDown => move_active_window_to(state, state.output_down()),
        A::MoveWindowToMonitorPrevious => move_active_window_to(state, state.output_previous()),
        A::MoveWindowToMonitorNext => move_active_window_to(state, state.output_next()),
        // Workspace navigation
        A::FocusWorkspaceUp => {
            state.layout.switch_workspace_up();
            queue_redraw_active_output(state);
        }
        A::FocusWorkspaceDown => {
            state.layout.switch_workspace_down();
            queue_redraw_active_output(state);
        }
        // Sizing
        A::MaximizeColumn => {
            state.layout.toggle_full_width();
            queue_redraw_active_output(state);
        }
        A::FullscreenWindow => {
            if let Some(surface) = state.keyboard_focus.surface().cloned() {
                // `Mapped::Id = smithay::desktop::Window` (the layout
                // identifies windows by the wrapped Window). The
                // inherent `Mapped::id()` returns `MappedId` (an
                // opaque numeric id) — we want the trait-level one,
                // i.e. the `window` field.
                let window = state
                    .layout
                    .find_window_and_output(&surface)
                    .map(|(mapped, _)| mapped.window.clone());
                if let Some(w) = window {
                    state.layout.toggle_fullscreen(&w);
                    queue_redraw_active_output(state);
                }
            }
        }
        A::CenterColumn => {
            state.layout.center_column();
            queue_redraw_active_output(state);
        }
        A::SwitchPresetColumnWidth => {
            // forwards=true == cycle through niri's preset widths in
            // user-facing order; matches Mod+R behaviour.
            state.layout.toggle_width(true);
            queue_redraw_active_output(state);
        }
        A::ConsumeWindowIntoColumn => {
            state.layout.consume_into_column();
            queue_redraw_active_output(state);
        }
        A::ExpelWindowFromColumn => {
            state.layout.expel_from_column();
            queue_redraw_active_output(state);
        }
        // Stubs for actions whose subsystems aren't ported yet.
        other => {
            tracing::debug!("action: unhandled {other:?}");
        }
    }
}

/// Fork+exec a child process, detached from prism. Mirrors niri's
/// `spawn`: ignore the child's stdio so its lifetime is fully
/// independent.
fn spawn(args: Vec<String>) {
    let Some(program) = args.first().cloned() else {
        tracing::warn!("action: Spawn with empty args");
        return;
    };
    let rest: Vec<String> = args.into_iter().skip(1).collect();
    tracing::info!("action: Spawn {program} {rest:?}");

    // Fork via Command. stdin → /dev/null (no TTY for child).
    // stdout/stderr inherit prism's so spawn failures land in our
    // log — hiding them via /dev/null is what made the alacritty /
    // fuzzel "spawn but never appear" silent. setsid detaches the
    // child from our process group so it survives prism exit.
    let mut cmd = Command::new(&program);
    cmd.args(&rest);
    cmd.stdin(std::process::Stdio::null());
    // SAFETY: setsid(2) is async-signal-safe and called between fork
    // and exec where the child is single-threaded. Detaches the child
    // from prism's process group so it survives compositor exit.
    unsafe {
        use std::os::unix::process::CommandExt as _;
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    match cmd.spawn() {
        Ok(child) => {
            tracing::debug!("spawned {program} pid={}", child.id());
            // We don't wait — the child is detached. The Child handle
            // drops here and the OS reparents to init (or, with
            // setsid, the child becomes its own session leader).
        }
        Err(e) => {
            tracing::warn!("spawn {program} failed: {e}");
        }
    }
}

/// Move the active column to `target`. No-op if the target is `None`
/// (i.e. there's no monitor in that direction). After moving, focus
/// follows the column to the new monitor (`activate=true`) and we
/// redraw both source and destination — both visibly changed.
fn move_active_column_to(state: &mut PrismState, target: Option<Output>) {
    let Some(out) = target else {
        return;
    };
    state.layout.move_column_to_output(&out, None, true);
    // focus_output is implied by activate=true above; the focus ring
    // moves to the target. Queue redraws on every output — the column
    // disappeared from one, appeared on another, and any column-shift
    // between siblings on the source also needs to repaint.
    queue_redraw_active_output(state);
}

/// Move just the active window (not the whole column) to `target`.
/// `move_to_output(None, ...)` picks "the active window" by convention,
/// matching niri's `MoveWindowToMonitor*` semantics.
fn move_active_window_to(state: &mut PrismState, target: Option<Output>) {
    let Some(out) = target else {
        return;
    };
    state
        .layout
        .move_to_output(None, &out, None, ActivateWindow::Smart);
    queue_redraw_active_output(state);
}

/// Queue a redraw on whatever output currently hosts the focus. The
/// layout updates may have moved things across outputs; conservatively
/// queue all outputs that own a workspace.
///
/// TODO: a per-action "what changed" hint so we don't redraw the
/// whole world for a focus move.
fn queue_redraw_active_output(state: &mut PrismState) {
    let ids: Vec<_> = state.outputs.keys().cloned().collect();
    for id in ids {
        state.output_redraw.entry(id).or_default().queue_redraw();
    }
}
