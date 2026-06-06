//! Action dispatcher — runs a config-bound `Action` against
//! `PrismState`.
//!
//! Ported from niri's `State::do_action` (input/mod.rs lines 676+),
//! covering the full layout-facing vocabulary. Known divergences, by
//! design:
//!
//!   - **No cursor warping.** Niri's `maybe_warp_cursor_to_focus` /
//!     `move_cursor_to_output` back the `warp-mouse-to-focus` config
//!     feature, which prism parses but does not implement yet. Every
//!     arm that niri warps in simply skips that step here.
//!   - **Whole-world redraws** via [`queue_redraw_active_output`] —
//!     same granularity niri uses (its arms carry `FIXME: granular`
//!     comments), see the TODO on that fn.
//!   - **No screenshot-UI / MRU-UI preludes.** Niri routes some arms
//!     (`MoveColumnLeft` etc.) into the screenshot UI when it's open;
//!     prism has no such UI yet.
//!
//! Actions whose subsystems don't exist yet fall through to the
//! catch-all at the bottom and log: screenshot UI, MRU switcher UI,
//! hotkey overlay, screencast targets, screen transitions, the
//! overview (the *layout* tracks overview state, but prism's render
//! and input paths don't support it — wiring the binds before that
//! would leave the screen not matching the layout's idea of itself),
//! `Suspend` (needs logind), keyboard-shortcuts-inhibit (protocol not
//! implemented), debug render toggles, and the `*UnderMouse` variants
//! (nothing in prism's input paths generates them).

use std::process::Command;
use std::sync::RwLock;
use std::thread;

use prism_config::Action;
use prism_ipc::LayoutSwitchTarget;
use prism_layout::layout::scrolling::ScrollDirection;
use prism_layout::layout::{ActivateWindow, LayoutElement as _};
use prism_protocols::PrismState;
use smithay::desktop::Window;
use smithay::output::Output;

pub fn handle_action(state: &mut PrismState, action: Action) {
    use Action as A;
    match action {
        A::Quit(_skip_confirmation) => {
            tracing::info!("action: Quit");
            state.should_stop = true;
        }
        A::ChangeVt(vt) => {
            // The hardcoded Ctrl+Alt+Fn escape hatch lives in
            // dispatch.rs; this is the programmatic/IPC route to the
            // same place.
            match state.session.as_ref() {
                Some(session) => match session.change_vt(vt) {
                    Ok(()) => tracing::info!("action: ChangeVt({vt})"),
                    Err(e) => tracing::warn!("change-vt {vt} failed: {e:#}"),
                },
                None => tracing::warn!(
                    "change-vt {vt} requested but no SeatSession bound \
                     (wayland-only / headless mode)"
                ),
            }
            // The VT switch may swallow the key releases; drop the
            // suppression state so they don't dangle (mirrors niri).
            state.suppressed_keys.clear();
        }
        A::Spawn(args) => spawn(args),
        A::SpawnSh(cmd) => spawn_sh(cmd),
        A::LoadConfigFile(path) => match &state.config_load_request {
            Some(request) => request(path),
            None => tracing::warn!("load-config-file: no config watcher running"),
        },
        A::SwitchLayout(target) => {
            let Some(keyboard) = state.seat.get_keyboard() else {
                return;
            };
            keyboard.with_xkb_state(state, |mut ctx| match target {
                LayoutSwitchTarget::Next => ctx.cycle_next_layout(),
                LayoutSwitchTarget::Prev => ctx.cycle_prev_layout(),
                LayoutSwitchTarget::Index(idx) => {
                    let num_layouts = ctx.xkb().lock().unwrap().layouts().count();
                    if usize::from(idx) >= num_layouts {
                        tracing::warn!(
                            "switch-layout: index {idx} out of range ({num_layouts} layouts)"
                        );
                    } else {
                        ctx.set_layout(smithay::input::keyboard::Layout(idx.into()));
                    }
                }
            });
        }

        // Window lifecycle
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
        A::CloseWindowById(id) => {
            let toplevel = state
                .layout
                .windows()
                .find(|(_, m)| m.id().get() == id)
                .map(|(_, m)| m.toplevel().clone());
            if let Some(t) = toplevel {
                t.send_close();
                queue_redraw_active_output(state);
            }
        }
        A::FullscreenWindow => {
            // Layout-focus rather than keyboard-focus, same reasoning
            // as CloseWindow. (`Mapped::Id = smithay::desktop::Window`;
            // the layout identifies windows by the wrapped Window, so
            // we clone the `window` field, not the `MappedId`.)
            if let Some(window) = focused_window(state) {
                state.layout.toggle_fullscreen(&window);
                queue_redraw_active_output(state);
            }
        }
        A::FullscreenWindowById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_fullscreen(&window);
                queue_redraw_active_output(state);
            }
        }
        A::ToggleWindowedFullscreen => {
            if let Some(window) = focused_window(state) {
                state.layout.toggle_windowed_fullscreen(&window);
                queue_redraw_active_output(state);
            }
        }
        A::ToggleWindowedFullscreenById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_windowed_fullscreen(&window);
                queue_redraw_active_output(state);
            }
        }
        A::MaximizeWindowToEdges => {
            if let Some(window) = focused_window(state) {
                state.layout.toggle_maximized(&window);
                queue_redraw_active_output(state);
            }
        }
        A::MaximizeWindowToEdgesById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_maximized(&window);
                queue_redraw_active_output(state);
            }
        }

        // Focus navigation
        A::FocusWindow(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.activate_window(&window);
                queue_redraw_active_output(state);
            }
        }
        A::FocusWindowPrevious => {
            // Most-recently-focused window other than the current one.
            // (Niri additionally commits MRU state here; prism has no
            // MRU UI yet, but the focus timestamps exist.)
            let current = state.layout.focus().map(|m| m.id());
            let previous = state
                .layout
                .windows()
                .map(|(_, m)| m)
                .filter(|m| Some(m.id()) != current)
                .max_by_key(|m| m.get_focus_timestamp())
                .map(|m| m.window.clone());
            if let Some(window) = previous {
                state.layout.activate_window(&window);
                queue_redraw_active_output(state);
            }
        }
        A::FocusWindowInColumn(index) => {
            state.layout.focus_window_in_column(index);
            queue_redraw_active_output(state);
        }
        A::FocusColumnLeft => {
            state.layout.focus_left();
            queue_redraw_active_output(state);
        }
        A::FocusColumnRight => {
            state.layout.focus_right();
            queue_redraw_active_output(state);
        }
        A::FocusColumnFirst => {
            state.layout.focus_column_first();
            queue_redraw_active_output(state);
        }
        A::FocusColumnLast => {
            state.layout.focus_column_last();
            queue_redraw_active_output(state);
        }
        A::FocusColumnRightOrFirst => {
            state.layout.focus_column_right_or_first();
            queue_redraw_active_output(state);
        }
        A::FocusColumnLeftOrLast => {
            state.layout.focus_column_left_or_last();
            queue_redraw_active_output(state);
        }
        A::FocusColumn(index) => {
            state.layout.focus_column(index);
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
        // "…OrMonitor…": stay within the workspace if there's somewhere
        // to go, otherwise hop to the neighboring monitor. The layout
        // method takes the target output and reports (via bool, unused
        // without cursor warping) whether the hop happened.
        A::FocusWindowOrMonitorUp => {
            if let Some(output) = state.output_up() {
                state.layout.focus_window_up_or_output(&output);
            } else {
                state.layout.focus_up();
            }
            queue_redraw_active_output(state);
        }
        A::FocusWindowOrMonitorDown => {
            if let Some(output) = state.output_down() {
                state.layout.focus_window_down_or_output(&output);
            } else {
                state.layout.focus_down();
            }
            queue_redraw_active_output(state);
        }
        A::FocusColumnOrMonitorLeft => {
            if let Some(output) = state.output_left() {
                state.layout.focus_column_left_or_output(&output);
            } else {
                state.layout.focus_left();
            }
            queue_redraw_active_output(state);
        }
        A::FocusColumnOrMonitorRight => {
            if let Some(output) = state.output_right() {
                state.layout.focus_column_right_or_output(&output);
            } else {
                state.layout.focus_right();
            }
            queue_redraw_active_output(state);
        }
        A::FocusWindowDownOrColumnLeft => {
            state.layout.focus_down_or_left();
            queue_redraw_active_output(state);
        }
        A::FocusWindowDownOrColumnRight => {
            state.layout.focus_down_or_right();
            queue_redraw_active_output(state);
        }
        A::FocusWindowUpOrColumnLeft => {
            state.layout.focus_up_or_left();
            queue_redraw_active_output(state);
        }
        A::FocusWindowUpOrColumnRight => {
            state.layout.focus_up_or_right();
            queue_redraw_active_output(state);
        }
        A::FocusWindowOrWorkspaceDown => {
            state.layout.focus_window_or_workspace_down();
            queue_redraw_active_output(state);
        }
        A::FocusWindowOrWorkspaceUp => {
            state.layout.focus_window_or_workspace_up();
            queue_redraw_active_output(state);
        }
        A::FocusWindowTop => {
            state.layout.focus_window_top();
            queue_redraw_active_output(state);
        }
        A::FocusWindowBottom => {
            state.layout.focus_window_bottom();
            queue_redraw_active_output(state);
        }
        A::FocusWindowDownOrTop => {
            state.layout.focus_window_down_or_top();
            queue_redraw_active_output(state);
        }
        A::FocusWindowUpOrBottom => {
            state.layout.focus_window_up_or_bottom();
            queue_redraw_active_output(state);
        }

        // Move column / window within the workspace
        A::MoveColumnLeft => {
            state.layout.move_left();
            queue_redraw_active_output(state);
        }
        A::MoveColumnRight => {
            state.layout.move_right();
            queue_redraw_active_output(state);
        }
        A::MoveColumnToFirst => {
            state.layout.move_column_to_first();
            queue_redraw_active_output(state);
        }
        A::MoveColumnToLast => {
            state.layout.move_column_to_last();
            queue_redraw_active_output(state);
        }
        A::MoveColumnToIndex(index) => {
            state.layout.move_column_to_index(index);
            queue_redraw_active_output(state);
        }
        A::MoveColumnLeftOrToMonitorLeft => {
            if let Some(output) = state.output_left() {
                state.layout.move_column_left_or_to_output(&output);
            } else {
                state.layout.move_left();
            }
            queue_redraw_active_output(state);
        }
        A::MoveColumnRightOrToMonitorRight => {
            if let Some(output) = state.output_right() {
                state.layout.move_column_right_or_to_output(&output);
            } else {
                state.layout.move_right();
            }
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
        A::MoveWindowDownOrToWorkspaceDown => {
            state.layout.move_down_or_to_workspace_down();
            queue_redraw_active_output(state);
        }
        A::MoveWindowUpOrToWorkspaceUp => {
            state.layout.move_up_or_to_workspace_up();
            queue_redraw_active_output(state);
        }
        A::ConsumeOrExpelWindowLeft => {
            state.layout.consume_or_expel_window_left(None);
            queue_redraw_active_output(state);
        }
        A::ConsumeOrExpelWindowLeftById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.consume_or_expel_window_left(Some(&window));
                queue_redraw_active_output(state);
            }
        }
        A::ConsumeOrExpelWindowRight => {
            state.layout.consume_or_expel_window_right(None);
            queue_redraw_active_output(state);
        }
        A::ConsumeOrExpelWindowRightById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.consume_or_expel_window_right(Some(&window));
                queue_redraw_active_output(state);
            }
        }
        A::ConsumeWindowIntoColumn => {
            state.layout.consume_into_column();
            queue_redraw_active_output(state);
        }
        A::ExpelWindowFromColumn => {
            state.layout.expel_from_column();
            queue_redraw_active_output(state);
        }
        A::SwapWindowLeft => {
            state.layout.swap_window_in_direction(ScrollDirection::Left);
            queue_redraw_active_output(state);
        }
        A::SwapWindowRight => {
            state
                .layout
                .swap_window_in_direction(ScrollDirection::Right);
            queue_redraw_active_output(state);
        }
        A::ToggleColumnTabbedDisplay => {
            state.layout.toggle_column_tabbed_display();
            queue_redraw_active_output(state);
        }
        A::SetColumnDisplay(display) => {
            state.layout.set_column_display(display);
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
        A::FocusMonitor(name) => {
            if let Some(out) = state.output_by_name_match(&name) {
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
        A::MoveColumnToMonitor(name) => {
            move_active_column_to(state, state.output_by_name_match(&name))
        }
        // Move a single window (the active one) — leaves the rest of
        // the column behind. Uses `Layout::move_to_output(None, ...)`
        // which picks "the active window" when window=None.
        A::MoveWindowToMonitorLeft => move_active_window_to(state, state.output_left()),
        A::MoveWindowToMonitorRight => move_active_window_to(state, state.output_right()),
        A::MoveWindowToMonitorUp => move_active_window_to(state, state.output_up()),
        A::MoveWindowToMonitorDown => move_active_window_to(state, state.output_down()),
        A::MoveWindowToMonitorPrevious => move_active_window_to(state, state.output_previous()),
        A::MoveWindowToMonitorNext => move_active_window_to(state, state.output_next()),
        A::MoveWindowToMonitor(name) => {
            move_active_window_to(state, state.output_by_name_match(&name))
        }
        A::MoveWindowToMonitorById { id, output } => {
            let Some(output) = state.output_by_name_match(&output) else {
                return;
            };
            if let Some(window) = find_window_by_id(state, id) {
                state
                    .layout
                    .move_to_output(Some(&window), &output, None, ActivateWindow::Smart);
                queue_redraw_active_output(state);
            }
        }
        // Move the active workspace wholesale to another monitor.
        A::MoveWorkspaceToMonitorLeft => move_active_workspace_to(state, state.output_left()),
        A::MoveWorkspaceToMonitorRight => move_active_workspace_to(state, state.output_right()),
        A::MoveWorkspaceToMonitorUp => move_active_workspace_to(state, state.output_up()),
        A::MoveWorkspaceToMonitorDown => move_active_workspace_to(state, state.output_down()),
        A::MoveWorkspaceToMonitorPrevious => {
            move_active_workspace_to(state, state.output_previous())
        }
        A::MoveWorkspaceToMonitorNext => move_active_workspace_to(state, state.output_next()),
        A::MoveWorkspaceToMonitor(name) => {
            move_active_workspace_to(state, state.output_by_name_match(&name))
        }
        A::MoveWorkspaceToMonitorByRef {
            output_name,
            reference,
        } => {
            let Some((old_output, old_idx)) = state.find_output_and_workspace_index(reference)
            else {
                return;
            };
            let Some(new_output) = state.output_by_name_match(&output_name) else {
                return;
            };
            if state
                .layout
                .move_workspace_to_output_by_id(old_idx, old_output, &new_output)
            {
                queue_redraw_active_output(state);
            }
        }

        // Workspace navigation
        A::FocusWorkspaceUp => {
            state.layout.switch_workspace_up();
            queue_redraw_active_output(state);
        }
        A::FocusWorkspaceDown => {
            state.layout.switch_workspace_down();
            queue_redraw_active_output(state);
        }
        A::FocusWorkspace(reference) => {
            let Some((mut output, index)) = state.find_output_and_workspace_index(reference) else {
                return;
            };
            // Same-output focus doesn't need the output hop (and only
            // then does auto-back-and-forth apply).
            if let Some(active) = state.layout.active_output() {
                if output.as_ref() == Some(active) {
                    output = None;
                }
            }
            if let Some(output) = output {
                state.layout.focus_output(&output);
                state.layout.switch_workspace(index);
            } else {
                let auto_back_and_forth = state.config.borrow().input.workspace_auto_back_and_forth;
                if auto_back_and_forth {
                    state.layout.switch_workspace_auto_back_and_forth(index);
                } else {
                    state.layout.switch_workspace(index);
                }
            }
            queue_redraw_active_output(state);
        }
        A::FocusWorkspacePrevious => {
            state.layout.switch_workspace_previous();
            queue_redraw_active_output(state);
        }
        A::MoveWindowToWorkspaceUp(focus) => {
            state.layout.move_to_workspace_up(focus);
            queue_redraw_active_output(state);
        }
        A::MoveWindowToWorkspaceDown(focus) => {
            state.layout.move_to_workspace_down(focus);
            queue_redraw_active_output(state);
        }
        A::MoveWindowToWorkspace(reference, focus) => {
            let Some((mut output, index)) = state.find_output_and_workspace_index(reference) else {
                return;
            };
            // The source is always the active output; same-output moves
            // go through move_to_workspace directly.
            if let Some(active) = state.layout.active_output() {
                if output.as_ref() == Some(active) {
                    output = None;
                }
            }
            let activate = if focus {
                ActivateWindow::Smart
            } else {
                ActivateWindow::No
            };
            if let Some(output) = output {
                state
                    .layout
                    .move_to_output(None, &output, Some(index), activate);
            } else {
                state.layout.move_to_workspace(None, index, activate);
            }
            queue_redraw_active_output(state);
        }
        A::MoveWindowToWorkspaceById {
            window_id,
            reference,
            focus,
        } => {
            let Some(window) = find_window_by_id(state, window_id) else {
                return;
            };
            let Some((output, index)) = state.find_output_and_workspace_index(reference) else {
                return;
            };
            let activate = if focus {
                ActivateWindow::Smart
            } else {
                ActivateWindow::No
            };
            if let Some(output) = output {
                state
                    .layout
                    .move_to_output(Some(&window), &output, Some(index), activate);
            } else {
                state
                    .layout
                    .move_to_workspace(Some(&window), index, activate);
            }
            queue_redraw_active_output(state);
        }
        A::MoveColumnToWorkspaceUp(focus) => {
            state.layout.move_column_to_workspace_up(focus);
            queue_redraw_active_output(state);
        }
        A::MoveColumnToWorkspaceDown(focus) => {
            state.layout.move_column_to_workspace_down(focus);
            queue_redraw_active_output(state);
        }
        A::MoveColumnToWorkspace(reference, focus) => {
            let Some((mut output, index)) = state.find_output_and_workspace_index(reference) else {
                return;
            };
            if let Some(active) = state.layout.active_output() {
                if output.as_ref() == Some(active) {
                    output = None;
                }
            }
            if let Some(output) = output {
                state
                    .layout
                    .move_column_to_output(&output, Some(index), focus);
            } else {
                state.layout.move_column_to_workspace(index, focus);
            }
            queue_redraw_active_output(state);
        }
        A::MoveWorkspaceUp => {
            state.layout.move_workspace_up();
            queue_redraw_active_output(state);
        }
        A::MoveWorkspaceDown => {
            state.layout.move_workspace_down();
            queue_redraw_active_output(state);
        }
        A::MoveWorkspaceToIndex(new_idx) => {
            // 1-based in config.
            state
                .layout
                .move_workspace_to_idx(None, new_idx.saturating_sub(1));
            queue_redraw_active_output(state);
        }
        A::MoveWorkspaceToIndexByRef { new_idx, reference } => {
            let Some(resolved) = state.find_output_and_workspace_index(reference) else {
                return;
            };
            state
                .layout
                .move_workspace_to_idx(Some(resolved), new_idx.saturating_sub(1));
            queue_redraw_active_output(state);
        }
        A::SetWorkspaceName(name) => {
            state.layout.set_workspace_name(name, None);
            queue_redraw_active_output(state);
        }
        A::SetWorkspaceNameByRef { name, reference } => {
            state.layout.set_workspace_name(name, Some(reference));
            queue_redraw_active_output(state);
        }
        A::UnsetWorkspaceName => {
            state.layout.unset_workspace_name(None);
            queue_redraw_active_output(state);
        }
        A::UnsetWorkSpaceNameByRef(reference) => {
            state.layout.unset_workspace_name(Some(reference));
            queue_redraw_active_output(state);
        }

        // Sizing
        A::MaximizeColumn => {
            state.layout.toggle_full_width();
            queue_redraw_active_output(state);
        }
        A::CenterColumn => {
            state.layout.center_column();
            queue_redraw_active_output(state);
        }
        A::CenterWindow => {
            state.layout.center_window(None);
            queue_redraw_active_output(state);
        }
        A::CenterWindowById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.center_window(Some(&window));
                queue_redraw_active_output(state);
            }
        }
        A::CenterVisibleColumns => {
            state.layout.center_visible_columns();
            queue_redraw_active_output(state);
        }
        A::SwitchPresetColumnWidth => {
            // forwards=true == cycle through niri's preset widths in
            // user-facing order; matches Mod+R behaviour.
            state.layout.toggle_width(true);
            queue_redraw_active_output(state);
        }
        A::SwitchPresetColumnWidthBack => {
            state.layout.toggle_width(false);
            queue_redraw_active_output(state);
        }
        A::SwitchPresetWindowWidth => {
            state.layout.toggle_window_width(None, true);
            queue_redraw_active_output(state);
        }
        A::SwitchPresetWindowWidthBack => {
            state.layout.toggle_window_width(None, false);
            queue_redraw_active_output(state);
        }
        A::SwitchPresetWindowWidthById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_window_width(Some(&window), true);
                queue_redraw_active_output(state);
            }
        }
        A::SwitchPresetWindowWidthBackById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_window_width(Some(&window), false);
                queue_redraw_active_output(state);
            }
        }
        A::SwitchPresetWindowHeight => {
            state.layout.toggle_window_height(None, true);
            queue_redraw_active_output(state);
        }
        A::SwitchPresetWindowHeightBack => {
            state.layout.toggle_window_height(None, false);
            queue_redraw_active_output(state);
        }
        A::SwitchPresetWindowHeightById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_window_height(Some(&window), true);
                queue_redraw_active_output(state);
            }
        }
        A::SwitchPresetWindowHeightBackById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_window_height(Some(&window), false);
                queue_redraw_active_output(state);
            }
        }
        A::SetColumnWidth(change) => {
            state.layout.set_column_width(change);
            queue_redraw_active_output(state);
        }
        A::SetWindowWidth(change) => {
            state.layout.set_window_width(None, change);
            queue_redraw_active_output(state);
        }
        A::SetWindowWidthById { id, change } => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.set_window_width(Some(&window), change);
                queue_redraw_active_output(state);
            }
        }
        A::SetWindowHeight(change) => {
            state.layout.set_window_height(None, change);
            queue_redraw_active_output(state);
        }
        A::SetWindowHeightById { id, change } => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.set_window_height(Some(&window), change);
                queue_redraw_active_output(state);
            }
        }
        A::ResetWindowHeight => {
            state.layout.reset_window_height(None);
            queue_redraw_active_output(state);
        }
        A::ResetWindowHeightById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.reset_window_height(Some(&window));
                queue_redraw_active_output(state);
            }
        }
        A::ExpandColumnToAvailableWidth => {
            state.layout.expand_column_to_available_width();
            queue_redraw_active_output(state);
        }

        // Floating
        A::ToggleWindowFloating => {
            state.layout.toggle_window_floating(None);
            queue_redraw_active_output(state);
        }
        A::ToggleWindowFloatingById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.toggle_window_floating(Some(&window));
                queue_redraw_active_output(state);
            }
        }
        A::MoveWindowToFloating => {
            state.layout.set_window_floating(None, true);
            queue_redraw_active_output(state);
        }
        A::MoveWindowToFloatingById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.set_window_floating(Some(&window), true);
                queue_redraw_active_output(state);
            }
        }
        A::MoveWindowToTiling => {
            state.layout.set_window_floating(None, false);
            queue_redraw_active_output(state);
        }
        A::MoveWindowToTilingById(id) => {
            if let Some(window) = find_window_by_id(state, id) {
                state.layout.set_window_floating(Some(&window), false);
                queue_redraw_active_output(state);
            }
        }
        A::FocusFloating => {
            state.layout.focus_floating();
            queue_redraw_active_output(state);
        }
        A::FocusTiling => {
            state.layout.focus_tiling();
            queue_redraw_active_output(state);
        }
        A::SwitchFocusBetweenFloatingAndTiling => {
            state.layout.switch_focus_floating_tiling();
            queue_redraw_active_output(state);
        }
        A::MoveFloatingWindowById { id, x, y } => {
            let window = match id {
                Some(id) => match find_window_by_id(state, id) {
                    Some(w) => Some(w),
                    None => return,
                },
                None => None,
            };
            state
                .layout
                .move_floating_window(window.as_ref(), x, y, true);
            queue_redraw_active_output(state);
        }

        // Window-rule + urgency tweaks
        A::ToggleWindowRuleOpacity => {
            let mut toggled = false;
            if let Some(window) = state
                .layout
                .active_workspace_mut()
                .and_then(|ws| ws.active_window_mut())
            {
                // Only meaningful when a rule actually set an opacity.
                if window.rules().opacity.is_some_and(|o| o != 1.) {
                    window.toggle_ignore_opacity_window_rule();
                    toggled = true;
                }
            }
            if toggled {
                queue_redraw_active_output(state);
            }
        }
        A::ToggleWindowRuleOpacityById(id) => {
            let mut toggled = false;
            if let Some(window) = state
                .layout
                .workspaces_mut()
                .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id))
            {
                if window.rules().opacity.is_some_and(|o| o != 1.) {
                    window.toggle_ignore_opacity_window_rule();
                    toggled = true;
                }
            }
            if toggled {
                queue_redraw_active_output(state);
            }
        }
        A::ToggleWindowUrgent(id) => {
            if let Some(window) = state
                .layout
                .workspaces_mut()
                .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id))
            {
                let urgent = window.is_urgent();
                window.set_urgent(!urgent);
            }
            queue_redraw_active_output(state);
        }
        A::SetWindowUrgent(id) => {
            if let Some(window) = state
                .layout
                .workspaces_mut()
                .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id))
            {
                window.set_urgent(true);
            }
            queue_redraw_active_output(state);
        }
        A::UnsetWindowUrgent(id) => {
            if let Some(window) = state
                .layout
                .workspaces_mut()
                .find_map(|ws| ws.windows_mut().find(|w| w.id().get() == id))
            {
                window.set_urgent(false);
            }
            queue_redraw_active_output(state);
        }

        A::PowerOffMonitors => {
            tracing::info!("action: PowerOffMonitors");
            state.set_all_monitors_powered(false);
        }
        A::PowerOnMonitors => {
            tracing::info!("action: PowerOnMonitors");
            state.set_all_monitors_powered(true);
        }

        // Stubs for actions whose subsystems aren't ported yet — see
        // the module doc for the list and reasons.
        other => {
            tracing::debug!("action: unhandled {other:?}");
        }
    }
}

/// Environment overrides applied to every spawned child — the config
/// `environment {}` block. `(name, Some(value))` sets the var,
/// `(name, None)` removes it. Populated once at startup via
/// [`set_child_env`]; read under a shared lock per spawn. A process-wide
/// static rather than a `PrismState` field because [`spawn`] runs on a
/// detached thread with no access to compositor state.
static CHILD_ENV: RwLock<Vec<(String, Option<String>)>> = RwLock::new(Vec::new());

/// Install the child-environment overrides from the config `environment {}`
/// block. Call once at startup, before spawning anything.
pub fn set_child_env(vars: Vec<(String, Option<String>)>) {
    *CHILD_ENV.write().unwrap() = vars;
}

/// Spawn a child process fully detached from prism. Public so the
/// startup-spawn path in `main` reuses the exact mechanism keybinds use.
///
/// Runs on a short-lived thread: the double-fork below means we must
/// `wait()` for the intermediate child, which is cheap but doesn't belong
/// on the compositor thread. See [`spawn_sync`] for the fork details.
pub fn spawn(args: Vec<String>) {
    if args.is_empty() {
        tracing::warn!("spawn: empty args");
        return;
    }
    if let Err(e) = thread::Builder::new()
        .name("spawn".to_owned())
        .spawn(move || spawn_sync(args))
    {
        tracing::warn!("spawn: could not start spawner thread: {e}");
    }
}

/// Spawn a command through `sh -c`, for `spawn-sh-at-startup` and the
/// `SpawnSh` bind. Hardcoded `sh -c`, consistent with sway/Hyprland/niri.
pub fn spawn_sh(command: String) {
    spawn(vec!["sh".to_string(), "-c".to_string(), command]);
}

/// Build and launch the child. Runs on the spawner thread.
///
/// stdin → /dev/null (no TTY for the child); stdout/stderr inherit prism's
/// so spawn failures land in our log — hiding them behind /dev/null is what
/// made the earlier alacritty / fuzzel "spawn but never appear" failures
/// silent.
fn spawn_sync(args: Vec<String>) {
    let program = args[0].clone();
    let rest = &args[1..];
    tracing::info!("spawn: {program} {rest:?}");

    let mut cmd = Command::new(&program);
    cmd.args(rest);
    cmd.stdin(std::process::Stdio::null());

    // Apply the configured child environment (`environment {}`).
    {
        let env = CHILD_ENV.read().unwrap();
        for (name, value) in env.iter() {
            match value {
                Some(v) => cmd.env(name, v),
                None => cmd.env_remove(name),
            };
        }
    }

    // SAFETY: the closure runs in the forked child, between fork and exec,
    // where it is single-threaded; fork(2)/setsid(2)/_exit(2) and the
    // sigemptyset/sigprocmask below are all async-signal-safe. Double-fork: the intermediate child exits
    // immediately, so the grandchild (which execs the program) reparents to
    // init and never lingers as a zombie of prism. The intermediate is
    // reaped by `wait()` below. setsid detaches the grandchild from prism's
    // session / controlling TTY so it survives compositor exit.
    unsafe {
        use std::os::unix::process::CommandExt as _;
        cmd.pre_exec(|| {
            match libc::fork() {
                -1 => return Err(std::io::Error::last_os_error()),
                0 => {}
                _ => libc::_exit(0),
            }
            libc::setsid();
            // Reset the signal mask to empty. calloop's `Signals` source
            // blocks SIGINT/SIGTERM on prism's main thread, and that blocked
            // mask is inherited across fork+exec — including into a terminal's
            // shell and the apps it runs. Without this, Ctrl-C can't cancel a
            // foreground app: the tty echoes `^C` and raises SIGINT, but it
            // stays blocked/pending and never fires. Mirrors the xwayland
            // satellite's `unblock_all`.
            let mut set: libc::sigset_t = std::mem::zeroed();
            if libc::sigemptyset(&mut set) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::sigprocmask(libc::SIG_SETMASK, &set, std::ptr::null_mut()) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    match cmd.spawn() {
        // Reap the intermediate child (it `_exit(0)`'d right after forking
        // the grandchild). The grandchild lives on, owned by init.
        Ok(mut child) => {
            let _ = child.wait();
        }
        Err(e) => tracing::warn!("spawn {program} failed: {e}"),
    }
}

/// The layout's focused window, as the `W::Id` the layout APIs take
/// (`Mapped::Id = smithay::desktop::Window`, i.e. the `window` field —
/// not the numeric `MappedId`).
fn focused_window(state: &PrismState) -> Option<Window> {
    state.layout.focus().map(|m| m.window.clone())
}

/// Resolve an `…ById(u64)` action target (a `MappedId`, e.g. from a
/// future IPC request) to the layout's window id. `None` if no such
/// window exists — e.g. it closed between request and dispatch.
fn find_window_by_id(state: &PrismState, id: u64) -> Option<Window> {
    state
        .layout
        .windows()
        .find(|(_, m)| m.id().get() == id)
        .map(|(_, m)| m.window.clone())
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

/// Move the active workspace wholesale to `target`. No-op if there's
/// no monitor in that direction (or no workspace to move — the layout
/// reports that via the bool).
fn move_active_workspace_to(state: &mut PrismState, target: Option<Output>) {
    let Some(out) = target else {
        return;
    };
    if state.layout.move_workspace_to_output(&out) {
        queue_redraw_active_output(state);
    }
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
