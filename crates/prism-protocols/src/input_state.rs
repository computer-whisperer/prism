//! Input-related state carried on `PrismState`.
//!
//! Distinct from `redraw.rs` (per-output redraw state machine) — this
//! is per-seat state that input dispatch reads and writes. The actual
//! dispatch logic lives in the `prism-input` crate; this file just
//! declares the types the dispatcher mutates.
//!
//! Currently slim because most of niri's `KeyboardFocus` variants
//! correspond to subsystems prism does not yet have (layer-shell, lock
//! screen, screenshot UI, exit-confirm dialog, overview, MRU). We
//! keep the enum form rather than collapsing to `Option<WlSurface>`
//! so the grow path back to parity is mechanical.

use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;

/// What the keyboard's events should be routed to.
#[derive(Debug, Clone)]
pub enum KeyboardFocus {
    /// The layout owns focus. `surface` is the focused window's
    /// toplevel surface (if any window is mapped), else `None`.
    Layout { surface: Option<WlSurface> },
    // Variants to re-add as the corresponding subsystems land:
    //   LayerShell { surface: WlSurface }      — wlr-layer-shell
    //   LockScreen { surface: Option<WlSurface> } — ext-session-lock
    //   ScreenshotUi                            — niri-style overlay
    //   ExitConfirmDialog                       — niri-style overlay
    //   Overview                                — niri-style overlay
    //   Mru                                     — alt-tab style switcher
}

impl Default for KeyboardFocus {
    fn default() -> Self {
        KeyboardFocus::Layout { surface: None }
    }
}

impl KeyboardFocus {
    pub fn surface(&self) -> Option<&WlSurface> {
        match self {
            KeyboardFocus::Layout { surface } => surface.as_ref(),
        }
    }

    pub fn into_surface(self) -> Option<WlSurface> {
        match self {
            KeyboardFocus::Layout { surface } => surface,
        }
    }

    pub fn is_layout(&self) -> bool {
        matches!(self, KeyboardFocus::Layout { .. })
    }
}

/// How the cursor is treated for hit-testing and rendering.
///
/// Ported from niri verbatim — auto-hide UX wants all three states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointerVisibility {
    /// The pointer is visible.
    #[default]
    Visible,
    /// The pointer is invisible but retains focus (used briefly after
    /// auto-hide so tooltips stay open and grabs stay alive).
    Hidden,
    /// The pointer is invisible and cannot focus. Set after touch
    /// input, or when contents under a Hidden pointer change.
    Disabled,
}

impl PointerVisibility {
    pub fn is_visible(&self) -> bool {
        matches!(self, Self::Visible)
    }
}
