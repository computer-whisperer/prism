//! Input-related state carried on `PrismState`.
//!
//! Distinct from `redraw.rs` (per-output redraw state machine) — this
//! is per-seat state that input dispatch reads and writes. The actual
//! dispatch logic lives in the `prism-input` crate; this file just
//! declares the types the dispatcher mutates.
//!
//! Currently slim because most of niri's `KeyboardFocus` variants
//! correspond to subsystems prism does not yet have (lock screen,
//! screenshot UI, exit-confirm dialog, overview, MRU). We keep the enum
//! form rather than collapsing to `Option<WlSurface>` so the grow path
//! back to parity is mechanical.

use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;

/// What the keyboard's events should be routed to.
///
/// This is the *effective* focus — the surface smithay's keyboard handle
/// is pointed at right now. It is computed by
/// [`PrismState::update_keyboard_focus`](crate::state::PrismState::update_keyboard_focus),
/// which arbitrates between layer-shell surfaces and the layout's focused
/// window. The layout-window candidate is read live from the layout's active
/// window (`Layout::focus`), so keyboard focus is derived from layout state
/// rather than stored separately.
#[derive(Debug, Clone)]
pub enum KeyboardFocus {
    /// The layout owns focus. `surface` is the focused window's
    /// toplevel surface (if any window is mapped), else `None`.
    Layout { surface: Option<WlSurface> },
    /// A `wlr_layer_shell` surface owns focus — either it advertised
    /// `Exclusive` interactivity (launcher / lock-style grab) or the user
    /// clicked an `OnDemand` surface. `surface` is the layer surface's
    /// role wl_surface.
    LayerShell { surface: WlSurface },
    /// The overview owns focus: no client surface receives keys; the
    /// input path routes unmodified Esc/Return/arrows to the hardcoded
    /// overview binds (niri's `hardcoded_overview_bind`).
    Overview,
    /// The session is locked (ext-session-lock-v1). `surface` is the
    /// lock surface keys go to — the one on the output under the cursor
    /// (falling back to the active/first output), `None` while the lock
    /// client has no surface there (keys then go to no client; binds
    /// are still filtered by `allow-when-locked`).
    LockScreen { surface: Option<WlSurface> },
    // Variants to re-add as the corresponding subsystems land:
    //   ScreenshotUi                            — niri-style overlay
    //   ExitConfirmDialog                       — niri-style overlay
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
            KeyboardFocus::LayerShell { surface } => Some(surface),
            KeyboardFocus::Overview => None,
            KeyboardFocus::LockScreen { surface } => surface.as_ref(),
        }
    }

    pub fn into_surface(self) -> Option<WlSurface> {
        match self {
            KeyboardFocus::Layout { surface } => surface,
            KeyboardFocus::LayerShell { surface } => Some(surface),
            KeyboardFocus::Overview => None,
            KeyboardFocus::LockScreen { surface } => surface,
        }
    }

    pub fn is_layout(&self) -> bool {
        matches!(self, KeyboardFocus::Layout { .. })
    }

    pub fn is_layer_shell(&self) -> bool {
        matches!(self, KeyboardFocus::LayerShell { .. })
    }

    pub fn is_overview(&self) -> bool {
        matches!(self, KeyboardFocus::Overview)
    }

    pub fn is_lock_screen(&self) -> bool {
        matches!(self, KeyboardFocus::LockScreen { .. })
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
