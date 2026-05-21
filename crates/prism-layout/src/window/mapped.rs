//! Mapped (post-initial-configure) window — minimal scaffold.
//!
//! Full port from `niri/src/window/mapped.rs` (1431 LOC of state machine
//! + LayoutElement impl + render path) is deferred until step 7 of the
//! niri port lands — porting `Mapped` standalone is wasted effort because
//! its `LayoutElement` impl is heavily coupled to `tile.rs`/`workspace.rs`
//! (interactive resize, configure throttling, focus-ring sizing, etc.).
//! Doing them together avoids two passes over the same code.
//!
//! Today this carries only enough surface for `super::WindowRef` to
//! compile: `toplevel()` and the boolean predicates `WindowRef` forwards
//! to. Everything else is a placeholder; constructing a `Mapped` will
//! `unimplemented!` until step 7 fleshes it out.

use smithay::desktop::Window;
use smithay::wayland::shell::xdg::ToplevelSurface;

#[derive(Debug)]
pub struct Mapped {
    pub window: Window,
}

impl Mapped {
    pub fn toplevel(&self) -> &ToplevelSurface {
        self.window.toplevel().expect("no X11 support")
    }

    pub fn is_focused(&self) -> bool {
        false
    }

    pub fn is_urgent(&self) -> bool {
        false
    }

    pub fn is_active_in_column(&self) -> bool {
        true
    }

    pub fn is_floating(&self) -> bool {
        false
    }

    pub fn is_window_cast_target(&self) -> bool {
        false
    }
}
