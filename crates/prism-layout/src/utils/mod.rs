//! Small free functions niri's `utils/mod.rs` collected. Subset only —
//! the heavier helpers (xdg-toplevel state munging, KDE-decorations
//! handshakes, output-name resolution) come across as the window/ port
//! gets there. Anything that requires niri's top-level `State` or
//! `ClientState` is deferred until prism has the equivalents wired.

pub mod id;
pub mod region;
pub mod scale;
pub mod transaction;
pub mod vblank_throttle;

use std::time::Duration;

use bitflags::bitflags;
use rustix::time::{clock_gettime, ClockId};
use smithay::input::pointer::CursorIcon;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;

bitflags! {
    /// Interactive-resize edge mask. Direct copy of niri's `ResizeEdge` —
    /// used by `LayoutElement::set_interactive_resize` and the window
    /// configure path to convey which edges are being dragged.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ResizeEdge: u32 {
        const TOP          = 0b0001;
        const BOTTOM       = 0b0010;
        const LEFT         = 0b0100;
        const RIGHT        = 0b1000;

        const TOP_LEFT     = Self::TOP.bits() | Self::LEFT.bits();
        const BOTTOM_LEFT  = Self::BOTTOM.bits() | Self::LEFT.bits();

        const TOP_RIGHT    = Self::TOP.bits() | Self::RIGHT.bits();
        const BOTTOM_RIGHT = Self::BOTTOM.bits() | Self::RIGHT.bits();

        const LEFT_RIGHT   = Self::LEFT.bits() | Self::RIGHT.bits();
        const TOP_BOTTOM   = Self::TOP.bits() | Self::BOTTOM.bits();
    }
}

impl From<xdg_toplevel::ResizeEdge> for ResizeEdge {
    #[inline]
    fn from(x: xdg_toplevel::ResizeEdge) -> Self {
        Self::from_bits(x as u32).unwrap()
    }
}

impl ResizeEdge {
    pub fn cursor_icon(self) -> CursorIcon {
        match self {
            Self::LEFT => CursorIcon::WResize,
            Self::RIGHT => CursorIcon::EResize,
            Self::TOP => CursorIcon::NResize,
            Self::BOTTOM => CursorIcon::SResize,
            Self::TOP_LEFT => CursorIcon::NwResize,
            Self::TOP_RIGHT => CursorIcon::NeResize,
            Self::BOTTOM_RIGHT => CursorIcon::SeResize,
            Self::BOTTOM_LEFT => CursorIcon::SwResize,
            _ => CursorIcon::Default,
        }
    }
}

/// Wall-clock-ish monotonic time. Matches niri's `get_monotonic_time`.
pub fn get_monotonic_time() -> Duration {
    let ts = clock_gettime(ClockId::Monotonic);
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// Round a logical-pixel coordinate to the nearest physical pixel at the
/// given output scale, then convert back to logical. Used for border /
/// shadow alignment so they don't sit on subpixel offsets.
pub fn round_logical_in_physical(scale: f64, logical: f64) -> f64 {
    (logical * scale).round() / scale
}

/// Like [`round_logical_in_physical`] but enforces a minimum of one
/// physical pixel for non-zero inputs. Border thicknesses use this so a
/// 1-px configured border doesn't render at zero on fractional scales.
pub fn round_logical_in_physical_max1(scale: f64, logical: f64) -> f64 {
    if logical == 0. {
        return 0.;
    }
    (logical * scale).max(1.).round() / scale
}

/// Floor-variant of [`round_logical_in_physical_max1`]. Tab-indicator
/// edges use floor to avoid overflowing the tab area on the inside edge.
pub fn floor_logical_in_physical_max1(scale: f64, logical: f64) -> f64 {
    if logical == 0. {
        return 0.;
    }
    (logical * scale).max(1.).floor() / scale
}

/// Run `f` against a toplevel's `XdgToplevelSurfaceRoleAttributes`,
/// holding the lock for the closure's duration. Ported verbatim from
/// niri/src/utils/mod.rs.
pub fn with_toplevel_role<T>(
    toplevel: &smithay::wayland::shell::xdg::ToplevelSurface,
    f: impl FnOnce(&mut smithay::wayland::shell::xdg::XdgToplevelSurfaceRoleAttributes) -> T,
) -> T {
    smithay::wayland::compositor::with_states(toplevel.wl_surface(), |states| {
        let mut role = states
            .data_map
            .get::<smithay::wayland::shell::xdg::XdgToplevelSurfaceData>()
            .unwrap()
            .lock()
            .unwrap();

        f(&mut role)
    })
}
