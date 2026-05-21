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

use rustix::time::{clock_gettime, ClockId};

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
