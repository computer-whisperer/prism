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

/// Returns whether `output`'s name matches the user-supplied
/// `target` (an output name from the config). Ported from niri.
pub fn output_matches_name(output: &smithay::output::Output, target: &str) -> bool {
    let name = output
        .user_data()
        .get::<prism_config::OutputName>()
        .unwrap();
    name.matches(target)
}

/// Logical-space size of `output` (applying the output's transform).
/// Niri's helper, ported verbatim from niri/src/utils.
pub fn output_size(
    output: &smithay::output::Output,
) -> smithay::utils::Size<f64, smithay::utils::Logical> {
    let output_scale = output.current_scale().fractional_scale();
    let output_transform = output.current_transform();
    let output_mode = output.current_mode().unwrap();
    let logical_size = output_mode.size.to_f64().to_logical(output_scale);
    output_transform.transform_size(logical_size)
}

/// "Baba is float" bob — tiny vertical offset based on wall time, used
/// for the joke window-rule that makes windows bob up and down.
/// Ported verbatim from niri/src/utils/mod.rs.
pub fn baba_is_float_offset(now: std::time::Duration, view_height: f64) -> f64 {
    let now = now.as_secs_f64();
    let amplitude = view_height / 96.;
    amplitude * ((core::f64::consts::TAU * now / 3.6).sin() - 1.)
}

/// Move `rect` so it sits within `area`, preferring to keep its
/// top-left corner anchored at the area's top-left if the rect is
/// larger than the area. Used by floating window placement to keep
/// windows on-screen across output reconfigure / display rotation.
/// Ported verbatim from niri/src/utils.
pub fn clamp_preferring_top_left_in_area(
    area: smithay::utils::Rectangle<f64, smithay::utils::Logical>,
    rect: &mut smithay::utils::Rectangle<f64, smithay::utils::Logical>,
) {
    rect.loc.x = f64::min(rect.loc.x, area.loc.x + area.size.w - rect.size.w);
    rect.loc.y = f64::min(rect.loc.y, area.loc.y + area.size.h - rect.size.h);
    rect.loc.x = f64::max(rect.loc.x, area.loc.x);
    rect.loc.y = f64::max(rect.loc.y, area.loc.y);
}

/// Return the top-left location that centers a `size`-sized box inside
/// `area`, but pinned to the area's top-left if the box is bigger than
/// the area on either axis. Used for placing floating windows
/// initially. Ported verbatim from niri/src/utils.
pub fn center_preferring_top_left_in_area(
    area: smithay::utils::Rectangle<f64, smithay::utils::Logical>,
    size: smithay::utils::Size<f64, smithay::utils::Logical>,
) -> smithay::utils::Point<f64, smithay::utils::Logical> {
    let area_size = area.size.to_point();
    let size = size.to_point();
    let mut offset = (area_size - size).downscale(2.);
    offset.x = f64::max(offset.x, 0.);
    offset.y = f64::max(offset.y, 0.);
    area.loc + offset
}

/// Clamp `x` to the `[min_size, max_size]` range, treating 0 as
/// "unbounded" for whichever endpoint is zero. Used to honor
/// xdg_toplevel min/max-size hints. Ported from niri/src/utils.
pub fn ensure_min_max_size(mut x: i32, min_size: i32, max_size: i32) -> i32 {
    if max_size > 0 {
        x = x.min(max_size);
    }
    if min_size > 0 {
        x = x.max(min_size);
    }
    x
}

/// Like [`ensure_min_max_size`] but special-cases `x == 0` (the
/// "configure picks a sensible size" sentinel): if the toplevel has a
/// fixed-size min==max, use that; otherwise stay zero. Ported from
/// niri/src/utils.
pub fn ensure_min_max_size_maybe_zero(x: i32, min_size: i32, max_size: i32) -> i32 {
    if x != 0 {
        ensure_min_max_size(x, min_size, max_size)
    } else if min_size > 0 && min_size == max_size {
        min_size
    } else {
        0
    }
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

/// Like [`with_toplevel_role`] but also exposes the toplevel's most
/// recently *acked* server-side state (a snapshot of the size/state
/// we configured the client with). Ported from niri/src/utils.
pub fn with_toplevel_role_and_current<T>(
    toplevel: &smithay::wayland::shell::xdg::ToplevelSurface,
    f: impl FnOnce(
        &mut smithay::wayland::shell::xdg::XdgToplevelSurfaceRoleAttributes,
        Option<&smithay::wayland::shell::xdg::ToplevelState>,
    ) -> T,
) -> T {
    smithay::wayland::compositor::with_states(toplevel.wl_surface(), |states| {
        let mut role = states
            .data_map
            .get::<smithay::wayland::shell::xdg::XdgToplevelSurfaceData>()
            .unwrap()
            .lock()
            .unwrap();

        let mut guard = states
            .cached_state
            .get::<smithay::wayland::shell::xdg::ToplevelCachedState>();
        let current = guard.current().last_acked.as_ref().map(|c| &c.state);

        f(&mut role, current)
    })
}

/// Look up the most recent in-flight (not-yet-acked) configure for a
/// toplevel. Used by the configure-throttling logic to decide whether
/// a new configure would be a redundant duplicate. Ported from
/// niri/src/utils.
pub fn with_toplevel_last_uncommitted_configure<T>(
    toplevel: &smithay::wayland::shell::xdg::ToplevelSurface,
    f: impl FnOnce(Option<&smithay::wayland::shell::xdg::ToplevelConfigure>) -> T,
) -> T {
    smithay::wayland::compositor::with_states(toplevel.wl_surface(), |states| {
        let role = states
            .data_map
            .get::<smithay::wayland::shell::xdg::XdgToplevelSurfaceData>()
            .unwrap()
            .lock()
            .unwrap();

        let mut guard = states
            .cached_state
            .get::<smithay::wayland::shell::xdg::ToplevelCachedState>();

        if let Some(last_pending) = role.pending_configures().last() {
            f(Some(last_pending))
        } else if let Some(last_acked) = &role.last_acked {
            let mut configure = Some(last_acked);
            if let Some(committed) = &guard.current().last_acked {
                if committed.serial.is_no_older_than(&last_acked.serial) {
                    configure = None;
                }
            }
            f(configure)
        } else {
            f(None)
        }
    })
}

/// Pump a surface's preferred output scale + transform out to the
/// client via wl_surface enter/leave + the fractional-scale protocol.
/// Ported verbatim from niri/src/utils.
pub fn send_scale_transform(
    surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    data: &smithay::wayland::compositor::SurfaceData,
    scale: smithay::output::Scale,
    transform: smithay::utils::Transform,
) {
    smithay::wayland::compositor::send_surface_state(surface, data, scale.integer_scale(), transform);
    smithay::wayland::fractional_scale::with_fractional_scale(data, |fractional| {
        fractional.set_preferred_scale(scale.fractional_scale());
    });
}

/// Configure the xdg-toplevel's "tiled" state hint based on the
/// user's CSD preference. Niri also consults `KdeDecorationsModeState`
/// (which lives in niri/src/handlers and isn't ported yet); prism's
/// version pretends the client never bound the KDE decoration global,
/// which is the correct behavior for clients that don't use it. When
/// the handlers port lands, the KDE branch will reappear here.
pub fn update_tiled_state(
    toplevel: &smithay::wayland::shell::xdg::ToplevelSurface,
    prefer_no_csd: bool,
    force_tiled: Option<bool>,
) {
    use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1;
    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;

    let should_tile = || {
        if let Some(mode) = toplevel.with_pending_state(|state| state.decoration_mode) {
            mode == zxdg_toplevel_decoration_v1::Mode::ServerSide
        } else {
            // niri also peeks at KdeDecorationsModeState here; until
            // the handlers port lands we treat the client as never
            // having bound the KDE decoration global.
            prefer_no_csd
        }
    };

    let should_tile = force_tiled.unwrap_or_else(should_tile);

    toplevel.with_pending_state(|state| {
        if should_tile {
            state.states.set(xdg_toplevel::State::TiledLeft);
            state.states.set(xdg_toplevel::State::TiledRight);
            state.states.set(xdg_toplevel::State::TiledTop);
            state.states.set(xdg_toplevel::State::TiledBottom);
        } else {
            state.states.unset(xdg_toplevel::State::TiledLeft);
            state.states.unset(xdg_toplevel::State::TiledRight);
            state.states.unset(xdg_toplevel::State::TiledTop);
            state.states.unset(xdg_toplevel::State::TiledBottom);
        }
    });
}

/// Fetch the OS-level (pid/uid/gid) credentials of the client that
/// owns `surface`. Niri caches these on `ClientState`; until that's
/// ported we return `None`, which downstream code treats as "client
/// credentials are unknown" — disables a few window-rule
/// process-matching predicates but is otherwise harmless.
pub fn get_credentials_for_surface(
    _surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
) -> Option<wayland_backend::server::Credentials> {
    // TODO: re-introduce ClientState lookup once the handlers port
    // lands; see niri/src/utils::get_credentials_for_surface.
    None
}
