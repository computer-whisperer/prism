//! Extension traits over smithay's `InputBackend`/`Device`.
//!
//! Every input handler signature in [`super::dispatch`] is generic over a
//! `PrismInputBackend` so the same code can drive libinput on a TTY and
//! the winit virtual backend in dev. The added trait method —
//! [`PrismInputDevice::output`] — answers "which `Output` should this
//! device's coordinates be relative to" (only meaningful for tablets so
//! far; pointers and keyboards return `None`).
//!
//! Ported from niri/src/input/backend_ext.rs. The `VirtualPointer` impl
//! is intentionally omitted — prism does not yet implement the
//! virtual-pointer protocol; re-add it here when that lands.

use ::input as libinput;
use prism_protocols::PrismState;
use smithay::backend::input;
use smithay::output::Output;

pub trait PrismInputBackend: input::InputBackend<Device = Self::PrismDevice> {
    type PrismDevice: PrismInputDevice;
}

impl<T: input::InputBackend> PrismInputBackend for T
where
    Self::Device: PrismInputDevice,
{
    type PrismDevice = Self::Device;
}

pub trait PrismInputDevice: input::Device {
    /// The output that this device's coordinates should be scoped to,
    /// if any. Used by tablet input where the tablet surface maps to a
    /// specific monitor. Pointers and keyboards return `None`.
    //
    // FIXME (mirrored from niri): this is per-device but could be
    // per-event; revisit once libinput needs it.
    fn output(&self, state: &PrismState) -> Option<Output>;
}

impl PrismInputDevice for libinput::Device {
    fn output(&self, _state: &PrismState) -> Option<Output> {
        // FIXME: per-device output mapping (config-driven). niri left
        // this as a TODO too.
        None
    }
}

// NB: niri also impls this for `WinitVirtualDevice` and `VirtualPointer`.
// Re-add either impl here when we grow a winit dev backend / wire the
// wlr virtual-pointer protocol.
