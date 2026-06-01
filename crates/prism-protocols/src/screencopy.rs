//! `zwlr_screencopy_v1` — output screenshot / capture for `grim`, `wf-recorder`,
//! `xdg-desktop-portal-wlr`, etc.
//!
//! Hand-rolled (smithay ships no screencopy). This is phase 2 of prism's
//! capture support (see `docs/screen-capture.md`): **SHM, synchronous**. When a
//! client calls `copy`, we capture the output's last composited frame *right
//! then* — `Renderer::capture` runs the sRGB capture encode on the persistent
//! intermediate and reads it back — and memcpy it into the client's `wl_shm`
//! buffer. `copy_with_damage` is serviced the same way (full-frame, full
//! damage). The damage-queued / dmabuf / PipeWire-screencast paths are later
//! phases.
//!
//! We advertise `Xrgb8888` SHM only (no `linux_dmabuf`), and capture into
//! `B8G8R8A8_UNORM` (memory `B,G,R,A`) so the bytes drop straight into an
//! `Xrgb8888` buffer with no CPU swizzle.
//!
//! Style mirrors [`crate::output_power`]: `GlobalDispatch` / `Dispatch` are
//! implemented directly on [`PrismState`], so there's no handler trait and no
//! `delegate_*` macro — the global is registered with
//! [`create_screencopy_global`] and the per-frame state lives in the frame
//! object's user data.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use prism_renderer::vk;
use smithay::output::Output;
use smithay::reexports::wayland_protocols_wlr::screencopy::v1::server::{
    zwlr_screencopy_frame_v1::{self, Flags, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::{self, ZwlrScreencopyManagerV1},
};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use smithay::wayland::shm::with_buffer_contents_mut;
use tracing::trace;

use crate::state::{OutputId, PrismState};

/// Protocol version we advertise. v3 adds `buffer_done` and the `damage` event,
/// which `copy_with_damage` clients (e.g. the wlr portal) expect.
const VERSION: u32 = 3;

/// Capture target format: memory order `B,G,R,A`, which matches `Xrgb8888`
/// (alpha lands in the ignored `X` byte). Lets the readback memcpy straight
/// into the client buffer.
const CAPTURE_VK_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Create the `zwlr_screencopy_manager_v1` global. The display keeps it alive;
/// we never remove it.
pub fn create_screencopy_global(dh: &DisplayHandle) {
    dh.create_global::<PrismState, ZwlrScreencopyManagerV1, ()>(VERSION, ());
}

/// Per-`zwlr_screencopy_frame_v1` user data. `Failed` ⇒ inert (we already sent
/// `failed`). `Pending` carries what a later `copy` needs.
pub enum ScreencopyFrameData {
    Failed,
    Pending {
        output_id: OutputId,
        /// Top-left of the captured region within the output, physical px
        /// (`0,0` for a full-output capture).
        region_x: i32,
        region_y: i32,
        /// Captured region size == client buffer size, physical px.
        width: i32,
        height: i32,
        /// Whether the client asked for the cursor composited in. Not yet
        /// honored (the cursor is a hardware plane, not in the intermediate);
        /// tracked so the eventual implementation has it. See the doc.
        overlay_cursor: bool,
        /// Set once `copy`/`copy_with_damage` is accepted; a second copy is a
        /// protocol error.
        copied: AtomicBool,
    },
}

impl GlobalDispatch<ZwlrScreencopyManagerV1, ()> for PrismState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwlrScreencopyManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwlrScreencopyManagerV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _manager: &ZwlrScreencopyManagerV1,
        request: <ZwlrScreencopyManagerV1 as Resource>::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use zwlr_screencopy_manager_v1::Request;

        // Resolve `output` → our connector id, and read its physical size.
        // Returns the smithay Output too (region math needs its scale).
        let resolve = |state: &PrismState, output: &_| -> Option<(OutputId, Output, i32, i32)> {
            let smithay_output = Output::from_resource(output)?;
            let id = smithay_output.name();
            let ctx = state.outputs.get(&id)?;
            let (w, h) = (ctx.extent.width as i32, ctx.extent.height as i32);
            Some((id, smithay_output, w, h))
        };

        let (frame, overlay_cursor, region_x, region_y, width, height, output_id) = match request {
            Request::CaptureOutput {
                frame,
                overlay_cursor,
                output,
            } => {
                let Some((id, _out, w, h)) = resolve(state, &output) else {
                    trace!("screencopy: capture of unknown/unrendered output");
                    let frame = data_init.init(frame, ScreencopyFrameData::Failed);
                    frame.failed();
                    return;
                };
                (frame, overlay_cursor, 0, 0, w, h, id)
            }
            Request::CaptureOutputRegion {
                frame,
                overlay_cursor,
                output,
                x,
                y,
                width,
                height,
            } => {
                if width <= 0 || height <= 0 {
                    trace!("screencopy: invalid region size");
                    let frame = data_init.init(frame, ScreencopyFrameData::Failed);
                    frame.failed();
                    return;
                }
                let Some((id, out, ow, oh)) = resolve(state, &output) else {
                    trace!("screencopy: region capture of unknown/unrendered output");
                    let frame = data_init.init(frame, ScreencopyFrameData::Failed);
                    frame.failed();
                    return;
                };
                // Region is given in output-logical coords; scale to physical.
                // Transform is assumed Normal (phase-2 simplification — see doc).
                let scale = out.current_scale().fractional_scale();
                let rx = (x as f64 * scale).round() as i32;
                let ry = (y as f64 * scale).round() as i32;
                let rw = (width as f64 * scale).round() as i32;
                let rh = (height as f64 * scale).round() as i32;
                // Clamp to the output rect.
                let x0 = rx.clamp(0, ow);
                let y0 = ry.clamp(0, oh);
                let x1 = (rx + rw).clamp(0, ow);
                let y1 = (ry + rh).clamp(0, oh);
                if x1 <= x0 || y1 <= y0 {
                    trace!("screencopy: region outside output");
                    let frame = data_init.init(frame, ScreencopyFrameData::Failed);
                    frame.failed();
                    return;
                }
                (frame, overlay_cursor, x0, y0, x1 - x0, y1 - y0, id)
            }
            Request::Destroy => return,
            _ => return,
        };

        let frame = data_init.init(
            frame,
            ScreencopyFrameData::Pending {
                output_id,
                region_x,
                region_y,
                width,
                height,
                overlay_cursor: overlay_cursor != 0,
                copied: AtomicBool::new(false),
            },
        );

        // Advertise the SHM buffer we want: Xrgb8888, region-sized, tight stride.
        frame.buffer(
            wl_shm::Format::Xrgb8888,
            width as u32,
            height as u32,
            (width * 4) as u32,
        );
        // v3: signal end of format enumeration. We deliberately send no
        // `linux_dmabuf` — SHM only this phase, so clients fall back to it.
        if frame.version() >= 3 {
            frame.buffer_done();
        }
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ScreencopyFrameData> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        frame: &ZwlrScreencopyFrameV1,
        request: <ZwlrScreencopyFrameV1 as Resource>::Request,
        data: &ScreencopyFrameData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        use zwlr_screencopy_frame_v1::Request;

        let (buffer, with_damage) = match request {
            Request::Copy { buffer } => (buffer, false),
            Request::CopyWithDamage { buffer } => (buffer, true),
            Request::Destroy => return,
            _ => return,
        };

        let ScreencopyFrameData::Pending {
            output_id,
            region_x,
            region_y,
            width,
            height,
            copied,
            ..
        } = data
        else {
            // Failed frame — ignore further requests.
            return;
        };

        if copied.swap(true, Ordering::SeqCst) {
            frame.post_error(
                zwlr_screencopy_frame_v1::Error::AlreadyUsed,
                "copy was already requested",
            );
            return;
        }

        state.service_screencopy_shm(
            frame,
            output_id,
            (*region_x, *region_y, *width, *height),
            &buffer,
            with_damage,
        );
    }
}

impl PrismState {
    /// Synchronously fulfill a screencopy `copy` into a `wl_shm` buffer: capture
    /// the output now, validate + fill the client buffer, send `ready` (or
    /// `failed`). `region` is `(x, y, w, h)` in physical px within the output.
    fn service_screencopy_shm(
        &mut self,
        frame: &ZwlrScreencopyFrameV1,
        output_id: &OutputId,
        region: (i32, i32, i32, i32),
        buffer: &WlBuffer,
        with_damage: bool,
    ) {
        let (rx, ry, rw, rh) = region;

        // Capture the full output (the renderer always captures the whole
        // intermediate); we crop to the region during the copy below. The
        // owned CaptureImage drops the `outputs` borrow before we touch the
        // client buffer.
        let capture = {
            let Some(ctx) = self.outputs.get_mut(output_id) else {
                trace!("screencopy: output {output_id} gone before copy");
                frame.failed();
                return;
            };
            let white = ctx.effective_sdr_reference_nits();
            match ctx.renderer.capture(CAPTURE_VK_FORMAT, white) {
                Ok(c) => c,
                Err(e) => {
                    trace!("screencopy: capture failed: {e}");
                    frame.failed();
                    return;
                }
            }
        };

        let cap_w = capture.width as i32;
        let cap_h = capture.height as i32;
        // The region must lie within the captured output (it was clamped at
        // request time, but the mode could have changed since).
        if rx < 0 || ry < 0 || rx + rw > cap_w || ry + rh > cap_h {
            trace!("screencopy: region no longer fits output");
            frame.failed();
            return;
        }

        let src_stride = (capture.width * 4) as usize;
        let row_bytes = (rw * 4) as usize;
        let region_off = (ry as usize * src_stride) + rx as usize * 4;

        // Copy region rows into the client buffer. The closure returns
        // `Err(())` on a buffer that doesn't match what we advertised.
        let copy_result = with_buffer_contents_mut(buffer, |ptr, len, bdata| {
            if bdata.format != wl_shm::Format::Xrgb8888
                || bdata.width != rw
                || bdata.height != rh
                || bdata.stride < rw * 4
            {
                return Err(());
            }
            let dst_stride = bdata.stride as usize;
            let dst_off = bdata.offset as usize;
            for row in 0..rh as usize {
                let src_start = region_off + row * src_stride;
                let dst_start = dst_off + row * dst_stride;
                if src_start + row_bytes > capture.pixels.len() || dst_start + row_bytes > len {
                    return Err(());
                }
                // SAFETY: bounds checked just above; src and dst are distinct
                // allocations (owned Vec vs. the mapped shm pool).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        capture.pixels.as_ptr().add(src_start),
                        ptr.add(dst_start),
                        row_bytes,
                    );
                }
            }
            Ok(())
        });

        match copy_result {
            Ok(Ok(())) => {}
            Ok(Err(())) => {
                frame.post_error(
                    zwlr_screencopy_frame_v1::Error::InvalidBuffer,
                    "buffer does not match the advertised format/size",
                );
                return;
            }
            Err(e) => {
                trace!("screencopy: shm access error: {e:?}");
                frame.failed();
                return;
            }
        }

        // Our capture is top-down with the same orientation as scanout, so no
        // Y-invert. For v3 with_damage clients, report the whole region damaged.
        frame.flags(Flags::empty());
        if with_damage && frame.version() >= 3 {
            frame.damage(0, 0, rw as u32, rh as u32);
        }
        let (sec_hi, sec_lo, nsec) = monotonic_timestamp();
        frame.ready(sec_hi, sec_lo, nsec);
    }
}

/// Current CLOCK_MONOTONIC time split into the `(tv_sec_hi, tv_sec_lo,
/// tv_nsec)` triple the `ready` event wants.
fn monotonic_timestamp() -> (u32, u32, u32) {
    let d = clock_monotonic_now();
    let secs = d.as_secs();
    (
        (secs >> 32) as u32,
        (secs & 0xFFFF_FFFF) as u32,
        d.subsec_nanos(),
    )
}

fn clock_monotonic_now() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid pointer to a timespec; CLOCK_MONOTONIC is always available.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    if rc != 0 {
        return Duration::ZERO;
    }
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}
