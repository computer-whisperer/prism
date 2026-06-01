//! `zwlr_screencopy_v1` — output screenshot / capture for `grim`, `wf-recorder`,
//! `xdg-desktop-portal-wlr`, etc.
//!
//! Hand-rolled (smithay ships no screencopy). See `docs/screen-capture.md`.
//! Two buffer paths, both rendering the sRGB capture encode of the output's
//! persistent intermediate (`Renderer`):
//!
//! - **SHM (synchronous)** — on `copy`, capture into an offscreen, read it back,
//!   and memcpy into the client `wl_shm` buffer, then `ready`. Supports whole-
//!   output and region capture (region cropped on the CPU copy). Blocks briefly
//!   on `queue_wait_idle`.
//! - **dmabuf (asynchronous, zero-copy)** — on `copy` with a `linux_dmabuf`
//!   buffer, import it as a `COLOR_ATTACHMENT` and render the capture straight
//!   into it; `ready` fires from a calloop sync_fd source once the GPU finishes
//!   (no stall). Whole-output only. This is the path recording clients want.
//!
//! We advertise `Xrgb8888` for both, and use `B8G8R8A8_UNORM` (memory
//! `B,G,R,A`) so the bytes match an `Xrgb8888` buffer with no swizzle. dmabuf is
//! advertised only for whole-output captures (the encode renders a full frame).
//!
//! Style mirrors [`crate::output_power`]: `GlobalDispatch` / `Dispatch` are
//! implemented directly on [`PrismState`], so there's no handler trait and no
//! `delegate_*` macro — the global is registered with
//! [`create_screencopy_global`] and the per-frame state lives in the frame
//! object's user data.

use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use prism_renderer::{vk, ImportedImage};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Fourcc};
use smithay::output::Output;
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{Interest, Mode, PostAction};
use smithay::reexports::wayland_protocols_wlr::screencopy::v1::server::{
    zwlr_screencopy_frame_v1::{self, Flags, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::{self, ZwlrScreencopyManagerV1},
};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use smithay::wayland::dmabuf::get_dmabuf;
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

        // `allow_dmabuf`: we only offer the zero-copy dmabuf path for
        // whole-output captures — the encode renders a full-output frame, so a
        // region (smaller, offset) buffer can't be filled without scaled
        // sampling. Region capture stays SHM-only (cropped on the CPU copy).
        let (frame, overlay_cursor, region_x, region_y, width, height, output_id, allow_dmabuf) =
            match request {
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
                    (frame, overlay_cursor, 0, 0, w, h, id, true)
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
                    (frame, overlay_cursor, x0, y0, x1 - x0, y1 - y0, id, false)
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
        // For whole-output captures also offer a dmabuf (zero-copy, async) — the
        // client picks whichever it prefers. v3 also needs `buffer_done` to end
        // enumeration. (v < 3 had no buffer_done and no dmabuf event.)
        if frame.version() >= 3 {
            if allow_dmabuf {
                frame.linux_dmabuf(Fourcc::Xrgb8888 as u32, width as u32, height as u32);
            }
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

        // A dmabuf client buffer → zero-copy async path; otherwise SHM.
        if let Ok(dmabuf) = get_dmabuf(&buffer) {
            state.service_screencopy_dmabuf(
                frame,
                output_id,
                (*region_x, *region_y, *width, *height),
                dmabuf,
                with_damage,
            );
        } else {
            state.service_screencopy_shm(
                frame,
                output_id,
                (*region_x, *region_y, *width, *height),
                &buffer,
                with_damage,
            );
        }
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

    /// Fulfill a screencopy `copy` into a client **dmabuf**: import it as a
    /// render target, render the capture encode straight into it (zero-copy),
    /// and fire `ready` asynchronously once the GPU's sync_fd signals — no
    /// `queue_wait_idle` stall. Only whole-output captures reach here (region
    /// captures don't advertise dmabuf). `region` is `(x, y, w, h)`; for a
    /// whole-output capture `x==y==0` and `w×h` is the full output.
    fn service_screencopy_dmabuf(
        &mut self,
        frame: &ZwlrScreencopyFrameV1,
        output_id: &OutputId,
        region: (i32, i32, i32, i32),
        dmabuf: &Dmabuf,
        with_damage: bool,
    ) {
        let (rx, ry, rw, rh) = region;

        // Validate the client dmabuf matches what we advertised.
        if dmabuf.format().code != Fourcc::Xrgb8888
            || dmabuf.width() != rw as u32
            || dmabuf.height() != rh as u32
            || rx != 0
            || ry != 0
        {
            frame.post_error(
                zwlr_screencopy_frame_v1::Error::InvalidBuffer,
                "dmabuf does not match the advertised format/size",
            );
            return;
        }

        // Import + render. The owned `ImportedImage` must outlive the GPU work
        // (it's moved into the completion source below).
        let (imported, sync_fd) = {
            let Some(ctx) = self.outputs.get_mut(output_id) else {
                trace!("screencopy: output {output_id} gone before dmabuf copy");
                frame.failed();
                return;
            };
            let device = ctx.renderer.device();
            let prism_dmabuf = match prism_frame::Dmabuf::from_smithay(dmabuf) {
                Ok(d) => d,
                Err(e) => {
                    trace!("screencopy: dmabuf clone failed: {e}");
                    frame.failed();
                    return;
                }
            };
            // Import with the capture format (B8G8R8A8 = Xrgb8888 memory order)
            // as a COLOR_ATTACHMENT so we can render into it.
            let imported = match ImportedImage::import(
                device,
                &prism_dmabuf,
                CAPTURE_VK_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ) {
                Ok(i) => i,
                Err(e) => {
                    trace!("screencopy: dmabuf import (COLOR_ATTACHMENT) failed: {e}");
                    frame.failed();
                    return;
                }
            };
            let white = ctx.effective_sdr_reference_nits();
            match ctx.renderer.capture_into_dmabuf(&imported, white) {
                Ok(fd) => (imported, fd),
                Err(e) => {
                    trace!("screencopy: capture_into_dmabuf failed: {e}");
                    frame.failed();
                    return;
                }
            }
        };

        // Park the in-flight capture on PrismState (NOT in the calloop closure):
        // the GPU is still writing `imported`, so it must outlive the wait, and
        // keeping it in state (not the closure) means an `insert_source` failure
        // can't drop it out from under the GPU. The completion callback gets
        // `&mut PrismState` and finishes it by id.
        let id = next_capture_id();
        self.screencopy_inflight.push(ScreencopyInflight {
            id,
            imported,
            frame: frame.clone(),
            with_damage,
            region_wh: (rw as u32, rh as u32),
        });

        let Some(loop_handle) = self.loop_handle.clone() else {
            // No event loop (unreachable post-init): block on the fd, then
            // complete inline so we never free the import while the GPU writes.
            block_on_sync_fd(&sync_fd);
            self.complete_screencopy_dmabuf(id);
            return;
        };

        let source = Generic::new(sync_fd, Interest::READ, Mode::OneShot);
        let res = loop_handle.insert_source(source, move |_, _, state: &mut PrismState| {
            state.complete_screencopy_dmabuf(id);
            Ok(PostAction::Remove)
        });
        if res.is_err() {
            // Practically unreachable (epoll registration failure). Pull the
            // entry back and leak its import rather than free it while the GPU
            // may still be writing — a one-off leak is safer than a UAF.
            trace!("screencopy: failed to register dmabuf completion source");
            if let Some(pos) = self.screencopy_inflight.iter().position(|c| c.id == id) {
                let ScreencopyInflight {
                    imported, frame, ..
                } = self.screencopy_inflight.remove(pos);
                std::mem::forget(imported);
                frame.failed();
            }
        }
    }

    /// Finish an in-flight dmabuf screencopy whose GPU sync_fd has signalled:
    /// send `ready` and drop the import (now safe — the GPU is done). Called
    /// from the calloop completion source. No-op if the entry is gone.
    pub fn complete_screencopy_dmabuf(&mut self, id: u64) {
        let Some(pos) = self.screencopy_inflight.iter().position(|c| c.id == id) else {
            return;
        };
        let c = self.screencopy_inflight.remove(pos);
        c.frame.flags(Flags::empty());
        if c.with_damage && c.frame.version() >= 3 {
            c.frame.damage(0, 0, c.region_wh.0, c.region_wh.1);
        }
        // Timestamp the completion *now* (the GPU just finished), not at request
        // time — this is the frame's actual presentation/ready instant.
        let (sec_hi, sec_lo, nsec) = monotonic_timestamp();
        c.frame.ready(sec_hi, sec_lo, nsec);
        // c.imported drops here — the GPU finished (its sync_fd signalled).
    }
}

/// An in-flight dmabuf screencopy: the GPU is rendering into `imported`; once
/// its sync_fd fires we send `ready` and drop the import. Held on
/// [`PrismState`] so an `insert_source` failure can't free the import while the
/// GPU still writes it.
pub struct ScreencopyInflight {
    id: u64,
    imported: ImportedImage,
    frame: ZwlrScreencopyFrameV1,
    with_damage: bool,
    region_wh: (u32, u32),
}

/// Monotonic id for matching a completion callback to its in-flight entry.
fn next_capture_id() -> u64 {
    use std::sync::atomic::AtomicU64;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Block the calling thread until a sync_file `fd` signals (GPU work complete).
/// Fallback for the (practically unreachable) no-event-loop path.
fn block_on_sync_fd(fd: &OwnedFd) {
    use std::os::fd::AsRawFd;
    let mut pfd = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: single valid pollfd; infinite timeout. Loops on EINTR.
    loop {
        let rc = unsafe { libc::poll(&mut pfd, 1, -1) };
        if rc >= 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
            break;
        }
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
