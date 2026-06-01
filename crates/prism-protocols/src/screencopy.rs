//! `zwlr_screencopy_v1` — output screenshot / capture for `grim`, `wf-recorder`,
//! `xdg-desktop-portal-wlr`, etc.
//!
//! Hand-rolled (smithay ships no screencopy). See `docs/screen-capture.md`.
//! Both buffer paths are **queued and serviced from the render loop**
//! (`submit_pending_screencopy`, right after the output's `present()`), so the
//! capture's GPU submit is correctly ordered after the frame, and **neither
//! stalls the main thread** — the key to non-laggy recording. An immediate
//! `copy` forces a frame even on an idle output; `copy_with_damage` rides the
//! next damage-driven frame (throttled to actual changes). Both render the sRGB
//! capture encode of the output's persistent intermediate (`Renderer`):
//!
//! - **dmabuf (zero-copy)** — `copy` with a `linux_dmabuf` buffer: import it as a
//!   `COLOR_ATTACHMENT` and render straight into it. Whole-output only. The path
//!   recording clients prefer.
//! - **SHM** — `copy` with a `wl_shm` buffer: render into an offscreen + an owned
//!   host readback (async), and on completion memcpy into the client buffer.
//!   Whole-output and region (region cropped on the copy).
//!
//! Either way, `ready` fires from a calloop sync_fd source once the GPU finishes
//! (the SHM completion memcpys first). We advertise `Xrgb8888` for both and
//! capture into `B8G8R8A8_UNORM` (memory `B,G,R,A`) so the bytes match with no
//! swizzle; dmabuf is advertised only for whole-output captures.
//!
//! Style mirrors [`crate::output_power`]: `GlobalDispatch` / `Dispatch` are
//! implemented directly on [`PrismState`], so there's no handler trait and no
//! `delegate_*` macro — the global is registered with
//! [`create_screencopy_global`] and the per-frame state lives in the frame
//! object's user data.

use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use prism_renderer::{vk, HostReadback, ImportedImage};
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
use smithay::wayland::shm::{with_buffer_contents, with_buffer_contents_mut};
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

    fn destroyed(
        state: &mut Self,
        _client: smithay::reexports::wayland_server::backend::ClientId,
        frame: &ZwlrScreencopyFrameV1,
        _data: &ScreencopyFrameData,
    ) {
        // Drop any *queued* (not-yet-submitted) dmabuf capture for this frame so
        // an abandoned `copy_with_damage` (e.g. recording stopped while waiting
        // for damage) doesn't leak its imported buffer. In-flight captures are
        // left alone — their GPU work is outstanding; they self-clean when the
        // sync_fd fires (`ready` on a dead frame is a harmless no-op).
        state.screencopy_pending.retain(|p| p.frame != *frame);
    }
}

impl PrismState {
    /// Accept a screencopy `copy` into a `wl_shm` buffer: validate it and
    /// **queue** the capture (like the dmabuf path). The render loop captures it
    /// asynchronously right after the output's frame and the completion memcpys
    /// the result in — no main-thread `queue_wait_idle` stall (that stall, hit
    /// per frame, is what made screen recording lag). `region` is `(x, y, w, h)`
    /// in physical px; a region capture is a sub-rect cropped on the copy.
    fn service_screencopy_shm(
        &mut self,
        frame: &ZwlrScreencopyFrameV1,
        output_id: &OutputId,
        region: (i32, i32, i32, i32),
        buffer: &WlBuffer,
        with_damage: bool,
    ) {
        let (_rx, _ry, rw, rh) = region;
        // Validate the SHM buffer up front (read-only) so a bad one fails now,
        // before we queue anything.
        let valid = with_buffer_contents(buffer, |_ptr, len, bdata| {
            bdata.format == wl_shm::Format::Xrgb8888
                && bdata.width == rw
                && bdata.height == rh
                && bdata.stride >= rw * 4
                && len >= (bdata.offset as usize) + (bdata.stride as usize) * (rh as usize)
        })
        .unwrap_or(false);
        if !valid {
            frame.post_error(
                zwlr_screencopy_frame_v1::Error::InvalidBuffer,
                "buffer does not match the advertised format/size",
            );
            return;
        }

        self.enqueue_screencopy(
            output_id,
            frame,
            with_damage,
            PendingKind::Shm {
                buffer: buffer.clone(),
                region,
            },
        );
    }

    /// Queue a capture and ensure the output renders so the render loop services
    /// it. An immediate `copy` forces a present even on an idle output;
    /// `copy_with_damage` rides the next frame damage produces (so a static
    /// screen yields no new frames, as intended).
    fn enqueue_screencopy(
        &mut self,
        output_id: &OutputId,
        frame: &ZwlrScreencopyFrameV1,
        with_damage: bool,
        kind: PendingKind,
    ) {
        self.screencopy_pending.push(PendingScreencopy {
            output_id: output_id.clone(),
            frame: frame.clone(),
            with_damage,
            kind,
        });
        if !with_damage {
            if let Some(ctx) = self.outputs.get_mut(output_id) {
                ctx.force_next_present();
            }
            self.output_redraw
                .entry(output_id.clone())
                .or_default()
                .queue_redraw();
        }
    }

    /// Accept a screencopy `copy` into a client **dmabuf**: validate + import it
    /// as a render target now, but **queue** it rather than rendering here. The
    /// render loop drains the queue right after the output's next `present()`
    /// ([`Self::submit_pending_screencopy`]), so the capture's GPU submit is
    /// sequenced after that frame's encode on the shared queue — the proper fix
    /// for the dmabuf ordering caveat. Whole-output only. `region` is
    /// `(x, y, w, h)` (for whole-output `x==y==0`).
    ///
    /// An immediate `copy` forces the output to render (even if idle);
    /// `copy_with_damage` waits for the next damage-driven frame.
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

        // Import the client dmabuf as a COLOR_ATTACHMENT render target (cheap;
        // same call the scanout uses). The actual render is deferred to the
        // render loop.
        let imported = {
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
            match ImportedImage::import(
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
            }
        };

        self.enqueue_screencopy(
            output_id,
            frame,
            with_damage,
            PendingKind::Dmabuf {
                imported,
                region_wh: (rw as u32, rh as u32),
            },
        );
    }

    /// Fail and drop any *queued* dmabuf captures for `output_id` — for when the
    /// output can't render them (e.g. it's powered off, so the render loop skips
    /// it). The imports were never GPU-submitted, so dropping them is safe; the
    /// client gets `failed` instead of hanging. In-flight captures are untouched
    /// (their GPU work self-cleans via the sync_fd).
    pub fn fail_pending_screencopy(&mut self, output_id: &str) {
        if self.screencopy_pending.is_empty() {
            return;
        }
        let (mine, rest): (Vec<_>, Vec<_>) = self
            .screencopy_pending
            .drain(..)
            .partition(|p| p.output_id == output_id);
        self.screencopy_pending = rest;
        for p in mine {
            p.frame.failed();
        }
    }

    /// Render and submit any queued screencopy captures (dmabuf or SHM) for
    /// `output_id`. Called from the render loop **immediately after** the
    /// output's frame is presented, so each capture's GPU submit is ordered after
    /// that frame's encode (and before the next frame's decode overwrites the
    /// intermediate) on the shared graphics queue. Each then completes
    /// asynchronously when its sync_fd signals — see [`Self::complete_screencopy`].
    pub fn submit_pending_screencopy(&mut self, output_id: &str) {
        if self.screencopy_pending.is_empty() {
            return;
        }
        // Take this output's queued captures; leave the rest queued.
        let (mine, rest): (Vec<_>, Vec<_>) = self
            .screencopy_pending
            .drain(..)
            .partition(|p| p.output_id == output_id);
        self.screencopy_pending = rest;

        for p in mine {
            let PendingScreencopy {
                frame,
                with_damage,
                kind,
                ..
            } = p;

            let (sync_fd, inflight_kind) = match self.submit_one_capture(output_id, kind) {
                Ok(v) => v,
                Err(()) => {
                    frame.failed();
                    continue;
                }
            };

            // Park the in-flight capture on PrismState (NOT the calloop closure):
            // the GPU is still writing the target (imported dmabuf / readback
            // buffer), so it must outlive the wait, and keeping it in state means
            // an `insert_source` failure can't drop it out from under the GPU.
            let id = next_capture_id();
            self.screencopy_inflight.push(ScreencopyInflight {
                id,
                frame,
                with_damage,
                kind: inflight_kind,
            });

            let Some(loop_handle) = self.loop_handle.clone() else {
                // No event loop (unreachable post-init): block on the fd, then
                // complete inline so we never free the target mid-write.
                block_on_sync_fd(&sync_fd);
                self.complete_screencopy(id);
                continue;
            };

            let source = Generic::new(sync_fd, Interest::READ, Mode::OneShot);
            let res = loop_handle.insert_source(source, move |_, _, state: &mut PrismState| {
                state.complete_screencopy(id);
                Ok(PostAction::Remove)
            });
            if res.is_err() {
                // Practically unreachable (epoll registration failure). Pull the
                // entry back; the GPU may still be writing its target, so leak the
                // GPU-referenced resource (forget the kind) rather than risk a UAF.
                trace!("screencopy: failed to register completion source");
                if let Some(pos) = self.screencopy_inflight.iter().position(|c| c.id == id) {
                    let ScreencopyInflight { frame, kind, .. } =
                        self.screencopy_inflight.remove(pos);
                    std::mem::forget(kind);
                    frame.failed();
                }
            }
        }
    }

    /// Submit one queued capture: render it (dmabuf → into the client buffer; SHM
    /// → into an offscreen + host readback) and return its sync_fd plus the
    /// in-flight kind to park. `Err(())` if the output is gone or the GPU submit
    /// failed (caller fails the frame).
    fn submit_one_capture(
        &mut self,
        output_id: &str,
        kind: PendingKind,
    ) -> Result<(OwnedFd, InflightKind), ()> {
        let Some(ctx) = self.outputs.get_mut(output_id) else {
            return Err(());
        };
        let white = ctx.effective_sdr_reference_nits();
        let src_width = ctx.extent.width;
        match kind {
            PendingKind::Dmabuf {
                imported,
                region_wh,
            } => {
                let fd = ctx
                    .renderer
                    .capture_into_dmabuf(&imported, white)
                    .map_err(|e| trace!("screencopy: capture_into_dmabuf failed: {e}"))?;
                Ok((
                    fd,
                    InflightKind::Dmabuf {
                        imported,
                        region_wh,
                    },
                ))
            }
            PendingKind::Shm { buffer, region } => {
                let (fd, readback) = ctx
                    .renderer
                    .capture_to_host(CAPTURE_VK_FORMAT, white)
                    .map_err(|e| trace!("screencopy: capture_to_host failed: {e}"))?;
                Ok((
                    fd,
                    InflightKind::Shm {
                        readback,
                        buffer,
                        region,
                        src_width,
                    },
                ))
            }
        }
    }

    /// Finish an in-flight screencopy whose GPU sync_fd has signalled: for SHM,
    /// memcpy the readback into the client buffer; then send `ready` and drop the
    /// capture's GPU resource (now safe — the GPU is done). Called from the
    /// calloop completion source. No-op if the entry is gone.
    pub fn complete_screencopy(&mut self, id: u64) {
        let Some(pos) = self.screencopy_inflight.iter().position(|c| c.id == id) else {
            return;
        };
        let c = self.screencopy_inflight.remove(pos);

        // Damage rect (whole captured region) for v3 with_damage clients.
        let (region_w, region_h) = match &c.kind {
            InflightKind::Dmabuf { region_wh, .. } => *region_wh,
            InflightKind::Shm { region, .. } => (region.2 as u32, region.3 as u32),
        };

        // SHM: copy the readback into the client buffer before signalling ready.
        if let InflightKind::Shm {
            readback,
            buffer,
            region,
            src_width,
        } = &c.kind
        {
            if copy_readback_to_shm(readback, *src_width, *region, buffer).is_err() {
                trace!("screencopy: shm readback copy failed");
                c.frame.failed();
                return;
            }
        }

        c.frame.flags(Flags::empty());
        if c.with_damage && c.frame.version() >= 3 {
            c.frame.damage(0, 0, region_w, region_h);
        }
        // Timestamp the completion *now* (the GPU just finished).
        let (sec_hi, sec_lo, nsec) = monotonic_timestamp();
        c.frame.ready(sec_hi, sec_lo, nsec);
        // c.kind drops here — GPU done (its sync_fd signalled).
    }
}

/// Copy a captured frame's `region` sub-rect out of a host readback (which holds
/// the whole output, `src_width` px wide, row-major Xrgb8888) into a client
/// `wl_shm` buffer. `Err(())` if the buffer no longer matches / is inaccessible.
fn copy_readback_to_shm(
    readback: &HostReadback,
    src_width: u32,
    region: (i32, i32, i32, i32),
    buffer: &WlBuffer,
) -> Result<(), ()> {
    let (rx, ry, rw, rh) = region;
    let src = readback.as_slice();
    let src_stride = (src_width * 4) as usize;
    let row_bytes = (rw * 4) as usize;
    let region_off = (ry as usize * src_stride) + rx as usize * 4;

    let r = with_buffer_contents_mut(buffer, |ptr, len, bdata| {
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
            if src_start + row_bytes > src.len() || dst_start + row_bytes > len {
                return Err(());
            }
            // SAFETY: bounds checked just above; src (readback) and dst (mapped
            // shm pool) are distinct allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_start),
                    ptr.add(dst_start),
                    row_bytes,
                );
            }
        }
        Ok(())
    });
    match r {
        Ok(Ok(())) => Ok(()),
        _ => Err(()),
    }
}

/// A queued screencopy awaiting its output's next frame. The render loop
/// captures it (via [`PrismState::submit_pending_screencopy`]) right after the
/// frame, so the capture is ordered correctly on the GPU queue.
pub struct PendingScreencopy {
    output_id: OutputId,
    frame: ZwlrScreencopyFrameV1,
    with_damage: bool,
    kind: PendingKind,
}

enum PendingKind {
    /// Client dmabuf, already imported as a `COLOR_ATTACHMENT` render target.
    /// `region_wh` is the whole-output size (dmabuf is whole-output only).
    Dmabuf {
        imported: ImportedImage,
        region_wh: (u32, u32),
    },
    /// Client `wl_shm` buffer + the `(x, y, w, h)` region to fill from the
    /// whole-output capture.
    Shm {
        buffer: WlBuffer,
        region: (i32, i32, i32, i32),
    },
}

/// An in-flight screencopy: the GPU is rendering into the capture target; once
/// its sync_fd fires we (for SHM) memcpy into the client buffer, send `ready`,
/// and drop the target. Held on [`PrismState`] so an `insert_source` failure
/// can't free the target while the GPU still writes it.
pub struct ScreencopyInflight {
    id: u64,
    frame: ZwlrScreencopyFrameV1,
    with_damage: bool,
    kind: InflightKind,
}

enum InflightKind {
    Dmabuf {
        /// Held only to keep the GPU render target alive until the sync_fd
        /// fires; dropped (freed) on completion. Not read by name (RAII).
        #[allow(dead_code)]
        imported: ImportedImage,
        region_wh: (u32, u32),
    },
    Shm {
        readback: HostReadback,
        buffer: WlBuffer,
        region: (i32, i32, i32, i32),
        /// Width of the captured frame (whole output) — the readback's row
        /// stride, for cropping `region` out of it.
        src_width: u32,
    },
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
