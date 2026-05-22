//! Per-output hardware cursor plane.
//!
//! Lives on `OutputContext` as `cursor: Option<CursorPlane>`. `None`
//! means the driver didn't expose a usable cursor plane, or claiming
//! it failed — outputs in that state render the rest of the scene as
//! normal and just don't show a cursor (software fallback is a
//! later phase).
//!
//! What this owns per output:
//!   - The claimed `plane::Handle` (held for the output's lifetime so
//!     no other CRTC steals it)
//!   - One ARGB8888 LINEAR GBM BO sized to the driver's `cursor_size`
//!     (or 64×64 fallback)
//!   - The DRM framebuffer for that BO
//!   - Current position in CRTC pixels, hotspot offset, visibility
//!
//! The hot path is `set_position` — pure field write, no syscalls.
//! Position only takes effect on the next page-flip via
//! [`CursorPlane::to_plane_state`]. Latency: up to one refresh
//! interval. Sub-frame cursor updates (legacy `drmModeMoveCursor` or a
//! cursor-only atomic commit) are a later phase.
//!
//! Sprite uploads happen lazily on cursor-icon changes via
//! [`CursorPlane::upload_sprite`] — copies RGBA8888 → ARGB8888 into
//! the BO with `gbm_bo_write`.

use anyhow::{Context, Result};
use gbm::BufferObject;
use smithay::backend::drm::{DrmSurface, PlaneClaim, PlaneState, PlaneConfig};
use smithay::reexports::drm::control::{framebuffer, plane};
use smithay::utils::{Rectangle, Transform};

use crate::scanout::add_framebuffer_for_bo;
use crate::{DrmCardContext, GbmDevice};

/// Per-output hardware cursor.
pub struct CursorPlane {
    /// Plane handle held via a `PlaneClaim` for the OutputContext's
    /// lifetime. Drop releases the claim so a re-add can reclaim it.
    _claim: PlaneClaim,
    handle: plane::Handle,
    /// CPU-writable ARGB8888 LINEAR scanout BO. Size matches
    /// `cursor_size` so any sprite ≤ that size fits — sub-size sprites
    /// are zero-padded around the hotspot during upload.
    _bo: BufferObject<()>,
    fb: framebuffer::Handle,
    /// BO size in pixels. The cursor plane shows the full BO; smaller
    /// sprites get padded into it.
    size: (u32, u32),
    /// Position in CRTC pixels, sprite top-left (NOT hotspot).
    pos: (i32, i32),
    /// Visibility flag — `false` ⇒ `to_plane_state` emits a "disable
    /// plane" config so the kernel skips this plane on commit.
    visible: bool,
}

impl CursorPlane {
    /// Attempt to claim a cursor plane on the given surface and back
    /// it with a fresh ARGB8888 BO sized to the driver's
    /// `cursor_size`. Returns `None` (with a log) if the driver
    /// doesn't expose a cursor plane or the claim fails — caller
    /// proceeds without hardware cursor on this output.
    pub fn try_new(
        card: &DrmCardContext,
        gbm: &GbmDevice,
        surface: &DrmSurface,
    ) -> Result<Option<Self>> {
        let cursor_planes = &surface.planes().cursor;
        let Some(plane_info) = cursor_planes.first() else {
            tracing::warn!("no cursor plane exposed; cursor will be invisible on this output");
            return Ok(None);
        };

        let Some(claim) = surface.claim_plane(plane_info.handle) else {
            tracing::warn!(
                plane = ?plane_info.handle,
                "cursor plane already claimed by another surface; cursor invisible on this output"
            );
            return Ok(None);
        };

        let size_buf = card.drm.cursor_size();
        let (cw, ch) = (size_buf.w, size_buf.h);

        let bo = gbm
            .allocate_cursor(cw, ch)
            .with_context(|| format!("allocate cursor BO {cw}x{ch}"))?;
        let fb = add_framebuffer_for_bo(&card.drm, &bo)
            .context("add_framebuffer_for_bo (cursor)")?;

        tracing::info!(
            "cursor plane claimed: handle={:?} bo={cw}x{ch}",
            plane_info.handle
        );

        Ok(Some(Self {
            _claim: claim,
            handle: plane_info.handle,
            _bo: bo,
            fb,
            size: (cw, ch),
            pos: (0, 0),
            visible: false,
        }))
    }

    /// BO size in pixels. The cursor sprite must fit within this; the
    /// driver scans out exactly this rectangle from the plane.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Copy an ARGB8888 sprite into the cursor BO. The sprite is
    /// expected to be ≤ the BO size; smaller sprites are zero-padded
    /// to fill (the BO is cleared first).
    ///
    /// `pixels` is in RGBA8888 byte order (the format xcursor returns)
    /// — we swizzle to ARGB8888 during the copy.
    pub fn upload_sprite(
        &mut self,
        pixels_rgba: &[u8],
        src_w: u32,
        src_h: u32,
    ) -> Result<()> {
        let (cw, ch) = self.size;
        anyhow::ensure!(
            src_w <= cw && src_h <= ch,
            "cursor sprite {src_w}x{src_h} larger than BO {cw}x{ch}"
        );
        anyhow::ensure!(
            pixels_rgba.len() == (src_w * src_h * 4) as usize,
            "cursor sprite length {} != {}*{}*4",
            pixels_rgba.len(),
            src_w,
            src_h
        );

        // Build a (cw × ch) ARGB8888 buffer, zero-filled, then blit
        // the sprite into the top-left. `gbm_bo_write` writes exactly
        // (cw * ch * 4) bytes into the BO regardless of stride; with
        // LINEAR + ARGB8888 the row pitch == cw * 4 so that's correct.
        let mut out = vec![0u8; (cw * ch * 4) as usize];
        for y in 0..src_h {
            let src_row =
                &pixels_rgba[(y * src_w * 4) as usize..((y + 1) * src_w * 4) as usize];
            let dst_off = (y * cw * 4) as usize;
            for x in 0..src_w {
                let i = (x * 4) as usize;
                // xcursor: RGBA (R, G, B, A); ARGB8888 little-endian
                // wire layout is (B, G, R, A) in memory — same as
                // BGRA8888. Convert RGBA→BGRA per pixel.
                let r = src_row[i];
                let g = src_row[i + 1];
                let b = src_row[i + 2];
                let a = src_row[i + 3];
                out[dst_off + i] = b;
                out[dst_off + i + 1] = g;
                out[dst_off + i + 2] = r;
                out[dst_off + i + 3] = a;
            }
        }

        self._bo
            .write(&out)
            .context("gbm_bo_write (cursor sprite)")?;
        Ok(())
    }

    /// Update the on-screen position. Cheap — the kernel only sees
    /// it on the next [`to_plane_state`] / page-flip.
    pub fn set_position(&mut self, x: i32, y: i32) {
        self.pos = (x, y);
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn position(&self) -> (i32, i32) {
        self.pos
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    /// Build the `PlaneState` to merge into the next atomic
    /// page-flip. When `visible == false` returns a "disable this
    /// plane" state (the kernel skips it).
    pub fn to_plane_state(&self) -> PlaneState<'static> {
        if !self.visible {
            return PlaneState {
                handle: self.handle,
                config: None,
            };
        }
        let (cw, ch) = self.size;
        let (x, y) = self.pos;
        let src = Rectangle::from_size((cw as i32, ch as i32).into()).to_f64();
        let dst = Rectangle::new(
            (x, y).into(),
            (cw as i32, ch as i32).into(),
        );
        PlaneState {
            handle: self.handle,
            config: Some(PlaneConfig {
                src,
                dst,
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: self.fb,
                fence: None,
            }),
        }
    }
}
