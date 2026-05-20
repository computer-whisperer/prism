//! `PrismState` — the smithay handler-trait carrier.
//!
//! Smithay's protocol dispatch model: one application-owned struct
//! (`PrismState` here) implements every protocol's `*Handler` trait that the
//! compositor wants to participate in, and `delegate_*!` macros wire the
//! protocol message dispatch into those traits.
//!
//! Scope of this scaffolding (task #46):
//!   - wl_compositor (surface lifecycle, basic commits)
//!   - xdg-shell (toplevel windows, configure / map / unmap)
//!   - wl_shm (software-rendered clients)
//!
//! Not yet wired (will come incrementally):
//!   - linux-dmabuf (hardware-rendered clients)
//!   - wl_seat / input
//!   - wl_output (display advertisement)
//!   - presentation-time, viewporter, fractional-scale, …
//!
//! On commit we currently only log; rendering hooks in once #47 (texture
//! import) and #48 (shader pipeline) are wired up.

use std::sync::Arc;

use anyhow::{Context, Result};
use prism_renderer::vk;
use smithay::backend::allocator::Format as DrmFormat;
use smithay::backend::allocator::dmabuf::Dmabuf as SmithayDmabuf;
use smithay::delegate_compositor;
use smithay::delegate_dmabuf;
use smithay::delegate_shm;
use smithay::delegate_xdg_shell;
use prism_frame::{DrmFourcc, DrmModifier};
use smithay::reexports::wayland_server::Client;
use smithay::reexports::wayland_server::backend::ClientData;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Display, DisplayHandle, Resource};
use smithay::utils::Serial;
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    CompositorClientState, CompositorHandler, CompositorState, get_role, with_states,
};
use smithay::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState};

use crate::client::PrismClient;

pub struct PrismState {
    pub display_handle: DisplayHandle,
    pub compositor: CompositorState,
    pub xdg_shell: XdgShellState,
    pub shm: ShmState,
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: DmabufGlobal,
    pub renderer_device: Arc<prism_renderer::Device>,
}

impl PrismState {
    pub fn new(
        display: &Display<PrismState>,
        renderer_device: Arc<prism_renderer::Device>,
    ) -> Self {
        let dh = display.handle();
        let compositor = CompositorState::new::<PrismState>(&dh);
        let xdg_shell = XdgShellState::new::<PrismState>(&dh);
        // Empty extra-formats list: ARGB8888 and XRGB8888 are mandatory and
        // smithay advertises them implicitly.
        let shm = ShmState::new::<PrismState>(&dh, []);

        // Hardcoded minimal dmabuf format set for now: XRGB8888 / ARGB8888
        // with LINEAR modifier. Both map to vk::Format::B8G8R8A8_UNORM. Tiled
        // modifiers will be added once we query the Vulkan device for
        // VK_EXT_image_drm_format_modifier support.
        let supported_formats = [
            DrmFormat {
                code: DrmFourcc::Xrgb8888,
                modifier: DrmModifier::Linear,
            },
            DrmFormat {
                code: DrmFourcc::Argb8888,
                modifier: DrmModifier::Linear,
            },
        ];
        let mut dmabuf_state = DmabufState::new();
        let dmabuf_global =
            dmabuf_state.create_global::<PrismState>(&dh, supported_formats.iter().copied());

        Self {
            display_handle: dh,
            compositor,
            xdg_shell,
            shm,
            dmabuf_state,
            dmabuf_global,
            renderer_device,
        }
    }
}

// ─── wl_compositor ──────────────────────────────────────────────────────────

impl CompositorHandler for PrismState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client
            .get_data::<PrismClient>()
            .expect("client missing PrismClient")
            .compositor
    }

    fn commit(&mut self, surface: &WlSurface) {
        let role = get_role(surface);
        tracing::debug!(?role, "wl_surface commit");

        // For xdg-shell toplevels, send an initial configure on first commit so
        // the client knows it can start drawing. Skipped here once already
        // configured.
        if let Some("xdg_toplevel") = role {
            let needs_initial_configure = with_states(surface, |states| {
                states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .map(|d| {
                        let attrs = d.lock().unwrap();
                        !attrs.initial_configure_sent
                    })
                    .unwrap_or(false)
            });
            if needs_initial_configure {
                if let Some(toplevel) = self
                    .xdg_shell
                    .toplevel_surfaces()
                    .iter()
                    .find(|t| t.wl_surface() == surface)
                    .cloned()
                {
                    toplevel.send_configure();
                    tracing::info!("sent initial configure to xdg_toplevel");
                }
            }
        }
    }
}

impl BufferHandler for PrismState {
    fn buffer_destroyed(&mut self, _buffer: &smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer) {}
}

delegate_compositor!(PrismState);

// ─── wl_shm ─────────────────────────────────────────────────────────────────

impl ShmHandler for PrismState {
    fn shm_state(&self) -> &ShmState {
        &self.shm
    }
}

delegate_shm!(PrismState);

// ─── xdg-shell ──────────────────────────────────────────────────────────────

impl XdgShellHandler for PrismState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        tracing::info!(
            surface_id = ?surface.wl_surface().id(),
            "new xdg_toplevel"
        );
        // Initial configure is sent on first commit via the CompositorHandler
        // hook above (so the client has a chance to set title / app_id first).
    }

    fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {
        tracing::info!("new xdg_popup (not yet handled)");
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: WlSeat, _serial: Serial) {
        // No popup grab handling yet — no input plumbing.
    }

    fn reposition_request(
        &mut self,
        _surface: PopupSurface,
        _positioner: PositionerState,
        _token: u32,
    ) {
    }
}

delegate_xdg_shell!(PrismState);

// ─── linux-dmabuf ───────────────────────────────────────────────────────────

impl DmabufHandler for PrismState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: SmithayDmabuf,
        notifier: ImportNotifier,
    ) {
        // Try a real Vulkan import as the validation. If it works we accept
        // the buffer; otherwise we tell the client this dmabuf can't be used.
        match try_import_dmabuf(&self.renderer_device, &dmabuf) {
            Ok(()) => {
                tracing::info!(
                    width = ?smithay::backend::allocator::Buffer::size(&dmabuf).w,
                    height = ?smithay::backend::allocator::Buffer::size(&dmabuf).h,
                    format = ?smithay::backend::allocator::Buffer::format(&dmabuf),
                    "imported client dmabuf as VkImage",
                );
                let _ = notifier.successful::<PrismState>();
            }
            Err(e) => {
                tracing::warn!("dmabuf import rejected: {e:#}");
                notifier.failed();
            }
        }
    }
}

delegate_dmabuf!(PrismState);

/// Test-import a client-provided dmabuf as a sampled `VkImage`. Throws away
/// the image afterward — this is only validation. Surface-tracked textures
/// will live in a per-surface cache once #48 wires the shader pipeline.
fn try_import_dmabuf(
    device: &Arc<prism_renderer::Device>,
    src: &SmithayDmabuf,
) -> Result<()> {
    let dmabuf = prism_frame::Dmabuf::from_smithay(src).context("Dmabuf::from_smithay")?;
    let vk_format = vk_format_for(dmabuf.format).with_context(|| {
        format!("no Vulkan format mapping for {:?}", dmabuf.format)
    })?;
    let _image = prism_renderer::ImportedImage::import(
        device.clone(),
        &dmabuf,
        vk_format,
        vk::ImageUsageFlags::SAMPLED,
    )
    .context("ImportedImage::import (SAMPLED)")?;
    Ok(())
}

/// Map a DRM fourcc to the Vulkan format we'd sample it as. Single-planar
/// 32-bit packed formats only for now.
fn vk_format_for(fourcc: DrmFourcc) -> Option<vk::Format> {
    Some(match fourcc {
        // DRM is little-endian-byte-order, so XRGB8888 in memory is B,G,R,X.
        // Vulkan's B8G8R8A8 reads exactly that byte order.
        DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
        _ => return None,
    })
}

// ─── Per-client data helper ─────────────────────────────────────────────────

/// Build the per-client data smithay attaches to each new client.
pub fn new_client_data() -> Arc<dyn ClientData> {
    Arc::new(PrismClient::default())
}

/// Convenience: create a fresh `Display<PrismState>`. Wrapped so callers
/// don't need a direct `wayland_server` dependency.
pub fn new_display() -> Result<Display<PrismState>> {
    Display::<PrismState>::new().context("wayland_server::Display::new")
}
