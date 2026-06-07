//! KMS frontend.
//!
//! Three-layer ownership:
//!
//!   - [`SeatSession`] — one per process. libseat grant.
//!   - [`DrmCardContext`] — one per `/dev/dri/cardN` driven. DrmDevice + GBM.
//!   - [`OutputContext`] — one per active connector. DrmSurface + scanout
//!     BOs + per-output Renderer.
//!
//! Static per-output configuration (depth, formats, encode-shader chain,
//! and — future — color description, calibration, tone-map) lives in
//! [`OutputConfig`].

pub mod breadcrumb;
pub mod card;
pub mod cursor_plane;
pub mod edid;
pub mod enumerate;
pub mod frame_clock;
pub mod gbm_dev;
pub mod hdr;
pub mod modifiers;
pub mod output_ctx;
pub mod scanout;
pub mod session;

pub use breadcrumb::{breadcrumb as drm_breadcrumb, flip_trace};
pub use card::{DrmCardContext, OutputConfig};
pub use cursor_plane::CursorPlane;
pub use edid::{ColorPrimaries, EdidInfo, HdrCapabilities};
pub use enumerate::{open_for_enumeration, summarize, ConnectorSummary, DeviceSummary, DrmFd};
pub use frame_clock::FrameClock;
pub use gbm_dev::GbmDevice;
pub use hdr::{HdrEotf, HdrProps, HdrSignaling};
pub use modifiers::pick_scanout_modifiers;
pub use output_ctx::{ColorOverride, LutSource, OutputContext, PresentOutcome};
pub use scanout::{
    add_framebuffer_for_bo, connector_is_non_desktop, find_property, pick_all_connected,
    pick_all_connected_with_config, pick_by_name, pick_by_name_with_config, pick_first_connected,
    rescan_non_desktop, set_connector_max_bpc, ConnectorScan, NonDesktopConnector, OutputPick,
    ScanoutDepth,
};
pub use session::SeatSession;
