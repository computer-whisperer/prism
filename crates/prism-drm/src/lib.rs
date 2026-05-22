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
pub mod enumerate;
pub mod frame_clock;
pub mod gbm_dev;
pub mod output_ctx;
pub mod scanout;
pub mod session;

pub use card::{DrmCardContext, OutputConfig};
pub use cursor_plane::CursorPlane;
pub use frame_clock::FrameClock;
pub use enumerate::{ConnectorSummary, DeviceSummary, DrmFd, open_for_enumeration, summarize};
pub use gbm_dev::GbmDevice;
pub use output_ctx::OutputContext;
pub use scanout::{
    OutputPick, ScanoutDepth, add_framebuffer_for_bo, find_property, pick_all_connected,
    pick_by_name, pick_first_connected, set_connector_max_bpc,
};
pub use session::SeatSession;
