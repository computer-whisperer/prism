//! KMS frontend.
//!
//! Owns per-GPU `DrmDevice`, per-output `DrmSurface`, scanout buffer pool.
//! Builds atomic commits combining renderer scanout output + color/HDR
//! connector properties.

pub mod enumerate;
pub mod gbm_dev;
pub mod output_ctx;
pub mod scanout;
pub mod session;

pub use enumerate::{ConnectorSummary, DeviceSummary, DrmFd, open_for_enumeration, summarize};
pub use gbm_dev::GbmDevice;
pub use output_ctx::{OutputContext, OutputNotifiers, OutputSetup};
pub use scanout::{
    OutputPick, ScanoutDepth, add_framebuffer_for_bo, find_property, pick_by_name,
    pick_first_connected, set_connector_max_bpc,
};
pub use session::SeatSession;
