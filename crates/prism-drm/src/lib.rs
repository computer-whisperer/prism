//! KMS frontend.
//!
//! Owns per-GPU `DrmDevice`, per-output `DrmSurface`, scanout buffer pool.
//! Builds atomic commits combining renderer scanout output + color/HDR
//! connector properties.

pub mod enumerate;
pub mod gbm_dev;
pub mod scanout;
pub mod session;

pub use enumerate::{ConnectorSummary, DeviceSummary, DrmFd, open_for_enumeration, summarize};
pub use gbm_dev::GbmDevice;
pub use scanout::{OutputPick, add_framebuffer_for_bo, pick_by_name, pick_first_connected};
pub use session::SeatSession;
