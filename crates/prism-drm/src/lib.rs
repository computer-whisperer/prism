//! KMS frontend.
//!
//! Owns per-GPU `DrmDevice`, per-output `DrmSurface`, scanout buffer pool.
//! Builds atomic commits combining renderer scanout output + color/HDR
//! connector properties.

pub mod enumerate;
pub mod gbm_dev;

pub use enumerate::{ConnectorSummary, DeviceSummary, DrmFd, open_for_enumeration, summarize};
pub use gbm_dev::{GbmDevice, GbmFd};
