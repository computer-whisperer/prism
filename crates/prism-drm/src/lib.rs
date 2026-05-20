//! KMS frontend.
//!
//! Owns per-GPU `DrmDevice`, per-output `DrmSurface`, scanout buffer pool.
//! Builds atomic commits combining renderer scanout output + color/HDR
//! connector properties.
