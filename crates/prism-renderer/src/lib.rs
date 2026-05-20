//! Vulkan (ash) renderer.
//!
//! One renderer instance per GPU. Consumes `prism_frame::FrameDescription`,
//! produces scanout buffers for `prism_drm` to submit.
