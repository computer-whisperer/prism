use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("prism=info,vulkan=info")),
        )
        .init();

    tracing::info!("prism compositor — pre-tracer skeleton");

    let instance = prism_renderer::Instance::new()?;

    // Smoke test: default-picked device.
    {
        let device = prism_renderer::Device::new(instance.clone(), None)?;
        tracing::info!(
            "default device: {}, graphics queue family {}",
            device.physical.name,
            device.physical.graphics_queue_family,
        );
    }

    // Smoke test: select by DRM node. Vega 20 is at render node 226:129 on this box
    // (the GPU driving DP-4 / LU28R55, our HDR tracer target).
    {
        let want = prism_renderer::DrmDevId {
            major: 226,
            minor: 129,
        };
        let device = prism_renderer::Device::new(instance, Some(want))?;
        tracing::info!(
            "drm-preferred device: {}, drm_render={:?}",
            device.physical.name,
            device.physical.drm_render,
        );
    }

    Ok(())
}
