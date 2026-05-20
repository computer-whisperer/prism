use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("prism=info")),
        )
        .init();

    tracing::info!("prism compositor — pre-tracer skeleton");

    Ok(())
}
