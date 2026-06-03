use ash::vk;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("vulkan loader: {0}")]
    Loading(#[from] ash::LoadingError),

    #[error("vulkan call failed ({context}): {result:?}")]
    Vk {
        context: &'static str,
        result: vk::Result,
    },

    #[error("no suitable Vulkan physical device found")]
    NoSuitableDevice,

    #[error("required Vulkan extension not supported on selected device: {0}")]
    MissingExtension(String),

    #[error("required Vulkan feature not supported on selected device: {0}")]
    MissingFeature(&'static str),

    #[error("{0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, RendererError>;

/// Extension trait to attach call-site context to a raw `vk::Result`.
pub(crate) trait VkResultExt<T> {
    fn vk_ctx(self, context: &'static str) -> Result<T>;
}

impl<T> VkResultExt<T> for std::result::Result<T, vk::Result> {
    fn vk_ctx(self, context: &'static str) -> Result<T> {
        self.map_err(|result| RendererError::Vk { context, result })
    }
}
