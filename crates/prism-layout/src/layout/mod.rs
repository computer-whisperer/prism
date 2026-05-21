//! Window layout — scrollable tiling, workspaces, monitors.
//!
//! Bulk of the niri port lands in this module. Today only the small
//! support pieces are in (focus_ring proper; shadow / tab_indicator /
//! insert_hint / opening_window / closing_window stubbed); the main
//! layout/tile/scrolling/workspace/monitor/floating code comes in the
//! next port chunk.

pub mod closing_window;
pub mod element;
pub mod focus_ring;
pub mod insert_hint_element;
pub mod opening_window;
pub mod shadow;
pub mod tab_indicator;

pub use closing_window::ClosingWindow;
pub use element::{
    ConfigureIntent, InteractiveResizeData, LayoutElement, RenderCtx, SizeFrac, SizingMode,
    ViewRect,
};
pub use focus_ring::FocusRing;
pub use insert_hint_element::InsertHintElement;
pub use opening_window::OpenAnimation;
pub use shadow::Shadow;
pub use tab_indicator::TabIndicator;
