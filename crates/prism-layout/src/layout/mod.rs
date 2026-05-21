//! Window layout — scrollable tiling, workspaces, monitors.
//!
//! Bulk of the niri port lands in this module. Today only the small
//! support pieces are in (focus_ring proper; shadow / tab_indicator /
//! insert_hint / opening_window / closing_window stubbed) plus `tile.rs`
//! (state machine ported, render emission stubbed) and the scaffolding
//! types `tile.rs` depends on (`Options`, `HitType`, the constant
//! `RESIZE_ANIMATION_THRESHOLD`, etc.). `workspace/monitor/floating/
//! scrolling/mod.rs` proper come in later port chunks.

use std::rc::Rc;
use std::time::Duration;

use prism_animation::Clock;
use prism_config::utils::MergeWith as _;
use prism_config::Config;
use smithay::output::Output;
use smithay::utils::{Logical, Point};

pub mod closing_window;
pub mod element;
pub mod focus_ring;
pub mod insert_hint_element;
pub mod opening_window;
pub mod shadow;
pub mod tab_indicator;
pub mod tile;

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
pub use tile::Tile;

use crate::utils::round_logical_in_physical_max1;

/// Size changes up to this many pixels don't animate.
pub const RESIZE_ANIMATION_THRESHOLD: f64 = 10.;

/// Pointer needs to move this far to pull a window from the layout.
pub const INTERACTIVE_MOVE_START_THRESHOLD: f64 = 256. * 256.;

/// Opacity of interactively moved tiles targeting the scrolling layout.
pub const INTERACTIVE_MOVE_ALPHA: f64 = 0.75;

/// Amount of touchpad movement to toggle the overview.
pub const OVERVIEW_GESTURE_MOVEMENT: f64 = 300.;

/// Snapshot of a `LayoutElement`'s render output, used for resize /
/// close animations.
///
/// Niri stores the baked GLES texture(s) of the pre-resize render here
/// (plus `size`) and crossfades from the snapshot to the live render.
/// Prism's render path doesn't have the offscreen-FBO infrastructure
/// yet, so only the `size` is retained — that's all the surrounding
/// state machine actually reads. When prism wires up offscreen renders,
/// this gains the baked-texture fields and the resize crossfade lights
/// up; until then the visual resize just snaps the tile to its
/// animated geometry, with no texture interpolation.
#[derive(Debug)]
pub struct LayoutElementRenderSnapshot {
    /// Visual size of the element at the moment the snapshot was taken.
    pub size: smithay::utils::Size<f64, smithay::utils::Logical>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Options {
    pub layout: prism_config::Layout,
    pub animations: prism_config::Animations,
    pub gestures: prism_config::Gestures,
    pub overview: prism_config::Overview,
    pub blur: prism_config::Blur,
    // Debug flags.
    pub disable_resize_throttling: bool,
    pub disable_transactions: bool,
    pub deactivate_unfocused_windows: bool,
}

impl Options {
    pub fn from_config(config: &Config) -> Self {
        Self {
            layout: config.layout.clone(),
            animations: config.animations.clone(),
            gestures: config.gestures,
            overview: config.overview,
            blur: config.blur,
            disable_resize_throttling: config.debug.disable_resize_throttling,
            disable_transactions: config.debug.disable_transactions,
            deactivate_unfocused_windows: config.debug.deactivate_unfocused_windows,
        }
    }

    pub fn with_merged_layout(mut self, part: Option<&prism_config::LayoutPart>) -> Self {
        if let Some(part) = part {
            self.layout.merge_with(part);
        }
        self
    }

    pub fn adjusted_for_scale(mut self, scale: f64) -> Self {
        self.layout.gaps = round_logical_in_physical_max1(scale, self.layout.gaps);
        self
    }
}

/// Type of the window hit from `window_under()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HitType {
    /// The hit is within a window's input region and can be used for sending events to it.
    Input {
        /// Position of the window's buffer.
        win_pos: Point<f64, Logical>,
    },
    /// The hit can activate a window, but it is not in the input region so cannot send events.
    ///
    /// For example, this could be clicking on a tile border outside the window.
    Activate {
        /// Whether the hit was on the tab indicator.
        is_tab_indicator: bool,
    },
}

/// Whether to activate a newly added window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ActivateWindow {
    /// Activate unconditionally.
    Yes,
    /// Activate based on heuristics.
    #[default]
    Smart,
    /// Do not activate.
    No,
}

/// Suppress dead-code warnings while large pieces of the layout port
/// (`Layout`, `Workspace`, `Monitor`) are still being filled in. Lets
/// the scaffolding compile clean before the rest lands.
#[allow(dead_code)]
const _: () = ();

/// `Layout`-mod private helpers re-exported up so the not-yet-ported
/// `Layout<W>` impl can be inserted alongside in a later port chunk
/// without a churn pass.
pub(crate) fn _suppress_unused_imports(
    _: Rc<Options>,
    _: Clock,
    _: Duration,
    _: &Output,
    _: Point<f64, Logical>,
) {
}
