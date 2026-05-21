//! Window layout — scrollable tiling, workspaces, monitors.
//!
//! Bulk of the niri port lands in this module. Today only the small
//! support pieces are in (focus_ring proper; shadow / tab_indicator /
//! insert_hint / opening_window / closing_window stubbed) plus `tile.rs`
//! (state machine ported, render emission stubbed) and the scaffolding
//! types `tile.rs` depends on (`Options`, `HitType`, the constant
//! `RESIZE_ANIMATION_THRESHOLD`, etc.). `workspace/monitor/floating/
//! scrolling/mod.rs` proper come in later port chunks.

use prism_config::utils::MergeWith as _;
use prism_config::{Config, CornerRadius, OutputName};
use smithay::output::Output;
use smithay::utils::{Logical, Point, Size};

use crate::utils::id::IdCounter;

pub mod closing_window;
pub mod element;
pub mod floating;
pub mod focus_ring;
pub mod insert_hint_element;
pub mod monitor;
pub mod opening_window;
pub mod scrolling;
pub mod shadow;
pub mod tab_indicator;
pub mod tile;
pub mod workspace;

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

impl HitType {
    pub fn offset_win_pos(mut self, offset: Point<f64, Logical>) -> Self {
        match &mut self {
            HitType::Input { win_pos } => *win_pos += offset,
            HitType::Activate { .. } => (),
        }
        self
    }

    /// Hit-test a tile at `tile_pos` against `point` (both in workspace
    /// logical coords). Returns the window + the hit type whose
    /// `Input.win_pos` (if any) has been offset into the workspace
    /// frame.
    pub fn hit_tile<W: LayoutElement>(
        tile: &Tile<W>,
        tile_pos: Point<f64, Logical>,
        point: Point<f64, Logical>,
    ) -> Option<(&W, Self)> {
        let pos_within_tile = point - tile_pos;
        tile.hit(pos_within_tile)
            .map(|hit| (tile.window(), hit.offset_win_pos(tile_pos)))
    }

    pub fn to_activate(self) -> Self {
        match self {
            HitType::Input { .. } => HitType::Activate {
                is_tab_indicator: false,
            },
            HitType::Activate { .. } => self,
        }
    }
}

impl ActivateWindow {
    pub fn map_smart(self, f: impl FnOnce() -> bool) -> bool {
        match self {
            ActivateWindow::Yes => true,
            ActivateWindow::Smart => f(),
            ActivateWindow::No => false,
        }
    }
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

/// Stable per-workspace ID. Lives at the mod root because both
/// `scrolling::ScrollingSpace` and the not-yet-ported `Workspace` /
/// `Layout` use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkspaceId(u64);

static WORKSPACE_ID_COUNTER: IdCounter = IdCounter::new();

impl WorkspaceId {
    pub fn next() -> WorkspaceId {
        WorkspaceId(WORKSPACE_ID_COUNTER.next())
    }

    pub fn get(self) -> u64 {
        self.0
    }

    pub fn specific(id: u64) -> Self {
        Self(id)
    }
}

/// Stable per-output identifier — wraps the configured output name so
/// workspaces can remember which output they came from across disconnects.
#[derive(Debug, Clone)]
pub struct OutputId(pub String);

impl OutputId {
    pub fn new(output: &Output) -> Self {
        let output_name = output.user_data().get::<OutputName>().unwrap();
        Self(output_name.format_make_model_serial_or_connector())
    }

    pub fn matches(&self, output: &Output) -> bool {
        let output_name = output.user_data().get::<OutputName>().unwrap();
        output_name.matches(&self.0)
    }
}

/// Where a window goes when dropped on a workspace — used by
/// interactive-move / DnD-into-workspace. Niri keeps this `pub(super)`
/// inside `monitor.rs`; here it lives at the layout-mod root so
/// `scrolling.rs` can reference it before `monitor.rs` lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertPosition {
    NewColumn(usize),
    InColumn(usize, usize),
    Floating,
}

/// Which workspace a drop targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertWorkspace {
    Existing(WorkspaceId),
    NewAt(usize),
}

/// Insert-hint metadata propagated during an interactive move so the
/// receiving workspace can paint the drop preview.
#[derive(Debug)]
pub struct InsertHint {
    pub workspace: InsertWorkspace,
    pub position: InsertPosition,
    pub corner_radius: CornerRadius,
}

/// State carried through an in-progress interactive resize. Lives at
/// the layout-mod root because both `scrolling.rs` (which originates
/// the resize gesture handling) and the not-yet-ported `workspace.rs`
/// hold one. Generic over `W: LayoutElement` because the window-ID
/// type is `W::Id`.
#[derive(Debug)]
pub struct InteractiveResize<W: LayoutElement> {
    pub window: W::Id,
    pub original_window_size: Size<f64, Logical>,
    pub data: InteractiveResizeData,
}

/// A user-requested width/height that may refer to the tile-including-
/// border or to the bare window.
#[derive(Debug, Clone, Copy)]
pub enum ResolvedSize {
    /// Size of the tile including borders.
    Tile(f64),
    /// Size of the window excluding borders.
    Window(f64),
}

/// Overview progress — three-state union niri keeps inside its
/// `Layout<W>` for the overview-mode animation. Lifted to the
/// layout-mod root so `monitor.rs` can convert into its own
/// 2-variant local enum without depending on `Layout<W>` yet.
#[derive(Debug)]
pub enum OverviewProgress {
    Animation(prism_animation::Animation),
    Gesture(OverviewGesture),
    Open,
}

/// In-progress overview-toggle swipe.
#[derive(Debug)]
pub struct OverviewGesture {
    pub tracker: crate::swipe_tracker::SwipeTracker,
    /// Start point.
    pub start: f64,
    /// Current progress.
    pub value: f64,
}

/// Overview-zoom curve. Niri's tiny helper, kept here so both
/// `monitor.rs` and the not-yet-ported `Layout<W>` can reach it.
pub fn compute_overview_zoom(options: &Options, overview_progress: Option<f64>) -> f64 {
    let zoom = options.overview.zoom.clamp(0.0001, 0.75);

    if let Some(p) = overview_progress {
        (1. - p * (1. - zoom)).max(0.0001)
    } else {
        1.
    }
}

/// A tile that was lifted out of the layout (interactive move, swap, …).
/// Niri keeps the layout-shape state alongside the tile so it can be
/// re-inserted into a different workspace with the same column geometry.
#[derive(Debug)]
pub struct RemovedTile<W: LayoutElement> {
    pub tile: Tile<W>,
    /// Width of the column the tile was in. Stored as an opaque
    /// `f64` ratio for now — full `ColumnWidth` type lives in
    /// `scrolling.rs` and is reachable as `scrolling::ColumnWidth`
    /// once that lands.
    pub width: scrolling::ColumnWidth,
    pub is_full_width: bool,
    pub is_floating: bool,
}
