//! Types for communicating with niri via IPC.
//!
//! After connecting to the niri socket, you can send [`Request`]s. Niri will process them one by
//! one, in order, and to each request it will respond with a single [`Reply`], which is a `Result`
//! wrapping a [`Response`].
//!
//! If you send a [`Request::EventStream`], niri will *stop* reading subsequent [`Request`]s, and
//! will start continuously writing compositor [`Event`]s to the socket. If you'd like to read an
//! event stream and write more requests at the same time, you need to use two IPC sockets.
//!
//! <div class="warning">
//!
//! Requests are *always* processed separately. Time passes between requests, even when sending
//! multiple requests to the socket at once. For example, sending [`Request::Workspaces`] and
//! [`Request::Windows`] together may not return consistent results (e.g. a window may open on a
//! new workspace in-between the two responses). This goes for actions too: sending
//! [`Action::FocusWindow`] and <code>[Action::CloseWindow] { id: None }</code> together may close
//! the wrong window because a different window got focused in-between these requests.
//!
//! </div>
//!
//! You can use the [`socket::Socket`] helper if you're fine with blocking communication. However,
//! it is a fairly simple helper, so if you need async, or if you're using a different language,
//! you are encouraged to communicate with the socket manually.
//!
//! 1. Read the socket filesystem path from [`socket::SOCKET_PATH_ENV`] (`$NIRI_SOCKET`).
//! 2. Connect to the socket and write a JSON-formatted [`Request`] on a single line. You can follow
//!    up with a line break and a flush, or just flush and shutdown the write end of the socket.
//! 3. Niri will respond with a single line JSON-formatted [`Reply`].
//! 4. You can keep writing [`Request`]s, each on a single line, and read [`Reply`]s, also each on a
//!    separate line.
//! 5. After you request an event stream, niri will keep responding with JSON-formatted [`Event`]s,
//!    on a single line each.
//!
//! ## Backwards compatibility
//!
//! This crate follows the niri version. It is **not** API-stable in terms of the Rust semver. In
//! particular, expect new struct fields and enum variants to be added in patch version bumps.
//!
//! Use an exact version requirement to avoid breaking changes:
//!
//! ```toml
//! [dependencies]
//! niri-ipc = "=26.4.0"
//! ```
//!
//! ## Features
//!
//! This crate defines the following features:
//! - `json-schema`: derives the [schemars](https://lib.rs/crates/schemars) `JsonSchema` trait for
//!   the types.
//! - `clap`: derives the clap CLI parsing traits for some types. Used internally by niri itself.
#![warn(missing_docs)]

use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub mod socket;
pub mod state;

/// Request from client to niri.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Request {
    /// Request the version string for the running niri instance.
    Version,
    /// Request information about connected outputs.
    Outputs,
    /// Request information about workspaces.
    Workspaces,
    /// Request information about open windows.
    Windows,
    /// Request information about layer-shell surfaces.
    Layers,
    /// Request information about the configured keyboard layouts.
    KeyboardLayouts,
    /// Request information about the focused output.
    FocusedOutput,
    /// Request information about the focused window.
    FocusedWindow,
    /// Request picking a window and get its information.
    PickWindow,
    /// Request picking a color from the screen.
    PickColor,
    /// Perform an action.
    Action(Action),
    /// Change output configuration temporarily.
    ///
    /// The configuration is changed temporarily and not saved into the config file. If the output
    /// configuration subsequently changes in the config file, these temporary changes will be
    /// forgotten.
    Output {
        /// Output name.
        output: String,
        /// Configuration to apply.
        action: OutputAction,
    },
    /// Start continuously receiving events from the compositor.
    ///
    /// The compositor should reply with `Reply::Ok(Response::Handled)`, then continuously send
    /// [`Event`]s, one per line.
    ///
    /// The event stream will always give you the full current state up-front. For example, the
    /// first workspace-related event you will receive will be [`Event::WorkspacesChanged`]
    /// containing the full current workspaces state. You *do not* need to separately send
    /// [`Request::Workspaces`] when using the event stream.
    ///
    /// Where reasonable, event stream state updates are atomic, though this is not always the
    /// case. For example, a window may end up with a workspace id for a workspace that had already
    /// been removed. This can happen if the corresponding [`Event::WorkspacesChanged`] arrives
    /// before the corresponding [`Event::WindowOpenedOrChanged`].
    EventStream,
    /// Respond with an error (for testing error handling).
    ReturnError,
    /// Request information about the overview.
    OverviewState,
    /// Request information about screencasts.
    Casts,
    /// Capture this output's most recent composited frame — the raw
    /// BT.2020 absolute-nits *intermediate* (the linear light buffer the
    /// encode pass reads, before LUT / response-curve / OETF panel
    /// correction). The compositor reads the whole intermediate back
    /// into a memfd and replies with `Response::FrameCaptured(meta)`
    /// **plus the memfd passed as an out-of-band file descriptor**
    /// (`SCM_RIGHTS`), which clients receive via
    /// [`socket::Socket::send_recv_fd`]. The client `mmap`s the fd and
    /// reads `meta.byte_len` bytes of pixel data laid out per `meta`.
    /// On-demand and synchronous on the compositor side (a single
    /// ~one-frame hitch); intended for a frame inspector, not streaming.
    CaptureFrame {
        /// Output connector name (e.g. `DisplayPort-4`).
        output: String,
    },
    /// Fetch the measured gamut-surface mesh configured for this output
    /// (KDL `color.gamut "file"` — the `.gamut.json` sidecar written by
    /// `prism-tune calibrate-lut3d`). The compositor reads and parses the
    /// file on demand and replies with `Response::GamutMesh(Some(..))`, or
    /// `Response::GamutMesh(None)` when no gamut file is configured. The
    /// mesh is the panel's actual reachable color boundary; intended for
    /// the gamut-cloud inspector to overlay as a lattice shell.
    GamutMesh {
        /// Output connector name (e.g. `DisplayPort-4`).
        output: String,
    },
    /// Fetch the calibration 3D LUT this output's encode pass is
    /// *currently running* — the effective LUT after the compositor's
    /// override precedence (IPC-pushed measurement → CTM/curve override
    /// synthesis → KDL `.lut` file → KDL synthesis), which may never have
    /// existed on disk. The compositor replies with
    /// `Response::Lut3d(meta)` **plus a memfd passed as an out-of-band
    /// file descriptor** (`SCM_RIGHTS`), received via
    /// [`socket::Socket::send_recv_fd`]. The fd's first
    /// [`byte_len`](Lut3dMeta::byte_len) bytes are `cube_edge³` RGB
    /// entries laid out per [`Lut3dMeta`]. Intended for the prism-tune
    /// LUT inspector.
    Lut3d {
        /// Output connector name (e.g. `DisplayPort-4`).
        output: String,
    },
}

/// Reply from niri to client.
///
/// Every request gets one reply.
///
/// * If an error had occurred, it will be an `Reply::Err`.
/// * If the request does not need any particular response, it will be
///   `Reply::Ok(Response::Handled)`. Kind of like an `Ok(())`.
/// * Otherwise, it will be `Reply::Ok(response)` with one of the other [`Response`] variants.
pub type Reply = Result<Response, String>;

/// Successful response from niri to client.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Response {
    /// A request that does not need a response was handled successfully.
    Handled,
    /// The version string for the running niri instance.
    Version(String),
    /// Information about connected outputs.
    ///
    /// Map from output name to output info.
    Outputs(HashMap<String, Output>),
    /// Information about workspaces.
    Workspaces(Vec<Workspace>),
    /// Information about open windows.
    Windows(Vec<Window>),
    /// Information about layer-shell surfaces.
    Layers(Vec<LayerSurface>),
    /// Information about the keyboard layout.
    KeyboardLayouts(KeyboardLayouts),
    /// Information about the focused output.
    FocusedOutput(Option<Output>),
    /// Information about the focused window.
    FocusedWindow(Option<Window>),
    /// Information about the picked window.
    PickedWindow(Option<Window>),
    /// Information about the picked color.
    PickedColor(Option<PickedColor>),
    /// Output configuration change result.
    OutputConfigChanged(OutputConfigChanged),
    /// Information about the overview.
    OverviewState(Overview),
    /// Information about screencasts.
    Casts(Vec<Cast>),
    /// Result of an `OutputAction::EncodeDiagnose` request. The
    /// scanout-format output of the encode pipeline, decoded back to
    /// linear cd/m². Lets a calibration tool compare against an
    /// independently-computed prediction (e.g.
    /// `trilinear_sample_lut(entries, pq_oetf(input))`) to localize
    /// shader/LUT bugs vs. panel non-additivity.
    EncodeDiagnose(EncodeDiagnoseResult),
    /// Geometry + layout for a `Request::CaptureFrame`. The pixel data
    /// itself travels out-of-band as a memfd file descriptor (see
    /// [`socket::Socket::send_recv_fd`]); this describes how to read it.
    FrameCaptured(FrameMeta),
    /// The measured gamut-surface mesh for a `Request::GamutMesh`, or
    /// `None` when the output has no `color.gamut` file configured (or it
    /// failed to load). Small enough to travel inline as JSON.
    GamutMesh(Option<GamutMesh>),
    /// Layout + provenance for a `Request::Lut3d`. The entry data itself
    /// travels out-of-band as a memfd file descriptor (see
    /// [`socket::Socket::send_recv_fd`]); this describes how to read it.
    Lut3d(Lut3dMeta),
}

/// Describes the memfd payload of a `Response::FrameCaptured`. The fd's
/// first `byte_len` bytes are `height` rows of `width` pixels, each row
/// `stride_bytes` wide, in the pixel `format`, holding BT.2020
/// absolute-nits *linear* light (the intermediate's domain).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct FrameMeta {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row (tightly packed: `width * texel_bytes(format)`).
    pub stride_bytes: u32,
    /// Total payload length in the memfd, in bytes.
    pub byte_len: u64,
    /// Pixel layout of each texel in the memfd.
    pub format: FrameFormat,
}

/// Pixel layout of a captured frame's texels.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum FrameFormat {
    /// Four little-endian `f32` channels per texel: R, G, B, A (16
    /// bytes). RGB are BT.2020 absolute-nits linear; A is unused.
    Rgba32Float,
}

/// Describes the memfd payload of a `Response::Lut3d`. The fd's first
/// `byte_len` bytes are `cube_edge³` LUT entries of three little-endian
/// `f32`s each (R, G, B — 12 bytes per entry), X-fastest then Y then Z —
/// the same layout as the `.lut` file's data section. Entry `(i, j, k)`
/// is the panel command the encode shader emits for the PQ-shaped
/// BT.2020 input coordinate `(i, j, k) / (cube_edge − 1)`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Lut3dMeta {
    /// Grid points per axis (typically 33); the payload holds
    /// `cube_edge³` entries.
    pub cube_edge: u32,
    /// Total payload length in the memfd, in bytes
    /// (`cube_edge³ × 12`).
    pub byte_len: u64,
    /// What the entry values mean — the output's encode-chain domain
    /// (mirrors the `.lut` v5 `out_space` header field).
    pub out_space: Lut3dDomain,
    /// Which precedence level produced the effective LUT.
    pub source: Lut3dSource,
}

/// Output domain of a 3D LUT's entries — what the numbers mean.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Lut3dDomain {
    /// Per-channel commanded panel luminance in cd/m² (PQ/linear HDR
    /// chains — the OutputTransfer stage clamps + encodes for the wire).
    Nits,
    /// Linear panel drive in `[0, 1]` (the parameter-free sRGB chain;
    /// values above 1 are wire-clamped by the shader).
    Drive,
}

/// Which level of the compositor's LUT precedence chain produced the
/// effective LUT (see `Request::Lut3d`).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Lut3dSource {
    /// Pushed live over IPC (`LoadLut3dFromFile` / `IdentityLut3d`) —
    /// measurement-derived data from a calibration tool.
    IpcOverride,
    /// Loaded from the KDL-configured `color.lut3d` file at startup.
    KdlFile,
    /// Synthesized from the effective CTM + response curve (no measured
    /// LUT in effect).
    Synthesized,
}

/// Schema tag the `.gamut.json` sidecar is written with (and the only
/// version [`GamutMesh::load_json`] accepts).
pub const GAMUT_MESH_SCHEMA: &str = "prism-gamut-mesh.v1";

/// A measured gamut-surface mesh: the panel's actual reachable color
/// boundary, probed by `prism-tune calibrate-lut3d` and stored as the
/// `.gamut.json` sidecar of a `.lut` file. The boundary is the surface of
/// the command-space RGB cube, adaptively subdivided into quad
/// [`patches`](GamutMesh::patches) over shared [`vertices`](GamutMesh::vertices).
///
/// This is the wire/file form: a deserialized-friendly subset of
/// prism-tune's in-memory `gamut::GamutMesh` (drops mesh-internal book-
/// keeping). All XYZ are absolute (cd/m²), pre-black-subtraction.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct GamutMesh {
    /// Measured white corner (code value `(1,1,1)`), absolute XYZ (cd/m²).
    pub white_xyz: [f64; 3],
    /// Per-channel command-axis saturation peaks (cd/m²).
    pub cmd_axis_max_nits: [f64; 3],
    /// Shared boundary vertices; patches index into this list.
    pub vertices: Vec<GamutVertex>,
    /// Quad patches tiling the cube surface (corners index `vertices`).
    pub patches: Vec<GamutPatch>,
}

/// One measured boundary vertex of a [`GamutMesh`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct GamutVertex {
    /// Requested RGB code value, each in `[0, 1]`.
    pub code_value: [f64; 3],
    /// Actual per-channel commanded luminance (cd/m²).
    pub cmd_nits: [f64; 3],
    /// Measured colorimeter reading, absolute XYZ (cd/m²).
    pub xyz: [f64; 3],
    /// CIELAB relative to the mesh's own measured white (informational —
    /// consumers re-anchoring to a different white recompute from `xyz`).
    pub lab: [f64; 3],
    /// Confidence flag from the measurement burst.
    pub trustworthy: bool,
}

/// One quad face of a [`GamutMesh`] boundary.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct GamutPatch {
    /// The cube face's fixed axis (0 = R, 1 = G, 2 = B).
    pub axis: usize,
    /// The fixed axis's value (0.0 or 1.0).
    pub value: f64,
    /// Four corner indices into [`GamutMesh::vertices`], in CCW order.
    pub corners: [u32; 4],
    /// Why subdivision of this patch stopped.
    pub status: GamutPatchStatus,
}

/// Why a [`GamutPatch`] was not subdivided further.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum GamutPatchStatus {
    /// Planar enough (centre within tolerance of the bilinear average).
    Flat,
    /// Corners collapsed in measured space — pipeline clamping detected.
    Folded,
    /// Hit the subdivision depth cap while still curved.
    MaxDepth,
    /// A corner was untrustworthy; stopped rather than chase noise.
    LowTrust,
}

impl GamutMesh {
    /// Load a `.gamut.json` sidecar (schema [`GAMUT_MESH_SCHEMA`]).
    /// Ignores the file's redundant per-patch `face` label and the
    /// document `schema` tag beyond validating it.
    pub fn load_json(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_json_reader(std::io::BufReader::new(file))
    }

    /// Parse a `.gamut.json` document from any reader (the testable core
    /// of [`load_json`]). Validates the schema tag.
    pub fn from_json_reader(reader: impl std::io::Read) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};

        /// The on-disk document wrapper — the mesh plus its schema tag.
        #[derive(Deserialize)]
        struct Doc {
            schema: String,
            #[serde(flatten)]
            mesh: GamutMesh,
        }

        let doc: Doc =
            serde_json::from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        if doc.schema != GAMUT_MESH_SCHEMA {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "gamut mesh schema {:?}, expected {:?}",
                    doc.schema, GAMUT_MESH_SCHEMA
                ),
            ));
        }
        Ok(doc.mesh)
    }
}

/// Per-channel decoded scanout value returned by `OutputAction::EncodeDiagnose`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct EncodeDiagnoseResult {
    /// What the compositor's encode pipeline + LUT actually emitted
    /// for the requested input, decoded from the scanout format back
    /// to the encode chain's LUT-output domain: linear cd/m² for HDR
    /// PQ scanout, linear panel drive `[0, 1]` for SDR sRGB scanout.
    /// (Field name predates the drive-domain reform.)
    pub scanout_nits: [f64; 3],
}

/// Overview information.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Overview {
    /// Whether the overview is currently open.
    pub is_open: bool,
}

/// Color picked from the screen.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct PickedColor {
    /// Color values as red, green, blue, each ranging from 0.0 to 1.0.
    pub rgb: [f64; 3],
}

/// Actions that niri can perform.
// Variants in this enum should match the spelling of the ones in niri-config. Most, but not all,
// variants from niri-config should be present here.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "clap", derive(clap::Parser))]
#[cfg_attr(feature = "clap", command(subcommand_value_name = "ACTION"))]
#[cfg_attr(feature = "clap", command(subcommand_help_heading = "Actions"))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Action {
    /// Exit niri.
    Quit {
        /// Skip the "Press Enter to confirm" prompt.
        #[cfg_attr(feature = "clap", arg(short, long))]
        skip_confirmation: bool,
    },
    /// Power off all monitors via DPMS.
    PowerOffMonitors {},
    /// Power on all monitors via DPMS.
    PowerOnMonitors {},
    /// Spawn a command.
    Spawn {
        /// Command to spawn.
        #[cfg_attr(feature = "clap", arg(last = true, required = true))]
        command: Vec<String>,
    },
    /// Spawn a command through the shell.
    SpawnSh {
        /// Command to run.
        #[cfg_attr(feature = "clap", arg(last = true, required = true))]
        command: String,
    },
    /// Do a screen transition.
    DoScreenTransition {
        /// Delay in milliseconds for the screen to freeze before starting the transition.
        #[cfg_attr(feature = "clap", arg(short, long))]
        delay_ms: Option<u16>,
    },
    /// Open the screenshot UI.
    Screenshot {
        ///  Whether to show the mouse pointer by default in the screenshot UI.
        #[cfg_attr(feature = "clap", arg(short = 'p', long, action = clap::ArgAction::Set, default_value_t = true))]
        show_pointer: bool,

        /// Path to save the screenshot to.
        ///
        /// The path must be absolute, otherwise an error is returned.
        ///
        /// If `None`, the screenshot is saved according to the `screenshot-path` config setting.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set))]
        path: Option<String>,
    },
    /// Screenshot the focused screen.
    ScreenshotScreen {
        /// Write the screenshot to disk in addition to putting it in your clipboard.
        ///
        /// The screenshot is saved according to the `screenshot-path` config setting.
        #[cfg_attr(feature = "clap", arg(short = 'd', long, action = clap::ArgAction::Set, default_value_t = true))]
        write_to_disk: bool,

        /// Whether to include the mouse pointer in the screenshot.
        #[cfg_attr(feature = "clap", arg(short = 'p', long, action = clap::ArgAction::Set, default_value_t = true))]
        show_pointer: bool,

        /// Path to save the screenshot to.
        ///
        /// The path must be absolute, otherwise an error is returned.
        ///
        /// If `None`, the screenshot is saved according to the `screenshot-path` config setting.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set))]
        path: Option<String>,
    },
    /// Screenshot a window.
    #[cfg_attr(feature = "clap", clap(about = "Screenshot the focused window"))]
    ScreenshotWindow {
        /// Id of the window to screenshot.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
        /// Write the screenshot to disk in addition to putting it in your clipboard.
        ///
        /// The screenshot is saved according to the `screenshot-path` config setting.
        #[cfg_attr(feature = "clap", arg(short = 'd', long, action = clap::ArgAction::Set, default_value_t = true))]
        write_to_disk: bool,

        /// Whether to include the mouse pointer in the screenshot.
        ///
        /// The pointer will be included only if the window is currently receiving pointer input
        /// (usually this means the pointer is on top of the window).
        #[cfg_attr(feature = "clap", arg(short = 'p', long, action = clap::ArgAction::Set, default_value_t = false))]
        show_pointer: bool,

        /// Path to save the screenshot to.
        ///
        /// The path must be absolute, otherwise an error is returned.
        ///
        /// If `None`, the screenshot is saved according to the `screenshot-path` config setting.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set))]
        path: Option<String>,
    },
    /// Enable or disable the keyboard shortcuts inhibitor (if any) for the focused surface.
    ToggleKeyboardShortcutsInhibit {},
    /// Close a window.
    #[cfg_attr(feature = "clap", clap(about = "Close the focused window"))]
    CloseWindow {
        /// Id of the window to close.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Toggle fullscreen on a window.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Toggle fullscreen on the focused window")
    )]
    FullscreenWindow {
        /// Id of the window to toggle fullscreen of.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Toggle windowed (fake) fullscreen on a window.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Toggle windowed (fake) fullscreen on the focused window")
    )]
    ToggleWindowedFullscreen {
        /// Id of the window to toggle windowed fullscreen of.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Focus a window by id.
    FocusWindow {
        /// Id of the window to focus.
        #[cfg_attr(feature = "clap", arg(long))]
        id: u64,
    },
    /// Focus a window in the focused column by index.
    FocusWindowInColumn {
        /// Index of the window in the column.
        ///
        /// The index starts from 1 for the topmost window.
        #[cfg_attr(feature = "clap", arg())]
        index: u8,
    },
    /// Focus the previously focused window.
    FocusWindowPrevious {},
    /// Focus the column to the left.
    FocusColumnLeft {},
    /// Focus the column to the right.
    FocusColumnRight {},
    /// Focus the first column.
    FocusColumnFirst {},
    /// Focus the last column.
    FocusColumnLast {},
    /// Focus the next column to the right, looping if at end.
    FocusColumnRightOrFirst {},
    /// Focus the next column to the left, looping if at start.
    FocusColumnLeftOrLast {},
    /// Focus a column by index.
    FocusColumn {
        /// Index of the column to focus.
        ///
        /// The index starts from 1 for the first column.
        #[cfg_attr(feature = "clap", arg())]
        index: usize,
    },
    /// Focus the window or the monitor above.
    FocusWindowOrMonitorUp {},
    /// Focus the window or the monitor below.
    FocusWindowOrMonitorDown {},
    /// Focus the column or the monitor to the left.
    FocusColumnOrMonitorLeft {},
    /// Focus the column or the monitor to the right.
    FocusColumnOrMonitorRight {},
    /// Focus the window below.
    FocusWindowDown {},
    /// Focus the window above.
    FocusWindowUp {},
    /// Focus the window below or the column to the left.
    FocusWindowDownOrColumnLeft {},
    /// Focus the window below or the column to the right.
    FocusWindowDownOrColumnRight {},
    /// Focus the window above or the column to the left.
    FocusWindowUpOrColumnLeft {},
    /// Focus the window above or the column to the right.
    FocusWindowUpOrColumnRight {},
    /// Focus the window or the workspace below.
    FocusWindowOrWorkspaceDown {},
    /// Focus the window or the workspace above.
    FocusWindowOrWorkspaceUp {},
    /// Focus the topmost window.
    FocusWindowTop {},
    /// Focus the bottommost window.
    FocusWindowBottom {},
    /// Focus the window below or the topmost window.
    FocusWindowDownOrTop {},
    /// Focus the window above or the bottommost window.
    FocusWindowUpOrBottom {},
    /// Move the focused column to the left.
    MoveColumnLeft {},
    /// Move the focused column to the right.
    MoveColumnRight {},
    /// Move the focused column to the start of the workspace.
    MoveColumnToFirst {},
    /// Move the focused column to the end of the workspace.
    MoveColumnToLast {},
    /// Move the focused column to the left or to the monitor to the left.
    MoveColumnLeftOrToMonitorLeft {},
    /// Move the focused column to the right or to the monitor to the right.
    MoveColumnRightOrToMonitorRight {},
    /// Move the focused column to a specific index on its workspace.
    MoveColumnToIndex {
        /// New index for the column.
        ///
        /// The index starts from 1 for the first column.
        #[cfg_attr(feature = "clap", arg())]
        index: usize,
    },
    /// Move the focused window down in a column.
    MoveWindowDown {},
    /// Move the focused window up in a column.
    MoveWindowUp {},
    /// Move the focused window down in a column or to the workspace below.
    MoveWindowDownOrToWorkspaceDown {},
    /// Move the focused window up in a column or to the workspace above.
    MoveWindowUpOrToWorkspaceUp {},
    /// Consume or expel a window left.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Consume or expel the focused window left")
    )]
    ConsumeOrExpelWindowLeft {
        /// Id of the window to consume or expel.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Consume or expel a window right.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Consume or expel the focused window right")
    )]
    ConsumeOrExpelWindowRight {
        /// Id of the window to consume or expel.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Consume the window to the right into the focused column.
    ConsumeWindowIntoColumn {},
    /// Expel the bottom window from the focused column.
    ExpelWindowFromColumn {},
    /// Swap focused window with one to the right.
    SwapWindowRight {},
    /// Swap focused window with one to the left.
    SwapWindowLeft {},
    /// Toggle the focused column between normal and tabbed display.
    ToggleColumnTabbedDisplay {},
    /// Set the display mode of the focused column.
    SetColumnDisplay {
        /// Display mode to set.
        #[cfg_attr(feature = "clap", arg())]
        display: ColumnDisplay,
    },
    /// Center the focused column on the screen.
    CenterColumn {},
    /// Center a window on the screen.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Center the focused window on the screen")
    )]
    CenterWindow {
        /// Id of the window to center.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Center all fully visible columns on the screen.
    CenterVisibleColumns {},
    /// Focus the workspace below.
    FocusWorkspaceDown {},
    /// Focus the workspace above.
    FocusWorkspaceUp {},
    /// Focus a workspace by reference (index or name).
    FocusWorkspace {
        /// Reference (index or name) of the workspace to focus.
        #[cfg_attr(feature = "clap", arg())]
        reference: WorkspaceReferenceArg,
    },
    /// Focus the previous workspace.
    FocusWorkspacePrevious {},
    /// Move the focused window to the workspace below.
    MoveWindowToWorkspaceDown {
        /// Whether the focus should follow the target workspace.
        ///
        /// If `true` (the default), the focus will follow the window to the new workspace. If
        /// `false`, the focus will remain on the original workspace.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]
        focus: bool,
    },
    /// Move the focused window to the workspace above.
    MoveWindowToWorkspaceUp {
        /// Whether the focus should follow the target workspace.
        ///
        /// If `true` (the default), the focus will follow the window to the new workspace. If
        /// `false`, the focus will remain on the original workspace.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]
        focus: bool,
    },
    /// Move a window to a workspace.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Move the focused window to a workspace by reference (index or name)")
    )]
    MoveWindowToWorkspace {
        /// Id of the window to move.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        window_id: Option<u64>,

        /// Reference (index or name) of the workspace to move the window to.
        #[cfg_attr(feature = "clap", arg())]
        reference: WorkspaceReferenceArg,

        /// Whether the focus should follow the moved window.
        ///
        /// If `true` (the default) and the window to move is focused, the focus will follow the
        /// window to the new workspace. If `false`, the focus will remain on the original
        /// workspace.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]
        focus: bool,
    },
    /// Move the focused column to the workspace below.
    MoveColumnToWorkspaceDown {
        /// Whether the focus should follow the target workspace.
        ///
        /// If `true` (the default), the focus will follow the column to the new workspace. If
        /// `false`, the focus will remain on the original workspace.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]
        focus: bool,
    },
    /// Move the focused column to the workspace above.
    MoveColumnToWorkspaceUp {
        /// Whether the focus should follow the target workspace.
        ///
        /// If `true` (the default), the focus will follow the column to the new workspace. If
        /// `false`, the focus will remain on the original workspace.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]
        focus: bool,
    },
    /// Move the focused column to a workspace by reference (index or name).
    MoveColumnToWorkspace {
        /// Reference (index or name) of the workspace to move the column to.
        #[cfg_attr(feature = "clap", arg())]
        reference: WorkspaceReferenceArg,

        /// Whether the focus should follow the target workspace.
        ///
        /// If `true` (the default), the focus will follow the column to the new workspace. If
        /// `false`, the focus will remain on the original workspace.
        #[cfg_attr(feature = "clap", arg(long, action = clap::ArgAction::Set, default_value_t = true))]
        focus: bool,
    },
    /// Move the focused workspace down.
    MoveWorkspaceDown {},
    /// Move the focused workspace up.
    MoveWorkspaceUp {},
    /// Move a workspace to a specific index on its monitor.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Move the focused workspace to a specific index on its monitor")
    )]
    MoveWorkspaceToIndex {
        /// New index for the workspace.
        #[cfg_attr(feature = "clap", arg())]
        index: usize,

        /// Reference (index or name) of the workspace to move.
        ///
        /// If `None`, uses the focused workspace.
        #[cfg_attr(feature = "clap", arg(long))]
        reference: Option<WorkspaceReferenceArg>,
    },
    /// Set the name of a workspace.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Set the name of the focused workspace")
    )]
    SetWorkspaceName {
        /// New name for the workspace.
        #[cfg_attr(feature = "clap", arg())]
        name: String,

        /// Reference (index or name) of the workspace to name.
        ///
        /// If `None`, uses the focused workspace.
        #[cfg_attr(feature = "clap", arg(long))]
        workspace: Option<WorkspaceReferenceArg>,
    },
    /// Unset the name of a workspace.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Unset the name of the focused workspace")
    )]
    UnsetWorkspaceName {
        /// Reference (index or name) of the workspace to unname.
        ///
        /// If `None`, uses the focused workspace.
        #[cfg_attr(feature = "clap", arg())]
        reference: Option<WorkspaceReferenceArg>,
    },
    /// Focus the monitor to the left.
    FocusMonitorLeft {},
    /// Focus the monitor to the right.
    FocusMonitorRight {},
    /// Focus the monitor below.
    FocusMonitorDown {},
    /// Focus the monitor above.
    FocusMonitorUp {},
    /// Focus the previous monitor.
    FocusMonitorPrevious {},
    /// Focus the next monitor.
    FocusMonitorNext {},
    /// Focus a monitor by name.
    FocusMonitor {
        /// Name of the output to focus.
        #[cfg_attr(feature = "clap", arg())]
        output: String,
    },
    /// Move the focused window to the monitor to the left.
    MoveWindowToMonitorLeft {},
    /// Move the focused window to the monitor to the right.
    MoveWindowToMonitorRight {},
    /// Move the focused window to the monitor below.
    MoveWindowToMonitorDown {},
    /// Move the focused window to the monitor above.
    MoveWindowToMonitorUp {},
    /// Move the focused window to the previous monitor.
    MoveWindowToMonitorPrevious {},
    /// Move the focused window to the next monitor.
    MoveWindowToMonitorNext {},
    /// Move a window to a specific monitor.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Move the focused window to a specific monitor")
    )]
    MoveWindowToMonitor {
        /// Id of the window to move.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,

        /// The target output name.
        #[cfg_attr(feature = "clap", arg())]
        output: String,
    },
    /// Move the focused column to the monitor to the left.
    MoveColumnToMonitorLeft {},
    /// Move the focused column to the monitor to the right.
    MoveColumnToMonitorRight {},
    /// Move the focused column to the monitor below.
    MoveColumnToMonitorDown {},
    /// Move the focused column to the monitor above.
    MoveColumnToMonitorUp {},
    /// Move the focused column to the previous monitor.
    MoveColumnToMonitorPrevious {},
    /// Move the focused column to the next monitor.
    MoveColumnToMonitorNext {},
    /// Move the focused column to a specific monitor.
    MoveColumnToMonitor {
        /// The target output name.
        #[cfg_attr(feature = "clap", arg())]
        output: String,
    },
    /// Change the width of a window.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Change the width of the focused window")
    )]
    SetWindowWidth {
        /// Id of the window whose width to set.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,

        /// How to change the width.
        #[cfg_attr(feature = "clap", arg(allow_hyphen_values = true))]
        change: SizeChange,
    },
    /// Change the height of a window.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Change the height of the focused window")
    )]
    SetWindowHeight {
        /// Id of the window whose height to set.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,

        /// How to change the height.
        #[cfg_attr(feature = "clap", arg(allow_hyphen_values = true))]
        change: SizeChange,
    },
    /// Reset the height of a window back to automatic.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Reset the height of the focused window back to automatic")
    )]
    ResetWindowHeight {
        /// Id of the window whose height to reset.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Switch between preset column widths.
    SwitchPresetColumnWidth {},
    /// Switch between preset column widths backwards.
    SwitchPresetColumnWidthBack {},
    /// Switch between preset window widths.
    SwitchPresetWindowWidth {
        /// Id of the window whose width to switch.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Switch between preset window widths backwards.
    SwitchPresetWindowWidthBack {
        /// Id of the window whose width to switch.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Switch between preset window heights.
    SwitchPresetWindowHeight {
        /// Id of the window whose height to switch.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Switch between preset window heights backwards.
    SwitchPresetWindowHeightBack {
        /// Id of the window whose height to switch.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Toggle the maximized state of the focused column.
    MaximizeColumn {},
    /// Toggle the maximized-to-edges state of the focused window.
    MaximizeWindowToEdges {
        /// Id of the window to maximize.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Change the width of the focused column.
    SetColumnWidth {
        /// How to change the width.
        #[cfg_attr(feature = "clap", arg(allow_hyphen_values = true))]
        change: SizeChange,
    },
    /// Expand the focused column to space not taken up by other fully visible columns.
    ExpandColumnToAvailableWidth {},
    /// Switch between keyboard layouts.
    SwitchLayout {
        /// Layout to switch to.
        #[cfg_attr(feature = "clap", arg())]
        layout: LayoutSwitchTarget,
    },
    /// Show the hotkey overlay.
    ShowHotkeyOverlay {},
    /// Move the focused workspace to the monitor to the left.
    MoveWorkspaceToMonitorLeft {},
    /// Move the focused workspace to the monitor to the right.
    MoveWorkspaceToMonitorRight {},
    /// Move the focused workspace to the monitor below.
    MoveWorkspaceToMonitorDown {},
    /// Move the focused workspace to the monitor above.
    MoveWorkspaceToMonitorUp {},
    /// Move the focused workspace to the previous monitor.
    MoveWorkspaceToMonitorPrevious {},
    /// Move the focused workspace to the next monitor.
    MoveWorkspaceToMonitorNext {},
    /// Move a workspace to a specific monitor.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Move the focused workspace to a specific monitor")
    )]
    MoveWorkspaceToMonitor {
        /// The target output name.
        #[cfg_attr(feature = "clap", arg())]
        output: String,

        // Reference (index or name) of the workspace to move.
        ///
        /// If `None`, uses the focused workspace.
        #[cfg_attr(feature = "clap", arg(long))]
        reference: Option<WorkspaceReferenceArg>,
    },
    /// Toggle a debug tint on windows.
    ToggleDebugTint {},
    /// Toggle visualization of render element opaque regions.
    DebugToggleOpaqueRegions {},
    /// Toggle visualization of output damage.
    DebugToggleDamage {},
    /// Move the focused window between the floating and the tiling layout.
    ToggleWindowFloating {
        /// Id of the window to move.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Move the focused window to the floating layout.
    MoveWindowToFloating {
        /// Id of the window to move.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Move the focused window to the tiling layout.
    MoveWindowToTiling {
        /// Id of the window to move.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Switches focus to the floating layout.
    FocusFloating {},
    /// Switches focus to the tiling layout.
    FocusTiling {},
    /// Toggles the focus between the floating and the tiling layout.
    SwitchFocusBetweenFloatingAndTiling {},
    /// Move a floating window on screen.
    #[cfg_attr(feature = "clap", clap(about = "Move the floating window on screen"))]
    MoveFloatingWindow {
        /// Id of the window to move.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,

        /// How to change the X position.
        #[cfg_attr(
            feature = "clap",
            arg(short, long, default_value = "+0", allow_hyphen_values = true)
        )]
        x: PositionChange,

        /// How to change the Y position.
        #[cfg_attr(
            feature = "clap",
            arg(short, long, default_value = "+0", allow_hyphen_values = true)
        )]
        y: PositionChange,
    },
    /// Toggle the opacity of a window.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Toggle the opacity of the focused window")
    )]
    ToggleWindowRuleOpacity {
        /// Id of the window.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Set the dynamic cast target to a window.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Set the dynamic cast target to the focused window")
    )]
    SetDynamicCastWindow {
        /// Id of the window to target.
        ///
        /// If `None`, uses the focused window.
        #[cfg_attr(feature = "clap", arg(long))]
        id: Option<u64>,
    },
    /// Set the dynamic cast target to a monitor.
    #[cfg_attr(
        feature = "clap",
        clap(about = "Set the dynamic cast target to the focused monitor")
    )]
    SetDynamicCastMonitor {
        /// Name of the output to target.
        ///
        /// If `None`, uses the focused output.
        #[cfg_attr(feature = "clap", arg())]
        output: Option<String>,
    },
    /// Clear the dynamic cast target, making it show nothing.
    ClearDynamicCastTarget {},
    /// Stop a PipeWire screencast.
    ///
    /// wlr-screencopy screencasts cannot currently be stopped via IPC.
    StopCast {
        /// Session ID of the screencast to stop.
        ///
        /// If the session has multiple screencast streams, this will stop all of them.
        #[cfg_attr(feature = "clap", arg(long))]
        session_id: u64,
    },
    /// Toggle (open/close) the Overview.
    ToggleOverview {},
    /// Open the Overview.
    OpenOverview {},
    /// Close the Overview.
    CloseOverview {},
    /// Toggle urgent status of a window.
    ToggleWindowUrgent {
        /// Id of the window to toggle urgent.
        #[cfg_attr(feature = "clap", arg(long))]
        id: u64,
    },
    /// Set urgent status of a window.
    SetWindowUrgent {
        /// Id of the window to set urgent.
        #[cfg_attr(feature = "clap", arg(long))]
        id: u64,
    },
    /// Unset urgent status of a window.
    UnsetWindowUrgent {
        /// Id of the window to unset urgent.
        #[cfg_attr(feature = "clap", arg(long))]
        id: u64,
    },
    /// Reload the config file.
    ///
    /// Can be useful for scripts changing the config file, to avoid waiting the small duration for
    /// niri's config file watcher to notice the changes.
    LoadConfigFile {
        /// Path of a new config file to load.
        ///
        /// If unset, reloads the current config file.
        #[cfg_attr(feature = "clap", arg(long))]
        path: Option<String>,
    },
}

/// Change in window or column size.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum SizeChange {
    /// Set the size in logical pixels.
    SetFixed(i32),
    /// Set the size as a proportion of the working area.
    SetProportion(f64),
    /// Add or subtract to the current size in logical pixels.
    AdjustFixed(i32),
    /// Add or subtract to the current size as a proportion of the working area.
    AdjustProportion(f64),
}

/// Change in floating window position.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum PositionChange {
    /// Set the position in logical pixels.
    SetFixed(f64),
    /// Set the position as a proportion of the working area.
    SetProportion(f64),
    /// Add or subtract to the current position in logical pixels.
    AdjustFixed(f64),
    /// Add or subtract to the current position as a proportion of the working area.
    AdjustProportion(f64),
}

/// Workspace reference (id, index or name) to operate on.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum WorkspaceReferenceArg {
    /// Id of the workspace.
    Id(u64),
    /// Index of the workspace.
    Index(u8),
    /// Name of the workspace.
    Name(String),
}

/// Layout to switch to.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum LayoutSwitchTarget {
    /// The next configured layout.
    Next,
    /// The previous configured layout.
    Prev,
    /// The specific layout by index.
    Index(u8),
}

/// How windows display in a column.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum ColumnDisplay {
    /// Windows are tiled vertically across the working area height.
    Normal,
    /// Windows are in tabs.
    Tabbed,
}

/// Output actions that niri can perform.
// Variants in this enum should match the spelling of the ones in niri-config. Most thigs from
// niri-config should be present here.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "clap", derive(clap::Parser))]
#[cfg_attr(feature = "clap", command(subcommand_value_name = "ACTION"))]
#[cfg_attr(feature = "clap", command(subcommand_help_heading = "Actions"))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum OutputAction {
    /// Turn off the output.
    Off,
    /// Turn on the output.
    On,
    /// Set the output mode.
    Mode {
        /// Mode to set, or "auto" for automatic selection.
        ///
        /// Run `niri msg outputs` to see the available modes.
        #[cfg_attr(feature = "clap", arg())]
        mode: ModeToSet,
    },
    /// Set a custom output mode.
    CustomMode {
        /// Custom mode to set.
        #[cfg_attr(feature = "clap", arg())]
        mode: ConfiguredMode,
    },
    /// Set a custom VESA CVT modeline.
    #[cfg_attr(feature = "clap", arg())]
    Modeline {
        /// The rate at which pixels are drawn in MHz.
        #[cfg_attr(feature = "clap", arg())]
        clock: f64,
        /// Horizontal active pixels.
        #[cfg_attr(feature = "clap", arg())]
        hdisplay: u16,
        /// Horizontal sync pulse start position in pixels.
        #[cfg_attr(feature = "clap", arg())]
        hsync_start: u16,
        /// Horizontal sync pulse end position in pixels.
        #[cfg_attr(feature = "clap", arg())]
        hsync_end: u16,
        /// Total horizontal number of pixels before resetting the horizontal drawing position to
        /// zero.
        #[cfg_attr(feature = "clap", arg())]
        htotal: u16,

        /// Vertical active pixels.
        #[cfg_attr(feature = "clap", arg())]
        vdisplay: u16,
        /// Vertical sync pulse start position in pixels.
        #[cfg_attr(feature = "clap", arg())]
        vsync_start: u16,
        /// Vertical sync pulse end position in pixels.
        #[cfg_attr(feature = "clap", arg())]
        vsync_end: u16,
        /// Total vertical number of pixels before resetting the vertical drawing position to zero.
        #[cfg_attr(feature = "clap", arg())]
        vtotal: u16,
        /// Horizontal sync polarity: "+hsync" or "-hsync".
        #[cfg_attr(feature = "clap", arg(allow_hyphen_values = true))]
        hsync_polarity: HSyncPolarity,
        /// Vertical sync polarity: "+vsync" or "-vsync".
        #[cfg_attr(feature = "clap", arg(allow_hyphen_values = true))]
        vsync_polarity: VSyncPolarity,
    },
    /// Set the output scale.
    Scale {
        /// Scale factor to set, or "auto" for automatic selection.
        #[cfg_attr(feature = "clap", arg())]
        scale: ScaleToSet,
    },
    /// Set the output transform.
    Transform {
        /// Transform to set, counter-clockwise.
        #[cfg_attr(feature = "clap", arg())]
        transform: Transform,
    },
    /// Set the output position.
    Position {
        /// Position to set, or "auto" for automatic selection.
        #[cfg_attr(feature = "clap", command(subcommand))]
        position: PositionToSet,
    },
    /// Set the variable refresh rate mode.
    Vrr {
        /// Variable refresh rate mode to set.
        #[cfg_attr(feature = "clap", command(flatten))]
        vrr: VrrToSet,
    },
    /// Set what absolute luminance (cd/m²) "SDR white" maps to for
    /// color-unaware clients on this output. Persists until cleared
    /// with `reset-color` or the compositor restarts (sticky across
    /// config-file reloads, by design — calibration runs take minutes
    /// and shouldn't race the file watcher). IEC sRGB default is 80.
    SdrReferenceNits {
        /// Reference luminance, cd/m². Clamped to [1, 10_000].
        #[cfg_attr(feature = "clap", arg())]
        nits: f64,
    },
    /// Set the panel response correction (per-channel gain + gamma).
    /// The encoder inverts the panel's measured
    /// `emitted = gain * commanded^gamma` so commanded nits match
    /// what the panel actually emits. Identity = gain 1, gamma 1.
    /// Persists until `reset-color` or restart.
    ResponseCurve {
        /// Red-channel gain.
        #[cfg_attr(feature = "clap", arg(long))]
        gain_r: f64,
        /// Green-channel gain.
        #[cfg_attr(feature = "clap", arg(long))]
        gain_g: f64,
        /// Blue-channel gain.
        #[cfg_attr(feature = "clap", arg(long))]
        gain_b: f64,
        /// Red-channel gamma exponent.
        #[cfg_attr(feature = "clap", arg(long))]
        gamma_r: f64,
        /// Green-channel gamma exponent.
        #[cfg_attr(feature = "clap", arg(long))]
        gamma_g: f64,
        /// Blue-channel gamma exponent.
        #[cfg_attr(feature = "clap", arg(long))]
        gamma_b: f64,
    },
    /// Set the per-channel panel peak luminance — the f32 IR's
    /// per-subpixel clamp ceiling and the basis for the
    /// HDR_OUTPUT_METADATA infoframe pushed to the connected sink.
    /// Real subpixel peaks differ (OLED ABL, LCD color-filter
    /// transmission), so the calibration pipeline measures each
    /// independently and applies them as a triple here. Sticky
    /// until `reset-color` or restart.
    PanelPeakNits {
        /// Red-channel peak luminance, cd/m². Clamped to [1, 10_000].
        #[cfg_attr(feature = "clap", arg(long))]
        nits_r: f64,
        /// Green-channel peak luminance, cd/m². Clamped to [1, 10_000].
        #[cfg_attr(feature = "clap", arg(long))]
        nits_g: f64,
        /// Blue-channel peak luminance, cd/m². Clamped to [1, 10_000].
        #[cfg_attr(feature = "clap", arg(long))]
        nits_b: f64,
    },
    /// Set the per-output 3×3 gamut-correction matrix. The encode
    /// shader applies `panel_rgb = M * bt2020_rgb` to map BT.2020 IR
    /// values into the panel's native-primary linear nits before the
    /// per-channel response curve and OETF. Derived from measured
    /// primaries: `M = panel_RGB_to_XYZ⁻¹ · BT2020_RGB_to_XYZ`. Field
    /// names are row-major: `rg` means "row R, column G" = how much
    /// input G contributes to output R. Identity matrix (1 on the
    /// diagonal, 0 elsewhere) is a no-op. Sticky until `reset-color`
    /// or restart.
    Ctm {
        /// Row R column R coefficient (input R contribution to output R).
        #[cfg_attr(feature = "clap", arg(long))]
        rr: f64,
        /// Row R column G coefficient (input G contribution to output R).
        #[cfg_attr(feature = "clap", arg(long))]
        rg: f64,
        /// Row R column B coefficient (input B contribution to output R).
        #[cfg_attr(feature = "clap", arg(long))]
        rb: f64,
        /// Row G column R coefficient (input R contribution to output G).
        #[cfg_attr(feature = "clap", arg(long))]
        gr: f64,
        /// Row G column G coefficient (input G contribution to output G).
        #[cfg_attr(feature = "clap", arg(long))]
        gg: f64,
        /// Row G column B coefficient (input B contribution to output G).
        #[cfg_attr(feature = "clap", arg(long))]
        gb: f64,
        /// Row B column R coefficient (input R contribution to output B).
        #[cfg_attr(feature = "clap", arg(long))]
        br: f64,
        /// Row B column G coefficient (input G contribution to output B).
        #[cfg_attr(feature = "clap", arg(long))]
        bg: f64,
        /// Row B column B coefficient (input B contribution to output B).
        #[cfg_attr(feature = "clap", arg(long))]
        bb: f64,
    },
    /// Load a binary 3D LUT from the named file and push its entries
    /// into this output's color pipeline. Takes precedence over both
    /// the (CTM, response-curve) IPC overrides AND the KDL `color.lut3d`
    /// file at config time — calibration tools use this to make a
    /// freshly-measured LUT live without restarting prism.
    ///
    /// File format: see `prism_renderer::lut3d::LutFileHeader`. Cube
    /// edge must match the renderer's allocated texture (17 in current
    /// builds) or the load is rejected. Sticky until `ResetColor`
    /// clears it.
    LoadLut3dFromFile {
        /// Absolute path to the `.lut` file. The prism daemon opens
        /// it directly — both ends must see the same filesystem.
        #[cfg_attr(feature = "clap", arg(long))]
        path: String,
    },
    /// Force this output's 3D LUT to identity, ignoring whatever the
    /// KDL config says. The panel renders raw commanded values — what
    /// `calibrate-lut3d`'s per-channel sweep needs so the measurements
    /// aren't pre-transformed by an already-active calibration.
    ///
    /// Without this, `ResetColor` only clears IPC overrides; KDL
    /// `color { ctm … response-curve … }` stays active and the encode
    /// shader's LUT is whatever those values synthesize to —
    /// silently mislabeling sweep measurements. Sticky until
    /// `ResetColor` clears the override.
    IdentityLut3d,
    /// Run the per-output encode pipeline once against a 1×1 scratch
    /// with `input_nits` as the synthetic intermediate value. The
    /// compositor reads back the scanout-format output, decodes it
    /// back to linear cd/m², and returns the result as
    /// `Response::EncodeDiagnose`. Lets calibration tools verify the
    /// LUT path produces what they think it does — closes the
    /// otherwise-feed-forward calibration loop independent of the
    /// colorimeter.
    EncodeDiagnose {
        /// Red-channel input to push into the diagnostic intermediate
        /// (linear cd/m², BT.2020 domain).
        #[cfg_attr(feature = "clap", arg(long))]
        r: f64,
        /// Green-channel input (linear cd/m², BT.2020).
        #[cfg_attr(feature = "clap", arg(long))]
        g: f64,
        /// Blue-channel input (linear cd/m², BT.2020).
        #[cfg_attr(feature = "clap", arg(long))]
        b: f64,
    },
    /// Set the absolute peak luminance (cd/m²) this output advertises to
    /// color-management clients as its mastering-display ceiling — the
    /// `mastering_luminance` max in the preferred `wp_color_management_v1`
    /// image description, i.e. the value a well-behaved client tone-maps
    /// against. Independent of the panel-facing `max-luminance` (which
    /// drives the HDR_OUTPUT_METADATA infoframe and the encode clamp):
    /// tuning this changes only what color-managed clients are told.
    /// Clients with a live surface-feedback object get a
    /// `preferred_changed2` so they re-query. No effect on SDR outputs.
    /// Sticky until `reset-color` or restart.
    AdvertisedPeakNits {
        /// Advertised peak luminance, cd/m². Clamped to [1, 10_000].
        #[cfg_attr(feature = "clap", arg(long))]
        nits: f64,
    },
    /// Clear all runtime color overrides for this output (sdr
    /// reference, response curve, panel peak nits, ctm, lut3d,
    /// advertised peak nits). Subsequent rendering reverts to whatever's
    /// in the persisted KDL config.
    ResetColor,
}

/// Output mode to set.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum ModeToSet {
    /// Niri will pick the mode automatically.
    Automatic,
    /// Specific mode.
    Specific(ConfiguredMode),
}

/// Output mode as set in the config file.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct ConfiguredMode {
    /// Width in physical pixels.
    pub width: u16,
    /// Height in physical pixels.
    pub height: u16,
    /// Refresh rate.
    pub refresh: Option<f64>,
}

/// Modeline horizontal syncing polarity.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum HSyncPolarity {
    /// Positive polarity.
    PHSync,
    /// Negative polarity.
    NHSync,
}

/// Modeline vertical syncing polarity.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum VSyncPolarity {
    /// Positive polarity.
    PVSync,
    /// Negative polarity.
    NVSync,
}

/// Output scale to set.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum ScaleToSet {
    /// Niri will pick the scale automatically.
    Automatic,
    /// Specific scale.
    Specific(f64),
}

/// Output position to set.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Subcommand))]
#[cfg_attr(feature = "clap", command(subcommand_value_name = "POSITION"))]
#[cfg_attr(feature = "clap", command(subcommand_help_heading = "Position Values"))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum PositionToSet {
    /// Position the output automatically.
    #[cfg_attr(feature = "clap", command(name = "auto"))]
    Automatic,
    /// Set a specific position.
    #[cfg_attr(feature = "clap", command(name = "set"))]
    Specific(ConfiguredPosition),
}

/// Output position as set in the config file.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct ConfiguredPosition {
    /// Logical X position.
    pub x: i32,
    /// Logical Y position.
    pub y: i32,
}

/// Output VRR to set.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct VrrToSet {
    /// Whether to enable variable refresh rate.
    #[cfg_attr(
        feature = "clap",
        arg(
            value_name = "ON|OFF",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new(),
            hide_possible_values = true,
        ),
    )]
    pub vrr: bool,
    /// Only enable when the output shows a window matching the variable-refresh-rate window rule.
    #[cfg_attr(feature = "clap", arg(long))]
    pub on_demand: bool,
}

/// Connected output.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Output {
    /// Name of the output.
    pub name: String,
    /// Textual description of the manufacturer.
    pub make: String,
    /// Textual description of the model.
    pub model: String,
    /// Serial of the output, if known.
    pub serial: Option<String>,
    /// Physical width and height of the output in millimeters, if known.
    pub physical_size: Option<(u32, u32)>,
    /// Available modes for the output.
    pub modes: Vec<Mode>,
    /// Index of the current mode in [`Self::modes`].
    ///
    /// `None` if the output is disabled.
    pub current_mode: Option<usize>,
    /// Whether the current_mode is a custom mode.
    pub is_custom_mode: bool,
    /// Whether the output supports variable refresh rate.
    pub vrr_supported: bool,
    /// Whether variable refresh rate is enabled on the output.
    pub vrr_enabled: bool,
    /// Logical output information.
    ///
    /// `None` if the output is not mapped to any logical output (for example, if it is disabled).
    pub logical: Option<LogicalOutput>,
    /// Current color-pipeline state — what the render path is actually
    /// using on this output right now, after runtime overrides and KDL
    /// config resolution. Calibration tools query this to decide whether
    /// to run in SDR or HDR mode and what the current per-channel peaks
    /// + response curve already are.
    pub color: ColorState,
}

/// Snapshot of the effective per-output color pipeline state. Reflects
/// runtime overrides (from `OutputAction::SdrReferenceNits` /
/// `ResponseCurve` / `PanelPeakNits` / `AdvertisedPeakNits`) and
/// persisted KDL config, resolved into the values the render path
/// (and color-management advertisement) actually uses.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct ColorState {
    /// True when the output's HDR signaling path is active — i.e.
    /// `wp_color_management_v1` PQ + BT.2020 with `HDR_OUTPUT_METADATA`
    /// pushed to the sink. False for SDR (sRGB) outputs.
    pub hdr_active: bool,
    /// Effective per-channel panel peak luminance (cd/m²) used by the
    /// decoder's display-referred clamp.
    pub panel_peak_nits: [f64; 3],
    /// Effective absolute luminance "SDR white" maps to (cd/m²). Used
    /// by color-unaware clients.
    pub sdr_reference_nits: f64,
    /// Effective per-channel response correction. `None` = identity
    /// (no correction applied).
    pub response_curve: Option<ResponseCurveState>,
    /// Effective 3×3 gamut-correction matrix, row-major. `None` =
    /// identity (no matrix applied; BT.2020 IR drives panel primaries
    /// directly without gamut correction).
    pub ctm: Option<[[f64; 3]; 3]>,
    /// Effective peak luminance (cd/m²) advertised to color-management
    /// clients as the mastering-display ceiling. `None` for SDR outputs
    /// (no mastering metadata advertised).
    pub advertised_peak_nits: Option<f64>,
}

/// Per-channel response correction snapshot. See [`ColorState::response_curve`].
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct ResponseCurveState {
    /// Per-channel gain (R, G, B).
    pub gain: [f64; 3],
    /// Per-channel gamma exponent (R, G, B).
    pub gamma: [f64; 3],
}

/// Output mode.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Mode {
    /// Width in physical pixels.
    pub width: u16,
    /// Height in physical pixels.
    pub height: u16,
    /// Refresh rate in millihertz.
    pub refresh_rate: u32,
    /// Whether this mode is preferred by the monitor.
    pub is_preferred: bool,
}

/// Logical output in the compositor's coordinate space.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct LogicalOutput {
    /// Logical X position.
    pub x: i32,
    /// Logical Y position.
    pub y: i32,
    /// Width in logical pixels.
    pub width: u32,
    /// Height in logical pixels.
    pub height: u32,
    /// Scale factor.
    pub scale: f64,
    /// Transform.
    pub transform: Transform,
}

/// Output transform, which goes counter-clockwise.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Transform {
    /// Untransformed.
    Normal,
    /// Rotated by 90°.
    #[serde(rename = "90")]
    _90,
    /// Rotated by 180°.
    #[serde(rename = "180")]
    _180,
    /// Rotated by 270°.
    #[serde(rename = "270")]
    _270,
    /// Flipped horizontally.
    Flipped,
    /// Rotated by 90° and flipped horizontally.
    #[cfg_attr(feature = "clap", value(name("flipped-90")))]
    Flipped90,
    /// Flipped vertically.
    #[cfg_attr(feature = "clap", value(name("flipped-180")))]
    Flipped180,
    /// Rotated by 270° and flipped horizontally.
    #[cfg_attr(feature = "clap", value(name("flipped-270")))]
    Flipped270,
}

/// Toplevel window.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Window {
    /// Unique id of this window.
    ///
    /// This id remains constant while this window is open.
    ///
    /// Do not assume that window ids will always increase without wrapping, or start at 1. That is
    /// an implementation detail subject to change. For example, ids may change to be randomly
    /// generated for each new window.
    pub id: u64,
    /// Title, if set.
    pub title: Option<String>,
    /// Application ID, if set.
    pub app_id: Option<String>,
    /// Process ID that created the Wayland connection for this window, if known.
    ///
    /// Currently, windows created by xdg-desktop-portal-gnome will have a `None` PID, but this may
    /// change in the future.
    pub pid: Option<i32>,
    /// Id of the workspace this window is on, if any.
    pub workspace_id: Option<u64>,
    /// Whether this window is currently focused.
    ///
    /// There can be either one focused window or zero (e.g. when a layer-shell surface has focus).
    pub is_focused: bool,
    /// Whether this window is currently floating.
    ///
    /// If the window isn't floating then it is in the tiling layout.
    pub is_floating: bool,
    /// Whether this window requests your attention.
    pub is_urgent: bool,
    /// Position- and size-related properties of the window.
    pub layout: WindowLayout,
    /// Timestamp when the window was most recently focused.
    ///
    /// This timestamp is intended for most-recently-used window switchers, i.e. Alt-Tab. It only
    /// updates after some debounce time so that quick window switching doesn't mark intermediate
    /// windows as recently focused.
    ///
    /// The timestamp comes from the monotonic clock.
    pub focus_timestamp: Option<Timestamp>,
}

/// A moment in time.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Timestamp {
    /// Number of whole seconds.
    pub secs: u64,
    /// Fractional part of the timestamp in nanoseconds (10<sup>-9</sup> seconds).
    pub nanos: u32,
}

/// Position- and size-related properties of a [`Window`].
///
/// Optional properties will be unset for some windows, do not rely on them being present. Whether
/// some optional properties are present or absent for certain window types may change across niri
/// releases.
///
/// All sizes and positions are in *logical pixels* unless stated otherwise. Logical sizes may be
/// fractional. For example, at 1.25 monitor scale, a 2-physical-pixel-wide window border is 1.6
/// logical pixels wide.
///
/// This struct contains positions and sizes both for full tiles ([`Self::tile_size`],
/// [`Self::tile_pos_in_workspace_view`]) and the window geometry ([`Self::window_size`],
/// [`Self::window_offset_in_tile`]). For visual displays, use the tile properties, as they
/// correspond to what the user visually considers "window". The window properties on the other
/// hand are mainly useful when you need to know the underlying Wayland window sizes, e.g. for
/// application debugging.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct WindowLayout {
    /// Location of a tiled window within a workspace: (column index, tile index in column).
    ///
    /// The indices are 1-based, i.e. the leftmost column is at index 1 and the topmost tile in a
    /// column is at index 1. This is consistent with [`Action::FocusColumn`] and
    /// [`Action::FocusWindowInColumn`].
    pub pos_in_scrolling_layout: Option<(usize, usize)>,
    /// Size of the tile this window is in, including decorations like borders.
    pub tile_size: (f64, f64),
    /// Size of the window's visual geometry itself.
    ///
    /// Does not include niri decorations like borders.
    ///
    /// Currently, Wayland toplevel windows can only be integer-sized in logical pixels, even
    /// though it doesn't necessarily align to physical pixels.
    pub window_size: (i32, i32),
    /// Tile position within the current view of the workspace.
    ///
    /// This is the same "workspace view" as in gradients' `relative-to` in the niri config.
    pub tile_pos_in_workspace_view: Option<(f64, f64)>,
    /// Location of the window's visual geometry within its tile.
    ///
    /// This includes things like border sizes. For fullscreened fixed-size windows this includes
    /// the distance from the corner of the black backdrop to the corner of the (centered) window
    /// contents.
    pub window_offset_in_tile: (f64, f64),
}

/// Output configuration change result.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum OutputConfigChanged {
    /// The target output was connected and the change was applied.
    Applied,
    /// The target output was not found, the change will be applied when it is connected.
    OutputWasMissing,
}

/// A workspace.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Workspace {
    /// Unique id of this workspace.
    ///
    /// This id remains constant regardless of the workspace moving around and across monitors.
    ///
    /// Do not assume that workspace ids will always increase without wrapping, or start at 1. That
    /// is an implementation detail subject to change. For example, ids may change to be randomly
    /// generated for each new workspace.
    pub id: u64,
    /// Index of the workspace on its monitor.
    ///
    /// This is the same index you can use for requests like `niri msg action focus-workspace`.
    ///
    /// This index *will change* as you move and re-order workspace. It is merely the workspace's
    /// current position on its monitor. Workspaces on different monitors can have the same index.
    ///
    /// If you need a unique workspace id that doesn't change, see [`Self::id`].
    pub idx: u8,
    /// Optional name of the workspace.
    pub name: Option<String>,
    /// Name of the output that the workspace is on.
    ///
    /// Can be `None` if no outputs are currently connected.
    pub output: Option<String>,
    /// Whether the workspace currently has an urgent window in its output.
    pub is_urgent: bool,
    /// Whether the workspace is currently active on its output.
    ///
    /// Every output has one active workspace, the one that is currently visible on that output.
    pub is_active: bool,
    /// Whether the workspace is currently focused.
    ///
    /// There's only one focused workspace across all outputs.
    pub is_focused: bool,
    /// Id of the active window on this workspace, if any.
    pub active_window_id: Option<u64>,
}

/// Configured keyboard layouts.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct KeyboardLayouts {
    /// XKB names of the configured layouts.
    pub names: Vec<String>,
    /// Index of the currently active layout in `names`.
    pub current_idx: u8,
}

/// A layer-shell layer.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Layer {
    /// The background layer.
    Background,
    /// The bottom layer.
    Bottom,
    /// The top layer.
    Top,
    /// The overlay layer.
    Overlay,
}

/// Keyboard interactivity modes for a layer-shell surface.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum LayerSurfaceKeyboardInteractivity {
    /// Surface cannot receive keyboard focus.
    None,
    /// Surface receives keyboard focus whenever possible.
    Exclusive,
    /// Surface receives keyboard focus on demand, e.g. when clicked.
    OnDemand,
}

/// A layer-shell surface.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct LayerSurface {
    /// Namespace provided by the layer-shell client.
    pub namespace: String,
    /// Name of the output the surface is on.
    pub output: String,
    /// Layer that the surface is on.
    pub layer: Layer,
    /// The surface's keyboard interactivity mode.
    pub keyboard_interactivity: LayerSurfaceKeyboardInteractivity,
}

/// A screencast.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct Cast {
    /// Stream ID of the screencast that uniquely identifies it.
    pub stream_id: u64,
    /// Session ID of the screencast.
    ///
    /// A session can have multiple screencast streams. Then multiple `Cast`s will have the same
    /// `session_id`. Though, usually there's only one stream per session.
    ///
    /// Do not confuse `session_id` with [`stream_id`](Self::stream_id).
    pub session_id: u64,
    /// Kind of this screencast.
    pub kind: CastKind,
    /// Target being captured.
    pub target: CastTarget,
    /// Whether this is a Dynamic Cast Target screencast.
    ///
    /// Meaning that actions like `SetDynamicCastWindow` will act on this screencast.
    ///
    /// Keep in mind that the target can change even if this is `false`.
    pub is_dynamic_target: bool,
    /// Whether the cast is currently streaming frames.
    ///
    /// This can be `false` for example when switching away to a different scene in OBS, which
    /// pauses the stream.
    pub is_active: bool,
    /// Process ID of the screencast consumer, if known.
    ///
    /// Currently, only wlr-screencopy screencasts can have a pid.
    pub pid: Option<i32>,
    /// PipeWire node ID of the screencast stream.
    ///
    /// This is `None` for wlr-screencopy casts, and also for PipeWire casts before the node is
    /// created (when the cast is just starting up).
    pub pw_node_id: Option<u32>,
}

/// Kind of screencast.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum CastKind {
    /// PipeWire screencast, typically via xdg-desktop-portal-gnome.
    PipeWire,
    /// wlr-screencopy protocol screencast.
    ///
    /// Tools like wf-recorder, and the xdg-desktop-portal-wlr portal.
    ///
    /// Only wlr-screencopy with damage tracking is reported here. Screencopy without damage is
    /// treated as a regular screenshot and not reported as a screencast.
    WlrScreencopy,
}

/// Target of a screencast.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum CastTarget {
    /// The target is not yet set, or was cleared.
    Nothing {},
    /// Casting an output.
    Output {
        /// Name of the screencasted output.
        name: String,
    },
    /// Casting a window.
    Window {
        /// ID of the screencasted window.
        id: u64,
    },
}

/// A compositor event.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub enum Event {
    /// The workspace configuration has changed.
    WorkspacesChanged {
        /// The new workspace configuration.
        ///
        /// This configuration completely replaces the previous configuration. I.e. if any
        /// workspaces are missing from here, then they were deleted.
        workspaces: Vec<Workspace>,
    },
    /// The workspace urgency changed.
    WorkspaceUrgencyChanged {
        /// Id of the workspace.
        id: u64,
        /// Whether this workspace has an urgent window.
        urgent: bool,
    },
    /// A workspace was activated on an output.
    ///
    /// This doesn't always mean the workspace became focused, just that it's now the active
    /// workspace on its output. All other workspaces on the same output become inactive.
    WorkspaceActivated {
        /// Id of the newly active workspace.
        id: u64,
        /// Whether this workspace also became focused.
        ///
        /// If `true`, this is now the single focused workspace. All other workspaces are no longer
        /// focused, but they may remain active on their respective outputs.
        focused: bool,
    },
    /// An active window changed on a workspace.
    WorkspaceActiveWindowChanged {
        /// Id of the workspace on which the active window changed.
        workspace_id: u64,
        /// Id of the new active window, if any.
        active_window_id: Option<u64>,
    },
    /// The window configuration has changed.
    WindowsChanged {
        /// The new window configuration.
        ///
        /// This configuration completely replaces the previous configuration. I.e. if any windows
        /// are missing from here, then they were closed.
        windows: Vec<Window>,
    },
    /// A new toplevel window was opened, or an existing toplevel window changed.
    WindowOpenedOrChanged {
        /// The new or updated window.
        ///
        /// If the window is focused, all other windows are no longer focused.
        window: Window,
    },
    /// A toplevel window was closed.
    WindowClosed {
        /// Id of the removed window.
        id: u64,
    },
    /// Window focus changed.
    ///
    /// All other windows are no longer focused.
    WindowFocusChanged {
        /// Id of the newly focused window, or `None` if no window is now focused.
        id: Option<u64>,
    },
    /// Window focus timestamp changed.
    ///
    /// This event is separate from [`Event::WindowFocusChanged`] because the focus timestamp only
    /// updates after some debounce time so that quick window switching doesn't mark intermediate
    /// windows as recently focused.
    WindowFocusTimestampChanged {
        /// Id of the window.
        id: u64,
        /// The new focus timestamp.
        focus_timestamp: Option<Timestamp>,
    },
    /// Window urgency changed.
    WindowUrgencyChanged {
        /// Id of the window.
        id: u64,
        /// The new urgency state of the window.
        urgent: bool,
    },
    /// The layout of one or more windows has changed.
    WindowLayoutsChanged {
        /// Pairs consisting of a window id and new layout information for the window.
        changes: Vec<(u64, WindowLayout)>,
    },
    /// The configured keyboard layouts have changed.
    KeyboardLayoutsChanged {
        /// The new keyboard layout configuration.
        keyboard_layouts: KeyboardLayouts,
    },
    /// The keyboard layout switched.
    KeyboardLayoutSwitched {
        /// Index of the newly active layout.
        idx: u8,
    },
    /// The overview was opened or closed.
    OverviewOpenedOrClosed {
        /// The new state of the overview.
        is_open: bool,
    },
    /// The configuration was reloaded.
    ///
    /// You will always receive this event when connecting to the event stream, indicating the last
    /// config load attempt.
    ConfigLoaded {
        /// Whether the loading failed.
        ///
        /// For example, the config file couldn't be parsed.
        failed: bool,
    },
    /// A screenshot was captured.
    ScreenshotCaptured {
        /// The file path where the screenshot was saved, if it was written to disk.
        ///
        /// If `None`, the screenshot was either only copied to the clipboard, or the path couldn't
        /// be converted to a `String` (e.g. contained invalid UTF-8 bytes).
        path: Option<String>,
    },
    /// The screencasts have changed.
    CastsChanged {
        /// The new screencast information.
        ///
        /// This configuration completely replaces the previous configuration. I.e. if any casts
        /// are missing from here, then they were stopped.
        casts: Vec<Cast>,
    },
    /// A screencast started, or an existing cast changed.
    CastStartedOrChanged {
        /// The cast that started or changed.
        cast: Cast,
    },
    /// A screencast stopped.
    CastStopped {
        /// Stream ID of the stopped screencast.
        stream_id: u64,
    },
}

impl From<Duration> for Timestamp {
    fn from(value: Duration) -> Self {
        Timestamp {
            secs: value.as_secs(),
            nanos: value.subsec_nanos(),
        }
    }
}

impl From<Timestamp> for Duration {
    fn from(value: Timestamp) -> Self {
        Duration::new(value.secs, value.nanos)
    }
}

impl FromStr for WorkspaceReferenceArg {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let reference = if let Ok(index) = s.parse::<i32>() {
            if let Ok(idx) = u8::try_from(index) {
                Self::Index(idx)
            } else {
                return Err("workspace index must be between 0 and 255");
            }
        } else {
            Self::Name(s.to_string())
        };

        Ok(reference)
    }
}

impl FromStr for SizeChange {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('%') {
            Some((value, empty)) => {
                if !empty.is_empty() {
                    return Err("trailing characters after '%' are not allowed");
                }

                match value.bytes().next() {
                    Some(b'-' | b'+') => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::AdjustProportion(value))
                    }
                    Some(_) => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::SetProportion(value))
                    }
                    None => Err("value is missing"),
                }
            }
            None => {
                let value = s;
                match value.bytes().next() {
                    Some(b'-' | b'+') => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::AdjustFixed(value))
                    }
                    Some(_) => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::SetFixed(value))
                    }
                    None => Err("value is missing"),
                }
            }
        }
    }
}

impl FromStr for PositionChange {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('%') {
            Some((value, empty)) => {
                if !empty.is_empty() {
                    return Err("trailing characters after '%' are not allowed");
                }

                match value.bytes().next() {
                    Some(b'-' | b'+') => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::AdjustProportion(value))
                    }
                    Some(_) => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::SetProportion(value))
                    }
                    None => Err("value is missing"),
                }
            }
            None => {
                let value = s;
                match value.bytes().next() {
                    Some(b'-' | b'+') => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::AdjustFixed(value))
                    }
                    Some(_) => {
                        let value = value.parse().map_err(|_| "error parsing value")?;
                        Ok(Self::SetFixed(value))
                    }
                    None => Err("value is missing"),
                }
            }
        }
    }
}

impl FromStr for LayoutSwitchTarget {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "next" => Ok(Self::Next),
            "prev" => Ok(Self::Prev),
            other => match other.parse() {
                Ok(layout) => Ok(Self::Index(layout)),
                _ => Err(r#"invalid layout action, can be "next", "prev" or a layout index"#),
            },
        }
    }
}

impl FromStr for ColumnDisplay {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "normal" => Ok(Self::Normal),
            "tabbed" => Ok(Self::Tabbed),
            _ => Err(r#"invalid column display, can be "normal" or "tabbed""#),
        }
    }
}

impl FromStr for Transform {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "normal" => Ok(Self::Normal),
            "90" => Ok(Self::_90),
            "180" => Ok(Self::_180),
            "270" => Ok(Self::_270),
            "flipped" => Ok(Self::Flipped),
            "flipped-90" => Ok(Self::Flipped90),
            "flipped-180" => Ok(Self::Flipped180),
            "flipped-270" => Ok(Self::Flipped270),
            _ => Err(concat!(
                r#"invalid transform, can be "90", "180", "270", "#,
                r#""flipped", "flipped-90", "flipped-180" or "flipped-270""#
            )),
        }
    }
}

impl FromStr for Layer {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "background" => Ok(Self::Background),
            "bottom" => Ok(Self::Bottom),
            "top" => Ok(Self::Top),
            "overlay" => Ok(Self::Overlay),
            _ => Err("invalid layer, can be \"background\", \"bottom\", \"top\" or \"overlay\""),
        }
    }
}

impl FromStr for ModeToSet {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Automatic);
        }

        let mode = s.parse()?;
        Ok(Self::Specific(mode))
    }
}

impl FromStr for ConfiguredMode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some((width, rest)) = s.split_once('x') else {
            return Err("no 'x' separator found");
        };

        let (height, refresh) = match rest.split_once('@') {
            Some((height, refresh)) => (height, Some(refresh)),
            None => (rest, None),
        };

        let width = width.parse().map_err(|_| "error parsing width")?;
        let height = height.parse().map_err(|_| "error parsing height")?;
        let refresh = refresh
            .map(str::parse)
            .transpose()
            .map_err(|_| "error parsing refresh rate")?;

        Ok(Self {
            width,
            height,
            refresh,
        })
    }
}

impl FromStr for HSyncPolarity {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+hsync" => Ok(Self::PHSync),
            "-hsync" => Ok(Self::NHSync),
            _ => Err(r#"invalid horizontal sync polarity, can be "+hsync" or "-hsync"#),
        }
    }
}

impl FromStr for VSyncPolarity {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+vsync" => Ok(Self::PVSync),
            "-vsync" => Ok(Self::NVSync),
            _ => Err(r#"invalid vertical sync polarity, can be "+vsync" or "-vsync"#),
        }
    }
}

impl FromStr for ScaleToSet {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Automatic);
        }

        let scale = s.parse().map_err(|_| "error parsing scale")?;
        Ok(Self::Specific(scale))
    }
}

macro_rules! ensure {
    ($cond:expr, $fmt:literal $($arg:tt)* ) => {
        if !$cond {
            return Err(format!($fmt $($arg)*));
        }
    };
}

impl OutputAction {
    /// Validates some required constraints on the modeline and custom mode.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            OutputAction::Modeline {
                hdisplay,
                hsync_start,
                hsync_end,
                htotal,
                vdisplay,
                vsync_start,
                vsync_end,
                vtotal,
                ..
            } => {
                ensure!(
                    hdisplay < hsync_start,
                    "hdisplay {} must be < hsync_start {}",
                    hdisplay,
                    hsync_start
                );
                ensure!(
                    hsync_start < hsync_end,
                    "hsync_start {} must be < hsync_end {}",
                    hsync_start,
                    hsync_end
                );
                ensure!(
                    hsync_end < htotal,
                    "hsync_end {} must be < htotal {}",
                    hsync_end,
                    htotal
                );
                ensure!(0 < *htotal, "htotal {} must be > 0", htotal);
                ensure!(
                    vdisplay < vsync_start,
                    "vdisplay {} must be < vsync_start {}",
                    vdisplay,
                    vsync_start
                );
                ensure!(
                    vsync_start < vsync_end,
                    "vsync_start {} must be < vsync_end {}",
                    vsync_start,
                    vsync_end
                );
                ensure!(
                    vsync_end < vtotal,
                    "vsync_end {} must be < vtotal {}",
                    vsync_end,
                    vtotal
                );
                ensure!(0 < *vtotal, "vtotal {} must be > 0", vtotal);
                Ok(())
            }
            OutputAction::CustomMode {
                mode: ConfiguredMode { refresh, .. },
            } => {
                if refresh.is_none() {
                    return Err("refresh rate is required for custom modes".to_string());
                }
                if let Some(refresh) = refresh {
                    if *refresh <= 0. {
                        return Err(format!("custom mode refresh rate {refresh} must be > 0"));
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_change() {
        assert_eq!(
            "10".parse::<SizeChange>().unwrap(),
            SizeChange::SetFixed(10),
        );
        assert_eq!(
            "+10".parse::<SizeChange>().unwrap(),
            SizeChange::AdjustFixed(10),
        );
        assert_eq!(
            "-10".parse::<SizeChange>().unwrap(),
            SizeChange::AdjustFixed(-10),
        );
        assert_eq!(
            "10%".parse::<SizeChange>().unwrap(),
            SizeChange::SetProportion(10.),
        );
        assert_eq!(
            "+10%".parse::<SizeChange>().unwrap(),
            SizeChange::AdjustProportion(10.),
        );
        assert_eq!(
            "-10%".parse::<SizeChange>().unwrap(),
            SizeChange::AdjustProportion(-10.),
        );

        assert!("-".parse::<SizeChange>().is_err());
        assert!("10% ".parse::<SizeChange>().is_err());
    }

    #[test]
    fn parse_position_change() {
        assert_eq!(
            "10".parse::<PositionChange>().unwrap(),
            PositionChange::SetFixed(10.),
        );
        assert_eq!(
            "+10".parse::<PositionChange>().unwrap(),
            PositionChange::AdjustFixed(10.),
        );
        assert_eq!(
            "-10".parse::<PositionChange>().unwrap(),
            PositionChange::AdjustFixed(-10.),
        );

        assert_eq!(
            "10%".parse::<PositionChange>().unwrap(),
            PositionChange::SetProportion(10.)
        );
        assert_eq!(
            "+10%".parse::<PositionChange>().unwrap(),
            PositionChange::AdjustProportion(10.)
        );
        assert_eq!(
            "-10%".parse::<PositionChange>().unwrap(),
            PositionChange::AdjustProportion(-10.)
        );
        assert!("-".parse::<PositionChange>().is_err());
        assert!("10% ".parse::<PositionChange>().is_err());
    }

    /// A `prism-gamut-mesh.v1` document — exactly the shape
    /// `prism-tune calibrate-lut3d` writes, including the redundant
    /// per-patch `face` label — round-trips through `from_json_reader`:
    /// schema accepted, `face`/`schema` ignored, snake_case status mapped.
    #[test]
    fn gamut_mesh_parses_v1_sidecar() {
        let json = r#"{
            "schema": "prism-gamut-mesh.v1",
            "white_xyz": [193.0, 203.0, 221.0],
            "cmd_axis_max_nits": [38.9, 113.9, 15.7],
            "vertices": [
                {"code_value": [0.0, 0.0, 0.0], "cmd_nits": [0.1, 0.1, 0.1],
                 "xyz": [0.2, 0.18, 0.34], "lab": [1.6, 0.0, 0.0], "trustworthy": true},
                {"code_value": [1.0, 0.0, 0.0], "cmd_nits": [38.9, 0.0, 0.0],
                 "xyz": [40.0, 20.0, 2.0], "lab": [51.0, 60.0, 40.0], "trustworthy": false}
            ],
            "patches": [
                {"face": "R=1", "axis": 0, "value": 1.0, "corners": [0, 1, 1, 0], "status": "folded"},
                {"face": "B=0", "axis": 2, "value": 0.0, "corners": [0, 1, 0, 1], "status": "max_depth"}
            ]
        }"#;
        let mesh = GamutMesh::from_json_reader(json.as_bytes()).expect("parse v1 sidecar");
        assert_eq!(mesh.white_xyz, [193.0, 203.0, 221.0]);
        assert_eq!(mesh.vertices.len(), 2);
        assert!(mesh.vertices[0].trustworthy && !mesh.vertices[1].trustworthy);
        assert_eq!(mesh.vertices[1].xyz, [40.0, 20.0, 2.0]);
        assert_eq!(mesh.patches.len(), 2);
        assert_eq!(mesh.patches[0].status, GamutPatchStatus::Folded);
        assert_eq!(mesh.patches[1].status, GamutPatchStatus::MaxDepth);
        assert_eq!(mesh.patches[0].corners, [0, 1, 1, 0]);
    }

    /// A wrong / future schema tag is rejected rather than silently
    /// misinterpreted.
    #[test]
    fn gamut_mesh_rejects_unknown_schema() {
        let json = r#"{"schema": "prism-gamut-mesh.v2", "white_xyz": [0,0,0],
            "cmd_axis_max_nits": [0,0,0], "vertices": [], "patches": []}"#;
        let err = GamutMesh::from_json_reader(json.as_bytes()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
