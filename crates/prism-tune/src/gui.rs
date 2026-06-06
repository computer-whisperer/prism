//! `prism-tune gui` — a damascene control panel for prism's per-output
//! color pipeline.
//!
//! First cut: the IPC surface only. It lists the connected outputs,
//! shows each one's live [`ColorState`], and lets you apply the runtime
//! color overrides (`SdrReferenceNits` / `ResponseCurve` /
//! `PanelPeakNits` / `AdvertisedPeakNits` / `ResetColor`) and see the
//! result reflected back.
//!
//! prism IPC is a fast one-shot socket round-trip, so every query and
//! every apply runs synchronously inside `on_event` — no worker thread.
//! The closed-loop calibration flows (which drive the colorimeter for
//! minutes) are a later increment and will need a background worker +
//! progress streaming; this cut deliberately stays on the cheap path to
//! prove the damascene-on-prism integration end to end.
//!
//! [`ColorState`]: prism_ipc::ColorState

use std::fs::File;
use std::os::unix::fs::FileExt;

use anyhow::{bail, Context, Result};
use damascene_core::prelude::*;
use damascene_core::scene::{PointShape, PointStyle, SceneSpec, SizeMode};
use prism_ipc::socket::Socket;
use prism_ipc::{
    ColorState, FrameFormat, FrameMeta, GamutMesh, Output, OutputAction, Request, Response,
    ResponseCurveState,
};

use crate::color3d::{self, GamutScene, GamutSpace, RefSet, REF_GAMUTS};
use crate::common::{send_action, srgb_oetf};

/// Cloud-sample decimation cap (per axis). The cloud is deduped in Lab
/// afterwards, so this only bounds the pre-dedup work, not the final
/// point count.
const CLOUD_MAX_SIDE: usize = 256;
/// Preview-image decimation cap (per axis) — keeps the preview texture
/// light while staying crisp enough to read.
const PREVIEW_MAX_SIDE: usize = 960;
/// Viewport width (logical px) at which the detail area switches from
/// the tabbed single-pane layout to the controls-rail + visualization
/// split. Below this there isn't room for the sidebar, a usable
/// controls column, and a chart side by side.
const WIDE_BREAKPOINT: f32 = 1280.0;
/// Width of the controls rail in the wide layout — enough for an R/G/B
/// triple of numeric inputs (each with its −/+ spinner buttons and a
/// 4-decimal value) inside a card without truncating.
const CONTROLS_PANE_WIDTH: f32 = 560.0;

/// Which detail pane is visible in the narrow (tabbed) layout. The
/// wide layout shows all three at once and ignores this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pane {
    Controls,
    Preview,
    Gamut,
}

impl Pane {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "controls" => Some(Pane::Controls),
            "preview" => Some(Pane::Preview),
            "gamut" => Some(Pane::Gamut),
            _ => None,
        }
    }
}

impl std::fmt::Display for Pane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Pane::Controls => "controls",
            Pane::Preview => "preview",
            Pane::Gamut => "gamut",
        })
    }
}

/// Query every connected output, sorted by connector name.
fn query_outputs() -> Result<Vec<Output>> {
    let mut socket = Socket::connect()
        .context("connect to PRISM_SOCKET (is prism running, and are you in its env?)")?;
    match socket
        .send(Request::Outputs)
        .context("send Outputs request")?
    {
        Ok(Response::Outputs(map)) => {
            let mut outputs: Vec<Output> = map.into_values().collect();
            outputs.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(outputs)
        }
        Ok(other) => bail!("unexpected reply to Outputs: {other:?}"),
        Err(e) => bail!("prism returned an error: {e}"),
    }
}

/// Fetch the measured gamut-surface mesh the compositor has configured
/// for `output` (KDL `color.gamut`). `Ok(None)` ⇒ none configured.
fn query_gamut_mesh(output: &str) -> Result<Option<GamutMesh>> {
    let mut socket = Socket::connect()
        .context("connect to PRISM_SOCKET (is prism running, and are you in its env?)")?;
    match socket
        .send(Request::GamutMesh {
            output: output.to_string(),
        })
        .context("send GamutMesh request")?
    {
        Ok(Response::GamutMesh(mesh)) => Ok(mesh),
        Ok(other) => bail!("unexpected reply to GamutMesh: {other:?}"),
        Err(e) => bail!("prism returned an error: {e}"),
    }
}

fn nits_opts() -> NumericInputOpts<'static> {
    NumericInputOpts::default()
        .min(1.0)
        .max(10_000.0)
        .step(1.0)
        .decimals(1)
}

fn curve_opts() -> NumericInputOpts<'static> {
    NumericInputOpts::default().min(0.0).step(0.01).decimals(4)
}

fn parse_field(s: &str, what: &str) -> Result<f64> {
    s.trim()
        .parse::<f64>()
        .with_context(|| format!("{what}: '{s}' is not a number"))
}

/// Editable color fields, held as `String`s (the controlled-input
/// contract — see [`numeric_input`]). Populated from the selected
/// output's [`ColorState`] each time the selection changes.
#[derive(Default)]
struct Fields {
    sdr_nits: String,
    gain: [String; 3],
    gamma: [String; 3],
    peak: [String; 3],
    /// Color-management advertised mastering peak (cd/m²). Empty for
    /// SDR outputs (no mastering metadata is advertised).
    advertised_peak: String,
}

impl Fields {
    fn from_color(color: &ColorState) -> Self {
        let (gain, gamma) = match color.response_curve {
            Some(rc) => (rc.gain, rc.gamma),
            None => ([1.0; 3], [1.0; 3]),
        };
        Fields {
            sdr_nits: format!("{:.1}", color.sdr_reference_nits),
            gain: gain.map(|v| format!("{v:.4}")),
            gamma: gamma.map(|v| format!("{v:.4}")),
            peak: color.panel_peak_nits.map(|v| format!("{v:.1}")),
            advertised_peak: color
                .advertised_peak_nits
                .map(|v| format!("{v:.1}"))
                .unwrap_or_default(),
        }
    }
}

struct TuneGui {
    outputs: Vec<Output>,
    /// Index into `outputs`, or `None` when the list is empty.
    selected: Option<usize>,
    fields: Fields,
    /// Last action / error message, shown as a banner.
    status: String,
    /// Tonemapped preview of the most recently fetched intermediate
    /// frame (decimated, BT.2020→sRGB). `None` until the first fetch.
    preview: Option<Image>,
    /// Raw BT.2020 absolute-nits samples from the last fetch, retained so
    /// the gamut view can be rebuilt when the coordinate space toggles
    /// without re-capturing.
    frame_samples: Option<Vec<[f32; 3]>>,
    /// 3D gamut point cloud + reference cages, built from `frame_samples`
    /// in the current `gamut_space`.
    gamut: Option<GamutScene>,
    /// Which coordinate space the gamut view plots in.
    gamut_space: GamutSpace,
    /// Which reference gamut cages are drawn, parallel to
    /// [`REF_GAMUTS`](crate::color3d::REF_GAMUTS).
    enabled_gamuts: RefSet,
    /// The measured gamut-surface mesh pulled from the compositor for the
    /// selected output (`color.gamut` config), retained so the shell can
    /// be rebuilt on a space toggle without re-fetching. `None` until
    /// fetched / when the output has none configured.
    gamut_mesh: Option<GamutMesh>,
    /// Whether the measured-gamut lattice shell is drawn.
    show_shell: bool,
    /// Active detail tab in the narrow layout.
    pane: Pane,
    selection: Selection,
}

impl TuneGui {
    fn new() -> Self {
        let mut gui = TuneGui {
            outputs: Vec::new(),
            selected: None,
            fields: Fields::default(),
            status: String::new(),
            preview: None,
            frame_samples: None,
            gamut: None,
            gamut_space: GamutSpace::Cielab,
            enabled_gamuts: [true; REF_GAMUTS.len()],
            gamut_mesh: None,
            show_shell: false,
            pane: Pane::Controls,
            selection: Selection::default(),
        };
        gui.reload(None);
        gui
    }

    /// Build a GUI seeded with a synthetic HDR output and no IPC. Used
    /// by the `gui-bundle` artifact dump so the panel can be laid out
    /// and linted headlessly — no running prism, no GPU.
    fn mock() -> Self {
        let output = Output {
            name: "DisplayPort-4".to_string(),
            make: "Acme".to_string(),
            model: "HDR Reference 27".to_string(),
            serial: None,
            physical_size: None,
            modes: Vec::new(),
            current_mode: None,
            is_custom_mode: false,
            vrr_supported: false,
            vrr_enabled: false,
            logical: None,
            color: ColorState {
                hdr_active: true,
                panel_peak_nits: [580.0, 600.0, 560.0],
                sdr_reference_nits: 203.0,
                response_curve: Some(ResponseCurveState {
                    gain: [0.45, 0.46, 0.43],
                    gamma: [1.08, 1.07, 1.10],
                }),
                ctm: None,
                advertised_peak_nits: Some(1000.0),
            },
        };
        // Synthetic frame stand-ins so the preview and gamut cards lay
        // out at their real (populated) sizes rather than collapsing to
        // their empty placeholders: a hue-sweep preview image and a
        // matching grid of BT.2020 absolute-nits samples.
        let (pw, ph) = (320u32, 180u32);
        let mut pixels = Vec::with_capacity((pw * ph * 4) as usize);
        let mut samples = Vec::with_capacity((pw * ph) as usize);
        for y in 0..ph {
            for x in 0..pw {
                let (fx, fy) = (x as f32 / pw as f32, y as f32 / ph as f32);
                let nits = 600.0 * (1.0 - fy);
                samples.push([nits * fx, nits * (1.0 - fx), nits * fy]);
                pixels.extend_from_slice(&[
                    (fx * 255.0) as u8,
                    ((1.0 - fx) * 255.0) as u8,
                    (fy * 255.0) as u8,
                    255,
                ]);
            }
        }
        let mut gui = TuneGui {
            outputs: vec![output],
            selected: Some(0),
            fields: Fields::default(),
            status: "Mock state — no prism connection.".to_string(),
            preview: Some(Image::from_rgba8(pw, ph, pixels)),
            frame_samples: Some(samples),
            gamut: None,
            gamut_space: GamutSpace::Cielab,
            enabled_gamuts: [true; REF_GAMUTS.len()],
            gamut_mesh: None,
            show_shell: false,
            pane: Pane::Controls,
            selection: Selection::default(),
        };
        gui.sync_fields();
        gui.rebuild_gamut();
        gui
    }

    /// The currently selected output, if any.
    fn current(&self) -> Option<&Output> {
        self.selected.and_then(|i| self.outputs.get(i))
    }

    /// Re-query the output list. If `keep` is set, try to re-select the
    /// output with that connector name; otherwise keep the current
    /// selection by name, falling back to the first output.
    fn reload(&mut self, keep: Option<String>) {
        let want = keep.or_else(|| self.current().map(|o| o.name.clone()));
        match query_outputs() {
            Ok(outputs) => {
                self.selected = match &want {
                    Some(name) => outputs
                        .iter()
                        .position(|o| &o.name == name)
                        .or(if outputs.is_empty() { None } else { Some(0) }),
                    None => (!outputs.is_empty()).then_some(0),
                };
                self.outputs = outputs;
                self.sync_fields();
                if self.outputs.is_empty() {
                    self.status = "No outputs reported by prism.".into();
                }
            }
            Err(e) => {
                self.status = format!("Query failed: {e:#}");
            }
        }
    }

    /// Re-populate the editable fields from the selected output's color
    /// state.
    fn sync_fields(&mut self) {
        if let Some(output) = self.current() {
            self.fields = Fields::from_color(&output.color);
        } else {
            self.fields = Fields::default();
        }
    }

    /// Send an action against the selected output, then reload so the
    /// shown state reflects what prism actually applied.
    fn apply(&mut self, action: OutputAction, label: &str) {
        let Some(name) = self.current().map(|o| o.name.clone()) else {
            self.status = "No output selected.".into();
            return;
        };
        match send_action(&name, action) {
            Ok(()) => {
                self.status = format!("Applied {label} to {name}.");
                self.reload(Some(name));
            }
            Err(e) => self.status = format!("{label} failed: {e:#}"),
        }
    }

    /// Fetch the selected output's most recent intermediate frame and
    /// store a tonemapped preview. On-demand: one IPC round-trip + a
    /// memfd read, all synchronous (the compositor does a single-frame
    /// readback). White point for the tonemap is the output's effective
    /// SDR reference nits.
    fn fetch_frame(&mut self) {
        let Some(output) = self.current().map(|o| o.name.clone()) else {
            self.status = "No output selected.".into();
            return;
        };
        let white = self
            .current()
            .map(|o| o.color.sdr_reference_nits)
            .unwrap_or(203.0);
        match capture_frame(&output, white) {
            Ok((image, samples, w, h)) => {
                self.preview = Some(image);
                self.frame_samples = Some(samples);
                self.rebuild_gamut();
                let colors = self.gamut.as_ref().map_or(0, |g| g.point_count);
                self.status =
                    format!("Fetched {w}×{h} frame from {output} · {colors} distinct colors.");
            }
            Err(e) => self.status = format!("Fetch failed: {e:#}"),
        }
    }

    /// Rebuild the gamut scene from the retained frame samples and/or the
    /// measured-gamut shell in the current coordinate space. Cheap (dedup
    /// over a decimated grid), so it runs on every space / cage / shell
    /// toggle without re-fetching. Builds a scene whenever there's either a
    /// cloud or a visible shell to show.
    fn rebuild_gamut(&mut self) {
        let shell = self
            .show_shell
            .then_some(self.gamut_mesh.as_ref())
            .flatten();
        let have_cloud = self.frame_samples.is_some();
        self.gamut = (have_cloud || shell.is_some()).then(|| {
            let empty: Vec<[f32; 3]> = Vec::new();
            let samples = self.frame_samples.as_deref().unwrap_or(&empty);
            color3d::build_gamut_scene(samples, self.gamut_space, self.enabled_gamuts, shell)
        });
    }

    /// Pull the selected output's measured gamut-surface mesh from the
    /// compositor (`color.gamut` config) and show it as a lattice shell.
    /// One IPC round-trip; the mesh is retained so space toggles rebuild
    /// without re-fetching.
    fn fetch_gamut_mesh(&mut self) {
        let Some(output) = self.current().map(|o| o.name.clone()) else {
            self.status = "No output selected.".into();
            return;
        };
        match query_gamut_mesh(&output) {
            Ok(Some(mesh)) => {
                let (v, p) = (mesh.vertices.len(), mesh.patches.len());
                self.gamut_mesh = Some(mesh);
                self.show_shell = true;
                self.rebuild_gamut();
                self.status =
                    format!("Loaded measured gamut for {output} · {v} vertices, {p} patches.");
            }
            Ok(None) => {
                self.gamut_mesh = None;
                self.show_shell = false;
                self.rebuild_gamut();
                self.status = format!("No measured gamut (color.gamut) configured for {output}.");
            }
            Err(e) => self.status = format!("Gamut fetch failed: {e:#}"),
        }
    }

    fn apply_sdr(&mut self) {
        match parse_field(&self.fields.sdr_nits, "SDR reference") {
            Ok(nits) => self.apply(OutputAction::SdrReferenceNits { nits }, "SDR reference"),
            Err(e) => self.status = format!("{e:#}"),
        }
    }

    fn apply_response(&mut self) {
        let parsed = (|| {
            Ok::<_, anyhow::Error>([
                parse_field(&self.fields.gain[0], "gain R")?,
                parse_field(&self.fields.gain[1], "gain G")?,
                parse_field(&self.fields.gain[2], "gain B")?,
                parse_field(&self.fields.gamma[0], "gamma R")?,
                parse_field(&self.fields.gamma[1], "gamma G")?,
                parse_field(&self.fields.gamma[2], "gamma B")?,
            ])
        })();
        match parsed {
            Ok([gain_r, gain_g, gain_b, gamma_r, gamma_g, gamma_b]) => self.apply(
                OutputAction::ResponseCurve {
                    gain_r,
                    gain_g,
                    gain_b,
                    gamma_r,
                    gamma_g,
                    gamma_b,
                },
                "response curve",
            ),
            Err(e) => self.status = format!("{e:#}"),
        }
    }

    fn apply_advertised(&mut self) {
        match parse_field(&self.fields.advertised_peak, "advertised peak") {
            Ok(nits) => self.apply(
                OutputAction::AdvertisedPeakNits { nits },
                "advertised peak nits",
            ),
            Err(e) => self.status = format!("{e:#}"),
        }
    }

    fn apply_peak(&mut self) {
        let parsed = (|| {
            Ok::<_, anyhow::Error>([
                parse_field(&self.fields.peak[0], "peak R")?,
                parse_field(&self.fields.peak[1], "peak G")?,
                parse_field(&self.fields.peak[2], "peak B")?,
            ])
        })();
        match parsed {
            Ok([nits_r, nits_g, nits_b]) => self.apply(
                OutputAction::PanelPeakNits {
                    nits_r,
                    nits_g,
                    nits_b,
                },
                "panel peak nits",
            ),
            Err(e) => self.status = format!("{e:#}"),
        }
    }

    /// Forward an event to every editable field; each `apply_event`
    /// gates on its own routed key, so non-matching fields ignore it.
    fn route_inputs(&mut self, event: &UiEvent) {
        let nits = nits_opts();
        let curve = curve_opts();
        numeric_input::apply_event(
            &mut self.fields.sdr_nits,
            &mut self.selection,
            "sdr",
            &nits,
            event,
        );
        numeric_input::apply_event(
            &mut self.fields.advertised_peak,
            &mut self.selection,
            "advertised",
            &nits,
            event,
        );
        let gain_keys = ["gain_r", "gain_g", "gain_b"];
        let gamma_keys = ["gamma_r", "gamma_g", "gamma_b"];
        let peak_keys = ["peak_r", "peak_g", "peak_b"];
        for i in 0..3 {
            numeric_input::apply_event(
                &mut self.fields.gain[i],
                &mut self.selection,
                gain_keys[i],
                &curve,
                event,
            );
            numeric_input::apply_event(
                &mut self.fields.gamma[i],
                &mut self.selection,
                gamma_keys[i],
                &curve,
                event,
            );
            numeric_input::apply_event(
                &mut self.fields.peak[i],
                &mut self.selection,
                peak_keys[i],
                &nits,
                event,
            );
        }
    }
}

/// A titled card whose content column stretches: the card takes the
/// height its parent assigns (the caller sets a `Fill` weight on the
/// returned El) and the body's last child absorbs the slack. Used for
/// the chart cards so the preview / 3D plot grow with the window
/// instead of sitting at a fixed height inside a scroll column.
fn fill_card<I, E>(title: &str, body: I) -> El
where
    I: IntoIterator<Item = E>,
    E: Into<El>,
{
    card([
        card_header([card_title(title)]),
        card_content([body_column(body).height(Size::Fill(1.0))]).height(Size::Fill(1.0)),
    ])
}

/// A card-body column with an explicit gap: `card_content` stacks its
/// children with no gap at all, which butts focusable rows against
/// each other — their expanded hit targets overlap and each row's
/// focus ring is painted over by the next (both lint findings).
fn body_column<I, E>(body: I) -> El
where
    I: IntoIterator<Item = E>,
    E: Into<El>,
{
    column(body.into_iter().map(Into::into).collect::<Vec<_>>())
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
}

/// [`titled_card`] with a gapped body — see [`body_column`].
fn gapped_card<I, E>(title: &str, body: I) -> El
where
    I: IntoIterator<Item = E>,
    E: Into<El>,
{
    titled_card(title, [body_column(body)])
}

/// Compact header for the selected output: connector name + status
/// badges on one line, make/model under it. The badges carry what the
/// old read-only state card reported that isn't already visible in an
/// editable field (mode, and whether a CTM / response correction is
/// live).
fn output_header(output: &Output) -> El {
    let color = &output.color;
    let mut title_row: Vec<El> = vec![h2(output.name.clone())];
    title_row.push(if color.hdr_active {
        badge("HDR").success()
    } else {
        badge("SDR").muted()
    });
    if color.response_curve.is_some() {
        title_row.push(badge("response curve").muted());
    }
    if color.ctm.is_some() {
        title_row.push(badge("CTM").muted());
    }
    column([
        row(title_row).gap(tokens::SPACE_2).align(Align::Center),
        text(format!("{} — {}", output.make, output.model))
            .muted()
            .small(),
    ])
    .gap(tokens::SPACE_1)
}

impl TuneGui {
    /// Sidebar: output picker + refresh.
    fn sidebar(&self) -> El {
        let mut picker: Vec<El> = vec![text("Outputs").bold()];
        for (i, output) in self.outputs.iter().enumerate() {
            let mut btn = button(output.name.clone()).key(format!("out:{}", output.name));
            btn = if Some(i) == self.selected {
                btn.primary()
            } else {
                btn.ghost()
            };
            picker.push(btn.width(Size::Fill(1.0)));
        }
        picker.push(spacer());
        picker.push(
            button("Refresh")
                .key("refresh")
                .secondary()
                .width(Size::Fill(1.0)),
        );
        column(picker)
            .gap(tokens::SPACE_2)
            .padding(tokens::SPACE_4)
            .width(Size::Fixed(220.0))
    }

    /// Luminance anchors in one card: the SDR reference, plus (HDR
    /// only) the color-management advertised mastering peak — the
    /// mastering_luminance max in the preferred image description,
    /// independent of the panel-facing max-luminance (infoframe +
    /// encode clamp).
    fn luminance_card(&self, hdr: bool) -> El {
        let mut rows = vec![nits_row(
            "SDR reference",
            &self.fields.sdr_nits,
            &self.selection,
            "sdr",
            "apply:sdr",
        )];
        if hdr {
            rows.push(nits_row(
                "Advertised peak",
                &self.fields.advertised_peak,
                &self.selection,
                "advertised",
                "apply:advertised",
            ));
            rows.push(
                text(
                    "Advertised peak is what color-managed clients tone-map \
                     against; separate from the panel's max-luminance.",
                )
                .muted()
                .small()
                .wrap_text(),
            );
        }
        gapped_card("Luminance", rows)
    }

    fn response_card(&self) -> El {
        gapped_card(
            "Response curve",
            [
                triple_row(
                    "Gain",
                    &self.fields.gain,
                    &self.selection,
                    ["gain_r", "gain_g", "gain_b"],
                    curve_opts(),
                ),
                triple_row(
                    "Gamma",
                    &self.fields.gamma,
                    &self.selection,
                    ["gamma_r", "gamma_g", "gamma_b"],
                    curve_opts(),
                ),
                row([button("Apply").key("apply:response").primary()]).justify(Justify::End),
            ],
        )
    }

    fn peak_card(&self) -> El {
        gapped_card(
            "Panel peak nits",
            [
                triple_row(
                    "RGB",
                    &self.fields.peak,
                    &self.selection,
                    ["peak_r", "peak_g", "peak_b"],
                    nits_opts(),
                ),
                row([button("Apply").key("apply:peak").primary()]).justify(Justify::End),
            ],
        )
    }

    /// The editable control cards, in the order they stack in the
    /// controls rail / Controls tab.
    fn control_cards(&self, output: &Output) -> Vec<El> {
        vec![
            self.luminance_card(output.color.hdr_active),
            self.response_card(),
            self.peak_card(),
            row([button("Reset all color overrides")
                .key("apply:reset")
                .destructive()])
            .justify(Justify::End),
        ]
    }

    /// Frame-preview card. The image absorbs whatever height the card
    /// is assigned, so the preview scales with the window.
    fn preview_card(&self) -> El {
        let body: El = match &self.preview {
            Some(img) => image(img.clone())
                .image_fit(ImageFit::Contain)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0)),
            None => column([text("No frame captured yet.").muted().small()])
                .height(Size::Fill(1.0))
                .justify(Justify::Center),
        };
        fill_card(
            "Recent frame · BT.2020 intermediate",
            [
                row([button("Fetch new frame").key("fetch").primary()]).justify(Justify::End),
                body,
            ],
        )
    }

    /// Gamut-cloud card: space + cage + shell toggles, then the 3D
    /// chart filling the remaining height.
    fn gamut_card(&self) -> El {
        let space_toggle = row([
            toggle_button(
                "CIELAB",
                "space:lab",
                self.gamut_space == GamutSpace::Cielab,
            ),
            toggle_button(
                "BT.2020 RGB (nits)",
                "space:rgb",
                self.gamut_space == GamutSpace::Bt2020Rgb,
            ),
        ])
        .gap(tokens::SPACE_2);

        // Per-gamut cage on/off, in REF_GAMUTS order. Each cage is
        // named in-plot at its green primary (see GamutScene).
        let mut cage_btns: Vec<El> = Vec::with_capacity(REF_GAMUTS.len());
        for (i, g) in REF_GAMUTS.iter().enumerate() {
            cage_btns.push(toggle_button(
                g.name,
                &format!("cage:{}", g.key),
                self.enabled_gamuts[i],
            ));
        }
        let cage_toggle = column([
            text("Reference gamuts").small().muted(),
            row(cage_btns).gap(tokens::SPACE_2),
        ])
        .gap(tokens::SPACE_1);

        // Measured-gamut shell: a button to pull the mesh from the
        // compositor, and (once loaded) a show/hide toggle.
        let mut shell_row: Vec<El> =
            vec![button("Load measured gamut").key("fetch:gamut").secondary()];
        if self.gamut_mesh.is_some() {
            shell_row.push(toggle_button("Show shell", "shell", self.show_shell));
        }
        let shell_controls = column([
            text("Measured gamut").small().muted(),
            row(shell_row).gap(tokens::SPACE_2),
        ])
        .gap(tokens::SPACE_1);

        let has_geometry = self.gamut.as_ref().is_some_and(|g| {
            g.point_count > 0
                || g.cage_segments > 0
                || g.shell_segments > 0
                || g.cage_label_count > 0
        });
        let gamut_body: El = match self.gamut.as_ref().filter(|_| has_geometry) {
            Some(g) => {
                // Axes are unclipped: both spaces are absolute, so
                // content brighter than reference white sits above
                // the cage whites instead of being clamped.
                let (tx, ty, tz) = match self.gamut_space {
                    GamutSpace::Cielab => ("a*", "L*", "b*"),
                    GamutSpace::Bt2020Rgb => ("R (nits)", "G (nits)", "B (nits)"),
                };
                // The damascene wgpu backend rejects empty geometry
                // buffers, so add each mark only when it has data —
                // the cloud (no frame yet), the cages (all toggled
                // off), the shell (no mesh), and the labels can each
                // be empty independently.
                let mut scene = SceneSpec::new();
                if g.point_count > 0 {
                    scene = scene.points_styled(
                        g.points.clone(),
                        PointStyle {
                            size: 5.0,
                            shape: PointShape::Circle,
                            size_mode: SizeMode::ScreenSpace,
                        },
                    );
                }
                if g.cage_segments > 0 {
                    scene = scene.lines(g.cages.clone());
                }
                if g.shell_segments > 0 {
                    scene = scene.lines(g.shell.clone());
                }
                if g.cage_label_count > 0 {
                    // A small square marker + persistent name at each
                    // enabled cage's green primary.
                    scene = scene.points_labeled(
                        g.cage_label_geo.clone(),
                        PointStyle {
                            size: 5.0,
                            shape: PointShape::Square,
                            size_mode: SizeMode::ScreenSpace,
                        },
                        g.cage_labels.clone(),
                    );
                }
                scene = scene.axis_titles(tx, ty, tz);
                chart3d(scene)
                    .width(Size::Fill(1.0))
                    .height(Size::Fill(1.0))
            }
            None => column([text("Fetch a frame or load the measured gamut to plot.")
                .muted()
                .small()])
            .height(Size::Fill(1.0))
            .justify(Justify::Center),
        };
        // Floating legend: the toggle cluster paints over the plot's
        // top-left corner on a translucent panel, so the 3D render
        // surface gets the card's whole content area. Painted after
        // the chart ⇒ hit-tested first, so the buttons win clicks;
        // drag / wheel anywhere else still orbits / zooms the scene.
        let legend =
            card([column([space_toggle, cage_toggle, shell_controls]).gap(tokens::SPACE_3)])
                .padding(tokens::SPACE_3)
                .fill(tokens::CARD.with_alpha(0.85))
                .radius(tokens::RADIUS_MD);
        let body = stack([
            gamut_body,
            // Transparent spacer wrapper: inset the legend from the
            // plot's corner (and keep its focus rings unclipped).
            column([legend]).padding(tokens::SPACE_2),
        ])
        .align(Align::Start)
        .justify(Justify::Start)
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0));
        fill_card("Gamut cloud — drag to orbit, wheel to zoom", [body])
    }

    /// Last action / error, as a sunken banner.
    fn status_banner(&self) -> El {
        column([text(self.status.clone()).small().wrap_text()])
            .padding(tokens::SPACE_3)
            .surface_role(SurfaceRole::Sunken)
    }

    /// Wide (≥ [`WIDE_BREAKPOINT`]) detail: a fixed-width controls
    /// rail next to a visualization column with the preview and gamut
    /// plot stacked — everything visible at once, charts sized by the
    /// window.
    fn wide_detail(&self, output: &Output) -> El {
        let mut controls: Vec<El> = Vec::new();
        if !self.status.is_empty() {
            controls.push(self.status_banner());
        }
        controls.push(output_header(output));
        controls.extend(self.control_cards(output));
        let controls_pane = scroll([column(controls)
            .gap(tokens::SPACE_4)
            .padding(tokens::SPACE_5)])
        .width(Size::Fixed(CONTROLS_PANE_WIDTH));
        // The gamut plot gets the larger share — it carries its own
        // toggle rows.
        let viz = column([
            self.preview_card().height(Size::Fill(2.0)),
            self.gamut_card().height(Size::Fill(3.0)),
        ])
        .gap(tokens::SPACE_4)
        .padding(tokens::SPACE_5)
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0));
        row([controls_pane, vertical_separator(), viz])
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }

    /// Narrow detail: header + tab strip, one pane at a time filling
    /// the window — no scrolling past charts to reach the controls.
    fn narrow_detail(&self, output: &Output) -> El {
        let mut children: Vec<El> = Vec::new();
        if !self.status.is_empty() {
            children.push(self.status_banner());
        }
        children.push(output_header(output));
        children.push(tabs_list(
            "pane",
            &self.pane,
            [
                (Pane::Controls, "Controls"),
                (Pane::Preview, "Preview"),
                (Pane::Gamut, "Gamut"),
            ],
        ));
        children.push(match self.pane {
            Pane::Controls => scroll([column(self.control_cards(output))
                .gap(tokens::SPACE_4)
                // Ring-width padding so focus rings at the column's
                // edges aren't clipped by the scroll scissor.
                .padding(tokens::RING_WIDTH)])
            .height(Size::Fill(1.0)),
            Pane::Preview => self.preview_card().height(Size::Fill(1.0)),
            Pane::Gamut => self.gamut_card().height(Size::Fill(1.0)),
        });
        column(children)
            .gap(tokens::SPACE_4)
            .padding(tokens::SPACE_5)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
    }
}

impl App for TuneGui {
    fn build(&self, cx: &BuildCx) -> El {
        // Tabbed below WIDE_BREAKPOINT (and when the host reports no
        // viewport), three-pane split above it.
        let wide = cx.viewport_width().is_some_and(|w| w >= WIDE_BREAKPOINT);
        let detail: El = match self.current() {
            Some(output) if wide => self.wide_detail(output),
            Some(output) => self.narrow_detail(output),
            None => {
                let mut children: Vec<El> = Vec::new();
                if !self.status.is_empty() {
                    children.push(self.status_banner());
                }
                children.push(text("No output selected.").muted());
                column(children)
                    .gap(tokens::SPACE_4)
                    .padding(tokens::SPACE_5)
                    .width(Size::Fill(1.0))
            }
        };
        row([self.sidebar(), vertical_separator(), detail]).height(Size::Fill(1.0))
    }

    fn on_event(&mut self, event: UiEvent, _cx: &EventCx) {
        if tabs::apply_event(&mut self.pane, &event, "pane", Pane::parse) {
            return;
        }
        if let Some(route) = event
            .route()
            .filter(|_| matches!(event.kind, UiEventKind::Click | UiEventKind::Activate))
        {
            match route {
                "refresh" => {
                    self.reload(None);
                    return;
                }
                "fetch" => {
                    self.fetch_frame();
                    return;
                }
                "fetch:gamut" => {
                    self.fetch_gamut_mesh();
                    return;
                }
                "shell" => {
                    self.show_shell = !self.show_shell;
                    self.rebuild_gamut();
                    return;
                }
                "space:lab" => {
                    self.gamut_space = GamutSpace::Cielab;
                    self.rebuild_gamut();
                    return;
                }
                "space:rgb" => {
                    self.gamut_space = GamutSpace::Bt2020Rgb;
                    self.rebuild_gamut();
                    return;
                }
                "apply:sdr" => {
                    self.apply_sdr();
                    return;
                }
                "apply:advertised" => {
                    self.apply_advertised();
                    return;
                }
                "apply:response" => {
                    self.apply_response();
                    return;
                }
                "apply:peak" => {
                    self.apply_peak();
                    return;
                }
                "apply:reset" => {
                    self.apply(OutputAction::ResetColor, "reset");
                    return;
                }
                r => {
                    if let Some(name) = r.strip_prefix("out:") {
                        if let Some(i) = self.outputs.iter().position(|o| o.name == name) {
                            self.selected = Some(i);
                            self.sync_fields();
                            self.status.clear();
                            // The measured gamut is per-output; drop it so a
                            // stale shell doesn't carry over to the new one.
                            self.gamut_mesh = None;
                            self.show_shell = false;
                            self.rebuild_gamut();
                        }
                        return;
                    }
                    if let Some(key) = r.strip_prefix("cage:") {
                        if let Some(i) = REF_GAMUTS.iter().position(|g| g.key == key) {
                            self.enabled_gamuts[i] = !self.enabled_gamuts[i];
                            self.rebuild_gamut();
                        }
                        return;
                    }
                }
            }
        }
        self.route_inputs(&event);
    }

    fn selection(&self) -> Selection {
        self.selection.clone()
    }
}

/// A toggle button — primary when active, ghost otherwise. Used for both
/// the coordinate-space selector and the per-gamut cage toggles.
fn toggle_button(label: &str, key: &str, active: bool) -> El {
    let btn = button(label.to_string()).key(key.to_string());
    if active {
        btn.primary()
    } else {
        btn.ghost()
    }
}

/// A labelled single-value nits input with its own Apply button. The
/// fixed label width keeps stacked rows' inputs aligned.
fn nits_row(label: &str, value: &str, selection: &Selection, key: &str, apply_key: &str) -> El {
    row([
        text(label).muted().width(Size::Fixed(130.0)),
        numeric_input(value, selection, key, nits_opts()).width(Size::Fill(1.0)),
        button("Apply").key(apply_key).primary(),
    ])
    .gap(tokens::SPACE_2)
    .align(Align::Center)
}

/// A labelled group of three numeric inputs (R/G/B-style triples): the
/// label sits above a full-width row of three equal-width fields so each
/// one spreads to a third of the card rather than collapsing to its
/// minimum and crowding the right edge.
fn triple_row(
    label: &str,
    values: &[String; 3],
    selection: &Selection,
    keys: [&str; 3],
    opts: NumericInputOpts<'_>,
) -> El {
    column([
        text(label).small().muted(),
        row([
            numeric_input(&values[0], selection, keys[0], opts).width(Size::Fill(1.0)),
            numeric_input(&values[1], selection, keys[1], opts).width(Size::Fill(1.0)),
            numeric_input(&values[2], selection, keys[2], opts).width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_2)
        .width(Size::Fill(1.0))
        .align(Align::Center),
    ])
    .gap(tokens::SPACE_1)
}

/// Request a frame capture for `output`, receive the memfd, and process
/// it into both a decimated sRGB preview and a coarser set of raw
/// BT.2020-linear samples for the gamut cloud. Returns those plus the
/// captured frame's full pixel dimensions (for the status line).
///
/// Uses a fresh one-shot connection so the `recvmsg` fd path in
/// [`Socket::send_recv_fd`] has no buffered read-ahead to race.
fn capture_frame(output: &str, white_nits: f64) -> Result<(Image, Vec<[f32; 3]>, u32, u32)> {
    let mut socket = Socket::connect()
        .context("connect to PRISM_SOCKET (is prism running, and are you in its env?)")?;
    let (reply, fd) = socket
        .send_recv_fd(Request::CaptureFrame {
            output: output.to_string(),
        })
        .context("send CaptureFrame request")?;
    let meta = match reply {
        Ok(Response::FrameCaptured(meta)) => meta,
        Ok(other) => bail!("unexpected reply to CaptureFrame: {other:?}"),
        Err(e) => bail!("prism returned an error: {e}"),
    };
    let fd = fd.ok_or_else(|| anyhow::anyhow!("server replied FrameCaptured without an fd"))?;
    let (image, samples) = process_frame(File::from(fd), &meta, white_nits)?;
    Ok((image, samples, meta.width, meta.height))
}

/// Read a captured BT.2020 intermediate frame once and derive two views:
/// a tonemapped sRGB preview `Image` (decimated to [`PREVIEW_MAX_SIDE`])
/// and a coarser grid of raw BT.2020-linear samples (to
/// [`CLOUD_MAX_SIDE`]) for the gamut cloud. The preview normalizes by
/// `white_nits` and converts BT.2020→sRGB (clamping out-of-gamut /
/// over-white); the cloud samples stay raw — the gamut builder owns that
/// color math.
fn process_frame(file: File, meta: &FrameMeta, white_nits: f64) -> Result<(Image, Vec<[f32; 3]>)> {
    match meta.format {
        FrameFormat::Rgba32Float => {}
    }
    let mut data = vec![0u8; meta.byte_len as usize];
    // `read_exact_at` is pread(2): reads at an explicit offset, ignoring
    // the file's (shared, SCM_RIGHTS-duplicated) position — so we always
    // start at byte 0 regardless of where the sender left the cursor.
    file.read_exact_at(&mut data, 0)
        .context("read captured frame from memfd")?;

    let w = meta.width as usize;
    let h = meta.height as usize;
    let stride = meta.stride_bytes as usize;
    if w == 0 || h == 0 {
        bail!("captured frame has zero dimension ({w}×{h})");
    }

    let read_rgb = |off: usize| -> [f32; 3] {
        [
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]),
            f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]),
            f32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]),
        ]
    };

    // Preview: tonemap a fine grid into an RGBA8 image.
    let pstep = (w.max(h) / PREVIEW_MAX_SIDE).max(1);
    let scale = if white_nits > 0.0 {
        1.0 / white_nits
    } else {
        1.0
    };
    let mut pixels: Vec<u8> = Vec::new();
    let (mut pw, mut ph) = (0u32, 0u32);
    let mut y = 0;
    while y < h {
        let row = y * stride;
        let mut cols = 0u32;
        let mut x = 0;
        while x < w {
            let rgb = read_rgb(row + x * 16);
            let [r, g, b] = bt2020_to_srgb8(
                rgb[0] as f64 * scale,
                rgb[1] as f64 * scale,
                rgb[2] as f64 * scale,
            );
            pixels.extend_from_slice(&[r, g, b, 255]);
            cols += 1;
            x += pstep;
        }
        pw = cols;
        ph += 1;
        y += pstep;
    }
    let image = Image::from_rgba8(pw, ph, pixels);

    // Cloud: a coarser grid of raw BT.2020-linear samples.
    let cstep = (w.max(h) / CLOUD_MAX_SIDE).max(1);
    let mut samples: Vec<[f32; 3]> = Vec::new();
    let mut y = 0;
    while y < h {
        let row = y * stride;
        let mut x = 0;
        while x < w {
            samples.push(read_rgb(row + x * 16));
            x += cstep;
        }
        y += cstep;
    }

    Ok((image, samples))
}

/// BT.2020 linear (relative to white = 1.0) → 8-bit sRGB. The 3×3 is
/// the standard BT.2020→BT.709 primary conversion; channels clip to
/// [0, 1] before the sRGB OETF.
fn bt2020_to_srgb8(r: f64, g: f64, b: f64) -> [u8; 3] {
    let sr = 1.660_491 * r - 0.587_641 * g - 0.072_850 * b;
    let sg = -0.124_550 * r + 1.132_900 * g - 0.008_349 * b;
    let sb = -0.018_151 * r - 0.100_579 * g + 1.118_730 * b;
    [enc(sr), enc(sg), enc(sb)]
}

fn enc(linear: f64) -> u8 {
    (srgb_oetf(linear.clamp(0.0, 1.0)) * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8
}

pub fn run() -> Result<()> {
    let viewport = Rect::new(0.0, 0.0, 900.0, 760.0);
    damascene_winit_wgpu::run("prism-tune", viewport, TuneGui::new())
        .map_err(|e| anyhow::anyhow!("damascene host error: {e}"))
}

/// Render the panel (with mock state) through damascene's bundle
/// pipeline and write the standard artifact set — `.svg`, `.tree.txt`,
/// `.draw_ops.txt`, `.shader_manifest.txt`, `.lint.txt` — to `dir`, then
/// echo the layout lint report to stderr. Headless: no prism, no GPU.
/// `width`×`height` is the logical-px viewport to lay out at, so the
/// panel can be checked at different window sizes.
pub fn dump_bundle(dir: &std::path::Path, width: f32, height: f32) -> Result<()> {
    let viewport = Rect::new(0.0, 0.0, width, height);
    let app = TuneGui::mock();
    let theme = Theme::default();
    let cx = BuildCx::new(&theme).with_viewport(width, height);
    let mut root = app.build(&cx);
    let bundle = render_bundle(&mut root, viewport);

    let written = write_bundle(&bundle, dir, "prism-tune-gui").context("write bundle artifacts")?;
    eprintln!("Wrote {} artifact(s) to {}:", written.len(), dir.display());
    for path in &written {
        if let Some(name) = path.file_name() {
            eprintln!("  {}", name.to_string_lossy());
        }
    }
    eprintln!("\n--- layout lint ---\n{}", bundle.lint.text());
    Ok(())
}
