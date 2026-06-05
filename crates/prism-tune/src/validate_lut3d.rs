//! `prism-tune validate-lut3d` — software validation of the live color
//! pipeline via the `EncodeDiagnose` 1-pixel IPC.
//!
//! Drives BT.2020 inputs through the compositor's real encode path (LUT +
//! output transfer) and reads back what the compositor emits *into the
//! display* (`scanout_nits`), with no colorimeter in the loop. Two checks:
//!
//!   1. **Neutral-balance drift.** Sweep a neutral ramp (R=G=B) across the
//!      luminance range — past the panel's white peak into the over-gamut
//!      region. The compositor's per-channel output should keep a constant
//!      ratio for a neutral input; a ratio that diverges as brightness rises
//!      means neutral white is being driven off-neutral (a visible tint —
//!      the "bright whites go pink" class of bug). Reports where the drift
//!      crosses a threshold and which channel falls off.
//!
//!   2. **GPU-vs-LUT-file drift** (optional, `--lut`). Compares each
//!      `EncodeDiagnose` result against the `.lut` file's own trilinear
//!      prediction. A mismatch isolates a shader / upload / sampling bug
//!      from the LUT's *content* — if the GPU matches the file but the
//!      ramp still drifts, the LUT itself encodes the tint.
//!
//! This deliberately needs no colorimeter: it validates what the compositor
//! computes and sends, not what the panel emits. A colorimeter-measured
//! mode (ground-truth displayed chromaticity) is a natural follow-up.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use prism_ipc::{OutputAction, Response};
use prism_renderer::{load_lut3d_file, pq_eotf, LoadedLut};

use crate::calibrate_lut3d::{pq_oetf_f64, trilinear_sample_lut};
use crate::common::{query_output_baseline, send_action_for_reply};

#[derive(clap::Args)]
pub struct ValidateLut3dArgs {
    /// Output connector name (e.g. `DisplayPort-4`). Long form, as
    /// `prism-tune msg outputs` lists it.
    pub output: String,

    /// Path to the `.lut` file backing this output. When given, each
    /// GPU `EncodeDiagnose` result is compared against the file's own
    /// trilinear prediction to separate shader/upload drift from LUT
    /// content. Omit to validate the live output alone.
    #[arg(long)]
    pub lut: Option<PathBuf>,

    /// Top of the neutral ramp, cd/m². Defaults to the panel's white
    /// peak (sum of measured per-channel peaks) × 1.5, so the sweep
    /// probes the over-gamut region where bright whites clip.
    #[arg(long)]
    pub max_nits: Option<f64>,

    /// Lowest non-zero ramp point, cd/m².
    #[arg(long, default_value_t = 1.0)]
    pub min_nits: f64,

    /// Number of ramp steps (perceptually / PQ-spaced).
    #[arg(long, default_value_t = 24)]
    pub steps: usize,

    /// Also sweep each primary (R, G, B alone), not just the neutral axis.
    #[arg(long)]
    pub primaries: bool,

    /// Neutral-balance drift threshold to flag, as a fraction (distance
    /// in unit-sum cmd-fraction space from the dim-end reference).
    #[arg(long, default_value_t = 0.02)]
    pub drift_threshold: f64,

    /// Write the full per-sample table to this path as CSV.
    #[arg(long)]
    pub csv: Option<PathBuf>,
}

/// One swept sample: the BT.2020 input and what the compositor emitted.
struct Sample {
    input_nits: f64,
    scanout: [f64; 3],
    /// Unit-sum fractions of `scanout` — the per-channel balance,
    /// independent of overall level.
    frac: [f64; 3],
    /// Distance of `frac` from the dim-end reference fractions.
    drift: f64,
    /// CPU LUT-file prediction (panel-native cmd nits) and the worst
    /// per-channel percentage difference vs the GPU `scanout`, when a
    /// `.lut` file was supplied.
    cpu: Option<([f64; 3], f64)>,
}

pub fn run(args: ValidateLut3dArgs) -> Result<()> {
    let baseline = query_output_baseline(&args.output)
        .with_context(|| format!("query baseline for {}", args.output))?;

    let white_peak: f64 = baseline.initial_panel_peak_nits.iter().sum();
    let max_nits = args
        .max_nits
        .unwrap_or(white_peak * 1.5)
        .max(args.min_nits * 2.0);

    let lut = match &args.lut {
        Some(path) => {
            Some(load_lut3d_file(path).with_context(|| format!("load LUT {}", path.display()))?)
        }
        None => None,
    };

    println!(
        "validate-lut3d {output}: mode={mode} sdr_ref={sdr:.0} panel_peak_rgb={peak:?} (white≈{white:.1} cd/m²)",
        output = args.output,
        mode = if baseline.hdr_active { "HDR" } else { "SDR" },
        sdr = baseline.sdr_reference_nits,
        peak = baseline.initial_panel_peak_nits,
        white = white_peak,
    );
    if let Some(l) = &lut {
        println!(
            "  LUT reference: cube_edge={} peaks={:?}",
            l.cube_edge, l.peak_nits
        );
    }
    println!(
        "  neutral ramp: {} steps, {:.1}..{:.1} cd/m² (PQ-spaced); drift threshold {:.3}\n",
        args.steps, args.min_nits, max_nits, args.drift_threshold,
    );

    let samples = sweep_neutral(&args, &lut, max_nits)?;
    report_neutral(&samples, args.drift_threshold);

    if args.primaries {
        for (idx, name) in [(0usize, "RED"), (1, "GREEN"), (2, "BLUE")] {
            println!("\n── {name} primary ramp ──");
            let prim = sweep_primary(&args, &lut, idx, max_nits)?;
            report_primary(&prim);
        }
    }

    if let Some(csv_path) = &args.csv {
        write_csv(csv_path, &samples)
            .with_context(|| format!("write CSV {}", csv_path.display()))?;
        println!("\nwrote {}", csv_path.display());
    }

    Ok(())
}

/// PQ-spaced ramp of input nits in `[min, max]`.
fn ramp(min_nits: f64, max_nits: f64, steps: usize) -> Vec<f64> {
    let e_min = pq_oetf_f64(min_nits);
    let e_max = pq_oetf_f64(max_nits);
    (0..steps)
        .map(|i| {
            let t = if steps <= 1 {
                0.0
            } else {
                i as f64 / (steps - 1) as f64
            };
            let e = e_min + (e_max - e_min) * t;
            pq_eotf(e as f32) as f64
        })
        .collect()
}

fn diagnose(output: &str, input: [f64; 3]) -> Result<[f64; 3]> {
    let resp = send_action_for_reply(
        output,
        OutputAction::EncodeDiagnose {
            r: input[0],
            g: input[1],
            b: input[2],
        },
    )?;
    match resp {
        Response::EncodeDiagnose(r) => Ok(r.scanout_nits),
        other => bail!("unexpected reply to EncodeDiagnose: {other:?}"),
    }
}

fn unit_sum(v: [f64; 3]) -> [f64; 3] {
    let s = v[0] + v[1] + v[2];
    if s <= 1e-9 {
        [1.0 / 3.0; 3]
    } else {
        [v[0] / s, v[1] / s, v[2] / s]
    }
}

fn frac_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn cpu_predict(lut: &LoadedLut, input_nits: f64) -> [f64; 3] {
    let coord = [pq_oetf_f64(input_nits); 3];
    let c = trilinear_sample_lut(&lut.entries, lut.cube_edge, coord);
    [c[0] as f64, c[1] as f64, c[2] as f64]
}

fn worst_pct_diff(gpu: [f64; 3], cpu: [f64; 3]) -> f64 {
    (0..3)
        .map(|i| (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-3) * 100.0)
        .fold(0.0_f64, f64::max)
}

fn sweep_neutral(
    args: &ValidateLut3dArgs,
    lut: &Option<LoadedLut>,
    max_nits: f64,
) -> Result<Vec<Sample>> {
    let nits = ramp(args.min_nits, max_nits, args.steps);
    let mut samples = Vec::with_capacity(nits.len());
    // Reference balance taken from the dimmest sample, where the LUT is
    // best-conditioned (calibration verifies the low end).
    let mut reference: Option<[f64; 3]> = None;
    for n in nits {
        let scanout = diagnose(&args.output, [n, n, n])?;
        let frac = unit_sum(scanout);
        let reference = *reference.get_or_insert(frac);
        let drift = frac_dist(frac, reference);
        let cpu = lut.as_ref().map(|l| {
            let p = cpu_predict(l, n);
            (p, worst_pct_diff(scanout, p))
        });
        samples.push(Sample {
            input_nits: n,
            scanout,
            frac,
            drift,
            cpu,
        });
    }
    Ok(samples)
}

fn report_neutral(samples: &[Sample], threshold: f64) {
    let has_cpu = samples.first().is_some_and(|s| s.cpu.is_some());
    if has_cpu {
        println!(
            "{:>9}  {:>27}  {:>21}  {:>7}  {:>9}",
            "input", "scanout R/G/B (cmd)", "balance R:G:B", "drift", "gpu-vs-lut"
        );
    } else {
        println!(
            "{:>9}  {:>27}  {:>21}  {:>7}",
            "input", "scanout R/G/B (cmd)", "balance R:G:B", "drift"
        );
    }
    for s in samples {
        let flag = if s.drift > threshold { " *" } else { "  " };
        let base = format!(
            "{:>9.2}  {:>8.2}{:>8.2}{:>8.2}     {:>6.3}:{:.3}:{:.3}  {:>6.4}{}",
            s.input_nits,
            s.scanout[0],
            s.scanout[1],
            s.scanout[2],
            s.frac[0],
            s.frac[1],
            s.frac[2],
            s.drift,
            flag,
        );
        match s.cpu {
            Some((_, pct)) => println!("{base}  {:>7.2}%", pct),
            None => println!("{base}"),
        }
    }

    // Verdict: first sample crossing the threshold.
    match samples.iter().find(|s| s.drift > threshold) {
        Some(first) => {
            let ref_frac = samples[0].frac;
            // Which channel moved most, and in which direction.
            let (ch, name) = (0..3)
                .map(|i| (i, (first.frac[i] - ref_frac[i]).abs()))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| (i, ["red", "green", "blue"][i]))
                .unwrap();
            let dir = if first.frac[ch] < ref_frac[ch] {
                "falls"
            } else {
                "rises"
            };
            println!(
                "\nVERDICT: neutral balance drifts past {:.3} at {:.1} cd/m² — {} {} \
                 (frac {:.3} vs dim-end {:.3}). Neutral input is being driven off-neutral; \
                 a {} deficit reads as the complementary tint on screen.",
                threshold, first.input_nits, name, dir, first.frac[ch], ref_frac[ch], name,
            );
        }
        None => {
            let max_drift = samples.iter().map(|s| s.drift).fold(0.0_f64, f64::max);
            println!(
                "\nVERDICT: neutral balance holds across the full ramp (max drift {:.4} ≤ {:.3}). \
                 The compositor's output stays neutral for neutral input — any pink originates \
                 upstream (client content / surface decode), not in this pipeline.",
                max_drift, threshold,
            );
        }
    }

    if has_cpu {
        let worst = samples
            .iter()
            .filter_map(|s| s.cpu.map(|(_, p)| (s.input_nits, p)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if let Some((nits, pct)) = worst {
            let verdict = if pct < 2.0 {
                "GPU matches the LUT file — the shader/upload path is faithful; \
                 any drift above is baked into the LUT content"
            } else {
                "GPU diverges from the LUT file — suspect the shader / upload / \
                 sampling path, not the LUT content"
            };
            println!("gpu-vs-lut: worst {pct:.2}% at {nits:.1} cd/m² — {verdict}.");
        }
    }
}

fn sweep_primary(
    args: &ValidateLut3dArgs,
    lut: &Option<LoadedLut>,
    channel: usize,
    max_nits: f64,
) -> Result<Vec<Sample>> {
    let nits = ramp(args.min_nits, max_nits, args.steps);
    let mut out = Vec::with_capacity(nits.len());
    for n in nits {
        let mut input = [0.0; 3];
        input[channel] = n;
        let scanout = diagnose(&args.output, input)?;
        let frac = unit_sum(scanout);
        let cpu = lut.as_ref().map(|l| {
            let coord = [
                pq_oetf_f64(input[0]),
                pq_oetf_f64(input[1]),
                pq_oetf_f64(input[2]),
            ];
            let c = trilinear_sample_lut(&l.entries, l.cube_edge, coord);
            let p = [c[0] as f64, c[1] as f64, c[2] as f64];
            (p, worst_pct_diff(scanout, p))
        });
        out.push(Sample {
            input_nits: n,
            scanout,
            frac,
            drift: 0.0,
            cpu,
        });
    }
    Ok(out)
}

fn report_primary(samples: &[Sample]) {
    for s in samples {
        let cpu = match s.cpu {
            Some((_, pct)) => format!("  gpu-vs-lut {pct:>6.2}%"),
            None => String::new(),
        };
        println!(
            "{:>9.2}  scanout {:>8.2}{:>8.2}{:>8.2}{}",
            s.input_nits, s.scanout[0], s.scanout[1], s.scanout[2], cpu,
        );
    }
}

fn write_csv(path: &PathBuf, samples: &[Sample]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "input_nits,scanout_r,scanout_g,scanout_b,frac_r,frac_g,frac_b,drift,cpu_r,cpu_g,cpu_b,gpu_vs_lut_pct"
    )?;
    for s in samples {
        let (cpu, pct) = match s.cpu {
            Some((c, p)) => (c, p),
            None => ([f64::NAN; 3], f64::NAN),
        };
        writeln!(
            f,
            "{:.4},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{:.3}",
            s.input_nits,
            s.scanout[0],
            s.scanout[1],
            s.scanout[2],
            s.frac[0],
            s.frac[1],
            s.frac[2],
            s.drift,
            cpu[0],
            cpu[1],
            cpu[2],
            pct,
        )?;
    }
    Ok(())
}
