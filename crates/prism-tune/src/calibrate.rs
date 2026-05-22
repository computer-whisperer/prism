//! `prism-tune calibrate` — closed-loop per-output panel response correction.
//!
//! Flow per iteration:
//! 1. Push the current response-curve estimate to prism over IPC.
//! 2. Walk a compact gray ramp on the chosen output (HDR PQ patches).
//! 3. Measure each with the tristim colorimeter.
//! 4. Log-linear fit of (target_nits, measured_Y) → updated (gain, gamma).
//!
//! After N iterations the fit converges to the best power-law approximation
//! of the panel's true response, sampled where we measured.
//!
//! Single curve applied identically to R/G/B (matches how the user has
//! been calibrating manually from gray sweeps). Per-channel from pure
//! primaries is a future refinement once we see how well the gray fit
//! holds for off-gray content.

use anyhow::{Context, Result};
use clap::Args;
use prism_ipc::socket::Socket;
use prism_ipc::{OutputAction, Request, Response};
use std::thread;
use std::time::Duration;
use tristim_display::{PatchSurface, PqDescriptionParams};
use tristim_driver::{Colorimeter, measurement::raw_to_xyz};

#[derive(Args)]
pub struct CalibrateArgs {
    /// Connector to calibrate (e.g. `DP-4`).
    #[arg(long)]
    pub output: String,
    /// Colorimeter calibration index (0..=6). 0 = the "General" preset.
    #[arg(long, default_value_t = 0)]
    pub cal: u8,
    /// Mastering / sweep peak nits for the HDR patches.
    #[arg(long, default_value_t = 400)]
    pub peak_nits: u32,
    /// Centered bright-window fraction (0..=1). Use 0.04–0.10 on
    /// ABL-throttled OLEDs to measure peak response; 1.0 fills the screen.
    #[arg(long, default_value_t = 0.10)]
    pub window: f64,
    /// Number of measure → fit → apply iterations.
    #[arg(long, default_value_t = 3)]
    pub iterations: u32,
    /// Seconds to wait for the puck to be placed before the first sweep.
    #[arg(long, default_value_t = 5)]
    pub prep_secs: u64,
    /// Settle time after each color change before measuring (ms).
    #[arg(long, default_value_t = 32)]
    pub settle_ms: u64,
    /// Leave the tuned curve active on exit (default: send ResetColor so
    /// the persisted KDL config wins again).
    #[arg(long)]
    pub keep: bool,
}

pub fn run(args: CalibrateArgs) -> Result<()> {
    // Start from a clean baseline — clear any prior runtime override.
    send_action(&args.output, OutputAction::ResetColor)
        .context("initial ResetColor")?;

    let mut device = Colorimeter::open_any().context("open colorimeter")?;
    let info = device.get_info().context("read colorimeter info")?;
    eprintln!(
        "Colorimeter: Spyder SN {} HW {}.{:02}",
        info.serial, info.hw_version.0, info.hw_version.1
    );
    let cal = device.get_calibration(args.cal).context("download cal matrix")?;
    let setup = device.get_setup(&cal).context("download setup")?;

    let params = PqDescriptionParams {
        // ~0.0005 cd/m² black point — fine for both LCD and OLED, the
        // compositor will scan it out and the panel does the rest.
        mastering_min_lum_ticks: 5,
        mastering_max_lum: args.peak_nits,
        max_cll: args.peak_nits,
        max_fall: args.peak_nits / 2,
    };
    let mut patch = PatchSurface::open_hdr(&args.output, params)
        .with_context(|| format!("open HDR patch on {}", args.output))?;
    patch.set_window_fraction(args.window).context("set window fraction")?;
    patch.set_nits([0.0, 0.0, 0.0]).context("initial black")?;

    eprintln!(
        "Place the puck flat on {} now. Calibration starts in {}s.",
        args.output, args.prep_secs
    );
    for s in (1..=args.prep_secs).rev() {
        eprintln!("  starting in {s}s...");
        thread::sleep(Duration::from_secs(1));
    }

    // Compact log-spaced target set. Skewed low — PQ packs precision into
    // the dark range so verifying there is the most informative.
    let peak = args.peak_nits as f64;
    let targets: Vec<f64> = vec![5.0, 25.0, 100.0, peak * 0.5, peak];

    let settle = Duration::from_millis(args.settle_ms);
    let mut gain = 1.0f64;
    let mut gamma = 1.0f64;

    for iter in 0..args.iterations {
        eprintln!(
            "\n--- iteration {} / {}  (current curve: gain={:.4}, gamma={:.4}) ---",
            iter + 1,
            args.iterations,
            gain,
            gamma
        );

        // One reset per iteration — Argyll-default per-measurement reset
        // is the dominant per-patch cost and overkill over a sub-10s batch.
        device.send_reset().context("colorimeter reset")?;

        let mut samples: Vec<(f64, f64)> = Vec::with_capacity(targets.len());
        for &t in &targets {
            patch.set_nits([t, t, t]).with_context(|| format!("set {t} nits"))?;
            thread::sleep(settle);
            let raw = device.measure_raw_no_reset(&setup).context("measure")?;
            let xyz = raw_to_xyz(&raw, &setup, &cal);
            eprintln!(
                "  target {:>6.1} cd/m²  →  measured Y = {:>7.3}",
                t, xyz.y
            );
            samples.push((t, xyz.y));
        }

        // After applying compositor curve V_k = (gain_k, gamma_k):
        //   commanded = (T / gain_k)^(1/gamma_k)
        //   Y         = panel_gain * commanded^panel_gamma
        // → log Y = log(panel_gain) + (panel_gamma/gamma_k) * (log T - log gain_k)
        // Linear fit log T → log Y gives slope b, intercept a, then:
        //   panel_gamma = b * gamma_k
        //   panel_gain  = exp(a) * gain_k^b
        let (slope, intercept) = log_linear_fit(&samples);
        let new_gamma = slope * gamma;
        let new_gain = intercept.exp() * gain.powf(slope);
        eprintln!(
            "  fit: panel_gain = {:.4}, panel_gamma = {:.4}",
            new_gain, new_gamma
        );
        gain = new_gain;
        gamma = new_gamma;

        send_action(
            &args.output,
            OutputAction::ResponseCurve {
                gain_r: gain,
                gain_g: gain,
                gain_b: gain,
                gamma_r: gamma,
                gamma_g: gamma,
                gamma_b: gamma,
            },
        )
        .context("apply ResponseCurve")?;
    }

    // Black before the user lifts the puck — both polite and ABL-friendly.
    let _ = patch.set_nits([0.0, 0.0, 0.0]);

    println!();
    println!("# Paste into the matching output block in your prism config:");
    println!("output \"{}\" {{", args.output);
    println!("    color {{");
    println!(
        "        response-curve gain-r={:.4} gain-g={:.4} gain-b={:.4} gamma-r={:.4} gamma-g={:.4} gamma-b={:.4}",
        gain, gain, gain, gamma, gamma, gamma
    );
    println!("    }}");
    println!("}}");

    if args.keep {
        eprintln!("\n--keep: tuned curve remains active until prism restart.");
    } else {
        eprintln!("\nRestoring KDL config defaults (use --keep to leave the tuned curve active).");
        send_action(&args.output, OutputAction::ResetColor)
            .context("final ResetColor")?;
    }

    Ok(())
}

/// Ordinary-least-squares fit on log-log. Skips non-positive points.
fn log_linear_fit(samples: &[(f64, f64)]) -> (f64, f64) {
    let pts: Vec<(f64, f64)> = samples
        .iter()
        .filter(|(t, y)| *t > 0.0 && *y > 0.0)
        .map(|(t, y)| (t.ln(), y.ln()))
        .collect();
    assert!(
        pts.len() >= 2,
        "need at least 2 positive (target, measured) samples to fit"
    );
    let n = pts.len() as f64;
    let mean_x = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let num: f64 = pts.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
    let den: f64 = pts.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

/// Fire a one-shot IPC request against the running prism. The server
/// closes after each reply, so each request opens a fresh connection.
fn send_action(output: &str, action: OutputAction) -> Result<()> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket
        .send(Request::Output {
            output: output.to_string(),
            action,
        })
        .context("send request / read reply")?;
    match reply {
        Ok(Response::OutputConfigChanged(_)) => Ok(()),
        Ok(other) => anyhow::bail!("unexpected reply: {other:?}"),
        Err(e) => anyhow::bail!("prism returned error: {e}"),
    }
}
