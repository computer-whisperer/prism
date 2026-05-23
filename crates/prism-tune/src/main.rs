//! `prism-tune` — closed-loop calibration + IPC client for the prism
//! compositor.
//!
//! Subcommands:
//!
//! - `msg` — swaymsg-equivalent: send one `prism_ipc::Request`, print the
//!   reply. Useful for one-shot queries and manual overrides.
//! - `calibrate` — closed-loop panel response correction. Drives the
//!   tristim USB colorimeter against an HDR PQ patch on the chosen
//!   output, fits `(gain, gamma)`, applies it live via IPC, and
//!   iterates a fixed number of times.
//!
//! Usage examples:
//!
//! ```text
//! prism-tune msg version
//! prism-tune msg outputs
//! prism-tune msg output DisplayPort-4 sdr-reference-nits 100
//! prism-tune msg output DisplayPort-4 response-curve \
//!     --gain-r 0.45 --gain-g 0.46 --gain-b 0.43 \
//!     --gamma-r 1.08 --gamma-g 1.07 --gamma-b 1.10
//! prism-tune msg output DisplayPort-4 reset-color
//!
//! prism-tune calibrate --output DisplayPort-4 --window 0.10
//! ```

mod calibrate;
mod calibrate_lut3d;
mod characterize;
mod common;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use prism_ipc::socket::Socket;
use prism_ipc::{OutputAction, Reply, Request, Response};

#[derive(Parser)]
#[command(name = "prism-tune", version, about = "Closed-loop color calibration + IPC client for prism")]
struct Cli {
    #[command(subcommand)]
    command: TopCommand,
}

#[derive(Subcommand)]
enum TopCommand {
    /// Send one Request to the prism IPC socket and print the reply.
    #[command(subcommand)]
    Msg(MsgCommand),
    /// Closed-loop per-channel panel calibration via the tristim
    /// colorimeter + a patch surface (HDR or SDR — branched from the
    /// output's current mode via prism IPC). Three phases: query
    /// state → per-channel saturation discovery → per-channel
    /// response refinement. Outputs all per-output color parameters
    /// (panel-peak-nits for HDR, response-curve always) as a paste-
    /// ready KDL block.
    Calibrate(calibrate::CalibrateArgs),
    /// Measurement-driven 3D LUT calibration. Sweeps each channel
    /// independently to capture the panel's actual `commanded → XYZ`
    /// response (no closed-form fit), then numerically inverts to
    /// produce a per-output 3D LUT. Writes a binary `.lut` file plus a
    /// paste-ready KDL snippet referencing it. Use when the closed-form
    /// `(gain, gamma)` model from `calibrate` doesn't reproduce
    /// white-point on a panel with intensity-dependent primaries.
    CalibrateLut3d(calibrate_lut3d::CalibrateLut3dArgs),
    /// Raw response-curve characterization — sweep a channel across a
    /// range of commanded values, log XYZ per sample. Diagnostic
    /// (no fitting, no compositor writes). Use to investigate panel
    /// behaviour that doesn't fit `calibrate`'s simple model.
    Characterize(characterize::CharacterizeArgs),
}

#[derive(Subcommand)]
enum MsgCommand {
    /// Query the running prism version string.
    Version,
    /// List all connected outputs.
    Outputs,
    /// Print info about the currently focused output.
    FocusedOutput,
    /// Apply a per-output action (color overrides, mode, etc.). See
    /// `--help` on the subcommand for the available actions.
    Output {
        /// Output connector name (e.g. `DisplayPort-4`, `HDMI-A-1`).
        /// Use the long form — recent prism builds match the
        /// connector-driver name verbatim, not the `DP-N` shorthand.
        /// Run `prism-tune msg outputs` to list available names.
        output: String,
        #[command(subcommand)]
        action: OutputAction,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        TopCommand::Msg(cmd) => run_msg(cmd),
        TopCommand::Calibrate(args) => calibrate::run(args),
        TopCommand::CalibrateLut3d(args) => calibrate_lut3d::run(args),
        TopCommand::Characterize(args) => characterize::run(args),
    }
}

fn run_msg(cmd: MsgCommand) -> Result<()> {
    let request = match cmd {
        MsgCommand::Version => Request::Version,
        MsgCommand::Outputs => Request::Outputs,
        MsgCommand::FocusedOutput => Request::FocusedOutput,
        MsgCommand::Output { output, action } => Request::Output { output, action },
    };

    let mut socket = Socket::connect()
        .context("connect to PRISM_SOCKET (is prism running, and are you in its env?)")?;
    let reply = socket
        .send(request)
        .context("send request / read reply")?;

    print_reply(reply)
}

fn print_reply(reply: Reply) -> Result<()> {
    // For Phase 1 we just pretty-print the JSON. Future polish: a
    // table view for `outputs`, etc., matching the `niri msg` style.
    match reply {
        Ok(Response::Version(v)) => {
            println!("{v}");
        }
        Ok(response) => {
            let pretty = serde_json::to_string_pretty(&response)
                .context("pretty-print response")?;
            println!("{pretty}");
        }
        Err(message) => {
            anyhow::bail!("prism reported an error: {message}");
        }
    }
    Ok(())
}
