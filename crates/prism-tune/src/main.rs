//! `prism-tune` — closed-loop calibration + IPC client for the prism
//! compositor.
//!
//! Phase 1 (this binary today): the `msg` subcommand. Connects to a
//! running prism's `PRISM_SOCKET`, sends a `prism_ipc::Request`, prints
//! the reply. Same shape as `niri msg` / `swaymsg`.
//!
//! Phase 2 (future): vendor in the spyder-driver + patch surface and
//! add a `calibrate` subcommand that runs the full sweep → fit →
//! apply → verify loop.
//!
//! Usage examples:
//!
//! ```text
//! prism-tune msg version
//! prism-tune msg outputs
//! prism-tune msg focused-output
//! prism-tune msg output DP-4 sdr-reference-nits 100
//! prism-tune msg output DP-4 response-curve \
//!     --gain-r 0.45 --gain-g 0.46 --gain-b 0.43 \
//!     --gamma-r 1.08 --gamma-g 1.07 --gamma-b 1.10
//! prism-tune msg output DP-4 reset-color
//! ```

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
        /// Output connector name (e.g. `DP-4`, `HDMI-A-1`).
        output: String,
        #[command(subcommand)]
        action: OutputAction,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        TopCommand::Msg(cmd) => run_msg(cmd),
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
