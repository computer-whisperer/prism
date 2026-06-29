//! Crash-resilient breadcrumb logging.
//!
//! Mirror of `prism::main::breadcrumb` so DRM-side code can leave a trail
//! that survives an external `SIGKILL` or a hard kernel wedge. Tracing
//! via stdio is buffered and disappears when the process is killed
//! ungracefully; this writes + `fsync`s per line.
//!
//! Opt-in: nothing is written unless `$PRISM_CRUMBS` (the output path) or
//! `$PRISM_FLIP_TRACE` is set. This keeps normal operation from leaking a
//! `prism.crumbs` file into the cwd of every session — the facility exists
//! for targeted lockup debugging, not steady-state.
//!
//! Path: `$PRISM_CRUMBS` if set, otherwise `./prism.crumbs` (cwd at
//! process start). Each line is `<unix-timestamp.fractional>: <msg>`.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

static CRUMBS_PATH: OnceLock<PathBuf> = OnceLock::new();
static CRUMBS_ENABLED: OnceLock<bool> = OnceLock::new();
static FLIP_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

fn env_set(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0")
}

/// Breadcrumbs are written only when explicitly opted into — either a
/// `$PRISM_CRUMBS` path is given, or `$PRISM_FLIP_TRACE` is set (which
/// implies the user wants the trail). Off by default.
fn crumbs_enabled() -> bool {
    *CRUMBS_ENABLED.get_or_init(|| env_set("PRISM_CRUMBS") || env_set("PRISM_FLIP_TRACE"))
}

fn crumbs_path() -> &'static PathBuf {
    CRUMBS_PATH.get_or_init(|| {
        std::env::var("PRISM_CRUMBS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("prism.crumbs"))
    })
}

/// Per-frame breadcrumbs gated behind `PRISM_FLIP_TRACE=1`. fsync per
/// line caps throughput at ~150 ops/sec on consumer SSDs — pages-flips
/// at 60Hz × N outputs blow past that. Only enable for targeted
/// debugging of the page_flip / vblank cadence.
pub fn flip_trace_enabled() -> bool {
    *FLIP_TRACE_ENABLED.get_or_init(|| env_set("PRISM_FLIP_TRACE"))
}

/// Append a per-flip / per-vblank breadcrumb iff `PRISM_FLIP_TRACE` is
/// set. Cheap no-op when disabled (single atomic load + branch).
pub fn flip_trace(msg: &str) {
    if !flip_trace_enabled() {
        return;
    }
    breadcrumb(msg);
}

/// Append one fsync'd line to the breadcrumbs file. No-op unless crumbs
/// are enabled (see [`crumbs_enabled`]); also silently no-ops if the file
/// can't be opened.
pub fn breadcrumb(msg: &str) {
    if !crumbs_enabled() {
        return;
    }
    let line = format!(
        "{:.3}: {msg}\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(crumbs_path())
    {
        let _ = f.write_all(line.as_bytes());
        let _ = f.sync_all();
    }
}
