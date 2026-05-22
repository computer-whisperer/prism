//! Crash-resilient breadcrumb logging.
//!
//! Mirror of `prism::main::breadcrumb` so DRM-side code can leave a trail
//! that survives `SIGKILL` (the watchdog) or a hard kernel wedge. Tracing
//! via stdio is buffered and disappears when the process is killed
//! ungracefully; this writes + `fsync`s per line.
//!
//! Path: `$PRISM_CRUMBS` if set, otherwise `./prism.crumbs` (cwd at
//! process start). Each line is `<unix-timestamp.fractional>: <msg>`.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

static CRUMBS_PATH: OnceLock<PathBuf> = OnceLock::new();
static FLIP_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

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
    *FLIP_TRACE_ENABLED.get_or_init(|| {
        std::env::var("PRISM_FLIP_TRACE")
            .ok()
            .filter(|v| !v.is_empty() && v != "0")
            .is_some()
    })
}

/// Append a per-flip / per-vblank breadcrumb iff `PRISM_FLIP_TRACE` is
/// set. Cheap no-op when disabled (single atomic load + branch).
pub fn flip_trace(msg: &str) {
    if !flip_trace_enabled() {
        return;
    }
    breadcrumb(msg);
}

/// Append one fsync'd line to the breadcrumbs file. Silently no-ops if
/// the file can't be opened.
pub fn breadcrumb(msg: &str) {
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
