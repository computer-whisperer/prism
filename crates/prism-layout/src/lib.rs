//! Window layout, input routing, per-surface state.
//!
//! Hosts the niri-ported `shell` of the compositor — `window/`, `layer/`,
//! `layout/`, `input/`, `cursor.rs` — collapsed into one crate because the
//! cross-module dependencies don't split cleanly. See the Cargo.toml
//! description for the replace/depend boundary against `render_helpers/`.
//!
//! Current state: utils foundation in place; window/layer/layout/input/cursor
//! are still being ported one chunk at a time.

pub mod utils;
