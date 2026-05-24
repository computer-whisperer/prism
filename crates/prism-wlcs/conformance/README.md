# WLCS conformance

[WLCS](https://github.com/MirServer/wlcs) (the Wayland Conformance Test
Suite) exercises prism's protocol behavior — surface/pointer coordinates,
enter/leave bookkeeping, subsurface input routing, output advertisement —
against a real client. It tests protocol, not pixels.

`prism-wlcs` builds a cdylib (`libprism_wlcs.so`) that boots the real
`PrismState` headless (a virtual 800×600 output, no DRM/scanout, a
timer-driven frame-callback pump) and exposes the
`wlcs_server_integration` entry point WLCS dlopen's.

## Status

Targeted subset (`test-filter.txt`): **38 passing, 6 expected failures**.
The 6 are a documented smithay-level subsurface-synchronization gap — see
the header of [`expected-failures.txt`](expected-failures.txt). We run a
curated subset rather than the whole suite because prism advertises only
some globals (no `wl_shell` / `zxdg_shell_v6` / `wl_touch`), so the rest
would SKIP or fail for unrelated "missing extension" reasons.

## Running locally

The WLCS runner is a separate C++ binary, not produced by `cargo`. Build
it once from the MirServer sources at the SHA smithay pins (see smithay's
`compile_wlcs.sh`); that yields a `wlcs` binary. It needs a working
Vulkan device — any local GPU, or lavapipe (Mesa software Vulkan) in
headless/CI environments.

```sh
cargo build -p prism-wlcs
crates/prism-wlcs/conformance/run.sh /path/to/wlcs
```

`run.sh` runs the filtered subset and compares the result to the
allowlist: it exits non-zero if a non-allowlisted test fails (a
regression) or if an allowlisted test starts passing (a stale entry to
promote out). Point it at a specific cdylib with a second argument or
`PRISM_WLCS_SO`.

## Files

- `test-filter.txt` — the `--gtest_filter` subset prism targets. Grow it
  as prism gains protocols.
- `expected-failures.txt` — known failures, each explained. Keep small.
- `run.sh` — runner + allowlist diff. Used locally and by CI.
