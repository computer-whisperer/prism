#!/usr/bin/env bash
#
# Run the WLCS conformance subset against prism's cdylib and compare the
# result to the expected-failures allowlist. Usable locally and in CI.
#
# Usage:
#   run.sh <path-to-wlcs-binary> [path-to-libprism_wlcs.so]
#
# Environment overrides:
#   WLCS_BIN     — wlcs binary (overrides arg 1)
#   PRISM_WLCS_SO — cdylib path (overrides arg 2)
#
# The wlcs runner is a separate C++ binary built from the MirServer wlcs
# sources (it is not produced by `cargo build`); pass its path in. The
# cdylib defaults to target/debug/libprism_wlcs.so relative to the repo
# root.
#
# Exit status:
#   0  every non-allowlisted test passed (allowlisted ones may fail)
#   1  a non-allowlisted test failed (regression), OR an allowlisted test
#      passed (stale entry — promote it out of expected-failures.txt)
#   2  usage / environment error

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"

wlcs_bin="${WLCS_BIN:-${1:-}}"
cdylib="${PRISM_WLCS_SO:-${2:-$repo_root/target/debug/libprism_wlcs.so}}"

if [[ -z "$wlcs_bin" || ! -x "$wlcs_bin" ]]; then
    echo "error: wlcs binary not found/executable: '${wlcs_bin:-<unset>}'" >&2
    echo "usage: run.sh <path-to-wlcs-binary> [path-to-libprism_wlcs.so]" >&2
    exit 2
fi
if [[ ! -f "$cdylib" ]]; then
    echo "error: cdylib not found: '$cdylib' (build with: cargo build -p prism-wlcs)" >&2
    exit 2
fi

filter="$(grep -vE '^\s*(#|$)' "$here/test-filter.txt" | head -n1)"
if [[ -z "$filter" ]]; then
    echo "error: empty test filter in $here/test-filter.txt" >&2
    exit 2
fi

# Normalize the allowlist to sorted, comment-stripped test names.
expected="$(grep -vE '^\s*(#|$)' "$here/expected-failures.txt" | sort -u)"

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

echo ">> wlcs $cdylib --gtest_filter='$filter'"
"$wlcs_bin" "$cdylib" --gtest_filter="$filter" >"$log" 2>&1
echo ">> wlcs exited $?"

# Distinct failing test names. gtest prints `[  FAILED  ] Suite.Test/0`
# both inline (with a " (N ms)" suffix) and again in the trailing summary
# (with ", where GetParam()..."); strip either suffix and dedup.
actual="$(grep -E '^\[  FAILED  \] .+/[0-9]' "$log" \
    | sed -E 's/^\[  FAILED  \] //; s/,? where.*//; s/ \([0-9]+ ms\)//' \
    | sort -u)"

passed_count="$(grep -cE '^\[       OK \] ' "$log")"

# regressions = failed but not allowlisted; stale = allowlisted but didn't fail.
regressions="$(comm -23 <(printf '%s\n' "$actual") <(printf '%s\n' "$expected"))"
stale="$(comm -13 <(printf '%s\n' "$actual") <(printf '%s\n' "$expected"))"

# `stale` includes allowlisted tests that PASSED *or* were filtered/SKIPPED.
# Only flag ones that actually ran and passed, to avoid false alarms when
# the filter doesn't select an allowlisted test.
unexpected_pass=""
while IFS= read -r t; do
    [[ -z "$t" ]] && continue
    if grep -qE "^\[       OK \] ${t//\//\\/}( |\$)" "$log"; then
        unexpected_pass+="$t"$'\n'
    fi
done <<<"$stale"

echo
echo "== WLCS conformance =="
echo "passed:            $passed_count"
echo "expected failures: $(grep -cvE '^\s*(#|$)' "$here/expected-failures.txt")"

status=0
if [[ -n "${regressions//[$'\n']/}" ]]; then
    echo
    echo "REGRESSIONS (failed, not in allowlist):"
    printf '  %s\n' $regressions
    status=1
fi
if [[ -n "${unexpected_pass//[$'\n']/}" ]]; then
    echo
    echo "UNEXPECTED PASSES (in allowlist but now passing — remove them):"
    printf '  %s\n' $unexpected_pass
    status=1
fi

if [[ $status -eq 0 ]]; then
    echo
    echo "OK: all non-allowlisted tests passed."
fi
exit $status
