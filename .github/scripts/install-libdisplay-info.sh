#!/usr/bin/env bash
# Build and install libdisplay-info from source.
#
# prism-drm parses EDID through the `libdisplay-info` Rust crate (0.3.x),
# whose `-sys` build script requires the C library >= 0.2.0 via pkg-config.
# Ubuntu (noble) only packages 0.1.1, so CI builds a matching version here.
# Pinned to the same release the dev box runs against.
set -euo pipefail

VERSION=0.3.0
SRC="$(mktemp -d)"

git clone --depth 1 --branch "$VERSION" \
  https://gitlab.freedesktop.org/emersion/libdisplay-info.git "$SRC"

# --prefix=/usr so the .pc and .so land on the default pkg-config / linker
# search paths (no PKG_CONFIG_PATH / LD_LIBRARY_PATH juggling needed).
meson setup --prefix=/usr "$SRC/build" "$SRC"
sudo meson install -C "$SRC/build"
sudo ldconfig
