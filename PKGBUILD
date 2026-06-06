# Maintainer: Christian Balcom <robot.inventor@gmail.com>

pkgname=prism
pkgver=0.1.0
pkgrel=1
pkgdesc='Vulkan-native, HDR-native Wayland compositor'
arch=('x86_64')
url='https://github.com/computer-whisperer/prism'
license=('GPL-3.0-or-later')
depends=(
    'gcc-libs'
    'glibc'
    'libdisplay-info'
    'libinput'
    'libusb' # prism-tune colorimeter access
    'libxkbcommon'
    'mesa' # libgbm
    'seatd' # libseat
    'systemd-libs' # libudev
    'vulkan-icd-loader'
    'wayland'
)
optdepends=(
    'xwayland-satellite: X11 application support (spawned on demand)'
)
makedepends=('cargo' 'glslang' 'pkgconf')
# Disable system LTO — Arch's default `-flto=auto` lands in CFLAGS and makes
# the C shims compiled by build scripts via the `cc` crate (wayland-backend,
# smithay) emit LTO-IR objects, which rust-lld can't resolve at the final
# Rust link step.
options=('!lto')
source=("$pkgname-$pkgver.tar.gz::$url/archive/refs/tags/v$pkgver.tar.gz")
# NOTE: placeholder hash from a local `git archive` — regenerate from the
# actual GitHub tag tarball (updpkgsums) once v0.1.0 is tagged and pushed.
sha256sums=('e6e030259291d691cf68569b0573f3a015602292d5810771a292eed15ba3b139')

prepare() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    cargo fetch --locked --target "$(rustc -vV | sed -n 's/host: //p')"
}

build() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    export CARGO_TARGET_DIR=target
    cargo build --release --frozen -p prism -p prism-tune
}

check() {
    cd "$pkgname-$pkgver"
    export RUSTUP_TOOLCHAIN=stable
    # prism-wlcs (WLCS integration shim) and prism-shmtest are dev-only
    # tools — neither packaged nor tested here.
    cargo test --release --frozen --workspace \
        --exclude prism-wlcs --exclude prism-shmtest
}

package() {
    cd "$pkgname-$pkgver"
    install -Dm755 target/release/prism "$pkgdir/usr/bin/prism"
    install -Dm755 target/release/prism-tune "$pkgdir/usr/bin/prism-tune"
    # Session launcher: hands the compositor to `systemd --user` so it runs
    # on the real session bus (keyring, portals). See resources/prism-session.
    install -Dm755 resources/prism-session "$pkgdir/usr/bin/prism-session"
    install -Dm644 resources/prism.service resources/prism-shutdown.target \
        -t "$pkgdir/usr/lib/systemd/user"
    # Display-manager session entry (Exec=prism-session).
    install -Dm644 resources/prism.desktop \
        "$pkgdir/usr/share/wayland-sessions/prism.desktop"
    # Reference copy of the built-in default config (also embedded in the
    # binary; `prism` runs with it when ~/.config/prism/config.kdl is absent).
    install -Dm644 resources/default-config.kdl README.md \
        -t "$pkgdir/usr/share/doc/$pkgname"
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
