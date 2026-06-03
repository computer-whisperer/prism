//! WLCS (Wayland Conformance Test Suite) integration shim for prism.
//!
//! Built as a `cdylib` that the WLCS C++ runner `dlopen`s:
//! `wlcs libprism_wlcs.so`. WLCS drives prism's protocol/state machine
//! headless — no DRM, no scanout — by:
//!   - starting the compositor on a thread ([`start_prism`] → [`main_loop`]),
//!   - injecting client connections over socket pairs ([`WlcsEvent::NewClient`]),
//!   - positioning windows and synthesizing pointer/touch input.
//!
//! This file is the FFI layer, ported near-verbatim from smithay's
//! `wlcs_anvil` (MIT) at prism's pinned smithay rev. The compositor side
//! that actually diverges from anvil lives in [`main_loop`].

mod main_loop;

use std::{
    io::{Error, ErrorKind},
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::net::UnixStream,
    },
    sync::atomic::{AtomicU32, Ordering},
    thread::JoinHandle,
};

use smithay::{
    reexports::calloop::channel::{channel, Sender},
    utils::{Logical, Point},
};

use wayland_sys::{
    // With `wayland-sys/dlopen`, `wl_display_get_fd` / `wl_proxy_get_id` are
    // reached through the lazily-loaded `wayland_client_handle()` via
    // `ffi_dispatch!`, not as free functions — so we import the handle accessor,
    // not the function symbols.
    client::{wayland_client_handle, wl_display, wl_proxy},
    common::{wl_fixed_t, wl_fixed_to_double},
    ffi_dispatch,
};
use wlcs::{
    extension_list,
    ffi_display_server_api::{
        WlcsExtensionDescriptor, WlcsIntegrationDescriptor, WlcsServerIntegration,
    },
    ffi_wrappers::wlcs_server,
    wlcs_server_integration, Wlcs,
};

wlcs_server_integration!(PrismDisplayServerHandle);

// Interfaces (and max versions) WLCS may bind against. Conservative for
// now; these should be reconciled with the exact versions prism advertises
// in its global setup — advertising a version higher than prism supports
// makes WLCS's bind fail and the test error rather than skip.
static SUPPORTED_EXTENSIONS: &[WlcsExtensionDescriptor] = extension_list!(
    ("wl_compositor", 4),
    ("wl_subcompositor", 1),
    ("wl_data_device_manager", 3),
    ("wl_seat", 7),
    ("wl_output", 4),
    ("xdg_wm_base", 3),
);

static DESCRIPTOR: WlcsIntegrationDescriptor = WlcsIntegrationDescriptor {
    version: 1,
    num_extensions: SUPPORTED_EXTENSIONS.len(),
    supported_extensions: SUPPORTED_EXTENSIONS.as_ptr(),
};

static DEVICE_ID: AtomicU32 = AtomicU32::new(0);

/// Command sent by WLCS (on its own thread) into the compositor thread's
/// event loop via a calloop channel.
#[derive(Debug)]
pub enum WlcsEvent {
    /// Stop the running server.
    Exit,
    /// Adopt a new client from the given connected socket.
    NewClient {
        stream: UnixStream,
        client_id: i32,
    },
    /// Position the given client's surface at an absolute logical location.
    PositionWindow {
        client_id: i32,
        surface_id: u32,
        location: Point<i32, Logical>,
    },
    /* Pointer */
    NewPointer {
        device_id: u32,
    },
    PointerMoveAbsolute {
        device_id: u32,
        location: Point<f64, Logical>,
    },
    PointerMoveRelative {
        device_id: u32,
        delta: Point<f64, Logical>,
    },
    PointerButtonDown {
        device_id: u32,
        button_id: i32,
    },
    PointerButtonUp {
        device_id: u32,
        button_id: i32,
    },
    PointerRemoved {
        device_id: u32,
    },
    /* Touch */
    NewTouch {
        device_id: u32,
    },
    TouchDown {
        device_id: u32,
        location: Point<f64, Logical>,
    },
    TouchMove {
        device_id: u32,
        location: Point<f64, Logical>,
    },
    TouchUp {
        device_id: u32,
    },
    TouchRemoved {
        device_id: u32,
    },
}

struct PrismDisplayServerHandle {
    server: Option<(Sender<WlcsEvent>, JoinHandle<()>)>,
}

impl Wlcs for PrismDisplayServerHandle {
    type Pointer = PointerHandle;
    type Touch = TouchHandle;

    fn new() -> Self {
        PrismDisplayServerHandle { server: None }
    }

    fn start(&mut self) {
        let (tx, rx) = channel();
        let join = crate::start_prism(rx);
        self.server = Some((tx, join));
    }

    fn stop(&mut self) {
        if let Some((sender, join)) = self.server.take() {
            let _ = sender.send(WlcsEvent::Exit);
            let _ = join.join();
        }
    }

    fn create_client_socket(&self) -> std::io::Result<OwnedFd> {
        if let Some((ref sender, _)) = self.server {
            if let Ok((client_side, server_side)) = UnixStream::pair() {
                if let Err(e) = sender.send(WlcsEvent::NewClient {
                    stream: server_side,
                    client_id: client_side.as_raw_fd(),
                }) {
                    return Err(Error::new(ErrorKind::ConnectionReset, e));
                }
                return Ok(client_side.into());
            }
        }
        Err(Error::from(ErrorKind::NotFound))
    }

    fn position_window_absolute(
        &self,
        display: *mut wl_display,
        surface: *mut wl_proxy,
        x: i32,
        y: i32,
    ) {
        let client_id =
            unsafe { ffi_dispatch!(wayland_client_handle(), wl_display_get_fd, display) };
        let surface_id =
            unsafe { ffi_dispatch!(wayland_client_handle(), wl_proxy_get_id, surface) };
        if let Some((ref sender, _)) = self.server {
            let _ = sender.send(WlcsEvent::PositionWindow {
                client_id,
                surface_id,
                location: (x, y).into(),
            });
        }
    }

    fn create_pointer(&mut self) -> Option<Self::Pointer> {
        let server = self.server.as_ref()?;
        Some(PointerHandle {
            device_id: DEVICE_ID.fetch_add(1, Ordering::Relaxed),
            sender: server.0.clone(),
        })
    }

    fn create_touch(&mut self) -> Option<Self::Touch> {
        let server = self.server.as_ref()?;
        Some(TouchHandle {
            device_id: DEVICE_ID.fetch_add(1, Ordering::Relaxed),
            sender: server.0.clone(),
        })
    }

    fn get_descriptor(&self) -> &WlcsIntegrationDescriptor {
        &crate::DESCRIPTOR
    }
}

struct PointerHandle {
    device_id: u32,
    sender: Sender<WlcsEvent>,
}

impl wlcs::Pointer for PointerHandle {
    fn move_absolute(&mut self, x: wl_fixed_t, y: wl_fixed_t) {
        let _ = self.sender.send(WlcsEvent::PointerMoveAbsolute {
            device_id: self.device_id,
            location: (wl_fixed_to_double(x), wl_fixed_to_double(y)).into(),
        });
    }

    fn move_relative(&mut self, dx: wl_fixed_t, dy: wl_fixed_t) {
        let _ = self.sender.send(WlcsEvent::PointerMoveRelative {
            device_id: self.device_id,
            delta: (wl_fixed_to_double(dx), wl_fixed_to_double(dy)).into(),
        });
    }

    fn button_up(&mut self, button: i32) {
        let _ = self.sender.send(WlcsEvent::PointerButtonUp {
            device_id: self.device_id,
            button_id: button,
        });
    }

    fn button_down(&mut self, button: i32) {
        let _ = self.sender.send(WlcsEvent::PointerButtonDown {
            device_id: self.device_id,
            button_id: button,
        });
    }

    fn destroy(&mut self) {}
}

struct TouchHandle {
    device_id: u32,
    sender: Sender<WlcsEvent>,
}

impl wlcs::Touch for TouchHandle {
    fn touch_down(&mut self, x: wl_fixed_t, y: wl_fixed_t) {
        let _ = self.sender.send(WlcsEvent::TouchDown {
            device_id: self.device_id,
            location: (wl_fixed_to_double(x), wl_fixed_to_double(y)).into(),
        });
    }

    fn touch_move(&mut self, x: wl_fixed_t, y: wl_fixed_t) {
        let _ = self.sender.send(WlcsEvent::TouchMove {
            device_id: self.device_id,
            location: (wl_fixed_to_double(x), wl_fixed_to_double(y)).into(),
        });
    }

    fn touch_up(&mut self) {
        let _ = self.sender.send(WlcsEvent::TouchUp {
            device_id: self.device_id,
        });
    }

    fn destroy(&mut self) {}
}

fn start_prism(
    channel: smithay::reexports::calloop::channel::Channel<WlcsEvent>,
) -> JoinHandle<()> {
    std::thread::spawn(move || main_loop::run(channel))
}
