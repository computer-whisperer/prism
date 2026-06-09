//! Helper for blocking communication over the prism socket.

use std::env;
use std::io::{self, BufRead, BufReader, IoSlice, IoSliceMut, Write};
use std::mem::MaybeUninit;
use std::net::Shutdown;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;

use rustix::net::{
    recvmsg, sendmsg, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags,
};

use crate::{Event, Reply, Request};

/// Name of the environment variable containing the prism IPC socket path.
pub const SOCKET_PATH_ENV: &str = "PRISM_SOCKET";

/// Helper for blocking communication over the prism socket.
///
/// This struct is used to communicate with the prism IPC server. It handles the socket connection
/// and serialization/deserialization of messages.
pub struct Socket {
    stream: BufReader<UnixStream>,
}

impl Socket {
    /// Connects to the default prism IPC socket.
    ///
    /// This is equivalent to calling [`Self::connect_to`] with the path taken from the
    /// [`SOCKET_PATH_ENV`] environment variable.
    pub fn connect() -> io::Result<Self> {
        let socket_path = env::var_os(SOCKET_PATH_ENV).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{SOCKET_PATH_ENV} is not set, are you running this within prism?"),
            )
        })?;
        Self::connect_to(socket_path)
    }

    /// Connects to the prism IPC socket at the given path.
    pub fn connect_to(path: impl AsRef<Path>) -> io::Result<Self> {
        let stream = UnixStream::connect(path.as_ref())?;
        let stream = BufReader::new(stream);
        Ok(Self { stream })
    }

    /// Sends a request to prism and returns the response.
    ///
    /// Return values:
    ///
    /// * `Ok(Ok(response))`: successful [`Response`](crate::Response) from prism
    /// * `Ok(Err(message))`: error message from prism
    /// * `Err(error)`: error communicating with prism
    pub fn send(&mut self, request: Request) -> io::Result<Reply> {
        let mut buf = serde_json::to_string(&request).unwrap();
        buf.push('\n');
        self.stream.get_mut().write_all(buf.as_bytes())?;

        buf.clear();
        self.stream.read_line(&mut buf)?;

        let reply = serde_json::from_str(&buf)?;
        Ok(reply)
    }

    /// Sends a request and reads a reply that may carry an out-of-band
    /// file descriptor (`SCM_RIGHTS`), e.g. [`Request::CaptureFrame`].
    ///
    /// Returns the parsed [`Reply`] plus the received fd, if the server
    /// attached one. The reply line is read with `recvmsg` rather than
    /// the buffered [`Self::send`] path, because ancillary fds are
    /// silently dropped by an ordinary `read()` that swallows the bytes
    /// they were queued against. Use this on a **fresh** connection
    /// (one request per socket, as prism's clients already do) so no
    /// buffered read-ahead can race the ancillary data.
    ///
    /// [`Request::CaptureFrame`]: crate::Request::CaptureFrame
    pub fn send_recv_fd(&mut self, request: Request) -> io::Result<(Reply, Option<OwnedFd>)> {
        let mut buf = serde_json::to_string(&request).unwrap();
        buf.push('\n');
        self.stream.get_mut().write_all(buf.as_bytes())?;

        let stream = self.stream.get_mut();
        let mut line: Vec<u8> = Vec::with_capacity(256);
        let mut received_fd: Option<OwnedFd> = None;
        let mut chunk = [0u8; 4096];
        loop {
            let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
            let mut ancillary = RecvAncillaryBuffer::new(&mut space);
            let mut iov = [IoSliceMut::new(&mut chunk)];
            let ret = recvmsg(&*stream, &mut iov, &mut ancillary, RecvFlags::empty())?;
            for msg in ancillary.drain() {
                if let RecvAncillaryMessage::ScmRights(fds) = msg {
                    // Keep the last fd; the protocol attaches exactly one.
                    received_fd = fds.into_iter().next_back().or(received_fd);
                }
            }
            if ret.bytes == 0 {
                break; // EOF before newline
            }
            line.extend_from_slice(&chunk[..ret.bytes]);
            if line.contains(&b'\n') {
                break;
            }
        }

        let text = String::from_utf8_lossy(&line);
        let reply: Reply = serde_json::from_str(text.trim_end())?;
        Ok((reply, received_fd))
    }

    /// Starts reading event stream [`Event`]s from the socket.
    ///
    /// The returned function will block until the next [`Event`] arrives, then return it.
    ///
    /// Use this only after requesting an [`EventStream`][Request::EventStream].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use prism_ipc::{Request, Response};
    /// use prism_ipc::socket::Socket;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut socket = Socket::connect()?;
    ///
    ///     let reply = socket.send(Request::EventStream)?;
    ///     if matches!(reply, Ok(Response::Handled)) {
    ///         let mut read_event = socket.read_events();
    ///         while let Ok(event) = read_event() {
    ///             println!("Received event: {event:?}");
    ///         }
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn read_events(self) -> impl FnMut() -> io::Result<Event> {
        let Self { mut stream } = self;
        let _ = stream.get_mut().shutdown(Shutdown::Write);

        let mut buf = String::new();
        move || {
            buf.clear();
            stream.read_line(&mut buf)?;
            let event = serde_json::from_str(&buf)?;
            Ok(event)
        }
    }
}

/// Create a memfd holding `bytes`, for attaching to a reply
/// via [`write_reply_with_fd`] — the server side of a bulk-data response
/// (e.g. `Response::Lut3d`). The receiver reads it with `pread(2)` from
/// offset 0, so the write cursor's final position doesn't matter.
pub fn memfd_from_bytes(name: &str, bytes: &[u8]) -> io::Result<OwnedFd> {
    let memfd =
        rustix::fs::memfd_create(name, rustix::fs::MemfdFlags::CLOEXEC).map_err(io::Error::from)?;
    let file = std::fs::File::from(memfd);
    let mut writer = &file;
    writer.write_all(bytes)?;
    Ok(file.into())
}

/// Write one JSON [`Reply`] line to `stream`, optionally attaching `fd`
/// as out-of-band ancillary data (`SCM_RIGHTS`). The server side of
/// [`Socket::send_recv_fd`]: when `fd` is `Some`, the reply is sent with
/// `sendmsg` so the descriptor rides alongside the reply bytes;
/// otherwise it's a plain write, identical to the normal reply path.
pub fn write_reply_with_fd(
    stream: &UnixStream,
    reply: &Reply,
    fd: Option<BorrowedFd<'_>>,
) -> io::Result<()> {
    let mut buf = serde_json::to_string(reply)?;
    buf.push('\n');

    let Some(fd) = fd else {
        let mut writer: &UnixStream = stream;
        return writer.write_all(buf.as_bytes());
    };

    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut ancillary = SendAncillaryBuffer::new(&mut space);
    let fds = [fd];
    let pushed = ancillary.push(SendAncillaryMessage::ScmRights(&fds));
    debug_assert!(pushed, "ancillary buffer too small for one fd");

    // The fd attaches to this first sendmsg; for our small replies it
    // carries the whole line, but loop defensively on short sends and
    // flush any remainder with plain writes (the fd's already delivered).
    let iov = [IoSlice::new(buf.as_bytes())];
    let mut sent = sendmsg(stream, &iov, &mut ancillary, SendFlags::empty())?;
    let mut writer: &UnixStream = stream;
    while sent < buf.len() {
        sent += writer.write(&buf.as_bytes()[sent..])?;
    }
    Ok(())
}

/// One write attempt on `stream` (which may be nonblocking), attaching
/// `fd` as out-of-band ancillary data (`SCM_RIGHTS`) when present. The
/// kernel ties the ancillary payload to the bytes accepted by the same
/// `sendmsg`, so `fd` is consumed only once at least one byte goes out
/// — a caller re-arming after `WouldBlock` neither drops the fd nor
/// sends it twice. Returns the byte count accepted; the caller loops
/// over the remainder. The incremental sibling of
/// [`write_reply_with_fd`] for servers that must not block.
pub fn write_chunk_with_fd(
    stream: &UnixStream,
    bytes: &[u8],
    fd: &mut Option<OwnedFd>,
) -> io::Result<usize> {
    let Some(borrowed) = fd.as_ref().map(|f| f.as_fd()) else {
        let mut writer: &UnixStream = stream;
        return writer.write(bytes);
    };

    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut ancillary = SendAncillaryBuffer::new(&mut space);
    let fds = [borrowed];
    let pushed = ancillary.push(SendAncillaryMessage::ScmRights(&fds));
    debug_assert!(pushed, "ancillary buffer too small for one fd");

    let iov = [IoSlice::new(bytes)];
    let sent = sendmsg(stream, &iov, &mut ancillary, SendFlags::empty())?;
    if sent > 0 {
        *fd = None;
    }
    Ok(sent)
}
