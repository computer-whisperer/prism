//! Implicit-fence extraction for sampled client dmabufs.
//!
//! A GL/EGL compositor (smithay/niri) gets producer→consumer sync for free:
//! when Mesa samples a dmabuf-backed EGLImage, the kernel inserts an implicit
//! dependency on the buffer's dma_resv write fence, so it never reads a buffer
//! the client's GPU is still writing. prism's Vulkan renderer does NOT get this
//! — Vulkan gives no implicit-sync guarantee for imported dmabufs, so we must
//! carry the dependency explicitly.
//!
//! This exports the buffer's "fence a reader must wait for" (the most recent
//! write/exclusive fence) as a `sync_file` fd via `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`.
//! The caller imports it as a `VkSemaphore` and adds it to the render submit's
//! wait list — the Vulkan analog of Mesa's free lunch.

use std::os::fd::{BorrowedFd, FromRawFd, OwnedFd};

use rustix::ioctl::{ioctl, opcode, Opcode, Updater};

/// `struct dma_buf_export_sync_file` (linux/dma-buf.h): `flags` is written by
/// us, `fd` is filled by the kernel with a fresh `sync_file` fd.
#[repr(C)]
struct DmaBufExportSyncFile {
    flags: u32,
    fd: i32,
}

/// `DMA_BUF_IOCTL_EXPORT_SYNC_FILE = _IOWR('b', 2, struct dma_buf_export_sync_file)`.
const EXPORT_SYNC_FILE: Opcode = opcode::read_write::<DmaBufExportSyncFile>(b'b', 2);

/// `DMA_BUF_SYNC_READ` — export the fence a read access must sync to (i.e. the
/// producer's write fence). What a sampling compositor waits on.
const DMA_BUF_SYNC_READ: u32 = 1 << 0;

/// Export the producer write fence of `dmabuf` (a dmabuf plane fd) as an owned
/// `sync_file` fd. The fd is signalled when the client's GPU finishes writing
/// the buffer; a reader waits on it before sampling.
///
/// Returns an error if the kernel lacks the ioctl (pre-5.20) or no fence is
/// attached — callers treat that as "no wait needed" and skip.
pub fn export_read_fence(dmabuf: BorrowedFd<'_>) -> rustix::io::Result<OwnedFd> {
    let mut arg = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ,
        fd: -1,
    };
    // SAFETY: EXPORT_SYNC_FILE is an _IOWR ioctl whose argument is exactly
    // `struct dma_buf_export_sync_file` (u32 flags in, s32 fd out) — matching
    // `DmaBufExportSyncFile` and the `EXPORT_SYNC_FILE` opcode.
    unsafe {
        ioctl(
            dmabuf,
            Updater::<EXPORT_SYNC_FILE, DmaBufExportSyncFile>::new(&mut arg),
        )?;
    }
    if arg.fd < 0 {
        return Err(rustix::io::Errno::INVAL);
    }
    // SAFETY: the ioctl succeeded and wrote a fresh owned sync_file fd into
    // `arg.fd`; we take ownership exactly once.
    Ok(unsafe { OwnedFd::from_raw_fd(arg.fd) })
}
