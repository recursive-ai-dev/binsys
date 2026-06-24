//! PuppyBoot V1 — Graphics Output Protocol (GOP) framebuffer discovery.
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  The AIOS native handover (boot.rs §C) passes a linear framebuffer
//!  descriptor to the kernel inside LOADER_PARAMS. The kernel draws directly
//!  into this memory after ExitBootServices, when GOP itself is gone — so we
//!  must capture the framebuffer's physical base, geometry, and stride
//!  *before* exiting boot services.
//!
//!  Selection policy:
//!    • Enumerate every GraphicsOutput handle.
//!    • Skip modes whose pixel format is BltOnly — those have no
//!      CPU-addressable linear framebuffer (UEFI §12.9: PixelBltOnly means
//!      the only access path is the Blt() boot service, which is unusable
//!      post-ExitBootServices).
//!    • Among the rest, pick the mode with the largest pixel area
//!      (width × height). Highest resolution gives the kernel the most
//!      screen real estate; ties are broken by first-found.
//!
//!  `stride` is reported in PIXELS per scanline (UEFI's PixelsPerScanLine),
//!  which may exceed `width` due to hardware row alignment. The kernel
//!  computes a pixel address as:
//!      addr = base + (y * stride + x) * bytes_per_pixel
//!  with bytes_per_pixel = 4 for both RGB and BGR 32-bpp formats.

#![allow(dead_code)]

extern crate alloc;

use log::{info, warn};

use uefi::proto::console::gop::{GraphicsOutput, PixelFormat};
use uefi::table::boot::{BootServices, SearchType};

// ════════════════════════════════════════════════════════════════════════════
//  §1 — Public descriptor
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct FramebufferInfo {
    /// Physical base address of the linear framebuffer (0 if none found).
    pub base:       u64,
    /// Visible width in pixels.
    pub width:      usize,
    /// Visible height in pixels.
    pub height:     usize,
    /// Scanline length in PIXELS (>= width; hardware-aligned).
    pub stride:     usize,
    /// Pixel layout (RGB / BGR / bitmask).
    pub pixel_kind: FbPixelKind,
    /// Total framebuffer size in bytes (as reported by GOP).
    pub fb_size:    usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbPixelKind {
    /// 8 bits each, ordered R, G, B, reserved (little-endian 0x00BBGGRR? no —
    /// UEFI RedGreenBlueReserved8BitPerColor: byte order R,G,B,Reserved).
    Rgb,
    /// UEFI BlueGreenRedReserved8BitPerColor: byte order B,G,R,Reserved.
    Bgr,
    /// Channel positions given by a bitmask (rare; kernel must read masks).
    Bitmask,
    /// No CPU-addressable framebuffer.
    None,
}

impl FramebufferInfo {
    /// Sentinel value used when no usable GOP framebuffer exists. The kernel
    /// interprets base == 0 as "headless / serial-only".
    pub const fn none() -> Self {
        Self {
            base:       0,
            width:      0,
            height:     0,
            stride:     0,
            pixel_kind: FbPixelKind::None,
            fb_size:    0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.base != 0 && self.width != 0 && self.height != 0
    }

    fn area(&self) -> usize {
        self.width.saturating_mul(self.height)
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §2 — Discovery
// ════════════════════════════════════════════════════════════════════════════

pub fn discover_framebuffer(bs: &BootServices) -> FramebufferInfo {
    let handles = match bs.locate_handle_buffer(SearchType::from_proto::<GraphicsOutput>()) {
        Ok(h)  => h,
        Err(_) => {
            warn!("No GraphicsOutput handles — headless framebuffer");
            return FramebufferInfo::none();
        }
    };

    let mut best = FramebufferInfo::none();

    for h in handles.iter() {
        let mut gop = match bs.open_protocol_exclusive::<GraphicsOutput>(*h) {
            Ok(g)  => g,
            Err(_) => continue,
        };

        let mode = gop.current_mode_info();
        let (width, height) = mode.resolution();
        let stride          = mode.stride();

        let pixel_kind = match mode.pixel_format() {
            PixelFormat::Rgb     => FbPixelKind::Rgb,
            PixelFormat::Bgr     => FbPixelKind::Bgr,
            PixelFormat::Bitmask => FbPixelKind::Bitmask,
            PixelFormat::BltOnly => {
                // No linear framebuffer — unusable after ExitBootServices.
                continue;
            }
        };

        // frame_buffer() borrows gop mutably; we extract base + size and drop
        // the borrow immediately. The physical base is stable across the
        // lifetime of the boot, so caching it now is sound.
        let (base, fb_size) = {
            let mut fb = gop.frame_buffer();
            (fb.as_mut_ptr() as u64, fb.size())
        };

        let candidate = FramebufferInfo {
            base, width, height, stride, pixel_kind, fb_size,
        };

        if candidate.is_valid() && candidate.area() > best.area() {
            best = candidate;
        }
    }

    if best.is_valid() {
        info!(
            "Framebuffer: {}x{} stride={}px {:?} @ 0x{:X} ({} bytes)",
            best.width, best.height, best.stride, best.pixel_kind, best.base, best.fb_size
        );
    } else {
        warn!("No usable linear framebuffer found");
    }

    best
}
