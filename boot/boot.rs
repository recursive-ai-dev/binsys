//! PuppyBoot V1 — Boot dispatch (Linux EFI-stub, Linux handover, AIOS native, chainload).
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  FOUR BOOT PATHS — selected by `EntryConfig::entry_type`
//!  ────────────────────────────────────────────────────────────────────────────
//!  §A   boot_linux_stub
//!         Primary modern Linux path. Kernel is a PE/COFF EFI binary
//!         (CONFIG_EFI_STUB=y, default for every Arch kernel since 2012).
//!         LoadImage / set LoadOptions with UTF-16 cmdline containing
//!         `initrd=` directives / StartImage. The kernel's own EFI stub
//!         handles GetMemoryMap, ExitBootServices, and boot_params
//!         construction.  Reference: Linux Documentation/admin-guide/efi-stub.rst.
//!
//!  §B   boot_linux_handover
//!         Legacy Linux path (kernels < 6.15 with handover_offset > 0).
//!         We allocate boot_params, copy the setup-header sectors of the
//!         bzImage in, populate efi_info at offset 0x1C0 ("EL64"), patch
//!         loader fields, then jump to handover_offset with MS x64 ABI
//!         registers (RCX, RDX, R8). Reference: Linux Documentation/x86/boot.rst
//!         §"EFI HANDOVER PROTOCOL".
//!
//!         IMPORTANT: This path was DEPRECATED upstream in Linux 6.2
//!         (Ard Biesheuvel, 2023-01-12) and REMOVED in 6.15 (May 2025).
//!         V1 keeps it for compatibility with kernels of that vintage.
//!
//!  §C   boot_aios
//!         AIOS native ELF kernel handover. Parses ELF64 PT_LOAD program
//!         headers, copies segments to their requested physical addresses
//!         (with relocation fallback), constructs an 80-byte LOADER_PARAMS
//!         block per Custom-OS-Manual §3 Handover Spec v2.3, appends the
//!         UEFI memory map immediately after it (kernel computes the map
//!         pointer as `(uint8_t*)LP + sizeof(LoaderParams)`), then calls
//!         ExitBootServices and jumps to the kernel via SysV AMD64 ABI
//!         (RDI = LP*).
//!
//!  §D   chainload_efi
//!         LoadImage from a partition DevicePath / StartImage. Used for
//!         booting Windows Boot Manager (bootmgfw.efi), systemd-boot,
//!         shim, or any third-party EFI binary.
//!
//!  All paths take ownership of the SystemTable (passed by value) when they
//!  intend to call ExitBootServices, so the type system enforces the
//!  invariant that no further BootServices calls are possible after exit.

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::mem;
use core::ptr;

use log::{error, info, warn};

use uefi::prelude::*;
use uefi::proto::device_path::DevicePath;
use uefi::proto::loaded_image::LoadedImage;
use uefi::table::boot::{AllocateType, LoadImageSource, MemoryType};
use uefi::table::runtime::ResetType;

use crate::config::EntryConfig;
use crate::disk;
use crate::gop;
use crate::partition::PartitionInfo;

// ════════════════════════════════════════════════════════════════════════════
//  §0 — Shared utilities
// ════════════════════════════════════════════════════════════════════════════

/// Hard reset on unrecoverable boot failure. Never returns.
///
/// Takes `&SystemTable<Boot>` (shared) so it can coexist with any in-flight
/// shared borrows the caller may hold (e.g. an inline `st.boot_services()`
/// call inside a match scrutinee). All UEFI services we call here —
/// `stall` and `reset` — accept `&self`, so the shared signature is sound.
fn die(st: &SystemTable<Boot>, ctx: &str, msg: impl AsRef<str>) -> ! {
    let msg = msg.as_ref();
    error!("BOOT FATAL [{}]: {}", ctx, msg);
    // Pause so the user can see the message on screen before reset.
    let _ = st.boot_services().stall(5_000_000); // 5 seconds
    st.runtime_services().reset(ResetType::COLD, Status::ABORTED, None);
}

/// Encode a Rust `&str` as UTF-16 LE bytes terminated by a single U+0000.
/// The returned Vec's length is in BYTES and includes the 2-byte NUL.
///
/// EFI LoadOptionsSize is documented in UEFI 2.10 §9.1.1 (EFI_LOADED_IMAGE_PROTOCOL)
/// as the size in bytes including the trailing null.
fn utf16le_with_nul(s: &str) -> Vec<u8> {
    // Capacity: each ASCII char → 2 bytes, each non-ASCII BMP → 2 bytes,
    // supplementary planes → 4 bytes; +2 for NUL.
    let mut out = Vec::with_capacity(s.len() * 2 + 2);
    for u in s.encode_utf16() {
        out.push((u & 0xFF) as u8);
        out.push((u >> 8)   as u8);
    }
    out.push(0);
    out.push(0);
    out
}

/// Translate POSIX-style path to UEFI backslash form.
/// `/EFI/arch/initramfs-linux.img` → `\EFI\arch\initramfs-linux.img`.
fn uefi_backslash(p: &str) -> String {
    p.chars().map(|c| if c == '/' { '\\' } else { c }).collect()
}

// ════════════════════════════════════════════════════════════════════════════
//  §A — boot_linux_stub  (modern Linux EFI stub, PREFERRED)
// ════════════════════════════════════════════════════════════════════════════
//
//  Mechanics (Linux Documentation/admin-guide/efi-stub.rst):
//    • Kernel is a PE/COFF binary.
//    • LoadOptions is a UTF-16 string. Tokens prefixed with `initrd=` are
//      consumed by the EFI stub: each tag specifies a file (backslash-path
//      relative to the loaded image's volume) to splice into the initramfs.
//    • Multiple initrd= directives are processed in declaration order and
//      concatenated. Microcode MUST appear first per
//      Documentation/x86/microcode.rst §"Early loading on UEFI".
//    • The kernel takes over from StartImage and never returns control.

pub fn boot_linux_stub(
    st:           &mut SystemTable<Boot>,
    image_handle: Handle,
    part:         &PartitionInfo,
    entry:        &EntryConfig,
) -> ! {
    info!("Linux-stub boot: {} ({})", entry.name, entry.kernel);

    // Kernel must reside on a FAT-family volume so the EFI stub can use
    // SimpleFileSystem to read initrd= targets. Non-FAT roots are blocked
    // because the stub cannot mount ext4/etc.
    match part.fstype.as_str() {
        "fat32" | "fat16" | "fat12" => {}
        other => die(st, "linux-stub", format!(
            "kernel partition fstype '{}' not readable by EFI stub — \
             move vmlinuz+initramfs to the ESP, or use type=linux-handover",
             other
        )),
    }

    // ── Construct command line ────────────────────────────────────────────
    let mut cmdline = entry.kernel_cmdline.clone();

    // Microcode MUST be the first initrd token.
    if let Some(ref uc) = entry.microcode {
        cmdline.push_str(&format!(" initrd={}", uefi_backslash(uc)));
    }
    for ird in &entry.initrd {
        cmdline.push_str(&format!(" initrd={}", uefi_backslash(ird)));
    }

    info!("  cmdline ({} chars): {}", cmdline.len(), cmdline);
    info!("  kernel path     : {}", entry.kernel);

    let ucs2 = utf16le_with_nul(cmdline.trim());

    // ── Open partition's DevicePath; keep guard alive across LoadImage ────
    let dp_guard = match st.boot_services().open_protocol_exclusive::<DevicePath>(part.handle) {
        Ok(g)  => g,
        Err(e) => die(st, "linux-stub", format!("partition DevicePath: {:?}", e)),
    };

    // ── Append a FilePath node onto the partition DP to point at vmlinuz ──
    //
    // Strategy: read the kernel image into a heap buffer, then LoadImage
    // FromBuffer with file_path set to the *partition* DP. The EFI stub
    // calls LoadedImageDevicePath() to discover the volume, and resolves
    // `initrd=\...` paths against the FILE_PROTOCOL it opens on that volume.
    //
    // This avoids the need to construct a synthetic FilePath device-path
    // node (which would require allocating + UTF-16 conversion + correct
    // length prefix and end-marker per UEFI §10.3.4.4). FromBuffer with
    // partition DP is the same convention systemd-boot uses.
    let kernel_bytes = match disk::read_fat_file(st.boot_services(), part.handle, &entry.kernel) {
        Ok(b)  => b,
        Err(e) => die(st, "linux-stub", format!("read kernel: {}", e)),
    };
    info!("  kernel image    : {} bytes", kernel_bytes.len());

    // PE signature sanity check.
    if kernel_bytes.len() < 0x40 || &kernel_bytes[..2] != b"MZ" {
        die(st, "linux-stub", "kernel image is not a PE/COFF binary \
                                   — try type=linux-handover instead");
    }

    let loaded = match st.boot_services().load_image(
        image_handle,
        LoadImageSource::FromBuffer {
            buffer:    &kernel_bytes,
            file_path: Some(&*dp_guard),
        },
    ) {
        Ok(h)  => h,
        Err(e) => die(st, "linux-stub", format!("load_image: {:?}", e)),
    };

    // dp_guard's lifetime is no longer needed past load_image — drop now.
    drop(dp_guard);
    drop(kernel_bytes); // The image has been copied into LOADER_CODE pages.

    // ── Attach LoadOptions ───────────────────────────────────────────────
    {
        let mut li_guard = match st.boot_services().open_protocol_exclusive::<LoadedImage>(loaded) {
            Ok(g)  => g,
            Err(e) => die(st, "linux-stub", format!("LoadedImage open: {:?}", e)),
        };
        // SAFETY: ucs2 lives until StartImage returns; LoadedImage stores a
        // raw pointer + size. The EFI stub consumes the buffer eagerly
        // (early in stub init) before its kernel main, but spec compliance
        // requires the buffer to remain valid for the lifetime of the
        // LoadedImage. We hold ucs2 in scope through start_image below.
        //
        // ScopedProtocol<LoadedImage> implements DerefMut → &mut LoadedImage,
        // so direct method invocation works.
        unsafe {
            li_guard.set_load_options(ucs2.as_ptr(), ucs2.len() as u32);
        }
    }

    // ── Start the kernel ──────────────────────────────────────────────────
    info!("  → StartImage");
    match st.boot_services().start_image(loaded) {
        Ok(_)  => warn!("kernel returned cleanly (unusual); unloading"),
        Err(e) => error!("kernel start failed: {:?}", e),
    }
    // If we reach here, the kernel did not take over the system.
    let _ = st.boot_services().unload_image(loaded);
    drop(ucs2);
    die(st, "linux-stub", "kernel returned to bootloader unexpectedly");
}

// ════════════════════════════════════════════════════════════════════════════
//  §B — boot_linux_handover  (legacy bzImage handover_offset path)
// ════════════════════════════════════════════════════════════════════════════
//
//  boot_params layout — selected offsets per arch/x86/boot/header.S:
//
//      0x1C0  efi_loader_signature[4]   "EL64"      ← was wrongly "BL36"
//      0x1C4  efi_systab                u32         ← low 32 bits
//      0x1C8  efi_memdesc_size          u32
//      0x1CC  efi_memdesc_version       u32
//      0x1D0  efi_memmap                u32         ← low 32 bits
//      0x1D4  efi_memmap_size           u32
//      0x1D8  efi_systab_hi             u32         ← high 32 bits
//      0x1DC  efi_memmap_hi             u32         ← high 32 bits
//      0x1F1  setup_sects               u8
//      0x210  type_of_loader            u8          ← we write 0xFF (other)
//      0x211  loadflags                 u8          ← LOADED_HIGH=0x01
//      0x218  ramdisk_image             u32         ← low 32 bits
//      0x21C  ramdisk_size              u32
//      0x228  cmd_line_ptr              u32         ← low 32 bits
//      0x234  kernel_alignment          u32
//      0x238  relocatable_kernel        u8
//      0x260  pref_address              u64
//      0x264  init_size                 u32
//      0x268  handover_offset           u32
//      0x26C  kernel_info_offset        u32
//
//  We do NOT call ExitBootServices: the handover spec says the kernel will.
//  The efi_memmap fields stay zero — kernel calls GetMemoryMap itself.
//
//  Calling convention at handover: Microsoft x64 (UEFI ABI)
//      RCX = image_handle
//      RDX = system_table_ptr
//      R8  = boot_params_ptr

const BP_EFI_INFO:            usize = 0x1C0;
const BP_SETUP_SECTS:         usize = 0x1F1;
const BP_TYPE_OF_LOADER:      usize = 0x210;
const BP_LOADFLAGS:           usize = 0x211;
const BP_CODE32_START:        usize = 0x214;
const BP_RAMDISK_IMAGE:       usize = 0x218;
const BP_RAMDISK_SIZE:        usize = 0x21C;
const BP_CMD_LINE_PTR:        usize = 0x228;
const BP_KERNEL_ALIGNMENT:    usize = 0x234;
const BP_RELOCATABLE_KERNEL:  usize = 0x238;
const BP_XLOADFLAGS:          usize = 0x23A;
const BP_CMDLINE_SIZE:        usize = 0x23C;
const BP_PREF_ADDRESS:        usize = 0x260;
const BP_INIT_SIZE:           usize = 0x268;
const BP_HANDOVER_OFFSET:     usize = 0x26C;
const BP_SIZE:                usize = 0x1000; // 4 KiB boot_params page

pub fn boot_linux_handover(
    st:           SystemTable<Boot>,
    image_handle: Handle,
    part:         &PartitionInfo,
    entry:        &EntryConfig,
) -> ! {
    info!("Linux-handover boot: {} ({})", entry.name, entry.kernel);


    // ── Load the bzImage into memory ──────────────────────────────────────
    let bz = match disk::read_file(st.boot_services(), part, &entry.kernel) {
        Ok(d)  => d,
        Err(e) => die(&st, "handover", format!("read kernel: {}", e)),
    };
    if bz.len() < 0x270 {
        die(&st, "handover", "bzImage too small (no setup header)");
    }
    if u16::from_le_bytes([bz[0x1FE], bz[0x1FF]]) != 0xAA55 {
        die(&st, "handover", "bzImage missing 0xAA55 boot signature");
    }
    if &bz[0x202..0x206] != b"HdrS" {
        die(&st, "handover", "bzImage missing 'HdrS' header magic");
    }

    let setup_sects = if bz[BP_SETUP_SECTS] == 0 { 4u8 } else { bz[BP_SETUP_SECTS] };
    let setup_bytes = (setup_sects as usize + 1) * 512;
    if setup_bytes > bz.len() {
        die(&st, "handover", "bzImage setup_sects truncated");
    }

    let init_size      = u32::from_le_bytes(bz[BP_INIT_SIZE      .. BP_INIT_SIZE+4].try_into().unwrap()) as usize;
    let pref_address   = u64::from_le_bytes(bz[BP_PREF_ADDRESS   .. BP_PREF_ADDRESS+8].try_into().unwrap());
    let handover_off   = u32::from_le_bytes(bz[BP_HANDOVER_OFFSET.. BP_HANDOVER_OFFSET+4].try_into().unwrap());
    let kernel_align   = u32::from_le_bytes(bz[BP_KERNEL_ALIGNMENT.. BP_KERNEL_ALIGNMENT+4].try_into().unwrap()).max(0x200000);
    let relocatable    = bz[BP_RELOCATABLE_KERNEL] != 0;

    if handover_off == 0 {
        die(&st, "handover",
            "kernel has no EFI handover entry (handover_offset == 0). \
             Use type=linux-stub for modern EFI-stub kernels.");
    }
    info!("  setup_sects={}  init_size=0x{:X}  handover_off=0x{:X}  pref=0x{:X}",
        setup_sects, init_size, handover_off, pref_address);

    // ── Allocate boot_params (4 KiB, must be < 4 GiB if loader sets only
    //    the low 32-bit half — but we set both halves, so anywhere is OK).
    let bp_pages: usize = (BP_SIZE + 4095) / 4096;
    let bp_addr = match st.boot_services().allocate_pages(
        AllocateType::AnyPages, MemoryType::LOADER_DATA, bp_pages,
    ) {
        Ok(a)  => a,
        Err(e) => die(&st, "handover", format!("alloc boot_params: {:?}", e)),
    };
    let bp = bp_addr as *mut u8;
    unsafe { ptr::write_bytes(bp, 0, BP_SIZE); }

    // Copy setup header from bzImage into boot_params (offset 0..setup_bytes).
    unsafe { ptr::copy_nonoverlapping(bz.as_ptr(), bp, setup_bytes); }

    // ── Allocate kernel destination ───────────────────────────────────────
    let kernel_size  = bz.len().saturating_sub(setup_bytes);
    let kernel_pages = ((init_size.max(kernel_size) + 4095) / 4096).max(1);

    // Choose destination: pref_address if relocatable=false; else honor it
    // when possible; else fall back to a high allocation respecting align.
    let kernel_dest = if pref_address != 0 {
        match st.boot_services().allocate_pages(
            AllocateType::Address(pref_address),
            MemoryType::LOADER_CODE,
            kernel_pages,
        ) {
            Ok(a) => a,
            Err(_) if relocatable => {
                // Allocate aligned anywhere and pad to alignment manually.
                let pad_pages = (kernel_align as usize / 4096).max(1);
                match st.boot_services().allocate_pages(
                    AllocateType::AnyPages, MemoryType::LOADER_CODE,
                    kernel_pages + pad_pages,
                ) {
                    Ok(raw) => {
                        let aligned = (raw + (kernel_align as u64 - 1))
                                    & !(kernel_align as u64 - 1);
                        aligned
                    }
                    Err(e) => die(&st, "handover", format!("alloc kernel: {:?}", e)),
                }
            }
            Err(e) => die(&st, "handover",
                format!("alloc at pref=0x{:X}: {:?}", pref_address, e)),
        }
    } else {
        match st.boot_services().allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_CODE, kernel_pages) {
            Ok(a)  => a,
            Err(e) => die(&st, "handover", format!("alloc kernel: {:?}", e)),
        }
    };

    // Zero kernel area, then copy protected-mode image.
    unsafe {
        ptr::write_bytes(kernel_dest as *mut u8, 0, kernel_pages * 4096);
        ptr::copy_nonoverlapping(
            bz.as_ptr().add(setup_bytes),
            kernel_dest as *mut u8,
            kernel_size,
        );
    }
    drop(bz);
    info!("  kernel @ 0x{:X} ({} pages)", kernel_dest, kernel_pages);

    // ── Build & attach cmdline (must be < 4 GiB unless XLF_CAN_BE_LOADED_ABOVE_4G) ──
    let xlf = u16::from_le_bytes(unsafe { *(bp.add(BP_XLOADFLAGS) as *const [u8; 2]) });
    let cmdline_size = u32::from_le_bytes(unsafe { *(bp.add(BP_CMDLINE_SIZE) as *const [u8; 4]) }) as usize;
    let cmdline_size = if cmdline_size == 0 { 4096 } else { cmdline_size.min(8192) };

    let cmdline_str = entry.kernel_cmdline.clone();
    let cmdline_bytes = cmdline_str.as_bytes();
    if cmdline_bytes.len() >= cmdline_size {
        die(&st, "handover", "cmdline exceeds kernel-declared cmdline_size");
    }

    // Place cmdline page just below the 4 GiB boundary for compatibility
    // with kernels that store only the low 32 bits of cmd_line_ptr.
    let cmdline_pages = ((cmdline_size + 4095) / 4096).max(1);
    let cmdline_addr  = if xlf & 0x0008 != 0 {
        // XLF_CAN_BE_LOADED_ABOVE_4G — anywhere is fine.
        match st.boot_services().allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, cmdline_pages) {
            Ok(a)  => a,
            Err(e) => die(&st, "handover", format!("alloc cmdline: {:?}", e)),
        }
    } else {
        match st.boot_services().allocate_pages(
            AllocateType::MaxAddress(0xFFFF_F000),
            MemoryType::LOADER_DATA,
            cmdline_pages,
        ) {
            Ok(a)  => a,
            Err(e) => die(&st, "handover", format!("alloc cmdline<4G: {:?}", e)),
        }
    };
    unsafe {
        ptr::write_bytes(cmdline_addr as *mut u8, 0, cmdline_pages * 4096);
        ptr::copy_nonoverlapping(
            cmdline_bytes.as_ptr(),
            cmdline_addr as *mut u8,
            cmdline_bytes.len(),
        );
    }

    // ── Load initramfs (with optional microcode-first concatenation) ──────
    //
    // The kernel sees a single contiguous initrd region. We compose it in
    // RAM by reading microcode + each declared initramfs file end-to-end.
    let mut ird_chunks: Vec<Vec<u8>> = Vec::new();
    if let Some(ref uc) = entry.microcode {
        match disk::read_file(st.boot_services(), part, uc) {
            Ok(d)  => ird_chunks.push(d),
            Err(e) => die(&st, "handover", format!("read microcode {}: {}", uc, e)),
        }
    }
    for ird in &entry.initrd {
        match disk::read_file(st.boot_services(), part, ird) {
            Ok(d)  => ird_chunks.push(d),
            Err(e) => die(&st, "handover", format!("read initrd {}: {}", ird, e)),
        }
    }
    let ird_total: usize = ird_chunks.iter().map(|c| c.len()).sum();

    let ird_addr: u64 = if ird_total == 0 {
        0
    } else {
        let ird_pages = ((ird_total + 4095) / 4096).max(1);
        let addr = if xlf & 0x0008 != 0 {
            st.boot_services().allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, ird_pages)
        } else {
            st.boot_services().allocate_pages(
                AllocateType::MaxAddress(0xFFFF_F000),
                MemoryType::LOADER_DATA,
                ird_pages,
            )
        }.unwrap_or_else(|e| die(&st, "handover", format!("alloc initrd: {:?}", e)));

        let mut off = 0usize;
        for c in &ird_chunks {
            unsafe {
                ptr::copy_nonoverlapping(
                    c.as_ptr(),
                    (addr as *mut u8).add(off),
                    c.len(),
                );
            }
            off += c.len();
        }
        addr
    };
    drop(ird_chunks);
    info!("  initrd @ 0x{:X} ({} bytes)", ird_addr, ird_total);

    // ── Patch boot_params header fields ───────────────────────────────────
    unsafe {
        // type_of_loader = 0xFF (other / 4-bit major+minor = 0xF,0xF)
        *bp.add(BP_TYPE_OF_LOADER) = 0xFF;
        // loadflags: bit0 LOADED_HIGH = 1
        *bp.add(BP_LOADFLAGS) |= 0x01;

        // code32_start: 32-bit kernel base
        ptr::copy_nonoverlapping(
            &(kernel_dest as u32).to_le_bytes() as *const _ as *const u8,
            bp.add(BP_CODE32_START), 4,
        );

        // cmd_line_ptr (low 32 bits) at 0x228
        ptr::copy_nonoverlapping(
            &(cmdline_addr as u32).to_le_bytes() as *const _ as *const u8,
            bp.add(BP_CMD_LINE_PTR), 4,
        );

        // ramdisk_image (low 32) / ramdisk_size
        ptr::copy_nonoverlapping(
            &(ird_addr as u32).to_le_bytes() as *const _ as *const u8,
            bp.add(BP_RAMDISK_IMAGE), 4,
        );
        ptr::copy_nonoverlapping(
            &(ird_total as u32).to_le_bytes() as *const _ as *const u8,
            bp.add(BP_RAMDISK_SIZE), 4,
        );
    }

    // ── Populate efi_info at offset 0x1C0 ─────────────────────────────────
    let systab_ptr = st.as_ptr() as u64;          // ACTUAL EFI_SYSTEM_TABLE*, not wrapper.
    unsafe {
        // efi_loader_signature[4] = "EL64"
        ptr::copy_nonoverlapping(b"EL64".as_ptr(), bp.add(BP_EFI_INFO),     4);
        // efi_systab (low 32)
        ptr::copy_nonoverlapping(
            &(systab_ptr as u32).to_le_bytes() as *const _ as *const u8,
            bp.add(BP_EFI_INFO + 0x04), 4,
        );
        // efi_memdesc_size, efi_memdesc_version, efi_memmap, efi_memmap_size
        // all left zero — kernel calls GetMemoryMap itself in the handover
        // path before ExitBootServices.
        ptr::write_bytes(bp.add(BP_EFI_INFO + 0x08), 0, 0x10);
        // efi_systab_hi
        ptr::copy_nonoverlapping(
            &((systab_ptr >> 32) as u32).to_le_bytes() as *const _ as *const u8,
            bp.add(BP_EFI_INFO + 0x18), 4,
        );
        // efi_memmap_hi
        ptr::write_bytes(bp.add(BP_EFI_INFO + 0x1C), 0, 4);
    }

    // ── Compute handover entry address ────────────────────────────────────
    //
    // Per Documentation/x86/boot.rst §EFI HANDOVER PROTOCOL:
    //   entry = kernel_dest + 0x200 + handover_offset
    //
    // The +0x200 skips the legacy real-mode setup_header at the start of
    // the protected-mode kernel image (which already lives in boot_params
    // copy now). The kernel's EFI handover stub then takes over.
    let handover_entry = kernel_dest + 0x200 + handover_off as u64;

    info!("  → handover entry 0x{:X}  bp 0x{:X}  systab 0x{:X}",
          handover_entry, bp_addr, systab_ptr);

    // ── Transfer control (MS x64 ABI, NO exit_boot_services) ──────────────
    //
    // The handover ABI is the same as the standard UEFI image entry:
    //   extern "win64" fn(handle, systab, bp) -> !
    // The kernel calls ExitBootServices itself.
    type HandoverFn = unsafe extern "win64" fn(Handle, *const c_void, *mut c_void) -> !;
    let entry_fn: HandoverFn = unsafe { mem::transmute(handover_entry as *const ()) };

    // Note: we deliberately do NOT drop `st` — the kernel needs the
    // system table pointer to remain at `systab_ptr` for the duration
    // of its boot-services-using window. Letting Rust drop it would
    // not actually free anything (UEFI allocations aren't tied to
    // Rust ownership) but we keep it for clarity.
    unsafe { entry_fn(image_handle, systab_ptr as *const c_void, bp as *mut c_void); }
    // unreachable; entry_fn is `!`.
}

// ════════════════════════════════════════════════════════════════════════════
//  §C — boot_aios  (AIOS native ELF + LOADER_PARAMS handover)
// ════════════════════════════════════════════════════════════════════════════
//
//  LOADER_PARAMS (Custom-OS-Manual §3 v2.3, 80 bytes, 8-byte aligned):
//      0x00  u64  mmap_total_size
//      0x08  u64  mmap_desc_size
//      0x10  u64  fb_base
//      0x18  u64  fb_width
//      0x20  u64  fb_height
//      0x28  u64  fb_stride
//      0x30  u64  config_table_ptr
//      0x38  u64  kernel_base_addr
//      0x40  u64  kernel_pages
//      0x48  u32  uefi_version             (major<<16) | minor
//      0x4C  u32  esp_root_size
//
//  Convention: the memory map IS NOT pointed to by LOADER_PARAMS — it
//  immediately follows the LP block at offset 0x50 within the same
//  allocation. The kernel computes the map base as `(uint8_t*)LP + 0x50`.
//  This matches the field set defined in `aios_process.py::LoaderParams`
//  exactly (we add nothing the kernel doesn't expect).

const LP_SIZE:                 usize = 0x50;       // 80 bytes, packed
const LP_MMAP_RESERVED_BYTES:  usize = 0x1_0000;   // 64 KiB scratch for memory map
                                                   // (UEFI memmaps are typically 4-16 KiB)
const LP_TOTAL_BYTES:          usize = LP_SIZE + LP_MMAP_RESERVED_BYTES;

const ELF_PT_LOAD: u32 = 1;

pub fn boot_aios(
    st:           SystemTable<Boot>,
    _image_handle: Handle,
    part:         &PartitionInfo,
    entry:        &EntryConfig,
) -> ! {
    info!("AIOS-native boot: {} ({})", entry.name, entry.kernel);


    // ── Read & validate ELF64 ─────────────────────────────────────────────
    let elf = match disk::read_file(st.boot_services(), part, &entry.kernel) {
        Ok(d)  => d,
        Err(e) => die(&st, "aios", format!("read ELF: {}", e)),
    };
    if elf.len() < 64                                   { die(&st, "aios", "ELF truncated"); }
    if &elf[0..4]  != b"\x7FELF"                        { die(&st, "aios", "bad ELF magic"); }
    if elf[4]      != 2                                 { die(&st, "aios", "not ELF64"); }
    if elf[5]      != 1                                 { die(&st, "aios", "not little-endian"); }
    if u16::from_le_bytes([elf[18], elf[19]]) != 0x3E   { die(&st, "aios", "not AMD64 ELF"); }

    let e_entry      = u64::from_le_bytes(elf[24..32].try_into().unwrap());
    let e_phoff      = u64::from_le_bytes(elf[32..40].try_into().unwrap()) as usize;
    let e_phentsize  = u16::from_le_bytes([elf[54], elf[55]]) as usize;
    let e_phnum      = u16::from_le_bytes([elf[56], elf[57]]) as usize;

    if e_phentsize != 56                                { die(&st, "aios", "phentsize != 56"); }
    if e_phoff.saturating_add(e_phentsize * e_phnum) > elf.len() {
        die(&st, "aios", "program headers out of file bounds");
    }
    info!("  ELF entry=0x{:X}  phnum={}", e_entry, e_phnum);

    // First pass: compute footprint across all PT_LOAD segments.
    let (mut min_pa, mut max_pa) = (u64::MAX, 0u64);
    for i in 0..e_phnum {
        let o = e_phoff + i * e_phentsize;
        let p_type = u32::from_le_bytes(elf[o..o+4].try_into().unwrap());
        if p_type != ELF_PT_LOAD { continue; }
        let p_paddr = u64::from_le_bytes(elf[o+24..o+32].try_into().unwrap());
        let p_memsz = u64::from_le_bytes(elf[o+40..o+48].try_into().unwrap());
        min_pa = min_pa.min(p_paddr);
        max_pa = max_pa.max(p_paddr.saturating_add(p_memsz));
    }
    if min_pa == u64::MAX { die(&st, "aios", "no PT_LOAD segments"); }
    let kernel_span_pages = (((max_pa - min_pa) + 4095) / 4096) as usize;

    // ── Allocate kernel image at requested physical address ───────────────
    //
    // Strategy: try exact-address allocation at min_pa first. If the
    // firmware has reserved that range, fall back to AnyPages and
    // relocate (offset all p_paddr by `offset = alloc_base - min_pa`).
    let (kernel_base, reloc_offset) = match st.boot_services().allocate_pages(
        AllocateType::Address(min_pa),
        MemoryType::LOADER_DATA,
        kernel_span_pages,
    ) {
        Ok(_)  => (min_pa, 0i64),
        Err(_) => {
            warn!("  exact alloc at 0x{:X} failed — relocating", min_pa);
            let a = st.boot_services().allocate_pages(
                AllocateType::AnyPages, MemoryType::LOADER_DATA, kernel_span_pages,
            ).unwrap_or_else(|e| die(&st, "aios",
                format!("alloc {} pages: {:?}", kernel_span_pages, e)));
            (a, a as i64 - min_pa as i64)
        }
    };
    // Zero the entire kernel destination so PT_LOAD's p_memsz > p_filesz tail
    // (i.e. BSS) is implicitly zeroed without a per-segment fill.
    unsafe {
        ptr::write_bytes(kernel_base as *mut u8, 0, kernel_span_pages * 4096);
    }

    // Second pass: copy PT_LOAD segments.
    for i in 0..e_phnum {
        let o = e_phoff + i * e_phentsize;
        let p_type = u32::from_le_bytes(elf[o..o+4].try_into().unwrap());
        if p_type != ELF_PT_LOAD { continue; }
        let p_offset = u64::from_le_bytes(elf[o+ 8..o+16].try_into().unwrap()) as usize;
        let p_paddr  = u64::from_le_bytes(elf[o+24..o+32].try_into().unwrap());
        let p_filesz = u64::from_le_bytes(elf[o+32..o+40].try_into().unwrap()) as usize;
        let p_memsz  = u64::from_le_bytes(elf[o+40..o+48].try_into().unwrap()) as usize;
        if p_filesz > p_memsz {
            die(&st, "aios", "PT_LOAD: filesz > memsz (corrupt ELF)");
        }
        if p_offset.saturating_add(p_filesz) > elf.len() {
            die(&st, "aios", "PT_LOAD: file region out of bounds");
        }
        let dst = ((p_paddr as i64) + reloc_offset) as u64 as *mut u8;
        unsafe {
            ptr::copy_nonoverlapping(
                elf.as_ptr().add(p_offset), dst, p_filesz,
            );
        }
    }
    drop(elf);

    let real_entry = ((e_entry as i64) + reloc_offset) as u64;
    info!("  kernel @ 0x{:X} ({} pages)  entry=0x{:X}",
          kernel_base, kernel_span_pages, real_entry);

    // ── Framebuffer (must query GOP BEFORE ExitBootServices) ──────────────
    let fb = gop::discover_framebuffer(st.boot_services());

    // ── Capture config-table pointer and UEFI revision BEFORE exit ────────
    let config_table_ptr  = st.config_table().as_ptr() as u64;
    let uefi_rev          = st.uefi_revision();
    // UEFI Revision encoding: high 16 bits = major, low 16 = minor (×10).
    let uefi_version: u32 = ((uefi_rev.major() as u32) << 16) | (uefi_rev.minor() as u32);

    // ── Pre-allocate LOADER_PARAMS block + memmap scratch (contiguous) ────
    let lp_pages = (LP_TOTAL_BYTES + 4095) / 4096;
    let lp_base  = match st.boot_services().allocate_pages(
        AllocateType::AnyPages, MemoryType::LOADER_DATA, lp_pages,
    ) {
        Ok(a)  => a,
        Err(e) => die(&st, "aios", format!("alloc LOADER_PARAMS: {:?}", e)),
    };
    unsafe { ptr::write_bytes(lp_base as *mut u8, 0, lp_pages * 4096); }

    info!("  LP block @ 0x{:X} ({} pages reserved)", lp_base, lp_pages);
    info!("  → ExitBootServices");

    // ── Exit Boot Services (consumes SystemTable<Boot>) ───────────────────
    //
    // uefi 0.28 API: `exit_boot_services(MemoryType)` internally retries
    // GetMemoryMap+ExitBootServices on map_key staleness, and returns
    // (SystemTable<Runtime>, MemoryMap) on success.
    //
    // SAFETY: from this point onward we MUST NOT:
    //   • allocate via Rust's global allocator (uefi::helpers wired it to BS pool)
    //   • call log::info!/warn!/error! (helpers wired logging to ConOut)
    //   • panic (helpers' panic handler uses BS)
    //
    // We perform only raw pointer writes and the final jump.
    let (_rt_st, mmap) = st.exit_boot_services(MemoryType::LOADER_DATA);

    // ── Serialize memory map directly after LOADER_PARAMS ─────────────────
    // uefi 0.28 MemoryMap exposes entries() but not the firmware desc_size.
    // We re-emit a packed array whose stride is the Rust MemoryDescriptor
    // size and report THAT stride to the kernel — fully self-consistent and
    // avoiding any read past each descriptor's well-defined bytes.
    let mmap_desc_size = core::mem::size_of::<uefi::table::boot::MemoryDescriptor>();
    let mmap_dst       = (lp_base as usize + LP_SIZE) as *mut u8;
    let mut written: usize = 0;

    for desc in mmap.entries() {
        if written + mmap_desc_size > LP_MMAP_RESERVED_BYTES {
            // Out of reserved space — best we can do is truncate. The kernel
            // will read mmap_total_size = written and miss the tail. The
            // 64 KiB scratch is sized to hold any plausible PC memmap.
            break;
        }
        unsafe {
            ptr::copy_nonoverlapping(
                desc as *const _ as *const u8,
                mmap_dst.add(written),
                mmap_desc_size,
            );
        }
        written += mmap_desc_size;
    }
    let mmap_total_size = written as u64;

    // ── Write LOADER_PARAMS fields ────────────────────────────────────────
    unsafe {
        let lp = lp_base as *mut u8;
        write_u64_le(lp, 0x00, mmap_total_size);
        write_u64_le(lp, 0x08, mmap_desc_size as u64);
        write_u64_le(lp, 0x10, fb.base);
        write_u64_le(lp, 0x18, fb.width  as u64);
        write_u64_le(lp, 0x20, fb.height as u64);
        write_u64_le(lp, 0x28, fb.stride as u64);
        write_u64_le(lp, 0x30, config_table_ptr);
        write_u64_le(lp, 0x38, kernel_base);
        write_u64_le(lp, 0x40, kernel_span_pages as u64);
        write_u32_le(lp, 0x48, uefi_version);
        write_u32_le(lp, 0x4C, 0); // esp_root_size: not measured in V1.
    }

    // ── Transfer control to AIOS kernel (SysV AMD64 ABI: RDI = LP*) ───────
    //
    // The Rust compiler emits the win64→sysv64 ABI transition for us:
    // RCX→RDI is the explicit move it inserts. After this call we never
    // return (`!`), so callee-saved register preservation does not matter.
    type AiosEntry = unsafe extern "sysv64" fn(*mut c_void) -> !;
    let entry_fn: AiosEntry = unsafe { mem::transmute(real_entry as *const ()) };
    unsafe { entry_fn(lp_base as *mut c_void); }
}

#[inline(always)]
unsafe fn write_u64_le(base: *mut u8, off: usize, v: u64) {
    ptr::copy_nonoverlapping(
        &v.to_le_bytes() as *const _ as *const u8,
        base.add(off),
        8,
    );
}

#[inline(always)]
unsafe fn write_u32_le(base: *mut u8, off: usize, v: u32) {
    ptr::copy_nonoverlapping(
        &v.to_le_bytes() as *const _ as *const u8,
        base.add(off),
        4,
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  §D — chainload_efi  (arbitrary EFI binary)
// ════════════════════════════════════════════════════════════════════════════
//
//  Use case: Windows Boot Manager (\EFI\Microsoft\Boot\bootmgfw.efi),
//  shim, GRUB, systemd-boot, rEFInd — anything that exposes itself as
//  a PE/COFF EFI application on a FAT volume.
//
//  Lifetime safety: the DevicePath guard MUST remain alive across the
//  load_image call. Reading `&*dp_guard` and then drop()ing the guard
    // See UEFI Specification Version 2.9, Section 7.4.
    // Note: The specific location where we unhook the memory map
    // before load_image returns is the bug the prior implementation had.
pub fn chainload_efi(
    st:           &mut SystemTable<Boot>,
    image_handle: Handle,
    part_handle:  Handle,
    path:         &str,
) -> ! {
    info!("Chainloading {} ...", path);

    let dp_guard = match st.boot_services().open_protocol_exclusive::<DevicePath>(part_handle) {
        Ok(g)  => g,
        Err(e) => die(st, "chainload", format!("DevicePath open: {:?}", e)),
    };

    // For chainload, we read the target into a buffer and use FromBuffer
    // with file_path = partition DP. This means we don't need to construct
    // a synthetic FilePath device-path node manually.
    //
    // Read first to verify the path exists; bubble a useful error otherwise.
    let img_bytes = {
        // We must use FAT here: chainload targets live on the ESP, by
        // construction (UEFI firmware boots only from FAT-family volumes).
        let bs = st.boot_services();
        match disk::read_fat_file(bs, part_handle, path) {
            Ok(b)  => b,
            Err(e) => {
                drop(dp_guard);
                die(st, "chainload", format!("read {}: {}", path, e));
            }
        }
    };

    let loaded = match st.boot_services().load_image(
        image_handle,
        LoadImageSource::FromBuffer {
            buffer:    &img_bytes,
            file_path: Some(&*dp_guard),
        },
    ) {
        Ok(h)  => h,
        Err(e) => {
            drop(dp_guard);
            die(st, "chainload", format!("load_image {}: {:?}", path, e));
        }
    };
    drop(dp_guard);
    drop(img_bytes);

    match st.boot_services().start_image(loaded) {
        Ok(s)  => info!("Chain image exited: {:?}", s),
        Err(e) => warn!("start_image: {:?}", e),
    }
    let _ = st.boot_services().unload_image(loaded);
    die(st, "chainload", "chain image returned");
}

/// Like `chainload_efi` but returns Status on failure instead of rebooting.
/// Used by the fallback chainload search at boot time.
pub fn try_chainload(
    st:           &mut SystemTable<Boot>,
    image_handle: Handle,
    part_handle:  Handle,
    path:         &str,
) -> Result<Status, Status> {
    let dp_guard = st.boot_services()
        .open_protocol_exclusive::<DevicePath>(part_handle)
        .map_err(|_| Status::NOT_FOUND)?;

    let img_bytes = disk::read_fat_file(st.boot_services(), part_handle, path)
        .map_err(|_| Status::NOT_FOUND)?;

    let loaded = match st.boot_services().load_image(
        image_handle,
        LoadImageSource::FromBuffer {
            buffer:    &img_bytes,
            file_path: Some(&*dp_guard),
        },
    ) {
        Ok(h)  => h,
        Err(e) => return Err(e.status()),
    };
    drop(dp_guard);

    st.boot_services().start_image(loaded)
        .map_err(|e| { let _ = st.boot_services().unload_image(loaded); e.status() })?;
    Ok(Status::SUCCESS)
}

// ════════════════════════════════════════════════════════════════════════════
//  §E — Module-level guard against accidental misuse
// ════════════════════════════════════════════════════════════════════════════

/// Compile-time sanity: LOADER_PARAMS field offsets used by `write_u64_le`/
/// `write_u32_le` in `boot_aios`. Mismatches against `aios_process.py`'s
/// `LoaderParams._PACK_FMT = "<9Q2I"` would produce a silently-wrong
/// handover. We assert the total size statically.
const _: () = assert!(LP_SIZE == 0x50);
const _: () = assert!(LP_SIZE == 9*8 + 2*4);
