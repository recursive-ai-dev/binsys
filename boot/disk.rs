//! PuppyBoot V1 — Unified disk I/O.
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  Two filesystem backends, one dispatch surface:
//!    • FAT12/16/32 — UEFI SimpleFileSystem + File protocols (firmware driver).
//!    • ext2/3/4    — pure-Rust Ext4FS over raw BlockIO reads (no firmware
//!                    ext driver exists; we parse the superblock, inode
//!                    table, and extent tree ourselves).
//!
//!  Public entry points (called by boot.rs / main.rs / menu.rs):
//!    read_file(bs, part, path)        → dispatch on part.fstype
//!    read_fat_file(bs, handle, path)  → FAT only (used where the caller
//!                                       already knows the volume is FAT,
//!                                       e.g. EFI-stub kernels and chainload)
//!    list_fat_dir(bs, handle, dir)    → directory enumeration on FAT
//!    write_fat_file(bs, handle, p, d) → install-time writes (not boot path)
//!
//!  Path convention: callers pass POSIX-style paths ("/EFI/arch/vmlinuz");
//!  uefi_path() translates '/' → '\' and builds the CString16 that
//!  File::open requires. ext4 lookup keeps '/' (its own parser splits on it).

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use uefi::proto::media::file::{File, FileAttribute, FileInfo, FileMode};
use uefi::proto::media::fs::SimpleFileSystem;
use uefi::table::boot::BootServices;
use uefi::{CString16, Handle};

use crate::partition::{self, PartitionInfo};

// ════════════════════════════════════════════════════════════════════════════
//  §1 — Path helper
// ════════════════════════════════════════════════════════════════════════════

/// Translate a POSIX-style path to a UEFI `CString16` with backslash
/// separators. `File::open` requires `&CStr16`; UEFI paths use '\'.
fn uefi_path(path: &str) -> Result<CString16, String> {
    let translated: String = path.chars()
        .map(|c| if c == '/' { '\\' } else { c })
        .collect();
    CString16::try_from(translated.as_str())
        .map_err(|e| format!("CString16('{}'): {:?}", path, e))
}

// ════════════════════════════════════════════════════════════════════════════
//  §2 — Dispatch
// ════════════════════════════════════════════════════════════════════════════

/// Read an entire file from any supported partition, dispatching on fstype.
pub fn read_file(bs: &BootServices, part: &PartitionInfo, path: &str) -> Result<Vec<u8>, String> {
    match part.fstype.as_str() {
        "fat32" | "fat16" | "fat12" => read_fat_file(bs, part.handle, path),
        "ext4"  | "ext3"  | "ext2"  => read_ext_file(bs, part, path),
        other => Err(format!("unsupported filesystem '{}' for read of {}", other, path)),
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §3 — FAT (SimpleFileSystem + File protocol)
// ════════════════════════════════════════════════════════════════════════════

pub fn read_fat_file(bs: &BootServices, part_handle: Handle, path: &str) -> Result<Vec<u8>, String> {
    let mut fs = bs
        .open_protocol_exclusive::<SimpleFileSystem>(part_handle)
        .map_err(|e| format!("SimpleFileSystem open: {:?}", e))?;

    let mut root = fs
        .open_volume()
        .map_err(|e| format!("open_volume: {:?}", e))?;

    let cpath = uefi_path(path)?;

    let mut file = root
        .open(&cpath, FileMode::Read, FileAttribute::empty())
        .map_err(|e| format!("open '{}': {:?}", path, e))?
        .into_regular_file()
        .ok_or_else(|| format!("'{}' is not a regular file", path))?;

    // FileInfo header is variable-length (it embeds the file name); 512 bytes
    // is comfortably larger than the fixed prefix plus any plausible name.
    let mut info_buf = [0u8; 512];
    let info = file
        .get_info::<FileInfo>(&mut info_buf)
        .map_err(|e| format!("get_info '{}': {:?}", path, e))?;

    let size = info.file_size() as usize;
    let mut buf = vec![0u8; size];
    let read = file
        .read(&mut buf)
        .map_err(|e| format!("read '{}': {:?}", path, e.status()))?;
    buf.truncate(read);
    Ok(buf)
}

/// List filenames in a FAT directory (names only, no path prefix).
/// Returns an empty Vec (not an error) when the directory is absent, so
/// callers scanning optional config dirs don't need to special-case it.
pub fn list_fat_dir(bs: &BootServices, part_handle: Handle, dir_path: &str) -> Result<Vec<String>, String> {
    let mut fs = bs
        .open_protocol_exclusive::<SimpleFileSystem>(part_handle)
        .map_err(|e| format!("SimpleFileSystem open: {:?}", e))?;

    let mut root = fs
        .open_volume()
        .map_err(|e| format!("open_volume: {:?}", e))?;

    let cpath = uefi_path(dir_path)?;

    let dir_handle = match root.open(&cpath, FileMode::Read, FileAttribute::empty()) {
        Ok(h)  => h,
        Err(_) => return Ok(Vec::new()), // directory does not exist → empty
    };
    let mut dir = dir_handle
        .into_directory()
        .ok_or_else(|| format!("'{}' is not a directory", dir_path))?;

    let mut names     = Vec::new();
    let mut entry_buf = [0u8; 512];

    loop {
        match dir.read_entry(&mut entry_buf) {
            Ok(Some(entry)) => {
                let name = entry.file_name().to_string();
                if name != "." && name != ".." && !name.is_empty() {
                    names.push(name);
                }
            }
            Ok(None) => break,
            Err(e)   => return Err(format!("read_entry '{}': {:?}", dir_path, e.status())),
        }
    }

    names.sort();
    Ok(names)
}

// ════════════════════════════════════════════════════════════════════════════
//  §4 — ext2/3/4 (raw BlockIO through Ext4FS)
// ════════════════════════════════════════════════════════════════════════════

const EXT4_ROOT_INO: u32 = 2;
const S_IFREG:       u16 = 0x8000;

fn read_ext_file(bs: &BootServices, part: &PartitionInfo, path: &str) -> Result<Vec<u8>, String> {
    let part_handle = part.handle;

    // BlockReader closure adapts partition::read_blocks to the signature
    // Ext4FS expects: Fn(lba, count) -> Option<Vec<u8>>. Each call opens
    // the BlockIO protocol briefly; the firmware caches the device, so the
    // per-call overhead is negligible relative to media latency.
    let reader = move |lba: u64, count: usize| -> Option<Vec<u8>> {
        partition::read_blocks(bs, part_handle, lba, count)
    };

    let fs = crate::ext4::Ext4FS::open(&reader)
        .ok_or_else(|| "ext4: superblock open failed".to_string())?;

    let target_ino = fs
        .lookup_path(&reader, EXT4_ROOT_INO, path)
        .ok_or_else(|| format!("ext4: path not found: {}", path))?;

    let target_inode = fs
        .read_inode(&reader, target_ino)
        .ok_or_else(|| format!("ext4: cannot read inode {} for {}", target_ino, path))?;

    if crate::ext4::Ext4FS::inode_mode(&target_inode) & S_IFREG == 0 {
        return Err(format!("ext4: not a regular file: {}", path));
    }

    fs.read_entire_file(&reader, &target_inode)
        .ok_or_else(|| format!("ext4: read failed: {}", path))
}

// ════════════════════════════════════════════════════════════════════════════
//  §5 — FAT writes (install tooling; never on the boot path)
// ════════════════════════════════════════════════════════════════════════════

pub fn write_fat_file(bs: &BootServices, part_handle: Handle, path: &str, data: &[u8]) -> Result<(), String> {
    let mut fs = bs
        .open_protocol_exclusive::<SimpleFileSystem>(part_handle)
        .map_err(|e| format!("SimpleFileSystem open: {:?}", e))?;

    let mut root = fs
        .open_volume()
        .map_err(|e| format!("open_volume: {:?}", e))?;

    if let Some(slash) = path.rfind('/') {
        let parent = &path[..slash];
        if !parent.is_empty() {
            ensure_dirs(&mut root, parent);
        }
    }

    let cpath = uefi_path(path)?;
    let mut file = root
        .open(&cpath, FileMode::CreateReadWrite, FileAttribute::empty())
        .map_err(|e| format!("create '{}': {:?}", path, e))?
        .into_regular_file()
        .ok_or_else(|| format!("'{}' is not a regular file", path))?;

    file.write(data).map_err(|e| format!("write '{}': {:?}", path, e.status()))?;
    file.flush().map_err(|e| format!("flush '{}': {:?}", path, e.status()))?;
    Ok(())
}

fn ensure_dirs(root: &mut uefi::proto::media::file::Directory, path: &str) {
    // Re-open from root for each cumulative component so we never hold two
    // mutable directory handles at once (UEFI File is single-threaded and
    // chained borrows would conflict).
    let mut cumulative = String::new();
    for component in path.split('/').filter(|c| !c.is_empty()) {
        cumulative.push('/');
        cumulative.push_str(component);
        let cpath = match uefi_path(&cumulative) {
            Ok(p)  => p,
            Err(_) => continue,
        };
        if root.open(&cpath, FileMode::Read, FileAttribute::DIRECTORY).is_err() {
            let _ = root.open(&cpath, FileMode::CreateReadWrite, FileAttribute::DIRECTORY);
        }
    }
}
