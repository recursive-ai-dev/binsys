//! PuppyBoot V1 — Partition discovery.
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  STRATEGY  (UEFI 2.7+ §13.18 EFI_PARTITION_INFO_PROTOCOL)
//!  ────────────────────────────────────────────────────────────────────────────
//!  UEFI 2.7+ exposes `EFI_PARTITION_INFO_PROTOCOL` on every partition handle,
//!  returning the firmware-parsed GPT (or MBR) entry directly. PuppyBoot V1
//!  uses this as its sole partition-metadata source — no manual GPT header
//!  parsing, no CRC32 validation of the entry array, and crucially no
//!  parent-disk resolution.
//!
//!  This sidesteps the most serious bug in the prior PuppyBoot:
//!  `find_parent_disk` returned the *first* whole-disk handle in the system.
//!  On any machine with two disks (USB installer + internal NVMe — the exact
//!  situation during an Arch install) every partition bound to the wrong
//!  disk and read a foreign GPT. The PartitionInfo protocol is resolved by
//!  the firmware against the correct parent, so the bug cannot recur.
//!
//!  Filesystem type and filesystem-level UUID are obtained from a direct
//!  4 KiB sniff of the partition's first sectors (the protocol gives the
//!  PARTUUID, not the filesystem UUID, so we still read the superblock for
//!  the latter — both are useful for matching config `part_uuid =` lines).
//!
//!  Scope: GPT only. MBR layouts are effectively extinct on UEFI-class
//!  x86_64 hardware (post-2014). Supporting them would add parsing surface
//!  for negligible real-world benefit on the target hardware.

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use log::{info, warn};

use uefi::prelude::*;
use uefi::proto::media::block::BlockIO;
use uefi::proto::media::partition::PartitionInfo as EfiPartInfo;
use uefi::table::boot::{BootServices, SearchType};

// ════════════════════════════════════════════════════════════════════════════
//  §1 — Standard GPT type GUIDs  (UEFI §5.3.3, mixed-endian on-disk form)
// ════════════════════════════════════════════════════════════════════════════
//
//  These are compared against `Guid::to_bytes()`, which yields the canonical
//  on-disk mixed-endian byte order: data1 (u32 LE), data2 (u16 LE),
//  data3 (u16 LE), data4 (8 bytes BE). Each constant below is written in
//  that exact byte order so a flat `==` comparison is correct.

/// EFI System Partition — C12A7328-F81F-11D2-BA4B-00A0C93EC93B
const ESP_GUID: [u8; 16] = [
    0x28, 0x73, 0x2A, 0xC1, 0x1F, 0xF8, 0xD2, 0x11,
    0xBA, 0x4B, 0x00, 0xA0, 0xC9, 0x3E, 0xC9, 0x3B,
];

/// Linux filesystem data — 0FC63DAF-8483-4772-8E79-3D69D8477DE4
const LINUX_FS_GUID: [u8; 16] = [
    0xAF, 0x3D, 0xC6, 0x0F, 0x83, 0x84, 0x72, 0x47,
    0x8E, 0x79, 0x3D, 0x69, 0xD8, 0x47, 0x7D, 0xE4,
];

/// Linux x86-64 root (auto-discoverable) — 4F68BCE3-E8CD-4DB1-96E7-FBCAF984B709
const LINUX_ROOT_X64_GUID: [u8; 16] = [
    0xE3, 0xBC, 0x68, 0x4F, 0xCD, 0xE8, 0xB1, 0x4D,
    0x96, 0xE7, 0xFB, 0xCA, 0xF9, 0x84, 0xB7, 0x09,
];

/// Linux /boot extended-boot (XBOOTLDR) — BC13C2FF-59E6-4262-A352-B275FD6F7172
const LINUX_XBOOTLDR_GUID: [u8; 16] = [
    0xFF, 0xC2, 0x13, 0xBC, 0xE6, 0x59, 0x62, 0x42,
    0xA3, 0x52, 0xB2, 0x75, 0xFD, 0x6F, 0x71, 0x72,
];

// ════════════════════════════════════════════════════════════════════════════
//  §2 — Public record
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PartitionInfo {
    /// UEFI handle exposing BlockIO and (for FAT) SimpleFileSystem.
    pub handle:         Handle,
    /// 1-based GPT partition number.
    pub part_num:       u32,
    /// Unique partition GUID — this is the PARTUUID systemd/Linux use.
    pub gpt_guid:       String,
    /// Filesystem-level UUID (ext{2,3,4} UUID, or FAT volume serial).
    pub fs_uuid:        String,
    /// GPT partition entry name (label).
    pub part_label:     String,
    /// "fat32" | "fat16" | "fat12" | "ext4" | "ext3" | "ext2" | "ntfs" | "unknown".
    pub fstype:         String,
    pub is_esp:         bool,
    pub is_linux:       bool,
    pub first_lba:      u64,
    pub last_lba:       u64,
    /// Convention prefix for PuppyBoot configs stored on this volume.
    pub mount_efi_path: String,
}

impl PartitionInfo {
    pub fn describe(&self) -> String {
        format!(
            "#{:<2} {} fs={:<7} esp={} linux={} uuid={} label='{}'",
            self.part_num, self.gpt_guid, self.fstype,
            self.is_esp, self.is_linux, self.fs_uuid, self.part_label
        )
    }

    /// Match against either the PARTUUID or the filesystem UUID,
    /// case-insensitive and dash-insensitive. This lets a config line
    /// `part_uuid = 1234ABCD-...` match whether the user copied the
    /// PARTUUID (from `lsblk -o PARTUUID`) or the FS UUID (`blkid`).
    pub fn matches_uuid(&self, needle: &str) -> bool {
        let norm = |s: &str| s.replace('-', "").to_ascii_lowercase();
        let n = norm(needle);
        if n.is_empty() { return false; }
        norm(&self.gpt_guid) == n || norm(&self.fs_uuid) == n
    }

    /// Match against the GPT partition label (case-insensitive).
    pub fn matches_label(&self, needle: &str) -> bool {
        !needle.is_empty() && self.part_label.eq_ignore_ascii_case(needle)
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §3 — Discovery
// ════════════════════════════════════════════════════════════════════════════

pub fn discover_partitions(bs: &BootServices) -> Result<Vec<PartitionInfo>, Status> {
    let handles = bs.locate_handle_buffer(SearchType::from_proto::<BlockIO>())
        .map_err(|e| e.status())?;
    let mut results = Vec::with_capacity(handles.len());

    for h in handles.iter() {
        match build_partition_info(bs, *h) {
            Ok(Some(mut p)) => {
                p.part_num = (results.len() + 1) as u32;
                info!("Partition: {}", p.describe());
                results.push(p);
            }
            Ok(None) => {} // whole-disk handle, non-GPT, or unreadable
            Err(e)   => warn!("Handle skipped: {}", e),
        }
    }

    // Deterministic enumeration order on every boot: ESPs first, then by
    // GPT partition number. The menu and default-entry resolution both
    // depend on a stable order.
    results.sort_by(|a, b| {
        b.is_esp.cmp(&a.is_esp).then(a.part_num.cmp(&b.part_num))
    });

    Ok(results)
}

fn build_partition_info(bs: &BootServices, h: Handle) -> Result<Option<PartitionInfo>, String> {
    // ── Gate on logical_partition via BlockIO media ───────────────────────
    // uefi 0.28's BlockIOMedia exposes `is_logical_partition()` but not a
    // partition number. We assign a synthetic 1-based sequence number in
    // discover_partitions (used only for display/sort; matching is by
    // UUID/label, never by number).
    {
        let bio_proto = bs
            .open_protocol_exclusive::<BlockIO>(h)
            .map_err(|e| format!("BlockIO open: {:?}", e))?;
        let bio = &*bio_proto;
        if !bio.media().is_logical_partition() {
            return Ok(None);
        }
    }

    let mut info = PartitionInfo {
        handle:         h,
        part_num:       0, // assigned by caller
        gpt_guid:       String::new(),
        fs_uuid:        String::new(),
        part_label:     String::new(),
        fstype:         "unknown".into(),
        is_esp:         false,
        is_linux:       false,
        first_lba:      0,
        last_lba:       0,
        mount_efi_path: "/EFI/puppyboot".into(),
    };

    // ── PRIMARY: EFI_PARTITION_INFO_PROTOCOL ──────────────────────────────
    if let Ok(part_proto) = bs.open_protocol_exclusive::<EfiPartInfo>(h) {
        let efi_pi = &*part_proto;
        if let Some(gpt) = efi_pi.gpt_partition_entry() {
            // `gpt` references a #[repr(packed)] struct; taking &field of a
            // packed struct is unaligned UB, so we copy each Copy field out
            // by value first (a by-value read of a Copy field is permitted
            // and lowers to an unaligned load).
            let unique     = gpt.unique_partition_guid;   // Guid: Copy
            let type_guid  = gpt.partition_type_guid;     // GptPartitionType: Copy
            let name       = gpt.partition_name;          // [Char16; 36]: Copy
            let start_lba  = gpt.starting_lba;            // u64
            let end_lba    = gpt.ending_lba;              // u64

            info.gpt_guid   = format!("{}", unique);
            info.part_label = decode_char16_array(&name);
            info.first_lba  = start_lba;
            info.last_lba   = end_lba;

            // GptPartitionType wraps a Guid (.0); to_bytes() consumes by value
            // and yields the mixed-endian on-disk form our constants use.
            let type_bytes = type_guid.0.to_bytes();
            info.is_esp    = type_bytes == ESP_GUID;
            info.is_linux  = type_bytes == LINUX_FS_GUID
                          || type_bytes == LINUX_ROOT_X64_GUID
                          || type_bytes == LINUX_XBOOTLDR_GUID;
        }
        // MBR partitions: fall through with empty gpt_guid; still detectable
        // by fstype and usable as chainload targets.
    }

    info.fstype  = detect_filesystem(bs, h);
    info.fs_uuid = get_fs_uuid(bs, h, &info.fstype);

    // ── Heuristic ESP detection ───────────────────────────────────────────
    // Some embedded/older firmware does not populate partition_type_guid.
    // Any FAT32 volume labelled like an ESP is treated as one.
    if !info.is_esp && info.fstype == "fat32" {
        let lbl = info.part_label.to_ascii_uppercase();
        if matches!(lbl.as_str(),
            "EFI" | "ESP" | "SYSTEM" | "BOOT" | "EFI SYSTEM PARTITION") {
            info.is_esp = true;
        }
    }

    Ok(Some(info))
}

// ════════════════════════════════════════════════════════════════════════════
//  §4 — Filesystem detection (4 KiB sniff from LBA 0)
// ════════════════════════════════════════════════════════════════════════════

fn detect_filesystem(bs: &BootServices, handle: Handle) -> String {
    let data = match read_blocks(bs, handle, 0, 8) {
        Some(d) => d,
        None    => return "unknown".into(),
    };

    // ── ext2/3/4: superblock at byte 1024, magic 0xEF53 at sb+56 ──────────
    if data.len() >= 1024 + 100 {
        let s_magic = u16::from_le_bytes([data[1024 + 56], data[1024 + 57]]);
        if s_magic == 0xEF53 {
            // s_feature_incompat at sb+96; INCOMPAT_EXTENTS = 0x40 is the
            // discriminator for ext4 vs ext2/3.
            let incompat = u32::from_le_bytes([
                data[1024 + 96], data[1024 + 97],
                data[1024 + 98], data[1024 + 99],
            ]);
            return if incompat & 0x40 != 0 { "ext4".into() } else { "ext2".into() };
        }
    }

    // ── FAT family: 0xAA55 boot signature at byte 510 ─────────────────────
    if data.len() >= 512 && u16::from_le_bytes([data[510], data[511]]) == 0xAA55 {
        // FAT32 BPB stores the type string at offset 82; FAT12/16 at 54.
        if data.len() >= 87 && &data[82..87] == b"FAT32" { return "fat32".into(); }
        if data.len() >= 59 && &data[54..59] == b"FAT16" { return "fat16".into(); }
        if data.len() >= 59 && &data[54..59] == b"FAT12" { return "fat12".into(); }
    }

    // ── NTFS: OEM ID "NTFS    " at offset 3 ───────────────────────────────
    if data.len() >= 11 && &data[3..11] == b"NTFS    " { return "ntfs".into(); }

    "unknown".into()
}

fn get_fs_uuid(bs: &BootServices, handle: Handle, fstype: &str) -> String {
    let data = match read_blocks(bs, handle, 0, 8) {
        Some(d) => d,
        None    => return String::new(),
    };

    match fstype {
        "ext4" | "ext3" | "ext2" => {
            // s_uuid at superblock offset 104, 16 bytes, stored big-endian
            // textual order (matches `blkid` output).
            if data.len() < 1024 + 120 { return String::new(); }
            let u = &data[1024 + 104..1024 + 120];
            format!(
                "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                u[0], u[1], u[2], u[3],
                u[4], u[5],
                u[6], u[7],
                u[8], u[9],
                u[10], u[11], u[12], u[13], u[14], u[15]
            )
        }
        "fat32" => {
            // Volume serial at byte 67, 4 bytes → "XXXX-XXXX".
            if data.len() < 71 { return String::new(); }
            let s = u32::from_le_bytes([data[67], data[68], data[69], data[70]]);
            format!("{:04X}-{:04X}", (s >> 16) & 0xFFFF, s & 0xFFFF)
        }
        "fat16" | "fat12" => {
            // Volume serial at byte 39, 4 bytes.
            if data.len() < 43 { return String::new(); }
            let s = u32::from_le_bytes([data[39], data[40], data[41], data[42]]);
            format!("{:04X}-{:04X}", (s >> 16) & 0xFFFF, s & 0xFFFF)
        }
        "ntfs" => {
            // Volume serial at byte 72, 8 bytes.
            if data.len() < 80 { return String::new(); }
            let s = u64::from_le_bytes([
                data[72], data[73], data[74], data[75],
                data[76], data[77], data[78], data[79],
            ]);
            format!("{:016X}", s)
        }
        _ => String::new(),
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §5 — Raw block read helper (also used by the ext4 reader closure)
// ════════════════════════════════════════════════════════════════════════════

pub fn read_blocks(bs: &BootServices, handle: Handle, lba: u64, count: usize) -> Option<Vec<u8>> {
    let proto    = bs.open_protocol_exclusive::<BlockIO>(handle).ok()?;
    let bio      = &*proto;
    let media    = bio.media();
    let media_id = media.media_id();
    let bsz      = media.block_size() as usize;
    if bsz == 0 || count == 0 { return None; }
    let mut buf  = vec![0u8; count * bsz];
    bio.read_blocks(media_id, lba, &mut buf).ok()?;
    Some(buf)
}

// ════════════════════════════════════════════════════════════════════════════
//  §6 — Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Decode a `[Char16; 36]` GPT name to a String, trimming at the first NUL.
/// `uefi::data_types::Char16` is `repr(transparent)` over `u16`; `u16::from`
/// is a safe value conversion.
fn decode_char16_array(raw: &[uefi::data_types::Char16; 36]) -> String {
    let u16s: Vec<u16> = raw.iter()
        .map(|c| u16::from(*c))
        .take_while(|&c| c != 0)
        .collect();
    String::from_utf16_lossy(&u16s)
}
