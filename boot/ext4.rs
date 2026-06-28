//! PuppyBoot — Minimal read-only ext4 reader.
//!
//! Handles: superblock, extent-tree traversal, directory lookup, file reads.
//! No journal replay, no write support.
//!
//! HARDENING:
//!  • Logical-to-Physical LBA conversion fixed (SAE FIX).
//!  • Extent tree depth limit (MAX_EXTENT_DEPTH) to prevent stack overflow.
//!  • Directory entry record length validation to prevent infinite loops.
//!  • Name length validation for directory lookups.
//!  • Integer overflow protection on offset calculations.

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use log::{info, warn};

// ─── Constants ────────────────────────────────────────────────────────────────

const EXT4_SUPER_MAGIC:  u16 = 0xEF53;
const EXT4_EXTENT_MAGIC: u16 = 0xF30A;
const S_IFDIR:           u16 = 0x4000;
const S_IFREG:           u16 = 0x8000;
const EXT4_NAME_LEN:     usize = 255;
const MAX_EXTENT_DEPTH:  u16 = 5; // Real ext4 trees are rarely deeper than 2-3

// ─── Extent tree on-disk structures ──────────────────────────────────────────

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct ExtentHeader {
    magic:      u16,
    entries:    u16,
    max:        u16,
    depth:      u16,
    generation: u32,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct ExtentLeaf {
    block:    u32,   // first logical block
    len:      u16,   // number of logical blocks in extent
    start_hi: u16,   // high 16 bits of physical block
    start_lo: u32,   // low 32 bits of physical block
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct ExtentIdx {
    block:   u32,
    leaf_lo: u32,
    leaf_hi: u16,
    _unused: u16,
}

// ─── Block reader callback ────────────────────────────────────────────────────

/// Reads `sector_count` sectors starting at `lba` (physical, partition-relative).
/// Returns `None` on I/O error.
pub type BlockReader<'a> = dyn Fn(u64, usize) -> Option<Vec<u8>> + 'a;

// ─── Ext4FS handle ────────────────────────────────────────────────────────────

pub struct Ext4FS {
    /// ext4 logical block size in bytes (1024 << log_block_size)
    pub block_size:        u64,
    /// Number of 512-byte sectors per ext4 logical block.
    pub sectors_per_block: u64,
    pub inodes_per_group:  u32,
    pub inode_size:        u32,
    pub desc_size:         u32,
    pub first_data_block:  u32,
    pub group_count:       u32,
    pub feature_incompat:  u32,
    pub uuid:              String,
}

impl Ext4FS {
    /// Open an ext4 filesystem.
    /// `reader` must read **sectors** (512-byte units), not ext4 blocks.
    pub fn open(reader: &BlockReader<'_>) -> Option<Self> {
        // The ext4 superblock starts at byte 1024 = LBA 2 (512-byte sectors).
        let data = reader(2, 8)?;
        if data.len() < 256 { return None; }

        let sb = &data[..];
        let magic = u16::from_le_bytes([sb[56], sb[57]]);
        if magic != EXT4_SUPER_MAGIC {
            info!("Bad ext4 magic: 0x{:X}", magic);
            return None;
        }

        let log_block_size   = u32::from_le_bytes([sb[24], sb[25], sb[26], sb[27]]);
        if log_block_size > 12 { // Limit block size to 4MiB (1024 << 12)
            warn!("Unsupported ext4 block size exponent: {}", log_block_size);
            return None;
        }

        let block_size       = 1024u64 << log_block_size;
        let sectors_per_block = block_size / 512;

        let inodes_per_group = u32::from_le_bytes([sb[40], sb[41], sb[42], sb[43]]);
        if inodes_per_group == 0 { return None; }

        let inode_size       = if sb.len() > 89 {
            u16::from_le_bytes([sb[88], sb[89]]) as u32
        } else {
            128
        };
        if inode_size < 128 || (inode_size & (inode_size - 1) != 0) {
            warn!("Invalid ext4 inode size: {}", inode_size);
            return None;
        }

        let feature_incompat = u32::from_le_bytes([sb[96], sb[97], sb[98], sb[99]]);
        let has_64bit        = feature_incompat & 0x80 != 0;
        let desc_size        = if has_64bit { 64u32 } else { 32u32 };

        let uuid_b = &sb[104..120];
        let uuid   = format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            uuid_b[3], uuid_b[2], uuid_b[1], uuid_b[0],
            uuid_b[5], uuid_b[4],
            uuid_b[7], uuid_b[6],
            uuid_b[8], uuid_b[9],
            uuid_b[10], uuid_b[11], uuid_b[12],
            uuid_b[13], uuid_b[14], uuid_b[15]
        );

        let group_count = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);

        info!(
            "ext4: block_size={} sectors/block={} inode_size={} groups={} uuid={}",
            block_size, sectors_per_block, inode_size, group_count, uuid
        );

        Some(Ext4FS {
            block_size,
            sectors_per_block,
            inodes_per_group,
            inode_size,
            desc_size,
            first_data_block: 0,
            group_count,
            feature_incompat,
            uuid,
        })
    }

    // ─── Block I/O wrapper ───────────────────────────────────────────────────

    fn read_block(&self, reader: &BlockReader<'_>, ext4_block: u64) -> Option<Vec<u8>> {
        let lba = ext4_block.checked_mul(self.sectors_per_block)?;
        let data = reader(lba, self.sectors_per_block as usize)?;
        if data.len() < self.block_size as usize { return None; }
        Some(data[..self.block_size as usize].to_vec())
    }

    fn read_blocks_n(&self, reader: &BlockReader<'_>, ext4_block: u64, n: usize) -> Option<Vec<u8>> {
        let lba = ext4_block.checked_mul(self.sectors_per_block)?;
        let total_sectors = self.sectors_per_block.checked_mul(n as u64)? as usize;
        let data = reader(lba, total_sectors)?;
        let expected_bytes = self.block_size.checked_mul(n as u64)? as usize;
        if data.len() < expected_bytes { return None; }
        Some(data[..expected_bytes].to_vec())
    }

    // ─── Inode reading ───────────────────────────────────────────────────────

    pub fn read_inode(&self, reader: &BlockReader<'_>, ino: u32) -> Option<Vec<u8>> {
        if ino == 0 { return None; }
        let ino0  = (ino - 1) as u64;
        let group = (ino0 / self.inodes_per_group as u64) as u32;
        let index = (ino0 % self.inodes_per_group as u64) as u32;

        let gdt_block  = if self.first_data_block == 0 { 1u64 } else { self.first_data_block as u64 + 1 };
        let gd_offset  = (group as u64).checked_mul(self.desc_size as u64)?;
        let gd_block   = gdt_block + (gd_offset / self.block_size);
        let gd_in_block = (gd_offset % self.block_size) as usize;

        let gd_data    = self.read_block(reader, gd_block)?;
        if gd_in_block + self.desc_size as usize > gd_data.len() { return None; }
        let gd         = &gd_data[gd_in_block..];

        // Inode table block: low 32 bits at offset 8 in group descriptor
        let inode_table_lo = u32::from_le_bytes([gd[8], gd[9], gd[10], gd[11]]) as u64;

        let inode_offset_bytes = (index as u64).checked_mul(self.inode_size as u64)?;
        let inode_block        = inode_table_lo + (inode_offset_bytes / self.block_size);
        let inode_in_block     = (inode_offset_bytes % self.block_size) as usize;

        let need_bytes = inode_in_block + self.inode_size as usize;
        let need_blocks = (need_bytes + self.block_size as usize - 1) / self.block_size as usize;
        let raw = self.read_blocks_n(reader, inode_block, need_blocks)?;

        let end = inode_in_block + self.inode_size as usize;
        if end > raw.len() { return None; }
        Some(raw[inode_in_block..end].to_vec())
    }

    pub fn inode_size(inode: &[u8]) -> u64 {
        if inode.len() < 128 { return 0; }
        let lo = u32::from_le_bytes([inode[4],  inode[5],  inode[6],  inode[7]]) as u64;
        let hi = if inode.len() >= 112 {
            u32::from_le_bytes([inode[108], inode[109], inode[110], inode[111]]) as u64
        } else { 0 };
        lo | (hi << 32)
    }

    pub fn inode_mode(inode: &[u8]) -> u16 {
        if inode.len() < 2 { return 0; }
        u16::from_le_bytes([inode[0], inode[1]])
    }

    // ─── Extent tree ─────────────────────────────────────────────────────────

    pub fn find_logical_block(
        &self,
        reader:        &BlockReader<'_>,
        inode:         &[u8],
        logical_block: u32,
    ) -> Option<u64> {
        self.search_extent_tree(reader, inode, 60, logical_block, 0)
    }

    fn search_extent_tree(
        &self,
        reader:  &BlockReader<'_>,
        root:    &[u8],
        offset:  usize,
        target:  u32,
        depth_call: u16,
    ) -> Option<u64> {
        if depth_call > MAX_EXTENT_DEPTH {
            warn!("ext4: extent tree too deep");
            return None;
        }
        if offset + 12 > root.len() { return None; }

        let hdr = unsafe {
            core::ptr::read_unaligned(root.as_ptr().add(offset) as *const ExtentHeader)
        };

        if u16::from_le(hdr.magic) != EXT4_EXTENT_MAGIC { return None; }

        let entries = u16::from_le(hdr.entries) as usize;
        let depth   = u16::from_le(hdr.depth);

        if depth == 0 {
            for i in 0..entries {
                let eoff = offset + 12 + i * 12;
                if eoff + 12 > root.len() { break; }
                let leaf = unsafe {
                    core::ptr::read_unaligned(root.as_ptr().add(eoff) as *const ExtentLeaf)
                };
                let lb  = u32::from_le(leaf.block);
                let len = u16::from_le(leaf.len) as u32;
                if target >= lb && target < lb + len {
                    let phys = (u16::from_le(leaf.start_hi) as u64) << 32
                        | u32::from_le(leaf.start_lo) as u64;
                    return Some(phys + (target - lb) as u64);
                }
            }
        } else {
            for i in 0..entries {
                let eoff = offset + 12 + i * 12;
                if eoff + 12 > root.len() { break; }
                let idx = unsafe {
                    core::ptr::read_unaligned(root.as_ptr().add(eoff) as *const ExtentIdx)
                };
                let ib = u32::from_le(idx.block);

                let next_b = if i + 1 < entries {
                    let noff = offset + 12 + (i + 1) * 12;
                    if noff + 12 <= root.len() {
                        let nidx = unsafe {
                            core::ptr::read_unaligned(root.as_ptr().add(noff) as *const ExtentIdx)
                        };
                        u32::from_le(nidx.block)
                    } else { u32::MAX }
                } else { u32::MAX };

                if target >= ib && target < next_b {
                    let child_phys = (u16::from_le(idx.leaf_hi) as u64) << 32
                        | u32::from_le(idx.leaf_lo) as u64;
                    let child_data = self.read_block(reader, child_phys)?;
                    return self.search_extent_tree(reader, &child_data, 0, target, depth_call + 1);
                }
            }
        }
        None
    }

    // ─── File data reading ───────────────────────────────────────────────────

    pub fn read_file(
        &self,
        reader: &BlockReader<'_>,
        inode:  &[u8],
        offset: u64,
        size:   u64,
    ) -> Option<Vec<u8>> {
        let file_size = Self::inode_size(inode);
        let end = (offset + size).min(file_size);
        if offset >= file_size { return Some(Vec::new()); }

        let mut result = Vec::with_capacity((end - offset) as usize);
        let mut pos    = offset;

        while pos < end {
            let logical_block = (pos / self.block_size) as u32;
            let block_offset  = (pos % self.block_size) as usize;
            let to_read       = ((end - pos) as usize).min(self.block_size as usize - block_offset);

            if let Some(phys_block) = self.find_logical_block(reader, inode, logical_block) {
                if let Some(data) = self.read_block(reader, phys_block) {
                    let avail = data.len().saturating_sub(block_offset);
                    let take  = to_read.min(avail);
                    result.extend_from_slice(&data[block_offset..block_offset + take]);
                    pos += take as u64;
                } else {
                    return None; // Return None on I/O error instead of silent zero-fill
                }
            } else {
                result.resize(result.len() + to_read, 0); // Sparse hole
                pos += to_read as u64;
            }
        }
        Some(result)
    }

    pub fn read_entire_file(&self, reader: &BlockReader<'_>, inode: &[u8]) -> Option<Vec<u8>> {
        let sz = Self::inode_size(inode);
        self.read_file(reader, inode, 0, sz)
    }

    // ─── Directory traversal ─────────────────────────────────────────────────

    pub fn lookup_path(&self, reader: &BlockReader<'_>, start_ino: u32, path: &str) -> Option<u32> {
        let mut current = start_ino;
        let trimmed     = path.trim_start_matches('/');
        if trimmed.is_empty() { return Some(current); }

        for component in trimmed.split('/') {
            if component.is_empty() || component == "." { continue; }
            if component == ".." {
                // Simplified: root's parent is root
                if current == 2 { continue; }
                // In a real implementation we'd look up ".." in current dir
            }

            let inode_data = self.read_inode(reader, current)?;
            if Self::inode_mode(&inode_data) & S_IFDIR == 0 { return None; }

            current = self.find_in_dir(reader, &inode_data, component)?;
        }
        Some(current)
    }

    fn find_in_dir(&self, reader: &BlockReader<'_>, dir_inode: &[u8], name: &str) -> Option<u32> {
        let size  = Self::inode_size(dir_inode);
        let mut offset = 0u64;

        while offset < size {
            let hdr = self.read_file(reader, dir_inode, offset, 8)?;
            if hdr.len() < 8 { break; }

            let entry_ino = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            let rec_len   = u16::from_le_bytes([hdr[4], hdr[5]]) as u64;
            let name_len  = hdr[6] as usize;

            // record length validation to prevent infinite loops
            if rec_len < 8 || rec_len > (size - offset) {
                warn!("ext4: invalid directory record length: {}", rec_len);
                break;
            }
            if entry_ino == 0 {
                offset += rec_len;
                continue;
            }

            if name_len > 0 && name_len <= EXT4_NAME_LEN && (name_len + 8) <= rec_len as usize {
                let name_data = self.read_file(reader, dir_inode, offset + 8, name_len as u64)?;
                if name_data.len() >= name_len {
                    if let Ok(s) = core::str::from_utf8(&name_data[..name_len]) {
                        if s == name { return Some(entry_ino); }
                    }
                }
            }
            offset += rec_len;
        }
        None
    }
}
