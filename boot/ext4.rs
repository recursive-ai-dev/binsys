//! PuppyBoot — Minimal read-only ext4 reader.
//!
//! Handles: superblock, extent-tree traversal, directory lookup, file reads.
//! No journal replay, no write support.
//!
//! SAE FIX vs original:
//!  • The BlockReader callback previously passed ext4 logical block numbers
//!    directly as physical sector LBAs.  An ext4 block is log_block_size
//!    sectors wide (e.g. block_size=4096, sector_size=512 → 8 sectors/block).
//!    Every read was therefore 8× too early in the partition, returning
//!    garbage data for any filesystem with block_size > sector_size.
//!
//!    Fix: Ext4FS stores sectors_per_block.  All internal calls to the
//!    BlockReader now convert:
//!      physical_lba = ext4_block_number * sectors_per_block
//!    The BlockReader itself remains agnostic — it takes (lba, sector_count).

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use log::info;

// ─── Constants ────────────────────────────────────────────────────────────────

const EXT4_SUPER_MAGIC:  u16 = 0xEF53;
const EXT4_EXTENT_MAGIC: u16 = 0xF30A;
const S_IFDIR:           u16 = 0x4000;
const S_IFREG:           u16 = 0x8000;
const EXT4_NAME_LEN:     usize = 255;

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
    /// SAE FIX: stored here so every reader call converts correctly.
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
        // We read 8 sectors (4 KiB) which is enough for any superblock variant.
        let data = reader(2, 8)?;
        if data.len() < 256 { return None; }

        // Superblock is at the very start of our buffer (offset 0 within it,
        // because we started reading at the superblock sector).
        // Actually: superblock is at byte offset 1024 from partition start.
        // LBA 2 = byte 1024. So sb starts at data[0].
        let sb = &data[..];

        let magic = u16::from_le_bytes([sb[56], sb[57]]);
        if magic != EXT4_SUPER_MAGIC {
            info!("Bad ext4 magic: 0x{:X}", magic);
            return None;
        }

        let log_block_size   = u32::from_le_bytes([sb[24], sb[25], sb[26], sb[27]]);
        let block_size       = 1024u64 << log_block_size;
        // SAE FIX: compute sectors per block (sector size = 512)
        let sectors_per_block = block_size / 512;

        let inodes_per_group = u32::from_le_bytes([sb[40], sb[41], sb[42], sb[43]]);
        let inode_size       = if sb.len() > 89 {
            u16::from_le_bytes([sb[88], sb[89]]) as u32
        } else {
            128
        };

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

    /// Read one ext4 logical block.
    ///
    /// SAE FIX: converts ext4 block number → physical LBA before calling reader.
    fn read_block(&self, reader: &BlockReader<'_>, ext4_block: u64) -> Option<Vec<u8>> {
        let lba           = ext4_block * self.sectors_per_block;
        let sector_count  = self.sectors_per_block as usize;
        let data = reader(lba, sector_count)?;
        // Trim to exactly one ext4 block (reader may return more if block > 512b)
        if data.len() < self.block_size as usize { return None; }
        Some(data[..self.block_size as usize].to_vec())
    }

    /// Read `n` consecutive ext4 logical blocks starting at `ext4_block`.
    fn read_blocks_n(&self, reader: &BlockReader<'_>, ext4_block: u64, n: usize) -> Option<Vec<u8>> {
        let lba          = ext4_block * self.sectors_per_block;
        let sector_count = self.sectors_per_block as usize * n;
        let data = reader(lba, sector_count)?;
        if (data.len() as u64) < self.block_size * n as u64 { return None; }
        Some(data[..self.block_size as usize * n].to_vec())
    }

    // ─── Inode reading ───────────────────────────────────────────────────────

    /// Read raw inode bytes for inode number `ino` (1-based).
    pub fn read_inode(&self, reader: &BlockReader<'_>, ino: u32) -> Option<Vec<u8>> {
        let ino0  = (ino - 1) as u64;
        let group = (ino0 / self.inodes_per_group as u64) as u32;
        let index = (ino0 % self.inodes_per_group as u64) as u32;

        // Group descriptor table starts at block (first_data_block + 1)
        let gdt_block  = if self.first_data_block == 0 { 1u64 } else { self.first_data_block as u64 + 1 };
        let gd_offset  = group as u64 * self.desc_size as u64;
        let gd_block   = gdt_block + gd_offset / self.block_size;
        let gd_in_block = (gd_offset % self.block_size) as usize;

        let gd_data    = self.read_block(reader, gd_block)?;
        if gd_in_block + self.desc_size as usize > gd_data.len() { return None; }
        let gd         = &gd_data[gd_in_block..];

        // Inode table block: low 32 bits at offset 8 in group descriptor
        let inode_table_lo = u32::from_le_bytes([gd[8], gd[9], gd[10], gd[11]]) as u64;

        let inode_offset_bytes = index as u64 * self.inode_size as u64;
        let inode_block        = inode_table_lo + inode_offset_bytes / self.block_size;
        let inode_in_block     = (inode_offset_bytes % self.block_size) as usize;

        // May span two blocks if inode straddles a block boundary
        let need_bytes = inode_in_block + self.inode_size as usize;
        let need_blocks = (need_bytes + self.block_size as usize - 1) / self.block_size as usize;
        let raw = self.read_blocks_n(reader, inode_block, need_blocks)?;

        let end = inode_in_block + self.inode_size as usize;
        if end > raw.len() { return None; }
        Some(raw[inode_in_block..end].to_vec())
    }

    /// Root inode is always 2.
    pub fn read_root_inode(&self, reader: &BlockReader<'_>) -> Option<Vec<u8>> {
        self.read_inode(reader, 2)
    }

    // ─── Inode field accessors ───────────────────────────────────────────────

    /// File size in bytes (combines lo at offset 4 and hi at offset 108).
    pub fn inode_size(inode: &[u8]) -> u64 {
        let lo = u32::from_le_bytes([inode[4],  inode[5],  inode[6],  inode[7]]) as u64;
        let hi = if inode.len() > 111 {
            u32::from_le_bytes([inode[108], inode[109], inode[110], inode[111]]) as u64
        } else { 0 };
        lo | (hi << 32)
    }

    /// File mode (offset 0).
    pub fn inode_mode(inode: &[u8]) -> u16 {
        u16::from_le_bytes([inode[0], inode[1]])
    }

    // ─── Extent tree ─────────────────────────────────────────────────────────

    /// Resolve logical block number → physical ext4 block number via extent tree.
    pub fn find_logical_block(
        &self,
        reader:        &BlockReader<'_>,
        inode:         &[u8],
        logical_block: u32,
    ) -> Option<u64> {
        self.search_extent_tree(reader, inode, 60, logical_block)
    }

    fn search_extent_tree(
        &self,
        reader:  &BlockReader<'_>,
        root:    &[u8],
        offset:  usize,
        target:  u32,
    ) -> Option<u64> {
        if offset + 12 > root.len() { return None; }

        let hdr = unsafe {
            core::ptr::read_unaligned(root.as_ptr().add(offset) as *const ExtentHeader)
        };

        if u16::from_le(hdr.magic) != EXT4_EXTENT_MAGIC { return None; }

        let entries = u16::from_le(hdr.entries) as usize;
        let depth   = u16::from_le(hdr.depth);

        if depth == 0 {
            // Leaf level: scan ExtentLeaf entries
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
            // Internal node: find child index subtree
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
                    // SAE FIX: read the child block using the correct physical-LBA conversion
                    let child_data = self.read_block(reader, child_phys)?;
                    return self.search_extent_tree(reader, &child_data, 0, target);
                }
            }
        }
        None
    }

    // ─── File data reading ───────────────────────────────────────────────────

    /// Read `size` bytes from a regular file starting at byte `offset`.
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
                // SAE FIX: read_block does the sectors_per_block conversion
                if let Some(data) = self.read_block(reader, phys_block) {
                    let avail = data.len().saturating_sub(block_offset);
                    let take  = to_read.min(avail);
                    result.extend_from_slice(&data[block_offset..block_offset + take]);
                    pos += take as u64;
                } else {
                    // I/O error — sparse-fill with zeros
                    result.resize(result.len() + to_read, 0);
                    pos += to_read as u64;
                }
            } else {
                // Sparse hole — fill with zeros
                result.resize(result.len() + to_read, 0);
                pos += to_read as u64;
            }
        }
        Some(result)
    }

    /// Read entire file contents.
    pub fn read_entire_file(&self, reader: &BlockReader<'_>, inode: &[u8]) -> Option<Vec<u8>> {
        let sz = Self::inode_size(inode);
        self.read_file(reader, inode, 0, sz)
    }

    // ─── Directory traversal ─────────────────────────────────────────────────

    /// Resolve an absolute path starting from inode `start_ino`.
    pub fn lookup_path(&self, reader: &BlockReader<'_>, start_ino: u32, path: &str) -> Option<u32> {
        let mut current = start_ino;
        let trimmed     = path.trim_start_matches('/');
        if trimmed.is_empty() { return Some(current); }

        for component in trimmed.split('/') {
            if component.is_empty() || component == "." { continue; }
            if component == ".." { current = 2; continue; } // simplified: root's parent is root

            let inode_data = self.read_inode(reader, current)?;
            if Self::inode_mode(&inode_data) & S_IFDIR == 0 { return None; }

            current = self.find_in_dir(reader, &inode_data, component)?;
        }
        Some(current)
    }

    /// Find a named entry within a directory inode. Returns its inode number.
    fn find_in_dir(&self, reader: &BlockReader<'_>, dir_inode: &[u8], name: &str) -> Option<u32> {
        let size  = Self::inode_size(dir_inode);
        let mut offset = 0u64;

        while offset < size {
            let hdr = self.read_file(reader, dir_inode, offset, 8)?;
            if hdr.len() < 8 { break; }

            let entry_ino = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            let rec_len   = u16::from_le_bytes([hdr[4], hdr[5]]) as u64;
            let name_len  = hdr[6] as usize;

            if rec_len == 0 || entry_ino == 0 { break; }

            if name_len > 0 && name_len <= EXT4_NAME_LEN {
                let name_data = self.read_file(reader, dir_inode, offset + 8, name_len as u64)?;
                if name_data.len() >= name_len {
                    if core::str::from_utf8(&name_data[..name_len]).unwrap_or("") == name {
                        return Some(entry_ino);
                    }
                }
            }
            offset += rec_len;
        }
        None
    }

    /// List all entries in a directory. Returns (name, inode, file_type).
    pub fn list_dir(&self, reader: &BlockReader<'_>, dir_inode: &[u8]) -> Vec<(String, u32, u8)> {
        let size  = Self::inode_size(dir_inode);
        let mut entries = Vec::new();
        let mut offset  = 0u64;

        while offset < size {
            let hdr = match self.read_file(reader, dir_inode, offset, 8) {
                Some(h) if h.len() >= 8 => h,
                _ => break,
            };

            let entry_ino = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
            let rec_len   = u16::from_le_bytes([hdr[4], hdr[5]]) as u64;
            let name_len  = hdr[6] as usize;
            let file_type = hdr[7];

            if rec_len == 0 || entry_ino == 0 { break; }

            if name_len > 0 && name_len <= EXT4_NAME_LEN {
                if let Some(nd) = self.read_file(reader, dir_inode, offset + 8, name_len as u64) {
                    if nd.len() >= name_len {
                        let s = String::from_utf8_lossy(&nd[..name_len]).into_owned();
                        entries.push((s, entry_ino, file_type));
                    }
                }
            }
            offset += rec_len;
        }
        entries
    }
}
