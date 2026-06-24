//! PuppyBoot V1 — Configuration parser.
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  Two config kinds, both INI-flavored (key = value, '#'/';' comments):
//!
//!    /EFI/puppyboot/loader.conf          → global BootManagerConfig
//!    /EFI/puppyboot/entries/<id>.conf    → one EntryConfig each
//!
//!  Each entry declares a `type` selecting one of four boot paths
//!  (EntryType). The dispatcher in boot.rs matches on this enum; getting the
//!  type wrong is the difference between LoadImage-ing a PE kernel and trying
//!  to ELF-parse it, so parse_entry_config validates the required fields per
//!  type before returning.

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use uefi::table::boot::BootServices;

use crate::disk;
use crate::partition::PartitionInfo;

// ════════════════════════════════════════════════════════════════════════════
//  §1 — Entry type
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryType {
    /// Modern Linux EFI-stub kernel: LoadImage + initrd= LoadOptions.
    LinuxStub,
    /// Legacy bzImage via the EFI handover protocol (handover_offset).
    LinuxHandover,
    /// AIOS native ELF64 kernel + LOADER_PARAMS handover.
    Aios,
    /// Arbitrary EFI binary (Windows Boot Manager, systemd-boot, …).
    Chain,
}

impl EntryType {
    fn parse(s: &str) -> EntryType {
        match s.trim().to_ascii_lowercase().as_str() {
            "linux-stub" | "stub" | "linux" | "efistub" | "efi-stub" => EntryType::LinuxStub,
            "linux-handover" | "handover" | "bzimage"                => EntryType::LinuxHandover,
            "aios" | "native" | "elf"                                => EntryType::Aios,
            "chain" | "chainload" | "efi"                            => EntryType::Chain,
            // Default to the safest, most broadly compatible path.
            _ => EntryType::LinuxStub,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §2 — Data structures
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ThemeConfig {
    pub title:        String,
    pub fg_color:     u8,
    pub bg_color:     u8,
    pub highlight_fg: u8,
    pub highlight_bg: u8,
    pub show_help:    bool,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            title:        "PuppyBoot V1".into(),
            fg_color:     0x0A, // light green
            bg_color:     0x00, // black
            highlight_fg: 0x00, // black text
            highlight_bg: 0x0E, // on yellow
            show_help:    true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BootManagerConfig {
    pub timeout:       u32,     // seconds; 0 = boot default immediately
    pub default_entry: String,  // entry id, name, or numeric index
    pub show_hidden:   bool,
    pub no_network:    bool,    // inject ip=off net.ifnames=0 globally
    pub entries_dir:   String,
    pub theme:         ThemeConfig,
}

impl Default for BootManagerConfig {
    fn default() -> Self {
        Self {
            timeout:       5,
            default_entry: "0".into(),
            show_hidden:   false,
            no_network:    false,
            entries_dir:   "/EFI/puppyboot/entries".into(),
            theme:         ThemeConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EntryConfig {
    pub id:             String,         // derived from filename (<id>.conf)
    pub name:           String,         // display name
    pub entry_type:     EntryType,
    pub kernel:         String,         // kernel/ELF/EFI path (POSIX form)
    pub initrd:         Vec<String>,    // zero or more initramfs images
    pub microcode:      Option<String>, // prepended before initrd (must be first)
    pub kernel_cmdline: String,
    pub part_uuid:      Option<String>, // PARTUUID / FS-UUID of kernel's partition
    pub part_label:     Option<String>, // GPT label fallback
    pub chain_target:   Option<String>, // for Chain: explicit EFI path
    pub load_addr:      String,         // for Aios/handover hints (hex string)
    pub hidden:         bool,
    pub no_network:     bool,
    pub recovery:       bool,
    pub ram_only:       bool,
    pub order:          u32,            // menu sort key (ascending)
}

impl Default for EntryConfig {
    fn default() -> Self {
        Self {
            id:             String::new(),
            name:           "Unnamed".into(),
            entry_type:     EntryType::LinuxStub,
            kernel:         String::new(),
            initrd:         Vec::new(),
            microcode:      None,
            kernel_cmdline: String::new(),
            part_uuid:      None,
            part_label:     None,
            chain_target:   None,
            load_addr:      "0x100000".into(),
            hidden:         false,
            no_network:     false,
            recovery:       false,
            ram_only:       false,
            order:          100,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §3 — Error type
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum ConfigError {
    Io(String),
    Utf8,
    Invalid(String),
}

// ════════════════════════════════════════════════════════════════════════════
//  §4 — Public API
// ════════════════════════════════════════════════════════════════════════════

pub fn parse_global_config(
    bs:   &BootServices,
    part: &PartitionInfo,
    path: &str,
) -> Result<BootManagerConfig, ConfigError> {
    let data = disk::read_file(bs, part, path).map_err(ConfigError::Io)?;
    let text = core::str::from_utf8(&data).map_err(|_| ConfigError::Utf8)?;

    let mut cfg = BootManagerConfig::default();
    for raw in text.lines() {
        let line = strip_comment(raw).trim();
        if line.is_empty() || is_section_header(line) { continue; }
        if let Some((k, v)) = split_kv(line) {
            apply_global_kv(&mut cfg, &k, &v);
        }
    }
    Ok(cfg)
}

pub fn parse_entry_config(
    bs:   &BootServices,
    part: &PartitionInfo,
    path: &str,
) -> Result<EntryConfig, ConfigError> {
    let data = disk::read_file(bs, part, path).map_err(ConfigError::Io)?;
    let text = core::str::from_utf8(&data).map_err(|_| ConfigError::Utf8)?;

    let mut ec = EntryConfig::default();
    if let Some(fname) = path.rsplit('/').next() {
        ec.id = fname.trim_end_matches(".conf").to_string();
        if ec.name == "Unnamed" {
            ec.name = ec.id.clone();
        }
    }

    for raw in text.lines() {
        let line = strip_comment(raw).trim();
        if line.is_empty() || is_section_header(line) { continue; }
        if let Some((k, v)) = split_kv(line) {
            apply_entry_kv(&mut ec, &k, &v);
        }
    }

    validate_entry(&ec)?;
    Ok(ec)
}

// ════════════════════════════════════════════════════════════════════════════
//  §5 — Per-type validation
// ════════════════════════════════════════════════════════════════════════════

fn validate_entry(ec: &EntryConfig) -> Result<(), ConfigError> {
    match ec.entry_type {
        EntryType::LinuxStub | EntryType::LinuxHandover | EntryType::Aios => {
            if ec.kernel.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "entry '{}' ({:?}) requires a 'kernel =' path", ec.id, ec.entry_type
                )));
            }
        }
        EntryType::Chain => {
            // Chain accepts either chain_target or kernel as the EFI path.
            if ec.chain_target.as_deref().unwrap_or("").trim().is_empty()
                && ec.kernel.trim().is_empty()
            {
                return Err(ConfigError::Invalid(format!(
                    "chain entry '{}' requires 'chainload =' (or 'kernel =') path", ec.id
                )));
            }
        }
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
//  §6 — Key/value application
// ════════════════════════════════════════════════════════════════════════════

fn apply_global_kv(cfg: &mut BootManagerConfig, k: &str, v: &str) {
    match k {
        "timeout"                          => cfg.timeout       = v.parse().unwrap_or(5),
        "default" | "default_entry"        => cfg.default_entry = unquote(v),
        "show_hidden"                      => cfg.show_hidden   = parse_bool(v),
        "no_network" | "nonet"             => cfg.no_network    = parse_bool(v),
        "entries_dir"                      => cfg.entries_dir   = unquote(v),
        "title"                            => cfg.theme.title         = unquote(v),
        "fg" | "fg_color"                  => cfg.theme.fg_color      = parse_color(v),
        "bg" | "bg_color"                  => cfg.theme.bg_color      = parse_color(v),
        "hi_fg" | "highlight_fg"           => cfg.theme.highlight_fg  = parse_color(v),
        "hi_bg" | "highlight_bg"           => cfg.theme.highlight_bg  = parse_color(v),
        "show_help"                        => cfg.theme.show_help     = parse_bool(v),
        _ => {}
    }
}

fn apply_entry_kv(ec: &mut EntryConfig, k: &str, v: &str) {
    match k {
        "name" | "title"                  => ec.name = unquote(v),
        "type"                            => ec.entry_type = EntryType::parse(v),
        "kernel" | "linux"                => ec.kernel = unquote(v),
        "initrd" | "initramfs"            => {
            let val = unquote(v);
            if val.eq_ignore_ascii_case("none") || val.is_empty() {
                ec.initrd.clear();
            } else {
                ec.initrd = val.split(',').map(|s| s.trim().to_string())
                               .filter(|s| !s.is_empty()).collect();
            }
        }
        "microcode" | "ucode"             => {
            let val = unquote(v);
            ec.microcode = if val.eq_ignore_ascii_case("none") || val.is_empty() {
                None
            } else {
                Some(val)
            };
        }
        "cmdline" | "options" | "kernel_cmdline" => ec.kernel_cmdline = unquote(v),
        "part_uuid" | "partuuid" | "uuid" => ec.part_uuid = some_nonempty(unquote(v)),
        "part_label" | "label"            => ec.part_label = some_nonempty(unquote(v)),
        "chainload" | "chain" | "efi"     => ec.chain_target = some_nonempty(unquote(v)),
        "load_addr" | "load_address"      => ec.load_addr = unquote(v),
        "hidden"                          => ec.hidden     = parse_bool(v),
        "no_network" | "nonet"            => ec.no_network = parse_bool(v),
        "recovery"                        => ec.recovery   = parse_bool(v),
        "ram_only" | "ramonly"            => ec.ram_only   = parse_bool(v),
        "order"                           => ec.order      = v.parse().unwrap_or(100),
        _ => {}
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §7 — Parsing helpers
// ════════════════════════════════════════════════════════════════════════════

fn strip_comment(line: &str) -> &str {
    let line = line.split('#').next().unwrap_or("");
    line.split(';').next().unwrap_or("")
}

fn is_section_header(line: &str) -> bool {
    line.starts_with('[') && line.ends_with(']')
}

fn split_kv(line: &str) -> Option<(String, String)> {
    let pos = line.find('=')?;
    let key = line[..pos].trim().to_ascii_lowercase();
    let val = line[pos + 1..].trim().to_string();
    if key.is_empty() { None } else { Some((key, val)) }
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"'  && bytes[bytes.len() - 1] == b'"')
        ||  (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn some_nonempty(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

pub fn parse_bool(s: &str) -> bool {
    matches!(s.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "on" | "enabled")
}

/// Parse a console color: hex ("0x0A"), decimal ("10"), or name ("green").
fn parse_color(s: &str) -> u8 {
    let t = s.trim();
    let lower = t.to_ascii_lowercase();
    match lower.as_str() {
        "black"        => return 0x00,
        "blue"         => return 0x01,
        "green"        => return 0x02,
        "cyan"         => return 0x03,
        "red"          => return 0x04,
        "magenta"      => return 0x05,
        "brown"        => return 0x06,
        "lightgray" | "lightgrey" | "gray" | "grey" => return 0x07,
        "darkgray" | "darkgrey" => return 0x08,
        "lightblue"    => return 0x09,
        "lightgreen"   => return 0x0A,
        "lightcyan"    => return 0x0B,
        "lightred"     => return 0x0C,
        "lightmagenta" => return 0x0D,
        "yellow"       => return 0x0E,
        "white"        => return 0x0F,
        _ => {}
    }
    if let Some(hex) = lower.strip_prefix("0x") {
        return u8::from_str_radix(hex, 16).unwrap_or(0x0A);
    }
    t.parse::<u8>().unwrap_or(0x0A)
}
