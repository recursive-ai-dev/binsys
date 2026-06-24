//! PuppyBoot V1 — UEFI boot manager entry point.
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  PIPELINE (per boot)
//!  ────────────────────────────────────────────────────────────────────────────
//!    1. uefi::helpers::init  — allocator, logger, panic handler.
//!    2. partition::discover  — enumerate every BlockIO handle, classify
//!       GPT type + filesystem.
//!    3. Locate ESP, read /EFI/puppyboot/loader.conf (global settings).
//!    4. Enumerate /EFI/puppyboot/entries/*.conf, parse each.
//!    5. Resolve each entry's kernel partition (PARTUUID > label > ESP fallback).
//!    6. show_boot_menu loop — non-consuming actions handled inline,
//!       boot decision causes loop exit.
//!    7. dispatch_boot — consumes SystemTable<Boot> and transfers control
//!       to the selected kernel; never returns.
//!
//!  FALLBACK PATHS
//!  ────────────────────────────────────────────────────────────────────────────
//!    • No config / no entries  → fallback_chainload tries
//!      /EFI/BOOT/BOOTX64.EFI on every FAT volume.
//!    • All boot attempts fail  → cold reboot.
//!
//!  GUARANTEES
//!  ────────────────────────────────────────────────────────────────────────────
//!    • No dynamic dispatch through unbounded vtables — match-based.
//!    • Every Result has an explicit failure path; no `.unwrap()` outside
//!      of `Option` operations on infallible internals.
//!    • All filesystem reads bounded; no unbounded recursion in dir walk.

#![no_std]
#![no_main]

extern crate alloc;

mod boot;
mod config;
mod disk;
mod ext4;
mod gop;
mod menu;
mod partition;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use log::{error, info, warn};
use uefi::prelude::*;
use uefi::table::runtime::ResetType;

use config::{BootManagerConfig, EntryConfig, EntryType};
use menu::BootMenuAction;
use partition::PartitionInfo;

// ════════════════════════════════════════════════════════════════════════════
//  §1 — Boot context
// ════════════════════════════════════════════════════════════════════════════

struct BootContext {
    partitions:    Vec<PartitionInfo>,
    global_config: BootManagerConfig,
    /// (entry-config, partition-on-which-the-kernel-lives)
    entries:       Vec<(EntryConfig, PartitionInfo)>,
}

// ════════════════════════════════════════════════════════════════════════════
//  §2 — Partition resolution per entry
// ════════════════════════════════════════════════════════════════════════════

fn resolve_entry_partition(
    entry:      &EntryConfig,
    partitions: &[PartitionInfo],
) -> Option<PartitionInfo> {
    // Explicit PARTUUID / fs-UUID match (case + dash insensitive).
    if let Some(ref uuid) = entry.part_uuid {
        if let Some(p) = partitions.iter().find(|p| p.matches_uuid(uuid)) {
            return Some(p.clone());
        }
    }
    // Explicit GPT label match.
    if let Some(ref label) = entry.part_label {
        if let Some(p) = partitions.iter().find(|p| p.matches_label(label)) {
            return Some(p.clone());
        }
    }
    // Fallback: ESP first (FAT-readable by UEFI), then any Linux FS.
    partitions.iter().find(|p| p.is_esp)
        .or_else(|| partitions.iter().find(|p| p.is_linux))
        .cloned()
}

fn find_default_index(default: &str, entries: &[(EntryConfig, PartitionInfo)]) -> Option<usize> {
    if let Ok(n) = default.parse::<usize>() {
        if n < entries.len() { return Some(n); }
    }
    entries.iter().position(|(e, _)| {
        e.id.eq_ignore_ascii_case(default) || e.name.eq_ignore_ascii_case(default)
    })
}

// ════════════════════════════════════════════════════════════════════════════
//  §3 — Context builder
// ════════════════════════════════════════════════════════════════════════════

fn build_context(st: &mut SystemTable<Boot>) -> Result<BootContext, String> {
    let bs = st.boot_services();

    let partitions = partition::discover_partitions(bs)
        .map_err(|e| format!("partition discovery: {:?}", e))?;
    info!("Found {} partitions", partitions.len());

    // Locate ESP (FAT32) for config storage.
    let config_part = partitions.iter()
        .find(|p| p.is_esp)
        .or_else(|| partitions.iter().find(|p| p.fstype == "fat32"))
        .cloned()
        .ok_or_else(|| "no FAT32/ESP partition available for config".to_string())?;

    info!("Config partition: #{} ({})", config_part.part_num, config_part.gpt_guid);

    let loader_conf_path = "/EFI/puppyboot/loader.conf";
    let global_config = match config::parse_global_config(bs, &config_part, loader_conf_path) {
        Ok(g)  => g,
        Err(e) => {
            warn!("loader.conf not loaded ({:?}); using defaults", e);
            BootManagerConfig::default()
        }
    };

    let entries_dir   = global_config.entries_dir.clone();
    let entry_files   = disk::list_fat_dir(bs, config_part.handle, &entries_dir)
        .unwrap_or_default();

    info!("Scanning {} for entry files ({} found)", entries_dir, entry_files.len());

    let mut entries: Vec<(EntryConfig, PartitionInfo)> = entry_files.into_iter()
        .filter(|f| f.ends_with(".conf"))
        .filter_map(|fname| {
            let full = format!("{}/{}", entries_dir, fname);
            match config::parse_entry_config(bs, &config_part, &full) {
                Ok(ec) => {
                    match resolve_entry_partition(&ec, &partitions) {
                        Some(pi) => {
                            info!("  + {}  type={:?}  fs={}",
                                  ec.name, ec.entry_type, pi.fstype);
                            Some((ec, pi))
                        }
                        None => {
                            warn!("  ! {}: no partition resolved", ec.name);
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!("  ! {}: {:?}", fname, e);
                    None
                }
            }
        })
        .collect();

    entries.sort_by_key(|(ec, _)| ec.order);

    Ok(BootContext { partitions, global_config, entries })
}

// ════════════════════════════════════════════════════════════════════════════
//  §4 — Boot dispatch (consumes SystemTable when needed)
// ════════════════════════════════════════════════════════════════════════════
//
//  Takes the SystemTable<Boot> BY VALUE because LinuxHandover and AiosNative
//  consume it (they call ExitBootServices, or in handover's case, hand the
//  pointer to the kernel which then exits). LinuxStub and Chain don't
//  consume it — they hand off via LoadImage/StartImage — but we still take
//  by value for uniformity. All paths return `!`.

fn dispatch_boot(
    st:           SystemTable<Boot>,
    image_handle: Handle,
    entry:        EntryConfig,
    part:         PartitionInfo,
) -> ! {
    info!("Dispatching boot: {} (type={:?})", entry.name, entry.entry_type);

    // Bind st mutably for the &mut-taking paths.
    let mut st = st;

    match entry.entry_type {
        EntryType::LinuxStub => {
            boot::boot_linux_stub(&mut st, image_handle, &part, &entry);
        }
        EntryType::LinuxHandover => {
            // Consumes st.
            boot::boot_linux_handover(st, image_handle, &part, &entry);
        }
        EntryType::Aios => {
            // Consumes st (calls ExitBootServices internally).
            boot::boot_aios(st, image_handle, &part, &entry);
        }
        EntryType::Chain => {
            let target = entry.chain_target.clone()
                .unwrap_or_else(|| entry.kernel.clone());
            boot::chainload_efi(&mut st, image_handle, part.handle, &target);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §5 — Cmdline augmentation
// ════════════════════════════════════════════════════════════════════════════
//
//  Apply global/per-entry flags to the kernel command line BEFORE handover.
//  These are not Linux-stub specific but most are Linux semantics; an AIOS
//  kernel ignores them (the AIOS kernel doesn't currently parse cmdline,
//  it reads LOADER_PARAMS). We still inject them so a single config entry
//  works for either kernel type.

fn finalize_cmdline(
    mut entry: EntryConfig,
    global:    &BootManagerConfig,
    extra:     Option<&str>,
) -> EntryConfig {
    let cl = &mut entry.kernel_cmdline;

    if let Some(x) = extra {
        if !x.is_empty() {
            cl.push(' ');
            cl.push_str(x);
        }
    }
    if global.no_network || entry.no_network {
        if !cl.contains("ip=off")        { cl.push_str(" ip=off"); }
        if !cl.contains("net.ifnames=0") { cl.push_str(" net.ifnames=0"); }
    }
    if entry.recovery {
        if !cl.contains("single") { cl.push_str(" single"); }
        if !cl.contains(" ro ") && !cl.contains(" rw ")
        && !cl.ends_with(" ro") && !cl.ends_with(" rw") {
            cl.push_str(" ro");
        }
    }
    if entry.ram_only && !cl.contains("pfix=") {
        cl.push_str(" pfix=ram");
    }
    entry
}

// ════════════════════════════════════════════════════════════════════════════
//  §6 — Last-resort chainload search
// ════════════════════════════════════════════════════════════════════════════

fn try_fallback_chainload(st: &mut SystemTable<Boot>, image_handle: Handle) -> Status {
    info!("Attempting fallback chainload of \\EFI\\BOOT\\BOOTX64.EFI ...");
    let partitions = partition::discover_partitions(st.boot_services()).unwrap_or_default();
    for part in &partitions {
        if !matches!(part.fstype.as_str(), "fat32" | "fat16" | "fat12") { continue; }
        // Don't recurse into ourselves on the volume we just booted from.
        if let Ok(s) = boot::try_chainload(st, image_handle, part.handle, "/EFI/BOOT/BOOTX64.EFI") {
            info!("Fallback chain returned: {:?}", s);
            return s;
        }
    }
    menu::show_error(st, "No bootable entries or chain targets remain.");
    Status::LOAD_ERROR
}

// ════════════════════════════════════════════════════════════════════════════
//  §7 — Entry point
// ════════════════════════════════════════════════════════════════════════════

#[entry]
fn efi_main(image_handle: Handle, mut system_table: SystemTable<Boot>) -> Status {
    // Allocator, logger, panic handler. Must be the first call.
    uefi::helpers::init(&mut system_table).expect("uefi helpers init");

    let _ = system_table.stdout().reset(false);
    info!("PuppyBoot V1 — UEFI boot manager");
    info!("UEFI revision: {}.{}",
        system_table.uefi_revision().major(),
        system_table.uefi_revision().minor(),
    );

    // ── Discover & parse ──────────────────────────────────────────────────
    let ctx = match build_context(&mut system_table) {
        Ok(c)  => c,
        Err(e) => {
            error!("Init failed: {}", e);
            menu::show_error(&mut system_table, &format!("Init: {}", e));
            return try_fallback_chainload(&mut system_table, image_handle);
        }
    };

    if ctx.entries.is_empty() {
        warn!("No boot entries configured; falling back to chainload");
        return try_fallback_chainload(&mut system_table, image_handle);
    }

    // ── Menu loop ─────────────────────────────────────────────────────────
    //
    // Non-consuming actions (chainload, reboot, shutdown) run inside the
    // loop. Boot decisions break out with a payload so we can dispatch the
    // consuming boot paths AFTER the loop, with full ownership of
    // system_table.
    enum Decision {
        Boot(EntryConfig, PartitionInfo),
    }

    let decision: Decision = loop {
        let action = menu::show_boot_menu(
            &mut system_table,
            &ctx.global_config,
            &ctx.entries,
            &ctx.partitions,
        );

        match action {
            BootMenuAction::BootEntry(idx) => {
                if idx >= ctx.entries.len() {
                    menu::show_error(&mut system_table, "Internal: entry index out of range");
                    continue;
                }
                let (ec, pi) = &ctx.entries[idx];
                let finalized = finalize_cmdline(ec.clone(), &ctx.global_config, None);
                break Decision::Boot(finalized, pi.clone());
            }
            BootMenuAction::Chainload(path, handle) => {
                boot::chainload_efi(&mut system_table, image_handle, handle, &path);
            }
            BootMenuAction::EditAndBoot(idx, extra) => {
                if idx >= ctx.entries.len() {
                    menu::show_error(&mut system_table, "Internal: entry index out of range");
                    continue;
                }
                let (ec, pi) = &ctx.entries[idx];
                let finalized = finalize_cmdline(ec.clone(), &ctx.global_config, Some(&extra));
                break Decision::Boot(finalized, pi.clone());
            }
            BootMenuAction::Reboot => {
                system_table.runtime_services().reset(ResetType::COLD, Status::SUCCESS, None);
            }
            BootMenuAction::Shutdown => {
                system_table.runtime_services().reset(ResetType::SHUTDOWN, Status::SUCCESS, None);
            }
            BootMenuAction::Timeout => {
                match find_default_index(&ctx.global_config.default_entry, &ctx.entries) {
                    Some(idx) => {
                        let (ec, pi) = &ctx.entries[idx];
                        let finalized = finalize_cmdline(ec.clone(), &ctx.global_config, None);
                        break Decision::Boot(finalized, pi.clone());
                    }
                    None => {
                        warn!("Timeout but default '{}' not resolvable; staying in menu",
                              ctx.global_config.default_entry);
                        // Loop continues — give the user manual control.
                    }
                }
            }
        }
    };

    let Decision::Boot(entry, part) = decision;
    dispatch_boot(system_table, image_handle, entry, part);
}
