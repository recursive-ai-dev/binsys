//! PuppyBoot V1 — Interactive boot menu.
//!
//! ════════════════════════════════════════════════════════════════════════════
//!  UI: double-line box border, countdown progress bar, arrow-key selection,
//!  in-place command-line edit, F-key shortcuts. Renders correctly on any
//!  UEFI graphical text console (Char16 = UCS-2).
//!
//!  KEYBOARD
//!    ↑ / ↓             Move selection
//!    Enter             Boot selected entry
//!    1 – 9             Quick-boot entry by number
//!    E                 Edit kernel command line for selected entry
//!    H  /  ?           Help screen
//!    F10  /  ESC       Reboot
//!    F11               Shutdown / power off
//!    F12               Toggle hidden entries
//!
//!  uefi 0.28 API notes (preserved from prior correct implementation):
//!    • output_string takes &CStr16, not &str → print_str helper.
//!    • Color is an enum, not constructible from u8 → match-based mapping.
//!    • ScanCode is a newtype around u16; match on .0.
//!    • create_event needs Tpl::APPLICATION.
//!    • TimerTrigger::Relative(100ns_units).
//!    • Char16 → char via u16::from then char::from_u32.

#![allow(dead_code)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use uefi::prelude::*;
use uefi::proto::console::text::{Color, Key, Output};
use uefi::table::boot::{EventType, TimerTrigger, Tpl};
use uefi::{CString16, Event};

use crate::config::{BootManagerConfig, EntryConfig};
use crate::partition::PartitionInfo;

// ════════════════════════════════════════════════════════════════════════════
//  §1 — Public action type
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum BootMenuAction {
    BootEntry(usize),
    Chainload(String, Handle),
    EditAndBoot(usize, String),
    Reboot,
    Shutdown,
    Timeout,
}

struct MenuItem {
    label:          String,
    detail:         String,
    is_hidden:      bool,
    entry_index:    usize,         // usize::MAX for chainload items
    chainload_path: Option<String>,
    chainload_part: Option<Handle>,
}

// ════════════════════════════════════════════════════════════════════════════
//  §2 — Color & output helpers
// ════════════════════════════════════════════════════════════════════════════

fn u8_to_fg_color(n: u8) -> Color {
    match n & 0x0F {
        0  => Color::Black,
        1  => Color::Blue,
        2  => Color::Green,
        3  => Color::Cyan,
        4  => Color::Red,
        5  => Color::Magenta,
        6  => Color::Brown,
        7  => Color::LightGray,
        8  => Color::DarkGray,
        9  => Color::LightBlue,
        10 => Color::LightGreen,
        11 => Color::LightCyan,
        12 => Color::LightRed,
        13 => Color::LightMagenta,
        14 => Color::Yellow,
        _  => Color::White,
    }
}

fn u8_to_bg_color(n: u8) -> Color {
    // UEFI 2.10 §12.4.2 — background palette restricted to colors 0..=7.
    match n & 0x07 {
        0 => Color::Black,
        1 => Color::Blue,
        2 => Color::Green,
        3 => Color::Cyan,
        4 => Color::Red,
        5 => Color::Magenta,
        6 => Color::Brown,
        _ => Color::LightGray,
    }
}

pub fn print_str(out: &mut Output, s: &str) {
    if let Ok(cs) = CString16::try_from(s) {
        let _ = out.output_string(&cs);
    }
}

fn set_fg_bg(out: &mut Output, fg: u8, bg: u8) {
    let _ = out.set_color(u8_to_fg_color(fg), u8_to_bg_color(bg));
}

fn set_colors(out: &mut Output, fg: Color, bg: Color) {
    let _ = out.set_color(fg, bg);
}

fn cursor(out: &mut Output, col: usize, row: usize) {
    let _ = out.set_cursor_position(col, row);
}

fn clear(out: &mut Output) {
    let _ = out.clear();
}

fn text_dims(out: &mut Output) -> (usize, usize) {
    out.current_mode()
        .ok()
        .flatten()
        .map(|m| (m.columns(), m.rows()))
        .unwrap_or((80, 25))
}

// ════════════════════════════════════════════════════════════════════════════
//  §3 — Box-drawing primitives (UCS-2 BMP — safe for CStr16)
// ════════════════════════════════════════════════════════════════════════════

const BOX_TL:  &str = "\u{2554}"; // ╔
const BOX_TR:  &str = "\u{2557}"; // ╗
const BOX_BL:  &str = "\u{255A}"; // ╚
const BOX_BR:  &str = "\u{255D}"; // ╝
const BOX_HH:  &str = "\u{2550}"; // ═
const BOX_VV:  &str = "\u{2551}"; // ║
const BOX_ML:  &str = "\u{2560}"; // ╠
const BOX_MR:  &str = "\u{2563}"; // ╣
const BLOCK_F: &str = "\u{2588}"; // █
const BLOCK_E: &str = "\u{2591}"; // ░
const SEL_ARR: &str = "\u{25BA}"; // ►

fn draw_hline(out: &mut Output, col: usize, row: usize, w: usize,
              left: &str, fill: &str, right: &str)
{
    cursor(out, col, row);
    print_str(out, left);
    for _ in 0..w.saturating_sub(2) { print_str(out, fill); }
    print_str(out, right);
}

fn print_centered(out: &mut Output, col: usize, row: usize, s: &str, width: usize) {
    let len = s.chars().count();
    let pad = width.saturating_sub(len) / 2;
    cursor(out, col + pad, row);
    print_str(out, s);
}

fn progress_bar(filled: usize, total: usize, width: usize) -> String {
    let f = if total > 0 { filled * width / total } else { width };
    core::iter::repeat(BLOCK_F).take(f)
        .chain(core::iter::repeat(BLOCK_E).take(width.saturating_sub(f)))
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
//  §4 — Input / timer
// ════════════════════════════════════════════════════════════════════════════

/// Wait up to `timeout_100ns` 100-ns units for a keypress.
///
/// 1 second = 10,000,000 × 100 ns; pass `u64::MAX` for indefinite wait.
///
/// Returns `Some(key)` if a key was pressed, `None` if the timer expired.
fn wait_for_key(st: &mut SystemTable<Boot>, timeout_100ns: u64) -> Option<Key> {
    // `wait_for_key_event` returns an owned `Event`; capturing it ends the
    // mutable borrow of stdin before we borrow boot_services below.
    let key_event: Event = st.stdin().wait_for_key_event()?;

    let bs    = st.boot_services();
    // SAFETY: creating a timer event with no callback is sound; the closure
    // arguments are None so no notify function is registered.
    let timer = unsafe {
        bs.create_event(EventType::TIMER, Tpl::APPLICATION, None, None).ok()?
    };
    if bs.set_timer(&timer, TimerTrigger::Relative(timeout_100ns)).is_err() {
        let _ = bs.close_event(timer);
        return None;
    }

    let mut events = [
        unsafe { key_event.unsafe_clone() },
        unsafe { timer.unsafe_clone() },
    ];
    // wait_for_event → Result<usize, Option<usize>>; Ok(index) is the index
    // of the signalled event in `events`.
    let signalled = match bs.wait_for_event(&mut events) {
        Ok(idx) => idx,
        Err(_)  => {
            let _ = bs.close_event(timer);
            return None;
        }
    };

    let _ = bs.set_timer(&timer, TimerTrigger::Cancel);
    let _ = bs.close_event(timer);

    if signalled == 0 {
        // Key event signalled — read the keystroke.
        st.stdin().read_key().ok().flatten()
    } else {
        None // timer (index 1) fired
    }
}

fn wait_for_any_key(st: &mut SystemTable<Boot>) -> Key {
    loop {
        if let Some(k) = wait_for_key(st, u64::MAX) { return k; }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §5 — Main menu loop
// ════════════════════════════════════════════════════════════════════════════

pub fn show_boot_menu(
    st:         &mut SystemTable<Boot>,
    global:     &BootManagerConfig,
    entries:    &[(EntryConfig, PartitionInfo)],
    partitions: &[PartitionInfo],
) -> BootMenuAction {
    let mut show_hidden = global.show_hidden;

    let mut items: Vec<MenuItem> = entries.iter().enumerate().map(|(i, (ec, pi))| {
        let guid_short = if pi.gpt_guid.len() >= 8 { &pi.gpt_guid[..8] } else { "?" };
        let detail = format!("{} \u{00B7} {}", guid_short, pi.fstype);
        MenuItem {
            label:          ec.name.clone(),
            detail,
            is_hidden:      ec.hidden,
            entry_index:    i,
            chainload_path: None,
            chainload_part: None,
        }
    }).collect();

    for (label, path, handle) in discover_chainload_targets(st, partitions) {
        items.push(MenuItem {
            label,
            detail:         "[chainload]".into(),
            is_hidden:      false,
            entry_index:    usize::MAX,
            chainload_path: Some(path),
            chainload_part: Some(handle),
        });
    }

    let mut selected: usize = 0;
    let mut countdown: i32  = global.timeout as i32;
    let timeout_total       = global.timeout as i32;
    let mut need_redraw     = true;

    loop {
        let visible: Vec<&MenuItem> = items.iter()
            .filter(|m| show_hidden || !m.is_hidden)
            .collect();

        if visible.is_empty() {
            show_error(st, "No boot entries found.");
            return BootMenuAction::Reboot;
        }
        if selected >= visible.len() { selected = visible.len() - 1; }

        if need_redraw {
            render_menu(st, global, &visible, selected, countdown, timeout_total);
            need_redraw = false;
        }

        // 1-second tick = 10,000,000 × 100 ns
        let key = wait_for_key(st, 10_000_000);

        match key {
            Some(k) => {
                countdown   = -1;
                need_redraw = true;
                match k {
                    Key::Special(s) => match s.0 {
                        0x01 => { // UP
                            selected = if selected == 0 { visible.len() - 1 } else { selected - 1 };
                        }
                        0x02 => { // DOWN
                            selected = if selected + 1 < visible.len() { selected + 1 } else { 0 };
                        }
                        0x14 => return BootMenuAction::Reboot,    // F10
                        0x15 => return BootMenuAction::Shutdown,  // F11
                        0x16 => { show_hidden = !show_hidden; }   // F12
                        0x17 => return BootMenuAction::Reboot,    // ESC
                        _    => {}
                    },
                    Key::Printable(c) => {
                        let ch = char::from_u32(u16::from(c) as u32).unwrap_or('\0');
                        match ch {
                            '\r' | '\n' => {
                                let m = visible[selected];
                                if m.entry_index == usize::MAX {
                                    return BootMenuAction::Chainload(
                                        m.chainload_path.clone().unwrap(),
                                        m.chainload_part.unwrap(),
                                    );
                                }
                                return BootMenuAction::BootEntry(m.entry_index);
                            }
                            'e' | 'E' => {
                                let m_idx = visible[selected].entry_index;
                                if m_idx != usize::MAX {
                                    if let Some(extra) = prompt_edit_cmdline(st) {
                                        return BootMenuAction::EditAndBoot(m_idx, extra);
                                    }
                                    need_redraw = true;
                                }
                            }
                            'h' | 'H' | '?' => {
                                show_help_screen(st);
                                need_redraw = true;
                            }
                            c if c.is_ascii_digit() => {
                                let n = (c as u8 - b'0') as usize;
                                if n > 0 && n <= visible.len() {
                                    let m = visible[n - 1];
                                    if m.entry_index == usize::MAX {
                                        return BootMenuAction::Chainload(
                                            m.chainload_path.clone().unwrap(),
                                            m.chainload_part.unwrap(),
                                        );
                                    }
                                    return BootMenuAction::BootEntry(m.entry_index);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            None => {
                // Timer tick (1 second).
                if countdown > 0 {
                    countdown -= 1;
                    need_redraw = true;
                    if countdown == 0 {
                        return BootMenuAction::Timeout;
                    }
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §6 — Menu rendering
// ════════════════════════════════════════════════════════════════════════════

const MENU_W:  usize = 78;
const OUTER_W: usize = 80;

fn render_menu(
    st:         &mut SystemTable<Boot>,
    global:     &BootManagerConfig,
    visible:    &[&MenuItem],
    selected:   usize,
    countdown:  i32,
    total_secs: i32,
) {
    let out = st.stdout();
    let (_, rows) = text_dims(out);
    clear(out);

    // ── Row 0: top border ─────────────────────────────────────────────────
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, 0, OUTER_W, BOX_TL, BOX_HH, BOX_TR);

    // ── Row 1: title bar ──────────────────────────────────────────────────
    set_colors(out, Color::White, Color::Blue);
    cursor(out, 0, 1);
    print_str(out, BOX_VV);
    print_centered(out, 1, 1, &global.theme.title, MENU_W);
    cursor(out, OUTER_W - 1, 1);
    print_str(out, BOX_VV);

    // ── Row 2: subtitle ───────────────────────────────────────────────────
    set_colors(out, Color::LightCyan, Color::Blue);
    cursor(out, 0, 2);
    print_str(out, BOX_VV);
    print_centered(out, 1, 2, "Boot manager  v1.0.0", MENU_W);
    cursor(out, OUTER_W - 1, 2);
    print_str(out, BOX_VV);

    // ── Row 3: divider ────────────────────────────────────────────────────
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, 3, OUTER_W, BOX_ML, BOX_HH, BOX_MR);

    // ── Row 4: blank ──────────────────────────────────────────────────────
    draw_side_row(out, 4);

    // ── Entries ───────────────────────────────────────────────────────────
    let entry_start_row = 5usize;
    for (i, item) in visible.iter().enumerate() {
        let row = entry_start_row + i;
        if row >= rows.saturating_sub(6) { break; }

        cursor(out, 0, row);
        set_colors(out, Color::Cyan, Color::Black);
        print_str(out, BOX_VV);

        if i == selected {
            set_colors(out, Color::Black, Color::Yellow);
            let line = format!(
                " {} {:<46}  {:>20} ",
                SEL_ARR, truncate(&item.label, 46), truncate(&item.detail, 20),
            );
            print_str(out, &line);
        } else if item.is_hidden {
            set_colors(out, Color::DarkGray, Color::Black);
            let line = format!(
                " * {:<46}  {:>20} ",
                truncate(&item.label, 46), truncate(&item.detail, 20),
            );
            print_str(out, &line);
        } else {
            set_fg_bg(out, global.theme.fg_color, global.theme.bg_color);
            let line = format!(
                "   {:<46}  {:>20} ",
                truncate(&item.label, 46), truncate(&item.detail, 20),
            );
            print_str(out, &line);
        }

        set_colors(out, Color::Cyan, Color::Black);
        cursor(out, OUTER_W - 1, row);
        print_str(out, BOX_VV);
    }

    // Blank row after entries.
    let blank_row = entry_start_row + visible.len();
    if blank_row < rows.saturating_sub(6) {
        draw_side_row(out, blank_row);
    }

    // ── Timer / progress divider ──────────────────────────────────────────
    let timer_top = rows.saturating_sub(5);
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, timer_top, OUTER_W, BOX_ML, BOX_HH, BOX_MR);

    // ── Countdown / manual row ────────────────────────────────────────────
    let timer_row = timer_top + 1;
    cursor(out, 0, timer_row);
    set_colors(out, Color::Cyan, Color::Black);
    print_str(out, BOX_VV);

    if countdown > 0 && total_secs > 0 {
        let bar_w   = 36usize;
        let elapsed = (total_secs - countdown) as usize;
        let bar     = progress_bar(elapsed, total_secs as usize, bar_w);
        set_colors(out, Color::Yellow, Color::Black);
        let status  = format!(" Auto-boot in {:2}s  [{}]  ", countdown, bar);
        print_str(out, &truncate_pad(&status, MENU_W));
    } else {
        set_colors(out, Color::LightGray, Color::Black);
        print_str(out, &truncate_pad("  Manual selection mode \u{2014} press Enter to boot", MENU_W));
    }

    cursor(out, OUTER_W - 1, timer_row);
    set_colors(out, Color::Cyan, Color::Black);
    print_str(out, BOX_VV);

    // ── Help divider ──────────────────────────────────────────────────────
    let help_div = timer_top + 2;
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, help_div, OUTER_W, BOX_ML, BOX_HH, BOX_MR);

    // ── Help row ──────────────────────────────────────────────────────────
    if global.theme.show_help {
        let help_row = help_div + 1;
        cursor(out, 0, help_row);
        set_colors(out, Color::Cyan, Color::Black);
        print_str(out, BOX_VV);
        set_colors(out, Color::LightGray, Color::Black);
        print_str(out, &truncate_pad(
            "  \u{2191}\u{2193}=Sel  Enter=Boot  E=Edit  H=Help  F12=Hidden  F10=Reset  F11=Off",
            MENU_W,
        ));
        cursor(out, OUTER_W - 1, help_row);
        set_colors(out, Color::Cyan, Color::Black);
        print_str(out, BOX_VV);
    }

    // ── Bottom border ─────────────────────────────────────────────────────
    let bot_row = rows.saturating_sub(1);
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, bot_row, OUTER_W, BOX_BL, BOX_HH, BOX_BR);

    set_fg_bg(out, global.theme.fg_color, global.theme.bg_color);
}

fn draw_side_row(out: &mut Output, row: usize) {
    cursor(out, 0, row);
    set_colors(out, Color::Cyan, Color::Black);
    print_str(out, BOX_VV);
    let blank: String = core::iter::repeat(' ').take(MENU_W).collect();
    set_colors(out, Color::White, Color::Black);
    print_str(out, &blank);
    cursor(out, OUTER_W - 1, row);
    set_colors(out, Color::Cyan, Color::Black);
    print_str(out, BOX_VV);
}

// ════════════════════════════════════════════════════════════════════════════
//  §7 — Command-line editor
// ════════════════════════════════════════════════════════════════════════════

fn prompt_edit_cmdline(st: &mut SystemTable<Boot>) -> Option<String> {
    let out = st.stdout();
    let (cols, rows) = text_dims(out);
    let row          = rows.saturating_sub(3);

    set_colors(out, Color::Cyan, Color::Black);
    cursor(out, 0, row);
    let blank: String = core::iter::repeat(' ').take(cols).collect();
    print_str(out, &blank);

    cursor(out, 1, row);
    set_colors(out, Color::Yellow, Color::Black);
    print_str(out, " Extra kernel args: ");
    set_colors(out, Color::White, Color::Black);

    let max_len = cols.saturating_sub(22);
    let mut buf = String::new();

    loop {
        let key = wait_for_key(st, u64::MAX)?;
        match key {
            Key::Printable(c) => {
                let ch = char::from_u32(u16::from(c) as u32).unwrap_or('\0');
                match ch {
                    '\r' | '\n' => return Some(buf),
                    '\x1b'      => return None,
                    '\x08' | '\x7F' => {
                        if buf.pop().is_some() {
                            let col = 21 + buf.len();
                            let out = st.stdout();
                            cursor(out, col, row);
                            print_str(out, " ");
                            cursor(out, col, row);
                        }
                    }
                    c if (c.is_ascii_graphic() || c == ' ') && buf.len() < max_len => {
                        buf.push(c);
                        let out = st.stdout();
                        print_str(out, &format!("{}", c));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  §8 — Help screen
// ════════════════════════════════════════════════════════════════════════════

fn show_help_screen(st: &mut SystemTable<Boot>) {
    let out = st.stdout();
    clear(out);
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, 0, OUTER_W, BOX_TL, BOX_HH, BOX_TR);

    set_colors(out, Color::White, Color::Blue);
    cursor(out, 0, 1);
    print_str(out, BOX_VV);
    print_centered(out, 1, 1, "PuppyBoot V1  \u{2014}  Help", MENU_W);
    cursor(out, OUTER_W - 1, 1);
    print_str(out, BOX_VV);

    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, 2, OUTER_W, BOX_ML, BOX_HH, BOX_MR);

    let lines: &[&str] = &[
        "",
        "  Keyboard shortcuts:",
        "    \u{2191} / \u{2193}        Move selection up / down",
        "    Enter           Boot selected entry",
        "    1 \u{2013} 9           Quick-boot entry by number",
        "    E               Edit kernel command line",
        "    H  /  ?         Show this help screen",
        "    ESC / F10       Reboot",
        "    F11             Shutdown / power off",
        "    F12             Toggle hidden entries",
        "",
        "  Entries marked  *  are hidden. Toggle with F12 or set",
        "  show_hidden = true in /EFI/puppyboot/loader.conf.",
        "",
        "  Press any key to return...",
    ];

    for (i, line) in lines.iter().enumerate() {
        let row = 3 + i;
        cursor(out, 0, row);
        set_colors(out, Color::Cyan, Color::Black);
        print_str(out, BOX_VV);
        set_colors(out, Color::LightGray, Color::Black);
        print_str(out, &truncate_pad(line, MENU_W));
        cursor(out, OUTER_W - 1, row);
        set_colors(out, Color::Cyan, Color::Black);
        print_str(out, BOX_VV);
    }

    let bot = 3 + lines.len();
    set_colors(out, Color::Cyan, Color::Black);
    draw_hline(out, 0, bot, OUTER_W, BOX_BL, BOX_HH, BOX_BR);

    wait_for_any_key(st);
}

// ════════════════════════════════════════════════════════════════════════════
//  §9 — Error display
// ════════════════════════════════════════════════════════════════════════════

pub fn show_error(st: &mut SystemTable<Boot>, msg: &str) {
    let out = st.stdout();
    set_colors(out, Color::White, Color::Red);
    cursor(out, 0, 0);
    print_str(out, &format!(" \u{2716} ERROR: {:<69} ", truncate(msg, 69)));
    set_colors(out, Color::LightGray, Color::Black);
    cursor(out, 0, 2);
    print_str(out, "  Press any key to continue...");
    wait_for_any_key(st);
}

// ════════════════════════════════════════════════════════════════════════════
//  §10 — Chainload discovery
// ════════════════════════════════════════════════════════════════════════════
//
//  Scan well-known EFI directories on every FAT-family partition for .efi
//  binaries we don't recognize as our own. The result is appended to the
//  menu so users can chainload Windows, GRUB, systemd-boot, etc., even
//  when no explicit entry config exists.

fn discover_chainload_targets(
    st:         &mut SystemTable<Boot>,
    partitions: &[PartitionInfo],
) -> Vec<(String, String, Handle)> {
    let bs = st.boot_services();
    partitions.iter()
        .filter(|p| matches!(p.fstype.as_str(), "fat32" | "fat16" | "fat12"))
        .flat_map(|part| {
            let search_dirs = ["/EFI", "/EFI/BOOT", "/EFI/Microsoft/Boot",
                               "/EFI/systemd", "/EFI/refind"];
            let handle = part.handle;
            let mut hits: Vec<(String, String, Handle)> = Vec::new();
            for dir in search_dirs.iter() {
                let files = crate::disk::list_fat_dir(bs, handle, dir).unwrap_or_default();
                for f in files {
                    let lower = f.to_ascii_lowercase();
                    if !lower.ends_with(".efi") { continue; }
                    if lower == "bootx64.efi" || lower == "puppyboot.efi"
                    || lower == "bootia32.efi" || lower == "bootaa64.efi" {
                        continue;
                    }
                    let label = format!("[chain] {}", f.trim_end_matches(".efi"));
                    let path  = format!("{}/{}", dir, f);
                    hits.push((label, path, handle));
                }
            }
            hits
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
//  §11 — String utilities
// ════════════════════════════════════════════════════════════════════════════

fn truncate(s: &str, max_chars: usize) -> String {
    // char_indices iterator — counts USV (Unicode scalar values), not bytes.
    let end = s.char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    s[..end].to_string()
}

fn truncate_pad(s: &str, width: usize) -> String {
    let t   = truncate(s, width);
    let len = t.chars().count();
    let pad: String = core::iter::repeat(' ')
        .take(width.saturating_sub(len))
        .collect();
    let mut r = t;
    r.push_str(&pad);
    r
}
