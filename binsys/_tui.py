"""Curses-based TUI for binsys system management."""

from __future__ import annotations

import curses
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from binsys._boot import (
    _ensure_bootloader,
)
from binsys._crypto import (
    do_app_unlock,
    do_encrypt,
    do_hash,
    do_lock,
    do_protect,
    do_unlock,
)
from binsys._frugal import (
    convert_to_frugal,
    do_frugal_list_snapshots,
    do_frugal_merge,
    do_frugal_rollback,
    do_frugal_save_snapshot,
)
from binsys._image import (
    do_check,
    do_clone,
    do_delete,
    do_export,
    do_import,
    do_mount,
    do_new,
    do_rename,
    do_resize,
    do_snap,
    do_umount,
)
from binsys._iso import do_iso_create
from binsys._qemu import _build_qcmd
from binsys._util import (
    MOUNTS,
    SCRIPTS_DIR,
    TYPES,
    WIZARD_SCRIPTS,
    _df_info,
    all_systems,
    human,
    is_mounted,
    load_keybindings,
    sh,
    sys_dir,
)

logger = logging.getLogger("binsys")


# ── spinner / progress helpers ────────────────────────────────────────────────


def _init_colors() -> None:
    if curses.has_colors():
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_BLUE, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)


def _spinner_gen() -> Any:
    while True:
        for ch in "⣾⣽⣻⢿⡿⣟⣯⣷":
            yield ch


def draw_spinner(win: Any, y: int, x: int, phase: int) -> None:
    chars = "⣾⣽⣻⢿⡿⣟⣯⣷"
    try:
        win.addstr(y, x, chars[phase % len(chars)])
        win.refresh()
    except curses.error:
        pass


def with_spinner(win: Any, msg: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a function while showing a spinner on the status line."""
    spinner = _spinner_gen()
    phase = 0
    win.nodelay(True)
    try:
        result = None
        exc = None

        def target() -> None:
            nonlocal result, exc
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                exc = e

        t = threading.Thread(target=target, daemon=True)
        t.start()

        while t.is_alive():
            try:
                ch = win.getch()
                if ch != -1:
                    break
            except Exception:
                pass
            spinner_ch = next(spinner)
            try:
                win.addstr(0, 0, f" {spinner_ch} {msg}   ")
                win.clrtoeol()
                win.refresh()
            except curses.error:
                pass
            phase += 1
            time.sleep(0.08)
        if exc:
            raise exc
        return result
    finally:
        win.nodelay(False)
        try:
            win.addstr(0, 0, " " * (len(msg) + 6))
            win.refresh()
        except curses.error:
            pass


# ── dialog helpers ────────────────────────────────────────────────────────────


def _type_icon(t: str) -> str:
    return {"ext4": "💾", "overlay": "📦", "squashfs": "🗜", "fat32": "💿", "iso": "📀", "frugal": "🪶"}.get(t, "❓")


def _type_color(t: str) -> int:
    return {"ext4": 1, "overlay": 5, "squashfs": 2, "fat32": 4, "iso": 6, "frugal": 5}.get(t, 3)


def _status_badges(meta: dict[str, Any]) -> str:
    parts: list[str] = []
    if meta.get("encrypted"):
        parts.append("🔒")
    if meta.get("frugal"):
        parts.append("🪶")
    if meta.get("source"):
        parts.append(f"📥{meta['source']}")
    return " ".join(parts)


def _safe_addstr(win: Any, y: int, x: int, s: str, *args: Any, **kwargs: Any) -> None:
    try:
        win.addstr(y, x, s, *args, **kwargs)
    except curses.error:
        pass


def _draw_box(win: Any, y1: int, x1: int, y2: int, x2: int) -> None:
    try:
        win.hline(y1, x1, curses.ACS_HLINE, x2 - x1)
        win.hline(y2, x1, curses.ACS_HLINE, x2 - x1)
        win.vline(y1, x1, curses.ACS_VLINE, y2 - y1)
        win.vline(y1, x2, curses.ACS_VLINE, y2 - y1)
        win.addch(y1, x1, curses.ACS_ULCORNER)
        win.addch(y1, x2, curses.ACS_URCORNER)
        win.addch(y2, x1, curses.ACS_LLCORNER)
        win.addch(y2, x2, curses.ACS_LRCORNER)
    except curses.error:
        pass


def _text_field(win: Any, y: int, x: int, width: int, label: str, default: str = "",
                password: bool = False) -> str:
    """Draw a labeled text field and return user input."""
    _safe_addstr(win, y, x, label)
    field_x = x + len(label) + 1
    field_win = win.derwin(1, width, y, field_x)
    curses.curs_set(1)
    result = ""
    while True:
        field_win.erase()
        display = "*" * len(result) if password else result
        field_win.addstr(0, 0, display[:width - 1])
        field_win.refresh()
        ch = field_win.getch()
        if ch in (curses.KEY_ENTER, 10, 13):
            result = result or default
            break
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            result = result[:-1]
        elif 32 <= ch < 127:
            if len(result) < width - 1:
                result += chr(ch)
        elif ch == 27:  # Escape
            result = default
            break
    curses.curs_set(0)
    return result


def _select_field(win: Any, y: int, x: int, label: str, options: list[str],
                  default_idx: int = 0) -> int:
    """Draw a labeled select field with left/right navigation."""
    _safe_addstr(win, y, x, label)
    idx = default_idx
    while True:
        _safe_addstr(win, y, x + len(label) + 1, f"◀ {options[idx]} ▶ ", curses.A_REVERSE)
        ch = win.getch()
        if ch in (curses.KEY_LEFT, ord(",")):
            idx = (idx - 1) % len(options)
        elif ch in (curses.KEY_RIGHT, ord(".")):
            idx = (idx + 1) % len(options)
        elif ch in (curses.KEY_ENTER, 10, 13, ord(" ")):
            break
        elif ch == 27:
            idx = default_idx
            break
    return idx


def input_dialog(stdscr: Any, title: str, fields: list[tuple[str, str, bool]],
                 width: int = 50) -> dict[str, str] | None:
    """Generic input dialog returning dict of field_name -> value, or None on cancel."""
    max_y, max_x = stdscr.getmaxyx()
    dialog_h = len(fields) + 6
    dialog_w = min(width, max_x - 4)
    y0 = (max_y - dialog_h) // 2
    x0 = (max_x - dialog_w) // 2

    sub = stdscr.derwin(dialog_h, dialog_w, y0, x0)
    sub.keypad(1)
    result: dict[str, str] = {}
    try:
        _draw_box(sub, 0, 0, dialog_h - 1, dialog_w - 1)
        _safe_addstr(sub, 1, 2, title, curses.A_BOLD)
        for i, (field_name, label, is_password) in enumerate(fields):
            val = _text_field(sub, i + 3, 2, dialog_w - 6, label, password=is_password)
            if val is None:
                return None
            result[field_name] = val
        return result
    except curses.error:
        return None
    finally:
        stdscr.touchwin()
        stdscr.refresh()


def confirm_dialog(stdscr: Any, msg: str, default: bool = False) -> bool:
    """Simple Yes/No confirmation dialog."""
    max_y, max_x = stdscr.getmaxyx()
    w = min(len(msg) + 10, max_x - 4)
    h = 5
    y0 = (max_y - h) // 2
    x0 = (max_x - w) // 2
    sub = stdscr.derwin(h, w, y0, x0)
    sub.keypad(1)
    selected = 0 if default else 1
    try:
        _draw_box(sub, 0, 0, h - 1, w - 1)
        _safe_addstr(sub, 1, 2, msg[:w - 4], curses.A_BOLD)
        while True:
            opts = [(" Yes ", 1 if selected == 0 else 0), (" No ", 1 if selected == 1 else 0)]
            sx = 2
            for label, hilite in opts:
                attr = curses.A_REVERSE if hilite else curses.A_NORMAL
                _safe_addstr(sub, 3, sx, label, attr)
                sx += len(label) + 1
            sub.refresh()
            ch = sub.getch()
            if ch == curses.KEY_LEFT:
                selected = max(0, selected - 1)
            elif ch == curses.KEY_RIGHT:
                selected = min(1, selected + 1)
            elif ch in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return selected == 0
            elif ch == 27:
                return default
    finally:
        stdscr.touchwin()
        stdscr.refresh()


def message_dialog(stdscr: Any, msg: str, title: str = "Info") -> None:
    """Show a message box and wait for key press."""
    max_y, max_x = stdscr.getmaxyx()
    lines = msg.split("\n")
    h = min(len(lines) + 4, max_y - 2)
    w = min(max(len(line) for line in lines) + 6, max_x - 4)
    y0 = (max_y - h) // 2
    x0 = (max_x - w) // 2
    sub = stdscr.derwin(h, w, y0, x0)
    try:
        _draw_box(sub, 0, 0, h - 1, w - 1)
        _safe_addstr(sub, 1, 2, title, curses.A_BOLD)
        for i, line in enumerate(lines):
            _safe_addstr(sub, i + 2, 2, line[:w - 4])
        _safe_addstr(sub, h - 2, 2, "Press any key to continue")
        sub.refresh()
        sub.getch()
    finally:
        stdscr.touchwin()
        stdscr.refresh()


# ── TUI main class ────────────────────────────────────────────────────────────


class BinSysTUI:
    """Curses-based interactive TUI for managing filesystem images."""

    def __init__(self, stdscr: Any) -> None:
        self.stdscr = stdscr
        self.keybindings = load_keybindings()
        self.selected = 0
        self.msg = ""
        _init_colors()
        curses.curs_set(0)
        curses.use_default_colors()
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_BLUE, -1)
            curses.init_pair(5, curses.COLOR_CYAN, -1)
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)

    def _reload(self) -> None:
        self.systems = all_systems()
        if self.selected >= len(self.systems):
            self.selected = max(0, len(self.systems) - 1)

    def _set_msg(self, s: str) -> None:
        self.msg = s

    def draw(self) -> None:
        self._reload()
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.erase()

        # Title bar
        title = f" BinSys — {len(self.systems)} systems "
        _safe_addstr(self.stdscr, 0, (max_x - len(title)) // 2, title, curses.A_BOLD)

        # System list
        if not self.systems:
            center = max_y // 2
            _safe_addstr(self.stdscr, center, max_x // 2 - 10,
                         " No systems — press 'n' to create one ",
                         curses.A_DIM)
        else:
            for i, meta in enumerate(self.systems):
                y = 2 + i
                if y >= max_y - 3:
                    break
                attrs = curses.A_REVERSE if i == self.selected else curses.A_NORMAL
                name = meta["name"]
                t = meta.get("type", "?")
                icon = _type_icon(t)
                size_str = ""
                d = sys_dir(name)
                if meta.get("disk"):
                    p = d / meta["disk"]
                    if p.exists():
                        size_str = human(p.stat().st_size)
                elif meta.get("base"):
                    p = d / meta["base"]
                    if p.exists():
                        size_str = human(p.stat().st_size)
                mounted = " 🔗" if is_mounted(MOUNTS / name) else ""
                badges = _status_badges(meta)
                line = f" {icon} {name:<20} {t:<10} {size_str:>8}{mounted} {badges}"
                _safe_addstr(self.stdscr, y, 2, line[:max_x - 4], attrs)

        # Status bar
        status = f" {self.msg} " if self.msg else ""
        _safe_addstr(self.stdscr, max_y - 1, 0, status[:max_x - 1],
                     curses.A_REVERSE if self.msg else curses.A_NORMAL)

        # Help (bottom)
        help_keys = ", ".join(f"{v}={k}" for k, v in sorted(self.keybindings.items(),
                            key=lambda x: x[0]))
        help_text = f" {help_keys} "
        _safe_addstr(self.stdscr, max_y - 1, max_x - len(help_text) - 1,
                     help_text, curses.A_DIM)

        self.stdscr.refresh()

    def _suspend(self) -> None:
        """Suspend the TUI (e.g., to run a subprocess)."""
        curses.endwin()
        os.system("stty sane" if shutil.which("stty") else "")

    def _resume(self) -> None:
        """Resume the TUI after suspension."""
        self.stdscr = curses.initscr()
        curses.curs_set(0)
        curses.use_default_colors()
        _init_colors()
        self.stdscr.refresh()

    def _shell_action(self, meta: dict[str, Any], name: str) -> None:
        self._suspend()
        d = sys_dir(name)
        print(f"--- Shell action: {name} ---")
        # Check what kind of action
        img_name = meta.get("disk", "disk.img")
        img_path = d / img_name
        if img_path.exists():
            self._set_msg(f"Image: {img_path}")
        else:
            self._set_msg("No image found")
        input("Press Enter to return to TUI...")
        self._resume()

    def action_new(self) -> None:
        self._suspend()
        try:
            name = input("System name: ").strip()
            if not name:
                return
            print("Types:", ", ".join(TYPES))
            t = input("Type [ext4]: ").strip() or "ext4"
            sz = input("Size [1G]: ").strip() or "1G"
            enc = input("Encrypt? (y/N): ").strip().lower() == "y"
            do_new(name, t, sz, encrypt=enc)
            self._set_msg(f"Created '{name}' ({t}, {sz})")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_convert_frugal(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            convert_to_frugal(name)
            self._set_msg(f"Converted '{name}' to frugal")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_fix_esp(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            efi_path = _ensure_bootloader(name)
            if not efi_path:
                self._set_msg("Bootloader not available")
                return
            d = sys_dir(name)
            if not (d / "disk.img").exists():
                self._set_msg("No disk image found")
                return
            r = sh(["losetup", "-j", str(d / "disk.img")], capture=True, sudo=True, check=False)
            loop_dev = r.stdout.strip().split(":")[0] if r.stdout else None
            if not loop_dev:
                r = sh(["losetup", "--find", "--show", "-P", str(d / "disk.img")],
                       capture=True, sudo=True)
                loop_dev = r.stdout.strip()
            esp_part = f"{loop_dev}p1" if loop_dev else None
            if not esp_part or not os.path.exists(esp_part):
                self._set_msg("ESP partition not found")
                return
            esp_mnt = d / ".esp_mnt"
            esp_mnt.mkdir(exist_ok=True)
            try:
                sh(["mount", esp_part, str(esp_mnt)], sudo=True)
                efi_dir = esp_mnt / "EFI" / "BOOT"
                efi_dir.mkdir(parents=True, exist_ok=True)
                sh(["cp", efi_path, str(efi_dir / "BOOTX64.EFI")])
                self._set_msg("Bootloader installed to ESP")
            finally:
                sh(["umount", str(esp_mnt)], sudo=True, check=False)
                shutil.rmtree(esp_mnt, ignore_errors=True)
                if loop_dev:
                    sh(["losetup", "-d", loop_dev], sudo=True, check=False)
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_delete(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        if confirm_dialog(self.stdscr, f"Delete '{name}'?", default=False):
            self._suspend()
            try:
                do_delete(name)
                self._set_msg(f"Deleted '{name}'")
            except RuntimeError as e:
                self._set_msg(str(e))
            finally:
                self._resume()
        else:
            self._set_msg("Canceled")

    def action_run(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            cmd = _build_qcmd(name, meta)
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)
        except RuntimeError as e:
            self._set_msg(str(e))
        except KeyboardInterrupt:
            pass
        finally:
            self._resume()

    def action_mount_toggle(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            if is_mounted(MOUNTS / name):
                do_umount(name)
                self._set_msg(f"Unmounted '{name}'")
            else:
                path = do_mount(name)
                self._set_msg(f"Mounted '{name}' at {path}")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_snap(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            do_snap(name)
            self._set_msg(f"Snapshotted '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_import(self, meta: dict[str, Any] | None = None) -> None:
        self._suspend()
        try:
            src = input("Source path: ").strip()
            if not src:
                return
            name = input("Name (enter for auto): ").strip() or None
            do_import(src, name)
            self._set_msg(f"Imported from {src}")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_resize(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            new_sz = input("New size (e.g. 4G): ").strip()
            if not new_sz:
                return
            do_resize(name, new_sz)
            self._set_msg(f"Resized '{name}' to {new_sz}")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_clone(self, meta: dict[str, Any]) -> None:
        src = meta["name"]
        self._suspend()
        try:
            dst = input(f"New name [{src}-copy]: ").strip() or f"{src}-copy"
            do_clone(src, dst)
            self._set_msg(f"Cloned '{src}' -> '{dst}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_rename(self, meta: dict[str, Any]) -> None:
        old = meta["name"]
        self._suspend()
        try:
            new = input("New name: ").strip()
            if not new:
                return
            do_rename(old, new)
            self._set_msg(f"Renamed '{old}' -> '{new}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_export(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            dst_path, size = do_export(name)
            self._set_msg(f"Exported '{name}' to {dst_path} ({human(size)})")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_check(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            do_check(name)
            self._set_msg(f"Check complete for '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_info(self, meta: dict[str, Any]) -> None:
        lines: list[str] = []
        lines.append(f"Name:      {meta['name']}")
        lines.append(f"Type:      {meta.get('type', '?')}")
        lines.append(f"Created:   {meta.get('created', '?')}")
        lines.append(f"Encrypted: {'Yes' if meta.get('encrypted') else 'No'}")
        lines.append(f"Frugal:    {'Yes' if meta.get('frugal') else 'No'}")
        if meta.get("source"):
            lines.append(f"Source:    {meta['source']}")
        d = sys_dir(meta["name"])
        if meta.get("disk"):
            p = d / meta["disk"]
            if p.exists():
                lines.append(f"Image:     {meta['disk']} ({human(p.stat().st_size)})")
        if meta.get("base"):
            p = d / meta["base"]
            if p.exists():
                lines.append(f"Base:      {meta['base']} ({human(p.stat().st_size)})")
        if meta.get("save"):
            p = d / meta["save"]
            if p.exists():
                lines.append(f"Save:      {meta['save']} ({human(p.stat().st_size)})")
        info = _df_info(str(MOUNTS / meta["name"])) if is_mounted(MOUNTS / meta["name"]) else None
        if info:
            used, total = info
            lines.append(f"Mounted:   {human(used)} / {human(total)}")
        message_dialog(self.stdscr, "\n".join(lines), title="System Info")

    def _ensure_unlocked(self, meta: dict[str, Any]) -> bool:
        """Ensure app-level protection is unlocked; return True if OK."""
        name = meta["name"]
        from binsys._crypto import _load_app_locks
        locks = _load_app_locks()
        entry = locks.get(name)
        if entry and not entry.get("unlocked", False):
            self._suspend()
            try:
                pw = input(f"App password for '{name}': ")
                if not pw:
                    return False
                do_app_unlock(name, pw)
                return True
            except RuntimeError as e:
                self._set_msg(str(e))
                return False
            finally:
                self._resume()
        return True

    def action_protect(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            pw = input("App password: ")
            if not pw:
                return
            do_protect(name, pw)
            self._set_msg(f"Protected '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_encrypt_toggle(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            if meta.get("encrypted"):
                do_lock(name)
                self._set_msg(f"Locked '{name}'")
            else:
                do_encrypt(name)
                self._set_msg(f"Encrypted '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_unlock(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            if not meta.get("encrypted"):
                self._set_msg("Not encrypted")
                return
            do_unlock(name)
            self._set_msg(f"Unlocked '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_hash(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            do_hash(name)
            self._set_msg(f"Hash computed for '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_frugal_menu(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        if meta.get("type") != "overlay" or not meta.get("frugal"):
            self._set_msg("Not a frugal system")
            return

        snaps = do_frugal_list_snapshots(name)
        snap_names = [s["name"] for s in snaps]

        if not snap_names:
            self._suspend()
            try:
                yn = input("No snapshots. Create one? (y/N): ").strip().lower()
                if yn == "y":
                    do_frugal_save_snapshot(name)
            finally:
                self._resume()
            return

        self._suspend()
        try:
            print("Snapshots:")
            for i, sn in enumerate(snap_names):
                print(f"  {i + 1}. {sn}")
            print("  r. Rollback")
            print("  m. Merge")
            print("  q. Back")
            choice = input("Choice: ").strip().lower()
            if choice == "r":
                snap_i = input("Snapshot number to rollback: ").strip()
                if snap_i.isdigit():
                    idx = int(snap_i) - 1
                    if 0 <= idx < len(snap_names):
                        do_frugal_rollback(name, snap_names[idx])
            elif choice == "m":
                do_frugal_merge(name)
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_iso(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            output = input("Output path (enter for auto): ").strip() or None
            do_iso_create(name, output)
            self._set_msg(f"ISO created for '{name}'")
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_wizard(self, meta: dict[str, Any] | None = None) -> None:
        self._suspend()
        try:
            print("Available wizards:")
            for n, d in WIZARD_SCRIPTS:
                print(f"  {n:<20} {d}")
            script = input("Wizard name (enter to cancel): ").strip()
            if not script:
                return
            wizard_path = SCRIPTS_DIR / script
            if not wizard_path.exists():
                self._set_msg(f"Wizard '{script}' not found")
                return
            sh([str(wizard_path)])
        except RuntimeError as e:
            self._set_msg(str(e))
        finally:
            self._resume()

    def action_help(self) -> None:
        lines = [
            "Key Bindings:",
        ]
        for action, key in sorted(self.keybindings.items(), key=lambda x: x[0]):
            lines.append(f"  {key:<4} {action}")
        message_dialog(self.stdscr, "\n".join(lines), title="Help")

    def _key_cmd(self, ch: int) -> str | None:
        """Map a key press to an action name."""
        for action, key in self.keybindings.items():
            if ord(key) == ch:
                return action
        return None

    def run(self) -> None:
        self._reload()
        while True:
            self.draw()
            ch = self.stdscr.getch()
            if ch == -1:
                continue
            action = self._key_cmd(ch)
            if action is None:
                continue
            self._set_msg("")

            # Actions that don't need a selected system
            if action == "quit":
                break
            elif action == "new":
                self.action_new()
                continue
            elif action == "import":
                self.action_import()
                continue
            elif action == "wizard":
                self.action_wizard()
                continue

            # Actions that need a selected system
            if not self.systems:
                self._set_msg("No systems — press 'n' to create one")
                continue
            meta = self.systems[self.selected]

            # Navigation
            if action == "help":
                self.action_help()
            elif action == "info":
                self.action_info(meta)
            elif action == "delete":
                self.action_delete(meta)
            elif action == "run":
                self.action_run(meta)
            elif action == "mount":
                self.action_mount_toggle(meta)
            elif action == "snap":
                self.action_snap(meta)
            elif action == "frugal":
                self.action_convert_frugal(meta)
            elif action == "fix_esp":
                self.action_fix_esp(meta)
            elif action == "clone":
                self.action_clone(meta)
            elif action == "rename":
                self.action_rename(meta)
            elif action == "export":
                self.action_export(meta)
            elif action == "check":
                self.action_check(meta)
            elif action == "resize":
                self.action_resize(meta)
            elif action == "protect":
                self.action_protect(meta)
            elif action == "encrypt":
                self.action_encrypt_toggle(meta)
            elif action == "unlock":
                self.action_unlock(meta)
            elif action == "hash":
                self.action_hash(meta)
            elif action == "frugal_save" or action == "frugal_merge" or action == "frugal_roll":
                self.action_frugal_menu(meta)
            elif action == "iso":
                self.action_iso(meta)
            elif ch == curses.KEY_UP:
                self.selected = max(0, self.selected - 1)
            elif ch == curses.KEY_DOWN:
                self.selected = min(len(self.systems) - 1, self.selected + 1)


def cmd_tui() -> None:
    """Launch the curses TUI."""
    sys.stderr.write("Starting TUI...\n")
    try:
        curses.wrapper(lambda stdscr: BinSysTUI(stdscr).run())
    except KeyboardInterrupt:
        pass
