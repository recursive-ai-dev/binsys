"""Curses-based TUI for binsys system management."""

from __future__ import annotations

import contextlib
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

from binsys import BinSysError
from binsys._boot import (
    _ensure_bootloader,
)
from binsys._crypto import (
    _load_app_locks,
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
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_BLUE, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Title bar
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Status bar


def _spinner_gen() -> Any:
    while True:
        yield from "⣾⣽⣻⢿⡿⣟⣯⣷"


def with_spinner(win: Any, msg: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a function while showing a spinner on the status line."""
    spinner = _spinner_gen()
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

        max_y, _ = win.getmaxyx()
        while t.is_alive():
            try:
                ch = win.getch()
                if ch == ord('q'): # Allow early exit request
                     pass
            except Exception:
                pass

            spinner_ch = next(spinner)
            try:
                status = f" {spinner_ch} {msg}... "
                win.addstr(max_y - 1, 0, status, curses.color_pair(8))
                win.clrtoeol()
                win.refresh()
            except curses.error:
                pass
            time.sleep(0.1)

        if exc:
            raise exc
        return result
    finally:
        win.nodelay(False)


# ── dialog helpers ────────────────────────────────────────────────────────────


def _type_icon(t: str) -> str:
    return {
        "ext4": "💾",
        "overlay": "📦",
        "squashfs": "🗜",
        "fat32": "💿",
        "iso": "📀",
        "frugal": "🪶"
    }.get(t, "❓")


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
    with contextlib.suppress(curses.error):
        win.addstr(y, x, s, *args, **kwargs)


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
        sub.erase()
        _draw_box(sub, 0, 0, h - 1, w - 1)
        _safe_addstr(sub, 1, 2, title, curses.A_BOLD)
        for i, line in enumerate(lines[:h-4]):
            _safe_addstr(sub, i + 2, 2, line[:w - 4])
        _safe_addstr(sub, h - 2, 2, "Press any key...", curses.A_DIM)
        sub.refresh()
        sub.getch()
    finally:
        stdscr.touchwin()
        stdscr.refresh()


def confirm_dialog(stdscr: Any, msg: str, default: bool = False) -> bool:
    """Simple Yes/No confirmation dialog."""
    max_y, max_x = stdscr.getmaxyx()
    w = min(len(msg) + 12, max_x - 4)
    h = 5
    y0 = (max_y - h) // 2
    x0 = (max_x - w) // 2
    sub = stdscr.derwin(h, w, y0, x0)
    sub.keypad(1)
    selected = 0 if default else 1
    try:
        while True:
            sub.erase()
            _draw_box(sub, 0, 0, h - 1, w - 1)
            _safe_addstr(sub, 1, 2, msg[:w - 4], curses.A_BOLD)

            opts = [(" Yes ", 0), (" No ", 1)]
            sx = 2
            for label, idx in opts:
                attr = curses.A_REVERSE if selected == idx else curses.A_NORMAL
                _safe_addstr(sub, 3, sx, label, attr)
                sx += len(label) + 2

            sub.refresh()
            ch = sub.getch()
            if ch == curses.KEY_LEFT:
                selected = 0
            elif ch == curses.KEY_RIGHT:
                selected = 1
            elif ch in (curses.KEY_ENTER, 10, 13):
                return selected == 0
            elif ch == 27: # Esc
                return default
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
        self.msg_attr = curses.A_NORMAL
        self.systems: list[dict[str, Any]] = []
        _init_colors()
        curses.curs_set(0)

    def _reload(self) -> None:
        try:
            self.systems = all_systems()
        except Exception as e:
            self._set_msg(f"Reload failed: {e}", error=True)
            self.systems = []

        if self.selected >= len(self.systems):
            self.selected = max(0, len(self.systems) - 1)

    def _set_msg(self, s: str, error: bool = False) -> None:
        self.msg = s
        self.msg_attr = curses.color_pair(3) if error else curses.A_NORMAL

    def draw(self) -> None:
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.erase()

        # Title bar
        title = f" BinSys — {len(self.systems)} systems "
        self.stdscr.attron(curses.color_pair(7))
        self.stdscr.hline(0, 0, " ", max_x)
        _safe_addstr(self.stdscr, 0, (max_x - len(title)) // 2, title, curses.A_BOLD)
        self.stdscr.attroff(curses.color_pair(7))

        # System list
        if not self.systems:
            center = max_y // 2
            _safe_addstr(self.stdscr, center, (max_x - 30) // 2,
                         "No systems. Press 'n' for new.",
                         curses.A_DIM)
        else:
            for i, meta in enumerate(self.systems):
                y = 2 + i
                if y >= max_y - 2:
                    break

                is_sel = (i == self.selected)
                attrs = curses.A_REVERSE if is_sel else curses.A_NORMAL

                name = meta["name"]
                t = meta.get("type", "?")
                icon = _type_icon(t)

                size_str = ""
                d = sys_dir(name)
                for key in ("disk", "base"):
                    if meta.get(key):
                        p = d / meta[key]
                        if p.exists():
                            size_str = human(p.stat().st_size)
                            break

                mounted = " 🔗" if is_mounted(MOUNTS / name) else ""
                badges = _status_badges(meta)

                line = f" {icon} {name:<20} {t:<10} {size_str:>8}{mounted} {badges}"
                _safe_addstr(self.stdscr, y, 2, line[:max_x - 4], attrs)

        # Status bar / Footer
        self.stdscr.attron(curses.color_pair(8))
        self.stdscr.hline(max_y - 1, 0, " ", max_x)
        status = f" {self.msg} " if self.msg else " Ready "
        _safe_addstr(self.stdscr, max_y - 1, 0, status[:max_x - 1], self.msg_attr)

        # Help hint
        help_hint = " [h] Help  [q] Quit "
        _safe_addstr(self.stdscr, max_y - 1, max_x - len(help_hint), help_hint)
        self.stdscr.attroff(curses.color_pair(8))

        self.stdscr.refresh()

    def _suspend(self) -> None:
        curses.endwin()

    def _resume(self) -> None:
        self.stdscr.refresh()
        curses.doupdate()

    def _run_op(self, msg: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run an operation with a spinner and error handling."""
        try:
            return with_spinner(self.stdscr, msg, fn, *args, **kwargs)
        except Exception as e:
            self._set_msg(f"Error: {e}", error=True)
            logger.exception("Operation failed: %s", msg)
            return None

    def action_new(self) -> None:
        self._suspend()
        try:
            print("\n--- Create New System ---")
            name = input("Name: ").strip()
            if not name: return
            t = input(f"Type ({'/'.join(TYPES)}) [ext4]: ").strip() or "ext4"
            sz = input("Size [1G]: ").strip() or "1G"
            enc = input("Encrypt? (y/N): ").strip().lower() == "y"

            self._resume()
            self._run_op(f"Creating {name}", do_new, name, t, sz, encrypt=enc)
            self._set_msg(f"Created '{name}'")
        except KeyboardInterrupt:
            self._resume()
        finally:
            self._resume()

    def action_delete(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        if confirm_dialog(self.stdscr, f"Delete '{name}' permanently?"):
            self._run_op(f"Deleting {name}", do_delete, name)
            self._set_msg(f"Deleted '{name}'")

    def action_run(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        self._suspend()
        try:
            cmd = _build_qcmd(name, meta)
            print(f"\n--- Booting {name} ---")
            print(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd)
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter...")
        finally:
            self._resume()

    def action_mount_toggle(self, meta: dict[str, Any]) -> None:
        name = meta["name"]
        if is_mounted(MOUNTS / name):
            self._run_op(f"Unmounting {name}", do_umount, name)
            self._set_msg(f"Unmounted '{name}'")
        else:
            path = self._run_op(f"Mounting {name}", do_mount, name)
            if path:
                self._set_msg(f"Mounted at {path}")

    def action_info(self, meta: dict[str, Any]) -> None:
        lines = [
            f"Name:      {meta['name']}",
            f"Type:      {meta.get('type', '?')}",
            f"Created:   {meta.get('created', '?')}",
            f"Encrypted: {'Yes' if meta.get('encrypted') else 'No'}",
            f"Frugal:    {'Yes' if meta.get('frugal') else 'No'}",
        ]
        if meta.get("source"):
            lines.append(f"Source:    {meta['source']}")

        d = sys_dir(meta["name"])
        for key in ("disk", "base", "save"):
            if val := meta.get(key):
                p = d / val
                sz = human(p.stat().st_size) if p.exists() else "???"
                lines.append(f"{key.capitalize():<10} {val} ({sz})")

        message_dialog(self.stdscr, "\n".join(lines), title="System Info")

    def action_help(self) -> None:
        lines = ["Key Bindings:"]
        for act, key in sorted(self.keybindings.items()):
            lines.append(f"  {key:<4} {act}")
        message_dialog(self.stdscr, "\n".join(lines), title="Help")

    def run(self) -> None:
        while True:
            self._reload()
            self.draw()

            try:
                ch = self.stdscr.getch()
            except KeyboardInterrupt:
                break

            if ch == -1: continue

            # Map key to action
            action = None
            for act, key in self.keybindings.items():
                if ord(key) == ch:
                    action = act
                    break

            if not action:
                if ch == curses.KEY_UP:
                    self.selected = max(0, self.selected - 1)
                elif ch == curses.KEY_DOWN:
                    self.selected = min(len(self.systems) - 1, self.selected + 1)
                continue

            self._set_msg("")

            if action == "quit": break
            elif action == "help": self.action_help()
            elif action == "new": self.action_new()
            elif action == "wizard":
                self._suspend()
                print("\n--- Wizards ---")
                for i, (n, d) in enumerate(WIZARD_SCRIPTS):
                    print(f" {i+1}. {n:<15} - {d}")
                input("\nPress Enter to return...")
                self._resume()

            # Selected-system actions
            if self.systems:
                meta = self.systems[self.selected]
                if action == "delete": self.action_delete(meta)
                elif action == "run": self.action_run(meta)
                elif action == "mount": self.action_mount_toggle(meta)
                elif action == "info": self.action_info(meta)
                elif action == "check": self._run_op(f"Checking {meta['name']}", do_check, meta['name'])
                elif action == "hash": self._run_op(f"Hashing {meta['name']}", do_hash, meta['name'])


def cmd_tui() -> None:
    """Launch the interactive TUI."""
    os.environ.setdefault("ESCDELAY", "25")
    with contextlib.suppress(KeyboardInterrupt):
        curses.wrapper(lambda stdscr: BinSysTUI(stdscr).run())
