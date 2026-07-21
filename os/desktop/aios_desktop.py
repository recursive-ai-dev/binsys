#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Desktop Environment v2                          aios_desktop.py     ║
║  Agentic Intelligence Operating System — Pure Python 3.9+ / stdlib only     ║
║  Zero Tkinter. Zero X11. Zero Wayland. Zero external dependencies.          ║
║                                                                              ║
║  "The framebuffer is yours. Every pixel, every decision."                   ║
║                                                                              ║
║  Pipeline:                                                                   ║
║    /dev/input/event* → EvdevInputDriver → EventQueue → DesktopShell         ║
║    CAWallpaper (Layer) ──┐                                                   ║
║    AppWindows (Layer[]) ─┼→ Compositor → FramebufferDriver → /dev/fb0       ║
║    Taskbar (Layer) ──────┘                                                   ║
║                                                                              ║
║  §0   Preamble & Imports                                                     ║
║  §1   Math Primitives — CORDIC (Volder 1959), NR-√, HSV→RGB (Foley 1990)  ║
║  §2   Evdev Input Driver — struct input_event, US keymap, pointer tracking  ║
║  §3   AppWindow Base — Window wrapper, repaint/event monkey-patch protocol  ║
║  §4   PTY Manager — pty.openpty, non-blocking I/O, TIOCSWINSZ              ║
║  §5   ANSI Parser + SGR State — VT100/VT220 CSI decoder, Color cell attrs  ║
║  §6   Terminal Surface — PTY → cell grid → Surface pixel rasterizer         ║
║  §7   CA Wallpaper — Brian's Brain 3-state CA → Surface (Dewdney 1989)     ║
║  §8   Neural Monitor — /proc/stat + /proc/meminfo sparklines               ║
║  §9   File Browser — os.scandir tree, CP437 icons, keyboard navigation     ║
║  §10  Taskbar — window buttons, clock, CPU/mem on a compositor Layer        ║
║  §11  App Launcher — right-click floating menu on a transient Layer         ║
║  §12  Desktop Shell — wires subsystems, main loop, SIGTERM                  ║
║  §13  Self-Test                                                              ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    CORDIC   : x_{i+1}=x_i-d_i·y_i·2^{-i},  y_{i+1}=y_i+d_i·x_i·2^{-i}  ║
║    Brian's Brain: DEAD→ALIVE iff exactly 2 ALIVE Moore neighbours           ║
║                   ALIVE→DYING,  DYING→DEAD  [Dewdney, SciAm 1989]          ║
║    Sparkline: y_px = h - 1 - floor(v·(h-1))  v∈[0,1]                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
import pty
import fcntl
import termios
import struct
import select
import signal
import queue
import threading
import time
import glob
from collections import deque
from dataclasses import dataclass, field
from typing import (Callable, Dict, List, NamedTuple, Optional,
                    Sequence, Tuple)

# ── aios_display import ──────────────────────────────────────────────────────
try:
    from aios_display import (
        DisplayManager, DisplayMode,
        WindowManager, Window, Layer, Compositor,
        Surface, Color, Rect, Palette,
        EventQueue, Event, EventType, KeyCode, KeyMod, MouseButton,
        FONT_W, FONT_H, TITLE_H, FRAME, MAX_WINDOWS,
    )
except ImportError as _exc:
    sys.exit(f"FATAL: aios_display.py not found or failed to import: {_exc}")

# ════════════════════════════════════════════════════════════════════════════
#  §1 — MATH PRIMITIVES (zero stdlib math imports)
# ════════════════════════════════════════════════════════════════════════════

# §1.1 CORDIC (Volder 1959) — rotation mode, 30 iterations
# x_{i+1}=x_i - d_i·y_i·2^{-i},  y_{i+1}=y_i + d_i·x_i·2^{-i},  z→0
_K      = 0.6072529350088812   # gain ∏cos(atan(2^{-i})), i=0..29
_ATAN   = (0.7853981633974483, 0.4636476090008172, 0.24497866312686414,
           0.12435499454676144, 0.06241880999595735, 0.031239833430268277,
           0.015623728620476831, 0.007812341060101111, 0.003906230131966972,
           0.0019531225164788188, 0.0009765621895593195, 0.0004882812111948985,
           0.00024414062014936177, 0.00012207031189367021, 6.103515617420877e-05,
           3.051757811552610e-05, 1.525878906131570e-05, 7.629394531101970e-06,
           3.814697265606496e-06, 1.907348632810187e-06, 9.536743164059955e-07,
           4.768371582030890e-07, 2.384185791015843e-07, 1.192092895507807e-07,
           5.960464477539055e-08, 2.980232238769530e-08, 1.490116119384766e-08,
           7.450580596923828e-09, 3.725290298461914e-09, 1.862645149230957e-09)
_PI      = 3.141592653589793
_TWO_PI  = 6.283185307179586
_HALF_PI = 1.5707963267948966

def cordic_sincos(angle: float) -> Tuple[float, float]:
    """Return (sin, cos) via CORDIC rotation. Error < 2^{-29}."""
    a = angle % _TWO_PI
    if a < 0.0:
        a += _TWO_PI
    q  = int(a / _HALF_PI)
    ar = a - q * _HALF_PI
    x, y, z, p = _K, 0.0, ar, 1.0
    for at in _ATAN:
        d = 1.0 if z >= 0.0 else -1.0
        x, y, z = x - d*y*p, y + d*x*p, z - d*at
        p *= 0.5
    if   q == 0: return  y,  x
    elif q == 1: return  x, -y
    elif q == 2: return -y, -x
    else:        return -x,  y

def sqrt_nr(x: float) -> float:
    """Newton–Raphson √x. Error < 1 ULP after 60 iterations."""
    if x <= 0.0:
        return 0.0
    g = x if x > 1.0 else 1.0
    for _ in range(60):
        g2 = 0.5 * (g + x / g)
        if abs(g2 - g) < 1e-15 * g:
            return g2
        g = g2
    return g

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Foley et al., Computer Graphics §13.3."""
    if s == 0.0:
        vi = int(v * 255)
        return vi, vi, vi
    h6 = (h % 360.0) / 60.0
    i  = int(h6)
    f  = h6 - i
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t  = v * (1.0 - s * (1.0 - f))
    r, g, b = ((v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q))[i]
    return int(r*255), int(g*255), int(b*255)

# ════════════════════════════════════════════════════════════════════════════
#  §2 — EVDEV INPUT DRIVER
#  Reads raw Linux input events from /dev/input/event* without X11 or Wayland.
#
#  Kernel struct (64-bit Linux, little-endian):
#    struct input_event {
#        long  tv_sec;   // 8 bytes
#        long  tv_usec;  // 8 bytes
#        __u16 type;
#        __u16 code;
#        __s32 value;
#    };                  // total 24 bytes
#
#  Reference: linux/input.h, linux/input-event-codes.h
# ════════════════════════════════════════════════════════════════════════════

_EVDEV_FMT = "@llHHi"           # native alignment (long×2, ushort×2, int)
_EVDEV_SZ  = struct.calcsize(_EVDEV_FMT)   # 24 bytes on 64-bit Linux

# --- evdev event type constants ---
_EV_SYN = 0x00
_EV_KEY = 0x01
_EV_REL = 0x02
_EV_ABS = 0x03

# --- EV_REL axis codes ---
_REL_X     = 0
_REL_Y     = 1
_REL_WHEEL = 8

# --- EV_KEY: mouse button codes (BTN_MOUSE range 0x110–0x117) ---
_BTN_LEFT   = 0x110
_BTN_RIGHT  = 0x111
_BTN_MIDDLE = 0x112
_BTN_MISC   = 0x100          # boundary: codes ≥ this are device-buttons not keys

# --- Modifier key Linux codes ---
_LSHIFT = 42; _RSHIFT = 54
_LCTRL  = 29; _RCTRL  = 97
_LALT   = 56; _RALT   = 100
_CAPSLOCK = 58

# --- US QWERTY keymap: linux_keycode → (normal, shifted) ---
_KEYMAP: Dict[int, Tuple[str, str]] = {
    2:  ('1','!'), 3:  ('2','@'), 4:  ('3','#'), 5:  ('4','$'),
    6:  ('5','%'), 7:  ('6','^'), 8:  ('7','&'), 9:  ('8','*'),
    10: ('9','('), 11: ('0',')'), 12: ('-','_'), 13: ('=','+'),
    16: ('q','Q'), 17: ('w','W'), 18: ('e','E'), 19: ('r','R'),
    20: ('t','T'), 21: ('y','Y'), 22: ('u','U'), 23: ('i','I'),
    24: ('o','O'), 25: ('p','P'), 26: ('[','{'), 27: (']','}'),
    30: ('a','A'), 31: ('s','S'), 32: ('d','D'), 33: ('f','F'),
    34: ('g','G'), 35: ('h','H'), 36: ('j','J'), 37: ('k','K'),
    38: ('l','L'), 39: (';',':'), 40: ("'",'"'), 41: ('`','~'),
    43: ('\\','|'), 44: ('z','Z'), 45: ('x','X'), 46: ('c','C'),
    47: ('v','V'), 48: ('b','B'), 49: ('n','N'), 50: ('m','M'),
    51: (',','<'), 52: ('.','>'), 53: ('/','?'), 57: (' ',' '),
    # numpad
    71: ('7','7'), 72: ('8','8'), 73: ('9','9'), 74: ('-','-'),
    75: ('4','4'), 76: ('5','5'), 77: ('6','6'), 78: ('+','+'),
    79: ('1','1'), 80: ('2','2'), 81: ('3','3'), 82: ('0','0'),
    83: ('.','.')
}

# --- Linux keycode → KeyCode (non-printable navigation/function keys) ---
_KEYCODES: Dict[int, int] = {
    1:   KeyCode.ESCAPE,    14:  KeyCode.BACKSPACE, 15:  KeyCode.TAB,
    28:  KeyCode.ENTER,     96:  KeyCode.ENTER,
    103: KeyCode.UP,        105: KeyCode.LEFT,
    106: KeyCode.RIGHT,     108: KeyCode.DOWN,
    102: KeyCode.HOME,      107: KeyCode.END,
    104: KeyCode.PAGE_UP,   109: KeyCode.PAGE_DOWN,
    111: KeyCode.DELETE,    110: KeyCode.INSERT,
    59:  KeyCode.F1,  60: KeyCode.F2,  61: KeyCode.F3,  62: KeyCode.F4,
    63:  KeyCode.F5,  64: KeyCode.F6,  65: KeyCode.F7,  66: KeyCode.F8,
    67:  KeyCode.F9,  68: KeyCode.F10, 87: KeyCode.F11, 88: KeyCode.F12,
}


class EvdevInputDriver:
    """
    Opens all readable /dev/input/event* devices and translates raw Linux
    input_event structs into aios_display.py Event objects pushed onto a
    shared EventQueue.

    Runs on a daemon thread. Tracks:
      - mouse_x, mouse_y (absolute pixel position, clamped to screen bounds)
      - shift, ctrl, alt, caps_lock modifier state
    """

    _MOUSE_ACCEL = 1.5      # multiplier applied to REL_X/REL_Y delta

    def __init__(self, eq: EventQueue, screen_w: int, screen_h: int) -> None:
        self._eq       = eq
        self._sw       = screen_w
        self._sh       = screen_h
        self.mouse_x   = screen_w // 2
        self.mouse_y   = screen_h // 2
        self._shift    = False
        self._ctrl     = False
        self._alt      = False
        self._caps     = False
        self._fds: List[int] = []
        self._alive    = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open all /dev/input/event* and launch reader thread."""
        for path in sorted(glob.glob("/dev/input/event*")):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                self._fds.append(fd)
            except OSError:
                pass          # device busy or permission denied — skip
        if not self._fds:
            # Degrade gracefully: no input hardware found (e.g. CI/VM)
            return
        self._alive  = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="evdev-reader")
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = []
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        buf = bytearray(_EVDEV_SZ)
        while self._alive and self._fds:
            try:
                readable, _, _ = select.select(self._fds, [], [], 0.05)
            except (ValueError, OSError):
                break
            for fd in readable:
                try:
                    n = os.readinto(fd, buf)   # type: ignore[attr-defined]
                except AttributeError:
                    # Python < 3.12: use os.read
                    try:
                        chunk = os.read(fd, _EVDEV_SZ)
                        if not chunk:
                            continue
                        buf[:len(chunk)] = chunk
                        n = len(chunk)
                    except OSError:
                        continue
                except OSError:
                    continue
                if n < _EVDEV_SZ:
                    continue
                _, _, ev_type, code, value = struct.unpack_from(_EVDEV_FMT, buf)
                self._dispatch(ev_type, code, value)

    def _dispatch(self, ev_type: int, code: int, value: int) -> None:
        if ev_type == _EV_SYN:
            return                         # sync frames — nothing to do

        if ev_type == _EV_REL:
            if code == _REL_X:
                self.mouse_x = int(clamp(
                    self.mouse_x + value * self._MOUSE_ACCEL, 0, self._sw - 1))
            elif code == _REL_Y:
                self.mouse_y = int(clamp(
                    self.mouse_y + value * self._MOUSE_ACCEL, 0, self._sh - 1))
            elif code == _REL_WHEEL:
                ev = Event(EventType.MOUSE_WHEEL,
                           x=self.mouse_x, y=self.mouse_y,
                           wheel=1 if value > 0 else -1)
                self._eq.push(ev)
            # push a MOUSE_MOVE after accumulating X/Y
            self._eq.push(Event(EventType.MOUSE_MOVE,
                                x=self.mouse_x, y=self.mouse_y))
            return

        if ev_type != _EV_KEY:
            return

        pressed = (value == 1 or value == 2)   # 1=press, 2=repeat, 0=release

        # ── modifier tracking ────────────────────────────────────────────────
        if code in (_LSHIFT, _RSHIFT):
            self._shift = pressed; return
        if code in (_LCTRL, _RCTRL):
            self._ctrl  = pressed; return
        if code in (_LALT, _RALT):
            self._alt   = pressed; return
        if code == _CAPSLOCK and value == 1:
            self._caps = not self._caps; return

        if not pressed:
            # Mouse button release
            if code in (_BTN_LEFT, _BTN_MIDDLE, _BTN_RIGHT):
                btn = {_BTN_LEFT: MouseButton.LEFT,
                       _BTN_MIDDLE: MouseButton.MIDDLE,
                       _BTN_RIGHT: MouseButton.RIGHT}[code]
                self._eq.push(Event(EventType.MOUSE_UP,
                                    x=self.mouse_x, y=self.mouse_y,
                                    button=int(btn)))
            return

        # ── mouse buttons (press) ────────────────────────────────────────────
        if code == _BTN_LEFT:
            self._eq.push(Event(EventType.MOUSE_DOWN,
                                x=self.mouse_x, y=self.mouse_y,
                                button=int(MouseButton.LEFT)))
            return
        if code == _BTN_RIGHT:
            self._eq.push(Event(EventType.MOUSE_DOWN,
                                x=self.mouse_x, y=self.mouse_y,
                                button=int(MouseButton.RIGHT)))
            return
        if code == _BTN_MIDDLE:
            self._eq.push(Event(EventType.MOUSE_DOWN,
                                x=self.mouse_x, y=self.mouse_y,
                                button=int(MouseButton.MIDDLE)))
            return

        # ── non-printable special keys ───────────────────────────────────────
        if code in _KEYCODES:
            kc   = _KEYCODES[code]
            mods = (KeyMod.SHIFT if self._shift else KeyMod.NONE) | \
                   (KeyMod.CTRL  if self._ctrl  else KeyMod.NONE) | \
                   (KeyMod.ALT   if self._alt   else KeyMod.NONE)
            self._eq.push(Event(EventType.KEY_DOWN, key=kc, mods=mods))
            return

        # ── printable keys ───────────────────────────────────────────────────
        if code not in _KEYMAP:
            return
        normal, shifted = _KEYMAP[code]
        # CAPS_LOCK flips shift for letters only
        use_shift = self._shift ^ (self._caps and normal.isalpha())
        ch = shifted if use_shift else normal

        if self._ctrl and ch.isalpha():
            # Ctrl+letter → ASCII control code
            ctrl_byte = ord(ch.lower()) - ord('a') + 1
            mods = KeyMod.CTRL | (KeyMod.SHIFT if self._shift else KeyMod.NONE)
            self._eq.push(Event(EventType.KEY_DOWN,
                                key=ctrl_byte, char=chr(ctrl_byte), mods=mods))
        else:
            mods = (KeyMod.SHIFT if use_shift else KeyMod.NONE) | \
                   (KeyMod.ALT   if self._alt  else KeyMod.NONE)
            self._eq.push(Event(EventType.KEY_DOWN,
                                key=ord(ch), char=ch, mods=mods))

    @property
    def modifiers(self) -> KeyMod:
        return ((KeyMod.SHIFT if self._shift else KeyMod.NONE) |
                (KeyMod.CTRL  if self._ctrl  else KeyMod.NONE) |
                (KeyMod.ALT   if self._alt   else KeyMod.NONE))


# ════════════════════════════════════════════════════════════════════════════
#  §3 — APPWINDOW BASE CLASS
#  Wraps a wm.Window (from aios_display.py) and installs monkey-patch hooks
#  so that the WindowManager's render loop calls our paint() and our
#  handle_event() instead of the default Widget-tree dispatch.
# ════════════════════════════════════════════════════════════════════════════

class AppWindow:
    """
    Base class for all AIOS desktop applications.

    Subclasses override:
        paint(surface, client_rect) — draw content each dirty frame
        handle_event(ev)            — return True if event was consumed

    The Window's title bar, frame, and close button are painted by
    _repaint_wrapper(), which calls paint() for the client area.
    """

    def __init__(self, wm: WindowManager,
                 title: str, x: int, y: int, w: int, h: int,
                 on_close: Optional[Callable[["AppWindow"], None]] = None) -> None:
        self._wm       = wm
        self._on_close = on_close
        # Create the Window through the WM so it gets a compositor Layer.
        self._win: Window = wm.create_window(title, x, y, w, h)
        # Install hooks: repaint → our wrapper, on_event → handle_event.
        self._win.repaint  = self._repaint_wrapper   # type: ignore[method-assign]
        self._win.on_event = self.handle_event        # type: ignore[method-assign]
        self._win._dirty   = True
        self._closed       = False

    # ── geometry helpers ─────────────────────────────────────────────────────
    @property
    def win(self) -> Window:
        return self._win

    def client_rect(self) -> Rect:
        return self._win.client_rect()

    def invalidate(self) -> None:
        """Mark window dirty so the WM schedules a repaint next tick."""
        self._win._dirty = True

    # ── lifecycle ────────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wm.destroy_window(self._win)
        if self._on_close:
            self._on_close(self)

    def check_closed(self) -> bool:
        """Returns True if the WM no longer owns this Window (closed via title bar x)."""
        return self._win not in self._wm.windows()

    # ── painting ─────────────────────────────────────────────────────────────
    def _repaint_wrapper(self) -> None:
        """
        Draw the AIOS title bar + frame, then delegate to paint().
        Mirrors Window.repaint() from aios_display.py §9, so decoration
        looks identical across all AppWindows.
        """
        s = self._win.surface
        s.set_clip(None)
        s.clear(Palette.PANEL_BG)
        focused    = self._win.focused
        frame_col  = Palette.BORDER_ACTIVE if focused else Palette.BORDER
        s.draw_rect(Rect(0, 0, self._win.rect.w, self._win.rect.h), frame_col)
        s.fill_rect(Rect(0, 0, self._win.rect.w, TITLE_H),
                    Palette.TITLE_BG if focused else Palette.PANEL_BG)
        s.draw_hline(0, TITLE_H - 1, self._win.rect.w, frame_col)
        max_title  = max(0, (self._win.rect.w - 2 * FONT_W) // FONT_W)
        s.draw_text(self._win.title[:max_title], FONT_W // 2, 2,
                    Palette.TEXT if focused else Palette.TEXT_DIM)
        s.draw_text("x", self._win.rect.w - FONT_W - 2, 2, Palette.ERROR)
        cr = self._win.client_rect()
        with s.clip(cr):
            self.paint(s, cr)
        self._win._dirty = False

    def paint(self, surface: Surface, client_rect: Rect) -> None:
        """Override: draw app content into client_rect on surface."""

    def handle_event(self, ev: Event) -> bool:
        """
        Override: process an event.  Mouse coordinates are in absolute
        screen pixels (same as Window.on_event receives them).
        Return True to consume the event.
        """
        return False


# ════════════════════════════════════════════════════════════════════════════
#  §4 — PTY MANAGER
#  Forks a shell behind a pseudo-terminal pair.  Non-blocking reads via
#  select(); PTY output queued for the terminal widget to drain each frame.
# ════════════════════════════════════════════════════════════════════════════

_SHELL = next((s for s in ('/bin/ash', '/bin/bash', '/bin/sh')
               if os.path.exists(s)), '/bin/sh')

import subprocess as _subprocess


class PtyManager:
    """Fork a shell behind a PTY and provide non-blocking read/write."""

    def __init__(self, shell: str = _SHELL,
                 env: Optional[Dict] = None) -> None:
        self.shell     = shell
        self.env       = env or {**os.environ, 'TERM': 'xterm-256color'}
        self.master_fd: Optional[int] = None
        self.proc: Optional[_subprocess.Popen] = None
        self._q: queue.Queue = queue.Queue(maxsize=512)
        self._alive    = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        self._set_winsize(24, 80, slave_fd)

        def _child_init() -> None:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

        self.proc = _subprocess.Popen(
            [self.shell],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            close_fds=True, preexec_fn=_child_init, env=self.env)
        os.close(slave_fd)
        self._alive  = True
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name='pty-reader')
        self._thread.start()

    def _set_winsize(self, rows: int, cols: int,
                     fd: Optional[int] = None) -> None:
        target = fd if fd is not None else self.master_fd
        if target is None:
            return
        try:
            fcntl.ioctl(target, termios.TIOCSWINSZ,
                        struct.pack('HHHH', rows, cols, 0, 0))
        except OSError:
            pass

    def _reader(self) -> None:
        while self._alive and self.master_fd is not None:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.05)
                if r:
                    data = os.read(self.master_fd, 8192)
                    if not data:
                        self._alive = False
                        break

                    while self._alive:
                        try:
                            self._q.put(data, timeout=0.1)
                            break
                        except queue.Full:
                            pass
            except OSError:
                self._alive = False
                break

    def write(self, data: bytes) -> None:
        if self.master_fd is not None and self._alive:
            try:
                os.write(self.master_fd, data)
            except OSError:
                pass

    def read_pending(self) -> List[bytes]:
        chunks: List[bytes] = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return chunks

    def resize(self, rows: int, cols: int) -> None:
        self._set_winsize(rows, cols)

    def stop(self) -> None:
        self._alive = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return self._alive and (self.proc is None or self.proc.poll() is None)


# ════════════════════════════════════════════════════════════════════════════
#  §5 — ANSI PARSER + SGR STATE
#  VT100/VT220 CSI sequence decoder (DEC RM-002).
#  SGR state tracks Color objects (not hex strings) so the terminal
#  rasterizer can pass them directly to Surface.draw_glyph().
# ════════════════════════════════════════════════════════════════════════════

# ANSI 16-color palette as Color objects (matching xterm defaults)
_ANSI16: Tuple[Color, ...] = (
    Color(28,  28,  28 ), Color(204, 51,  51 ), Color(51,  204, 51 ),
    Color(204, 204, 51 ), Color(51,  51,  204), Color(204, 51,  204),
    Color(51,  204, 204), Color(204, 204, 204), Color(136, 136, 136),
    Color(255, 85,  85 ), Color(85,  255, 85 ), Color(255, 255, 85 ),
    Color(85,  85,  255), Color(255, 85,  255), Color(85,  255, 255),
    Color(255, 255, 255),
)

def _ansi256_color(n: int) -> Color:
    """Map xterm-256 index → Color. Ref: xterm source chart 256colors2.pl."""
    if n < 16:
        return _ANSI16[n]
    if n < 232:
        n -= 16
        def cv(c: int) -> int:
            return 0 if c == 0 else 55 + 40 * c
        return Color(cv(n // 36), cv((n // 6) % 6), cv(n % 6))
    v = 8 + (n - 232) * 10
    return Color(v, v, v)


@dataclass
class SgrState:
    """Current SGR (Select Graphic Rendition) colour and attribute state."""
    fg:        Optional[Color] = None   # None → use terminal default
    bg:        Optional[Color] = None
    bold:      bool = False
    italic:    bool = False
    underline: bool = False
    reverse:   bool = False

    def reset(self) -> None:
        self.fg = self.bg = None
        self.bold = self.italic = self.underline = self.reverse = False

    def apply_sgr(self, params: str) -> None:
        nums = [int(x) if x.strip().isdigit() else 0
                for x in params.split(';')]
        i = 0
        while i < len(nums):
            n = nums[i]
            if   n == 0:  self.reset()
            elif n == 1:  self.bold      = True
            elif n == 3:  self.italic    = True
            elif n == 4:  self.underline = True
            elif n == 7:  self.reverse   = True
            elif n == 22: self.bold      = False
            elif n == 23: self.italic    = False
            elif n == 24: self.underline = False
            elif n == 27: self.reverse   = False
            elif 30 <= n <= 37:
                self.fg = _ANSI16[n - 30]
            elif n == 38:
                if i + 1 < len(nums):
                    if nums[i+1] == 5 and i + 2 < len(nums):
                        self.fg = _ansi256_color(nums[i+2]); i += 2
                    elif nums[i+1] == 2 and i + 4 < len(nums):
                        self.fg = Color(nums[i+2], nums[i+3], nums[i+4]); i += 4
            elif n == 39: self.fg = None
            elif 40 <= n <= 47:
                self.bg = _ANSI16[n - 40]
            elif n == 48:
                if i + 1 < len(nums):
                    if nums[i+1] == 5 and i + 2 < len(nums):
                        self.bg = _ansi256_color(nums[i+2]); i += 2
                    elif nums[i+1] == 2 and i + 4 < len(nums):
                        self.bg = Color(nums[i+2], nums[i+3], nums[i+4]); i += 4
            elif n == 49:  self.bg = None
            elif 90 <= n <= 97:  self.fg = _ANSI16[n - 90 + 8]
            elif 100 <= n <= 107: self.bg = _ANSI16[n - 100 + 8]
            i += 1

    def effective_fg(self, default: Color) -> Color:
        return self.fg if self.fg is not None else default

    def effective_bg(self, default: Color) -> Color:
        return self.bg if self.bg is not None else default


class AnsiParser:
    """
    Incremental VT100/VT220 CSI/SGR decoder.
    Emits (action: str, params: str) tuples per parsed token.
    Handles: CSI cursor/erase/scroll/insert/delete, OSC title, SS3 F-keys,
    UTF-8 multi-byte characters.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        self._buf.extend(data)
        i = 0
        n = len(self._buf)
        while i < n:
            b = self._buf[i]
            if b == 0x1b:
                if i + 1 >= n:
                    break
                nb = self._buf[i + 1]
                if nb == ord('['):             # CSI
                    j = i + 2
                    while j < n and (chr(self._buf[j]).isdigit()
                                     or self._buf[j] in b';?!> '):
                        j += 1
                    if j >= n:
                        break
                    fin = chr(self._buf[j])
                    ps  = self._buf[i+2:j].decode('ascii', errors='replace')
                    events.extend(self._csi(fin, ps))
                    i = j + 1
                elif nb == ord('O'):           # SS3 (F1–F4)
                    if i + 2 >= n:
                        break
                    fin = chr(self._buf[i + 2])
                    km  = {'P': 'F1', 'Q': 'F2', 'R': 'F3', 'S': 'F4',
                           'A': 'CURSOR_UP',   'B': 'CURSOR_DOWN',
                           'C': 'CURSOR_FWD',  'D': 'CURSOR_BACK',
                           'H': 'CURSOR_POS',  'F': 'CURSOR_POS'}
                    if fin in km:
                        events.append((km[fin], '1'))
                    i += 3
                elif nb == ord(']'):           # OSC — skip to BEL or ST
                    j = i + 2
                    while j < n and self._buf[j] != 0x07:
                        if self._buf[j] == 0x1b and j+1 < n \
                                and self._buf[j+1] == ord('\\'):
                            j += 2; break
                        j += 1
                    if j < n and self._buf[j] == 0x07:
                        j += 1
                    i = j
                elif nb == ord('M'):
                    events.append(('REVERSE_INDEX', ''))
                    i += 2
                elif nb == ord('c'):
                    events.append(('RESET', ''))
                    i += 2
                elif nb in (ord('('), ord(')')):
                    i += 3 if i + 2 < n else i + 2
                else:
                    i += 2
            elif b == 0x0d: events.append(('CR',  '')); i += 1
            elif b == 0x0a: events.append(('LF',  '')); i += 1
            elif b == 0x08: events.append(('BS',  '')); i += 1
            elif b == 0x07: events.append(('BEL', '')); i += 1
            elif b == 0x09: events.append(('TAB', '')); i += 1
            elif b == 0x7f: events.append(('DEL', '')); i += 1
            elif b < 0x20:  i += 1                      # other controls
            else:
                # UTF-8 decode
                if b < 0x80:
                    events.append(('CHAR', chr(b))); i += 1
                elif b < 0xc0:
                    i += 1           # continuation byte without starter
                elif b < 0xe0:
                    if i + 1 < n:
                        try:    ch = self._buf[i:i+2].decode('utf-8')
                        except: ch = '?'
                        events.append(('CHAR', ch)); i += 2
                    else: break
                elif b < 0xf0:
                    if i + 2 < n:
                        try:    ch = self._buf[i:i+3].decode('utf-8')
                        except: ch = '?'
                        events.append(('CHAR', ch)); i += 3
                    else: break
                else:
                    if i + 3 < n:
                        try:    ch = self._buf[i:i+4].decode('utf-8')
                        except: ch = '?'
                        events.append(('CHAR', ch)); i += 4
                    else: break
        self._buf = self._buf[i:]
        return events

    def _csi(self, f: str, p: str) -> List[Tuple[str, str]]:
        m = {
            'm': ('SGR', p or '0'),
            'A': ('CURSOR_UP',    p or '1'), 'B': ('CURSOR_DOWN',  p or '1'),
            'C': ('CURSOR_FWD',   p or '1'), 'D': ('CURSOR_BACK',  p or '1'),
            'H': ('CURSOR_POS',   p or '1;1'), 'f': ('CURSOR_POS', p or '1;1'),
            'G': ('CURSOR_COL',   p or '1'),
            'J': ('ERASE_SCREEN', p or '0'), 'K': ('ERASE_LINE',   p or '0'),
            'S': ('SCROLL_UP',    p or '1'), 'T': ('SCROLL_DOWN',  p or '1'),
            'P': ('DELETE_CHARS', p or '1'), '@': ('INSERT_CHARS', p or '1'),
            'L': ('INSERT_LINES', p or '1'), 'M': ('DELETE_LINES', p or '1'),
            'n': ('DEVICE_STATUS', p),
        }
        if f in m:
            return [m[f]]
        if f in ('h', 'l'):
            return [('MODE_SET', p + f)]
        if f == 'r':
            return [('SET_SCROLL_RGN', p)]
        return []


# Key code → terminal escape-sequence bytes
_KEYCODE_ESC: Dict[int, bytes] = {
    KeyCode.ENTER:     b'\r',
    KeyCode.BACKSPACE: b'\x7f',
    KeyCode.TAB:       b'\t',
    KeyCode.ESCAPE:    b'\x1b',
    KeyCode.UP:        b'\x1b[A', KeyCode.DOWN:      b'\x1b[B',
    KeyCode.RIGHT:     b'\x1b[C', KeyCode.LEFT:      b'\x1b[D',
    KeyCode.HOME:      b'\x1b[H', KeyCode.END:       b'\x1b[F',
    KeyCode.PAGE_UP:   b'\x1b[5~', KeyCode.PAGE_DOWN: b'\x1b[6~',
    KeyCode.DELETE:    b'\x1b[3~', KeyCode.INSERT:    b'\x1b[2~',
    KeyCode.F1:  b'\x1bOP',  KeyCode.F2:  b'\x1bOQ',
    KeyCode.F3:  b'\x1bOR',  KeyCode.F4:  b'\x1bOS',
    KeyCode.F5:  b'\x1b[15~', KeyCode.F6:  b'\x1b[17~',
    KeyCode.F7:  b'\x1b[18~', KeyCode.F8:  b'\x1b[19~',
    KeyCode.F9:  b'\x1b[20~', KeyCode.F10: b'\x1b[21~',
    KeyCode.F11: b'\x1b[23~', KeyCode.F12: b'\x1b[24~',
}


# ════════════════════════════════════════════════════════════════════════════
#  §6 — TERMINAL SURFACE
#  A PTY-backed AppWindow that renders terminal output as a cell grid onto
#  an aios_display.py Surface.  Implements the subset of VT100/VT220 used by
#  busybox ash / bash / vim / htop.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class _Cell:
    """One terminal character cell."""
    ch:        int   = 32    # Unicode codepoint, masked to 0xFF for CP437
    fg:        Optional[Color] = None
    bg:        Optional[Color] = None
    bold:      bool  = False
    underline: bool  = False
    reverse:   bool  = False
    dirty:     bool  = True


class TerminalSurface(AppWindow):
    """
    Terminal emulator that rasterizes PTY output into a Surface cell grid.

    Cell model:
        cols = client_w // FONT_W,   rows = client_h // FONT_H
        grid[y * cols + x] = _Cell
    All VT100/VT220 cursor movement, erase, and scroll operations operate on
    this grid.  paint() iterates dirty cells and calls Surface.draw_glyph().
    """

    _DEFAULT_FG = Palette.TEXT
    _DEFAULT_BG = Palette.DARK_BG

    def __init__(self, wm: WindowManager, title: str,
                 x: int, y: int, w: int, h: int,
                 shell: str = _SHELL,
                 on_close: Optional[Callable[["AppWindow"], None]] = None) -> None:
        super().__init__(wm, title, x, y, w, h, on_close=on_close)
        cr           = self.client_rect()
        self._cols   = max(1, cr.w // FONT_W)
        self._rows   = max(1, cr.h // FONT_H)
        self._grid: List[_Cell] = [_Cell() for _ in range(self._cols * self._rows)]
        self._cx     = 0       # cursor column
        self._cy     = 0       # cursor row
        self._top    = 0       # scroll region top
        self._bot    = self._rows - 1    # scroll region bottom
        self._cursor_visible = True
        self._sgr    = SgrState()
        self._parser = AnsiParser()
        self._pty    = PtyManager(shell=shell)
        self._lock   = threading.Lock()
        try:
            self._pty.start()
            self._pty.resize(self._rows, self._cols)
        except Exception as e:
            self._write_error(f"[PTY error: {e}]")
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name='term-poll')
        self._poll_thread.start()

    # ── PTY → cell grid ─────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: drain PTY queue every 12 ms."""
        while not self._closed:
            chunks = self._pty.read_pending()
            if chunks:
                data = b''.join(chunks)
                with self._lock:
                    self._dispatch(self._parser.feed(data))
                self.invalidate()
            time.sleep(0.012)

    def _write_error(self, msg: str) -> None:
        for ch in msg:
            self._put_char(ch)

    def _put_char(self, ch: str) -> None:
        """Write one character at the cursor position, advance cursor."""
        cp  = ord(ch) & 0xFF     # mask to CP437 range
        sgr = self._sgr
        fg  = sgr.fg
        bg  = sgr.bg
        rv  = sgr.reverse
        bld = sgr.bold
        ul  = sgr.underline
        if 0 <= self._cx < self._cols and 0 <= self._cy < self._rows:
            idx = self._cy * self._cols + self._cx
            c   = self._grid[idx]
            c.ch = cp; c.fg = fg; c.bg = bg
            c.bold = bld; c.underline = ul; c.reverse = rv
            c.dirty = True
        self._cx += 1
        if self._cx >= self._cols:
            self._cx = 0
            self._lf()

    def _lf(self) -> None:
        """Line-feed with scroll region awareness."""
        if self._cy < self._bot:
            self._cy += 1
        else:
            self._scroll_up(1)

    def _scroll_up(self, n: int) -> None:
        """Scroll the scroll region up by n lines, fill at bottom with blanks."""
        n   = max(1, n)
        top = self._top
        bot = self._bot
        cols = self._cols
        for _ in range(n):
            for row in range(top, bot):
                self._grid[row*cols:(row+1)*cols] = \
                    self._grid[(row+1)*cols:(row+2)*cols]
                for c in self._grid[row*cols:(row+1)*cols]:
                    c.dirty = True
            for col in range(cols):
                self._grid[bot*cols + col] = _Cell(dirty=True)

    def _scroll_down(self, n: int) -> None:
        """Scroll the scroll region down by n lines, fill at top with blanks."""
        n   = max(1, n)
        top = self._top
        bot = self._bot
        cols = self._cols
        for _ in range(n):
            for row in range(bot, top, -1):
                self._grid[row*cols:(row+1)*cols] = \
                    self._grid[(row-1)*cols:row*cols]
                for c in self._grid[row*cols:(row+1)*cols]:
                    c.dirty = True
            for col in range(cols):
                self._grid[top*cols + col] = _Cell(dirty=True)

    def _erase_cells(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Erase cells in rectangle [x0,x1) × [y0,y1)."""
        cols = self._cols
        for y in range(max(0,y0), min(self._rows, y1)):
            for x in range(max(0,x0), min(cols, x1)):
                self._grid[y*cols + x] = _Cell(dirty=True)

    def _dispatch(self, events: List[Tuple[str, str]]) -> None:
        """Apply parsed ANSI events to the cell grid (call under self._lock)."""
        for action, params in events:
            if action == 'CHAR':
                self._put_char(params)
            elif action == 'SGR':
                self._sgr.apply_sgr(params)
            elif action == 'CR':
                self._cx = 0
            elif action == 'LF':
                self._lf()
            elif action == 'BS':
                self._cx = max(0, self._cx - 1)
            elif action == 'DEL':
                # Delete char at cursor: shift left
                row = self._cy
                for c in range(self._cx, self._cols - 1):
                    self._grid[row*self._cols + c] = self._grid[row*self._cols + c + 1]
                    self._grid[row*self._cols + c].dirty = True
                self._grid[row*self._cols + self._cols - 1] = _Cell(dirty=True)
            elif action == 'TAB':
                self._cx = min(self._cols - 1,
                               self._cx + (8 - (self._cx % 8)))
            elif action == 'BEL':
                pass   # no audio on bare metal
            elif action == 'CURSOR_UP':
                n = max(1, int(params) if params.isdigit() else 1)
                self._cy = max(self._top, self._cy - n)
            elif action == 'CURSOR_DOWN':
                n = max(1, int(params) if params.isdigit() else 1)
                self._cy = min(self._bot, self._cy + n)
            elif action == 'CURSOR_FWD':
                n = max(1, int(params) if params.isdigit() else 1)
                self._cx = min(self._cols - 1, self._cx + n)
            elif action == 'CURSOR_BACK':
                n = max(1, int(params) if params.isdigit() else 1)
                self._cx = max(0, self._cx - n)
            elif action == 'CURSOR_POS':
                ps  = params.split(';')
                row = max(1, int(ps[0]) if ps[0].isdigit() else 1) - 1
                col = max(0, (int(ps[1]) if len(ps) > 1 and ps[1].isdigit()
                              else 1) - 1)
                self._cy = min(self._rows - 1, row)
                self._cx = min(self._cols - 1, col)
            elif action == 'CURSOR_COL':
                n = max(1, int(params) if params.isdigit() else 1)
                self._cx = min(self._cols - 1, n - 1)
            elif action == 'ERASE_LINE':
                n = int(params) if params.isdigit() else 0
                if   n == 0: self._erase_cells(self._cx, self._cy, self._cols, self._cy+1)
                elif n == 1: self._erase_cells(0, self._cy, self._cx+1, self._cy+1)
                elif n == 2: self._erase_cells(0, self._cy, self._cols,  self._cy+1)
            elif action == 'ERASE_SCREEN':
                n = int(params) if params.isdigit() else 0
                if   n == 0: self._erase_cells(self._cx, self._cy, self._cols, self._rows)
                elif n == 1: self._erase_cells(0, 0, self._cx+1, self._cy+1)
                elif n == 2:
                    self._erase_cells(0, 0, self._cols, self._rows)
                    self._cx = self._cy = 0
            elif action == 'SCROLL_UP':
                n = max(1, int(params) if params.isdigit() else 1)
                self._scroll_up(n)
            elif action == 'SCROLL_DOWN':
                n = max(1, int(params) if params.isdigit() else 1)
                self._scroll_down(n)
            elif action == 'DELETE_CHARS':
                n = max(1, int(params) if params.isdigit() else 1)
                row = self._cy; cols = self._cols
                for c in range(self._cx, cols - n):
                    self._grid[row*cols + c] = self._grid[row*cols + c + n]
                    self._grid[row*cols + c].dirty = True
                for c in range(max(self._cx, cols - n), cols):
                    self._grid[row*cols + c] = _Cell(dirty=True)
            elif action == 'INSERT_CHARS':
                n = max(1, int(params) if params.isdigit() else 1)
                row = self._cy; cols = self._cols
                for c in range(cols - 1, self._cx + n - 1, -1):
                    self._grid[row*cols + c] = self._grid[row*cols + c - n]
                    self._grid[row*cols + c].dirty = True
                for c in range(self._cx, min(self._cx + n, cols)):
                    self._grid[row*cols + c] = _Cell(dirty=True)
            elif action == 'INSERT_LINES':
                n = max(1, int(params) if params.isdigit() else 1)
                self._scroll_down(n)
            elif action == 'DELETE_LINES':
                n = max(1, int(params) if params.isdigit() else 1)
                self._scroll_up(n)
            elif action == 'SET_SCROLL_RGN':
                ps = params.split(';')
                t  = max(0, int(ps[0]) - 1) if ps[0].isdigit() else 0
                b  = (int(ps[1]) - 1) if len(ps) > 1 and ps[1].isdigit() \
                     else self._rows - 1
                self._top = max(0, min(t, self._rows - 1))
                self._bot = max(self._top, min(b, self._rows - 1))
                self._cx = self._cy = 0
            elif action == 'MODE_SET':
                p    = params
                on   = p.endswith('h')
                code = p.lstrip('?').rstrip('hl')
                if code == '25':
                    self._cursor_visible = on
                # ?1049: alternate screen — we keep a single screen buffer
            elif action == 'REVERSE_INDEX':
                if self._cy > self._top:
                    self._cy -= 1
                else:
                    self._scroll_down(1)
            elif action == 'RESET':
                self._sgr.reset()
                self._erase_cells(0, 0, self._cols, self._rows)
                self._cx = self._cy = 0
                self._top = 0
                self._bot = self._rows - 1

    # ── paint ────────────────────────────────────────────────────────────────

    def paint(self, surf: Surface, cr: Rect) -> None:
        with self._lock:
            surf.fill_rect(cr, self._DEFAULT_BG)
            for y in range(self._rows):
                for x in range(self._cols):
                    cell = self._grid[y * self._cols + x]
                    fg = cell.fg if cell.fg is not None else self._DEFAULT_FG
                    bg = cell.bg if cell.bg is not None else self._DEFAULT_BG
                    if cell.reverse:
                        fg, bg = bg, fg
                    px = cr.x + x * FONT_W
                    py = cr.y + y * FONT_H
                    surf.draw_glyph(cell.ch & 0xFF, px, py, fg, bg)
                    cell.dirty = False
            # Cursor
            if (self._cursor_visible
                    and 0 <= self._cy < self._rows
                    and 0 <= self._cx < self._cols):
                px = cr.x + self._cx * FONT_W
                py = cr.y + self._cy * FONT_H
                surf.fill_rect(Rect(px, py, FONT_W, FONT_H), Palette.ACCENT)
                cell = self._grid[self._cy * self._cols + self._cx]
                if cell.ch != 32:
                    inv_fg = (cell.bg or self._DEFAULT_BG)
                    surf.draw_glyph(cell.ch & 0xFF, px, py, inv_fg)

    # ── events ───────────────────────────────────────────────────────────────

    def handle_event(self, ev: Event) -> bool:
        if ev.type != EventType.KEY_DOWN:
            return False
        b = self._ev_to_bytes(ev)
        if b:
            self._pty.write(b)
            return True
        return False

    @staticmethod
    def _ev_to_bytes(ev: Event) -> Optional[bytes]:
        if ev.mods & KeyMod.CTRL:
            if 1 <= ev.key <= 26:    # already a control character
                return bytes([ev.key])
            if ev.char and ev.char.isalpha():
                return bytes([ord(ev.char.lower()) - ord('a') + 1])
        if ev.char and ord(ev.char) >= 32:
            return ev.char.encode('utf-8', errors='replace')
        kc = ev.key
        if kc in _KEYCODE_ESC:
            return _KEYCODE_ESC[kc]
        return None

    def close(self) -> None:
        self._pty.stop()
        super().close()


# ════════════════════════════════════════════════════════════════════════════
#  §7 — CA WALLPAPER
#  Brian's Brain 3-state cellular automaton on a compositor Layer.
#
#  States: DEAD=0, ALIVE=1, DYING=2
#  Rules  (Moore neighbourhood, all 8 neighbours):
#    DEAD  → ALIVE  iff exactly 2 ALIVE neighbours
#    ALIVE → DYING
#    DYING → DEAD
#  [Dewdney, Scientific American 261(4):102-105, 1989]
# ════════════════════════════════════════════════════════════════════════════

_DEAD  = 0
_ALIVE = 1
_DYING = 2

_CA_CELL_PX = 6      # pixels per CA cell (6×6 → ~320×180 cells on 1920×1080)
_CA_FPS     = 8      # CA update rate

_CA_ALIVE_COLOR = Color(61,  180, 100, 180)   # semi-transparent accent green
_CA_DYING_COLOR = Color(30,  90,  50,  90)    # dim, fading
_CA_BG_COLOR    = Palette.DARK_BG


class CAWallpaper:
    """
    Brian's Brain CA rendered into a compositor Layer behind all windows.

    The Surface is written directly (bypassing put_pixel per-pixel locking)
    for performance: entire rows are written via bytearray slice assignment.
    """

    def __init__(self, compositor: Compositor,
                 screen_w: int, screen_h: int) -> None:
        self._sw     = screen_w
        self._sh     = screen_h
        self._cp     = _CA_CELL_PX
        self._cols   = screen_w  // _CA_CELL_PX
        self._rows   = screen_h  // _CA_CELL_PX
        total        = self._cols * self._rows
        self._grid   = bytearray(total)
        self._next   = bytearray(total)
        self._surf   = Surface(screen_w, screen_h, fill=_CA_BG_COLOR)
        self._layer  = Layer(self._surf, 0, 0, z=-1000, name='ca-wallpaper')
        compositor.add_layer(self._layer)
        self._comp   = compositor
        self._alive  = False
        self._thread: Optional[threading.Thread] = None
        self._seed()

    def _seed(self) -> None:
        """Randomly populate ~20% of cells as ALIVE."""
        import random
        for i in range(len(self._grid)):
            self._grid[i] = _ALIVE if random.random() < 0.20 else _DEAD

    def start(self) -> None:
        self._alive  = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='ca-wallpaper')
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = 1.0 / _CA_FPS
        while self._alive:
            t0 = time.monotonic()
            self._step()
            self._render()
            self._comp.mark_dirty(Rect(0, 0, self._sw, self._sh))
            elapsed = time.monotonic() - t0
            wait    = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    def _step(self) -> None:
        """Apply Brian's Brain rules to produce self._next from self._grid."""
        cols  = self._cols
        rows  = self._rows
        g     = self._grid
        nxt   = self._next
        for y in range(rows):
            y_up   = (y - 1) % rows
            y_dn   = (y + 1) % rows
            row_u  = y_up  * cols
            row_c  = y     * cols
            row_d  = y_dn  * cols
            for x in range(cols):
                state = g[row_c + x]
                if state == _ALIVE:
                    nxt[row_c + x] = _DYING
                elif state == _DYING:
                    nxt[row_c + x] = _DEAD
                else:  # DEAD: count ALIVE neighbours
                    xl = (x - 1) % cols
                    xr = (x + 1) % cols
                    alive_n = (
                        (g[row_u + xl] == _ALIVE) +
                        (g[row_u +  x] == _ALIVE) +
                        (g[row_u + xr] == _ALIVE) +
                        (g[row_c + xl] == _ALIVE) +
                        (g[row_c + xr] == _ALIVE) +
                        (g[row_d + xl] == _ALIVE) +
                        (g[row_d +  x] == _ALIVE) +
                        (g[row_d + xr] == _ALIVE)
                    )
                    nxt[row_c + x] = _ALIVE if alive_n == 2 else _DEAD
        self._grid, self._next = self._next, self._grid

    def _render(self) -> None:
        """Rasterize the CA grid into the surface buffer (single lock hold)."""
        cp   = self._cp
        sw   = self._sw
        buf  = self._surf._buf
        bg_b = bytes((_CA_BG_COLOR.b, _CA_BG_COLOR.g, _CA_BG_COLOR.r, 255))
        al_b = bytes((_CA_ALIVE_COLOR.b, _CA_ALIVE_COLOR.g,
                      _CA_ALIVE_COLOR.r, _CA_ALIVE_COLOR.a))
        dy_b = bytes((_CA_DYING_COLOR.b, _CA_DYING_COLOR.g,
                      _CA_DYING_COLOR.r, _CA_DYING_COLOR.a))
        bg_row = bg_b * sw
        with self._surf._lock:
            # Clear to background
            for py in range(self._sh):
                off = py * sw * 4
                buf[off:off + sw * 4] = bg_row
            # Draw cells
            for cy in range(self._rows):
                for cx in range(self._cols):
                    state = self._grid[cy * self._cols + cx]
                    if state == _DEAD:
                        continue
                    c_bytes = al_b if state == _ALIVE else dy_b
                    row_px  = c_bytes * cp
                    px      = cx * cp
                    for dy in range(cp):
                        py  = cy * cp + dy
                        off = (py * sw + px) * 4
                        buf[off:off + cp * 4] = row_px


# ════════════════════════════════════════════════════════════════════════════
#  §8 — NEURAL MONITOR
#  Reads /proc/stat (CPU) and /proc/meminfo (memory) and renders sparklines
#  on an AppWindow Surface.  Updates sample history at 1 Hz on a daemon thread.
# ════════════════════════════════════════════════════════════════════════════

def _read_cpu_pct() -> float:
    """Read CPU% (0.0–100.0) from /proc/stat using two samples."""
    try:
        with open('/proc/stat') as f:
            t0 = f.readline().split()
        time.sleep(0.1)
        with open('/proc/stat') as f:
            t1 = f.readline().split()
        vals0 = [int(x) for x in t0[1:8]]
        vals1 = [int(x) for x in t1[1:8]]
        idle0 = vals0[3]; idle1 = vals1[3]
        tot0  = sum(vals0); tot1 = sum(vals1)
        dt    = tot1 - tot0
        if dt == 0:
            return 0.0
        return 100.0 * (1.0 - (idle1 - idle0) / dt)
    except Exception:
        return 0.0


def _read_mem_pct() -> Tuple[float, float, float]:
    """Return (used%, total_mb, used_mb) from /proc/meminfo."""
    try:
        info: Dict[str, int] = {}
        with open('/proc/meminfo') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
        total = info.get('MemTotal', 0)
        avail = info.get('MemAvailable', info.get('MemFree', 0))
        if total == 0:
            return 0.0, 0.0, 0.0
        used     = total - avail
        used_pct = 100.0 * used / total
        return used_pct, total / 1024, used / 1024
    except Exception:
        return 0.0, 0.0, 0.0


_SPARK_LEN = 60      # number of history points kept

_COL_CPU  = Color(80, 160, 255)    # accent blue
_COL_MEM  = Color(80, 200, 110)    # success green
_COL_WARN = Color(245, 158, 11)    # amber
_COL_CRIT = Color(225, 85,  85)    # error red
_COL_GRID = Color(50,  50,  70, 255)


class NeuralMonitor(AppWindow):
    """CPU and memory sparkline monitor."""

    def __init__(self, wm: WindowManager,
                 x: int, y: int, w: int, h: int,
                 on_close: Optional[Callable[["AppWindow"], None]] = None) -> None:
        super().__init__(wm, 'Neural Monitor', x, y, w, h, on_close=on_close)
        self._cpu_hist: deque = deque([0.0] * _SPARK_LEN, maxlen=_SPARK_LEN)
        self._mem_hist: deque = deque([0.0] * _SPARK_LEN, maxlen=_SPARK_LEN)
        self._cpu_pct  = 0.0
        self._mem_pct  = 0.0
        self._mem_mb   = 0.0
        self._lock_d   = threading.Lock()
        self._sample_thread = threading.Thread(
            target=self._sampler, daemon=True, name='neural-monitor')
        self._sample_thread.start()

    def _sampler(self) -> None:
        while not self._closed:
            cpu = _read_cpu_pct()
            mem, total_mb, used_mb = _read_mem_pct()
            with self._lock_d:
                self._cpu_pct = cpu
                self._mem_pct = mem
                self._mem_mb  = used_mb
                self._cpu_hist.append(cpu)
                self._mem_hist.append(mem)
            self.invalidate()
            time.sleep(1.0)

    def _draw_sparkline(self, surf: Surface, history: Sequence[float],
                        rx: int, ry: int, rw: int, rh: int,
                        color: Color, label: str, pct: float) -> None:
        # Background + border
        surf.fill_rect(Rect(rx, ry, rw, rh), Palette.PANEL_BG)
        surf.draw_rect(Rect(rx, ry, rw, rh), _COL_GRID)
        # Horizontal grid lines at 25%, 50%, 75%
        for frac in (0.25, 0.50, 0.75):
            gy = ry + rh - 1 - int(frac * (rh - 2))
            surf.draw_hline(rx + 1, gy, rw - 2, _COL_GRID)
        # Determine color by value
        col = (_COL_CRIT if pct > 85 else
               _COL_WARN if pct > 60 else color)
        # Plot sparkline — Bresenham lines between successive sample points
        vals = list(history)
        n    = len(vals)
        if n < 2:
            return
        plot_w = rw - 2
        plot_h = rh - 2
        def sample_px(i: int) -> Tuple[int, int]:
            sx = rx + 1 + int(i * plot_w / (n - 1))
            sy = ry + plot_h - int(clamp(vals[i], 0, 100) / 100.0 * plot_h)
            return sx, sy
        prev = sample_px(0)
        for i in range(1, n):
            cur = sample_px(i)
            surf.draw_line(prev[0], prev[1], cur[0], cur[1], col)
            prev = cur
        # Label text
        surf.draw_text(label, rx + 2, ry + 2, Palette.TEXT_DIM)
        surf.draw_text(f"{pct:5.1f}%", rx + rw - 6 * FONT_W - 2, ry + 2,
                       col)

    def paint(self, surf: Surface, cr: Rect) -> None:
        surf.fill_rect(cr, Palette.DARK_BG)
        with self._lock_d:
            cpu_h   = list(self._cpu_hist)
            mem_h   = list(self._mem_hist)
            cpu_pct = self._cpu_pct
            mem_pct = self._mem_pct
            mem_mb  = self._mem_mb
        pad  = 4
        half = (cr.h - 3 * pad) // 2
        self._draw_sparkline(surf, cpu_h,
                             cr.x + pad, cr.y + pad,
                             cr.w - 2*pad, half,
                             _COL_CPU, 'CPU', cpu_pct)
        self._draw_sparkline(surf, mem_h,
                             cr.x + pad, cr.y + 2*pad + half,
                             cr.w - 2*pad, half,
                             _COL_MEM, f'MEM {mem_mb:.0f}MB', mem_pct)


# ════════════════════════════════════════════════════════════════════════════
#  §9 — FILE BROWSER
#  os.scandir tree view.  CP437 glyphs used as type icons (no external fonts).
#  Keyboard navigation: UP/DOWN arrows, ENTER to descend, BACKSPACE to ascend.
# ════════════════════════════════════════════════════════════════════════════

_ICON_DIR  = 0xFE   # ■ — directory
_ICON_EXEC = 0x10   # ► — executable
_ICON_FILE = 0xFA   # · — regular file
_ICON_LINK = 0xC4   # ─ — symlink

_COL_DIR   = Palette.ACCENT
_COL_EXEC  = Color(80, 200, 110)
_COL_LINK  = Color(180, 140, 255)
_COL_FILE  = Palette.TEXT
_COL_SEL   = Color(44, 62, 100)     # selected row background


@dataclass
class _Entry:
    name:  str
    is_dir: bool
    is_exe: bool
    is_lnk: bool
    size:  int


class FileBrowser(AppWindow):
    """Directory tree viewer backed by os.scandir."""

    def __init__(self, wm: WindowManager,
                 start: str = '/', x: int = 0, y: int = 0,
                 w: int = 400, h: int = 480,
                 on_close: Optional[Callable[["AppWindow"], None]] = None) -> None:
        super().__init__(wm, f'Files: {start}', x, y, w, h, on_close=on_close)
        self._path    = os.path.abspath(start)
        self._entries: List[_Entry] = []
        self._sel     = 0
        self._scroll  = 0
        self._lock_e  = threading.Lock()
        self._load()

    def _load(self) -> None:
        entries: List[_Entry] = []
        try:
            for e in sorted(os.scandir(self._path),
                            key=lambda x: (not x.is_dir(follow_symlinks=False),
                                           x.name.lower())):
                stat = e.stat(follow_symlinks=False)
                entries.append(_Entry(
                    name   = e.name,
                    is_dir = e.is_dir(follow_symlinks=False),
                    is_exe = bool(stat.st_mode & 0o111),
                    is_lnk = e.is_symlink(),
                    size   = stat.st_size,
                ))
        except PermissionError:
            pass
        with self._lock_e:
            self._entries = entries
            self._sel     = 0
            self._scroll  = 0
        self._win.title = f'Files: {self._path}'
        self.invalidate()

    def paint(self, surf: Surface, cr: Rect) -> None:
        surf.fill_rect(cr, Palette.DARK_BG)
        # Path bar
        surf.fill_rect(Rect(cr.x, cr.y, cr.w, FONT_H + 2), Palette.PANEL_BG)
        path_text = self._path[-((cr.w // FONT_W) - 1):]
        surf.draw_text(path_text, cr.x + 2, cr.y + 1, Palette.ACCENT)
        # Entries
        row_h    = FONT_H + 2
        y_start  = cr.y + FONT_H + 4
        visible  = max(1, (cr.h - FONT_H - 4) // row_h)
        with self._lock_e:
            entries = self._entries[self._scroll:self._scroll + visible]
            sel_abs = self._sel
        for i, entry in enumerate(entries):
            abs_i = self._scroll + i
            ey    = y_start + i * row_h
            if abs_i == sel_abs:
                surf.fill_rect(Rect(cr.x, ey, cr.w, row_h), _COL_SEL)
            if entry.is_lnk:
                icon, col = _ICON_LINK, _COL_LINK
            elif entry.is_dir:
                icon, col = _ICON_DIR, _COL_DIR
            elif entry.is_exe:
                icon, col = _ICON_EXEC, _COL_EXEC
            else:
                icon, col = _ICON_FILE, _COL_FILE
            surf.draw_glyph(icon, cr.x + 2, ey, col, None)
            name_max = max(0, (cr.w // FONT_W) - 3)
            surf.draw_text(entry.name[:name_max], cr.x + FONT_W + 4, ey, col)

    def handle_event(self, ev: Event) -> bool:
        if ev.type != EventType.KEY_DOWN:
            return False
        with self._lock_e:
            n    = len(self._entries)
            sel  = self._sel
        cr       = self.client_rect()
        row_h    = FONT_H + 2
        visible  = max(1, (cr.h - FONT_H - 4) // row_h)
        kc       = ev.key
        if kc == KeyCode.UP:
            self._sel = max(0, sel - 1)
            if self._sel < self._scroll:
                self._scroll = self._sel
            self.invalidate(); return True
        if kc == KeyCode.DOWN:
            self._sel = min(n - 1, sel + 1)
            if self._sel >= self._scroll + visible:
                self._scroll = self._sel - visible + 1
            self.invalidate(); return True
        if kc == KeyCode.ENTER:
            with self._lock_e:
                if 0 <= sel < len(self._entries):
                    entry = self._entries[sel]
            if entry.is_dir:
                self._path = os.path.join(self._path, entry.name)
                self._load()
            return True
        if kc == KeyCode.BACKSPACE:
            parent = os.path.dirname(self._path)
            if parent != self._path:
                self._path = parent
                self._load()
            return True
        return False


# ════════════════════════════════════════════════════════════════════════════
#  §10 — TASKBAR
#  A fixed-height compositor Layer pinned to the bottom of the screen.
#  Draws window buttons, system clock, CPU%, and MEM%.
#  Click events are intercepted by DesktopShell before being forwarded to
#  the WM, so taskbar buttons work even without a window under the cursor.
# ════════════════════════════════════════════════════════════════════════════

TASKBAR_H = 22          # pixels

_TB_BG    = Color(12,  16,  30,  255)
_TB_BTN   = Color(30,  40,  68,  255)
_TB_BTN_A = Color(44,  62, 100,  255)   # active / hovered
_TB_SEP   = Color(40,  50,  80,  255)


class Taskbar:
    """Renders the taskbar onto a dedicated compositor Layer."""

    LAUNCH_W = 48    # width of the AIOS launcher button (pixels)

    def __init__(self, compositor: Compositor, wm: WindowManager,
                 screen_w: int, screen_h: int) -> None:
        self._wm   = wm
        self._sw   = screen_w
        self._sh   = screen_h
        self._surf = Surface(screen_w, TASKBAR_H, fill=_TB_BG)
        self._layer = Layer(self._surf,
                            x=0, y=screen_h - TASKBAR_H,
                            z=999999, name='taskbar')
        compositor.add_layer(self._layer)
        self._comp = compositor
        self._alive  = False
        self._thread: Optional[threading.Thread] = None
        self._launch_cb: Optional[Callable] = None
        self._cpu    = 0.0
        self._mem    = 0.0

    def set_launch_callback(self, cb: Callable) -> None:
        self._launch_cb = cb

    def start(self) -> None:
        self._alive  = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='taskbar')
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while self._alive:
            self._cpu = _read_cpu_pct()
            mp, _, _  = _read_mem_pct()
            self._mem = mp
            self._draw()
            self._comp.mark_dirty(Rect(0, self._sh - TASKBAR_H,
                                       self._sw, TASKBAR_H))
            time.sleep(1.0)

    def _draw(self) -> None:
        s   = self._surf
        sw  = self._sw
        H   = TASKBAR_H
        with s._lock:
            # Background
            bg_row = bytes((_TB_BG.b, _TB_BG.g, _TB_BG.r, 255)) * sw
            for y in range(H):
                off = y * sw * 4
                s._buf[off:off + sw * 4] = bg_row
        # Top separator line
        s.draw_hline(0, 0, sw, _TB_SEP)
        # Launcher button
        lw  = self.LAUNCH_W
        s.fill_rect(Rect(1, 1, lw - 2, H - 2), Palette.ACCENT)
        s.draw_text('AIOS', 4, (H - FONT_H) // 2, Color(0, 0, 0))
        # Window buttons
        bx  = lw + 4
        bw  = 96
        for win in self._wm.windows():
            if bx + bw >= sw - 160:
                break
            focused = win.focused
            bg      = _TB_BTN_A if focused else _TB_BTN
            s.fill_rect(Rect(bx, 2, bw, H - 4), bg)
            s.draw_rect(Rect(bx, 2, bw, H - 4), _TB_SEP)
            title = win.title[:(bw // FONT_W) - 1]
            s.draw_text(title, bx + 3, (H - FONT_H) // 2,
                        Palette.TEXT if focused else Palette.TEXT_DIM)
            bx += bw + 3
        # Clock + CPU/MEM
        ts   = time.strftime('%H:%M:%S')
        cpu_col = (_COL_CRIT if self._cpu > 85 else
                   _COL_WARN if self._cpu > 60 else _COL_CPU)
        mem_col = (_COL_CRIT if self._mem > 85 else
                   _COL_WARN if self._mem > 70 else _COL_MEM)
        rx = sw - 2
        for label, col in (
                (ts, Palette.TEXT),
                (f'MEM{self._mem:4.0f}%', mem_col),
                (f'CPU{self._cpu:4.0f}%', cpu_col)):
            tw = len(label) * FONT_W
            rx -= tw + 6
            s.draw_text(label, rx, (H - FONT_H) // 2, col)

    def hit_test_launch(self, mx: int) -> bool:
        return 1 <= mx <= self.LAUNCH_W

    def hit_test_window(self, mx: int) -> Optional[Window]:
        """Return the Window whose taskbar button was clicked, or None."""
        bx = self.LAUNCH_W + 4
        bw = 96
        for win in self._wm.windows():
            if bx + bw >= self._sw - 160:
                break
            if bx <= mx < bx + bw:
                return win
            bx += bw + 3
        return None

    def in_taskbar(self, my: int) -> bool:
        return my >= self._sh - TASKBAR_H


# ════════════════════════════════════════════════════════════════════════════
#  §11 — APP LAUNCHER
#  Right-click context menu rendered as a transient compositor Layer.
#  Appears at mouse position; disappears on click-outside or selection.
# ════════════════════════════════════════════════════════════════════════════

_MENU_ITEM_H = FONT_H + 6
_MENU_W      = 160
_MENU_BG     = Color(18, 22, 40, 245)
_MENU_SEP_C  = Color(44, 54, 88, 255)
_MENU_HOV    = Color(40, 80, 160, 220)

@dataclass
class _MenuItem:
    label:     str
    action:    str           # internal action identifier or '' for separator
    separator: bool = False


_MENU_ITEMS: Tuple[_MenuItem, ...] = (
    _MenuItem('▶  New Terminal',   'terminal'),
    _MenuItem('▶  AIOS REPL',      'repl'),
    _MenuItem('',                  '',  separator=True),
    _MenuItem('▶  File Browser',   'files'),
    _MenuItem('▶  Neural Monitor', 'monitor'),
    _MenuItem('',                  '',  separator=True),
    _MenuItem('⏻  Shutdown',        'shutdown'),
    _MenuItem('↺  Reboot',          'reboot'),
)


class AppLauncher:
    """
    Floating context menu.  open(mx, my) creates a compositor Layer; the
    DesktopShell calls handle_click(mx, my) each frame until the menu closes.
    """

    def __init__(self, compositor: Compositor,
                 screen_w: int, screen_h: int) -> None:
        self._comp   = compositor
        self._sw     = screen_w
        self._sh     = screen_h
        self._layer: Optional[Layer] = None
        self._mx     = 0
        self._my     = 0
        self._action_cb: Optional[Callable[[str], None]] = None

    def set_action_callback(self, cb: Callable[[str], None]) -> None:
        self._action_cb = cb

    def is_open(self) -> bool:
        return self._layer is not None

    def open(self, mx: int, my: int) -> None:
        """Show the menu anchored at (mx, my)."""
        if self._layer is not None:
            self.close()
        h = len(_MENU_ITEMS) * _MENU_ITEM_H + 4
        w = _MENU_W
        # Clamp to screen
        ox = min(mx, self._sw - w - 2)
        oy = min(my - h, self._sh - TASKBAR_H - h - 2)
        oy = max(0, oy)
        surf = Surface(w, h, fill=_MENU_BG)
        surf.draw_rect(Rect(0, 0, w, h), _MENU_SEP_C)
        for i, item in enumerate(_MENU_ITEMS):
            iy = 2 + i * _MENU_ITEM_H
            if item.separator:
                surf.draw_hline(4, iy + _MENU_ITEM_H // 2, w - 8, _MENU_SEP_C)
            else:
                surf.draw_text(item.label, 6, iy + 3, Palette.TEXT)
        self._layer = Layer(surf, ox, oy, z=900000, name='launcher')
        self._comp.add_layer(self._layer)
        self._comp.mark_dirty(Rect(ox, oy, w, h))

    def close(self) -> None:
        if self._layer is not None:
            self._comp.remove_layer(self._layer)
            self._comp.mark_dirty(Rect(self._layer.x, self._layer.y,
                                       self._layer.surface.width,
                                       self._layer.surface.height))
            self._layer = None

    def handle_click(self, mx: int, my: int) -> bool:
        """
        Called when a mouse-down event fires with menu open.
        Returns True if the click was inside the menu (consumed).
        """
        if self._layer is None:
            return False
        lx = self._layer.x
        ly = self._layer.y
        lw = self._layer.surface.width
        lh = self._layer.surface.height
        if not (lx <= mx < lx + lw and ly <= my < ly + lh):
            self.close()
            return False      # click outside → close, do not consume
        rel_y = my - ly - 2
        idx   = rel_y // _MENU_ITEM_H
        if 0 <= idx < len(_MENU_ITEMS):
            item = _MENU_ITEMS[idx]
            if not item.separator and item.action and self._action_cb:
                self._action_cb(item.action)
        self.close()
        return True


# ════════════════════════════════════════════════════════════════════════════
#  §12 — DESKTOP SHELL
#  Top-level orchestrator.  Creates and wires every subsystem, then runs the
#  frame loop:  evdev → dispatch → CA/taskbar update → WM render → present.
# ════════════════════════════════════════════════════════════════════════════

_CURSOR_BITMAP = bytes([    # 8×8 arrow cursor, pointing top-left
    0b11111110,
    0b11111100,
    0b11111000,
    0b11110000,
    0b11111100,
    0b11001100,
    0b10000110,
    0b00000011,
])
_CURSOR_FG = Palette.TEXT
_CURSOR_BG = Color(0, 0, 0, 0)    # transparent background


class DesktopShell:
    """
    Boots the AIOS desktop without X11, Wayland, or Tkinter.
    Requires /dev/fb0 (or FRAMEBUFFER mode) and /dev/input/event* (evdev).

    Target resolution: whatever /dev/fb0 reports (or the fallback 1024×768).
    """

    TARGET_FPS = 30

    def __init__(self, fb_device: str = '/dev/fb0') -> None:
        self._running  = False
        self._apps: List[AppWindow] = []
        self._app_lock = threading.Lock()

        # ── 1. Display Manager ───────────────────────────────────────────────
        self.dm = DisplayManager(mode=DisplayMode.FRAMEBUFFER,
                                 fb_device=fb_device,
                                 target_fps=self.TARGET_FPS)
        self.sw = self.dm.width
        self.sh = self.dm.height

        # ── 2. Evdev input ───────────────────────────────────────────────────
        self._evdev = EvdevInputDriver(self.dm.events, self.sw, self.sh)

        # ── 3. CA Wallpaper ──────────────────────────────────────────────────
        self._ca = CAWallpaper(self.dm.wm.compositor, self.sw, self.sh)

        # ── 4. Cursor overlay ────────────────────────────────────────────────
        cursor_surf = self._make_cursor_surface()
        self._cursor_layer = Layer(cursor_surf,
                                   x=self.sw // 2, y=self.sh // 2,
                                   z=1000000, name='cursor')
        self.dm.wm.compositor.add_layer(self._cursor_layer)

        # ── 5. Taskbar ───────────────────────────────────────────────────────
        self._taskbar = Taskbar(self.dm.wm.compositor, self.dm.wm,
                                self.sw, self.sh)

        # ── 6. App Launcher ──────────────────────────────────────────────────
        self._launcher = AppLauncher(self.dm.wm.compositor, self.sw, self.sh)
        self._launcher.set_action_callback(self._on_launch_action)
        self._taskbar.set_launch_callback(lambda: self._open_launcher(4, self.sh))

    # ── cursor ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_cursor_surface() -> Surface:
        surf = Surface(8, 8, fill=_CURSOR_BG)
        for row in range(8):
            bits = _CURSOR_BITMAP[row]
            for col in range(8):
                if bits & (1 << (7 - col)):
                    surf.put_pixel(col, row, _CURSOR_FG)
        return surf

    def _update_cursor(self) -> None:
        mx = self._evdev.mouse_x
        my = self._evdev.mouse_y
        if self._cursor_layer.x != mx or self._cursor_layer.y != my:
            old = Rect(self._cursor_layer.x, self._cursor_layer.y, 8, 8)
            self._cursor_layer.x = mx
            self._cursor_layer.y = my
            self.dm.wm.compositor.mark_dirty(old)
            self.dm.wm.compositor.mark_dirty(Rect(mx, my, 8, 8))

    # ── app management ───────────────────────────────────────────────────────

    def _spawn(self, app: AppWindow) -> None:
        with self._app_lock:
            self._apps.append(app)

    def _reap_closed(self) -> None:
        """Remove apps whose WM windows have been destroyed (title-bar close)."""
        with self._app_lock:
            live    = [a for a in self._apps if not a.check_closed()]
            removed = [a for a in self._apps if a.check_closed()]
        for a in removed:
            a._closed = True          # prevent double-cleanup
        with self._app_lock:
            self._apps = live

    def _next_window_pos(self) -> Tuple[int, int, int, int]:
        """Cascade new windows across the usable area."""
        idx   = len(self._apps)
        step  = 24
        pad   = 20
        usable_h = self.sh - TASKBAR_H
        x = pad + (idx * step) % (self.sw // 3)
        y = pad + (idx * step) % (usable_h // 3)
        w = self.sw - x - pad
        h = usable_h - y - pad
        return x, y, w, h

    # ── launcher actions ─────────────────────────────────────────────────────

    def _open_launcher(self, mx: int, my: int) -> None:
        if self._launcher.is_open():
            self._launcher.close()
        else:
            self._launcher.open(mx, my)

    def _on_launch_action(self, action: str) -> None:
        if action == 'terminal':
            self.open_terminal()
        elif action == 'repl':
            self.open_repl()
        elif action == 'files':
            self.open_files()
        elif action == 'monitor':
            self.open_monitor()
        elif action == 'shutdown':
            self._shutdown()
        elif action == 'reboot':
            self._reboot()

    def open_terminal(self, shell: str = _SHELL) -> TerminalSurface:
        x, y, w, h = self._next_window_pos()
        t = TerminalSurface(self.dm.wm, 'AIOS Terminal',
                            x, y, max(w, 400), max(h, 300),
                            shell=shell,
                            on_close=self._on_app_close)
        self._spawn(t)
        return t

    def open_repl(self) -> TerminalSurface:
        env = {**os.environ, 'TERM': 'xterm-256color', 'PYTHONSTARTUP': ''}
        startup = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'aios_core.py')
        if os.path.exists(startup):
            env['PYTHONSTARTUP'] = startup
        x, y, w, h = self._next_window_pos()
        t = TerminalSurface(self.dm.wm, 'AIOS REPL',
                            x, y, max(w, 400), max(h, 300),
                            shell=f'{sys.executable} -i',
                            on_close=self._on_app_close)
        t._pty.env = env
        self._spawn(t)
        return t

    def open_files(self, start: str = '/') -> FileBrowser:
        x, y, _, _ = self._next_window_pos()
        fb = FileBrowser(self.dm.wm, start,
                         x, y, 420, min(500, self.sh - TASKBAR_H - y - 20),
                         on_close=self._on_app_close)
        self._spawn(fb)
        return fb

    def open_monitor(self) -> NeuralMonitor:
        x, y, _, _ = self._next_window_pos()
        nm = NeuralMonitor(self.dm.wm, x, y, 280, 220,
                           on_close=self._on_app_close)
        self._spawn(nm)
        return nm

    def _on_app_close(self, app: AppWindow) -> None:
        with self._app_lock:
            try:
                self._apps.remove(app)
            except ValueError:
                pass

    def _shutdown(self) -> None:
        self._running = False
        try:
            _subprocess.Popen(['poweroff'])
        except Exception:
            pass

    def _reboot(self) -> None:
        self._running = False
        try:
            _subprocess.Popen(['reboot'])
        except Exception:
            pass

    # ── input pre-routing ────────────────────────────────────────────────────

    def _preprocess_event(self, ev: Event) -> bool:
        """
        Intercept events that belong to desktop chrome (taskbar, launcher).
        Returns True if the event was consumed (do NOT forward to WM).
        """
        if ev.type == EventType.MOUSE_DOWN:
            # Launcher menu gets first shot
            if self._launcher.is_open():
                if self._launcher.handle_click(ev.x, ev.y):
                    return True

            # Taskbar region
            if self._taskbar.in_taskbar(ev.y):
                if ev.button == int(MouseButton.LEFT):
                    if self._taskbar.hit_test_launch(ev.x):
                        self._open_launcher(ev.x, ev.y)
                        return True
                    win = self._taskbar.hit_test_window(ev.x)
                    if win is not None:
                        self.dm.wm.focus_window(win)
                        return True
                return True     # absorb all clicks in taskbar area

            # Right-click anywhere else → launcher
            if ev.button == int(MouseButton.RIGHT):
                self._open_launcher(ev.x, ev.y)
                return True

        # F2 → new terminal
        if ev.type == EventType.KEY_DOWN and ev.key == KeyCode.F2:
            self.open_terminal()
            return False    # don't consume so WM can also process (harmless)

        return False

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._sighandler)
        signal.signal(signal.SIGINT,  self._sighandler)

        self._running = True

        # Start subsystems
        self._evdev.start()
        self._ca.start()
        self._taskbar.start()

        # Open a welcome terminal
        self.open_terminal()

        frame_budget = 1.0 / self.TARGET_FPS

        while self._running:
            t0 = time.monotonic()

            # Drain shared event queue, routing desktop chrome events first
            pending = self.dm.events.drain()
            for ev in pending:
                if ev.type == EventType.QUIT:
                    self._running = False
                    break
                if not self._preprocess_event(ev):
                    self.dm.wm.dispatch(ev)

            # Close menu if user clicked elsewhere (already handled above)

            # Update cursor position to latest mouse position
            self._update_cursor()

            # Reap windows closed via title-bar button
            self._reap_closed()

            # Render all dirty windows + present
            surface, spans = self.dm.wm.render_frame()
            if spans:
                self.dm.driver.present(surface, spans)

            self.dm._frame_count += 1

            elapsed = time.monotonic() - t0
            wait    = frame_budget - elapsed
            if wait > 0:
                time.sleep(wait)

        # Teardown
        self._evdev.stop()
        self._ca.stop()
        self._taskbar.stop()
        for app in list(self._apps):
            app.close()

    def _sighandler(self, signum: int, frame: object) -> None:
        self._running = False


# ════════════════════════════════════════════════════════════════════════════
#  §13 — SELF-TEST
#  Headless smoke tests (no framebuffer required).  Run:
#    python3 aios_desktop.py --selftest
# ════════════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    failures: List[str] = []

    def check(cond: bool, label: str) -> None:
        if cond:
            print(f"  \x1b[32mPASS\x1b[0m  {label}")
        else:
            print(f"  \x1b[31mFAIL\x1b[0m  {label}")
            failures.append(label)

    print("AIOS Desktop Environment v2 — self-test")
    print("-" * 60)

    # §1 Math
    print("§1 math")
    s, c = cordic_sincos(0.0)
    check(abs(s) < 1e-6 and abs(c - 1.0) < 1e-6, "cordic sin(0)=0, cos(0)=1")
    s, c = cordic_sincos(_HALF_PI)
    check(abs(s - 1.0) < 1e-5, "cordic sin(π/2)=1")
    check(abs(sqrt_nr(9.0) - 3.0) < 1e-12, "sqrt_nr(9)=3")
    r, g, b = hsv_to_rgb(120, 1.0, 1.0)
    check(r == 0 and g == 255 and b == 0, "hsv_to_rgb(120,1,1) = pure green")

    # §2 Evdev keymap
    print("§2 evdev keymap")
    check(16 in _KEYMAP and _KEYMAP[16][0] == 'q', "keymap[16] = ('q','Q')")
    check(_KEYCODES[1] == KeyCode.ESCAPE, "keycode 1 → ESCAPE")
    check(_KEYCODES[103] == KeyCode.UP, "keycode 103 → UP")

    # §3 AppWindow — create in ANSI mode (no real fb needed)
    print("§3 AppWindow")
    dm = DisplayManager(mode=DisplayMode.ANSI_TERMINAL, width=120, height=40)
    wm = dm.wm
    paint_called = [False]
    class _TestApp(AppWindow):
        def paint(self, surf, cr):
            paint_called[0] = True
    app = _TestApp(wm, 'test', 4, 4, 80, 40)
    check(app.win in wm.windows(), "AppWindow creates WM window")
    wm.render_frame()
    check(paint_called[0], "AppWindow.paint() called by WM render")

    # §5 AnsiParser
    print("§5 AnsiParser")
    p = AnsiParser()
    evs = p.feed(b'hello\x1b[1;32mworld\x1b[0m')
    types = [e[0] for e in evs]
    check('CHAR' in types and 'SGR' in types, "AnsiParser emits CHAR + SGR")

    # §5 SgrState
    print("§5 SgrState")
    sgr = SgrState()
    sgr.apply_sgr('1;32')
    check(sgr.bold and sgr.fg == _ANSI16[2], "SGR bold + green fg")
    sgr.apply_sgr('0')
    check(not sgr.bold and sgr.fg is None, "SGR reset clears state")
    check(_ansi256_color(196) == Color(255, 0, 0), "ansi256(196)=red")

    # §7 Brian's Brain
    print("§7 Brian's Brain")
    comp_dummy = DisplayManager(mode=DisplayMode.ANSI_TERMINAL,
                                 width=64, height=32).wm.compositor
    ca = CAWallpaper(comp_dummy, 64, 32)
    # Seed a known 3-ALIVE cluster: with exactly 2 alive neighbours, DEAD→ALIVE
    ca._grid[:] = bytearray(len(ca._grid))     # all dead
    ca._grid[ca._cols + 1] = _ALIVE            # (1,1)
    ca._grid[ca._cols + 2] = _ALIVE            # (2,1)
    ca._grid[2 * ca._cols + 1] = _ALIVE        # (1,2)
    ca._step()
    # Cell (2,2) has exactly 2 alive neighbours at (1,1) and (2,1) — but also (1,2)
    # that's 3 alive neighbours, so cell (2,2) stays dead. Cell (0,2) has 2 alive
    # neighbours (1,1) and (1,2). Verify original alive cells became DYING.
    check(ca._grid[ca._cols + 1] == _DYING, "ALIVE→DYING after step")
    check(ca._grid[ca._cols + 2] == _DYING, "ALIVE→DYING (2)")

    # §8 proc readers (graceful on non-Linux)
    print("§8 proc readers")
    mp, _, _ = _read_mem_pct()
    check(0.0 <= mp <= 100.0, f"/proc/meminfo→mem_pct in [0,100]: {mp:.1f}")

    # §10 Taskbar geometry
    print("§10 Taskbar")
    check(TASKBAR_H == 22, "TASKBAR_H == 22")

    print("-" * 60)
    if failures:
        print(f"\x1b[31m{len(failures)} FAILURE(S)\x1b[0m: " + "; ".join(failures))
    else:
        print("\x1b[32mALL CHECKS PASSED\x1b[0m")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if '--selftest' in sys.argv:
        sys.exit(_selftest())
    fb = '/dev/fb0'
    for arg in sys.argv[1:]:
        if arg.startswith('--fb='):
            fb = arg[5:]
    shell = DesktopShell(fb_device=fb)
    shell.run()


if __name__ == '__main__':
    main()
