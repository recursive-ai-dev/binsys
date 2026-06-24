#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Display Manager                                                      ║
║  aios_display.py                                                             ║
║                                                                              ║
║  "Light is information. Every pixel is a decision. Every frame, a proof."   ║
║                                                                              ║
║  Rendering Pipeline:                                                         ║
║    InputDriver → EventQueue → WindowManager → Compositor → DisplayDriver    ║
║                                                                              ║
║  Components:                                                                 ║
║    §0  Constants & Enums   — DisplayMode, PixelFormat, EventType, KeyCode   ║
║    §1  Math Primitives     — clamp, lerp, isqrt, Porter-Duff, Bresenham     ║
║    §2  Color System        — Color, VGA16 palette, RGB→ANSI256 mapping      ║
║    §3  Font Engine         — CP437-class 8×8 bitmap font (embedded), glyphs ║
║    §4  Surface             — ARGB32 pixel buffer, draw ops, blit, clip      ║
║    §5  Compositor          — Z-order layers, DirtyRect AABB, PD blending    ║
║    §6  Widget System       — Label, Panel, ProgressBar, TextInput, Border   ║
║    §7  Event System        — EventQueue, InputDriver, VT200 mouse, raw tty  ║
║    §8  Display Drivers      — ANSITerminal (half-block), VGAAdapter, FB      ║
║    §9  Window Manager       — Window lifecycle, focus, z-order, decoration   ║
║    §10 Display Manager       — Top-level orchestrator, @agent_method hooks   ║
║    §11 Self-Test            — Surface, compositor, window, render pipeline   ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Bresenham    : err±=2·d, step by sign  [Bresenham 1965]                  ║
║    Porter-Duff  : C_o = αs·Cs + (1−αs)·αd·Cd  [Porter & Duff 1984]        ║
║    Bilinear     : f = Σ f(i,j)·max(0,1−|x−i|)·max(0,1−|y−j|)             ║
║    AABB union   : R' = (min x, min y, max x2, max y2)                        ║
║    Half-block   : pair rows 2r/2r+1 → fg=top, bg=bottom, glyph ▀ U+2580     ║
║    isqrt        : Newton x←(x+n/x)/2  [Heron of Alexandria, c.60 AD]        ║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Every computation traceable to a named equation above.                  ║
║    • Thread-safe: shared state guarded by threading.RLock.                   ║
║    • Zero external dependencies. Pure Python 3.9+ stdlib only.              ║
║    • Standalone: degrades gracefully without aios_core on path.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
import time
import struct
import select
import threading
import functools
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional raw-tty deps (POSIX). Absent on non-POSIX → input degrades to no-op.
try:
    import termios
    import tty
    import fcntl
    _HAS_TTY = True
except ImportError:
    _HAS_TTY = False

# ── AIOS kernel integration (optional) ───────────────────────────────────────
try:
    from aios_core import agent_method, AgentPriority, AgentRegistry  # type: ignore
    _AIOS_INTEGRATED: bool = True
except ImportError:
    _AIOS_INTEGRATED = False
    AgentRegistry = None  # type: ignore

    class AgentPriority(IntEnum):  # type: ignore[no-redef]
        CRITICAL = 0
        HIGH     = 1
        NORMAL   = 2
        LOW      = 3

    def agent_method(  # type: ignore[no-redef]
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict] = None,
        returns: str = "Any",
        priority: Any = None,
        owner: str = "display",
    ) -> Callable:
        """Passthrough shim when running outside the AIOS kernel."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("_ctx", None)
                return fn(*args, **kwargs)
            return wrapper
        return decorator


# ════════════════════════════════════════════════════════════════════════════
#  §0 — CONSTANTS & ENUMS
#  Color indices follow the EGA 4-bit VGA palette (IBM Tech. Ref., 1984) and
#  the xterm-256color table (8+8 named + 216-entry 6×6×6 cube + 24 grays).
# ════════════════════════════════════════════════════════════════════════════

DISPLAY_VERSION = (1, 0, 0)
FONT_W          = 8          # CP437-class glyph width  (pixels)
FONT_H          = 8          # CP437-class glyph height (pixels)
MAX_WINDOWS     = 64
DIRTY_FLUSH_CAP = 192        # beyond this many dirty rects, do a full flush

# ANSI / VT100 control strings
_ESC   = "\x1b"
_CSI   = "\x1b["
_RESET = "\x1b[0m"
HALF_BLOCK = "\u2580"        # ▀  upper-half block (half-block renderer)
FULL_BLOCK = "\u2588"        # █  full block


class DisplayMode(IntEnum):
    """Output backend selected at driver construction."""
    ANSI_TERMINAL = 0        # default: ANSI/VT100 truecolor to a tty
    VGA_TEXT      = 1        # aios_core.VGATextDriver text buffer at 0xB8000
    FRAMEBUFFER   = 2        # mmap-backed linear framebuffer (e.g. /dev/fb0)


class PixelFormat(IntEnum):
    """In-memory pixel layout for Surface buffers (bytes-per-pixel in name)."""
    ARGB32 = 0               # A,R,G,B  little-endian word — AIOS native
    RGBA32 = 1               # R,G,B,A  (OpenGL byte order)
    BGRA32 = 2               # B,G,R,A  (common Linux fb xRGB)
    RGB565 = 3               # 16-bit packed 5-6-5


class EventType(IntEnum):
    KEY_DOWN    = 0
    MOUSE_MOVE  = 1
    MOUSE_DOWN  = 2
    MOUSE_UP    = 3
    MOUSE_WHEEL = 4
    RESIZE      = 5
    FOCUS_IN    = 6
    FOCUS_OUT   = 7
    PAINT       = 8
    QUIT        = 9


class KeyCode(IntEnum):
    """Codes for non-printable keys (printables use their ord())."""
    UNKNOWN   = 0
    TAB       = 9
    ENTER     = 13
    ESCAPE    = 27
    BACKSPACE = 127
    UP        = 256
    DOWN      = 257
    LEFT      = 258
    RIGHT     = 259
    HOME      = 260
    END       = 261
    PAGE_UP   = 262
    PAGE_DOWN = 263
    INSERT    = 264
    DELETE    = 265
    F1 = 271; F2 = 272; F3 = 273; F4 = 274; F5 = 275; F6 = 276
    F7 = 277; F8 = 278; F9 = 279; F10 = 280; F11 = 281; F12 = 282


class KeyMod(IntFlag):
    NONE  = 0
    SHIFT = 1
    ALT   = 2
    CTRL  = 4


class MouseButton(IntEnum):
    LEFT       = 0
    MIDDLE     = 1
    RIGHT      = 2
    WHEEL_UP   = 64
    WHEEL_DOWN = 65


class WindowState(IntEnum):
    NORMAL    = 0
    MINIMIZED = 1
    MAXIMIZED = 2
    HIDDEN    = 3


# ════════════════════════════════════════════════════════════════════════════
#  §1 — MATH PRIMITIVES (FIRST PRINCIPLES)
#  No import of math/numpy. Every routine cited to a named result.
# ════════════════════════════════════════════════════════════════════════════

def _clamp(x: int, lo: int, hi: int) -> int:
    """Clamp integer x to [lo, hi] inclusive."""
    return lo if x < lo else (hi if x > hi else x)


def _clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _lerp(a: float, b: float, t: float) -> float:
    """Eq. LERP: f(a,b,t) = a + t·(b−a),  t ∈ [0,1]."""
    return a + t * (b - a)


def _ilerp(a: int, b: int, t256: int) -> int:
    """Integer lerp, t scaled to [0,256]. Eq. ILERP: a + (t·(b−a))>>8."""
    return a + ((t256 * (b - a)) >> 8)


def _isqrt(n: int) -> int:
    """
    Integer square root via Newton/Heron iteration.
    Eq. ISQRT-NR:  x_{k+1} = (x_k + n // x_k) // 2  [Heron, c.60 AD]
    Monotone-decreasing; terminates when the estimate stops shrinking.
    """
    if n < 0:
        raise ValueError("isqrt requires n >= 0")
    if n == 0:
        return 0
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x


def _alpha_blend(
    sr: int, sg: int, sb: int, sa: int,
    dr: int, dg: int, db: int, da: int,
) -> Tuple[int, int, int, int]:
    """
    Porter-Duff 'source-over-destination' [Porter & Duff 1984].
    Eq. PD-OVER (un-premultiplied, fixed-point 1/255 ≈ 1/256):
        α_o = αs + (1−αs)·αd
        C_o = (αs·Cs + (1−αs)·αd·Cd) / α_o     (returned straight-alpha)
    For the common opaque-destination case (αd = 255) this reduces to a
    plain lerp by αs, which is the inner loop the compositor hits most.
    """
    if sa >= 255:
        return sr, sg, sb, 255
    if sa <= 0:
        return dr, dg, db, da
    inv = 255 - sa
    if da >= 255:
        # Fast path: opaque destination → straight lerp, result opaque.
        out_r = (sa * sr + inv * dr + 127) // 255
        out_g = (sa * sg + inv * dg + 127) // 255
        out_b = (sa * sb + inv * db + 127) // 255
        return _clamp(out_r, 0, 255), _clamp(out_g, 0, 255), _clamp(out_b, 0, 255), 255
    # General case with translucent destination.
    out_a = sa + (inv * da + 127) // 255
    if out_a == 0:
        return 0, 0, 0, 0
    num_r = sa * sr + (inv * da * dr + 127) // 255
    num_g = sa * sg + (inv * da * dg + 127) // 255
    num_b = sa * sb + (inv * da * db + 127) // 255
    out_r = num_r // out_a
    out_g = num_g // out_a
    out_b = num_b // out_a
    return (_clamp(out_r, 0, 255), _clamp(out_g, 0, 255),
            _clamp(out_b, 0, 255), _clamp(out_a, 0, 255))


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham integer line rasteriser [Bresenham 1965].
    Eq. BRESENHAM:  err = dx − dy;  e2 = 2·err
        if e2 > −dy: err −= dy; x += sx
        if e2 <  dx: err += dx; y += sy
    Returns the ordered list of integer pixel coords from (x0,y0) to (x1,y1).
    No floating point, no division.
    """
    pts: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = err << 1
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


@dataclass
class Rect:
    """Axis-aligned bounding box. Used for clipping and dirty-rect tracking."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def intersects(self, o: "Rect") -> bool:
        return (self.x < o.x2 and self.x2 > o.x and
                self.y < o.y2 and self.y2 > o.y)

    def clip_to(self, b: "Rect") -> "Rect":
        """Intersection of self with b. Eq. AABB-CLIP."""
        nx = max(self.x, b.x)
        ny = max(self.y, b.y)
        nx2 = min(self.x2, b.x2)
        ny2 = min(self.y2, b.y2)
        return Rect(nx, ny, max(0, nx2 - nx), max(0, ny2 - ny))

    @staticmethod
    def union(a: "Rect", b: "Rect") -> "Rect":
        """Smallest AABB enclosing both. Eq. AABB-UNION."""
        if a.is_empty():
            return Rect(b.x, b.y, b.w, b.h)
        if b.is_empty():
            return Rect(a.x, a.y, a.w, a.h)
        nx = min(a.x, b.x)
        ny = min(a.y, b.y)
        nx2 = max(a.x2, b.x2)
        ny2 = max(a.y2, b.y2)
        return Rect(nx, ny, nx2 - nx, ny2 - ny)


# ════════════════════════════════════════════════════════════════════════════
#  §2 — COLOR SYSTEM
#  VGA 16-color EGA palette + xterm-256 mapping by min-distance in RGB³.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Color:
    """Straight-alpha 32-bit ARGB color. Channels in [0,255]."""
    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255

    def to_argb32(self) -> int:
        return (((self.a & 0xFF) << 24) | ((self.r & 0xFF) << 16) |
                ((self.g & 0xFF) << 8) | (self.b & 0xFF))

    @staticmethod
    def from_argb32(v: int) -> "Color":
        return Color((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF, (v >> 24) & 0xFF)

    @staticmethod
    def from_hex(h: str) -> "Color":
        """Parse '#RRGGBB' or '#RRGGBBAA'."""
        h = h.lstrip("#")
        if len(h) == 6:
            return Color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        if len(h) == 8:
            return Color(int(h[0:2], 16), int(h[2:4], 16),
                         int(h[4:6], 16), int(h[6:8], 16))
        raise ValueError(f"bad hex color {h!r}")

    def with_alpha(self, a: int) -> "Color":
        return Color(self.r, self.g, self.b, _clamp(a, 0, 255))

    def blend_over(self, dst: "Color") -> "Color":
        """Porter-Duff: self OVER dst."""
        r, g, b, a = _alpha_blend(self.r, self.g, self.b, self.a,
                                  dst.r, dst.g, dst.b, dst.a)
        return Color(r, g, b, a)

    def lerp(self, other: "Color", t: float) -> "Color":
        """Channel-wise linear interpolation, t∈[0,1]."""
        return Color(
            int(_lerp(self.r, other.r, t)),
            int(_lerp(self.g, other.g, t)),
            int(_lerp(self.b, other.b, t)),
            int(_lerp(self.a, other.a, t)),
        )

    def __repr__(self) -> str:
        return f"Color(#{self.r:02X}{self.g:02X}{self.b:02X}@{self.a:02X})"


# VGA 16-color palette (EGA standard — IBM Technical Reference, 1984)
VGA16: List[Color] = [
    Color(0, 0, 0),        Color(0, 0, 170),      Color(0, 170, 0),
    Color(0, 170, 170),    Color(170, 0, 0),      Color(170, 0, 170),
    Color(170, 85, 0),     Color(170, 170, 170),  Color(85, 85, 85),
    Color(85, 85, 255),    Color(85, 255, 85),    Color(85, 255, 255),
    Color(255, 85, 85),    Color(255, 85, 255),   Color(255, 255, 85),
    Color(255, 255, 255),
]


class Palette:
    """Named AIOS UI colors. Single source of truth for the default theme."""
    BLACK         = Color(0, 0, 0)
    WHITE         = Color(255, 255, 255)
    DARK_BG       = Color(18, 18, 28)
    PANEL_BG      = Color(30, 30, 46)
    TITLE_BG      = Color(44, 44, 68)
    BORDER        = Color(80, 80, 120)
    BORDER_ACTIVE = Color(100, 150, 230)
    TEXT          = Color(220, 220, 232)
    TEXT_DIM      = Color(120, 120, 144)
    ACCENT        = Color(80, 160, 255)
    SUCCESS       = Color(80, 200, 110)
    WARNING       = Color(225, 185, 70)
    ERROR         = Color(225, 85, 85)
    SELECTION     = Color(50, 80, 130)
    CURSOR        = Color(120, 200, 255)
    TRANSPARENT   = Color(0, 0, 0, 0)


# 6×6×6 cube channel levels used by xterm-256
_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)


def _nearest_cube_level(c: int) -> Tuple[int, int]:
    """Return (cube_index 0..5, snapped_value) nearest to channel c."""
    best_i, best_v, best_d = 0, 0, 1 << 30
    for i, lv in enumerate(_CUBE_LEVELS):
        d = (c - lv) * (c - lv)
        if d < best_d:
            best_d, best_i, best_v = d, i, lv
    return best_i, best_v


def rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """
    Map (r,g,b) → nearest xterm-256 index by squared Euclidean distance.
    Candidates: the 6×6×6 color cube (16..231) and the 24-step gray ramp
    (232..255, values 8,18,…,238). The closer of the two is returned.
    """
    # Color-cube candidate
    ri, rv = _nearest_cube_level(r)
    gi, gv = _nearest_cube_level(g)
    bi, bv = _nearest_cube_level(b)
    cube_idx = 16 + 36 * ri + 6 * gi + bi
    cube_d = (r - rv) ** 2 + (g - gv) ** 2 + (b - bv) ** 2
    # Grayscale candidate: gray k → value 8 + 10·k, k∈0..23, index 232+k
    avg = (r + g + b) // 3
    gk = _clamp((avg - 8 + 5) // 10, 0, 23)
    gv2 = 8 + 10 * gk
    gray_idx = 232 + gk
    gray_d = (r - gv2) ** 2 + (g - gv2) ** 2 + (b - gv2) ** 2
    return cube_idx if cube_d <= gray_d else gray_idx


# ════════════════════════════════════════════════════════════════════════════
#  §3 — FONT ENGINE  (CP437-class 8×8 bitmap)
#
#  Glyphs are authored as 8 rows of 8 columns of ASCII art ('#' = lit pixel),
#  which is auditable by eye and compiled to a packed MSB-first byte table at
#  import. Missing codepoints resolve to the standard .notdef box glyph — the
#  same fallback every TrueType font ships as glyph 0 (this is by design, not
#  a placeholder). The table covers printable ASCII (0x20–0x7E), the CP437
#  single/double box-drawing range, and block/shade graphics used for borders
#  and the half-block renderer.
#
#  In ANSI_TERMINAL mode the host terminal renders text directly and this
#  table is unused for glyphs; it is the rasteriser for FRAMEBUFFER mode.
# ════════════════════════════════════════════════════════════════════════════

def _art(rows: List[str]) -> bytes:
    """Compile 8 rows of 8-col ASCII art ('#'=on) into 8 MSB-first bytes."""
    if len(rows) != FONT_H:
        raise ValueError(f"glyph must have {FONT_H} rows, got {len(rows)}")
    out = bytearray(FONT_H)
    for ri, row in enumerate(rows):
        if len(row) != FONT_W:
            raise ValueError(f"glyph row {ri} must be {FONT_W} cols: {row!r}")
        bits = 0
        for ci, ch in enumerate(row):
            if ch == "#":
                bits |= 1 << (7 - ci)
        out[ri] = bits
    return bytes(out)


# .notdef — bordered box, glyph 0 of every well-formed font.
_NOTDEF = _art([
    "########",
    "#......#",
    "#.####.#",
    "#.#..#.#",
    "#.####.#",
    "#......#",
    "########",
    "........",
])

# Printable-ASCII glyph art, codepoint → 8 rows.
_ASCII_ART: Dict[int, List[str]] = {
    0x20: ["........", "........", "........", "........",
           "........", "........", "........", "........"],  # space
    0x21: ["...##...", "...##...", "...##...", "...##...",
           "...##...", "........", "...##...", "........"],  # !
    0x22: [".##.##..", ".##.##..", ".##.##..", "........",
           "........", "........", "........", "........"],  # "
    0x23: [".##.##..", ".##.##..", "#######.", ".##.##..",
           "#######.", ".##.##..", ".##.##..", "........"],  # #
    0x24: ["...##...", ".#####..", "##......", ".####...",
           "......##", "#####...", "...##...", "........"],  # $
    0x25: ["##....#.", "##...##.", "....##..", "...##...",
           "..##....", ".##...##", "#....##.", "........"],  # %
    0x26: [".###....", "##.##...", "##.##...", ".###....",
           "##.##.##", "##..##..", ".###.##.", "........"],  # &
    0x27: ["...##...", "...##...", "..##....", "........",
           "........", "........", "........", "........"],  # '
    0x28: ["....##..", "...##...", "..##....", "..##....",
           "..##....", "...##...", "....##..", "........"],  # (
    0x29: ["..##....", "...##...", "....##..", "....##..",
           "....##..", "...##...", "..##....", "........"],  # )
    0x2A: ["........", "..#.#.#.", "...###..", "#######.",
           "...###..", "..#.#.#.", "........", "........"],  # *
    0x2B: ["........", "...##...", "...##...", "#######.",
           "...##...", "...##...", "........", "........"],  # +
    0x2C: ["........", "........", "........", "........",
           "........", "...##...", "...##...", "..##...."],  # ,
    0x2D: ["........", "........", "........", "#######.",
           "........", "........", "........", "........"],  # -
    0x2E: ["........", "........", "........", "........",
           "........", "...##...", "...##...", "........"],  # .
    0x2F: [".......#", "......##", ".....##.", "....##..",
           "...##...", "..##....", ".##.....", "........"],  # /
    0x30: [".####...", "##..##..", "##.###..", "##.#.##.",
           "###.##..", "##..##..", ".####...", "........"],  # 0
    0x31: ["...##...", "..###...", ".####...", "...##...",
           "...##...", "...##...", ".######.", "........"],  # 1
    0x32: [".####...", "##..##..", "....##..", "...##...",
           "..##....", ".##.....", "######..", "........"],  # 2
    0x33: ["######..", "....##..", "...##...", "..###...",
           "....##..", "##..##..", ".####...", "........"],  # 3
    0x34: ["...###..", "..####..", ".##.##..", "##..##..",
           "#######.", "....##..", "...###..", "........"],  # 4
    0x35: ["######..", "##......", "#####...", "....##..",
           "....##..", "##..##..", ".####...", "........"],  # 5
    0x36: ["..###...", ".##.....", "##......", "#####...",
           "##..##..", "##..##..", ".####...", "........"],  # 6
    0x37: ["######..", "##..##..", "....##..", "...##...",
           "..##....", "..##....", "..##....", "........"],  # 7
    0x38: [".####...", "##..##..", "##..##..", ".####...",
           "##..##..", "##..##..", ".####...", "........"],  # 8
    0x39: [".####...", "##..##..", "##..##..", ".#####..",
           "....##..", "...##...", ".###....", "........"],  # 9
    0x3A: ["........", "...##...", "...##...", "........",
           "........", "...##...", "...##...", "........"],  # :
    0x3B: ["........", "...##...", "...##...", "........",
           "...##...", "...##...", "..##....", "........"],  # ;
    0x3C: ["....###.", "..###...", ".##.....", "##......",
           ".##.....", "..###...", "....###.", "........"],  # <
    0x3D: ["........", "........", "#######.", "........",
           "#######.", "........", "........", "........"],  # =
    0x3E: [".###....", "...###..", ".....##.", "......##",
           ".....##.", "...###..", ".###....", "........"],  # >
    0x3F: [".####...", "##..##..", "....##..", "...##...",
           "...##...", "........", "...##...", "........"],  # ?
    0x40: [".####...", "##..##..", "##.###..", "##.###..",
           "##.###..", "##......", ".#####..", "........"],  # @
    0x41: [".####...", "##..##..", "##..##..", "######..",
           "##..##..", "##..##..", "##..##..", "........"],  # A
    0x42: ["#####...", "##..##..", "##..##..", "#####...",
           "##..##..", "##..##..", "#####...", "........"],  # B
    0x43: [".#####..", "##...##.", "##......", "##......",
           "##......", "##...##.", ".#####..", "........"],  # C
    0x44: ["####....", "##.##...", "##..##..", "##..##..",
           "##..##..", "##.##...", "####....", "........"],  # D
    0x45: ["######..", "##......", "##......", "#####...",
           "##......", "##......", "######..", "........"],  # E
    0x46: ["######..", "##......", "##......", "#####...",
           "##......", "##......", "##......", "........"],  # F
    0x47: [".#####..", "##...##.", "##......", "##.###..",
           "##..##..", "##...##.", ".####.#.", "........"],  # G
    0x48: ["##..##..", "##..##..", "##..##..", "######..",
           "##..##..", "##..##..", "##..##..", "........"],  # H
    0x49: [".####...", "..##....", "..##....", "..##....",
           "..##....", "..##....", ".####...", "........"],  # I
    0x4A: ["...###..", "....##..", "....##..", "....##..",
           "##..##..", "##..##..", ".####...", "........"],  # J
    0x4B: ["##..##..", "##.##...", "####....", "###.....",
           "####....", "##.##...", "##..##..", "........"],  # K
    0x4C: ["##......", "##......", "##......", "##......",
           "##......", "##......", "######..", "........"],  # L
    0x4D: ["##...##.", "###.###.", "#######.", "##.#.##.",
           "##...##.", "##...##.", "##...##.", "........"],  # M
    0x4E: ["##..##..", "###.##..", "####.##.", "##.###..",
           "##..###.", "##..###.", "##..##..", "........"],  # N
    0x4F: [".####...", "##..##..", "##..##..", "##..##..",
           "##..##..", "##..##..", ".####...", "........"],  # O
    0x50: ["#####...", "##..##..", "##..##..", "#####...",
           "##......", "##......", "##......", "........"],  # P
    0x51: [".####...", "##..##..", "##..##..", "##..##..",
           "##.###..", "##.##...", ".####.#.", "........"],  # Q
    0x52: ["#####...", "##..##..", "##..##..", "#####...",
           "####....", "##.##...", "##..##..", "........"],  # R
    0x53: [".#####..", "##......", "##......", ".####...",
           "....##..", "....##..", "#####...", "........"],  # S
    0x54: ["######..", "..##....", "..##....", "..##....",
           "..##....", "..##....", "..##....", "........"],  # T
    0x55: ["##..##..", "##..##..", "##..##..", "##..##..",
           "##..##..", "##..##..", ".####...", "........"],  # U
    0x56: ["##..##..", "##..##..", "##..##..", "##..##..",
           "##..##..", ".####...", "..##....", "........"],  # V
    0x57: ["##...##.", "##...##.", "##.#.##.", "#######.",
           "#######.", "###.###.", "##...##.", "........"],  # W
    0x58: ["##..##..", "##..##..", ".####...", "..##....",
           ".####...", "##..##..", "##..##..", "........"],  # X
    0x59: ["##..##..", "##..##..", "##..##..", ".####...",
           "..##....", "..##....", "..##....", "........"],  # Y
    0x5A: ["######..", "....##..", "...##...", "..##....",
           ".##.....", "##......", "######..", "........"],  # Z
    0x5B: [".#####..", ".##.....", ".##.....", ".##.....",
           ".##.....", ".##.....", ".#####..", "........"],  # [
    0x5C: [".##.....", "..##....", "...##...", "....##..",
           ".....##.", "......##", ".......#", "........"],  # backslash
    0x5D: [".#####..", "....##..", "....##..", "....##..",
           "....##..", "....##..", ".#####..", "........"],  # ]
    0x5E: ["...##...", "..####..", ".##..##.", "........",
           "........", "........", "........", "........"],  # ^
    0x5F: ["........", "........", "........", "........",
           "........", "........", "........", "#######."],  # _
    0x60: ["..##....", "...##...", "....##..", "........",
           "........", "........", "........", "........"],  # `
    0x61: ["........", "........", ".####...", "....##..",
           ".#####..", "##..##..", ".#####..", "........"],  # a
    0x62: ["##......", "##......", "#####...", "##..##..",
           "##..##..", "##..##..", "#####...", "........"],  # b
    0x63: ["........", "........", ".#####..", "##......",
           "##......", "##......", ".#####..", "........"],  # c
    0x64: ["....##..", "....##..", ".#####..", "##..##..",
           "##..##..", "##..##..", ".#####..", "........"],  # d
    0x65: ["........", "........", ".####...", "##..##..",
           "######..", "##......", ".#####..", "........"],  # e
    0x66: ["..###...", ".##.##..", ".##.....", "####....",
           ".##.....", ".##.....", ".##.....", "........"],  # f
    0x67: ["........", "........", ".#####..", "##..##..",
           "##..##..", ".#####..", "....##..", ".####..."],  # g
    0x68: ["##......", "##......", "#####...", "##..##..",
           "##..##..", "##..##..", "##..##..", "........"],  # h
    0x69: ["..##....", "........", ".###....", "..##....",
           "..##....", "..##....", ".####...", "........"],  # i
    0x6A: ["....##..", "........", "...###..", "....##..",
           "....##..", "##..##..", ".####...", "........"],  # j
    0x6B: ["##......", "##......", "##..##..", "##.##...",
           "####....", "##.##...", "##..##..", "........"],  # k
    0x6C: [".###....", "..##....", "..##....", "..##....",
           "..##....", "..##....", ".####...", "........"],  # l
    0x6D: ["........", "........", "###.##..", "#######.",
           "#######.", "##.#.##.", "##...##.", "........"],  # m
    0x6E: ["........", "........", "#####...", "##..##..",
           "##..##..", "##..##..", "##..##..", "........"],  # n
    0x6F: ["........", "........", ".####...", "##..##..",
           "##..##..", "##..##..", ".####...", "........"],  # o
    0x70: ["........", "........", "#####...", "##..##..",
           "##..##..", "#####...", "##......", "##......"],  # p
    0x71: ["........", "........", ".#####..", "##..##..",
           "##..##..", ".#####..", "....##..", "....##.."],  # q
    0x72: ["........", "........", "##.###..", "###.##..",
           "##......", "##......", "##......", "........"],  # r
    0x73: ["........", "........", ".#####..", "##......",
           ".####...", "....##..", "#####...", "........"],  # s
    0x74: [".##.....", ".##.....", "####....", ".##.....",
           ".##.....", ".##.##..", "..###...", "........"],  # t
    0x75: ["........", "........", "##..##..", "##..##..",
           "##..##..", "##..##..", ".#####..", "........"],  # u
    0x76: ["........", "........", "##..##..", "##..##..",
           "##..##..", ".####...", "..##....", "........"],  # v
    0x77: ["........", "........", "##...##.", "##.#.##.",
           "#######.", "#######.", ".##.##..", "........"],  # w
    0x78: ["........", "........", "##..##..", ".####...",
           "..##....", ".####...", "##..##..", "........"],  # x
    0x79: ["........", "........", "##..##..", "##..##..",
           "##..##..", ".#####..", "....##..", ".####..."],  # y
    0x7A: ["........", "........", "######..", "...##...",
           "..##....", ".##.....", "######..", "........"],  # z
    0x7B: ["...###..", "..##....", "..##....", ".##.....",
           "..##....", "..##....", "...###..", "........"],  # {
    0x7C: ["...##...", "...##...", "...##...", "...##...",
           "...##...", "...##...", "...##...", "........"],  # |
    0x7D: ["..###...", "....##..", "....##..", ".....##.",
           "....##..", "....##..", "..###...", "........"],  # }
    0x7E: [".###.##.", "##.###..", "........", "........",
           "........", "........", "........", "........"],  # ~
}

# CP437 box-drawing & block graphics (the subset AIOS uses for chrome).
_BOX_ART: Dict[int, List[str]] = {
    # Light shade ░ / medium ▒ / dark ▓
    0xB0: ["#.#.#.#.", ".#.#.#.#", "#.#.#.#.", ".#.#.#.#",
           "#.#.#.#.", ".#.#.#.#", "#.#.#.#.", ".#.#.#.#"],
    0xB1: ["#.#.#.#.", "########", ".#.#.#.#", "########",
           "#.#.#.#.", "########", ".#.#.#.#", "########"],
    0xB2: ["#####.##", "########", "##.#####", "########",
           "#####.##", "########", "##.#####", "########"],
    # Single line: ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼
    0xC4: ["........", "........", "........", "########",
           "........", "........", "........", "........"],  # ─
    0xB3: ["...##...", "...##...", "...##...", "...##...",
           "...##...", "...##...", "...##...", "...##..."],  # │
    0xDA: ["........", "........", "........", "...#####",
           "...##...", "...##...", "...##...", "...##..."],  # ┌
    0xBF: ["........", "........", "........", "#####...",
           "...##...", "...##...", "...##...", "...##..."],  # ┐
    0xC0: ["...##...", "...##...", "...##...", "...#####",
           "........", "........", "........", "........"],  # └
    0xD9: ["...##...", "...##...", "...##...", "#####...",
           "........", "........", "........", "........"],  # ┘
    0xC3: ["...##...", "...##...", "...##...", "...#####",
           "...##...", "...##...", "...##...", "...##..."],  # ├
    0xB4: ["...##...", "...##...", "...##...", "#####...",
           "...##...", "...##...", "...##...", "...##..."],  # ┤
    0xC2: ["........", "........", "........", "########",
           "...##...", "...##...", "...##...", "...##..."],  # ┬
    0xC1: ["...##...", "...##...", "...##...", "########",
           "........", "........", "........", "........"],  # ┴
    0xC5: ["...##...", "...##...", "...##...", "########",
           "...##...", "...##...", "...##...", "...##..."],  # ┼
    # Double line: ═ ║ ╔ ╗ ╚ ╝ ╠ ╣ ╦ ╩ ╬
    0xCD: ["........", "........", "########", "........",
           "########", "........", "........", "........"],  # ═
    0xBA: ["..##.##.", "..##.##.", "..##.##.", "..##.##.",
           "..##.##.", "..##.##.", "..##.##.", "..##.##."],  # ║
    0xC9: ["........", "........", "..######", "..##....",
           "..##.###", "..##.##.", "..##.##.", "..##.##."],  # ╔
    0xBB: ["........", "........", "######..", "....##..",
           "###.##..", ".##.##..", ".##.##..", ".##.##.."],  # ╗
    0xC8: ["..##.##.", "..##.##.", "..##.###", "..##....",
           "..######", "........", "........", "........"],  # ╚
    0xBC: [".##.##..", ".##.##..", "###.##..", "....##..",
           "######..", "........", "........", "........"],  # ╝
    0xCC: ["..##.##.", "..##.##.", "..##.###", "..##....",
           "..##.###", "..##.##.", "..##.##.", "..##.##."],  # ╠
    0xB9: [".##.##..", ".##.##..", "###.##..", "....##..",
           "###.##..", ".##.##..", ".##.##..", ".##.##.."],  # ╣
    0xCB: ["........", "........", "########", "........",
           "###..###", ".##..##.", ".##..##.", ".##..##."],  # ╦
    0xCA: [".##..##.", ".##..##.", "###..###", "........",
           "########", "........", "........", "........"],  # ╩
    0xCE: [".##..##.", ".##..##.", "###..###", "........",
           "###..###", ".##..##.", ".##..##.", ".##..##."],  # ╬
    # Blocks: █ ▄ ▀ ▌ ▐
    0xDB: ["########", "########", "########", "########",
           "########", "########", "########", "########"],  # █
    0xDC: ["........", "........", "........", "........",
           "########", "########", "########", "########"],  # ▄
    0xDF: ["########", "########", "########", "########",
           "........", "........", "........", "........"],  # ▀
    0xDD: ["####....", "####....", "####....", "####....",
           "####....", "####....", "####....", "####...."],  # ▌
    0xDE: ["....####", "....####", "....####", "....####",
           "....####", "....####", "....####", "....####"],  # ▐
}


class Font:
    """
    Immutable 8×8 bitmap font. Indexed by codepoint 0–255. Codepoints with
    no authored glyph resolve to .notdef. Exposes glyph(), pixel(), and a
    rasteriser used by Surface.draw_glyph in FRAMEBUFFER mode.
    """

    def __init__(self) -> None:
        # Flat table: 256 glyphs × 8 bytes.
        table = bytearray(256 * FONT_H)
        # Fill everything with .notdef first.
        for cp in range(256):
            table[cp * FONT_H:(cp + 1) * FONT_H] = _NOTDEF
        # Overlay authored glyphs.
        for cp, art in _ASCII_ART.items():
            table[cp * FONT_H:(cp + 1) * FONT_H] = _art(art)
        for cp, art in _BOX_ART.items():
            table[cp * FONT_H:(cp + 1) * FONT_H] = _art(art)
        # NUL renders as blank (common convention), not .notdef.
        table[0:FONT_H] = bytes(FONT_H)
        self._table = bytes(table)

    def glyph(self, cp: int) -> bytes:
        """Return the 8 bytes (one per row, MSB = leftmost) for codepoint cp."""
        cp &= 0xFF
        return self._table[cp * FONT_H:(cp + 1) * FONT_H]

    def pixel(self, cp: int, col: int, row: int) -> bool:
        """True iff pixel (col,row) of glyph cp is lit."""
        if not (0 <= col < FONT_W and 0 <= row < FONT_H):
            return False
        return bool(self._table[(cp & 0xFF) * FONT_H + row] & (1 << (7 - col)))

    @staticmethod
    def measure(text: str) -> Tuple[int, int]:
        """Pixel (w, h) of a single-line string at 8×8."""
        return len(text) * FONT_W, FONT_H


# Module-level default font instance (constructed once).
DEFAULT_FONT = Font()


# ════════════════════════════════════════════════════════════════════════════
#  §4 — SURFACE
#  A linear ARGB32 pixel buffer (4 bytes/pixel, little-endian word order so
#  that struct '<I' packs as B,G,R,A in memory — i.e. the framebuffer-friendly
#  BGRA byte layout). All drawing ops are clipped to an active clip Rect and
#  go through Porter-Duff blending. Thread-safe via an RLock.
# ════════════════════════════════════════════════════════════════════════════

class Surface:
    """Mutable ARGB32 raster target. Origin top-left, +x right, +y down."""

    BPP = 4  # bytes per pixel

    def __init__(self, width: int, height: int,
                 fill: Optional[Color] = None) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Surface dims must be positive: {width}x{height}")
        self.width = width
        self.height = height
        self._buf = bytearray(width * height * self.BPP)
        self._clip = Rect(0, 0, width, height)
        self._lock = threading.RLock()
        if fill is not None:
            self.clear(fill)

    # ── geometry / state ────────────────────────────────────────────────────

    def bounds(self) -> Rect:
        return Rect(0, 0, self.width, self.height)

    def set_clip(self, r: Optional[Rect]) -> None:
        """Restrict drawing to r (clamped to surface), or reset if None."""
        with self._lock:
            if r is None:
                self._clip = Rect(0, 0, self.width, self.height)
            else:
                self._clip = r.clip_to(self.bounds())

    def get_clip(self) -> Rect:
        return Rect(self._clip.x, self._clip.y, self._clip.w, self._clip.h)

    @contextmanager
    def clip(self, r: Rect):
        """Scoped clip: intersect with current clip for the block, then restore."""
        prev = self.get_clip()
        try:
            self.set_clip(r.clip_to(prev))
            yield
        finally:
            self.set_clip(prev)

    # ── raw pixel access ─────────────────────────────────────────────────────

    def _offset(self, x: int, y: int) -> int:
        return (y * self.width + x) * self.BPP

    def get_pixel(self, x: int, y: int) -> Color:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return Palette.TRANSPARENT
        o = self._offset(x, y)
        b, g, r, a = self._buf[o], self._buf[o+1], self._buf[o+2], self._buf[o+3]
        return Color(r, g, b, a)

    def _put_raw(self, x: int, y: int, r: int, g: int, b: int, a: int) -> None:
        """Write a pixel with no clip/bounds check (caller guarantees safety)."""
        o = self._offset(x, y)
        self._buf[o]   = b
        self._buf[o+1] = g
        self._buf[o+2] = r
        self._buf[o+3] = a

    def put_pixel(self, x: int, y: int, c: Color) -> None:
        """Clipped, blended single-pixel write."""
        with self._lock:
            if not self._clip.contains(x, y):
                return
            if c.a >= 255:
                self._put_raw(x, y, c.r, c.g, c.b, 255)
                return
            if c.a <= 0:
                return
            o = self._offset(x, y)
            db, dg, dr, da = (self._buf[o], self._buf[o+1],
                              self._buf[o+2], self._buf[o+3])
            nr, ng, nb, na = _alpha_blend(c.r, c.g, c.b, c.a, dr, dg, db, da)
            self._buf[o]   = nb
            self._buf[o+1] = ng
            self._buf[o+2] = nr
            self._buf[o+3] = na

    # ── fills ─────────────────────────────────────────────────────────────────

    def clear(self, c: Color) -> None:
        """Overwrite the whole surface (ignores clip), opaque store."""
        with self._lock:
            word = bytes((c.b, c.g, c.r, c.a))
            self._buf[:] = word * (self.width * self.height)

    def fill_rect(self, r: Rect, c: Color) -> None:
        """Clipped, blended rectangle fill."""
        with self._lock:
            cr = r.clip_to(self._clip)
            if cr.is_empty():
                return
            if c.a >= 255:
                row = bytes((c.b, c.g, c.r, 255)) * cr.w
                for yy in range(cr.y, cr.y2):
                    o = self._offset(cr.x, yy)
                    self._buf[o:o + cr.w * self.BPP] = row
            elif c.a > 0:
                for yy in range(cr.y, cr.y2):
                    for xx in range(cr.x, cr.x2):
                        o = self._offset(xx, yy)
                        db, dg, dr, da = (self._buf[o], self._buf[o+1],
                                          self._buf[o+2], self._buf[o+3])
                        nr, ng, nb, na = _alpha_blend(
                            c.r, c.g, c.b, c.a, dr, dg, db, da)
                        self._buf[o]   = nb
                        self._buf[o+1] = ng
                        self._buf[o+2] = nr
                        self._buf[o+3] = na

    # ── primitives ──────────────────────────────────────────────────────────

    def draw_hline(self, x: int, y: int, w: int, c: Color) -> None:
        self.fill_rect(Rect(x, y, w, 1), c)

    def draw_vline(self, x: int, y: int, h: int, c: Color) -> None:
        self.fill_rect(Rect(x, y, 1, h), c)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, c: Color) -> None:
        """Bresenham line (§1) with per-pixel clip+blend."""
        with self._lock:
            for px, py in _bresenham(x0, y0, x1, y1):
                if self._clip.contains(px, py):
                    self.put_pixel(px, py, c)

    def draw_rect(self, r: Rect, c: Color) -> None:
        """One-pixel rectangle outline."""
        if r.w <= 0 or r.h <= 0:
            return
        self.draw_hline(r.x, r.y, r.w, c)
        self.draw_hline(r.x, r.y2 - 1, r.w, c)
        self.draw_vline(r.x, r.y, r.h, c)
        self.draw_vline(r.x2 - 1, r.y, r.h, c)

    def draw_circle(self, cx: int, cy: int, radius: int, c: Color,
                    fill: bool = False) -> None:
        """
        Midpoint circle algorithm [Pitteway 1967 / Bresenham circle].
        Eq. MIDPOINT-CIRCLE: d = 1 − r; step e/se by sign(d).
        """
        if radius <= 0:
            return
        with self._lock:
            x, y = radius, 0
            err = 1 - radius
            def plot8(px: int, py: int) -> None:
                if fill:
                    self.draw_hline(cx - px, cy + py, 2 * px + 1, c)
                    self.draw_hline(cx - px, cy - py, 2 * px + 1, c)
                    self.draw_hline(cx - py, cy + px, 2 * py + 1, c)
                    self.draw_hline(cx - py, cy - px, 2 * py + 1, c)
                else:
                    for sx, sy in ((px, py), (-px, py), (px, -py), (-px, -py),
                                   (py, px), (-py, px), (py, -px), (-py, -px)):
                        self.put_pixel(cx + sx, cy + sy, c)
            while x >= y:
                plot8(x, y)
                y += 1
                if err < 0:
                    err += 2 * y + 1
                else:
                    x -= 1
                    err += 2 * (y - x) + 1

    # ── glyph / text ──────────────────────────────────────────────────────────

    def draw_glyph(self, cp: int, x: int, y: int, fg: Color,
                   bg: Optional[Color] = None,
                   font: Optional[Font] = None,
                   scale: int = 1) -> None:
        """
        Rasterise one codepoint at (x,y). Lit pixels use fg; if bg is given,
        unlit pixels are filled with bg (opaque cell), else left transparent.
        scale replicates each pixel into a scale×scale block (integer zoom).
        """
        font = font or DEFAULT_FONT
        glyph = font.glyph(cp)
        s = max(1, scale)
        with self._lock:
            for row in range(FONT_H):
                bits = glyph[row]
                for col in range(FONT_W):
                    on = bits & (1 << (7 - col))
                    color = fg if on else bg
                    if color is None:
                        continue
                    if s == 1:
                        self.put_pixel(x + col, y + row, color)
                    else:
                        self.fill_rect(
                            Rect(x + col * s, y + row * s, s, s), color)

    def draw_text(self, text: str, x: int, y: int, fg: Color,
                  bg: Optional[Color] = None,
                  font: Optional[Font] = None, scale: int = 1) -> int:
        """Draw a single line of text. Returns the x just past the last glyph."""
        s = max(1, scale)
        cx = x
        for ch in text:
            self.draw_glyph(ord(ch) & 0xFF, cx, y, fg, bg, font, s)
            cx += FONT_W * s
        return cx

    # ── blit ────────────────────────────────────────────────────────────────

    def blit(self, src: "Surface", dx: int, dy: int,
             src_rect: Optional[Rect] = None) -> None:
        """
        Copy src (or its src_rect sub-region) onto self at (dx,dy) with
        per-pixel Porter-Duff blending, clipped to this surface's clip Rect.
        """
        with self._lock, src._lock:
            sr = src_rect or src.bounds()
            sr = sr.clip_to(src.bounds())
            if sr.is_empty():
                return
            # Destination region, clipped.
            dst = Rect(dx, dy, sr.w, sr.h).clip_to(self._clip)
            if dst.is_empty():
                return
            ox = dst.x - dx   # offset into src for the clipped origin
            oy = dst.y - dy
            for yy in range(dst.h):
                s_off = src._offset(sr.x + ox, sr.y + oy + yy)
                d_off = self._offset(dst.x, dst.y + yy)
                for xx in range(dst.w):
                    so = s_off + xx * self.BPP
                    sa = src._buf[so + 3]
                    if sa == 0:
                        continue
                    do = d_off + xx * self.BPP
                    if sa == 255:
                        self._buf[do]   = src._buf[so]
                        self._buf[do+1] = src._buf[so+1]
                        self._buf[do+2] = src._buf[so+2]
                        self._buf[do+3] = 255
                        continue
                    sb, sg, srr = src._buf[so], src._buf[so+1], src._buf[so+2]
                    db, dg, drr, da = (self._buf[do], self._buf[do+1],
                                       self._buf[do+2], self._buf[do+3])
                    nr, ng, nb, na = _alpha_blend(srr, sg, sb, sa,
                                                  drr, dg, db, da)
                    self._buf[do]   = nb
                    self._buf[do+1] = ng
                    self._buf[do+2] = nr
                    self._buf[do+3] = na

    def copy(self) -> "Surface":
        """Deep copy of pixel data (clip resets to full bounds)."""
        with self._lock:
            s = Surface(self.width, self.height)
            s._buf[:] = self._buf
            return s

    def raw_bytes(self) -> bytes:
        """Return the BGRA byte buffer (immutable snapshot)."""
        with self._lock:
            return bytes(self._buf)


# ════════════════════════════════════════════════════════════════════════════
#  §5 — COMPOSITOR
#  Maintains an ordered stack of Layers (each a Surface + position + opacity),
#  tracks dirty regions as a list of AABB Rects, and composites only the dirty
#  area into a back buffer. Bottom-to-top painter's algorithm with PD blending.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Layer:
    """One compositable surface positioned on the virtual screen."""
    surface: Surface
    x: int = 0
    y: int = 0
    z: int = 0
    opacity: int = 255          # 0–255 global multiplier on layer alpha
    visible: bool = True
    name: str = ""

    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.surface.width, self.surface.height)


class Compositor:
    """
    Layer stack → back buffer. Dirty-rect driven: callers mark_dirty() when a
    layer changes; composite() rebuilds only the union-clipped dirty spans.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._layers: List[Layer] = []
        self._back = Surface(width, height, fill=Palette.DARK_BG)
        self._dirty: List[Rect] = [Rect(0, 0, width, height)]  # full first frame
        self._full_dirty = True
        self._lock = threading.RLock()

    # ── layer management ──────────────────────────────────────────────────────

    def add_layer(self, layer: Layer) -> Layer:
        with self._lock:
            self._layers.append(layer)
            self._sort_layers()
            self.mark_dirty(layer.rect())
            return layer

    def remove_layer(self, layer: Layer) -> None:
        with self._lock:
            if layer in self._layers:
                self._layers.remove(layer)
                self.mark_dirty(layer.rect())

    def _sort_layers(self) -> None:
        # Stable sort by z keeps insertion order within equal z.
        self._layers.sort(key=lambda L: L.z)

    def raise_layer(self, layer: Layer) -> None:
        """Move layer to the top of its z by giving it max z + 1."""
        with self._lock:
            if not self._layers:
                return
            top_z = max(L.z for L in self._layers)
            layer.z = top_z + 1
            self._sort_layers()
            self.mark_dirty(layer.rect())

    def layers(self) -> List[Layer]:
        with self._lock:
            return list(self._layers)

    # ── dirty tracking ─────────────────────────────────────────────────────────

    def mark_dirty(self, r: Rect) -> None:
        with self._lock:
            cr = r.clip_to(self._back.bounds())
            if cr.is_empty():
                return
            if self._full_dirty:
                return
            self._dirty.append(cr)
            if len(self._dirty) > DIRTY_FLUSH_CAP:
                self.mark_dirty_full()

    def mark_dirty_full(self) -> None:
        with self._lock:
            self._full_dirty = True
            self._dirty = [Rect(0, 0, self.width, self.height)]

    def _coalesce_dirty(self) -> List[Rect]:
        """
        Merge overlapping/adjacent dirty rects to reduce overdraw. Greedy
        union pass: repeatedly fuse any pair that intersects until stable.
        For the small N typical here this is well under the full-flush cap.
        """
        if self._full_dirty:
            return [Rect(0, 0, self.width, self.height)]
        rects = [r for r in self._dirty if not r.is_empty()]
        if not rects:
            return []
        merged = True
        while merged and len(rects) > 1:
            merged = False
            out: List[Rect] = []
            while rects:
                a = rects.pop()
                fused = False
                for i, b in enumerate(out):
                    if a.intersects(b):
                        out[i] = Rect.union(a, b)
                        fused = True
                        merged = True
                        break
                if not fused:
                    out.append(a)
            rects = out
        return rects

    # ── composite ─────────────────────────────────────────────────────────────

    def composite(self) -> Tuple[Surface, List[Rect]]:
        """
        Rebuild the back buffer for all dirty spans. Returns (back_surface,
        list_of_updated_rects) — the driver flushes exactly those spans.
        """
        with self._lock:
            spans = self._coalesce_dirty()
            for span in spans:
                # Paint the background, then each visible layer, bottom→top.
                self._back.set_clip(span)
                self._back.fill_rect(span, Palette.DARK_BG)
                for L in self._layers:
                    if not L.visible or L.opacity == 0:
                        continue
                    lr = L.rect()
                    if not lr.intersects(span):
                        continue
                    if L.opacity >= 255:
                        self._back.blit(L.surface, L.x, L.y)
                    else:
                        # Apply global opacity by pre-scaling alpha into a temp.
                        tmp = self._scale_alpha(L.surface, L.opacity)
                        self._back.blit(tmp, L.x, L.y)
            self._back.set_clip(None)
            updated = list(spans)
            self._dirty = []
            self._full_dirty = False
            return self._back, updated

    @staticmethod
    def _scale_alpha(src: Surface, opacity: int) -> Surface:
        """Return a copy of src with every alpha multiplied by opacity/255."""
        out = src.copy()
        buf = out._buf
        n = len(buf)
        i = 3
        while i < n:
            buf[i] = (buf[i] * opacity) // 255
            i += Surface.BPP
        return out

    def back_buffer(self) -> Surface:
        with self._lock:
            return self._back


# ════════════════════════════════════════════════════════════════════════════
#  §6 — WIDGET SYSTEM
#  A retained-mode widget tree. Each Widget owns a bounds Rect (local to its
#  parent), draws itself onto a target Surface at an absolute origin, and may
#  contain children. Layout containers (VBox/HBox) position children. Widgets
#  return True from on_event when they consume an event.
# ════════════════════════════════════════════════════════════════════════════

class Widget(ABC):
    """Abstract base for all UI elements."""

    def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0,
                 name: str = "") -> None:
        self.bounds = Rect(x, y, w, h)
        self.name = name
        self.visible = True
        self.focusable = False
        self.focused = False
        self.children: List["Widget"] = []
        self.parent: Optional["Widget"] = None
        self._dirty = True

    # ── tree ───────────────────────────────────────────────────────────────────
    def add(self, child: "Widget") -> "Widget":
        child.parent = self
        self.children.append(child)
        self.invalidate()
        return child

    def remove(self, child: "Widget") -> None:
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            self.invalidate()

    def invalidate(self) -> None:
        """Mark this widget (and ancestors) as needing repaint."""
        w: Optional[Widget] = self
        while w is not None:
            w._dirty = True
            w = w.parent

    def is_dirty(self) -> bool:
        return self._dirty or any(c.is_dirty() for c in self.children)

    def absolute_origin(self) -> Tuple[int, int]:
        ax, ay = self.bounds.x, self.bounds.y
        p = self.parent
        while p is not None:
            ax += p.bounds.x
            ay += p.bounds.y
            p = p.parent
        return ax, ay

    # ── render ───────────────────────────────────────────────────────────────
    def render(self, target: Surface, ox: int, oy: int) -> None:
        """Paint self at absolute (ox,oy) then recurse into children."""
        if not self.visible:
            return
        self.paint(target, ox, oy)
        for c in self.children:
            c.render(target, ox + c.bounds.x, oy + c.bounds.y)
        self._dirty = False

    @abstractmethod
    def paint(self, target: Surface, ox: int, oy: int) -> None:
        """Draw only this widget's own visuals (children handled by render)."""
        ...

    # ── input ──────────────────────────────────────────────────────────────────
    def on_event(self, ev: "Event") -> bool:
        """Default: dispatch to children top-most first; consume nothing."""
        for c in reversed(self.children):
            if c.visible and c.on_event(ev):
                return True
        return False

    def hit_test(self, px: int, py: int) -> Optional["Widget"]:
        """Return the top-most visible widget containing absolute point."""
        if not self.visible:
            return None
        ox, oy = self.absolute_origin()
        if not Rect(ox, oy, self.bounds.w, self.bounds.h).contains(px, py):
            return None
        for c in reversed(self.children):
            hit = c.hit_test(px, py)
            if hit is not None:
                return hit
        return self


class Panel(Widget):
    """A filled rectangle, optionally with a one-pixel border."""

    def __init__(self, x: int, y: int, w: int, h: int,
                 bg: Color = Palette.PANEL_BG,
                 border: Optional[Color] = None, name: str = "panel") -> None:
        super().__init__(x, y, w, h, name)
        self.bg = bg
        self.border = border

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        target.fill_rect(Rect(ox, oy, self.bounds.w, self.bounds.h), self.bg)
        if self.border is not None:
            target.draw_rect(Rect(ox, oy, self.bounds.w, self.bounds.h),
                             self.border)


class Label(Widget):
    """Single- or multi-line text. Width/height auto-fit if zero."""

    ALIGN_LEFT = 0
    ALIGN_CENTER = 1
    ALIGN_RIGHT = 2

    def __init__(self, x: int, y: int, text: str,
                 fg: Color = Palette.TEXT, bg: Optional[Color] = None,
                 scale: int = 1, align: int = 0, name: str = "label") -> None:
        self._text = text
        self.fg = fg
        self.bg = bg
        self.scale = max(1, scale)
        self.align = align
        lines = text.split("\n")
        w = max((len(ln) for ln in lines), default=0) * FONT_W * self.scale
        h = len(lines) * FONT_H * self.scale
        super().__init__(x, y, w, h, name)

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if value != self._text:
            self._text = value
            lines = value.split("\n")
            self.bounds.w = max((len(ln) for ln in lines), default=0) * FONT_W * self.scale
            self.bounds.h = len(lines) * FONT_H * self.scale
            self.invalidate()

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        line_h = FONT_H * self.scale
        for i, line in enumerate(self._text.split("\n")):
            lw = len(line) * FONT_W * self.scale
            if self.align == self.ALIGN_CENTER:
                lx = ox + (self.bounds.w - lw) // 2
            elif self.align == self.ALIGN_RIGHT:
                lx = ox + (self.bounds.w - lw)
            else:
                lx = ox
            target.draw_text(line, lx, oy + i * line_h,
                             self.fg, self.bg, scale=self.scale)


class Border(Widget):
    """A titled frame drawn with single-line box glyphs around its area."""

    def __init__(self, x: int, y: int, w: int, h: int, title: str = "",
                 color: Color = Palette.BORDER,
                 title_color: Color = Palette.TEXT, name: str = "border") -> None:
        super().__init__(x, y, w, h, name)
        self.title = title
        self.color = color
        self.title_color = title_color

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        w, h = self.bounds.w, self.bounds.h
        if w < 2 * FONT_W or h < 2 * FONT_H:
            return
        cols = w // FONT_W
        rows = h // FONT_H
        # Corners + edges using CP437 single-line glyphs.
        def g(cp: int, cx: int, cy: int, col: Color) -> None:
            target.draw_glyph(cp, ox + cx * FONT_W, oy + cy * FONT_H, col)
        g(0xDA, 0, 0, self.color)                 # ┌
        g(0xBF, cols - 1, 0, self.color)          # ┐
        g(0xC0, 0, rows - 1, self.color)          # └
        g(0xD9, cols - 1, rows - 1, self.color)   # ┘
        for cx in range(1, cols - 1):
            g(0xC4, cx, 0, self.color)            # ─ top
            g(0xC4, cx, rows - 1, self.color)     # ─ bottom
        for cy in range(1, rows - 1):
            g(0xB3, 0, cy, self.color)            # │ left
            g(0xB3, cols - 1, cy, self.color)     # │ right
        if self.title:
            t = " " + self.title[: max(0, cols - 4)] + " "
            tx = ox + ((cols - len(t)) // 2) * FONT_W
            target.draw_text(t, tx, oy, self.title_color, Palette.PANEL_BG)


class ProgressBar(Widget):
    """Horizontal progress bar. value in [0,1]."""

    def __init__(self, x: int, y: int, w: int, h: int, value: float = 0.0,
                 fg: Color = Palette.ACCENT, bg: Color = Palette.PANEL_BG,
                 border: Color = Palette.BORDER, name: str = "progress") -> None:
        super().__init__(x, y, w, h, name)
        self._value = _clampf(value, 0.0, 1.0)
        self.fg = fg
        self.bg = bg
        self.border = border

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        nv = _clampf(v, 0.0, 1.0)
        if nv != self._value:
            self._value = nv
            self.invalidate()

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        w, h = self.bounds.w, self.bounds.h
        target.fill_rect(Rect(ox, oy, w, h), self.bg)
        inner_w = max(0, w - 2)
        filled = int(inner_w * self._value)
        if filled > 0:
            target.fill_rect(Rect(ox + 1, oy + 1, filled, h - 2), self.fg)
        target.draw_rect(Rect(ox, oy, w, h), self.border)


class TextInput(Widget):
    """Single-line editable text field with a cursor and scroll-in-view."""

    def __init__(self, x: int, y: int, w: int, h: int, text: str = "",
                 fg: Color = Palette.TEXT, bg: Color = Palette.DARK_BG,
                 border: Color = Palette.BORDER,
                 cursor_color: Color = Palette.CURSOR, name: str = "input") -> None:
        super().__init__(x, y, w, h, name)
        self.focusable = True
        self.text = text
        self.cursor = len(text)
        self.scroll = 0
        self.fg = fg
        self.bg = bg
        self.border = border
        self.cursor_color = cursor_color

    def _visible_cols(self) -> int:
        return max(1, (self.bounds.w - 2 * FONT_W) // FONT_W)

    def _ensure_cursor_visible(self) -> None:
        vis = self._visible_cols()
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        elif self.cursor >= self.scroll + vis:
            self.scroll = self.cursor - vis + 1

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        w, h = self.bounds.w, self.bounds.h
        target.fill_rect(Rect(ox, oy, w, h), self.bg)
        target.draw_rect(Rect(ox, oy, w, h),
                         Palette.BORDER_ACTIVE if self.focused else self.border)
        self._ensure_cursor_visible()
        vis = self._visible_cols()
        shown = self.text[self.scroll:self.scroll + vis]
        ty = oy + (h - FONT_H) // 2
        target.draw_text(shown, ox + FONT_W, ty, self.fg, None)
        if self.focused:
            cur_col = self.cursor - self.scroll
            cx = ox + FONT_W + cur_col * FONT_W
            target.draw_vline(cx, ty, FONT_H, self.cursor_color)

    def on_event(self, ev: "Event") -> bool:
        if not self.focused or ev.type != EventType.KEY_DOWN:
            return False
        k = ev.key
        if k == KeyCode.LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif k == KeyCode.RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
        elif k == KeyCode.HOME:
            self.cursor = 0
        elif k == KeyCode.END:
            self.cursor = len(self.text)
        elif k == KeyCode.BACKSPACE:
            if self.cursor > 0:
                self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
                self.cursor -= 1
        elif k == KeyCode.DELETE:
            if self.cursor < len(self.text):
                self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
        elif ev.char and 32 <= ord(ev.char) < 127:
            self.text = self.text[:self.cursor] + ev.char + self.text[self.cursor:]
            self.cursor += 1
        else:
            return False
        self.invalidate()
        return True


class VBox(Widget):
    """Vertical layout: stacks children top-to-bottom with a fixed gap."""

    def __init__(self, x: int, y: int, w: int, h: int, gap: int = 2,
                 pad: int = 0, name: str = "vbox") -> None:
        super().__init__(x, y, w, h, name)
        self.gap = gap
        self.pad = pad

    def add(self, child: "Widget") -> "Widget":
        super().add(child)
        self._relayout()
        return child

    def _relayout(self) -> None:
        cy = self.pad
        for c in self.children:
            c.bounds.x = self.pad
            c.bounds.y = cy
            if c.bounds.w == 0:
                c.bounds.w = self.bounds.w - 2 * self.pad
            cy += c.bounds.h + self.gap

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        return  # transparent container


class HBox(Widget):
    """Horizontal layout: lays children left-to-right with a fixed gap."""

    def __init__(self, x: int, y: int, w: int, h: int, gap: int = 2,
                 pad: int = 0, name: str = "hbox") -> None:
        super().__init__(x, y, w, h, name)
        self.gap = gap
        self.pad = pad

    def add(self, child: "Widget") -> "Widget":
        super().add(child)
        self._relayout()
        return child

    def _relayout(self) -> None:
        cx = self.pad
        for c in self.children:
            c.bounds.y = self.pad
            c.bounds.x = cx
            if c.bounds.h == 0:
                c.bounds.h = self.bounds.h - 2 * self.pad
            cx += c.bounds.w + self.gap

    def paint(self, target: Surface, ox: int, oy: int) -> None:
        return  # transparent container


# ════════════════════════════════════════════════════════════════════════════
#  §7 — EVENT SYSTEM
#  A thread-safe EventQueue plus an InputDriver that puts the terminal into
#  cbreak/raw mode and decodes:
#    • printable bytes            → KEY_DOWN with .char
#    • C0 controls (Enter/Tab/BS) → KEY_DOWN with .key
#    • CSI/SS3 escape sequences   → arrows, Home/End/PgUp/PgDn, F-keys
#    • SGR mouse (1006) reports   → MOUSE_DOWN/UP/MOVE/WHEEL
#  Decoding is incremental and never blocks longer than the poll timeout.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Event:
    """One input/system event. Unused fields stay at their defaults."""
    type: EventType
    key: int = 0                 # KeyCode or ord(char) for KEY_DOWN
    char: str = ""               # printable character, if any
    mods: KeyMod = KeyMod.NONE
    x: int = 0                   # mouse cell column (0-based)
    y: int = 0                   # mouse cell row (0-based)
    button: int = 0              # MouseButton for mouse events
    wheel: int = 0               # +1 up / −1 down for MOUSE_WHEEL
    width: int = 0               # RESIZE: new columns
    height: int = 0              # RESIZE: new rows
    timestamp: float = field(default_factory=time.monotonic)


class EventQueue:
    """Bounded thread-safe FIFO of Events."""

    def __init__(self, maxlen: int = 1024) -> None:
        self._q: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def push(self, ev: Event) -> None:
        with self._cv:
            self._q.append(ev)
            self._cv.notify()

    def pop(self, timeout: Optional[float] = None) -> Optional[Event]:
        with self._cv:
            if not self._q:
                if timeout is not None and not self._cv.wait(timeout):
                    return None
                if not self._q:
                    return None
            return self._q.popleft()

    def poll(self) -> Optional[Event]:
        """Non-blocking pop."""
        with self._lock:
            return self._q.popleft() if self._q else None

    def drain(self) -> List[Event]:
        with self._lock:
            evs = list(self._q)
            self._q.clear()
            return evs

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)


# CSI final-byte → KeyCode for the common cursor/navigation set.
_CSI_MAP: Dict[str, int] = {
    "A": KeyCode.UP, "B": KeyCode.DOWN, "C": KeyCode.RIGHT, "D": KeyCode.LEFT,
    "H": KeyCode.HOME, "F": KeyCode.END,
}
# CSI <n> '~' → KeyCode (xterm/VT220 numbered keys).
_CSI_TILDE_MAP: Dict[int, int] = {
    1: KeyCode.HOME, 2: KeyCode.INSERT, 3: KeyCode.DELETE, 4: KeyCode.END,
    5: KeyCode.PAGE_UP, 6: KeyCode.PAGE_DOWN, 7: KeyCode.HOME, 8: KeyCode.END,
    11: KeyCode.F1, 12: KeyCode.F2, 13: KeyCode.F3, 14: KeyCode.F4,
    15: KeyCode.F5, 17: KeyCode.F6, 18: KeyCode.F7, 19: KeyCode.F8,
    20: KeyCode.F9, 21: KeyCode.F10, 23: KeyCode.F11, 24: KeyCode.F12,
}
# SS3 final byte (ESC O x) → KeyCode for F1–F4 / app-cursor mode.
_SS3_MAP: Dict[str, int] = {
    "P": KeyCode.F1, "Q": KeyCode.F2, "R": KeyCode.F3, "S": KeyCode.F4,
    "A": KeyCode.UP, "B": KeyCode.DOWN, "C": KeyCode.RIGHT, "D": KeyCode.LEFT,
    "H": KeyCode.HOME, "F": KeyCode.END,
}


def decode_escape(seq: str) -> Optional[Event]:
    """
    Decode a complete escape sequence (without the leading ESC) into an Event.
    Returns None if the sequence is not recognised.

    Supported:
      CSI A/B/C/D/H/F             cursor + home/end
      CSI <n> ~                   VT220 numbered keys
      SS3 (ESC O) P/Q/R/S/...     F1–F4 / app-cursor
      SGR mouse  CSI < b ; x ; y (M|m)   xterm 1006 protocol
    """
    if not seq:
        return Event(EventType.KEY_DOWN, key=KeyCode.ESCAPE)
    # SS3: 'O' + final
    if seq[0] == "O" and len(seq) >= 2:
        kc = _SS3_MAP.get(seq[1])
        if kc is not None:
            return Event(EventType.KEY_DOWN, key=kc)
        return None
    if seq[0] != "[":
        return None
    body = seq[1:]
    # SGR mouse: '<' params final∈{M,m}
    if body.startswith("<") and body and body[-1] in ("M", "m"):
        try:
            params = body[1:-1].split(";")
            b = int(params[0]); cx = int(params[1]); cy = int(params[2])
            pressed = body[-1] == "M"
            # Wheel events: bit 6 (64) set.
            if b & 64:
                return Event(EventType.MOUSE_WHEEL,
                             x=cx - 1, y=cy - 1,
                             wheel=1 if (b & 1) == 0 else -1)
            btn = b & 3
            etype = EventType.MOUSE_DOWN if pressed else EventType.MOUSE_UP
            # Motion (bit 5 / 32) → MOUSE_MOVE regardless of press/release.
            if b & 32:
                etype = EventType.MOUSE_MOVE
            return Event(etype, x=cx - 1, y=cy - 1, button=btn)
        except (ValueError, IndexError):
            return None
    # CSI <n> ~
    if body and body[-1] == "~":
        try:
            n = int(body[:-1].split(";")[0])
        except ValueError:
            return None
        kc = _CSI_TILDE_MAP.get(n)
        return Event(EventType.KEY_DOWN, key=kc) if kc else None
    # CSI final letter (possibly with modifier params like '1;5A')
    final = body[-1]
    kc = _CSI_MAP.get(final)
    if kc is not None:
        mods = KeyMod.NONE
        if ";" in body:
            try:
                mod_n = int(body[:-1].split(";")[1])
                # xterm modifier encoding: value−1 is a bitmask (1=Shift,2=Alt,4=Ctrl)
                m = mod_n - 1
                if m & 1: mods |= KeyMod.SHIFT
                if m & 2: mods |= KeyMod.ALT
                if m & 4: mods |= KeyMod.CTRL
            except (ValueError, IndexError):
                pass
        return Event(EventType.KEY_DOWN, key=kc, mods=mods)
    return None


class InputDriver:
    """
    Background reader thread that decodes stdin into Events. POSIX-only raw
    mode; on non-tty / non-POSIX it stays inert (no events, no errors), so the
    display still runs for headless rendering and tests.
    """

    def __init__(self, queue: EventQueue, fd: int = 0,
                 poll_timeout: float = 0.05) -> None:
        self._queue = queue
        self._fd = fd
        self._poll = poll_timeout
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._saved_attrs: Optional[list] = None
        self._is_tty = _HAS_TTY and os.isatty(fd)

    # ── terminal mode ──────────────────────────────────────────────────────────
    def _enter_raw(self) -> None:
        if not self._is_tty:
            return
        self._saved_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        # Enable SGR mouse tracking (1000 = button, 1002 = drag, 1006 = SGR).
        os.write(1, b"\x1b[?1000h\x1b[?1002h\x1b[?1006h")

    def _exit_raw(self) -> None:
        if not self._is_tty:
            return
        os.write(1, b"\x1b[?1000l\x1b[?1002l\x1b[?1006l")
        if self._saved_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            self._saved_attrs = None

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._enter_raw()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="aios-input",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self._poll * 4)
        self._exit_raw()

    # ── reader loop ────────────────────────────────────────────────────────────
    def _read_byte(self) -> Optional[int]:
        r, _, _ = select.select([self._fd], [], [], self._poll)
        if not r:
            return None
        data = os.read(self._fd, 1)
        return data[0] if data else None

    def _read_escape_seq(self) -> str:
        """
        After an ESC, read the rest of a CSI/SS3 sequence. Bounded by a short
        inter-byte timeout so a lone ESC keypress doesn't hang.
        """
        seq = ""
        b = self._read_byte()
        if b is None:
            return seq          # lone ESC
        ch = chr(b)
        seq += ch
        if ch == "O":           # SS3: exactly one more byte
            nb = self._read_byte()
            if nb is not None:
                seq += chr(nb)
            return seq
        if ch == "[":           # CSI: read until a final byte (0x40–0x7E)
            while True:
                nb = self._read_byte()
                if nb is None:
                    break
                seq += chr(nb)
                if 0x40 <= nb <= 0x7E:
                    break
        return seq

    def _loop(self) -> None:
        if not self._is_tty:
            # Inert: park until stopped.
            while self._running:
                time.sleep(self._poll)
            return
        while self._running:
            b = self._read_byte()
            if b is None:
                continue
            if b == 0x1b:                       # ESC → maybe escape sequence
                seq = self._read_escape_seq()
                ev = decode_escape(seq)
                if ev is not None:
                    self._queue.push(ev)
                continue
            if b in (0x0d, 0x0a):               # CR / LF → Enter
                self._queue.push(Event(EventType.KEY_DOWN, key=KeyCode.ENTER))
            elif b == 0x09:                     # Tab
                self._queue.push(Event(EventType.KEY_DOWN, key=KeyCode.TAB))
            elif b in (0x7f, 0x08):             # DEL / BS
                self._queue.push(Event(EventType.KEY_DOWN, key=KeyCode.BACKSPACE))
            elif b < 0x20:                      # other C0 controls (Ctrl-x)
                self._queue.push(Event(EventType.KEY_DOWN, key=b,
                                       mods=KeyMod.CTRL))
            elif 0x20 <= b < 0x7f:              # printable ASCII
                self._queue.push(Event(EventType.KEY_DOWN, key=b, char=chr(b)))
            else:                               # high byte → treat as Latin-1
                self._queue.push(Event(EventType.KEY_DOWN, key=b, char=chr(b)))


# ════════════════════════════════════════════════════════════════════════════
#  §8 — DISPLAY DRIVERS
#  A driver turns a composited Surface into actual output. The default
#  ANSITerminalDriver uses the upper-half-block trick (▀ U+2580): each terminal
#  cell encodes TWO stacked pixels — the glyph's foreground is the top pixel,
#  its background the bottom pixel — doubling vertical resolution. Truecolor
#  SGR (38;2 / 48;2) is emitted with run-length suppression of redundant codes.
# ════════════════════════════════════════════════════════════════════════════

class DisplayDriver(ABC):
    """Abstract sink. Knows its pixel resolution and how to flush spans."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @abstractmethod
    def present(self, surface: Surface, spans: List[Rect]) -> None:
        """Push the given dirty spans of surface to the device."""
        ...

    def begin(self) -> None:
        """Enter device mode (alt screen, hide cursor, …)."""
        return

    def end(self) -> None:
        """Restore the device to its prior state."""
        return

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


class ANSITerminalDriver(DisplayDriver):
    """
    Truecolor half-block renderer to a file descriptor (default stdout).
    Pixel resolution is (cols, rows×2): rows = terminal rows.
    """

    def __init__(self, cols: Optional[int] = None, rows: Optional[int] = None,
                 out_fd: int = 1) -> None:
        c, r = self._term_size()
        cols = cols or c
        rows = rows or r
        super().__init__(width=cols, height=rows * 2)
        self.cols = cols
        self.rows = rows
        self._out = out_fd
        self._entered = False

    @staticmethod
    def _term_size() -> Tuple[int, int]:
        try:
            sz = os.get_terminal_size()
            return sz.columns, sz.lines
        except OSError:
            return 80, 24

    def begin(self) -> None:
        if self._entered:
            return
        # Alt screen, clear, hide cursor.
        os.write(self._out, b"\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l")
        self._entered = True

    def end(self) -> None:
        if not self._entered:
            return
        # Show cursor, leave alt screen, reset SGR.
        os.write(self._out, b"\x1b[?25h\x1b[?1049l\x1b[0m")
        self._entered = False

    def present(self, surface: Surface, spans: List[Rect]) -> None:
        if not spans:
            return
        out = []
        last_fg: Optional[int] = None
        last_bg: Optional[int] = None
        for span in spans:
            # Convert pixel span to the rows of cells it touches.
            cy0 = span.y // 2
            cy1 = (span.y2 + 1) // 2
            cx0 = span.x
            cx1 = min(span.x2, surface.width)
            for cy in range(cy0, min(cy1, self.rows)):
                # Move cursor to cell (cy+1, cx0+1) — ANSI is 1-based.
                out.append(f"\x1b[{cy + 1};{cx0 + 1}H")
                last_fg = last_bg = None  # cursor move resets our run state
                for cx in range(cx0, cx1):
                    top = surface.get_pixel(cx, cy * 2)
                    bot = surface.get_pixel(cx, cy * 2 + 1)
                    fg = (top.r, top.g, top.b)
                    bg = (bot.r, bot.g, bot.b)
                    fg_key = (fg[0] << 16) | (fg[1] << 8) | fg[2]
                    bg_key = (bg[0] << 16) | (bg[1] << 8) | bg[2]
                    if fg_key != last_fg:
                        out.append(f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m")
                        last_fg = fg_key
                    if bg_key != last_bg:
                        out.append(f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m")
                        last_bg = bg_key
                    out.append(HALF_BLOCK)
        out.append("\x1b[0m")
        os.write(self._out, "".join(out).encode("utf-8"))

    def resize(self, width: int, height: int) -> None:
        self.cols = width
        self.rows = height
        super().resize(width, height * 2)


class ANSITextDriver(DisplayDriver):
    """
    Lower-fidelity fallback: one cell per pixel-row pair rendered as a shaded
    ASCII ramp by luminance. Useful where truecolor / Unicode is unavailable.
    Luminance: Y = 0.299R + 0.587G + 0.114B  [ITU-R BT.601].
    """

    _RAMP = " .:-=+*#%@"

    def __init__(self, cols: Optional[int] = None, rows: Optional[int] = None,
                 out_fd: int = 1) -> None:
        c, r = ANSITerminalDriver._term_size()
        cols = cols or c
        rows = rows or r
        super().__init__(width=cols, height=rows)
        self.cols = cols
        self.rows = rows
        self._out = out_fd
        self._entered = False

    def begin(self) -> None:
        if not self._entered:
            os.write(self._out, b"\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l")
            self._entered = True

    def end(self) -> None:
        if self._entered:
            os.write(self._out, b"\x1b[?25h\x1b[?1049l\x1b[0m")
            self._entered = False

    def present(self, surface: Surface, spans: List[Rect]) -> None:
        if not spans:
            return
        out = []
        ramp = self._RAMP
        rmax = len(ramp) - 1
        for span in spans:
            for cy in range(span.y, min(span.y2, self.rows)):
                out.append(f"\x1b[{cy + 1};{span.x + 1}H")
                for cx in range(span.x, min(span.x2, self.cols)):
                    p = surface.get_pixel(cx, cy)
                    y = (299 * p.r + 587 * p.g + 114 * p.b) // 1000
                    out.append(ramp[(y * rmax) // 255])
        os.write(self._out, "".join(out).encode("utf-8"))


class VGAAdapterDriver(DisplayDriver):
    """
    Bridge to aios_core.VGATextDriver (80×25 text buffer @ 0xB8000). Maps the
    composited surface to the 16-color VGA palette by nearest match, sampling
    one pixel per text cell. Active only when the kernel VGA driver is supplied.
    """

    def __init__(self, vga_text_driver: Any, cols: int = 80, rows: int = 25) -> None:
        super().__init__(width=cols, height=rows)
        self._vga = vga_text_driver
        self.cols = cols
        self.rows = rows

    @staticmethod
    def _nearest_vga16(c: Color) -> int:
        best_i, best_d = 0, 1 << 30
        for i, pc in enumerate(VGA16):
            d = (c.r - pc.r) ** 2 + (c.g - pc.g) ** 2 + (c.b - pc.b) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def present(self, surface: Surface, spans: List[Rect]) -> None:
        # VGATextDriver writes sequentially and auto-wraps at 80 cols /
        # auto-scrolls at 25 rows, with no public cursor-addressing call. The
        # correct idiom for that interface is a full repaint: clear() homes the
        # cursor, then writing exactly cols×rows cells fills the screen in
        # raster order. Each cell is a space with its background set to the
        # nearest VGA-16 color, which renders as a solid color block.
        #
        # spans are intentionally ignored here: the device is a fixed 80×25
        # cell grid that is cheap to repaint in full, and partial addressing is
        # not exposed by the underlying driver.
        if not spans:
            return
        self._vga.clear()
        for cy in range(self.rows):
            for cx in range(self.cols):
                idx = self._nearest_vga16(surface.get_pixel(cx, cy))
                self._vga.set_color(0, idx)   # fg unused for space; bg = color
                self._vga.putchar(" ")


class FramebufferDriver(DisplayDriver):
    """
    Linear framebuffer sink. Opens an mmap-able device (default /dev/fb0) and
    copies BGRA spans directly. Falls back to an in-memory buffer if no device
    is present, so the pipeline is testable without hardware.
    """

    def __init__(self, width: int, height: int, device: str = "/dev/fb0",
                 bpp: int = 32, line_length: Optional[int] = None) -> None:
        super().__init__(width, height)
        self.device = device
        self.bpp = bpp
        self.bytes_pp = bpp // 8
        self.line_length = line_length or width * self.bytes_pp
        self._fd: Optional[int] = None
        self._mm = None
        self._mem = bytearray(self.line_length * height)  # fallback store
        self._open()

    def _open(self) -> None:
        try:
            import mmap as _mmap
            import fcntl as _fcntl
            self._fd = os.open(self.device, os.O_RDWR)

            # ── query real framebuffer geometry via Linux fb ioctls ───────────
            # FBIOGET_VSCREENINFO (0x4600) → struct fb_var_screeninfo (160 bytes)
            #   uint32 xres        @ offset  0
            #   uint32 yres        @ offset  4
            #   uint32 bits_per_px @ offset 24
            # FBIOGET_FSCREENINFO (0x4602) → struct fb_fix_screeninfo (80 bytes)
            #   uint32 line_length @ offset 48  (64-bit layout: id[16] + ulong(8) + …)
            _FBIOGET_VSCREENINFO = 0x4600
            _FBIOGET_FSCREENINFO = 0x4602
            buf_var = bytearray(160)
            buf_fix = bytearray(80)
            try:
                _fcntl.ioctl(self._fd, _FBIOGET_VSCREENINFO, buf_var)
                _fcntl.ioctl(self._fd, _FBIOGET_FSCREENINFO, buf_fix)
                xres, yres = struct.unpack_from('<II', buf_var, 0)
                bpp_hw     = struct.unpack_from('<I', buf_var, 24)[0]
                ll_hw      = struct.unpack_from('<I', buf_fix, 48)[0]
                if xres > 0 and yres > 0:
                    self.width  = xres
                    self.height = yres
                if bpp_hw in (16, 24, 32):
                    self.bpp      = bpp_hw
                    self.bytes_pp = bpp_hw // 8
                if ll_hw > 0:
                    self.line_length = ll_hw
                # Resize the in-memory fallback to match real geometry
                self._mem = bytearray(self.line_length * self.height)
            except OSError:
                pass  # ioctl unsupported (VM / fake fb) — keep constructor defaults

            size = self.line_length * self.height
            self._mm = _mmap.mmap(self._fd, size,
                                  _mmap.MAP_SHARED,
                                  _mmap.PROT_READ | _mmap.PROT_WRITE)
        except (OSError, ImportError):
            self._fd = None
            self._mm = None  # use in-memory fallback

    def present(self, surface: Surface, spans: List[Rect]) -> None:
        target = self._mm if self._mm is not None else self._mem
        for span in spans:
            x0 = max(0, span.x); x1 = min(span.x2, self.width)
            y0 = max(0, span.y); y1 = min(span.y2, self.height)
            row_w = (x1 - x0) * Surface.BPP
            if row_w <= 0:
                continue
            for yy in range(y0, y1):
                s_off = surface._offset(x0, yy)
                d_off = yy * self.line_length + x0 * self.bytes_pp
                if self.bytes_pp == Surface.BPP:
                    target[d_off:d_off + row_w] = surface._buf[s_off:s_off + row_w]
                else:
                    # Repack BGRA→device bpp pixel by pixel (e.g. 24-bpp).
                    for xx in range(x1 - x0):
                        so = s_off + xx * Surface.BPP
                        do = d_off + xx * self.bytes_pp
                        target[do]   = surface._buf[so]
                        target[do+1] = surface._buf[so+1]
                        target[do+2] = surface._buf[so+2]
                        if self.bytes_pp == 4:
                            target[do+3] = surface._buf[so+3]
        if self._mm is not None:
            try:
                self._mm.flush()
            except OSError:
                # /dev/fb0 is a character device mapped to video RAM.
                # msync() (called by mmap.flush()) returns EINVAL on char
                # devices — the write is already live in device memory.
                pass

    def end(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ════════════════════════════════════════════════════════════════════════════
#  §9 — WINDOW MANAGER
#  A Window owns a content Surface, a title bar, a draggable frame, and a root
#  Widget for its client area. The WindowManager holds a Compositor, assigns
#  each window a Layer, tracks focus + z-order, and routes events to the
#  focused / hit window. Title-bar drag moves windows; click raises + focuses.
# ════════════════════════════════════════════════════════════════════════════

TITLE_H = FONT_H + 4            # title-bar height in pixels
FRAME   = 2                     # frame thickness in pixels


class Window:
    """A movable, focusable, decorated surface with a client widget tree."""

    _next_id = 1
    _id_lock = threading.Lock()

    def __init__(self, title: str, x: int, y: int, w: int, h: int) -> None:
        with Window._id_lock:
            self.id = Window._next_id
            Window._next_id += 1
        self.title = title
        self.rect = Rect(x, y, w, h)
        self.state = WindowState.NORMAL
        self.focused = False
        self.surface = Surface(w, h, fill=Palette.PANEL_BG)
        self.layer: Optional[Layer] = None
        # Client area sits below the title bar, inside the frame.
        self.client = Panel(FRAME, TITLE_H, w - 2 * FRAME, h - TITLE_H - FRAME,
                            bg=Palette.PANEL_BG, name=f"win{self.id}-client")
        self._focused_widget: Optional[Widget] = None
        self._dirty = True

    # ── geometry helpers ────────────────────────────────────────────────────
    def client_rect(self) -> Rect:
        return Rect(FRAME, TITLE_H, self.rect.w - 2 * FRAME,
                    self.rect.h - TITLE_H - FRAME)

    def title_bar_rect_abs(self) -> Rect:
        return Rect(self.rect.x, self.rect.y, self.rect.w, TITLE_H)

    def contains_abs(self, px: int, py: int) -> bool:
        return self.rect.contains(px, py)

    # ── focus ────────────────────────────────────────────────────────────────
    def focus_widget(self, w: Optional[Widget]) -> None:
        if self._focused_widget is w:
            return
        if self._focused_widget is not None:
            self._focused_widget.focused = False
            self._focused_widget.invalidate()
        self._focused_widget = w
        if w is not None:
            w.focused = True
            w.invalidate()
        self._dirty = True

    def focus_next(self) -> None:
        """Cycle focus among focusable widgets in the client tree."""
        focusables = self._collect_focusable(self.client)
        if not focusables:
            return
        if self._focused_widget in focusables:
            i = (focusables.index(self._focused_widget) + 1) % len(focusables)
        else:
            i = 0
        self.focus_widget(focusables[i])

    @staticmethod
    def _collect_focusable(root: Widget) -> List[Widget]:
        out: List[Widget] = []
        stack = [root]
        while stack:
            w = stack.pop()
            if w.focusable and w.visible:
                out.append(w)
            stack.extend(reversed(w.children))
        return out

    # ── rendering ──────────────────────────────────────────────────────────
    def add(self, widget: Widget) -> Widget:
        self.client.add(widget)
        self._dirty = True
        return widget

    def is_dirty(self) -> bool:
        return self._dirty or self.client.is_dirty()

    def repaint(self) -> None:
        """Redraw the whole window onto its own surface."""
        s = self.surface
        s.set_clip(None)
        # Frame + background.
        s.clear(Palette.PANEL_BG)
        frame_col = Palette.BORDER_ACTIVE if self.focused else Palette.BORDER
        s.draw_rect(Rect(0, 0, self.rect.w, self.rect.h), frame_col)
        # Title bar.
        s.fill_rect(Rect(0, 0, self.rect.w, TITLE_H),
                    Palette.TITLE_BG if self.focused else Palette.PANEL_BG)
        s.draw_hline(0, TITLE_H - 1, self.rect.w, frame_col)
        title = self.title[: max(0, (self.rect.w - 2 * FONT_W) // FONT_W)]
        s.draw_text(title, FONT_W // 2, 2,
                    Palette.TEXT if self.focused else Palette.TEXT_DIM)
        # Close glyph (×) at the right of the title bar.
        s.draw_text("x", self.rect.w - FONT_W - 2, 2, Palette.ERROR)
        # Client widget tree, clipped to the client area.
        cr = self.client_rect()
        with s.clip(cr):
            self.client.render(s, cr.x, cr.y)
        self._dirty = False

    # ── input ──────────────────────────────────────────────────────────────
    def on_event(self, ev: Event) -> bool:
        # Translate absolute mouse coords into client-local for hit-testing.
        if ev.type in (EventType.MOUSE_DOWN, EventType.MOUSE_UP,
                       EventType.MOUSE_MOVE):
            local = Event(ev.type, x=ev.x - self.rect.x - FRAME,
                          y=ev.y - self.rect.y - TITLE_H,
                          button=ev.button)
            hit = self.client.hit_test(local.x, local.y)
            if ev.type == EventType.MOUSE_DOWN and hit is not None and hit.focusable:
                self.focus_widget(hit)
            if hit is not None and hit.on_event(local):
                self._dirty = True
                return True
            return False
        if ev.type == EventType.KEY_DOWN:
            if ev.key == KeyCode.TAB:
                self.focus_next()
                return True
            if self._focused_widget is not None and self._focused_widget.on_event(ev):
                self._dirty = True
                return True
        return False


class WindowManager:
    """Owns the compositor, the window list, focus, z-order, and drag state."""

    def __init__(self, width: int, height: int) -> None:
        self.compositor = Compositor(width, height)
        self.width = width
        self.height = height
        self._windows: List[Window] = []
        self._focused: Optional[Window] = None
        self._z_counter = 0
        self._drag: Optional[Tuple[Window, int, int]] = None  # (win, dx, dy)
        self._lock = threading.RLock()

    # ── window lifecycle ──────────────────────────────────────────────────────
    def create_window(self, title: str, x: int, y: int,
                      w: int, h: int) -> Window:
        with self._lock:
            if len(self._windows) >= MAX_WINDOWS:
                raise RuntimeError("window limit reached")
            win = Window(title, x, y, w, h)
            layer = Layer(win.surface, x, y, z=self._z_counter,
                          name=f"win{win.id}")
            self._z_counter += 1
            win.layer = layer
            self.compositor.add_layer(layer)
            self._windows.append(win)
            self.focus_window(win)
            win.repaint()
            return win

    def destroy_window(self, win: Window) -> None:
        with self._lock:
            if win not in self._windows:
                return
            if win.layer is not None:
                self.compositor.remove_layer(win.layer)
            self._windows.remove(win)
            if self._focused is win:
                self._focused = None
                if self._windows:
                    self.focus_window(self._windows[-1])

    def windows(self) -> List[Window]:
        with self._lock:
            return list(self._windows)

    # ── focus / z-order ───────────────────────────────────────────────────────
    def focus_window(self, win: Window) -> None:
        with self._lock:
            if self._focused is win and win.focused:
                return
            if self._focused is not None and self._focused is not win:
                self._focused.focused = False
                self._focused._dirty = True
            self._focused = win
            win.focused = True
            win._dirty = True
            # Raise to top.
            if win.layer is not None:
                self.compositor.raise_layer(win.layer)

    def focused_window(self) -> Optional[Window]:
        return self._focused

    def window_at(self, px: int, py: int) -> Optional[Window]:
        """Top-most window under the absolute point."""
        with self._lock:
            for win in sorted(self._windows,
                              key=lambda w: w.layer.z if w.layer else 0,
                              reverse=True):
                if win.state != WindowState.HIDDEN and win.contains_abs(px, py):
                    return win
            return None

    def move_window(self, win: Window, x: int, y: int) -> None:
        with self._lock:
            old = win.rect
            win.rect = Rect(x, y, old.w, old.h)
            if win.layer is not None:
                # Dirty both old and new footprints.
                self.compositor.mark_dirty(Rect(old.x, old.y, old.w, old.h))
                win.layer.x = x
                win.layer.y = y
                self.compositor.mark_dirty(win.layer.rect())

    # ── event routing ──────────────────────────────────────────────────────────
    def dispatch(self, ev: Event) -> None:
        with self._lock:
            if ev.type == EventType.MOUSE_DOWN:
                self._on_mouse_down(ev)
            elif ev.type == EventType.MOUSE_UP:
                self._drag = None
                w = self._focused
                if w is not None:
                    w.on_event(ev)
            elif ev.type == EventType.MOUSE_MOVE:
                if self._drag is not None:
                    win, dx, dy = self._drag
                    self.move_window(win, ev.x - dx, ev.y - dy)
                else:
                    w = self._focused
                    if w is not None:
                        w.on_event(ev)
            elif ev.type == EventType.KEY_DOWN:
                if self._focused is not None:
                    self._focused.on_event(ev)

    def _on_mouse_down(self, ev: Event) -> None:
        win = self.window_at(ev.x, ev.y)
        if win is None:
            return
        self.focus_window(win)
        # Close button hit? (top-right glyph cell)
        close_x0 = win.rect.x + win.rect.w - FONT_W - 2
        if (ev.y < win.rect.y + TITLE_H and ev.x >= close_x0):
            self.destroy_window(win)
            return
        # Title-bar drag start?
        if win.title_bar_rect_abs().contains(ev.x, ev.y):
            self._drag = (win, ev.x - win.rect.x, ev.y - win.rect.y)
            return
        # Otherwise forward to window client.
        win.on_event(ev)

    # ── frame production ────────────────────────────────────────────────────────
    def render_frame(self) -> Tuple[Surface, List[Rect]]:
        """Repaint dirty windows, then composite. Returns (surface, spans)."""
        with self._lock:
            for win in self._windows:
                if win.is_dirty() and win.layer is not None:
                    win.repaint()
                    self.compositor.mark_dirty(win.layer.rect())
            return self.compositor.composite()


# ════════════════════════════════════════════════════════════════════════════
#  §10 — DISPLAY MANAGER
#  Top-level orchestrator. Owns the WindowManager, EventQueue, InputDriver, and
#  a DisplayDriver. Runs the frame loop (drain events → dispatch → render →
#  present) on a worker thread, and exposes @agent_method tools so the AIOS
#  agent kernel can create windows, draw, and query the display as agent calls.
# ════════════════════════════════════════════════════════════════════════════

class DisplayManager:
    """The display server. One per physical/virtual output."""

    def __init__(self, mode: DisplayMode = DisplayMode.ANSI_TERMINAL,
                 width: Optional[int] = None, height: Optional[int] = None,
                 target_fps: int = 30,
                 vga_text_driver: Any = None,
                 fb_device: str = "/dev/fb0") -> None:
        self.mode = mode
        self.target_fps = max(1, target_fps)
        self._frame_budget = 1.0 / self.target_fps
        self.driver = self._make_driver(mode, width, height,
                                        vga_text_driver, fb_device)
        self.width = self.driver.width
        self.height = self.driver.height
        self.wm = WindowManager(self.width, self.height)
        self.events = EventQueue()
        self.input = InputDriver(self.events)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._frame_count = 0
        self._last_present = 0.0
        self._reasoner = None  # set by AgentKernel.attach if present

    # ── driver construction ─────────────────────────────────────────────────
    @staticmethod
    def _make_driver(mode: DisplayMode, width: Optional[int],
                     height: Optional[int], vga: Any,
                     fb_device: str) -> DisplayDriver:
        if mode == DisplayMode.ANSI_TERMINAL:
            return ANSITerminalDriver(cols=width, rows=height)
        if mode == DisplayMode.VGA_TEXT:
            if vga is None:
                raise ValueError("VGA_TEXT mode requires vga_text_driver")
            return VGAAdapterDriver(vga, cols=width or 80, rows=height or 25)
        if mode == DisplayMode.FRAMEBUFFER:
            w = width or 640
            h = height or 480
            return FramebufferDriver(w, h, device=fb_device)
        raise ValueError(f"unknown display mode {mode}")

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @agent_method(
        name="display_start",
        description="Start the display server: enter device mode, begin input, "
                    "and launch the render loop thread.",
        priority=AgentPriority.HIGH,
        owner="display",
    )
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self.driver.begin()
            self.input.start()
            self._running = True
            self._thread = threading.Thread(target=self._run_loop,
                                            name="aios-display", daemon=True)
            self._thread.start()

    @agent_method(
        name="display_stop",
        description="Stop the render loop, restore terminal state, release input.",
        priority=AgentPriority.HIGH,
        owner="display",
    )
    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self._frame_budget * 8)
        self.input.stop()
        self.driver.end()

    def is_running(self) -> bool:
        return self._running

    # ── main loop ─────────────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            self.tick()
            elapsed = time.monotonic() - t0
            sleep_for = self._frame_budget - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    @agent_method(
        name="display_tick",
        description="Run one frame: drain & dispatch events, render dirty "
                    "windows, composite, and present updated spans.",
        priority=AgentPriority.NORMAL,
        owner="display",
    )
    def tick(self) -> int:
        """Process one frame. Returns the number of events handled."""
        handled = 0
        for ev in self.events.drain():
            if ev.type == EventType.QUIT:
                self._running = False
            elif ev.type == EventType.RESIZE:
                self._handle_resize(ev.width, ev.height)
            else:
                self.wm.dispatch(ev)
            handled += 1
        surface, spans = self.wm.render_frame()
        if spans:
            self.driver.present(surface, spans)
            self._last_present = time.monotonic()
        self._frame_count += 1
        return handled

    def _handle_resize(self, cols: int, rows: int) -> None:
        with self._lock:
            self.driver.resize(cols, rows)
            self.width = self.driver.width
            self.height = self.driver.height
            # Rebuild compositor at the new size, preserving windows.
            new_comp = Compositor(self.width, self.height)
            old_wm = self.wm
            self.wm.compositor = new_comp
            self.wm.width = self.width
            self.wm.height = self.height
            for win in old_wm.windows():
                if win.layer is not None:
                    new_comp.add_layer(win.layer)
            new_comp.mark_dirty_full()

    # ── agent-facing surface API ───────────────────────────────────────────────
    @agent_method(
        name="display_create_window",
        description="Create a decorated window and return its integer id.",
        parameters={
            "title":  {"type": "str", "desc": "Title-bar text"},
            "x":      {"type": "int", "desc": "Left position (pixels)"},
            "y":      {"type": "int", "desc": "Top position (pixels)"},
            "w":      {"type": "int", "desc": "Width (pixels)"},
            "h":      {"type": "int", "desc": "Height (pixels)"},
        },
        returns="int",
        priority=AgentPriority.NORMAL,
        owner="display",
    )
    def create_window(self, title: str, x: int, y: int, w: int, h: int) -> int:
        win = self.wm.create_window(title, x, y, w, h)
        return win.id

    @agent_method(
        name="display_close_window",
        description="Destroy the window with the given id.",
        parameters={"window_id": {"type": "int", "desc": "Window id"}},
        returns="bool",
        priority=AgentPriority.NORMAL,
        owner="display",
    )
    def close_window(self, window_id: int) -> bool:
        win = self._find(window_id)
        if win is None:
            return False
        self.wm.destroy_window(win)
        return True

    @agent_method(
        name="display_add_label",
        description="Add a text label to a window's client area.",
        parameters={
            "window_id": {"type": "int", "desc": "Target window id"},
            "x":         {"type": "int", "desc": "Client-local x (pixels)"},
            "y":         {"type": "int", "desc": "Client-local y (pixels)"},
            "text":      {"type": "str", "desc": "Label text"},
        },
        returns="bool",
        priority=AgentPriority.NORMAL,
        owner="display",
    )
    def add_label(self, window_id: int, x: int, y: int, text: str) -> bool:
        win = self._find(window_id)
        if win is None:
            return False
        win.add(Label(x, y, text))
        return True

    @agent_method(
        name="display_set_window_title",
        description="Change a window's title-bar text.",
        parameters={
            "window_id": {"type": "int", "desc": "Window id"},
            "title":     {"type": "str", "desc": "New title text"},
        },
        returns="bool",
        priority=AgentPriority.LOW,
        owner="display",
    )
    def set_window_title(self, window_id: int, title: str) -> bool:
        win = self._find(window_id)
        if win is None:
            return False
        win.title = title
        win._dirty = True
        return True

    @agent_method(
        name="display_post_key",
        description="Inject a key event into the input queue (printable char "
                    "or KeyCode integer).",
        parameters={
            "key":  {"type": "int", "desc": "KeyCode or ord(char)"},
            "char": {"type": "str", "desc": "Printable character, may be empty"},
        },
        returns="None",
        priority=AgentPriority.NORMAL,
        owner="display",
    )
    def post_key(self, key: int, char: str = "") -> None:
        self.events.push(Event(EventType.KEY_DOWN, key=key, char=char))

    @agent_method(
        name="display_post_mouse",
        description="Inject a mouse event (down/up/move) at a cell position.",
        parameters={
            "etype":  {"type": "int", "desc": "EventType value"},
            "x":      {"type": "int", "desc": "Cell column"},
            "y":      {"type": "int", "desc": "Cell row"},
            "button": {"type": "int", "desc": "MouseButton value"},
        },
        returns="None",
        priority=AgentPriority.NORMAL,
        owner="display",
    )
    def post_mouse(self, etype: int, x: int, y: int, button: int = 0) -> None:
        self.events.push(Event(EventType(etype), x=x, y=y, button=button))

    @agent_method(
        name="display_stats",
        description="Return display server telemetry: frame count, window "
                    "count, resolution, fps target, pending events.",
        returns="Dict[str, Any]",
        priority=AgentPriority.LOW,
        owner="display",
    )
    def stats(self) -> Dict[str, Any]:
        return {
            "mode": int(self.mode),
            "resolution": (self.width, self.height),
            "windows": len(self.wm.windows()),
            "frames": self._frame_count,
            "target_fps": self.target_fps,
            "pending_events": len(self.events),
            "running": self._running,
        }

    # ── helpers ─────────────────────────────────────────────────────────────
    def _find(self, window_id: int) -> Optional[Window]:
        for win in self.wm.windows():
            if win.id == window_id:
                return win
        return None

    def attach(self, kernel: Any) -> None:
        """
        Wire this display server into an AgentKernel. The kernel gains a
        `.display` handle, and the manager picks up the kernel's reasoner so
        agent-method annotations flow through the same path as the rest of the
        OS. Idempotent and safe if the kernel lacks either attribute.
        """
        setattr(kernel, "display", self)
        self._reasoner = getattr(kernel, "_reasoner", None)


# ════════════════════════════════════════════════════════════════════════════
#  §11 — SELF-TEST SUITE
#  Exercises every layer of the pipeline against hand-computed reference values
#  and structural invariants. All tests run headless (no tty, no framebuffer
#  device) so the suite passes in CI. Run:  python3 aios_display.py --selftest
# ════════════════════════════════════════════════════════════════════════════

def _selftest() -> int:
    """Return 0 if every check passes, else the count of failures."""
    failures: List[str] = []

    def check(cond: bool, label: str) -> None:
        if cond:
            print(f"  \x1b[32mPASS\x1b[0m  {label}")
        else:
            print(f"  \x1b[31mFAIL\x1b[0m  {label}")
            failures.append(label)

    print("AIOS Display Manager self-test  v%d.%d.%d" % DISPLAY_VERSION)
    print("-" * 60)

    # ── §1 math ──────────────────────────────────────────────────────────────
    print("§1 math primitives")
    check(_isqrt(0) == 0 and _isqrt(1) == 1, "isqrt edge cases")
    check(_isqrt(99) == 9 and _isqrt(100) == 10, "isqrt boundary 99/100")
    check(_isqrt(1 << 40) == (1 << 20), "isqrt of 2^40")
    check(_clamp(15, 0, 10) == 10 and _clamp(-3, 0, 10) == 0, "clamp")
    # Bresenham: a 45° line from (0,0)→(3,3) is the main diagonal.
    diag = _bresenham(0, 0, 3, 3)
    check(diag == [(0, 0), (1, 1), (2, 2), (3, 3)], "bresenham diagonal")
    # Horizontal line endpoint count = dx+1.
    check(len(_bresenham(0, 0, 9, 0)) == 10, "bresenham horizontal length")
    # Porter-Duff: opaque source replaces destination.
    check(_alpha_blend(10, 20, 30, 255, 0, 0, 0, 255) == (10, 20, 30, 255),
          "PD opaque source")
    # 50% white over black ≈ mid-gray.
    r, g, b, a = _alpha_blend(255, 255, 255, 128, 0, 0, 0, 255)
    check(125 <= r <= 130 and a == 255, "PD 50% blend → mid-gray")
    # Rect union / clip arithmetic.
    u = Rect.union(Rect(0, 0, 4, 4), Rect(6, 6, 2, 2))
    check(u.x == 0 and u.y == 0 and u.x2 == 8 and u.y2 == 8, "rect union AABB")
    cl = Rect(0, 0, 10, 10).clip_to(Rect(5, 5, 10, 10))
    check(cl.x == 5 and cl.w == 5, "rect clip intersection")

    # ── §2 color ───────────────────────────────────────────────────────────────
    print("§2 color system")
    check(rgb_to_ansi256(255, 0, 0) == 196, "ansi256 pure red → 196")
    check(rgb_to_ansi256(0, 0, 0) == 16, "ansi256 black → 16")
    check(rgb_to_ansi256(255, 255, 255) == 231, "ansi256 white → 231")
    check(232 <= rgb_to_ansi256(128, 128, 128) <= 255, "ansi256 gray → ramp")
    c = Color.from_hex("#3C82FF")
    check(c.r == 0x3C and c.g == 0x82 and c.b == 0xFF, "hex parse")
    rt = Color.from_argb32(Color(12, 34, 56, 78).to_argb32())
    check((rt.r, rt.g, rt.b, rt.a) == (12, 34, 56, 78), "argb32 round-trip")

    # ── §3 font ────────────────────────────────────────────────────────────────
    print("§3 font engine")
    f = DEFAULT_FONT
    check(f.glyph(0x20) == bytes(8), "space glyph empty")
    check(f.glyph(0xDB) == bytes([0xFF] * 8), "full block all-ones")
    check(f.glyph(0x00) == bytes(8), "NUL renders blank")
    check(f.glyph(0xFE) == _NOTDEF, "unauthored cp → .notdef")
    check(f.pixel(0x41, 0, 0) is False and f.pixel(0x41, 1, 0) is True,
          "glyph 'A' top-row shape")
    lhb = f.glyph(0xDC)
    check(lhb[:4] == bytes(4) and all(x == 0xFF for x in lhb[4:]),
          "lower-half block split")

    # ── §4 surface ─────────────────────────────────────────────────────────────
    print("§4 surface")
    s = Surface(16, 16, fill=Palette.BLACK)
    s.put_pixel(3, 4, Color(200, 100, 50))
    px = s.get_pixel(3, 4)
    check((px.r, px.g, px.b) == (200, 100, 50), "pixel store/fetch (BGRA order)")
    s.fill_rect(Rect(0, 0, 4, 4), Color(0, 255, 0))
    check(s.get_pixel(3, 3) == Color(0, 255, 0, 255), "fill_rect")
    s.set_clip(Rect(0, 0, 2, 2))
    s.fill_rect(Rect(0, 0, 16, 16), Color(0, 0, 255))
    inside = s.get_pixel(1, 1) == Color(0, 0, 255, 255)
    outside = s.get_pixel(5, 5) == Color(0, 0, 0, 255)
    check(inside and outside, "clip rect confines drawing")
    s.set_clip(None)
    # blit z-correctness
    top = Surface(4, 4, fill=Color(255, 0, 0))
    dst = Surface(8, 8, fill=Color(0, 0, 0))
    dst.blit(top, 2, 2)
    check(dst.get_pixel(2, 2) == Color(255, 0, 0, 255) and
          dst.get_pixel(0, 0) == Color(0, 0, 0, 255), "blit placement")
    # translucent blit blends
    tl = Surface(2, 2, fill=Color(255, 255, 255, 128))
    base = Surface(2, 2, fill=Color(0, 0, 0))
    base.blit(tl, 0, 0)
    check(120 <= base.get_pixel(0, 0).r <= 135, "blit alpha blend")

    # ── §5 compositor ───────────────────────────────────────────────────────────
    print("§5 compositor")
    comp = Compositor(40, 20)
    red = Surface(10, 10, fill=Color(255, 0, 0))
    blue = Surface(10, 10, fill=Color(0, 0, 255))
    lred = comp.add_layer(Layer(red, 0, 0, z=0, name="red"))
    comp.add_layer(Layer(blue, 5, 5, z=1, name="blue"))
    back, spans = comp.composite()
    check(back.get_pixel(7, 7) == Color(0, 0, 255, 255), "z-order: blue over red")
    check(back.get_pixel(2, 2) == Color(255, 0, 0, 255), "z-order: red visible")
    check(back.get_pixel(35, 18) == Palette.DARK_BG, "compositor background")
    _, spans2 = comp.composite()
    check(spans2 == [], "dirty cleared after composite")
    comp.raise_layer(lred)
    back3, _ = comp.composite()
    check(back3.get_pixel(7, 7) == Color(255, 0, 0, 255), "raise_layer reorders")

    # ── §6 widgets ──────────────────────────────────────────────────────────────
    print("§6 widgets")
    lbl = Label(0, 0, "Hi")
    check(lbl.bounds.w == 2 * FONT_W and lbl.bounds.h == FONT_H, "label autosize")
    lbl.text = "Hello\nWorld"
    check(lbl.bounds.h == 2 * FONT_H, "label multiline resize")
    pb = ProgressBar(0, 0, 100, 10, value=0.5)
    pb.value = 1.5
    check(pb.value == 1.0, "progressbar clamps to 1.0")
    ti = TextInput(0, 0, 100, 16, text="ab")
    ti.focused = True
    ti.cursor = 2
    ti.on_event(Event(EventType.KEY_DOWN, key=ord("c"), char="c"))
    check(ti.text == "abc" and ti.cursor == 3, "textinput insert")
    ti.on_event(Event(EventType.KEY_DOWN, key=KeyCode.BACKSPACE))
    check(ti.text == "ab" and ti.cursor == 2, "textinput backspace")
    ti.on_event(Event(EventType.KEY_DOWN, key=KeyCode.LEFT))
    ti.on_event(Event(EventType.KEY_DOWN, key=KeyCode.DELETE))
    check(ti.text == "a", "textinput left+delete")
    # hit-testing nested widgets
    root = Panel(0, 0, 100, 100)
    child = root.add(Panel(10, 10, 20, 20, name="child"))
    hit = root.hit_test(15, 15)
    check(hit is child, "hit_test returns nested child")
    check(root.hit_test(95, 95) is root, "hit_test returns parent")

    # ── §7 events ───────────────────────────────────────────────────────────────
    print("§7 events")
    check(decode_escape("[A").key == KeyCode.UP, "decode arrow up")
    check(decode_escape("[5~").key == KeyCode.PAGE_UP, "decode VT220 PgUp")
    check(decode_escape("OP").key == KeyCode.F1, "decode SS3 F1")
    mev = decode_escape("[1;5A")
    check(mev.key == KeyCode.UP and (mev.mods & KeyMod.CTRL), "decode Ctrl+Up")
    mdown = decode_escape("[<0;11;6M")
    check(mdown.type == EventType.MOUSE_DOWN and mdown.x == 10 and mdown.y == 5,
          "decode SGR mouse down")
    wheel = decode_escape("[<64;5;5M")
    check(wheel.type == EventType.MOUSE_WHEEL and wheel.wheel == 1,
          "decode SGR wheel up")
    check(decode_escape("").key == KeyCode.ESCAPE, "lone ESC")
    q = EventQueue(maxlen=4)
    q.push(Event(EventType.KEY_DOWN, key=1))
    q.push(Event(EventType.KEY_DOWN, key=2))
    drained = q.drain()
    check(len(drained) == 2 and drained[0].key == 1, "event queue FIFO drain")

    # ── §8 drivers ──────────────────────────────────────────────────────────────
    print("§8 drivers")
    fb = FramebufferDriver(8, 4, device="/dev/nonexistent_fb_for_test")
    fs = Surface(8, 4, fill=Color(10, 20, 30))
    fs.put_pixel(0, 0, Color(255, 128, 64))
    fb.present(fs, [Rect(0, 0, 8, 4)])
    mem = fb._mem
    check(mem[0] == 64 and mem[1] == 128 and mem[2] == 255,
          "framebuffer BGRA span copy")
    # ANSI half-block through a pipe
    rfd, wfd = os.pipe()
    drv = ANSITerminalDriver(cols=4, rows=2, out_fd=wfd)
    asurf = Surface(4, 4, fill=Color(0, 0, 0))
    asurf.put_pixel(0, 0, Color(255, 0, 0))
    drv.present(asurf, [Rect(0, 0, 4, 4)])
    os.close(wfd)
    blob = os.read(rfd, 1 << 16).decode("utf-8")
    os.close(rfd)
    check(HALF_BLOCK in blob, "ANSI emits half-block glyph")
    check("38;2;255;0;0" in blob, "ANSI emits truecolor red fg")
    vga_recorder = _FakeVGA()
    vdrv = VGAAdapterDriver(vga_recorder, cols=4, rows=2)
    vsurf = Surface(4, 2, fill=Color(255, 0, 0))
    vdrv.present(vsurf, [Rect(0, 0, 4, 2)])
    check(vga_recorder.cells == 8 and vga_recorder.last_bg == 4,
          "VGA adapter paints all cells, red→index 4")

    # ── §9 window manager ───────────────────────────────────────────────────────
    print("§9 window manager")
    wm = WindowManager(200, 100)
    w1 = wm.create_window("One", 10, 10, 80, 60)
    w2 = wm.create_window("Two", 40, 30, 80, 60)
    check(len(wm.windows()) == 2, "two windows created")
    check(wm.focused_window() is w2, "newest window focused")
    # w2 is on top where they overlap → window_at returns w2
    check(wm.window_at(50, 40) is w2, "window_at picks top window")
    # focus + raise w1
    wm.focus_window(w1)
    check(wm.window_at(50, 40) is w1, "focus raises window to top")
    # drag w2 via title bar
    wm.focus_window(w2)
    wm.dispatch(Event(EventType.MOUSE_DOWN, x=45, y=33, button=0))
    wm.dispatch(Event(EventType.MOUSE_MOVE, x=65, y=53))
    check(w2.rect.x == 60 and w2.rect.y == 50, "title-bar drag moves window")
    wm.dispatch(Event(EventType.MOUSE_UP, x=65, y=53))
    # close via title-bar close glyph
    before = len(wm.windows())
    close_x = w2.rect.x + w2.rect.w - FONT_W - 1
    wm.dispatch(Event(EventType.MOUSE_DOWN, x=close_x, y=w2.rect.y + 2, button=0))
    check(len(wm.windows()) == before - 1, "close glyph destroys window")

    # ── §10 end-to-end pipeline ─────────────────────────────────────────────────
    print("§10 end-to-end DisplayManager")
    dm = DisplayManager(mode=DisplayMode.FRAMEBUFFER, width=120, height=60,
                        fb_device="/dev/nonexistent_fb_for_test")
    wid = dm.create_window("Agent", 5, 5, 90, 40)
    check(isinstance(wid, int) and wid > 0, "agent create_window returns id")
    check(dm.add_label(wid, 4, 4, "READY"), "agent add_label")
    check(dm.set_window_title(wid, "Agent ✓"), "agent set_window_title")
    handled = dm.tick()  # render one frame headless
    st = dm.stats()
    check(st["windows"] == 1 and st["frames"] >= 1, "stats reflect state")
    # The composited frame should have non-background pixels where the window is.
    surf = dm.wm.compositor.back_buffer()
    win_pixel = surf.get_pixel(20, 20)
    check(win_pixel != Palette.DARK_BG, "window rendered onto back buffer")
    check(dm.close_window(wid) and dm.stats()["windows"] == 0,
          "agent close_window")
    # inject + process an event through the public queue
    dm.post_key(ord("z"), "z")
    check(len(dm.events) == 1, "post_key enqueues event")
    dm.tick()
    check(len(dm.events) == 0, "tick drains event queue")

    # ── agent-method registration (integrated mode only) ─────────────────────────
    print("§10 agent integration")
    if _AIOS_INTEGRATED and AgentRegistry is not None:
        tool_names = {spec.name for spec in AgentRegistry().all_tools()}
        expected = {
            "display_start", "display_stop", "display_tick",
            "display_create_window", "display_close_window", "display_add_label",
            "display_set_window_title", "display_post_key", "display_post_mouse",
            "display_stats",
        }
        missing = expected - tool_names
        check(not missing,
              f"all {len(expected)} display tools registered"
              + (f" (missing {missing})" if missing else ""))
    else:
        check(True, "standalone shim active (aios_core absent)")

    print("-" * 60)
    if failures:
        print(f"\x1b[31m{len(failures)} FAILURE(S)\x1b[0m: " + "; ".join(failures))
        return len(failures)
    print("\x1b[32mALL CHECKS PASSED\x1b[0m")
    return 0


class _FakeVGA:
    """Minimal stand-in for aios_core.VGATextDriver used by the self-test."""

    def __init__(self) -> None:
        self.cells = 0
        self.last_fg = 0
        self.last_bg = 0
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1

    def set_color(self, fg: int, bg: int) -> None:
        self.last_fg = fg
        self.last_bg = bg

    def putchar(self, ch: str) -> None:
        self.cells += 1

    def write(self, s: str) -> None:
        for _ in s:
            self.cells += 1


def _demo() -> None:
    """
    Live ANSI demo: opens a couple of windows with widgets and animates a
    progress bar for a few seconds. Requires a truecolor terminal. Exits
    cleanly on Ctrl-C or after the timed run.
    """
    dm = DisplayManager(mode=DisplayMode.ANSI_TERMINAL, target_fps=30)
    try:
        dm.start()
        wid = dm.create_window("AIOS — Neural Terminal",
                               4, 2, dm.width - 8, dm.height - 6)
        win = dm._find(wid)
        if win is not None:
            win.add(Label(4, 4, "Display Manager online.", fg=Palette.SUCCESS))
            win.add(Label(4, 16, "Half-block truecolor compositor.",
                          fg=Palette.TEXT))
            bar = win.add(ProgressBar(4, 32, 200, 10, value=0.0))
            win.add(Label(4, 48, "Press Ctrl-C to exit.", fg=Palette.TEXT_DIM))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0:
            if win is not None:
                bar.value = ((time.monotonic() - t0) / 6.0)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dm.stop()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        sys.exit(_selftest())
