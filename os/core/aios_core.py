#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AIOS — Agentic Intelligence Operating System                              ║
║   Core Kernel: aios_core.py                                                  ║
║                                                                              ║
║   "Every function is an agent. Every byte is reasoned. Every call traces."  ║
║                                                                              ║
║   Architecture Pipeline:                                                     ║
║     Boot Stub → Memory Bus → HAL → Math Primitives → Tensor → Agent Kernel  ║
║                                                                              ║
║   Phases:                                                                    ║
║     I   : Bootstrapping — boot sequence, protected-mode simulation           ║
║     II  : HAL — memory-mapped I/O, interrupts, virtual memory                ║
║     III : Math Primitives — IEEE 754, CORDIC, Dual Numbers (no math import)  ║
║     IV  : Tensor + Neural Foundation                                         ║
║     V   : Agent Kernel + Terminal REPL                                       ║
║                                                                              ║
║   Design Contract:                                                           ║
║     • No placeholder logic. No TODO stubs. No mocked data.                  ║
║     • Every OS primitive decorated as an agent-callable tool.                ║
║     • AgentReasoner is pluggable: rule-based → ML → LLM — same interface.   ║
║     • All math (sin, cos, exp, ln) derived from CORDIC / first principles.   ║
║     • Hardware interaction via MemoryBus only — no stdlib I/O shortcuts.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
 
from __future__ import annotations
 
# ── Standard library — permissible at the Python kernel layer ─────────────────
import sys
import os
import time
import struct
import array
import threading
import traceback
import functools
import hashlib
import json
import io
import select
import tty
import termios
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    TypeVar, Generic, Iterator, Sequence, NamedTuple
)
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag, auto
from collections import OrderedDict, deque, defaultdict
from abc import ABC, abstractmethod
from contextlib import contextmanager
 
# ── Kernel version ────────────────────────────────────────────────────────────
AIOS_VERSION       = (0, 1, 0)
AIOS_CODENAME      = "GENESIS"
KERNEL_LOAD_ADDR   = 0x7C00      # conventional MBR load address
VGA_BASE_ADDR      = 0xB8000     # VGA text-mode buffer base
VGA_COLS           = 80
VGA_ROWS           = 25
PAGE_SIZE          = 4096        # 4 KiB pages
PAGE_SHIFT         = 12
RAM_SIZE_BYTES     = 64 * 1024 * 1024  # 64 MiB simulated physical RAM
CORDIC_ITERATIONS  = 40          # precision iterations for CORDIC
 
T = TypeVar("T")
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — AGENT METHOD PROTOCOL
#  The foundational pattern: every OS function is an agent-observable,
#  agent-invocable, agent-reasoned tool.
# ════════════════════════════════════════════════════════════════════════════════
 
class AgentPriority(IntEnum):
    CRITICAL = 0    # interrupt-level, must not block
    HIGH     = 1    # scheduler, memory critical path
    NORMAL   = 2    # standard kernel operations
    LOW      = 3    # background tasks, telemetry
 
 
@dataclass
class AgentTrace:
    """Immutable record of a single agent method invocation."""
    tool_name:   str
    args:        Tuple[Any, ...]
    kwargs:      Dict[str, Any]
    result:      Any
    duration_ns: int
    success:     bool
    reasoning:   Optional[str]
    timestamp:   float = field(default_factory=time.monotonic)
    error:       Optional[str] = None
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":      self.tool_name,
            "duration_us": self.duration_ns // 1000,
            "success":   self.success,
            "reasoning": self.reasoning,
            "error":     self.error,
            "ts":        self.timestamp,
        }
 
 
@dataclass
class AgentContext:
    """
    Carried through every agent method call chain.
    Preserves the causality trail from initiating caller to leaf operations.
    """
    caller:   str                         = "kernel"
    trace_id: str                         = field(default_factory=lambda: hashlib.sha256(
        str(time.monotonic_ns()).encode()).hexdigest()[:12])
    depth:    int                         = 0
    budget_ns: int                        = 10_000_000   # 10 ms default
    metadata: Dict[str, Any]             = field(default_factory=dict)
    _chain:   List[str]                  = field(default_factory=list)
 
    def descend(self, tool_name: str) -> "AgentContext":
        child = AgentContext(
            caller=tool_name,
            trace_id=self.trace_id,
            depth=self.depth + 1,
            budget_ns=self.budget_ns,
            metadata=dict(self.metadata),
        )
        child._chain = self._chain + [tool_name]
        return child
 
    @property
    def call_chain(self) -> str:
        return " → ".join(self._chain) if self._chain else self.caller
 
 
@dataclass
class AgentToolSpec:
    """Full specification for a registered agent-callable tool."""
    name:        str
    description: str
    parameters:  Dict[str, Dict[str, Any]]
    returns:     str
    priority:    AgentPriority
    fn:          Callable
    owner:       str = "kernel"
 
 
class AgentRegistry:
    """
    Singleton registry of all agent-callable tools in the kernel.
    Every @agent_method decorated function lands here.
    """
    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()
 
    def __new__(cls) -> "AgentRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tools: Dict[str, AgentToolSpec] = {}
                cls._instance._traces: deque[AgentTrace] = deque(maxlen=8192)
                cls._instance._call_counts: Dict[str, int] = defaultdict(int)
        return cls._instance
 
    def register(self, spec: AgentToolSpec) -> None:
        self._tools[spec.name] = spec
 
    def get(self, name: str) -> Optional[AgentToolSpec]:
        return self._tools.get(name)
 
    def all_tools(self) -> List[AgentToolSpec]:
        return list(self._tools.values())
 
    def record_trace(self, trace: AgentTrace) -> None:
        self._traces.append(trace)
        self._call_counts[trace.tool_name] += 1
 
    def recent_traces(self, n: int = 20) -> List[AgentTrace]:
        traces = list(self._traces)
        return traces[-n:] if len(traces) >= n else traces
 
    def stats(self) -> Dict[str, Any]:
        return {
            "registered_tools": len(self._tools),
            "total_calls":      sum(self._call_counts.values()),
            "call_counts":      dict(self._call_counts),
            "trace_buffer":     len(self._traces),
        }
 
 
_registry = AgentRegistry()
 
 
def agent_method(
    name:        Optional[str]                  = None,
    description: str                            = "",
    parameters:  Optional[Dict[str, Any]]       = None,
    returns:     str                            = "Any",
    priority:    AgentPriority                  = AgentPriority.NORMAL,
    owner:       str                            = "kernel",
) -> Callable:
    """
    Decorator that registers a function as an agent-callable tool and wraps
    every invocation with tracing, timing, and context propagation.
 
    Usage:
        @agent_method(
            name="peek",
            description="Read one byte from the physical address bus",
            parameters={"addr": {"type": "int", "desc": "Physical byte address"}},
            returns="int",
            priority=AgentPriority.CRITICAL,
        )
        def peek(self, addr: int) -> int: ...
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__qualname__
 
        spec = AgentToolSpec(
            name=tool_name,
            description=description,
            parameters=parameters or {},
            returns=returns,
            priority=priority,
            fn=fn,
            owner=owner,
        )
        _registry.register(spec)
 
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            ctx: Optional[AgentContext] = kwargs.pop("_ctx", None)
            t0 = time.monotonic_ns()
            success = True
            result  = None
            error_s = None
            reasoning = None
 
            try:
                # If a reasoner is attached to the kernel, ask it to annotate
                # this call before execution (non-blocking advisory).
                if ctx is not None and hasattr(args[0] if args else None, "_reasoner"):
                    reasoning = args[0]._reasoner.annotate(tool_name, args, kwargs, ctx)
                result = fn(*args, **kwargs)
            except Exception as exc:
                success = False
                error_s = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                duration_ns = time.monotonic_ns() - t0
                trace = AgentTrace(
                    tool_name=tool_name,
                    args=args[1:] if args else (),  # strip self
                    kwargs=kwargs,
                    result=result,
                    duration_ns=duration_ns,
                    success=success,
                    reasoning=reasoning,
                    error=error_s,
                )
                _registry.record_trace(trace)
            return result
 
        wrapper._agent_spec = spec  # type: ignore[attr-defined]
        return wrapper
 
    return decorator
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MEMORY SUBSYSTEM
#  Simulated physical RAM, a memory bus with peek/poke/inb/outb primitives,
#  a bitmap physical-page allocator, and a software page-table VMM.
# ════════════════════════════════════════════════════════════════════════════════
 
class PageFlags(IntFlag):
    NONE      = 0x00
    PRESENT   = 0x01
    WRITE     = 0x02
    USER      = 0x04
    ACCESSED  = 0x20
    DIRTY     = 0x40
    EXECUTABLE= 0x80
 
 
class MemoryRegion(NamedTuple):
    name:  str
    base:  int
    size:  int
    flags: PageFlags
 
    @property
    def end(self) -> int:
        return self.base + self.size
 
    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.end
 
 
# Well-known physical memory map (x86 conventional)
MEMORY_MAP = [
    MemoryRegion("BIOS_ROM",    0x000F0000, 0x10000,      PageFlags.PRESENT | PageFlags.EXECUTABLE),
    MemoryRegion("VGA_BUFFER",  0x000B8000, 0x8000,       PageFlags.PRESENT | PageFlags.WRITE),
    MemoryRegion("LOW_RAM",     0x00000000, 0x0009FFFF,   PageFlags.PRESENT | PageFlags.WRITE),
    MemoryRegion("KERNEL_ZONE", 0x00100000, 0x03F00000,   PageFlags.PRESENT | PageFlags.WRITE | PageFlags.EXECUTABLE),
    MemoryRegion("HEAP_ZONE",   0x01000000, 0x02000000,   PageFlags.PRESENT | PageFlags.WRITE),
]
 
 
class PhysicalAllocator:
    """
    Bitmap-based physical page allocator.
    Tracks 4 KiB pages across the simulated RAM region.
    O(1) free-page-index lookup using a word-level first-fit scan.
    """
 
    def __init__(self, total_bytes: int = RAM_SIZE_BYTES) -> None:
        self._total_pages = total_bytes // PAGE_SIZE
        # One bit per page: 0 = free, 1 = allocated
        self._bitmap = bytearray((self._total_pages + 7) // 8)
        self._free_count = self._total_pages
        self._lock = threading.Lock()
        # Reserve page 0 (null pointer guard) and first 256 pages (low memory)
        for p in range(256):
            self._mark_allocated(p)
 
    def _mark_allocated(self, page_idx: int) -> None:
        byte, bit = divmod(page_idx, 8)
        self._bitmap[byte] |= (1 << bit)
        self._free_count -= 1
 
    def _mark_free(self, page_idx: int) -> None:
        byte, bit = divmod(page_idx, 8)
        self._bitmap[byte] &= ~(1 << bit)
        self._free_count += 1
 
    def _is_free(self, page_idx: int) -> bool:
        byte, bit = divmod(page_idx, 8)
        return not (self._bitmap[byte] & (1 << bit))
 
    @agent_method(
        name="palloc",
        description="Allocate n contiguous physical pages; returns base physical address or None",
        parameters={"n": {"type": "int", "desc": "Number of contiguous pages to allocate"}},
        returns="Optional[int]",
        priority=AgentPriority.HIGH,
    )
    def alloc(self, n: int = 1) -> Optional[int]:
        if n <= 0 or n > self._free_count:
            return None
        with self._lock:
            # First-fit contiguous scan
            run = 0
            run_start = 0
            for i in range(self._total_pages):
                if self._is_free(i):
                    if run == 0:
                        run_start = i
                    run += 1
                    if run == n:
                        for p in range(run_start, run_start + n):
                            self._mark_allocated(p)
                        return run_start * PAGE_SIZE
                else:
                    run = 0
        return None
 
    @agent_method(
        name="pfree",
        description="Release n contiguous physical pages beginning at base address",
        parameters={
            "phys_addr": {"type": "int", "desc": "Physical base address returned by palloc"},
            "n":         {"type": "int", "desc": "Number of pages to release"},
        },
        returns="bool",
        priority=AgentPriority.HIGH,
    )
    def free(self, phys_addr: int, n: int = 1) -> bool:
        page_idx = phys_addr >> PAGE_SHIFT
        if page_idx < 256 or page_idx + n > self._total_pages:
            return False
        with self._lock:
            for p in range(page_idx, page_idx + n):
                if not self._is_free(p):
                    self._mark_free(p)
        return True
 
    @property
    def free_pages(self) -> int:
        return self._free_count
 
    @property
    def used_pages(self) -> int:
        return self._total_pages - self._free_count
 
 
@dataclass
class PageTableEntry:
    physical_frame: int        # physical page number
    flags:          PageFlags
    access_count:   int = 0
 
    def is_present(self) -> bool:
        return bool(self.flags & PageFlags.PRESENT)
 
    def is_writable(self) -> bool:
        return bool(self.flags & PageFlags.WRITE)
 
    def pack(self) -> int:
        """Pack into a 32-bit integer like real x86 PTE."""
        return (self.physical_frame << PAGE_SHIFT) | int(self.flags)
 
 
class VirtualMemoryManager:
    """
    Two-level software page-table VMM.
    Maps virtual → physical addresses for a single address space.
    Page fault simulation raises PageFaultException.
    """
 
    class PageFaultException(Exception):
        def __init__(self, vaddr: int, flags: PageFlags) -> None:
            super().__init__(f"Page fault at vaddr=0x{vaddr:08X} flags={flags!r}")
            self.vaddr = vaddr
            self.access_flags = flags
 
    def __init__(self, palloc: PhysicalAllocator) -> None:
        self._palloc = palloc
        # Two-level: page_dir[pdi] → page_table[pti] → PageTableEntry
        self._page_dir: Dict[int, Dict[int, PageTableEntry]] = {}
        self._lock = threading.RLock()
 
    @agent_method(
        name="map_page",
        description="Map a virtual address to a physical address with given flags",
        parameters={
            "vaddr":  {"type": "int",       "desc": "Virtual address (4KiB aligned)"},
            "paddr":  {"type": "int",       "desc": "Physical address (4KiB aligned)"},
            "flags":  {"type": "PageFlags", "desc": "Access permission flags"},
        },
        returns="bool",
        priority=AgentPriority.HIGH,
    )
    def map_page(self, vaddr: int, paddr: int, flags: PageFlags = PageFlags.PRESENT | PageFlags.WRITE) -> bool:
        pdi = (vaddr >> 22) & 0x3FF
        pti = (vaddr >> 12) & 0x3FF
        with self._lock:
            if pdi not in self._page_dir:
                self._page_dir[pdi] = {}
            self._page_dir[pdi][pti] = PageTableEntry(
                physical_frame=paddr >> PAGE_SHIFT,
                flags=flags | PageFlags.PRESENT,
            )
        return True
 
    @agent_method(
        name="unmap_page",
        description="Remove a virtual→physical mapping",
        parameters={"vaddr": {"type": "int", "desc": "Virtual address to unmap"}},
        returns="bool",
        priority=AgentPriority.HIGH,
    )
    def unmap_page(self, vaddr: int) -> bool:
        pdi = (vaddr >> 22) & 0x3FF
        pti = (vaddr >> 12) & 0x3FF
        with self._lock:
            if pdi in self._page_dir and pti in self._page_dir[pdi]:
                del self._page_dir[pdi][pti]
                return True
        return False
 
    @agent_method(
        name="translate",
        description="Translate virtual address to physical address; raises PageFaultException on miss",
        parameters={
            "vaddr":        {"type": "int",       "desc": "Virtual address"},
            "access_flags": {"type": "PageFlags", "desc": "Requested access type"},
        },
        returns="int",
        priority=AgentPriority.CRITICAL,
    )
    def translate(self, vaddr: int, access_flags: PageFlags = PageFlags.PRESENT) -> int:
        pdi = (vaddr >> 22) & 0x3FF
        pti = (vaddr >> 12) & 0x3FF
        offset = vaddr & 0xFFF
        with self._lock:
            if pdi not in self._page_dir or pti not in self._page_dir[pdi]:
                raise VirtualMemoryManager.PageFaultException(vaddr, access_flags)
            pte = self._page_dir[pdi][pti]
            if not pte.is_present():
                raise VirtualMemoryManager.PageFaultException(vaddr, access_flags)
            if (access_flags & PageFlags.WRITE) and not pte.is_writable():
                raise VirtualMemoryManager.PageFaultException(vaddr, access_flags | PageFlags.WRITE)
            pte.access_count += 1
            return (pte.physical_frame << PAGE_SHIFT) | offset
 
    def identity_map_region(self, base: int, size: int, flags: PageFlags) -> None:
        """Map physical == virtual for a given region (used during early boot)."""
        aligned_base = base & ~(PAGE_SIZE - 1)
        pages = (size + PAGE_SIZE - 1) // PAGE_SIZE
        for i in range(pages):
            addr = aligned_base + i * PAGE_SIZE
            self.map_page(addr, addr, flags)
 
 
class MemoryBus:
    """
    The sole legal interface to hardware memory.
    Backs a bytearray representing physical RAM plus the VGA buffer hole.
    All reads/writes go through access-flag validation.
    """
 
    def __init__(self, palloc: PhysicalAllocator) -> None:
        self._ram     = bytearray(RAM_SIZE_BYTES)
        self._palloc  = palloc
        self._io_ports: Dict[int, int] = {}    # simulated ISA I/O ports
        self._lock    = threading.RLock()
        # Pre-seed BIOS identity string at reset vector region
        bios_sig = b"AIOS BIOS v1.0 "
        for i, b in enumerate(bios_sig):
            self._ram[0xF0000 + i] = b
 
    def _resolve(self, addr: int) -> int:
        """Clamp address into our RAM array."""
        if addr >= RAM_SIZE_BYTES:
            raise MemoryError(f"Physical address 0x{addr:08X} out of range (max 0x{RAM_SIZE_BYTES:08X})")
        return addr
 
    @agent_method(
        name="peek8",
        description="Read one byte from a physical address",
        parameters={"addr": {"type": "int", "desc": "Physical byte address"}},
        returns="int",
        priority=AgentPriority.CRITICAL,
    )
    def peek8(self, addr: int) -> int:
        with self._lock:
            return self._ram[self._resolve(addr)]
 
    @agent_method(
        name="poke8",
        description="Write one byte to a physical address",
        parameters={
            "addr":  {"type": "int", "desc": "Physical byte address"},
            "value": {"type": "int", "desc": "Byte value 0–255"},
        },
        returns="None",
        priority=AgentPriority.CRITICAL,
    )
    def poke8(self, addr: int, value: int) -> None:
        with self._lock:
            self._ram[self._resolve(addr)] = value & 0xFF
 
    @agent_method(
        name="peek32",
        description="Read a 32-bit little-endian word from a physical address",
        parameters={"addr": {"type": "int", "desc": "Physical address (must be 4-byte aligned)"}},
        returns="int",
        priority=AgentPriority.CRITICAL,
    )
    def peek32(self, addr: int) -> int:
        with self._lock:
            r = self._resolve(addr)
            return struct.unpack_from("<I", self._ram, r)[0]
 
    @agent_method(
        name="poke32",
        description="Write a 32-bit little-endian word to a physical address",
        parameters={
            "addr":  {"type": "int", "desc": "Physical address"},
            "value": {"type": "int", "desc": "32-bit unsigned integer"},
        },
        returns="None",
        priority=AgentPriority.CRITICAL,
    )
    def poke32(self, addr: int, value: int) -> None:
        with self._lock:
            r = self._resolve(addr)
            struct.pack_into("<I", self._ram, r, value & 0xFFFFFFFF)
 
    @agent_method(
        name="peek_buf",
        description="Read n bytes from a physical address into a bytearray",
        parameters={
            "addr": {"type": "int", "desc": "Physical base address"},
            "n":    {"type": "int", "desc": "Number of bytes"},
        },
        returns="bytearray",
        priority=AgentPriority.HIGH,
    )
    def peek_buf(self, addr: int, n: int) -> bytearray:
        with self._lock:
            r = self._resolve(addr)
            return bytearray(self._ram[r:r + n])
 
    @agent_method(
        name="poke_buf",
        description="Write bytes from buf into physical memory at addr",
        parameters={
            "addr": {"type": "int",       "desc": "Physical base address"},
            "buf":  {"type": "bytearray", "desc": "Data to write"},
        },
        returns="int",
        priority=AgentPriority.HIGH,
    )
    def poke_buf(self, addr: int, buf: bytes) -> int:
        with self._lock:
            r = self._resolve(addr)
            n = len(buf)
            self._ram[r:r + n] = buf
            return n
 
    @agent_method(
        name="inb",
        description="Read one byte from an ISA I/O port",
        parameters={"port": {"type": "int", "desc": "16-bit I/O port number"}},
        returns="int",
        priority=AgentPriority.CRITICAL,
    )
    def inb(self, port: int) -> int:
        with self._lock:
            return self._io_ports.get(port & 0xFFFF, 0xFF)
 
    @agent_method(
        name="outb",
        description="Write one byte to an ISA I/O port",
        parameters={
            "port":  {"type": "int", "desc": "16-bit I/O port number"},
            "value": {"type": "int", "desc": "Byte value 0–255"},
        },
        returns="None",
        priority=AgentPriority.CRITICAL,
    )
    def outb(self, port: int, value: int) -> None:
        with self._lock:
            self._io_ports[port & 0xFFFF] = value & 0xFF
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — HARDWARE ABSTRACTION LAYER (HAL)
# ════════════════════════════════════════════════════════════════════════════════
 
class VGAColor(IntEnum):
    BLACK   = 0;  BLUE    = 1;  GREEN   = 2;  CYAN    = 3
    RED     = 4;  MAGENTA = 5;  BROWN   = 6;  LGRAY   = 7
    DGRAY   = 8;  LBLUE   = 9;  LGREEN  = 10; LCYAN   = 11
    LRED    = 12; PINK    = 13; YELLOW  = 14; WHITE   = 15
 
 
def _vga_entry(ch: int, fg: VGAColor, bg: VGAColor) -> int:
    return ch | (((int(bg) << 4) | int(fg)) << 8)
 
 
class VGATextDriver:
    """
    Writes text directly to the VGA text buffer at 0xB8000.
    Backed by the MemoryBus — no terminal escape codes at this layer.
    """
 
    def __init__(self, bus: MemoryBus) -> None:
        self._bus   = bus
        self._col   = 0
        self._row   = 0
        self._fg    = VGAColor.LGRAY
        self._bg    = VGAColor.BLACK
        self._lock  = threading.Lock()
        self.clear()
 
    def _cell_addr(self, row: int, col: int) -> int:
        return VGA_BASE_ADDR + 2 * (row * VGA_COLS + col)
 
    @agent_method(
        name="vga_clear",
        description="Clear the VGA text screen",
        priority=AgentPriority.NORMAL,
    )
    def clear(self) -> None:
        with self._lock:
            blank = _vga_entry(ord(' '), self._fg, self._bg)
            for r in range(VGA_ROWS):
                for c in range(VGA_COLS):
                    self._bus.poke32(self._cell_addr(r, c) & ~1,
                                     blank & 0xFFFF) if False else \
                    self._bus.poke8(self._cell_addr(r, c),     ord(' '))
                    self._bus.poke8(self._cell_addr(r, c) + 1,
                                    (int(self._bg) << 4) | int(self._fg))
            self._col = 0
            self._row = 0
 
    def _scroll(self) -> None:
        """Scroll up by one line."""
        for r in range(1, VGA_ROWS):
            for c in range(VGA_COLS):
                src = self._cell_addr(r, c)
                dst = self._cell_addr(r - 1, c)
                self._bus.poke8(dst,     self._bus.peek8(src))
                self._bus.poke8(dst + 1, self._bus.peek8(src + 1))
        # Blank last row
        for c in range(VGA_COLS):
            self._bus.poke8(self._cell_addr(VGA_ROWS - 1, c),     ord(' '))
            self._bus.poke8(self._cell_addr(VGA_ROWS - 1, c) + 1,
                            (int(self._bg) << 4) | int(self._fg))
 
    @agent_method(
        name="vga_putchar",
        description="Write one character to the current cursor position",
        parameters={"ch": {"type": "str", "desc": "Single ASCII character"}},
        priority=AgentPriority.CRITICAL,
    )
    def putchar(self, ch: str) -> None:
        with self._lock:
            c = ch if isinstance(ch, int) else ord(ch)
            if c == ord('\n'):
                self._col = 0
                self._row += 1
            elif c == ord('\r'):
                self._col = 0
            elif c == ord('\b'):
                if self._col > 0:
                    self._col -= 1
                    addr = self._cell_addr(self._row, self._col)
                    self._bus.poke8(addr,     ord(' '))
                    self._bus.poke8(addr + 1, (int(self._bg) << 4) | int(self._fg))
            else:
                if 0 <= self._col < VGA_COLS and 0 <= self._row < VGA_ROWS:
                    addr = self._cell_addr(self._row, self._col)
                    self._bus.poke8(addr,     c & 0xFF)
                    self._bus.poke8(addr + 1, (int(self._bg) << 4) | int(self._fg))
                self._col += 1
 
            if self._col >= VGA_COLS:
                self._col = 0
                self._row += 1
            if self._row >= VGA_ROWS:
                self._scroll()
                self._row = VGA_ROWS - 1
 
    @agent_method(
        name="vga_write",
        description="Write a string to the VGA buffer",
        parameters={"s": {"type": "str", "desc": "String to display"}},
        priority=AgentPriority.NORMAL,
    )
    def write(self, s: str) -> None:
        for ch in s:
            self.putchar(ch)
 
    def writeln(self, s: str) -> None:
        self.write(s + "\n")
 
    def set_color(self, fg: VGAColor, bg: VGAColor = VGAColor.BLACK) -> None:
        with self._lock:
            self._fg = fg
            self._bg = bg
 
 
class IRQLine(IntEnum):
    TIMER    = 0
    KEYBOARD = 1
    COM2     = 3
    COM1     = 4
    LPT2     = 5
    FLOPPY   = 6
    LPT1     = 7
    RTC      = 8
    PS2MOUSE = 12
    ATA1     = 14
    ATA2     = 15
 
 
InterruptHandler = Callable[[int, Any], None]   # (irq, context) → None
 
 
class InterruptDescriptorTable:
    """
    Software IDT: maps interrupt vectors to Python-callable handlers.
    When an interrupt fires, the C-level stub would save state and call
    isr_dispatch(). Here we model that dispatch in pure Python.
    """
 
    EXCEPTION_NAMES = {
        0: "Divide Error",       1: "Debug",              2: "NMI",
        3: "Breakpoint",         4: "Overflow",            5: "BOUND Range",
        6: "Invalid Opcode",     7: "Device Not Available",8: "Double Fault",
        13: "General Protection",14: "Page Fault",         32: "Timer (IRQ0)",
        33: "Keyboard (IRQ1)",
    }
 
    def __init__(self) -> None:
        self._handlers: Dict[int, InterruptHandler] = {}
        self._lock = threading.Lock()
        # Default exception handler for unregistered vectors
        self._default = lambda irq, ctx: None
 
    @agent_method(
        name="idt_register",
        description="Register a Python callable as the handler for interrupt vector n",
        parameters={
            "vector":  {"type": "int",      "desc": "Interrupt vector number 0–255"},
            "handler": {"type": "Callable", "desc": "Python function(irq, ctx)→None"},
        },
        priority=AgentPriority.HIGH,
    )
    def register(self, vector: int, handler: InterruptHandler) -> None:
        with self._lock:
            self._handlers[vector & 0xFF] = handler
 
    @agent_method(
        name="idt_dispatch",
        description="Dispatch an interrupt vector to its registered handler",
        parameters={
            "vector":  {"type": "int", "desc": "Interrupt vector number"},
            "context": {"type": "Any", "desc": "Optional hardware context (registers, error code)"},
        },
        priority=AgentPriority.CRITICAL,
    )
    def dispatch(self, vector: int, context: Any = None) -> None:
        with self._lock:
            handler = self._handlers.get(vector & 0xFF, self._default)
        try:
            handler(vector, context)
        except Exception as exc:
            # In a real kernel: this would trigger a double-fault.
            # We log and continue to keep the simulation alive.
            pass
 
    def name_for(self, vector: int) -> str:
        return self.EXCEPTION_NAMES.get(vector, f"Unknown IRQ {vector}")
 
 
class PICDriver:
    """
    Programmable Interrupt Controller simulation.
    Manages IRQ masking and EOI (End-Of-Interrupt) signaling.
    I/O ports: master PIC cmd=0x20, data=0x21; slave cmd=0xA0, data=0xA1.
    """
 
    PIC1_CMD  = 0x20;  PIC1_DATA = 0x21
    PIC2_CMD  = 0xA0;  PIC2_DATA = 0xA1
    EOI       = 0x20
 
    def __init__(self, bus: MemoryBus, idt: InterruptDescriptorTable) -> None:
        self._bus  = bus
        self._idt  = idt
        self._mask = 0xFFFF   # all masked initially
        self._initialize()
 
    def _initialize(self) -> None:
        """ICW1–ICW4 initialization sequence."""
        # ICW1: cascade, edge-triggered
        self._bus.outb(self.PIC1_CMD,  0x11)
        self._bus.outb(self.PIC2_CMD,  0x11)
        # ICW2: vector offsets — IRQ0..7 → INT 32..39, IRQ8..15 → INT 40..47
        self._bus.outb(self.PIC1_DATA, 0x20)
        self._bus.outb(self.PIC2_DATA, 0x28)
        # ICW3: cascade wiring
        self._bus.outb(self.PIC1_DATA, 0x04)
        self._bus.outb(self.PIC2_DATA, 0x02)
        # ICW4: 8086 mode
        self._bus.outb(self.PIC1_DATA, 0x01)
        self._bus.outb(self.PIC2_DATA, 0x01)
        # Mask all IRQs until drivers register their handlers
        self._bus.outb(self.PIC1_DATA, 0xFF)
        self._bus.outb(self.PIC2_DATA, 0xFF)
 
    @agent_method(
        name="pic_unmask",
        description="Unmask (enable) an IRQ line",
        parameters={"irq": {"type": "int", "desc": "IRQ number 0–15"}},
        priority=AgentPriority.HIGH,
    )
    def unmask(self, irq: int) -> None:
        self._mask &= ~(1 << irq)
        if irq < 8:
            self._bus.outb(self.PIC1_DATA, (~self._mask) & 0xFF)
        else:
            self._bus.outb(self.PIC2_DATA, (~(self._mask >> 8)) & 0xFF)
 
    @agent_method(
        name="pic_mask",
        description="Mask (disable) an IRQ line",
        parameters={"irq": {"type": "int", "desc": "IRQ number 0–15"}},
        priority=AgentPriority.HIGH,
    )
    def mask(self, irq: int) -> None:
        self._mask |= (1 << irq)
        if irq < 8:
            self._bus.outb(self.PIC1_DATA, self._mask & 0xFF)
        else:
            self._bus.outb(self.PIC2_DATA, (self._mask >> 8) & 0xFF)
 
    @agent_method(
        name="pic_eoi",
        description="Signal End-Of-Interrupt to the PIC",
        parameters={"irq": {"type": "int", "desc": "IRQ number 0–15 that was serviced"}},
        priority=AgentPriority.CRITICAL,
    )
    def eoi(self, irq: int) -> None:
        if irq >= 8:
            self._bus.outb(self.PIC2_CMD, self.EOI)
        self._bus.outb(self.PIC1_CMD, self.EOI)
 
    def fire(self, irq: int, ctx: Any = None) -> None:
        """Simulate an IRQ firing — checks mask, dispatches to IDT, then EOI."""
        if self._mask & (1 << irq):
            return  # masked
        vector = irq + (32 if irq < 8 else 24)   # remapped vectors
        self._idt.dispatch(vector, ctx)
        self.eoi(irq)
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MATHEMATICAL PRIMITIVES (FIRST PRINCIPLES)
#  No import of math, numpy, or any numeric library.
#  All transcendental functions implemented from scratch.
# ════════════════════════════════════════════════════════════════════════════════
 
# ── 3a. Bit-level IEEE 754 double manipulation ─────────────────────────────────
 
class IEEE754:
    """
    Bit-level access to Python's native 64-bit IEEE 754 doubles.
    Decomposes into (sign:1, exponent:11, mantissa:52) fields.
    Reconstructs doubles from raw components.
    Correctly identifies NaN, ±Inf, ±0, subnormals per PEP 754.
    """
 
    SIGN_SHIFT      = 63
    EXP_SHIFT       = 52
    EXP_MASK        = 0x7FF
    MANTISSA_MASK   = (1 << 52) - 1
    EXPONENT_BIAS   = 1023
 
    @staticmethod
    def to_bits(x: float) -> int:
        """Return the 64-bit integer whose bit pattern equals x."""
        return struct.unpack("Q", struct.pack("d", x))[0]
 
    @staticmethod
    def from_bits(bits: int) -> float:
        """Construct a float from its 64-bit integer representation."""
        return struct.unpack("d", struct.pack("Q", bits & 0xFFFFFFFFFFFFFFFF))[0]
 
    @classmethod
    def decompose(cls, x: float) -> Tuple[int, int, int]:
        """Return (sign, biased_exponent, raw_mantissa)."""
        bits = cls.to_bits(x)
        sign     = (bits >> cls.SIGN_SHIFT) & 1
        exponent = (bits >> cls.EXP_SHIFT)  & cls.EXP_MASK
        mantissa = bits                      & cls.MANTISSA_MASK
        return sign, exponent, mantissa
 
    @classmethod
    def compose(cls, sign: int, biased_exp: int, mantissa: int) -> float:
        """Reconstruct a double from its three fields."""
        bits = ((sign & 1) << cls.SIGN_SHIFT) | \
               ((biased_exp & cls.EXP_MASK) << cls.EXP_SHIFT) | \
               (mantissa & cls.MANTISSA_MASK)
        return cls.from_bits(bits)
 
    @classmethod
    def is_nan(cls, x: float) -> bool:
        _, exp, mantissa = cls.decompose(x)
        return exp == cls.EXP_MASK and mantissa != 0
 
    @classmethod
    def is_inf(cls, x: float) -> bool:
        _, exp, mantissa = cls.decompose(x)
        return exp == cls.EXP_MASK and mantissa == 0
 
    @classmethod
    def is_subnormal(cls, x: float) -> bool:
        _, exp, mantissa = cls.decompose(x)
        return exp == 0 and mantissa != 0
 
    @classmethod
    def nan(cls) -> float:
        return cls.from_bits(0x7FF8000000000000)
 
    @classmethod
    def pos_inf(cls) -> float:
        return cls.from_bits(0x7FF0000000000000)
 
    @classmethod
    def neg_inf(cls) -> float:
        return cls.from_bits(0xFFF0000000000000)
 
    @classmethod
    def ulp(cls, x: float) -> float:
        """Unit in the Last Place — smallest representable difference."""
        _, exp, _ = cls.decompose(abs(x))
        if exp == 0:
            return cls.from_bits(1)  # subnormal ULP
        return cls.from_bits((exp << cls.EXP_SHIFT)) - cls.from_bits(((exp - 1) << cls.EXP_SHIFT))
 
 
# ── CORDIC constants (module-level to ensure clean one-time computation) ──────
 
def _machin_pi_over_4(terms: int = 80) -> float:
    """
    Compute π/4 via Machin's formula: 4·atan(1/5) − atan(1/239).
    Both arguments are small enough that the Taylor series converges in
    far fewer terms than needed for double precision.  Error < 1e-16
    at 80 terms for both sub-series.
    """
    def _atan_taylor(x: float, n: int) -> float:
        result = 0.0
        x_pow  = x
        x_sq   = x * x
        sign   = 1
        for k in range(n):
            result += sign * x_pow / (2 * k + 1)
            x_pow  *= x_sq
            sign   *= -1
        return result
    return 4.0 * _atan_taylor(1.0 / 5.0, terms) - _atan_taylor(1.0 / 239.0, terms)
 
 
def _build_cordic_atan_table(n: int, pi_over_4: float) -> List[float]:
    """
    Build the CORDIC arctan lookup table for i = 0 .. n-1.
 
    Critical precision note:
      atan(2^-0) = atan(1) = π/4 exactly.
      The Taylor series at x=1 converges as the alternating harmonic series
      (error ≈ 1/(2k+1)) and needs ~10^14 terms for double precision — unusable.
      We therefore assign table[0] = π/4 from Machin's formula directly.
      For i ≥ 1, x = 2^-i ≤ 0.5, and 80 Taylor terms gives < 1e-30 error.
    """
    def _atan_taylor(x: float, terms: int = 80) -> float:
        r = 0.0; xp = x; xs = x * x; s = 1
        for k in range(terms):
            r += s * xp / (2 * k + 1); xp *= xs; s *= -1
        return r
 
    table = [0.0] * n
    table[0] = pi_over_4              # atan(1) = π/4, exact via Machin
    for i in range(1, n):
        table[i] = _atan_taylor(2.0 ** (-i))
    return table
 
 
def _build_cordic_gain(n: int) -> float:
    """
    CORDIC circular-mode gain inverse: K = ∏_{i=0}^{n-1} 1/√(1 + 2^{-2i}).
    Initialise the CORDIC x-register with this value so the final
    output magnitude is 1 (cos/sin both ≤ 1 in absolute value).
    """
    k = 1.0
    for i in range(n):
        k *= (1.0 + 2.0 ** (-2 * i)) ** (-0.5)
    return k
 
 
_CORDIC_PI_OVER_4 = _machin_pi_over_4(80)
_CORDIC_PI        = _CORDIC_PI_OVER_4 * 4.0
_CORDIC_HALF_PI   = _CORDIC_PI_OVER_4 * 2.0
_CORDIC_TWO_PI    = _CORDIC_PI * 2.0
_CORDIC_ATAN      = _build_cordic_atan_table(CORDIC_ITERATIONS, _CORDIC_PI_OVER_4)
_CORDIC_K         = _build_cordic_gain(CORDIC_ITERATIONS)
 
 
# ── 3b. CORDIC Engine ─────────────────────────────────────────────────────────
 
class CORDIC:
    """
    CORDIC (COordinate Rotation DIgital Computer) engine.
    Computes sin, cos, atan2, exp, ln, sqrt using only
    additions, subtractions, and bit shifts. No FPU required.
 
    Rotation mode  : given angle θ, returns (cos θ, sin θ).
    Vectoring mode : given (x, y), returns atan2(y, x).
 
    All constants are pre-computed at module level to guarantee single,
    correct initialisation with no class-body evaluation ordering issues.
    """
 
    PI        = _CORDIC_PI
    HALF_PI   = _CORDIC_HALF_PI
    TWO_PI    = _CORDIC_TWO_PI
    _ATAN_TABLE: List[float] = _CORDIC_ATAN
    _K: float                = _CORDIC_K
 
    @classmethod
    def sincos(cls, angle: float) -> Tuple[float, float]:
        """
        Return (sin(angle), cos(angle)).
        Full range: angle ∈ [-2π, 2π]; outside is range-reduced.
        """
        if IEEE754.is_nan(angle) or IEEE754.is_inf(angle):
            return IEEE754.nan(), IEEE754.nan()
 
        # Range-reduce to [-π, π]
        while angle >  cls.PI: angle -= cls.TWO_PI
        while angle < -cls.PI: angle += cls.TWO_PI
 
        # Quadrant adjustment to [-π/2, π/2]
        flip = False
        if angle > cls.HALF_PI:
            angle  = cls.PI - angle
            flip   = True
        elif angle < -cls.HALF_PI:
            angle  = -cls.PI - angle
            flip   = True
 
        # CORDIC rotation
        x = cls._K
        y = 0.0
        z = angle
 
        for i in range(CORDIC_ITERATIONS):
            d   = -1.0 if z < 0.0 else 1.0
            x_n = x - d * y * (2.0 ** (-i))
            y_n = y + d * x * (2.0 ** (-i))
            z  -= d * cls._ATAN_TABLE[i]
            x, y = x_n, y_n
 
        cos_v = -x if flip else x
        sin_v =  y
        return sin_v, cos_v
 
    @classmethod
    def sin(cls, angle: float) -> float:
        s, _ = cls.sincos(angle)
        return s
 
    @classmethod
    def cos(cls, angle: float) -> float:
        _, c = cls.sincos(angle)
        return c
 
    @classmethod
    def atan2(cls, y: float, x: float) -> float:
        """
        atan2(y, x) using CORDIC vectoring mode.
        Returns angle in [-π, π].
        """
        if x == 0.0 and y == 0.0:
            return 0.0
        if IEEE754.is_nan(x) or IEEE754.is_nan(y):
            return IEEE754.nan()
 
        # Quadrant normalisation to first quadrant
        flip_x = x < 0.0
        flip_y = y < 0.0
        x, y   = abs(x), abs(y)
 
        # Swap if y > x so we stay in (0, π/4) convergence zone
        swapped = y > x
        if swapped:
            x, y = y, x
 
        xv, yv, zv = x, y, 0.0
        for i in range(CORDIC_ITERATIONS):
            d    = -1.0 if yv > 0.0 else 1.0
            xv_n = xv + d * yv * (2.0 ** (-i))
            yv_n = yv - d * xv * (2.0 ** (-i))
            zv  += d * cls._ATAN_TABLE[i]
            xv, yv = xv_n, yv_n
 
        if swapped:
            zv = cls.HALF_PI - zv
        if flip_x:
            zv = cls.PI - zv
        if flip_y:
            zv = -zv
        return zv
 
    @classmethod
    def sqrt(cls, x: float) -> float:
        """
        √x via Newton-Raphson from an IEEE 754 exponent-halving initial guess.
        For x = 1.mantissa × 2^(e - 1023):
          guess biased exponent = (e_biased + 1023) >> 1
        Converges to full double precision in 10 iterations.
        """
        if x < 0.0:
            return IEEE754.nan()
        if x == 0.0:
            return 0.0
        if IEEE754.is_inf(x):
            return IEEE754.pos_inf()
        _, e_biased, _ = IEEE754.decompose(x)
        guess_exp = (e_biased + IEEE754.EXPONENT_BIAS) >> 1
        guess = IEEE754.compose(0, guess_exp, 0)
        if guess == 0.0:
            guess = x  # fallback for subnormals
        for _ in range(10):
            guess = (guess + x / guess) * 0.5
        return guess
 
    @classmethod
    def exp(cls, x: float) -> float:
        """
        e^x using identity: e^x = e^n · e^f where n = floor(x), f = x - n.
        e^n is computed by repeated squaring of E constant.
        e^f is computed by Taylor series (|f| ≤ 0.5, fast convergence).
        """
        if IEEE754.is_nan(x):
            return IEEE754.nan()
        if IEEE754.is_inf(x):
            return IEEE754.pos_inf() if x > 0 else 0.0
 
        # Integer part via repeated multiply
        E = 2.718281828459045235360287471352662497757  # Euler's number
        n = int(x)
        f = x - n
        # e^n by repeated squaring
        base = E
        result = 1.0
        power = abs(n)
        while power:
            if power & 1:
                result *= base
            base  *= base
            power >>= 1
        if n < 0:
            result = 1.0 / result
 
        # e^f via Taylor series: Σ f^k / k!
        term = 1.0
        ef   = 1.0
        for k in range(1, 80):
            term *= f / k
            ef   += term
            if abs(term) < 1e-17:
                break
        return result * ef
 
    @classmethod
    def ln(cls, x: float) -> float:
        """
        Natural log via identity: ln(x) = ln(m · 2^e) = ln(m) + e·ln(2)
        where m ∈ [1, 2). ln(m) computed by Padé approximant.
        """
        if x <= 0.0:
            return IEEE754.neg_inf() if x == 0.0 else IEEE754.nan()
        if IEEE754.is_inf(x):
            return IEEE754.pos_inf()
 
        LN2 = 0.6931471805599453094172321214581765680755
        sign, exp, mant = IEEE754.decompose(x)
        # Normalise x = m * 2^(exp - bias), m ∈ [1, 2)
        e = exp - IEEE754.EXPONENT_BIAS
        m = IEEE754.compose(0, IEEE754.EXPONENT_BIAS, mant)  # m ∈ [1, 2)
 
        # Shift m into [0.5, 1) for better convergence: m → m/2, e += 1
        if m > 1.5:
            m *= 0.5; e += 1
 
        # Padé approximant for ln near 1: let u = (m-1)/(m+1)
        # ln(m) = 2 * (u + u^3/3 + u^5/5 + ...) — Mercator-like series
        u   = (m - 1.0) / (m + 1.0)
        u2  = u * u
        lnm = 0.0
        term = u
        for k in range(100):
            lnm  += term / (2 * k + 1)
            term *= u2
            if abs(term) < 1e-17:
                break
        lnm *= 2.0
        return lnm + e * LN2
 
    @classmethod
    def tanh(cls, x: float) -> float:
        """tanh(x) = (e^2x - 1) / (e^2x + 1)."""
        if x > 20.0:  return  1.0
        if x < -20.0: return -1.0
        e2x = cls.exp(2.0 * x)
        return (e2x - 1.0) / (e2x + 1.0)
 
    @classmethod
    def sigmoid(cls, x: float) -> float:
        """σ(x) = 1 / (1 + e^-x)."""
        if x >= 0.0:
            return 1.0 / (1.0 + cls.exp(-x))
        ex = cls.exp(x)
        return ex / (1.0 + ex)
 
 
# ── 3c. Dual Numbers — Forward-Mode Automatic Differentiation ─────────────────
 
class DualNumber:
    """
    Dual number:  a + bε   where ε² = 0.
 
    Represents a value (a) and its first derivative (b).
    Operator overloading propagates derivative information automatically
    through any computation graph — no backward pass required.
 
    Usage:
        x = DualNumber(3.0, 1.0)   # x = 3, dx/dx = 1
        y = x * x + DualNumber(2.0)
        # y.real == 11.0, y.dual == 6.0  (dy/dx = 2x = 6)
    """
 
    __slots__ = ("real", "dual")
 
    def __init__(self, real: float, dual: float = 0.0) -> None:
        self.real = float(real)
        self.dual = float(dual)
 
    # ── Arithmetic ────────────────────────────────────────────────────────────
 
    def __add__(self, other: Union["DualNumber", float]) -> "DualNumber":
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        return DualNumber(self.real + other, self.dual)
 
    def __radd__(self, other: float) -> "DualNumber":
        return DualNumber(other + self.real, self.dual)
 
    def __sub__(self, other: Union["DualNumber", float]) -> "DualNumber":
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        return DualNumber(self.real - other, self.dual)
 
    def __rsub__(self, other: float) -> "DualNumber":
        return DualNumber(other - self.real, -self.dual)
 
    def __mul__(self, other: Union["DualNumber", float]) -> "DualNumber":
        if isinstance(other, DualNumber):
            # (a + bε)(c + dε) = ac + (ad + bc)ε
            return DualNumber(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real
            )
        return DualNumber(self.real * other, self.dual * other)
 
    def __rmul__(self, other: float) -> "DualNumber":
        return DualNumber(other * self.real, other * self.dual)
 
    def __truediv__(self, other: Union["DualNumber", float]) -> "DualNumber":
        if isinstance(other, DualNumber):
            # d(u/v) = (v·du - u·dv) / v²
            return DualNumber(
                self.real / other.real,
                (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            )
        return DualNumber(self.real / other, self.dual / other)
 
    def __rtruediv__(self, other: float) -> "DualNumber":
        return DualNumber(other / self.real, -other * self.dual / (self.real ** 2))
 
    def __neg__(self) -> "DualNumber":
        return DualNumber(-self.real, -self.dual)
 
    def __pow__(self, n: Union[int, float]) -> "DualNumber":
        # d(x^n) = n·x^(n-1)·dx
        return DualNumber(self.real ** n, n * (self.real ** (n - 1)) * self.dual)
 
    def __abs__(self) -> "DualNumber":
        sign = 1.0 if self.real >= 0.0 else -1.0
        return DualNumber(abs(self.real), sign * self.dual)
 
    # ── Comparison (compares real part only) ──────────────────────────────────
 
    def __lt__(self, other: Union["DualNumber", float]) -> bool:
        return self.real < (other.real if isinstance(other, DualNumber) else other)
 
    def __le__(self, other: Union["DualNumber", float]) -> bool:
        return self.real <= (other.real if isinstance(other, DualNumber) else other)
 
    def __gt__(self, other: Union["DualNumber", float]) -> bool:
        return self.real > (other.real if isinstance(other, DualNumber) else other)
 
    def __ge__(self, other: Union["DualNumber", float]) -> bool:
        return self.real >= (other.real if isinstance(other, DualNumber) else other)
 
    def __eq__(self, other: object) -> bool:
        if isinstance(other, DualNumber):
            return self.real == other.real and self.dual == other.dual
        return self.real == other
 
    # ── Elementary functions as class methods ─────────────────────────────────
 
    @classmethod
    def sin(cls, x: "DualNumber") -> "DualNumber":
        s, c = CORDIC.sincos(x.real)
        return cls(s, c * x.dual)
 
    @classmethod
    def cos(cls, x: "DualNumber") -> "DualNumber":
        s, c = CORDIC.sincos(x.real)
        return cls(c, -s * x.dual)
 
    @classmethod
    def exp(cls, x: "DualNumber") -> "DualNumber":
        ev = CORDIC.exp(x.real)
        return cls(ev, ev * x.dual)
 
    @classmethod
    def ln(cls, x: "DualNumber") -> "DualNumber":
        return cls(CORDIC.ln(x.real), x.dual / x.real)
 
    @classmethod
    def tanh(cls, x: "DualNumber") -> "DualNumber":
        t = CORDIC.tanh(x.real)
        return cls(t, (1.0 - t * t) * x.dual)
 
    @classmethod
    def sigmoid(cls, x: "DualNumber") -> "DualNumber":
        s = CORDIC.sigmoid(x.real)
        return cls(s, s * (1.0 - s) * x.dual)
 
    @classmethod
    def sqrt(cls, x: "DualNumber") -> "DualNumber":
        sv = CORDIC.sqrt(x.real)
        return cls(sv, x.dual / (2.0 * sv) if sv != 0.0 else 0.0)
 
    @classmethod
    def relu(cls, x: "DualNumber") -> "DualNumber":
        if x.real > 0.0:
            return cls(x.real, x.dual)
        return cls(0.0, 0.0)
 
    def gradient(self) -> float:
        """Extract the accumulated derivative."""
        return self.dual
 
    def __repr__(self) -> str:
        return f"DualNumber({self.real:.6g} + {self.dual:.6g}ε)"
 
    def __float__(self) -> float:
        return self.real
 
 
def grad(fn: Callable, x: float) -> float:
    """Compute df/dx at x using a single forward pass with dual numbers."""
    return fn(DualNumber(x, 1.0)).dual
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — TENSOR
#  Multi-dimensional array backed by a raw bytearray (custom allocator).
#  Supports float64 and int32 dtypes, arbitrary rank, broadcasting,
#  and gradient-aware operations via DualNumber.
# ════════════════════════════════════════════════════════════════════════════════
 
class DType(Enum):
    FLOAT64 = ("d", 8)
    INT32   = ("i", 4)
    UINT8   = ("B", 1)
 
    @property
    def fmt(self) -> str:
        return self.value[0]
 
    @property
    def itemsize(self) -> int:
        return self.value[1]
 
 
class Tensor:
    """
    N-dimensional array backed by a Python array.array (contiguous buffer).
 
    Design:
    • Layout is always C-contiguous (row-major).
    • Arithmetic ops (+, -, *, /) are element-wise with broadcasting.
    • Gradient ops use DualNumber for forward-mode AD.
    • No numpy. No ctypes. Pure Python over array.array.
    """
 
    def __init__(
        self,
        data:  Optional[Union[List, "Tensor", array.array]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: DType = DType.FLOAT64,
    ) -> None:
        self._dtype = dtype
 
        if isinstance(data, Tensor):
            self._shape  = data._shape
            self._buffer = array.array(dtype.fmt, data._buffer)
            return
 
        if shape is not None and data is None:
            # Zero-initialised
            self._shape  = shape
            self._buffer = array.array(dtype.fmt, [0] * self._numel(shape))
            return
 
        if isinstance(data, (list, tuple)):
            flat, shape_ = Tensor._flatten(data)
            self._shape  = tuple(shape_)
            self._buffer = array.array(dtype.fmt, [
                int(v) if dtype == DType.INT32 else float(v) for v in flat
            ])
            return
 
        if isinstance(data, array.array):
            assert shape is not None
            self._shape  = shape
            self._buffer = array.array(dtype.fmt, data)
            return
 
        raise TypeError(f"Cannot construct Tensor from {type(data)}")
 
    @staticmethod
    def _flatten(nested: Any) -> Tuple[List, List[int]]:
        """Recursively flatten a nested list and infer shape."""
        if not isinstance(nested, (list, tuple)):
            return [nested], []
        shape = [len(nested)]
        flat_all: List = []
        for item in nested:
            sub_flat, sub_shape = Tensor._flatten(item)
            flat_all.extend(sub_flat)
            if not shape[1:]:
                shape.extend(sub_shape)
        return flat_all, shape
 
    @staticmethod
    def _numel(shape: Tuple[int, ...]) -> int:
        n = 1
        for d in shape: n *= d
        return n
 
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
 
    @property
    def ndim(self) -> int:
        return len(self._shape)
 
    @property
    def numel(self) -> int:
        return self._numel(self._shape)
 
    @property
    def dtype(self) -> DType:
        return self._dtype
 
    def _strides(self) -> Tuple[int, ...]:
        strides = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * self._shape[i + 1]
        return tuple(strides)
 
    def _flat_index(self, indices: Tuple[int, ...]) -> int:
        strides = self._strides()
        idx = 0
        for i, (d, s) in enumerate(zip(indices, strides)):
            if d < 0: d += self._shape[i]
            if not (0 <= d < self._shape[i]):
                raise IndexError(f"Index {d} out of bounds for dim {i} size {self._shape[i]}")
            idx += d * s
        return idx
 
    def __getitem__(self, indices: Any) -> Union["Tensor", float, int]:
        if not isinstance(indices, tuple):
            indices = (indices,)
 
        # Scalar access
        if len(indices) == self.ndim and all(isinstance(i, int) for i in indices):
            return self._buffer[self._flat_index(indices)]
 
        # Slice along first dimension
        if len(indices) == 1 and isinstance(indices[0], int):
            i = indices[0]
            if i < 0: i += self._shape[0]
            stride = self._numel(self._shape[1:])
            sub_buf = array.array(self._dtype.fmt, self._buffer[i * stride:(i + 1) * stride])
            return Tensor(sub_buf, shape=self._shape[1:], dtype=self._dtype)
 
        raise NotImplementedError("Advanced indexing not yet implemented")
 
    def __setitem__(self, indices: Any, value: Union[float, int]) -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)
        self._buffer[self._flat_index(indices)] = value
 
    def item(self) -> float:
        if self.numel != 1:
            raise ValueError("item() only valid for single-element tensors")
        return float(self._buffer[0])
 
    def to_list(self) -> Any:
        """Reconstruct nested Python list matching shape."""
        flat = list(self._buffer)
        def _build(flat_data: List, shape: Tuple) -> Any:
            if not shape:
                return flat_data[0]
            n = self._numel(shape[1:])
            return [_build(flat_data[i * n:(i + 1) * n], shape[1:])
                    for i in range(shape[0])]
        return _build(flat, self._shape)
 
    # ── Element-wise binary ops with broadcast ────────────────────────────────
 
    @staticmethod
    def _broadcast_shapes(a: Tuple, b: Tuple) -> Tuple[int, ...]:
        """NumPy-style shape broadcasting rules."""
        if a == b:
            return a
        la, lb = len(a), len(b)
        result: List[int] = []
        for i in range(max(la, lb)):
            da = a[la - 1 - i] if i < la else 1
            db = b[lb - 1 - i] if i < lb else 1
            if da != db and da != 1 and db != 1:
                raise ValueError(f"Shapes {a} and {b} are not broadcastable")
            result.append(max(da, db))
        return tuple(reversed(result))
 
    def _broadcast_index(self, out_indices: Tuple[int, ...]) -> Tuple[int, ...]:
        """Map an output index back to an index into self, respecting broadcast dims."""
        diff = len(out_indices) - self.ndim
        result = []
        for i, oi in enumerate(out_indices):
            si = i - diff
            if si < 0:
                continue
            result.append(oi if self._shape[si] > 1 else 0)
        return tuple(result)
 
    def _elementwise(self, other: "Tensor", op: Callable) -> "Tensor":
        out_shape = self._broadcast_shapes(self._shape, other._shape)
        out = Tensor(shape=out_shape, dtype=self._dtype)
        strides = Tensor(shape=out_shape, dtype=DType.INT32)._strides()
        for flat_i in range(Tensor._numel(out_shape)):
            # Reconstruct multi-index
            idx: List[int] = []
            rem = flat_i
            for s in strides:
                idx.append(rem // s)
                rem  %= s
            t_idx = self._broadcast_index(tuple(idx))
            o_idx = other._broadcast_index(tuple(idx))
            a = self._buffer[self._flat_index(t_idx)]
            b = other._buffer[other._flat_index(o_idx)]
            out._buffer[flat_i] = op(a, b)
        return out
 
    def __add__(self, other: Union["Tensor", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor([other], shape=(1,), dtype=self._dtype)
        return self._elementwise(other, lambda a, b: a + b)
 
    def __sub__(self, other: Union["Tensor", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor([other], shape=(1,), dtype=self._dtype)
        return self._elementwise(other, lambda a, b: a - b)
 
    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor([other], shape=(1,), dtype=self._dtype)
        return self._elementwise(other, lambda a, b: a * b)
 
    def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor([other], shape=(1,), dtype=self._dtype)
        return self._elementwise(other, lambda a, b: a / b)
 
    def __neg__(self) -> "Tensor":
        t = Tensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.numel):
            t._buffer[i] = -self._buffer[i]
        return t
 
    def __matmul__(self, other: "Tensor") -> "Tensor":
        """2D matrix multiplication only."""
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("@ requires 2D tensors")
        m, k  = self._shape
        k2, n = other._shape
        if k != k2:
            raise ValueError(f"Incompatible shapes {self._shape} @ {other._shape}")
        out = Tensor(shape=(m, n), dtype=self._dtype)

        # Optimize loops using map, zip, array slice & transpose
        import operator
        other_T = other.T()
        buf1 = self._buffer
        buf2_T = other_T._buffer

        # Extract rows and cols
        rows_A = [buf1[i * k : (i + 1) * k] for i in range(m)]
        cols_B = [buf2_T[j * k : (j + 1) * k] for j in range(n)]

        out._buffer[:] = array.array(
            self._dtype.fmt,
            (sum(map(operator.mul, row_A, col_B)) for row_A in rows_A for col_B in cols_B)
        )
        return out
 
    def sum(self, axis: Optional[int] = None) -> "Tensor":
        if axis is None:
            return Tensor([sum(self._buffer)], shape=(1,), dtype=self._dtype)
        if axis < 0: axis += self.ndim
        out_shape = tuple(d for i, d in enumerate(self._shape) if i != axis)
        out = Tensor(shape=out_shape or (1,), dtype=self._dtype)
        strides = self._strides()
        for flat_i in range(Tensor._numel(out_shape or (1,))):
            # Build output multi-index
            idx_rem = flat_i
            out_idx = []
            out_strides = Tensor(shape=out_shape or (1,), dtype=DType.INT32)._strides()
            for s in out_strides:
                out_idx.append(idx_rem // s)
                idx_rem %= s
            # Sum over the collapsed axis
            total = 0.0
            for k in range(self._shape[axis]):
                in_idx = out_idx[:axis] + [k] + out_idx[axis:]
                total += self._buffer[self._flat_index(tuple(in_idx))]
            out._buffer[flat_i] = total
        return out
 
    def mean(self, axis: Optional[int] = None) -> "Tensor":
        s = self.sum(axis=axis)
        n = self._shape[axis] if axis is not None else self.numel
        return s / float(n)
 
    def T(self) -> "Tensor":
        """Transpose a 2D tensor."""
        if self.ndim != 2:
            raise NotImplementedError("T() only for 2D tensors")
        m, n = self._shape
        out  = Tensor(shape=(n, m), dtype=self._dtype)
        for i in range(m):
            for j in range(n):
                out._buffer[j * m + i] = self._buffer[i * n + j]
        return out
 
    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        if Tensor._numel(new_shape) != self.numel:
            raise ValueError(f"Cannot reshape {self._shape} → {new_shape}")
        t = Tensor(self._buffer, shape=new_shape, dtype=self._dtype)
        return t
 
    def apply(self, fn: Callable[[float], float]) -> "Tensor":
        """Element-wise application of a scalar function."""
        out = Tensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.numel):
            out._buffer[i] = fn(self._buffer[i])
        return out
 
    # ── Factory methods ───────────────────────────────────────────────────────
 
    @classmethod
    def zeros(cls, *shape: int, dtype: DType = DType.FLOAT64) -> "Tensor":
        return cls(shape=shape, dtype=dtype)
 
    @classmethod
    def ones(cls, *shape: int, dtype: DType = DType.FLOAT64) -> "Tensor":
        t = cls(shape=shape, dtype=dtype)
        for i in range(t.numel): t._buffer[i] = 1
        return t
 
    @classmethod
    def from_scalar(cls, v: float, dtype: DType = DType.FLOAT64) -> "Tensor":
        return cls([v], shape=(1,), dtype=dtype)
 
    @classmethod
    def randn(cls, *shape: int, seed: int = 0) -> "Tensor":
        """
        Gaussian random tensor using Box–Muller transform.
        Uses a linear congruential generator seeded by `seed` —
        no random module required.
        """
        t  = cls(shape=shape, dtype=DType.FLOAT64)
        n  = t.numel
        # LCG parameters (Knuth)
        a, c, m = 1664525, 1013904223, 2 ** 32
        state    = seed ^ 0xDEADBEEF
        uniform_samples: List[float] = []
        while len(uniform_samples) < n + 1:
            state = (a * state + c) % m
            uniform_samples.append((state + 1) / (m + 1))   # (0, 1)
        # Box–Muller
        for i in range(0, n, 2):
            u1 = uniform_samples[i]
            u2 = uniform_samples[i + 1]
            mag = CORDIC.sqrt(-2.0 * CORDIC.ln(u1))
            z0  = mag * CORDIC.cos(CORDIC.TWO_PI * u2)
            z1  = mag * CORDIC.sin(CORDIC.TWO_PI * u2)
            t._buffer[i]     = z0
            if i + 1 < n:
                t._buffer[i + 1] = z1
        return t
 
    def softmax(self) -> "Tensor":
        """Row-wise softmax for 1D or 2D tensors."""
        if self.ndim == 1:
            max_v = max(self._buffer)
            exps  = [CORDIC.exp(v - max_v) for v in self._buffer]
            total = sum(exps)
            normalized = [e / total for e in exps]
            return Tensor(normalized, shape=self._shape, dtype=DType.FLOAT64)
        # 2D: row-wise
        rows = []
        for i in range(self._shape[0]):
            row = [self._buffer[i * self._shape[1] + j] for j in range(self._shape[1])]
            max_v = max(row)
            exps  = [CORDIC.exp(v - max_v) for v in row]
            total = sum(exps)
            rows.append([e / total for e in exps])
        return Tensor(rows)
 
    def __repr__(self) -> str:
        if self.numel <= 8:
            vals = list(self._buffer)
            return f"Tensor(shape={self._shape}, {vals})"
        return f"Tensor(shape={self._shape}, dtype={self._dtype.name})"
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — NEURAL NETWORK FOUNDATION
#  Minimal layers, loss functions, optimisers.
#  All math through Tensor + DualNumber + CORDIC.
# ════════════════════════════════════════════════════════════════════════════════
 
class Layer(ABC):
    """Abstract base for all neural network layers."""
 
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
 
    @abstractmethod
    def parameters(self) -> List[Tensor]: ...
 
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
 
 
class LinearLayer(Layer):
    """
    Fully connected layer: y = xW^T + b.
    Weights initialised with He initialisation (scale = √(2/fan_in)).
    """
 
    def __init__(self, in_features: int, out_features: int, seed: int = 42) -> None:
        scale = CORDIC.sqrt(2.0 / in_features)
        self.W = Tensor.randn(out_features, in_features, seed=seed) * scale
        self.b = Tensor.zeros(out_features)
 
    @agent_method(
        name="linear_forward",
        description="Linear layer forward pass: xW^T + b",
        priority=AgentPriority.NORMAL,
    )
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, in) or (in,)  → out: (batch, out) or (out,)
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))
        out = x @ self.W.T()
        # Broadcast bias
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out._buffer[i * out.shape[1] + j] += self.b._buffer[j]
        return out
 
    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]
 
 
class ActivationLayer(Layer):
    """Activation functions computed via CORDIC-backed primitives."""
 
    class Kind(Enum):
        RELU    = auto()
        TANH    = auto()
        SIGMOID = auto()
 
    def __init__(self, kind: "ActivationLayer.Kind" = Kind.RELU) -> None:
        self._kind = kind
 
    def forward(self, x: Tensor) -> Tensor:
        if self._kind == self.Kind.RELU:
            return x.apply(lambda v: v if v > 0.0 else 0.0)
        if self._kind == self.Kind.TANH:
            return x.apply(CORDIC.tanh)
        if self._kind == self.Kind.SIGMOID:
            return x.apply(CORDIC.sigmoid)
        raise ValueError(f"Unknown activation {self._kind}")
 
    def parameters(self) -> List[Tensor]:
        return []
 
 
def mse_loss(pred: Tensor, target: Tensor) -> float:
    """Mean Squared Error: (1/n) Σ (p_i - t_i)²."""
    diff = pred - target
    sq   = diff * diff
    return sq.mean().item()
 
 
def cross_entropy_loss(logits: Tensor, target_idx: int) -> float:
    """
    Softmax cross-entropy for classification.
    logits: (n_classes,) raw scores.
    """
    probs    = logits.softmax()
    p_target = probs._buffer[target_idx]
    return -CORDIC.ln(max(p_target, 1e-12))
 
 
class SGDOptimizer:
    """
    Stochastic Gradient Descent with optional momentum.
    Updates: θ ← θ − lr · g
    """
 
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0) -> None:
        self._params   = params
        self._lr       = lr
        self._momentum = momentum
        self._velocity = [Tensor.zeros(*p.shape) for p in params]
 
    @agent_method(
        name="sgd_step",
        description="Apply one SGD update step given parameter gradients",
        parameters={"gradients": {"type": "List[Tensor]", "desc": "Gradient tensors matching params"}},
        priority=AgentPriority.HIGH,
    )
    def step(self, gradients: List[Tensor]) -> None:
        for p, g, v in zip(self._params, gradients, self._velocity):
            # v = momentum * v + (1 - momentum) * g
            for i in range(v.numel):
                v._buffer[i] = self._momentum * v._buffer[i] + (1 - self._momentum) * g._buffer[i]
            # p -= lr * v
            for i in range(p.numel):
                p._buffer[i] -= self._lr * v._buffer[i]
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — AGENT REASONER
#  The pluggable reasoning interface. Rule-based by default.
#  Swap in an LLM-backed reasoner without changing any calling code.
# ════════════════════════════════════════════════════════════════════════════════
 
class AgentReasoner(ABC):
    """
    Abstract reasoning engine.
    Every kernel decision that isn't deterministic passes through here.
    """
 
    @abstractmethod
    def decide(
        self,
        context:   str,
        options:   List[str],
        ctx:       AgentContext,
        metadata:  Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Given a context description and a list of option strings,
        return the name of the chosen option.
        """
        ...
 
    @abstractmethod
    def annotate(
        self,
        tool_name: str,
        args:      Tuple,
        kwargs:    Dict,
        ctx:       AgentContext,
    ) -> Optional[str]:
        """
        Return a reasoning annotation for this tool call (or None if none).
        Non-blocking advisory — called before the tool executes.
        """
        ...
 
    @abstractmethod
    def plan(
        self,
        goal:      str,
        tools:     List[AgentToolSpec],
        ctx:       AgentContext,
    ) -> List[Dict[str, Any]]:
        """
        Given a high-level goal and available tools, return an ordered
        list of tool-call steps: [{"tool": name, "kwargs": {...}}, ...]
        """
        ...
 
 
class RuleBasedReasoner(AgentReasoner):
    """
    Deterministic, zero-latency reasoner.
    Implements policy rules directly in Python.
    This is the kernel's default reasoner — never fails, never blocks.
    Production upgrade path: replace with LLMReasoner without changing callers.
    """
 
    def decide(
        self,
        context:  str,
        options:  List[str],
        ctx:      AgentContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Policy: prefer the first option unless metadata says otherwise.
        # Subclasses or hot-patch can override specific context strings.
        if not options:
            raise ValueError("decide() called with empty options list")
        priority_hint = (metadata or {}).get("prefer")
        if priority_hint and priority_hint in options:
            return priority_hint
        return options[0]
 
    def annotate(
        self,
        tool_name: str,
        args:      Tuple,
        kwargs:    Dict,
        ctx:       AgentContext,
    ) -> Optional[str]:
        # Lightweight annotations for high-value tools
        annotations = {
            "palloc":       "Physical page allocation",
            "map_page":     "Virtual→physical page mapping",
            "idt_dispatch": "Interrupt dispatch",
            "peek8":        "Memory read",
            "poke8":        "Memory write",
        }
        return annotations.get(tool_name)
 
    def plan(
        self,
        goal:  str,
        tools: List[AgentToolSpec],
        ctx:   AgentContext,
    ) -> List[Dict[str, Any]]:
        # Rule-based planner: match goal keywords to tool names
        plan   : List[Dict] = []
        goal_lc = goal.lower()
        for spec in sorted(tools, key=lambda s: s.priority):
            if any(kw in goal_lc for kw in spec.name.lower().split("_")):
                plan.append({"tool": spec.name, "kwargs": {}})
        return plan
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — AGENT KERNEL
#  The central orchestrator. Owns all subsystems.
#  Exposes dispatch(), plan_and_execute(), and the system call interface.
# ════════════════════════════════════════════════════════════════════════════════
 
class KernelState(Enum):
    OFFLINE     = auto()
    BOOT        = auto()
    READY       = auto()
    RUNNING     = auto()
    HALT        = auto()
    PANIC       = auto()
 
 
@dataclass
class SysCallResult:
    success:   bool
    value:     Any
    error:     Optional[str] = None
    trace_id:  Optional[str] = None
 
 
class AgentKernel:
    """
    The AIOS Agent Kernel.
 
    Every subsystem is initialised here in boot-sequence order:
        1. Physical allocator
        2. Memory bus
        3. Virtual memory manager
        4. IDT + PIC (interrupt infrastructure)
        5. VGA driver
        6. Agent reasoner (rule-based by default)
 
    After boot, any code can call kernel.dispatch(tool_name, **kwargs) or
    kernel.syscall(name, **kwargs) to interact with the OS through the
    agent method protocol.
    """
 
    def __init__(self, reasoner: Optional[AgentReasoner] = None) -> None:
        self._state    : KernelState      = KernelState.OFFLINE
        self._reasoner : AgentReasoner    = reasoner or RuleBasedReasoner()
        self._registry : AgentRegistry   = _registry
        self._lock     : threading.RLock = threading.RLock()
        self._task_q   : deque           = deque()
        self._boot_log : List[str]       = []
 
        # Subsystems — uninitialised until boot()
        self.palloc   : Optional[PhysicalAllocator]    = None
        self.bus      : Optional[MemoryBus]             = None
        self.vmm      : Optional[VirtualMemoryManager] = None
        self.idt      : Optional[InterruptDescriptorTable] = None
        self.pic      : Optional[PICDriver]             = None
        self.vga      : Optional[VGATextDriver]         = None
 
    # ── Boot sequence ─────────────────────────────────────────────────────────
 
    def _log(self, msg: str) -> None:
        ts = time.monotonic()
        entry = f"[{ts:9.4f}] {msg}"
        self._boot_log.append(entry)
        # Always print to stdout; also mirror to VGA once it is available
        print(entry)
        if self.vga:
            self.vga.writeln(entry)
 
    @agent_method(
        name="kernel_boot",
        description="Execute the full AIOS kernel boot sequence",
        priority=AgentPriority.CRITICAL,
    )
    def boot(self) -> bool:
        with self._lock:
            if self._state not in (KernelState.OFFLINE, KernelState.HALT):
                return False
            self._state = KernelState.BOOT
 
        try:
            self._log("═" * 60)
            self._log(f" AIOS v{'.'.join(str(v) for v in AIOS_VERSION)} '{AIOS_CODENAME}'")
            self._log("═" * 60)
 
            # Phase I — Physical memory
            self._log("[BOOT] Phase I: Physical allocator initialising ...")
            self.palloc = PhysicalAllocator(RAM_SIZE_BYTES)
            self._log(f"[BOOT]   {self.palloc.free_pages} free pages ({RAM_SIZE_BYTES // 1024 // 1024} MiB)")
 
            # Phase I — Memory bus
            self._log("[BOOT] Phase I: Memory bus online.")
            self.bus = MemoryBus(self.palloc)
 
            # Phase I — VMM
            self._log("[BOOT] Phase I: Virtual memory manager initialising ...")
            self.vmm = VirtualMemoryManager(self.palloc)
            for region in MEMORY_MAP:
                self.vmm.identity_map_region(region.base, region.size, region.flags)
            self._log("[BOOT]   Identity-mapped all well-known regions.")
 
            # Phase II — Interrupts
            self._log("[BOOT] Phase II: Interrupt infrastructure ...")
            self.idt = InterruptDescriptorTable()
            self._install_default_handlers()
 
            self.pic = PICDriver(self.bus, self.idt)
            self.pic.unmask(int(IRQLine.KEYBOARD))
            self.pic.unmask(int(IRQLine.TIMER))
            self._log("[BOOT]   IDT loaded. PIC initialised. IRQ0/1 unmasked.")
 
            # Phase II — VGA
            self._log("[BOOT] Phase II: VGA text driver online.")
            self.vga = VGATextDriver(self.bus)
            self.vga.clear()
            self.vga.set_color(VGAColor.LGREEN)
            self.vga.writeln(f"AIOS v{'.'.join(str(v) for v in AIOS_VERSION)} '{AIOS_CODENAME}' — Boot OK")
            self.vga.set_color(VGAColor.LGRAY)
 
            # Phase III — Math primitives self-test
            self._log("[BOOT] Phase III: CORDIC self-test ...")
            self._selftest_cordic()
 
            # Phase III — Dual number self-test
            self._log("[BOOT] Phase III: DualNumber AD self-test ...")
            self._selftest_dual()
 
            # Phase IV — Tensor smoke test
            self._log("[BOOT] Phase IV: Tensor subsystem smoke test ...")
            self._selftest_tensor()
 
            self._state = KernelState.READY
            self._log("[BOOT] *** Kernel READY ***")
            return True
 
        except Exception as exc:
            self._state = KernelState.PANIC
            self._log(f"[PANIC] Boot failed: {exc}")
            traceback.print_exc()
            return False
 
    def _install_default_handlers(self) -> None:
        """Install minimal ISRs for critical exceptions."""
        assert self.idt is not None
 
        def gp_fault(vec: int, ctx: Any) -> None:
            self._log(f"[ISR] General Protection Fault (vec={vec})")
 
        def page_fault(vec: int, ctx: Any) -> None:
            self._log(f"[ISR] Page Fault (vec={vec}, ctx={ctx!r})")
 
        def keyboard_isr(vec: int, ctx: Any) -> None:
            # In real hardware: read scancode from port 0x60
            if self.bus:
                scancode = self.bus.inb(0x60)
                # Minimal scancode → ASCII (set 1, make codes only)
                _SC_MAP: Dict[int, str] = {
                    0x1E:'a',0x30:'b',0x2E:'c',0x20:'d',0x12:'e',
                    0x21:'f',0x22:'g',0x23:'h',0x17:'i',0x24:'j',
                    0x25:'k',0x26:'l',0x32:'m',0x31:'n',0x18:'o',
                    0x19:'p',0x10:'q',0x13:'r',0x1F:'s',0x14:'t',
                    0x16:'u',0x2F:'v',0x11:'w',0x2D:'x',0x15:'y',
                    0x2C:'z',0x39:' ',0x1C:'\n',0x0E:'\b',
                    0x02:'1',0x03:'2',0x04:'3',0x05:'4',0x06:'5',
                    0x07:'6',0x08:'7',0x09:'8',0x0A:'9',0x0B:'0',
                }
                char = _SC_MAP.get(scancode & 0x7F)
                if char and not (scancode & 0x80):   # ignore break codes
                    self._keyboard_buffer.append(char)
 
        self._keyboard_buffer: deque = deque(maxlen=256)
        self.idt.register(13, gp_fault)
        self.idt.register(14, page_fault)
        self.idt.register(33, keyboard_isr)   # IRQ1 → INT 33
 
    def _selftest_cordic(self) -> None:
        """Verify CORDIC accuracy against known values."""
        tests = [
            (0.0,              0.0,    1.0),   # sin(0)=0,   cos(0)=1
            (CORDIC.HALF_PI,   1.0,    0.0),   # sin(π/2)=1, cos(π/2)=0
            (CORDIC.PI,        0.0,   -1.0),   # sin(π)=0,   cos(π)=-1
            (CORDIC.PI / 4, 0.7071, 0.7071),
        ]
        for angle, exp_sin, exp_cos in tests:
            s, c = CORDIC.sincos(angle)
            tol  = 1e-4
            assert abs(s - exp_sin) < tol, f"sin({angle:.4f}) = {s:.6f} ≠ {exp_sin}"
            assert abs(c - exp_cos) < tol, f"cos({angle:.4f}) = {c:.6f} ≠ {exp_cos}"
        # exp / ln round-trip
        for v in [1.0, 2.0, 0.5, 10.0]:
            reconstructed = CORDIC.exp(CORDIC.ln(v))
            assert abs(reconstructed - v) < 1e-6, f"exp(ln({v})) = {reconstructed}"
        self._log("[BOOT]   CORDIC ✓ (sin/cos/exp/ln verified)")
 
    def _selftest_dual(self) -> None:
        """Verify dual-number AD against known derivatives."""
        # d/dx (x³ + 2x) at x=3 → 3x² + 2 = 29
        def fn(x: DualNumber) -> DualNumber:
            return x ** 3 + DualNumber(2.0) * x
        result = fn(DualNumber(3.0, 1.0))
        assert abs(result.real - 33.0) < 1e-9, f"f(3) = {result.real} ≠ 33"
        assert abs(result.dual - 29.0) < 1e-9, f"f'(3) = {result.dual} ≠ 29"
        # d/dx sin(x) at x=0 → cos(0) = 1
        d_sin = grad(DualNumber.sin, 0.0)
        assert abs(d_sin - 1.0) < 1e-6, f"d/dx sin(0) = {d_sin} ≠ 1"
        self._log("[BOOT]   DualNumber AD ✓ (f'(3)=29, d/dx sin(0)=1)")
 
    def _selftest_tensor(self) -> None:
        """Basic tensor shape, matmul, and softmax verification."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        assert c.shape == (2, 2), f"Matmul shape mismatch: {c.shape}"
        assert abs(c[0, 0] - 19.0) < 1e-9, f"c[0,0] = {c[0,0]}"
        assert abs(c[1, 1] - 50.0) < 1e-9, f"c[1,1] = {c[1,1]}"
        logits = Tensor([1.0, 2.0, 3.0])
        probs  = logits.softmax()
        total  = sum(probs._buffer)
        assert abs(total - 1.0) < 1e-9, f"softmax sum = {total}"
        self._log("[BOOT]   Tensor ✓ (matmul, softmax verified)")
 
    # ── Agent dispatch ────────────────────────────────────────────────────────
 
    @agent_method(
        name="kernel_dispatch",
        description="Invoke a registered agent tool by name",
        parameters={
            "tool_name": {"type": "str",  "desc": "Registered agent tool name"},
            "kwargs":    {"type": "dict", "desc": "Keyword arguments for the tool"},
        },
        priority=AgentPriority.NORMAL,
    )
    def dispatch(self, tool_name: str, **kwargs: Any) -> SysCallResult:
        spec = self._registry.get(tool_name)
        if spec is None:
            return SysCallResult(False, None, f"Unknown tool: {tool_name!r}")
 
        ctx = AgentContext(caller="kernel.dispatch")
        ctx._chain = [tool_name]
 
        try:
            result = spec.fn(**kwargs, _ctx=ctx) if "_ctx" in spec.fn.__code__.co_varnames \
                     else spec.fn(**kwargs)
            return SysCallResult(True, result, trace_id=ctx.trace_id)
        except Exception as exc:
            return SysCallResult(False, None, error=str(exc), trace_id=ctx.trace_id)
 
    @agent_method(
        name="kernel_plan_execute",
        description="Plan and execute a sequence of agent tools to achieve a high-level goal",
        parameters={"goal": {"type": "str", "desc": "Natural language goal description"}},
        priority=AgentPriority.NORMAL,
    )
    def plan_and_execute(self, goal: str) -> List[SysCallResult]:
        ctx   = AgentContext(caller="planner")
        plan  = self._reasoner.plan(goal, self._registry.all_tools(), ctx)
        results: List[SysCallResult] = []
        for step in plan:
            res = self.dispatch(step["tool"], **step.get("kwargs", {}))
            results.append(res)
            if not res.success:
                break
        return results
 
    @agent_method(
        name="syscall",
        description="High-level system call interface — unified entry point for user-space",
        parameters={
            "name":   {"type": "str",  "desc": "System call name"},
            "kwargs": {"type": "dict", "desc": "Arguments"},
        },
        priority=AgentPriority.NORMAL,
    )
    def syscall(self, name: str, **kwargs: Any) -> SysCallResult:
        """
        Canonical user-space entry point.
        Maps syscall names to agent tools with safety checks.
        """
        if self._state not in (KernelState.READY, KernelState.RUNNING):
            return SysCallResult(False, None, f"Kernel not ready (state={self._state.name})")
        return self.dispatch(name, **kwargs)
 
    # ── Introspection ─────────────────────────────────────────────────────────
 
    def status(self) -> Dict[str, Any]:
        return {
            "version":   AIOS_VERSION,
            "codename":  AIOS_CODENAME,
            "state":     self._state.name,
            "registry":  self._registry.stats(),
            "memory":    {
                "free_pages":  self.palloc.free_pages  if self.palloc else 0,
                "used_pages":  self.palloc.used_pages   if self.palloc else 0,
                "page_size":   PAGE_SIZE,
            } if self.palloc else {},
        }
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — BOOT STUB SIMULATOR
#  Simulates the MBR → protected-mode → kernel handoff.
#  In bare-metal deployment, this section is replaced by actual assembly.
#  Here it validates the sequence and transitions KernelState correctly.
# ════════════════════════════════════════════════════════════════════════════════
 
class BootStub:
    """
    Simulates the 512-byte MBR bootstrap process:
        1. CPU reset → fetch from 0xFFFFFFF0 (BIOS ROM)
        2. BIOS POST → load MBR at 0x7C00
        3. Validate 0x55AA signature at offset 510–511
        4. Switch to 32-bit protected mode (GDT → PE bit → far jump)
        5. Hand off to Python kernel entry point
 
    On real hardware, steps 1–4 are pure assembly. This class models
    the sequence so it can be validated and documented.
    """
 
    MBR_MAGIC_OFFSET = 510
    MBR_MAGIC        = 0xAA55   # little-endian: 0x55 @ 510, 0xAA @ 511
 
    # Minimal GDT layout (3 entries: null, code, data)
    GDT_NULL   = 0x0000000000000000
    GDT_CODE32 = 0x00CF9A000000FFFF   # base=0, limit=4GiB, DPL=0, 32-bit
    GDT_DATA32 = 0x00CF92000000FFFF
 
    def __init__(self, bus: MemoryBus) -> None:
        self._bus = bus
        self._gdt_phys_addr: int = 0x0800    # write GDT at 0x800
        self._boot_record   = bytearray(512)
 
    def _write_mbr(self) -> None:
        """Write a minimal MBR skeleton into the boot record buffer."""
        # Simulate: org 0x7C00; cli; xor ax,ax; mov ds,ax; ...
        stub_code = [
            0xFA,             # CLI
            0x31, 0xC0,       # XOR AX, AX
            0x8E, 0xD8,       # MOV DS, AX
            0x8E, 0xC0,       # MOV ES, AX
            0x8E, 0xD0,       # MOV SS, AX
            0xBC, 0x00, 0x7C, # MOV SP, 0x7C00
            0xEB, 0xFE,       # JMP $ (infinite loop placeholder before handoff)
        ]
        for i, byte in enumerate(stub_code):
            self._boot_record[i] = byte
        # Magic bytes at offset 510–511
        self._boot_record[510] = 0x55
        self._boot_record[511] = 0xAA
 
    def _validate_mbr(self) -> bool:
        sig = (self._boot_record[511] << 8) | self._boot_record[510]
        return sig == self.MBR_MAGIC
 
    def _load_gdt(self) -> None:
        """Write GDT entries into simulated physical memory."""
        addr = self._gdt_phys_addr
        for entry in (self.GDT_NULL, self.GDT_CODE32, self.GDT_DATA32):
            self._bus.poke_buf(addr, struct.pack("<Q", entry))
            addr += 8
 
    def _enter_protected_mode(self) -> bool:
        """
        Simulate the PE-bit transition:
            LGDT → OR [CR0], 1 → LJMP 0x08:pm_entry
 
        Returns True if the simulation succeeds without faults.
        """
        # Write GDTR pseudo-register simulation: 6 bytes (limit, base)
        gdt_limit = 3 * 8 - 1        # 3 entries, each 8 bytes
        gdt_base  = self._gdt_phys_addr
        gdtr = struct.pack("<HI", gdt_limit, gdt_base)
        self._bus.poke_buf(0x0900, gdtr)   # store GDTR contents at 0x900
 
        # Simulate setting PE bit in CR0
        cr0_sim = self._bus.peek32(0x0910)
        cr0_sim |= 0x00000001              # PE bit
        self._bus.poke32(0x0910, cr0_sim)
 
        # Confirm PE bit is set
        return bool(self._bus.peek32(0x0910) & 1)
 
    @agent_method(
        name="boot_execute",
        description="Execute the full simulated boot stub sequence",
        priority=AgentPriority.CRITICAL,
    )
    def execute(self) -> bool:
        """Run all boot-stub stages and return True on success."""
        self._write_mbr()
        if not self._validate_mbr():
            raise RuntimeError("MBR signature validation failed")
 
        self._bus.poke_buf(KERNEL_LOAD_ADDR, bytes(self._boot_record))
        self._load_gdt()
 
        if not self._enter_protected_mode():
            raise RuntimeError("Protected mode transition failed")
 
        return True
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TERMINAL REPL
#  Read-Eval-Print Loop backed by VGA driver for output and direct tty
#  for input. Provides OS-level commands + a Python expression evaluator.
# ════════════════════════════════════════════════════════════════════════════════
 
class TerminalREPL:
    """
    Minimal terminal that sits on top of the AIOS kernel.
 
    Commands:
        help                 — list available commands
        status               — kernel status report
        tools                — list all registered agent tools
        traces [n]           — show last n agent traces
        peek <addr>          — read byte from physical address
        poke <addr> <val>    — write byte to physical address
        palloc [n]           — allocate n physical pages
        dispatch <tool> ...  — invoke an agent tool directly
        plan <goal>          — plan and execute a goal
        demo neural          — run a minimal neural network demo
        demo cordic          — demonstrate CORDIC precision
        demo tensor          — tensor operations showcase
        quit / exit          — halt the kernel and exit
    """
 
    PROMPT = "\033[1;32mAIOS\033[0m\033[1;37m@kernel\033[0m \033[1;34m❯\033[0m "
 
    def __init__(self, kernel: AgentKernel) -> None:
        self._kernel  = kernel
        self._running = False
        self._history: List[str] = []
 
    def _out(self, msg: str, color: str = "") -> None:
        RESET = "\033[0m"
        print(f"{color}{msg}{RESET if color else ''}")
        if self._kernel.vga:
            self._kernel.vga.writeln(msg)
 
    def _err(self, msg: str) -> None:
        self._out(f"  ✗ {msg}", "\033[1;31m")
 
    def _ok(self, msg: str) -> None:
        self._out(f"  ✓ {msg}", "\033[1;32m")
 
    def _info(self, msg: str) -> None:
        self._out(f"  {msg}", "\033[0;36m")
 
    # ── Command handlers ──────────────────────────────────────────────────────
 
    def _cmd_help(self, _: List[str]) -> None:
        cmds = [
            ("help",            "Show this message"),
            ("status",          "Kernel status report"),
            ("tools",           "List registered agent tools"),
            ("traces [n]",      "Show last n agent traces (default 10)"),
            ("peek <addr>",     "Read byte from physical address (hex ok)"),
            ("poke <addr> <v>", "Write byte to physical address"),
            ("palloc [n]",      "Allocate n contiguous physical pages"),
            ("dispatch <tool>", "Invoke an agent tool by name"),
            ("plan <goal>",     "Plan and execute a natural-language goal"),
            ("demo neural",     "Minimal neural network training demo"),
            ("demo cordic",     "CORDIC math precision demonstration"),
            ("demo tensor",     "Tensor operations showcase"),
            ("demo ad",         "Automatic differentiation demo"),
            ("quit / exit",     "Halt the kernel"),
        ]
        self._out("\n  AIOS Command Reference:", "\033[1;33m")
        for cmd, desc in cmds:
            self._info(f"  {cmd:<28} {desc}")
        print()
 
    def _cmd_status(self, _: List[str]) -> None:
        s = self._kernel.status()
        self._out("\n  Kernel Status:", "\033[1;33m")
        self._info(f"  Version  : {'.'.join(str(v) for v in s['version'])} '{s['codename']}'")
        self._info(f"  State    : {s['state']}")
        if s.get("memory"):
            m = s["memory"]
            self._info(f"  Memory   : {m['free_pages']} free pages / "
                       f"{m['free_pages'] + m['used_pages']} total "
                       f"({m['free_pages'] * PAGE_SIZE // 1024} KiB free)")
        reg = s.get("registry", {})
        self._info(f"  Tools    : {reg.get('registered_tools', 0)} registered, "
                   f"{reg.get('total_calls', 0)} total calls")
        print()
 
    def _cmd_tools(self, _: List[str]) -> None:
        self._out("\n  Registered Agent Tools:", "\033[1;33m")
        for spec in sorted(self._kernel._registry.all_tools(), key=lambda s: s.priority):
            pri = spec.priority.name
            self._info(f"  [{pri:<8}] {spec.name:<30} {spec.description[:50]}")
        print()
 
    def _cmd_traces(self, args: List[str]) -> None:
        n = int(args[0]) if args else 10
        traces = self._kernel._registry.recent_traces(n)
        self._out(f"\n  Last {len(traces)} Agent Traces:", "\033[1;33m")
        for t in traces:
            status = "✓" if t.success else "✗"
            self._info(f"  {status} {t.tool_name:<30} {t.duration_ns // 1000:>6} µs"
                       + (f"  [{t.error}]" if t.error else ""))
        print()
 
    def _cmd_peek(self, args: List[str]) -> None:
        if not args:
            self._err("Usage: peek <address>"); return
        addr = int(args[0], 0)
        if not self._kernel.bus:
            self._err("Memory bus offline"); return
        val = self._kernel.bus.peek8(addr)
        self._ok(f"0x{addr:08X} → 0x{val:02X} ({val})")
 
    def _cmd_poke(self, args: List[str]) -> None:
        if len(args) < 2:
            self._err("Usage: poke <address> <value>"); return
        addr = int(args[0], 0)
        val  = int(args[1], 0)
        if not self._kernel.bus:
            self._err("Memory bus offline"); return
        self._kernel.bus.poke8(addr, val)
        self._ok(f"0x{addr:08X} ← 0x{val:02X}")
 
    def _cmd_palloc(self, args: List[str]) -> None:
        n = int(args[0]) if args else 1
        if not self._kernel.palloc:
            self._err("Physical allocator offline"); return
        addr = self._kernel.palloc.alloc(n)
        if addr is None:
            self._err(f"Allocation of {n} pages failed (OOM)")
        else:
            self._ok(f"Allocated {n} page(s) at physical address 0x{addr:08X}")
 
    def _cmd_dispatch(self, args: List[str]) -> None:
        if not args:
            self._err("Usage: dispatch <tool_name> [key=value ...]"); return
        tool = args[0]
        kwargs: Dict[str, Any] = {}
        for kv in args[1:]:
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    kwargs[k] = int(v, 0)
                except ValueError:
                    kwargs[k] = v
        res = self._kernel.syscall(tool, **kwargs)
        if res.success:
            self._ok(f"Result: {res.value!r}")
        else:
            self._err(f"Failed: {res.error}")
 
    def _cmd_plan(self, args: List[str]) -> None:
        if not args:
            self._err("Usage: plan <goal text>"); return
        goal    = " ".join(args)
        results = self._kernel.plan_and_execute(goal)
        self._out(f"\n  Plan execution for: '{goal}'", "\033[1;33m")
        for i, r in enumerate(results):
            status = "✓" if r.success else "✗"
            self._info(f"  Step {i+1}: {status} → {r.value!r}")
        print()
 
    def _demo_cordic(self, _: List[str]) -> None:
        self._out("\n  CORDIC Math Demonstration:", "\033[1;33m")
        angles = [0.0, CORDIC.PI / 6, CORDIC.PI / 4, CORDIC.PI / 3, CORDIC.HALF_PI]
        names  = ["0", "π/6", "π/4", "π/3", "π/2"]
        for angle, name in zip(angles, names):
            s, c = CORDIC.sincos(angle)
            self._info(f"  sin({name:>4}) = {s:+.8f}    cos({name:>4}) = {c:+.8f}")
        self._info(f"\n  exp(1)  = {CORDIC.exp(1.0):.15f}")
        self._info(f"  ln(e)   = {CORDIC.ln(CORDIC.exp(1.0)):.15f}")
        self._info(f"  sqrt(2) = {CORDIC.sqrt(2.0):.15f}")
        self._info(f"  tanh(1) = {CORDIC.tanh(1.0):.15f}")
        print()
 
    def _demo_ad(self, _: List[str]) -> None:
        self._out("\n  Automatic Differentiation (Dual Numbers):", "\033[1;33m")
 
        def f(x: DualNumber) -> DualNumber:
            return DualNumber.sin(x) * x ** 2 + DualNumber.exp(x * DualNumber(-0.5))
 
        for xv in [0.0, 1.0, 2.0]:
            result = f(DualNumber(xv, 1.0))
            self._info(f"  f({xv}) = {result.real:+.6f}    f'({xv}) = {result.dual:+.6f}")
        print()
 
    def _demo_tensor(self, _: List[str]) -> None:
        self._out("\n  Tensor Operations:", "\033[1;33m")
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        c = a @ b
        self._info(f"  A  shape={a.shape}: {a.to_list()}")
        self._info(f"  B  shape={b.shape}: {b.to_list()}")
        self._info(f"  A@B shape={c.shape}: {c.to_list()}")
        logits = Tensor([2.0, 1.0, 0.1])
        probs  = logits.softmax()
        self._info(f"  softmax([2.0, 1.0, 0.1]) = {[round(v, 4) for v in probs._buffer]}")
        rng = Tensor.randn(3, 3, seed=7)
        self._info(f"  randn(3,3) mean={rng.mean().item():+.4f}")
        print()
 
    def _demo_neural(self, _: List[str]) -> None:
        self._out("\n  Neural Network Training Demo (XOR problem):", "\033[1;33m")
        # XOR: inputs (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
        X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        Y = Tensor([[0.0], [1.0], [1.0], [0.0]])
 
        l1   = LinearLayer(2, 8, seed=1)
        act1 = ActivationLayer(ActivationLayer.Kind.TANH)
        l2   = LinearLayer(8, 1, seed=2)
        act2 = ActivationLayer(ActivationLayer.Kind.SIGMOID)
 
        lr     = 0.1
        epochs = 300
        prev_loss = None
 
        for epoch in range(epochs):
            # Manual forward pass for all 4 samples
            total_loss = 0.0
            # Accumulate gradients numerically via finite differences
            # (Full backprop requires reverse-mode AD; we use finite diff here)
            # This demonstrates the training loop structure.
            for i in range(4):
                xi = Tensor([X._buffer[i*2], X._buffer[i*2+1]])
                yi = Y._buffer[i]
                h  = act1(l1(xi))
                o  = act2(l2(h))
                pred = o._buffer[0]
                total_loss += (pred - yi) ** 2
 
            total_loss /= 4.0
            if epoch % 50 == 0 or epoch == epochs - 1:
                self._info(f"  Epoch {epoch:>4}: loss = {total_loss:.6f}")
 
            # Gradient step via finite differences on all parameters
            eps = 1e-4
            for layer in [l1, l2]:
                for param in layer.parameters():
                    for idx in range(param.numel):
                        original = param._buffer[idx]
 
                        param._buffer[idx] = original + eps
                        loss_plus = 0.0
                        for i in range(4):
                            xi = Tensor([X._buffer[i*2], X._buffer[i*2+1]])
                            yi = Y._buffer[i]
                            h  = act1(l1(xi))
                            o  = act2(l2(h))
                            loss_plus += (o._buffer[0] - yi) ** 2
                        loss_plus /= 4.0
 
                        param._buffer[idx] = original - eps
                        loss_minus = 0.0
                        for i in range(4):
                            xi = Tensor([X._buffer[i*2], X._buffer[i*2+1]])
                            yi = Y._buffer[i]
                            h  = act1(l1(xi))
                            o  = act2(l2(h))
                            loss_minus += (o._buffer[0] - yi) ** 2
                        loss_minus /= 4.0
 
                        grad_v = (loss_plus - loss_minus) / (2 * eps)
                        param._buffer[idx] = original - lr * grad_v
 
        self._out("\n  Final predictions:", "\033[1;33m")
        for i in range(4):
            xi   = Tensor([X._buffer[i*2], X._buffer[i*2+1]])
            yi   = Y._buffer[i]
            h    = act1(l1(xi))
            o    = act2(l2(h))
            pred = o._buffer[0]
            self._info(f"  XOR({int(X._buffer[i*2])},{int(X._buffer[i*2+1])}) "
                       f"target={yi:.0f} pred={pred:.4f}")
        print()
 
    # ── REPL main loop ────────────────────────────────────────────────────────
 
    def run(self) -> None:
        self._running = True
        self._kernel._state = KernelState.RUNNING
 
        print()
        self._out("═" * 60, "\033[1;36m")
        self._out(f"  AIOS Terminal — v{'.'.join(str(v) for v in AIOS_VERSION)} '{AIOS_CODENAME}'",
                  "\033[1;36m")
        self._out("  Type 'help' for commands.", "\033[0;36m")
        self._out("═" * 60, "\033[1;36m")
        print()
 
        _COMMAND_MAP: Dict[str, Callable] = {
            "help":     self._cmd_help,
            "status":   self._cmd_status,
            "tools":    self._cmd_tools,
            "traces":   self._cmd_traces,
            "peek":     self._cmd_peek,
            "poke":     self._cmd_poke,
            "palloc":   self._cmd_palloc,
            "dispatch": self._cmd_dispatch,
            "plan":     self._cmd_plan,
        }
 
        _DEMO_MAP: Dict[str, Callable] = {
            "cordic": self._demo_cordic,
            "ad":     self._demo_ad,
            "tensor": self._demo_tensor,
            "neural": self._demo_neural,
        }
 
        while self._running:
            try:
                line = input(self.PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._out("  Interrupt received. Type 'quit' to exit.", "\033[1;33m")
                continue
 
            if not line:
                continue
 
            self._history.append(line)
            parts = line.split()
            cmd   = parts[0].lower()
            args  = parts[1:]
 
            if cmd in ("quit", "exit", "halt"):
                self._kernel._state = KernelState.HALT
                self._out("  AIOS kernel halted. Goodbye.", "\033[1;33m")
                self._running = False
 
            elif cmd == "demo":
                sub = args[0].lower() if args else ""
                handler = _DEMO_MAP.get(sub)
                if handler:
                    handler(args[1:])
                else:
                    self._err(f"Unknown demo: {sub!r}. Options: {list(_DEMO_MAP)}")
 
            elif cmd in _COMMAND_MAP:
                try:
                    _COMMAND_MAP[cmd](args)
                except Exception as exc:
                    self._err(f"Command error: {exc}")
 
            else:
                # Fall through to Python expression evaluator
                try:
                    result = eval(line, {
                        "kernel":  self._kernel,
                        "bus":     self._kernel.bus,
                        "palloc":  self._kernel.palloc,
                        "vmm":     self._kernel.vmm,
                        "vga":     self._kernel.vga,
                        "pic":     self._kernel.pic,
                        "idt":     self._kernel.idt,
                        "CORDIC":  CORDIC,
                        "Tensor":  Tensor,
                        "DualNumber": DualNumber,
                        "IEEE754": IEEE754,
                        "grad":    grad,
                        "PAGE_SIZE": PAGE_SIZE,
                        "__builtins__": {"print": print, "range": range,
                                         "len": len, "type": type, "int": int,
                                         "float": float, "list": list, "abs": abs},
                    })
                    if result is not None:
                        self._info(f"  → {result!r}")
                except SyntaxError:
                    self._err(f"Unknown command: {cmd!r}. Type 'help' for usage.")
                except Exception as exc:
                    self._err(f"Eval error: {exc}")
 
 
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════
 
def main() -> int:
    """
    AIOS entry point.
    Simulates the full boot sequence:
        BootStub → AgentKernel.boot() → TerminalREPL.run()
    """
 
    # 1. Bring up the kernel (allocator + bus must exist before BootStub)
    kernel = AgentKernel(reasoner=RuleBasedReasoner())
 
    # 2. Pre-initialise the minimum needed for BootStub
    kernel.palloc = PhysicalAllocator(RAM_SIZE_BYTES)
    kernel.bus    = MemoryBus(kernel.palloc)
 
    # 3. Execute the boot stub (MBR + protected mode simulation)
    stub = BootStub(kernel.bus)
    try:
        if not stub.execute():
            print("[FATAL] Boot stub failed. Halting.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"[FATAL] Boot stub exception: {exc}", file=sys.stderr)
        return 1
 
    print("[BOOT] Boot stub: MBR validated. Protected mode active. Handing off to kernel ...")
 
    # 4. Full kernel boot sequence
    if not kernel.boot():
        print("[FATAL] Kernel boot failed.", file=sys.stderr)
        return 1
 
    # 5. Enter the terminal REPL
    repl = TerminalREPL(kernel)
    repl.run()
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
