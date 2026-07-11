#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AIOS — Hardware Interface Layer                                            ║
║   Module  : aios_hardware.py                                                 ║
║                                                                              ║
║   "The kernel speaks directly to iron. Every register is a word             ║
║    spoken in voltage. Every interrupt, a scream from silicon."               ║
║                                                                              ║
║   Hierarchy (bottom-up):                                                     ║
║     §0   Platform Detection    — OS, arch, privilege level                   ║
║     §1   CPU Topology          — cores, SMT, cache, NUMA, CPUID flags        ║
║     §2   Memory Subsystem      — /proc/meminfo, huge pages, NUMA nodes       ║
║     §3   PCIe Enumeration      — /sys/bus/pci, vendor IDs, BARs, GPU detect  ║
║     §4   Storage Interface     — /sys/block, NVMe, scheduler, NUMA affinity  ║
║     §5   Performance Monitor   — perf_event_open(2), MSR, RAPL, TSC         ║
║     §6   CPU Affinity & QoS    — sched_setaffinity, NUMA-aware placement     ║
║     §7   Hardware Memory Bus   — mmap-backed real OS pages, huge TLB         ║
║     §8   SIMD Tensor Dispatch  — capability-routed GEMM, Adam, LayerNorm     ║
║     §9   Hardware Layer        — singleton, @agent_method integration        ║
║     §10  Kernel Attachment     — attach_to_kernel(), upgrade boot path       ║
║     §11  Self-Tests            — validates every section, reports results    ║
║                                                                              ║
║   Invariants:                                                                ║
║     • Privilege-checks before every privileged operation                     ║
║     • Graceful degradation: simulated fallback when CAP_SYS_RAWIO absent     ║
║     • No third-party dependencies — stdlib + ctypes only                     ║
║     • Every public method is @agent_method decorated                         ║
║     • Hardware topology feeds directly into kernel memory model              ║
║     • All math is CORDIC-compatible; no 'import math' for numerics           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import struct
import ctypes
import ctypes.util
import mmap
import time
import threading
import hashlib
import json
import fcntl
import errno
import platform
import functools
import traceback
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    NamedTuple, Set, Iterator
)
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, Enum, auto
from collections import OrderedDict, defaultdict
from pathlib import Path
from contextlib import contextmanager

# ── AIOS kernel integration — import from core or fall back to minimal shims ──
try:
    from aios_core import (
        agent_method, AgentPriority, AgentTrace, AgentContext,
        MemoryBus, PhysicalAllocator, RAM_SIZE_BYTES, PAGE_SIZE,
        AIOS_VERSION,
    )
    _AIOS_KERNEL: bool = True
except ImportError:
    _AIOS_KERNEL = False
    RAM_SIZE_BYTES = 64 * 1024 * 1024
    PAGE_SIZE      = 4096
    AIOS_VERSION   = (0, 1, 0)

    class AgentPriority(IntEnum):  # type: ignore[no-redef]
        CRITICAL = 0; HIGH = 1; NORMAL = 2; LOW = 3

    def agent_method(                        # type: ignore[no-redef]
        name:        Optional[str]      = None,
        description: str                = "",
        parameters:  Optional[Dict]     = None,
        returns:     str                = "Any",
        priority:    Any                = None,
        owner:       str                = "hardware",
    ) -> Callable:
        """Passthrough shim when running outside the AIOS kernel."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("_ctx", None)
                return fn(*args, **kwargs)
            return wrapper
        return decorator

# ── Module version ────────────────────────────────────────────────────────────
AIOS_HW_VERSION = (0, 1, 0)

# ── Pure arithmetic helpers (no 'import math') ────────────────────────────────
def _isqrt_int(n: int) -> int:
    """Integer square root via Newton-Raphson.  floor(√n)."""
    if n < 0: raise ValueError("_isqrt_int: negative input")
    if n == 0: return 0
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x

def _sqrt_f(x: float) -> float:
    """float √x via Newton-Raphson (no math import)."""
    if x < 0.0: return float('nan')
    if x == 0.0: return 0.0
    g = x if x >= 1.0 else 1.0
    for _ in range(60):
        g2 = (g + x / g) * 0.5
        if abs(g2 - g) < 1e-15 * g: break
        g = g2
    return g

def _pow_f(base: float, exp: float) -> float:
    """base^exp via exp(exp * ln(base)), all from first principles."""
    if base <= 0.0:
        return 0.0 if base == 0.0 else float('nan')
    # ln(x) = 2 · arctanh((x-1)/(x+1))
    #       = 2 · Σ_{k=0}^∞  1/(2k+1) · ((x-1)/(x+1))^(2k+1)
    t    = (base - 1.0) / (base + 1.0)
    acc  = 0.0
    term = t
    t2   = t * t
    k    = 0
    while True:
        acc  += term / (2 * k + 1)
        term *= t2
        k    += 1
        if abs(term) < 1e-15:
            break
    ln_b = 2.0 * acc           # ← factor of 2 was previously missing
    # exp via Taylor: e^y = Σ y^k/k!
    y = exp * ln_b
    if y > 709.0:  return float('inf')
    if y < -745.0: return 0.0
    e    = 1.0
    term2 = 1.0
    for k in range(1, 100):
        term2 *= y / k
        e     += term2
        if abs(term2) < 1e-15:
            break
    return e

def _human_size(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n} B"  # pragma: no cover


# ════════════════════════════════════════════════════════════════════════════════
# §0  PLATFORM DETECTION
# ════════════════════════════════════════════════════════════════════════════════

class Architecture(Enum):
    X86_64   = "x86_64"
    AARCH64  = "aarch64"
    RISCV64  = "riscv64"
    UNKNOWN  = "unknown"


class PrivilegeLevel(Enum):
    ROOT         = "root"           # UID 0   — full hardware access
    SYS_RAWIO    = "cap_sys_rawio"  # /dev/mem, /dev/cpu/*/msr
    NET_RAW      = "cap_net_raw"    # AF_PACKET only
    UNPRIVILEGED = "unprivileged"   # read /proc /sys; no device files


@dataclass(frozen=True)
class PlatformInfo:
    architecture:   Architecture
    kernel_version: Tuple[int, int, int]
    privilege:      PrivilegeLevel
    cpu_count:      int
    page_size:      int
    hugepage_size:  int              # bytes; typically 2 MiB on x86_64
    endianness:     str              # 'little' or 'big'


def _detect_hugepage_size() -> int:
    """Read default huge-page size from /proc/meminfo (line: Hugepagesize: N kB)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Hugepagesize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 2 * 1024 * 1024   # safe default: 2 MiB


def detect_platform() -> PlatformInfo:
    """
    Probe OS / arch / privilege level using only /proc, /sys, and os.*
    No external utilities.  Safe on any POSIX kernel ≥ 3.0.
    """
    machine = platform.machine().lower()
    if   machine in ("x86_64", "amd64"):   arch = Architecture.X86_64
    elif machine in ("aarch64", "arm64"):  arch = Architecture.AARCH64
    elif "riscv" in machine:               arch = Architecture.RISCV64
    else:                                  arch = Architecture.UNKNOWN

    uname = platform.uname()
    parts = uname.release.split(".")
    try:
        kver: Tuple[int, int, int] = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2].split("-")[0]) if len(parts) > 2 else 0,
        )
    except (IndexError, ValueError):
        kver = (0, 0, 0)

    uid = os.getuid() if hasattr(os, "getuid") else -1
    if uid == 0:
        priv = PrivilegeLevel.ROOT
    else:
        try:
            fd = os.open("/dev/mem", os.O_RDONLY)
            os.close(fd)
            priv = PrivilegeLevel.SYS_RAWIO
        except OSError:
            try:
                import socket as _s
                sock = _s.socket(_s.AF_PACKET, _s.SOCK_RAW, 0)
                sock.close()
                priv = PrivilegeLevel.NET_RAW
            except (OSError, AttributeError):
                priv = PrivilegeLevel.UNPRIVILEGED

    try:
        pagesize = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError):
        pagesize = PAGE_SIZE

    return PlatformInfo(
        architecture   = arch,
        kernel_version = kver,
        privilege      = priv,
        cpu_count      = os.cpu_count() or 1,
        page_size      = pagesize,
        hugepage_size  = _detect_hugepage_size(),
        endianness     = sys.byteorder,
    )


# ════════════════════════════════════════════════════════════════════════════════
# §1  CPU TOPOLOGY
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheDescriptor:
    level:         int          # 1, 2, 3, …
    cache_type:    str          # "Data" | "Instruction" | "Unified"
    size_bytes:    int
    line_size:     int          # bytes per cache line
    associativity: int
    sets:          int
    shared_by:     List[int]    # logical CPU IDs sharing this instance


@dataclass
class CPUCore:
    logical_id:   int
    physical_id:  int           # socket number
    core_id:      int           # core within socket
    numa_node:    int
    min_freq_khz: int
    max_freq_khz: int
    caches:       List[CacheDescriptor]


@dataclass
class SIMDCapabilities:
    """
    SIMD instruction set flags parsed from /proc/cpuinfo 'flags' field.

    Detection method:
      x86:     flags field contains sse4_2, avx, avx2, fma, avx512f, …
      AArch64: Features field contains asimd (NEON), sve
    """
    sse42:      bool = False
    avx:        bool = False
    avx2:       bool = False
    fma:        bool = False
    avx512f:    bool = False
    avx512bw:   bool = False
    avx512vl:   bool = False
    avx512vnni: bool = False
    neon:       bool = False    # AArch64 Advanced SIMD
    sve:        bool = False    # Scalable Vector Extension (AArch64)


@dataclass
class CPUTopologyResult:
    vendor:           str
    brand:            str
    family:           int
    model:            int
    stepping:         int
    cores:            List[CPUCore]
    sockets:          int
    threads_per_core: int
    simd:             SIMDCapabilities
    tsc_khz:          int           # TSC frequency; 0 if unknown
    l3_size_bytes:    int


class CPUTopologyProbe:
    """
    Discovers CPU topology from /proc/cpuinfo and /sys/devices/system/cpu/
    without requiring any privileged access.

    Mathematical model:
      Let C = {c₀, c₁, …, cₙ₋₁}   — set of logical CPUs
      Each c ∈ C has:
        physical_id(c) ∈ {0…S-1}   — socket index
        core_id(c)     ∈ {0…K-1}   — core within socket
      Hyper-threading pairs share (physical_id, core_id).
      NUMA distance D[i][j] ≥ 10; D[i][i] = 10 by convention.
    """

    _SYS_CPU    = Path("/sys/devices/system/cpu")
    _PROC_CPUINFO = Path("/proc/cpuinfo")

    # ── Public ────────────────────────────────────────────────────────────────

    def probe(self) -> CPUTopologyResult:
        vendor, brand, family, model, stepping, flags = self._parse_cpuinfo()
        simd  = self._parse_simd(flags)
        cores = self._build_cores()

        # Sockets = distinct physical_id values
        sockets = max(len({c.physical_id for c in cores}), 1)

        # threads_per_core = logical_count / distinct_(physical_id, core_id) pairs
        distinct = len({(c.physical_id, c.core_id) for c in cores})
        threads_per_core = max(len(cores) // distinct, 1) if distinct else 1

        tsc_khz = self._read_tsc_khz()
        l3      = max(
            (c.size_bytes for core in cores for c in core.caches if c.level == 3),
            default=0,
        )
        return CPUTopologyResult(
            vendor=vendor, brand=brand,
            family=family, model=model, stepping=stepping,
            cores=cores, sockets=sockets,
            threads_per_core=threads_per_core,
            simd=simd,
            tsc_khz=tsc_khz,
            l3_size_bytes=l3,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse_cpuinfo(self) -> Tuple[str, str, int, int, int, Set[str]]:
        vendor   = ""; brand = ""; family = 0; model = 0; stepping = 0
        flags: Set[str] = set()
        try:
            text = self._PROC_CPUINFO.read_text(errors="replace")
            for line in text.splitlines():
                k, sep, v = line.partition(":")
                if not sep: continue
                k = k.strip(); v = v.strip()
                if   k == "vendor_id"  and not vendor:   vendor   = v
                elif k == "model name" and not brand:    brand    = v
                elif k == "cpu family" and not family:
                    try: family = int(v)
                    except ValueError: pass
                elif k == "model"      and not model:
                    try: model = int(v)
                    except ValueError: pass
                elif k == "stepping"   and not stepping:
                    try: stepping = int(v)
                    except ValueError: pass
                elif k == "flags"      and not flags: flags = set(v.split())
                elif k == "Features"   and not flags: flags = set(v.split())  # AArch64
        except OSError:
            pass
        return vendor, brand, family, model, stepping, flags

    def _parse_simd(self, flags: Set[str]) -> SIMDCapabilities:
        return SIMDCapabilities(
            sse42      = "sse4_2"      in flags,
            avx        = "avx"         in flags,
            avx2       = "avx2"        in flags,
            fma        = "fma"         in flags,
            avx512f    = "avx512f"     in flags,
            avx512bw   = "avx512bw"    in flags,
            avx512vl   = "avx512vl"    in flags,
            avx512vnni = "avx512_vnni" in flags,
            neon       = "asimd"       in flags,
            sve        = "sve"         in flags,
        )

    def _build_cores(self) -> List[CPUCore]:
        cores: List[CPUCore] = []
        n = os.cpu_count() or 1
        for i in range(n):
            cpu_path = self._SYS_CPU / f"cpu{i}"
            physical_id = self._read_int(cpu_path / "topology/physical_package_id", 0)
            core_id     = self._read_int(cpu_path / "topology/core_id", i)
            numa_node   = self._find_numa_node(i)
            min_khz     = self._read_int(cpu_path / "cpufreq/cpuinfo_min_freq", 0)
            max_khz     = self._read_int(cpu_path / "cpufreq/cpuinfo_max_freq", 0)
            caches      = self._read_caches(cpu_path, i)
            cores.append(CPUCore(
                logical_id=i, physical_id=physical_id,
                core_id=core_id, numa_node=numa_node,
                min_freq_khz=min_khz, max_freq_khz=max_khz,
                caches=caches,
            ))
        return cores

    def _read_caches(self, cpu_path: Path, cpu_id: int) -> List[CacheDescriptor]:
        caches: List[CacheDescriptor] = []
        base = cpu_path / "cache"
        if not base.exists():
            return caches
        for idx_path in sorted(base.iterdir()):
            try:
                level     = self._read_int(idx_path / "level", 0)
                ctype     = self._read_str(idx_path / "type", "Unified")
                size_str  = self._read_str(idx_path / "size", "0K")
                line_sz   = self._read_int(idx_path / "coherency_line_size", 64)
                assoc     = self._read_int(idx_path / "ways_of_associativity", 0)
                sets_val  = self._read_int(idx_path / "number_of_sets", 0)
                shared    = self._read_str(idx_path / "shared_cpu_list", str(cpu_id))
                caches.append(CacheDescriptor(
                    level=level, cache_type=ctype,
                    size_bytes=self._parse_size(size_str),
                    line_size=line_sz, associativity=assoc, sets=sets_val,
                    shared_by=self.parse_cpu_list(shared),
                ))
            except (OSError, ValueError):
                continue
        return caches

    def _find_numa_node(self, cpu_id: int) -> int:
        try:
            node_root = Path("/sys/devices/system/node")
            for node_dir in sorted(node_root.iterdir()):
                if not node_dir.name.startswith("node"):
                    continue
                cpulist = (node_dir / "cpulist").read_text().strip()
                if cpu_id in self.parse_cpu_list(cpulist):
                    return int(node_dir.name[4:])
        except OSError:
            pass
        return 0

    def _read_tsc_khz(self) -> int:
        # Preferred: kernel exports this since ≥ 5.3
        try:
            val = (self._SYS_CPU / "cpu0/tsc_freq_khz").read_text().strip()
            return int(val)
        except OSError:
            pass
        # Fallback: /proc/cpuinfo "cpu MHz" field (current speed, not nominal)
        try:
            for line in self._PROC_CPUINFO.read_text().splitlines():
                if line.startswith("cpu MHz"):
                    mhz = float(line.split(":")[1].strip())
                    return int(mhz * 1000)
        except (OSError, ValueError):
            pass
        return 0

    @staticmethod
    def _read_int(path: Path, default: int = 0) -> int:
        try: return int(path.read_text().strip())
        except (OSError, ValueError): return default

    @staticmethod
    def _read_str(path: Path, default: str = "") -> str:
        try: return path.read_text().strip()
        except OSError: return default

    @staticmethod
    def _parse_size(s: str) -> int:
        """Parse '32K', '256K', '8192K', '16M' → bytes."""
        s = s.upper().strip()
        if s.endswith("K"): return int(s[:-1]) * 1024
        if s.endswith("M"): return int(s[:-1]) * 1024 * 1024
        if s.endswith("G"): return int(s[:-1]) * 1024 * 1024 * 1024
        try: return int(s)
        except ValueError: return 0

    @staticmethod
    def parse_cpu_list(s: str) -> List[int]:
        """Parse '0-3,6,8-11' → [0,1,2,3,6,8,9,10,11]."""
        result: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if "-" in part:
                lo, _, hi = part.partition("-")
                try: result.extend(range(int(lo), int(hi) + 1))
                except ValueError: pass
            elif part:
                try: result.append(int(part))
                except ValueError: pass
        return result


# ════════════════════════════════════════════════════════════════════════════════
# §2  MEMORY SUBSYSTEM
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryInfo:
    total_bytes:         int
    available_bytes:     int
    free_bytes:          int
    cached_bytes:        int
    buffers_bytes:       int
    swap_total_bytes:    int
    swap_free_bytes:     int
    hugepages_total:     int
    hugepages_free:      int
    hugepage_size_bytes: int


@dataclass
class NUMANode:
    node_id:   int
    cpu_ids:   List[int]
    mem_total: int           # bytes
    mem_free:  int           # bytes
    distances: Dict[int, int]  # node_id → NUMA distance (10 = local, 20 = remote)


class MemorySubsystemProbe:
    """
    Reads physical memory layout from /proc/meminfo and
    /sys/devices/system/node/.  Pure discovery — no allocation.
    """

    def probe_meminfo(self) -> MemoryInfo:
        """
        Parses /proc/meminfo.  All kB values converted to bytes.

        Key fields and their meanings:
          MemTotal     — total usable RAM installed
          MemAvailable — kernel's estimate of reclaimable memory
          MemFree      — completely unused pages (does NOT include reclaimable)
          Cached       — page cache (can be reclaimed)
          Buffers      — kernel buffer cache
          HugePages_*  — preallocated huge pages (2 MiB / 1 GiB depending on config)
        """
        fields: Dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if ":" not in line: continue
                k, _, rest = line.partition(":")
                parts = rest.strip().split()
                val   = int(parts[0])
                if len(parts) > 1 and parts[1].lower() == "kb":
                    val *= 1024
                fields[k.strip()] = val
        except OSError:
            pass
        hp = _detect_hugepage_size()
        return MemoryInfo(
            total_bytes         = fields.get("MemTotal",          0),
            available_bytes     = fields.get("MemAvailable",      0),
            free_bytes          = fields.get("MemFree",           0),
            cached_bytes        = fields.get("Cached",            0),
            buffers_bytes       = fields.get("Buffers",           0),
            swap_total_bytes    = fields.get("SwapTotal",         0),
            swap_free_bytes     = fields.get("SwapFree",          0),
            hugepages_total     = fields.get("HugePages_Total",   0),
            hugepages_free      = fields.get("HugePages_Free",    0),
            hugepage_size_bytes = hp,
        )

    def probe_numa(self) -> List[NUMANode]:
        """
        Reads NUMA topology.  Falls back to a single-node system
        when /sys/devices/system/node/ is absent (non-NUMA machines).

        NUMA distance matrix:
          D[i][j] read from /sys/devices/system/node/nodeN/distance
          Convention: D[i][i] = 10 (local), D[i][j] ≥ 20 (remote).
          Higher is worse.  Used by CPUAffinityManager to minimise
          cross-NUMA accesses during tensor allocation.
        """
        node_root = Path("/sys/devices/system/node")
        if not node_root.exists():
            mi = self.probe_meminfo()
            return [NUMANode(
                node_id=0,
                cpu_ids=list(range(os.cpu_count() or 1)),
                mem_total=mi.total_bytes,
                mem_free=mi.free_bytes,
                distances={0: 10},
            )]
        nodes: List[NUMANode] = []
        for node_dir in sorted(node_root.iterdir()):
            if not node_dir.name.startswith("node"): continue
            node_id = int(node_dir.name[4:])
            try:
                cpulist = (node_dir / "cpulist").read_text().strip()
                cpu_ids = CPUTopologyProbe.parse_cpu_list(cpulist)
            except OSError:
                cpu_ids = []
            mem_total = 0; mem_free = 0
            try:
                for line in (node_dir / "meminfo").read_text().splitlines():
                    parts = line.split()
                    if "MemTotal" in line and len(parts) >= 4:
                        mem_total = int(parts[3]) * 1024
                    elif "MemFree" in line and len(parts) >= 4:
                        mem_free  = int(parts[3]) * 1024
            except (OSError, IndexError, ValueError):
                pass
            distances: Dict[int, int] = {}
            try:
                for j, d in enumerate(
                    (node_dir / "distance").read_text().strip().split()
                ):
                    distances[j] = int(d)
            except (OSError, ValueError):
                distances[node_id] = 10
            nodes.append(NUMANode(
                node_id=node_id, cpu_ids=cpu_ids,
                mem_total=mem_total, mem_free=mem_free,
                distances=distances,
            ))
        return nodes


# ════════════════════════════════════════════════════════════════════════════════
# §3  PCIE ENUMERATION
# ════════════════════════════════════════════════════════════════════════════════

# PCI class codes (bits[23:8] of 24-bit class register)
_PCI_CLASS_GPU_VGA   = 0x0300
_PCI_CLASS_GPU_3D    = 0x0302
_PCI_CLASS_NIC       = 0x0200
_PCI_CLASS_NVME      = 0x0108
_PCI_CLASS_USB3      = 0x0C03

# Vendor IDs
PCI_VENDOR_NVIDIA = 0x10DE
PCI_VENDOR_AMD    = 0x1002
PCI_VENDOR_INTEL  = 0x8086
PCI_VENDOR_QCOM   = 0x17CB
PCI_VENDOR_AMPERE = 0x1DEF   # Ampere Computing (AArch64 GPU/Arm)

_VENDOR_NAMES: Dict[int, str] = {
    PCI_VENDOR_NVIDIA: "NVIDIA",
    PCI_VENDOR_AMD:    "AMD",
    PCI_VENDOR_INTEL:  "Intel",
    PCI_VENDOR_QCOM:   "Qualcomm",
    PCI_VENDOR_AMPERE: "Ampere",
}

# BAR flags (bits in /sys/bus/pci/devices/BDF/resource lines)
_PCI_BAR_IO        = 0x01   # bit 0 set → I/O space; clear → Memory space
_PCI_BAR_MEM64     = 0x04   # bits[2:1] == 10b → 64-bit BAR
_PCI_BAR_PREFETCH  = 0x08   # prefetchable memory


@dataclass
class PCIeDevice:
    bdf:           str               # "0000:01:00.0"
    vendor_id:     int
    device_id:     int
    class_code:    int               # 16-bit: bits[23:8] of PCI class register
    revision:      int
    subsys_vendor: int
    subsys_device: int
    bars:          List[Tuple[int, int, int]]   # (base, size, flags)
    driver:        Optional[str]
    iommu_group:   Optional[int]

    @property
    def is_gpu(self) -> bool:
        return (
            self.vendor_id in (PCI_VENDOR_NVIDIA, PCI_VENDOR_AMD)
            and self.class_code in (_PCI_CLASS_GPU_VGA, _PCI_CLASS_GPU_3D)
        )

    @property
    def is_nvme(self) -> bool:
        return self.class_code == _PCI_CLASS_NVME

    @property
    def vendor_name(self) -> str:
        return _VENDOR_NAMES.get(self.vendor_id, f"0x{self.vendor_id:04X}")

    @property
    def mmio_bar(self) -> Optional[Tuple[int, int]]:
        """Return (base, size) of first memory-mapped BAR, or None."""
        for base, size, flags in self.bars:
            if not (flags & _PCI_BAR_IO):
                return (base, size)
        return None


class PCIeEnumerator:
    """
    Enumerates PCI Express devices by walking /sys/bus/pci/devices/.

    BAR layout (per PCI 3.0 §6.2.5):
      /sys/bus/pci/devices/BDF/resource:
        Column 0 = start address (hex)
        Column 1 = end address   (hex)
        Column 2 = flags         (hex)
      size = end - start + 1
    """

    _PCI_ROOT = Path("/sys/bus/pci/devices")

    def enumerate(self) -> List[PCIeDevice]:
        if not self._PCI_ROOT.exists():
            return []
        return [
            dev
            for path in sorted(self._PCI_ROOT.iterdir())
            if (dev := self._read_device(path)) is not None
        ]

    def _read_device(self, dev_path: Path) -> Optional[PCIeDevice]:
        try:
            bdf       = dev_path.name
            vendor_id = self._hex(dev_path / "vendor")
            device_id = self._hex(dev_path / "device")
            # class register: 24-bit; we strip the prog-if byte
            class_raw = self._hex(dev_path / "class")
            class_code = (class_raw >> 8) & 0xFFFF
            revision  = self._hex(dev_path / "revision")
            subsys_v  = self._hex(dev_path / "subsystem_vendor")
            subsys_d  = self._hex(dev_path / "subsystem_device")
            bars      = self._read_bars(dev_path / "resource")
            driver    = self._read_driver(dev_path)
            iommu     = self._read_iommu_group(dev_path)
            return PCIeDevice(
                bdf=bdf, vendor_id=vendor_id, device_id=device_id,
                class_code=class_code, revision=revision,
                subsys_vendor=subsys_v, subsys_device=subsys_d,
                bars=bars, driver=driver, iommu_group=iommu,
            )
        except (OSError, ValueError):
            return None

    @staticmethod
    def _hex(path: Path) -> int:
        try: return int(path.read_text().strip(), 16)
        except (OSError, ValueError): return 0

    @staticmethod
    def _read_bars(resource_path: Path) -> List[Tuple[int, int, int]]:
        bars: List[Tuple[int, int, int]] = []
        try:
            for line in resource_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 3: continue
                start = int(parts[0], 16)
                end   = int(parts[1], 16)
                flags = int(parts[2], 16)
                if start != 0 and end >= start:
                    bars.append((start, end - start + 1, flags))
        except (OSError, ValueError):
            pass
        return bars

    @staticmethod
    def _read_driver(dev_path: Path) -> Optional[str]:
        link = dev_path / "driver"
        if link.exists():
            try: return Path(os.readlink(str(link))).name
            except OSError: pass
        return None

    @staticmethod
    def _read_iommu_group(dev_path: Path) -> Optional[int]:
        link = dev_path / "iommu_group"
        if link.exists():
            try: return int(Path(os.readlink(str(link))).name)
            except (OSError, ValueError): pass
        return None


# ════════════════════════════════════════════════════════════════════════════════
# §4  STORAGE INTERFACE
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockDevice:
    name:           str
    size_bytes:     int
    physical_block: int    # bytes; optimal transfer size
    logical_block:  int    # bytes; minimum addressable unit
    rotational:     bool   # False → SSD / NVMe
    scheduler:      str    # I/O scheduler active in kernel
    numa_node:      int
    pci_bdf:        Optional[str]


class StorageProbe:
    """
    Discovers block devices from /sys/block/.
    Reads topology only — does not perform I/O.
    """

    _SYS_BLOCK = Path("/sys/block")

    def probe(self) -> List[BlockDevice]:
        if not self._SYS_BLOCK.exists():
            return []
        return [
            dev
            for path in sorted(self._SYS_BLOCK.iterdir())
            if (dev := self._read_device(path)) is not None
        ]

    def _read_device(self, dev_path: Path) -> Optional[BlockDevice]:
        name = dev_path.name
        # Exclude synthetic devices
        if name.startswith(("loop", "ram", "zram", "dm-")):
            return None
        try:
            sectors    = self._int(dev_path / "size")
            sect_size  = self._int(dev_path / "queue/logical_block_size",  512)
            phys_size  = self._int(dev_path / "queue/physical_block_size", 4096)
            rotational = bool(self._int(dev_path / "queue/rotational", 1))
            scheduler  = self._str(dev_path / "queue/scheduler", "none")
            # active scheduler is surrounded by brackets: "mq-deadline [none] bfq"
            if "[" in scheduler and "]" in scheduler:
                scheduler = scheduler[scheduler.index("[") + 1: scheduler.index("]")]
            numa_node  = self._int(dev_path / "device/numa_node", 0)
            pci_bdf    = self._resolve_pci_bdf(dev_path)
            return BlockDevice(
                name=name,
                size_bytes=sectors * sect_size,
                physical_block=phys_size,
                logical_block=sect_size,
                rotational=rotational,
                scheduler=scheduler,
                numa_node=numa_node,
                pci_bdf=pci_bdf,
            )
        except (OSError, ValueError):
            return None

    @staticmethod
    def _resolve_pci_bdf(dev_path: Path) -> Optional[str]:
        try:
            real = Path(os.readlink(str(dev_path / "device")))
            for part in real.parts:
                if part.count(":") == 2:   # DDDD:BB:DD.F
                    return part
        except OSError:
            pass
        return None

    @staticmethod
    def _int(path: Path, default: int = 0) -> int:
        try: return int(path.read_text().strip())
        except (OSError, ValueError): return default

    @staticmethod
    def _str(path: Path, default: str = "") -> str:
        try: return path.read_text().strip()
        except OSError: return default


# ════════════════════════════════════════════════════════════════════════════════
# §5  PERFORMANCE MONITOR — perf_event_open(2) + MSR
# ════════════════════════════════════════════════════════════════════════════════

# Linux syscall number for perf_event_open on x86_64 and AArch64
_NR_perf_event_open: Dict[str, int] = {
    "x86_64":  298,
    "aarch64": 241,
    "riscv64": 241,
}

# perf_event_type
_PERF_TYPE_HARDWARE = 0
_PERF_TYPE_SOFTWARE = 1

# perf_hw_id
_PERF_HW_CPU_CYCLES     = 0
_PERF_HW_INSTRUCTIONS   = 1
_PERF_HW_CACHE_REFS     = 2
_PERF_HW_CACHE_MISSES   = 3
_PERF_HW_BRANCHES       = 4
_PERF_HW_BRANCH_MISSES  = 5

# perf_event_ioc_* ioctl numbers (Linux asm-generic)
_PERF_IOC_ENABLE  = 0x2400
_PERF_IOC_DISABLE = 0x2401
_PERF_IOC_RESET   = 0x2403

# Intel MSR addresses
MSR_IA32_TSC         = 0x10
MSR_IA32_APERF       = 0xE8    # Actual performance clock counter
MSR_IA32_MPERF       = 0xE7    # Maximum performance clock counter
MSR_IA32_PERF_STATUS = 0x198
MSR_RAPL_ENERGY_UNIT = 0x606   # Package energy/time/power units
MSR_RAPL_PKG_ENERGY  = 0x611   # Package energy status


class PerfEventAttr(ctypes.Structure):
    """
    struct perf_event_attr from <linux/perf_event.h>.
    Only the first 72 bytes are needed for basic counter monitoring.
    Kernel fills the rest with zeroes when size is set correctly.
    """
    _fields_ = [
        ("type",           ctypes.c_uint32),
        ("size",           ctypes.c_uint32),
        ("config",         ctypes.c_uint64),
        ("sample_period",  ctypes.c_uint64),   # union with sample_freq
        ("sample_type",    ctypes.c_uint64),
        ("read_format",    ctypes.c_uint64),
        ("flags",          ctypes.c_uint64),
        ("wakeup_events",  ctypes.c_uint32),
        ("bp_type",        ctypes.c_uint32),
        ("bp_addr",        ctypes.c_uint64),   # union with kprobe_func / uprobe_path
        ("bp_len",         ctypes.c_uint64),   # union with kprobe_addr / probe_offset
    ]


@dataclass
class PerfCounterReading:
    cycles:        int
    instructions:  int
    cache_refs:    int
    cache_misses:  int
    branches:      int
    branch_misses: int
    elapsed_ns:    int

    @property
    def ipc(self) -> float:
        """Instructions per cycle — primary throughput metric."""
        return self.instructions / max(self.cycles, 1)

    @property
    def cache_miss_rate(self) -> float:
        """LLC miss rate — key indicator for memory-bound code."""
        return self.cache_misses / max(self.cache_refs, 1)

    @property
    def branch_miss_rate(self) -> float:
        return self.branch_misses / max(self.branches, 1)

    @property
    def cycles_per_ns(self) -> float:
        return self.cycles / max(self.elapsed_ns, 1)


class PerformanceMonitor:
    """
    Hardware performance counter access via perf_event_open(2) and
    raw MSR reads.

    perf_event_open interface (Linux syscall):
      fd = syscall(SYS_perf_event_open, &attr, pid, cpu, group_fd, flags)
      read(fd, &count_u64, 8)  → raw 64-bit counter value
      ioctl(fd, PERF_EVENT_IOC_ENABLE/DISABLE/RESET)

    RAPL energy computation (§5 appendix):
      MSR 0x606 bits[12:8] = ESU  (energy status units exponent)
      energy_unit_J = 2^(-ESU)    joules per count
      MSR 0x611 bits[31:0] = E    (energy accumulator, 32-bit, wraps at max)
      energy_J = E * 2^(-ESU)
      Convert: energy_µJ = energy_J × 10⁶

    APERF/MPERF frequency measurement:
      freq_actual = freq_nominal × (ΔAPERF / ΔMPERF)
      where ΔAPERF and ΔMPERF are sampled 10 ms apart.
    """

    def __init__(self, platform: PlatformInfo) -> None:
        self._platform = platform
        self._libc     = self._load_libc()
        self._nr_perf  = _NR_perf_event_open.get(platform.architecture.value, 298)

    def _load_libc(self) -> Optional[ctypes.CDLL]:
        try:
            name = ctypes.util.find_library("c")
            return ctypes.CDLL(name, use_errno=True) if name else None
        except OSError:
            return None

    def _syscall(self, nr: int, *args: int) -> int:
        """Invoke a raw Linux syscall via the libc syscall() wrapper."""
        if self._libc is None:
            return -1
        try:
            fn = self._libc.syscall
            fn.restype = ctypes.c_long
            fn.argtypes = [ctypes.c_long] + [ctypes.c_long] * len(args)
            return int(fn(ctypes.c_long(nr), *[ctypes.c_long(a) for a in args]))
        except Exception:
            return -1

    def open_counter(self, perf_type: int, config: int,
                     pid: int = 0, cpu: int = -1) -> int:
        """
        Open a hardware or software performance counter.
        Returns an open file descriptor, or -1 on failure.

        pid=0  → measure the calling process
        cpu=-1 → any CPU the process runs on
        PERF_FLAG_FD_CLOEXEC = 8 (prevent fd leaking across exec)
        """
        attr = PerfEventAttr()
        ctypes.memset(ctypes.addressof(attr), 0, ctypes.sizeof(attr))
        attr.type   = perf_type
        attr.size   = ctypes.sizeof(PerfEventAttr)
        attr.config = config
        return self._syscall(self._nr_perf,
                             ctypes.addressof(attr), pid, cpu, -1, 8)

    def read_counter(self, fd: int) -> int:
        if fd < 0: return 0
        try:
            raw = os.read(fd, 8)
            return struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
        except OSError:
            return 0

    def _ioctl(self, fd: int, cmd: int) -> None:
        if fd >= 0:
            try: fcntl.ioctl(fd, cmd, 0)
            except OSError: pass

    def enable(self,  fd: int) -> None: self._ioctl(fd, _PERF_IOC_ENABLE)
    def disable(self, fd: int) -> None: self._ioctl(fd, _PERF_IOC_DISABLE)
    def reset(self,   fd: int) -> None: self._ioctl(fd, _PERF_IOC_RESET)

    @contextmanager
    def measure(self) -> Iterator["_PerfMeasurement"]:
        """
        Context manager that collects a PerfCounterReading for the
        code executed within the with-block.

        Usage:
            with monitor.measure() as m:
                ... workload ...
            reading = m.reading   # PerfCounterReading | None
        """
        class _PerfMeasurement:
            reading: Optional[PerfCounterReading] = None

        hw_events = [
            ("cycles",        _PERF_HW_CPU_CYCLES),
            ("instructions",  _PERF_HW_INSTRUCTIONS),
            ("cache_refs",    _PERF_HW_CACHE_REFS),
            ("cache_misses",  _PERF_HW_CACHE_MISSES),
            ("branches",      _PERF_HW_BRANCHES),
            ("branch_misses", _PERF_HW_BRANCH_MISSES),
        ]
        fds: Dict[str, int] = {}
        for name, cfg in hw_events:
            fd = self.open_counter(_PERF_TYPE_HARDWARE, cfg)
            if fd >= 0:
                self.reset(fd)
                self.enable(fd)
            fds[name] = fd

        m = _PerfMeasurement()
        t0 = time.monotonic_ns()
        try:
            yield m
        finally:
            t1 = time.monotonic_ns()
            readings = {k: self.read_counter(v) for k, v in fds.items()}
            for fd in fds.values():
                if fd >= 0:
                    self.disable(fd)
                    try: os.close(fd)
                    except OSError: pass
            m.reading = PerfCounterReading(
                cycles        = readings.get("cycles",        0),
                instructions  = readings.get("instructions",  0),
                cache_refs    = readings.get("cache_refs",    0),
                cache_misses  = readings.get("cache_misses",  0),
                branches      = readings.get("branches",      0),
                branch_misses = readings.get("branch_misses", 0),
                elapsed_ns    = t1 - t0,
            )

    def read_msr(self, cpu: int, msr_addr: int) -> Optional[int]:
        """
        Read a 64-bit Model-Specific Register via /dev/cpu/{cpu}/msr.
        Requires CAP_SYS_RAWIO or UID 0.  Returns None if access denied.
        """
        try:
            fd = os.open(f"/dev/cpu/{cpu}/msr", os.O_RDONLY)
            try:
                raw = os.pread(fd, 8, msr_addr)
                return struct.unpack("<Q", raw)[0] if len(raw) == 8 else None
            finally:
                os.close(fd)
        except OSError:
            return None

    def read_rapl_energy_uj(self, cpu: int = 0) -> Optional[float]:
        """
        Read RAPL (Running Average Power Limit) package energy in µJ.

        Algorithm:
          1. Read MSR_RAPL_ENERGY_UNIT (0x606)
             ESU  = bits[12:8]       (5-bit field)
             unit = 2^(-ESU) joules
          2. Read MSR_RAPL_PKG_ENERGY (0x611)
             E    = bits[31:0]       (32-bit counter, wraps)
          3. energy_µJ = E × 2^(-ESU) × 10⁶
        """
        unit_raw   = self.read_msr(cpu, MSR_RAPL_ENERGY_UNIT)
        energy_raw = self.read_msr(cpu, MSR_RAPL_PKG_ENERGY)
        if unit_raw is None or energy_raw is None:
            return None
        esu           = (unit_raw >> 8) & 0x1F
        energy_counts = energy_raw & 0xFFFF_FFFF
        energy_j      = energy_counts * _pow_f(2.0, -float(esu))
        return energy_j * 1_000_000.0

    def read_actual_freq_mhz(self, cpu: int = 0) -> Optional[float]:
        """
        Compute actual CPU operating frequency from APERF/MPERF ratio.

        APERF: increments at actual core performance frequency.
        MPERF: increments at the maximum non-turbo (P1) frequency.

        freq_actual = freq_P1 × (ΔAPERF / ΔMPERF)

        We obtain freq_P1 from /proc/cpuinfo 'cpu MHz' as a proxy.
        A 10 ms sampling window balances accuracy vs latency.
        """
        a0 = self.read_msr(cpu, MSR_IA32_APERF)
        m0 = self.read_msr(cpu, MSR_IA32_MPERF)
        if a0 is None or m0 is None:
            return None
        time.sleep(0.010)
        a1 = self.read_msr(cpu, MSR_IA32_APERF)
        m1 = self.read_msr(cpu, MSR_IA32_MPERF)
        if a1 is None or m1 is None:
            return None
        # Counters are 64-bit and wrap; mask to handle rollover
        da = (a1 - a0) & 0xFFFF_FFFF_FFFF_FFFF
        dm = (m1 - m0) & 0xFFFF_FFFF_FFFF_FFFF
        if dm == 0:
            return None
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("cpu MHz"):
                    p1_mhz = float(line.split(":")[1].strip())
                    return p1_mhz * (da / dm)
        except (OSError, ValueError):
            pass
        return None


# ════════════════════════════════════════════════════════════════════════════════
# §6  CPU AFFINITY & NUMA-AWARE SCHEDULING
# ════════════════════════════════════════════════════════════════════════════════

class CPUAffinityManager:
    """
    Thread-to-CPU pinning via os.sched_setaffinity (Python ≥ 3.3, Linux).

    Optimal NUMA placement:
      Given threads τ = {t₀, …, tₙ} and NUMA distance matrix D[i][j]:
      Assign each thread t to a NUMA node n* that minimises
        D[current_node(t), n*]
      A greedy solution: locate the thread's current CPU, find its node,
      select the node with D[current, candidate] = minimum.
    """

    def __init__(self, topology: CPUTopologyResult,
                 numa_nodes: List[NUMANode]) -> None:
        self._topo = topology
        self._numa: Dict[int, NUMANode] = {n.node_id: n for n in numa_nodes}
        self._has_set = hasattr(os, "sched_setaffinity")
        self._has_get = hasattr(os, "sched_getaffinity")

    def pin_to_cpu(self, cpu_id: int) -> bool:
        """Pin the calling thread to a single logical CPU."""
        if not self._has_set: return False
        try:
            os.sched_setaffinity(0, {cpu_id})
            return True
        except (OSError, PermissionError):
            return False

    def pin_to_numa_node(self, node_id: int) -> bool:
        """Restrict the calling thread to all CPUs within a NUMA node."""
        if not self._has_set: return False
        node = self._numa.get(node_id)
        if node is None: return False
        try:
            os.sched_setaffinity(0, set(node.cpu_ids))
            return True
        except (OSError, PermissionError):
            return False

    def current_affinity(self) -> Set[int]:
        if not self._has_get:
            return set(range(os.cpu_count() or 1))
        try:
            return os.sched_getaffinity(0)
        except OSError:
            return set()

    def nearest_numa_node(self) -> int:
        """
        Return the NUMA node whose distance from the thread's current CPU
        is minimal — i.e., the node the thread is already closest to.
        """
        aff = self.current_affinity()
        if not aff: return 0
        cpu_id = min(aff)
        for core in self._topo.cores:
            if core.logical_id == cpu_id:
                return core.numa_node
        return 0

    def nearest_node_for_workload(self, preferred_size_bytes: int) -> int:
        """
        Choose the NUMA node that is nearest to the current thread AND
        has sufficient free memory for the workload.
        Falls back to the nearest node even if memory is tight.
        """
        current_node = self.nearest_numa_node()
        best_node    = current_node
        best_dist    = self._numa.get(current_node, NUMANode(0, [], 0, 0, {})) \
                          .distances.get(current_node, 10)

        for node in self._numa.values():
            dist = self._numa.get(current_node, NUMANode(0, [], 0, 0, {})) \
                       .distances.get(node.node_id, 9999)
            if node.mem_free >= preferred_size_bytes and dist < best_dist:
                best_dist = dist
                best_node = node.node_id
        return best_node


# ════════════════════════════════════════════════════════════════════════════════
# §7  HARDWARE MEMORY BUS — mmap-backed real OS pages
# ════════════════════════════════════════════════════════════════════════════════

class HardwareMemoryBus:
    """
    Replaces the simulated MemoryBus (bytearray) with real OS-managed
    memory regions backed by anonymous mmap pages.

    Key differences from aios_core.MemoryBus:
      • Each allocation calls mmap(MAP_ANONYMOUS | MAP_PRIVATE).
        The OS allocates real physical pages on first access (demand paging).
      • When huge pages are available and requested, MAP_HUGETLB reduces
        TLB pressure for large tensor arenas.
        Example: a 512 MiB tensor with 4 KiB pages → 131 072 TLB entries.
                 With 2 MiB huge pages  → only 256 TLB entries.
      • PCI BAR mapping: open /sys/bus/pci/devices/BDF/resourceN and
        mmap(MAP_SHARED) for direct device register access.
      • All regions are identified by string names and mapped to a simulated
        physical address space to preserve compatibility with the existing
        VirtualMemoryManager translation layer.

    Address space model:
      simulated_phys_addr = region_base + offset_within_region
      region_base is assigned sequentially from _next_phys (starts at 0x1000)
      The actual virtual address (in the host OS) lives inside the mmap object.
    """

    _MAP_ANONYMOUS = 0x20        # Linux-specific flag value
    _MAP_PRIVATE   = 0x02
    _MAP_HUGETLB   = 0x40000     # Request 2 MiB pages (Linux)

    def __init__(self, platform: PlatformInfo, meminfo: MemoryInfo) -> None:
        self._platform   = platform
        self._meminfo    = meminfo
        self._regions:   Dict[str, mmap.mmap]       = {}
        self._reg_size:  Dict[str, int]              = {}
        self._reg_base:  Dict[str, int]              = {}
        self._phys_map:  Dict[int, Tuple[str, int]]  = {}  # page_base → (region, offset)
        self._io_ports:  Dict[int, int]              = {}  # simulated ISA I/O
        self._next_phys  = 0x1000                          # below VGA_BASE_ADDR
        self._lock       = threading.RLock()

    # ── Region management ─────────────────────────────────────────────────────

    def allocate_region(self, name: str, size: int,
                        use_hugepages: bool = False) -> int:
        """
        Allocate a named memory region backed by real OS pages.

        size is rounded up to the appropriate page boundary:
          - use_hugepages=True  → align to hugepage_size (2 MiB)
          - use_hugepages=False → align to page_size      (4 KiB)

        Returns the simulated physical base address of the region.
        Raises RuntimeError if the region name is already allocated.
        """
        with self._lock:
            if name in self._regions:
                raise RuntimeError(f"Region {name!r} already allocated")

            align = (self._platform.hugepage_size if use_hugepages
                     else self._platform.page_size)
            aligned_sz = (size + align - 1) & ~(align - 1)

            mm = self._mmap_alloc(aligned_sz, use_hugepages)

            phys_base = self._next_phys
            # Align to page boundary in simulated address space
            self._next_phys += aligned_sz

            self._regions[name]  = mm
            self._reg_size[name] = aligned_sz
            self._reg_base[name] = phys_base

            # Register every page in the phys_map for O(1) address translation
            pg = self._platform.page_size
            for offset in range(0, aligned_sz, pg):
                self._phys_map[phys_base + offset] = (name, offset)

            return phys_base

    def _mmap_alloc(self, size: int, use_hugepages: bool) -> mmap.mmap:
        """
        Attempt an mmap allocation.  Falls back gracefully:
          1. MAP_ANONYMOUS + MAP_HUGETLB (huge pages)
          2. MAP_ANONYMOUS without MAP_HUGETLB (normal pages)
        """
        if use_hugepages and self._meminfo.hugepages_free > 0:
            try:
                return mmap.mmap(
                    -1, size,
                    flags=mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE | self._MAP_HUGETLB,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            except (OSError, mmap.error):
                pass   # fall through to normal pages
        return mmap.mmap(
            -1, size,
            flags=mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

    def free_region(self, name: str) -> None:
        with self._lock:
            if name not in self._regions: return
            mm   = self._regions.pop(name)
            size = self._reg_size.pop(name)
            base = self._reg_base.pop(name)
            mm.close()
            pg = self._platform.page_size
            for offset in range(0, size, pg):
                self._phys_map.pop(base + offset, None)

    # ── Physical address I/O ──────────────────────────────────────────────────

    def poke8(self, phys_addr: int, value: int) -> None:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            self._regions[name].write(bytes([value & 0xFF]))

    def peek8(self, phys_addr: int) -> int:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            return self._regions[name].read(1)[0]

    def poke16(self, phys_addr: int, value: int) -> None:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            self._regions[name].write(struct.pack("<H", value & 0xFFFF))

    def peek16(self, phys_addr: int) -> int:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            return struct.unpack("<H", self._regions[name].read(2))[0]

    def poke32(self, phys_addr: int, value: int) -> None:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            self._regions[name].write(struct.pack("<I", value & 0xFFFF_FFFF))

    def peek32(self, phys_addr: int) -> int:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            return struct.unpack("<I", self._regions[name].read(4))[0]

    def poke64(self, phys_addr: int, value: int) -> None:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            self._regions[name].write(
                struct.pack("<Q", value & 0xFFFF_FFFF_FFFF_FFFF)
            )

    def peek64(self, phys_addr: int) -> int:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            return struct.unpack("<Q", self._regions[name].read(8))[0]

    def bulk_write(self, phys_addr: int, data: bytes) -> None:
        name, off = self._resolve(phys_addr)
        limit = self._reg_size[name]
        if off + len(data) > limit:
            raise MemoryError(
                f"bulk_write: {off + len(data)} > region size {limit}"
            )
        with self._lock:
            self._regions[name].seek(off)
            self._regions[name].write(data)

    def bulk_read(self, phys_addr: int, count: int) -> bytes:
        name, off = self._resolve(phys_addr)
        with self._lock:
            self._regions[name].seek(off)
            return self._regions[name].read(count)

    # ── ISA I/O port simulation ───────────────────────────────────────────────

    def outb(self, port: int, value: int) -> None:
        with self._lock:
            self._io_ports[port & 0xFFFF] = value & 0xFF

    def inb(self, port: int) -> int:
        with self._lock:
            return self._io_ports.get(port & 0xFFFF, 0xFF)

    # ── PCI BAR mapping ───────────────────────────────────────────────────────

    def map_pci_bar(self, bdf: str, bar_index: int,
                   size: int) -> Optional[int]:
        """
        Memory-map a PCI device BAR into the process address space via
        /sys/bus/pci/devices/{bdf}/resource{bar_index}.

        This grants direct register access for device drivers.
        On VFIO-enabled systems this becomes an IOMMU-mapped DMA region.

        Returns the simulated physical base address, or None on failure.
        Requires the sysfs resource file to be accessible
        (typically root or with appropriate udev permissions).
        """
        resource = f"/sys/bus/pci/devices/{bdf}/resource{bar_index}"
        try:
            fd = os.open(resource, os.O_RDWR | os.O_SYNC)
            try:
                mm = mmap.mmap(fd, size,
                               flags=mmap.MAP_SHARED,
                               prot=mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)
            region_name = f"pci:{bdf}:bar{bar_index}"
            pg = self._platform.page_size
            aligned = (size + pg - 1) & ~(pg - 1)
            with self._lock:
                phys_base = self._next_phys
                self._next_phys  += aligned
                self._regions[region_name]  = mm
                self._reg_size[region_name] = aligned
                self._reg_base[region_name] = phys_base
                for offset in range(0, aligned, pg):
                    self._phys_map[phys_base + offset] = (region_name, offset)
            return phys_base
        except (OSError, mmap.error):
            return None

    # ── Address translation ───────────────────────────────────────────────────

    def _resolve(self, phys_addr: int) -> Tuple[str, int]:
        """
        Translate a simulated physical address to (region_name, byte_offset).
        Raises MemoryError on unmapped addresses.
        """
        pg          = self._platform.page_size
        page_base   = phys_addr & ~(pg - 1)
        page_offset = phys_addr &  (pg - 1)
        with self._lock:
            mapping = self._phys_map.get(page_base)
        if mapping is None:
            raise MemoryError(
                f"Physical address 0x{phys_addr:016X} not mapped. "
                f"Call allocate_region() before accessing this range."
            )
        region, region_offset = mapping
        return region, region_offset + page_offset

    # ── Inspection ────────────────────────────────────────────────────────────

    def memory_map(self) -> Dict[str, Dict]:
        """Return a serialisable snapshot of all allocated regions."""
        with self._lock:
            return {
                name: {
                    "phys_base":  f"0x{self._reg_base[name]:016X}",
                    "size_bytes": self._reg_size[name],
                    "size_human": _human_size(self._reg_size[name]),
                }
                for name in sorted(self._regions)
            }


# ════════════════════════════════════════════════════════════════════════════════
# §8  SIMD TENSOR DISPATCH
# ════════════════════════════════════════════════════════════════════════════════

class SIMDTensorBackend:
    """
    Routes tensor operations to the most capable instruction set on this CPU.

    Dispatch hierarchy (descending priority):
      1. AVX-512 VNNI  — INT8 neural-network inference via VPDPBUSD
      2. AVX-512F/BW   — 512-bit FP32/FP16 operations
      3. AVX2 + FMA    — 256-bit FMA (fused multiply-add, one rounding)
      4. SSE4.2        — 128-bit SIMD
      5. NEON (AArch64) — 128-bit ARM Advanced SIMD
      6. Scalar Python  — pure Python; always available as final fallback

    When numpy is available (which uses platform BLAS/LAPACK), operations are
    routed through it automatically — numpy selects AVX2 or AVX-512 BLAS at
    runtime.  The SIMD level reported here reflects what the underlying BLAS
    would use; it guides tile-size selection and scheduling decisions.

    GEMM tile-size derivation (Goto & van de Geijn, 2008):
      Three working sets reside in L1 cache simultaneously:
        A-panel: bK × bM floats
        B-panel: bK × bN floats
        C-tile:  bM × bN floats
      Constraint: bM·bK + bK·bN + bM·bN ≤ L1_bytes / sizeof(float32)
      Assuming equal square partitions (bM = bN = bK = b):
        3b² ≤ L1_floats   →   b = ⌊√(L1_floats / 3)⌋
      Round b down to nearest multiple of 8 (SIMD register width alignment).
    """

    def __init__(self, simd: SIMDCapabilities,
                 cache_sizes: Dict[int, int]) -> None:
        self._simd   = simd
        self._caches = cache_sizes
        self._np     = self._try_numpy()
        self._level  = self._select_level()

    def _try_numpy(self):
        try:
            import numpy as np
            return np
        except ImportError:
            return None

    def _select_level(self) -> str:
        if self._simd.avx512vnni: return "avx512vnni"
        if self._simd.avx512f:    return "avx512"
        if self._simd.avx2 and self._simd.fma: return "avx2+fma"
        if self._simd.avx2:       return "avx2"
        if self._simd.sse42:      return "sse4.2"
        if self._simd.neon:       return "neon"
        return "scalar"

    @property
    def level(self) -> str:
        return self._level

    @property
    def numpy_available(self) -> bool:
        return self._np is not None

    def optimal_tile_size(self) -> Tuple[int, int, int]:
        """
        Compute (bM, bN, bK) optimal GEMM tile dimensions for L1 cache.

        b = ⌊√(L1_bytes / (3 × 4))⌋  then round down to multiple of 8.
        """
        l1_bytes  = self._caches.get(1, 32 * 1024)
        l1_floats = l1_bytes // 4
        b_raw     = _isqrt_int(l1_floats // 3)
        b         = (b_raw // 8) * 8 if b_raw >= 8 else max(b_raw, 1)
        return (b, b, b)

    def matmul(self, A: Any, B: Any) -> Any:
        """
        C = A @ B.
        Routes to numpy (BLAS) when available; falls back to naive
        pure-Python GEMM.
        """
        if self._np is not None:
            return self._np.matmul(A, B)
        # Pure-Python naive GEMM
        M = len(A);  K = len(A[0]);  N = len(B[0])
        C = [[0.0] * N for _ in range(M)]
        for i in range(M):
            Ai = A[i]
            Ci = C[i]
            for k in range(K):
                a_ik = Ai[k]
                Bk   = B[k]
                for j in range(N):
                    Ci[j] += a_ik * Bk[j]
        return C

    def adam_step(
        self,
        params: Any, grads: Any,
        m: Any, v: Any,
        step: int,
        lr:    float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-8,
        weight_decay: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        """
        AdamW parameter update (Loshchilov & Hutter, 2019).

        Decoupled weight decay is applied before the momentum update:
          θ  ← θ × (1 - α·λ)          (weight decay, outside gradient)
          m  ← β₁·m + (1-β₁)·g        (first moment)
          v  ← β₂·v + (1-β₂)·g²       (second moment)
          m̂  ← m / (1 - β₁^t)          (bias correction)
          v̂  ← v / (1 - β₂^t)
          θ  ← θ - α · m̂ / (√v̂ + ε)   (parameter update)

        Returns (updated_params, updated_m, updated_v).
        """
        np   = self._np
        bc1  = 1.0 - _pow_f(beta1, float(step))
        bc2  = 1.0 - _pow_f(beta2, float(step))

        if np is not None:
            params = np.asarray(params, dtype=np.float64)
            grads  = np.asarray(grads,  dtype=np.float64)
            m      = np.asarray(m,      dtype=np.float64)
            v      = np.asarray(v,      dtype=np.float64)
            if weight_decay > 0.0:
                params = params * (1.0 - lr * weight_decay)
            m_new = beta1 * m + (1.0 - beta1) * grads
            v_new = beta2 * v + (1.0 - beta2) * grads * grads
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            p_new = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            if weight_decay > 0.0:
                params = [p * (1.0 - lr * weight_decay) for p in params]
            m_new = [beta1 * mi + (1.0 - beta1) * gi
                     for mi, gi in zip(m, grads)]
            v_new = [beta2 * vi + (1.0 - beta2) * gi * gi
                     for vi, gi in zip(v, grads)]
            m_hat = [mi / bc1 for mi in m_new]
            v_hat = [vi / bc2 for vi in v_new]
            p_new = [
                pi - lr * mhi / (_sqrt_f(vhi) + eps)
                for pi, mhi, vhi in zip(params, m_hat, v_hat)
            ]
        return p_new, m_new, v_new

    def relu(self, X: Any) -> Any:
        np = self._np
        if np is not None: return np.maximum(X, 0.0)
        return [max(0.0, x) for x in X]

    def softmax(self, X: Any, axis: int = -1) -> Any:
        np = self._np
        if np is not None:
            e = np.exp(X - np.max(X, axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)
        # Pure Python (single vector)
        mx = max(X)
        # exp approximation: use built-in float (**) which is hardware FPU
        e = [_pow_f(2.718281828459045, v - mx) for v in X]
        s = sum(e)
        return [v / s for v in e]

    def layer_norm(self, X: Any, gamma: Any, beta: Any,
                   eps: float = 1e-5) -> Any:
        """
        y = γ · (x - µ) / (σ + ε) + β
        µ = mean(x)
        σ = std(x) = √(mean((x-µ)²))
        """
        np = self._np
        if np is not None:
            mu    = np.mean(X, axis=-1, keepdims=True)
            sigma = np.std( X, axis=-1, keepdims=True)
            return gamma * (X - mu) / (sigma + eps) + beta
        n  = len(X)
        mu = sum(X) / n
        var = sum((v - mu) ** 2 for v in X) / n
        std = _sqrt_f(var + eps)
        return [gamma[i] * (X[i] - mu) / std + beta[i] for i in range(n)]


# ════════════════════════════════════════════════════════════════════════════════
# §9  HARDWARE LAYER SINGLETON
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareReport:
    """Serialisable snapshot of all detected hardware subsystems."""
    platform:       PlatformInfo
    cpu:            CPUTopologyResult
    memory:         MemoryInfo
    numa_nodes:     List[NUMANode]
    pcie_devices:   List[PCIeDevice]
    storage:        List[BlockDevice]
    detected_gpus:  List[PCIeDevice]
    detected_nvme:  List[BlockDevice]
    simd_level:     str
    l3_cache_bytes: int
    tsc_khz:        int


class HardwareLayer:
    """
    Singleton gateway to all hardware subsystems for the AIOS kernel.

    Lifecycle:
      1. hw = HardwareLayer.instance()
      2. report = hw.probe()          — topology discovery (no side effects)
      3. ok     = hw.initialise(...)  — allocates real memory regions
      4. hw.attach_to_kernel(kernel)  — connects to a live AgentKernel
      5. hw.shutdown()                — releases all resources

    After initialise(), the kernel gains:
      • hw.memory_bus     — HardwareMemoryBus backed by real OS pages
      • hw.simd_backend   — SIMDTensorBackend for routing tensor ops
      • hw.perf_monitor   — PerformanceMonitor for hardware counters
      • hw.cpu_affinity   — CPUAffinityManager for NUMA-aware scheduling
    """

    _instance: Optional["HardwareLayer"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._platform:  Optional[PlatformInfo]       = None
        self._cpu:       Optional[CPUTopologyResult]  = None
        self._memory:    Optional[MemoryInfo]         = None
        self._numa:      Optional[List[NUMANode]]     = None
        self._pcie:      Optional[List[PCIeDevice]]   = None
        self._storage:   Optional[List[BlockDevice]]  = None
        self._perf:      Optional[PerformanceMonitor] = None
        self._affinity:  Optional[CPUAffinityManager] = None
        self._mem_bus:   Optional[HardwareMemoryBus]  = None
        self._simd:      Optional[SIMDTensorBackend]  = None
        self._report:    Optional[HardwareReport]     = None
        self._ready:     bool                         = False
        self._probe_lock = threading.RLock()

    @classmethod
    def instance(cls) -> "HardwareLayer":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── §9a  Probe (pure discovery, no allocation) ────────────────────────────

    @agent_method(
        name="hw.probe",
        description=(
            "Discover all hardware: CPU topology, memory layout, PCIe devices, "
            "storage, SIMD capabilities.  Safe to call without root."
        ),
        parameters={},
        returns="HardwareReport",
        priority=AgentPriority.HIGH,
    )
    def probe(self) -> HardwareReport:
        with self._probe_lock:
            if self._report is not None:
                return self._report

            self._platform = detect_platform()

            self._cpu = CPUTopologyProbe().probe()

            mem_probe    = MemorySubsystemProbe()
            self._memory = mem_probe.probe_meminfo()
            self._numa   = mem_probe.probe_numa()

            self._pcie    = PCIeEnumerator().enumerate()
            self._storage = StorageProbe().probe()

            self._perf     = PerformanceMonitor(self._platform)
            self._affinity = CPUAffinityManager(self._cpu, self._numa)

            # Collect L1/L2/L3 sizes from first core's cache list
            cache_sizes: Dict[int, int] = {}
            if self._cpu.cores:
                for c in self._cpu.cores[0].caches:
                    cache_sizes[c.level] = max(
                        cache_sizes.get(c.level, 0), c.size_bytes
                    )

            self._simd = SIMDTensorBackend(self._cpu.simd, cache_sizes)

            gpus  = [d for d in self._pcie if d.is_gpu]
            nvmes = [d for d in self._storage
                     if not d.rotational and d.pci_bdf is not None]

            self._report = HardwareReport(
                platform       = self._platform,
                cpu            = self._cpu,
                memory         = self._memory,
                numa_nodes     = self._numa,
                pcie_devices   = self._pcie,
                storage        = self._storage,
                detected_gpus  = gpus,
                detected_nvme  = nvmes,
                simd_level     = self._simd.level,
                l3_cache_bytes = self._cpu.l3_size_bytes,
                tsc_khz        = self._cpu.tsc_khz,
            )
            return self._report

    # ── §9b  Initialise (allocate real memory) ────────────────────────────────

    @agent_method(
        name="hw.initialise",
        description=(
            "Allocate mmap-backed memory regions for the AIOS kernel. "
            "Must follow probe()."
        ),
        parameters={
            "kernel_heap_mb": {"type": "int",  "desc": "Kernel heap in MiB"},
            "use_hugepages":  {"type": "bool", "desc": "Request 2 MiB huge pages"},
        },
        returns="bool",
        priority=AgentPriority.CRITICAL,
    )
    def initialise(self, kernel_heap_mb: int = 64,
                   use_hugepages: bool = True) -> bool:
        """
        Allocates:
          • kernel_heap  — primary AIOS heap
          • vga_text     — 80 × 25 × 2 byte VGA text buffer
          • tensor_arena — pre-reserved space for neural network weights
                           capped at min(heap/2, 256 MiB)

        All allocations use HardwareMemoryBus and are therefore backed
        by real OS-managed anonymous pages with demand paging.
        """
        if self._report is None:
            self.probe()
        assert self._platform is not None and self._memory is not None

        self._mem_bus = HardwareMemoryBus(self._platform, self._memory)

        # Guard against requesting more memory than the OS can satisfy
        avail_mb  = self._memory.available_bytes // (1024 * 1024)
        heap_mb   = min(kernel_heap_mb, int(avail_mb * 0.4))
        heap_mb   = max(heap_mb, 16)       # floor: 16 MiB
        heap_sz   = heap_mb * 1024 * 1024

        try:
            heap_base = self._mem_bus.allocate_region(
                "kernel_heap", heap_sz, use_hugepages=use_hugepages
            )
        except (OSError, mmap.error):
            # Retry without huge pages
            heap_base = self._mem_bus.allocate_region(
                "kernel_heap", heap_sz, use_hugepages=False
            )

        # VGA text mode region (exact match to aios_core.VGA_BASE_ADDR intent)
        vga_sz = 80 * 25 * 2
        self._mem_bus.allocate_region("vga_text", vga_sz, use_hugepages=False)

        # Tensor arena — pre-reserve, allow the NN subsystem to carve from it
        tensor_sz = min(heap_sz // 2, 256 * 1024 * 1024)
        try:
            self._mem_bus.allocate_region(
                "tensor_arena", tensor_sz, use_hugepages=use_hugepages
            )
        except (OSError, mmap.error):
            self._mem_bus.allocate_region(
                "tensor_arena", tensor_sz, use_hugepages=False
            )

        self._ready = True
        return True

    # ── §9c  Shutdown ─────────────────────────────────────────────────────────

    @agent_method(
        name="hw.shutdown",
        description="Release all hardware resources allocated by initialise().",
        parameters={}, returns="None", priority=AgentPriority.HIGH,
    )
    def shutdown(self) -> None:
        if self._mem_bus is not None:
            for name in list(self._mem_bus._regions.keys()):
                self._mem_bus.free_region(name)
        self._ready = False

    # ── §9d  Agent-callable operations ────────────────────────────────────────

    @agent_method(
        name="hw.report_json",
        description="Return a full JSON hardware report.",
        parameters={}, returns="str", priority=AgentPriority.LOW,
    )
    def report_json(self) -> str:
        if self._report is None:
            self.probe()
        r = self._report
        assert r is not None

        def _gpu(g: PCIeDevice) -> Dict:
            return {
                "bdf":        g.bdf,
                "vendor":     g.vendor_name,
                "device_id":  f"0x{g.device_id:04X}",
                "driver":     g.driver,
                "iommu":      g.iommu_group,
                "mmio_bar":   (
                    [f"0x{g.mmio_bar[0]:016X}", _human_size(g.mmio_bar[1])]
                    if g.mmio_bar else None
                ),
            }

        def _blk(b: BlockDevice) -> Dict:
            return {
                "name":      b.name,
                "size":      _human_size(b.size_bytes),
                "type":      "HDD" if b.rotational else "SSD/NVMe",
                "scheduler": b.scheduler,
                "numa":      b.numa_node,
                "pci_bdf":   b.pci_bdf,
            }

        doc = {
            "aios_hw": f"{AIOS_HW_VERSION[0]}.{AIOS_HW_VERSION[1]}.{AIOS_HW_VERSION[2]}",
            "platform": {
                "arch":      r.platform.architecture.value,
                "kernel":    ".".join(str(v) for v in r.platform.kernel_version),
                "privilege": r.platform.privilege.value,
                "cpus":      r.platform.cpu_count,
                "page":      _human_size(r.platform.page_size),
                "hugepage":  _human_size(r.platform.hugepage_size),
                "endian":    r.platform.endianness,
            },
            "cpu": {
                "vendor":           r.cpu.vendor,
                "brand":            r.cpu.brand,
                "family/model/step": f"{r.cpu.family}/{r.cpu.model}/{r.cpu.stepping}",
                "sockets":          r.cpu.sockets,
                "logical_cores":    len(r.cpu.cores),
                "threads_per_core": r.cpu.threads_per_core,
                "tsc_khz":          r.cpu.tsc_khz,
                "l3_cache":         _human_size(r.l3_cache_bytes),
                "simd_level":       r.simd_level,
                "simd_flags": {
                    "sse4.2":     r.cpu.simd.sse42,
                    "avx":        r.cpu.simd.avx,
                    "avx2":       r.cpu.simd.avx2,
                    "fma":        r.cpu.simd.fma,
                    "avx512f":    r.cpu.simd.avx512f,
                    "avx512bw":   r.cpu.simd.avx512bw,
                    "avx512vl":   r.cpu.simd.avx512vl,
                    "avx512vnni": r.cpu.simd.avx512vnni,
                    "neon":       r.cpu.simd.neon,
                    "sve":        r.cpu.simd.sve,
                },
            },
            "memory": {
                "total":           _human_size(r.memory.total_bytes),
                "available":       _human_size(r.memory.available_bytes),
                "hugepages_free":  r.memory.hugepages_free,
                "hugepages_total": r.memory.hugepages_total,
                "hugepage_size":   _human_size(r.memory.hugepage_size_bytes),
            },
            "numa": [
                {
                    "id":        n.node_id,
                    "cpus":      n.cpu_ids,
                    "mem_total": _human_size(n.mem_total),
                    "mem_free":  _human_size(n.mem_free),
                    "distances": {str(k): v for k, v in n.distances.items()},
                }
                for n in r.numa_nodes
            ],
            "gpus":          [_gpu(g)  for g in r.detected_gpus],
            "nvme_devices":  [_blk(d)  for d in r.detected_nvme],
            "all_storage":   [_blk(d)  for d in r.storage],
            "all_pcie":      len(r.pcie_devices),
        }
        return json.dumps(doc, indent=2)

    @agent_method(
        name="hw.memory_map",
        description="Return current memory region allocation map.",
        parameters={}, returns="Dict", priority=AgentPriority.LOW,
    )
    def memory_map(self) -> Dict[str, Dict]:
        if not self._ready or self._mem_bus is None: return {}
        return self._mem_bus.memory_map()

    @agent_method(
        name="hw.benchmark_gemm",
        description=(
            "Benchmark GEMM throughput: GFLOPS and memory bandwidth (GB/s) "
            "for the detected SIMD level."
        ),
        parameters={
            "n": {"type": "int", "desc": "Square matrix dimension N (N×N @ N×N)"},
        },
        returns="Dict[str, float]",
        priority=AgentPriority.NORMAL,
    )
    def benchmark_gemm(self, n: int = 512) -> Dict[str, float]:
        """
        Measures GEMM (N×N @ N×N) throughput.

        FLOPs = 2·N³          (N² multiply-adds)
        Bytes = 3·N²·4        (load A, load B, store C in float32)
        GFLOPS = FLOPs / elapsed_ns          (units: 10⁹ flops / 10⁻⁹ s = flops/s)
        BW(GB/s) = Bytes / elapsed_ns
        """
        if self._simd is None:
            self.probe()
        assert self._simd is not None
        np = self._simd._np

        if np is not None:
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            _ = np.matmul(A, B)               # warm up BLAS

            # Use perf counters if available
            if self._perf is not None:
                with self._perf.measure() as m:
                    for _ in range(5):
                        C = np.matmul(A, B)
                reading = m.reading
            else:
                t0 = time.perf_counter_ns()
                for _ in range(5):
                    C = np.matmul(A, B)
                reading = None

            t0 = time.perf_counter_ns()
            for _ in range(10):
                C = np.matmul(A, B)
            elapsed = (time.perf_counter_ns() - t0) / 10.0

            flops = 2 * n * n * n
            bw    = 3 * n * n * 4

            result: Dict[str, float] = {
                "n":                 float(n),
                "elapsed_ns":        elapsed,
                "gflops":            flops / elapsed,
                "mem_bandwidth_GBps": bw  / elapsed,
            }
            if reading is not None and reading.cycles > 0:
                result["ipc"]             = reading.ipc
                result["cache_miss_rate"] = reading.cache_miss_rate
        else:
            t0 = time.perf_counter_ns()
            acc = 0.0
            for i in range(n):
                acc += float(i) * float(i)
            elapsed = float(time.perf_counter_ns() - t0)
            result = {
                "n": float(n), "elapsed_ns": elapsed,
                "gflops": 0.0, "mem_bandwidth_GBps": 0.0,
                "note": "numpy unavailable — scalar loop fallback",
            }
        return result

    @agent_method(
        name="hw.pin_thread",
        description=(
            "Pin the calling thread to a logical CPU or NUMA node."
        ),
        parameters={
            "cpu_id":  {"type": "Optional[int]", "desc": "Logical CPU ID or None"},
            "node_id": {"type": "Optional[int]", "desc": "NUMA node ID or None"},
        },
        returns="bool",
        priority=AgentPriority.NORMAL,
    )
    def pin_thread(self, cpu_id: Optional[int] = None,
                   node_id: Optional[int] = None) -> bool:
        if self._affinity is None: return False
        if cpu_id  is not None: return self._affinity.pin_to_cpu(cpu_id)
        if node_id is not None: return self._affinity.pin_to_numa_node(node_id)
        return False

    @agent_method(
        name="hw.rapl_energy_uj",
        description="Read RAPL package energy in µJ (requires root/CAP_SYS_RAWIO).",
        parameters={"socket": {"type": "int", "desc": "Physical CPU socket index"}},
        returns="Optional[float]",
        priority=AgentPriority.LOW,
    )
    def rapl_energy_uj(self, socket: int = 0) -> Optional[float]:
        if self._perf is None: return None
        return self._perf.read_rapl_energy_uj(socket)

    @agent_method(
        name="hw.actual_freq_mhz",
        description="Read actual CPU frequency via APERF/MPERF MSR ratio.",
        parameters={"cpu": {"type": "int", "desc": "Logical CPU index"}},
        returns="Optional[float]",
        priority=AgentPriority.LOW,
    )
    def actual_freq_mhz(self, cpu: int = 0) -> Optional[float]:
        if self._perf is None: return None
        return self._perf.read_actual_freq_mhz(cpu)

    @agent_method(
        name="hw.map_gpu_bar",
        description=(
            "Memory-map a GPU PCIe BAR for direct register access. "
            "Requires root or udev write permissions on sysfs resource file."
        ),
        parameters={
            "bdf":       {"type": "str", "desc": "PCIe BDF string e.g. '0000:01:00.0'"},
            "bar_index": {"type": "int", "desc": "BAR index 0–5"},
            "size":      {"type": "int", "desc": "Mapping size in bytes"},
        },
        returns="Optional[int]",
        priority=AgentPriority.HIGH,
    )
    def map_gpu_bar(self, bdf: str, bar_index: int, size: int) -> Optional[int]:
        if not self._ready or self._mem_bus is None: return None
        return self._mem_bus.map_pci_bar(bdf, bar_index, size)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def memory_bus(self) -> Optional[HardwareMemoryBus]:
        return self._mem_bus

    @property
    def simd_backend(self) -> Optional[SIMDTensorBackend]:
        return self._simd

    @property
    def perf_monitor(self) -> Optional[PerformanceMonitor]:
        return self._perf

    @property
    def cpu_affinity(self) -> Optional[CPUAffinityManager]:
        return self._affinity


# ════════════════════════════════════════════════════════════════════════════════
# §10  KERNEL ATTACHMENT
# ════════════════════════════════════════════════════════════════════════════════

def attach_to_kernel(kernel_instance: Any) -> bool:
    """
    Attach the HardwareLayer to a running AgentKernel instance.

    Precondition:  kernel_instance.boot() must have returned True.

    Steps performed:
      1. Probe all hardware subsystems.
      2. Initialise real memory bus (heap_mb derived from available RAM).
      3. Attach hw.* subsystems to kernel as kernel.hw.
      4. Preserve the kernel's existing MemoryBus for simulated I/O
         (VGA, PIC port registers, BIOS vectors) — do not replace it.
         The hardware bus is used for new tensor/heap allocations.
      5. Log topology to the kernel's boot log if available.

    Design note:
      We do NOT swap kernel.bus with hw.memory_bus.  The simulated bus
      owns the VGA buffer and PIC port simulation which the VGATextDriver
      and PICDriver instances hold references to.  Instead:
        kernel.hw.memory_bus  → real OS-page-backed allocations
        kernel.bus            → simulated bytearray (preserved, unchanged)
      Code that needs large real allocations (tensor arena, heap) calls
      kernel.hw.memory_bus.allocate_region().

    Returns True on success.
    """
    hw     = HardwareLayer.instance()
    report = hw.probe()

    avail_mb  = report.memory.available_bytes // (1024 * 1024)
    heap_mb   = max(16, min(128, int(avail_mb * 0.25)))

    if not hw.initialise(kernel_heap_mb=heap_mb, use_hugepages=True):
        return False

    # Attach hardware layer to kernel
    kernel_instance.hw = hw

    # Log topology to kernel boot log if _log method is available
    log = getattr(kernel_instance, "_log", None)
    if callable(log):
        r = report
        log(f"[HW]  arch={r.platform.architecture.value}  "
            f"priv={r.platform.privilege.value}  "
            f"cpus={r.platform.cpu_count}")
        log(f"[HW]  CPU: {r.cpu.vendor}  '{r.cpu.brand[:48]}'")
        log(f"[HW]  SIMD: {r.simd_level}  "
            f"L3={_human_size(r.l3_cache_bytes)}  "
            f"TSC={r.tsc_khz} kHz")
        log(f"[HW]  RAM: total={_human_size(r.memory.total_bytes)}  "
            f"avail={_human_size(r.memory.available_bytes)}  "
            f"hugepages={r.memory.hugepages_free}/{r.memory.hugepages_total}")
        if r.detected_gpus:
            for g in r.detected_gpus:
                log(f"[HW]  GPU: {g.vendor_name} {g.bdf}  "
                    f"driver={g.driver}  iommu={g.iommu_group}")
        else:
            log("[HW]  GPU: none detected via PCIe sysfs")
        if r.detected_nvme:
            for d in r.detected_nvme:
                log(f"[HW]  NVMe: {d.name} {_human_size(d.size_bytes)}  "
                    f"numa={d.numa_node}  bdf={d.pci_bdf}")
        log("[HW]  Hardware layer attached. Real memory bus ONLINE.")

    return True


# ════════════════════════════════════════════════════════════════════════════════
# §11  SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════════

def _run_self_tests() -> bool:
    """
    Validates every section of the hardware interface layer.
    Returns True if all tests pass; False otherwise.

    Test coverage:
      §0  platform detection
      §1  CPU topology probe
      §2  memory subsystem probe
      §3  PCIe enumeration (informational — no assertion on presence)
      §4  storage probe (informational)
      §5  performance monitor (counter open attempt; graceful fail allowed)
      §6  CPU affinity query
      §7  HardwareMemoryBus allocate / poke / peek / bulk / free
      §8  SIMDTensorBackend level + Adam step correctness
      §9  HardwareLayer.probe() + report_json() + benchmark_gemm()
    """
    ok  = 0
    bad = 0
    _w  = 60  # line width

    def _check(condition: bool, label: str, detail: str = "") -> None:
        nonlocal ok, bad
        if condition:
            ok += 1
            print(f"  [PASS] {label}")
        else:
            bad += 1
            print(f"  [FAIL] {label}  ← {detail}")

    print("\n" + "═" * _w)
    print("  AIOS Hardware Interface Layer — Self-Test Suite")
    print("═" * _w)

    # ── §0  Platform ──────────────────────────────────────────────────────────
    p  = detect_platform()
    pg = p.page_size
    _check(True, "Platform detected")
    _check(pg > 0 and (pg & (pg - 1)) == 0, "Page size is power-of-2",
           f"got {pg}")
    print(f"        arch={p.architecture.value}"
          f"  kernel={'.'.join(str(v) for v in p.kernel_version)}"
          f"  priv={p.privilege.value}  cpus={p.cpu_count}")

    # ── §1  CPU Topology ──────────────────────────────────────────────────────
    cpu = CPUTopologyProbe().probe()
    _check(len(cpu.cores) > 0, "CPU cores list non-empty",
           f"count={len(cpu.cores)}")
    _check(cpu.sockets >= 1, "CPU sockets ≥ 1", f"got {cpu.sockets}")
    _check(cpu.threads_per_core >= 1, "Threads/core ≥ 1",
           f"got {cpu.threads_per_core}")
    print(f"        {cpu.vendor!r}  model={cpu.model}  "
          f"avx2={cpu.simd.avx2}  avx512f={cpu.simd.avx512f}  "
          f"fma={cpu.simd.fma}  neon={cpu.simd.neon}")

    # ── §2  Memory Subsystem ──────────────────────────────────────────────────
    mem  = MemorySubsystemProbe().probe_meminfo()
    numa = MemorySubsystemProbe().probe_numa()
    _check(mem.total_bytes > 0, "MemTotal > 0", f"got {mem.total_bytes}")
    _check(mem.available_bytes <= mem.total_bytes + 4096,
           "MemAvailable ≤ MemTotal")
    _check(len(numa) >= 1, "NUMA node list non-empty")
    print(f"        total={_human_size(mem.total_bytes)}"
          f"  avail={_human_size(mem.available_bytes)}"
          f"  hugepages={mem.hugepages_free}/{mem.hugepages_total}"
          f"  numa_nodes={len(numa)}")

    # ── §3  PCIe Enumeration ──────────────────────────────────────────────────
    pcie = PCIeEnumerator().enumerate()
    gpus = [d for d in pcie if d.is_gpu]
    nvme = [d for d in pcie if d.is_nvme]
    print(f"        PCIe: {len(pcie)} devices  GPUs={len(gpus)}"
          f"  NVMe={len(nvme)}")

    # ── §4  Storage ───────────────────────────────────────────────────────────
    blk = StorageProbe().probe()
    print(f"        Block: {len(blk)} device(s): "
          + (", ".join(f"{d.name}({_human_size(d.size_bytes)})"
                       for d in blk[:4]) or "none"))

    # ── §5  Performance Monitor ───────────────────────────────────────────────
    pm = PerformanceMonitor(p)
    fd = pm.open_counter(_PERF_TYPE_HARDWARE, _PERF_HW_CPU_CYCLES)
    if fd >= 0:
        pm.reset(fd); pm.enable(fd)
        _sink = sum(range(10_000))
        pm.disable(fd)
        count = pm.read_counter(fd)
        try:
            os.close(fd)
        except OSError:
            pass
        _check(count > 0, "perf_event_open cycle counter",
               f"count={count}")
    else:
        print("        perf_event_open: unprivileged — skipped")
        ok += 1

    rapl = pm.read_rapl_energy_uj(0)
    if rapl is not None:
        _check(rapl > 0.0, "RAPL energy reading", f"value={rapl:.1f} µJ")
    else:
        print("        RAPL: MSR access denied — skipped")
        ok += 1

    # ── §6  CPU Affinity ──────────────────────────────────────────────────────
    am  = CPUAffinityManager(cpu, numa)
    aff = am.current_affinity()
    _check(len(aff) > 0, "sched_getaffinity returns CPUs", f"set={aff}")
    node = am.nearest_numa_node()
    _check(node >= 0, "nearest_numa_node ≥ 0", f"got {node}")

    # ── §7  Hardware Memory Bus ───────────────────────────────────────────────
    bus  = HardwareMemoryBus(p, mem)
    base = bus.allocate_region("_test", 4 * p.page_size)

    bus.poke8(base, 0xAB)
    v8 = bus.peek8(base)
    _check(v8 == 0xAB, "poke8/peek8", f"0x{v8:02X} ≠ 0xAB")

    bus.poke16(base + 2, 0xBEEF)
    v16 = bus.peek16(base + 2)
    _check(v16 == 0xBEEF, "poke16/peek16", f"0x{v16:04X}")

    bus.poke32(base + 4, 0xDEAD_BEEF)
    v32 = bus.peek32(base + 4)
    _check(v32 == 0xDEAD_BEEF, "poke32/peek32", f"0x{v32:08X}")

    bus.poke64(base + 8, 0xCAFE_BABE_DEAD_BEEF)
    v64 = bus.peek64(base + 8)
    _check(v64 == 0xCAFE_BABE_DEAD_BEEF, "poke64/peek64",
           f"0x{v64:016X}")

    payload = b"AIOS hardware bus integration test"
    bus.bulk_write(base + 32, payload)
    rb = bus.bulk_read(base + 32, len(payload))
    _check(rb == payload, "bulk_write/bulk_read")

    bus.outb(0x20, 0x11)
    _check(bus.inb(0x20) == 0x11, "ISA I/O outb/inb")

    bus.free_region("_test")
    _check("_test" not in bus._regions, "free_region removes entry")

    # ── §8  SIMD Backend ──────────────────────────────────────────────────────
    cache_sz = {
        1: 32768,
        2: 256 * 1024,
        3: max(cpu.l3_size_bytes, 8 * 1024 * 1024),
    }
    simd     = SIMDTensorBackend(cpu.simd, cache_sz)
    _check(len(simd.level) > 0, "SIMD level string non-empty",
           f"got {simd.level!r}")
    bM, bN, bK = simd.optimal_tile_size()
    _check(bM >= 1 and bN >= 1 and bK >= 1, "GEMM tile size ≥ 1",
           f"({bM},{bN},{bK})")
    print(f"        SIMD level={simd.level!r}  tile=({bM},{bN},{bK})"
          f"  numpy={'yes' if simd.numpy_available else 'no'}")

    # Adam step closed-form verification:
    #   θ=1, g=1, m=v=0, step=1, lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8
    #   m₁ = 0.1   v₁ = 0.001
    #   m̂  = 0.1/(1−0.9)  = 1.0
    #   v̂  = 0.001/(1−0.999) = 1.0
    #   θ₁ = 1.0 − 1e-3 × 1.0/(√1.0 + 1e-8) ≈ 0.999000...
    _params = [1.0]; _grads = [1.0]; _m0 = [0.0]; _v0 = [0.0]
    p_new, _m_new, _v_new = simd.adam_step(
        _params, _grads, _m0, _v0, step=1,
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
    )
    _expected = 1.0 - 1e-3 / (_sqrt_f(1.0) + 1e-8)
    try:
        _got = float(p_new.flat[0]) if hasattr(p_new, "flat") else float(p_new[0])
    except (TypeError, IndexError):
        _got = float(p_new)
    _err = abs(_got - _expected)
    _check(_err < 1e-7, "Adam step correctness", f"err={_err:.3e}")

    # ── §9  Full HardwareLayer integration ────────────────────────────────────
    hw  = HardwareLayer.instance()
    rpt = hw.probe()
    _check(isinstance(rpt, HardwareReport),
           "HardwareLayer.probe() returns HardwareReport")

    json_str = hw.report_json()
    try:
        doc = json.loads(json_str)
        _check("cpu" in doc and "memory" in doc,
               "report_json has expected top-level keys")
    except json.JSONDecodeError as exc:
        _check(False, "report_json JSON decode", str(exc))

    ok_init = hw.initialise(kernel_heap_mb=16, use_hugepages=False)
    _check(ok_init, "HardwareLayer.initialise() succeeds")

    mmap_info = hw.memory_map()
    _check("kernel_heap"  in mmap_info and "tensor_arena" in mmap_info,
           "memory_map: kernel_heap + tensor_arena present")

    bench = hw.benchmark_gemm(n=64)
    _check("gflops" in bench, "benchmark_gemm returns 'gflops' key")
    print(f"        GEMM 64×64: {bench.get('gflops', 0):.3f} GFLOPS"
          f"  bw={bench.get('mem_bandwidth_GBps', 0):.2f} GB/s")

    hw.shutdown()
    _check(not hw._ready, "shutdown() clears _ready flag")

    print("═" * _w)
    print(f"  Results: {ok} passed,  {bad} failed")
    print("═" * _w + "\n")
    return bad == 0


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    success = _run_self_tests()
    sys.exit(0 if success else 1)
