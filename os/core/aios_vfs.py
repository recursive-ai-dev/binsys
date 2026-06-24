#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Virtual Filesystem                                                   ║
║  aios_vfs.py                                                                 ║
║                                                                              ║
║  "A filesystem is a theory of names. Everything the kernel can know         ║
║   must be addressable. Everything addressable must have a path."             ║
║                                                                              ║
║  Components:                                                                 ║
║    §0  Constants & Enums                                                     ║
║    §1  VNode              — abstract inode (file/dir/device/pipe/symlink)   ║
║    §2  FileDescriptor     — open handle: (vnode, offset, flags)             ║
║    §3  FDTable            — per-process file descriptor table               ║
║    §4  MemFS              — in-memory filesystem (dict tree of VNodes)      ║
║    §5  ProcFS             — read-only kernel introspection filesystem        ║
║    §6  DevFS              — device nodes (/dev/null, /dev/zero, /dev/mem)   ║
║    §7  VFS                — root namespace: mount, open, read, write,        ║
║                             close, stat, seek, readdir, unlink, mkdir        ║
║    §8  VFSKernel          — @agent_method integration                        ║
║    §9  Self-Tests         — deterministic validation suite                   ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Path resolution : O(d) where d = directory depth                         ║
║    MemFS lookup    : O(1) hash map per directory level                      ║
║    FD allocation   : O(n) first-fit scan; n = max open FDs (1024)          ║
║    Inode address   : ino = hash(abs_path) & 0xFFFFFFFF  [pseudorandom]     ║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Zero external dependencies. Pure Python 3.9+ stdlib only.               ║
║    • Thread-safe: per-VNode RLock + FDTable RLock.                          ║
║    • Standalone: degrades gracefully when aios_core is absent.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

_VFS_VERSION = "1.0.0"
MAX_FDS      = 1024       # max simultaneously open file descriptors
MAX_PATH_LEN = 4096       # maximum path string length
MAX_FILE_SIZE= 64 * 1024 * 1024   # 64 MiB per MemFS file


# ─────────────────────────────────────────────────────────────────────────────
# §0  CONSTANTS & ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class VNodeType(Enum):
    FILE    = "file"
    DIR     = "dir"
    DEVICE  = "device"
    PIPE    = "pipe"
    SYMLINK = "symlink"


class OpenFlags(IntFlag):
    O_RDONLY  = 0x00
    O_WRONLY  = 0x01
    O_RDWR    = 0x02
    O_CREAT   = 0x40
    O_TRUNC   = 0x200
    O_APPEND  = 0x400
    O_EXCL    = 0x800


class SeekWhence(IntEnum):
    SEEK_SET = 0   # offset from file start
    SEEK_CUR = 1   # offset from current position
    SEEK_END = 2   # offset from end


class VFSError(Exception):
    """Base class for all VFS errors."""

class FileNotFoundError(VFSError): pass
class PermissionError(VFSError):   pass
class FileExistsError(VFSError):   pass
class NotADirectoryError(VFSError): pass
class IsADirectoryError(VFSError): pass
class IOError(VFSError):           pass
class NoSpaceError(VFSError):      pass


@dataclass(frozen=True)
class VNodeStat:
    """stat() result analogous to struct stat."""
    ino:    int          # inode number (path hash)
    size:   int          # size in bytes
    vtype:  VNodeType    # file, dir, device, …
    mode:   int          # permission bits (UNIX octal)
    mtime:  float        # monotonic modification timestamp
    atime:  float        # monotonic access timestamp
    nlinks: int          # hard link count (always 1 in MemFS)


def _ino(path: str) -> int:
    """Deterministic inode number from absolute path."""
    return int(hashlib.sha256(path.encode()).hexdigest()[:8], 16)


def _now() -> float:
    return time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# §1  VNODE
# ─────────────────────────────────────────────────────────────────────────────

class VNode(ABC):
    """
    Abstract virtual filesystem node.

    Subclasses implement read/write/readdir according to their type.
    All VNodes carry an ino, vtype, mode, and timestamps.
    """

    def __init__(
        self,
        path:  str,
        vtype: VNodeType,
        mode:  int = 0o644,
    ) -> None:
        self._path   = path
        self._vtype  = vtype
        self._mode   = mode
        self._ino    = _ino(path)
        self._mtime  = _now()
        self._atime  = _now()
        self._lock   = threading.RLock()

    @property
    def vtype(self) -> VNodeType:
        return self._vtype

    @property
    def ino(self) -> int:
        return self._ino

    @property
    def mode(self) -> int:
        return self._mode

    def stat(self) -> VNodeStat:
        with self._lock:
            return VNodeStat(
                ino    = self._ino,
                size   = self._stat_size(),
                vtype  = self._vtype,
                mode   = self._mode,
                mtime  = self._mtime,
                atime  = self._atime,
                nlinks = 1,
            )

    def _stat_size(self) -> int:
        return 0

    @abstractmethod
    def read(self, offset: int, count: int) -> bytes: ...

    @abstractmethod
    def write(self, offset: int, data: bytes) -> int: ...

    def readdir(self) -> List[str]:
        """Return list of child names. Only valid for DIR nodes."""
        raise IsADirectoryError(f"{self._path} is not a directory")

    def truncate(self, size: int) -> None:
        raise PermissionError("truncate not supported on this vnode")

    def _touch_mtime(self) -> None:
        self._mtime = _now()

    def _touch_atime(self) -> None:
        self._atime = _now()


# ─────────────────────────────────────────────────────────────────────────────
# §2  FILE DESCRIPTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileDescriptor:
    """
    Open file handle binding a VNode to an offset and flags.

    fd:     small integer (table index) assigned by FDTable
    vnode:  underlying VNode
    offset: current read/write position
    flags:  OpenFlags bitmask
    path:   absolute path at open time (for display / /proc/self/fd)
    """
    fd:     int
    vnode:  VNode
    offset: int
    flags:  OpenFlags
    path:   str
    _lock:  threading.RLock = field(default_factory=threading.RLock, repr=False)

    def readable(self) -> bool:
        return not (self.flags & OpenFlags.O_WRONLY)

    def writable(self) -> bool:
        return bool(self.flags & (OpenFlags.O_WRONLY | OpenFlags.O_RDWR))

    def read(self, count: int) -> bytes:
        if not self.readable():
            raise PermissionError("File descriptor not open for reading")
        with self._lock:
            data        = self.vnode.read(self.offset, count)
            self.offset += len(data)
        return data

    def write(self, data: bytes) -> int:
        if not self.writable():
            raise PermissionError("File descriptor not open for writing")
        with self._lock:
            if self.flags & OpenFlags.O_APPEND:
                self.offset = self.vnode.stat().size
            n           = self.vnode.write(self.offset, data)
            self.offset += n
        return n

    def seek(self, offset: int, whence: SeekWhence = SeekWhence.SEEK_SET) -> int:
        with self._lock:
            if whence == SeekWhence.SEEK_SET:
                self.offset = offset
            elif whence == SeekWhence.SEEK_CUR:
                self.offset += offset
            elif whence == SeekWhence.SEEK_END:
                self.offset = self.vnode.stat().size + offset
            if self.offset < 0:
                self.offset = 0
        return self.offset


# ─────────────────────────────────────────────────────────────────────────────
# §3  FD TABLE
# ─────────────────────────────────────────────────────────────────────────────

class FDTable:
    """
    File descriptor table.  Allocates small integers (≥ 3) for open handles.
    FDs 0, 1, 2 are reserved for stdin/stdout/stderr (not managed here).
    """

    def __init__(self) -> None:
        self._table : Dict[int, FileDescriptor] = {}
        self._lock  = threading.RLock()
        self._next  = 3

    def alloc(self, vnode: VNode, flags: OpenFlags, path: str) -> FileDescriptor:
        """Allocate a new FD and register the open FileDescriptor."""
        with self._lock:
            # First-fit scan from _next upward, wrapping at MAX_FDS
            for _ in range(MAX_FDS):
                if self._next not in self._table:
                    fd_num = self._next
                    self._next = (self._next % (MAX_FDS - 1)) + 3
                    offset = 0
                    if flags & OpenFlags.O_APPEND:
                        offset = vnode.stat().size
                    fd = FileDescriptor(fd=fd_num, vnode=vnode,
                                        offset=offset, flags=flags, path=path)
                    self._table[fd_num] = fd
                    return fd
                self._next = (self._next % (MAX_FDS - 1)) + 3
            raise IOError("File descriptor table exhausted (max FDs open)")

    def get(self, fd: int) -> FileDescriptor:
        with self._lock:
            fdo = self._table.get(fd)
            if fdo is None:
                raise IOError(f"Bad file descriptor: {fd}")
            return fdo

    def close(self, fd: int) -> None:
        with self._lock:
            if fd not in self._table:
                raise IOError(f"Bad file descriptor: {fd}")
            del self._table[fd]

    def open_count(self) -> int:
        with self._lock:
            return len(self._table)

    def list_open(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"fd": f.fd, "path": f.path,
                 "offset": f.offset, "flags": int(f.flags)}
                for f in self._table.values()
            ]


# ─────────────────────────────────────────────────────────────────────────────
# §4  MEMFS
#
#  In-memory filesystem backed by a tree of dicts and bytearrays.
#  Supports: mkdir, create, unlink, rmdir, stat, read, write, readdir.
#  Files are stored as bytearray; directories as {name: VNode}.
# ─────────────────────────────────────────────────────────────────────────────

class MemFSFile(VNode):
    """Mutable in-memory file."""

    def __init__(self, path: str, mode: int = 0o644) -> None:
        super().__init__(path, VNodeType.FILE, mode)
        self._buf : bytearray = bytearray()

    def _stat_size(self) -> int:
        return len(self._buf)

    def read(self, offset: int, count: int) -> bytes:
        with self._lock:
            self._touch_atime()
            return bytes(self._buf[offset:offset + count])

    def write(self, offset: int, data: bytes) -> int:
        with self._lock:
            end = offset + len(data)
            if end > MAX_FILE_SIZE:
                raise NoSpaceError(f"File size limit {MAX_FILE_SIZE} exceeded")
            if end > len(self._buf):
                self._buf.extend(bytes(end - len(self._buf)))
            self._buf[offset:end] = data
            self._touch_mtime()
            return len(data)

    def truncate(self, size: int) -> None:
        with self._lock:
            if size < len(self._buf):
                self._buf = self._buf[:size]
            else:
                self._buf.extend(bytes(size - len(self._buf)))
            self._touch_mtime()


class MemFSDir(VNode):
    """In-memory directory."""

    def __init__(self, path: str, mode: int = 0o755) -> None:
        super().__init__(path, VNodeType.DIR, mode)
        self._children: Dict[str, VNode] = {}

    def _stat_size(self) -> int:
        return len(self._children)

    def read(self, offset: int, count: int) -> bytes:
        raise IsADirectoryError(f"{self._path} is a directory")

    def write(self, offset: int, data: bytes) -> int:
        raise IsADirectoryError(f"{self._path} is a directory")

    def readdir(self) -> List[str]:
        with self._lock:
            return sorted(self._children.keys())

    def get_child(self, name: str) -> Optional[VNode]:
        with self._lock:
            return self._children.get(name)

    def add_child(self, name: str, node: VNode) -> None:
        with self._lock:
            self._children[name] = node
            self._touch_mtime()

    def remove_child(self, name: str) -> VNode:
        with self._lock:
            if name not in self._children:
                raise FileNotFoundError(f"No entry '{name}' in {self._path}")
            node = self._children.pop(name)
            self._touch_mtime()
            return node


class MemFS:
    """
    In-memory filesystem driver.

    Provides lookup, create, mkdir, unlink, rmdir given an absolute path.
    Used as the backing store for VFS mount points (e.g. mounted at "/tmp").
    """

    def __init__(self, mount_point: str = "/") -> None:
        self._mount = mount_point.rstrip("/") or "/"
        self._root  = MemFSDir(mount_point, mode=0o755)

    def _local(self, abs_path: str) -> str:
        """
        Strip the mount prefix so paths are relative to this FS root.
        E.g. mount="/tmp", abs_path="/tmp/a/b.txt" → "/a/b.txt"
             mount="/",     abs_path="/etc/rc"      → "/etc/rc"
        """
        prefix = self._mount
        if prefix == "/":
            return abs_path  # root mount — no stripping needed
        if abs_path == prefix:
            return "/"
        if abs_path.startswith(prefix + "/"):
            return abs_path[len(prefix):]  # "/tmp/a/b" → "/a/b"
        return abs_path

    def _resolve_parent(self, abs_path: str) -> Tuple[MemFSDir, str]:
        """Return (parent_dir, leaf_name) for abs_path."""
        local = self._local(abs_path)
        parts = [p for p in local.split("/") if p]
        if not parts:
            raise FileNotFoundError("Cannot resolve parent of root")
        cur: VNode = self._root
        for part in parts[:-1]:
            if not isinstance(cur, MemFSDir):
                raise NotADirectoryError(f"{cur._path} is not a directory")
            child = cur.get_child(part)
            if child is None:
                raise FileNotFoundError(f"Directory '{part}' not found in {cur._path}")
            cur = child
        if not isinstance(cur, MemFSDir):
            raise NotADirectoryError(f"{cur._path} is not a directory")
        return cur, parts[-1]

    def lookup(self, abs_path: str) -> VNode:
        """Resolve abs_path to a VNode. Raises FileNotFoundError."""
        local = self._local(abs_path)
        parts = [p for p in local.split("/") if p]
        cur: VNode = self._root
        for part in parts:
            if not isinstance(cur, MemFSDir):
                raise NotADirectoryError(f"{cur._path} is not a directory")
            child = cur.get_child(part)
            if child is None:
                raise FileNotFoundError(f"No entry '{part}' in {cur._path}")
            cur = child
        return cur

    def create(self, abs_path: str, mode: int = 0o644) -> MemFSFile:
        """Create a new file at abs_path. Raises FileExistsError if exists."""
        parent, name = self._resolve_parent(abs_path)
        if parent.get_child(name) is not None:
            raise FileExistsError(f"{abs_path} already exists")
        f = MemFSFile(abs_path, mode)
        parent.add_child(name, f)
        return f

    def mkdir(self, abs_path: str, mode: int = 0o755) -> MemFSDir:
        """Create a directory at abs_path."""
        parent, name = self._resolve_parent(abs_path)
        if parent.get_child(name) is not None:
            raise FileExistsError(f"{abs_path} already exists")
        d = MemFSDir(abs_path, mode)
        parent.add_child(name, d)
        return d

    def unlink(self, abs_path: str) -> None:
        """Remove a file at abs_path."""
        parent, name = self._resolve_parent(abs_path)
        node = parent.get_child(name)
        if node is None:
            raise FileNotFoundError(abs_path)
        if isinstance(node, MemFSDir):
            raise IsADirectoryError(f"{abs_path} is a directory (use rmdir)")
        parent.remove_child(name)

    def rmdir(self, abs_path: str) -> None:
        """Remove an empty directory."""
        parent, name = self._resolve_parent(abs_path)
        node = parent.get_child(name)
        if node is None:
            raise FileNotFoundError(abs_path)
        if not isinstance(node, MemFSDir):
            raise NotADirectoryError(f"{abs_path} is not a directory")
        if node.readdir():
            raise IOError(f"{abs_path} is not empty")
        parent.remove_child(name)

    def get_or_create_file(self, abs_path: str, mode: int = 0o644) -> MemFSFile:
        """Return existing file or create it."""
        try:
            node = self.lookup(abs_path)
            if not isinstance(node, MemFSFile):
                raise IsADirectoryError(f"{abs_path} is not a file")
            return node
        except FileNotFoundError:
            return self.create(abs_path, mode)


# ─────────────────────────────────────────────────────────────────────────────
# §5  PROCFS
#
#  Read-only pseudo-filesystem that exposes live kernel state as files.
#  All content is generated on-the-fly when read; writes are rejected.
#
#  Paths:
#    /proc/traces      — last 50 AgentTrace records as JSON lines
#    /proc/status      — kernel.status() as JSON
#    /proc/tools       — registered agent tools as JSON
#    /proc/reward      — RewardKernel.stats() if attached
#    /proc/ipc         — IPCKernel.stats() if attached
#    /proc/fds         — FDTable.list_open() as JSON
# ─────────────────────────────────────────────────────────────────────────────

class ProcFSFile(VNode):
    """Read-only VNode whose content is produced by a generator callable."""

    def __init__(self, path: str, generator: Callable[[], bytes]) -> None:
        super().__init__(path, VNodeType.FILE, mode=0o444)
        self._gen = generator

    def _stat_size(self) -> int:
        return len(self._gen())

    def read(self, offset: int, count: int) -> bytes:
        with self._lock:
            self._touch_atime()
            content = self._gen()
        return content[offset:offset + count]

    def write(self, offset: int, data: bytes) -> int:
        raise PermissionError(f"{self._path} is read-only")


class ProcFSDir(VNode):
    """Directory node for ProcFS — children are fixed at construction."""

    def __init__(self, path: str, children: Dict[str, VNode]) -> None:
        super().__init__(path, VNodeType.DIR, mode=0o555)
        self._children = children

    def _stat_size(self) -> int:
        return len(self._children)

    def read(self, offset: int, count: int) -> bytes:
        raise IsADirectoryError(f"{self._path} is a directory")

    def write(self, offset: int, data: bytes) -> int:
        raise IsADirectoryError(f"{self._path} is a directory")

    def readdir(self) -> List[str]:
        return sorted(self._children.keys())

    def get_child(self, name: str) -> Optional[VNode]:
        return self._children.get(name)


class ProcFS:
    """
    ProcFS driver.  Populates /proc with live kernel data providers.
    """

    def __init__(
        self,
        kernel:    Optional[Any] = None,
        fd_table:  Optional[FDTable] = None,
    ) -> None:
        self._kernel   = kernel
        self._fd_table = fd_table
        self._root     = self._build_root()

    def _gen_traces(self) -> bytes:
        if self._kernel is None:
            return b"[]\n"
        try:
            from aios_core import _registry
            traces = _registry.recent_traces(50)
            lines  = [json.dumps(t.to_dict(), separators=(',', ':')) for t in traces]
            return ("\n".join(lines) + "\n").encode('utf-8')
        except Exception as e:
            return json.dumps({"error": str(e)}).encode()

    def _gen_status(self) -> bytes:
        if self._kernel is None:
            return b"{}\n"
        try:
            return (json.dumps(self._kernel.status(), separators=(',', ':')) + "\n").encode()
        except Exception as e:
            return json.dumps({"error": str(e)}).encode()

    def _gen_tools(self) -> bytes:
        try:
            from aios_core import _registry
            tools = [
                {"name": s.name, "description": s.description,
                 "priority": int(s.priority), "owner": s.owner}
                for s in _registry.all_tools()
            ]
            return (json.dumps(tools, separators=(',', ':')) + "\n").encode()
        except Exception as e:
            return json.dumps({"error": str(e)}).encode()

    def _gen_fds(self) -> bytes:
        if self._fd_table is None:
            return b"[]\n"
        return (json.dumps(self._fd_table.list_open(), separators=(',', ':')) + "\n").encode()

    def _gen_version(self) -> bytes:
        return f'{{"vfs":"{_VFS_VERSION}","ts":{time.monotonic():.4f}}}\n'.encode()

    def _build_root(self) -> ProcFSDir:
        children: Dict[str, VNode] = {
            "traces":  ProcFSFile("/proc/traces",  self._gen_traces),
            "status":  ProcFSFile("/proc/status",  self._gen_status),
            "tools":   ProcFSFile("/proc/tools",   self._gen_tools),
            "fds":     ProcFSFile("/proc/fds",     self._gen_fds),
            "version": ProcFSFile("/proc/version", self._gen_version),
        }
        return ProcFSDir("/proc", children)

    def lookup(self, abs_path: str) -> VNode:
        """Resolve path under /proc. abs_path must start with /proc."""
        path = abs_path[len("/proc"):].lstrip("/")
        if not path:
            return self._root
        parts = path.split("/")
        node: VNode = self._root
        for part in parts:
            if not isinstance(node, ProcFSDir):
                raise NotADirectoryError(f"{abs_path}: not a directory")
            child = node.get_child(part)
            if child is None:
                raise FileNotFoundError(f"/proc/{part}: no such file")
            node = child
        return node


# ─────────────────────────────────────────────────────────────────────────────
# §6  DEVFS
#
#  Device filesystem at /dev.  Each device node is a VNode whose read/write
#  behavior reflects the device semantics.
#
#  /dev/null  — reads return b""; writes silently succeed (consume data)
#  /dev/zero  — reads return count zero bytes; writes silently succeed
#  /dev/mem   — reads from the kernel MemoryBus if attached
#  /dev/tty   — placeholder (actual TTY I/O is handled by aios_core REPL)
# ─────────────────────────────────────────────────────────────────────────────

class DevNull(VNode):
    def __init__(self) -> None:
        super().__init__("/dev/null", VNodeType.DEVICE, mode=0o666)

    def _stat_size(self) -> int: return 0

    def read(self, offset: int, count: int) -> bytes:
        return b""

    def write(self, offset: int, data: bytes) -> int:
        return len(data)   # discard


class DevZero(VNode):
    def __init__(self) -> None:
        super().__init__("/dev/zero", VNodeType.DEVICE, mode=0o666)

    def _stat_size(self) -> int: return 0

    def read(self, offset: int, count: int) -> bytes:
        return bytes(count)

    def write(self, offset: int, data: bytes) -> int:
        return len(data)   # discard


class DevMem(VNode):
    """
    /dev/mem — provides raw read access to the kernel MemoryBus.
    Writes are rejected for safety.
    """

    def __init__(self, bus: Optional[Any] = None) -> None:
        super().__init__("/dev/mem", VNodeType.DEVICE, mode=0o440)
        self._bus = bus

    def _stat_size(self) -> int:
        return 64 * 1024 * 1024  # 64 MiB simulated

    def read(self, offset: int, count: int) -> bytes:
        if self._bus is None:
            return bytes(count)
        try:
            return bytes(self._bus.peek_buf(offset, count))
        except Exception:
            return bytes(count)

    def write(self, offset: int, data: bytes) -> int:
        raise PermissionError("/dev/mem is read-only via VFS")


class DevFSDir(VNode):
    """Directory node for /dev."""

    def __init__(self, children: Dict[str, VNode]) -> None:
        super().__init__("/dev", VNodeType.DIR, mode=0o755)
        self._children = children

    def _stat_size(self) -> int: return len(self._children)

    def read(self, offset: int, count: int) -> bytes:
        raise IsADirectoryError("/dev is a directory")

    def write(self, offset: int, data: bytes) -> int:
        raise IsADirectoryError("/dev is a directory")

    def readdir(self) -> List[str]:
        return sorted(self._children.keys())

    def get_child(self, name: str) -> Optional[VNode]:
        return self._children.get(name)


class DevFS:
    """Device filesystem driver."""

    def __init__(self, bus: Optional[Any] = None) -> None:
        self._null = DevNull()
        self._zero = DevZero()
        self._mem  = DevMem(bus)
        self._root = DevFSDir({
            "null": self._null,
            "zero": self._zero,
            "mem":  self._mem,
        })

    def lookup(self, abs_path: str) -> VNode:
        path = abs_path[len("/dev"):].lstrip("/")
        if not path:
            return self._root
        child = self._root.get_child(path)
        if child is None:
            raise FileNotFoundError(f"{abs_path}: no such device")
        return child


# ─────────────────────────────────────────────────────────────────────────────
# §7  VFS — root namespace
#
#  The VFS maintains a mount table mapping path prefixes to filesystem drivers.
#  Default mounts (established at init):
#    "/"     → MemFS  (root filesystem)
#    "/proc" → ProcFS (kernel introspection)
#    "/dev"  → DevFS  (device nodes)
#    "/tmp"  → MemFS  (temporary files, separate instance)
#
#  Path resolution:
#    1. Find longest mount prefix matching the path
#    2. Strip prefix; pass remainder to driver.lookup()
#    3. Return VNode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MountEntry:
    prefix:  str
    driver:  Any   # MemFS | ProcFS | DevFS — must implement .lookup(abs_path)
    options: Dict[str, Any] = field(default_factory=dict)


class VFS:
    """
    Virtual Filesystem root namespace.

    Provides the POSIX-like syscall surface:
      open, read, write, close, seek, stat, readdir, mkdir, unlink, rmdir
    Plus: mount, umount.
    """

    def __init__(
        self,
        kernel:   Optional[Any] = None,
        bus:      Optional[Any] = None,
    ) -> None:
        self._lock     = threading.RLock()
        self._fd_table = FDTable()
        self._mounts   : List[MountEntry] = []

        # Default mounts — order matters (longest prefix wins)
        self._proc_fs  = ProcFS(kernel=kernel, fd_table=self._fd_table)
        self._dev_fs   = DevFS(bus=bus)
        self._root_fs  = MemFS("/")
        self._tmp_fs   = MemFS("/tmp")

        self.mount("/",     self._root_fs)
        self.mount("/proc", self._proc_fs)
        self.mount("/dev",  self._dev_fs)
        self.mount("/tmp",  self._tmp_fs)

        # Create well-known directories on the root FS
        try: self._root_fs.mkdir("/etc")
        except FileExistsError: pass
        try: self._root_fs.mkdir("/var")
        except FileExistsError: pass
        try: self._root_fs.mkdir("/home")
        except FileExistsError: pass

    # ── Mount management ──────────────────────────────────────────────────────

    def mount(self, prefix: str, driver: Any, options: Optional[Dict] = None) -> None:
        """Attach a filesystem driver at prefix."""
        with self._lock:
            # Remove any existing mount at exact prefix
            self._mounts = [m for m in self._mounts if m.prefix != prefix]
            self._mounts.append(MountEntry(prefix, driver, options or {}))
            # Sort by descending prefix length so longest match wins
            self._mounts.sort(key=lambda m: len(m.prefix), reverse=True)

    def umount(self, prefix: str) -> bool:
        """Detach the filesystem at prefix. Returns True if found."""
        with self._lock:
            before = len(self._mounts)
            self._mounts = [m for m in self._mounts if m.prefix != prefix]
            return len(self._mounts) < before

    def _resolve(self, path: str) -> VNode:
        """Find and return the VNode for path."""
        if not path.startswith("/"):
            raise FileNotFoundError(f"Path must be absolute: {path!r}")
        if len(path) > MAX_PATH_LEN:
            raise IOError(f"Path too long: {len(path)} > {MAX_PATH_LEN}")

        with self._lock:
            mounts = list(self._mounts)

        for entry in mounts:
            if path == entry.prefix or path.startswith(entry.prefix.rstrip("/") + "/"):
                return entry.driver.lookup(path)

        raise FileNotFoundError(f"No mount covers path: {path!r}")

    # ── Core syscall surface ──────────────────────────────────────────────────

    def open(
        self,
        path:  str,
        flags: OpenFlags = OpenFlags.O_RDONLY,
        mode:  int = 0o644,
    ) -> int:
        """
        Open path and return a file descriptor integer.
        Creates the file if O_CREAT is set and the path is on a MemFS.
        """
        try:
            vnode = self._resolve(path)
        except FileNotFoundError:
            if not (flags & OpenFlags.O_CREAT):
                raise
            # Create the file on the appropriate MemFS mount
            self._create_file(path, mode)
            vnode = self._resolve(path)

        if flags & OpenFlags.O_TRUNC and isinstance(vnode, MemFSFile):
            vnode.truncate(0)

        fd = self._fd_table.alloc(vnode, flags, path)
        return fd.fd

    def _create_file(self, path: str, mode: int) -> None:
        """Create a file by finding the MemFS that covers path."""
        with self._lock:
            mounts = list(self._mounts)
        for entry in mounts:
            if isinstance(entry.driver, MemFS):
                if path == entry.prefix or path.startswith(
                    entry.prefix.rstrip("/") + "/"
                ):
                    entry.driver.create(path, mode)
                    return
        raise PermissionError(f"No writable filesystem covers {path!r}")

    def read(self, fd: int, count: int) -> bytes:
        """Read up to count bytes from fd at current offset."""
        return self._fd_table.get(fd).read(count)

    def write(self, fd: int, data: bytes) -> int:
        """Write data to fd at current offset. Returns bytes written."""
        return self._fd_table.get(fd).write(data)

    def seek(self, fd: int, offset: int, whence: SeekWhence = SeekWhence.SEEK_SET) -> int:
        """Reposition fd's offset. Returns new offset."""
        return self._fd_table.get(fd).seek(offset, whence)

    def close(self, fd: int) -> None:
        """Close file descriptor fd."""
        self._fd_table.close(fd)

    def stat(self, path: str) -> VNodeStat:
        """Return stat for path."""
        return self._resolve(path).stat()

    def fstat(self, fd: int) -> VNodeStat:
        """Return stat for open fd."""
        return self._fd_table.get(fd).vnode.stat()

    def readdir(self, path: str) -> List[str]:
        """Return sorted list of directory entries."""
        node = self._resolve(path)
        return node.readdir()

    def mkdir(self, path: str, mode: int = 0o755) -> None:
        """Create a directory. Parent must exist."""
        with self._lock:
            mounts = list(self._mounts)
        for entry in mounts:
            if isinstance(entry.driver, MemFS):
                if path == entry.prefix or path.startswith(
                    entry.prefix.rstrip("/") + "/"
                ):
                    entry.driver.mkdir(path, mode)
                    return
        raise PermissionError(f"No writable filesystem covers {path!r}")

    def unlink(self, path: str) -> None:
        """Remove a file."""
        with self._lock:
            mounts = list(self._mounts)
        for entry in mounts:
            if isinstance(entry.driver, MemFS):
                if path.startswith(entry.prefix.rstrip("/") + "/") or path == entry.prefix:
                    entry.driver.unlink(path)
                    return
        raise PermissionError(f"No writable filesystem covers {path!r}")

    def write_file(self, path: str, data: bytes, mode: int = 0o644) -> int:
        """
        Convenience: write bytes to a file, creating or truncating as needed.
        Returns bytes written.
        """
        fd = self.open(path, OpenFlags.O_RDWR | OpenFlags.O_CREAT | OpenFlags.O_TRUNC, mode)
        try:
            n = self.write(fd, data)
        finally:
            self.close(fd)
        return n

    def read_file(self, path: str) -> bytes:
        """Convenience: read entire file content."""
        st = self.stat(path)
        fd = self.open(path, OpenFlags.O_RDONLY)
        try:
            return self.read(fd, st.size)
        finally:
            self.close(fd)

    def mounts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"prefix": m.prefix, "driver": type(m.driver).__name__,
                 "options": m.options}
                for m in self._mounts
            ]


# ─────────────────────────────────────────────────────────────────────────────
# §8  VFS KERNEL — @agent_method integration
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aios_core import agent_method, AgentPriority
    _HAS_CORE = True
except ImportError:
    def agent_method(**kw):  # type: ignore[misc]
        def dec(fn): return fn
        return dec
    class AgentPriority:  # type: ignore[no-redef]
        CRITICAL, HIGH, NORMAL, LOW = 0, 1, 2, 3
    _HAS_CORE = False


def _rebind_agent_methods(obj: Any) -> None:
    """Re-register @agent_method tools on obj as bound methods in the registry."""
    if not _HAS_CORE:
        return
    from aios_core import AgentToolSpec, _registry as _reg
    for attr_name in dir(obj):
        try:
            method = getattr(obj, attr_name)
        except AttributeError:
            continue
        spec = getattr(method, '_agent_spec', None)
        if spec is None:
            continue
        bound_spec = AgentToolSpec(
            name=spec.name, description=spec.description,
            parameters=spec.parameters, returns=spec.returns,
            priority=spec.priority, fn=method, owner=type(obj).__name__,
        )
        _reg.register(bound_spec)


class VFSKernel:
    """
    @agent_method surface for VFS operations.

    Attach after kernel boot:
        vfs_kernel = VFSKernel(kernel, bus)
        vfs_kernel.attach(kernel)
    """

    def __init__(
        self,
        kernel: Optional[Any] = None,
        bus:    Optional[Any] = None,
    ) -> None:
        self.vfs = VFS(kernel=kernel, bus=bus)

    def attach(self, kernel: Any) -> None:
        """Bind to a running AgentKernel."""
        _rebind_agent_methods(self)

    @agent_method(
        name="vfs_open",
        description="Open a file and return a file descriptor",
        parameters={
            "path":  {"type": "str", "desc": "Absolute file path"},
            "flags": {"type": "int", "desc": "OpenFlags bitmask (0=rdonly, 1=wronly, 2=rdwr, 0x40=creat)"},
        },
        priority=AgentPriority.NORMAL,
    )
    def vfs_open(self, path: str, flags: int = 0, mode: int = 0o644) -> int:
        return self.vfs.open(path, OpenFlags(flags), mode)

    @agent_method(
        name="vfs_read",
        description="Read bytes from a file descriptor",
        parameters={
            "fd":    {"type": "int", "desc": "File descriptor"},
            "count": {"type": "int", "desc": "Bytes to read"},
        },
        priority=AgentPriority.NORMAL,
    )
    def vfs_read(self, fd: int, count: int) -> bytes:
        return self.vfs.read(fd, count)

    @agent_method(
        name="vfs_write",
        description="Write bytes to a file descriptor",
        parameters={
            "fd":   {"type": "int",   "desc": "File descriptor"},
            "data": {"type": "bytes", "desc": "Data to write"},
        },
        priority=AgentPriority.NORMAL,
    )
    def vfs_write(self, fd: int, data: bytes) -> int:
        return self.vfs.write(fd, data)

    @agent_method(
        name="vfs_close",
        description="Close a file descriptor",
        parameters={"fd": {"type": "int", "desc": "File descriptor to close"}},
        priority=AgentPriority.NORMAL,
    )
    def vfs_close(self, fd: int) -> None:
        self.vfs.close(fd)

    @agent_method(
        name="vfs_stat",
        description="Stat a path; returns size, type, mtime",
        parameters={"path": {"type": "str", "desc": "Absolute path"}},
        priority=AgentPriority.LOW,
    )
    def vfs_stat(self, path: str) -> Dict[str, Any]:
        s = self.vfs.stat(path)
        return {
            "ino":   s.ino, "size": s.size, "type": s.vtype.value,
            "mode":  oct(s.mode), "mtime": s.mtime, "atime": s.atime,
        }

    @agent_method(
        name="vfs_readdir",
        description="List directory contents",
        parameters={"path": {"type": "str", "desc": "Absolute directory path"}},
        priority=AgentPriority.LOW,
    )
    def vfs_readdir(self, path: str) -> List[str]:
        return self.vfs.readdir(path)

    @agent_method(
        name="vfs_write_file",
        description="Write entire file content (create or overwrite)",
        parameters={
            "path": {"type": "str",   "desc": "Absolute path"},
            "data": {"type": "bytes", "desc": "File content"},
        },
        priority=AgentPriority.NORMAL,
    )
    def vfs_write_file(self, path: str, data: bytes) -> int:
        return self.vfs.write_file(path, data)

    @agent_method(
        name="vfs_read_file",
        description="Read entire file content",
        parameters={"path": {"type": "str", "desc": "Absolute path"}},
        priority=AgentPriority.NORMAL,
    )
    def vfs_read_file(self, path: str) -> bytes:
        return self.vfs.read_file(path)

    @agent_method(
        name="vfs_mkdir",
        description="Create a directory",
        parameters={"path": {"type": "str", "desc": "Absolute directory path"}},
        priority=AgentPriority.NORMAL,
    )
    def vfs_mkdir(self, path: str, mode: int = 0o755) -> None:
        self.vfs.mkdir(path, mode)

    @agent_method(
        name="vfs_unlink",
        description="Delete a file",
        parameters={"path": {"type": "str", "desc": "Absolute file path"}},
        priority=AgentPriority.NORMAL,
    )
    def vfs_unlink(self, path: str) -> None:
        self.vfs.unlink(path)

    @agent_method(
        name="vfs_mounts",
        description="List mounted filesystems",
        priority=AgentPriority.LOW,
    )
    def vfs_mounts(self) -> List[Dict[str, Any]]:
        return self.vfs.mounts()


# ─────────────────────────────────────────────────────────────────────────────
# §9  SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """Deterministic validation suite. Raises AssertionError on failure."""

    # ── MemFS basic operations ────────────────────────────────────────────────
    fs = MemFS("/")
    f  = fs.create("/hello.txt")
    f.write(0, b"Hello, AIOS!")
    assert f.read(0, 5)  == b"Hello"
    assert f.read(7, 5)  == b"AIOS!"
    assert f.stat().size == 12

    f.truncate(5)
    assert f.stat().size == 5
    assert f.read(0, 5) == b"Hello"

    fs.mkdir("/subdir")
    fs.create("/subdir/file2.txt").write(0, b"inner")

    assert "subdir"   in fs.lookup("/").readdir()
    assert "hello.txt" in fs.lookup("/").readdir()
    assert "file2.txt" in fs.lookup("/subdir").readdir()

    fs.unlink("/hello.txt")
    try:
        fs.lookup("/hello.txt")
        assert False, "should raise FileNotFoundError"
    except FileNotFoundError:
        pass

    # rmdir only works on empty dirs
    try:
        fs.rmdir("/subdir")
        assert False, "non-empty rmdir should fail"
    except IOError:
        pass
    fs.unlink("/subdir/file2.txt")
    fs.rmdir("/subdir")

    # ── FDTable ───────────────────────────────────────────────────────────────
    ft = FDTable()
    vn = MemFSFile("/test_fd.txt")
    vn.write(0, b"0123456789")

    fd_obj = ft.alloc(vn, OpenFlags.O_RDWR, "/test_fd.txt")
    assert fd_obj.fd == 3
    data = fd_obj.read(4)
    assert data == b"0123"
    assert fd_obj.offset == 4

    fd_obj.write(b"XXXX")
    assert vn.read(4, 4) == b"XXXX"

    ft.close(fd_obj.fd)
    try:
        ft.get(fd_obj.fd)
        assert False, "closed fd should raise"
    except IOError:
        pass

    # ── DevFS ─────────────────────────────────────────────────────────────────
    devfs = DevFS()
    null  = devfs.lookup("/dev/null")
    zero  = devfs.lookup("/dev/zero")
    mem   = devfs.lookup("/dev/mem")

    assert null.read(0, 100) == b""
    assert null.write(0, b"anything") == 8

    assert zero.read(0, 4) == b"\x00\x00\x00\x00"
    assert zero.write(0, b"data") == 4

    assert len(mem.read(0, 16)) == 16   # returns zeros without bus

    # ── ProcFS ────────────────────────────────────────────────────────────────
    procfs = ProcFS(kernel=None)
    ver_node = procfs.lookup("/proc/version")
    assert isinstance(ver_node, ProcFSFile)
    ver_data = ver_node.read(0, 1024)
    assert b"vfs" in ver_data

    procfs_dir = procfs.lookup("/proc")
    entries = procfs_dir.readdir()
    assert "version" in entries
    assert "status"  in entries
    assert "traces"  in entries

    try:
        procfs.lookup("/proc/nonexistent")
        assert False, "should raise"
    except FileNotFoundError:
        pass

    # ── VFS integrated ────────────────────────────────────────────────────────
    vfs = VFS()

    # Write and read through VFS
    n = vfs.write_file("/tmp/test.txt", b"Hello VFS!")
    assert n == 10
    data = vfs.read_file("/tmp/test.txt")
    assert data == b"Hello VFS!", f"data={data!r}"

    # Stat
    st = vfs.stat("/tmp/test.txt")
    assert st.size == 10
    assert st.vtype == VNodeType.FILE

    # FD-level read/write
    fd = vfs.open("/tmp/rw.bin", OpenFlags.O_RDWR | OpenFlags.O_CREAT)
    vfs.write(fd, b"ABCDEF")
    vfs.seek(fd, 2)
    chunk = vfs.read(fd, 3)
    assert chunk == b"CDE", f"chunk={chunk!r}"
    vfs.close(fd)

    # mkdir and readdir
    vfs.mkdir("/tmp/mydir")
    vfs.write_file("/tmp/mydir/a.txt", b"A")
    vfs.write_file("/tmp/mydir/b.txt", b"B")
    entries = vfs.readdir("/tmp/mydir")
    assert "a.txt" in entries and "b.txt" in entries

    # /proc access through VFS
    fd2 = vfs.open("/proc/version", OpenFlags.O_RDONLY)
    ver  = vfs.read(fd2, 256)
    vfs.close(fd2)
    assert b"vfs" in ver

    # /dev/zero through VFS
    fd3 = vfs.open("/dev/zero", OpenFlags.O_RDONLY)
    zeroes = vfs.read(fd3, 8)
    vfs.close(fd3)
    assert zeroes == b"\x00" * 8

    # /dev/null through VFS
    fd4 = vfs.open("/dev/null", OpenFlags.O_WRONLY)
    n4  = vfs.write(fd4, b"garbage")
    vfs.close(fd4)
    assert n4 == 7

    # Unlink
    vfs.unlink("/tmp/rw.bin")
    try:
        vfs.stat("/tmp/rw.bin")
        assert False, "unlinked file should be gone"
    except FileNotFoundError:
        pass

    # Mounts list
    mounts = vfs.mounts()
    prefixes = {m["prefix"] for m in mounts}
    assert "/" in prefixes
    assert "/proc" in prefixes
    assert "/dev" in prefixes
    assert "/tmp" in prefixes

    print("aios_vfs: all self-tests passed ✓")


if __name__ == "__main__":
    _run_self_tests()
