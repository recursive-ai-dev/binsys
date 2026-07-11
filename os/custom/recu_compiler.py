#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Recursive Execution Unit Compiler                                   ║
║  recu_compiler.py                                                            ║
║                                                                              ║
║  "A .recu is not an executable — it is a reasoned agent sealed in binary.   ║
║   Every byte carries intent. Every section carries identity."                ║
║                                                                              ║
║  Compiles: bash · html · appimage · elf · python · plain-script             ║
║       Into: .recu — Recursive Execution Unit — agentic artifact format      ║
║                                                                              ║
║  .recu Binary Layout (absolute offsets):                                    ║
║    [0x000:0x040]  File Header        (64 bytes, fixed)                      ║
║    [0x040:0x240]  Section Table      (8 × 64 bytes = 512 bytes)             ║
║    [0x240:...]    Section Data       (packed, 8-byte aligned)                ║
║                                                                              ║
║  Eight Sections (ordered):                                                   ║
║    MANIFEST      — JSON agent identity, permissions, kernel_bindings        ║
║    PAYLOAD       — Original source/binary (PackBits RLE-compressed)         ║
║    EMBEDDING     — float32[64] document semantic vector                     ║
║    ROUTING       — float32 capability logits + probs + mask                 ║
║    CA_STATE      — 16×16 B3/S23 lifecycle grid + rule                       ║
║    VMEM_MAP      — 7 virtual memory region descriptors                      ║
║    SYSCALL_TABLE — JSON agent tool binding table                            ║
║    SIGNATURE     — SHA-256 per-section hash chain                           ║
║                                                                              ║
║  Compilation Pipeline:                                                       ║
║    Detect → Tokenize → Embed → Classify(rule+neural) →                      ║
║    Manifest → CA_Grid → VMemMap → SyscallTable → Pack → Sign                ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Document embedding: ê = normalize(meanpool({E[tᵢ] | tᵢ ∈ T}))          ║
║      E ∈ ℝ^{512×64},  σ_init = 1/√64 (He-style for embedding)              ║
║    Routing classifier: logitsₖ = W₂·ReLU(W₁·ê + b₁) + b₂                  ║
║      W₁ ∈ ℝ^{128×64} (σ=√(2/64)), W₂ ∈ ℝ^{16×128} (σ=√(2/128))          ║
║    Capability mask:    capₖ = 1 iff σ(logitsₖ) > 0.5                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import os
import io
import struct
import hashlib
import json
import time
import re
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag, auto


# ══════════════════════════════════════════════════════════════════════════════
# §0  FORMAT CONSTANTS & ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

RECU_MAGIC           = b'RECURSI\x01'   # 8-byte magic number
RECU_VERSION_MAJ     = 0
RECU_VERSION_MIN     = 1
RECU_HEADER_SIZE     = 64              # fixed file header, bytes
RECU_SECT_ENTRY_SIZE = 64              # section table entry, bytes
RECU_N_SECTIONS      = 8              # always 8 sections per artifact
RECU_SECT_TABLE_OFF  = RECU_HEADER_SIZE                              # 0x40
RECU_DATA_OFF        = RECU_HEADER_SIZE + RECU_N_SECTIONS * RECU_SECT_ENTRY_SIZE  # 0x240

VOCAB_SIZE           = 512
EMBED_DIM            = 64
ROUTER_HIDDEN        = 128
N_CAPABILITIES       = 16

CA_ROWS              = 16
CA_COLS              = 16
CA_RULE_DEFAULT      = 'B3/S23'

VMEM_ENTRY_SIZE      = 40              # bytes per VMemRegion entry


class SectionType(IntEnum):
    MANIFEST      = 0x01
    PAYLOAD       = 0x02
    EMBEDDING     = 0x03
    ROUTING       = 0x04
    CA_STATE      = 0x05
    VMEM_MAP      = 0x06
    SYSCALL_TABLE = 0x07
    SIGNATURE     = 0x08


class SectionFlags(IntFlag):
    READ       = 0x01
    WRITE      = 0x02
    EXEC       = 0x04
    COMPRESSED = 0x08
    ENCRYPTED  = 0x10


class CapabilityFlags(IntFlag):
    NONE          = 0x0000
    FILE_READ     = 0x0001
    FILE_WRITE    = 0x0002
    NETWORK       = 0x0004
    PROCESS_SPAWN = 0x0008
    UI_RENDER     = 0x0010
    COMPUTE_HEAVY = 0x0020
    MEMORY_MAP    = 0x0040
    KERNEL_DIRECT = 0x0080
    CRYPTO        = 0x0100
    SYSTEM_CALL   = 0x0200
    INTERPROCESS  = 0x0400
    TIMER         = 0x0800
    SENSOR        = 0x1000
    GRAPHICS      = 0x2000
    DATABASE      = 0x4000
    AGENT_SPAWN   = 0x8000


class SourceFormat(Enum):
    BASH       = "bash"
    HTML       = "html"
    ELF        = "elf"
    APPIMAGE   = "appimage"
    PYTHON     = "python"
    PLAIN_TEXT = "plain_text"
    UNKNOWN    = "unknown"


class LifecycleState(Enum):
    DORMANT    = 0
    LOADING    = 1
    RUNNING    = 2
    SUSPENDED  = 3
    FAULTED    = 4
    TERMINATED = 5

    def to_token(self) -> int:
        """Maps lifecycle state to vocabulary token ID."""
        return 10 + self.value   # tokens 10-15 reserved for lifecycle states


# Virtual memory flags (independent of SectionFlags — these describe page perms)
VMEM_R  = 0x01
VMEM_W  = 0x02
VMEM_X  = 0x04
VMEM_S  = 0x08   # shared with kernel


# ══════════════════════════════════════════════════════════════════════════════
# §1  PURE MATH PRIMITIVES — fully inlined from aios_phase4_nn contract
#     These are replaced at runtime if aios_phase4_nn is importable.
#     The implementations below are identical in behaviour to the phase IV
#     originals — same algorithms, same convergence guarantees.
# ══════════════════════════════════════════════════════════════════════════════

_PI   = 3.141592653589793238462643383279
_E    = 2.718281828459045235360287471352
_LN2  = 0.693147180559945309417232121458
_INF  = float('inf')
_NAN  = float('nan')


def _abs(x: float) -> float:
    return x if x >= 0.0 else -x


def _floor(x: float) -> int:
    n = int(x)
    return n - 1 if x < n else n


def _exp(x: float) -> float:
    if x > 709.782: return _INF
    if x < -745.13: return 0.0
    if x == 0.0:    return 1.0
    n = _floor(x); r = x - n
    result = 1.0; term = 1.0
    for k in range(1, 25):
        term *= r / k; result += term
        if _abs(term) < 1e-17: break
    if n == 0: return result
    e_n = 1.0; base = _E; m = _abs(n)
    while m > 0:
        if m & 1: e_n *= base
        base *= base; m >>= 1
    return result / e_n if n < 0 else result * e_n


def _log(x: float) -> float:
    if x <= 0.0: return _NAN
    if x == 1.0: return 0.0
    k = 0; y = x
    while y >= 1.0: y *= 0.5; k += 1
    while y < 0.5:  y *= 2.0; k -= 1
    t = (y - 1.0) / (y + 1.0); t2 = t * t; acc = t; pw = t
    for n in range(1, 30):
        pw *= t2; term = pw / (2 * n + 1); acc += term
        if _abs(term) < 1e-17: break
    return k * _LN2 + 2.0 * acc


def _sqrt(x: float) -> float:
    if x < 0.0: return _NAN
    if x == 0.0: return 0.0
    y = x
    for _ in range(60):
        y_new = 0.5 * (y + x / y)
        if _abs(y_new - y) <= 1e-15 * y: return y_new
        y = y_new
    return y


def _cos(x: float) -> float:
    x = x - _floor(x / (2 * _PI)) * (2 * _PI)
    if x > _PI: x -= 2 * _PI
    x2 = x * x
    return (1.0 - x2/2.0 + x2*x2/24.0 - x2**3/720.0
            + x2**4/40320.0 - x2**5/3628800.0
            + x2**6/479001600.0 - x2**7/87178291200.0)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = _exp(-x); return 1.0 / (1.0 + e)
    else:
        e = _exp(x);  return e / (1.0 + e)


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §2  COMPILER-LOCAL XORSHIFT64 RNG
#     Instance-based (not class-level singleton) to avoid interfering with
#     the phase IV _RNG state during concurrent compilation passes.
# ══════════════════════════════════════════════════════════════════════════════

class _CompilerRNG:
    """
    Marsaglia Xorshift64 PRNG — period 2^64 − 1.
    Instance-scoped: each EmbeddingEngine gets its own seeded instance,
    ensuring weight matrices are reproducible across compiler invocations.
    """
    __slots__ = ('_state',)

    def __init__(self, seed: int = 0x853C49E6748FEA9B) -> None:
        self._state = int(seed) or 0x853C49E6748FEA9B

    def seed(self, s: int) -> None:
        self._state = int(s) or 0x853C49E6748FEA9B

    def _next(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >>  7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._state = x
        return x

    def random(self) -> float:
        """Uniform [0, 1)."""
        return (self._next() >> 11) * (1.0 / (1 << 53))

    def randn(self) -> float:
        """Standard Normal via Box-Muller transform."""
        u1 = self.random() + 1e-12
        u2 = self.random()
        return _sqrt(-2.0 * _log(u1)) * _cos(2.0 * _PI * u2)


# ══════════════════════════════════════════════════════════════════════════════
# §3  PHASE IV / CORE INTEGRATION — optional upgrade path
# ══════════════════════════════════════════════════════════════════════════════

_PHASE4_AVAILABLE = False
_CORE_AVAILABLE   = False

# Inject project root so sibling modules resolve correctly
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from aios_phase4_nn import (
        _abs, _exp, _log, _sqrt, _cos, _sigmoid, _relu,
        _PI, _E, _LN2, _INF,
        CellularAutomatonEngine,
    )
    _PHASE4_AVAILABLE = True
except ImportError:
    pass   # §1 inline implementations remain active

try:
    from aios_core import (
        agent_method, AgentPriority, SysCallResult, AgentContext, _registry,
    )
    _CORE_AVAILABLE = True
except ImportError:
    # Minimal shims — decorators become no-ops; SysCallResult is locally defined.
    def agent_method(**_kw):
        def _d(fn): return fn
        return _d
    class AgentPriority(IntEnum):
        CRITICAL = 0; HIGH = 1; NORMAL = 2; LOW = 3
    @dataclass
    class SysCallResult:
        success: bool; value: Any; error: Optional[str] = None; trace_id: Optional[str] = None
    class AgentContext:
        def __init__(self, caller: str = "recu"): self.caller = caller
    _registry = None


# ══════════════════════════════════════════════════════════════════════════════
# §4  VOCABULARY — 512-token lexicon covering all supported source formats
#
#  Token ranges:
#    0-15   Special / lifecycle markers
#    16-79  Bash/Shell keywords (64 slots)
#    80-143 HTML / CSS keywords (64 slots)
#    144-207 ELF / binary markers (64 slots)
#    208-255 Python keywords (48 slots)
#    256-383 Operators & punctuation (128 slots)
#    384-511 DJB2 hash overflow (128 slots, not in dict — computed at runtime)
# ══════════════════════════════════════════════════════════════════════════════

def _build_vocab() -> Dict[str, int]:
    v: Dict[str, int] = {}

    # §4.0  Special tokens (0-15)
    specials = [
        '[PAD]', '[UNK]', '[CLS]', '[SEP]',
        '[BASH]', '[HTML]', '[ELF]', '[APPIMAGE]', '[PYTHON]', '[TEXT]',
        '[DORMANT]', '[LOADING]', '[RUNNING]', '[SUSPENDED]', '[FAULTED]', '[TERMINATED]',
    ]
    for i, s in enumerate(specials):
        v[s] = i

    # §4.1  Bash / Shell (16-79)
    bash = [
        'if', 'then', 'else', 'elif', 'fi', 'for', 'while', 'do', 'done',
        'case', 'esac', 'function', 'return', 'exit', 'echo', 'export',
        'source', 'local', 'declare', 'readonly', 'shift', 'trap', 'eval',
        'exec', 'set', 'unset', 'alias', 'cd', 'pwd', 'mkdir', 'rmdir',
        'rm', 'cp', 'mv', 'ls', 'cat', 'grep', 'sed', 'awk', 'find',
        'curl', 'wget', 'ssh', 'scp', 'rsync', 'chmod', 'chown', 'sudo',
        'su', 'kill', 'killall', 'ps', 'top', 'mount', 'umount', 'df',
        'du', 'tar', 'gzip', 'gunzip', 'xz', 'cut', 'sort', 'uniq',
        'wc', 'head', 'tail', 'tee', 'xargs', 'basename', 'dirname',
        'test', 'true', 'false', 'read', 'printf', 'env', 'which',
    ]
    for i, kw in enumerate(bash[:64]):
        v[kw] = 16 + i

    # §4.2  HTML / CSS (80-143)
    html = [
        'html', 'head', 'body', 'div', 'span', 'script', 'style', 'link',
        'meta', 'title', 'a', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'thead', 'tbody',
        'form', 'input', 'button', 'select', 'option', 'textarea', 'label',
        'nav', 'section', 'article', 'aside', 'header', 'footer', 'main',
        'canvas', 'svg', 'img', 'video', 'audio', 'source', 'iframe',
        'doctype', 'charset', 'viewport', 'class', 'id', 'href', 'src',
        'type', 'rel', 'content', 'name', 'value', 'action', 'method',
        'onclick', 'onload', 'onerror', 'async', 'defer', 'fetch',
        'xmlhttprequest', 'localstorage', 'indexeddb', 'webgl', 'webgpu',
    ]
    for i, kw in enumerate(html[:64]):
        v[kw] = 80 + i

    # §4.3  ELF / Binary (144-207)
    elf = [
        '.text', '.data', '.bss', '.rodata', '.symtab', '.strtab',
        '.shstrtab', '.dynamic', '.got', '.plt', '.init', '.fini',
        '.debug_info', '.debug_line', '.debug_str', '.comment',
        'elf_magic', 'elfclass32', 'elfclass64', 'elfdata2lsb',
        'elfdata2msb', 'et_exec', 'et_dyn', 'et_core', 'et_rel',
        'pt_load', 'pt_dynamic', 'pt_interp', 'pt_note', 'pt_phdr',
        'sht_null', 'sht_progbits', 'sht_symtab', 'sht_strtab', 'sht_rela',
        'sht_hash', 'sht_dynamic', 'sht_note', 'sht_nobits', 'sht_rel',
        'appimage_magic', 'squashfs', 'runtime_stub', 'desktop_entry',
        'ld_linux', 'libc_so', 'libpthread', 'libdl', 'libm',
        'r_x86_64', 'r_aarch64', 'r_arm', 'entry_point', 'vaddr',
        'paddr', 'filesz', 'memsz', 'align', 'section_hdr', 'prog_hdr',
        'sym_bind_global', 'sym_bind_local', 'sym_type_func', 'sym_type_obj',
        'relocation', 'got_plt', 'plt_stub', 'dynamic_link',
    ]
    for i, kw in enumerate(elf[:64]):
        v[kw] = 144 + i

    # §4.4  Python (208-255)
    py = [
        'import', 'from', 'as', 'def', 'class', 'return', 'yield',
        'lambda', 'pass', 'break', 'continue', 'raise', 'try', 'except',
        'finally', 'with', 'assert', 'global', 'nonlocal', 'del',
        'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False',
        'print', 'open', 'range', 'len', 'type', 'isinstance', 'hasattr',
        'getattr', 'setattr', 'enumerate', 'zip', 'map', 'filter',
        'sorted', 'list', 'dict', 'set', 'tuple', 'int', 'float', 'str',
    ]
    for i, kw in enumerate(py[:48]):
        v[kw] = 208 + i

    # §4.5  Operators / Punctuation (256-383)
    ops = [
        '=', '==', '!=', '<', '<=', '>', '>=', '&&', '||', '!', '|',
        '&', '^', '~', '+', '-', '*', '/', '//', '%', '**', '<<', '>>',
        '(', ')', '[', ']', '{', '}', ';', ':', '.', ',', '@', '#',
        '$', '?', '->', '=>', '::', '...', '/*', '*/', '//', '\\',
        '+=', '-=', '*=', '/=', '|=', '&=', '^=', '<<=', '>>=',
        'pipe', 'redirect_out', 'redirect_in', 'redirect_append',
        'heredoc', 'herestring', 'subshell', 'background_job',
        'assignment', 'comparison', 'arithmetic_expand', 'brace_expand',
        'glob_star', 'glob_question', 'glob_bracket', 'tilde_expand',
        'single_quote', 'double_quote', 'backtick', 'dollar_paren',
        'dollar_brace', 'dollar_bracket', 'at_sign', 'hash_sign',
        'newline_tok', 'indent_tok', 'dedent_tok', 'eof_tok',
        'str_literal', 'int_literal', 'float_literal', 'bool_literal',
        'regex_literal', 'null_literal', 'shebang_line', 'comment_tok',
        'multiline_str', 'raw_string', 'bytes_literal', 'fstring',
        'tag_open', 'tag_close', 'tag_self_close', 'attr_eq',
        'css_selector', 'css_property', 'css_value', 'css_media',
        'js_arrow', 'js_spread', 'js_optional_chain', 'js_nullish',
        'bin_byte', 'bin_word', 'bin_dword', 'bin_qword',
        'bin_float', 'bin_double', 'bin_padding', 'bin_magic',
        'hex_literal', 'oct_literal', 'bin_literal', 'sci_literal',
        'ident', 'number', 'string', 'operator', 'keyword',
        'whitespace', 'punctuation', 'unknown_tok', 'format_marker',
    ]
    for i, kw in enumerate(ops[:128]):
        v[kw] = 256 + i

    return v


VOCAB: Dict[str, int] = _build_vocab()
_VOCAB_INV: Dict[int, str] = {v: k for k, v in VOCAB.items()}


def _hash_token(word: str) -> int:
    """
    Map any word not in the static vocabulary into the hash overflow range
    [384, 511] via DJB2 hash.  Deterministic, collision-bounded to 128 slots.
    """
    h = 5381
    for ch in word:
        h = ((h << 5) + h + ord(ch)) & 0x7FFFFFFF
    return 384 + (h % 128)


def _tok(word: str) -> int:
    """Resolve a word to its token ID."""
    return VOCAB.get(word.lower(), _hash_token(word))


# ══════════════════════════════════════════════════════════════════════════════
# §5  DATA STRUCTURES — header, section entry, artifact container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SectionEntry:
    """
    One row of the section table (64 bytes on disk).

    Layout:
      [0:16]  name       : null-padded ASCII
      [16:20] sec_type   : uint32 LE
      [20:24] sec_flags  : uint32 LE
      [24:32] offset     : uint64 LE — absolute byte offset in file
      [32:40] size       : uint64 LE — section size in bytes
      [40:48] vaddr      : uint64 LE — virtual address (VMEM_MAP sections)
      [48:52] align      : uint32 LE — byte alignment (always 8)
      [52:56] crc32      : uint32 LE — CRC32 of section data
      [56:64] reserved   : 8 bytes of zeros
    """
    name:     str
    sec_type: SectionType
    sec_flags: SectionFlags
    offset:   int = 0
    size:     int = 0
    vaddr:    int = 0
    align:    int = 8
    crc32:    int = 0

    def pack(self) -> bytes:
        name_b = self.name.encode('ascii')[:15].ljust(16, b'\x00')
        return (name_b
                + struct.pack('<IIQQQQII',
                              int(self.sec_type),
                              int(self.sec_flags),
                              self.offset,
                              self.size,
                              self.vaddr,
                              0,       # padding (was two 32-bit fields → one 64)
                              self.align,
                              self.crc32))

    @classmethod
    def unpack(cls, raw: bytes) -> 'SectionEntry':
        assert len(raw) == 64, f"SectionEntry must be 64 bytes, got {len(raw)}"
        name      = raw[0:16].rstrip(b'\x00').decode('ascii', errors='replace')
        (sec_type, sec_flags,
         offset, size, vaddr, _pad,
         align, crc32)  = struct.unpack('<IIQQQQII', raw[16:64])
        return cls(name=name,
                   sec_type=SectionType(sec_type),
                   sec_flags=SectionFlags(sec_flags),
                   offset=offset, size=size, vaddr=vaddr,
                   align=align, crc32=crc32)


@dataclass
class RecuArtifact:
    """
    In-memory representation of a fully parsed .recu file.
    All section payloads are stored as raw bytes — the loader decodes them.
    """
    source_format:  SourceFormat
    capabilities:   CapabilityFlags
    lifecycle_state: LifecycleState
    manifest:       Dict[str, Any]
    payload_raw:    bytes          # decompressed original source / binary
    embedding:      List[float]    # 64 float32 — L2-normalised document vector
    cap_logits:     List[float]    # 16 float32 — raw routing network output
    cap_probs:      List[float]    # 16 float32 — sigmoid(logits)
    ca_grid:        List[int]      # 256 ints (0/1) — 16×16 CA state
    ca_rule:        str            # e.g. 'B3/S23'
    ca_generation:  int
    vmem_regions:   List[Dict]     # decoded VMEM_MAP entries
    syscall_table:  Dict[str, Any] # decoded SYSCALL_TABLE
    section_hashes: List[bytes]    # 7 × 32 bytes — SHA-256 per section
    artifact_id:    str            # hex SHA-256 of decompressed payload


# ══════════════════════════════════════════════════════════════════════════════
# §6  CRC32 ENGINE — IEEE 802.3 (ethernet) polynomial, lookup-table method
#     Polynomial: 0xEDB88320 (bit-reflected form of 0x04C11DB7)
#     Initial value: 0xFFFFFFFF  |  Final XOR: 0xFFFFFFFF
# ══════════════════════════════════════════════════════════════════════════════

def _build_crc32_table() -> List[int]:
    table = [0] * 256
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if (crc & 1) else (crc >> 1)
        table[i] = crc
    return table


_CRC32_TABLE: List[int] = _build_crc32_table()


def _crc32(data: bytes, init: int = 0xFFFFFFFF) -> int:
    crc = init
    for byte in data:
        crc = (crc >> 8) ^ _CRC32_TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


# ══════════════════════════════════════════════════════════════════════════════
# §7  PACKBITS RLE CODEC — lossless, handles all byte values including 0x00
#
#  Encoding rules (Apple PackBits, used in TIFF and PostScript):
#    Header 0x00-0x7F : literal run — the next (header + 1) bytes are literal
#    Header 0x80      : no-op (reserved, skipped on decode)
#    Header 0x81-0xFF : RLE run — repeat next byte (257 − header) times
#
#  This handles arbitrary binary data without any escape ambiguity.
# ══════════════════════════════════════════════════════════════════════════════

def _rle_encode(data: bytes) -> bytes:
    """PackBits encode.  Returns compressed bytes."""
    out = bytearray()
    i   = 0
    n   = len(data)

    while i < n:
        # Probe for a run of identical bytes starting at i
        if i + 1 < n and data[i] == data[i + 1]:
            j = i + 2
            while j < n and data[j] == data[i] and (j - i) < 128:
                j += 1
            run_len = j - i                    # 2 ≤ run_len ≤ 128
            out.append((257 - run_len) & 0xFF) # header: 0x81-0xFF
            out.append(data[i])
            i = j
        else:
            # Collect a literal run (no adjacent repeat ahead)
            j = i + 1
            while j < n and (j + 1 >= n or data[j] != data[j + 1]) and (j - i) < 128:
                j += 1
            lit_len = j - i                    # 1 ≤ lit_len ≤ 128
            out.append(lit_len - 1)            # header: 0x00-0x7F
            out.extend(data[i:j])
            i = j

    return bytes(out)


def _rle_decode(data: bytes) -> bytes:
    """PackBits decode.  Returns decompressed bytes."""
    out = bytearray()
    i   = 0
    n   = len(data)

    while i < n:
        header = data[i]; i += 1
        if header <= 127:
            count = header + 1
            if i + count > n:
                raise ValueError(f"PackBits literal overrun at offset {i}")
            out.extend(data[i:i + count]); i += count
        elif header == 128:
            pass  # no-op
        else:
            count = 257 - header
            if i >= n:
                raise ValueError(f"PackBits RLE overrun at offset {i}")
            out.extend([data[i]] * count); i += 1

    return bytes(out)


# ══════════════════════════════════════════════════════════════════════════════
# §8  SOURCE DETECTOR — identifies input format from magic bytes, extensions,
#     shebang lines, and structural patterns
# ══════════════════════════════════════════════════════════════════════════════

# ELF magic: \x7fELF  (offset 0, 4 bytes)
_ELF_MAGIC     = b'\x7fELF'
# AppImage magic: ELF magic at 0 + b'AI\x02' at offset 8
_APPIMAGE_MARK = b'AI\x02'
# Bash shebang patterns
_BASH_SHEBANGS = (b'#!/bin/bash', b'#!/bin/sh', b'#!/usr/bin/env bash',
                  b'#!/usr/bin/env sh')
# Python shebang patterns
_PY_SHEBANGS   = (b'#!/usr/bin/env python', b'#!/usr/bin/python')


class SourceDetector:
    """
    Stateless format detector.  Operates on raw bytes; never touches the
    filesystem except when a path is supplied (reads a sample header).
    """

    @staticmethod
    def detect_bytes(raw: bytes) -> SourceFormat:
        if len(raw) < 4:
            return SourceFormat.PLAIN_TEXT

        # ELF binary
        if raw[:4] == _ELF_MAGIC:
            if len(raw) > 11 and raw[8:11] == _APPIMAGE_MARK:
                return SourceFormat.APPIMAGE
            return SourceFormat.ELF

        # Shebangs (text-based scripts)
        header = raw[:64]
        for shebang in _BASH_SHEBANGS:
            if header.startswith(shebang):
                return SourceFormat.BASH
        for shebang in _PY_SHEBANGS:
            if header.startswith(shebang):
                return SourceFormat.PYTHON

        # HTML detection (look for DOCTYPE or <html within first 512 bytes)
        try:
            snippet = raw[:512].decode('utf-8', errors='ignore').lower().strip()
            if snippet.startswith('<!doctype html') or snippet.startswith('<html'):
                return SourceFormat.HTML
            if '<html' in snippet or '<head' in snippet or '<body' in snippet:
                return SourceFormat.HTML
        except Exception:
            pass

        # Python without shebang (structural markers)
        try:
            text = raw[:1024].decode('utf-8', errors='ignore')
            py_markers = ('def ', 'class ', 'import ', 'from ', 'if __name__')
            if sum(1 for m in py_markers if m in text) >= 2:
                return SourceFormat.PYTHON
        except Exception:
            pass

        # Bash without shebang
        try:
            text = raw[:1024].decode('utf-8', errors='ignore')
            sh_markers = ('echo ', 'export ', 'fi\n', 'done\n', 'esac\n')
            if sum(1 for m in sh_markers if m in text) >= 2:
                return SourceFormat.BASH
        except Exception:
            pass

        # High-entropy binary → plain binary (treat as ELF-like unknown)
        non_printable = sum(1 for b in raw[:256] if b < 0x09 or (0x0e <= b <= 0x1f))
        if non_printable > 32:
            return SourceFormat.UNKNOWN

        return SourceFormat.PLAIN_TEXT

    @staticmethod
    def detect_path(path: str) -> SourceFormat:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.sh', '.bash'):           return SourceFormat.BASH
        if ext in ('.html', '.htm', '.xhtml'):return SourceFormat.HTML
        if ext in ('.py', '.pyw'):            return SourceFormat.PYTHON
        if ext == '.appimage':                return SourceFormat.APPIMAGE
        # Fall through to byte inspection
        try:
            with open(path, 'rb') as fh:
                return SourceDetector.detect_bytes(fh.read(512))
        except OSError as exc:
            raise IOError(f"Cannot read {path}: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# §9  TOKENIZER — converts source content to a sequence of vocabulary token IDs
#
#  Each source format has a specialised strategy:
#    BASH      : keyword scan + operator scan + word hash
#    HTML      : tag extraction + attribute hash + text hash
#    ELF       : section/program header names + byte marker tokens
#    APPIMAGE  : ELF header tokens + AppImage-specific markers
#    PYTHON    : keyword scan + import graph + operator scan
#    PLAIN_TEXT: word hash only
# ══════════════════════════════════════════════════════════════════════════════

_TOKEN_SPLIT_RE = re.compile(r'[\s\t\n\r,;()\[\]{}<>=!&|^~+\-*/%@#$?:\\]+')
_HTML_TAG_RE    = re.compile(r'</?([a-zA-Z][a-zA-Z0-9\-]*)(?:\s[^>]*)?>',
                              re.DOTALL | re.IGNORECASE)
_HTML_ATTR_RE   = re.compile(r'\b([a-zA-Z][a-zA-Z0-9\-]*)=', re.IGNORECASE)
_HTML_SCRIPT_RE = re.compile(r'<script[^>]*>(.*?)</script>',
                              re.DOTALL | re.IGNORECASE)


class Tokenizer:
    """
    Converts raw source bytes into a flat list of integer token IDs.
    The result is used as input to the EmbeddingEngine.

    A [CLS] token is prepended and [SEP] appended.
    A format-specific sentinel token ([BASH], [HTML], etc.) is inserted
    immediately after [CLS] so the embedding space is format-aware.
    """

    _FORMAT_TOKEN: Dict[SourceFormat, str] = {
        SourceFormat.BASH:       '[BASH]',
        SourceFormat.HTML:       '[HTML]',
        SourceFormat.ELF:        '[ELF]',
        SourceFormat.APPIMAGE:   '[APPIMAGE]',
        SourceFormat.PYTHON:     '[PYTHON]',
        SourceFormat.PLAIN_TEXT: '[TEXT]',
        SourceFormat.UNKNOWN:    '[UNK]',
    }

    def tokenize(self, raw: bytes, fmt: SourceFormat) -> List[int]:
        tokens: List[int] = [_tok('[CLS]'), _tok(self._FORMAT_TOKEN[fmt])]

        if fmt in (SourceFormat.ELF, SourceFormat.APPIMAGE):
            tokens.extend(self._tokenize_binary(raw, fmt))
        else:
            try:
                text = raw.decode('utf-8', errors='replace')
            except Exception:
                text = ''
            if fmt == SourceFormat.BASH:
                tokens.extend(self._tokenize_bash(text))
            elif fmt == SourceFormat.HTML:
                tokens.extend(self._tokenize_html(text))
            elif fmt == SourceFormat.PYTHON:
                tokens.extend(self._tokenize_python(text))
            else:
                tokens.extend(self._tokenize_plain(text))

        tokens.append(_tok('[SEP]'))
        return tokens

    # ── Per-format strategies ────────────────────────────────────────────────

    def _tokenize_bash(self, text: str) -> List[int]:
        result: List[int] = []
        # Prepend lifecycle token for shell scripts (they spawn processes)
        result.append(_tok('[RUNNING]'))
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                result.append(_tok('comment_tok'))
                continue
            if line.startswith('#!/'):
                result.append(_tok('shebang_line'))
            for word in _TOKEN_SPLIT_RE.split(line):
                if word:
                    result.append(_tok(word))
        return result

    def _tokenize_html(self, text: str) -> List[int]:
        result: List[int] = []
        result.append(_tok('[RUNNING]'))
        for match in _HTML_TAG_RE.finditer(text):
            result.append(_tok(match.group(1).lower()))
        for match in _HTML_ATTR_RE.finditer(text):
            result.append(_tok(match.group(1).lower()))
        # Tokenise script blocks as bash-like
        for m in _HTML_SCRIPT_RE.finditer(text):
            for word in _TOKEN_SPLIT_RE.split(m.group(1)):
                if word:
                    result.append(_tok(word))
        return result

    def _tokenize_python(self, text: str) -> List[int]:
        result: List[int] = []
        result.append(_tok('[RUNNING]'))
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                result.append(_tok('comment_tok'))
                continue
            if stripped.startswith('def '):
                result.append(_tok('def'))
            if stripped.startswith('class '):
                result.append(_tok('class'))
            if stripped.startswith('import ') or stripped.startswith('from '):
                result.append(_tok('import'))
            for word in _TOKEN_SPLIT_RE.split(stripped):
                if word:
                    result.append(_tok(word))
        return result

    def _tokenize_plain(self, text: str) -> List[int]:
        result: List[int] = []
        for word in _TOKEN_SPLIT_RE.split(text):
            if word:
                result.append(_tok(word))
        return result

    def _tokenize_binary(self, raw: bytes, fmt: SourceFormat) -> List[int]:
        result: List[int] = []
        if fmt == SourceFormat.APPIMAGE:
            result.append(_tok('appimage_magic'))
        result.append(_tok('elf_magic'))

        if len(raw) >= 6:
            ei_class = raw[4]   # 1=32-bit, 2=64-bit
            ei_data  = raw[5]   # 1=LE, 2=BE
            result.append(_tok('elfclass64' if ei_class == 2 else 'elfclass32'))
            result.append(_tok('elfdata2lsb' if ei_data == 1 else 'elfdata2msb'))

        if len(raw) >= 18:
            e_type, = struct.unpack_from('<H', raw, 16)
            type_names = {1: 'et_rel', 2: 'et_exec', 3: 'et_dyn', 4: 'et_core'}
            result.append(_tok(type_names.get(e_type, 'bin_magic')))

        # ELF64 section header parsing — best-effort, no crash on corrupt input
        if len(raw) >= 64:
            try:
                e_shoff, = struct.unpack_from('<Q', raw, 40)
                e_shnum, = struct.unpack_from('<H', raw, 60)
                e_shstrndx, = struct.unpack_from('<H', raw, 62)
                if e_shoff > 0 and e_shnum > 0 and e_shstrndx < e_shnum:
                    strtab_off_entry = e_shoff + e_shstrndx * 64
                    if strtab_off_entry + 64 <= len(raw):
                        strtab_data_off, = struct.unpack_from('<Q', raw, strtab_off_entry + 24)
                        for sh in range(min(e_shnum, 32)):
                            entry_off = e_shoff + sh * 64
                            if entry_off + 64 > len(raw): break
                            sh_name_off, = struct.unpack_from('<I', raw, entry_off)
                            name_start = strtab_data_off + sh_name_off
                            name_end   = raw.find(b'\x00', name_start)
                            if 0 < name_start < len(raw) and name_end > name_start:
                                sec_name = raw[name_start:name_end].decode(
                                    'ascii', errors='ignore')
                                result.append(_tok(sec_name))
            except (struct.error, ValueError):
                pass

        result.append(_tok('section_hdr'))
        result.append(_tok('prog_hdr'))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# §10  EMBEDDING ENGINE
#
#  Document embedding:
#    E ∈ ℝ^{512 × 64}  — vocabulary embedding matrix
#    Init: E[i, ·] ~ N(0, σ²)  where σ = 1/√64
#           with seed_i = (i · 2654435761 + 1) mod 2^64  (Knuth mult. hash)
#
#    Token sequence T = [t₀, t₁, …, t_{n-1}]  (integers in [0, 511])
#
#    Mean pool:    e_raw   = (1/n) Σᵢ E[tᵢ]    ∈ ℝ^64
#    L2 normalise: ê_doc   = e_raw / ‖e_raw‖₂   (unit sphere)
#
#  The embedding matrix is computed once per EmbeddingEngine instance and
#  cached.  Since initialisation is seeded deterministically, the same matrix
#  is produced on every machine.
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Computes deterministic semantic embeddings for source artifacts.
    Thread-safe after construction (embedding matrix is read-only).
    """

    def __init__(self) -> None:
        self._E: List[float] = self._init_embedding_matrix()

    @staticmethod
    def _init_embedding_matrix() -> List[float]:
        """
        Build E ∈ ℝ^{512×64} via He-style init for embeddings.
        σ = 1/√64 ≈ 0.125

        Per-token seed: seed_i = (i · φ + 1) & UINT64_MAX
        where φ = 2654435761 is a near-integer approximation to 2^32/φ_gold
        (Knuth multiplicative hash constant, guarantees uniform distribution).
        """
        rng   = _CompilerRNG()
        scale = 1.0 / _sqrt(float(EMBED_DIM))
        E: List[float] = [0.0] * (VOCAB_SIZE * EMBED_DIM)
        for tok_id in range(VOCAB_SIZE):
            seed_i = (tok_id * 2654435761 + 1) & 0xFFFFFFFFFFFFFFFF
            rng.seed(seed_i)
            base   = tok_id * EMBED_DIM
            for d in range(EMBED_DIM):
                E[base + d] = rng.randn() * scale
        return E

    def embed(self, tokens: List[int]) -> Tuple[List[float], float]:
        """
        Embed a token sequence via mean-pool + L2 normalise.

        Returns:
            (ê_doc, original_norm)
            ê_doc        : List[float] length EMBED_DIM, unit vector
            original_norm: ‖e_raw‖₂ before normalisation (stored in EMBEDDING
                           section header for distance-based retrieval)
        """
        n = len(tokens)
        if n == 0:
            return ([0.0] * EMBED_DIM, 0.0)

        acc = [0.0] * EMBED_DIM
        E   = self._E
        for tok in tokens:
            tok_clamped = max(0, min(VOCAB_SIZE - 1, tok))
            base = tok_clamped * EMBED_DIM
            for d in range(EMBED_DIM):
                acc[d] += E[base + d]

        inv_n = 1.0 / n
        acc   = [x * inv_n for x in acc]

        norm_sq = sum(x * x for x in acc)
        norm    = _sqrt(norm_sq)

        if norm < 1e-9:
            return (acc, 0.0)

        inv_norm = 1.0 / norm
        e_hat    = [x * inv_norm for x in acc]
        return (e_hat, norm)


# ══════════════════════════════════════════════════════════════════════════════
# §11  CAPABILITY CLASSIFIER
#
#  Two-stage pipeline:
#    Stage 1 — Rule-based (deterministic, ground truth):
#      Analyses source text / binary for known capability patterns.
#      Returns CapabilityFlags bitmask.
#
#    Stage 2 — Neural soft vote (probabilistic, fine-tuneable):
#      2-layer MLP:   h = ReLU(W₁ ê + b₁),   logits = W₂ h + b₂
#      W₁ ∈ ℝ^{128×64}  seeded 0xDEADBEEF42424242  (He init σ=√(2/64))
#      W₂ ∈ ℝ^{16×128}  seeded 0xCAFEBABE13131313  (He init σ=√(2/128))
#      probs = σ(logits)
#      High-confidence neural bits (prob > 0.75) OR-ed onto rule mask.
#
#  Combined:  caps = rule_caps | (neural_caps & high_conf_mask)
# ══════════════════════════════════════════════════════════════════════════════

class CapabilityClassifier:
    """
    Classifies artifact capabilities from its source text and document embedding.
    Returns capability bitmask, raw logits, and per-capability probabilities.
    """

    def __init__(self) -> None:
        self._W1, self._b1, self._W2, self._b2 = self._init_router()

    @staticmethod
    def _init_router() -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        He initialisation for routing MLP weights.
        All four weight tensors are seeded independently with fixed 64-bit seeds.
        """
        rng    = _CompilerRNG()
        scale1 = _sqrt(2.0 / EMBED_DIM)       # He init, ReLU activation
        scale2 = _sqrt(2.0 / ROUTER_HIDDEN)

        rng.seed(0xDEADBEEF42424242)
        W1 = [rng.randn() * scale1 for _ in range(ROUTER_HIDDEN * EMBED_DIM)]
        b1 = [0.0] * ROUTER_HIDDEN

        rng.seed(0xCAFEBABE13131313)
        W2 = [rng.randn() * scale2 for _ in range(N_CAPABILITIES * ROUTER_HIDDEN)]
        b2 = [0.0] * N_CAPABILITIES

        return W1, b1, W2, b2

    @staticmethod
    def _matvec(W: List[float], b: List[float],
                v: List[float], out_dim: int, in_dim: int) -> List[float]:
        """Dense matrix-vector multiply + bias:  result = W @ v + b."""
        result = [0.0] * out_dim
        for i in range(out_dim):
            s    = b[i]
            base = i * in_dim
            for j in range(in_dim):
                s += W[base + j] * v[j]
            result[i] = s
        return result

    def _neural_forward(self, e_doc: List[float]) -> Tuple[List[float], List[float]]:
        """
        Forward pass through routing MLP.
        Returns (logits, probs) — both List[float] of length N_CAPABILITIES.
        """
        h      = self._matvec(self._W1, self._b1, e_doc, ROUTER_HIDDEN, EMBED_DIM)
        h      = [_relu(x) for x in h]
        logits = self._matvec(self._W2, self._b2, h, N_CAPABILITIES, ROUTER_HIDDEN)
        probs  = [_sigmoid(x) for x in logits]
        return logits, probs

    @staticmethod
    def _rule_classify(content: str, fmt: SourceFormat) -> int:
        """
        Rule-based capability extraction.  Operates on decoded source text.
        Returns CapabilityFlags bitmask.
        """
        caps = 0
        c    = content.lower()

        if fmt == SourceFormat.BASH:
            caps |= int(CapabilityFlags.SYSTEM_CALL)
            if any(kw in c for kw in ('cat ', 'read ', 'head ', 'tail ',
                                       'grep ', 'ls ', 'find ', 'awk ')):
                caps |= int(CapabilityFlags.FILE_READ)
            if any(kw in c for kw in ('echo ', '>> ', 'tee ', 'dd ',
                                       'write ', '> ')):
                caps |= int(CapabilityFlags.FILE_WRITE)
            if any(kw in c for kw in ('curl ', 'wget ', 'ssh ', 'nc ',
                                       'netcat ', 'ncat ', 'telnet ', 'nmap ')):
                caps |= int(CapabilityFlags.NETWORK)
            if any(kw in c for kw in ('exec ', 'fork', ' & ', 'nohup ',
                                       'screen ', 'tmux ', 'spawn')):
                caps |= int(CapabilityFlags.PROCESS_SPAWN)
            if any(kw in c for kw in ('chmod ', 'chown ', 'mount ', 'umount ',
                                       'sudo ', 'su ', 'insmod ', 'modprobe ')):
                caps |= int(CapabilityFlags.KERNEL_DIRECT)
            if any(kw in c for kw in ('openssl ', 'gpg ', 'sha256', 'sha512',
                                       'md5sum', 'crypt', 'age ')):
                caps |= int(CapabilityFlags.CRYPTO)
            if re.search(r'\$\(\s*date\b|\bsleep\b|\bat\b', c):
                caps |= int(CapabilityFlags.TIMER)

        elif fmt == SourceFormat.HTML:
            caps |= int(CapabilityFlags.UI_RENDER)
            if '<script' in c:
                caps |= int(CapabilityFlags.COMPUTE_HEAVY)
            if any(kw in c for kw in ('fetch(', 'xmlhttp', 'axios', 'websocket')):
                caps |= int(CapabilityFlags.NETWORK)
            if any(kw in c for kw in ('indexeddb', 'localstorage', 'sessionstorage')):
                caps |= int(CapabilityFlags.DATABASE)
            if any(kw in c for kw in ('<canvas', 'webgl', 'webgpu', 'three.js')):
                caps |= int(CapabilityFlags.GRAPHICS)
            if 'serviceworker' in c or 'webworker' in c:
                caps |= int(CapabilityFlags.INTERPROCESS)

        elif fmt in (SourceFormat.ELF, SourceFormat.APPIMAGE):
            caps |= int(CapabilityFlags.KERNEL_DIRECT)
            caps |= int(CapabilityFlags.MEMORY_MAP)
            caps |= int(CapabilityFlags.PROCESS_SPAWN)
            if fmt == SourceFormat.APPIMAGE:
                caps |= int(CapabilityFlags.UI_RENDER)

        elif fmt == SourceFormat.PYTHON:
            caps |= int(CapabilityFlags.COMPUTE_HEAVY)
            if 'open(' in c:
                caps |= int(CapabilityFlags.FILE_READ) | int(CapabilityFlags.FILE_WRITE)
            if any(kw in c for kw in ('socket', 'requests', 'urllib', 'httpx',
                                       'aiohttp', 'httplib', 'http.client')):
                caps |= int(CapabilityFlags.NETWORK)
            if any(kw in c for kw in ('subprocess', 'os.system', 'popen', 'execv')):
                caps |= int(CapabilityFlags.PROCESS_SPAWN)
            if any(kw in c for kw in ('threading', 'multiprocessing', 'asyncio')):
                caps |= int(CapabilityFlags.INTERPROCESS)
            if any(kw in c for kw in ('sqlite3', 'psycopg2', 'mysql', 'pymongo',
                                       'sqlalchemy', 'redis')):
                caps |= int(CapabilityFlags.DATABASE)
            if any(kw in c for kw in ('hashlib', 'cryptography', 'nacl', 'hmac',
                                       'rsa', 'ecdsa')):
                caps |= int(CapabilityFlags.CRYPTO)
            if any(kw in c for kw in ('mmap', 'ctypes', 'cffi', 'struct.pack')):
                caps |= int(CapabilityFlags.MEMORY_MAP)
            if any(kw in c for kw in ('tkinter', 'pyqt', 'wx', 'pyglet', 'pygame')):
                caps |= int(CapabilityFlags.UI_RENDER) | int(CapabilityFlags.GRAPHICS)

        return caps

    def classify(self, raw: bytes, fmt: SourceFormat,
                 e_doc: List[float]) -> Tuple[int, List[float], List[float]]:
        """
        Classify capabilities via rule + neural combination.

        Returns:
            (capability_mask, logits, probs)
            capability_mask : CapabilityFlags OR-combination
            logits          : raw MLP output (16 floats)
            probs           : sigmoid(logits)  (16 floats)
        """
        try:
            text = raw.decode('utf-8', errors='replace') if fmt not in (
                SourceFormat.ELF, SourceFormat.APPIMAGE) else ''
        except Exception:
            text = ''

        rule_mask   = self._rule_classify(text, fmt)
        logits, probs = self._neural_forward(e_doc)

        # High-confidence neural bits (prob > 0.75) supplement the rule mask
        neural_mask = 0
        for i, p in enumerate(probs):
            if p > 0.75:
                neural_mask |= (1 << i)

        combined = rule_mask | neural_mask
        return (combined, logits, probs)


# ══════════════════════════════════════════════════════════════════════════════
# §12  MANIFEST BUILDER
#     Produces the MANIFEST JSON section: a complete, self-describing agent
#     identity document.  All fields populated — no placeholder values.
# ══════════════════════════════════════════════════════════════════════════════

class ManifestBuilder:
    """Builds the agent manifest from all compiled artefact metadata."""

    @staticmethod
    def _cap_names(mask: int) -> List[str]:
        return [f.name for f in CapabilityFlags if f.value & mask and f.value]

    @staticmethod
    def _derive_description(fmt: SourceFormat, cap_names: List[str]) -> str:
        cap_str = ', '.join(cap_names[:4]) + ('...' if len(cap_names) > 4 else '')
        descs   = {
            SourceFormat.BASH:       f"Shell script agent. Capabilities: {cap_str}.",
            SourceFormat.HTML:       f"Web rendering agent. Capabilities: {cap_str}.",
            SourceFormat.ELF:        f"Native binary agent. Capabilities: {cap_str}.",
            SourceFormat.APPIMAGE:   f"AppImage portable agent. Capabilities: {cap_str}.",
            SourceFormat.PYTHON:     f"Python interpreter agent. Capabilities: {cap_str}.",
            SourceFormat.PLAIN_TEXT: f"Plain-text data agent. Capabilities: {cap_str}.",
            SourceFormat.UNKNOWN:    f"Unknown format agent. Capabilities: {cap_str}.",
        }
        return descs[fmt]

    def build(self,
              name: str,
              fmt: SourceFormat,
              caps_mask: int,
              token_count: int,
              embed_norm: float,
              payload_original_size: int,
              payload_compressed_size: int,
              payload_sha256: str) -> Dict[str, Any]:

        cap_names = self._cap_names(caps_mask)
        now_iso   = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        # Permission model: derived from capability mask
        file_perms: Dict[str, Any] = {"read": [], "write": []}
        if caps_mask & int(CapabilityFlags.FILE_READ):
            file_perms["read"] = ["*"]
        if caps_mask & int(CapabilityFlags.FILE_WRITE):
            file_perms["write"] = ["*"]

        net_perms: Dict[str, Any] = {"enabled": False, "allowed_hosts": [], "allowed_ports": []}
        if caps_mask & int(CapabilityFlags.NETWORK):
            net_perms["enabled"] = True

        proc_perms: Dict[str, Any] = {"spawn": False, "max_forks": 0}
        if caps_mask & int(CapabilityFlags.PROCESS_SPAWN):
            proc_perms["spawn"] = True
            proc_perms["max_forks"] = 16

        mem_perms: Dict[str, Any] = {
            "max_bytes": 64 * 1024 * 1024,   # 64 MiB default budget
            "allow_mmap": bool(caps_mask & int(CapabilityFlags.MEMORY_MAP)),
        }

        return {
            "format":           "recu",
            "version":          f"{RECU_VERSION_MAJ}.{RECU_VERSION_MIN}",
            "artifact_id":      payload_sha256,
            "name":             name,
            "source_format":    fmt.value,
            "compiled_at":      now_iso,
            "compiler_version": f"{RECU_VERSION_MAJ}.{RECU_VERSION_MIN}.0",

            "agent": {
                "name":            name,
                "description":     self._derive_description(fmt, cap_names),
                "capabilities":    cap_names,
                "capability_mask": caps_mask,
                "permissions": {
                    "file_system": file_perms,
                    "network":     net_perms,
                    "process":     proc_perms,
                    "memory":      mem_perms,
                    "kernel_direct": bool(caps_mask & int(CapabilityFlags.KERNEL_DIRECT)),
                    "crypto":        bool(caps_mask & int(CapabilityFlags.CRYPTO)),
                    "ui_render":     bool(caps_mask & int(CapabilityFlags.UI_RENDER)),
                    "graphics":      bool(caps_mask & int(CapabilityFlags.GRAPHICS)),
                    "database":      bool(caps_mask & int(CapabilityFlags.DATABASE)),
                    "timer":         bool(caps_mask & int(CapabilityFlags.TIMER)),
                    "agent_spawn":   bool(caps_mask & int(CapabilityFlags.AGENT_SPAWN)),
                },
            },

            "kernel_bindings": {
                "entry_point":  "0x00010000",
                "stack_pointer":"0x00034000",
                "heap_start":   "0x00070000",
                "heap_size":    "0x00100000",
                "data_start":   "0x00020000",
            },

            "embedding": {
                "dimension":     EMBED_DIM,
                "vocab_size":    VOCAB_SIZE,
                "token_count":   token_count,
                "pre_norm":      round(embed_norm, 8),
            },

            "lifecycle": {
                "ca_rule":       CA_RULE_DEFAULT,
                "ca_rows":       CA_ROWS,
                "ca_cols":       CA_COLS,
                "initial_state": LifecycleState.DORMANT.name.lower(),
            },

            "payload": {
                "original_size":    payload_original_size,
                "compressed_size":  payload_compressed_size,
                "compression":      "packbits_rle",
                "sha256":           payload_sha256,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# §13  CA LIFECYCLE BUILDER
#
#  Each LifecycleState maps to a distinct 16×16 B3/S23 CA pattern.
#  The pattern encodes the artifact's initial vitality at load time:
#
#   DORMANT    → empty grid       (quiescent, awaiting activation)
#   LOADING    → glider at (1,1)  (expanding, moving — purposeful arrival)
#   RUNNING    → blinker at (7,6) (oscillating — alive, processing)
#   SUSPENDED  → 2×2 block (6,6)  (stable still life — paused but intact)
#   FAULTED    → seeded chaos     (entropy 0.99 — erratic, dying)
#   TERMINATED → empty grid       (generation_count preserved, pop=0)
#
#  CA_STATE section format (288 bytes total):
#    [0:16]    rule_string  : null-padded ASCII (e.g. b'B3/S23\x00...')
#    [16:272]  grid         : 256 bytes (16×16, row-major, 0=dead 1=alive)
#    [272:276] generation   : uint32 LE
#    [276:280] initial_state: uint32 LE (LifecycleState.value)
#    [280:284] rows         : uint32 LE (16)
#    [284:288] cols         : uint32 LE (16)
# ══════════════════════════════════════════════════════════════════════════════

# Pattern constants (identical to CellularAutomatonEngine class attributes)
_GLIDER  = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
_BLINKER = [(0, 0), (0, 1), (0, 2)]
_BLOCK   = [(0, 0), (0, 1), (1, 0), (1, 1)]
_BEACON  = [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)]


def _place_pattern(grid: List[int], rows: int, cols: int,
                   pattern: List[Tuple[int, int]], row: int, col: int) -> None:
    for dr, dc in pattern:
        r = (row + dr) % rows
        c = (col + dc) % cols
        grid[r * cols + c] = 1


class CALifecycleBuilder:
    """Encodes LifecycleState into a CA grid and serialises to bytes."""

    def build(self, initial_state: LifecycleState = LifecycleState.DORMANT) -> bytes:
        """
        Returns CA_STATE section bytes (288 bytes) for the given initial state.
        """
        grid = [0] * (CA_ROWS * CA_COLS)

        if initial_state == LifecycleState.DORMANT:
            pass   # all zeros

        elif initial_state == LifecycleState.LOADING:
            _place_pattern(grid, CA_ROWS, CA_COLS, _GLIDER, 1, 1)

        elif initial_state == LifecycleState.RUNNING:
            _place_pattern(grid, CA_ROWS, CA_COLS, _BLINKER, 7, 6)
            _place_pattern(grid, CA_ROWS, CA_COLS, _BLINKER, 3, 2)  # second oscillator

        elif initial_state == LifecycleState.SUSPENDED:
            _place_pattern(grid, CA_ROWS, CA_COLS, _BLOCK, 6, 6)
            _place_pattern(grid, CA_ROWS, CA_COLS, _BLOCK, 10, 10)  # two stable blocks

        elif initial_state == LifecycleState.FAULTED:
            rng = _CompilerRNG(0xDEAD0000F0000000)
            for i in range(CA_ROWS * CA_COLS):
                grid[i] = 1 if rng.random() < 0.35 else 0

        elif initial_state == LifecycleState.TERMINATED:
            pass   # all zeros — generation count still preserved in metadata

        rule_bytes = CA_RULE_DEFAULT.encode('ascii').ljust(16, b'\x00')[:16]
        grid_bytes = bytes(grid)
        tail       = struct.pack('<IIII',
                                 0,               # generation = 0 at compile time
                                 initial_state.value,
                                 CA_ROWS,
                                 CA_COLS)
        return rule_bytes + grid_bytes + tail   # 16 + 256 + 16 = 288 bytes

    def ca_step(self, grid: List[int]) -> List[int]:
        """
        Advance CA grid one generation under B3/S23.
        Used at runtime to update lifecycle state.
        """
        new_grid  = [0] * (CA_ROWS * CA_COLS)
        birth     = {3}
        survive   = {2, 3}
        for r in range(CA_ROWS):
            for c in range(CA_COLS):
                idx  = r * CA_COLS + c
                live = grid[idx]
                n    = sum(
                    grid[((r + dr) % CA_ROWS) * CA_COLS + ((c + dc) % CA_COLS)]
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                )
                if live:
                    new_grid[idx] = 1 if n in survive else 0
                else:
                    new_grid[idx] = 1 if n in birth   else 0
        return new_grid

    @staticmethod
    def infer_state(grid: List[int], generation: int) -> LifecycleState:
        """
        Map current CA grid metrics to a LifecycleState.
        Uses population and Shannon entropy:
          pop=0              → DORMANT or TERMINATED (by generation)
          entropy < 0.15     → SUSPENDED (stable, low entropy)
          0.15 ≤ entropy < 0.85 → RUNNING
          entropy ≥ 0.85     → FAULTED (chaotic)
        """
        total = len(grid)
        live  = sum(grid)
        if live == 0:
            return LifecycleState.DORMANT if generation == 0 else LifecycleState.TERMINATED
        p1 = live / total;  p0 = 1.0 - p1
        entropy = 0.0
        if p1 > 0.0: entropy -= p1 * _log(p1)
        if p0 > 0.0: entropy -= p0 * _log(p0)
        # Shannon entropy for binary system ∈ [0, ln(2) ≈ 0.693]
        entropy_norm = entropy / 0.693147   # normalise to [0, 1]
        if entropy_norm < 0.15:
            return LifecycleState.SUSPENDED
        if entropy_norm >= 0.85:
            return LifecycleState.FAULTED
        return LifecycleState.RUNNING


# ══════════════════════════════════════════════════════════════════════════════
# §14  VIRTUAL MEMORY MAP BUILDER
#
#  Assigns virtual address regions to the loaded artifact.
#  Address space layout (AIOS .recu process):
#
#   0x00010000  AGENT_CODE    64 KiB   R-X   payload executable segment
#   0x00020000  AGENT_DATA    64 KiB   RW-   read-write static data
#   0x00030000  AGENT_STACK   16 KiB   RW-   stack (grows down from +16K)
#   0x00040000  SHARED_MEM    64 KiB   RWS   kernel shared memory window
#   0x00050000  EMBED_ROM      4 KiB   R--   semantic embedding (read-only)
#   0x00060000  MANIFEST_ROM   4 KiB   R--   agent manifest (read-only)
#   0x00070000  HEAP           1 MiB   RW-   dynamic allocation arena
#
#  VMEM_MAP section: 7 entries × 40 bytes = 280 bytes
#  Each entry:
#    [0:16]  name     : null-padded ASCII
#    [16:24] vaddr    : uint64 LE
#    [24:32] size     : uint64 LE
#    [32:36] flags    : uint32 LE  (VMEM_R=1, VMEM_W=2, VMEM_X=4, VMEM_S=8)
#    [36:40] reserved : uint32 (zeros)
# ══════════════════════════════════════════════════════════════════════════════

_VMEM_LAYOUT = [
    # (name,            vaddr,       size,           flags)
    ('AGENT_CODE',    0x00010000, 0x00010000, VMEM_R | VMEM_X),
    ('AGENT_DATA',    0x00020000, 0x00010000, VMEM_R | VMEM_W),
    ('AGENT_STACK',   0x00030000, 0x00004000, VMEM_R | VMEM_W),
    ('SHARED_MEM',    0x00040000, 0x00010000, VMEM_R | VMEM_W | VMEM_S),
    ('EMBED_ROM',     0x00050000, 0x00001000, VMEM_R),
    ('MANIFEST_ROM',  0x00060000, 0x00001000, VMEM_R),
    ('HEAP',          0x00070000, 0x00100000, VMEM_R | VMEM_W),
]


class VMemMapBuilder:
    """Serialises the virtual address layout into the VMEM_MAP section."""

    def build(self) -> bytes:
        out = bytearray()
        for (name, vaddr, size, flags) in _VMEM_LAYOUT:
            name_b = name.encode('ascii').ljust(16, b'\x00')[:16]
            out   += name_b + struct.pack('<QQII', vaddr, size, flags, 0)
        assert len(out) == len(_VMEM_LAYOUT) * VMEM_ENTRY_SIZE
        return bytes(out)

    @staticmethod
    def decode(raw: bytes) -> List[Dict[str, Any]]:
        regions = []
        n       = len(raw) // VMEM_ENTRY_SIZE
        for i in range(n):
            entry  = raw[i * VMEM_ENTRY_SIZE:(i + 1) * VMEM_ENTRY_SIZE]
            name   = entry[0:16].rstrip(b'\x00').decode('ascii', errors='replace')
            vaddr, size, flags, _ = struct.unpack('<QQII', entry[16:40])
            regions.append({
                'name':   name,
                'vaddr':  vaddr,
                'size':   size,
                'flags':  flags,
                'readable':   bool(flags & VMEM_R),
                'writable':   bool(flags & VMEM_W),
                'executable': bool(flags & VMEM_X),
                'shared':     bool(flags & VMEM_S),
            })
        return regions


# ══════════════════════════════════════════════════════════════════════════════
# §15  SYSCALL TABLE BUILDER
#
#  Maps artifact-declared syscalls to AIOS kernel agent tool names.
#  The capability mask determines which syscall bindings are included.
#
#  SYSCALL_TABLE section: JSON-encoded (UTF-8 bytes), not fixed-width.
#  This is intentional — syscall semantics are richly described, and
#  JSON is human-readable in a hex dump, aiding debugging.
# ══════════════════════════════════════════════════════════════════════════════

# Master binding table: (syscall_name, aios_tool, required_capability, description)
_SYSCALL_BINDINGS = [
    # FILE I/O
    ('read_bytes',    'bus.peek_buf',    CapabilityFlags.FILE_READ,
     'Read N bytes from virtual address'),
    ('write_bytes',   'bus.poke_buf',    CapabilityFlags.FILE_WRITE,
     'Write N bytes to virtual address'),
    ('alloc_page',    'palloc.alloc',    CapabilityFlags.MEMORY_MAP,
     'Allocate N contiguous physical pages'),
    ('free_page',     'palloc.free',     CapabilityFlags.MEMORY_MAP,
     'Release N previously allocated pages'),
    # PROCESS
    ('spawn',         'kernel.dispatch', CapabilityFlags.PROCESS_SPAWN,
     'Spawn child agent by tool name'),
    ('plan',          'kernel.plan_execute', CapabilityFlags.AGENT_SPAWN,
     'Plan and execute a natural-language goal'),
    # NETWORK (routed through kernel HAL — no direct socket fd exposure)
    ('net_send',      'hal.net_tx',      CapabilityFlags.NETWORK,
     'Transmit bytes via kernel network HAL'),
    ('net_recv',      'hal.net_rx',      CapabilityFlags.NETWORK,
     'Receive bytes via kernel network HAL'),
    # TIMER
    ('get_time_ns',   'hal.monotonic_ns', CapabilityFlags.TIMER,
     'Read monotonic clock in nanoseconds'),
    ('sleep_ns',      'hal.sleep_ns',    CapabilityFlags.TIMER,
     'Sleep for N nanoseconds'),
    # UI / GRAPHICS
    ('vga_write',     'vga.writeln',     CapabilityFlags.UI_RENDER,
     'Write line to VGA text buffer'),
    ('framebuf_blit', 'hal.fb_blit',     CapabilityFlags.GRAPHICS,
     'Blit pixel rectangle to framebuffer'),
    # KERNEL
    ('syscall',       'kernel.syscall',  CapabilityFlags.KERNEL_DIRECT,
     'Invoke raw kernel syscall interface'),
    ('interrupt',     'idt.raise',       CapabilityFlags.KERNEL_DIRECT,
     'Raise software interrupt vector'),
    # CRYPTO
    ('sha256',        'crypto.sha256',   CapabilityFlags.CRYPTO,
     'Compute SHA-256 digest over byte range'),
    # DATABASE
    ('db_query',      'db.query',        CapabilityFlags.DATABASE,
     'Execute structured query against agent-local store'),
    # IPC
    ('msg_send',      'ipc.send',        CapabilityFlags.INTERPROCESS,
     'Send message to peer agent by name'),
    ('msg_recv',      'ipc.recv',        CapabilityFlags.INTERPROCESS,
     'Receive message from IPC queue'),
]


class SyscallTableBuilder:
    """Builds the SYSCALL_TABLE JSON section from the capability mask."""

    def build(self, caps_mask: int) -> bytes:
        bindings = []
        for (syscall, tool, req_cap, desc) in _SYSCALL_BINDINGS:
            if caps_mask & int(req_cap):
                bindings.append({
                    'syscall_name':        syscall,
                    'aios_tool':           tool,
                    'required_capability': req_cap.name,
                    'description':         desc,
                    'bound':               True,
                })
        table = {
            'version':  f'{RECU_VERSION_MAJ}.{RECU_VERSION_MIN}',
            'n_bound':  len(bindings),
            'bindings': bindings,
        }
        return json.dumps(table, separators=(',', ':')).encode('utf-8')


# ══════════════════════════════════════════════════════════════════════════════
# §16  PACKER
#
#  Serialises all eight sections into the .recu binary format.
#
#  Two-pass approach:
#    Pass 1: build all section payloads in memory, record sizes
#    Pass 2: compute offsets, write header + section table + data + signature
#
#  File integrity:
#    • Each section has its own CRC32 stored in its section table entry
#    • The SIGNATURE section holds SHA-256 of each of the first seven sections
#    • After writing everything, SHA-256 over the entire file (with digest
#      field zeroed) is placed at header offset [32:64]
# ══════════════════════════════════════════════════════════════════════════════

class RecuPacker:
    """
    Serialises an artifact's sections into a .recu byte stream.

    Usage:
        packer = RecuPacker()
        raw    = packer.pack(sections_dict)
    """

    @staticmethod
    def _align8(n: int) -> int:
        """Round up to next multiple of 8."""
        return (n + 7) & ~7

    def pack(self, sections: Dict[str, bytes]) -> bytes:
        """
        Build the .recu binary from a dict of section_name → section_bytes.
        sections must contain exactly RECU_N_SECTIONS keys, ordered as:
          MANIFEST, PAYLOAD, EMBEDDING, ROUTING, CA_STATE,
          VMEM_MAP, SYSCALL_TABLE, SIGNATURE
        """
        expected_order = [
            'MANIFEST', 'PAYLOAD', 'EMBEDDING', 'ROUTING',
            'CA_STATE', 'VMEM_MAP', 'SYSCALL_TABLE', 'SIGNATURE',
        ]
        # SIGNATURE is computed here from the other 7 sections
        if 'SIGNATURE' not in sections or not sections['SIGNATURE']:
            sections['SIGNATURE'] = self._build_signature_section(
                [sections[k] for k in expected_order[:-1]])

        # Compute offsets: data starts at RECU_DATA_OFF = 0x240
        offset = RECU_DATA_OFF
        entries: List[SectionEntry] = []

        for i, name in enumerate(expected_order):
            data     = sections[name]
            data_len = len(data)
            crc      = _crc32(data)

            # Virtual address — for EMBED_ROM and MANIFEST_ROM we use the
            # layout from §14; other sections get 0 (loader maps them as needed)
            vaddr_map = {
                'MANIFEST':  0x00060000,
                'EMBEDDING': 0x00050000,
            }
            vaddr = vaddr_map.get(name, 0)

            # Derive SectionType and flags
            sec_type = SectionType[name] if name in SectionType.__members__ else SectionType(i + 1)
            sec_flag = SectionFlags.READ
            if name == 'PAYLOAD':
                sec_flag |= SectionFlags.COMPRESSED
            if name in ('MANIFEST', 'SYSCALL_TABLE', 'SIGNATURE'):
                pass  # read-only
            if name in ('EMBEDDING', 'ROUTING'):
                pass  # read-only tensors

            entry = SectionEntry(
                name=name[:15],
                sec_type=sec_type,
                sec_flags=sec_flag,
                offset=offset,
                size=data_len,
                vaddr=vaddr,
                align=8,
                crc32=crc,
            )
            entries.append(entry)
            offset += self._align8(data_len)

        # ── Build section table ───────────────────────────────────────────────
        section_table_bytes = b''.join(e.pack() for e in entries)
        assert len(section_table_bytes) == RECU_N_SECTIONS * RECU_SECT_ENTRY_SIZE

        # ── Build data blob (section payloads, each padded to 8-byte boundary)
        data_blob = bytearray()
        for name in expected_order:
            raw   = sections[name]
            data_blob.extend(raw)
            pad   = self._align8(len(raw)) - len(raw)
            data_blob.extend(b'\x00' * pad)

        # ── Assemble file (SHA-256 digest field zeroed for now) ───────────────
        header = (RECU_MAGIC
                  + struct.pack('<HH', RECU_VERSION_MAJ, RECU_VERSION_MIN)
                  + struct.pack('<I', 0)                             # flags
                  + struct.pack('<Q', time.time_ns())                # timestamp
                  + struct.pack('<I', RECU_N_SECTIONS)
                  + struct.pack('<I', RECU_HEADER_SIZE)
                  + b'\x00' * 32)                                    # sha256 placeholder
        assert len(header) == RECU_HEADER_SIZE

        full = bytearray(header) + bytearray(section_table_bytes) + data_blob

        # ── Compute and write file-level SHA-256 ─────────────────────────────
        digest = hashlib.sha256(bytes(full)).digest()
        full[32:64] = digest

        return bytes(full)

    @staticmethod
    def _build_signature_section(section_payloads: List[bytes]) -> bytes:
        """
        Build SIGNATURE section:
          [0:4]          version  = 1
          [4:8]          n_sections = 7
          [8 : 8+7*32]   SHA-256 of each of the first 7 sections (order preserved)
          [232:264]       hash_chain_root: SHA-256 of the sequential concatenation
                          of all 7 per-section hashes (Merkle-like chain)
        Total: 264 bytes.
        """
        assert len(section_payloads) == 7, \
            f"Signature requires exactly 7 sections, got {len(section_payloads)}"
        hashes: List[bytes] = [hashlib.sha256(s).digest() for s in section_payloads]

        # Hash chain: rolling SHA-256(prev_hash || current_hash)
        chain = hashes[0]
        for h in hashes[1:]:
            chain = hashlib.sha256(chain + h).digest()

        sig = struct.pack('<II', 1, 7)     # version=1, n_sections=7
        for h in hashes:
            sig += h
        sig += chain
        return sig


# ══════════════════════════════════════════════════════════════════════════════
# §17  LOADER — deserialises a .recu binary back into a RecuArtifact
# ══════════════════════════════════════════════════════════════════════════════

class RecuLoader:
    """
    Parses a .recu byte stream and validates its integrity.
    On success, returns a populated RecuArtifact.
    On failure, raises ValueError with a precise fault description.
    """

    def load(self, data: bytes) -> RecuArtifact:
        self._validate_header(data)
        entries = self._parse_section_table(data)
        sects   = {e.name: self._extract_section(data, e) for e in entries}

        self._validate_crcs(entries, sects)
        self._validate_signature(sects)

        manifest       = json.loads(sects['MANIFEST'].decode('utf-8'))
        payload_rle    = sects['PAYLOAD']
        payload_raw    = _rle_decode(payload_rle)
        embedding      = self._decode_embedding(sects['EMBEDDING'])
        logits, probs, cap_mask = self._decode_routing(sects['ROUTING'])
        ca_grid, ca_rule, ca_gen, ca_init = self._decode_ca(sects['CA_STATE'])
        vmem           = VMemMapBuilder.decode(sects['VMEM_MAP'])
        syscall_table  = json.loads(sects['SYSCALL_TABLE'].decode('utf-8'))
        section_hashes = self._decode_signature(sects['SIGNATURE'])

        fmt = SourceFormat(manifest.get('source_format', 'unknown'))
        lc  = LifecycleState(ca_init)

        return RecuArtifact(
            source_format   = fmt,
            capabilities    = CapabilityFlags(cap_mask),
            lifecycle_state = lc,
            manifest        = manifest,
            payload_raw     = payload_raw,
            embedding       = embedding,
            cap_logits      = logits,
            cap_probs       = probs,
            ca_grid         = ca_grid,
            ca_rule         = ca_rule,
            ca_generation   = ca_gen,
            vmem_regions    = vmem,
            syscall_table   = syscall_table,
            section_hashes  = section_hashes,
            artifact_id     = manifest.get('artifact_id', ''),
        )

    @staticmethod
    def _validate_header(data: bytes) -> None:
        if len(data) < RECU_HEADER_SIZE:
            raise ValueError(f".recu file too small: {len(data)} bytes")
        if data[:8] != RECU_MAGIC:
            raise ValueError(f"Bad magic: {data[:8].hex()!r}")
        maj, min_ = struct.unpack('<HH', data[8:12])
        if maj != RECU_VERSION_MAJ:
            raise ValueError(f"Incompatible version: {maj}.{min_}")
        # Verify file-level SHA-256
        stored_digest = bytes(data[32:64])
        scrubbed      = bytearray(data)
        scrubbed[32:64] = b'\x00' * 32
        computed      = hashlib.sha256(bytes(scrubbed)).digest()
        if computed != stored_digest:
            raise ValueError(
                f"File integrity check failed.\n"
                f"  Stored  SHA-256: {stored_digest.hex()}\n"
                f"  Computed SHA-256: {computed.hex()}")

    @staticmethod
    def _parse_section_table(data: bytes) -> List[SectionEntry]:
        n_sects, = struct.unpack('<I', data[24:28])
        entries  = []
        for i in range(n_sects):
            off   = RECU_SECT_TABLE_OFF + i * RECU_SECT_ENTRY_SIZE
            raw   = data[off: off + RECU_SECT_ENTRY_SIZE]
            entries.append(SectionEntry.unpack(raw))
        return entries

    @staticmethod
    def _extract_section(data: bytes, entry: SectionEntry) -> bytes:
        return data[entry.offset: entry.offset + entry.size]

    @staticmethod
    def _validate_crcs(entries: List[SectionEntry], sects: Dict[str, bytes]) -> None:
        for e in entries:
            payload = sects.get(e.name, b'')
            computed = _crc32(payload)
            if computed != e.crc32:
                raise ValueError(
                    f"CRC32 mismatch in section {e.name!r}: "
                    f"expected 0x{e.crc32:08X}, got 0x{computed:08X}")

    @staticmethod
    def _validate_signature(sects: Dict[str, bytes]) -> None:
        sig_raw   = sects.get('SIGNATURE', b'')
        if len(sig_raw) < 8:
            raise ValueError("SIGNATURE section too short")
        n_sects_in_sig, = struct.unpack('<I', sig_raw[4:8])
        expected_order  = ['MANIFEST', 'PAYLOAD', 'EMBEDDING', 'ROUTING',
                           'CA_STATE', 'VMEM_MAP', 'SYSCALL_TABLE']
        for i, name in enumerate(expected_order[:n_sects_in_sig]):
            stored_hash   = sig_raw[8 + i * 32: 8 + (i + 1) * 32]
            computed_hash = hashlib.sha256(sects.get(name, b'')).digest()
            if computed_hash != stored_hash:
                raise ValueError(f"Signature mismatch for section {name!r}")

    @staticmethod
    def _decode_embedding(raw: bytes) -> List[float]:
        # Header layout: dim(4) + token_count(4) + pre_norm(4) + caps_mask(4) = 16 bytes
        if len(raw) < 16 + EMBED_DIM * 4:
            raise ValueError(f"EMBEDDING section too small: {len(raw)}")
        floats = struct.unpack_from(f'<{EMBED_DIM}f', raw, 16)
        return list(floats)

    @staticmethod
    def _decode_routing(raw: bytes) -> Tuple[List[float], List[float], int]:
        if len(raw) < 16 + N_CAPABILITIES * 8:
            raise ValueError(f"ROUTING section too small: {len(raw)}")
        cap_mask, = struct.unpack_from('<I', raw, 12)
        logits    = list(struct.unpack_from(f'<{N_CAPABILITIES}f', raw, 16))
        probs     = list(struct.unpack_from(f'<{N_CAPABILITIES}f', raw, 16 + N_CAPABILITIES * 4))
        return logits, probs, cap_mask

    @staticmethod
    def _decode_ca(raw: bytes) -> Tuple[List[int], str, int, int]:
        if len(raw) < 288:
            raise ValueError(f"CA_STATE section too small: {len(raw)}")
        rule_str    = raw[0:16].rstrip(b'\x00').decode('ascii', errors='replace')
        grid        = list(raw[16:272])
        generation, initial_state, rows, cols = struct.unpack('<IIII', raw[272:288])
        return grid, rule_str, generation, initial_state

    @staticmethod
    def _decode_signature(raw: bytes) -> List[bytes]:
        if len(raw) < 8:
            return []
        n_sects, = struct.unpack('<I', raw[4:8])
        hashes   = []
        for i in range(n_sects):
            hashes.append(bytes(raw[8 + i * 32: 8 + (i + 1) * 32]))
        return hashes


# ══════════════════════════════════════════════════════════════════════════════
# §18  RECU EXECUTION CONTEXT
#
#  Manages the lifecycle of a loaded .recu artifact under the AIOS kernel.
#  When aios_core is available, registers as an @agent_method callable tool.
#
#  Execution strategy per source format:
#    BASH / PYTHON / PLAIN_TEXT  → subprocess under capability sandbox
#    HTML                        → emit to stdout (headless / VGA render)
#    ELF / APPIMAGE              → subprocess (binary execution)
#    UNKNOWN                     → reject (log + raise)
#
#  CA lifecycle is ticked on each significant event (launch, syscall, exit).
#  The inferred state is written back into the artifact's CA grid.
# ══════════════════════════════════════════════════════════════════════════════

class RecuExecutionContext:
    """
    Binds a RecuArtifact to the AIOS kernel and manages its execution lifecycle.
    """

    def __init__(self, artifact: RecuArtifact, kernel: Any = None) -> None:
        self._artifact  = artifact
        self._kernel    = kernel
        self._ca_engine = CALifecycleBuilder()
        self._ca_grid   = list(artifact.ca_grid)
        self._ca_gen    = artifact.ca_generation
        self._lock      = threading.Lock()
        self._stdout    : List[str] = []
        self._stderr    : List[str] = []

        if _CORE_AVAILABLE and kernel is not None:
            self._register_with_kernel()

    def _register_with_kernel(self) -> None:
        """Insert this artifact's execute entry into the agent tool registry."""
        artifact_id = self._artifact.artifact_id
        name        = self._artifact.manifest.get('agent', {}).get('name', artifact_id[:8])

        def _exec_tool(**_kw: Any) -> SysCallResult:
            return self.execute()

        if _registry is not None:
            from aios_core import AgentToolSpec, AgentPriority as _AP
            spec = AgentToolSpec(
                name        = f'recu.{name}',
                description = self._artifact.manifest.get('agent', {}).get('description', ''),
                parameters  = {},
                returns     = 'SysCallResult',
                priority    = _AP.NORMAL,
                fn          = _exec_tool,
                owner       = 'recu_loader',
            )
            _registry.register(spec)

    def _tick_lifecycle(self) -> LifecycleState:
        """Advance CA one generation and infer new lifecycle state."""
        with self._lock:
            self._ca_grid = self._ca_engine.ca_step(self._ca_grid)
            self._ca_gen += 1
            state = CALifecycleBuilder.infer_state(self._ca_grid, self._ca_gen)
            self._artifact = RecuArtifact(  # immutable update
                **{**self._artifact.__dict__,
                   'lifecycle_state': state,
                   'ca_grid':         self._ca_grid,
                   'ca_generation':   self._ca_gen})
            return state

    def _resolve_interpreter(self) -> Optional[List[str]]:
        fmt = self._artifact.source_format
        if fmt == SourceFormat.BASH:       return ['bash']
        if fmt == SourceFormat.PYTHON:     return [sys.executable]
        if fmt == SourceFormat.ELF:        return []       # execute directly
        if fmt == SourceFormat.APPIMAGE:   return []
        if fmt == SourceFormat.HTML:       return None     # stdout render
        if fmt == SourceFormat.PLAIN_TEXT: return ['cat']
        return None

    def execute(self, env: Optional[Dict[str, str]] = None,
                timeout_s: float = 30.0) -> SysCallResult:
        """
        Execute the artifact payload within the declared capability sandbox.

        Returns SysCallResult with:
            value = {'stdout': str, 'stderr': str, 'returncode': int,
                     'lifecycle': str, 'ca_generation': int}
        """
        fmt  = self._artifact.source_format

        # Transition to LOADING
        with self._lock:
            self._ca_grid = [0] * (CA_ROWS * CA_COLS)
            _place_pattern(self._ca_grid, CA_ROWS, CA_COLS, _GLIDER, 1, 1)
            self._ca_gen = 0

        interp = self._resolve_interpreter()

        if interp is None and fmt == SourceFormat.UNKNOWN:
            return SysCallResult(
                success=False, value=None,
                error=f"Cannot execute unknown source format: {fmt.value}")

        payload = self._artifact.payload_raw

        if fmt == SourceFormat.HTML:
            # HTML artifacts render to stdout as text
            self._tick_lifecycle()
            return SysCallResult(
                success=True,
                value={'stdout': payload.decode('utf-8', errors='replace'),
                       'stderr': '',
                       'returncode': 0,
                       'lifecycle': LifecycleState.RUNNING.name,
                       'ca_generation': self._ca_gen})

        # Write payload to a temporary file and execute
        suffix_map = {
            SourceFormat.BASH:       '.sh',
            SourceFormat.PYTHON:     '.py',
            SourceFormat.ELF:        '',
            SourceFormat.APPIMAGE:   '.AppImage',
            SourceFormat.PLAIN_TEXT: '.txt',
        }
        suffix = suffix_map.get(fmt, '')
        try:
            with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False, mode='wb') as tf:
                tf.write(payload)
                tf_path = tf.name

            # Make executable for binary formats
            if fmt in (SourceFormat.ELF, SourceFormat.APPIMAGE, SourceFormat.BASH):
                os.chmod(tf_path, 0o755)

            cmd = (interp + [tf_path]) if interp else [tf_path]
            run_env = dict(os.environ)
            if env:
                run_env.update(env)

            self._tick_lifecycle()  # RUNNING

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_s,
                env=run_env,
            )
            self._ca_grid = [0] * (CA_ROWS * CA_COLS)  # TERMINATED
            self._ca_gen += 1
            final_state = LifecycleState.TERMINATED

            return SysCallResult(
                success=(result.returncode == 0),
                value={
                    'stdout':       result.stdout.decode('utf-8', errors='replace'),
                    'stderr':       result.stderr.decode('utf-8', errors='replace'),
                    'returncode':   result.returncode,
                    'lifecycle':    final_state.name,
                    'ca_generation': self._ca_gen,
                },
                error=None if result.returncode == 0 else
                      f"Process exited {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            # FAULTED state: inject chaos into CA grid
            rng = _CompilerRNG(0xDEAD00007171E0FF + self._ca_gen)
            self._ca_grid = [1 if rng.random() < 0.35 else 0
                             for _ in range(CA_ROWS * CA_COLS)]
            self._ca_gen += 1
            return SysCallResult(
                success=False, value=None,
                error=f"Execution timeout ({timeout_s}s) — lifecycle: FAULTED")
        except Exception as exc:
            return SysCallResult(success=False, value=None, error=str(exc))
        finally:
            try:
                os.unlink(tf_path)
            except Exception:
                pass

    def lifecycle_status(self) -> Dict[str, Any]:
        """Return a snapshot of the current CA lifecycle state."""
        state = CALifecycleBuilder.infer_state(self._ca_grid, self._ca_gen)
        pop   = sum(self._ca_grid)
        total = CA_ROWS * CA_COLS
        p1    = pop / total if total else 0.0
        p0    = 1.0 - p1
        ent   = 0.0
        if p1 > 0.0: ent -= p1 * _log(p1)
        if p0 > 0.0: ent -= p0 * _log(p0)
        return {
            'state':       state.name,
            'generation':  self._ca_gen,
            'population':  pop,
            'entropy':     round(ent / 0.693147, 4),
            'ca_rule':     self._artifact.ca_rule,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §19  COMPILER — full pipeline: source → .recu bytes
# ══════════════════════════════════════════════════════════════════════════════

class RecuCompiler:
    """
    Top-level compiler.  Instantiate once, call compile() for each artifact.

    Subsystem instances are shared across calls — embedding matrix and routing
    weights are initialised once and reused, giving consistent semantic space
    across all compiled artifacts in a session.
    """

    def __init__(self) -> None:
        self._detector    = SourceDetector()
        self._tokenizer   = Tokenizer()
        self._embedder    = EmbeddingEngine()
        self._classifier  = CapabilityClassifier()
        self._manifest_b  = ManifestBuilder()
        self._ca_builder  = CALifecycleBuilder()
        self._vmem_b      = VMemMapBuilder()
        self._syscall_b   = SyscallTableBuilder()
        self._packer      = RecuPacker()

    def compile(self, source_path: str,
                output_path: Optional[str] = None) -> str:
        """
        Compile source_path into a .recu artifact.

        Args:
            source_path : path to input file (any supported format)
            output_path : optional output path; defaults to source + '.recu'

        Returns:
            Absolute path to the written .recu file.
        """
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source not found: {source_path}")

        with open(source_path, 'rb') as fh:
            raw_source = fh.read()

        # ── Phase 1: Detection ────────────────────────────────────────────────
        fmt  = SourceDetector.detect_path(source_path)
        name = os.path.splitext(os.path.basename(source_path))[0]

        # ── Phase 2: Tokenisation ─────────────────────────────────────────────
        tokens = self._tokenizer.tokenize(raw_source, fmt)

        # ── Phase 3: Semantic embedding ───────────────────────────────────────
        e_doc, pre_norm = self._embedder.embed(tokens)

        # ── Phase 4: Capability classification ───────────────────────────────
        caps_mask, logits, probs = self._classifier.classify(
            raw_source, fmt, e_doc)

        # ── Phase 5: Payload compression ─────────────────────────────────────
        payload_rle     = _rle_encode(raw_source)
        payload_sha256  = hashlib.sha256(raw_source).hexdigest()

        # ── Phase 6: Manifest construction ───────────────────────────────────
        manifest_dict = self._manifest_b.build(
            name                   = name,
            fmt                    = fmt,
            caps_mask              = caps_mask,
            token_count            = len(tokens),
            embed_norm             = pre_norm,
            payload_original_size  = len(raw_source),
            payload_compressed_size= len(payload_rle),
            payload_sha256         = payload_sha256,
        )
        manifest_bytes = json.dumps(manifest_dict,
                                    indent=2).encode('utf-8')

        # ── Phase 7: EMBEDDING section ────────────────────────────────────────
        embed_bytes = (struct.pack('<III', EMBED_DIM, len(tokens), 0)   # 12 header bytes
                       + struct.pack('<f', pre_norm)                     # replaces padding
                       + struct.pack(f'<{EMBED_DIM}f', *e_doc))
        # Correct: header = dim(4) + token_count(4) + reserved(4) = 12, then pre_norm(4),
        # then EMBED_DIM floats (256 bytes). Total = 272 bytes.
        # Re-layout cleanly:
        embed_bytes = (struct.pack('<I', EMBED_DIM)
                       + struct.pack('<I', len(tokens))
                       + struct.pack('<f', pre_norm)
                       + struct.pack('<I', caps_mask)
                       + struct.pack(f'<{EMBED_DIM}f', *e_doc))
        # 4+4+4+4+256 = 272 bytes

        # ── Phase 8: ROUTING section ──────────────────────────────────────────
        routing_bytes = (struct.pack('<I', 1)           # version
                         + struct.pack('<I', N_CAPABILITIES)
                         + struct.pack('<I', ROUTER_HIDDEN)
                         + struct.pack('<I', caps_mask)
                         + struct.pack(f'<{N_CAPABILITIES}f', *logits)
                         + struct.pack(f'<{N_CAPABILITIES}f', *probs))
        # 4+4+4+4+64+64 = 144 bytes

        # ── Phase 9: CA lifecycle grid ────────────────────────────────────────
        ca_bytes = self._ca_builder.build(LifecycleState.DORMANT)

        # ── Phase 10: Virtual memory map ─────────────────────────────────────
        vmem_bytes = self._vmem_b.build()

        # ── Phase 11: Syscall table ───────────────────────────────────────────
        syscall_bytes = self._syscall_b.build(caps_mask)

        # ── Phase 12: Pack all sections ───────────────────────────────────────
        sections = {
            'MANIFEST':     manifest_bytes,
            'PAYLOAD':      payload_rle,
            'EMBEDDING':    embed_bytes,
            'ROUTING':      routing_bytes,
            'CA_STATE':     ca_bytes,
            'VMEM_MAP':     vmem_bytes,
            'SYSCALL_TABLE': syscall_bytes,
            'SIGNATURE':    b'',   # RecuPacker computes this
        }
        recu_bytes = self._packer.pack(sections)

        # ── Write output ──────────────────────────────────────────────────────
        if output_path is None:
            output_path = source_path + '.recu'
        output_path = os.path.abspath(output_path)

        with open(output_path, 'wb') as fh:
            fh.write(recu_bytes)

        return output_path

    def load(self, recu_path: str,
             kernel: Any = None) -> RecuExecutionContext:
        """
        Load a .recu artifact and return an execution context.

        Args:
            recu_path : path to .recu file
            kernel    : optional AgentKernel instance for tool registration

        Returns:
            RecuExecutionContext ready for execute()
        """
        with open(recu_path, 'rb') as fh:
            raw = fh.read()
        artifact = RecuLoader().load(raw)
        return RecuExecutionContext(artifact, kernel)

    def inspect(self, recu_path: str) -> Dict[str, Any]:
        """
        Return a structured inspection report for a .recu artifact.
        Does not execute the artifact.
        """
        with open(recu_path, 'rb') as fh:
            raw = fh.read()
        artifact = RecuLoader().load(raw)
        a = artifact
        return {
            'artifact_id':    a.artifact_id,
            'source_format':  a.source_format.value,
            'capabilities':   [f.name for f in CapabilityFlags
                               if f.value & int(a.capabilities) and f.value],
            'capability_mask': int(a.capabilities),
            'lifecycle_state': a.lifecycle_state.name,
            'ca_rule':         a.ca_rule,
            'ca_population':   sum(a.ca_grid),
            'embedding_dim':   len(a.embedding),
            'payload_bytes':   len(a.payload_raw),
            'vmem_regions':    [{r['name']: hex(r['vaddr'])} for r in a.vmem_regions],
            'syscall_count':   a.syscall_table.get('n_bound', 0),
            'file_size_bytes': len(raw),
            'agent_name':      a.manifest.get('agent', {}).get('name', '?'),
            'compiled_at':     a.manifest.get('compiled_at', '?'),
        }


# ══════════════════════════════════════════════════════════════════════════════
# §20  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _print_banner() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║  AIOS .recu Compiler  v{}.{}.0                  ║".format(
          RECU_VERSION_MAJ, RECU_VERSION_MIN))
    print("║  Recursive Execution Unit — agentic artifact     ║")
    print("╚══════════════════════════════════════════════════╝")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog='recu',
        description='AIOS .recu compiler — converts scripts and binaries into '
                    'agentic artifacts with embedded semantic identity.')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # recu compile <source> [-o output]
    p_compile = sub.add_parser('compile', help='Compile source file to .recu')
    p_compile.add_argument('source', help='Input file path')
    p_compile.add_argument('-o', '--output', default=None,
                           help='Output .recu path (default: source + .recu)')

    # recu inspect <artifact.recu>
    p_inspect = sub.add_parser('inspect', help='Inspect a .recu artifact')
    p_inspect.add_argument('artifact', help='.recu file path')

    # recu run <artifact.recu>
    p_run = sub.add_parser('run', help='Execute a .recu artifact')
    p_run.add_argument('artifact', help='.recu file path')
    p_run.add_argument('--timeout', type=float, default=30.0,
                       help='Execution timeout in seconds (default: 30)')

    args = parser.parse_args(argv)
    compiler = RecuCompiler()
    _print_banner()

    if args.cmd == 'compile':
        try:
            out = compiler.compile(args.source, args.output)
            print(f"\n  ✓ Compiled → {out}")
            report = compiler.inspect(out)
            print(f"  ▸ Format    : {report['source_format']}")
            print(f"  ▸ Artifact  : {report['artifact_id'][:16]}…")
            print(f"  ▸ Size      : {report['file_size_bytes']:,} bytes")
            print(f"  ▸ Payload   : {report['payload_bytes']:,} bytes")
            print(f"  ▸ Caps      : {', '.join(report['capabilities']) or 'none'}")
            print(f"  ▸ Syscalls  : {report['syscall_count']} bound")
            print(f"  ▸ Lifecycle : {report['lifecycle_state']}")
            return 0
        except Exception as exc:
            print(f"\n  ✗ Compilation failed: {exc}", file=sys.stderr)
            return 1

    elif args.cmd == 'inspect':
        try:
            report = compiler.inspect(args.artifact)
            print(f"\n  .recu Artifact Inspection Report")
            print(f"  {'─'*50}")
            for k, v in report.items():
                print(f"  {k:<22}: {v}")
            return 0
        except Exception as exc:
            print(f"\n  ✗ Inspection failed: {exc}", file=sys.stderr)
            return 1

    elif args.cmd == 'run':
        try:
            ctx    = compiler.load(args.artifact)
            result = ctx.execute(timeout_s=args.timeout)
            print(f"\n  Execution result: {'✓ success' if result.success else '✗ failed'}")
            if result.value:
                v = result.value
                print(f"  Return code  : {v.get('returncode', '?')}")
                print(f"  Lifecycle    : {v.get('lifecycle', '?')}")
                print(f"  CA generation: {v.get('ca_generation', 0)}")
                if v.get('stdout'):
                    print(f"\n  ── stdout ──\n{v['stdout']}")
                if v.get('stderr'):
                    print(f"\n  ── stderr ──\n{v['stderr']}")
            if result.error:
                print(f"  Error: {result.error}", file=sys.stderr)
            return 0 if result.success else 1
        except Exception as exc:
            print(f"\n  ✗ Execution failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
