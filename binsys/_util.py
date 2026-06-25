"""Shared utilities for binsys: constants, helpers, subprocess wrapper, metadata."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

logger = logging.getLogger("binsys")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

STORE = Path.home() / ".binsys"
IMAGES = STORE / "images"
MOUNTS = STORE / "mounts"

REPO_DIR = Path(__file__).resolve().parent.parent
BOOT_DIR = REPO_DIR / "boot"
EFI_BIN = BOOT_DIR / "target" / "x86_64-unknown-uefi" / "release" / "puppyboot.efi"
SCRIPTS_DIR = REPO_DIR / "scripts"

WIZARD_SCRIPTS: list[tuple[str, str]] = [
    ("build-frugal", "Build a frugal overlay system (base.sfs + save.img)"),
    ("build-iso", "Create an ISO9660 image from a system or directory"),
    ("quick-vm", "Launch a VM with hardware presets"),
    ("snapshot-manager", "Manage frugal save-layer snapshots"),
]

OVMF_CANDIDATES: list[tuple[str, str | None]] = [
    ("/usr/share/OVMF/OVMF_CODE_4M.fd", "/usr/share/OVMF/OVMF_VARS_4M.fd"),
    ("/usr/share/OVMF/OVMF_CODE.fd", "/usr/share/OVMF/OVMF_VARS.fd"),
    ("/usr/share/edk2-ovmf/x64/OVMF_CODE.fd", "/usr/share/edk2-ovmf/x64/OVMF_VARS.fd"),
    ("/usr/share/qemu/OVMF.fd", None),
]

SIZE_PRESETS: dict[str, str] = {
    "nano": "256M",
    "mini": "512M",
    "small": "1G",
    "medium": "2G",
    "large": "4G",
    "xl": "8G",
    "huge": "16G",
}

TYPES: list[str] = ["ext4", "overlay", "squashfs", "fat32", "frugal", "iso"]

DEFAULT_KEYBINDINGS: dict[str, str] = {
    "new": "n",
    "delete": "d",
    "run": "r",
    "mount": "m",
    "snap": "s",
    "import": "i",
    "frugal": "f",
    "fix_esp": "b",
    "clone": "c",
    "rename": "e",
    "export": "x",
    "check": "k",
    "resize": "z",
    "info": "?",
    "help": "h",
    "protect": "P",
    "encrypt": "E",
    "unlock": "L",
    "hash": "#",
    "frugal_save": "S",
    "frugal_merge": "M",
    "frugal_roll": "R",
    "iso": "I",
    "wizard": "W",
    "quit": "q",
}

QEMU_ARCHES: dict[str, tuple[str, list[str]]] = {
    "x86_64": ("qemu-system-x86_64", []),
    "aarch64": ("qemu-system-aarch64", ["-machine", "virt", "-cpu", "cortex-a57"]),
    "arm": ("qemu-system-arm", ["-machine", "virt"]),
    "riscv64": ("qemu-system-riscv64", ["-machine", "virt"]),
    "i386": ("qemu-system-i386", []),
}


def load_keybindings() -> dict[str, str]:
    """Load keybindings from config file (~/.config/binsys/keybindings.json)."""
    cfg_paths: list[Path] = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        cfg_paths.append(Path(xdg) / "binsys" / "keybindings.json")
    cfg_paths.append(Path.home() / ".config" / "binsys" / "keybindings.json")
    for p in cfg_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                kb = DEFAULT_KEYBINDINGS.copy()
                for k, v in data.items():
                    if isinstance(v, str) and len(v) == 1:
                        kb[k] = v
                return kb
            except Exception:
                logger.exception("Failed to load keybindings from %s", p)
    return DEFAULT_KEYBINDINGS.copy()


# ── core helpers ──────────────────────────────────────────────────────────────


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# Required system dependencies for various operations
REQUIRED_BINARIES: dict[str, list[str]] = {
    "image_creation": ["truncate", "mkfs.ext4", "mkfs.fat", "mksquashfs"],
    "image_mount": ["mount", "umount"],
    "encryption": ["cryptsetup", "losetup"],
    "qemu": ["qemu-system-x86_64"],
    "gpt": ["sgdisk"],
    "iso": ["isoinfo"],
    "fsck": ["e2fsck", "fsck.fat"],
}

ISO_CREATORS = ["mkisofs", "genisoimage"]


def _any_present(binaries: list[str]) -> bool:
    return any(shutil.which(b) for b in binaries)


def check_dependencies(operation: str | None = None) -> dict[str, list[str]]:
    """Check for required system dependencies. Returns dict of missing deps."""
    missing: dict[str, list[str]] = {}

    for category, binaries in REQUIRED_BINARIES.items():
        if operation and operation != category:
            continue
        category_missing = [b for b in binaries if not shutil.which(b)]
        if category_missing:
            missing[category] = category_missing

    if (operation is None or operation == "iso") and not _any_present(ISO_CREATORS):
        missing.setdefault("iso", []).extend(ISO_CREATORS)

    return missing


def check_dependencies_or_warn(operation: str | None = None) -> None:
    """Check dependencies and warn about missing ones."""
    missing = check_dependencies(operation)
    if missing:
        logger.warning("Missing system dependencies:")
        for category, binaries in missing.items():
            logger.warning(f"  {category}: {', '.join(binaries)}")
        logger.warning("Some features may not work correctly.")


def sh(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    sudo: bool = False,
    quiet: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[Any]:
    if capture and quiet:
        raise ValueError("capture and quiet are mutually exclusive")
    if sudo and os.geteuid() != 0:
        cmd = ["sudo", *list(cmd)]
    logger.debug("sh: %s (sudo=%s, check=%s, capture=%s)", cmd, sudo, check, capture)
    kw: dict[str, Any] = {"check": check}
    if capture:
        kw |= {"capture_output": True, "text": True}
    elif quiet:
        kw |= {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if env is not None:
        kw["env"] = {**os.environ, **env}
    if cwd is not None:
        kw["cwd"] = cwd
    try:
        return subprocess.run(cmd, **kw)
    except subprocess.CalledProcessError as e:
        if check:
            msg = (e.stderr or "").strip() or str(e)
            raise RuntimeError(f"Command `{' '.join(str(a) for a in cmd)}` failed: {msg}")
        raise


def ensure_dirs() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    MOUNTS.mkdir(parents=True, exist_ok=True)


def sys_dir(name: str) -> Path:
    return IMAGES / name


def load_meta(name: str) -> dict[str, Any] | None:
    p = sys_dir(name) / "meta.json"
    if not p.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


def save_meta(name: str, meta: dict[str, Any]) -> None:
    p = sys_dir(name) / "meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(p)


def resolve_size(s: str | None) -> str:
    """Normalize a size string or preset to its canonical form."""
    if not s:
        return "1G"
    s = str(s).strip()
    return SIZE_PRESETS.get(s.lower(), s)


def _df_info(path: str) -> tuple[int, int] | None:
    """Return (used_bytes, total_bytes) via statvfs, or None on error."""
    try:
        st = os.statvfs(path)
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        return used, total
    except OSError:
        return None


def is_mounted(path: Path) -> bool:
    """Return True if the given path is an active mount point."""
    try:
        return os.path.ismount(str(path))
    except Exception:
        return False


def mounted_set() -> set[str]:
    """Return a set of mounted paths under the global mounts directory."""
    if not MOUNTS.exists():
        return set()
    return {str(p) for p in MOUNTS.iterdir() if p.exists() and is_mounted(p)}


def all_systems() -> list[dict[str, Any]]:
    """Return metadata for all known systems in the images store."""
    ensure_dirs()
    systems: list[dict[str, Any]] = []
    if not IMAGES.exists():
        return systems
    for p in sorted(IMAGES.iterdir()):
        if not p.is_dir():
            continue
        meta = load_meta(p.name)
        if meta:
            systems.append(meta)
    return systems


def _unique_snap_name(base: str) -> str:
    """Return base, base-2, base-3, … — whichever doesn't exist yet."""
    if not sys_dir(base).exists():
        return base
    n = 2
    while sys_dir(f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def _size_to_bytes(s: str) -> int:
    """Parse size like '1K','1KiB','2G' into bytes."""
    s = str(s).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([KkMmGgTt]i?B?|B)?$", s)
    if not m:
        raise ValueError(f"invalid size: {s}")
    n = float(m.group(1))
    suf = (m.group(2) or "B").upper()
    _mult = {
        "B": 1,
        "K": 1000, "KB": 1000,
        "KI": 1024, "KIB": 1024,
        "M": 1000 ** 2, "MB": 1000 ** 2,
        "MI": 1024 ** 2, "MIB": 1024 ** 2,
        "G": 1000 ** 3, "GB": 1000 ** 3,
        "GI": 1024 ** 3, "GIB": 1024 ** 3,
        "T": 1000 ** 4, "TB": 1000 ** 4,
        "TI": 1024 ** 4, "TIB": 1024 ** 4,
    }
    if suf in _mult:
        return int(n * _mult[suf])
    raise ValueError(f"invalid size suffix: {suf}")


def human(n: int) -> str:
    if n == 0:
        return "0B"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}PiB"


def sanitize_filename(name: str) -> str:
    """Remove or replace characters unsafe for filenames (simple)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    s = s.strip(". ")
    s = re.sub(r"_+", "_", s)
    return s or "unnamed"


def _validate_name(name: str) -> None:
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        raise RuntimeError(f"invalid name: {name}")


def _validate_size(size_str: str) -> int:
    """Validate and parse a size string, returning bytes.

    Raises RuntimeError if the size string is invalid.
    """
    try:
        bytes_val = _size_to_bytes(size_str)
        if bytes_val <= 0:
            raise ValueError("size must be positive")
        return bytes_val
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"invalid size '{size_str}': {e}") from e


def _validate_positive_int(value: str, name: str = "value") -> int:
    """Validate that a string can be parsed as a positive integer."""
    try:
        val = int(value)
        if val <= 0:
            raise ValueError(f"{name} must be positive")
        return val
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"invalid {name}: {e}") from e
