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

# Setup logger for the binsys package
logger = logging.getLogger("binsys")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ── Paths & Constants ─────────────────────────────────────────────────────────

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


# ── Core Helpers ──────────────────────────────────────────────────────────────


def die(msg: str) -> NoReturn:
    """Print an error message to stderr and exit with status 1."""
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

ISO_CREATORS = ["mkisofs", "genisoimage", "xorrisofs"]


def _any_present(binaries: list[str]) -> bool:
    """Return True if any of the listed binaries are present in PATH."""
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
    cwd: str | Path | None = None,
    input_data: str | bytes | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Execute a shell command with options for sudo, capture, and error checking."""
    if capture and quiet:
        raise ValueError("capture and quiet are mutually exclusive")

    if sudo and os.geteuid() != 0:
        cmd = ["sudo", *list(cmd)]

    logger.debug("sh: %s (sudo=%s, check=%s, capture=%s)", cmd, sudo, check, capture)

    kw: dict[str, Any] = {"check": check}
    if capture:
        kw["capture_output"] = True
        kw["text"] = not isinstance(input_data, bytes)
    elif quiet:
        kw["stdout"] = subprocess.DEVNULL
        kw["stderr"] = subprocess.DEVNULL

    if input_data is not None and not isinstance(input_data, bytes):
        kw["text"] = True

    if env is not None:
        kw["env"] = {**os.environ, **env}
    if cwd is not None:
        kw["cwd"] = str(cwd)
    if input_data is not None:
        kw["input"] = input_data

    try:
        return subprocess.run(cmd, **kw)
    except subprocess.CalledProcessError as e:
        if check:
            # Safely extract error message from stderr
            err_msg = ""
            if hasattr(e, "stderr") and e.stderr:
                err_msg = e.stderr.strip() if isinstance(e.stderr, str) else e.stderr.decode(errors="replace").strip()
            msg = err_msg or str(e)
            raise RuntimeError(f"Command `{' '.join(str(a) for a in cmd)}` failed: {msg}") from e
        raise


def ensure_dirs() -> None:
    """Ensure that the local store directories exist."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    MOUNTS.mkdir(parents=True, exist_ok=True)


def sys_dir(name: str) -> Path:
    """Return the data directory for a given system name."""
    return IMAGES / name


def load_meta(name: str) -> dict[str, Any] | None:
    """Load and return system metadata from its meta.json."""
    p = sys_dir(name) / "meta.json"
    if not p.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


def save_meta(name: str, meta: dict[str, Any]) -> None:
    """Atomically save system metadata to its meta.json."""
    d = sys_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "meta.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(p)


def resolve_size(s: str | None) -> str:
    """Normalize a size string or preset (e.g., 'large') to its canonical form ('4G')."""
    if not s:
        return "1G"
    s = str(s).strip()
    return SIZE_PRESETS.get(s.lower(), s)


def _df_info(path: str | Path) -> tuple[int, int] | None:
    """Return (used_bytes, total_bytes) via statvfs, or None on error."""
    try:
        st = os.statvfs(str(path))
        total = st.f_blocks * st.f_frsize
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        return used, total
    except (OSError, ValueError):
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
    """Return metadata for all known systems in the images store, sorted by name."""
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
    """Return base, base-2, base-3, … — whichever doesn't exist yet as a snapshot."""
    if not sys_dir(base).exists():
        return base
    n = 2
    while sys_dir(f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def _size_to_bytes(s: str) -> int:
    """Parse human-readable size like '1K', '1KiB', '2G' into integer bytes.
    Unitless numeric strings are treated as bytes.
    """
    s = str(s).strip()
    if not s:
        raise ValueError("empty size string")

    # Match number and optional unit
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KkMmGgTt]i?B?|B)?$", s)
    if not m:
        raise ValueError(f"invalid size format: {s}")

    n = float(m.group(1))
    unit = (m.group(2) or "B").upper()

    # Normalize units
    if unit.endswith("B") and len(unit) > 1:
        unit = unit[:-1]

    _mult = {
        "B": 1,
        "K": 1000,
        "KI": 1024,
        "M": 1000**2,
        "MI": 1024**2,
        "G": 1000**3,
        "GI": 1024**3,
        "T": 1000**4,
        "TI": 1024**4,
    }

    if unit in _mult:
        return int(n * _mult[unit])

    raise ValueError(f"invalid size unit: {unit}")


def human(n: int) -> str:
    """Return a human-readable string for a byte count using binary prefixes (KiB, MiB, ...)."""
    if n < 0:
        return str(n)
    if n == 0:
        return "0B"

    # Use binary prefixes (base 1024)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if n < 1024.0:
            if unit == "B":
                return f"{int(n)}B"
            val = f"{n:.2f}".rstrip("0").rstrip(".")
            return f"{val}{unit}"
        n /= 1024.0

    return f"{n:.2f}EiB"


def sanitize_filename(name: str) -> str:
    """Remove or replace characters unsafe for filenames (simple)."""
    # Replace unsafe chars with underscore
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    s = s.strip(". ")
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s or "unnamed"


def _validate_name(name: str) -> None:
    """Ensure system name uses only safe characters."""
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        raise RuntimeError(
            f"invalid name: '{name}' (use only alphanumeric, dots, underscores, dashes)"
        )


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
