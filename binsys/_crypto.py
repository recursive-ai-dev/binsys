"""Encryption, hashing, and app-level protection for disk images."""

from __future__ import annotations

import base64
import contextlib
import getpass
import hashlib
import hmac
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

from binsys._util import (
    MOUNTS,
    STORE,
    human,
    is_mounted,
    load_meta,
    logger,
    save_meta,
    sh,
    sys_dir,
)

# ── PBKDF2 Configuration ──────────────────────────────────────────────────────

PBKDF2_ITERATIONS = 600_000  # OWASP recommended for HMAC-SHA256
SALT_SIZE = 16


# ── disk encryption (LUKS2) ───────────────────────────────────────────────────


def do_encrypt(name: str, hash_algo: str = "sha256", passphrase: str | None = None) -> None:
    """Encrypt an ext4/fat32 disk image in-place with LUKS2.
    Passphrase is piped to cryptsetup via stdin to avoid temp files.
    """
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if meta.get("encrypted"):
        raise RuntimeError(f"'{name}' is already encrypted")
    if meta["type"] not in ("ext4", "fat32"):
        raise RuntimeError(f"encryption not supported for type '{meta['type']}'")
    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")

    d = sys_dir(name)
    img = d / meta["disk"]
    if not img.exists():
        raise RuntimeError(f"image not found: {img}")

    if not shutil.which("cryptsetup"):
        raise RuntimeError("cryptsetup not found — install cryptsetup-bin or similar")

    if not passphrase:
        pp = getpass.getpass("Enter encryption passphrase: ")
        cf = getpass.getpass("Confirm passphrase: ")
        if not hmac.compare_digest(pp, cf):
            raise RuntimeError("passphrases do not match")
        passphrase = pp

    if not shutil.which("losetup"):
        raise RuntimeError("losetup not found")

    # Allocate loop device
    r = sh(["losetup", "--find", "--show", str(img)], capture=True, sudo=True)
    loop_dev = r.stdout.strip()
    if not loop_dev:
        raise RuntimeError("failed to allocate loop device")

    try:
        # Re-encrypt in-place. We pipe the passphrase to stdin.
        # cryptsetup reencrypt --encrypt requires --key-file - to read from stdin
        sh(
            [
                "cryptsetup", "reencrypt", "--encrypt", "--type", "luks2",
                "--hash", hash_algo, "--key-file", "-", loop_dev
            ],
            sudo=True,
            input_data=passphrase + "\n"
        )

        # Open the new LUKS device
        mapper = f"binsys-{name}-{os.urandom(4).hex()}"
        sh(
            ["cryptsetup", "open", "--key-file", "-", loop_dev, mapper],
            sudo=True,
            input_data=passphrase + "\n"
        )

        meta["luks_mapper"] = mapper
        meta["encrypted"] = True
        save_meta(name, meta)
        logger.info("Encrypted '%s' (mapper=%s)", name, mapper)
    finally:
        # Always detach loop device
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)


def do_unlock(name: str, passphrase: str | None = None) -> None:
    """Open a LUKS-encrypted image (set up dm-crypt mapper)."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if not meta.get("encrypted"):
        raise RuntimeError(f"'{name}' is not encrypted")

    mapper = meta.get("luks_mapper")
    if mapper and Path(f"/dev/mapper/{mapper}").exists():
        logger.info("'%s' already unlocked (%s)", name, mapper)
        return

    if not passphrase:
        passphrase = getpass.getpass(f"Enter passphrase for '{name}': ")

    d = sys_dir(name)
    img = d / meta["disk"]
    if not img.exists():
        raise RuntimeError(f"image not found: {img}")

    r = sh(["losetup", "--find", "--show", str(img)], capture=True, sudo=True)
    loop_dev = r.stdout.strip()
    if not loop_dev:
        raise RuntimeError("failed to allocate loop device")

    new_mapper = f"binsys-{name}-{os.urandom(4).hex()}"
    try:
        sh(
            ["cryptsetup", "open", "--key-file", "-", loop_dev, new_mapper],
            sudo=True,
            input_data=passphrase + "\n"
        )
        meta["luks_mapper"] = new_mapper
        save_meta(name, meta)
        logger.info("Unlocked '%s' (mapper=%s)", name, new_mapper)
    except Exception:
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)
        raise
    finally:
        # cryptsetup open keeps the device busy, but we can detach the loop
        # device if it was opened with --autoclear, or let cryptsetup handle it.
        # Actually, for LUKS, the loop device must stay.
        pass


def do_lock(name: str) -> None:
    """Close a LUKS-encrypted image."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")

    mapper = meta.get("luks_mapper")
    if not mapper:
        # Check if maybe it's still in /dev/mapper under some name?
        # For now, just assume it's locked if no mapper in meta.
        logger.warning("'%s' has no active mapper in metadata", name)
        return

    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")

    # Try to find the loop device to detach it after closing
    loop_dev = None
    try:
        r = sh(["cryptsetup", "status", mapper], capture=True, sudo=True, check=False)
        for line in r.stdout.splitlines():
            if "device:" in line:
                loop_dev = line.split(":")[1].strip()
                break
    except Exception:
        pass

    sh(["cryptsetup", "close", mapper], sudo=True)

    if loop_dev and loop_dev.startswith("/dev/loop"):
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)

    meta.pop("luks_mapper", None)
    save_meta(name, meta)
    logger.info("Locked '%s'", name)


def do_hash(name: str, algo: str = "sha256") -> None:
    """Compute and print a checksum of a system's image."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")

    d = sys_dir(name)
    if meta["type"] in ("ext4", "fat32", "iso", "iso9660"):
        path = d / meta["disk"]
    elif meta["type"] in ("squashfs", "overlay", "frugal"):
        path = d / meta["base"]
    else:
        raise RuntimeError(f"no hash strategy for type '{meta['type']}'")

    if not path.exists():
        raise RuntimeError(f"file not found: {path}")

    logger.info("Computing %s hash for %s...", algo.upper(), path.name)
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    size = path.stat().st_size
    print(f"{algo.upper()} ({path.name}) = {h.hexdigest()}")
    print(f"Size: {human(size)} ({size} bytes)")


# ── app-level protection (PBKDF2-HMAC-SHA256) ─────────────────────────────────

# Rate limiting state (in-memory)
_auth_failures: dict[str, tuple[int, float]] = {}
_RATE_LIMIT_MAX = 5
_RATE_LIMIT_WINDOW = 300  # 5 minutes


def _load_app_locks() -> dict[str, Any]:
    """Load application-level locks from the central store."""
    p = STORE / "app_locks.json"
    if p.exists():
        try:
            return cast(dict[str, Any], json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            logger.error("Failed to parse app_locks.json, returning empty")
            return {}
    return {}


def _save_app_locks(locks: dict[str, Any]) -> None:
    """Save application-level locks to the central store."""
    STORE.mkdir(parents=True, exist_ok=True)
    p = STORE / "app_locks.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(locks, indent=2))
    tmp.replace(p)


def _check_rate_limit(name: str) -> None:
    """Check and enforce authentication rate limiting for a system."""
    now = time.time()
    failures, last_time = _auth_failures.get(name, (0, 0.0))

    # Reset window if it has passed
    if now - last_time > _RATE_LIMIT_WINDOW:
        _auth_failures[name] = (0, now)
        return

    if failures >= _RATE_LIMIT_MAX:
        remaining = int(_RATE_LIMIT_WINDOW - (now - last_time))
        raise RuntimeError(
            f"Too many failed attempts for '{name}'. Please try again in {remaining}s."
        )


def _record_failure(name: str) -> None:
    """Record a failed authentication attempt."""
    failures, last_time = _auth_failures.get(name, (0, 0.0))
    # If first failure in a while, update last_time to now
    if now := time.time():
        if failures == 0 or (now - last_time > _RATE_LIMIT_WINDOW):
            _auth_failures[name] = (1, now)
        else:
            _auth_failures[name] = (failures + 1, last_time)


def _clear_failures(name: str) -> None:
    """Clear rate limiting state after successful authentication."""
    _auth_failures.pop(name, None)


def _app_lock_hash(password: str, salt: bytes, keyfile: str | None = None) -> str:
    """Derive a secure hash from password, salt, and optional keyfile content."""
    # Combine password with keyfile content if provided
    key_material = password.encode()
    if keyfile:
        kp = Path(keyfile)
        if kp.exists():
            key_material += kp.read_bytes()

    # Use PBKDF2-HMAC-SHA256
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        key_material,
        salt,
        PBKDF2_ITERATIONS
    )
    return derived.hex()


def _ensure_app_unlocked(name: str) -> None:
    """Raise RuntimeError if the system is protected and currently locked."""
    locks = _load_app_locks()
    entry = locks.get(name)
    if entry and not entry.get("unlocked", False):
        raise RuntimeError(f"'{name}' is app-locked. Run 'binsys auth {name}' first.")


def do_protect(name: str, password: str | None = None, keyfile: str | None = None) -> None:
    """Set app-level PBKDF2 protection on a system."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")

    if not password:
        password = getpass.getpass(f"Set app password for '{name}': ")
        cf = getpass.getpass("Confirm password: ")
        if not hmac.compare_digest(password, cf):
            raise RuntimeError("passwords do not match")

    salt = os.urandom(SALT_SIZE)
    locks = _load_app_locks()
    locks[name] = {
        "hash": _app_lock_hash(password, salt, keyfile),
        "salt": salt.hex(),
        "keyfile": str(Path(keyfile).resolve()) if keyfile else None,
        "unlocked": False,
        "algo": "pbkdf2-sha256",
        "iterations": PBKDF2_ITERATIONS,
    }
    _save_app_locks(locks)
    logger.info("Protected '%s' with app-level lock", name)


def do_unprotect(name: str) -> None:
    """Remove app-level protection from a system."""
    locks = _load_app_locks()
    if name in locks:
        locks.pop(name)
        _save_app_locks(locks)
        logger.info("Removed app-level protection from '%s'", name)
    else:
        logger.info("'%s' is not protected", name)


def do_app_unlock(name: str, password: str | None = None) -> None:
    """Authenticate to unlock a protected system for the current session."""
    _check_rate_limit(name)
    locks = _load_app_locks()
    entry = locks.get(name)
    if not entry:
        raise RuntimeError(f"'{name}' is not protected")

    if not password:
        password = getpass.getpass(f"App password for '{name}': ")

    salt = bytes.fromhex(entry["salt"])
    keyfile = entry.get("keyfile")

    actual_hash = _app_lock_hash(password, salt, keyfile)
    expected_hash = entry["hash"]

    if not hmac.compare_digest(actual_hash, expected_hash):
        _record_failure(name)
        raise RuntimeError("incorrect password")

    _clear_failures(name)
    entry["unlocked"] = True
    entry["unlocked_at"] = time.time()
    _save_app_locks(locks)
    logger.info("Unlocked '%s' for this session", name)


def do_app_lock(name: str) -> None:
    """Re-lock a protected system in the current session."""
    locks = _load_app_locks()
    entry = locks.get(name)
    if not entry:
        raise RuntimeError(f"'{name}' is not protected")
    entry["unlocked"] = False
    _save_app_locks(locks)
    logger.info("Locked '%s'", name)
