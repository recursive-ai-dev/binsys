"""Encryption, hashing, and app-level protection for disk images."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import shutil
import tempfile as _binsys_tf
from pathlib import Path
from typing import Any

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

# ── low-level helpers ─────────────────────────────────────────────────────────


def _crypt_keyfile(passphrase: str) -> tuple[str, Any]:
    """Write passphrase to a secure temp file and return path + close function."""
    old_umask = os.umask(0o077)
    try:
        fd, path = _binsys_tf.mkstemp(prefix="binsys-crypt-", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(passphrase)
                f.write("\n")
        except Exception:
            os.unlink(path, missing_ok=True)
            raise
    finally:
        os.umask(old_umask)

    def cleanup() -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    return path, cleanup


# ── disk encryption ───────────────────────────────────────────────────────────


def do_encrypt(name: str, hash_algo: str = "sha256", passphrase: str | None = None) -> None:
    """Encrypt an ext4/fat32 disk image in-place with LUKS2."""
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
        raise RuntimeError("cryptsetup not found — sudo apt install cryptsetup")

    import hmac
    
    if not passphrase:
        pp = getpass.getpass("Enter encryption passphrase: ")
        cf = getpass.getpass("Confirm passphrase: ")
        # Constant-time comparison
        if not hmac.compare_digest(pp, cf):
            raise RuntimeError("passphrases do not match")
        passphrase = pp

    if not shutil.which("losetup"):
        raise RuntimeError("losetup not found")

    r = sh(["losetup", "--find", "--show", str(img)], capture=True, sudo=True)
    loop_dev = r.stdout.strip()
    if not loop_dev:
        raise RuntimeError("failed to allocate loop device")
    keyfile, cleanup_key = _crypt_keyfile(passphrase)
    try:
        sh(["cryptsetup", "reencrypt", "--encrypt", "--type", "luks2",
            "--hash", hash_algo, "--key-file", keyfile, loop_dev], sudo=True)
        mapper = f"binsys-{name}-{os.urandom(4).hex()}"
        sh(["cryptsetup", "open", "--key-file", keyfile, loop_dev, mapper], sudo=True)
        meta["luks_mapper"] = mapper
        meta["encrypted"] = True
        save_meta(name, meta)
        logger.info("Encrypted %s (mapper=%s)", name, mapper)
    finally:
        cleanup_key()
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)


def do_unlock(name: str, passphrase: str | None = None) -> None:
    """Open a LUKS-encrypted image (set up dm-crypt mapper)."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if not meta.get("encrypted"):
        raise RuntimeError(f"'{name}' is not encrypted")
    if meta.get("luks_mapper") and Path(f"/dev/mapper/{meta['luks_mapper']}").exists():
        logger.info("'%s' already unlocked (%s)", name, meta["luks_mapper"])
        return

    if not passphrase:
        passphrase = getpass.getpass("Enter passphrase: ")

    d = sys_dir(name)
    img = d / meta["disk"]
    if not img.exists():
        raise RuntimeError(f"image not found: {img}")

    r = sh(["losetup", "--find", "--show", str(img)], capture=True, sudo=True)
    loop_dev = r.stdout.strip()
    keyfile, cleanup_key = _crypt_keyfile(passphrase)
    try:
        mapper = f"binsys-{name}-{os.urandom(4).hex()}"
        sh(["cryptsetup", "open", "--key-file", keyfile, loop_dev, mapper], sudo=True)
        meta["luks_mapper"] = mapper
        save_meta(name, meta)
        logger.info("Unlocked %s (mapper=%s)", name, mapper)
    finally:
        cleanup_key()
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)


def do_lock(name: str) -> None:
    """Close a LUKS-encrypted image."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    mapper = meta.get("luks_mapper")
    if not mapper:
        raise RuntimeError(f"'{name}' has no active mapper")
    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")
    sh(["cryptsetup", "close", mapper], sudo=True)
    meta.pop("luks_mapper", None)
    save_meta(name, meta)
    logger.info("Locked %s", name)


def do_hash(name: str, algo: str = "sha256") -> None:
    """Compute and print a checksum of a system's image."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    d = sys_dir(name)
    if meta["type"] in ("ext4", "fat32", "iso", "iso9660"):
        path = d / meta["disk"]
    elif meta["type"] in ("squashfs", "overlay"):
        path = d / meta["base"]
    else:
        raise RuntimeError(f"no hash strategy for type '{meta['type']}'")
    if not path.exists():
        raise RuntimeError(f"file not found: {path}")
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    size = path.stat().st_size
    print(f"{algo.upper()} ({path.name}) = {h.hexdigest()}")
    print(f"Size: {human(size)} ({size} bytes)")


# ── app-level protection ──────────────────────────────────────────────────────

# Rate limiting state (in-memory, cleared on restart)
_auth_failures: dict[str, tuple[int, float]] = {}
_RATE_LIMIT_MAX = 5
_RATE_LIMIT_WINDOW = 300  # 5 minutes


def _load_app_locks() -> dict[str, Any]:
    p = STORE / "app_locks.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_app_locks(locks: dict[str, Any]) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    (STORE / "app_locks.json").write_text(json.dumps(locks, indent=2))


def _is_app_unlocked(name: str) -> bool:
    locks = _load_app_locks()
    return locks.get(name, {}).get("unlocked", False)


def _check_rate_limit(name: str) -> None:
    """Check and enforce authentication rate limiting."""
    import time
    now = time.time()
    failures, last_time = _auth_failures.get(name, (0, 0.0))
    if now - last_time > _RATE_LIMIT_WINDOW:
        _auth_failures[name] = (0, now)
        return
    if failures >= _RATE_LIMIT_MAX:
        remaining = int(_RATE_LIMIT_WINDOW - (now - last_time))
        raise RuntimeError(
            f"too many failed attempts for '{name}' — try again in {remaining}s"
        )


def _record_failure(name: str) -> None:
    """Record a failed authentication attempt."""
    import time
    failures, last_time = _auth_failures.get(name, (0, 0.0))
    _auth_failures[name] = (failures + 1, last_time if failures > 0 else time.time())


def _clear_failures(name: str) -> None:
    """Clear rate limiting state after successful auth."""
    _auth_failures.pop(name, None)


def _app_lock_hash(password: str, keyfile: str | None = None) -> str:
    h = hashlib.sha256(password.encode())
    if keyfile:
        kp = Path(keyfile)
        if kp.exists():
            h.update(kp.read_bytes())
    return h.hexdigest()


def _ensure_app_unlocked(name: str) -> None:
    locks = _load_app_locks()
    entry = locks.get(name)
    if entry and not entry.get("unlocked", False):
        raise RuntimeError(f"'{name}' is locked — run 'binsys auth {name}' first")


def do_protect(name: str, password: str | None = None, keyfile: str | None = None) -> None:
    """Set app-level password+keyfile protection on a system."""
    import hmac
    
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if not password:
        password = getpass.getpass("App password: ")
        cf = getpass.getpass("Confirm: ")
        # Constant-time comparison
        if not hmac.compare_digest(password, cf):
            raise RuntimeError("passwords do not match")
    locks = _load_app_locks()
    locks[name] = {
        "hash": _app_lock_hash(password, keyfile),
        "keyfile": str(keyfile) if keyfile else None,
        "unlocked": False,
    }
    _save_app_locks(locks)


def do_unprotect(name: str) -> None:
    """Remove app-level protection from a system."""
    locks = _load_app_locks()
    locks.pop(name, None)
    _save_app_locks(locks)


def do_app_unlock(name: str, password: str | None = None) -> None:
    """Authenticate to unlock a protected system for this session."""
    import hmac
    import time
    
    _check_rate_limit(name)
    locks = _load_app_locks()
    entry = locks.get(name)
    if not entry:
        raise RuntimeError(f"'{name}' is not protected")
    if not password:
        password = getpass.getpass("App password: ")
    keyfile = entry.get("keyfile")
    actual_hash = _app_lock_hash(password, keyfile)
    expected_hash = entry["hash"]
    
    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(actual_hash, expected_hash):
        _record_failure(name)
        raise RuntimeError("incorrect password")
    
    _clear_failures(name)
    entry["unlocked"] = True
    entry["unlocked_at"] = time.time()
    _save_app_locks(locks)


def do_app_lock(name: str) -> None:
    """Re-lock a protected system in the current session."""
    locks = _load_app_locks()
    entry = locks.get(name)
    if not entry:
        raise RuntimeError(f"'{name}' is not protected")
    entry["unlocked"] = False
    _save_app_locks(locks)
