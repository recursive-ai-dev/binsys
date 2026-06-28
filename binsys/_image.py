"""Core image operations: create, delete, clone, rename, export, check, mount, import, resize, snapshot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from binsys._crypto import (
    _ensure_app_unlocked,
    do_encrypt,
)
from binsys._util import (
    MOUNTS,
    TYPES,
    _size_to_bytes,
    _unique_snap_name,
    _validate_name,
    _validate_size,
    ensure_dirs,
    human,
    is_mounted,
    load_meta,
    logger,
    resolve_size,
    save_meta,
    sh,
    sys_dir,
)

# ── Distro Metadata ───────────────────────────────────────────────────────────

DISTRO_IMAGES: Final[dict[str, dict[str, str]]] = {
    "ubuntu": {
        "url": "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
        "sha256": "53fdde898feed8b027d94baa9cfe8229867f330a1d9c49dc7d84465ee7f229f7",
    },
    "debian": {
        "url": "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2",
        "sha256": "6a05a330409e14759533317787d6921e08651628c1b68c8b37675555a1557f5",
    },
    "arch": {
        "url": "https://geo.mirror.pkgbuild.com/images/latest/Arch-Linux-x86_64-basic.qcow2",
        "sha256": "f0afc371014e559a3ff92cc0af1bb3d5e9e87da226f54446a9b9a93f29c1e124",
    },
    "fedora": {
        "url": "https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/x86_64/images/Fedora-Cloud-Base-40-1.14.x86_64.qcow2",
        "sha256": "5f7830e60e9a507a8c9787e5e5a7342f7161e8b0b62e356587394c0a8e8f578",
    },
    "alpine": {
        "url": "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/alpine-virt-3.20.3-x86_64.iso",
        "sha256": "81df854fbd7327d293c726b1eeeb82061d3bc8f5a86a6f77eea720f6be372261",
    },
    "void": {
        "url": "https://repo-default.voidlinux.org/live/current/void-x86_64-20240314.iso",
        "sha256": "0f7439f500740f62dd18972cae448cec7d8a85032c7eb8f1bf946100d9a92161",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _flash_source(distro: str, d: Path, size: str) -> None:
    """Download a distro image and write it to disk.img inside the system dir."""
    info = DISTRO_IMAGES.get(distro)
    if not info:
        raise RuntimeError(
            f"unknown distro '{distro}' — choose from {', '.join(DISTRO_IMAGES.keys())}"
        )

    url = info["url"]
    expected_hash = info["sha256"]

    img_path = d / "disk.img"
    tmp_path = d / "disk.img.tmp"

    logger.info("Downloading %s …", url)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if expected_hash:
            sha256 = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            if actual_hash != expected_hash:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"download hash mismatch for {distro}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )
        tmp_path.rename(img_path)
    except (urllib.error.URLError, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {e}") from e

    # Optionally resize to requested size (raw ext4 images only)
    if size:
        target = _size_to_bytes(size)
        actual = img_path.stat().st_size
        if target > actual:
            # Detect format — skip resize for qcow2/ISO
            with open(img_path, "rb") as f:
                magic = f.read(32774)

            # qcow2 magic: QFI\xfb
            if magic[:4] == b"QFI\xfb":
                logger.warning("qcow2 format detected — resize skipped (use qemu-img)")
                return

            # ISO magic: CD001 at offset 32768
            if len(magic) >= 32774 and magic[32769:32774] == b"CD001":
                logger.warning("ISO format detected — resize skipped")
                return

            # Simple ext4 check (Superblock magic at 0x438 within the first block of the group)
            # This is very simplified. For raw images, we just truncate and resize.

            logger.info("Resizing %s → %s", human(actual), human(target))
            with open(img_path, "ab") as f:
                f.truncate(target)
            # Try to resize the filesystem inside the image
            sh(["e2fsck", "-p", "-f", str(img_path)], sudo=True, check=False)
            sh(["resize2fs", str(img_path)], sudo=True, check=False)


# ── do_* operations ───────────────────────────────────────────────────────────


def do_new(
    name: str,
    img_type: str = "ext4",
    size: str = "1G",
    label: str | None = None,
    distro: str | None = None,
    encrypt: bool = False,
    boot: bool = False,
    bootloader: bool = False,
    auto_esp: bool = False,
    save_size: str | None = None,
) -> None:
    """Create a new filesystem image. Uses a temp directory for atomicity."""
    _validate_name(name)
    if img_type not in TYPES:
        raise RuntimeError(f"unknown type '{img_type}' — choose from {', '.join(TYPES)}")

    final_dir = sys_dir(name)
    if final_dir.exists():
        raise RuntimeError(f"'{name}' already exists")

    _validate_size(size)
    if save_size:
        _validate_size(save_size)

    size = resolve_size(size)
    ensure_dirs()

    # Create in a temporary directory
    tmp_dir_path = Path(tempfile.mkdtemp(dir=final_dir.parent, prefix=f".new-{name}-"))
    try:
        tmp_dir = Path(tmp_dir_path)
        meta: dict[str, Any] = {
            "name": name,
            "type": img_type,
            "created": datetime.now().isoformat(timespec="seconds"),
        }

        if img_type == "ext4":
            img = tmp_dir / "disk.img"
            sh(["truncate", "-s", size, str(img)])
            sh(["mkfs.ext4", "-F", str(img)])
            if label:
                sh(["e2label", str(img), label])
            meta.update({"disk": "disk.img", "fstype": "ext4"})

        elif img_type == "overlay" or img_type == "frugal":
            base_img = tmp_dir / "base.sfs"
            save_img = tmp_dir / "save.img"
            save_sz = resolve_size(save_size or "512M")
            # Create a temp ext4, fill it with a marker, squash it
            tmp_ext4 = tmp_dir / ".tmp_ext4"
            try:
                sh(["truncate", "-s", size, str(tmp_ext4)])
                sh(["mkfs.ext4", "-F", str(tmp_ext4)])
                tmp_mnt = tmp_dir / ".tmp_mnt"
                tmp_mnt.mkdir(exist_ok=True)
                sh(["mount", "-o", "loop", str(tmp_ext4), str(tmp_mnt)], sudo=True)
                try:
                    (tmp_mnt / "binsys.txt").write_text(f"binsys {img_type} — {name}\n")
                finally:
                    sh(["umount", str(tmp_mnt)], sudo=True)
                shutil.rmtree(tmp_mnt)
                sh(["mksquashfs", str(tmp_ext4), str(base_img), "-noappend"], sudo=True)
            finally:
                if tmp_ext4.exists():
                    tmp_ext4.unlink()
            sh(["truncate", "-s", save_sz, str(save_img)])
            sh(["mkfs.ext4", "-F", str(save_img)])
            meta.update({
                "base": "base.sfs",
                "save": "save.img",
                "fstype": "overlay",
                "frugal": (img_type == "frugal")
            })

        elif img_type == "squashfs":
            base_img = tmp_dir / "base.sfs"
            tmp_ext4 = tmp_dir / ".tmp_ext4"
            try:
                sh(["truncate", "-s", size, str(tmp_ext4)])
                sh(["mkfs.ext4", "-F", str(tmp_ext4)])
                tmp_mnt = tmp_dir / ".tmp_mnt"
                tmp_mnt.mkdir(exist_ok=True)
                sh(["mount", "-o", "loop", str(tmp_ext4), str(tmp_mnt)], sudo=True)
                try:
                    (tmp_mnt / "binsys.txt").write_text(f"binsys squashfs — {name}\n")
                finally:
                    sh(["umount", str(tmp_mnt)], sudo=True)
                shutil.rmtree(tmp_mnt)
                sh(["mksquashfs", str(tmp_ext4), str(base_img), "-comp", "zstd", "-noappend"], sudo=True)
            finally:
                if tmp_ext4.exists():
                    tmp_ext4.unlink()
            meta.update({"base": "base.sfs", "fstype": "squashfs"})

        elif img_type == "fat32":
            img = tmp_dir / "disk.img"
            sh(["truncate", "-s", size, str(img)])
            sh(["mkfs.fat", "-F32", str(img)])
            if label:
                sh(["fatlabel", str(img), label[:11]])
            meta.update({"disk": "disk.img", "fstype": "fat32"})

        elif img_type == "iso":
            img = tmp_dir / "disk.img"
            sh(["truncate", "-s", size, str(img)])
            sh(["mkfs.ext4", "-F", str(img)])
            meta.update({"disk": "disk.img", "fstype": "iso9660"})

        if distro:
            _flash_source(distro, tmp_dir, size)
            meta["source"] = distro

        # Write metadata into the temp directory
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Rename temp directory to final destination
        try:
            tmp_dir.rename(final_dir)
        except OSError as e:
            raise RuntimeError(f"failed to finalize system directory: {e}") from e
    except Exception:
        if tmp_dir_path.exists():
            shutil.rmtree(tmp_dir_path)
        raise

    # Handle encryption after move
    if encrypt:
        try:
            do_encrypt(name)
        except Exception as e:
            logger.error("Encryption failed after creation: %s", e)
            logger.info("The system '%s' remains unencrypted.", name)


def do_delete(name: str) -> None:
    """Delete a system directory and all its contents."""
    _ensure_app_unlocked(name)
    d = sys_dir(name)
    if not d.exists():
        raise RuntimeError(f"'{name}' not found")

    # Check for active mounts
    active_mounts = [
        m for m in [MOUNTS / name, MOUNTS / f"{name}_base", MOUNTS / f"{name}_save"]
        if is_mounted(m)
    ]
    if active_mounts:
        raise RuntimeError(
            f"'{name}' has active mounts: {', '.join(str(m) for m in active_mounts)}. Umount first."
        )

    shutil.rmtree(d)
    logger.info("Deleted system '%s'", name)


def do_clone(src_name: str, dst_name: str) -> None:
    """Deep-copy a system under a new name."""
    _ensure_app_unlocked(src_name)
    _validate_name(dst_name)
    src_dir = sys_dir(src_name)
    dst_dir = sys_dir(dst_name)

    if not src_dir.exists():
        raise RuntimeError(f"'{src_name}' not found")
    if is_mounted(MOUNTS / src_name):
        raise RuntimeError(f"'{src_name}' is mounted — umount first")
    if dst_dir.exists():
        raise RuntimeError(f"'{dst_name}' already exists")

    shutil.copytree(src_dir, dst_dir)
    meta = load_meta(dst_name)
    if meta:
        meta["name"] = dst_name
        meta["created"] = datetime.now().isoformat(timespec="seconds")
        meta.pop("source", None)
        save_meta(dst_name, meta)
    logger.info("Cloned '%s' to '%s'", src_name, dst_name)


def do_rename(old_name: str, new_name: str) -> None:
    """Rename a system: moves its directory and updates meta."""
    _ensure_app_unlocked(old_name)
    _validate_name(new_name)
    old_dir = sys_dir(old_name)
    new_dir = sys_dir(new_name)

    if not old_dir.exists():
        raise RuntimeError(f"'{old_name}' not found")
    if is_mounted(MOUNTS / old_name):
        raise RuntimeError(f"'{old_name}' is mounted — umount first")
    if new_dir.exists():
        raise RuntimeError(f"'{new_name}' already exists")

    old_dir.rename(new_dir)
    meta = load_meta(new_name)
    if meta:
        meta["name"] = new_name
        save_meta(new_name, meta)
    logger.info("Renamed '%s' to '%s'", old_name, new_name)


def do_export(name: str, dest_path: str | None = None) -> tuple[Path, int]:
    """Copy the primary image file to dest_path (directory or file path)."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")

    d = sys_dir(name)
    kind = meta["type"]
    if kind in ("ext4", "fat32", "iso", "iso9660"):
        src = d / meta["disk"]
        suffix = ".img"
    elif kind in ("squashfs", "overlay", "frugal"):
        src = d / meta["base"]
        suffix = ".sfs"
    else:
        raise RuntimeError(f"no export strategy for type '{kind}'")

    if not src.exists():
        raise RuntimeError(f"source image not found: {src}")

    dst = Path(dest_path) if dest_path else Path.cwd()
    if dst.is_dir():
        dst = dst / (name + suffix)

    shutil.copy2(src, dst)
    return dst, src.stat().st_size


def do_check(name: str) -> None:
    """Run a filesystem integrity check on the primary image."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")

    kind = meta["type"]
    d = sys_dir(name)
    if kind == "ext4":
        sh(["e2fsck", "-p", "-f", "-v", str(d / meta["disk"])], sudo=True, check=False)
    elif kind == "overlay" or kind == "frugal":
        sh(["e2fsck", "-p", "-f", "-v", str(d / meta["save"])], sudo=True, check=False)
    elif kind == "fat32":
        sh(["fsck.fat", "-v", "-a", str(d / meta["disk"])], sudo=True, check=False)
    elif kind in ("iso", "iso9660"):
        img = d / meta.get("disk", "disk.img")
        sh(["isoinfo", "-d", "-i", str(img)])
    elif kind == "squashfs":
        sh(["unsquashfs", "-s", str(d / meta["base"])])
    else:
        raise RuntimeError(f"no check method for type '{kind}'")


def _mount_image(kind: str, d: Path, meta: dict[str, Any], mnt: Path) -> None:
    """Low-level mount logic for different image types."""
    # For encrypted images, use the LUKS mapper device if unlocked
    mapper = meta.get("luks_mapper")
    dev_mapper = Path(f"/dev/mapper/{mapper}") if mapper else None
    dev = dev_mapper if dev_mapper and dev_mapper.exists() else None

    if kind == "ext4":
        src = dev if dev else (d / meta["disk"])
        opts: list[str] = [] if dev else ["-o", "loop"]
        sh(["mount", *opts, str(src), str(mnt)], sudo=True)

    elif kind == "fat32":
        src = dev if dev else (d / meta["disk"])
        opts = [] if dev else ["-o", f"loop,uid={os.getuid()},gid={os.getgid()}"]
        sh(["mount", *opts, str(src), str(mnt)], sudo=True)

    elif kind in ("iso", "iso9660"):
        img = d / meta.get("disk", "disk.img")
        if not img.exists():
            raise RuntimeError(f"ISO image not found: {img}")
        sh(["mount", "-o", "loop,ro", str(img), str(mnt)], sudo=True)

    elif kind == "squashfs":
        sh(["mount", "-o", "loop,ro", str(d / meta["base"]), str(mnt)], sudo=True)

    elif kind == "overlay" or kind == "frugal":
        name = meta["name"]
        base_mnt = MOUNTS / f"{name}_base"
        save_mnt = MOUNTS / f"{name}_save"
        base_mnt.mkdir(parents=True, exist_ok=True)
        save_mnt.mkdir(parents=True, exist_ok=True)

        try:
            sh(["mount", "-o", "loop,ro", str(d / meta["base"]), str(base_mnt)], sudo=True)
            try:
                sh(["mount", "-o", "loop", str(d / meta["save"]), str(save_mnt)], sudo=True)
            except Exception:
                sh(["umount", str(base_mnt)], sudo=True, check=False)
                raise

            upper = save_mnt / "upper"
            work = save_mnt / ".work"
            sh(["mkdir", "-p", str(upper), str(work)], sudo=True)

            sh(["mount", "-t", "overlay", "overlay",
                "-o", f"lowerdir={base_mnt},upperdir={upper},workdir={work}",
                str(mnt)], sudo=True)
        except Exception:
            # Cleanup on failure
            for p in (mnt, save_mnt, base_mnt):
                if is_mounted(p):
                    sh(["umount", str(p)], sudo=True, check=False)
            raise


def do_mount(name: str) -> str:
    """Mount a system's image to the global mounts directory."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    mnt = MOUNTS / name
    if is_mounted(mnt):
        raise RuntimeError(f"already mounted at {mnt}")
    mnt.mkdir(parents=True, exist_ok=True)
    try:
        _mount_image(meta["type"], sys_dir(name), meta, mnt)
    except Exception:
        with contextlib.suppress(OSError):
            mnt.rmdir()
        raise
    return str(mnt)


def do_umount(name: str) -> None:
    """Unmount a system's image and any subsidiary layers (LIFO)."""
    meta = load_meta(name)
    mnt = MOUNTS / name

    if not is_mounted(mnt):
        # Even if main mount is gone, subsidiary layers might be there.
        # But for regular images, we want to raise if not mounted.
        # If meta is missing, assume it's a regular image and raise.
        if not meta or meta["type"] not in ("overlay", "frugal"):
            raise RuntimeError(f"'{name}' is not mounted")

    # 1. Unmount the main overlay/image (LIFO)
    if is_mounted(mnt):
        sh(["umount", str(mnt)], sudo=True)
        with contextlib.suppress(OSError):
            mnt.rmdir()

    # 2. Unmount layers for overlay/frugal types
    for sub in (f"{name}_save", f"{name}_base"):
        p = MOUNTS / sub
        if is_mounted(p):
            sh(["umount", str(p)], sudo=True, check=False)
            with contextlib.suppress(OSError):
                p.rmdir()

    # 3. Handle LUKS cleanup if encrypted
    if meta and meta.get("encrypted"):
        mapper = meta.get("luks_mapper")
        if mapper and Path(f"/dev/mapper/{mapper}").exists():
            # do_lock handles cryptsetup close AND loop detach
            from binsys._crypto import do_lock
            do_lock(name)


def do_snap(name: str) -> None:
    """Take a snapshot of an overlay save layer."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if meta["type"] not in ("overlay", "frugal"):
        raise RuntimeError(f"snapshot only supported for overlay types, not '{meta['type']}'")

    d = sys_dir(name)
    save_img = d / meta["save"]
    snap_dir = d / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    snap_name = _unique_snap_name(f"{name}-save")
    snap_path = snap_dir / f"{snap_name}.img"
    sh(["cp", "-a", str(save_img), str(snap_path)], sudo=True)
    logger.info("Snapshot saved: %s", snap_path)


def do_import(src: str, name: str | None = None, img_type: str = "ext4") -> None:
    """Import a pre-existing disk image into binsys."""
    src_path = Path(src)
    if not src_path.exists():
        raise RuntimeError(f"source not found: {src}")

    base_name = name or src_path.stem
    _validate_name(base_name)

    final_dir = sys_dir(base_name)
    if final_dir.exists():
        raise RuntimeError(f"'{base_name}' already exists")

    ensure_dirs()
    tmp_dir_path = Path(tempfile.mkdtemp(dir=final_dir.parent, prefix=f".import-{base_name}-"))
    try:
        tmp_dir = Path(tmp_dir_path)
        disk_name = "disk.img"
        dst = tmp_dir / disk_name
        shutil.copy2(src_path, dst)

        meta: dict[str, Any] = {
            "name": base_name,
            "type": img_type,
            "disk": disk_name,
            "fstype": "ext4" if img_type == "ext4" else img_type,
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": str(src_path.resolve()),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        tmp_dir.rename(final_dir)
    except Exception:
        if tmp_dir_path.exists():
            shutil.rmtree(tmp_dir_path)
        raise

    logger.info("Imported '%s' from %s", base_name, src)


def do_resize(name: str, new_size: str) -> None:
    """Resize the primary filesystem image."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")

    _validate_size(new_size)
    d = sys_dir(name)
    target_bytes = _size_to_bytes(new_size)

    if meta["type"] in ("ext4", "fat32"):
        img = d / meta["disk"]
        old_size = img.stat().st_size
        if target_bytes == old_size:
            logger.info("Already %s", human(target_bytes))
            return

        if target_bytes > old_size:
            sh(["truncate", "-s", str(target_bytes), str(img)])
            if meta["type"] == "ext4":
                sh(["e2fsck", "-p", "-f", str(img)], sudo=True, check=False)
                sh(["resize2fs", str(img)], sudo=True)
        elif meta["type"] == "ext4":
            sh(["e2fsck", "-p", "-f", str(img)], sudo=True, check=False)
            sh(["resize2fs", str(img), new_size], sudo=True)
            sh(["truncate", "-s", str(target_bytes), str(img)])
        else:
            raise RuntimeError("shrinking fat32 not supported")

    elif meta["type"] in ("overlay", "frugal"):
        save_img = d / meta["save"]
        old_size = save_img.stat().st_size
        if target_bytes == old_size:
            return
        if is_mounted(MOUNTS / f"{name}_save"):
            raise RuntimeError("save layer is mounted — umount first")

        sh(["truncate", "-s", str(target_bytes), str(save_img)])
        sh(["e2fsck", "-p", "-f", str(save_img)], sudo=True, check=False)
        sh(["resize2fs", str(save_img)], sudo=True)
    else:
        raise RuntimeError(f"resize not supported for type '{meta['type']}'")

    logger.info("Resized '%s' to %s", name, human(target_bytes))
