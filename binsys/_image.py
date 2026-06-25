"""Core image operations: create, delete, clone, rename, export, check, mount, import, resize, snapshot."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

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

# try:
#     import argcomplete
#     _HAS_ARGCOMPLETE = True
# except ImportError:
#     _HAS_ARGCOMPLETE = False


# ── helpers ───────────────────────────────────────────────────────────────────


def _flash_source(distro: str, d: Path, size: str) -> None:
    """Download a distro image and write it to disk.img inside the system dir."""
    # Quick sanity: only allow known distros
    known = ("ubuntu", "debian", "arch", "fedora", "alpine", "void", "tinycore")
    if distro not in known:
        raise RuntimeError(f"unknown distro '{distro}' — choose from {', '.join(known)}")

    url_map: dict[str, str] = {
        "ubuntu":  "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
        "debian":  "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-generic-amd64.qcow2",
        "arch":    "https://geo.mirror.pkgbuild.com/images/latest/Arch-Linux-x86_64-basic.qcow2",
        "fedora":  "https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/x86_64/images/Fedora-Cloud-Base-40-1.14.x86_64.qcow2",
        "alpine":  "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/alpine-virt-3.20.3-x86_64.iso",
        "void":    "https://repo-default.voidlinux.org/live/current/void-x86_64-20240314.iso",
    }
    # SHA256 hashes for integrity verification (update when URLs change)
    hash_map: dict[str, str] = {
        "ubuntu":  "53fdde898feed8b027d94baa9cfe8229867f330a1d9c49dc7d84465ee7f229f7",  # noble-server-cloudimg-amd64.img
        "debian":  "6a05a330409e14759533317787d6921e08651628c1b68c8b37675555a1557f5",  # debian-12-generic-amd64.qcow2
        "arch":    "f0afc371014e559a3ff92cc0af1bb3d5e9e87da226f54446a9b9a93f29c1e124",  # Arch-Linux-x86_64-basic.qcow2
        "fedora":  "5f7830e60e9a507a8c9787e5e5a7342f7161e8b0b62e356587394c0a8e8f578",  # Fedora-Cloud-Base-40-1.14.x86_64.qcow2
        "alpine":  "81df854fbd7327d293c726b1eeeb82061d3bc8f5a86a6f77eea720f6be372261",  # alpine-virt-3.20.3-x86_64.iso
        "void":    "0f7439f500740f62dd18972cae448cec7d8a85032c7eb8f1bf946100d9a92161",  # void-x86_64-20250202-base.iso
    }
    url = url_map.get(distro)
    expected_hash = hash_map.get(distro)
    if not url:
        raise RuntimeError(f"no known URL for '{distro}'")

    img_path = d / "disk.img"
    tmp_path = d / "disk.img.tmp"
    logger.info("Downloading %s …", url)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if expected_hash and expected_hash != "0" * 64:
            sha256 = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            if actual_hash != expected_hash:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"download hash mismatch for {distro}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )
        tmp_path.rename(img_path)
    except urllib.error.URLError as e:
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
            if magic[:4] == b'QFI\xfb':
                logger.warning("qcow2 format detected — resize skipped (use qemu-img)")
                return
            if len(magic) < 32774 or magic[32768:32774] != b'\x53\xef\x01':
                logger.warning("non-ext4 format detected — resize skipped")
                return
            logger.info("Resizing %s → %s", human(actual), human(target))
            with open(img_path, "ab") as f:
                f.truncate(target)
            sh(["e2fsck", "-f", str(img_path)], sudo=True, check=False)
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
    """Create a new filesystem image."""
    _validate_name(name)
    if img_type not in TYPES:
        raise RuntimeError(f"unknown type '{img_type}' — choose from {', '.join(TYPES)}")
    if sys_dir(name).exists():
        raise RuntimeError(f"'{name}' already exists")

    # Validate sizes
    _validate_size(size)
    if save_size:
        _validate_size(save_size)

    size = resolve_size(size)
    ensure_dirs()
    d = sys_dir(name)
    d.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "name": name,
        "type": img_type,
        "created": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        if img_type == "ext4":
            img = d / "disk.img"
            sh(["truncate", "-s", size, str(img)])
            sh(["mkfs.ext4", "-F", str(img)])
            if label:
                sh(["e2label", str(img), label])
            meta.update({"disk": "disk.img", "fstype": "ext4"})

        elif img_type == "overlay":
            base_img = d / "base.sfs"
            save_img = d / "save.img"
            save_sz = resolve_size(save_size or "512M")
            # Create a temp ext4, fill it with a marker, squash it
            tmp = d / ".tmp_ext4"
            try:
                sh(["truncate", "-s", size, str(tmp)])
                sh(["mkfs.ext4", "-F", str(tmp)])
                tmp_mnt = d / ".tmp_mnt"
                tmp_mnt.mkdir(exist_ok=True)
                sh(["mount", "-o", "loop", str(tmp), str(tmp_mnt)], sudo=True)
                (tmp_mnt / "binsys.txt").write_text(f"binsys overlay — {name}\n")
                sh(["umount", str(tmp_mnt)], sudo=True)
                shutil.rmtree(tmp_mnt, ignore_errors=True)
                sh(["mksquashfs", str(tmp), str(base_img), "-noappend"], sudo=True)
            finally:
                if tmp.exists():
                    tmp.unlink()
            sh(["truncate", "-s", save_sz, str(save_img)])
            sh(["mkfs.ext4", "-F", str(save_img)])
            meta.update({"base": "base.sfs", "save": "save.img", "fstype": "overlay"})

        elif img_type == "squashfs":
            base_img = d / "base.sfs"
            tmp = d / ".tmp_ext4"
            try:
                sh(["truncate", "-s", size, str(tmp)])
                sh(["mkfs.ext4", "-F", str(tmp)])
                tmp_mnt = d / ".tmp_mnt"
                tmp_mnt.mkdir(exist_ok=True)
                sh(["mount", "-o", "loop", str(tmp), str(tmp_mnt)], sudo=True)
                (tmp_mnt / "binsys.txt").write_text(f"binsys squashfs — {name}\n")
                sh(["umount", str(tmp_mnt)], sudo=True)
                shutil.rmtree(tmp_mnt, ignore_errors=True)
                sh(["mksquashfs", str(tmp), str(base_img), "-comp", "zstd", "-noappend"], sudo=True)
            finally:
                if tmp.exists():
                    tmp.unlink()
            meta.update({"base": "base.sfs", "fstype": "squashfs"})

        elif img_type == "fat32":
            img = d / "disk.img"
            sh(["truncate", "-s", size, str(img)])
            sh(["mkfs.fat", "-F32", str(img)])
            if label:
                sh(["fatlabel", str(img), label])
            meta.update({"disk": "disk.img", "fstype": "fat32"})

        elif img_type == "frugal":
            # frugal is a synonym for overlay in our model
            base_img = d / "base.sfs"
            save_img = d / "save.img"
            save_sz = resolve_size(save_size or "512M")
            tmp = d / ".tmp_ext4"
            try:
                sh(["truncate", "-s", size, str(tmp)])
                sh(["mkfs.ext4", "-F", str(tmp)])
                tmp_mnt = d / ".tmp_mnt"
                tmp_mnt.mkdir(exist_ok=True)
                sh(["mount", "-o", "loop", str(tmp), str(tmp_mnt)], sudo=True)
                (tmp_mnt / "binsys.txt").write_text(f"binsys frugal — {name}\n")
                sh(["umount", str(tmp_mnt)], sudo=True)
                shutil.rmtree(tmp_mnt, ignore_errors=True)
                sh(["mksquashfs", str(tmp), str(base_img), "-noappend"], sudo=True)
            finally:
                if tmp.exists():
                    tmp.unlink()
            sh(["truncate", "-s", save_sz, str(save_img)])
            sh(["mkfs.ext4", "-F", str(save_img)])
            meta.update({
                "base": "base.sfs",
                "save": "save.img",
                "fstype": "overlay",
                "frugal": True,
            })

        elif img_type == "iso":
            img = d / "disk.img"
            sh(["truncate", "-s", size, str(img)])
            sh(["mkfs.ext4", "-F", str(img)])
            meta.update({"disk": "disk.img", "fstype": "iso9660"})

        else:
            raise RuntimeError(f"unhandled type '{img_type}'")

        if distro:
            _flash_source(distro, d, size)
            meta["source"] = distro

        if encrypt:
            # Defer meta save — encryption will update it
            save_meta(name, meta)
            try:
                do_encrypt(name)
            except Exception as e:
                logger.warning("Encryption failed after creation: %s", e)
        else:
            save_meta(name, meta)

    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def do_delete(name: str) -> None:
    _ensure_app_unlocked(name)
    d = sys_dir(name)
    if not d.exists():
        raise RuntimeError(f"'{name}' not found")
    if is_mounted(MOUNTS / name) or is_mounted(MOUNTS / f"{name}_base") or is_mounted(MOUNTS / f"{name}_save"):
        raise RuntimeError(f"'{name}' or its sub-mounts are mounted — umount first")
    shutil.rmtree(d)


def do_clone(src_name: str, dst_name: str) -> None:
    """Deep-copy a system under a new name."""
    _ensure_app_unlocked(src_name)
    _validate_name(dst_name)
    if not sys_dir(src_name).exists():
        raise RuntimeError(f"'{src_name}' not found")
    if is_mounted(MOUNTS / src_name):
        raise RuntimeError(f"'{src_name}' is mounted — umount first")
    if sys_dir(dst_name).exists():
        raise RuntimeError(f"'{dst_name}' already exists")
    shutil.copytree(sys_dir(src_name), sys_dir(dst_name))
    meta = load_meta(dst_name)
    if meta:
        meta["name"] = dst_name
        meta["created"] = datetime.now().isoformat(timespec="seconds")
        meta.pop("source", None)
        save_meta(dst_name, meta)


def do_rename(old_name: str, new_name: str) -> None:
    """Rename a system: moves its directory and updates meta."""
    _ensure_app_unlocked(old_name)
    _validate_name(new_name)
    if not sys_dir(old_name).exists():
        raise RuntimeError(f"'{old_name}' not found")
    if is_mounted(MOUNTS / old_name):
        raise RuntimeError(f"'{old_name}' is mounted — umount first")
    if sys_dir(new_name).exists():
        raise RuntimeError(f"'{new_name}' already exists")
    sys_dir(old_name).rename(sys_dir(new_name))
    meta = load_meta(new_name)
    if meta:
        meta["name"] = new_name
        save_meta(new_name, meta)


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
    elif kind in ("squashfs", "overlay"):
        src = d / meta["base"]
        suffix = ".sfs"
    else:
        raise RuntimeError(f"no export strategy for type '{kind}'")
    dst = Path(dest_path) if dest_path else Path.cwd()
    if dst.is_dir():
        dst = dst / (name + suffix)
    shutil.copy2(src, dst)
    return dst, src.stat().st_size


def do_check(name: str) -> None:
    """Run a filesystem integrity check on the writable image."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")
    kind = meta["type"]
    d = sys_dir(name)
    if kind == "ext4":
        sh(["e2fsck", "-f", "-v", str(d / meta["disk"])], sudo=True, check=False)
    elif kind == "overlay":
        sh(["e2fsck", "-f", "-v", str(d / meta["save"])], sudo=True, check=False)
    elif kind == "fat32":
        sh(["fsck.fat", "-v", str(d / meta["disk"])], sudo=True, check=False)
    elif kind in ("iso", "iso9660"):
        img = d / meta.get("disk", "disk.img")
        r = sh(["isoinfo", "-d", "-i", str(img)], capture=True, check=False)
        if r.returncode != 0:
            print(f"Error: ISO read failed: {(r.stderr or r.stdout).strip()}")
        else:
            for line in (r.stdout or "").splitlines():
                if any(k in line for k in ("Volume id", "Volume size", "Block size")):
                    print(f"  {line.strip()}")
            r2 = sh(["isoinfo", "-l", "-i", str(img)], capture=True, check=False)
            if r2.returncode == 0:
                entries = len([x for x in (r2.stdout or "").splitlines() if x.strip()])
                print(f"ISO OK — {entries} directory entries")
            else:
                print("Error: ISO directory listing failed")
    elif kind == "squashfs":
        r = sh(["unsquashfs", "-s", str(d / meta["base"])], capture=True, check=False)
        if r.returncode != 0:
            print(f"Error: squashfs superblock read failed: {(r.stderr or r.stdout).strip()}")
            return
        r2 = sh(["unsquashfs", "-l", str(d / meta["base"])], capture=True, check=False)
        if r2.returncode != 0:
            print(f"Error: squashfs data verification failed: {(r2.stderr or r.stdout).strip()}")
        else:
            print("Squashfs integrity OK")
    else:
        raise RuntimeError(f"no check method for type '{kind}'")


def _mount_image(kind: str, d: Path, meta: dict[str, Any], mnt: Path) -> None:
    # For encrypted images, use the LUKS mapper device instead of the raw file
    mapper = meta.get("luks_mapper")
    dev = Path(f"/dev/mapper/{mapper}") if mapper and Path(f"/dev/mapper/{mapper}").exists() else None
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
        name = meta["name"]
        base_mnt = MOUNTS / f"{name}_base"
        base_mnt.mkdir(parents=True, exist_ok=True)
        sh(["mount", "-o", "loop,ro", str(d / meta["base"]), str(base_mnt)], sudo=True)
        if meta.get("save"):
            save_mnt = MOUNTS / f"{name}_save"
            save_mnt.mkdir(parents=True, exist_ok=True)
            try:
                sh(["mount", "-o", "loop", str(d / meta["save"]), str(save_mnt)], sudo=True)
            except Exception:
                sh(["umount", str(base_mnt)], sudo=True, check=False)
                raise
            upper = save_mnt / "upper"
            work = save_mnt / ".work"
            sh(["mkdir", "-p", str(upper), str(work)], sudo=True)
            try:
                sh(["mount", "-t", "overlay", "overlay",
                    "-o", f"lowerdir={base_mnt},upperdir={upper},workdir={work}",
                    str(mnt)], sudo=True)
            except Exception:
                sh(["umount", str(save_mnt)], sudo=True, check=False)
                sh(["umount", str(base_mnt)], sudo=True, check=False)
                with contextlib.suppress(Exception):
                    save_mnt.rmdir()
                with contextlib.suppress(Exception):
                    base_mnt.rmdir()
                raise
        else:
            sh(["mount", "-o", "loop,ro", str(d / meta["base"]), str(mnt)], sudo=True)
    elif kind == "overlay":
        name = meta["name"]
        base_mnt = MOUNTS / f"{name}_base"
        save_mnt = MOUNTS / f"{name}_save"
        base_mnt.mkdir(parents=True, exist_ok=True)
        save_mnt.mkdir(parents=True, exist_ok=True)
        sh(["mount", "-o", "loop,ro", str(d / meta["base"]), str(base_mnt)], sudo=True)
        try:
            sh(["mount", "-o", "loop", str(d / meta["save"]), str(save_mnt)], sudo=True)
        except Exception:
            sh(["umount", str(base_mnt)], sudo=True, check=False)
            raise
        upper = save_mnt / "upper"
        work = save_mnt / ".work"
        sh(["mkdir", "-p", str(upper), str(work)], sudo=True)
        try:
            sh(["mount", "-t", "overlay", "overlay",
                "-o", f"lowerdir={base_mnt},upperdir={upper},workdir={work}",
                str(mnt)], sudo=True)
        except Exception:
            sh(["umount", str(save_mnt)], sudo=True, check=False)
            sh(["umount", str(base_mnt)], sudo=True, check=False)
            raise


def do_mount(name: str) -> str:
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    mnt = MOUNTS / name
    if is_mounted(mnt):
        raise RuntimeError(f"already mounted at {mnt}")
    mnt.mkdir(parents=True, exist_ok=True)
    _mount_image(meta["type"], sys_dir(name), meta, mnt)
    return str(mnt)


def do_umount(name: str) -> None:
    meta = load_meta(name)
    mnt = MOUNTS / name
    if not is_mounted(mnt):
        raise RuntimeError(f"'{name}' is not mounted")
    sh(["umount", str(mnt)], sudo=True)
    is_overlay = (meta and meta["type"] == "overlay") or is_mounted(MOUNTS / f"{name}_base")
    if is_overlay:
        for sub in (f"{name}_base", f"{name}_save"):
            p = MOUNTS / sub
            if is_mounted(p):
                sh(["umount", str(p)], sudo=True, check=False)
            try:
                shutil.rmtree(p)
            except OSError:
                logger.warning("Failed to remove mount point %s", p)
    try:
        shutil.rmtree(mnt)
    except OSError:
        logger.warning("Failed to remove mount point %s", mnt)
    if meta and meta.get("encrypted"):
        mapper = meta.get("luks_mapper")
        if mapper and Path(f"/dev/mapper/{mapper}").exists():
            sh(["cryptsetup", "close", mapper], sudo=True, check=False)


def do_snap(name: str) -> None:
    """Take a snapshot of an overlay save layer."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if meta["type"] not in ("overlay",):
        raise RuntimeError(f"snapshot only supported for overlay type, not '{meta['type']}'")

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
    if sys_dir(base_name).exists():
        raise RuntimeError(f"'{base_name}' already exists")

    ensure_dirs()
    d = sys_dir(base_name)
    d.mkdir(parents=True, exist_ok=True)
    disk_name = "disk.img"
    dst = d / disk_name
    shutil.copy2(src_path, dst)
    meta: dict[str, Any] = {
        "name": base_name,
        "type": img_type,
        "disk": disk_name,
        "fstype": img_type,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(src_path),
    }
    save_meta(base_name, meta)


def do_resize(name: str, new_size: str) -> None:
    """Resize the primary filesystem image."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if is_mounted(MOUNTS / name):
        raise RuntimeError(f"'{name}' is mounted — umount first")

    # Validate new_size early
    _validate_size(new_size)

    d = sys_dir(name)
    new_size = resolve_size(new_size)
    target_bytes = _size_to_bytes(new_size)

    if meta["type"] in ("ext4", "fat32"):
        img = d / meta["disk"]
        old_size = img.stat().st_size
        if target_bytes == old_size:
            logger.info("Already %s", new_size)
            return
        if target_bytes > old_size:
            sh(["truncate", "-s", str(target_bytes), str(img)])
            if meta["type"] == "ext4":
                sh(["e2fsck", "-f", str(img)], sudo=True, check=False)
                sh(["resize2fs", str(img)], sudo=True)
        elif meta["type"] == "ext4":
            sh(["e2fsck", "-f", str(img)], sudo=True, check=False)
            sh(["resize2fs", str(img), new_size], sudo=True)
            sh(["truncate", "-s", str(target_bytes), str(img)])
        else:
            raise RuntimeError("shrinking fat32 not supported")
    elif meta["type"] == "overlay":
        save_img = d / meta["save"]
        old_size = save_img.stat().st_size
        if target_bytes == old_size:
            return
        if is_mounted(MOUNTS / f"{name}_save"):
            raise RuntimeError("save layer is mounted — umount first")
        sh(["truncate", "-s", str(target_bytes), str(save_img)])
        sh(["e2fsck", "-f", str(save_img)], sudo=True, check=False)
        sh(["resize2fs", str(save_img)], sudo=True)
    else:
        raise RuntimeError(f"resize not supported for type '{meta['type']}'")

    save_meta(name, meta)
