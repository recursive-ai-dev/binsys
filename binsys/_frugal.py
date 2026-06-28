"""Frugal overlay operations: conversion, snapshots, rollback, merge."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from binsys import BinSysError
from binsys._crypto import _ensure_app_unlocked
from binsys._util import (
    MOUNTS,
    human,
    is_mounted,
    load_meta,
    logger,
    save_meta,
    sh,
    sys_dir,
)


def convert_to_frugal(name: str) -> None:
    """Non-interactive conversion to frugal overlay: creates base.sfs and save.img."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise BinSysError(f"'{name}' not found")

    d = sys_dir(name)
    if meta.get("type") == "overlay" and meta.get("frugal"):
        logger.info("'%s' is already a frugal system", name)
        return

    old_type = meta["type"]
    if old_type not in ("ext4", "squashfs"):
        raise BinSysError(f"cannot convert type '{old_type}' to frugal")

    src_name = meta.get("disk") or meta.get("base")
    if not src_name:
        raise BinSysError(f"no source image defined for '{name}'")

    img_path = d / src_name
    if not img_path.exists():
        raise BinSysError(f"source image not found: {img_path}")

    base_img = d / "base.sfs"
    save_img = d / "save.img"
    save_sz = "512M"

    # 1. Create base.sfs from source
    if old_type == "ext4":
        tmp_mnt_path = Path(tempfile.mkdtemp(prefix=f".convert-{name}-", dir=d))
        try:
            sh(["mount", "-o", "loop,ro", str(img_path), str(tmp_mnt_path)], sudo=True)
            try:
                logger.info("Creating squashfs base layer for '%s'...", name)
                sh(["mksquashfs", str(tmp_mnt_path), str(base_img), "-noappend"], sudo=True)
            finally:
                sh(["umount", str(tmp_mnt_path)], sudo=True)
        finally:
            with contextlib.suppress(OSError):
                tmp_mnt_path.rmdir()

        # Backup original image
        shutil.move(str(img_path), str(img_path.with_suffix(".img.orig")))
    else:
        # For squashfs, just move it to base.sfs
        shutil.move(str(img_path), str(base_img))

    # 2. Create save layer
    logger.info("Creating ext4 save layer (size=%s)...", save_sz)
    sh(["truncate", "-s", save_sz, str(save_img)])
    sh(["mkfs.ext4", "-F", "-L", f"save-{name}"[:16], str(save_img)])

    # 3. Update metadata
    meta["type"] = "overlay"
    meta["frugal"] = True
    meta["base"] = "base.sfs"
    meta["save"] = "save.img"
    meta["fstype"] = "overlay"
    meta.pop("disk", None)
    save_meta(name, meta)

    logger.info("Successfully converted '%s' to frugal overlay", name)


def do_frugal_save_snapshot(name: str, label: str | None = None) -> None:
    """Snapshot the save layer of a frugal system."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise BinSysError(f"'{name}' not found")

    if meta.get("type") != "overlay" or not meta.get("frugal"):
        raise BinSysError(f"'{name}' is not a frugal system")

    d = sys_dir(name)
    save_img = d / meta["save"]
    if not save_img.exists():
        raise BinSysError(f"save layer not found at {save_img}")

    snap_dir = d / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    snap_name = f"save_{ts}{suffix}.img"
    snap_path = snap_dir / snap_name

    if snap_path.exists():
        raise BinSysError(f"snapshot '{snap_name}' already exists")

    logger.info("Saving snapshot for '%s' to %s...", name, snap_name)
    sh(["cp", "-a", str(save_img), str(snap_path)], sudo=True)
    print(f"Snapshot saved: {snap_name} ({human(snap_path.stat().st_size)})")


def do_frugal_list_snapshots(name: str) -> list[dict[str, Any]]:
    """List save snapshots for a frugal system."""
    meta = load_meta(name)
    if not meta:
        raise BinSysError(f"'{name}' not found")

    snap_dir = sys_dir(name) / "snapshots"
    if not snap_dir.exists():
        return []

    snaps: list[dict[str, Any]] = []
    for p in sorted(snap_dir.iterdir(), key=os.path.getmtime, reverse=True):
        if p.suffix == ".img" and p.name.startswith("save_"):
            snaps.append({
                "name": p.name,
                "path": str(p),
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return snaps


def do_frugal_rollback(name: str, snap: str) -> None:
    """Restore a save snapshot to the current save layer."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise BinSysError(f"'{name}' not found")

    d = sys_dir(name)
    snap_path = d / "snapshots" / snap
    if not snap_path.exists():
        raise BinSysError(f"snapshot '{snap}' not found")

    save_img = d / meta["save"]

    # Check if system is mounted
    if is_mounted(MOUNTS / name) or is_mounted(MOUNTS / f"{name}_save"):
        raise BinSysError(f"system '{name}' is currently mounted — umount first")

    logger.info("Rolling back '%s' to snapshot '%s'...", name, snap)
    sh(["cp", "-a", str(snap_path), str(save_img)], sudo=True)
    logger.info("Rollback complete.")


def do_frugal_merge(name: str) -> None:
    """Merge save layer into base.sfs (flatten overlay and reset save layer)."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise BinSysError(f"'{name}' not found")

    if meta.get("type") != "overlay" or not meta.get("frugal"):
        raise BinSysError(f"'{name}' is not a frugal system")

    d = sys_dir(name)
    base_img = d / meta["base"]
    save_img = d / meta["save"]

    if is_mounted(MOUNTS / name):
        raise BinSysError(f"system '{name}' is currently mounted — umount first")

    # Staging paths
    tmp_merge_root = Path(tempfile.mkdtemp(prefix=f".merge-{name}-", dir=d))
    tmp_new_base = d / "base.sfs.new"

    base_mnt = MOUNTS / f"{name}_base"
    save_mnt = MOUNTS / f"{name}_save"
    merge_mnt = tmp_merge_root / "mnt"
    merge_mnt.mkdir()

    try:
        base_mnt.mkdir(parents=True, exist_ok=True)
        save_mnt.mkdir(parents=True, exist_ok=True)

        # 1. Mount layers
        sh(["mount", "-o", "loop,ro", str(base_img), str(base_mnt)], sudo=True)
        try:
            sh(["mount", "-o", "loop,ro", str(save_img), str(save_mnt)], sudo=True)
            try:
                # 2. Compose overlay to flatten
                upper = save_mnt / "upper"
                if not upper.exists():
                     raise BinSysError("save layer does not contain an 'upper' directory")

                logger.info("Merging layers for '%s'...", name)
                sh(["mount", "-t", "overlay", "overlay",
                    "-o", f"lowerdir={upper}:{base_mnt}",
                    str(merge_mnt)], sudo=True)

                try:
                    # 3. Create new base squashfs
                    sh(["mksquashfs", str(merge_mnt), str(tmp_new_base), "-noappend"], sudo=True)
                finally:
                    sh(["umount", str(merge_mnt)], sudo=True)
            finally:
                sh(["umount", str(save_mnt)], sudo=True)
        finally:
            sh(["umount", str(base_mnt)], sudo=True)

        # 4. Atomically swap base layer
        base_img.replace(base_img.with_suffix(".sfs.bak"))
        tmp_new_base.replace(base_img)

        # 5. Reset save layer
        logger.info("Resetting save layer...")
        sh(["truncate", "-s", "512M", str(save_img)])
        sh(["mkfs.ext4", "-F", "-L", f"save-{name}"[:16], str(save_img)])

        logger.info("Merge complete for '%s'. Save layer has been reset.", name)

    finally:
        shutil.rmtree(tmp_merge_root, ignore_errors=True)
        for p in (base_mnt, save_mnt):
            with contextlib.suppress(OSError):
                if not is_mounted(p):
                    p.rmdir()
