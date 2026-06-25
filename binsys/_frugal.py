"""Frugal overlay operations: conversion, snapshots, rollback, merge."""

from __future__ import annotations

import contextlib
import shutil
from datetime import datetime
from typing import Any

from binsys._crypto import _ensure_app_unlocked
from binsys._util import (
    MOUNTS,
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
        raise RuntimeError(f"'{name}' not found")
    d = sys_dir(name)
    if meta.get("type") == "overlay" and meta.get("frugal"):
        logger.info("'%s' already frugal/overlay", name)
        return

    old_type = meta["type"]
    if old_type not in ("ext4", "squashfs"):
        raise RuntimeError(f"cannot convert type '{old_type}' to frugal")

    src_name = meta.get("disk") or meta.get("base")
    if not src_name:
        raise RuntimeError("no source image name in metadata")
    img_path = d / src_name
    if not img_path.exists():
        raise RuntimeError(f"source image not found: {img_path}")

    # If ext4, we need to mount it, squash it, then create a save layer
    base_img = d / "base.sfs"
    save_img = d / "save.img"
    save_sz = "512M"

    if old_type == "ext4":
        tmp_mnt = d / ".convert_mnt"
        tmp_mnt.mkdir(exist_ok=True)
        try:
            sh(["mount", "-o", "loop", str(img_path), str(tmp_mnt)], sudo=True)
            sh(["mksquashfs", str(tmp_mnt), str(base_img), "-noappend"], sudo=True)
            sh(["umount", str(tmp_mnt)], sudo=True)
        finally:
            shutil.rmtree(tmp_mnt, ignore_errors=True)
        shutil.move(str(img_path), str(img_path.with_suffix(".orig")))
    else:
        shutil.move(str(img_path), str(img_path.with_suffix(".orig")))
        shutil.copy2(str(img_path.with_suffix(".orig")), str(base_img))

    sh(["truncate", "-s", save_sz, str(save_img)])
    sh(["mkfs.ext4", "-F", str(save_img)])

    meta["type"] = "overlay"
    meta["frugal"] = True
    meta["base"] = "base.sfs"
    meta["save"] = "save.img"
    meta.pop("disk", None)
    save_meta(name, meta)
    logger.info("Converted '%s' to frugal overlay", name)


def do_frugal_save_snapshot(name: str, label: str | None = None) -> None:
    """Snapshot the save layer of a frugal system."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if meta.get("type") != "overlay" or not meta.get("frugal"):
        raise RuntimeError(f"'{name}' is not a frugal system")
    d = sys_dir(name)
    save_img = d / meta["save"]
    snap_dir = d / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    snap_name = f"save_{ts}{suffix}.img"
    snap_path = snap_dir / snap_name
    if snap_path.exists():
        raise RuntimeError(f"snapshot '{snap_name}' already exists")
    sh(["cp", "-a", str(save_img), str(snap_path)], sudo=True)
    print(f"Snapshot saved: {snap_path}")


def do_frugal_list_snapshots(name: str) -> list[dict[str, Any]]:
    """List save snapshots for a frugal system."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    snap_dir = sys_dir(name) / "snapshots"
    if not snap_dir.exists():
        return []
    snaps: list[dict[str, Any]] = []
    for p in sorted(snap_dir.iterdir()):
        if p.suffix == ".img":
            snaps.append({
                "name": p.name,
                "path": str(p),
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return snaps


def do_frugal_rollback(name: str, snap: str) -> None:
    """Restore a save snapshot."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    d = sys_dir(name)
    snap_path = d / "snapshots" / snap
    if not snap_path.exists():
        raise RuntimeError(f"snapshot '{snap}' not found")
    save_img = d / meta["save"]
    if is_mounted(MOUNTS / name) or is_mounted(MOUNTS / f"{name}_save"):
        raise RuntimeError("system is mounted — umount first")
    sh(["cp", "-a", str(snap_path), str(save_img)], sudo=True)
    logger.info("Rolled back '%s' to snapshot '%s'", name, snap)


def do_frugal_merge(name: str) -> None:
    """Merge save layer into base.sfs (flatten overlay)."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    if meta.get("type") != "overlay" or not meta.get("frugal"):
        raise RuntimeError(f"'{name}' is not a frugal system")
    d = sys_dir(name)
    base_img = d / meta["base"]
    save_img = d / meta["save"]

    tmp_mnt = d / ".merge_mnt"
    tmp_save = d / ".merge_save"
    tmp_mnt.mkdir(exist_ok=True)
    try:
        base_mnt = MOUNTS / f"{name}_base"
        save_mnt = MOUNTS / f"{name}_save"
        base_mnt.mkdir(parents=True, exist_ok=True)
        save_mnt.mkdir(parents=True, exist_ok=True)
        sh(["mount", "-o", "loop,ro", str(base_img), str(base_mnt)], sudo=True)
        sh(["mount", "-o", "loop", str(save_img), str(save_mnt)], sudo=True)
        upper = save_mnt / "upper"
        sh(["mount", "-t", "overlay", "overlay",
            "-o", f"lowerdir={base_mnt},upperdir={upper},workdir={save_mnt / '.work'}",
            str(tmp_mnt)], sudo=True)
        sh(["mksquashfs", str(tmp_mnt), str(tmp_save), "-noappend"], sudo=True)
        sh(["umount", str(tmp_mnt)], sudo=True)
        sh(["umount", str(save_mnt)], sudo=True)
        sh(["umount", str(base_mnt)], sudo=True)
    finally:
        shutil.rmtree(tmp_mnt, ignore_errors=True)
        for p in (MOUNTS / f"{name}_base", MOUNTS / f"{name}_save"):
            with contextlib.suppress(OSError):
                shutil.rmtree(p)

    shutil.move(str(base_img), str(base_img.with_suffix(".bak")))
    shutil.move(str(tmp_save), str(base_img))
    sh(["truncate", "-s", "512M", str(save_img)])
    sh(["mkfs.ext4", "-F", str(save_img)])
    logger.info("Merged '%s' — save layer reset", name)
