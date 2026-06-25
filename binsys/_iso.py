"""ISO9660 image creation from systems or directories."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from binsys._crypto import _ensure_app_unlocked
from binsys._util import (
    ISO_CREATORS,
    load_meta,
    logger,
    sh,
    sys_dir,
)


def _find_iso_tool() -> str:
    for binary in ISO_CREATORS:
        if shutil.which(binary):
            return binary
    raise RuntimeError("no ISO creation tool found (install mkisofs or genisoimage)")


def do_iso_create(name: str, output: str | None = None) -> None:
    """Create a bootable ISO from an existing system."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    d = sys_dir(name)
    iso_name = output or (name + ".iso")
    iso_path = Path(iso_name)
    if not iso_path.parent.exists():
        raise RuntimeError(f"output directory does not exist: {iso_path.parent}")

    tmp = Path(tempfile.mkdtemp(prefix="binsys-iso-"))
    try:
        iso_dir = tmp / "iso"
        iso_dir.mkdir(parents=True, exist_ok=True)
        kind = meta["type"]
        label = f"binsys-{name}"[:32]
        if kind in ("ext4", "fat32"):
            img = d / meta["disk"]
            sh(["cp", str(img), str(iso_dir / "system.img")])
        elif kind == "overlay":
            base_img = d / meta["base"]
            sh(["cp", str(base_img), str(iso_dir / "base.sfs")])
            if meta.get("save"):
                save_img = d / meta["save"]
                sh(["cp", str(save_img), str(iso_dir / "save.img")])
        elif kind in ("squashfs",):
            base_img = d / meta["base"]
            sh(["cp", str(base_img), str(iso_dir / "base.sfs")])
        else:
            raise RuntimeError(f"ISO creation not supported for type '{kind}'")

        sh([_find_iso_tool(), "-o", str(iso_path),
            "-V", label, "-R", "-J",
            str(iso_dir)])
        logger.info("ISO created: %s", iso_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def do_iso_from_dir(source_dir: str, output: str | None = None, label: str | None = None, bootable: bool = False) -> None:
    """Create an ISO from a directory."""
    src = Path(source_dir)
    if not src.is_dir():
        raise RuntimeError(f"not a directory: {source_dir}")
    vol_label = label or f"binsys-{src.name}"[:32]
    iso_path = Path(output or (src.name + ".iso"))
    if not iso_path.parent.exists():
        raise RuntimeError(f"output directory does not exist: {iso_path.parent}")

    cmd = [_find_iso_tool(), "-o", str(iso_path), "-V", vol_label, "-R", "-J"]
    if bootable:
        # Look for an isolinux or EFI boot image in the source
        isolinux = src / "isolinux" / "isolinux.bin"
        if isolinux.exists():
            cmd += ["-b", "isolinux/isolinux.bin",
                    "-c", "boot.catalog",
                    "-no-emul-boot", "-boot-load-size", "4",
                    "-boot-info-table"]
        elif (src / "EFI" / "BOOT" / "BOOTX64.EFI").exists():
            cmd += ["-eltorito-alt-boot",
                    "-e", "EFI/BOOT/BOOTX64.EFI",
                    "-no-emul-boot"]
    cmd.append(str(src))
    sh(cmd)
    logger.info("ISO created: %s", iso_path)
