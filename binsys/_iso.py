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
    """Locate an available ISO creation tool in the system PATH."""
    for binary in ISO_CREATORS:
        if shutil.which(binary):
            return binary
    raise RuntimeError(
        f"no ISO creation tool found — install one of: {', '.join(ISO_CREATORS)}"
    )


def do_iso_create(name: str, output: str | None = None) -> None:
    """Create a bootable ISO from an existing system's primary files."""
    _ensure_app_unlocked(name)
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")

    d = sys_dir(name)
    iso_name = output or f"{name}.iso"
    iso_path = Path(iso_name).resolve()

    # Check output location
    if iso_path.exists() and not iso_path.is_file():
        raise RuntimeError(f"ISO output path exists and is not a file: {iso_path}")

    # Use a secure temp directory for staging the ISO contents
    with tempfile.TemporaryDirectory(prefix=f"binsys-iso-{name}-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        iso_root = tmp_dir / "iso_root"
        iso_root.mkdir()

        kind = meta["type"]
        vol_label = f"binsys-{name}"[:32]

        # Determine what to include based on system type
        if kind in ("ext4", "fat32", "iso", "iso9660"):
            disk_name = meta.get("disk", "disk.img")
            if (d / disk_name).exists():
                shutil.copy2(d / disk_name, iso_root / "system.img")
        elif kind in ("overlay", "frugal", "squashfs"):
            base_name = meta.get("base", "base.sfs")
            if (d / base_name).exists():
                shutil.copy2(d / base_name, iso_root / "base.sfs")

            save_name = meta.get("save")
            if save_name and (d / save_name).exists():
                shutil.copy2(d / save_name, iso_root / "save.img")
        else:
            raise RuntimeError(f"ISO creation not supported for type '{kind}'")

        # Include metadata
        shutil.copy2(d / "meta.json", iso_root / "meta.json")

        logger.info("Generating ISO: %s", iso_path)
        sh([
            _find_iso_tool(), "-o", str(iso_path),
            "-V", vol_label, "-R", "-J",
            "-input-charset", "utf-8",
            str(iso_root)
        ])

    logger.info("ISO created successfully: %s (%s)", iso_path,
                Path(iso_path).stat().st_size)


def do_iso_from_dir(
    source_dir: str,
    output: str | None = None,
    label: str | None = None,
    bootable: bool = False
) -> None:
    """Create an ISO image from an arbitrary directory."""
    src = Path(source_dir).resolve()
    if not src.is_dir():
        raise RuntimeError(f"source is not a directory: {source_dir}")

    vol_label = label or f"binsys-{src.name}"[:32]
    iso_path = Path(output or f"{src.name}.iso").resolve()

    # ISO tool detection
    tool = _find_iso_tool()

    cmd = [
        tool, "-o", str(iso_path),
        "-V", vol_label, "-R", "-J",
        "-input-charset", "utf-8"
    ]

    if bootable:
        # Check for isolinux
        if (src / "isolinux" / "isolinux.bin").exists():
            cmd += [
                "-b", "isolinux/isolinux.bin",
                "-c", "isolinux/boot.cat",
                "-no-emul-boot", "-boot-load-size", "4",
                "-boot-info-table"
            ]
        # Check for EFI
        elif (src / "EFI" / "BOOT" / "BOOTX64.EFI").exists():
            cmd += [
                "-eltorito-alt-boot",
                "-e", "EFI/BOOT/BOOTX64.EFI",
                "-no-emul-boot"
            ]
        else:
            logger.warning("Bootable ISO requested but no bootloader found in source")

    cmd.append(str(src))

    logger.info("Building ISO from directory '%s'...", src)
    sh(cmd)
    logger.info("ISO created: %s", iso_path)
