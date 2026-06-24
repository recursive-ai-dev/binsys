"""Boot disk assembly: GPT partitioning, bootloader installation, EFI handling."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from binsys._util import (
    EFI_BIN,
    REPO_DIR,
    _size_to_bytes,
    _validate_name,
    ensure_dirs,
    load_meta,
    logger,
    resolve_size,
    save_meta,
    sh,
    sys_dir,
)


class Partition:
    """Describes a single GPT partition."""

    def __init__(self, label: str, size: str, fs: str, flags: list[str] | None = None) -> None:
        self.label = label
        self.size = size
        self.fs = fs
        self.flags = flags or []

    def __repr__(self) -> str:
        return f"Partition({self.label}, {self.size}, {self.fs})"


def layout_of(name: str) -> list[Partition]:
    """Return the partition layout for a given system."""
    meta = load_meta(name)
    if not meta:
        raise RuntimeError(f"'{name}' not found")
    return meta.get("partitions", [])


def is_gpt_layout(name: str) -> bool:
    meta = load_meta(name)
    return bool(meta and meta.get("partitions"))


def gpt_layout_for(name: str) -> list[Partition]:
    return layout_of(name)


def _rootfs_partnum(parts: list[Partition]) -> int:
    for i, p in enumerate(parts):
        if p.label == "rootfs":
            return i + 1
    return 2  # default


def _partnum(parts: list[Partition], label: str) -> int | None:
    for i, p in enumerate(parts):
        if p.label == label:
            return i + 1
    return None


def _partlabel(name: str, parts: list[Partition], idx: int) -> str:
    if 0 <= idx - 1 < len(parts):
        return parts[idx - 1].label
    return f"p{idx}"


def _gpt_kernel_cmdline(boot_name: str, parts: list[Partition]) -> str:
    root_part = _rootfs_partnum(parts)
    return f"root=PARTLABEL={boot_name}-p{root_part} ro quiet"


def _gpt_partition_plan(total_size: int, esp_size: int = 512 * 1024 * 1024) -> list[Partition]:
    """Build a classic partition plan: ESP + rootfs."""
    return [
        Partition("esp", str(esp_size), "fat32", ["boot"]),
        Partition("rootfs", str(total_size - esp_size), "ext4", []),
    ]


def _align_up(val: int, align: int) -> int:
    return ((val + align - 1) // align) * align


def _assemble_gpt(
    img: Path,
    parts: list[Partition],
    boot_name: str,
) -> None:
    """Write GPT partition table + format each partition."""
    img_size = img.stat().st_size
    sector = 512
    part_start = 2048 * sector  # 1 MiB offset for GPT

    r = sh(["losetup", "--find", "--show", "-P", str(img)], capture=True, sudo=True)
    loop_dev = r.stdout.strip()
    if not loop_dev:
        raise RuntimeError("failed to allocate loop device")
    try:
        for i, p in enumerate(parts, 1):
            size_bytes = _size_to_bytes(p.size)
            if i == len(parts):
                size_bytes = img_size - part_start
            else:
                size_bytes = min(size_bytes, img_size - part_start)
            size_bytes = _align_up(size_bytes, 1024 * 1024)

            part_label = f"{boot_name}-p{i}"
            part_dev = f"{loop_dev}p{i}"
            if p.fs == "fat32":
                sh(["sgdisk", "-n", f"{i}:{part_start // sector}:+{size_bytes // sector}",
                    "-t", f"{i}:ef00",
                    "-c", f"{i}:{part_label[:36]}",
                    str(img)])
                sh(["mkfs.fat", "-F32", part_dev])
                sh(["fatlabel", part_dev, p.label[:11]])
            elif p.fs == "ext4":
                sh(["sgdisk", "-n", f"{i}:{part_start // sector}:+{size_bytes // sector}",
                    "-t", f"{i}:8300",
                    "-c", f"{i}:{part_label[:36]}",
                    str(img)])
                sh(["mkfs.ext4", "-F", part_dev])
                sh(["e2label", part_dev, p.label[:16]])

            part_start += size_bytes
    finally:
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)


def _find_ovmf() -> tuple[str | None, str | None]:
    """Find OVMF UEFI firmware on the host."""
    from binsys._util import OVMF_CANDIDATES
    for code, vars_ in OVMF_CANDIDATES:
        if os.path.exists(code):
            return code, vars_
    return None, None


def _ensure_bootloader(name: str) -> str | None:
    """Ensure the puppyboot EFI binary exists, building if needed."""
    if not EFI_BIN.exists():
        logger.info("PuppyBoot EFI not found at %s", EFI_BIN)
        cargo = shutil.which("cargo")
        if not cargo:
            logger.warning("cargo not installed — cannot build bootloader")
            return None
        boot_dir = REPO_DIR / "boot"
        logger.info("Building PuppyBoot …")
        sh([cargo, "build", "--release", "--target", "x86_64-unknown-uefi"],
           cwd=str(boot_dir))
    return str(EFI_BIN) if EFI_BIN.exists() else None


def build_bootdisk(
    name: str,
    size: str = "4G",
    esp_size: str = "512M",
    kernel: str | None = None,
    initrd: str | None = None,
    cmdline: str | None = None,
    bootloader: bool = False,
    auto_esp: bool = False,
) -> None:
    """Build a bootable disk image with GPT partitioning and optional bootloader."""
    _validate_name(name)
    ensure_dirs()
    d = sys_dir(name)
    if d.exists():
        raise RuntimeError(f"'{name}' already exists")

    d.mkdir(parents=True, exist_ok=True)
    img = d / "disk.img"
    total_size = _size_to_bytes(resolve_size(size))
    esp_bytes = _size_to_bytes(resolve_size(esp_size))

    sh(["truncate", "-s", str(total_size), str(img)])

    parts = _gpt_partition_plan(total_size, esp_bytes)
    _assemble_gpt(img, parts, name)

    # Mount ESP and install bootloader if requested
    if bootloader:
        efi_path = _ensure_bootloader(name)
        if efi_path:
            esp_mnt = d / ".esp_mnt"
            esp_mnt.mkdir(exist_ok=True)
            r = sh(["losetup", "--find", "--show", "-P", str(img)], capture=True, sudo=True)
            loop_dev = r.stdout.strip()
            if not loop_dev:
                raise RuntimeError("failed to allocate loop device")
            try:
                esp_part = f"{loop_dev}p1"
                sh(["mount", str(esp_part), str(esp_mnt)], sudo=True)
                efi_dir = esp_mnt / "EFI" / "BOOT"
                efi_dir.mkdir(parents=True, exist_ok=True)
                sh(["cp", efi_path, str(efi_dir / "BOOTX64.EFI")])

                puppyboot_dir = esp_mnt / "EFI" / "puppyboot"
                puppyboot_dir.mkdir(parents=True, exist_ok=True)
                entries_dir = puppyboot_dir / "entries"
                entries_dir.mkdir(parents=True, exist_ok=True)

                cfg = """default 0
timeout 5
editor yes
"""
                (puppyboot_dir / "loader.conf").write_text(cfg)

                entry_cmdline = cmdline or _gpt_kernel_cmdline(name, parts)
                if kernel:
                    entry = f"""title   {name}
type    linux-stub
kernel  /EFI/arch/{Path(kernel).name}
initrd  /EFI/arch/{Path(initrd).name if initrd else 'initramfs.img'}
cmdline {entry_cmdline}
"""
                    (entries_dir / f"{name}.conf").write_text(entry)

                sh(["umount", str(esp_mnt)], sudo=True)
            finally:
                sh(["losetup", "-d", loop_dev], sudo=True, check=False)
                shutil.rmtree(esp_mnt, ignore_errors=True)

    meta: dict[str, Any] = {
        "name": name,
        "type": "ext4",
        "disk": "disk.img",
        "fstype": "ext4",
        "created": datetime.now().isoformat(timespec="seconds"),
        "partitions": [{"label": p.label, "size": p.size, "fs": p.fs} for p in parts],
    }
    save_meta(name, meta)
