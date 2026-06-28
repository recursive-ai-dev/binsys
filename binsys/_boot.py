"""Boot disk assembly: GPT partitioning, bootloader installation, EFI handling."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from binsys._util import (
    EFI_BIN,
    OVMF_CANDIDATES,
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
    raw = meta.get("partitions", [])
    if not isinstance(raw, list):
        return []
    return [Partition(p["label"], p["size"], p["fs"], p.get("flags", [])) for p in raw if isinstance(p, dict)]


def is_gpt_layout(name: str) -> bool:
    """Return True if the system has a GPT partition layout."""
    meta = load_meta(name)
    return bool(meta and meta.get("partitions"))


def _rootfs_partnum(parts: list[Partition]) -> int:
    """Find the index of the partition labeled 'rootfs' (1-based)."""
    for i, p in enumerate(parts):
        if p.label == "rootfs":
            return i + 1
    return 2  # default fallback


def _gpt_kernel_cmdline(boot_name: str, parts: list[Partition]) -> str:
    """Generate a default kernel command line for a GPT layout."""
    root_part = _rootfs_partnum(parts)
    # Using PARTLABEL for robust discovery by the initramfs
    return f"root=PARTLABEL={boot_name}-p{root_part} ro quiet console=ttyS0 console=tty1"


def _gpt_partition_plan(total_size: int, esp_size: int = 512 * 1024 * 1024) -> list[Partition]:
    """Build a classic partition plan: ESP + rootfs."""
    # Ensure minimum rootfs size (e.g., 64MiB)
    rootfs_size = max(64 * 1024 * 1024, total_size - esp_size - (2 * 1024 * 1024))
    return [
        Partition("esp", str(esp_size), "fat32", ["boot", "esp"]),
        Partition("rootfs", str(rootfs_size), "ext4", []),
    ]


def _align_up(val: int, align: int) -> int:
    """Align a value up to the nearest multiple of align."""
    return ((val + align - 1) // align) * align


def _assemble_gpt(
    img: Path,
    parts: list[Partition],
    boot_name: str,
) -> None:
    """Write GPT partition table + format each partition with 1MiB alignment."""
    img_size = img.stat().st_size
    sector_size = 512
    # 1 MiB alignment for performance and compatibility
    alignment = 1024 * 1024
    current_start = alignment  # Start first partition at 1 MiB

    # Zap existing table and create new GPT
    sh(["sgdisk", "-Z", str(img)], sudo=True)
    sh(["sgdisk", "-g", str(img)], sudo=True)

    r = sh(["losetup", "--find", "--show", "-P", str(img)], capture=True, sudo=True)
    loop_dev = r.stdout.strip()
    if not loop_dev:
        raise RuntimeError("failed to allocate loop device")

    try:
        for i, p in enumerate(parts, 1):
            size_bytes = _size_to_bytes(p.size)

            # For the last partition, use remaining space minus 1MB for GPT backup
            if i == len(parts):
                end_lba = (img_size - alignment) // sector_size - 1
            else:
                end_lba = (current_start + size_bytes) // sector_size - 1

            start_lba = current_start // sector_size

            # Partition types: EF00 for ESP, 8300 for Linux filesystem
            type_code = "ef00" if p.fs == "fat32" and "esp" in p.flags else "8300"
            part_label = f"{boot_name}-p{i}"
            part_dev = f"{loop_dev}p{i}"

            logger.info("Creating partition %d: %s (%s) at LBA %d-%d", i, p.label, p.fs, start_lba, end_lba)

            sh([
                "sgdisk",
                "-n", f"{i}:{start_lba}:{end_lba}",
                "-t", f"{i}:{type_code}",
                "-c", f"{i}:{part_label[:36]}",
                str(img)
            ], sudo=True)

            # Format the partition
            if p.fs == "fat32":
                sh(["mkfs.fat", "-F32", "-n", p.label[:11].upper(), part_dev], sudo=True)
            elif p.fs == "ext4":
                sh(["mkfs.ext4", "-F", "-L", p.label[:16], part_dev], sudo=True)

            # Advance current_start for next partition, aligned
            current_start = _align_up((end_lba + 1) * sector_size, alignment)

    finally:
        sh(["losetup", "-d", loop_dev], sudo=True, check=False)


def _find_ovmf() -> tuple[str | None, str | None]:
    """Find OVMF UEFI firmware on the host."""
    for code, vars_ in OVMF_CANDIDATES:
        if os.path.exists(code):
            return code, vars_
    return None, None


def _ensure_bootloader() -> str | None:
    """Ensure the puppyboot EFI binary exists, building if needed."""
    if not EFI_BIN.exists():
        logger.info("PuppyBoot EFI not found at %s. Attempting build...", EFI_BIN)
        cargo = shutil.which("cargo")
        if not cargo:
            logger.warning("cargo not installed — cannot build bootloader")
            return None

        boot_dir = REPO_DIR / "boot"
        try:
            sh([cargo, "build", "--release", "--target", "x86_64-unknown-uefi"],
               cwd=boot_dir)
        except Exception as e:
            logger.error("Failed to build bootloader: %s", e)
            return None

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

    try:
        total_size = _size_to_bytes(resolve_size(size))
        esp_bytes = _size_to_bytes(resolve_size(esp_size))

        sh(["truncate", "-s", str(total_size), str(img)])

        parts = _gpt_partition_plan(total_size, esp_bytes)
        _assemble_gpt(img, parts, name)

        # Install bootloader and configuration to ESP
        if bootloader:
            efi_bin_path = _ensure_bootloader()
            if not efi_bin_path:
                logger.warning("Bootloader binary not available; skipping installation.")
            else:
                _install_bootloader_to_esp(img, name, parts, efi_bin_path, kernel, initrd, cmdline)

        meta: dict[str, Any] = {
            "name": name,
            "type": "ext4",
            "disk": "disk.img",
            "fstype": "ext4",
            "created": datetime.now().isoformat(timespec="seconds"),
            "partitions": [{"label": p.label, "size": p.size, "fs": p.fs} for p in parts],
        }
        save_meta(name, meta)
        logger.info("Successfully built bootable disk '%s'", name)

    except Exception:
        # Cleanup on failure to avoid leaving a corrupt partial system
        if d.exists():
            shutil.rmtree(d)
        raise


def _install_bootloader_to_esp(
    img_path: Path,
    name: str,
    parts: list[Partition],
    efi_bin_path: str,
    kernel: str | None,
    initrd: str | None,
    cmdline: str | None,
) -> None:
    """Mount ESP and install the bootloader and its configuration."""
    with tempfile.TemporaryDirectory() as tmp_mnt_str:
        esp_mnt = Path(tmp_mnt_str)

        r = sh(["losetup", "--find", "--show", "-P", str(img_path)], capture=True, sudo=True)
        loop_dev = r.stdout.strip()
        if not loop_dev:
            raise RuntimeError("failed to allocate loop device for ESP installation")

        try:
            esp_part = f"{loop_dev}p1"
            sh(["mount", esp_part, str(esp_mnt)], sudo=True)

            try:
                # 1. Standard UEFI boot path
                efi_boot_dir = esp_mnt / "EFI" / "BOOT"
                sh(["mkdir", "-p", str(efi_boot_dir)], sudo=True)
                sh(["cp", efi_bin_path, str(efi_boot_dir / "BOOTX64.EFI")], sudo=True)

                # 2. PuppyBoot configuration
                pb_dir = esp_mnt / "EFI" / "puppyboot"
                entries_dir = pb_dir / "entries"
                sh(["mkdir", "-p", str(entries_dir)], sudo=True)

                loader_cfg = "default 0\ntimeout 5\neditor yes\n"
                # Use a temp file and sudo cp to handle permissions on the mounted FAT fs if needed
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
                    tf.write(loader_cfg)
                    tf_path = tf.name
                sh(["mv", tf_path, str(pb_dir / "loader.conf")], sudo=True)

                # 3. Kernel and initrd installation
                if kernel:
                    arch_dir = esp_mnt / "EFI" / "arch"
                    sh(["mkdir", "-p", str(arch_dir)], sudo=True)

                    kernel_src = Path(kernel)
                    if not kernel_src.exists():
                        raise RuntimeError(f"kernel not found: {kernel}")

                    sh(["cp", str(kernel_src), str(arch_dir / kernel_src.name)], sudo=True)

                    initrd_line = ""
                    if initrd:
                        initrd_src = Path(initrd)
                        if not initrd_src.exists():
                            raise RuntimeError(f"initrd not found: {initrd}")
                        sh(["cp", str(initrd_src), str(arch_dir / initrd_src.name)], sudo=True)
                        initrd_line = f"initrd  /EFI/arch/{initrd_src.name}"

                    entry_cmdline = cmdline or _gpt_kernel_cmdline(name, parts)
                    entry_content = f"""title   {name}
type    linux-stub
kernel  /EFI/arch/{kernel_src.name}
{initrd_line}
cmdline {entry_cmdline}
"""
                    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
                        tf.write(entry_content)
                        tf_path = tf.name
                    sh(["mv", tf_path, str(entries_dir / f"{name}.conf")], sudo=True)

            finally:
                sh(["umount", str(esp_mnt)], sudo=True)

        finally:
            sh(["losetup", "-d", loop_dev], sudo=True, check=False)
