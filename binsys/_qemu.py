"""QEMU virtual machine launch helpers."""

from __future__ import annotations

import os
import shutil
from typing import Any

from binsys._boot import _find_ovmf
from binsys._util import (
    QEMU_ARCHES,
    sys_dir,
)


def _build_qcmd(
    name: str,
    meta: dict[str, Any],
    kvm: bool = True,
    gdb: bool = False,
    boot: str = "menu",
    uefi: bool = True,
    memory: str = "2048",
    preset: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build the QEMU command line for a given system."""
    # Detect architecture
    arch_key = meta.get("arch", "x86_64")
    qemu_bin, _arch_opts = QEMU_ARCHES.get(arch_key, QEMU_ARCHES["x86_64"])

    if not shutil.which(qemu_bin):
        raise RuntimeError(f"{qemu_bin} not found — install qemu-system-{arch_key}")

    d = sys_dir(name)
    cmd: list[str] = [qemu_bin]
    cmd += ["-m", memory]
    cmd += ["-machine", "q35" if arch_key == "x86_64" else "virt", f"accel={'kvm:tcg' if kvm else 'tcg'}"]
    cmd += ["-cpu", "host" if kvm else "max"]
    cmd += ["-smp", (os.cpu_count() and str(min(os.cpu_count() or 4, 8))) or "4"]

    if gdb:
        cmd += ["-s", "-S"]

    # OVMF (UEFI) or BIOS
    if uefi:
        code, vars_ = _find_ovmf()
        if code:
            cmd += ["-drive", f"if=pflash,format=raw,readonly=on,file={code}"]
        if vars_:
            vars_copy = d / ".OVMF_VARS.fd"
            if not vars_copy.exists():
                shutil.copy2(vars_, vars_copy)
            cmd += ["-drive", f"if=pflash,format=raw,file={vars_copy}"]

    # If meta has kernel+initrd, boot via -kernel (direct kernel boot)
    kernel_path = d / meta["kernel"] if meta.get("kernel") else None
    initrd_path = d / meta["initrd"] if meta.get("initrd") else None
    if kernel_path and kernel_path.exists() and initrd_path and initrd_path.exists():
        cmd += ["-kernel", str(kernel_path)]
        cmd += ["-initrd", str(initrd_path)]
        cmdline = meta.get("cmdline", "console=ttyS0")
        cmd += ["-append", cmdline]
        # Also attach a data disk with base.sfs if present (for frugal overlay boots)
        data_img = d / meta.get("data_disk", "data.img")
        if data_img.exists():
            cmd += ["-drive", f"file={data_img},format=raw,if=virtio"]
        elif meta.get("base"):
            # For frugal: attach the save layer as a writable drive, plus
            # create a data disk image with base.sfs + save dir
            # Try to use the boot drive directly
            pass
    else:
        # Attach the disk image
        img_path = d / meta["disk"]
        if meta["type"] in ("ext4", "fat32", "iso", "iso9660"):
            if img_path.suffix == ".iso":
                cmd += ["-drive", f"file={img_path},format=raw,media=cdrom"]
                cmd += ["-boot", "d"]
            else:
                cmd += ["-drive", f"file={img_path},format=raw,if=virtio"]
                cmd += ["-boot", boot]
        elif meta["type"] in ("squashfs", "overlay"):
            cmd += ["-drive", f"file={img_path},format=raw,if=virtio"]
            cmd += ["-boot", boot]

    # Network
    cmd += ["-nic", "user,model=virtio-net-pci"]

    # Display — default to nographic on headless hosts; allow override via env
    display = os.environ.get("BINSYS_QEMU_DISPLAY", "").lower()
    if display == "none" or display == "nographic":
        cmd += ["-nographic"]
        use_serial_stdio = False
    elif display == "gtk" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        cmd += ["-display", "gtk,gl=off"]
        use_serial_stdio = True
    else:
        cmd += ["-nographic"]
        use_serial_stdio = False
    cmd += ["-vga", "virtio"]

    # Audio — only enabled when explicitly requested or a pulseaudio server is present
    enable_audio = os.environ.get("BINSYS_QEMU_AUDIO", "").lower() in ("1", "true", "yes")
    if not enable_audio and shutil.which("pactl"):
        enable_audio = True
    if enable_audio:
        cmd += ["-audiodev", "pa,id=pa", "-device", "intel-hda", "-device", "hda-duplex,audiodev=pa"]

    # Serial for debug (redundant with -nographic, so skip in that case)
    if use_serial_stdio:
        cmd += ["-serial", "stdio"]

    if extra:
        cmd += extra

    return cmd
