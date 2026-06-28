"""QEMU virtual machine launch helpers."""

from __future__ import annotations

import os
import shutil
from typing import Any

from binsys._boot import _find_ovmf
from binsys._util import (
    QEMU_ARCHES,
    logger,
    sys_dir,
)


def _build_qcmd(
    name: str,
    meta: dict[str, Any],
    kvm: bool = True,
    gdb: bool = False,
    boot: str = "menu=on",
    uefi: bool = True,
    memory: str = "2048",
    preset: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build the QEMU command line for a given system."""
    # Detect architecture
    arch_key = meta.get("arch", "x86_64")
    qemu_bin, arch_opts = QEMU_ARCHES.get(arch_key, QEMU_ARCHES["x86_64"])

    if not shutil.which(qemu_bin):
        raise RuntimeError(f"{qemu_bin} not found — install qemu-system-{arch_key}")

    d = sys_dir(name)
    cmd: list[str] = [qemu_bin]
    cmd += ["-m", memory]

    # Machine and CPU selection
    is_x86 = arch_key in ("x86_64", "i386")
    if is_x86:
        cmd += ["-machine", "q35"]
    else:
        cmd += ["-machine", "virt"]

    # KVM/Acceleration
    if kvm and os.path.exists("/dev/kvm"):
        cmd += ["-accel", "kvm", "-cpu", "host"]
    else:
        if kvm:
            logger.warning("KVM requested but /dev/kvm not found — falling back to TCG")
        cmd += ["-accel", "tcg", "-cpu", "max"]

    # SMP (CPU cores)
    cpu_count = os.cpu_count() or 1
    cmd += ["-smp", str(min(cpu_count, 8))]

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
        else:
            logger.warning("UEFI requested but OVMF firmware not found — trying BIOS boot")

    # If meta has kernel+initrd, boot via -kernel (direct kernel boot)
    kernel_path = d / meta["kernel"] if meta.get("kernel") else None
    initrd_path = d / meta["initrd"] if meta.get("initrd") else None
    if kernel_path and kernel_path.exists() and initrd_path and initrd_path.exists():
        cmd += ["-kernel", str(kernel_path)]
        cmd += ["-initrd", str(initrd_path)]
        cmdline = meta.get("cmdline", "console=ttyS0 console=tty1 root=/dev/vda ro")
        cmd += ["-append", cmdline]

        # Primary disk for direct kernel boot
        img_name = meta.get("disk") or meta.get("base")
        if img_name:
            img_path = d / img_name
            if img_path.exists():
                cmd += ["-drive", f"file={img_path},format=raw,if=virtio"]
    else:
        # Standard disk-based boot
        img_name = meta.get("disk") or meta.get("base")
        if not img_name:
             raise RuntimeError(f"no primary disk image defined for '{name}'")

        img_path = d / img_name
        if not img_path.exists():
            raise RuntimeError(f"disk image not found: {img_path}")

        # Determine drive options based on type
        if meta["type"] in ("iso", "iso9660") or img_path.suffix == ".iso":
            cmd += ["-drive", f"file={img_path},format=raw,media=cdrom,readonly=on"]
            cmd += ["-boot", "d"]
        else:
            cmd += ["-drive", f"file={img_path},format=raw,if=virtio"]
            cmd += ["-boot", boot]

    # Secondary save layer for overlay/frugal types (if not already handled)
    if meta.get("type") in ("overlay", "frugal") and meta.get("save"):
        save_path = d / meta["save"]
        if save_path.exists():
            cmd += ["-drive", f"file={save_path},format=raw,if=virtio"]

    # Network
    cmd += ["-nic", "user,model=virtio-net-pci"]

    # Display — default to nographic on headless hosts; allow override via env
    display = os.environ.get("BINSYS_QEMU_DISPLAY", "").lower()
    if display in ("none", "nographic"):
        cmd += ["-nographic"]
        use_serial_stdio = False
    elif display == "gtk" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        # Auto-detect display if possible
        cmd += ["-display", "gtk,gl=off" if display == "gtk" else "default"]
        use_serial_stdio = True
    else:
        cmd += ["-nographic"]
        use_serial_stdio = False

    cmd += ["-vga", "virtio"]

    # Audio
    enable_audio = os.environ.get("BINSYS_QEMU_AUDIO", "").lower() in ("1", "true", "yes")
    if enable_audio:
        cmd += ["-audiodev", "pa,id=pa", "-device", "intel-hda", "-device", "hda-duplex,audiodev=pa"]

    # Serial for debug / headless console
    if use_serial_stdio:
        cmd += ["-serial", "stdio"]

    # Architecture specific extra opts
    cmd += arch_opts

    if extra:
        cmd += extra

    return cmd
