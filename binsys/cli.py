"""CLI entry point: argument parsing and command dispatch."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from binsys._boot import build_bootdisk
from binsys._crypto import (
    do_app_lock,
    do_app_unlock,
    do_encrypt,
    do_hash,
    do_lock,
    do_protect,
    do_unlock,
    do_unprotect,
)
from binsys._frugal import do_frugal_list_snapshots, do_frugal_merge, do_frugal_rollback, do_frugal_save_snapshot
from binsys._image import (
    do_check,
    do_clone,
    do_delete,
    do_export,
    do_import,
    do_mount,
    do_new,
    do_rename,
    do_resize,
    do_snap,
    do_umount,
)
from binsys._iso import do_iso_create, do_iso_from_dir
from binsys._qemu import _build_qcmd
from binsys._util import (
    MOUNTS,
    SCRIPTS_DIR,
    TYPES,
    WIZARD_SCRIPTS,
    all_systems,
    check_dependencies_or_warn,
    die,
    human,
    is_mounted,
    load_meta,
    logger,
    sh,
    sys_dir,
)

try:
    import argcomplete
    _HAS_ARGCOMPLETE = True
except ImportError:
    _HAS_ARGCOMPLETE = False


# ── CLI command handlers ──────────────────────────────────────────────────────


def cmd_new(args: argparse.Namespace) -> None:
    do_new(
        args.name,
        img_type=args.type,
        size=args.size,
        label=args.label,
        distro=args.distro,
        encrypt=args.encrypt,
        boot=args.boot,
        bootloader=args.bootloader,
        auto_esp=args.auto_esp,
        save_size=args.save_size,
    )


def cmd_snap(args: argparse.Namespace) -> None:
    do_snap(args.name)


def cmd_run(args: argparse.Namespace) -> None:
    meta = load_meta(args.name)
    if not meta:
        die(f"'{args.name}' not found")
    extra: list[str] = []
    if args.boot:
        extra += ["-boot", args.boot]
    cmd = _build_qcmd(args.name, meta, kvm=not args.no_kvm, gdb=args.gdb,
                      memory=args.memory, extra=extra)
    print(f"Running: {' '.join(cmd)}")
    sh(cmd, check=False)


def cmd_shell(args: argparse.Namespace) -> None:
    meta = load_meta(args.name)
    if not meta:
        die(f"'{args.name}' not found")
    d = sys_dir(args.name)
    img_path = d / meta["disk"]
    print(f"System: {args.name}")
    print(f"Image:  {img_path}")
    os.system(os.environ.get("SHELL", "/bin/bash"))


def cmd_mount(args: argparse.Namespace) -> None:
    path = do_mount(args.name)
    print(f"Mounted at {path}")


def cmd_umount(args: argparse.Namespace) -> None:
    do_umount(args.name)
    print(f"Unmounted '{args.name}'")


def cmd_list(args: argparse.Namespace) -> None:
    systems = all_systems()
    if not systems:
        print("No systems.")
        return
    for meta in systems:
        name = meta["name"]
        t = meta.get("type", "?")
        d = sys_dir(name)
        size_str = ""
        if meta.get("disk"):
            p = d / meta["disk"]
            if p.exists():
                size_str = human(p.stat().st_size)
        elif meta.get("base"):
            p = d / meta["base"]
            if p.exists():
                size_str = human(p.stat().st_size)
        mounted = " 🔗" if is_mounted(MOUNTS / name) else ""
        extra = ""
        if meta.get("encrypted"):
            extra += " [enc]"
        if meta.get("frugal"):
            extra += " [frugal]"
        print(f"  {name:<20} {t:<10} {size_str:>8}{mounted}{extra}")


def cmd_layouts(args: argparse.Namespace) -> None:
    """Show partition layout for a bootable image."""
    from binsys._boot import layout_of
    parts = layout_of(args.name)
    if not parts:
        print(f"No partition layout for '{args.name}'")
        return
    print(f"Partition layout for '{args.name}':")
    for i, p in enumerate(parts, 1):
        print(f"  {i}. {p.label:<10} {p.size:>8}  {p.fs}")


def cmd_info(args: argparse.Namespace) -> None:
    meta = load_meta(args.name)
    if not meta:
        die(f"'{args.name}' not found")
    d = sys_dir(args.name)
    print(f"Name:      {meta['name']}")
    print(f"Type:      {meta.get('type', '?')}")
    print(f"Created:   {meta.get('created', '?')}")
    print(f"Encrypted: {'Yes' if meta.get('encrypted') else 'No'}")
    print(f"Frugal:    {'Yes' if meta.get('frugal') else 'No'}")
    if meta.get("source"):
        print(f"Source:    {meta['source']}")
    if meta.get("disk"):
        p = d / meta["disk"]
        if p.exists():
            print(f"Image:     {meta['disk']} ({human(p.stat().st_size)})")
    if meta.get("base"):
        p = d / meta["base"]
        if p.exists():
            print(f"Base:      {meta['base']} ({human(p.stat().st_size)})")
    if meta.get("save"):
        p = d / meta["save"]
        if p.exists():
            print(f"Save:      {meta['save']} ({human(p.stat().st_size)})")


def cmd_resize(args: argparse.Namespace) -> None:
    do_resize(args.name, args.size)


def cmd_import(args: argparse.Namespace) -> None:
    do_import(args.src, name=args.name, img_type=args.type)


def cmd_delete(args: argparse.Namespace) -> None:
    if not args.force:
        import sys
        response = input(f"Really delete '{args.name}'? This cannot be undone. [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Deleted cancelled.")
            sys.exit(0)
    do_delete(args.name)


def cmd_clone(args: argparse.Namespace) -> None:
    do_clone(args.src, args.dst or f"{args.src}-copy")


def cmd_rename(args: argparse.Namespace) -> None:
    do_rename(args.old, args.new)


def cmd_export(args: argparse.Namespace) -> None:
    dst, size = do_export(args.name, args.dest)
    print(f"Exported to {dst} ({human(size)})")


def cmd_check(args: argparse.Namespace) -> None:
    do_check(args.name)


def cmd_encrypt(args: argparse.Namespace) -> None:
    do_encrypt(args.name, hash_algo=args.hash_algo)


def cmd_unlock(args: argparse.Namespace) -> None:
    do_unlock(args.name)


def cmd_lock(args: argparse.Namespace) -> None:
    do_lock(args.name)


def cmd_protect(args: argparse.Namespace) -> None:
    do_protect(args.name, password=args.password, keyfile=args.keyfile)


def cmd_unprotect(args: argparse.Namespace) -> None:
    do_unprotect(args.name)


def cmd_auth(args: argparse.Namespace) -> None:
    do_app_unlock(args.name, password=args.password)


def cmd_app_lock(args: argparse.Namespace) -> None:
    do_app_lock(args.name)


def cmd_hash(args: argparse.Namespace) -> None:
    do_hash(args.name, algo=args.algo)


def cmd_frugal(args: argparse.Namespace) -> None:
    name = args.name
    if args.frugal_cmd == "snapshot":
        do_frugal_save_snapshot(name, label=args.label)
    elif args.frugal_cmd == "list":
        snaps = do_frugal_list_snapshots(name)
        if not snaps:
            print("No snapshots.")
        for s in snaps:
            print(f"  {s['name']:<40} {human(s['size']):>8}")
    elif args.frugal_cmd == "rollback":
        do_frugal_rollback(name, args.snap)
    elif args.frugal_cmd == "merge":
        do_frugal_merge(name)


def cmd_wizard(args: argparse.Namespace) -> None:
    if args.list:
        print("Available wizards:")
        for n, d in WIZARD_SCRIPTS:
            print(f"  {n:<20} {d}")
        return
    if not args.name:
        print("Usage: binsys wizard <script>")
        print("Available:", ", ".join(n for n, d in WIZARD_SCRIPTS))
        return
    wizard_path = SCRIPTS_DIR / args.name
    if not wizard_path.exists():
        die(f"wizard '{args.name}' not found")
    sh([str(wizard_path)])


def cmd_iso(args: argparse.Namespace) -> None:
    src = Path(args.source)
    if src.is_dir():
        do_iso_from_dir(str(src), output=args.output, label=args.name, bootable=args.bootable)
    else:
        do_iso_create(args.source, output=args.output)


def cmd_boot(args: argparse.Namespace) -> None:
    build_bootdisk(
        args.name,
        size=args.size,
        esp_size=args.esp_size,
        kernel=args.kernel,
        initrd=args.initrd,
        cmdline=args.cmdline,
        bootloader=args.bootloader,
        auto_esp=args.auto_esp,
    )


# ── argument parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="binsys",
        description="Create and run filesystem images like VMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Supported types:
  ext4     — raw writable image (default)
  overlay  — squashfs base + ext4 save layer (Puppy frugal style)
  squashfs — compressed read-only snapshot (MX snapshot style)
  fat32    — FAT32 image (Ventoy-compatible)
  frugal   — alias for overlay with frugal metadata
""",
    )
    p.add_argument("--version", action="version", version="binsys 1.0.0")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=False)

    # tui (default)
    sub.add_parser("tui", help="Launch the interactive TUI")

    # new
    pn = sub.add_parser("new", help="Create a new filesystem image")
    pn.add_argument("name")
    pn.add_argument("--type", "-t", default="ext4", choices=TYPES,
                    help="Image type (default: ext4)")
    pn.add_argument("--size", "-s", default="1G",
                    help="Image size (e.g. 2G, 512M) or preset (nano, mini, small, ...)")
    pn.add_argument("--label", "-l", help="Filesystem label")
    pn.add_argument("--distro", "-d",
                    help="Pre-populate from a known distro (ubuntu, debian, arch, ...)")
    pn.add_argument("--encrypt", "-e", action="store_true",
                    help="Encrypt the image with LUKS2 after creation")
    pn.add_argument("--boot", action="store_true", help="Make the image bootable (GPT)")
    pn.add_argument("--bootloader", action="store_true",
                    help="Install puppyboot UEFI bootloader")
    pn.add_argument("--auto-esp", action="store_true",
                    help="Automatically size the ESP partition")
    pn.add_argument("--save-size", help="Save layer size (overlay/frugal only)")

    # snap
    ps = sub.add_parser("snap", help="Snapshot an overlay save layer")
    ps.add_argument("name")

    # run
    pr = sub.add_parser("run", help="Boot a system in QEMU")
    pr.add_argument("name")
    pr.add_argument("--no-kvm", action="store_true", help="Disable KVM acceleration")
    pr.add_argument("--gdb", "-g", action="store_true", help="Wait for GDB connection")
    pr.add_argument("--boot", help="Boot device (e.g. d for CD-ROM)")
    pr.add_argument("--memory", "-m", default="2048", help="RAM in MB (default: 2048)")

    # shell
    psh = sub.add_parser("shell", help="Open a shell in the system directory")
    psh.add_argument("name")

    # mount / umount
    pm = sub.add_parser("mount", help="Mount a system's image")
    pm.add_argument("name")
    pu = sub.add_parser("umount", help="Unmount a system's image")
    pu.add_argument("name")

    # list
    sub.add_parser("list", aliases=["ls"], help="List all systems")

    # layouts
    play = sub.add_parser("layouts", help="Show partition layout")
    play.add_argument("name")

    # info
    pi = sub.add_parser("info", help="Show detailed system info")
    pi.add_argument("name")

    # resize
    prz = sub.add_parser("resize", help="Resize a system's image")
    prz.add_argument("name")
    prz.add_argument("size", help="New size (e.g. 4G)")

    # import
    pim = sub.add_parser("import", help="Import an existing disk image")
    pim.add_argument("src")
    pim.add_argument("name", nargs="?", help="System name (default: from filename)")
    pim.add_argument("--type", "-t", default="ext4", choices=TYPES)

    # delete
    pd = sub.add_parser("delete", help="Delete a system (irreversible)")
    pd.add_argument("name")
    pd.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    # clone
    pcl = sub.add_parser("clone", help="Clone a system")
    pcl.add_argument("src")
    pcl.add_argument("dst", nargs="?", help="Destination name (default: <src>-copy)")

    # rename
    prn = sub.add_parser("rename", help="Rename a system")
    prn.add_argument("old")
    prn.add_argument("new")

    # export
    pex = sub.add_parser("export", help="Export the primary image file to a path")
    pex.add_argument("name")
    pex.add_argument("dest", nargs="?", help="Destination file or directory (default: current dir)")

    # encrypt
    pe = sub.add_parser("encrypt", help="Encrypt a disk image with LUKS2")
    pe.add_argument("name")
    pe.add_argument("--hash", dest="hash_algo", default="sha256",
                    choices=["sha256", "sha512", "sha384", "sha1", "ripemd160"],
                    help="Hash algorithm for LUKS key derivation (default: sha256)")

    # unlock / lock
    pu = sub.add_parser("unlock", help="Open a LUKS-encrypted image")
    pu.add_argument("name")
    plk = sub.add_parser("lock", help="Close a LUKS-encrypted image")
    plk.add_argument("name")

    # protect / unprotect / auth / app-lock
    pp = sub.add_parser("protect", help="Set app-level password+keyfile protection")
    pp.add_argument("name")
    pp.add_argument("--password", help="App password (omit for interactive)")
    pp.add_argument("--keyfile", help="Path to a keyfile to require alongside password")
    pup = sub.add_parser("unprotect", help="Remove app-level protection")
    pup.add_argument("name")
    pa = sub.add_parser("auth", help="Authenticate to unlock a protected system")
    pa.add_argument("name")
    pa.add_argument("--password", help="App password (omit for interactive)")
    pal = sub.add_parser("app-lock", help="Re-lock a protected system in the current session")
    pal.add_argument("name")

    # hash
    ph = sub.add_parser("hash", help="Compute/verify checksum of a system image")
    ph.add_argument("name")
    ph.add_argument("--algo", default="sha256",
                    choices=["sha256", "sha512", "sha384", "sha1", "md5"],
                    help="Hash algorithm (default: sha256)")

    # frugal
    pf = sub.add_parser("frugal", help="Manage frugal (overlay) systems")
    pf_sub = pf.add_subparsers(dest="frugal_cmd", metavar="FRUGAL_CMD", required=True)
    pf_snap = pf_sub.add_parser("snapshot", help="Snapshot the save layer")
    pf_snap.add_argument("name")
    pf_snap.add_argument("--label", "-l", help="Optional label for the snapshot")
    pf_list = pf_sub.add_parser("list", help="List save snapshots")
    pf_list.add_argument("name")
    pf_roll = pf_sub.add_parser("rollback", help="Restore a save snapshot")
    pf_roll.add_argument("name")
    pf_roll.add_argument("snap", help="Snapshot file name (from 'binsys frugal list')")
    pf_merge = pf_sub.add_parser("merge", help="Merge save layer into base.sfs")
    pf_merge.add_argument("name")

    # wizard
    pw = sub.add_parser("wizard", help="Launch a guided automation script",
        description="Available wizards:\n" + "\n".join(f"  {n:<20} {d}" for n, d in WIZARD_SCRIPTS))
    pw.add_argument("name", nargs="?", metavar="SCRIPT",
                    help=f"wizard name ({', '.join(n for n, d in WIZARD_SCRIPTS)})")
    pw.add_argument("--list", action="store_true", help="List available wizards")

    # iso
    piso = sub.add_parser("iso", help="Create an ISO9660 image from a system or directory")
    piso.add_argument("source", help="System name or directory path")
    piso.add_argument("--output", "-o", metavar="PATH", help="Output ISO path")
    piso.add_argument("--name", "-n", help="Volume label (for directory source)")
    piso.add_argument("--bootable", action="store_true", help="Create a bootable ISO (El-Torito)")

    # check
    pck = sub.add_parser("check", help="Run a filesystem integrity check")
    pck.add_argument("name")

    # boot
    pb = sub.add_parser("boot", help="Build a bootable disk image with GPT partitions")
    pb.add_argument("name")
    pb.add_argument("--size", default="4G", help="Total disk size (default: 4G)")
    pb.add_argument("--esp-size", default="512M", help="ESP partition size (default: 512M)")
    pb.add_argument("--kernel", help="Path to a Linux kernel vmlinuz to install")
    pb.add_argument("--initrd", help="Path to an initramfs to install")
    pb.add_argument("--cmdline", help="Kernel command line")
    pb.add_argument("--bootloader", action="store_true", help="Install puppyboot UEFI bootloader")
    pb.add_argument("--auto-esp", action="store_true", help="Auto-size the ESP")

    return p


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = build_parser()

    if _HAS_ARGCOMPLETE:
        argcomplete.autocomplete(p)

    args = p.parse_args(argv)
    
    # Check for missing system dependencies
    if args.command and args.command != "tui":
        check_dependencies_or_warn()

    if args.verbose:
        logging.getLogger("binsys").setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if args.command is None or args.command == "tui":
        from binsys._tui import cmd_tui
        cmd_tui()
        return

    try:
        cmd_map: dict[str, Any] = {
            "new": cmd_new,
            "snap": cmd_snap,
            "run": cmd_run,
            "shell": cmd_shell,
            "mount": cmd_mount,
            "umount": cmd_umount,
            "list": cmd_list,
            "ls": cmd_list,
            "layouts": cmd_layouts,
            "info": cmd_info,
            "resize": cmd_resize,
            "import": cmd_import,
            "delete": cmd_delete,
            "clone": cmd_clone,
            "rename": cmd_rename,
            "export": cmd_export,
            "check": cmd_check,
            "encrypt": cmd_encrypt,
            "unlock": cmd_unlock,
            "lock": cmd_lock,
            "protect": cmd_protect,
            "unprotect": cmd_unprotect,
            "auth": cmd_auth,
            "app-lock": cmd_app_lock,
            "hash": cmd_hash,
            "frugal": cmd_frugal,
            "iso": cmd_iso,
            "wizard": cmd_wizard,
            "boot": cmd_boot,
        }
        handler = cmd_map[args.command]
        handler(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
