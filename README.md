# binsys — Filesystem Images, Meet VMs

**binsys** creates and runs filesystem images like virtual machines. Think
"container ergonomics with VM isolation" — a single CLI to spin up ext4,
squashfs, FAT32, or Puppy-style frugal overlay images and boot them in QEMU.

```
$ binsys new mybox --type ext4 --size 4G
$ binsys run mybox
```

## Features

- **Four image types** — ext4, squashfs, FAT32, and **frugal overlay** (base.sfs + save.img)
- **Frugal overlay** — Puppy Linux–style: read-only squashfs base + writable ext4 save layer with snapshot, rollback, and merge
- **Bootable disk builder** — GPT partitioning with ESP, puppyboot UEFI bootloader, kernel + initrd installation
- **QEMU runner** — automatic KVM detection, UEFI (OVMF) support, serial console
- **LUKS2 encryption** for save layers (frugal) or full images
- **App-level protection** — password + keyfile locking per system
- **Snapshot manager** — take, list, and roll back snapshots of overlay save layers
- **ISO builder** — create ISO9660 images from systems or directories
- **TUI / CLI** — rich interactive TUI or headless command-line
- **Wizard scripts** — guided builders for frugal systems, ISOs, and VMs
- **Plugin system** (FrugalOS) — hot-loadable squashfs plugins with dependency resolution

## Quick start

```bash
# Install
pip install binsys    # or: python3 -m pip install .

# Create an ext4 system from a distro cloud image
binsys new ubuntu-box --type ext4 --distro ubuntu --size 8G

# List systems
binsys list

# Boot it
binsys run ubuntu-box --memory 4096
```

## Image types

| Type | Description | Use case |
|------|-------------|----------|
| `ext4` | Raw writable ext4 image | General-purpose VMs, distro snapshots |
| `squashfs` | Compressed read-only image | MX Linux–style snapshots, distro archives |
| `fat32` | FAT32 image | Ventoy‑compatible USB images |
| `frugal` / `overlay` | squashfs base + ext4 save | Puppy‑style frugal installs, FrugalOS |

## Frugal overlay management

```bash
# Create a frugal system
binsys new myfrugal --type frugal --size 2G --save-size 512M

# Convert an existing ext4 system to frugal (in-place)
binsys frugal convert mybox

# Take a snapshot of the save layer
binsys frugal snapshot myfrugal --label "before-update"

# List snapshots
binsys frugal list myfrugal

# Roll back to a snapshot
binsys frugal rollback myfrugal save_20260624_120000_before-update.img

# Merge the save layer into base.sfs (flatten overlay, reset save)
binsys frugal merge myfrugal

# Run in QEMU
binsys run myfrugal
```

## Bootable disk builder

```bash
binsys boot my-boot-disk --size 4G \
    --kernel /boot/vmlinuz \
    --initrd /boot/initrd.img \
    --bootloader
```

Creates a GPT disk with:
- Partition 1: **ESP** (FAT32, 512M default) — contains BOOTX64.EFI (puppyboot) + kernel + initrd
- Partition 2: **rootfs** (ext4) — for your root filesystem

## Encryption & protection

```bash
# LUKS2-encrypt an image
binsys encrypt securebox

# App-level password + keyfile protection
binsys protect securebox --password
binsys auth securebox --password    # unlock for the session
binsys app-lock securebox           # re-lock
```


## Commands

```
new          Create a new filesystem image
run          Boot a system in QEMU
list / ls    List all systems
info         Show detailed system info
mount        Mount a system's image
umount       Unmount a system
snap         Snapshot an overlay save layer
resize       Resize a system's image
clone        Clone a system
rename       Rename a system
delete       Delete a system (irreversible)
export       Export the primary image
import       Import an existing disk image
check        Filesystem integrity check
encrypt      LUKS2-encrypt a disk image
unlock / lock   LUKS2 open / close
protect / unprotect   App-level protection
auth         Authenticate to unlock a protected system
app-lock     Re-lock a protected system
hash         Compute / verify checksum
frugal       Frugal overlay sub-commands (snapshot, list, rollback, merge)
iso          Create an ISO9660 image
boot         Build a bootable disk image with GPT partitions
wizard       Launch a guided automation script
tui          Launch the interactive TUI
```


---

# FrugalOS — The custom-frugal Subsystem

[`custom-frugal/`](./custom-frugal) contains a self-contained **FrugalOS** — a
tiny, musl+BusyBox initrd + squashfs base that boots into a plugin-based
overlay root filesystem. It fits in ~2.6 MB of compressed artifacts.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Boot device (partition labeled FRUGALOS or by UUID)      │
│                                                           │
│  /os/base.sfs           — read-only squashfs base layer    │
│  /plugins/available/    — all .sfs plugin packages         │
│  /plugins/enabled/      — symlinks to boot-enabled plugins │
│  /save/upper/           — overlayfs upper dir (writable)   │
│  /save/work/            — overlayfs work dir               │
│                                                           │
│  /boot/vmlinuz          — Linux kernel                     │
│  /boot/initrd.gz        — FrugalOS initramfs               │
└───────────────────────────────────────────────────────────┘
```

### Layers

| Priority | Layer | Source | Contents |
|----------|-------|--------|----------|
| 0 (base) | Lower | `base.sfs` | musl libc, BusyBox, plugin‑ctl, init scripts |
| 1–99 | Mid | `plugins/` | Hot‑loadable .sfs plugin packages |
| ∞ | Upper | `save/upper/` | Persistent writable overlay (ext4 on save.img) |

### Initrd boot flow (`/init`)

1. **Stage 0** — mount proc, sys, dev, run; verify kernel features (overlayfs, squashfs)
2. **Stage 1** — parse `frugal.dev=`, `frugal.uuid=`, `frugal.label=` from cmdline; locate & mount boot device
3. **Stage 2** — discover `.sfs` files in `plugins/enabled/`, parse manifests, sort by LOAD_ORDER
4. **Stage 3** — loop-mount each .sfs; record loop devices for plugin‑ctl hot‑load/unload
5. **Stage 4** — build lowerdir string (base + plugins in order), mount OverlayFS → `/newroot`
6. **Stage 5** — write runtime state to `/run/plugin-state/`, move mounts, `pivot_root`, `exec /sbin/init`

### Plugin system (`plugin-ctl`)

```bash
# List plugins
plugin-ctl list
plugin-ctl list active
plugin-ctl list boot

# Enable / disable at boot
plugin-ctl enable my-plugin
plugin-ctl disable my-plugin

# Hot-load / hot-unload (LIFO — must unload in reverse order)
plugin-ctl load my-plugin
plugin-ctl unload my-plugin

# Build a plugin from a staging directory
plugin-ctl build ./my-plugin-dir my-plugin 1.0.0

# Dependency check
plugin-ctl check my-plugin

# Dev cycle
plugin-ctl reload my-plugin
```

Each plugin is a squashfs archive containing:
- `.plugin/manifest` — metadata (NAME, VERSION, DEPENDS, CONFLICTS, LOAD_ORDER, PROVIDES)
- `.plugin/on_load` — executable hook called after hot-load
- `.plugin/on_unload` — executable hook called before hot-unload
- Standard FHS directories (`usr/bin/`, `usr/lib/`, `etc/`, etc.)

### Build artifacts

| File | Size | Description | Builder script |
|------|------|-------------|----------------|
| `output/base.sfs` | ~1.6 MB | SquashFS base layer (musl + BusyBox + plugin-ctl) | `build-base-sfs.sh` |
| `output/initrd.gz` | ~1.0 MB | Initramfs with /init, BusyBox, musl | `pack-initrd.sh` |
| `boot-disk/disk.img` | 1 GB | Bootable GPT disk with ESP + data partition | manual (see below) |

### Build from source

```bash
cd custom-frugal

# 1. Build base.sfs (requires root for chroot + mount)
sudo bash build-base-sfs.sh

# 2. Build initrd.gz (requires root for cpio ownership)
sudo bash pack-initrd.sh --list

# 3. Build bootable disk image (optional)
sudo bash -c '
  BOOT_DIR=boot-disk; mkdir -p "$BOOT_DIR"
  truncate -s 1G "$BOOT_DIR/disk.img"
  sgdisk -Z "$BOOT_DIR/disk.img"
  sgdisk -n 1:2048:+256M -t 1:ef00 -c 1:FRUGAL-ESP \
          -n 2:0:0    -t 2:8300 -c 2:FRUGAL-DATA \
          "$BOOT_DIR/disk.img"
  LODEV=$(losetup --find --show -P "$BOOT_DIR/disk.img")
  mkfs.fat -F32 -n FRUGAL_ESP "${LODEV}p1"
  mkfs.ext4 -F -L FRUGALOS   "${LODEV}p2"
  mount "${LODEV}p1" /mnt/frugal-boot-esp
  mount "${LODEV}p2" /mnt/frugal-boot-root
  cp output/initrd.gz /mnt/frugal-boot-esp/
  cp /boot/vmlinuz    /mnt/frugal-boot-esp/
  mkdir -p /mnt/frugal-boot-root/os
  cp output/base.sfs  /mnt/frugal-boot-root/os/
  mkdir -p /mnt/frugal-boot-root/{plugins/{enabled,available},save/{upper,work}}
  umount /mnt/frugal-boot-esp /mnt/frugal-boot-root
  losetup -d "$LODEV"
'

# 4. Run in QEMU
qemu-system-x86_64 -m 1G \
    -drive file=boot-disk/disk.img,format=raw,if=virtio \
    -nic user \
    -serial stdio
```

### Kernel command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `frugal.dev=` | *(auto)* | Explicit block device (e.g. `/dev/sda2`) |
| `frugal.uuid=` | *(auto)* | Find boot device by UUID |
| `frugal.label=` | `FRUGALOS` | Find boot device by filesystem label |
| `frugal.save=` | `save` | Relative path to save directory on boot device |
| `frugal.init=` | `/sbin/init` | Init binary in new root |
| `frugal.debug=1` | off | Enable verbose initrd logging |

## License

MIT — see [LICENSE.md](./LICENSE.md).

