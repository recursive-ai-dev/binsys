# PuppyBoot V1 — Build & Install

PuppyBoot is a UEFI boot manager for the AIOS project. It boots, in order of
preference, modern Linux kernels via their EFI stub, legacy Linux kernels via
the EFI handover protocol, AIOS native ELF kernels via a LOADER_PARAMS
handover, and arbitrary EFI binaries via chainload.

It compiles to a single `BOOTX64.EFI` and depends on exactly one external
crate (`uefi` 0.28) plus `log`.

---

## 1. Toolchain

```sh
# Rust (stable) + the UEFI target
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup target add x86_64-unknown-uefi
```

No C compiler, linker script, or custom target JSON is required: the
`x86_64-unknown-uefi` target emits a PE32+ image with the
`EFI Application` subsystem directly.

---

## 2. Build

```sh
cd puppyboot
cargo build --release --target x86_64-unknown-uefi
```

Output:

```
target/x86_64-unknown-uefi/release/puppyboot.efi
```

A correct build is a `PE32+ executable (EFI application) x86-64` of roughly
130 KB and produces **zero warnings**.

---

## 3. Install onto an ESP

PuppyBoot runs as the firmware's default loader. Lay the ESP out like this
(the `esp_layout/` directory in this repo mirrors it):

```
<ESP>/
├── EFI/
│   ├── BOOT/
│   │   └── BOOTX64.EFI            ← puppyboot.efi, renamed
│   ├── puppyboot/
│   │   ├── loader.conf            ← global menu config
│   │   └── entries/
│   │       ├── arch.conf          ← one file per boot target
│   │       ├── windows.conf
│   │       └── aios.conf
│   └── arch/                      ← kernel + initrds (for the stub path)
│       ├── vmlinuz-linux
│       ├── initramfs-linux.img
│       └── intel-ucode.img
```

Assuming the ESP is mounted at `/mnt/esp`:

```sh
install -Dm644 target/x86_64-unknown-uefi/release/puppyboot.efi \
        /mnt/esp/EFI/BOOT/BOOTX64.EFI
cp -r esp_layout/EFI/puppyboot /mnt/esp/EFI/
# then copy your kernel + initramfs into /mnt/esp/EFI/arch/ and edit
# entries/arch.conf so root=PARTUUID=... matches your install.
```

To register it as a named UEFI boot option instead of relying on the
removable-media fallback path:

```sh
efibootmgr --create --disk /dev/sdX --part 1 \
           --loader '\EFI\BOOT\BOOTX64.EFI' --label 'PuppyBoot'
```

### The one value you must edit

In `entries/arch.conf`, replace the placeholder in:

```
cmdline = root=PARTUUID=REPLACE-WITH-ROOT-PARTUUID rw quiet loglevel=3
```

Find your root partition's PARTUUID with `lsblk -o NAME,PARTUUID` (or use
`root=UUID=...` from `blkid`).

---

## 4. Test under QEMU (OVMF)

```sh
# Debian/Ubuntu: apt install ovmf qemu-system-x86
# Arch:          pacman -S edk2-ovmf qemu-full

# Build a throwaway FAT image as the ESP:
mkdir -p esproot/EFI/BOOT
cp target/x86_64-unknown-uefi/release/puppyboot.efi esproot/EFI/BOOT/BOOTX64.EFI
cp -r esp_layout/EFI/puppyboot esproot/EFI/
# (drop a vmlinuz/initramfs into esproot/EFI/arch/ to exercise a real boot)

qemu-system-x86_64 \
    -machine q35,accel=kvm:tcg \
    -m 2048 \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd \
    -drive if=pflash,format=raw,file=OVMF_VARS.copy.fd \
    -drive format=raw,file=fat:rw:esproot \
    -serial stdio
```

`-machine accel=kvm:tcg` uses KVM when available and falls back to pure
software emulation otherwise. Copy the OVMF vars file first so the original
stays pristine:

```sh
cp /usr/share/OVMF/OVMF_VARS.fd OVMF_VARS.copy.fd
```

For a BIOS/CSM machine there is no PE/COFF EFI execution; PuppyBoot is a UEFI
loader and expects a UEFI (or OVMF) firmware environment.

---

## 5. Menu key bindings

| Key            | Action                                   |
|----------------|------------------------------------------|
| ↑ / ↓          | Move selection                           |
| Enter          | Boot the highlighted entry               |
| 1 – 9          | Boot that entry directly                 |
| E              | Edit the kernel command line, then boot  |
| H              | Toggle the help footer                   |
| F12            | Reveal hidden entries                    |
| F10 / Esc      | Reboot (cold)                            |
| F11            | Shut down                                |

The countdown auto-boots `default` when it reaches zero; any keypress
cancels it.

---

## 6. Boot-path reference

| `type =`         | What PuppyBoot does                                              | ExitBootServices called by |
|------------------|-----------------------------------------------------------------|----------------------------|
| `linux-stub`     | LoadImage(vmlinuz) + `initrd=` LoadOptions                      | the kernel's EFI stub      |
| `linux-handover` | allocate boot_params, set `EL64`, jump to handover_offset      | the kernel                 |
| `aios`           | parse ELF64, build LOADER_PARAMS, jump with RDI=&params        | PuppyBoot                  |
| `chain`          | LoadImage/StartImage an arbitrary EFI binary                   | the target                 |

Prefer `linux-stub` for any kernel ≥ 5.8. It is the most robust path because
the kernel constructs its own boot parameters and memory map.
