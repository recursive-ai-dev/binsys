#!/bin/sh
#═══════════════════════════════════════════════════════════════════════════════
#  pack-initrd.sh — FrugalOS initrd.gz Builder
#
#  Produces a cpio newc archive containing the absolute minimum needed to:
#    - mount proc/sys/dev
#    - find and mount the boot device
#    - loop-mount N squashfs layers
#    - build and mount the overlay
#    - pivot_root → exec /sbin/init
#
#  The initrd is NOT a general-purpose environment. It lives in RAM,
#  runs for ~1 second, and hands off to the real root. Every byte counts.
#
#  Output: ./output/initrd.gz
#
#  Requirements (host):
#    busybox static binary   — provide via BUSYBOX_BIN or auto-located
#    musl dynamic linker     — provide via MUSL_BIN or auto-located
#    cpio, gzip, find        — standard on any Linux host
#    our /init script        — must exist at INIT_SCRIPT path
#
#  Usage:
#    ./pack-initrd.sh [options]
#
#  Options:
#    --busybox  <path>   path to static busybox binary
#    --musl     <path>   path to ld-musl-x86_64.so.1
#    --init     <path>   path to FrugalOS /init script
#    --out      <path>   output initrd.gz path
#    --keep             keep staging directory after build (for inspection)
#    --list             list cpio archive contents after build
#═══════════════════════════════════════════════════════════════════════════════

set -e
PATH=/usr/bin:/usr/sbin:/bin:/sbin
export PATH

#───────────────────────────────────────────────────────────────────────────────
# DEFAULTS
#───────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-locate from build-base-sfs output if not specified
BUSYBOX_BIN="${BUSYBOX_BIN:-}"
MUSL_BIN="${MUSL_BIN:-}"
INIT_SCRIPT="${INIT_SCRIPT:-${SCRIPT_DIR}/init}"

OUTPUT_DIR="${SCRIPT_DIR}/output"
OUTPUT_INITRD="${OUTPUT_DIR}/initrd.gz"

STAGING_DIR="${SCRIPT_DIR}/build/initrd-stage"
KEEP_STAGING=0
LIST_CONTENTS=0

#───────────────────────────────────────────────────────────────────────────────
# LOGGING
#───────────────────────────────────────────────────────────────────────────────

log()  { printf '\033[1;32m=>\033[0m %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
step() { printf '\n\033[1;37m── %s\033[0m\n' "$*"; }

#───────────────────────────────────────────────────────────────────────────────
# ARG PARSING
#───────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF

Usage: pack-initrd.sh [options]

Options:
  --busybox <path>   static busybox binary (default: auto-locate)
  --musl    <path>   ld-musl-x86_64.so.1   (default: auto-locate)
  --init    <path>   FrugalOS /init script  (default: ./init)
  --out     <path>   output initrd.gz       (default: ./output/initrd.gz)
  --keep             keep staging dir after build
  --list             list archive contents after build

Environment overrides:
  BUSYBOX_BIN   MUSL_BIN   INIT_SCRIPT

EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --busybox) BUSYBOX_BIN="$2"; shift 2 ;;
        --musl)    MUSL_BIN="$2";    shift 2 ;;
        --init)    INIT_SCRIPT="$2"; shift 2 ;;
        --out)     OUTPUT_INITRD="$2"; shift 2 ;;
        --keep)    KEEP_STAGING=1; shift ;;
        --list)    LIST_CONTENTS=1; shift ;;
        --help|-h) usage ;;
        *) die "unknown option: $1" ;;
    esac
done

#───────────────────────────────────────────────────────────────────────────────
# AUTO-LOCATE BINARIES
# Look for busybox static in common places: build chroot, host system, PATH
#───────────────────────────────────────────────────────────────────────────────

_locate_busybox() {
    [ -n "$BUSYBOX_BIN" ] && [ -f "$BUSYBOX_BIN" ] && return 0

    local chroot_bb="${SCRIPT_DIR}/build/alpine-chroot/bin/busybox.static"
    local chroot_bb2="${SCRIPT_DIR}/build/alpine-chroot/bin/busybox"

    for candidate in \
        "$chroot_bb" "$chroot_bb2" \
        /bin/busybox /usr/bin/busybox \
        "$(command -v busybox 2>/dev/null || true)"
    do
        [ -f "$candidate" ] || continue
        # Verify it's actually static
        if file "$candidate" 2>/dev/null | grep -q "statically linked"; then
            BUSYBOX_BIN="$candidate"
            info "auto-located busybox (static): $BUSYBOX_BIN"
            return 0
        fi
    done

    return 1
}

_locate_musl() {
    [ -n "$MUSL_BIN" ] && [ -f "$MUSL_BIN" ] && return 0

    local chroot_musl
    chroot_musl=$(find "${SCRIPT_DIR}/build/alpine-chroot/lib" \
        -name "ld-musl-x86_64.so.1" 2>/dev/null | head -1)

    for candidate in \
        "$chroot_musl" \
        /lib/ld-musl-x86_64.so.1 \
        /usr/lib/x86_64-linux-musl/libc.so \
        "${SCRIPT_DIR}/build/base-stage/lib/ld-musl-x86_64.so.1"
    do
        [ -f "$candidate" ] && {
            MUSL_BIN="$candidate"
            info "auto-located musl: $MUSL_BIN"
            return 0
        }
    done

    return 1
}

#───────────────────────────────────────────────────────────────────────────────
# PREFLIGHT
#───────────────────────────────────────────────────────────────────────────────

preflight() {
    step "Preflight"

    for cmd in cpio gzip find; do
        command -v "$cmd" >/dev/null 2>&1 \
            || die "required tool not found: $cmd"
        ok "$cmd"
    done

    _locate_busybox \
        || die "busybox (static) not found. Use --busybox <path> or run build-base-sfs.sh first"
    ok "busybox: $BUSYBOX_BIN"

    _locate_musl \
        || warn "musl not found — initrd will rely on statically-linked busybox only (acceptable)"
    [ -n "$MUSL_BIN" ] && ok "musl: $MUSL_BIN"

    [ -f "$INIT_SCRIPT" ] \
        || die "/init script not found at $INIT_SCRIPT — use --init <path>"
    [ -x "$INIT_SCRIPT" ] \
        || warn "/init is not executable — will chmod +x in staging"
    ok "init: $INIT_SCRIPT"

    mkdir -p "$OUTPUT_DIR"
}

#───────────────────────────────────────────────────────────────────────────────
# STAGE STRUCTURE
# The initrd cpio is tiny — just what /init needs before pivot_root.
# We do NOT include package managers, compressors, or anything post-pivot.
#───────────────────────────────────────────────────────────────────────────────

# Exact list of busybox applets /init.sh calls or might need for emergency shell.
# Sorted by category. All symlink to /bin/busybox.
_BIN_APPLETS="
ash sh
cat echo printf
ls find
mkdir rm cp mv ln chmod chown
mount umount
grep sed awk sort cut tr head tail wc
mktemp sleep date
id whoami
kill fuser
chroot
"

_SBIN_APPLETS="
modprobe depmod insmod rmmod
losetup
blkid
pivot_root switch_root
ifconfig
halt reboot poweroff
getty
"

build_structure() {
    step "Building initrd staging tree"

    # Clean any previous staging
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"

    local S="$STAGING_DIR"

    # ── directory skeleton ─────────────────────────────────────────────────
    # Ordered so parent dirs exist before children
    for d in \
        bin sbin lib lib64 \
        proc sys \
        dev dev/pts \
        run tmp \
        mnt mnt/boot \
        mnt/layers \
        newroot
    do
        mkdir -p "${S}/${d}"
    done

    chmod 1777 "${S}/tmp"

    log "directory skeleton created"
}

install_busybox() {
    step "Installing busybox"

    local S="$STAGING_DIR"

    cp "$BUSYBOX_BIN" "${S}/bin/busybox"
    chmod 755 "${S}/bin/busybox"

    # ── /bin symlinks ──────────────────────────────────────────────────────
    for applet in $_BIN_APPLETS; do
        [ -z "$applet" ] && continue
        [ -e "${S}/bin/${applet}" ] && continue
        ln -s busybox "${S}/bin/${applet}"
    done

    # ── /sbin symlinks — point to /bin/busybox ────────────────────────────
    for applet in $_SBIN_APPLETS; do
        [ -z "$applet" ] && continue
        [ -e "${S}/sbin/${applet}" ] && continue
        ln -s /bin/busybox "${S}/sbin/${applet}"
    done

    # Verify busybox actually runs in our staging context
    "${S}/bin/busybox" --help >/dev/null 2>&1 \
        || die "busybox sanity check failed — binary may not be executable on this host arch"

    ok "busybox installed with $(ls "${S}/bin" | wc -l) /bin + $(ls "${S}/sbin" | wc -l) /sbin applets"
}

install_musl() {
    step "Installing musl dynamic linker"

    [ -n "$MUSL_BIN" ] || { info "skipping — musl not found"; return 0; }

    local S="$STAGING_DIR"

    cp "$MUSL_BIN" "${S}/lib/ld-musl-x86_64.so.1"
    chmod 755 "${S}/lib/ld-musl-x86_64.so.1"

    # Compatibility symlinks
    ln -sf ld-musl-x86_64.so.1 "${S}/lib/libc.so"
    ln -sf ld-musl-x86_64.so.1 "${S}/lib/libc.musl-x86_64.so.1"
    ln -sf /lib/ld-musl-x86_64.so.1 "${S}/lib/ld-linux-x86-64.so.2"

    # /lib64 → /lib (x86_64 glibc compat path)
    ln -sf /lib "${S}/lib64"

    ok "musl installed: $(du -sh "${S}/lib/ld-musl-x86_64.so.1" | cut -f1)"
}

install_init() {
    step "Installing /init"

    local S="$STAGING_DIR"

    cp "$INIT_SCRIPT" "${S}/init"
    chmod 755 "${S}/init"

    # Validate: must start with a shebang the kernel will honour
    local first_line
    first_line=$(head -1 "${S}/init")
    case "$first_line" in
        '#!/bin/sh'|'#!/bin/ash'|'#!/bin/busybox sh'|'#!/bin/busybox ash') ;;
        *) warn "/init shebang is '$first_line' — kernel expects #!/bin/sh or #!/bin/ash" ;;
    esac

    ok "/init installed ($(wc -l < "${S}/init") lines)"
}

install_dev_nodes() {
    step "Installing static /dev nodes"

    local S="$STAGING_DIR"

    # The kernel will populate /dev via devtmpfs on boot.
    # We only need a handful of nodes for the case where devtmpfs fails
    # or if someone drops to emergency shell before /dev is mounted.
    #
    # mknod requires root — skip gracefully if not root, warn, and continue.
    # devtmpfs should handle it either way.

    if [ "$(id -u)" != "0" ]; then
        warn "not root — skipping static /dev node creation (devtmpfs will handle at boot)"
        return 0
    fi

    mknod -m 660 "${S}/dev/console"  c 5 1  2>/dev/null || true
    mknod -m 660 "${S}/dev/tty"      c 5 0  2>/dev/null || true
    mknod -m 660 "${S}/dev/tty0"     c 4 0  2>/dev/null || true
    mknod -m 660 "${S}/dev/tty1"     c 4 1  2>/dev/null || true
    mknod -m 660 "${S}/dev/ttyS0"    c 4 64 2>/dev/null || true  # serial console
    mknod -m 640 "${S}/dev/mem"      c 1 1  2>/dev/null || true
    mknod -m 666 "${S}/dev/null"     c 1 3  2>/dev/null || true
    mknod -m 666 "${S}/dev/zero"     c 1 5  2>/dev/null || true
    mknod -m 444 "${S}/dev/random"   c 1 8  2>/dev/null || true
    mknod -m 444 "${S}/dev/urandom"  c 1 9  2>/dev/null || true

    ok "static /dev nodes installed"
}

#───────────────────────────────────────────────────────────────────────────────
# AUDIT
# Show what's in the staging dir and rough size before packing
#───────────────────────────────────────────────────────────────────────────────

audit_staging() {
    step "Staging tree audit"

    local S="$STAGING_DIR"

    printf '\n  %-32s %s\n' "Path" "Size"
    printf '  %s\n' "────────────────────────────────────────"

    for d in bin sbin lib init; do
        [ -e "${S}/${d}" ] || continue
        local sz
        sz=$(du -sh "${S}/${d}" 2>/dev/null | cut -f1)
        printf '  %-32s %s\n' "/$d" "$sz"
    done

    printf '\n'

    local total_files
    total_files=$(find "$S" | wc -l)
    local total_real
    total_real=$(du -sh "$S" | cut -f1)
    info "  total: $total_files entries, $total_real uncompressed"
    printf '\n'
}

#───────────────────────────────────────────────────────────────────────────────
# PACK
# cpio newc format is what the Linux kernel expects.
# We sort find output for reproducible archives.
# -R 0:0 sets uid:gid to root:root for all entries.
#───────────────────────────────────────────────────────────────────────────────

pack_initrd() {
    step "Packing initrd.gz"

    local S="$STAGING_DIR"

    # Ensure output directory exists
    mkdir -p "$(dirname "$OUTPUT_INITRD")"

    # Remove any previous output
    [ -f "$OUTPUT_INITRD" ] && rm -f "$OUTPUT_INITRD"

    log "cpio newc | gzip -9 → $OUTPUT_INITRD"

    (
        cd "$S"
        # find . outputs paths relative to S
        # sort for reproducible ordering
        # cpio -R 0:0 forces root ownership
        # gzip -9 for maximum compression (initrd lives in RAM, smaller = better)
        find . | sort | cpio --quiet -R 0:0 -H newc -o | gzip -9 > "$OUTPUT_INITRD"
    )

    [ -f "$OUTPUT_INITRD" ] || die "cpio/gzip produced no output"

    local compressed_size
    compressed_size=$(du -sh "$OUTPUT_INITRD" | cut -f1)

    ok "initrd.gz: $OUTPUT_INITRD ($compressed_size)"

    # Cross-check: uncompressed size for reference
    local uncompressed_size
    uncompressed_size=$(gzip -l "$OUTPUT_INITRD" 2>/dev/null | awk 'NR==2 {printf "%.1fM", $2/1048576}' || echo "?")
    info "  uncompressed: ${uncompressed_size} (this is what loads into RAM)"
}

list_contents() {
    step "Archive contents"
    zcat "$OUTPUT_INITRD" | cpio --quiet -t | sort | while read -r entry; do
        printf '  %s\n' "$entry"
    done
    printf '\n'
    info "total entries: $(zcat "$OUTPUT_INITRD" | cpio --quiet -t | wc -l)"
}

#───────────────────────────────────────────────────────────────────────────────
# BOOTLOADER HINT
# Print a ready-to-paste GRUB stanza for the generated artifacts
#───────────────────────────────────────────────────────────────────────────────

print_grub_hint() {
    step "Bootloader configuration (GRUB example)"

    cat <<EOF

  # /boot/grub/grub.cfg — paste this entry:

  menuentry "FrugalOS" {
      set root=(hd0,1)           # adjust to your boot partition
      linux  /boot/vmlinuz       \\
             frugal.label=FRUGALOS \\
             frugal.save=save      \\
             quiet
      initrd /boot/initrd.gz
  }

  # For a USB stick (GRUB installed on USB, booted on any machine):
  menuentry "FrugalOS (USB)" {
      search --set=root --label FRUGALOS
      linux  /boot/vmlinuz       \\
             frugal.label=FRUGALOS \\
             frugal.save=save      \\
             console=tty1 console=ttyS0,115200n8
      initrd /boot/initrd.gz
  }

  # Expected boot device layout:
  #   /boot/vmlinuz         — kernel
  #   /boot/initrd.gz       — this file
  #   /os/base.sfs          — base layer
  #   /plugins/enabled/     — symlinks to enabled plugins
  #   /plugins/available/   — all available .sfs files
  #   /save/upper/          — persistent writable layer
  #   /save/work/           — overlayfs workdir

EOF
}

#───────────────────────────────────────────────────────────────────────────────
# CLEANUP
#───────────────────────────────────────────────────────────────────────────────

cleanup() {
    if [ "$KEEP_STAGING" = "1" ]; then
        log "staging preserved: $STAGING_DIR"
    else
        rm -rf "$STAGING_DIR"
    fi
}

trap cleanup EXIT

#───────────────────────────────────────────────────────────────────────────────
# MAIN
#───────────────────────────────────────────────────────────────────────────────

printf '\n'
printf '\033[1;37m═══════════════════════════════════════════════\033[0m\n'
printf '\033[1;37m  FrugalOS initrd Packer\033[0m\n'
printf '\033[1;37m═══════════════════════════════════════════════\033[0m\n\n'

preflight
build_structure
install_busybox
install_musl
install_init
install_dev_nodes
audit_staging
pack_initrd

[ "$LIST_CONTENTS" = "1" ] && list_contents

print_grub_hint

printf '\033[1;32m  Done.\033[0m  All three artifacts:\n\n'
printf '    vmlinuz     →  arch/x86/boot/bzImage  (from kernel build)\n'
printf '    initrd.gz   →  %s\n' "$OUTPUT_INITRD"
printf '    base.sfs    →  output/base.sfs        (from build-base-sfs.sh)\n'
printf '\n'
