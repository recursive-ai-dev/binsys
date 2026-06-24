#!/bin/sh
#═══════════════════════════════════════════════════════════════════════════════
#  build-base-sfs.sh — FrugalOS base.sfs Builder
#  Run on any x86_64 Linux host with: root, wget/curl, chroot, mksquashfs
#
#  What this builds:
#    musl libc         — C runtime, dynamic linker
#    busybox (static)  — entire POSIX userland in one binary
#    squashfs-tools    — mksquashfs / unsquashfs (for plugin-ctl build)
#    frugalos scripts  — plugin-ctl, manifest-new, init scripts
#    FHS skeleton      — /proc /sys /dev /run /tmp /mnt /etc /root ...
#    busybox init      — inittab-based process manager
#
#  Output: ./output/base.sfs  (~6-12 MB with XZ compression)
#
#  Alpine Linux minirootfs is used as the build chroot because it is
#  musl-native and apk gives us exactly the packages we need with zero
#  glibc contamination. Nothing from Alpine ends up in base.sfs except
#  the binaries we explicitly select.
#═══════════════════════════════════════════════════════════════════════════════

set -e
PATH=/usr/bin:/usr/sbin:/bin:/sbin
export PATH

#───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
#───────────────────────────────────────────────────────────────────────────────

ALPINE_VERSION="3.19"
ALPINE_ARCH="x86_64"
ALPINE_MINIROOTFS="alpine-minirootfs-${ALPINE_VERSION}.0-${ALPINE_ARCH}.tar.gz"
ALPINE_URL="https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION}/releases/${ALPINE_ARCH}/${ALPINE_MINIROOTFS}"

BUILD_DIR="$(pwd)/build"
CHROOT_DIR="${BUILD_DIR}/alpine-chroot"
STAGE_DIR="${BUILD_DIR}/base-stage"
OUTPUT_DIR="$(pwd)/output"
OUTPUT_SFS="${OUTPUT_DIR}/base.sfs"

FRUGALOS_VERSION="0.1.0"
KERNEL_VERSION=""   # set to e.g. "6.6.30" to pre-populate /lib/modules structure

# squashfs compression — xz for best ratio, zstd for faster boot
SFS_COMP="xz"
SFS_COMP_FLAGS="-Xbcj x86"

#───────────────────────────────────────────────────────────────────────────────
# LOGGING
#───────────────────────────────────────────────────────────────────────────────

log()  { printf '\033[1;32m=>\033[0m %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;37m── %s\033[0m\n' "$*"; }

#───────────────────────────────────────────────────────────────────────────────
# PREFLIGHT
#───────────────────────────────────────────────────────────────────────────────

preflight() {
    step "Preflight checks"

    [ "$(id -u)" = "0" ] || die "must run as root (needs chroot, mount)"

    for cmd in wget tar chroot mount umount mksquashfs; do
        command -v "$cmd" >/dev/null 2>&1 \
            || die "required tool not found: $cmd"
        log "$cmd found"
    done

    # Verify mksquashfs supports the chosen compressor
    mksquashfs -help 2>&1 | grep -qi "${SFS_COMP}" \
        || die "mksquashfs does not support '${SFS_COMP}' — install squashfs-tools with ${SFS_COMP} support"
}

#───────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP ALPINE CHROOT
#───────────────────────────────────────────────────────────────────────────────

bootstrap_alpine() {
    step "Bootstrapping Alpine ${ALPINE_VERSION} chroot"

    mkdir -p "$CHROOT_DIR" "$STAGE_DIR" "$OUTPUT_DIR"

    if [ ! -f "${BUILD_DIR}/${ALPINE_MINIROOTFS}" ]; then
        log "downloading Alpine minirootfs"
        wget -q --show-progress -O "${BUILD_DIR}/${ALPINE_MINIROOTFS}" "$ALPINE_URL" \
            || die "download failed: $ALPINE_URL"
    else
        log "using cached: ${ALPINE_MINIROOTFS}"
    fi

    log "extracting minirootfs"
    tar -xzf "${BUILD_DIR}/${ALPINE_MINIROOTFS}" -C "$CHROOT_DIR"

    # Copy host DNS so apk can reach mirrors
    cp /etc/resolv.conf "${CHROOT_DIR}/etc/resolv.conf" 2>/dev/null || true

    # Mount pseudo-filesystems into chroot
    mount -t proc  proc  "${CHROOT_DIR}/proc"
    mount --bind   /sys  "${CHROOT_DIR}/sys"
    mount --bind   /dev  "${CHROOT_DIR}/dev"

    log "updating apk index"
    chroot "$CHROOT_DIR" apk update --quiet

    log "installing build packages"
    chroot "$CHROOT_DIR" apk add --quiet --no-progress \
        busybox-static \
        busybox-extras \
        musl \
        squashfs-tools \
        file \
        xz

    log "alpine chroot ready"
}

teardown_alpine() {
    # Always called, even on error
    umount -l "${CHROOT_DIR}/proc" 2>/dev/null || true
    umount -l "${CHROOT_DIR}/sys"  2>/dev/null || true
    umount -l "${CHROOT_DIR}/dev"  2>/dev/null || true
}

#───────────────────────────────────────────────────────────────────────────────
# BUILD FHS SKELETON
# We build the stage directory manually — nothing from Alpine's root goes
# in except the specific binaries we extract below.
#───────────────────────────────────────────────────────────────────────────────

build_fhs() {
    step "Building FHS skeleton"

    # Standard directories
    for d in \
        bin sbin lib lib64 \
        usr/bin usr/sbin usr/lib usr/lib64 usr/libexec usr/share usr/include \
        etc etc/frugalos etc/frugalos/init.d \
        run tmp var/log var/tmp \
        proc sys dev dev/pts \
        mnt mnt/boot \
        root home \
        opt \
        .plugin
    do
        mkdir -p "${STAGE_DIR}/${d}"
    done

    # /var/run and /var/lock → /run (FHS 3.0)
    ln -sf /run     "${STAGE_DIR}/var/run"
    ln -sf /run/lock "${STAGE_DIR}/var/lock" 2>/dev/null || true

    # /lib64 compat symlink (glibc ABI compatibility for any binary that hardcodes it)
    ln -sf /lib     "${STAGE_DIR}/lib64"
    ln -sf /usr/lib "${STAGE_DIR}/usr/lib64"

    # /tmp sticky bit
    chmod 1777 "${STAGE_DIR}/tmp"

    log "FHS skeleton created"
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL BUSYBOX
# We use the static busybox binary from Alpine — statically linked against
# musl so it carries zero runtime deps. We install all applet symlinks.
#───────────────────────────────────────────────────────────────────────────────

install_busybox() {
    step "Installing busybox (static)"

    local bb_src="${CHROOT_DIR}/bin/busybox.static"
    [ -f "$bb_src" ] || bb_src="${CHROOT_DIR}/bin/busybox"
    [ -f "$bb_src" ] || die "busybox binary not found in chroot"

    cp "$bb_src" "${STAGE_DIR}/bin/busybox"
    chmod 755 "${STAGE_DIR}/bin/busybox"

    log "installing applet symlinks"
    # Run busybox --list inside a minimal chroot of our STAGE_DIR
    # We can't just chroot into STAGE_DIR yet (no /proc), so list applets
    # directly from the binary
    "${STAGE_DIR}/bin/busybox" --list 2>/dev/null | while read -r applet; do
        [ -z "$applet" ] && continue
        # Determine bin vs sbin
        case "$applet" in
            # sbin applets — system administration tools
            init|getty|login|syslogd|klogd|mdev|ifconfig|route|\
            insmod|rmmod|modprobe|depmod|lsmod|\
            fdisk|mkfs|fsck|mount|umount|swapon|swapoff|\
            losetup|pivot_root|switch_root|blkid|\
            halt|reboot|poweroff|arp|ip|\
            ifup|ifdown|udhcpc|nameif|vconfig)
                local target="${STAGE_DIR}/sbin/${applet}"
                ;;
            # everything else → /bin
            *)
                local target="${STAGE_DIR}/bin/${applet}"
                ;;
        esac
        # Only create if not already a real file
        [ -e "$target" ] || ln -sf /bin/busybox "$target"
    done

    # Ensure critical symlinks exist regardless of --list output
    for applet in sh ash; do
        ln -sf /bin/busybox "${STAGE_DIR}/bin/${applet}" 2>/dev/null || true
    done

    # /usr/bin → symlinks to /bin for PATH compat
    # Plugins will overmount /usr/bin with their own content; base provides
    # a passthrough so /usr/bin/sh etc. work before any plugin is loaded.
    ln -sf /bin/busybox "${STAGE_DIR}/usr/bin/busybox" 2>/dev/null || true
    ln -sf /bin/sh      "${STAGE_DIR}/usr/bin/sh"      2>/dev/null || true

    log "busybox installed: $(${STAGE_DIR}/bin/busybox --help 2>&1 | head -1)"
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL MUSL LIBC
# musl combines the C library and dynamic linker into one file.
# We install it so dynamically-linked plugin binaries have a runtime.
#───────────────────────────────────────────────────────────────────────────────

install_musl() {
    step "Installing musl libc"

    local musl_src
    # Alpine installs musl's dynamic linker here:
    musl_src=$(find "${CHROOT_DIR}/lib" -name "ld-musl-x86_64.so.1" 2>/dev/null | head -1)
    [ -n "$musl_src" ] || die "ld-musl-x86_64.so.1 not found in chroot"

    cp "$musl_src" "${STAGE_DIR}/lib/ld-musl-x86_64.so.1"
    chmod 755 "${STAGE_DIR}/lib/ld-musl-x86_64.so.1"

    # Standard compat symlinks
    ln -sf ld-musl-x86_64.so.1 "${STAGE_DIR}/lib/libc.so"
    ln -sf ld-musl-x86_64.so.1 "${STAGE_DIR}/lib/libc.musl-x86_64.so.1"

    # glibc ABI compat: many binaries try to dlopen ld-linux-x86-64.so.2
    ln -sf /lib/ld-musl-x86_64.so.1 "${STAGE_DIR}/lib/ld-linux-x86-64.so.2"

    log "musl: $(${CHROOT_DIR}/lib/ld-musl-x86_64.so.1 2>&1 | head -1 || true)"
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL SQUASHFS-TOOLS
# mksquashfs + unsquashfs — needed for plugin-ctl build and verify commands.
# Dynamically linked against musl in Alpine, which is exactly what we want.
#───────────────────────────────────────────────────────────────────────────────

install_squashfs_tools() {
    step "Installing squashfs-tools"

    for bin in mksquashfs unsquashfs; do
        local src="${CHROOT_DIR}/usr/bin/${bin}"
        [ -f "$src" ] || die "$bin not found in chroot"
        cp "$src" "${STAGE_DIR}/usr/bin/${bin}"
        chmod 755 "${STAGE_DIR}/usr/bin/${bin}"
        log "$bin installed"
    done

    # Copy any musl-based shared libs that squashfs-tools links against
    # (typically just libc, lz4, lzo, xz, zstd — find them via ldd equivalent)
    chroot "$CHROOT_DIR" /bin/sh -c \
        "ldd /usr/bin/mksquashfs 2>/dev/null | awk '/=>/ {print \$3}'" \
    | while read -r lib; do
        [ -f "${CHROOT_DIR}${lib}" ] || continue
        local dest_dir="${STAGE_DIR}$(dirname "$lib")"
        mkdir -p "$dest_dir"
        cp -n "${CHROOT_DIR}${lib}" "${dest_dir}/" 2>/dev/null || true
    done

    log "squashfs-tools installed"
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL KERNEL MODULE DIRECTORY STRUCTURE
# The actual .ko files are provided by a kernel-modules plugin.
# base.sfs just provides the skeleton so modprobe doesn't panic.
#───────────────────────────────────────────────────────────────────────────────

install_modules_skeleton() {
    step "Creating kernel modules skeleton"

    if [ -n "$KERNEL_VERSION" ]; then
        mkdir -p "${STAGE_DIR}/lib/modules/${KERNEL_VERSION}"
        # Empty modules.dep so depmod -a has somewhere to write
        touch "${STAGE_DIR}/lib/modules/${KERNEL_VERSION}/modules.dep"
        touch "${STAGE_DIR}/lib/modules/${KERNEL_VERSION}/modules.alias"
        touch "${STAGE_DIR}/lib/modules/${KERNEL_VERSION}/modules.symbols"
        log "created skeleton for kernel $KERNEL_VERSION"
    else
        mkdir -p "${STAGE_DIR}/lib/modules"
        log "KERNEL_VERSION not set — empty /lib/modules/ (modules plugin will populate)"
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL /ETC FILES
#───────────────────────────────────────────────────────────────────────────────

install_etc() {
    step "Populating /etc"

    # ── os-release ─────────────────────────────────────────────────────────
    cat > "${STAGE_DIR}/etc/os-release" <<EOF
NAME="FrugalOS"
VERSION="${FRUGALOS_VERSION}"
ID=frugalos
ID_LIKE=
PRETTY_NAME="FrugalOS ${FRUGALOS_VERSION}"
HOME_URL="https://github.com/frugalos/frugalos"
BUILD_ID=$(date -u +%Y%m%d)
EOF

    # ── fstab ───────────────────────────────────────────────────────────────
    # Intentionally minimal — overlay is handled by /init before pivot_root
    cat > "${STAGE_DIR}/etc/fstab" <<EOF
proc      /proc     proc    defaults          0 0
sysfs     /sys      sysfs   defaults          0 0
devpts    /dev/pts  devpts  gid=5,mode=0620   0 0
tmpfs     /tmp      tmpfs   defaults,nosuid   0 0
tmpfs     /run      tmpfs   defaults,nosuid   0 0
EOF

    # ── hostname ────────────────────────────────────────────────────────────
    printf 'frugalos\n' > "${STAGE_DIR}/etc/hostname"

    # ── shells ──────────────────────────────────────────────────────────────
    printf '/bin/sh\n/bin/ash\n' > "${STAGE_DIR}/etc/shells"

    # ── passwd / shadow / group ─────────────────────────────────────────────
    # Root only. No password (console is trusted in this model;
    # network auth is a plugin responsibility).
    cat > "${STAGE_DIR}/etc/passwd" <<EOF
root:x:0:0:root:/root:/bin/sh
nobody:x:65534:65534:nobody:/:/sbin/nologin
EOF
    cat > "${STAGE_DIR}/etc/shadow" <<EOF
root:::0:::::
nobody:!:::::::
EOF
    cat > "${STAGE_DIR}/etc/group" <<EOF
root:x:0:
tty:x:5:
disk:x:6:
audio:x:29:
video:x:44:
input:x:101:
nobody:x:65534:
EOF
    chmod 640 "${STAGE_DIR}/etc/shadow"

    # ── profile (login shell) ───────────────────────────────────────────────
    cat > "${STAGE_DIR}/etc/profile" <<'EOF'
# /etc/profile — FrugalOS login shell environment
export PATH=/usr/bin:/usr/sbin:/bin:/sbin
export PS1='\[\033[1;32m\]\u@\h\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]\$ '
export TERM="${TERM:-linux}"
export HOME="${HOME:-/root}"

# Source any plugin-provided profile fragments
for f in /etc/profile.d/*.sh; do
    [ -f "$f" ] && . "$f"
done
EOF

    mkdir -p "${STAGE_DIR}/etc/profile.d"

    # ── inittab (busybox init) ──────────────────────────────────────────────
    cat > "${STAGE_DIR}/etc/inittab" <<EOF
# /etc/inittab — FrugalOS busybox init
# Run system init scripts
::sysinit:/etc/frugalos/init.d/sysinit
# Wait for boot services to complete
::wait:/etc/frugalos/init.d/boot
# Virtual consoles
tty1::respawn:/sbin/getty -L tty1 0 vt100
tty2::respawn:/sbin/getty -L tty2 0 vt100
# Ctrl-Alt-Del
::ctrlaltdel:/sbin/reboot
# Clean shutdown
::shutdown:/etc/frugalos/init.d/shutdown
EOF

    log "/etc populated"
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL FRUGALOS INIT SCRIPTS
# /sbin/init → busybox init (via applet symlink already created)
# /etc/frugalos/init.d/ — our system lifecycle scripts
#───────────────────────────────────────────────────────────────────────────────

install_init_scripts() {
    step "Installing FrugalOS init scripts"

    local initd="${STAGE_DIR}/etc/frugalos/init.d"

    # ── sysinit ────────────────────────────────────────────────────────────
    cat > "${initd}/sysinit" <<'SYSINIT'
#!/bin/sh
# /etc/frugalos/init.d/sysinit
# Runs once as PID 1's first child (sysinit target)
# At this point we are already running on the overlay root

# Mount anything fstab missed (proc/sys are usually already up from /init)
mount -a 2>/dev/null || true

# Set hostname
hostname -F /etc/hostname 2>/dev/null || true

# Bring up loopback
ifconfig lo 127.0.0.1 netmask 255.0.0.0 up 2>/dev/null || true

# Seed /dev with any static nodes mdev might have missed
mdev -s 2>/dev/null || true

# Start mdev as hotplug agent
printf '/sbin/mdev\n' > /proc/sys/kernel/hotplug 2>/dev/null || true

# Set system clock from hardware clock if available
hwclock -s 2>/dev/null || true

# Make /tmp and /run clean
chmod 1777 /tmp /run 2>/dev/null || true

# Run any plugin sysinit hooks in /etc/frugalos/init.d/sysinit.d/
for hook in /etc/frugalos/init.d/sysinit.d/*.sh; do
    [ -f "$hook" ] || continue
    sh "$hook"
done

printf '\033[1;32m  FrugalOS %s — sysinit complete\033[0m\n' \
    "$(. /etc/os-release 2>/dev/null; printf '%s' "$VERSION")"
SYSINIT

    # ── boot ───────────────────────────────────────────────────────────────
    cat > "${initd}/boot" <<'BOOT'
#!/bin/sh
# /etc/frugalos/init.d/boot
# Runs once after sysinit completes, before gettys spawn.
# Plugins add service scripts to /etc/frugalos/init.d/boot.d/

for svc in /etc/frugalos/init.d/boot.d/*.sh; do
    [ -f "$svc" ] || continue
    sh "$svc"
done
BOOT

    # ── shutdown ───────────────────────────────────────────────────────────
    cat > "${initd}/shutdown" <<'SHUTDOWN'
#!/bin/sh
# /etc/frugalos/init.d/shutdown
# Called on reboot/halt/poweroff

for svc in /etc/frugalos/init.d/shutdown.d/*.sh; do
    [ -f "$svc" ] || continue
    sh "$svc"
done

# Sync filesystems, then lazy unmount everything
sync
umount -a -r 2>/dev/null || true
SHUTDOWN

    # Make all scripts executable
    chmod 755 "${initd}/sysinit" "${initd}/boot" "${initd}/shutdown"

    # Create hook directories for plugins to drop into
    mkdir -p "${initd}/sysinit.d" "${initd}/boot.d" "${initd}/shutdown.d"

    log "init scripts installed"
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL FRUGALOS TOOLS
# Copies plugin-ctl and manifest-new from the directory containing this script
#───────────────────────────────────────────────────────────────────────────────

install_frugalos_tools() {
    step "Installing FrugalOS tools"

    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"

    for tool in plugin-ctl manifest-new; do
        local src="${script_dir}/${tool}"
        if [ -f "$src" ]; then
            cp "$src" "${STAGE_DIR}/usr/bin/${tool}"
            chmod 755 "${STAGE_DIR}/usr/bin/${tool}"
            log "$tool installed from $src"
        else
            warn "$tool not found at $src — skipping (build it first)"
        fi
    done

    # Version info
    mkdir -p "${STAGE_DIR}/usr/share/frugalos"
    cat > "${STAGE_DIR}/usr/share/frugalos/version" <<EOF
FRUGALOS_VERSION=${FRUGALOS_VERSION}
BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUILD_HOST=$(hostname 2>/dev/null || echo unknown)
EOF
}

#───────────────────────────────────────────────────────────────────────────────
# INSTALL BASE PLUGIN MANIFEST
# base.sfs is itself a plugin — LOAD_ORDER 0, cannot be unloaded
#───────────────────────────────────────────────────────────────────────────────

install_base_manifest() {
    step "Installing base plugin manifest"

    cat > "${STAGE_DIR}/.plugin/manifest" <<EOF
NAME="base"
VERSION="${FRUGALOS_VERSION}"
DESCRIPTION="FrugalOS base layer — musl, busybox, init, plugin-ctl"
ARCH="x86_64"
DEPENDS=""
CONFLICTS=""
PROVIDES="base libc init shell"
LOAD_ORDER=0
EOF

    log "base manifest written"
}

#───────────────────────────────────────────────────────────────────────────────
# PERMISSIONS PASS
# Ensure setuid bits and ownership are correct before squashing
#───────────────────────────────────────────────────────────────────────────────

fix_permissions() {
    step "Fixing permissions"

    # busybox su needs setuid root
    chmod u+s "${STAGE_DIR}/bin/busybox" 2>/dev/null || true

    # shadow must not be world-readable
    chmod 640  "${STAGE_DIR}/etc/shadow"

    # /root home directory
    chmod 700 "${STAGE_DIR}/root"

    # /tmp sticky
    chmod 1777 "${STAGE_DIR}/tmp"

    # All regular files in /etc should not be world-writable
    find "${STAGE_DIR}/etc" -type f | while read -r f; do
        chmod o-w "$f" 2>/dev/null || true
    done

    log "permissions fixed"
}

#───────────────────────────────────────────────────────────────────────────────
# SIZE AUDIT
# Print a breakdown before compressing so we know where the bytes are
#───────────────────────────────────────────────────────────────────────────────

size_audit() {
    step "Stage directory size breakdown"
    du -sh "${STAGE_DIR}"/* 2>/dev/null | sort -h | while read -r size path; do
        printf '   %-8s %s\n' "$size" "${path#${STAGE_DIR}/}"
    done
    printf '\n'
    info "total uncompressed: $(du -sh "$STAGE_DIR" | cut -f1)"
}

#───────────────────────────────────────────────────────────────────────────────
# BUILD SFS
#───────────────────────────────────────────────────────────────────────────────

build_sfs() {
    step "Building base.sfs (compression: ${SFS_COMP})"

    [ -f "$OUTPUT_SFS" ] && {
        log "removing previous: $OUTPUT_SFS"
        rm -f "$OUTPUT_SFS"
    }

    mksquashfs "$STAGE_DIR" "$OUTPUT_SFS" \
        -comp "$SFS_COMP" $SFS_COMP_FLAGS \
        -noappend \
        -no-exports \
        -no-progress \
        2>&1 | while IFS= read -r line; do info "  $line"; done

    [ -f "$OUTPUT_SFS" ] || die "mksquashfs produced no output"

    local size
    size=$(du -sh "$OUTPUT_SFS" | cut -f1)
    log "base.sfs built: $OUTPUT_SFS ($size)"

    # Quick sanity: verify we can unsquash at least the manifest
    unsquashfs -l "$OUTPUT_SFS" .plugin/manifest >/dev/null 2>&1 \
        && log "squashfs integrity: OK" \
        || warn "integrity spot-check failed — investigate before deploying"
}

#───────────────────────────────────────────────────────────────────────────────
# CLEANUP
#───────────────────────────────────────────────────────────────────────────────

cleanup() {
    teardown_alpine
    log "build complete. Chroot preserved at: $CHROOT_DIR"
    log "re-run with --clean to remove build artifacts"
}

do_clean() {
    log "cleaning build artifacts"
    teardown_alpine 2>/dev/null || true
    rm -rf "$BUILD_DIR"
    log "clean complete"
}

#───────────────────────────────────────────────────────────────────────────────
# MAIN
#───────────────────────────────────────────────────────────────────────────────

trap 'teardown_alpine 2>/dev/null || true' EXIT INT TERM

case "${1:-}" in
    --clean|-c)
        do_clean
        exit 0
        ;;
    --stage-only|-s)
        # Stop after building stage dir, don't mksquashfs
        STAGE_ONLY=1
        ;;
    --help|-h)
        cat <<EOF
Usage: $0 [options]

Options:
  (none)       full build: bootstrap → stage → base.sfs
  --clean      remove build artifacts
  --stage-only build stage directory only (skip mksquashfs)

Environment:
  ALPINE_VERSION   Alpine version to use (default: ${ALPINE_VERSION})
  KERNEL_VERSION   pre-populate /lib/modules/<ver>/ skeleton
  SFS_COMP         squashfs compression: xz|zstd|lz4 (default: xz)
  FRUGALOS_VERSION embedded in os-release (default: ${FRUGALOS_VERSION})

Output: ${OUTPUT_SFS}
EOF
        exit 0
        ;;
esac

printf '\n'
printf '\033[1;37m═══════════════════════════════════════════════\033[0m\n'
printf '\033[1;37m  FrugalOS base.sfs Builder v%s\033[0m\n' "$FRUGALOS_VERSION"
printf '\033[1;37m═══════════════════════════════════════════════\033[0m\n\n'

preflight
bootstrap_alpine
build_fhs
install_busybox
install_musl
install_squashfs_tools
install_modules_skeleton
install_etc
install_init_scripts
install_frugalos_tools
install_base_manifest
fix_permissions
size_audit

if [ "${STAGE_ONLY:-0}" = "1" ]; then
    log "stage-only mode — skipping mksquashfs"
    log "stage dir: $STAGE_DIR"
else
    build_sfs
fi

cleanup

printf '\n'
printf '\033[1;32m  Done.\033[0m\n\n'
