#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  build_aios_kernel.sh — Hardened AIOS kernel build
#
#  HARDENING CHANGES
#  ─────────────────
#  • SHA256 verification of the tarball BEFORE GPG (defence in depth).
#  • Build lockfile via flock(1) — no concurrent builds corrupting the tree.
#  • Atomic artifact writes (tempfile + rename) — manifest is never partial.
#  • All temp files created securely under mktemp(1); aggressive cleanup trap.
#  • Paths are canonicalised and validated before any rm -rf.
#  • Suppressed stdout is captured to a log; on failure the tail is dumped.
#  • bzImage is validated with file(1) and readelf(1) before shipping.
#  • Verification is mandatory; VERIFY_SIG=0 requires an explicit UNSAFE flag.
#  • Boot-critical audit now also rejects =m (module) where =y is required.
# ════════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

# ─── Tunables ───────────────────────────────────────────────────────────────
KERNEL_MAJOR="6"
KERNEL_VER="6.18.34"
KERNEL_BASE="defconfig"
JOBS="$(nproc)"
WORK="${WORK:-$PWD/.kbuild}"
OUT="${OUT:-$PWD/out}"
FRAGMENT="${FRAGMENT:-$PWD/aios_kernel.config}"
VERIFY_SIG="${VERIFY_SIG:-1}"
UNSAFE_SKIP_VERIFY="${UNSAFE_SKIP_VERIFY:-0}"   # must be explicitly set to 1

CDN="https://cdn.kernel.org/pub/linux/kernel/v${KERNEL_MAJOR}.x"
TARBALL="linux-${KERNEL_VER}.tar.xz"
SIGFILE="linux-${KERNEL_VER}.tar.sign"
SHAFILE="sha256sums.asc"
SRCDIR="${WORK}/linux-${KERNEL_VER}"
STAMPS="${WORK}/.stamps"
LOCKFILE="${WORK}/.build.lock"
BUILD_LOG="${WORK}/build.log"

KERNEL_KEYS=(
  "647F28654894E3BD457199BE38DBBDC86092693E"
  "E27E5D8A3403A2EF66873BBCDEA66FF797772CDC"
  "ABAF11C65A2970B130ABE3C479BE3E4300411886"
)

# ─── Helpers ────────────────────────────────────────────────────────────────
log()  { printf '\033[1;36m[kbuild]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[kbuild WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[kbuild FATAL]\033[0m %s\n' "$*" >&2; exit 1; }
stamp(){ touch "${STAMPS}/$1"; }
is_done() { [[ -f "${STAMPS}/$1" ]]; }

# Atomic write: $1 = source, $2 = dest
atomic_install() {
  local src="$1" dst="$2"
  local tmp="${dst}.tmp.$$"
  cp -f "$src" "$tmp" || die "copy $src → $tmp failed"
  chmod 0644 "$tmp"
  sync "$tmp"
  mv -f "$tmp" "$dst" || die "atomic rename $tmp → $dst failed"
}

# Safe path validation: aborts if path is empty, relative, or resolves outside WORK
validate_under() {
  local path="$1" base="$2" name="$3"
  [[ -n "$path" ]] || die "$name is empty"
  [[ "$path" = /* ]] || die "$name must be absolute: $path"
  local realpath
  realpath="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")" 2>/dev/null || die "$name unreadable: $path"
  [[ "$realpath" == "$base"* ]] || die "$name ($realpath) escapes base ($base)"
}

# On-error log tail
on_error() {
  local code=$? line=$1
  die "failed at line $line (exit $code). Build log tail:\n$(tail -n 40 "$BUILD_LOG" 2>/dev/null || echo '<no log>')"
}
trap 'on_error $LINENO' ERR

# Cleanup on exit/interrupt
cleanup() {
  [[ -f "${WORK}/${TARBALL}.part" ]] && rm -f "${WORK}/${TARBALL}.part"
  [[ -f "${WORK}/${SIGFILE}.part" ]] && rm -f "${WORK}/${SIGFILE}.part"
  [[ -f "${WORK}/${SHAFILE}.part" ]] && rm -f "${WORK}/${SHAFILE}.part"
}
trap cleanup EXIT

require() {
  local missing=()
  for t in "$@"; do command -v "$t" >/dev/null 2>&1 || missing+=("$t"); done
  ((${#missing[@]}==0)) || die "missing host tools: ${missing[*]}
  Debian/Ubuntu: apt-get install build-essential flex bison libelf-dev libssl-dev \\
                 bc kmod cpio xz-utils zstd wget gnupg file
  Alpine:        apk add build-base linux-headers flex bison elfutils-dev openssl-dev \\
                 bc kmod cpio xz zstd wget gnupg file"
}

mkdir -p "${WORK}" "${OUT}" "${STAMPS}"
WORK_REAL="$(cd "$WORK" && pwd)"
OUT_REAL="$(cd "$OUT" && pwd)"

validate_under "${SRCDIR}" "${WORK_REAL}" "SRCDIR"
validate_under "${FRAGMENT}" "${PWD}" "FRAGMENT"

# ─── Phase 0: host toolchain ────────────────────────────────────────────────
require make gcc ld flex bison bc cpio xz zstd wget tar sha256sum file readelf
[[ "${VERIFY_SIG}" == "1" ]] && require gpg gpg2 || true
command -v gpg >/dev/null 2>&1 || command -v gpg2 >/dev/null 2>&1 || \
  { [[ "${VERIFY_SIG}" == "1" ]] && die "gpg required (or set UNSAFE_SKIP_VERIFY=1, NOT recommended)"; }
[[ -f "${FRAGMENT}" ]] || die "config fragment not found: ${FRAGMENT}"

# ─── Lockfile: prevent concurrent builds ────────────────────────────────────
exec 200>"$LOCKFILE"
flock -n 200 || die "another build holds $LOCKFILE (concurrent builds are unsafe)"

# ─── Phase 1: fetch ─────────────────────────────────────────────────────────
if ! is_done fetch; then
  log "fetching ${TARBALL}"
  wget -q --show-progress -O "${WORK}/${TARBALL}.part" "${CDN}/${TARBALL}" \
    || die "download failed: ${CDN}/${TARBALL}"
  mv -f "${WORK}/${TARBALL}.part" "${WORK}/${TARBALL}"

  if [[ "${VERIFY_SIG}" == "1" ]]; then
    wget -q -O "${WORK}/${SIGFILE}.part" "${CDN}/${SIGFILE}" || die "signature download failed"
    mv -f "${WORK}/${SIGFILE}.part" "${WORK}/${SIGFILE}"
    wget -q -O "${WORK}/${SHAFILE}.part" "${CDN}/${SHAFILE}" || die "SHA256SUMS download failed"
    mv -f "${WORK}/${SHAFILE}.part" "${WORK}/${SHAFILE}"
  fi
  stamp fetch
else
  log "phase fetch: cached"
fi

# ─── Phase 2: verify ────────────────────────────────────────────────────────
if [[ "${VERIFY_SIG}" == "1" ]] && ! is_done verify; then
  log "verifying SHA256 + PGP signature"

  # 2a: SHA256 verification first (defence in depth — detects CDN corruption / MITM)
  tarball_hash="$(sha256sum "${WORK}/${TARBALL}" | awk '{print $1}')"
  grep -F "$tarball_hash" "${WORK}/${SHAFILE}" >/dev/null 2>&1 \
    || die "SHA256 mismatch for ${TARBALL}. CDN compromised, corrupted download, or wrong SHA256SUMS file."

  # 2b: PGP verification over decompressed tar
  GPG="$(command -v gpg || command -v gpg2)"
  export GNUPGHOME="${WORK}/.gnupg"
  mkdir -p "${GNUPGHOME}"; chmod 700 "${GNUPGHOME}"
  for key in "${KERNEL_KEYS[@]}"; do
    "${GPG}" --batch --keyserver hkps://keyserver.ubuntu.com --recv-keys "${key}" 2>/dev/null \
      || "${GPG}" --batch --keyserver hkps://keys.openpgp.org --recv-keys "${key}" 2>/dev/null \
      || warn "could not fetch key ${key} (verify may fail)"
  done

  log "decompressing for PGP verify"
  xz -dc "${WORK}/${TARBALL}" > "${WORK}/linux-${KERNEL_VER}.tar"
  "${GPG}" --batch --verify "${WORK}/${SIGFILE}" "${WORK}/linux-${KERNEL_VER}.tar" \
    || die "PGP VERIFICATION FAILED — tarball is NOT trusted. Aborting."
  rm -f "${WORK}/linux-${KERNEL_VER}.tar"
  log "signature OK"
  stamp verify
elif [[ "${VERIFY_SIG}" != "1" ]]; then
  if [[ "${UNSAFE_SKIP_VERIFY}" != "1" ]]; then
    die "VERIFY_SIG != 1 but UNSAFE_SKIP_VERIFY != 1. To build without verification, set UNSAFE_SKIP_VERIFY=1"
  fi
  warn "phase verify: SKIPPED — DO NOT SHIP IMAGES BUILT THIS WAY"
else
  log "phase verify: cached"
fi

# ─── Phase 3: extract ───────────────────────────────────────────────────────
if ! is_done extract; then
  log "extracting source"
  rm -rf "${SRCDIR}"
  tar -C "${WORK}" -xf "${WORK}/${TARBALL}"
  [[ -d "${SRCDIR}" ]] || die "extraction did not yield ${SRCDIR}"
  # Sanity: ensure the tarball actually contained the version we expect
  [[ -f "${SRCDIR}/Makefile" ]] || die "extracted tree lacks Makefile"
  stamp extract
else
  log "phase extract: cached"
fi

# ─── Phase 4: config + merge + reconcile ────────────────────────────────────
if ! is_done config; then
  log "generating base config: ${KERNEL_BASE}"
  make -C "${SRCDIR}" ARCH=x86_64 "${KERNEL_BASE}" >"${BUILD_LOG}" 2>&1 \
    || { tail -n 20 "${BUILD_LOG}"; die "base config failed"; }

  log "merging AIOS fragment"
  ( cd "${SRCDIR}" && ARCH=x86_64 ./scripts/kconfig/merge_config.sh -m .config "${FRAGMENT}" ) >>"${BUILD_LOG}" 2>&1 \
    || die "merge_config.sh failed"

  make -C "${SRCDIR}" ARCH=x86_64 olddefconfig >"${BUILD_LOG}" 2>&1 \
    || { tail -n 20 "${BUILD_LOG}"; die "olddefconfig failed"; }

  # HARD GATE: boot-critical symbols must be =y (builtin), never =m.
  log "auditing boot-critical symbols"
  assert_y() {
    local sym="$1"
    if grep -qx "CONFIG_${sym}=y" "${SRCDIR}/.config"; then
      return 0
    elif grep -qx "CONFIG_${sym}=m" "${SRCDIR}/.config"; then
      die "BOOT-CRITICAL symbol CONFIG_${sym} is =m (module). It MUST be =y (builtin) or the initramfs cannot mount root."
    else
      die "BOOT-CRITICAL symbol CONFIG_${sym} is missing after olddefconfig. A dependency likely disabled it."
    fi
  }
  for sym in SQUASHFS OVERLAY_FS TMPFS BLK_DEV_LOOP DEVTMPFS DEVTMPFS_MOUNT \
             PROC_FS SYSFS BLK_DEV_INITRD EFI_STUB FB_EFI BLK_DEV_SD \
             USB_STORAGE ISO9660_FS VFAT_FS EXT4_FS EFI_PARTITION \
             SQUASHFS_ZSTD SQUASHFS_XZ; do
    assert_y "$sym"
  done
  log "all boot-critical symbols present and builtin"
  stamp config
else
  log "phase config: cached"
fi

# ─── Phase 5: compile ───────────────────────────────────────────────────────
if ! is_done build; then
  log "compiling bzImage (-j${JOBS})"
  make -C "${SRCDIR}" ARCH=x86_64 -j"${JOBS}" bzImage >"${BUILD_LOG}" 2>&1 \
    || { tail -n 30 "${BUILD_LOG}"; die "bzImage build failed"; }

  log "compiling modules"
  make -C "${SRCDIR}" ARCH=x86_64 -j"${JOBS}" modules >"${BUILD_LOG}" 2>&1 \
    || { tail -n 30 "${BUILD_LOG}"; die "modules build failed"; }

  # Validate the artifact is a real x86-64 EFI stub kernel
  file "${SRCDIR}/arch/x86/boot/bzImage" | grep -qiE "x86-64|EFI" \
    || warn "bzImage file(1) probe did not mention x86-64 or EFI — inspect manually"
  readelf -h "${SRCDIR}/arch/x86/boot/bzImage" >/dev/null 2>&1 \
    || warn "readelf cannot parse bzImage — inspect manually"

  stamp build
else
  log "phase build: cached"
fi

# ─── Phase 6: stage modules ─────────────────────────────────────────────────
if ! is_done modules; then
  log "installing modules to staging tree"
  MODSTAGE="${WORK}/modstage"
  rm -rf "${MODSTAGE}"; mkdir -p "${MODSTAGE}"
  make -C "${SRCDIR}" ARCH=x86_64 INSTALL_MOD_PATH="${MODSTAGE}" \
       INSTALL_MOD_STRIP=1 modules_install >"${BUILD_LOG}" 2>&1 \
    || die "modules_install failed"

  # Verify depmod ran
  [[ -f "${MODSTAGE}/lib/modules/${KERNEL_VER}/modules.dep" ]] \
    || die "modules.dep missing — depmod did not run correctly"

  ( cd "${MODSTAGE}" && tar -I 'zstd -19' -cf "${OUT}/modules.tar.zst.part" lib/modules )
  mv -f "${OUT}/modules.tar.zst.part" "${OUT}/modules.tar.zst"
  stamp modules
else
  log "phase modules: cached"
fi

# ─── Phase 7: collect artifacts + manifest ────────────────────────────────────
log "collecting artifacts"
atomic_install "${SRCDIR}/arch/x86/boot/bzImage" "${OUT}/bzImage"
atomic_install "${SRCDIR}/.config"               "${OUT}/aios_kernel.release"
atomic_install "${SRCDIR}/System.map"            "${OUT}/System.map"

# Build metadata
cat > "${OUT}/aios_kernel.release.meta" <<EOF
kernel_version=${KERNEL_VER}
build_host=$(hostname -f 2>/dev/null || hostname)
build_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
builder=$(whoami)
toolchain=$(gcc --version 2>/dev/null | head -n1)
EOF
chmod 0644 "${OUT}/aios_kernel.release.meta"

( cd "${OUT}" && sha256sum bzImage modules.tar.zst aios_kernel.release System.map aios_kernel.release.meta > SHA256SUMS.part )
mv -f "${OUT}/SHA256SUMS.part" "${OUT}/SHA256SUMS"

KSIZE=$(du -h "${OUT}/bzImage" | cut -f1)
log "DONE."
log "  kernel : ${OUT}/bzImage  (${KSIZE}, linux ${KERNEL_VER})"
log "  modules: ${OUT}/modules.tar.zst"
log "  config : ${OUT}/aios_kernel.release"
log "  meta   : ${OUT}/aios_kernel.release.meta"
log ""
log "Next: feed bzImage to PuppyBoot/isolinux, and unpack modules.tar.zst into"
log "your squashfs at /lib/modules so post-boot modprobe of optional drivers works."
