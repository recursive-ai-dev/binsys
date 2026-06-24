//! PuppyBoot V1 — build script.
//!
//! Intentionally minimal. The `x86_64-unknown-uefi` target is a Tier-2
//! built-in target in rustc; it already emits a PE32+ image with the
//! `EFI Application` subsystem (IMAGE_SUBSYSTEM_EFI_APPLICATION = 10) and
//! the correct entry-point thunk for `efi_main`. No linker scripts, no
//! custom target JSON, and no post-processing are required.
//!
//! We keep this file (rather than deleting it) so that future additions —
//! embedding a build timestamp, a version string, or a compressed splash
//! image via `include_bytes!` — have an obvious home, and so `cargo` does
//! not warn about an unexpected absence when tooling expects one.

fn main() {
    // Re-run only if this script itself changes.
    println!("cargo:rerun-if-changed=build.rs");
}
