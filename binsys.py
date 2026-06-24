#!/usr/bin/env python3
"""binsys — create and run filesystem images like VMs

Supports:
  ext4     — raw writable image (default)
  overlay  — squashfs base + ext4 save layer  (Puppy frugal style)
  squashfs — compressed read-only snapshot    (MX snapshot style)
  fat32    — FAT32 image                      (Ventoy-compatible)
"""

from binsys.cli import main

if __name__ == "__main__":
    main()
