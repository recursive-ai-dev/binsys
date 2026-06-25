#!/usr/bin/env python3
"""Small utilities for binsys: deterministic_id, CLI prompts, colors, and rename helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Colors (ANSI)
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"


def deterministic_id(prefix: str, *parts: str) -> str:
    content = "|".join(parts)
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def prompt(msg: str, default: str | None = None) -> str:
    if default is not None:
        v = input(f"{msg} [{default}]: ")
        return v.strip() or default
    return input(f"{msg}: ").strip()


def confirm(msg: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    resp = input(f"{msg} ({d}): ").strip().lower()
    if resp == "":
        return default
    return resp in ("y", "yes")


def write_undo_log(path: Path, ops: list[tuple[str, str]]):
    """Write undo log mapping new -> old so renames can be reverted."""
    data = {"ops": [{"src": s, "dst": d} for s, d in ops], "created": None}
    path.write_text(json.dumps(data, indent=2))


def preview_and_apply_renames(ops: list[tuple[Path, Path]], dry_run: bool = True, undo_log: Path | None = None) -> None:
    """Show planned renames, optionally apply them and write undo log.

    ops: list of (src, dst)
    """
    print("Planned renames:")
    for s, d in ops:
        print(f"  {s} -> {d}")
    if dry_run:
        print("Dry run only; no changes made.")
        return
    # apply safely: ensure no dst exists or is same as src
    applied = []
    for s, d in ops:
        s = Path(s)
        d = Path(d)
        if not s.exists():
            print(f"Skipping missing source: {s}")
            continue
        if d.exists():
            print(f"Skipping existing destination: {d}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)
        applied.append((str(d), str(s)))
    if undo_log and applied:
        write_undo_log(undo_log, applied)
