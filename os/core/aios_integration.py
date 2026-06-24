#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Integration Wiring Reference                                         ║
║  aios_integration.py                                                         ║
║                                                                              ║
║  "The kernel was always waiting for its nervous system. Now it has one."     ║
║                                                                              ║
║  This file documents — with runnable code — exactly how to attach the four  ║
║  new subsystems (reward, IPC, VFS, feedback) to a live AgentKernel.         ║
║                                                                              ║
║  Minimal integration (add to aios_core.py after kernel.boot()):             ║
║                                                                              ║
║      from aios_reward   import RewardKernel                                  ║
║      from aios_ipc      import IPCKernel                                    ║
║      from aios_vfs      import VFSKernel                                    ║
║      from aios_feedback import FeedbackKernel                               ║
║                                                                              ║
║      reward_kernel   = RewardKernel()                                        ║
║      reward_kernel.attach(kernel)                                            ║
║                                                                              ║
║      ipc_kernel = IPCKernel()                                                ║
║      ipc_kernel.attach(kernel)                                               ║
║                                                                              ║
║      vfs_kernel = VFSKernel(kernel=kernel, bus=kernel.bus)                  ║
║      vfs_kernel.attach(kernel)                                               ║
║                                                                              ║
║      feedback_kernel = FeedbackKernel()                                      ║
║      feedback_kernel.attach(kernel)     # hot-swaps kernel._reasoner        ║
║      feedback_kernel.start()            # starts daemon training thread     ║
║                                                                              ║
║  After this sequence:                                                        ║
║    • kernel._reasoner is now a LearningReasoner (with RuleBased fallback)  ║
║    • Every kernel.dispatch() → AgentTrace → reward signal → RL buffer       ║
║    • Background thread trains PPO every EPISODE_LEN (64) new traces         ║
║    • After MIN_TRAIN_STEPS (50) training steps, RL policy activates         ║
║    • kernel.dispatch("vfs_write_file", path=…, data=…) works               ║
║    • kernel.dispatch("ipc_publish", topic=…, payload=…) works              ║
║    • kernel.dispatch("feedback_stats") introspects the learning loop        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION ENTRY POINT
#
# Call attach_all_subsystems(kernel) immediately after kernel.boot() returns.
# Returns a dict of all attached subsystem objects for introspection.
# ─────────────────────────────────────────────────────────────────────────────

def attach_all_subsystems(kernel: Any) -> dict:
    """
    Attach reward, IPC, VFS, and feedback subsystems to a booted AgentKernel.

    Args:
        kernel: A fully booted AgentKernel instance (state must be READY or RUNNING).

    Returns:
        dict with keys: reward_kernel, ipc_kernel, vfs_kernel, feedback_kernel
    """

    # ── 1. Reward subsystem ───────────────────────────────────────────────────
    #    Defines the intrinsic reward signal for every @agent_method call.
    #    Must be attached before FeedbackKernel so the shaper and encoder
    #    are available for ExperienceBuffer construction.
    from aios_reward import RewardKernel
    reward_kernel = RewardKernel()
    reward_kernel.attach(kernel)
    _log(kernel, "[AIOS] RewardKernel attached — reward shaping active")

    # ── 2. IPC subsystem ──────────────────────────────────────────────────────
    #    Message queues, event bus, channels, shared memory, semaphores.
    #    Enables agent↔agent communication without direct object references.
    from aios_ipc import IPCKernel, IPCNamespace
    ipc_kernel = IPCKernel()
    ipc_kernel.attach(kernel)
    _log(kernel, "[AIOS] IPCKernel attached — event bus and message queues live")

    # ── 3. VFS subsystem ──────────────────────────────────────────────────────
    #    Virtual filesystem: open/read/write/close through agent dispatch.
    #    Default mounts: /→MemFS, /proc→ProcFS, /dev→DevFS, /tmp→MemFS
    bus = getattr(kernel, 'bus', None)
    from aios_vfs import VFSKernel
    vfs_kernel = VFSKernel(kernel=kernel, bus=bus)
    vfs_kernel.attach(kernel)
    _log(kernel, "[AIOS] VFSKernel attached — /proc /dev /tmp /etc mounted")

    # ── 4. Feedback loop — must be last ──────────────────────────────────────
    #    Hot-swaps kernel._reasoner to LearningReasoner (which wraps
    #    RuleBasedReasoner as cold-start fallback until trained).
    #    Starts daemon thread reading AgentRegistry traces and training PPO.
    from aios_feedback import FeedbackKernel
    feedback_kernel = FeedbackKernel()
    feedback_kernel.attach(kernel)    # installs LearningReasoner
    feedback_kernel.start()           # starts background training thread
    _log(kernel, "[AIOS] FeedbackKernel attached — RL feedback loop running")
    _log(kernel, f"[AIOS]   reasoner = {type(kernel._reasoner).__name__}")

    return {
        "reward_kernel":   reward_kernel,
        "ipc_kernel":      ipc_kernel,
        "vfs_kernel":      vfs_kernel,
        "feedback_kernel": feedback_kernel,
    }


def _log(kernel: Any, msg: str) -> None:
    """Log to kernel._log if available, else print."""
    if hasattr(kernel, '_log'):
        kernel._log(msg)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# PATCH FOR aios_core.py AgentKernel.boot()
#
# Locate the boot() method in AgentKernel (around line 2100 in aios_core.py)
# and add the following at the very end, just before the final state transition:
#
#     # ── §7.5  Extended subsystems ─────────────────────────────────────────
#     try:
#         from aios_integration import attach_all_subsystems
#         self._subsystems = attach_all_subsystems(self)
#         self._log("[BOOT]  Extended subsystems ✓ (reward/ipc/vfs/feedback)")
#     except ImportError as e:
#         self._log(f"[BOOT]  Extended subsystems skipped: {e}")
#
# This makes the import optional — the kernel boots fine without the new
# modules present (e.g. in minimal/embedded deployments).
# ─────────────────────────────────────────────────────────────────────────────

BOOT_PATCH = '''
    # ── §7.5  Extended subsystems ─────────────────────────────────────────────
    # Attach reward, IPC, VFS, and RL feedback subsystems if available.
    # Import is guarded so minimal kernels boot without these modules.
    try:
        from aios_integration import attach_all_subsystems
        self._subsystems = attach_all_subsystems(self)
        self._log("[BOOT]  Extended subsystems ✓  (reward / ipc / vfs / feedback)")
    except ImportError as _e:
        self._subsystems = {}
        self._log(f"[BOOT]  Extended subsystems skipped ({_e})")
'''


# ─────────────────────────────────────────────────────────────────────────────
# REPL EXTENSIONS
#
# Add these commands to TerminalREPL in aios_core.py to expose the new
# subsystems from the kernel terminal:
#
#   feedback          → print FeedbackKernel.stats()
#   vfs ls <path>     → print VFS readdir
#   vfs cat <path>    → print VFS read_file
#   ipc emit <topic>  → publish to kernel event bus
#   reward stats      → print RewardKernel.stats()
# ─────────────────────────────────────────────────────────────────────────────

REPL_COMMANDS = {
    "feedback": "self._kernel.dispatch('feedback_stats')",
    "vfs ls":   "self._kernel.dispatch('vfs_readdir', path=args[1] if args else '/')",
    "vfs cat":  "self._kernel.dispatch('vfs_read_file', path=args[1])",
    "ipc emit": "self._kernel.dispatch('ipc_publish', topic=args[1], payload=args[2] if len(args)>2 else {})",
    "reward":   "self._kernel.dispatch('reward_stats')",
}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST — run standalone to verify the full wiring
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Full integration smoke test.
    Boots a real AgentKernel and verifies all four subsystems attach correctly.
    """
    # aios_core must be on the path
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'project')
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)

    try:
        from aios_core import AgentKernel
    except ImportError:
        print("aios_integration smoke test: aios_core not on path — skipping kernel boot test")
        return

    print("Booting AgentKernel…")
    kernel = AgentKernel()
    kernel.boot()
    print(f"Kernel state: {kernel._state.name}")

    print("Attaching subsystems…")
    subs = attach_all_subsystems(kernel)
    assert "reward_kernel"   in subs
    assert "ipc_kernel"      in subs
    assert "vfs_kernel"      in subs
    assert "feedback_kernel" in subs

    # Verify dispatch routing for each subsystem
    r = kernel.dispatch("reward_stats")
    assert r.success, f"reward_stats failed: {r.error}"
    print(f"  reward_stats: state_dim={r.value['state_dim']}")

    r = kernel.dispatch("ipc_stats")
    assert r.success, f"ipc_stats failed: {r.error}"
    print(f"  ipc_stats: buses={list(r.value['namespace']['buses'].keys())}")

    r = kernel.dispatch("vfs_mounts")
    assert r.success, f"vfs_mounts failed: {r.error}"
    prefixes = [m['prefix'] for m in r.value]
    assert "/" in prefixes and "/proc" in prefixes and "/dev" in prefixes
    print(f"  vfs_mounts: {prefixes}")

    r = kernel.dispatch("feedback_stats")
    assert r.success, f"feedback_stats failed: {r.error}"
    print(f"  feedback_stats: loop.state={r.value['loop']['state']}")

    # Verify VFS works end-to-end through dispatch
    r = kernel.dispatch("vfs_write_file", path="/tmp/smoke.txt", data=b"AIOS is alive")
    assert r.success, f"vfs_write_file failed: {r.error}"
    r = kernel.dispatch("vfs_read_file", path="/tmp/smoke.txt")
    assert r.success and r.value == b"AIOS is alive"
    print("  VFS read/write via dispatch: ✓")

    # Verify IPC publish/subscribe round-trip
    recv_q = subs["ipc_kernel"]._ns.get_or_create_queue("smoke_q")
    sub_id = subs["ipc_kernel"]._bus.subscribe(
        "smoke.*",
        queue=recv_q,
    )
    r = kernel.dispatch("ipc_publish", topic="smoke.test", payload={"ok": True})
    assert r.success
    msg = recv_q.recv(timeout=1.0)
    assert msg is not None and msg.payload == {"ok": True}
    print("  IPC publish/subscribe round-trip: ✓")

    # Allow feedback loop a moment to ingest the traces we just generated
    time.sleep(0.3)
    r = kernel.dispatch("feedback_stats")
    assert r.success
    stats = r.value["loop"]
    print(f"  Feedback loop: updates={stats['update_count']}, ingested={stats['total_ingested']}")

    # Stop the feedback loop cleanly
    subs["feedback_kernel"].stop()
    print("\naios_integration: full smoke test passed ✓")


if __name__ == "__main__":
    _smoke_test()
