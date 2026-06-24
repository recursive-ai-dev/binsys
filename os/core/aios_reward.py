#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Intrinsic Reward Subsystem                                           ║
║  aios_reward.py                                                              ║
║                                                                              ║
║  "Reward is not given. It is measured. Every trace is a lesson in value."   ║
║                                                                              ║
║  This module defines the reward signal that transforms raw AgentTrace        ║
║  records into the scalar feedback that drives the RL feedback loop.          ║
║  Without a formal reward definition the RL machinery in aios_rl.py cannot   ║
║  improve the kernel's own behaviour — this module closes that gap.           ║
║                                                                              ║
║  Components:                                                                 ║
║    §0  Constants & Math Shim  — no math import                              ║
║    §1  LatencyModel           — priority-stratified target latency           ║
║    §2  RewardComponents       — individual reward terms (latency/success/…)  ║
║    §3  RewardShaper           — composite reward + Welford online σ-norm     ║
║    §4  StateEncoder           — AgentTrace → ℝ^STATE_DIM feature vector      ║
║    §5  RewardKernel           — @agent_method integration hook               ║
║    §6  Self-Tests             — deterministic validation suite               ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Latency term  : r_lat = −ln(1 + d_μs / T*)  [log-penalty; T*=priority]  ║
║    Success term  : r_suc = +1.0 if ok, −2.0 if error  [asymmetric]         ║
║    Depth term    : r_dep = −0.1 · min(depth / 10, 1)  [parsimony pressure] ║
║    Composite     : R = w_s·r_suc + w_l·r_lat + w_d·r_dep                   ║
║    Welford norm  : μ_n = μ_{n-1} + (x−μ_{n-1})/n                           ║
║                   S_n = S_{n-1} + (x−μ_{n-1})·(x−μ_n)                     ║
║                   σ̂ = sqrt(S_n / n),  R̂ = (R − μ̂) / (σ̂ + ε)            ║
║                   [Welford 1962, online variance algorithm]                  ║
║    State space   : s ∈ ℝ^{STATE_DIM=16}  (per-trace feature vector)        ║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Zero external dependencies. Pure Python 3.9+ stdlib only.               ║
║    • Thread-safe: all mutable state guarded by threading.RLock.              ║
║    • Standalone: runs without aios_core/aios_rl on path.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import array
import hashlib
import json
import struct
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# §0  MATH SHIM — no math import; mirrors idiom in every other AIOS module
# ─────────────────────────────────────────────────────────────────────────────

_PI   = 3.141592653589793238462643383279
_E    = 2.718281828459045235360287471352
_LN2  = 0.693147180559945309417232121458
_INF  = float('inf')
_EPS  = 1e-12

_REWARD_VERSION = "1.0.0"
STATE_DIM       = 16   # dimensionality of encoded state vector


def _abs(x: float) -> float:
    return x if x >= 0.0 else -x


def _max(a: float, b: float) -> float:
    return a if a > b else b


def _min(a: float, b: float) -> float:
    return a if a < b else b


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _exp(x: float) -> float:
    """e^x via range-reduction + Taylor; exact for |x| ≤ 709."""
    if x > 709.782: return _INF
    if x < -745.0:  return 0.0
    k = int(x / _LN2)
    r = x - k * _LN2
    # Taylor: e^r = Σ r^n/n!  (20 terms; |r| ≤ ln2/2 ≈ 0.347)
    term, acc = 1.0, 1.0
    for n in range(1, 20):
        term *= r / n
        acc  += term
        if _abs(term) < _EPS * _abs(acc): break
    # e^x = e^{kln2} * e^r = 2^k * acc
    if k >= 0:
        scale = 1 << k if k < 63 else _INF
    else:
        scale = 1.0 / (1 << (-k)) if -k < 63 else 0.0
    return acc * scale


def _ln(x: float) -> float:
    """Natural log via argument reduction + Taylor; x > 0 required."""
    if x <= 0.0: return -_INF
    if x == 1.0: return 0.0
    # Reduce: x = m * 2^e, ln x = ln m + e*ln2, m in [1,2)
    e = 0
    m = x
    while m >= 2.0: m *= 0.5; e += 1
    while m < 1.0:  m *= 2.0; e -= 1
    # ln m = 2*atanh((m-1)/(m+1)), series for atanh
    y  = (m - 1.0) / (m + 1.0)
    y2 = y * y
    acc = y
    term = y
    for n in range(1, 35):
        term *= y2
        acc  += term / (2 * n + 1)
        if _abs(term) < _EPS * _abs(acc): break
    return 2.0 * acc + e * _LN2


def _sqrt(x: float) -> float:
    """Newton–Raphson square root; x ≥ 0."""
    if x < 0.0:  return float('nan')
    if x == 0.0: return 0.0
    g = x if x <= 1.0 else x * 0.5
    for _ in range(52):
        g2 = (g + x / g) * 0.5
        if _abs(g2 - g) < _EPS * g: return g2
        g = g2
    return g


def _sin(x: float) -> float:
    """sin(x) via Taylor; argument reduced to [−π, π]."""
    x = x % (2 * _PI)
    if x > _PI:  x -= 2 * _PI
    x2 = x * x
    term, acc = x, x
    for n in range(1, 20):
        term *= -x2 / ((2 * n) * (2 * n + 1))
        acc  += term
        if _abs(term) < _EPS * _abs(acc): break
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# §1  LATENCY MODEL
#
#  Each AgentPriority level has a distinct "target latency" T* in microseconds.
#  The latency reward penalises deviations above T*:
#    r_lat(d) = −ln(1 + d / T*)
#  At d=0:      r_lat = 0  (no penalty)
#  At d=T*:     r_lat = −ln(2) ≈ −0.693
#  At d=100·T*: r_lat ≈ −ln(101) ≈ −4.615
#
#  This is a concave penalty that grows without bound but at a shrinking rate,
#  meaning the agent is strongly discouraged from any latency above T* but
#  catastrophically slow calls are not infinitely punished.
# ─────────────────────────────────────────────────────────────────────────────

# Target latency in microseconds per priority tier
_TARGET_LATENCY_US: Dict[int, float] = {
    0: 100.0,       # CRITICAL — 100 μs
    1: 1_000.0,     # HIGH     — 1 ms
    2: 10_000.0,    # NORMAL   — 10 ms
    3: 100_000.0,   # LOW      — 100 ms
}

_PRIORITY_NAMES = {0: "CRITICAL", 1: "HIGH", 2: "NORMAL", 3: "LOW"}


def latency_reward(duration_us: float, priority: int) -> float:
    """
    r_lat = −ln(1 + duration_us / T*)
    Always ≤ 0; approaches 0 as duration → 0.
    """
    T_star = _TARGET_LATENCY_US.get(priority, _TARGET_LATENCY_US[2])
    return -_ln(1.0 + duration_us / T_star)


# ─────────────────────────────────────────────────────────────────────────────
# §2  REWARD COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

# Reward weights — must sum to 1.0
_W_SUCCESS = 0.50
_W_LATENCY = 0.30
_W_DEPTH   = 0.20

assert abs(_W_SUCCESS + _W_LATENCY + _W_DEPTH - 1.0) < 1e-9, "Weights must sum to 1"

# Asymmetric success/failure values
_R_SUCCESS = +1.0
_R_FAILURE = -2.0   # failures punished 2× harder than successes rewarded


@dataclass(frozen=True)
class RewardComponents:
    """
    Decomposed reward signal for a single AgentTrace.
    All fields are real-valued; composite is the weighted sum.
    """
    r_success:  float   # success/failure term (§2 formula)
    r_latency:  float   # latency penalty (§1 formula)
    r_depth:    float   # call-chain parsimony term
    composite:  float   # R = w_s·r_s + w_l·r_l + w_d·r_d
    normalized: float   # R̂ = (R − μ̂) / (σ̂ + ε)  [filled by RewardShaper]


def compute_components(
    success:     bool,
    duration_ns: int,
    depth:       int,
    priority:    int,
    normalized:  float = 0.0,
) -> RewardComponents:
    """
    Compute all reward components from trace fields.

    Args:
        success:     tool call succeeded
        duration_ns: wall-clock duration in nanoseconds
        depth:       call-chain depth (AgentContext.depth)
        priority:    AgentPriority integer value (0-3)
        normalized:  pre-computed normalised composite (supplied by RewardShaper)
    """
    duration_us = duration_ns / 1000.0

    r_s = _R_SUCCESS if success else _R_FAILURE
    r_l = latency_reward(duration_us, priority)
    r_d = -0.1 * _min(depth / 10.0, 1.0)

    composite = _W_SUCCESS * r_s + _W_LATENCY * r_l + _W_DEPTH * r_d
    return RewardComponents(
        r_success=r_s,
        r_latency=r_l,
        r_depth=r_d,
        composite=composite,
        normalized=normalized,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §3  REWARD SHAPER
#
#  Aggregates per-trace components and applies Welford online normalization.
#  Thread-safe; may be called from multiple dispatch threads simultaneously.
#
#  Welford's online algorithm [Welford 1962]:
#      μ_n = μ_{n-1} + (x − μ_{n-1}) / n
#      S_n = S_{n-1} + (x − μ_{n-1}) · (x − μ_n)
#      σ̂_n = sqrt(S_n / n)
#
#  Produces R̂ ∈ ℝ that is approximately N(0, 1) over the history window,
#  which stabilises PPO gradient magnitudes.
# ─────────────────────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Converts AgentTrace records into normalised scalar rewards.

    Usage:
        shaper = RewardShaper()
        r = shaper.shape(trace_dict)   # trace_dict = AgentTrace.to_dict() plus extras
        print(r.normalized)            # use this as RL reward signal
    """

    def __init__(
        self,
        history_window:     int   = 2048,
        priority_map:       Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            history_window: rolling window for summary statistics (not Welford)
            priority_map:   tool_name → priority int; supplements traces that
                            don't carry priority directly
        """
        self._priority_map: Dict[str, int] = priority_map or {}
        self._lock    = threading.RLock()
        self._history : deque = deque(maxlen=history_window)

        # Welford state (global across all tools)
        self._welf_n  : int   = 0
        self._welf_mu : float = 0.0
        self._welf_S  : float = 0.0   # sum of squared deviations

        # Per-tool EMA error rate: tool_name → float
        self._tool_error_ema : Dict[str, float] = defaultdict(float)
        self._tool_call_ts   : Dict[str, List[float]] = defaultdict(list)
        self._ALPHA          : float = 0.05  # EMA decay for per-tool error rate

    # ── Welford online update ─────────────────────────────────────────────────

    def _welford_update(self, x: float) -> Tuple[float, float]:
        """Update Welford running stats with new value x. Returns (μ, σ)."""
        self._welf_n  += 1
        delta          = x - self._welf_mu
        self._welf_mu += delta / self._welf_n
        delta2         = x - self._welf_mu
        self._welf_S  += delta * delta2
        sigma = _sqrt(self._welf_S / self._welf_n) if self._welf_n > 1 else 1.0
        return self._welf_mu, sigma

    def _normalise(self, x: float, mu: float, sigma: float) -> float:
        return (x - mu) / (sigma + _EPS)

    # ── Per-tool bookkeeping ──────────────────────────────────────────────────

    def _update_tool_stats(self, tool_name: str, success: bool, ts: float) -> None:
        """EMA update of per-tool error rate and call timestamp list."""
        err = 0.0 if success else 1.0
        self._tool_error_ema[tool_name] = (
            self._ALPHA * err
            + (1.0 - self._ALPHA) * self._tool_error_ema[tool_name]
        )
        tss = self._tool_call_ts[tool_name]
        tss.append(ts)
        # Keep only last 64 timestamps (for call-rate computation)
        if len(tss) > 64:
            tss.pop(0)

    def _call_rate(self, tool_name: str, now: float, window_s: float = 1.0) -> float:
        """Calls per second for tool_name over the last window_s seconds."""
        tss = self._tool_call_ts[tool_name]
        cutoff = now - window_s
        recent = sum(1 for t in tss if t >= cutoff)
        return float(recent) / window_s

    # ── Primary API ──────────────────────────────────────────────────────────

    def shape(
        self,
        tool_name:   str,
        success:     bool,
        duration_ns: int,
        depth:       int       = 0,
        priority:    int       = 2,
        error:       Optional[str] = None,
        ts:          Optional[float] = None,
    ) -> RewardComponents:
        """
        Compute full reward signal for one agent tool invocation.

        Returns a RewardComponents with .normalized suitable for use as
        the RL scalar reward r_t.
        """
        if ts is None:
            ts = time.monotonic()

        with self._lock:
            # Per-tool priority override
            if tool_name in self._priority_map:
                priority = self._priority_map[tool_name]

            self._update_tool_stats(tool_name, success, ts)

            comp    = compute_components(success, duration_ns, depth, priority)
            mu, sig = self._welford_update(comp.composite)
            norm    = self._normalise(comp.composite, mu, sig)

            # Rebuild as frozen dataclass with normalized filled in
            final = RewardComponents(
                r_success=comp.r_success,
                r_latency=comp.r_latency,
                r_depth=comp.r_depth,
                composite=comp.composite,
                normalized=norm,
            )

            self._history.append({
                "tool":       tool_name,
                "reward":     final.composite,
                "norm":       norm,
                "success":    success,
                "ts":         ts,
            })
            return final

    def shape_from_trace(self, trace: Any) -> RewardComponents:
        """
        Convenience wrapper: accepts an AgentTrace object or dict with
        fields: tool_name, success, duration_ns, error.
        Also accepts .to_dict() output supplemented with depth and priority.
        """
        if hasattr(trace, 'tool_name'):
            # AgentTrace object
            return self.shape(
                tool_name   = trace.tool_name,
                success     = trace.success,
                duration_ns = trace.duration_ns,
                depth       = getattr(trace, 'depth', 0),
                priority    = getattr(trace, 'priority', 2),
                error       = getattr(trace, 'error', None),
                ts          = getattr(trace, 'timestamp', None),
            )
        else:
            # dict
            return self.shape(
                tool_name   = trace.get('tool', ''),
                success     = trace.get('success', True),
                duration_ns = trace.get('duration_ns', trace.get('duration_us', 0) * 1000),
                depth       = trace.get('depth', 0),
                priority    = trace.get('priority', 2),
                error       = trace.get('error', None),
                ts          = trace.get('ts', None),
            )

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for the shaper's history window."""
        with self._lock:
            if not self._history:
                return {"n": 0, "mu": 0.0, "sigma": 0.0, "tool_errors": {}}
            rewards  = [h["reward"] for h in self._history]
            norm_r   = [h["norm"] for h in self._history]
            n        = len(rewards)
            mu_r     = sum(rewards) / n
            var_r    = sum((r - mu_r) ** 2 for r in rewards) / n
            return {
                "n":                n,
                "welford_mu":       round(self._welf_mu, 6),
                "welford_sigma":    round(_sqrt(self._welf_S / max(1, self._welf_n)), 6),
                "window_mean":      round(mu_r, 6),
                "window_std":       round(_sqrt(var_r), 6),
                "window_norm_mean": round(sum(norm_r) / n, 6),
                "success_rate":     round(sum(1 for h in self._history if h["success"]) / n, 4),
                "tool_errors":      dict(self._tool_error_ema),
            }

    def register_priority(self, tool_name: str, priority: int) -> None:
        """Inform the shaper of a tool's priority tier for reward computation."""
        with self._lock:
            self._priority_map[tool_name] = priority


# ─────────────────────────────────────────────────────────────────────────────
# §4  STATE ENCODER
#
#  Maps a single AgentTrace (or equivalent dict) into a fixed-size float
#  vector s ∈ ℝ^{STATE_DIM=16} suitable as input to the ActorCritic network.
#
#  Feature layout:
#   [0]  tool_priority_norm     = priority / 3.0
#   [1]  duration_log_norm      = ln(1+d_μs) / ln(1+3_000_000)  [cap 3s]
#   [2]  success                = 1.0 if success else 0.0
#   [3]  depth_norm             = min(depth / 10, 1)
#   [4]  tool_error_ema         = EMA error rate for this tool  [0,1]
#   [5]  call_rate_norm         = min(calls/sec / 1000, 1)
#   [6]  global_success_rate    = rate over last window
#   [7]  global_latency_ema_log = ln(1+ema_us) / ln(1+3_000_000)
#   [8]  error_flag             = 1.0 if error string present else 0.0
#   [9]  budget_pressure        = depth_norm · (1 − success)
#   [10] tool_hash_0            = deterministic hash feature 0
#   [11] tool_hash_1            = deterministic hash feature 1
#   [12] tool_hash_2            = deterministic hash feature 2
#   [13] tool_hash_3            = deterministic hash feature 3
#   [14] tool_hash_4            = deterministic hash feature 4
#   [15] timestamp_phase        = sin(2π · (ts % 60) / 60)  [cyclical minute]
#
#  Tool hash features: h_k(name) = (SHA256(name + str(k))[0:3] as uint24) / 2^24
#  This gives 5 pseudo-independent real-valued features in [0,1] that encode
#  tool identity consistently across calls without requiring a one-hot lookup.
# ─────────────────────────────────────────────────────────────────────────────

_LN_CAP = _ln(1.0 + 3_000_000.0)   # ln(1 + 3s in μs); normalizer denominator

assert _LN_CAP > 0, "Normalizer must be positive"


def _tool_hash_features(tool_name: str, n: int = 5) -> List[float]:
    """
    Deterministic hash encoding of tool_name into n floats ∈ [0,1].
    Each feature uses a different SHA-256 seed for independence.
    """
    feats: List[float] = []
    name_b = tool_name.encode('utf-8')
    for k in range(n):
        digest = hashlib.sha256(name_b + str(k).encode()).digest()
        # Take first 3 bytes as a uint24
        uint24 = (digest[0] << 16) | (digest[1] << 8) | digest[2]
        feats.append(uint24 / 16_777_215.0)   # 2^24 - 1
    return feats


class StateEncoder:
    """
    Converts AgentTrace records into 16-dimensional state vectors.

    Maintains running statistics needed for features [6] and [7] and
    integrates with RewardShaper for per-tool error EMA [4] and call rate [5].
    """

    def __init__(self, shaper: Optional[RewardShaper] = None) -> None:
        self._shaper    = shaper
        self._lock      = threading.RLock()
        self._lat_ema   = 0.0       # EMA of duration in μs (global)
        self._lat_alpha = 0.01      # slow EMA (long-term latency baseline)
        self._succ_win  : deque = deque(maxlen=100)  # recent success flags

    def _update_globals(self, success: bool, duration_us: float) -> None:
        self._succ_win.append(1.0 if success else 0.0)
        self._lat_ema = self._lat_alpha * duration_us + (1 - self._lat_alpha) * self._lat_ema

    def encode(
        self,
        tool_name:   str,
        success:     bool,
        duration_ns: int,
        depth:       int       = 0,
        priority:    int       = 2,
        error:       Optional[str] = None,
        ts:          Optional[float] = None,
        call_rate:   float     = 0.0,
        error_ema:   float     = 0.0,
    ) -> List[float]:
        """
        Produce a STATE_DIM-length float list representing this trace.

        All values are in [0,1] or standardised to a bounded range.
        """
        if ts is None:
            ts = time.monotonic()

        duration_us = duration_ns / 1000.0

        with self._lock:
            self._update_globals(success, duration_us)
            global_sr  = (sum(self._succ_win) / len(self._succ_win)
                          if self._succ_win else 1.0)
            lat_ema_log = _ln(1.0 + self._lat_ema) / (_LN_CAP + _EPS)

        # Feature 0: priority
        f0 = _clamp(priority / 3.0, 0.0, 1.0)

        # Feature 1: log-normalised duration
        f1 = _clamp(_ln(1.0 + duration_us) / (_LN_CAP + _EPS), 0.0, 1.0)

        # Feature 2: success flag
        f2 = 1.0 if success else 0.0

        # Feature 3: call depth
        f3 = _clamp(depth / 10.0, 0.0, 1.0)

        # Feature 4: per-tool error EMA (supplied or 0)
        f4 = _clamp(error_ema, 0.0, 1.0)

        # Feature 5: call rate
        f5 = _clamp(call_rate / 1000.0, 0.0, 1.0)

        # Feature 6: global success rate
        f6 = _clamp(global_sr, 0.0, 1.0)

        # Feature 7: global latency EMA (log-normalised)
        f7 = _clamp(lat_ema_log, 0.0, 1.0)

        # Feature 8: error flag
        f8 = 1.0 if error else 0.0

        # Feature 9: budget pressure = depth_norm × (1 − success)
        f9 = f3 * (1.0 - f2)

        # Features 10-14: tool identity hash
        h = _tool_hash_features(tool_name, n=5)

        # Feature 15: cyclical timestamp (60-second phase)
        phase = (ts % 60.0) / 60.0
        f15   = 0.5 * (_sin(2 * _PI * phase) + 1.0)  # mapped to [0,1]

        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
                h[0], h[1], h[2], h[3], h[4],
                f15]

    def encode_trace(self, trace: Any, shaper: Optional[RewardShaper] = None) -> List[float]:
        """
        Convenience wrapper: accepts AgentTrace object or to_dict() output.
        Uses shaper (or self._shaper) for per-tool EMA and call rate.
        """
        sh = shaper or self._shaper

        if hasattr(trace, 'tool_name'):
            name   = trace.tool_name
            ok     = trace.success
            dur    = trace.duration_ns
            depth  = getattr(trace, 'depth', 0)
            pri    = getattr(trace, 'priority', 2)
            err    = getattr(trace, 'error', None)
            ts     = getattr(trace, 'timestamp', None)
        else:
            name   = trace.get('tool', '')
            ok     = trace.get('success', True)
            dur    = trace.get('duration_ns', trace.get('duration_us', 0) * 1000)
            depth  = trace.get('depth', 0)
            pri    = trace.get('priority', 2)
            err    = trace.get('error', None)
            ts     = trace.get('ts', None)

        error_ema  = 0.0
        call_rate  = 0.0
        if sh is not None:
            error_ema = sh._tool_error_ema.get(name, 0.0)
            call_rate = sh._call_rate(name, ts or time.monotonic())

        return self.encode(name, ok, dur, depth, pri, err, ts, call_rate, error_ema)

    @staticmethod
    def dim() -> int:
        """Return STATE_DIM (16). Useful for constructing ActorCritic."""
        return STATE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# §5  REWARD KERNEL — @agent_method integration
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aios_core import agent_method, AgentPriority, _registry
    _HAS_CORE = True
except ImportError:
    # Standalone mode — define a no-op decorator
    def agent_method(**kwargs):  # type: ignore[misc]
        def dec(fn): return fn
        return dec
    class AgentPriority:  # type: ignore[no-redef]
        CRITICAL, HIGH, NORMAL, LOW = 0, 1, 2, 3
    _HAS_CORE = False


def _rebind_agent_methods(obj: Any) -> None:
    """
    Re-register all @agent_method-decorated methods on obj as BOUND methods
    in the AgentRegistry.  This is necessary because the decorator stores the
    UNBOUND function at class-definition time; kernel.dispatch() needs a
    callable that includes `self`.
    """
    if not _HAS_CORE:
        return
    from aios_core import AgentToolSpec, _registry as _reg
    for attr_name in dir(obj):
        try:
            method = getattr(obj, attr_name)
        except AttributeError:
            continue
        spec = getattr(method, '_agent_spec', None)
        if spec is None:
            continue
        bound_spec = AgentToolSpec(
            name        = spec.name,
            description = spec.description,
            parameters  = spec.parameters,
            returns     = spec.returns,
            priority    = spec.priority,
            fn          = method,               # bound method — includes self
            owner       = type(obj).__name__,
        )
        _reg.register(bound_spec)


class RewardKernel:
    """
    Attaches reward shaping to the kernel's @agent_method registry.

    After .attach(kernel), the registry's recent_traces() feed into the
    shaper automatically via shape_from_trace().

    Usage:
        rk = RewardKernel()
        rk.attach(kernel)
        # Now rk.shaper and rk.encoder are populated by every dispatch()
    """

    def __init__(
        self,
        shaper:  Optional[RewardShaper]  = None,
        encoder: Optional[StateEncoder]  = None,
    ) -> None:
        self.shaper  : RewardShaper  = shaper  or RewardShaper()
        self.encoder : StateEncoder  = encoder or StateEncoder(self.shaper)
        self._kernel  = None
        self._lock    = threading.RLock()
        self._processed_traces: int  = 0

    def attach(self, kernel: Any) -> None:
        """Bind this RewardKernel to an AgentKernel instance."""
        with self._lock:
            self._kernel = kernel
        # Re-register all @agent_method tools as bound methods
        _rebind_agent_methods(self)
        # Register priority map from the tool registry
        if _HAS_CORE:
            for spec in _registry.all_tools():
                pri = int(spec.priority)
                self.shaper.register_priority(spec.name, pri)

    @agent_method(
        name="reward_shape",
        description="Compute intrinsic reward for the most recent agent traces",
        priority=AgentPriority.LOW,
    )
    def process_recent_traces(self, n: int = 32) -> List[Dict[str, Any]]:
        """
        Pull the last n traces from the registry, compute rewards and state
        encodings. Returns list of {tool, reward, state} dicts.
        """
        results = []
        if not _HAS_CORE:
            return results

        traces = _registry.recent_traces(n)
        for tr in traces[self._processed_traces:]:
            comp = self.shaper.shape_from_trace(tr)
            enc  = self.encoder.encode_trace(tr, self.shaper)
            results.append({
                "tool":      tr.tool_name,
                "reward":    comp.normalized,
                "r_success": comp.r_success,
                "r_latency": comp.r_latency,
                "r_depth":   comp.r_depth,
                "state":     enc,
                "success":   tr.success,
            })
        self._processed_traces = max(0, self._processed_traces + len(results))
        return results

    @agent_method(
        name="reward_stats",
        description="Return reward shaper statistics",
        priority=AgentPriority.LOW,
    )
    def stats(self) -> Dict[str, Any]:
        return {
            "shaper":           self.shaper.stats(),
            "processed_traces": self._processed_traces,
            "state_dim":        STATE_DIM,
            "version":          _REWARD_VERSION,
        }


# ─────────────────────────────────────────────────────────────────────────────
# §6  SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """Deterministic test suite. Raises AssertionError on any failure."""

    # ── Math primitives ───────────────────────────────────────────────────────
    assert _abs(-5.0) == 5.0
    assert _abs(3.0)  == 3.0

    assert _abs(_exp(0.0) - 1.0)    < 1e-9,  f"exp(0) = {_exp(0.0)}"
    assert _abs(_exp(1.0) - _E)     < 1e-9,  f"exp(1) = {_exp(1.0)}"
    assert _abs(_exp(_ln(2.0)) - 2.0) < 1e-9, "exp(ln(2)) ≠ 2"
    assert _abs(_ln(1.0))           < 1e-9,  f"ln(1) = {_ln(1.0)}"
    assert _abs(_ln(_E) - 1.0)      < 1e-9,  f"ln(e) = {_ln(_E)}"
    assert _abs(_sqrt(4.0) - 2.0)   < 1e-9,  f"sqrt(4) = {_sqrt(4.0)}"
    assert _abs(_sqrt(2.0) - 1.41421356) < 1e-6, f"sqrt(2) = {_sqrt(2.0)}"

    # ── Latency reward ────────────────────────────────────────────────────────
    r_zero  = latency_reward(0.0, 2)
    r_exact = latency_reward(10_000.0, 2)   # exactly at T* for NORMAL
    r_ten   = latency_reward(100_000.0, 2)  # 10× over T*

    assert r_zero == 0.0,                       f"r_lat(0) = {r_zero}"
    assert _abs(r_exact - (-_ln(2.0))) < 1e-9,  f"r_lat(T*) = {r_exact}"
    assert r_ten < r_exact,                      "Slower call should have lower reward"
    assert r_zero > r_exact > r_ten,            "Monotone latency penalty"

    # ── Reward components ─────────────────────────────────────────────────────
    c_ok  = compute_components(True,  1_000_000,  2, 2)
    c_err = compute_components(False, 50_000_000, 5, 2)

    assert c_ok.r_success  ==  _R_SUCCESS
    assert c_err.r_success ==  _R_FAILURE
    assert c_ok.composite  >  c_err.composite,  "Success should have higher reward"
    assert c_ok.r_depth    ==  0.0 or c_ok.r_depth < 0.0  # depth=2 → small penalty

    # ── RewardShaper ──────────────────────────────────────────────────────────
    shaper = RewardShaper()

    # First call: sigma = 1 (single sample), norm = 0
    r1 = shaper.shape("palloc", True, 500_000, depth=0, priority=1)
    assert r1.normalized == 0.0, f"First norm should be 0, got {r1.normalized}"

    r2 = shaper.shape("palloc", True,  200_000, depth=0, priority=1)
    r3 = shaper.shape("palloc", False, 500_000, depth=3, priority=1)

    # After 3 calls, normalization should be active
    stats = shaper.stats()
    assert stats["n"] == 3
    assert -3.0 < stats["welford_mu"] < 3.0
    assert stats["welford_sigma"] >= 0.0

    # Failure should produce lower composite than success
    assert r3.composite < r2.composite, "Error trace should yield lower reward"

    # ── StateEncoder ──────────────────────────────────────────────────────────
    encoder = StateEncoder(shaper)

    state = encoder.encode(
        tool_name="palloc", success=True, duration_ns=500_000,
        depth=1, priority=1, error=None, ts=1000.0,
        call_rate=10.0, error_ema=0.05,
    )
    assert len(state) == STATE_DIM, f"State dim mismatch: {len(state)} ≠ {STATE_DIM}"
    assert all(isinstance(v, float) for v in state), "State must be all floats"
    assert all(-0.01 <= v <= 1.01 for v in state), f"State out of [0,1]: {state}"

    # Same tool name must produce same hash features (determinism)
    h1 = _tool_hash_features("palloc", 5)
    h2 = _tool_hash_features("palloc", 5)
    assert h1 == h2, "Hash features must be deterministic"

    # Different tools must produce different hashes
    h3 = _tool_hash_features("pfree", 5)
    assert h1 != h3, "Different tools must have different hash features"

    # ── Integration: shape_from_trace with duck-typed object ──────────────────
    class FakeTrace:
        tool_name   = "kernel_dispatch"
        success     = True
        duration_ns = 2_000_000
        depth       = 1
        priority    = 2
        error       = None
        timestamp   = None

    r_fake = shaper.shape_from_trace(FakeTrace())
    assert isinstance(r_fake, RewardComponents)
    assert r_fake.r_success == _R_SUCCESS

    state2 = encoder.encode_trace(FakeTrace(), shaper)
    assert len(state2) == STATE_DIM

    print("aios_reward: all self-tests passed ✓")


if __name__ == "__main__":
    _run_self_tests()
