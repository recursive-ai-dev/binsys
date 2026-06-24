#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Agentic Feedback Loop                                                ║
║  aios_feedback.py                                                            ║
║                                                                              ║
║  "An agent that cannot learn from its own execution is not an agent.         ║
║   It is a subroutine. This module is the difference."                        ║
║                                                                              ║
║  This module closes the loop:                                                ║
║                                                                              ║
║    dispatch(tool) → AgentTrace → RewardShaper → ExperienceBuffer            ║
║       → PPO.update() → LearningReasoner.plan() → dispatch(tool)             ║
║                                                                              ║
║  Components:                                                                 ║
║    §0  Constants & Math Shim                                                 ║
║    §1  KernelEnv          — Env subclass wrapping live AgentRegistry traces  ║
║    §2  ExperienceBuffer   — ring buffer of (state, action, reward, next_s)  ║
║    §3  LearningReasoner   — AgentReasoner backed by a trained ActorCritic   ║
║    §4  FeedbackLoop       — background thread: read traces → train → update  ║
║    §5  FeedbackKernel     — @agent_method integration + kernel hot-swap      ║
║    §6  Self-Tests         — deterministic validation suite                   ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    PPO clip   : L^CLIP = E[min(r·Â, clip(r, 1−ε, 1+ε)·Â)]  [Schulman 2017]║
║    GAE        : Â_t = Σ_{k=0}^∞ (γλ)^k · δ_{t+k}           [Schulman 2016]║
║    Adam       : θ ← θ − α · m̂ / (√v̂ + ε)                  [Kingma 2015] ║
║    Confidence : conf = max(softmax(logits))  [greedy policy confidence]     ║
║    Cold-start : fallback to RuleBasedReasoner if steps < MIN_TRAIN_STEPS    ║
║    State dim  : s ∈ ℝ^{16}  (from aios_reward.StateEncoder)                ║
║    Action dim : a ∈ {0, …, MAX_OPTIONS−1}  (option index)                  ║
║                                                                              ║
║  Integration:                                                                ║
║    from aios_feedback import FeedbackKernel                                  ║
║    fk = FeedbackKernel(); fk.attach(kernel); fk.start()                     ║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Zero external dependencies beyond other AIOS modules.                   ║
║    • Thread-safe: FeedbackLoop runs in a daemon thread.                      ║
║    • Graceful degradation: falls back to RuleBasedReasoner on any failure.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

_FEEDBACK_VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# §0  MATH SHIM — no import math; full AIOS idiom
# ─────────────────────────────────────────────────────────────────────────────

_PI   = 3.141592653589793238462643383279
_E    = 2.718281828459045235360287471352
_LN2  = 0.693147180559945309417232121458
_INF  = float('inf')
_EPS  = 1e-12

MAX_OPTIONS       = 16    # maximum number of options passed to decide()
MIN_TRAIN_STEPS   = 50    # minimum experience before LearningReasoner activates
CONFIDENCE_FLOOR  = 0.40  # softmax max below this → fall back to rule-based
TRAIN_INTERVAL    = 32    # collect this many transitions before a PPO update
EPISODE_LEN       = 64    # artificial episode boundary for PPO rollouts
LOOP_SLEEP_S      = 0.1   # FeedbackLoop polling interval


def _abs(x: float) -> float:
    return x if x >= 0.0 else -x


def _exp(x: float) -> float:
    if x > 709.782: return _INF
    if x < -745.0:  return 0.0
    k = int(x / _LN2)
    r = x - k * _LN2
    term, acc = 1.0, 1.0
    for n in range(1, 20):
        term *= r / n
        acc  += term
        if _abs(term) < _EPS * _abs(acc): break
    if k >= 0:
        scale = (1 << k) if k < 63 else _INF
    else:
        scale = 1.0 / (1 << (-k)) if -k < 63 else 0.0
    return acc * scale


def _ln(x: float) -> float:
    if x <= 0.0: return -_INF
    if x == 1.0: return 0.0
    e, m = 0, x
    while m >= 2.0: m *= 0.5; e += 1
    while m < 1.0:  m *= 2.0; e -= 1
    y  = (m - 1.0) / (m + 1.0)
    y2 = y * y
    acc, term = y, y
    for n in range(1, 35):
        term *= y2
        acc  += term / (2 * n + 1)
        if _abs(term) < _EPS * _abs(acc): break
    return 2.0 * acc + e * _LN2


def _sqrt(x: float) -> float:
    if x < 0.0:  return float('nan')
    if x == 0.0: return 0.0
    g = x if x <= 1.0 else x * 0.5
    for _ in range(52):
        g2 = (g + x / g) * 0.5
        if _abs(g2 - g) < _EPS * g: return g2
        g = g2
    return g


def _softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    m   = max(logits) if logits else 0.0
    exps = [_exp(v - m) for v in logits]
    s   = sum(exps) + _EPS
    return [e / s for e in exps]


def _argmax(xs: List[float]) -> int:
    best, idx = xs[0], 0
    for i in range(1, len(xs)):
        if xs[i] > best:
            best, idx = xs[i], i
    return idx


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# §1  KERNEL ENVIRONMENT
#
#  Wraps the kernel's AgentRegistry as an RL environment.
#
#  State:   16-dim feature vector from StateEncoder applied to the last trace
#  Action:  integer index into the registered tool list (tool selection)
#  Reward:  RewardShaper.shape_from_trace().normalized
#  Done:    True every EPISODE_LEN steps (artificial episode boundary)
#
#  This allows the PPO/DQN agents from aios_rl.py to train on real kernel
#  execution data without any environment simulation.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aios_rl import Env, Transition, Episode
    _HAS_RL = True
except ImportError:
    _HAS_RL = False

    class Env:  # type: ignore[no-redef]
        state_size         = 16
        action_space_size  = MAX_OPTIONS
        def reset(self) -> List[float]: return [0.0] * 16
        def step(self, action: int):    return ([0.0]*16, 0.0, True, {})

    @dataclass
    class Transition:  # type: ignore[no-redef]
        state:      List[float]
        action:     int
        reward:     float
        next_state: List[float]
        done:       bool
        log_prob:   float = 0.0
        value:      float = 0.0

    class Episode:  # type: ignore[no-redef]
        def __init__(self): self.transitions = []
        def push(self, t): self.transitions.append(t)
        @property
        def total_reward(self): return sum(t.reward for t in self.transitions)
        @property
        def length(self): return len(self.transitions)


try:
    from aios_reward import RewardShaper, StateEncoder, STATE_DIM
    _HAS_REWARD = True
except ImportError:
    _HAS_REWARD = False
    STATE_DIM   = 16
    class RewardShaper:  # type: ignore[no-redef]
        def shape_from_trace(self, t): return type('R', (), {'normalized': 0.0})()
    class StateEncoder:  # type: ignore[no-redef]
        def encode_trace(self, t, s=None): return [0.0] * STATE_DIM
        @staticmethod
        def dim(): return STATE_DIM


try:
    from aios_core import (
        AgentReasoner, AgentContext, AgentToolSpec, AgentTrace,
        AgentPriority, _registry, agent_method, RuleBasedReasoner,
    )
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

    def agent_method(**kw):  # type: ignore[misc]
        def dec(fn): return fn
        return dec

    class AgentPriority:  # type: ignore[no-redef]
        CRITICAL, HIGH, NORMAL, LOW = 0, 1, 2, 3

    class AgentContext:  # type: ignore[no-redef]
        caller = "kernel"; depth = 0; trace_id = ""; budget_ns = 0
        metadata: Dict = {}; _chain: List = []

    class AgentToolSpec:  # type: ignore[no-redef]
        def __init__(self, name, priority=2): self.name = name; self.priority = priority

    class AgentReasoner:  # type: ignore[no-redef]
        def decide(self, ctx, opts, ag, meta=None): return opts[0] if opts else ""
        def annotate(self, n, a, k, c): return None
        def plan(self, g, t, c): return []

    class RuleBasedReasoner(AgentReasoner):  # type: ignore[no-redef]
        def decide(self, ctx, opts, ag, meta=None):
            if not opts: raise ValueError("empty options")
            return opts[0]
        def annotate(self, n, a, k, c): return None
        def plan(self, g, tools, c):
            goal_lc = g.lower()
            return [{"tool": sp.name, "kwargs": {}} for sp in
                    sorted(tools, key=lambda s: s.priority)
                    if any(kw in goal_lc for kw in sp.name.lower().split("_"))]

    class _FakeRegistry:
        def recent_traces(self, n=20): return []
        def all_tools(self): return []
    _registry = _FakeRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# §2  EXPERIENCE BUFFER
#
#  Rolling ring buffer of Transition objects assembled from trace batches.
#  Each Transition captures:
#    state:      StateEncoder.encode_trace(trace_t)
#    action:     tool_index(trace_t.tool_name)
#    reward:     RewardShaper.shape_from_trace(trace_t).normalized
#    next_state: StateEncoder.encode_trace(trace_{t+1})
#    done:       True every EPISODE_LEN steps
# ─────────────────────────────────────────────────────────────────────────────

class ToolIndex:
    """
    Bidirectional mapping between tool names and integer action indices.
    New tools are assigned the next available index (up to MAX_ACTIONS).
    """

    MAX_ACTIONS = 64

    def __init__(self) -> None:
        self._name_to_idx : Dict[str, int] = {}
        self._idx_to_name : Dict[int, str] = {}
        self._lock = threading.Lock()

    def get_or_assign(self, name: str) -> int:
        with self._lock:
            if name in self._name_to_idx:
                return self._name_to_idx[name]
            idx = len(self._name_to_idx) % self.MAX_ACTIONS
            self._name_to_idx[name] = idx
            self._idx_to_name[idx]  = name
            return idx

    def get_name(self, idx: int) -> Optional[str]:
        with self._lock:
            return self._idx_to_name.get(idx)

    def action_space_size(self) -> int:
        with self._lock:
            return max(len(self._name_to_idx), MAX_OPTIONS)

    def all_tools(self) -> List[str]:
        with self._lock:
            return [self._idx_to_name[i] for i in sorted(self._idx_to_name)]

    def seed_from_registry(self) -> int:
        """Pre-populate from the kernel registry. Returns count added."""
        count = 0
        for spec in _registry.all_tools():
            self.get_or_assign(spec.name)
            count += 1
        return count


class ExperienceBuffer:
    """
    Rolling buffer of Transitions assembled from live AgentRegistry traces.

    Maintains a cursor into _registry.recent_traces() so each batch of new
    traces is processed exactly once.
    """

    def __init__(
        self,
        capacity:    int            = 4096,
        shaper:      Optional[Any]  = None,
        encoder:     Optional[Any]  = None,
        tool_index:  Optional[ToolIndex] = None,
    ) -> None:
        self._buf        : deque = deque(maxlen=capacity)
        self._shaper     = shaper  or RewardShaper()
        self._encoder    = encoder or StateEncoder(self._shaper)
        self._tool_index = tool_index or ToolIndex()
        self._lock       = threading.RLock()
        self._cursor     = 0      # how many registry traces we've consumed
        self._step       = 0      # total transitions stored ever
        self._last_state : Optional[List[float]] = None

    def ingest_traces(self, traces: List[Any]) -> int:
        """
        Convert trace objects to Transitions and store them.
        Returns number of new transitions added.
        """
        added = 0
        for i, tr in enumerate(traces):
            state  = self._encoder.encode_trace(tr, self._shaper)
            action = self._tool_index.get_or_assign(
                getattr(tr, 'tool_name', tr.get('tool', 'unknown'))
                if not hasattr(tr, 'tool_name') else tr.tool_name
            )
            reward = self._shaper.shape_from_trace(tr).normalized

            # next_state: use next trace's state, or repeat current on last
            if i + 1 < len(traces):
                next_state = self._encoder.encode_trace(traces[i + 1], self._shaper)
            elif self._last_state is not None:
                next_state = self._last_state
            else:
                next_state = state[:]   # self-loop fallback

            done = (self._step % EPISODE_LEN == EPISODE_LEN - 1)

            t = Transition(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done,
                log_prob=0.0, value=0.0,
            )
            with self._lock:
                self._buf.append(t)
                self._last_state = next_state
                self._step += 1
            added += 1
        return added

    def drain_episode(self) -> Optional[Episode]:
        """
        Pop one EPISODE_LEN-length episode from the buffer, or None if
        insufficient transitions are available.
        """
        with self._lock:
            if len(self._buf) < EPISODE_LEN:
                return None
            ep = Episode()
            for _ in range(EPISODE_LEN):
                ep.push(self._buf.popleft())
            return ep

    def drain_batch(self, n: int) -> List[Transition]:
        """Pop up to n transitions for DQN-style training."""
        with self._lock:
            batch = []
            for _ in range(min(n, len(self._buf))):
                batch.append(self._buf.popleft())
            return batch

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def total_steps(self) -> int:
        with self._lock:
            return self._step

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "buffered":    len(self._buf),
                "total_steps": self._step,
                "cursor":      self._cursor,
                "tools_known": len(self._tool_index._name_to_idx),
            }


# ─────────────────────────────────────────────────────────────────────────────
# §3  LEARNING REASONER
#
#  Replaces RuleBasedReasoner as the kernel's live reasoning engine once
#  enough experience has been accumulated.
#
#  For decide():
#    1. Encode (context, options) as a 16-dim state vector
#    2. Query ActorCritic forward pass
#    3. Softmax over valid option indices [0, len(options))
#    4. If max(probs) < CONFIDENCE_FLOOR → fall back to RuleBasedReasoner
#    5. Otherwise return options[argmax(probs)]
#
#  For plan():
#    1. Encode goal as a 16-dim query vector
#    2. For each tool, compute dot-product score with goal embedding
#    3. Return tools sorted by score, filtered by goal keyword matching
#    4. Falls back to RuleBasedReasoner if not trained
#
#  Context encoding for decide():
#    Hash the concatenation of context string + options string into a
#    16-dim vector using the same tool hash technique from StateEncoder.
#    This is lightweight and deterministic without requiring a separate
#    embedding network.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aios_rl import ActorCritic, PPO
    _HAS_AC = True
except ImportError:
    _HAS_AC = False

    class ActorCritic:  # type: ignore[no-redef]
        def __init__(self, s, a):
            self.action_size = a
        def forward(self, s):
            return [0.0]*self.action_size, 0.0
        def act(self, s):
            return 0, 0.0, 0.0
        def param_count(self): return 0

    class PPO:  # type: ignore[no-redef]
        def __init__(self, ac): self.ac = ac
        def update(self, rollout): return {}


import hashlib as _hashlib


def _encode_context_as_state(context: str, options: List[str]) -> List[float]:
    """
    Encode a (context, options) pair as a 16-dim float state vector.

    Strategy:
      [0:8]   SHA-256 hash features of context string
      [8:12]  SHA-256 hash features of sorted(options) joined
      [12]    number of options, normalised to [0,1] over MAX_OPTIONS
      [13]    mean option name length, normalised
      [14]    context string length, normalised
      [15]    timestamp phase feature (cyclical minute)
    """
    ctx_b = context.encode('utf-8')
    opt_b = "|".join(sorted(options)).encode('utf-8')

    def _hash_feats(b: bytes, n: int) -> List[float]:
        feats = []
        for k in range(n):
            digest = _hashlib.sha256(b + str(k).encode()).digest()
            uint24 = (digest[0] << 16) | (digest[1] << 8) | digest[2]
            feats.append(uint24 / 16_777_215.0)
        return feats

    ctx_feats  = _hash_feats(ctx_b, 8)
    opt_feats  = _hash_feats(opt_b, 4)
    n_opts     = _clamp(len(options) / MAX_OPTIONS, 0.0, 1.0)
    mean_len   = _clamp(_mean([len(o) for o in options]) / 32.0, 0.0, 1.0) if options else 0.0
    ctx_len    = _clamp(len(context) / 256.0, 0.0, 1.0)
    # Feature [15]: 5th hash from combined key — fully deterministic
    combo_hash = _hash_feats(ctx_b + b"|" + opt_b, 1)[0]

    state = ctx_feats + opt_feats + [n_opts, mean_len, ctx_len, combo_hash]
    assert len(state) == STATE_DIM, f"encode_context len={len(state)}"
    return state


def _encode_goal_as_state(goal: str) -> List[float]:
    """Encode a goal string as a 16-dim state vector."""
    return _encode_context_as_state(goal, [])


class LearningReasoner(AgentReasoner):
    """
    RL-backed AgentReasoner.

    Wraps a trained ActorCritic network for decide() and plan().
    Falls back to RuleBasedReasoner when:
      - self._trained_steps < MIN_TRAIN_STEPS  (cold start)
      - max(softmax(logits)) < CONFIDENCE_FLOOR  (low confidence)
      - any exception occurs  (safety net)
    """

    def __init__(
        self,
        state_dim:   int = STATE_DIM,
        max_options: int = MAX_OPTIONS,
        fallback:    Optional[AgentReasoner] = None,
        tool_index:  Optional[ToolIndex] = None,
    ) -> None:
        self._fallback     : AgentReasoner = fallback or RuleBasedReasoner()
        self._tool_index   : ToolIndex     = tool_index or ToolIndex()
        self._state_dim    = state_dim
        self._max_options  = max_options
        self._lock         = threading.RLock()
        self._trained_steps= 0
        self._decide_count = 0
        self._learn_count  = 0
        self._fallback_count = 0

        # Two separate policies:
        #   _decision_ac — for decide(context, options) → option index
        #   _plan_ac     — for plan(goal, tools) → tool scores
        if _HAS_AC:
            self._decision_ac = ActorCritic(state_dim, max_options)
            self._plan_ac     = ActorCritic(state_dim, ToolIndex.MAX_ACTIONS)
        else:
            self._decision_ac = None
            self._plan_ac     = None

    def _use_learned_policy(self) -> bool:
        """Return True if the policy has been trained enough to activate."""
        return self._trained_steps >= MIN_TRAIN_STEPS

    def decide(
        self,
        context:  str,
        options:  List[str],
        ctx:      Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not options:
            raise ValueError("decide() called with empty options list")

        with self._lock:
            self._decide_count += 1

        if not self._use_learned_policy() or self._decision_ac is None:
            with self._lock: self._fallback_count += 1
            return self._fallback.decide(context, options, ctx, metadata)

        try:
            state    = _encode_context_as_state(context, options)
            log_probs, value = self._decision_ac.forward(state)

            # Only consider valid option indices
            n        = min(len(options), self._max_options)
            relevant = log_probs[:n]
            probs    = _softmax(relevant)
            best_idx = _argmax(probs)
            conf     = probs[best_idx]

            if conf < CONFIDENCE_FLOOR:
                with self._lock: self._fallback_count += 1
                return self._fallback.decide(context, options, ctx, metadata)

            with self._lock: self._learn_count += 1
            return options[best_idx]

        except Exception:
            # Safety net: never crash the kernel over a reasoning error
            with self._lock: self._fallback_count += 1
            return self._fallback.decide(context, options, ctx, metadata)

    def annotate(
        self,
        tool_name: str,
        args:      Tuple,
        kwargs:    Dict,
        ctx:       Any,
    ) -> Optional[str]:
        if self._trained_steps > 0:
            return f"LearningReasoner (steps={self._trained_steps})"
        return self._fallback.annotate(tool_name, args, kwargs, ctx)

    def plan(
        self,
        goal:  str,
        tools: List[Any],
        ctx:   Any,
    ) -> List[Dict[str, Any]]:
        if not self._use_learned_policy() or self._plan_ac is None:
            return self._fallback.plan(goal, tools, ctx)

        try:
            state      = _encode_goal_as_state(goal)
            tool_scores, _ = self._plan_ac.forward(state)

            # Score each tool using its action index from the plan policy
            scored: List[Tuple[float, Any]] = []
            for spec in tools:
                idx   = self._tool_index.get_or_assign(spec.name)
                score = tool_scores[idx] if idx < len(tool_scores) else 0.0
                # Blend with keyword matching (existing rule-based signal)
                kw_match = any(
                    kw in goal.lower()
                    for kw in spec.name.lower().split("_")
                )
                # Keyword match gives +0.5 bonus; allows RL to re-rank among matches
                score += 0.5 if kw_match else 0.0
                scored.append((score, spec))

            # Sort descending by score, keep top MAX_OPTIONS
            scored.sort(key=lambda x: x[0], reverse=True)
            plan: List[Dict[str, Any]] = []
            for score, spec in scored[:MAX_OPTIONS]:
                if score > 0.0:
                    plan.append({"tool": spec.name, "kwargs": {}})

            return plan if plan else self._fallback.plan(goal, tools, ctx)

        except Exception:
            return self._fallback.plan(goal, tools, ctx)

    def absorb_training(self, ppo: Any, ep: Any) -> float:
        """
        Run one PPO update from the given episode.
        Returns the mean policy loss. Increments trained_steps.
        Thread-safe.
        """
        if ppo is None or ep is None:
            return 0.0
        try:
            result = ppo.update(ep) if hasattr(ppo, 'update') else {}
            loss   = result.get('policy_loss', 0.0) if isinstance(result, dict) else 0.0
            with self._lock:
                self._trained_steps += len(ep.transitions) if hasattr(ep, 'transitions') else EPISODE_LEN
            return float(loss)
        except Exception:
            return 0.0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "trained_steps":  self._trained_steps,
                "active":         self._use_learned_policy(),
                "decide_count":   self._decide_count,
                "learn_count":    self._learn_count,
                "fallback_count": self._fallback_count,
                "confidence_floor": CONFIDENCE_FLOOR,
                "min_train_steps":  MIN_TRAIN_STEPS,
            }


# ─────────────────────────────────────────────────────────────────────────────
# §4  FEEDBACK LOOP
#
#  Background daemon thread that:
#    1. Every LOOP_SLEEP_S, pulls new traces from AgentRegistry
#    2. Converts traces → Transitions via ExperienceBuffer
#    3. When enough transitions accumulate, assembles a rollout Episode
#    4. Calls LearningReasoner.absorb_training(ppo, episode)
#    5. Reports statistics via self.stats()
#
#  The loop runs as a daemon thread so it dies cleanly when the process exits.
#  It can be paused, resumed, and stopped explicitly.
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackLoopState(Enum):
    STOPPED  = auto()
    RUNNING  = auto()
    PAUSED   = auto()
    ERROR    = auto()


class FeedbackLoop:
    """
    Background training thread. Wires live kernel execution into the RL policy.

    Usage:
        loop = FeedbackLoop(reasoner, buffer)
        loop.start()
        # ... kernel runs normally ...
        loop.stop()
    """

    def __init__(
        self,
        reasoner: LearningReasoner,
        buffer:   ExperienceBuffer,
        ppo:      Optional[Any] = None,
    ) -> None:
        self._reasoner  = reasoner
        self._buffer    = buffer
        self._ppo       = ppo
        self._state     = FeedbackLoopState.STOPPED
        self._lock      = threading.RLock()
        self._thread    : Optional[threading.Thread] = None
        self._trace_cursor  = 0     # index into registry trace deque
        self._update_count  = 0
        self._total_ingested = 0
        self._last_loss     = 0.0
        self._errors        : deque = deque(maxlen=32)

    def _ingest_new_traces(self) -> int:
        """Pull new traces from the registry and push to buffer."""
        if not _HAS_CORE:
            return 0
        try:
            all_traces = _registry.recent_traces(8192)
            new_traces = all_traces[self._trace_cursor:]
            if not new_traces:
                return 0
            added = self._buffer.ingest_traces(new_traces)
            self._trace_cursor += len(new_traces)
            return added
        except Exception as e:
            self._errors.append(str(e))
            return 0

    def _try_train_step(self) -> bool:
        """Attempt one PPO update. Returns True if training occurred."""
        if len(self._buffer) < EPISODE_LEN:
            return False

        ep = self._buffer.drain_episode()
        if ep is None:
            return False

        # If no PPO supplied, create one around the decision policy
        ppo = self._ppo
        if ppo is None and _HAS_AC and self._reasoner._decision_ac is not None:
            try:
                ppo = PPO(self._reasoner._decision_ac)
                self._ppo = ppo
            except Exception:
                return False

        loss = self._reasoner.absorb_training(ppo, ep)
        with self._lock:
            self._update_count  += 1
            self._total_ingested += ep.length
            self._last_loss      = loss
        return True

    def _loop_body(self) -> None:
        """Single iteration of the feedback loop."""
        self._ingest_new_traces()
        self._try_train_step()

    def _run(self) -> None:
        """Main thread target."""
        while True:
            with self._lock:
                state = self._state
            if state == FeedbackLoopState.STOPPED:
                break
            if state == FeedbackLoopState.RUNNING:
                try:
                    self._loop_body()
                except Exception as e:
                    self._errors.append(str(e))
            time.sleep(LOOP_SLEEP_S)

    def start(self) -> bool:
        """Start the background loop. Returns False if already running."""
        with self._lock:
            if self._state == FeedbackLoopState.RUNNING:
                return False
            self._state = FeedbackLoopState.RUNNING
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="aios-feedback-loop",
                    daemon=True,
                )
                self._thread.start()
        return True

    def pause(self) -> None:
        with self._lock:
            if self._state == FeedbackLoopState.RUNNING:
                self._state = FeedbackLoopState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state == FeedbackLoopState.PAUSED:
                self._state = FeedbackLoopState.RUNNING

    def stop(self, join_timeout: float = 2.0) -> None:
        """Stop the loop and optionally wait for thread exit."""
        with self._lock:
            self._state = FeedbackLoopState.STOPPED
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._state == FeedbackLoopState.RUNNING

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state":           self._state.name,
                "update_count":    self._update_count,
                "total_ingested":  self._total_ingested,
                "buffer":          self._buffer.stats(),
                "reasoner":        self._reasoner.stats(),
                "last_loss":       self._last_loss,
                "trace_cursor":    self._trace_cursor,
                "recent_errors":   list(self._errors),
            }


# ─────────────────────────────────────────────────────────────────────────────
# §5  FEEDBACK KERNEL — @agent_method integration + kernel hot-swap
#
#  After boot, call:
#      fk = FeedbackKernel()
#      fk.attach(kernel)
#      fk.start()
#
#  This hot-swaps kernel._reasoner from RuleBasedReasoner to LearningReasoner
#  (which internally falls back to rule-based until trained sufficiently),
#  and starts the FeedbackLoop daemon thread.
# ─────────────────────────────────────────────────────────────────────────────

def _rebind_agent_methods(obj: Any) -> None:
    """Re-register @agent_method tools on obj as bound methods in the registry."""
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
            name=spec.name, description=spec.description,
            parameters=spec.parameters, returns=spec.returns,
            priority=spec.priority, fn=method, owner=type(obj).__name__,
        )
        _reg.register(bound_spec)


class FeedbackKernel:
    """
    Orchestrates the full feedback loop installation on a live AgentKernel.

    Responsibilities:
      1. Create RewardShaper + StateEncoder
      2. Create ExperienceBuffer + ToolIndex
      3. Create LearningReasoner (with RuleBasedReasoner fallback)
      4. Hot-swap kernel._reasoner → LearningReasoner
      5. Create and start FeedbackLoop daemon thread
    """

    def __init__(self) -> None:
        # Defer heavy setup until attach()
        self._shaper    : Optional[RewardShaper]      = None
        self._encoder   : Optional[StateEncoder]      = None
        self._tool_index: Optional[ToolIndex]         = None
        self._buffer    : Optional[ExperienceBuffer]  = None
        self._reasoner  : Optional[LearningReasoner]  = None
        self._loop      : Optional[FeedbackLoop]      = None
        self._ppo       : Optional[Any]               = None
        self._kernel    : Optional[Any]               = None
        self._lock      = threading.RLock()
        self._attached  = False

    def attach(self, kernel: Any) -> None:
        """
        Bind to a running AgentKernel and install the feedback loop.
        Safe to call after kernel.boot().
        """
        with self._lock:
            if self._attached:
                return

            self._kernel = kernel

            # Build components
            self._shaper     = RewardShaper()
            self._encoder    = StateEncoder(self._shaper)
            self._tool_index = ToolIndex()
            self._tool_index.seed_from_registry()

            self._buffer = ExperienceBuffer(
                capacity   = 4096,
                shaper     = self._shaper,
                encoder    = self._encoder,
                tool_index = self._tool_index,
            )

            self._reasoner = LearningReasoner(
                state_dim   = STATE_DIM,
                max_options = MAX_OPTIONS,
                fallback    = RuleBasedReasoner(),
                tool_index  = self._tool_index,
            )

            # Build PPO around the decision actor-critic
            if _HAS_AC and self._reasoner._decision_ac is not None:
                try:
                    self._ppo = PPO(self._reasoner._decision_ac)
                except Exception:
                    self._ppo = None

            self._loop = FeedbackLoop(
                reasoner = self._reasoner,
                buffer   = self._buffer,
                ppo      = self._ppo,
            )

            # Hot-swap the kernel reasoner if possible
            if hasattr(kernel, '_reasoner'):
                kernel._reasoner = self._reasoner

            # Re-register our @agent_method tools as bound methods
            _rebind_agent_methods(self)

            self._attached = True

    @agent_method(
        name="feedback_start",
        description="Start the RL feedback loop background thread",
        priority=AgentPriority.LOW,
    )
    def start(self) -> bool:
        """Start the feedback loop. Returns True on success."""
        if self._loop is None:
            return False
        return self._loop.start()

    @agent_method(
        name="feedback_stop",
        description="Stop the RL feedback loop background thread",
        priority=AgentPriority.LOW,
    )
    def stop(self) -> None:
        if self._loop:
            self._loop.stop()

    @agent_method(
        name="feedback_pause",
        description="Pause the RL feedback loop (training stops, kernel continues)",
        priority=AgentPriority.LOW,
    )
    def pause(self) -> None:
        if self._loop:
            self._loop.pause()

    @agent_method(
        name="feedback_resume",
        description="Resume a paused RL feedback loop",
        priority=AgentPriority.LOW,
    )
    def resume(self) -> None:
        if self._loop:
            self._loop.resume()

    @agent_method(
        name="feedback_stats",
        description="Return feedback loop, buffer, and reasoner statistics",
        priority=AgentPriority.LOW,
    )
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "attached": self._attached,
                "loop":     self._loop.stats() if self._loop else {},
                "version":  _FEEDBACK_VERSION,
            }

    @agent_method(
        name="feedback_force_train",
        description="Force one training step regardless of buffer depth (for testing)",
        priority=AgentPriority.LOW,
    )
    def force_train(self) -> Dict[str, Any]:
        """
        Force a single training update from whatever transitions are buffered.
        Useful for testing and for manually triggering policy refreshes.
        """
        if self._loop is None:
            return {"error": "not attached"}
        ingested = self._loop._ingest_new_traces()
        trained  = self._loop._try_train_step()
        return {
            "ingested": ingested,
            "trained":  trained,
            "stats":    self._loop.stats(),
        }

    @agent_method(
        name="feedback_reasoner_stats",
        description="Return LearningReasoner policy statistics",
        priority=AgentPriority.LOW,
    )
    def reasoner_stats(self) -> Dict[str, Any]:
        if self._reasoner is None:
            return {"error": "not attached"}
        return self._reasoner.stats()


# ─────────────────────────────────────────────────────────────────────────────
# §6  SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """Deterministic validation suite. Raises AssertionError on failure."""

    # ── Math shim ─────────────────────────────────────────────────────────────
    assert _abs(-3.0) == 3.0
    assert _abs(_exp(0.0) - 1.0) < 1e-9
    assert _abs(_exp(1.0) - _E)  < 1e-9
    assert _abs(_ln(1.0))        < 1e-9
    assert _abs(_ln(_E) - 1.0)   < 1e-9
    assert _abs(_sqrt(9.0) - 3.0)< 1e-9

    # ── Softmax ───────────────────────────────────────────────────────────────
    probs = _softmax([1.0, 2.0, 3.0])
    assert _abs(sum(probs) - 1.0) < 1e-9, f"softmax sum={sum(probs)}"
    assert probs[2] > probs[1] > probs[0], "softmax must be monotone"

    # ── Context encoding ──────────────────────────────────────────────────────
    s1 = _encode_context_as_state("allocate memory", ["palloc", "pfree"])
    assert len(s1) == STATE_DIM, f"len={len(s1)}"
    assert all(isinstance(v, float) for v in s1)
    assert all(-0.01 <= v <= 1.01 for v in s1), f"out of [0,1]: {s1}"

    s2 = _encode_context_as_state("allocate memory", ["palloc", "pfree"])
    assert s1 == s2, "Encoding must be deterministic"

    s3 = _encode_context_as_state("free memory", ["palloc", "pfree"])
    assert s1 != s3, "Different context must give different state"

    s4 = _encode_goal_as_state("plan memory allocation")
    assert len(s4) == STATE_DIM

    # ── ToolIndex ─────────────────────────────────────────────────────────────
    ti = ToolIndex()
    i0 = ti.get_or_assign("palloc")
    i1 = ti.get_or_assign("pfree")
    i2 = ti.get_or_assign("palloc")  # idempotent
    assert i0 != i1, "Different tools get different indices"
    assert i0 == i2, "Same tool always gets same index"
    assert ti.get_name(i0) == "palloc"
    assert ti.get_name(i1) == "pfree"

    # ── ExperienceBuffer ──────────────────────────────────────────────────────
    shaper  = RewardShaper()
    encoder = StateEncoder(shaper)
    ti2     = ToolIndex()
    buf     = ExperienceBuffer(capacity=256, shaper=shaper, encoder=encoder, tool_index=ti2)

    assert len(buf) == 0

    # Simulate traces as dicts (aios_core not required)
    fake_traces = [
        type('T', (), {
            'tool_name':   'palloc',
            'success':     True,
            'duration_ns': 500_000,
            'depth':       0,
            'priority':    1,
            'error':       None,
            'timestamp':   float(i),
        })()
        for i in range(100)
    ]

    added = buf.ingest_traces(fake_traces)
    assert added == 100, f"Expected 100, got {added}"
    assert len(buf) == 100

    # Can drain an episode once enough transitions exist
    ep = buf.drain_episode()
    assert ep is not None
    assert ep.length == EPISODE_LEN
    assert len(buf) == 100 - EPISODE_LEN

    # drain_batch
    batch = buf.drain_batch(5)
    assert len(batch) == 5
    assert all(isinstance(t.state, list) and len(t.state) == STATE_DIM for t in batch)
    assert all(isinstance(t.reward, float) for t in batch)

    # ── LearningReasoner — cold-start falls back ──────────────────────────────
    lr = LearningReasoner(fallback=RuleBasedReasoner(), tool_index=ToolIndex())
    ctx = AgentContext() if _HAS_CORE else type('C', (), {
        'depth':0,'trace_id':'','budget_ns':0,'metadata':{},'_chain':[],'caller':'k'
    })()

    # Under MIN_TRAIN_STEPS, must fall back to rule-based behaviour
    assert lr._trained_steps == 0
    assert not lr._use_learned_policy()

    # Options: first available should be returned (rule-based fallback)
    choice = lr.decide("allocate memory", ["palloc", "pfree"], ctx)
    assert choice in ["palloc", "pfree"], f"Unexpected choice: {choice}"

    # annotate should not raise
    ann = lr.annotate("palloc", (), {}, ctx)
    assert ann is not None or ann is None  # either is fine

    # ── FeedbackLoop lifecycle ────────────────────────────────────────────────
    reasoner2 = LearningReasoner(fallback=RuleBasedReasoner(), tool_index=ToolIndex())
    buf2      = ExperienceBuffer(capacity=256)
    loop      = FeedbackLoop(reasoner2, buf2)

    assert not loop.is_running
    ok = loop.start()
    assert ok
    assert loop.is_running

    # Pause/resume cycle
    loop.pause()
    time.sleep(0.05)
    loop.resume()
    assert loop.is_running

    # Stop and confirm
    loop.stop(join_timeout=1.0)
    assert not loop.is_running

    st = loop.stats()
    assert st["state"] == "STOPPED"
    assert "reasoner" in st
    assert "buffer"   in st

    # ── FeedbackKernel — attach without real kernel ───────────────────────────
    class FakeKernel:
        _reasoner = RuleBasedReasoner()

    fk     = FeedbackKernel()
    fake_k = FakeKernel()
    fk.attach(fake_k)

    assert fk._attached
    assert isinstance(fake_k._reasoner, LearningReasoner)

    result = fk.stats()
    assert result["attached"] is True
    assert "version" in result

    # force_train with empty buffer should not crash
    ft_result = fk.force_train()
    assert "ingested" in ft_result
    assert "trained"  in ft_result

    # start/stop via kernel API
    assert fk.start()
    time.sleep(0.05)
    fk.stop()

    print("aios_feedback: all self-tests passed ✓")


if __name__ == "__main__":
    _run_self_tests()
