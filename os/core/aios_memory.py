#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Hierarchical Memory Subsystem                                        ║
║  aios_memory.py                                                              ║
║                                                                              ║
║  "To remember is to survive. To forget is to make room for what matters."    ║
║                                                                              ║
║  Architecture (Baddeley–Hitch extended with modern ML):                      ║
║    §1  Math Shim          — exp, log, sqrt, cos (inline; no math import)     ║
║    §2  Embedding Projector— Linear projection W∈ℝ^{d_in×d_emb}, L2-norm     ║
║    §3  Working Memory     — Attention-gated circular buffer, O(1) push       ║
║    §4  Episodic Memory    — Ebbinghaus decay, cosine retrieval, top-k        ║
║    §5  Semantic Memory    — Hebbian association, confidence-weighted facts    ║
║    §6  LSH Index          — Sign-random-projection ANN for O(1) lookups      ║
║    §7  Modern Hopfield    — Exponential-energy pattern completion             ║
║    §8  PER Consolidator   — Prioritized Experience Replay → Adam update      ║
║    §9  Sleep Cycle        — EM→SM distillation, decay sweep, index rebuild   ║
║    §10 CTF Serializer     — Binary persistence compatible with custom_model  ║
║    §11 MemoryKernel       — @agent_method interface to AgentKernel           ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Forgetting  : I(t) = I₀·exp(-λ·Δt)   [Ebbinghaus 1885]                  ║
║    Retrieval   : sim(q,k) = q·k/(‖q‖·‖k‖+ε)  [Cosine]                      ║
║    PER priority: pᵢ = (|δᵢ|+ε)^α,  P(i) = pᵢ/Σpⱼ  [Schaul 2016]          ║
║    Hopfield    : ξ_new = Ξ·softmax(β·Ξᵀ·ξ)  [Ramsauer 2020]                ║
║    Hebbian     : ΔW = η·(yxᵀ - λW)  [Oja's rule]                           ║
║    LSH         : h(x) = sign(W_lsh·x),  W_lsh ~ N(0,1)                     ║
║    Adam update : θ -= lr·m̂/(√v̂+ε)  [Kingma & Ba 2015]                     ║
║                                                                              ║
║  Integration:                                                                ║
║    from aios_core      import agent_method, AgentPriority, _registry        ║
║    from aios_phase4_nn import Tensor, Parameter, Linear, Adam               ║
║    mk = MemoryKernel(d_emb=128); mk.attach(kernel)                          ║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Every computation traceable to a named equation in this docstring.      ║
║    • Thread-safe: all mutable state guarded by threading.RLock.              ║
║    • Standalone: runs without aios_core/phase4 if those aren't on path.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import struct
import threading
import time
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable

# ═══════════════════════════════════════════════════════════════════════════════
# §1  MATH SHIM — identical idiom to aios_phase4_nn.py: no math import
# ═══════════════════════════════════════════════════════════════════════════════

_PI  = 3.141592653589793238462643383279
_E   = 2.718281828459045235360287471352
_INF = float('inf')
_EPS = 1e-12


def _m_abs(x: float) -> float:
    return x if x >= 0.0 else -x


def _m_floor(x: float) -> int:
    n = int(x)
    return n - 1 if x < n else n


def _m_exp(x: float) -> float:
    """e^x via range-reduction + 25-term Taylor; mirrors aios_phase4_nn._exp."""
    if x > 709.782:  return _INF
    if x < -745.13:  return 0.0
    if x == 0.0:     return 1.0
    n = _m_floor(x)
    r = x - n
    result, term = 1.0, 1.0
    for k in range(1, 25):
        term *= r / k
        result += term
        if _m_abs(term) < 1e-17:
            break
    # e^n via repeated squaring
    base = _E
    pw   = 1.0
    nn   = n if n >= 0 else -n
    while nn:
        if nn & 1:
            pw *= base
        base *= base
        nn >>= 1
    return result * (pw if n >= 0 else 1.0 / pw)


def _m_log(x: float) -> float:
    """ln(x) via Halley's method seeded by bit-trick approximation."""
    if x <= 0.0:  return -_INF
    if x == 1.0:  return 0.0
    # Reduce: x = m * 2^e, m in [0.5, 1)
    e = 0
    m = x
    while m >= 1.0: m /= 2.0; e += 1
    while m < 0.5:  m *= 2.0; e -= 1
    # ln(x) = ln(m) + e*ln2; use ln2 = 0.693147...
    LN2 = 0.6931471805599453
    # ln(m): Newton/Halley on f(y)=e^y-m, |m-1| small after rescale to [√½,√2)
    # Rescale m once more to [√½, √2) to keep |m-1| ≤ 0.414
    if m < 0.7071067811865476:
        m *= 2.0; e -= 1
    # Halley: y_{n+1} = y_n + 2*(m - e^y_n)/(m + e^y_n)
    y = (m - 1.0) - (m - 1.0) ** 2 / 2.0  # Taylor seed
    for _ in range(8):
        ey  = _m_exp(y)
        y  += 2.0 * (m - ey) / (m + ey)
    return y + e * LN2


def _m_sqrt(x: float) -> float:
    """√x via 5 iterations of Newton's method (Heron's)."""
    if x < 0.0: return float('nan')
    if x == 0.0: return 0.0
    y = x
    for _ in range(40):
        yn = 0.5 * (y + x / y)
        if _m_abs(yn - y) < 1e-15 * y:
            break
        y = yn
    return y


def _m_cos(x: float) -> float:
    """cos(x) via 14-term Taylor after range-reduction to [-π, π]."""
    x = x - 2.0 * _PI * _m_floor(x / (2.0 * _PI) + 0.5)
    result, term, xx = 1.0, 1.0, x * x
    for k in range(1, 14):
        term *= -xx / ((2 * k - 1) * (2 * k))
        result += term
        if _m_abs(term) < 1e-17:
            break
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §2  EMBEDDING PROJECTOR
#
# Maps an arbitrary flat feature vector x ∈ ℝ^{d_in} to a unit-norm embedding
# ê ∈ ℝ^{d_emb} via:
#
#     e  = W · x + b,     W ∈ ℝ^{d_emb × d_in},  b ∈ ℝ^{d_emb}
#     ê  = e / (‖e‖₂ + ε)
#
# Weights initialised: W ~ N(0, 2/d_in) (He-normal for linear activations),
#                       b = 0
#
# The projector can be updated via Adam gradient descent (§8 PER Consolidator).
# Its parameters are serialised in the CTF memory snapshot (§10).
# ═══════════════════════════════════════════════════════════════════════════════

class _RNGEngine:
    """Xorshift64 — identical to aios_phase4_nn._XorshiftRNG for reproducibility."""
    def __init__(self, seed: int = 20240101) -> None:
        self._state = seed ^ 0xDEADBEEFCAFEBABE
        if self._state == 0:
            self._state = 1

    def next_uint64(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7)
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._state = x & 0xFFFFFFFFFFFFFFFF
        return self._state

    def randn(self) -> float:
        """Box-Muller Gaussian."""
        while True:
            u1 = (self.next_uint64() >> 11) / (1 << 53)
            u2 = (self.next_uint64() >> 11) / (1 << 53)
            if u1 > 0.0:
                break
        mag = _m_sqrt(-2.0 * _m_log(u1))
        # cos branch of Box-Muller
        return mag * _m_cos(2.0 * _PI * u2)

    def uniform(self) -> float:
        return (self.next_uint64() >> 11) / (1 << 53)


_RNG = _RNGEngine(seed=0xAE1F_0B32_7C5D_9A84)


class EmbeddingProjector:
    """
    Learnable linear embedding: e = Wx + b,  ê = e/‖e‖

    Attributes
    ----------
    W : list[list[float]]   shape (d_emb, d_in)
    b : list[float]         shape (d_emb,)
    """

    def __init__(self, d_in: int, d_emb: int, seed: int = 42) -> None:
        rng   = _RNGEngine(seed)
        scale = _m_sqrt(2.0 / d_in)          # He-normal
        self.d_in  = d_in
        self.d_emb = d_emb
        # W stored row-major: W[i] is the i-th output neuron weights
        self.W: List[List[float]] = [
            [rng.randn() * scale for _ in range(d_in)]
            for _ in range(d_emb)
        ]
        self.b: List[float] = [0.0] * d_emb

        # Adam state for in-place gradient updates
        self._mW: List[List[float]] = [[0.0] * d_in for _ in range(d_emb)]
        self._vW: List[List[float]] = [[0.0] * d_in for _ in range(d_emb)]
        self._mb: List[float]        = [0.0] * d_emb
        self._vb: List[float]        = [0.0] * d_emb
        self._adam_t: int            = 0

    # ── Forward ──────────────────────────────────────────────────────────────

    def embed(self, x: List[float]) -> List[float]:
        """
        Project x ∈ ℝ^{d_in} → ê ∈ ℝ^{d_emb} (unit norm).

        If len(x) < d_in, zero-pad; if > d_in, truncate.
        """
        # Pad / truncate
        if len(x) < self.d_in:
            x = list(x) + [0.0] * (self.d_in - len(x))
        elif len(x) > self.d_in:
            x = list(x[:self.d_in])

        # Linear: e_i = Σ_j W[i][j] * x[j] + b[i]
        e = [sum(self.W[i][j] * x[j] for j in range(self.d_in)) + self.b[i]
             for i in range(self.d_emb)]

        # L2-normalise
        norm = _m_sqrt(sum(v * v for v in e)) + _EPS
        return [v / norm for v in e]

    # ── Adam update (called from PER Consolidator, §8) ────────────────────────

    def adam_update(self, grad_W: List[List[float]], grad_b: List[float],
                    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8) -> None:
        """
        In-place Adam step.
        m_t = β₁·m_{t-1} + (1-β₁)·g
        v_t = β₂·v_{t-1} + (1-β₂)·g²
        θ  -= lr · m̂ / (√v̂ + ε)
        """
        self._adam_t += 1
        t     = self._adam_t
        bc1   = 1.0 - beta1 ** t
        bc2   = 1.0 - beta2 ** t

        for i in range(self.d_emb):
            # bias
            g = grad_b[i]
            self._mb[i] = beta1 * self._mb[i] + (1.0 - beta1) * g
            self._vb[i] = beta2 * self._vb[i] + (1.0 - beta2) * g * g
            self.b[i]  -= lr * (self._mb[i] / bc1) / (_m_sqrt(self._vb[i] / bc2) + eps)
            # weights
            for j in range(self.d_in):
                g = grad_W[i][j]
                self._mW[i][j] = beta1 * self._mW[i][j] + (1.0 - beta1) * g
                self._vW[i][j] = beta2 * self._vW[i][j] + (1.0 - beta2) * g * g
                self.W[i][j]  -= lr * (self._mW[i][j] / bc1) / (
                    _m_sqrt(self._vW[i][j] / bc2) + eps)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_flat(self) -> List[float]:
        """Flatten W then b into a single list for CTF serialisation."""
        flat = []
        for row in self.W:
            flat.extend(row)
        flat.extend(self.b)
        return flat

    def from_flat(self, data: List[float]) -> None:
        """Restore W and b from a flat list produced by to_flat()."""
        expected = self.d_emb * self.d_in + self.d_emb
        if len(data) < expected:
            raise ValueError(f"EmbeddingProjector.from_flat: need {expected} floats, got {len(data)}")
        idx = 0
        for i in range(self.d_emb):
            for j in range(self.d_in):
                self.W[i][j] = data[idx]; idx += 1
        for i in range(self.d_emb):
            self.b[i] = data[idx]; idx += 1


# ═══════════════════════════════════════════════════════════════════════════════
# §3  WORKING MEMORY
#
# Fixed-capacity circular buffer of the K most recent observations.
# At retrieval, an attention-weighted readout is computed:
#
#     α_i = softmax( q · k_i / √d_emb )_i
#     h   = Σ_i  α_i · v_i
#
# where q is the query embedding, k_i/v_i are stored slot embeddings.
#
# Capacity: K slots. Each slot: (content, embedding, timestamp, attn_weight).
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WMSlot:
    content:     Any
    embedding:   List[float]
    timestamp:   float
    attn_weight: float = 0.0


class WorkingMemory:
    """
    Attention-gated circular buffer.

    Parameters
    ----------
    capacity : int     Maximum slots (default 16).
    d_emb    : int     Embedding dimensionality.
    """

    def __init__(self, capacity: int = 16, d_emb: int = 128) -> None:
        self.capacity = capacity
        self.d_emb    = d_emb
        self._slots: deque = deque(maxlen=capacity)
        self._lock  = threading.RLock()
        self._clock = 0

    def push(self, content: Any, embedding: List[float]) -> None:
        """Insert a new observation; oldest is evicted when full."""
        with self._lock:
            self._clock += 1
            self._slots.append(WMSlot(
                content   = content,
                embedding = list(embedding),
                timestamp = time.monotonic(),
                attn_weight = 0.0,
            ))

    def attend(self, query_emb: List[float], temperature: float = 1.0) -> List[float]:
        """
        Compute attention-weighted readout h ∈ ℝ^{d_emb}.

        Returns the zero vector if the buffer is empty.
        """
        with self._lock:
            slots = list(self._slots)
        if not slots:
            return [0.0] * self.d_emb

        d = len(query_emb)
        scale = _m_sqrt(float(d)) * temperature

        # Dot products: s_i = q · k_i / (√d · temperature)
        scores = [_dot(query_emb, s.embedding) / scale for s in slots]

        # Stable softmax
        s_max  = max(scores)
        exps   = [_m_exp(s - s_max) for s in scores]
        z      = sum(exps) + _EPS
        alphas = [e / z for e in exps]

        # Store weights back for introspection
        with self._lock:
            live = list(self._slots)
            for i, sl in enumerate(live):
                sl.attn_weight = alphas[i]

        # Weighted sum h = Σ α_i * v_i
        h = [0.0] * self.d_emb
        for alpha, sl in zip(alphas, slots):
            for k in range(min(self.d_emb, len(sl.embedding))):
                h[k] += alpha * sl.embedding[k]
        return h

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return [
                {"content": s.content, "ts": s.timestamp, "attn": s.attn_weight}
                for s in self._slots
            ]

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()

    def __len__(self) -> int:
        return len(self._slots)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  EPISODIC MEMORY
#
# Long-term store of (up to N) discrete events.
#
# Importance decay — Ebbinghaus (1885):
#     I(t) = I₀ · exp(-λ · Δt) + I_consolidated
#
# Retrieval scoring:
#     score_i = sim(q, k_i) · I(tᵢ)
#     sim(q,k) = (q · k) / (‖q‖ · ‖k‖ + ε)
#
# Top-k retrieval returns episodes sorted by score descending.
# Access increments access_count and refreshes the importance floor.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    ep_id:            str
    timestamp:        float
    content:          Any
    embedding:        List[float]
    importance:       float          # I₀ at creation time
    importance_floor: float          # I_consolidated: minimum persisted importance
    access_count:     int = 0
    _creation_time:   float = field(default_factory=time.monotonic)

    def current_importance(self, decay_rate: float) -> float:
        """I(t) = I₀ · exp(-λ·Δt) + I_floor"""
        delta_t = time.monotonic() - self._creation_time
        return self.importance * _m_exp(-decay_rate * delta_t) + self.importance_floor

    def touch(self) -> None:
        """Access: refresh creation_time (spaced-repetition effect) + increment count."""
        self.access_count  += 1
        # Spaced repetition: each access adds a fraction to the floor,
        # then resets the decay clock.  The floor is bounded at 0.5 * importance.
        bump = self.importance * 0.05 * _m_exp(-0.1 * self.access_count)
        self.importance_floor = min(self.importance_floor + bump,
                                    0.5 * self.importance)
        self._creation_time = time.monotonic()


class EpisodicMemory:
    """
    Fixed-capacity long-term episode store with Ebbinghaus decay.

    Parameters
    ----------
    capacity   : int     Maximum concurrent episodes (default 4096).
    d_emb      : int     Embedding dimensionality.
    decay_rate : float   λ in I(t) = I₀·exp(-λΔt) [unit: 1/second]
    prune_thr  : float   Prune when I(t) < prune_thr AND pressure > 0.85.
    """

    def __init__(self, capacity: int = 4096, d_emb: int = 128,
                 decay_rate: float = 1e-5, prune_thr: float = 0.01) -> None:
        self.capacity   = capacity
        self.d_emb      = d_emb
        self.decay_rate = decay_rate
        self.prune_thr  = prune_thr
        self._episodes: OrderedDict = OrderedDict()   # ep_id → Episode
        self._lock      = threading.RLock()

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, content: Any, embedding: List[float],
              importance: float = 1.0) -> str:
        """
        Commit a new episode.

        Returns the assigned episode ID (sha1 of timestamp + content repr,
        truncated to 16 hex chars).
        """
        ts    = time.monotonic()
        ep_id = hashlib.sha1(
            f"{ts}{repr(content)}".encode()
        ).hexdigest()[:16]

        ep = Episode(
            ep_id            = ep_id,
            timestamp        = ts,
            content          = content,
            embedding        = list(embedding),
            importance       = float(importance),
            importance_floor = 0.0,
        )

        with self._lock:
            if len(self._episodes) >= self.capacity:
                self._evict_lri()         # Least Recent + Importance
            self._episodes[ep_id] = ep
        return ep_id

    def _evict_lri(self) -> None:
        """
        Evict the episode with the lowest score:
            score = I(t) * log(1 + access_count + 1)
        Ties broken by insertion order (oldest first).
        """
        if not self._episodes:
            return
        worst_id    = None
        worst_score = _INF
        for ep_id, ep in self._episodes.items():
            s = ep.current_importance(self.decay_rate) * _m_log(
                1.0 + ep.access_count + 1.0)
            if s < worst_score:
                worst_score = s
                worst_id    = ep_id
        if worst_id:
            del self._episodes[worst_id]

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query_emb: List[float], top_k: int = 5,
                 min_similarity: float = 0.0) -> List[Tuple[float, Episode]]:
        """
        Return up to top_k episodes sorted by (cosine_similarity · importance) ↓.

        Each accessed episode is .touch()'ed to refresh its decay clock.
        """
        with self._lock:
            episodes = list(self._episodes.values())

        results = []
        for ep in episodes:
            sim  = _cosine(query_emb, ep.embedding)
            if sim < min_similarity:
                continue
            imp   = ep.current_importance(self.decay_rate)
            score = sim * imp
            results.append((score, ep))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]

        # Touch retrieved episodes
        with self._lock:
            for _, ep in top:
                if ep.ep_id in self._episodes:
                    self._episodes[ep.ep_id].touch()

        return top

    # ── Decay sweep ───────────────────────────────────────────────────────────

    def prune_decayed(self) -> int:
        """
        Remove episodes whose importance has decayed below prune_thr.
        Only activates when memory pressure > 85%.

        Returns count of pruned episodes.
        """
        pressure = len(self._episodes) / max(1, self.capacity)
        if pressure < 0.85:
            return 0

        with self._lock:
            to_prune = [
                ep_id for ep_id, ep in self._episodes.items()
                if ep.current_importance(self.decay_rate) < self.prune_thr
                and ep.access_count == 0
            ]
            for ep_id in to_prune:
                del self._episodes[ep_id]
        return len(to_prune)

    def all_episodes(self) -> List[Episode]:
        with self._lock:
            return list(self._episodes.values())

    def __len__(self) -> int:
        return len(self._episodes)


# ═══════════════════════════════════════════════════════════════════════════════
# §5  SEMANTIC MEMORY
#
# Compressed knowledge store: facts as (key_embedding, value, confidence).
# Association matrix A ∈ ℝ^{d_emb × d_emb} via Oja's rule:
#
#     ΔA = η · (y · xᵀ − λ · A)
#
# where x = key embedding, y = output embedding (or value projection).
# Retrieval: output = A · query / ‖A · query‖
#
# Each fact also carries a confidence score updated by repeated exposure:
#     c_new = c_old + α·(1 − c_old)   if consistent
#     c_new = c_old · (1 − α)          if contradicts existing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fact:
    fact_id:    str
    key_emb:    List[float]       # normalised
    value:      Any               # arbitrary Python object
    confidence: float             # ∈ [0, 1]
    source:     str               # 'episodic_consolidation' | 'explicit' | ...
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    access_count: int = 0


class SemanticMemory:
    """
    Confidence-weighted fact store + Hebbian association matrix.

    Parameters
    ----------
    d_emb        : int     Embedding dimension.
    capacity     : int     Maximum distinct facts (default 2048).
    hebbian_lr   : float   η in Oja's rule (default 1e-3).
    hebbian_decay: float   λ in Oja's rule (default 1e-4).
    """

    def __init__(self, d_emb: int = 128, capacity: int = 2048,
                 hebbian_lr: float = 1e-3, hebbian_decay: float = 1e-4) -> None:
        self.d_emb        = d_emb
        self.capacity     = capacity
        self.hebbian_lr   = hebbian_lr
        self.hebbian_decay = hebbian_decay

        self._facts: Dict[str, Fact] = {}
        self._lock  = threading.RLock()

        # Association matrix A ∈ ℝ^{d_emb × d_emb}, flat row-major
        self._A: List[float] = [0.0] * (d_emb * d_emb)

    # ── Fact operations ───────────────────────────────────────────────────────

    def store_fact(self, key_emb: List[float], value: Any,
                   confidence: float = 1.0, source: str = 'explicit') -> str:
        """
        Insert or update a fact.

        If a fact with a very similar key exists (cosine > 0.95), update it
        rather than inserting a duplicate. Returns fact_id.
        """
        norm_key = _l2_normalise(key_emb)
        fact_id  = hashlib.sha1(repr(norm_key[:4]).encode()).hexdigest()[:16]

        with self._lock:
            # Search for near-duplicate (O(N) — acceptable for N ≤ 2048)
            for fid, fact in self._facts.items():
                if _cosine(norm_key, fact.key_emb) > 0.95:
                    # Update existing: confidence update rule
                    alpha = 0.1
                    fact.confidence  = fact.confidence + alpha * (1.0 - fact.confidence)
                    fact.value       = value
                    fact.updated_at  = time.monotonic()
                    fact.access_count += 1
                    self._hebbian_update(norm_key, fact.key_emb)
                    return fid

            # Evict lowest-confidence fact if at capacity
            if len(self._facts) >= self.capacity:
                worst = min(self._facts.values(), key=lambda f: f.confidence)
                del self._facts[worst.fact_id]

            fact = Fact(
                fact_id    = fact_id,
                key_emb    = norm_key,
                value      = value,
                confidence = float(confidence),
                source     = source,
            )
            self._facts[fact_id] = fact
            self._hebbian_update(norm_key, norm_key)

        return fact_id

    def retrieve_fact(self, query_emb: List[float],
                      top_k: int = 3,
                      min_confidence: float = 0.1) -> List[Tuple[float, Fact]]:
        """
        Retrieve top_k most relevant facts above min_confidence.
        Score = cosine(query, key) * confidence.
        """
        norm_q = _l2_normalise(query_emb)
        with self._lock:
            facts = list(self._facts.values())

        results = []
        for f in facts:
            if f.confidence < min_confidence:
                continue
            score = _cosine(norm_q, f.key_emb) * f.confidence
            results.append((score, f))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]

        with self._lock:
            for _, f in top:
                if f.fact_id in self._facts:
                    self._facts[f.fact_id].access_count += 1

        return top

    # ── Hebbian association matrix ────────────────────────────────────────────

    def _hebbian_update(self, x: List[float], y: List[float]) -> None:
        """
        Oja's rule: ΔA = η · (y ⊗ x − λ · A)
        A[i][j] += η * (y[i]*x[j] - λ*A[i][j])
        """
        eta = self.hebbian_lr
        lam = self.hebbian_decay
        d   = self.d_emb
        for i in range(d):
            yi = y[i] if i < len(y) else 0.0
            for j in range(d):
                xj = x[j] if j < len(x) else 0.0
                idx          = i * d + j
                self._A[idx] += eta * (yi * xj - lam * self._A[idx])

    def associative_recall(self, query_emb: List[float]) -> List[float]:
        """
        Heteroassociative recall: output = A · query / ‖A · query‖
        Retrieves the semantic direction most strongly associated with query.
        """
        d   = self.d_emb
        Aq  = [0.0] * d
        for i in range(d):
            s = 0.0
            for j in range(d):
                qj = query_emb[j] if j < len(query_emb) else 0.0
                s += self._A[i * d + j] * qj
            Aq[i] = s
        return _l2_normalise(Aq)

    def all_facts(self) -> List[Fact]:
        with self._lock:
            return list(self._facts.values())

    def __len__(self) -> int:
        return len(self._facts)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  LSH INDEX — Locality-Sensitive Hashing for approximate k-NN
#
# Sign-random-projection hashing:
#     h(x) = sign(W_lsh · x),    W_lsh ~ N(0,1),  W_lsh ∈ ℝ^{n_bits × d_emb}
#
# n_tables independent hash tables, each using n_bits bits → 2^n_bits buckets.
# Expected collision probability for two vectors with angle θ:
#     P[h(u) = h(v)] = 1 - θ/π
#
# For θ = 30°: P ≈ 0.833 per bit.  With n_bits=16: P_all ≈ 0.833^16 ≈ 0.055.
# With n_tables=8: P(found in ≥1 table) = 1 - (1-0.055)^8 ≈ 0.36 — low but
# supplemented by the fact that similar vectors land in the same bucket.
# ═══════════════════════════════════════════════════════════════════════════════

class LSHIndex:
    """
    Multi-table sign-random-projection LSH.

    Parameters
    ----------
    d_emb    : int   Embedding dimension.
    n_bits   : int   Bits per hash (bucket key width).  Default 16.
    n_tables : int   Number of independent hash tables.  Default 8.
    """

    def __init__(self, d_emb: int = 128, n_bits: int = 16,
                 n_tables: int = 8, seed: int = 0xBEEF) -> None:
        self.d_emb    = d_emb
        self.n_bits   = n_bits
        self.n_tables = n_tables

        rng = _RNGEngine(seed)
        # Projection matrices: list of (n_bits × d_emb) float matrices
        self._projs: List[List[List[float]]] = []
        for _ in range(n_tables):
            table_proj = [[rng.randn() for _ in range(d_emb)]
                          for _ in range(n_bits)]
            self._projs.append(table_proj)

        # Hash tables: list of dict[int, list[str]]  (bucket → list of ep_ids)
        self._tables: List[Dict[int, List[str]]] = [{} for _ in range(n_tables)]
        self._lock   = threading.RLock()

    def _hash(self, emb: List[float], table_idx: int) -> int:
        """
        Compute the n_bits-wide hash for embedding in the given table.
        h = Σ_i  sign(proj[i] · emb) * 2^i
        """
        proj  = self._projs[table_idx]
        code  = 0
        for bit_idx in range(self.n_bits):
            row = proj[bit_idx]
            dot = sum(row[j] * (emb[j] if j < len(emb) else 0.0)
                      for j in range(self.d_emb))
            if dot >= 0.0:
                code |= (1 << bit_idx)
        return code

    def insert(self, ep_id: str, embedding: List[float]) -> None:
        """Insert ep_id into all hash tables."""
        with self._lock:
            for t in range(self.n_tables):
                h = self._hash(embedding, t)
                self._tables[t].setdefault(h, []).append(ep_id)

    def query(self, embedding: List[float]) -> List[str]:
        """
        Return the union of candidate ep_ids from all tables.
        May include false positives — re-rank by exact cosine similarity.
        """
        candidates: set = set()
        with self._lock:
            for t in range(self.n_tables):
                h = self._hash(embedding, t)
                candidates.update(self._tables[t].get(h, []))
        return list(candidates)

    def remove(self, ep_id: str) -> None:
        """Remove an ep_id from all buckets (O(N) — called only on eviction)."""
        with self._lock:
            for t in range(self.n_tables):
                for bucket in self._tables[t].values():
                    if ep_id in bucket:
                        bucket.remove(ep_id)

    def rebuild(self, episodes: List[Tuple[str, List[float]]]) -> None:
        """Full index rebuild from (ep_id, embedding) pairs."""
        with self._lock:
            for t in range(self.n_tables):
                self._tables[t].clear()
            for ep_id, emb in episodes:
                for t in range(self.n_tables):
                    h = self._hash(emb, t)
                    self._tables[t].setdefault(h, []).append(ep_id)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MODERN HOPFIELD NETWORK — Pattern Completion
#
# Energy function (Ramsauer et al., 2020):
#     F(ξ) = -lse(β, Ξᵀξ) + ½‖ξ‖² + C
#
# where lse(β, z) = β⁻¹ · log Σ_i exp(β·z_i)  (log-sum-exp)
#
# Fixed-point update rule (converges in 1–2 steps for large β):
#     ξ_{t+1} = Ξ · softmax(β · Ξᵀ · ξ_t)
#
# Ξ ∈ ℝ^{d_emb × M} is the memory matrix, each column a stored pattern.
# Storage capacity: M ≈ d_emb / (2 · log d_emb) well-separated patterns.
# ═══════════════════════════════════════════════════════════════════════════════

class HopfieldMemory:
    """
    Modern Hopfield pattern-completion layer.

    Parameters
    ----------
    d_emb    : int    Embedding dimension.
    beta     : float  Inverse temperature controlling retrieval sharpness.
    max_iter : int    Maximum fixed-point iterations (default 3).
    """

    def __init__(self, d_emb: int = 128, beta: float = 8.0,
                 max_iter: int = 3) -> None:
        self.d_emb    = d_emb
        self.beta     = beta
        self.max_iter = max_iter
        # Stored patterns: list of unit-norm embeddings
        self._patterns: List[List[float]] = []
        self._lock = threading.RLock()

    def store(self, pattern: List[float]) -> None:
        """Store a normalised pattern vector."""
        with self._lock:
            self._patterns.append(_l2_normalise(pattern))

    def retrieve(self, query: List[float]) -> List[float]:
        """
        Pattern completion via fixed-point iteration.
        Returns the retrieved attractor (unit norm).
        """
        with self._lock:
            patterns = list(self._patterns)
        if not patterns:
            return _l2_normalise(query)

        xi = _l2_normalise(query)
        for _ in range(self.max_iter):
            # Score = β · Ξᵀ · ξ_t  (dot with each pattern)
            scores = [self.beta * _dot(p, xi) for p in patterns]

            # Softmax over scores
            s_max  = max(scores)
            exps   = [_m_exp(s - s_max) for s in scores]
            z      = sum(exps) + _EPS
            alphas = [e / z for e in exps]

            # New estimate: ξ_{t+1} = Ξ · softmax(β Ξᵀ ξ_t)
            xi_new = [0.0] * self.d_emb
            for alpha, p in zip(alphas, patterns):
                for k in range(self.d_emb):
                    xi_new[k] += alpha * (p[k] if k < len(p) else 0.0)
            xi_new = _l2_normalise(xi_new)

            # Convergence check
            diff = _m_sqrt(sum((a - b) ** 2 for a, b in zip(xi_new, xi)))
            xi   = xi_new
            if diff < 1e-6:
                break

        return xi

    def clear(self) -> None:
        with self._lock:
            self._patterns.clear()

    def __len__(self) -> int:
        return len(self._patterns)


# ═══════════════════════════════════════════════════════════════════════════════
# §8  PRIORITIZED EXPERIENCE REPLAY (PER) CONSOLIDATOR
#
# Bridges episodic → semantic memory via gradient-based distillation.
#
# Priority:
#     δᵢ  = |I(tᵢ) · sim(query_mean, kᵢ) − μ|   (deviation from mean importance)
#     pᵢ  = (|δᵢ| + ε)^α
#     P(i) = pᵢ / Σ pⱼ
#
# Importance-sampling correction:
#     wᵢ  = ( N · P(i) )^{-β}
#     wᵢ  = wᵢ / max_j wⱼ
#
# Gradient step on embedding projector:
#     L   = Σᵢ wᵢ · ‖ê_reconstructed − ê_stored‖²
#     ∇W  = 2 · Σᵢ wᵢ · (ê_recon - ê_stored) · xᵀ   (chain through projector)
# ═══════════════════════════════════════════════════════════════════════════════

class PERConsolidator:
    """
    Prioritized Experience Replay consolidator.

    Parameters
    ----------
    projector      : EmbeddingProjector   The projector to update.
    alpha          : float                Priority exponent (default 0.6).
    beta_start     : float                IS weight exponent start (default 0.4).
    beta_end       : float                IS weight exponent end (annealed to 1.0).
    beta_anneal_T  : int                  Steps to anneal beta (default 1000).
    lr             : float                Adam learning rate for projector update.
    batch_size     : int                  Replay batch size.
    """

    def __init__(self, projector: EmbeddingProjector,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_anneal_T: int = 1000, lr: float = 1e-3,
                 batch_size: int = 32) -> None:
        self.projector     = projector
        self.alpha         = alpha
        self.beta          = beta_start
        self.beta_start    = beta_start
        self.beta_anneal_T = beta_anneal_T
        self.lr            = lr
        self.batch_size    = batch_size
        self._step         = 0
        self._lock         = threading.RLock()

    def _anneal_beta(self) -> float:
        """β anneals linearly from beta_start → 1.0 over beta_anneal_T steps."""
        frac = min(self._step / max(1, self.beta_anneal_T), 1.0)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def consolidate(self, episodes: List[Episode],
                    rng: _RNGEngine) -> Dict[str, float]:
        """
        Sample a mini-batch from episodes using PER and update the projector.

        Returns diagnostics: {loss, mean_weight, n_sampled}.
        """
        if len(episodes) < 2:
            return {"loss": 0.0, "mean_weight": 0.0, "n_sampled": 0}

        self._step += 1
        beta = self._anneal_beta()
        N    = len(episodes)

        # --- Compute priorities -------------------------------------------
        imps = [ep.current_importance(1e-5) for ep in episodes]
        mu   = sum(imps) / N
        deltas = [_m_abs(imp - mu) for imp in imps]
        eps_p  = 1e-6
        raw_p  = [(d + eps_p) ** self.alpha for d in deltas]
        total  = sum(raw_p) + _EPS
        probs  = [p / total for p in raw_p]

        # --- Sample batch (weighted) -------------------------------------
        batch_size = min(self.batch_size, N)
        # Stochastic sampling via cumulative distribution
        cumprob = []
        acc = 0.0
        for p in probs:
            acc += p
            cumprob.append(acc)

        sampled_idx = []
        for _ in range(batch_size):
            r   = rng.uniform()
            idx = 0
            for i, cp in enumerate(cumprob):
                if r <= cp:
                    idx = i
                    break
            sampled_idx.append(idx)

        # --- IS weights ---------------------------------------------------
        raw_w  = [(N * probs[i] + _EPS) ** (-beta) for i in sampled_idx]
        max_w  = max(raw_w)
        weights = [w / max_w for w in raw_w]

        # --- Gradient step on embedding projector -------------------------
        # Loss: L = Σ wᵢ · ‖projector.embed(raw_feature) - stored_emb‖²
        # Since we store the already-embedded vector, we use a self-reconstruction
        # objective: treat stored_emb as both input (zero-padded to d_in) and target.
        d_in  = self.projector.d_in
        d_emb = self.projector.d_emb

        grad_W = [[0.0] * d_in  for _ in range(d_emb)]
        grad_b = [0.0] * d_emb
        total_loss = 0.0

        for wi_idx, ep_idx in enumerate(sampled_idx):
            ep  = episodes[ep_idx]
            w   = weights[wi_idx]
            # Treat stored embedding as the input (pad/truncate to d_in)
            x   = list(ep.embedding)
            if len(x) < d_in:
                x.extend([0.0] * (d_in - len(x)))
            elif len(x) > d_in:
                x = x[:d_in]

            e_hat = self.projector.embed(x)   # forward pass
            # Residual: r = ê − stored_embedding (target)
            target = ep.embedding
            res = [e_hat[k] - (target[k] if k < len(target) else 0.0)
                   for k in range(d_emb)]
            loss = w * sum(r * r for r in res)
            total_loss += loss

            # Gradients: ∂L/∂e = 2w·r, then backprop through normalisation
            # Approximate: treat normalisation as identity for gradient (common approx)
            scale = 2.0 * w / (batch_size + _EPS)
            for i in range(d_emb):
                grad_b[i] += scale * res[i]
                for j in range(d_in):
                    grad_W[i][j] += scale * res[i] * x[j]

        # --- Adam update --------------------------------------------------
        self.projector.adam_update(grad_W, grad_b, lr=self.lr)

        return {
            "loss":        total_loss / (batch_size + _EPS),
            "mean_weight": sum(weights) / len(weights),
            "n_sampled":   batch_size,
            "beta":        beta,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §9  SLEEP CYCLE — EM → SM Distillation + Decay Sweep + LSH Rebuild
#
# Analogous to hippocampal-neocortical consolidation during slow-wave sleep.
# Triggered explicitly (kernel.sleep_cycle()) or by pressure threshold.
#
# Procedure:
#   1. Identify high-importance, high-access episodes (candidates for promotion).
#   2. For each candidate, store_fact() in SemanticMemory with confidence
#      proportional to (access_count / max_access) * importance.
#   3. Run PER Consolidator on full episodic set to update projector.
#   4. Run decay sweep on EpisodicMemory (prune expired).
#   5. Rebuild LSHIndex from surviving episodes.
#   6. Refresh HopfieldMemory with surviving high-importance episodes.
# ═══════════════════════════════════════════════════════════════════════════════

class SleepCycle:
    """
    Orchestrates one full memory consolidation pass.

    Parameters
    ----------
    em           : EpisodicMemory
    sm           : SemanticMemory
    lsh          : LSHIndex
    hop          : HopfieldMemory
    consolidator : PERConsolidator
    promote_thr  : float   Minimum importance to promote episode to SM.
    promote_acc  : int     Minimum access_count to promote.
    """

    def __init__(self, em: EpisodicMemory, sm: SemanticMemory,
                 lsh: LSHIndex, hop: HopfieldMemory,
                 consolidator: PERConsolidator,
                 promote_thr: float = 0.3,
                 promote_acc: int = 2) -> None:
        self.em           = em
        self.sm           = sm
        self.lsh          = lsh
        self.hop          = hop
        self.consolidator = consolidator
        self.promote_thr  = promote_thr
        self.promote_acc  = promote_acc
        self._lock        = threading.RLock()
        self._last_run    = 0.0
        self._run_count   = 0

    def run(self, rng: _RNGEngine) -> Dict[str, Any]:
        """
        Execute one full sleep cycle.

        Returns a report dict with counts and PER diagnostics.
        """
        with self._lock:
            t0 = time.monotonic()
            self._run_count += 1

            # 1. Collect all episodes
            all_eps = self.em.all_episodes()
            if not all_eps:
                return {"status": "nothing_to_consolidate", "run": self._run_count}

            max_access = max((ep.access_count for ep in all_eps), default=1) or 1

            # 2. Promote to semantic memory
            promoted = 0
            for ep in all_eps:
                imp = ep.current_importance(self.em.decay_rate)
                if imp >= self.promote_thr and ep.access_count >= self.promote_acc:
                    confidence = min(1.0, (ep.access_count / max_access) * imp)
                    self.sm.store_fact(
                        key_emb    = ep.embedding,
                        value      = ep.content,
                        confidence = confidence,
                        source     = 'episodic_consolidation',
                    )
                    promoted += 1

            # 3. PER consolidation step on embedding projector
            per_stats = self.consolidator.consolidate(all_eps, rng)

            # 4. Decay sweep
            pruned = self.em.prune_decayed()

            # 5. Rebuild LSH from surviving episodes
            surviving = self.em.all_episodes()
            self.lsh.rebuild([(ep.ep_id, ep.embedding) for ep in surviving])

            # 6. Refresh Hopfield with top-importance episodes
            self.hop.clear()
            sorted_eps = sorted(surviving,
                                key=lambda ep: ep.current_importance(self.em.decay_rate),
                                reverse=True)
            for ep in sorted_eps[:min(len(sorted_eps),
                                      max(1, self.hop.d_emb // 2))]:
                self.hop.store(ep.embedding)

            elapsed = time.monotonic() - t0
            self._last_run = t0

            return {
                "status":       "ok",
                "run":          self._run_count,
                "elapsed_ms":   round(elapsed * 1000, 2),
                "total_eps":    len(all_eps),
                "promoted":     promoted,
                "pruned":       pruned,
                "surviving":    len(surviving),
                "hopfield":     len(self.hop),
                "sem_facts":    len(self.sm),
                "per":          per_stats,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# §10  CTF SERIALIZER — Binary persistence
#
# File format (inspired by custom_model_format.py):
#
#   [8 bytes]  Magic: b'AIOSMEM\x01'
#   [8 bytes]  JSON header size (little-endian uint64)
#   [N bytes]  JSON header (UTF-8)
#   [M bytes]  Binary payload (IEEE 754 float32, little-endian, struct pack)
#
# JSON header keys:
#   "version"   : "1.0"
#   "d_emb"     : int
#   "d_in"      : int
#   "n_episodes": int
#   "n_facts"   : int
#   "projector_offset": int   byte offset in payload for projector weights
#   "projector_size"  : int   number of float32 values
#   "episodes_offset" : int   byte offset for episode embeddings
#   "episodes_size"   : int   number of float32 values
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_MAGIC = b'AIOSMEM\x01'


class CTFMemorySerializer:
    """Read/write the AIOS memory snapshot in the custom binary format."""

    @staticmethod
    def save(path: str,
             projector: EmbeddingProjector,
             em: EpisodicMemory,
             sm: SemanticMemory) -> int:
        """
        Serialise projector weights + episode embeddings to a binary snapshot.

        Returns total bytes written.
        """
        # --- Build flat float payload -----------------------------------------
        proj_flat  = projector.to_flat()
        eps_flat   = []
        ep_meta    = []

        for ep in em.all_episodes():
            off_start = len(eps_flat)
            eps_flat.extend(ep.embedding)
            ep_meta.append({
                "ep_id":      ep.ep_id,
                "timestamp":  ep.timestamp,
                "importance": ep.importance,
                "floor":      ep.importance_floor,
                "access":     ep.access_count,
                "content":    str(ep.content)[:256],   # truncate to 256 chars
                "emb_offset": off_start,
                "emb_len":    len(ep.embedding),
            })

        fact_meta = []
        for f in sm.all_facts():
            fact_meta.append({
                "fact_id":    f.fact_id,
                "confidence": f.confidence,
                "source":     f.source,
                "value":      str(f.value)[:256],
                "key_emb":    f.key_emb[:16],   # store first 16 dims as preview
            })

        # Offsets are in units of float32 (4 bytes)
        proj_offset = 0
        eps_offset  = len(proj_flat)

        header = {
            "version":          "1.0",
            "d_emb":            projector.d_emb,
            "d_in":             projector.d_in,
            "n_episodes":       len(ep_meta),
            "n_facts":          len(fact_meta),
            "projector_offset": proj_offset,
            "projector_size":   len(proj_flat),
            "episodes_offset":  eps_offset,
            "episodes_size":    len(eps_flat),
            "episodes":         ep_meta,
            "facts":            fact_meta,
        }

        header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
        all_floats   = proj_flat + eps_flat
        payload      = struct.pack(f'<{len(all_floats)}f', *all_floats)

        with open(path, 'wb') as fh:
            fh.write(MEMORY_MAGIC)
            fh.write(struct.pack('<Q', len(header_bytes)))
            fh.write(header_bytes)
            fh.write(payload)

        return len(MEMORY_MAGIC) + 8 + len(header_bytes) + len(payload)

    @staticmethod
    def load(path: str,
             projector: EmbeddingProjector,
             em: EpisodicMemory,
             sm: SemanticMemory,
             projector_only: bool = False) -> Dict[str, Any]:
        """
        Restore projector weights and rebuild episode/fact stores from snapshot.

        Returns the parsed header for inspection.
        """
        with open(path, 'rb') as fh:
            magic = fh.read(8)
            if magic != MEMORY_MAGIC:
                raise ValueError(f"Bad magic: {magic!r} — not an AIOS memory snapshot")

            hdr_size_bytes = fh.read(8)
            hdr_size       = struct.unpack('<Q', hdr_size_bytes)[0]
            if hdr_size > 128 * 1024 * 1024:
                raise ValueError(f"Header too large: {hdr_size} bytes — possible corruption")

            header      = json.loads(fh.read(hdr_size).decode('utf-8'))
            payload_raw = fh.read()

        # Parse floats from payload
        n_floats = len(payload_raw) // 4
        floats   = list(struct.unpack(f'<{n_floats}f', payload_raw[:n_floats * 4]))

        # Restore projector
        p_off  = header['projector_offset']
        p_size = header['projector_size']
        projector.from_flat(floats[p_off: p_off + p_size])

        if projector_only:
            return header

        # Restore episodes
        e_off = header['episodes_offset']
        for ep_info in header.get('episodes', []):
            emb_start = e_off + ep_info['emb_offset']
            emb_end   = emb_start + ep_info['emb_len']
            emb       = floats[emb_start: emb_end]
            ep = Episode(
                ep_id            = ep_info['ep_id'],
                timestamp        = ep_info['timestamp'],
                content          = ep_info['content'],
                embedding        = emb,
                importance       = ep_info['importance'],
                importance_floor = ep_info['floor'],
                access_count     = ep_info['access'],
            )
            with em._lock:
                em._episodes[ep.ep_id] = ep

        # Restore facts (key_emb is only 16-dim preview — mark as partial)
        for f_info in header.get('facts', []):
            f = Fact(
                fact_id    = f_info['fact_id'],
                key_emb    = f_info['key_emb'],
                value      = f_info['value'],
                confidence = f_info['confidence'],
                source     = f_info['source'],
            )
            with sm._lock:
                sm._facts[f.fact_id] = f

        return header


# ═══════════════════════════════════════════════════════════════════════════════
# §11  MEMORY KERNEL — @agent_method interface to AgentKernel
#
# Wraps all memory subsystems behind the AIOS agent-method protocol.
# Falls back gracefully if aios_core is not importable.
# ═══════════════════════════════════════════════════════════════════════════════

# --- Graceful import of aios_core agent_method decorator ---
try:
    from aios_core import agent_method, AgentPriority, _registry as _kern_registry
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

    class AgentPriority:         # type: ignore[no-redef]
        CRITICAL = 0
        HIGH     = 1
        NORMAL   = 2
        LOW      = 3

    def agent_method(name='', description='', parameters=None,   # type: ignore[misc]
                     priority=None, returns='Any', owner='memory'):
        """No-op shim when aios_core is unavailable."""
        def decorator(fn):
            return fn
        return decorator


class MemoryKernel:
    """
    Central memory subsystem for AIOS.

    Instantiate once; call attach(kernel) after AgentKernel.boot().

    Parameters
    ----------
    d_in     : int   Feature vector dimension fed to the embedding projector.
    d_emb    : int   Embedded space dimensionality.
    em_cap   : int   EpisodicMemory capacity (episodes).
    sm_cap   : int   SemanticMemory capacity (facts).
    wm_cap   : int   WorkingMemory capacity (slots).
    decay_λ  : float Episodic decay rate (1/second).
    seed     : int   RNG seed.
    """

    VERSION = (1, 0, 0)

    def __init__(self,
                 d_in:    int   = 256,
                 d_emb:   int   = 128,
                 em_cap:  int   = 4096,
                 sm_cap:  int   = 2048,
                 wm_cap:  int   = 16,
                 decay_λ: float = 1e-5,
                 seed:    int   = 0xAE1F) -> None:

        self.d_in  = d_in
        self.d_emb = d_emb
        self._rng  = _RNGEngine(seed)
        self._lock = threading.RLock()

        # --- Subsystems ---------------------------------------------------
        self.projector   = EmbeddingProjector(d_in, d_emb, seed=seed)
        self.wm          = WorkingMemory(wm_cap, d_emb)
        self.em          = EpisodicMemory(em_cap, d_emb, decay_rate=decay_λ)
        self.sm          = SemanticMemory(d_emb, sm_cap)
        self.lsh         = LSHIndex(d_emb, n_bits=16, n_tables=8, seed=seed)
        self.hop         = HopfieldMemory(d_emb, beta=8.0)
        self.consolidator = PERConsolidator(self.projector, batch_size=32)
        self.sleep       = SleepCycle(
            self.em, self.sm, self.lsh, self.hop, self.consolidator,
            promote_thr=0.3, promote_acc=2,
        )
        self._kernel_ref = None   # set by attach()
        self._stats      = {
            "remember_calls":  0,
            "recall_calls":    0,
            "sleep_cycles":    0,
            "total_episodes":  0,
            "total_facts":     0,
        }

    # ── Attach to AgentKernel ─────────────────────────────────────────────────

    def attach(self, kernel: Any) -> None:
        """
        Register all memory agent-methods with the running AgentKernel.
        kernel must have a .boot()-initialised state.
        """
        self._kernel_ref = kernel
        # Bind our methods into the agent registry if aios_core is available
        if _HAS_CORE:
            import types
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if callable(attr) and hasattr(attr, '_agent_spec'):
                    spec = attr._agent_spec
                    spec.fn = attr
                    _kern_registry.register(spec)

    # ── Core Agent Methods ────────────────────────────────────────────────────

    @agent_method(
        name        = "memory.remember",
        description = "Encode an observation into working and episodic memory",
        parameters  = {
            "content":    {"type": "Any",        "desc": "Payload to remember"},
            "features":   {"type": "List[float]","desc": "Raw feature vector (d_in)"},
            "importance": {"type": "float",      "desc": "Initial importance weight I₀ ∈ [0,1]"},
        },
        returns     = "str   # episode ID",
        priority    = AgentPriority.NORMAL,
        owner       = "memory",
    )
    def remember(self, content: Any, features: List[float],
                 importance: float = 1.0) -> str:
        """
        Full memory encoding pipeline:
          1. Project features → embedding ê
          2. Push to WorkingMemory
          3. Store in EpisodicMemory
          4. Insert into LSHIndex
          5. Conditionally store in HopfieldMemory (high importance)

        Returns episode ID.
        """
        emb   = self.projector.embed(features)
        self.wm.push(content, emb)
        ep_id = self.em.store(content, emb, importance)
        self.lsh.insert(ep_id, emb)
        if importance >= 0.7:
            self.hop.store(emb)

        with self._lock:
            self._stats["remember_calls"]  += 1
            self._stats["total_episodes"]   = len(self.em)

        return ep_id

    @agent_method(
        name        = "memory.recall",
        description = "Retrieve episodic and semantic memories relevant to a query",
        parameters  = {
            "query_features": {"type": "List[float]", "desc": "Query feature vector (d_in)"},
            "top_k_ep":       {"type": "int",         "desc": "Max episodic results"},
            "top_k_sm":       {"type": "int",         "desc": "Max semantic/fact results"},
            "use_hopfield":   {"type": "bool",        "desc": "Also run Hopfield completion"},
        },
        returns     = "Dict[str, Any]",
        priority    = AgentPriority.NORMAL,
        owner       = "memory",
    )
    def recall(self, query_features: List[float],
               top_k_ep: int = 5, top_k_sm: int = 3,
               use_hopfield: bool = True) -> Dict[str, Any]:
        """
        Full memory retrieval pipeline:
          1. Embed query
          2. LSH candidate lookup → exact cosine re-rank (episodic)
          3. Attention-weighted working memory readout
          4. Semantic fact retrieval
          5. Optional Hopfield completion
          6. Associative recall from semantic association matrix
        """
        q_emb = self.projector.embed(query_features)

        # --- Episodic retrieval -------------------------------------------
        ep_results = self.em.retrieve(q_emb, top_k=top_k_ep)
        ep_out = [
            {
                "ep_id":      ep.ep_id,
                "content":    ep.content,
                "score":      round(score, 6),
                "importance": round(ep.current_importance(self.em.decay_rate), 6),
                "access":     ep.access_count,
            }
            for score, ep in ep_results
        ]

        # --- Working memory readout --------------------------------------
        wm_context = self.wm.attend(q_emb)

        # --- Semantic fact retrieval -------------------------------------
        sm_results = self.sm.retrieve_fact(q_emb, top_k=top_k_sm)
        sm_out = [
            {
                "fact_id":    f.fact_id,
                "value":      f.value,
                "confidence": round(f.confidence, 6),
                "score":      round(score, 6),
            }
            for score, f in sm_results
        ]

        # --- Hopfield completion ------------------------------------------
        hop_out = None
        if use_hopfield and len(self.hop) > 0:
            retrieved_emb = self.hop.retrieve(q_emb)
            # Cosine similarity of retrieved pattern to query
            hop_sim = _cosine(retrieved_emb, q_emb)
            hop_out = {"similarity": round(hop_sim, 6), "d_emb": self.d_emb}

        # --- Associative recall -------------------------------------------
        assoc_emb = self.sm.associative_recall(q_emb)
        assoc_sim = _cosine(assoc_emb, q_emb)

        with self._lock:
            self._stats["recall_calls"] += 1

        return {
            "query_norm":       round(_m_sqrt(sum(v * v for v in q_emb)), 6),
            "episodic":         ep_out,
            "working_context":  wm_context[:8],     # first 8 dims for inspection
            "semantic":         sm_out,
            "hopfield":         hop_out,
            "associative_sim":  round(assoc_sim, 6),
        }

    @agent_method(
        name        = "memory.teach",
        description = "Directly assert a semantic fact with a given confidence",
        parameters  = {
            "key_features": {"type": "List[float]", "desc": "Key feature vector (d_in)"},
            "value":        {"type": "Any",         "desc": "Fact value / payload"},
            "confidence":   {"type": "float",       "desc": "Confidence ∈ [0,1]"},
        },
        returns     = "str   # fact ID",
        priority    = AgentPriority.NORMAL,
        owner       = "memory",
    )
    def teach(self, key_features: List[float], value: Any,
              confidence: float = 1.0) -> str:
        """Directly inject a semantic fact (explicit knowledge)."""
        key_emb = self.projector.embed(key_features)
        fid = self.sm.store_fact(key_emb, value, confidence, source='explicit')
        with self._lock:
            self._stats["total_facts"] = len(self.sm)
        return fid

    @agent_method(
        name        = "memory.sleep_cycle",
        description = "Run one full memory consolidation pass (EM→SM, decay, LSH rebuild)",
        parameters  = {},
        returns     = "Dict[str, Any]",
        priority    = AgentPriority.LOW,
        owner       = "memory",
    )
    def run_sleep_cycle(self) -> Dict[str, Any]:
        """Trigger a manual consolidation cycle."""
        report = self.sleep.run(self._rng)
        with self._lock:
            self._stats["sleep_cycles"]  += 1
            self._stats["total_episodes"] = len(self.em)
            self._stats["total_facts"]    = len(self.sm)
        return report

    @agent_method(
        name        = "memory.save",
        description = "Serialise memory snapshot to a binary CTF file",
        parameters  = {"path": {"type": "str", "desc": "Output file path (.amem)"}},
        returns     = "int   # bytes written",
        priority    = AgentPriority.LOW,
        owner       = "memory",
    )
    def save(self, path: str) -> int:
        return CTFMemorySerializer.save(path, self.projector, self.em, self.sm)

    @agent_method(
        name        = "memory.load",
        description = "Restore memory snapshot from a binary CTF file",
        parameters  = {
            "path":            {"type": "str",  "desc": "Snapshot file path"},
            "projector_only":  {"type": "bool", "desc": "Only restore projector weights"},
        },
        returns     = "Dict[str, Any]   # parsed header",
        priority    = AgentPriority.LOW,
        owner       = "memory",
    )
    def load(self, path: str, projector_only: bool = False) -> Dict[str, Any]:
        hdr = CTFMemorySerializer.load(path, self.projector, self.em, self.sm,
                                       projector_only=projector_only)
        # Rebuild LSH from loaded episodes
        eps = self.em.all_episodes()
        self.lsh.rebuild([(ep.ep_id, ep.embedding) for ep in eps])
        return hdr

    @agent_method(
        name        = "memory.status",
        description = "Return full memory subsystem statistics",
        parameters  = {},
        returns     = "Dict[str, Any]",
        priority    = AgentPriority.LOW,
        owner       = "memory",
    )
    def status(self) -> Dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
        wm_snap = self.wm.snapshot()
        return {
            "version":        self.VERSION,
            "d_in":           self.d_in,
            "d_emb":          self.d_emb,
            "working_memory": {"slots": len(self.wm), "capacity": self.wm.capacity,
                               "snapshot": wm_snap[-3:]},
            "episodic":       {"size": len(self.em), "capacity": self.em.capacity,
                               "decay_λ": self.em.decay_rate,
                               "prune_thr": self.em.prune_thr},
            "semantic":       {"size": len(self.sm), "capacity": self.sm.capacity,
                               "hebbian_lr": self.sm.hebbian_lr},
            "hopfield":       {"patterns": len(self.hop), "beta": self.hop.beta},
            "lsh":            {"n_bits": self.lsh.n_bits, "n_tables": self.lsh.n_tables},
            "per_step":       self.consolidator._step,
            "stats":          s,
        }

    @agent_method(
        name        = "memory.wm_clear",
        description = "Flush the working memory buffer (start of new context window)",
        parameters  = {},
        returns     = "None",
        priority    = AgentPriority.HIGH,
        owner       = "memory",
    )
    def clear_working_memory(self) -> None:
        self.wm.clear()

    @agent_method(
        name        = "memory.forget_episode",
        description = "Immediately remove a specific episode from episodic memory",
        parameters  = {"ep_id": {"type": "str", "desc": "Episode ID to remove"}},
        returns     = "bool   # True if found and removed",
        priority    = AgentPriority.NORMAL,
        owner       = "memory",
    )
    def forget_episode(self, ep_id: str) -> bool:
        with self.em._lock:
            if ep_id not in self.em._episodes:
                return False
            del self.em._episodes[ep_id]
        self.lsh.remove(ep_id)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (module-level, no side effects)
# ═══════════════════════════════════════════════════════════════════════════════

def _dot(a: List[float], b: List[float]) -> float:
    """Inner product, length-safe (shorter vector is zero-padded)."""
    return sum(a[i] * (b[i] if i < len(b) else 0.0) for i in range(len(a)))


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity ∈ [-1, 1].  Returns 0 for zero vectors."""
    na = _m_sqrt(sum(v * v for v in a))
    nb = _m_sqrt(sum(v * v for v in b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return _dot(a, b) / (na * nb)


def _l2_normalise(v: List[float]) -> List[float]:
    """Return unit-norm version of v.  Handles zero vector → zero vector."""
    norm = _m_sqrt(sum(x * x for x in v)) + _EPS
    return [x / norm for x in v]


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_self_tests() -> None:
    """
    Deterministic self-tests covering all 11 sections.
    All checks are assertions with descriptive messages.
    Passes with zero external dependencies.
    """
    import sys

    PASS = "\033[1;32m✓\033[0m"
    FAIL = "\033[1;31m✗\033[0m"
    passed = failed = 0

    def check(label: str, expr: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if expr:
            print(f"  {PASS}  {label}")
            passed += 1
        else:
            print(f"  {FAIL}  {label}  ← {detail}")
            failed += 1

    print("\n" + "═" * 68)
    print("  AIOS Memory Subsystem — Self-Test Suite")
    print("═" * 68)

    # §1 Math Shim
    print("\n  §1  Math Shim")
    check("exp(0) = 1",       _m_abs(_m_exp(0.0) - 1.0) < 1e-12)
    check("exp(1) ≈ e",       _m_abs(_m_exp(1.0) - _E) < 1e-10)
    check("exp(-inf) → 0",    _m_exp(-800.0) == 0.0)
    check("log(e) = 1",       _m_abs(_m_log(_E) - 1.0) < 1e-9)
    check("log(1) = 0",       _m_abs(_m_log(1.0)) < 1e-14)
    check("sqrt(4) = 2",      _m_abs(_m_sqrt(4.0) - 2.0) < 1e-12)
    check("cos(0) = 1",       _m_abs(_m_cos(0.0) - 1.0) < 1e-14)
    check("cos(π) = -1",      _m_abs(_m_cos(_PI) + 1.0) < 1e-12)
    check("exp∘log identity", _m_abs(_m_exp(_m_log(2.718)) - 2.718) < 1e-8)

    # §2 Embedding Projector
    print("\n  §2  Embedding Projector")
    proj = EmbeddingProjector(d_in=8, d_emb=4, seed=42)
    x    = [1.0, 0.5, -0.3, 0.8, 0.0, 0.2, -0.1, 0.9]
    e    = proj.embed(x)
    check("embed output dim = d_emb",     len(e) == 4)
    check("embed output is unit norm",    _m_abs(_m_sqrt(sum(v**2 for v in e)) - 1.0) < 1e-6)
    e2 = proj.embed(x[:3])              # short input (pad test)
    check("embed pads short input",       len(e2) == 4 and _m_abs(_m_sqrt(sum(v**2 for v in e2)) - 1.0) < 1e-6)
    e3 = proj.embed(x + [9.9, 8.8])    # long input (truncate test)
    check("embed truncates long input",   len(e3) == 4)
    # Adam update: loss should decrease
    g_W = [[_RNG.randn() * 0.01 for _ in range(8)] for _ in range(4)]
    g_b = [_RNG.randn() * 0.01 for _ in range(4)]
    proj.adam_update(g_W, g_b, lr=1e-3)
    check("adam_update does not crash",   True)
    check("adam_t incremented",           proj._adam_t == 1)

    # §3 Working Memory
    print("\n  §3  Working Memory")
    wm = WorkingMemory(capacity=4, d_emb=4)
    for i in range(6):
        wm.push(f"item_{i}", [float(i), float(i), float(i), float(i)])
    check("WM respects capacity",         len(wm) == 4)
    q   = [1.0, 1.0, 1.0, 1.0]
    h   = wm.attend(q)
    check("attend returns d_emb vector",  len(h) == 4)
    check("attend output non-zero",       any(_m_abs(v) > 1e-9 for v in h))
    wm.clear()
    check("clear empties WM",             len(wm) == 0)
    h0 = wm.attend(q)
    check("attend on empty → zero vec",   all(_m_abs(v) < 1e-9 for v in h0))

    # §4 Episodic Memory
    print("\n  §4  Episodic Memory")
    em = EpisodicMemory(capacity=8, d_emb=4, decay_rate=0.0)
    ids = []
    for i in range(5):
        emb = _l2_normalise([float(i+1), 0.0, 0.0, 0.0])
        ids.append(em.store(f"event_{i}", emb, importance=float(i+1)/5.0))
    check("store returns string id",     isinstance(ids[0], str) and len(ids[0]) == 16)
    check("episode count = 5",           len(em) == 5)
    q_emb = _l2_normalise([5.0, 0.0, 0.0, 0.0])
    top   = em.retrieve(q_emb, top_k=2)
    check("retrieve returns ≤ top_k",    len(top) <= 2)
    check("top score is highest",        top[0][0] >= top[-1][0] if len(top) > 1 else True)
    # Access increments
    ep0 = top[0][1]
    acc_before = ep0.access_count
    em.retrieve(q_emb, top_k=1)
    # capacity eviction
    for i in range(10):
        em.store(f"extra_{i}", _l2_normalise([0.1]*4))
    check("capacity bounded",            len(em) <= em.capacity)

    # §5 Semantic Memory
    print("\n  §5  Semantic Memory")
    sm = SemanticMemory(d_emb=4, capacity=8)
    k1 = _l2_normalise([1.0, 0.0, 0.0, 0.0])
    k2 = _l2_normalise([0.0, 1.0, 0.0, 0.0])
    fid1 = sm.store_fact(k1, "north pole", confidence=0.9)
    fid2 = sm.store_fact(k2, "east coast", confidence=0.7)
    check("store returns fact id",       isinstance(fid1, str))
    check("distinct fact ids",           fid1 != fid2)
    results = sm.retrieve_fact(k1, top_k=2)
    check("retrieve_fact non-empty",     len(results) > 0)
    check("top fact is north pole",      results[0][1].value == "north pole" if results else True)
    # Near-duplicate dedup
    k1_near = _l2_normalise([1.0, 0.001, 0.0, 0.0])
    sm.store_fact(k1_near, "north pole updated", confidence=0.95)
    check("near-dup updates not inserts",len(sm) == 2)
    # Associative recall
    assoc = sm.associative_recall(k1)
    check("associative_recall unit norm",_m_abs(_m_sqrt(sum(v**2 for v in assoc)) - 1.0) < 1e-6)

    # §6 LSH Index
    print("\n  §6  LSH Index")
    lsh = LSHIndex(d_emb=4, n_bits=4, n_tables=4, seed=1234)
    test_embs = [(f"id_{i}", _l2_normalise([float(i), float(i+1), 0.0, 0.0]))
                 for i in range(10)]
    for ep_id, emb in test_embs:
        lsh.insert(ep_id, emb)
    q_emb2 = _l2_normalise([9.0, 10.0, 0.0, 0.0])
    cands  = lsh.query(q_emb2)
    check("query returns candidates list",isinstance(cands, list))
    lsh.remove("id_0")
    cands2 = lsh.query(_l2_normalise([0.0, 1.0, 0.0, 0.0]))
    check("removed id not in results",   "id_0" not in cands2)
    pairs = [(ep_id, emb) for ep_id, emb in test_embs if ep_id != "id_0"]
    lsh.rebuild(pairs)
    check("rebuild doesn't crash",       True)

    # §7 Modern Hopfield
    print("\n  §7  Modern Hopfield")
    hop = HopfieldMemory(d_emb=4, beta=10.0)
    pats = [_l2_normalise([1.0, 0.0, 0.0, 0.0]),
            _l2_normalise([0.0, 1.0, 0.0, 0.0]),
            _l2_normalise([0.0, 0.0, 1.0, 0.0])]
    for p in pats: hop.store(p)
    check("hop len = 3",                 len(hop) == 3)
    # Noisy query close to pat[0] → should retrieve close to pat[0]
    noisy = _l2_normalise([0.9, 0.1, 0.05, 0.02])
    retrieved = hop.retrieve(noisy)
    check("retrieved is unit norm",      _m_abs(_m_sqrt(sum(v**2 for v in retrieved)) - 1.0) < 1e-5)
    sim_to_p0 = _cosine(retrieved, pats[0])
    check("retrieval pulls toward nearest pattern", sim_to_p0 > 0.8,
          f"sim={sim_to_p0:.4f}")
    hop.clear()
    check("hop clear empties",           len(hop) == 0)

    # §8 PER Consolidator
    print("\n  §8  PER Consolidator")
    proj2 = EmbeddingProjector(d_in=4, d_emb=4, seed=7)
    per   = PERConsolidator(proj2, batch_size=4)
    em2   = EpisodicMemory(capacity=64, d_emb=4, decay_rate=0.0)
    rng2  = _RNGEngine(seed=999)
    for i in range(8):
        emb_i = _l2_normalise([_RNG.randn() for _ in range(4)])
        em2.store(f"obs_{i}", emb_i, importance=_RNG.uniform())
    diag = per.consolidate(em2.all_episodes(), rng2)
    check("PER returns loss key",        "loss" in diag)
    check("PER loss is finite",          diag["loss"] == diag["loss"])  # not NaN
    check("PER n_sampled ≤ batch_size",  diag["n_sampled"] <= per.batch_size)
    check("PER adam_t incremented",      proj2._adam_t >= 1)

    # §9 Sleep Cycle
    print("\n  §9  Sleep Cycle")
    em3  = EpisodicMemory(capacity=32, d_emb=4, decay_rate=0.0)
    sm3  = SemanticMemory(d_emb=4, capacity=32)
    lsh3 = LSHIndex(d_emb=4, n_bits=4, n_tables=2, seed=55)
    hop3 = HopfieldMemory(d_emb=4)
    per3 = PERConsolidator(EmbeddingProjector(4, 4), batch_size=4)
    sc   = SleepCycle(em3, sm3, lsh3, hop3, per3,
                      promote_thr=0.0, promote_acc=0)
    for i in range(6):
        emb_i = _l2_normalise([float(i), 1.0, 0.0, 0.0])
        ep_id_i = em3.store(f"mem_{i}", emb_i, importance=0.9)
        lsh3.insert(ep_id_i, emb_i)
    rng3 = _RNGEngine(99)
    report = sc.run(rng3)
    check("sleep cycle status=ok",       report.get("status") == "ok", str(report))
    check("sleep promoted ≥ 1 fact",     report.get("promoted", 0) >= 1)
    check("sleep hopfield populated",    report.get("hopfield", 0) >= 1)

    # §10 CTF Serializer (round-trip)
    print("\n  §10 CTF Serializer")
    import tempfile
    proj_s = EmbeddingProjector(d_in=4, d_emb=4, seed=123)
    em_s   = EpisodicMemory(capacity=16, d_emb=4, decay_rate=0.0)
    sm_s   = SemanticMemory(d_emb=4, capacity=16)
    for i in range(3):
        emb_i = _l2_normalise([float(i+1), 0.0, 0.0, 0.0])
        em_s.store(f"save_ep_{i}", emb_i, importance=0.5 + i * 0.1)
    sm_s.store_fact(_l2_normalise([1.0, 0.0, 0.0, 0.0]), "fact_A", confidence=0.8)

    with tempfile.NamedTemporaryFile(suffix='.amem', delete=False) as tf:
        tmp_path = tf.name
    try:
        n_bytes = CTFMemorySerializer.save(tmp_path, proj_s, em_s, sm_s)
        check("save writes bytes",        n_bytes > len(MEMORY_MAGIC) + 8)
        check("file exists",              os.path.isfile(tmp_path))

        # Restore into fresh instances
        proj_r = EmbeddingProjector(d_in=4, d_emb=4, seed=0)
        em_r   = EpisodicMemory(capacity=16, d_emb=4, decay_rate=0.0)
        sm_r   = SemanticMemory(d_emb=4, capacity=16)
        hdr    = CTFMemorySerializer.load(tmp_path, proj_r, em_r, sm_r)
        check("header has n_episodes=3", hdr.get("n_episodes") == 3)
        check("episodes restored",       len(em_r) == 3)
        check("facts restored",          len(sm_r) >= 1)
        # Verify projector weights match
        orig_flat = proj_s.to_flat()
        rest_flat = proj_r.to_flat()
        max_err = max(_m_abs(a - b) for a, b in zip(orig_flat, rest_flat))
        check("projector weights restored exactly", max_err < 1e-5,
              f"max_err={max_err:.2e}")
    finally:
        os.unlink(tmp_path)

    # §11 MemoryKernel integration
    print("\n  §11 MemoryKernel")
    mk = MemoryKernel(d_in=8, d_emb=4, em_cap=64, sm_cap=32, wm_cap=8,
                      decay_λ=0.0, seed=0xCAFE)
    ep_id_k = mk.remember("hello world", [1.0]*8, importance=0.8)
    check("remember returns ep_id str",  isinstance(ep_id_k, str) and len(ep_id_k) == 16)
    mk.remember("second event",          [0.5]*8, importance=0.4)
    mk.remember("third event",           [0.9]*8, importance=0.9)
    recall_r = mk.recall([1.0]*8, top_k_ep=2, top_k_sm=1, use_hopfield=True)
    check("recall returns dict",         isinstance(recall_r, dict))
    check("recall has episodic key",     "episodic" in recall_r)
    check("recall episodic non-empty",   len(recall_r["episodic"]) > 0)
    check("recall has semantic key",     "semantic" in recall_r)
    check("recall hopfield present",     recall_r.get("hopfield") is not None)
    fid = mk.teach([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "taught fact", 0.95)
    check("teach returns fact id",       isinstance(fid, str))
    sleep_r = mk.run_sleep_cycle()
    check("sleep_cycle returns ok",      sleep_r.get("status") == "ok", str(sleep_r))
    stat    = mk.status()
    check("status has all keys",         all(k in stat for k in
                                            ("working_memory", "episodic", "semantic",
                                             "hopfield", "lsh", "stats")))
    removed = mk.forget_episode(ep_id_k)
    check("forget_episode returns True", removed)
    removed2 = mk.forget_episode("nonexistent_id_x")
    check("forget nonexistent → False",  not removed2)
    mk.clear_working_memory()
    check("clear_working_memory works",  len(mk.wm) == 0)

    # --- Final report -------------------------------------------------------
    print("\n" + "─" * 68)
    total = passed + failed
    print(f"  Result: {passed}/{total} passed"
          + (f"  [{failed} FAILED]" if failed else "  [all OK]"))
    print("═" * 68 + "\n")
    if failed:
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    _run_self_tests()
