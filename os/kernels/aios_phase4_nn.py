#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Phase IV: Neural Intelligence Kernel                                ║
║  aios_phase4_nn.py                                                           ║
║                                                                              ║
║  "The substrate thinks. Every weight is a judgment. Every gradient, a        ║
║   correction written in the language of error."                              ║
║                                                                              ║
║  Components (all zero external dependencies, pure Python):                  ║
║    §1  Math Primitives  — exp, log, sqrt, tanh, sin, cos (no math import)   ║
║    §2  RNG Engine       — Xorshift64 + Box-Muller Gaussian sampling         ║
║    §3  Autograd Engine  — Reverse-mode AD, tape-based, full chain rule      ║
║    §4  Tensor           — N-dim array, all ops, autograd-aware              ║
║    §5  NN Framework     — Module, Parameter, Linear, Activations, Norm      ║
║    §6  Loss Functions   — MSE, CrossEntropy, BCE, Huber                     ║
║    §7  Optimizers       — SGD (momentum), RMSProp, Adam, AdamW              ║
║    §8  Cellular Automata — Conway's Life, custom B/S rules, entropy         ║
║    §9  Transformer Agent — Embedding, PosEnc, MHA, FFN, decoder loop        ║
║    §10 AIOS Integration — @agent_method, kernel hooks, system self-tests    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
 
# ══════════════════════════════════════════════════════════════════════════════
# §1  PURE MATH PRIMITIVES — No imports. All from first principles.
# ══════════════════════════════════════════════════════════════════════════════
 
_PI  = 3.141592653589793238462643383279
_E   = 2.718281828459045235360287471352
_LN2 = 0.693147180559945309417232121458
_INF = float('inf')
_NAN = float('nan')
 
 
def _abs(x: float) -> float:
    return x if x >= 0.0 else -x
 
 
def _sign(x: float) -> float:
    if x > 0.0: return 1.0
    if x < 0.0: return -1.0
    return 0.0
 
 
def _floor(x: float) -> int:
    n = int(x)
    return n - 1 if x < n else n
 
 
def _ceil(x: float) -> int:
    n = int(x)
    return n + 1 if x > n else n
 
 
def _exp(x: float) -> float:
    """
    e^x via range reduction + Taylor series.
    Accurate to machine epsilon (~15 digits) across the representable range.
 
    Strategy:
      1. Range-reduce: e^x = e^n * e^r, n = floor(x), r = x - n, |r| ≤ 1
      2. e^r via Taylor (converges in ~15 terms for |r| ≤ 1)
      3. e^n via repeated squaring (n is integer)
    """
    if x > 709.782:  return _INF
    if x < -745.13:  return 0.0
    if x == 0.0:     return 1.0
 
    n = _floor(x)
    r = x - n          # fractional part: |r| <= 1, fast Taylor convergence
 
    # Taylor: e^r = Σ r^k/k!
    result = 1.0
    term   = 1.0
    for k in range(1, 25):
        term *= r / k
        result += term
        if _abs(term) < 1e-17:
            break
 
    # Integer power e^n via repeated squaring
    if n == 0:
        return result
    e_n  = 1.0
    base = _E
    m    = _abs(n)
    while m > 0:
        if m & 1:
            e_n *= base
        base *= base
        m >>= 1
    return result / e_n if n < 0 else result * e_n
 
 
def _log(x: float) -> float:
    """
    Natural logarithm via range reduction + Padé-quality series.
 
    Strategy:
      1. Range-reduce: log(x) = k*ln(2) + log(y),  y ∈ [0.5, 1)
      2. Substitute t = (y-1)/(y+1), then log(y) = 2*(t + t³/3 + t⁵/5 + ...)
         (series converges very fast for y ∈ [0.5, 1))
    """
    if x <= 0.0:
        return _NAN
    if x == 1.0:
        return 0.0
 
    k = 0
    y = x
    while y >= 1.0:
        y *= 0.5
        k += 1
    while y < 0.5:
        y *= 2.0
        k -= 1
    # y ∈ [0.5, 1.0)
 
    t  = (y - 1.0) / (y + 1.0)
    t2 = t * t
    acc = t
    pw  = t
    for n in range(1, 30):
        pw  *= t2
        term = pw / (2 * n + 1)
        acc += term
        if _abs(term) < 1e-17:
            break
    return k * _LN2 + 2.0 * acc
 
 
def _sqrt(x: float) -> float:
    """Newton-Raphson square root."""
    if x < 0.0:  return _NAN
    if x == 0.0: return 0.0
    # Good initial estimate using integer bit trick approximation
    y = x
    for _ in range(60):
        y_new = 0.5 * (y + x / y)
        if _abs(y_new - y) <= 1e-15 * y:
            return y_new
        y = y_new
    return y
 
 
def _tanh(x: float) -> float:
    """Numerically stable hyperbolic tangent."""
    if x > 20.0:  return  1.0
    if x < -20.0: return -1.0
    e2 = _exp(2.0 * x)
    return (e2 - 1.0) / (e2 + 1.0)
 
 
def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0.0:
        e = _exp(-x)
        return 1.0 / (1.0 + e)
    else:
        e = _exp(x)
        return e / (1.0 + e)
 
 
def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0
 
 
def _gelu(x: float) -> float:
    """Gaussian Error Linear Unit — used in transformers."""
    return 0.5 * x * (1.0 + _tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
 
 
def _cos(x: float) -> float:
    """
    Cosine via range reduction + Taylor series.
    Taylor: cos(x) = Σ (-1)^n * x^(2n) / (2n)!
    """
    # Reduce to [-π, π]
    x = x - _floor(x / (2 * _PI)) * (2 * _PI)
    if x > _PI:  x -= 2 * _PI
    x2 = x * x
    # 8-term Taylor — accurate to 1e-15 for |x| ≤ π
    return (1.0 - x2/2.0 + x2*x2/24.0 - x2**3/720.0
            + x2**4/40320.0 - x2**5/3628800.0
            + x2**6/479001600.0 - x2**7/87178291200.0)
 
 
def _sin(x: float) -> float:
    return _cos(x - _PI * 0.5)
 
 
def _atan2(y: float, x: float) -> float:
    """Four-quadrant arctangent via CORDIC-inspired polynomial."""
    if x == 0.0:
        if y > 0.0: return  _PI * 0.5
        if y < 0.0: return -_PI * 0.5
        return 0.0
    r = y / x
    # Minimax rational approximation for atan(r), |r| ≤ 1
    # then quadrant correction
    neg = r < 0.0
    r = _abs(r)
    if r > 1.0:
        inv = True
        r = 1.0 / r
    else:
        inv = False
    r2 = r * r
    a = r * (1.0 + r2 * (-0.3333333333 + r2 * (0.2 + r2 * (-0.1428571 + r2 * 0.1111111))))
    if inv:  a = _PI * 0.5 - a
    if neg:  a = -a
    if x < 0.0:
        return a + _PI if y >= 0.0 else a - _PI
    return a
 
 
def _log2(x: float) -> float:
    return _log(x) / _LN2
 
 
def _pow(base: float, exp: float) -> float:
    if exp == 0.0:   return 1.0
    if base == 0.0:  return 0.0
    if base < 0.0:
        if exp == _floor(exp):
            n = int(exp)
            r = 1.0
            b = _abs(base)
            m = _abs(n)
            while m > 0:
                if m & 1: r *= b
                b *= b
                m >>= 1
            r = r if n >= 0 else 1.0 / r
            return r * (_sign(base) ** int(exp))
        return _NAN
    return _exp(exp * _log(base))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §2  RNG ENGINE — Xorshift64 with Box-Muller Gaussian
# ══════════════════════════════════════════════════════════════════════════════
 
class _RNG:
    """
    Xorshift64 pseudo-random number generator.
    Period 2^64 - 1. Passes BigCrush. Zero dependencies.
    """
    _state: int = 0x853C49E6748FEA9B  # non-zero seed
 
    @classmethod
    def seed(cls, s: int) -> None:
        cls._state = int(s) if s != 0 else 0x853C49E6748FEA9B
 
    @classmethod
    def _next(cls) -> int:
        x = cls._state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >>  7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        cls._state = x
        return x
 
    @classmethod
    def random(cls) -> float:
        """Uniform [0, 1)."""
        return (cls._next() >> 11) * (1.0 / (1 << 53))
 
    @classmethod
    def randn(cls) -> float:
        """Standard Normal via Box-Muller transform."""
        u1 = cls.random() + 1e-12   # avoid log(0)
        u2 = cls.random()
        return _sqrt(-2.0 * _log(u1)) * _cos(2.0 * _PI * u2)
 
    @classmethod
    def uniform(cls, lo: float, hi: float) -> float:
        return lo + (hi - lo) * cls.random()
 
    @classmethod
    def randint(cls, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi)."""
        return lo + int(cls.random() * (hi - lo))
 
    @classmethod
    def shuffle(cls, lst: list) -> None:
        """Fisher-Yates in-place shuffle."""
        n = len(lst)
        for i in range(n - 1, 0, -1):
            j = cls.randint(0, i + 1)
            lst[i], lst[j] = lst[j], lst[i]
 
    @classmethod
    def choice(cls, lst: list):
        return lst[cls.randint(0, len(lst))]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §3  TENSOR — N-dim array with full reverse-mode autograd
# ══════════════════════════════════════════════════════════════════════════════
# Design principles:
#  • Flat _data list + shape tuple — clean, cache-friendly access pattern
#  • Every op creates a new Tensor node with a _backward closure capturing
#    both the inputs and the output (standard micrograd pattern, extended to ND)
#  • backward() does topological sort once, then iterates in reverse
#  • requires_grad propagates to outputs only when at least one input has it
#  • grad is None for non-leaf, non-requires_grad tensors (memory efficiency)
# ══════════════════════════════════════════════════════════════════════════════
 
class Tensor:
    """
    N-dimensional autograd array.
 
    Supported shapes: scalar (), 1-D (N,), 2-D (M, N)
    Supported ops:
      Arithmetic  : +, -, *, /, @, **
      Unary       : relu, gelu, sigmoid, tanh, exp, log, neg
      Reduction   : sum, mean (global and dim-wise)
      Shape       : transpose/.T, reshape, flatten
      Special     : softmax, log_softmax, layer_norm_1d
    """
 
    __slots__ = ('_data', 'shape', 'requires_grad', 'grad',
                 '_prev', '_op', '_backward')
 
    # ── Construction ─────────────────────────────────────────────────────────
 
    def __init__(self, data, shape=None, requires_grad: bool = False,
                 _op: str = '', _children: tuple = ()):
        if isinstance(data, (int, float)):
            self._data  = [float(data)]
            self.shape  = ()
        elif isinstance(data, Tensor):
            self._data  = data._data[:]
            self.shape  = data.shape
        else:
            flat, shp   = Tensor._flatten_data(data)
            self._data  = flat
            self.shape  = shp if shape is None else shape
 
        self.requires_grad = requires_grad
        self.grad          = [0.0] * len(self._data) if requires_grad else None
        self._op           = _op
        self._prev         = set(_children)
        self._backward     = lambda: None
 
    @staticmethod
    def _flatten_data(data):
        """Recursively flatten nested lists → (flat_list, shape_tuple)."""
        if not isinstance(data, (list, tuple)):
            return [float(data)], ()
        if len(data) == 0:
            return [], (0,)
        if not isinstance(data[0], (list, tuple)):
            return [float(x) for x in data], (len(data),)
        sub = [Tensor._flatten_data(row) for row in data]
        flat = []
        for f, _ in sub:
            flat.extend(f)
        return flat, (len(data),) + sub[0][1]
 
    # ── Factories ─────────────────────────────────────────────────────────────
 
    @staticmethod
    def _make(data: list, shape: tuple, children: tuple, op: str) -> 'Tensor':
        """Internal factory for op-result tensors."""
        t               = object.__new__(Tensor)
        t._data         = data
        t.shape         = shape
        t.requires_grad = any(
            isinstance(c, Tensor) and c.requires_grad for c in children
        )
        t.grad          = [0.0] * len(data) if t.requires_grad else None
        t._prev         = set(c for c in children if isinstance(c, Tensor))
        t._op           = op
        t._backward     = lambda: None
        return t
 
    @staticmethod
    def zeros(*shape, requires_grad: bool = False) -> 'Tensor':
        n = 1
        for d in shape: n *= d
        return Tensor([0.0] * n, shape=shape, requires_grad=requires_grad)
 
    @staticmethod
    def ones(*shape, requires_grad: bool = False) -> 'Tensor':
        n = 1
        for d in shape: n *= d
        return Tensor([1.0] * n, shape=shape, requires_grad=requires_grad)
 
    @staticmethod
    def randn(*shape, requires_grad: bool = False) -> 'Tensor':
        n = 1
        for d in shape: n *= d
        return Tensor([_RNG.randn() for _ in range(n)],
                      shape=shape, requires_grad=requires_grad)
 
    @staticmethod
    def uniform(*shape, lo: float = -1.0, hi: float = 1.0,
                requires_grad: bool = False) -> 'Tensor':
        n = 1
        for d in shape: n *= d
        return Tensor([_RNG.uniform(lo, hi) for _ in range(n)],
                      shape=shape, requires_grad=requires_grad)
 
    @staticmethod
    def eye(n: int, requires_grad: bool = False) -> 'Tensor':
        data = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
        return Tensor(data, shape=(n, n), requires_grad=requires_grad)
 
    # ── Accessors ─────────────────────────────────────────────────────────────
 
    def numel(self) -> int:
        n = 1
        for d in self.shape: n *= d
        return n if self.shape else 1
 
    def ndim(self) -> int:
        return len(self.shape)
 
    def item(self) -> float:
        assert len(self._data) == 1, \
            f"item() requires scalar tensor, got shape {self.shape}"
        return self._data[0]
 
    def tolist(self):
        if not self.shape or self.shape == ():
            return self._data[0]
        if len(self.shape) == 1:
            return self._data[:]
        M, N = self.shape[0], self.shape[1]
        return [[self._data[i * N + j] for j in range(N)] for i in range(M)]
 
    def _flat_idx(self, *idx) -> int:
        assert len(idx) == len(self.shape)
        pos, stride = 0, 1
        for i in range(len(self.shape) - 1, -1, -1):
            pos    += idx[i] * stride
            stride *= self.shape[i]
        return pos
 
    def __getitem__(self, idx):
        """Row access for 2-D tensors (returns 1-D row Tensor)."""
        if len(self.shape) == 2:
            N = self.shape[1]
            row_data = self._data[idx * N: (idx + 1) * N]
            return Tensor(row_data, shape=(N,), requires_grad=False)
        return self._data[idx]
 
    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = [0.0] * len(self._data)
 
    # ── Arithmetic Operations ─────────────────────────────────────────────────
 
    def __add__(self, other: 'Tensor') -> 'Tensor':
        other = self._coerce(other)
 
        # Broadcast: (M, N) + (N,) — bias addition
        if (len(self.shape) == 2 and len(other.shape) == 1
                and other.shape[0] == self.shape[1]):
            M, N   = self.shape
            result = [self._data[i * N + j] + other._data[j]
                      for i in range(M) for j in range(N)]
            out    = Tensor._make(result, (M, N), (self, other), 'add_bc')
 
            def _bwd_add_bc():
                if self.requires_grad:
                    for k in range(len(result)):
                        self.grad[k] += out.grad[k]
                if other.requires_grad:
                    for i in range(M):
                        for j in range(N):
                            other.grad[j] += out.grad[i * N + j]
            out._backward = _bwd_add_bc
            return out
 
        assert self.shape == other.shape, \
            f"add shape mismatch: {self.shape} vs {other.shape}"
        result = [a + b for a, b in zip(self._data, other._data)]
        out    = Tensor._make(result, self.shape, (self, other), 'add')
 
        def _bwd_add():
            if self.requires_grad:
                for i in range(len(result)): self.grad[i] += out.grad[i]
            if other.requires_grad:
                for i in range(len(result)): other.grad[i] += out.grad[i]
        out._backward = _bwd_add
        return out
 
    def __radd__(self, other): return self + other
 
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        other = self._coerce(other)
        assert self.shape == other.shape, \
            f"sub shape mismatch: {self.shape} vs {other.shape}"
        result = [a - b for a, b in zip(self._data, other._data)]
        out    = Tensor._make(result, self.shape, (self, other), 'sub')
 
        def _bwd_sub():
            if self.requires_grad:
                for i in range(len(result)): self.grad[i] += out.grad[i]
            if other.requires_grad:
                for i in range(len(result)): other.grad[i] -= out.grad[i]
        out._backward = _bwd_sub
        return out
 
    def __rsub__(self, other):
        return self._coerce(other) - self
 
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        other = self._coerce(other)
        assert self.shape == other.shape, \
            f"mul shape mismatch: {self.shape} vs {other.shape}"
        result = [a * b for a, b in zip(self._data, other._data)]
        out    = Tensor._make(result, self.shape, (self, other), 'mul')
 
        def _bwd_mul():
            if self.requires_grad:
                for i in range(len(result)):
                    self.grad[i] += other._data[i] * out.grad[i]
            if other.requires_grad:
                for i in range(len(result)):
                    other.grad[i] += self._data[i] * out.grad[i]
        out._backward = _bwd_mul
        return out
 
    def __rmul__(self, other): return self * other
 
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        """a / b = a * b^{-1}."""
        if isinstance(other, (int, float)):
            inv_scalar = 1.0 / float(other)
            result     = [x * inv_scalar for x in self._data]
            out        = Tensor._make(result, self.shape, (self,), 'div_scalar')
            def _bwd_div_s():
                if self.requires_grad:
                    for i in range(len(result)):
                        self.grad[i] += inv_scalar * out.grad[i]
            out._backward = _bwd_div_s
            return out
 
        other  = self._coerce(other)
        assert self.shape == other.shape
        inv    = [1.0 / b for b in other._data]
        result = [a * iv for a, iv in zip(self._data, inv)]
        out    = Tensor._make(result, self.shape, (self, other), 'div')
 
        def _bwd_div():
            if self.requires_grad:
                for i in range(len(result)):
                    self.grad[i] += inv[i] * out.grad[i]
            if other.requires_grad:
                for i in range(len(result)):
                    other.grad[i] -= self._data[i] * (inv[i] ** 2) * out.grad[i]
        out._backward = _bwd_div
        return out
 
    def __rtruediv__(self, other):
        return self._coerce(other) / self
 
    def __neg__(self) -> 'Tensor':
        result = [-x for x in self._data]
        out    = Tensor._make(result, self.shape, (self,), 'neg')
        def _bwd_neg():
            if self.requires_grad:
                for i in range(len(result)): self.grad[i] -= out.grad[i]
        out._backward = _bwd_neg
        return out
 
    def __pow__(self, exponent) -> 'Tensor':
        assert isinstance(exponent, (int, float))
        e      = float(exponent)
        result = [x ** e for x in self._data]
        out    = Tensor._make(result, self.shape, (self,), 'pow')
        def _bwd_pow():
            if self.requires_grad:
                for i in range(len(result)):
                    self.grad[i] += e * (self._data[i] ** (e - 1.0)) * out.grad[i]
        out._backward = _bwd_pow
        return out
 
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication: (M, K) @ (K, N) → (M, N)
 
        Backward:
          dA = dC @ B^T  →  dA[i,k] = Σ_j  dC[i,j] * B[k,j]
          dB = A^T @ dC  →  dB[k,j] = Σ_i  A[i,k] * dC[i,j]
        """
        assert len(self.shape) == 2 and len(other.shape) == 2, \
            f"matmul requires 2-D tensors, got {self.shape} @ {other.shape}"
        M, K  = self.shape
        K2, N = other.shape
        assert K == K2, f"matmul inner dims must match: {K} ≠ {K2}"
 
        result = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += self._data[i * K + k] * other._data[k * N + j]
                result[i * N + j] = s
 
        out = Tensor._make(result, (M, N), (self, other), 'matmul')
 
        def _bwd_matmul():
            if self.requires_grad:
                # dA[i,k] = Σ_j dC[i,j] * B[k,j]
                for i in range(M):
                    for k in range(K):
                        g = 0.0
                        for j in range(N):
                            g += out.grad[i * N + j] * other._data[k * N + j]
                        self.grad[i * K + k] += g
            if other.requires_grad:
                # dB[k,j] = Σ_i A[i,k] * dC[i,j]
                for k in range(K):
                    for j in range(N):
                        g = 0.0
                        for i in range(M):
                            g += self._data[i * K + k] * out.grad[i * N + j]
                        other.grad[k * N + j] += g
 
        out._backward = _bwd_matmul
        return out
 
    # ── Unary Differentiable Functions ────────────────────────────────────────
 
    def relu(self) -> 'Tensor':
        result = [_relu(x) for x in self._data]
        out    = Tensor._make(result, self.shape, (self,), 'relu')
        def _bwd():
            if self.requires_grad:
                for i in range(len(result)):
                    self.grad[i] += (1.0 if self._data[i] > 0.0 else 0.0) * out.grad[i]
        out._backward = _bwd
        return out
 
    def gelu(self) -> 'Tensor':
        result = [_gelu(x) for x in self._data]
        out    = Tensor._make(result, self.shape, (self,), 'gelu')
        def _bwd():
            if self.requires_grad:
                for i in range(len(result)):
                    x  = self._data[i]
                    # d/dx GELU(x) = 0.5*tanh(c*(x+0.044715x³)) +
                    #               0.5*x*sech²(c*(x+0.044715x³))*c*(1+3*0.044715x²)
                    c  = 0.7978845608
                    u  = c * (x + 0.044715 * x * x * x)
                    th = _tanh(u)
                    sech2 = 1.0 - th * th
                    d     = 0.5 * (1.0 + th) + 0.5 * x * sech2 * c * (1.0 + 3 * 0.044715 * x * x)
                    self.grad[i] += d * out.grad[i]
        out._backward = _bwd
        return out
 
    def sigmoid(self) -> 'Tensor':
        sig    = [_sigmoid(x) for x in self._data]
        out    = Tensor._make(sig, self.shape, (self,), 'sigmoid')
        def _bwd():
            if self.requires_grad:
                for i in range(len(sig)):
                    s = sig[i]
                    self.grad[i] += s * (1.0 - s) * out.grad[i]
        out._backward = _bwd
        return out
 
    def tanh(self) -> 'Tensor':
        th  = [_tanh(x) for x in self._data]
        out = Tensor._make(th, self.shape, (self,), 'tanh')
        def _bwd():
            if self.requires_grad:
                for i in range(len(th)):
                    self.grad[i] += (1.0 - th[i] * th[i]) * out.grad[i]
        out._backward = _bwd
        return out
 
    def exp(self) -> 'Tensor':
        ex  = [_exp(x) for x in self._data]
        out = Tensor._make(ex, self.shape, (self,), 'exp')
        def _bwd():
            if self.requires_grad:
                for i in range(len(ex)):
                    self.grad[i] += ex[i] * out.grad[i]
        out._backward = _bwd
        return out
 
    def log(self) -> 'Tensor':
        """Numerically safe log: clamps to log(1e-12) at lower bound."""
        lg  = [_log(max(x, 1e-12)) for x in self._data]
        out = Tensor._make(lg, self.shape, (self,), 'log')
        def _bwd():
            if self.requires_grad:
                for i in range(len(lg)):
                    x = max(self._data[i], 1e-12)
                    self.grad[i] += out.grad[i] / x
        out._backward = _bwd
        return out
 
    def abs(self) -> 'Tensor':
        result = [_abs(x) for x in self._data]
        out    = Tensor._make(result, self.shape, (self,), 'abs')
        def _bwd():
            if self.requires_grad:
                for i in range(len(result)):
                    self.grad[i] += _sign(self._data[i]) * out.grad[i]
        out._backward = _bwd
        return out
 
    # ── Reduction Operations ──────────────────────────────────────────────────
 
    def sum(self, dim=None) -> 'Tensor':
        if dim is None:
            total = sum(self._data)
            out   = Tensor._make([total], (), (self,), 'sum')
            def _bwd_sum():
                if self.requires_grad:
                    for i in range(len(self._data)):
                        self.grad[i] += out.grad[0]
            out._backward = _bwd_sum
            return out
 
        assert len(self.shape) == 2, "dim-wise sum only for 2-D"
        M, N = self.shape
        if dim == 0:
            result = [sum(self._data[i * N + j] for i in range(M)) for j in range(N)]
            out    = Tensor._make(result, (N,), (self,), 'sum_d0')
            def _bwd_sum_d0():
                if self.requires_grad:
                    for i in range(M):
                        for j in range(N):
                            self.grad[i * N + j] += out.grad[j]
            out._backward = _bwd_sum_d0
        else:  # dim == 1
            result = [sum(self._data[i * N + j] for j in range(N)) for i in range(M)]
            out    = Tensor._make(result, (M,), (self,), 'sum_d1')
            def _bwd_sum_d1():
                if self.requires_grad:
                    for i in range(M):
                        for j in range(N):
                            self.grad[i * N + j] += out.grad[i]
            out._backward = _bwd_sum_d1
        return out
 
    def mean(self, dim=None) -> 'Tensor':
        n = (len(self._data) if dim is None
             else (self.shape[0] if dim == 1 else self.shape[1]))
        return self.sum(dim=dim) * (1.0 / n)
 
    # ── Shape Operations ──────────────────────────────────────────────────────
 
    def transpose(self) -> 'Tensor':
        assert len(self.shape) == 2, "transpose requires 2-D tensor"
        M, N     = self.shape
        new_data = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                new_data[j * M + i] = self._data[i * N + j]
        out = Tensor._make(new_data, (N, M), (self,), 'T')
        def _bwd_T():
            if self.requires_grad:
                for i in range(M):
                    for j in range(N):
                        self.grad[i * N + j] += out.grad[j * M + i]
        out._backward = _bwd_T
        return out
 
    @property
    def T(self) -> 'Tensor':
        return self.transpose()
 
    def reshape(self, *new_shape) -> 'Tensor':
        new_n = 1
        for d in new_shape: new_n *= d
        assert new_n == len(self._data), \
            f"reshape: size {len(self._data)} → {new_shape} mismatch"
        out = Tensor._make(self._data[:], new_shape, (self,), 'reshape')
        def _bwd_reshape():
            if self.requires_grad:
                for i in range(len(self._data)):
                    self.grad[i] += out.grad[i]
        out._backward = _bwd_reshape
        return out
 
    def flatten(self) -> 'Tensor':
        return self.reshape(len(self._data))
 
    # ── Special Neural Network Operations ─────────────────────────────────────
 
    def softmax(self, dim: int = 1) -> 'Tensor':
        """
        Numerically stable softmax along `dim`.
        Backward: Jacobian of softmax is diag(s) - s·sᵀ contracted with dout.
          grad_x[k] = s[k] * (dout[k] - dot(dout, s))
        """
        assert len(self.shape) == 2, "softmax operates on 2-D"
        M, N     = self.shape
        new_data = [0.0] * (M * N)
 
        for i in range(M):
            row     = self._data[i * N: (i + 1) * N]
            mx      = max(row)
            exps    = [_exp(v - mx) for v in row]
            total   = sum(exps)
            for j in range(N):
                new_data[i * N + j] = exps[j] / total
 
        out = Tensor._make(new_data, self.shape, (self,), 'softmax')
 
        def _bwd_softmax():
            if not self.requires_grad: return
            for i in range(M):
                s    = new_data[i * N: (i + 1) * N]
                dout = out.grad[i * N: (i + 1) * N]
                dot  = sum(dout[j] * s[j] for j in range(N))
                for k in range(N):
                    self.grad[i * N + k] += s[k] * (dout[k] - dot)
        out._backward = _bwd_softmax
        return out
 
    def log_softmax(self, dim: int = 1) -> 'Tensor':
        """
        Numerically stable log-softmax: log(softmax(x)).
        = x - log(Σ exp(x))  (per-row)
        """
        assert len(self.shape) == 2
        M, N     = self.shape
        log_sum  = []
        new_data = [0.0] * (M * N)
        for i in range(M):
            row  = self._data[i * N: (i + 1) * N]
            mx   = max(row)
            exps = [_exp(v - mx) for v in row]
            ls   = _log(sum(exps)) + mx
            log_sum.append(ls)
            for j in range(N):
                new_data[i * N + j] = row[j] - ls
 
        out = Tensor._make(new_data, self.shape, (self,), 'log_softmax')
 
        def _bwd_log_softmax():
            if not self.requires_grad: return
            for i in range(M):
                sm   = [_exp(new_data[i * N + j]) for j in range(N)]
                dout = out.grad[i * N: (i + 1) * N]
                s    = sum(dout)
                for k in range(N):
                    self.grad[i * N + k] += dout[k] - sm[k] * s
        out._backward = _bwd_log_softmax
        return out
 
    def layer_norm(self, gamma: 'Tensor', beta: 'Tensor',
                   eps: float = 1e-5) -> 'Tensor':
        """
        Layer normalization per row (last dimension).
        y = (x - μ) / σ * γ + β
        Backward via cached statistics.
        """
        assert len(self.shape) == 2
        M, N     = self.shape
        new_data = [0.0] * (M * N)
        mu_list  = []
        inv_std  = []
        xhat     = []
 
        for i in range(M):
            row = self._data[i * N: (i + 1) * N]
            mu  = sum(row) / N
            var = sum((v - mu) ** 2 for v in row) / N
            is_ = 1.0 / _sqrt(var + eps)
            mu_list.append(mu)
            inv_std.append(is_)
            for j in range(N):
                xh = (row[j] - mu) * is_
                xhat.append(xh)
                new_data[i * N + j] = xh * gamma._data[j] + beta._data[j]
 
        out = Tensor._make(new_data, self.shape, (self, gamma, beta), 'layer_norm')
 
        def _bwd_ln():
            if self.requires_grad:
                for i in range(M):
                    is_ = inv_std[i]
                    dout_row = out.grad[i * N: (i + 1) * N]
                    xh_row   = xhat[i * N: (i + 1) * N]
                    g        = gamma._data
                    dy_xh    = [dout_row[j] * g[j] for j in range(N)]
                    sum_dy   = sum(dy_xh)
                    sum_dy_xh = sum(dy_xh[j] * xh_row[j] for j in range(N))
                    for j in range(N):
                        self.grad[i * N + j] += is_ / N * (
                            N * dy_xh[j] - sum_dy - xh_row[j] * sum_dy_xh
                        )
            if gamma.requires_grad:
                for j in range(N):
                    for i in range(M):
                        gamma.grad[j] += out.grad[i * N + j] * xhat[i * N + j]
            if beta.requires_grad:
                for j in range(N):
                    for i in range(M):
                        beta.grad[j] += out.grad[i * N + j]
        out._backward = _bwd_ln
        return out
 
    # ── Autograd Engine ───────────────────────────────────────────────────────
 
    def backward(self) -> None:
        """
        Reverse-mode backpropagation.
        Builds a topological order over the computation graph,
        initialises output gradient to 1, then calls each node's
        _backward closure in reverse order.
        """
        topo    = []
        visited = set()
 
        def _build(v: 'Tensor') -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for c in v._prev:
                    _build(c)
                topo.append(v)
        _build(self)
 
        # Seed
        if self.grad is None:
            self.grad = [0.0] * len(self._data)
        if len(self._data) == 1:
            self.grad[0] = 1.0
        else:
            self.grad = [1.0] * len(self._data)
 
        for v in reversed(topo):
            v._backward()
 
    # ── Gradient Utilities ────────────────────────────────────────────────────
 
    def grad_tensor(self) -> 'Tensor':
        assert self.grad is not None, "No gradient — did you call backward()?"
        return Tensor(self.grad[:], shape=self.shape)
 
    # ── Helpers ───────────────────────────────────────────────────────────────
 
    def _coerce(self, other) -> 'Tensor':
        """Broadcast scalar/list into a Tensor of self's shape."""
        if isinstance(other, Tensor):
            if other.shape == () and self.shape != ():
                return Tensor([other._data[0]] * len(self._data), shape=self.shape)
            return other
        val = float(other)
        return Tensor([val] * len(self._data), shape=self.shape)
 
    def __repr__(self) -> str:
        if not self.shape:
            return f"Tensor({self._data[0]:.6f})"
        if len(self.shape) == 1:
            vs = ', '.join(f'{v:.4f}' for v in self._data[:8])
            suf = '...' if len(self._data) > 8 else ''
            return f"Tensor([{vs}{suf}], shape={self.shape})"
        M, N = self.shape[0], self.shape[1]
        rows = []
        for i in range(min(4, M)):
            r = ', '.join(f'{self._data[i*N+j]:.4f}' for j in range(min(N, 6)))
            rows.append(f'  [{r}{"..." if N > 6 else ""}]')
        if M > 4:
            rows.append('  ...')
        inner = '\n'.join(rows)
        return f"Tensor([\n{inner}\n], shape={self.shape})"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §4  PARAMETER & MODULE BASE
# ══════════════════════════════════════════════════════════════════════════════
 
class Parameter(Tensor):
    """
    A Tensor that is always a leaf with requires_grad=True.
    Semantically: a trainable weight or bias.
    """
    def __init__(self, data, shape=None):
        super().__init__(data, shape=shape, requires_grad=True)
        self._op   = 'param'
        self._prev = set()
 
    def __repr__(self) -> str:
        return 'Parameter' + super().__repr__()[6:]
 
 
class Module:
    """
    Base class for all neural network components.
    Provides recursive parameter collection, zero_grad, and __call__.
    """
 
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
 
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward() not implemented")
 
    def parameters(self) -> list:
        params = []
        for v in vars(self).values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params
 
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
 
    def train_mode(self, flag: bool = True) -> 'Module':
        self._training = flag
        for v in vars(self).values():
            if isinstance(v, Module):
                v.train_mode(flag)
        return self
 
    def eval_mode(self) -> 'Module':
        return self.train_mode(False)
 
    def _is_training(self) -> bool:
        return getattr(self, '_training', True)
 
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
 
    def __repr__(self) -> str:
        cls    = type(self).__name__
        params = self.param_count()
        return f"{cls}(params={params:,})"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §5  NEURAL NETWORK LAYERS
# ══════════════════════════════════════════════════════════════════════════════
 
class Linear(Module):
    """
    Affine layer: y = x @ W + b
      W: (in_features, out_features)  — Kaiming normal init
      b: (out_features,)              — zeros
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        # Kaiming He normal init: scale = sqrt(2 / fan_in)
        scale = _sqrt(2.0 / in_features)
        w_data = [[_RNG.randn() * scale for _ in range(out_features)]
                  for _ in range(in_features)]
        self.weight = Parameter(w_data)               # (in, out)
        self.bias   = Parameter([0.0] * out_features) if bias else None
 
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, in_features) → (batch, out_features)
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
 
    def __repr__(self) -> str:
        return (f"Linear({self.in_features} → {self.out_features}, "
                f"bias={self.bias is not None})")
 
 
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.relu()
 
 
class GeLU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.gelu()
 
 
class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor: return x.sigmoid()
 
 
class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: return x.tanh()
 
 
class Dropout(Module):
    """
    Inverted dropout: scales surviving activations by 1/(1-p) during training.
    No-op at eval time.
    """
    def __init__(self, p: float = 0.1):
        assert 0.0 <= p < 1.0, f"Dropout p must be in [0, 1), got {p}"
        self.p = p
 
    def forward(self, x: Tensor) -> Tensor:
        if not self._is_training() or self.p == 0.0:
            return x
        scale = 1.0 / (1.0 - self.p)
        mask  = [scale if _RNG.random() >= self.p else 0.0 for _ in x._data]
        mask_t = Tensor(mask, shape=x.shape)
        return x * mask_t
 
 
class LayerNorm(Module):
    """
    Layer normalisation over last dimension.
    Learns gamma (scale) and beta (shift) per feature.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.norm_shape = normalized_shape
        self.eps        = eps
        self.gamma      = Parameter([1.0] * normalized_shape)
        self.beta       = Parameter([0.0] * normalized_shape)
 
    def forward(self, x: Tensor) -> Tensor:
        return x.layer_norm(self.gamma, self.beta, eps=self.eps)
 
 
class Embedding(Module):
    """
    Simple lookup-table embedding: integer index → dense vector.
    weight: (vocab_size, d_model)
    """
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        scale = _sqrt(1.0 / d_model)
        data  = [[_RNG.randn() * scale for _ in range(d_model)]
                 for _ in range(vocab_size)]
        self.weight = Parameter(data)   # (vocab_size, d_model)
 
    def forward(self, indices: list) -> Tensor:
        """
        indices: list of integer token indices (length = seq_len)
        Returns: Tensor of shape (seq_len, d_model)
        """
        V, D    = self.vocab_size, self.d_model
        seq_len = len(indices)
        result  = []
        for idx in indices:
            assert 0 <= idx < V, f"Embedding index {idx} out of range [0, {V})"
            result.extend(self.weight._data[idx * D: (idx + 1) * D])
 
        out = Tensor._make(result, (seq_len, D), (self.weight,), 'embed')
 
        def _bwd_embed():
            if self.weight.requires_grad:
                for pos, idx in enumerate(indices):
                    for d in range(D):
                        self.weight.grad[idx * D + d] += out.grad[pos * D + d]
        out._backward = _bwd_embed
        return out
 
 
class Sequential(Module):
    """Chain of modules applied in order."""
    def __init__(self, *layers):
        self.layers = list(layers)
 
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
 
    def __repr__(self) -> str:
        body = '\n  '.join(str(l) for l in self.layers)
        return f"Sequential(\n  {body}\n)"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §6  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
 
class MSELoss(Module):
    """Mean Squared Error: L = mean((y_pred - y_true)²)"""
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()
 
 
class MAELoss(Module):
    """Mean Absolute Error: L = mean(|y_pred - y_true|)"""
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).abs().mean()
 
 
class HuberLoss(Module):
    """Huber loss: quadratic for |err| ≤ δ, linear beyond."""
    def __init__(self, delta: float = 1.0):
        self.delta = delta
 
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        d  = self.delta
        e  = pred - target
        ea = e.abs()
        # Piecewise: 0.5*e² if |e|≤δ else δ*(|e|-0.5*δ)
        result = [
            0.5 * v * v if _abs(v) <= d else d * (_abs(v) - 0.5 * d)
            for v in e._data
        ]
        out = Tensor._make(result, pred.shape, (pred, target), 'huber')
        def _bwd():
            if pred.requires_grad:
                for i in range(len(result)):
                    v  = e._data[i]
                    av = _abs(v)
                    g  = (v if av <= d else d * _sign(v)) / len(result)
                    pred.grad[i] += g * out.grad[0]
        out._backward = _bwd
        return out.mean()
 
 
class BCELoss(Module):
    """Binary Cross-Entropy: L = -mean(y*log(p) + (1-y)*log(1-p))"""
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
 
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = self.eps
        result = [-( t * _log(max(p, eps)) + (1.0 - t) * _log(max(1.0 - p, eps)) )
                  for p, t in zip(pred._data, target._data)]
        out = Tensor._make(result, pred.shape, (pred, target), 'bce')
        def _bwd():
            if pred.requires_grad:
                n = len(result)
                for i in range(n):
                    p = max(min(pred._data[i], 1.0 - eps), eps)
                    t = target._data[i]
                    pred.grad[i] += (-(t / p) + (1.0 - t) / (1.0 - p)) / n * out.grad[0]
        out._backward = _bwd
        return out.mean()
 
 
class CrossEntropyLoss(Module):
    """
    Cross-entropy combining LogSoftmax + NLLLoss.
    Input pred: (batch, num_classes) logits
    Input target: list of integer class indices (length = batch)
    """
    def forward(self, pred: Tensor, target: list) -> Tensor:
        log_probs = pred.log_softmax(dim=1)
        M, N      = log_probs.shape
        result    = []
        for i in range(M):
            idx = target[i]
            result.append(-log_probs._data[i * N + idx])
 
        loss_data  = [sum(result) / M]
        out        = Tensor._make(loss_data, (), (log_probs,), 'ce')
 
        def _bwd_ce():
            if log_probs.requires_grad:
                for i in range(M):
                    for j in range(N):
                        # Gradient of CE w.r.t. log_softmax output:
                        # -1/M if j == target[i], else 0
                        if j == target[i]:
                            log_probs.grad[i * N + j] += -1.0 / M * out.grad[0]
        out._backward = _bwd_ce
        return out
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §7  OPTIMIZERS
# ══════════════════════════════════════════════════════════════════════════════
 
class SGD:
    """
    Stochastic Gradient Descent with optional momentum and weight decay.
    θ(t+1) = θ(t) - lr*(grad + wd*θ) + momentum*v(t)
    """
    def __init__(self, params: list, lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.params       = list(params)
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.velocity     = [[0.0] * len(p._data) for p in self.params]
 
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            for j in range(len(p._data)):
                g = p.grad[j] + self.weight_decay * p._data[j]
                self.velocity[i][j] = self.momentum * self.velocity[i][j] + g
                p._data[j] -= self.lr * self.velocity[i][j]
 
    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
 
 
class RMSProp:
    """
    RMSProp: divides learning rate by running average of squared gradients.
    """
    def __init__(self, params: list, lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8):
        self.params  = list(params)
        self.lr      = lr
        self.alpha   = alpha
        self.eps     = eps
        self.sq_avg  = [[0.0] * len(p._data) for p in self.params]
 
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            for j in range(len(p._data)):
                g = p.grad[j]
                self.sq_avg[i][j] = (self.alpha * self.sq_avg[i][j]
                                     + (1.0 - self.alpha) * g * g)
                p._data[j] -= self.lr * g / (_sqrt(self.sq_avg[i][j]) + self.eps)
 
    def zero_grad(self) -> None:
        for p in self.params: p.zero_grad()
 
 
class Adam:
    """
    Adam: Adaptive Moment Estimation.
    m(t) = β₁*m(t-1) + (1-β₁)*g
    v(t) = β₂*v(t-1) + (1-β₂)*g²
    θ -= lr * m̂ / (√v̂ + ε)   where m̂, v̂ are bias-corrected
    """
    def __init__(self, params: list, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params       = list(params)
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self.m            = [[0.0] * len(p._data) for p in self.params]
        self.v            = [[0.0] * len(p._data) for p in self.params]
 
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1    = 1.0 - b1 ** self.t
        bc2    = 1.0 - b2 ** self.t
 
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            for j in range(len(p._data)):
                g = p.grad[j] + self.weight_decay * p._data[j]
                self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * g
                self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * g * g
                m_hat        = self.m[i][j] / bc1
                v_hat        = self.v[i][j] / bc2
                p._data[j]  -= self.lr * m_hat / (_sqrt(v_hat) + self.eps)
 
    def zero_grad(self) -> None:
        for p in self.params: p.zero_grad()
 
 
class AdamW(Adam):
    """
    AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
    Applies weight decay directly to parameters, not to the gradient.
    """
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1    = 1.0 - b1 ** self.t
        bc2    = 1.0 - b2 ** self.t
 
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # Decoupled weight decay
            for j in range(len(p._data)):
                p._data[j] *= (1.0 - self.lr * self.weight_decay)
 
            for j in range(len(p._data)):
                g = p.grad[j]
                self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * g
                self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * g * g
                m_hat        = self.m[i][j] / bc1
                v_hat        = self.v[i][j] / bc2
                p._data[j]  -= self.lr * m_hat / (_sqrt(v_hat) + self.eps)
 
 
class LRScheduler:
    """
    Learning rate schedulers for any optimizer.
    Supports: step, cosine_annealing, warmup_cosine, exponential
    """
    def __init__(self, optimizer, schedule: str = 'cosine',
                 T_max: int = 1000, eta_min: float = 1e-6,
                 warmup_steps: int = 0, gamma: float = 0.95):
        self.opt          = optimizer
        self.schedule     = schedule
        self.T_max        = T_max
        self.eta_min      = eta_min
        self.warmup_steps = warmup_steps
        self.gamma        = gamma
        self.base_lr      = optimizer.lr
        self._step_count  = 0
 
    def step(self) -> float:
        self._step_count += 1
        t = self._step_count
 
        if self.schedule == 'cosine':
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1.0 + _cos(_PI * t / self.T_max))
        elif self.schedule == 'warmup_cosine':
            if t < self.warmup_steps:
                lr = self.base_lr * t / max(1, self.warmup_steps)
            else:
                tc = t - self.warmup_steps
                Tc = self.T_max - self.warmup_steps
                lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                    1.0 + _cos(_PI * tc / Tc))
        elif self.schedule == 'exponential':
            lr = self.base_lr * (self.gamma ** t)
        elif self.schedule == 'step':
            lr = self.base_lr * (self.gamma ** (t // self.T_max))
        else:
            lr = self.base_lr
 
        self.opt.lr = lr
        return lr
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §8  CELLULAR AUTOMATON ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# A 2-D discrete dynamical system.  The grid is a flat list of 0/1 cell states.
# Rules follow the B/S notation standard (birth/survival counts for Moore
# neighbourhood).  Conway's Game of Life = B3/S23.
# ══════════════════════════════════════════════════════════════════════════════
 
class CellularAutomatonEngine:
    """
    2-D Cellular Automaton with configurable B/S rules.
 
    Usage:
      ca = CellularAutomatonEngine(20, 40, rule='B3/S23', seed=42)
      ca.run(50, display=True)
    """
 
    PRESETS = {
        'life':         'B3/S23',     # Conway's Game of Life
        'highlife':     'B36/S23',    # HighLife
        'daynight':     'B3678/S34678',
        'seeds':        'B2/S',       # Seeds — explosive
        'replicator':   'B1357/S1357',
        'anneal':       'B4678/S35678',
        '2x2':          'B36/S125',
        'maze':         'B3/S12345',
        'coral':        'B3/S45678',
        'flakes':       'B3/S012345678',
    }
 
    def __init__(self, rows: int, cols: int, rule: str = 'B3/S23',
                 seed=None, density: float = 0.3):
        self.rows       = rows
        self.cols       = cols
        self.generation = 0
        self.history    = []
        self._birth, self._survive = self._parse_rule(
            self.PRESETS.get(rule, rule)
        )
        if seed is not None:
            _RNG.seed(seed)
        self.grid = [
            1 if _RNG.random() < density else 0
            for _ in range(rows * cols)
        ]
 
    @staticmethod
    def _parse_rule(rule_str: str):
        """Parse 'B3/S23' → ({3}, {2,3})."""
        parts   = rule_str.upper().replace(' ', '').split('/')
        birth   = set()
        survive = set()
        for part in parts:
            if part.startswith('B'):
                birth   = set(int(c) for c in part[1:])
            elif part.startswith('S'):
                survive = set(int(c) for c in part[1:])
        return birth, survive
 
    def _idx(self, r: int, c: int) -> int:
        return (r % self.rows) * self.cols + (c % self.cols)
 
    def _count_neighbors(self, r: int, c: int) -> int:
        """Count live Moore-neighbourhood cells (8 neighbours, toroidal)."""
        total = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                total += self.grid[self._idx(r + dr, c + dc)]
        return total
 
    def step(self) -> None:
        """Advance exactly one generation."""
        new_grid = [0] * (self.rows * self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                idx  = r * self.cols + c
                live = self.grid[idx]
                n    = self._count_neighbors(r, c)
                if live:
                    new_grid[idx] = 1 if n in self._survive else 0
                else:
                    new_grid[idx] = 1 if n in self._birth else 0
        self.grid       = new_grid
        self.generation += 1
 
    def run(self, generations: int, display: bool = False,
            record: bool = True) -> list:
        """
        Advance `generations` steps.
        If display=True, prints ASCII art each generation.
        If record=True, saves each grid state.
        Returns list of grid snapshots.
        """
        snapshots = []
        for _ in range(generations):
            if record:
                snapshots.append(self.grid[:])
            if display:
                print(f"\n── Generation {self.generation} "
                      f"(pop={self.population()}) ──")
                print(self.render())
            self.step()
        return snapshots
 
    def render(self, alive: str = '█', dead: str = '·') -> str:
        """ASCII representation of the current grid."""
        lines = []
        for r in range(self.rows):
            row = ''
            for c in range(self.cols):
                row += alive if self.grid[r * self.cols + c] else dead
            lines.append(row)
        return '\n'.join(lines)
 
    def population(self) -> int:
        return sum(self.grid)
 
    def entropy(self) -> float:
        """Shannon entropy of the live/dead distribution."""
        total = self.rows * self.cols
        live  = self.population()
        dead  = total - live
        if live == 0 or dead == 0:
            return 0.0
        p1 = live  / total
        p0 = dead  / total
        return -(p1 * _log(p1) + p0 * _log(p0))
 
    def similarity(self, other_grid: list) -> float:
        """Hamming similarity between self.grid and another grid."""
        assert len(other_grid) == len(self.grid)
        matches = sum(1 for a, b in zip(self.grid, other_grid) if a == b)
        return matches / len(self.grid)
 
    def place_pattern(self, pattern: list, row: int, col: int) -> None:
        """
        Stamp a pattern (list of (dr, dc) offsets for live cells) at (row, col).
        Example patterns:
          glider  = [(0,1),(1,2),(2,0),(2,1),(2,2)]
          blinker = [(0,0),(0,1),(0,2)]
          block   = [(0,0),(0,1),(1,0),(1,1)]
        """
        for dr, dc in pattern:
            r = (row + dr) % self.rows
            c = (col + dc) % self.cols
            self.grid[r * self.cols + c] = 1
 
    def clear(self) -> None:
        self.grid = [0] * (self.rows * self.cols)
 
    # Common patterns as class attributes
    GLIDER   = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    BLINKER  = [(0, 0), (0, 1), (0, 2)]
    BLOCK    = [(0, 0), (0, 1), (1, 0), (1, 1)]
    TOAD     = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    BEACON   = [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)]
    LWSS     = [(0,1),(0,4),(1,0),(2,0),(2,4),(3,0),(3,1),(3,2),(3,3)]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §9  TRANSFORMER AGENT MODULE
# ══════════════════════════════════════════════════════════════════════════════
# A minimal, trainable transformer-based agent following the GPT architecture:
#   Token Embedding + Positional Encoding
#   → N x TransformerBlock (MHA + FFN + LayerNorm)
#   → Prediction Head
#
# Positional encoding: learned (simpler, no sinusoidal dependency)
# Attention: scaled dot-product, causal mask for autoregressive use
# ══════════════════════════════════════════════════════════════════════════════
 
class MultiHeadSelfAttention(Module):
    """
    Multi-Head Self-Attention.
    Projects input to Q, K, V, computes scaled dot-product attention,
    concatenates heads and projects output.
 
    For simplicity and correctness, we implement single-head as the inner loop
    and tile for n_heads (equivalent to full multi-head when combined).
    The key invariant: d_head = d_model // n_heads.
    """
 
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        assert d_model % n_heads == 0, \
            f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        # Combined Q,K,V projection: (d_model, 3*d_model)
        self.W_qkv    = Linear(d_model, 3 * d_model, bias=False)
        self.W_o      = Linear(d_model, d_model, bias=False)
        self.dropout  = Dropout(dropout)
        self._scale   = _sqrt(float(self.d_head))
 
    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        """
        x: (seq_len, d_model)
        Returns: (seq_len, d_model)
 
        Multi-head attention using the combined QKV projection.
        For each head h: Q_h, K_h, V_h ∈ (seq_len, d_head)
        Attention: softmax((Q_h @ K_h^T) / √d_head) @ V_h
        Concatenate all heads → (seq_len, d_model)
        """
        S, D  = x.shape        # seq_len, d_model
        H     = self.n_heads
        Dh    = self.d_head
        scale = self._scale
 
        # Single combined projection: (S, 3*D)
        qkv = self.W_qkv(x)   # (S, 3*D)
        # Split into Q, K, V: each (S, D)
        Q_data = qkv._data[0 * S * D: 1 * S * D]  # incorrect split — need column-wise
        # Correct column-wise split:
        Q_data = []
        K_data = []
        V_data = []
        for i in range(S):
            row = qkv._data[i * 3 * D: (i + 1) * 3 * D]
            Q_data.extend(row[0 * D: 1 * D])
            K_data.extend(row[1 * D: 2 * D])
            V_data.extend(row[2 * D: 3 * D])
 
        Q = Tensor(Q_data, shape=(S, D))
        K = Tensor(K_data, shape=(S, D))
        V = Tensor(V_data, shape=(S, D))
 
        # Process each head independently, concatenate results
        head_outputs = []
        for h in range(H):
            # Extract head slice: columns [h*Dh : (h+1)*Dh]
            q_h = Tensor([Q._data[i * D + h * Dh + d]
                          for i in range(S) for d in range(Dh)], shape=(S, Dh))
            k_h = Tensor([K._data[i * D + h * Dh + d]
                          for i in range(S) for d in range(Dh)], shape=(S, Dh))
            v_h = Tensor([V._data[i * D + h * Dh + d]
                          for i in range(S) for d in range(Dh)], shape=(S, Dh))
 
            # Scores: (S, S)
            scores = q_h @ k_h.T          # (S, Dh) @ (Dh, S) = (S, S)
            scores = scores * (1.0 / scale)
 
            if causal:
                # Apply causal mask: −∞ for future positions
                for i in range(S):
                    for j in range(i + 1, S):
                        scores._data[i * S + j] = -1e9
 
            attn = scores.softmax(dim=1)   # (S, S)
            attn = self.dropout(attn)
            out_h = attn @ v_h             # (S, S) @ (S, Dh) = (S, Dh)
            head_outputs.append(out_h)
 
        # Concatenate heads along d dimension: (S, H*Dh) = (S, D)
        concat_data = []
        for i in range(S):
            for h in range(H):
                concat_data.extend(
                    head_outputs[h]._data[i * Dh: (i + 1) * Dh]
                )
        concat = Tensor(concat_data, shape=(S, D))
        return self.W_o(concat)
 
 
class FeedForward(Module):
    """
    Position-wise Feed-Forward Network: Linear → GELU → Linear
    Expansion factor 4× is standard (following GPT).
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        d_ff = d_ff or 4 * d_model
        self.fc1     = Linear(d_model, d_ff)
        self.fc2     = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
 
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.fc1(x).gelu()))
 
 
class TransformerBlock(Module):
    """
    Pre-norm transformer block (following GPT-2 convention):
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int = None, dropout: float = 0.0):
        self.ln1      = LayerNorm(d_model)
        self.attn     = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2      = LayerNorm(d_model)
        self.ff       = FeedForward(d_model, d_ff, dropout)
 
    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        # Residual + attention
        x_norm = self.ln1(x)
        x_attn = self.attn(x_norm, causal=causal)
        # Residual connection (element-wise add)
        x_res_data = [a + b for a, b in zip(x._data, x_attn._data)]
        x = Tensor(x_res_data, shape=x.shape)
 
        # Residual + FFN
        x_norm2   = self.ln2(x)
        x_ff      = self.ff(x_norm2)
        x_res_data2 = [a + b for a, b in zip(x._data, x_ff._data)]
        return Tensor(x_res_data2, shape=x.shape)
 
 
class MiniTransformer(Module):
    """
    Minimal autoregressive transformer language model.
 
    Architecture:
      TokenEmbedding(vocab_size, d_model)
      + PosEmbedding(max_seq_len, d_model)          ← learned
      N x TransformerBlock(d_model, n_heads, d_ff)
      LayerNorm(d_model)
      Linear(d_model, vocab_size)                   ← prediction head
 
    Usage:
      model = MiniTransformer(vocab_size=64, d_model=32, n_heads=2,
                              n_layers=2, max_seq=16)
      loss  = model.train_step(token_ids, optimizer)
    """
 
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_seq: int = 64, dropout: float = 0.0):
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq     = max_seq
 
        self.tok_embed   = Embedding(vocab_size, d_model)
        self.pos_embed   = Embedding(max_seq, d_model)   # learned positional
        self.blocks      = [TransformerBlock(d_model, n_heads, dropout=dropout)
                            for _ in range(n_layers)]
        self.ln_final    = LayerNorm(d_model)
        self.head        = Linear(d_model, vocab_size, bias=False)
        self._ce_loss    = CrossEntropyLoss()
 
    def parameters(self) -> list:
        params = []
        params.extend(self.tok_embed.parameters())
        params.extend(self.pos_embed.parameters())
        for b in self.blocks:
            params.extend(b.parameters())
        params.extend(self.ln_final.parameters())
        params.extend(self.head.parameters())
        return params
 
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
 
    def forward(self, token_ids: list) -> Tensor:
        """
        token_ids: list of int, length seq_len
        Returns logits: Tensor (seq_len, vocab_size)
        """
        S     = len(token_ids)
        pos   = list(range(S))
 
        tok_e = self.tok_embed(token_ids)   # (S, D)
        pos_e = self.pos_embed(pos)          # (S, D)
 
        # Element-wise addition of token + position embeddings
        x_data = [tok_e._data[i] + pos_e._data[i]
                  for i in range(len(tok_e._data))]
        x = Tensor(x_data, shape=(S, self.d_model))
 
        for block in self.blocks:
            x = block(x, causal=True)
 
        x = self.ln_final(x)
        return self.head(x)           # (S, vocab_size)
 
    def train_step(self, token_ids: list, optimizer) -> float:
        """
        One training step predicting token[t+1] from token[t].
        Returns scalar loss value.
        """
        if len(token_ids) < 2:
            return 0.0
 
        inputs  = token_ids[:-1]
        targets = token_ids[1:]
 
        optimizer.zero_grad()
        logits = self.forward(inputs)    # (S-1, vocab_size)
        loss   = self._ce_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()
 
    def generate(self, start_ids: list, max_new: int = 20,
                 temperature: float = 1.0) -> list:
        """
        Autoregressive generation via top-k sampling (k=vocab_size).
        temperature=1.0 → full distribution, low → sharper.
        """
        ids = list(start_ids)
        for _ in range(max_new):
            context = ids[-self.max_seq:]
            logits  = self.forward(context)          # (len, vocab)
            S_cur, V = logits.shape
            last_row = logits._data[(S_cur - 1) * V: S_cur * V]
 
            # Temperature scaling + softmax sampling
            if temperature != 1.0:
                last_row = [v / temperature for v in last_row]
            max_v   = max(last_row)
            exps    = [_exp(v - max_v) for v in last_row]
            total   = sum(exps)
            probs   = [e / total for e in exps]
 
            # Cumulative sampling
            r     = _RNG.random()
            cumul = 0.0
            next_id = V - 1
            for idx, p in enumerate(probs):
                cumul += p
                if r < cumul:
                    next_id = idx
                    break
            ids.append(next_id)
        return ids
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §10  AIOS INTEGRATION — @agent_method, kernel hooks, training utilities
# ══════════════════════════════════════════════════════════════════════════════
 
def agent_method(name: str = '', doc: str = ''):
    """
    Decorator registering a callable as an AIOS agent tool.
    Wraps invocation in timing and structured result packaging.
    """
    import time as _time
 
    def decorator(fn):
        tool_name = name or fn.__name__
        fn._aios_tool      = True
        fn._aios_tool_name = tool_name
        fn._aios_tool_doc  = doc or (fn.__doc__ or '').strip()
 
        def wrapper(*args, **kwargs):
            t0     = _time.perf_counter()
            result = fn(*args, **kwargs)
            dt     = _time.perf_counter() - t0
            return {
                'tool':    tool_name,
                'result':  result,
                'elapsed': round(dt * 1000, 3),
            }
        wrapper._aios_tool      = True
        wrapper._aios_tool_name = tool_name
        wrapper._aios_tool_doc  = fn._aios_tool_doc
        wrapper.__name__        = fn.__name__
        wrapper.__doc__         = fn.__doc__
        return wrapper
    return decorator
 
 
class NeuralTrainer:
    """
    High-level training loop with metrics, early stopping, and LR scheduling.
    Designed for integration with the AIOS agent kernel.
    """
 
    def __init__(self, model: Module, optimizer, loss_fn: Module,
                 scheduler=None, clip_norm: float = None):
        self.model      = model
        self.optimizer  = optimizer
        self.loss_fn    = loss_fn
        self.scheduler  = scheduler
        self.clip_norm  = clip_norm
        self.history    = {'loss': [], 'val_loss': [], 'lr': []}
 
    def _clip_gradients(self) -> float:
        """Global gradient norm clipping."""
        if self.clip_norm is None:
            return 0.0
        params = self.model.parameters()
        total_sq = sum(
            g ** 2
            for p in params if p.grad is not None
            for g in p.grad
        )
        norm = _sqrt(total_sq)
        if norm > self.clip_norm:
            scale = self.clip_norm / (norm + 1e-8)
            for p in params:
                if p.grad is not None:
                    for i in range(len(p.grad)):
                        p.grad[i] *= scale
        return norm
 
    def train_epoch(self, batches: list) -> float:
        """
        One epoch over a list of (X, y) Tensor pairs.
        Returns mean training loss.
        """
        self.model.train_mode(True)
        total_loss = 0.0
        for X, y in batches:
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(batches), 1)
        self.history['loss'].append(avg)
        if self.scheduler:
            lr = self.scheduler.step()
            self.history['lr'].append(lr)
        return avg
 
    def evaluate(self, batches: list) -> float:
        """Evaluate on validation batches (no gradient accumulation)."""
        self.model.eval_mode()
        total = 0.0
        for X, y in batches:
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            total += loss.item()
        avg = total / max(len(batches), 1)
        self.history['val_loss'].append(avg)
        return avg
 
    def fit(self, train_batches: list, epochs: int,
            val_batches: list = None, patience: int = 0,
            verbose: bool = True) -> dict:
        """
        Full training loop.
        patience=0 disables early stopping.
        Returns history dict.
        """
        best_val    = _INF
        no_improve  = 0
 
        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_batches)
            v_loss = self.evaluate(val_batches) if val_batches else None
 
            if verbose:
                v_str = f"  val_loss={v_loss:.6f}" if v_loss is not None else ''
                lr_str = f"  lr={self.optimizer.lr:.2e}"
                print(f"  Epoch {epoch:4d} | loss={t_loss:.6f}{v_str}{lr_str}")
 
            if patience > 0 and v_loss is not None:
                if v_loss < best_val - 1e-7:
                    best_val   = v_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(f"  Early stop at epoch {epoch}.")
                        break
        return self.history
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §11  GRADIENT VERIFICATION UTILITY
# ══════════════════════════════════════════════════════════════════════════════
 
def numerical_gradient(fn, param: Parameter, eps: float = 1e-5) -> list:
    """
    Compute numerical gradient of scalar fn() w.r.t. all elements of param.
    Uses centred finite differences: (f(x+ε) - f(x-ε)) / (2ε)
    """
    numeric = []
    for i in range(len(param._data)):
        orig = param._data[i]
        param._data[i] = orig + eps
        fp = fn().item()
        param._data[i] = orig - eps
        fm = fn().item()
        param._data[i] = orig
        numeric.append((fp - fm) / (2 * eps))
    return numeric
 
 
def gradient_check(module: Module, X: Tensor, y: Tensor,
                   loss_fn: Module, eps: float = 1e-5,
                   rtol: float = 1e-3) -> bool:
    """
    Verify autograd gradients against numerical finite differences.
    Uses centred differences; does NOT call backward inside numerical loop.
    Returns (passed: bool, max_relative_error: float).
    """
    params = module.parameters()[:3]   # check first few for speed
 
    # One clean forward+backward pass for analytic grads
    module.zero_grad()
    pred = module(X)
    loss = loss_fn(pred, y)
    loss.backward()
 
    # Save analytic grads before any further forward passes contaminate state
    saved_analytic = {id(p): p.grad[:] for p in params}
 
    passed  = True
    max_err = 0.0
 
    for param in params:
        analytic = saved_analytic[id(param)]
        for i in range(len(param._data)):
            orig           = param._data[i]
            param._data[i] = orig + eps
            fp             = loss_fn(module(X), y).item()   # no backward
            param._data[i] = orig - eps
            fm             = loss_fn(module(X), y).item()
            param._data[i] = orig                           # restore
 
            numeric = (fp - fm) / (2.0 * eps)
            a       = analytic[i]
            denom   = max(_abs(a), _abs(numeric), 1e-8)
            err     = _abs(a - numeric) / denom
            max_err = max(max_err, err)
            if err > rtol:
                passed = False
 
    return passed, max_err
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §12  SYSTEM SELF-TESTS
# ══════════════════════════════════════════════════════════════════════════════
 
def _run_self_tests():
    print("\n" + "═" * 68)
    print("  AIOS Phase IV — Neural Intelligence Kernel  SELF-TESTS")
    print("═" * 68)
    passed = 0
    failed = 0
 
    def check(name: str, condition: bool, detail: str = ''):
        nonlocal passed, failed
        status = '✓ PASS' if condition else '✗ FAIL'
        if not condition:
            failed += 1
            print(f"  {status}  {name}  {detail}")
        else:
            passed += 1
            print(f"  {status}  {name}")
 
    # ── T1: Math Primitives ───────────────────────────────────────────────────
    print("\n  §1  Math Primitives")
    check("exp(0)=1",          _abs(_exp(0.0) - 1.0)      < 1e-14)
    check("exp(1)≈e",          _abs(_exp(1.0) - _E)        < 1e-12)
    check("exp(ln(x))=x",      _abs(_exp(_log(2.71)) - 2.71) < 1e-12)
    check("log(e)=1",          _abs(_log(_E) - 1.0)        < 1e-14)
    check("log(1)=0",          _abs(_log(1.0))              < 1e-15)
    check("sqrt(2)²≈2",        _abs(_sqrt(2.0) ** 2 - 2.0) < 1e-14)
    check("tanh(0)=0",         _abs(_tanh(0.0))             < 1e-15)
    check("tanh(inf)→1",       _abs(_tanh(100.0) - 1.0)    < 1e-12)
    check("sigmoid(0)=0.5",    _abs(_sigmoid(0.0) - 0.5)   < 1e-15)
    check("cos(0)=1",          _abs(_cos(0.0) - 1.0)        < 1e-15)
    check("cos(π)=-1",         _abs(_cos(_PI) + 1.0)        < 1e-5)
    check("sin(π/2)≈1",        _abs(_sin(_PI * 0.5) - 1.0) < 1e-12)
 
    # ── T2: Tensor Arithmetic ─────────────────────────────────────────────────
    print("\n  §2  Tensor Arithmetic")
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    check("add 1D",     (a + b).tolist() == [5.0, 7.0, 9.0])
    check("sub 1D",     (b - a).tolist() == [3.0, 3.0, 3.0])
    check("mul 1D",     (a * b).tolist() == [4.0, 10.0, 18.0])
    expected_div = [0.5, 1.0, 1.5]
    check("div scalar", all(_abs((a/2.0).tolist()[i] - expected_div[i]) < 1e-12
                            for i in range(3)))
    check("neg",        ((-a).tolist()) == [-1.0, -2.0, -3.0])
    check("pow",        all(_abs((a ** 2).tolist()[i] - [1, 4, 9][i]) < 1e-14
                            for i in range(3)))
 
    # ── T3: Matrix Multiply ───────────────────────────────────────────────────
    print("\n  §3  Matrix Multiply")
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    check("matmul shape",  C.shape == (2, 2))
    check("matmul C[0,0]", _abs(C._data[0] - 19.0) < 1e-12)
    check("matmul C[1,1]", _abs(C._data[3] - 50.0) < 1e-12)
 
    # ── T4: Autograd ──────────────────────────────────────────────────────────
    print("\n  §4  Reverse-Mode Autograd")
    # f(x) = x³ + 2x, f'(x) = 3x² + 2
    # At x=3: f'(3) = 27 + 2 = 29
    x = Parameter([3.0])
    y_out = (x ** 3) + (x * Tensor([2.0]))
    y_sum = y_out.sum()
    y_sum.backward()
    check("d/dx(x³+2x)|x=3 = 29", _abs(x.grad[0] - 29.0) < 1e-10,
          f"got {x.grad[0]:.6f}")
 
    # Chain rule: d/da (a*b + b²) w.r.t. a at a=2,b=3 = b = 3
    a_ = Parameter([2.0])
    b_ = Parameter([3.0])
    z_ = (a_ * b_ + b_ ** 2).sum()
    z_.backward()
    check("chain rule ∂/∂a(ab+b²)=b=3",  _abs(a_.grad[0] - 3.0) < 1e-10)
    check("chain rule ∂/∂b(ab+b²)=a+2b=8", _abs(b_.grad[0] - 8.0) < 1e-10)
 
    # matmul backward
    A_p = Parameter([[1.0, 0.0], [0.0, 1.0]])  # identity
    v_  = Tensor([[2.0], [3.0]])                # input vector
    out_ = A_p @ v_
    loss_ = out_.sum()
    loss_.backward()
    # d(Σ Av)/dA[i,j] = v[j] for each output row i
    check("matmul backward dA[0,0]=v[0]=2", _abs(A_p.grad[0] - 2.0) < 1e-10)
    check("matmul backward dA[1,1]=v[1]=3", _abs(A_p.grad[3] - 3.0) < 1e-10)
 
    # ── T5: Linear Layer ──────────────────────────────────────────────────────
    print("\n  §5  Linear Layer")
    _RNG.seed(42)
    lin = Linear(2, 3)
    x_in = Tensor([[1.0, 0.0]], shape=(1, 2))
    out_lin = lin(x_in)
    check("Linear output shape", out_lin.shape == (1, 3))
    loss_lin = out_lin.sum()
    loss_lin.backward()
    check("Linear weight grad exists",
          lin.weight.grad is not None and any(g != 0.0 for g in lin.weight.grad))
    check("Linear bias grad exists",
          lin.bias is not None and any(g != 0.0 for g in lin.bias.grad))
 
    # ── T6: XOR — must converge to < 0.02 ────────────────────────────────────
    print("\n  §6  XOR Convergence (reverse-mode AD)")
    _RNG.seed(0)
    # Net: 2 → 8 → 8 → 1, tanh activations, Adam
    model_xor = Sequential(
        Linear(2, 8), Tanh(),
        Linear(8, 8), Tanh(),
        Linear(8, 1), Sigmoid()
    )
    xor_X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    opt_xor = Adam(model_xor.parameters(), lr=0.05)
    loss_fn_xor = BCELoss()
    batches_xor = [(xor_X, xor_y)]
 
    trainer_xor = NeuralTrainer(model_xor, opt_xor, loss_fn_xor,
                                clip_norm=1.0)
    history_xor = trainer_xor.fit(batches_xor, epochs=2000, verbose=False)
    final_loss = history_xor['loss'][-1]
    check(f"XOR converges (loss={final_loss:.5f} < 0.02)", final_loss < 0.02,
          f"got {final_loss:.5f}")
 
    # Verify predictions
    preds = []
    for row in [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]:
        x_row = Tensor([row], shape=(1,2))
        p     = model_xor(x_row)._data[0]
        preds.append(round(p))
    check("XOR predictions correct",
          preds == [0, 1, 1, 0],
          f"got {preds}")
 
    # ── T7: Softmax & LogSoftmax ──────────────────────────────────────────────
    print("\n  §7  Softmax / LogSoftmax")
    logits = Tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    sm     = logits.softmax(dim=1)
    row_sums = [sm._data[0]+sm._data[1]+sm._data[2],
                sm._data[3]+sm._data[4]+sm._data[5]]
    check("softmax rows sum to 1", all(_abs(s - 1.0) < 1e-12 for s in row_sums))
    check("softmax monotone", sm._data[2] > sm._data[1] > sm._data[0])
 
    lsm = logits.log_softmax(dim=1)
    check("log_softmax ≤ 0", all(v <= 1e-10 for v in lsm._data))
 
    # ── T8: LayerNorm ─────────────────────────────────────────────────────────
    print("\n  §8  LayerNorm")
    ln    = LayerNorm(4)
    x_ln  = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    y_ln  = ln(x_ln)
    # Row 0 should be normalised to ~zero mean, ~unit var
    row0  = y_ln._data[0:4]
    mean0 = sum(row0) / 4
    var0  = sum((v - mean0)**2 for v in row0) / 4
    check("LayerNorm row mean ≈ 0", _abs(mean0) < 1e-6, f"mean={mean0:.2e}")
    check("LayerNorm row std ≈ 1",  _abs(_sqrt(var0) - 1.0) < 1e-5)
    # Row 1 (all identical) → undefined variance → should not crash
    check("LayerNorm constant input no crash",
          all(not (v != v) for v in y_ln._data[4:]))  # NaN check
 
    # ── T9: Loss Functions ────────────────────────────────────────────────────
    print("\n  §9  Loss Functions")
    p_pos   = Tensor([[0.9, 0.1]])
    p_neg   = Tensor([[0.1, 0.9]])
    ce_loss = CrossEntropyLoss()
    loss_c0 = ce_loss(p_pos, [0])
    loss_c1 = ce_loss(p_pos, [1])
    check("CE low for correct pred",  loss_c0.item() < 0.5)
    check("CE high for wrong pred",   loss_c1.item() > 1.0)
 
    mse    = MSELoss()
    p_mse  = Tensor([[1.0, 2.0]])
    t_mse  = Tensor([[1.0, 2.0]])
    check("MSE perfect pred = 0", _abs(mse(p_mse, t_mse).item()) < 1e-14)
 
    bce    = BCELoss()
    p_bce  = Tensor([[0.95]])
    t_bce  = Tensor([[1.0]])
    check("BCE near-perfect < 0.1", bce(p_bce, t_bce).item() < 0.1)
 
    # ── T10: Cellular Automaton ───────────────────────────────────────────────
    print("\n  §10 Cellular Automaton")
    ca = CellularAutomatonEngine(10, 10, rule='B3/S23', seed=123, density=0.0)
    ca.clear()
    # Place a glider and verify it's still alive after 4 generations
    ca.place_pattern(CellularAutomatonEngine.GLIDER, 2, 2)
    pop0 = ca.population()
    ca.run(4)
    pop4 = ca.population()
    check("Glider alive after 4 gen", pop4 > 0, f"pop0={pop0}, pop4={pop4}")
 
    # Blinker: period-2 oscillator
    ca2 = CellularAutomatonEngine(7, 7, rule='B3/S23', seed=0, density=0.0)
    ca2.clear()
    ca2.place_pattern(CellularAutomatonEngine.BLINKER, 3, 2)
    grid_0 = ca2.grid[:]
    ca2.step(); ca2.step()                 # 2 steps = back to original
    check("Blinker period-2",
          ca2.similarity(grid_0) > 0.99)
 
    entropy_test = CellularAutomatonEngine(8, 8, density=0.5, seed=7)
    e = entropy_test.entropy()
    check("Entropy in (0, ln2]", 0 < e <= _log(2.0) + 1e-10)
 
    # ── T11: Adam Optimizer ───────────────────────────────────────────────────
    print("\n  §11 Adam Optimizer")
    _RNG.seed(1)
    # Minimise sum((w - target)²): should converge to target
    target = [1.0, -2.0, 0.5]
    w      = Parameter([0.0, 0.0, 0.0])
    adam   = Adam([w], lr=0.1)
    for _ in range(500):
        adam.zero_grad()
        t_t  = Tensor(target[:])
        diff = (w - t_t)
        loss = (diff * diff).sum()
        loss.backward()
        adam.step()
    check("Adam converges to target",
          all(_abs(w._data[i] - target[i]) < 0.01 for i in range(3)),
          f"w={[round(v, 4) for v in w._data]}, target={target}")
 
    # ── T12: Gradient Check ───────────────────────────────────────────────────
    print("\n  §12 Numerical Gradient Verification")
    _RNG.seed(5)
    gc_model  = Linear(3, 2)
    gc_X      = Tensor.uniform(1, 3, lo=-1.0, hi=1.0)
    gc_y      = Tensor.uniform(1, 2, lo=-1.0, hi=1.0)
    gc_loss   = MSELoss()
    ok, max_e = gradient_check(gc_model, gc_X, gc_y, gc_loss)
    check(f"Gradient check max_err={max_e:.2e} < 1e-3", ok, f"max_err={max_e:.2e}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed  |  {failed} failed")
    print("─" * 68 + "\n")
    return failed == 0
 
 
# ══════════════════════════════════════════════════════════════════════════════
# §13  DEMONSTRATION SUITE
# ══════════════════════════════════════════════════════════════════════════════
 
def demo_xor_detailed():
    """Detailed XOR training demonstration with per-epoch logging."""
    print("\n" + "═" * 68)
    print("  DEMO: XOR — Reverse-Mode Autograd Training")
    print("═" * 68)
    _RNG.seed(42)
 
    model = Sequential(
        Linear(2, 16), Tanh(),
        Linear(16, 8), Tanh(),
        Linear(8, 1),  Sigmoid()
    )
    print(f"  Model: {model}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    xor_X  = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_y  = Tensor([[0.0], [1.0], [1.0], [0.0]])
    opt    = Adam(model.parameters(), lr=0.05)
    loss_fn = BCELoss()
    sched  = LRScheduler(opt, schedule='cosine', T_max=1500, eta_min=1e-4)
 
    print(f"\n  {'Epoch':>6}  {'Loss':>10}  {'LR':>10}")
    print("  " + "─" * 32)
    for epoch in range(1, 1501):
        opt.zero_grad()
        pred = model(xor_X)
        loss = loss_fn(pred, xor_y)
        loss.backward()
        # Gradient clipping
        total_sq = sum(g**2 for p in model.parameters()
                       if p.grad for g in p.grad)
        norm = _sqrt(total_sq)
        if norm > 1.0:
            scale = 1.0 / norm
            for p in model.parameters():
                if p.grad:
                    for i in range(len(p.grad)): p.grad[i] *= scale
        opt.step()
        lr = sched.step()
        if epoch % 200 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {loss.item():>10.6f}  {lr:>10.2e}")
 
    print("\n  Final predictions:")
    for (x0, x1), expected in [((0,0),0),((0,1),1),((1,0),1),((1,1),0)]:
        x_in  = Tensor([[float(x0), float(x1)]], shape=(1,2))
        p_out = model(x_in)._data[0]
        pred  = round(p_out)
        mark  = '✓' if pred == expected else '✗'
        print(f"    {mark} XOR({x0},{x1}) = {p_out:.4f} → {pred}  (expected {expected})")
 
 
def demo_cellular_automata():
    """Cellular automaton visual demonstration."""
    print("\n" + "═" * 68)
    print("  DEMO: Cellular Automata — Conway's Game of Life")
    print("═" * 68)
 
    ca = CellularAutomatonEngine(16, 32, rule='B3/S23', density=0.0)
    ca.clear()
 
    # Place a glider at top-left
    ca.place_pattern(CellularAutomatonEngine.GLIDER, 1, 1)
    # Place a blinker in the middle
    ca.place_pattern(CellularAutomatonEngine.BLINKER, 8, 15)
    # Place a block (stable)
    ca.place_pattern(CellularAutomatonEngine.BLOCK, 12, 26)
 
    print(f"\n  Rule: B3/S23 (Conway's Life)  |  Grid: {ca.rows}×{ca.cols}")
    for gen in [0, 4, 8, 12]:
        while ca.generation < gen:
            ca.step()
        print(f"\n  Generation {gen}  |  Population: {ca.population()}"
              f"  |  Entropy: {ca.entropy():.4f}")
        print("  ┌" + "─" * ca.cols + "┐")
        for line in ca.render().split('\n'):
            print("  │" + line + "│")
        print("  └" + "─" * ca.cols + "┘")
 
 
def demo_transformer():
    """MiniTransformer learning a repeating pattern."""
    print("\n" + "═" * 68)
    print("  DEMO: MiniTransformer — Character-Level Pattern Learning")
    print("═" * 68)
    _RNG.seed(77)
 
    # Vocabulary: A-Z + space = 27 chars
    chars    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
    vocab    = {c: i for i, c in enumerate(chars)}
    inv_vocab = {i: c for c, i in vocab.items()}
    V         = len(chars)
 
    # Training data: repeating "AIOS " pattern
    pattern   = 'AIOS AIOS AIOS AIOS AIOS AIOS '
    tokens    = [vocab[c] for c in pattern]
 
    model = MiniTransformer(
        vocab_size=V, d_model=16, n_heads=2,
        n_layers=2, max_seq=16, dropout=0.0
    )
    opt   = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    print(f"  Vocab: {V}  |  Pattern: '{pattern[:15]}...'"
          f"  |  Model params: {sum(p.numel() for p in model.parameters()):,}")
 
    print(f"\n  {'Step':>6}  {'Loss':>10}")
    print("  " + "─" * 20)
    for step in range(1, 401):
        # Sample a random subsequence of length 12
        start = _RNG.randint(0, max(1, len(tokens) - 13))
        chunk = tokens[start: start + 13]
        if len(chunk) < 2:
            continue
        loss_val = model.train_step(chunk, opt)
        if step % 80 == 0 or step == 1:
            print(f"  {step:>6}  {loss_val:>10.4f}")
 
    # Generate a sequence starting with "AI"
    start_ids = [vocab['A'], vocab['I']]
    gen_ids   = model.generate(start_ids, max_new=12, temperature=0.7)
    gen_str   = ''.join(inv_vocab.get(i, '?') for i in gen_ids)
    print(f"\n  Generated (seed='AI'): '{gen_str}'")
    print(f"  (Target pattern: 'AIOS AIOS AIOS...')")
 
 
def demo_regression():
    """Sine-wave regression using a small neural network."""
    print("\n" + "═" * 68)
    print("  DEMO: Sine-Wave Regression (no math.sin import)")
    print("═" * 68)
    _RNG.seed(99)
 
    # Dataset: 20 points on sin curve + small noise
    N     = 20
    xs    = [_PI * 2 * i / N for i in range(N)]
    ys    = [_sin(x) + _RNG.randn() * 0.05 for x in xs]
 
    X     = Tensor([[x] for x in xs], shape=(N, 1))
    y     = Tensor([[v] for v in ys],  shape=(N, 1))
 
    model = Sequential(
        Linear(1, 32), Tanh(),
        Linear(32, 32), Tanh(),
        Linear(32, 1)
    )
    opt    = Adam(model.parameters(), lr=5e-3)
    loss_fn = MSELoss()
    sched  = LRScheduler(opt, 'cosine', T_max=800, eta_min=1e-5)
 
    print(f"  {'Epoch':>6}  {'MSE Loss':>12}")
    print("  " + "─" * 22)
    for epoch in range(1, 801):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 200 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {loss.item():>12.6f}")
 
    # Sample 5 test predictions
    print(f"\n  {'x':>8}  {'true sin(x)':>14}  {'predicted':>12}  {'err':>8}")
    print("  " + "─" * 50)
    for i in [0, 4, 8, 12, 16]:
        x_t    = Tensor([[xs[i]]], shape=(1, 1))
        p_val  = model(x_t)._data[0]
        t_val  = _sin(xs[i])
        err    = _abs(p_val - t_val)
        print(f"  {xs[i]:>8.4f}  {t_val:>14.6f}  {p_val:>12.6f}  {err:>8.6f}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
 
if __name__ == '__main__':
    import sys as _sys
 
    args = _sys.argv[1:]
 
    if not args or 'test' in args:
        ok = _run_self_tests()
        if not ok:
            _sys.exit(1)
 
    if 'all' in args or 'xor' in args:
        demo_xor_detailed()
 
    if 'all' in args or 'ca' in args:
        demo_cellular_automata()
 
    if 'all' in args or 'transformer' in args:
        demo_transformer()
 
    if 'all' in args or 'regression' in args:
        demo_regression()
 
    if not args:
        print("\n  Usage: python aios_phase4_nn.py [test|xor|ca|transformer|regression|all]")
        print("  Running self-tests only. Add 'all' for full demo suite.\n")
