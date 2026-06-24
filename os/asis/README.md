# ASIS 2.0 — Algebraic Swarm Intelligence System

> *A deterministic, rule-based multi-agent architecture implementing a Symbolic Algebra of Concepts (SAC), now with real-time cyberpunk visualization.*

---

## What Makes This Remarkable

### 1. Formal Algebraic Foundation
ASIS is built on a **typed lambda calculus variant** where:
- **ConceptAtoms** are typed atomic elements (Entity, Action, Goal, Constraint, etc.)
- **10 algebraic operators** form a closed algebra over expression trees (⊗, ⊕, ¬, π, ι, β, ρ, τ, γ, μ)
- **Immutable DAG-based expressions** with content-addressed SHA256 identifiers
- **Forward-chaining rule engine** with unification and pattern matching
- **Deterministic discrete event simulation** — zero randomness, full traceability

### 2. Live Cyberpunk Dashboard
A self-contained, zero-dependency HTML dashboard featuring:
- **Force-directed agent network** with 6 specialized agent types
- **Animated message packets** traveling between agents in real-time
- **Neon glow effects** and particle systems
- **Live expression tree visualization**
- **Interactive task injection** — click "Inject Task" and watch the swarm solve it
- **Convergence detection** with animated banners
- **Full keyboard shortcuts** (Space to pause, Ctrl+Enter to inject, Escape to close)

### 3. Production-Grade Agent Architecture
| Agent | Role | Capability |
|-------|------|------------|
| **Orchestrator** | Coordination | Task decomposition, routing, result aggregation |
| **Analyst** | Analysis | Requirement analysis, constraint identification, feasibility assessment |
| **Planner** | Planning | Multi-step plan generation with validation gates |
| **Executor** | Execution | Step-by-step execution with logging |
| **Validator** | Validation | Success/failure detection, feedback loops |
| **Synthesizer** | Synthesis | Final output assembly and delivery |

### 4. Determinism Guarantees
- Zero non-determinism (no random, async, or threading in core)
- Content-addressed SHA256 identifiers
- Canonical ordering via sorted processing and tuple-based bindings
- Full audit trail via global log, versioned blackboard, and execution snapshots

---

## Quick Start

### Open the Live Dashboard
Simply open `asis_dashboard.html` in any modern browser:
```bash
open asis_dashboard.html
# or
firefox asis_dashboard.html
# or
chrome asis_dashboard.html
```

The dashboard runs a **live simulation** of the swarm in real-time. Watch as:
1. Tasks are injected into the Orchestrator
2. Messages pulse through the network as glowing packets
3. Agents light up when processing
4. The system converges to a fixed point

### Run the Enhanced Engine
```bash
python asis_enhanced.py
```

This executes a full symbolic simulation and exports a JSON trace to `asis_trace.json`.

### Programmatic Usage
```python
from asis_enhanced import *

# Create swarm
swarm = create_default_swarm()

# Inject a complex algebraic task
task = C.compose(
    C.goal("optimize_system"),
    C.constraint("latency < 100ms"),
    C.constraint("throughput > 1000rps")
)
swarm.inject_task(task)

# Run until algebraic fixed point (convergence)
result = swarm.run_until_convergence(max_steps=50)
print(f"Converged in {result['steps_executed']} steps")

# Export full trace for visualization
swarm.save_trace("my_trace.json")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ASIS 2.0 ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 5: Swarm Controller    │ Step-based execution      │
│  LAYER 4: Agent Hierarchy     │ 6 specialized roles         │
│  LAYER 3: Communication       │ AlgebraicMessage, Channels  │
│  LAYER 2: Rule Engine         │ Forward-chaining + unification│
│  LAYER 1: Algebraic Core      │ SAC with 10 operators       │
└─────────────────────────────────────────────────────────────┘
```

### Algebraic Operators
| Symbol | Name | Semantics |
|--------|------|-----------|
| ⊗ | COMPOSE | Sequential composition |
| ⊕ | UNION | Parallel combination |
| ¬ | NEGATE | Logical negation |
| π | PROJECT | Extract / select |
| ι | INJECT | Embed into space |
| β | BIND | Parameterize |
| ρ | REDUCE | Aggregate |
| τ | TRANSFORM | Cross-domain map |
| γ | GUARD | Conditional |
| μ | FIXPOINT | Iterate to convergence |

---

## Files

| File | Description |
|------|-------------|
| `asis_enhanced.py` | Production-grade ASIS engine with rule engine, full agent hierarchy, and trace export |
| `asis_dashboard.html` | Self-contained interactive cyberpunk visualization (40KB, zero dependencies) |
| `asis_trace.json` | Sample execution trace from a 3-task simulation |

---

## Keyboard Shortcuts (Dashboard)

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume simulation |
| `Ctrl + Enter` | Open task injection modal |
| `Escape` | Close modal |
| `Click agent` | Select agent (see details) |

---

## Mathematical Properties

- **Closed Algebra**: All operators produce valid Expression trees
- **Associativity**: COMPOSE and UNION are associative (flattened automatically)
- **Identity**: `_identity` element for COMPOSE
- **Absorption**: `_zero` element absorbs in COMPOSE
- **Idempotence**: A ⊕ A = A
- **Double Negation**: ¬¬A = A

---

*Built on the Symbolic Algebra of Concepts — where agents communicate through expression trees, not natural language.*
