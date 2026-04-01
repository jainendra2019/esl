# ESL algorithm documentation — index

This repository maintains **two** algorithm writeups plus the **PRD**:

| Document | Role |
|----------|------|
| **[ALGORITHM_CURRENT.md](ALGORITHM_CURRENT.md)** | **Implementation-faithful:** what `esl.trainer.run_esl` and the interaction protocol actually do (variable \(L_t\), \(Q\)-based prototype steps, etc.). |
| **[ALGORITHM_TARGET.md](ALGORITHM_TARGET.md)** | **Theory-aligned target:** intended math (predictive \(\hat\pi\), synthetic population, SA interpretation, MCE / belief metrics). |
| **[ESL_PSEUDOCODE.md](ESL_PSEUDOCODE.md)** | **Cormen-style pseudocode:** consolidated implementation-faithful pseudocode + a flagship recovery configuration block. |
| **[PRD.md](PRD.md)** | Product / MVP spec, experiments, and normative requirements; includes an **addendum** on target protocol and evaluation. |

**Start here for coding reviews:** [ALGORITHM_CURRENT.md](ALGORITHM_CURRENT.md).  
**Start here for paper alignment:** [ALGORITHM_TARGET.md](ALGORITHM_TARGET.md) + [PRD.md](PRD.md).

---

## Quick pointer

- **Trainer entrypoint:** `esl/trainer.py` → `run_esl`
- **Pair sampling / \(L_t\):** `esl/interaction_protocol.py`
- **Config:** `esl/config.py` → `ESLConfig`
- **Metrics (MCE, belief CE/KL vs type):** `esl/metrics.py`
- **Synthetic ground truth (not used by trainer):** `esl/synthetic_population.py`
- **Consolidated pseudocode:** `ESL_PSEUDOCODE.md`
