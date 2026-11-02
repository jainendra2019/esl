# Fixed benchmark suite

The accepted NeurIPS 2026 evaluation uses the twelve committed **3×3** games
listed in `flagship_3x3.manifest.json` (SHA-256 of each JSON file included).
Load these files; do not resample payoffs.

The original 24-instance `manifest.json` also records an earlier 2×2 suite.
Those 2×2 files remain for protocol compatibility and are not the headline benchmark.

**Interaction regime ≠ payoff realization.**

This directory holds a **committed**, versioned set of matrix-game instances used by the
Robustness Extension Suite (RES). Instances were generated **once** by
`scripts/generate_res_benchmark.py` under family constraints with a diversity filter.
Experiments **load these files**; they do not resample payoffs online.

## Families

| Family | Defining structure (2×2) |
|---|---|
| `cooperative/` | Prisoner's Dilemma inequalities \(T>R>P>S\) |
| `coordination/` | Stag-Hunt inequalities \(R>T \ge P > S\) |
| `adversarial/` | Zero-sum (`col = -row`), varying magnitudes |

3×3 instances use the same family labels with regime-analogue constructions
(documented in `esl/res/benchmark.py`).

## Quality criterion

Instances maximize diversity within each regime while satisfying family constraints.
Near-duplicate payoff matrices are rejected at generation time.

## Do not

- Edit instance JSON ad hoc without regenerating + re-auditing diversity
- Sample new payoffs inside training / evaluation loops
