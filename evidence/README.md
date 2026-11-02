# Frozen numerical evidence

Compact cell-level tables used to recompute the accepted manuscript numbers.
These files are inputs to `scripts/reproduce_tables.py` and
`scripts/reproduce_figures.py`. They are not full interaction trajectories.

| Directory | Contents |
|---|---|
| `flagship/` | Twelve-game × 10-seed cells (SHA-256 `243c8ac5…e740e9`) |
| `k_sensitivity/` | ESL K∈{2,3,5} sweep; K=3 matches flagship ESL cells |
| `omd/` | Held-out OMD/MWU bake-off, including paired Δ and CI |
| `ablation/` | Matched-component ablation cells and paired tests |
| `lambda/` | Inverse-temperature sensitivity (split opponent RNG) |
| `alpha/` | Prototype-update coefficient sensitivity |
| `batch/` | Batch-size sensitivity |
| `robustness/` | Observation missingness and noise |
| `opponent_count/` | J∈{3,10,100} encounter-clock diagnostics |
| `controlled_shift/` | Calibrated behavioral-shift ladder |
| `rm_diagnostic/` | Diminishing-step Robbins–Monro timeseries |
| `psrv/` | Population recovery cells for N=20 and N=100 |

Do not concatenate files across protocols. Flagship λ=2 cells are not the
held-out OMD λ=10 cells. Lambda-sensitivity uses a split exogenous RNG, so
its λ=2 mean is not the flagship PSR.
