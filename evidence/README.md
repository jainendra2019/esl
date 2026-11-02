# Frozen numerical evidence

Compact cell-level tables used to recompute the accepted manuscript numbers.
These files are inputs to `scripts/reproduce_tables.py` and
`scripts/reproduce_figures.py`. They are not full interaction trajectories.
`reproduce_figures.py` writes supporting evidence-derived plots, not the exact
manuscript figure PDFs.

**Cell sources (independently recomputed in tests and `reproduce_tables.py`):**
`flagship/cells.csv`, `k_sensitivity/k_sweep_cells.csv`,
`omd/esl_lambda_star_cells.csv`, `omd/omd_per_opponent_cells.csv`,
`psrv/n20_cells.csv`, `psrv/n100_cells.csv`.

**Precomputed statistical outputs** (copied through; not re-derived here):
`k_sensitivity/summary.csv`, `k_sensitivity/paired_vs_k3.csv` (includes the
game-clustered bootstrap), `k_sensitivity/family_paired_vs_k3.csv`,
`omd/omd_psr_summary.csv`, `omd/omd_vs_esl_paired.json` (global-OMD contrasts
have no cell file in this package), and the ablation / λ / α / batch /
robustness / shift summary CSVs.

| Directory | Contents |
|---|---|
| `flagship/` | Twelve-game × 10-seed cells (SHA-256 `243c8ac5…e740e9`) |
| `k_sensitivity/` | ESL K∈{2,3,5} sweep; K=3 matches flagship ESL cells |
| `omd/` | Held-out OMD/MWU comparison, including paired Δ and CI |
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
