# Epistemic Social Learning (ESL)

Code and frozen numerical evidence for the NeurIPS 2026 paper:

> **Epistemic Social Learning: Latent Behavioral Structure under Endogenous Multi-Agent Interaction**

**Authors:** Jainendra Shukla, Dhruv Jaiswal, Divyanshi Beniwal, and Kiriti Kanjilal.  
**Affiliation:** IIIT Delhi, New Delhi, India.  
**Contact:** {jainendra,dhruv23200,divyanshi23209,kanjilal}@iiitd.ac.in.

ESL is a two-timescale learner for repeated multi-agent interaction. Fast Bayesian beliefs over shared behavioral prototypes are coupled to a slower prototype update driven by belief-weighted observations. Because the learner’s actions shape the data it sees, the observation process is endogenous.

This repository is a camera-ready research release: the accepted algorithm, the committed twelve-game benchmark, experiment drivers, and compact cell-level evidence needed to recompute the published tables and figures.

## Repository contents

| Path | Role |
|---|---|
| `esl/` | ESL core, beliefs, prototypes, metrics, and experiment drivers |
| `esl/res/` | Twelve-game (3×3) evaluation protocol and baselines |
| `benchmark/` | Frozen payoff matrices (load these; do not resample) |
| `evidence/` | Compact cell-level CSVs used for the accepted numbers |
| `scripts/` | Verification, table/figure regeneration, and rerun helpers |
| `tests/` | Protocol, metric, and numerical-regression tests |
| `third_party/PPO-PyTorch/` | Vendored official `PPO.py` (MIT) for the PPO baseline |

## Installation

Python 3.10–3.13. The accepted numerical artifacts were produced with **NumPy 2.2.6** and **SciPy 1.16.0**.

```bash
git clone https://github.com/jainendra2019/esl.git
cd esl
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

Optional, only if you rerun the PPO baseline:

```bash
python -m pip install torch
```

Table and figure regeneration do not require PyTorch.

## Quick verification

Modest CPU. Should finish in a few minutes.

```bash
python -m pytest -q
python scripts/verify_release.py
```

This runs unit/protocol tests, regenerates the twelve 3×3 matrices in memory and compares them to the committed files, checks evidence hashes, and runs a 40-step ESL smoke cell.

## Reproduce reported tables and figures

These commands **do not** rerun the expensive simulations. They recompute published summaries from `evidence/`.

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

Outputs:

- `outputs/tables/` — method, per-game, K-sensitivity, OMD, ablation, and sensitivity CSVs
- `outputs/figures/` — PDF/PNG bar and line plots from those tables

Accepted flagship ESL means (n=120 game–seed cells, λ=2):

| Metric | Mean |
|---|---:|
| PSR | 162.97 |
| DPR | 106.67 |
| LTE | 3.225421 (display 3.23) |

PSR and DPR use the inclusive post-switch window `0 ≤ τ ≤ H`. LTE uses a time-varying Hungarian alignment of pre-observation beliefs and averages over `τ < H`.

ESL has the lowest PSR point estimate among {ESL, Fixed-K Bayesian, SA-FP, Online EM} on **8 of 12** games, and a lower PSR point estimate than SA-FP on **11 of 12** games. These are distinct claims.

The held-out OMD comparison is a **separate** protocol (development seeds 0–2, evaluation seeds 3–9, n=84, ESL λ=10, OMD η=1.0) and is not mixed with the headline λ=2 flagship.

## Rerun experiments from scratch

Full paper-scale runs are CPU-heavy (twelve games × ten seeds × methods × T=1000). Wall-clock time is hours to days depending on cores and whether PPO/torch is included. Cross-platform bitwise identity of new simulation output is **not** claimed; accepted numbers are those in `evidence/`.

Headline twelve-game adaptation (λ=2; includes ESL, Fixed-K, SA-FP, Online EM; PPO if torch is installed):

```bash
python -m esl.res --num-actions 3 --full --out-dir outputs/flagship
```

Held-out OMD/MWU bake-off:

```bash
python scripts/run_omd_clean_psr.py
```

Matched-component ablations and α / batch-size sweeps:

```bash
python -m esl.res.ablation_a3 ablation --out-dir outputs/ablation_a3
python -m esl.res.ablation_a3 alpha --out-dir outputs/alpha_sensitivity
python -m esl.res.ablation_a3 batch --out-dir outputs/batch_sensitivity
```

Inverse-temperature, opponent count, and calibrated shifts:

```bash
python scripts/run_focal_lambda_sensitivity.py --out outputs/lambda_sensitivity
python -m esl.res.opponent_count_sensitivity run --J 10 --mode switch_matched
python scripts/run_controlled_behavioral_shift.py --out outputs/mechanistic_analysis_a3
python scripts/run_k_sensitivity.py --out outputs/k_sensitivity
```

Observation noise / missingness uses the same categorical cell with `--p-obs` and `--obs-noise`:

```bash
python -m esl.res --num-actions 3 --full --p-obs 0.5 --out-dir outputs/visibility
python -m esl.res --num-actions 3 --full --obs-noise 0.1 --out-dir outputs/noise
```

Population recovery (Algorithm 1, N=20 and N=100):

```bash
python scripts/run_psrv_evaluation.py --help
```

Robbins–Monro diminishing-step diagnostic:

```bash
python -m esl.experiments.figure4_rm_dynamics --help
```

See `configs/README.md` for protocol locks (seeds, H=100, switch interval, no resampling).

## Hardware and runtime

- Quick verification: laptop CPU, a few minutes, no GPU.
- Table/figure regeneration: seconds.
- Full flagship rerun: many CPU-hours. PPO uses CPU torch in this protocol.
- The accepted evidence package was produced on CPU with NumPy 2.2.6.

## Reproducibility limitations

- Canonical payoff matrices are the committed JSON files (SHA-256 in `reproducibility/SHA256SUMS` and `benchmark/flagship_3x3.manifest.json`). In-memory `generate_3x3_suite(seed=20260728)` matches those files bit-for-bit on macOS with NumPy 2.2.6. Other platforms or NumPy builds may draw a different stream; evaluation must load the JSON files and must not resample.
- Online EM’s reported LTE near 1.10 equals `-log(1/3)` in this implementation because the three prototypes collapse and beliefs remain uniform. It is not accurate regime recovery.
- The OMD comparison uses a held-out seed split and a separately selected λ. Do not compare those PSR values to the headline λ=2 flagship.
- This release does not include sequential-state or Overcooked environments.
- Constant-step and diminishing-step dynamics plots are diagnostics, not convergence certificates.

## License and citation

This software is released under the MIT License (see `LICENSE`).

**If you use this software, the benchmark, or its results, please cite our NeurIPS 2026 paper:**

```bibtex
@inproceedings{shukla2026epistemic,
  title     = {Epistemic Social Learning: Latent Behavioral Structure under Endogenous Multi-Agent Interaction},
  author    = {Shukla, Jainendra and Jaiswal, Dhruv and Beniwal, Divyanshi and Kanjilal, Kiriti},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

GitHub's citation feature also uses [`CITATION.cff`](CITATION.cff). A DOI and proceedings URL will be added when assigned.
