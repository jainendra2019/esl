# ESL — Epistemic Social Learning

Code for **multi-agent belief dynamics** and **slow prototype (latent-type) learning** in repeated 2-action games. ESL is a **feedback-coupled** system: observations depend on actions driven by **current** beliefs and (in adaptation mode) best responses—so data are **endogenous**, not i.i.d. latent-variable samples. Beliefs update every environmental step; prototype logits update on a slower schedule with **averaged batch gradients** over full interaction batches (two-timescale stochastic approximation; see **PRD.md** §3, §5, §8 and **ALGORITHM.md**).

**Repository:** [github.com/jainendra2019/esl](https://github.com/jainendra2019/esl)

## Features

- **Recovery mode:** fixed hidden policies (e.g. Always Cooperate / Always Defect), random ordered-pair observations, Bayes updates followed by Euclidean projection onto the δ-floored simplex Δ_K^δ.
- **Interaction protocol:** each environment round samples \(L_t\) distinct ordered pairs without replacement (`interaction_pairs_min` / `max`); prototype SGD runs every **\(Q\)** interaction events (`prototype_update_every` after `validate()`), not once per round when \(L_t>1\).
- **Metrics:** Hungarian-matched cross-entropy vs. true policies, belief entropy, argmax accuracy.
- **Paper-style experiments:** two-type separation suite (`main` / `symmetric` / `freeze prototype`) plus plotting helpers for observability and mechanism figures.
- **Tests:** verification (math / algorithm), validation (end-to-end), edge cases; heavy checks marked **`slow`**.

## Requirements

- Python 3.10+ recommended (3.13 works in CI-style local runs).
- Dependencies: `numpy`, `scipy`, `matplotlib`; `pytest` for tests.

## Install

```bash
git clone https://github.com/jainendra2019/esl.git
cd esl
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .              # editable install of the `esl` package (optional)
```

## Tests

```bash
# Fast loop (excludes @pytest.mark.slow)
pytest -q

# Include long two-type separation validations
pytest -q --runslow
```

See **`tests/TEST_PLAN.md`** for the verification vs. validation map.

## Quick start (~5 minutes the first time)

Rough budget from a fresh clone: **venv + pip ~1–3 min**, **`pytest -q` ~30–60 s**, **smoke experiment ~20–50 s** (three conditions × 50 rounds + matplotlib). After dependencies are installed, the test + smoke loop is usually **~1–2 minutes**.

From the repo root, with your venv active:

```bash
# 1) Fast test suite (excludes @pytest.mark.slow)
pytest -q

# 2) Smoke run: three conditions (main / symmetric / freeze prototype) + PNGs
python3 -m esl.experiment_two_type_separation \
  --out runs/quickstart_smoke \
  --rounds 50 \
  --m 5 \
  --seed 42 \
  --lr-scale 18 \
  --obs-prob 1.0
```

**CLI essentials** (`esl.experiment_two_type_separation`; **`--out` is required**):

| Flag | Role | Quick-start value |
|------|------|-------------------|
| `--out` | Output root (created if missing) | `runs/quickstart_smoke` |
| `--rounds` | Environment rounds per condition (`num_rounds`) | `50` smoke; `250`+ for smoother curves |
| `--m` | `prototype_update_every` → **\(Q\)** in config (prototype step every **\(Q\) interaction events**). This driver uses the default **one ordered pair per round**, so **\(Q\)** equals “every *m* rounds” here. | `5` |
| `--seed` | RNG seed | `42` |
| `--lr-scale` | `prototype_lr_scale` for **main** and **freeze** conditions (symmetric baseline uses `1.0`) | `18` |
| `--obs-prob` | Observation probability \(p_{\mathrm{obs}}\) (`1.0` = full observability) | `1.0` |
| `--n-agents` | Agent count (default **6** = three “always C” + three “always D” when \(N=6\), \(K=2\)) | default `6` |
| `--n-prototypes` | \(K\) | default `2` |
| `--delta` | Simplex floor \(\delta\) (`delta_simplex`) | default `1e-4` |
| `--no-plots` | Skip figure PNGs; metrics/CSVs only | — |
| `--until-converged` | Stop early when convergence tests pass; `--rounds` is **T_max** | optional; tune with `--conv-window`, `--conv-eps-*` |

For **multi-pair rounds** (\(L_t>1\)) or flagship-style recovery, configure `ESLConfig` in Python or another driver; this experiment script keeps the default **one random pair per round** so the table above stays accurate.

**Optional faster demo** (single recovery or adaptation run, default 200 rounds, writes under `runs/...` if you pass `--out`):

```bash
python3 -m esl --rounds 80 --out runs/esl_demo
```

**You should see** (under `runs/quickstart_smoke/` after step 2):

| Path | What it is |
|------|------------|
| `experiment_manifest.json` | Run settings (`rounds`, `m`, `seed`, `lr_scale`, `obs_prob`, …) |
| `figure_prototype_separation.png` | Three-way prototype dynamics (+ separation panel) |
| `figure_matched_ce.png`, `figure_belief_entropy.png` | Matched CE and mean entropy |
| `main/summary_metrics.json` | Final metrics for the **main** condition |
| `main/`, `symmetric/`, `freeze_proto_baseline/` | Per-condition CSVs + `config.json` |

Figures will look **noisy / incomplete** at 50 rounds; they are only for a **fast pipeline check**. For smoother curves, use **`--rounds 250`** (or 500+) as in **Main commands** below.

### One-shot script

```bash
chmod +x scripts/reproduce_paper_figures.sh   # once
./scripts/reproduce_paper_figures.sh
```

Optional **longer** bundle (main @ 250 rounds + mechanism plot @ `p_obs=0.5`, ~400 rounds):

```bash
ESL_FULL_REPRO=1 ./scripts/reproduce_paper_figures.sh
```

## Main commands

**Two-type separation experiment** (three conditions, CSV/JSON artifacts, figures):

```bash
python3 -m esl.experiment_two_type_separation \
  --rounds 500 --m 5 --seed 42 --lr-scale 18 --obs-prob 1.0 \
  --out runs/my_run
```

- **`--out`** is required. Use **`--no-plots`** for metrics only.
- Outputs under `runs/<name>/`: `main/`, `symmetric/`, `freeze_proto_baseline/`, `experiment_manifest.json`, and figure PNGs unless disabled.

**Hand trace** (fixed-θ sanity check, CSV + optional cooperate stream):

```bash
python3 -m esl.hand_trace --help
```

**Post-hoc figures** (after a run exists; beliefs use Euclidean projection onto $\Delta_K^\delta$ throughout):

```bash
# Separation vs. observability (one …/main path per p_obs curve)
python3 -m esl.plot_observability_separation \
  --runs runs/two_type_sep_obs10_long/main runs/two_type_sep_obs05/main runs/two_type_sep_obs02/main \
  --out runs/paper_figures/figure_separation_vs_obs.png

# Mechanism: belief entropy + prototype separation (main folder only)
python3 -m esl.plot_esl_mechanism \
  --main-dir runs/two_type_sep_mechanism_obs05/main \
  --out runs/paper_figures/figure_mechanism_obs05.png
```

**`runs/`** is mostly gitignored; canonical flagship recovery outputs live under `runs/recovery_best_bal20_*_seed42/` and `..._seed43/` and are tracked. Reproduce other figures locally with the commands above.

## NeurIPS-style recovery experiments (presets / sweeps / figures)

Presets match the **20-agent balanced** flagship geometry (variable \(L_t\in[5,15]\), \(Q=15\), `prototype_lr_scale=22`, `init_noise=0.05`, …). Outputs default under **`runs/neurips/`** (gitignored except your own exceptions).

```bash
# List presets
python3 -m esl.experiments list-presets

# One smoke run (5 rounds — for CI / wiring checks)
python3 -m esl.experiments run --preset recovery_flagship --smoke

# Full flagship (10k rounds — long)
python3 -m esl.experiments run --preset recovery_flagship --seed 42

# Sparse observability (set p_obs in --variant)
python3 -m esl.experiments run --preset recovery_sparse_obs --variant 0.5 --seed 42

# Fixed interaction budget (500 events, L_t=10 per round → 50 rounds)
python3 -m esl.experiments run --preset recovery_short_horizon --variant 500 --seed 42

# Ablation knobs (variant = numeric parameter)
python3 -m esl.experiments run --preset recovery_lr_sweep --variant 18 --seed 42
python3 -m esl.experiments run --preset recovery_init_noise_sweep --variant 0.05 --seed 42
python3 -m esl.experiments run --preset recovery_Q_sweep --variant 15 --seed 42

# Weak recovery contrast + fixed-prototype baseline (beliefs only)
python3 -m esl.experiments run --preset recovery_failure_case --seed 42
python3 -m esl.experiments run --preset recovery_fixed_prototype_baseline --seed 42
```

**Sweep helper** (sequential full runs — long): see **`scripts/run_neurips_sweeps.sh`**.

**Aggregate CSV** (schema v1) over completed runs under a root:

```bash
python3 -m esl.experiments aggregate runs/neurips -o runs/neurips/_aggregates/summary_all_runs.csv
```

**Plots** (after runs exist):

```bash
python3 -m esl.plot_neurips flagship runs/neurips/recovery_flagship/long/seed_42 \
  -o runs/neurips/_figures/flagship_panels.png
python3 -m esl.plot_neurips robustness runs/neurips/_aggregates/summary_all_runs.csv \
  -o runs/neurips/_figures/robustness.png
python3 -m esl.plot_neurips compare runs/neurips/recovery_failure_case/weak/seed_42 \
  runs/neurips/recovery_flagship/long/seed_42 -o runs/neurips/_figures/compare.png
```

Each run writes **`summary_metrics.json`** (including `final_mce`, `final_prototype_gap`, `num_interaction_events_executed`, `prototype_update_every_q`, …) plus **`run_manifest.json`** (preset / variant / optional `target_interaction_budget`).

## Package layout

| Path | Role |
|------|------|
| `esl/config.py` | `ESLConfig`, seeds, schedules |
| `esl/trainer.py` | Main loop, batching, logging |
| `esl/interaction_protocol.py` | \(L_t\) and ordered-pair sampling |
| `esl/beliefs.py`, `esl/prototypes.py`, `esl/metrics.py` | Core math |
| `esl/synthetic_population.py` | Optional simulator ground truth (evaluation only; not used by `run_esl`) |
| `esl/experiment_two_type_separation.py` | Three-condition experiment driver |
| `esl/experiments/` | NeurIPS presets, `run_manifest.json`, aggregate CSV builder |
| `esl/plot_neurips.py` | Flagship / robustness / compare figures |
| `esl/plot_observability_separation.py` | Overlay separation curves across `p_obs` |
| `esl/plot_esl_mechanism.py` | Two-panel mechanism figure |
| `esl/hand_trace.py` | Traceable single-run diagnostics |
| `tests/` | Pytest suite + plan |
| `scripts/reproduce_paper_figures.sh` | One-shot quick (and optional full) local reproduction |
| `scripts/run_neurips_sweeps.sh` | Batch sparse / budget / lr / noise / Q sweeps into `runs/neurips/` |

## Documentation

- **`PRD.md`** — product / algorithm requirements; **§19** is the v2 addendum (variable \(L_t\), \(Q\), \(\hat\pi\), synthetic eval, MCE naming, SA wording).
- **`ALGORITHM.md`** — **only** algorithm doc: *Current implementation* (narrative), *Theory-aligned target*, and *Pseudocode (Cormen-style)* in one file (no separate pseudocode markdown).
- **`tests/TEST_PLAN.md`** — verification vs. validation map; keep in sync with `ALGORITHM.md` when the trainer contract changes.

## License

Released under the **MIT License** (see **`LICENSE`**): open and permissive, with **no warranty** and **no liability** for use of the software. Intended especially for **research and academic** work; cite the project or paper when appropriate.

## Citation

If you use this code in research, please cite the accompanying paper when available and link this repository.
