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

## Quick start: minimal replication (~5 minutes)

From the repo root, with your venv active:

```bash
# 1) Sanity-check the implementation (~30–60 s)
pytest -q

# 2) Short end-to-end experiment (~20–40 s): three conditions + figures
python3 -m esl.experiment_two_type_separation \
  --rounds 50 --m 5 --seed 42 --lr-scale 18 --obs-prob 1.0 \
  --out runs/quickstart_smoke
```

**You should see** (under `runs/quickstart_smoke/`):

| Path | What it is |
|------|------------|
| `experiment_manifest.json` | Run settings (`rounds`, `m`, `seed`, `lr_scale`, `obs_prob`) |
| `figure_prototype_separation.png` | Three-way prototype dynamics (+ separation panel) |
| `figure_matched_ce.png`, `figure_belief_entropy.png` | Matched CE and mean entropy |
| `main/summary_metrics.json` | Final metrics for the **main** condition |
| `main/`, `symmetric/`, `freeze_proto_baseline/` | Per-condition CSVs + `config.json` |

Figures will look **noisy / incomplete** at 50 rounds; they are only for a **fast pipeline check**. For smoother curves, use **`--rounds 250`** (or 500+) as in the section below.

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

## Package layout

| Path | Role |
|------|------|
| `esl/config.py` | `ESLConfig`, seeds, schedules |
| `esl/trainer.py` | Main loop, batching, logging |
| `esl/interaction_protocol.py` | \(L_t\) and ordered-pair sampling |
| `esl/beliefs.py`, `esl/prototypes.py`, `esl/metrics.py` | Core math |
| `esl/synthetic_population.py` | Optional simulator ground truth (evaluation only; not used by `run_esl`) |
| `esl/experiment_two_type_separation.py` | Three-condition experiment driver |
| `esl/plot_observability_separation.py` | Overlay separation curves across `p_obs` |
| `esl/plot_esl_mechanism.py` | Two-panel mechanism figure |
| `esl/hand_trace.py` | Traceable single-run diagnostics |
| `tests/` | Pytest suite + plan |
| `scripts/reproduce_paper_figures.sh` | One-shot quick (and optional full) local reproduction |

## Documentation

- **`PRD.md`** — product / algorithm requirements; **§19** is the v2 addendum (variable \(L_t\), \(Q\), \(\hat\pi\), synthetic eval, MCE naming, SA wording).
- **`ALGORITHM.md`** — **only** algorithm doc: *Current implementation* (narrative), *Theory-aligned target*, and *Pseudocode (Cormen-style)* in one file (no separate pseudocode markdown).
- **`tests/TEST_PLAN.md`** — verification vs. validation map; keep in sync with `ALGORITHM.md` when the trainer contract changes.

## License

Released under the **MIT License** (see **`LICENSE`**): open and permissive, with **no warranty** and **no liability** for use of the software. Intended especially for **research and academic** work; cite the project or paper when appropriate.

## Citation

If you use this code in research, please cite the accompanying paper when available and link this repository.
