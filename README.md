# ESL — Epistemic Social Learning

Code for **multi-agent belief dynamics** and **slow prototype (latent-type) learning** in repeated 2-action games. Beliefs update every environmental step; prototype logits update on a slower schedule with **averaged batch gradients** (two-timescale ESL). See **`PRD.md`** for the full specification.

**Repository:** [github.com/jainendra2019/esl](https://github.com/jainendra2019/esl)

## Features

- **Recovery mode:** fixed hidden policies (e.g. Always Cooperate / Always Defect), random ordered-pair observations, Bayes belief updates on the simplex with floor projection.
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

**Post-hoc figures** (after a run exists):

```bash
# Separation vs. observability (point at each run’s …/main folder)
python3 -m esl.plot_observability_separation \
  --runs runs/run_a/main runs/run_b/main runs/run_c/main \
  --out runs/figure_separation_vs_obs.png

# Mechanism: belief entropy + prototype separation (main folder only)
python3 -m esl.plot_esl_mechanism --main-dir runs/my_run/main
```

Generated **`runs/`** is gitignored; reproduce figures locally with the commands above.

## Package layout

| Path | Role |
|------|------|
| `esl/config.py` | `ESLConfig`, seeds, schedules |
| `esl/trainer.py` | Main loop, batching, logging |
| `esl/beliefs.py`, `esl/prototypes.py`, `esl/metrics.py` | Core math |
| `esl/experiment_two_type_separation.py` | Three-condition experiment driver |
| `esl/plot_observability_separation.py` | Overlay separation curves across `p_obs` |
| `esl/plot_esl_mechanism.py` | Two-panel mechanism figure |
| `esl/hand_trace.py` | Traceable single-run diagnostics |
| `tests/` | Pytest suite + plan |

## Documentation

- **`PRD.md`** — product / algorithm requirements used to align implementation and tests.

## License

Released under the **MIT License** (see **`LICENSE`**): open and permissive, with **no warranty** and **no liability** for use of the software. Intended especially for **research and academic** work; cite the project or paper when appropriate.

## Citation

If you use this code in research, please cite the accompanying paper when available and link this repository.
