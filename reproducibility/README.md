# RES 3×3 benchmark freeze

This directory pins the original Robustness Extension Suite (RES) matrix-game
benchmark used in the expanded-adaptation experiments, plus the minimal code
needed to regenerate and load it.

Frozen JSON instances are **not** resampled at evaluation time. Do **not** run
`scripts/generate_res_benchmark.py` against the committed `benchmark/` tree:
that writer deletes existing JSON and would replace the freeze.

## Twelve 3×3 instances (headline suite)

Effective generator seed: **`20260728`**.

| ID | Source path |
|---|---|
| `coop3_01` | `benchmark/cooperative/coop3_01.json` |
| `coop3_02` | `benchmark/cooperative/coop3_02.json` |
| `coop3_03` | `benchmark/cooperative/coop3_03.json` |
| `coop3_04` | `benchmark/cooperative/coop3_04.json` |
| `coord3_01` | `benchmark/coordination/coord3_01.json` |
| `coord3_02` | `benchmark/coordination/coord3_02.json` |
| `coord3_03` | `benchmark/coordination/coord3_03.json` |
| `coord3_04` | `benchmark/coordination/coord3_04.json` |
| `adv3_01` | `benchmark/adversarial/adv3_01.json` |
| `adv3_02` | `benchmark/adversarial/adv3_02.json` |
| `adv3_03` | `benchmark/adversarial/adv3_03.json` |
| `adv3_04` | `benchmark/adversarial/adv3_04.json` |

Payoffs are stored at full Python `float` / NumPy `float64` precision in JSON.
Each 3×3 file records `"generator_seed": 20260728` and `"notes": "3x3_sampled"`.

`benchmark/manifest.json` also lists twelve 2×2 instances (`coop_01`–`04`,
`coord_01`–`04`, `adv_01`–`04`). Those files are frozen with the suite so the
original 24-instance manifest is complete. They are **not** the 3×3 headline
benchmark.

## Generator configuration

| Knob | Value |
|---|---|
| Module | `esl.res.benchmark.generate_3x3_suite` |
| CLI | `scripts/generate_res_benchmark.py` |
| CLI default `--seed` | `DEFAULT_GENERATOR_SEED = 20260725` (2×2 stream) |
| 3×3 seed | **`20260725 + 3 = 20260728`** |
| Candidates per family | 60 |
| Selected per family | 4 |
| Families (order) | cooperative, coordination, adversarial |
| Diversity threshold | `MIN_PAIRWISE_DISTANCE = 1.25` |
| Fallback fill threshold | `0.5 × 1.25 = 0.625` |
| RNG | `numpy.random.default_rng(seed)` |

**Do not pass `--seed 20260728` to the CLI.** That sets the 2×2 stream to
`20260728` and the 3×3 stream to `20260731`. The reproducing 3×3 call is:

```python
from esl.res.benchmark import generate_3x3_suite
specs = generate_3x3_suite(seed=20260728, per_family=4)
```

Equivalent CLI (writes files; do not use on the frozen tree):

```bash
PYTHONPATH=. python3 scripts/generate_res_benchmark.py --seed 20260725
```

The cooperative 3×3 sampler draws an unused `rng.uniform(0, 1, size=(3, 3))`
per candidate. That draw is part of the RNG stream and must be left in place
for bit-exact regeneration.

Greedy selection (`_greedy_diverse`) keeps the first candidate, then rejects
near-duplicates with min L2 distance `< 1.25` in the sampler feature space.
If fewer than four instances pass, it fills by maximum min-distance and may
accept distances down to `0.625`. The frozen coordination 3×3 group has
minimum pairwise distance **`0.8277621663576695`** (flattened-row audit
feature). That value is below `1.25` because fallback fill was used; it is
not a file error.

Audit distances for the frozen 3×3 files (`benchmark/diversity_audit.json`
and `outputs/res/eval/m2/gate.json`):

- `cooperative_A3`: `1.6531797267520525`
- `coordination_A3`: `0.8277621663576695`
- `adversarial_A3`: `1.9798215935340548`

## Dependencies

Verified regeneration environment:

- Python 3
- **NumPy 2.2.6** (bit-exact match confirmed)
- SciPy `>= 1.7` (only for `tests/test_res_benchmark.py::test_paired_stats_smoke`)
- pytest `>= 7.0` (optional; for the focused tests)

`numpy.random.Generator` (`default_rng`) is required. Other NumPy versions
that implement the same PCG64 `default_rng(20260728)` stream should match;
this freeze was checked against 2.2.6 only.

## Deterministic regeneration (in memory; do not overwrite `benchmark/`)

From the repository root:

```bash
PYTHONPATH=. python3 reproducibility/verify_benchmark_freeze.py
```

That script regenerates the twelve 3×3 matrices in memory, asserts
`numpy.array_equal` on every row and column payoff, checks instance IDs,
and verifies `reproducibility/SHA256SUMS`.

## Evaluation loading (no resampling)

`esl.res.benchmark.load_suite(root, num_actions=3)` reads
`benchmark/<family>/*.json` via `GameSpec.load_json`. It does not call
`generate_3x3_suite` or sample payoffs.

The RES cell runner (`esl.res.runner.run_res_m1` / `run_res_cell`, not part
of this minimal freeze) copies those matrices onto `MatrixGameTask.payoff_row`
/ `payoff_col`. Headline 3×3 methods share this load path.

Flagship protocol (from the frozen gate file): focal 1+3, `A=K=3`,
`T=1000`, `H=100`, switch interval 200, hazard 0.01, seeds 0–9, `p_obs=1`,
`obs_noise=0`, focal role `row`.

## Flagship cell table

Byte-identical copies of the local evaluation artifacts (the live `outputs/`
tree remains gitignored):

- `reproducibility/artifacts/expanded_adaptation_symmetric_cells.csv`
  (original path: `outputs/res/eval/expanded_adaptation/symmetric/cells.csv`)
- `reproducibility/artifacts/expanded_adaptation_symmetric_gate.json`
  (original path: `outputs/res/eval/expanded_adaptation/symmetric/gate.json`)

840 rows = 12 games × 10 seeds × 7 methods, including ESL, Online EM,
Fixed-K Bayesian, and SA-FP.

## SHA-256 manifest

`reproducibility/SHA256SUMS` covers the frozen JSON, generator, tests, and
flagship copies. Verify with:

```bash
shasum -a 256 -c reproducibility/SHA256SUMS
```

## Provenance limits

- Archived CSVs store `instance_id` (and metrics), **not** payoff hashes.
  Binding a historical row to a matrix is by ID plus this frozen JSON.
- `benchmark/diversity_audit.json` records a machine-local `manifest` path;
  that path is not used as a loader input.
- This freeze does not include the asymmetric suite (`benchmark_asym3/`).
- Two tests in `tests/test_res_benchmark.py` (`test_config_allows_num_actions_3`,
  `test_confusion_channel_rows_sum_to_one`) need later A=3 `ESLConfig` /
  observation-noise APIs. They are not required to regenerate the matrices.
