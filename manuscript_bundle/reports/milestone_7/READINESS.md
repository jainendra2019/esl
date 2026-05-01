# Milestone 7 — Submission readiness

- **Overall:** READY
- **Matching pennies in main text:** True
- **Heuristic detail:** {"n_rows": 10, "eligible": true, "mean": 0.421, "std": 0.021429470880593906, "cv": 0.05090135601091189, "range": 0.0615, "reason": "ok"}

## Statistical audit

- Spot checks flag very wide CIs relative to the mean on selected aggregate columns.
- No wide-CI warnings on spot-checked aggregates.

## Before submission

- Re-run without smoke using frozen JSON horizons.
- If PPO is required in main text, set ``skip_ppo: false`` in ``milestone7_frozen.json`` and ensure ``third_party/PPO-PyTorch`` is present.
- Replace placeholder interpretation with numbers pulled from staged CSVs after your final run.
