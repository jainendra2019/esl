# Milestone 3 — Core ESL experimental validation (paper-facing)

| Field | Value |
|-------|-------|
| Status | PASS |
| Scope | Recovery primary; sparse observability; Q sweep; freeze-θ diagnostic |
| Freeze-θ diagnostic OK | True |

## Interpretation (short)

- **Sparse sweep** (`sparse_pobs_summary.csv`): ESL **MCE** and belief argmax vs **p_obs**, with **transductive** K-means and FCM on the same logged stream (batch static baselines). Lower **p_obs** increases gradient noise and typically hurts recovery vs full observability.
- **Q sweep** (`q_recovery_summary.csv`): Varying **prototype_update_every** trades off how often slow θ updates occur; extremes can under- or over-fit relative to the flagship default.
- **Freeze-θ diagnostic**: `prototype_logits_override` unchanged through rollout when `freeze_prototype_parameters=True` (self-consistency).

## Manuscript bundle paths copied

- `tables/milestone3_sparse_pobs_summary.csv`
- `tables/milestone3_q_recovery_summary.csv`
- `metrics/milestone3_runs_aggregate.csv`
- `figures/milestone3_final_ce_vs_p_obs.png`
- `figures/milestone3_sparse_pobs_mce_belief.png`
- `figures/milestone3_q_vs_mce.png`
- `metrics/milestone3_freeze_theta_diagnostic.json`
- `manifests/milestone3_example_run_manifest.json`
- `configs/milestone3_example_config.json`

## Paper fidelity

- NeurIPS-style figures use `publication_figure_dpi()` (set `ESL_PUBLICATION_DPI=300` for print export).
- No M-FOS/MBOM; batch baselines restricted to **K-means + FCM** for Milestone 3 tables/figures.

## Next

Await confirmation before further external baselines or longer flagship horizons.
