# Main Performance Grid Protocol Fix Report

## Decision

**Protocol is fixed for smoke evaluation, but the 3 x 3 figure is not ready for full paper runs until PPO/SOM appendix status is accepted and M-FOS/MBOM policy is decided.**

## Shared Protocol

- Focal learner: agent `0`.
- Reported metric: `focal_mean_payoff_per_round` only.
- Every included method uses `run_shared_focal_protocol` in `esl/experiments/main_performance_grid.py`.
- Opponent IDs/types and random uniforms are generated once per task/regime/seed and reused across methods.
- Baseline adapters' separate synthetic `train(...)` loops are not used for this corrected figure.
- Smoke seeds: `[0, 1]`; horizon: `12`.

## Method Validity

| Method | Status | Fidelity | Notes |
|---|---|---|---|
| ESL | VALID_SHARED_PROTOCOL | official | Agent 0 acts in every regime, including fixed types. |
| Independent PPO | VALID_SHARED_PROTOCOL_APPENDIX_ONLY | official PPO core, custom shared wrapper | Uses official `PPO.py` policy/update inside the shared focal loop. |
| Simple Opponent Model | VALID_SHARED_PROTOCOL_APPENDIX_ONLY | reduced | Faithful simple conditional opponent model inside shared loop; official Lua/Torch DRON is not executed. |
| M-FOS | SKIPPED | skipped | SKIPPED: official M-FOS meta-policy logic is not wrapped in the shared focal-agent protocol. |
| MBOM | SKIPPED | skipped | SKIPPED: official MBOM recursive imagination/environment-model stack is not wrapped in the shared focal-agent protocol. |

## Smoke Payoff Summary

| Task | Regime | Method | n | Mean | 95% CI |
|---|---|---|---:|---:|---:|
| ipd | adaptive_agents | ESL | 2 | 1.667 | [-2.569, 5.902] |
| ipd | adaptive_agents | Independent PPO | 2 | 0.958 | [-1.689, 3.605] |
| ipd | adaptive_agents | Simple Opponent Model | 2 | 1.750 | [-4.603, 8.103] |
| ipd | belief_conditioned_agents | ESL | 2 | 1.583 | [-3.711, 6.878] |
| ipd | belief_conditioned_agents | Independent PPO | 2 | 0.667 | [-3.569, 4.902] |
| ipd | belief_conditioned_agents | Simple Opponent Model | 2 | 1.625 | [-3.140, 6.390] |
| ipd | fixed_types | ESL | 2 | 2.583 | [-0.593, 5.760] |
| ipd | fixed_types | Independent PPO | 2 | 2.417 | [-1.819, 6.652] |
| ipd | fixed_types | Simple Opponent Model | 2 | 2.667 | [-1.569, 6.902] |
| matching_pennies | adaptive_agents | ESL | 2 | 0.083 | [-3.093, 3.260] |
| matching_pennies | adaptive_agents | Independent PPO | 2 | 0.083 | [-7.329, 7.495] |
| matching_pennies | adaptive_agents | Simple Opponent Model | 2 | 0.333 | [-1.784, 2.451] |
| matching_pennies | belief_conditioned_agents | ESL | 2 | 0.083 | [-5.211, 5.378] |
| matching_pennies | belief_conditioned_agents | Independent PPO | 2 | 0.167 | [-1.951, 2.284] |
| matching_pennies | belief_conditioned_agents | Simple Opponent Model | 2 | 0.167 | [-6.186, 6.520] |
| matching_pennies | fixed_types | ESL | 2 | 0.083 | [-5.211, 5.378] |
| matching_pennies | fixed_types | Independent PPO | 2 | -0.417 | [-3.593, 2.760] |
| matching_pennies | fixed_types | Simple Opponent Model | 2 | -0.167 | [-2.284, 1.951] |
| stag_hunt | adaptive_agents | ESL | 2 | 2.000 | [2.000, 2.000] |
| stag_hunt | adaptive_agents | Independent PPO | 2 | 1.667 | [-4.686, 8.020] |
| stag_hunt | adaptive_agents | Simple Opponent Model | 2 | 1.958 | [0.370, 3.547] |
| stag_hunt | belief_conditioned_agents | ESL | 2 | 1.833 | [-1.343, 5.010] |
| stag_hunt | belief_conditioned_agents | Independent PPO | 2 | 1.667 | [-0.451, 3.784] |
| stag_hunt | belief_conditioned_agents | Simple Opponent Model | 2 | 1.833 | [0.774, 2.892] |
| stag_hunt | fixed_types | ESL | 2 | 2.042 | [1.512, 2.571] |
| stag_hunt | fixed_types | Independent PPO | 2 | 2.333 | [2.333, 2.333] |
| stag_hunt | fixed_types | Simple Opponent Model | 2 | 2.667 | [1.608, 3.726] |

## Readiness

- ESL average smoke payoff across cells: `1.3287`.
- PPO average smoke payoff across cells: `1.0602`.
- Simple Opponent Model average smoke payoff across cells: `1.4259`.
- Figure output: `manuscript_bundle/main/figures/main_performance_grid_protocol_fixed_smoke.png`.
- ESL payoff is now meaningful as focal-agent payoff under a shared protocol.
- M-FOS and MBOM are intentionally absent from the corrected v1 figure instead of silently using reduced substitutes.

## Recommendation

Keep the corrected 3 x 3 layout. Before full runs, decide whether the main figure should contain only ESL + PPO + Simple Opponent Model, or whether full official M-FOS/MBOM wrappers are required. If M-FOS/MBOM are required in main text, do not run full sweeps until those official wrappers exist.
