# Main Performance Grid Report

## Decision

**Full corrected shared-protocol run completed for ESL, ESL ablations, PPO, and SOM.**

## Shared Protocol

- Focal learner: agent `0`.
- Reported metric: `focal_mean_payoff_per_round` only.
- Every included method uses `run_shared_focal_protocol` in `esl/experiments/main_performance_grid.py`.
- Opponent IDs/types and random uniforms are generated once per task/regime/seed and reused across methods.
- Baseline adapters' separate synthetic `train(...)` loops are not used for this corrected figure.
- Full seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`; horizon: `500`.

## Method Validity

| Method | Status | Fidelity | Notes |
|---|---|---|---|
| ESL | VALID_SHARED_PROTOCOL | official | Agent 0 acts in every regime, including fixed types. |
| ESL K=1 | VALID_SHARED_PROTOCOL_ABLATION | ablation | Single-prototype ESL ablation under the same focal-agent loop. |
| ESL without belief updates | VALID_SHARED_PROTOCOL_ABLATION | ablation | Beliefs stay uniform while prototype updates continue under the same focal-agent loop. |
| PPO | VALID_SHARED_PROTOCOL_APPENDIX_ONLY | official PPO core, custom shared wrapper | Uses official `PPO.py` policy/update inside the shared focal loop. |
| SOM | VALID_SHARED_PROTOCOL_APPENDIX_ONLY | reduced | Faithful simple conditional opponent model inside shared loop; official Lua/Torch DRON is not executed. |
| M-FOS | SKIPPED | skipped | SKIPPED: official M-FOS meta-policy logic is not wrapped in the shared focal-agent protocol. |
| MBOM | SKIPPED | skipped | SKIPPED: official MBOM recursive imagination/environment-model stack is not wrapped in the shared focal-agent protocol. |

## Full Payoff Summary

| Task | Regime | Method | n | Mean | 95% CI |
|---|---|---|---:|---:|---:|
| ipd | adaptive_agents | ESL | 10 | 1.760 | [1.725, 1.796] |
| ipd | adaptive_agents | ESL K=1 | 10 | 1.753 | [1.719, 1.786] |
| ipd | adaptive_agents | ESL without belief updates | 10 | 1.760 | [1.725, 1.796] |
| ipd | adaptive_agents | PPO | 10 | 1.205 | [1.106, 1.305] |
| ipd | adaptive_agents | SOM | 10 | 1.721 | [1.689, 1.752] |
| ipd | belief_conditioned_agents | ESL | 10 | 1.369 | [1.318, 1.419] |
| ipd | belief_conditioned_agents | ESL K=1 | 10 | 1.367 | [1.317, 1.417] |
| ipd | belief_conditioned_agents | ESL without belief updates | 10 | 1.369 | [1.318, 1.419] |
| ipd | belief_conditioned_agents | PPO | 10 | 0.934 | [0.832, 1.035] |
| ipd | belief_conditioned_agents | SOM | 10 | 1.336 | [1.285, 1.387] |
| ipd | fixed_types | ESL | 10 | 2.375 | [2.333, 2.418] |
| ipd | fixed_types | ESL K=1 | 10 | 2.322 | [2.275, 2.368] |
| ipd | fixed_types | ESL without belief updates | 10 | 2.328 | [2.275, 2.380] |
| ipd | fixed_types | PPO | 10 | 2.228 | [2.189, 2.267] |
| ipd | fixed_types | SOM | 10 | 2.360 | [2.325, 2.395] |
| matching_pennies | adaptive_agents | ESL | 10 | 0.009 | [-0.022, 0.039] |
| matching_pennies | adaptive_agents | ESL K=1 | 10 | -0.000 | [-0.033, 0.033] |
| matching_pennies | adaptive_agents | ESL without belief updates | 10 | 0.010 | [-0.022, 0.042] |
| matching_pennies | adaptive_agents | PPO | 10 | -0.004 | [-0.017, 0.009] |
| matching_pennies | adaptive_agents | SOM | 10 | -0.046 | [-0.075, -0.017] |
| matching_pennies | belief_conditioned_agents | ESL | 10 | 0.018 | [-0.022, 0.058] |
| matching_pennies | belief_conditioned_agents | ESL K=1 | 10 | 0.036 | [0.004, 0.067] |
| matching_pennies | belief_conditioned_agents | ESL without belief updates | 10 | 0.018 | [-0.022, 0.059] |
| matching_pennies | belief_conditioned_agents | PPO | 10 | -0.013 | [-0.050, 0.024] |
| matching_pennies | belief_conditioned_agents | SOM | 10 | -0.009 | [-0.046, 0.028] |
| matching_pennies | fixed_types | ESL | 10 | 0.859 | [0.834, 0.884] |
| matching_pennies | fixed_types | ESL K=1 | 10 | 0.064 | [0.022, 0.106] |
| matching_pennies | fixed_types | ESL without belief updates | 10 | 0.041 | [0.012, 0.070] |
| matching_pennies | fixed_types | PPO | 10 | 0.015 | [-0.024, 0.054] |
| matching_pennies | fixed_types | SOM | 10 | 0.782 | [0.721, 0.843] |
| stag_hunt | adaptive_agents | ESL | 10 | 2.047 | [2.030, 2.063] |
| stag_hunt | adaptive_agents | ESL K=1 | 10 | 2.043 | [2.029, 2.057] |
| stag_hunt | adaptive_agents | ESL without belief updates | 10 | 2.047 | [2.030, 2.063] |
| stag_hunt | adaptive_agents | PPO | 10 | 1.974 | [1.886, 2.061] |
| stag_hunt | adaptive_agents | SOM | 10 | 2.015 | [1.998, 2.032] |
| stag_hunt | belief_conditioned_agents | ESL | 10 | 1.987 | [1.980, 1.995] |
| stag_hunt | belief_conditioned_agents | ESL K=1 | 10 | 1.990 | [1.980, 1.999] |
| stag_hunt | belief_conditioned_agents | ESL without belief updates | 10 | 1.987 | [1.980, 1.995] |
| stag_hunt | belief_conditioned_agents | PPO | 10 | 1.852 | [1.735, 1.969] |
| stag_hunt | belief_conditioned_agents | SOM | 10 | 1.963 | [1.953, 1.974] |
| stag_hunt | fixed_types | ESL | 10 | 2.603 | [2.533, 2.674] |
| stag_hunt | fixed_types | ESL K=1 | 10 | 2.272 | [2.240, 2.305] |
| stag_hunt | fixed_types | ESL without belief updates | 10 | 2.264 | [2.239, 2.289] |
| stag_hunt | fixed_types | PPO | 10 | 2.252 | [2.209, 2.295] |
| stag_hunt | fixed_types | SOM | 10 | 2.589 | [2.556, 2.621] |

## Readiness

- ESL average full payoff across cells: `1.4475`.
- ESL K=1 average full payoff across cells: `1.3162`.
- ESL without belief updates average full payoff across cells: `1.3138`.
- PPO average full payoff across cells: `1.1603`.
- SOM average full payoff across cells: `1.4124`.
- ESL beats PPO in `9/9` cells; mean ESL-PPO difference `0.2871`.
- ESL beats SOM in `9/9` cells; mean ESL-SOM difference `0.0351`.
- ESL beats ESL K=1 in `7/9` cells; mean difference `0.1313`.
- ESL beats ESL without belief updates in `3/9` cells; mean difference `0.1337`.
- Figure output: `manuscript_bundle/main/figures/main_performance_grid.png`.
- Caption: 10 seeds, 500 rounds, focal agent payoff, shared opponent schedule.
- ESL payoff is now meaningful as focal-agent payoff under a shared protocol.
- M-FOS and MBOM are intentionally absent from the corrected v1 figure instead of silently using reduced substitutes.

## Recommendation

Use this as the corrected v1 main performance grid with ESL ablations. M-FOS and MBOM remain excluded based on `manuscript_bundle/reports/mfos_mbom_wrapper_feasibility.md`.
