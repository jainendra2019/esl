# Main Performance Grid Correctness Audit

## Decision

**Fix protocol first. Do not proceed to a full paper run from the current `main_performance_grid.png`.**

The 3 x 3 layout is useful as a presentation target, and the smoke pipeline successfully exercises file generation, CI aggregation, plotting, and provenance. The scientific comparison is not yet valid: ESL and the external baselines are not all evaluated in the same interaction protocol, `fixed_types` does not make ESL act as a focal learner, and the current ESL payoff is a row-position average rather than a strict focal-agent payoff.

## Smoke Table Audit

All entries use `n=2` seeds and 95% t intervals. Higher payoff is better for all three games. The payoff scale is focal row payoff: IPD ranges from 0 to 5 per interaction, Stag Hunt from 0 to 4, and Matching Pennies from -1 to 1.

| Task | Regime | Method | Mean | 95% CI | Scale check |
|---|---|---|---:|---:|---|
| IPD | Adaptive agents | ESL | 1.125 | [1.125, 1.125] | OK, higher better |
| IPD | Adaptive agents | Independent PPO | 1.500 | [-3.794, 6.794] | OK, but CI impossible for payoff support due n=2 t interval |
| IPD | Adaptive agents | M-FOS | 0.958 | [-4.865, 6.782] | OK, reduced adapter |
| IPD | Adaptive agents | MBOM | 2.000 | [-6.471, 10.471] | OK, reduced adapter |
| IPD | Adaptive agents | Simple Opponent Model | 1.667 | [-2.569, 5.902] | OK, reduced adapter |
| IPD | Belief-conditioned agents | ESL | 2.000 | [-1.177, 5.177] | OK, but not strict focal-only average |
| IPD | Belief-conditioned agents | Independent PPO | 1.708 | [0.120, 3.297] | OK |
| IPD | Belief-conditioned agents | M-FOS | 0.375 | [-0.154, 0.904] | OK, reduced adapter |
| IPD | Belief-conditioned agents | MBOM | 4.500 | [2.382, 6.618] | Suspicious: near temptation payoff via synthetic opponent |
| IPD | Belief-conditioned agents | Simple Opponent Model | 4.167 | [-0.069, 8.402] | Suspicious: synthetic opponent exploit |
| IPD | Fixed types | ESL | 2.292 | [-1.414, 5.998] | OK scale, but ESL is not acting |
| IPD | Fixed types | Independent PPO | 2.292 | [1.762, 2.821] | OK |
| IPD | Fixed types | M-FOS | 1.875 | [0.287, 3.463] | OK, reduced adapter |
| IPD | Fixed types | MBOM | 2.125 | [1.596, 2.654] | OK, reduced adapter |
| IPD | Fixed types | Simple Opponent Model | 2.083 | [1.024, 3.142] | OK, reduced adapter |
| Matching Pennies | Adaptive agents | ESL | -0.250 | [-2.368, 1.868] | OK scale; wide CI |
| Matching Pennies | Adaptive agents | Independent PPO | 0.083 | [-0.976, 1.142] | OK scale; CI exceeds support |
| Matching Pennies | Adaptive agents | M-FOS | 0.000 | [0.000, 0.000] | OK scale; reduced adapter is effectively random |
| Matching Pennies | Adaptive agents | MBOM | -0.167 | [-2.284, 1.951] | OK scale; CI exceeds support |
| Matching Pennies | Adaptive agents | Simple Opponent Model | 0.167 | [0.167, 0.167] | OK scale; tiny sample |
| Matching Pennies | Belief-conditioned agents | ESL | -0.167 | [-1.226, 0.892] | OK scale |
| Matching Pennies | Belief-conditioned agents | Independent PPO | -0.417 | [-5.711, 4.878] | OK scale; CI exceeds support |
| Matching Pennies | Belief-conditioned agents | M-FOS | 0.167 | [-1.951, 2.284] | OK scale; reduced adapter |
| Matching Pennies | Belief-conditioned agents | MBOM | -0.167 | [-0.167, -0.167] | OK scale; reduced adapter |
| Matching Pennies | Belief-conditioned agents | Simple Opponent Model | 0.250 | [-0.809, 1.309] | OK scale; reduced adapter |
| Matching Pennies | Fixed types | ESL | -0.042 | [-0.571, 0.488] | OK scale, but ESL is not acting |
| Matching Pennies | Fixed types | Independent PPO | -0.167 | [-2.284, 1.951] | OK scale; CI exceeds support |
| Matching Pennies | Fixed types | M-FOS | 0.000 | [-2.118, 2.118] | OK scale; reduced adapter |
| Matching Pennies | Fixed types | MBOM | -0.333 | [-2.451, 1.784] | OK scale; reduced adapter |
| Matching Pennies | Fixed types | Simple Opponent Model | -0.083 | [-1.142, 0.976] | OK scale; reduced adapter |
| Stag Hunt | Adaptive agents | ESL | 1.812 | [0.489, 3.136] | OK scale |
| Stag Hunt | Adaptive agents | Independent PPO | 1.792 | [-2.973, 6.556] | OK scale; CI exceeds support |
| Stag Hunt | Adaptive agents | M-FOS | 2.333 | [1.274, 3.392] | OK, reduced adapter |
| Stag Hunt | Adaptive agents | MBOM | 2.167 | [1.108, 3.226] | OK, reduced adapter |
| Stag Hunt | Adaptive agents | Simple Opponent Model | 2.083 | [-0.034, 4.201] | OK, reduced adapter |
| Stag Hunt | Belief-conditioned agents | ESL | 1.792 | [1.792, 1.792] | OK scale, but not strict focal-only average |
| Stag Hunt | Belief-conditioned agents | Independent PPO | 2.250 | [0.132, 4.368] | OK |
| Stag Hunt | Belief-conditioned agents | M-FOS | 0.750 | [-0.309, 1.809] | OK, reduced adapter |
| Stag Hunt | Belief-conditioned agents | MBOM | 2.833 | [-1.402, 7.069] | OK scale; reduced adapter |
| Stag Hunt | Belief-conditioned agents | Simple Opponent Model | 2.417 | [1.358, 3.476] | OK, reduced adapter |
| Stag Hunt | Fixed types | ESL | 2.229 | [-0.683, 5.141] | OK scale, but ESL is not acting |
| Stag Hunt | Fixed types | Independent PPO | 2.125 | [0.537, 3.713] | OK |
| Stag Hunt | Fixed types | M-FOS | 2.208 | [1.679, 2.738] | OK, reduced adapter |
| Stag Hunt | Fixed types | MBOM | 2.250 | [2.250, 2.250] | OK, reduced adapter |
| Stag Hunt | Fixed types | Simple Opponent Model | 2.333 | [1.274, 3.392] | OK, reduced adapter |

## Regime Correctness

| Regime | Classification | Finding |
|---|---|---|
| Fixed types | **SUSPECT** | The runner maps this to `mode="recovery"`. In recovery mode, actions come from fixed hidden policies, so ESL learns prototypes but does not act as a focal learner. The reported payoff is therefore not ESL policy performance. |
| Adaptive agents | **SUSPECT** | ESL adaptation is active (`mode="adaptation"`, no `adaptation_esl_agent_indices`, so all agents use ESL best response), but the metric averages row-player rewards across sampled population interactions rather than a fixed focal agent. |
| Belief-conditioned agents | **SUSPECT** | Focal ESL activation is configured (`adaptation_esl_agent_indices=[0]`), but the payoff aggregation still averages all row-player interactions, not only focal agent 0 interactions. |

Additional protocol checks:

- Focal payoff is used for the plotted smoke table, not social payoff. However, for ESL it is the mean of `r_i` over all sampled row positions, not guaranteed focal-agent payoff.
- Seeds and horizons are matched at the file level: seeds `{0, 1}`, horizon `12`, two ordered pairs per round, 24 ESL interaction events per run.
- Opponents and schedules are not matched across methods. ESL uses the trainer population protocol; external baselines use adapter-local synthetic opponent processes.
- ESL action distributions are not auditable from current outputs because `log_interaction_observations=false`; `reward_trajectory.csv` lacks actions.
- ESL-K1, no-belief, random, and fixed-policy baselines are not present in this grid, so they cannot explain the smoke gaps.

## Cell Validity

Every task x regime cell is currently **SUSPECT** for scientific comparison, even when payoff scale is correct.

| Task | Fixed types | Adaptive agents | Belief-conditioned agents |
|---|---|---|---|
| IPD | SUSPECT: ESL not acting; baselines use separate synthetic opponent loop | SUSPECT: all ESL agents adapt, but payoff is not focal-only and opponents differ by method | SUSPECT: focal mask exists, but metric is not focal-only; reduced MBOM/SOM exploit synthetic opponents |
| Stag Hunt | SUSPECT: ESL not acting; baselines use separate synthetic opponent loop | SUSPECT: all ESL agents adapt, but payoff is not focal-only and opponents differ by method | SUSPECT: focal mask exists, but metric is not focal-only and baselines are reduced |
| Matching Pennies | SUSPECT: payoff sign/support now correct, but ESL not acting | SUSPECT: zero-sum scale is correct, but payoff protocol differs across methods | SUSPECT: focal mask exists, but metric and opponent schedule are not matched |

## Method Readiness

| Method | Classification | Official code use | Reason |
|---|---|---|---|
| ESL | **NOT_READY** for this main grid | ESL core trainer is used, but grid protocol is wrong | Fixed-types cell does not evaluate an acting ESL focal agent; payoff is not strict focal-agent payoff; actions are not logged for audit. |
| Independent PPO | **APPENDIX_ONLY** until protocol is fixed | Official `third_party/PPO-PyTorch/PPO.py` is imported and used | The policy core is official, but the environment loop is a custom adapter and does not share ESL's opponent schedule/population. |
| M-FOS | **NOT_READY** | Official source is pinned but not executed | Current adapter is a reduced cooperation-shaping heuristic, not M-FOS meta-policy training. |
| MBOM | **NOT_READY** | Official source is pinned but not executed | Current adapter is a reduced opponent-frequency model; it does not run MBOM recursive imagination or the official environment-model stack. |
| Simple Opponent Model | **NOT_READY** | Official source is pinned but not executed | Current adapter is a reduced conditional opponent-action model; official Lua/Torch DRON code is not run. |

## ESL Underperformance Diagnosis

The smoke underperformance should not be interpreted as a scientific result.

Likely causes:

- **Protocol mismatch:** external baselines are not playing the same population schedule or opponent process as ESL.
- **Fixed-types bug for performance:** `fixed_types` currently uses recovery mode, so ESL does not act.
- **Metric mismatch:** ESL payoff is not strict focal-agent payoff, while external adapters are single-row-player loops.
- **Smoke horizon too short:** each ESL run has only 12 rounds and 24 interactions. ESL summaries show belief entropy remains near `0.693`, prototype gaps remain near zero, and MCE remains around `0.69-0.85`; this indicates prototypes have not separated.
- **Reduced adapters can exploit adapter-local opponents:** in IPD belief-conditioned smoke, reduced MBOM and SOM repeatedly defect against mostly cooperating synthetic opponents, yielding payoffs around `4.5` and `4.17`. That is an adapter artifact, not evidence against ESL.

Not supported by current artifacts:

- No ESL focal action distribution is logged.
- No ESL-K1/no-belief/random/fixed controls are included.
- No belief-conditioned comparison can be validated against matched observed action logs.

## Recommendation Before Full Run

**Do not run the full paper sweep yet. Fix protocol first.**

Required fixes:

1. Define one shared evaluation environment for all methods: same task, same seed, same horizon, same opponent population, same pair schedule, same observability.
2. Make every method produce a focal-agent payoff for the same focal agent, preferably agent `0`.
3. For `fixed_types`, evaluate an acting focal learner against fixed opponents, not ESL recovery-mode hidden-policy payoff.
4. Log focal actions and opponent actions for ESL (`interaction_observations` or a dedicated focal trajectory) so action distributions can be audited.
5. Add at least random/fixed, ESL-K1, and no-belief controls to the smoke audit before interpreting underperformance.
6. Keep M-FOS, MBOM, and Simple Opponent Model out of the main paper grid until their official training logic is wrapped; otherwise rename the reduced adapters as appendix diagnostics.

Keep the 3 x 3 layout, but treat the current figure as a smoke-layout artifact only. The next milestone should be a protocol repair milestone, not tuning and not a full run.
