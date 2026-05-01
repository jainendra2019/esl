# Figure 4 RM Dynamics Report

## Diminishing-Step Diagnostic for Asymptotic Dynamics

Main Figures 1--3 evaluate finite-horizon adaptation using constant-step updates. This appendix diagnostic instead runs ESL in the Robbins--Monro regime assumed by Theorem 1.

## Protocol
- Method: ESL only; games: `ipd, stag_hunt, matching_pennies`; seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`; horizon: `3000`.
- Environment: same type-shifting protocol as Figure 3.
- Fixed configuration: `c=0.5`, `eta_reg=0.001`, frozen batch size `M=50`.
- Step size: `gamma_m = c * (m + 1)^(-0.9)`.
- The schedule analytically satisfies the Robbins--Monro conditions: `sum gamma_m = infinity` and `sum gamma_m^2 < infinity`.
- Nonzero L2 regularization instantiates the inward-drift condition used in Lemma 1.
- Larger frozen batches reduce gradient noise and better approximate the slow-timescale dynamics assumed in the analysis.

While the Robbins--Monro schedule guarantees the relevant asymptotic step-size conditions, the finite-horizon results here should be interpreted as diagnostics of trajectory behavior rather than convergence itself.

## Gate Checks
- Overall gate: PASS.
- `all_finite`: PASS.
- `max_norm_finite`: PASS.
- `norm_growth_bounded`: PASS.
- `late_norm_not_diverging`: PASS.
- `movement_decreases_or_controlled`: PASS.
- `effective_step_decreases_or_controlled`: PASS.
- `usage_jsd_stable`: PASS.
- `action_jsd_stable`: PASS.
- `gamma_decreases`: PASS.
- `gamma_cumulative_diagnostics_finite`: PASS.

## Aggregate Numbers
- `norm_growth_ratio`: mean `0.9169`; 95% CI [`0.9129`, `0.9209`].
- `late_movement`: mean `0.0023`; 95% CI [`0.0022`, `0.0024`].
- `movement_ratio`: mean `0.1243`; 95% CI [`0.1160`, `0.1326`].
- `late_effective_step`: mean `0.0023`; 95% CI [`0.0022`, `0.0024`].
- `effective_step_ratio`: mean `0.1246`; 95% CI [`0.1163`, `0.1330`].
- `late_usage_jsd_mean`: mean `0.0491`; 95% CI [`0.0409`, `0.0573`].
- `late_action_jsd_mean`: mean `0.0055`; 95% CI [`0.0031`, `0.0079`].

## Interpretation Constraint
This experiment is not meant to beat constant-step ESL on PSR. It is a theorem-alignment diagnostic checking bounded norms, decreasing update magnitudes, and stable late-window empirical distributions under Robbins--Monro conditions.
Do not claim convergence to ICT sets. These results are consistent with the dynamical structure implied by the differential inclusion, but they are not an empirical proof of convergence.

## Constant-Step vs Robbins--Monro
The Robbins--Monro diagnostic emphasizes slower movement and stabilizing updates, while the constant-step Figure 3 setting permits persistent movement for adaptive tracking. The two settings are complementary regimes: asymptotic convergence diagnostics versus finite-horizon adaptive tracking.

## Caption
\caption{\textbf{Diminishing-step diagnostic for ESL dynamics.} (a) Prototype norms remain bounded under Robbins--Monro updates. (b) Prototype update magnitudes decrease over slow time. (c) Belief entropy stabilizes into a recurrent regime rather than collapsing by construction. (d) Late-window Jensen--Shannon divergence for prototype usage and induced action distributions is small. This diagnostic uses diminishing step sizes, nonzero L2 regularization, and larger frozen batches to test trajectory signatures consistent with the dynamical structure implied by the differential inclusion; it is not a claim of ICT convergence and does not replace the constant-step adaptation results.}
