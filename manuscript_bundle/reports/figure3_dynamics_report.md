# Figure 3 Dynamics Report

## Protocol
- Method: ESL only; games: `ipd, stag_hunt, matching_pennies`; seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`; horizon: `1000`.
- Environment: same type-shifting protocol as Figures 1-2.
- Panel (c) uses normalized belief entropy by default.

## Metric Definitions
- Prototype norm is the Frobenius norm of the prototype matrix.
- Movement and update norms are slow-update prototype displacements.
- The invariant-regime proxy uses adjacent-window Jensen-Shannon divergence in nats.

## Gate Checks
- bounded prototype dynamics: PASS; mean norm growth ratio `1.0130`.
- usage invariant-regime proxy JSD < 0.25: PASS; mean `0.0098`.
- action invariant-regime proxy JSD < 0.25: PASS; mean `0.0100`.
- JSD gates are reporting diagnostics, not hard scientific claims; higher values indicate less stable empirical distributions in that setting.

## Aggregate Numbers
- `norm_growth_ratio`: mean `1.0130`; 95% CI [`0.8989`, `1.1271`].
- `late_movement`: mean `0.3539`; 95% CI [`0.3415`, `0.3662`].
- `late_usage_jsd_mean`: mean `0.0098`; 95% CI [`0.0066`, `0.0131`].
- `late_action_jsd_mean`: mean `0.0100`; 95% CI [`0.0060`, `0.0140`].
- `update_norm_ratio`: mean `1.1801`; 95% CI [`1.1197`, `1.2405`].

## Conservative Interpretation
These diagnostics do not prove convergence to an ICT set. Rather, they test whether the empirical ESL trajectories exhibit signatures predicted by the theory: bounded prototype dynamics, non-divergent slow updates, and stabilization of windowed belief/action distributions. The results are therefore interpreted as consistency evidence for the differential-inclusion characterization, not as a separate theoretical guarantee.

This is not a proof of ICT convergence; it is a diagnostic consistent with the differential-inclusion characterization.

## Caption
\caption{\textbf{Empirical signatures of ESL's closed-loop learning dynamics.} (a) Prototype parameters remain bounded over slow updates. (b) Prototype movement decreases but need not immediately collapse to zero, consistent with the theory allowing both fixed points and more general internally chain transitive regimes. (c) Belief statistics stabilize over training, indicating recurrent epistemic regimes. (d) Adjacent-window Jensen--Shannon divergence over the final training phase is small for both prototype usage and induced action distributions, providing an empirical proxy for invariant-regime averaging. These diagnostics do not prove ICT convergence, but show behavior consistent with the differential-inclusion characterization.}
