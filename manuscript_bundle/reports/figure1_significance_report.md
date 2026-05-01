# Figure 1 Paired Significance

## Protocol
- Source: `runs/figure1_dynamic_adaptation/figure1_dynamic_run_summaries.csv`.
- Comparison: ESL vs Fixed-K.
- Metric: PSR.
- Unit of replication: matched seed after averaging PSR across games; timesteps and post-switch rows are not treated as independent samples.
- Test: paired two-sided t-test implemented as a one-sample t-test on paired differences `d_s = PSR_Fixed-K(s) - PSR_ESL(s)`.

## Results
- n pairs: `10`.
- mean ESL PSR: `96.7016`.
- mean Fixed-K PSR: `131.4683`.
- mean paired reduction: `34.7667`.
- 95% CI: [`27.8776`, `41.6557`].
- t-statistic: `11.4163`.
- p-value: `1.17613e-06`.
- Cohen's dz: `3.6101` (large; thresholds: small ~0.2, medium ~0.5, large ~0.8).

## Manuscript Sentence
Across matched seeds, ESL significantly reduces PSR relative to the strongest baseline, Fixed-K (paired t-test, mean paired reduction = 34.77, 95% CI [27.88, 41.66], p < 0.001).
