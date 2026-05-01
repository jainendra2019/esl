# Experiment report — claims, figures, and evidence

**Takeaway.** Belief-coupled latent opponent models improve long-run payoffs when feedback about others is limited.

## Figure index

Machine-readable mapping: ``figure_index.json`` (includes source resolution and caption paths).

## Claim → figure → evidence

| Claim | Figure | Evidence (where to look) |
|-------|--------|---------------------------|
| Recovery is stable when actions are only sometimes observed | **Figure 1** (`main/figures/fig1.png`) | Aggregated runs in `main/tables`; CI bands on the figure. |
| Recovery depends on how frequently latent parameters are updated from data | **Figure 2** (`main/figures/fig2.png`) | Same tables; cadence on the horizontal axis. |
| Focal payoffs evolve under repeated play with a mixed partner population | **Figure 3** (`main/figures/fig3.png`) | Time series with cross-run bands; per-run logs under the paper run root. |
| Structured opponent learning compares favorably to clustering-only alternatives | **Figure 4** (`main/figures/fig4.png`) | Bar summary with CIs; companion CSV in `main/tables`. |
| Payoff rises as inferred structure aligns with a fixed two-type reference | **Figure 5** (`main/figures/fig5.png`) | Scatter over seeds/configurations; underlying rows in appendix tables for extended robustness. |

## Appendix (robustness and diagnostics)

- **Figures:** `appendix/figures/` — initialization and latent-cardinality sweeps from the recovery study; supplementary adaptation-stress panels; optional pooled scatter including the oscillatory diagnostic.
- **Tables:** `appendix/tables/` — aggregated robustness tables for adaptation stress and belief ablation.

## Reproducibility

Paper runs are produced by the frozen orchestration config (`esl/experiments/configs/milestone7_frozen.json`) and staged by Milestone 7; Milestone 8 only renames, moves, and rewrites narrative files without re-running experiments.
