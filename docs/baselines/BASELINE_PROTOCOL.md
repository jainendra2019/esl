# Offline baselines protocol

## Fairness (same stream, no sequential learning)

**All baselines operate on the same observation stream (including the same observability mask `w`) but do not perform sequential belief updates or use interaction feedback during fitting; ESL is the only method that couples beliefs, observations, and prototype learning online.**

Rows with **`w ≤ 0`** are excluded from baseline statistics and likelihoods (same convention as ESL’s Bayes path, which skips updates when unobserved).

## Data

- **`interaction_observations.csv`**: columns `round`, `i`, `j`, `w`, `s`, `a_i`, `a_j`. Observer is `i`, target is `j`; in the trainer, `s = a_j` (signal equals column player action).
- **`config.json`**: `num_agents`, `num_prototypes`, optional `force_agent_true_types`.
- **Schema**: `observation_manifest.json` records `schema_version`, `sha256`, `n_rows`.

## Features (K-means / FCM)

Per agent \(j\) (as **target** in rows with `w>0`), Laplace-smoothed

\[
\phi_j = \big[\mathbb{P}(a_j{=}C \mid a_i{=}C),\;\mathbb{P}(a_j{=}C \mid a_i{=}D)\big],
\]

with action 0 = Cooperate. Default smoothing \(\alpha=1\) per bin (see `conditional_coop_features`).

## EM (primary vs ablation)

- **Conditional mixture (primary):** latent type per agent; emissions \(\pi_k^{(C)}\), \(\pi_k^{(D)}\) = P(cooperate \(\mid\) opponent row action C vs D in the **same** interaction row). Agent-level EM with multiple random restarts.
- **Marginal mixture (ablation):** single \(\pi_k\) = P(cooperate) per type, ignoring \(a_i\).

**MCE mapping:** learned rows are \((K,2)\) logits matching `softmax` vs `games.true_type_distributions`: conditional EM uses the **empirical** frequency of \(a_i{=}C\) in **`w>0`** rows to form a **marginal** cooperate probability \(\bar{\pi}_k = q\,\pi_k^{(C)} + (1-q)\,\pi_k^{(D)}\), then \(\log \bar{\pi}_k\), \(\log(1-\bar{\pi}_k)\).

## Oracle

Uses **`force_agent_true_types`** (or cyclic default) to pool labeled targets and form empirical \((p(C), p(D))\) per true type. **Not** a fair competitor—reports an approximate **upper bound** on static MCE achievable with perfect type labels on the same log.

## Outputs

**Protocol T (transductive full log):**  
`python -m esl.baselines run-all --run-dir RUN` → **`RUN/baselines/baselines_summary.json`**. Offline methods refit on **all** `w>0` rows; ESL bar uses **final θ** after the full run. This is a useful **diagnostic** (“is \(\phi\) separable if we pool the entire log?”) but **not** a fair headline comparison against online ESL, because batch methods see the **entire future** of the interaction stream.

**Protocol P (prefix-matched budget):**  
`python -m esl.baselines run-all --run-dir RUN --prefix-max-round R` → **`RUN/baselines/baselines_summary_prefix_round_R.json`**. Baselines use only rows with **`round ≤ R`**. ESL’s MCE uses **θ at the end of round \(R\)** from `prototype_trajectory.csv` (same information horizon). **Prefer this for main-paper baseline bars** when the claim is online learning under sequential data.

Plots default to **Oracle omitted** (often visually redundant when EM \(\approx 0\)); use `python -m esl.baselines plot --include-oracle` if needed.

**Interpretation (short runs):** On **small \(T\)**, batch sufficient statistics can still dominate online θ. Use **converged** horizons or learning curves. JSON includes **`esl_mce_recomputed_from_trajectory`** and **`esl_mce_summary_matches_trajectory`** (full-log protocol) to validate the trainer’s `final_mce` against `prototype_trajectory.csv`.

## Metrics

**MCE** = matched cross-entropy via Hungarian alignment (`esl.metrics.mce_value`).

## Suggested figure roles (main text vs appendix)

| Role | Figure / artifact | Command or module |
|------|-------------------|-------------------|
| **Main** | Belief argmax accuracy vs time (ESL-only) | `python -m esl.plot_neurips belief-acc RUN_DIR -o fig.png` |
| **Main** | Sparse \(p_{\mathrm{obs}}\): ESL MCE vs batch EM/K-means + belief panel | `python -m esl.experiments sparse-pobs-sweep --out-root runs/sparse_pobs_sweep` → `sparse_pobs_mce_belief.png` |
| **Main** | Prefix sample-efficiency curves | `python -m esl.experiments prefix-learning-curve --run-dir RUN` |
| **Main** | Ablation: full vs frozen θ vs frozen beliefs | `python -m esl.experiments ablation-ladder --out-root runs/ablation_ladder` |
| **Main** | Adaptation payoff vs \(\lambda\) (ESL-only caption) | `python -m esl.experiments adaptation-payoff-sweep` |
| **Appendix** | Transductive MCE bar (full log) | `esl.baselines plot` on `baselines_summary.json` |
| **Appendix** | Prefix-matched MCE bar | `esl.baselines plot --summary baselines_summary_prefix_round_R.json` |

Use **`--no-baselines`** on the sparse sweep only for fast debugging (skips `interaction_observations` + offline fits).
