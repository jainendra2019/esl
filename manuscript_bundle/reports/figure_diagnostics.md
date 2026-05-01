# Figure Diagnostics Report

_Automated validation only — no experiments re-run, no figures edited._

## Pre-flight (Step 0)

- Manuscript bundle exists: **True**
- `third_party/PPO-PyTorch` present: **True** (omit `--skip-ppo` only if true)
- `ESL_PUBLICATION_DPI`: (unset — export 300 before plotting full runs)
- Milestone 4 smoke flag from `statistical_audit.json`: **False** (`null` if audit missing)

## Milestone 4 (Recovery)

### Effect size
- M4 sparse p_obs x=1.0: ESL vs K-means MCE are close (|Δ| < 0.03: ESL 0.0227, KM 0.0008)
- M4 Q recovery x=1: ESL vs K-means MCE are close (|Δ| < 0.03: ESL 0.0216, KM 0.0008)
- M4 Q recovery x=5: ESL vs K-means MCE are close (|Δ| < 0.03: ESL 0.0222, KM 0.0008)
- M4 Q recovery x=15: ESL vs K-means MCE are close (|Δ| < 0.03: ESL 0.0227, KM 0.0008)

### CI quality
- _(no wide-CI flags)_

### Stability / data volume
- _(no flags)_

**Recommendation:** Review flags above before main-text inclusion.

## Milestone 5 (Adaptation)

### Effect size
- M5 ESL − K-means (aggregate mean payoff): +0.0103
- M5 paired ESL−KMeans across seeds: mean Δ=0.0102, 95% CI [0.0075, 0.0130], n=10

### CI quality
- _(no wide-CI flags)_

### Stability / data volume
- _(no flags)_

**Recommendation:** Review flags above before main-text inclusion.

## Milestone 6 (Mechanism)

### Effect size
- _(no notes)_

### CI quality
- _(no wide-CI flags)_

### Stability / data volume
- _(no flags)_

**Recommendation:** No automated issues; still verify narrative fit.

## Final figure decisions (main paper)

| Figure | Decision | Bytes | Notes |
|--------|----------|------:|-------|
| fig1 | **KEEP** | 120070 | M4 sparse observability: present and above size threshold |
| fig2 | **KEEP** | 83256 | M4 Q / batch cadence: present and above size threshold |
| fig3 | **KEEP** | 343722 | M5 adaptation trajectories: present and above size threshold |
| fig4 | **KEEP** | 110973 | M5 method comparison: present and above size threshold |
| fig5 | **KEEP** | 125926 | M6 payoff vs structure (main): present and above size threshold |

### MODIFY vs appendix

- **MOVE** here means: keep for supplementary PDF or appendix until full runs increase resolution / CI tightness.
- **DROP** means: missing asset or unsuitable for camera-ready bundle without re-staging.

## Acceptance checklist (manual)

- [ ] M4–M6 executed **without** `--smoke`
- [ ] `export ESL_PUBLICATION_DPI=300` (or equivalent) before sweeps that plot
- [ ] `manuscript_bundle/main/tables/` aggregates include mean + 95% CI columns
- [ ] Milestone 7 staged `main/figures/` sources, then optional Milestone 8 for `fig1`–`fig5`
- [ ] This report reviewed; STOP before M8/M9 until figure decisions confirmed
