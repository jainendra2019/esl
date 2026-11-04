# Experiment configurations

Protocol locks for the accepted twelve-game evaluation. Experiments load
committed JSON from `benchmark/`; they do not resample payoffs.

## Flagship adaptation (headline)

- Games: `coop3_01`–`coop3_04`, `coord3_01`–`coord3_04`, `adv3_01`–`adv3_04`
- Generator seed for the 3×3 suite: `20260728`
- Focal learner plus three opponents, row player
- Actions A=3, true types K=3, learned K=3 unless noted
- Horizon T=1000, evaluation window H=100
- Switch interval 200, switch hazard 0.01
- Seeds 0–9 (n=120 cells/method)
- Methods (flagship): ESL, PPO, Fixed-K Bayesian, SA-FP, FP, SOM, Online EM (840 cells). OMD/MWU is a separate held-out protocol.
- ESL: λ=2, prototype step γ=0.9, batch M=2, p_obs=1, obs_noise=0
- PSR/DPR: inclusive `0 ≤ τ ≤ H`
- LTE: Hungarian alignment of pre-observation beliefs, mean over `τ < H`

Entry point (requires optional PyTorch for PPO):

```bash
python -m esl.res --num-actions 3 --full \
  --method-set flagship-a3 \
  --out-dir outputs/flagship
```

Without PPO (`flagship-a3-no-ppo`, 720 cells):

```bash
python -m esl.res --num-actions 3 --full \
  --method-set flagship-a3-no-ppo \
  --out-dir outputs/flagship_no_ppo
```

The CLI default `--method-set tier1` keeps the historical five-method suite and does **not** match the published seven-method flagship.

## Held-out OMD/MWU

- Same twelve games and T=1000 protocol
- Development seeds 0–2; confirmatory seeds 3–9 (n=84)
- ESL selected λ=10; OMD/MWU selected η=1.0
- Do not mix with the headline λ=2 table

Entry point: `python scripts/run_omd_clean_psr.py`

## Other families

| Family | Entry point |
|---|---|
| Matched ablations | `python -m esl.res.ablation_a3 ablation` |
| α sensitivity | `python -m esl.res.ablation_a3 alpha` |
| Batch size M | `python -m esl.res.ablation_a3 batch` |
| Inverse temperature λ | `python scripts/run_focal_lambda_sensitivity.py` |
| K ∈ {2,3,5} | `python scripts/run_k_sensitivity.py` |
| Opponent count | `python -m esl.res.opponent_count_sensitivity` |
| Calibrated shifts | `python scripts/run_controlled_behavioral_shift.py` |
| Visibility / noise | `python -m esl.res --num-actions 3 --full --method-set flagship-a3 --p-obs …` / `--obs-noise …` |
| Diminishing-step RM | `python -m esl.experiments.figure4_rm_dynamics` |
