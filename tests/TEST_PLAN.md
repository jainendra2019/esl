# ESL MVP — verification vs validation test plan

**Verification** = did we implement the algorithm correctly?  
**Validation** = does the system behave as ESL / PRD intend?

## How to run

| Command | What runs |
|---------|-----------|
| `pytest -q` | Default: all tests **except** `@pytest.mark.slow` (fast dev loop). |
| `pytest -q --runslow` | Full suite, including slow (e.g. two-type separation VAL9). |
| `pytest -q -m slow` | Slow tests only (heavy scientific checks). |
| `pytest -q -m validation` | Validation tests **excluding** `@pytest.mark.slow`. |
| `pytest -q --runslow -m validation` | All validation tests, **including** slow (e.g. VAL9 two-type separation). |

## 1. Verification (`@pytest.mark.verification`)

| ID | Test function | Intent |
|----|----------------|--------|
| V1 | `test_v1_softmax_sums_stable` | Softmax sums to 1, in [0,1], finite for extreme logits |
| V2 | `test_v2_log_likelihood_finite_both_actions` | Clamped log L finite for extreme logits, both actions |
| V3 | `test_v3_gradient_matches_finite_differences_specific` | Analytic grad vs FD at θ=[0.7,-0.2], s=0 |
| V4 | `test_v4_bayes_preserves_simplex` | Posterior on simplex, nonnegative |
| V5 | `test_v5_floor_projection_enforces_delta` | Iterated floor: min ≥ δ − ε_floor, sum 1 |
| V6 | `test_v6_no_belief_update_when_unobserved` | W=0 ⇒ belief unchanged |
| V7 | `test_v7_batch_stores_b_ij_t_before_bayes_not_b_ij_t_plus_1` | Batch = $b_{i\to j,t}$ (pre-Bayes), not $b_{t+1}$; gradient uses snapshot |
| V8 | `test_v8_batch_averaging_duplicate_observations` | Duplicate batch ⇒ same θ update as single (mean) |
| V9 | `test_v9_belief_likelihood_cooperate_shifts_posterior` | Observe C ⇒ mass on C-like prototype up |
| V10 | `test_v10_single_prototype_step_increases_likelihood_of_signal` | SGD step increases L(signal) |
| V11 | `test_v11_deterministic_seed_reproducibility` | Same seed ⇒ same logits + summary |
| V12 | `test_v12_batch_appends_w_zero_and_dilutes_mean_gradient` | `w=0` rows add no grad but count in mean denominator (**ALGORITHM.md**) |
| V13 | `test_v13_weighted_gradient_equals_w_times_b_times_es_minus_p` | Batch grad = `w · b_snap[k] · (e_s − p_k)` per prototype |
| V14 | `test_v14_true_type_distributions_cycles_behaviors_when_k_gt_2` | `true_type_distributions(K)` cycles AC/AD when K>2 |
| V15 | `test_v15_recovery_mode_never_calls_act_agent` | Recovery path never invokes `act_agent` |
| V16 | `test_v16_adaptation_mode_calls_act_agent` | Adaptation path invokes `act_agent` each pair play |
| HT1 | `test_hand_trace_*` in `test_hand_trace_always_defect.py` | Defect + **Cooperate mirror**: fixed θ; beliefs vs isolated Bayes; entropy ↓; P(C) rises under s=0 batch |
| — | `python -m esl.hand_trace` / `--cooperate` | CSV + prototype update log for Defect or Cooperate stream |
| — | **`ALGORITHM.md`** (repo root) | Implementation-faithful pseudocode; keep in sync with `trainer.py` |

## 2. Validation (`@pytest.mark.validation`)

| ID | Test function | Intent |
|----|----------------|--------|
| VAL1 | `test_val1_symmetric_init_no_separation` | Identical init ⇒ prototypes stay close |
| VAL2 | `test_val2_asymmetry_induces_specialization` | Noise init ⇒ separation + lower CE |
| VAL3 | `test_val3_two_type_recovery_short_horizon_multi_seed` | Same as above on **seeds 12, 42, 99** (parametrize) |
| VAL4 | `test_val4_sparse_observability_degrades_recovery` | Lower p_obs ⇒ worse CE (aggregated) |
| VAL5 | `test_val5_two_timescale_prototype_schedule` | Prototype updates only on schedule + count |
| VAL6 | `test_val6_batch_log_likelihood_trend` | Later rounds: higher mean batch LL |
| VAL7 | `test_val7_adaptation_learning_beats_frozen_baseline` | ESL learning > learning_frozen payoff |
| VAL8 | `test_val8_entropy_falls_before_payoff_rises` | Early entropy > late; payoff non-decreasing trend |
| VAL9 | `test_two_type_*` in `test_two_type_separation.py` | Main (N=6, K=2): matched purity thresholds; symmetric θ=0 baseline non-separated; freeze-prototype logits fixed (**`@pytest.mark.slow`**) |
| DASH | `test_dashboard_summary_fields` | Summary JSON exposes required dashboard keys |
| — | `python -m esl.experiment_two_type_separation --out <dir>` | **Required `--out`**. Runs `main` / `symmetric` / `freeze_proto_baseline`; manifest + `figure_*.png` (unless `--no-plots`) |

## 3. Edge cases (`@pytest.mark.edge`)

| ID | Test function | Intent |
|----|----------------|--------|
| E1 | `test_e1_all_agents_same_hidden_type` | K=1 or uniform types: stable, no crash |
| E2 | `test_e2_overparameterized_k_four` | K=4, two behaviors: no crash, metrics finite |
| E3 | `test_e3_extreme_sparsity_runs` | p_obs=0.01: finite, no NaNs |
| E4 | `test_e4_degenerate_extreme_logits` | Huge logits: softmax + Bayes stable |

## 4. Minimum bar (must pass before trusting MVP)

Gradient FD, Bayes simplex, W=0 no-update, pre-Bayes snapshot, symmetric init, asymmetric separation, short recovery, sparse degradation, reproducibility — all covered above.
