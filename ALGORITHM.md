# ESL core algorithm (implementation-faithful)

This document matches the behavior of `esl.trainer.run_esl` and related modules. For the full product spec, see **`PRD.md`**.

## Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | `num_agents` |
| \(K\) | `num_prototypes` — **learned** prototype count (logits \(\theta \in \mathbb{R}^{K \times |A|}\)) |
| \(\tau_a \in \{0,\ldots,K-1\}\) | Agent \(a\)’s **assigned type index** in code (`true_types`); used for metrics and for choosing a **fixed** hidden policy in recovery mode. |
| True types vs learned prototypes (paper) | **Conceptually separate:** the paper treats true latent types and learned prototypes as distinct objects. This repo aligns them only where needed for metrics (e.g. Hungarian match over \(K\) rows). |
| \(M\) | `prototype_update_every` — slow timescale |
| \(w \in \{0,1\}\) | Observation mask (`sample_observation_mask`; sparse \(w \sim \mathrm{Bernoulli}(p_{\mathrm{obs}})\)) |
| \(b^{\mathrm{snap}}_{i\to j}\) | Belief **before** Bayes on the current signal (what the batch stores). |

### Implementation note: \(K\) larger than base behavioral templates

In the **current recovery-mode implementation**, v1 only registers two **base** hidden policies (Always Cooperate / Always Defect). When **\(K\)** (or an agent type index) exceeds the number of those templates, **fixed** hidden policies are assigned by **cycling** through the available templates (`type_index % n_behavioral`; the same cycling defines rows of `true_type_distributions(K)` for Hungarian costs).

**This is an implementation convenience** for overparameterized and edge tests. **It is not a theoretical claim:** a fuller model could equip each true type with its own policy without cycling. Reviewers should not read the modulo rule as part of the ESL *definition*—only as what this codebase does when \(K>2\) with the built-in policy registry.

## Initialization

```
Initialize θ ∈ ℝ^{K×|A|}     (override, or symmetric / noisy draw)
Initialize B[i,j] ∈ Δ^{K-1} for all ordered pairs i ≠ j (uniform; diagonal unused)
batch ← empty list
m ← 0   // prototype SGD step index

Assign τ_a for each agent a (config override or cyclic 0..K-1)
Build hidden policies π_a from τ_a
    // v1: policy template = f(τ_a mod n_behavioral) — see "Implementation note" above when K>2

If mode == adaptation:
    mark ESL-controlled agents (v1: all) for act_agent
```

## Main loop (rounds t = 0 … T−1)

### 1. Interaction pair

```
If force_ordered_pair set: (i,j) ← that pair
Else: sample uniformly random ordered pair (i,j), i ≠ j
```

### 2. Actions (**split by mode**)

```
If mode == recovery:
    a_i ← hidden_policy[i].act(rng, last_opponent_action=...)
    a_j ← hidden_policy[j].act(...)
Else if mode == adaptation:
    a_i ← act_agent(i, j, …, belief_tensor=B, logits=θ, payoffs, …)   // ESL softmax best response
    a_j ← act_agent(j, i, …)

Update last_opponent_action hooks from (a_i, a_j).
Record PD payoffs (bookkeeping only; not used in belief/prototype gradients).
```

### 3. Observation

```
w ← observation mask (1 under full observability; else Bernoulli(p_obs))
s ← encode(a_j)   // signal = observed action of target j
```

### 4. Fast timescale — beliefs and batch

**If `learning_frozen`:**

- Do **not** update \(B\).
- Do **not** append to `batch`.
- Beliefs stay at their initialization (except any pre-loop edits).

**Else (learning enabled):**

```
b_snap ← copy of B[i,j]

If w > 0:
    For each prototype k:
        L_k ← softmax(θ_k)[s]          // unclipped; used only for Bayes
    tilde_b ← Bayes(B[i,j], L)         // normalized posterior; ε in denominator only if configured
    B[i,j] ← Π_{Δ_K^δ}(tilde_b)        // Euclidean projection onto {b : sum b = 1, b_k ≥ δ}

Append batch record (i, j, s, w, b_snap).
```

**Important:** records are **always** appended when learning is not frozen, **including \(w=0\)**. For \(w=0\), \(B[i,j]\) is unchanged and \(b^{\mathrm{snap}}\) equals the prior belief for that step. In the prototype step, records with `w ≤ 0` contribute **zero** to the accumulated gradient, but they **still count** toward batch length \(|B|\) in the **mean** (see below).

### 5. Slow timescale — prototype SGD

If **not** `learning_frozen` and \((t+1) \bmod M = 0\) and `batch` non-empty:

```
If freeze_prototype_parameters:
    clear batch (no θ update)
Else:
    g_accum ← 0 ∈ ℝ^{K×|A|}
    For each record (i,j,s,w,b_snap) in batch:
        If w ≤ 0:
            skip contribution to g_accum
        Else:
            For each prototype k:
                p_k ← softmax(θ_k)
                g_k ← w · b_snap[k] · (e_s − p_k)   // row k of grad_log_likelihood, scaled
            Add weighted gradient to g_accum (see batch_weighted_prototype_gradient)

    g_mean ← g_accum / max(|batch|, 1)
    γ ← prototype_lr(prototype_step_m)
    θ ← θ + γ · g_mean
    m ← m + 1
    clear batch
```

Clamping of \(\log p\) for telemetry uses `softmax_log_likelihood_clamped` and does **not** define the gradient; gradients use `grad_log_likelihood` (unclamped softmax).

### 6. Logging (each round)

Written to `metrics_trajectory.csv`, `prototype_trajectory.csv`, and (dense) `belief_trajectory.csv`:

- Summary scalars: mean belief entropy over **off-diagonal** \(B[i,j]\), Hungarian **matched** cross-entropy vs `true_type_distributions(K)`, belief argmax accuracy, batch log-likelihood telemetry, prototype step index / update norm, etc.
- Full **per-round** prototype logits and softmax rows.
- Full **per-round** tensor of all pairwise beliefs (off-diagonals); this is heavy but is what the trainer emits today.

Representative / downsampled belief series for experiments live in `belief_metrics.csv` from the experiment driver, not in the raw trainer alone.

### After the last round

If `batch` non-empty, learning on, and prototypes not frozen: one **final flush** SGD step (same averaging rule), then clear. The recorded `env_round_ended` for that event is the **last completed environment round** (0-based index), which equals `T−1` for a full run and is smaller if the main loop stopped early (adaptive mode).

---

## Optional adaptive stopping (`stop_on_convergence`)

When `ESLConfig.stop_on_convergence` is **true**, `num_rounds` is interpreted as a hard cap \(T_{\max}\). After each round \(t\) once \(t+1 \ge W\) (`convergence_window_w` = \(W\)), the trainer checks:

1. **Entropy (instantaneous):** \(\bar H_t < \varepsilon_H\) using `belief_entropy_mean` for round \(t\).
2. **Separation (instantaneous):** Let \(\Delta_t = |P(\mathrm{C})_{\pi(0)} - P(\mathrm{C})_{\pi(1)}|\) where \(\pi\) is the Hungarian match at **current** \(\theta\) between true-type rows \(0,1\) and prototypes (same row assignment as matched CE). Require \(\Delta_t > \varepsilon_\Delta\).
3. **Prototype stability (window):** \(\max_{s \in \{t-W+1,\ldots,t\}} \texttt{prototype\_update\_norm}_s < \varepsilon_\theta\). Non–prototype-update rounds log norm \(0\); this uses the same per-round scalar already written to `metrics_trajectory.csv`, not \(\|\theta_t-\theta_{t-1}\|\) on every round.
4. **Belief stability (window):** \(\max_{s \in \{t-W+1,\ldots,t\}} \texttt{belief\_change\_norm}_s < \varepsilon_B\).

All \(\varepsilon\)’s and \(W\) are **user-set before the run** (config / CLI). If the conditions never hold, the run stops at \(T_{\max}\). Summary JSON includes `num_rounds_executed`, `stopped_on_convergence`, `convergence_round` (0-based \(t\) when stopped early, else `null`), and `convergence_thresholds`.

Default is **fixed horizon**: `stop_on_convergence` false, behavior unchanged.

---

## Metrics

Hungarian matching pairs **rows** of `true_type_probs` (shape \(K \times |A|\)) to learned prototypes to minimize sum of \(\mathrm{CE}(p^{\mathrm{true}}_t \,\|\, \mathrm{softmax}(\theta_k))\). Belief accuracy uses the same permutation to map “predicted prototype index” to “true type index.”

---

## File reference

| Piece | Location |
|-------|----------|
| Main loop | `esl/trainer.py` → `run_esl` |
| Bayes + Π_{Δ_K^δ} | `esl/beliefs.py`, `esl/utils/simplex.py` |
| Likelihoods / gradients | `esl/prototypes.py` |
| True type soft rows | `esl/games.py` → `true_type_distributions` |
| Config | `esl/config.py` → `ESLConfig` |
