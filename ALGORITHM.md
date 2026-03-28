# ESL core algorithm (implementation-faithful)

This document matches the behavior of `esl.trainer.run_esl` and related modules. For the full product spec, see **`PRD.md`**.

## Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | `num_agents` |
| \(K\) | `num_prototypes` — **learned** prototype count (logits \(\theta \in \mathbb{R}^{K \times |A|}\)) |
| \(\tau_a \in \{0,\ldots,K-1\}\) | Agent \(a\)’s **assigned type index** (stored in `true_types`; used for metrics and, modulo the behavioral registry, for hidden policies) |
| Behavioral policies | `games.HIDDEN_POLICY_BUILDERS` defines built-in behaviors (Always C / Always D). If \(K>2\), **row \(k\)** of the true action distribution matrix still cycles those behaviors: `true_type_distributions(K)` has shape \((K,|A|)\). So **learned \(K\)** may **exceed** the number of *distinct* behavioral templates; indices map via `k % n_base`. |
| \(M\) | `prototype_update_every` — slow timescale |
| \(w \in \{0,1\}\) | Observation mask (`sample_observation_mask`; sparse \(w \sim \mathrm{Bernoulli}(p_{\mathrm{obs}})\)) |
| \(b^{\mathrm{snap}}_{i\to j}\) | Belief **before** Bayes on the current signal (what the batch stores). |

## Initialization

```
Initialize θ ∈ ℝ^{K×|A|}     (override, or symmetric / noisy draw)
Initialize B[i,j] ∈ Δ^{K-1} for all ordered pairs i ≠ j (uniform; diagonal unused)
batch ← empty list
m ← 0   // prototype SGD step index

Assign τ_a for each agent a (config override or cyclic 0..K-1)
Build hidden policies π_a from τ_a (policy class = f(τ_a mod n_behavioral))

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
    B[i,j] ← BayesThenSimplexFloor(B[i,j], L, δ, ε_floor, …)

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

If `batch` non-empty, learning on, and prototypes not frozen: one **final flush** SGD step (same averaging rule), then clear.

---

## Metrics

Hungarian matching pairs **rows** of `true_type_probs` (shape \(K \times |A|\)) to learned prototypes to minimize sum of \(\mathrm{CE}(p^{\mathrm{true}}_t \,\|\, \mathrm{softmax}(\theta_k))\). Belief accuracy uses the same permutation to map “predicted prototype index” to “true type index.”

---

## File reference

| Piece | Location |
|-------|----------|
| Main loop | `esl/trainer.py` → `run_esl` |
| Bayes + floor | `esl/beliefs.py` |
| Likelihoods / gradients | `esl/prototypes.py` |
| True type soft rows | `esl/games.py` → `true_type_distributions` |
| Config | `esl/config.py` → `ESLConfig` |
