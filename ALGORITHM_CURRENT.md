# ESL core algorithm — **current implementation**

This document matches **`esl.trainer.run_esl`** and related modules **as of this repository revision**. For the **target** (theory-aligned) specification, see **`ALGORITHM_TARGET.md`**. For product goals and math spec, see **`PRD.md`**. The root **`ALGORITHM.md`** is an index.

For consolidated **Cormen-style pseudocode**, see **[`ESL_PSEUDOCODE.md`](ESL_PSEUDOCODE.md)**.

---

## Scheduler (environment rounds)

Each **environment round** \(t\):

1. If `force_ordered_pair` is set (**locked precedence**): \(\mathcal{E}_t = \{(i,j)\}\) with \(L_t=1\); `interaction_pairs_min` / `max` are ignored for that run.
2. Else: sample \(L_t \in [\texttt{interaction\_pairs\_min}, \texttt{interaction\_pairs\_max}]\) uniformly (if min \(=\) max, **no RNG** is consumed for \(L_t\)). Sample \(L_t\) **distinct** ordered pairs **without replacement** into \(\mathcal{E}_t\). Order is the **interaction order** (sequential belief updates).
3. For each \((i,j) \in \mathcal{E}_t\) in order: run **one interaction** (actions → payoffs → observation → Bayes/batch as below).

**Prototype SGD:** after each batch append (when learning is enabled), let **`n`** be the global count of such appends. When **`n mod Q == 0`** and the batch is non-empty, run one prototype step (or clear batch only if `freeze_prototype_parameters`). **`Q = prototype_Q()`** after `ESLConfig.validate()` (field `prototype_update_every`; optional `prototype_update_every_interactions` overwrites it on validate).

**Defaults** `interaction_pairs_min = interaction_pairs_max = 1` recover **one random ordered pair per round** with the same **`rng.integers`** consumption on that draw as the pre–multi-pair implementation.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | `num_agents` |
| \(K\) | `num_prototypes` |
| \(Q\) | `prototype_Q()` — interactions per prototype checkpoint |
| \(\tau_a\) | `true_types[a]` — for metrics / recovery hidden policies |
| \(w\) | Observation mask |
| \(b^{\mathrm{snap}}_{i\to j}\) | Pre-Bayes belief stored in the batch record |

### Implementation note: \(K\) larger than base behavioral templates

See **PRD** / **ALGORITHM_TARGET.md**: cycling through AC/AD templates when \(K>2\) is an **implementation convenience**, not a theoretical claim.

---

## State

- **Prototypes:** \(\theta \in \mathbb{R}^{K \times |A|}\) (`logits`)
- **Beliefs:** \(B[i,j] \in \Delta_K^\delta\) for \(i \neq j\)
- **Batch:** `BatchRecord` list
- **`m`:** prototype SGD step index
- **`n`:** global interaction-event counter (increments on each batch append when not `learning_frozen`)

---

## Initialization

Same as before: \(\theta\), uniform \(B[i,j]\), empty batch, assign \(\tau_a\), build `hidden_policies`, adaptation mask if needed.

---

## Per-interaction block (inside round \(t\), for each \((i,j) \in \mathcal{E}_t\))

### Actions

- **Recovery:** `hidden_policies[i].act`, `hidden_policies[j].act` (fixed policies).
- **Adaptation:** `act_agent` uses **predictive opponent law** \(\hat\pi_{j\mid i} = b_{i\to j}^\top \sigma(\theta)\) (implemented as `marginal_opponent_probs` → logit best response).

### Observation

`w` from `sample_observation_mask`; `s = action_to_signal(a_j)`.

### Belief + batch

If `learning_frozen`: no \(B\) update, no append, **`n`** unchanged.

Else: snapshot \(b^{\mathrm{snap}}\), Bayes + \(\Pi_{\Delta^\delta}\) if \(w>0\), append record, **`n \leftarrow n+1`**. If **`n \bmod Q = 0`** and batch non-empty: prototype SGD (unless freeze-prototypes → clear only) with mean gradient over **full** `len(batch)` including \(w=0\) rows; optional **`prototype_l2_eta`**: \(\theta \leftarrow \theta + \gamma(\bar g - \eta_{\mathrm{reg}}\theta)\).

---

## End of round \(t\)

- **`belief_change_norm`:** \(\sum_{i,j}|B_{t}^{\mathrm{end}}-B_{t}^{\mathrm{start}}|\) over the full tensor (start = tensor at beginning of round).
- **`summary_rows`:** one row — metrics from state **after** all interactions in \(\mathcal{E}_t\) (and any in-round SGD).
- **`prototype_rows`:** one row — \(\theta\) after last SGD in the round, if any.
- **`belief_trajectory`:** default one snapshot per round; if `log_beliefs_every_interaction`, append after **each** interaction.

---

## Final flush

If batch non-empty after the last round, same prototype step as before (see `run_esl`).

---

## Metrics (trainer)

Hungarian **matched cross-entropy** (MCE) between `true_type_distributions(K)` rows and learned softmax prototypes; belief argmax accuracy uses the same permutation. See **`esl/metrics.py`** for MCE helpers and belief CE/KL vs true type for evaluation drivers.

---

## File reference

| Piece | Location |
|-------|----------|
| Main loop | `esl/trainer.py` → `run_esl` |
| Pair / \(L_t\) sampling | `esl/interaction_protocol.py` |
| Bayes + simplex | `esl/beliefs.py`, `esl/utils/simplex.py` |
| Likelihoods | `esl/prototypes.py` |
| Config | `esl/config.py` |
| **Synthetic eval only** | `esl/synthetic_population.py` (**not** imported by `trainer`) |

---

## Adaptive stopping

Window **`W`** is in **environment rounds**; each `summary_rows` entry is **end-of-round**. See `run_esl` + `convergence_criteria_met`.
