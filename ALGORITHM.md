# ESL — algorithm reference (single file)

This document is the **only** algorithm narrative for the repository: **what the code does today**, the **theory-aligned target**, and **Cormen-style pseudocode**. Product requirements, experiments, and MVP scope live in **[PRD.md](PRD.md)**; the v2 protocol addendum is **PRD §19** and should stay consistent with the two narrative sections below.

**Coding reviews:** start with [Current implementation](#current-implementation). **Paper alignment:** [Theory-aligned target](#theory-aligned-target) + PRD §19. **Executable-style spec:** [Pseudocode (Cormen-style)](#pseudocode-cormen-style).

| Code entry | Location |
|------------|----------|
| Main loop | `esl/trainer.py` → `run_esl` |
| Pair sampling / \(L_t\) | `esl/interaction_protocol.py` |
| Config | `esl/config.py` → `ESLConfig` |
| Metrics (MCE, belief CE/KL vs type) | `esl/metrics.py` |
| Synthetic ground truth (not used by trainer) | `esl/synthetic_population.py` |

### Run outputs (`summary_metrics.json`)

`run_esl` writes a JSON summary including, among others: **`final_matched_cross_entropy`** (Hungarian **total** CE over matched pairs — legacy field), **`final_mce`** (paper-style \((1/K)\) × that total, via `mce_value`), **`final_belief_entropy`**, **`final_belief_argmax_accuracy`**, **`final_prototype_gap`** (Hungarian-matched \(|P(C)\) separation between true types 0 and 1), **`final_prototype_softmax`**, **`prototype_update_count`**, **`prototype_update_every_q`**, **`num_interaction_events_executed`**, **`p_obs`**, **`prototype_lr_scale`**, **`init_noise`**, **`freeze_prototype_parameters`**. Preset runs add **`run_manifest.json`** (see `esl/experiments/`).

---

## Current implementation

This section matches **`esl.trainer.run_esl`** and related modules **as of this repository revision**.

### Scheduler (environment rounds)

Each **environment round** \(t\):

1. If `force_ordered_pair` is set (**locked precedence**): \(\mathcal{E}_t = \{(i,j)\}\) with \(L_t=1\); `interaction_pairs_min` / `max` are ignored for that run.
2. Else: sample \(L_t \in [\texttt{interaction\_pairs\_min}, \texttt{interaction\_pairs\_max}]\) uniformly (if min \(=\) max, **no RNG** is consumed for \(L_t\)). Sample \(L_t\) **distinct** ordered pairs **without replacement** into \(\mathcal{E}_t\). Order is the **interaction order** (sequential belief updates).
3. For each \((i,j) \in \mathcal{E}_t\) in order: run **one interaction** (actions → payoffs → observation → Bayes/batch as below).

**Prototype SGD:** after each batch append (when learning is enabled), let **`n`** be the global count of such appends. When **`n mod Q == 0`** and the batch is non-empty, run one prototype step (or clear batch only if `freeze_prototype_parameters`). **`Q = prototype_Q()`** after `ESLConfig.validate()` (field `prototype_update_every`; optional `prototype_update_every_interactions` overwrites it on validate).

**Defaults** `interaction_pairs_min = interaction_pairs_max = 1` recover **one random ordered pair per round** with the same **`rng.integers`** consumption on that draw as the pre–multi-pair implementation.

### Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | `num_agents` |
| \(K\) | `num_prototypes` |
| \(Q\) | `prototype_Q()` — interactions per prototype checkpoint |
| \(\tau_a\) | `true_types[a]` — for metrics / recovery hidden policies |
| \(w\) | Observation mask |
| \(b^{\mathrm{snap}}_{i\to j}\) | Pre-Bayes belief stored in the batch record |

#### Implementation note: \(K\) larger than base behavioral templates

See **PRD** and [Theory-aligned target](#theory-aligned-target): cycling through AC/AD templates when \(K>2\) is an **implementation convenience**, not a theoretical claim.

### State

- **Prototypes:** \(\theta \in \mathbb{R}^{K \times |A|}\) (`logits`)
- **Beliefs:** \(B[i,j] \in \Delta_K^\delta\) for \(i \neq j\)
- **Batch:** `BatchRecord` list
- **`m`:** prototype SGD step index
- **`n`:** global interaction-event counter (increments on each batch append when not `learning_frozen`)

### Initialization

Same as before: \(\theta\), uniform \(B[i,j]\), empty batch, assign \(\tau_a\), build `hidden_policies`, adaptation mask if needed.

### Per-interaction block (inside round \(t\), for each \((i,j) \in \mathcal{E}_t\))

#### Actions

- **Recovery:** `hidden_policies[i].act`, `hidden_policies[j].act` (fixed policies).
- **Adaptation:** `act_agent` uses **predictive opponent law** \(\hat\pi_{j\mid i} = b_{i\to j}^\top \sigma(\theta)\) (implemented as `marginal_opponent_probs` → logit best response).

#### Observation

`w` from `sample_observation_mask`; `s = action_to_signal(a_j)`.

#### Belief + batch

If `learning_frozen`: no \(B\) update, no append, **`n`** unchanged.

Else: snapshot \(b^{\mathrm{snap}}\), Bayes + \(\Pi_{\Delta^\delta}\) if \(w>0\), append record, **`n \leftarrow n+1`**. If **`n \bmod Q = 0`** and batch non-empty: prototype SGD (unless freeze-prototypes → clear only) with mean gradient over **full** `len(batch)` including \(w=0\) rows; optional **`prototype_l2_eta`**: \(\theta \leftarrow \theta + \gamma(\bar g - \eta_{\mathrm{reg}}\theta)\).

### End of round \(t\)

- **`belief_change_norm`:** \(\sum_{i,j}|B_{t}^{\mathrm{end}}-B_{t}^{\mathrm{start}}|\) over the full tensor (start = tensor at beginning of round).
- **`summary_rows`:** one row — metrics from state **after** all interactions in \(\mathcal{E}_t\) (and any in-round SGD).
- **`prototype_rows`:** one row — \(\theta\) after last SGD in the round, if any.
- **`belief_trajectory`:** default one snapshot per round; if `log_beliefs_every_interaction`, append after **each** interaction.

### Final flush

If batch non-empty after the last round, same prototype step as before (see `run_esl`).

### Metrics (trainer)

Hungarian **matched cross-entropy** (MCE) between `true_type_distributions(K)` rows and learned softmax prototypes; belief argmax accuracy uses the same permutation. See **`esl/metrics.py`** for MCE helpers and belief CE/KL vs true type for evaluation drivers.

### File reference

| Piece | Location |
|-------|----------|
| Main loop | `esl/trainer.py` → `run_esl` |
| Pair / \(L_t\) sampling | `esl/interaction_protocol.py` |
| Bayes + simplex | `esl/beliefs.py`, `esl/utils/simplex.py` |
| Likelihoods | `esl/prototypes.py` |
| Config | `esl/config.py` |
| **Synthetic eval only** | `esl/synthetic_population.py` (**not** imported by `trainer`) |

### Adaptive stopping

Window **`W`** is in **environment rounds**; each `summary_rows` entry is **end-of-round**. See `run_esl` + `convergence_criteria_met`.

---

## Theory-aligned target

This section states the **target** mathematical protocol and update rules aligned with the paper-style design. The **running code** may lag; [Current implementation](#current-implementation) is what `run_esl` does today.

### State

- **Prototypes:** \(\Theta_m = \{\theta_k\}_{k=1}^K\)
- **Beliefs:** \(B_t = \{b_{i\to j,t}\}_{i\neq j}\)
- **Batch:** \(\mathcal{D}_m\) (list of tuples after each observed interaction slot)
- **Global interaction counter:** \(n\) (monotone; typically increments once per processed interaction record appended to the slow buffer)
- **Prototype step index:** \(m\)

### Initialization

\(\Theta_0\), uniform \(b_{i\to j,0}\), \(\mathcal{D}_0=\emptyset\), \(n=0\), \(m=0\).

### Main loop (rounds \(t=0,1,\ldots\))

#### 1. Sample interaction count

\(L_t \sim \mathcal{L}\) supported on \(\{L_{\min},\ldots,L_{\max}\}\). (Degenerate \(L_{\min}=L_{\max}\) needs no RNG draw for \(L_t\) if bit-reproducibility is required.)

#### 2. Sample ordered pairs

\(\mathcal{E}_t \subset \{(i,j): i\neq j\}\), \(|\mathcal{E}_t|=L_t\), **without replacement**. Process pairs in a fixed order (e.g. the sample order).

**Debug override:** a single fixed pair forces \(L_t=1\) and \(\mathcal{E}_t=\{(i,j)\}\), ignoring \(L_{\min},L_{\max}\).

#### 3. Sequential interaction processing

For each \((i,j)\in\mathcal{E}_t\) in order:

**(a) Actions**

- **Recovery:** policies exogenous to \(\Theta\)-learning (fixed or simulator-drawn).
- **Adaptation:** opponent predictive law  
  \[
  \hat\pi_{j\mid i,t}(a_j)=\sum_{k=1}^K b_{i\to j,t}[k]\,\pi_k(a_j\mid\theta_k),
  \]
  then row/column utilities and logit best response for the acting agent.

**(b) Observation:** \(W_{ij,t}\sim\mathrm{Bernoulli}(p_{\mathrm{obs}})\); signal \(s_{ij,t}\) from the game if \(W=1\).

**(c) Snapshot:** \(b^{\mathrm{snap}}=b_{i\to j,t}\).

**(d) Belief update:** Bayes with unclamped \(L_k(s\mid\theta_k)\), then \(\Pi_{\Delta_K^\delta}\).

**(e) Append** \((i,j,s,W,b^{\mathrm{snap}})\) to \(\mathcal{D}\); **\(n\leftarrow n+1\)** (convention: count structural slots consistent with the implementation’s batch semantics).

**(f) Slow update:** if \(n \equiv 0 \pmod Q\) and \(\mathcal{D}\neq\emptyset\):  
\[
\bar g_k = \frac{1}{|\mathcal{D}|}\sum_{\mathcal{D}} w\, b^{\mathrm{snap}}[k]\,\nabla_{\theta_k}\log L_k(s\mid\theta_k),
\]
\[
\theta_k \leftarrow \theta_k + \gamma_m\bigl(\bar g_k - \eta_{\mathrm{reg}}\theta_k\bigr),
\]
then \(\mathcal{D}\leftarrow\emptyset\), \(m\leftarrow m+1\) (optional \(\eta_{\mathrm{reg}}=0\)).

Unobserved rows (\(w=0\)) contribute **0** to the gradient numerator but **count** in \(|\mathcal{D}|\).

#### 4. End-of-round telemetry

Log beliefs / \(\Theta\) / scalars **after** all pairs in \(\mathcal{E}_t\) (see **PRD** / **Current implementation** for CSV shapes).

### Synthetic population (evaluation / simulation only)

True parameters \(\Theta^\star=\{\theta_k^\star\}\), latent types \(z_i\sim\mathrm{Categorical}(\rho)\), idiosyncratic noise \(\phi_i\sim\mathcal{D}_{z_i}\), agent logits \(\tilde\theta_i=\theta^\star_{z_i}+\phi_i\). **Not** observed by the learner; used only to generate actions in experiments and to score metrics. Implementations: **`esl/synthetic_population.py`**.

### Limiting interpretation (stochastic approximation)

The slow iterate is understood as a **stochastic approximation** driven by the **fast** belief Markov chain. In general, averaging and limits are phrased via **invariant measures** and **differential inclusions**; a **mean-field ODE** is a **special case** under stronger mixing / uniqueness assumptions—not the general claim.

### Metrics (evaluation)

- **MCE:** \(\min_{\sigma\in S_K}\frac{1}{K}\sum_k \mathrm{CE}\bigl(\pi^\star_k \,\|\, \pi_{\sigma(k)}(\Theta)\bigr)\) over softmax policies (Hungarian in code).
- **Beliefs vs \(z_j\):** CE\((e_{z_j}, b_{i\to j})\), KL\((e_{z_j}\| b_{i\to j})\), argmax accuracy (permutation-aligned where applicable).

---

## Pseudocode (Cormen-style)

Cormen-style pseudocode for the ESL algorithm in this repository.

- **Implementation-faithful pseudocode:** matches [`esl/trainer.py`](esl/trainer.py) `run_esl` and [`esl/config.py`](esl/config.py).
- **Flagship block:** specialization of the same algorithm for the canonical balanced recovery run (multi-pair rounds, strong prototype learning).

Narrative: [Current implementation](#current-implementation). Product spec: **PRD.md** (§19 addendum).

### A.1 Inputs (configuration parameters and meaning)

Let `CFG` be a record with (key fields only):

```text
# Core
CFG.seed                      # RNG seed
CFG.mode ∈ {recovery, adaptation}
CFG.num_agents = N
CFG.num_prototypes = K
CFG.num_actions = 2
CFG.num_rounds = R            # environment rounds (hard cap)

# Pair protocol
CFG.force_ordered_pair        # optional fixed pair (i,j); else None
CFG.interaction_pairs_min     # L_min (≥1)
CFG.interaction_pairs_max     # L_max (≥L_min)
CFG.interaction_pairs_law     # "uniform" on {L_min,…,L_max}

# Observability / beliefs
CFG.observability ∈ {full, sparse}
CFG.p_obs ∈ [0,1]
CFG.delta_simplex = δ
CFG.bayes_denominator_eps = ε_bayes

# Prototypes / schedules
CFG.prototype_update_every                # Q after validate() (interaction-event schedule)
CFG.prototype_update_every_interactions   # if not None: overwrite Q in validate()
CFG.prototype_lr_scale
CFG.lr_prototype_gamma_exponent
CFG.prototype_l2_eta = η_reg ≥ 0

# Initialization
CFG.prototype_logits_override  # optional θ init (K×2)
CFG.base_init
CFG.init_noise
CFG.symmetric_init

# Control / ablations
CFG.learning_frozen
CFG.freeze_prototype_parameters

# Logging (affects output size; not learning)
CFG.log_beliefs_tensor
CFG.log_beliefs_every_interaction

# True-type assignment for recovery / metrics
CFG.force_agent_true_types     # optional list length N; each in {0,…,K−1}
```

After `CFG.validate()`:

- `Q ← CFG.prototype_Q()` returns `CFG.prototype_update_every` (possibly overwritten from `prototype_update_every_interactions`).
- Pair feasibility holds: \(1 \le L_{\min}\le L_{\max} \le N(N-1)\).
- If `force_ordered_pair` is set, indices are in range and `i ≠ j`.

### A.2 Data structures

```text
θ            # prototype logits, shape (K, 2)
B            # belief tensor, shape (N, N, K)
batch        # list of BatchRecord
LOG          # run log structure (summary_rows, prototype_rows, reward_rows, belief_rows, prototype_update_events)

BatchRecord = (i, j, s, w, b_snap)
    i, j     # ordered pair (observer i, target j)
    s        # signal (here: observed action of j)
    w        # observation mask ∈ {0,1} or {0.0,1.0}
    b_snap   # snapshot of belief B[i,j,:] BEFORE Bayes update for this signal
```

### A.3 Helper procedures (faithful to current code)

#### A.3.1 Beliefs

```text
BELIEF_INIT(N, K):
    # matches esl.beliefs.init_beliefs
    B ← array of shape (N,N,K) filled with 1/K
    for i ← 0 to N−1:
        B[i,i,:] ← (0,…,0)          # diagonal defined as zeros (unused)
    return B

BAYES_POSTERIOR_BEFORE_PROJECTION(prior, L, ε):
    # stabilized posterior; normalized up to eps
    num[k] ← prior[k] * L[k]
    den    ← Σ_q num[q] + ε
    post[k] ← num[k] / den
    return post

UPDATE_BELIEF_PAIR(prior, L, δ, ε):
    post ← BAYES_POSTERIOR_BEFORE_PROJECTION(prior, L, ε)
    return PROJECT_TO_DELTA_K_FLOOR(post, δ)   # Euclidean projection onto Δ_K^δ
```

#### A.3.2 Interaction sampling

```text
ALL_ORDERED_PAIRS(N):
    P ← empty list
    for i ← 0..N−1:
        for j ← 0..N−1:
            if i ≠ j: append (i,j) to P
    return P

SAMPLE_L_t(RNG, L_min, L_max):
    # locked parity: if L_min==L_max, return constant without consuming RNG
    if L_min == L_max:
        return L_min
    return RNG.randint(L_min, L_max+1)

SAMPLE_ORDERED_PAIRS_WITHOUT_REPLACEMENT(RNG, N, L_t):
    P ← ALL_ORDERED_PAIRS(N)            # fixed row-major order
    M ← |P|
    assert 1 ≤ L_t ≤ M

    if L_t == 1:
        idx ← RNG.randint(0, M)         # single integers draw (legacy parity)
        return [ P[idx] ]

    σ ← RNG.permutation(0..M−1)
    return [ P[σ[k]] for k=0..L_t−1 ]   # distinct ordered pairs, in sampled order
```

#### A.3.3 Prototype SGD step (with optional L2)

```text
PROTOTYPE_LR(CFG, m):
    return CFG.prototype_lr_scale * (m+1)^(CFG.lr_prototype_gamma_exponent)

PROTOTYPE_SGD_STEP_FROM_BATCH(batch, θ, CFG, m):
    # mean gradient over FULL |batch|, including w=0 rows in denominator
    G_accum ← zeros_like(θ)

    for each (i,j,s,w,b_snap) in batch:
        if w ≤ 0:
            continue
        for k ← 0..K−1:
            p_k ← softmax(θ[k,:])                # stable softmax
            G_accum[k,:] ← G_accum[k,:] + w*b_snap[k]*(e_s − p_k)

    denom ← max(|batch|, 1)
    G_mean ← G_accum / denom

    η_reg ← CFG.prototype_l2_eta
    if η_reg > 0:
        G_step ← G_mean − η_reg * θ
    else:
        G_step ← G_mean

    γ ← PROTOTYPE_LR(CFG, m)
    Δθ ← γ * G_step
    θ_new ← θ + Δθ
    return (θ_new, ||Δθ||_2)
```

### A.4 Main algorithm

#### A.4.1 RUN_ESL

```text
RUN_ESL(CFG):

    CFG.validate()
    RNG ← RandomGenerator(CFG.seed)

    N ← CFG.num_agents
    K ← CFG.num_prototypes
    Q ← CFG.prototype_Q()

    payoffs ← PrisonersDilemmaPayoffs(CFG)
    true_type_probs ← TrueTypeDistributions(K)     # implementation convenience when K>2 may cycle templates

    # True types (for recovery policies + metrics)
    if CFG.force_agent_true_types exists:
        TRUE_TYPES[a] ← CFG.force_agent_true_types[a]  for a=0..N−1
    else:
        TRUE_TYPES[a] ← a mod K

    # Hidden policies for recovery mode (registry size may be < K; modulo cycling is an implementation convenience)
    HIDDEN_POLICIES[a] ← BuildHiddenPolicy(TRUE_TYPES[a] mod registry_size)

    # Initialize θ
    if CFG.prototype_logits_override exists:
        θ ← copy(CFG.prototype_logits_override)
    else:
        noise ← 0 if CFG.symmetric_init else CFG.init_noise
        for k ← 0..K−1:
            for a ← 0..1:
                θ[k,a] ← CFG.base_init + noise * RNG.standard_normal()

    # Initialize beliefs
    B ← BELIEF_INIT(N, K)

    LOG ← NewRunLog()
    batch ← empty list
    m ← 0                      # prototype step index
    n ← 0                      # global interaction-event counter (increments on each batch append)
    LAST_OPP[a] ← None for a=0..N−1

    stopped_on_convergence ← FALSE
    convergence_round ← NIL

    for t ← 0..CFG.num_rounds−1:

        (θ, B, batch, m, n, LAST_OPP, stopped_on_convergence, convergence_round, LOG)
            ← RUN_ONE_ROUND(CFG, RNG, t, θ, B, batch, m, n, LAST_OPP, LOG,
                             TRUE_TYPES, HIDDEN_POLICIES, payoffs, true_type_probs)

        if stopped_on_convergence:
            break

    # Final flush (event-based schedule semantics)
    # Applies one additional prototype step on the trailing incomplete batch even if n is not a multiple of Q.
    if (NOT CFG.learning_frozen) AND (batch not empty):
        if CFG.freeze_prototype_parameters:
            CLEAR(batch)
        else:
            θ_before ← copy(θ)
            (θ, norm) ← PROTOTYPE_SGD_STEP_FROM_BATCH(batch, θ, CFG, m)
            APPEND_PROTOTYPE_UPDATE_EVENT(LOG, CFG, m, env_round_ended = last_completed_round,
                                          θ_before = θ_before, θ_after = θ,
                                          prototype_update_norm = norm,
                                          final_flush = TRUE,
                                          batch = batch)
            m ← m + 1
            CLEAR(batch)

    return (LOG, θ, B, SUMMARY(LOG, θ, true_type_probs), run_dir)
```

#### A.4.2 RUN_ONE_ROUND

```text
RUN_ONE_ROUND(CFG, RNG, t, θ, B, batch, m, n, LAST_OPP, LOG,
              TRUE_TYPES, HIDDEN_POLICIES, payoffs, true_type_probs):

    N ← CFG.num_agents
    K ← CFG.num_prototypes
    Q ← CFG.prototype_Q()

    B_start ← copy(B)                 # used for belief_change_norm over the round
    proto_norm_last_update_in_round ← 0.0
    last_interaction_batch_loglik ← 0.0

    # 1) Choose E_t (ordered pairs for this round)
    if CFG.force_ordered_pair exists:
        E_t ← [ CFG.force_ordered_pair ]           # locked precedence: fixed pair ⇒ L_t=1; ignore L_min/L_max
    else:
        L_t ← SAMPLE_L_t(RNG, CFG.interaction_pairs_min, CFG.interaction_pairs_max)
        E_t ← SAMPLE_ORDERED_PAIRS_WITHOUT_REPLACEMENT(RNG, N, L_t)

    # 2) Sequentially process each (i,j)
    for each (i,j) in E_t:

        # 2a) Actions
        if CFG.mode == recovery:
            a_j ← HIDDEN_POLICIES[j].act(RNG, LAST_OPP[j])
            a_i ← HIDDEN_POLICIES[i].act(RNG, LAST_OPP[i])

        else:   # adaptation (explicit predictive distribution)
            # π̂_{j|i}(·) = Σ_k B[i,j,k] * softmax(θ[k,:])
            # ActAgent computes expected utilities vs π̂ and draws logit best response.
            a_i ← ActAgent(i, j, B, θ, payoffs, LAST_OPP, CFG, RNG, is_row_player=TRUE)
            a_j ← ActAgent(j, i, B, θ, payoffs, LAST_OPP, CFG, RNG, is_row_player=FALSE)

        LAST_OPP[i] ← a_j
        LAST_OPP[j] ← a_i

        # 2b) Reward logging (telemetry only)
        (r_i, r_j) ← PlayPairPayoffs(a_i, a_j, payoffs)
        APPEND(LOG.reward_rows, {round:t, i:i, j:j, r_i:r_i, r_j:r_j})

        # 2c) Observe
        w ← SampleObservationMask(CFG, RNG)
        s ← ActionToSignal(a_j)

        if CFG.learning_frozen:
            b_now ← copy(B[i,j,:])
            if w > 0:
                last_interaction_batch_loglik ← Σ_k b_now[k] * w * log_clamped_softmax_prob(θ[k,:], s)
            else:
                last_interaction_batch_loglik ← 0

        else:
            # 2d) Snapshot and Bayes update
            b_snap ← copy(B[i,j,:])

            if w > 0:
                for k ← 0..K−1:
                    L[k] ← softmax(θ[k,:])[s]                 # unclipped likelihood for Bayes
                B[i,j,:] ← UPDATE_BELIEF_PAIR(B[i,j,:], L, δ=CFG.delta_simplex, ε=CFG.bayes_denominator_eps)

            # 2e) Append batch record (always, including w=0)
            APPEND(batch, (i, j, s, w, b_snap))
            n ← n + 1

            # 2f) Telemetry log-likelihood (uses b_snap; clamp for logs only)
            if w > 0:
                last_interaction_batch_loglik ← Σ_k b_snap[k] * w * log_clamped_softmax_prob(θ[k,:], s)
            else:
                last_interaction_batch_loglik ← 0

            # 2g) Scheduled prototype update (multiple per round possible)
            if (n mod Q == 0) AND (batch not empty):
                if CFG.freeze_prototype_parameters:
                    CLEAR(batch)        # collects records, then clears at schedule times without changing θ
                else:
                    θ_before ← copy(θ)
                    (θ, norm) ← PROTOTYPE_SGD_STEP_FROM_BATCH(batch, θ, CFG, m)
                    proto_norm_last_update_in_round ← norm      # logs LAST update norm if multiple occur
                    APPEND_PROTOTYPE_UPDATE_EVENT(LOG, CFG, m, env_round_ended=t,
                                                  θ_before=θ_before, θ_after=θ,
                                                  prototype_update_norm=norm,
                                                  final_flush=FALSE,
                                                  batch=batch)
                    m ← m + 1
                    CLEAR(batch)

        # Optional belief tensor logging per interaction (output-size control)
        if CFG.log_beliefs_tensor AND CFG.log_beliefs_every_interaction:
            APPEND_BELIEF_ROWS(LOG, B, t)

    # 3) End-of-round metrics (after all interactions and any in-round prototype steps)
    belief_change_norm ← Σ_{i,j,k} |B[i,j,k] − B_start[i,j,k]|     # diagonal is well-defined as 0
    α_logged ← CFG.belief_lr(t)                                    # bookkeeping; beliefs update is Bayes

    (perm, total_ce) ← MATCH_PROTOTYPES_TO_TYPES(true_type_probs, θ)
    H_mean ← BELIEF_ENTROPY_OFFDIAGONAL(B)
    acc ← BELIEF_ARGMAX_ACCURACY(B, TRUE_TYPES, perm)

    APPEND(LOG.summary_rows, {
        round: t,
        matched_cross_entropy: total_ce,
        belief_entropy_mean: H_mean,
        belief_argmax_accuracy: acc,
        batch_log_likelihood: last_interaction_batch_loglik,        # explicitly “last interaction” telemetry
        alpha_logged: α_logged,
        prototype_step_m: m,
        prototype_update_norm: proto_norm_last_update_in_round,     # last update norm in this round (0 if none)
        belief_change_norm: belief_change_norm
    })

    # 4) End-of-round prototype row (θ after the last in-round update, if any)
    APPEND(LOG.prototype_rows, PrototypeRowFromTheta(θ, t, m))

    # 5) Belief tensor logging once per round if enabled and not per-interaction
    if CFG.log_beliefs_tensor AND (NOT CFG.log_beliefs_every_interaction):
        APPEND_BELIEF_ROWS(LOG, B, t)

    # 6) Optional convergence stopping (window in rounds)
    if CFG.stop_on_convergence AND t+1 ≥ CFG.convergence_window_w:
        if CONVERGENCE_CRITERIA_MET(LOG.summary_rows, t, CFG, θ, true_type_probs):
            return (θ, B, batch, m, n, LAST_OPP, TRUE, t, LOG)

    return (θ, B, batch, m, n, LAST_OPP, FALSE, NIL, LOG)
```

### B. Flagship recovery configuration (specialized pseudocode)

This section specializes the above algorithm to the flagship recovery experiment.

#### B.1 Parameterization (constants)

```text
CFG.mode = recovery
CFG.num_agents = 20
CFG.num_prototypes = 2
CFG.num_actions = 2
CFG.num_rounds = 10000

# Balanced two-type recovery (type 0 = AC template, type 1 = AD template)
CFG.force_agent_true_types = [0]*10 + [1]*10

# Multi-pair rounds
CFG.force_ordered_pair = None
CFG.interaction_pairs_min = 5
CFG.interaction_pairs_max = 15
CFG.interaction_pairs_law = "uniform"

# Observability
CFG.observability = full
CFG.p_obs = 1.0

# Beliefs / simplex
CFG.delta_simplex = 0.02
CFG.bayes_denominator_eps = 1e-12

# Prototype schedule (event-based)
CFG.prototype_update_every_interactions = None
CFG.prototype_update_every = 15              # Q = 15 interactions per prototype step

# Initialization asymmetry
CFG.base_init = 0.0
CFG.init_noise = 0.05
CFG.symmetric_init = False

# Prototype learning strength
CFG.prototype_lr_scale = 22
CFG.lr_prototype_gamma_exponent = −0.9
CFG.prototype_l2_eta = 0.0

# Logging (long-run safety; does not affect learning)
CFG.log_beliefs_tensor = False
CFG.log_beliefs_every_interaction = False

# Seed
CFG.seed = 42   (and a comparison run uses seed = 43)
```

#### B.2 Algorithm (same as RUN_ESL, with constants)

```text
FlagshipRecoveryRun():

    Initialize RNG with seed (42 or 43)

    θ ← base_init + init_noise * Normal(0,1) noise per entry (K×2)
    B ← uniform off-diagonal beliefs; diagonal rows are zeros

    batch ← empty
    m ← 0
    n ← 0

    for t = 0..9999:

        L_t ← UniformInteger(5, 15)
        E_t ← L_t distinct ordered pairs sampled without replacement

        for (i,j) in E_t (sequentially):

            # Recovery actions
            a_i ← hidden_policy(TRUE_TYPES[i])       # AC or AD
            a_j ← hidden_policy(TRUE_TYPES[j])

            w ← 1
            s ← a_j

            b_snap ← copy(B[i,j])
            B[i,j] ← Bayes(b_snap, softmax(θ)[s]) then project to Δ_2^δ
            append (i,j,s,w,b_snap) to batch
            n ← n + 1

            if n mod 15 == 0:
                θ ← θ + γ_m * mean_batch_gradient(batch)     # denominator = |batch| = 15
                m ← m + 1
                clear batch

        # end-of-round: compute and log MCE, belief entropy, argmax accuracy, θ softmax rows

    # final flush if trailing batch nonempty
```

### C. Faithfulness statement

- **Implementation:** `esl/trainer.py` + `esl/interaction_protocol.py` are faithful to **Section A** (pseudocode above).
- **Narrative:** [Current implementation](#current-implementation) is the prose counterpart; **PRD.md** is the product spec.
