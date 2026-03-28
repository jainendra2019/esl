PRD: Epistemic Social Learning (ESL) MVP — Senior-Reviewed Version

1. Goal

Implement a minimal, correct, interpretable version of Epistemic Social Learning (ESL) for repeated 2-action matrix games.

The MVP should demonstrate that:
	1.	agents maintain pairwise beliefs over latent behavioral prototypes,
	2.	beliefs update from observed actions/signals via Bayes’ rule,
	3.	prototype parameters update via belief-weighted stochastic gradient ascent on a likelihood objective,
	4.	under slight initialization asymmetry, the learned prototypes specialize and recover latent behavioral types up to permutation.

This MVP is for:
	•	validation of the core algorithm,
	•	debugging,
	•	producing first recovery plots,
	•	preparing the empirical section of the paper.

It is not the final scalable implementation.

⸻

2. Explicit MVP scope

In scope
	•	repeated 2-action matrix games
	•	fixed finite set of latent prototypes K
	•	signal = observed action
	•	categorical likelihood with softmax logits
	•	pairwise beliefs over prototypes
	•	sparse or full observability via W_{ij,t}
	•	belief update via Bayes + Euclidean projection onto $\Delta_K^\delta$
	•	prototype update via stochastic gradient ascent
	•	evaluation in both recovery and adaptation modes

Out of scope
	•	fuzzy clustering / FCM
	•	contextual/state-dependent environments
	•	neural likelihood models
	•	continuous actions
	•	full decentralized multi-learner training
	•	theorem verification code
	•	performance optimization

⸻

3. Two supported experiment modes

Mode A — Recovery mode (required first)

Purpose: verify prototype recovery cleanly.
	•	Data-generating agents use fixed hidden policies
	•  These hidden policies are exogenous data generators and are not
	   updated during training.
	•	ESL learner(s) observe them and learn prototypes
	•	Recommended hidden types:
	•	Always Cooperate
	•	Always Defect
	•	Tit-for-Tat (optional later)

This is the first mode to implement.

⸻

Mode B — Strategic adaptation mode (second)

Purpose: test whether ESL improves payoff via adaptation.
	•	ESL agents choose actions via logit best response
	•	recovery is still tracked, but payoff becomes a main metric

**MVP default:** all agents use logit best response (homogeneous adaptation). Mixed populations (some fixed hidden opponents, some BR learners) are a **possible extension** via a mask or role flags; they are **not** required for MVP correctness and need not appear in the core PRD checklist.

⸻

4. Mathematical specification

4.1 Action space

Use a finite 2-action set:
\mathcal A = \{0,1\}

Recommended semantic mapping:
	•	0 = Cooperate
	•	1 = Defect

⸻

4.2 Signal space

For MVP:
\mathcal S = \mathcal A
and
s_{ij,t} = a_j^t
whenever W_{ij,t}=1.

This means the observed signal is the observed opponent action.

⸻

4.3 Context

For MVP:
\omega_t \equiv \text{constant}
and should not appear in code except possibly as a placeholder.

Do not build a context/state system in v1.

⸻

4.4 Observability

For each ordered pair (i,j):
W_{ij,t}\in\{0,1\}

Interpretation:
	•	W_{ij,t}=1: agent i observes j’s action
	•	W_{ij,t}=0: no signal is available

Support two modes:
	1.	full observation
	2.	Bernoulli sparse observation with parameter p_{\text{obs}}

⸻

4.5 Prototype parameters

Each prototype k\in\{1,\dots,K\} is represented by:
\theta_k\in\mathbb R^{|\mathcal A|}

For 2 actions:
\theta_k = (\theta_{k,0},\theta_{k,1})

These are logits, not probabilities.

⸻

4.6 Likelihood model

Define the prototype likelihood:
L_k(s=a\mid\theta_k)
=
\frac{\exp(\theta_{k,a})}{\sum_{a'}\exp(\theta_{k,a'})}

Implementation requirement:
	•	use numerically stable softmax
	•	when computing $\log L_k(s\mid\theta_k)$ **for the weighted log-likelihood objective or telemetry**, clamp the relevant softmax probability below a small $\varepsilon_{\log}$ (e.g., `1e-8`) **before** taking logs, to avoid numerical instability
	•	**Bayesian belief updates must use the raw softmax probabilities** $L_k(s\mid\theta_k)$ with **no** clamping; the clamp is **not** part of the generative likelihood for the posterior

⸻

4.7 Pairwise beliefs

For each ordered pair (i,j), i\neq j, maintain:
b_{i\to j,t}\in\Delta_K^\delta

with initialization:
b_{i\to j,0}[k]=1/K

Use:
\Delta_K^\delta=\{b\in\mathbb R^K:\sum_k b[k]=1,\; b[k]\ge \delta\}

**Implementation:** after each Bayes step, beliefs are mapped to $\Delta_K^\delta$ by **Euclidean projection** $\Pi_{\Delta_K^\delta}(x)=\arg\min_{b\in\Delta_K^\delta}\|b-x\|_2^2$ (sorting-based algorithm in `esl/utils/simplex.py`). Floating-point output satisfies $b[k]\ge \delta$ and $\sum_k b[k]=1$ up to machine epsilon.

⸻

4.8 Belief update

If W_{ij,t}=1, compute the **unnormalized** Bayes posterior:
\tilde b_{i\to j,t+1}[k]
=
\frac{
b_{i\to j,t}[k]L_k(s_{ij,t}\mid\theta_{k,t})
}{
\sum_q b_{i\to j,t}[q]L_q(s_{ij,t}\mid\theta_{q,t})
}

using **unclamped** likelihoods $L_k$ from the softmax model.

Then apply **Euclidean projection** onto $\Delta_K^\delta$:
\[
b_{i\to j,t+1}=\Pi_{\Delta_K^\delta}(\tilde b).
\]

If W_{ij,t}=0, keep:
b_{i\to j,t+1}=b_{i\to j,t}

Implementation guardrails:
	•	add small $\varepsilon$ in the Bayes denominator only for numerical stability (`bayes_denominator_eps` in code; not part of the feasible set definition)
	•	require $K\delta\le 1$; when $K\delta=1$ the feasible set is the single point $b[k]=\delta$

⸻

4.9 Prototype learning objective

Population objective:
\mathcal L(\Theta)
=
\mathbb E\left[
\sum_{i\neq j}\sum_{k=1}^K
b_{i\to j}[k]\log L_k(s_{ij}\mid\theta_k)
\right]

In code, use a stochastic batch estimate:
\hat{\mathcal L}_t(\Theta)
=
\sum_{(i,j)\in\text{batch}}\sum_{k=1}^K
W_{ij,t}\,b_{i\to j,t}[k]\log L_k(s_{ij,t}\mid\theta_k)

⸻

4.10 Gradient

Let
p_k = \text{softmax}(\theta_k)
and let e_s denote the one-hot encoding of observed signal s.

Then:
\nabla_{\theta_k}\log L_k(s\mid\theta_k)
=
e_s - p_k

This must be implemented directly and also unit-tested against finite differences.

⸻

4.11 Prototype update

At prototype-update step m:
\theta_{k,m+1}
=
\theta_{k,m}
+
\gamma_m
\sum_{(i,j)\in\text{batch}}
W_{ij}\,b_{i\to j}[k]\,(e_{s_{ij}}-p_k)

Implementation detail:
The belief weights $b_{i\to j}[k]$ used in the gradient must be the
snapshot taken \textbf{before} applying the Bayes update from the
current signal $s_{ij,t}$. This ensures consistency with the stochastic
objective defined in Section 4.9.

Important:
	•	prototype update uses batch observations
	•	beliefs are treated as fixed during that gradient step
	•  By default, gradients should be \textbf{averaged over the batch size}
	   rather than summed. If summation is used instead, the learning rate
	   $\gamma_m$ must be scaled accordingly to keep updates stable across
	   different batch sizes and observability levels.

⸻

4.12 Policy

Recovery mode

Do not use endogenous strategic policies.
Use fixed hidden true policies.

Adaptation mode

Use:
\pi_i(a_i\mid B_t)\propto \exp\{\lambda U_i(a_i\mid B_t)\}

The expected payoff U_i should be computed against the inferred action distributions of currently interacting opponents.

Do not implement adaptation mode until recovery mode is stable.

⸻

5. Interaction protocol

Recovery mode default

At each round:
	1. sample one ordered observer-target pair (i,j) with i \neq j
	2.	agent j emits action according to its true hidden type
	3.	optional: agent i also acts if needed for payoff logging
	4.	sample visibility W_{ij,t}
	5.	if visible, update b_{i\to j,t}
	6.	accumulate batch for prototype update

This protocol must be explicit in code.

Do not use “all-vs-all every round” in v1.

⸻

6. Initialization

Beliefs

Uniform for all ordered pairs.

Prototypes

Must not be initialized identically.

Use:

theta = base + init_noise * np.random.randn(K, num_actions)

Required config fields:
	•	base_init
	•	init_noise

Default:
	•	base_init = 0
	•	init_noise = 0.01

Also support a “symmetric init” flag for failure-case experiments.

⸻

7. Step sizes

Use two counters:
	•	round counter t for belief-scale bookkeeping
	•	prototype-update counter m for slow updates

Recommended defaults:
\alpha_t=(t+1)^{-0.6}
\gamma_m=(m+1)^{-0.9}

If beliefs are updated without explicit damping, still log \alpha_t as the effective fast-scale step.

Prototype updates occur every `prototype_update_every` rounds (slow counter **m** advances only on those steps).

⸻

### Two-timescale update contract (critical implementation detail)

The ESL algorithm follows a **two-timescale** stochastic-approximation structure:

**Fast timescale (beliefs)**  
- Updated on **every** interaction step in which a signal is observed (`W_{ij,t}=1`).  
- Uses likelihoods evaluated at the **current** prototype parameters `\Theta_t` (Bayes / filtered update).  
- `\alpha_t=(t+1)^{-0.6}` is **logged** as the fast-scale bookkeeping rate even when the belief map is undamped full Bayes (no explicit `\alpha_t` in the Bayes formula).

**Slow timescale (prototypes)**  
- Updated **only** on a fixed schedule (batched SGD), not every environment step.  
- Step size `\gamma_m` uses the prototype-update index `m` (increments once per scheduled update).  
- Must be **slower** in aggregate than belief dynamics: large effective belief step every round vs. one prototype step every `M` rounds, and typically smaller `\gamma_m` than the logged `\alpha_t` (tune via `prototype_lr_scale` and `M`).

⸻

#### Implementation rules

**1. Strict update ordering per environment step `t`**

1. Observe signal(s): opponent action `s_{ij,t}=a_j^t` when `W_{ij,t}=1` (and sample `W_{ij,t}`).  
2. **Update beliefs** `B_t \rightarrow B_{t+1}` wherever the signal is observed (likelihoods `L_k(\cdot\mid\theta_{k,t})`).  
3. **Append the batch buffer** with this step’s training tuple.  
4. **Do not** update prototypes in this block.

Prototype parameter updates run **only** in a separate scheduled block (below).

**Batch record vs. §4.9.** The stochastic objective uses weights `b_{i\to j,t}` at the **start** of step `t` for that pair (before the Bayes update from `s_{ij,t}`). Implementation must snapshot `b_{i\to j,t}` **before** applying Bayes for this signal, perform `B_t\to B_{t+1}`, then append the buffer with that frozen `b_{i\to j,t}` so it matches §4.9–§4.11.

**2. Prototype update schedule**

Define `prototype_update_every = M` (default **5**; **10** also common). `M=1` is allowed only for debugging / fast ablations.

- Every `M` environment steps: consume the **accumulated** batch, apply **one** prototype SGD step, increment `m`, clear the batch.  
- If the run ends with a partially filled batch, perform a **final** scheduled prototype update on that remainder (one `m` increment).

**3. Step-size separation**

Target `\gamma_m \ll \alpha_t` in a practical sense:

- Beliefs: full update whenever visible (fast).  
- Prototypes: infrequent updates (`M\ge 2` in real experiments) plus `\gamma_m=(m+1)^{-0.9}` scaled by `prototype_lr_scale` as needed.

Recommended code defaults:

```text
alpha_t = (t+1)**(-0.6)    # logged; implicit if beliefs are pure Bayes
gamma_m = prototype_lr_scale * (m+1)**(-0.9)
```

**4. Frozen-belief assumption during a prototype update**

When applying a prototype step:

- Treat stored `b_{i\to j}` entries in the batch as **constants**.  
- Accumulate `\sum W_{ij}\,b_{i\to j}[k]\,(e_s-\mathrm{softmax}(\theta_k))` over the batch at **current** `\Theta`, then **divide by the batch length** `|B|` (number of environment steps in the buffer) and apply `\Theta \leftarrow \Theta + \gamma_m \cdot (\mathrm{mean})`, matching §4.11. Equivalently: `\Theta \leftarrow \Theta + (\gamma_m/|B|)\cdot(\mathrm{sum})`.

This approximates gradients of the inner objective at `B^\star(\Theta)` in spirit.

⸻

8. Required modules

config.py

Must contain all experiment parameters and seed.

games.py

Implement repeated matrix games and hidden opponent policies.

signals.py

Optional helper for generating observable signals from actions.

beliefs.py

Belief init, Bayes update, Euclidean projection onto $\Delta_K^\delta$.

prototypes.py

Softmax likelihood, gradient, prototype update.

trainer.py

Main training loop, batching, logging, evaluation hooks.

metrics.py

Belief accuracy, entropy, prototype recovery up to permutation, reward stats.

plotting.py

Belief trajectories, prototype trajectories, recovery plots.

⸻

9. Required outputs

For every run save:
	•	config file
	•	random seed
	•	prototype trajectory CSV
	•	belief trajectory CSV
	•	reward trajectory CSV
	•	summary metrics JSON
	•	plots

All outputs must be organized by timestamped run folder.

⸻

10. Required metrics

**Primary recovery metric (MVP):** **matched cross-entropy** — after Hungarian matching of learned prototypes to true behavioral types, sum (or report) cross-entropy between each true type’s action distribution and the matched prototype’s $\mathrm{softmax}(\theta_k)$ (with clipped $q$ inside the CE formula for numerical safety, as in evaluation code).

Recovery metrics (exported under the **canonical field names** below)
	•	`belief_argmax_accuracy` — belief accuracy vs true type (under the current matching)
	•	`belief_entropy_mean` — mean entropy of off-diagonal pairwise beliefs
	•	`matched_cross_entropy` — per-round trajectory; `final_matched_cross_entropy` in run summary JSON
	•	matched **KL** is **optional** and **not** required in MVP exports unless explicitly needed for plots

Dynamics metrics (canonical names)
	•	`prototype_update_norm` — norm of the applied prototype parameter step $\|\gamma_m \bar g\|$ (per scheduled update; logged on the round where an update occurs)
	•	`belief_change_norm` — L1 total variation $\sum_{i,j,k}|b_{i\to j,t+1}[k]-b_{i\to j,t}[k]|$ over the stored belief tensor after vs. before the step (diagonal entries remain zero)
	•	`batch_log_likelihood` — weighted log-likelihood term for the observed step (clamped log, §4.6)

Payoff metrics (summary JSON)
	•	`mean_payoff_per_agent_per_round`
	•	`cumulative_social_payoff`

**Schema:** use the same identifiers in `metrics_trajectory.csv`, `summary_metrics.json`, and plotting code that reads those files.

⸻

11. Mandatory evaluation rule

All prototype comparisons must be computed up to permutation.

Use:
	•	Hungarian matching if convenient
	•	brute-force permutation for small K

This is non-optional.

⸻

12. Mandatory unit tests

Implement at least:
	1.	softmax(theta).sum() == 1
	2.	Bayes update preserves simplex
	3.	Euclidean projection onto $\Delta_K^\delta$ enforces $b[k]\ge \delta$ and normalization (within float tolerance)
	4.	gradient matches finite differences
	5.	symmetric init run does not separate prototypes
	6.	asymmetric init run does separate prototypes
	7.	prototype recovery improves over time in simple 2-type setting

⸻

13. Stopping rule

For MVP use fixed horizon only.

Optional diagnostics:
	•	stop early if prototype parameter change norm stays below threshold for M updates
	•	stop early if belief change norm stays below threshold

But do not rely on early stopping in v1 experiments.

⸻

14. Required first experiments

Experiment 1 — Symmetry preservation
	•	symmetric initialization
	•	show no specialization

Experiment 2 — Recovery under perturbation
	•	tiny asymmetric initialization
	•	50–200 steps
	•	show beliefs sharpen and prototypes split

Experiment 3 — Sparse observability
	•	vary observation probability p_{\text{obs}}
	•	evaluate recovery degradation

Only after these work:

Experiment 4 — Strategic adaptation
	•	ESL vs baseline without prototype learning

⸻

15. Guardrails

Do not add
	•	FCM
	•	neural networks
	•	context/state
	•	more than 2 actions
	•	multiple interacting abstractions at once

Do enforce
	•	stable softmax
	•	deterministic seeds
	•	assertions on all probabilities
	•	permutation-invariant evaluation
	•	explicit experiment mode separation

⸻

16. Cursor implementation brief

Build a minimal Python implementation of Epistemic Social Learning (ESL) for repeated 2-action matrix games. Support two modes: (A) recovery mode with fixed hidden opponent policies and (B) adaptation mode with logit best-response learners, but implement recovery mode first. Use K latent prototypes, each parameterized by action logits theta_k in R^{|A|}. Define signals as observed opponent actions. Use likelihood L_k(s=a | theta_k) = softmax(theta_k)[a]. Maintain pairwise beliefs b_{i->j} over prototypes. Update beliefs with Bayes rule, then **Euclidean projection** onto $\Delta_K^\delta$ (§4.8). Update prototype parameters by stochastic gradient ascent on the belief-weighted batch log-likelihood using gradient e_s - softmax(theta_k). Use sparse or full observability through W_{ij,t}. Use repeated Prisoner’s Dilemma as the first environment. Implement stable softmax, deterministic seeding, permutation-invariant prototype matching, unit tests for gradients and belief updates, and logging of prototype trajectories, belief trajectories, and recovery metrics.

⸻

17. Implementation clarifications (MVP; keep aligned with code)

These statements are **normative for the reference implementation** and avoid ambiguity for tests and reviewers.

1. **Simplex constraint:** After Bayes, beliefs are mapped by **exact Euclidean projection** $\Pi_{\Delta_K^\delta}$ onto $\{b:\sum_k b_k=1,\,b_k\ge\delta\}$ (§4.8). Implementation: `esl/utils/simplex.py` + `esl/beliefs.py`.

2. **Mode B:** MVP adaptation uses **all** logit best-response agents unless extended. Mixed fixed/BR populations are an **optional** implementation note, not a core requirement.

3. **Recovery evaluation:** **Matched cross-entropy** is the **primary** exported recovery distance; a separate matched **KL** field is **optional** and omitted from MVP outputs unless explicitly required.

4. **Likelihood clamping:** **Unclipped** softmax probabilities feed **Bayes** updates. **$\varepsilon$-clamping applies only** when computing $\log L_k$ for the **weighted log-likelihood** telemetry / objective logging, not for the posterior likelihood ratio.

5. **Outputs:** Use **one canonical naming scheme** for metrics across JSON, CSV, and plotting readers (§10). Avoid duplicate aliases in the same artifact.