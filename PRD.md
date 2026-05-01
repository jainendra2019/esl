PRD: Epistemic Social Learning (ESL) MVP — Senior-Reviewed Version

1. Goal

Implement a minimal, correct, interpretable version of Epistemic Social Learning (ESL) as a **feedback-coupled latent-structure learning system** in repeated multi-agent interaction.

The MVP should demonstrate:

	1.	Agents maintain **pairwise beliefs over latent behavioral prototypes**.
	2.	Beliefs update via **Bayesian filtering under endogenous data** (observations drawn from policies that depend on the current ESL state).
	3.	Prototype parameters update via **belief-weighted stochastic gradient ascent** on a likelihood-based objective estimated from interaction batches.
	4.	**Data is endogenously generated**: beliefs → actions → observations → belief updates → (scheduled) prototype updates.
	5.	The system exhibits **emergent specialization of prototypes** under initialization asymmetry (recovery) and, in adaptation mode, strategic behavior under beliefs.

This is **not** a standard EM / i.i.d. latent-variable setting: observations are not exogenous and identically distributed; latent assignments and beliefs affect **future** data; learning is a **closed-loop stochastic process**.


This MVP is for validation of the core algorithm, debugging, first recovery plots, and the empirical section of the paper. It is not the final scalable implementation.

Validation standard for this PRD 
	•	The implementation phase must remain strictly faithful to the current paper draft, especially the two-timescale ordering, the belief-update semantics, the frozen-batch prototype update schedule, the recovery/adaptation experiment split, and the canonical evaluation metrics already defined in this document and in ALGORITHM.md.
	•	Development must be **test-driven first**: no milestone is considered complete until its required tests pass and its declared artifacts are generated.
	•	Execution must be **gated by milestone confirmation**: after each milestone, Cursor must stop, summarize status, report generated deliverables, list any deviations or open issues, and wait for explicit confirmation before proceeding to the next milestone.
	•	All experiment outputs intended for the paper must ultimately be collected into a single reproducible manuscript bundle directory (see new bundling section below).

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
	•	explicit **two-timescale** coupling: fast beliefs vs. slow prototypes (see §8)

Out of scope
	•	fuzzy clustering / FCM
	•	contextual/state-dependent environments
	•	neural likelihood models
	•	continuous actions
	•	full decentralized multi-learner training
	•	theorem verification code
	•	performance optimization

⸻

3. Core system property: feedback-coupled learning

Unlike classical latent-variable models with exogenous data:

- **Observations are not fixed in distribution**; the data-generating process depends on the current beliefs, policies, and (in adaptation mode) best responses.
- The **sampling distribution of signals evolves** with the system state $(B_t,\Theta_t)$.

At each environment step the intended causal chain is:

**beliefs → actions → observations → belief updates → (scheduled) prototype updates**

This induces **non-stationarity**, **Markovian dependence** across rounds, and **state-dependent noise** in stochastic gradient estimates for $\Theta$.

**Implementation requirement:** the reference code **must** respect this ordering (snapshot beliefs before Bayes on the current signal; use current $\Theta_t$ in likelihoods for the belief update; apply prototype steps only on the slow schedule; average gradients over full batch length $|B|$ including $W=0$ rows—see §5.11 and §8).

⸻

4. Two supported experiment modes

Mode A — Recovery mode (required first)

Purpose: verify prototype recovery cleanly.
	•	Data-generating agents use fixed hidden policies (not updated during training).
	•	Those policies are **exogenous to the learner’s parameter updates**, but the **ESL state** $(B_t,\Theta_t)$ still evolves in closed loop: sampled pairs, actions, and sparse observations drive Bayes and batched prototype steps.
	•	ESL learner(s) observe opponents and learn prototypes
	•	Recommended hidden types:
	•	Always Cooperate
	•	Always Defect
	•	Tit-for-Tat (optional later)

This is the first mode to implement.

⸻

Mode B — Strategic adaptation mode (second)

Purpose: test whether ESL improves payoff via adaptation.
	•	ESL agents choose actions via logit best response using **current** beliefs and $\Theta_t$ — actions (and thus observations) are **fully endogenous** to the ESL state.
	•	Recovery metrics are still tracked; payoff becomes a main metric

**MVP default:** all agents use logit best response (homogeneous adaptation). Mixed populations (some fixed hidden opponents, some BR learners) are a **possible extension** via a mask or role flags; they are **not** required for MVP correctness and need not appear in the core PRD checklist.

⸻

5. Mathematical specification

5.1 Action space

Use a finite 2-action set:
\mathcal A = \{0,1\}

Recommended semantic mapping:
	•	0 = Cooperate
	•	1 = Defect

⸻

5.2 Signal space

For MVP:
\mathcal S = \mathcal A
and
s_{ij,t} = a_j^t
whenever W_{ij,t}=1.

This means the observed signal is the observed opponent action.

⸻

5.3 Context

For MVP:
\omega_t \equiv \text{constant}
and should not appear in code except possibly as a placeholder.

Do not build a context/state system in v1.

⸻

5.4 Observability

For each ordered pair (i,j):
W_{ij,t}\in\{0,1\}

Interpretation:
	•	W_{ij,t}=1: agent i observes j’s action
	•	W_{ij,t}=0: no signal is available

Support two modes:
	1.	full observation
	2.	Bernoulli sparse observation with parameter p_{\text{obs}}

⸻

5.5 Prototype parameters

Each prototype k\in\{1,\dots,K\} is represented by:
\theta_k\in\mathbb R^{|\mathcal A|}

For 2 actions:
\theta_k = (\theta_{k,0},\theta_{k,1})

These are logits, not probabilities.

⸻

5.6 Likelihood model

Define the prototype likelihood:
L_k(s=a\mid\theta_k)
=
\frac{\exp(\theta_{k,a})}{\sum_{a'}\exp(\theta_{k,a'})}

Implementation requirement:
	•	use numerically stable softmax
	•	when computing $\log L_k(s\mid\theta_k)$ **for the weighted log-likelihood objective or telemetry**, clamp the relevant softmax probability below a small $\varepsilon_{\log}$ (e.g., `1e-8`) **before** taking logs, to avoid numerical instability
	•	**Bayesian belief updates must use the raw softmax probabilities** $L_k(s\mid\theta_k)$ with **no** clamping; the clamp is **not** part of the generative likelihood for the posterior

⸻

5.7 Pairwise beliefs

For each ordered pair (i,j), i\neq j, maintain:
b_{i\to j,t}\in\Delta_K^\delta

with initialization:
b_{i\to j,0}[k]=1/K

Use:
\Delta_K^\delta=\{b\in\mathbb R^K:\sum_k b[k]=1,\; b[k]\ge \delta\}

**Implementation:** after each Bayes step, beliefs are mapped to $\Delta_K^\delta$ by **Euclidean projection** $\Pi_{\Delta_K^\delta}(x)=\arg\min_{b\in\Delta_K^\delta}\|b-x\|_2^2$ (sorting-based algorithm in `esl/utils/simplex.py`). Floating-point output satisfies $b[k]\ge \delta$ and $\sum_k b[k]=1$ up to machine epsilon.

⸻

5.8 Belief update

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

5.9 Prototype learning objective

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

**Interpretation (not classical EM):** this is **not** a fixed log-likelihood under i.i.d. latent assignments. The distribution of $(i,j,s_{ij,t},W_{ij,t})$ in the batch is **state-dependent**, induced by current beliefs $B_t$, the sampling of interaction pairs, and (in adaptation mode) policies that depend on $B_t$ and payoffs. Stochastic gradient estimates for $\Theta$ are therefore **biased** finite-sample SA directions; in the two-timescale regime they are understood to track a **mean-field drift** (ODE limit) rather than an exact score of a static marginal likelihood.

⸻

5.10 Gradient

Let
p_k = \text{softmax}(\theta_k)
and let e_s denote the one-hot encoding of observed signal s.

Then:
\nabla_{\theta_k}\log L_k(s\mid\theta_k)
=
e_s - p_k

This must be implemented directly and also unit-tested against finite differences.

⸻

5.11 Prototype update

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
objective defined in Section 5.9.

Important:
	•	prototype update uses batch observations
	•	beliefs are treated as fixed during that gradient step
	•  By default, gradients should be \textbf{averaged over the batch size}
	   rather than summed. If summation is used instead, the learning rate
	   $\gamma_m$ must be scaled accordingly to keep updates stable across
	   different batch sizes and observability levels.

⸻

5.12 Policy

Recovery mode

Do not use endogenous strategic policies.
Use fixed hidden true policies.

Adaptation mode

Use:
\pi_i(a_i\mid B_t)\propto \exp\{\lambda U_i(a_i\mid B_t)\}

The expected payoff U_i should be computed against the inferred action distributions of currently interacting opponents.

Do not implement adaptation mode until recovery mode is stable.

⸻

6. Interaction protocol

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

7. Initialization

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

8. Step sizes

Use two counters:
	•	round counter t for belief-scale bookkeeping
	•	prototype-update counter m for slow updates

Recommended defaults:
\alpha_t=(t+1)^{-0.6}
\gamma_m=(m+1)^{-0.9}

If beliefs are updated without explicit damping, still log \alpha_t as the effective fast-scale step.

Prototype updates occur every **`Q` interaction events** (`prototype_update_every` after `validate()`; see §19). When `interaction_pairs_min = interaction_pairs_max = 1`, this matches one interaction per environment round, i.e. every `Q` rounds. Slow counter **m** advances only on those steps.

⸻

### Two-timescale update contract (critical implementation detail)

The ESL algorithm follows a **two-timescale** stochastic-approximation structure. **Formal contract:**

- **Fast process:** pairwise beliefs $B$ are updated on every step where learning is enabled and an observation occurs (`W_{ij,t}=1$), using likelihoods at **current** $\Theta_t$.
- **Slow process:** prototype parameters $\Theta$ are updated **only** every $M$ environment steps (batched SGD), not every step.
- **Constraint:** prototype dynamics must be **strictly slower** than belief dynamics in aggregate (typically $M\ge 2$, tuned step sizes). This separation supports the standard SA / ODE tracking interpretation of the slow iterate.

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

**Batch record vs. §5.9.** The stochastic objective uses weights `b_{i\to j,t}` at the **start** of step `t` for that pair (before the Bayes update from `s_{ij,t}`). Implementation must snapshot `b_{i\to j,t}` **before** applying Bayes for this signal, perform `B_t\to B_{t+1}`, then append the buffer with that frozen `b_{i\to j,t}` so it matches §5.9–§5.11.

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
- Accumulate `\sum W_{ij}\,b_{i\to j}[k]\,(e_s-\mathrm{softmax}(\theta_k))` over the batch at **current** `\Theta`, then **divide by the batch length** `|B|` (number of **interaction steps** in the buffer, i.e. environment rounds accumulated—not the count of $W_{ij}=1$ rows) and apply `\Theta \leftarrow \Theta + \gamma_m \cdot (\mathrm{mean})`, matching §5.11. Equivalently: `\Theta \leftarrow \Theta + (\gamma_m/|B|)\cdot(\mathrm{sum})`.

**IMPORTANT:** Unobserved steps (`W_{ij}=0`) contribute **zero** to the gradient numerator but **still increment** $|B|$ in the denominator. This matches two-timescale stochastic-approximation scaling under sparse observability.

This approximates gradients of the inner objective at `B^\star(\Theta)` in spirit.

⸻


8A. Baseline integration policy (official code first, minimal reimplementation)

For the paper experiments, baseline comparison should prefer official or author-released implementations when they are available and compatible with the ESL evaluation protocol. The purpose is to reduce implementation error, preserve fidelity to published methods, and minimize unnecessary engineering effort.

Baseline policy:
- Use official / author-released code where feasible.
- Wrap external baselines behind a thin adapter layer rather than rewriting them from scratch.
- Reimplement a baseline only if: (i) no usable code is available, (ii) the released code is fundamentally incompatible with the ESL matrix-game setting, or (iii) the baseline is intentionally a simple reference model (e.g., FCM/GMM).
- All baseline runs must be logged under the same experiment manifest structure as ESL, with explicit notes on any deviations from the original codebase or training protocol.

Canonical baseline sources for this project:
1. Independent PPO
   - Preferred source: https://github.com/nikhilbarhate99/PPO-PyTorch
   - Use as a lightweight policy-gradient baseline without opponent modeling.
   - Integrate via a thin wrapper in the project rather than copying logic into ESL core modules.

2. M-FOS
   - Preferred source: ICML 2022 supplementary archive at https://media.icml.cc/Conferences/ICML2022/supplementary/lu22d-supp.zip
   - Treat the released src/ as the primary reference implementation.
   - Port or wrap only the minimal pieces required to run fair comparisons in the repeated matrix-game setting.

3. MBOM
   - Preferred source: https://github.com/PKU-RL/MBOM
   - Use the official codebase as the primary implementation reference.
   - If the full environment-model stack is too tightly coupled to original environments, isolate the opponent-modeling logic and document any simplifications explicitly.

4. Simple Opponent Model
   - Preferred source: https://github.com/hhexiy/opponent
   - Use this as the main external reference for a history-based opponent-modeling baseline.
   - If exact integration is not feasible, implement the closest faithful reduced version and document the gap in the experiment manifest.

5. FCM/GMM Clustering
   - These should be implemented in-project.
   - Rationale: they are simple static clustering baselines, easy to validate directly, and do not require external code.

Repository placement and adapter contract:
- Place third-party baselines under a dedicated directory such as third_party/ or external/ with clear subfolders per baseline.
- Do not merge third-party code into ESL core modules.
- Create project-local adapters under esl/baselines/ that expose a common interface for:
  - train(...)
  - evaluate(...)
  - export_summary(...)
  - export_predictions(...) if applicable
- Every adapter must translate baseline outputs into the canonical ESL metric schema (MCE, payoff, belief-related metrics when meaningful, runtime, and manifest metadata).

Fairness and protocol requirements for external baselines:
- Use matched horizons, matched seeds where possible, and matched interaction logs or environment budgets.
- Keep the game definitions, observability masks, and evaluation populations identical to ESL.
- If a baseline cannot naturally support a regime (e.g., belief-conditioned or adaptive setting), the run must be skipped explicitly and the reason recorded.
- Never silently replace an official baseline with a custom variant without documenting it in the manifest and report.

Test-first requirement for baseline onboarding:
Before any large sweep is run, each external baseline must pass a small onboarding checklist:
1. repository snapshot / source path recorded;
2. adapter smoke test passes on one tiny run;
3. outputs can be converted to the canonical experiment schema;
4. runtime and seed are logged;
5. any incompatibilities with the ESL setting are documented.

The goal of this policy is not full benchmark breadth, but faithful, low-risk comparison against the agreed baseline set with minimum avoidable implementation error.

9. Required modules

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

experiment_registry.py

Experiment registry, canonical manifests, and helpers for fair baseline comparison.

baseline adapters under esl/baselines/

Thin wrappers exposing a common interface for ESL-compatible baseline training, evaluation, and summary export.

third-party baseline sources under third_party/ or external/

Pinned external baseline code snapshots kept separate from ESL core modules.
⸻

10. Required outputs

For every run save:
	•	config file
	•	random seed
	•	prototype trajectory CSV
	•	belief trajectory CSV
	•	reward trajectory CSV
	•	summary metrics JSON
	•	plots
	•	baseline manifest JSON for every external baseline run
	•	source provenance record (URL, commit/hash if available, local patch note)
	•	canonical summary export compatible with ESL experiment aggregation
	•	milestone status report markdown after each completed milestone
	•	manuscript-bundle-ready copies of final figures/tables/manifests

All outputs must be organized by timestamped run folder.

⸻

11. Required metrics

**Primary recovery metric (MVP):** **matched cross-entropy** — after Hungarian matching of learned prototypes to true behavioral types, sum (or report) cross-entropy between each true type’s action distribution and the matched prototype’s $\mathrm{softmax}(\theta_k)$ (with clipped $q$ inside the CE formula for numerical safety, as in evaluation code).

**Offline baselines (comparison only):** The package **`esl/baselines/`** fits static K-means, fuzzy c-means, and Bernoulli mixture EM on an optional **`interaction_observations.csv`** log (same `w` mask as ESL; no sequential beliefs). Same **MCE** metric via `esl.metrics.mce_value`. Fairness statement and feature definitions: **`docs/baselines/BASELINE_PROTOCOL.md`**.

Recovery metrics (exported under the **canonical field names** below)
	•	`belief_argmax_accuracy` — belief accuracy vs true type (under the current matching)
	•	`belief_entropy_mean` — mean entropy of off-diagonal pairwise beliefs
	•	`matched_cross_entropy` — per-round trajectory; `final_matched_cross_entropy` in run summary JSON
	•	matched **KL** is **optional** and **not** required in MVP exports unless explicitly needed for plots

Dynamics metrics (canonical names)
	•	`prototype_update_norm` — norm of the applied prototype parameter step $\|\gamma_m \bar g\|$ (per scheduled update; logged on the round where an update occurs)
	•	`belief_change_norm` — L1 total variation $\sum_{i,j,k}|b_{i\to j,t+1}[k]-b_{i\to j,t}[k]|$ over the stored belief tensor after vs. before the step (diagonal entries remain zero)
	•	`batch_log_likelihood` — weighted log-likelihood term for the observed step (clamped log, §5.6)

Payoff metrics (summary JSON)
	•	`mean_payoff_per_agent_per_round`
	•	`cumulative_social_payoff`

**Schema:** use the same identifiers in `metrics_trajectory.csv`, `summary_metrics.json`, and plotting code that reads those files.

⸻

12. Mandatory evaluation rule

All prototype comparisons must be computed up to permutation.

Use:
	•	Hungarian matching if convenient
	•	brute-force permutation for small K

This is non-optional.

⸻

13. Mandatory unit tests

Implement at least:
	1.	softmax(theta).sum() == 1
	2.	Bayes update preserves simplex
	3.	Euclidean projection onto $\Delta_K^\delta$ enforces $b[k]\ge \delta$ and normalization (within float tolerance)
	4.	gradient matches finite differences
	5.	symmetric init run does not separate prototypes
	6.	asymmetric init run does separate prototypes
	7.	prototype recovery improves over time in simple 2-type setting
	8.	experiment manifest schema validates for ESL and baseline runs
	9.	milestone artifact bundle is created for a smoke-test run
	10.	freeze-Theta self-consistency runner preserves prototype parameters exactly during rollout

⸻

13A. Milestone-gated, test-driven execution protocol

The implementation and experimental phase must proceed in strictly gated milestones.

Milestone rule:
	•	Each milestone begins with tests or validation checks for the functionality being added.
	•	Each milestone must produce a short status report including:
		•	what was implemented,
		•	which tests passed or failed,
		•	which artifacts were produced,
		•	any deviations from the PRD or paper,
		•	recommended next step.
	•	After producing that report, Cursor must stop and wait for explicit user confirmation before proceeding.
	•	If a milestone fails tests or produces ambiguous outputs, the next milestone must not begin.

Required milestone report fields:
	•	Milestone name
	•	Status: PASS / PARTIAL / FAIL
	•	Files added or changed
	•	Tests executed
	•	Artifacts generated
	•	Open issues / caveats
	•	Paper-faithfulness check
	•	Ready-for-next-step recommendation

This gating rule is mandatory for both ESL core functionality and baseline onboarding.

14. Stopping rule

For MVP use fixed horizon only.

Optional diagnostics:
	•	stop early if prototype parameter change norm stays below threshold for M updates
	•	stop early if belief change norm stays below threshold

But do not rely on early stopping in v1 experiments.

⸻


14A. Baseline implementation plan for the validation phase

The experimental phase should not begin with from-scratch baseline engineering. Instead, use the following staged onboarding plan:

Stage B1 — Baseline acquisition
- Create a dedicated baseline source directory (e.g., third_party/).
- Add subdirectories or pinned source snapshots for:
  - PPO-PyTorch
  - M-FOS supplementary src
  - MBOM official code
  - opponent-model reference code
- Record source URLs, commit hashes if available, and local patch notes.

Stage B2 — Adapter wrappers
- Implement thin wrappers in esl/baselines/ for each external baseline.
- Wrappers must expose a common runner interface compatible with the ESL experiment registry and output schema.
- Wrappers must not change ESL core trainer semantics.

Stage B3 — Smoke-test comparison
- For each baseline, run one tiny recovery-mode job in a repeated 2-action matrix game.
- Confirm that:
  - training executes end-to-end or fails with an explicit incompatibility reason,
  - outputs can be parsed,
  - summary metrics are emitted,
  - runtime is logged.

Stage B4 — Fairness validation
- Verify horizon matching, evaluation population matching, and observability-mask matching against ESL.
- Verify that baseline-specific hyperparameters are recorded separately from ESL hyperparameters.
- If the official code cannot be used directly, freeze a minimal adapted version and document the exact deviations.

Only after Stages B1--B4 are complete should full paper sweeps be launched.

14B. Manuscript bundle requirement

At the end of the experimental phase, all paper-facing artifacts must be collected into a single reproducible manuscript bundle.

Create a directory of the form:
- manuscript_bundle/

Required contents:
- manuscript_bundle/figures/                # final paper figures in PNG/PDF
- manuscript_bundle/tables/                 # CSV + LaTeX-ready tables
- manuscript_bundle/configs/                # frozen configs used for paper runs
- manuscript_bundle/manifests/              # manifests for all included runs
- manuscript_bundle/metrics/                # aggregated CSV/JSON summaries
- manuscript_bundle/reports/                # milestone reports + final experiment report
- manuscript_bundle/provenance/             # source URLs, commits/hashes, local patch notes
- manuscript_bundle/README.md               # how to regenerate the bundle
- manuscript_bundle/EXPERIMENT_REPORT.md    # final summary of results and caveats

Bundling rules:
	•	Only include artifacts that correspond to the final paper protocol or explicitly labeled appendix-only analyses.
	•	Every figure in the bundle must be traceable to a manifest and a config.
	•	Every table in the bundle must be reproducible from saved metrics files.
	•	If a figure depends on manual curation, that must be recorded explicitly in the bundle README.
	•	The bundle must be sufficient for manuscript assembly without hunting across ad hoc run folders.

15. Required first experiments

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

16. Guardrails

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
	•	official baseline code first where available
	•	thin wrappers instead of deep rewrites for external baselines
	•	explicit documentation of baseline incompatibilities or adaptations
	•	matched evaluation protocol across ESL and baselines
	•	stop-after-milestone reporting and explicit confirmation before continuation
	•	manuscript bundle assembly as a first-class deliverable, not an afterthought
	•	paper-faithfulness check in every milestone report

⸻

17. Cursor implementation brief

Build and maintain a minimal Python implementation of Epistemic Social Learning (ESL) for repeated 2-action matrix games, and complete the experiment-validation layer around it. Support two modes: (A) recovery mode with fixed hidden opponent policies and (B) adaptation mode with logit best-response learners, but keep recovery mode as the first correctness target. Use K latent prototypes, each parameterized by action logits theta_k in R^{|A|}. Define signals as observed opponent actions. Use likelihood L_k(s=a | theta_k) = softmax(theta_k)[a]. Maintain pairwise beliefs b_{i->j} over prototypes. Update beliefs with Bayes rule, then Euclidean projection onto $\Delta_K^\delta$ (§5.8). Update prototype parameters by stochastic gradient ascent on the belief-weighted batch log-likelihood using gradient e_s - softmax(theta_k). Use sparse or full observability through W_{ij,t}. Use repeated Prisoner’s Dilemma as the first environment. Implement stable softmax, deterministic seeding, permutation-invariant prototype matching, unit tests for gradients and belief updates, and logging of prototype trajectories, belief trajectories, and recovery metrics. For baseline comparison, prefer official or author-released implementations where available, integrate them under a dedicated third_party/ (or equivalent) folder, and expose them through thin adapters in esl/baselines/ rather than reimplementing complex baselines from scratch. Work in milestone-gated fashion: after each milestone, emit a concise status report with tests, outputs, caveats, and paper-faithfulness notes, then stop for user confirmation before moving on. Ensure all final paper artifacts are collected into a manuscript_bundle/ directory.

⸻

18. Implementation clarifications (MVP; keep aligned with code)

These statements are **normative for the reference implementation** and avoid ambiguity for tests and reviewers.

1. **Simplex constraint:** After Bayes, beliefs are mapped by **exact Euclidean projection** $\Pi_{\Delta_K^\delta}$ onto $\{b:\sum_k b_k=1,\,b_k\ge\delta\}$ (§5.8). Implementation: `esl/utils/simplex.py` + `esl/beliefs.py`.

2. **Mode B:** MVP adaptation uses **all** logit best-response agents unless extended. Mixed fixed/BR populations are an **optional** implementation note, not a core requirement.

3. **Recovery evaluation:** **Matched cross-entropy** is the **primary** exported recovery distance; a separate matched **KL** field is **optional** and omitted from MVP outputs unless explicitly required.

4. **Likelihood clamping:** **Unclipped** softmax probabilities feed **Bayes** updates. **$\varepsilon$-clamping applies only** when computing $\log L_k$ for the **weighted log-likelihood** telemetry / objective logging, not for the posterior likelihood ratio.

5. **Outputs:** Use **one canonical naming scheme** for metrics across JSON, CSV, and plotting readers (§11). Avoid duplicate aliases in the same artifact.

⸻

## 19. Addendum — Target protocol, predictive policies, synthetic evaluation, SA language (v2)

This addendum aligns the **product spec** with the paper-style target. The **reference code** may implement a strict subset. **Single algorithm reference:** **`ALGORITHM.md`** — (*Current implementation* = what `run_esl` does), (*Theory-aligned target* = paper-style math), (*Pseudocode (Cormen-style)* = consolidated procedures and flagship block). There is **no** separate pseudocode file. **NeurIPS-style recovery presets, sweep aggregation, and plotting** live under **`esl/experiments/`** and **`esl/plot_neurips.py`** (see **README**); they reuse `run_esl` and the reporting fields described in **ALGORITHM.md** (*Run outputs*).

### 19.1 Target interaction protocol

Each **environment round** \(t\):

1. Sample \(L_t \in \{L_{\min},\ldots,L_{\max}\}\) (degenerate case: constant \(L_t\) without extra RNG if required for reproducibility).
2. Sample \(\mathcal{E}_t \subset \{(i,j): i\neq j\}\) with \(|\mathcal{E}_t|=L_t\) **without replacement**.
3. For each \((i,j)\in\mathcal{E}_t\) **in order**: act, observe \(W_{ij,t}\), snapshot belief, Bayes + projection, append slow-batch record, increment global interaction counter \(n\).
4. **Prototype update** after every **\(Q\)** interaction events (not “every \(M\) rounds” when \(L_t\) varies). Config: `prototype_update_every` holds \(Q\) after `validate()`; optional `prototype_update_every_interactions` overwrites it.

If **`force_ordered_pair`** is set (debug): \(L_t=1\) and \(\mathcal{E}_t\) is that pair only; **`interaction_pairs_min` / `max` are ignored** for that run.

### 19.2 Belief-dependent policy (adaptation mode)

Predictive opponent distribution:
\[
\hat\pi_{j\mid i,t}(a_j)=\sum_{k=1}^K b_{i\to j,t}[k]\,\pi_k(a_j\mid\theta_k),
\quad \pi_k=\mathrm{softmax}(\theta_k).
\]
Expected utility for candidate \(a_i\): \(\sum_{a_j}\hat\pi_{j\mid i,t}(a_j)\,u_i(a_i,a_j)\). Logit best response:
\[
\pi_i(a_i\mid b,\Theta)\propto \exp\{\lambda\, U_i(a_i)\}.
\]
**Implementation:** `esl.trainer.marginal_opponent_probs` implements \(\hat\pi\) as \(b_{i\to j}^\top \sigma(\Theta)\).

### 19.3 Synthetic population and evaluation truth (never for learning)

For **simulation and metrics only**:
\[
z_i\sim\mathrm{Categorical}(\rho),\qquad \phi_i\sim\mathcal{D}_{z_i},\qquad \tilde\theta_i=\theta^\star_{z_i}+\phi_i.
\]
Actions in the simulator may draw from \(\mathrm{softmax}(\tilde\theta_i)\). **\(\Theta^\star\), \(z\), \(\phi\) are not observed** by the ESL learner and **must not** appear in `run_esl`’s learning path. Module: **`esl/synthetic_population.py`** (do **not** import from `esl.trainer`).

### 19.4 Evaluation metrics (names)

- **MCE (matched cross-entropy):**  
  \(\displaystyle \mathrm{MCE}(\Theta,\Theta^\star)=\min_{\sigma\in S_K}\frac{1}{K}\sum_{k=1}^K \mathrm{CE}\bigl(\pi^\star_k \,\|\, \mathrm{softmax}(\theta_{\sigma(k)})\bigr)\)  
  (implementation: Hungarian on CE cost matrix; see `esl.metrics.match_prototypes_to_types` / `mce_value`).
- **Beliefs vs true type \(z_j\)** (evaluation): cross-entropy \(\mathrm{CE}(e_{z_j}, b_{i\to j})\), KL\((e_{z_j}\| b_{i\to j})\), argmax accuracy (Hungarian-aligned where used).

### 19.5 Theoretical interpretation (wording)

The system is a **two-timescale stochastic approximation** with **Markovian** fast dynamics (beliefs). **Limiting** behavior should be described, in general, via **invariant measures** of the fast process and an **averaged drift** / **differential inclusion** for the slow parameter; a **mean-field ODE** is appropriate **only** under **stronger** mixing or uniqueness assumptions—not as the universal story.
