# ESL target algorithm (theory-aligned)

This document states the **target** mathematical protocol and update rules aligned with the paper-style design. The **running code** may lag; see **`ALGORITHM_CURRENT.md`** for what `run_esl` actually does today.

---

## State

- **Prototypes:** \(\Theta_m = \{\theta_k\}_{k=1}^K\)
- **Beliefs:** \(B_t = \{b_{i\to j,t}\}_{i\neq j}\)
- **Batch:** \(\mathcal{D}_m\) (list of tuples after each observed interaction slot)
- **Global interaction counter:** \(n\) (monotone; typically increments once per processed interaction record appended to the slow buffer)
- **Prototype step index:** \(m\)

---

## Initialization

\(\Theta_0\), uniform \(b_{i\to j,0}\), \(\mathcal{D}_0=\emptyset\), \(n=0\), \(m=0\).

---

## Main loop (rounds \(t=0,1,\ldots\))

### 1. Sample interaction count

\(L_t \sim \mathcal{L}\) supported on \(\{L_{\min},\ldots,L_{\max}\}\). (Degenerate \(L_{\min}=L_{\max}\) needs no RNG draw for \(L_t\) if bit-reproducibility is required.)

### 2. Sample ordered pairs

\(\mathcal{E}_t \subset \{(i,j): i\neq j\}\), \(|\mathcal{E}_t|=L_t\), **without replacement**. Process pairs in a fixed order (e.g. the sample order).

**Debug override:** a single fixed pair forces \(L_t=1\) and \(\mathcal{E}_t=\{(i,j)\}\), ignoring \(L_{\min},L_{\max}\).

### 3. Sequential interaction processing

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

### 4. End-of-round telemetry

Log beliefs / \(\Theta\) / scalars **after** all pairs in \(\mathcal{E}_t\) (see **PRD** / **CURRENT** for CSV shapes).

---

## Synthetic population (evaluation / simulation only)

True parameters \(\Theta^\star=\{\theta_k^\star\}\), latent types \(z_i\sim\mathrm{Categorical}(\rho)\), idiosyncratic noise \(\phi_i\sim\mathcal{D}_{z_i}\), agent logits \(\tilde\theta_i=\theta^\star_{z_i}+\phi_i\). **Not** observed by the learner; used only to generate actions in experiments and to score metrics. Implementations: **`esl/synthetic_population.py`**.

---

## Limiting interpretation (stochastic approximation)

The slow iterate is understood as a **stochastic approximation** driven by the **fast** belief Markov chain. In general, averaging and limits are phrased via **invariant measures** and **differential inclusions**; a **mean-field ODE** is a **special case** under stronger mixing / uniqueness assumptions—not the general claim.

---

## Metrics (evaluation)

- **MCE:** \(\min_{\sigma\in S_K}\frac{1}{K}\sum_k \mathrm{CE}\bigl(\pi^\star_k \,\|\, \pi_{\sigma(k)}(\Theta)\bigr)\) over softmax policies (Hungarian in code).
- **Beliefs vs \(z_j\):** CE\((e_{z_j}, b_{i\to j})\), KL\((e_{z_j}\| b_{i\to j})\), argmax accuracy (permutation-aligned where applicable).
