# Figure 2 Ablation Report

- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`; horizon: `1000`; H: `100`.
- Error bars: 95% CI over seed-task cells.
- Gate: PASS.

## Caption
Figure 2: Component-level analysis under behavioral shifts. Removing shared structure, belief conditioning, or prototype learning increases post-switch adaptation cost. ESL achieves the lowest prediction and adaptation error. Methods that achieve low latent identification error (e.g., EM) do not necessarily adapt well, showing that identification alone is insufficient under endogenous interaction.

## PSR
- ESL vs No-Sharing: `101.2683` vs `129.2349`.
- ESL vs Fixed-K Bayesian: `101.2683` vs `132.8683`.
- ESL vs No-Belief: `101.2683` vs `150.3349`.
- ESL vs Online EM: `101.2683` vs `153.9016`.

## Mechanism
- ESL has the lowest DPR: `73.9118`.
- EM low LTE but high DPR: `1.0986` LTE, `86.9729` DPR.
- Removing any component worsens PSR relative to full ESL.
- No-sharing degrades PSR relative to ESL, supporting the necessity of shared structure.
- Identification alone is insufficient under endogenous interaction.
