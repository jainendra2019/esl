# Figure 1 Dynamic Adaptation Report

- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`; horizon: `1000`; H: `100`.
- Full gate: PASS.
- Games: IPD, Stag Hunt, Matching Pennies; Panel (c) reports early gap-to-oracle AUC across the same settings.

## PSR Comparison
- ESL vs PPO: `win` (`96.7016` vs `268.9349`).
- ESL vs FP: `win` (`96.7016` vs `187.8683`).
- ESL vs SOM: `win` (`96.7016` vs `159.9349`).
- ESL vs Fixed-K Bayesian: `win` (`96.7016` vs `131.4683`).
- ESL vs Online EM: `win` (`96.7016` vs `149.5683`).

## DPR Comparison
- ESL vs PPO: `win` (`74.2763` vs `88.7277`).
- ESL vs FP: `win` (`74.2763` vs `103.0335`).
- ESL vs SOM: `win` (`74.2763` vs `85.5026`).
- ESL vs Fixed-K Bayesian: `win` (`74.2763` vs `102.0808`).
- ESL vs Online EM: `win` (`74.2763` vs `86.9729`).

## LTE Comparison
- ESL: `3.2892`.
- PPO: `N/A`.
- FP: `N/A`.
- SOM: `N/A`.
- Fixed-K Bayesian: `3.6441`.
- Online EM: `1.0986`.
- Non-latent methods are marked N/A for latent identification because they do not maintain type beliefs.
- Online EM LTE is reported as a responsibility-based diagnostic, not as the primary mechanism claim, because its i.i.d. clusters need not correspond cleanly to ESL/Fixed-K latent prototypes.
- Interpretation: low identification error alone is not sufficient for adaptation; Online EM can look strong on LTE while remaining worse on decision-relevant prediction regret and post-switch oracle regret.

## PPO Audit
- Source: `official_ppo_pytorch`.
- Total PPO update calls across included cells: `1860`.
- Action path: `official_select_action`; the old uniform fallback is not used.
- Observation context: `constant_no_regime_info`; no true type, switch label, oracle payoff, or future information is provided.
- PPO uses the official PyTorch implementation under the same interaction protocol; since PPO has no explicit opponent model, DPR is computed using a uniform predictive prior and should be interpreted only as a no-modeling prediction baseline.

## Interpretation
- Panel (a): ESL reduces post-switch oracle regret.
- Panel (b1): EM may identify latent type well, but identification alone is not sufficient.
- Panel (b2): ESL has lowest decision-relevant prediction regret.
- Panel (c): early adaptation gap summarizes recovery immediately after switches.
- PPO: official PPO-PyTorch policy-gradient learner used through `select_action`; it has no explicit opponent model, so prediction fields are marked not applicable and are used only to preserve the shared metrics schema.
- FP: flat belief over opponent actions.
- SOM: history-based model without shared latent structure.
- Fixed-K Bayesian: fixed misspecified prototypes, no prototype learning.
- Online EM: sliding-window latent clustering under an i.i.d. assumption; useful for identification but not decision-coupled adaptation.
- ESL: learned reusable latent behavioral structure under endogenous interaction.

ESL reduces post-switch adaptation cost by leveraging learned latent structure under endogenous interaction.
