# Milestone 2B — FCM / static clustering finalization

This milestone finalizes **in-project offline static clustering** only:

- **K-means** on conditional-cooperate features (`esl.baselines.kmeans_actions`) — bundle: [`../milestone_2b_kmeans/`](../milestone_2b_kmeans/MILESTONE_STATUS.md)
- **FCM** (fuzzy c-means, `esl.baselines.fuzzy_cmeans`) — bundle: [`../milestone_2b_fcm/`](../milestone_2b_fcm/MILESTONE_STATUS.md)

**Out of scope for 2B:** M-FOS, MBOM, PPO (Milestone 2A), and **EM / mixture (“GMM”)** baselines — those remain separate onboarding items.

Canonical exports: `esl.baselines.clustering_smoke_export` (manifest, provenance, `summary_metrics.json` with numeric **MCE**, trajectory stub, optional `clustering_logits.json`).
