"""Oracle upper bound: empirical action distribution per true type (labeled agents)."""

from __future__ import annotations

import numpy as np

from esl.baselines.dataset import RunDataset, observed_rows


def oracle_logits_from_true_types(
    ds: RunDataset,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    For each true type k, pool w>0 observations where target j has true_types[j]==k.
    Returns (K, 2) logits [log p(C), log p(D)] matching softmax parameterization for mce_value.
    """
    k = ds.num_prototypes
    n_actions = ds.num_actions
    if n_actions != 2:
        raise ValueError("oracle v1 supports 2 actions only")
    counts = np.zeros((k, 2), dtype=np.float64)
    for r in observed_rows(ds):
        t = int(ds.true_types[r.j])
        counts[t, int(r.a_j)] += 1.0
    logits = np.zeros((k, 2), dtype=np.float64)
    for t in range(k):
        tot = counts[t].sum()
        if tot <= 0:
            logits[t, :] = np.log(np.array([0.5, 0.5]))
            continue
        p = (counts[t] + eps) / (tot + 2 * eps)
        logits[t, 0] = np.log(p[0])
        logits[t, 1] = np.log(p[1])
    return logits
