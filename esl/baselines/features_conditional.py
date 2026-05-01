"""Per-agent conditional cooperate rates: P(a_j=C|a_i=C), P(a_j=C|a_i=D) from w>0 rows."""

from __future__ import annotations

import numpy as np

from esl.baselines.dataset import RunDataset, observed_rows


def conditional_coop_features(
    ds: RunDataset,
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each agent j (as observed target), empirical P(C|opp C) and P(C|opp D) with Laplace smoothing.
    Action 0 = cooperate (games.ACTION_COOPERATE).

    Returns:
        phi: (N, 2) with columns [p_c_given_ai_c, p_c_given_ai_d]
        counts: (N, 4) optional diagnostic — n_(ai=C), n_coop_given_ai=C, n_(ai=D), n_coop_given_ai=D
    """
    n_agents = ds.num_agents
    # per j: n when ai=0, coop when ai=0, n when ai=1, coop when ai=1
    c = np.zeros((n_agents, 4), dtype=np.float64)
    for r in observed_rows(ds):
        j = r.j
        ai = r.a_i
        coop_j = 1.0 if r.a_j == 0 else 0.0
        if ai == 0:
            c[j, 0] += 1.0
            c[j, 1] += coop_j
        elif ai == 1:
            c[j, 2] += 1.0
            c[j, 3] += coop_j
    phi = np.zeros((n_agents, 2), dtype=np.float64)
    two_a = 2.0 * alpha
    for j in range(n_agents):
        n0, coop0, n1, coop1 = c[j]
        phi[j, 0] = (coop0 + alpha) / (n0 + two_a) if (n0 + two_a) > 0 else 0.5
        phi[j, 1] = (coop1 + alpha) / (n1 + two_a) if (n1 + two_a) > 0 else 0.5
    return phi, c
