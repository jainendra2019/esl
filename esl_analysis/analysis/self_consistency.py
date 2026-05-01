from __future__ import annotations

from typing import Any

import numpy as np

from esl_analysis.metrics.divergence import kl_divergence


def empirical_policy(actions: list[int], *, t: int, window: int = 20, action_dim: int = 2) -> np.ndarray:
    start = max(0, t - window)
    window_actions = actions[start:t]
    if not window_actions:
        return np.full(action_dim, 1.0 / action_dim, dtype=np.float64)
    counts = np.bincount(window_actions, minlength=action_dim).astype(np.float64)
    return counts / counts.sum()


def compute_kl_trajectory(rows: list[dict[str, Any]], *, window: int = 20) -> np.ndarray:
    ordered = sorted(rows, key=lambda r: int(r["round"]))
    actions = [int(r["opponent_action"]) for r in ordered]
    vals: list[float] = []
    for idx, row in enumerate(ordered):
        if idx < window:
            continue
        pred = np.array([float(row["pred_p_coop"]), float(row["pred_p_defect"])], dtype=np.float64)
        emp = empirical_policy(actions, t=idx, window=window, action_dim=2)
        vals.append(kl_divergence(pred, emp))
    return np.asarray(vals, dtype=np.float64)
