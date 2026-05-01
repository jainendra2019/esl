from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def mean_ci95(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1.0, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(arr.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * stats.sem(arr, ddof=1))
    return {"n": float(n), "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def pointwise_mean_ci(curves: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not curves:
        return {
            "mean": np.array([], dtype=np.float64),
            "ci95_low": np.array([], dtype=np.float64),
            "ci95_high": np.array([], dtype=np.float64),
        }
    arr = np.vstack(curves).astype(np.float64)
    mean = arr.mean(axis=0)
    if arr.shape[0] == 1:
        return {"mean": mean, "ci95_low": mean.copy(), "ci95_high": mean.copy()}
    sem = stats.sem(arr, axis=0, ddof=1)
    half = stats.t.ppf(0.975, arr.shape[0] - 1) * sem
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}
