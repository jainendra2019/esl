"""Mean, sample std, and 95% CI for the mean (t-based); used in Milestone 4 aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def mean_std_ci95(values: Sequence[float]) -> dict[str, float]:
    """
    Return ``mean``, sample ``std`` (ddof=1), and equal-tailed **95% CI** for the mean
    using Student's t on ``n-1`` df (SciPy). For ``n < 2``, CI collapses to the point estimate.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1.0, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(arr.std(ddof=1))
    sem = float(stats.sem(arr, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sem)
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }
