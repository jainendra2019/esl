"""Paired CIs and effect sizes for RES robustness reporting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def paired_mean_ci95(diffs: Sequence[float]) -> dict[str, float]:
    """Paired mean difference with Student-t 95% CI."""
    arr = np.asarray(list(diffs), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "cohen_dz": float("nan"),
        }
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1.0, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "cohen_dz": float("nan")}
    std = float(arr.std(ddof=1))
    sem = float(stats.sem(arr, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sem)
    dz = float(mean / std) if std > 0.0 else float("nan")
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "cohen_dz": dz,
    }


def method_mean_order(method_to_mean: dict[str, float], *, lower_is_better: bool = True) -> list[str]:
    """Return methods sorted by mean metric (default: lower better, e.g. PSR)."""
    return sorted(method_to_mean.keys(), key=lambda m: (method_to_mean[m], m), reverse=not lower_is_better)
