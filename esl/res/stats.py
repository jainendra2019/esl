"""Paired CIs and effect sizes for RES robustness reporting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjustment; returns values in the original order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: float(pvals[i]))
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (n - rank) * float(pvals[i]))
        running = max(running, val)
        adj[i] = running
    return adj


def filter_seeds(rows: Sequence[dict], seeds: Sequence[int]) -> list[dict]:
    want = {int(s) for s in seeds}
    return [r for r in rows if int(r["seed"]) in want]


def paired_psr_diffs(
    left_rows: Sequence[dict],
    right_by_key: dict[tuple[str, int], float],
) -> list[float]:
    """OMD-style pairing: walk left rows in given order; skip unmatched keys."""
    diffs: list[float] = []
    for r in left_rows:
        key = (str(r["instance_id"]), int(r["seed"]))
        if key not in right_by_key:
            continue
        diffs.append(float(r["psr"]) - float(right_by_key[key]))
    return diffs


def bootstrap_mean_ci95(
    values: Sequence[float],
    *,
    seed: int = 42,
    n_boot: int = 10_000,
) -> dict[str, float]:
    """Percentile bootstrap 95% CI for the mean (matches the held-out OMD pairing)."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        mean = float(arr.mean()) if n else float("nan")
        return {
            "n": float(n),
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    mean = float(arr.mean())
    rng = np.random.default_rng(int(seed))
    boot = sorted(float(np.mean(arr[rng.integers(0, n, size=n)])) for _ in range(int(n_boot)))
    return {
        "n": float(n),
        "mean": mean,
        "ci95_low": boot[int(0.025 * len(boot))],
        "ci95_high": boot[int(0.975 * len(boot))],
    }


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
