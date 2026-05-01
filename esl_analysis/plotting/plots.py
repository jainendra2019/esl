from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {"ESL": "ESL", "Fixed-K Bayesian": "Fixed-K", "Online EM": "EM"}


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or values.shape[0] < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def plot_curves(curves: dict[str, dict[str, np.ndarray]], *, ylabel: str, title: str, path: Path, smooth: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, series in curves.items():
        mean = np.asarray(series["mean"], dtype=np.float64)
        lo = np.asarray(series["ci95_low"], dtype=np.float64)
        hi = np.asarray(series["ci95_high"], dtype=np.float64)
        if smooth:
            mean = moving_average(mean)
            lo = moving_average(lo)
            hi = moving_average(hi)
        x = np.arange(mean.shape[0])
        ax.plot(x, mean, label=METHOD_LABELS.get(method, method), linewidth=2.0)
        ax.fill_between(x, lo, hi, alpha=0.18)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
