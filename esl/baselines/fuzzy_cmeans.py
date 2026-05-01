"""Fuzzy C-means (Euclidean), hard labels for MCE via argmax membership."""

from __future__ import annotations

import time

import numpy as np


def fuzzy_cmeans(
    x: np.ndarray,
    k: int,
    *,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Standard FCM. x: (n, d). Returns (hard_labels (n,), centers (k, d), objective).
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    # init centers from random data points
    idx = rng.choice(n, size=k, replace=False)
    centers = x[idx].astype(np.float64).copy()
    u = np.ones((n, k), dtype=np.float64) / k
    prev_obj = np.inf
    for _ in range(max_iter):
        dist = np.stack(
            [
                np.maximum(np.sum((x - centers[c]) ** 2, axis=1), 1e-12)
                for c in range(k)
            ],
            axis=1,
        )
        exp = 2.0 / (m - 1.0)
        d_pow = dist ** (-exp / 2.0)
        u = d_pow / np.sum(d_pow, axis=1, keepdims=True)
        u_m = u**m
        for c in range(k):
            wsum = u_m[:, c].sum()
            if wsum > 1e-12:
                centers[c] = (u_m[:, c : c + 1] * x).sum(axis=0) / wsum
        obj = float(np.sum(u_m * dist))
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
    labels = np.argmax(u, axis=1)
    return labels, centers, obj


def fuzzy_cmeans_with_timing(
    x: np.ndarray, k: int, **kwargs: object
) -> tuple[np.ndarray, np.ndarray, float, float]:
    t0 = time.perf_counter()
    labels, centers, obj = fuzzy_cmeans(x, k, **kwargs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    return labels, centers, obj, elapsed
