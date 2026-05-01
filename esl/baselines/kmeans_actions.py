"""K-means on per-agent feature vectors."""

from __future__ import annotations

import time

import numpy as np


def _init_centers_kmeans_pp(
    x: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """K-means++ initialization (Euclidean). x: (n, d)."""
    n, d = x.shape
    idx0 = int(rng.integers(0, n))
    centers = np.zeros((k, d), dtype=np.float64)
    centers[0] = x[idx0]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(np.sum(closest))
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            probs = (closest.astype(np.float64) / total).clip(0.0, 1.0)
            s = float(np.sum(probs))
            if s <= 0.0 or not np.isfinite(s):
                probs = np.full(n, 1.0 / n, dtype=np.float64)
            else:
                probs /= s
        j = int(rng.choice(n, p=probs))
        centers[c] = x[j]
        dist = np.sum((x - centers[c]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    return centers


def kmeans_lloyd(
    x: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    n_init: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (labels (n,), centers (k, d), inertia) for best of n_init runs.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    best_inertia = np.inf
    best_labels = np.zeros(n, dtype=int)
    best_centers = np.zeros((k, d), dtype=np.float64)
    for _ in range(n_init):
        centers = _init_centers_kmeans_pp(x, k, rng)
        for _it in range(max_iter):
            dist = np.stack(
                [np.sum((x - centers[c]) ** 2, axis=1) for c in range(k)], axis=1
            )
            labels = np.argmin(dist, axis=1)
            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centers[c] = x[mask].mean(axis=0)
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift < tol:
                break
        inertia = float(
            np.sum((x - centers[labels]) ** 2)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def kmeans_with_timing(
    x: np.ndarray, k: int, **kwargs: object
) -> tuple[np.ndarray, np.ndarray, float, float]:
    t0 = time.perf_counter()
    labels, centers, inertia = kmeans_lloyd(x, k, **kwargs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    return labels, centers, inertia, elapsed
