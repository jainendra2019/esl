"""Euclidean projection onto the δ-floored simplex Δ_K^δ."""

from __future__ import annotations

import numpy as np


def project_to_simplex_with_floor(x: np.ndarray, delta: float) -> np.ndarray:
    """
    Euclidean projection onto Δ_K^δ = { b ∈ R^K : sum_k b_k = 1, b_k ≥ δ }.

    Equivalently: b = δ + y where y ≥ 0, sum y = τ = 1 - Kδ, and y minimizes ||y - (x-δ)||_2.

    Args:
        x: shape (K,)
        delta: floor mass per component; require K * delta <= 1. When K*delta < 1, τ = 1−Kδ > 0.

    Returns:
        b: shape (K,)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    k = int(x.shape[0])
    if k == 0:
        raise ValueError("empty belief vector")
    kd = float(k * delta)
    if kd > 1.0 + 1e-14:
        raise ValueError("infeasible: K * delta must be <= 1")
    tau = 1.0 - kd
    # Unique feasible point iff K*delta = 1 exactly: then b = (δ,…,δ) and sum b = Kδ = 1.
    # If K*delta < 1, τ > 0 and the sorting path applies; returning uniform δ would sum to K*δ ≠ 1.
    # Use exact float equality so we only short-circuit when kd rounds to 1.0 (e.g. k*(1.0/k)).
    if tau == 0.0:
        return np.full(k, delta, dtype=np.float64)

    z = x - delta
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, k + 1, dtype=np.float64)
    cond = u > (cssv - tau) / j
    idx = np.nonzero(cond)[0]
    if idx.size == 0:
        y = np.zeros(k, dtype=np.float64)
    else:
        rho = int(idx[-1])
        theta = float((cssv[rho] - tau) / (rho + 1))
        y = np.maximum(z - theta, 0.0)
    b = y + delta
    return b
