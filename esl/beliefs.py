"""Pairwise beliefs: initialization, Bayes update, iterative simplex floor (Δ_K^δ up to tolerance)."""

from __future__ import annotations

import numpy as np


def init_beliefs(num_agents: int, k: int, dtype: type = np.float64) -> np.ndarray:
    """Shape (N, N, K) with uniform beliefs on ordered pairs; diagonal entries unused (zeros)."""
    b = np.full((num_agents, num_agents, k), 1.0 / k, dtype=dtype)
    for i in range(num_agents):
        b[i, i, :] = 0.0
    return b


def bayes_update_raw(prior: np.ndarray, likelihoods_k: np.ndarray, eps: float) -> np.ndarray:
    """
    prior, likelihoods_k: shape (K,)
    tilde_b[k] propto prior[k] * L_k
    """
    num = prior * likelihoods_k
    den = num.sum() + eps
    return num / den


def project_simplex_floor_iterate(
    b: np.ndarray,
    delta: float,
    *,
    floor_tolerance: float = 1e-8,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Practical surrogate for Δ_K^δ: repeat clip to δ and renormalize until
    min_k b[k] ≥ δ - floor_tolerance and sum b = 1 (within float tolerance).

    A single clip+renorm need not satisfy the floor after normalization; iterating fixes that.
    """
    k = len(b)
    if delta * k > 1.0 + 1e-12:
        raise ValueError("infeasible: K*delta must be <= 1")
    x = np.asarray(b, dtype=np.float64)
    for _ in range(max_iter):
        x = np.maximum(x, delta)
        s = x.sum()
        if s <= 0:
            raise ValueError("belief projection: zero sum after clip")
        x = x / s
        if float(np.min(x)) >= delta - floor_tolerance and np.isclose(x.sum(), 1.0, rtol=1e-10, atol=1e-10):
            break
    else:
        raise ValueError(f"simplex floor projection did not converge in {max_iter} iterations")
    if float(np.min(x)) < delta - floor_tolerance - 1e-12:
        raise AssertionError("floor constraint violated after projection")
    if not np.isclose(x.sum(), 1.0, rtol=1e-9, atol=1e-9):
        raise AssertionError("belief posterior must lie on simplex")
    return x


def update_belief_pair(
    prior: np.ndarray,
    likelihoods_k: np.ndarray,
    delta: float,
    eps: float,
    *,
    floor_tolerance: float = 1e-8,
    max_floor_iter: int = 10_000,
) -> np.ndarray:
    raw = bayes_update_raw(prior, likelihoods_k, eps)
    return project_simplex_floor_iterate(
        raw,
        delta,
        floor_tolerance=floor_tolerance,
        max_iter=max_floor_iter,
    )
