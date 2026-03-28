"""Pairwise beliefs: initialization, Bayes update, Euclidean projection onto Δ_K^δ."""

from __future__ import annotations

import numpy as np

from esl.utils.simplex import project_to_simplex_with_floor


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


def update_belief_pair(
    prior: np.ndarray,
    likelihoods_k: np.ndarray,
    delta: float,
    eps: float,
) -> np.ndarray:
    raw = bayes_update_raw(prior, likelihoods_k, eps)
    return project_to_simplex_with_floor(raw, float(delta))
