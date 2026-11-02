"""Entropy-regularized OMD via multiplicative-weights (MWU) update.

Primary experimental variant: per-opponent log-weights z_j (interaction-specific).
Secondary ablation: global z shared across opponents.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

EPS = 1e-12
# Candidate step sizes for held-out selection; report η is frozen after the grid.
OMD_ETA_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
OMD_PRIMARY_C = 0.1  # grid member / legacy default; not the frozen report η until selected
OMD_SENSITIVITY_C: tuple[float, ...] = (0.03, 0.1, 0.3)
OmdScope = Literal["per_opponent", "global"]
OMD_DEFAULT_SCOPE: OmdScope = "per_opponent"


def omd_init_log_weights(num_actions: int) -> np.ndarray:
    return np.zeros(int(num_actions), dtype=np.float64)


def omd_init_per_opponent(
    num_actions: int,
    opponent_ids: tuple[int, ...] | list[int],
) -> dict[int, np.ndarray]:
    """Identical zero initialization for each opponent's log-weight vector."""
    return {int(j): omd_init_log_weights(num_actions) for j in opponent_ids}


def omd_policy(log_weights: np.ndarray) -> np.ndarray:
    z = np.asarray(log_weights, dtype=np.float64)
    z = z - np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def omd_step(log_weights: np.ndarray, rewards: np.ndarray, *, eta: float) -> np.ndarray:
    z = np.asarray(log_weights, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64)
    if z.shape != r.shape:
        raise ValueError(f"shape mismatch z={z.shape} r={r.shape}")
    return z + float(eta) * r


def normalize_matrix_once(u: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Affine map of a focal payoff matrix to [-1, 1]; computed once per matrix."""
    m = np.asarray(u, dtype=np.float64)
    lo = float(m.min())
    hi = float(m.max())
    if hi - lo < 1e-12:
        return np.zeros_like(m), lo, hi
    scaled = 2.0 * (m - lo) / (hi - lo) - 1.0
    return scaled, lo, hi


def focal_reward_vector(
    u_focal: np.ndarray,
    *,
    opponent_action: int,
    u_min: float,
    u_max: float,
) -> np.ndarray:
    """
    Full-information scaled rewards for every focal action against observed opponent action:

        r̃(a) = 2 (U(a, a_{-i}) - min U) / (max U - min U) - 1

    Favorable feedback: after observing a_opp, the learner receives the payoffs that
    every focal action would have obtained against that action (not only the realized scalar).

    If max U = min U, return the zero vector.
    """
    col = np.asarray(u_focal, dtype=np.float64)[:, int(opponent_action)]
    if abs(float(u_max) - float(u_min)) < 1e-12:
        return np.zeros_like(col)
    return 2.0 * (col - float(u_min)) / (float(u_max) - float(u_min)) - 1.0
