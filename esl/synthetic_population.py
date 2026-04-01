"""Synthetic population for simulation and evaluation only.

Ground truth Θ★, latent types z, and idiosyncratic shocks φ must **not** feed the online
learner in ``run_esl``. This module is for experiment drivers and metric sanity checks.

**Layering:** ``esl.trainer`` must not import this package (enforced by a unit test).
"""

from __future__ import annotations

import numpy as np

from esl.games import HiddenPolicy
from esl.prototypes import stable_softmax


def sample_latent_types(
    rng: np.random.Generator,
    rho: np.ndarray,
    n_agents: int,
) -> np.ndarray:
    """
    z_i ~ Categorical(rho), i = 0..n_agents-1.

    rho: shape (K_types,) nonnegative, will be normalized.
    Returns: integer array shape (n_agents,) with values in 0..len(rho)-1.
    """
    rho = np.asarray(rho, dtype=np.float64).ravel()
    if rho.size < 1 or np.any(rho < 0):
        raise ValueError("rho must be nonempty and nonnegative")
    s = rho.sum()
    if s <= 0:
        raise ValueError("rho must sum to a positive value")
    rho = rho / s
    return rng.choice(len(rho), size=int(n_agents), p=rho).astype(np.int64)


def sample_gaussian_parameter_noise(
    rng: np.random.Generator,
    n_agents: int,
    num_actions: int,
    sigma: float,
) -> np.ndarray:
    """phi_i ~ N(0, sigma^2 I) in R^{|A|}, shape (n_agents, num_actions)."""
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return float(sigma) * rng.standard_normal(size=(int(n_agents), int(num_actions)))


def agent_logits_from_star(
    theta_star: np.ndarray,
    z: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    tilde_theta[i] = theta_star[z[i]] + phi[i].

    theta_star: (K, A); z: (N,); phi: (N, A).
    """
    theta_star = np.asarray(theta_star, dtype=np.float64)
    z = np.asarray(z, dtype=np.int64)
    phi = np.asarray(phi, dtype=np.float64)
    if theta_star.ndim != 2:
        raise ValueError("theta_star must be (K, A)")
    k, a = theta_star.shape
    if z.shape != (phi.shape[0],) or phi.shape[1] != a:
        raise ValueError("shape mismatch between z, phi, and theta_star")
    if np.any(z < 0) or np.any(z >= k):
        raise ValueError("z entries must be in [0, K)")
    return theta_star[z] + phi


class SoftmaxLogitsPolicy(HiddenPolicy):
    """Independent softmax policy from a fixed 2-action logit vector (recovery / sim)."""

    def __init__(self, logits: np.ndarray) -> None:
        self._logits = np.asarray(logits, dtype=np.float64).ravel()
        if self._logits.shape[0] != 2:
            raise ValueError("v1 supports exactly 2 actions")

    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        p = stable_softmax(self._logits.reshape(1, -1))[0]
        return int(rng.choice(len(p), p=p))

    def action_probs(self) -> np.ndarray:
        return stable_softmax(self._logits.reshape(1, -1))[0]


def build_softmax_policies(agent_logits: np.ndarray) -> list[SoftmaxLogitsPolicy]:
    """One SoftmaxLogitsPolicy per row of agent_logits (N, 2)."""
    agent_logits = np.asarray(agent_logits, dtype=np.float64)
    if agent_logits.ndim != 2 or agent_logits.shape[1] != 2:
        raise ValueError("agent_logits must be (N, 2)")
    return [SoftmaxLogitsPolicy(agent_logits[i]) for i in range(agent_logits.shape[0])]
