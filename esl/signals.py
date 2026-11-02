"""Observation channel: identity or categorical confusion noise."""

from __future__ import annotations

import numpy as np


def action_to_signal(action: int) -> int:
    """MVP / paper default: signal equals observed opponent action."""
    return int(action)


def confusion_channel(num_actions: int, eps: float) -> np.ndarray:
    """
    Symmetric categorical channel P(ã|a):

        P(ã=a) = 1-ε
        P(ã≠a) = ε/(A-1)
    """
    a = int(num_actions)
    e = float(eps)
    if a < 2:
        raise ValueError("num_actions must be >= 2")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must be in [0, 1)")
    if e == 0.0:
        return np.eye(a, dtype=np.float64)
    off = e / float(a - 1)
    p = np.full((a, a), off, dtype=np.float64)
    np.fill_diagonal(p, 1.0 - e)
    return p


def sample_noisy_action(
    action: int,
    *,
    num_actions: int,
    eps: float,
    rng: np.random.Generator,
) -> int:
    """Draw ã ~ P(·|a) from the symmetric confusion channel."""
    if float(eps) <= 0.0:
        return int(action)
    channel = confusion_channel(num_actions, eps)
    probs = channel[int(action)]
    return int(rng.choice(num_actions, p=probs))
