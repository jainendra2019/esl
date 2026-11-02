"""HMM-L: transition-aware filtered responsibilities + Online-EM-matched categorical M-step.

Emission updates consume γ_t ∝ b̄(k) φ_k(a) with b̄ = T^⊤ b (sticky Markov predict).
Does NOT call transition-free Online EM E-step overwriting the filter.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from esl.experiments.figure1_dynamic_adaptation import markov_predict
from esl.prototypes import stable_softmax
from esl.res.online_em_categorical import ONLINE_EM_INNER_ITERS, K_TYPES

EPS = 1e-12


def filtered_responsibility(
    belief: np.ndarray,
    logits: np.ndarray,
    action: int,
    *,
    p_stay: float,
    predictive_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (b̄, γ) with γ ∝ b̄(k) φ_k(a). If predictive_prior is given, reuse it (once-predict)."""
    if predictive_prior is None:
        bar = markov_predict(belief, float(p_stay))
    else:
        bar = np.asarray(predictive_prior, dtype=np.float64).reshape(-1).copy()
    phi = stable_softmax(np.asarray(logits, dtype=np.float64))
    a = int(action)
    if a < 0 or a >= phi.shape[1]:
        raise ValueError(f"action {a} out of range for A={phi.shape[1]}")
    lik = phi[:, a]
    gamma = bar * lik
    total = float(gamma.sum())
    if total <= EPS:
        gamma = np.full(bar.shape[0], 1.0 / float(bar.shape[0]), dtype=np.float64)
    else:
        gamma = gamma / total
    return bar, gamma


def m_step_from_responsibilities(
    window: Sequence[tuple[int, np.ndarray]],
    *,
    num_actions: int,
    k_types: int = K_TYPES,
    n_iters: int = ONLINE_EM_INNER_ITERS,
) -> np.ndarray:
    """Dirichlet-smoothed categorical M-step using stored (action, γ) pairs.

    Responsibilities are held fixed (transition-aware); repeating the M-step
    ``n_iters`` times matches Online EM's iteration count without a free E-step.
    """
    a = int(num_actions)
    k = int(k_types)
    if not window:
        raise ValueError("m_step_from_responsibilities requires a non-empty window")
    counts = np.zeros((k, a), dtype=np.float64)
    weights = np.zeros(k, dtype=np.float64)
    for act, gamma in window:
        g = np.asarray(gamma, dtype=np.float64).reshape(-1)
        if g.shape[0] != k:
            raise ValueError(f"gamma shape {g.shape} != ({k},)")
        counts[:, int(act)] += g
        weights += g
    out = np.zeros((k, a), dtype=np.float64)
    for _ in range(max(int(n_iters), 1)):
        p = (counts + 1.0) / (weights.reshape(-1, 1) + float(a))
        out = np.log(np.clip(p, EPS, 1.0))
    return out


def update_hmml_emissions(
    logits: np.ndarray,
    window: Sequence[tuple[int, np.ndarray]],
    *,
    num_actions: int,
    k_types: int = K_TYPES,
    n_iters: int = ONLINE_EM_INNER_ITERS,
) -> np.ndarray:
    """Update shared emission logits from a window of filtered responsibilities."""
    if not window:
        return np.asarray(logits, dtype=np.float64).copy()
    return m_step_from_responsibilities(
        window, num_actions=num_actions, k_types=k_types, n_iters=n_iters
    )


def empty_hmml_window(maxlen: int) -> deque[tuple[int, np.ndarray]]:
    return deque(maxlen=int(maxlen))
