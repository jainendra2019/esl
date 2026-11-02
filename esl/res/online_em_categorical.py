"""Categorical Online EM for A≥2 (rejects Bernoulli-only Figure-1 M-step for A>2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from esl.prototypes import stable_softmax

EPS = 1e-12
ONLINE_EM_INNER_ITERS = 3
K_TYPES = 3


def update_online_em_categorical(
    logits: np.ndarray,
    *,
    beliefs: dict[int, np.ndarray],
    em_window: Sequence[tuple[int, int]],
    num_actions: int,
    k_types: int = K_TYPES,
    n_iters: int = ONLINE_EM_INNER_ITERS,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Soft EM on a sliding window of (opponent_id, action) with categorical emissions.

    logits: (K, A) unnormalized log-emission parameters.
    Returns updated logits and per-opponent beliefs.
    """
    a = int(num_actions)
    k = int(k_types)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (k, a):
        raise ValueError(f"logits shape {logits.shape} != ({k}, {a})")
    if not em_window:
        return logits.copy(), {opp: b.copy() for opp, b in beliefs.items()}

    actions = np.array([act for _opp, act in em_window], dtype=int)
    if np.any(actions < 0) or np.any(actions >= a):
        raise ValueError("em_window actions out of range")

    out = logits.copy()
    for _ in range(int(n_iters)):
        probs = stable_softmax(out)  # (K, A)
        # responsibilities R[t, k] ∝ probs[k, action_t]
        resp = probs[:, actions].T  # (T, K)
        resp = resp / np.maximum(resp.sum(axis=1, keepdims=True), EPS)
        # M-step: Dirichlet-smoothed categorical frequencies per prototype
        weights = resp.sum(axis=0)  # (K,)
        counts = np.zeros((k, a), dtype=np.float64)
        for t, act in enumerate(actions):
            counts[:, int(act)] += resp[t]
        # prior alpha=1 ⇒ add 1 to each action bin
        p = (counts + 1.0) / (weights.reshape(-1, 1) + float(a))
        out = np.log(np.clip(p, EPS, 1.0))

    probs = stable_softmax(out)
    new_beliefs: dict[int, np.ndarray] = {}
    for opp in beliefs:
        opp_actions = [act for o, act in em_window if o == opp]
        if not opp_actions:
            new_beliefs[opp] = np.asarray(beliefs[opp], dtype=np.float64).copy()
            continue
        log_lik = np.zeros(k, dtype=np.float64)
        for action in opp_actions:
            log_lik += np.log(np.clip(probs[:, int(action)], EPS, 1.0))
        log_lik -= np.max(log_lik)
        b = np.exp(log_lik)
        new_beliefs[opp] = b / np.maximum(b.sum(), EPS)
    # Preserve any belief keys not touched
    for opp, b in beliefs.items():
        if opp not in new_beliefs:
            new_beliefs[opp] = np.asarray(b, dtype=np.float64).copy()
    return out, new_beliefs
