"""
Stable softmax, likelihoods, and gradients for prototype logits.

Bayesian belief updates must use unclipped softmax probabilities from likelihoods().
Apply epsilon clamping only when taking logs for the weighted log-likelihood objective / telemetry.
"""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last axis.
    logits: shape (..., A)
    """
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x - m)
    s = np.sum(ex, axis=-1, keepdims=True)
    out = ex / s
    if not (np.all(out >= -1e-15) and np.allclose(out.sum(axis=-1), 1.0)):
        raise ValueError("softmax invariant violated")
    return out


def softmax_log_likelihood(logits: np.ndarray, action: int) -> np.ndarray:
    """
    log L_k(s=action | theta_k) for each prototype k (exact log-softmax).
    logits: (K, A)
    returns: shape (K,)
    """
    log_p = stable_log_softmax(logits)
    return log_p[:, int(action)]


def softmax_log_likelihood_clamped(logits: np.ndarray, action: int, log_prob_min: float) -> np.ndarray:
    """
    §4.6: log L_k with log(max(p(a), log_prob_min)) for stable telemetry (not used in gradients).
    """
    p = stable_softmax(logits)
    pa = np.clip(p[:, int(action)], log_prob_min, 1.0)
    return np.log(pa)


def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x - m)
    log_sum = np.log(np.sum(ex, axis=-1, keepdims=True)) + m
    return x - log_sum


def likelihoods(logits: np.ndarray, action: int) -> np.ndarray:
    """L_k(s=action|theta_k) for each k; logits (K, A)."""
    return np.exp(softmax_log_likelihood(logits, action))


def grad_log_likelihood(logits: np.ndarray, action: int) -> np.ndarray:
    """
    grad_{theta_k} log L_k(s|theta_k) = e_s - softmax(theta_k).
    Returns shape (K, A).
    """
    p = stable_softmax(logits)
    g = -p
    g[:, int(action)] += 1.0
    return g


def batch_weighted_prototype_gradient(
    logits: np.ndarray,
    weights_per_k: np.ndarray,
    action: int,
) -> np.ndarray:
    """
    sum_k is outside; this returns grad for a single (i,j) term:
    sum_k w_k * (e_s - p_k) ... actually per prototype k the weight is w_k.

    logits: (K, A)
    weights_per_k: (K,)  e.g. W_ij * b_ij[k]
    returns: (K, A)  same as w_k * (e_s - p_k) elementwise per k
    """
    base = grad_log_likelihood(logits, action)
    return base * weights_per_k[:, np.newaxis]
