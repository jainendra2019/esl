from __future__ import annotations

import numpy as np

EPS = 1e-8


def _normalize(dist: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(dist, dtype=np.float64) + eps
    return arr / arr.sum()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p_norm = _normalize(p, eps)
    q_norm = _normalize(q, eps)
    m = 0.5 * (p_norm + q_norm)
    return float(0.5 * np.sum(p_norm * np.log(p_norm / m)) + 0.5 * np.sum(q_norm * np.log(q_norm / m)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p_norm = _normalize(p, eps)
    q_norm = _normalize(q, eps)
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))
