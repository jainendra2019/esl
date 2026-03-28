"""Softmax invariants and gradient vs finite differences."""

from __future__ import annotations

import numpy as np

from esl.prototypes import (
    grad_log_likelihood,
    softmax_log_likelihood,
    softmax_log_likelihood_clamped,
    stable_softmax,
)


def test_softmax_sums_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal((5, 2))
        p = stable_softmax(x)
        assert np.allclose(p.sum(axis=-1), 1.0)
        assert np.all(p >= 0)


def test_clamped_log_likelihood_finite_for_extreme_logits():
    logits = np.array([[-100.0, 100.0], [50.0, -50.0]], dtype=np.float64)
    ll = softmax_log_likelihood_clamped(logits, action=0, log_prob_min=1e-8)
    assert np.all(np.isfinite(ll))
    assert np.all(ll >= np.log(1e-8) - 1e-6)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((3, 2))
    action = 0
    eps = 1e-5
    g = grad_log_likelihood(logits, action)
    for k in range(3):
        for a in range(2):
            lp = logits.copy()
            lp[k, a] += eps
            lm = logits.copy()
            lm[k, a] -= eps
            fd = (
                softmax_log_likelihood(lp, action)[k] - softmax_log_likelihood(lm, action)[k]
            ) / (2 * eps)
            assert np.isclose(g[k, a], fd, rtol=1e-4, atol=1e-4)
