"""Bayes update and Euclidean projection onto Δ_K^δ."""

from __future__ import annotations

import numpy as np
import pytest

from esl.beliefs import bayes_update_raw, update_belief_pair
from esl.utils.simplex import project_to_simplex_with_floor


def test_bayes_preserves_simplex():
    rng = np.random.default_rng(2)
    for _ in range(30):
        prior = rng.random(4)
        prior /= prior.sum()
        lk = rng.random(4)
        lk /= lk.max()
        post = bayes_update_raw(prior, lk, eps=1e-15)
        assert np.isclose(post.sum(), 1.0)
        assert np.all(post >= 0)


def test_euclidean_projection_after_bayes_enforces_delta():
    delta = 0.05
    prior = np.array([0.97, 0.02, 0.01])
    lk = np.array([0.6, 0.3, 0.1])
    raw = bayes_update_raw(prior, lk, eps=1e-12)
    out = project_to_simplex_with_floor(raw, delta)
    assert np.isclose(out.sum(), 1.0, rtol=1e-9, atol=1e-9)
    assert np.all(out >= delta - 1e-10)
    naive = np.maximum(raw, delta)
    naive /= naive.sum()
    assert not np.allclose(out, naive, rtol=1e-3, atol=1e-3)


def test_update_belief_pair_sums_to_one_and_respects_floor():
    delta = 1e-4
    prior = np.ones(3) / 3
    lk = np.array([0.9, 0.05, 0.05])
    out = update_belief_pair(prior, lk, delta=delta, eps=1e-12)
    assert np.isclose(out.sum(), 1.0, rtol=1e-9, atol=1e-9)
    assert np.all(out >= delta - 1e-10)


def test_simplex_projection_infeasible_raises():
    with pytest.raises(ValueError, match="infeasible"):
        project_to_simplex_with_floor(np.ones(3) / 3, delta=0.5)
