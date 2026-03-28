"""Bayes update, simplex, and iterative floor projection onto Δ_K^δ (within tolerance)."""

from __future__ import annotations

import numpy as np
import pytest

from esl.beliefs import (
    bayes_update_raw,
    project_simplex_floor_iterate,
    update_belief_pair,
)


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


def test_iterative_floor_enforces_delta_within_tolerance():
    delta = 0.05
    floor_tol = 1e-8
    prior = np.array([0.97, 0.02, 0.01])
    lk = np.array([0.6, 0.3, 0.1])
    raw = bayes_update_raw(prior, lk, eps=1e-12)
    out = project_simplex_floor_iterate(raw, delta, floor_tolerance=floor_tol)
    assert np.isclose(out.sum(), 1.0, rtol=1e-9, atol=1e-9)
    assert np.all(out >= delta - floor_tol - 1e-12)
    assert not np.allclose(out, np.maximum(raw, delta) / np.maximum(raw, delta).sum())


def test_update_belief_pair_sums_to_one_and_respects_floor():
    delta = 1e-4
    floor_tol = 1e-8
    prior = np.ones(3) / 3
    lk = np.array([0.9, 0.05, 0.05])
    out = update_belief_pair(
        prior,
        lk,
        delta=delta,
        eps=1e-12,
        floor_tolerance=floor_tol,
    )
    assert np.isclose(out.sum(), 1.0, rtol=1e-9, atol=1e-9)
    assert np.all(out >= delta - floor_tol - 1e-12)


def test_floor_projection_infeasible_raises():
    with pytest.raises(ValueError, match="infeasible"):
        project_simplex_floor_iterate(np.ones(3) / 3, delta=0.5)
