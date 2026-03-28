"""Euclidean projection onto the δ-floored simplex."""

from __future__ import annotations

import numpy as np

from esl.utils.simplex import project_to_simplex_with_floor


def test_projection_properties_user_example():
    x = np.array([0.9, 0.05, 0.05])
    delta = 0.1
    b = project_to_simplex_with_floor(x, delta)
    assert np.all(b >= delta - 1e-12)
    assert abs(float(np.sum(b)) - 1.0) < 1e-8


def test_projection_tau_zero_is_uniform_delta():
    k = 3
    delta = 1.0 / k
    x = np.array([0.2, 0.5, 0.3])
    b = project_to_simplex_with_floor(x, delta)
    np.testing.assert_allclose(b, np.full(k, delta), rtol=1e-12, atol=1e-12)


def test_interior_kd_less_than_one_never_returns_uniform_delta_sum_mismatch():
    """If K*delta < 1, the result must sum to 1, not K*delta (uniform δ would be wrong)."""
    k, delta = 5, 0.1
    assert k * delta < 1.0
    x = np.array([0.5, 0.15, 0.15, 0.1, 0.1])
    b = project_to_simplex_with_floor(x, delta)
    assert np.isclose(b.sum(), 1.0)
    assert not np.allclose(b, np.full(k, delta)), "should not short-circuit to uniform δ"


def test_random_points_obey_constraints():
    rng = np.random.default_rng(1)
    delta = 0.03
    for _ in range(50):
        x = rng.random(6) * 2.0 - 0.2
        b = project_to_simplex_with_floor(x, delta)
        assert np.all(b >= delta - 1e-10)
        assert np.isclose(b.sum(), 1.0, rtol=1e-10, atol=1e-10)
