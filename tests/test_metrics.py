"""Permutation matching agrees with brute force for small K."""

from __future__ import annotations

import numpy as np

from esl.metrics import brute_force_min_permutation, hungarian_min_cost_permutation


def test_hungarian_matches_bruteforce():
    rng = np.random.default_rng(3)
    for k in range(2, 5):
        cost = rng.random((k, k))
        p1, c1 = brute_force_min_permutation(cost)
        p2, c2 = hungarian_min_cost_permutation(cost)
        assert np.isclose(c1, c2, rtol=1e-6)
