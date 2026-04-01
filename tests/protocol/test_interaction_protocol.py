"""Tests for variable-L_t interaction sampling (test-first protocol layer)."""

import numpy as np
import pytest

from esl.interaction_protocol import (
    all_ordered_pairs,
    num_ordered_pairs,
    sample_L_t,
    sample_ordered_pairs_without_replacement,
)


def test_num_ordered_pairs():
    assert num_ordered_pairs(4) == 12
    assert num_ordered_pairs(2) == 2
    assert num_ordered_pairs(1) == 0


def test_all_ordered_pairs_count_and_uniqueness():
    n = 5
    pairs = all_ordered_pairs(n)
    assert len(pairs) == num_ordered_pairs(n)
    assert len(set(pairs)) == len(pairs)
    for i, j in pairs:
        assert i != j and 0 <= i < n and 0 <= j < n


def test_sample_L_t_range_uniform():
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = sample_L_t(rng, 2, 7, law="uniform")
        assert 2 <= L <= 7


def test_sample_L_t_degenerate_no_rng_drift():
    """L_min == L_max must not consume RNG (locked L8 / golden parity)."""
    ra = np.random.default_rng(12345)
    rb = np.random.default_rng(12345)
    L = sample_L_t(ra, 3, 3, law="uniform")
    assert L == 3
    assert ra.random() == rb.random()


def test_sample_ordered_pairs_L1_single_integers_draw_parity():
    """L_t==1 path: same as one integers draw over flattened pair index."""
    pool = all_ordered_pairs(4)
    rng = np.random.default_rng(7)
    idx = int(rng.integers(0, len(pool)))
    rng2 = np.random.default_rng(7)
    out = sample_ordered_pairs_without_replacement(rng2, 4, 1)
    assert out == [pool[idx]]


def test_sample_ordered_pairs_no_duplicates_within_round():
    rng = np.random.default_rng(1)
    for n in (4, 6):
        max_p = num_ordered_pairs(n)
        for L in (1, 3, max_p):
            pairs = sample_ordered_pairs_without_replacement(rng, n, L)
            assert len(pairs) == L
            assert len(set(pairs)) == L


def test_sample_ordered_pairs_L_t_exceeds_pairs_errors():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceeds"):
        sample_ordered_pairs_without_replacement(rng, 3, 10)


def test_sequential_belief_updates_visible_to_next_pair():
    """Later slots read tensor state after earlier in-round writes (contract for inner loop)."""
    from esl import beliefs as belief_ops

    K = 2
    n_agents = 3
    B = belief_ops.init_beliefs(n_agents, K)
    uniform_01 = B[0, 1].copy()
    B[0, 1] = np.array([0.7, 0.3], dtype=np.float64)
    assert not np.allclose(uniform_01, B[0, 1])
    # Unrelated cell unchanged — next interaction (different i,j) still sees its own row
    B2 = belief_ops.init_beliefs(n_agents, K)
    assert np.allclose(B[1, 0], B2[1, 0])
