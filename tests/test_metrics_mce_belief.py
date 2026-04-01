"""MCE normalization, permutation invariance, belief metrics vs true type."""

import numpy as np
import pytest

from esl.metrics import (
    belief_cross_entropy_vs_type,
    belief_kl_true_vs_belief,
    match_prototypes_to_types,
    mce_value,
)


def test_mce_permutation_invariant_under_learned_row_shuffle():
    """Permuting learned prototype rows (same multiset of softmax rows) → same MCE."""
    true_probs = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    theta = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    m0 = mce_value(true_probs, theta)
    theta_swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    m1 = mce_value(true_probs, theta_swap)
    assert m0 == pytest.approx(m1, rel=0, abs=1e-9)


def test_match_total_ce_equals_k_times_mce():
    true_probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    theta = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    _, total = match_prototypes_to_types(true_probs, theta)
    m = mce_value(true_probs, theta)
    assert total == pytest.approx(2 * m)


def test_belief_ce_and_kl_vs_one_hot():
    b = np.array([0.25, 0.75], dtype=np.float64)
    ce = belief_cross_entropy_vs_type(b, true_type_j=1, k_proto=2)
    kl = belief_kl_true_vs_belief(b, true_type_j=1, k_proto=2)
    assert ce == pytest.approx(-np.log(0.75))
    # KL(e_1 || b) = log(1 / b[1]) when b[1] > 0
    assert kl == pytest.approx(-np.log(0.75))
    ce0 = belief_cross_entropy_vs_type(b, true_type_j=0, k_proto=2)
    assert ce0 == pytest.approx(-np.log(0.25))

    b_sharp = np.array([0.0, 1.0], dtype=np.float64)
    kl_sharp = belief_kl_true_vs_belief(b_sharp, true_type_j=1, k_proto=2)
    assert kl_sharp == pytest.approx(0.0, abs=1e-6)
