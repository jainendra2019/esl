"""Verification tests V1–V11: algorithm and math correctness."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from esl.beliefs import bayes_update_raw, init_beliefs, update_belief_pair
from esl.config import ESLConfig
from esl.prototypes import (
    batch_weighted_prototype_gradient,
    grad_log_likelihood,
    likelihoods,
    softmax_log_likelihood,
    softmax_log_likelihood_clamped,
    stable_softmax,
)
from esl.trainer import (
    BatchRecord,
    observe_signal_update_belief,
    prototype_sgd_step_from_batch,
    run_esl,
)


@pytest.mark.verification
def test_v1_softmax_sums_stable():
    cases = [
        np.array([[0.5, 0.5]], dtype=np.float64),
        np.array([[10.0, -10.0]], dtype=np.float64),
        np.array([[-100.0, 100.0]], dtype=np.float64),
    ]
    for theta in cases:
        p = stable_softmax(theta)
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0) and np.all(p <= 1.0)
        assert abs(float(p.sum()) - 1.0) < 1e-8


@pytest.mark.verification
def test_v2_log_likelihood_finite_both_actions():
    logits = np.array([[-100.0, 100.0], [50.0, -50.0]], dtype=np.float64)
    eps = 1e-8
    for a in (0, 1):
        ll = softmax_log_likelihood_clamped(logits, action=a, log_prob_min=eps)
        assert np.all(np.isfinite(ll))
        assert np.all(ll >= np.log(eps) - 1e-10)


@pytest.mark.verification
def test_v3_gradient_matches_finite_differences_specific():
    logits = np.array([[0.7, -0.2]], dtype=np.float64)
    action = 0
    g = grad_log_likelihood(logits, action)
    eps = 1e-6
    for a in range(2):
        lp = logits.copy()
        lp[0, a] += eps
        lm = logits.copy()
        lm[0, a] -= eps
        fd = (softmax_log_likelihood(lp, action)[0] - softmax_log_likelihood(lm, action)[0]) / (2 * eps)
        assert abs(float(g[0, a]) - float(fd)) < 1e-5


@pytest.mark.verification
def test_v4_bayes_preserves_simplex():
    b = np.array([0.6, 0.4], dtype=np.float64)
    lk = np.array([0.8, 0.2], dtype=np.float64)
    post = bayes_update_raw(b, lk, eps=1e-15)
    assert np.isclose(post.sum(), 1.0)
    assert np.all(post >= 0)


@pytest.mark.verification
def test_v5_delta_simplex_projection_enforces_delta():
    from esl.utils.simplex import project_to_simplex_with_floor

    b = np.array([1e-12, 1.0 - 1e-12], dtype=np.float64)
    delta = 0.05
    out = project_to_simplex_with_floor(b, delta)
    assert np.isclose(out.sum(), 1.0, rtol=1e-9, atol=1e-9)
    assert np.all(out >= delta - 1e-12)


@pytest.mark.verification
def test_v6_no_belief_update_when_unobserved():
    cfg = ESLConfig()
    B = init_beliefs(3, 2)
    logits = np.zeros((2, 2), dtype=np.float64)
    before = B[0, 1].copy()
    rec = observe_signal_update_belief(B, logits, i=0, j=1, signal=0, w=0.0, cfg=cfg)
    assert np.allclose(B[0, 1], before)
    assert np.allclose(rec.b_ij, before)


@pytest.mark.verification
def test_v7_batch_stores_b_ij_t_before_bayes_not_b_ij_t_plus_1():
    """
    Subtle contract (PRD §4.9 / §4.11): batch weight must be b_{i→j,t} **before** the Bayes
    step from the current signal — never the posterior b_{i→j,t+1} after updating on s_{ij,t}.
    """
    cfg = ESLConfig(delta_simplex=1e-4)
    B = init_beliefs(2, 2)
    logits = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    b_t = B[0, 1].copy()
    rec = observe_signal_update_belief(B, logits, i=0, j=1, signal=0, w=1.0, cfg=cfg)
    b_t_plus_1 = B[0, 1].copy()

    assert np.allclose(rec.b_ij, b_t), "batch must store pre-update belief b_{t}"
    assert not np.allclose(b_t_plus_1, b_t), "Bayes must change belief for this fixture"
    assert not np.allclose(rec.b_ij, b_t_plus_1), "batch must NOT store posterior b_{t+1}"

    wk = rec.b_ij * 1.0
    g = batch_weighted_prototype_gradient(logits, wk, 0)
    g_expected = grad_log_likelihood(logits, 0) * wk[:, np.newaxis]
    assert np.allclose(g, g_expected)


@pytest.mark.verification
def test_v8_batch_averaging_duplicate_observations():
    cfg = ESLConfig(prototype_update_every=1, prototype_lr_scale=1.0)
    logits0 = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float64)
    b = np.array([0.5, 0.5], dtype=np.float64)
    rec = BatchRecord(i=0, j=1, signal=0, w=1.0, b_ij=b.copy())
    u1, _ = prototype_sgd_step_from_batch([rec], logits0.copy(), cfg, 0)
    u2, _ = prototype_sgd_step_from_batch([rec, rec], logits0.copy(), cfg, 0)
    np.testing.assert_allclose(u1, u2, rtol=1e-10, atol=1e-10)


@pytest.mark.verification
def test_v9_belief_likelihood_cooperate_shifts_posterior():
    cfg = ESLConfig(delta_simplex=1e-4)
    prior = np.ones(2) / 2
    logits = np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    lk = likelihoods(logits, action=0)
    post = update_belief_pair(prior, lk, cfg.delta_simplex, cfg.bayes_denominator_eps)
    assert post[0] > prior[0]
    assert post[1] < prior[1]


@pytest.mark.verification
def test_v10_single_prototype_step_increases_likelihood_of_signal():
    cfg = ESLConfig(prototype_update_every=1, prototype_lr_scale=0.5)
    logits = np.array([[-1.5, 1.5]], dtype=np.float64)
    s = 0
    p_old = float(stable_softmax(logits)[0, s])
    rec = BatchRecord(0, 1, s, 1.0, np.array([1.0]))
    logits_new, _ = prototype_sgd_step_from_batch([rec], logits.copy(), cfg, 0)
    p_new = float(stable_softmax(logits_new)[0, s])
    assert p_new > p_old


@pytest.mark.verification
def test_v11_deterministic_seed_reproducibility():
    cfg = ESLConfig(
        seed=12345,
        num_rounds=30,
        num_agents=3,
        prototype_update_every=1,
        observability="full",
    )
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _, l1, _, s1, _ = run_esl(cfg, run_dir=Path(a))
        _, l2, _, s2, _ = run_esl(cfg, run_dir=Path(b))
    np.testing.assert_allclose(l1, l2, rtol=0, atol=0)
    for key in ("final_matched_cross_entropy", "cumulative_social_payoff", "prototype_update_count"):
        np.testing.assert_allclose(float(s1[key]), float(s2[key]), rtol=1e-12, atol=1e-12)
