"""Contracts aligned with ALGORITHM.md (recovery vs adaptation, batch w=0, K cycling)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import esl.trainer as trainer_mod
from esl.config import ESLConfig
from esl.games import true_type_distributions
from esl.prototypes import batch_weighted_prototype_gradient, grad_log_likelihood, stable_softmax
from esl.trainer import BatchRecord, prototype_sgd_step_from_batch, run_esl


@pytest.mark.verification
def test_v12_batch_appends_w_zero_and_dilutes_mean_gradient():
    """
    Records with w=0 are appended; they add no gradient mass but increase |batch|
    in the denominator of the mean (ALGORITHM.md §5).
    """
    cfg = ESLConfig(prototype_update_every=1, prototype_lr_scale=1.0, lr_prototype_gamma_exponent=0.0)
    logits = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float64)
    b = np.array([0.6, 0.4], dtype=np.float64)
    rec_on = BatchRecord(0, 1, 0, 1.0, b.copy())
    rec_off = BatchRecord(0, 2, 1, 0.0, b.copy())

    u_single, _ = prototype_sgd_step_from_batch([rec_on], logits.copy(), cfg, 0)
    u_mixed, _ = prototype_sgd_step_from_batch([rec_off, rec_on], logits.copy(), cfg, 0)

    g_one = batch_weighted_prototype_gradient(logits, b * 1.0, 0)
    g_mean_single = g_one / 1.0
    g_mean_mixed = g_one / 2.0
    np.testing.assert_allclose(u_single, logits + g_mean_single, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(u_mixed, logits + g_mean_mixed, rtol=1e-10, atol=1e-10)
    assert not np.allclose(u_single, u_mixed)


@pytest.mark.verification
def test_v13_weighted_gradient_equals_w_times_b_times_es_minus_p():
    """Per-record gradient: (w * b_snap[k]) * (e_s - p_k) per row k."""
    logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    s = 0
    w = 0.7
    b_snap = np.array([0.25, 0.75], dtype=np.float64)
    wk = b_snap * w
    g = batch_weighted_prototype_gradient(logits, wk, s)
    p = stable_softmax(logits)
    for k in range(2):
        ek = np.zeros(2, dtype=np.float64)
        ek[s] = 1.0
        expected_row = wk[k] * (ek - p[k])
        np.testing.assert_allclose(g[k], expected_row, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g, grad_log_likelihood(logits, s) * wk[:, np.newaxis], rtol=1e-12, atol=1e-12)


@pytest.mark.verification
def test_v14_true_type_distributions_cycles_behaviors_when_k_gt_2():
    d4 = true_type_distributions(4)
    assert d4.shape == (4, 2)
    # Row 0 AC, 1 AD, 2 AC, 3 AD
    np.testing.assert_allclose(d4[0], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d4[1], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(d4[2], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d4[3], [0.0, 1.0], atol=1e-12)


@pytest.mark.verification
def test_v15_recovery_mode_never_calls_act_agent(monkeypatch):
    def _forbidden(*_a, **_k):
        raise AssertionError("act_agent must not be used in recovery mode")

    monkeypatch.setattr(trainer_mod, "act_agent", _forbidden)
    cfg = ESLConfig(
        mode="recovery",
        num_rounds=8,
        num_agents=3,
        num_prototypes=2,
        observability="full",
        prototype_update_every=5,
        seed=0,
    )
    with tempfile.TemporaryDirectory() as td:
        run_esl(cfg, run_dir=Path(td))


@pytest.mark.verification
def test_v16_adaptation_mode_calls_act_agent(monkeypatch):
    calls = {"n": 0}
    real = trainer_mod.act_agent

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(trainer_mod, "act_agent", _count)
    cfg = ESLConfig(
        mode="adaptation",
        num_rounds=4,
        num_agents=3,
        num_prototypes=2,
        observability="full",
        prototype_update_every=99,
        seed=1,
    )
    with tempfile.TemporaryDirectory() as td:
        run_esl(cfg, run_dir=Path(td))
    assert calls["n"] >= 8
