"""Edge-case robustness tests E1–E4."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from esl.beliefs import update_belief_pair
from esl.config import ESLConfig
from esl.prototypes import likelihoods, stable_softmax
from esl.trainer import run_esl


@pytest.mark.edge
def test_e1_all_agents_same_hidden_type():
    cfg = ESLConfig(
        seed=20,
        num_prototypes=1,
        num_agents=4,
        num_rounds=80,
        observability="full",
        prototype_update_every=1,
        init_noise=0.02,
    )
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert np.all(np.isfinite(logits))
    assert summary["final_matched_cross_entropy"] < 1.0
    assert not np.any(np.isnan(logits))


@pytest.mark.edge
def test_e2_overparameterized_k_four():
    cfg = ESLConfig(
        seed=21,
        num_prototypes=4,
        num_agents=5,
        num_rounds=60,
        observability="full",
        prototype_update_every=2,
        init_noise=0.05,
    )
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert logits.shape == (4, 2)
    assert np.all(np.isfinite(logits))
    assert "permutation_true_to_learned" in summary
    assert len(summary["permutation_true_to_learned"]) == 4


@pytest.mark.edge
def test_e3_extreme_sparsity_runs():
    cfg = ESLConfig(
        seed=22,
        num_rounds=200,
        observability="sparse",
        p_obs=0.01,
        prototype_update_every=5,
    )
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert np.all(np.isfinite(logits))
    assert not np.any(np.isnan(logits))


@pytest.mark.edge
def test_e4_degenerate_extreme_logits():
    huge = np.array([[500.0, -500.0], [-500.0, 500.0]], dtype=np.float64)
    p = stable_softmax(huge)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=-1), 1.0)
    lk0 = likelihoods(huge, 0)
    assert np.all(np.isfinite(lk0))
    assert float(lk0.sum()) > 0
    prior = np.ones(2) / 2
    post = update_belief_pair(prior, lk0, delta=1e-4, eps=1e-12, floor_tolerance=1e-8)
    assert np.isfinite(post).all()
