"""End-to-end recovery dynamics and initialization experiments."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from esl.config import ESLConfig
from esl.metrics import match_prototypes_to_types
from esl.trainer import init_prototype_logits, run_esl
from esl import games


def test_symmetric_init_prototypes_stay_close():
    cfg = ESLConfig(
        seed=10,
        symmetric_init=True,
        init_noise=0.0,
        num_rounds=80,
        num_agents=3,
        num_prototypes=2,
        observability="full",
        prototype_update_every=1,
    )
    rng = cfg.make_rng()
    theta0 = init_prototype_logits(cfg, rng)
    sep0 = float(np.linalg.norm(theta0[0] - theta0[1]))
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, _, _ = run_esl(cfg, run_dir=Path(td))
    sep1 = float(np.linalg.norm(logits[0] - logits[1]))
    assert sep1 <= sep0 + 0.5


def test_asymmetric_init_prototype_separation():
    cfg = ESLConfig(
        seed=11,
        symmetric_init=False,
        init_noise=0.2,
        num_rounds=150,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=1,
    )
    rng = cfg.make_rng()
    theta0 = init_prototype_logits(cfg, rng)
    sep0 = float(np.linalg.norm(theta0[0] - theta0[1]))
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, _, _ = run_esl(cfg, run_dir=Path(td))
    sep1 = float(np.linalg.norm(logits[0] - logits[1]))
    assert sep1 >= sep0 * 0.5


def test_adaptation_mode_runs():
    cfg = ESLConfig(
        seed=99,
        mode="adaptation",
        num_rounds=40,
        num_agents=3,
        num_prototypes=2,
        observability="full",
        prototype_update_every=1,
    )
    with tempfile.TemporaryDirectory() as td:
        _, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert summary["mode"] == "adaptation"


def test_recovery_improves_matched_cross_entropy():
    cfg = ESLConfig(
        seed=12,
        num_rounds=400,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        init_noise=0.08,
        symmetric_init=False,
        prototype_update_every=5,
    )
    true_probs = games.true_type_distributions(cfg.num_prototypes)
    rng = cfg.make_rng()
    theta_init = init_prototype_logits(cfg, rng)
    perm0, ce0 = match_prototypes_to_types(true_probs, theta_init)
    assert perm0.shape == (2,)
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, _, _ = run_esl(cfg, run_dir=Path(td))
    _, ce1 = match_prototypes_to_types(true_probs, logits)
    assert ce1 < ce0 or ce1 < 0.15
