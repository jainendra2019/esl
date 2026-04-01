"""Integration tests: variable L_t rounds and Q interaction-based prototype steps."""

from pathlib import Path
import tempfile

import numpy as np
import pytest

from esl.config import ESLConfig
from esl.trainer import run_esl


def test_prototype_update_every_interactions_syncs_legacy_field():
    cfg = ESLConfig(
        prototype_update_every=3,
        prototype_update_every_interactions=7,
        num_agents=4,
    )
    cfg.validate()
    assert cfg.prototype_update_every == 7
    assert cfg.prototype_Q() == 7


def test_q_triggers_multiple_sgd_within_one_round():
    """L_t=4 interactions, Q=2 => two prototype steps in the same env round."""
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(
            seed=1,
            mode="recovery",
            num_agents=4,
            num_prototypes=2,
            num_rounds=1,
            interaction_pairs_min=4,
            interaction_pairs_max=4,
            prototype_update_every=2,
            observability="full",
        )
        cfg.validate()
        log, _, _, _, _ = run_esl(cfg, run_dir=Path(td))
    assert len(log.prototype_update_events) == 2
    assert all(not ev["final_flush"] for ev in log.prototype_update_events)
    assert log.summary_rows[0]["round"] == 0


def test_prototype_l2_eta_changes_update_from_baseline():
    """Nonzero L2 pulls logits toward zero vs identical run with eta=0."""
    base = dict(
        seed=0,
        mode="recovery",
        num_agents=4,
        num_prototypes=2,
        num_rounds=20,
        prototype_update_every=1,
        prototype_lr_scale=2.0,
        lr_prototype_gamma_exponent=0.0,
    )
    with tempfile.TemporaryDirectory() as t0, tempfile.TemporaryDirectory() as t1:
        c0 = ESLConfig(**base, prototype_l2_eta=0.0)
        c1 = ESLConfig(**base, prototype_l2_eta=0.5)
        c0.validate()
        c1.validate()
        _, logits0, _, _, _ = run_esl(c0, run_dir=Path(t0))
        _, logits1, _, _, _ = run_esl(c1, run_dir=Path(t1))
    assert not np.allclose(logits0, logits1)


def test_degenerate_protocol_matches_single_pair_per_round_schedule():
    """Default L_min=L_max=1 with Q=5: five rounds => one prototype step at interaction 5."""
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(
            seed=42,
            mode="recovery",
            num_agents=4,
            num_prototypes=2,
            num_rounds=5,
            interaction_pairs_min=1,
            interaction_pairs_max=1,
            prototype_update_every=5,
        )
        cfg.validate()
        log, _, _, _, _ = run_esl(cfg, run_dir=Path(td))
    assert len(log.prototype_update_events) == 1
    assert log.prototype_update_events[0]["env_round_ended"] == 4
