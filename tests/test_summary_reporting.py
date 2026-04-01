"""NeurIPS-style reporting: summary keys, MCE, freeze baseline, Q schedule, belief perm, no leakage."""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from esl.beliefs import init_beliefs
from esl.config import ESLConfig
from esl.games import true_type_distributions
from esl.metrics import belief_argmax_accuracy, mce_value
from esl.trainer import observe_signal_update_belief, run_esl


NEURIPS_SUMMARY_KEYS = (
    "num_interaction_events_executed",
    "prototype_update_every_q",
    "p_obs",
    "prototype_lr_scale",
    "init_noise",
    "final_prototype_gap",
    "final_prototype_softmax",
    "final_mce",
)


def test_summary_includes_neurips_reporting_keys():
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(
            seed=0,
            mode="recovery",
            num_rounds=3,
            num_agents=4,
            num_prototypes=2,
            prototype_update_every=2,
            observability="sparse",
            p_obs=0.7,
            prototype_lr_scale=3.5,
            init_noise=0.03,
        )
        cfg.validate()
        _, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    for k in NEURIPS_SUMMARY_KEYS:
        assert k in summary, f"missing {k}"
    assert summary["p_obs"] == pytest.approx(0.7)
    assert summary["prototype_lr_scale"] == pytest.approx(3.5)
    assert summary["init_noise"] == pytest.approx(0.03)
    assert summary["prototype_update_every_q"] == cfg.prototype_Q()
    assert isinstance(summary["final_prototype_softmax"], list)
    assert len(summary["final_prototype_softmax"]) == 2


def test_num_interaction_events_equals_reward_rows_when_learning_enabled():
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(
            seed=1,
            mode="recovery",
            num_rounds=4,
            num_agents=3,
            num_prototypes=2,
            prototype_update_every=5,
        )
        cfg.validate()
        log, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert summary["num_interaction_events_executed"] == len(log.reward_rows)


def test_final_mce_matches_mce_value():
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(seed=2, mode="recovery", num_rounds=2, num_agents=4, num_prototypes=2)
        cfg.validate()
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    true_p = true_type_distributions(cfg.num_prototypes)
    expected = mce_value(true_p, logits)
    assert summary["final_mce"] == pytest.approx(expected, rel=0, abs=1e-9)


def test_final_prototype_gap_matches_separation():
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(seed=3, mode="recovery", num_rounds=2, num_agents=4, num_prototypes=2)
        cfg.validate()
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    from esl.trainer import matched_true_type_separation_p_coop

    true_p = true_type_distributions(cfg.num_prototypes)
    g = matched_true_type_separation_p_coop(true_p, logits)
    assert summary["final_prototype_gap"] == pytest.approx(g, rel=0, abs=1e-9)


def test_freeze_prototype_theta_unchanged_and_no_updates():
    theta = [[0.4, -0.4], [-0.2, 0.2]]
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(
            seed=99,
            mode="recovery",
            num_rounds=8,
            num_agents=4,
            num_prototypes=2,
            prototype_update_every=1,
            freeze_prototype_parameters=True,
            prototype_logits_override=[list(r) for r in theta],
        )
        cfg.validate()
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert np.allclose(logits, np.array(theta, dtype=np.float64))
    assert summary["prototype_update_count"] == 0


def test_prototype_scheduled_events_occur_at_multiples_of_q():
    """Non-final-flush updates happen when interaction_n is a multiple of Q."""
    with tempfile.TemporaryDirectory() as td:
        cfg = ESLConfig(
            seed=0,
            mode="recovery",
            num_rounds=2,
            num_agents=4,
            num_prototypes=2,
            interaction_pairs_min=3,
            interaction_pairs_max=3,
            prototype_update_every=4,
        )
        cfg.validate()
        Q = cfg.prototype_Q()
        log, _, _, _, _ = run_esl(cfg, run_dir=Path(td))
    for ev in log.prototype_update_events:
        if ev.get("final_flush"):
            continue
        n_at = ev.get("interaction_n_at_update")
        assert n_at is not None
        assert n_at % Q == 0, (n_at, Q)


def test_belief_argmax_accuracy_respects_hungarian_perm():
    """Argmax must match perm[true_type[j]], not raw prototype index == true type."""
    n, k = 2, 2
    B = init_beliefs(n, k)
    # perm = [1,0]: true type 0 -> learned row 1; true type 1 -> learned row 0.
    B[0, 1, :] = np.array([0.9, 0.1], dtype=np.float64)  # j=1 type 1 -> row 0
    B[1, 0, :] = np.array([0.1, 0.9], dtype=np.float64)  # j=0 type 0 -> row 1
    true_types = np.array([0, 1], dtype=int)
    perm = np.array([1, 0], dtype=int)
    acc = belief_argmax_accuracy(B, true_types, perm, n)
    assert acc == 1.0


def test_observe_signal_update_belief_has_no_true_types_param():
    sig = inspect.signature(observe_signal_update_belief)
    assert "true_types" not in sig.parameters


def test_summary_metrics_json_roundtrip_includes_new_keys():
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        cfg = ESLConfig(seed=0, mode="recovery", num_rounds=1, num_agents=3, num_prototypes=2)
        cfg.validate()
        run_esl(cfg, run_dir=rd)
        data = json.loads((rd / "summary_metrics.json").read_text(encoding="utf-8"))
    assert "final_mce" in data
    assert "num_interaction_events_executed" in data
