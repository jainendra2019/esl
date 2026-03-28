"""Adaptive convergence stopping (stop_on_convergence + windowed norms)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from esl.config import ESLConfig
from esl.trainer import convergence_criteria_met, run_esl


def test_convergence_stops_early_when_criteria_met():
    """Frozen learning + separated θ: entropy high enough, zero norms, separation full."""
    cfg = ESLConfig(
        seed=0,
        mode="recovery",
        num_agents=4,
        num_prototypes=2,
        num_rounds=10_000,
        learning_frozen=True,
        prototype_logits_override=[[10.0, -10.0], [-10.0, 10.0]],
        stop_on_convergence=True,
        convergence_window_w=3,
        convergence_epsilon_h=2.0,
        convergence_epsilon_delta=0.5,
        convergence_epsilon_theta=0.1,
        convergence_epsilon_b=0.1,
        prototype_update_every=5,
    )
    with tempfile.TemporaryDirectory() as td:
        _, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert summary["stopped_on_convergence"] is True
    assert summary["convergence_round"] == 2
    assert summary["num_rounds_executed"] == 3
    assert summary["num_rounds"] == 10_000
    assert summary["convergence_thresholds"]["window_w"] == 3


def test_convergence_runs_full_cap_when_entropy_never_low_enough():
    cfg = ESLConfig(
        seed=0,
        mode="recovery",
        num_agents=4,
        num_prototypes=2,
        num_rounds=12,
        learning_frozen=True,
        prototype_logits_override=[[10.0, -10.0], [-10.0, 10.0]],
        stop_on_convergence=True,
        convergence_window_w=3,
        convergence_epsilon_h=0.01,
        convergence_epsilon_delta=0.5,
        convergence_epsilon_theta=0.1,
        convergence_epsilon_b=0.1,
        prototype_update_every=5,
    )
    with tempfile.TemporaryDirectory() as td:
        _, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    assert summary["stopped_on_convergence"] is False
    assert summary["convergence_round"] is None
    assert summary["num_rounds_executed"] == 12


def test_convergence_criteria_met_requires_full_window():
    rows = [
        {
            "belief_entropy_mean": 0.1,
            "prototype_update_norm": 0.0,
            "belief_change_norm": 0.0,
        }
        for _ in range(2)
    ]
    cfg = ESLConfig(
        stop_on_convergence=True,
        convergence_window_w=3,
        convergence_epsilon_h=1.0,
        convergence_epsilon_delta=0.0,
        convergence_epsilon_theta=1.0,
        convergence_epsilon_b=1.0,
        num_prototypes=2,
    )
    logits = np.array([[10.0, -10.0], [-10.0, 10.0]], dtype=np.float64)
    true_type_probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert convergence_criteria_met(rows, 1, cfg, logits, true_type_probs) is False


def test_config_rejects_convergence_epsilon_delta_out_of_range():
    with pytest.raises(ValueError, match="convergence_epsilon_delta"):
        ESLConfig(
            stop_on_convergence=True,
            convergence_epsilon_delta=1.0,
            num_prototypes=2,
        ).validate()
