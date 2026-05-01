"""Prefix learning curve experiment."""

from __future__ import annotations

import tempfile
from pathlib import Path

from esl.config import ESLConfig
from esl.experiments.prefix_learning_curve import run_prefix_learning_curve
from esl.trainer import run_esl


def test_prefix_learning_curve_csv_and_png():
    cfg = ESLConfig(
        seed=55,
        num_rounds=12,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=2,
        log_interaction_observations=True,
        force_ordered_pair=(0, 1),
        force_agent_true_types=[0, 0, 1, 1],
    )
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "r"
        run_esl(cfg, run_dir=run_dir)
        csv_path, png_path = run_prefix_learning_curve(
            run_dir,
            n_prefix_points=4,
            em_restarts=2,
            plot=True,
        )
        assert csv_path.is_file()
        assert png_path is not None and png_path.is_file()
        text = csv_path.read_text(encoding="utf-8")
        assert "esl_mce_theta_at_round" in text
        assert "em_conditional_mce" in text
