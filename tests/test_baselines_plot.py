"""Baselines MCE bar chart smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from esl.baselines.evaluate import write_baselines_summary
from esl.baselines.plot import plot_baseline_mce_comparison
from esl.config import ESLConfig
from esl.trainer import run_esl


def test_plot_baseline_mce_comparison_writes_png():
    cfg = ESLConfig(
        seed=31,
        num_rounds=40,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=2,
        log_interaction_observations=True,
        force_agent_true_types=[0, 0, 1, 1],
    )
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "r"
        run_esl(cfg, run_dir=run_dir)
        summary = write_baselines_summary(run_dir, em_restarts=2)
        data = __import__("json").loads(summary.read_text(encoding="utf-8"))
        out = Path(td) / "mce.png"
        plot_baseline_mce_comparison(data, out)
        assert out.is_file()
        assert out.stat().st_size > 500
