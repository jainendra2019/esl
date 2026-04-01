"""Smoke: NeurIPS plotting writes a PNG from minimal CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from esl.plot_neurips import plot_flagship_panels, plot_robustness_from_csv


def test_plot_flagship_panels_writes_file(tmp_path: Path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "metrics_trajectory.csv").write_text(
        "round,belief_entropy_mean,matched_cross_entropy,belief_argmax_accuracy,"
        "batch_log_likelihood,alpha_logged,prototype_step_m,prototype_update_norm,belief_change_norm\n"
        "0,0.7,1.0,0.5,0.0,1.0,0,0.0,0.1\n"
        "1,0.6,0.8,0.6,0.0,1.0,1,0.1,0.1\n",
        encoding="utf-8",
    )
    (rd / "prototype_trajectory.csv").write_text(
        "round,prototype_step_m,theta_0_0,theta_0_1,theta_1_0,theta_1_1,"
        "softmax_0_0,softmax_0_1,softmax_1_0,softmax_1_1\n"
        "0,0,0,0,0,0,0.6,0.4,0.55,0.45\n"
        "1,1,1,0,0,1,0.7,0.3,0.5,0.5\n",
        encoding="utf-8",
    )
    out = tmp_path / "fig.png"
    plot_flagship_panels(rd, out)
    assert out.is_file()
    assert out.stat().st_size > 100


def test_plot_robustness_from_csv_writes_file(tmp_path: Path):
    csv_path = tmp_path / "agg.csv"
    csv_path.write_text(
        "run_id,p_obs,final_mce,num_interaction_events_executed,prototype_lr_scale,final_prototype_gap\n"
        "a,1.0,0.1,1000,22,0.5\n"
        "b,0.5,0.3,1000,22,0.2\n",
        encoding="utf-8",
    )
    out = tmp_path / "rob.png"
    plot_robustness_from_csv(csv_path, out)
    assert out.is_file()
    assert out.stat().st_size > 100
