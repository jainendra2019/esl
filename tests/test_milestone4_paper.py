"""Milestone 4: multi-seed aggregation, CI stats, smoke paper pack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esl.experiments.milestone4_paper import run_milestone4_paper
from esl.experiments.stats_ci import mean_std_ci95


def test_mean_std_ci95_two_samples():
    s = mean_std_ci95([0.4, 0.6])
    assert s["n"] == 2.0
    assert s["mean"] == pytest.approx(0.5)
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]


def test_mean_std_ci95_single_sample():
    s = mean_std_ci95([1.2])
    assert s["mean"] == pytest.approx(1.2)
    assert s["ci95_low"] == pytest.approx(1.2)


def test_milestone4_smoke_pack_runs(tmp_path: Path) -> None:
    mb = tmp_path / "mb"
    out = tmp_path / "m4"
    meta = run_milestone4_paper(
        out,
        smoke=True,
        seeds=(0, 1),
        manuscript_bundle=mb,
        rounds_sparse=4,
        rounds_q=4,
        rounds_init=4,
        rounds_k=4,
        p_obs_grid=(1.0, 0.5),
        q_grid=(1, 2),
        noise_grid=(0.05,),
        k_grid=(2,),
        em_restarts=2,
    )
    assert (out / "aggregate" / "sparse_pobs_agg.csv").is_file()
    assert (out / "figures" / "sparse_pobs_mce_ci.png").is_file()
    assert meta["smoke"] is True
    assert (mb / "tables" / "milestone4_sparse_pobs_agg.csv").is_file()
    assert (mb / "reports" / "milestone_4" / "MILESTONE_STATUS.md").is_file()


def test_milestone4_meta_smoke_flag(tmp_path: Path) -> None:
    out = tmp_path / "m4b"
    run_milestone4_paper(out, smoke=True, seeds=(0,), rounds_sparse=3, p_obs_grid=(1.0,))
    meta = __import__("json").loads((out / "aggregate" / "milestone4_run_meta.json").read_text())
    assert meta["smoke"] is True
    assert meta["seeds"] == [0]
