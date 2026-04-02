"""Sparse p_obs mini-sweep (tiny rounds)."""

from __future__ import annotations

from pathlib import Path

import pytest

from esl.experiments.sparse_pobs_sweep import (
    POBS_SWEEP_VALUES,
    plot_final_ce_vs_pobs,
    run_sparse_pobs_sweep,
)


def test_pobs_sweep_values():
    assert POBS_SWEEP_VALUES == (1.0, 0.5, 0.3, 0.2)


@pytest.mark.parametrize("p", [1.0, 0.5, 0.3, 0.2])
def test_sparse_obs_accepts_all_sweep_pobs(p: float):
    from esl.experiments.presets import recovery_sparse_obs_cfg

    c = recovery_sparse_obs_cfg(p_obs=p, seed=0)
    c.validate()
    assert c.p_obs == pytest.approx(p)


def test_run_sparse_pobs_sweep_writes_csv_and_png(tmp_path: Path):
    root = tmp_path / "sweep"
    csv_path, png_path = run_sparse_pobs_sweep(
        out_root=root, num_rounds=2, seed=0, plot=True
    )
    assert csv_path.is_file()
    assert png_path is not None and png_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert "p_obs" in text
    assert text.count("\n") >= 5  # header + 4 rows
    assert "0.3" in text


def test_plot_only_from_csv(tmp_path: Path):
    root = tmp_path / "sweep"
    (root).mkdir()
    csv_path = root / "sparse_pobs_summary.csv"
    csv_path.write_text(
        "p_obs,run_id,num_rounds_executed,final_matched_cross_entropy,final_mce,"
        "stopped_on_convergence,convergence_round\n"
        "1.0,a,10,0.5,0.25,False,\n"
        "0.5,b,10,0.7,0.35,False,\n",
        encoding="utf-8",
    )
    out = root / "fig.png"
    plot_final_ce_vs_pobs(csv_path, out)
    assert out.stat().st_size > 80
