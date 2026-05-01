"""Milestone 3: sparse p_obs + Q sweep + pack (ESL + K-means + FCM only for batch path)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from esl.baselines.evaluate import run_baselines_on_dataset
from esl.baselines.dataset import load_run_dataset
from esl.config import ESLConfig
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.milestone3_pack import run_milestone3_smoke_pack, run_freeze_theta_self_consistency
from esl.experiments.q_recovery_sweep import run_q_recovery_sweep
from esl.experiments.sparse_pobs_sweep import run_sparse_pobs_sweep
from esl.trainer import run_esl


def test_clustering_only_baselines_skips_em_oracle(tmp_path: Path) -> None:
    cfg = ESLConfig(
        seed=1,
        num_rounds=25,
        num_agents=4,
        num_prototypes=2,
        log_interaction_observations=True,
        force_agent_true_types=[0, 0, 1, 1],
    )
    cfg.validate()
    run_dir = tmp_path / "esl"
    run_esl(cfg, run_dir=run_dir)
    ds = load_run_dataset(run_dir)
    full = run_baselines_on_dataset(ds, em_restarts=2, clustering_only=False)
    assert "em_conditional_mce" in full
    part = run_baselines_on_dataset(ds, em_restarts=2, clustering_only=True)
    assert "kmeans_mce" in part and "fcm_mce" in part
    assert "em_conditional_mce" not in part
    assert "oracle_mce" not in part


def test_sparse_pobs_sweep_includes_fcm_column_and_clustering_only(tmp_path: Path) -> None:
    root = tmp_path / "sp"
    csv_path, _ = run_sparse_pobs_sweep(
        out_root=root,
        num_rounds=2,
        seed=0,
        plot=False,
        include_baselines=True,
        em_restarts=2,
        clustering_only_baselines=True,
    )
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert len(rows) == 4
    assert all(r.get("fcm_mce", "").strip() != "" for r in rows)


def test_q_recovery_sweep_smoke(tmp_path: Path) -> None:
    root = tmp_path / "q"
    csv_path, png = run_q_recovery_sweep(
        out_root=root,
        num_rounds=2,
        seed=0,
        q_values=(1, 2),
        plot=True,
        include_baselines=True,
        em_restarts=2,
        clustering_only_baselines=True,
    )
    assert csv_path.is_file()
    assert png is not None and png.is_file()


def test_milestone3_smoke_pack_with_manuscript(tmp_path: Path) -> None:
    mb = tmp_path / "manuscript_bundle"
    out = tmp_path / "m3"
    info = run_milestone3_smoke_pack(
        out,
        manuscript_bundle=mb,
        sparse_rounds=2,
        q_rounds=2,
        q_values=(1, 2),
        seed=0,
        plot=True,
    )
    assert info["freeze_ok"] is True
    assert (mb / "tables" / "milestone3_sparse_pobs_summary.csv").is_file()
    assert (mb / "figures" / "milestone3_q_vs_mce.png").is_file()
    assert (mb / "reports" / "milestone_3" / "MILESTONE_STATUS.md").is_file()


def test_publication_dpi_env(monkeypatch) -> None:
    monkeypatch.setenv("ESL_PUBLICATION_DPI", "300")
    assert publication_figure_dpi() == 300
    monkeypatch.delenv("ESL_PUBLICATION_DPI", raising=False)
    assert publication_figure_dpi() == 150


def test_freeze_diagnostic_json(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    out = run_freeze_theta_self_consistency(p, seed=3)
    assert out["ok"] is True
    assert json.loads(p.read_text(encoding="utf-8"))["ok"] is True
