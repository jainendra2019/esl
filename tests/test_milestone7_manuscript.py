"""Milestone 7 orchestration (smoke) and MP cleanliness heuristic."""

from __future__ import annotations

from pathlib import Path

from esl.experiments.milestone7_manuscript import (
    _matching_pennies_clean_for_main,
    ensure_milestone7_manuscript_layout,
    run_milestone7_paper,
)


def test_ensure_m7_layout_creates_main_appendix(tmp_path: Path) -> None:
    ensure_milestone7_manuscript_layout(tmp_path / "mb")
    assert (tmp_path / "mb" / "main" / "figures").is_dir()
    assert (tmp_path / "mb" / "appendix" / "tables").is_dir()


def test_mp_cleanliness_heuristic(tmp_path: Path) -> None:
    p = tmp_path / "long.csv"
    p.write_text(
        "experiment,seed,focal_mean_payoff_per_round,final_mce\n"
        "matching_pennies,0,1.0,0.5\n"
        "matching_pennies,1,1.05,0.52\n"
        "matching_pennies,2,0.98,0.48\n",
        encoding="utf-8",
    )
    ok, meta = _matching_pennies_clean_for_main(p)
    assert ok is True
    assert meta.get("eligible") is True


def test_milestone7_smoke_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "out"
    mb = tmp_path / "mb"
    run_milestone7_paper(out_root=out, manuscript_bundle=mb, smoke=True)
    assert (mb / "EXPERIMENT_REPORT.md").is_file()
    assert (mb / "statistical_audit.json").is_file()
    assert (mb / "main" / "figures").exists()
    assert (mb / "reports" / "milestone_7" / "READINESS.md").is_file()
