"""Milestone 9 submission bundle and verification."""

from __future__ import annotations

import json
from pathlib import Path

from esl.experiments.milestone7_manuscript import ensure_milestone7_manuscript_layout
from esl.experiments.milestone9_submission import run_milestone9_submission


def test_milestone9_assembles_and_checklist(tmp_path: Path) -> None:
    mb = tmp_path / "mb"
    ensure_milestone7_manuscript_layout(mb)
    fig = mb / "main/figures"
    for name in (
        "sparse_pobs_mce_ci.png",
        "q_recovery_mce_ci.png",
        "milestone5_adaptation_focal_payoff.png",
        "milestone5_method_comparison_bars.png",
        "m6_payoff_vs_mce_main_text.png",
        "m6_k_misspec_payoff_ci.png",
    ):
        (fig / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 600)

    tbl = mb / "main/tables"
    tbl.mkdir(parents=True, exist_ok=True)
    (tbl / "m6_k_misspec_aggregate.csv").write_text("h\n0\n", encoding="utf-8")

    info = run_milestone9_submission(mb, min_fig_bytes=1)
    assert info["checklist"]["all_pass"] is True
    assert (mb / "submission_bundle/paper/figures/fig1.png").is_file()
    assert (mb / "submission_bundle/paper/text/figure_inclusion_stub.tex").is_file()
    assert "includegraphics" in (mb / "submission_bundle/paper/text/figure_inclusion_stub.tex").read_text(
        encoding="utf-8"
    )
    assert (mb / "reports/milestone_9/REVIEWER_SIMULATION.md").is_file()
    assert (mb / "reports/milestone_9/ACCEPT_REJECT_RISK.md").is_file()
    assert (mb / "text/main/abstract_refined.md").is_file()
    chk = json.loads((mb / "reports/milestone_9/checklist_results.json").read_text(encoding="utf-8"))
    assert chk["main_figures_present"] is True
