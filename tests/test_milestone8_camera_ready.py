"""Milestone 8 camera-ready polish (no experiment re-runs)."""

from __future__ import annotations

import json
from pathlib import Path

from esl.experiments.milestone7_manuscript import ensure_milestone7_manuscript_layout
from esl.experiments.milestone8_camera_ready import (
    ONE_LINE_TAKEAWAY,
    run_milestone8_camera_ready,
)


def test_milestone8_renames_and_index(tmp_path: Path) -> None:
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
        (fig / name).write_bytes(b"\x89PNG\r\n\x1a\n")

    tbl = mb / "main/tables"
    for name in ("m6_k_misspec_aggregate.csv", "milestone5_method_comparison.csv"):
        (tbl / name).write_text("h\n0\n", encoding="utf-8")

    stale_cap = mb / "text/main/captions"
    stale_cap.mkdir(parents=True, exist_ok=True)
    (stale_cap / "fig_m4_legacy.md").write_text("old\n", encoding="utf-8")

    info = run_milestone8_camera_ready(mb)
    assert (fig / "fig1.png").is_file()
    assert (fig / "fig5.png").is_file()
    assert not (fig / "sparse_pobs_mce_ci.png").exists()
    assert (mb / "appendix/figures/m6_k_misspec_payoff_ci.png").is_file()
    assert (mb / "figure_index.json").is_file()
    assert ONE_LINE_TAKEAWAY in (mb / "text/main/abstract.txt").read_text(encoding="utf-8")
    assert not info.get("missing") or info["missing"] == []
    assert not (stale_cap / "fig_m4_legacy.md").exists()
    idx = json.loads((mb / "figure_index.json").read_text(encoding="utf-8"))
    by_id = {e["paper_id"]: e for e in idx["main_figures"]}
    assert by_id["fig1"]["paper_figure"] == "main/figures/fig1.png"
    assert by_id["fig1"]["original_artifact"] == "main/figures/sparse_pobs_mce_ci.png"
    assert by_id["fig5"]["original_artifact"] == "main/figures/m6_payoff_vs_mce_main_text.png"
