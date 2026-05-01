from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np

from esl_analysis.pipeline import run_analysis_pipeline


def test_appendix_pipeline_outputs_and_reproducibility(tmp_path: Path) -> None:
    out1 = run_analysis_pipeline(
        output_dir=tmp_path / "a",
        seeds=[0],
        games=("stag_hunt",),
        methods=("ESL", "Fixed-K Bayesian", "Online EM"),
        horizon=260,
        t_entropy=50,
    )
    out2 = run_analysis_pipeline(
        output_dir=tmp_path / "b",
        seeds=[0],
        games=("stag_hunt",),
        methods=("ESL", "Fixed-K Bayesian", "Online EM"),
        horizon=260,
        t_entropy=50,
    )

    for key in (
        "switch_difficulty",
        "switch_difficulty_comparison",
        "entropy_curves",
        "accuracy_curves",
        "kl_curves",
        "entropy_plot",
        "accuracy_plot",
        "kl_plot",
    ):
        assert out1[key].is_file()

    entropy = np.load(out1["entropy_curves"], allow_pickle=True).item()
    accuracy = np.load(out1["accuracy_curves"], allow_pickle=True).item()
    kl = np.load(out1["kl_curves"], allow_pickle=True).item()
    assert entropy["ESL"]["mean"].shape[0] == 50
    assert accuracy["ESL"]["mean"].shape[0] == 50
    assert kl["ESL"]["mean"].shape[0] > 0
    assert np.isfinite(entropy["ESL"]["mean"]).all()
    assert np.isfinite(accuracy["ESL"]["mean"]).all()
    assert np.isfinite(kl["ESL"]["mean"]).all()
    assert not np.allclose(entropy["Online EM"]["mean"], np.log(3.0))

    entropy2 = np.load(out2["entropy_curves"], allow_pickle=True).item()
    assert np.allclose(entropy["ESL"]["mean"], entropy2["ESL"]["mean"])

    manifest = json.loads(out1["manifest"].read_text(encoding="utf-8"))
    assert "excluded_windows" in manifest
    assert manifest["controlled_switch_protocol"]["shifts"]["small"] == [0.2, 0.3]
    switch_rows = out1["switch_difficulty"].read_text(encoding="utf-8")
    assert "small" in switch_rows
    assert "medium" in switch_rows
    assert "large" in switch_rows
    comparison = list(csv.DictReader(out1["switch_difficulty_comparison"].open()))
    assert {r["difficulty"] for r in comparison} == {"small", "medium", "large"}
    for row in comparison:
        assert np.isfinite(float(row["esl_vs_fixed_k_delta"]))
        assert np.isfinite(float(row["esl_vs_em_delta"]))
        assert np.isfinite(float(row["esl_vs_best_baseline_delta"]))
    paired_rows = out1["paired_tests"].read_text(encoding="utf-8")
    assert "small" in paired_rows
    assert "medium" in paired_rows
    assert "large" in paired_rows
    report = out1["report"].read_text(encoding="utf-8")
    assert "KL is necessary but not sufficient for adaptation" in report
    assert "ESL's advantage is most visible under large behavioral shifts" in report
    assert "% lower PSR" in report
    assert "controlled synthetic shifts and illustrate trends rather than universal scaling laws" in report
    assert "Compared to Online EM" in report
    assert "avoiding premature commitment" in report
    assert "associated with higher posterior accuracy" in report
    assert "KL evaluates predictive fit to observed actions, not counterfactual decision quality" in report
    assert "accurate prediction does not imply optimal response" in report
