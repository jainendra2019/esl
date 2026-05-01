from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from esl.experiments.significance import (
    load_figure1_seed_pairs,
    paired_ttest_summary,
    write_figure1_significance,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "task", "seed", "horizon", "h_window", "focal_agent_id", "psr"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matched_rows(*, baseline_label: str = "Fixed-K Bayesian", offset: float = 10.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in range(3):
        for task, base in (("ipd", 100.0 + seed), ("stag_hunt", 50.0 + seed)):
            rows.append(
                {
                    "method": "ESL",
                    "task": task,
                    "seed": seed,
                    "horizon": 1000,
                    "h_window": 100,
                    "focal_agent_id": 0,
                    "psr": base,
                }
            )
            rows.append(
                {
                    "method": baseline_label,
                    "task": task,
                    "seed": seed,
                    "horizon": 1000,
                    "h_window": 100,
                    "focal_agent_id": 0,
                    "psr": base + offset,
                }
            )
    return rows


def test_seed_level_pairs_use_fixedk_minus_esl_direction_and_aliases(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    _write_rows(csv_path, _matched_rows(baseline_label="fixed_k", offset=12.0))

    pairs = load_figure1_seed_pairs(csv_path)

    assert len(pairs) == 3
    assert all(np.isclose(pair["diff"], 12.0) for pair in pairs)
    assert np.isclose(pairs[0]["psr_esl"], 75.0)
    assert np.isclose(pairs[0]["psr_baseline"], 87.0)


def test_missing_matched_seed_or_game_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    rows = _matched_rows()
    rows = [row for row in rows if not (row["method"] == "Fixed-K Bayesian" and row["seed"] == 2 and row["task"] == "ipd")]
    _write_rows(csv_path, rows)

    with pytest.raises(ValueError, match="missing matched method row"):
        load_figure1_seed_pairs(csv_path)


def test_fewer_than_three_pairs_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    rows = [row for row in _matched_rows() if int(row["seed"]) < 2]
    _write_rows(csv_path, rows)

    with pytest.raises(ValueError, match="at least 3"):
        load_figure1_seed_pairs(csv_path)


def test_degenerate_zero_variance_differences_are_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    _write_rows(csv_path, _matched_rows(offset=5.0))

    result = paired_ttest_summary(load_figure1_seed_pairs(csv_path))

    assert result["mean_diff"] == 5.0
    assert result["ci95_low"] == 5.0
    assert result["ci95_high"] == 5.0
    assert np.isinf(float(result["t_stat"]))
    assert result["p_value"] == 0.0
    assert np.isinf(float(result["cohen_dz"]))

    outputs = write_figure1_significance(summary_csv=csv_path, manuscript_bundle=tmp_path / "bundle_degenerate")
    report = outputs["report"].read_text(encoding="utf-8")
    assert "p < 0.001" in report
    assert "p = <" not in report


def test_outputs_and_non_significant_report_wording(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    rows = _matched_rows(offset=0.0)
    # Seed-level differences are positive but noisy enough to be non-significant.
    for row in rows:
        if row["method"] == "Fixed-K Bayesian" and row["seed"] == 0:
            row["psr"] = float(row["psr"]) + 3.0
        if row["method"] == "Fixed-K Bayesian" and row["seed"] == 1:
            row["psr"] = float(row["psr"]) + 3.0
        if row["method"] == "Fixed-K Bayesian" and row["seed"] == 2:
            row["psr"] = float(row["psr"]) - 1.0
    _write_rows(csv_path, rows)

    outputs = write_figure1_significance(summary_csv=csv_path, manuscript_bundle=tmp_path / "bundle")

    assert outputs["table"].is_file()
    assert outputs["report"].is_file()
    report = outputs["report"].read_text(encoding="utf-8")
    manuscript_sentence = report.split("## Manuscript Sentence", maxsplit=1)[1]
    assert "significant" not in manuscript_sentence
    assert "paired 95% CI" in manuscript_sentence
