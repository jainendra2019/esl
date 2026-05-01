from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from esl.experiments.figure3_dynamics import REQUIRED_SUMMARY_COLUMNS, js_divergence, run_figure3_dynamics


def test_js_divergence_properties() -> None:
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.2, 0.7, 0.1])
    assert math.isclose(js_divergence(p, p), 0.0, abs_tol=1e-12)
    assert math.isclose(js_divergence(p, q), js_divergence(q, p), rel_tol=1e-12)
    assert js_divergence(p, q) > 0.0


def test_figure3_smoke_outputs_schema_and_gates(tmp_path: Path) -> None:
    outputs = run_figure3_dynamics(
        out_dir=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        seeds=[0, 1],
        games=("ipd", "stag_hunt", "matching_pennies"),
        horizon=400,
        batch_size=2,
        rolling_window=5,
    )

    assert outputs["figure"].is_file()
    assert outputs["figure"].stat().st_size > 10_000
    assert outputs["summary_csv"].is_file()
    assert outputs["report"].is_file()
    assert outputs["timeseries_csv"].is_file()

    rows = list(csv.DictReader(outputs["summary_csv"].open()))
    assert len(rows) == 6
    assert set(REQUIRED_SUMMARY_COLUMNS).issubset(rows[0].keys())

    numeric_cols = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in ("game", "seed")]
    for row in rows:
        for col in numeric_cols:
            assert np.isfinite(float(row[col])), col
        assert float(row["norm_growth_ratio"]) < 5.0

    assert np.mean([float(r["late_usage_jsd_mean"]) for r in rows]) < 0.25
    assert np.mean([float(r["late_action_jsd_mean"]) for r in rows]) < 0.25

    report = outputs["report"].read_text(encoding="utf-8")
    assert "bounded prototype dynamics" in report
    assert "invariant-regime proxy" in report
    assert "not a proof of ICT convergence" in report
    assert "consistent with the differential-inclusion characterization" in report
    assert "JSD gates are reporting diagnostics" in report
