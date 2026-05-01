from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from esl.experiments.figure4_rm_dynamics import SUMMARY_COLUMNS, run_figure4_rm_dynamics


def test_figure4_rm_smoke_outputs_schema_and_theory_diagnostics(tmp_path: Path) -> None:
    outputs = run_figure4_rm_dynamics(
        out_dir=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        seeds=[0, 1],
        games=("stag_hunt",),
        horizon=400,
        batch_size=20,
        rm_c=0.5,
        eta_reg=1e-3,
        rolling_window=5,
    )

    for key in ("figure", "summary_csv", "report", "gate_json", "timeseries_csv"):
        assert outputs[key].is_file(), key
    assert outputs["figure"].stat().st_size > 10_000

    summary_rows = list(csv.DictReader(outputs["summary_csv"].open()))
    assert len(summary_rows) == 2
    assert set(SUMMARY_COLUMNS).issubset(summary_rows[0].keys())

    numeric_cols = [c for c in SUMMARY_COLUMNS if c not in ("game", "seed")]
    for row in summary_rows:
        for col in numeric_cols:
            assert np.isfinite(float(row[col])), col
        assert float(row["rm_c"]) == 0.5
        assert float(row["eta_reg"]) == 1e-3
        assert int(float(row["batch_size"])) == 20
        assert float(row["gamma_first"]) > float(row["gamma_last"])
        assert float(row["cumulative_sum_gamma_last"]) > 0.0
        assert float(row["cumulative_sum_gamma_sq_last"]) > 0.0
        assert np.isfinite(float(row["max_prototype_norm"]))
        assert float(row["effective_step_ratio"]) < 1.25

    timeseries_rows = list(csv.DictReader(outputs["timeseries_csv"].open()))
    assert timeseries_rows
    for row in timeseries_rows:
        for col in (
            "gamma_m",
            "cumulative_sum_gamma",
            "cumulative_sum_gamma_sq",
            "effective_step_ratio",
            "prototype_norm",
            "prototype_step_size",
        ):
            assert np.isfinite(float(row[col])), col

    gate = json.loads(outputs["gate_json"].read_text(encoding="utf-8"))
    assert gate["pass"] is True
    assert gate["checks"]["gamma_decreases"] is True
    assert gate["checks"]["gamma_cumulative_diagnostics_finite"] is True
    assert gate["checks"]["effective_step_decreases_or_controlled"] is True

    report = outputs["report"].read_text(encoding="utf-8")
    assert "Diminishing-Step Diagnostic for Asymptotic Dynamics" in report
    assert "consistent with the dynamical structure implied by the differential inclusion" in report
    assert "not meant to beat constant-step ESL on PSR" in report
    assert "not an empirical proof of convergence" in report
