"""Aggregate CSV builder for NeurIPS runs."""

from __future__ import annotations

from pathlib import Path

from esl.experiments.aggregate import (
    AGGREGATE_COLUMNS_V1,
    find_run_directories,
    row_from_run_dir,
    write_aggregate_csv,
)


def test_find_run_directories_finds_fixture():
    root = Path(__file__).resolve().parent / "fixtures" / "neurips_mini"
    dirs = find_run_directories(root)
    assert len(dirs) == 1
    assert (dirs[0] / "summary_metrics.json").is_file()


def test_row_from_run_dir_shape():
    root = Path(__file__).resolve().parent / "fixtures" / "neurips_mini"
    rd = root / "flagship_smoke" / "seed_42"
    row = row_from_run_dir(rd, root=root)
    for col in AGGREGATE_COLUMNS_V1:
        assert col in row
    assert row["preset"] == "recovery_flagship"
    assert row["final_mce"] == 0.05
    assert row["run_id"].replace("\\", "/") == "flagship_smoke/seed_42"


def test_write_aggregate_csv(tmp_path: Path):
    import shutil

    src = Path(__file__).resolve().parent / "fixtures" / "neurips_mini"
    dst = tmp_path / "neurips_mini"
    shutil.copytree(src, dst)
    out = tmp_path / "out.csv"
    write_aggregate_csv(dst, out)
    text = out.read_text(encoding="utf-8")
    assert "run_id" in text
    assert "recovery_flagship" in text
    assert "0.05" in text
