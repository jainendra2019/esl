"""Ablation ladder smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

from esl.experiments.ablation_ladder import run_ablation_ladder


def test_ablation_ladder_writes_csv_png():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "abl"
        csv_path, png_path = run_ablation_ladder(
            out_root=root, num_rounds=3, seed=0, plot=True
        )
        assert csv_path.is_file()
        assert png_path is not None and png_path.is_file()
        txt = csv_path.read_text(encoding="utf-8")
        assert "full" in txt
        assert "freeze_prototype" in txt
        assert "learning_frozen" in txt
