"""Adaptation payoff sweep smoke."""

from __future__ import annotations

import tempfile
from pathlib import Path

from esl.experiments.adaptation_payoff_sweep import run_adaptation_payoff_sweep


def test_adaptation_payoff_sweep_smoke():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ad"
        cpath, ppath = run_adaptation_payoff_sweep(
            out_root=root, num_rounds=4, seed=1, plot=True
        )
        assert cpath.is_file()
        assert ppath is not None and ppath.is_file()
        assert "adaptation_lambda" in cpath.read_text(encoding="utf-8")
