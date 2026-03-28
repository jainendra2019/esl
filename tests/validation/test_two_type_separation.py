"""Main two-type recovery experiment: separation vs symmetric baseline (validation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from esl.experiment_two_type_separation import (
    acceptance_report,
    two_type_separation_config,
)
from esl.trainer import run_esl


@pytest.mark.slow
@pytest.mark.validation
def test_two_type_main_prototype_separation_thresholds():
    """Asymmetric init + prototype learning → matched purity thresholds; seed 42, T=250, lr=18."""
    cfg = two_type_separation_config("main", num_rounds=250, seed=42)
    assert cfg.prototype_lr_scale == 18.0
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    rep = acceptance_report(logits, summary)
    assert rep["prototype_separation_thresholds_met"], rep
    assert rep["belief_accuracy_threshold_90pct"]
    assert float(summary["final_matched_cross_entropy"]) < 0.6


@pytest.mark.slow
@pytest.mark.validation
def test_two_type_symmetric_baseline_stays_non_separated():
    """θ_1=θ_2=0: prototypes should not specialize (matched purities stay ~0.5)."""
    cfg = two_type_separation_config("symmetric_baseline", num_rounds=250, seed=42)
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    rep = acceptance_report(logits, summary)
    assert not rep["prototype_separation_thresholds_met"]
    assert rep["belief_argmax_accuracy"] < 0.6
    assert 0.4 < rep["matched_P_coop_type_on_C"] < 0.62
    assert 0.38 < rep["matched_P_defect_type_on_D"] < 0.62


@pytest.mark.slow
@pytest.mark.validation
def test_two_type_freeze_proto_logits_unchanged():
    cfg = two_type_separation_config("freeze_proto_baseline", num_rounds=30, seed=0)
    theta0 = np.array(cfg.prototype_logits_override, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    np.testing.assert_allclose(logits, theta0, rtol=1e-9, atol=1e-9)
    assert summary["prototype_update_count"] == 0
