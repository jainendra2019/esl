"""Milestone 2A: independent PPO baseline (official third_party) — smoke + schema only."""

from __future__ import annotations

from pathlib import Path

import pytest

from esl.baselines.ppo_adapter import third_party_ppo_root
from esl.experiment_registry import resolve_experiment_block
from esl.experiments.canonical_io import (
    MILESTONE_BUNDLE_PPO_BASELINE_FILES,
    create_milestone_artifact_bundle,
    ensure_manuscript_bundle_layout,
    manuscript_bundle_baseline_ppo_smoke_copy,
)
from esl.experiments.schema import validate_baseline_ppo_smoke_directory


def _require_official_ppo() -> None:
    if not (third_party_ppo_root() / "PPO.py").is_file():
        pytest.skip("third_party/PPO-PyTorch missing (git submodule update --init)")


def test_registry_has_baseline_ppo_block() -> None:
    b = resolve_experiment_block("baseline.ppo.recovery_smoke")
    assert b.run_kind == "baseline"


def test_ppo_recovery_smoke_writes_canonical_outputs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _require_official_ppo()
    from esl.baselines.ppo_adapter import run_recovery_mode_smoke

    out = tmp_path / "ppo_smoke"
    run_recovery_mode_smoke(
        out,
        seed=7,
        esl_run_dir="runs/milestone1_smoke",
        max_training_timesteps=96,
        update_timestep=32,
    )
    validate_baseline_ppo_smoke_directory(out)


def test_milestone_2a_bundle_and_manuscript_copy(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _require_official_ppo()
    from esl.baselines.ppo_adapter import run_recovery_mode_smoke

    run_dir = tmp_path / "ppo_for_bundle"
    run_recovery_mode_smoke(run_dir, seed=3, esl_run_dir="runs/milestone1_smoke")
    reports = tmp_path / "mb" / "reports"
    bundle = create_milestone_artifact_bundle(
        source_run_dir=run_dir,
        bundle_parent=reports,
        milestone_slug="milestone_2a_ppo",
        experiment_id="baseline.ppo.recovery_smoke",
        milestone_name="Milestone 2A — Independent PPO baseline onboarding",
        status="PASS",
        tests_summary="pytest tests/test_ppo_baseline_milestone2a.py",
        paper_faithfulness="Official PPO.py only; recovery-style fixed opponent in repeated PD; manifest + provenance per PRD §8A.",
        readiness="Stop after 2A; no sweeps. Await confirmation before further baselines.",
        open_issues="MCE omitted (null): policy-gradient baseline is not a prototype model.",
        relative_files=MILESTONE_BUNDLE_PPO_BASELINE_FILES,
    )
    art = bundle / "artifacts"
    for name in MILESTONE_BUNDLE_PPO_BASELINE_FILES:
        assert (art / name).is_file()
    assert (bundle / "MILESTONE_STATUS.md").is_file()

    mb = tmp_path / "manuscript_bundle"
    ensure_manuscript_bundle_layout(mb)
    copied = manuscript_bundle_baseline_ppo_smoke_copy(
        source_run_dir=run_dir, manuscript_bundle_root=mb, tag="m2a_ppo"
    )
    assert copied["manifest.json"].is_file()
