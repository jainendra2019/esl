"""Milestone 1: registry, canonical manifests, summary schema, bundle scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest

from esl.experiment_registry import (
    build_baseline_manifest_dict,
    list_experiment_block_ids,
    resolve_experiment_block,
    esl_config_for_block,
)
from esl.experiments.canonical_io import (
    create_milestone_artifact_bundle,
    ensure_manuscript_bundle_layout,
    manuscript_bundle_smoke_copy,
)
from esl.experiments.registry_smoke import run_registry_esl
from esl.experiments.schema import (
    validate_esl_run_directory,
    validate_manifest_baseline,
    validate_summary_metrics,
)


def test_registry_resolves_all_named_blocks() -> None:
    ids = list_experiment_block_ids()
    assert "milestone.smoke_lock" in ids
    assert "baseline.offline.protocol_stub" in ids
    for bid in ids:
        b = resolve_experiment_block(bid)
        assert b.id == bid
    # ESL blocks (excluding baseline stub) must yield a config
    for bid in ids:
        block = resolve_experiment_block(bid)
        if block.run_kind != "esl":
            continue
        cfg, slug = esl_config_for_block(bid, seed=7, smoke=True)
        assert slug
        cfg.validate()


def test_baseline_stub_is_not_esl_runnable() -> None:
    with pytest.raises(ValueError, match="not an ESL"):
        esl_config_for_block("baseline.offline.protocol_stub", seed=0, smoke=True)


def test_smoke_run_emits_required_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "smoke_run"
    run_registry_esl("milestone.smoke_lock", run_dir, seed=11, smoke=True)
    for name in ("config.json", "manifest.json", "summary_metrics.json", "metrics_trajectory.csv"):
        assert (run_dir / name).is_file(), f"missing {name}"


def test_schema_validation_esl_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "esl_schema"
    run_registry_esl("milestone.smoke_lock", run_dir, seed=1, smoke=True)
    validate_esl_run_directory(run_dir)


def test_schema_validation_baseline_manifest() -> None:
    m = build_baseline_manifest_dict(
        experiment_id="baseline.offline.em_conditional",
        seed=42,
        esl_run_dir="runs/example/seed_42",
        adapter="em_conditional_bernoulli",
        deviations=["stub milestone-1 only"],
        provenance={"url": "https://example.invalid", "commit": None},
    )
    validate_manifest_baseline(m)


def test_milestone_artifact_bundle_created(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_src"
    run_registry_esl("milestone.smoke_lock", run_dir, seed=2, smoke=True)
    reports = tmp_path / "manuscript_bundle" / "reports"
    bundle = create_milestone_artifact_bundle(
        source_run_dir=run_dir,
        bundle_parent=reports,
        milestone_slug="milestone_1_smoke",
        experiment_id="milestone.smoke_lock",
        milestone_name="Milestone 1 — Experiment registry and result schema lock",
        status="PASS",
        tests_summary="pytest tests/test_experiment_registry_milestone1.py (this file)",
        paper_faithfulness="Manifest + trajectory field names align with PRD §10–11 and ALGORITHM.md run outputs.",
        readiness="Registry and schema validators are in place; await confirmation before Milestone 2.",
        open_issues="None for M1 smoke scope.",
    )
    art = bundle / "artifacts"
    for name in ("config.json", "manifest.json", "summary_metrics.json", "metrics_trajectory.csv"):
        assert (art / name).is_file()
    assert (bundle / "MILESTONE_STATUS.md").is_file()


def test_manuscript_bundle_smoke_layout_and_copy(tmp_path: Path) -> None:
    root = tmp_path / "manuscript_bundle"
    ensure_manuscript_bundle_layout(root)
    assert (root / "figures").is_dir()
    assert (root / "reports" / "templates").is_dir()
    run_dir = tmp_path / "run_for_copy"
    run_registry_esl("milestone.smoke_lock", run_dir, seed=3, smoke=True)
    out = manuscript_bundle_smoke_copy(
        source_run_dir=run_dir, manuscript_bundle_root=root, tag="m1"
    )
    assert out["manifest"].is_file()
    data = __import__("json").loads(out["summary_metrics"].read_text(encoding="utf-8"))
    validate_summary_metrics(data)


def test_summary_metrics_includes_schema_version(tmp_path: Path) -> None:
    """Regression: trainer must emit schema_version for canonical validation."""
    from esl.trainer import run_esl
    from esl.config import ESLConfig

    rd = tmp_path / "t"
    cfg = ESLConfig(seed=0, mode="recovery", num_rounds=1, num_agents=3, num_prototypes=2)
    cfg.validate()
    run_esl(cfg, run_dir=rd)
    from esl.experiments.schema import load_json

    data = load_json(rd / "summary_metrics.json")
    validate_summary_metrics(data)
