"""Milestone 2B: in-project K-means + FCM static clustering — canonical export + schema."""

from __future__ import annotations

import tempfile
from pathlib import Path

from esl.baselines.clustering_smoke_export import (
    export_both_clustering_smokes,
    export_static_clustering_baseline_smoke,
)
from esl.baselines.evaluate import fit_static_clustering_baseline
from esl.baselines.dataset import load_run_dataset
from esl.config import ESLConfig
from esl.experiment_registry import resolve_experiment_block
from esl.experiments.canonical_io import (
    MILESTONE_BUNDLE_OFFLINE_BASELINE_FILES,
    create_milestone_artifact_bundle,
    ensure_manuscript_bundle_layout,
    manuscript_bundle_baseline_ppo_smoke_copy,
)
from esl.experiments.schema import validate_baseline_clustering_smoke_directory
from esl.trainer import run_esl


def _tiny_esl_run_with_obs_log(tmp_path: Path) -> Path:
    cfg = ESLConfig(
        seed=5,
        mode="recovery",
        num_rounds=40,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=2,
        log_interaction_observations=True,
        force_agent_true_types=[0, 0, 1, 1],
        symmetric_init=False,
        init_noise=0.05,
    )
    cfg.validate()
    run_dir = tmp_path / "esl_for_cluster"
    run_esl(cfg, run_dir=run_dir)
    assert (run_dir / "interaction_observations.csv").is_file()
    return run_dir


def test_registry_has_clustering_blocks() -> None:
    assert resolve_experiment_block("baseline.cluster.kmeans_recovery_smoke").run_kind == "baseline"
    assert resolve_experiment_block("baseline.cluster.fcm_recovery_smoke").run_kind == "baseline"


def test_fit_static_clustering_kmeans_and_fcm() -> None:
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "r"
        cfg = ESLConfig(
            seed=1,
            num_rounds=30,
            num_agents=4,
            num_prototypes=2,
            log_interaction_observations=True,
            force_agent_true_types=[0, 1, 0, 1],
        )
        cfg.validate()
        run_esl(cfg, run_dir=run_dir)
        ds = load_run_dataset(run_dir)
        km = fit_static_clustering_baseline(ds, method="kmeans", kmeans_seed=0, kmeans_n_init=2)
        fcm = fit_static_clustering_baseline(ds, method="fcm", fcm_seed=0)
    assert km["final_mce"] == km["final_mce"]
    assert fcm["method"] == "fcm"
    assert km["final_mce"] >= 0.0


def test_clustering_smoke_exports_validate(tmp_path: Path) -> None:
    esl = _tiny_esl_run_with_obs_log(tmp_path)
    root = tmp_path / "cluster_exports"
    km_dir, fcm_dir = export_both_clustering_smokes(
        esl, root, seed=9, esl_run_dir_manifest="runs/example_esl"
    )
    validate_baseline_clustering_smoke_directory(km_dir)
    validate_baseline_clustering_smoke_directory(fcm_dir)
    assert (km_dir / "clustering_logits.json").is_file()


def test_clustering_single_fcm_export(tmp_path: Path) -> None:
    esl = _tiny_esl_run_with_obs_log(tmp_path)
    out = tmp_path / "fcm_only"
    export_static_clustering_baseline_smoke(
        esl,
        out,
        method="fcm",
        seed=2,
        experiment_id="baseline.cluster.fcm_recovery_smoke",
        esl_run_dir_manifest=str(esl),
    )
    validate_baseline_clustering_smoke_directory(out)


def test_milestone_2b_bundle_kmeans(tmp_path: Path) -> None:
    esl = _tiny_esl_run_with_obs_log(tmp_path)
    run_dir = tmp_path / "km_out"
    export_static_clustering_baseline_smoke(
        esl,
        run_dir,
        method="kmeans",
        seed=11,
        experiment_id="baseline.cluster.kmeans_recovery_smoke",
        esl_run_dir_manifest="runs/m2b_ref",
    )
    validate_baseline_clustering_smoke_directory(run_dir)
    reports = tmp_path / "mb" / "reports"
    bundle = create_milestone_artifact_bundle(
        source_run_dir=run_dir,
        bundle_parent=reports,
        milestone_slug="milestone_2b_kmeans",
        experiment_id="baseline.cluster.kmeans_recovery_smoke",
        milestone_name="Milestone 2B — K-means static clustering (smoke)",
        status="PASS",
        tests_summary="pytest tests/test_clustering_baseline_milestone2b.py",
        paper_faithfulness="Offline K-means on phi features; MCE via Hungarian vs true_type_distributions; PRD §8A in-project baseline.",
        readiness="Stop after 2B; no M-FOS/MBOM.",
        open_issues="PRD 'GMM' mixture is EM (separate milestone); 2B is K-means + FCM only.",
        relative_files=MILESTONE_BUNDLE_OFFLINE_BASELINE_FILES,
    )
    for name in MILESTONE_BUNDLE_OFFLINE_BASELINE_FILES:
        assert (bundle / "artifacts" / name).is_file()

    mb = tmp_path / "manuscript_bundle"
    ensure_manuscript_bundle_layout(mb)
    copied = manuscript_bundle_baseline_ppo_smoke_copy(
        source_run_dir=run_dir, manuscript_bundle_root=mb, tag="m2b_kmeans"
    )
    assert copied["manifest.json"].is_file()
