"""
Milestone 2B: canonical export for in-project static clustering (K-means, FCM).

Offline / static only (PRD §8A, BASELINE_PROTOCOL.md). No EM / Oracle in this export path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from esl.baselines.evaluate import (
    StaticClusteringMethod,
    fit_static_clustering_baseline,
)
from esl.baselines.dataset import load_run_dataset
from esl.experiment_registry import build_baseline_manifest_dict
from esl.experiments.canonical_io import write_manifest_json

ESL_REPO_URL = "https://github.com/jainendra2019/esl"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_static_clustering_baseline_smoke(
    esl_run_dir: Path,
    out_dir: Path,
    *,
    method: StaticClusteringMethod,
    seed: int,
    experiment_id: str,
    esl_run_dir_manifest: str,
    kmeans_n_init: int = 3,
    fcm_m: float = 2.0,
) -> Path:
    """
    Fit one clustering baseline on ``esl_run_dir`` observation log; write canonical artifacts.

    Requires ``interaction_observations.csv`` + ``config.json`` under ``esl_run_dir``.
    """
    esl_run_dir = esl_run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_run_dataset(esl_run_dir)

    fit = fit_static_clustering_baseline(
        ds,
        method=method,
        kmeans_seed=seed,
        fcm_seed=seed,
        fcm_m=fcm_m,
        kmeans_n_init=kmeans_n_init,
    )

    adapter = (
        "esl.baselines.kmeans_actions"
        if method == "kmeans"
        else "esl.baselines.fuzzy_cmeans"
    )
    run_cfg: dict[str, Any] = {
        "adapter": adapter,
        "clustering_method": method,
        "seed": seed,
        "esl_run_dir": str(esl_run_dir),
        "kmeans_n_init": kmeans_n_init,
        "fcm_m": fcm_m,
        "phi_feature_hash_sha256_prefix": fit["phi_feature_hash_sha256_prefix"],
        "protocol_doc": "docs/baselines/BASELINE_PROTOCOL.md",
    }
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2, sort_keys=True), encoding="utf-8")

    provenance = {
        "sources": [
            {
                "name": "ESL in-project static clustering",
                "url": ESL_REPO_URL,
                "local_path": "esl/baselines",
                "commit": None,
                "modules": [adapter, "esl.baselines.features_conditional", "esl.baselines.evaluate"],
            }
        ],
        "notes": "No third-party train loop; features and MCE mapping per docs/baselines/BASELINE_PROTOCOL.md.",
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    deviations = [
        "Batch static fit on full interaction_observations.csv (Protocol T style for smoke).",
        "No sequential belief or prototype learning in baseline path.",
    ]
    manifest = build_baseline_manifest_dict(
        experiment_id=experiment_id,
        seed=seed,
        esl_run_dir=esl_run_dir_manifest,
        adapter=adapter,
        deviations=deviations,
        provenance=provenance["sources"][0],
        extra={"clustering_method": method, "mce_mapping": "phi_matrix_to_logits + Hungarian MCE"},
    )
    write_manifest_json(out_dir / "manifest.json", manifest)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "baseline",
        "baseline_family": "offline_static_clustering",
        "clustering_method": method,
        "baseline_adapter": adapter,
        "seed": seed,
        "wall_time_sec": float(fit["wall_time_sec"]),
        "timesteps": int(fit["n_observed_w_positive"]),
        "final_mce": float(fit["final_mce"]),
        "final_matched_cross_entropy": float(fit["final_matched_cross_entropy"]),
        "final_belief_entropy": None,
        "final_belief_argmax_accuracy": None,
        "mean_payoff_per_agent_per_round": None,
        "cumulative_social_payoff": None,
        "mode": "recovery_static_fit_offline",
        "mce_note": "Hungarian-matched CE vs games.true_type_distributions after phi→logits map.",
        "objective": float(fit["objective"]),
        "code_origin": "in_project",
        "protocol_doc": "docs/baselines/BASELINE_PROTOCOL.md",
    }
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    traj = [
        {
            "round": 0,
            "fit_stage": "final",
            "objective_value": float(fit["objective"]),
            "final_mce": float(fit["final_mce"]),
        }
    ]
    _write_csv(out_dir / "metrics_trajectory.csv", traj)

    logits_path = out_dir / "clustering_logits.json"
    logits_path.write_text(
        json.dumps(np.asarray(fit["logits"]).tolist(), indent=2),
        encoding="utf-8",
    )

    return out_dir


def export_both_clustering_smokes(
    esl_run_dir: Path,
    out_root: Path,
    *,
    seed: int,
    esl_run_dir_manifest: str,
) -> tuple[Path, Path]:
    """Convenience for tests: K-means and FCM sibling folders under ``out_root``."""
    km = export_static_clustering_baseline_smoke(
        esl_run_dir,
        out_root / "baseline_kmeans_smoke",
        method="kmeans",
        seed=seed,
        experiment_id="baseline.cluster.kmeans_recovery_smoke",
        esl_run_dir_manifest=esl_run_dir_manifest,
    )
    fcm = export_static_clustering_baseline_smoke(
        esl_run_dir,
        out_root / "baseline_fcm_smoke",
        method="fcm",
        seed=seed + 1,
        experiment_id="baseline.cluster.fcm_recovery_smoke",
        esl_run_dir_manifest=esl_run_dir_manifest,
    )
    return km, fcm
