"""JSON-side validation for canonical manifests and metric exports (PRD §11, §13)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from esl.experiment_registry import MANIFEST_SCHEMA_VERSION, SUMMARY_METRICS_SCHEMA_VERSION

_TRAJECTORY_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "round",
        "belief_entropy_mean",
        "matched_cross_entropy",
        "belief_argmax_accuracy",
        "batch_log_likelihood",
        "prototype_update_norm",
        "belief_change_norm",
    }
)

_SUMMARY_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "final_mce",
        "final_matched_cross_entropy",
        "final_belief_entropy",
        "final_belief_argmax_accuracy",
        "mean_payoff_per_agent_per_round",
        "cumulative_social_payoff",
        "seed",
        "mode",
        "schema_version",
    }
)

_BASELINE_PPO_SUMMARY_REQUIRED: frozenset[str] = frozenset(
    {
        "schema_version",
        "run_kind",
        "baseline_adapter",
        "seed",
        "wall_time_sec",
        "timesteps",
        "mean_payoff_per_agent_per_round",
        "cumulative_social_payoff",
        "official_repo_commit",
        "mce_note",
        "final_mce",
        "final_matched_cross_entropy",
        "final_belief_entropy",
        "final_belief_argmax_accuracy",
        "mode",
    }
)

_BASELINE_PPO_TRAJECTORY_COLS: frozenset[str] = frozenset(
    {"round", "timestep", "ppo_step_reward", "opponent_action", "row_action"}
)

_POLICY_BASELINE_FAMILIES: frozenset[str] = frozenset(
    {"independent_ppo", "mfos", "mbom", "simple_opponent_model"}
)

_BASELINE_POLICY_SUMMARY_REQUIRED: frozenset[str] = frozenset(
    {
        "schema_version",
        "run_kind",
        "baseline_adapter",
        "baseline_family",
        "seed",
        "wall_time_sec",
        "timesteps",
        "mean_payoff_per_agent_per_round",
        "cumulative_social_payoff",
        "official_repo_commit",
        "mce_note",
        "final_mce",
        "final_matched_cross_entropy",
        "final_belief_entropy",
        "final_belief_argmax_accuracy",
        "mode",
        "task",
        "regime",
        "availability_status",
    }
)

_BASELINE_POLICY_TRAJECTORY_COLS: frozenset[str] = frozenset(
    {
        "round",
        "timestep",
        "task",
        "regime",
        "method",
        "row_action",
        "opponent_action",
        "row_reward",
        "opponent_reward",
        "mean_payoff_per_agent",
    }
)

_BASELINE_CLUSTER_SUMMARY_REQUIRED: frozenset[str] = frozenset(
    {
        "schema_version",
        "run_kind",
        "baseline_family",
        "clustering_method",
        "baseline_adapter",
        "seed",
        "wall_time_sec",
        "timesteps",
        "final_mce",
        "final_matched_cross_entropy",
        "final_belief_entropy",
        "final_belief_argmax_accuracy",
        "mean_payoff_per_agent_per_round",
        "cumulative_social_payoff",
        "mode",
        "mce_note",
        "code_origin",
        "protocol_doc",
    }
)

_BASELINE_CLUSTER_TRAJECTORY_COLS: frozenset[str] = frozenset(
    {"round", "fit_stage", "objective_value", "final_mce"}
)

_PROVENANCE_REQUIRED_TOP: frozenset[str] = frozenset({"sources"})


def validate_manifest_esl(m: dict[str, Any]) -> None:
    if m.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest: schema_version mismatch for ESL")
    if m.get("run_kind") != "esl":
        raise ValueError("manifest: run_kind must be 'esl'")
    for k in ("experiment_id", "seed", "esl"):
        if k not in m:
            raise ValueError(f"manifest: missing key {k!r}")
    esl = m["esl"]
    if not isinstance(esl, dict):
        raise ValueError("manifest: esl must be an object")
    if "variant" not in esl:
        raise ValueError("manifest: esl.variant required")


def validate_manifest_baseline(m: dict[str, Any]) -> None:
    if m.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest: schema_version mismatch for baseline")
    if m.get("run_kind") != "baseline":
        raise ValueError("manifest: run_kind must be 'baseline'")
    for k in ("experiment_id", "seed", "baseline"):
        if k not in m:
            raise ValueError(f"manifest: missing key {k!r}")
    b = m["baseline"]
    if not isinstance(b, dict):
        raise ValueError("manifest: baseline must be an object")
    for k in ("adapter", "esl_run_dir"):
        if k not in b:
            raise ValueError(f"manifest: baseline.{k} required")


def validate_summary_metrics_baseline_clustering(obj: dict[str, Any]) -> None:
    if obj.get("schema_version") != SUMMARY_METRICS_SCHEMA_VERSION:
        raise ValueError("summary_metrics (baseline clustering): schema_version mismatch")
    if obj.get("run_kind") != "baseline":
        raise ValueError("summary_metrics (baseline clustering): run_kind must be 'baseline'")
    if obj.get("baseline_family") != "offline_static_clustering":
        raise ValueError("summary_metrics (baseline clustering): baseline_family mismatch")
    missing = sorted(_BASELINE_CLUSTER_SUMMARY_REQUIRED - obj.keys())
    if missing:
        raise ValueError(f"summary_metrics (baseline clustering): missing keys: {missing}")
    if obj.get("clustering_method") not in ("kmeans", "fcm"):
        raise ValueError("clustering_method must be 'kmeans' or 'fcm'")


def validate_summary_metrics_baseline_ppo(obj: dict[str, Any]) -> None:
    if obj.get("schema_version") != SUMMARY_METRICS_SCHEMA_VERSION:
        raise ValueError("summary_metrics (baseline PPO): schema_version mismatch")
    if obj.get("run_kind") != "baseline":
        raise ValueError("summary_metrics (baseline PPO): run_kind must be 'baseline'")
    missing = sorted(_BASELINE_PPO_SUMMARY_REQUIRED - obj.keys())
    if missing:
        raise ValueError(f"summary_metrics (baseline PPO): missing keys: {missing}")


def validate_summary_metrics_baseline_policy(obj: dict[str, Any]) -> None:
    if obj.get("schema_version") != SUMMARY_METRICS_SCHEMA_VERSION:
        raise ValueError("summary_metrics (baseline policy): schema_version mismatch")
    if obj.get("run_kind") != "baseline":
        raise ValueError("summary_metrics (baseline policy): run_kind must be 'baseline'")
    if obj.get("baseline_family") not in _POLICY_BASELINE_FAMILIES:
        raise ValueError("summary_metrics (baseline policy): baseline_family mismatch")
    missing = sorted(_BASELINE_POLICY_SUMMARY_REQUIRED - obj.keys())
    if missing:
        raise ValueError(f"summary_metrics (baseline policy): missing keys: {missing}")


def validate_summary_metrics(obj: dict[str, Any]) -> None:
    if obj.get("schema_version") != SUMMARY_METRICS_SCHEMA_VERSION:
        raise ValueError("summary_metrics: schema_version missing or mismatch")
    if obj.get("run_kind") == "baseline":
        fam = obj.get("baseline_family")
        if fam == "offline_static_clustering":
            validate_summary_metrics_baseline_clustering(obj)
        elif fam == "independent_ppo" or obj.get("baseline_adapter") == "third_party.PPO-PyTorch":
            if "task" in obj and "regime" in obj:
                validate_summary_metrics_baseline_policy(obj)
            else:
                validate_summary_metrics_baseline_ppo(obj)
        elif fam in _POLICY_BASELINE_FAMILIES:
            validate_summary_metrics_baseline_policy(obj)
        else:
            raise ValueError(
                f"summary_metrics: unknown baseline_family for baseline run: {fam!r}"
            )
        return
    # ESL trainer output (no run_kind field before Milestone 2A)
    missing = sorted(_SUMMARY_REQUIRED_KEYS - obj.keys())
    if missing:
        raise ValueError(f"summary_metrics: missing keys: {missing}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metrics_trajectory_baseline_clustering(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("metrics_trajectory.csv (baseline clustering): empty")
    reader = csv.DictReader(raw.splitlines())
    if reader.fieldnames is None:
        raise ValueError("metrics_trajectory.csv: no header")
    fields = set(reader.fieldnames)
    missing = sorted(_BASELINE_CLUSTER_TRAJECTORY_COLS - fields)
    if missing:
        raise ValueError(f"metrics_trajectory.csv (baseline clustering): missing columns {missing}")


def validate_metrics_trajectory_baseline_ppo(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("metrics_trajectory.csv (baseline PPO): empty")
    reader = csv.DictReader(raw.splitlines())
    if reader.fieldnames is None:
        raise ValueError("metrics_trajectory.csv: no header")
    fields = set(reader.fieldnames)
    missing = sorted(_BASELINE_PPO_TRAJECTORY_COLS - fields)
    if missing:
        raise ValueError(f"metrics_trajectory.csv (baseline PPO): missing columns {missing}")


def validate_metrics_trajectory_baseline_policy(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("metrics_trajectory.csv (baseline policy): empty")
    reader = csv.DictReader(raw.splitlines())
    if reader.fieldnames is None:
        raise ValueError("metrics_trajectory.csv: no header")
    fields = set(reader.fieldnames)
    missing = sorted(_BASELINE_POLICY_TRAJECTORY_COLS - fields)
    if missing:
        raise ValueError(f"metrics_trajectory.csv (baseline policy): missing columns {missing}")


def validate_provenance_json(path: Path) -> None:
    data = load_json(path)
    missing = sorted(_PROVENANCE_REQUIRED_TOP - data.keys())
    if missing:
        raise ValueError(f"provenance.json: missing keys {missing}")
    srcs = data["sources"]
    if not isinstance(srcs, list) or not srcs:
        raise ValueError("provenance.json: sources must be a non-empty list")
    for i, s in enumerate(srcs):
        if not isinstance(s, dict):
            raise ValueError(f"provenance.json: sources[{i}] must be an object")
        for k in ("name", "url", "local_path"):
            if k not in s:
                raise ValueError(f"provenance.json: sources[{i}] missing {k!r}")


def validate_metrics_trajectory_csv(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("metrics_trajectory.csv is empty but schema requires per-round rows for ESL")
    reader = csv.DictReader(raw.splitlines())
    if reader.fieldnames is None:
        raise ValueError("metrics_trajectory.csv: no header")
    fields = set(reader.fieldnames)
    missing = sorted(_TRAJECTORY_REQUIRED_COLUMNS - fields)
    if missing:
        raise ValueError(f"metrics_trajectory.csv: missing columns {missing}")


def validate_esl_run_directory(run_dir: Path) -> None:
    """Validate manifest + summary + trajectory for an ESL run folder."""
    validate_manifest_esl(load_json(run_dir / "manifest.json"))
    validate_summary_metrics(load_json(run_dir / "summary_metrics.json"))
    validate_metrics_trajectory_csv(run_dir / "metrics_trajectory.csv")


def validate_baseline_ppo_smoke_directory(run_dir: Path) -> None:
    """Milestone 2A: canonical baseline smoke outputs for independent PPO."""
    validate_manifest_baseline(load_json(run_dir / "manifest.json"))
    validate_summary_metrics(load_json(run_dir / "summary_metrics.json"))
    validate_metrics_trajectory_baseline_ppo(run_dir / "metrics_trajectory.csv")
    validate_provenance_json(run_dir / "provenance.json")


def validate_baseline_clustering_smoke_directory(run_dir: Path) -> None:
    """Milestone 2B: K-means / FCM static clustering baseline smoke."""
    validate_manifest_baseline(load_json(run_dir / "manifest.json"))
    validate_summary_metrics(load_json(run_dir / "summary_metrics.json"))
    validate_metrics_trajectory_baseline_clustering(run_dir / "metrics_trajectory.csv")
    validate_provenance_json(run_dir / "provenance.json")


def validate_baseline_policy_smoke_directory(run_dir: Path) -> None:
    """Canonical smoke outputs for policy-performance baselines."""
    validate_manifest_baseline(load_json(run_dir / "manifest.json"))
    validate_summary_metrics(load_json(run_dir / "summary_metrics.json"))
    validate_metrics_trajectory_baseline_policy(run_dir / "metrics_trajectory.csv")
    validate_provenance_json(run_dir / "provenance.json")
