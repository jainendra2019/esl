"""Canonical JSON writers, manuscript bundle layout, and milestone artifact bundles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from esl.experiment_registry import MANIFEST_SCHEMA_VERSION, OUTPUT_CONVENTION_ID

MILESTONE_BUNDLE_RELATIVE_FILES: tuple[str, ...] = (
    "config.json",
    "manifest.json",
    "summary_metrics.json",
    "metrics_trajectory.csv",
)

# Shared by PPO (M2A) and offline clustering (M2B) baseline smoke bundles.
MILESTONE_BUNDLE_OFFLINE_BASELINE_FILES: tuple[str, ...] = (
    "config.json",
    "manifest.json",
    "summary_metrics.json",
    "metrics_trajectory.csv",
    "provenance.json",
)

MILESTONE_BUNDLE_PPO_BASELINE_FILES = MILESTONE_BUNDLE_OFFLINE_BASELINE_FILES


def write_manifest_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def ensure_manuscript_bundle_layout(bundle_root: Path) -> None:
    """
    PRD §14B directory scaffolding for paper-facing outputs (smoke / CI).

    Creates standard subfolders and placeholder docs if missing.
    """
    subdirs = (
        "figures",
        "tables",
        "configs",
        "manifests",
        "metrics",
        "reports",
        "reports/templates",
        "provenance",
    )
    for s in subdirs:
        (bundle_root / s).mkdir(parents=True, exist_ok=True)
    readme = bundle_root / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# Manuscript bundle\n\n"
            "Paper-facing artifacts per PRD §14B. Populate via experiment milestones "
            "and final aggregation scripts.\n\n"
            "## Regenerate\n\n"
            "Run the experiment registry smoke (`tests/test_experiment_registry_milestone1.py`) "
            "and milestone bundle helpers from `esl.experiments.canonical_io`.\n",
            encoding="utf-8",
        )
    report = bundle_root / "EXPERIMENT_REPORT.md"
    if not report.is_file():
        report.write_text(
            "# Experiment report (stub)\n\n"
            "Final results and caveats will be written after all milestones complete.\n",
            encoding="utf-8",
        )


def create_milestone_artifact_bundle(
    *,
    source_run_dir: Path,
    bundle_parent: Path,
    milestone_slug: str,
    experiment_id: str,
    milestone_name: str,
    status: str,
    tests_summary: str,
    paper_faithfulness: str,
    readiness: str,
    open_issues: str,
    relative_files: tuple[str, ...] | None = None,
) -> Path:
    """
    Copy canonical run files and emit a PRD §13A milestone status report.

    ``bundle_parent`` is typically ``manuscript_bundle/reports/``.
    """
    bundle_dir = (bundle_parent / milestone_slug).resolve()
    art = bundle_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    files = relative_files if relative_files is not None else MILESTONE_BUNDLE_RELATIVE_FILES
    for name in files:
        src = source_run_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"milestone bundle missing required file: {src}")
        shutil.copy2(src, art / name)
    tmpl = Path(__file__).resolve().parent / "templates" / "milestone_status_report.md"
    text = tmpl.read_text(encoding="utf-8")
    text = (
        text.replace("{{MILESTONE_NAME}}", milestone_name)
        .replace("{{STATUS}}", status)
        .replace("{{EXPERIMENT_ID}}", experiment_id)
        .replace("{{TESTS_SUMMARY}}", tests_summary)
        .replace("{{ARTIFACTS_DIR}}", str(art))
        .replace("{{OPEN_ISSUES}}", open_issues)
        .replace("{{PAPER_FAITHFULNESS}}", paper_faithfulness)
        .replace("{{READINESS}}", readiness)
        .replace("{{MANIFEST_SCHEMA_VERSION}}", str(MANIFEST_SCHEMA_VERSION))
        .replace("{{OUTPUT_CONVENTION_ID}}", OUTPUT_CONVENTION_ID)
    )
    (bundle_dir / "MILESTONE_STATUS.md").write_text(text, encoding="utf-8")
    return bundle_dir


def manuscript_bundle_smoke_copy(
    *,
    source_run_dir: Path,
    manuscript_bundle_root: Path,
    tag: str,
) -> dict[str, Path]:
    """
    Copy a smoke run's key files into manuscript_bundle/{configs,manifests,metrics}.

    Returns map logical_name -> written path (for tests).
    """
    ensure_manuscript_bundle_layout(manuscript_bundle_root)
    out: dict[str, Path] = {}
    cfg_dst = manuscript_bundle_root / "configs" / f"smoke_{tag}_config.json"
    shutil.copy2(source_run_dir / "config.json", cfg_dst)
    out["config"] = cfg_dst
    man_dst = manuscript_bundle_root / "manifests" / f"smoke_{tag}_manifest.json"
    shutil.copy2(source_run_dir / "manifest.json", man_dst)
    out["manifest"] = man_dst
    met_dst = manuscript_bundle_root / "metrics" / f"smoke_{tag}_summary_metrics.json"
    shutil.copy2(source_run_dir / "summary_metrics.json", met_dst)
    out["summary_metrics"] = met_dst
    traj_dst = manuscript_bundle_root / "metrics" / f"smoke_{tag}_metrics_trajectory.csv"
    shutil.copy2(source_run_dir / "metrics_trajectory.csv", traj_dst)
    out["metrics_trajectory"] = traj_dst
    return out


def manuscript_bundle_baseline_ppo_smoke_copy(
    *,
    source_run_dir: Path,
    manuscript_bundle_root: Path,
    tag: str,
) -> dict[str, Path]:
    """Copy PPO baseline smoke artifacts into manuscript_bundle subfolders."""
    ensure_manuscript_bundle_layout(manuscript_bundle_root)
    out: dict[str, Path] = {}
    mapping = (
        ("config.json", "configs", f"smoke_{tag}_config.json"),
        ("manifest.json", "manifests", f"smoke_{tag}_manifest.json"),
        ("summary_metrics.json", "metrics", f"smoke_{tag}_summary_metrics.json"),
        ("metrics_trajectory.csv", "metrics", f"smoke_{tag}_metrics_trajectory.csv"),
        ("provenance.json", "provenance", f"smoke_{tag}_provenance.json"),
    )
    for rel, sub, fname in mapping:
        dst = manuscript_bundle_root / sub / fname
        shutil.copy2(source_run_dir / rel, dst)
        out[rel] = dst
    return out
