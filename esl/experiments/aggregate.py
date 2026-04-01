"""Scan run directories and emit a single NeurIPS-style summary CSV (schema v1)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from esl.experiments.manifest import read_run_manifest

AGGREGATE_COLUMNS_V1: list[str] = [
    "run_id",
    "preset",
    "variant",
    "seed",
    "p_obs",
    "prototype_lr_scale",
    "init_noise",
    "prototype_update_every_q",
    "target_interaction_budget",
    "num_interaction_events_executed",
    "num_rounds_executed",
    "final_matched_cross_entropy",
    "final_mce",
    "final_belief_entropy",
    "final_belief_argmax_accuracy",
    "final_prototype_gap",
    "prototype_update_count",
]


def find_run_directories(root: Path) -> list[Path]:
    """Directories under root that contain summary_metrics.json."""
    root = root.resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("summary_metrics.json"):
        if "_aggregates" in p.parts:
            continue
        out.append(p.parent.resolve())
    # de-duplicate (rglob shouldn't duplicate)
    return sorted(set(out))


def row_from_run_dir(run_dir: Path, *, root: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    sm_path = run_dir / "summary_metrics.json"
    summary = json.loads(sm_path.read_text(encoding="utf-8"))
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    man = read_run_manifest(run_dir / "run_manifest.json") or {}

    if root is not None:
        try:
            run_id = str(run_dir.relative_to(root.resolve()))
        except ValueError:
            run_id = str(run_dir)
    else:
        run_id = str(run_dir)

    q = summary.get("prototype_update_every_q", cfg.get("prototype_update_every"))
    tb = man.get("target_interaction_budget", "")
    if tb is None:
        tb = ""

    return {
        "run_id": run_id,
        "preset": man.get("preset", ""),
        "variant": man.get("variant", ""),
        "seed": summary.get("seed", cfg.get("seed", "")),
        "p_obs": summary.get("p_obs", cfg.get("p_obs", "")),
        "prototype_lr_scale": summary.get("prototype_lr_scale", cfg.get("prototype_lr_scale", "")),
        "init_noise": summary.get("init_noise", cfg.get("init_noise", "")),
        "prototype_update_every_q": q,
        "target_interaction_budget": tb,
        "num_interaction_events_executed": summary.get("num_interaction_events_executed", ""),
        "num_rounds_executed": summary.get("num_rounds_executed", ""),
        "final_matched_cross_entropy": summary.get("final_matched_cross_entropy", ""),
        "final_mce": summary.get("final_mce", ""),
        "final_belief_entropy": summary.get("final_belief_entropy", ""),
        "final_belief_argmax_accuracy": summary.get("final_belief_argmax_accuracy", ""),
        "final_prototype_gap": summary.get("final_prototype_gap", ""),
        "prototype_update_count": summary.get("prototype_update_count", ""),
    }


def write_aggregate_csv(
    root: Path,
    output: Path,
    *,
    columns: list[str] | None = None,
) -> Path:
    cols = columns or AGGREGATE_COLUMNS_V1
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rd in find_run_directories(root):
        if rd.resolve() == output.parent.resolve() or "_aggregates" in rd.parts:
            continue
        rows.append(row_from_run_dir(rd, root=root))
    rows.sort(key=lambda r: r["run_id"])
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return output
