"""Clean 3×3 baseline refresh: merge retrieved Tier-1 cells with newly run Online EM / OMD/MWU."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from esl.res.omd_mwu import OMD_PRIMARY_C, OMD_SENSITIVITY_C
from esl.res.runner import (
    RETRIEVABLE_A3_METHODS,
    RUN_NEW_A3_METHODS,
    aggregate_all_metrics,
    paired_gaps_all_metrics,
    run_res_m1,
    _write_csv,
)


PAIRING_FIELDS = ("instance_id", "seed", "focal_role", "horizon", "p_obs", "obs_noise")


def pairing_key(row: dict[str, Any]) -> tuple:
    return (
        str(row["instance_id"]),
        int(row["seed"]),
        str(row.get("focal_role", "row")),
        int(float(row["horizon"])),
        float(row.get("p_obs", 1.0)),
        float(row.get("obs_noise", 0.0)),
    )


def method_pairing_key(row: dict[str, Any]) -> tuple:
    return pairing_key(row) + (str(row["method"]),)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_retrievable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if str(r["method"]) not in RETRIEVABLE_A3_METHODS:
            continue
        # Clean protocol only
        if abs(float(r.get("p_obs", 1.0)) - 1.0) > 1e-12:
            continue
        if abs(float(r.get("obs_noise", 0.0))) > 1e-12:
            continue
        out.append(dict(r))
    return out


def merge_baseline_cells(
    retrieved: Sequence[dict[str, Any]],
    fresh: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union by (pairing_key, method); fresh rows win on collision."""
    by_key: dict[tuple, dict[str, Any]] = {}
    for r in retrieved:
        by_key[method_pairing_key(r)] = dict(r)
    for r in fresh:
        by_key[method_pairing_key(r)] = dict(r)
    return list(by_key.values())


def assert_complete_pairing(
    rows: Sequence[dict[str, Any]],
    *,
    required_methods: Sequence[str],
) -> None:
    """Every method must share identical pairing keys with ESL."""
    by_method: dict[str, set[tuple]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), set()).add(pairing_key(r))
    if "ESL" not in by_method:
        raise AssertionError("merged cells missing ESL")
    base = by_method["ESL"]
    for m in required_methods:
        if m not in by_method:
            raise AssertionError(f"merged cells missing method {m}")
        if by_method[m] != base:
            missing = base - by_method[m]
            extra = by_method[m] - base
            raise AssertionError(
                f"pairing mismatch for {m}: missing={len(missing)} extra={len(extra)}"
            )


def run_new_methods_suite(
    *,
    out_dir: Path,
    benchmark_root: Path,
    seeds: Sequence[int],
    horizon: int,
    focal_roles: Sequence[str],
    omd_c: float,
    methods: Sequence[str] = RUN_NEW_A3_METHODS,
    prefix: str,
) -> list[dict[str, Any]]:
    gate = run_res_m1(
        out_dir=out_dir,
        benchmark_root=benchmark_root,
        seeds=seeds,
        horizon=horizon,
        methods=methods,
        num_actions=3,
        p_obs=1.0,
        obs_noise=0.0,
        force_categorical=True,
        milestone_override=prefix,
        prefix_override=prefix,
        focal_roles=focal_roles,
        omd_c=float(omd_c),
    )
    cells_path = Path(gate["cells_csv"])
    return _read_csv(cells_path)


def write_refresh_artifacts(
    *,
    out_dir: Path,
    summaries: list[dict[str, Any]],
    prefix: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalize optional columns
    cols = [
        "family",
        "instance_id",
        "num_actions",
        "method",
        "seed",
        "horizon",
        "psr",
        "dpr",
        "lte_auc",
        "early_gap_auc",
        "p_obs",
        "obs_noise",
        "sa_fp_p_stay",
        "focal_role",
        "omd_c",
        "online_em_window",
    ]
    cells_csv = out_dir / f"{prefix}_cells.csv"
    _write_csv(cells_csv, summaries, cols)

    family_rows = aggregate_all_metrics(summaries)
    family_csv = out_dir / f"{prefix}_family_summary.csv"
    _write_csv(
        family_csv,
        family_rows,
        ["family", "method", "metric", "n", "mean", "ci95_low", "ci95_high"],
    )
    paired_rows = paired_gaps_all_metrics(summaries)
    paired_csv = out_dir / f"{prefix}_paired.csv"
    _write_csv(
        paired_csv,
        paired_rows,
        ["method_a", "method_b", "metric", "diff_meaning", "n", "mean", "std", "ci95_low", "ci95_high", "cohen_dz"],
    )
    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "cells_csv": str(cells_csv),
        "family_csv": str(family_csv),
        "paired_csv": str(paired_csv),
        "manifest": str(manifest_path),
        "n_summaries": len(summaries),
        "omd_primary_c": OMD_PRIMARY_C,
        "omd_sensitivity_c": list(OMD_SENSITIVITY_C),
    }
