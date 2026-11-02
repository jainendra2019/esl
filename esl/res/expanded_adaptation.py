"""Expanded 3×3 Adaptation Benchmark — flagship registry, protocol locks, packaging helpers.

Canonical empirical evaluation for the revised manuscript (replaces 2×2 Figure 1 as primary).
OMD/MWU is excluded from this workstream. PPO is pending until after the clean flagship freeze.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from esl.experiments.figure1_dynamic_adaptation import (
    FOCAL_AGENT_ID,
    H_WINDOW,
    K_TYPES,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
)
from esl.res.runner import paired_method_gaps

# Methods that already run on A=3 categorical cells — used for M2 Figure-1 replacement freeze.
FLAGSHIP_A3_METHODS_EXISTING: tuple[str, ...] = (
    "ESL",
    "Fixed-K Bayesian",
    "SA-FP",
    "FP",
    "SOM",
    "Online EM",
)

# Full intended set after M3 PPO port (PPO may be omit-gated).
FLAGSHIP_A3_METHODS_FULL: tuple[str, ...] = (
    "ESL",
    "PPO",
    "Fixed-K Bayesian",
    "SA-FP",
    "FP",
    "SOM",
    "Online EM",
)

EXCLUDED_FROM_FLAGSHIP: tuple[str, ...] = ("OMD/MWU",)

LATENT_LTE_METHODS: tuple[str, ...] = (
    "ESL",
    "Fixed-K Bayesian",
    "SA-FP",
    "Online EM",
)

PREDICTIVE_DPR_METHODS: tuple[str, ...] = FLAGSHIP_A3_METHODS_EXISTING  # all existing emit DPR

PROTOCOL = {
    "focal_agent_id": FOCAL_AGENT_ID,
    "num_opponents": 3,
    "opponent_ids": (1, 2, 3),
    "k_types": K_TYPES,
    "num_actions_canonical": 3,
    "horizon": 1000,
    "h_window": H_WINDOW,
    "switch_interval": SWITCH_INTERVAL,
    "switch_hazard": SWITCH_HAZARD,
    "primary_metric": "psr",
    "secondary_metrics": ("dpr", "lte_auc", "early_gap_auc"),
}

ALLOWED_CLAIMS: tuple[str, ...] = (
    "ESL_lowest_PSR_among_flagship_when_supported",
    "SA_FP_strongest_switch_aware_baseline",
    "Online_EM_strongest_LTE_when_supported",
    "DPR_reported_honestly",
    "LTE_reported_honestly_not_equated_to_adaptation",
    "2x2_vs_3x3_separated",
    "sym_vs_asym_separated",
    "clean_vs_visibility_vs_noise_separated",
    "OMD_not_in_this_workstream",
    "opp6_not_primary_if_run",
)

DEFAULT_SOURCE_SYM = Path("outputs/res/eval/baseline_a3_refresh/symmetric/baseline_sym_cells.csv")
DEFAULT_OUT_SYM = Path("outputs/res/eval/expanded_adaptation/symmetric")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cells_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_flagship_cells(
    rows: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str] = FLAGSHIP_A3_METHODS_EXISTING,
    num_actions: int = 3,
    p_obs: float = 1.0,
    obs_noise: float = 0.0,
) -> list[dict[str, Any]]:
    """Keep clean flagship cells; drop OMD and non-matching obs settings."""
    method_set = set(methods)
    out: list[dict[str, Any]] = []
    for r in rows:
        m = str(r.get("method", ""))
        if m in EXCLUDED_FROM_FLAGSHIP or m not in method_set:
            continue
        if int(float(r.get("num_actions", num_actions))) != int(num_actions):
            continue
        if abs(float(r.get("p_obs", 1.0)) - float(p_obs)) > 1e-12:
            continue
        if abs(float(r.get("obs_noise", 0.0)) - float(obs_noise)) > 1e-12:
            continue
        out.append(dict(r))
    return out


def assert_grid_complete(
    rows: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str],
    n_instances: int = 12,
    n_seeds: int = 10,
) -> dict[str, Any]:
    """Integrity check: each method has n_instances × n_seeds unique cells."""
    by_method: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for r in rows:
        by_method[str(r["method"])].add((str(r["instance_id"]), int(r["seed"])))
    expected = int(n_instances) * int(n_seeds)
    missing: dict[str, int] = {}
    for m in methods:
        got = len(by_method.get(m, set()))
        if got != expected:
            missing[m] = got
    families: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        families[str(r["family"])].add(str(r["instance_id"]))
    return {
        "expected_per_method": expected,
        "methods_ok": len(missing) == 0,
        "missing_or_short": missing,
        "n_families": {k: len(v) for k, v in sorted(families.items())},
        "n_instances": len({str(r["instance_id"]) for r in rows}),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        cols = list(columns or [])
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        return
    cols = list(columns) if columns is not None else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def aggregate_method_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overall mean±CI95 per method × metric (cell-level)."""
    from esl.experiments.stats_ci import mean_std_ci95

    metrics = ("psr", "dpr", "lte_auc", "early_gap_auc")
    out: list[dict[str, Any]] = []
    methods = sorted({str(r["method"]) for r in rows})
    for method in methods:
        subset = [r for r in rows if str(r["method"]) == method]
        for metric in metrics:
            vals: list[float] = []
            for r in subset:
                raw = r.get(metric, "")
                if raw in ("", None, "N/A"):
                    continue
                vals.append(float(raw))
            if not vals:
                out.append(
                    {
                        "method": method,
                        "metric": metric,
                        "metric_defined": False,
                        "n": 0,
                        "mean": "",
                        "ci95_low": "",
                        "ci95_high": "",
                    }
                )
                continue
            st = mean_std_ci95(vals)
            out.append(
                {
                    "method": method,
                    "metric": metric,
                    "metric_defined": True,
                    "n": int(st["n"]),
                    "mean": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                }
            )
    return out


def instance_summary(rows: Sequence[dict[str, Any]], *, metric: str = "psr") -> list[dict[str, Any]]:
    from esl.experiments.stats_ci import mean_std_ci95

    by: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        raw = r.get(metric, "")
        if raw in ("", None, "N/A"):
            continue
        key = (str(r["family"]), str(r["instance_id"]), str(r["method"]))
        by[key].append(float(raw))
    out: list[dict[str, Any]] = []
    for (family, instance_id, method), vals in sorted(by.items()):
        st = mean_std_ci95(vals)
        out.append(
            {
                "family": family,
                "instance_id": instance_id,
                "method": method,
                "metric": metric,
                "n": int(st["n"]),
                "mean": st["mean"],
                "ci95_low": st["ci95_low"],
                "ci95_high": st["ci95_high"],
            }
        )
    return out


def paired_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    contrasts: Iterable[tuple[str, str]] = (
        ("SA-FP", "ESL"),
        ("Fixed-K Bayesian", "ESL"),
        ("Online EM", "ESL"),
        ("FP", "ESL"),
        ("SOM", "ESL"),
    ),
    metrics: Sequence[str] = ("psr", "dpr", "lte_auc", "early_gap_auc"),
) -> list[dict[str, Any]]:
    summaries = [dict(r) for r in rows]
    out: list[dict[str, Any]] = []
    for a, b in contrasts:
        for metric in metrics:
            gap = paired_method_gaps(summaries, method_a=a, method_b=b, metric=metric)
            out.append(gap)
    return out


def hier_boot_paired_diff(
    rows: Sequence[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    metric: str = "psr",
    B: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Hierarchical bootstrap of mean paired (a-b) difference: resample instances, then seeds."""
    by_inst: dict[str, list[float]] = defaultdict(list)
    by_key: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        raw = r.get(metric, "")
        if raw in ("", None, "N/A"):
            continue
        key = (str(r["instance_id"]), int(r["seed"]))
        by_key[key][str(r["method"])] = float(raw)
    for (inst, seed_i), methods in by_key.items():
        if method_a in methods and method_b in methods:
            by_inst[inst].append(methods[method_a] - methods[method_b])
    instances = sorted(by_inst)
    if not instances:
        return {"n_instances": 0.0, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    cell_diffs = np.concatenate([np.asarray(by_inst[i], dtype=np.float64) for i in instances])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        drawn = rng.choice(instances, size=len(instances), replace=True)
        parts = []
        for inst in drawn:
            v = np.asarray(by_inst[inst], dtype=np.float64)
            parts.append(v[rng.integers(0, len(v), size=len(v))])
        boots.append(float(np.concatenate(parts).mean()))
    boots_arr = np.asarray(boots, dtype=np.float64)
    return {
        "n_instances": float(len(instances)),
        "n_pairs": float(cell_diffs.size),
        "mean": float(cell_diffs.mean()),
        "ci_lo": float(np.quantile(boots_arr, 0.025)),
        "ci_hi": float(np.quantile(boots_arr, 0.975)),
    }


def hierarchical_bootstrap_table(
    rows: Sequence[dict[str, Any]],
    *,
    baselines: Sequence[str] = ("SA-FP", "Fixed-K Bayesian", "Online EM", "FP", "SOM"),
    metric: str = "psr",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in baselines:
        hb = hier_boot_paired_diff(rows, method_a=b, method_b="ESL", metric=metric, B=4000, seed=11)
        out.append(
            {
                "method_a": b,
                "method_b": "ESL",
                "metric": metric,
                "diff_meaning": f"{b}_minus_ESL",
                **hb,
            }
        )
    return out


def package_gate_payload(
    *,
    integrity: dict[str, Any],
    source_hash: str,
    source_path: str,
    n_cells: int,
    methods: Sequence[str],
) -> dict[str, Any]:
    return {
        "label": "expanded_adaptation_symmetric_m2",
        "protocol": PROTOCOL,
        "methods": list(methods),
        "excluded": list(EXCLUDED_FROM_FLAGSHIP),
        "source_csv": source_path,
        "source_sha256": source_hash,
        "n_cells": n_cells,
        "integrity": integrity,
        "figure1_replacement_gate": bool(integrity.get("methods_ok")),
        "freeze_note": (
            "Do not modify benchmark suite, hyperparameters, switching process, "
            "or baseline implementations after this gate without invalidating/restarting M2."
        ),
    }
