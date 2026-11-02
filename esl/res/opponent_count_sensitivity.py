"""Opponent-count sensitivity for focal adaptation (clean 3×3 ESL only).

Switch clock audit: type switches are indexed by **global round** t (see
outputs/opponent_count/SWITCH_CLOCK_AUDIT.md). Horizon-only matching (T scaled)
does not match switch exposure; use switch_matched=True for fair J=10 comparison.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl.experiments.figure1_dynamic_adaptation import (
    FIGURE1_MAIN_HORIZON,
    FIGURE1_MAIN_SEEDS,
    H_WINDOW,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
    compute_post_switch_oracle_regret,
)
from esl.res.benchmark import load_suite
from esl.res.categorical_cell import (
    DEFAULT_NUM_OPPONENTS,
    exposure_matched_horizon,
    exposure_matched_switch_hazard,
    exposure_matched_switch_interval,
    opponent_ids,
    run_categorical_cell,
)
from esl.res.game_spec import GameSpec
from esl.res.runner import game_spec_to_task

NUM_ACTIONS = 3
SEEDS: tuple[int, ...] = FIGURE1_MAIN_SEEDS

MatchMode = Literal["flagship", "horizon_only", "switch_matched"]


def _window_rows(rows: list[dict[str, Any]], *, h_window: int = H_WINDOW) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if int(r.get("switch_id", -1)) >= 0 and 0 <= int(r.get("tau", h_window + 1)) <= h_window
    ]


def compute_psr_diagnostics(
    rows: list[dict[str, Any]],
    *,
    num_opponents: int,
    h_window: int = H_WINDOW,
) -> dict[str, float]:
    """Normalized PSR quantities for cross-J comparison."""
    window = _window_rows(rows, h_window=h_window)
    psr = float(sum(float(r["psr_gap"]) for r in window))
    switch_keys = {(int(r["opponent_id"]), int(r["switch_id"])) for r in window}
    n_switches = len(switch_keys)
    n_window_rows = len(window)

    enc = Counter(int(r["opponent_id"]) for r in rows)
    j = int(num_opponents)
    mean_enc = float(np.mean([enc.get(opp, 0) for opp in opponent_ids(j)]))

    # Encounters between successive switches (per opponent, per switch transition)
    gaps: list[float] = []
    by_opp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_opp[int(r["opponent_id"])].append(r)
    for opp, opp_rows in by_opp.items():
        opp_rows.sort(key=lambda x: int(x["round"]))
        last_sid = None
        since = 0
        for r in opp_rows:
            sid = int(r["switch_id"])
            if last_sid is None:
                last_sid = sid
                since = 1
                continue
            if sid != last_sid:
                if since > 0:
                    gaps.append(float(since))
                last_sid = sid
                since = 1
            else:
                since += 1

    return {
        "psr": psr,
        "n_evaluated_switches": float(n_switches),
        "n_psr_window_rows": float(n_window_rows),
        "psr_per_switch": psr / n_switches if n_switches > 0 else float("nan"),
        "psr_per_window_row": psr / n_window_rows if n_window_rows > 0 else float("nan"),
        "mean_encounters_per_opponent": mean_enc,
        "mean_encounters_between_opponent_switches": float(np.mean(gaps)) if gaps else float("nan"),
    }


def protocol_knobs(num_opponents: int, *, mode: MatchMode) -> dict[str, Any]:
    j = int(num_opponents)
    if mode == "flagship":
        return {
            "horizon": int(FIGURE1_MAIN_HORIZON),
            "switch_interval": int(SWITCH_INTERVAL),
            "switch_hazard": float(SWITCH_HAZARD),
            "match_mode": mode,
        }
    if mode == "horizon_only":
        return {
            "horizon": exposure_matched_horizon(j),
            "switch_interval": int(SWITCH_INTERVAL),
            "switch_hazard": float(SWITCH_HAZARD),
            "match_mode": mode,
        }
    # switch_matched: scale T, interval, and hazard
    return {
        "horizon": exposure_matched_horizon(j),
        "switch_interval": exposure_matched_switch_interval(j),
        "switch_hazard": exposure_matched_switch_hazard(j),
        "match_mode": mode,
    }


def run_and_diagnose_cell(
    *,
    spec: GameSpec,
    seed: int,
    num_opponents: int,
    mode: MatchMode,
) -> dict[str, Any]:
    task = game_spec_to_task(spec)
    knobs = protocol_knobs(num_opponents, mode=mode)
    result = run_categorical_cell(
        task=task,
        method="ESL",
        seed=int(seed),
        horizon=int(knobs["horizon"]),
        num_actions=NUM_ACTIONS,
        num_opponents=int(num_opponents),
        switch_interval=int(knobs["switch_interval"]),
        switch_hazard=float(knobs["switch_hazard"]),
        p_obs=1.0,
        obs_noise=0.0,
    )
    diag = compute_psr_diagnostics(result["trajectory"], num_opponents=num_opponents)
    row = {
        **result["summary"],
        **diag,
        **knobs,
        "family": spec.family,
        "instance_id": spec.instance_id,
    }
    return row


def run_opponent_count_grid(
    *,
    num_opponents: int,
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    mode: MatchMode = "switch_matched",
    tag: str | None = None,
) -> Path:
    """Run ESL-only clean 3×3 grid with diagnostics."""
    j = int(num_opponents)
    specs = load_suite(num_actions=NUM_ACTIONS)
    if not specs:
        raise RuntimeError("No 3×3 benchmark games found")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = tag or mode
    csv_path = out_dir / f"J{j}_{suffix}_diagnostics.csv"
    rows_out: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            rows_out.append(
                run_and_diagnose_cell(spec=spec, seed=int(seed), num_opponents=j, mode=mode)
            )

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return csv_path


def _first_spec_per_family(specs: list[GameSpec]) -> list[GameSpec]:
    seen: set[str] = set()
    out: list[GameSpec] = []
    for spec in specs:
        if spec.family not in seen:
            seen.add(spec.family)
            out.append(spec)
    return out


def run_timing_probe(
    *,
    num_opponents: int = 100,
    seed: int = 0,
    mode: MatchMode = "switch_matched",
    out_dir: Path = Path("outputs/opponent_count"),
) -> dict[str, Any]:
    """Full-protocol timing probe: one cell per family (seed 0)."""
    specs = _first_spec_per_family(load_suite(num_actions=NUM_ACTIONS))
    if len(specs) < 3:
        raise RuntimeError("expected at least one game per family in 3×3 suite")

    out_dir.mkdir(parents=True, exist_ok=True)
    knobs = protocol_knobs(num_opponents, mode=mode)
    rows: list[dict[str, Any]] = []
    t0_total = time.perf_counter()

    for spec in specs:
        t0 = time.perf_counter()
        row = run_and_diagnose_cell(
            spec=spec, seed=int(seed), num_opponents=int(num_opponents), mode=mode
        )
        wall = time.perf_counter() - t0
        row["wall_time_sec"] = round(wall, 3)
        rows.append(row)
        print(
            f"[probe] {spec.family}/{spec.instance_id} seed={seed} "
            f"wall={wall:.1f}s psr/switch={row['psr_per_switch']:.3f} "
            f"n_switches={row['n_evaluated_switches']:.0f}",
            flush=True,
        )

    total_wall = time.perf_counter() - t0_total
    walls = [float(r["wall_time_sec"]) for r in rows]
    median_wall = float(np.median(walls))
    extrapolated_120 = median_wall * 120.0

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = ""

    manifest: dict[str, Any] = {
        "probe_type": "timing_3_family",
        "num_opponents": int(num_opponents),
        "match_mode": mode,
        "seed": int(seed),
        "protocol": knobs,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "commit_hash": commit,
        "cells": rows,
        "summary": {
            "n_probe_cells": len(rows),
            "wall_time_sec_per_cell": walls,
            "median_wall_time_sec": round(median_wall, 3),
            "mean_wall_time_sec": round(float(np.mean(walls)), 3),
            "min_wall_time_sec": round(float(np.min(walls)), 3),
            "max_wall_time_sec": round(float(np.max(walls)), 3),
            "total_probe_wall_time_sec": round(total_wall, 3),
            "extrapolated_120_cell_sequential_sec": round(extrapolated_120, 1),
            "extrapolated_120_cell_sequential_hours": round(extrapolated_120 / 3600.0, 2),
            "extrapolated_120_cell_4_workers_hours": round(extrapolated_120 / 3600.0 / 4.0, 2),
        },
        "exposure_targets_j3_reference": {
            "mean_encounters_per_opponent": 333.0,
            "mean_encounters_between_switches": 22.7,
        },
    }
    manifest_path = out_dir / f"J{num_opponents}_timing_probe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    probe_csv = out_dir / f"J{num_opponents}_timing_probe.csv"
    with probe_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(
        f"\n[probe] median={median_wall:.1f}s/cell → "
        f"120 cells sequential ≈ {extrapolated_120/3600:.1f}h "
        f"(4 workers ≈ {extrapolated_120/3600/4:.1f}h)",
        flush=True,
    )
    print(f"[probe] manifest → {manifest_path}", flush=True)
    return manifest


def hier_boot_mean(vals_by_inst: dict[str, np.ndarray], *, B: int = 4000, seed: int = 0) -> dict[str, float]:
    instances = sorted(vals_by_inst)
    if not instances:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        drawn = rng.choice(instances, size=len(instances), replace=True)
        parts = [vals_by_inst[i][rng.integers(0, len(vals_by_inst[i]), size=len(vals_by_inst[i]))] for i in drawn]
        boots.append(float(np.concatenate(parts).mean()))
    boots_arr = np.asarray(boots)
    return {
        "mean": float(np.mean(boots_arr)),
        "ci_lo": float(np.quantile(boots_arr, 0.025)),
        "ci_hi": float(np.quantile(boots_arr, 0.975)),
    }


def summarize_diagnostics(csv_path: Path, *, metric: str = "psr_per_switch") -> dict[str, Any]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    vals = np.array([float(r[metric]) for r in rows if r.get(metric, "") not in ("", "nan")])
    n = len(vals)
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    by_inst: dict[str, np.ndarray] = {}
    for r in rows:
        key = str(r.get("instance_id", r.get("task", "")))
        by_inst.setdefault(key, []).append(float(r[metric]))
    by_inst = {k: np.asarray(v) for k, v in by_inst.items()}
    hb = hier_boot_mean(by_inst)

    by_family: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_family[str(r["family"])].append(float(r[metric]))

    family_stats = {
        fam: {
            "mean": float(np.mean(v)),
            "se": float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0,
            "n": len(v),
        }
        for fam, v in sorted(by_family.items())
    }

    # Aggregate protocol diagnostics
    proto = {
        "horizon_mean": float(np.mean([float(r["horizon"]) for r in rows])),
        "switch_interval": int(float(rows[0]["switch_interval"])),
        "switch_hazard": float(rows[0]["switch_hazard"]),
        "mean_encounters_per_opponent": float(np.mean([float(r["mean_encounters_per_opponent"]) for r in rows])),
        "mean_encounters_between_switches": float(
            np.nanmean([float(r["mean_encounters_between_opponent_switches"]) for r in rows])
        ),
        "n_evaluated_switches_mean": float(np.mean([float(r["n_evaluated_switches"]) for r in rows])),
        "psr_mean": float(np.mean([float(r["psr"]) for r in rows])),
        "psr_per_switch_mean": float(np.mean([float(r["psr_per_switch"]) for r in rows])),
        "psr_per_window_row_mean": float(np.mean([float(r["psr_per_window_row"]) for r in rows])),
    }

    return {
        "metric": metric,
        "n_cells": n,
        "cell_mean": mean,
        "cell_se": se,
        "cell_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "hier_boot": hb,
        "family": family_stats,
        "protocol": proto,
        "match_mode": rows[0].get("match_mode", ""),
        "num_opponents": int(float(rows[0]["num_opponents"])),
    }


def compute_matched_game_seed_contrast(
    baseline_csv: Path,
    variant_csv: Path,
    *,
    contrast: str,
    metric: str = "psr_per_switch",
    B: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Matched game–seed contrast Δ = metric(variant) − metric(baseline)."""
    import csv as csv_mod
    from scipy import stats as sp_stats

    with open(baseline_csv) as f:
        base = {(r["instance_id"], int(r["seed"])): float(r[metric]) for r in csv_mod.DictReader(f)}
    with open(variant_csv) as f:
        var = {(r["instance_id"], int(r["seed"])): float(r[metric]) for r in csv_mod.DictReader(f)}

    keys = sorted(set(base.keys()) & set(var.keys()))
    if not keys:
        raise ValueError("no overlapping (instance_id, seed) keys")

    diffs = np.array([var[k] - base[k] for k in keys])
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = sd_diff / math.sqrt(n) if n > 1 else float("inf")
    dz = mean_diff / sd_diff if sd_diff > 0 else float("inf")

    t_stat, p_val = sp_stats.ttest_1samp(diffs, 0.0) if n > 1 else (float("nan"), float("nan"))
    t_ci = sp_stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se_diff) if n > 1 else (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boot_cell = sorted(float(np.mean(diffs[rng.choice(n, n, True)])) for _ in range(B))

    by_inst: dict[str, list[float]] = defaultdict(list)
    for k in keys:
        by_inst[k[0]].append(float(var[k] - base[k]))
    instances = sorted(by_inst)
    boot_inst = []
    for _ in range(B):
        drawn = rng.choice(instances, size=len(instances), replace=True)
        parts = []
        for inst in drawn:
            vals = np.asarray(by_inst[inst], dtype=np.float64)
            parts.extend(vals[rng.integers(0, len(vals), size=len(vals))].tolist())
        boot_inst.append(float(np.mean(parts)))
    boot_inst.sort()

    return {
        "metric": metric,
        "contrast": contrast,
        "n_matched_cells": n,
        "mean_delta": round(mean_diff, 4),
        "paired_ci95_t": [round(float(t_ci[0]), 4), round(float(t_ci[1]), 4)],
        "bootstrap_ci95_cell": [round(boot_cell[int(0.025 * B)], 4), round(boot_cell[int(0.975 * B)], 4)],
        "hier_boot_ci95_instance": [round(boot_inst[int(0.025 * B)], 4), round(boot_inst[int(0.975 * B)], 4)],
        "t_stat": round(float(t_stat), 4) if n > 1 else None,
        "p_value": float(p_val) if n > 1 else None,
        "cohens_dz": round(dz, 4),
        "ci_includes_zero": bool(t_ci[0] <= 0.0 <= t_ci[1]) if n > 1 else None,
    }


def compute_instance_seed_contrast(
    j3_csv: Path,
    j10_csv: Path,
    *,
    metric: str = "psr_per_switch",
    B: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Backward-compatible wrapper for J10 − J3 contrast."""
    out = compute_matched_game_seed_contrast(
        j3_csv, j10_csv, contrast="J10_switch_matched minus J3_flagship", metric=metric, B=B, seed=seed
    )
    out["n_paired_cells"] = out["n_matched_cells"]
    return out


def plot_opponent_count_ladder(
    summaries: list[dict[str, Any]],
    *,
    out_path: Path,
    metric: str = "psr_per_switch",
    ylabel: str = "Mean PSR per evaluated switch ↓",
) -> None:
    """Point-and-interval sensitivity ladder over J (no connecting line)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = sorted(summaries, key=lambda s: int(s["num_opponents"]))
    xs = [int(s["num_opponents"]) for s in summaries]
    means = [s["hier_boot"]["mean"] for s in summaries]
    ci_lo = [s["hier_boot"]["mean"] - s["hier_boot"]["ci_lo"] for s in summaries]
    ci_hi = [s["hier_boot"]["ci_hi"] - s["hier_boot"]["mean"] for s in summaries]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    # Unconnected points: ordered sensitivity ladder (no implied interpolation).
    ax.errorbar(
        xs, means, yerr=[ci_lo, ci_hi], fmt="o", capsize=5, color="#2c7bb6",
        markersize=8, markeredgecolor="k", markeredgewidth=0.5, linewidth=1.2, linestyle="none",
    )
    ax.set_xticks(xs)
    ax.set_xlabel("Number of opponents J")
    ax.set_ylabel(ylabel)
    ax.set_title("ESL opponent-count sensitivity (clean 3×3)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_comparison(
    summaries: list[dict[str, Any]],
    *,
    out_path: Path,
    metric: str = "psr_per_switch",
    ylabel: str = "Mean PSR per evaluated switch ↓",
    contrast: dict[str, Any] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_map = {
        (3, "flagship"): "J=3\n(flagship)",
        (10, "switch_matched"): "J=10\n(matched exposure)",
    }
    labels = []
    means = []
    ci_lo = []
    ci_hi = []
    for s in summaries:
        key = (s["num_opponents"], s["match_mode"])
        labels.append(label_map.get(key, f"J={s['num_opponents']}"))
        hb = s["hier_boot"]
        means.append(hb["mean"])
        ci_lo.append(hb["mean"] - hb["ci_lo"])
        ci_hi.append(hb["ci_hi"] - hb["mean"])

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=[ci_lo, ci_hi], capsize=5,
        color=["#2c7bb6", "#abd9e9"][: len(labels)], edgecolor="k", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title("ESL opponent-count sensitivity (clean 3×3)")
    if contrast is not None:
        d = contrast["mean_delta"]
        ci = contrast["hier_boot_ci95_instance"]
        ax.text(
            0.5, 0.02,
            f"Δ(J10−J3) = {d:+.2f}; instance-clustered 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]",
            transform=ax.transAxes, ha="center", fontsize=8,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_full_analysis(*, out_dir: Path = Path("outputs/opponent_count")) -> dict[str, Any]:
    """Phase 1 correction: diagnostics + switch-matched J=10."""
    out_dir.mkdir(parents=True, exist_ok=True)

    j3_path = run_opponent_count_grid(num_opponents=3, out_dir=out_dir, mode="flagship")
    j10_h_path = run_opponent_count_grid(num_opponents=10, out_dir=out_dir, mode="horizon_only")
    j10_s_path = run_opponent_count_grid(num_opponents=10, out_dir=out_dir, mode="switch_matched")

    metric = "psr_per_switch"
    s3 = summarize_diagnostics(j3_path, metric=metric)
    s10h = summarize_diagnostics(j10_h_path, metric=metric)
    s10s = summarize_diagnostics(j10_s_path, metric=metric)

    contrast = compute_instance_seed_contrast(j3_path, j10_s_path, metric=metric)

    plot_normalized_comparison(
        [s3, s10s],
        out_path=out_dir / "opponent_count_psr_per_switch.pdf",
        metric=metric,
        ylabel="Mean PSR per evaluated switch ↓",
        contrast=contrast,
    )
    plot_normalized_comparison(
        [s3, s10s],
        out_path=Path("manuscript_bundle/main/figures/opponent_count_psr_per_switch.pdf"),
        metric=metric,
        ylabel="Mean PSR per evaluated switch ↓",
        contrast=contrast,
    )

    report = {
        "switch_clock": "global_round",
        "primary_metric": metric,
        "J3_flagship": s3,
        "J10_horizon_only_internal": s10h,
        "J10_switch_matched": s10s,
        "direct_contrast_J10_minus_J3": contrast,
        "figure_caption": (
            "ESL opponent-count sensitivity on the clean 3×3 suite. "
            "Bars show hierarchical bootstrap 95% CIs over game instances (not standard errors). "
            "J=3 uses the flagship protocol (T=1000, switch interval=200, hazard=0.01). "
            "J=10 uses switch-matched exposure (T=3333, interval=667, hazard=0.003) so that "
            "encounters per opponent and encounters between behavioral switches match J=3."
        ),
        "paths": {
            "J3": str(j3_path),
            "J10_horizon_only": str(j10_h_path),
            "J10_switch_matched": str(j10_s_path),
        },
    }
    (out_dir / "normalized_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    # LaTeX snippet for contrast table
    c = contrast
    tex = (
        f"\\begin{{table}}[h]\\centering\\small\n"
        f"\\caption{{Direct contrast: PSR per evaluated switch (J=10 switch-matched minus J=3 flagship; "
        f"$n={c['n_paired_cells']}$ cells).}}\n"
        f"\\begin{{tabular}}{{lr}}\\toprule\n"
        f"Mean $\\Delta$ & {c['mean_delta']:+.3f} \\\\\n"
        f"Paired 95\\% CI & [{c['paired_ci95_t'][0]:+.3f}, {c['paired_ci95_t'][1]:+.3f}] \\\\\n"
        f"Instance-clustered boot. 95\\% CI & [{c['hier_boot_ci95_instance'][0]:+.3f}, {c['hier_boot_ci95_instance'][1]:+.3f}] \\\\\n"
        f"Cohen's $d_z$ & {c['cohens_dz']:.3f} \\\\\n"
        f"$p$-value & {c['p_value']:.4f} \\\\\n"
        f"\\bottomrule\\end{{tabular}}\\end{{table}}\n"
    )
    (out_dir / "paired_contrast.tex").write_text(tex)
    return report


def summarize_ladder(*, out_dir: Path = Path("outputs/opponent_count")) -> dict[str, Any]:
    """Summarize J=3 / J=10 / J=100 from existing CSVs and write ladder figure."""
    j3_path = out_dir / "J3_flagship_diagnostics.csv"
    j10_s_path = out_dir / "J10_switch_matched_diagnostics.csv"
    j100_s_path = out_dir / "J100_switch_matched_diagnostics.csv"
    for p in (j3_path, j10_s_path, j100_s_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing diagnostics CSV: {p}")

    metric = "psr_per_switch"
    s3 = summarize_diagnostics(j3_path, metric=metric)
    s10s = summarize_diagnostics(j10_s_path, metric=metric)
    s100s = summarize_diagnostics(j100_s_path, metric=metric)

    c10 = compute_matched_game_seed_contrast(
        j3_path, j10_s_path, contrast="J10_switch_matched minus J3_flagship", metric=metric
    )
    c100 = compute_matched_game_seed_contrast(
        j3_path, j100_s_path, contrast="J100_switch_matched minus J3_flagship", metric=metric
    )
    c100_10 = compute_matched_game_seed_contrast(
        j10_s_path, j100_s_path, contrast="J100_switch_matched minus J10_switch_matched", metric=metric
    )

    ladder = [s3, s10s, s100s]
    for out_path in (
        out_dir / "opponent_count_psr_per_switch.pdf",
        Path("manuscript_bundle/main/figures/opponent_count_psr_per_switch.pdf"),
    ):
        plot_opponent_count_ladder(ladder, out_path=out_path, metric=metric)

    caption = (
        "ESL opponent-count sensitivity on the clean 3×3 suite. "
        "Points show hierarchical bootstrap 95% CIs over game instances (not standard errors). "
        "J=3 uses the flagship protocol (T=1000, switch interval=200, hazard=0.01). "
        "J=10 and J=100 use switch-matched exposure: T, switch interval, and hazard were scaled "
        "so encounters per opponent and encounters between behavioral switches match J=3."
    )
    report = {
        "switch_clock": "global_round",
        "primary_metric": metric,
        "J3_flagship": s3,
        "J10_switch_matched": s10s,
        "J100_switch_matched": s100s,
        "matched_contrast_J10_minus_J3": c10,
        "matched_contrast_J100_minus_J3": c100,
        "matched_contrast_J100_minus_J10": c100_10,
        "figure_caption": caption,
        "paths": {
            "J3": str(j3_path),
            "J10_switch_matched": str(j10_s_path),
            "J100_switch_matched": str(j100_s_path),
        },
    }
    (out_dir / "normalized_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    tex_lines = [
        "\\begin{table}[h]\\centering\\small",
        "\\caption{Matched game--seed contrasts: PSR per evaluated switch ($n=120$ cells per contrast). "
        "Instance-clustered bootstrap 95\\% CIs are primary.}",
        "\\begin{tabular}{lrrr}\\toprule",
        "Contrast & Mean $\\Delta$ & Inst.-clustered 95\\% CI & CI incl.\\ 0? \\\\",
        "\\midrule",
    ]
    for c in (c10, c100, c100_10):
        ci = c["hier_boot_ci95_instance"]
        incl = "Yes" if c["ci_includes_zero"] else "No"
        tex_lines.append(
            f"{c['contrast'].replace('_', ' ')} & {c['mean_delta']:+.3f} & "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] & {incl} \\\\"
        )
    tex_lines.extend(["\\bottomrule\\end{tabular}\\end{table}"])
    (out_dir / "paired_contrast.tex").write_text("\n".join(tex_lines) + "\n")
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Opponent-count sensitivity (ESL, clean 3×3)")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run one grid")
    p_run.add_argument("--J", type=int, default=10)
    p_run.add_argument("--mode", choices=["flagship", "horizon_only", "switch_matched"], default="switch_matched")
    p_run.add_argument("--out-dir", type=Path, default=Path("outputs/opponent_count"))

    sub.add_parser("analyze", help="Run J=3 + J=10 horizon + J=10 switch-matched analysis")
    sub.add_parser("summarize", help="Recompute stats/figure from existing diagnostic CSVs")
    p_probe = sub.add_parser("probe", help="3-family full-protocol timing probe (one cell per family)")
    p_probe.add_argument("--J", type=int, default=100)
    p_probe.add_argument("--seed", type=int, default=0)
    p_probe.add_argument("--mode", choices=["flagship", "horizon_only", "switch_matched"], default="switch_matched")
    p_probe.add_argument("--out-dir", type=Path, default=Path("outputs/opponent_count"))

    args = p.parse_args()
    if args.cmd == "run":
        path = run_opponent_count_grid(num_opponents=args.J, out_dir=args.out_dir, mode=args.mode)
        print(f"Grid → {path}")
    elif args.cmd == "analyze":
        report = run_full_analysis()
        print(json.dumps(report, indent=2))
    elif args.cmd == "summarize":
        report = summarize_ladder()
        print(json.dumps(
            {
                "J3_psr_per_switch": report["J3_flagship"]["hier_boot"],
                "J10_psr_per_switch": report["J10_switch_matched"]["hier_boot"],
                "J100_psr_per_switch": report["J100_switch_matched"]["hier_boot"],
                "J100_minus_J3": report["matched_contrast_J100_minus_J3"],
            },
            indent=2,
        ))
    elif args.cmd == "probe":
        manifest = run_timing_probe(
            num_opponents=int(args.J),
            seed=int(args.seed),
            mode=args.mode,
            out_dir=args.out_dir,
        )
        print(json.dumps(manifest["summary"], indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
