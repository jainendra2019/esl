#!/usr/bin/env python3
"""Regenerate published numerical summaries from the frozen evidence package.

Does not rerun simulations. Independently recomputes the flagship means, the
K-sensitivity paired contrasts, and the held-out OMD paired Δ from cell CSVs.
Other files under evidence/ that are copied through are labeled precomputed.

Inputs:  evidence/**/*.csv and evidence/**/*.json
Outputs: outputs/tables/*.csv and a markdown summary on stdout
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esl.res.stats import (
    bootstrap_mean_ci95,
    filter_seeds,
    holm_adjust,
    paired_mean_ci95,
    paired_psr_diffs,
)
from scipy import stats

EVIDENCE = ROOT / "evidence"
OUT = ROOT / "outputs" / "tables"
FOUR_METHODS = ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM")
HELD_OUT_SEEDS = (3, 4, 5, 6, 7, 8, 9)
K_GRID = (2, 3, 5)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals))


def se(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def flagship_tables() -> dict[str, Any]:
    rows = load_csv(EVIDENCE / "flagship" / "cells.csv")
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    summary = []
    for method, cells in sorted(by_method.items()):
        rec = {"method": method, "n": len(cells)}
        for metric in ("psr", "dpr", "lte_auc"):
            vals = [float(c[metric]) for c in cells if c.get(metric, "") not in ("", "N/A")]
            rec[f"{metric}_mean"] = _mean(vals) if vals else ""
            rec[f"{metric}_se"] = se(vals) if vals else ""
            rec[f"{metric}_n"] = len(vals)
        summary.append(rec)

    esl = by_method["ESL"]
    esl_psr = _mean([float(r["psr"]) for r in esl])
    esl_dpr = _mean([float(r["dpr"]) for r in esl])
    esl_lte = _mean([float(r["lte_auc"]) for r in esl])

    per_game = []
    games = sorted({r["instance_id"] for r in rows})
    cell_means: dict[tuple[str, str], float] = {}
    for g in games:
        rec: dict[str, Any] = {"instance_id": g}
        for m in FOUR_METHODS:
            vals = [float(r["psr"]) for r in rows if r["instance_id"] == g and r["method"] == m]
            rec[m] = _mean(vals)
            cell_means[(g, m)] = rec[m]
        per_game.append(rec)
    lowest_four = sum(1 for g in games if cell_means[(g, "ESL")] == min(cell_means[(g, m)] for m in FOUR_METHODS))
    lt_safp = sum(1 for g in games if cell_means[(g, "ESL")] < cell_means[(g, "SA-FP")])

    write_csv(
        OUT / "flagship_method_summary.csv",
        summary,
        ["method", "n", "psr_mean", "psr_se", "dpr_mean", "dpr_se", "lte_auc_mean", "lte_auc_se", "psr_n", "dpr_n", "lte_auc_n"],
    )
    write_csv(
        OUT / "flagship_per_game_psr.csv",
        per_game,
        ["instance_id", *FOUR_METHODS],
    )
    return {
        "esl_psr": esl_psr,
        "esl_dpr": esl_dpr,
        "esl_lte": esl_lte,
        "n": len(esl),
        "lowest_four": lowest_four,
        "lt_safp": lt_safp,
    }


def k_table() -> dict[str, Any]:
    cells = load_csv(EVIDENCE / "k_sensitivity" / "k_sweep_cells.csv")
    by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in cells:
        by_k[int(float(r["k_model"]))].append(r)
    summary = []
    for k in K_GRID:
        rec: dict[str, Any] = {"k_model": k, "n": len(by_k[k])}
        for metric in ("psr", "dpr"):
            vals = [float(r[metric]) for r in by_k[k]]
            rec[f"{metric}_mean"] = _mean(vals)
            rec[f"{metric}_se"] = se(vals)
        summary.append(rec)
    write_csv(
        OUT / "k_sensitivity_summary.csv",
        summary,
        ["k_model", "n", "psr_mean", "psr_se", "dpr_mean", "dpr_se"],
    )

    paired_rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for k_alt in (2, 5):
        for metric in ("psr", "dpr"):
            map3 = {(r["instance_id"], int(r["seed"])): float(r[metric]) for r in by_k[3]}
            mapk = {(r["instance_id"], int(r["seed"])): float(r[metric]) for r in by_k[k_alt]}
            keys = sorted(set(map3) & set(mapk))
            diffs = [mapk[key] - map3[key] for key in keys]
            st = paired_mean_ci95(diffs)
            tstat, pval = stats.ttest_rel(
                [mapk[key] for key in keys], [map3[key] for key in keys]
            )
            rec = {
                "contrast": f"K{k_alt}-K3",
                "metric": metric,
                "n": int(st["n"]),
                "mean_diff": st["mean"],
                "std_diff": st["std"],
                "ci95_low": st["ci95_low"],
                "ci95_high": st["ci95_high"],
                "cohen_dz": st["cohen_dz"],
                "ttest_rel_stat": float(tstat),
                "p_unadj": float(pval),
                "ci_includes_0": bool(st["ci95_low"] <= 0.0 <= st["ci95_high"]),
            }
            paired_rows.append(rec)
            tests.append(rec)
    for rec, adj in zip(tests, holm_adjust([float(r["p_unadj"]) for r in tests])):
        rec["p_holm"] = adj
        rec["holm_sig_0.05"] = bool(adj < 0.05)
    write_csv(
        OUT / "k_sensitivity_paired_recomputed.csv",
        paired_rows,
        [
            "contrast",
            "metric",
            "n",
            "mean_diff",
            "std_diff",
            "ci95_low",
            "ci95_high",
            "cohen_dz",
            "ttest_rel_stat",
            "p_unadj",
            "p_holm",
            "holm_sig_0.05",
            "ci_includes_0",
        ],
    )
    return {"summary": summary, "paired": paired_rows}


def omd_table() -> dict[str, Any]:
    esl_rows = filter_seeds(load_csv(EVIDENCE / "omd" / "esl_lambda_star_cells.csv"), HELD_OUT_SEEDS)
    omd_rows = filter_seeds(load_csv(EVIDENCE / "omd" / "omd_per_opponent_cells.csv"), HELD_OUT_SEEDS)
    esl_map = {(str(r["instance_id"]), int(r["seed"])): float(r["psr"]) for r in esl_rows}
    esl_vals = [float(r["psr"]) for r in esl_rows]
    omd_vals = [float(r["psr"]) for r in omd_rows]
    diffs = paired_psr_diffs(omd_rows, esl_map)
    boot = bootstrap_mean_ci95(diffs, seed=42, n_boot=10_000)
    sd = math.sqrt(sum((x - boot["mean"]) ** 2 for x in diffs) / (len(diffs) - 1))
    recomputed = {
        "n": int(boot["n"]),
        "esl_star_psr_mean": _mean(esl_vals),
        "omd_psr_mean": _mean(omd_vals),
        "mean_delta": boot["mean"],
        "ci95_lo": boot["ci95_low"],
        "ci95_hi": boot["ci95_high"],
        "se": sd / math.sqrt(len(diffs)),
        "dz": boot["mean"] / sd if sd > 0 else float("inf"),
        "note": "recomputed from cell CSVs; percentile bootstrap seed=42, B=10000; Δ = OMD − ESL λ*",
    }
    (OUT / "omd_vs_esl_paired_recomputed.json").write_text(
        json.dumps(recomputed, indent=2) + "\n", encoding="utf-8"
    )
    precomputed = json.loads((EVIDENCE / "omd" / "omd_vs_esl_paired.json").read_text(encoding="utf-8"))
    (OUT / "omd_vs_esl_paired_precomputed.json").write_text(
        json.dumps(precomputed, indent=2) + "\n", encoding="utf-8"
    )
    return {"recomputed": recomputed, "precomputed": precomputed}


def copy_simple(src: Path, dst_name: str, *, precomputed: bool = True) -> None:
    if not src.is_file():
        return
    rows = load_csv(src)
    if not rows:
        return
    write_csv(OUT / dst_name, rows, list(rows[0].keys()))
    if precomputed:
        print(f"(precomputed copy) {src.relative_to(ROOT)} → {dst_name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flag = flagship_tables()
    k = k_table()
    omd = omd_table()
    copy_simple(EVIDENCE / "k_sensitivity" / "paired_vs_k3.csv", "k_sensitivity_paired_precomputed.csv")
    copy_simple(EVIDENCE / "ablation" / "ablation_summary.csv", "ablation_summary.csv")
    copy_simple(EVIDENCE / "lambda" / "summary.csv", "lambda_summary.csv")
    copy_simple(EVIDENCE / "alpha" / "alpha_summary.csv", "alpha_summary.csv")
    copy_simple(EVIDENCE / "batch" / "batch_summary.csv", "batch_summary.csv")
    copy_simple(EVIDENCE / "robustness" / "visibility_family_summary.csv", "visibility_family_summary.csv")
    copy_simple(EVIDENCE / "robustness" / "noise_family_summary.csv", "noise_family_summary.csv")
    copy_simple(EVIDENCE / "controlled_shift" / "shift_ladder_summary.csv", "shift_ladder_summary.csv")
    copy_simple(EVIDENCE / "psrv" / "n20_cells.csv", "psrv_n20_cells.csv")
    copy_simple(EVIDENCE / "psrv" / "n100_cells.csv", "psrv_n100_cells.csv")
    n20 = load_csv(EVIDENCE / "psrv" / "n20_cells.csv")
    n100 = load_csv(EVIDENCE / "psrv" / "n100_cells.csv")
    acc20 = _mean([float(r["recovery_accuracy"]) for r in n20])
    acc100 = _mean([float(r["recovery_accuracy"]) for r in n100])

    print("# Camera-ready numerical summaries")
    print()
    print("## Flagship twelve-game ESL (n=120 cells, λ=2)")
    print(f"- PSR = {flag['esl_psr']:.2f}  (accepted 162.97)")
    print(f"- DPR = {flag['esl_dpr']:.2f}  (accepted 106.67)")
    print(f"- LTE = {flag['esl_lte']:.6f}  (accepted 3.225421; display 3.23)")
    print(f"- ESL lowest PSR among {{ESL, Fixed-K, SA-FP, Online EM}}: {flag['lowest_four']}/12")
    print(f"- ESL PSR < SA-FP PSR: {flag['lt_safp']}/12")
    print()
    print("## K-sensitivity (ESL only; recomputed from k_sweep_cells.csv)")
    for r in k["summary"]:
        print(f"- K={r['k_model']} PSR = {r['psr_mean']:.2f}  DPR = {r['dpr_mean']:.2f}")
    print()
    print("## Held-out OMD comparison (seeds 3–9, n=84, ESL λ=10, OMD η=1.0; recomputed)")
    d = omd["recomputed"]
    print(f"- ESL PSR = {d['esl_star_psr_mean']:.2f}  (accepted 86.73)")
    print(f"- OMD PSR = {d['omd_psr_mean']:.2f}  (accepted 116.67)")
    print(
        f"- paired OMD−ESL = {float(d['mean_delta']):.2f} "
        f"[{float(d['ci95_lo']):.2f}, {float(d['ci95_hi']):.2f}]"
    )
    print()
    print("## Population recovery (PSRV)")
    print(f"- N=20 recovery accuracy = {acc20:.3f}  (accepted 0.955)")
    print(f"- N=100 recovery accuracy = {acc100:.3f}  (accepted 0.847)")
    print()
    print(f"Wrote CSVs under {OUT}")


if __name__ == "__main__":
    main()
