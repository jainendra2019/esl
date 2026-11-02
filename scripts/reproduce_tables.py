#!/usr/bin/env python3
"""Regenerate published numerical summaries from the frozen evidence package.

Does not rerun simulations.

Inputs:  evidence/**/*.csv and evidence/**/*.json
Outputs: outputs/tables/*.csv and a markdown summary on stdout
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "outputs" / "tables"
FOUR_METHODS = ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM")


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


def k_table() -> list[dict[str, str]]:
    rows = load_csv(EVIDENCE / "k_sensitivity" / "summary.csv")
    write_csv(OUT / "k_sensitivity_summary.csv", rows, list(rows[0].keys()))
    return rows


def omd_table() -> dict[str, Any]:
    rows = load_csv(EVIDENCE / "omd" / "omd_psr_summary.csv")
    write_csv(OUT / "omd_psr_summary.csv", rows, list(rows[0].keys()))
    paired = json.loads((EVIDENCE / "omd" / "omd_vs_esl_paired.json").read_text(encoding="utf-8"))
    (OUT / "omd_vs_esl_paired.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    primary = {r["method"]: r for r in rows if r["layer"] == "primary_held_out"}
    d = paired["primary_held_out_seeds_3_9"]["per_opponent_minus_esl_star"]
    return {"primary": primary, "delta": d}


def copy_simple(src: Path, dst_name: str) -> None:
    if not src.is_file():
        return
    rows = load_csv(src)
    if not rows:
        return
    write_csv(OUT / dst_name, rows, list(rows[0].keys()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flag = flagship_tables()
    k_rows = k_table()
    omd = omd_table()
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
    print("## K-sensitivity (ESL only)")
    for r in k_rows:
        if r["metric"] in ("psr", "dpr"):
            print(f"- K={r['k_model']} {r['metric'].upper()} = {float(r['mean']):.2f}")
    print()
    print("## Held-out OMD comparison (seeds 3–9, n=84, ESL λ=10, OMD η=1.0)")
    p = omd["primary"]
    print(f"- ESL PSR = {float(p['ESL-decision-tuned']['psr_mean']):.2f}  (accepted 86.73)")
    print(f"- OMD PSR = {float(p['Per-opponent OMD/MWU']['psr_mean']):.2f}  (accepted 116.67)")
    d = omd["delta"]
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
