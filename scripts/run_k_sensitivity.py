#!/usr/bin/env python3
"""Rerun the matched ESL K∈{2,3,5} sweep on the committed twelve-game suite.

This is expensive (360 full-horizon cells). K=3 is the identity path and should
reproduce evidence/flagship/cells.csv ESL rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from esl.res.k_sensitivity import K_GRID, SEEDS, HORIZON, load_suite_3x3, run_esl_k_cell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/k_sensitivity"))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--smoke", action="store_true", help="One game, one seed, short horizon")
    args = p.parse_args()
    specs = load_suite_3x3()
    seeds = SEEDS
    k_grid = K_GRID
    horizon = int(args.horizon)
    if args.smoke:
        specs = specs[:1]
        seeds = (0,)
        horizon = 40
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for seed in seeds:
            for k in k_grid:
                result = run_esl_k_cell(spec=spec, seed=seed, k_model=k, horizon=horizon)
                s = result["summary"]
                rows.append(
                    {
                        "family": s["family"],
                        "instance_id": s["instance_id"],
                        "seed": int(s["seed"]),
                        "k_model": int(s["k_model"]),
                        "k_true": int(s["k_true"]),
                        "method": "ESL",
                        "psr": s["psr"],
                        "dpr": s["dpr"],
                        "lte_auc": s["lte_auc"],
                    }
                )
                print(
                    f"{spec.instance_id} seed={seed} K={k} PSR={float(s['psr']):.3f}",
                    flush=True,
                )
    path = args.out / "k_sweep_cells.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
