#!/usr/bin/env python3
"""PSRV evaluation: Algorithm 1 recovery at N=20 and N=100 (K=A=3).

Supports --resume and incremental CSV checkpoints after every completed cell so
sleep/disconnect does not lose aggregate progress (per-run dirs are always saved).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from esl.population_spec import PopulationSpec
from esl.psrv.presets import population_n20, population_n100, recovery_psrv_cfg
from esl.res.benchmark import default_benchmark_root, load_suite
from esl.res.game_spec import GameSpec
from esl.trainer import run_esl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "population"
SEEDS = tuple(range(10))
HORIZON = 10_000
LATENCY_THRESHOLD = 0.9
CELL_COLUMNS = (
    "n",
    "l_min",
    "l_max",
    "q",
    "family",
    "instance_id",
    "seed",
    "horizon",
    "num_actions",
    "num_prototypes",
    "recovery_accuracy",
    "mce",
    "belief_entropy",
    "matched_ce",
    "prototype_gap",
    "prototype_update_count",
    "latency_to_acc_0_9",
    "run_dir",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def _append_or_upsert_cell(path: Path, row: dict[str, Any], columns: Sequence[str]) -> None:
    """Atomically rewrite cells CSV with this row upserted (safe incremental checkpoint)."""
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    key = (str(row["instance_id"]), int(row["seed"]))
    kept = [r for r in existing if (str(r["instance_id"]), int(r["seed"])) != key]
    kept.append({c: row.get(c, "") for c in columns})
    kept.sort(key=lambda r: (str(r["family"]), str(r["instance_id"]), int(r["seed"])))
    _write_csv(path, kept, columns)


def _latency_to_threshold(metrics_csv: Path, threshold: float = LATENCY_THRESHOLD) -> float | str:
    if not metrics_csv.is_file():
        return ""
    with metrics_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = row.get("belief_argmax_accuracy", "")
            if acc == "":
                continue
            if float(acc) >= float(threshold):
                return float(row["round"])
    return ""


def _cell_row_from_run(
    *,
    run_dir: Path,
    spec: GameSpec,
    population: PopulationSpec,
    seed: int,
    horizon: int,
) -> dict[str, Any] | None:
    """Load a completed full-horizon cell from disk, or None if incomplete/smoke."""
    sm_path = run_dir / "summary_metrics.json"
    cfg_path = run_dir / "config.json"
    if not sm_path.is_file():
        return None
    summary = json.loads(sm_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    executed = int(summary.get("num_rounds_executed", summary.get("num_rounds", 0)))
    configured = int(cfg.get("num_rounds", executed))
    if executed < int(horizon) or configured < int(horizon):
        return None
    latency = _latency_to_threshold(run_dir / "metrics_trajectory.csv")
    return {
        "n": population.n,
        "l_min": population.l_min,
        "l_max": population.l_max,
        "q": population.q,
        "family": spec.family,
        "instance_id": spec.instance_id,
        "seed": seed,
        "horizon": horizon,
        "num_actions": 3,
        "num_prototypes": 3,
        "recovery_accuracy": float(summary["final_belief_argmax_accuracy"]),
        "mce": float(summary["final_mce"]),
        "belief_entropy": float(summary["final_belief_entropy"]),
        "matched_ce": float(summary["final_matched_cross_entropy"]),
        "prototype_gap": summary.get("final_prototype_gap", ""),
        "prototype_update_count": int(summary["prototype_update_count"]),
        "latency_to_acc_0_9": latency,
        "run_dir": str(run_dir),
    }


def _run_one_cell(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: one (environment, seed) recovery cell."""
    spec = GameSpec.from_dict(payload["spec"])
    population = PopulationSpec(
        n=int(payload["n"]),
        l_min=int(payload["l_min"]),
        l_max=int(payload["l_max"]),
        q=int(payload["q"]),
    )
    seed = int(payload["seed"])
    horizon = int(payload["horizon"])
    run_dir = Path(payload["run_dir"])
    # Clear incomplete / smoke leftovers before a full run.
    if run_dir.is_dir():
        sm = run_dir / "summary_metrics.json"
        cfgp = run_dir / "config.json"
        ok = False
        if sm.is_file():
            summary = json.loads(sm.read_text(encoding="utf-8"))
            cfg = json.loads(cfgp.read_text(encoding="utf-8")) if cfgp.is_file() else {}
            executed = int(summary.get("num_rounds_executed", 0))
            configured = int(cfg.get("num_rounds", 0))
            ok = executed >= horizon and configured >= horizon
        if not ok:
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)
    cfg = recovery_psrv_cfg(
        seed=seed,
        spec=spec,
        population=population,
        num_rounds=horizon,
    )
    _, _, _, summary, rd = run_esl(cfg, run_dir=run_dir)
    latency = _latency_to_threshold(Path(rd) / "metrics_trajectory.csv")
    return {
        "n": population.n,
        "l_min": population.l_min,
        "l_max": population.l_max,
        "q": population.q,
        "family": spec.family,
        "instance_id": spec.instance_id,
        "seed": seed,
        "horizon": horizon,
        "num_actions": 3,
        "num_prototypes": 3,
        "recovery_accuracy": float(summary["final_belief_argmax_accuracy"]),
        "mce": float(summary["final_mce"]),
        "belief_entropy": float(summary["final_belief_entropy"]),
        "matched_ce": float(summary["final_matched_cross_entropy"]),
        "prototype_gap": summary.get("final_prototype_gap", ""),
        "prototype_update_count": int(summary["prototype_update_count"]),
        "latency_to_acc_0_9": latency,
        "run_dir": str(rd),
    }


def rebuild_cells_from_runs(
    *,
    out_dir: Path,
    population: PopulationSpec,
    seeds: Sequence[int],
    horizon: int,
    label: str,
) -> list[dict[str, Any]]:
    """Scan run dirs and write aggregate CSV from completed full-horizon cells."""
    specs = load_suite(default_benchmark_root(ROOT), num_actions=3)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            run_dir = out_dir / "runs" / f"{spec.instance_id}_seed{seed}"
            row = _cell_row_from_run(
                run_dir=run_dir,
                spec=spec,
                population=population,
                seed=int(seed),
                horizon=int(horizon),
            )
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (r["family"], r["instance_id"], r["seed"]))
    cells_csv = out_dir / f"psrv_{label}_cells.csv"
    _write_csv(cells_csv, rows, CELL_COLUMNS)
    print(f"[rebuild] {label}: wrote {len(rows)} completed cells → {cells_csv}", flush=True)
    return rows


def run_grid(
    *,
    out_dir: Path,
    population: PopulationSpec,
    seeds: Sequence[int],
    horizon: int,
    label: str,
    jobs: int = 1,
    resume: bool = True,
) -> list[dict[str, Any]]:
    specs = load_suite(default_benchmark_root(ROOT), num_actions=3)
    if len(specs) != 12:
        raise RuntimeError(f"expected 12 committed 3×3 environments, got {len(specs)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = out_dir / f"psrv_{label}_cells.csv"

    # Always rebuild aggregate from disk first (authoritative checkpoint).
    summaries = rebuild_cells_from_runs(
        out_dir=out_dir,
        population=population,
        seeds=seeds,
        horizon=horizon,
        label=label,
    )
    done_keys = {(str(r["instance_id"]), int(r["seed"])) for r in summaries}

    payloads: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            key = (spec.instance_id, int(seed))
            if resume and key in done_keys:
                continue
            payloads.append(
                {
                    "spec": spec.to_dict(),
                    "n": population.n,
                    "l_min": population.l_min,
                    "l_max": population.l_max,
                    "q": population.q,
                    "seed": int(seed),
                    "horizon": int(horizon),
                    "run_dir": str(out_dir / "runs" / f"{spec.instance_id}_seed{seed}"),
                }
            )

    total_target = len(specs) * len(seeds)
    print(
        f"[{label}] resume={resume}: {len(summaries)}/{total_target} done, "
        f"{len(payloads)} remaining, jobs={max(1, int(jobs))}",
        flush=True,
    )
    if not payloads:
        write_family_summary(summaries, out_dir / f"psrv_{label}_family_summary.csv")
        (out_dir / "gate.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "n": population.n,
                    "n_cells": len(summaries),
                    "horizon": horizon,
                    "seeds": list(seeds),
                    "jobs": max(1, int(jobs)),
                    "resumed_complete": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return summaries

    t0 = time.time()
    workers = max(1, int(jobs))
    finished_new = 0

    def _ingest(row: dict[str, Any]) -> None:
        nonlocal finished_new
        summaries.append(row)
        _append_or_upsert_cell(cells_csv, row, CELL_COLUMNS)
        finished_new += 1
        print(
            f"[{label}] checkpoint {len(summaries)}/{total_target} "
            f"(+{finished_new} this session, elapsed {time.time()-t0:.0f}s) "
            f"{row['instance_id']} seed={row['seed']} "
            f"acc={row['recovery_accuracy']:.3f} mce={row['mce']:.3f}",
            flush=True,
        )

    if workers == 1:
        for payload in payloads:
            _ingest(_run_one_cell(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one_cell, p): p for p in payloads}
            for fut in as_completed(futs):
                _ingest(fut.result())

    summaries.sort(key=lambda r: (r["family"], r["instance_id"], r["seed"]))
    _write_csv(cells_csv, summaries, CELL_COLUMNS)
    write_family_summary(summaries, out_dir / f"psrv_{label}_family_summary.csv")
    (out_dir / "gate.json").write_text(
        json.dumps(
            {
                "label": label,
                "n": population.n,
                "n_cells": len(summaries),
                "elapsed_s_session": time.time() - t0,
                "horizon": horizon,
                "seeds": list(seeds),
                "jobs": workers,
                "resume": resume,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries


def write_family_summary(cells: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for family in sorted({c["family"] for c in cells}) + ["all"]:
        for metric in ("recovery_accuracy", "mce", "belief_entropy"):
            vals = [
                float(c[metric])
                for c in cells
                if metric in c and c.get(metric, "") != "" and (family == "all" or c["family"] == family)
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            n = len(arr)
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "ci95_low": mean - 1.96 * se,
                    "ci95_high": mean + 1.96 * se,
                }
            )
    _write_csv(path, rows, ["family", "metric", "n", "mean", "ci95_low", "ci95_high"])


def main() -> None:
    p = argparse.ArgumentParser(description="PSRV Algorithm 1 population recovery evaluation")
    p.add_argument("--out-root", type=Path, default=OUT)
    p.add_argument("--only", choices=("n20", "n100", "all", "smoke", "rebuild"), default="all")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--jobs", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--no-resume", action="store_true", help="Re-run all cells (ignore completed)")
    args = p.parse_args()
    seeds: Sequence[int] = tuple(args.seeds) if args.seeds is not None else SEEDS
    horizon = 50 if args.only == "smoke" else int(args.horizon)
    if args.only == "smoke":
        seeds = (0, 1)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = int(args.jobs)
    resume = not bool(args.no_resume)

    if args.only == "rebuild":
        rebuild_cells_from_runs(
            out_dir=out_root / "N20",
            population=population_n20(),
            seeds=seeds,
            horizon=horizon,
            label="n20",
        )
        if (out_root / "N100" / "runs").is_dir():
            rebuild_cells_from_runs(
                out_dir=out_root / "N100",
                population=population_n100(),
                seeds=seeds,
                horizon=horizon,
                label="n100",
            )
        return

    if args.only in ("n20", "all", "smoke"):
        print("=== PSRV N=20 K=A=3 ===", flush=True)
        run_grid(
            out_dir=out_root / "N20",
            population=population_n20(),
            seeds=seeds,
            horizon=horizon,
            label="n20",
            jobs=jobs,
            resume=resume,
        )

    if args.only in ("n100", "all"):
        print("=== PSRV N=100 K=A=3 ===", flush=True)
        run_grid(
            out_dir=out_root / "N100",
            population=population_n100(),
            seeds=seeds,
            horizon=horizon,
            label="n100",
            jobs=jobs,
            resume=resume,
        )


if __name__ == "__main__":
    main()
