"""Small sparse-observability sweep: p_obs grid + final CE vs p_obs plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from dataclasses import replace

from esl.experiments.manifest import write_run_manifest
from esl.experiments.presets import recovery_sparse_obs_cfg
from esl.trainer import run_esl

# Ordered for plot (high to low observability).
POBS_SWEEP_VALUES: tuple[float, ...] = (1.0, 0.5, 0.3, 0.2)


def _p_slug(p: float) -> str:
    return "p_obs_" + str(p).replace(".", "p")


def run_sparse_pobs_sweep(
    *,
    out_root: Path,
    num_rounds: int,
    seed: int = 42,
    plot: bool = True,
) -> tuple[Path, Path | None]:
    """
    Run flagship-geometry recovery at each p_obs in POBS_SWEEP_VALUES with fixed num_rounds.

    Writes:
      - one run directory per p_obs under out_root/<p_slug>/seed_<seed>/
      - out_root/sparse_pobs_summary.csv
      - out_root/final_ce_vs_p_obs.png (if plot=True)
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for p_obs in POBS_SWEEP_VALUES:
        cfg = recovery_sparse_obs_cfg(p_obs=p_obs, seed=seed)
        cfg = replace(cfg, num_rounds=int(num_rounds))
        cfg.validate()
        slug = _p_slug(p_obs)
        run_dir = out_root / slug / f"seed_{seed}"
        write_run_manifest(
            run_dir / "run_manifest.json",
            {
                "preset": "sparse_pobs_sweep",
                "variant": slug,
                "seed": seed,
                "p_obs": p_obs,
                "num_rounds_requested": num_rounds,
            },
        )
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
        try:
            rel = run_dir.relative_to(out_root)
        except ValueError:
            rel = Path(run_dir.name)
        rows.append(
            {
                "p_obs": p_obs,
                "run_id": str(rel).replace("\\", "/"),
                "num_rounds_executed": summary["num_rounds_executed"],
                "final_matched_cross_entropy": summary["final_matched_cross_entropy"],
                "final_mce": summary.get("final_mce", ""),
                "stopped_on_convergence": summary.get("stopped_on_convergence", ""),
                "convergence_round": summary.get("convergence_round", ""),
            }
        )

    csv_path = out_root / "sparse_pobs_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "p_obs",
                "run_id",
                "num_rounds_executed",
                "final_matched_cross_entropy",
                "final_mce",
                "stopped_on_convergence",
                "convergence_round",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = out_root / "final_ce_vs_p_obs.png"
        plot_final_ce_vs_pobs(csv_path, png_path)

    return csv_path, png_path


def plot_final_ce_vs_pobs(summary_csv: Path, out_png: Path) -> None:
    """Line/scatter plot: final Hungarian total CE vs p_obs (reads sparse_pobs_summary.csv)."""
    summary_csv = summary_csv.resolve()
    pts: list[tuple[float, float]] = []
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append((float(row["p_obs"]), float(row["final_matched_cross_entropy"])))
    pts.sort(key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, ys, "o-", color="C0", markersize=8, linewidth=1.5)
    ax.set_xlabel(r"$p_{\mathrm{obs}}$")
    ax.set_ylabel("Final matched CE (Hungarian total)")
    ax.set_title("Recovery vs observation probability")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main_sparse_pobs(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sparse p_obs sweep {1.0,0.5,0.3,0.2} + CE plot")
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sparse_pobs_sweep"),
        help="output root for runs + summary CSV + PNG",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=3000,
        help="environment rounds per p_obs (same flagship geometry, shorter than 10k default)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true", help="only write CSV and runs")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="only read existing sparse_pobs_summary.csv and rewrite PNG",
    )
    args = p.parse_args(argv)
    root = args.out_root.resolve()
    csv_path = root / "sparse_pobs_summary.csv"
    if args.plot_only:
        if not csv_path.is_file():
            raise SystemExit(f"missing {csv_path}; run without --plot-only first")
        out_png = root / "final_ce_vs_p_obs.png"
        plot_final_ce_vs_pobs(csv_path, out_png)
        print(f"Wrote: {out_png}")
        return
    cpath, ppath = run_sparse_pobs_sweep(
        out_root=root,
        num_rounds=args.rounds,
        seed=args.seed,
        plot=not args.no_plot,
    )
    print(f"Wrote: {cpath}")
    if ppath:
        print(f"Wrote: {ppath}")
