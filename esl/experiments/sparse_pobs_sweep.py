"""Small sparse-observability sweep: p_obs grid + ESL vs offline baseline MCE + belief metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from dataclasses import replace

from esl.baselines.evaluate import write_baselines_summary
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.manifest import write_run_manifest
from esl.experiments.presets import recovery_sparse_obs_cfg
from esl.trainer import run_esl

# Ordered for plot (high to low observability).
POBS_SWEEP_VALUES: tuple[float, ...] = (1.0, 0.5, 0.3, 0.2)


def _p_slug(p: float) -> str:
    return "p_obs_" + str(p).replace(".", "p")


CSV_FIELDNAMES: list[str] = [
    "p_obs",
    "run_id",
    "num_rounds_executed",
    "final_matched_cross_entropy",
    "final_mce",
    "final_belief_argmax_accuracy",
    "em_conditional_mce",
    "kmeans_mce",
    "fcm_mce",
    "n_observed_w_positive",
    "stopped_on_convergence",
    "convergence_round",
]


def run_sparse_pobs_sweep(
    *,
    out_root: Path,
    num_rounds: int,
    seed: int = 42,
    plot: bool = True,
    include_baselines: bool = True,
    em_restarts: int = 8,
    clustering_only_baselines: bool = False,
    p_obs_values: tuple[float, ...] | None = None,
) -> tuple[Path, Path | None]:
    """
    Run flagship-geometry recovery at each p_obs in POBS_SWEEP_VALUES with fixed num_rounds.

    When ``include_baselines``, sets ``log_interaction_observations=True`` and runs
    ``write_baselines_summary`` per run (transductive full-log batch MCE).

    When ``clustering_only_baselines`` is True, only K-means + FCM are fitted (no EM / Oracle;
    Milestone 3 paper-facing pack).

    Writes:
      - one run directory per p_obs under out_root/<p_slug>/seed_<seed>/
      - out_root/sparse_pobs_summary.csv
      - out_root/final_ce_vs_p_obs.png (if plot=True)
      - out_root/sparse_pobs_mce_belief.png (if plot=True)
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    grid = p_obs_values if p_obs_values is not None else POBS_SWEEP_VALUES

    for p_obs in grid:
        cfg = recovery_sparse_obs_cfg(p_obs=p_obs, seed=seed)
        cfg = replace(
            cfg,
            num_rounds=int(num_rounds),
            log_interaction_observations=include_baselines,
        )
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
                "include_baselines": include_baselines,
                "clustering_only_baselines": clustering_only_baselines,
            },
        )
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
        try:
            rel = run_dir.relative_to(out_root)
        except ValueError:
            rel = Path(run_dir.name)

        row: dict[str, Any] = {
            "p_obs": p_obs,
            "run_id": str(rel).replace("\\", "/"),
            "num_rounds_executed": summary["num_rounds_executed"],
            "final_matched_cross_entropy": summary["final_matched_cross_entropy"],
            "final_mce": summary.get("final_mce", ""),
            "final_belief_argmax_accuracy": summary.get("final_belief_argmax_accuracy", ""),
            "em_conditional_mce": "",
            "kmeans_mce": "",
            "fcm_mce": "",
            "n_observed_w_positive": "",
            "stopped_on_convergence": summary.get("stopped_on_convergence", ""),
            "convergence_round": summary.get("convergence_round", ""),
        }
        if include_baselines:
            bs_path = write_baselines_summary(
                run_dir,
                em_restarts=em_restarts,
                clustering_only=clustering_only_baselines,
            )
            bj = json.loads(bs_path.read_text(encoding="utf-8"))
            row["em_conditional_mce"] = bj.get("em_conditional_mce") or ""
            row["kmeans_mce"] = bj.get("kmeans_mce") or ""
            row["fcm_mce"] = bj.get("fcm_mce") or ""
            row["n_observed_w_positive"] = bj.get("n_observed_w_positive", "")
        rows.append(row)

    csv_path = out_root / "sparse_pobs_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = out_root / "final_ce_vs_p_obs.png"
        plot_final_ce_vs_pobs(csv_path, png_path)
        plot_sparse_pobs_mce_belief(csv_path, out_root / "sparse_pobs_mce_belief.png")

    return csv_path, png_path


def plot_final_ce_vs_pobs(summary_csv: Path, out_png: Path, *, dpi: int | None = None) -> None:
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
    fig.savefig(out_png, dpi=dpi or publication_figure_dpi())
    plt.close(fig)


def plot_sparse_pobs_mce_belief(summary_csv: Path, out_png: Path, *, dpi: int | None = None) -> None:
    """
    Two panels: (left) ESL vs batch MCE vs p_obs; (right) belief argmax accuracy (ESL only) vs p_obs.
    """
    summary_csv = summary_csv.resolve()
    rows: list[dict[str, str]] = []
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    rows.sort(key=lambda r: float(r["p_obs"]))
    xs = [float(r["p_obs"]) for r in rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)

    esl_mce = [float(r["final_mce"]) for r in rows if r.get("final_mce", "").strip() != ""]
    if len(esl_mce) == len(rows):
        ax0.plot(xs, esl_mce, "o-", color="C0", label="ESL final MCE", linewidth=1.5, markersize=7)
    if rows[0].get("em_conditional_mce", "").strip() != "":
        em_m = [float(r["em_conditional_mce"]) for r in rows]
        ax0.plot(xs, em_m, "s--", color="C1", label="EM (cond.) transductive", linewidth=1.5, markersize=6)
    if rows[0].get("kmeans_mce", "").strip() != "":
        km = [float(r["kmeans_mce"]) for r in rows]
        ax0.plot(xs, km, "^:", color="C2", label="K-means transductive", linewidth=1.5, markersize=6)
    if rows[0].get("fcm_mce", "").strip() != "":
        fc = [float(r["fcm_mce"]) for r in rows]
        ax0.plot(xs, fc, "d-.", color="C5", label="FCM transductive", linewidth=1.5, markersize=6)
    ax0.set_ylabel("MCE")
    ax0.set_title("Prototype MCE vs observability")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlabel(r"$p_{\mathrm{obs}}$")

    if rows[0].get("final_belief_argmax_accuracy", "").strip() != "":
        ba = [float(r["final_belief_argmax_accuracy"]) for r in rows]
        ax1.plot(xs, ba, "o-", color="C3", markersize=7, linewidth=1.5)
        ax1.set_ylabel("Belief argmax accuracy")
        ax1.set_ylim(0.0, 1.05)
    ax1.set_title("ESL belief quality (no batch counterpart)")
    ax1.set_xlabel(r"$p_{\mathrm{obs}}$")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(xs)

    fig.suptitle("Sparse observations: ESL vs offline baselines on same logged stream", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi or publication_figure_dpi())
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
    p.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip interaction log + offline baselines (faster; smaller CSV)",
    )
    p.add_argument(
        "--clustering-only-baselines",
        action="store_true",
        help="K-means + FCM only (no EM / Oracle); Milestone 3 default for paper-facing sweeps",
    )
    p.add_argument("--em-restarts", type=int, default=8)
    args = p.parse_args(argv)
    root = args.out_root.resolve()
    csv_path = root / "sparse_pobs_summary.csv"
    if args.plot_only:
        if not csv_path.is_file():
            raise SystemExit(f"missing {csv_path}; run without --plot-only first")
        out_png = root / "final_ce_vs_p_obs.png"
        plot_final_ce_vs_pobs(csv_path, out_png)
        plot_sparse_pobs_mce_belief(csv_path, root / "sparse_pobs_mce_belief.png")
        print(f"Wrote: {out_png}")
        print(f"Wrote: {root / 'sparse_pobs_mce_belief.png'}")
        return
    cpath, ppath = run_sparse_pobs_sweep(
        out_root=root,
        num_rounds=args.rounds,
        seed=args.seed,
        plot=not args.no_plot,
        include_baselines=not args.no_baselines,
        em_restarts=args.em_restarts,
        clustering_only_baselines=args.clustering_only_baselines,
    )
    print(f"Wrote: {cpath}")
    if ppath:
        print(f"Wrote: {ppath}")
        print(f"Wrote: {root / 'sparse_pobs_mce_belief.png'}")
