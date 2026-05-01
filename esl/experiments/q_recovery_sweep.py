"""Recovery sweep over prototype batch horizon Q (``prototype_update_every``), flagship geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

from esl.baselines.evaluate import write_baselines_summary
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.manifest import write_run_manifest
from esl.experiments.presets import recovery_Q_sweep_cfg
from esl.trainer import run_esl

DEFAULT_Q_VALUES: tuple[int, ...] = (1, 5, 15)

Q_CSV_FIELDS: list[str] = [
    "prototype_update_every_q",
    "run_id",
    "num_rounds_executed",
    "final_mce",
    "final_matched_cross_entropy",
    "final_belief_argmax_accuracy",
    "kmeans_mce",
    "fcm_mce",
    "n_observed_w_positive",
    "prototype_update_count",
]


def run_q_recovery_sweep(
    *,
    out_root: Path,
    num_rounds: int,
    seed: int = 42,
    q_values: tuple[int, ...] = DEFAULT_Q_VALUES,
    plot: bool = True,
    include_baselines: bool = True,
    em_restarts: int = 4,
    clustering_only_baselines: bool = False,
) -> tuple[Path, Path | None]:
    """
    Flagship recovery geometry with varying prototype SGD cadence ``Q``.

    Writes ``out_root/q_recovery_summary.csv`` and optional ``out_root/q_vs_mce.png``.
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for q in q_values:
        cfg = recovery_Q_sweep_cfg(Q=int(q), seed=seed)
        cfg = replace(
            cfg,
            num_rounds=int(num_rounds),
            log_interaction_observations=include_baselines,
        )
        cfg.validate()
        slug = f"Q_{q}"
        run_dir = out_root / slug / f"seed_{seed}"
        write_run_manifest(
            run_dir / "run_manifest.json",
            {
                "preset": "q_recovery_sweep",
                "variant": slug,
                "seed": seed,
                "prototype_update_every_q": q,
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
            "prototype_update_every_q": int(q),
            "run_id": str(rel).replace("\\", "/"),
            "num_rounds_executed": summary["num_rounds_executed"],
            "final_mce": summary.get("final_mce", ""),
            "final_matched_cross_entropy": summary.get("final_matched_cross_entropy", ""),
            "final_belief_argmax_accuracy": summary.get("final_belief_argmax_accuracy", ""),
            "kmeans_mce": "",
            "fcm_mce": "",
            "n_observed_w_positive": "",
            "prototype_update_count": summary.get("prototype_update_count", ""),
        }
        if include_baselines:
            bs_path = write_baselines_summary(
                run_dir,
                em_restarts=em_restarts,
                clustering_only=clustering_only_baselines,
            )
            bj = json.loads(bs_path.read_text(encoding="utf-8"))
            row["kmeans_mce"] = bj.get("kmeans_mce", "")
            row["fcm_mce"] = bj.get("fcm_mce", "")
            row["n_observed_w_positive"] = bj.get("n_observed_w_positive", "")
        rows.append(row)

    csv_path = out_root / "q_recovery_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=Q_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = out_root / "q_vs_mce.png"
        plot_q_vs_mce(csv_path, png_path)
    return csv_path, png_path


def plot_q_vs_mce(summary_csv: Path, out_png: Path, *, dpi: int | None = None) -> None:
    summary_csv = summary_csv.resolve()
    pts: list[tuple[int, float, float, float]] = []
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qv = int(float(row["prototype_update_every_q"]))
            esl = float(row["final_mce"]) if row.get("final_mce", "").strip() else float("nan")
            km = (
                float(row["kmeans_mce"])
                if row.get("kmeans_mce", "").strip() != ""
                else float("nan")
            )
            fc = (
                float(row["fcm_mce"])
                if row.get("fcm_mce", "").strip() != ""
                else float("nan")
            )
            pts.append((qv, esl, km, fc))
    pts.sort(key=lambda t: t[0])
    qs = [p[0] for p in pts]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(qs, [p[1] for p in pts], "o-", label="ESL MCE", color="C0", linewidth=1.5, markersize=7)
    if not all(np.isnan(p[2]) for p in pts):
        ax.plot(qs, [p[2] for p in pts], "s--", label="K-means", color="C2", linewidth=1.5, markersize=6)
    if not all(np.isnan(p[3]) for p in pts):
        ax.plot(qs, [p[3] for p in pts], "d-.", label="FCM", color="C5", linewidth=1.5, markersize=6)
    ax.set_xlabel(r"Prototype update horizon $Q$ (interactions per SGD step)")
    ax.set_ylabel("MCE")
    ax.set_title("Recovery: ESL vs offline clustering vs batch cadence")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi or publication_figure_dpi())
    plt.close(fig)


def main_q_recovery(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Recovery sweep over prototype_update_every Q")
    p.add_argument("--out-root", type=Path, default=Path("runs/q_recovery_sweep"))
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--q-values",
        type=str,
        default="1,5,15",
        help="comma-separated Q list, e.g. 1,5,15",
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--clustering-only-baselines", action="store_true")
    p.add_argument("--em-restarts", type=int, default=8)
    args = p.parse_args(argv)
    qvals = tuple(int(x.strip()) for x in args.q_values.split(",") if x.strip())
    cpath, ppath = run_q_recovery_sweep(
        out_root=args.out_root.resolve(),
        num_rounds=args.rounds,
        seed=args.seed,
        q_values=qvals,
        plot=not args.no_plot,
        include_baselines=not args.no_baselines,
        em_restarts=args.em_restarts,
        clustering_only_baselines=args.clustering_only_baselines,
    )
    print(f"Wrote: {cpath}")
    if ppath:
        print(f"Wrote: {ppath}")
