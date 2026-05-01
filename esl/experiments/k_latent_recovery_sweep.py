"""Recovery sweep over number of latent prototypes K (flagship geometry, cyclic types)."""

from __future__ import annotations

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
from esl.experiments.presets import recovery_flagship_cfg
from esl.trainer import run_esl

DEFAULT_K_VALUES: tuple[int, ...] = (2, 3)

K_CSV_FIELDS: list[str] = [
    "num_prototypes",
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


def run_k_latent_recovery_sweep(
    *,
    out_root: Path,
    num_rounds: int,
    seed: int = 42,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    plot: bool = True,
    include_baselines: bool = True,
    em_restarts: int = 4,
    clustering_only_baselines: bool = False,
) -> tuple[Path, Path | None]:
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for k in k_values:
        cfg = replace(recovery_flagship_cfg(seed=seed), num_prototypes=int(k))
        cfg = replace(
            cfg,
            num_rounds=int(num_rounds),
            log_interaction_observations=include_baselines,
        )
        cfg.validate()
        slug = f"K_{k}"
        run_dir = out_root / slug / f"seed_{seed}"
        write_run_manifest(
            run_dir / "run_manifest.json",
            {
                "preset": "k_latent_recovery_sweep",
                "variant": slug,
                "seed": seed,
                "num_prototypes": int(k),
                "num_rounds_requested": num_rounds,
                "clustering_only_baselines": clustering_only_baselines,
            },
        )
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
        try:
            rel = run_dir.relative_to(out_root)
        except ValueError:
            rel = Path(run_dir.name)

        row: dict[str, Any] = {
            "num_prototypes": int(k),
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
            row["kmeans_mce"] = bj.get("kmeans_mce") or ""
            row["fcm_mce"] = bj.get("fcm_mce") or ""
            row["n_observed_w_positive"] = bj.get("n_observed_w_positive", "")
        rows.append(row)

    csv_path = out_root / "k_latent_recovery_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=K_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = out_root / "k_vs_mce.png"
        plot_k_vs_mce(csv_path, png_path)
    return csv_path, png_path


def plot_k_vs_mce(summary_csv: Path, out_png: Path, *, dpi: int | None = None) -> None:
    pts: list[tuple[int, float, float, float]] = []
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kv = int(row["num_prototypes"])
            esl = float(row["final_mce"]) if row.get("final_mce", "").strip() else float("nan")
            km = float(row["kmeans_mce"]) if row.get("kmeans_mce", "").strip() else float("nan")
            fc = float(row["fcm_mce"]) if row.get("fcm_mce", "").strip() else float("nan")
            pts.append((kv, esl, km, fc))
    pts.sort(key=lambda t: t[0])
    ks = [p[0] for p in pts]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(ks, [p[1] for p in pts], "o-", label="ESL MCE", color="C0", linewidth=1.5, markersize=8)
    if not all(np.isnan(p[2]) for p in pts):
        ax.plot(ks, [p[2] for p in pts], "s--", label="K-means", color="C2", linewidth=1.5, markersize=7)
    if not all(np.isnan(p[3]) for p in pts):
        ax.plot(ks, [p[3] for p in pts], "d-.", label="FCM", color="C5", linewidth=1.5, markersize=7)
    ax.set_xlabel(r"Number of latent types $K$")
    ax.set_xticks(ks)
    ax.set_ylabel("MCE")
    ax.set_title("Latent cardinality (recovery, flagship geometry)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi or publication_figure_dpi())
    plt.close(fig)
