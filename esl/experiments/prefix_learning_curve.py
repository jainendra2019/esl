"""Prefix learning curve: ESL θ MCE vs batch baselines at increasing round horizons (same run)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from esl.baselines.dataset import load_run_dataset, with_observation_prefix
from esl.baselines.evaluate import (
    esl_mce_from_prototype_trajectory,
    run_baselines_on_dataset,
)
from esl.experiments.manifest import write_run_manifest


def _belief_argmax_at_round(run_dir: Path, target_round: int) -> float | None:
    """Last metrics_trajectory row with round <= target_round."""
    path = run_dir / "metrics_trajectory.csv"
    if not path.is_file():
        return None
    best: dict[str, str] | None = None
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["round"])
            if r <= target_round:
                best = row
    if best is None:
        return None
    return float(best["belief_argmax_accuracy"])


def prefix_grid(max_round: int, *, n_points: int) -> list[int]:
    """Inclusive environment round indices from early to max_round (≈ evenly spaced)."""
    if max_round < 0:
        return []
    out: list[int] = []
    for k in range(1, n_points + 1):
        r = int(max_round * k / n_points)
        out.append(min(max_round, max(0, r)))
    return sorted(set(out))


def run_prefix_learning_curve(
    run_dir: Path,
    *,
    n_prefix_points: int = 10,
    em_restarts: int = 4,
    manifest_extra: dict[str, Any] | None = None,
    plot: bool = True,
) -> tuple[Path, Path | None]:
    """
    Requires ``interaction_observations.csv`` and ``prototype_trajectory.csv`` in ``run_dir``.

    Writes ``run_dir/prefix_learning_curve.csv`` and optional ``prefix_learning_curve.png``.
    """
    run_dir = Path(run_dir).resolve()
    ds_full = load_run_dataset(run_dir)
    sm_path = run_dir / "summary_metrics.json"
    if not sm_path.is_file():
        raise FileNotFoundError(f"Missing {sm_path}")
    sm = json.loads(sm_path.read_text(encoding="utf-8"))
    n_exec = int(sm.get("num_rounds_executed", 0))
    if n_exec < 1:
        raise ValueError("Run has no executed rounds")
    t_max = n_exec - 1
    rounds_list = prefix_grid(t_max, n_points=n_prefix_points)

    manifest = {
        "experiment": "prefix_learning_curve",
        "run_dir": str(run_dir),
        "n_prefix_points": n_prefix_points,
        "em_restarts": em_restarts,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    write_run_manifest(run_dir / "prefix_learning_curve_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for R in rounds_list:
        esl_m = esl_mce_from_prototype_trajectory(ds_full, max_round_inclusive=R)
        ds_p = with_observation_prefix(ds_full, R)
        out_b = run_baselines_on_dataset(ds_p, em_restarts=em_restarts)
        n_obs = out_b["n_observed_w_positive"]
        ba = _belief_argmax_at_round(run_dir, R)
        rows.append(
            {
                "prefix_max_round": R,
                "n_observed_w_positive": n_obs,
                "esl_mce_theta_at_round": esl_m if esl_m is not None else "",
                "em_conditional_mce": out_b["em_conditional_mce"],
                "kmeans_mce": out_b["kmeans_mce"],
                "belief_argmax_accuracy": ba if ba is not None else "",
            }
        )

    csv_path = run_dir / "prefix_learning_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "prefix_max_round",
                "n_observed_w_positive",
                "esl_mce_theta_at_round",
                "em_conditional_mce",
                "kmeans_mce",
                "belief_argmax_accuracy",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = run_dir / "prefix_learning_curve.png"
        plot_prefix_learning_curve(csv_path, png_path)
    return csv_path, png_path


def plot_prefix_learning_curve(csv_path: Path, out_png: Path) -> None:
    csv_path = Path(csv_path)
    pts: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append(row)
    if len(pts) < 2:
        return
    x = [int(r["n_observed_w_positive"]) for r in pts]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.8))

    def _flt(key: str, i: int) -> float:
        s = pts[i].get(key, "").strip()
        return float(s) if s != "" else float("nan")

    esl = [_flt("esl_mce_theta_at_round", i) for i in range(len(pts))]
    emc = [_flt("em_conditional_mce", i) for i in range(len(pts))]
    km = [_flt("kmeans_mce", i) for i in range(len(pts))]
    ax0.plot(x, esl, "o-", label="ESL θ (online @ round)", color="C0", linewidth=1.5)
    ax0.plot(x, emc, "s--", label="EM (cond.) batch @ prefix", color="C1", linewidth=1.5)
    ax0.plot(x, km, "^:", label="K-means @ prefix", color="C2", linewidth=1.5)
    ax0.set_xlabel("Cumulative observed interactions (w>0)")
    ax0.set_ylabel("MCE")
    ax0.set_title("Sample efficiency (prefix-matched batch)")
    ax0.legend(fontsize=7)
    ax0.grid(True, alpha=0.3)

    has_ba = any(r.get("belief_argmax_accuracy", "").strip() for r in pts)
    if has_ba:
        ba = [_flt("belief_argmax_accuracy", i) for i in range(len(pts))]
        ax1.plot(x, ba, "o-", color="C3", linewidth=1.5)
        ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Cumulative observed interactions (w>0)")
    ax1.set_ylabel("Belief argmax accuracy")
    ax1.set_title("ESL beliefs @ same horizon")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Prefix learning curve from one logged ESL run")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n-prefix-points", type=int, default=10)
    p.add_argument("--em-restarts", type=int, default=4)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    csv_path, png_path = run_prefix_learning_curve(
        args.run_dir,
        n_prefix_points=args.n_prefix_points,
        em_restarts=args.em_restarts,
        plot=not args.no_plot,
    )
    print(csv_path)
    if png_path:
        print(png_path)


if __name__ == "__main__":
    main()
