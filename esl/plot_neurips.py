"""NeurIPS-style figures: flagship panels, robustness from aggregate CSV, failure vs success overlay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_metrics_trajectory(run_dir: Path) -> tuple[list[int], dict[str, list[float]]]:
    path = run_dir / "metrics_trajectory.csv"
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return [], {}
    rounds = [int(row["round"]) for row in rows]
    keys = [k for k in rows[0] if k != "round"]
    data = {k: [float(row[k]) for row in rows] for k in keys}
    return rounds, data


def _read_prototype_p_coop(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    path = run_dir / "prototype_trajectory.csv"
    rounds: list[int] = []
    p0: list[float] = []
    p1: list[float] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            p0.append(float(row["softmax_0_0"]))
            p1.append(float(row["softmax_1_0"]))
    return rounds, p0, p1


def plot_flagship_panels(run_dir: Path, out_path: Path, *, title: str | None = None) -> None:
    """Four panels: matched CE, belief entropy, argmax accuracy, P(C) per prototype row."""
    run_dir = run_dir.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rounds, mt = _read_metrics_trajectory(run_dir)
    _, p0, p1 = _read_prototype_p_coop(run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(rounds, mt["matched_cross_entropy"], color="C0")
    ax.set_ylabel("Matched CE")
    ax.set_title("Hungarian matched cross-entropy")

    ax = axes[0, 1]
    ax.plot(rounds, mt["belief_entropy_mean"], color="C1")
    ax.set_ylabel("Mean entropy")
    ax.set_title("Belief entropy (mean off-diagonal)")

    ax = axes[1, 0]
    ax.plot(rounds, mt["belief_argmax_accuracy"], color="C2")
    ax.set_ylabel("Accuracy")
    ax.set_title("Belief argmax accuracy")
    ax.set_xlabel("Environment round")

    ax = axes[1, 1]
    ax.plot(rounds, p0, label="P(C) row 0", color="C3")
    ax.plot(rounds, p1, label="P(C) row 1", color="C4", alpha=0.85)
    ax.set_ylabel("P(cooperate)")
    ax.set_title("Prototype P(C) by row")
    ax.set_xlabel("Environment round")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(title or f"Flagship-style run: {run_dir.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_robustness_from_csv(csv_path: Path, out_path: Path) -> None:
    """Scatter/line panels: MCE vs p_obs, vs interaction budget, vs lr (requires columns present)."""
    csv_path = csv_path.resolve()
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")

    def ffloat(x: str) -> float | None:
        if x is None or x == "":
            return None
        return float(x)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    xs_p = []
    ys_p = []
    for row in rows:
        x = ffloat(str(row.get("p_obs", "")))
        y = ffloat(str(row.get("final_mce", row.get("final_matched_cross_entropy", ""))))
        if x is not None and y is not None:
            xs_p.append(x)
            ys_p.append(y)
    axes[0].scatter(xs_p, ys_p, c="C0")
    axes[0].set_xlabel("p_obs")
    axes[0].set_ylabel("Final MCE")
    axes[0].set_title("MCE vs p_obs")

    xs_b = []
    ys_b = []
    for row in rows:
        x = ffloat(str(row.get("num_interaction_events_executed", "")))
        y = ffloat(str(row.get("final_mce", row.get("final_matched_cross_entropy", ""))))
        if x is not None and y is not None:
            xs_b.append(x)
            ys_b.append(y)
    axes[1].scatter(xs_b, ys_b, c="C1")
    axes[1].set_xlabel("Interaction events")
    axes[1].set_ylabel("Final MCE")
    axes[1].set_title("MCE vs interactions")

    xs_lr = []
    ys_lr = []
    for row in rows:
        x = ffloat(str(row.get("prototype_lr_scale", "")))
        y = ffloat(str(row.get("final_prototype_gap", "")))
        if x is not None and y is not None:
            xs_lr.append(x)
            ys_lr.append(y)
    axes[2].scatter(xs_lr, ys_lr, c="C2")
    axes[2].set_xlabel("prototype_lr_scale")
    axes[2].set_ylabel("Final prototype gap")
    axes[2].set_title("Prototype gap vs LR")

    fig.suptitle(f"Robustness summary ({csv_path.name})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_failure_vs_success(weak_dir: Path, strong_dir: Path, out_path: Path) -> None:
    """Overlay matched CE trajectories for two runs."""
    r_w, m_w = _read_metrics_trajectory(weak_dir)
    r_s, m_s = _read_metrics_trajectory(strong_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r_w, m_w["matched_cross_entropy"], label="weak / failure", alpha=0.9)
    ax.plot(r_s, m_s["matched_cross_entropy"], label="strong / flagship", alpha=0.85)
    ax.set_xlabel("Round")
    ax.set_ylabel("Matched CE")
    ax.legend()
    ax.set_title("Failure vs success (matched CE)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="NeurIPS-style ESL plots")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("flagship", help="four-panel figure from one run directory")
    pf.add_argument("run_dir", type=Path)
    pf.add_argument("-o", "--out", type=Path, required=True)

    pr = sub.add_parser("robustness", help="panels from aggregate CSV")
    pr.add_argument("csv_path", type=Path)
    pr.add_argument("-o", "--out", type=Path, required=True)

    pc = sub.add_parser("compare", help="overlay two runs (matched CE)")
    pc.add_argument("weak_dir", type=Path)
    pc.add_argument("strong_dir", type=Path)
    pc.add_argument("-o", "--out", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "flagship":
        plot_flagship_panels(args.run_dir, args.out)
    elif args.cmd == "robustness":
        plot_robustness_from_csv(args.csv_path, args.out)
    elif args.cmd == "compare":
        plot_failure_vs_success(args.weak_dir, args.strong_dir, args.out)


if __name__ == "__main__":
    main()
