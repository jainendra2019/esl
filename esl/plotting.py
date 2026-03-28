"""Belief, prototype, and recovery plots from run CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return [], []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        rows.append(dict(zip(header, parts)))
    return header, rows


def plot_run(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    proto_h, proto_rows = _read_csv_dicts(run_dir / "prototype_trajectory.csv")
    met_h, met_rows = _read_csv_dicts(run_dir / "metrics_trajectory.csv")

    if proto_rows and proto_h:
        rounds = [int(r["round"]) for r in proto_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in proto_h:
            if key.startswith("softmax_") and key.endswith("_0"):
                ax.plot(rounds, [float(r[key]) for r in proto_rows], label=key)
        ax.set_xlabel("round")
        ax.set_ylabel("P(Cooperate)")
        ax.set_title("Prototype trajectories (softmax, cooperate dim)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(run_dir / "plot_prototype_softmax.png", dpi=150)
        plt.close(fig)

    if met_rows and met_h:
        rounds = [int(r["round"]) for r in met_rows]
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(rounds, [float(r["matched_cross_entropy"]) for r in met_rows], color="C0")
        axes[0].set_ylabel("matched_cross_entropy")
        axes[0].set_title("Recovery (Hungarian-matched cross-entropy)")
        axes[1].plot(rounds, [float(r["belief_entropy_mean"]) for r in met_rows], color="C1")
        axes[1].set_ylabel("Mean belief entropy")
        axes[1].set_xlabel("round")
        fig.tight_layout()
        fig.savefig(run_dir / "plot_recovery_metrics.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(rounds, [float(r["batch_log_likelihood"]) for r in met_rows], color="C2", alpha=0.7)
        ax.set_xlabel("round")
        ax.set_ylabel("Batch log-likelihood")
        fig.tight_layout()
        fig.savefig(run_dir / "plot_log_likelihood.png", dpi=150)
        plt.close(fig)
