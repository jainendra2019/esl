"""Focal-agent payoff curves from reward_trajectory.csv (Milestone 5)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def focal_payoff_per_round_from_run_dir(run_dir: Path, focal: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns ``(rounds, mean_focal_payoff_in_round)`` where the mean is over interactions
    where the focal agent played in that environment round (0 if none).
    """
    path = run_dir / "reward_trajectory.csv"
    if not path.is_file():
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    sums: dict[int, float] = defaultdict(float)
    cnts: dict[int, int] = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["round"])
            i, j = int(row["i"]), int(row["j"])
            if i == focal:
                sums[t] += float(row["r_i"])
                cnts[t] += 1
            if j == focal:
                sums[t] += float(row["r_j"])
                cnts[t] += 1
    if not sums:
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    t_max = max(sums.keys())
    rounds = np.arange(t_max + 1, dtype=int)
    y = np.zeros(t_max + 1, dtype=np.float64)
    for t in range(t_max + 1):
        c = cnts.get(t, 0)
        y[t] = float(sums[t] / c) if c > 0 else float("nan")
    return rounds, y


def focal_cumulative_payoff(run_dir: Path, focal: int = 0) -> float:
    path = run_dir / "reward_trajectory.csv"
    if not path.is_file():
        return 0.0
    total = 0.0
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            i, j = int(row["i"]), int(row["j"])
            if i == focal:
                total += float(row["r_i"])
            if j == focal:
                total += float(row["r_j"])
    return float(total)


def write_focal_sidecar(run_dir: Path, focal: int = 0) -> Path:
    """Writes ``focal_payoff_per_round.csv`` next to the run for downstream plotting."""
    rounds, y = focal_payoff_per_round_from_run_dir(run_dir, focal=focal)
    out = run_dir / "focal_payoff_per_round.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "focal_mean_payoff_in_round", "focal_index"])
        for t, v in zip(rounds.tolist(), y.tolist(), strict=False):
            w.writerow([int(t), v, focal])
    meta: dict[str, Any] = {
        "focal_agent_index": focal,
        "focal_cumulative_payoff": focal_cumulative_payoff(run_dir, focal=focal),
    }
    (run_dir / "focal_metrics.json").write_text(
        __import__("json").dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out
