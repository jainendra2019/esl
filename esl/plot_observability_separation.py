"""
Overlay separation(t) = |P(C)_AC-match − P(C)_AD-match| for several p_obs runs.

Uses each run's final θ for a fixed Hungarian label map (same convention as experiment figures).
Prototype logits are piecewise constant between SGD steps, so we forward-fill from
main/prototype_trajectory.csv (sparse update rows + init row).

Usage:
  python3 -m esl.plot_observability_separation \\
    --out runs/figure_separation_vs_obs.png \\
    --runs runs/two_type_sep_main runs/two_type_sep_pobs05_T2000 runs/two_type_sep_obs02_long \\
    --labels '$p_{\\mathrm{obs}}=1.0$' '$p_{\\mathrm{obs}}=0.5$' '$p_{\\mathrm{obs}}=0.2$'
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from esl.metrics import match_prototypes_to_types

_TRUE = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

# First round index where separation(t) reaches this level (convergence-time diagnostic).
DEFAULT_SEPARATION_THRESHOLD = 0.5


def _first_round_at_or_above(sep: np.ndarray, level: float) -> int | None:
    hit = np.where(sep >= level)[0]
    return int(hit[0]) if len(hit) else None


def _read_traj_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def separation_curve_from_main_dir(main_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (rounds 0..T-1, separation(t), p_obs) for the main condition in this folder.
    """
    main_dir = Path(main_dir)
    summary = json.loads((main_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    T = int(summary["num_rounds"])
    p_obs = float(summary["obs_prob"])

    rows = _read_traj_rows(main_dir / "prototype_trajectory.csv")
    if not rows:
        raise ValueError(f"empty trajectory: {main_dir / 'prototype_trajectory.csv'}")

    rows.sort(key=lambda d: int(d["round"]))
    last = rows[-1]
    logits = np.array(
        [
            [float(last["theta_0_a0"]), float(last["theta_0_a1"])],
            [float(last["theta_1_a0"]), float(last["theta_1_a1"])],
        ],
        dtype=np.float64,
    )
    perm, _ = match_prototypes_to_types(_TRUE, logits)
    k_ac, k_ad = int(perm[0]), int(perm[1])

    sep_at_marker: list[float] = []
    for r in rows:
        pac = float(r[f"p_{k_ac}_cooperate"])
        pad = float(r[f"p_{k_ad}_cooperate"])
        sep_at_marker.append(abs(pac - pad))

    t_axis = np.arange(T, dtype=np.int64)
    sep = np.zeros(T, dtype=np.float64)
    j = 0
    for t in range(T):
        while j + 1 < len(rows) and int(rows[j + 1]["round"]) <= t:
            j += 1
        sep[t] = sep_at_marker[j]
    return t_axis, sep, p_obs


def plot_observability_separation_summary(
    main_dirs: list[Path],
    labels: list[str],
    out_path: Path,
    *,
    title: str | None = None,
    separation_threshold: float = DEFAULT_SEPARATION_THRESHOLD,
) -> None:
    if len(main_dirs) != len(labels):
        raise ValueError("main_dirs and labels must have same length")
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    thr = float(separation_threshold)
    ax.axhline(
        thr,
        color="0.45",
        linestyle="--",
        linewidth=1.25,
        zorder=0,
        label=rf"Separation $= {thr:g}$ (convergence threshold)",
    )
    for idx, (d, lab) in enumerate(zip(main_dirs, labels)):
        t, s, _p = separation_curve_from_main_dir(d)
        (line,) = ax.plot(t, s, label=lab, linewidth=1.8, zorder=2)
        t_hit = _first_round_at_or_above(s, thr)
        if t_hit is not None:
            y_hit = float(s[t_hit])
            c = line.get_color()
            ax.scatter(
                [t_hit],
                [y_hit],
                color=c,
                s=52,
                zorder=5,
                edgecolors="0.15",
                linewidths=0.6,
            )
            ax.annotate(
                rf"$T_{{{thr:g}}}={t_hit}$",
                xy=(t_hit, y_hit),
                xytext=(6, 10 + idx * 16),
                textcoords="offset points",
                fontsize=9,
                color=c,
            )
    ax.set_xlabel("round $t$")
    ax.set_ylabel(
        r"Separation $\Delta(t) = \left| P_{\mathrm{AC}}(C) - P_{\mathrm{AD}}(C) \right|$"
    )
    ax.set_title(
        title
        or "Effect of observability on convergence speed (main, matched labels at final $\\theta$)"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay separation(t) for several p_obs experiment runs")
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to each experiment's **main** subdirectory (…/main), in desired plot order",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (same length as --runs). Default: infer from summary obs_prob",
    )
    p.add_argument("--out", type=Path, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SEPARATION_THRESHOLD,
        help="Horizontal reference and T_h first-hit level (default: 0.5)",
    )
    args = p.parse_args()

    dirs = [Path(x).resolve() for x in args.runs]
    if args.labels is not None:
        if len(args.labels) != len(dirs):
            raise SystemExit("--labels length must match --runs")
        labels = args.labels
    else:
        labels = []
        for d in dirs:
            summ = json.loads((d / "summary_metrics.json").read_text(encoding="utf-8"))
            po = float(summ["obs_prob"])
            labels.append(f"$p_{{\\mathrm{{obs}}}}={po:g}$")

    plot_observability_separation_summary(
        dirs,
        labels,
        args.out,
        title=args.title,
        separation_threshold=args.threshold,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
