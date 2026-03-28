"""
Two-panel mechanism figure: fast belief entropy vs slow prototype separation (main only).

Reads:
  {main_dir}/belief_metrics.csv  (avg_belief_entropy per round)
  {main_dir}/prototype_trajectory.csv + summary_metrics.json (separation via final Hungarian)

Optional: vertical dashed line at first round with avg belief entropy < 0.5.

Usage:
  python3 -m esl.plot_esl_mechanism --main-dir runs/two_type_sep_mechanism_obs05/main
  python3 -m esl.plot_esl_mechanism --main-dir .../main --out runs/paper_figures/figure_mechanism_obs05.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from esl.plot_observability_separation import separation_curve_from_main_dir

DEFAULT_ENTROPY_VLINE_THRESHOLD = 0.5


def _read_belief_entropy_series(main_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(main_dir) / "belief_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty or missing: {path}")
    rows.sort(key=lambda r: int(r["round"]))
    t = np.array([int(r["round"]) for r in rows], dtype=np.int64)
    h = np.array([float(r["avg_belief_entropy"]) for r in rows], dtype=np.float64)
    return t, h


def _first_round_below(entropy: np.ndarray, rounds: np.ndarray, level: float) -> int | None:
    hit = np.where(entropy < level)[0]
    return int(rounds[hit[0]]) if len(hit) else None


def plot_esl_mechanism_figure(
    main_dir: Path,
    out_path: Path | None = None,
    *,
    entropy_vline_threshold: float | None = DEFAULT_ENTROPY_VLINE_THRESHOLD,
) -> Path:
    """
    Save figure_mechanism.png under main_dir unless out_path is set.
    """
    main_dir = Path(main_dir).resolve()
    out_path = Path(out_path) if out_path is not None else main_dir / "figure_mechanism.png"

    t_h, h = _read_belief_entropy_series(main_dir)
    t_s, sep, p_obs = separation_curve_from_main_dir(main_dir)
    t_end = int(max(t_h.max(), t_s.max()))

    thr: float | None = None
    t_star: int | None = None
    if entropy_vline_threshold is not None:
        thr = float(entropy_vline_threshold)
        t_star = _first_round_below(h, t_h, thr)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, layout="constrained")
    ax0, ax1 = axes

    if t_star is not None:
        for ax in axes:
            ax.axvspan(
                t_star,
                t_end,
                facecolor="0.45",
                alpha=0.1,
                zorder=0,
                linewidth=0,
            )

    ax0.plot(t_h, h, color="C0", linewidth=1.5, zorder=2)
    ax0.set_ylabel("mean belief entropy (nats)", fontsize=10)
    ax0.set_title(
        "Mechanism of Epistemic Social Learning: fast belief inference drives slow prototype learning",
        fontsize=10,
    )
    ax0.grid(True, alpha=0.3, zorder=1)
    ax0.set_ylim(bottom=0.0)
    ax0.set_xlim(0, t_end)

    ax1.plot(t_s, sep, color="C1", linewidth=1.5, zorder=2)
    ax1.set_ylabel(
        r"$\Delta(t) = \left| P_{\mathrm{AC}}(C) - P_{\mathrm{AD}}(C) \right|$",
        fontsize=10,
    )
    ax1.set_xlabel("round $t$")
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, t_end)

    if t_star is not None and thr is not None:
        for ax in axes:
            ax.axvline(
                t_star,
                color="0.35",
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
                zorder=3,
            )
            ax.annotate(
                "beliefs become\ninformative",
                xy=(t_star, 0.52),
                xycoords=("data", "axes fraction"),
                xytext=(9, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.8,
                color="0.28",
                zorder=4,
            )
        ax0.text(
            0.02,
            0.97,
            rf"$\bar{{H}}(B_t) < {thr:g}$ first at $t^* = {t_star}$",
            transform=ax0.transAxes,
            fontsize=7.5,
            verticalalignment="top",
            color="0.35",
        )
        ax1.text(
            (t_star + t_end) / 2,
            0.93,
            "learning phase",
            ha="center",
            va="top",
            fontsize=8,
            color="0.4",
            zorder=4,
        )

    fig.suptitle(
        rf"$p_{{\mathrm{{obs}}}}={p_obs:g}$, main condition (matched labels at final $\theta$)",
        fontsize=9,
        y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="ESL mechanism: entropy + prototype separation (2 panels)")
    p.add_argument(
        "--main-dir",
        type=Path,
        required=True,
        help="Path to experiment **main** subdirectory (contains belief_metrics.csv)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: {main-dir}/figure_mechanism.png)",
    )
    p.add_argument(
        "--entropy-vline",
        type=float,
        default=DEFAULT_ENTROPY_VLINE_THRESHOLD,
        help="Draw vertical line at first round with avg entropy below this (nats). "
        "Use negative to disable.",
    )
    args = p.parse_args()

    ev = None if args.entropy_vline < 0 else float(args.entropy_vline)
    path = plot_esl_mechanism_figure(args.main_dir, args.out, entropy_vline_threshold=ev)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
