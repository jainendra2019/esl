"""Bar charts for baseline MCE summaries (transductive diagnostic vs prefix-matched)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

AnnotationStyle = Literal["none", "minimal", "full"]


def load_baselines_summary(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def default_title(summary: dict[str, Any]) -> str:
    proto = summary.get("comparison_protocol", "transductive_full_log")
    if proto == "prefix_matched_budget":
        r = summary.get("prefix_max_round")
        return f"Prefix-matched budget (rounds ≤ {r}): offline refits vs ESL θ at same horizon"
    return "Full observation log: transductive offline refits vs online ESL (diagnostic)"


def _esl_mce_for_display(summary: dict[str, Any]) -> tuple[float | None, str]:
    v = summary.get("esl_mce_recomputed_from_trajectory")
    if v is None:
        v = summary.get("esl_final_mce_from_summary")
    if v is None:
        return None, "ESL"
    if summary.get("comparison_protocol") == "prefix_matched_budget":
        r = summary.get("prefix_max_round")
        return float(v), f"ESL (θ at round {r})"
    return float(v), "ESL (online, final θ)"


def plot_baseline_mce_comparison(
    summary: dict[str, Any],
    out_path: Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.8),
    include_oracle: bool = False,
    annotation: AnnotationStyle = "minimal",
    ylim_floor: float = 0.08,
) -> None:
    """
    ``include_oracle=False`` by default (Oracle ≈ 0 often duplicates EM visually).

    ``ylim_floor``: upper axis limit is at least this value so tiny batch MCE bars do not
    exaggerate the gap vs ESL (paper readability).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    series: list[tuple[str, float, str]] = []
    esl_v, esl_lbl = _esl_mce_for_display(summary)
    if esl_v is not None:
        series.append((esl_lbl, esl_v, "C0"))
    series.extend(
        [
            ("K-means", float(summary["kmeans_mce"]), "C1"),
            ("FCM", float(summary["fcm_mce"]), "C2"),
            ("EM (cond.)", float(summary["em_conditional_mce"]), "C3"),
            ("EM (marg.)", float(summary["em_marginal_mce"]), "C4"),
        ]
    )
    if include_oracle:
        series.append(("Oracle", float(summary["oracle_mce"]), "0.55"))

    labels = [s[0] for s in series]
    values = np.array([s[1] for s in series], dtype=np.float64)
    colors = [s[2] for s in series]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("MCE (matched cross-entropy)")
    ax.set_title(title or default_title(summary))
    raw_max = float(np.nanmax(values)) if len(values) else 0.0
    ymax = max(raw_max * 1.12, ylim_floor, 0.05)
    ax.set_ylim(0.0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    if annotation == "full":
        nr = summary.get("esl_num_rounds_executed")
        ni = summary.get("esl_num_interaction_events_executed")
        match_ok = summary.get("esl_mce_summary_matches_trajectory")
        lines = [
            "Metric: mce_value(true templates, ·); baselines on same w>0 rows as ESL.",
            "Transductive: batch refit on full log. Prefix: same data budget as θ snapshot.",
        ]
        if nr is not None and ni is not None:
            lines.append(f"Full run: T={nr} rounds, {ni} interaction events.")
        if match_ok is False:
            lines.append("WARNING: trajectory MCE ≠ summary final_mce.")
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.35},
        )
    elif annotation == "minimal":
        line = (
            "Same mce_value(·); prefix figures match offline fit to ESL’s data budget."
            if summary.get("comparison_protocol") == "prefix_matched_budget"
            else "Diagnostic: batch methods refit on the full log (transductive)."
        )
        ax.text(
            0.02,
            0.98,
            line,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.28},
        )

    for b, v in zip(bars, values, strict=True):
        if np.isfinite(v):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                min(v + ymax * 0.015, ymax * 0.97),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_from_run_dir(
    run_dir: Path,
    out_path: Path | None = None,
    *,
    title: str | None = None,
    summary_filename: str = "baselines_summary.json",
    include_oracle: bool = False,
    annotation: AnnotationStyle = "minimal",
) -> Path:
    run_dir = Path(run_dir)
    json_path = run_dir / "baselines" / summary_filename
    summary = load_baselines_summary(json_path)
    dest = out_path or (run_dir / "baselines" / "mce_comparison.png")
    plot_baseline_mce_comparison(
        summary,
        dest,
        title=title,
        include_oracle=include_oracle,
        annotation=annotation,
    )
    return Path(dest)
