"""Ablation ladder: full ESL vs freeze_prototype_parameters vs learning_frozen (matched geometry)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

from esl.config import ESLConfig
from esl.experiments.manifest import write_run_manifest
from esl.trainer import run_esl


def _ablation_base_cfg(*, seed: int, num_rounds: int) -> ESLConfig:
    """Compact recovery geometry for reproducible ablations."""
    return ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=6,
        num_prototypes=2,
        num_rounds=num_rounds,
        force_agent_true_types=[0, 0, 0, 1, 1, 1],
        observability="full",
        p_obs=1.0,
        prototype_update_every=3,
        prototype_lr_scale=18.0,
        init_noise=0.05,
        symmetric_init=False,
        log_beliefs_tensor=True,
        log_beliefs_every_interaction=False,
        delta_simplex=1e-4,
        learning_frozen=False,
        freeze_prototype_parameters=False,
    )


def run_ablation_ladder(
    *,
    out_root: Path,
    num_rounds: int,
    seed: int = 42,
    plot: bool = True,
) -> tuple[Path, Path | None]:
    """
    Runs three conditions under ``out_root/<condition>/seed_<seed>/``.

    Writes ``out_root/ablation_ladder_summary.csv`` and ``ablation_mce_belief.png``.
    """
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    base = _ablation_base_cfg(seed=seed, num_rounds=num_rounds)
    variants: list[tuple[str, ESLConfig]] = [
        ("full", base),
        (
            "freeze_prototype",
            replace(base, freeze_prototype_parameters=True),
        ),
        (
            "learning_frozen",
            replace(base, learning_frozen=True),
        ),
    ]

    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    mces: list[float] = []
    beliefs: list[float] = []
    payoffs: list[float] = []

    for name, cfg in variants:
        cfg.validate()
        run_dir = out_root / name / f"seed_{seed}"
        write_run_manifest(
            run_dir / "run_manifest.json",
            {"experiment": "ablation_ladder", "variant": name, "seed": seed},
        )
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
        mce = float(summary.get("final_mce", 0.0))
        ba = float(summary.get("final_belief_argmax_accuracy", 0.0))
        pay = float(summary.get("mean_payoff_per_agent_per_round", 0.0))
        rows.append(
            {
                "variant": name,
                "run_dir": str(run_dir),
                "final_mce": mce,
                "final_belief_argmax_accuracy": ba,
                "mean_payoff_per_agent_per_round": pay,
                "num_interaction_events_executed": summary.get(
                    "num_interaction_events_executed", ""
                ),
            }
        )
        labels.append(name.replace("_", "\n"))
        mces.append(mce)
        beliefs.append(ba)
        payoffs.append(pay)

    csv_path = out_root / "ablation_ladder_summary.csv"
    fields = [
        "variant",
        "run_dir",
        "final_mce",
        "final_belief_argmax_accuracy",
        "mean_payoff_per_agent_per_round",
        "num_interaction_events_executed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = out_root / "ablation_mce_belief.png"
        _plot_ablation_bars(labels, mces, beliefs, png_path)

    return csv_path, png_path


def _plot_ablation_bars(
    labels: list[str],
    mces: list[float],
    beliefs: list[float],
    out_png: Path,
) -> None:
    x = np.arange(len(labels))
    w = 0.35
    fig, ax0 = plt.subplots(figsize=(6, 4))
    ax0.bar(x - w / 2, mces, width=w, label="Final MCE", color="C0")
    ax0.set_ylabel("MCE")
    ax0.set_xticks(x, labels, fontsize=8)
    ax0.legend(loc="upper left")
    ax0.grid(axis="y", alpha=0.3)

    ax1 = ax0.twinx()
    ax1.bar(x + w / 2, beliefs, width=w, label="Belief argmax acc.", color="C1", alpha=0.85)
    ax1.set_ylabel("Belief argmax accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right")

    ax0.set_title("ESL ablations (matched recovery geometry)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ablation ladder CSV + figure")
    p.add_argument("--out-root", type=Path, default=Path("runs/ablation_ladder"))
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    cpath, ppath = run_ablation_ladder(
        out_root=args.out_root,
        num_rounds=args.rounds,
        seed=args.seed,
        plot=not args.no_plot,
    )
    print(cpath)
    if ppath:
        print(ppath)


if __name__ == "__main__":
    main()
