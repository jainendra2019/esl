"""Adaptation mode: mean payoff vs adaptation_lambda (ESL-focused; no batch payoff baseline)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from esl.config import ESLConfig
from esl.experiments.manifest import write_run_manifest
from esl.trainer import run_esl

# Logit best-response temperature grid (interpretable ablation).
ADAPTATION_LAMBDA_GRID: tuple[float, ...] = (1.0, 2.0, 4.0)


def _adaptation_base(*, seed: int, num_rounds: int, lam: float) -> ESLConfig:
    return ESLConfig(
        seed=seed,
        mode="adaptation",
        num_agents=6,
        num_prototypes=2,
        num_rounds=num_rounds,
        force_agent_true_types=[0, 0, 0, 1, 1, 1],
        observability="full",
        p_obs=1.0,
        prototype_update_every=4,
        prototype_lr_scale=12.0,
        init_noise=0.08,
        symmetric_init=False,
        adaptation_lambda=float(lam),
        interaction_pairs_min=1,
        interaction_pairs_max=2,
        log_beliefs_tensor=False,
    )


def run_adaptation_payoff_sweep(
    *,
    out_root: Path,
    num_rounds: int,
    seed: int = 42,
    plot: bool = True,
) -> tuple[Path, Path | None]:
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for lam in ADAPTATION_LAMBDA_GRID:
        cfg = _adaptation_base(seed=seed, num_rounds=num_rounds, lam=lam)
        cfg.validate()
        slug = f"lambda_{str(lam).replace('.', 'p')}"
        run_dir = out_root / slug / f"seed_{seed}"
        write_run_manifest(
            run_dir / "run_manifest.json",
            {
                "experiment": "adaptation_payoff_sweep",
                "adaptation_lambda": lam,
                "seed": seed,
            },
        )
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
        rows.append(
            {
                "adaptation_lambda": lam,
                "run_dir": str(run_dir),
                "mean_payoff_per_agent_per_round": summary.get(
                    "mean_payoff_per_agent_per_round", ""
                ),
                "final_mce": summary.get("final_mce", ""),
                "final_belief_argmax_accuracy": summary.get(
                    "final_belief_argmax_accuracy", ""
                ),
            }
        )

    csv_path = out_root / "adaptation_payoff_summary.csv"
    fields = [
        "adaptation_lambda",
        "run_dir",
        "mean_payoff_per_agent_per_round",
        "final_mce",
        "final_belief_argmax_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    png_path: Path | None = None
    if plot:
        png_path = out_root / "adaptation_payoff_vs_lambda.png"
        _plot_payoff_vs_lambda(rows, png_path)

    return csv_path, png_path


def _plot_payoff_vs_lambda(rows: list[dict[str, Any]], out_png: Path) -> None:
    xs = [float(r["adaptation_lambda"]) for r in rows]
    ys = [float(r["mean_payoff_per_agent_per_round"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, ys, "o-", color="C0", linewidth=1.5, markersize=8)
    ax.set_xlabel(r"Adaptation $\lambda$ (logit best-response)")
    ax.set_ylabel("Mean payoff / agent / round")
    ax.set_title("Adaptation mode (ESL); batch baselines have no payoff here")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Adaptation payoff vs lambda")
    p.add_argument("--out-root", type=Path, default=Path("runs/adaptation_payoff_sweep"))
    p.add_argument("--rounds", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    cpath, ppath = run_adaptation_payoff_sweep(
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
