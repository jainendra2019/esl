#!/usr/bin/env python3
"""Regenerate baseline comparison figures under docs/baselines/.

Writes **two** PNGs:
  1. ``baseline_mce_comparison.png`` — **Protocol P** (prefix-matched budget): offline methods
     refit on rounds ≤ R; ESL bar is θ at end of round R (fairer for main-paper messaging).
  2. ``baseline_transductive_diagnostic.png`` — **Protocol T** (full log): transductive batch
     refits vs final online θ (appendix / sanity: separability of φ on the full log).

Default training run: N=12, multi-pair rounds, lr_scale=22 (~2200 env rounds). Use ``--quick`` for smoke only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esl.baselines.evaluate import write_baselines_summary
from esl.baselines.plot import plot_from_run_dir
from esl.config import ESLConfig
from esl.trainer import run_esl


def _cfg_paper_style(*, seed: int, num_rounds: int) -> ESLConfig:
    types = [0] * 6 + [1] * 6
    return ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=12,
        num_prototypes=2,
        num_rounds=num_rounds,
        force_agent_true_types=types,
        delta_simplex=0.02,
        bayes_denominator_eps=1e-12,
        base_init=0.0,
        init_noise=0.05,
        symmetric_init=False,
        prototype_lr_scale=22.0,
        lr_prototype_gamma_exponent=-0.9,
        prototype_update_every=10,
        prototype_l2_eta=0.0,
        interaction_pairs_min=3,
        interaction_pairs_max=6,
        interaction_pairs_law="uniform",
        observability="full",
        p_obs=1.0,
        log_beliefs_tensor=False,
        log_beliefs_every_interaction=False,
        log_interaction_observations=True,
        learning_frozen=False,
    )


def _cfg_quick(*, seed: int, num_rounds: int) -> ESLConfig:
    return ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=4,
        num_prototypes=2,
        num_rounds=num_rounds,
        observability="full",
        prototype_update_every=3,
        init_noise=0.06,
        symmetric_init=False,
        log_interaction_observations=True,
        log_beliefs_tensor=False,
        force_agent_true_types=[0, 0, 1, 1],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("runs/baseline_mce_demo"))
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("docs/baselines/baseline_mce_comparison.png"),
        help="Prefix-matched (Protocol P) figure",
    )
    p.add_argument(
        "--out-transductive",
        type=Path,
        default=Path("docs/baselines/baseline_transductive_diagnostic.png"),
        help="Full-log transductive (Protocol T) diagnostic figure",
    )
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--prefix-fraction",
        type=float,
        default=0.28,
        help="R = floor(fraction * num_rounds_executed), min 1",
    )
    p.add_argument("--em-restarts", type=int, default=8)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        rounds = 120 if args.rounds is None else args.rounds
        cfg = _cfg_quick(seed=args.seed, num_rounds=rounds)
    else:
        rounds = 2200 if args.rounds is None else args.rounds
        cfg = _cfg_paper_style(seed=args.seed, num_rounds=rounds)
    cfg.validate()
    run_esl(cfg, run_dir=run_dir)

    write_baselines_summary(run_dir, em_restarts=args.em_restarts)
    args.out_transductive.resolve().parent.mkdir(parents=True, exist_ok=True)
    plot_from_run_dir(
        run_dir,
        args.out_transductive.resolve(),
        summary_filename="baselines_summary.json",
        title=None,
        include_oracle=False,
        annotation="minimal",
    )

    sm = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    t_exec = int(sm.get("num_rounds_executed", 0))
    pref = max(1, int(args.prefix_fraction * t_exec))
    write_baselines_summary(
        run_dir, em_restarts=args.em_restarts, prefix_max_round=pref
    )
    summ_name = f"baselines_summary_prefix_round_{pref}.json"
    args.out_prefix.resolve().parent.mkdir(parents=True, exist_ok=True)
    plot_from_run_dir(
        run_dir,
        args.out_prefix.resolve(),
        summary_filename=summ_name,
        title=None,
        include_oracle=False,
        annotation="minimal",
    )
    print(args.out_prefix.resolve())
    print(args.out_transductive.resolve())


if __name__ == "__main__":
    main()
