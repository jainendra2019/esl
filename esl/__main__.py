"""python -m esl [--adapt] runs a short demo and writes plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from esl.config import ESLConfig
from esl.plotting import plot_run
from esl.trainer import run_esl


def main() -> None:
    p = argparse.ArgumentParser(description="ESL demo run")
    p.add_argument("--adapt", action="store_true", help="adaptation mode (logit best response)")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = ESLConfig(
        seed=args.seed,
        mode="adaptation" if args.adapt else "recovery",
        num_rounds=args.rounds,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        symmetric_init=False,
        init_noise=0.05,
    )
    _, _, _, _, rd = run_esl(cfg, run_dir=args.out)
    plot_run(rd)


if __name__ == "__main__":
    main()
