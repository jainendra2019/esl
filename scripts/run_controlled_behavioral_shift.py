#!/usr/bin/env python3
"""Run Controlled Behavioral Shift Analysis (A=3 appendix)."""

from __future__ import annotations

import argparse
from pathlib import Path

from esl.res.controlled_behavioral_shift import DEFAULT_SEEDS, ROTATIONS, run_grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/mechanistic_analysis_a3"),
        help="Output directory",
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="Tiny grid for wiring checks")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--rotations", type=int, nargs="*", default=None)
    p.add_argument("--horizon", type=int, default=220)
    args = p.parse_args()

    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    rotations = tuple(args.rotations) if args.rotations is not None else ROTATIONS
    report = run_grid(
        out_dir=args.out,
        seeds=seeds,
        rotations=rotations,
        horizon=int(args.horizon),
        jobs=int(args.jobs),
        smoke=bool(args.smoke),
    )
    print(
        f"Wrote {report['n_cells']} cells → {args.out} "
        f"(esl_uniformly_best={report['esl_uniformly_best']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
