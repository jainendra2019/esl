#!/usr/bin/env python3
"""Run focal-λ sensitivity for ESL on the clean 3×3 adaptation suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from esl.res.focal_lambda_sensitivity import FOCAL_LAMBDA_GRID, run_grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/lambda_sensitivity"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--horizon", type=int, default=1000)
    args = p.parse_args()
    report = run_grid(
        out_dir=args.out,
        lambdas=FOCAL_LAMBDA_GRID,
        horizon=int(args.horizon),
        jobs=int(args.jobs),
        smoke=bool(args.smoke),
    )
    print(
        f"Wrote {report['n_cells']} cells → {args.out}; "
        f"dpr_invariant={report['dpr_approximately_invariant']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
