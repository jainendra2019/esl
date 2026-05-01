"""CLI: python -m esl.baselines run-all | plot"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from esl.baselines.evaluate import write_baselines_summary
from esl.baselines.plot import plot_from_run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ESL offline baselines on interaction_observations.csv")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_all = sub.add_parser("run-all", help="Compute baselines and write baselines/*.json")
    run_all.add_argument("--run-dir", type=Path, required=True, help="ESL run directory with config + CSV")
    run_all.add_argument(
        "--prefix-max-round",
        type=int,
        default=None,
        help="Protocol P: fit baselines on rounds ≤ R; ESL bar uses θ at end of round R",
    )
    run_all.add_argument("--kmeans-seed", type=int, default=0)
    run_all.add_argument("--fcm-seed", type=int, default=0)
    run_all.add_argument("--fcm-m", type=float, default=2.0)
    run_all.add_argument("--em-seed", type=int, default=0)
    run_all.add_argument("--em-restarts", type=int, default=8)

    plot_p = sub.add_parser("plot", help="Bar chart from a baselines summary JSON")
    plot_p.add_argument("--run-dir", type=Path, required=True)
    plot_p.add_argument(
        "--summary",
        type=str,
        default="baselines_summary.json",
        help="Filename under RUN_DIR/baselines/ (e.g. baselines_summary_prefix_round_400.json)",
    )
    plot_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG path (default: RUN_DIR/baselines/mce_comparison.png)",
    )
    plot_p.add_argument("--title", type=str, default=None)
    plot_p.add_argument(
        "--include-oracle",
        action="store_true",
        help="Add Oracle bar (often redundant when EM ≈ 0)",
    )
    plot_p.add_argument(
        "--annotation",
        choices=("none", "minimal", "full"),
        default="minimal",
    )
    plot_p.add_argument(
        "--run-all-first",
        action="store_true",
        help="Run full-log baselines first if summary JSON is missing",
    )
    plot_p.add_argument("--prefix-max-round", type=int, default=None)
    plot_p.add_argument("--kmeans-seed", type=int, default=0)
    plot_p.add_argument("--fcm-seed", type=int, default=0)
    plot_p.add_argument("--fcm-m", type=float, default=2.0)
    plot_p.add_argument("--em-seed", type=int, default=0)
    plot_p.add_argument("--em-restarts", type=int, default=8)

    args = p.parse_args(argv)
    if args.cmd == "run-all":
        path = write_baselines_summary(
            args.run_dir,
            kmeans_seed=args.kmeans_seed,
            fcm_seed=args.fcm_seed,
            fcm_m=args.fcm_m,
            em_seed=args.em_seed,
            em_restarts=args.em_restarts,
            prefix_max_round=args.prefix_max_round,
        )
        print(path)
        return 0
    if args.cmd == "plot":
        summ_path = args.run_dir / "baselines" / args.summary
        if args.run_all_first and not summ_path.is_file():
            write_baselines_summary(
                args.run_dir,
                kmeans_seed=args.kmeans_seed,
                fcm_seed=args.fcm_seed,
                fcm_m=args.fcm_m,
                em_seed=args.em_seed,
                em_restarts=args.em_restarts,
                prefix_max_round=args.prefix_max_round,
            )
        out_default = args.run_dir / "baselines" / "mce_comparison.png"
        if args.summary != "baselines_summary.json":
            stem = Path(args.summary).stem
            out_default = args.run_dir / "baselines" / f"{stem}.png"
        out = plot_from_run_dir(
            args.run_dir,
            args.out or out_default,
            title=args.title,
            summary_filename=args.summary,
            include_oracle=args.include_oracle,
            annotation=args.annotation,  # type: ignore[arg-type]
        )
        print(out)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
