#!/usr/bin/env python3
"""Generate-once RES benchmark suite and write committed JSON under benchmark/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esl.res.benchmark import (
    DEFAULT_GENERATOR_SEED,
    assert_family_constraints,
    default_benchmark_root,
    generate_2x2_suite,
    generate_3x3_suite,
    min_pairwise_feature_distance,
    write_suite,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate fixed RES benchmark suite (run once, then commit)")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_GENERATOR_SEED)
    p.add_argument("--per-family", type=int, default=4)
    p.add_argument("--include-3x3", action="store_true", default=True)
    p.add_argument("--no-3x3", action="store_true")
    args = p.parse_args()
    root = args.root or default_benchmark_root()
    include_3x3 = bool(args.include_3x3) and not bool(args.no_3x3)

    specs = generate_2x2_suite(seed=int(args.seed), per_family=int(args.per_family))
    if include_3x3:
        specs.extend(generate_3x3_suite(seed=int(args.seed) + 3, per_family=int(args.per_family)))

    for spec in specs:
        assert_family_constraints(spec)

    # Clear existing instance JSONs but keep README if present
    if root.exists():
        for path in root.rglob("*.json"):
            if path.name == "README.md":
                continue
            path.unlink()

    man = write_suite(specs, root)
    diversity = min_pairwise_feature_distance(specs)
    audit = {
        "n_total": len(specs),
        "n_2x2": sum(1 for s in specs if s.num_actions == 2),
        "n_3x3": sum(1 for s in specs if s.num_actions == 3),
        "diversity_min_pairwise": diversity,
        "manifest": str(man),
    }
    (root / "diversity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
