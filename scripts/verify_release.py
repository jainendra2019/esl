#!/usr/bin/env python3
"""Quick camera-ready verification: benchmark integrity, smoke cell, frozen numbers.

Default (portable): committed JSON hashes, load of the twelve games, evidence
hashes, accepted numerical checks, and a 40-step ESL smoke cell.

Pass ``--regen`` to additionally run the strict in-memory generator comparison
(known platform-dependent). Integrity failures always fail this script.

Inputs: committed benchmark/, evidence/, and the installed esl package.
Outputs: PASS/FAIL on stdout. Does not write files.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLAGSHIP_SHA256 = "243c8ac5fe8c37132f75f9f2a61f84aa1d8b55b6ff6748901e0729ef7de740e9"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    regen = "--regen" in args
    print("== benchmark freeze (hashes, load, no resample) ==")
    freeze = [sys.executable, str(ROOT / "reproducibility" / "verify_benchmark_freeze.py")]
    _run(freeze)

    if regen:
        print("== in-memory generator regeneration (strict, platform-dependent) ==")
        _run(freeze + ["--regen"])

    print("== flagship evidence hash ==")
    cells = ROOT / "evidence" / "flagship" / "cells.csv"
    digest = hashlib.sha256(cells.read_bytes()).hexdigest()
    if digest != FLAGSHIP_SHA256:
        raise SystemExit(f"flagship cells hash mismatch: {digest}")
    print(f"PASS cells.csv SHA-256 {digest}")

    print("== accepted numerical checks ==")
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_release_numbers.py",
            "tests/test_res_benchmark.py",
        ]
    )

    print("== tiny ESL smoke cell ==")
    from esl.res.benchmark import load_suite
    from esl.res.categorical_cell import run_categorical_cell
    from esl.res.runner import game_spec_to_task

    spec = next(s for s in load_suite(num_actions=3) if s.instance_id == "coop3_01")
    out = run_categorical_cell(
        task=game_spec_to_task(spec),
        method="ESL",
        seed=0,
        horizon=40,
        num_actions=3,
    )
    psr = float(out["summary"]["psr"])
    if not (psr == psr):  # NaN
        raise SystemExit("smoke PSR is NaN")
    print(f"PASS smoke ESL coop3_01 seed=0 horizon=40 PSR={psr:.4f}")
    print("verify_release: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
