#!/usr/bin/env python3
"""Bit-exact freeze check: regenerate 3×3 matrices in memory; never write benchmark/."""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esl.res.benchmark import (  # noqa: E402
    generate_3x3_suite,
    load_suite,
)
from esl.res.game_spec import GameSpec  # noqa: E402

EXPECTED_IDS = (
    "coop3_01",
    "coop3_02",
    "coop3_03",
    "coop3_04",
    "coord3_01",
    "coord3_02",
    "coord3_03",
    "coord3_04",
    "adv3_01",
    "adv3_02",
    "adv3_03",
    "adv3_04",
)
THREE_X_THREE_SEED = 20260728
SHA256SUMS = ROOT / "reproducibility" / "SHA256SUMS"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sums(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        if rel.startswith("*"):
            rel = rel[1:]
        rows.append((digest, rel))
    return rows


def verify_hashes() -> None:
    missing = []
    mismatches = []
    for digest, rel in _parse_sums(SHA256SUMS):
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        got = _sha256(path)
        if got != digest:
            mismatches.append((rel, digest, got))
    if missing or mismatches:
        raise AssertionError(f"SHA-256 failures missing={missing} mismatches={mismatches}")


def verify_regeneration() -> None:
    frozen = {s.instance_id: s for s in load_suite(ROOT / "benchmark", num_actions=3)}
    if tuple(frozen) != EXPECTED_IDS:
        # load_suite order is family then sorted filename; IDs must still match as a set/order.
        got = tuple(frozen)
        if set(got) != set(EXPECTED_IDS) or len(got) != 12:
            raise AssertionError(f"unexpected 3x3 IDs: {got}")
        if got != EXPECTED_IDS:
            raise AssertionError(f"3x3 ID order {got} != {EXPECTED_IDS}")

    regen = generate_3x3_suite(seed=THREE_X_THREE_SEED, per_family=4)
    regen_ids = tuple(s.instance_id for s in regen)
    if regen_ids != EXPECTED_IDS:
        raise AssertionError(f"regenerated IDs {regen_ids} != {EXPECTED_IDS}")

    for spec in regen:
        disk = frozen[spec.instance_id]
        row_a = np.asarray(disk.payoff_row, dtype=np.float64)
        row_b = np.asarray(spec.payoff_row, dtype=np.float64)
        col_a = np.asarray(disk.payoff_col, dtype=np.float64)
        col_b = np.asarray(spec.payoff_col, dtype=np.float64)
        if not np.array_equal(row_a, row_b) or not np.array_equal(col_a, col_b):
            row_err = float(np.max(np.abs(row_a - row_b)))
            col_err = float(np.max(np.abs(col_a - col_b)))
            raise AssertionError(
                f"payoff mismatch for {spec.instance_id}: "
                f"max_abs_row={row_err} max_abs_col={col_err}"
            )
        if int(disk.generator_seed) != THREE_X_THREE_SEED:
            raise AssertionError(f"{spec.instance_id} generator_seed={disk.generator_seed}")

        json_path = ROOT / "benchmark" / disk.family / f"{disk.instance_id}.json"
        loaded = GameSpec.load_json(json_path)
        if loaded.instance_id != spec.instance_id:
            raise AssertionError(f"JSON id mismatch {json_path}")
        if not np.array_equal(
            np.asarray(loaded.payoff_row, dtype=np.float64), row_b
        ) or not np.array_equal(np.asarray(loaded.payoff_col, dtype=np.float64), col_b):
            raise AssertionError(f"JSON load mismatch {json_path}")


def verify_loader_does_not_resample() -> None:
    src = inspect.getsource(load_suite)
    forbidden = ("generate_3x3_suite", "generate_2x2_suite", "default_rng", "_sample_3x3")
    for token in forbidden:
        if token in src:
            raise AssertionError(f"load_suite source contains {token!r}; resampling risk")
    if "GameSpec.load_json" not in src and "load_json" not in src:
        raise AssertionError("load_suite does not load JSON files")
    if "glob" not in src:
        raise AssertionError("load_suite does not iterate committed JSON files")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import platform

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--allow-regen-mismatch",
        action="store_true",
        help="Treat in-memory regeneration mismatch as a documented portability warning. "
        "Committed JSON hashes still must match. Canonical matrices are the JSON files.",
    )
    args = p.parse_args(argv)
    before = {rel: _sha256(ROOT / rel) for _, rel in _parse_sums(SHA256SUMS)}
    verify_hashes()
    verify_loader_does_not_resample()
    regen_ok = True
    try:
        verify_regeneration()
    except AssertionError as exc:
        regen_ok = False
        print(f"REGEN_MISMATCH: {exc}")
        print(
            f"platform={platform.platform()} python={sys.version.split()[0]} numpy={np.__version__}"
        )
        print(
            "Canonical source: committed benchmark/*.json (SHA-256). "
            "In-memory generate_3x3_suite(seed=20260728) is verified bit-exact on "
            "macOS / NumPy 2.2.6; other platforms may differ."
        )
        if not args.allow_regen_mismatch:
            raise
    after = {rel: _sha256(ROOT / rel) for _, rel in _parse_sums(SHA256SUMS)}
    if before != after:
        raise AssertionError("benchmark freeze files changed during verification")
    if regen_ok:
        print("PASS: twelve 3x3 matrices bit-exact; SHA-256 ok; loader is file-backed")
    else:
        print("PASS: SHA-256 of committed matrices ok; regeneration mismatch recorded")
    print(f"numpy={np.__version__} seed={THREE_X_THREE_SEED} n={len(EXPECTED_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
