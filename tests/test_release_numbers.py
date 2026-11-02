"""Recompute accepted manuscript numbers from the frozen evidence package."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"

FLAGSHIP_SHA256 = "243c8ac5fe8c37132f75f9f2a61f84aa1d8b55b6ff6748901e0729ef7de740e9"
FOUR_METHODS = ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM")


def _load(rel: str) -> list[dict[str, str]]:
    with (EVIDENCE / rel).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key, "") not in ("", "N/A")]
    return sum(vals) / len(vals)


def test_flagship_cells_hash():
    import hashlib

    digest = hashlib.sha256((EVIDENCE / "flagship" / "cells.csv").read_bytes()).hexdigest()
    assert digest == FLAGSHIP_SHA256


def test_flagship_esl_psr_dpr_lte():
    esl = [r for r in _load("flagship/cells.csv") if r["method"] == "ESL"]
    assert len(esl) == 120
    assert round(_mean(esl, "psr"), 2) == 162.97
    assert round(_mean(esl, "dpr"), 2) == 106.67
    assert round(_mean(esl, "lte_auc"), 6) == pytest.approx(3.225421, abs=5e-7)
    assert f"{_mean(esl, 'lte_auc'):.2f}" == "3.23"


def test_esl_lowest_psr_counts():
    rows = _load("flagship/cells.csv")
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by[(r["instance_id"], r["method"])].append(float(r["psr"]))
    games = sorted({r["instance_id"] for r in rows})
    lowest_four = 0
    lt_safp = 0
    for g in games:
        means = {m: sum(by[(g, m)]) / len(by[(g, m)]) for m in FOUR_METHODS}
        if means["ESL"] == min(means.values()):
            lowest_four += 1
        if means["ESL"] < means["SA-FP"]:
            lt_safp += 1
    assert lowest_four == 8
    assert lt_safp == 11


def test_k_sensitivity_means():
    rows = _load("k_sensitivity/summary.csv")
    got = {(int(r["k_model"]), r["metric"]): round(float(r["mean"]), 2) for r in rows}
    assert got[(2, "psr")] == 175.91
    assert got[(2, "dpr")] == 120.19
    assert got[(3, "psr")] == 162.97
    assert got[(3, "dpr")] == 106.67
    assert got[(5, "psr")] == 161.65
    assert got[(5, "dpr")] == 98.96


def test_k3_matches_flagship_cells():
    flag = {
        (r["instance_id"], int(r["seed"])): (float(r["psr"]), float(r["dpr"]), float(r["lte_auc"]))
        for r in _load("flagship/cells.csv")
        if r["method"] == "ESL"
    }
    sweep = [
        r
        for r in _load("k_sensitivity/k_sweep_cells.csv")
        if int(float(r["k_model"])) == 3
    ]
    assert len(sweep) == 120
    for r in sweep:
        key = (r["instance_id"], int(r["seed"]))
        psr, dpr, lte = flag[key]
        assert float(r["psr"]) == pytest.approx(psr, abs=0.0, rel=0.0)
        assert float(r["dpr"]) == pytest.approx(dpr, abs=0.0, rel=0.0)
        assert float(r["lte_auc"]) == pytest.approx(lte, abs=0.0, rel=0.0)


def test_psrv_recovery_accuracy():
    n20 = _load("psrv/n20_cells.csv")
    n100 = _load("psrv/n100_cells.csv")
    assert len(n20) == 120
    assert len(n100) == 120
    acc20 = _mean(n20, "recovery_accuracy")
    acc100 = _mean(n100, "recovery_accuracy")
    assert round(acc20, 3) == 0.955
    assert round(acc100, 3) == 0.847


def test_omd_held_out_means():
    import json

    summary = _load("omd/omd_psr_summary.csv")
    primary = [r for r in summary if r["layer"] == "primary_held_out"]
    by = {r["method"]: r for r in primary}
    assert round(float(by["ESL-decision-tuned"]["psr_mean"]), 2) == 86.73
    assert round(float(by["Per-opponent OMD/MWU"]["psr_mean"]), 2) == 116.67
    paired = json.loads((EVIDENCE / "omd" / "omd_vs_esl_paired.json").read_text(encoding="utf-8"))
    d = paired["primary_held_out_seeds_3_9"]["per_opponent_minus_esl_star"]
    assert d["n"] == 84
    assert round(float(d["mean_delta"]), 2) == 29.94
    assert round(float(d["ci95_lo"]), 2) == 13.79
    assert round(float(d["ci95_hi"]), 2) == 48.07
    assert float(paired["tuning"]["frozen_lambda"]) == 10.0
    assert float(paired["tuning"]["frozen_eta"]) == 1.0
