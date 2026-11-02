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


def _k_cells_by_k() -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in _load("k_sensitivity/k_sweep_cells.csv"):
        by[int(float(r["k_model"]))].append(r)
    return by


def test_k_sensitivity_means():
    by = _k_cells_by_k()
    assert {k: len(v) for k, v in by.items()} == {2: 120, 3: 120, 5: 120}
    means = {
        (k, metric): round(_mean(rows, metric), 2)
        for k, rows in by.items()
        for metric in ("psr", "dpr")
    }
    assert means[(2, "psr")] == 175.91
    assert means[(2, "dpr")] == 120.19
    assert means[(3, "psr")] == 162.97
    assert means[(3, "dpr")] == 106.67
    assert means[(5, "psr")] == 161.65
    assert means[(5, "dpr")] == 98.96


def test_k_sensitivity_paired_holm_recomputed():
    from scipy import stats as sp_stats

    from esl.res.stats import holm_adjust, paired_mean_ci95

    by = _k_cells_by_k()
    tests = []
    for k_alt in (2, 5):
        for metric in ("psr", "dpr"):
            map3 = {(r["instance_id"], int(r["seed"])): float(r[metric]) for r in by[3]}
            mapk = {(r["instance_id"], int(r["seed"])): float(r[metric]) for r in by[k_alt]}
            keys = sorted(set(map3) & set(mapk))
            diffs = [mapk[key] - map3[key] for key in keys]
            st = paired_mean_ci95(diffs)
            _tstat, pval = sp_stats.ttest_rel(
                [mapk[key] for key in keys], [map3[key] for key in keys]
            )
            tests.append(
                {
                    "contrast": f"K{k_alt}-K3",
                    "metric": metric,
                    "mean_diff": st["mean"],
                    "p_unadj": float(pval),
                    "ci_includes_0": bool(st["ci95_low"] <= 0.0 <= st["ci95_high"]),
                }
            )
    for rec, adj in zip(tests, holm_adjust([r["p_unadj"] for r in tests])):
        rec["p_holm"] = adj
        rec["holm_sig_0.05"] = bool(adj < 0.05)
    by_key = {(r["contrast"], r["metric"]): r for r in tests}
    k2p = by_key[("K2-K3", "psr")]
    k2d = by_key[("K2-K3", "dpr")]
    k5p = by_key[("K5-K3", "psr")]
    k5d = by_key[("K5-K3", "dpr")]
    assert round(k2p["mean_diff"], 2) == 12.94
    assert round(k5p["mean_diff"], 2) == -1.32
    assert round(k2d["mean_diff"], 2) == 13.52
    assert round(k5d["mean_diff"], 2) == -7.71
    assert k2p["holm_sig_0.05"] is True
    assert k2d["holm_sig_0.05"] is True
    assert k5p["holm_sig_0.05"] is False
    assert k5d["holm_sig_0.05"] is True
    assert k5p["ci_includes_0"] is True
    assert k2p["p_holm"] == pytest.approx(5.236241553340677e-05, rel=1e-9)
    assert k5p["p_holm"] == pytest.approx(0.4705938367561794, rel=1e-9)


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


def test_omd_held_out_means_recomputed():
    from esl.res.stats import bootstrap_mean_ci95, filter_seeds, paired_psr_diffs

    esl_rows = filter_seeds(_load("omd/esl_lambda_star_cells.csv"), (3, 4, 5, 6, 7, 8, 9))
    omd_rows = filter_seeds(_load("omd/omd_per_opponent_cells.csv"), (3, 4, 5, 6, 7, 8, 9))
    esl = {(str(r["instance_id"]), int(r["seed"])): float(r["psr"]) for r in esl_rows}
    omd = {(str(r["instance_id"]), int(r["seed"])): float(r["psr"]) for r in omd_rows}
    keys = set(esl) & set(omd)
    assert len(keys) == 84
    esl_mean = sum(esl[k] for k in keys) / len(keys)
    omd_mean = sum(omd[k] for k in keys) / len(keys)
    diffs = paired_psr_diffs(omd_rows, esl)
    boot = bootstrap_mean_ci95(diffs, seed=42, n_boot=10_000)
    assert round(esl_mean, 2) == 86.73
    assert round(omd_mean, 2) == 116.67
    assert round(boot["mean"], 2) == 29.94
    assert round(boot["ci95_low"], 2) == 13.79
    assert round(boot["ci95_high"], 2) == 48.07
    assert boot["mean"] == pytest.approx(29.944589146999935, abs=1e-9)
    assert boot["ci95_low"] == pytest.approx(13.792863472207264, abs=1e-9)
    assert boot["ci95_high"] == pytest.approx(48.07397529072643, abs=1e-9)


def test_freeze_cli_does_not_silently_pass_regen_mismatch():
    src = (ROOT / "reproducibility" / "verify_benchmark_freeze.py").read_text(encoding="utf-8")
    assert "allow-regen-mismatch" not in src
    assert "regeneration mismatch recorded" not in src
    assert "--regen" in src
