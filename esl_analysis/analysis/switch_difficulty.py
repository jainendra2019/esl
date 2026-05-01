from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from esl_analysis.metrics.divergence import js_divergence
from esl_analysis.metrics.regret import compute_psr_per_switch


def assign_bin(js_value: float) -> str:
    if js_value < 0.3:
        return "low"
    if js_value < 0.8:
        return "medium"
    return "high"


def _policy(row: dict[str, Any]) -> np.ndarray:
    return np.array([float(row["true_p_coop"]), float(row["true_p_defect"])], dtype=np.float64)


def _switch_keys(rows: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    seen: dict[int, int] = {}
    for row in rows:
        sid = int(row.get("switch_id", -1))
        if sid < 0:
            continue
        switch_round = row.get("switch_round", "")
        if switch_round == "":
            continue
        round_i = int(switch_round)
        seen.setdefault(round_i, sid)
    return [(0, sid, round_i) for round_i, sid in sorted(seen.items())]


def compute_switch_difficulty(
    rows: list[dict[str, Any]],
    *,
    w_pre: int = 20,
    w_post: int = 20,
    psr_horizon: int = 100,
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    excluded = 0
    for opponent_id, switch_id, switch_round in _switch_keys(rows):
        pre = [
            _policy(row)
            for row in rows
            if switch_round - w_pre <= int(row["round"]) < switch_round
        ]
        post = [
            _policy(row)
            for row in rows
            if switch_round <= int(row["round"]) < switch_round + w_post
        ]
        psr_rows = [
            row
            for row in rows
            if switch_round <= int(row["round"]) <= switch_round + psr_horizon
        ]
        if len(pre) < w_pre or len(post) < w_post or not psr_rows:
            excluded += 1
            continue
        difficulty = js_divergence(np.mean(pre, axis=0), np.mean(post, axis=0))
        out.append(
            {
                "opponent_id": opponent_id,
                "switch_id": switch_id,
                "switch_round": switch_round,
                "difficulty_label": next((str(row.get("difficulty_label")) for row in rows if int(row["round"]) == switch_round and row.get("difficulty_label", "")), ""),
                "difficulty": difficulty,
                "difficulty_bin": assign_bin(difficulty),
                "psr": compute_psr_per_switch(psr_rows, switch_time=switch_round, horizon=psr_horizon),
            }
        )
    return out, excluded


def aggregate_switch_difficulty(run_switches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Equal weighting per run: average switches within run/bin first, then aggregate runs.
    by_run_bin: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in run_switches:
        difficulty = row.get("difficulty_label") or row["difficulty_bin"]
        by_run_bin[(difficulty, row["method"], row["game"], int(row["seed"]))].append(float(row["psr"]))
    run_rows: list[dict[str, Any]] = []
    for (difficulty, method, game, seed), vals in by_run_bin.items():
        run_rows.append({"difficulty": difficulty, "method": method, "game": game, "seed": seed, "psr": float(np.mean(vals))})
    out: list[dict[str, Any]] = []
    difficulty_order = ("small", "medium", "large", "low", "high")
    for difficulty in difficulty_order:
        for method in sorted({r["method"] for r in run_rows}):
            vals = [r["psr"] for r in run_rows if r["difficulty"] == difficulty and r["method"] == method]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "psr_mean": float(arr.mean()),
                    "psr_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n_runs": len(arr),
                }
            )
    return out
