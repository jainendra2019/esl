"""Controlled Behavioral Shift Analysis (A=3 appendix diagnostic).

Purpose: identify regimes in which online prototype adaptation becomes beneficial.

Protocol locks:
- 12 committed 3×3 games, focal 1+3
- At t=switch_round, all three opponents switch π0 → π_s (independent samples)
- Intuitive behavioral ladder; TV/JS/KL reported as descriptors
- Cyclic action-label rotations aggregated
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl import games
from esl.baselines.external_common import MatrixGameTask
from esl.experiments.figure1_dynamic_adaptation import (
    FIGURE1_DEFAULT_ESL_UPDATE,
    H_WINDOW,
    SA_FP_DEFAULT_P_STAY,
    DynamicScheduleStep,
    ESLUpdateConfig,
    _schedule_task_index,
    compute_dynamic_prediction_regret,
)
from esl.experiments.figure1_learning_curves import _task_payoff
from esl.experiments.stats_ci import mean_std_ci95
from esl.metrics import cross_entropy
from esl.res.benchmark import load_suite
from esl.res.categorical_cell import (
    _focal_action,
    _method_state,
    _oracle_payoff,
    _ppo_reward_update,
    _predicted_probs,
    _sa_fp_missing_or_full_update,
    _update,
)
from esl.res.runner import game_spec_to_task
from esl.res.stats import paired_mean_ci95
from esl_analysis.metrics.divergence import js_divergence, kl_divergence

ShiftName = Literal["small", "medium", "large"]

BEHAVIORAL_SHIFTS: dict[str, np.ndarray] = {
    "pre": np.array([0.8, 0.1, 0.1], dtype=np.float64),
    "small": np.array([0.7, 0.2, 0.1], dtype=np.float64),
    "medium": np.array([0.5, 0.3, 0.2], dtype=np.float64),
    "large": np.array([0.1, 0.1, 0.8], dtype=np.float64),
}
SHIFT_ORDER: tuple[ShiftName, ...] = ("small", "medium", "large")
METHODS: tuple[str, ...] = ("ESL", "SA-FP", "Fixed-K Bayesian", "Online EM", "PPO")
PAIRED_VS_ESL: tuple[str, ...] = ("SA-FP", "Fixed-K Bayesian", "Online EM")
ROTATIONS: tuple[int, ...] = (0, 1, 2)

DEFAULT_HORIZON = 220
DEFAULT_SWITCH_ROUND = 100
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))


def rotate_policy(probs: np.ndarray, rotation: int) -> np.ndarray:
    """Cyclic permutation of action labels: mass at i moves to (i + rotation) % A."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    r = int(rotation) % int(p.size)
    if r == 0:
        return p.copy()
    return np.roll(p, r)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))))


def shift_divergences(shift: str, *, rotation: int = 0) -> dict[str, float]:
    pre = rotate_policy(BEHAVIORAL_SHIFTS["pre"], rotation)
    post = rotate_policy(BEHAVIORAL_SHIFTS[shift], rotation)
    return {
        "tv": tv_distance(pre, post),
        "js": js_divergence(pre, post),
        "kl_pre_post": kl_divergence(pre, post),  # D_KL(π0 || π_s)
    }


def descriptor_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shift in SHIFT_ORDER:
        d = shift_divergences(shift, rotation=0)
        rows.append(
            {
                "shift": shift,
                "tv": d["tv"],
                "js": d["js"],
                "kl_pre_post": d["kl_pre_post"],
                "kl_direction": "D_KL(pi0 || pi_s)",
            }
        )
    return rows


def oracle_br_action(pay_row: np.ndarray, opp_probs: np.ndarray) -> int:
    utilities = np.asarray(pay_row, dtype=np.float64) @ np.asarray(opp_probs, dtype=np.float64)
    return int(np.argmax(utilities))


def oracle_br_changed(pay_row: np.ndarray, *, shift: str, rotation: int) -> bool:
    pre = rotate_policy(BEHAVIORAL_SHIFTS["pre"], rotation)
    post = rotate_policy(BEHAVIORAL_SHIFTS[shift], rotation)
    return oracle_br_action(pay_row, pre) != oracle_br_action(pay_row, post)


def build_controlled_shift_schedule(
    *,
    task: MatrixGameTask,
    seed: int,
    horizon: int,
    shift: str,
    rotation: int = 0,
    switch_round: int = DEFAULT_SWITCH_ROUND,
) -> list[DynamicScheduleStep]:
    if shift not in SHIFT_ORDER:
        raise ValueError(f"unknown shift {shift!r}; expected one of {SHIFT_ORDER}")
    task_index = _schedule_task_index(task)
    # Method-independent schedule RNG (same recipe family as categorical flagship).
    rng = np.random.default_rng(seed + 1009 * task_index + 17 + 97 * int(rotation))
    pre = rotate_policy(BEHAVIORAL_SHIFTS["pre"], rotation)
    post = rotate_policy(BEHAVIORAL_SHIFTS[shift], rotation)
    steps: list[DynamicScheduleStep] = []
    for t in range(int(horizon)):
        switched = t >= int(switch_round)
        probs = post if switched else pre
        opponent_id = int(rng.choice([1, 2, 3]))
        steps.append(
            DynamicScheduleStep(
                round=t,
                opponent_id=opponent_id,
                true_type=0,  # controlled emission; type id unused
                type_age=t if not switched else t - int(switch_round),
                rounds_since_switch=(t - int(switch_round)) if switched else H_WINDOW + 1,
                switch_id=0 if switched else -1,
                switch_round=int(switch_round) if switched else "",
                old_type="",
                new_type="",
                random_u=float(rng.random()),
                true_action_probs=probs.copy(),
            )
        )
    return steps


def run_controlled_shift_cell(
    *,
    task: MatrixGameTask,
    method: str,
    seed: int,
    shift: str,
    rotation: int = 0,
    horizon: int = DEFAULT_HORIZON,
    h_window: int = H_WINDOW,
    switch_round: int = DEFAULT_SWITCH_ROUND,
    p_stay: float = SA_FP_DEFAULT_P_STAY,
    esl_update: ESLUpdateConfig = FIGURE1_DEFAULT_ESL_UPDATE,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method {method!r} not in appendix METHODS {METHODS}")
    num_actions = 3
    pay = _task_payoff(task)
    if pay.row.shape != (num_actions, num_actions):
        raise ValueError(f"payoff shape {pay.row.shape} != (3,3)")

    schedule = build_controlled_shift_schedule(
        task=task,
        seed=seed,
        horizon=horizon,
        shift=shift,
        rotation=rotation,
        switch_round=switch_round,
    )
    state = _method_state(
        method,  # type: ignore[arg-type]
        seed + 7919,
        num_actions=num_actions,
        p_stay=p_stay,
        pay_row=pay.row,
        horizon=horizon,
    )

    rows: list[dict[str, Any]] = []
    for step in schedule:
        true_probs = np.asarray(step.true_action_probs, dtype=np.float64)
        pred_probs = _predicted_probs(state, step.opponent_id)
        focal_action = _focal_action(state, pay, step.opponent_id, step=step)
        opponent_action = int(state.rng.choice(num_actions, p=true_probs))
        focal_payoff, _ = games.play_pair_payoffs(focal_action, opponent_action, pay)
        oracle_payoff = _oracle_payoff(true_probs, pay)
        prediction_ce = cross_entropy(np.eye(num_actions)[opponent_action], pred_probs)
        oracle_ce = cross_entropy(np.eye(num_actions)[opponent_action], true_probs)
        row = {
            "round": step.round,
            "tau": step.rounds_since_switch,
            "task": task.key,
            "method": method,
            "seed": seed,
            "shift": shift,
            "rotation": rotation,
            "opponent_id": step.opponent_id,
            "focal_action": focal_action,
            "opponent_action": opponent_action,
            "focal_payoff": float(focal_payoff),
            "oracle_payoff": float(oracle_payoff),
            "psr_gap": float(oracle_payoff - focal_payoff),
            "prediction_ce": float(prediction_ce),
            "oracle_ce": float(oracle_ce),
            "dpr_gap": float(prediction_ce - oracle_ce),
        }
        if method == "PPO":
            _ppo_reward_update(
                state,
                step=step,
                focal_action=focal_action,
                focal_payoff=float(focal_payoff),
                last_opponent_action=opponent_action,
            )
        elif method == "SA-FP":
            _sa_fp_missing_or_full_update(
                state,
                step=step,
                observed=True,
                observed_action=opponent_action,
            )
            state.last_focal_action = focal_action
            state.last_opponent_action = opponent_action
        else:
            _update(
                state,
                step=step,
                focal_action=focal_action,
                opponent_action=opponent_action,
                esl_update=esl_update,
            )
        rows.append(row)

    post = [r for r in rows if int(r["round"]) >= int(switch_round) and int(r["tau"]) < int(h_window)]
    psr = float(sum(float(r["psr_gap"]) for r in post))
    dpr = compute_dynamic_prediction_regret(rows, h_window=h_window)
    return {
        "method": method,
        "task": task.key,
        "instance_id": task.key,
        "seed": int(seed),
        "shift": shift,
        "rotation": int(rotation),
        "horizon": int(horizon),
        "switch_round": int(switch_round),
        "h_window": int(h_window),
        "psr": psr,
        "dpr": dpr if dpr != "" else float("nan"),
        "n_post_switch": len(post),
        **shift_divergences(shift, rotation=rotation),
    }


def summarize_psr_table(cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mean PSR by shift × method; honesty flag for uniform ESL dominance."""
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in cell_rows:
        by[(str(r["shift"]), str(r["method"]))].append(float(r["psr"]))

    table: list[dict[str, Any]] = []
    means: dict[str, dict[str, float]] = {s: {} for s in SHIFT_ORDER}
    for shift in SHIFT_ORDER:
        row: dict[str, Any] = {"shift": shift}
        for method in METHODS:
            vals = by.get((shift, method), [])
            st = mean_std_ci95(vals)
            means[shift][method] = float(st["mean"])
            row[method] = float(st["mean"])
            row[f"{method}_ci_low"] = float(st["ci95_low"])
            row[f"{method}_ci_high"] = float(st["ci95_high"])
            row[f"{method}_n"] = int(st["n"])
        table.append(row)

    esl_uniform = True
    for shift in SHIFT_ORDER:
        esl_m = means[shift].get("ESL", float("nan"))
        for method in METHODS:
            if method == "ESL":
                continue
            if means[shift].get(method, float("inf")) < esl_m:
                esl_uniform = False
                break
        if not esl_uniform:
            break

    claim = (
        "ESL mean PSR is lowest at every shift magnitude in this table."
        if esl_uniform
        else "ESL is not uniformly best across shift magnitudes; results are regime-dependent."
    )
    return {
        "table": table,
        "means": means,
        "esl_uniformly_best": bool(esl_uniform),
        "allowed_claim_text": claim,
    }


def paired_contrasts_by_shift(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired (competitor − ESL) PSR CIs within each shift, pairing on (instance, seed, rotation)."""
    out: list[dict[str, Any]] = []
    index: dict[tuple[Any, ...], float] = {}
    for r in cell_rows:
        key = (r["shift"], r["instance_id"], int(r["seed"]), int(r["rotation"]), r["method"])
        index[key] = float(r["psr"])

    for shift in SHIFT_ORDER:
        for competitor in PAIRED_VS_ESL:
            diffs: list[float] = []
            # Discover pairing keys from ESL rows
            for r in cell_rows:
                if r["shift"] != shift or r["method"] != "ESL":
                    continue
                k_esl = (shift, r["instance_id"], int(r["seed"]), int(r["rotation"]), "ESL")
                k_c = (shift, r["instance_id"], int(r["seed"]), int(r["rotation"]), competitor)
                if k_esl in index and k_c in index:
                    diffs.append(index[k_c] - index[k_esl])
            st = paired_mean_ci95(diffs)
            out.append(
                {
                    "shift": shift,
                    "comparison": f"{competitor} - ESL",
                    "delta_psr": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                    "cohen_dz": st["cohen_dz"],
                    "n": int(st["n"]),
                }
            )
    return out


def oracle_br_change_rates(
    *,
    shifts: tuple[str, ...] = SHIFT_ORDER,
    rotations: tuple[int, ...] = ROTATIONS,
) -> list[dict[str, Any]]:
    specs = load_suite(num_actions=3)
    rows: list[dict[str, Any]] = []
    for shift in shifts:
        n = 0
        changed = 0
        for spec in specs:
            pay = np.asarray(spec.payoff_row, dtype=np.float64)
            for rot in rotations:
                n += 1
                if oracle_br_changed(pay, shift=shift, rotation=rot):
                    changed += 1
        rows.append(
            {
                "shift": shift,
                "n_conditions": n,
                "n_br_changed": changed,
                "pct_oracle_br_changed": 100.0 * changed / max(n, 1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = columns or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def run_grid(
    *,
    out_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    rotations: Sequence[int] = ROTATIONS,
    methods: Sequence[str] = METHODS,
    shifts: Sequence[str] = SHIFT_ORDER,
    horizon: int = DEFAULT_HORIZON,
    h_window: int = H_WINDOW,
    switch_round: int = DEFAULT_SWITCH_ROUND,
    jobs: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    specs = load_suite(num_actions=3)
    if smoke:
        specs = specs[:1]
        seeds = tuple(seeds)[:1]
        rotations = (0,)
        methods = ("ESL", "SA-FP", "Fixed-K Bayesian")
        shifts = ("small", "large")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_list: list[dict[str, Any]] = []
    for spec in specs:
        task = game_spec_to_task(spec)
        for seed in seeds:
            for rotation in rotations:
                for shift in shifts:
                    for method in methods:
                        jobs_list.append(
                            {
                                "instance_id": spec.instance_id,
                                "family": spec.family,
                                "method": method,
                                "seed": int(seed),
                                "shift": shift,
                                "rotation": int(rotation),
                                "horizon": int(horizon),
                                "h_window": int(h_window),
                                "switch_round": int(switch_round),
                                "payoff_row": np.asarray(spec.payoff_row, dtype=np.float64).tolist(),
                                "payoff_col": np.asarray(spec.payoff_col, dtype=np.float64).tolist(),
                            }
                        )

    cells: list[dict[str, Any]] = []
    if int(jobs) <= 1:
        for j in jobs_list:
            cells.append(_run_job(j))
    else:
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            futs = [ex.submit(_run_job, j) for j in jobs_list]
            for fut in as_completed(futs):
                cells.append(fut.result())

    cells_sorted = sorted(
        cells,
        key=lambda r: (r["shift"], r["instance_id"], r["rotation"], r["seed"], r["method"]),
    )
    _write_csv(out_dir / "shift_ladder_cells.csv", cells_sorted)

    summary = summarize_psr_table(cells_sorted)
    _write_csv(out_dir / "shift_ladder_summary.csv", summary["table"])
    paired = paired_contrasts_by_shift(cells_sorted)
    _write_csv(out_dir / "shift_ladder_paired.csv", paired)

    desc = descriptor_table()
    br = oracle_br_change_rates(shifts=tuple(shifts), rotations=tuple(rotations))  # type: ignore[arg-type]
    # Merge BR % into descriptors
    br_map = {r["shift"]: r for r in br}
    desc_merged = []
    for d in desc:
        if d["shift"] not in shifts:
            continue
        m = dict(d)
        b = br_map.get(d["shift"], {})
        m["pct_oracle_br_changed"] = b.get("pct_oracle_br_changed", "")
        m["n_br_conditions"] = b.get("n_conditions", "")
        desc_merged.append(m)
    _write_csv(out_dir / "shift_descriptors.csv", desc_merged)

    report = {
        "purpose": "Identify regimes in which online prototype adaptation becomes beneficial.",
        "protocol": {
            "games": len(specs),
            "seeds": list(seeds),
            "rotations": list(rotations),
            "methods": list(methods),
            "shifts": list(shifts),
            "horizon": horizon,
            "switch_round": switch_round,
            "h_window": h_window,
            "all_opponents_switch": True,
        },
        "n_cells": len(cells_sorted),
        "esl_uniformly_best": summary["esl_uniformly_best"],
        "allowed_claim_text": summary["allowed_claim_text"],
        "interpretation_template": (
            "If ESL's relative advantage grows with shift magnitude, this is consistent with "
            "adaptive prototypes helping when fixed representations become increasingly mismatched."
        ),
        "descriptors": desc_merged,
        "paired": paired,
        "summary_table": summary["table"],
    }
    (out_dir / "mechanistic_a3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(out_dir / "mechanistic_a3_report.md", report)
    _write_summary_tex(out_dir / "shift_ladder_summary.tex", summary["table"])
    return report


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    task = MatrixGameTask(
        key=job["instance_id"],
        label=job["instance_id"],
        payoff_game="prisoners_dilemma",
        payoff_row=np.asarray(job["payoff_row"], dtype=np.float64),
        payoff_col=np.asarray(job["payoff_col"], dtype=np.float64),
    )
    out = run_controlled_shift_cell(
        task=task,
        method=job["method"],
        seed=int(job["seed"]),
        shift=job["shift"],
        rotation=int(job["rotation"]),
        horizon=int(job["horizon"]),
        h_window=int(job["h_window"]),
        switch_round=int(job["switch_round"]),
    )
    out["family"] = job.get("family", "")
    return out


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Controlled Behavioral Shift Analysis",
        "",
        report["purpose"],
        "",
        "## Descriptors",
        "",
        "| Shift | TV | JS | D_KL(π0‖πs) | Oracle BR changed (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in report["descriptors"]:
        lines.append(
            f"| {d['shift']} | {d['tv']:.4f} | {d['js']:.4f} | {d['kl_pre_post']:.4f} | "
            f"{d.get('pct_oracle_br_changed', '')} |"
        )
    lines += ["", "## Mean PSR (lower better)", "", "| Shift | ESL | SA-FP | Fixed-K | Online EM | PPO |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["summary_table"]:
        lines.append(
            "| {shift} | {ESL:.2f} | {SA-FP:.2f} | {Fixed-K Bayesian:.2f} | {Online EM:.2f} | {PPO:.2f} |".format(
                **{k: row.get(k, float("nan")) for k in ("shift", "ESL", "SA-FP", "Fixed-K Bayesian", "Online EM", "PPO")}
            )
        )
    lines += ["", "## Paired contrasts (competitor − ESL)", ""]
    for p in report["paired"]:
        lines.append(
            f"- {p['shift']}: {p['comparison']}: ΔPSR={p['delta_psr']:.2f} "
            f"[{p['ci95_low']:.2f}, {p['ci95_high']:.2f}] (dz={p['cohen_dz']:.2f}, n={p['n']})"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        report["allowed_claim_text"],
        "",
        report["interpretation_template"],
        "",
        "PPO is reported descriptively and is not central to this mechanistic comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_tex(path: Path, table: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Shift & ESL & SA-FP & Fixed-$K$ & Online EM & PPO \\",
        r"\midrule",
    ]
    for row in table:
        lines.append(
            f"{row['shift']} & {row['ESL']:.2f} & {row['SA-FP']:.2f} & "
            f"{row['Fixed-K Bayesian']:.2f} & {row['Online EM']:.2f} & {row['PPO']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
