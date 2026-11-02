"""Focal inverse-temperature (λ) sensitivity for ESL on the clean 3×3 suite.

Locks:
- Vary only λ_focal; default λ=2 remains pre-specified (not retuned).
- Separate env_rng vs focal_rng so opponent actions/signals match across λ.
- Opponent emissions use schedule true_action_probs only (no focal-dependent type-2 override).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from esl import games
from esl.baselines.external_common import MatrixGameTask
from esl.experiments.figure1_dynamic_adaptation import (
    FIGURE1_DEFAULT_ESL_UPDATE,
    FIGURE1_MAIN_SEEDS,
    H_WINDOW,
    LAMBDA_BR,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
    ESLUpdateConfig,
)
from esl.experiments.figure1_learning_curves import _task_payoff
from esl.experiments.stats_ci import mean_std_ci95
from esl.metrics import cross_entropy
from esl.res.benchmark import load_suite
from esl.res.categorical_cell import (
    _method_state,
    _oracle_payoff,
    _predicted_probs,
    _softmax,
    _update,
    build_categorical_schedule,
)
from esl.res.runner import game_spec_to_task
from esl.res.stats import paired_mean_ci95

FOCAL_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.5, 2.0, 5.0, 10.0)
DEFAULT_LAMBDA = float(LAMBDA_BR)  # 2.0 — pre-locked
HORIZON = 1000
EPS = 1e-12


def policy_probs_from_utilities(utilities: np.ndarray, lam: float) -> np.ndarray:
    """Softmax(λU); λ=0 is exactly uniform (explicit branch)."""
    u = np.asarray(utilities, dtype=np.float64).reshape(-1)
    if float(lam) == 0.0:
        return np.full(u.shape[0], 1.0 / float(u.shape[0]), dtype=np.float64)
    return _softmax(u, lam=float(lam))


def action_entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def best_action_prob(probs: np.ndarray, utilities: np.ndarray) -> float:
    u = np.asarray(utilities, dtype=np.float64)
    # Break ties by lowest index for determinism
    best = int(np.argmax(u))
    return float(np.asarray(probs, dtype=np.float64)[best])


def run_focal_lambda_cell(
    *,
    task: MatrixGameTask,
    seed: int,
    lam_focal: float,
    horizon: int = HORIZON,
    h_window: int = H_WINDOW,
    switch_interval: int = SWITCH_INTERVAL,
    switch_hazard: float = SWITCH_HAZARD,
    esl_update: ESLUpdateConfig = FIGURE1_DEFAULT_ESL_UPDATE,
) -> dict[str, Any]:
    """ESL cell with split RNGs: env stream independent of λ; focal stream uses λ."""
    num_actions = 3
    pay = _task_payoff(task)
    if pay.row.shape != (num_actions, num_actions):
        raise ValueError(f"expected 3x3 payoffs, got {pay.row.shape}")

    # Environment stream: schedule + opponent actions — depends on (task, seed) only.
    schedule = build_categorical_schedule(
        task=task,
        seed=seed,
        horizon=horizon,
        num_actions=num_actions,
        switch_interval=switch_interval,
        switch_hazard=switch_hazard,
    )
    env_rng = np.random.default_rng(seed + 4242)

    # Learner state RNG only used for ESL init noise; keep λ-independent so θ0 matches.
    state = _method_state(
        "ESL",
        seed + 7919,
        num_actions=num_actions,
        p_stay=1.0,
        pay_row=pay.row,
        horizon=horizon,
    )
    # Focal action sampling stream — may depend on λ (isolated from env).
    focal_rng = np.random.default_rng(seed + 17_389 + int(round(float(lam_focal) * 1000)))

    rows: list[dict[str, Any]] = []
    entropies: list[float] = []
    p_bests: list[float] = []
    opp_actions: list[int] = []
    focal_actions: list[int] = []
    true_types: list[int] = []
    signals: list[int] = []

    for step in schedule:
        # Exogenous emissions: use schedule probs only (no last_focal override).
        true_probs = np.asarray(step.true_action_probs, dtype=np.float64)
        true_probs = true_probs / true_probs.sum()
        pred_probs = _predicted_probs(state, step.opponent_id)
        utilities = np.array(
            [float(np.sum(pred_probs * pay.row[act, :])) for act in range(num_actions)],
            dtype=np.float64,
        )
        pi = policy_probs_from_utilities(utilities, float(lam_focal))
        entropies.append(action_entropy(pi))
        p_bests.append(best_action_prob(pi, utilities))
        focal_action = int(focal_rng.choice(num_actions, p=pi))
        opponent_action = int(env_rng.choice(num_actions, p=true_probs))
        focal_payoff, _ = games.play_pair_payoffs(focal_action, opponent_action, pay)
        oracle_payoff = _oracle_payoff(true_probs, pay)
        prediction_ce = cross_entropy(np.eye(num_actions)[opponent_action], pred_probs)
        oracle_ce = cross_entropy(np.eye(num_actions)[opponent_action], true_probs)

        # Full observation (clean protocol).
        _update(
            state,
            step=step,
            focal_action=focal_action,
            opponent_action=opponent_action,
            esl_update=esl_update,
        )

        rows.append(
            {
                "round": step.round,
                "tau": step.rounds_since_switch,
                "opponent_id": step.opponent_id,
                "true_type": step.true_type,
                "focal_action": focal_action,
                "opponent_action": opponent_action,
                "observed_action": opponent_action,
                "psr_gap": float(oracle_payoff - focal_payoff),
                "dpr_gap": float(prediction_ce - oracle_ce),
            }
        )
        opp_actions.append(opponent_action)
        focal_actions.append(focal_action)
        true_types.append(int(step.true_type))
        signals.append(opponent_action)

    post = [r for r in rows if int(r["tau"]) < int(h_window) and int(r.get("tau", 10**9)) >= 0]
    # Match flagship PSR: sum gaps where rounds_since_switch < H (post-switch windows).
    psr = float(sum(float(r["psr_gap"]) for r in rows if int(r["tau"]) < int(h_window)))
    dpr = float(sum(float(r["dpr_gap"]) for r in rows if int(r["tau"]) < int(h_window)))

    return {
        "method": "ESL",
        "instance_id": task.key,
        "seed": int(seed),
        "lam_focal": float(lam_focal),
        "horizon": int(horizon),
        "h_window": int(h_window),
        "psr": psr,
        "dpr": dpr,
        "mean_action_entropy": float(np.mean(entropies)),
        "mean_p_best": float(np.mean(p_bests)),
        "opponent_actions": opp_actions,
        "focal_actions": focal_actions,
        "true_types": true_types,
        "signals": signals,
        "n_post_rows": len([r for r in rows if int(r["tau"]) < int(h_window)]),
    }


def summarize_lambda_grid(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean PSR/entropy and paired Δ vs λ=2 per (instance, seed)."""
    by_lam: dict[float, list[dict[str, Any]]] = defaultdict(list)
    index: dict[tuple[Any, ...], float] = {}
    for r in cells:
        lam = float(r["lam_focal"])
        by_lam[lam].append(r)
        index[(r["instance_id"], int(r["seed"]), lam)] = float(r["psr"])

    table: list[dict[str, Any]] = []
    for lam in FOCAL_LAMBDA_GRID:
        rows = by_lam.get(float(lam), [])
        psr_vals = [float(r["psr"]) for r in rows]
        ent_vals = [float(r["mean_action_entropy"]) for r in rows]
        pbest_vals = [float(r["mean_p_best"]) for r in rows]
        dpr_vals = [float(r["dpr"]) for r in rows]
        st = mean_std_ci95(psr_vals)
        st_e = mean_std_ci95(ent_vals)
        st_pb = mean_std_ci95(pbest_vals)
        st_d = mean_std_ci95(dpr_vals)

        deltas: list[float] = []
        for r in rows:
            key2 = (r["instance_id"], int(r["seed"]), DEFAULT_LAMBDA)
            key = (r["instance_id"], int(r["seed"]), float(lam))
            if key2 in index and key in index:
                deltas.append(index[key] - index[key2])
        st_dlt = paired_mean_ci95(deltas)

        table.append(
            {
                "lam_focal": float(lam),
                "n": int(st["n"]),
                "mean_psr": st["mean"],
                "psr_ci95_low": st["ci95_low"],
                "psr_ci95_high": st["ci95_high"],
                "delta_vs_default": st_dlt["mean"],
                "delta_ci95_low": st_dlt["ci95_low"],
                "delta_ci95_high": st_dlt["ci95_high"],
                "delta_cohen_dz": st_dlt["cohen_dz"],
                "mean_action_entropy": st_e["mean"],
                "mean_p_best": st_pb["mean"],
                "mean_dpr": st_d["mean"],
                "dpr_ci95_low": st_d["ci95_low"],
                "dpr_ci95_high": st_d["ci95_high"],
            }
        )

    # Structural invariance check: DPR nearly constant across λ
    dpr_means = [float(r["mean_dpr"]) for r in table if math.isfinite(r["mean_dpr"])]
    dpr_span = (max(dpr_means) - min(dpr_means)) if dpr_means else float("nan")
    return {
        "table": table,
        "default_lambda": DEFAULT_LAMBDA,
        "dpr_span_across_lambda": dpr_span,
        "dpr_approximately_invariant": bool(dpr_span < 1e-6) if math.isfinite(dpr_span) else False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = columns or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def run_grid(
    *,
    out_dir: Path,
    seeds: Sequence[int] = FIGURE1_MAIN_SEEDS,
    lambdas: Sequence[float] = FOCAL_LAMBDA_GRID,
    horizon: int = HORIZON,
    jobs: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    specs = load_suite(num_actions=3)
    if smoke:
        specs = specs[:1]
        seeds = tuple(seeds)[:1]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs_list: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            for lam in lambdas:
                jobs_list.append(
                    {
                        "instance_id": spec.instance_id,
                        "family": spec.family,
                        "seed": int(seed),
                        "lam_focal": float(lam),
                        "horizon": int(horizon),
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

    # Drop heavy trajectory arrays from on-disk cells
    slim = []
    for r in cells:
        slim.append({k: v for k, v in r.items() if k not in ("opponent_actions", "focal_actions", "true_types", "signals")})
    slim_sorted = sorted(slim, key=lambda r: (float(r["lam_focal"]), r["instance_id"], int(r["seed"])))
    _write_csv(out_dir / "cells.csv", slim_sorted)

    summary = summarize_lambda_grid(slim_sorted)
    _write_csv(out_dir / "summary.csv", summary["table"])
    _write_summary_tex(out_dir / "summary.tex", summary["table"])
    _plot_psr(out_dir / "psr_vs_lambda.pdf", summary["table"])
    _plot_psr(out_dir / "psr_vs_lambda.png", summary["table"])

    report = {
        "question": "How sensitive is ESL to focal policy inverse temperature λ?",
        "default_lambda": DEFAULT_LAMBDA,
        "grid": list(lambdas),
        "n_cells": len(slim_sorted),
        "protocol": {
            "games": len(specs),
            "seeds": list(seeds),
            "horizon": horizon,
            "h_window": H_WINDOW,
            "p_obs": 1.0,
            "obs_noise": 0.0,
            "methods": ["ESL"],
            "separate_env_focal_rng": True,
            "exogenous_opponent_emissions": True,
        },
        "summary": summary["table"],
        "dpr_span_across_lambda": summary["dpr_span_across_lambda"],
        "dpr_approximately_invariant": summary["dpr_approximately_invariant"],
        "interpretation_note": (
            "DPR/LTE are expected to be nearly invariant because opponent emissions and "
            "observation streams are held fixed across λ; only focal decision stochasticity changes."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    task = MatrixGameTask(
        key=job["instance_id"],
        label=job["instance_id"],
        payoff_game="prisoners_dilemma",
        payoff_row=np.asarray(job["payoff_row"], dtype=np.float64),
        payoff_col=np.asarray(job["payoff_col"], dtype=np.float64),
    )
    out = run_focal_lambda_cell(
        task=task,
        seed=int(job["seed"]),
        lam_focal=float(job["lam_focal"]),
        horizon=int(job["horizon"]),
    )
    out["family"] = job.get("family", "")
    return out


def _write_summary_tex(path: Path, table: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\lambda$ & PSR & $\Delta$ vs $\lambda{=}2$ & 95\% CI($\Delta$) & Entropy & $p$(best) \\",
        r"\midrule",
    ]
    for r in table:
        lines.append(
            f"{r['lam_focal']:g} & {r['mean_psr']:.2f} & {r['delta_vs_default']:+.2f} & "
            f"[{r['delta_ci95_low']:.2f},\\,{r['delta_ci95_high']:.2f}] & "
            f"{r['mean_action_entropy']:.3f} & {r['mean_p_best']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_psr(path: Path, table: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    xs = [float(r["lam_focal"]) for r in table]
    ys = [float(r["mean_psr"]) for r in table]
    lo = [float(r["psr_ci95_low"]) for r in table]
    hi = [float(r["psr_ci95_high"]) for r in table]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.fill_between(xs, lo, hi, alpha=0.2, color="#1f77b4")
    ax.plot(xs, ys, "-o", color="#1f77b4", label="ESL PSR")
    ax.axvline(DEFAULT_LAMBDA, color="0.4", ls="--", lw=1.0, label=r"default $\lambda=2$")
    ax.set_xlabel(r"Focal inverse temperature $\lambda$")
    ax.set_ylabel("PSR (lower better)")
    ax.set_title("Focal-$\\lambda$ sensitivity (clean $3\\times 3$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
