"""Figure 4: Robbins--Monro dynamics diagnostic for ESL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from esl import beliefs as belief_ops
from esl import games as game_lib
from esl.baselines.external_common import MatrixGameTask, matrix_game_tasks
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure1_dynamic_adaptation import (
    ACTION_DIM,
    EPS,
    K_TYPES,
    LAMBDA_BR,
    build_type_shifting_schedule,
)
from esl.experiments.figure1_learning_curves import _task_payoff
from esl.experiments.figure3_dynamics import (
    _belief_entropy,
    _distribution_entropy,
    _esl_initial_prototypes,
    _rolling,
    _sample_binary,
    _softmax,
    js_divergence,
)
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.stats_ci import mean_std_ci95
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax

DEFAULT_GAMES = ("ipd", "stag_hunt", "matching_pennies")
DEFAULT_BATCH_SIZE = 50
DEFAULT_RM_C = 0.5
DEFAULT_ETA_REG = 1e-3
DEFAULT_HORIZON = 3000
DEFAULT_ROLLING_WINDOW = 5
GAMMA_EXPONENT = -0.9

SUMMARY_COLUMNS: tuple[str, ...] = (
    "game",
    "seed",
    "rm_c",
    "eta_reg",
    "batch_size",
    "num_slow_steps",
    "initial_prototype_norm",
    "max_prototype_norm",
    "final_prototype_norm",
    "early_norm_mean",
    "late_norm_mean",
    "norm_growth_ratio",
    "norm_late_early_ratio",
    "early_movement",
    "late_movement",
    "movement_ratio",
    "early_effective_step",
    "late_effective_step",
    "effective_step_ratio",
    "early_entropy",
    "late_entropy",
    "entropy_change",
    "late_usage_jsd_mean",
    "late_usage_jsd_max",
    "late_action_jsd_mean",
    "late_action_jsd_max",
    "gamma_first",
    "gamma_last",
    "cumulative_sum_gamma_last",
    "cumulative_sum_gamma_sq_last",
)
TIMESERIES_COLUMNS: tuple[str, ...] = (
    "slow_step",
    "env_step",
    "seed",
    "game",
    "theta_flat",
    "prototype_norm",
    "prototype_step_size",
    "mean_belief_entropy",
    "prototype_usage",
    "usage_entropy",
    "belief_total_variation",
    "mean_update_norm",
    "window_action_distribution",
    "window_gradient_norm",
    "gamma_m",
    "cumulative_sum_gamma",
    "cumulative_sum_gamma_sq",
    "effective_step_ratio",
    "rm_c",
    "eta_reg",
    "batch_size",
)


@dataclass
class RMDynamicsState:
    rng: np.random.Generator
    logits: np.ndarray
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    batch_actions: list[int] = field(default_factory=list)
    batch_beliefs_after: list[np.ndarray] = field(default_factory=list)
    previous_beliefs: dict[int, np.ndarray] = field(default_factory=dict)


def rm_gamma(step_m: int, c: float = DEFAULT_RM_C) -> float:
    return float(c) * float((int(step_m) + 1) ** GAMMA_EXPONENT)


def _state(seed: int) -> RMDynamicsState:
    beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    return RMDynamicsState(
        rng=np.random.default_rng(seed),
        logits=_esl_initial_prototypes(seed),
        beliefs={opp: b.copy() for opp, b in beliefs.items()},
        previous_beliefs={opp: b.copy() for opp, b in beliefs.items()},
    )


def _predicted_opponent_probs(state: RMDynamicsState, opponent_id: int) -> np.ndarray:
    return state.beliefs[opponent_id] @ stable_softmax(state.logits)


def _action_probs(pred: np.ndarray, pay: game_lib.PayoffMatrices) -> np.ndarray:
    utilities = np.array([np.sum(pred * pay.row[a, :]) for a in range(ACTION_DIM)], dtype=np.float64)
    return _softmax(utilities, lam=LAMBDA_BR)


def _bayes_step(prior: np.ndarray, prototypes: np.ndarray, opponent_action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, opponent_action), 1e-4, EPS)


def run_rm_dynamics_cell(
    *,
    task: MatrixGameTask,
    seed: int,
    horizon: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rm_c: float = DEFAULT_RM_C,
    eta_reg: float = DEFAULT_ETA_REG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pay = _task_payoff(task)
    schedule = build_type_shifting_schedule(task=task, seed=seed, horizon=horizon)
    state = _state(seed + 29001)
    initial_norm = float(np.linalg.norm(state.logits))
    records: list[dict[str, Any]] = []
    slow_step = 0
    cumulative_gamma = 0.0
    cumulative_gamma_sq = 0.0

    for step in schedule:
        pred = _predicted_opponent_probs(state, step.opponent_id)
        focal_action = _sample_binary(state.rng, _action_probs(pred, pay))
        opponent_action = game_lib.ACTION_COOPERATE if step.random_u < step.true_action_probs[0] else game_lib.ACTION_DEFECT
        belief_before = state.beliefs[step.opponent_id].copy()
        belief_after = _bayes_step(belief_before, state.logits, opponent_action)
        state.previous_beliefs[step.opponent_id] = belief_before
        state.beliefs[step.opponent_id] = belief_after
        state.batch.append((opponent_action, belief_before))
        state.batch_actions.append(focal_action)
        state.batch_beliefs_after.append(belief_after.copy())

        if len(state.batch) < batch_size:
            continue

        theta_before = state.logits.copy()
        grad = np.zeros_like(state.logits)
        for signal, belief in state.batch:
            grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
        grad = grad / len(state.batch)
        gamma = rm_gamma(slow_step, c=rm_c)
        cumulative_gamma += gamma
        cumulative_gamma_sq += gamma * gamma
        gradient_norm = float(np.linalg.norm(grad))
        update_direction = grad - float(eta_reg) * state.logits
        state.logits = state.logits + gamma * update_direction
        delta = state.logits - theta_before
        usage = np.mean(np.asarray(state.batch_beliefs_after, dtype=np.float64), axis=0)
        usage = usage / np.maximum(usage.sum(), EPS)
        action_counts = np.bincount(state.batch_actions, minlength=ACTION_DIM).astype(np.float64)
        action_dist = action_counts / np.maximum(action_counts.sum(), EPS)
        belief_tv = float(
            np.mean(
                [
                    0.5 * np.sum(np.abs(state.beliefs[opp] - state.previous_beliefs[opp]))
                    for opp in sorted(state.beliefs)
                ]
            )
        )
        update_norm = float(np.linalg.norm(delta))
        effective_step = float(gamma * gradient_norm)
        records.append(
            {
                "slow_step": slow_step,
                "env_step": step.round,
                "seed": seed,
                "game": task.key,
                "theta_flat": json.dumps([float(x) for x in state.logits.reshape(-1)]),
                "prototype_norm": float(np.linalg.norm(state.logits)),
                "prototype_step_size": update_norm,
                "mean_belief_entropy": float(np.mean([_belief_entropy(b) for b in state.beliefs.values()])),
                "prototype_usage": json.dumps([float(x) for x in usage]),
                "usage_entropy": _distribution_entropy(usage),
                "belief_total_variation": belief_tv,
                "mean_update_norm": update_norm,
                "window_action_distribution": json.dumps([float(x) for x in action_dist]),
                "window_gradient_norm": gradient_norm,
                "gamma_m": gamma,
                "cumulative_sum_gamma": cumulative_gamma,
                "cumulative_sum_gamma_sq": cumulative_gamma_sq,
                "effective_step_ratio": effective_step,
                "rm_c": float(rm_c),
                "eta_reg": float(eta_reg),
                "batch_size": int(batch_size),
            }
        )
        slow_step += 1
        state.batch.clear()
        state.batch_actions.clear()
        state.batch_beliefs_after.clear()

    summary = summarize_rm_records(
        records,
        game=task.key,
        seed=seed,
        initial_norm=initial_norm,
        batch_size=batch_size,
        rm_c=rm_c,
        eta_reg=eta_reg,
    )
    return records, summary


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _first_last(values: np.ndarray, frac: float = 0.25) -> tuple[float, float]:
    n = max(1, int(math.ceil(len(values) * frac)))
    return _safe_mean(values[:n]), _safe_mean(values[-n:])


def _late_window_jsd(records: list[dict[str, Any]], key: str, *, num_windows: int = 5) -> tuple[float, float]:
    if len(records) < 2:
        return 0.0, 0.0
    late = records[len(records) // 2 :]
    chunks = [c for c in np.array_split(np.arange(len(late)), min(num_windows, len(late))) if len(c) > 0]
    dists: list[np.ndarray] = []
    for chunk in chunks:
        vals = [np.asarray(json.loads(late[int(i)][key]), dtype=np.float64) for i in chunk]
        dist = np.mean(vals, axis=0)
        dists.append(dist / np.maximum(dist.sum(), EPS))
    jsds = [js_divergence(dists[i], dists[i + 1]) for i in range(len(dists) - 1)]
    if not jsds:
        return 0.0, 0.0
    return float(np.mean(jsds)), float(np.max(jsds))


def summarize_rm_records(
    records: list[dict[str, Any]],
    *,
    game: str,
    seed: int,
    initial_norm: float,
    batch_size: int,
    rm_c: float,
    eta_reg: float,
) -> dict[str, Any]:
    if not records:
        out = {col: 0.0 for col in SUMMARY_COLUMNS}
        out.update({"game": game, "seed": seed, "rm_c": rm_c, "eta_reg": eta_reg, "batch_size": batch_size})
        return out
    norms = np.asarray([float(r["prototype_norm"]) for r in records], dtype=np.float64)
    movement = np.asarray([float(r["prototype_step_size"]) for r in records], dtype=np.float64)
    entropy = np.asarray([float(r["mean_belief_entropy"]) for r in records], dtype=np.float64)
    effective = np.asarray([float(r["effective_step_ratio"]) for r in records], dtype=np.float64)
    early_norm, late_norm = _first_last(norms)
    early_movement, late_movement = _first_last(movement)
    early_entropy, late_entropy = _first_last(entropy)
    early_effective, late_effective = _first_last(effective)
    usage_jsd_mean, usage_jsd_max = _late_window_jsd(records, "prototype_usage")
    action_jsd_mean, action_jsd_max = _late_window_jsd(records, "window_action_distribution")
    gamma_first = float(records[0]["gamma_m"])
    gamma_last = float(records[-1]["gamma_m"])
    return {
        "game": game,
        "seed": int(seed),
        "rm_c": float(rm_c),
        "eta_reg": float(eta_reg),
        "batch_size": int(batch_size),
        "num_slow_steps": int(len(records)),
        "initial_prototype_norm": float(initial_norm),
        "max_prototype_norm": float(np.max(norms)),
        "final_prototype_norm": float(norms[-1]),
        "early_norm_mean": early_norm,
        "late_norm_mean": late_norm,
        "norm_growth_ratio": float(norms[-1] / max(initial_norm, EPS)),
        "norm_late_early_ratio": float(late_norm / max(early_norm, EPS)),
        "early_movement": early_movement,
        "late_movement": late_movement,
        "movement_ratio": float(late_movement / max(early_movement, EPS)),
        "early_effective_step": early_effective,
        "late_effective_step": late_effective,
        "effective_step_ratio": float(late_effective / max(early_effective, EPS)),
        "early_entropy": early_entropy,
        "late_entropy": late_entropy,
        "entropy_change": float(late_entropy - early_entropy),
        "late_usage_jsd_mean": usage_jsd_mean,
        "late_usage_jsd_max": usage_jsd_max,
        "late_action_jsd_mean": action_jsd_mean,
        "late_action_jsd_max": action_jsd_max,
        "gamma_first": gamma_first,
        "gamma_last": gamma_last,
        "cumulative_sum_gamma_last": float(records[-1]["cumulative_sum_gamma"]),
        "cumulative_sum_gamma_sq_last": float(records[-1]["cumulative_sum_gamma_sq"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _aggregate_timeseries(records: list[dict[str, Any]], key: str) -> list[dict[str, float]]:
    by_step: dict[int, list[float]] = {}
    for row in records:
        by_step.setdefault(int(row["slow_step"]), []).append(float(row[key]))
    out: list[dict[str, float]] = []
    for step in sorted(by_step):
        st = mean_std_ci95(by_step[step])
        out.append({"slow_step": float(step), **st})
    return out


def _metric_ci(summary_rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    return mean_std_ci95([float(r[metric]) for r in summary_rows])


def _all_finite(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> bool:
    for row in rows:
        for col in columns:
            if col in ("game", "seed"):
                continue
            try:
                if not np.isfinite(float(row[col])):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _gate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = _all_finite(summary_rows, SUMMARY_COLUMNS)
    max_norm = max(float(r["max_prototype_norm"]) for r in summary_rows) if summary_rows else float("nan")
    norm_growth = max(float(r["norm_growth_ratio"]) for r in summary_rows) if summary_rows else float("nan")
    norm_late_early = max(float(r["norm_late_early_ratio"]) for r in summary_rows) if summary_rows else float("nan")
    movement_ratio = max(float(r["movement_ratio"]) for r in summary_rows) if summary_rows else float("nan")
    effective_ratio = max(float(r["effective_step_ratio"]) for r in summary_rows) if summary_rows else float("nan")
    usage_jsd = _metric_ci(summary_rows, "late_usage_jsd_mean")["mean"] if summary_rows else float("nan")
    action_jsd = _metric_ci(summary_rows, "late_action_jsd_mean")["mean"] if summary_rows else float("nan")
    gamma_decreases = all(float(r["gamma_first"]) > float(r["gamma_last"]) for r in summary_rows)
    gamma_sums_finite = all(
        np.isfinite(float(r["cumulative_sum_gamma_last"]))
        and np.isfinite(float(r["cumulative_sum_gamma_sq_last"]))
        and float(r["cumulative_sum_gamma_last"]) > 0.0
        and float(r["cumulative_sum_gamma_sq_last"]) > 0.0
        for r in summary_rows
    )
    checks = {
        "all_finite": bool(finite),
        "max_norm_finite": bool(np.isfinite(max_norm)),
        "norm_growth_bounded": bool(np.isfinite(norm_growth) and norm_growth < 5.0),
        "late_norm_not_diverging": bool(np.isfinite(norm_late_early) and norm_late_early < 2.5),
        "movement_decreases_or_controlled": bool(np.isfinite(movement_ratio) and movement_ratio < 1.25),
        "effective_step_decreases_or_controlled": bool(np.isfinite(effective_ratio) and effective_ratio < 1.25),
        "usage_jsd_stable": bool(np.isfinite(usage_jsd) and usage_jsd < 0.25),
        "action_jsd_stable": bool(np.isfinite(action_jsd) and action_jsd < 0.25),
        "gamma_decreases": bool(gamma_decreases),
        "gamma_cumulative_diagnostics_finite": bool(gamma_sums_finite),
    }
    values = {
        "max_prototype_norm": float(max_norm),
        "max_norm_growth_ratio": float(norm_growth),
        "max_norm_late_early_ratio": float(norm_late_early),
        "max_movement_ratio": float(movement_ratio),
        "max_effective_step_ratio": float(effective_ratio),
        "mean_late_usage_jsd": float(usage_jsd),
        "mean_late_action_jsd": float(action_jsd),
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "values": values}


def _plot_figure(records: list[dict[str, Any]], summary_rows: list[dict[str, Any]], path: Path, *, rolling_window: int) -> None:
    norm = _aggregate_timeseries(records, "prototype_norm")
    movement = _aggregate_timeseries(records, "prototype_step_size")
    entropy = _aggregate_timeseries(records, "mean_belief_entropy")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.reshape(-1)
    panels = [
        (ax_a, "(a)", "Bounded prototype norms under RM updates", r"$\|\Theta\|_F$"),
        (ax_b, "(b)", "Vanishing update magnitudes", r"$\|\Delta \Theta\|_F$"),
        (ax_c, "(c)", "Belief-state regime formation", "Normalized belief entropy"),
        (ax_d, "(d)", "Stable late-window distributions", "Adjacent-window JSD"),
    ]
    for ax, label, title, ylabel in panels:
        ax.text(-0.08, 1.10, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax, series, smooth in ((ax_a, norm, False), (ax_b, movement, True), (ax_c, entropy, False)):
        x = np.asarray([r["slow_step"] for r in series], dtype=np.float64)
        mean = np.asarray([r["mean"] for r in series], dtype=np.float64)
        lo = np.asarray([r["ci95_low"] for r in series], dtype=np.float64)
        hi = np.asarray([r["ci95_high"] for r in series], dtype=np.float64)
        if smooth:
            mean = _rolling(mean, rolling_window)
            lo = _rolling(lo, rolling_window)
            hi = _rolling(hi, rolling_window)
        ax.plot(x, mean, color="#1f77b4", linewidth=2.0)
        ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
        ax.set_xlabel("Slow update")
    bars = [
        ("Prototype usage JSD", _metric_ci(summary_rows, "late_usage_jsd_mean")),
        ("Action distribution JSD", _metric_ci(summary_rows, "late_action_jsd_mean")),
    ]
    x = np.arange(len(bars))
    means = [b[1]["mean"] for b in bars]
    yerr = np.array(
        [
            [max(0.0, means[i] - bars[i][1]["ci95_low"]), max(0.0, bars[i][1]["ci95_high"] - means[i])]
            for i in range(len(bars))
        ]
    ).T
    ax_d.bar(x, means, yerr=yerr, capsize=4, color=["#4c72b0", "#55a868"])
    ax_d.set_xticks(x, [b[0] for b in bars], rotation=15, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=publication_figure_dpi())
    plt.close(fig)


def _write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    *,
    seeds: list[int],
    games: tuple[str, ...],
    horizon: int,
    batch_size: int,
    rm_c: float,
    eta_reg: float,
) -> None:
    metrics = {
        m: _metric_ci(summary_rows, m)
        for m in (
            "norm_growth_ratio",
            "late_movement",
            "movement_ratio",
            "late_effective_step",
            "effective_step_ratio",
            "late_usage_jsd_mean",
            "late_action_jsd_mean",
        )
    }
    lines = [
        "# Figure 4 RM Dynamics Report",
        "",
        "## Diminishing-Step Diagnostic for Asymptotic Dynamics",
        "",
        "Main Figures 1--3 evaluate finite-horizon adaptation using constant-step updates. "
        "This appendix diagnostic instead runs ESL in the Robbins--Monro regime assumed by Theorem 1.",
        "",
        "## Protocol",
        f"- Method: ESL only; games: `{', '.join(games)}`; seeds: `{seeds}`; horizon: `{horizon}`.",
        "- Environment: same type-shifting protocol as Figure 3.",
        f"- Fixed configuration: `c={rm_c:g}`, `eta_reg={eta_reg:g}`, frozen batch size `M={batch_size}`.",
        "- Step size: `gamma_m = c * (m + 1)^(-0.9)`.",
        "- The schedule analytically satisfies the Robbins--Monro conditions: "
        "`sum gamma_m = infinity` and `sum gamma_m^2 < infinity`.",
        "- Nonzero L2 regularization instantiates the inward-drift condition used in Lemma 1.",
        "- Larger frozen batches reduce gradient noise and better approximate the slow-timescale dynamics assumed in the analysis.",
        "",
        "While the Robbins--Monro schedule guarantees the relevant asymptotic step-size conditions, "
        "the finite-horizon results here should be interpreted as diagnostics of trajectory behavior rather than convergence itself.",
        "",
        "## Gate Checks",
        f"- Overall gate: {'PASS' if gate['pass'] else 'REPORT'}.",
    ]
    for name, ok in gate["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if ok else 'REPORT'}.")
    lines.extend(
        [
            "",
            "## Aggregate Numbers",
        ]
    )
    for metric, st in metrics.items():
        lines.append(f"- `{metric}`: mean `{st['mean']:.4f}`; 95% CI [`{st['ci95_low']:.4f}`, `{st['ci95_high']:.4f}`].")
    lines.extend(
        [
            "",
            "## Interpretation Constraint",
            (
                "This experiment is not meant to beat constant-step ESL on PSR. It is a theorem-alignment diagnostic "
                "checking bounded norms, decreasing update magnitudes, and stable late-window empirical distributions "
                "under Robbins--Monro conditions."
            ),
            (
                "Do not claim convergence to ICT sets. These results are consistent with the dynamical structure implied "
                "by the differential inclusion, but they are not an empirical proof of convergence."
            ),
            "",
            "## Constant-Step vs Robbins--Monro",
            (
                "The Robbins--Monro diagnostic emphasizes slower movement and stabilizing updates, while the constant-step "
                "Figure 3 setting permits persistent movement for adaptive tracking. The two settings are complementary "
                "regimes: asymptotic convergence diagnostics versus finite-horizon adaptive tracking."
            ),
            "",
            "## Caption",
            (
                "\\caption{\\textbf{Diminishing-step diagnostic for ESL dynamics.} "
                "(a) Prototype norms remain bounded under Robbins--Monro updates. "
                "(b) Prototype update magnitudes decrease over slow time. "
                "(c) Belief entropy stabilizes into a recurrent regime rather than collapsing by construction. "
                "(d) Late-window Jensen--Shannon divergence for prototype usage and induced action distributions is small. "
                "This diagnostic uses diminishing step sizes, nonzero L2 regularization, and larger frozen batches to test "
                "trajectory signatures consistent with the dynamical structure implied by the differential inclusion; "
                "it is not a claim of ICT convergence and does not replace the constant-step adaptation results.}"
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure4_rm_dynamics(
    *,
    out_dir: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    games: tuple[str, ...],
    horizon: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rm_c: float = DEFAULT_RM_C,
    eta_reg: float = DEFAULT_ETA_REG,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for game in games:
        task = task_by_key[game]
        for seed in seeds:
            ts, summary = run_rm_dynamics_cell(
                task=task,
                seed=seed,
                horizon=horizon,
                batch_size=batch_size,
                rm_c=rm_c,
                eta_reg=eta_reg,
            )
            records.extend(ts)
            summaries.append(summary)

    figure = manuscript_bundle / "main" / "figures" / "figure4_rm_dynamics.pdf"
    summary_csv = manuscript_bundle / "main" / "tables" / "figure4_rm_dynamics_summary.csv"
    report = manuscript_bundle / "reports" / "figure4_rm_dynamics_report.md"
    timeseries_csv = manuscript_bundle / "reports" / "figure4_rm_dynamics_timeseries.csv"
    gate_json = out_dir / "figure4_rm_dynamics_gate.json"
    _write_csv(summary_csv, summaries, SUMMARY_COLUMNS)
    _write_csv(timeseries_csv, records, TIMESERIES_COLUMNS)
    _plot_figure(records, summaries, figure, rolling_window=rolling_window)
    gate = _gate(summaries)
    gate_json.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        report,
        summaries,
        gate,
        seeds=seeds,
        games=games,
        horizon=horizon,
        batch_size=batch_size,
        rm_c=rm_c,
        eta_reg=eta_reg,
    )
    return {
        "figure": figure,
        "summary_csv": summary_csv,
        "report": report,
        "timeseries_csv": timeseries_csv,
        "gate_json": gate_json,
    }


def _parse_csv_or_space(values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return default
    out: list[str] = []
    for value in values:
        out.extend([x.strip() for x in value.split(",") if x.strip()])
    return tuple(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 4 Robbins--Monro ESL dynamics diagnostics")
    p.add_argument("--out-dir", type=Path, default=Path("runs/figure4_rm_dynamics"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", nargs="*", default=["0,1,2,3,4,5,6,7,8,9"])
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--games", nargs="*", default=list(DEFAULT_GAMES))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--rm-c", type=float, default=DEFAULT_RM_C)
    p.add_argument("--eta-reg", type=float, default=DEFAULT_ETA_REG)
    p.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    args = p.parse_args(argv)
    seeds = [int(x) for x in _parse_csv_or_space(args.seeds, tuple(str(i) for i in range(10)))]
    games = _parse_csv_or_space(args.games, DEFAULT_GAMES)
    outputs = run_figure4_rm_dynamics(
        out_dir=args.out_dir,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        games=games,
        horizon=args.horizon,
        batch_size=args.batch_size,
        rm_c=args.rm_c,
        eta_reg=args.eta_reg,
        rolling_window=args.rolling_window,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
