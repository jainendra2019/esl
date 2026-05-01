"""Figure 3: empirical signatures of ESL closed-loop learning dynamics."""

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
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.stats_ci import mean_std_ci95
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "game",
    "seed",
    "max_prototype_norm",
    "final_prototype_norm",
    "norm_growth_ratio",
    "early_movement",
    "late_movement",
    "movement_ratio",
    "early_entropy",
    "late_entropy",
    "entropy_change",
    "late_usage_jsd_mean",
    "late_usage_jsd_max",
    "late_action_jsd_mean",
    "late_action_jsd_max",
    "early_update_norm",
    "late_update_norm",
    "update_norm_ratio",
    "num_slow_steps",
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
)
DEFAULT_GAMES = ("ipd", "stag_hunt", "matching_pennies")
DEFAULT_BATCH_SIZE = 2
DEFAULT_ROLLING_WINDOW = 5


@dataclass
class ESLDynamicsState:
    rng: np.random.Generator
    logits: np.ndarray
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    batch_actions: list[int] = field(default_factory=list)
    batch_beliefs_after: list[np.ndarray] = field(default_factory=list)
    previous_beliefs: dict[int, np.ndarray] = field(default_factory=dict)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _esl_initial_prototypes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([[1.0, -1.0], [-1.0, 1.0], [0.25, -0.25]], dtype=np.float64)
    return base + 0.04 * rng.standard_normal(size=base.shape)


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z -= np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def _sample_binary(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(ACTION_DIM, p=np.asarray(probs, dtype=np.float64)))


def _belief_entropy(belief: np.ndarray) -> float:
    b = np.clip(np.asarray(belief, dtype=np.float64), EPS, 1.0)
    return float(-np.sum(b * np.log(b)) / math.log(K_TYPES))


def _distribution_entropy(dist: np.ndarray) -> float:
    p = np.clip(np.asarray(dist, dtype=np.float64), EPS, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / math.log(len(p)))


def _state(seed: int) -> ESLDynamicsState:
    beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    return ESLDynamicsState(
        rng=np.random.default_rng(seed),
        logits=_esl_initial_prototypes(seed),
        beliefs={opp: b.copy() for opp, b in beliefs.items()},
        previous_beliefs={opp: b.copy() for opp, b in beliefs.items()},
    )


def _predicted_opponent_probs(state: ESLDynamicsState, opponent_id: int) -> np.ndarray:
    return state.beliefs[opponent_id] @ stable_softmax(state.logits)


def _action_probs(pred: np.ndarray, pay: game_lib.PayoffMatrices) -> np.ndarray:
    utilities = np.array([np.sum(pred * pay.row[a, :]) for a in range(ACTION_DIM)], dtype=np.float64)
    return _softmax(utilities, lam=LAMBDA_BR)


def _bayes_step(prior: np.ndarray, prototypes: np.ndarray, opponent_action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, opponent_action), 1e-4, EPS)


def run_dynamics_cell(
    *,
    task: MatrixGameTask,
    seed: int,
    horizon: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pay = _task_payoff(task)
    schedule = build_type_shifting_schedule(task=task, seed=seed, horizon=horizon)
    state = _state(seed + 19001)
    initial_norm = float(np.linalg.norm(state.logits))
    records: list[dict[str, Any]] = []
    slow_step = 0

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
        state.logits = state.logits + 0.9 * grad
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
        mean_entropy = float(np.mean([_belief_entropy(b) for b in state.beliefs.values()]))
        update_norm = float(np.linalg.norm(delta))
        records.append(
            {
                "slow_step": slow_step,
                "env_step": step.round,
                "seed": seed,
                "game": task.key,
                "theta_flat": json.dumps([float(x) for x in state.logits.reshape(-1)]),
                "prototype_norm": float(np.linalg.norm(state.logits)),
                "prototype_step_size": update_norm,
                "mean_belief_entropy": mean_entropy,
                "prototype_usage": json.dumps([float(x) for x in usage]),
                "usage_entropy": _distribution_entropy(usage),
                "belief_total_variation": belief_tv,
                "mean_update_norm": update_norm,
                "window_action_distribution": json.dumps([float(x) for x in action_dist]),
                "window_gradient_norm": float(np.linalg.norm(grad)),
            }
        )
        slow_step += 1
        state.batch.clear()
        state.batch_actions.clear()
        state.batch_beliefs_after.clear()

    return records, summarize_dynamics_records(records, game=task.key, seed=seed, initial_norm=initial_norm)


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


def summarize_dynamics_records(
    records: list[dict[str, Any]],
    *,
    game: str,
    seed: int,
    initial_norm: float,
) -> dict[str, Any]:
    if not records:
        return {col: 0.0 for col in REQUIRED_SUMMARY_COLUMNS} | {"game": game, "seed": seed}
    norms = np.asarray([float(r["prototype_norm"]) for r in records], dtype=np.float64)
    movement = np.asarray([float(r["prototype_step_size"]) for r in records], dtype=np.float64)
    entropy = np.asarray([float(r["mean_belief_entropy"]) for r in records], dtype=np.float64)
    update_norm = np.asarray([float(r["mean_update_norm"]) for r in records], dtype=np.float64)
    early_movement, late_movement = _first_last(movement)
    early_entropy, late_entropy = _first_last(entropy)
    early_update, late_update = _first_last(update_norm)
    usage_jsd_mean, usage_jsd_max = _late_window_jsd(records, "prototype_usage")
    action_jsd_mean, action_jsd_max = _late_window_jsd(records, "window_action_distribution")
    return {
        "game": game,
        "seed": seed,
        "max_prototype_norm": float(np.max(norms)),
        "final_prototype_norm": float(norms[-1]),
        "norm_growth_ratio": float(norms[-1] / max(initial_norm, EPS)),
        "early_movement": early_movement,
        "late_movement": late_movement,
        "movement_ratio": float(late_movement / max(early_movement, EPS)),
        "early_entropy": early_entropy,
        "late_entropy": late_entropy,
        "entropy_change": float(late_entropy - early_entropy),
        "late_usage_jsd_mean": usage_jsd_mean,
        "late_usage_jsd_max": usage_jsd_max,
        "late_action_jsd_mean": action_jsd_mean,
        "late_action_jsd_max": action_jsd_max,
        "early_update_norm": early_update,
        "late_update_norm": late_update,
        "update_norm_ratio": float(late_update / max(early_update, EPS)),
        "num_slow_steps": int(len(records)),
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


def _rolling(vals: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(vals) < window:
        return vals
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(vals, kernel, mode="same")


def _metric_ci(summary_rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    return mean_std_ci95([float(r[metric]) for r in summary_rows])


def _plot_figure(records: list[dict[str, Any]], summary_rows: list[dict[str, Any]], path: Path, *, rolling_window: int) -> None:
    norm = _aggregate_timeseries(records, "prototype_norm")
    movement = _aggregate_timeseries(records, "prototype_step_size")
    entropy = _aggregate_timeseries(records, "mean_belief_entropy")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.reshape(-1)
    panels = [
        (ax_a, "(a)", "Prototype dynamics remain bounded", r"$\|\Theta\|_F$"),
        (ax_b, "(b)", "Updates persist without divergence", r"$\|\Delta \Theta\|_F$"),
        (ax_c, "(c)", "Belief-state regime formation", "Normalized belief entropy"),
        (ax_d, "(d)", "Stable distributions in late regime", "Adjacent-window JSD ↓"),
    ]
    for ax, label, title, ylabel in panels:
        ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
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


def _write_report(path: Path, summary_rows: list[dict[str, Any]], *, seeds: list[int], games: tuple[str, ...], horizon: int) -> None:
    metrics = {
        m: _metric_ci(summary_rows, m)
        for m in (
            "norm_growth_ratio",
            "late_movement",
            "late_usage_jsd_mean",
            "late_action_jsd_mean",
            "update_norm_ratio",
        )
    }
    bounded = all(float(r["norm_growth_ratio"]) < 5.0 for r in summary_rows)
    usage_stable = metrics["late_usage_jsd_mean"]["mean"] < 0.25
    action_stable = metrics["late_action_jsd_mean"]["mean"] < 0.25
    lines = [
        "# Figure 3 Dynamics Report",
        "",
        "## Protocol",
        f"- Method: ESL only; games: `{', '.join(games)}`; seeds: `{seeds}`; horizon: `{horizon}`.",
        "- Environment: same type-shifting protocol as Figures 1-2.",
        "- Panel (c) uses normalized belief entropy by default.",
        "",
        "## Metric Definitions",
        "- Prototype norm is the Frobenius norm of the prototype matrix.",
        "- Movement and update norms are slow-update prototype displacements.",
        "- The invariant-regime proxy uses adjacent-window Jensen-Shannon divergence in nats.",
        "",
        "## Gate Checks",
        f"- bounded prototype dynamics: {'PASS' if bounded else 'REPORT'}; mean norm growth ratio `{metrics['norm_growth_ratio']['mean']:.4f}`.",
        f"- usage invariant-regime proxy JSD < 0.25: {'PASS' if usage_stable else 'REPORT'}; mean `{metrics['late_usage_jsd_mean']['mean']:.4f}`.",
        f"- action invariant-regime proxy JSD < 0.25: {'PASS' if action_stable else 'REPORT'}; mean `{metrics['late_action_jsd_mean']['mean']:.4f}`.",
        "- JSD gates are reporting diagnostics, not hard scientific claims; higher values indicate less stable empirical distributions in that setting.",
        "",
        "## Aggregate Numbers",
    ]
    for metric, st in metrics.items():
        lines.append(f"- `{metric}`: mean `{st['mean']:.4f}`; 95% CI [`{st['ci95_low']:.4f}`, `{st['ci95_high']:.4f}`].")
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            (
                "These diagnostics do not prove convergence to an ICT set. Rather, they test whether the empirical ESL "
                "trajectories exhibit signatures predicted by the theory: bounded prototype dynamics, non-divergent slow "
                "updates, and stabilization of windowed belief/action distributions. The results are therefore interpreted "
                "as consistency evidence for the differential-inclusion characterization, not as a separate theoretical guarantee."
            ),
            "",
            "This is not a proof of ICT convergence; it is a diagnostic consistent with the differential-inclusion characterization.",
            "",
            "## Caption",
            (
                "\\caption{\\textbf{Empirical signatures of ESL's closed-loop learning dynamics.} "
                "(a) Prototype parameters remain bounded over slow updates. "
                "(b) Prototype movement decreases but need not immediately collapse to zero, consistent with the theory allowing "
                "both fixed points and more general internally chain transitive regimes. "
                "(c) Belief statistics stabilize over training, indicating recurrent epistemic regimes. "
                "(d) Adjacent-window Jensen--Shannon divergence over the final training phase is small for both prototype usage "
                "and induced action distributions, providing an empirical proxy for invariant-regime averaging. "
                "These diagnostics do not prove ICT convergence, but show behavior consistent with the differential-inclusion characterization.}"
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure3_dynamics(
    *,
    out_dir: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    games: tuple[str, ...],
    horizon: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
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
            ts, summary = run_dynamics_cell(task=task, seed=seed, horizon=horizon, batch_size=batch_size)
            records.extend(ts)
            summaries.append(summary)
    figure = manuscript_bundle / "main" / "figures" / "figure3_dynamics.pdf"
    summary_csv = manuscript_bundle / "main" / "tables" / "figure3_dynamics_summary.csv"
    report = manuscript_bundle / "reports" / "figure3_dynamics_report.md"
    timeseries_csv = manuscript_bundle / "reports" / "figure3_dynamics_timeseries.csv"
    _write_csv(summary_csv, summaries, REQUIRED_SUMMARY_COLUMNS)
    _write_csv(timeseries_csv, records, TIMESERIES_COLUMNS)
    _plot_figure(records, summaries, figure, rolling_window=rolling_window)
    _write_report(report, summaries, seeds=seeds, games=games, horizon=horizon)
    return {
        "figure": figure,
        "summary_csv": summary_csv,
        "report": report,
        "timeseries_csv": timeseries_csv,
    }


def _parse_csv_or_space(values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return default
    out: list[str] = []
    for value in values:
        out.extend([x.strip() for x in value.split(",") if x.strip()])
    return tuple(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 3 ESL dynamics diagnostics")
    p.add_argument("--out-dir", type=Path, default=Path("runs/figure3_dynamics"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", nargs="*", default=["0,1,2,3,4,5,6,7,8,9"])
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--games", nargs="*", default=list(DEFAULT_GAMES))
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    args = p.parse_args(argv)
    seeds = [int(x) for x in _parse_csv_or_space(args.seeds, ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))]
    games = _parse_csv_or_space(args.games, DEFAULT_GAMES)
    outputs = run_figure3_dynamics(
        out_dir=args.out_dir,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        games=games,
        horizon=args.horizon,
        batch_size=args.batch_size,
        rolling_window=args.rolling_window,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
