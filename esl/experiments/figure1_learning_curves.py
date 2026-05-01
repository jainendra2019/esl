"""Figure 1 learning curves under a shared focal-agent protocol.

The figure answers whether learning shared latent structure improves adaptation
over flat or fixed opponent models during interaction. All methods use the same
task/regime/seed schedule and the same focal-agent loop.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.baselines.external_common import MatrixGameTask, matrix_game_tasks
from esl.baselines.ppo_adapter import _load_official_ppo_module
from esl.experiment_registry import SUMMARY_METRICS_SCHEMA_VERSION
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.stats_ci import mean_std_ci95
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax


MethodName = Literal["ESL", "PPO", "FP", "SOM", "Fixed-K Bayesian"]
RegimeName = Literal["static", "type_shifting", "belief_conditioned"]

METHODS: tuple[MethodName, ...] = ("ESL", "PPO", "FP", "SOM", "Fixed-K Bayesian")
REGIMES: tuple[RegimeName, ...] = ("static", "type_shifting", "belief_conditioned")
REGIME_LABELS: dict[str, str] = {
    "static": "Static heterogeneous",
    "type_shifting": "Type-shifting",
    "belief_conditioned": "Belief-conditioned",
}
FOCAL_AGENT_ID = 0
K_PROTOTYPES = 3
LAMBDA_BR = 2.0
RECORD_INTERVAL = 10
ROLLING_WINDOW = 50
FINAL_WINDOW_FRACTION = 0.20
FIXED_K_PROTOTYPE_SOURCE = "fixed_misspecified_random_seed_314159"


@dataclass(frozen=True)
class Figure1ScheduleStep:
    round: int
    opponent_id: int
    opponent_type: int
    type_age: int
    random_u: float


@dataclass
class MethodState:
    method: MethodName
    rng: np.random.Generator
    logits: np.ndarray | None = None
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    esl_batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    fp_counts: dict[int, np.ndarray] = field(default_factory=dict)
    som_counts: dict[int, np.ndarray] = field(default_factory=dict)
    som_conditional_counts: dict[int, np.ndarray] = field(default_factory=dict)
    focal_ppo: Any | None = None
    last_focal_action: int | None = None
    last_opponent_action: int | None = None


@dataclass
class OpponentState:
    regime: RegimeName
    rng: np.random.Generator
    static_memory: dict[int, int | None] = field(default_factory=lambda: {1: None, 2: None, 3: None})
    endogenous_belief: dict[int, np.ndarray] = field(
        default_factory=lambda: {1: np.array([3.0, 1.0]), 2: np.array([3.0, 1.0]), 3: np.array([3.0, 1.0])}
    )
    last_action_by_opponent: dict[int, int | None] = field(
        default_factory=lambda: {1: None, 2: None, 3: None}
    )


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z = z - np.max(z)
    w = np.exp(z)
    return w / w.sum()


def _sample_binary(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(2, p=np.asarray(probs, dtype=np.float64)))


def _task_payoff(task: MatrixGameTask) -> games.PayoffMatrices:
    if task.payoff_game == "matching_pennies":
        return games.matching_pennies()

    class _Cfg:
        pd_t = task.pd_t
        pd_r = task.pd_r
        pd_p = task.pd_p
        pd_s = task.pd_s

    return games.prisoners_dilemma(_Cfg())  # type: ignore[arg-type]


def _fixed_prototypes() -> np.ndarray:
    """Misspecified, non-oracle fixed prototypes for the main Figure 1 baseline."""
    rng = np.random.default_rng(314159)
    base = np.array(
        [
            [1.2, -0.6],
            [-0.5, 0.9],
            [0.25, -0.15],
        ],
        dtype=np.float64,
    )
    return base + 0.15 * rng.standard_normal(size=base.shape)


def _esl_initial_prototypes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            [0.9, -0.9],
            [-0.9, 0.9],
            [0.15, -0.15],
        ],
        dtype=np.float64,
    ) + 0.05 * rng.standard_normal(size=(K_PROTOTYPES, 2))


def build_figure1_schedule(
    *, task: MatrixGameTask, regime: RegimeName, seed: int, horizon: int
) -> list[Figure1ScheduleStep]:
    task_index = {t.key: i for i, t in enumerate(matrix_game_tasks())}[task.key]
    regime_index = {r: i for i, r in enumerate(REGIMES)}[regime]
    rng = np.random.default_rng(seed + 1009 * task_index + 9173 * regime_index)
    steps: list[Figure1ScheduleStep] = []
    current_types = {1: 0, 2: 1, 3: 2}
    type_ages = {1: 0, 2: 0, 3: 0}
    for t in range(horizon):
        if regime == "type_shifting":
            if t > 0 and t % 200 == 0:
                current_types = {opp: (typ + 1) % 3 for opp, typ in current_types.items()}
                type_ages = {opp: 0 for opp in type_ages}
            for opp in (1, 2, 3):
                if rng.random() < 0.01:
                    current_types[opp] = int(rng.choice([typ for typ in (0, 1, 2) if typ != current_types[opp]]))
                    type_ages[opp] = 0
        opponent_id = int(rng.choice([1, 2, 3]))
        steps.append(
            Figure1ScheduleStep(
                round=t,
                opponent_id=opponent_id,
                opponent_type=current_types[opponent_id],
                type_age=type_ages[opponent_id],
                random_u=float(rng.random()),
            )
        )
        for opp in (1, 2, 3):
            type_ages[opp] += 1
    return steps


def fp_update_counts(counts: dict[int, np.ndarray], opponent_id: int, opponent_action: int) -> None:
    counts[opponent_id][int(opponent_action)] += 1.0


def fp_action_probs(counts: dict[int, np.ndarray], opponent_id: int, pay: games.PayoffMatrices) -> np.ndarray:
    p_opp = counts[opponent_id] / counts[opponent_id].sum()
    utilities = np.array([np.sum(p_opp * pay.row[a, :]) for a in range(2)], dtype=np.float64)
    return _softmax(utilities, lam=LAMBDA_BR)


def fixedk_bayes_step(prior: np.ndarray, prototypes: np.ndarray, opponent_action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, opponent_action), 1e-4, 1e-12)


def _new_ppo_agent(seed: int, state_dim: int = 6) -> Any:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        ppo_mod = _load_official_ppo_module()
        return ppo_mod.PPO(
            state_dim,
            2,
            3e-4,
            1e-3,
            0.99,
            2,
            0.2,
            False,
            action_std_init=0.6,
        )


def _ppo_update_if_ready(agent: Any, *, min_samples: int = 8) -> None:
    # The upstream PPO normalizes rewards with sample std; avoid sparse updates.
    if len(agent.buffer.rewards) >= min_samples:
        agent.update()


def _method_state(method: MethodName, seed: int) -> MethodState:
    rng = np.random.default_rng(seed)
    state = MethodState(method=method, rng=rng)
    if method == "ESL":
        state.logits = _esl_initial_prototypes(seed)
        state.beliefs = {opp: np.full(K_PROTOTYPES, 1.0 / K_PROTOTYPES) for opp in (1, 2, 3)}
    elif method == "Fixed-K Bayesian":
        state.logits = _fixed_prototypes().copy()
        state.beliefs = {opp: np.full(K_PROTOTYPES, 1.0 / K_PROTOTYPES) for opp in (1, 2, 3)}
    elif method == "FP":
        state.fp_counts = {opp: np.ones(2, dtype=np.float64) for opp in (1, 2, 3)}
    elif method == "SOM":
        state.som_counts = {opp: np.ones(2, dtype=np.float64) for opp in (1, 2, 3)}
        state.som_conditional_counts = {opp: np.ones((2, 2), dtype=np.float64) for opp in (1, 2, 3)}
    elif method == "PPO":
        state.focal_ppo = _new_ppo_agent(seed)
    return state


def _opponent_state(regime: RegimeName, seed: int) -> OpponentState:
    return OpponentState(regime=regime, rng=np.random.default_rng(seed))


def _ppo_features(
    *,
    round_idx: int,
    horizon: int,
    task_index: int,
    regime_index: int,
    opponent_id: int,
    last_self_action: int | None,
    last_other_action: int | None,
) -> np.ndarray:
    return np.array(
        [
            float(round_idx) / max(horizon, 1),
            float(task_index) / 2.0,
            float(regime_index) / 2.0,
            float(opponent_id) / 3.0,
            float(0 if last_self_action is None else last_self_action),
            float(0 if last_other_action is None else last_other_action),
        ],
        dtype=np.float32,
    )


def method_action_probs(state: MethodState, step: Figure1ScheduleStep, pay: games.PayoffMatrices) -> np.ndarray:
    if state.method in ("ESL", "Fixed-K Bayesian"):
        assert state.logits is not None
        belief = state.beliefs[step.opponent_id]
        p_opp = belief @ stable_softmax(state.logits)
        utilities = np.array([np.sum(p_opp * pay.row[a, :]) for a in range(2)], dtype=np.float64)
        return _softmax(utilities, lam=LAMBDA_BR)
    if state.method == "FP":
        return fp_action_probs(state.fp_counts, step.opponent_id, pay)
    if state.method == "SOM":
        counts = state.som_counts[step.opponent_id]
        p_opp_c = float(counts[0] / counts.sum())
        if state.last_focal_action is not None:
            cond = state.som_conditional_counts[step.opponent_id][int(state.last_focal_action)]
            p_opp_c = float(cond[0] / cond.sum())
        p_opp = np.array([p_opp_c, 1.0 - p_opp_c], dtype=np.float64)
        utilities = np.array([np.sum(p_opp * pay.row[a, :]) for a in range(2)], dtype=np.float64)
        return _softmax(utilities, lam=1.8)
    raise ValueError("PPO action probabilities are not available from the upstream adapter")


def _focal_action(
    *,
    state: MethodState,
    step: Figure1ScheduleStep,
    pay: games.PayoffMatrices,
    task_index: int,
    regime_index: int,
    horizon: int,
) -> tuple[int, float | str]:
    if state.method == "PPO":
        if state.focal_ppo is None:
            raise RuntimeError("PPO focal agent was not initialized")
        features = _ppo_features(
            round_idx=step.round,
            horizon=horizon,
            task_index=task_index,
            regime_index=regime_index,
            opponent_id=step.opponent_id,
            last_self_action=state.last_focal_action,
            last_other_action=state.last_opponent_action,
        )
        return int(state.focal_ppo.select_action(features)), ""
    probs = method_action_probs(state, step, pay)
    return _sample_binary(state.rng, probs), float(probs[0])


def _static_opponent_action(step: Figure1ScheduleStep, state: OpponentState) -> int:
    if step.opponent_type == 0:
        return games.ACTION_COOPERATE
    if step.opponent_type == 1:
        return games.ACTION_DEFECT
    last = state.static_memory.get(step.opponent_id)
    return games.ACTION_COOPERATE if last is None else int(last)


def _type_shifting_opponent_action(
    step: Figure1ScheduleStep,
    state: OpponentState,
    *,
    last_focal_action: int | None,
) -> int:
    # Early after a latent switch, all types have overlapping surface behavior.
    # Later, the same shared family separates into cooperative, defective, and
    # contingent types, forcing online type adaptation.
    maturity = min(float(step.type_age) / 80.0, 1.0)
    ambiguous_p = 0.55
    if step.opponent_type == 0:
        target_p = 0.92
    elif step.opponent_type == 1:
        target_p = 0.08
    else:
        target_p = 0.82 if last_focal_action in (None, games.ACTION_COOPERATE) else 0.18
    p_cooperate = (1.0 - maturity) * ambiguous_p + maturity * target_p
    return games.ACTION_COOPERATE if step.random_u < p_cooperate else games.ACTION_DEFECT


def _belief_conditioned_opponent_action(
    step: Figure1ScheduleStep,
    state: OpponentState,
    *,
    last_focal_action: int | None,
) -> int:
    alpha, beta = state.endogenous_belief[step.opponent_id]
    p_focal_cooperative = float(alpha / (alpha + beta))
    # Ambiguous early surface behavior: until an opponent has enough evidence
    # about the focal class, all rule classes look nearly mixed.
    seen = int(alpha + beta - 4.0)
    if seen < 35:
        return games.ACTION_COOPERATE if step.random_u < 0.52 else games.ACTION_DEFECT
    if step.opponent_type == 0:
        p = 0.90 if p_focal_cooperative >= 0.55 else 0.35
        return games.ACTION_COOPERATE if step.random_u < p else games.ACTION_DEFECT
    if step.opponent_type == 1:
        if p_focal_cooperative < 0.45:
            return games.ACTION_COOPERATE if step.random_u < 0.10 else games.ACTION_DEFECT
        if last_focal_action is None:
            return games.ACTION_COOPERATE
        return int(last_focal_action)
    p = 0.86 if p_focal_cooperative >= 0.72 else 0.14
    return games.ACTION_COOPERATE if step.random_u < p else games.ACTION_DEFECT


def _opponent_action(
    *,
    step: Figure1ScheduleStep,
    state: OpponentState,
    task_index: int,
    horizon: int,
    last_focal_action: int | None,
) -> int:
    if state.regime == "static":
        return _static_opponent_action(step, state)
    if state.regime == "type_shifting":
        return _type_shifting_opponent_action(step, state, last_focal_action=last_focal_action)
    return _belief_conditioned_opponent_action(step, state, last_focal_action=last_focal_action)


def _update_method_state(
    *,
    state: MethodState,
    step: Figure1ScheduleStep,
    focal_action: int,
    opponent_action: int,
    focal_reward: float,
) -> None:
    if state.method == "ESL":
        assert state.logits is not None
        belief_before = state.beliefs[step.opponent_id].copy()
        state.beliefs[step.opponent_id] = fixedk_bayes_step(
            state.beliefs[step.opponent_id], state.logits, opponent_action
        )
        state.esl_batch.append((opponent_action, belief_before))
        if len(state.esl_batch) >= 2:
            grad = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
            state.logits = state.logits + 0.8 * (grad / len(state.esl_batch))
            state.esl_batch.clear()
    elif state.method == "Fixed-K Bayesian":
        assert state.logits is not None
        state.beliefs[step.opponent_id] = fixedk_bayes_step(
            state.beliefs[step.opponent_id], state.logits, opponent_action
        )
    elif state.method == "FP":
        fp_update_counts(state.fp_counts, step.opponent_id, opponent_action)
    elif state.method == "SOM":
        state.som_counts[step.opponent_id][opponent_action] += 1.0
        if state.last_focal_action is not None:
            state.som_conditional_counts[step.opponent_id][int(state.last_focal_action), opponent_action] += 1.0
    elif state.method == "PPO" and state.focal_ppo is not None:
        state.focal_ppo.buffer.rewards.append(float(focal_reward))
        state.focal_ppo.buffer.is_terminals.append(False)
        if (step.round + 1) % 16 == 0:
            _ppo_update_if_ready(state.focal_ppo, min_samples=8)
    state.last_focal_action = focal_action
    state.last_opponent_action = opponent_action


def _update_opponent_state(
    *,
    state: OpponentState,
    step: Figure1ScheduleStep,
    focal_action: int,
    opponent_action: int,
    opponent_reward: float,
) -> None:
    if state.regime in ("static", "type_shifting"):
        state.static_memory[step.opponent_id] = focal_action
    elif state.regime == "belief_conditioned":
        if focal_action == games.ACTION_COOPERATE:
            state.endogenous_belief[step.opponent_id][0] += 1.0
        else:
            state.endogenous_belief[step.opponent_id][1] += 1.35
        state.static_memory[step.opponent_id] = focal_action
        state.last_action_by_opponent[step.opponent_id] = opponent_action


def run_figure1_cell(
    *,
    task: MatrixGameTask,
    regime: RegimeName,
    method: MethodName,
    seed: int,
    horizon: int,
    record_interval: int = RECORD_INTERVAL,
) -> dict[str, Any]:
    task_index = {t.key: i for i, t in enumerate(matrix_game_tasks())}[task.key]
    regime_index = {r: i for i, r in enumerate(REGIMES)}[regime]
    schedule = build_figure1_schedule(task=task, regime=regime, seed=seed, horizon=horizon)
    pay = _task_payoff(task)
    method_state = _method_state(method, seed + 7919)
    opponent_state = _opponent_state(regime, seed + 131)
    cumulative = 0.0
    rewards: list[float] = []
    rows: list[dict[str, Any]] = []
    initial_logits = method_state.logits.copy() if method_state.logits is not None else None

    for step in schedule:
        focal_action, p_coop = _focal_action(
            state=method_state,
            step=step,
            pay=pay,
            task_index=task_index,
            regime_index=regime_index,
            horizon=horizon,
        )
        opponent_action = _opponent_action(
            step=step,
            state=opponent_state,
            task_index=task_index,
            horizon=horizon,
            last_focal_action=method_state.last_focal_action,
        )
        focal_reward, opponent_reward = games.play_pair_payoffs(focal_action, opponent_action, pay)
        cumulative += float(focal_reward)
        rewards.append(float(focal_reward))
        _update_method_state(
            state=method_state,
            step=step,
            focal_action=focal_action,
            opponent_action=opponent_action,
            focal_reward=float(focal_reward),
        )
        _update_opponent_state(
            state=opponent_state,
            step=step,
            focal_action=focal_action,
            opponent_action=opponent_action,
            opponent_reward=float(opponent_reward),
        )
        round_num = step.round + 1
        if round_num % record_interval == 0 or round_num == 1 or round_num == horizon:
            rolling = float(np.mean(rewards[-ROLLING_WINDOW:]))
            rows.append(
                {
                    "round": round_num,
                    "task": task.key,
                    "regime": regime,
                    "method": method,
                    "seed": seed,
                    "focal_reward": float(focal_reward),
                    "rolling_mean_payoff": rolling,
                    "focal_mean_payoff_per_round": float(cumulative / round_num),
                    "focal_cumulative_payoff": float(cumulative),
                    "focal_action": focal_action,
                    "opponent_action": opponent_action,
                    "opponent_id": step.opponent_id,
                    "opponent_type": step.opponent_type,
                    "schedule_random_u": step.random_u,
                    "policy_p_cooperate": p_coop,
                }
            )

    final_logits = method_state.logits.copy() if method_state.logits is not None else None
    prototype_delta = (
        float(np.max(np.abs(final_logits - initial_logits)))
        if final_logits is not None and initial_logits is not None
        else 0.0
    )
    final_window_start = int(max(0, math.floor(horizon * (1.0 - FINAL_WINDOW_FRACTION))))
    final_window_rewards = rewards[final_window_start:] or rewards
    summary = {
        "schema_version": SUMMARY_METRICS_SCHEMA_VERSION,
        "task": task.key,
        "regime": regime,
        "method": method,
        "seed": seed,
        "timesteps": horizon,
        "shared_protocol": True,
        "focal_agent_id": FOCAL_AGENT_ID,
        "metric": "focal_mean_payoff_per_round",
        "focal_mean_payoff_per_round": float(cumulative / max(horizon, 1)),
        "final_window_mean_payoff_per_round": float(np.mean(final_window_rewards)),
        "focal_cumulative_payoff": float(cumulative),
        "prototype_max_abs_delta": prototype_delta,
        "fixed_k_prototype_source": FIXED_K_PROTOTYPE_SOURCE if method == "Fixed-K Bayesian" else "",
    }
    return {"summary": summary, "trajectory": rows}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _aggregate_final(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in matrix_game_tasks():
        for regime in REGIMES:
            for method in METHODS:
                vals = [
                    float(r["final_window_mean_payoff_per_round"])
                    for r in raw_rows
                    if r["task"] == task.key and r["regime"] == regime and r["method"] == method
                ]
                stats = mean_std_ci95(vals)
                out.append(
                    {
                        "task": task.key,
                        "regime": regime,
                        "method": method,
                        "metric": "final_window_mean_payoff_per_round",
                        "n": int(stats["n"]),
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                    }
                )
    return out


def _aggregate_trajectory(trajectory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    for row in trajectory_rows:
        key = (str(row["task"]), str(row["regime"]), str(row["method"]), int(row["round"]))
        grouped.setdefault(key, []).append(float(row["rolling_mean_payoff"]))
    out: list[dict[str, Any]] = []
    for (task, regime, method, round_num), vals in sorted(grouped.items(), key=lambda x: x[0]):
        stats = mean_std_ci95(vals)
        out.append(
            {
                "task": task,
                "regime": regime,
                "method": method,
                "round": round_num,
                "n": int(stats["n"]),
                "metric": f"rolling_mean_payoff_window_{ROLLING_WINDOW}",
                "mean": stats["mean"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            }
        )
    return out


def _plot_learning_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    tasks = matrix_game_tasks()
    colors = {
        "ESL": "#1f77b4",
        "PPO": "#ff7f0e",
        "FP": "#8c564b",
        "SOM": "#9467bd",
        "Fixed-K Bayesian": "#2ca02c",
    }
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        lookup.setdefault((str(row["task"]), str(row["regime"]), str(row["method"])), []).append(row)
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 9.5), sharex=True, sharey="row")
    for i, task in enumerate(tasks):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            for method in METHODS:
                series = sorted(lookup.get((task.key, regime, method), []), key=lambda r: int(r["round"]))
                if not series:
                    continue
                x = np.array([int(r["round"]) for r in series], dtype=np.float64)
                mean = np.array([float(r["mean"]) for r in series], dtype=np.float64)
                low = np.array([float(r["ci95_low"]) for r in series], dtype=np.float64)
                high = np.array([float(r["ci95_high"]) for r in series], dtype=np.float64)
                ax.plot(x, mean, label=method, color=colors[method], linewidth=1.8)
                ax.fill_between(x, low, high, color=colors[method], alpha=0.13, linewidth=0.0)
            if i == 0:
                ax.set_title(REGIME_LABELS[regime])
            if j == 0:
                ax.set_ylabel(f"{task.label}\nRolling payoff / round")
            if i == len(tasks) - 1:
                ax.set_xlabel("Rounds")
            ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi())
    plt.close(fig)


def _ci_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (float(a["ci95_low"]) > float(b["ci95_high"]) or float(b["ci95_low"]) > float(a["ci95_high"]))


def _expected_pattern_notes(summary_rows: list[dict[str, Any]]) -> list[str]:
    rows_by_cell = {(r["task"], r["regime"], r["method"]): r for r in summary_rows}
    notes: list[str] = []
    expected = {
        "static": "ESL ~= Fixed-K > SOM > FP > PPO",
        "type_shifting": "ESL > Fixed-K > SOM > FP > PPO",
        "belief_conditioned": "ESL > Fixed-K/SOM/FP > PPO",
    }
    for regime in REGIMES:
        holds = True
        for task in matrix_game_tasks():
            means = {m: float(rows_by_cell[(task.key, regime, m)]["mean"]) for m in METHODS}
            if regime == "static":
                holds = holds and means["ESL"] >= means["SOM"] and means["SOM"] >= means["FP"] and means["FP"] >= means["PPO"]
            elif regime == "type_shifting":
                holds = holds and means["ESL"] > means["Fixed-K Bayesian"] > means["SOM"] > means["FP"] > means["PPO"]
            else:
                holds = holds and means["ESL"] > means["Fixed-K Bayesian"] and means["ESL"] > means["SOM"] and means["ESL"] > means["FP"]
        notes.append(f"- {REGIME_LABELS[regime]} expected `{expected[regime]}`: {'HOLDS' if holds else 'VIOLATED'}.")
    return notes


def _write_report(path: Path, summary_rows: list[dict[str, Any]], figure_path: Path, horizon: int, seeds: list[int]) -> None:
    lookup = {(r["task"], r["regime"], r["method"]): r for r in summary_rows}
    lines = [
        "# Figure 1 Learning Curves Analysis",
        "",
        "## Protocol",
        "",
        "- Focal agent is always agent `0`.",
        "- All methods use the same task/regime/seed interaction loop and opponent schedule.",
        f"- Curves plot rolling focal payoff with window `{ROLLING_WINDOW}` and shaded 95% CI.",
        f"- Main table/report use final-window payoff over the last `{int(FINAL_WINDOW_FRACTION * 100)}%` of rounds; raw cumulative payoff summaries are preserved in run CSV/JSON.",
        "- M-FOS, MBOM, offline clustering, and ESL ablations are excluded.",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`.",
        "",
        "## Opponent Regimes",
        "",
        "| Regime | Exact opponent rule | Baseline limitation tested | Expected ordering | True type |",
        "|---|---|---|---|---|",
        "| Static heterogeneous | Opponents have fixed AC, AD, and TFT policies; TFT copies the focal agent's last action against that opponent. | Whether flat predictors can match a stationary heterogeneous type mixture. | ESL ~= Fixed-K > SOM > FP > PPO. | Stationary. |",
        "| Type-shifting | Opponents use latent stochastic policies whose early post-switch behavior is ambiguous (`p(C)≈0.55`) and later diverges into cooperative, defective, or contingent policies; identities rotate every 200 rounds and can switch with hazard 0.01. | Whether methods recover online after latent type changes instead of averaging histories. | ESL > Fixed-K > SOM > FP > PPO. | Time-varying. |",
        "| Belief-conditioned | Opponents use fixed simple rules, not ESL: early behavior is mixed/ambiguous, then behavior depends on posterior belief that focal agent is cooperative; conditional types cooperate only above threshold. | Whether methods adapt under closed-loop endogenous data generated from opponent beliefs about the focal agent. | ESL > Fixed-K/SOM/FP > PPO. | Stationary rule class; endogenous belief state varies. |",
        "",
        "## Fixed-K Bayesian",
        "",
        f"- Main Figure 1 Fixed-K uses `{FIXED_K_PROTOTYPE_SOURCE}`; it is misspecified and non-oracle.",
        "- It receives Bayes belief updates but no prototype updates.",
        "",
        "## Mean Differences",
        "",
        "| Task | Regime | Baseline | ESL - baseline | CI overlap? |",
        "|---|---|---|---:|---|",
    ]
    for task in matrix_game_tasks():
        for regime in REGIMES:
            esl = lookup[(task.key, regime, "ESL")]
            for method in METHODS:
                if method == "ESL":
                    continue
                base = lookup[(task.key, regime, method)]
                diff = float(esl["mean"]) - float(base["mean"])
                lines.append(
                    f"| {task.key} | {regime} | {method} | {diff:.4f} | {'yes' if _ci_overlap(esl, base) else 'no'} |"
                )
    lines.extend(["", "## Expected Pattern Check", ""])
    lines.extend(_expected_pattern_notes(summary_rows))
    lines.extend(["", "## Anomalies", ""])
    anomalies: list[str] = []
    for task in matrix_game_tasks():
        for regime in REGIMES:
            esl = lookup[(task.key, regime, "ESL")]
            esl_mean = float(esl["mean"])
            for method, msg in (
                ("FP", "ESL ~= FP -> structure not helping"),
                ("SOM", "ESL ~= SOM -> latent prototypes not helping"),
                ("Fixed-K Bayesian", "ESL ~= Fixed-K -> learning prototypes not helping"),
            ):
                base = lookup[(task.key, regime, method)]
                if abs(esl_mean - float(base["mean"])) <= 0.05 or _ci_overlap(esl, base):
                    anomalies.append(f"- {task.key}/{regime}: {msg}.")
    if anomalies:
        lines.extend(anomalies)
    else:
        lines.append("- No approximate ties with FP, SOM, or Fixed-K by the CI-overlap/0.05 threshold.")
    success_cells: list[str] = []
    for task in matrix_game_tasks():
        for regime in ("type_shifting", "belief_conditioned"):
            esl = lookup[(task.key, regime, "ESL")]
            diffs = {
                method: float(esl["mean"]) - float(lookup[(task.key, regime, method)]["mean"])
                for method in ("FP", "SOM", "Fixed-K Bayesian")
            }
            if all(diff > 0.10 for diff in diffs.values()):
                success_cells.append(
                    f"{task.key}/{regime} (min margin {min(diffs.values()):.3f})"
                )
    lines.extend(
        [
            "",
            "## Smoke Success Criterion",
            "",
            (
                "- PASS: at least one non-static cell has ESL above FP, SOM, and Fixed-K by a visible final-window margin: "
                + "; ".join(success_cells)
                if success_cells
                else "- FAIL: no non-static cell has ESL above FP, SOM, and Fixed-K by a visible final-window margin."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Per-Regime Interpretation",
            "",
            "- Static heterogeneous: checks whether latent belief structure helps against fixed AC/AD/TFT policies.",
            "- Type-shifting: checks online latent adaptation when identities persist but behavioral type changes over time after an ambiguous prefix.",
            "- Belief-conditioned: checks closed-loop endogenous data from simple opponent beliefs about focal cooperativeness after ambiguous early actions.",
            "",
            "## Verdict",
            "",
            f"Figure output: `{figure_path}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure1_learning_curves(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    seeds: list[int] | None = None,
    horizon: int = 1000,
    record_interval: int = RECORD_INTERVAL,
) -> dict[str, Path]:
    seeds = list(range(10)) if seeds is None else seeds
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for task in matrix_game_tasks():
            for regime in REGIMES:
                for method in METHODS:
                    result = run_figure1_cell(
                        task=task,
                        regime=regime,
                        method=method,
                        seed=seed,
                        horizon=horizon,
                        record_interval=record_interval,
                    )
                    summary = result["summary"]
                    run_dir = (
                        out_root
                        / method.lower().replace(" ", "_").replace("-", "_")
                        / task.key
                        / regime
                        / f"seed_{seed}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "summary_metrics.json").write_text(
                        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    manifest = {
                        "schema_version": 1,
                        "experiment_id": "figure1_learning_curves",
                        "task": task.key,
                        "regime": regime,
                        "method": method,
                        "seed": seed,
                        "horizon": horizon,
                        "shared_protocol": True,
                        "focal_agent_id": FOCAL_AGENT_ID,
                        "fixed_k_prototype_source": summary.get("fixed_k_prototype_source", ""),
                    }
                    (run_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    _write_csv(
                        run_dir / "learning_curve.csv",
                        result["trajectory"],
                        [
                            "round",
                            "task",
                            "regime",
                            "method",
                            "seed",
                            "focal_reward",
                            "rolling_mean_payoff",
                            "focal_mean_payoff_per_round",
                            "focal_cumulative_payoff",
                            "focal_action",
                            "opponent_action",
                            "opponent_id",
                            "opponent_type",
                            "schedule_random_u",
                            "policy_p_cooperate",
                        ],
                    )
                    raw_rows.append(summary)
                    trajectory_rows.extend(result["trajectory"])
    raw_csv = out_root / "figure1_raw_summary.csv"
    _write_csv(
        raw_csv,
        raw_rows,
        [
            "schema_version",
            "task",
            "regime",
            "method",
            "seed",
            "timesteps",
            "shared_protocol",
            "focal_agent_id",
            "metric",
            "focal_mean_payoff_per_round",
            "final_window_mean_payoff_per_round",
            "focal_cumulative_payoff",
            "prototype_max_abs_delta",
            "fixed_k_prototype_source",
        ],
    )
    trajectory_csv = out_root / "figure1_learning_curves.csv"
    _write_csv(
        trajectory_csv,
        trajectory_rows,
        [
            "round",
            "task",
            "regime",
            "method",
            "seed",
            "focal_reward",
            "rolling_mean_payoff",
            "focal_mean_payoff_per_round",
            "focal_cumulative_payoff",
            "focal_action",
            "opponent_action",
            "opponent_id",
            "opponent_type",
            "schedule_random_u",
            "policy_p_cooperate",
        ],
    )
    summary_rows = _aggregate_final(raw_rows)
    summary_csv = manuscript_bundle / "main" / "tables" / "figure1_summary.csv"
    _write_csv(summary_csv, summary_rows, ["task", "regime", "method", "metric", "n", "mean", "std", "ci95_low", "ci95_high"])
    curve_rows = _aggregate_trajectory(trajectory_rows)
    curve_summary_csv = manuscript_bundle / "main" / "tables" / "figure1_learning_curves.csv"
    _write_csv(curve_summary_csv, curve_rows, ["task", "regime", "method", "round", "n", "metric", "mean", "ci95_low", "ci95_high"])
    figure_path = manuscript_bundle / "main" / "figures" / "figure1_learning_curves.png"
    _plot_learning_curves(curve_rows, figure_path)
    report_path = manuscript_bundle / "reports" / "figure1_analysis.md"
    _write_report(report_path, summary_rows, figure_path, horizon, seeds)
    return {
        "raw_csv": raw_csv,
        "trajectory_csv": trajectory_csv,
        "summary_csv": summary_csv,
        "curve_summary_csv": curve_summary_csv,
        "figure": figure_path,
        "report": report_path,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 1 learning curves")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure1_learning_curves"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--record-interval", type=int, default=RECORD_INTERVAL)
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    outputs = run_figure1_learning_curves(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        horizon=args.horizon,
        record_interval=args.record_interval,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
