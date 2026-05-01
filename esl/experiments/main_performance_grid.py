"""Main-paper performance grid smoke milestone.

Rows are tasks, columns are opponent regimes, and bars are methods. This module
intentionally avoids MCE/recovery diagnostics and only reports payoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.baselines.external_common import (
    BaselineAdapter,
    BaselineSpec,
    MatrixGameTask,
    OpponentRegime,
    matrix_game_tasks,
    opponent_regimes,
    task_config,
)
from esl.baselines.mbom_adapter import build_adapter as build_mbom_adapter
from esl.baselines.mfos_adapter import build_adapter as build_mfos_adapter
from esl.baselines.ppo_adapter import build_adapter as build_ppo_adapter
from esl.baselines.simple_opponent_adapter import build_adapter as build_simple_opponent_adapter
from esl.experiment_registry import SUMMARY_METRICS_SCHEMA_VERSION, build_esl_manifest_dict
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout, write_manifest_json
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.schema import validate_baseline_policy_smoke_directory
from esl.experiments.stats_ci import mean_std_ci95
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax
from esl.trainer import run_esl


def external_adapters() -> list[BaselineAdapter]:
    return [
        build_ppo_adapter(),
        build_mfos_adapter(),
        build_mbom_adapter(),
        build_simple_opponent_adapter(),
    ]


FOCAL_AGENT_ID = 0
SUPPORTED_SHARED_METHODS: tuple[str, ...] = (
    "ESL",
    "ESL K=1",
    "ESL without belief updates",
    "PPO",
    "SOM",
)
SKIPPED_SHARED_METHODS: dict[str, str] = {
    "M-FOS": "SKIPPED: official M-FOS meta-policy logic is not wrapped in the shared focal-agent protocol.",
    "MBOM": "SKIPPED: official MBOM recursive imagination/environment-model stack is not wrapped in the shared focal-agent protocol.",
}


@dataclass(frozen=True)
class SharedScheduleStep:
    round: int
    opponent_id: int
    opponent_type: int
    random_u: float


@dataclass
class SharedPolicyState:
    method: str
    rng: np.random.Generator
    logits: np.ndarray | None = None
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    esl_batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    opponent_counts: dict[int, np.ndarray] = field(default_factory=dict)
    conditional_counts: dict[int, np.ndarray] = field(default_factory=dict)
    last_focal_action: int | None = None
    last_opponent_action: int | None = None
    focal_action_count: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ppo_agent: Any | None = None


def _is_esl_method(method: str) -> bool:
    return method in {"ESL", "ESL K=1", "ESL without belief updates"}


def _payoff_for_task(task: MatrixGameTask) -> games.PayoffMatrices:
    if task.payoff_game == "matching_pennies":
        return games.matching_pennies()
    cfg = task_config(task, opponent_regimes()[0], seed=0, horizon=1)
    return games.prisoners_dilemma(cfg)


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z = z - np.max(z)
    w = np.exp(z)
    return w / w.sum()


def _sample_logit_action(rng: np.random.Generator, utilities: np.ndarray, lam: float) -> int:
    return int(rng.choice(2, p=_softmax(utilities, lam=lam)))


def build_shared_opponent_schedule(
    *, task: MatrixGameTask, regime: OpponentRegime, seed: int, horizon: int
) -> list[SharedScheduleStep]:
    task_index = {t.key: i for i, t in enumerate(matrix_game_tasks())}[task.key]
    regime_index = {r.key: i for i, r in enumerate(opponent_regimes())}[regime.key]
    rng = np.random.default_rng(seed + 1009 * task_index + 9173 * regime_index)
    steps: list[SharedScheduleStep] = []
    for t in range(horizon):
        opponent_id = int(rng.choice([1, 2, 3]))
        # Agent 1 = AC, agent 2 = AD, agent 3 = TFT-like.
        opponent_type = {1: 0, 2: 1, 3: 2}[opponent_id]
        steps.append(
            SharedScheduleStep(
                round=t,
                opponent_id=opponent_id,
                opponent_type=opponent_type,
                random_u=float(rng.random()),
            )
        )
    return steps


def _opponent_action_shared(
    *,
    task: MatrixGameTask,
    regime: OpponentRegime,
    step: SharedScheduleStep,
    focal_history: list[int],
    opponent_memory: dict[int, int | None],
) -> int:
    if regime.key == "fixed_types":
        if step.opponent_type == 0:
            return games.ACTION_COOPERATE
        if step.opponent_type == 1:
            return games.ACTION_DEFECT
        last = opponent_memory.get(step.opponent_id)
        return games.ACTION_COOPERATE if last is None else int(last)

    coop_rate = (float(sum(1 for a in focal_history if a == games.ACTION_COOPERATE)) + 1.0) / (
        len(focal_history) + 2.0
    )
    pay = _payoff_for_task(task)
    if regime.key == "adaptive_agents":
        if task.key == "matching_pennies":
            # Column wants to mismatch the row player.
            p_col_cooperate = 1.0 - coop_rate
        else:
            p_row = np.array([coop_rate, 1.0 - coop_rate], dtype=np.float64)
            util = np.array([np.sum(p_row * pay.col[:, a]) for a in range(2)], dtype=np.float64)
            p_col_cooperate = float(_softmax(util, lam=1.2)[games.ACTION_COOPERATE])
        return games.ACTION_COOPERATE if step.random_u < p_col_cooperate else games.ACTION_DEFECT

    if regime.key == "belief_conditioned_agents":
        if task.key == "matching_pennies":
            p_col_cooperate = 1.0 - coop_rate
        else:
            # A simple agreed belief-conditioned/logit response to the focal empirical policy.
            p_row = np.array([coop_rate, 1.0 - coop_rate], dtype=np.float64)
            util = np.array([np.sum(p_row * pay.col[:, a]) for a in range(2)], dtype=np.float64)
            p_col_cooperate = float(_softmax(util, lam=1.8)[games.ACTION_COOPERATE])
        return games.ACTION_COOPERATE if step.random_u < p_col_cooperate else games.ACTION_DEFECT

    raise ValueError(f"unknown regime: {regime.key}")


def _init_shared_policy_state(method: str, *, seed: int, smoke: bool) -> SharedPolicyState:
    rng = np.random.default_rng(seed)
    state = SharedPolicyState(method=method, rng=rng)
    if _is_esl_method(method):
        k = 1 if method == "ESL K=1" else 2
        state.logits = 0.05 * rng.standard_normal(size=(k, 2))
        state.beliefs = {opp: np.full(k, 1.0 / k, dtype=np.float64) for opp in (1, 2, 3)}
    elif method == "SOM":
        state.opponent_counts = {opp: np.ones(2, dtype=np.float64) for opp in (1, 2, 3)}
        state.conditional_counts = {
            opp: np.ones((2, 2), dtype=np.float64) for opp in (1, 2, 3)
        }
    elif method == "PPO":
        from esl.baselines.ppo_adapter import _load_official_ppo_module

        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        ppo_mod = _load_official_ppo_module()
        state.ppo_agent = ppo_mod.PPO(
            5,
            2,
            3e-4,
            1e-3,
            0.99,
            2 if smoke else 4,
            0.2,
            False,
            action_std_init=0.6,
        )
    return state


def _act_shared_method(
    *,
    state: SharedPolicyState,
    task: MatrixGameTask,
    regime: OpponentRegime,
    step: SharedScheduleStep,
    horizon: int,
) -> tuple[int, dict[str, Any]]:
    pay = _payoff_for_task(task)
    if _is_esl_method(state.method):
        assert state.logits is not None
        belief = state.beliefs[step.opponent_id]
        p_opp = belief @ stable_softmax(state.logits)
        utilities = np.array([np.sum(p_opp * pay.row[a, :]) for a in range(2)], dtype=np.float64)
        action = _sample_logit_action(state.rng, utilities, lam=2.0)
        return action, {"policy_p_cooperate": float(_softmax(utilities, lam=2.0)[0])}

    if state.method == "SOM":
        counts = state.opponent_counts[step.opponent_id]
        p_opp_c = float(counts[0] / counts.sum())
        if state.last_focal_action is not None:
            cond = state.conditional_counts[step.opponent_id][int(state.last_focal_action)]
            p_opp_c = float(cond[0] / cond.sum())
        p_opp = np.array([p_opp_c, 1.0 - p_opp_c], dtype=np.float64)
        utilities = np.array([np.sum(p_opp * pay.row[a, :]) for a in range(2)], dtype=np.float64)
        action = _sample_logit_action(state.rng, utilities, lam=1.8)
        return action, {"policy_p_cooperate": float(_softmax(utilities, lam=1.8)[0])}

    if state.method == "PPO":
        if state.ppo_agent is None:
            raise RuntimeError("PPO agent was not initialized")
        state_vec = np.array(
            [
                float(step.round) / max(horizon, 1),
                float(step.opponent_id) / 3.0,
                float(state.last_opponent_action if state.last_opponent_action is not None else 0),
                float(state.last_focal_action if state.last_focal_action is not None else 0),
                {"fixed_types": 0.0, "adaptive_agents": 0.5, "belief_conditioned_agents": 1.0}[regime.key],
            ],
            dtype=np.float32,
        )
        action = int(state.ppo_agent.select_action(state_vec))
        return action, {"policy_p_cooperate": ""}

    raise ValueError(f"unsupported shared method: {state.method}")


def _update_shared_method(
    *,
    state: SharedPolicyState,
    step: SharedScheduleStep,
    focal_action: int,
    opponent_action: int,
    focal_reward: float,
    smoke: bool,
) -> None:
    if _is_esl_method(state.method):
        assert state.logits is not None
        belief_before = state.beliefs[step.opponent_id].copy()
        if state.method != "ESL without belief updates":
            state.beliefs[step.opponent_id] = belief_ops.update_belief_pair(
                state.beliefs[step.opponent_id],
                likelihoods(state.logits, opponent_action),
                1e-4,
                1e-12,
            )
        state.esl_batch.append((opponent_action, belief_before))
        if len(state.esl_batch) >= 2:
            g = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                g += batch_weighted_prototype_gradient(state.logits, belief, signal)
            state.logits = state.logits + 0.8 * (g / max(len(state.esl_batch), 1))
            state.esl_batch.clear()
    elif state.method == "SOM":
        state.opponent_counts[step.opponent_id][opponent_action] += 1.0
        if state.last_focal_action is not None:
            state.conditional_counts[step.opponent_id][int(state.last_focal_action), opponent_action] += 1.0
    elif state.method == "PPO":
        if state.ppo_agent is not None:
            state.ppo_agent.buffer.rewards.append(float(focal_reward))
            state.ppo_agent.buffer.is_terminals.append(False)
            update_every = 4 if smoke else 16
            if (step.round + 1) % update_every == 0:
                state.ppo_agent.update()
    state.focal_action_count[focal_action] += 1.0
    state.last_focal_action = focal_action
    state.last_opponent_action = opponent_action


def run_shared_focal_protocol(
    *,
    method: str,
    task: MatrixGameTask,
    regime: OpponentRegime,
    seed: int,
    horizon: int,
    smoke: bool,
) -> dict[str, Any]:
    if method in SKIPPED_SHARED_METHODS:
        return {
            "status": "SKIPPED",
            "skip_reason": SKIPPED_SHARED_METHODS[method],
            "rows": [],
            "summary": {
                "task": task.key,
                "regime": regime.key,
                "method": method,
                "seed": seed,
                "shared_protocol": True,
                "focal_agent_id": FOCAL_AGENT_ID,
                "baseline_fidelity": "skipped",
            },
        }

    schedule = build_shared_opponent_schedule(task=task, regime=regime, seed=seed, horizon=horizon)
    state = _init_shared_policy_state(method, seed=seed + 7919, smoke=smoke)
    pay = _payoff_for_task(task)
    opponent_memory: dict[int, int | None] = {1: None, 2: None, 3: None}
    focal_history: list[int] = []
    rows: list[dict[str, Any]] = []
    cumulative = 0.0

    for step in schedule:
        focal_action, policy_info = _act_shared_method(
            state=state, task=task, regime=regime, step=step, horizon=horizon
        )
        opponent_action = _opponent_action_shared(
            task=task,
            regime=regime,
            step=step,
            focal_history=focal_history,
            opponent_memory=opponent_memory,
        )
        focal_reward, opponent_reward = games.play_pair_payoffs(focal_action, opponent_action, pay)
        cumulative += float(focal_reward)
        _update_shared_method(
            state=state,
            step=step,
            focal_action=focal_action,
            opponent_action=opponent_action,
            focal_reward=float(focal_reward),
            smoke=smoke,
        )
        opponent_memory[step.opponent_id] = focal_action
        focal_history.append(focal_action)
        rows.append(
            {
                "round": step.round,
                "focal_agent_id": FOCAL_AGENT_ID,
                "opponent_id": step.opponent_id,
                "opponent_type": step.opponent_type,
                "task": task.key,
                "regime": regime.key,
                "method": method,
                "focal_action": focal_action,
                "opponent_action": opponent_action,
                "focal_reward": float(focal_reward),
                "opponent_reward": float(opponent_reward),
                "focal_cumulative_payoff": float(cumulative),
                "focal_mean_payoff_per_round": float(cumulative / (step.round + 1)),
                "schedule_random_u": step.random_u,
                "policy_p_cooperate": policy_info.get("policy_p_cooperate", ""),
            }
        )

    fidelity = {
        "ESL": "official",
        "ESL K=1": "ablation",
        "ESL without belief updates": "ablation",
        "PPO": "official",
        "SOM": "reduced",
    }[method]
    summary = {
        "schema_version": SUMMARY_METRICS_SCHEMA_VERSION,
        "run_kind": "esl" if _is_esl_method(method) else "baseline",
        "task": task.key,
        "regime": regime.key,
        "method": method,
        "seed": seed,
        "timesteps": horizon,
        "shared_protocol": True,
        "focal_agent_id": FOCAL_AGENT_ID,
        "baseline_fidelity": fidelity,
        "focal_cumulative_payoff": float(cumulative),
        "focal_mean_payoff_per_round": float(cumulative / max(horizon, 1)),
        "focal_cooperate_rate": float(state.focal_action_count[0] / max(horizon, 1)),
        "status": "READY" if method in ("ESL", "ESL K=1", "ESL without belief updates", "PPO") else "APPENDIX_ONLY",
    }
    if _is_esl_method(method):
        summary["final_prototype_softmax"] = stable_softmax(state.logits).tolist() if state.logits is not None else None
    return {"status": "READY", "skip_reason": "", "rows": rows, "summary": summary}


def _write_shared_run_artifacts(
    *,
    run_dir: Path,
    result: dict[str, Any],
    task: MatrixGameTask,
    regime: OpponentRegime,
    method: str,
    seed: int,
    horizon: int,
    smoke: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": task.key,
        "regime": regime.key,
        "method": method,
        "seed": seed,
        "horizon": horizon,
        "smoke": smoke,
        "shared_protocol": True,
        "focal_agent_id": FOCAL_AGENT_ID,
        "baseline_fidelity": result["summary"].get("baseline_fidelity", "skipped"),
        "skip_reason": result.get("skip_reason", ""),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        run_dir / "focal_trajectory.csv",
        result.get("rows", []),
        [
            "round",
            "focal_agent_id",
            "opponent_id",
            "opponent_type",
            "task",
            "regime",
            "method",
            "focal_action",
            "opponent_action",
            "focal_reward",
            "opponent_reward",
            "focal_cumulative_payoff",
            "focal_mean_payoff_per_round",
            "schedule_random_u",
            "policy_p_cooperate",
        ],
    )
    manifest = {
        "schema_version": 1,
        "run_kind": "esl" if _is_esl_method(method) else "baseline",
        "experiment_id": "main_performance_grid.protocol_fixed_smoke",
        "seed": seed,
        "task": task.key,
        "regime": regime.key,
        "method": method,
        "shared_protocol": True,
        "focal_agent_id": FOCAL_AGENT_ID,
        "baseline_fidelity": result["summary"].get("baseline_fidelity", "skipped"),
        "skip_reason": result.get("skip_reason", ""),
    }
    write_manifest_json(run_dir / "manifest.json", manifest)
    provenance = {
        "sources": [{"name": method, "url": "", "local_path": "esl/experiments/main_performance_grid.py"}],
        "shared_protocol": True,
        "focal_agent_id": FOCAL_AGENT_ID,
        "baseline_fidelity": result["summary"].get("baseline_fidelity", "skipped"),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_row_reward(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        vals = [float(r["r_i"]) for r in reader if r.get("r_i") not in (None, "")]
    return float(np.mean(vals)) if vals else float("nan")


def _spec_to_audit(spec: BaselineSpec) -> dict[str, Any]:
    return {
        "status": spec.availability.status,
        "source_path": spec.source.local_path,
        "source_url": spec.source.url,
        "commit": spec.source.commit,
        "hash": spec.source.commit,
        "license": spec.source.license,
        "adapter_path": spec.adapter_path,
        "can_run_repeated_2action_matrix_games": spec.availability.can_run_repeated_2x2,
        "required_deviations_from_official_code": list(spec.deviations),
        "availability_reason": spec.availability.reason,
    }


def write_baseline_audit(path: Path, adapters: list[BaselineAdapter]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "milestone": "main_performance_grid",
        "baselines": {adapter.spec.family: _spec_to_audit(adapter.spec) for adapter in adapters},
        "notes": [
            "PPO keeps the existing official PPO.py adapter.",
            "M-FOS, MBOM, and Simple Opponent Model sources are pinned under third_party.",
            "Reduced adapters are marked PARTIAL and document deviations in provenance.",
        ],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def _run_esl_cell(
    *,
    out_root: Path,
    seed: int,
    task_key: str,
    regime_key: str,
    horizon: int,
    smoke: bool,
) -> Path:
    tasks = {t.key: t for t in matrix_game_tasks()}
    regimes = {r.key: r for r in opponent_regimes()}
    task = tasks[task_key]
    regime = regimes[regime_key]
    cfg = task_config(task, regime, seed=seed, horizon=horizon)
    run_dir = out_root / "ESL" / task.key / regime.key / f"seed_{seed}"
    _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
    manifest = build_esl_manifest_dict(
        experiment_id="esl.main_performance_grid.smoke",
        seed=seed,
        smoke=smoke,
        manifest_variant=f"{task.key}.{regime.key}",
        neurips_preset=None,
        extra={"task": task.key, "regime": regime.key, "horizon": horizon},
    )
    write_manifest_json(run_dir / "manifest.json", manifest)
    summary["schema_version"] = SUMMARY_METRICS_SCHEMA_VERSION
    return run_dir


def _run_onboarding_smokes(
    *,
    out_root: Path,
    reports_dir: Path,
    adapters: list[BaselineAdapter],
    horizon: int,
) -> list[dict[str, Any]]:
    task = matrix_game_tasks()[0]
    regime = opponent_regimes()[0]
    rows: list[dict[str, Any]] = []
    for adapter in adapters:
        out = out_root / "_onboarding" / adapter.spec.family
        row: dict[str, Any] = {
            "method": adapter.spec.method_name,
            "baseline_family": adapter.spec.family,
            "status": adapter.spec.availability.status,
            "official_or_reduced": "official" if adapter.spec.availability.status == "READY" else "reduced",
            "run_dir": str(out),
        }
        try:
            if not adapter.spec.availability.can_run_repeated_2x2:
                raise RuntimeError(adapter.spec.availability.reason)
            adapter.train(out, seed=0, task=task, regime=regime, horizon=horizon, smoke=True)
            validate_baseline_policy_smoke_directory(out)
            summary = _read_summary(out / "summary_metrics.json")
            payoff = float(summary["mean_payoff_per_agent_per_round"])
            if not math.isfinite(payoff):
                raise ValueError("payoff is not finite")
            row["payoff"] = payoff
            row["result"] = "PASS"
            row["reason"] = adapter.spec.availability.reason
        except Exception as exc:  # noqa: BLE001 - recorded as explicit onboarding status.
            row["result"] = "UNAVAILABLE"
            row["reason"] = str(exc)
        rows.append(row)

    lines = [
        "# Baseline Onboarding Status",
        "",
        "| Method | Family | Status | Smoke result | Official/reduced | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {baseline_family} | {status} | {result} | {official_or_reduced} | {reason} |".format(
                **row
            )
        )
    (reports_dir / "baseline_onboarding_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    status: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (str(row["task"]), str(row["regime"]), str(row["method"]))
        try:
            payoff = float(row["payoff"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(payoff):
            continue
        grouped.setdefault(key, []).append(payoff)
        status[key] = str(row.get("status", ""))
    out: list[dict[str, Any]] = []
    for (task, regime, method), values in sorted(grouped.items()):
        st = mean_std_ci95(values)
        out.append(
            {
                "task": task,
                "regime": regime,
                "method": method,
                "status": status.get((task, regime, method), ""),
                "n": int(st["n"]),
                "mean": st["mean"],
                "std": st["std"],
                "ci95_low": st["ci95_low"],
                "ci95_high": st["ci95_high"],
            }
        )
    return out


def _plot_grid(summary_rows: list[dict[str, Any]], out_path: Path, *, horizon: int) -> None:
    tasks = matrix_game_tasks()
    regimes = opponent_regimes()
    methods = list(SUPPORTED_SHARED_METHODS)
    colors = {
        "ESL": "#1f77b4",
        "ESL K=1": "#17becf",
        "ESL without belief updates": "#2ca02c",
        "PPO": "#ff7f0e",
        "SOM": "#9467bd",
    }
    lookup = {(r["task"], r["regime"], r["method"]): r for r in summary_rows}
    fig, axes = plt.subplots(len(tasks), len(regimes), figsize=(13.5, 9.0), sharey=False)
    row_limits: dict[str, tuple[float, float]] = {}
    for task in tasks:
        row_values: list[float] = []
        for regime in regimes:
            for method in methods:
                row = lookup.get((task.key, regime.key, method))
                if row is None:
                    continue
                row_values.extend([float(row["ci95_low"]), float(row["ci95_high"]), 0.0])
        if row_values:
            lo = min(row_values)
            hi = max(row_values)
            pad = max((hi - lo) * 0.12, 0.08)
            row_limits[task.key] = (lo - pad, hi + pad)
    for i, task in enumerate(tasks):
        for j, regime in enumerate(regimes):
            ax = axes[i, j]
            vals: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for method in methods:
                row = lookup.get((task.key, regime.key, method))
                if row is None:
                    vals.append(float("nan"))
                    lows.append(float("nan"))
                    highs.append(float("nan"))
                    continue
                mean = float(row["mean"])
                vals.append(mean)
                lows.append(mean - float(row["ci95_low"]))
                highs.append(float(row["ci95_high"]) - mean)
            x = np.arange(len(methods))
            ax.bar(
                x,
                vals,
                yerr=np.array([lows, highs], dtype=np.float64),
                color=[colors[m] for m in methods],
                capsize=3,
                alpha=0.88,
            )
            ax.axhline(0.0, color="#333333", linewidth=0.8)
            if task.key in row_limits:
                ax.set_ylim(*row_limits[task.key])
            if i == 0:
                ax.set_title(regime.label, fontsize=12, pad=10)
            if j == 0:
                ax.set_ylabel(f"{task.label}\nFocal payoff / round", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False, length=0)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in methods]
    fig.legend(handles, methods, loc="lower center", ncol=len(methods), frameon=False, fontsize=10)
    fig.text(
        0.5,
        0.035,
        f"Bars show focal agent 0 mean payoff per round with 95% CI across 10 seeds and {horizon} rounds; all methods use the same opponent schedule for each task, regime, and seed.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi())
    plt.close(fig)


def _write_report(
    *,
    path: Path,
    audit: dict[str, Any],
    onboarding_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    figure_path: Path,
) -> None:
    ready = [
        b["source_path"]
        for b in audit["baselines"].values()
        if b["status"] == "READY"
    ]
    partial = [
        f"{name}: {b['availability_reason']}"
        for name, b in audit["baselines"].items()
        if b["status"] != "READY"
    ]
    esl_cells = [r for r in summary_rows if r["method"] == "ESL"]
    external_cells = [r for r in summary_rows if r["method"] != "ESL"]
    esl_mean = float(np.nanmean([float(r["mean"]) for r in esl_cells])) if esl_cells else float("nan")
    ext_mean = (
        float(np.nanmean([float(r["mean"]) for r in external_cells])) if external_cells else float("nan")
    )
    separation = esl_mean - ext_mean if math.isfinite(esl_mean) and math.isfinite(ext_mean) else float("nan")
    lines = [
        "# Main Performance Grid Smoke Report",
        "",
        "## Baseline Readiness",
        "",
        f"- READY sources: {', '.join(ready) if ready else 'none'}",
        f"- PARTIAL / unavailable: {'; '.join(partial) if partial else 'none'}",
        "",
        "## Smoke Onboarding",
        "",
    ]
    for row in onboarding_rows:
        lines.append(f"- {row['method']}: {row['result']} ({row['reason']})")
    lines.extend(
        [
            "",
            "## Figure",
            "",
            f"- Output: `{figure_path}`",
            "- Layout: usable 3 x 3 task/regime grid with payoff bars and 95% CI error bars.",
            f"- ESL average minus external average across smoke cells: {separation:.4f}",
            "",
            "## Before Full Paper Runs",
            "",
            "- Review the PARTIAL adapters and decide whether reduced implementations are acceptable.",
            "- Wrap full official M-FOS and MBOM training stacks if they must be claimed as official baselines.",
            "- Decide how to handle the Lua/Torch Simple Opponent Model source for camera-ready claims.",
            "- Increase seeds and horizons only after this audit is accepted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_protocol_fix_report(
    *,
    path: Path,
    audit: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    figure_path: Path,
    horizon: int,
    seeds: list[int],
    smoke: bool,
) -> None:
    method_status = {
        "ESL": "VALID_SHARED_PROTOCOL",
        "ESL K=1": "VALID_SHARED_PROTOCOL_ABLATION",
        "ESL without belief updates": "VALID_SHARED_PROTOCOL_ABLATION",
        "PPO": "VALID_SHARED_PROTOCOL_APPENDIX_ONLY",
        "SOM": "VALID_SHARED_PROTOCOL_APPENDIX_ONLY",
        "M-FOS": "SKIPPED",
        "MBOM": "SKIPPED",
    }
    skipped_reasons: dict[str, str] = {}
    for row in skipped_rows:
        skipped_reasons.setdefault(str(row["method"]), str(row["reason"]))
    esl_rows = [r for r in summary_rows if r["method"] == "ESL"]
    k1_rows = [r for r in summary_rows if r["method"] == "ESL K=1"]
    no_belief_rows = [r for r in summary_rows if r["method"] == "ESL without belief updates"]
    ppo_rows = [r for r in summary_rows if r["method"] == "PPO"]
    som_rows = [r for r in summary_rows if r["method"] == "SOM"]
    def _avg(rows: list[dict[str, Any]]) -> float:
        return float(np.nanmean([float(r["mean"]) for r in rows])) if rows else float("nan")

    def _cell_diff(other: str) -> tuple[int, int, float]:
        esl_by_cell = {(r["task"], r["regime"]): float(r["mean"]) for r in esl_rows}
        other_rows = [r for r in summary_rows if r["method"] == other]
        diffs = [
            esl_by_cell[(r["task"], r["regime"])] - float(r["mean"])
            for r in other_rows
            if (r["task"], r["regime"]) in esl_by_cell
        ]
        wins = sum(1 for d in diffs if d > 0)
        return wins, len(diffs), float(np.mean(diffs)) if diffs else float("nan")

    ppo_wins, ppo_n, ppo_mean_diff = _cell_diff("PPO")
    som_wins, som_n, som_mean_diff = _cell_diff("SOM")
    k1_wins, k1_n, k1_mean_diff = _cell_diff("ESL K=1")
    no_belief_wins, no_belief_n, no_belief_mean_diff = _cell_diff("ESL without belief updates")

    lines = [
        "# Main Performance Grid Report",
        "",
        "## Decision",
        "",
        (
            "**Protocol is fixed for smoke evaluation, but the 3 x 3 figure is not ready for full paper runs until PPO/SOM appendix status is accepted and M-FOS/MBOM policy is decided.**"
            if smoke
            else "**Full corrected shared-protocol run completed for ESL, ESL ablations, PPO, and SOM.**"
        ),
        "",
        "## Shared Protocol",
        "",
        f"- Focal learner: agent `{FOCAL_AGENT_ID}`.",
        "- Reported metric: `focal_mean_payoff_per_round` only.",
        "- Every included method uses `run_shared_focal_protocol` in `esl/experiments/main_performance_grid.py`.",
        "- Opponent IDs/types and random uniforms are generated once per task/regime/seed and reused across methods.",
        "- Baseline adapters' separate synthetic `train(...)` loops are not used for this corrected figure.",
        f"- {'Smoke' if smoke else 'Full'} seeds: `{seeds}`; horizon: `{horizon}`.",
        "",
        "## Method Validity",
        "",
        "| Method | Status | Fidelity | Notes |",
        "|---|---|---|---|",
        "| ESL | VALID_SHARED_PROTOCOL | official | Agent 0 acts in every regime, including fixed types. |",
        "| ESL K=1 | VALID_SHARED_PROTOCOL_ABLATION | ablation | Single-prototype ESL ablation under the same focal-agent loop. |",
        "| ESL without belief updates | VALID_SHARED_PROTOCOL_ABLATION | ablation | Beliefs stay uniform while prototype updates continue under the same focal-agent loop. |",
        "| PPO | VALID_SHARED_PROTOCOL_APPENDIX_ONLY | official PPO core, custom shared wrapper | Uses official `PPO.py` policy/update inside the shared focal loop. |",
        "| SOM | VALID_SHARED_PROTOCOL_APPENDIX_ONLY | reduced | Faithful simple conditional opponent model inside shared loop; official Lua/Torch DRON is not executed. |",
        f"| M-FOS | SKIPPED | skipped | {skipped_reasons.get('M-FOS', SKIPPED_SHARED_METHODS['M-FOS'])} |",
        f"| MBOM | SKIPPED | skipped | {skipped_reasons.get('MBOM', SKIPPED_SHARED_METHODS['MBOM'])} |",
        "",
            f"## {'Smoke' if smoke else 'Full'} Payoff Summary",
        "",
        "| Task | Regime | Method | n | Mean | 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['regime']} | {row['method']} | {row['n']} | "
            f"{float(row['mean']):.3f} | [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- ESL average {'smoke' if smoke else 'full'} payoff across cells: `{_avg(esl_rows):.4f}`.",
            f"- ESL K=1 average {'smoke' if smoke else 'full'} payoff across cells: `{_avg(k1_rows):.4f}`.",
            f"- ESL without belief updates average {'smoke' if smoke else 'full'} payoff across cells: `{_avg(no_belief_rows):.4f}`.",
            f"- PPO average {'smoke' if smoke else 'full'} payoff across cells: `{_avg(ppo_rows):.4f}`.",
            f"- SOM average {'smoke' if smoke else 'full'} payoff across cells: `{_avg(som_rows):.4f}`.",
            f"- ESL beats PPO in `{ppo_wins}/{ppo_n}` cells; mean ESL-PPO difference `{ppo_mean_diff:.4f}`.",
            f"- ESL beats SOM in `{som_wins}/{som_n}` cells; mean ESL-SOM difference `{som_mean_diff:.4f}`.",
            f"- ESL beats ESL K=1 in `{k1_wins}/{k1_n}` cells; mean difference `{k1_mean_diff:.4f}`.",
            f"- ESL beats ESL without belief updates in `{no_belief_wins}/{no_belief_n}` cells; mean difference `{no_belief_mean_diff:.4f}`.",
            f"- Figure output: `{figure_path}`.",
            f"- Caption: 10 seeds, {horizon} rounds, focal agent payoff, shared opponent schedule.",
            "- ESL payoff is now meaningful as focal-agent payoff under a shared protocol.",
            "- M-FOS and MBOM are intentionally absent from the corrected v1 figure instead of silently using reduced substitutes.",
            "",
            "## Recommendation",
            "",
            (
                "Keep the corrected 3 x 3 layout. Before full runs, decide whether the main figure should contain only ESL + PPO + SOM, or whether full official M-FOS/MBOM wrappers are required. If M-FOS/MBOM are required in main text, do not run full sweeps until those official wrappers exist."
                if smoke
                else "Use this as the corrected v1 main performance grid with ESL ablations. M-FOS and MBOM remain excluded based on `manuscript_bundle/reports/mfos_mbom_wrapper_feasibility.md`."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_main_performance_grid(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    smoke: bool = True,
    seeds: list[int] | None = None,
    horizon: int | None = None,
) -> dict[str, Path]:
    seeds = seeds if seeds is not None else ([0, 1] if smoke else list(range(10)))
    horizon = horizon if horizon is not None else (12 if smoke else 500)
    ensure_manuscript_bundle_layout(manuscript_bundle)
    main_fig_dir = manuscript_bundle / "main" / "figures"
    main_table_dir = manuscript_bundle / "main" / "tables"
    main_caption_dir = manuscript_bundle / "text" / "main" / "captions"
    reports_dir = manuscript_bundle / "reports"
    for d in (main_fig_dir, main_table_dir, main_caption_dir, reports_dir, out_root):
        d.mkdir(parents=True, exist_ok=True)

    adapters = external_adapters()
    audit = write_baseline_audit(reports_dir / "baseline_audit.json", adapters)

    raw_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    methods = list(SUPPORTED_SHARED_METHODS) + list(SKIPPED_SHARED_METHODS.keys())
    for seed in seeds:
        for task in matrix_game_tasks():
            for regime in opponent_regimes():
                for method in methods:
                    family = method.lower().replace(" ", "_").replace("-", "_")
                    run_dir = out_root / family / task.key / regime.key / f"seed_{seed}"
                    try:
                        result = run_shared_focal_protocol(
                            method=method,
                            task=task,
                            regime=regime,
                            seed=seed,
                            horizon=horizon,
                            smoke=smoke,
                        )
                    except Exception as exc:  # noqa: BLE001 - explicit unavailable method cell.
                        result = {
                            "status": "SKIPPED",
                            "skip_reason": str(exc),
                            "rows": [],
                            "summary": {
                                "task": task.key,
                                "regime": regime.key,
                                "method": method,
                                "seed": seed,
                                "shared_protocol": True,
                                "focal_agent_id": FOCAL_AGENT_ID,
                                "baseline_fidelity": "skipped",
                            },
                        }
                    _write_shared_run_artifacts(
                        run_dir=run_dir,
                        result=result,
                        task=task,
                        regime=regime,
                        method=method,
                        seed=seed,
                        horizon=horizon,
                        smoke=smoke,
                    )
                    summary = result["summary"]
                    if result["status"] == "SKIPPED":
                        skip_rows.append(
                            {
                                "task": task.key,
                                "regime": regime.key,
                                "method": method,
                                "seed": seed,
                                "status": "SKIPPED",
                                "reason": result.get("skip_reason", ""),
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    raw_rows.append(
                        {
                            "task": task.key,
                            "regime": regime.key,
                            "method": method,
                            "seed": seed,
                            "status": summary.get("status", "READY"),
                            "payoff": summary["focal_mean_payoff_per_round"],
                            "run_dir": str(run_dir),
                            "shared_protocol": True,
                            "focal_agent_id": FOCAL_AGENT_ID,
                            "baseline_fidelity": summary.get("baseline_fidelity", ""),
                        }
                    )

    raw_csv = out_root / (
        "main_performance_grid_protocol_fixed_raw.csv" if smoke else "main_performance_grid_raw.csv"
    )
    _write_csv(
        raw_csv,
        raw_rows,
        [
            "task",
            "regime",
            "method",
            "seed",
            "status",
            "payoff",
            "run_dir",
            "shared_protocol",
            "focal_agent_id",
            "baseline_fidelity",
        ],
    )
    skipped_csv = out_root / (
        "main_performance_grid_protocol_fixed_skipped.csv" if smoke else "main_performance_grid_skipped.csv"
    )
    _write_csv(
        skipped_csv,
        skip_rows,
        ["task", "regime", "method", "seed", "status", "reason", "run_dir"],
    )
    summary_rows = _aggregate_rows(raw_rows)
    summary_csv = main_table_dir / (
        "main_performance_grid_protocol_fixed_smoke_summary.csv"
        if smoke
        else "main_performance_grid_summary.csv"
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ["task", "regime", "method", "status", "n", "mean", "std", "ci95_low", "ci95_high"],
    )
    figure_path = main_fig_dir / (
        "main_performance_grid_protocol_fixed_smoke.png" if smoke else "main_performance_grid.png"
    )
    _plot_grid(summary_rows, figure_path, horizon=horizon)
    report_path = reports_dir / (
        "main_performance_grid_protocol_fix_report.md"
        if smoke
        else "main_performance_grid_full_report.md"
    )
    _write_protocol_fix_report(
        path=report_path,
        audit=audit,
        summary_rows=summary_rows,
        skipped_rows=skip_rows,
        figure_path=figure_path,
        horizon=horizon,
        seeds=seeds,
        smoke=smoke,
    )
    caption_path = main_caption_dir / (
        "main_performance_grid_protocol_fixed_smoke.md"
        if smoke
        else "main_performance_grid.md"
    )
    caption_path.write_text(
        (
            "Main performance grid over IPD, Stag Hunt, and Matching Pennies under fixed-type, "
            "adaptive-agent, and belief-conditioned-agent regimes. Bars report focal agent 0 mean "
            f"payoff per round with 95% confidence intervals over {len(seeds)} seeds and {horizon} "
            "rounds; for each task, regime, and seed, every method receives the same opponent "
            "schedule."
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "audit": reports_dir / "baseline_audit.json",
        "raw_csv": raw_csv,
        "skipped_csv": skipped_csv,
        "summary_csv": summary_csv,
        "figure": figure_path,
        "report": report_path,
        "caption": caption_path,
    }


def main_performance_grid_cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Main performance grid smoke milestone")
    p.add_argument("--out-root", type=Path, default=Path("runs/main_performance_grid"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--paper", action="store_true")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--horizon", type=int, default=None)
    args = p.parse_args(argv)
    smoke = True if args.smoke or not args.paper else False
    seeds = None if args.seeds is None else [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    outputs = run_main_performance_grid(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        smoke=smoke,
        seeds=seeds,
        horizon=args.horizon,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main_performance_grid_cli()
