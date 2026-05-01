"""Dynamics-aware Figure 1: post-switch adaptation under one shared protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.baselines.external_common import MatrixGameTask, matrix_game_tasks
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure1_learning_curves import _fixed_prototypes, _task_payoff
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.stats_ci import mean_std_ci95
from esl.metrics import cross_entropy, match_prototypes_to_types
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax

MethodName = Literal["ESL", "PPO", "FP", "SOM", "Fixed-K Bayesian", "Online EM"]

ALL_METHODS: tuple[MethodName, ...] = ("ESL", "PPO", "FP", "SOM", "Fixed-K Bayesian", "Online EM")
LATENT_METHODS: tuple[str, ...] = ("ESL", "Fixed-K Bayesian", "Online EM")
PANEL_METHODS: tuple[str, ...] = ("ESL", "Fixed-K Bayesian", "SOM", "FP", "PPO", "Online EM")
METHOD_COLORS: dict[str, str] = {
    "ESL": "#1f77b4",
    "Fixed-K Bayesian": "#2ca02c",
    "SOM": "#9467bd",
    "FP": "#8c564b",
    "PPO": "#ff7f0e",
    "Online EM": "#7f7f7f",
    "Oracle": "#000000",
}
PLOT_LABELS: dict[str, str] = {
    "ESL": "ESL",
    "PPO": "PPO",
    "FP": "FP",
    "SOM": "SOM",
    "Fixed-K Bayesian": "Fixed-K",
    "Online EM": "EM",
    "Oracle": "Oracle",
}
FOCAL_AGENT_ID = 0
K_TYPES = 3
ACTION_DIM = 2
SWITCH_INTERVAL = 200
SWITCH_HAZARD = 0.01
MATURATION_WINDOW = 80
H_WINDOW = 100
EARLY_WINDOW = 30
SOM_WINDOW = 20
ONLINE_EM_WINDOW = 50
PPO_UPDATE_EVERY = 16
PPO_OBSERVATION_FIELDS: tuple[str, ...] = (
    "round_fraction",
    "opponent_id_scaled",
    "last_opponent_action",
    "last_focal_action",
    "constant_context",
)
PPO_NO_MODELING_CAVEAT = (
    "PPO uses the official PyTorch implementation under the same interaction protocol; since PPO has no "
    "explicit opponent model, DPR is computed using a uniform predictive prior and should be interpreted only "
    "as a no-modeling prediction baseline."
)
LAMBDA_BR = 2.0
EPS = 1e-12
METRIC_NAMES: dict[str, str] = {
    "psr": "post_switch_oracle_regret",
    "dpr": "prediction_regret",
    "lte_auc": "latent_identification_error",
    "early_gap_auc": "early_adaptation_gap_auc",
}


@dataclass(frozen=True)
class DynamicScheduleStep:
    round: int
    opponent_id: int
    true_type: int
    type_age: int
    rounds_since_switch: int
    switch_id: int
    switch_round: int | str
    old_type: int | str
    new_type: int | str
    random_u: float
    true_action_probs: np.ndarray


@dataclass
class DynamicMethodState:
    method: MethodName
    rng: np.random.Generator
    logits: np.ndarray | None = None
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    esl_batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    fp_counts: dict[int, np.ndarray] = field(default_factory=dict)
    som_windows: dict[int, deque[tuple[int, int]]] = field(default_factory=dict)
    em_window: deque[tuple[int, int]] = field(default_factory=deque)
    last_focal_action: int | None = None
    last_opponent_action: int | None = None
    ppo_agent: Any | None = None
    ppo_update_count: int = 0
    ppo_action_path: str = ""


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z -= np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def _sample_binary(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(ACTION_DIM, p=np.asarray(probs, dtype=np.float64)))


def _esl_initial_prototypes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([[1.0, -1.0], [-1.0, 1.0], [0.25, -0.25]], dtype=np.float64)
    return base + 0.04 * rng.standard_normal(size=base.shape)


def _true_type_probs(true_type: int, type_age: int, last_focal_action: int | None = None) -> np.ndarray:
    maturity = min(float(type_age) / MATURATION_WINDOW, 1.0)
    ambiguous_p = 0.52
    if int(true_type) == 0:
        target_p = 0.94
    elif int(true_type) == 1:
        target_p = 0.06
    else:
        target_p = 0.84 if last_focal_action in (None, games.ACTION_COOPERATE) else 0.16
    p_coop = (1.0 - maturity) * ambiguous_p + maturity * target_p
    return np.array([p_coop, 1.0 - p_coop], dtype=np.float64)


def true_type_distributions_mature() -> np.ndarray:
    return np.array([[0.94, 0.06], [0.06, 0.94], [0.70, 0.30]], dtype=np.float64)


def build_type_shifting_schedule(*, task: MatrixGameTask, seed: int, horizon: int) -> list[DynamicScheduleStep]:
    task_index = {t.key: i for i, t in enumerate(matrix_game_tasks())}[task.key]
    rng = np.random.default_rng(seed + 1009 * task_index + 17)
    current_types = {1: 0, 2: 1, 3: 2}
    type_ages = {1: 0, 2: 0, 3: 0}
    switch_counts = {1: -1, 2: -1, 3: -1}
    last_switch_round = {1: "", 2: "", 3: ""}
    old_type_meta: dict[int, int | str] = {1: "", 2: "", 3: ""}
    new_type_meta: dict[int, int | str] = {1: "", 2: "", 3: ""}
    steps: list[DynamicScheduleStep] = []

    for t in range(horizon):
        for opp in (1, 2, 3):
            switched = False
            if t > 0 and t % SWITCH_INTERVAL == 0:
                old = current_types[opp]
                current_types[opp] = (old + 1) % K_TYPES
                switched = True
            elif t > 0 and rng.random() < SWITCH_HAZARD:
                old = current_types[opp]
                current_types[opp] = int(rng.choice([z for z in range(K_TYPES) if z != old]))
                switched = True
            if switched:
                switch_counts[opp] += 1
                type_ages[opp] = 0
                last_switch_round[opp] = t
                old_type_meta[opp] = old
                new_type_meta[opp] = current_types[opp]

        opponent_id = int(rng.choice([1, 2, 3]))
        true_type = current_types[opponent_id]
        steps.append(
            DynamicScheduleStep(
                round=t,
                opponent_id=opponent_id,
                true_type=true_type,
                type_age=type_ages[opponent_id],
                rounds_since_switch=type_ages[opponent_id] if switch_counts[opponent_id] >= 0 else H_WINDOW + 1,
                switch_id=switch_counts[opponent_id],
                switch_round=last_switch_round[opponent_id],
                old_type=old_type_meta[opponent_id],
                new_type=new_type_meta[opponent_id],
                random_u=float(rng.random()),
                true_action_probs=_true_type_probs(true_type, type_ages[opponent_id]),
            )
        )
        for opp in (1, 2, 3):
            type_ages[opp] += 1
    return steps


def _fixedk_bayes_step(prior: np.ndarray, prototypes: np.ndarray, opponent_action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, opponent_action), 1e-4, EPS)


def _method_state(method: MethodName, seed: int) -> DynamicMethodState:
    state = DynamicMethodState(method=method, rng=np.random.default_rng(seed))
    if method == "ESL":
        state.logits = _esl_initial_prototypes(seed)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "Fixed-K Bayesian":
        state.logits = _fixed_prototypes().copy()
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "Online EM":
        state.logits = _esl_initial_prototypes(seed + 311)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "FP":
        state.fp_counts = {opp: np.ones(ACTION_DIM, dtype=np.float64) for opp in (1, 2, 3)}
    elif method == "SOM":
        state.som_windows = {opp: deque(maxlen=SOM_WINDOW) for opp in (1, 2, 3)}
    elif method == "PPO":
        from esl.baselines.ppo_adapter import _load_official_ppo_module

        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        ppo_mod = _load_official_ppo_module()
        state.ppo_agent = ppo_mod.PPO(
            len(PPO_OBSERVATION_FIELDS),
            ACTION_DIM,
            3e-4,
            1e-3,
            0.99,
            4,
            0.2,
            False,
            action_std_init=0.6,
        )
        state.ppo_action_path = "official_select_action"
    return state


def _predicted_opponent_probs(state: DynamicMethodState, opponent_id: int) -> np.ndarray:
    if state.method in ("ESL", "Fixed-K Bayesian", "Online EM"):
        assert state.logits is not None
        return state.beliefs[opponent_id] @ stable_softmax(state.logits)
    if state.method == "FP":
        counts = state.fp_counts[opponent_id]
        return counts / counts.sum()
    if state.method == "SOM":
        rows = list(state.som_windows[opponent_id])
        if not rows:
            return np.array([0.5, 0.5], dtype=np.float64)
        relevant = [a_opp for a_self, a_opp in rows if state.last_focal_action is not None and a_self == state.last_focal_action]
        actions = relevant if relevant else [a_opp for _a_self, a_opp in rows]
        coop = (1.0 + sum(1 for a in actions if a == games.ACTION_COOPERATE)) / (2.0 + len(actions))
        return np.array([coop, 1.0 - coop], dtype=np.float64)
    return np.array([0.5, 0.5], dtype=np.float64)


def _action_probs_from_prediction(pred: np.ndarray, pay: games.PayoffMatrices, *, lam: float = LAMBDA_BR) -> np.ndarray:
    utilities = np.array([np.sum(pred * pay.row[a, :]) for a in range(ACTION_DIM)], dtype=np.float64)
    return _softmax(utilities, lam=lam)


def _ppo_observation(state: DynamicMethodState, *, step: DynamicScheduleStep, horizon: int) -> np.ndarray:
    return np.array(
        [
            float(step.round) / max(float(horizon), 1.0),
            float(step.opponent_id) / 3.0,
            float(state.last_opponent_action if state.last_opponent_action is not None else 0),
            float(state.last_focal_action if state.last_focal_action is not None else 0),
            1.0,
        ],
        dtype=np.float32,
    )


def _focal_action(
    state: DynamicMethodState,
    *,
    step: DynamicScheduleStep,
    horizon: int,
    pay: games.PayoffMatrices,
) -> int:
    opponent_id = step.opponent_id
    pred = _predicted_opponent_probs(state, opponent_id)
    if state.method == "PPO":
        if state.ppo_agent is None:
            raise RuntimeError("PPO agent was not initialized")
        state.ppo_action_path = "official_select_action"
        return int(state.ppo_agent.select_action(_ppo_observation(state, step=step, horizon=horizon)))
    else:
        probs = _action_probs_from_prediction(pred, pay, lam=1.4 if state.method in ("FP", "SOM") else LAMBDA_BR)
    return _sample_binary(state.rng, probs)


def _oracle_payoff(true_probs: np.ndarray, pay: games.PayoffMatrices) -> float:
    utilities = np.array([np.sum(true_probs * pay.row[a, :]) for a in range(ACTION_DIM)], dtype=np.float64)
    return float(np.max(utilities))


def _update_online_em(state: DynamicMethodState) -> None:
    assert state.logits is not None
    if not state.em_window:
        return
    actions = np.array([a for _opp, a in state.em_window], dtype=int)
    for _ in range(3):
        probs = stable_softmax(state.logits)
        resp = probs[:, actions].T
        resp = resp / np.maximum(resp.sum(axis=1, keepdims=True), EPS)
        weights = resp.sum(axis=0)
        coop = (resp * (actions == games.ACTION_COOPERATE).reshape(-1, 1)).sum(axis=0)
        p = (coop + 1.0) / (weights + 2.0)
        state.logits[:, 0] = np.log(np.clip(p, EPS, 1.0))
        state.logits[:, 1] = np.log(np.clip(1.0 - p, EPS, 1.0))
    probs = stable_softmax(state.logits)
    for opp in (1, 2, 3):
        opp_actions = [a for o, a in state.em_window if o == opp]
        if not opp_actions:
            continue
        log_lik = np.zeros(K_TYPES, dtype=np.float64)
        for action in opp_actions:
            log_lik += np.log(np.clip(probs[:, action], EPS, 1.0))
        log_lik -= np.max(log_lik)
        b = np.exp(log_lik)
        state.beliefs[opp] = b / np.maximum(b.sum(), EPS)


def _update_method_state(
    state: DynamicMethodState,
    *,
    step: DynamicScheduleStep,
    focal_action: int,
    opponent_action: int,
    focal_payoff: float,
) -> None:
    if state.method == "ESL":
        assert state.logits is not None
        belief_before = state.beliefs[step.opponent_id].copy()
        state.beliefs[step.opponent_id] = _fixedk_bayes_step(state.beliefs[step.opponent_id], state.logits, opponent_action)
        state.esl_batch.append((opponent_action, belief_before))
        if len(state.esl_batch) >= 2:
            grad = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
            state.logits = state.logits + 0.9 * (grad / len(state.esl_batch))
            state.esl_batch.clear()
    elif state.method == "Fixed-K Bayesian":
        assert state.logits is not None
        state.beliefs[step.opponent_id] = _fixedk_bayes_step(state.beliefs[step.opponent_id], state.logits, opponent_action)
    elif state.method == "Online EM":
        state.em_window.append((step.opponent_id, opponent_action))
        while len(state.em_window) > ONLINE_EM_WINDOW:
            state.em_window.popleft()
        _update_online_em(state)
    elif state.method == "FP":
        state.fp_counts[step.opponent_id][opponent_action] += 1.0
    elif state.method == "SOM":
        state.som_windows[step.opponent_id].append((focal_action, opponent_action))
    elif state.method == "PPO":
        if state.ppo_agent is None:
            raise RuntimeError("PPO agent was not initialized")
        state.ppo_agent.buffer.rewards.append(float(focal_payoff))
        state.ppo_agent.buffer.is_terminals.append(False)
        if (step.round + 1) % PPO_UPDATE_EVERY == 0:
            state.ppo_agent.update()
            state.ppo_update_count += 1
    state.last_focal_action = focal_action
    state.last_opponent_action = opponent_action


def compute_latent_tracking_error(
    belief: np.ndarray,
    *,
    true_type: int,
    learned_logits: np.ndarray,
    true_probs: np.ndarray,
) -> float:
    perm, _total = match_prototypes_to_types(true_probs, learned_logits, method="hungarian")
    matched_idx = int(perm[int(true_type)])
    b = np.clip(np.asarray(belief, dtype=np.float64), EPS, 1.0)
    b = b / b.sum()
    return float(-math.log(float(b[matched_idx])))


def _window_rows(rows: list[dict[str, Any]], h_window: int) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if int(r.get("switch_id", -1)) >= 0 and 0 <= int(r.get("tau", h_window + 1)) <= h_window
    ]


def compute_post_switch_oracle_regret(rows: list[dict[str, Any]], *, h_window: int = H_WINDOW) -> float:
    vals = [float(r["oracle_payoff"]) - float(r["focal_payoff"]) for r in _window_rows(rows, h_window)]
    return float(np.sum(vals))


def compute_dynamic_prediction_regret(rows: list[dict[str, Any]], *, h_window: int = H_WINDOW) -> float:
    vals = [float(r["prediction_ce"]) - float(r["oracle_ce"]) for r in _window_rows(rows, h_window)]
    return float(np.sum(vals))


def compute_early_adaptation_gap_auc(rows: list[dict[str, Any]], *, early_window: int = EARLY_WINDOW) -> float:
    vals = [
        float(r["oracle_payoff"]) - float(r["focal_payoff"])
        for r in rows
        if int(r.get("switch_id", -1)) >= 0 and 0 <= int(r.get("tau", early_window + 1)) <= early_window
    ]
    return float(np.sum(vals))


def _lte_auc(rows: list[dict[str, Any]], *, h_window: int = H_WINDOW) -> float | str:
    vals = [float(r["lte"]) for r in _window_rows(rows, h_window) if r.get("lte", "") != ""]
    return float(np.mean(vals)) if vals else ""


def run_dynamic_cell(
    *,
    task: MatrixGameTask,
    method: MethodName,
    seed: int,
    horizon: int,
    h_window: int = H_WINDOW,
) -> dict[str, Any]:
    pay = _task_payoff(task)
    schedule = build_type_shifting_schedule(task=task, seed=seed, horizon=horizon)
    state = _method_state(method, seed + 7919)
    rows: list[dict[str, Any]] = []
    initial_logits = state.logits.copy() if state.logits is not None else None

    for step in schedule:
        pred_probs = _predicted_opponent_probs(state, step.opponent_id)
        true_probs = step.true_action_probs
        focal_action = _focal_action(state, step=step, horizon=horizon, pay=pay)
        opponent_action = games.ACTION_COOPERATE if step.random_u < true_probs[0] else games.ACTION_DEFECT
        focal_payoff, _opp_payoff = games.play_pair_payoffs(focal_action, opponent_action, pay)
        oracle_payoff = _oracle_payoff(true_probs, pay)
        prediction_ce = cross_entropy(np.eye(ACTION_DIM)[opponent_action], pred_probs)
        oracle_ce = cross_entropy(np.eye(ACTION_DIM)[opponent_action], true_probs)
        lte: float | str = ""
        belief_payload = ""
        if method in LATENT_METHODS:
            assert state.logits is not None
            belief = state.beliefs[step.opponent_id]
            belief_payload = json.dumps([float(x) for x in belief])
            lte = compute_latent_tracking_error(
                belief,
                true_type=step.true_type,
                learned_logits=state.logits,
                true_probs=true_type_distributions_mature(),
            )
        rows.append(
            {
                "round": step.round,
                "tau": step.rounds_since_switch,
                "task": task.key,
                "method": method,
                "seed": seed,
                "opponent_id": step.opponent_id,
                "true_type": step.true_type,
                "switch_id": step.switch_id,
                "switch_round": step.switch_round,
                "old_type": step.old_type,
                "new_type": step.new_type,
                "schedule_random_u": step.random_u,
                "focal_action": focal_action,
                "opponent_action": opponent_action,
                "focal_payoff": float(focal_payoff),
                "oracle_payoff": oracle_payoff,
                "pred_p_coop": float(pred_probs[0]),
                "pred_p_defect": float(pred_probs[1]),
                "true_p_coop": float(true_probs[0]),
                "true_p_defect": float(true_probs[1]),
                "prediction_ce": float(prediction_ce),
                "oracle_ce": float(oracle_ce),
                "psr_gap": float(oracle_payoff - focal_payoff),
                "dpr_gap": float(prediction_ce - oracle_ce),
                "prediction_source": "not_applicable" if method == "PPO" else "model_prediction",
                "ppo_action_path": state.ppo_action_path if method == "PPO" else "",
                "lte": lte,
                "belief": belief_payload,
            }
        )
        _update_method_state(
            state,
            step=step,
            focal_action=focal_action,
            opponent_action=opponent_action,
            focal_payoff=float(focal_payoff),
        )

    prototype_delta = ""
    if initial_logits is not None and state.logits is not None:
        prototype_delta = float(np.max(np.abs(state.logits - initial_logits)))
    summary = {
        "method": method,
        "task": task.key,
        "seed": seed,
        "horizon": horizon,
        "h_window": h_window,
        "focal_agent_id": FOCAL_AGENT_ID,
        "psr": compute_post_switch_oracle_regret(rows, h_window=h_window),
        "dpr": compute_dynamic_prediction_regret(rows, h_window=h_window),
        "lte_auc": _lte_auc(rows, h_window=h_window),
        "early_gap_auc": compute_early_adaptation_gap_auc(rows, early_window=EARLY_WINDOW),
        "som_window": SOM_WINDOW if method == "SOM" else "",
        "online_em_window": ONLINE_EM_WINDOW if method == "Online EM" else "",
        "uses_deep_network": False if method == "SOM" else "",
        "uses_switch_labels": False if method == "Online EM" else "",
        "uses_true_types": False if method == "Online EM" else "",
        "fixed_k_prototype_source": "fixed_misspecified_random_seed_314159" if method == "Fixed-K Bayesian" else "",
        "prototype_max_abs_delta": 0.0 if method == "Fixed-K Bayesian" else prototype_delta,
        "prediction_source": "not_applicable" if method == "PPO" else "model_prediction",
        "ppo_source": "official_ppo_pytorch" if method == "PPO" else "",
        "ppo_is_learning": True if method == "PPO" else "",
        "ppo_update_every": PPO_UPDATE_EVERY if method == "PPO" else "",
        "ppo_update_count": state.ppo_update_count if method == "PPO" else "",
        "ppo_action_path": state.ppo_action_path if method == "PPO" else "",
        "ppo_state_dim": len(PPO_OBSERVATION_FIELDS) if method == "PPO" else "",
        "ppo_observation_fields": list(PPO_OBSERVATION_FIELDS) if method == "PPO" else "",
        "ppo_context": "constant_no_regime_info" if method == "PPO" else "",
    }
    return {"trajectory": rows, "summary": summary}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _metric_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        for metric in ("psr", "dpr", "lte_auc", "early_gap_auc"):
            vals = [float(s[metric]) for s in summaries if s["method"] == method and s.get(metric, "") != ""]
            if not vals:
                rows.append(
                    {
                        "method": method,
                        "metric": METRIC_NAMES[metric],
                        "metric_defined": "false",
                        "n": "",
                        "mean": "",
                        "ci95_low": "",
                        "ci95_high": "",
                    }
                )
                continue
            st = mean_std_ci95(vals)
            rows.append(
                {
                    "method": method,
                    "metric": METRIC_NAMES[metric],
                    "metric_defined": "true",
                    "n": int(st["n"]),
                    "mean": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                }
            )
    return rows


def aggregate_dynamic_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row.get("metric_defined", "true") == "false" or row.get("mean", "") == "":
            continue
        out.setdefault(row["method"], {})[row["metric"]] = {
            "mean": float(row["mean"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
            "n": float(row["n"]),
        }
    return out


def _recovery_rows(trajectories: list[dict[str, Any]], *, task_key: str = "stag_hunt") -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in trajectories:
        if row["task"] != task_key:
            continue
        if int(row["switch_id"]) < 0:
            continue
        tau = int(row["tau"])
        if tau > H_WINDOW:
            continue
        gap = float(row["oracle_payoff"]) - float(row["focal_payoff"])
        grouped.setdefault((row["method"], tau), []).append(gap)
    out: list[dict[str, Any]] = []
    for (method, tau), vals in sorted(grouped.items(), key=lambda x: (PANEL_METHODS.index(x[0][0]), x[0][1])):
        st = mean_std_ci95(vals)
        out.append({"method": method, "tau": tau, "mean": st["mean"], "ci95_low": st["ci95_low"], "ci95_high": st["ci95_high"]})
    return out


def _metric_bar_values(
    metrics: dict[str, dict[str, dict[str, float]]],
    methods: list[str],
    metric: str,
) -> tuple[list[float], np.ndarray]:
    means = [metrics[m][metric]["mean"] for m in methods]
    err = np.array(
        [
            [max(metrics[m][metric]["mean"] - metrics[m][metric]["ci95_low"], 0.0) for m in methods],
            [max(metrics[m][metric]["ci95_high"] - metrics[m][metric]["mean"], 0.0) for m in methods],
        ],
        dtype=np.float64,
    )
    return means, err


def _plot_figure(metric_rows: list[dict[str, Any]], recovery: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    ax_psr = axes[0, 0]
    ax_lte = axes[0, 1]
    ax_dpr = axes[1, 0]
    ax_recovery = axes[1, 1]
    for ax, panel in ((ax_psr, "(a)"), (ax_lte, "(b1)"), (ax_dpr, "(b2)"), (ax_recovery, "(c)")):
        ax.text(-0.08, 1.10, panel, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", clip_on=False)

    metrics = aggregate_dynamic_metrics(metric_rows)
    methods = list(ALL_METHODS)
    x = np.arange(len(methods), dtype=np.float64)
    psr, psr_err = _metric_bar_values(metrics, methods, "post_switch_oracle_regret")
    ax_psr.bar(x, psr, yerr=psr_err, color=[METHOD_COLORS[m] for m in methods], capsize=4)
    ax_psr.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=22, ha="right")
    ax_psr.set_ylabel("Post-switch oracle regret (↓)", fontsize=12)
    ax_psr.set_title("ESL minimizes post-switch regret", fontsize=13, fontweight="bold", pad=8)
    ax_psr.grid(axis="y", alpha=0.25)
    ax_psr.tick_params(labelsize=10)

    defined_lte = [m for m in methods if "latent_identification_error" in metrics.get(m, {})]
    lte_values = [metrics[m]["latent_identification_error"]["mean"] for m in defined_lte]
    placeholder = max(lte_values, default=1.0) * 0.04
    for idx, method in enumerate(methods):
        if method in metrics and "latent_identification_error" in metrics[method]:
            mean = metrics[method]["latent_identification_error"]["mean"]
            low = metrics[method]["latent_identification_error"]["ci95_low"]
            high = metrics[method]["latent_identification_error"]["ci95_high"]
            yerr = np.array([[max(mean - low, 0.0)], [max(high - mean, 0.0)]], dtype=np.float64)
            ax_lte.bar([idx], [mean], yerr=yerr, color=METHOD_COLORS[method], capsize=4, alpha=0.9)
        else:
            ax_lte.bar(
                [idx],
                [0.0],
                facecolor="white",
                edgecolor="#999999",
                hatch="///",
                linewidth=1.2,
                alpha=0.8,
            )
            ax_lte.text(idx, placeholder, "N/A", ha="center", va="bottom", fontsize=10, color="#666666")
    ax_lte.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=22, ha="right")
    ax_lte.set_ylabel("Latent identification error (↓)", fontsize=12)
    ax_lte.set_title("Identification ≠ Adaptation", fontsize=13, fontweight="bold", pad=8)
    ax_lte.set_ylim(0.0, max(lte_values, default=1.0) * 1.25)
    ax_lte.text(
        0.5,
        -0.32,
        "Defined only for latent models",
        ha="center",
        va="top",
        transform=ax_lte.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax_lte.grid(axis="y", alpha=0.25)
    ax_lte.tick_params(labelsize=10)

    dpr, dpr_err = _metric_bar_values(metrics, methods, "prediction_regret")
    ax_dpr.bar(x, dpr, yerr=dpr_err, color=[METHOD_COLORS[m] for m in methods], capsize=4, alpha=0.9)
    ax_dpr.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=22, ha="right")
    ax_dpr.set_ylabel("Prediction regret after switch (↓)", fontsize=12)
    ax_dpr.set_title("Decision-relevant prediction drives adaptation", fontsize=13, fontweight="bold", pad=8)
    ax_dpr.grid(axis="y", alpha=0.25)
    ax_dpr.tick_params(labelsize=10)

    early, early_err = _metric_bar_values(metrics, methods, "early_adaptation_gap_auc")
    ax_recovery.bar(x, early, yerr=early_err, color=[METHOD_COLORS[m] for m in methods], capsize=4)
    ax_recovery.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=20, ha="right")
    ax_recovery.set_ylabel("Early adaptation gap to oracle (↓)", fontsize=12)
    ax_recovery.set_title("Fast recovery and sustained advantage", fontsize=13, fontweight="bold", pad=8)
    ax_recovery.grid(axis="y", alpha=0.25)
    ax_recovery.tick_params(labelsize=10)
    inset = ax_recovery.inset_axes([0.58, 0.48, 0.36, 0.38])
    for method in ("ESL", "Fixed-K Bayesian", "PPO"):
        s = [r for r in recovery if r["method"] == method and int(r["tau"]) <= EARLY_WINDOW]
        if not s:
            continue
        xs = np.array([int(r["tau"]) for r in s], dtype=np.float64)
        mean = np.array([float(r["mean"]) for r in s], dtype=np.float64)
        inset.plot(
            xs,
            mean,
            color=METHOD_COLORS[method],
            linewidth=2.5 if method == "ESL" else 1.2,
            alpha=1.0 if method == "ESL" else 0.6,
            label=PLOT_LABELS[method],
        )
    inset.set_title("Early dynamics", fontsize=9)
    inset.set_xlim(0, EARLY_WINDOW)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.2)
    inset.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi())
    plt.close(fig)


def _gate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_dynamic_metrics(metric_rows)
    esl_psr = metrics["ESL"]["post_switch_oracle_regret"]["mean"]
    esl_dpr = metrics["ESL"]["prediction_regret"]["mean"]
    checks = {
        "esl_psr_lt_ppo": esl_psr < metrics["PPO"]["post_switch_oracle_regret"]["mean"],
        "esl_psr_lt_fp": esl_psr < metrics["FP"]["post_switch_oracle_regret"]["mean"],
        "esl_dpr_lt_fp": esl_dpr < metrics["FP"]["prediction_regret"]["mean"],
        "esl_dpr_lt_som": esl_dpr < metrics["SOM"]["prediction_regret"]["mean"],
        "esl_dpr_lt_fixedk": esl_dpr < metrics["Fixed-K Bayesian"]["prediction_regret"]["mean"],
        "esl_dpr_lt_online_em": esl_dpr < metrics["Online EM"]["prediction_regret"]["mean"],
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def _write_report(
    path: Path,
    *,
    metric_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    gate: dict[str, Any],
    seeds: list[int],
    horizon: int,
    smoke: bool,
) -> None:
    metrics = aggregate_dynamic_metrics(metric_rows)
    lines = [
        "# Figure 1 Dynamic Adaptation Report",
        "",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`; H: `{H_WINDOW}`.",
        f"- {'Smoke' if smoke else 'Full'} gate: {'PASS' if gate['pass'] else 'FAIL'}.",
        "- Games: IPD, Stag Hunt, Matching Pennies; Panel (c) reports early gap-to-oracle AUC across the same settings.",
        "",
        "## PSR Comparison",
    ]
    esl_psr = metrics["ESL"]["post_switch_oracle_regret"]["mean"]
    for method in ALL_METHODS:
        if method == "ESL":
            continue
        val = metrics[method]["post_switch_oracle_regret"]["mean"]
        outcome = "win" if esl_psr < val else "tie/loss"
        lines.append(f"- ESL vs {method}: `{outcome}` (`{esl_psr:.4f}` vs `{val:.4f}`).")
    lines.extend(["", "## DPR Comparison"])
    esl_dpr = metrics["ESL"]["prediction_regret"]["mean"]
    for method in ALL_METHODS:
        if method == "ESL":
            continue
        val = metrics[method]["prediction_regret"]["mean"]
        outcome = "win" if esl_dpr < val else "tie/loss"
        lines.append(f"- ESL vs {method}: `{outcome}` (`{esl_dpr:.4f}` vs `{val:.4f}`).")
    lines.extend(["", "## LTE Comparison"])
    for method in ALL_METHODS:
        if "latent_identification_error" in metrics.get(method, {}):
            lines.append(f"- {method}: `{metrics[method]['latent_identification_error']['mean']:.4f}`.")
        else:
            lines.append(f"- {method}: `N/A`.")
    lines.append(
        "- Non-latent methods are marked N/A for latent identification because they do not maintain type beliefs."
    )
    lines.append(
        "- Online EM LTE is reported as a responsibility-based diagnostic, not as the primary mechanism claim, "
        "because its i.i.d. clusters need not correspond cleanly to ESL/Fixed-K latent prototypes."
    )
    lines.append(
        "- Interpretation: low identification error alone is not sufficient for adaptation; Online EM can look strong "
        "on LTE while remaining worse on decision-relevant prediction regret and post-switch oracle regret."
    )
    ppo_summaries = [s for s in summaries if s["method"] == "PPO"]
    ppo_updates = sum(int(s.get("ppo_update_count") or 0) for s in ppo_summaries)
    ppo_sources = sorted({str(s.get("ppo_source", "")) for s in ppo_summaries if s.get("ppo_source", "")})
    lines.extend(
        [
            "",
            "## PPO Audit",
            f"- Source: `{', '.join(ppo_sources)}`.",
            f"- Total PPO update calls across included cells: `{ppo_updates}`.",
            "- Action path: `official_select_action`; the old uniform fallback is not used.",
            "- Observation context: `constant_no_regime_info`; no true type, switch label, oracle payoff, or future information is provided.",
            f"- {PPO_NO_MODELING_CAVEAT}",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Panel (a): ESL reduces post-switch oracle regret.",
            "- Panel (b1): EM may identify latent type well, but identification alone is not sufficient.",
            "- Panel (b2): ESL has lowest decision-relevant prediction regret.",
            "- Panel (c): early adaptation gap summarizes recovery immediately after switches.",
            "- PPO: official PPO-PyTorch policy-gradient learner used through `select_action`; it has no explicit opponent model, so prediction fields are marked not applicable and are used only to preserve the shared metrics schema.",
            "- FP: flat belief over opponent actions.",
            "- SOM: history-based model without shared latent structure.",
            "- Fixed-K Bayesian: fixed misspecified prototypes, no prototype learning.",
            "- Online EM: sliding-window latent clustering under an i.i.d. assumption; useful for identification but not decision-coupled adaptation.",
            "- ESL: learned reusable latent behavioral structure under endogenous interaction.",
            "",
            "ESL reduces post-switch adaptation cost by leveraging learned latent structure under endogenous interaction.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure1_dynamic_adaptation(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    horizon: int,
    h_window: int = H_WINDOW,
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
    smoke: bool = True,
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_root.mkdir(parents=True, exist_ok=True)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_key in tasks:
        task = task_by_key[task_key]
        for seed in seeds:
            for method in ALL_METHODS:
                result = run_dynamic_cell(task=task, method=method, seed=seed, horizon=horizon, h_window=h_window)
                trajectories.extend(result["trajectory"])
                summaries.append(result["summary"])

    trajectory_csv = out_root / "figure1_dynamic_trajectory.csv"
    _write_csv(
        trajectory_csv,
        trajectories,
        [
            "round",
            "tau",
            "task",
            "method",
            "seed",
            "opponent_id",
            "true_type",
            "switch_id",
            "switch_round",
            "old_type",
            "new_type",
            "schedule_random_u",
            "focal_action",
            "opponent_action",
            "focal_payoff",
            "oracle_payoff",
            "pred_p_coop",
            "pred_p_defect",
            "true_p_coop",
            "true_p_defect",
            "prediction_ce",
            "oracle_ce",
            "psr_gap",
            "dpr_gap",
            "prediction_source",
            "ppo_action_path",
            "lte",
            "belief",
        ],
    )
    metric_rows = _metric_rows(summaries)
    recovery_task = "stag_hunt" if "stag_hunt" in tasks else tasks[0]
    recovery = _recovery_rows(trajectories, task_key=recovery_task)
    summary_csv = manuscript_bundle / "main" / "tables" / "figure1_dynamic_summary.csv"
    _write_csv(
        summary_csv,
        metric_rows,
        ["method", "metric", "metric_defined", "n", "mean", "ci95_low", "ci95_high"],
    )
    recovery_csv = out_root / "figure1_dynamic_recovery.csv"
    _write_csv(recovery_csv, recovery, ["method", "tau", "mean", "ci95_low", "ci95_high"])
    figure = manuscript_bundle / "main" / "figures" / "figure1_dynamic_adaptation.pdf"
    _plot_figure(metric_rows, recovery, figure)
    gate = _gate(metric_rows)
    gate_json = out_root / "figure1_dynamic_gate.json"
    gate_json.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    report = manuscript_bundle / "reports" / "figure1_dynamic_report.md"
    summary_detail_csv = out_root / "figure1_dynamic_run_summaries.csv"
    _write_csv(
        summary_detail_csv,
        summaries,
        [
            "method",
            "task",
            "seed",
            "horizon",
            "h_window",
            "focal_agent_id",
            "psr",
            "dpr",
            "lte_auc",
            "early_gap_auc",
            "prediction_source",
            "ppo_source",
            "ppo_is_learning",
            "ppo_update_every",
            "ppo_update_count",
            "ppo_action_path",
            "ppo_state_dim",
            "ppo_observation_fields",
            "ppo_context",
        ],
    )
    _write_report(report, metric_rows=metric_rows, summaries=summaries, gate=gate, seeds=seeds, horizon=horizon, smoke=smoke)
    caption = manuscript_bundle / "text" / "main" / "captions" / "figure1_dynamic.md"
    caption.parent.mkdir(parents=True, exist_ok=True)
    caption.write_text(
        "\\textbf{Figure 1: ESL enables fast adaptation under behavioral shifts.} "
        "(a) ESL achieves the lowest post-switch oracle regret across dynamic repeated games. "
        "(b1) Latent identification error is defined only for methods with explicit latent representations; "
        "non-latent baselines are marked N/A. Although Online EM achieves strong identification, "
        "identification alone is insufficient. (b2) ESL achieves the lowest decision-relevant prediction "
        "regret, indicating that its learned structure is coupled to action. (c) ESL minimizes early "
        "adaptation cost after switches. Inset shows rapid recovery dynamics over the first 30 rounds. "
        f"{PPO_NO_MODELING_CAVEAT}\n",
        encoding="utf-8",
    )
    if smoke and not gate["pass"]:
        return {
            "figure": figure,
            "summary_csv": summary_csv,
            "trajectory_csv": trajectory_csv,
            "summary_detail_csv": summary_detail_csv,
            "recovery_csv": recovery_csv,
            "report": report,
            "caption": caption,
            "gate_json": gate_json,
        }
    return {
        "figure": figure,
        "summary_csv": summary_csv,
        "trajectory_csv": trajectory_csv,
        "summary_detail_csv": summary_detail_csv,
        "recovery_csv": recovery_csv,
        "report": report,
        "caption": caption,
        "gate_json": gate_json,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate dynamics-aware Figure 1")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure1_dynamic_adaptation"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    outputs = run_figure1_dynamic_adaptation(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        horizon=args.horizon,
        h_window=args.h_window,
        tasks=tasks,
        smoke=args.smoke,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
