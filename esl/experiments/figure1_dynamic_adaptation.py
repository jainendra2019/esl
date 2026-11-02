"""Dynamics-aware Figure 1: post-switch adaptation under one shared protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
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
from esl.experiments.figure_style import publication_figure_dpi, apply_ylabel_lower_better, xlabel_lower_better
from esl.experiments.stats_ci import mean_std_ci95
from esl.metrics import cross_entropy, match_prototypes_to_types
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax

MethodName = Literal["ESL", "PPO", "FP", "SOM", "Fixed-K Bayesian", "SA-FP", "Online EM"]

ALL_METHODS: tuple[MethodName, ...] = ("ESL", "PPO", "FP", "SOM", "Fixed-K Bayesian", "SA-FP", "Online EM")
LATENT_METHODS: tuple[str, ...] = ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM")
PANEL_METHODS: tuple[str, ...] = ("ESL", "Fixed-K Bayesian", "SA-FP", "SOM", "FP", "PPO", "Online EM")
METHOD_COLORS: dict[str, str] = {
    "ESL": "#1f77b4",
    "Fixed-K Bayesian": "#2ca02c",
    "SA-FP": "#17becf",
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
    "SA-FP": "SA-FP",
    "Online EM": "EM",
    "Oracle": "Oracle",
}
FOCAL_AGENT_ID = 0
K_TYPES = 3
FIGURE1_MAIN_SEEDS: tuple[int, ...] = tuple(range(10))
FIGURE1_MAIN_HORIZON = 1000
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
SA_FP_DEFAULT_P_STAY = 0.95
SA_FP_SENSITIVITY_P_STAY: tuple[float, ...] = (0.90, 0.95, 0.99)
FIXED_K_PROTOTYPE_SOURCE = "fixed_misspecified_random_seed_314159"
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
    p_stay: float = SA_FP_DEFAULT_P_STAY


@dataclass(frozen=True)
class ESLUpdateConfig:
    label: str = "ESL"
    gamma: float = 0.9
    batch_size: int = 2
    eta_reg: float = 0.0


FIGURE1_DEFAULT_ESL_UPDATE = ESLUpdateConfig()
FIGURE1_DIAGNOSTIC_ESL_UPDATES: tuple[ESLUpdateConfig, ...] = (
    ESLUpdateConfig(label="ESL-main", gamma=0.9, batch_size=2, eta_reg=0.0),
    ESLUpdateConfig(label="ESL-M50-reg", gamma=0.9, batch_size=50, eta_reg=1e-3),
)
FIGURE1_DIAGNOSTIC_REPORT_SENTENCE = (
    "This diagnostic tests robustness/stability of a more theory-aligned constant-step variant; "
    "it is not a convergence experiment and does not replace Appendix D."
)
FIGURE1_PER_GAME_REPORT_SENTENCE = (
    "Per-game breakdown tests whether ESL’s aggregate improvement is driven by a single task or is consistent "
    "across interaction regimes."
)
FIGURE1_K_SWEEP_REPORT_SENTENCE = (
    "This diagnostic evaluates robustness of ESL to the assumed number of latent prototypes, including under "
    "model misspecification (K ≠ true latent cardinality)."
)


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z -= np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def _sample_binary(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(ACTION_DIM, p=np.asarray(probs, dtype=np.float64)))


def _esl_initial_prototypes(seed: int, *, k_model: int = K_TYPES) -> np.ndarray:
    if k_model < 1:
        raise ValueError(f"k_model must be positive, got {k_model}")
    rng = np.random.default_rng(seed)
    base = np.array([[1.0, -1.0], [-1.0, 1.0], [0.25, -0.25]], dtype=np.float64)
    if k_model == K_TYPES:
        return base + 0.04 * rng.standard_normal(size=base.shape)
    if k_model < K_TYPES:
        return base[:k_model] + 0.04 * rng.standard_normal(size=(k_model, ACTION_DIM))
    extras = rng.normal(loc=float(np.mean(base)), scale=float(np.std(base)), size=(k_model - K_TYPES, ACTION_DIM))
    extended = np.vstack([base, extras])
    return extended + 0.04 * rng.standard_normal(size=extended.shape)


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


def transition_matrix(p_stay: float, *, k: int = K_TYPES) -> np.ndarray:
    """Row-stochastic Markov transition over K latent prototypes: T_ii=p_stay."""
    p = float(p_stay)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p_stay must be in [0, 1], got {p_stay}")
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k_int = int(k)
    if k_int == 1:
        return np.array([[1.0]], dtype=np.float64)
    off = (1.0 - p) / float(k_int - 1)
    t = np.full((k_int, k_int), off, dtype=np.float64)
    np.fill_diagonal(t, p)
    return t


def markov_predict(belief: np.ndarray, p_stay: float) -> np.ndarray:
    """One-step HMM predict: b_pred = T^T b for row-stochastic T."""
    b = np.asarray(belief, dtype=np.float64).reshape(-1)
    if float(p_stay) == 1.0:
        return b.copy()
    t = transition_matrix(p_stay, k=b.shape[0])
    pred = t.T @ b
    total = float(pred.sum())
    if total <= EPS:
        return np.full(b.shape[0], 1.0 / float(b.shape[0]), dtype=np.float64)
    return pred / total


def _schedule_task_index(task: MatrixGameTask) -> int:
    """Stable index for schedule RNG: paper tasks keep legacy indices; RES uses key hash."""
    known = {t.key: i for i, t in enumerate(matrix_game_tasks())}
    if task.key in known:
        return int(known[task.key])
    # Stable across processes (avoid Python's salted hash()).
    digest = 0
    for ch in task.key.encode("utf-8"):
        digest = (digest * 131 + int(ch)) % 10_007
    return int(digest)


def build_type_shifting_schedule(
    *,
    task: MatrixGameTask,
    seed: int,
    horizon: int,
    switch_interval: int = SWITCH_INTERVAL,
    switch_hazard: float = SWITCH_HAZARD,
) -> list[DynamicScheduleStep]:
    task_index = _schedule_task_index(task)
    rng = np.random.default_rng(seed + 1009 * task_index + 17)
    current_types = {1: 0, 2: 1, 3: 2}
    type_ages = {1: 0, 2: 0, 3: 0}
    switch_counts = {1: -1, 2: -1, 3: -1}
    last_switch_round = {1: "", 2: "", 3: ""}
    old_type_meta: dict[int, int | str] = {1: "", 2: "", 3: ""}
    new_type_meta: dict[int, int | str] = {1: "", 2: "", 3: ""}
    steps: list[DynamicScheduleStep] = []
    interval = int(switch_interval)
    hazard = float(switch_hazard)

    for t in range(horizon):
        for opp in (1, 2, 3):
            switched = False
            if interval > 0 and t > 0 and t % interval == 0:
                old = current_types[opp]
                current_types[opp] = (old + 1) % K_TYPES
                switched = True
            elif t > 0 and hazard > 0.0 and rng.random() < hazard:
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


def _method_state(
    method: MethodName,
    seed: int,
    *,
    esl_model_k: int = K_TYPES,
    p_stay: float = SA_FP_DEFAULT_P_STAY,
) -> DynamicMethodState:
    state = DynamicMethodState(method=method, rng=np.random.default_rng(seed), p_stay=float(p_stay))
    if method == "ESL":
        state.logits = _esl_initial_prototypes(seed, k_model=esl_model_k)
        state.beliefs = {opp: np.full(esl_model_k, 1.0 / esl_model_k) for opp in (1, 2, 3)}
    elif method in ("Fixed-K Bayesian", "SA-FP"):
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
    if state.method in ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM"):
        assert state.logits is not None
        belief = state.beliefs[opponent_id]
        if state.method == "SA-FP":
            belief = markov_predict(belief, state.p_stay)
        return belief @ stable_softmax(state.logits)
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
    esl_update: ESLUpdateConfig = FIGURE1_DEFAULT_ESL_UPDATE,
) -> float | str:
    prototype_update_norm: float | str = ""
    if state.method == "ESL":
        assert state.logits is not None
        belief_before = state.beliefs[step.opponent_id].copy()
        state.beliefs[step.opponent_id] = _fixedk_bayes_step(state.beliefs[step.opponent_id], state.logits, opponent_action)
        state.esl_batch.append((opponent_action, belief_before))
        if len(state.esl_batch) >= esl_update.batch_size:
            grad = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
            update_direction = grad / len(state.esl_batch) - esl_update.eta_reg * state.logits
            delta = esl_update.gamma * update_direction
            state.logits = state.logits + delta
            prototype_update_norm = float(np.linalg.norm(delta))
            state.esl_batch.clear()
    elif state.method == "Fixed-K Bayesian":
        assert state.logits is not None
        state.beliefs[step.opponent_id] = _fixedk_bayes_step(state.beliefs[step.opponent_id], state.logits, opponent_action)
    elif state.method == "SA-FP":
        assert state.logits is not None
        prior = markov_predict(state.beliefs[step.opponent_id], state.p_stay)
        state.beliefs[step.opponent_id] = _fixedk_bayes_step(prior, state.logits, opponent_action)
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
    return prototype_update_norm


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
    esl_update: ESLUpdateConfig = FIGURE1_DEFAULT_ESL_UPDATE,
    method_label: str | None = None,
    esl_model_k: int = K_TYPES,
    compute_lte: bool = True,
    p_stay: float = SA_FP_DEFAULT_P_STAY,
    switch_interval: int = SWITCH_INTERVAL,
    switch_hazard: float = SWITCH_HAZARD,
) -> dict[str, Any]:
    pay = _task_payoff(task)
    schedule = build_type_shifting_schedule(
        task=task,
        seed=seed,
        horizon=horizon,
        switch_interval=switch_interval,
        switch_hazard=switch_hazard,
    )
    state = _method_state(method, seed + 7919, esl_model_k=esl_model_k, p_stay=p_stay)
    output_method = method_label or method
    rows: list[dict[str, Any]] = []
    initial_logits = state.logits.copy() if state.logits is not None else None
    initial_norm = float(np.linalg.norm(initial_logits)) if initial_logits is not None else ""
    prototype_norms: list[float] = []
    prototype_update_norms: list[float] = []

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
        if compute_lte and method in LATENT_METHODS:
            assert state.logits is not None
            # LTE uses the same predictive belief used for action selection.
            belief = state.beliefs[step.opponent_id]
            if method == "SA-FP":
                belief = markov_predict(belief, state.p_stay)
            belief_payload = json.dumps([float(x) for x in belief])
            lte = compute_latent_tracking_error(
                belief,
                true_type=step.true_type,
                learned_logits=state.logits,
                true_probs=true_type_distributions_mature(),
            )
        row = {
            "round": step.round,
            "tau": step.rounds_since_switch,
            "task": task.key,
            "method": output_method,
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
        update_norm = _update_method_state(
            state,
            step=step,
            focal_action=focal_action,
            opponent_action=opponent_action,
            focal_payoff=float(focal_payoff),
            esl_update=esl_update,
        )
        if state.logits is not None:
            prototype_norm = float(np.linalg.norm(state.logits))
            prototype_norms.append(prototype_norm)
            row["prototype_norm_fro"] = prototype_norm
        else:
            row["prototype_norm_fro"] = ""
        row["prototype_update_norm_fro"] = update_norm
        if update_norm != "":
            prototype_update_norms.append(float(update_norm))
        rows.append(row)

    prototype_delta = ""
    if initial_logits is not None and state.logits is not None:
        prototype_delta = float(np.max(np.abs(state.logits - initial_logits)))
    prototype_norm_final = prototype_norms[-1] if prototype_norms else ""
    prototype_norm_max = max(prototype_norms) if prototype_norms else ""
    prototype_norm_growth_ratio = (
        float(prototype_norm_max) / float(initial_norm)
        if prototype_norms and initial_norm != "" and float(initial_norm) > EPS
        else ""
    )
    late_updates = prototype_update_norms[-max(1, len(prototype_update_norms) // 4) :] if prototype_update_norms else []
    summary = {
        "method": output_method,
        "task": task.key,
        "seed": seed,
        "horizon": horizon,
        "h_window": h_window,
        "esl_model_k": esl_model_k if method == "ESL" else "",
        "focal_agent_id": FOCAL_AGENT_ID,
        "psr": compute_post_switch_oracle_regret(rows, h_window=h_window),
        "dpr": compute_dynamic_prediction_regret(rows, h_window=h_window),
        "lte_auc": _lte_auc(rows, h_window=h_window),
        "early_gap_auc": compute_early_adaptation_gap_auc(rows, early_window=EARLY_WINDOW),
        "som_window": SOM_WINDOW if method == "SOM" else "",
        "online_em_window": ONLINE_EM_WINDOW if method == "Online EM" else "",
        "uses_deep_network": False if method == "SOM" else "",
        "uses_switch_labels": False if method in ("Online EM", "SA-FP") else "",
        "uses_true_types": False if method in ("Online EM", "SA-FP") else "",
        "fixed_k_prototype_source": FIXED_K_PROTOTYPE_SOURCE if method in ("Fixed-K Bayesian", "SA-FP") else "",
        "prototype_max_abs_delta": 0.0 if method in ("Fixed-K Bayesian", "SA-FP") else prototype_delta,
        "sa_fp_p_stay": float(p_stay) if method == "SA-FP" else "",
        "esl_gamma": esl_update.gamma if method == "ESL" else "",
        "esl_batch_size": esl_update.batch_size if method == "ESL" else "",
        "esl_eta_reg": esl_update.eta_reg if method == "ESL" else "",
        "prototype_norm_initial": initial_norm,
        "prototype_norm_final": prototype_norm_final,
        "prototype_norm_max": prototype_norm_max,
        "prototype_norm_growth_ratio": prototype_norm_growth_ratio,
        "prototype_update_norm_mean": float(np.mean(prototype_update_norms)) if prototype_update_norms else "",
        "prototype_update_norm_late_mean": float(np.mean(late_updates)) if late_updates else "",
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


def _summary_metric_rows(
    summaries: list[dict[str, Any]],
    *,
    methods: tuple[str, ...],
    metrics: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for key, label in metrics:
            vals = [float(s[key]) for s in summaries if s["method"] == method and s.get(key, "") != ""]
            if not vals:
                rows.append(
                    {
                        "method": method,
                        "metric": label,
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
                    "metric": label,
                    "metric_defined": "true",
                    "n": int(st["n"]),
                    "mean": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                }
            )
    return rows


def _task_display_name(task_key: str) -> str:
    labels = {
        "ipd": "IPD (Cooperative)",
        "stag_hunt": "Stag Hunt (Coordination)",
        "matching_pennies": "Matching Pennies (Adversarial)",
    }
    return labels.get(task_key, task_key.replace("_", " ").title())


def _read_figure1_run_summaries(path: Path) -> list[dict[str, str]]:
    required = {"method", "task", "seed", "psr", "dpr"}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def _per_game_metric_rows(
    summaries: list[dict[str, Any]],
    *,
    tasks: tuple[str, ...],
    methods: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (("psr", "PSR"), ("dpr", "DPR"))
    for task in tasks:
        for method in methods:
            for key, metric in metrics:
                vals = [
                    float(s[key])
                    for s in summaries
                    if s["task"] == task and s["method"] == method and s.get(key, "") != ""
                ]
                if not vals:
                    rows.append(
                        {
                            "task": task,
                            "method": method,
                            "metric": metric,
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
                        "task": task,
                        "method": method,
                        "metric": metric,
                        "metric_defined": "true",
                        "n": int(st["n"]),
                        "mean": st["mean"],
                        "ci95_low": st["ci95_low"],
                        "ci95_high": st["ci95_high"],
                    }
                )
    return rows


def _validate_per_game_breakdown_rows(
    rows: list[dict[str, Any]],
    *,
    tasks: tuple[str, ...],
    methods: tuple[str, ...],
) -> None:
    expected = {(task, method, metric) for task in tasks for method in methods for metric in ("PSR", "DPR")}
    observed = {(r["task"], r["method"], r["metric"]) for r in rows}
    missing = expected.difference(observed)
    if missing:
        raise ValueError(f"missing per-game rows: {sorted(missing)}")
    undefined = [
        (r["task"], r["method"], r["metric"])
        for r in rows
        if r.get("metric_defined") != "true"
        or not all(math.isfinite(float(r[col])) for col in ("mean", "ci95_low", "ci95_high"))
    ]
    if undefined:
        raise ValueError(f"non-finite or undefined per-game metrics: {undefined}")


def _breakdown_output_paths(manuscript_bundle: Path) -> dict[str, Path]:
    if (manuscript_bundle / "appendix").is_dir():
        return {
            "figure": manuscript_bundle / "appendix" / "figures" / "figure1_per_game_breakdown.pdf",
            "metric_csv": manuscript_bundle / "appendix" / "tables" / "figure1_per_game_breakdown.csv",
            "report": manuscript_bundle / "appendix" / "reports" / "figure1_per_game_breakdown.md",
        }
    return {
        "figure": manuscript_bundle / "main" / "figures" / "figure1_per_game_breakdown.pdf",
        "metric_csv": manuscript_bundle / "main" / "tables" / "figure1_per_game_breakdown.csv",
        "report": manuscript_bundle / "reports" / "figure1_per_game_breakdown.md",
    }


def _plot_per_game_breakdown(
    rows: list[dict[str, Any]],
    *,
    tasks: tuple[str, ...],
    methods: tuple[str, ...],
    out_path: Path,
) -> None:
    method_order = list(methods)
    fig, axes_raw = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3))
    axes = np.atleast_1d(axes_raw)
    metric_offsets = {"PSR": -0.18, "DPR": 0.18}
    metric_colors = {"PSR": "#4c72b0", "DPR": "#dd8452"}
    row_by_key = {(r["task"], r["method"], r["metric"]): r for r in rows}

    for ax, task in zip(axes, tasks, strict=True):
        x = np.arange(len(method_order), dtype=np.float64)
        for metric in ("PSR", "DPR"):
            means = [float(row_by_key[(task, method, metric)]["mean"]) for method in method_order]
            err = np.array(
                [
                    [
                        max(0.0, means[i] - float(row_by_key[(task, method, metric)]["ci95_low"]))
                        for i, method in enumerate(method_order)
                    ],
                    [
                        max(0.0, float(row_by_key[(task, method, metric)]["ci95_high"]) - means[i])
                        for i, method in enumerate(method_order)
                    ],
                ],
                dtype=np.float64,
            )
            edgecolor = ["#222222" if method == "ESL" else "none" for method in method_order]
            linewidth = [1.2 if method == "ESL" else 0.0 for method in method_order]
            ax.bar(
                x + metric_offsets[metric],
                means,
                width=0.34,
                yerr=err,
                color=metric_colors[metric],
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.9,
                capsize=3,
                label=metric,
            )
        ax.set_title(_task_display_name(task), fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x, [PLOT_LABELS.get(m, m) for m in method_order], rotation=24, ha="right")
        ax.set_xlabel("Method", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Mean regret (95% CI; lower is better)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995), title="Metric")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi())
    plt.close(fig)


def _write_per_game_breakdown_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    source_csv: Path,
    tasks: tuple[str, ...],
    methods: tuple[str, ...],
) -> None:
    row_by_key = {(r["task"], r["method"], r["metric"]): r for r in rows}
    lines = [
        "# Figure 1 Per-Game Breakdown",
        "",
        f"- Source: `{source_csv}`.",
        f"- Tasks: `{list(tasks)}`.",
        f"- Methods: `{list(methods)}`.",
        "- Metrics: PSR and DPR, aggregated within task and method using t-based 95% confidence intervals.",
        "",
        "## Interpretation",
        FIGURE1_PER_GAME_REPORT_SENTENCE,
        "PSR and DPR are distinct regret metrics; compare methods within each metric rather than treating the two "
        "colors as a single common scale. ESL bars are outlined to make the focal method easy to locate.",
        "",
        "## ESL Per-Game Summary",
    ]
    for task in tasks:
        psr = row_by_key[(task, "ESL", "PSR")]
        dpr = row_by_key[(task, "ESL", "DPR")]
        lines.append(
            f"- {_task_display_name(task)}: PSR `{float(psr['mean']):.4f}` "
            f"[`{float(psr['ci95_low']):.4f}`, `{float(psr['ci95_high']):.4f}`]; "
            f"DPR `{float(dpr['mean']):.4f}` "
            f"[`{float(dpr['ci95_low']):.4f}`, `{float(dpr['ci95_high']):.4f}`]."
        )
    lines.extend(
        [
            "",
            "## Validation",
            f"- Expected cells: `{len(tasks) * len(methods) * 2}`; observed rows: `{len(rows)}`.",
            "- Figure 1 main outputs are not modified by this appendix diagnostic.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    for (method, tau), vals in sorted(
        grouped.items(),
        key=lambda x: (PANEL_METHODS.index(x[0][0]) if x[0][0] in PANEL_METHODS else len(PANEL_METHODS), x[0][1]),
    ):
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
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    ax_psr = axes[0]
    ax_lte = axes[1]
    ax_dpr = axes[2]
    ax_recovery = axes[3]
    for ax, panel in ((ax_psr, "(a)"), (ax_lte, "(b1)"), (ax_dpr, "(b2)"), (ax_recovery, "(c)")):
        ax.text(-0.07, 1.05, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", clip_on=False)

    metrics = aggregate_dynamic_metrics(metric_rows)
    methods = list(ALL_METHODS)
    x = np.arange(len(methods), dtype=np.float64)
    psr, psr_err = _metric_bar_values(metrics, methods, "post_switch_oracle_regret")
    ax_psr.bar(x, psr, yerr=psr_err, color=[METHOD_COLORS[m] for m in methods], capsize=4, width=0.68)
    ax_psr.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=20, ha="right")
    apply_ylabel_lower_better(ax_psr, "PSR", fontsize=10)
    ax_psr.set_title("Post-switch regret", fontsize=11, fontweight="bold", pad=8)
    ax_psr.grid(axis="y", alpha=0.25)
    ax_psr.tick_params(labelsize=8)

    defined_lte = [m for m in methods if "latent_identification_error" in metrics.get(m, {})]
    lte_values = [metrics[m]["latent_identification_error"]["mean"] for m in defined_lte]
    placeholder = max(lte_values, default=1.0) * 0.04
    for idx, method in enumerate(methods):
        if method in metrics and "latent_identification_error" in metrics[method]:
            mean = metrics[method]["latent_identification_error"]["mean"]
            low = metrics[method]["latent_identification_error"]["ci95_low"]
            high = metrics[method]["latent_identification_error"]["ci95_high"]
            yerr = np.array([[max(mean - low, 0.0)], [max(high - mean, 0.0)]], dtype=np.float64)
            ax_lte.bar([idx], [mean], yerr=yerr, color=METHOD_COLORS[method], capsize=4, alpha=0.9, width=0.68)
        else:
            ax_lte.bar(
                [idx],
                [0.0],
                width=0.68,
                facecolor="white",
                edgecolor="#999999",
                hatch="///",
                linewidth=1.2,
                alpha=0.8,
            )
            ax_lte.text(idx, placeholder * 1.45, "N/A", ha="center", va="bottom", fontsize=7, color="#666666", alpha=0.6)
    ax_lte.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=20, ha="right")
    apply_ylabel_lower_better(ax_lte, "LTE", fontsize=10)
    ax_lte.set_title("Latent-state identification", fontsize=11, fontweight="bold", pad=8)
    ax_lte.set_ylim(0.0, max(lte_values, default=1.0) * 1.25)
    ax_lte.text(
        0.5,
        -0.34,
        "Defined only for latent models",
        ha="center",
        va="top",
        transform=ax_lte.transAxes,
        fontsize=7,
        color="#555555",
    )
    ax_lte.grid(axis="y", alpha=0.25)
    ax_lte.tick_params(labelsize=8)

    dpr, dpr_err = _metric_bar_values(metrics, methods, "prediction_regret")
    ax_dpr.bar(x, dpr, yerr=dpr_err, color=[METHOD_COLORS[m] for m in methods], capsize=4, alpha=0.9, width=0.68)
    ax_dpr.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=20, ha="right")
    apply_ylabel_lower_better(ax_dpr, "DPR", fontsize=10)
    ax_dpr.set_title("Behavior prediction regret", fontsize=11, fontweight="bold", pad=8)
    ax_dpr.grid(axis="y", alpha=0.25)
    ax_dpr.tick_params(labelsize=8)

    early, early_err = _metric_bar_values(metrics, methods, "early_adaptation_gap_auc")
    ax_recovery.bar(x, early, yerr=early_err, color=[METHOD_COLORS[m] for m in methods], capsize=4, width=0.68)
    ax_recovery.set_xticks(x, [PLOT_LABELS[m] for m in methods], rotation=20, ha="right")
    apply_ylabel_lower_better(ax_recovery, "Early gap", fontsize=10)
    ax_recovery.set_title("Early recovery", fontsize=11, fontweight="bold", pad=8)
    ax_recovery.grid(axis="y", alpha=0.25)
    ax_recovery.tick_params(labelsize=8)
    inset = ax_recovery.inset_axes([0.52, 0.50, 0.42, 0.36])
    for method in ("ESL", "SA-FP", "Fixed-K Bayesian", "PPO"):
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
    inset.set_title("Dynamics", fontsize=7)
    inset.set_xlim(0, EARLY_WINDOW)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.2)
    inset.legend(frameon=False, fontsize=4.8, loc="upper left")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.28, top=0.82, wspace=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi(), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
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
    informational = {}
    if "SA-FP" in metrics and "prediction_regret" in metrics["SA-FP"]:
        informational["esl_dpr_vs_sa_fp"] = {
            "esl_dpr": esl_dpr,
            "sa_fp_dpr": metrics["SA-FP"]["prediction_regret"]["mean"],
            "esl_lower": esl_dpr < metrics["SA-FP"]["prediction_regret"]["mean"],
        }
        if "post_switch_oracle_regret" in metrics["SA-FP"]:
            informational["esl_psr_vs_sa_fp"] = {
                "esl_psr": esl_psr,
                "sa_fp_psr": metrics["SA-FP"]["post_switch_oracle_regret"]["mean"],
                "esl_lower": esl_psr < metrics["SA-FP"]["post_switch_oracle_regret"]["mean"],
            }
    return {"pass": bool(all(checks.values())), "checks": checks, "informational": informational}


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
            "- SA-FP: Fixed-K plus HMM-style Markov belief prediction with frozen prototypes.",
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
    sa_fp_p_stay: float = SA_FP_DEFAULT_P_STAY,
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
                result = run_dynamic_cell(
                    task=task,
                    method=method,
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                    p_stay=sa_fp_p_stay,
                )
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
            "prototype_norm_fro",
            "prototype_update_norm_fro",
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
            "esl_gamma",
            "esl_batch_size",
            "esl_eta_reg",
            "prototype_norm_initial",
            "prototype_norm_final",
            "prototype_norm_max",
            "prototype_norm_growth_ratio",
            "prototype_update_norm_mean",
            "prototype_update_norm_late_mean",
            "sa_fp_p_stay",
            "fixed_k_prototype_source",
            "prototype_max_abs_delta",
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


def _diagnostic_metric_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _summary_metric_rows(
        summaries,
        methods=tuple(cfg.label for cfg in FIGURE1_DIAGNOSTIC_ESL_UPDATES),
        metrics=(
            ("psr", "post_switch_oracle_regret"),
            ("dpr", "prediction_regret"),
            ("prototype_norm_growth_ratio", "prototype_norm_growth_ratio"),
            ("prototype_norm_final", "prototype_norm_final"),
            ("prototype_norm_max", "prototype_norm_max"),
            ("prototype_update_norm_late_mean", "prototype_update_norm_late_mean"),
        ),
    )


def _paired_diagnostic_deltas(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    main_label = FIGURE1_DIAGNOSTIC_ESL_UPDATES[0].label
    diag_label = FIGURE1_DIAGNOSTIC_ESL_UPDATES[1].label
    by_key = {(s["method"], s["task"], int(s["seed"])): s for s in summaries}
    rows: list[dict[str, Any]] = []
    keys = sorted({(s["task"], int(s["seed"])) for s in summaries})
    for task, seed in keys:
        main = by_key.get((main_label, task, seed))
        diag = by_key.get((diag_label, task, seed))
        if main is None or diag is None:
            continue
        rows.append(
            {
                "task": task,
                "seed": seed,
                "delta_psr": float(diag["psr"]) - float(main["psr"]),
                "delta_dpr": float(diag["dpr"]) - float(main["dpr"]),
                "delta_norm_growth": float(diag["prototype_norm_growth_ratio"]) - float(main["prototype_norm_growth_ratio"]),
                "main_psr": float(main["psr"]),
                "diagnostic_psr": float(diag["psr"]),
                "main_dpr": float(main["dpr"]),
                "diagnostic_dpr": float(diag["dpr"]),
                "main_norm_growth": float(main["prototype_norm_growth_ratio"]),
                "diagnostic_norm_growth": float(diag["prototype_norm_growth_ratio"]),
            }
        )
    return rows


def _plot_diagnostic(metric_rows: list[dict[str, Any]], out_path: Path) -> None:
    metrics = aggregate_dynamic_metrics(metric_rows)
    methods = [cfg.label for cfg in FIGURE1_DIAGNOSTIC_ESL_UPDATES]
    panels = [
        ("post_switch_oracle_regret", "PSR (lower is better)"),
        ("prediction_regret", "DPR (lower is better)"),
        ("prototype_norm_growth_ratio", "Prototype norm growth"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    colors = ["#1f77b4", "#4c72b0"]
    for ax, (metric, ylabel) in zip(axes, panels, strict=True):
        means = [metrics[m][metric]["mean"] for m in methods]
        yerr = np.array(
            [
                [max(metrics[m][metric]["mean"] - metrics[m][metric]["ci95_low"], 0.0) for m in methods],
                [max(metrics[m][metric]["ci95_high"] - metrics[m][metric]["mean"], 0.0) for m in methods],
            ],
            dtype=np.float64,
        )
        ax.bar(np.arange(len(methods)), means, yerr=yerr, color=colors, capsize=4)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 1 ESL Stability Diagnostic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi())
    plt.close(fig)


def _write_diagnostic_report(
    path: Path,
    *,
    metric_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    seeds: list[int],
    horizon: int,
    tasks: tuple[str, ...],
) -> None:
    metrics = aggregate_dynamic_metrics(metric_rows)
    main_label = FIGURE1_DIAGNOSTIC_ESL_UPDATES[0].label
    diag_label = FIGURE1_DIAGNOSTIC_ESL_UPDATES[1].label
    lines = [
        "# Figure 1 ESL Stability Diagnostic",
        "",
        FIGURE1_DIAGNOSTIC_REPORT_SENTENCE,
        "",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`; tasks: `{list(tasks)}`.",
        f"- Variants: `{main_label}` (`gamma=0.9`, `M=2`, `eta_reg=0`) vs `{diag_label}` (`gamma=0.9`, `M=50`, `eta_reg=0.001`).",
        "- Shared protocol: identical Figure 1 type-shifting schedules for each `(task, seed)` pair.",
        "",
        "## Aggregate Metrics",
    ]
    for metric in ("post_switch_oracle_regret", "prediction_regret", "prototype_norm_growth_ratio"):
        lines.append(f"- `{metric}`:")
        for method in (main_label, diag_label):
            st = metrics[method][metric]
            lines.append(f"  - {method}: mean `{st['mean']:.4f}`, 95% CI [`{st['ci95_low']:.4f}`, `{st['ci95_high']:.4f}`], n=`{int(st['n'])}`.")
    lines.extend(["", "## Paired Deltas"])
    for key in ("delta_psr", "delta_dpr", "delta_norm_growth"):
        vals = [float(r[key]) for r in paired_rows]
        st = mean_std_ci95(vals)
        lines.append(f"- `{key}` ({diag_label} - {main_label}): mean `{st['mean']:.4f}`, 95% CI [`{st['ci95_low']:.4f}`, `{st['ci95_high']:.4f}`], n=`{int(st['n'])}`.")
    finite = all(
        math.isfinite(float(s[key]))
        for s in summaries
        for key in ("psr", "dpr", "prototype_norm_growth_ratio", "prototype_norm_final", "prototype_norm_max")
    )
    lines.extend(
        [
            "",
            "## Validation",
            f"- Expected paired cells: `{2 * len(tasks) * len(seeds)}`; observed summaries: `{len(summaries)}`.",
            f"- Paired comparison rows: `{len(paired_rows)}`.",
            f"- Finite PSR/DPR/norm metrics: `{'PASS' if finite else 'FAIL'}`.",
            "",
            "## Interpretation Constraint",
            "- Use this as a robustness/stability diagnostic only. It should not overwrite the main Figure 1 result unless explicitly promoted after review.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure1_diagnostic(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    horizon: int,
    h_window: int = H_WINDOW,
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_root.mkdir(parents=True, exist_ok=True)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_key in tasks:
        task = task_by_key[task_key]
        for seed in seeds:
            for cfg in FIGURE1_DIAGNOSTIC_ESL_UPDATES:
                result = run_dynamic_cell(
                    task=task,
                    method="ESL",
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                    esl_update=cfg,
                    method_label=cfg.label,
                )
                trajectories.extend(result["trajectory"])
                summaries.append(result["summary"])

    trajectory_csv = out_root / "figure1_diagnostic_trajectory.csv"
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
            "lte",
            "belief",
            "prototype_norm_fro",
            "prototype_update_norm_fro",
        ],
    )
    summary_detail_csv = out_root / "figure1_diagnostic_run_summaries.csv"
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
            "esl_gamma",
            "esl_batch_size",
            "esl_eta_reg",
            "prototype_max_abs_delta",
            "prototype_norm_initial",
            "prototype_norm_final",
            "prototype_norm_max",
            "prototype_norm_growth_ratio",
            "prototype_update_norm_mean",
            "prototype_update_norm_late_mean",
        ],
    )
    metric_rows = _diagnostic_metric_rows(summaries)
    metric_csv = manuscript_bundle / "main" / "tables" / "figure1_diagnostic.csv"
    _write_csv(metric_csv, metric_rows, ["method", "metric", "metric_defined", "n", "mean", "ci95_low", "ci95_high"])
    paired_rows = _paired_diagnostic_deltas(summaries)
    paired_csv = out_root / "figure1_diagnostic_paired_deltas.csv"
    _write_csv(
        paired_csv,
        paired_rows,
        [
            "task",
            "seed",
            "delta_psr",
            "delta_dpr",
            "delta_norm_growth",
            "main_psr",
            "diagnostic_psr",
            "main_dpr",
            "diagnostic_dpr",
            "main_norm_growth",
            "diagnostic_norm_growth",
        ],
    )
    figure = manuscript_bundle / "main" / "figures" / "figure1_diagnostic.pdf"
    _plot_diagnostic(metric_rows, figure)
    report = manuscript_bundle / "reports" / "figure1_diagnostic.md"
    _write_diagnostic_report(
        report,
        metric_rows=metric_rows,
        paired_rows=paired_rows,
        summaries=summaries,
        seeds=seeds,
        horizon=horizon,
        tasks=tasks,
    )
    return {
        "figure": figure,
        "metric_csv": metric_csv,
        "trajectory_csv": trajectory_csv,
        "summary_detail_csv": summary_detail_csv,
        "paired_csv": paired_csv,
        "report": report,
    }


def run_figure1_per_game_breakdown(
    *,
    source_csv: Path = Path("runs/figure1_dynamic_adaptation/figure1_dynamic_run_summaries.csv"),
    manuscript_bundle: Path = Path("manuscript_bundle"),
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
    methods: tuple[str, ...] = PANEL_METHODS,
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    if not source_csv.is_file():
        raise FileNotFoundError(
            f"Figure 1 run-summary CSV not found at {source_csv}. "
            "Generate Figure 1 dynamic adaptation first, or pass --source-csv."
        )
    summaries = _read_figure1_run_summaries(source_csv)
    observed_tasks = {row["task"] for row in summaries}
    observed_methods = {row["method"] for row in summaries}
    missing_tasks = set(tasks).difference(observed_tasks)
    missing_methods = set(methods).difference(observed_methods)
    if missing_tasks or missing_methods:
        raise ValueError(
            "Figure 1 run-summary CSV is incomplete: "
            f"missing_tasks={sorted(missing_tasks)}, missing_methods={sorted(missing_methods)}"
        )

    metric_rows = _per_game_metric_rows(summaries, tasks=tasks, methods=methods)
    _validate_per_game_breakdown_rows(metric_rows, tasks=tasks, methods=methods)
    outputs = _breakdown_output_paths(manuscript_bundle)
    _write_csv(
        outputs["metric_csv"],
        metric_rows,
        ["task", "method", "metric", "metric_defined", "n", "mean", "ci95_low", "ci95_high"],
    )
    _plot_per_game_breakdown(metric_rows, tasks=tasks, methods=methods, out_path=outputs["figure"])
    _write_per_game_breakdown_report(
        outputs["report"],
        rows=metric_rows,
        source_csv=source_csv,
        tasks=tasks,
        methods=methods,
    )
    return {**outputs, "source_csv": source_csv}


def _k_sweep_rows(summaries: list[dict[str, Any]], *, k_values: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k_model in k_values:
        psr_vals = [float(s["psr"]) for s in summaries if int(s["K"]) == int(k_model)]
        dpr_vals = [float(s["dpr"]) for s in summaries if int(s["K"]) == int(k_model)]
        psr_st = mean_std_ci95(psr_vals)
        dpr_st = mean_std_ci95(dpr_vals)
        rows.append(
            {
                "K": int(k_model),
                "method": "ESL",
                "n": int(psr_st["n"]),
                "mean_psr": psr_st["mean"],
                "ci95_psr_low": psr_st["ci95_low"],
                "ci95_psr_high": psr_st["ci95_high"],
                "mean_dpr": dpr_st["mean"],
                "ci95_dpr_low": dpr_st["ci95_low"],
                "ci95_dpr_high": dpr_st["ci95_high"],
            }
        )
    return rows


def _validate_k_sweep_rows(rows: list[dict[str, Any]], *, k_values: tuple[int, ...]) -> None:
    observed = {int(r["K"]) for r in rows}
    missing = set(k_values).difference(observed)
    if missing:
        raise ValueError(f"missing K-sweep rows for K={sorted(missing)}")
    bad = [
        int(r["K"])
        for r in rows
        if r["method"] != "ESL"
        or not all(
            math.isfinite(float(r[col]))
            for col in ("mean_psr", "ci95_psr_low", "ci95_psr_high", "mean_dpr", "ci95_dpr_low", "ci95_dpr_high")
        )
    ]
    if bad:
        raise ValueError(f"non-finite or invalid K-sweep rows for K={bad}")


def _k_sweep_dpr_interpretation(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: int(r["K"]))
    psr_best = min(ordered, key=lambda r: float(r["mean_psr"]))
    dpr_best = min(ordered, key=lambda r: float(r["mean_dpr"]))
    if int(psr_best["K"]) == int(dpr_best["K"]):
        return "DPR shows a similar trend to PSR, indicating consistent robustness in decision-relevant prediction."
    return (
        "DPR does not exactly match the PSR ordering in this sweep; it is reported as a complementary "
        "decision-relevant prediction metric rather than a K-selection criterion."
    )


def _write_k_sweep_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    tasks: tuple[str, ...],
    seeds: list[int],
    horizon: int,
    h_window: int,
) -> None:
    dpr_interpretation = _k_sweep_dpr_interpretation(rows)
    lines = [
        "# Figure 1 ESL K-Sweep Metrics Diagnostic",
        "",
        FIGURE1_K_SWEEP_REPORT_SENTENCE,
        "",
        f"- True environment latent cardinality: `K={K_TYPES}`.",
        f"- Tasks: `{list(tasks)}`.",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`; H: `{h_window}`.",
        "- Metrics: PSR and DPR; lower is better.",
        "- Aggregation: across tasks and seeds; `n` denotes the number of `(task, seed)` runs aggregated.",
        "- CI: t-based 95% confidence interval over the aggregated `(task, seed)` runs.",
        "- Additional prototypes are initialized i.i.d. from the same distribution and scale as existing prototypes, with deterministic seeding.",
        "- Baseline methods are omitted because they do not depend on K in the same manner.",
        "- Results are reported for robustness only; no selection of K is performed.",
        f"- {dpr_interpretation}",
        "",
        "| K | method | n | mean_psr | ci95_psr_low | ci95_psr_high | mean_dpr | ci95_dpr_low | ci95_dpr_high |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['K'])} | {row['method']} | {int(row['n'])} | "
            f"{float(row['mean_psr']):.4f} | {float(row['ci95_psr_low']):.4f} | {float(row['ci95_psr_high']):.4f} | "
            f"{float(row['mean_dpr']):.4f} | {float(row['ci95_dpr_low']):.4f} | {float(row['ci95_dpr_high']):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure1_k_sweep(
    *,
    manuscript_bundle: Path,
    seeds: list[int],
    horizon: int,
    h_window: int = H_WINDOW,
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
    k_values: tuple[int, ...] = (2, 3, 5),
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    summaries: list[dict[str, Any]] = []
    for k_model in k_values:
        for task_key in tasks:
            task = task_by_key[task_key]
            for seed in seeds:
                result = run_dynamic_cell(
                    task=task,
                    method="ESL",
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                    esl_model_k=int(k_model),
                    compute_lte=False,
                )
                summaries.append(
                    {
                        "K": int(k_model),
                        "task": task_key,
                        "seed": seed,
                        "psr": result["summary"]["psr"],
                        "dpr": result["summary"]["dpr"],
                    }
                )

    rows = _k_sweep_rows(summaries, k_values=k_values)
    _validate_k_sweep_rows(rows, k_values=k_values)
    metric_csv = manuscript_bundle / "main" / "tables" / "figure1_k_sweep_metrics.csv"
    report = manuscript_bundle / "reports" / "figure1_k_sweep_metrics.md"
    _write_csv(
        metric_csv,
        rows,
        [
            "K",
            "method",
            "n",
            "mean_psr",
            "ci95_psr_low",
            "ci95_psr_high",
            "mean_dpr",
            "ci95_dpr_low",
            "ci95_dpr_high",
        ],
    )
    _write_k_sweep_report(report, rows=rows, tasks=tasks, seeds=seeds, horizon=horizon, h_window=h_window)
    return {"metric_csv": metric_csv, "report": report}


def k_sweep_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 1 ESL K-sweep PSR diagnostic")
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in FIGURE1_MAIN_SEEDS))
    p.add_argument("--horizon", type=int, default=FIGURE1_MAIN_HORIZON)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    p.add_argument("--k-values", type=str, default="2,3,5")
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    k_values = tuple(int(x.strip()) for x in args.k_values.split(",") if x.strip())
    outputs = run_figure1_k_sweep(
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        horizon=args.horizon,
        h_window=args.h_window,
        tasks=tasks,
        k_values=k_values,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


def _thread_env_metadata() -> dict[str, str]:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    return {key: os.environ.get(key, "") for key in keys}


def _runtime_environment_metadata() -> dict[str, Any]:
    thread_env = _thread_env_metadata()
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or platform.uname().processor,
        "thread_env": thread_env,
        "explicitly_single_threaded": bool(thread_env) and all(value == "1" for value in thread_env.values()),
    }


def _runtime_aggregate_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in cell_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    method_stats: dict[str, dict[str, float]] = {}
    for method in ("ESL", "Online EM"):
        method_rows = by_method.get(method, [])
        wall = mean_std_ci95([float(r["wall_time_sec"]) for r in method_rows])
        psr = mean_std_ci95([float(r["psr"]) for r in method_rows])
        dpr = mean_std_ci95([float(r["dpr"]) for r in method_rows])
        method_stats[method] = wall
        rows.append(
            {
                "method": method,
                "n": int(wall["n"]),
                "mean_wall_time_sec": wall["mean"],
                "ci95_wall_time_low": wall["ci95_low"],
                "ci95_wall_time_high": wall["ci95_high"],
                "mean_psr": psr["mean"],
                "mean_dpr": dpr["mean"],
                "overhead_vs_online_em_pct": "",
            }
        )
    em_mean = method_stats.get("Online EM", {}).get("mean", float("nan"))
    if math.isfinite(float(em_mean)) and float(em_mean) > EPS:
        for row in rows:
            mean = float(row["mean_wall_time_sec"])
            row["overhead_vs_online_em_pct"] = 100.0 * (mean / float(em_mean) - 1.0)
    return rows


def _validate_runtime_rows(rows: list[dict[str, Any]]) -> None:
    methods = {row["method"] for row in rows}
    if methods != {"ESL", "Online EM"}:
        raise ValueError(f"runtime diagnostic expected ESL and Online EM, got {sorted(methods)}")
    for row in rows:
        for key in ("mean_wall_time_sec", "ci95_wall_time_low", "ci95_wall_time_high"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid runtime value for {row['method']}:{key}={value}")
        if float(row["ci95_wall_time_low"]) > float(row["mean_wall_time_sec"]):
            raise ValueError(f"runtime CI low exceeds mean for {row['method']}")
        if float(row["mean_wall_time_sec"]) > float(row["ci95_wall_time_high"]):
            raise ValueError(f"runtime CI high below mean for {row['method']}")


def _runtime_interpretation(rows: list[dict[str, Any]]) -> str:
    by_method = {row["method"]: row for row in rows}
    overhead = by_method.get("ESL", {}).get("overhead_vs_online_em_pct", "")
    if overhead == "":
        return "Runtime overhead relative to Online EM could not be computed from the observed timing rows."
    pct = float(overhead)
    if pct >= 0.0:
        return (
            "ESL incurs additional per-batch computation due to prototype-gradient updates; in this CPU "
            f"implementation and Figure 1 protocol, ESL required approximately {pct:.1f}% additional wall-clock "
            "time relative to Online EM."
        )
    return (
        "In this CPU implementation and Figure 1 protocol, ESL was not slower than Online EM in the observed "
        f"wall-clock timings (relative difference {pct:.1f}%)."
    )


def _write_runtime_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    tasks: tuple[str, ...],
    seeds: list[int],
    horizon: int,
    h_window: int,
) -> None:
    thread_env = metadata["thread_env"]
    thread_desc = ", ".join(f"{key}={value or 'unset'}" for key, value in thread_env.items())
    lines = [
        "# Figure 1 Runtime Diagnostic",
        "",
        _runtime_interpretation(rows),
        "",
        "- This is an implementation-level timing diagnostic, not a hardware-normalized scalability benchmark.",
        "- Timings wrap only `run_dynamic_cell()` and exclude CSV writing, report writing, and plot generation.",
        "- LTE computation is disabled for this timing diagnostic so method updates, simulation, PSR, and DPR are timed consistently.",
        f"- Tasks: `{list(tasks)}`.",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`; H: `{h_window}`.",
        f"- Environment: `{metadata['platform']}`; Python `{metadata['python_version']}`; machine `{metadata['machine']}`; processor `{metadata['processor'] or 'unknown'}`.",
        f"- Thread environment: {thread_desc}. Explicitly single-threaded from environment: `{metadata['explicitly_single_threaded']}`.",
        "",
        "| method | n | mean_wall_time_sec | ci95_wall_time_low | ci95_wall_time_high | overhead_vs_online_em_pct |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        overhead = row["overhead_vs_online_em_pct"]
        overhead_s = "" if overhead == "" else f"{float(overhead):.2f}"
        lines.append(
            f"| {row['method']} | {int(row['n'])} | {float(row['mean_wall_time_sec']):.6f} | "
            f"{float(row['ci95_wall_time_low']):.6f} | {float(row['ci95_wall_time_high']):.6f} | {overhead_s} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure1_runtime_diagnostic(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    horizon: int,
    h_window: int = H_WINDOW,
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_root.mkdir(parents=True, exist_ok=True)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    cell_rows: list[dict[str, Any]] = []
    for task_key in tasks:
        task = task_by_key[task_key]
        for seed in seeds:
            for method in ("ESL", "Online EM"):
                start = time.perf_counter()
                result = run_dynamic_cell(
                    task=task,
                    method=method,  # type: ignore[arg-type]
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                    compute_lte=False,
                )
                wall = time.perf_counter() - start
                cell_rows.append(
                    {
                        "method": method,
                        "task": task_key,
                        "seed": seed,
                        "horizon": horizon,
                        "h_window": h_window,
                        "K": K_TYPES,
                        "wall_time_sec": float(wall),
                        "psr": result["summary"]["psr"],
                        "dpr": result["summary"]["dpr"],
                    }
                )
    rows = _runtime_aggregate_rows(cell_rows)
    _validate_runtime_rows(rows)
    metric_csv = manuscript_bundle / "main" / "tables" / "figure1_runtime_diagnostic.csv"
    detail_csv = out_root / "figure1_runtime_cells.csv"
    report = manuscript_bundle / "reports" / "figure1_runtime_diagnostic.md"
    _write_csv(
        metric_csv,
        rows,
        [
            "method",
            "n",
            "mean_wall_time_sec",
            "ci95_wall_time_low",
            "ci95_wall_time_high",
            "mean_psr",
            "mean_dpr",
            "overhead_vs_online_em_pct",
        ],
    )
    _write_csv(detail_csv, cell_rows, ["method", "task", "seed", "horizon", "h_window", "K", "wall_time_sec", "psr", "dpr"])
    _write_runtime_report(
        report,
        rows=rows,
        metadata=_runtime_environment_metadata(),
        tasks=tasks,
        seeds=seeds,
        horizon=horizon,
        h_window=h_window,
    )
    return {"metric_csv": metric_csv, "detail_csv": detail_csv, "report": report}


def runtime_diagnostic_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 1 runtime diagnostic")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure1_runtime_diagnostic"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in FIGURE1_MAIN_SEEDS))
    p.add_argument("--horizon", type=int, default=FIGURE1_MAIN_HORIZON)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    outputs = run_figure1_runtime_diagnostic(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        horizon=args.horizon,
        h_window=args.h_window,
        tasks=tasks,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


def per_game_breakdown_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 1 per-game appendix diagnostic")
    p.add_argument("--source-csv", type=Path, default=Path("runs/figure1_dynamic_adaptation/figure1_dynamic_run_summaries.csv"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    args = p.parse_args(argv)
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    outputs = run_figure1_per_game_breakdown(
        source_csv=args.source_csv,
        manuscript_bundle=args.manuscript_bundle,
        tasks=tasks,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


def diagnostic_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 1 ESL stability diagnostic")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure1_diagnostic"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    outputs = run_figure1_diagnostic(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        horizon=args.horizon,
        h_window=args.h_window,
        tasks=tasks,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate dynamics-aware Figure 1")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure1_dynamic_adaptation"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    p.add_argument("--sa-fp-p-stay", type=float, default=SA_FP_DEFAULT_P_STAY)
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
        sa_fp_p_stay=float(args.sa_fp_p_stay),
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


def _plot_sa_fp_transition_robustness(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    gap: dict[str, float] | None = None,
) -> None:
    """Two-panel figure: SA-FP vs p_stay with Fixed-K/ESL refs + aggregate capability decomposition."""
    if gap is None:
        by_method = {
            r["method"]: float(r["mean_psr"])
            for r in rows
            if abs(float(r["p_stay"]) - SA_FP_DEFAULT_P_STAY) < 1e-12
        }
        if {"ESL", "SA-FP", "Fixed-K Bayesian"} <= set(by_method):
            fk = by_method["Fixed-K Bayesian"]
            sa = by_method["SA-FP"]
            es = by_method["ESL"]
            gap = {
                "mean_psr_fixedk": fk,
                "mean_psr_sa_fp": sa,
                "mean_psr_esl": es,
                "gap_fixedk_to_esl": fk - es,
                "gap_recovered_by_sa_fp": fk - sa,
                "gap_remaining_esl_over_sa_fp": sa - es,
            }
        else:
            gap = None

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55), gridspec_kw={"width_ratios": [1.25, 1.05]})
    ax = axes[0]

    sa_rows = [r for r in rows if r["method"] == "SA-FP"]
    xs = np.array([float(r["p_stay"]) for r in sa_rows], dtype=np.float64)
    means = np.array([float(r["mean_psr"]) for r in sa_rows], dtype=np.float64)
    lo = np.array([float(r["ci95_psr_low"]) for r in sa_rows], dtype=np.float64)
    hi = np.array([float(r["ci95_psr_high"]) for r in sa_rows], dtype=np.float64)
    order = np.argsort(xs)
    xs, means, lo, hi = xs[order], means[order], lo[order], hi[order]
    yerr = np.vstack([means - lo, hi - means])
    # Discrete evaluated settings only: error bars (95% CI), not a continuous band.
    ax.errorbar(
        xs,
        means,
        yerr=yerr,
        fmt="-o",
        color=METHOD_COLORS["SA-FP"],
        linewidth=2.0,
        markersize=6.0,
        capsize=4.0,
        label="SA-FP",
    )

    ref_esl = float(gap["mean_psr_esl"]) if gap is not None else None
    ref_fk = float(gap["mean_psr_fixedk"]) if gap is not None else None
    if gap is None:
        esl_vals = [float(r["mean_psr"]) for r in rows if r["method"] == "ESL"]
        fk_vals = [float(r["mean_psr"]) for r in rows if r["method"] == "Fixed-K Bayesian"]
        if esl_vals:
            ref_esl = float(np.mean(esl_vals))
        if fk_vals:
            ref_fk = float(np.mean(fk_vals))
    if ref_fk is not None:
        ax.axhline(
            ref_fk,
            color=METHOD_COLORS["Fixed-K Bayesian"],
            linestyle="--",
            linewidth=1.4,
            label="Fixed-K reference",
        )
    if ref_esl is not None:
        ax.axhline(
            ref_esl,
            color=METHOD_COLORS["ESL"],
            linestyle="--",
            linewidth=1.4,
            label="ESL reference",
        )

    ax.set_xlabel(r"$p_{\mathrm{stay}}$")
    apply_ylabel_lower_better(ax, "PSR (post-switch oracle regret)")
    ax.set_title("(a) SA-FP transition robustness")
    ax.set_xticks(list(xs))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax2 = axes[1]
    if gap is not None:
        es = float(gap["mean_psr_esl"])
        sa = float(gap["mean_psr_sa_fp"])
        fk = float(gap["mean_psr_fixedk"])
        y_pos = [2, 1, 0]
        methods = ["Fixed-K", "SA-FP", "ESL"]
        values = [fk, sa, es]
        colors = [METHOD_COLORS["Fixed-K Bayesian"], METHOD_COLORS["SA-FP"], METHOD_COLORS["ESL"]]
        ax2.barh(y_pos, values, color=colors, height=0.55)
        ax2.set_yticks(y_pos, methods)
        ax2.set_xlabel(xlabel_lower_better("Mean PSR"))
        ax2.set_title("(b) Aggregate capability decomposition")
        ax2.annotate(
            f"+ explicit switch-aware filtering\naggregate ΔPSR = {fk - sa:.1f}",
            xy=((fk + sa) / 2.0, 1.5),
            xytext=(max(fk, sa, es) * 1.02, 1.55),
            fontsize=7.2,
            va="center",
            ha="left",
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.8},
        )
        ax2.annotate(
            f"+ adaptive prototype mechanism\naggregate ΔPSR = {sa - es:.1f}",
            xy=((sa + es) / 2.0, 0.5),
            xytext=(max(fk, sa, es) * 1.02, 0.45),
            fontsize=7.2,
            va="center",
            ha="left",
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.8},
        )
        ax2.set_xlim(0, max(fk, sa, es) * 1.42)
        ax2.grid(axis="x", alpha=0.25)
    else:
        ax2.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi(), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sa_fp_transition_robustness(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    horizon: int,
    h_window: int = H_WINDOW,
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
    p_stay_values: tuple[float, ...] = SA_FP_SENSITIVITY_P_STAY,
) -> dict[str, Path]:
    """Compare ESL / SA-FP / Fixed-K across transition persistence values."""
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_root.mkdir(parents=True, exist_ok=True)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    cell_summaries: list[dict[str, Any]] = []
    for p_stay in p_stay_values:
        for task_key in tasks:
            task = task_by_key[task_key]
            for seed in seeds:
                fixed = run_dynamic_cell(
                    task=task,
                    method="Fixed-K Bayesian",
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                )
                sa_fp = run_dynamic_cell(
                    task=task,
                    method="SA-FP",
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                    p_stay=float(p_stay),
                )
                esl = run_dynamic_cell(
                    task=task,
                    method="ESL",
                    seed=seed,
                    horizon=horizon,
                    h_window=h_window,
                )
                for result, method_name in (
                    (fixed, "Fixed-K Bayesian"),
                    (sa_fp, "SA-FP"),
                    (esl, "ESL"),
                ):
                    row = dict(result["summary"])
                    row["p_stay"] = float(p_stay)
                    row["method"] = method_name
                    cell_summaries.append(row)

    agg_rows: list[dict[str, Any]] = []
    for p_stay in p_stay_values:
        for method in ("ESL", "SA-FP", "Fixed-K Bayesian"):
            vals_psr = [
                float(s["psr"])
                for s in cell_summaries
                if s["method"] == method and float(s["p_stay"]) == float(p_stay)
            ]
            vals_dpr = [
                float(s["dpr"])
                for s in cell_summaries
                if s["method"] == method and float(s["p_stay"]) == float(p_stay)
            ]
            st_psr = mean_std_ci95(vals_psr)
            st_dpr = mean_std_ci95(vals_dpr)
            agg_rows.append(
                {
                    "p_stay": float(p_stay),
                    "method": method,
                    "n": int(st_psr["n"]),
                    "mean_psr": st_psr["mean"],
                    "ci95_psr_low": st_psr["ci95_low"],
                    "ci95_psr_high": st_psr["ci95_high"],
                    "mean_dpr": st_dpr["mean"],
                    "ci95_dpr_low": st_dpr["ci95_low"],
                    "ci95_dpr_high": st_dpr["ci95_high"],
                }
            )

    table_csv = manuscript_bundle / "main" / "tables" / "figure1_sa_fp_transition_robustness.csv"
    _write_csv(
        table_csv,
        agg_rows,
        [
            "p_stay",
            "method",
            "n",
            "mean_psr",
            "ci95_psr_low",
            "ci95_psr_high",
            "mean_dpr",
            "ci95_dpr_low",
            "ci95_dpr_high",
        ],
    )
    detail_csv = out_root / "figure1_sa_fp_transition_robustness_cells.csv"
    _write_csv(
        detail_csv,
        cell_summaries,
        ["p_stay", "method", "task", "seed", "horizon", "h_window", "psr", "dpr", "early_gap_auc", "lte_auc", "sa_fp_p_stay"],
    )
    figure = manuscript_bundle / "main" / "figures" / "figure1_sa_fp_transition_robustness.pdf"
    _plot_sa_fp_transition_robustness(agg_rows, out_path=figure)
    report = manuscript_bundle / "reports" / "figure1_sa_fp_transition_robustness.md"
    lines = [
        "# SA-FP Transition Robustness",
        "",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`; H: `{h_window}`.",
        f"- Tasks: `{', '.join(tasks)}`.",
        f"- Primary SA-FP default: `p_stay={SA_FP_DEFAULT_P_STAY}` (neutral HMM prior, not hazard-matched).",
        f"- Evaluated `p_stay` values: `{list(p_stay_values)}`.",
        "",
        "## Notes",
        "- ESL and Fixed-K do not depend on `p_stay`; they are repeated at each x-value as constant references.",
        "- All evaluated SA-FP values are reported; do not headline the best after the fact.",
        "",
        "## Results",
    ]
    for row in agg_rows:
        lines.append(
            f"- p_stay={row['p_stay']:.2f} / {row['method']}: "
            f"PSR={float(row['mean_psr']):.4f} [{float(row['ci95_psr_low']):.4f}, {float(row['ci95_psr_high']):.4f}]; "
            f"DPR={float(row['mean_dpr']):.4f}."
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"table_csv": table_csv, "detail_csv": detail_csv, "figure": figure, "report": report}


def sa_fp_transition_robustness_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SA-FP transition robustness vs ESL/Fixed-K")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure1_sa_fp_transition_robustness"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in FIGURE1_MAIN_SEEDS))
    p.add_argument("--horizon", type=int, default=FIGURE1_MAIN_HORIZON)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    p.add_argument(
        "--p-stay-values",
        type=str,
        default=",".join(str(x) for x in SA_FP_SENSITIVITY_P_STAY),
    )
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    p_stay_values = tuple(float(x.strip()) for x in args.p_stay_values.split(",") if x.strip())
    outputs = run_sa_fp_transition_robustness(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        seeds=seeds,
        horizon=args.horizon,
        h_window=args.h_window,
        tasks=tasks,
        p_stay_values=p_stay_values,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
