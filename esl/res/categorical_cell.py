"""Categorical (A≥2) Tier-1 dynamic cell for RES — does not alter the paper Figure-1 A=2 path."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.baselines.external_common import MatrixGameTask
from esl.experiments.figure1_dynamic_adaptation import (
    EARLY_WINDOW,
    FOCAL_AGENT_ID,
    H_WINDOW,
    K_TYPES,
    LAMBDA_BR,
    MATURATION_WINDOW,
    ONLINE_EM_WINDOW,
    PPO_OBSERVATION_FIELDS,
    PPO_UPDATE_EVERY,
    SA_FP_DEFAULT_P_STAY,
    SOM_WINDOW,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
    DynamicScheduleStep,
    ESLUpdateConfig,
    FIGURE1_DEFAULT_ESL_UPDATE,
    FIXED_K_PROTOTYPE_SOURCE,
    _schedule_task_index,
    compute_dynamic_prediction_regret,
    compute_early_adaptation_gap_auc,
    compute_post_switch_oracle_regret,
    markov_predict,
)
from esl.experiments.figure1_learning_curves import _task_payoff
from esl.metrics import cross_entropy, match_prototypes_to_types
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax
from esl.res.omd_mwu import (
    OMD_DEFAULT_SCOPE,
    OMD_PRIMARY_C,
    OmdScope,
    focal_reward_vector,
    normalize_matrix_once,
    omd_init_log_weights,
    omd_init_per_opponent,
    omd_policy,
    omd_step,
)
from esl.res.hmml_categorical import (
    empty_hmml_window,
    filtered_responsibility,
    update_hmml_emissions,
)
from esl.res.online_em_categorical import update_online_em_categorical

Tier1Method = Literal[
    "ESL",
    "Fixed-K Bayesian",
    "SA-FP",
    "FP",
    "SOM",
    "OMD/MWU",
    "Online EM",
    "HMM-L",
    "PPO",
]
EPS = 1e-12
DEFAULT_NUM_OPPONENTS = 3


def opponent_ids(num_opponents: int) -> tuple[int, ...]:
    j = int(num_opponents)
    if j < 1:
        raise ValueError("num_opponents must be >= 1")
    return tuple(range(1, j + 1))


def exposure_matched_horizon(num_opponents: int, *, base_horizon: int = 1000, base_opponents: int = 3) -> int:
    """Match expected focal–opponent encounters per opponent (one opp sampled per round)."""
    return int(round(int(base_horizon) * int(num_opponents) / int(base_opponents)))


def exposure_matched_switch_interval(
    num_opponents: int,
    *,
    base_interval: int = SWITCH_INTERVAL,
    base_opponents: int = 3,
) -> int:
    """Scale global switch interval so opponents see ~same encounters between interval switches."""
    return int(round(int(base_interval) * int(num_opponents) / int(base_opponents)))


def exposure_matched_switch_hazard(
    num_opponents: int,
    *,
    base_hazard: float = SWITCH_HAZARD,
    base_opponents: int = 3,
) -> float:
    """Scale per-round hazard so expected global hazard switches per opponent is stable."""
    return float(base_hazard) * float(base_opponents) / float(num_opponents)


# --- Encounter-clock opponent-count protocol (apples-to-apples primary) ---
# Switch and evaluate in per-opponent encounter time, not global rounds.
ENC_PRE_SWITCH = 60  # encounters under current type before each switch
ENC_H_E = 30  # post-switch evaluation encounters H_e
ENC_N_SWITCHES = 5  # evaluated switches per opponent
ENC_CYCLE = ENC_PRE_SWITCH + ENC_H_E  # 90
ENC_PER_OPPONENT = ENC_N_SWITCHES * ENC_CYCLE  # 450
ENC_SWITCH_POINTS = tuple(ENC_PRE_SWITCH + i * ENC_CYCLE for i in range(ENC_N_SWITCHES))
# 60, 150, 240, 330, 420 — after last switch, H_e encounters remain within 450


def encounter_clock_horizon(num_opponents: int) -> int:
    """Global T so each opponent expects ENC_PER_OPPONENT uniform samples."""
    return int(ENC_PER_OPPONENT) * int(num_opponents)


def build_encounter_clock_schedule(
    *,
    task: MatrixGameTask,
    seed: int,
    num_actions: int,
    num_opponents: int,
    encounters_per_opponent: int = ENC_PER_OPPONENT,
    switch_points: tuple[int, ...] = ENC_SWITCH_POINTS,
) -> list[DynamicScheduleStep]:
    """Switch types on per-opponent encounter counts (not global round).

    For each opponent j, maintain encounter counter c_j. After the c_j-th
    encounter, if c_j is a locked switch point, j's latent type advances for
    subsequent encounters. One opponent is sampled uniformly each global round;
    horizon is encounters_per_opponent * J.

    Locked points (60, 150, 240, 330, 420) yield 60 pre-switch encounters under
    the old type, then H_e=30 post-switch encounters under the new type, ×5.
    """
    j = int(num_opponents)
    opp_list = opponent_ids(j)
    horizon = int(encounters_per_opponent) * j
    switch_set = {int(x) for x in switch_points}
    task_index = _schedule_task_index(task)
    rng = np.random.default_rng(seed + 1009 * task_index + 17)

    # Exact encounters per opponent (shuffled order): stronger than expectation-only uniform.
    encounter_order = np.array(
        [opp for opp in opp_list for _ in range(int(encounters_per_opponent))],
        dtype=np.int64,
    )
    rng.shuffle(encounter_order)

    current_types = {opp: (opp - 1) % K_TYPES for opp in opp_list}
    # Encounters completed under current type (0-based age for the upcoming encounter).
    type_ages = {opp: 0 for opp in opp_list}
    switch_counts = {opp: -1 for opp in opp_list}
    encounter_counts = {opp: 0 for opp in opp_list}
    last_switch_round: dict[int, int | str] = {opp: "" for opp in opp_list}
    old_type_meta: dict[int, int | str] = {opp: "" for opp in opp_list}
    new_type_meta: dict[int, int | str] = {opp: "" for opp in opp_list}
    steps: list[DynamicScheduleStep] = []

    for t in range(horizon):
        opponent_id = int(encounter_order[t])
        true_type = current_types[opponent_id]
        age = type_ages[opponent_id]
        # tau = encounters since last switch for this opponent (0 on first post-switch encounter).
        tau = age if switch_counts[opponent_id] >= 0 else ENC_H_E + 1
        steps.append(
            DynamicScheduleStep(
                round=t,
                opponent_id=opponent_id,
                true_type=true_type,
                type_age=age,
                rounds_since_switch=tau,
                switch_id=switch_counts[opponent_id],
                switch_round=last_switch_round[opponent_id],
                old_type=old_type_meta[opponent_id],
                new_type=new_type_meta[opponent_id],
                random_u=float(rng.random()),
                true_action_probs=true_type_probs_categorical(true_type, age, num_actions),
            )
        )
        encounter_counts[opponent_id] += 1
        type_ages[opponent_id] += 1
        c = encounter_counts[opponent_id]
        if c in switch_set:
            old = current_types[opponent_id]
            current_types[opponent_id] = (old + 1) % K_TYPES
            switch_counts[opponent_id] += 1
            type_ages[opponent_id] = 0
            last_switch_round[opponent_id] = t
            old_type_meta[opponent_id] = old
            new_type_meta[opponent_id] = current_types[opponent_id]
    return steps


@dataclass
class CategoricalMethodState:
    method: Tier1Method
    rng: np.random.Generator
    num_actions: int
    logits: np.ndarray | None = None
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    esl_batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    fp_counts: dict[int, np.ndarray] = field(default_factory=dict)
    som_windows: dict[int, deque[tuple[int, int]]] = field(default_factory=dict)
    em_window: deque[tuple[int, int]] = field(default_factory=deque)
    # HMM-L: sliding window of (action, transition-aware γ); predictive cache for once-predict.
    hmml_window: deque[tuple[int, np.ndarray]] = field(default_factory=deque)
    hmml_pred_cache: dict[int, np.ndarray] = field(default_factory=dict)
    omd_log_weights: np.ndarray | None = None
    omd_log_weights_by_opp: dict[int, np.ndarray] = field(default_factory=dict)
    omd_scope: OmdScope = OMD_DEFAULT_SCOPE
    omd_u_min: float = 0.0
    omd_u_max: float = 0.0
    omd_c: float = OMD_PRIMARY_C
    ppo_agent: Any | None = None
    ppo_update_count: int = 0
    ppo_horizon: int = 1000
    last_focal_action: int | None = None
    last_opponent_action: int | None = None
    p_stay: float = SA_FP_DEFAULT_P_STAY
    num_opponents: int = DEFAULT_NUM_OPPONENTS
    lam_br: float = LAMBDA_BR


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z -= np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def true_type_probs_categorical(
    true_type: int,
    type_age: int,
    num_actions: int,
    *,
    last_focal_action: int | None = None,
) -> np.ndarray:
    """Peaked categorical emissions that reduce to the paper Bernoulli targets when A=2."""
    a = int(num_actions)
    maturity = min(float(type_age) / MATURATION_WINDOW, 1.0)
    if a == 2:
        ambiguous = np.array([0.52, 0.48], dtype=np.float64)
        if int(true_type) == 0:
            target = np.array([0.94, 0.06], dtype=np.float64)
        elif int(true_type) == 1:
            target = np.array([0.06, 0.94], dtype=np.float64)
        else:
            if last_focal_action in (None, 0):
                target = np.array([0.84, 0.16], dtype=np.float64)
            else:
                target = np.array([0.16, 0.84], dtype=np.float64)
        return (1.0 - maturity) * ambiguous + maturity * target

    # A≥3: type k prefers action k % A, with residual mass uniform elsewhere
    amb = np.full(a, 1.0 / a, dtype=np.float64)
    preferred = int(true_type) % a
    if int(true_type) == 2 and last_focal_action is not None:
        preferred = int(last_focal_action) % a
    peak = 0.80
    target = np.full(a, (1.0 - peak) / float(a - 1), dtype=np.float64)
    target[preferred] = peak
    return (1.0 - maturity) * amb + maturity * target


def mature_true_type_distributions(num_actions: int) -> np.ndarray:
    return np.stack(
        [true_type_probs_categorical(k, MATURATION_WINDOW, num_actions) for k in range(K_TYPES)],
        axis=0,
    )


def _esl_initial_logits(seed: int, *, k_model: int, num_actions: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = int(num_actions)
    base = rng.normal(0.0, 0.5, size=(k_model, a))
    # Encourage prototype k to prefer action k % A
    for k in range(k_model):
        base[k, k % a] += 1.0
    return base


# Provenance: pre-specified, game-independent, non-oracle misspecified random logits.
# Seed 314159; mild preference of prototype k for action k%A. Not fitted to the benchmark
# and not derived from true type emissions. Same source string as FIXED_K_PROTOTYPE_SOURCE.
CATEGORICAL_FIXED_LOGITS_SOURCE = FIXED_K_PROTOTYPE_SOURCE


def _fixed_logits(num_actions: int) -> np.ndarray:
    """Deterministic misspecified random logits (see CATEGORICAL_FIXED_LOGITS_SOURCE).

    Callers that share this tensor across methods must use ``.copy()`` to avoid
    in-place ESL updates mutating another method's state.
    """
    rng = np.random.default_rng(314159)
    a = int(num_actions)
    base = rng.normal(0.0, 0.4, size=(K_TYPES, a))
    for k in range(K_TYPES):
        base[k, k % a] += 0.8
    return base


PrototypeInit = Literal["default", "fixed"]


def build_categorical_schedule(
    *,
    task: MatrixGameTask,
    seed: int,
    horizon: int,
    num_actions: int,
    num_opponents: int = DEFAULT_NUM_OPPONENTS,
    switch_interval: int = SWITCH_INTERVAL,
    switch_hazard: float = SWITCH_HAZARD,
) -> list[DynamicScheduleStep]:
    opp_list = opponent_ids(num_opponents)
    task_index = _schedule_task_index(task)
    rng = np.random.default_rng(seed + 1009 * task_index + 17)
    current_types = {opp: (opp - 1) % K_TYPES for opp in opp_list}
    type_ages = {opp: 0 for opp in opp_list}
    switch_counts = {opp: -1 for opp in opp_list}
    last_switch_round: dict[int, int | str] = {opp: "" for opp in opp_list}
    old_type_meta: dict[int, int | str] = {opp: "" for opp in opp_list}
    new_type_meta: dict[int, int | str] = {opp: "" for opp in opp_list}
    steps: list[DynamicScheduleStep] = []
    interval = int(switch_interval)
    hazard = float(switch_hazard)

    for t in range(horizon):
        for opp in opp_list:
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

        opponent_id = int(rng.choice(opp_list))
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
                true_action_probs=true_type_probs_categorical(
                    true_type, type_ages[opponent_id], num_actions
                ),
            )
        )
        for opp in opp_list:
            type_ages[opp] += 1
    return steps


def _method_state(
    method: Tier1Method,
    seed: int,
    *,
    num_actions: int,
    num_opponents: int = DEFAULT_NUM_OPPONENTS,
    p_stay: float,
    omd_c: float = OMD_PRIMARY_C,
    omd_scope: OmdScope = OMD_DEFAULT_SCOPE,
    pay_row: np.ndarray | None = None,
    horizon: int = 1000,
    lam_br: float = LAMBDA_BR,
    prototype_init: PrototypeInit = "default",
) -> CategoricalMethodState:
    if prototype_init not in ("default", "fixed"):
        raise ValueError(f"prototype_init must be default|fixed, got {prototype_init!r}")
    a = int(num_actions)
    opps = opponent_ids(num_opponents)
    state = CategoricalMethodState(
        method=method,
        rng=np.random.default_rng(seed),
        num_actions=a,
        p_stay=float(p_stay),
        omd_c=float(omd_c),
        omd_scope=omd_scope,
        ppo_horizon=int(horizon),
        num_opponents=int(num_opponents),
        lam_br=float(lam_br),
    )
    if method == "ESL":
        if prototype_init == "fixed":
            state.logits = _fixed_logits(a).copy()
        else:
            state.logits = _esl_initial_logits(seed, k_model=K_TYPES, num_actions=a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in opps}
    elif method in ("Fixed-K Bayesian", "SA-FP"):
        # Always frozen misspecified dictionary; .copy() prevents cross-cell leakage.
        state.logits = _fixed_logits(a).copy()
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in opps}
    elif method == "Online EM":
        state.logits = _esl_initial_logits(seed + 311, k_model=K_TYPES, num_actions=a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in opps}
        state.em_window = deque(maxlen=ONLINE_EM_WINDOW)
    elif method == "HMM-L":
        # Exact Online-EM-matched init (same seed offset, window length, beliefs).
        state.logits = _esl_initial_logits(seed + 311, k_model=K_TYPES, num_actions=a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in opps}
        state.hmml_window = empty_hmml_window(ONLINE_EM_WINDOW)
        state.hmml_pred_cache = {}
    elif method == "FP":
        state.fp_counts = {opp: np.ones(a, dtype=np.float64) for opp in opps}
    elif method == "SOM":
        state.som_windows = {opp: deque(maxlen=SOM_WINDOW) for opp in opps}
    elif method == "PPO":
        from esl.baselines.ppo_adapter import _load_official_ppo_module

        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        ppo_mod = _load_official_ppo_module()
        # Same hyperparameters as Figure-1; only action_dim follows num_actions (no retuning).
        state.ppo_agent = ppo_mod.PPO(
            len(PPO_OBSERVATION_FIELDS),
            a,
            3e-4,
            1e-3,
            0.99,
            4,
            0.2,
            False,
            action_std_init=0.6,
        )
    elif method == "OMD/MWU":
        if pay_row is None:
            raise ValueError("OMD/MWU requires pay_row for once-per-matrix normalization")
        _scaled, lo, hi = normalize_matrix_once(pay_row)
        state.omd_u_min = float(lo)
        state.omd_u_max = float(hi)
        if omd_scope == "per_opponent":
            state.omd_log_weights_by_opp = omd_init_per_opponent(a, opps)
            state.omd_log_weights = None
        elif omd_scope == "global":
            state.omd_log_weights = omd_init_log_weights(a)
            state.omd_log_weights_by_opp = {}
        else:
            raise ValueError(f"unknown omd_scope={omd_scope!r}")
    return state


def _ppo_observation(state: CategoricalMethodState, *, step: DynamicScheduleStep) -> np.ndarray:
    return np.array(
        [
            float(step.round) / max(float(state.ppo_horizon), 1.0),
            float(step.opponent_id) / max(float(state.num_opponents), 1.0),
            float(state.last_opponent_action if state.last_opponent_action is not None else 0),
            float(state.last_focal_action if state.last_focal_action is not None else 0),
            1.0,
        ],
        dtype=np.float32,
    )


def _hmml_predictive_belief(state: CategoricalMethodState, opponent_id: int) -> np.ndarray:
    """Once-predict cache: b̄ = T^⊤ b for this opponent; reused by the observation update."""
    cached = state.hmml_pred_cache.get(int(opponent_id))
    if cached is not None:
        return cached
    bar = markov_predict(state.beliefs[int(opponent_id)], state.p_stay)
    state.hmml_pred_cache[int(opponent_id)] = bar
    return bar


def _predicted_probs(state: CategoricalMethodState, opponent_id: int) -> np.ndarray:
    a = state.num_actions
    if state.method in ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM", "HMM-L"):
        assert state.logits is not None
        belief = state.beliefs[opponent_id]
        if state.method == "SA-FP":
            belief = markov_predict(belief, state.p_stay)
        elif state.method == "HMM-L":
            belief = _hmml_predictive_belief(state, opponent_id)
        return belief @ stable_softmax(state.logits)
    if state.method == "FP":
        counts = state.fp_counts[opponent_id]
        return counts / counts.sum()
    if state.method in ("OMD/MWU", "PPO"):
        # No opponent model — uniform predictive prior (Figure-1 PPO convention).
        return np.full(a, 1.0 / a, dtype=np.float64)
    # SOM: conditional empirical categorical
    rows = list(state.som_windows[opponent_id])
    if not rows:
        return np.full(a, 1.0 / a, dtype=np.float64)
    relevant = [
        a_opp
        for a_self, a_opp in rows
        if state.last_focal_action is not None and a_self == state.last_focal_action
    ]
    actions = relevant if relevant else [a_opp for _a_self, a_opp in rows]
    counts = np.ones(a, dtype=np.float64)
    for act in actions:
        counts[int(act)] += 1.0
    return counts / counts.sum()


def _focal_action(
    state: CategoricalMethodState,
    pay: games.PayoffMatrices,
    opponent_id: int,
    *,
    step: DynamicScheduleStep | None = None,
) -> int:
    if state.method == "OMD/MWU":
        if state.omd_scope == "per_opponent":
            z = state.omd_log_weights_by_opp[int(opponent_id)]
        else:
            assert state.omd_log_weights is not None
            z = state.omd_log_weights
        probs = omd_policy(z)
        return int(state.rng.choice(state.num_actions, p=probs))
    if state.method == "PPO":
        if state.ppo_agent is None or step is None:
            raise RuntimeError("PPO agent/step was not initialized")
        return int(state.ppo_agent.select_action(_ppo_observation(state, step=step)))
    pred = _predicted_probs(state, opponent_id)
    utilities = np.array([np.sum(pred * pay.row[act, :]) for act in range(state.num_actions)], dtype=np.float64)
    lam = 1.4 if state.method in ("FP", "SOM") else float(state.lam_br)
    probs = _softmax(utilities, lam=lam)
    return int(state.rng.choice(state.num_actions, p=probs))


def _oracle_payoff(true_probs: np.ndarray, pay: games.PayoffMatrices) -> float:
    a = int(pay.row.shape[0])
    utilities = np.array([np.sum(true_probs * pay.row[act, :]) for act in range(a)], dtype=np.float64)
    return float(np.max(utilities))


def _bayes_step(prior: np.ndarray, prototypes: np.ndarray, opponent_action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, opponent_action), 1e-4, EPS)


def _sa_fp_missing_or_full_update(
    state: CategoricalMethodState,
    *,
    step: DynamicScheduleStep,
    observed: bool,
    observed_action: int,
) -> None:
    """
    SA-FP Option A (HMM): always Markov-predict; Bayes correct only if observed.

    ESL / Fixed-K remain event-driven (no update when unobserved) — see SAFP_W_SEMANTICS.md.
    """
    assert state.logits is not None
    prior = markov_predict(state.beliefs[step.opponent_id], state.p_stay)
    if observed:
        state.beliefs[step.opponent_id] = _bayes_step(prior, state.logits, int(observed_action))
    else:
        state.beliefs[step.opponent_id] = prior


def _hmml_missing_or_full_update(
    state: CategoricalMethodState,
    *,
    step: DynamicScheduleStep,
    observed: bool,
    observed_action: int,
    freeze_emissions: bool = False,
) -> None:
    """HMM-L: once-predict (cached) → filtered γ; M-step on stored γ only (no free E-step)."""
    assert state.logits is not None
    opp = int(step.opponent_id)
    bar = _hmml_predictive_belief(state, opp)
    if not observed:
        state.beliefs[opp] = bar.copy()
        state.hmml_pred_cache.pop(opp, None)
        return
    _bar, gamma = filtered_responsibility(
        state.beliefs[opp],
        state.logits,
        int(observed_action),
        p_stay=state.p_stay,
        predictive_prior=bar,
    )
    state.beliefs[opp] = gamma.copy()
    state.hmml_window.append((int(observed_action), gamma.copy()))
    if not freeze_emissions and state.hmml_window:
        state.logits = update_hmml_emissions(
            state.logits,
            list(state.hmml_window),
            num_actions=state.num_actions,
        )
    state.hmml_pred_cache.pop(opp, None)


def _update(
    state: CategoricalMethodState,
    *,
    step: DynamicScheduleStep,
    focal_action: int,
    opponent_action: int,
    esl_update: ESLUpdateConfig,
) -> float | str:
    prototype_update_norm: float | str = ""
    if state.method == "ESL":
        assert state.logits is not None
        belief_before = state.beliefs[step.opponent_id].copy()
        state.beliefs[step.opponent_id] = _bayes_step(state.beliefs[step.opponent_id], state.logits, opponent_action)
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
        state.beliefs[step.opponent_id] = _bayes_step(state.beliefs[step.opponent_id], state.logits, opponent_action)
    elif state.method == "SA-FP":
        _sa_fp_missing_or_full_update(
            state, step=step, observed=True, observed_action=int(opponent_action)
        )
    elif state.method == "FP":
        state.fp_counts[step.opponent_id][opponent_action] += 1.0
    elif state.method == "SOM":
        state.som_windows[step.opponent_id].append((focal_action, opponent_action))
    elif state.method == "Online EM":
        assert state.logits is not None
        state.em_window.append((step.opponent_id, opponent_action))
        state.logits, state.beliefs = update_online_em_categorical(
            state.logits,
            beliefs=state.beliefs,
            em_window=list(state.em_window),
            num_actions=state.num_actions,
        )
    elif state.method == "OMD/MWU":
        raise RuntimeError("OMD/MWU updates go through _omd_observe_update with pay_row")
    elif state.method == "PPO":
        raise RuntimeError("PPO updates go through _ppo_reward_update")
    state.last_focal_action = focal_action
    state.last_opponent_action = opponent_action
    return prototype_update_norm


def _ppo_reward_update(
    state: CategoricalMethodState,
    *,
    step: DynamicScheduleStep,
    focal_action: int,
    focal_payoff: float,
    last_opponent_action: int | None,
) -> None:
    """PPO learns from focal payoff every round (Figure-1 convention; independent of W)."""
    if state.ppo_agent is None:
        raise RuntimeError("PPO agent was not initialized")
    state.ppo_agent.buffer.rewards.append(float(focal_payoff))
    state.ppo_agent.buffer.is_terminals.append(False)
    if (int(step.round) + 1) % int(PPO_UPDATE_EVERY) == 0:
        state.ppo_agent.update()
        state.ppo_update_count += 1
    state.last_focal_action = focal_action
    if last_opponent_action is not None:
        state.last_opponent_action = int(last_opponent_action)


def _lte(
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


def _lte_auc(rows: list[dict[str, Any]], *, h_window: int) -> float | str:
    vals = [float(r["lte"]) for r in rows if r.get("lte", "") != "" and int(r["tau"]) < h_window]
    if not vals:
        return ""
    return float(np.mean(vals))


def _omd_observe_update(
    state: CategoricalMethodState,
    *,
    pay_row: np.ndarray,
    opponent_action: int,
    opponent_id: int,
) -> None:
    r = focal_reward_vector(
        pay_row,
        opponent_action=int(opponent_action),
        u_min=state.omd_u_min,
        u_max=state.omd_u_max,
    )
    eta = float(state.omd_c)
    if state.omd_scope == "per_opponent":
        j = int(opponent_id)
        state.omd_log_weights_by_opp[j] = omd_step(state.omd_log_weights_by_opp[j], r, eta=eta)
    else:
        assert state.omd_log_weights is not None
        state.omd_log_weights = omd_step(state.omd_log_weights, r, eta=eta)


def run_categorical_cell(
    *,
    task: MatrixGameTask,
    method: Tier1Method,
    seed: int,
    horizon: int,
    num_actions: int,
    num_opponents: int = DEFAULT_NUM_OPPONENTS,
    h_window: int = H_WINDOW,
    p_stay: float = SA_FP_DEFAULT_P_STAY,
    switch_interval: int = SWITCH_INTERVAL,
    switch_hazard: float = SWITCH_HAZARD,
    esl_update: ESLUpdateConfig = FIGURE1_DEFAULT_ESL_UPDATE,
    p_obs: float = 1.0,
    obs_noise: float = 0.0,
    omd_c: float = OMD_PRIMARY_C,
    omd_scope: OmdScope = OMD_DEFAULT_SCOPE,
    switch_clock: Literal["global", "encounter"] = "global",
    lam_br: float = LAMBDA_BR,
    prototype_init: PrototypeInit = "default",
    split_exogenous_rng: bool = False,
) -> dict[str, Any]:
    from esl.signals import sample_noisy_action

    if not (0.0 <= float(p_obs) <= 1.0):
        raise ValueError("p_obs must be in [0, 1]")
    if not (0.0 <= float(obs_noise) < 1.0):
        raise ValueError("obs_noise must be in [0, 1)")
    if omd_scope not in ("per_opponent", "global"):
        raise ValueError(f"omd_scope must be per_opponent|global, got {omd_scope!r}")
    if prototype_init not in ("default", "fixed"):
        raise ValueError(f"prototype_init must be default|fixed, got {prototype_init!r}")

    pay = _task_payoff(task)
    if pay.row.shape != (num_actions, num_actions):
        raise ValueError(f"payoff shape {pay.row.shape} != ({num_actions},{num_actions})")
    if switch_clock == "encounter":
        schedule = build_encounter_clock_schedule(
            task=task,
            seed=seed,
            num_actions=num_actions,
            num_opponents=num_opponents,
        )
        horizon = len(schedule)
    elif switch_clock == "global":
        schedule = build_categorical_schedule(
            task=task,
            seed=seed,
            horizon=horizon,
            num_actions=num_actions,
            num_opponents=num_opponents,
            switch_interval=switch_interval,
            switch_hazard=switch_hazard,
        )
    else:
        raise ValueError(f"unknown switch_clock={switch_clock!r}")
    state = _method_state(
        method,
        seed + 7919,
        num_actions=num_actions,
        num_opponents=num_opponents,
        p_stay=p_stay,
        omd_c=float(omd_c),
        omd_scope=omd_scope,
        pay_row=pay.row,
        horizon=horizon,
        lam_br=float(lam_br),
        prototype_init=prototype_init,
    )
    mature = mature_true_type_distributions(num_actions)
    rows: list[dict[str, Any]] = []
    latent = method in ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM", "HMM-L")
    # Observation stream: always cell-keyed, independent of method RNG consumption.
    obs_rng = np.random.default_rng(seed + 4242)
    # Opponent-action stream: optional split so focal sampling cannot desynchronize
    # exogenous opponent draws across methods (matched-init control). Default False
    # preserves flagship bit-exactness (opponent draws shared state.rng with focal).
    opp_rng = np.random.default_rng(seed + 17_389) if split_exogenous_rng else state.rng

    for step in schedule:
        pred_probs = _predicted_probs(state, step.opponent_id)
        true_probs = step.true_action_probs
        if int(step.true_type) == 2:
            true_probs = true_type_probs_categorical(
                step.true_type,
                step.type_age,
                num_actions,
                last_focal_action=state.last_focal_action,
            )
        focal_action = _focal_action(state, pay, step.opponent_id, step=step)
        opponent_action = int(opp_rng.choice(num_actions, p=np.asarray(true_probs, dtype=np.float64)))
        focal_payoff, _opp = games.play_pair_payoffs(focal_action, opponent_action, pay)
        oracle_payoff = _oracle_payoff(true_probs, pay)
        prediction_ce = cross_entropy(np.eye(num_actions)[opponent_action], pred_probs)
        oracle_ce = cross_entropy(np.eye(num_actions)[opponent_action], true_probs)
        lte: float | str = ""
        belief_payload = ""
        if latent:
            assert state.logits is not None
            belief = state.beliefs[step.opponent_id]
            if method == "SA-FP":
                belief = markov_predict(belief, state.p_stay)
            elif method == "HMM-L":
                belief = _hmml_predictive_belief(state, step.opponent_id)
            belief_payload = json.dumps([float(x) for x in belief])
            lte = _lte(belief, true_type=step.true_type, learned_logits=state.logits, true_probs=mature)

        # Visibility W ~ Bernoulli(p_obs); independently, observation noise on the signal.
        observed = float(obs_rng.random()) < float(p_obs)
        observed_action = opponent_action
        if observed and float(obs_noise) > 0.0:
            observed_action = sample_noisy_action(
                opponent_action,
                num_actions=num_actions,
                eps=float(obs_noise),
                rng=obs_rng,
            )

        row = {
            "round": step.round,
            "tau": step.rounds_since_switch,
            "task": task.key,
            "method": method,
            "seed": seed,
            "opponent_id": step.opponent_id,
            "true_type": step.true_type,
            "switch_id": step.switch_id,
            "focal_action": focal_action,
            "opponent_action": opponent_action,
            "observed": int(observed),
            "observed_action": observed_action if observed else "",
            "focal_payoff": float(focal_payoff),
            "oracle_payoff": oracle_payoff,
            "prediction_ce": float(prediction_ce),
            "oracle_ce": float(oracle_ce),
            "psr_gap": float(oracle_payoff - focal_payoff),
            "dpr_gap": float(prediction_ce - oracle_ce) if method != "OMD/MWU" else "",
            "lte": lte,
            "belief": belief_payload,
        }
        if method == "PPO":
            # Reward always available; W only gates opponent-signal methods.
            _ppo_reward_update(
                state,
                step=step,
                focal_action=focal_action,
                focal_payoff=float(focal_payoff),
                last_opponent_action=int(observed_action) if observed else None,
            )
        elif observed:
            if state.method == "SA-FP":
                _sa_fp_missing_or_full_update(
                    state,
                    step=step,
                    observed=True,
                    observed_action=int(observed_action),
                )
                state.last_focal_action = focal_action
                state.last_opponent_action = int(observed_action)
            elif state.method == "HMM-L":
                _hmml_missing_or_full_update(
                    state,
                    step=step,
                    observed=True,
                    observed_action=int(observed_action),
                )
                state.last_focal_action = focal_action
                state.last_opponent_action = int(observed_action)
            elif state.method == "OMD/MWU":
                _omd_observe_update(
                    state,
                    pay_row=pay.row,
                    opponent_action=int(observed_action),
                    opponent_id=int(step.opponent_id),
                )
                state.last_focal_action = focal_action
                state.last_opponent_action = int(observed_action)
            else:
                _update(
                    state,
                    step=step,
                    focal_action=focal_action,
                    opponent_action=int(observed_action),
                    esl_update=esl_update,
                )
        else:
            if state.method == "SA-FP":
                # Option A: time update without measurement
                _sa_fp_missing_or_full_update(
                    state,
                    step=step,
                    observed=False,
                    observed_action=0,
                )
            elif state.method == "HMM-L":
                _hmml_missing_or_full_update(
                    state,
                    step=step,
                    observed=False,
                    observed_action=0,
                )
            # ESL / Fixed-K / FP / SOM / Online EM / OMD: no learning update when W=0
            state.last_focal_action = focal_action
        rows.append(row)

    dpr_val: float | str
    if method == "OMD/MWU":
        dpr_val = "N/A"
    else:
        dpr_val = compute_dynamic_prediction_regret(rows, h_window=h_window)

    summary = {
        "method": method,
        "task": task.key,
        "seed": seed,
        "horizon": horizon,
        "h_window": h_window,
        "num_actions": num_actions,
        "num_opponents": int(num_opponents),
        "p_obs": float(p_obs),
        "obs_noise": float(obs_noise),
        "lam_br": float(lam_br),
        "prototype_init": str(prototype_init),
        "split_exogenous_rng": bool(split_exogenous_rng),
        "focal_agent_id": FOCAL_AGENT_ID,
        "psr": compute_post_switch_oracle_regret(rows, h_window=h_window),
        "dpr": dpr_val,
        "lte_auc": _lte_auc(rows, h_window=h_window) if latent else "",
        "early_gap_auc": compute_early_adaptation_gap_auc(rows, early_window=EARLY_WINDOW),
        "sa_fp_p_stay": float(p_stay) if method in ("SA-FP", "HMM-L") else "",
        "fixed_k_prototype_source": (
            CATEGORICAL_FIXED_LOGITS_SOURCE
            if method in ("Fixed-K Bayesian", "SA-FP")
            or (method == "ESL" and prototype_init == "fixed")
            else ""
        ),
        "som_window": SOM_WINDOW if method == "SOM" else "",
        "online_em_window": ONLINE_EM_WINDOW if method in ("Online EM", "HMM-L") else "",
        "hmml_transition_aware": True if method == "HMM-L" else "",
        "omd_c": float(omd_c) if method == "OMD/MWU" else "",
        "omd_scope": str(omd_scope) if method == "OMD/MWU" else "",
        "omd_u_min": float(state.omd_u_min) if method == "OMD/MWU" else "",
        "omd_u_max": float(state.omd_u_max) if method == "OMD/MWU" else "",
        "ppo_update_count": int(state.ppo_update_count) if method == "PPO" else "",
        "switch_clock": str(switch_clock),
    }
    return {"trajectory": rows, "summary": summary}
