"""A=3 ablation grid + α/M sensitivity sweeps — reviewer response experiments.

Methods:
  ESL, No-Sharing, Fixed-K, No-Belief, Online EM
Each ablation removes exactly one mechanism (see outputs/ablation_a3/ablation_semantics.md).
Critically, No-Sharing and No-Belief use the **same** γ and M as ESL so differences are
purely mechanistic (unlike the old 2×2 code which used different rates).

Also provides α-sweep and M-sweep runners (ESL-only, clean benchmark).
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl import beliefs as belief_ops, games
from esl.baselines.external_common import MatrixGameTask
from esl.experiments.figure1_dynamic_adaptation import (
    EARLY_WINDOW,
    ESLUpdateConfig,
    FIGURE1_DEFAULT_ESL_UPDATE,
    FIGURE1_MAIN_HORIZON,
    FIGURE1_MAIN_SEEDS,
    FIXED_K_PROTOTYPE_SOURCE,
    H_WINDOW,
    K_TYPES,
    LAMBDA_BR,
    MATURATION_WINDOW,
    ONLINE_EM_WINDOW,
    SA_FP_DEFAULT_P_STAY,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
    DynamicScheduleStep,
    compute_dynamic_prediction_regret,
    compute_early_adaptation_gap_auc,
    compute_post_switch_oracle_regret,
    markov_predict,
)
from esl.experiments.figure1_learning_curves import _task_payoff
from esl.metrics import cross_entropy, match_prototypes_to_types
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax
from esl.res.benchmark import load_suite
from esl.res.categorical_cell import (
    CategoricalMethodState,
    EPS,
    _esl_initial_logits,
    _fixed_logits,
    _focal_action,
    _lte,
    _lte_auc,
    _method_state,
    _oracle_payoff,
    _predicted_probs,
    _sa_fp_missing_or_full_update,
    _update,
    build_categorical_schedule,
    mature_true_type_distributions,
    true_type_probs_categorical,
)
from esl.res.online_em_categorical import update_online_em_categorical
from esl.res.runner import game_spec_to_task

# ---------------------------------------------------------------------------
# Ablation method type
# ---------------------------------------------------------------------------
AblationMethod = Literal["ESL", "No-Sharing", "Fixed-K", "No-Belief", "Online EM"]
ABLATION_METHODS: tuple[AblationMethod, ...] = ("ESL", "No-Sharing", "Fixed-K", "No-Belief", "Online EM")

HORIZON = FIGURE1_MAIN_HORIZON
SEEDS: tuple[int, ...] = FIGURE1_MAIN_SEEDS  # 0..9
NUM_ACTIONS = 3

# α and M sweep grids (defaults locked)
ALPHA_GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
ALPHA_DEFAULT = 0.9
BATCH_GRID: tuple[int, ...] = (1, 2, 5, 20)
BATCH_DEFAULT = 2


# ---------------------------------------------------------------------------
# Ablation-specific state and update
# ---------------------------------------------------------------------------

@dataclass
class AblationState:
    method: AblationMethod
    rng: np.random.Generator
    num_actions: int = NUM_ACTIONS
    logits: np.ndarray | None = None
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    esl_batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    per_opponent_logits: dict[int, np.ndarray] = field(default_factory=dict)
    per_opponent_batch: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    em_window: deque[tuple[int, int]] = field(default_factory=deque)
    last_focal_action: int | None = None
    last_opponent_action: int | None = None
    p_stay: float = SA_FP_DEFAULT_P_STAY


def _init_ablation_state(
    method: AblationMethod,
    seed: int,
    *,
    num_actions: int = NUM_ACTIONS,
    gamma: float = ALPHA_DEFAULT,
    batch_size: int = BATCH_DEFAULT,
) -> AblationState:
    a = int(num_actions)
    state = AblationState(method=method, rng=np.random.default_rng(seed), num_actions=a)
    if method == "ESL":
        state.logits = _esl_initial_logits(seed, k_model=K_TYPES, num_actions=a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "No-Sharing":
        state.per_opponent_logits = {
            opp: _esl_initial_logits(seed + 101 * opp, k_model=K_TYPES, num_actions=a) for opp in (1, 2, 3)
        }
        state.per_opponent_batch = {opp: [] for opp in (1, 2, 3)}
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "Fixed-K":
        state.logits = _fixed_logits(a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "No-Belief":
        state.logits = _esl_initial_logits(seed, k_model=K_TYPES, num_actions=a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "Online EM":
        state.logits = _esl_initial_logits(seed + 311, k_model=K_TYPES, num_actions=a)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
        state.em_window = deque(maxlen=ONLINE_EM_WINDOW)
    return state


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * np.asarray(values, dtype=np.float64)
    z -= np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def _bayes_step(prior: np.ndarray, prototypes: np.ndarray, action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, action), 1e-4, EPS)


def _current_logits(state: AblationState, opponent_id: int) -> np.ndarray:
    if state.method == "No-Sharing":
        return state.per_opponent_logits[opponent_id]
    assert state.logits is not None
    return state.logits


def _predicted_opponent_probs(state: AblationState, opponent_id: int) -> np.ndarray:
    logits = _current_logits(state, opponent_id)
    if state.method == "No-Belief":
        belief = np.full(K_TYPES, 1.0 / K_TYPES)
    else:
        belief = state.beliefs[opponent_id]
    return belief @ stable_softmax(logits)


def _action_probs(pred: np.ndarray, pay: games.PayoffMatrices) -> np.ndarray:
    a = pay.row.shape[0]
    utilities = np.array([np.sum(pred * pay.row[i, :]) for i in range(a)], dtype=np.float64)
    return _softmax(utilities, lam=LAMBDA_BR)


def _oracle_payoff_abl(true_probs: np.ndarray, pay: games.PayoffMatrices) -> float:
    a = pay.row.shape[0]
    utilities = np.array([np.sum(true_probs * pay.row[i, :]) for i in range(a)], dtype=np.float64)
    return float(np.max(utilities))


def _update_ablation(
    state: AblationState,
    *,
    opponent_id: int,
    opponent_action: int,
    gamma: float,
    batch_size: int,
) -> None:
    if state.method == "ESL":
        assert state.logits is not None
        belief_before = state.beliefs[opponent_id].copy()
        state.beliefs[opponent_id] = _bayes_step(state.beliefs[opponent_id], state.logits, opponent_action)
        state.esl_batch.append((opponent_action, belief_before))
        if len(state.esl_batch) >= batch_size:
            grad = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
            state.logits = state.logits + gamma * (grad / len(state.esl_batch))
            state.esl_batch.clear()
    elif state.method == "No-Sharing":
        logits = state.per_opponent_logits[opponent_id]
        belief_before = state.beliefs[opponent_id].copy()
        state.beliefs[opponent_id] = _bayes_step(state.beliefs[opponent_id], logits, opponent_action)
        batch = state.per_opponent_batch[opponent_id]
        batch.append((opponent_action, belief_before))
        if len(batch) >= batch_size:
            grad = np.zeros_like(logits)
            for signal, belief in batch:
                grad += batch_weighted_prototype_gradient(logits, belief, signal)
            state.per_opponent_logits[opponent_id] = logits + gamma * (grad / len(batch))
            batch.clear()
    elif state.method == "Fixed-K":
        assert state.logits is not None
        state.beliefs[opponent_id] = _bayes_step(state.beliefs[opponent_id], state.logits, opponent_action)
    elif state.method == "No-Belief":
        assert state.logits is not None
        uniform = np.full(K_TYPES, 1.0 / K_TYPES)
        state.esl_batch.append((opponent_action, uniform))
        if len(state.esl_batch) >= batch_size:
            grad = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
            state.logits = state.logits + gamma * (grad / len(state.esl_batch))
            state.esl_batch.clear()
    elif state.method == "Online EM":
        assert state.logits is not None
        state.em_window.append((opponent_id, opponent_action))
        state.logits, state.beliefs = update_online_em_categorical(
            state.logits,
            beliefs=state.beliefs,
            em_window=list(state.em_window),
            num_actions=state.num_actions,
        )
    state.last_opponent_action = opponent_action


def run_ablation_cell(
    *,
    task: MatrixGameTask,
    method: AblationMethod,
    seed: int,
    gamma: float = ALPHA_DEFAULT,
    batch_size: int = BATCH_DEFAULT,
    horizon: int = HORIZON,
    num_actions: int = NUM_ACTIONS,
    h_window: int = H_WINDOW,
) -> dict[str, Any]:
    """Run one ablation cell and return summary dict."""
    pay = _task_payoff(task)
    if pay.row.shape != (num_actions, num_actions):
        raise ValueError(f"payoff shape {pay.row.shape} != ({num_actions},{num_actions})")

    schedule = build_categorical_schedule(
        task=task,
        seed=seed,
        horizon=horizon,
        num_actions=num_actions,
        switch_interval=SWITCH_INTERVAL,
        switch_hazard=SWITCH_HAZARD,
    )
    state = _init_ablation_state(method, seed + 7919, num_actions=num_actions, gamma=gamma, batch_size=batch_size)
    mature = mature_true_type_distributions(num_actions)
    rows: list[dict[str, Any]] = []
    latent = method in ("ESL", "No-Sharing", "Fixed-K", "No-Belief", "Online EM")

    for step in schedule:
        pred_probs = _predicted_opponent_probs(state, step.opponent_id)
        true_probs = step.true_action_probs
        if int(step.true_type) == 2:
            true_probs = true_type_probs_categorical(
                step.true_type, step.type_age, num_actions,
                last_focal_action=state.last_focal_action,
            )
        focal_probs = _action_probs(pred_probs, pay)
        focal_action = int(state.rng.choice(num_actions, p=focal_probs))
        opponent_action = int(state.rng.choice(num_actions, p=np.asarray(true_probs, dtype=np.float64)))
        focal_payoff, _ = games.play_pair_payoffs(focal_action, opponent_action, pay)
        oracle_payoff = _oracle_payoff_abl(true_probs, pay)
        prediction_ce = cross_entropy(np.eye(num_actions)[opponent_action], pred_probs)
        oracle_ce = cross_entropy(np.eye(num_actions)[opponent_action], true_probs)

        lte: float | str = ""
        if latent and state.logits is not None:
            belief = state.beliefs.get(step.opponent_id)
            if belief is not None:
                lte = _lte(belief, true_type=step.true_type, learned_logits=state.logits, true_probs=mature)

        row = {
            "round": step.round,
            "tau": step.rounds_since_switch,
            "task": task.key,
            "method": method,
            "seed": seed,
            "gamma": gamma,
            "batch_size": batch_size,
            "opponent_id": step.opponent_id,
            "true_type": step.true_type,
            "switch_id": step.switch_id,
            "focal_action": focal_action,
            "opponent_action": opponent_action,
            "focal_payoff": float(focal_payoff),
            "oracle_payoff": oracle_payoff,
            "prediction_ce": float(prediction_ce),
            "oracle_ce": float(oracle_ce),
            "psr_gap": float(oracle_payoff - focal_payoff),
            "dpr_gap": float(prediction_ce - oracle_ce),
            "lte": lte,
        }
        _update_ablation(state, opponent_id=step.opponent_id, opponent_action=opponent_action,
                         gamma=gamma, batch_size=batch_size)
        state.last_focal_action = focal_action
        rows.append(row)

    return {
        "method": method,
        "task": task.key,
        "seed": seed,
        "gamma": gamma,
        "batch_size": batch_size,
        "psr": compute_post_switch_oracle_regret(rows, h_window=h_window),
        "dpr": compute_dynamic_prediction_regret(rows, h_window=h_window),
        "lte_auc": _lte_auc(rows, h_window=h_window) if latent else "",
        "early_gap_auc": compute_early_adaptation_gap_auc(rows, early_window=EARLY_WINDOW),
    }


# ---------------------------------------------------------------------------
# Grid runners
# ---------------------------------------------------------------------------

def run_ablation_grid(
    *,
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    methods: tuple[AblationMethod, ...] = ABLATION_METHODS,
    gamma: float = ALPHA_DEFAULT,
    batch_size: int = BATCH_DEFAULT,
) -> Path:
    """Run full ablation grid: 12 games × seeds × methods. Returns CSV path."""
    specs = load_suite(num_actions=NUM_ACTIONS)
    if not specs:
        raise RuntimeError("No 3×3 benchmark games found")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_cells.csv"
    first = True
    for spec in specs:
        task = game_spec_to_task(spec)
        for seed in seeds:
            for method in methods:
                result = run_ablation_cell(
                    task=task, method=method, seed=seed,
                    gamma=gamma, batch_size=batch_size,
                )
                if first:
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(result.keys()))
                        w.writeheader()
                        w.writerow(result)
                    first = False
                else:
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(result.keys()))
                        w.writerow(result)
    return csv_path


def run_alpha_sweep(
    *,
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> Path:
    """ESL-only α sensitivity sweep. Returns CSV path."""
    specs = load_suite(num_actions=NUM_ACTIONS)
    if not specs:
        raise RuntimeError("No 3×3 benchmark games found")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "alpha_cells.csv"
    first = True
    for spec in specs:
        task = game_spec_to_task(spec)
        for seed in seeds:
            for alpha in alphas:
                result = run_ablation_cell(
                    task=task, method="ESL", seed=seed,
                    gamma=alpha, batch_size=BATCH_DEFAULT,
                )
                if first:
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(result.keys()))
                        w.writeheader()
                        w.writerow(result)
                    first = False
                else:
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(result.keys()))
                        w.writerow(result)
    return csv_path


def run_batch_sweep(
    *,
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    batches: tuple[int, ...] = BATCH_GRID,
) -> Path:
    """ESL-only batch-size sensitivity sweep. Returns CSV path."""
    specs = load_suite(num_actions=NUM_ACTIONS)
    if not specs:
        raise RuntimeError("No 3×3 benchmark games found")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "batch_cells.csv"
    first = True
    for spec in specs:
        task = game_spec_to_task(spec)
        for seed in seeds:
            for m in batches:
                result = run_ablation_cell(
                    task=task, method="ESL", seed=seed,
                    gamma=ALPHA_DEFAULT, batch_size=m,
                )
                if first:
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(result.keys()))
                        w.writeheader()
                        w.writerow(result)
                    first = False
                else:
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(result.keys()))
                        w.writerow(result)
    return csv_path


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def compute_paired_statistics(csv_path: Path) -> dict[str, Any]:
    """Compute paired bootstrap CI, t-test, Cohen's dz for ESL vs each ablation on PSR."""
    import csv as csv_mod
    from scipy import stats as sp_stats

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    esl_rows = {(r["task"], r["seed"]): float(r["psr"]) for r in rows if r["method"] == "ESL"}
    comparisons = [m for m in ABLATION_METHODS if m != "ESL"]
    results = []
    for method in comparisons:
        method_rows = {(r["task"], r["seed"]): float(r["psr"]) for r in rows if r["method"] == method}
        keys = sorted(set(esl_rows.keys()) & set(method_rows.keys()))
        n = len(keys)
        if n == 0:
            continue
        diffs = np.array([esl_rows[k] - method_rows[k] for k in keys])
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        se = sd_diff / math.sqrt(n) if n > 1 else float("inf")
        dz = mean_diff / sd_diff if sd_diff > 0 else float("inf")

        # paired t-test
        t_stat, p_val = sp_stats.ttest_1samp(diffs, 0)

        # bootstrap 95% CI
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(10000):
            idx = rng.choice(n, size=n, replace=True)
            boot_means.append(float(np.mean(diffs[idx])))
        boot_means.sort()
        ci_lo = boot_means[int(0.025 * len(boot_means))]
        ci_hi = boot_means[int(0.975 * len(boot_means))]

        results.append({
            "comparison": f"ESL vs {method}",
            "n": n,
            "mean_delta_psr": round(mean_diff, 4),
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_val),
            "dz": round(dz, 4),
        })

    # Holm correction
    results.sort(key=lambda r: r["p_value"])
    m_tests = len(results)
    for i, r in enumerate(results):
        r["p_holm"] = min(r["p_value"] * (m_tests - i), 1.0)

    return {"n_paired": results[0]["n"] if results else 0, "comparisons": results}


def compute_correlations(csv_path: Path) -> dict[str, Any]:
    """Pearson/Spearman PSR vs DPR and PSR vs LTE across all ablation cells."""
    from scipy import stats as sp_stats
    import csv as csv_mod

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    psr = [float(r["psr"]) for r in rows if r["psr"] not in ("", "nan")]
    dpr = [float(r["dpr"]) for r in rows if r["dpr"] not in ("", "nan", "N/A")]
    lte = [float(r["lte_auc"]) for r in rows if r["lte_auc"] not in ("", "nan")]

    out: dict[str, Any] = {}
    if len(psr) == len(dpr) and len(psr) > 2:
        pr, pp = sp_stats.pearsonr(psr, dpr)
        sr, sp = sp_stats.spearmanr(psr, dpr)
        out["psr_dpr"] = {"pearson_r": round(pr, 4), "pearson_p": pp, "spearman_r": round(sr, 4), "spearman_p": sp}

    psr_lte = [(float(r["psr"]), float(r["lte_auc"])) for r in rows
               if r["lte_auc"] not in ("", "nan") and r["psr"] not in ("", "nan")]
    if len(psr_lte) > 2:
        ps, ls = zip(*psr_lte)
        pr, pp = sp_stats.pearsonr(ps, ls)
        sr, sp = sp_stats.spearmanr(ps, ls)
        out["psr_lte"] = {"pearson_r": round(pr, 4), "pearson_p": pp, "spearman_r": round(sr, 4), "spearman_p": sp}

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="A=3 ablation / α / M experiments")
    sub = parser.add_subparsers(dest="cmd")

    p_abl = sub.add_parser("ablation", help="Run ablation grid")
    p_abl.add_argument("--out-dir", type=Path, default=Path("outputs/ablation_a3"))

    p_alpha = sub.add_parser("alpha", help="Run α sensitivity sweep")
    p_alpha.add_argument("--out-dir", type=Path, default=Path("outputs/alpha_sensitivity"))

    p_batch = sub.add_parser("batch", help="Run batch-size sensitivity sweep")
    p_batch.add_argument("--out-dir", type=Path, default=Path("outputs/batch_sensitivity"))

    p_stats = sub.add_parser("stats", help="Compute paired statistics from ablation CSV")
    p_stats.add_argument("csv", type=Path)
    p_stats.add_argument("--out-dir", type=Path, default=Path("outputs/ablation_a3"))

    args = parser.parse_args()
    if args.cmd == "ablation":
        path = run_ablation_grid(out_dir=args.out_dir)
        print(f"Ablation grid → {path}")
    elif args.cmd == "alpha":
        path = run_alpha_sweep(out_dir=args.out_dir)
        print(f"Alpha sweep → {path}")
    elif args.cmd == "batch":
        path = run_batch_sweep(out_dir=args.out_dir)
        print(f"Batch sweep → {path}")
    elif args.cmd == "stats":
        paired = compute_paired_statistics(args.csv)
        corr = compute_correlations(args.csv)
        out = args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "paired_statistics.json", "w") as f:
            json.dump(paired, f, indent=2)
        with open(out / "correlations.json", "w") as f:
            json.dump(corr, f, indent=2)
        print(f"Stats → {out}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
