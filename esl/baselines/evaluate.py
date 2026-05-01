"""Run offline baselines on a run directory; write baselines_summary.json."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl.baselines.dataset import (
    RunDataset,
    load_run_dataset,
    observed_rows,
    with_observation_prefix,
)
from esl.baselines.em_bernoulli_marginal import em_marginal_with_timing, logits_from_marginal_pi
from esl.baselines.em_conditional_bernoulli import (
    em_conditional_with_timing,
    marginalize_logits_from_conditional,
)
from esl.baselines.features_conditional import conditional_coop_features
from esl.baselines.fuzzy_cmeans import fuzzy_cmeans_with_timing
from esl.baselines.kmeans_actions import kmeans_with_timing
from esl.baselines.oracle import oracle_logits_from_true_types
from esl.metrics import match_prototypes_to_types, mce_value


def esl_mce_from_prototype_trajectory(
    ds: RunDataset,
    *,
    max_round_inclusive: int | None = None,
) -> float | None:
    """
    MCE from prototype_trajectory.csv: last row with ``round <= max_round_inclusive``,
    or final row if ``max_round_inclusive`` is None (end of run).
    """
    path = ds.run_dir / "prototype_trajectory.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    chosen = rows[-1]
    if max_round_inclusive is not None:
        cand = [r for r in rows if int(r["round"]) <= int(max_round_inclusive)]
        if not cand:
            return None
        chosen = cand[-1]
    k = ds.num_prototypes
    a = ds.num_actions
    logits = np.zeros((k, a), dtype=np.float64)
    try:
        for ki in range(k):
            for ai in range(a):
                logits[ki, ai] = float(chosen[f"theta_{ki}_{ai}"])
    except KeyError:
        return None
    return mce_value(ds.true_type_probs, logits)


def esl_mce_from_last_prototype_trajectory(ds: RunDataset) -> float | None:
    """End-of-run θ (same as ``esl_mce_from_prototype_trajectory(ds, max_round_inclusive=None)``)."""
    return esl_mce_from_prototype_trajectory(ds, max_round_inclusive=None)


def _empirical_q_ai_cooperate(ds: RunDataset) -> float:
    obs = observed_rows(ds)
    if not obs:
        return 0.5
    n_c = sum(1 for r in obs if r.a_i == 0)
    return n_c / len(obs)


def phi_matrix_to_logits(phi_rows: np.ndarray, ds: RunDataset, eps: float = 1e-12) -> np.ndarray:
    """Map (K,2) conditional-cooperate features to (K,2) logits under empirical P(a_i)."""
    q = _empirical_q_ai_cooperate(ds)
    k = phi_rows.shape[0]
    logits = np.zeros((k, 2), dtype=np.float64)
    for t in range(k):
        p_coop = q * phi_rows[t, 0] + (1.0 - q) * phi_rows[t, 1]
        p_coop = float(np.clip(p_coop, eps, 1.0 - eps))
        p_def = 1.0 - p_coop
        logits[t, 0] = np.log(p_coop)
        logits[t, 1] = np.log(p_def)
    return logits


StaticClusteringMethod = Literal["kmeans", "fcm"]


def fit_static_clustering_baseline(
    ds: RunDataset,
    *,
    method: StaticClusteringMethod,
    kmeans_seed: int = 0,
    fcm_seed: int = 0,
    fcm_m: float = 2.0,
    kmeans_n_init: int = 10,
) -> dict[str, Any]:
    """
    Offline static clustering only (K-means or FCM on conditional-cooperate features).

    PRD §8A: in-project reference; no sequential learning. Used for Milestone 2B smoke exports.
    """
    k = ds.num_prototypes
    true_probs = ds.true_type_probs
    phi, _counts = conditional_coop_features(ds)
    phi_hash = hashlib.sha256(np.ascontiguousarray(phi).tobytes()).hexdigest()[:16]

    if method == "kmeans":
        _lk, centers, inertia, t_fit = kmeans_with_timing(
            phi, k, seed=kmeans_seed, n_init=kmeans_n_init
        )
        objective = float(inertia)
    else:
        _lf, centers, obj, t_fit = fuzzy_cmeans_with_timing(
            phi, k, seed=fcm_seed, m=fcm_m
        )
        objective = float(obj)

    logits = phi_matrix_to_logits(centers, ds)
    mce = float(mce_value(true_probs, logits))
    _perm, total_ce = match_prototypes_to_types(true_probs, logits)
    total_ce_f = float(total_ce)

    return {
        "method": method,
        "final_mce": mce,
        "final_matched_cross_entropy": total_ce_f,
        "wall_time_sec": float(t_fit),
        "logits": logits,
        "centers": centers,
        "phi_feature_hash_sha256_prefix": phi_hash,
        "objective": objective,
        "n_observed_w_positive": len(observed_rows(ds)),
        "fcm_m": float(fcm_m) if method == "fcm" else None,
        "kmeans_n_init": int(kmeans_n_init) if method == "kmeans" else None,
    }


def run_baselines_on_dataset(
    ds: RunDataset,
    *,
    kmeans_seed: int = 0,
    fcm_seed: int = 0,
    fcm_m: float = 2.0,
    em_seed: int = 0,
    em_restarts: int = 8,
    prefix_max_round: int | None = None,
    clustering_only: bool = False,
) -> dict[str, Any]:
    """
    If ``prefix_max_round`` is set (Protocol P), baselines fit only on observations with
    ``round <= prefix_max_round``; ESL MCE uses θ at the end of that same round from
    ``prototype_trajectory.csv``. Otherwise (Protocol T), baselines use the full log and
    ESL uses final θ.

    When ``clustering_only`` is True (Milestone 3 paper pack: ESL + K-means + FCM only),
    EM and Oracle fits are skipped entirely.
    """
    full_ds = ds
    if prefix_max_round is not None:
        ds = with_observation_prefix(full_ds, prefix_max_round)

    k = ds.num_prototypes
    true_probs = ds.true_type_probs
    phi, _counts = conditional_coop_features(ds)
    phi_hash = hashlib.sha256(np.ascontiguousarray(phi).tobytes()).hexdigest()[:16]

    out: dict[str, Any] = {
        "run_dir": str(ds.run_dir),
        "n_agents": ds.num_agents,
        "num_prototypes": k,
        "n_observation_rows_total": len(ds.observations),
        "n_observed_w_positive": len(observed_rows(ds)),
        "phi_feature_hash_sha256_prefix": phi_hash,
        "comparison_protocol": (
            "prefix_matched_budget" if prefix_max_round is not None else "transductive_full_log"
        ),
        "prefix_max_round": prefix_max_round,
        "n_observation_rows_full_run": len(full_ds.observations),
    }

    _lk, centers_km, _in_km, t_km = kmeans_with_timing(
        phi, k, seed=kmeans_seed, n_init=10
    )
    logits_km = phi_matrix_to_logits(centers_km, ds)
    out["kmeans_mce"] = mce_value(true_probs, logits_km)
    out["wall_time_sec_kmeans"] = t_km
    out["kmeans_inertia"] = float(_in_km)

    _lf, centers_fcm, _obj_fcm, t_fcm = fuzzy_cmeans_with_timing(
        phi, k, seed=fcm_seed, m=fcm_m
    )
    logits_fcm = phi_matrix_to_logits(centers_fcm, ds)
    out["fcm_mce"] = mce_value(true_probs, logits_fcm)
    out["wall_time_sec_fcm"] = t_fcm
    out["fcm_m"] = fcm_m

    if not clustering_only:
        _rc, pi_c, pi_d, _gc, it_c, ll_c, t_em_c = em_conditional_with_timing(
            ds, n_restarts=em_restarts, base_seed=em_seed
        )
        logits_em_c = marginalize_logits_from_conditional(pi_c, pi_d, ds)
        out["em_conditional_mce"] = mce_value(true_probs, logits_em_c)
        out["wall_time_sec_em_conditional"] = t_em_c
        out["em_conditional_iterations"] = it_c
        out["em_conditional_log_likelihood"] = ll_c
        out["em_conditional_n_restarts"] = em_restarts

        _rm, pi_m, _gm, it_m, ll_m, t_em_m = em_marginal_with_timing(
            ds, n_restarts=em_restarts, base_seed=em_seed + 99
        )
        logits_em_m = logits_from_marginal_pi(pi_m)
        out["em_marginal_mce"] = mce_value(true_probs, logits_em_m)
        out["wall_time_sec_em_marginal"] = t_em_m
        out["em_marginal_iterations"] = it_m
        out["em_marginal_log_likelihood"] = ll_m

        logits_o = oracle_logits_from_true_types(ds)
        out["oracle_mce"] = mce_value(true_probs, logits_o)
        out["wall_time_sec_oracle"] = 0.0

    sm_path = full_ds.run_dir / "summary_metrics.json"
    if sm_path.is_file():
        sm = json.loads(sm_path.read_text(encoding="utf-8"))
        out["esl_final_mce_from_summary"] = sm.get("final_mce")
        out["esl_num_rounds_executed"] = sm.get("num_rounds_executed")
        out["esl_num_interaction_events_executed"] = sm.get("num_interaction_events_executed")

    if prefix_max_round is not None:
        rec = esl_mce_from_prototype_trajectory(
            full_ds, max_round_inclusive=prefix_max_round
        )
        if rec is not None:
            out["esl_mce_recomputed_from_trajectory"] = rec
        out["esl_mce_summary_matches_trajectory"] = None
        sm_mce = out.get("esl_final_mce_from_summary")
        if rec is not None and sm_mce is not None:
            out["esl_final_mce_full_run_reference"] = float(sm_mce)
    else:
        rec = esl_mce_from_last_prototype_trajectory(full_ds)
        if rec is not None:
            out["esl_mce_recomputed_from_trajectory"] = rec
        sm_mce = out.get("esl_final_mce_from_summary")
        if rec is not None and sm_mce is not None:
            out["esl_mce_summary_matches_trajectory"] = abs(float(sm_mce) - rec) < 1e-5
        elif rec is not None:
            out["esl_mce_summary_matches_trajectory"] = None
        else:
            out["esl_mce_summary_matches_trajectory"] = None

    return out


def write_baselines_summary(run_dir: Path | str, **kwargs: Any) -> Path:
    run_dir = Path(run_dir)
    prefix = kwargs.get("prefix_max_round")
    ds = load_run_dataset(run_dir)
    summary = run_baselines_on_dataset(ds, **kwargs)
    out_dir = run_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefix is not None:
        path = out_dir / f"baselines_summary_prefix_round_{int(prefix)}.json"
    else:
        path = out_dir / "baselines_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
