"""Training loop: recovery and adaptation modes, batching, logging hooks.

Implementation-level pseudocode (faithful to this file) lives in **ALGORITHM.md** at repo root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.config import ESLConfig
from esl.metrics import (
    belief_argmax_accuracy,
    belief_entropy,
    match_prototypes_to_types,
    pairwise_assignment_cost,
)
from esl.prototypes import (
    batch_weighted_prototype_gradient,
    grad_log_likelihood,
    likelihoods,
    softmax_log_likelihood_clamped,
    stable_softmax,
)
from esl.signals import action_to_signal


@dataclass
class BatchRecord:
    i: int
    j: int
    signal: int
    w: float
    b_ij: np.ndarray


def prototype_sgd_step_from_batch(
    batch: list[BatchRecord],
    logits: np.ndarray,
    cfg: ESLConfig,
    prototype_step_m: int,
) -> tuple[np.ndarray, float]:
    """
    §4.11: For each record with w > 0, accumulate per-prototype gradients
    g_k = w · b_snap[k] · (e_s − softmax(θ_k)); then **mean over full |batch|**
    (records with w ≤ 0 contribute 0 to the sum but still increase the divisor).
    Finally θ ← θ + γ_m · g_mean.
    """
    g_accum = np.zeros_like(logits)
    for rec in batch:
        if rec.w <= 0:
            continue
        wk = rec.b_ij * rec.w
        g_accum += batch_weighted_prototype_gradient(logits, wk, rec.signal)
    denom = max(len(batch), 1)
    g_mean = g_accum / denom
    gamma = cfg.prototype_lr(prototype_step_m)
    delta_theta = gamma * g_mean
    return logits + delta_theta, float(np.linalg.norm(delta_theta))


def observe_signal_update_belief(
    belief_tensor: np.ndarray,
    logits: np.ndarray,
    *,
    i: int,
    j: int,
    signal: int,
    w: float,
    cfg: ESLConfig,
) -> BatchRecord:
    """
    PRD §7 ordering: snapshot b_{i→j,t} → (if w>0) Bayes update to B_{t+1} → return BatchRecord.

    Always returns a record with pre-update ``b_ij`` in the batch row. When ``w == 0``,
    beliefs are unchanged and the record still carries ``w=0`` so the caller can append it;
    ``prototype_sgd_step_from_batch`` skips gradient from such rows but they still count
    toward the batch-length denominator for the mean gradient.
    """
    b_ij_t = belief_tensor[i, j].copy()
    if w > 0:
        # Unclipped softmax likelihoods for Bayes (PRD §4.6); never clamp L_k for the update.
        lk = likelihoods(logits, signal)
        belief_tensor[i, j] = belief_ops.update_belief_pair(
            belief_tensor[i, j],
            lk,
            cfg.delta_simplex,
            cfg.belief_floor_eps,
            floor_tolerance=cfg.belief_floor_tolerance,
            max_floor_iter=cfg.belief_floor_max_iter,
        )
    return BatchRecord(i=i, j=j, signal=signal, w=w, b_ij=b_ij_t)


@dataclass
class RunLog:
    prototype_rows: list[dict[str, Any]] = field(default_factory=list)
    belief_rows: list[dict[str, Any]] = field(default_factory=list)
    reward_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    # One entry per prototype SGD step (scheduled or final flush).
    prototype_update_events: list[dict[str, Any]] = field(default_factory=list)


def _append_prototype_update_event(
    log: RunLog,
    *,
    cfg: ESLConfig,
    update_index_m: int,
    env_round_ended: int,
    theta_before: np.ndarray,
    theta_after: np.ndarray,
    prototype_update_norm: float,
    final_flush: bool,
) -> None:
    p_before = stable_softmax(np.asarray(theta_before, dtype=np.float64))
    p_after = stable_softmax(np.asarray(theta_after, dtype=np.float64))
    log.prototype_update_events.append(
        {
            "prototype_update_every": cfg.prototype_update_every,
            "prototype_update_index_m": int(update_index_m),
            "env_round_ended": int(env_round_ended),
            "final_flush": bool(final_flush),
            "prototype_update_norm": float(prototype_update_norm),
            "theta_before": theta_before.tolist(),
            "theta_after": theta_after.tolist(),
            "p_before": p_before.tolist(),
            "p_after": p_after.tolist(),
        }
    )


def init_prototype_logits(cfg: ESLConfig, rng: np.random.Generator) -> np.ndarray:
    noise = 0.0 if cfg.symmetric_init else cfg.init_noise
    shape = (cfg.num_prototypes, cfg.num_actions)
    return cfg.base_init + noise * rng.standard_normal(size=shape)


def sample_observation_mask(cfg: ESLConfig, rng: np.random.Generator) -> float:
    if cfg.observability == "full":
        return 1.0
    return float(rng.binomial(1, cfg.p_obs))


def marginal_opponent_probs(
    observer: int,
    opponent: int,
    belief_tensor: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    bvec = belief_tensor[observer, opponent]
    p = stable_softmax(logits)
    return bvec @ p


def sample_logit_best_response(
    rng: np.random.Generator,
    utilities: np.ndarray,
    lam: float,
) -> int:
    u = lam * np.asarray(utilities, dtype=np.float64)
    u = u - np.max(u)
    w = np.exp(u)
    w /= w.sum()
    return int(rng.choice(len(w), p=w))


def act_agent(
    agent_id: int,
    opponent_id: int,
    *,
    cfg: ESLConfig,
    rng: np.random.Generator,
    is_row_player: bool,
    hidden_policies: list[games.HiddenPolicy],
    esl_mask: np.ndarray,
    belief_tensor: np.ndarray,
    logits: np.ndarray,
    pay: games.PayoffMatrices,
    last_opp: dict[int, int | None],
) -> int:
    if not esl_mask[agent_id]:
        return hidden_policies[agent_id].act(rng, last_opponent_action=last_opp.get(agent_id))

    if is_row_player:
        p_opp = marginal_opponent_probs(agent_id, opponent_id, belief_tensor, logits)
        util = np.array([np.sum(p_opp * pay.row[a, :]) for a in range(cfg.num_actions)], dtype=np.float64)
    else:
        p_opp = marginal_opponent_probs(agent_id, opponent_id, belief_tensor, logits)
        util = np.array([np.sum(p_opp * pay.col[:, a]) for a in range(cfg.num_actions)], dtype=np.float64)
    return sample_logit_best_response(rng, util, cfg.adaptation_lambda)


def run_esl(
    cfg: ESLConfig,
    run_dir: Path | None = None,
) -> tuple[RunLog, np.ndarray, np.ndarray, dict[str, Any], Path]:
    """
    Main ESL loop. **Recovery:** actions from fixed ``hidden_policies`` only (never ``act_agent``).
    **Adaptation:** actions from ``act_agent`` (ESL softmax best response vs beliefs + θ).

    One random ordered pair (i,j) per round unless ``force_ordered_pair`` is set; signal s = a_j.
    Belief / batch / prototype ordering: see PRD §7 and **ALGORITHM.md**.

    If ``learning_frozen``: no Bayes updates and no batch appends (beliefs stay at init).
    """
    cfg.validate()
    rng = cfg.make_rng()
    run_dir = run_dir or Path("runs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(run_dir / "config.json")

    pay = games.prisoners_dilemma(cfg)
    true_type_probs = games.true_type_distributions(cfg.num_prototypes)

    # Assign each agent a true latent type index in 0..K-1 (cyclic if N > K)
    if cfg.force_agent_true_types is not None:
        true_types = np.array(cfg.force_agent_true_types, dtype=int)
    else:
        true_types = np.arange(cfg.num_agents, dtype=int) % cfg.num_prototypes
    _nb = len(games.HIDDEN_POLICY_BUILDERS)
    hidden_policies = [
        games.build_hidden_policy(int(true_types[a]) % _nb) for a in range(cfg.num_agents)
    ]

    esl_mask = np.zeros(cfg.num_agents, dtype=bool)
    if cfg.mode == "adaptation":
        esl_mask[:] = True
    # Recovery: fixed hidden policies for everyone (no ESL action model)

    if cfg.prototype_logits_override is not None:
        logits = np.array(cfg.prototype_logits_override, dtype=np.float64)
    else:
        logits = init_prototype_logits(cfg, rng)
    belief_tensor = belief_ops.init_beliefs(cfg.num_agents, cfg.num_prototypes)

    log = RunLog()
    batch: list[BatchRecord] = []
    prototype_step_m = 0
    last_grad_norm = 0.0
    last_opp: dict[int, int | None] = {i: None for i in range(cfg.num_agents)}

    for t in range(cfg.num_rounds):
        belief_before = belief_tensor.copy()
        if cfg.force_ordered_pair is not None:
            i, j = cfg.force_ordered_pair
        else:
            pairs = [(i, j) for i in range(cfg.num_agents) for j in range(cfg.num_agents) if i != j]
            idx = int(rng.integers(0, len(pairs)))
            i, j = pairs[idx]

        if cfg.mode == "recovery":
            a_j = hidden_policies[j].act(rng, last_opponent_action=last_opp[j])
            a_i = hidden_policies[i].act(rng, last_opponent_action=last_opp[i])
        else:
            a_i = act_agent(
                i,
                j,
                cfg=cfg,
                rng=rng,
                is_row_player=True,
                hidden_policies=hidden_policies,
                esl_mask=esl_mask,
                belief_tensor=belief_tensor,
                logits=logits,
                pay=pay,
                last_opp=last_opp,
            )
            a_j = act_agent(
                j,
                i,
                cfg=cfg,
                rng=rng,
                is_row_player=False,
                hidden_policies=hidden_policies,
                esl_mask=esl_mask,
                belief_tensor=belief_tensor,
                logits=logits,
                pay=pay,
                last_opp=last_opp,
            )

        last_opp[i] = a_j
        last_opp[j] = a_i

        r_i, r_j = games.play_pair_payoffs(a_i, a_j, pay)
        log.reward_rows.append({"round": t, "i": i, "j": j, "r_i": r_i, "r_j": r_j})

        # --- Two-timescale contract (PRD §7): observe -> beliefs -> batch; prototypes later ---
        w = sample_observation_mask(cfg, rng)
        s = action_to_signal(a_j)
        if cfg.learning_frozen:
            b_frozen = belief_tensor[i, j].copy()
            batch_ll = (
                float(
                    np.sum(
                        b_frozen
                        * w
                        * softmax_log_likelihood_clamped(logits, s, cfg.log_prob_min)
                    )
                )
                if w > 0
                else 0.0
            )
        else:
            rec = observe_signal_update_belief(
                belief_tensor, logits, i=i, j=j, signal=s, w=w, cfg=cfg
            )
            batch.append(rec)
            batch_ll = (
                float(
                    np.sum(
                        rec.b_ij
                        * w
                        * softmax_log_likelihood_clamped(logits, s, cfg.log_prob_min)
                    )
                )
                if w > 0
                else 0.0
            )

        belief_change_norm = float(np.sum(np.abs(belief_tensor - belief_before)))

        alpha_eff = cfg.belief_lr(t)
        proto_norm_this_round = 0.0
        if not cfg.learning_frozen and (t + 1) % cfg.prototype_update_every == 0 and batch:
            if cfg.freeze_prototype_parameters:
                batch.clear()
            else:
                theta_before = logits.copy()
                m_step = prototype_step_m
                logits, proto_norm_this_round = prototype_sgd_step_from_batch(
                    batch, logits, cfg, m_step
                )
                _append_prototype_update_event(
                    log,
                    cfg=cfg,
                    update_index_m=m_step,
                    env_round_ended=t,
                    theta_before=theta_before,
                    theta_after=logits,
                    prototype_update_norm=proto_norm_this_round,
                    final_flush=False,
                )
                last_grad_norm = proto_norm_this_round
                prototype_step_m += 1
                batch.clear()

        perm, total_ce = match_prototypes_to_types(true_type_probs, logits)
        summary = {
            "round": t,
            "belief_entropy_mean": belief_entropy(belief_tensor, cfg.num_agents, cfg.num_prototypes),
            "matched_cross_entropy": total_ce,
            "belief_argmax_accuracy": belief_argmax_accuracy(
                belief_tensor, true_types, perm, cfg.num_agents
            ),
            "batch_log_likelihood": batch_ll,
            "alpha_logged": alpha_eff,
            "prototype_step_m": prototype_step_m,
            "prototype_update_norm": proto_norm_this_round,
            "belief_change_norm": belief_change_norm,
        }
        log.summary_rows.append(summary)

        # Prototype trajectory row
        row: dict[str, Any] = {"round": t, "prototype_step_m": prototype_step_m}
        for k in range(cfg.num_prototypes):
            for a in range(cfg.num_actions):
                row[f"theta_{k}_{a}"] = float(logits[k, a])
        for k in range(cfg.num_prototypes):
            sm = stable_softmax(logits[k : k + 1])[0]
            for a in range(cfg.num_actions):
                row[f"softmax_{k}_{a}"] = float(sm[a])
        log.prototype_rows.append(row)

        for ii in range(cfg.num_agents):
            for jj in range(cfg.num_agents):
                if ii == jj:
                    continue
                br = {"round": t, "i": ii, "j": jj}
                for k in range(cfg.num_prototypes):
                    br[f"b_{k}"] = float(belief_tensor[ii, jj, k])
                log.belief_rows.append(br)

    if batch and not cfg.learning_frozen:
        if cfg.freeze_prototype_parameters:
            batch.clear()
        else:
            theta_before = logits.copy()
            m_step = prototype_step_m
            logits, last_grad_norm = prototype_sgd_step_from_batch(
                batch, logits, cfg, m_step
            )
            _append_prototype_update_event(
                log,
                cfg=cfg,
                update_index_m=m_step,
                env_round_ended=cfg.num_rounds - 1,
                theta_before=theta_before,
                theta_after=logits,
                prototype_update_norm=last_grad_norm,
                final_flush=True,
            )
            prototype_step_m += 1
            batch.clear()

    final_perm, final_ce = match_prototypes_to_types(true_type_probs, logits)
    cost_mat = pairwise_assignment_cost(true_type_probs, logits)
    rewards = log.reward_rows
    if rewards:
        cum_social = float(sum(float(r["r_i"]) + float(r["r_j"]) for r in rewards))
        mean_per_round = cum_social / (2.0 * len(rewards))
    else:
        cum_social = 0.0
        mean_per_round = 0.0
    summary_out: dict[str, Any] = {
        "final_matched_cross_entropy": final_ce,
        "permutation_true_to_learned": final_perm.tolist(),
        "cost_matrix": cost_mat.tolist(),
        "final_belief_entropy": belief_entropy(belief_tensor, cfg.num_agents, cfg.num_prototypes),
        "final_belief_argmax_accuracy": belief_argmax_accuracy(
            belief_tensor, true_types, final_perm, cfg.num_agents
        ),
        "cumulative_social_payoff": cum_social,
        "mean_payoff_per_agent_per_round": mean_per_round,
        "prototype_update_count": prototype_step_m,
        "num_rounds": cfg.num_rounds,
        "learning_frozen": cfg.learning_frozen,
        "seed": cfg.seed,
        "mode": cfg.mode,
    }
    (run_dir / "summary_metrics.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    _write_csv(run_dir / "prototype_trajectory.csv", log.prototype_rows)
    _write_csv(run_dir / "belief_trajectory.csv", log.belief_rows)
    _write_csv(run_dir / "reward_trajectory.csv", log.reward_rows)
    _write_csv(run_dir / "metrics_trajectory.csv", log.summary_rows)

    return log, logits, belief_tensor, summary_out, run_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in keys))
    path.write_text("\n".join(lines), encoding="utf-8")


# Expose gradient for tests
def log_likelihood_grad_reference(logits: np.ndarray, action: int) -> np.ndarray:
    return grad_log_likelihood(logits, action)
