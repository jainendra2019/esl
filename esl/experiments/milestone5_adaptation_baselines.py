"""
Milestone 5 baselines: heterogeneous IPD adaptation without changing ESL core.

- **Clustering + logit BR**: same Bayes beliefs and interaction protocol as ``run_esl`` in
  adaptation mode, but prototype rows are refreshed from **batch clustering** (K-means or
  FCM on belief+signal features) instead of SGD.
- **Independent PPO focal**: no beliefs or prototypes; focal agent trains official PPO
  against fixed hidden opponents under the same pair schedule as ESL.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.baselines.fuzzy_cmeans import fuzzy_cmeans
from esl.baselines.kmeans_actions import kmeans_lloyd
from esl.baselines.ppo_adapter import _load_official_ppo_module, official_ppo_commit
from esl.config import ESLConfig
from esl.experiment_registry import SUMMARY_METRICS_SCHEMA_VERSION
from esl.interaction_protocol import sample_L_t, sample_ordered_pairs_without_replacement
from esl.metrics import (
    belief_argmax_accuracy,
    belief_entropy,
    match_prototypes_to_types,
    mce_value,
    pairwise_assignment_cost,
)
from esl.prototypes import softmax_log_likelihood_clamped, stable_softmax
from esl.signals import action_to_signal
from esl.trainer import (
    BatchRecord,
    RunLog,
    _append_prototype_update_event,
    _write_csv,
    _write_prototype_update_steps_csv,
    act_agent,
    convergence_criteria_met,
    init_prototype_logits,
    matched_true_type_separation_p_coop,
    observe_signal_update_belief,
    sample_observation_mask,
)


def _cluster_logits_from_batch(
    batch: list[BatchRecord],
    logits: np.ndarray,
    cfg: ESLConfig,
    *,
    backend: Literal["kmeans", "fcm"],
    rng: np.random.Generator,
    prototype_step_m: int,
) -> tuple[np.ndarray, float]:
    """Assign batch rows to clusters; set each cluster row logits from empirical signal rates."""
    k_proto, a_dim = cfg.num_prototypes, cfg.num_actions
    if not batch:
        return logits, 0.0
    X_rows: list[np.ndarray] = []
    for r in batch:
        b = np.asarray(r.b_ij[:k_proto], dtype=np.float64)
        oh = np.zeros(a_dim, dtype=np.float64)
        oh[int(r.signal)] = 1.0
        X_rows.append(np.concatenate([b, oh]))
    x = np.stack(X_rows, axis=0)
    n = x.shape[0]
    if n < k_proto:
        return logits, 0.0
    if backend == "kmeans":
        labels, _, _ = kmeans_lloyd(x, k_proto, seed=int(cfg.seed) + prototype_step_m)
    else:
        labels, _, _ = fuzzy_cmeans(x, k_proto, seed=int(cfg.seed) + prototype_step_m * 17)
    new_logits = logits.copy()
    for c in range(k_proto):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        cnt = np.zeros(a_dim, dtype=np.float64)
        for ix in idxs:
            rec = batch[int(ix)]
            if rec.w > 0:
                cnt[int(rec.signal)] += 1.0
        if cnt.sum() < 1e-12:
            continue
        p = cnt / cnt.sum()
        p = np.clip(p, 1e-3, 1.0 - 1e-3)
        new_logits[c] = np.log(p)
    delta = float(np.linalg.norm(new_logits - logits))
    return new_logits, delta


def _write_summary_and_csvs(
    run_dir: Path,
    cfg: ESLConfig,
    log: RunLog,
    logits: np.ndarray,
    belief_tensor: np.ndarray,
    true_types: np.ndarray,
    true_type_probs: np.ndarray,
    prototype_step_m: int,
    interaction_n: int,
    stopped_on_convergence: bool,
    convergence_round: int | None,
    extra_summary: dict[str, Any],
) -> dict[str, Any]:
    final_perm, final_ce = match_prototypes_to_types(true_type_probs, logits)
    cost_mat = pairwise_assignment_cost(true_type_probs, logits)
    rewards = log.reward_rows
    if rewards:
        cum_social = float(sum(float(r["r_i"]) + float(r["r_j"]) for r in rewards))
        mean_per_round = cum_social / (2.0 * len(rewards))
    else:
        cum_social = 0.0
        mean_per_round = 0.0
    learned_p = stable_softmax(logits)
    summary_out: dict[str, Any] = {
        "schema_version": SUMMARY_METRICS_SCHEMA_VERSION,
        "final_matched_cross_entropy": final_ce,
        "final_mce": mce_value(true_type_probs, logits),
        "permutation_true_to_learned": final_perm.tolist(),
        "cost_matrix": cost_mat.tolist(),
        "final_belief_entropy": belief_entropy(belief_tensor, cfg.num_agents, cfg.num_prototypes),
        "final_belief_argmax_accuracy": belief_argmax_accuracy(
            belief_tensor, true_types, final_perm, cfg.num_agents
        ),
        "final_prototype_gap": matched_true_type_separation_p_coop(true_type_probs, logits),
        "final_prototype_softmax": learned_p.tolist(),
        "cumulative_social_payoff": cum_social,
        "mean_payoff_per_agent_per_round": mean_per_round,
        "prototype_update_count": prototype_step_m,
        "prototype_update_every_q": cfg.prototype_Q(),
        "num_interaction_events_executed": int(interaction_n),
        "p_obs": float(cfg.p_obs),
        "prototype_lr_scale": float(cfg.prototype_lr_scale),
        "init_noise": float(cfg.init_noise),
        "num_rounds": cfg.num_rounds,
        "num_rounds_executed": len(log.summary_rows),
        "stopped_on_convergence": stopped_on_convergence,
        "convergence_round": convergence_round,
        "convergence_thresholds": None,
        "learning_frozen": cfg.learning_frozen,
        "freeze_prototype_parameters": bool(cfg.freeze_prototype_parameters),
        "seed": cfg.seed,
        "mode": cfg.mode,
    }
    summary_out.update(extra_summary)
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "run_config_snapshot.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )
    _write_csv(run_dir / "prototype_trajectory.csv", log.prototype_rows)
    _write_csv(run_dir / "belief_trajectory.csv", log.belief_rows)
    _write_csv(run_dir / "reward_trajectory.csv", log.reward_rows)
    _write_csv(run_dir / "metrics_trajectory.csv", log.summary_rows)
    _write_prototype_update_steps_csv(run_dir / "prototype_update_steps.csv", log.prototype_update_events)
    return summary_out


def run_clustering_br_adaptation(
    cfg: ESLConfig,
    run_dir: Path,
    *,
    backend: Literal["kmeans", "fcm"],
) -> dict[str, Any]:
    """
    Clustering + Bayes beliefs + logit best response for ESL-masked agents (same as ``run_esl``),
    with prototype rows updated by batch clustering instead of SGD.
    """
    if cfg.mode != "adaptation":
        raise ValueError("run_clustering_br_adaptation requires mode='adaptation'")
    if cfg.learning_frozen or cfg.freeze_prototype_parameters:
        raise ValueError("Milestone 5 clustering baseline expects normal learning flags off")
    cfg.validate()
    rng = cfg.make_rng()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(run_dir / "config.json")

    pay = games.game_payoffs(cfg)
    true_type_probs = games.true_type_distributions(cfg.num_prototypes)
    if cfg.force_agent_true_types is not None:
        true_types = np.array(cfg.force_agent_true_types, dtype=int)
    else:
        true_types = np.arange(cfg.num_agents, dtype=int) % cfg.num_prototypes
    _nb = len(games.HIDDEN_POLICY_BUILDERS)
    if cfg.force_hidden_policy_by_agent is not None:
        hidden_policies = [
            games.build_hidden_policy(int(cfg.force_hidden_policy_by_agent[a]) % _nb)
            for a in range(cfg.num_agents)
        ]
    else:
        hidden_policies = [
            games.build_hidden_policy(int(true_types[a]) % _nb) for a in range(cfg.num_agents)
        ]

    esl_mask = np.zeros(cfg.num_agents, dtype=bool)
    if cfg.adaptation_esl_agent_indices is not None:
        for idx in cfg.adaptation_esl_agent_indices:
            esl_mask[int(idx)] = True
    else:
        esl_mask[:] = True

    logits = init_prototype_logits(cfg, rng)
    belief_tensor = belief_ops.init_beliefs(cfg.num_agents, cfg.num_prototypes)
    log = RunLog()
    batch: list[BatchRecord] = []
    prototype_step_m = 0
    last_grad_norm = 0.0
    last_opp: dict[int, int | None] = {i: None for i in range(cfg.num_agents)}
    Q = cfg.prototype_Q()
    interaction_n = 0
    t = 0
    stopped_on_convergence = False
    convergence_round: int | None = None

    while t < cfg.num_rounds:
        belief_before_round = belief_tensor.copy()
        proto_norm_this_round = 0.0
        batch_ll_round = 0.0

        if cfg.force_ordered_pair is not None:
            e_t = [cfg.force_ordered_pair]
        else:
            l_t = sample_L_t(
                rng,
                cfg.interaction_pairs_min,
                cfg.interaction_pairs_max,
                law=cfg.interaction_pairs_law,
            )
            e_t = sample_ordered_pairs_without_replacement(rng, cfg.num_agents, l_t)

        for (i, j) in e_t:
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

            w = sample_observation_mask(cfg, rng)
            s = action_to_signal(a_j)
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
            interaction_n += 1
            if interaction_n % Q == 0 and batch:
                theta_before = logits.copy()
                m_step = prototype_step_m
                logits, proto_norm_this_round = _cluster_logits_from_batch(
                    batch, logits, cfg, backend=backend, rng=rng, prototype_step_m=m_step
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
                    interaction_n_at_update=interaction_n,
                    batch=batch,
                )
                last_grad_norm = proto_norm_this_round
                prototype_step_m += 1
                batch.clear()
            batch_ll_round = batch_ll

        belief_change_norm = float(np.sum(np.abs(belief_tensor - belief_before_round)))
        alpha_eff = cfg.belief_lr(t)
        perm, total_ce = match_prototypes_to_types(true_type_probs, logits)
        summary = {
            "round": t,
            "belief_entropy_mean": belief_entropy(belief_tensor, cfg.num_agents, cfg.num_prototypes),
            "matched_cross_entropy": total_ce,
            "belief_argmax_accuracy": belief_argmax_accuracy(
                belief_tensor, true_types, perm, cfg.num_agents
            ),
            "batch_log_likelihood": batch_ll_round,
            "alpha_logged": alpha_eff,
            "prototype_step_m": prototype_step_m,
            "prototype_update_norm": proto_norm_this_round,
            "belief_change_norm": belief_change_norm,
        }
        log.summary_rows.append(summary)

        row: dict[str, Any] = {"round": t, "prototype_step_m": prototype_step_m}
        for k in range(cfg.num_prototypes):
            for a in range(cfg.num_actions):
                row[f"theta_{k}_{a}"] = float(logits[k, a])
        for k in range(cfg.num_prototypes):
            sm = stable_softmax(logits[k : k + 1])[0]
            for a in range(cfg.num_actions):
                row[f"softmax_{k}_{a}"] = float(sm[a])
        log.prototype_rows.append(row)

        if cfg.log_beliefs_tensor and (not cfg.log_beliefs_every_interaction):
            for ii in range(cfg.num_agents):
                for jj in range(cfg.num_agents):
                    if ii == jj:
                        continue
                    br = {"round": t, "i": ii, "j": jj}
                    for k in range(cfg.num_prototypes):
                        br[f"b_{k}"] = float(belief_tensor[ii, jj, k])
                    log.belief_rows.append(br)

        if cfg.stop_on_convergence and (t + 1) >= cfg.convergence_window_w:
            if convergence_criteria_met(log.summary_rows, t, cfg, logits, true_type_probs):
                stopped_on_convergence = True
                convergence_round = t
                break
        t += 1

    last_env_round = len(log.summary_rows) - 1 if log.summary_rows else -1
    if batch:
        theta_before = logits.copy()
        m_step = prototype_step_m
        logits, last_grad_norm = _cluster_logits_from_batch(
            batch, logits, cfg, backend=backend, rng=rng, prototype_step_m=m_step
        )
        _append_prototype_update_event(
            log,
            cfg=cfg,
            update_index_m=m_step,
            env_round_ended=max(0, last_env_round),
            theta_before=theta_before,
            theta_after=logits,
            prototype_update_norm=last_grad_norm,
            final_flush=True,
            interaction_n_at_update=interaction_n,
            batch=batch,
        )
        prototype_step_m += 1
        batch.clear()

    extra = {
        "milestone5_method": f"clustering_br_{backend}",
        "milestone5_clustering_backend": backend,
    }
    return _write_summary_and_csvs(
        run_dir,
        cfg,
        log,
        logits,
        belief_tensor,
        true_types,
        true_type_probs,
        prototype_step_m,
        interaction_n,
        stopped_on_convergence,
        convergence_round,
        extra,
    )


def run_ppo_focal_heterogeneous_ipd(
    cfg: ESLConfig,
    run_dir: Path,
    *,
    focal_idx: int = 0,
    update_timestep: int = 32,
    K_epochs: int = 4,
) -> dict[str, Any]:
    """
    Independent PPO (official submodule) for a single focal agent; others use hidden policies.
    No beliefs or prototypes. Writes the same ``reward_trajectory.csv`` schema for metrics tooling.
    """
    if cfg.mode != "adaptation":
        raise ValueError("PPO baseline expects adaptation-style population config (mode may be ignored)")
    cfg.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(run_dir / "config.json")

    import torch

    ppo_mod = _load_official_ppo_module()
    PPO = ppo_mod.PPO

    rng = cfg.make_rng()
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    pay = games.game_payoffs(cfg)
    if cfg.force_agent_true_types is not None:
        true_types = np.array(cfg.force_agent_true_types, dtype=int)
    else:
        true_types = np.arange(cfg.num_agents, dtype=int) % cfg.num_prototypes
    _nb = len(games.HIDDEN_POLICY_BUILDERS)
    if cfg.force_hidden_policy_by_agent is not None:
        hidden_policies = [
            games.build_hidden_policy(int(cfg.force_hidden_policy_by_agent[a]) % _nb)
            for a in range(cfg.num_agents)
        ]
    else:
        hidden_policies = [
            games.build_hidden_policy(int(true_types[a]) % _nb) for a in range(cfg.num_agents)
        ]

    state_dim = max(1, cfg.num_agents - 1)
    action_dim = 2
    ppo_agent = PPO(
        state_dim,
        action_dim,
        3e-4,
        1e-3,
        0.99,
        K_epochs,
        0.2,
        False,
        action_std_init=0.6,
    )

    last_action_by: dict[int, int | None] = {i: None for i in range(cfg.num_agents)}
    last_opp: dict[int, int | None] = {i: None for i in range(cfg.num_agents)}
    log_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    time_step = 0

    def state_for_focal(f: int) -> np.ndarray:
        vec: list[float] = []
        for x in range(cfg.num_agents):
            if x == f:
                continue
            v = last_action_by.get(x)
            vec.append(0.5 if v is None else float(v))
        while len(vec) < state_dim:
            vec.append(0.5)
        return np.asarray(vec[:state_dim], dtype=np.float32)

    for t in range(cfg.num_rounds):
        if cfg.force_ordered_pair is not None:
            e_t = [cfg.force_ordered_pair]
        else:
            l_t = sample_L_t(
                rng,
                cfg.interaction_pairs_min,
                cfg.interaction_pairs_max,
                law=cfg.interaction_pairs_law,
            )
            e_t = sample_ordered_pairs_without_replacement(rng, cfg.num_agents, l_t)

        round_pay_focal = 0.0
        n_focal = 0
        for (i, j) in e_t:
            if i == focal_idx:
                st = state_for_focal(i)
                a_i = int(ppo_agent.select_action(st))
                a_j = hidden_policies[j].act(rng, last_opponent_action=last_opp[j])
            elif j == focal_idx:
                st = state_for_focal(j)
                a_j = int(ppo_agent.select_action(st))
                a_i = hidden_policies[i].act(rng, last_opponent_action=last_opp[i])
            else:
                a_i = hidden_policies[i].act(rng, last_opponent_action=last_opp[i])
                a_j = hidden_policies[j].act(rng, last_opponent_action=last_opp[j])

            last_opp[i] = a_j
            last_opp[j] = a_i
            last_action_by[i] = a_i
            last_action_by[j] = a_j

            r_i, r_j = games.play_pair_payoffs(a_i, a_j, pay)
            reward_rows.append({"round": t, "i": i, "j": j, "r_i": r_i, "r_j": r_j})
            if i == focal_idx:
                round_pay_focal += r_i
                n_focal += 1
                rew = float(r_i)
            elif j == focal_idx:
                round_pay_focal += r_j
                n_focal += 1
                rew = float(r_j)
            else:
                rew = 0.0

            if i == focal_idx or j == focal_idx:
                time_step += 1
                ppo_agent.buffer.rewards.append(rew)
                ppo_agent.buffer.is_terminals.append(False)
                if time_step > 0 and time_step % update_timestep == 0:
                    ppo_agent.update()
                log_rows.append(
                    {
                        "round": t,
                        "timestep": time_step,
                        "ppo_step_reward": rew,
                        "row_action": a_i if i == focal_idx else a_j,
                    }
                )

        summary_rows.append(
            {
                "round": t,
                "belief_entropy_mean": float("nan"),
                "matched_cross_entropy": float("nan"),
                "belief_argmax_accuracy": float("nan"),
                "batch_log_likelihood": float("nan"),
                "alpha_logged": float("nan"),
                "prototype_step_m": 0,
                "prototype_update_norm": 0.0,
                "belief_change_norm": float("nan"),
                "focal_mean_payoff_in_round": float(round_pay_focal / max(n_focal, 1)),
            }
        )

    cum_social = float(sum(float(r["r_i"]) + float(r["r_j"]) for r in reward_rows))
    mean_per_round = cum_social / max(2.0 * len(reward_rows), 1.0)
    commit = official_ppo_commit() or "unknown"
    summary_out: dict[str, Any] = {
        "schema_version": SUMMARY_METRICS_SCHEMA_VERSION,
        "final_matched_cross_entropy": None,
        "final_mce": None,
        "permutation_true_to_learned": [],
        "cost_matrix": [],
        "final_belief_entropy": None,
        "final_belief_argmax_accuracy": None,
        "final_prototype_gap": None,
        "final_prototype_softmax": [],
        "cumulative_social_payoff": cum_social,
        "mean_payoff_per_agent_per_round": mean_per_round,
        "prototype_update_count": 0,
        "prototype_update_every_q": cfg.prototype_Q(),
        "num_interaction_events_executed": len(reward_rows),
        "p_obs": float(cfg.p_obs),
        "prototype_lr_scale": float(cfg.prototype_lr_scale),
        "init_noise": float(cfg.init_noise),
        "num_rounds": cfg.num_rounds,
        "num_rounds_executed": len(summary_rows),
        "stopped_on_convergence": False,
        "convergence_round": None,
        "convergence_thresholds": None,
        "learning_frozen": False,
        "freeze_prototype_parameters": False,
        "seed": cfg.seed,
        "mode": "adaptation_ppo_baseline",
        "milestone5_method": "independent_ppo_focal",
        "mce_note": "MCE undefined: PPO baseline has no ESL prototypes.",
        "official_repo_commit": commit,
    }
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(summary_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(run_dir / "reward_trajectory.csv", reward_rows)
    _write_csv(run_dir / "metrics_trajectory.csv", summary_rows)
    if log_rows:
        keys = list(log_rows[0].keys())
        lines = [",".join(keys)]
        for r in log_rows:
            lines.append(",".join(str(r[k]) for k in keys))
        (run_dir / "metrics_trajectory_ppo_steps.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    prov = {
        "sources": [
            {
                "name": "PPO-PyTorch",
                "commit": commit,
                "entry_module": "PPO.py",
            }
        ],
        "notes": "Milestone 5 heterogeneous IPD; focal PPO only.",
    }
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    body = (run_dir / "reward_trajectory.csv").read_bytes()
    (run_dir / "reward_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "file": "reward_trajectory.csv",
                "sha256": hashlib.sha256(body).hexdigest(),
                "n_rows": len(reward_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_out
