"""Prototype recovery sweep: K=2, 3, 5 stochastic prototypes.

Supports IPD, Stag Hunt, and Matching Pennies with game-specific default
prototypes. Results stored under runs/ with per-run CSVs/JSONs and
comparison PNGs at the experiment root.

Usage:
    # IPD (default)
    python -m esl.experiments.prototype_recovery_sweep \\
        --seeds 5 --rounds 2000 --out runs/ipd_sweep

    # Stag Hunt
    python -m esl.experiments.prototype_recovery_sweep \\
        --game stag_hunt --seeds 3 --rounds 2000 --out runs/sh_sweep

    # Matching Pennies with sparse observability
    python -m esl.experiments.prototype_recovery_sweep \\
        --game matching_pennies --p-obs 0.5 --seeds 3 --rounds 3000 --out runs/mp_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from esl.config import ESLConfig
from esl.metrics import (
    belief_argmax_accuracy,
    match_prototypes_to_types,
    mean_belief_ce_vs_types,
)
from esl.prototypes import stable_softmax
from esl.trainer import run_esl

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  EXPERIMENT CONFIGURATION — edit here to change hyperparameters         │
# └──────────────────────────────────────────────────────────────────────────┘

# ── Per-game ground-truth prototype definitions ───────────────────────────
# Each list is a K×2 matrix: [P(action_0), P(action_1)] per prototype.
#   IPD:              action_0 = Cooperate,  action_1 = Defect
#   Stag Hunt:        action_0 = Stag,       action_1 = Hare
#   Matching Pennies: action_0 = Heads,      action_1 = Tails

GAME_PROTOTYPES: dict[str, dict[str, list[list[float]]]] = {
    "ipd": {
        "K2": [[0.98, 0.02], [0.02, 0.98]],                                          # cooperator vs defector
        "K3": [[0.98, 0.02], [0.75, 0.25], [0.02, 0.98]],                             # + mixed cooperator
        "K5": [[0.98, 0.02], [0.75, 0.25], [0.60, 0.40], [0.50, 0.50], [0.25, 0.75]], # full spectrum
    },
    "stag_hunt": {
        "K2": [[0.95, 0.05], [0.05, 0.95]],                                           # coordinator vs safety
        "K3": [[0.95, 0.05], [0.50, 0.50], [0.05, 0.95]],                              # + indecisive mixer
        "K4": [[0.95, 0.05], [0.75, 0.25], [0.25, 0.75], [0.05, 0.95]], # full spectrum
    },
    "matching_pennies": {
        # "K2": [[0.85, 0.15], [0.15, 0.85]],                                            # heads-biased vs tails-biased
        # "K3": [[0.85, 0.15], [0.50, 0.50], [0.15, 0.85]],                              # + Nash player
        "K5": [[0.85, 0.15], [0.65, 0.35], [0.50, 0.50], [0.35, 0.65], [0.15, 0.85]], # gradient of biases
    },
}

# ── Fixed hyperparameters ─────────────────────────────────────────────────
# Change these constants OR pass CLI flags to override.

DEFAULT_NUM_AGENTS = 20              # N: population size
DEFAULT_PROTOTYPE_UPDATE_EVERY = 7   # M: rounds between prototype SGD steps
DEFAULT_INTERACTION_PAIRS_MIN = 5    # L_t lower bound
DEFAULT_INTERACTION_PAIRS_MAX = 15   # L_t upper bound
DEFAULT_POPULATION_NOISE_SIGMA = 0.1 # σ: per-agent Gaussian noise on logits
DEFAULT_PROTOTYPE_LR_SCALE = 22.0    # γ: prototype learning rate scale
DEFAULT_LR_EXPONENT = -0.9           # prototype LR decay exponent
DEFAULT_DELTA_SIMPLEX = 0.02         # belief simplex floor
DEFAULT_INIT_NOISE = 0.05            # initial prototype logit perturbation
DEFAULT_P_OBS = 1.0                  # observation probability (1.0 = full observability)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  IMPLEMENTATION — you normally don't need to edit below this line       │
# └──────────────────────────────────────────────────────────────────────────┘


def make_balanced_types(num_agents: int, K: int) -> list[int]:
    """Assign agents to types as evenly as possible."""
    types: list[int] = []
    for k in range(K):
        types.extend([k] * (num_agents // K))
    for r in range(num_agents - len(types)):
        types.append(r % K)
    types.sort()
    return types


def make_cfg(
    protos: list[list[float]],
    seed: int,
    num_rounds: int,
    num_agents: int,
    prototype_update_every: int,
    noise_sigma: float,
    lr_scale: float,
    game_type: str = "ipd",
    p_obs: float = 1.0,
) -> ESLConfig:
    """Build an ESLConfig for a recovery experiment with custom prototypes."""
    K = len(protos)
    types = make_balanced_types(num_agents, K)

    cfg = ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=num_agents,
        num_prototypes=K,
        num_actions=2,
        num_rounds=num_rounds,
        force_agent_true_types=types,
        ground_truth_probs=protos,
        population_noise_sigma=noise_sigma,
        # Game
        game_type=game_type,
        # Prototype learning
        delta_simplex=DEFAULT_DELTA_SIMPLEX,
        base_init=0.0,
        init_noise=DEFAULT_INIT_NOISE,
        symmetric_init=False,
        prototype_lr_scale=lr_scale,
        lr_prototype_gamma_exponent=DEFAULT_LR_EXPONENT,
        prototype_update_every=prototype_update_every,
        prototype_l2_eta=0.0,
        # Interaction protocol
        interaction_pairs_min=DEFAULT_INTERACTION_PAIRS_MIN,
        interaction_pairs_max=DEFAULT_INTERACTION_PAIRS_MAX,
        interaction_pairs_law="uniform",
        # Observability
        observability="full" if p_obs >= 1.0 else "sparse",
        p_obs=min(1.0, max(0.0, p_obs)),
        # Logging — disable per-round belief CSV to avoid OOM on large populations.
        # Metrics (entropy, argmax accuracy) are still computed from the live tensor.
        log_beliefs_tensor=False,
        log_beliefs_every_interaction=False,
    )
    cfg.validate()
    return cfg


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_single(
    protos: list[list[float]],
    seed: int,
    num_rounds: int,
    run_dir: Path,
    num_agents: int,
    prototype_update_every: int,
    noise_sigma: float,
    lr_scale: float,
    game_type: str = "ipd",
    p_obs: float = 1.0,
) -> dict[str, Any]:
    """Run one experiment, save per-run outputs, and return metrics."""
    cfg = make_cfg(
        protos,
        seed=seed,
        num_rounds=num_rounds,
        num_agents=num_agents,
        prototype_update_every=prototype_update_every,
        noise_sigma=noise_sigma,
        lr_scale=lr_scale,
        game_type=game_type,
        p_obs=p_obs,
    )
    K = len(protos)
    true_type_probs = np.array(protos, dtype=np.float64)
    true_types = np.array(cfg.force_agent_true_types, dtype=int)

    # Run ESL — results saved to run_dir by trainer
    log, logits, beliefs, summary, _ = run_esl(cfg, run_dir=run_dir)

    # Compute final metrics
    perm, total_ce = match_prototypes_to_types(true_type_probs, logits)
    final_mce = float(total_ce / K)
    final_belief_acc = belief_argmax_accuracy(beliefs, true_types, perm, cfg.num_agents)
    final_belief_ce = mean_belief_ce_vs_types(beliefs, true_types, cfg.num_agents, K)
    learned_p = stable_softmax(logits)

    # Average per-agent return
    total_reward = sum(r["r_i"] + r["r_j"] for r in log.reward_rows)
    num_interactions = len(log.reward_rows)
    avg_return = total_reward / (2 * max(num_interactions, 1))

    # Save summary_metrics.json (matches existing convention)
    summary_payload = {
        "experiment": "prototype_recovery_sweep",
        "K": K,
        "seed": seed,
        "num_rounds": num_rounds,
        "num_agents": num_agents,
        "prototype_update_every": prototype_update_every,
        "population_noise_sigma": noise_sigma,
        "ground_truth_probs": protos,
        "final_mce": final_mce,
        "final_belief_accuracy": final_belief_acc,
        "final_belief_ce": final_belief_ce,
        "avg_return": avg_return,
        "learned_logits": logits.tolist(),
        "learned_probs": learned_p.tolist(),
        "permutation_true_to_learned": perm.tolist(),
        "prototype_update_count": int(summary.get("prototype_update_count", 0)),
        "agents_per_type": {str(k): int(np.sum(true_types == k)) for k in range(K)},
    }
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    # MCE trajectory for plotting
    mce_traj = [
        {"round": int(row["round"]), "mce": float(row.get("matched_cross_entropy", 0.0)) / K}
        for row in log.summary_rows
    ]
    _write_csv(run_dir / "mce_trajectory.csv", mce_traj)

    # Belief accuracy trajectory
    acc_traj = [
        {"round": int(row["round"]), "belief_accuracy": float(row.get("belief_argmax_accuracy", 0.0))}
        for row in log.summary_rows
    ]
    _write_csv(run_dir / "belief_accuracy_trajectory.csv", acc_traj)

    return {
        "seed": seed,
        "K": K,
        "final_mce": final_mce,
        "final_belief_accuracy": final_belief_acc,
        "final_belief_ce": final_belief_ce,
        "avg_return": avg_return,
        "learned_probs": learned_p.tolist(),
        "mce_traj": [r["mce"] for r in mce_traj],
        "acc_traj": [r["belief_accuracy"] for r in acc_traj],
        "rounds": list(range(len(mce_traj))),
    }


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_mce_trajectories(
    all_results: dict[str, list[dict[str, Any]]], out_path: Path
) -> None:
    """MCE over rounds for each K, averaged across seeds with shaded CI."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"K2": "#2196F3", "K3": "#FF9800", "K5": "#4CAF50"}

    for name, results in all_results.items():
        trajs = [np.array(r["mce_traj"]) for r in results]
        min_len = min(len(t) for t in trajs)
        trajs = np.array([t[:min_len] for t in trajs])
        mean = trajs.mean(axis=0)
        if trajs.shape[0] > 1:
            ci = 1.96 * trajs.std(axis=0, ddof=1) / np.sqrt(trajs.shape[0])
        else:
            ci = np.zeros_like(mean)
        rounds = np.arange(min_len)
        ax.plot(rounds, mean, label=name, color=colors.get(name, None), linewidth=1.5)
        ax.fill_between(rounds, mean - ci, mean + ci, alpha=0.2, color=colors.get(name, None))

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Matched Cross-Entropy (MCE, nats)", fontsize=12)
    ax.set_title("Prototype Recovery: MCE Trajectory", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_belief_accuracy_trajectories(
    all_results: dict[str, list[dict[str, Any]]], out_path: Path
) -> None:
    """Belief argmax accuracy over rounds for each K."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"K2": "#2196F3", "K3": "#FF9800", "K5": "#4CAF50"}

    for name, results in all_results.items():
        trajs = [np.array(r["acc_traj"]) for r in results]
        min_len = min(len(t) for t in trajs)
        trajs = np.array([t[:min_len] for t in trajs])
        mean = trajs.mean(axis=0)
        if trajs.shape[0] > 1:
            ci = 1.96 * trajs.std(axis=0, ddof=1) / np.sqrt(trajs.shape[0])
        else:
            ci = np.zeros_like(mean)
        rounds = np.arange(min_len)
        ax.plot(rounds, mean, label=name, color=colors.get(name, None), linewidth=1.5)
        ax.fill_between(rounds, mean - ci, mean + ci, alpha=0.2, color=colors.get(name, None))

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Belief Argmax Accuracy", fontsize=12)
    ax.set_title("Prototype Recovery: Belief Accuracy", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prototype_comparison(
    all_results: dict[str, list[dict[str, Any]]],
    out_path: Path,
    experiments: dict[str, list[list[float]]],
) -> None:
    """Bar chart: ground-truth P(action_0) vs learned P(action_0), last seed per K."""
    n_exp = len(all_results)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]

    for ax, (name, results) in zip(axes, all_results.items()):
        last = results[-1]
        K = last["K"]
        gt_probs = experiments[name]
        learned_probs = last["learned_probs"]

        # Sort learned probs by Hungarian matching for visual alignment
        gt_arr = np.array(gt_probs)
        learned_arr = np.array(learned_probs)
        from esl.metrics import pairwise_assignment_cost, hungarian_min_cost_permutation
        cost = pairwise_assignment_cost(gt_arr, np.log(np.clip(learned_arr, 1e-8, 1.0)))
        perm, _ = hungarian_min_cost_permutation(cost)

        x = np.arange(K)
        width = 0.35
        gt_coop = [gt_probs[k][0] for k in range(K)]
        learned_coop = [learned_probs[int(perm[k])][0] for k in range(K)]

        ax.bar(x - width / 2, gt_coop, width, label="Ground Truth", color="#2196F3", alpha=0.8)
        ax.bar(x + width / 2, learned_coop, width, label="Learned", color="#FF9800", alpha=0.8)
        ax.set_xlabel("Prototype Index", fontsize=11)
        ax.set_ylabel("P(Cooperate)", fontsize=11)
        ax.set_title(f"{name} (seed={last['seed']})", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"θ★_{k}" for k in range(K)])
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Ground-Truth vs Learned P(Cooperate)", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_bar(
    all_results: dict[str, list[dict[str, Any]]], out_path: Path
) -> None:
    """Bar chart of final metrics (MCE, Belief Acc, Belief CE) per K with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        ("final_mce", "MCE (nats) ↓", "#E53935"),
        ("final_belief_accuracy", "Belief Accuracy ↑", "#43A047"),
        ("final_belief_ce", "Belief CE (nats) ↓", "#1E88E5"),
    ]
    names = list(all_results.keys())
    x = np.arange(len(names))

    for ax, (metric, ylabel, color) in zip(axes, metrics):
        means, cis = [], []
        for name in names:
            vals = [r[metric] for r in all_results[name]]
            m = np.mean(vals)
            c = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            means.append(m)
            cis.append(c)
        ax.bar(x, means, yerr=cis, capsize=5, color=color, alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        # Add value labels on bars
        for i, (m, c) in enumerate(zip(means, cis)):
            ax.text(i, m + c + 0.02, f"{m:.3f}", ha="center", fontsize=9)

    fig.suptitle("Aggregate Metrics (mean ± 95% CI)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main sweep logic ─────────────────────────────────────────────────────


def run_sweep(
    out_root: Path,
    num_seeds: int,
    num_rounds: int,
    num_agents: int,
    prototype_update_every: int,
    noise_sigma: float,
    lr_scale: float,
    game_type: str = "ipd",
    p_obs: float = 1.0,
    no_plots: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Run all K configurations across seeds, save everything."""
    out_root = Path(out_root).resolve()
    all_results: dict[str, list[dict[str, Any]]] = {}

    # Select prototypes for this game
    experiments = GAME_PROTOTYPES.get(game_type, GAME_PROTOTYPES["ipd"])
    game_label = {"ipd": "Prisoner's Dilemma", "stag_hunt": "Stag Hunt", "matching_pennies": "Matching Pennies"}
    obs_label = f"full (p=1.0)" if p_obs >= 1.0 else f"sparse (p={p_obs})"
    print(f"\n  Game: {game_label.get(game_type, game_type)}  |  Observability: {obs_label}")

    for name, protos in experiments.items():
        K = len(protos)
        print(f"\n{'='*60}")
        print(f"  {name}  |  K={K}  |  Prototypes: {protos}")
        print(f"{'='*60}")
        results = []
        for s in range(num_seeds):
            seed = 42 + s
            run_dir = out_root / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Seed {seed} → {run_dir} ... ", end="", flush=True)
            r = run_single(
                protos,
                seed=seed,
                num_rounds=num_rounds,
                run_dir=run_dir,
                num_agents=num_agents,
                prototype_update_every=prototype_update_every,
                noise_sigma=noise_sigma,
                lr_scale=lr_scale,
                game_type=game_type,
                p_obs=p_obs,
            )
            print(
                f"MCE={r['final_mce']:.4f}  "
                f"BeliefAcc={r['final_belief_accuracy']:.3f}  "
                f"BeliefCE={r['final_belief_ce']:.4f}  "
                f"AvgReturn={r['avg_return']:.2f}"
            )
            results.append(r)
        all_results[name] = results

    # Save aggregate results
    aggregate = {}
    for name, results in all_results.items():
        agg: dict[str, Any] = {"K": results[0]["K"], "seeds": len(results)}
        for metric in ("final_mce", "final_belief_accuracy", "final_belief_ce", "avg_return"):
            vals = [r[metric] for r in results]
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{metric}_ci95"] = (
                1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            )
        aggregate[name] = agg
    (out_root / "aggregate_results.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    # Save experiment manifest
    manifest = {
        "experiment": "prototype_recovery_sweep",
        "game_type": game_type,
        "observability": "full" if p_obs >= 1.0 else "sparse",
        "p_obs": p_obs,
        "num_agents": num_agents,
        "prototype_update_every_M": prototype_update_every,
        "interaction_pairs_L_t": [DEFAULT_INTERACTION_PAIRS_MIN, DEFAULT_INTERACTION_PAIRS_MAX],
        "population_noise_sigma": noise_sigma,
        "prototype_lr_scale": lr_scale,
        "num_rounds": num_rounds,
        "num_seeds": num_seeds,
        "experiments": {name: protos for name, protos in experiments.items()},
    }
    (out_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Print aggregate table
    print(f"\n\n{'='*80}")
    print(f"  AGGREGATE RESULTS (mean ± 95% CI)")
    print(f"{'='*80}")
    print(f"  N={num_agents}, M={prototype_update_every}, L_t∈[{DEFAULT_INTERACTION_PAIRS_MIN},{DEFAULT_INTERACTION_PAIRS_MAX}], σ={noise_sigma}, p_obs={p_obs}")
    print(f"{'='*80}")

    def fmt_ci(vals: list[float], f: str) -> str:
        m = np.mean(vals)
        c = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return f"{m:{f}} ± {c:{f}}"

    print(f"{'Exp':<8} {'MCE ↓':>16} {'Belief Acc ↑':>16} {'Belief CE ↓':>16} {'Avg Return':>16}")
    print(f"{'-'*8} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")
    for name, results in all_results.items():
        m = [r["final_mce"] for r in results]
        a = [r["final_belief_accuracy"] for r in results]
        c = [r["final_belief_ce"] for r in results]
        ret = [r["avg_return"] for r in results]
        print(f"{name:<8} {fmt_ci(m, '.4f'):>16} {fmt_ci(a, '.3f'):>16} {fmt_ci(c, '.4f'):>16} {fmt_ci(ret, '.2f'):>16}")

    # Learned prototypes (last seed)
    print(f"\n{'='*80}")
    print(f"  LEARNED PROTOTYPES (last seed)")
    print(f"{'='*80}")
    for name, results in all_results.items():
        last = results[-1]
        gt = experiments[name]
        print(f"\n  {name}:")
        print(f"    Ground-truth:  {['[{:.2f}, {:.2f}]'.format(*p) for p in gt]}")
        print(f"    Learned:       {['[{:.2f}, {:.2f}]'.format(*p) for p in last['learned_probs']]}")

    # Generate plots
    if not no_plots:
        print(f"\n  Generating plots...")
        plot_mce_trajectories(all_results, out_root / "figure_mce_trajectory.png")
        plot_belief_accuracy_trajectories(all_results, out_root / "figure_belief_accuracy.png")
        plot_prototype_comparison(all_results, out_root / "figure_prototype_comparison.png", experiments)
        plot_aggregate_bar(all_results, out_root / "figure_aggregate_metrics.png")
        print(f"  Saved figures to {out_root}/figure_*.png")

    print(f"\n  All outputs: {out_root}/")
    return all_results


def main() -> None:
    p = argparse.ArgumentParser(
        description="ESL prototype recovery sweep (IPD / Stag Hunt / Matching Pennies)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=Path("runs/prototype_recovery_sweep"),
                    help="Output directory root")
    p.add_argument("--game", type=str, default="ipd",
                    choices=["ipd", "stag_hunt", "matching_pennies"],
                    help="Stage game to use")
    p.add_argument("--seeds", type=int, default=5,
                    help="Number of random seeds (seed_42, seed_43, ...)")
    p.add_argument("--rounds", type=int, default=2000,
                    help="Number of rounds per run")
    p.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS,
                    help="Population size N")
    p.add_argument("--M", type=int, default=DEFAULT_PROTOTYPE_UPDATE_EVERY,
                    dest="prototype_update_every",
                    help="Rounds between prototype SGD steps")
    p.add_argument("--noise-sigma", type=float, default=DEFAULT_POPULATION_NOISE_SIGMA,
                    help="Per-agent Gaussian noise σ on logits")
    p.add_argument("--lr-scale", type=float, default=DEFAULT_PROTOTYPE_LR_SCALE,
                    help="Prototype learning rate scale γ")
    p.add_argument("--p-obs", type=float, default=DEFAULT_P_OBS,
                    help="Observation probability (1.0=full, <1.0=sparse)")
    p.add_argument("--no-plots", action="store_true",
                    help="Skip plot generation (metrics only)")
    args = p.parse_args()

    run_sweep(
        out_root=args.out,
        num_seeds=args.seeds,
        num_rounds=args.rounds,
        num_agents=args.num_agents,
        prototype_update_every=args.prototype_update_every,
        noise_sigma=args.noise_sigma,
        lr_scale=args.lr_scale,
        game_type=args.game,
        p_obs=args.p_obs,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
