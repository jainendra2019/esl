"""
Deterministic “paper trace” scenario: 2 agents, observer 0, target 1 (Always Defect),
fixed logits θ₁=[0.2,0], θ₂=[0,0.2], full observability, signal Defect every round.

Run:
  python -m esl.hand_trace [--rounds 10] [--m 5]           # repeated Defect (default)
  python -m esl.hand_trace --cooperate [--rounds 10] [--m 5]  # repeated Cooperate mirror

Prints:
  • prototype_update_every (M), and for each SGD step: index m, env_round_ended,
    prototype_update_norm, θ_k before/after, softmax p_k before/after
  • CSV: round,b0,b1,entropy for b_{0→1}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from esl import beliefs as belief_ops
from esl.config import ESLConfig
from esl.prototypes import likelihoods, stable_softmax
from esl.trainer import RunLog, run_esl

RepeatedSignal = Literal["cooperate", "defect"]


def paper_trace_config(
    *,
    num_rounds: int = 10,
    prototype_update_every: int = 100,
    seed: int = 0,
    repeated_signal: RepeatedSignal = "defect",
) -> ESLConfig:
    """
    Walkthrough: K=2, N=2, (i,j)=(0,1), fixed θ.

    * repeated_signal=defect: target j is Always Defect → s=1 every round (mirror of paper AD trace).
    * repeated_signal=cooperate: both agents AC → j always Cooperate → s=0 every round.
    """
    if repeated_signal == "defect":
        force_types = [0, 1]
    else:
        force_types = [0, 0]
    return ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=2,
        num_prototypes=2,
        num_rounds=num_rounds,
        prototype_update_every=prototype_update_every,
        observability="full",
        force_ordered_pair=(0, 1),
        force_agent_true_types=force_types,
        prototype_logits_override=[[0.2, 0.0], [0.0, 0.2]],
        init_noise=0.01,
        symmetric_init=False,
    )


def print_prototype_update_log(log: RunLog, cfg: ESLConfig) -> None:
    """Pretty-print θ, softmax p, update index m, and ‖γ·ḡ‖ for each SGD step."""
    events = log.prototype_update_events
    if not events:
        print("\n(no prototype updates — increase rounds or lower --m)")
        return
    print(f"\nprototype_update_every (M) = {cfg.prototype_update_every}")
    for ev in events:
        m = ev["prototype_update_index_m"]
        t_end = ev["env_round_ended"]
        ff = ev["final_flush"]
        norm = ev["prototype_update_norm"]
        tb = ev["theta_before"]
        ta = ev["theta_after"]
        pb = ev["p_before"]
        pa = ev["p_after"]
        tag = "FINAL_FLUSH" if ff else "scheduled"
        print(
            f"\n--- prototype update #{m} ({tag}) | "
            f"env_round_ended={t_end} (0-based) | prototype_update_norm={norm:.6g} ---"
        )
        for k in range(len(tb)):
            print(f"  theta_{k}_before = {tb[k]}")
            print(f"  theta_{k}_after  = {ta[k]}")
            print(f"  p_{k}_before (softmax) = {pb[k]}")
            print(f"  p_{k}_after  (softmax) = {pa[k]}")


def belief_entropy_pair(b: np.ndarray) -> float:
    b = np.clip(b, 1e-12, 1.0)
    return float(-np.sum(b * np.log(b)))


def isolated_belief_trajectory(cfg: ESLConfig, logits: np.ndarray, signal: int, steps: int) -> list[np.ndarray]:
    """Reference: repeated Bayes with fixed logits and fixed signal (same as trainer belief math)."""
    b = np.ones(cfg.num_prototypes) / cfg.num_prototypes
    traj = [b.copy()]
    lk = likelihoods(logits, signal)
    for _ in range(steps):
        b = belief_ops.update_belief_pair(
            b,
            lk,
            cfg.delta_simplex,
            cfg.bayes_denominator_eps,
        )
        traj.append(b.copy())
    return traj


def main() -> None:
    p = argparse.ArgumentParser(description="Print hand-trace belief CSV (paper scenario)")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument(
        "--m",
        type=int,
        default=100,
        help="prototype_update_every (use >> rounds to keep θ fixed for belief-only comparison)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="write CSV to this path")
    p.add_argument(
        "--cooperate",
        action="store_true",
        help="mirror trace: target always Cooperate (s=0) instead of Defect (s=1)",
    )
    args = p.parse_args()

    rep: RepeatedSignal = "cooperate" if args.cooperate else "defect"
    obs_action = 0 if rep == "cooperate" else 1
    cfg = paper_trace_config(
        num_rounds=args.rounds,
        prototype_update_every=args.m,
        seed=args.seed,
        repeated_signal=rep,
    )
    logits = np.array(cfg.prototype_logits_override, dtype=np.float64)
    lk = likelihoods(logits, obs_action)
    lab = "Cooperate" if obs_action == 0 else "Defect"
    print(f"Repeated observed signal: {lab} (s={obs_action})")
    print(f"Likelihoods L_k(s={obs_action}):", lk)
    print("Softmax(prototype k):")
    for k in range(2):
        print(f"  k={k}", stable_softmax(logits[k : k + 1])[0])

    run_dir = args.out.parent if args.out else None
    log, _, _, summary, rd = run_esl(cfg, run_dir=run_dir)

    print_prototype_update_log(log, cfg)

    # Row t = belief b_{0→1} **after** the Bayes update in environment step t (user “round t+1”).
    lines = ["round,b0,b1,entropy_b01"]
    for t in range(cfg.num_rounds):
        row = next(
            r
            for r in log.belief_rows
            if int(r["round"]) == t and int(r["i"]) == 0 and int(r["j"]) == 1
        )
        b0, b1 = float(row["b_0"]), float(row["b_1"])
        b = np.array([b0, b1])
        lines.append(f"{t},{b0:.6f},{b1:.6f},{belief_entropy_pair(b):.6f}")

    text = "\n".join(lines)
    print(text)
    print("---")
    print(f"run_dir={rd}")
    print(f"final_matched_cross_entropy={summary['final_matched_cross_entropy']:.6f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
