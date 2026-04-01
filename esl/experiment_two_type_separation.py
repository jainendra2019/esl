"""
Two-type latent recovery experiment: main vs symmetric vs freeze-prototype baseline.

Recovery mode only; one random ordered pair per round; prototypes every M rounds.

Usage:
  python -m esl.experiment_two_type_separation \\
    --rounds 250 --m 5 --seed 42 --lr-scale 18 --out runs/two_type_sep_main
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from esl.config import ESLConfig
from esl.metrics import match_prototypes_to_types
from esl.prototypes import stable_softmax
from esl.trainer import run_esl

# True one-shot policies for labeling: type 0 = Always C, type 1 = Always D
_TRUE_TYPE_SOFTMAX = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

Condition = Literal["main", "symmetric", "freeze_proto_baseline"]

SIX_AGENT_TWO_TYPE = [0, 0, 0, 1, 1, 1]
MAIN_THETA = [[0.01, -0.01], [-0.01, 0.01]]
DEFAULT_LR_SCALE = 18.0
CONDITION_ORDER: tuple[Condition, ...] = ("main", "symmetric", "freeze_proto_baseline")


def _normalize_condition(name: str) -> Condition:
    if name == "symmetric_baseline":
        return "symmetric"
    if name in ("main", "symmetric", "freeze_proto_baseline"):
        return name  # type: ignore[return-value]
    raise ValueError(f"unknown condition: {name!r}")


def build_two_type_separation_config(
    condition: str,
    *,
    num_rounds: int = 250,
    prototype_update_every: int = 5,
    seed: int = 42,
    lr_scale: float = DEFAULT_LR_SCALE,
    prototype_lr_scale: float | None = None,
    p_obs: float = 1.0,
    num_agents: int = 6,
    num_prototypes: int = 2,
    delta_simplex: float = 1e-4,
    game: str = "ipd",
) -> ESLConfig:
    """Build ESLConfig for one of three experiment conditions."""
    if prototype_lr_scale is not None:
        lr_scale = float(prototype_lr_scale)
    if game != "ipd":
        raise ValueError("only game='ipd' is supported in this experiment driver")
    c = _normalize_condition(condition)
    base = ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=num_agents,
        num_prototypes=num_prototypes,
        num_rounds=num_rounds,
        prototype_update_every=prototype_update_every,
        observability="full" if p_obs >= 1.0 else "sparse",
        p_obs=min(1.0, max(0.0, float(p_obs))),
        delta_simplex=delta_simplex,
        force_agent_true_types=list(SIX_AGENT_TWO_TYPE[:num_agents])
        if num_agents == 6
        else [i % num_prototypes for i in range(num_agents)],
        symmetric_init=False,
        init_noise=0.0,
        learning_frozen=False,
        freeze_prototype_parameters=False,
        prototype_lr_scale=1.0,
    )
    if c == "main":
        return replace(
            base,
            prototype_logits_override=[list(row) for row in MAIN_THETA],
            symmetric_init=False,
            prototype_lr_scale=float(lr_scale),
        )
    if c == "symmetric":
        return replace(
            base,
            prototype_logits_override=None,
            symmetric_init=True,
            init_noise=0.0,
            prototype_lr_scale=1.0,
        )
    if c == "freeze_proto_baseline":
        return replace(
            base,
            prototype_logits_override=[list(row) for row in MAIN_THETA],
            freeze_prototype_parameters=True,
            prototype_lr_scale=float(lr_scale),
        )
    raise ValueError(c)


# Backward-compatible name for tests
two_type_separation_config = build_two_type_separation_config


def acceptance_report(logits: np.ndarray, summary: dict[str, Any]) -> dict[str, Any]:
    """Hungarian-matched prototype purity vs AC/AD and belief accuracy."""
    true_probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    perm, final_ce = match_prototypes_to_types(true_probs, logits)
    learned_p = stable_softmax(logits)
    purities: dict[str, float] = {}
    ok_sep = True
    for t in range(2):
        k = int(perm[t])
        p = learned_p[k]
        if t == 0:
            purities["matched_P_coop_type_on_C"] = float(p[0])
            if p[0] <= 0.85:
                ok_sep = False
        else:
            purities["matched_P_defect_type_on_D"] = float(p[1])
            if p[1] <= 0.85:
                ok_sep = False
    acc = float(summary["final_belief_argmax_accuracy"])
    return {
        "permutation_true_to_learned": perm.tolist(),
        "final_matched_cross_entropy": float(final_ce),
        "prototype_separation_thresholds_met": ok_sep,
        "belief_argmax_accuracy": acc,
        "belief_accuracy_threshold_90pct": acc >= 0.90,
        **purities,
    }


def run_condition(cfg: ESLConfig, label: str, run_dir: Path) -> dict[str, Any]:
    """Run one condition; returns artifacts for saving and plotting."""
    run_dir = Path(run_dir)
    log, logits, belief_tensor, summary, run_dir = run_esl(cfg, run_dir=run_dir)
    return {
        "label": label,
        "cfg": cfg,
        "log": log,
        "logits": logits,
        "belief_tensor": belief_tensor,
        "summary": summary,
        "run_dir": run_dir,
    }


def _spec_prototype_rows(
    run: dict[str, Any],
    *,
    condition: str,
    seed: int,
) -> list[dict[str, Any]]:
    log = run["log"]
    cfg: ESLConfig = run["cfg"]
    rows: list[dict[str, Any]] = []
    if not log.prototype_rows:
        return rows
    pr0 = log.prototype_rows[0]
    init: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "round": 0,
        "prototype_update_idx": -1,
        "prototype_update_norm": 0.0,
    }
    for k in range(cfg.num_prototypes):
        for a in range(cfg.num_actions):
            init[f"theta_{k}_a{a}"] = float(pr0[f"theta_{k}_{a}"])
    for k in range(cfg.num_prototypes):
        init[f"p_{k}_cooperate"] = float(pr0[f"softmax_{k}_0"])
        init[f"p_{k}_defect"] = float(pr0[f"softmax_{k}_1"])
    rows.append(init)

    for ev in log.prototype_update_events:
        r = {
            "condition": condition,
            "seed": seed,
            "round": int(ev["env_round_ended"]),
            "prototype_update_idx": int(ev["prototype_update_index_m"]),
            "prototype_update_norm": float(ev["prototype_update_norm"]),
        }
        theta_after = np.asarray(ev["theta_after"], dtype=np.float64)
        p_after = np.asarray(ev["p_after"], dtype=np.float64)
        for k in range(cfg.num_prototypes):
            for a in range(cfg.num_actions):
                r[f"theta_{k}_a{a}"] = float(theta_after[k, a])
        for k in range(cfg.num_prototypes):
            r[f"p_{k}_cooperate"] = float(p_after[k, 0])
            r[f"p_{k}_defect"] = float(p_after[k, 1])
        rows.append(r)
    return rows


def _belief_metric_rows(
    run: dict[str, Any],
    *,
    condition: str,
    seed: int,
    csv_every: int,
    p_obs: float,
) -> list[dict[str, Any]]:
    log = run["log"]
    cfg: ESLConfig = run["cfg"]
    out: list[dict[str, Any]] = []
    coop_j, def_j = 1, min(5, cfg.num_agents - 1)
    obs_i = 0

    def belief_rep(round_t: int, i: int, j: int, k: int) -> float:
        for br in log.belief_rows:
            if int(br["round"]) == round_t and int(br["i"]) == i and int(br["j"]) == j:
                return float(br[f"b_{k}"])
        return float("nan")

    n_exec = len(log.summary_rows)
    for t in range(0, n_exec, csv_every):
        srow = log.summary_rows[t]
        if p_obs >= 1.0:
            obs_count = t + 1
        else:
            obs_count = int(round((t + 1) * p_obs))
        row = {
            "condition": condition,
            "seed": seed,
            "round": t,
            "avg_belief_entropy": float(srow["belief_entropy_mean"]),
            "belief_argmax_accuracy": float(srow["belief_argmax_accuracy"]),
            "matched_cross_entropy": float(srow["matched_cross_entropy"]),
            "obs_count_so_far": obs_count,
            "belief_rep_coop_type0": belief_rep(t, obs_i, coop_j, 0),
            "belief_rep_def_type0": belief_rep(t, obs_i, def_j, 0),
        }
        out.append(row)
    return out


def _write_csv_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_condition_outputs(run: dict[str, Any], out_dir: Path, *, csv_every: int, p_obs: float) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg: ESLConfig = run["cfg"]
    label = str(run["label"])
    seed = cfg.seed

    cfg.save_json(out_dir / "config.json")

    n_exec = len(run["log"].summary_rows)
    learned = stable_softmax(run["logits"])
    summary_payload = {
        "condition": label,
        "seed": seed,
        "num_rounds": cfg.num_rounds,
        "num_rounds_executed": int(run["summary"].get("num_rounds_executed", n_exec)),
        "stopped_on_convergence": bool(run["summary"].get("stopped_on_convergence", False)),
        "convergence_round": run["summary"].get("convergence_round"),
        "prototype_update_every": cfg.prototype_update_every,
        "prototype_update_count": int(run["summary"]["prototype_update_count"]),
        "learning_frozen": cfg.learning_frozen,
        "freeze_prototype_parameters": cfg.freeze_prototype_parameters,
        "final_matched_cross_entropy": float(run["summary"]["final_matched_cross_entropy"]),
        "final_belief_argmax_accuracy": float(run["summary"]["final_belief_argmax_accuracy"]),
        "final_avg_belief_entropy": float(run["summary"]["final_belief_entropy"]),
        "final_p_0_cooperate": float(learned[0, 0]),
        "final_p_1_cooperate": float(learned[1, 0]),
        "prototype_separation_abs": float(abs(learned[0, 0] - learned[1, 0])),
        "true_types": list(cfg.force_agent_true_types or []),
        "obs_prob": float(cfg.p_obs),
    }
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    proto_rows = _spec_prototype_rows(run, condition=label, seed=seed)
    _write_csv_dicts(out_dir / "prototype_trajectory.csv", proto_rows)

    belief_rows = _belief_metric_rows(
        run, condition=label, seed=seed, csv_every=csv_every, p_obs=p_obs
    )
    _write_csv_dicts(out_dir / "belief_metrics.csv", belief_rows)


def _final_match_perm(run: dict[str, Any]) -> np.ndarray:
    """Hungarian map true_type -> learned prototype index (fixed using final logits)."""
    perm, _ = match_prototypes_to_types(_TRUE_TYPE_SOFTMAX, run["logits"])
    return perm


def _series_matched_p_coop(
    run: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per round: P(C) for protos matched (at final time) to Always Cooperate vs Always Defect,
    and |ΔP(C)| between them. Labels are stable for the whole trajectory.
    """
    perm = _final_match_perm(run)
    k_coop, k_def = int(perm[0]), int(perm[1])
    log = run["log"]
    r = np.array([int(x["round"]) for x in log.prototype_rows], dtype=np.int64)
    p_ac = np.array([float(x[f"softmax_{k_coop}_0"]) for x in log.prototype_rows], dtype=np.float64)
    p_ad = np.array([float(x[f"softmax_{k_def}_0"]) for x in log.prototype_rows], dtype=np.float64)
    sep = np.abs(p_ac - p_ad)
    return r, p_ac, p_ad, sep


def plot_prototype_separation(all_runs: dict[str, dict[str, Any]], out_path: Path) -> None:
    """
    Panels 0–2: P(C) for protos matched (final Hungarian) to Always C vs Always D.
    Panel 3: |P(C)_AC-match − P(C)_AD-match| for all three conditions.
    """
    out_path = Path(out_path)
    fig = plt.figure(figsize=(8, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.72], hspace=0.38)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    titles = {
        "main": "Main (asymmetric init)",
        "symmetric": "Symmetric control",
        "freeze_proto_baseline": "Freeze prototype",
    }
    label_ac = r"$P$(C) — proto matched to Always Cooperate"
    label_ad = r"$P$(C) — proto matched to Always Defect"

    for ax, key in zip(axes[:3], CONDITION_ORDER):
        run = all_runs[key]
        rounds, p_ac, p_ad, _sep = _series_matched_p_coop(run)
        ax.plot(rounds, p_ac, label=label_ac, color="C0")
        ax.plot(rounds, p_ad, label=label_ad, color="C1", linestyle="--")
        ax.set_ylabel(r"$P$(cooperate)")
        ax.set_title(titles.get(key, key))
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="right", fontsize=7)

    ax_sep = axes[3]
    pretty = {"main": "main", "symmetric": "symmetric", "freeze_proto_baseline": "freeze proto"}
    for key in CONDITION_ORDER:
        run = all_runs[key]
        rounds, _a, _b, sep = _series_matched_p_coop(run)
        ax_sep.plot(rounds, sep, label=pretty[key])
    ax_sep.set_ylabel("|ΔP(C)| (matched protos, final θ)")
    ax_sep.set_xlabel("round")
    ax_sep.set_title("Prototype separation (matched labels, final permutation)")
    ax_sep.set_ylim(-0.05, 1.05)
    ax_sep.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Prototype cooperation (curves use Hungarian match at final θ; same index map for all rounds)",
        y=0.995,
        fontsize=10,
    )
    fig.subplots_adjust(top=0.93)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matched_ce(all_runs: dict[str, dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for key in CONDITION_ORDER:
        run = all_runs[key]
        log = run["log"]
        r = [int(s["round"]) for s in log.summary_rows]
        ce = [float(s["matched_cross_entropy"]) for s in log.summary_rows]
        ax.plot(r, ce, label=key)
    ax.set_xlabel("round")
    ax.set_ylabel("matched cross-entropy (nats)")
    ax.legend(fontsize=8)
    ax.set_title("Matched CE vs true AC/AD (Hungarian, per-round optimal match)")
    ax.set_ylim(0.0, 2.0)
    fig.text(
        0.5,
        0.02,
        "Note: y-axis clipped at 2 nats. Early spikes often reflect permutation mismatch before prototypes specialize.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="0.35",
    )
    fig.subplots_adjust(bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_belief_entropy(all_runs: dict[str, dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in CONDITION_ORDER:
        run = all_runs[key]
        log = run["log"]
        r = [int(s["round"]) for s in log.summary_rows]
        h = [float(s["belief_entropy_mean"]) for s in log.summary_rows]
        ax.plot(r, h, label=key)
    ax.set_xlabel("round")
    ax.set_ylabel("Mean belief entropy (nats)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    manifest: dict[str, Any] = {
        "experiment": "two_type_separation",
        "rounds": args.rounds,
        "m": args.m,
        "seed": args.seed,
        "lr_scale": args.lr_scale,
        "obs_prob": args.obs_prob,
        "conditions": list(CONDITION_ORDER),
        "until_converged": bool(getattr(args, "until_converged", False)),
    }
    if manifest["until_converged"]:
        manifest["conv_window"] = args.conv_window
        manifest["conv_eps_h"] = args.conv_eps_h
        manifest["conv_eps_delta"] = args.conv_eps_delta
        manifest["conv_eps_theta"] = args.conv_eps_theta
        manifest["conv_eps_b"] = args.conv_eps_b
    (Path(out_dir) / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Two-type prototype separation experiment (3 conditions)")
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--m", type=int, default=5, help="prototype update every M rounds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr-scale", type=float, default=DEFAULT_LR_SCALE, dest="lr_scale")
    p.add_argument("--out", type=Path, required=True, help="output directory root")
    p.add_argument("--obs-prob", type=float, default=1.0, dest="obs_prob")
    p.add_argument("--n-agents", type=int, default=6, dest="n_agents")
    p.add_argument("--n-prototypes", type=int, default=2, dest="n_prototypes")
    p.add_argument("--delta", type=float, default=1e-4, dest="delta_simplex")
    p.add_argument("--no-plots", action="store_true", help="write metrics only")
    p.add_argument("--csv-every", type=int, default=1, dest="csv_every")
    p.add_argument("--game", type=str, default="ipd")
    p.add_argument(
        "--until-converged",
        action="store_true",
        help="stop early when convergence criteria hold; --rounds is T_max (see ALGORITHM.md)",
    )
    p.add_argument("--conv-window", type=int, default=50, dest="conv_window")
    p.add_argument("--conv-eps-h", type=float, default=0.1, dest="conv_eps_h")
    p.add_argument("--conv-eps-delta", type=float, default=0.8, dest="conv_eps_delta")
    p.add_argument("--conv-eps-theta", type=float, default=0.01, dest="conv_eps_theta")
    p.add_argument("--conv-eps-b", type=float, default=0.01, dest="conv_eps_b")
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_runs: dict[str, dict[str, Any]] = {}

    for cond in CONDITION_ORDER:
        cfg = build_two_type_separation_config(
            cond,
            num_rounds=args.rounds,
            prototype_update_every=args.m,
            seed=args.seed,
            lr_scale=args.lr_scale,
            p_obs=args.obs_prob,
            num_agents=args.n_agents,
            num_prototypes=args.n_prototypes,
            delta_simplex=args.delta_simplex,
            game=args.game,
        )
        if args.until_converged:
            cfg = replace(
                cfg,
                stop_on_convergence=True,
                convergence_window_w=args.conv_window,
                convergence_epsilon_h=args.conv_eps_h,
                convergence_epsilon_delta=args.conv_eps_delta,
                convergence_epsilon_theta=args.conv_eps_theta,
                convergence_epsilon_b=args.conv_eps_b,
            )
        sub = out_root / cond
        run = run_condition(cfg, cond, sub)
        save_condition_outputs(run, sub, csv_every=args.csv_every, p_obs=args.obs_prob)
        all_runs[cond] = run

        # Guardrails
        assert (sub / "config.json").is_file()
        assert (sub / "summary_metrics.json").is_file()
        assert (sub / "prototype_trajectory.csv").is_file()
        assert (sub / "belief_metrics.csv").is_file()
        proto_lines = (sub / "prototype_trajectory.csv").read_text(encoding="utf-8").strip().splitlines()
        n_proto_data = max(0, len(proto_lines) - 1)
        learns = not cfg.freeze_prototype_parameters and not cfg.learning_frozen
        if learns:
            assert n_proto_data >= 1
            # At least one real prototype update row (idx >= 0) when updates should occur
            assert int(run["summary"]["prototype_update_count"]) >= 1

    if not args.no_plots:
        plot_prototype_separation(all_runs, out_root / "figure_prototype_separation.png")
        plot_matched_ce(all_runs, out_root / "figure_matched_ce.png")
        plot_belief_entropy(all_runs, out_root / "figure_belief_entropy.png")
        assert (out_root / "figure_prototype_separation.png").is_file()

    save_manifest(args, out_root)
    assert (out_root / "experiment_manifest.json").is_file()

    print(f"Wrote experiment under: {out_root}")


if __name__ == "__main__":
    main()
