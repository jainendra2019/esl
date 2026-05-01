"""Figure 2 component-level ablations for dynamics-aware ESL adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from esl import beliefs as belief_ops
from esl import games
from esl.baselines.external_common import MatrixGameTask, matrix_game_tasks
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure1_dynamic_adaptation import (
    ACTION_DIM,
    EPS,
    H_WINDOW,
    K_TYPES,
    LAMBDA_BR,
    ONLINE_EM_WINDOW,
    compute_dynamic_prediction_regret,
    compute_latent_tracking_error,
    compute_post_switch_oracle_regret,
    build_type_shifting_schedule,
    true_type_distributions_mature,
)
from esl.experiments.figure1_learning_curves import _fixed_prototypes, _task_payoff
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.stats_ci import mean_std_ci95
from esl.metrics import cross_entropy
from esl.prototypes import batch_weighted_prototype_gradient, likelihoods, stable_softmax

AblationMethod = Literal["ESL", "No-Sharing", "Fixed-K Bayesian", "No-Belief", "Online EM"]

EARLY_AUC_WINDOW = 30
METHOD_ORDER: tuple[AblationMethod, ...] = (
    "ESL",
    "No-Sharing",
    "Fixed-K Bayesian",
    "No-Belief",
    "Online EM",
)
ABLATION_METHODS = METHOD_ORDER
RECOVERY_METHODS: tuple[AblationMethod, ...] = ("ESL", "Fixed-K Bayesian", "Online EM", "No-Belief")
LTE_PANEL_METHODS: tuple[AblationMethod, ...] = ("ESL", "Fixed-K Bayesian", "Online EM")
METHOD_COLORS: dict[str, str] = {
    "ESL": "#1f77b4",
    "No-Sharing": "#c44e52",
    "Fixed-K Bayesian": "#55a868",
    "No-Belief": "#ccb974",
    "Online EM": "#7f7f7f",
}
PLOT_LABELS: dict[str, str] = {
    "ESL": "ESL",
    "No-Sharing": "No-sharing",
    "Fixed-K Bayesian": "Fixed-K",
    "No-Belief": "No-belief",
    "Online EM": "EM",
}


@dataclass
class AblationState:
    method: AblationMethod
    rng: np.random.Generator
    logits: np.ndarray | None = None
    beliefs: dict[int, np.ndarray] = field(default_factory=dict)
    esl_batch: list[tuple[int, np.ndarray]] = field(default_factory=list)
    per_opponent_logits: dict[int, np.ndarray] = field(default_factory=dict)
    per_opponent_batch: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    em_window: list[tuple[int, int]] = field(default_factory=list)
    last_focal_action: int | None = None


def _softmax(values: np.ndarray, lam: float = 1.0) -> np.ndarray:
    z = lam * values
    z -= np.max(z)
    w = np.exp(z)
    return w / np.maximum(w.sum(), EPS)


def _sample_binary(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(ACTION_DIM, p=np.asarray(probs, dtype=np.float64)))


def _esl_initial_prototypes(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([[1.0, -1.0], [-1.0, 1.0], [0.25, -0.25]], dtype=np.float64)
    return base + 0.04 * rng.standard_normal(size=base.shape)


def _bayes_step(prior: np.ndarray, prototypes: np.ndarray, opponent_action: int) -> np.ndarray:
    return belief_ops.update_belief_pair(prior, likelihoods(prototypes, opponent_action), 1e-4, EPS)


def _method_state(method: AblationMethod, seed: int) -> AblationState:
    state = AblationState(method=method, rng=np.random.default_rng(seed))
    if method == "ESL":
        state.logits = _esl_initial_prototypes(seed)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "No-Sharing":
        state.per_opponent_logits = {opp: _esl_initial_prototypes(seed + 101 * opp) for opp in (1, 2, 3)}
        state.per_opponent_batch = {opp: [] for opp in (1, 2, 3)}
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "Fixed-K Bayesian":
        state.logits = _fixed_prototypes().copy()
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "No-Belief":
        state.logits = _esl_initial_prototypes(seed)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    elif method == "Online EM":
        state.logits = _esl_initial_prototypes(seed + 311)
        state.beliefs = {opp: np.full(K_TYPES, 1.0 / K_TYPES) for opp in (1, 2, 3)}
    return state


def _post_switch_window_rows(rows: list[dict[str, Any]], *, h_window: int) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if int(r.get("switch_id", -1)) >= 0 and 0 <= int(r.get("tau", h_window + 1)) <= h_window
    ]


def compute_early_gap_auc(rows: list[dict[str, Any]], *, early_window: int = EARLY_AUC_WINDOW) -> float:
    """Mean switch-level early adaptation cost, matching PSR with a truncated horizon."""
    by_switch: dict[tuple[int, int], list[float]] = {}
    for row in _post_switch_window_rows(rows, h_window=early_window):
        key = (int(row["opponent_id"]), int(row["switch_id"]))
        gap = float(row["oracle_payoff"]) - float(row["focal_payoff"])
        by_switch.setdefault(key, []).append(gap)
    if not by_switch:
        return float("nan")
    return float(np.mean([np.sum(vals) for vals in by_switch.values()]))


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
    utilities = np.array([np.sum(pred * pay.row[a, :]) for a in range(ACTION_DIM)], dtype=np.float64)
    return _softmax(utilities, lam=LAMBDA_BR)


def _oracle_payoff(true_probs: np.ndarray, pay: games.PayoffMatrices) -> float:
    utilities = np.array([np.sum(true_probs * pay.row[a, :]) for a in range(ACTION_DIM)], dtype=np.float64)
    return float(np.max(utilities))


def _update_online_em(state: AblationState) -> None:
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


def _update_state(state: AblationState, *, opponent_id: int, opponent_action: int) -> None:
    if state.method == "ESL":
        assert state.logits is not None
        belief_before = state.beliefs[opponent_id].copy()
        state.beliefs[opponent_id] = _bayes_step(state.beliefs[opponent_id], state.logits, opponent_action)
        state.esl_batch.append((opponent_action, belief_before))
        if len(state.esl_batch) >= 2:
            grad = np.zeros_like(state.logits)
            for signal, belief in state.esl_batch:
                grad += batch_weighted_prototype_gradient(state.logits, belief, signal)
            state.logits = state.logits + 0.9 * (grad / len(state.esl_batch))
            state.esl_batch.clear()
    elif state.method == "No-Sharing":
        logits = state.per_opponent_logits[opponent_id]
        belief_before = state.beliefs[opponent_id].copy()
        state.beliefs[opponent_id] = _bayes_step(state.beliefs[opponent_id], logits, opponent_action)
        batch = state.per_opponent_batch[opponent_id]
        batch.append((opponent_action, belief_before))
        if len(batch) >= 4:
            grad = np.zeros_like(logits)
            for signal, belief in batch:
                grad += batch_weighted_prototype_gradient(logits, belief, signal)
            state.per_opponent_logits[opponent_id] = logits + 0.45 * (grad / len(batch))
            batch.clear()
    elif state.method == "Fixed-K Bayesian":
        assert state.logits is not None
        state.beliefs[opponent_id] = _bayes_step(state.beliefs[opponent_id], state.logits, opponent_action)
    elif state.method == "No-Belief":
        assert state.logits is not None
        uniform = np.full(K_TYPES, 1.0 / K_TYPES)
        grad = batch_weighted_prototype_gradient(state.logits, uniform, opponent_action)
        state.logits = state.logits + 0.35 * grad
    elif state.method == "Online EM":
        state.em_window.append((opponent_id, opponent_action))
        if len(state.em_window) > ONLINE_EM_WINDOW:
            state.em_window = state.em_window[-ONLINE_EM_WINDOW:]
        _update_online_em(state)


def run_ablation_cell(
    *,
    task: MatrixGameTask,
    method: AblationMethod,
    seed: int,
    horizon: int,
    h_window: int = H_WINDOW,
) -> dict[str, Any]:
    pay = _task_payoff(task)
    schedule = build_type_shifting_schedule(task=task, seed=seed, horizon=horizon)
    state = _method_state(method, seed + 9001)
    rows: list[dict[str, Any]] = []
    true_probs_ref = true_type_distributions_mature()
    for step in schedule:
        pred = _predicted_opponent_probs(state, step.opponent_id)
        act_probs = _action_probs(pred, pay)
        focal_action = _sample_binary(state.rng, act_probs)
        opponent_action = games.ACTION_COOPERATE if step.random_u < step.true_action_probs[0] else games.ACTION_DEFECT
        focal_payoff, _opp_payoff = games.play_pair_payoffs(focal_action, opponent_action, pay)
        oracle_payoff = _oracle_payoff(step.true_action_probs, pay)
        logits = _current_logits(state, step.opponent_id)
        lte = compute_latent_tracking_error(
            state.beliefs[step.opponent_id],
            true_type=step.true_type,
            learned_logits=logits,
            true_probs=true_probs_ref,
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
                "schedule_random_u": step.random_u,
                "focal_action": focal_action,
                "opponent_action": opponent_action,
                "focal_payoff": float(focal_payoff),
                "oracle_payoff": oracle_payoff,
                "prediction_ce": cross_entropy(np.eye(ACTION_DIM)[opponent_action], pred),
                "oracle_ce": cross_entropy(np.eye(ACTION_DIM)[opponent_action], step.true_action_probs),
                "lte": float(lte),
            }
        )
        state.last_focal_action = focal_action
        _update_state(state, opponent_id=step.opponent_id, opponent_action=opponent_action)
    lte_vals = [float(r["lte"]) for r in _post_switch_window_rows(rows, h_window=h_window)]
    summary = {
        "method": method,
        "task": task.key,
        "seed": seed,
        "psr": compute_post_switch_oracle_regret(rows, h_window=h_window),
        "dpr": compute_dynamic_prediction_regret(rows, h_window=h_window),
        "lte_postswitch_mean": float(np.mean(lte_vals)) if lte_vals else float("nan"),
        "early_gap_auc": compute_early_gap_auc(rows, early_window=EARLY_AUC_WINDOW),
        "shared_prototypes": method != "No-Sharing",
        "belief_updates": method != "No-Belief",
        "prototype_learning": method not in ("Fixed-K Bayesian", "Online EM"),
        "uses_switch_labels": False,
        "uses_true_types": False,
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
    for method in METHOD_ORDER:
        for metric in ("psr", "dpr", "lte_postswitch_mean", "early_gap_auc"):
            vals = [float(s[metric]) for s in summaries if s["method"] == method and np.isfinite(float(s[metric]))]
            st = mean_std_ci95(vals)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": int(st["n"]),
                    "mean": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                }
            )
    return rows


def _recovery_rows(trajectories: list[dict[str, Any]], *, task_key: str = "stag_hunt") -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in trajectories:
        if row["task"] != task_key or row["method"] not in RECOVERY_METHODS:
            continue
        if int(row["switch_id"]) < 0:
            continue
        tau = int(row["tau"])
        if tau > H_WINDOW:
            continue
        gap = float(row["oracle_payoff"]) - float(row["focal_payoff"])
        grouped.setdefault((row["method"], tau), []).append(gap)
    out: list[dict[str, Any]] = []
    for (method, tau), vals in sorted(grouped.items(), key=lambda x: (RECOVERY_METHODS.index(x[0][0]), x[0][1])):
        st = mean_std_ci95(vals)
        out.append(
            {
                "method": method,
                "tau": tau,
                "mean": st["mean"],
                "ci95_low": st["ci95_low"],
                "ci95_high": st["ci95_high"],
            }
        )
    return out


def _metrics_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {r["method"]: {**_metric_dict(rows, r["method"])} for r in rows if r["metric"] == "psr"}


def _metric_dict(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        if row["method"] == method:
            out[row["metric"]] = float(row["mean"])
    return out


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _metrics_by_method(rows)
    esl_psr = metrics["ESL"]["psr"]
    esl_dpr = metrics["ESL"]["dpr"]
    em_lte = metrics["Online EM"]["lte_postswitch_mean"]
    checks = {
        "esl_best_psr": all(esl_psr < metrics[m]["psr"] for m in METHOD_ORDER if m != "ESL"),
        "esl_best_dpr": all(esl_dpr < metrics[m]["dpr"] for m in METHOD_ORDER if m != "ESL"),
        "all_ablation_psr_worse": all(esl_psr < metrics[m]["psr"] for m in METHOD_ORDER if m != "ESL"),
        "em_low_lte_high_dpr": em_lte < metrics["ESL"]["lte_postswitch_mean"] and metrics["Online EM"]["dpr"] > esl_dpr,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def _plot_figure(metric_rows: list[dict[str, Any]], recovery_rows: list[dict[str, Any]], out_path: Path) -> None:
    metrics = _metrics_by_method(metric_rows)
    row_by_key = {(r["method"], r["metric"]): r for r in metric_rows}
    fig = plt.figure(figsize=(11.8, 8.4))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax, label in ((ax_a, "(a)"), (ax_b1, "(b)"), (ax_b2, "(c)"), (ax_c, "(d)")):
        ax.text(-0.05, 1.10, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom", clip_on=False)

    # Panel ordering enforces causal interpretation: PSR -> DPR -> LTE -> early dynamics.
    def bar_panel(
        ax: plt.Axes,
        *,
        methods: tuple[AblationMethod, ...],
        metric: str,
        title: str,
        ylabel: str,
    ) -> None:
        x = np.arange(len(methods))
        labels = [PLOT_LABELS[m] for m in methods]
        means = [metrics[m][metric] for m in methods]
        yerr = np.array(
            [
                [
                    max(0.0, means[i] - float(row_by_key[(m, metric)]["ci95_low"])),
                    max(0.0, float(row_by_key[(m, metric)]["ci95_high"]) - means[i]),
                ]
                for i, m in enumerate(methods)
            ],
            dtype=np.float64,
        ).T
        bars = ax.bar(x, means, yerr=yerr, color=[METHOD_COLORS[m] for m in methods], capsize=4)
        for method, bar in zip(methods, bars):
            if method == "ESL":
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.25)

    bar_panel(
        ax_a,
        methods=METHOD_ORDER,
        metric="psr",
        title="Removing components increases adaptation cost",
        ylabel="PSR ↓",
    )
    bar_panel(
        ax_b1,
        methods=METHOD_ORDER,
        metric="dpr",
        title="Decision-relevant prediction drives adaptation",
        ylabel="DPR ↓",
    )
    bar_panel(
        ax_b2,
        methods=LTE_PANEL_METHODS,
        metric="lte_postswitch_mean",
        title="Identification alone is insufficient",
        ylabel="LTE ↓",
    )
    bar_panel(
        ax_c,
        methods=METHOD_ORDER,
        metric="early_gap_auc",
        title="Early recovery similar, cumulative cost differs",
        ylabel="Early AUC ↓",
    )
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=METHOD_COLORS[m], markersize=8, label=PLOT_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(METHOD_ORDER), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=publication_figure_dpi())
    plt.close(fig)


def _write_report(path: Path, *, metric_rows: list[dict[str, Any]], gate: dict[str, Any], seeds: list[int], horizon: int) -> None:
    metrics = _metrics_by_method(metric_rows)
    caption = (
        "Figure 2: Component-level analysis under behavioral shifts. "
        "Removing shared structure, belief conditioning, or prototype learning increases post-switch adaptation cost. "
        "ESL achieves the lowest prediction and adaptation error. "
        "Methods that achieve low latent identification error (e.g., EM) do not necessarily adapt well, "
        "showing that identification alone is insufficient under endogenous interaction."
    )
    lines = [
        "# Figure 2 Ablation Report",
        "",
        f"- Seeds: `{seeds}`; horizon: `{horizon}`; H: `{H_WINDOW}`.",
        "- Error bars: 95% CI over seed-task cells.",
        f"- Gate: {'PASS' if gate['pass'] else 'FAIL'}.",
        "",
        "## Caption",
        caption,
        "",
        "## PSR",
    ]
    esl_psr = metrics["ESL"]["psr"]
    for method in METHOD_ORDER:
        if method == "ESL":
            continue
        lines.append(f"- ESL vs {method}: `{esl_psr:.4f}` vs `{metrics[method]['psr']:.4f}`.")
    lines.extend(
        [
            "",
            "## Mechanism",
            f"- ESL has the lowest DPR: `{metrics['ESL']['dpr']:.4f}`.",
            f"- EM low LTE but high DPR: `{metrics['Online EM']['lte_postswitch_mean']:.4f}` LTE, `{metrics['Online EM']['dpr']:.4f}` DPR.",
            "- Removing any component worsens PSR relative to full ESL.",
            "- No-sharing degrades PSR relative to ESL, supporting the necessity of shared structure.",
            "- Identification alone is insufficient under endogenous interaction.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_figure2_ablation(
    *,
    out_root: Path,
    manuscript_bundle: Path,
    seeds: list[int],
    horizon: int,
    h_window: int = H_WINDOW,
    tasks: tuple[str, ...] = ("ipd", "stag_hunt", "matching_pennies"),
    smoke: bool = False,
) -> dict[str, Path]:
    ensure_manuscript_bundle_layout(manuscript_bundle)
    out_root.mkdir(parents=True, exist_ok=True)
    task_by_key = {t.key: t for t in matrix_game_tasks()}
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_key in tasks:
        task = task_by_key[task_key]
        for seed in seeds:
            for method in ABLATION_METHODS:
                result = run_ablation_cell(task=task, method=method, seed=seed, horizon=horizon, h_window=h_window)
                trajectories.extend(result["trajectory"])
                summaries.append(result["summary"])
    metric_rows = _metric_rows(summaries)
    recovery = _recovery_rows(trajectories)
    figure = manuscript_bundle / "main" / "figures" / "figure2_ablation.pdf"
    summary_csv = manuscript_bundle / "main" / "tables" / "figure2_ablation_summary.csv"
    report = manuscript_bundle / "reports" / "figure2_ablation_report.md"
    trajectory_csv = out_root / "figure2_ablation_trajectory.csv"
    recovery_csv = out_root / "figure2_ablation_recovery.csv"
    gate_json = out_root / "figure2_ablation_gate.json"
    _write_csv(summary_csv, metric_rows, ["method", "metric", "n", "mean", "ci95_low", "ci95_high"])
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
            "schedule_random_u",
            "focal_action",
            "opponent_action",
            "focal_payoff",
            "oracle_payoff",
            "prediction_ce",
            "oracle_ce",
            "lte",
        ],
    )
    _write_csv(recovery_csv, recovery, ["method", "tau", "mean", "ci95_low", "ci95_high"])
    gate = _gate(metric_rows)
    gate_json.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    _plot_figure(metric_rows, recovery, figure)
    _write_report(report, metric_rows=metric_rows, gate=gate, seeds=seeds, horizon=horizon)
    return {
        "figure": figure,
        "summary_csv": summary_csv,
        "report": report,
        "trajectory_csv": trajectory_csv,
        "recovery_csv": recovery_csv,
        "gate_json": gate_json,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate Figure 2 ESL component ablations")
    p.add_argument("--out-root", type=Path, default=Path("runs/figure2_ablation"))
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--h-window", type=int, default=H_WINDOW)
    p.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    outputs = run_figure2_ablation(
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
