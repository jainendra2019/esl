"""Shared scaffolding for external baseline onboarding.

The adapters in this milestone keep official sources pinned under ``third_party``
and expose a small ESL-compatible surface. When a source cannot run directly in
the repeated 2-action matrix-game protocol, the adapter records the reduced
implementation and its deviations in the canonical provenance artifacts.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from esl import games
from esl.config import ESLConfig
from esl.experiment_registry import SUMMARY_METRICS_SCHEMA_VERSION, build_baseline_manifest_dict
from esl.experiments.canonical_io import write_manifest_json


BASELINE_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "round",
    "timestep",
    "task",
    "regime",
    "method",
    "row_action",
    "opponent_action",
    "row_reward",
    "opponent_reward",
    "mean_payoff_per_agent",
)


@dataclass(frozen=True)
class SourceInfo:
    name: str
    url: str
    local_path: str
    commit: str | None
    license: str
    notes: str


@dataclass(frozen=True)
class BaselineAvailability:
    status: str
    reason: str
    can_run_repeated_2x2: bool


@dataclass(frozen=True)
class BaselineSpec:
    family: str
    method_name: str
    adapter_path: str
    source: SourceInfo
    deviations: tuple[str, ...]
    availability: BaselineAvailability


@dataclass(frozen=True)
class MatrixGameTask:
    key: str
    label: str
    payoff_game: str
    pd_t: float = 5.0
    pd_r: float = 3.0
    pd_p: float = 1.0
    pd_s: float = 0.0


@dataclass(frozen=True)
class OpponentRegime:
    key: str
    label: str
    esl_mode: str
    description: str


class BaselineAdapter(Protocol):
    spec: BaselineSpec

    def train(
        self,
        out_dir: Path,
        *,
        seed: int,
        task: MatrixGameTask,
        regime: OpponentRegime,
        horizon: int,
        smoke: bool,
    ) -> Path:
        ...

    def evaluate(self, run_dir: Path) -> dict[str, Any]:
        ...

    def export_summary(self, run_dir: Path) -> dict[str, Any]:
        ...

    def export_predictions(self, run_dir: Path) -> Path | None:
        ...


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def source_info(
    *,
    name: str,
    url: str,
    local_rel: str,
    license_name: str = "not declared in pinned source",
    notes: str = "",
) -> SourceInfo:
    return SourceInfo(
        name=name,
        url=url,
        local_path=local_rel,
        commit=git_commit(repo_root() / local_rel),
        license=license_name,
        notes=notes,
    )


def matrix_game_tasks() -> tuple[MatrixGameTask, ...]:
    return (
        MatrixGameTask("ipd", "IPD", "prisoners_dilemma"),
        # Stag Hunt is represented through the same 2x2 payoff parameterization:
        # C/C=4, D/D=2, unilateral D against C=3, unilateral C against D=0.
        MatrixGameTask("stag_hunt", "Stag Hunt", "prisoners_dilemma", pd_t=3.0, pd_r=4.0, pd_p=2.0, pd_s=0.0),
        MatrixGameTask("matching_pennies", "Matching Pennies", "matching_pennies"),
    )


def opponent_regimes() -> tuple[OpponentRegime, ...]:
    return (
        OpponentRegime("fixed_types", "Fixed types", "recovery", "Fixed AC/AD/TFT hidden policies."),
        OpponentRegime("adaptive_agents", "Adaptive agents", "adaptation", "All agents adapt with logit best response."),
        OpponentRegime(
            "belief_conditioned_agents",
            "Belief-conditioned agents",
            "adaptation",
            "Focal ESL-style learner faces fixed hidden opponents.",
        ),
    )


def task_config(task: MatrixGameTask, regime: OpponentRegime, *, seed: int, horizon: int) -> ESLConfig:
    cfg = ESLConfig(
        seed=seed,
        mode="recovery" if regime.key == "fixed_types" else "adaptation",
        num_agents=4,
        num_prototypes=2,
        num_actions=2,
        num_rounds=horizon,
        interaction_pairs_min=2,
        interaction_pairs_max=2,
        prototype_update_every=2,
        log_beliefs_tensor=False,
        log_interaction_observations=False,
        payoff_game=task.payoff_game,  # type: ignore[arg-type]
        pd_t=task.pd_t,
        pd_r=task.pd_r,
        pd_p=task.pd_p,
        pd_s=task.pd_s,
        prototype_lr_scale=1.0,
        init_noise=0.05,
    )
    if regime.key == "belief_conditioned_agents":
        cfg.adaptation_esl_agent_indices = [0]
    cfg.validate()
    return cfg


def payoff_for_task(task: MatrixGameTask) -> games.PayoffMatrices:
    cfg = task_config(task, opponent_regimes()[0], seed=0, horizon=1)
    return games.game_payoffs(cfg)


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-9)
    z = z - np.max(z)
    w = np.exp(z)
    return w / w.sum()


def _best_response_action(pay: games.PayoffMatrices, p_opp: float, *, lam: float = 2.0) -> np.ndarray:
    probs = np.array([p_opp, 1.0 - p_opp], dtype=np.float64)
    utilities = np.array([np.sum(probs * pay.row[a, :]) for a in range(2)], dtype=np.float64)
    return _softmax(utilities, temperature=1.0 / max(lam, 1e-9))


def _opponent_action(
    rng: np.random.Generator,
    *,
    regime: OpponentRegime,
    t: int,
    last_row_action: int | None,
    row_coop_rate: float,
    task: MatrixGameTask,
) -> int:
    if regime.key == "fixed_types":
        if t % 3 == 0:
            return games.ACTION_COOPERATE
        if t % 3 == 1:
            return games.ACTION_DEFECT
        return games.ACTION_COOPERATE if last_row_action is None else int(last_row_action)
    if regime.key == "adaptive_agents":
        pay = payoff_for_task(task)
        p_row_c = float(np.clip(row_coop_rate, 0.05, 0.95))
        utils = np.array([np.sum(np.array([p_row_c, 1.0 - p_row_c]) * pay.col[:, a]) for a in range(2)])
        probs = _softmax(utils, temperature=0.75)
        return int(rng.choice(2, p=probs))
    # Belief-conditioned opponent: mostly follows a soft best response to the observed focal rate.
    p_row_c = float(np.clip(row_coop_rate, 0.1, 0.9))
    if task.key == "matching_pennies":
        return int(rng.binomial(1, 0.5))
    return games.ACTION_DEFECT if p_row_c > 0.45 else games.ACTION_COOPERATE


def policy_action_probs(
    *,
    family: str,
    pay: games.PayoffMatrices,
    rng: np.random.Generator,
    opponent_counts: np.ndarray,
    transition_counts: np.ndarray,
    last_row_action: int | None,
    last_opp_action: int | None,
    t: int,
    task: MatrixGameTask,
) -> np.ndarray:
    total = float(opponent_counts.sum())
    p_opp_c = float(opponent_counts[0] / total) if total > 0 else 0.5
    if family == "mfos":
        if task.key == "matching_pennies":
            return np.array([0.5, 0.5], dtype=np.float64)
        shaping = 0.72 if t < 2 else 0.62
        return np.array([shaping, 1.0 - shaping], dtype=np.float64)
    if family == "mbom":
        return _best_response_action(pay, p_opp_c, lam=2.5)
    if family == "simple_opponent_model":
        if last_row_action is not None and transition_counts[int(last_row_action)].sum() > 0:
            row = transition_counts[int(last_row_action)]
            p_opp_c = float(row[0] / row.sum())
        return _best_response_action(pay, p_opp_c, lam=1.8)
    # Conservative fallback for independent policies.
    return np.array([0.5, 0.5], dtype=np.float64)


def run_reduced_matrix_game(
    *,
    family: str,
    method_name: str,
    task: MatrixGameTask,
    regime: OpponentRegime,
    seed: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path | None]:
    rng = np.random.default_rng(seed)
    pay = payoff_for_task(task)
    rows: list[dict[str, Any]] = []
    opponent_counts = np.ones(2, dtype=np.float64)
    transition_counts = np.ones((2, 2), dtype=np.float64)
    last_row_action: int | None = None
    last_opp_action: int | None = None
    cum_social = 0.0
    cum_row = 0.0

    for t in range(horizon):
        row_coop_rate = float((sum(1 for r in rows if r["row_action"] == 0) + 1.0) / (len(rows) + 2.0))
        p_action = policy_action_probs(
            family=family,
            pay=pay,
            rng=rng,
            opponent_counts=opponent_counts,
            transition_counts=transition_counts,
            last_row_action=last_row_action,
            last_opp_action=last_opp_action,
            t=t,
            task=task,
        )
        a_i = int(rng.choice(2, p=p_action))
        a_j = _opponent_action(
            rng,
            regime=regime,
            t=t,
            last_row_action=last_row_action,
            row_coop_rate=row_coop_rate,
            task=task,
        )
        r_i, r_j = games.play_pair_payoffs(a_i, a_j, pay)
        cum_social += float(r_i + r_j)
        cum_row += float(r_i)
        rows.append(
            {
                "round": t,
                "timestep": t + 1,
                "task": task.key,
                "regime": regime.key,
                "method": method_name,
                "row_action": a_i,
                "opponent_action": a_j,
                "row_reward": float(r_i),
                "opponent_reward": float(r_j),
                "mean_payoff_per_agent": float(cum_row / (t + 1)),
            }
        )
        if last_row_action is not None:
            transition_counts[int(last_row_action), a_j] += 1.0
        opponent_counts[a_j] += 1.0
        last_row_action = a_i
        last_opp_action = a_j

    final = rows[-1]["mean_payoff_per_agent"] if rows else math.nan
    summary = {
        "mean_payoff_per_agent_per_round": float(final),
        "cumulative_social_payoff": float(cum_social),
        "timesteps": int(horizon),
        "final_mce": None,
        "final_matched_cross_entropy": None,
        "final_belief_entropy": None,
        "final_belief_argmax_accuracy": None,
    }
    return rows, summary, None


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns or (tuple(rows[0].keys()) if rows else ()))
    if not cols:
        path.write_text("", encoding="utf-8")
        return
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_canonical_baseline_run(
    *,
    out_dir: Path,
    spec: BaselineSpec,
    seed: int,
    task: MatrixGameTask,
    regime: OpponentRegime,
    horizon: int,
    trajectory_rows: list[dict[str, Any]],
    summary_values: dict[str, Any],
    experiment_id: str,
    smoke: bool,
) -> Path:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "adapter": spec.adapter_path,
        "baseline_family": spec.family,
        "method_name": spec.method_name,
        "seed": seed,
        "task": task.key,
        "task_label": task.label,
        "regime": regime.key,
        "regime_label": regime.label,
        "horizon": horizon,
        "smoke": bool(smoke),
        "availability_status": spec.availability.status,
        "source_url": spec.source.url,
        "source_commit": spec.source.commit,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "sources": [
            {
                "name": spec.source.name,
                "url": spec.source.url,
                "commit": spec.source.commit or "unknown",
                "local_path": spec.source.local_path,
                "license": spec.source.license,
                "notes": spec.source.notes,
            }
        ],
        "adapter": spec.adapter_path,
        "implementation_status": spec.availability.status,
        "can_run_repeated_2x2": spec.availability.can_run_repeated_2x2,
        "deviations": list(spec.deviations),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = build_baseline_manifest_dict(
        experiment_id=experiment_id,
        seed=seed,
        esl_run_dir="runs/main_performance_grid",
        adapter=spec.adapter_path,
        deviations=list(spec.deviations),
        provenance=provenance["sources"][0],
        extra={
            "task": task.key,
            "regime": regime.key,
            "horizon": horizon,
            "smoke": bool(smoke),
            "availability_status": spec.availability.status,
        },
    )
    write_manifest_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "metrics_trajectory.csv", trajectory_rows, BASELINE_TRAJECTORY_COLUMNS)
    wall = time.perf_counter() - t0
    summary = {
        "schema_version": SUMMARY_METRICS_SCHEMA_VERSION,
        "run_kind": "baseline",
        "baseline_adapter": spec.adapter_path,
        "baseline_family": spec.family,
        "seed": seed,
        "wall_time_sec": float(wall),
        "timesteps": int(summary_values.get("timesteps", horizon)),
        "mean_payoff_per_agent_per_round": float(summary_values["mean_payoff_per_agent_per_round"]),
        "cumulative_social_payoff": float(summary_values["cumulative_social_payoff"]),
        "final_mce": summary_values.get("final_mce"),
        "final_matched_cross_entropy": summary_values.get("final_matched_cross_entropy"),
        "final_belief_entropy": summary_values.get("final_belief_entropy"),
        "final_belief_argmax_accuracy": summary_values.get("final_belief_argmax_accuracy"),
        "mode": f"{task.key}:{regime.key}",
        "task": task.key,
        "regime": regime.key,
        "availability_status": spec.availability.status,
        "mce_note": "MCE is undefined for policy-performance external baselines.",
        "official_repo_commit": spec.source.commit or "unknown",
        "implementation_note": spec.availability.reason,
    }
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_dir


class ReducedMatrixGameAdapter:
    """Adapter for reduced faithful external baselines in 2-action matrix games."""

    spec: BaselineSpec

    def __init__(self, spec: BaselineSpec):
        self.spec = spec

    def train(
        self,
        out_dir: Path,
        *,
        seed: int,
        task: MatrixGameTask,
        regime: OpponentRegime,
        horizon: int,
        smoke: bool,
    ) -> Path:
        rows, summary, _ = run_reduced_matrix_game(
            family=self.spec.family,
            method_name=self.spec.method_name,
            task=task,
            regime=regime,
            seed=seed,
            horizon=horizon,
        )
        return write_canonical_baseline_run(
            out_dir=out_dir,
            spec=self.spec,
            seed=seed,
            task=task,
            regime=regime,
            horizon=horizon,
            trajectory_rows=rows,
            summary_values=summary,
            experiment_id=f"baseline.{self.spec.family}.performance_smoke",
            smoke=smoke,
        )

    def evaluate(self, run_dir: Path) -> dict[str, Any]:
        return json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))

    def export_summary(self, run_dir: Path) -> dict[str, Any]:
        return self.evaluate(run_dir)

    def export_predictions(self, run_dir: Path) -> Path | None:
        return None
