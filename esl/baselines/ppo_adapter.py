"""
Thin adapter for official PPO-PyTorch (PRD §8A: independent PPO baseline).

Uses only ``third_party/PPO-PyTorch/PPO.py`` — not the upstream ``train.py`` (roboschool/gym).
Trains a discrete policy in a tiny repeated Prisoner's Dilemma loop against a **fixed**
hidden opponent (recovery-style exogenous opponent), then writes canonical manifests
and summary JSON for Milestone 2A smoke only.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from esl import games
from esl.baselines.external_common import (
    BaselineAvailability,
    BaselineSpec,
    MatrixGameTask,
    OpponentRegime,
    payoff_for_task,
    source_info,
    write_canonical_baseline_run,
)
from esl.config import ESLConfig
from esl.experiment_registry import build_baseline_manifest_dict
from esl.experiments.canonical_io import write_manifest_json

OFFICIAL_PPO_REPO_URL = "https://github.com/nikhilbarhate99/PPO-PyTorch"


def third_party_ppo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "PPO-PyTorch"


def official_ppo_commit() -> str | None:
    root = third_party_ppo_root()
    if not (root / "PPO.py").is_file():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_official_ppo_module() -> Any:
    root = third_party_ppo_root()
    path = root / "PPO.py"
    if not path.is_file():
        raise FileNotFoundError(f"Official PPO not found at {path}")
    spec = importlib.util.spec_from_file_location("official_ppo_pytorch", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official PPO.py")
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def ppo_spec() -> BaselineSpec:
    root = third_party_ppo_root()
    ready = (root / "PPO.py").is_file()
    status = "READY" if ready else "NOT_INTEGRATED"
    reason = (
        "Official PPO.py is present and loaded directly by the adapter."
        if ready
        else "third_party/PPO-PyTorch/PPO.py is missing."
    )
    source = source_info(
        name="PPO-PyTorch",
        url=OFFICIAL_PPO_REPO_URL,
        local_rel="third_party/PPO-PyTorch",
        license_name="MIT",
        notes="Official PPO.py is imported directly; upstream train.py is not used.",
    )
    return BaselineSpec(
        family="independent_ppo",
        method_name="Independent PPO",
        adapter_path="esl.baselines.ppo_adapter",
        source=source,
        deviations=(
            "Upstream train.py targets Gym/Roboschool-style environments; adapter supplies an ESL repeated "
            "2-action matrix-game loop around official PPO.ActorCritic and PPO.update.",
            "PPO has no ESL prototypes or beliefs, so MCE and belief metrics are undefined.",
        ),
        availability=BaselineAvailability(
            status=status,
            reason=reason,
            can_run_repeated_2x2=ready,
        ),
    )


class PPOAdapter:
    """Common interface wrapper around the official PPO.py implementation."""

    def __init__(self) -> None:
        self.spec = ppo_spec()

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
        import torch

        ppo_mod = _load_official_ppo_module()
        PPO = ppo_mod.PPO
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        pay = payoff_for_task(task)
        ppo_agent = PPO(
            2,
            2,
            3e-4,
            1e-3,
            0.99,
            2 if smoke else 4,
            0.2,
            False,
            action_std_init=0.6,
        )
        rows: list[dict[str, Any]] = []
        cum_social = 0.0
        cum_row = 0.0
        last_opp = 0
        row_coop = 0
        for t in range(horizon):
            state = np.array([float(t) / max(horizon, 1), float(last_opp)], dtype=np.float32)
            action = int(ppo_agent.select_action(state))
            if regime.key == "fixed_types":
                opp = games.ACTION_COOPERATE if t % 2 == 0 else games.ACTION_DEFECT
            elif regime.key == "adaptive_agents":
                opp = games.ACTION_DEFECT if row_coop / max(t, 1) > 0.45 else games.ACTION_COOPERATE
            else:
                opp = int(rng.binomial(1, 0.5 if task.key == "matching_pennies" else 0.65))
            r_i, r_j = games.play_pair_payoffs(action, opp, pay)
            ppo_agent.buffer.rewards.append(float(r_i))
            ppo_agent.buffer.is_terminals.append(t == horizon - 1)
            if (t + 1) % max(4, min(16, horizon)) == 0:
                ppo_agent.update()
            row_coop += int(action == games.ACTION_COOPERATE)
            last_opp = int(opp)
            cum_social += float(r_i + r_j)
            cum_row += float(r_i)
            rows.append(
                {
                    "round": t,
                    "timestep": t + 1,
                    "task": task.key,
                    "regime": regime.key,
                    "method": self.spec.method_name,
                    "row_action": action,
                    "opponent_action": opp,
                    "row_reward": float(r_i),
                    "opponent_reward": float(r_j),
                    "mean_payoff_per_agent": float(cum_row / (t + 1)),
                }
            )
        summary = {
            "mean_payoff_per_agent_per_round": float(cum_row / max(horizon, 1)),
            "cumulative_social_payoff": float(cum_social),
            "timesteps": int(horizon),
            "final_mce": None,
            "final_matched_cross_entropy": None,
            "final_belief_entropy": None,
            "final_belief_argmax_accuracy": None,
        }
        return write_canonical_baseline_run(
            out_dir=out_dir,
            spec=self.spec,
            seed=seed,
            task=task,
            regime=regime,
            horizon=horizon,
            trajectory_rows=rows,
            summary_values=summary,
            experiment_id="baseline.ppo.performance_smoke",
            smoke=smoke,
        )

    def evaluate(self, run_dir: Path) -> dict[str, Any]:
        return json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))

    def export_summary(self, run_dir: Path) -> dict[str, Any]:
        return self.evaluate(run_dir)

    def export_predictions(self, run_dir: Path) -> Path | None:
        return None


def build_adapter() -> PPOAdapter:
    return PPOAdapter()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_recovery_mode_smoke(
    out_dir: Path,
    *,
    seed: int,
    esl_run_dir: str,
    experiment_id: str = "baseline.ppo.recovery_smoke",
    opponent_action: int = games.ACTION_DEFECT,
    max_training_timesteps: int = 96,
    update_timestep: int = 32,
    max_ep_len: int = 12,
    K_epochs: int = 4,
) -> Path:
    """
    One tiny PPO smoke run: discrete actions, fixed opponent, PD payoffs from ``ESLConfig``.

    Writes under ``out_dir``: ``config.json``, ``manifest.json``, ``summary_metrics.json``,
    ``metrics_trajectory.csv``, ``provenance.json``.
    """
    import torch

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_mod = _load_official_ppo_module()
    PPO = ppo_mod.PPO

    cfg_pd = ESLConfig(seed=seed, num_rounds=1, num_agents=2, num_prototypes=2)
    cfg_pd.validate()
    pay = games.prisoners_dilemma(cfg_pd)

    state_dim = 2
    action_dim = 2
    has_continuous_action_space = False
    lr_actor = 3e-4
    lr_critic = 1e-3
    gamma = 0.99
    eps_clip = 0.2

    torch.manual_seed(seed)
    np.random.seed(seed)

    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
    )

    t0 = time.perf_counter()
    time_step = 0
    i_episode = 0
    traj_rows: list[dict[str, Any]] = []
    cum_social = 0.0
    n_steps = 0

    while time_step < max_training_timesteps:
        state = np.array([1.0, float(opponent_action)], dtype=np.float32)
        current_ep_reward = 0.0

        for t in range(1, max_ep_len + 1):
            action = int(ppo_agent.select_action(state))
            opp = opponent_action
            r_row, r_col = games.play_pair_payoffs(action, opp, pay)
            reward = float(r_row)
            cum_social += float(r_row + r_col)
            n_steps += 1

            done = t == max_ep_len
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step > 0 and time_step % update_timestep == 0:
                ppo_agent.update()

            traj_rows.append(
                {
                    "round": i_episode,
                    "timestep": time_step,
                    "ppo_step_reward": reward,
                    "opponent_action": opp,
                    "row_action": action,
                }
            )

            if time_step >= max_training_timesteps:
                break

        i_episode += 1
        if time_step >= max_training_timesteps:
            break

    wall = time.perf_counter() - t0
    mean_per_step = cum_social / max(2 * n_steps, 1)

    run_cfg = {
        "adapter": "esl.baselines.ppo_adapter",
        "official_repo": OFFICIAL_PPO_REPO_URL,
        "seed": seed,
        "opponent_action": int(opponent_action),
        "max_training_timesteps": max_training_timesteps,
        "update_timestep": update_timestep,
        "max_ep_len": max_ep_len,
        "K_epochs": K_epochs,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "pd_t": cfg_pd.pd_t,
        "pd_r": cfg_pd.pd_r,
        "pd_p": cfg_pd.pd_p,
        "pd_s": cfg_pd.pd_s,
    }
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    commit = official_ppo_commit() or "unknown"
    provenance = {
        "sources": [
            {
                "name": "PPO-PyTorch",
                "url": OFFICIAL_PPO_REPO_URL,
                "commit": commit,
                "local_path": "third_party/PPO-PyTorch",
                "entry_module": "PPO.py",
            }
        ],
        "notes": "Adapter does not use upstream train.py (roboschool). Only PPO.ActorCritic + PPO.update.",
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    deviations = [
        "Upstream train.py targets Roboschool; adapter runs a custom repeated PD vs fixed opponent.",
        "No ESL beliefs or prototypes; MCE is not defined for this baseline.",
    ]
    manifest = build_baseline_manifest_dict(
        experiment_id=experiment_id,
        seed=seed,
        esl_run_dir=esl_run_dir,
        adapter="third_party.PPO-PyTorch",
        deviations=deviations,
        provenance=provenance["sources"][0],
        extra={"wall_time_sec": wall, "timesteps": time_step},
    )
    write_manifest_json(out_dir / "manifest.json", manifest)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "baseline",
        "baseline_adapter": "third_party.PPO-PyTorch",
        "baseline_family": "independent_ppo",
        "seed": seed,
        "wall_time_sec": float(wall),
        "timesteps": int(time_step),
        "episodes": int(i_episode),
        "mean_payoff_per_agent_per_round": float(mean_per_step),
        "cumulative_social_payoff": float(cum_social),
        "final_mce": None,
        "final_matched_cross_entropy": None,
        "final_belief_entropy": None,
        "final_belief_argmax_accuracy": None,
        "mode": "recovery_style_fixed_opponent",
        "mce_note": "MCE is undefined: PPO does not learn ESL prototypes.",
        "official_repo_commit": commit,
    }
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    _write_csv(out_dir / "metrics_trajectory.csv", traj_rows)
    return out_dir
