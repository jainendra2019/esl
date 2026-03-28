"""Experiment configuration and deterministic seeding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

ExperimentMode = Literal["recovery", "adaptation"]

# Paper / debug: fixed (observer, target) and explicit θ (list-of-lists JSON-serializable).
ForceOrderedPair = tuple[int, int] | None
PrototypeLogitsOverride = list[list[float]] | None


@dataclass
class ESLConfig:
    """All experiment parameters in one place."""

    seed: int = 42
    mode: ExperimentMode = "recovery"
    # Baseline / ablation: no belief updates, no batching, no prototype SGD (uniform b stays; θ fixed).
    learning_frozen: bool = False
    # Beliefs update and batch accumulates, but prototype logits never apply SGD (slow scale off).
    freeze_prototype_parameters: bool = False
    # If set, every round uses this ordered pair (i observes j) instead of sampling.
    force_ordered_pair: ForceOrderedPair = None
    # If set, skip random prototype init and use these logits (shape K × |A|).
    prototype_logits_override: PrototypeLogitsOverride = None
    # If set, length must equal num_agents; overrides cyclic true-type assignment (debug / hand traces).
    force_agent_true_types: list[int] | None = None

    num_agents: int = 4
    num_prototypes: int = 2
    num_actions: int = 2

    delta_simplex: float = 1e-4
    # Small additive term in the Bayes posterior denominator for numerical stability only.
    bayes_denominator_eps: float = 1e-12
    # §4.6: clamp softmax mass before log(p) in log-likelihood telemetry
    log_prob_min: float = 1e-8

    base_init: float = 0.0
    init_noise: float = 0.01
    symmetric_init: bool = False

    num_rounds: int = 200
    # When True, stop early once convergence criteria hold for the current round (see trainer);
    # num_rounds is a hard cap T_max. Default False preserves fixed-horizon behavior.
    stop_on_convergence: bool = False
    # Rolling window length W (rounds). Criteria need t+1 >= W before any check.
    convergence_window_w: int = 50
    convergence_epsilon_h: float = 0.1
    # Hungarian-matched |P(C) for true type 0 − P(C) for true type 1| at current θ.
    convergence_epsilon_delta: float = 0.8
    # max over the last W rounds of logged prototype_update_norm must be < this.
    convergence_epsilon_theta: float = 0.01
    # max over the last W rounds of belief_change_norm must be < this.
    convergence_epsilon_b: float = 0.01
    # Slow timescale: prototype SGD every M env steps (PRD §7; default M=5).
    prototype_update_every: int = 5

    observability: Literal["full", "sparse"] = "full"
    p_obs: float = 1.0

    lr_belief_alpha_exponent: float = -0.6
    lr_prototype_gamma_exponent: float = -0.9
    prototype_lr_scale: float = 1.0

    adaptation_lambda: float = 2.0

    pd_t: float = 5.0
    pd_r: float = 3.0
    pd_p: float = 1.0
    pd_s: float = 0.0

    def validate(self) -> None:
        if self.num_actions != 2:
            raise ValueError("v1 supports exactly 2 actions")
        if not (0.0 <= self.p_obs <= 1.0):
            raise ValueError("p_obs must be in [0, 1]")
        if self.delta_simplex * self.num_prototypes > 1.0:
            raise ValueError("delta_simplex too large for K (K*delta must be <= 1)")
        if self.prototype_update_every < 1:
            raise ValueError("prototype_update_every must be >= 1")
        if self.force_ordered_pair is not None:
            io, jo = self.force_ordered_pair
            if not (0 <= io < self.num_agents and 0 <= jo < self.num_agents):
                raise ValueError("force_ordered_pair indices out of range")
            if io == jo:
                raise ValueError("force_ordered_pair requires i != j")
        if self.prototype_logits_override is not None:
            arr = np.array(self.prototype_logits_override, dtype=np.float64)
            if arr.shape != (self.num_prototypes, self.num_actions):
                raise ValueError(
                    f"prototype_logits_override must be ({self.num_prototypes}, {self.num_actions}), got {arr.shape}"
                )
        if self.force_agent_true_types is not None:
            if len(self.force_agent_true_types) != self.num_agents:
                raise ValueError("force_agent_true_types length must equal num_agents")
            for t in self.force_agent_true_types:
                if not (0 <= int(t) < self.num_prototypes):
                    raise ValueError("force_agent_true_types entries must be in [0, num_prototypes)")
        if self.stop_on_convergence:
            if self.num_prototypes < 2:
                raise ValueError("stop_on_convergence requires num_prototypes >= 2")
            if self.convergence_window_w < 1:
                raise ValueError("convergence_window_w must be >= 1")
            for name, x in (
                ("convergence_epsilon_h", self.convergence_epsilon_h),
                ("convergence_epsilon_theta", self.convergence_epsilon_theta),
                ("convergence_epsilon_b", self.convergence_epsilon_b),
            ):
                if not (x > 0.0):
                    raise ValueError(f"{name} must be > 0")
            if not (0.0 <= self.convergence_epsilon_delta < 1.0):
                raise ValueError("convergence_epsilon_delta must be in [0, 1)")

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def belief_lr(self, round_t: int) -> float:
        return float((round_t + 1) ** self.lr_belief_alpha_exponent)

    def prototype_lr(self, step_m: int) -> float:
        return self.prototype_lr_scale * float((step_m + 1) ** self.lr_prototype_gamma_exponent)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
