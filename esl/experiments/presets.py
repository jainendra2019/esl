"""Recovery experiment presets aligned with NeurIPS robustness / ablation checklist."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from esl.config import ESLConfig

# Balanced 10×AC / 10×AD (matches tracked flagship runs).
FLAGSHIP_AGENT_TYPES: list[int] = [0] * 10 + [1] * 10


def recovery_flagship_cfg(*, seed: int = 42) -> ESLConfig:
    """Canonical long-run recovery (variable L_t in [5,15], Q=15, N=20, balanced types)."""
    c = ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=20,
        num_prototypes=2,
        num_actions=2,
        num_rounds=10_000,
        force_agent_true_types=list(FLAGSHIP_AGENT_TYPES),
        delta_simplex=0.02,
        bayes_denominator_eps=1e-12,
        base_init=0.0,
        init_noise=0.05,
        symmetric_init=False,
        prototype_lr_scale=22.0,
        lr_prototype_gamma_exponent=-0.9,
        prototype_update_every=15,
        prototype_l2_eta=0.0,
        interaction_pairs_min=5,
        interaction_pairs_max=15,
        interaction_pairs_law="uniform",
        observability="full",
        p_obs=1.0,
        log_beliefs_tensor=False,
        log_beliefs_every_interaction=False,
        freeze_prototype_parameters=False,
        learning_frozen=False,
    )
    c.validate()
    return c


def recovery_sparse_obs_cfg(
    *,
    p_obs: Literal[1.0, 0.5, 0.2] | float,
    seed: int = 42,
) -> ESLConfig:
    """Flagship geometry with sparse observability at given p_obs."""
    po = float(p_obs)
    obs: Literal["full", "sparse"] = "full" if po >= 1.0 else "sparse"
    c = replace(recovery_flagship_cfg(seed=seed), observability=obs, p_obs=min(1.0, max(0.0, po)))
    c.validate()
    return c


def recovery_short_horizon_cfg(
    *,
    interaction_budget: int,
    interactions_per_round: int = 10,
    seed: int = 42,
) -> ESLConfig:
    """
    Fixed L_t = interactions_per_round each round so total interactions == budget exactly
    (num_rounds = budget // interactions_per_round).
    """
    if interaction_budget % interactions_per_round != 0:
        raise ValueError("interaction_budget must be divisible by interactions_per_round")
    rounds = interaction_budget // interactions_per_round
    max_pairs = 20 * 19
    if not (1 <= interactions_per_round <= max_pairs):
        raise ValueError("interactions_per_round out of range for N=20")
    c = replace(
        recovery_flagship_cfg(seed=seed),
        num_rounds=rounds,
        interaction_pairs_min=interactions_per_round,
        interaction_pairs_max=interactions_per_round,
    )
    c.validate()
    return c


def recovery_lr_sweep_cfg(*, prototype_lr_scale: float, seed: int = 42) -> ESLConfig:
    c = replace(recovery_flagship_cfg(seed=seed), prototype_lr_scale=float(prototype_lr_scale))
    c.validate()
    return c


def recovery_init_noise_sweep_cfg(*, init_noise: float, seed: int = 42) -> ESLConfig:
    c = replace(recovery_flagship_cfg(seed=seed), init_noise=float(init_noise))
    c.validate()
    return c


def recovery_Q_sweep_cfg(*, Q: int, seed: int = 42) -> ESLConfig:
    c = replace(recovery_flagship_cfg(seed=seed), prototype_update_every=int(Q))
    c.validate()
    return c


def recovery_failure_case_cfg(*, seed: int = 42) -> ESLConfig:
    """
    Weak / incomplete recovery contrast: symmetric θ init, tiny noise, lower LR, shorter horizon.
    """
    c = replace(
        recovery_flagship_cfg(seed=seed),
        num_rounds=2500,
        symmetric_init=True,
        init_noise=0.01,
        prototype_lr_scale=8.0,
        prototype_logits_override=None,
    )
    c.validate()
    return c


def recovery_fixed_prototype_baseline_cfg(*, seed: int = 42) -> ESLConfig:
    """Same flagship data geometry; beliefs update but prototype SGD is off."""
    c = replace(recovery_flagship_cfg(seed=seed), freeze_prototype_parameters=True)
    c.validate()
    return c


PRESET_BUILDERS: dict[str, object] = {
    "recovery_flagship": recovery_flagship_cfg,
    "recovery_fixed_prototype_baseline": recovery_fixed_prototype_baseline_cfg,
    "recovery_failure_case": recovery_failure_case_cfg,
}
