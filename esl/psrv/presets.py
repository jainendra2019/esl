"""PSRV presets: Algorithm 1 recovery at K=A=3 with density-scaled populations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from esl.config import ESLConfig
from esl.experiments.presets import recovery_flagship_cfg
from esl.population_spec import PopulationSpec, apply_population_spec
from esl.res.game_spec import GameSpec


def balanced_types(n: int, k: int) -> list[int]:
    """Near-equal type counts summing to n."""
    n, k = int(n), int(k)
    base = n // k
    rem = n % k
    out: list[int] = []
    for t in range(k):
        out.extend([t] * (base + (1 if t < rem else 0)))
    return out


def recovery_psrv_cfg(
    *,
    seed: int,
    spec: GameSpec,
    population: PopulationSpec,
    num_rounds: int = 10_000,
) -> ESLConfig:
    """
    Algorithm 1 recovery on a committed 3×3 payoff environment.

    K=A=3, game-derived archetypes, shared θ, pair beliefs, Q from PopulationSpec.
    """
    a = int(spec.num_actions)
    if a != 3:
        raise ValueError(f"PSRV primary requires A=3 GameSpec; got num_actions={a}")
    base = recovery_flagship_cfg(seed=seed)
    # Start from flagship hyperparams, then override geometry / types / payoffs.
    cfg = replace(
        base,
        seed=int(seed),
        num_actions=3,
        num_prototypes=3,
        num_rounds=int(num_rounds),
        force_agent_true_types=balanced_types(population.n, 3),
        payoff_row=[list(map(float, r)) for r in spec.payoff_row],
        payoff_col=[list(map(float, r)) for r in spec.payoff_col],
        force_hidden_policy_by_agent=None,
    )
    cfg = apply_population_spec(cfg, population)
    cfg.validate()
    return cfg


def population_n20() -> PopulationSpec:
    return PopulationSpec.for_n(20, q=15)


def population_n100() -> PopulationSpec:
    return PopulationSpec.for_n(100, q=15)
