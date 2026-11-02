"""PopulationSpec: N, density-scaled L bounds, Q for Algorithm 1 recovery."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from esl.config import ESLConfig


def density_bounds(n: int) -> tuple[int, int]:
    """Lmin, Lmax preserving ~0.5 sampled interactions per agent per round."""
    n = int(n)
    if n < 2:
        raise ValueError("n must be >= 2")
    l_min = floor(n / 4)
    l_max = floor(3 * n / 4)
    max_pairs = n * (n - 1)
    l_min = max(1, l_min)
    l_max = min(max(l_max, l_min), max_pairs)
    return l_min, l_max


@dataclass(frozen=True)
class PopulationSpec:
    n: int
    l_min: int
    l_max: int
    q: int = 15

    @classmethod
    def for_n(cls, n: int, *, q: int = 15) -> PopulationSpec:
        lo, hi = density_bounds(n)
        return cls(n=int(n), l_min=lo, l_max=hi, q=int(q))


def apply_population_spec(cfg: ESLConfig, spec: PopulationSpec) -> ESLConfig:
    """Return a copy of cfg with population size, L bounds, and Q from spec."""
    from dataclasses import replace

    out = replace(
        cfg,
        num_agents=int(spec.n),
        interaction_pairs_min=int(spec.l_min),
        interaction_pairs_max=int(spec.l_max),
        prototype_update_every=int(spec.q),
    )
    out.validate()
    return out
