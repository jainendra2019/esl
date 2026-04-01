"""Sampling the interaction protocol: L_t interactions per round and ordered pairs E_t.

Locked semantics (see plan): L_min == L_max => no RNG draw for L_t (golden parity).
L_t == 1 => single random pair via one rng.integers draw (matches legacy trainer).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

LtLaw = Literal["uniform"]


def num_ordered_pairs(num_agents: int) -> int:
    """Count of (i, j) with i != j."""
    if num_agents < 2:
        return 0
    return num_agents * (num_agents - 1)


def all_ordered_pairs(num_agents: int) -> list[tuple[int, int]]:
    """Same order as legacy trainer: row-major i, then j."""
    return [(i, j) for i in range(num_agents) for j in range(num_agents) if i != j]


def sample_L_t(
    rng: np.random.Generator,
    L_min: int,
    L_max: int,
    *,
    law: LtLaw = "uniform",
) -> int:
    """
    Sample L_t in [L_min, L_max]. If L_min == L_max, return that value without consuming RNG.
    """
    if L_min < 1 or L_max < L_min:
        raise ValueError(f"need 1 <= L_min <= L_max, got L_min={L_min}, L_max={L_max}")
    if law != "uniform":
        raise ValueError(f"unsupported L_t law: {law}")
    if L_min == L_max:
        return int(L_min)
    return int(rng.integers(L_min, L_max + 1))


def sample_ordered_pairs_without_replacement(
    rng: np.random.Generator,
    num_agents: int,
    L_t: int,
) -> list[tuple[int, int]]:
    """
    Sample L_t distinct ordered pairs. Order is the sequence in which interactions run.

    Locked parity: L_t == 1 uses exactly one rng.integers(0, len(pool)) draw (legacy).
    L_t > 1 uses rng.permutation(len(pool))[:L_t] for distinct pairs in random order.
    """
    pool = all_ordered_pairs(num_agents)
    max_p = len(pool)
    if L_t < 1:
        raise ValueError("L_t must be >= 1")
    if L_t > max_p:
        raise ValueError(f"L_t={L_t} exceeds available ordered pairs {max_p} for N={num_agents}")
    if L_t == 1:
        idx = int(rng.integers(0, max_p))
        return [pool[idx]]
    perm = rng.permutation(max_p)
    return [pool[int(perm[k])] for k in range(L_t)]
