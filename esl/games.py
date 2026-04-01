"""Repeated 2-action matrix games and fixed hidden policies (recovery mode)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from esl.config import ESLConfig


# Semantics: 0 = Cooperate, 1 = Defect
ACTION_COOPERATE = 0
ACTION_DEFECT = 1


@dataclass(frozen=True)
class PayoffMatrices:
    """Row player i, column player j: payoff_i[a_i, a_j]."""

    row: np.ndarray  # shape (2, 2)
    col: np.ndarray  # shape (2, 2)


def prisoners_dilemma(cfg: ESLConfig) -> PayoffMatrices:
    T, R, P, S = cfg.pd_t, cfg.pd_r, cfg.pd_p, cfg.pd_s
    row = np.array([[R, S], [T, P]], dtype=np.float64)
    col = np.array([[R, T], [S, P]], dtype=np.float64)
    return PayoffMatrices(row=row, col=col)


class HiddenPolicy(ABC):
    """Fixed policy for an agent (no learned state in v1 beyond last-action hooks if needed)."""

    @abstractmethod
    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        pass

    @abstractmethod
    def action_probs(self) -> np.ndarray:
        """Deterministic distribution over actions for metrics (shape (2,))."""


class AlwaysCooperate(HiddenPolicy):
    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        return ACTION_COOPERATE

    def action_probs(self) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64)


class AlwaysDefect(HiddenPolicy):
    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        return ACTION_DEFECT

    def action_probs(self) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float64)


# Registry: prototype / type index -> policy instance
HIDDEN_POLICY_BUILDERS: dict[int, type[HiddenPolicy]] = {
    0: AlwaysCooperate,
    1: AlwaysDefect,
}


def build_hidden_policy(type_index: int) -> HiddenPolicy:
    if type_index not in HIDDEN_POLICY_BUILDERS:
        raise ValueError(f"Unknown hidden type index {type_index}; v1 supports {sorted(HIDDEN_POLICY_BUILDERS)}")
    return HIDDEN_POLICY_BUILDERS[type_index]()


def true_type_distributions(num_types: int) -> np.ndarray:
    """
    Shape (K, 2): row k is p(a | nominal type k) used for Hungarian CE / metrics.

    When K exceeds the number of registered base policies (v1: AC and AD only),
    rows **cycle** through those templates (``k % n_behavioral``). That is an
    **implementation convenience** for overparameterized / edge tests—not a
    theoretical restriction; see **ALGORITHM.md** (*Current implementation* — “Implementation note: K larger
    than base behavioral templates”). Trainer hidden policies use the same
    modulo when building ``HiddenPolicy`` from a type index.
    """
    if num_types < 1:
        raise ValueError("num_types must be >= 1")
    nbase = len(HIDDEN_POLICY_BUILDERS)
    rows = [build_hidden_policy(k % nbase).action_probs() for k in range(num_types)]
    return np.stack(rows, axis=0)


def play_pair_payoffs(
    a_i: int,
    a_j: int,
    pay: PayoffMatrices,
) -> tuple[float, float]:
    return float(pay.row[a_i, a_j]), float(pay.col[a_i, a_j])
