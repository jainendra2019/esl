"""Repeated matrix games and fixed hidden policies (recovery mode).

PayoffMatrices support general A×A shapes; v1 paper protocol and many baselines
still assume A=2 (C=0, D=1). RES unlocks A∈{2,3} via GameSpec injection.
PSRV A=3 recovery uses game-derived archetypes (soft-BR to opponent pure action k).
"""

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

# Softmax temperature for game-derived archetype policies (implementation default).
ARCHETYPE_LAMBDA = 2.0


@dataclass(frozen=True)
class PayoffMatrices:
    """Row player i, column player j: payoff_i[a_i, a_j]."""

    row: np.ndarray  # shape (A, A)
    col: np.ndarray  # shape (A, A)

    def __post_init__(self) -> None:
        row = np.asarray(self.row)
        col = np.asarray(self.col)
        if row.ndim != 2 or col.ndim != 2 or row.shape != col.shape or row.shape[0] != row.shape[1]:
            raise ValueError(f"payoffs must be square (A,A); got row={row.shape} col={col.shape}")


def prisoners_dilemma(cfg: ESLConfig) -> PayoffMatrices:
    T, R, P, S = cfg.pd_t, cfg.pd_r, cfg.pd_p, cfg.pd_s
    row = np.array([[R, S], [T, P]], dtype=np.float64)
    col = np.array([[R, T], [S, P]], dtype=np.float64)
    return PayoffMatrices(row=row, col=col)


def matching_pennies() -> PayoffMatrices:
    """Zero-sum matching pennies: row wants to match actions; column is opponent."""
    row = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    col = -row
    return PayoffMatrices(row=row, col=col)


def game_payoffs(cfg: ESLConfig) -> PayoffMatrices:
    """Dispatch custom payoff override, matching pennies, or PD."""
    row_ov = getattr(cfg, "payoff_row", None)
    if row_ov is not None:
        row = np.asarray(row_ov, dtype=np.float64)
        col_src = getattr(cfg, "payoff_col", None)
        col = np.asarray(col_src if col_src is not None else row_ov, dtype=np.float64)
        return PayoffMatrices(row=row, col=col)
    name = getattr(cfg, "payoff_game", "prisoners_dilemma")
    if name == "matching_pennies":
        return matching_pennies()
    return prisoners_dilemma(cfg)


class HiddenPolicy(ABC):
    """Fixed policy for an agent (no learned state in v1 beyond last-action hooks if needed)."""

    @abstractmethod
    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        pass

    @abstractmethod
    def action_probs(self) -> np.ndarray:
        """Distribution over actions for metrics (shape (A,))."""


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


class TitForTat(HiddenPolicy):
    """
    Cooperate on the first observed encounter; thereafter copy the opponent's last action.

    The trainer passes ``last_opponent_action`` as the most recent action played by whoever
    the agent last faced (single scalar per agent). This is a pragmatic v1 hook for
    well-mixed populations, not full pairwise TFT memory.
    """

    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        if last_opponent_action is None:
            return ACTION_COOPERATE
        return int(last_opponent_action)

    def action_probs(self) -> np.ndarray:
        return np.array([0.5, 0.5], dtype=np.float64)


class SoftmaxArchetypePolicy(HiddenPolicy):
    """
    Game-derived latent type k: soft best response to opponent pure action k,

        π_k(a) ∝ exp(λ U(a, k))

    using the focal (row) payoff matrix. Used for PSRV K=A≥3 recovery.
    """

    def __init__(self, probs: np.ndarray):
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p.ndim != 1 or p.size < 2:
            raise ValueError("archetype probs must be a 1-D simplex of length A>=2")
        s = float(p.sum())
        if s <= 0:
            raise ValueError("archetype probs must have positive mass")
        self._probs = p / s

    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        return int(rng.choice(self._probs.size, p=self._probs))

    def action_probs(self) -> np.ndarray:
        return self._probs.copy()


def archetype_action_probs(
    pay_row: np.ndarray,
    type_k: int,
    *,
    lam: float = ARCHETYPE_LAMBDA,
) -> np.ndarray:
    """π_k(a) ∝ exp(λ U(a, k)) for row matrix U."""
    u = np.asarray(pay_row, dtype=np.float64)
    a = int(u.shape[0])
    k = int(type_k) % a
    logits = float(lam) * u[:, k]
    logits = logits - np.max(logits)
    w = np.exp(logits)
    return w / np.maximum(w.sum(), 1e-12)


def build_archetype_policy(
    pay_row: np.ndarray,
    type_k: int,
    *,
    lam: float = ARCHETYPE_LAMBDA,
) -> SoftmaxArchetypePolicy:
    return SoftmaxArchetypePolicy(archetype_action_probs(pay_row, type_k, lam=lam))


# Registry: prototype / type index -> policy instance (A=2 paper recovery)
HIDDEN_POLICY_BUILDERS: dict[int, type[HiddenPolicy]] = {
    0: AlwaysCooperate,
    1: AlwaysDefect,
    2: TitForTat,
}


def build_hidden_policy(type_index: int) -> HiddenPolicy:
    if type_index not in HIDDEN_POLICY_BUILDERS:
        raise ValueError(f"Unknown hidden type index {type_index}; v1 supports {sorted(HIDDEN_POLICY_BUILDERS)}")
    return HIDDEN_POLICY_BUILDERS[type_index]()


def true_type_distributions(
    num_types: int,
    *,
    num_actions: int = 2,
    pay_row: np.ndarray | None = None,
    lam: float = ARCHETYPE_LAMBDA,
) -> np.ndarray:
    """
    Shape (K, A): row k is p(a | nominal type k) used for Hungarian CE / metrics.

    - A=2 / no pay_row: cycle AC/AD/TFT templates (flagship recovery).
    - A>=3 or pay_row set: game-derived soft-BR archetypes from pay_row (requires pay_row).
    """
    if num_types < 1:
        raise ValueError("num_types must be >= 1")
    a = int(num_actions)
    if pay_row is not None or a > 2:
        if pay_row is None:
            raise ValueError("pay_row is required for true_type_distributions when num_actions > 2")
        u = np.asarray(pay_row, dtype=np.float64)
        if u.shape != (a, a):
            raise ValueError(f"pay_row shape {u.shape} != ({a},{a})")
        rows = [archetype_action_probs(u, k, lam=lam) for k in range(num_types)]
        return np.stack(rows, axis=0)
    nbase = len(HIDDEN_POLICY_BUILDERS)
    rows = [build_hidden_policy(k % nbase).action_probs() for k in range(num_types)]
    return np.stack(rows, axis=0)


def play_pair_payoffs(
    a_i: int,
    a_j: int,
    pay: PayoffMatrices,
) -> tuple[float, float]:
    return float(pay.row[a_i, a_j]), float(pay.col[a_i, a_j])
