"""Repeated 2-action matrix games and fixed hidden policies (recovery mode)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from esl.config import ESLConfig


# Action semantics (game-dependent):
#   IPD:              0 = Cooperate, 1 = Defect
#   Stag Hunt:        0 = Stag,      1 = Hare
#   Matching Pennies: 0 = Heads,     1 = Tails
ACTION_COOPERATE = 0
ACTION_DEFECT = 1


@dataclass(frozen=True)
class PayoffMatrices:
    """Row player i, column player j: payoff_i[a_i, a_j]."""

    row: np.ndarray  # shape (2, 2)
    col: np.ndarray  # shape (2, 2)


def prisoners_dilemma(cfg: ESLConfig) -> PayoffMatrices:
    """Prisoner's Dilemma: T > R > P > S, 2R > T + S.

    Default: T=5, R=3, P=1, S=0.
    Actions: 0=Cooperate, 1=Defect.
    """
    T, R, P, S = cfg.pd_t, cfg.pd_r, cfg.pd_p, cfg.pd_s
    row = np.array([[R, S], [T, P]], dtype=np.float64)
    col = np.array([[R, T], [S, P]], dtype=np.float64)
    return PayoffMatrices(row=row, col=col)


def stag_hunt(cfg: ESLConfig) -> PayoffMatrices:
    """Stag Hunt: coordination game with two pure Nash equilibria.

    Payoff structure (A > B > C):
                    Stag(0)    Hare(1)
      Stag(0)       A, A       C, B
      Hare(1)       B, C       B, B

    (Stag, Stag) is payoff-dominant; (Hare, Hare) is risk-dominant.
    Default: A=4, B=3, C=0.
    """
    A = getattr(cfg, 'sh_a', 4.0)
    B = getattr(cfg, 'sh_b', 3.0)
    C = getattr(cfg, 'sh_c', 0.0)
    row = np.array([[A, C], [B, B]], dtype=np.float64)
    col = np.array([[A, B], [C, B]], dtype=np.float64)
    return PayoffMatrices(row=row, col=col)


def matching_pennies(cfg: ESLConfig) -> PayoffMatrices:
    """Matching Pennies: zero-sum game, no pure Nash equilibrium.

    Payoff structure:
                    Heads(0)   Tails(1)
      Heads(0)      +W, -W     -W, +W
      Tails(1)      -W, +W     +W, -W

    Row player wins on match; column player wins on mismatch.
    Unique mixed Nash: (0.5, 0.5) for both players.
    Default: W=1.
    """
    W = getattr(cfg, 'mp_w', 1.0)
    row = np.array([[W, -W], [-W, W]], dtype=np.float64)
    col = np.array([[-W, W], [W, -W]], dtype=np.float64)
    return PayoffMatrices(row=row, col=col)


GAME_BUILDERS = {
    "ipd": prisoners_dilemma,
    "stag_hunt": stag_hunt,
    "matching_pennies": matching_pennies,
}


def build_game(cfg: ESLConfig) -> PayoffMatrices:
    """Select and build the stage game from cfg.game_type."""
    name = getattr(cfg, 'game_type', 'ipd')
    if name not in GAME_BUILDERS:
        raise ValueError(f"Unknown game_type {name!r}; supported: {sorted(GAME_BUILDERS)}")
    return GAME_BUILDERS[name](cfg)


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


class SoftmaxLogitsPolicy(HiddenPolicy):
    """Stochastic policy from a fixed logit vector: P(a) = softmax(logits)[a].

    Used for synthetic populations where agents are drawn from ground-truth
    prototypes with optional per-agent noise.
    """

    def __init__(self, logits: np.ndarray) -> None:
        self._logits = np.asarray(logits, dtype=np.float64).ravel()

    def act(self, rng: np.random.Generator, *, last_opponent_action: int | None) -> int:
        p = self._softmax()
        return int(rng.choice(len(p), p=p))

    def action_probs(self) -> np.ndarray:
        return self._softmax()

    def _softmax(self) -> np.ndarray:
        x = self._logits - np.max(self._logits)
        e = np.exp(x)
        return e / e.sum()


def probs_to_logits(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert probability vectors to logits: θ_a = log(max(p_a, eps)).

    Input:  shape (..., A) probability distributions (rows should sum to 1).
    Output: shape (..., A) logits such that softmax(output) ≈ input.
    """
    probs = np.asarray(probs, dtype=np.float64)
    return np.log(np.clip(probs, eps, 1.0))


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
