"""GameSpec (committed instance) and ExperimentSpec (run-time knobs)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl import games

InteractionFamily = Literal["cooperative", "coordination", "adversarial"]


@dataclass(frozen=True)
class GameSpec:
    """Committed benchmark instance (regime ≠ realization)."""

    instance_id: str
    family: InteractionFamily
    num_actions: int
    payoff_row: tuple[tuple[float, ...], ...]
    payoff_col: tuple[tuple[float, ...], ...]
    generator_seed: int = 0
    notes: str = ""
    params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        row = np.asarray(self.payoff_row, dtype=np.float64)
        col = np.asarray(self.payoff_col, dtype=np.float64)
        a = int(self.num_actions)
        if a < 2:
            raise ValueError("num_actions must be >= 2")
        if row.shape != (a, a) or col.shape != (a, a):
            raise ValueError(f"payoffs must be ({a},{a}); got row={row.shape} col={col.shape}")
        if self.family not in ("cooperative", "coordination", "adversarial"):
            raise ValueError(f"unknown family: {self.family}")

    def to_payoffs(self) -> games.PayoffMatrices:
        return games.PayoffMatrices(
            row=np.asarray(self.payoff_row, dtype=np.float64),
            col=np.asarray(self.payoff_col, dtype=np.float64),
        )

    def feature_vector(self) -> np.ndarray:
        """Flattened row payoffs for diversity distance (symmetric games: row suffices)."""
        return np.asarray(self.payoff_row, dtype=np.float64).reshape(-1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("params") is None:
            d.pop("params", None)
        return d

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> GameSpec:
        row = tuple(tuple(float(x) for x in r) for r in obj["payoff_row"])
        col = tuple(tuple(float(x) for x in r) for r in obj["payoff_col"])
        params = obj.get("params")
        return cls(
            instance_id=str(obj["instance_id"]),
            family=obj["family"],  # type: ignore[arg-type]
            num_actions=int(obj["num_actions"]),
            payoff_row=row,
            payoff_col=col,
            generator_seed=int(obj.get("generator_seed", 0)),
            notes=str(obj.get("notes", "")),
            params=dict(params) if params is not None else None,
        )

    @classmethod
    def load_json(cls, path: Path) -> GameSpec:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ExperimentSpec:
    """Run-time protocol + environmental perturbations around a GameSpec."""

    game: GameSpec
    seed: int
    horizon: int = 1000
    h_window: int = 100
    p_obs: float = 1.0
    obs_noise: float = 0.0
    switch_interval: int = 200
    switch_hazard: float = 0.01
    sa_fp_p_stay: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.p_obs) <= 1.0):
            raise ValueError("p_obs must be in [0, 1]")
        if not (0.0 <= float(self.obs_noise) < 1.0):
            raise ValueError("obs_noise must be in [0, 1)")
