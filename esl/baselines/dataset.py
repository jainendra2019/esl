"""Load run directory: config, observation CSV, true types."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from esl import games
from esl.config import ESLConfig


@dataclass(frozen=True)
class ObservationRow:
    round: int
    i: int
    j: int
    w: float
    s: int
    a_i: int
    a_j: int


@dataclass(frozen=True)
class RunDataset:
    """Parsed run artifacts for baseline fitting."""

    run_dir: Path
    cfg: ESLConfig
    observations: list[ObservationRow]
    true_type_probs: np.ndarray
    true_types: np.ndarray

    @property
    def num_agents(self) -> int:
        return int(self.cfg.num_agents)

    @property
    def num_prototypes(self) -> int:
        return int(self.cfg.num_prototypes)

    @property
    def num_actions(self) -> int:
        return int(self.cfg.num_actions)


def load_config(run_dir: Path) -> ESLConfig:
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(ESLConfig)}
    return ESLConfig(**{k: v for k, v in data.items() if k in known})


def load_observations_csv(path: Path) -> list[ObservationRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    rows: list[ObservationRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"round", "i", "j", "w", "s", "a_i", "a_j"}
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")
        if not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must have columns {expected}, got {reader.fieldnames}")
        for r in reader:
            rows.append(
                ObservationRow(
                    round=int(r["round"]),
                    i=int(r["i"]),
                    j=int(r["j"]),
                    w=float(r["w"]),
                    s=int(r["s"]),
                    a_i=int(r["a_i"]),
                    a_j=int(r["a_j"]),
                )
            )
    return rows


def true_types_from_config(cfg: ESLConfig) -> np.ndarray:
    if cfg.force_agent_true_types is not None:
        return np.array(cfg.force_agent_true_types, dtype=int)
    return np.arange(cfg.num_agents, dtype=int) % cfg.num_prototypes


def load_run_dataset(run_dir: Path | str) -> RunDataset:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    cfg.validate()
    obs = load_observations_csv(run_dir / "interaction_observations.csv")
    true_probs = games.true_type_distributions(cfg.num_prototypes)
    ttypes = true_types_from_config(cfg)
    return RunDataset(
        run_dir=run_dir,
        cfg=cfg,
        observations=obs,
        true_type_probs=true_probs,
        true_types=ttypes,
    )


def observed_rows(ds: RunDataset) -> list[ObservationRow]:
    """Rows with positive observability mask (same convention as ESL Bayes path)."""
    return [r for r in ds.observations if r.w > 0]


def with_observation_prefix(ds: RunDataset, max_round_inclusive: int) -> RunDataset:
    """
    Protocol P: keep only interactions with ``round <= max_round_inclusive`` (chronological prefix).
    Same config and paths as ``ds``; observations list is filtered.
    """
    if max_round_inclusive < 0:
        raise ValueError("max_round_inclusive must be >= 0")
    obs = [r for r in ds.observations if r.round <= max_round_inclusive]
    return RunDataset(
        run_dir=ds.run_dir,
        cfg=ds.cfg,
        observations=obs,
        true_type_probs=ds.true_type_probs,
        true_types=ds.true_types,
    )
