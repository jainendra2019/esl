"""interaction_observations.csv alignment with trainer."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from esl.config import ESLConfig
from esl.trainer import run_esl


def test_interaction_log_row_count_matches_interaction_n_when_learning():
    cfg = ESLConfig(
        seed=7,
        num_rounds=8,
        num_agents=3,
        num_prototypes=2,
        observability="full",
        prototype_update_every=1,
        force_ordered_pair=(0, 1),
        log_interaction_observations=True,
        learning_frozen=False,
    )
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _log, _, _, summary, run_dir = run_esl(cfg, run_dir=td_path / "r")
        assert run_dir.is_dir()
        csv_path = run_dir / "interaction_observations.csv"
        assert csv_path.is_file()
        manifest = run_dir / "observation_manifest.json"
        assert manifest.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == summary["num_interaction_events_executed"]
        assert len(rows) > 0


def test_interaction_log_written_when_learning_frozen():
    cfg = ESLConfig(
        seed=8,
        num_rounds=4,
        num_agents=2,
        num_prototypes=2,
        observability="full",
        force_ordered_pair=(0, 1),
        log_interaction_observations=True,
        learning_frozen=True,
    )
    with tempfile.TemporaryDirectory() as td:
        _, _, _, summary, run_dir = run_esl(cfg, run_dir=Path(td) / "r")
        csv_path = run_dir / "interaction_observations.csv"
        assert csv_path.is_file()
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == cfg.num_rounds
        assert summary["num_interaction_events_executed"] == 0
