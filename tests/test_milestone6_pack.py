"""Milestone 6 mechanism & robustness pack (smoke) and metric helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esl.config import ESLConfig
from esl.experiments.milestone6_pack import run_milestone6_pack
from esl.games import game_payoffs, matching_pennies
from esl.metrics import mce_value, pairwise_assignment_cost
from esl.prototypes import stable_softmax
from esl.trainer import run_esl


def test_pairwise_cost_rectangular_2x5() -> None:
    true_p = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0], [3.0, -1.0]], dtype=np.float64)
    c = pairwise_assignment_cost(true_p, logits)
    assert c.shape == (2, 5)


def test_mce_kt2_km1_is_mean_ce_to_single_prototype() -> None:
    true_p = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    logits = np.array([[0.5, 0.5]], dtype=np.float64)
    m = mce_value(true_p, logits)
    q = stable_softmax(logits)[0]
    from esl.metrics import cross_entropy

    expected = 0.5 * (cross_entropy(true_p[0], q) + cross_entropy(true_p[1], q))
    assert abs(m - expected) < 1e-9


def test_matching_pennies_zero_sum_row() -> None:
    pay = matching_pennies()
    r0, c0 = pay.row[0, 0], pay.col[0, 0]
    assert abs(r0 + c0) < 1e-12


def test_game_payoffs_dispatches_matching_pennies() -> None:
    cfg = ESLConfig(seed=0, payoff_game="matching_pennies", num_rounds=1)
    cfg.validate()
    pay = game_payoffs(cfg)
    assert pay.row.shape == (2, 2)


def test_belief_updates_disabled_runs(tmp_path: Path) -> None:
    cfg = ESLConfig(
        seed=0,
        mode="adaptation",
        num_agents=4,
        num_prototypes=2,
        num_rounds=6,
        belief_updates_enabled=False,
        observability="full",
        p_obs=1.0,
        interaction_pairs_min=1,
        interaction_pairs_max=1,
        log_beliefs_tensor=False,
    )
    cfg.validate()
    _, _, _, summary, _ = run_esl(cfg, run_dir=tmp_path / "noup")
    assert summary["num_rounds_executed"] == 6


def test_metrics_num_true_types_mce_rectangular(tmp_path: Path) -> None:
    cfg = ESLConfig(
        seed=1,
        mode="adaptation",
        num_agents=4,
        num_prototypes=3,
        metrics_num_true_types=2,
        num_rounds=5,
        observability="full",
        p_obs=1.0,
        interaction_pairs_min=1,
        interaction_pairs_max=1,
        log_beliefs_tensor=False,
    )
    cfg.validate()
    _, _, _, summary, _ = run_esl(cfg, run_dir=tmp_path / "mrect")
    assert summary.get("final_mce") is not None


def test_milestone6_smoke_pack(tmp_path: Path) -> None:
    meta = run_milestone6_pack(
        out_root=tmp_path / "m6",
        manuscript_bundle=None,
        smoke=True,
        num_rounds=12,
        run_matching_pennies=False,
    )
    assert meta["smoke"] is True
    assert (tmp_path / "m6" / "aggregate" / "milestone6_all_runs.csv").is_file()
