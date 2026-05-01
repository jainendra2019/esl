from __future__ import annotations

import numpy as np

from esl import games
from esl.baselines.external_common import matrix_game_tasks
from esl.experiments.figure1_learning_curves import (
    FOCAL_AGENT_ID,
    METHODS,
    REGIMES,
    _fixed_prototypes,
    build_figure1_schedule,
    fixedk_bayes_step,
    fp_action_probs,
    fp_update_counts,
    run_figure1_cell,
    run_figure1_learning_curves,
)
from esl.prototypes import likelihoods
from esl import beliefs as belief_ops


def test_fp_belief_update_correctness() -> None:
    counts = {1: np.ones(2, dtype=np.float64)}
    fp_update_counts(counts, opponent_id=1, opponent_action=games.ACTION_COOPERATE)
    fp_update_counts(counts, opponent_id=1, opponent_action=games.ACTION_DEFECT)
    fp_update_counts(counts, opponent_id=1, opponent_action=games.ACTION_DEFECT)
    assert counts[1].tolist() == [2.0, 3.0]
    assert np.allclose(counts[1] / counts[1].sum(), [0.4, 0.6])


def test_fp_best_response_correctness() -> None:
    counts = {1: np.array([11.0, 1.0], dtype=np.float64)}
    pay = games.PayoffMatrices(
        row=np.array([[4.0, 0.0], [3.0, 2.0]], dtype=np.float64),
        col=np.array([[4.0, 3.0], [0.0, 2.0]], dtype=np.float64),
    )
    probs = fp_action_probs(counts, 1, pay)
    assert probs[games.ACTION_COOPERATE] > probs[games.ACTION_DEFECT]


def test_fixedk_belief_update_matches_esl_bayes_step() -> None:
    prior = np.full(3, 1.0 / 3.0)
    prototypes = _fixed_prototypes()
    got = fixedk_bayes_step(prior, prototypes, games.ACTION_COOPERATE)
    expected = belief_ops.update_belief_pair(
        prior,
        likelihoods(prototypes, games.ACTION_COOPERATE),
        1e-4,
        1e-12,
    )
    assert np.allclose(got, expected)


def test_fixedk_has_zero_prototype_updates() -> None:
    result = run_figure1_cell(
        task=matrix_game_tasks()[0],
        regime="static",
        method="Fixed-K Bayesian",
        seed=0,
        horizon=8,
        record_interval=4,
    )
    assert result["summary"]["prototype_max_abs_delta"] == 0.0


def test_all_methods_produce_focal_mean_payoff() -> None:
    task = matrix_game_tasks()[0]
    for method in METHODS:
        result = run_figure1_cell(
            task=task,
            regime="static",
            method=method,
            seed=1,
            horizon=8,
            record_interval=4,
        )
        summary = result["summary"]
        assert summary["focal_agent_id"] == FOCAL_AGENT_ID
        assert summary["metric"] == "focal_mean_payoff_per_round"
        assert isinstance(summary["focal_mean_payoff_per_round"], float)


def test_all_methods_use_identical_opponent_schedules() -> None:
    task = matrix_game_tasks()[1]
    for regime in REGIMES:
        expected = build_figure1_schedule(task=task, regime=regime, seed=4, horizon=12)
        expected_pairs = [(s.opponent_id, s.opponent_type, s.random_u) for s in expected]
        for method in METHODS:
            result = run_figure1_cell(
                task=task,
                regime=regime,
                method=method,
                seed=4,
                horizon=12,
                record_interval=1,
            )
            got = [
                (int(r["opponent_id"]), int(r["opponent_type"]), float(r["schedule_random_u"]))
                for r in result["trajectory"]
            ]
            assert got == expected_pairs


def test_figure1_smoke_outputs(tmp_path) -> None:
    outputs = run_figure1_learning_curves(
        out_root=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        seeds=[0],
        horizon=8,
        record_interval=4,
    )
    assert outputs["figure"].is_file()
    assert outputs["summary_csv"].is_file()
    assert outputs["report"].is_file()
    assert "ESL K=1" not in outputs["summary_csv"].read_text(encoding="utf-8")
