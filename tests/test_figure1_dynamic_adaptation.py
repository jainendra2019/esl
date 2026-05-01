from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from esl.baselines.external_common import matrix_game_tasks
from esl.experiments.figure1_dynamic_adaptation import (
    ALL_METHODS,
    H_WINDOW,
    LATENT_METHODS,
    ONLINE_EM_WINDOW,
    PPO_OBSERVATION_FIELDS,
    PPO_UPDATE_EVERY,
    SOM_WINDOW,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
    aggregate_dynamic_metrics,
    build_type_shifting_schedule,
    compute_dynamic_prediction_regret,
    compute_early_adaptation_gap_auc,
    compute_latent_tracking_error,
    compute_post_switch_oracle_regret,
    run_dynamic_cell,
    run_figure1_dynamic_adaptation,
)


def test_schedule_switches_overlap_then_diverge() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["stag_hunt"]
    schedule = build_type_shifting_schedule(task=task, seed=0, horizon=450)
    assert SWITCH_INTERVAL == 200
    assert SWITCH_HAZARD == 0.01
    assert any(s.switch_id >= 0 for s in schedule)

    early = [s.true_action_probs[0] for s in schedule if 0 <= s.rounds_since_switch < 10]
    mature = [s.true_action_probs[0] for s in schedule if 80 <= s.rounds_since_switch < 120]
    assert np.std(early) < np.std(mature)


def test_all_methods_share_same_schedule_and_visibility() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["stag_hunt"]
    expected = build_type_shifting_schedule(task=task, seed=3, horizon=80)
    expected_key = [(s.round, s.opponent_id, s.true_type, s.switch_id, s.random_u) for s in expected]
    for method in ALL_METHODS:
        result = run_dynamic_cell(task=task, method=method, seed=3, horizon=80, h_window=H_WINDOW)
        got_key = [
            (int(r["round"]), int(r["opponent_id"]), int(r["true_type"]), int(r["switch_id"]), float(r["schedule_random_u"]))
            for r in result["trajectory"]
        ]
        assert got_key == expected_key
        has_lte = any(r["lte"] != "" for r in result["trajectory"])
        assert has_lte is (method in LATENT_METHODS)


def test_method_constraints_are_logged() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["stag_hunt"]
    som = run_dynamic_cell(task=task, method="SOM", seed=1, horizon=40, h_window=H_WINDOW)
    fixed = run_dynamic_cell(task=task, method="Fixed-K Bayesian", seed=1, horizon=40, h_window=H_WINDOW)
    online_em = run_dynamic_cell(task=task, method="Online EM", seed=1, horizon=40, h_window=H_WINDOW)

    assert som["summary"]["som_window"] == SOM_WINDOW
    assert som["summary"]["uses_deep_network"] is False
    assert fixed["summary"]["prototype_max_abs_delta"] == 0.0
    assert fixed["summary"]["fixed_k_prototype_source"]
    assert online_em["summary"]["online_em_window"] == ONLINE_EM_WINDOW
    assert online_em["summary"]["uses_switch_labels"] is False
    assert online_em["summary"]["uses_true_types"] is False


def test_figure1_ppo_uses_official_agent_not_uniform_placeholder() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["ipd"]
    result = run_dynamic_cell(task=task, method="PPO", seed=0, horizon=80, h_window=H_WINDOW)
    summary = result["summary"]
    assert summary["ppo_source"] == "official_ppo_pytorch"
    assert summary["ppo_is_learning"] is True
    assert int(summary["ppo_update_count"]) > 0
    assert summary["ppo_action_path"] == "official_select_action"
    assert summary["prediction_source"] == "not_applicable"
    assert {r["ppo_action_path"] for r in result["trajectory"]} == {"official_select_action"}


def test_figure1_ppo_observation_contract_has_no_leakage() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["ipd"]
    result = run_dynamic_cell(task=task, method="PPO", seed=0, horizon=32, h_window=H_WINDOW)
    summary = result["summary"]
    assert summary["ppo_state_dim"] == 5
    assert summary["ppo_observation_fields"] == list(PPO_OBSERVATION_FIELDS)
    assert summary["ppo_context"] == "constant_no_regime_info"
    forbidden = {"true_type", "switch_id", "switch_round", "oracle_payoff", "oracle_action"}
    assert forbidden.isdisjoint(set(summary["ppo_observation_fields"]))
    for row in result["trajectory"]:
        assert row["prediction_source"] == "not_applicable"


def test_figure1_ppo_logs_common_metric_fields() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["ipd"]
    result = run_dynamic_cell(task=task, method="PPO", seed=1, horizon=40, h_window=H_WINDOW)
    required = {
        "focal_payoff",
        "oracle_payoff",
        "pred_p_coop",
        "pred_p_defect",
        "prediction_ce",
        "oracle_ce",
        "psr_gap",
        "dpr_gap",
        "prediction_source",
    }
    assert required.issubset(result["trajectory"][0])
    assert np.isfinite(float(result["summary"]["psr"]))
    assert np.isfinite(float(result["summary"]["dpr"]))


def test_figure1_ppo_update_cadence_matches_main_grid_full_cadence() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["ipd"]
    horizon = PPO_UPDATE_EVERY * 3
    result = run_dynamic_cell(task=task, method="PPO", seed=2, horizon=horizon, h_window=H_WINDOW)
    assert result["summary"]["ppo_update_every"] == PPO_UPDATE_EVERY
    assert result["summary"]["ppo_update_count"] == 3


def test_metric_math_matches_hand_computed_values() -> None:
    rows = [
        {"switch_id": 0, "tau": 0, "oracle_payoff": 3.0, "focal_payoff": 2.0, "prediction_ce": 0.7, "oracle_ce": 0.2},
        {"switch_id": 0, "tau": 1, "oracle_payoff": 2.0, "focal_payoff": 2.5, "prediction_ce": 0.6, "oracle_ce": 0.3},
        {"switch_id": 0, "tau": 2, "oracle_payoff": 1.0, "focal_payoff": 0.0, "prediction_ce": 0.5, "oracle_ce": 0.4},
        {"switch_id": 0, "tau": 31, "oracle_payoff": 10.0, "focal_payoff": 0.0, "prediction_ce": 0.5, "oracle_ce": 0.4},
    ]
    psr = compute_post_switch_oracle_regret(rows, h_window=2)
    dpr = compute_dynamic_prediction_regret(rows, h_window=2)
    early = compute_early_adaptation_gap_auc(rows, early_window=30)
    assert np.isclose(psr, 1.5)
    assert np.isclose(dpr, 0.9)
    assert np.isclose(early, 1.5)


def test_lte_uses_hungarian_matching() -> None:
    beliefs = np.array([0.05, 0.90, 0.05], dtype=np.float64)
    true_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.55, 0.45]], dtype=np.float64)
    learned_logits = np.log(np.array([[0.1, 0.9], [0.9, 0.1], [0.55, 0.45]], dtype=np.float64))
    lte = compute_latent_tracking_error(beliefs, true_type=0, learned_logits=learned_logits, true_probs=true_probs)
    assert lte < 0.2


def test_dynamic_smoke_outputs_and_gates(tmp_path: Path) -> None:
    outputs = run_figure1_dynamic_adaptation(
        out_root=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        seeds=[0, 1, 2],
        horizon=500,
        h_window=H_WINDOW,
        tasks=("stag_hunt",),
        smoke=True,
    )
    for key in ("figure", "summary_csv", "report", "caption"):
        assert outputs[key].is_file()

    report = outputs["report"].read_text(encoding="utf-8")
    assert "Smoke gate: PASS" in report
    assert "ESL reduces post-switch adaptation cost" in report
    assert (
        "PPO uses the official PyTorch implementation under the same interaction protocol; since PPO has no "
        "explicit opponent model, DPR is computed using a uniform predictive prior and should be interpreted only "
        "as a no-modeling prediction baseline."
    ) in report

    rows = list(csv.DictReader(outputs["summary_csv"].open()))
    metrics = {(r["method"], r["metric"]) for r in rows}
    expected_metrics = {
        "post_switch_oracle_regret",
        "latent_identification_error",
        "prediction_regret",
        "early_adaptation_gap_auc",
    }
    assert metrics == {(method, metric) for method in ALL_METHODS for metric in expected_metrics}

    by_key = {(r["method"], r["metric"]): r for r in rows}
    for method in ("PPO", "FP", "SOM"):
        row = by_key[(method, "latent_identification_error")]
        assert row["metric_defined"] == "false"
        assert row["mean"] == ""
        assert row["ci95_low"] == ""
        assert row["ci95_high"] == ""
    for method in ("ESL", "Fixed-K Bayesian", "Online EM"):
        assert by_key[(method, "latent_identification_error")]["metric_defined"] == "true"
    assert by_key[("ESL", "post_switch_oracle_regret")]["metric_defined"] == "true"
    assert by_key[("Online EM", "prediction_regret")]["metric_defined"] == "true"

    gate = json.loads(outputs["gate_json"].read_text(encoding="utf-8"))
    assert gate["pass"] is True
    assert (
        aggregate_dynamic_metrics(rows)["ESL"]["post_switch_oracle_regret"]["mean"]
        < aggregate_dynamic_metrics(rows)["PPO"]["post_switch_oracle_regret"]["mean"]
    )
    caption = outputs["caption"].read_text(encoding="utf-8")
    assert "marked N/A" in caption
    assert "Inset shows rapid recovery dynamics" in caption
    assert "DPR is computed using a uniform predictive prior" in caption
    assert "Non-latent methods are marked N/A" in report
