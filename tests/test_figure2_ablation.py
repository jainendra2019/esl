from __future__ import annotations

import csv
import json
from pathlib import Path

from esl.baselines.external_common import matrix_game_tasks
from esl.experiments.figure1_dynamic_adaptation import H_WINDOW, build_type_shifting_schedule
from esl.experiments.figure2_ablation import (
    ABLATION_METHODS,
    EARLY_AUC_WINDOW,
    LTE_PANEL_METHODS,
    METHOD_ORDER,
    RECOVERY_METHODS,
    compute_early_gap_auc,
    run_ablation_cell,
    run_figure2_ablation,
)


def test_figure2_methods_and_shared_schedule() -> None:
    assert ABLATION_METHODS == ("ESL", "No-Sharing", "Fixed-K Bayesian", "No-Belief", "Online EM")
    assert METHOD_ORDER == ABLATION_METHODS
    assert LTE_PANEL_METHODS == ("ESL", "Fixed-K Bayesian", "Online EM")
    assert RECOVERY_METHODS == ("ESL", "Fixed-K Bayesian", "Online EM", "No-Belief")
    task = {t.key: t for t in matrix_game_tasks()}["stag_hunt"]
    expected = build_type_shifting_schedule(task=task, seed=2, horizon=80)
    expected_key = [(s.round, s.opponent_id, s.true_type, s.switch_id, s.random_u) for s in expected]
    for method in ABLATION_METHODS:
        result = run_ablation_cell(task=task, method=method, seed=2, horizon=80, h_window=H_WINDOW)
        got_key = [
            (int(r["round"]), int(r["opponent_id"]), int(r["true_type"]), int(r["switch_id"]), float(r["schedule_random_u"]))
            for r in result["trajectory"]
        ]
        assert got_key == expected_key


def test_early_gap_auc_is_mean_switch_auc() -> None:
    rows = [
        {"opponent_id": 1, "switch_id": 0, "tau": 0, "oracle_payoff": 3.0, "focal_payoff": 2.0},
        {"opponent_id": 1, "switch_id": 0, "tau": 1, "oracle_payoff": 3.0, "focal_payoff": 1.0},
        {"opponent_id": 2, "switch_id": 0, "tau": 0, "oracle_payoff": 5.0, "focal_payoff": 1.0},
        {"opponent_id": 2, "switch_id": 0, "tau": EARLY_AUC_WINDOW + 1, "oracle_payoff": 99.0, "focal_payoff": 0.0},
        {"opponent_id": 3, "switch_id": -1, "tau": 0, "oracle_payoff": 99.0, "focal_payoff": 0.0},
    ]
    assert compute_early_gap_auc(rows, early_window=EARLY_AUC_WINDOW) == 3.5


def test_ablation_component_constraints() -> None:
    task = {t.key: t for t in matrix_game_tasks()}["stag_hunt"]
    no_sharing = run_ablation_cell(task=task, method="No-Sharing", seed=0, horizon=80, h_window=H_WINDOW)
    no_belief = run_ablation_cell(task=task, method="No-Belief", seed=0, horizon=80, h_window=H_WINDOW)
    fixed = run_ablation_cell(task=task, method="Fixed-K Bayesian", seed=0, horizon=80, h_window=H_WINDOW)
    online_em = run_ablation_cell(task=task, method="Online EM", seed=0, horizon=80, h_window=H_WINDOW)

    assert no_sharing["summary"]["shared_prototypes"] is False
    assert no_belief["summary"]["belief_updates"] is False
    assert fixed["summary"]["prototype_learning"] is False
    assert online_em["summary"]["uses_switch_labels"] is False
    assert online_em["summary"]["uses_true_types"] is False
    assert all(r["lte"] != "" for r in no_belief["trajectory"])


def test_figure2_smoke_outputs_and_gates(tmp_path: Path) -> None:
    outputs = run_figure2_ablation(
        out_root=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        seeds=[0, 1, 2],
        horizon=500,
        h_window=H_WINDOW,
        tasks=("stag_hunt",),
        smoke=True,
    )
    for key in ("figure", "summary_csv", "report", "gate_json"):
        assert outputs[key].is_file()

    rows = list(csv.DictReader(outputs["summary_csv"].open()))
    metrics = {(r["method"], r["metric"]) for r in rows}
    for method in ABLATION_METHODS:
        assert (method, "psr") in metrics
        assert (method, "dpr") in metrics
        assert (method, "lte_postswitch_mean") in metrics
        assert (method, "early_gap_auc") in metrics
        assert (method, "lte_auc") not in metrics

    gate = json.loads(outputs["gate_json"].read_text(encoding="utf-8"))
    assert gate["pass"] is True
    assert gate["checks"]["esl_best_psr"] is True
    assert gate["checks"]["esl_best_dpr"] is True
    assert gate["checks"]["all_ablation_psr_worse"] is True
    assert gate["checks"]["em_low_lte_high_dpr"] is True

    report = outputs["report"].read_text(encoding="utf-8")
    assert "Removing any component worsens PSR" in report
    assert "EM low LTE but high DPR" in report
    assert "Identification alone is insufficient" in report
    assert "identification alone is insufficient under endogenous interaction" in report
    assert "No-sharing degrades PSR relative to ESL" in report
    assert "Error bars: 95% CI over seed-task cells" in report
