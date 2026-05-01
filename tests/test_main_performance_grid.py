from __future__ import annotations

import json
from pathlib import Path

import pytest

from esl.baselines.external_common import (
    BaselineAvailability,
    BaselineSpec,
    ReducedMatrixGameAdapter,
    SourceInfo,
    matrix_game_tasks,
    opponent_regimes,
)
from esl.baselines.mbom_adapter import build_adapter as build_mbom_adapter
from esl.baselines.mfos_adapter import build_adapter as build_mfos_adapter
from esl.baselines.ppo_adapter import build_adapter as build_ppo_adapter
from esl.baselines.simple_opponent_adapter import build_adapter as build_simple_adapter
from esl.experiment_registry import resolve_experiment_block
from esl.experiments.main_performance_grid import (
    FOCAL_AGENT_ID,
    _run_onboarding_smokes,
    build_shared_opponent_schedule,
    run_shared_focal_protocol,
    run_main_performance_grid,
    write_baseline_audit,
)
from esl.experiments.schema import validate_baseline_policy_smoke_directory


def _tiny_task_regime():
    return matrix_game_tasks()[0], opponent_regimes()[0]


@pytest.mark.parametrize(
    "block_id",
    [
        "baseline.ppo.performance_smoke",
        "baseline.mfos.performance_smoke",
        "baseline.mbom.performance_smoke",
        "baseline.simple_opponent.performance_smoke",
    ],
)
def test_performance_registry_blocks(block_id: str) -> None:
    assert resolve_experiment_block(block_id).run_kind == "baseline"


@pytest.mark.parametrize(
    "adapter_factory",
    [build_mfos_adapter, build_mbom_adapter, build_simple_adapter],
)
def test_reduced_external_adapter_writes_canonical_outputs(tmp_path: Path, adapter_factory) -> None:
    adapter = adapter_factory()
    task, regime = _tiny_task_regime()
    run_dir = adapter.train(
        tmp_path / adapter.spec.family,
        seed=1,
        task=task,
        regime=regime,
        horizon=4,
        smoke=True,
    )
    validate_baseline_policy_smoke_directory(run_dir)
    summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary["baseline_family"] == adapter.spec.family
    assert isinstance(summary["mean_payoff_per_agent_per_round"], float)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["sources"][0]["local_path"].startswith("third_party/")


def test_ppo_performance_adapter_smoke_if_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    adapter = build_ppo_adapter()
    if not adapter.spec.availability.can_run_repeated_2x2:
        pytest.skip(adapter.spec.availability.reason)
    task, regime = _tiny_task_regime()
    run_dir = adapter.train(tmp_path / "ppo", seed=2, task=task, regime=regime, horizon=4, smoke=True)
    validate_baseline_policy_smoke_directory(run_dir)


def test_baseline_audit_and_unavailable_onboarding(tmp_path: Path) -> None:
    spec = BaselineSpec(
        family="fake_unavailable",
        method_name="Fake",
        adapter_path="tests.fake",
        source=SourceInfo(
            name="fake",
            url="https://example.invalid/fake",
            local_path="third_party/fake",
            commit=None,
            license="unknown",
            notes="test only",
        ),
        deviations=("test deviation",),
        availability=BaselineAvailability(
            status="NOT_INTEGRATED",
            reason="test unavailable",
            can_run_repeated_2x2=False,
        ),
    )
    adapter = ReducedMatrixGameAdapter(spec)
    audit = write_baseline_audit(tmp_path / "reports" / "baseline_audit.json", [adapter])
    assert audit["baselines"]["fake_unavailable"]["status"] == "NOT_INTEGRATED"
    rows = _run_onboarding_smokes(
        out_root=tmp_path / "runs",
        reports_dir=tmp_path / "reports",
        adapters=[adapter],
        horizon=2,
    )
    assert rows[0]["result"] == "UNAVAILABLE"


def test_main_performance_grid_smoke_outputs(tmp_path: Path) -> None:
    outputs = run_main_performance_grid(
        out_root=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        smoke=True,
        seeds=[0],
        horizon=3,
    )
    assert outputs["audit"].is_file()
    assert outputs["summary_csv"].is_file()
    assert outputs["figure"].is_file()
    assert outputs["report"].is_file()
    assert outputs["figure"].name == "main_performance_grid_protocol_fixed_smoke.png"


def test_shared_protocol_focal_payoff_is_agent_zero_only() -> None:
    task, regime = _tiny_task_regime()
    result = run_shared_focal_protocol(
        method="SOM",
        task=task,
        regime=regime,
        seed=11,
        horizon=5,
        smoke=True,
    )
    rows = result["rows"]
    assert {r["focal_agent_id"] for r in rows} == {FOCAL_AGENT_ID}
    reward_sum = sum(float(r["focal_reward"]) for r in rows)
    assert result["summary"]["focal_cumulative_payoff"] == pytest.approx(reward_sum)
    assert result["summary"]["focal_mean_payoff_per_round"] == pytest.approx(reward_sum / len(rows))


def test_esl_acts_in_fixed_types_regime() -> None:
    task, regime = _tiny_task_regime()
    result = run_shared_focal_protocol(
        method="ESL",
        task=task,
        regime=regime,
        seed=3,
        horizon=8,
        smoke=True,
    )
    rows = result["rows"]
    assert result["summary"]["shared_protocol"] is True
    assert result["summary"]["focal_agent_id"] == 0
    assert len({int(r["focal_action"]) for r in rows}) >= 1
    assert all(r["method"] == "ESL" for r in rows)


def test_methods_receive_same_opponent_schedule_for_same_seed() -> None:
    task, regime = _tiny_task_regime()
    expected = build_shared_opponent_schedule(task=task, regime=regime, seed=4, horizon=6)
    expected_pairs = [(s.opponent_id, s.opponent_type, s.random_u) for s in expected]
    for method in ("ESL", "ESL K=1", "ESL without belief updates", "PPO", "SOM"):
        result = run_shared_focal_protocol(
            method=method,
            task=task,
            regime=regime,
            seed=4,
            horizon=6,
            smoke=True,
        )
        got = [
            (int(r["opponent_id"]), int(r["opponent_type"]), float(r["schedule_random_u"]))
            for r in result["rows"]
        ]
        assert got == expected_pairs


def test_unsupported_baselines_are_skipped_not_synthetic() -> None:
    task, regime = _tiny_task_regime()
    for method in ("M-FOS", "MBOM"):
        result = run_shared_focal_protocol(
            method=method,
            task=task,
            regime=regime,
            seed=0,
            horizon=4,
            smoke=True,
        )
        assert result["status"] == "SKIPPED"
        assert result["rows"] == []
        assert result["summary"]["baseline_fidelity"] == "skipped"
        assert "official" in result["skip_reason"]


def test_protocol_fixed_outputs_mark_shared_protocol(tmp_path: Path) -> None:
    outputs = run_main_performance_grid(
        out_root=tmp_path / "runs",
        manuscript_bundle=tmp_path / "manuscript_bundle",
        smoke=True,
        seeds=[0],
        horizon=3,
    )
    raw = outputs["raw_csv"].read_text(encoding="utf-8")
    skipped = outputs["skipped_csv"].read_text(encoding="utf-8")
    assert "True,0,official" in raw
    assert "ESL K=1" in raw
    assert "ESL without belief updates" in raw
    assert "M-FOS" in skipped
    assert "MBOM" in skipped


def test_esl_ablations_are_lower_capacity_shared_protocol() -> None:
    task, regime = _tiny_task_regime()
    for method, fidelity in (
        ("ESL K=1", "ablation"),
        ("ESL without belief updates", "ablation"),
    ):
        result = run_shared_focal_protocol(
            method=method,
            task=task,
            regime=regime,
            seed=5,
            horizon=6,
            smoke=True,
        )
        assert result["summary"]["shared_protocol"] is True
        assert result["summary"]["baseline_fidelity"] == fidelity
        assert result["summary"]["focal_agent_id"] == FOCAL_AGENT_ID
