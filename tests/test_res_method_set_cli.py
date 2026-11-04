"""Regression tests for RES CLI method-set selection (flagship vs tier1).

These tests monkeypatch ``run_res_m1`` rather than regenerating the 840-cell
flagship suite. They lock documentation contracts without changing accepted
numerical evidence.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from esl.res import runner as res_runner
from esl.res.runner import (
    FLAGSHIP_A3_METHODS,
    FLAGSHIP_A3_METHODS_NO_PPO,
    TIER1_METHODS,
    aggregate_all_metrics,
    paired_gaps_all_metrics,
    resolve_method_set,
)


def test_tier1_default_five_methods() -> None:
    assert TIER1_METHODS == ("ESL", "Fixed-K Bayesian", "SA-FP", "FP", "SOM")
    assert len(TIER1_METHODS) == 5


def test_flagship_a3_is_seven_methods_without_omd() -> None:
    assert FLAGSHIP_A3_METHODS == (
        "ESL",
        "PPO",
        "Fixed-K Bayesian",
        "SA-FP",
        "FP",
        "SOM",
        "Online EM",
    )
    assert len(FLAGSHIP_A3_METHODS) == 7
    assert "OMD/MWU" not in FLAGSHIP_A3_METHODS


def test_flagship_a3_no_ppo_is_six_methods() -> None:
    assert FLAGSHIP_A3_METHODS_NO_PPO == (
        "ESL",
        "Fixed-K Bayesian",
        "SA-FP",
        "FP",
        "SOM",
        "Online EM",
    )
    assert len(FLAGSHIP_A3_METHODS_NO_PPO) == 6
    assert "PPO" not in FLAGSHIP_A3_METHODS_NO_PPO
    assert "OMD/MWU" not in FLAGSHIP_A3_METHODS_NO_PPO


def test_resolve_method_set_defaults_and_restrictions() -> None:
    assert resolve_method_set("tier1", num_actions=2) == TIER1_METHODS
    assert resolve_method_set("tier1", num_actions=3) == TIER1_METHODS
    assert resolve_method_set("flagship-a3", num_actions=3) == FLAGSHIP_A3_METHODS
    assert resolve_method_set("flagship-a3-no-ppo", num_actions=3) == FLAGSHIP_A3_METHODS_NO_PPO
    with pytest.raises(ValueError, match="only supported for --num-actions 3"):
        resolve_method_set("flagship-a3", num_actions=2)
    with pytest.raises(ValueError, match="only supported for --num-actions 3"):
        resolve_method_set("flagship-a3-no-ppo", num_actions=2)
    with pytest.raises(ValueError, match="Unknown --method-set"):
        resolve_method_set("not-a-set", num_actions=3)


def test_full_mode_cell_count_contracts() -> None:
    n_games, n_seeds = 12, 10
    assert n_games * n_seeds * len(TIER1_METHODS) == 600
    assert n_games * n_seeds * len(FLAGSHIP_A3_METHODS_NO_PPO) == 720
    assert n_games * n_seeds * len(FLAGSHIP_A3_METHODS) == 840


def test_cli_default_still_uses_tier1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_res_m1(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"n_summaries": 0, "ok": True}

    monkeypatch.setattr(res_runner, "run_res_m1", fake_run_res_m1)
    res_runner.main(
        ["--num-actions", "3", "--full", "--out-dir", str(tmp_path / "tier1")]
    )
    assert captured["methods"] == TIER1_METHODS
    assert tuple(captured["seeds"]) == tuple(range(10))
    assert captured["horizon"] == 1000
    assert captured["num_actions"] == 3


def test_cli_flagship_a3_passes_seven_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_res_m1(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"n_summaries": 0, "ok": True}

    monkeypatch.setattr(res_runner, "run_res_m1", fake_run_res_m1)
    monkeypatch.setattr(res_runner, "require_torch_for_ppo", lambda: None)
    res_runner.main(
        [
            "--num-actions",
            "3",
            "--full",
            "--method-set",
            "flagship-a3",
            "--out-dir",
            str(tmp_path / "flagship"),
        ]
    )
    assert captured["methods"] == FLAGSHIP_A3_METHODS
    assert len(captured["methods"]) == 7
    assert tuple(captured["seeds"]) == tuple(range(10))
    assert captured["horizon"] == 1000


def test_cli_flagship_no_ppo_passes_six_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_res_m1(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"n_summaries": 0, "ok": True}

    monkeypatch.setattr(res_runner, "run_res_m1", fake_run_res_m1)
    res_runner.main(
        [
            "--num-actions",
            "3",
            "--full",
            "--method-set",
            "flagship-a3-no-ppo",
            "--out-dir",
            str(tmp_path / "flagship_no_ppo"),
        ]
    )
    assert captured["methods"] == FLAGSHIP_A3_METHODS_NO_PPO
    assert "PPO" not in captured["methods"]


def test_cli_flagship_requires_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom() -> None:
        raise SystemExit(
            "PyTorch is required for --method-set flagship-a3 (PPO). "
            "Install torch, or use --method-set flagship-a3-no-ppo."
        )

    monkeypatch.setattr(res_runner, "require_torch_for_ppo", boom)
    with pytest.raises(SystemExit, match="PyTorch is required"):
        res_runner.main(
            [
                "--num-actions",
                "3",
                "--full",
                "--method-set",
                "flagship-a3",
                "--out-dir",
                str(tmp_path / "need_torch"),
            ]
        )


def test_aggregates_include_ppo_and_online_em() -> None:
    summaries = []
    for method in FLAGSHIP_A3_METHODS:
        summaries.append(
            {
                "family": "cooperative",
                "instance_id": "coop3_01",
                "method": method,
                "seed": 0,
                "psr": 1.0,
                "dpr": 2.0 if method != "PPO" else 2.0,
                "lte_auc": 3.0 if method in ("ESL", "Fixed-K Bayesian", "SA-FP", "Online EM") else "",
                "early_gap_auc": 0.5,
                "focal_role": "row",
            }
        )
    rows = aggregate_all_metrics(summaries)
    methods_in_psr = {r["method"] for r in rows if r["metric"] == "psr" and r["family"] == "all"}
    assert methods_in_psr == set(FLAGSHIP_A3_METHODS)
    paired = paired_gaps_all_metrics(summaries)
    pair_keys = {(r["method_a"], r["method_b"], r["metric"]) for r in paired}
    assert ("PPO", "ESL", "psr") in pair_keys
    assert ("Online EM", "ESL", "psr") in pair_keys


def test_online_em_short_integration_smoke() -> None:
    """Short Online EM cell; not a flagship evaluation result."""
    from esl.res.benchmark import load_suite
    from esl.res.runner import run_res_cell

    specs = [s for s in load_suite(num_actions=3) if s.instance_id == "coop3_01"]
    assert specs
    out = run_res_cell(spec=specs[0], method="Online EM", seed=0, horizon=40)
    assert out["summary"]["method"] == "Online EM"
    assert float(out["summary"]["psr"]) == float(out["summary"]["psr"])  # finite


def test_ppo_short_integration_smoke_when_torch_available() -> None:
    """Short PPO cell when torch is installed; skip otherwise. Not flagship eval."""
    pytest.importorskip("torch")
    from esl.res.benchmark import load_suite
    from esl.res.runner import run_res_cell

    specs = [s for s in load_suite(num_actions=3) if s.instance_id == "coop3_01"]
    assert specs
    out = run_res_cell(spec=specs[0], method="PPO", seed=0, horizon=40)
    assert out["summary"]["method"] == "PPO"
    assert float(out["summary"]["psr"]) == float(out["summary"]["psr"])


def test_cli_does_not_call_suite_generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Method-set selection must not resample the committed benchmark."""
    captured: dict[str, Any] = {}

    def fake_run_res_m1(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"n_summaries": 0}

    def ban_generate(*_a: Any, **_k: Any) -> None:
        raise AssertionError("generate_3x3_suite must not be called from CLI method-set path")

    monkeypatch.setattr(res_runner, "run_res_m1", fake_run_res_m1)
    monkeypatch.setattr(res_runner, "require_torch_for_ppo", lambda: None)
    bench = importlib.import_module("esl.res.benchmark")
    monkeypatch.setattr(bench, "generate_3x3_suite", ban_generate)
    res_runner.main(
        [
            "--num-actions",
            "3",
            "--full",
            "--method-set",
            "flagship-a3",
            "--out-dir",
            str(tmp_path / "nosample"),
        ]
    )
    assert captured["methods"] == FLAGSHIP_A3_METHODS
