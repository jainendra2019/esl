"""Milestone 5: heterogeneous adaptation IPD pack (smoke)."""

from __future__ import annotations

from pathlib import Path

import pytest

from esl.config import ESLConfig
from esl.experiments.milestone5_pack import build_milestone5_esl_config, run_milestone5_pack
from esl.games import TitForTat, build_hidden_policy
from esl.trainer import run_esl


def test_tit_for_tat_first_move_cooperates_then_copies() -> None:
    rng = __import__("numpy").random.default_rng(0)
    tft = TitForTat()
    assert tft.act(rng, last_opponent_action=None) == 0
    assert tft.act(rng, last_opponent_action=1) == 1
    assert tft.act(rng, last_opponent_action=0) == 0


def test_build_hidden_policy_type2_is_tft() -> None:
    p = build_hidden_policy(2)
    assert isinstance(p, TitForTat)


def test_force_hidden_policy_by_agent_k1_adaptation_runs(tmp_path: Path) -> None:
    cfg = ESLConfig(
        seed=1,
        mode="adaptation",
        num_agents=4,
        num_prototypes=1,
        num_rounds=8,
        force_agent_true_types=[0, 0, 0, 0],
        force_hidden_policy_by_agent=[0, 0, 1, 1],
        adaptation_esl_agent_indices=[0],
        observability="full",
        p_obs=1.0,
        interaction_pairs_min=1,
        interaction_pairs_max=1,
        log_beliefs_tensor=False,
    )
    cfg.validate()
    _, _, _, summary, _ = run_esl(cfg, run_dir=tmp_path / "mix_k1")
    assert summary["num_rounds_executed"] == 8


def test_milestone5_smoke_pack_skip_ppo(tmp_path: Path) -> None:
    meta = run_milestone5_pack(
        out_root=tmp_path / "m5",
        manuscript_bundle=None,
        smoke=True,
        num_rounds=12,
        include_tft=False,
        skip_ppo=True,
    )
    assert meta["smoke"] is True
    assert (tmp_path / "m5" / "aggregate" / "milestone5_long.csv").is_file()


def test_milestone5_resume_skip_complete_second_pass(tmp_path: Path) -> None:
    root = tmp_path / "m5resume"
    run_milestone5_pack(
        out_root=root,
        manuscript_bundle=None,
        smoke=True,
        num_rounds=12,
        skip_ppo=True,
        resume_skip_complete=False,
    )
    rows_first = (root / "aggregate" / "milestone5_long.csv").read_text(encoding="utf-8")
    run_milestone5_pack(
        out_root=root,
        manuscript_bundle=None,
        smoke=True,
        num_rounds=12,
        skip_ppo=True,
        resume_skip_complete=True,
    )
    rows_second = (root / "aggregate" / "milestone5_long.csv").read_text(encoding="utf-8")
    assert rows_first == rows_second


def test_milestone5_smoke_writes_manuscript_bundle(tmp_path: Path) -> None:
    mb = tmp_path / "mb"
    run_milestone5_pack(
        out_root=tmp_path / "m5b",
        manuscript_bundle=mb,
        smoke=True,
        num_rounds=8,
        skip_ppo=True,
    )
    assert (mb / "reports" / "milestone_5" / "MILESTONE_STATUS.md").is_file()
    assert (mb / "tables" / "milestone5_method_comparison.csv").is_file()


def test_build_config_raises_tft_with_k1_ablation() -> None:
    with pytest.raises(ValueError):
        build_milestone5_esl_config(
            seed=0,
            num_rounds=10,
            include_tft=True,
            esl_ablation_k1=True,
            smoke=True,
        )
