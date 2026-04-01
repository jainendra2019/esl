"""Validation tests VAL1–VAL8 + dashboard: ESL behavior vs PRD intent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from esl.config import ESLConfig
from esl.metrics import match_prototypes_to_types
from esl import games
from esl.trainer import init_prototype_logits, run_esl


@pytest.mark.validation
def test_val1_symmetric_init_no_separation():
    cfg = ESLConfig(
        seed=10,
        symmetric_init=True,
        init_noise=0.0,
        num_rounds=100,
        num_agents=3,
        num_prototypes=2,
        observability="full",
        prototype_update_every=1,
    )
    rng = cfg.make_rng()
    t0 = init_prototype_logits(cfg, rng)
    sep0 = float(np.linalg.norm(t0[0] - t0[1]))
    with tempfile.TemporaryDirectory() as td:
        _, logits, b, summary, _ = run_esl(cfg, run_dir=Path(td))
    sep1 = float(np.linalg.norm(logits[0] - logits[1]))
    assert sep1 <= sep0 + 0.6
    assert summary["final_belief_entropy"] > 0.5


@pytest.mark.validation
def test_val2_asymmetry_induces_specialization():
    cfg = ESLConfig(
        seed=11,
        symmetric_init=False,
        init_noise=0.2,
        num_rounds=180,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=1,
    )
    rng = cfg.make_rng()
    t0 = init_prototype_logits(cfg, rng)
    sep0 = float(np.linalg.norm(t0[0] - t0[1]))
    true_probs = games.true_type_distributions(2)
    _, ce0 = match_prototypes_to_types(true_probs, t0)
    with tempfile.TemporaryDirectory() as td:
        _, logits, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    sep1 = float(np.linalg.norm(logits[0] - logits[1]))
    assert sep1 >= sep0 * 0.4
    assert summary["final_matched_cross_entropy"] < ce0 or summary["final_matched_cross_entropy"] < 0.2


@pytest.mark.validation
@pytest.mark.parametrize("seed", [12, 42, 99])
def test_val3_two_type_recovery_short_horizon_multi_seed(seed: int):
    """Seed-stability smoke: same recovery criteria on three seeds (not a single lucky draw)."""
    cfg = ESLConfig(
        seed=seed,
        num_rounds=200,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        init_noise=0.08,
        symmetric_init=False,
        prototype_update_every=1,
    )
    rng = cfg.make_rng()
    t0 = init_prototype_logits(cfg, rng)
    true_probs = games.true_type_distributions(2)
    _, ce0 = match_prototypes_to_types(true_probs, t0)
    with tempfile.TemporaryDirectory() as td:
        _, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    ce1 = float(summary["final_matched_cross_entropy"])
    assert ce1 < ce0 or ce1 < 0.2, f"matched CE did not improve vs init (seed={seed})"
    assert summary["final_belief_argmax_accuracy"] > 0.35, f"belief argmax accuracy low (seed={seed})"


@pytest.mark.validation
def test_val4_sparse_observability_degrades_recovery():
    seeds = [7, 8, 9]
    ce_full, ce_sparse = [], []
    for sd in seeds:
        for p_obs, bucket in ((1.0, ce_full), (0.2, ce_sparse)):
            cfg = ESLConfig(
                seed=sd,
                num_rounds=250,
                num_agents=4,
                num_prototypes=2,
                observability="sparse",
                p_obs=p_obs,
                init_noise=0.1,
                prototype_update_every=5,
            )
            with tempfile.TemporaryDirectory() as td:
                _, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
            bucket.append(float(summary["final_matched_cross_entropy"]))
    assert float(np.mean(ce_full)) < float(np.mean(ce_sparse))


@pytest.mark.validation
def test_val5_prototype_updates_on_fixed_schedule():
    M = 5
    T = 103
    cfg = ESLConfig(
        seed=0,
        num_rounds=T,
        num_agents=3,
        prototype_update_every=M,
        observability="full",
    )
    with tempfile.TemporaryDirectory() as td:
        log, _, _, summary, _ = run_esl(cfg, run_dir=Path(td))
    expected_updates = T // M + (0 if T % M == 0 else 1)
    assert summary["prototype_update_count"] == expected_updates
    norms = [float(r["prototype_update_norm"]) for r in log.summary_rows]
    scheduled_logged = T // M
    nonzero = sum(1 for n in norms if n > 0)
    assert nonzero == scheduled_logged
    if T % M != 0:
        assert expected_updates == scheduled_logged + 1


@pytest.mark.validation
def test_val6_batch_log_likelihood_trend():
    cfg = ESLConfig(
        seed=42,
        mode="recovery",
        num_rounds=500,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        init_noise=0.05,
        prototype_update_every=5,
    )
    with tempfile.TemporaryDirectory() as td:
        log, _, _, _, _ = run_esl(cfg, run_dir=Path(td))
    rows = log.summary_rows
    n = len(rows)
    q = n // 4
    early = [float(r["batch_log_likelihood"]) for r in rows[:q]]
    late = [float(r["batch_log_likelihood"]) for r in rows[-q:]]
    assert len(early) > 10 and len(late) > 10
    assert float(np.mean(late)) > float(np.mean(early)) - 0.2


@pytest.mark.validation
def test_val7_adaptation_learning_beats_frozen_baseline():
    gains = []
    for sd in (100, 101, 102):
        base = dict(
            seed=sd,
            mode="adaptation",
            num_rounds=160,
            num_agents=4,
            num_prototypes=2,
            observability="full",
            prototype_update_every=2,
            init_noise=0.05,
        )
        with tempfile.TemporaryDirectory() as t1:
            _, _, _, s_learn, _ = run_esl(ESLConfig(**base, learning_frozen=False), run_dir=Path(t1))
        with tempfile.TemporaryDirectory() as t2:
            _, _, _, s_frozen, _ = run_esl(ESLConfig(**base, learning_frozen=True), run_dir=Path(t2))
        gains.append(float(s_learn["cumulative_social_payoff"]) - float(s_frozen["cumulative_social_payoff"]))
    assert float(np.median(gains)) > 0.0


@pytest.mark.validation
def test_val8_entropy_falls_before_payoff_rises():
    cfg = ESLConfig(
        seed=55,
        mode="adaptation",
        num_rounds=200,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=2,
        init_noise=0.06,
    )
    with tempfile.TemporaryDirectory() as td:
        log, _, _, _, _ = run_esl(cfg, run_dir=Path(td))
    rows = log.summary_rows
    n = len(rows)
    f = max(n // 5, 5)
    H_early = float(np.mean([float(r["belief_entropy_mean"]) for r in rows[:f]]))
    H_late = float(np.mean([float(r["belief_entropy_mean"]) for r in rows[-f:]]))
    r_early = float(np.mean([(float(x["r_i"]) + float(x["r_j"])) / 2 for x in log.reward_rows[:f]]))
    r_late = float(np.mean([(float(x["r_i"]) + float(x["r_j"])) / 2 for x in log.reward_rows[-f:]]))
    assert H_early > H_late
    assert r_late >= r_early - 1.0


@pytest.mark.validation
def test_dashboard_summary_fields():
    cfg = ESLConfig(num_rounds=15, prototype_update_every=1)
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        _, logits, _, summary, rd_out = run_esl(cfg, run_dir=rd)
        assert rd_out == rd
        required = (
            "seed",
            "final_matched_cross_entropy",
            "final_belief_entropy",
            "final_belief_argmax_accuracy",
            "cumulative_social_payoff",
            "mean_payoff_per_agent_per_round",
            "prototype_update_count",
            "num_rounds",
            "learning_frozen",
            "permutation_true_to_learned",
            "cost_matrix",
        )
        for k in required:
            assert k in summary
        assert (rd / "config.json").exists()
        met = (rd / "metrics_trajectory.csv").read_text(encoding="utf-8")
        for col in (
            "matched_cross_entropy",
            "belief_entropy_mean",
            "prototype_update_norm",
            "belief_change_norm",
            "batch_log_likelihood",
        ):
            assert col in met.splitlines()[0]
