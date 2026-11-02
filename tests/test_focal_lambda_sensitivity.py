"""TDD for focal λ sensitivity (split RNGs, entropy, paired Δ vs λ=2)."""

from __future__ import annotations

import numpy as np
import pytest

from esl.res.benchmark import load_suite
from esl.res.focal_lambda_sensitivity import (
    DEFAULT_LAMBDA,
    FOCAL_LAMBDA_GRID,
    best_action_prob,
    policy_probs_from_utilities,
    run_focal_lambda_cell,
    summarize_lambda_grid,
)
from esl.res.runner import game_spec_to_task


def test_lam0_exact_uniform() -> None:
    u = np.array([3.0, 1.0, 0.0], dtype=np.float64)
    p = policy_probs_from_utilities(u, 0.0)
    assert p.shape == (3,)
    assert np.allclose(p, np.full(3, 1.0 / 3.0))


def test_increasing_lam_raises_mass_on_unique_br() -> None:
    u = np.array([2.0, 0.5, 0.0], dtype=np.float64)
    p0 = policy_probs_from_utilities(u, 0.0)[0]
    p05 = policy_probs_from_utilities(u, 0.5)[0]
    p2 = policy_probs_from_utilities(u, 2.0)[0]
    p10 = policy_probs_from_utilities(u, 10.0)[0]
    assert p0 < p05 < p2 < p10
    assert best_action_prob(policy_probs_from_utilities(u, 10.0), u) == pytest.approx(p10)


def test_finite_lam_remains_stochastic() -> None:
    u = np.array([1.0, 0.8, 0.5], dtype=np.float64)
    p = policy_probs_from_utilities(u, 5.0)
    assert np.all(p > 1e-6)
    assert p.sum() == pytest.approx(1.0)


def test_identical_opponent_stream_across_lambda() -> None:
    specs = load_suite(num_actions=3)
    task = game_spec_to_task(specs[0])
    outs = [
        run_focal_lambda_cell(task=task, seed=0, lam_focal=lam, horizon=80)
        for lam in (0.0, 2.0, 10.0)
    ]
    base = outs[0]
    for o in outs[1:]:
        assert o["true_types"] == base["true_types"]
        assert o["opponent_actions"] == base["opponent_actions"]
        assert o["signals"] == base["signals"]
    # Focal actions must be allowed to differ for some λ (λ=0 vs λ=10)
    assert outs[0]["focal_actions"] != outs[-1]["focal_actions"]


def test_psr_entropy_finite_and_entropy_falls_with_lam() -> None:
    specs = load_suite(num_actions=3)
    task = game_spec_to_task(specs[0])
    rows = []
    for lam in FOCAL_LAMBDA_GRID:
        o = run_focal_lambda_cell(task=task, seed=1, lam_focal=lam, horizon=120)
        assert np.isfinite(o["psr"])
        assert np.isfinite(o["mean_action_entropy"])
        assert np.isfinite(o["mean_p_best"])
        rows.append(o)
    # Mean entropy should be non-increasing as λ increases (policy sharpens)
    ents = [r["mean_action_entropy"] for r in rows]
    assert ents[0] >= ents[-1] - 1e-9
    pbest = [r["mean_p_best"] for r in rows]
    assert pbest[0] <= pbest[-1] + 1e-9


def test_summarize_paired_delta_vs_default() -> None:
    fake = []
    for seed in (0, 1):
        for lam, psr in ((0.0, 30.0), (2.0, 20.0), (10.0, 22.0)):
            fake.append(
                {
                    "instance_id": "g",
                    "seed": seed,
                    "lam_focal": lam,
                    "psr": psr + seed,
                    "dpr": 5.0,
                    "mean_action_entropy": 1.0,
                    "mean_p_best": 0.5,
                }
            )
    summary = summarize_lambda_grid(fake)
    by = {float(r["lam_focal"]): r for r in summary["table"]}
    assert by[2.0]["delta_vs_default"] == pytest.approx(0.0)
    assert by[0.0]["delta_vs_default"] == pytest.approx(10.0)
    assert DEFAULT_LAMBDA == 2.0
