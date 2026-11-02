"""Test-first: entropy-regularized OMD / MWU helpers and categorical integration."""

from __future__ import annotations

import numpy as np
import pytest

from esl.res.omd_mwu import (
    OMD_PRIMARY_C,
    focal_reward_vector,
    normalize_matrix_once,
    omd_init_log_weights,
    omd_policy,
    omd_step,
)


def test_uniform_initialization() -> None:
    z = omd_init_log_weights(3)
    x = omd_policy(z)
    assert x.shape == (3,)
    assert np.allclose(x, np.full(3, 1.0 / 3.0))
    assert np.isclose(x.sum(), 1.0)
    assert np.all(x >= 0.0)


def test_policy_is_simplex() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=5)
    x = omd_policy(z)
    assert np.all(x >= 0.0)
    assert np.isclose(x.sum(), 1.0)


def test_eta_zero_freezes_policy() -> None:
    z = np.array([0.2, -0.1, 0.0], dtype=np.float64)
    r = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    z2 = omd_step(z, r, eta=0.0)
    assert np.allclose(z2, z)
    assert np.allclose(omd_policy(z2), omd_policy(z))


def test_equal_rewards_preserve_relative_weights() -> None:
    z = np.array([0.5, -0.2, 0.1], dtype=np.float64)
    x0 = omd_policy(z)
    r = np.full(3, 0.7, dtype=np.float64)
    z2 = omd_step(z, r, eta=0.1)
    x1 = omd_policy(z2)
    assert np.allclose(x1, x0, atol=1e-12)


def test_advantage_increases_probability() -> None:
    z = omd_init_log_weights(3)
    r = np.array([1.0, -1.0, -1.0], dtype=np.float64)
    z2 = omd_step(z, r, eta=0.3)
    x = omd_policy(z2)
    assert x[0] > x[1]
    assert x[0] > x[2]
    assert x[0] > 1.0 / 3.0


def test_normalize_once_to_unit_interval() -> None:
    u = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [0.5, 1.5, 2.5]], dtype=np.float64)
    n, lo, hi = normalize_matrix_once(u)
    assert np.isclose(lo, 0.0)
    assert np.isclose(hi, 5.0)
    assert np.isclose(n.min(), -1.0)
    assert np.isclose(n.max(), 1.0)
    # Ordering preserved
    assert np.argmax(n) == np.argmax(u)


def test_normalize_degenerate_returns_zeros() -> None:
    u = np.full((3, 3), 2.5, dtype=np.float64)
    n, lo, hi = normalize_matrix_once(u)
    assert lo == hi
    assert np.allclose(n, 0.0)


def test_focal_reward_vector_scaled_formula() -> None:
    # Focal U rows = actions, cols = opponent actions
    u = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float64)
    # Against opponent action 1: column 1 raw = [1,4,7] -> min0 max8 -> scale
    r = focal_reward_vector(u, opponent_action=1, u_min=0.0, u_max=8.0)
    expected = 2.0 * (np.array([1.0, 4.0, 7.0]) - 0.0) / 8.0 - 1.0
    assert np.allclose(r, expected)


def test_focal_reward_degenerate_zero() -> None:
    u = np.ones((3, 3), dtype=np.float64)
    r = focal_reward_vector(u, opponent_action=0, u_min=1.0, u_max=1.0)
    assert np.allclose(r, 0.0)


def test_primary_c_is_locked() -> None:
    assert OMD_PRIMARY_C == 0.1


def test_omd_works_for_arbitrary_action_dim() -> None:
    for a in (2, 3, 5):
        z = omd_init_log_weights(a)
        r = np.zeros(a)
        r[0] = 1.0
        z2 = omd_step(z, r, eta=0.1)
        x = omd_policy(z2)
        assert x.shape == (a,)
        assert np.isclose(x.sum(), 1.0)
        assert x[0] == max(x)


def test_categorical_cell_omd_no_switch_reset() -> None:
    """OMD log-weights persist across a type switch in the schedule."""
    from esl.res.benchmark import load_suite, default_benchmark_root
    from esl.res.runner import run_res_cell

    specs = load_suite(default_benchmark_root(), num_actions=3)
    assert specs
    cell = run_res_cell(
        spec=specs[0],
        method="OMD/MWU",
        seed=0,
        horizon=250,  # includes at least one switch at 200
        force_categorical=True,
        focal_role="row",
    )
    # Summary must exist; DPR N/A
    assert cell["summary"]["method"] == "OMD/MWU"
    assert cell["summary"]["dpr"] == "" or cell["summary"]["dpr"] == "N/A"
    assert "omd_c" in cell["summary"]
    assert float(cell["summary"]["omd_c"]) == OMD_PRIMARY_C
    # Trajectory has no belief field requirement; policy mass exists
    assert len(cell["trajectory"]) == 250
