"""Golden-field tests for NeurIPS recovery presets (no long runs)."""

from __future__ import annotations

import pytest

from esl.experiments.presets import (
    FLAGSHIP_AGENT_TYPES,
    recovery_failure_case_cfg,
    recovery_fixed_prototype_baseline_cfg,
    recovery_flagship_cfg,
    recovery_init_noise_sweep_cfg,
    recovery_lr_sweep_cfg,
    recovery_Q_sweep_cfg,
    recovery_short_horizon_cfg,
    recovery_sparse_obs_cfg,
)


def test_init_noise_sweep_changes_noise_only():
    c = recovery_init_noise_sweep_cfg(init_noise=0.12, seed=7)
    assert c.init_noise == pytest.approx(0.12)
    assert c.num_rounds == recovery_flagship_cfg(seed=7).num_rounds


def test_flagship_matches_tracked_reference_fields():
    c = recovery_flagship_cfg(seed=42)
    assert c.num_agents == 20
    assert c.num_prototypes == 2
    assert c.num_rounds == 10_000
    assert c.force_agent_true_types == FLAGSHIP_AGENT_TYPES
    assert c.interaction_pairs_min == 5
    assert c.interaction_pairs_max == 15
    assert c.prototype_Q() == 15
    assert c.prototype_lr_scale == 22.0
    assert c.init_noise == 0.05
    assert c.delta_simplex == 0.02
    assert c.p_obs == 1.0
    assert c.observability == "full"
    assert c.log_beliefs_tensor is False
    assert c.freeze_prototype_parameters is False


def test_sparse_only_changes_observability_and_p_obs():
    base = recovery_flagship_cfg(seed=0)
    s = recovery_sparse_obs_cfg(p_obs=0.5, seed=0)
    assert s.p_obs == pytest.approx(0.5)
    assert s.observability == "sparse"
    assert s.num_rounds == base.num_rounds
    assert s.prototype_lr_scale == base.prototype_lr_scale


def test_short_horizon_fixed_interaction_budget():
    c = recovery_short_horizon_cfg(interaction_budget=500, interactions_per_round=10, seed=1)
    assert c.num_rounds == 50
    assert c.interaction_pairs_min == c.interaction_pairs_max == 10


def test_short_horizon_budget_must_divide():
    with pytest.raises(ValueError, match="divisible"):
        recovery_short_horizon_cfg(interaction_budget=501, interactions_per_round=10, seed=1)


def test_lr_sweep_changes_scale_only():
    c = recovery_lr_sweep_cfg(prototype_lr_scale=12.0, seed=2)
    assert c.prototype_lr_scale == pytest.approx(12.0)
    assert c.num_rounds == recovery_flagship_cfg(seed=2).num_rounds


def test_Q_sweep_changes_Q_only():
    c = recovery_Q_sweep_cfg(Q=20, seed=3)
    assert c.prototype_Q() == 20


def test_failure_case_is_symmetric_lower_lr_shorter():
    f = recovery_failure_case_cfg(seed=0)
    b = recovery_flagship_cfg(seed=0)
    assert f.symmetric_init is True
    assert f.prototype_lr_scale < b.prototype_lr_scale
    assert f.num_rounds < b.num_rounds


def test_fixed_prototype_sets_freeze():
    c = recovery_fixed_prototype_baseline_cfg(seed=0)
    assert c.freeze_prototype_parameters is True
