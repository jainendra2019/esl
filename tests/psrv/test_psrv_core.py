"""TDD: K=A=3 game-derived archetypes and PopulationSpec / density."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esl.games import (
    ARCHETYPE_LAMBDA,
    archetype_action_probs,
    build_archetype_policy,
    true_type_distributions,
)
from esl.population_spec import PopulationSpec, apply_population_spec, density_bounds
from esl.psrv.presets import balanced_types, population_n20, population_n100, recovery_psrv_cfg
from esl.res.benchmark import default_benchmark_root, load_suite
from esl.trainer import run_esl


def test_density_bounds_n20_matches_flagship() -> None:
    assert density_bounds(20) == (5, 15)
    spec = population_n20()
    assert spec.n == 20 and spec.l_min == 5 and spec.l_max == 15 and spec.q == 15


def test_density_bounds_n100() -> None:
    assert density_bounds(100) == (25, 75)
    spec = population_n100()
    assert (spec.l_min, spec.l_max, spec.q) == (25, 75, 15)


def test_mean_density_near_half_interactions_per_agent() -> None:
    """E[L] ≈ N/2 ⇒ roughly 0.5 directed samples per agent role-slot per round."""
    for n in (20, 100):
        lo, hi = density_bounds(n)
        mean_l = 0.5 * (lo + hi)
        assert abs(mean_l - n / 2) < 1.0


def test_population20_apply_matches_flagship_geometry() -> None:
    from esl.experiments.presets import recovery_flagship_cfg

    cfg = apply_population_spec(recovery_flagship_cfg(seed=0), population_n20())
    assert cfg.num_agents == 20
    assert cfg.interaction_pairs_min == 5
    assert cfg.interaction_pairs_max == 15
    assert cfg.prototype_update_every == 15


def test_pair_sampling_no_self_no_dup() -> None:
    from esl.interaction_protocol import sample_ordered_pairs_without_replacement

    rng = np.random.default_rng(0)
    for n, lt in ((20, 10), (100, 50)):
        pairs = sample_ordered_pairs_without_replacement(rng, n, lt)
        assert len(pairs) == lt
        assert len(set(pairs)) == lt
        assert all(i != j for i, j in pairs)
        assert all(0 <= i < n and 0 <= j < n for i, j in pairs)


def test_archetype_soft_br_to_opponent_pure() -> None:
    # Identity-like U: best reply to opponent action k is action k
    u = np.eye(3, dtype=np.float64)
    for k in range(3):
        p = archetype_action_probs(u, k, lam=ARCHETYPE_LAMBDA)
        assert p.shape == (3,)
        assert np.isclose(p.sum(), 1.0)
        assert int(np.argmax(p)) == k


def test_archetypes_differ_across_payoff_environments() -> None:
    specs = load_suite(default_benchmark_root(), num_actions=3)
    assert len(specs) >= 2
    u0 = np.asarray(specs[0].payoff_row, dtype=np.float64)
    u1 = np.asarray(specs[1].payoff_row, dtype=np.float64)
    p0 = np.stack([archetype_action_probs(u0, k) for k in range(3)])
    p1 = np.stack([archetype_action_probs(u1, k) for k in range(3)])
    assert not np.allclose(p0, p1)


def test_true_type_distributions_a3() -> None:
    u = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = true_type_distributions(3, num_actions=3, pay_row=u)
    assert dist.shape == (3, 3)
    assert np.allclose(dist.sum(axis=1), 1.0)


def test_true_type_distributions_a2_unchanged_shape() -> None:
    dist = true_type_distributions(2)
    assert dist.shape == (2, 2)


def test_balanced_types() -> None:
    t = balanced_types(20, 3)
    assert len(t) == 20
    assert sorted(t.count(k) for k in range(3)) == [6, 7, 7]


def test_recovery_pipeline_runs_a3_smoke(tmp_path: Path) -> None:
    specs = load_suite(default_benchmark_root(), num_actions=3)
    pop = PopulationSpec.for_n(20, q=15)
    cfg = recovery_psrv_cfg(seed=0, spec=specs[0], population=pop, num_rounds=25)
    assert cfg.num_actions == 3 and cfg.num_prototypes == 3
    assert cfg.num_agents == 20
    _, logits, beliefs, summary, _ = run_esl(cfg, run_dir=tmp_path / "smoke")
    assert logits.shape == (3, 3)
    assert beliefs.shape == (20, 20, 3)
    assert np.isfinite(float(summary["final_mce"]))
    assert 0.0 <= float(summary["final_belief_argmax_accuracy"]) <= 1.0
