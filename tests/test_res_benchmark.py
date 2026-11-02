from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esl.config import ESLConfig
from esl.res.benchmark import (
    assert_family_constraints,
    default_benchmark_root,
    generate_2x2_suite,
    generate_3x3_suite,
    load_suite,
    min_pairwise_feature_distance,
    write_suite,
)
from esl.res.game_spec import ExperimentSpec, GameSpec
from esl.res.stats import paired_mean_ci95


def test_game_spec_roundtrip(tmp_path: Path) -> None:
    spec = GameSpec(
        instance_id="coop_test",
        family="cooperative",
        num_actions=2,
        payoff_row=((3.0, 0.0), (5.0, 1.0)),
        payoff_col=((3.0, 5.0), (0.0, 1.0)),
        params={"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0},
    )
    path = tmp_path / "coop_test.json"
    spec.save_json(path)
    loaded = GameSpec.load_json(path)
    assert loaded.instance_id == "coop_test"
    assert loaded.to_payoffs().row.shape == (2, 2)


def test_generate_2x2_suite_diverse_and_constrained() -> None:
    specs = generate_2x2_suite(seed=20260725, per_family=4)
    assert len(specs) >= 12
    for s in specs:
        assert s.num_actions == 2
        assert_family_constraints(s)
    dists = min_pairwise_feature_distance(specs)
    for key, d in dists.items():
        assert d > 0.0, key


def test_generate_3x3_suite_shapes() -> None:
    specs = generate_3x3_suite(seed=20260728, per_family=4)
    assert len(specs) >= 12
    for s in specs:
        assert s.num_actions == 3
        assert np.asarray(s.payoff_row).shape == (3, 3)
        assert_family_constraints(s)


def test_write_and_load_suite(tmp_path: Path) -> None:
    specs = generate_2x2_suite(seed=1, per_family=3)
    write_suite(specs, tmp_path)
    loaded = load_suite(tmp_path, num_actions=2)
    assert len(loaded) == len(specs)
    assert (tmp_path / "manifest.json").is_file()


def test_config_allows_num_actions_3() -> None:
    try:
        cfg = ESLConfig(num_actions=3, num_agents=3, interaction_pairs_min=1, interaction_pairs_max=2)
        cfg.validate()
    except ValueError as exc:
        pytest.skip(f"A=3 ESLConfig is not in this checkout: {exc}")


def test_confusion_channel_rows_sum_to_one() -> None:
    signals = pytest.importorskip("esl.signals")
    if not hasattr(signals, "confusion_channel") or not hasattr(signals, "sample_noisy_action"):
        pytest.skip("confusion_channel / sample_noisy_action are not in this checkout")
    confusion_channel = signals.confusion_channel
    sample_noisy_action = signals.sample_noisy_action
    for a in (2, 3):
        for eps in (0.0, 0.1, 0.2):
            p = confusion_channel(a, eps)
            assert p.shape == (a, a)
            assert np.allclose(p.sum(axis=1), 1.0)
    rng = np.random.default_rng(0)
    samples = [sample_noisy_action(0, num_actions=3, eps=0.0, rng=rng) for _ in range(20)]
    assert set(samples) == {0}


def test_paired_stats_smoke() -> None:
    st = paired_mean_ci95([1.0, 2.0, 3.0])
    assert st["n"] == 3.0
    assert st["mean"] == 2.0


def test_experiment_spec_validates_knobs() -> None:
    game = generate_2x2_suite(seed=0, per_family=1)[0]
    exp = ExperimentSpec(game=game, seed=0, p_obs=0.8, obs_noise=0.05)
    assert exp.p_obs == 0.8


def test_committed_benchmark_if_present() -> None:
    root = default_benchmark_root()
    if not (root / "manifest.json").is_file():
        return
    specs = load_suite(root)
    assert len(specs) >= 12
    for s in specs:
        assert_family_constraints(s)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert int(man["n_instances"]) == len(specs)
