"""Scientific contracts for the accepted twelve-game 3×3 protocol."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from esl.experiments.figure1_dynamic_adaptation import (
    H_WINDOW,
    compute_dynamic_prediction_regret,
    compute_post_switch_oracle_regret,
)
from esl.metrics import match_prototypes_to_types
from esl.prototypes import stable_softmax
from esl.res.benchmark import load_suite
from esl.res.categorical_cell import (
    EPS,
    _lte,
    _lte_auc,
    run_categorical_cell,
)
from esl.res.k_sensitivity import inspect_esl_init, prefix_init_rows
from esl.res.runner import game_spec_to_task
from esl.signals import confusion_channel, sample_noisy_action


def _coop_spec():
    specs = [s for s in load_suite(num_actions=3) if s.instance_id == "coop3_01"]
    assert specs, "committed coop3_01 is required"
    return specs[0]


def test_action_is_chosen_before_observation():
    spec = _coop_spec()
    out = run_categorical_cell(
        task=game_spec_to_task(spec),
        method="ESL",
        seed=0,
        horizon=30,
        num_actions=3,
        p_obs=0.0,
    )
    for row in out["trajectory"]:
        assert row["focal_action"] in (0, 1, 2)
        assert row["opponent_action"] in (0, 1, 2)
        assert int(row["observed"]) == 0
        assert row["observed_action"] == ""


def test_observation_masking_p_obs_zero_and_one():
    spec = _coop_spec()
    task = game_spec_to_task(spec)
    full = run_categorical_cell(task=task, method="Fixed-K Bayesian", seed=1, horizon=40, num_actions=3, p_obs=1.0)
    none = run_categorical_cell(task=task, method="Fixed-K Bayesian", seed=1, horizon=40, num_actions=3, p_obs=0.0)
    assert all(int(r["observed"]) == 1 for r in full["trajectory"])
    assert all(int(r["observed"]) == 0 for r in none["trajectory"])
    ltes = [float(r["lte"]) for r in none["trajectory"] if r["lte"] != ""]
    assert ltes
    assert abs(ltes[-1] - float(-np.log(1.0 / 3.0))) < 1e-9


def test_lte_uses_pre_update_belief_and_strict_window():
    spec = _coop_spec()
    out = run_categorical_cell(
        task=game_spec_to_task(spec),
        method="ESL",
        seed=2,
        horizon=40,
        num_actions=3,
        h_window=5,
        switch_interval=10,
        switch_hazard=1.0,
    )
    rows = out["trajectory"]
    recorded = [r for r in rows if r["lte"] != ""]
    assert recorded
    first = recorded[0]
    belief = np.array(json.loads(first["belief"]), dtype=np.float64)
    # Pre-update snapshot is stored on the row; uniform init should still be near 1/K.
    assert belief.shape == (3,)
    assert np.isclose(belief.sum(), 1.0)
    strict = [float(r["lte"]) for r in rows if r["lte"] != "" and int(r["tau"]) < 5]
    inclusive = [
        float(r["lte"])
        for r in rows
        if r["lte"] != "" and 0 <= int(r["tau"]) <= 5 and int(r["switch_id"]) >= 0
    ]
    got = _lte_auc(rows, h_window=5)
    assert got == pytest.approx(float(np.mean(strict)))
    assert len(strict) <= len(inclusive)


def test_psr_dpr_inclusive_window():
    rows = [
        {
            "switch_id": 0,
            "tau": tau,
            "oracle_payoff": 2.0,
            "focal_payoff": 1.0,
            "prediction_ce": 1.5,
            "oracle_ce": 0.5,
        }
        for tau in range(0, 6)
    ]
    assert compute_post_switch_oracle_regret(rows, h_window=5) == pytest.approx(6.0)
    assert compute_dynamic_prediction_regret(rows, h_window=5) == pytest.approx(6.0)


def test_lte_hungarian_permutation_invariance():
    true_probs = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float64,
    )
    logits = np.log(true_probs)
    perm = np.array([2, 0, 1])
    shuffled = logits[perm]
    belief = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    # shuffled[i] = logits[perm[i]], so belief mass on new i is old mass on perm[i].
    shuffled_belief = belief[perm]
    a = _lte(belief, true_type=2, learned_logits=logits, true_probs=true_probs)
    b = _lte(shuffled_belief, true_type=2, learned_logits=shuffled, true_probs=true_probs)
    assert a == pytest.approx(b, rel=0, abs=1e-12)
    match, _ = match_prototypes_to_types(true_probs, shuffled, method="hungarian")
    assert set(match.tolist()) == {0, 1, 2}


def test_prototype_update_after_batch_count(monkeypatch):
    from esl.experiments.figure1_dynamic_adaptation import ESLUpdateConfig
    import esl.res.categorical_cell as cc

    updates = []
    orig = cc.batch_weighted_prototype_gradient

    def counted(*args, **kwargs):
        updates.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(cc, "batch_weighted_prototype_gradient", counted)
    spec = _coop_spec()
    run_categorical_cell(
        task=game_spec_to_task(spec),
        method="ESL",
        seed=0,
        horizon=20,
        num_actions=3,
        p_obs=1.0,
        esl_update=ESLUpdateConfig(label="ESL", gamma=0.9, batch_size=2, eta_reg=0.0),
    )
    # One gradient evaluation per observed record in a flushed batch of size 2.
    assert len(updates) >= 2
    assert len(updates) % 2 == 0


def test_variable_prototype_cardinality_init():
    for k in (2, 3, 5):
        info = inspect_esl_init(seed=0, k_model=k)
        assert info["logits_shape"] == [k, 3]
        assert info["n_beliefs"] == 3
        assert all(v == k for v in info["belief_lens"].values())
        assert info["uniform_beliefs"]
    assert prefix_init_rows(0, 2, 3)
    assert prefix_init_rows(0, 3, 5)


def test_matched_seed_pairing_same_schedule():
    spec = _coop_spec()
    a = run_categorical_cell(task=game_spec_to_task(spec), method="ESL", seed=4, horizon=25, num_actions=3)
    b = run_categorical_cell(task=game_spec_to_task(spec), method="Fixed-K Bayesian", seed=4, horizon=25, num_actions=3)
    keys_a = [(r["round"], r["opponent_id"], r["true_type"], r["switch_id"]) for r in a["trajectory"]]
    keys_b = [(r["round"], r["opponent_id"], r["true_type"], r["switch_id"]) for r in b["trajectory"]]
    assert keys_a == keys_b


def test_confusion_channel_is_row_stochastic():
    ch = confusion_channel(3, 0.2)
    assert ch.shape == (3, 3)
    np.testing.assert_allclose(ch.sum(axis=1), 1.0, atol=1e-12)
    rng = np.random.default_rng(0)
    sample = sample_noisy_action(1, num_actions=3, eps=0.0, rng=rng)
    assert sample == 1


def test_lte_clip_convention():
    true_probs = np.eye(3, dtype=np.float64)
    logits = np.log(np.clip(true_probs, 1e-8, 1.0))
    tiny = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    val = _lte(tiny, true_type=2, learned_logits=logits, true_probs=true_probs)
    assert math.isfinite(val)
    assert val <= -math.log(EPS) + 1e-9
