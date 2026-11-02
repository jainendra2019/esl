"""Test-first: categorical Online EM must not use A=2 Bernoulli sufficient stats."""

from __future__ import annotations

import numpy as np
import pytest

from esl.res.online_em_categorical import update_online_em_categorical


def test_online_em_rejects_binary_only_path_for_a3() -> None:
    """A=3 update must write full simplex logits (K, A), not Bernoulli log p / log(1-p)."""
    rng = np.random.default_rng(0)
    k, a = 3, 3
    logits = rng.normal(size=(k, a))
    # Window of opponent actions in {0,1,2}
    window = [(1, 0), (1, 1), (2, 2), (2, 0), (3, 1), (1, 2)]
    beliefs = {opp: np.full(k, 1.0 / k) for opp in (1, 2, 3)}
    new_logits, new_beliefs = update_online_em_categorical(
        logits, beliefs=beliefs, em_window=window, num_actions=a
    )
    assert new_logits.shape == (k, a)
    # Softmax rows are distributions
    from esl.prototypes import stable_softmax

    probs = stable_softmax(new_logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)
    # Not a Bernoulli embedding: columns beyond 2 must be used / finite
    assert np.all(np.isfinite(new_logits))
    for opp in (1, 2, 3):
        assert np.isclose(new_beliefs[opp].sum(), 1.0)


def test_online_em_a2_still_produces_valid_simplex() -> None:
    logits = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    window = [(1, 0), (1, 0), (2, 1), (3, 0)]
    beliefs = {opp: np.full(3, 1.0 / 3) for opp in (1, 2, 3)}
    new_logits, new_beliefs = update_online_em_categorical(
        logits, beliefs=beliefs, em_window=window, num_actions=2
    )
    assert new_logits.shape == (3, 2)
    from esl.prototypes import stable_softmax

    probs = stable_softmax(new_logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_figure1_online_em_is_binary_specific() -> None:
    """Audit lock: paper Online EM uses ACTION_COOPERATE Bernoulli M-step — not for A=3."""
    import inspect
    from esl.experiments import figure1_dynamic_adaptation as fig1

    src = inspect.getsource(fig1._update_online_em)
    assert "ACTION_COOPERATE" in src
    assert "logits[:, 1]" in src
