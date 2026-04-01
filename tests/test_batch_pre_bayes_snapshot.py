"""PRD §8: batch stores pre-Bayes b_{i->j,t}; beliefs advance before append."""

from __future__ import annotations

import numpy as np

from esl.beliefs import init_beliefs
from esl.config import ESLConfig
from esl.trainer import observe_signal_update_belief


def test_batch_carries_prior_belief_snapshot():
    cfg = ESLConfig(delta_simplex=1e-4)
    b = init_beliefs(2, 2)
    logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    prior = b[0, 1].copy()
    rec = observe_signal_update_belief(b, logits, i=0, j=1, signal=0, w=1.0, cfg=cfg)
    assert np.allclose(rec.b_ij, prior)
    assert not np.allclose(b[0, 1], prior)
