from __future__ import annotations

import numpy as np
import pytest

from esl_analysis.metrics.divergence import js_divergence, kl_divergence


def test_js_divergence_properties() -> None:
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert js_divergence(p, q) == pytest.approx(0.0)
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert js_divergence(p, q) > 0.5


def test_kl_stability() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    val = kl_divergence(p, q)
    assert np.isfinite(val)
