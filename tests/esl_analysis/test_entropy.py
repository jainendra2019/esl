from __future__ import annotations

import numpy as np
import pytest

from esl_analysis.metrics.entropy import belief_entropy


def test_entropy_bounds() -> None:
    assert belief_entropy(np.array([0.5, 0.5])) > 0.6
    assert belief_entropy(np.array([1.0, 0.0])) == pytest.approx(0.0)
