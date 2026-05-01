from __future__ import annotations

import numpy as np

from esl_analysis.analysis.ambiguity import extract_entropy_trajectory, posterior_accuracy


def test_accuracy_computation() -> None:
    b = np.array([0.1, 0.9])
    assert posterior_accuracy(b, 1) == 1


def test_entropy_trajectory_aligns_to_switch_tau() -> None:
    rows = []
    for tau in range(3):
        rows.append(
            {
                "switch_id": 0,
                "switch_round": 10,
                "tau": tau,
                "belief": "[0.5, 0.5]",
                "true_type": 0,
            }
        )
    curve = extract_entropy_trajectory(rows, switch_time=10, t_window=2)
    assert curve.shape[0] == 2
    assert np.isfinite(curve).all()
