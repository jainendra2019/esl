from __future__ import annotations

import numpy as np

from esl_analysis.analysis.self_consistency import compute_kl_trajectory, empirical_policy


def test_empirical_policy_excludes_current_action() -> None:
    actions = [0, 0, 1, 1]
    got = empirical_policy(actions, t=3, window=2, action_dim=2)
    assert np.allclose(got, np.array([0.5, 0.5]))


def test_kl_trajectory_is_finite_and_windowed() -> None:
    rows = []
    for t, action in enumerate([0, 0, 1, 1, 0, 1]):
        rows.append(
            {
                "round": t,
                "pred_p_coop": 0.6,
                "pred_p_defect": 0.4,
                "opponent_action": action,
            }
        )
    curve = compute_kl_trajectory(rows, window=2)
    assert curve.shape[0] == 4
    assert np.isfinite(curve).all()
