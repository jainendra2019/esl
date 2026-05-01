from __future__ import annotations

import pytest

from esl_analysis.analysis.switch_difficulty import assign_bin, compute_switch_difficulty
from esl_analysis.pipeline import CONTROLLED_SHIFTS, controlled_shift_js
from esl_analysis.metrics.regret import compute_psr_per_switch


def _row(t: int, *, switch_id: int = -1, switch_round: int | str = "", p: float = 0.5, gap: float = 1.0) -> dict:
    return {
        "round": t,
        "tau": t - int(switch_round) if switch_round != "" else 999,
        "opponent_id": 1,
        "switch_id": switch_id,
        "switch_round": switch_round,
        "true_p_coop": p,
        "true_p_defect": 1.0 - p,
        "oracle_payoff": gap,
        "focal_payoff": 0.0,
        "psr_gap": gap,
    }


def test_switch_bin_assignment() -> None:
    assert assign_bin(0.1) == "low"
    assert assign_bin(0.5) == "medium"
    assert assign_bin(1.2) == "high"


def test_controlled_shift_js_ordering() -> None:
    vals = {label: controlled_shift_js(*shift) for label, shift in CONTROLLED_SHIFTS.items()}
    assert vals["small"] < vals["medium"] < vals["large"]


def test_switch_windows_do_not_mix_pre_post() -> None:
    rows = [_row(t, p=0.1) for t in range(5)]
    rows += [_row(5, switch_id=0, switch_round=5, p=0.9)]
    rows += [_row(t, switch_id=0, switch_round=5, p=0.9) for t in range(6, 10)]
    switches, excluded = compute_switch_difficulty(rows, w_pre=5, w_post=5)
    assert excluded == 0
    assert len(switches) == 1
    assert switches[0]["difficulty"] > 0.3


def test_psr_per_switch_sums_post_switch_horizon() -> None:
    rows = [_row(t, switch_id=0, switch_round=5, gap=2.0) for t in range(5, 9)]
    assert compute_psr_per_switch(rows, switch_time=5, horizon=3) == pytest.approx(8.0)
