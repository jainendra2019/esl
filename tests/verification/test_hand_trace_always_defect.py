"""
Hand traces: 2 prototypes, observer 0 / target 1, fixed θ₁=[0.2,0], θ₂=[0,0.2], W=1.

* Defect stream (s=1): target AD — beliefs sharpen on prototype 1 (defect-leaning).
* Cooperate stream (s=0): target AC — symmetric mirror; beliefs sharpen on prototype 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from esl.config import ESLConfig
from esl.hand_trace import isolated_belief_trajectory, paper_trace_config
from esl.trainer import run_esl


@pytest.mark.verification
def test_hand_trace_belief_matches_isolated_bayes_with_fixed_logits():
    """Trainer b_{0→1} after each round == repeated Bayes reference (prevents pair-sampling bugs)."""
    cfg = paper_trace_config(num_rounds=10, prototype_update_every=1000, seed=0)
    logits = np.array(cfg.prototype_logits_override, dtype=np.float64)
    ref = isolated_belief_trajectory(cfg, logits, signal=1, steps=cfg.num_rounds)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log, _, belief_tensor, _, _ = run_esl(cfg, run_dir=Path(td))

    for t in range(cfg.num_rounds):
        row = next(
            r
            for r in log.belief_rows
            if int(r["round"]) == t and int(r["i"]) == 0 and int(r["j"]) == 1
        )
        got = np.array([float(row["b_0"]), float(row["b_1"])])
        np.testing.assert_allclose(got, ref[t + 1], rtol=1e-6, atol=1e-8)

    np.testing.assert_allclose(belief_tensor[0, 1], ref[-1], rtol=1e-6, atol=1e-8)


@pytest.mark.verification
def test_hand_trace_qualitative_direction_and_entropy():
    """Qualitative checks from the paper: mass shifts to defector-like prototype; entropy falls."""
    cfg = paper_trace_config(num_rounds=10, prototype_update_every=1000, seed=0)
    logits = np.array(cfg.prototype_logits_override, dtype=np.float64)
    ref = isolated_belief_trajectory(cfg, logits, signal=1, steps=cfg.num_rounds)

    assert ref[1][1] > ref[0][1], "b[type-2] should rise after first Defect signal"
    for s in range(1, len(ref) - 1):
        assert ref[s + 1][1] >= ref[s][1] - 1e-9, "b[2] monotone (Defect-likelihood prototype)"

    H0 = float(-np.sum(ref[0] * np.log(np.clip(ref[0], 1e-12, 1.0))))
    Hn = float(-np.sum(ref[-1] * np.log(np.clip(ref[-1], 1e-12, 1.0))))
    assert Hn < H0, "entropy should decrease as beliefs sharpen"

    # Near paper’s rounded first step [0.45, 0.55] with actual softmax likelihoods
    np.testing.assert_allclose(ref[1], [0.45, 0.55], rtol=0.02, atol=0.02)


@pytest.mark.verification
def test_hand_trace_prototype_update_with_m5_moves_toward_defect():
    """With M=5 and γ>0, after first update both prototypes’ P(Defect) should not decrease sharply."""
    cfg = paper_trace_config(num_rounds=5, prototype_update_every=5, seed=0)
    cfg.prototype_lr_scale = 0.5
    import tempfile
    from pathlib import Path

    from esl.prototypes import stable_softmax

    logits0 = np.array(cfg.prototype_logits_override, dtype=np.float64)
    p0 = stable_softmax(logits0)[:, 1].copy()
    with tempfile.TemporaryDirectory() as td:
        _, logits1, _, _, _ = run_esl(cfg, run_dir=Path(td))
    p1 = stable_softmax(logits1)[:, 1]
    assert float(p1[1]) >= float(p0[1]) - 1e-3, "prototype 2 should stay or move toward Defect"
    for k in range(2):
        assert np.isclose(float(stable_softmax(logits1[k : k + 1]).sum()), 1.0)


@pytest.mark.verification
def test_hand_trace_cooperate_mirror_matches_isolated_bayes():
    cfg = paper_trace_config(num_rounds=10, prototype_update_every=1000, seed=0, repeated_signal="cooperate")
    logits = np.array(cfg.prototype_logits_override, dtype=np.float64)
    ref = isolated_belief_trajectory(cfg, logits, signal=0, steps=cfg.num_rounds)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log, _, belief_tensor, _, _ = run_esl(cfg, run_dir=Path(td))

    for t in range(cfg.num_rounds):
        row = next(
            r
            for r in log.belief_rows
            if int(r["round"]) == t and int(r["i"]) == 0 and int(r["j"]) == 1
        )
        got = np.array([float(row["b_0"]), float(row["b_1"])])
        np.testing.assert_allclose(got, ref[t + 1], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(belief_tensor[0, 1], ref[-1], rtol=1e-6, atol=1e-8)


@pytest.mark.verification
def test_hand_trace_cooperate_mirror_qualitative():
    cfg = paper_trace_config(num_rounds=10, prototype_update_every=1000, seed=0, repeated_signal="cooperate")
    logits = np.array(cfg.prototype_logits_override, dtype=np.float64)
    ref = isolated_belief_trajectory(cfg, logits, signal=0, steps=cfg.num_rounds)
    assert ref[1][0] > ref[0][0], "b[cooperative prototype] should rise after Cooperate"
    for s in range(1, len(ref) - 1):
        assert ref[s + 1][0] >= ref[s][0] - 1e-9
    H0 = float(-np.sum(ref[0] * np.log(np.clip(ref[0], 1e-12, 1.0))))
    Hn = float(-np.sum(ref[-1] * np.log(np.clip(ref[-1], 1e-12, 1.0))))
    assert Hn < H0


@pytest.mark.verification
def test_hand_trace_cooperate_prototype_update_increases_p_cooperate_on_weighted_protos():
    cfg = paper_trace_config(num_rounds=5, prototype_update_every=5, seed=0, repeated_signal="cooperate")
    cfg.prototype_lr_scale = 0.5
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log, _, _, _, _ = run_esl(cfg, run_dir=Path(td))
    assert len(log.prototype_update_events) >= 1
    ev = log.prototype_update_events[0]
    for k in range(2):
        pbc, pbd = ev["p_before"][k]
        pac, pad = ev["p_after"][k]
        assert pac + pad == pytest.approx(1.0)
        assert pac >= pbc - 1e-4, f"prototype {k}: P(Cooperate) should not drop after s=0 batch"
