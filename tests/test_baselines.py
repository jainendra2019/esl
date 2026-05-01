"""Offline baselines package: smoke and recovery sanity."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from esl.baselines.evaluate import write_baselines_summary
from esl.config import ESLConfig
from esl.trainer import run_esl


def test_baselines_run_all_after_esl_with_observation_log():
    cfg = ESLConfig(
        seed=21,
        num_rounds=120,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=3,
        init_noise=0.06,
        symmetric_init=False,
        log_interaction_observations=True,
        force_agent_true_types=[0, 0, 1, 1],
    )
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        run_esl(cfg, run_dir=run_dir)
        out = write_baselines_summary(run_dir, em_restarts=4)
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
    assert "kmeans_mce" in data
    assert "fcm_mce" in data
    assert "em_conditional_mce" in data
    assert "em_marginal_mce" in data
    assert "oracle_mce" in data
    assert data["wall_time_sec_kmeans"] >= 0.0
    for key in (
        "kmeans_mce",
        "fcm_mce",
        "em_conditional_mce",
        "em_marginal_mce",
        "oracle_mce",
    ):
        assert isinstance(data[key], (int, float))
        assert data[key] == data[key]  # not NaN
    assert data["n_observed_w_positive"] == data["n_observation_rows_total"]
    assert data.get("esl_mce_summary_matches_trajectory") is True
    assert data.get("esl_mce_recomputed_from_trajectory") is not None


def test_prefix_protocol_summary_and_plot():
    cfg = ESLConfig(
        seed=44,
        num_rounds=25,
        num_agents=4,
        num_prototypes=2,
        observability="full",
        prototype_update_every=2,
        log_interaction_observations=True,
        force_ordered_pair=(0, 1),
        force_agent_true_types=[0, 0, 1, 1],
    )
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "r"
        run_esl(cfg, run_dir=run_dir)
        pref = 8
        p = write_baselines_summary(run_dir, prefix_max_round=pref, em_restarts=2)
        assert p.name == f"baselines_summary_prefix_round_{pref}.json"
        data = json.loads(p.read_text(encoding="utf-8"))
    assert data["comparison_protocol"] == "prefix_matched_budget"
    assert data["prefix_max_round"] == pref
    assert data["n_observation_rows_total"] < data["n_observation_rows_full_run"]
    assert data.get("esl_mce_recomputed_from_trajectory") is not None


def test_kmeans_pp_init_degenerate_points():
    """Identical rows → zero distances; init must not pass invalid probs to rng.choice."""
    import numpy as np
    from numpy.random import default_rng

    from esl.baselines.kmeans_actions import _init_centers_kmeans_pp

    x = np.ones((15, 2), dtype=np.float64)
    rng = default_rng(42)
    centers = _init_centers_kmeans_pp(x, k=3, rng=rng)
    assert centers.shape == (3, 2)
    assert np.allclose(centers, 1.0)
