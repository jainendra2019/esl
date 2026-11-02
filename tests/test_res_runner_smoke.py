from __future__ import annotations

from pathlib import Path

from esl.res.benchmark import generate_2x2_suite, write_suite
from esl.res.runner import TIER1_METHODS, game_spec_to_task, run_res_cell, run_res_m1
from esl.experiments.figure1_learning_curves import _task_payoff
import numpy as np


def test_game_spec_to_task_payoffs() -> None:
    spec = generate_2x2_suite(seed=0, per_family=1)[0]
    task = game_spec_to_task(spec)
    pay = _task_payoff(task)
    assert pay.row.shape == (2, 2)


def test_run_res_cell_smoke() -> None:
    spec = [s for s in generate_2x2_suite(seed=0, per_family=1) if s.family == "cooperative"][0]
    out = run_res_cell(spec=spec, method="ESL", seed=0, horizon=40)
    assert "psr" in out["summary"]
    assert out["summary"]["instance_id"] == spec.instance_id


def test_run_res_m1_smoke(tmp_path: Path) -> None:
    specs = generate_2x2_suite(seed=11, per_family=2)
    write_suite(specs, tmp_path)
    gate = run_res_m1(
        out_dir=tmp_path / "out",
        benchmark_root=tmp_path,
        seeds=(0,),
        horizon=30,
        methods=("ESL", "SA-FP", "Fixed-K Bayesian"),
    )
    assert gate["n_summaries"] == 6 * 3  # 3 families × 2 instances × 3 methods × 1 seed
    assert (tmp_path / "out" / "res_m1_report.md").is_file()
    assert set(TIER1_METHODS) >= {"ESL", "SA-FP", "Fixed-K Bayesian"}


def test_categorical_cell_a3_smoke() -> None:
    from esl.res.benchmark import generate_3x3_suite
    from esl.res.categorical_cell import run_categorical_cell

    spec = generate_3x3_suite(seed=0, per_family=1)[0]
    task = game_spec_to_task(spec)
    out = run_categorical_cell(task=task, method="ESL", seed=0, horizon=40, num_actions=3)
    assert out["summary"]["num_actions"] == 3
    assert np.isfinite(float(out["summary"]["psr"]))


def test_run_res_m2_a3_smoke(tmp_path: Path) -> None:
    from esl.res.benchmark import generate_3x3_suite

    specs = generate_3x3_suite(seed=7, per_family=1)
    write_suite(specs, tmp_path)
    gate = run_res_m1(
        out_dir=tmp_path / "out3",
        benchmark_root=tmp_path,
        seeds=(0,),
        horizon=25,
        methods=("ESL", "FP", "SOM"),
        num_actions=3,
    )
    assert gate["num_actions"] == 3
    assert gate["n_summaries"] == 3 * 3  # 3 families × 1 × 3 methods
    assert (tmp_path / "out3" / "res_m2_a3_report.md").is_file()


def test_visibility_and_noise_smoke() -> None:
    from esl.res.categorical_cell import run_categorical_cell

    spec = generate_2x2_suite(seed=0, per_family=1)[0]
    task = game_spec_to_task(spec)
    full = run_categorical_cell(
        task=task, method="Fixed-K Bayesian", seed=0, horizon=50, num_actions=2, p_obs=1.0
    )
    none = run_categorical_cell(
        task=task, method="Fixed-K Bayesian", seed=0, horizon=50, num_actions=2, p_obs=0.0
    )
    noisy = run_categorical_cell(
        task=task, method="ESL", seed=0, horizon=50, num_actions=2, obs_noise=0.2
    )
    assert all(int(r["observed"]) == 0 for r in none["trajectory"])
    assert all(int(r["observed"]) == 1 for r in full["trajectory"])
    assert np.isfinite(float(noisy["summary"]["psr"]))
    # With no observations, Fixed-K beliefs stay uniform ⇒ LTE stays at -log(1/K)
    ltes = [float(r["lte"]) for r in none["trajectory"] if r["lte"] != ""]
    assert ltes and abs(ltes[-1] - float(-np.log(1.0 / 3.0))) < 1e-9
