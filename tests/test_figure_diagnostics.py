"""Figure diagnostics smoke test."""

from __future__ import annotations

from pathlib import Path

from esl.experiments.figure_diagnostics import run_figure_diagnostics
from esl.experiments.milestone7_manuscript import ensure_milestone7_manuscript_layout


def _write_csv(path: Path, header: str, row: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + row + "\n", encoding="utf-8")


def test_figure_diagnostics_writes_report(tmp_path: Path) -> None:
    mb = tmp_path / "mb"
    ensure_milestone7_manuscript_layout(mb)
    mt = mb / "main/tables"
    _write_csv(
        mt / "sparse_pobs_agg.csv",
        "p_obs,final_mce_mean,final_mce_std,final_mce_ci95_low,final_mce_ci95_high,final_mce_n,kmeans_mce_mean",
        "1.0,0.5,0.1,0.4,0.6,10,0.8",
    )
    _write_csv(
        mt / "q_recovery_agg.csv",
        "prototype_update_every_q,final_mce_mean,final_mce_std,final_mce_ci95_low,final_mce_ci95_high,final_mce_n,kmeans_mce_mean",
        "5,0.5,0.1,0.4,0.6,10,0.8",
    )
    _write_csv(
        mt / "milestone5_method_comparison.csv",
        "method,focal_mean_per_round_mean,focal_mean_per_round_std,focal_mean_per_round_ci95_low,focal_mean_per_round_ci95_high,focal_mean_per_round_n",
        "esl,1.5,0.1,1.3,1.7,10\nclustering_kmeans,1.4,0.1,1.2,1.6,10",
    )
    _write_csv(
        mt / "milestone5_long.csv",
        "seed,method,focal_mean_payoff_per_round",
        "0,esl,1.5\n0,clustering_kmeans,1.4\n1,esl,1.6\n1,clustering_kmeans,1.3",
    )
    _write_csv(
        mt / "m6_k_misspec_aggregate.csv",
        "num_prototypes,focal_mean_payoff_per_round_mean,focal_mean_payoff_per_round_std,focal_mean_payoff_per_round_ci95_low,focal_mean_payoff_per_round_ci95_high,focal_mean_payoff_per_round_n",
        "2,1.5,0.1,1.3,1.7,10",
    )
    fig = mb / "main/figures"
    fig.mkdir(parents=True, exist_ok=True)
    (fig / "fig1.png").write_bytes(b"x" * 9000)

    info = run_figure_diagnostics(mb, min_seeds=3, min_fig_bytes=8000)
    assert (mb / "reports" / "figure_diagnostics.md").is_file()
    assert "fig1" in (mb / "reports" / "figure_diagnostics.md").read_text(encoding="utf-8")
    assert info["report"].endswith("figure_diagnostics.md")
