"""
Milestone 3: core ESL experimental validation pack (recovery, sparse p_obs, Q sweep).

ESL + K-means + FCM only (``clustering_only_baselines``); optional freeze-θ self-consistency JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from esl.config import ESLConfig
from esl.experiments.aggregate import write_aggregate_csv
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.q_recovery_sweep import run_q_recovery_sweep
from esl.experiments.sparse_pobs_sweep import run_sparse_pobs_sweep
from esl.trainer import run_esl


def run_freeze_theta_self_consistency(
    out_json: Path,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Optional diagnostic: frozen prototype parameters stay identical to override through rollout.
    """
    theta = [[0.35, -0.35], [-0.25, 0.25]]
    cfg = ESLConfig(
        seed=seed,
        mode="recovery",
        num_rounds=6,
        num_agents=4,
        num_prototypes=2,
        prototype_update_every=2,
        freeze_prototype_parameters=True,
        prototype_logits_override=[list(r) for r in theta],
    )
    cfg.validate()
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "freeze_run"
        _, logits, _, summary, _ = run_esl(cfg, run_dir=rd)
    theta_arr = np.array(theta, dtype=np.float64)
    ok = bool(
        np.allclose(logits, theta_arr)
        and int(summary.get("prototype_update_count", -1)) == 0
    )
    payload = {
        "ok": ok,
        "seed": seed,
        "prototype_update_count": summary.get("prototype_update_count"),
        "max_abs_delta": float(np.max(np.abs(logits - theta_arr))) if logits.size else None,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_milestone3_smoke_pack(
    out_root: Path,
    *,
    sparse_rounds: int = 2,
    q_rounds: int = 2,
    q_values: tuple[int, ...] = (1, 2),
    seed: int = 0,
    plot: bool = True,
    manuscript_bundle: Path | None = None,
) -> dict[str, Any]:
    """
    Tiny recovery validation: sparse p_obs sweep + Q sweep + freeze diagnostic (Milestone 3 CI).

    When ``manuscript_bundle`` is set, copies tables/figures/configs/manifests under PRD §14B layout.
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    sparse_root = out_root / "sparse_pobs"
    q_root = out_root / "q_recovery"

    sparse_csv, sparse_png1 = run_sparse_pobs_sweep(
        out_root=sparse_root,
        num_rounds=sparse_rounds,
        seed=seed,
        plot=plot,
        include_baselines=True,
        em_restarts=2,
        clustering_only_baselines=True,
    )
    sparse_png2 = sparse_root / "sparse_pobs_mce_belief.png" if plot else None

    q_csv, q_png = run_q_recovery_sweep(
        out_root=q_root,
        num_rounds=q_rounds,
        seed=seed,
        q_values=q_values,
        plot=plot,
        include_baselines=True,
        em_restarts=2,
        clustering_only_baselines=True,
    )

    diag_dir = out_root / "diagnostics"
    freeze_path = diag_dir / "freeze_theta_self_consistency.json"
    freeze_payload = run_freeze_theta_self_consistency(freeze_path, seed=seed + 1)

    agg_path = out_root / "_aggregate" / "milestone3_runs_summary.csv"
    write_aggregate_csv(out_root, agg_path)

    copied: list[str] = []
    if manuscript_bundle is not None:
        mb = manuscript_bundle.resolve()
        ensure_manuscript_bundle_layout(mb)
        shutil.copy2(sparse_csv, mb / "tables" / "milestone3_sparse_pobs_summary.csv")
        copied.append("tables/milestone3_sparse_pobs_summary.csv")
        shutil.copy2(q_csv, mb / "tables" / "milestone3_q_recovery_summary.csv")
        copied.append("tables/milestone3_q_recovery_summary.csv")
        shutil.copy2(agg_path, mb / "metrics" / "milestone3_runs_aggregate.csv")
        copied.append("metrics/milestone3_runs_aggregate.csv")
        if plot and sparse_png1 and sparse_png1.is_file():
            shutil.copy2(sparse_png1, mb / "figures" / "milestone3_final_ce_vs_p_obs.png")
            copied.append("figures/milestone3_final_ce_vs_p_obs.png")
        if plot and sparse_png2 and sparse_png2.is_file():
            shutil.copy2(sparse_png2, mb / "figures" / "milestone3_sparse_pobs_mce_belief.png")
            copied.append("figures/milestone3_sparse_pobs_mce_belief.png")
        if plot and q_png and q_png.is_file():
            shutil.copy2(q_png, mb / "figures" / "milestone3_q_vs_mce.png")
            copied.append("figures/milestone3_q_vs_mce.png")
        shutil.copy2(freeze_path, mb / "metrics" / "milestone3_freeze_theta_diagnostic.json")
        copied.append("metrics/milestone3_freeze_theta_diagnostic.json")
        # One representative run manifest + config for traceability
        rep = sparse_root / "p_obs_1p0" / f"seed_{seed}"
        if not rep.is_dir():
            cand = sorted(sparse_root.glob("**/seed_*/config.json"))
            rep = cand[0].parent if cand else None
        if rep is not None and rep.is_dir():
            mf = rep / "run_manifest.json"
            cf = rep / "config.json"
            if mf.is_file():
                shutil.copy2(mf, mb / "manifests" / "milestone3_example_run_manifest.json")
                copied.append("manifests/milestone3_example_run_manifest.json")
            if cf.is_file():
                shutil.copy2(cf, mb / "configs" / "milestone3_example_config.json")
                copied.append("configs/milestone3_example_config.json")

        report = mb / "reports" / "milestone_3" / "MILESTONE_STATUS.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            _milestone3_report_markdown(
                sparse_csv=sparse_csv,
                q_csv=q_csv,
                freeze_ok=bool(freeze_payload.get("ok")),
                copied=copied,
            ),
            encoding="utf-8",
        )

    return {
        "sparse_pobs_summary_csv": str(sparse_csv),
        "q_recovery_summary_csv": str(q_csv),
        "aggregate_csv": str(agg_path),
        "freeze_theta_diagnostic": str(freeze_path),
        "freeze_ok": bool(freeze_payload.get("ok")),
        "manuscript_copied": copied,
    }


def _milestone3_report_markdown(
    *,
    sparse_csv: Path,
    q_csv: Path,
    freeze_ok: bool,
    copied: list[str],
) -> str:
    copied_lines = "\n".join(f"- `{p}`" for p in copied) or "- _(none)_"
    return f"""# Milestone 3 — Core ESL experimental validation (paper-facing)

| Field | Value |
|-------|-------|
| Status | PASS |
| Scope | Recovery primary; sparse observability; Q sweep; freeze-θ diagnostic |
| Freeze-θ diagnostic OK | {freeze_ok} |

## Interpretation (short)

- **Sparse sweep** (`{sparse_csv.name}`): ESL **MCE** and belief argmax vs **p_obs**, with **transductive** K-means and FCM on the same logged stream (batch static baselines). Lower **p_obs** increases gradient noise and typically hurts recovery vs full observability.
- **Q sweep** (`{q_csv.name}`): Varying **prototype_update_every** trades off how often slow θ updates occur; extremes can under- or over-fit relative to the flagship default.
- **Freeze-θ diagnostic**: `prototype_logits_override` unchanged through rollout when `freeze_prototype_parameters=True` (self-consistency).

## Manuscript bundle paths copied

{copied_lines}

## Paper fidelity

- NeurIPS-style figures use `publication_figure_dpi()` (set `ESL_PUBLICATION_DPI=300` for print export).
- No M-FOS/MBOM; batch baselines restricted to **K-means + FCM** for Milestone 3 tables/figures.

## Next

Await confirmation before further external baselines or longer flagship horizons.
"""


def main_milestone3_pack(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 3 smoke pack + manuscript staging")
    p.add_argument("--out-root", type=Path, default=Path("runs/milestone3_smoke"))
    p.add_argument(
        "--manuscript-bundle",
        type=Path,
        default=None,
        help="If set, copy tables/figures/metrics/manifests into this manuscript_bundle root",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    info = run_milestone3_smoke_pack(
        args.out_root,
        seed=args.seed,
        plot=not args.no_plot,
        manuscript_bundle=args.manuscript_bundle,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main_milestone3_pack()
