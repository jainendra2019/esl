"""
Milestone 7: full paper runs (M4–M6), statistical audit, main vs appendix staging,
``EXPERIMENT_REPORT.md``, captions, and submission readiness notes.

Does not add baselines, experimental families, or algorithm changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from esl.experiments.canonical_io import ensure_manuscript_bundle_layout, write_manifest_json
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.milestone4_paper import run_milestone4_paper
from esl.experiments.milestone5_pack import run_milestone5_pack
from esl.experiments.milestone6_pack import run_milestone6_pack
from esl.experiments.paper_run_contract import validate_frozen_for_full_paper
from esl.experiments.progress_echo import log_step, progress_heartbeat
from esl.experiments.stats_ci import mean_std_ci95

_FROZEN_JSON = Path(__file__).resolve().parent / "configs" / "milestone7_frozen.json"


def _load_frozen() -> dict[str, Any]:
    return json.loads(_FROZEN_JSON.read_text(encoding="utf-8"))


def ensure_milestone7_manuscript_layout(bundle_root: Path) -> None:
    """Extends the standard bundle with main vs appendix paper trees."""
    ensure_manuscript_bundle_layout(bundle_root)
    for sub in (
        "main/figures",
        "main/tables",
        "appendix/figures",
        "appendix/tables",
        "text/main/captions",
        "text/main/summaries",
        "text/appendix/captions",
        "text/appendix/summaries",
        "reports/milestone_7",
    ):
        (bundle_root / sub).mkdir(parents=True, exist_ok=True)


def _copy_globs(src: Path, dst: Path, patterns: tuple[str, ...], *, rel_prefix: str) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for pat in patterns:
        for p in sorted(src.glob(pat)):
            if p.is_file():
                shutil.copy2(p, dst / p.name)
                copied.append(f"{rel_prefix}/{p.name}")
    return copied


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_payoff_vs_mce_filtered(
    rows: list[dict[str, Any]],
    out_png: Path,
    *,
    exclude_experiments: set[str],
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        if str(r.get("experiment", "")) in exclude_experiments:
            continue
        try:
            x = float(r.get("final_mce", "nan"))
            y = float(r.get("focal_mean_payoff_per_round", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(xs, ys, alpha=0.65, s=36, c="C0", edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Final MCE")
    ax.set_ylabel("Focal mean payoff / round")
    ax.set_title("Payoff vs structure quality (excl. matching pennies)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=publication_figure_dpi())
    plt.close(fig)


def _matching_pennies_clean_for_main(m6_long_csv: Path) -> tuple[bool, dict[str, Any]]:
    """
    Heuristic: matching pennies is **main-text eligible** if focal payoffs across seeds are
    not wildly dispersed (CV below threshold and enough seeds).
    """
    rows = [r for r in _read_csv(m6_long_csv) if r.get("experiment") == "matching_pennies"]
    meta: dict[str, Any] = {"n_rows": len(rows), "eligible": False}
    if len(rows) < 3:
        meta["reason"] = "fewer_than_3_matching_pennies_runs"
        return False, meta
    vals = []
    for r in rows:
        try:
            vals.append(float(r["focal_mean_payoff_per_round"]))
        except (KeyError, ValueError):
            continue
    if len(vals) < 3:
        meta["reason"] = "insufficient_numeric_focal_payoffs"
        return False, meta
    arr = np.asarray(vals, dtype=np.float64)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cv = float(s / abs(m)) if abs(m) > 1e-9 else float("inf")
    rng = float(np.max(arr) - np.min(arr))
    meta.update({"mean": m, "std": s, "cv": cv, "range": rng})
    if cv > 0.45:
        meta["reason"] = "high_coefficient_of_variation"
        return False, meta
    if rng > 1.5:
        meta["reason"] = "large_seed_range"
        return False, meta
    meta["reason"] = "ok"
    meta["eligible"] = True
    return True, meta


def _audit_numeric_csv(path: Path, *, value_cols: Sequence[str], min_n: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "warnings": [], "columns": {}}
    rows = _read_csv(path)
    if not rows:
        out["warnings"].append("empty_or_missing")
        return out
    cols = value_cols or [c for c in rows[0].keys() if "_mean" in c or c.endswith("_mean")]
    for col in cols:
        xs = []
        for r in rows:
            try:
                xs.append(float(r.get(col, "nan")))
            except (TypeError, ValueError):
                continue
        xs = [x for x in xs if math.isfinite(x)]
        st = mean_std_ci95(xs) if xs else {"n": 0.0, "mean": float("nan"), "std": float("nan")}
        ci_w = float("nan")
        if int(st.get("n", 0) or 0) >= min_n and math.isfinite(st["mean"]):
            ci_w = float(st.get("ci95_high", 0) - st.get("ci95_low", 0))
        out["columns"][col] = {"n": st.get("n", 0), "ci_width": ci_w}
        if int(st.get("n", 0) or 0) < min_n:
            out["warnings"].append(f"{col}:n_lt_{min_n}")
        if math.isfinite(ci_w) and abs(st.get("mean", 0.0) or 0) > 1e-9 and ci_w > 3 * abs(st["mean"]):
            out["warnings"].append(f"{col}:wide_ci_vs_mean")
    return out


def _load_m4_meta_from_disk(m4_root: Path) -> dict[str, Any]:
    p = m4_root / "aggregate" / "milestone4_run_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"Cannot skip M4: missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def run_milestone7_paper(
    *,
    out_root: Path,
    manuscript_bundle: Path | None,
    smoke: bool = False,
    frozen_path: Path | None = None,
    skip_m4_if_ready: bool = False,
    m5_resume_skip_complete: bool = False,
) -> dict[str, Any]:
    """
    Run M4 → M6 with frozen (or smoke) settings; optionally stage into ``manuscript_bundle``
    with **main/** vs **appendix/** separation and generated narrative files.
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    raw = json.loads((frozen_path or _FROZEN_JSON).read_text(encoding="utf-8"))
    if not smoke:
        validate_frozen_for_full_paper(raw)
    sm = raw.get("smoke_override", {}) if smoke else {}

    m4_cfg = {**raw["milestone4"], **sm.get("milestone4", {})}
    m5_cfg = {**raw["milestone5"], **sm.get("milestone5", {})}
    m6_cfg = {**raw["milestone6"], **sm.get("milestone6", {})}

    seeds4 = tuple(int(x) for x in m4_cfg["seeds"])
    m4_root = out_root / "milestone4"
    seeds5 = tuple(int(x) for x in m5_cfg["seeds"])
    m5_root = out_root / "milestone5"
    seeds6 = tuple(int(x) for x in m6_cfg["seeds"])
    m6_root = out_root / "milestone6"
    progress = not smoke

    with progress_heartbeat("milestone7-paper M4→M6", enabled=progress):
        m4_agg_csv = m4_root / "aggregate" / "milestone4_per_run_metrics.csv"
        if skip_m4_if_ready and m4_agg_csv.is_file() and m4_agg_csv.stat().st_size > 32:
            log_step(
                f"Milestone 4 — skip (reuse existing under {m4_root}); "
                f"aggregate CSV present ({m4_agg_csv.stat().st_size} bytes)",
                enabled=progress,
            )
            m4_meta = _load_m4_meta_from_disk(m4_root)
        else:
            log_step(
                f"Milestone 4 — {len(seeds4)} seeds, rounds sparse/q/init/k = "
                f"{m4_cfg['rounds_sparse']}/{m4_cfg['rounds_q']}/{m4_cfg['rounds_init']}/{m4_cfg['rounds_k']} → {m4_root}",
                enabled=progress,
            )
            m4_meta = run_milestone4_paper(
                m4_root,
                smoke=smoke,
                seeds=seeds4,
                manuscript_bundle=None,
                rounds_sparse=int(m4_cfg["rounds_sparse"]),
                rounds_q=int(m4_cfg["rounds_q"]),
                rounds_init=int(m4_cfg["rounds_init"]),
                rounds_k=int(m4_cfg["rounds_k"]),
                em_restarts=int(m4_cfg.get("em_restarts", 8)),
            )
            log_step("Milestone 4 — complete", enabled=progress)

        log_step(
            f"Milestone 5 — {len(seeds5)} seeds, {m5_cfg['rounds']} rounds, skip_ppo={m5_cfg.get('skip_ppo', True)} → {m5_root}",
            enabled=progress,
        )
        m5_meta = run_milestone5_pack(
            out_root=m5_root,
            manuscript_bundle=None,
            smoke=smoke,
            seeds=seeds5,
            num_rounds=int(m5_cfg["rounds"]),
            include_tft=bool(m5_cfg.get("include_tft", False)),
            skip_ppo=bool(m5_cfg.get("skip_ppo", True)),
            resume_skip_complete=m5_resume_skip_complete,
        )
        log_step("Milestone 5 — complete", enabled=progress)

        log_step(
            f"Milestone 6 — {len(seeds6)} seeds, {m6_cfg['rounds']} rounds, matching_pennies="
            f"{bool(m6_cfg.get('run_matching_pennies', False))} → {m6_root}",
            enabled=progress,
        )
        m6_meta = run_milestone6_pack(
            out_root=m6_root,
            manuscript_bundle=None,
            smoke=smoke,
            seeds=seeds6,
            num_rounds=int(m6_cfg["rounds"]),
            run_matching_pennies=bool(m6_cfg.get("run_matching_pennies", False)),
        )
        log_step("Milestone 6 — complete", enabled=progress)

    with progress_heartbeat("milestone7-paper audit + staging", enabled=progress):
        log_step("Milestone 7 — numeric audit, derived scatter, manuscript staging", enabled=progress)
        audit: dict[str, Any] = {
            "milestone4": {},
            "milestone5": {},
            "milestone6": {},
        }
        for name, rel, cols in [
        (
            "m5_methods",
            m5_root / "aggregate" / "milestone5_method_comparison.csv",
            ("focal_mean_per_round_mean",),
        ),
        (
            "m6_k",
            m6_root / "aggregate" / "m6_k_misspec_aggregate.csv",
            ("focal_mean_payoff_per_round_mean",),
        ),
            (
                "m6_p",
                m6_root / "aggregate" / "m6_p_obs_aggregate.csv",
                ("focal_mean_payoff_per_round_mean",),
            ),
        ]:
            p = rel
            if p.is_file():
                audit["milestone5" if "m5" in name else "milestone6"][name] = _audit_numeric_csv(
                    p,
                    value_cols=cols,
                    min_n=2,
                )

        m6_long = m6_root / "aggregate" / "milestone6_all_runs.csv"
        mp_main, mp_meta = (False, {"reason": "no_csv"})
        if m6_long.is_file():
            mp_main, mp_meta = _matching_pennies_clean_for_main(m6_long)

        # Filtered structure plot for main text (exclude oscillatory regime from scatter)
        main_scatter = out_root / "milestone7_derived" / "m6_payoff_vs_mce_main_text.png"
        main_scatter.parent.mkdir(parents=True, exist_ok=True)
        if m6_long.is_file():
            long_rows: list[dict[str, Any]] = []
            with m6_long.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    long_rows.append(dict(row))
            _plot_payoff_vs_mce_filtered(
                long_rows,
                main_scatter,
                exclude_experiments={"matching_pennies"},
            )

        staged: dict[str, Any] = {"main": [], "appendix": []}
        if manuscript_bundle is not None:
            mb = manuscript_bundle.resolve()
            ensure_milestone7_manuscript_layout(mb)

            # --- Main: recovery + adaptation core + mechanism (no raw MP scatter) ---
            m_main_f = mb / "main/figures"
            m_main_t = mb / "main/tables"
            staged["main"] += _copy_globs(m4_root / "figures", m_main_f, ("*.png",), rel_prefix="main/figures")
            staged["main"] += _copy_globs(m4_root / "aggregate", m_main_t, ("*.csv",), rel_prefix="main/tables")
            staged["main"] += _copy_globs(
                m5_root / "figures", m_main_f, ("milestone5_*.png",), rel_prefix="main/figures"
            )
            staged["main"] += _copy_globs(m5_root / "aggregate", m_main_t, ("*.csv",), rel_prefix="main/tables")
            staged["main"] += _copy_globs(
                m6_root / "figures",
                m_main_f,
                ("m6_k_*.png", "m6_p_*.png", "m6_belief_*.png"),
                rel_prefix="main/figures",
            )
            staged["main"] += _copy_globs(
                m6_root / "aggregate",
                m_main_t,
                ("m6_*aggregate.csv", "milestone6_all_runs.csv", "milestone6_run_meta.json"),
                rel_prefix="main/tables",
            )
            if main_scatter.is_file():
                shutil.copy2(main_scatter, m_main_f / "m6_payoff_vs_mce_main_text.png")
                staged["main"].append("main/figures/m6_payoff_vs_mce_main_text.png")

            # --- Appendix: full M6 scatter + optional MP-only figures ---
            m_apx_f = mb / "appendix/figures"
            m_apx_t = mb / "appendix/tables"
            staged["appendix"] += _copy_globs(
                m6_root / "figures", m_apx_f, ("m6_payoff_vs_mce.png",), rel_prefix="appendix/figures"
            )
            if not mp_main and (m6_root / "figures" / "m6_payoff_vs_mce.png").is_file():
                staged["appendix"].append(
                    "(matching_pennies: full scatter retained in appendix; main uses filtered plot)"
                )

            if mp_main and (m6_root / "figures" / "m6_payoff_vs_mce.png").is_file():
                shutil.copy2(m6_root / "figures" / "m6_payoff_vs_mce.png", m_main_f / "m6_payoff_vs_mce_full_including_mp.png")
                staged["main"].append("main/figures/m6_payoff_vs_mce_full_including_mp.png")

            audit_path = mb / "statistical_audit.json"
            audit_bundle = {
                "milestone4_meta": {k: v for k, v in m4_meta.items() if k != "manuscript_copied"},
                "milestone5_meta": m5_meta,
                "milestone6_meta": m6_meta,
                "matching_pennies_main_text": mp_main,
                "matching_pennies_audit": mp_meta,
                "numeric_spot_checks": audit,
            }
            audit_path.write_text(json.dumps(audit_bundle, indent=2), encoding="utf-8")

            _write_experiment_report(mb, audit_bundle=audit_bundle, mp_main=mp_main)
            _write_captions_and_summaries(mb, mp_main=mp_main, m5_meta=m5_meta)
            _write_readiness(mb, audit_bundle=audit_bundle)

            write_manifest_json(
                mb / "manifests" / "milestone7_bundle_manifest.json",
                {
                    "schema_version": 1,
                    "milestone": "milestone7_paper",
                    "smoke": smoke,
                    "out_root": str(out_root),
                    "staged_main": staged["main"][:50],
                    "staged_main_count": len(staged["main"]),
                    "staged_appendix_count": len(staged["appendix"]),
                    "matching_pennies_in_main": mp_main,
                },
            )
        else:
            staged = {"main": [], "appendix": []}

    return {
        "out_root": str(out_root),
        "smoke": smoke,
        "m4": {k: v for k, v in m4_meta.items() if k != "manuscript_copied"},
        "m5": m5_meta,
        "m6": m6_meta,
        "matching_pennies_main_text": mp_main,
        "matching_pennies_meta": mp_meta,
        "audit": audit,
    }


def _write_experiment_report(mb: Path, *, audit_bundle: dict[str, Any], mp_main: bool) -> None:
    mp_note = (
        "Matching Pennies panels are included in the **main** figure set (cleanliness heuristic passed)."
        if mp_main
        else "Matching Pennies are staged as **appendix-only** (full pooled scatter); the main text uses a filtered payoff–MCE figure excluding that regime."
    )
    mp_row = (
        "| Matching pennies (optional regime) | Oscillatory / non-PD telemetry | **main** (`main/figures/m6_payoff_vs_mce_full_including_mp.png`) |"
        if mp_main
        else "| Matching pennies (optional regime) | Oscillatory / non-PD telemetry | **appendix** (`appendix/figures/m6_payoff_vs_mce.png`) |"
    )
    body = f"""# EXPERIMENT_REPORT — ESL manuscript integration (Milestone 7)

This file maps **paper claims** to **reproducible artifacts** produced by frozen Milestones 4–6 runs.

## Reproduction

- Frozen defaults: ``esl/experiments/configs/milestone7_frozen.json``
- Orchestrator: ``python -m esl.experiments.milestone7_manuscript`` (or ``milestone7-paper`` via runner)
- Raw run roots under the chosen ``--out-root`` (e.g. ``out_root/milestone4`` … ``milestone6``).

## Claim → artifact map

| Claim (short) | Primary evidence | Location |
|---------------|------------------|----------|
| Recovery under sparse observations | MCE vs `p_obs` with CI bands | ``main/figures`` (Milestone 4 sparse), ``main/tables`` ``*_agg.csv`` |
| Q-robustness (prototype cadence) | MCE vs Q | M4 ``q_recovery`` aggregates / figures |
| Init noise & latent K robustness | M4 init-noise and K sweeps | ``main/tables``, ``main/figures`` |
| Strategic adaptation vs baselines | Focal payoff trajectories + method bars | ``main/figures/milestone5_*.png``, ``main/tables/milestone5_*.csv`` |
| Structure ↔ adaptation performance | Payoff vs MCE | ``main/figures/m6_payoff_vs_mce_main_text.png`` |
| Misspecified latent cardinality | Payoff vs `K_model` (metrics `K_true=2`) | ``main/figures/m6_k_misspec_payoff_ci.png``, ``main/tables/m6_k_misspec_aggregate.csv`` |
| Observability stress in adaptation | Payoff vs `p_obs` | ``main/figures/m6_p_obs_payoff_ci.png``, ``main/tables/m6_p_obs_aggregate.csv`` |
| Value of Bayesian beliefs | Uniform-belief ablation | ``main/figures/m6_belief_ablation_bars.png`` |
{mp_row}

## Matching pennies policy

{mp_note}

## Statistical audit snapshot

See ``statistical_audit.json`` at bundle root for seed lists, MP eligibility rationale, and spot-checks on aggregate tables.

## Provenance

Per-run ``config.json`` / ``summary_metrics.json`` remain under ``out_root/milestone*/…``; staged tables are aggregation copies for the paper bundle.
"""
    (mb / "EXPERIMENT_REPORT.md").write_text(body, encoding="utf-8")


def _write_captions_and_summaries(mb: Path, *, mp_main: bool, m5_meta: dict[str, Any]) -> None:
    cap = mb / "text/main/captions"
    summ = mb / "text/main/summaries"
    (mb / "text/appendix/summaries").mkdir(parents=True, exist_ok=True)

    (cap / "fig_m4_sparse_mce.md").write_text(
        "**Figure (M4).** Mean matched cross-entropy (±95% CI) vs observation probability `p_obs` "
        "for ESL and clustering baselines under the recovery flagship protocol.\n",
        encoding="utf-8",
    )
    (cap / "fig_m5_adaptation_focal.md").write_text(
        "**Figure (M5).** Focal-agent mean payoff per environment round (±95% CI across seeds) "
        "for ESL, K-means + BR, FCM + BR"
        + (" (PPO omitted in frozen config)." if m5_meta.get("ppo_skipped") else ", and independent PPO.")
        + "\n",
        encoding="utf-8",
    )
    (cap / "fig_m6_structure_payoff.md").write_text(
        "**Figure (M6).** Focal mean payoff vs final MCE after excluding matching-pennies diagnostics "
        "(main-text scatter).\n",
        encoding="utf-8",
    )
    (summ / "results_m4_recovery.md").write_text(
        "Milestone 4 aggregates quantify recovery stability across seeds for sparse `p_obs`, batch horizon Q, "
        "initialization noise, and latent cardinality; CI bands summarize Monte Carlo variability.\n",
        encoding="utf-8",
    )
    (summ / "results_m5_adaptation.md").write_text(
        "Milestone 5 compares ESL logit best response against clustering + BR and (when enabled) independent PPO "
        "in heterogeneous IPD; bars report cross-seed mean focal payoff with 95% CI.\n",
        encoding="utf-8",
    )
    (summ / "results_m6_mechanism.md").write_text(
        "Milestone 6 links latent-structure quality (MCE vs two-type reference) to focal payoffs, stresses `p_obs`, "
        "and isolates Bayesian belief updates vs uniform beliefs while holding prototype learning on.\n",
        encoding="utf-8",
    )
    apx = "**Appendix.** Full pooled scatter including matching pennies.\n"
    if mp_main:
        apx = "**Main text.** Matching pennies passed stability heuristic; full scatter may accompany supplement.\n"
    (mb / "text/appendix/summaries/results_matching_pennies_placement.md").write_text(apx, encoding="utf-8")


def _write_readiness(mb: Path, *, audit_bundle: dict[str, Any]) -> None:
    warns = []
    for section in audit_bundle.get("numeric_spot_checks", {}).values():
        if isinstance(section, dict):
            for _k, v in section.items():
                if isinstance(v, dict) and v.get("warnings"):
                    warns.extend(v["warnings"])
    status = "READY_WITH_NOTES" if warns else "READY"
    lines = [
        "# Milestone 7 — Submission readiness",
        "",
        f"- **Overall:** {status}",
        f"- **Matching pennies in main text:** {audit_bundle.get('matching_pennies_main_text')}",
        f"- **Heuristic detail:** {json.dumps(audit_bundle.get('matching_pennies_audit', {}))}",
        "",
        "## Statistical audit",
        "",
        "- Spot checks flag very wide CIs relative to the mean on selected aggregate columns.",
        f"- Raw warnings: `{warns}`" if warns else "- No wide-CI warnings on spot-checked aggregates.",
        "",
        "## Before submission",
        "",
        "- Re-run without smoke using frozen JSON horizons.",
        "- If PPO is required in main text, set ``skip_ppo: false`` in ``milestone7_frozen.json`` and ensure ``third_party/PPO-PyTorch`` is present.",
        "- Replace placeholder interpretation with numbers pulled from staged CSVs after your final run.",
        "",
    ]
    (mb / "reports" / "milestone_7" / "READINESS.md").write_text("\n".join(lines), encoding="utf-8")


def main_milestone7(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 7 full paper bundle + manuscript integration")
    p.add_argument("--out-root", type=Path, default=Path("runs/milestone7_paper"))
    p.add_argument("--manuscript-bundle", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="use smoke_override from frozen JSON")
    p.add_argument("--frozen", type=Path, default=None, help="alternate frozen JSON path")
    p.add_argument(
        "--skip-m4-if-ready",
        action="store_true",
        help="reuse existing out_root/milestone4 if aggregate/milestone4_per_run_metrics.csv exists",
    )
    p.add_argument(
        "--m5-resume-skip-complete",
        action="store_true",
        help="skip M5 (seed, method) cells that already have summary_metrics.json + focal_payoff_per_round.csv",
    )
    args = p.parse_args(argv)
    info = run_milestone7_paper(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        smoke=args.smoke,
        frozen_path=args.frozen,
        skip_m4_if_ready=args.skip_m4_if_ready,
        m5_resume_skip_complete=args.m5_resume_skip_complete,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main_milestone7()
