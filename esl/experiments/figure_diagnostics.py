"""
Figure diagnostics for manuscript_bundle (validation only).

Reads staged aggregates and long-form CSVs after Milestone 7 (and optional fig1–fig5
PNG sizes). Does not run experiments, alter algorithms, or edit figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from esl.experiments.canonical_io import write_manifest_json
from esl.experiments.stats_ci import mean_std_ci95

# Heuristic thresholds (tune for your venue / scale)
DEFAULT_MIN_SEEDS = 10
DEFAULT_MIN_FIG_BYTES = 8000
CI_RATIO_WARN = 1.5
WEAK_PAYOFF_COHENS_D = 0.2
WEAK_MCE_ABS = 0.03


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str | None) -> float:
    if x is None or str(x).strip() == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def _ci_ratio(mean: float, lo: float, hi: float) -> float:
    if not math.isfinite(mean) or abs(mean) < 1e-12:
        return float("inf") if (hi - lo) > 0 else 0.0
    return (hi - lo) / abs(mean)


def _min_n_m4(rows: list[dict[str, str]]) -> float:
    ns = []
    for r in rows:
        n = _f(r.get("final_mce_n"))
        if math.isfinite(n) and n > 0:
            ns.append(n)
    return min(ns) if ns else 0.0


def _m4_section(mb: Path, *, min_seeds: int) -> dict[str, Any]:
    mt = mb / "main/tables"
    sparse = _read_csv(mt / "sparse_pobs_agg.csv")
    qrec = _read_csv(mt / "q_recovery_agg.csv")
    flags: list[str] = []
    ci_notes: list[str] = []
    eff_notes: list[str] = []

    for label, rows, xk in (
        ("sparse p_obs", sparse, "p_obs"),
        ("Q recovery", qrec, "prototype_update_every_q"),
    ):
        if not rows:
            flags.append(f"M4 {label}: missing aggregate CSV")
            continue
        nmin = _min_n_m4(rows)
        if nmin < min_seeds:
            flags.append(f"M4 {label}: ESL final_mce_n minimum is {int(nmin)} (< {min_seeds} seeds target)")
        for r in rows:
            xm = r.get(xk, "")
            fm = _f(r.get("final_mce_mean"))
            lo = _f(r.get("final_mce_ci95_low"))
            hi = _f(r.get("final_mce_ci95_high"))
            km = _f(r.get("kmeans_mce_mean"))
            ratio = _ci_ratio(fm, lo, hi)
            if math.isfinite(ratio) and ratio > CI_RATIO_WARN:
                ci_notes.append(f"M4 {label} x={xm}: ESL MCE CI/mean={ratio:.2f} (wide)")
            if math.isfinite(fm) and math.isfinite(km) and abs(fm - km) < WEAK_MCE_ABS:
                eff_notes.append(
                    f"M4 {label} x={xm}: ESL vs K-means MCE are close (|Δ| < {WEAK_MCE_ABS}: ESL {fm:.4f}, KM {km:.4f})"
                )

    per_run = _read_csv(mt / "milestone4_per_run_metrics.csv")
    smoke_hint = ""
    if per_run:
        seeds = {int(_f(r.get("seed"))) for r in per_run if math.isfinite(_f(r.get("seed")))}
        rounds = {_f(r.get("num_rounds_executed")) for r in per_run if r.get("num_rounds_executed")}
        if rounds and max(r for r in rounds if math.isfinite(r)) < 500:
            smoke_hint = "Per-run `num_rounds_executed` suggests smoke-scale horizons (<500)."
        if len(seeds) < min_seeds:
            flags.append(f"M4 per-run: only {len(seeds)} distinct seeds in milestone4_per_run_metrics.csv")

    return {
        "flags": flags,
        "ci_notes": ci_notes,
        "eff_notes": eff_notes,
        "smoke_hint": smoke_hint,
        "has_sparse": bool(sparse),
        "has_q": bool(qrec),
    }


def _paired_m5_deltas(long_rows: list[dict[str, str]], base: str, treat: str) -> list[float]:
    by_seed: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for r in long_rows:
        m = (r.get("method") or "").strip()
        sd = int(_f(r.get("seed")))
        if not math.isfinite(sd):
            continue
        y = _f(r.get("focal_mean_payoff_per_round"))
        if not math.isfinite(y):
            continue
        by_seed[sd].append((m, y))
    deltas: list[float] = []
    for sd, pairs in by_seed.items():
        d = dict(pairs)
        if base in d and treat in d:
            deltas.append(d[treat] - d[base])
    return deltas


def _m5_section(mb: Path, *, min_seeds: int) -> dict[str, Any]:
    mt = mb / "main/tables"
    cmp_rows = _read_csv(mt / "milestone5_method_comparison.csv")
    long_rows = _read_csv(mt / "milestone5_long.csv")
    flags: list[str] = []
    ci_notes: list[str] = []
    eff_notes: list[str] = []

    for r in cmp_rows:
        method = (r.get("method") or "").strip()
        m = _f(r.get("focal_mean_per_round_mean"))
        lo = _f(r.get("focal_mean_per_round_ci95_low"))
        hi = _f(r.get("focal_mean_per_round_ci95_high"))
        n = _f(r.get("focal_mean_per_round_n"))
        ratio = _ci_ratio(m, lo, hi)
        if math.isfinite(n) and n < min_seeds:
            flags.append(f"M5 method `{method}`: n={int(n)} (< {min_seeds})")
        if math.isfinite(ratio) and ratio > CI_RATIO_WARN:
            ci_notes.append(f"M5 `{method}`: focal payoff CI/mean={ratio:.2f}")

    esl_row = next((r for r in cmp_rows if (r.get("method") or "").strip() == "esl"), None)
    km_row = next((r for r in cmp_rows if (r.get("method") or "").strip() == "clustering_kmeans"), None)
    if esl_row and km_row:
        d_mean = _f(esl_row.get("focal_mean_per_round_mean")) - _f(km_row.get("focal_mean_per_round_mean"))
        eff_notes.append(f"M5 ESL − K-means (aggregate mean payoff): {d_mean:+.4f}")

    deltas = _paired_m5_deltas(long_rows, "clustering_kmeans", "esl")
    if len(deltas) >= 2:
        st = mean_std_ci95(deltas)
        pooled = float(np.std(np.asarray(deltas, dtype=np.float64), ddof=1) or 1e-9)
        cohen_d = float(st["mean"] / pooled) if pooled > 1e-12 else float("nan")
        eff_notes.append(
            f"M5 paired ESL−KMeans across seeds: mean Δ={st['mean']:.4f}, 95% CI [{st['ci95_low']:.4f}, {st['ci95_high']:.4f}], n={int(st['n'])}"
        )
        if abs(cohen_d) < WEAK_PAYOFF_COHENS_D:
            flags.append(f"M5: |Cohen's d| for paired payoff gain vs K-means is small ({cohen_d:.3f} < {WEAK_PAYOFF_COHENS_D})")
        signs = sum(1 for x in deltas if x > 0)
        if signs not in (0, len(deltas)) and len(deltas) >= 3:
            flags.append(
                f"M5: inconsistent sign of ESL vs K-means gain across seeds ({signs}/{len(deltas)} positive)"
            )
    elif long_rows:
        flags.append("M5: could not pair ESL vs clustering_kmeans by seed (missing rows).")

    return {"flags": flags, "ci_notes": ci_notes, "eff_notes": eff_notes, "has_cmp": bool(cmp_rows)}


def _m6_section(mb: Path, *, min_seeds: int) -> dict[str, Any]:
    mt = mb / "main/tables"
    apx_t = mb / "appendix/tables"
    flags: list[str] = []
    ci_notes: list[str] = []

    for name, label in (
        ("m6_k_misspec_aggregate.csv", "K misspec"),
        ("m6_p_obs_aggregate.csv", "p_obs stress"),
        ("m6_belief_ablation_aggregate.csv", "belief ablation"),
    ):
        p = mt / name if (mt / name).is_file() else apx_t / name
        rows = _read_csv(p)
        if not rows:
            flags.append(f"M6 {label}: missing `{name}` in main/tables and appendix/tables")
            continue
        for r in rows:
            keys = [k for k in r if k.endswith("_mean") and "focal_mean_payoff" in k]
            if not keys:
                keys = [k for k in r if k.endswith("_mean")]
            for mk in keys[:1]:
                m = _f(r.get(mk))
                base = mk[: -len("_mean")]
                lo = _f(r.get(f"{base}_ci95_low"))
                hi = _f(r.get(f"{base}_ci95_high"))
                n = _f(r.get(f"{base}_n"))
                ratio = _ci_ratio(m, lo, hi)
                if math.isfinite(n) and n < min_seeds:
                    flags.append(f"M6 {label}: n={int(n)} (< {min_seeds})")
                if math.isfinite(ratio) and ratio > CI_RATIO_WARN:
                    ci_notes.append(f"M6 {label}: CI/mean={ratio:.2f} on `{mk}`")

    long_p = mt / "milestone6_all_runs.csv"
    if not long_p.is_file():
        long_p = apx_t / "milestone6_all_runs.csv"
    long_rows = _read_csv(long_p)
    if long_rows:
        by_exp: dict[str, list[float]] = defaultdict(list)
        for r in long_rows:
            exp = (r.get("experiment") or "").strip()
            fp = _f(r.get("focal_mean_payoff_per_round"))
            if exp and math.isfinite(fp):
                by_exp[exp].append(fp)
        for exp, vals in by_exp.items():
            if len(vals) >= 2:
                v = float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))
                if v > 0.5:
                    flags.append(f"M6 experiment `{exp}`: high cross-seed std on focal payoff ({v:.3f})")

    return {"flags": flags, "ci_notes": ci_notes, "has_long": bool(long_rows)}


def _audit_smoke(mb: Path) -> bool | None:
    p = mb / "statistical_audit.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return bool(data.get("milestone4_meta", {}).get("smoke"))
    except (json.JSONDecodeError, OSError):
        return None


def _figure_decisions(mb: Path, *, min_fig_bytes: int) -> list[dict[str, Any]]:
    specs = [
        ("fig1", "sparse_pobs_mce_ci.png", "M4 sparse observability"),
        ("fig2", "q_recovery_mce_ci.png", "M4 Q / batch cadence"),
        ("fig3", "milestone5_adaptation_focal_payoff.png", "M5 adaptation trajectories"),
        ("fig4", "milestone5_method_comparison_bars.png", "M5 method comparison"),
        ("fig5", "m6_payoff_vs_mce_main_text.png", "M6 payoff vs structure (main)"),
    ]
    main_f = mb / "main/figures"
    out: list[dict[str, Any]] = []
    for fid, legacy_name, desc in specs:
        p = main_f / f"{fid}.png"
        if not p.is_file():
            p = main_f / legacy_name
        sz = p.stat().st_size if p.is_file() else 0
        decision = "DROP"
        reason: list[str] = []
        if not p.is_file():
            reason.append("PNG missing under main/figures")
        elif sz < min_fig_bytes:
            decision = "MOVE"
            reason.append(f"file size {sz}B < {min_fig_bytes}B (likely smoke or low-DPI)")
        else:
            decision = "KEEP"
            reason.append("present and above size threshold")
        out.append(
            {
                "figure": fid,
                "path": str(p.relative_to(mb)) if p.is_file() else f"main/figures/{fid}.png",
                "bytes": sz,
                "decision": decision,
                "description": desc,
                "reasons": reason,
            }
        )
    return out


def _preflight(mb: Path) -> dict[str, Any]:
    repo = mb.resolve()
    while repo != repo.parent and not (repo / "esl").is_dir():
        repo = repo.parent
    ppo = repo / "third_party" / "PPO-PyTorch"
    dpi = os.environ.get("ESL_PUBLICATION_DPI", "")
    return {
        "manuscript_bundle_exists": mb.is_dir(),
        "ppo_submodule_present": ppo.is_dir(),
        "ESL_PUBLICATION_DPI": dpi or "(unset — export 300 before plotting full runs)",
    }


def run_figure_diagnostics(
    manuscript_bundle: Path,
    *,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    min_fig_bytes: int = DEFAULT_MIN_FIG_BYTES,
) -> dict[str, Any]:
    mb = manuscript_bundle.resolve()

    pre = _preflight(mb)
    smoke = _audit_smoke(mb)
    m4 = _m4_section(mb, min_seeds=min_seeds)
    m5 = _m5_section(mb, min_seeds=min_seeds)
    m6 = _m6_section(mb, min_seeds=min_seeds)
    figs = _figure_decisions(mb, min_fig_bytes=min_fig_bytes)

    lines = [
        "# Figure Diagnostics Report",
        "",
        "_Automated validation only — no experiments re-run, no figures edited._",
        "",
        "## Pre-flight (Step 0)",
        "",
        f"- Manuscript bundle exists: **{pre['manuscript_bundle_exists']}**",
        f"- `third_party/PPO-PyTorch` present: **{pre['ppo_submodule_present']}** (omit `--skip-ppo` only if true)",
        f"- `ESL_PUBLICATION_DPI`: {pre['ESL_PUBLICATION_DPI']}",
        f"- Milestone 4 smoke flag from `statistical_audit.json`: **{smoke}** (`null` if audit missing)",
        "",
    ]
    if smoke is True:
        lines.append(
            "> **Warning:** audit indicates Milestone 4 ran in **smoke** mode — full paper acceptance requires non-smoke M4–M7.\n"
        )

    def _emit_ms(title: str, block: dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("### Effect size")
        for x in block.get("eff_notes") or ["_(no notes)_"]:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("### CI quality")
        for x in block.get("ci_notes") or ["_(no wide-CI flags)_"]:
            lines.append(f"- {x}")
        lines.append("")
        lines.append("### Stability / data volume")
        for x in block.get("flags") or ["_(no flags)_"]:
            lines.append(f"- {x}")
        if block.get("smoke_hint"):
            lines.append(f"- {block['smoke_hint']}")
        lines.append("")
        rec = "Review flags above before main-text inclusion."
        if not block.get("flags") and not block.get("ci_notes") and not block.get("eff_notes"):
            rec = "No automated issues; still verify narrative fit."
        lines.append(f"**Recommendation:** {rec}")
        lines.append("")

    _emit_ms("Milestone 4 (Recovery)", m4)
    _emit_ms("Milestone 5 (Adaptation)", m5)
    _emit_ms("Milestone 6 (Mechanism)", m6)

    lines.append("## Final figure decisions (main paper)")
    lines.append("")
    lines.append("| Figure | Decision | Bytes | Notes |")
    lines.append("|--------|----------|------:|-------|")
    for f in figs:
        rs = "; ".join(f["reasons"])
        lines.append(f"| {f['figure']} | **{f['decision']}** | {f['bytes']} | {f['description']}: {rs} |")
    lines.append("")
    lines.append("### MODIFY vs appendix")
    lines.append("")
    lines.append(
        "- **MOVE** here means: keep for supplementary PDF or appendix until full runs increase resolution / CI tightness."
    )
    lines.append("- **DROP** means: missing asset or unsuitable for camera-ready bundle without re-staging.")
    lines.append("")
    lines.append("## Acceptance checklist (manual)")
    lines.append("")
    lines.append("- [ ] M4–M6 executed **without** `--smoke`")
    lines.append("- [ ] `export ESL_PUBLICATION_DPI=300` (or equivalent) before sweeps that plot")
    lines.append("- [ ] `manuscript_bundle/main/tables/` aggregates include mean + 95% CI columns")
    lines.append("- [ ] Milestone 7 staged `main/figures/` sources, then optional Milestone 8 for `fig1`–`fig5`")
    lines.append("- [ ] This report reviewed; STOP before M8/M9 until figure decisions confirmed")
    lines.append("")

    report_path = mb / "reports" / "figure_diagnostics.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    write_manifest_json(
        mb / "manifests" / "figure_diagnostics_manifest.json",
        {
            "schema_version": 1,
            "milestone4_flags": m4["flags"],
            "milestone5_flags": m5["flags"],
            "milestone6_flags": m6["flags"],
            "figure_decisions": figs,
            "audit_smoke": smoke,
        },
    )

    return {
        "report": str(report_path),
        "m4": m4,
        "m5": m5,
        "m6": m6,
        "figures": figs,
        "audit_smoke": smoke,
    }


def main_figure_diagnostics(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Figure diagnostics report (post Milestone 7)")
    p.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    p.add_argument(
        "--min-seeds",
        type=int,
        default=DEFAULT_MIN_SEEDS,
        help="flag aggregates with fewer independent seeds than this (default 10)",
    )
    p.add_argument(
        "--min-fig-bytes",
        type=int,
        default=DEFAULT_MIN_FIG_BYTES,
        help="KEEP if fig PNG at least this many bytes (default 8000)",
    )
    args = p.parse_args(argv)
    info = run_figure_diagnostics(
        args.manuscript_bundle,
        min_seeds=args.min_seeds,
        min_fig_bytes=args.min_fig_bytes,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "m4" and k != "m5" and k != "m6"}, indent=2))


if __name__ == "__main__":
    main_figure_diagnostics()
