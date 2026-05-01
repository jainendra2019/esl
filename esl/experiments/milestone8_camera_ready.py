"""
Milestone 8: camera-ready narrative and artifact polish (presentation only).

Renames main figures to fig1–fig5, moves auxiliary panels to appendix, writes
``figure_index.json``, rewrites ``EXPERIMENT_REPORT.md``, and aligns one-line takeaway
across abstract / intro / conclusion stubs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from esl.experiments.canonical_io import write_manifest_json
from esl.experiments.milestone7_manuscript import ensure_milestone7_manuscript_layout

# Single takeaway used everywhere (abstract, intro, conclusion).
ONE_LINE_TAKEAWAY = (
    "Belief-coupled latent opponent models improve long-run payoffs when feedback about others is limited."
)


def _first_existing(fig_dir: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = fig_dir / n
        if p.is_file():
            return p
    return None


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_move_to_appendix(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src.unlink()
    return True


def _rel_under(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _cleanup_stale_narrative_files(mb: Path) -> list[str]:
    """Remove pre–Milestone-8 caption/summary filenames so the bundle stays camera-ready."""
    removed: list[str] = []
    cap = mb / "text/main/captions"
    keep_caps = {f"fig{i}.md" for i in range(1, 6)}
    if cap.is_dir():
        for p in list(cap.glob("*.md")):
            if p.name not in keep_caps:
                p.unlink(missing_ok=True)
                removed.append(str(p.relative_to(mb)))
    summ = mb / "text/main/summaries"
    keep_sum = {"results_overview.md"}
    if summ.is_dir():
        for p in list(summ.glob("*.md")):
            if p.name not in keep_sum:
                p.unlink(missing_ok=True)
                removed.append(str(p.relative_to(mb)))
    return removed


def run_milestone8_camera_ready(
    manuscript_bundle: Path,
) -> dict[str, Any]:
    mb = manuscript_bundle.resolve()
    ensure_milestone7_manuscript_layout(mb)
    main_f = mb / "main/figures"
    main_t = mb / "main/tables"
    apx_f = mb / "appendix/figures"
    apx_t = mb / "appendix/tables"
    for d in (main_f, main_t, apx_f, apx_t):
        d.mkdir(parents=True, exist_ok=True)

    # --- Paper figure map: fig1–fig5 (main text only) ---
    figure_defs: list[dict[str, Any]] = [
        {
            "paper_id": "fig1",
            "output_file": "fig1.png",
            "source_candidates": ("sparse_pobs_mce_ci.png",),
            "claim": "Recovery remains stable as observation probability varies.",
            "evidence": "Cross-seed mean and 95% CI for matched cross-entropy across observation rates.",
        },
        {
            "paper_id": "fig2",
            "output_file": "fig2.png",
            "source_candidates": ("q_recovery_mce_ci.png",),
            "claim": "Performance is sensitive to how often latent type parameters are updated from data.",
            "evidence": "Mean matched cross-entropy (±95% CI) across update cadences.",
        },
        {
            "paper_id": "fig3",
            "output_file": "fig3.png",
            "source_candidates": ("milestone5_adaptation_focal_payoff.png",),
            "claim": "Payoffs evolve over repeated interaction when learners face a heterogeneous population.",
            "evidence": "Focal-agent mean payoff per round with cross-seed uncertainty bands.",
        },
        {
            "paper_id": "fig4",
            "output_file": "fig4.png",
            "source_candidates": ("milestone5_method_comparison_bars.png",),
            "claim": "Structured opponent modeling compares favorably to learning-free clustering baselines.",
            "evidence": "End-horizon focal payoff (mean ±95% CI) across learning approaches.",
        },
        {
            "paper_id": "fig5",
            "output_file": "fig5.png",
            "source_candidates": (
                "m6_payoff_vs_mce_main_text.png",
                "m6_payoff_vs_mce.png",
            ),
            "claim": "Better alignment between inferred and reference opponent structure co-occurs with higher payoffs.",
            "evidence": "Focal mean payoff versus matched cross-entropy after excluding the oscillatory diagnostic regime.",
        },
    ]

    index_entries: list[dict[str, Any]] = []
    resolved: list[str] = []
    missing: list[str] = []

    # Move robustness-only figures before renaming, so main/ stays unambiguous.
    appendix_moves: list[str] = []
    early_appendix_figures = (
        "init_noise_mce_ci.png",
        "k_latent_mce_ci.png",
        "m6_k_misspec_payoff_ci.png",
        "m6_p_obs_payoff_ci.png",
        "m6_belief_ablation_bars.png",
        "m6_payoff_vs_mce.png",
        "m6_payoff_vs_mce_full_including_mp.png",
    )
    for name in early_appendix_figures:
        if _safe_move_to_appendix(main_f / name, apx_f / name):
            appendix_moves.append(f"appendix/figures/{name}")

    for spec in figure_defs:
        search_dirs = (main_f, apx_f) if spec["paper_id"] == "fig5" else (main_f,)
        src: Path | None = None
        for d in search_dirs:
            hit = _first_existing(d, tuple(spec["source_candidates"]))
            if hit is not None:
                src = hit
                break
        out = main_f / spec["output_file"]
        if src is None:
            missing.append(spec["paper_id"])
            index_entries.append(
                {
                    "paper_id": spec["paper_id"],
                    "paper_figure": f"main/figures/{spec['output_file']}",
                    "original_artifact": None,
                    "output_file": spec["output_file"],
                    "source_resolved": None,
                    "status": "missing_source",
                    "claim": spec["claim"],
                    "caption_path": f"text/main/captions/{spec['paper_id']}.md",
                }
            )
            continue
        original_rel = _rel_under(mb, src)
        _safe_copy(src, out)
        resolved.append(spec["paper_id"])
        if src.resolve() != out.resolve() and src.parent == main_f:
            try:
                src.unlink()
            except OSError:
                pass
        index_entries.append(
            {
                "paper_id": spec["paper_id"],
                "paper_figure": f"main/figures/{spec['output_file']}",
                "original_artifact": original_rel,
                "output_file": spec["output_file"],
                "source_resolved": src.name,
                "status": "ok",
                "claim": spec["claim"],
                "evidence": spec["evidence"],
                "caption_path": f"text/main/captions/{spec['paper_id']}.md",
            }
        )

    # Any remaining non–camera-ready filenames under main/figures → appendix.
    for p in list(main_f.glob("*.png")):
        if p.name not in {f"fig{i}.png" for i in range(1, 6)}:
            if _safe_move_to_appendix(p, apx_f / p.name):
                appendix_moves.append(f"appendix/figures/{p.name}")

    appendix_tables = (
        "m6_k_misspec_aggregate.csv",
        "m6_p_obs_aggregate.csv",
        "m6_belief_ablation_aggregate.csv",
        "milestone6_all_runs.csv",
    )
    for name in appendix_tables:
        p = main_t / name
        if p.is_file():
            if _safe_move_to_appendix(p, apx_t / name):
                appendix_moves.append(f"appendix/tables/{name}")

    # --- Captions (camera-ready wording) ---
    cap_dir = mb / "text/main/captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    captions_body = {
        "fig1": (
            "**Figure 1.** Recovery quality as a function of how often actions are observed. "
            "Curves show cross-run averages; shaded bands are 95% confidence intervals for the mean."
        ),
        "fig2": (
            "**Figure 2.** Recovery quality across different rates of latent-parameter updates from data. "
            "Shaded bands summarize cross-run uncertainty."
        ),
        "fig3": (
            "**Figure 3.** Focal-agent payoffs over repeated rounds against a mixed population of fixed partners. "
            "Bands reflect cross-run variability."
        ),
        "fig4": (
            "**Figure 4.** End-of-horizon focal payoffs across learning approaches. "
            "Error bars show 95% confidence intervals for the mean across runs."
        ),
        "fig5": (
            "**Figure 5.** Relationship between focal payoff and mismatch between inferred and reference opponent structure. "
            "Each point is one experimental configuration and seed; the diagnostic zero-sum regime is excluded here (see appendix)."
        ),
    }
    for fid, text in captions_body.items():
        (cap_dir / f"{fid}.md").write_text(text + "\n", encoding="utf-8")

    stale_removed = _cleanup_stale_narrative_files(mb)

    # --- One-line takeaway (identical everywhere) ---
    takeaway_path = mb / "text/main/ONE_LINE_TAKEAWAY.txt"
    takeaway_path.parent.mkdir(parents=True, exist_ok=True)
    takeaway_path.write_text(ONE_LINE_TAKEAWAY + "\n", encoding="utf-8")
    for name in ("abstract.txt", "intro_one_line.txt", "conclusion_one_line.txt"):
        (mb / "text/main" / name).write_text(ONE_LINE_TAKEAWAY + "\n", encoding="utf-8")

    # --- Summary (non-implementation) ---
    (mb / "text/main/summaries/results_overview.md").write_text(
        f"{ONE_LINE_TAKEAWAY}\n\n"
        "The main text emphasizes recovery under limited observations, adaptation against a heterogeneous "
        "population, and the payoff–structure relationship. Robustness sweeps and supplementary plots are "
        "collected in the appendix bundle.\n",
        encoding="utf-8",
    )

    # --- figure_index.json ---
    apx_readme = mb / "appendix/README.md"
    apx_readme.parent.mkdir(parents=True, exist_ok=True)
    apx_readme.write_text(
        "# Appendix bundle\n\n"
        "Figures and tables here support robustness claims: extra recovery sweeps, adaptation-stress "
        "panels beyond the single payoff–structure figure in the main text, long-form run logs, and "
        "optional pooled diagnostics. Paths mirror the staging layout after Milestone 7–8.\n",
        encoding="utf-8",
    )

    fig_index = {
        "schema_version": 1,
        "milestone": "milestone8_camera_ready",
        "one_line_takeaway": ONE_LINE_TAKEAWAY,
        "main_figures": index_entries,
        "appendix_moves": appendix_moves,
        "missing_main_figures": missing,
        "stale_narrative_files_removed": stale_removed,
    }
    (mb / "figure_index.json").write_text(json.dumps(fig_index, indent=2), encoding="utf-8")

    # --- EXPERIMENT_REPORT.md (claim → figure → evidence) ---
    report = f"""# Experiment report — claims, figures, and evidence

**Takeaway.** {ONE_LINE_TAKEAWAY}

## Figure index

Machine-readable mapping: ``figure_index.json`` (includes source resolution and caption paths).

## Claim → figure → evidence

| Claim | Figure | Evidence (where to look) |
|-------|--------|---------------------------|
| Recovery is stable when actions are only sometimes observed | **Figure 1** (`main/figures/fig1.png`) | Aggregated runs in `main/tables`; CI bands on the figure. |
| Recovery depends on how frequently latent parameters are updated from data | **Figure 2** (`main/figures/fig2.png`) | Same tables; cadence on the horizontal axis. |
| Focal payoffs evolve under repeated play with a mixed partner population | **Figure 3** (`main/figures/fig3.png`) | Time series with cross-run bands; per-run logs under the paper run root. |
| Structured opponent learning compares favorably to clustering-only alternatives | **Figure 4** (`main/figures/fig4.png`) | Bar summary with CIs; companion CSV in `main/tables`. |
| Payoff rises as inferred structure aligns with a fixed two-type reference | **Figure 5** (`main/figures/fig5.png`) | Scatter over seeds/configurations; underlying rows in appendix tables for extended robustness. |

## Appendix (robustness and diagnostics)

- **Figures:** `appendix/figures/` — initialization and latent-cardinality sweeps from the recovery study; supplementary adaptation-stress panels; optional pooled scatter including the oscillatory diagnostic.
- **Tables:** `appendix/tables/` — aggregated robustness tables for adaptation stress and belief ablation.

## Reproducibility

Paper runs are produced by the frozen orchestration config (`esl/experiments/configs/milestone7_frozen.json`) and staged by Milestone 7; Milestone 8 only renames, moves, and rewrites narrative files without re-running experiments.
"""
    (mb / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")

    # --- Readiness note ---
    (mb / "reports/milestone_8").mkdir(parents=True, exist_ok=True)
    (mb / "reports/milestone_8" / "READINESS.md").write_text(
        "## Milestone 8 — Camera-ready polish\n\n"
        "- Main text figures are standardized as `fig1.png`–`fig5.png`.\n"
        "- Auxiliary recovery and adaptation-stress panels live under `appendix/`.\n"
        f"- Missing main figure slots (if any): `{missing}`\n\n"
        "### Submission checklist\n\n"
        "- [ ] Replace placeholder prose in LaTeX with numbers from staged tables.\n"
        "- [ ] Verify every `fig*.png` is cited in the manuscript.\n"
        "- [ ] Confirm appendix PDF includes moved panels and tables.\n",
        encoding="utf-8",
    )

    write_manifest_json(
        mb / "manifests" / "milestone8_bundle_manifest.json",
        {
            "schema_version": 1,
            "milestone": "milestone8_camera_ready",
            "resolved_main_figures": resolved,
            "missing_main_figures": missing,
            "appendix_moves_count": len(appendix_moves),
        },
    )

    return {
        "manuscript_bundle": str(mb),
        "resolved": resolved,
        "missing": missing,
        "appendix_moves": appendix_moves,
        "stale_narrative_files_removed": stale_removed,
    }


def main_milestone8(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 8 camera-ready manuscript polish")
    p.add_argument(
        "--manuscript-bundle",
        type=Path,
        default=Path("manuscript_bundle"),
        help="path to manuscript bundle root",
    )
    args = p.parse_args(argv)
    info = run_milestone8_camera_ready(args.manuscript_bundle)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main_milestone8()
