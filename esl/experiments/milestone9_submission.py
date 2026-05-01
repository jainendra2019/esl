"""
Milestone 9: submission hardening, camera-ready assembly, notation audit,
reviewer-style critique, and accept/reject risk notes (presentation only).

Does not re-run experiments, change algorithms, or add new figures.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from esl.experiments.canonical_io import write_manifest_json
from esl.experiments.milestone7_manuscript import ensure_milestone7_manuscript_layout
from esl.experiments.milestone8_camera_ready import ONE_LINE_TAKEAWAY, run_milestone8_camera_ready


MIN_MAIN_FIG_BYTES = 512  # smoke PNGs are tiny; real paper assets should exceed this


def _read_takeaway(mb: Path) -> str:
    p = mb / "text/main/ONE_LINE_TAKEAWAY.txt"
    if p.is_file():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return ONE_LINE_TAKEAWAY


def _copy_tree_filtered(src: Path, dst: Path, *, glob_pat: str = "*") -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    if not src.is_dir():
        return out
    for p in sorted(src.glob(glob_pat)):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            out.append(str(p.name))
    return out


def _copy_recursive_text(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def _load_figure_index(mb: Path) -> dict[str, Any] | None:
    idx = mb / "figure_index.json"
    if not idx.is_file():
        return None
    return json.loads(idx.read_text(encoding="utf-8"))


def _verify_main_figures(mb: Path, min_bytes: int) -> dict[str, Any]:
    main_f = mb / "main/figures"
    cap_dir = mb / "text/main/captions"
    checks: list[dict[str, Any]] = []
    for i in range(1, 6):
        fig = main_f / f"fig{i}.png"
        cap = cap_dir / f"fig{i}.md"
        st = fig.stat() if fig.is_file() else None
        checks.append(
            {
                "figure": f"fig{i}",
                "path": f"main/figures/fig{i}.png",
                "exists": fig.is_file(),
                "bytes": st.st_size if st else 0,
                "size_ok": bool(st and st.st_size >= min_bytes),
                "caption_path": f"text/main/captions/fig{i}.md",
                "caption_exists": cap.is_file(),
                "caption_nonempty": cap.is_file() and len(cap.read_text(encoding="utf-8").strip()) > 20,
            }
        )
    extras = [p.name for p in main_f.glob("*.png")] if main_f.is_dir() else []
    extras = [n for n in extras if not re.match(r"fig[1-5]\.png$", n)]
    return {"per_figure": checks, "extra_png_in_main_figures": extras}


def _gather_paper_text(mb: Path) -> str:
    parts: list[str] = []
    for rel in (
        "EXPERIMENT_REPORT.md",
        "text/main/abstract.txt",
        "text/main/intro_one_line.txt",
        "text/main/conclusion_one_line.txt",
        "text/main/summaries/results_overview.md",
    ):
        p = mb / rel
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
    cap = mb / "text/main/captions"
    if cap.is_dir():
        for p in sorted(cap.glob("fig*.md")):
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _notation_audit(blob: str) -> dict[str, Any]:
    """Heuristic consistency hints (not a full LaTeX parse)."""
    issues: list[dict[str, str]] = []
    has_mce = bool(re.search(r"\bMCE\b", blob))
    has_matched_ce = "matched cross-entropy" in blob.lower() or "matched cross entropy" in blob.lower()
    has_plain_ce = bool(re.search(r"cross[- ]entropy", blob, re.I)) and not has_matched_ce
    if has_mce and has_plain_ce and not has_matched_ce:
        issues.append(
            {
                "kind": "metric_naming",
                "detail": "Uses both 'MCE' and generic 'cross-entropy' without defining 'matched'. Prefer 'MCE (matched cross-entropy to reference types)' once, then MCE.",
            }
        )
    # Baseline naming variants
    variants = {
        "clustering+BR": r"clustering\s*\+\s*BR|clustering\+best[- ]response|cluster[- ]action\s*\+\s*logit",
        "K-means": r"\bK[- ]?means\b",
        "PPO": r"\bPPO\b",
        "FCM": r"\bFCM\b|fuzzy\s*c[- ]?means",
    }
    found = {k: bool(re.search(pat, blob, re.I)) for k, pat in variants.items()}
    if found["K-means"] and found["clustering+BR"]:
        issues.append(
            {
                "kind": "baseline_naming",
                "detail": "Both 'K-means' and 'clustering+BR' phrasing appear; ensure the manuscript maps one display name to the implementation once (e.g., 'clustering + logit best-response (K-means features)').",
            }
        )
    if found["FCM"]:
        issues.append(
            {
                "kind": "baseline_scope",
                "detail": "FCM / fuzzy c-means appears in bundle text. If the main paper does not report FCM, remove stray mentions from paper-facing stubs to avoid reviewer confusion.",
            }
        )
    return {"issues": issues, "baseline_flags": found, "has_mce_token": has_mce}


def _write_abstract_refined(mb: Path, takeaway: str) -> Path:
    body = (
        "**Problem.** In repeated matrix games, agents often receive only partial feedback about others' "
        "actions yet must infer latent strategic structure to act well over long horizons.\n\n"
        "**Method.** We study Epistemic Social Learning (ESL), a feedback-coupled latent-opponent model: "
        "pairwise beliefs over a finite set of behavioral prototypes update quickly from observed actions, "
        "while prototype parameters move on a slower batch schedule.\n\n"
        "**Key results.** Across recovery and adaptation experiments, ESL maintains reliable alignment "
        "between inferred and reference opponent structure under sparse observations; tracks focal payoffs "
        "against heterogeneous fixed partners; compares favorably to clustering-only learning at the end "
        "of the horizon; and shows co-variation between payoff and structure alignment when a diagnostic "
        "oscillatory regime is excluded from the primary scatter (see appendix).\n\n"
        "**Implication.** "
        f"{takeaway}\n"
    )
    plain_body = (
        "In repeated matrix games, agents often receive only partial feedback about others' actions yet "
        "must infer latent strategic structure to act well over long horizons. "
        "We study Epistemic Social Learning (ESL), a feedback-coupled latent-opponent model: pairwise "
        "beliefs over a finite set of behavioral prototypes update quickly from observed actions, while "
        "prototype parameters move on a slower batch schedule. "
        "Across recovery and adaptation experiments, ESL maintains reliable alignment between inferred "
        "and reference opponent structure under sparse observations; tracks focal payoffs against "
        "heterogeneous fixed partners; compares favorably to clustering-only learning at the end of the "
        "horizon; and shows co-variation between payoff and structure alignment when a diagnostic "
        "oscillatory regime is excluded from the primary scatter (see appendix). "
        f"{takeaway}"
    )
    out = mb / "text/main/abstract_refined.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    (mb / "text/main/abstract_refined_plain.txt").write_text(plain_body + "\n", encoding="utf-8")
    return out


def _write_intro_opening(mb: Path, takeaway: str) -> None:
    p = mb / "text/main/intro_opening_refined.txt"
    p.write_text(
        "Many learning settings expose agents to partners whose behavior is only partially observed. "
        "We focus on repeated two-action games and ask when a belief-coupled latent-type model can support "
        "both interpretable recovery of opponent structure and competitive payoffs under that uncertainty. "
        f"{takeaway}\n",
        encoding="utf-8",
    )


def _write_notation_glossary(mb: Path) -> None:
    g = mb / "text/main/NOTATION_GLOSSARY.md"
    g.write_text(
        "# Notation and naming (paper-facing)\n\n"
        "- **ESL** — Epistemic Social Learning: pairwise beliefs over latent behavioral prototypes with "
        "batch prototype updates (two-timescale).\n"
        "- **MCE** — Matched cross-entropy between inferred opponent structure and a fixed reference "
        "assignment in recovery-style experiments; lower is better alignment.\n"
        "- **Clustering + logit BR** — Unsupervised action clustering with a softmax policy and logit "
        "best-response actions (display name should match bars in Figure 4).\n"
        "- **PPO** — Proximal policy optimization focal baseline where enabled.\n"
        "- **K=1 ablation** — ESL with a single prototype, isolating the value of multi-type structure.\n"
        "Use the same strings in abstract, axis labels, and figure captions.\n",
        encoding="utf-8",
    )


def _write_core_claims_main_only(mb: Path) -> None:
    p = mb / "submission_bundle/CORE_CLAIMS_MAIN_TEXT_ONLY.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Core claims supported without reading the appendix\n\n"
        "1. **Sparse observations (Figure 1).** Recovery quality summarized on the vertical axis remains "
        "stable across observation rates on the horizontal axis (main figure only).\n"
        "2. **Update cadence (Figure 2).** Sensitivity of recovery to how often latent parameters are updated "
        "from batches (main figure only).\n"
        "3. **Adaptation trajectory (Figure 3).** Focal payoffs over rounds against a mixed population "
        "(main figure only).\n"
        "4. **Baselines (Figure 4).** End-horizon focal payoff relative to clustering-style and RL baselines "
        "where run (main figure only).\n"
        "5. **Payoff vs structure (Figure 5).** Co-variation of payoff with structure mismatch under the "
        "stated exclusion; appendix hosts the pooled diagnostic and robustness sweeps.\n\n"
        "The appendix adds stress tests and extra sweeps; it should be cited for transparency but is not "
        "required to verify the five claims above.\n",
        encoding="utf-8",
    )


def _write_reviewer_simulation(
    mb: Path,
    takeaway: str,
    verification: dict[str, Any],
    notation: dict[str, Any],
    checklist: dict[str, Any],
) -> None:
    rep = mb / "reports/milestone_9"
    rep.mkdir(parents=True, exist_ok=True)
    amb: list[str] = [
        "Abstract: quantify 'partial feedback' with the observation model (e.g., Bernoulli observation "
        "mask) on first use.",
        "Figure 2: define the horizontal 'update cadence' variable in notation and tie it to batch length M "
        "or logged schedule so it is not read as wall-clock.",
        "Figure 5: state explicitly which experimental regimes populate points and why the oscillatory "
        "regime is excluded from the main scatter.",
    ]
    if notation["issues"]:
        amb.append("Notation scan flagged naming collisions; see `notation_consistency.md`.")
    if verification.get("extra_png_in_main_figures"):
        amb.append(
            "Extra PNGs remain under `main/figures/` outside fig1–fig5; Milestone 8 should sweep these—"
            "reviewers opening the bundle may see clutter."
        )
    strengths = [
        "Clear separation between recovery (structure metrics) and adaptation (payoff + baselines).",
        "Figure index maps each numbered figure to a claim and caption path.",
        "Appendix bundle isolates robustness panels without blocking the five main claims.",
    ]
    concerns = [
        "Risk: reviewers may ask for equilibrium or regret-style guarantees—scope stays empirical.",
        "Risk: PPO availability depends on environment; ensure captions state when omitted.",
        "Risk: Figure 2 is inherently noisier than Figure 1; if bands overlap, prepare a short FAQ answer.",
    ]
    sim = f"""# Ten-minute reviewer simulation (Milestone 9)

## Minutes 1–2 — Contribution skim
A reader sees a latent-type model with **belief–action feedback**, evaluated on **recovery** and **heterogeneous adaptation**, with baselines. One-line claim: {takeaway}

## Minutes 3–5 — Abstract / intro ambiguity
{chr(10).join(f"- {a}" for a in amb)}

## Minutes 6–8 — Figure pass
- **Figure 1:** Clear axis semantics for observation rate; confirm units match text.
- **Figure 3:** Clarify partner mix (fixed bots vs learners) in caption on first submission pass.
- **Figure 4:** Legend strings must match `NOTATION_GLOSSARY.md` baseline names.
- **Figure 5:** Point cloud density; appendix should hold the full-pool variant.

## Minutes 9–10 — Baselines and metrics
{chr(10).join(f"- {i['detail']}" for i in notation.get('issues', [])) or "- No automated baseline/metric flags."}

## Programmatic checklist (this run)
```json
{json.dumps(checklist, indent=2)}
```

## Strengths (reviewer voice)
{chr(10).join(f"- {s}" for s in strengths)}

## Likely criticisms
{chr(10).join(f"- {c}" for c in concerns)}
"""
    (rep / "REVIEWER_SIMULATION.md").write_text(sim, encoding="utf-8")


def _write_risk_assessment(mb: Path, checklist_ok: bool, notation: dict[str, Any]) -> None:
    rep = mb / "reports/milestone_9"
    band = "borderline / revision likely" if not checklist_ok else "competitive if claims match numbers in final LaTeX"
    critique = "notation or packaging issues detected" if notation["issues"] else "no automated notation flags"
    text = f"""# Accept / reject risk (automated heuristic)

**Band:** {band}

**Strengths**
- Coherent milestone pipeline (M7→M8→M9) with frozen configs and reproducible bundle layout.
- Main text limited to five numbered figures with machine-readable `figure_index.json`.
- Appendix separated for robustness; core claims documented in `CORE_CLAIMS_MAIN_TEXT_ONLY.md`.

**Potential reviewer criticisms**
- Empirical paper without formal convergence guarantees for the coupled belief–prototype system.
- Dependence on synthetic matrix-game partners; external validity limited.
- Several moving parts (observation sparsity, update cadence, partner mix); clarity must stay high in prose.

**Verdict guidance**
- If `checklist.all_pass` is true and LaTeX mirrors glossary naming: **accept risk moderate** (venue-dependent).
- If checklist fails or appendix is needed to understand Figure 5: **reject risk elevated** until fixed.

_Automated note:_ {critique}; this is not a substitute for human area-chair judgment.
"""
    (rep / "ACCEPT_REJECT_RISK.md").write_text(text, encoding="utf-8")


def _assemble_submission_bundle(mb: Path) -> dict[str, Any]:
    root = mb / "submission_bundle"
    paper = root / "paper"
    supp = root / "supplementary"
    for d in (
        paper / "figures",
        paper / "tables",
        paper / "text",
        supp / "figures",
        supp / "tables",
        supp / "text",
    ):
        d.mkdir(parents=True, exist_ok=True)

    copied_main_figs = _copy_tree_filtered(mb / "main/figures", paper / "figures", glob_pat="fig*.png")
    _copy_tree_filtered(mb / "main/tables", paper / "tables", glob_pat="*")
    _copy_recursive_text(mb / "text/main", paper / "text")

    _copy_tree_filtered(mb / "appendix/figures", supp / "figures", glob_pat="*")
    _copy_tree_filtered(mb / "appendix/tables", supp / "tables", glob_pat="*")
    if (mb / "text/appendix").is_dir():
        _copy_recursive_text(mb / "text/appendix", supp / "text")

    for name in ("figure_index.json", "EXPERIMENT_REPORT.md"):
        p = mb / name
        if p.is_file():
            shutil.copy2(p, root / name)

    # Legacy flat `figures/` at bundle root is not camera-ready; park copies under supplementary only.
    legacy_src = mb / "figures"
    if legacy_src.is_dir():
        leg_dst = supp / "legacy_unbundled_figures"
        leg_dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(legacy_src.glob("*")):
            if p.is_file():
                shutil.copy2(p, leg_dst / p.name)
        (leg_dst / "README.md").write_text(
            "These files lived under the pre–Milestone-7 flat `figures/` directory. "
            "Do not use them for the main PDF unless explicitly revived; prefer `paper/figures/fig1.png`–`fig5.png`.\n",
            encoding="utf-8",
        )

    stub = paper / "text" / "figure_inclusion_stub.tex"
    stub.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated Milestone 9 — merge paths with your LaTeX layout.",
        "% Each figure is referenced once for traceability.",
    ]
    for i in range(1, 6):
        lines.append(f"\\includegraphics[width=\\linewidth]{{figures/fig{i}.png}} % Figure {i}")
    stub.write_text("\n".join(lines) + "\n", encoding="utf-8")

    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# Submission bundle (Milestone 9)\n\n"
        "- **`paper/`** — Camera-ready main text assets: `figures/fig1.png`–`fig5.png`, `tables/`, "
        "and `text/` (abstract stubs, captions, glossary).\n"
        "- **`supplementary/`** — Appendix figures/tables and optional appendix text.\n"
        "- **`figure_index.json`** — Maps original artifacts to numbered figures.\n"
        "- **`CORE_CLAIMS_MAIN_TEXT_ONLY.md`** — Claims that do not require the appendix.\n"
        "- **`SUBMISSION_CHECKLIST.md`** — Pass/fail from the latest Milestone 9 run.\n",
        encoding="utf-8",
    )
    return {"submission_root": str(root), "paper_figures_copied": copied_main_figs}


def _weak_png_warnings(fig_dir: Path, *, threshold: int = 2048) -> list[str]:
    warn: list[str] = []
    if not fig_dir.is_dir():
        return warn
    for p in fig_dir.glob("*.png"):
        try:
            if p.stat().st_size < threshold:
                warn.append(f"{p.name} is very small ({p.stat().st_size} bytes); confirm not a smoke placeholder.")
        except OSError:
            continue
    return warn


def run_milestone9_submission(manuscript_bundle: Path, *, min_fig_bytes: int = MIN_MAIN_FIG_BYTES) -> dict[str, Any]:
    mb = manuscript_bundle.resolve()
    ensure_milestone7_manuscript_layout(mb)
    if _load_figure_index(mb) is None:
        run_milestone8_camera_ready(mb)

    takeaway = _read_takeaway(mb)
    fig_index = _load_figure_index(mb) or {}
    missing = list(fig_index.get("missing_main_figures") or [])

    verification = _verify_main_figures(mb, min_fig_bytes)
    all_caps = all(c["caption_exists"] and c["caption_nonempty"] for c in verification["per_figure"])
    all_exist = all(c["exists"] for c in verification["per_figure"])
    size_ok = all(c["size_ok"] for c in verification["per_figure"])
    no_extras = len(verification["extra_png_in_main_figures"]) == 0
    idx_ok = len(missing) == 0

    blob = _gather_paper_text(mb)
    notation = _notation_audit(blob)

    _write_abstract_refined(mb, takeaway)
    _write_intro_opening(mb, takeaway)
    _write_notation_glossary(mb)
    _write_core_claims_main_only(mb)

    asm = _assemble_submission_bundle(mb)

    weak_main = _weak_png_warnings(mb / "main/figures")
    weak_apx = _weak_png_warnings(mb / "appendix/figures")

    checklist: dict[str, Any] = {
        "figure_index_resolved": idx_ok,
        "main_figures_present": all_exist,
        "captions_present": all_caps,
        "main_figure_bytes_sufficient": size_ok,
        "main_figures_only_fig1_to_5": no_extras,
        "abstract_refined_exists": (mb / "text/main/abstract_refined.md").is_file(),
        "notation_glossary_exists": (mb / "text/main/NOTATION_GLOSSARY.md").is_file(),
        "submission_bundle_assembled": (mb / "submission_bundle/paper/figures").is_dir(),
        "stub_references_all_figures": (mb / "submission_bundle/paper/text/figure_inclusion_stub.tex").is_file(),
        "weak_main_figure_warnings": weak_main,
        "weak_appendix_figure_warnings": weak_apx,
    }
    checklist["all_pass"] = all(
        bool(checklist[k])
        for k in (
            "figure_index_resolved",
            "main_figures_present",
            "captions_present",
            "main_figure_bytes_sufficient",
            "main_figures_only_fig1_to_5",
            "abstract_refined_exists",
            "notation_glossary_exists",
            "submission_bundle_assembled",
            "stub_references_all_figures",
        )
    )

    rep = mb / "reports/milestone_9"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "figure_verification.json").write_text(
        json.dumps(verification | {"missing_from_index": missing}, indent=2),
        encoding="utf-8",
    )
    (rep / "notation_consistency.md").write_text(
        "# Notation and naming scan\n\n"
        + (
            "\n".join(f"- **{i['kind']}:** {i['detail']}" for i in notation["issues"])
            or "- No automated issues detected."
        )
        + "\n\n## Detected baseline tokens\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in notation["baseline_flags"].items())
        + "\n",
        encoding="utf-8",
    )
    (rep / "checklist_results.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")

    _write_reviewer_simulation(mb, takeaway, verification, notation, checklist)
    _write_risk_assessment(mb, bool(checklist["all_pass"]), notation)

    (mb / "submission_bundle/SUBMISSION_CHECKLIST.md").write_text(
        "# Submission checklist (Milestone 9)\n\n"
        + "\n".join(f"- [{'x' if v else ' '}] `{k}`" for k, v in checklist.items() if k != "weak_main_figure_warnings" and k != "weak_appendix_figure_warnings")
        + "\n\n## Warnings\n\n"
        + (
            "\n".join(f"- {w}" for w in (weak_main + weak_apx))
            or "- (none)"
        )
        + "\n",
        encoding="utf-8",
    )

    write_manifest_json(
        mb / "manifests" / "milestone9_bundle_manifest.json",
        {
            "schema_version": 1,
            "milestone": "milestone9_submission",
            "checklist_all_pass": checklist["all_pass"],
            "missing_main_figures": missing,
            "notation_issue_count": len(notation["issues"]),
            "paper_bundle": asm["submission_root"] + "/paper",
        },
    )

    return {
        "manuscript_bundle": str(mb),
        "checklist": checklist,
        "missing_main_figures": missing,
        "notation_issues": notation["issues"],
        "assembly": asm,
    }


def main_milestone9(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 9 submission hardening and reviewer simulation")
    p.add_argument(
        "--manuscript-bundle",
        type=Path,
        default=Path("manuscript_bundle"),
        help="path to manuscript bundle root",
    )
    p.add_argument(
        "--min-fig-bytes",
        type=int,
        default=MIN_MAIN_FIG_BYTES,
        help="minimum bytes per main fig (raise for full paper assets)",
    )
    args = p.parse_args(argv)
    info = run_milestone9_submission(args.manuscript_bundle, min_fig_bytes=args.min_fig_bytes)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main_milestone9()
