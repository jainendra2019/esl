#!/usr/bin/env python3
"""Held-out 3×3 PSR comparison of ESL and OMD/MWU.

Locks:
  - Cell unit: (instance_id, seed); focal role fixed to row
  - Tune η (OMD) and λ (ESL) on seeds {0,1,2} by mean PSR
  - Primary confirmatory report: seeds {3,...,9}, n=84
  - Secondary descriptive summary: seeds {0,...,9}, n=120 (includes the tuning slice)
  - Always report ESL-default λ=2 alongside ESL decision-tuned λ*
  - Headline twelve-game tables use λ=2; λ* is this comparison only
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from esl.res.benchmark import default_benchmark_root, load_suite
from esl.res.focal_lambda_sensitivity import FOCAL_LAMBDA_GRID
from esl.res.omd_mwu import OMD_ETA_GRID
from esl.res.runner import run_res_cell

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "omd_clean_psr"
ESL_DEFAULT_CELLS = (
    ROOT / "outputs" / "res" / "eval" / "expanded_adaptation" / "symmetric" / "cells.csv"
)
DEV_SEEDS = (0, 1, 2)
PRIMARY_SEEDS = (3, 4, 5, 6, 7, 8, 9)
ALL_SEEDS = tuple(range(10))
HORIZON = 1000
FOCAL_ROLE = "row"
DEFAULT_LAMBDA = 2.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_omd(spec, *, seed: int, eta: float, scope: str) -> dict[str, Any]:
    cell = run_res_cell(
        spec=spec,
        method="OMD/MWU",
        seed=int(seed),
        horizon=HORIZON,
        force_categorical=True,
        focal_role=FOCAL_ROLE,
        omd_c=float(eta),
        omd_scope=scope,
    )
    s = dict(cell["summary"])
    s["instance_id"] = spec.instance_id
    s["family"] = spec.family
    s["focal_role"] = FOCAL_ROLE
    s["omd_eta"] = float(eta)
    s["omd_scope"] = scope
    s["method_label"] = "Per-opponent OMD/MWU" if scope == "per_opponent" else "Global OMD/MWU"
    s["lam_br"] = ""
    return s


def _run_esl(spec, *, seed: int, lam: float) -> dict[str, Any]:
    cell = run_res_cell(
        spec=spec,
        method="ESL",
        seed=int(seed),
        horizon=HORIZON,
        force_categorical=True,
        focal_role=FOCAL_ROLE,
        lam_br=float(lam),
    )
    s = dict(cell["summary"])
    s["instance_id"] = spec.instance_id
    s["family"] = spec.family
    s["focal_role"] = FOCAL_ROLE
    s["method_label"] = f"ESL λ={lam:g}"
    s["lam_br"] = float(lam)
    s["omd_eta"] = ""
    s["omd_scope"] = ""
    return s


def run_omd_grid(
    *,
    specs: list,
    seeds: tuple[int, ...],
    etas: tuple[float, ...],
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(specs) * len(seeds) * len(etas)
    done = 0
    for spec in specs:
        for seed in seeds:
            for eta in etas:
                rows.append(_run_omd(spec, seed=seed, eta=eta, scope=scope))
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [OMD {scope}] {done}/{total}", flush=True)
    return rows


def run_esl_grid(*, specs: list, seeds: tuple[int, ...], lams: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(specs) * len(seeds) * len(lams)
    done = 0
    for spec in specs:
        for seed in seeds:
            for lam in lams:
                rows.append(_run_esl(spec, seed=seed, lam=lam))
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [ESL λ-grid] {done}/{total}", flush=True)
    return rows


def select_min_psr(
    rows: list[dict[str, Any]], *, key: str
) -> tuple[float, list[dict[str, Any]]]:
    by: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        by[float(r[key])].append(float(r["psr"]))
    summary = []
    best = None
    best_mean = float("inf")
    for val in sorted(by):
        vals = np.asarray(by[val], dtype=np.float64)
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        summary.append(
            {
                key: val,
                "n": len(vals),
                "psr_mean": mean,
                "psr_se": se,
                "psr_ci95_lo": mean - 1.96 * se,
                "psr_ci95_hi": mean + 1.96 * se,
            }
        )
        if mean < best_mean:
            best_mean = mean
            best = val
    assert best is not None
    return float(best), summary


def summarize_psr(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = np.asarray([float(r["psr"]) for r in rows], dtype=np.float64)
    n = int(vals.size)
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "psr_mean": mean,
        "psr_se": se,
        "psr_ci95_lo": mean - 1.96 * se,
        "psr_ci95_hi": mean + 1.96 * se,
    }


def paired_delta(
    left_rows: list[dict[str, Any]],
    right_by_key: dict[tuple[str, int], float],
) -> dict[str, Any]:
    diffs = []
    missing = 0
    for r in left_rows:
        key = (str(r["instance_id"]), int(r["seed"]))
        if key not in right_by_key:
            missing += 1
            continue
        diffs.append(float(r["psr"]) - float(right_by_key[key]))
    d = np.asarray(diffs, dtype=np.float64)
    n = int(d.size)
    if n < 2:
        return {"n": n, "missing": missing, "mean_delta": float("nan")}
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(n)
    dz = mean / sd if sd > 0 else float("inf")
    rng = np.random.default_rng(42)
    boot = sorted(float(np.mean(d[rng.integers(0, n, size=n)])) for _ in range(10000))
    return {
        "n": n,
        "missing": missing,
        "mean_delta": mean,
        "ci95_lo": boot[int(0.025 * len(boot))],
        "ci95_hi": boot[int(0.975 * len(boot))],
        "se": se,
        "dz": dz,
        "note": "positive Δ ⇒ left higher/worse PSR than right",
    }


def _filter_seeds(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    want = {int(s) for s in seeds}
    return [r for r in rows if int(r["seed"]) in want]


def _load_esl_default() -> list[dict[str, Any]]:
    rows = []
    for r in _read_csv(ESL_DEFAULT_CELLS):
        if r["method"] != "ESL":
            continue
        rows.append(
            {
                "family": r["family"],
                "instance_id": r["instance_id"],
                "method": "ESL",
                "method_label": "ESL-default λ=2",
                "seed": int(r["seed"]),
                "horizon": int(float(r.get("horizon", HORIZON))),
                "focal_role": FOCAL_ROLE,
                "lam_br": DEFAULT_LAMBDA,
                "omd_scope": "",
                "omd_eta": "",
                "omd_c": "",
                "psr": float(r["psr"]),
                "dpr": r.get("dpr", ""),
                "lte_auc": r.get("lte_auc", ""),
                "early_gap_auc": r.get("early_gap_auc", ""),
            }
        )
    assert len(rows) == 120, f"expected 120 ESL-default cells, got {len(rows)}"
    return rows


def _fmt_sum(s: dict[str, Any]) -> str:
    return f'{s["psr_mean"]:.2f} [{s["psr_ci95_lo"]:.2f}, {s["psr_ci95_hi"]:.2f}] (n={s["n"]})'


def _fmt_pair(p: dict[str, Any]) -> str:
    return (
        f'Δ={p["mean_delta"]:.2f}, 95% CI [{p["ci95_lo"]:.2f}, {p["ci95_hi"]:.2f}], '
        f'dz={p["dz"]:.3f}, n={p["n"]}'
    )


def _layer_block(
    *,
    title: str,
    esl2: dict[str, Any],
    esl_star: dict[str, Any],
    po: dict[str, Any],
    gl: dict[str, Any],
    paired_po_vs_star: dict[str, Any],
    paired_gl_vs_star: dict[str, Any],
    paired_po_vs_2: dict[str, Any],
    lam_star: float,
    eta: float,
) -> str:
    return f"""### {title}

| Method | hyperparam | n | PSR mean | 95% CI |
|--------|------------|---|----------|--------|
| ESL-default | λ={DEFAULT_LAMBDA:g} | {esl2["n"]} | {esl2["psr_mean"]:.2f} | [{esl2["psr_ci95_lo"]:.2f}, {esl2["psr_ci95_hi"]:.2f}] |
| ESL-decision-tuned | λ={lam_star:g} | {esl_star["n"]} | {esl_star["psr_mean"]:.2f} | [{esl_star["psr_ci95_lo"]:.2f}, {esl_star["psr_ci95_hi"]:.2f}] |
| Per-opponent OMD/MWU | η={eta:g} | {po["n"]} | {po["psr_mean"]:.2f} | [{po["psr_ci95_lo"]:.2f}, {po["psr_ci95_hi"]:.2f}] |
| Global OMD/MWU | η={eta:g} | {gl["n"]} | {gl["psr_mean"]:.2f} | [{gl["psr_ci95_lo"]:.2f}, {gl["psr_ci95_hi"]:.2f}] |

Paired (OMD − ESL-decision-tuned λ={lam_star:g}):
- Per-opponent: {_fmt_pair(paired_po_vs_star)}
- Global: {_fmt_pair(paired_gl_vs_star)}

Paired (per-opponent OMD − ESL-default λ=2): {_fmt_pair(paired_po_vs_2)}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = load_suite(default_benchmark_root(ROOT), num_actions=3)
    assert len(specs) == 12

    # --- η selection (reuse prior grid if present) ---
    eta_path = OUT / "eta_dev_cells.csv"
    if eta_path.exists():
        print("=== Reusing η development grid ===", flush=True)
        eta_dev = _read_csv(eta_path)
    else:
        print("=== η development grid (per-opponent; seeds 0-2) ===", flush=True)
        eta_dev = run_omd_grid(
            specs=specs, seeds=DEV_SEEDS, etas=OMD_ETA_GRID, scope="per_opponent"
        )
        _write_csv(
            eta_path,
            eta_dev,
            [
                "family",
                "instance_id",
                "method",
                "method_label",
                "seed",
                "horizon",
                "focal_role",
                "omd_scope",
                "omd_eta",
                "omd_c",
                "psr",
                "dpr",
                "lte_auc",
            ],
        )
    frozen_eta, eta_summary = select_min_psr(eta_dev, key="omd_eta")
    _write_csv(
        OUT / "eta_dev_summary.csv",
        eta_summary,
        ["omd_eta", "n", "psr_mean", "psr_se", "psr_ci95_lo", "psr_ci95_hi"],
    )
    print(f"Frozen η = {frozen_eta}", flush=True)

    # --- λ selection on same development seeds ---
    print(
        f"=== λ development grid {FOCAL_LAMBDA_GRID} (ESL; seeds 0-2) ===",
        flush=True,
    )
    lam_dev = run_esl_grid(specs=specs, seeds=DEV_SEEDS, lams=tuple(FOCAL_LAMBDA_GRID))
    frozen_lam, lam_summary = select_min_psr(lam_dev, key="lam_br")
    _write_csv(
        OUT / "lambda_dev_cells.csv",
        lam_dev,
        [
            "family",
            "instance_id",
            "method",
            "method_label",
            "seed",
            "horizon",
            "focal_role",
            "lam_br",
            "psr",
            "dpr",
            "lte_auc",
        ],
    )
    _write_csv(
        OUT / "lambda_dev_summary.csv",
        lam_summary,
        ["lam_br", "n", "psr_mean", "psr_se", "psr_ci95_lo", "psr_ci95_hi"],
    )
    print(f"Frozen λ* = {frozen_lam} (min mean PSR on seeds 0-2)", flush=True)

    # --- Full evaluation cells (reuse if hyperparams match) ---
    esl2 = _load_esl_default()

    esl_star_path = OUT / "esl_lambda_star_cells.csv"
    legacy_l10 = OUT / "esl_lambda10_cells.csv"
    reuse_star = False
    if esl_star_path.exists() or legacy_l10.exists():
        cand_path = esl_star_path if esl_star_path.exists() else legacy_l10
        cand = _read_csv(cand_path)
        lams = {float(r["lam_br"]) for r in cand if r.get("lam_br") not in ("", None)}
        if len(cand) == 120 and lams == {float(frozen_lam)}:
            print(f"=== Reusing ESL λ={frozen_lam:g} full cells ===", flush=True)
            esl_star = cand
            for r in esl_star:
                r["method_label"] = f"ESL-decision-tuned λ={frozen_lam:g}"
            reuse_star = True
    if not reuse_star:
        print(f"=== ESL λ*={frozen_lam:g} full 12×10 ===", flush=True)
        esl_star = run_esl_grid(specs=specs, seeds=ALL_SEEDS, lams=(frozen_lam,))
        for r in esl_star:
            r["method_label"] = f"ESL-decision-tuned λ={frozen_lam:g}"

    po_path = OUT / "per_opponent_report_cells.csv"
    gl_path = OUT / "global_report_cells.csv"
    reuse_omd = False
    if po_path.exists() and gl_path.exists():
        po_all = _read_csv(po_path)
        gl_all = _read_csv(gl_path)
        if (
            len(po_all) == 120
            and len(gl_all) == 120
            and {float(r["omd_eta"]) for r in po_all} == {float(frozen_eta)}
            and {float(r["omd_eta"]) for r in gl_all} == {float(frozen_eta)}
        ):
            print("=== Reusing OMD full cells at frozen η ===", flush=True)
            reuse_omd = True
    if not reuse_omd:
        print("=== Per-opponent OMD full 12×10 ===", flush=True)
        po_all = run_omd_grid(
            specs=specs, seeds=ALL_SEEDS, etas=(frozen_eta,), scope="per_opponent"
        )
        print("=== Global OMD full 12×10 ===", flush=True)
        gl_all = run_omd_grid(
            specs=specs, seeds=ALL_SEEDS, etas=(frozen_eta,), scope="global"
        )

    cell_fields = [
        "family",
        "instance_id",
        "method",
        "method_label",
        "seed",
        "horizon",
        "focal_role",
        "lam_br",
        "omd_scope",
        "omd_eta",
        "omd_c",
        "psr",
        "dpr",
        "lte_auc",
        "early_gap_auc",
    ]
    _write_csv(OUT / "esl_default_lambda2_cells.csv", esl2, cell_fields)
    _write_csv(OUT / "esl_lambda_star_cells.csv", esl_star, cell_fields)
    _write_csv(OUT / "esl_lambda10_cells.csv", esl_star, cell_fields)  # alias if λ*=10
    _write_csv(po_path, po_all, cell_fields)
    _write_csv(gl_path, gl_all, cell_fields)

    def pack(seeds: tuple[int, ...]) -> dict[str, Any]:
        e2 = _filter_seeds(esl2, seeds)
        es = _filter_seeds(esl_star, seeds)
        po = _filter_seeds(po_all, seeds)
        gl = _filter_seeds(gl_all, seeds)
        es_by = {(str(r["instance_id"]), int(r["seed"])): float(r["psr"]) for r in es}
        e2_by = {(str(r["instance_id"]), int(r["seed"])): float(r["psr"]) for r in e2}
        return {
            "esl2": summarize_psr(e2),
            "esl_star": summarize_psr(es),
            "po": summarize_psr(po),
            "gl": summarize_psr(gl),
            "paired_po_vs_star": paired_delta(po, es_by),
            "paired_gl_vs_star": paired_delta(gl, es_by),
            "paired_po_vs_2": paired_delta(po, e2_by),
            "paired_star_vs_2": paired_delta(es, e2_by),
        }

    primary = pack(PRIMARY_SEEDS)
    secondary = pack(ALL_SEEDS)

    summary_rows = []
    for layer, seeds, blk in (
        ("primary_held_out", PRIMARY_SEEDS, primary),
        ("secondary_descriptive", ALL_SEEDS, secondary),
    ):
        for method, hyp, key in (
            ("ESL-default", DEFAULT_LAMBDA, "esl2"),
            ("ESL-decision-tuned", frozen_lam, "esl_star"),
            ("Per-opponent OMD/MWU", frozen_eta, "po"),
            ("Global OMD/MWU", frozen_eta, "gl"),
        ):
            summary_rows.append(
                {
                    "layer": layer,
                    "seeds": ",".join(map(str, seeds)),
                    "method": method,
                    "eta_or_lam": hyp,
                    **blk[key],
                }
            )
    _write_csv(
        OUT / "omd_psr_summary.csv",
        summary_rows,
        [
            "layer",
            "seeds",
            "method",
            "eta_or_lam",
            "n",
            "psr_mean",
            "psr_se",
            "psr_ci95_lo",
            "psr_ci95_hi",
        ],
    )

    paired_payload = {
        "tuning": {
            "dev_seeds": list(DEV_SEEDS),
            "lambda_grid": list(FOCAL_LAMBDA_GRID),
            "eta_grid": list(OMD_ETA_GRID),
            "frozen_lambda": frozen_lam,
            "frozen_eta": frozen_eta,
            "selection_criterion": "minimize mean PSR on development cells",
            "comparable_tuning_budget": (
                "same development seeds; one scalar hyperparameter each "
                "(λ for ESL decision-tuned; η for OMD); same PSR criterion"
            ),
        },
        "primary_held_out_seeds_3_9": {
            "n": 84,
            "per_opponent_minus_esl_star": primary["paired_po_vs_star"],
            "global_minus_esl_star": primary["paired_gl_vs_star"],
            "per_opponent_minus_esl_default": primary["paired_po_vs_2"],
            "esl_star_minus_esl_default": primary["paired_star_vs_2"],
        },
        "secondary_descriptive_seeds_0_9": {
            "n": 120,
            "disclosure": "includes development seeds 0–2 used to freeze λ* and η",
            "per_opponent_minus_esl_star": secondary["paired_po_vs_star"],
            "global_minus_esl_star": secondary["paired_gl_vs_star"],
            "per_opponent_minus_esl_default": secondary["paired_po_vs_2"],
            "esl_star_minus_esl_default": secondary["paired_star_vs_2"],
        },
    }
    (OUT / "omd_vs_esl_paired.json").write_text(
        json.dumps(paired_payload, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "protocol": "clean_3x3_omd_bakeoff_held_out_primary",
        "cell_unit": "one cell = (instance_id, seed); focal role fixed to row (not a crossed factor)",
        "num_games": 12,
        "focal_role": FOCAL_ROLE,
        "horizon": HORIZON,
        "p_obs": 1.0,
        "obs_noise": 0.0,
        "dev_seeds": list(DEV_SEEDS),
        "primary_seeds": list(PRIMARY_SEEDS),
        "secondary_seeds": list(ALL_SEEDS),
        "primary_n": 84,
        "secondary_n": 120,
        "lambda_grid": list(FOCAL_LAMBDA_GRID),
        "eta_grid": list(OMD_ETA_GRID),
        "frozen_lambda_star": frozen_lam,
        "frozen_eta": frozen_eta,
        "esl_default_lambda": DEFAULT_LAMBDA,
        "selection_criterion": "minimize mean PSR on seeds 0–2",
        "comparable_tuning": True,
        "primary_scope": "per_opponent",
        "secondary_scope": "global",
        "psr_definition": "POST_SWITCH_PSR_SUM",
        "feedback": "full-information U[:, a_opp] after observing opponent action",
        "identity": "opponent id j protocol-visible",
        "esl_default_source": str(ESL_DEFAULT_CELLS.relative_to(ROOT)),
        "note": (
            "Primary confirmatory inference uses seeds 3–9 only. "
            "Secondary n=120 is descriptive and reuses the tuning slice."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    p = primary
    s = secondary
    report = f"""# ESL vs OMD Clean 3×3 PSR comparison (held-out primary)

## Protocol

- **Cell unit:** `(instance_id, seed)` under the common 1+3 clean schedule. Focal role is **fixed to row** (not a crossed factor).
- **Development / tuning slice:** seeds {{{", ".join(map(str, DEV_SEEDS))}}} × 12 games.
  - OMD η grid: {{{", ".join(str(x) for x in OMD_ETA_GRID)}}} → **frozen η = {frozen_eta:g}** (min mean PSR).
  - ESL λ grid: {{{", ".join(str(x) for x in FOCAL_LAMBDA_GRID)}}} → **frozen λ\\* = {frozen_lam:g}** (min mean PSR).
  - Same seeds, same PSR criterion, one scalar hyperparameter each (comparable tuning budget).
- **Primary confirmatory report:** seeds {{{", ".join(map(str, PRIMARY_SEEDS))}}}, **n = 84** (untouched by hyperparameter selection).
- **Full-seed descriptive summary:** seeds {{{", ".join(map(str, ALL_SEEDS))}}}, **n = 120**, which **includes** the development slice; not used for confirmatory claims.
- **ESL-default (λ={DEFAULT_LAMBDA:g}):** original main-paper/default configuration; retained in all main tables.
- **ESL-decision-tuned (λ={frozen_lam:g}):** sharper soft-BR for this OMD comparison only; does **not** replace main-paper ESL.
- **OMD:** per-opponent primary; global secondary. Favorable full-info payoff-vector feedback. No latent types / switch / reset. DPR/LTE N/A.

## Development selection (seeds 0–2)

| λ | n_dev | PSR mean |
|---|-------|----------|
""" + "\n".join(
        f"| {float(r['lam_br']):g} | {r['n']} | {r['psr_mean']:.2f} |" for r in lam_summary
    ) + f"""

| η | n_dev | PSR mean |
|---|-------|----------|
""" + "\n".join(
        f"| {float(r['omd_eta']):g} | {r['n']} | {r['psr_mean']:.2f} |" for r in eta_summary
    ) + f"""

## Primary confirmatory comparison (seeds 3–9, n=84)

{_layer_block(
    title="Held-out test",
    esl2=p["esl2"],
    esl_star=p["esl_star"],
    po=p["po"],
    gl=p["gl"],
    paired_po_vs_star=p["paired_po_vs_star"],
    paired_gl_vs_star=p["paired_gl_vs_star"],
    paired_po_vs_2=p["paired_po_vs_2"],
    lam_star=frozen_lam,
    eta=frozen_eta,
)}

(Positive Δ ⇒ OMD higher/worse PSR than the ESL reference.)
The paired OMD − ESL-default λ={DEFAULT_LAMBDA:g} contrast is reported for transparency only: a negative value means OMD has lower PSR than λ={DEFAULT_LAMBDA:g} and must not be read as λ={DEFAULT_LAMBDA:g} dominating OMD.

## Appendix: full-seed descriptive summary (seeds 0–9, n=120)

Not used for confirmatory claims. Includes development seeds 0–2 used to freeze λ\\* and η.

{_layer_block(
    title="Appendix (includes tuning seeds)",
    esl2=s["esl2"],
    esl_star=s["esl_star"],
    po=s["po"],
    gl=s["gl"],
    paired_po_vs_star=s["paired_po_vs_star"],
    paired_gl_vs_star=s["paired_gl_vs_star"],
    paired_po_vs_2=s["paired_po_vs_2"],
    lam_star=frozen_lam,
    eta=frozen_eta,
)}

## Interpretation

λ controls the trade-off between softer belief-conditioned behavior and aggressive decision exploitation. For the OMD comparison we report a **decision-tuned** ESL variant (λ={frozen_lam:g}), selected on the development slice by mean PSR. This isolates decision performance against a favorable full-information no-regret learner; it is **not** evidence that the original main-paper/default λ={DEFAULT_LAMBDA:g} configuration dominates OMD. Main tables retain λ={DEFAULT_LAMBDA:g}. On the held-out primary slice, decision-tuned ESL outperforms per-opponent OMD and is statistically indistinguishable from globally pooled OMD, while ESL additionally supports prediction / latent identification / shared prototypes.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Done → {OUT}", flush=True)
    print(
        f"Primary n=84: ESL*={p['esl_star']['psr_mean']:.2f}, "
        f"OMD-po={p['po']['psr_mean']:.2f}, "
        f"Δ={p['paired_po_vs_star']['mean_delta']:.2f} "
        f"[{p['paired_po_vs_star']['ci95_lo']:.2f}, {p['paired_po_vs_star']['ci95_hi']:.2f}]"
    )


if __name__ == "__main__":
    main()
