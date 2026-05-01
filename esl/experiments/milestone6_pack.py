"""
Milestone 6: mechanism and robustness (structure–performance, K misspec, p_obs stress,
uniform-belief ablation, optional matching pennies).

Reuses Milestone 5 heterogeneous adaptation presets and ``run_esl``; adds rectangular MCE
via ``metrics_num_true_types`` and ``belief_updates_enabled`` for the belief ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from esl.config import ESLConfig
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.manifest import write_run_manifest
from esl.experiments.milestone5_metrics import focal_cumulative_payoff, write_focal_sidecar
from esl.experiments.milestone5_pack import build_milestone5_esl_config
from esl.experiments.progress_echo import log_step
from esl.experiments.stats_ci import mean_std_ci95
from esl.trainer import run_esl

PAPER_SEEDS: tuple[int, ...] = tuple(range(5))
SMOKE_SEEDS: tuple[int, ...] = (0, 1)
K_MODEL_GRID: tuple[int, ...] = (1, 2, 3, 5)
P_OBS_GRID: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)


def _m6_base(seed: int, horizon: int, smoke: bool) -> ESLConfig:
    """Milestone 6 uses the Milestone 5 heterogeneous adaptation preset (focal ESL)."""
    return build_milestone5_esl_config(
        seed=seed,
        num_rounds=horizon,
        include_tft=False,
        esl_ablation_k1=False,
        smoke=smoke,
    )


def _finalize_cfg(cfg: ESLConfig) -> ESLConfig:
    cfg.validate()
    return cfg


def _run_esl_tagged(cfg: ESLConfig, run_dir: Path, manifest_extra: dict[str, Any]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir / "run_manifest.json",
        {"experiment": "milestone6_robustness", **manifest_extra, "seed": cfg.seed},
    )
    _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
    write_focal_sidecar(run_dir, focal=0)
    return summary


def _focal_mean_per_round(run_dir: Path, summary: dict[str, Any]) -> float:
    cum = focal_cumulative_payoff(run_dir, focal=0)
    n = int(summary.get("num_rounds_executed") or 1)
    return float(cum / max(n, 1))


def _append_long_row(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    cfg: ESLConfig,
    run_dir: Path,
    summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    mce = summary.get("final_mce", "")
    fpr = _focal_mean_per_round(run_dir, summary)
    row: dict[str, Any] = {
        "experiment": experiment,
        "seed": cfg.seed,
        "num_prototypes": cfg.num_prototypes,
        "metrics_num_true_types": cfg.metrics_num_true_types or cfg.num_prototypes,
        "p_obs": cfg.p_obs,
        "observability": cfg.observability,
        "belief_updates_enabled": cfg.belief_updates_enabled,
        "payoff_game": cfg.payoff_game,
        "belief_variant": "",
        "final_mce": mce,
        "focal_mean_payoff_per_round": fpr,
        "mean_payoff_per_agent_per_round": summary.get("mean_payoff_per_agent_per_round", ""),
        "run_dir": str(run_dir),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def _plot_scatter_payoff_vs_mce(rows: list[dict[str, Any]], out_png: Path) -> None:
    if not rows:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        try:
            x = float(r.get("final_mce", "nan"))
            y = float(r.get("focal_mean_payoff_per_round", "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(xs, ys, alpha=0.65, s=36, c="C0", edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Final MCE (true types vs learned prototypes)")
    ax.set_ylabel("Focal mean payoff / round")
    ax.set_title("Milestone 6 — payoff vs latent-structure quality")
    fig.tight_layout()
    fig.savefig(out_png, dpi=publication_figure_dpi())
    plt.close(fig)


def _aggregate_by_key(
    rows: list[dict[str, Any]], key: str, metric: str = "focal_mean_payoff_per_round"
) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        g = str(r.get(key, ""))
        try:
            v = float(r[metric])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(v):
            buckets.setdefault(g, []).append(v)
    out: list[dict[str, Any]] = []

    def _sort_key(x: str) -> tuple[int, float | str]:
        if key == "num_prototypes":
            try:
                return (0, float(x))
            except ValueError:
                return (0, 0.0)
        if key == "p_obs":
            try:
                return (0, float(x))
            except ValueError:
                return (0, 0.0)
        return (1, x)

    for g in sorted(buckets.keys(), key=_sort_key):
        st = mean_std_ci95(buckets[g])
        out.append(
            {
                key: g,
                f"{metric}_mean": st["mean"],
                f"{metric}_std": st["std"],
                f"{metric}_ci95_low": st["ci95_low"],
                f"{metric}_ci95_high": st["ci95_high"],
                f"{metric}_n": int(st["n"]),
            }
        )
    return out


def _plot_x_with_ci(
    out_png: Path,
    *,
    agg: list[dict[str, Any]],
    x_key: str,
    x_label: str,
    title: str,
    x_numeric: bool = True,
) -> None:
    if not agg:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    labs = [r[x_key] for r in agg]
    xs = [float(x) if x_numeric else i for i, x in enumerate(labs)]
    m = [r["focal_mean_payoff_per_round_mean"] for r in agg]
    lo = [r["focal_mean_payoff_per_round_ci95_low"] for r in agg]
    hi = [r["focal_mean_payoff_per_round_ci95_high"] for r in agg]
    ax.errorbar(xs, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="o-", capsize=4, color="C0")
    if not x_numeric:
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=18, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Focal mean payoff / round (± 95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=publication_figure_dpi())
    plt.close(fig)


def _milestone6_report(path: Path, *, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 6 — Mechanism and robustness validation",
        "",
        "## Causal interpretation (design-level)",
        "",
        "- **Structure → performance:** lower **MCE** (better alignment of learned prototypes to the",
        "  fixed two-type reference ``metrics_num_true_types=2``) should co-vary with higher **focal**",
        "  payoffs when opponent modeling actually informs logit best responses.",
        "- **Misspecified K:** with ``K_true=2`` held in metrics, increasing ``num_prototypes`` beyond 2",
        "  adds capacity but also estimation load; ``K<2`` forces a single opponent hypothesis.",
        "- **Observability stress:** lowering ``p_obs`` in adaptation degrades signals feeding Bayes",
        "  updates (unless beliefs are frozen ablation), so payoff should fall smoothly if learning",
        "  depends on observations.",
        "- **Uniform beliefs:** disabling Bayes while keeping prototype SGD isolates the value of",
        "  **tracking** opponents vs only pooling observations through θ.",
        "- **Matching pennies (optional):** zero-sum matching incentives can yield oscillatory",
        "  prototype / belief telemetry rather than PD-style cooperation gradients.",
        "",
        "## Outputs",
        "",
        f"- Meta: ``{meta.get('aggregate_json', '')}``",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_milestone6_pack(
    *,
    out_root: Path,
    manuscript_bundle: Path | None = None,
    smoke: bool = False,
    seeds: tuple[int, ...] | None = None,
    num_rounds: int = 500,
    run_matching_pennies: bool = False,
) -> dict[str, Any]:
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seeds_t = tuple(SMOKE_SEEDS if smoke else (seeds or PAPER_SEEDS))
    horizon = 24 if smoke else int(num_rounds)
    k_grid = (1, 2) if smoke else K_MODEL_GRID
    p_grid = (1.0, 0.5) if smoke else P_OBS_GRID

    long_rows: list[dict[str, Any]] = []
    _prog = not smoke

    # Exp 2 — K_model misspecification (K_true for metrics fixed at 2)
    for k_model in k_grid:
        log_step(f"Milestone 6 — K misspec sweep k_model={k_model}", enabled=_prog)
        for seed in seeds_t:
            base = _m6_base(seed, horizon, smoke)
            if int(k_model) == 1:
                n = base.num_agents
                cfg = _finalize_cfg(
                    replace(
                        base,
                        num_prototypes=1,
                        metrics_num_true_types=2,
                        force_agent_true_types=[0] * n,
                        force_hidden_policy_by_agent=[0, 0, 0, 1, 1, 1][:n],
                    )
                )
            else:
                cfg = _finalize_cfg(
                    replace(
                        base,
                        num_prototypes=int(k_model),
                        metrics_num_true_types=2,
                    )
                )
            rd = out_root / "exp2_k_misspec" / f"k_model_{k_model}" / f"seed_{seed}"
            summ = _run_esl_tagged(cfg, rd, {"exp": "k_misspec", "k_model": k_model})
            _append_long_row(long_rows, experiment="k_misspec", cfg=cfg, run_dir=rd, summary=summ)

    # Exp 3 — p_obs stress (adaptation, full focal ESL)
    for p_obs in p_grid:
        log_step(f"Milestone 6 — p_obs stress p_obs={p_obs}", enabled=_prog)
        for seed in seeds_t:
            base = _m6_base(seed, horizon, smoke)
            obs = "sparse" if float(p_obs) < 1.0 else "full"
            cfg = _finalize_cfg(
                replace(
                    base,
                    num_prototypes=2,
                    metrics_num_true_types=2,
                    observability=obs,
                    p_obs=float(p_obs),
                )
            )
            slug = str(p_obs).replace(".", "p")
            rd = out_root / "exp3_p_obs" / f"p_obs_{slug}" / f"seed_{seed}"
            summ = _run_esl_tagged(cfg, rd, {"exp": "p_obs", "p_obs": p_obs})
            _append_long_row(long_rows, experiment="p_obs", cfg=cfg, run_dir=rd, summary=summ)

    # Exp 4 — belief ablation
    for belief_on in (True, False):
        log_step(f"Milestone 6 — belief ablation belief_updates_enabled={belief_on}", enabled=_prog)
        for seed in seeds_t:
            base = _m6_base(seed, horizon, smoke)
            cfg = _finalize_cfg(
                replace(
                    base,
                    num_prototypes=2,
                    metrics_num_true_types=2,
                    belief_updates_enabled=belief_on,
                )
            )
            lab = "bayes_on" if belief_on else "uniform_beliefs"
            rd = out_root / "exp4_belief_ablation" / lab / f"seed_{seed}"
            summ = _run_esl_tagged(cfg, rd, {"exp": "belief_ablation", "belief_updates": belief_on})
            _append_long_row(
                long_rows,
                experiment="belief_ablation",
                cfg=cfg,
                run_dir=rd,
                summary=summ,
                extra={"belief_variant": lab},
            )

    # Exp 5 — optional matching pennies (oscillatory regime)
    if run_matching_pennies:
        log_step("Milestone 6 — matching pennies (optional)", enabled=_prog)
        for seed in seeds_t:
            base = _m6_base(seed, horizon, smoke)
            cfg = _finalize_cfg(
                replace(
                    base,
                    num_prototypes=2,
                    metrics_num_true_types=2,
                    payoff_game="matching_pennies",
                )
            )
            rd = out_root / "exp5_matching_pennies" / f"seed_{seed}"
            summ = _run_esl_tagged(cfg, rd, {"exp": "matching_pennies"})
            _append_long_row(long_rows, experiment="matching_pennies", cfg=cfg, run_dir=rd, summary=summ)

    agg_dir = out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    long_path = agg_dir / "milestone6_all_runs.csv"
    if long_rows:
        keys = sorted(set().union(*(r.keys() for r in long_rows)))
        with long_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in long_rows:
                w.writerow({k: r.get(k, "") for k in keys})

    # Exp 1 — correlation figure from pooled runs (configs 2–4)
    corr_subset = [r for r in long_rows if r.get("experiment") in ("k_misspec", "p_obs", "belief_ablation")]
    _plot_scatter_payoff_vs_mce(corr_subset, out_root / "figures" / "m6_payoff_vs_mce.png")

    # Aggregates + CI plots
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    km_rows = [r for r in long_rows if r.get("experiment") == "k_misspec"]
    po_rows = [r for r in long_rows if r.get("experiment") == "p_obs"]
    agg_k = _aggregate_by_key(km_rows, "num_prototypes")
    if agg_k:
        with (agg_dir / "m6_k_misspec_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
            fn = list(agg_k[0].keys())
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(agg_k)
        _plot_x_with_ci(
            fig_dir / "m6_k_misspec_payoff_ci.png",
            agg=agg_k,
            x_key="num_prototypes",
            x_label="K_model (num_prototypes)",
            title="Milestone 6 — focal payoff vs K_model (K_true=2 in metrics)",
        )
    agg_p = _aggregate_by_key(po_rows, "p_obs")
    if agg_p:
        with (agg_dir / "m6_p_obs_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
            fn = list(agg_p[0].keys())
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(agg_p)
        _plot_x_with_ci(
            fig_dir / "m6_p_obs_payoff_ci.png",
            agg=agg_p,
            x_key="p_obs",
            x_label="p_obs",
            title="Milestone 6 — observability stress (adaptation)",
        )
    bel = [r for r in long_rows if r.get("experiment") == "belief_ablation"]
    agg_b = _aggregate_by_key(bel, "belief_variant", metric="focal_mean_payoff_per_round")
    if agg_b:
        with (agg_dir / "m6_belief_ablation_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
            fn = list(agg_b[0].keys())
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            w.writerows(agg_b)
        _plot_x_with_ci(
            fig_dir / "m6_belief_ablation_bars.png",
            agg=agg_b,
            x_key="belief_variant",
            x_label="Belief treatment",
            title="Milestone 6 — Bayes vs uniform beliefs (prototype SGD on)",
            x_numeric=False,
        )

    meta_path = agg_dir / "milestone6_run_meta.json"
    meta: dict[str, Any] = {
        "seeds": list(seeds_t),
        "smoke": smoke,
        "rounds": horizon,
        "k_model_grid": list(k_grid),
        "p_obs_grid": list(p_grid),
        "matching_pennies_ran": bool(run_matching_pennies),
        "long_csv": str(long_path) if long_rows else "",
        "aggregate_json": str(meta_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if manuscript_bundle is not None:
        mb = manuscript_bundle.resolve()
        ensure_manuscript_bundle_layout(mb)
        rep = mb / "reports" / "milestone_6"
        rep.mkdir(parents=True, exist_ok=True)
        _milestone6_report(rep / "MILESTONE_STATUS.md", meta=meta)
        if long_path.is_file():
            shutil.copy2(long_path, mb / "tables" / "milestone6_all_runs.csv")
        shutil.copy2(meta_path, mb / "manifests" / "milestone6_run_meta.json")
        for p in fig_dir.glob("m6_*.png"):
            shutil.copy2(p, mb / "figures" / p.name)
        for p in agg_dir.glob("m6_*aggregate.csv"):
            shutil.copy2(p, mb / "tables" / p.name)
        ex = next(out_root.glob("exp2_k_misspec/**/config.json"), None)
        if ex and ex.is_file():
            shutil.copy2(ex, mb / "configs" / "milestone6_example_config.json")

    return meta


def main_milestone6(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 6 mechanism & robustness pack")
    p.add_argument("--out-root", type=Path, default=Path("runs/milestone6_robustness"))
    p.add_argument("--manuscript-bundle", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--matching-pennies", action="store_true", help="run optional exp5 (slow if not smoke)")
    args = p.parse_args(argv)
    seeds: tuple[int, ...] | None = None
    if args.seeds:
        seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    meta = run_milestone6_pack(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        smoke=args.smoke,
        seeds=seeds,
        num_rounds=args.rounds,
        run_matching_pennies=args.matching_pennies,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main_milestone6()
