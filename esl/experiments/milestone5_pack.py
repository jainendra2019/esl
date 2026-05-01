"""
Milestone 5: strategic adaptation validation (heterogeneous IPD, multi-seed, CI, manuscript bundle).

Methods: ESL (K>=2), ESL ablation ``num_prototypes=1``, clustering+logit BR (K-means + FCM),
independent PPO focal (official submodule when present).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from esl.config import ESLConfig
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.manifest import write_run_manifest
from esl.experiments.milestone5_adaptation_baselines import (
    run_clustering_br_adaptation,
    run_ppo_focal_heterogeneous_ipd,
)
from esl.experiments.milestone5_metrics import (
    focal_cumulative_payoff,
    focal_payoff_per_round_from_run_dir,
    write_focal_sidecar,
)
from esl.experiments.progress_echo import log_step
from esl.experiments.stats_ci import mean_std_ci95
from esl.trainer import run_esl

PAPER_SEEDS: tuple[int, ...] = tuple(range(5))
SMOKE_SEEDS: tuple[int, ...] = (0, 1)


def build_milestone5_esl_config(
    *,
    seed: int,
    num_rounds: int,
    include_tft: bool,
    esl_ablation_k1: bool,
    smoke: bool,
) -> ESLConfig:
    """Heterogeneous population; focal agent 0 uses ESL; others fixed (AC/AD/TFT)."""
    if include_tft and esl_ablation_k1:
        raise ValueError("include_tft and esl_ablation_k1 together are not supported (K=1 ablation).")
    n = 7 if include_tft else 6
    if include_tft:
        k = 3
        force_types = [0, 0, 1, 1, 2, 1, 2]
        hidden_override = None
    elif esl_ablation_k1:
        k = 1
        force_types = [0] * n
        hidden_override = [0, 0, 0, 1, 1, 1][:n]
    else:
        k = 2
        force_types = [0, 0, 0, 1, 1, 1][:n]
        hidden_override = None

    cfg = ESLConfig(
        seed=seed,
        mode="adaptation",
        num_agents=n,
        num_prototypes=k,
        num_rounds=30 if smoke else num_rounds,
        force_agent_true_types=list(force_types),
        force_hidden_policy_by_agent=list(hidden_override) if hidden_override is not None else None,
        adaptation_esl_agent_indices=[0],
        observability="full",
        p_obs=1.0,
        prototype_update_every=4,
        prototype_lr_scale=12.0,
        init_noise=0.08,
        symmetric_init=False,
        adaptation_lambda=2.0,
        interaction_pairs_min=1,
        interaction_pairs_max=2,
        log_beliefs_tensor=not smoke,
        stop_on_convergence=False,
    )
    cfg.validate()
    return cfg


def _method_slug(method: str) -> str:
    return method.replace("/", "_")


def _m5_run_dir_complete(run_dir: Path) -> bool:
    """A finished M5 cell writes focal sidecar and summary after the trainer/baseline."""
    return (run_dir / "focal_payoff_per_round.csv").is_file() and (run_dir / "summary_metrics.json").is_file()


def _load_m5_summary_from_run_dir(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))


def _run_one_method(
    cfg: ESLConfig,
    run_dir: Path,
    method: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir / "run_manifest.json",
        {"experiment": "milestone5_adaptation_ipd", "method": method, "seed": cfg.seed},
    )
    if method == "esl":
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
    elif method == "esl_ablation_k1":
        _, _, _, summary, _ = run_esl(cfg, run_dir=run_dir)
    elif method == "clustering_kmeans":
        summary = run_clustering_br_adaptation(cfg, run_dir, backend="kmeans")
    elif method == "clustering_fcm":
        summary = run_clustering_br_adaptation(cfg, run_dir, backend="fcm")
    elif method == "ppo_focal":
        summary = run_ppo_focal_heterogeneous_ipd(cfg, run_dir, focal_idx=0)
    else:
        raise ValueError(f"unknown method {method}")
    write_focal_sidecar(run_dir, focal=0)
    return summary


def _aggregate_focal_mean_per_round(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for r in long_rows:
        g = str(r.get("method", ""))
        v = r.get("focal_mean_payoff_per_round")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        fv = float(v)
        if np.isfinite(fv):
            buckets.setdefault(g, []).append(fv)
    out: list[dict[str, Any]] = []
    for g in sorted(buckets.keys()):
        st = mean_std_ci95(buckets[g])
        out.append(
            {
                "method": g,
                "focal_mean_per_round_mean": st["mean"],
                "focal_mean_per_round_std": st["std"],
                "focal_mean_per_round_ci95_low": st["ci95_low"],
                "focal_mean_per_round_ci95_high": st["ci95_high"],
                "focal_mean_per_round_n": int(st["n"]),
            }
        )
    return out


def _plot_adaptation_curves(
    out_png: Path,
    *,
    curve_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """``curve_by_method[method]`` is a list of (rounds, y) arrays per seed (aligned max T)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {
        "esl": "C0",
        "esl_ablation_k1": "C1",
        "clustering_kmeans": "C2",
        "clustering_fcm": "C3",
        "ppo_focal": "C4",
    }
    for method, curves in curve_by_method.items():
        if not curves:
            continue
        t_max = max(int(c[0][-1]) for c in curves if c[0].size > 0)
        if t_max < 0:
            continue
        grid = np.arange(t_max + 1, dtype=int)
        mat = np.full((len(curves), t_max + 1), np.nan, dtype=np.float64)
        for si, (ts, ys) in enumerate(curves):
            for ti, yi in zip(ts.tolist(), ys.tolist(), strict=False):
                if 0 <= int(ti) <= t_max:
                    mat[si, int(ti)] = float(yi)
        mean = np.full(t_max + 1, np.nan, dtype=np.float64)
        lo = np.full_like(mean, np.nan)
        hi = np.full_like(mean, np.nan)
        for t in range(t_max + 1):
            col = mat[:, t]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            st = mean_std_ci95(col.tolist())
            mean[t] = st["mean"]
            lo[t] = st["ci95_low"]
            hi[t] = st["ci95_high"]
        c = colors.get(method, "C7")
        ax.plot(grid, mean, label=method, color=c, linewidth=1.8)
        ax.fill_between(grid, lo, hi, color=c, alpha=0.18, linewidth=0)
    ax.set_xlabel("Environment round")
    ax.set_ylabel("Focal mean payoff (in-round)")
    ax.set_title("Milestone 5 — adaptation performance over time (mean ± 95% CI)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=publication_figure_dpi())
    plt.close(fig)


def _plot_method_comparison_bars(
    out_png: Path,
    *,
    agg_rows: list[dict[str, Any]],
) -> None:
    if not agg_rows:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["method"] for r in agg_rows]
    means = [r["focal_mean_per_round_mean"] for r in agg_rows]
    lo = [r["focal_mean_per_round_ci95_low"] for r in agg_rows]
    hi = [r["focal_mean_per_round_ci95_high"] for r in agg_rows]
    err = [[m - l for m, l in zip(means, lo, strict=False)], [h - m for m, h in zip(means, hi, strict=False)]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x, means, yerr=err, capsize=4, color=["C0", "C1", "C2", "C3", "C4"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Focal mean payoff / round (± 95% CI)")
    ax.set_title("Milestone 5 — method comparison (heterogeneous IPD)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=publication_figure_dpi())
    plt.close(fig)


def _milestone5_report(path: Path, *, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Milestone 5 — Strategic adaptation validation",
        "",
        "## Design",
        "",
        "- **Game:** Iterated Prisoner's Dilemma (repo PD payoffs).",
        "- **Population:** focal agent 0 runs the method under test; opponents are heterogeneous "
        "(Always Cooperate, Always Defect, optional Tit-for-Tat). TFT uses the trainer's scalar "
        "`last_opponent_action` hook (well-mixed approximation; see `games.TitForTat`).",
        "- **ESL:** logit best response vs belief-mixture opponent models; focal only (`adaptation_esl_agent_indices=[0]`).",
        "- **Clustering baselines:** same beliefs and observation protocol as ESL, but prototype logits "
        "are refit from **K-means** or **FCM** clustering on (belief snapshot, one-hot signal) features.",
        "- **PPO:** independent actor–critic (official `PPO.py`) on the focal agent only; no prototypes or beliefs.",
        "- **Ablation:** `num_prototypes=1` with `force_hidden_policy_by_agent` preserving AC/AD bots (no latent mixture).",
        "",
        "## Statistics",
        "",
        f"- Seeds: `{meta.get('seeds')}`",
        f"- Smoke: `{meta.get('smoke')}`",
        f"- `include_tft`: `{meta.get('include_tft')}`",
        "",
        "Metrics stress **focal mean payoff per round** and **cumulative focal payoff** (sidecars under each run). "
        "Belief accuracy and MCE are reported for ESL-like runs for reference; PPO leaves MCE undefined.",
        "",
        "## Interpretation (high level)",
        "",
        "- If ESL curves lie above clustering, structured opponent modeling + SGD helps beyond "
        "unsupervised cluster refits of the same telemetry.",
        "- If PPO tracks or beats ESL, the environment may be too short / too stationary for belief "
        "learning to amortize, or the PPO state carries enough signal for myopic RL.",
        "- The K=1 ablation tests whether latent **mixture** structure matters on top of logit BR.",
        "",
        "## Outputs",
        "",
        "- `aggregate/milestone5_long.csv` — per-seed summaries.",
        "- `aggregate/milestone5_method_comparison.csv` — mean ± 95% CI for focal payoff.",
        "- `figures/milestone5_*.png` — adaptation curves and bar comparison.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_milestone5_pack(
    *,
    out_root: Path,
    manuscript_bundle: Path | None = None,
    smoke: bool = False,
    seeds: tuple[int, ...] | None = None,
    num_rounds: int = 800,
    include_tft: bool = False,
    skip_ppo: bool = False,
    resume_skip_complete: bool = False,
) -> dict[str, Any]:
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seed_tuple = tuple(SMOKE_SEEDS if smoke else (seeds or PAPER_SEEDS))
    methods: list[str] = [
        "esl",
        "esl_ablation_k1",
        "clustering_kmeans",
        "clustering_fcm",
    ]
    if not skip_ppo:
        methods.append("ppo_focal")

    long_rows: list[dict[str, Any]] = []
    curve_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in methods}

    ppo_skipped = False
    _prog = not smoke
    for seed in seed_tuple:
        log_step(f"Milestone 5 — seed {seed}", enabled=_prog)
        base_cfg = build_milestone5_esl_config(
            seed=seed,
            num_rounds=num_rounds,
            include_tft=include_tft,
            esl_ablation_k1=False,
            smoke=smoke,
        )
        ab_cfg = build_milestone5_esl_config(
            seed=seed,
            num_rounds=num_rounds,
            include_tft=include_tft,
            esl_ablation_k1=True,
            smoke=smoke,
        )
        cfgs: dict[str, ESLConfig] = {
            "esl": base_cfg,
            "esl_ablation_k1": ab_cfg,
            "clustering_kmeans": base_cfg,
            "clustering_fcm": base_cfg,
            "ppo_focal": base_cfg,
        }
        for method in methods:
            if method == "ppo_focal" and skip_ppo:
                continue
            log_step(f"Milestone 5 — seed {seed} method `{method}`", enabled=_prog)
            cfg = cfgs[method]
            slug = _method_slug(method)
            run_dir = out_root / slug / f"seed_{seed}"
            if resume_skip_complete and run_dir.is_dir() and not _m5_run_dir_complete(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
            if resume_skip_complete and _m5_run_dir_complete(run_dir):
                log_step(
                    f"Milestone 5 — reuse completed run (skip) seed {seed} method `{method}`",
                    enabled=_prog,
                )
                summary = _load_m5_summary_from_run_dir(run_dir)
            else:
                try:
                    summary = _run_one_method(cfg, run_dir, method)
                except FileNotFoundError:
                    if method == "ppo_focal":
                        ppo_skipped = True
                        continue
                    raise
            rounds, ys = focal_payoff_per_round_from_run_dir(run_dir, focal=0)
            curve_by_method[method].append((rounds, ys))
            n_exec = int(summary.get("num_rounds_executed", cfg.num_rounds) or cfg.num_rounds)
            focal_cum = focal_cumulative_payoff(run_dir, focal=0)
            focal_mpr = float(focal_cum / max(n_exec, 1))
            long_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "focal_cumulative_payoff": focal_cum,
                    "focal_mean_payoff_per_round": focal_mpr,
                    "mean_payoff_per_agent_per_round": summary.get("mean_payoff_per_agent_per_round", ""),
                    "final_mce": summary.get("final_mce", ""),
                    "final_belief_argmax_accuracy": summary.get("final_belief_argmax_accuracy", ""),
                    "run_dir": str(run_dir),
                }
            )

    agg_dir = out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    long_csv = agg_dir / "milestone5_long.csv"
    with long_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "seed",
            "method",
            "focal_cumulative_payoff",
            "focal_mean_payoff_per_round",
            "mean_payoff_per_agent_per_round",
            "final_mce",
            "final_belief_argmax_accuracy",
            "run_dir",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in long_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    cmp_rows = _aggregate_focal_mean_per_round(long_rows)
    cmp_path = agg_dir / "milestone5_method_comparison.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        fn = [
            "method",
            "focal_mean_per_round_mean",
            "focal_mean_per_round_std",
            "focal_mean_per_round_ci95_low",
            "focal_mean_per_round_ci95_high",
            "focal_mean_per_round_n",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in cmp_rows:
            w.writerow({k: r.get(k, "") for k in fn})

    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_adaptation_curves(fig_dir / "milestone5_adaptation_focal_payoff.png", curve_by_method=curve_by_method)
    _plot_method_comparison_bars(fig_dir / "milestone5_method_comparison_bars.png", agg_rows=cmp_rows)

    meta = {
        "seeds": list(seed_tuple),
        "smoke": smoke,
        "include_tft": include_tft,
        "num_rounds_requested": num_rounds,
        "methods": methods,
        "ppo_skipped": ppo_skipped,
        "long_csv": str(long_csv),
        "comparison_csv": str(cmp_path),
    }
    (agg_dir / "milestone5_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if manuscript_bundle is not None:
        mb = manuscript_bundle.resolve()
        ensure_manuscript_bundle_layout(mb)
        rep = mb / "reports" / "milestone_5"
        rep.mkdir(parents=True, exist_ok=True)
        _milestone5_report(rep / "MILESTONE_STATUS.md", meta=meta)
        shutil.copy2(cmp_path, mb / "tables" / "milestone5_method_comparison.csv")
        shutil.copy2(long_csv, mb / "tables" / "milestone5_long.csv")
        shutil.copy2(agg_dir / "milestone5_run_meta.json", mb / "manifests" / "milestone5_run_meta.json")
        for png in fig_dir.glob("milestone5_*.png"):
            shutil.copy2(png, mb / "figures" / png.name)
        ex = next((out_root / "esl").glob("seed_*/config.json"), None)
        if ex and ex.is_file():
            shutil.copy2(ex, mb / "configs" / "milestone5_example_esl_config.json")
    return meta


def main_milestone5(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 5 adaptation IPD pack")
    p.add_argument("--out-root", type=Path, default=Path("runs/milestone5_adaptation"))
    p.add_argument("--manuscript-bundle", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=str, default=None, help="comma-separated ints")
    p.add_argument("--rounds", type=int, default=800)
    p.add_argument("--include-tft", action="store_true")
    p.add_argument("--skip-ppo", action="store_true", help="skip PPO even if submodule is present")
    p.add_argument(
        "--resume-skip-complete",
        action="store_true",
        help="reuse finished (seed, method) runs; wipe and redo incomplete directories",
    )
    args = p.parse_args(argv)
    seeds: tuple[int, ...] | None = None
    if args.seeds:
        seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip() != "")
    meta = run_milestone5_pack(
        out_root=args.out_root,
        manuscript_bundle=args.manuscript_bundle,
        smoke=args.smoke,
        seeds=seeds,
        num_rounds=args.rounds,
        include_tft=args.include_tft,
        skip_ppo=args.skip_ppo,
        resume_skip_complete=args.resume_skip_complete,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main_milestone5()
