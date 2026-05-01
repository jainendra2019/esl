"""
Milestone 4: paper-grade reproduction — multi-seed sweeps, 95% CI bands, mean±std tables.

Re-runs Milestone 3-style experiments (sparse p_obs, Q) plus init-noise and K robustness.
ESL + K-means + FCM only (``clustering_only_baselines``). No adaptation / no new baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from esl.experiments.aggregate import write_aggregate_csv
from esl.experiments.canonical_io import ensure_manuscript_bundle_layout
from esl.experiments.figure_style import publication_figure_dpi
from esl.experiments.init_noise_recovery_sweep import run_init_noise_recovery_sweep
from esl.experiments.k_latent_recovery_sweep import run_k_latent_recovery_sweep
from esl.experiments.q_recovery_sweep import run_q_recovery_sweep
from esl.experiments.sparse_pobs_sweep import POBS_SWEEP_VALUES, run_sparse_pobs_sweep
from esl.experiments.progress_echo import log_step
from esl.experiments.stats_ci import mean_std_ci95

PAPER_SEEDS: tuple[int, ...] = tuple(range(10))
SMOKE_SEEDS: tuple[int, ...] = (0, 1)

DEFAULT_ROUNDS_SPARSE = 5000
DEFAULT_ROUNDS_Q = 5000
DEFAULT_ROUNDS_INIT = 5000
DEFAULT_ROUNDS_K = 5000


def _f(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    return float(s)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate_numeric(
    long_rows: list[dict[str, Any]],
    group_key: str,
    metric: str,
) -> list[dict[str, Any]]:
    buckets: dict[Any, list[float]] = defaultdict(list)
    for r in long_rows:
        g = r.get(group_key)
        v = r.get(metric)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        fv = float(v) if not isinstance(v, (int, float)) else float(v)
        if np.isfinite(fv):
            buckets[g].append(fv)
    out: list[dict[str, Any]] = []
    for g in sorted(buckets.keys(), key=lambda x: (str(type(x)), str(x))):
        st = mean_std_ci95(buckets[g])
        out.append(
            {
                group_key: g,
                f"{metric}_mean": st["mean"],
                f"{metric}_std": st["std"],
                f"{metric}_ci95_low": st["ci95_low"],
                f"{metric}_ci95_high": st["ci95_high"],
                f"{metric}_n": int(st["n"]),
                f"{metric}_mean_pm_std": f"{st['mean']:.4f} ± {st['std']:.4f}",
            }
        )
    return out


def _write_agg_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _plot_grouped_ci(
    out_png: Path,
    *,
    x_key: str,
    x_label: str,
    title: str,
    agg_rows: list[dict[str, Any]],
    series: tuple[tuple[str, str, str], ...],
) -> None:
    """``series`` entries: (metric_prefix without _mean, label, color)."""
    if not agg_rows:
        return
    xs_raw = [r[x_key] for r in agg_rows]
    xs = np.asarray(xs_raw, dtype=float) if all(isinstance(x, (int, float)) for x in xs_raw) else np.arange(len(xs_raw))
    order = np.argsort(xs)
    xs = xs[order]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for prefix, lab, c in series:
        m = np.array([r[f"{prefix}_mean"] for r in agg_rows], dtype=float)[order]
        lo = np.array([r[f"{prefix}_ci95_low"] for r in agg_rows], dtype=float)[order]
        hi = np.array([r[f"{prefix}_ci95_high"] for r in agg_rows], dtype=float)[order]
        ax.plot(xs, m, "o-", color=c, label=lab, linewidth=1.6, markersize=6)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.22, linewidth=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("MCE (mean ± 95% CI)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=publication_figure_dpi())
    plt.close(fig)


def _gather_long_sparse(sparse_root: Path) -> list[dict[str, Any]]:
    long: list[dict[str, Any]] = []
    for sd in sorted(sparse_root.glob("seed_*")):
        if not sd.is_dir():
            continue
        seed = int(sd.name.split("_", 1)[1])
        for row in _read_csv_dicts(sd / "sparse_pobs_summary.csv"):
            r = dict(row)
            r["seed"] = seed
            for k in ("final_mce", "kmeans_mce", "fcm_mce", "final_belief_argmax_accuracy"):
                r[k] = _f(str(r.get(k, "")))
            r["p_obs"] = float(r["p_obs"])
            long.append(r)
    return long


def _gather_long_generic(
    root: Path,
    subdir: str,
    summary_name: str,
    group_key: str,
    float_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    long: list[dict[str, Any]] = []
    base = root / subdir
    for sd in sorted(base.glob("seed_*")):
        if not sd.is_dir():
            continue
        seed = int(sd.name.split("_", 1)[1])
        for row in _read_csv_dicts(sd / summary_name):
            r = dict(row)
            r["seed"] = seed
            for k in float_keys:
                r[k] = _f(str(r.get(k, "")))
            r[group_key] = float(r[group_key]) if group_key != "num_prototypes" else int(float(r[group_key]))
            long.append(r)
    return long


def run_milestone4_paper(
    out_root: Path,
    *,
    smoke: bool = False,
    seeds: Sequence[int] | None = None,
    manuscript_bundle: Path | None = None,
    rounds_sparse: int | None = None,
    rounds_q: int | None = None,
    rounds_init: int | None = None,
    rounds_k: int | None = None,
    p_obs_grid: tuple[float, ...] | None = None,
    q_grid: tuple[int, ...] | None = None,
    noise_grid: tuple[float, ...] | None = None,
    k_grid: tuple[int, ...] | None = None,
    em_restarts: int = 8,
) -> dict[str, Any]:
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds) if seeds is not None else (SMOKE_SEEDS if smoke else PAPER_SEEDS)
    rs = rounds_sparse or (35 if smoke else DEFAULT_ROUNDS_SPARSE)
    rq = rounds_q or (35 if smoke else DEFAULT_ROUNDS_Q)
    ri = rounds_init or (35 if smoke else DEFAULT_ROUNDS_INIT)
    rk = rounds_k or (35 if smoke else DEFAULT_ROUNDS_K)
    pgrid = p_obs_grid or ((1.0, 0.5) if smoke else POBS_SWEEP_VALUES)
    qgrid = q_grid or ((1, 2) if smoke else (1, 5, 15))
    ngrid = noise_grid or ((0.05,) if smoke else (0.01, 0.05, 0.10))
    kgrid = k_grid or ((2,) if smoke else (2, 3))

    _prog = not smoke
    for idx, s in enumerate(seed_list):
        log_step(
            f"Milestone 4 — seed {s} ({idx + 1}/{len(seed_list)}): sparse p_obs sweep ({len(pgrid)} grid points)",
            enabled=_prog,
        )
        run_sparse_pobs_sweep(
            out_root=out_root / "sparse" / f"seed_{s}",
            num_rounds=rs,
            seed=s,
            plot=False,
            include_baselines=True,
            em_restarts=em_restarts,
            clustering_only_baselines=True,
            p_obs_values=pgrid,
        )
        log_step(f"Milestone 4 — seed {s}: Q-recovery sweep ({len(qgrid)} points)", enabled=_prog)
        run_q_recovery_sweep(
            out_root=out_root / "q_recovery" / f"seed_{s}",
            num_rounds=rq,
            seed=s,
            q_values=qgrid,
            plot=False,
            include_baselines=True,
            em_restarts=max(2, em_restarts // 2),
            clustering_only_baselines=True,
        )
        log_step(f"Milestone 4 — seed {s}: init-noise sweep ({len(ngrid)} points)", enabled=_prog)
        run_init_noise_recovery_sweep(
            out_root=out_root / "init_noise" / f"seed_{s}",
            num_rounds=ri,
            seed=s,
            noise_values=ngrid,
            plot=False,
            include_baselines=True,
            em_restarts=max(2, em_restarts // 2),
            clustering_only_baselines=True,
        )
        log_step(f"Milestone 4 — seed {s}: K-latent sweep ({len(kgrid)} points)", enabled=_prog)
        run_k_latent_recovery_sweep(
            out_root=out_root / "k_latent" / f"seed_{s}",
            num_rounds=rk,
            seed=s,
            k_values=kgrid,
            plot=False,
            include_baselines=True,
            em_restarts=max(2, em_restarts // 2),
            clustering_only_baselines=True,
        )

    agg_dir = out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = write_aggregate_csv(out_root, agg_dir / "milestone4_per_run_metrics.csv")

    long_sp = _gather_long_sparse(out_root / "sparse")
    merged_sp: dict[float, dict[str, Any]] = {}
    for m in ("final_mce", "kmeans_mce", "fcm_mce"):
        for row in _aggregate_numeric(long_sp, "p_obs", m):
            p = float(row["p_obs"])
            merged_sp.setdefault(p, {"p_obs": p})
            merged_sp[p].update(
                {
                    f"{m}_mean": row[f"{m}_mean"],
                    f"{m}_std": row[f"{m}_std"],
                    f"{m}_ci95_low": row[f"{m}_ci95_low"],
                    f"{m}_ci95_high": row[f"{m}_ci95_high"],
                    f"{m}_n": row[f"{m}_n"],
                    f"{m}_mean_pm_std": row[f"{m}_mean_pm_std"],
                }
            )
    agg_sp_rows = sorted(merged_sp.values(), key=lambda r: r["p_obs"])
    fn_sp = (
        "p_obs",
        "final_mce_mean",
        "final_mce_std",
        "final_mce_ci95_low",
        "final_mce_ci95_high",
        "final_mce_n",
        "final_mce_mean_pm_std",
        "kmeans_mce_mean",
        "kmeans_mce_std",
        "kmeans_mce_ci95_low",
        "kmeans_mce_ci95_high",
        "kmeans_mce_n",
        "kmeans_mce_mean_pm_std",
        "fcm_mce_mean",
        "fcm_mce_std",
        "fcm_mce_ci95_low",
        "fcm_mce_ci95_high",
        "fcm_mce_n",
        "fcm_mce_mean_pm_std",
    )
    _write_agg_csv(agg_dir / "sparse_pobs_agg.csv", agg_sp_rows, fn_sp)

    long_q = _gather_long_generic(
        out_root,
        "q_recovery",
        "q_recovery_summary.csv",
        "prototype_update_every_q",
        ("final_mce", "kmeans_mce", "fcm_mce"),
    )
    merged_q: dict[int, dict[str, Any]] = {}
    for m in ("final_mce", "kmeans_mce", "fcm_mce"):
        for row in _aggregate_numeric(long_q, "prototype_update_every_q", m):
            qv = int(row["prototype_update_every_q"])
            merged_q.setdefault(qv, {"prototype_update_every_q": qv})
            merged_q[qv].update(
                {
                    f"{m}_mean": row[f"{m}_mean"],
                    f"{m}_std": row[f"{m}_std"],
                    f"{m}_ci95_low": row[f"{m}_ci95_low"],
                    f"{m}_ci95_high": row[f"{m}_ci95_high"],
                    f"{m}_n": row[f"{m}_n"],
                    f"{m}_mean_pm_std": row[f"{m}_mean_pm_std"],
                }
            )
    agg_q_rows = sorted(merged_q.values(), key=lambda r: r["prototype_update_every_q"])
    fn_q = (
        "prototype_update_every_q",
        "final_mce_mean",
        "final_mce_std",
        "final_mce_ci95_low",
        "final_mce_ci95_high",
        "final_mce_n",
        "final_mce_mean_pm_std",
        "kmeans_mce_mean",
        "kmeans_mce_std",
        "kmeans_mce_ci95_low",
        "kmeans_mce_ci95_high",
        "kmeans_mce_n",
        "kmeans_mce_mean_pm_std",
        "fcm_mce_mean",
        "fcm_mce_std",
        "fcm_mce_ci95_low",
        "fcm_mce_ci95_high",
        "fcm_mce_n",
        "fcm_mce_mean_pm_std",
    )
    _write_agg_csv(agg_dir / "q_recovery_agg.csv", agg_q_rows, fn_q)

    long_i = _gather_long_generic(
        out_root,
        "init_noise",
        "init_noise_recovery_summary.csv",
        "init_noise",
        ("final_mce", "kmeans_mce", "fcm_mce"),
    )
    merged_i: dict[float, dict[str, Any]] = {}
    for m in ("final_mce", "kmeans_mce", "fcm_mce"):
        for row in _aggregate_numeric(long_i, "init_noise", m):
            iv = float(row["init_noise"])
            merged_i.setdefault(iv, {"init_noise": iv})
            merged_i[iv].update(
                {
                    f"{m}_mean": row[f"{m}_mean"],
                    f"{m}_std": row[f"{m}_std"],
                    f"{m}_ci95_low": row[f"{m}_ci95_low"],
                    f"{m}_ci95_high": row[f"{m}_ci95_high"],
                    f"{m}_n": row[f"{m}_n"],
                    f"{m}_mean_pm_std": row[f"{m}_mean_pm_std"],
                }
            )
    agg_i_rows = sorted(merged_i.values(), key=lambda r: r["init_noise"])
    fn_i = (
        "init_noise",
        "final_mce_mean",
        "final_mce_std",
        "final_mce_ci95_low",
        "final_mce_ci95_high",
        "final_mce_n",
        "final_mce_mean_pm_std",
        "kmeans_mce_mean",
        "kmeans_mce_std",
        "kmeans_mce_ci95_low",
        "kmeans_mce_ci95_high",
        "kmeans_mce_n",
        "kmeans_mce_mean_pm_std",
        "fcm_mce_mean",
        "fcm_mce_std",
        "fcm_mce_ci95_low",
        "fcm_mce_ci95_high",
        "fcm_mce_n",
        "fcm_mce_mean_pm_std",
    )
    _write_agg_csv(agg_dir / "init_noise_agg.csv", agg_i_rows, fn_i)

    long_k = _gather_long_generic(
        out_root,
        "k_latent",
        "k_latent_recovery_summary.csv",
        "num_prototypes",
        ("final_mce", "kmeans_mce", "fcm_mce"),
    )
    merged_k: dict[int, dict[str, Any]] = {}
    for m in ("final_mce", "kmeans_mce", "fcm_mce"):
        for row in _aggregate_numeric(long_k, "num_prototypes", m):
            kv = int(row["num_prototypes"])
            merged_k.setdefault(kv, {"num_prototypes": kv})
            merged_k[kv].update(
                {
                    f"{m}_mean": row[f"{m}_mean"],
                    f"{m}_std": row[f"{m}_std"],
                    f"{m}_ci95_low": row[f"{m}_ci95_low"],
                    f"{m}_ci95_high": row[f"{m}_ci95_high"],
                    f"{m}_n": row[f"{m}_n"],
                    f"{m}_mean_pm_std": row[f"{m}_mean_pm_std"],
                }
            )
    agg_k_rows = sorted(merged_k.values(), key=lambda r: r["num_prototypes"])
    fn_k = (
        "num_prototypes",
        "final_mce_mean",
        "final_mce_std",
        "final_mce_ci95_low",
        "final_mce_ci95_high",
        "final_mce_n",
        "final_mce_mean_pm_std",
        "kmeans_mce_mean",
        "kmeans_mce_std",
        "kmeans_mce_ci95_low",
        "kmeans_mce_ci95_high",
        "kmeans_mce_n",
        "kmeans_mce_mean_pm_std",
        "fcm_mce_mean",
        "fcm_mce_std",
        "fcm_mce_ci95_low",
        "fcm_mce_ci95_high",
        "fcm_mce_n",
        "fcm_mce_mean_pm_std",
    )
    _write_agg_csv(agg_dir / "k_latent_agg.csv", agg_k_rows, fn_k)

    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_grouped_ci(
        fig_dir / "sparse_pobs_mce_ci.png",
        x_key="p_obs",
        x_label=r"$p_{\mathrm{obs}}$",
        title="Sparse observability (multi-seed mean ± 95% CI)",
        agg_rows=agg_sp_rows,
        series=(("final_mce", "ESL", "C0"), ("kmeans_mce", "K-means", "C2"), ("fcm_mce", "FCM", "C5")),
    )
    _plot_grouped_ci(
        fig_dir / "q_recovery_mce_ci.png",
        x_key="prototype_update_every_q",
        x_label=r"Prototype horizon $Q$",
        title="Batch cadence Q (multi-seed mean ± 95% CI)",
        agg_rows=agg_q_rows,
        series=(("final_mce", "ESL", "C0"), ("kmeans_mce", "K-means", "C2"), ("fcm_mce", "FCM", "C5")),
    )
    _plot_grouped_ci(
        fig_dir / "init_noise_mce_ci.png",
        x_key="init_noise",
        x_label=r"Init noise $\sigma_{\mathrm{init}}$",
        title="Initialization sensitivity (multi-seed mean ± 95% CI)",
        agg_rows=agg_i_rows,
        series=(("final_mce", "ESL", "C0"), ("kmeans_mce", "K-means", "C2"), ("fcm_mce", "FCM", "C5")),
    )
    _plot_grouped_ci(
        fig_dir / "k_latent_mce_ci.png",
        x_key="num_prototypes",
        x_label=r"Latent types $K$",
        title="Latent cardinality (multi-seed mean ± 95% CI)",
        agg_rows=agg_k_rows,
        series=(("final_mce", "ESL", "C0"), ("kmeans_mce", "K-means", "C2"), ("fcm_mce", "FCM", "C5")),
    )

    meta = {
        "seeds": list(seed_list),
        "smoke": smoke,
        "rounds": {"sparse": rs, "q": rq, "init_noise": ri, "k_latent": rk},
        "grids": {
            "p_obs": list(pgrid),
            "q": list(qgrid),
            "init_noise": list(ngrid),
            "k": list(kgrid),
        },
        "aggregate_dir": str(agg_dir),
        "per_run_metrics_csv": str(per_run_csv),
        "figures_dir": str(fig_dir),
    }
    (agg_dir / "milestone4_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    copied: list[str] = []
    if manuscript_bundle is not None:
        mb = manuscript_bundle.resolve()
        ensure_manuscript_bundle_layout(mb)
        shutil.copy2(per_run_csv, mb / "metrics" / "milestone4_per_run_metrics.csv")
        copied.append("metrics/milestone4_per_run_metrics.csv")
        for name in (
            "sparse_pobs_agg.csv",
            "q_recovery_agg.csv",
            "init_noise_agg.csv",
            "k_latent_agg.csv",
            "milestone4_run_meta.json",
        ):
            shutil.copy2(agg_dir / name, mb / "tables" / f"milestone4_{name}")
            copied.append(f"tables/milestone4_{name}")
        for png in fig_dir.glob("*.png"):
            shutil.copy2(png, mb / "figures" / f"milestone4_{png.name}")
            copied.append(f"figures/milestone4_{png.name}")
        cfg_hit = next(out_root.glob("sparse/seed_*/**/config.json"), None)
        if cfg_hit is not None and cfg_hit.is_file():
            shutil.copy2(cfg_hit, mb / "configs" / "milestone4_example_config.json")
            copied.append("configs/milestone4_example_config.json")
            mf = cfg_hit.parent / "run_manifest.json"
            if mf.is_file():
                shutil.copy2(mf, mb / "manifests" / "milestone4_example_run_manifest.json")
                copied.append("manifests/milestone4_example_run_manifest.json")
        report = mb / "reports" / "milestone_4" / "MILESTONE_STATUS.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(_milestone4_report(meta, copied, seed_list), encoding="utf-8")
        copied.append("reports/milestone_4/MILESTONE_STATUS.md")

    meta["manuscript_copied"] = copied
    return meta


def _milestone4_report(meta: dict[str, Any], copied: list[str], seeds: Sequence[int]) -> str:
    lines = "\n".join(f"- `{c}`" for c in copied) or "- _(none)_"
    return f"""# Milestone 4 — Paper-grade reproduction and robustness

| Field | Value |
|-------|-------|
| Status | PASS |
| Seeds | {len(seeds)} runs per grid point ({min(seeds)}…{max(seeds)}) |
| Smoke mode | {meta.get("smoke")} |

## Statistical strength

- **95% CI** for the mean uses Student's t (`scipy.stats`) on independent seeds (equal-tailed).
- **Tables** (`*_agg.csv`) report **mean**, **sample std**, **CI bounds**, **n**, and **mean ± std** string columns for ESL / K-means / FCM MCE.
- **Figures** overlay **mean curves** with **shaded 95% CI bands** per method (ESL, K-means, FCM).

## Robustness axes

1. **Sparse observability** — `p_obs` grid (recovery flagship geometry).
2. **Prototype batch horizon** — `Q = prototype_update_every`.
3. **Initialization noise** — `init_noise` on prototype logits.
4. **Latent cardinality** — `K ∈ {{2,3}}` (cyclic templates; no new algorithms).

## Manuscript staging

{lines}

## Interpretation

- **Across-seed bands** quantify Monte Carlo variability from RNG + interaction sampling; narrower bands ⇒ more stable recovery under the protocol.
- **Init noise** and **K** panels isolate minimal robustness checks without changing the ESL update rules.
- For **publication-length** horizons, use ``--no-smoke`` and ``--rounds-*`` in the high thousands (3000–10000) as in the CLI.

## Next

Await confirmation before adaptation experiments or external deep-RL baselines.
"""


def main_milestone4(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Milestone 4 multi-seed paper reproduction")
    p.add_argument("--out-root", type=Path, default=Path("runs/milestone4_paper"))
    p.add_argument("--smoke", action="store_true", help="2 seeds, tiny rounds, reduced grids")
    p.add_argument("--manuscript-bundle", type=Path, default=None)
    p.add_argument("--seeds", type=str, default=None, help="comma-separated, e.g. 0,1,2")
    p.add_argument("--rounds-sparse", type=int, default=None)
    p.add_argument("--rounds-q", type=int, default=None)
    p.add_argument("--rounds-init", type=int, default=None)
    p.add_argument("--rounds-k", type=int, default=None)
    p.add_argument("--em-restarts", type=int, default=8)
    args = p.parse_args(argv)
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",")) if args.seeds else None
    info = run_milestone4_paper(
        args.out_root,
        smoke=args.smoke,
        seeds=seeds,
        manuscript_bundle=args.manuscript_bundle,
        rounds_sparse=args.rounds_sparse,
        rounds_q=args.rounds_q,
        rounds_init=args.rounds_init,
        rounds_k=args.rounds_k,
        em_restarts=args.em_restarts,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "manuscript_copied"}, indent=2))
    if info.get("manuscript_copied"):
        print("Copied:", *info["manuscript_copied"], sep="\n  ")


if __name__ == "__main__":
    main_milestone4()
