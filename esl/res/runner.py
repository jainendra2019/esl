"""RES experiment runner: Tier-1 methods on the committed GameSpec suite (Figure-1 protocol)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from esl.baselines.external_common import MatrixGameTask
from esl.experiments.figure1_dynamic_adaptation import (
    H_WINDOW,
    SA_FP_DEFAULT_P_STAY,
    SWITCH_HAZARD,
    SWITCH_INTERVAL,
    run_dynamic_cell,
)
from esl.experiments.stats_ci import mean_std_ci95
from esl.res.benchmark import (
    assert_family_constraints,
    default_benchmark_root,
    load_suite,
    min_pairwise_feature_distance,
)
from esl.res.game_spec import ExperimentSpec, GameSpec
from esl.res.stats import method_mean_order, paired_mean_ci95

TIER1_METHODS: tuple[str, ...] = ("ESL", "Fixed-K Bayesian", "SA-FP", "FP", "SOM")
# Extended clean 3×3 refresh methods (includes Online EM + OMD/MWU).
BASELINE_A3_METHODS: tuple[str, ...] = (
    "ESL",
    "Fixed-K Bayesian",
    "SA-FP",
    "FP",
    "SOM",
    "Online EM",
    "OMD/MWU",
)
# Expanded Adaptation flagship (no OMD); PPO after M2 freeze.
FLAGSHIP_A3_METHODS: tuple[str, ...] = (
    "ESL",
    "PPO",
    "Fixed-K Bayesian",
    "SA-FP",
    "FP",
    "SOM",
    "Online EM",
)
# Same as flagship without PPO (for environments without PyTorch).
FLAGSHIP_A3_METHODS_NO_PPO: tuple[str, ...] = tuple(
    m for m in FLAGSHIP_A3_METHODS if m != "PPO"
)
METHOD_SET_CHOICES: tuple[str, ...] = ("tier1", "flagship-a3", "flagship-a3-no-ppo")
RETRIEVABLE_A3_METHODS: tuple[str, ...] = ("ESL", "Fixed-K Bayesian", "SA-FP", "FP", "SOM")
RUN_NEW_A3_METHODS: tuple[str, ...] = ("Online EM", "OMD/MWU", "HMM-L")
# Bake-off-only allow-list (not injected into flagship packaging).
HMML_BAKEOFF_METHODS: tuple[str, ...] = ("ESL", "SA-FP", "Online EM", "HMM-L")


def resolve_method_set(name: str, *, num_actions: int) -> tuple[str, ...]:
    """Map a CLI ``--method-set`` name to an ordered method tuple.

    Flagship sets are restricted to the three-action protocol. ``tier1`` preserves
    the historical five-method default for both A=2 and A=3.
    """
    key = str(name).strip().lower()
    if key == "tier1":
        return TIER1_METHODS
    if key in ("flagship-a3", "flagship-a3-no-ppo"):
        if int(num_actions) != 3:
            raise ValueError(
                f"--method-set {key} is only supported for --num-actions 3 "
                f"(got num_actions={num_actions})"
            )
        if key == "flagship-a3":
            return FLAGSHIP_A3_METHODS
        return FLAGSHIP_A3_METHODS_NO_PPO
    raise ValueError(
        f"Unknown --method-set {name!r}; choose one of {list(METHOD_SET_CHOICES)}"
    )


def require_torch_for_ppo() -> None:
    """Fail fast with an actionable message when PPO is requested without PyTorch."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for --method-set flagship-a3 (PPO). "
            "Install torch (see README / requirements for optional PPO deps), "
            "or use --method-set flagship-a3-no-ppo."
        ) from exc


def _aggregate_method_order() -> list[str]:
    """Stable order covering tier-1, flagship (incl. PPO), and baseline-A3 methods."""
    return list(
        dict.fromkeys([*FLAGSHIP_A3_METHODS, *TIER1_METHODS, *BASELINE_A3_METHODS])
    )


def game_spec_to_task(spec: GameSpec) -> MatrixGameTask:
    """Adapter: committed GameSpec → MatrixGameTask with free-form payoffs."""
    return MatrixGameTask(
        key=spec.instance_id,
        label=f"{spec.family}:{spec.instance_id}",
        payoff_game="prisoners_dilemma",  # unused when payoff_row is set
        payoff_row=spec.payoff_row,
        payoff_col=spec.payoff_col,
    )


def run_res_cell(
    *,
    spec: GameSpec,
    method: str,
    seed: int,
    horizon: int = 1000,
    h_window: int = H_WINDOW,
    p_stay: float = SA_FP_DEFAULT_P_STAY,
    switch_interval: int = SWITCH_INTERVAL,
    switch_hazard: float = SWITCH_HAZARD,
    p_obs: float = 1.0,
    obs_noise: float = 0.0,
    force_categorical: bool = False,
    focal_role: str = "row",
    omd_c: float | None = None,
    omd_scope: str = "per_opponent",
    lam_br: float | None = None,
    prototype_init: str = "default",
    split_exogenous_rng: bool = False,
) -> dict[str, Any]:
    """
    Run one (game, method, seed) cell under the Figure-1-style dynamic protocol.

    - A=2 uses the paper `run_dynamic_cell` path (bit-exact with existing Figure 1 machinery).
    - A≥3 uses the categorical Tier-1 cell (`esl.res.categorical_cell`).
    - Set ``force_categorical=True`` for M3/M4 sweeps so p_obs=1 / ε=0 share the same cell path.
    - ``focal_role='col'`` swaps payoff matrices so the focal learner uses the column player's payoffs.
    - ``omd_scope``: ``per_opponent`` (primary) or ``global`` (secondary ablation).
    - ``lam_br``: optional soft-BR temperature for ESL / Fixed-K / SA-FP / Online EM (default paper λ=2).
    - ``prototype_init``: ``default`` (flagship) or ``fixed`` (ESL starts from shared `_fixed_logits`).
    - ``split_exogenous_rng``: if True, opponent actions use a cell-keyed stream separate from focal RNG
      (matched-init control). Default False preserves flagship bit-exactness.
    """
    allowed = set(TIER1_METHODS) | set(BASELINE_A3_METHODS) | set(FLAGSHIP_A3_METHODS) | set(HMML_BAKEOFF_METHODS)
    if method not in allowed:
        raise ValueError(f"Unknown RES method {method}; allowed={sorted(allowed)}")
    # A=2 paper path does not include Online EM / OMD/MWU / SA-FP extensions for categorical-only methods
    if method in (*RUN_NEW_A3_METHODS, "PPO") and not (
        force_categorical or int(spec.num_actions) != 2 or float(p_obs) != 1.0 or float(obs_noise) != 0.0
    ):
        # Force categorical path for new methods / PPO-A3 even on A=2 if ever requested
        force_categorical = True
    role = str(focal_role).lower().strip()
    if role not in ("row", "col"):
        raise ValueError(f"focal_role must be 'row' or 'col'; got {focal_role}")
    run_spec = spec
    if role == "col":
        from esl.res.asymmetric_benchmark import swap_roles

        run_spec = swap_roles(spec)
    task = game_spec_to_task(run_spec)
    use_categorical = (
        bool(force_categorical)
        or int(spec.num_actions) != 2
        or float(p_obs) != 1.0
        or float(obs_noise) != 0.0
        or method in RUN_NEW_A3_METHODS
        or method == "PPO"
    )
    if use_categorical:
        from esl.res.categorical_cell import run_categorical_cell
        from esl.res.omd_mwu import OMD_DEFAULT_SCOPE, OMD_PRIMARY_C

        kwargs: dict[str, Any] = dict(
            task=task,
            method=method,  # type: ignore[arg-type]
            seed=seed,
            horizon=horizon,
            num_actions=int(run_spec.num_actions),
            h_window=h_window,
            p_stay=p_stay,
            switch_interval=switch_interval,
            switch_hazard=switch_hazard,
            p_obs=float(p_obs),
            obs_noise=float(obs_noise),
        )
        if lam_br is not None:
            kwargs["lam_br"] = float(lam_br)
        if prototype_init != "default":
            kwargs["prototype_init"] = str(prototype_init)
        if split_exogenous_rng:
            kwargs["split_exogenous_rng"] = True
        if method == "OMD/MWU":
            kwargs["omd_c"] = float(OMD_PRIMARY_C if omd_c is None else omd_c)
            scope = str(omd_scope or OMD_DEFAULT_SCOPE)
            if scope not in ("per_opponent", "global"):
                raise ValueError(f"omd_scope must be per_opponent|global, got {scope!r}")
            kwargs["omd_scope"] = scope
        result = run_categorical_cell(**kwargs)
    else:
        if method not in TIER1_METHODS and method != "Online EM":
            # Figure-1 path supports Online EM; OMD is categorical-only
            raise ValueError(f"Method {method} requires the categorical cell path")
        result = run_dynamic_cell(
            task=task,
            method=method,  # type: ignore[arg-type]
            seed=seed,
            horizon=horizon,
            h_window=h_window,
            p_stay=p_stay,
            switch_interval=switch_interval,
            switch_hazard=switch_hazard,
        )
    summary = dict(result["summary"])
    summary["family"] = spec.family
    summary["instance_id"] = spec.instance_id
    summary["num_actions"] = spec.num_actions
    summary["p_obs"] = float(p_obs)
    summary["obs_noise"] = float(obs_noise)
    summary["focal_role"] = role
    return {"trajectory": result["trajectory"], "summary": summary}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def aggregate_family_metrics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = sorted({str(s["family"]) for s in summaries})
    method_order = _aggregate_method_order()
    methods = [m for m in method_order if any(s["method"] == m for s in summaries)]
    for family in families + ["all"]:
        for method in methods:
            vals = [
                float(s["psr"])
                for s in summaries
                if s["method"] == method and (family == "all" or s["family"] == family)
            ]
            if not vals:
                continue
            st = mean_std_ci95(vals)
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "metric": "psr",
                    "n": int(st["n"]),
                    "mean": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                }
            )
    return rows


def paired_method_gaps(
    summaries: list[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    metric: str = "psr",
) -> dict[str, Any]:
    """Paired (seed, instance[, focal_role]) gaps: metric(a) - metric(b); positive ⇒ a worse if lower-is-better."""
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for s in summaries:
        raw = s.get(metric, "")
        if raw == "" or raw is None or raw == "N/A":
            continue
        role = str(s.get("focal_role", "row"))
        key = (str(s["instance_id"]), int(s["seed"]), role)
        by_key[key][str(s["method"])] = float(raw)
    diffs: list[float] = []
    for key, methods in by_key.items():
        if method_a in methods and method_b in methods:
            diffs.append(methods[method_a] - methods[method_b])
    stats = paired_mean_ci95(diffs)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "diff_meaning": f"{method_a}_minus_{method_b}",
        **stats,
    }


def qualitative_stability_note(family_rows: list[dict[str, Any]]) -> str:
    """Summarize whether ESL remains competitive vs SA-FP / Fixed-K on mean PSR."""
    lines: list[str] = []
    for family in sorted({r["family"] for r in family_rows}):
        means = {
            r["method"]: float(r["mean"])
            for r in family_rows
            if r["family"] == family and r["metric"] == "psr"
        }
        if not means:
            continue
        order = method_mean_order(means, lower_is_better=True)
        lines.append(f"- `{family}` mean-PSR order (lower better): {' < '.join(order)}")
        if "ESL" in means and "SA-FP" in means:
            gap = means["SA-FP"] - means["ESL"]
            lines.append(f"  ESL vs SA-FP mean gap (SA-FP−ESL): {gap:.3f}")
    return "\n".join(lines) if lines else "- (no family aggregates)"


def aggregate_all_metrics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Family + all aggregates for PSR/DPR/LTE/recovery (early_gap_auc)."""
    metric_keys = (
        ("psr", "psr"),
        ("dpr", "dpr"),
        ("lte_auc", "lte"),
        ("early_gap_auc", "recovery"),
    )
    rows: list[dict[str, Any]] = []
    families = sorted({str(s["family"]) for s in summaries})
    method_order = _aggregate_method_order()
    methods = [m for m in method_order if any(s["method"] == m for s in summaries)]
    for family in families + ["all"]:
        for method in methods:
            for src, label in metric_keys:
                vals = [
                    float(s[src])
                    for s in summaries
                    if s["method"] == method
                    and (family == "all" or s["family"] == family)
                    and s.get(src, "") not in ("", "N/A", None)
                ]
                if not vals:
                    continue
                st = mean_std_ci95(vals)
                rows.append(
                    {
                        "family": family,
                        "method": method,
                        "metric": label,
                        "n": int(st["n"]),
                        "mean": st["mean"],
                        "ci95_low": st["ci95_low"],
                        "ci95_high": st["ci95_high"],
                    }
                )
    return rows


def paired_gaps_all_metrics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = {str(s["method"]) for s in summaries}
    pairs = [
        ("SA-FP", "ESL"),
        ("Fixed-K Bayesian", "ESL"),
        ("Fixed-K Bayesian", "SA-FP"),
    ]
    for m in ("OMD/MWU", "Online EM", "SOM", "FP", "PPO"):
        if m in present and "ESL" in present:
            pairs.append((m, "ESL"))
    metrics = ("psr", "dpr", "lte_auc", "early_gap_auc")
    rows: list[dict[str, Any]] = []
    for a, b in pairs:
        for metric in metrics:
            row = paired_method_gaps(summaries, method_a=a, method_b=b, metric=metric)
            rows.append(row)
    return rows


def run_res_m1(
    *,
    out_dir: Path,
    benchmark_root: Path | None = None,
    seeds: Sequence[int] = (0, 1),
    horizon: int = 200,
    methods: Sequence[str] = TIER1_METHODS,
    num_actions: int = 2,
    p_obs: float = 1.0,
    obs_noise: float = 0.0,
    force_categorical: bool = False,
    milestone_override: str | None = None,
    prefix_override: str | None = None,
    focal_roles: Sequence[str] = ("row",),
    omd_c: float | None = None,
) -> dict[str, Any]:
    """
    M1/M2/M3/M4 runner over committed suite filtered by ``num_actions``.

    Default horizon/seeds are smoke-friendly; pass paper settings for full runs.
    Use a fixed subset via ``benchmark_root`` filter or pre-filtered suite for sweeps.
    Pass ``focal_roles=('row','col')`` for asymmetric dual-role evaluation.
    """
    root = benchmark_root or default_benchmark_root()
    specs = load_suite(root, num_actions=num_actions)
    if not specs:
        raise FileNotFoundError(f"No GameSpec instances under {root} for num_actions={num_actions}")
    for spec in specs:
        assert_family_constraints(spec)

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    roles = tuple(str(r) for r in focal_roles) or ("row",)
    for spec in specs:
        for role in roles:
            for method in methods:
                for seed in seeds:
                    cell = run_res_cell(
                        spec=spec,
                        method=method,
                        seed=int(seed),
                        horizon=int(horizon),
                        p_obs=float(p_obs),
                        obs_noise=float(obs_noise),
                        force_categorical=bool(force_categorical),
                        focal_role=role,
                        omd_c=omd_c,
                    )
                    summaries.append(cell["summary"])

    if milestone_override:
        milestone = milestone_override
    elif float(obs_noise) > 0.0:
        milestone = "M4"
    elif float(p_obs) < 1.0:
        milestone = "M3"
    elif num_actions == 2:
        milestone = "M1"
    else:
        milestone = f"M2_A{num_actions}"

    if prefix_override:
        prefix = prefix_override
    elif milestone == "M4":
        prefix = "res_m4"
    elif milestone == "M3":
        prefix = "res_m3"
    elif num_actions == 2:
        prefix = "res_m1"
    else:
        prefix = f"res_m2_a{num_actions}"

    cell_cols = [
        "family",
        "instance_id",
        "num_actions",
        "method",
        "seed",
        "horizon",
        "psr",
        "dpr",
        "lte_auc",
        "early_gap_auc",
        "p_obs",
        "obs_noise",
        "sa_fp_p_stay",
        "focal_role",
        "omd_c",
        "online_em_window",
    ]
    cells_csv = out_dir / f"{prefix}_cells.csv"
    _write_csv(cells_csv, summaries, cell_cols)

    family_rows = aggregate_all_metrics(summaries)
    family_csv = out_dir / f"{prefix}_family_summary.csv"
    _write_csv(
        family_csv,
        family_rows,
        ["family", "method", "metric", "n", "mean", "ci95_low", "ci95_high"],
    )

    paired_rows = paired_gaps_all_metrics(summaries)
    paired_csv = out_dir / f"{prefix}_paired.csv"
    # Keep legacy PSR-only filename for smoke tests that still look for it.
    paired_psr_only = [r for r in paired_rows if r["metric"] == "psr"]
    _write_csv(
        out_dir / f"{prefix}_paired_psr.csv",
        paired_psr_only,
        ["method_a", "method_b", "metric", "diff_meaning", "n", "mean", "std", "ci95_low", "ci95_high", "cohen_dz"],
    )
    _write_csv(
        paired_csv,
        paired_rows,
        ["method_a", "method_b", "metric", "diff_meaning", "n", "mean", "std", "ci95_low", "ci95_high", "cohen_dz"],
    )

    diversity = min_pairwise_feature_distance(specs)
    # qualitative note uses PSR family rows
    psr_family = [r for r in family_rows if r["metric"] == "psr"]
    note = qualitative_stability_note(psr_family)
    report = out_dir / f"{prefix}_report.md"
    report.write_text(
        "\n".join(
            [
                f"# RES {milestone} Report (A={num_actions} fixed suite)",
                "",
                f"- Benchmark root: `{root}`",
                f"- Instances: `{len(specs)}`",
                f"- Seeds: `{list(seeds)}`",
                f"- Horizon: `{horizon}`",
                f"- Methods: `{list(methods)}`",
                f"- p_obs: `{p_obs}`",
                f"- obs_noise: `{obs_noise}`",
                f"- force_categorical: `{force_categorical}`",
                "",
                "## Diversity audit (min pairwise feature distance)",
                "",
                *[f"- `{k}`: `{v}`" for k, v in diversity.items()],
                "",
                "## Qualitative stability (mean PSR)",
                "",
                note,
                "",
                "## Exit checklist",
                "",
                f"- [x] Tier-1 baselines run on committed A={num_actions} suite",
                "- [x] Family-level summary written (PSR/DPR/LTE/recovery)",
                "- [x] Paired CI / effect sizes computed",
                "- [x] Diversity distances reported",
                "",
            ]
        ),
        encoding="utf-8",
    )
    gate = {
        "milestone": milestone,
        "num_actions": num_actions,
        "p_obs": float(p_obs),
        "obs_noise": float(obs_noise),
        "force_categorical": bool(force_categorical),
        "n_instances": len(specs),
        "n_summaries": len(summaries),
        "diversity_min_pairwise": diversity,
        "cells_csv": str(cells_csv),
        "family_csv": str(family_csv),
        "paired_csv": str(paired_csv),
        "report": str(report),
    }
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="RES runner (committed GameSpec suite)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/res/m1"))
    p.add_argument("--benchmark-root", type=Path, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--num-actions", type=int, default=2, choices=(2, 3))
    p.add_argument("--p-obs", type=float, default=1.0)
    p.add_argument("--obs-noise", type=float, default=0.0)
    p.add_argument("--full", action="store_true", help="Paper-like seeds 0..9 and horizon 1000")
    p.add_argument(
        "--method-set",
        type=str,
        default="tier1",
        choices=METHOD_SET_CHOICES,
        help=(
            "Method suite: 'tier1' (default; 5 methods), "
            "'flagship-a3' (7 methods including PPO; A=3 only), "
            "'flagship-a3-no-ppo' (6 methods; A=3 only). "
            "OMD/MWU is not part of these sets."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    seeds: Sequence[int] = tuple(range(10)) if args.full else tuple(args.seeds)
    horizon = 1000 if args.full else int(args.horizon)
    methods = resolve_method_set(args.method_set, num_actions=int(args.num_actions))
    if "PPO" in methods:
        require_torch_for_ppo()
    gate = run_res_m1(
        out_dir=args.out_dir,
        benchmark_root=args.benchmark_root,
        seeds=seeds,
        horizon=horizon,
        methods=methods,
        num_actions=int(args.num_actions),
        p_obs=float(args.p_obs),
        obs_noise=float(args.obs_noise),
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
