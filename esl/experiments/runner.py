"""Execute preset configs under runs/neurips/... with run_manifest.json sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from esl.config import ESLConfig
from esl.experiments import presets as neurips_presets
from esl.experiments.aggregate import write_aggregate_csv
from esl.experiments.canonical_io import write_manifest_json
from esl.experiment_registry import build_esl_manifest_dict
from esl.experiments.manifest import write_run_manifest
from esl.trainer import run_esl


def default_variant_for_preset(preset: str) -> str:
    """Default CLI variant when none is passed (registry / programmatic callers)."""
    defaults: dict[str, str] = {
        "recovery_sparse_obs": "1.0",
        "recovery_short_horizon": "500",
        "recovery_lr_sweep": "12",
        "recovery_init_noise_sweep": "0.05",
        "recovery_Q_sweep": "15",
    }
    return defaults.get(preset, "")


def esl_config_for_preset(
    preset: str,
    *,
    seed: int,
    smoke: bool,
    variant: str | None = None,
) -> tuple[ESLConfig, str]:
    """
    Public API: build (ESLConfig, manifest_variant_slug) for a named NeurIPS preset.

    When ``variant`` is None, uses :func:`default_variant_for_preset`.
    """
    v = variant if variant is not None else default_variant_for_preset(preset)
    return _config_for_cli_preset(preset, seed=seed, smoke=smoke, variant=v)


def _out_dir_for_run(out_root: Path, preset: str, variant: str, seed: int) -> Path:
    slug = variant.replace(".", "p").replace(" ", "_") if variant else "default"
    return (out_root / preset / slug / f"seed_{seed}").resolve()


def run_named_preset(
    preset: str,
    *,
    seed: int,
    out_root: Path,
    smoke: bool = False,
    variant: str = "",
    target_interaction_budget: int | None = None,
    extra_manifest: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> Path:
    cfg, manifest_variant = _config_for_cli_preset(
        preset, seed=seed, smoke=smoke, variant=variant
    )
    run_dir = _out_dir_for_run(out_root, preset, manifest_variant, seed)
    manifest: dict[str, Any] = {
        "preset": preset,
        "variant": manifest_variant,
        "seed": seed,
        "smoke": smoke,
        "target_interaction_budget": target_interaction_budget,
    }
    if manifest["target_interaction_budget"] is None and preset == "recovery_short_horizon":
        vs = variant.strip()
        if vs.isdigit():
            manifest["target_interaction_budget"] = int(vs)
        elif smoke:
            manifest["target_interaction_budget"] = 30
    if extra_manifest:
        manifest.update(extra_manifest)
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    run_esl(cfg, run_dir=run_dir)
    eid = experiment_id or f"esl.neurips.{preset}"
    canonical = build_esl_manifest_dict(
        experiment_id=eid,
        seed=seed,
        smoke=smoke,
        manifest_variant=manifest_variant,
        neurips_preset=preset,
        extra={"run_manifest": manifest},
    )
    write_manifest_json(run_dir / "manifest.json", canonical)
    return run_dir


def _config_for_cli_preset(
    preset: str,
    *,
    seed: int,
    smoke: bool,
    variant: str,
) -> tuple[ESLConfig, str]:
    from dataclasses import replace

    v = variant.strip()
    if preset == "recovery_flagship":
        cfg = neurips_presets.recovery_flagship_cfg(seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=5)
            cfg.validate()
        return cfg, v or "long"

    if preset == "recovery_fixed_prototype_baseline":
        cfg = neurips_presets.recovery_fixed_prototype_baseline_cfg(seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=5)
            cfg.validate()
        return cfg, v or "freeze"

    if preset == "recovery_failure_case":
        cfg = neurips_presets.recovery_failure_case_cfg(seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=8)
            cfg.validate()
        return cfg, v or "weak"

    if preset == "recovery_sparse_obs":
        if not v:
            raise ValueError("recovery_sparse_obs requires --variant with p_obs, e.g. 0.5")
        p_obs = float(v)
        cfg = neurips_presets.recovery_sparse_obs_cfg(p_obs=p_obs, seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=5)
            cfg.validate()
        return cfg, f"p_obs_{p_obs}"

    if preset == "recovery_short_horizon":
        if not v:
            raise ValueError("recovery_short_horizon requires --variant with interaction budget, e.g. 500")
        budget = int(v)
        if smoke:
            cfg = neurips_presets.recovery_short_horizon_cfg(
                interaction_budget=30, interactions_per_round=10, seed=seed
            )
            # Unique path per requested paper budget even though smoke uses 30 interactions.
            return cfg, f"budget_{budget}_smoke"
        cfg = neurips_presets.recovery_short_horizon_cfg(
            interaction_budget=budget, interactions_per_round=10, seed=seed
        )
        return cfg, f"budget_{budget}"

    if preset == "recovery_lr_sweep":
        if not v:
            raise ValueError("recovery_lr_sweep requires --variant lr value, e.g. 12")
        lr = float(v)
        cfg = neurips_presets.recovery_lr_sweep_cfg(prototype_lr_scale=lr, seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=5)
            cfg.validate()
        return cfg, f"lr_{lr}"

    if preset == "recovery_init_noise_sweep":
        if not v:
            raise ValueError("recovery_init_noise_sweep requires --variant noise, e.g. 0.05")
        noise = float(v)
        cfg = neurips_presets.recovery_init_noise_sweep_cfg(init_noise=noise, seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=5)
            cfg.validate()
        return cfg, f"init_noise_{noise}"

    if preset == "recovery_Q_sweep":
        if not v:
            raise ValueError("recovery_Q_sweep requires --variant Q, e.g. 15")
        q = int(float(v))
        cfg = neurips_presets.recovery_Q_sweep_cfg(Q=q, seed=seed)
        if smoke:
            cfg = replace(cfg, num_rounds=5)
            cfg.validate()
        return cfg, f"Q_{q}"

    raise ValueError(f"unknown preset {preset!r}; use --list-presets")


def list_preset_names() -> list[str]:
    return [
        "recovery_flagship",
        "recovery_fixed_prototype_baseline",
        "recovery_failure_case",
        "recovery_sparse_obs",
        "recovery_short_horizon",
        "recovery_lr_sweep",
        "recovery_init_noise_sweep",
        "recovery_Q_sweep",
    ]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="ESL NeurIPS-style experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run one preset")
    pr.add_argument("--preset", required=True)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--out-root", type=Path, default=Path("runs/neurips"))
    pr.add_argument("--variant", default="")
    pr.add_argument("--smoke", action="store_true")

    pl = sub.add_parser("list-presets", help="print preset names")

    pa = sub.add_parser("aggregate", help="build summary CSV from runs under a root")
    pa.add_argument("root", type=Path, nargs="?", default=Path("runs/neurips"))
    pa.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("runs/neurips/_aggregates/summary_all_runs.csv"),
    )

    ps = sub.add_parser(
        "sparse-pobs-sweep",
        help="p_obs in {1.0,0.5,0.3,0.2} (flagship geometry) + final CE vs p_obs plot",
    )
    ps.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sparse_pobs_sweep"),
    )
    ps.add_argument(
        "--rounds",
        type=int,
        default=3000,
        help="environment rounds per point (use 10000 for paper-length)",
    )
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--no-plot", action="store_true")
    ps.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate PNG from existing sparse_pobs_summary.csv",
    )
    ps.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip observation log + offline baselines per run",
    )
    ps.add_argument(
        "--clustering-only-baselines",
        action="store_true",
        help="K-means + FCM only (no EM/Oracle); Milestone 3 paper-facing",
    )
    ps.add_argument("--em-restarts", type=int, default=8)

    pq = sub.add_parser(
        "q-recovery-sweep",
        help="recovery flagship: sweep prototype_update_every Q + MCE plot",
    )
    pq.add_argument("--out-root", type=Path, default=Path("runs/q_recovery_sweep"))
    pq.add_argument("--rounds", type=int, default=3000)
    pq.add_argument("--seed", type=int, default=42)
    pq.add_argument("--q-values", type=str, default="1,5,15")
    pq.add_argument("--no-plot", action="store_true")
    pq.add_argument("--no-baselines", action="store_true")
    pq.add_argument("--clustering-only-baselines", action="store_true")
    pq.add_argument("--em-restarts", type=int, default=8)

    pm4 = sub.add_parser(
        "milestone4-paper",
        help="Milestone 4: multi-seed sweeps, CI bands, mean±std tables; optional manuscript staging",
    )
    pm4.add_argument("--out-root", type=Path, default=Path("runs/milestone4_paper"))
    pm4.add_argument("--smoke", action="store_true")
    pm4.add_argument("--manuscript-bundle", type=Path, default=None)
    pm4.add_argument("--seeds", type=str, default=None)
    pm4.add_argument("--rounds-sparse", type=int, default=None)
    pm4.add_argument("--rounds-q", type=int, default=None)
    pm4.add_argument("--rounds-init", type=int, default=None)
    pm4.add_argument("--rounds-k", type=int, default=None)
    pm4.add_argument("--em-restarts", type=int, default=8)

    pm5 = sub.add_parser(
        "milestone5-pack",
        help="Milestone 5: heterogeneous IPD adaptation (ESL, K=1 ablation, clustering+BR, PPO)",
    )
    pm5.add_argument("--out-root", type=Path, default=Path("runs/milestone5_adaptation"))
    pm5.add_argument("--manuscript-bundle", type=Path, default=None)
    pm5.add_argument("--smoke", action="store_true")
    pm5.add_argument("--seeds", type=str, default=None)
    pm5.add_argument("--rounds", type=int, default=800)
    pm5.add_argument("--include-tft", action="store_true")
    pm5.add_argument("--skip-ppo", action="store_true")
    pm5.add_argument(
        "--resume-skip-complete",
        action="store_true",
        help="skip (seed, method) runs that already have summary + focal sidecar on disk",
    )

    pm6 = sub.add_parser(
        "milestone6-pack",
        help="Milestone 6: structure–performance, K misspec, p_obs stress, belief ablation",
    )
    pm6.add_argument("--out-root", type=Path, default=Path("runs/milestone6_robustness"))
    pm6.add_argument("--manuscript-bundle", type=Path, default=None)
    pm6.add_argument("--smoke", action="store_true")
    pm6.add_argument("--seeds", type=str, default=None)
    pm6.add_argument("--rounds", type=int, default=500)
    pm6.add_argument(
        "--matching-pennies",
        action="store_true",
        help="include optional matching pennies experiment",
    )

    pm7 = sub.add_parser(
        "milestone7-paper",
        help="Milestone 7: frozen M4–M6 paper runs, audit, main/appendix manuscript bundle",
    )
    pm7.add_argument("--out-root", type=Path, default=Path("runs/milestone7_paper"))
    pm7.add_argument("--manuscript-bundle", type=Path, default=None)
    pm7.add_argument("--smoke", action="store_true")
    pm7.add_argument(
        "--frozen",
        type=Path,
        default=None,
        help="path to milestone7_frozen.json (default: esl/experiments/configs/…)",
    )
    pm7.add_argument(
        "--skip-m4-if-ready",
        action="store_true",
        help="reuse existing milestone4 under --out-root when aggregate CSV exists",
    )
    pm7.add_argument(
        "--m5-resume-skip-complete",
        action="store_true",
        help="skip M5 cells that already finished (summary + focal sidecar on disk)",
    )

    pm8 = sub.add_parser(
        "milestone8-camera-ready",
        help="Milestone 8: fig1–fig5 mapping, figure_index.json, narrative polish (no re-runs)",
    )
    pm8.add_argument(
        "--manuscript-bundle",
        type=Path,
        default=Path("manuscript_bundle"),
    )

    pm9 = sub.add_parser(
        "milestone9-submission",
        help="Milestone 9: submission bundle, figure verification, reviewer sim, notation audit",
    )
    pm9.add_argument(
        "--manuscript-bundle",
        type=Path,
        default=Path("manuscript_bundle"),
    )
    pm9.add_argument(
        "--min-fig-bytes",
        type=int,
        default=512,
        help="minimum file size for each main figure (use ~5k+ for non-smoke assets)",
    )

    pfd = sub.add_parser(
        "figure-diagnostics",
        help="Post–M7: effect sizes, CI flags, seed stability, KEEP/MOVE/DROP figure report",
    )
    pfd.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    pfd.add_argument(
        "--min-seeds",
        type=int,
        default=10,
        help="minimum seeds before flagging thin aggregates",
    )
    pfd.add_argument(
        "--min-fig-bytes",
        type=int,
        default=8000,
        help="KEEP if each main figure PNG is at least this many bytes",
    )

    pm3 = sub.add_parser(
        "milestone3-pack",
        help="Milestone 3 CI smoke: sparse + Q sweep + freeze diagnostic; optional manuscript staging",
    )
    pm3.add_argument("--out-root", type=Path, default=Path("runs/milestone3_smoke"))
    pm3.add_argument(
        "--manuscript-bundle",
        type=Path,
        default=None,
        help="e.g. manuscript_bundle to copy tables/figures",
    )
    pm3.add_argument("--seed", type=int, default=0)
    pm3.add_argument("--no-plot", action="store_true")

    plc = sub.add_parser(
        "prefix-learning-curve",
        help="ESL vs batch MCE along prefix horizons (needs logged run)",
    )
    plc.add_argument("--run-dir", type=Path, required=True)
    plc.add_argument("--n-prefix-points", type=int, default=10)
    plc.add_argument("--em-restarts", type=int, default=4)
    plc.add_argument("--no-plot", action="store_true")

    ab = sub.add_parser(
        "ablation-ladder",
        help="full vs freeze_prototype vs learning_frozen (CSV + bar figure)",
    )
    ab.add_argument("--out-root", type=Path, default=Path("runs/ablation_ladder"))
    ab.add_argument("--rounds", type=int, default=200)
    ab.add_argument("--seed", type=int, default=42)
    ab.add_argument("--no-plot", action="store_true")

    ad = sub.add_parser(
        "adaptation-payoff-sweep",
        help="adaptation mode: mean payoff vs adaptation_lambda (ESL-only figure)",
    )
    ad.add_argument("--out-root", type=Path, default=Path("runs/adaptation_payoff_sweep"))
    ad.add_argument("--rounds", type=int, default=120)
    ad.add_argument("--seed", type=int, default=42)
    ad.add_argument("--no-plot", action="store_true")

    mpg = sub.add_parser(
        "main-performance-grid",
        help="Main performance smoke: tasks x regimes x external baseline audit",
    )
    mpg.add_argument("--out-root", type=Path, default=Path("runs/main_performance_grid"))
    mpg.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    mpg.add_argument("--smoke", action="store_true")
    mpg.add_argument("--paper", action="store_true")
    mpg.add_argument("--seeds", type=str, default=None)
    mpg.add_argument("--horizon", type=int, default=None)

    f1dyn = sub.add_parser(
        "figure1-dynamic-adaptation",
        help="Figure 1: dynamics-aware post-switch adaptation evaluation",
    )
    f1dyn.add_argument("--out-root", type=Path, default=Path("runs/figure1_dynamic_adaptation"))
    f1dyn.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    f1dyn.add_argument("--seeds", type=str, default="0,1,2")
    f1dyn.add_argument("--horizon", type=int, default=500)
    f1dyn.add_argument("--h-window", type=int, default=100)
    f1dyn.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    f1dyn.add_argument("--smoke", action="store_true")

    f2abl = sub.add_parser(
        "figure2-ablation",
        help="Figure 2: component-level ablation analysis under type shifts",
    )
    f2abl.add_argument("--out-root", type=Path, default=Path("runs/figure2_ablation"))
    f2abl.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    f2abl.add_argument("--seeds", type=str, default="0,1,2")
    f2abl.add_argument("--horizon", type=int, default=500)
    f2abl.add_argument("--h-window", type=int, default=100)
    f2abl.add_argument("--tasks", type=str, default="ipd,stag_hunt,matching_pennies")
    f2abl.add_argument("--smoke", action="store_true")

    f3dyn = sub.add_parser(
        "figure3-dynamics",
        help="Figure 3: ESL closed-loop dynamics diagnostics",
    )
    f3dyn.add_argument("--out-dir", type=Path, default=Path("runs/figure3_dynamics"))
    f3dyn.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    f3dyn.add_argument("--seeds", nargs="*", default=["0,1,2,3,4,5,6,7,8,9"])
    f3dyn.add_argument("--horizon", type=int, default=1000)
    f3dyn.add_argument("--games", nargs="*", default=["ipd", "stag_hunt", "matching_pennies"])
    f3dyn.add_argument("--batch-size", type=int, default=2)
    f3dyn.add_argument("--rolling-window", type=int, default=5)

    f4rm = sub.add_parser(
        "figure4-rm-dynamics",
        help="Figure 4: Robbins--Monro ESL dynamics diagnostic",
    )
    f4rm.add_argument("--out-dir", type=Path, default=Path("runs/figure4_rm_dynamics"))
    f4rm.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    f4rm.add_argument("--seeds", nargs="*", default=["0,1,2,3,4,5,6,7,8,9"])
    f4rm.add_argument("--horizon", type=int, default=3000)
    f4rm.add_argument("--games", nargs="*", default=["ipd", "stag_hunt", "matching_pennies"])
    f4rm.add_argument("--batch-size", type=int, default=50)
    f4rm.add_argument("--rm-c", type=float, default=0.5)
    f4rm.add_argument("--eta-reg", type=float, default=1e-3)
    f4rm.add_argument("--rolling-window", type=int, default=5)

    appmech = sub.add_parser(
        "appendix-mechanistic-analysis",
        help="Appendix: switch difficulty, ambiguity, and self-consistency analysis",
    )
    appmech.add_argument("--output-dir", type=Path, default=Path("outputs/mechanistic_analysis"))
    appmech.add_argument("--seeds", nargs="*", default=["0,1,2,3,4,5,6,7,8,9"])
    appmech.add_argument("--games", nargs="*", default=["ipd", "stag_hunt", "matching_pennies"])
    appmech.add_argument("--methods", nargs="*", default=["ESL", "Fixed-K Bayesian", "Online EM"])
    appmech.add_argument("--horizon", type=int, default=1000)

    args = p.parse_args(argv)

    if args.cmd == "list-presets":
        for n in list_preset_names():
            print(n)
        return
    if args.cmd == "run":
        run_dir = run_named_preset(
            args.preset,
            seed=args.seed,
            out_root=args.out_root,
            smoke=args.smoke,
            variant=args.variant,
            target_interaction_budget=None,
        )
        print(f"Wrote: {run_dir}")
        return
    if args.cmd == "aggregate":
        out = write_aggregate_csv(args.root, args.output)
        print(f"Wrote: {out}")
        return
    if args.cmd == "prefix-learning-curve":
        from esl.experiments.prefix_learning_curve import run_prefix_learning_curve

        cpath, ppath = run_prefix_learning_curve(
            args.run_dir,
            n_prefix_points=args.n_prefix_points,
            em_restarts=args.em_restarts,
            plot=not args.no_plot,
        )
        print(f"Wrote: {cpath}")
        if ppath:
            print(f"Wrote: {ppath}")
        return

    if args.cmd == "ablation-ladder":
        from esl.experiments.ablation_ladder import run_ablation_ladder

        cpath, ppath = run_ablation_ladder(
            out_root=args.out_root,
            num_rounds=args.rounds,
            seed=args.seed,
            plot=not args.no_plot,
        )
        print(f"Wrote: {cpath}")
        if ppath:
            print(f"Wrote: {ppath}")
        return

    if args.cmd == "adaptation-payoff-sweep":
        from esl.experiments.adaptation_payoff_sweep import run_adaptation_payoff_sweep

        cpath, ppath = run_adaptation_payoff_sweep(
            out_root=args.out_root,
            num_rounds=args.rounds,
            seed=args.seed,
            plot=not args.no_plot,
        )
        print(f"Wrote: {cpath}")
        if ppath:
            print(f"Wrote: {ppath}")
        return

    if args.cmd == "main-performance-grid":
        from esl.experiments.main_performance_grid import main_performance_grid_cli

        argv_mpg = ["--out-root", str(args.out_root), "--manuscript-bundle", str(args.manuscript_bundle)]
        if args.smoke:
            argv_mpg.append("--smoke")
        if args.paper:
            argv_mpg.append("--paper")
        if args.seeds:
            argv_mpg += ["--seeds", args.seeds]
        if args.horizon is not None:
            argv_mpg += ["--horizon", str(args.horizon)]
        main_performance_grid_cli(argv_mpg)
        return

    if args.cmd == "figure1-dynamic-adaptation":
        from esl.experiments.figure1_dynamic_adaptation import main as figure1_dynamic_main

        argv_dyn = [
            "--out-root",
            str(args.out_root),
            "--manuscript-bundle",
            str(args.manuscript_bundle),
            "--seeds",
            str(args.seeds),
            "--horizon",
            str(args.horizon),
            "--h-window",
            str(args.h_window),
            "--tasks",
            str(args.tasks),
        ]
        if args.smoke:
            argv_dyn.append("--smoke")
        figure1_dynamic_main(argv_dyn)
        return

    if args.cmd == "figure2-ablation":
        from esl.experiments.figure2_ablation import main as figure2_ablation_main

        argv_f2 = [
            "--out-root",
            str(args.out_root),
            "--manuscript-bundle",
            str(args.manuscript_bundle),
            "--seeds",
            str(args.seeds),
            "--horizon",
            str(args.horizon),
            "--h-window",
            str(args.h_window),
            "--tasks",
            str(args.tasks),
        ]
        if args.smoke:
            argv_f2.append("--smoke")
        figure2_ablation_main(argv_f2)
        return

    if args.cmd == "figure3-dynamics":
        from esl.experiments.figure3_dynamics import main as figure3_dynamics_main

        argv_f3 = [
            "--out-dir",
            str(args.out_dir),
            "--manuscript-bundle",
            str(args.manuscript_bundle),
            "--horizon",
            str(args.horizon),
            "--batch-size",
            str(args.batch_size),
            "--rolling-window",
            str(args.rolling_window),
            "--seeds",
            *[str(x) for x in args.seeds],
            "--games",
            *[str(x) for x in args.games],
        ]
        figure3_dynamics_main(argv_f3)
        return

    if args.cmd == "figure4-rm-dynamics":
        from esl.experiments.figure4_rm_dynamics import main as figure4_rm_dynamics_main

        argv_f4 = [
            "--out-dir",
            str(args.out_dir),
            "--manuscript-bundle",
            str(args.manuscript_bundle),
            "--horizon",
            str(args.horizon),
            "--batch-size",
            str(args.batch_size),
            "--rm-c",
            str(args.rm_c),
            "--eta-reg",
            str(args.eta_reg),
            "--rolling-window",
            str(args.rolling_window),
            "--seeds",
            *[str(x) for x in args.seeds],
            "--games",
            *[str(x) for x in args.games],
        ]
        figure4_rm_dynamics_main(argv_f4)
        return

    if args.cmd == "appendix-mechanistic-analysis":
        from esl.experiments.appendix_mechanistic_analysis import main as appendix_mechanistic_main

        argv_app = [
            "--output-dir",
            str(args.output_dir),
            "--horizon",
            str(args.horizon),
            "--seeds",
            *[str(x) for x in args.seeds],
            "--games",
            *[str(x) for x in args.games],
            "--methods",
            *[str(x) for x in args.methods],
        ]
        appendix_mechanistic_main(argv_app)
        return

    if args.cmd == "sparse-pobs-sweep":
        from esl.experiments.sparse_pobs_sweep import (
            main_sparse_pobs,
        )

        main_sparse_pobs(
            [
                "--out-root",
                str(args.out_root),
                "--rounds",
                str(args.rounds),
                "--seed",
                str(args.seed),
                "--em-restarts",
                str(args.em_restarts),
            ]
            + (["--no-plot"] if args.no_plot else [])
            + (["--plot-only"] if args.plot_only else [])
            + (["--no-baselines"] if args.no_baselines else [])
            + (["--clustering-only-baselines"] if args.clustering_only_baselines else [])
        )
        return

    if args.cmd == "q-recovery-sweep":
        from esl.experiments.q_recovery_sweep import main_q_recovery

        main_q_recovery(
            [
                "--out-root",
                str(args.out_root),
                "--rounds",
                str(args.rounds),
                "--seed",
                str(args.seed),
                "--q-values",
                str(args.q_values),
                "--em-restarts",
                str(args.em_restarts),
            ]
            + (["--no-plot"] if args.no_plot else [])
            + (["--no-baselines"] if args.no_baselines else [])
            + (["--clustering-only-baselines"] if args.clustering_only_baselines else [])
        )
        return

    if args.cmd == "milestone3-pack":
        from esl.experiments.milestone3_pack import main_milestone3_pack

        argv_m3 = ["--out-root", str(args.out_root), "--seed", str(args.seed)]
        if args.manuscript_bundle is not None:
            argv_m3 += ["--manuscript-bundle", str(args.manuscript_bundle)]
        if args.no_plot:
            argv_m3.append("--no-plot")
        main_milestone3_pack(argv_m3)
        return

    if args.cmd == "milestone4-paper":
        from esl.experiments.milestone4_paper import main_milestone4

        argv4 = ["--out-root", str(args.out_root)]
        if args.smoke:
            argv4.append("--smoke")
        if args.manuscript_bundle is not None:
            argv4 += ["--manuscript-bundle", str(args.manuscript_bundle)]
        if args.seeds:
            argv4 += ["--seeds", args.seeds]
        if args.rounds_sparse is not None:
            argv4 += ["--rounds-sparse", str(args.rounds_sparse)]
        if args.rounds_q is not None:
            argv4 += ["--rounds-q", str(args.rounds_q)]
        if args.rounds_init is not None:
            argv4 += ["--rounds-init", str(args.rounds_init)]
        if args.rounds_k is not None:
            argv4 += ["--rounds-k", str(args.rounds_k)]
        argv4 += ["--em-restarts", str(args.em_restarts)]
        main_milestone4(argv4)
        return

    if args.cmd == "milestone5-pack":
        from esl.experiments.milestone5_pack import main_milestone5

        argv5 = ["--out-root", str(args.out_root), "--rounds", str(args.rounds)]
        if args.smoke:
            argv5.append("--smoke")
        if args.manuscript_bundle is not None:
            argv5 += ["--manuscript-bundle", str(args.manuscript_bundle)]
        if args.seeds:
            argv5 += ["--seeds", args.seeds]
        if args.include_tft:
            argv5.append("--include-tft")
        if args.skip_ppo:
            argv5.append("--skip-ppo")
        if args.resume_skip_complete:
            argv5.append("--resume-skip-complete")
        main_milestone5(argv5)
        return

    if args.cmd == "milestone6-pack":
        from esl.experiments.milestone6_pack import main_milestone6

        argv6 = ["--out-root", str(args.out_root), "--rounds", str(args.rounds)]
        if args.smoke:
            argv6.append("--smoke")
        if args.manuscript_bundle is not None:
            argv6 += ["--manuscript-bundle", str(args.manuscript_bundle)]
        if args.seeds:
            argv6 += ["--seeds", args.seeds]
        if args.matching_pennies:
            argv6.append("--matching-pennies")
        main_milestone6(argv6)
        return

    if args.cmd == "milestone7-paper":
        from esl.experiments.milestone7_manuscript import main_milestone7

        argv7 = ["--out-root", str(args.out_root)]
        if args.smoke:
            argv7.append("--smoke")
        if args.manuscript_bundle is not None:
            argv7 += ["--manuscript-bundle", str(args.manuscript_bundle)]
        if args.frozen is not None:
            argv7 += ["--frozen", str(args.frozen)]
        if args.skip_m4_if_ready:
            argv7.append("--skip-m4-if-ready")
        if args.m5_resume_skip_complete:
            argv7.append("--m5-resume-skip-complete")
        main_milestone7(argv7)
        return

    if args.cmd == "milestone8-camera-ready":
        from esl.experiments.milestone8_camera_ready import main_milestone8

        main_milestone8(["--manuscript-bundle", str(args.manuscript_bundle)])
        return

    if args.cmd == "milestone9-submission":
        from esl.experiments.milestone9_submission import main_milestone9

        main_milestone9(
            [
                "--manuscript-bundle",
                str(args.manuscript_bundle),
                "--min-fig-bytes",
                str(args.min_fig_bytes),
            ]
        )
        return

    if args.cmd == "figure-diagnostics":
        from esl.experiments.figure_diagnostics import main_figure_diagnostics

        main_figure_diagnostics(
            [
                "--manuscript-bundle",
                str(args.manuscript_bundle),
                "--min-seeds",
                str(args.min_seeds),
                "--min-fig-bytes",
                str(args.min_fig_bytes),
            ]
        )
        return


if __name__ == "__main__":
    main()
