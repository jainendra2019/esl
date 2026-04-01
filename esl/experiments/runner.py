"""Execute preset configs under runs/neurips/... with run_manifest.json sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from esl.config import ESLConfig
from esl.experiments import presets as neurips_presets
from esl.experiments.aggregate import write_aggregate_csv
from esl.experiments.manifest import write_run_manifest
from esl.trainer import run_esl


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
