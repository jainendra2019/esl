"""Registry-driven ESL smoke runs + canonical manifest.json emission."""

from __future__ import annotations

from pathlib import Path

from esl.experiment_registry import (
    build_esl_manifest_dict,
    esl_config_for_block,
    resolve_experiment_block,
)
from esl.experiments.canonical_io import write_manifest_json
from esl.trainer import run_esl


def run_registry_esl(
    experiment_id: str,
    run_dir: Path,
    *,
    seed: int = 0,
    smoke: bool = True,
    variant: str | None = None,
) -> Path:
    """
    Execute an ESL registry block and write ``manifest.json`` (canonical, PRD §10).

    Does not write ``run_manifest.json`` (optional NeurIPS sidecar).
    """
    block = resolve_experiment_block(experiment_id)
    if block.run_kind != "esl":
        raise ValueError(f"{experiment_id!r} is not an ESL block")
    cfg, variant_slug = esl_config_for_block(
        experiment_id, seed=seed, smoke=smoke, variant=variant
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_esl(cfg, run_dir=run_dir)
    preset = block.neurips_preset
    manifest = build_esl_manifest_dict(
        experiment_id=experiment_id,
        seed=seed,
        smoke=smoke,
        manifest_variant=variant_slug,
        neurips_preset=preset,
    )
    write_manifest_json(run_dir / "manifest.json", manifest)
    return run_dir
