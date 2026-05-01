"""
Experiment registry: named blocks, canonical manifest payloads, and ESL smoke configs.

PRD §9 (experiment_registry), §10–11 (manifests / metrics identifiers), §13A (milestones).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from esl.config import ESLConfig

RunKind = Literal["esl", "baseline"]

MANIFEST_SCHEMA_VERSION: Final[int] = 1
SUMMARY_METRICS_SCHEMA_VERSION: Final[int] = 1
OUTPUT_CONVENTION_ID: Final[str] = "esl_canonical_v1"


@dataclass(frozen=True)
class ExperimentBlock:
    """A named experiment block (registry row)."""

    id: str
    run_kind: RunKind
    title: str
    # For ESL blocks driven by NeurIPS preset runner (None for inline smoke / baseline stubs).
    neurips_preset: str | None = None
    default_variant: str = ""


def milestone_smoke_lock_cfg(*, seed: int = 0) -> ESLConfig:
    """Tiny ESL run for Milestone 1 CI: few rounds, one pair per round, small N."""
    c = ESLConfig(
        seed=seed,
        mode="recovery",
        num_agents=4,
        num_prototypes=2,
        num_actions=2,
        num_rounds=3,
        interaction_pairs_min=1,
        interaction_pairs_max=1,
        prototype_update_every=1,
        log_beliefs_tensor=False,
        symmetric_init=False,
        init_noise=0.05,
        prototype_lr_scale=1.0,
    )
    c.validate()
    return c


_EXPERIMENT_BLOCKS: tuple[ExperimentBlock, ...] = (
    ExperimentBlock(
        "milestone.smoke_lock",
        "esl",
        "Milestone 1: minimal ESL smoke for schema and bundle tests",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "esl.neurips.recovery_flagship",
        "esl",
        "NeurIPS recovery flagship geometry",
        neurips_preset="recovery_flagship",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_fixed_prototype_baseline",
        "esl",
        "Freeze-prototype baseline (same geometry)",
        neurips_preset="recovery_fixed_prototype_baseline",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_failure_case",
        "esl",
        "Weak-init / failure stress geometry",
        neurips_preset="recovery_failure_case",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_sparse_obs",
        "esl",
        "Flagship with sparse observability (default p_obs=1.0 variant)",
        neurips_preset="recovery_sparse_obs",
        default_variant="1.0",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_short_horizon",
        "esl",
        "Short interaction budget (default 500 interactions)",
        neurips_preset="recovery_short_horizon",
        default_variant="500",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_lr_sweep",
        "esl",
        "LR ablation (default lr=12)",
        neurips_preset="recovery_lr_sweep",
        default_variant="12",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_init_noise_sweep",
        "esl",
        "Init-noise ablation (default 0.05)",
        neurips_preset="recovery_init_noise_sweep",
        default_variant="0.05",
    ),
    ExperimentBlock(
        "esl.neurips.recovery_Q_sweep",
        "esl",
        "Prototype Q ablation (default Q=15)",
        neurips_preset="recovery_Q_sweep",
        default_variant="15",
    ),
    ExperimentBlock(
        "baseline.offline.protocol_stub",
        "baseline",
        "Schema-only stub for offline baseline manifests (no train step in M1)",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.ppo.recovery_smoke",
        "baseline",
        "Milestone 2A: official PPO-PyTorch tiny recovery-style PD smoke (thin adapter)",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.ppo.performance_smoke",
        "baseline",
        "Main performance smoke: independent PPO policy baseline",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.mfos.performance_smoke",
        "baseline",
        "Main performance smoke: M-FOS source onboarding baseline",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.mbom.performance_smoke",
        "baseline",
        "Main performance smoke: MBOM source onboarding baseline",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.simple_opponent.performance_smoke",
        "baseline",
        "Main performance smoke: Simple Opponent Model source onboarding baseline",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.cluster.kmeans_recovery_smoke",
        "baseline",
        "Milestone 2B: K-means on conditional-cooperate features (offline static)",
        neurips_preset=None,
    ),
    ExperimentBlock(
        "baseline.cluster.fcm_recovery_smoke",
        "baseline",
        "Milestone 2B: FCM on conditional-cooperate features (offline static)",
        neurips_preset=None,
    ),
)

_ID_TO_BLOCK: dict[str, ExperimentBlock] = {b.id: b for b in _EXPERIMENT_BLOCKS}


def list_experiment_block_ids() -> list[str]:
    return sorted(_ID_TO_BLOCK.keys())


def resolve_experiment_block(block_id: str) -> ExperimentBlock:
    if block_id not in _ID_TO_BLOCK:
        raise KeyError(f"unknown experiment block id: {block_id!r}")
    return _ID_TO_BLOCK[block_id]


def esl_config_for_block(
    block_id: str,
    *,
    seed: int,
    smoke: bool = True,
    variant: str | None = None,
) -> tuple[ESLConfig, str]:
    """
    Return (cfg, manifest_variant_slug) for an ESL registry block.

    Raises:
        KeyError: unknown id
        ValueError: block is not an ESL runnable (e.g. baseline stub)
    """
    block = resolve_experiment_block(block_id)
    if block.run_kind != "esl":
        raise ValueError(f"block {block_id!r} is not an ESL run")
    if block.id == "milestone.smoke_lock":
        return milestone_smoke_lock_cfg(seed=seed), "inline_smoke"
    if block.neurips_preset is None:
        raise ValueError(f"block {block_id!r} has no neurips_preset")
    from esl.experiments.runner import esl_config_for_preset

    v = variant if variant is not None else block.default_variant
    return esl_config_for_preset(block.neurips_preset, seed=seed, smoke=smoke, variant=v)


def build_esl_manifest_dict(
    *,
    experiment_id: str,
    seed: int,
    smoke: bool,
    manifest_variant: str,
    neurips_preset: str | None,
    run_dir_relative: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical manifest.json body for an ESL run (PRD §10)."""
    body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": "esl",
        "experiment_id": experiment_id,
        "seed": int(seed),
        "smoke": bool(smoke),
        "output_convention": OUTPUT_CONVENTION_ID,
        "esl": {
            "neurips_preset": neurips_preset,
            "variant": manifest_variant,
        },
        "expected_outputs": {
            "required_relative": [
                "config.json",
                "manifest.json",
                "summary_metrics.json",
                "metrics_trajectory.csv",
            ],
            "optional_relative": [
                "prototype_trajectory.csv",
                "belief_trajectory.csv",
                "reward_trajectory.csv",
                "run_manifest.json",
                "interaction_observations.csv",
                "observation_manifest.json",
            ],
        },
    }
    if run_dir_relative is not None:
        body["run_dir"] = run_dir_relative
    if extra:
        body["extras"] = extra
    return body


def build_baseline_manifest_dict(
    *,
    experiment_id: str,
    seed: int,
    esl_run_dir: str,
    adapter: str,
    source_protocol: str = "docs/baselines/BASELINE_PROTOCOL.md",
    deviations: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical manifest.json for an offline / external baseline run."""
    body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": "baseline",
        "experiment_id": experiment_id,
        "seed": int(seed),
        "output_convention": OUTPUT_CONVENTION_ID,
        "baseline": {
            "adapter": adapter,
            "esl_run_dir": esl_run_dir,
            "source_protocol": source_protocol,
            "deviations": list(deviations or []),
            "provenance": provenance or {},
        },
        "expected_outputs": {
            "required_relative": [
                "manifest.json",
                "summary_metrics.json",
            ],
            "optional_relative": [
                "metrics_trajectory.csv",
                "predictions.json",
                "config.json",
                "provenance.json",
            ],
        },
    }
    if extra:
        body["extras"] = extra
    return body
