"""NeurIPS-style experiment presets, runner, and aggregate reporting."""

from esl.experiments.presets import (
    recovery_failure_case_cfg,
    recovery_fixed_prototype_baseline_cfg,
    recovery_flagship_cfg,
    recovery_init_noise_sweep_cfg,
    recovery_lr_sweep_cfg,
    recovery_Q_sweep_cfg,
    recovery_short_horizon_cfg,
    recovery_sparse_obs_cfg,
)

__all__ = [
    "recovery_flagship_cfg",
    "recovery_sparse_obs_cfg",
    "recovery_short_horizon_cfg",
    "recovery_lr_sweep_cfg",
    "recovery_init_noise_sweep_cfg",
    "recovery_Q_sweep_cfg",
    "recovery_failure_case_cfg",
    "recovery_fixed_prototype_baseline_cfg",
]
