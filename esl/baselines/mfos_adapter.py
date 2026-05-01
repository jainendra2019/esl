"""Thin ESL adapter for M-FOS source onboarding."""

from __future__ import annotations

from esl.baselines.external_common import (
    BaselineAvailability,
    BaselineSpec,
    ReducedMatrixGameAdapter,
    source_info,
)

MFOS_SOURCE_URL = "https://github.com/luchris429/Model-Free-Opponent-Shaping"


def mfos_spec() -> BaselineSpec:
    source = source_info(
        name="Model-Free-Opponent-Shaping",
        url=MFOS_SOURCE_URL,
        local_rel="third_party/Model-Free-Opponent-Shaping",
        notes="Official PyTorch M-FOS source acquired; adapter uses a reduced matrix-game policy-shaping smoke.",
    )
    return BaselineSpec(
        family="mfos",
        method_name="M-FOS",
        adapter_path="esl.baselines.mfos_adapter",
        source=source,
        deviations=(
            "Official M-FOS trains a meta-policy via src/main_mfos_ppo.py; smoke adapter uses a "
            "reduced cooperation-shaping policy for repeated 2-action matrix games.",
            "No official environment code is modified; reduced implementation is isolated in ESL adapter code.",
        ),
        availability=BaselineAvailability(
            status="PARTIAL",
            reason="Official source is pinned, but full M-FOS meta-game training is not yet wrapped.",
            can_run_repeated_2x2=True,
        ),
    )


class MFOSAdapter(ReducedMatrixGameAdapter):
    def __init__(self) -> None:
        super().__init__(mfos_spec())


def build_adapter() -> MFOSAdapter:
    return MFOSAdapter()
