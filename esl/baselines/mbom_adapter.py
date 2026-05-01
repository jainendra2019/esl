"""Thin ESL adapter for MBOM source onboarding."""

from __future__ import annotations

from esl.baselines.external_common import (
    BaselineAvailability,
    BaselineSpec,
    ReducedMatrixGameAdapter,
    source_info,
)

MBOM_SOURCE_URL = "https://github.com/PKU-RL/MBOM"


def mbom_spec() -> BaselineSpec:
    source = source_info(
        name="MBOM",
        url=MBOM_SOURCE_URL,
        local_rel="third_party/MBOM",
        notes="Official MBOM source acquired; upstream targets older Gym/gfootball environments.",
    )
    return BaselineSpec(
        family="mbom",
        method_name="MBOM",
        adapter_path="esl.baselines.mbom_adapter",
        source=source,
        deviations=(
            "Official MBOM depends on Python 3.6, gym 0.17.2, gfootball, and a full environment model stack.",
            "Smoke adapter uses a reduced model-based opponent-frequency estimator with soft best response.",
            "The reduced adapter records payoff only; full recursive imagination is not claimed.",
        ),
        availability=BaselineAvailability(
            status="PARTIAL",
            reason="Official code is pinned but incompatible with the current lightweight matrix-game runtime.",
            can_run_repeated_2x2=True,
        ),
    )


class MBOMAdapter(ReducedMatrixGameAdapter):
    def __init__(self) -> None:
        super().__init__(mbom_spec())


def build_adapter() -> MBOMAdapter:
    return MBOMAdapter()
