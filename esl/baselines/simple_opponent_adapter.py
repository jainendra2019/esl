"""Thin ESL adapter for the ICML 2016 Simple Opponent Model source."""

from __future__ import annotations

from esl.baselines.external_common import (
    BaselineAvailability,
    BaselineSpec,
    ReducedMatrixGameAdapter,
    source_info,
)

SIMPLE_OPPONENT_SOURCE_URL = "https://github.com/hhexiy/opponent"


def simple_opponent_spec() -> BaselineSpec:
    source = source_info(
        name="opponent",
        url=SIMPLE_OPPONENT_SOURCE_URL,
        local_rel="third_party/opponent",
        license_name="MIT-style license file present in pinned source",
        notes="Official DRON/opponent-modeling source acquired; upstream is Lua/Torch and dataset-specific.",
    )
    return BaselineSpec(
        family="simple_opponent_model",
        method_name="Simple Opponent Model",
        adapter_path="esl.baselines.simple_opponent_adapter",
        source=source,
        deviations=(
            "Official code is Lua/Torch and targets quiz bowl / soccer experiments with external data.",
            "Smoke adapter uses a reduced conditional opponent-action model and soft best response.",
            "No official Lua code is executed in the smoke milestone.",
        ),
        availability=BaselineAvailability(
            status="PARTIAL",
            reason="Official source is pinned but cannot run directly in this Python matrix-game stack.",
            can_run_repeated_2x2=True,
        ),
    )


class SimpleOpponentAdapter(ReducedMatrixGameAdapter):
    def __init__(self) -> None:
        super().__init__(simple_opponent_spec())


def build_adapter() -> SimpleOpponentAdapter:
    return SimpleOpponentAdapter()
