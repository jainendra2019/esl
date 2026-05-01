from __future__ import annotations

import time

import pytest

from esl.experiments.progress_echo import progress_heartbeat


def test_heartbeat_runs_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESL_PROGRESS_HEARTBEAT_SEC", "0")
    t0 = time.monotonic()
    with progress_heartbeat("test", enabled=True):
        time.sleep(0.05)
    assert time.monotonic() - t0 < 2.0


def test_heartbeat_disabled_no_delay() -> None:
    with progress_heartbeat("off", enabled=False):
        pass
