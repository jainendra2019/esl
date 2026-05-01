"""Optional terminal heartbeat for long-running experiment batches."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


def heartbeat_interval_seconds() -> float:
    raw = os.environ.get("ESL_PROGRESS_HEARTBEAT_SEC", "60").strip()
    if raw == "0":
        return 0.0
    try:
        v = float(raw)
    except ValueError:
        return 60.0
    return max(5.0, v)


@contextmanager
def progress_heartbeat(label: str, *, enabled: bool) -> Iterator[None]:
    """
    While the context is active, print a line every ``ESL_PROGRESS_HEARTBEAT_SEC`` seconds
    (default 60) so long runs show signs of life. Set ``ESL_PROGRESS_HEARTBEAT_SEC=0`` to
    disable the timer thread (label still prints start/stop if enabled).
    """
    if not enabled:
        yield
        return

    interval = heartbeat_interval_seconds()
    stop = threading.Event()
    start = time.monotonic()

    def _run() -> None:
        n = 0
        while not stop.wait(interval):
            n += 1
            elapsed = time.monotonic() - start
            print(
                f"[progress] {label} — still running (heartbeat #{n}, {elapsed:.0f}s elapsed)",
                flush=True,
            )

    print(f"[progress] {label} — starting", flush=True)
    t: threading.Thread | None = None
    if interval > 0:
        t = threading.Thread(target=_run, name="progress-heartbeat", daemon=True)
        t.start()
    try:
        yield
    finally:
        stop.set()
        if t is not None:
            t.join(timeout=interval + 2.0)
        print(
            f"[progress] {label} — finished ({time.monotonic() - start:.0f}s total)",
            flush=True,
        )


def log_step(msg: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[progress] {msg}", flush=True)
