"""
Hard gates for full (non-smoke) paper runs: seeds, horizons, publication DPI.

Used by ``run_milestone7_paper`` when ``smoke=False``. Keeps M4–M6 aligned with
submission-quality settings without ad-hoc CLI drift.
"""

from __future__ import annotations

import os
from typing import Any


class PaperRunContractError(ValueError):
    pass


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise PaperRunContractError(msg)


def validate_frozen_for_full_paper(raw: dict[str, Any]) -> None:
    """
    Enforce:
    - ≥10 seeds on M4, M5, M6
    - M4: each of rounds_sparse, rounds_q, rounds_init, rounds_k in [5000, 10000]
    - M5: rounds in [2000, 5000]
    - M6: rounds in [1000, 3000]
    - ``ESL_PUBLICATION_DPI`` environment variable exactly ``300``
    """
    dpi = os.environ.get("ESL_PUBLICATION_DPI", "").strip()
    _need(dpi == "300", "Set ESL_PUBLICATION_DPI=300 before a full paper run (got %r)." % dpi)

    for key in ("milestone4", "milestone5", "milestone6"):
        _need(key in raw, f"Frozen JSON missing {key!r}")

    m4, m5, m6 = raw["milestone4"], raw["milestone5"], raw["milestone6"]

    for label, cfg in ("M4", m4), ("M5", m5), ("M6", m6):
        seeds = cfg.get("seeds")
        _need(isinstance(seeds, list), f"{label}: seeds must be a list")
        _need(len(seeds) >= 10, f"{label}: need at least 10 seeds, got {len(seeds)}")

    for rk in ("rounds_sparse", "rounds_q", "rounds_init", "rounds_k"):
        _need(rk in m4, f"M4: missing {rk!r}")
        v = int(m4[rk])
        _need(5000 <= v <= 10000, f"M4: {rk}={v} must be in [5000, 10000]")

    r5 = int(m5["rounds"])
    _need(2000 <= r5 <= 5000, f"M5: rounds={r5} must be in [2000, 5000]")

    r6 = int(m6["rounds"])
    _need(1000 <= r6 <= 3000, f"M6: rounds={r6} must be in [1000, 3000]")


def validate_frozen_json_file(path: Any) -> None:
    import json
    from pathlib import Path

    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    validate_frozen_for_full_paper(raw)
