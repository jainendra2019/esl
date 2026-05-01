from __future__ import annotations

from typing import Any


def _gap(row: dict[str, Any]) -> float:
    if "psr_gap" in row and row["psr_gap"] != "":
        expected = float(row["oracle_payoff"]) - float(row["focal_payoff"])
        got = float(row["psr_gap"])
        if abs(expected - got) > 1e-8:
            raise ValueError("psr_gap does not match oracle_payoff - focal_payoff")
        return got
    return float(row["oracle_payoff"]) - float(row["focal_payoff"])


def compute_psr_per_switch(rows: list[dict[str, Any]], *, switch_time: int, horizon: int = 100) -> float:
    vals = [
        _gap(row)
        for row in rows
        if switch_time <= int(row["round"]) <= switch_time + horizon
    ]
    return float(sum(vals))
