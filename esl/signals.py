"""Observable signals from actions (MVP: signal equals observed opponent action)."""

from __future__ import annotations


def action_to_signal(action: int) -> int:
    return int(action)
