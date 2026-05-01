from __future__ import annotations

import json
from typing import Any

import numpy as np

from esl_analysis.metrics.entropy import belief_entropy


def _belief(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.loads(row["belief"]), dtype=np.float64)


def _switch_round(row: dict[str, Any]) -> int | None:
    value = row.get("switch_round", "")
    if value == "":
        return None
    return int(value)


def posterior_accuracy(belief: np.ndarray, true_type: int) -> int:
    return int(int(np.argmax(belief)) == int(true_type))


def extract_entropy_trajectory(rows: list[dict[str, Any]], *, switch_time: int, t_window: int = 50) -> np.ndarray:
    vals: list[float] = []
    for tau in range(t_window):
        matches = [
            row
            for row in rows
            if row.get("belief", "") != "" and _switch_round(row) == switch_time and int(row["tau"]) == tau
        ]
        if not matches:
            return np.array([], dtype=np.float64)
        vals.append(float(np.mean([belief_entropy(_belief(row)) for row in matches])))
    return np.asarray(vals, dtype=np.float64)


def extract_accuracy_trajectory(rows: list[dict[str, Any]], *, switch_time: int, t_window: int = 50) -> np.ndarray:
    vals: list[float] = []
    for tau in range(t_window):
        matches = [
            row
            for row in rows
            if row.get("belief", "") != "" and _switch_round(row) == switch_time and int(row["tau"]) == tau
        ]
        if not matches:
            return np.array([], dtype=np.float64)
        vals.append(float(np.mean([posterior_accuracy(_belief(row), int(row["true_type"])) for row in matches])))
    return np.asarray(vals, dtype=np.float64)


def extract_aligned_trajectories(rows: list[dict[str, Any]], *, t_window: int = 50) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    switch_times = sorted(
        {
            int(row["switch_round"])
            for row in rows
            if row.get("belief", "") != "" and row.get("switch_round", "") != "" and int(row.get("switch_id", -1)) >= 0
        }
    )
    entropy: list[np.ndarray] = []
    accuracy: list[np.ndarray] = []
    excluded = 0
    for switch_time in switch_times:
        e = extract_entropy_trajectory(rows, switch_time=switch_time, t_window=t_window)
        a = extract_accuracy_trajectory(rows, switch_time=switch_time, t_window=t_window)
        if e.shape[0] != t_window or a.shape[0] != t_window:
            excluded += 1
            continue
        entropy.append(e)
        accuracy.append(a)
    return entropy, accuracy, excluded
