"""Publication figure settings (NeurIPS-ready DPI)."""

from __future__ import annotations

import os
import re
from typing import Any

_DEFAULT_DPI = 150
_PUBLICATION_DPI = 300


def publication_figure_dpi() -> int:
    """
    Return matplotlib ``dpi`` for saved figures.

    Set ``ESL_PUBLICATION_DPI=300`` (or ``1`` / ``true`` to use 300) for print-quality exports;
    default 150 keeps CI/tests fast.
    """
    v = os.environ.get("ESL_PUBLICATION_DPI", "").strip().lower()
    if v in ("1", "true", "yes", "300"):
        return _PUBLICATION_DPI
    if v.isdigit():
        return max(72, min(int(v), 600))
    return _DEFAULT_DPI


def ylabel_lower_better(name: str) -> str:
    """Plain metric ylabel (no arrow glyph).

    Unicode ↓ on rotated ylabels either becomes → or floats beside the top letter;
    detached annotations floated near panel letters. Captions state lower-is-better.
    """
    return name.rstrip()


def xlabel_lower_better(name: str) -> str:
    """X-axis label; optional lower-better note without a fragile arrow glyph."""
    return f"{name.rstrip()} (lower better)"


def apply_ylabel_lower_better(ax: Any, name: str, *, fontsize: float = 10) -> None:
    """Set a standard vertical ylabel with the metric name only."""
    ax.set_ylabel(ylabel_lower_better(name), fontsize=fontsize)


def fix_rotated_ylabel_arrows(label: str) -> str:
    """Strip legacy arrow glyphs from ylabel strings; keep the metric name."""
    cleaned = label
    for token in (
        " ↓",
        "↑",
        "↓",
        " ←",
        "←",
        " →",
        "→",
        r" $\downarrow$",
        r"$\downarrow$",
        r"\!\downarrow",
        r"\downarrow",
        "\n",
        " (lower better)",
    ):
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"\$\\mathrm\{([^}]*)\}\$", r"\1", cleaned)
    cleaned = cleaned.replace(r"\ ", " ").replace("$", "").strip()
    return ylabel_lower_better(cleaned) if cleaned else cleaned
