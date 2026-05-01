"""Publication figure settings (Milestone 3: NeurIPS-ready DPI)."""

from __future__ import annotations

import os

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
