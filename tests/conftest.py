"""
Pytest configuration: markers and shared fixtures.

Markers:
  verification — algorithm / math correctness (§1 of TEST_PLAN.md)
  validation   — ESL behavior vs PRD intent (§2)
  edge           — robustness (§3)
  slow           — long runs; omitted from default `pytest` (pass `--runslow` to include)

Default `pytest` drops `@pytest.mark.slow` tests so the dev loop stays fast.
`pytest -m slow` still runs them: the hook skips filtering when the `-m` expression positively includes `slow` (see `_markexpr_includes_slow_positive`; `not slow` is treated as exclusion).
Full suite: `pytest --runslow`
"""

from __future__ import annotations

import re

import pytest


def _markexpr_includes_slow_positive(markexpr: str) -> bool:
    """True if MARKEXPR requests slow tests (handles `not slow` as exclusion)."""
    if not (markexpr or "").strip():
        return False
    simplified = re.sub(r"\bnot\s+slow\b", "__neg_slow__", markexpr.lower())
    return bool(re.search(r"\bslow\b", simplified))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="include @pytest.mark.slow tests in the default run",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "verification: unit and integration correctness")
    config.addinivalue_line("markers", "validation: end-to-end ESL behavior")
    config.addinivalue_line("markers", "edge: edge-case robustness")
    config.addinivalue_line(
        "markers", "slow: long runs excluded unless --runslow or -m slow only"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("runslow", default=False):
        return
    if not items:
        return
    # `-m slow` is applied after this hook; do not drop slow tests when the expr asks for them.
    markexpr = config.getoption("markexpr") or ""
    if _markexpr_includes_slow_positive(markexpr):
        return
    if all(item.get_closest_marker("slow") for item in items):
        return
    items[:] = [item for item in items if not item.get_closest_marker("slow")]
