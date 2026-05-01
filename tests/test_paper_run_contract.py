"""Paper run contract gates."""

from __future__ import annotations

import pytest

from esl.experiments.paper_run_contract import PaperRunContractError, validate_frozen_for_full_paper


def test_contract_passes_on_default_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESL_PUBLICATION_DPI", "300")
    raw = {
        "milestone4": {
            "seeds": list(range(10)),
            "rounds_sparse": 5000,
            "rounds_q": 5000,
            "rounds_init": 5000,
            "rounds_k": 5000,
        },
        "milestone5": {"seeds": list(range(10)), "rounds": 4000},
        "milestone6": {"seeds": list(range(10)), "rounds": 2000},
    }
    validate_frozen_for_full_paper(raw)


def test_contract_fails_wrong_dpi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESL_PUBLICATION_DPI", "150")
    raw = {
        "milestone4": {
            "seeds": list(range(10)),
            "rounds_sparse": 5000,
            "rounds_q": 5000,
            "rounds_init": 5000,
            "rounds_k": 5000,
        },
        "milestone5": {"seeds": list(range(10)), "rounds": 2000},
        "milestone6": {"seeds": list(range(10)), "rounds": 2000},
    }
    with pytest.raises(PaperRunContractError):
        validate_frozen_for_full_paper(raw)


def test_contract_fails_m6_rounds_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESL_PUBLICATION_DPI", "300")
    raw = {
        "milestone4": {
            "seeds": list(range(10)),
            "rounds_sparse": 5000,
            "rounds_q": 5000,
            "rounds_init": 5000,
            "rounds_k": 5000,
        },
        "milestone5": {"seeds": list(range(10)), "rounds": 2000},
        "milestone6": {"seeds": list(range(10)), "rounds": 500},
    }
    with pytest.raises(PaperRunContractError):
        validate_frozen_for_full_paper(raw)
