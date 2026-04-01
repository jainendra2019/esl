"""Architecture: learning path must not depend on synthetic ground-truth modules."""

from pathlib import Path


def test_trainer_source_does_not_import_synthetic_population():
    root = Path(__file__).resolve().parents[1]
    text = (root / "esl" / "trainer.py").read_text(encoding="utf-8")
    banned = (
        "from esl.synthetic_population",
        "import esl.synthetic_population",
    )
    for b in banned:
        assert b not in text, f"trainer.py must not contain {b!r}"
