"""python -m esl — short help for the camera-ready release."""

from __future__ import annotations


def main() -> None:
    print(
        "Epistemic Social Learning (ESL) — NeurIPS 2026 camera-ready code.\n"
        "\n"
        "Quick verification:\n"
        "  python -m pytest -q\n"
        "  python scripts/verify_release.py\n"
        "\n"
        "Regenerate published tables and figures from frozen evidence:\n"
        "  python scripts/reproduce_tables.py\n"
        "  python scripts/reproduce_figures.py\n"
        "\n"
        "Run experiments (expensive):\n"
        "  python -m esl.res --help\n"
        "  python -m esl.res --num-actions 3 --full --out-dir outputs/flagship\n"
        "\n"
        "See README.md for the full reproduction map."
    )


if __name__ == "__main__":
    main()
