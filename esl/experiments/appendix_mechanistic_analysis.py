"""CLI wrapper for the mechanistic appendix analysis pipeline."""

from __future__ import annotations

from esl_analysis.pipeline import main, run_analysis_pipeline

__all__ = ["main", "run_analysis_pipeline"]


if __name__ == "__main__":
    main()
