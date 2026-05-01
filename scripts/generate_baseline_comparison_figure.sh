#!/usr/bin/env bash
# Regenerate docs/baselines/baseline_mce_comparison.png
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 scripts/generate_baseline_comparison_figure.py "$@"
