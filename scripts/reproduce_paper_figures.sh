#!/usr/bin/env bash
# Reproduce core outputs locally (runs/ is gitignored).
#
# Default (~1–3 min): fast tests + short experiment + figures.
# Full paper-style bundle (~3–8 min): set ESL_FULL_REPRO=1
#
# Usage:
#   ./scripts/reproduce_paper_figures.sh
#   ESL_FULL_REPRO=1 ./scripts/reproduce_paper_figures.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/esl-mpl-cache}"
mkdir -p "$MPLCONFIGDIR"

echo "==> pytest (fast suite, excludes @slow)"
python3 -m pytest -q

echo "==> Quick smoke: 50 rounds, full obs, three conditions + PNGs → runs/quickstart_smoke/"
python3 -m esl.experiment_two_type_separation \
  --rounds 50 \
  --m 5 \
  --seed 42 \
  --lr-scale 18 \
  --obs-prob 1.0 \
  --out runs/quickstart_smoke

echo ""
echo "Quickstart artifacts (check these exist):"
echo "  runs/quickstart_smoke/experiment_manifest.json"
echo "  runs/quickstart_smoke/figure_prototype_separation.png"
echo "  runs/quickstart_smoke/main/summary_metrics.json"

if [[ "${ESL_FULL_REPRO:-0}" == "1" ]]; then
  echo ""
  echo "==> Full repro: 250 rounds full obs → runs/paper_ref_main/"
  python3 -m esl.experiment_two_type_separation \
    --rounds 250 \
    --m 5 \
    --seed 42 \
    --lr-scale 18 \
    --obs-prob 1.0 \
    --out runs/paper_ref_main

  echo "==> Mechanism figure: p_obs=0.5, 400 rounds (metrics + mechanism PNG)"
  python3 -m esl.experiment_two_type_separation \
    --rounds 400 \
    --m 5 \
    --seed 42 \
    --lr-scale 18 \
    --obs-prob 0.5 \
    --no-plots \
    --out runs/paper_ref_mechanism

  python3 -m esl.plot_esl_mechanism --main-dir runs/paper_ref_mechanism/main

  echo ""
  echo "Reference artifacts:"
  echo "  runs/paper_ref_main/figure_*.png"
  echo "  runs/paper_ref_mechanism/main/figure_mechanism.png"
else
  echo ""
  echo "Tip: for a longer paper-style bundle (250r main + mechanism @ p_obs=0.5), run:"
  echo "  ESL_FULL_REPRO=1 ./scripts/reproduce_paper_figures.sh"
fi

echo ""
echo "Done."
