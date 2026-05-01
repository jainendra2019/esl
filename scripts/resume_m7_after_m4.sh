#!/usr/bin/env bash
# Resume Milestone 7 after a successful M4 tree: skip M4, reuse finished M5 cells, run M5+M6+M7 staging.
# Requires prior full M4 under OUT/milestone4/ (aggregate/milestone4_per_run_metrics.csv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-runs/m7_full}"
export ESL_PUBLICATION_DPI="${ESL_PUBLICATION_DPI:-300}"
export ESL_PROGRESS_HEARTBEAT_SEC="${ESL_PROGRESS_HEARTBEAT_SEC:-120}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "OUT=$OUT"
test -f "$OUT/milestone4/aggregate/milestone4_per_run_metrics.csv" || {
  echo "Missing M4 aggregate; run full M4 first or point OUT= to the directory that has it."
  exit 1
}

python3 -m esl.experiments milestone7-paper \
  --out-root "$OUT" \
  --manuscript-bundle manuscript_bundle \
  --skip-m4-if-ready \
  --m5-resume-skip-complete

echo "Done. Next (optional): milestone8-camera-ready, figure-diagnostics."
