#!/usr/bin/env bash
# Full paper run: Milestone 7 (M4→M6) without smoke + figure diagnostics.
# Contract enforced in code: ≥10 seeds, M4 rounds 5k–10k, M5 2k–5k, M6 1k–3k, ESL_PUBLICATION_DPI=300.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export ESL_PUBLICATION_DPI=300
# Heartbeat interval (seconds) for full runs; set to 0 to disable timer thread only.
export ESL_PROGRESS_HEARTBEAT_SEC="${ESL_PROGRESS_HEARTBEAT_SEC:-120}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "== Step 0: contract + bundle =="
test -d manuscript_bundle || { echo "missing manuscript_bundle/"; exit 1; }
python3 -c "
from pathlib import Path
from esl.experiments.paper_run_contract import validate_frozen_json_file
validate_frozen_json_file(Path('esl/experiments/configs/milestone7_frozen.json'))
print('paper_run_contract: OK')
"
echo "ESL_PUBLICATION_DPI=$ESL_PUBLICATION_DPI"
echo "ESL_PROGRESS_HEARTBEAT_SEC=$ESL_PROGRESS_HEARTBEAT_SEC (terminal progress pings)"

echo "== Milestone 7 (full M4→M6, stage manuscript) =="
python3 -m esl.experiments milestone7-paper \
  --out-root runs/m7_full \
  --manuscript-bundle manuscript_bundle

echo "== Figure diagnostics =="
python3 -m esl.experiments figure-diagnostics \
  --manuscript-bundle manuscript_bundle \
  --min-seeds 10 \
  --min-fig-bytes 8000

echo "Done. See manuscript_bundle/reports/figure_diagnostics.md"
echo "STOP before M8/M9 until you confirm figure decisions."
