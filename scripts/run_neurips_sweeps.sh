#!/usr/bin/env bash
# Full NeurIPS sweep grid (long). Requires active venv with esl installed.
# Usage: ./scripts/run_neurips_sweeps.sh
# Override: NEURIPS_SEED=43 NEURIPS_OUT=runs/neurips ./scripts/run_neurips_sweeps.sh
# Quick structural demo (tiny rounds, not publication numbers):
#   NEURIPS_SMOKE=1 ./scripts/run_neurips_sweeps.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${NEURIPS_SEED:-42}"
OUT="${NEURIPS_OUT:-runs/neurips}"
SMOKE=()
if [[ "${NEURIPS_SMOKE:-0}" == "1" ]]; then
  SMOKE=(--smoke)
  echo "NEURIPS_SMOKE=1: using --smoke (few rounds per run; for full paper runs omit NEURIPS_SMOKE)"
fi
RUN=(python3 -m esl.experiments run --seed "$SEED" --out-root "$OUT" "${SMOKE[@]}")

echo "==> Sparse p_obs"
for p in 1.0 0.5 0.2; do
  "${RUN[@]}" --preset recovery_sparse_obs --variant "$p"
done

echo "==> Short horizons (fixed L_t=10)"
for b in 500 1000 2000; do
  "${RUN[@]}" --preset recovery_short_horizon --variant "$b"
done

echo "==> LR sweep"
for lr in 8 12 18 22; do
  "${RUN[@]}" --preset recovery_lr_sweep --variant "$lr"
done

echo "==> init_noise sweep"
for n in 0.0 0.01 0.05 0.12; do
  "${RUN[@]}" --preset recovery_init_noise_sweep --variant "$n"
done

echo "==> Q sweep"
for q in 10 15 20; do
  "${RUN[@]}" --preset recovery_Q_sweep --variant "$q"
done

echo "==> Flagship long + failure + fixed-prototype baseline"
"${RUN[@]}" --preset recovery_flagship
"${RUN[@]}" --preset recovery_failure_case
"${RUN[@]}" --preset recovery_fixed_prototype_baseline

echo "==> Aggregate CSV"
python3 -m esl.experiments aggregate "$OUT" -o "$OUT/_aggregates/summary_all_runs.csv"
echo "Done. Figures: python3 -m esl.plot_neurips robustness $OUT/_aggregates/summary_all_runs.csv -o $OUT/_figures/robustness.png"
