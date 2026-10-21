#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

EXP_NAME="mnist-4vs9-CNN-valpha-main"
NS=(75 250 1000 4000)
EPOCH=200
N_SEEDS=5
N_SELECTOR_SEEDS=30

require_run_grid "${EXP_NAME}" "${EPOCH}" "${N_SEEDS}" "${N_SELECTOR_SEEDS}" "${NS[@]}"
parse_samplewise_results "${EXP_NAME}" "${EPOCH}" 2 "${N_SEEDS}" "${N_SELECTOR_SEEDS}" "${NS[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}" MPLBACKEND=Agg \
  "${PYTHON_BIN}" -B -m scripts.plot_valpha_ratio_to_error \
  --input "${RESULTS_DIR}/${EXP_NAME}/results.pkl" \
  --output "${FIGURES_DIR}/figure1_mnist_valpha_ratio.pdf" \
  --epoch "${EPOCH}" \
  --ns "${NS[@]}" \
  --linear_ce \
  --include_t11 \
  --include_t13
