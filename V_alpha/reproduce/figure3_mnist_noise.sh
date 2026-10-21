#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

NS=(75 250 1000 4000)
EPOCH=200
N_SEEDS=5
N_SELECTOR_SEEDS=30
EXPS=(
  "mnist-4vs9-CNN-noise-0.05-valpha-main"
  "mnist-4vs9-CNN-noise-0.1-valpha-main"
  "mnist-4vs9-CNN-noise-0.15-valpha-main"
)

for exp_name in "${EXPS[@]}"; do
  require_run_grid "${exp_name}" "${EPOCH}" "${N_SEEDS}" "${N_SELECTOR_SEEDS}" "${NS[@]}"
  parse_samplewise_results "${exp_name}" "${EPOCH}" 2 "${N_SEEDS}" "${N_SELECTOR_SEEDS}" "${NS[@]}"
done

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}" MPLBACKEND=Agg \
  "${PYTHON_BIN}" -B -m scripts.plot_valpha_ratio_grid \
  --inputs \
    "${RESULTS_DIR}/${EXPS[0]}/results.pkl" \
    "${RESULTS_DIR}/${EXPS[1]}/results.pkl" \
    "${RESULTS_DIR}/${EXPS[2]}/results.pkl" \
  --labels 'delta=0.05' 'delta=0.10' 'delta=0.15' \
  --output "${FIGURES_DIR}/figure3_mnist_noise_valpha_ratio.pdf" \
  --epoch "${EPOCH}" \
  --ns "${NS[@]}"
