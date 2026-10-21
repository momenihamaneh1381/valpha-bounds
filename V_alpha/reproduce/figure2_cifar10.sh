#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

EXP_NAME="cifar10-pretrained-resnet50-valpha-main"
NS=(1000 5000 20000)
EPOCH=40
N_SEEDS=2
N_SELECTOR_SEEDS=40
REQUIRE_DETERMINISTIC_SUPERSAMPLE_EVAL=1

require_run_grid "${EXP_NAME}" "${EPOCH}" "${N_SEEDS}" "${N_SELECTOR_SEEDS}" "${NS[@]}"
if [[ "${SKIP_PARSE}" != "1" ]]; then
"${PYTHON_BIN}" -B -m scripts.check_deterministic_supersamples \
  --results_dir "${RESULTS_DIR}" \
  --exp_name "${EXP_NAME}" \
  --ns "${NS[@]}" \
  --n_seeds "${N_SEEDS}" \
  --n_selector_seeds "${N_SELECTOR_SEEDS}"
fi
parse_samplewise_results "${EXP_NAME}" "${EPOCH}" 10 "${N_SEEDS}" "${N_SELECTOR_SEEDS}" "${NS[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}" MPLBACKEND=Agg \
  "${PYTHON_BIN}" -B -m scripts.plot_valpha_ratio_to_error \
  --input "${RESULTS_DIR}/${EXP_NAME}/results.pkl" \
  --output "${FIGURES_DIR}/figure2_cifar10_valpha_ratio.pdf" \
  --epoch "${EPOCH}" \
  --ns "${NS[@]}" \
  --linear_ce \
  --include_t11 \
  --include_t13
