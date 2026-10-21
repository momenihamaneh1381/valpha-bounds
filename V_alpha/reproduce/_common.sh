#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${REPRO_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PARSE_DEVICE="${PARSE_DEVICE:-cpu}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
FIGURES_DIR="${FIGURES_DIR:-${REPO_ROOT}/figures}"
SKIP_PARSE="${SKIP_PARSE:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python or create .venv as documented." >&2
  exit 2
fi
mkdir -p "${FIGURES_DIR}"

require_run_grid() {
  local exp_name="$1"
  local epoch="$2"
  local n_seeds="$3"
  local n_selector_seeds="$4"
  shift 4
  local ns=("$@")
  if [[ "${SKIP_PARSE}" == "1" ]]; then
    if [[ ! -f "${RESULTS_DIR}/${exp_name}/results.pkl" ]]; then
      echo "Missing aggregate: ${RESULTS_DIR}/${exp_name}/results.pkl" >&2
      exit 3
    fi
    return
  fi
  local missing=0
  local n seed selector run_dir
  for n in "${ns[@]}"; do
    for ((seed=0; seed<n_seeds; seed++)); do
      for ((selector=0; selector<n_selector_seeds; selector++)); do
        run_dir="${RESULTS_DIR}/${exp_name}/n=${n},seed=${seed},S_seed=${selector}"
        if [[ ! -f "${run_dir}/saved_data.pkl" ]]; then
          echo "Missing ${run_dir}/saved_data.pkl" >&2
          missing=$((missing + 1))
          continue
        fi
        if [[ ! -f "${run_dir}/checkpoints/epoch$((epoch - 1)).mdl" ]]; then
          echo "Missing ${run_dir}/checkpoints/epoch$((epoch - 1)).mdl" >&2
          missing=$((missing + 1))
        fi
      done
    done
  done
  if (( missing > 0 )); then
    echo "Preflight failed: ${missing} required artifact(s) are missing." >&2
    echo "Training is not rerun automatically. Restore the saved run directories first." >&2
    exit 3
  fi
}

parse_samplewise_results() {
  local exp_name="$1"
  local epoch="$2"
  local num_classes="$3"
  local n_seeds="$4"
  local n_selector_seeds="$5"
  shift 5
  local ns=("$@")

  if [[ "${SKIP_PARSE}" == "1" ]]; then
    echo "SKIP_PARSE=1: using existing ${RESULTS_DIR}/${exp_name}/results.pkl"
    return
  fi

  local aggregate_file="${RESULTS_DIR}/${exp_name}/results.pkl"
  local backup_file="${RESULTS_DIR}/${exp_name}/results.pre_c1.pkl"
  if [[ -f "${aggregate_file}" && ! -f "${backup_file}" ]]; then
    cp "${aggregate_file}" "${backup_file}"
    echo "Backed up previous aggregate to ${backup_file}"
  fi

  local deterministic_eval_args=()
  if [[ "${REQUIRE_DETERMINISTIC_SUPERSAMPLE_EVAL:-0}" == "1" ]]; then
    deterministic_eval_args+=(--require_deterministic_supersample_eval)
  fi

  "${PYTHON_BIN}" -B -m scripts.parse_results \
    -d "${PARSE_DEVICE}" \
    --results_dir "${RESULTS_DIR}" \
    --exp_name "${exp_name}" \
    --ns "${ns[@]}" \
    --n_seeds "${n_seeds}" \
    --n_S_seeds "${n_selector_seeds}" \
    --epochs "${epoch}" \
    --num_classes "${num_classes}" \
    "${deterministic_eval_args[@]}" \
    --include_theorem8_samplewise_loss_difference_valpha_bound \
    --include_theorem9_samplewise_single_loss_valpha_bound \
    --include_theorem10_samplewise_loss_pair_valpha_bound \
    --include_theorem11_samplewise_binarized_single_loss_valpha_bound \
    --include_theorem12_samplewise_binarized_loss_difference_valpha_bound \
    --include_theorem13_samplewise_binarized_loss_pair_valpha_bound \
    --valpha_alphas 1.005 1.1 1.25 1.5 2.0 3.0 5.0 \
    --theorem8_num_bins 30 \
    --theorem9_num_bins 30 \
    --theorem10_num_bins 30 \
    --theorem9_symmetrization average \
    --theorem11_symmetrization average
}
