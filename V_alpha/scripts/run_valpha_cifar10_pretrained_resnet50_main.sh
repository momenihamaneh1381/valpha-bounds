#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
PARSE_DEVICE="${PARSE_DEVICE:-${DEVICE}}"

# Optional destination for validated completed runs.
PERSIST_RESULTS_ROOT="${PERSIST_RESULTS_ROOT:-}"

EXP_NAME="cifar10-pretrained-resnet50-valpha-main"
CONFIG="configs/pretrained-resnet50-cifar10.json"
DATASET="cifar10"
BATCH_SIZE=64
EPOCHS=40
SAVE_ITER=40
VIS_ITER=10000
OPTIMIZER="sgd"
LR=0.01
MOMENTUM=0.9

LOCAL_EXP_ROOT="results/${EXP_NAME}"
PERSIST_EXP_ROOT=""
PARSE_RESULTS_ROOT="results"

is_complete_run() {
  local run_dir="$1"
  [[ -f "${run_dir}/saved_data.pkl" ]] && \
    { [[ -f "${run_dir}/checkpoints/epoch39.mdl" ]] || \
      [[ -f "${run_dir}/checkpoints/final.mdl" ]]; }
}

copy_directory_contents() {
  local source_dir="$1"
  local destination_dir="$2"

  mkdir -p "${destination_dir}"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${source_dir}/" "${destination_dir}/"
  else
    cp -a "${source_dir}/." "${destination_dir}/"
  fi
}

persist_completed_run() {
  local local_run="$1"
  local persistent_run="$2"

  [[ -n "${PERSIST_EXP_ROOT}" ]] || return 0

  if ! is_complete_run "${local_run}"; then
    echo "Refusing to persist incomplete local run: ${local_run}" >&2
    return 1
  fi

  if is_complete_run "${persistent_run}"; then
    return 0
  fi

  local key
  local temporary
  key="$(basename "${local_run}")"
  temporary="${PERSIST_EXP_ROOT}/.${key}.tmp-$$"

  mkdir -p "${PERSIST_EXP_ROOT}"
  rm -rf "${temporary}"

  echo "Persisting completed run to Drive: ${key}"
  copy_directory_contents "${local_run}" "${temporary}"

  if ! is_complete_run "${temporary}"; then
    echo "Persistent copy failed validation: ${key}" >&2
    rm -rf "${temporary}"
    return 1
  fi

  if [[ -e "${persistent_run}" ]]; then
    mv "${persistent_run}" \
      "${persistent_run}.incomplete-$(date +%Y%m%d-%H%M%S)"
  fi

  # Publish only after the copy has passed validation.
  mv "${temporary}" "${persistent_run}"
}

if [[ -n "${PERSIST_RESULTS_ROOT}" ]]; then
  PERSIST_EXP_ROOT="${PERSIST_RESULTS_ROOT%/}/${EXP_NAME}"
  PARSE_RESULTS_ROOT="${PERSIST_RESULTS_ROOT%/}"
  mkdir -p "${PERSIST_EXP_ROOT}"
  echo "Using persistent run index: ${PERSIST_EXP_ROOT}"
fi

mkdir -p "${LOCAL_EXP_ROOT}"

NS=(1000 5000 20000)
SEEDS=(0 1)
S_SEEDS=($(seq 0 39))

for n in "${NS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for S_seed in "${S_SEEDS[@]}"; do
      key="n=${n},seed=${seed},S_seed=${S_seed}"
      local_run="${LOCAL_EXP_ROOT}/${key}"
      persistent_run=""

      if [[ -n "${PERSIST_EXP_ROOT}" ]]; then
        persistent_run="${PERSIST_EXP_ROOT}/${key}"

        # Only two final artifacts are stat'ed on Drive. Completed runs are not
        # copied back into the runtime.
        if is_complete_run "${persistent_run}"; then
          echo "Skipping completed persistent run: ${key}"
          # The persistent copy is authoritative; no local mirror is needed.
          rm -rf "${local_run}"
          continue
        fi

        if [[ -d "${persistent_run}" ]]; then
          echo "Persistent run exists but is incomplete; it will be replaced after retraining: ${key}" >&2
        fi
      fi

      # A complete local run can occur when the launcher is restarted while the
      # same VM is still alive. Persist it if necessary, then skip it.
      if is_complete_run "${local_run}"; then
        if [[ -n "${PERSIST_EXP_ROOT}" ]]; then
          persist_completed_run "${local_run}" "${persistent_run}"
        fi
        echo "Skipping completed local run: ${key}"
        continue
      fi

      # train_classifier no-ops when saved_data.pkl is present. Remove any
      # incomplete local run so this combination is retrained cleanly.
      if [[ -d "${local_run}" ]]; then
        echo "Removing incomplete local run before retraining: ${key}" >&2
        rm -rf "${local_run}"
      fi

      "${PYTHON_BIN}" -B -m scripts.train_classifier \
        -c "${CONFIG}" \
        -d "${DEVICE}" \
        -b "${BATCH_SIZE}" \
        -e "${EPOCHS}" \
        -s "${SAVE_ITER}" \
        -v "${VIS_ITER}" \
        --exp_name "${EXP_NAME}" \
        -D "${DATASET}" \
        -m StandardClassifier \
        -A \
        --resize_to_imagenet \
        --optimizer "${OPTIMIZER}" \
        --lr "${LR}" \
        --momentum "${MOMENTUM}" \
        --n "${n}" \
        --seed "${seed}" \
        --S_seed "${S_seed}"

      if ! is_complete_run "${local_run}"; then
        echo "Run did not finish correctly: ${local_run}" >&2
        exit 1
      fi

      if [[ -n "${PERSIST_EXP_ROOT}" ]]; then
        persist_completed_run "${local_run}" "${persistent_run}"
        # Parsing reads directly from Drive, so release local disk immediately.
        rm -rf "${local_run}"
      fi
    done
  done
done

# When persistence is enabled, parse directly from Drive. No completed run
# directories need to be restored into /content.
"${PYTHON_BIN}" -B -m scripts.parse_results \
  -d "${PARSE_DEVICE}" \
  --results_dir "${PARSE_RESULTS_ROOT}" \
  --exp_name "${EXP_NAME}" \
  --ns 1000 5000 20000 \
  --n_seeds 2 \
  --n_S_seeds 40 \
  --epochs 40 \
  --num_classes 10 \
  --batch_size "${BATCH_SIZE}" \
  --require_deterministic_supersample_eval \
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
