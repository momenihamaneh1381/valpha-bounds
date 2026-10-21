#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
PARSE_DEVICE="${PARSE_DEVICE:-${DEVICE}}"

EXP_NAME="mnist-4vs9-CNN-noise-0.15-valpha-main"
CONFIG="configs/binary-mnist-4layer-CNN.json"
DATASET="mnist"
BATCH_SIZE=128
EPOCHS=200
SAVE_ITER=20
VIS_ITER=10000
ERROR_PROB=0.15

NS=(75 250 1000 4000)
SEEDS=(0 1 2 3 4)
S_SEEDS=($(seq 0 29))

for n in "${NS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for S_seed in "${S_SEEDS[@]}"; do
      run_dir="results/${EXP_NAME}/n=${n},seed=${seed},S_seed=${S_seed}"
      epoch_ckpt="${run_dir}/checkpoints/epoch199.mdl"
      final_ckpt="${run_dir}/checkpoints/final.mdl"
      saved_data="${run_dir}/saved_data.pkl"

      if [[ -f "${epoch_ckpt}" || -f "${final_ckpt}" ]]; then
        if [[ ! -f "${saved_data}" ]]; then
          echo "Found checkpoint but missing saved_data.pkl for ${run_dir}; refusing to skip an incomplete run." >&2
          exit 1
        fi
        echo "Skipping completed run: n=${n}, seed=${seed}, S_seed=${S_seed}"
        continue
      fi

      if [[ -f "${saved_data}" ]]; then
        echo "Found saved_data.pkl but no epoch199.mdl/final.mdl for ${run_dir}; train_classifier will no-op." >&2
        echo "Remove or repair this incomplete run directory before continuing." >&2
        exit 1
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
        --which_labels 4 9 \
        --deterministic \
        --error_prob "${ERROR_PROB}" \
        --n "${n}" \
        --seed "${seed}" \
        --S_seed "${S_seed}"
    done
  done
done

"${PYTHON_BIN}" -B -m scripts.parse_results \
  -d "${PARSE_DEVICE}" \
  --exp_name "${EXP_NAME}" \
  --ns 75 250 1000 4000 \
  --n_seeds 5 \
  --n_S_seeds 30 \
  --epochs 200 \
  --num_classes 2 \
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
