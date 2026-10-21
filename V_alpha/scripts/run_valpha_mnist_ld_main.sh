#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
PARSE_DEVICE="${PARSE_DEVICE:-${DEVICE}}"

EXP_NAME="mnist-4vs9-CNN-LD-valpha-main"
CONFIG="configs/binary-mnist-4layer-CNN.json"
DATASET="mnist"
BATCH_SIZE=100
EPOCHS=40
SAVE_ITER=4
VIS_ITER=10000
LD_LR=0.004
LD_BETA=10.0

NS=(4000)
SEEDS=(0 1 2 3 4)
S_SEEDS=($(seq 0 29))
PARSE_EPOCHS=(4 8 12 16 20 24 28 32 36 40)

for n in "${NS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for S_seed in "${S_SEEDS[@]}"; do
      run_dir="results/${EXP_NAME}/n=${n},seed=${seed},S_seed=${S_seed}"
      epoch_ckpt="${run_dir}/checkpoints/epoch39.mdl"
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
        echo "Found saved_data.pkl but no epoch39.mdl/final.mdl for ${run_dir}; train_classifier will no-op." >&2
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
        -m LangevinDynamics \
        --ld_lr "${LD_LR}" \
        --ld_beta "${LD_BETA}" \
        --n "${n}" \
        --seed "${seed}" \
        --S_seed "${S_seed}"
    done
  done
done

"${PYTHON_BIN}" -B -m scripts.parse_results \
  -d "${PARSE_DEVICE}" \
  --exp_name "${EXP_NAME}" \
  --ns 4000 \
  --n_seeds 5 \
  --n_S_seeds 30 \
  --epochs "${PARSE_EPOCHS[@]}" \
  --num_classes 2 \
  --batch_size "${BATCH_SIZE}" \
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
