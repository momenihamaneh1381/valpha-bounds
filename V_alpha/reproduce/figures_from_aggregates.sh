#!/usr/bin/env bash
set -euo pipefail
REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKIP_PARSE=1
bash "${REPRO_DIR}/figure1_mnist.sh"
bash "${REPRO_DIR}/figure2_cifar10.sh"
bash "${REPRO_DIR}/figure3_mnist_noise.sh"
