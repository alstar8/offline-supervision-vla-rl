#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_seed_args "$@"
cd "${SIMPLER_ENV_DIR}"

RUN_NAME="PPO_seed${SEED}"

XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="${RUN_NAME}" \
  --env_id="${ENV_ID}" \
  --vla_path="${VLA_PATH}" \
  --vla_unnorm_key="${VLA_UNNORM_KEY}" \
  --seed="${SEED}" \
  "${FORWARD_ARGS[@]}"
