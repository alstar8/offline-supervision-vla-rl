#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

parse_seed_args "$@"
cd "${SIMPLER_ENV_DIR}"

RUN_NAME="DataBC_seed${SEED}"

XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo_sft.py \
  --name="${RUN_NAME}" \
  --env_id="${ENV_ID}" \
  --vla_path="${VLA_PATH}" \
  --seed="${SEED}" \
  --vla_unnorm_key="${VLA_UNNORM_KEY}" \
  --bc_to_ref_enabled \
  --no_sft_image_aug \
  --sft_data_root_dir=../datasets \
  --sft_dataset_name=sft \
  --bc_to_ref_coef=0.6 \
  --bc_to_ref_hold_steps=100000 \
  --bc_to_ref_decay_steps=300000 \
  "${FORWARD_ARGS[@]}"
