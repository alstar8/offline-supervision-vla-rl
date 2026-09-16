#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CKPT="${CKPT:-${SCRIPT_DIR}/wandb/offline-run-20260915_093914-9ehszx0r/glob/steps_0009}"
PY="${PY:-/home/admin/.conda/envs/rlvla_env/bin/python}"
export RC5_PYTHON_API_ROOT="${RC5_PYTHON_API_ROOT:-/home/admin/Desktop/RC5_Hand_OpenVLA/python_api}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO}/openvla${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET=1

cd "${REPO}"
exec "${PY}" sim2real/real_replay/eval_openvla_real.py \
  --checkpoint "${CKPT}" \
  --instruction "Pick red cube" \
  --unnorm-key sft \
  "$@"
