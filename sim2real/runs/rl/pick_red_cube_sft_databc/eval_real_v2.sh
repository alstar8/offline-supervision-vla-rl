#!/usr/bin/env bash
# OpenVLA-V2 closed-loop eval on the real RC5 + AeroHand: separate scene and wrist
# images plus 7D proprio (see sim2real/real_replay/eval_openvla_v2_real.py).
# Same layout as eval_real.sh, which runs the V1 composited-image eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CKPT="${CKPT:-${SCRIPT_DIR}/wandb/offline-run-20260915_093914-9ehszx0r/glob/steps_0009}"
PY="${PY:-/home/admin/.conda/envs/rlvla_env/bin/python}"
export RC5_PYTHON_API_ROOT="${RC5_PYTHON_API_ROOT:-/home/admin/Desktop/RC5_Hand_OpenVLA/python_api}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# V2 builds its config through the base model's remote code (openvla/openvla-7b),
# which is cached in this HF home next to a link to the base weights.
export HF_HOME="${HF_HOME:-/home/aermakov/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO}"
exec "${PY}" sim2real/real_replay/eval_openvla_v2_real.py \
  --checkpoint "${CKPT}" \
  --instruction "Pick red cube" \
  --unnorm-key sft_v2 \
  "$@"
