#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PY="${PY:-/home/admin/.conda/envs/rlvla_env/bin/python}"
TARGET="${1:?usage: test_action_square.sh sim|real [extra args]}"
shift || true

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/action_square/${STAMP}}"
OUT_DIR="${OUT_ROOT}/${TARGET}"

export RC5_PYTHON_API_ROOT="${RC5_PYTHON_API_ROOT:-/home/admin/Desktop/RC5_Hand_OpenVLA/python_api}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Do not put sim2real first: it shadows ManiSkill.
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi

mkdir -p "${OUT_DIR}"
echo "target=${TARGET}  output=${OUT_DIR}  gpu=${CUDA_VISIBLE_DEVICES}"
cd "${REPO}"
exec "${PY}" sim2real/real_replay/test_openvla_action_square.py \
  --target "${TARGET}" \
  --output-dir "${OUT_DIR}" \
  "$@"
