#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-${REPO}/openvla/checkpoints/sft/scratch}"
SFT_RUN_DIR="${SFT_RUN_ROOT}/steps_${MAX_STEPS:-1000}-no_aug"
PIPELINE_LOG="${PIPELINE_LOG:-${RUN_DIR}/pipeline_scratch.log}"
SFT_LOG="${SFT_LOG:-${RUN_DIR}/sft_train_scratch.log}"
DATABC_LOG="${DATABC_LOG:-${RUN_DIR}/databc_train_scratch.log}"
export DATABC_NAME="${DATABC_NAME:-DataBC_pick_red_cube_scratch}"
export STEPS_MAX="${STEPS_MAX:-2000000}"
export ALG_ENTROPY_COEF="${ALG_ENTROPY_COEF:-0.005}"
export REWARD_REACH_COEF="${REWARD_REACH_COEF:-0.3}"
export SFT_RUN_ROOT
LOCK="${RUN_DIR}/pipeline.lock"

mkdir -p "${RUN_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "pipeline already running" >&2
  exit 0
fi

pick_sft_lora() {
  local best=""
  local preferred
  for preferred in lora_000500 lora_000750 lora_000250 lora_001000; do
    if [[ -d "${SFT_RUN_DIR}/${preferred}" && -f "${SFT_RUN_DIR}/${preferred}/adapter_config.json" ]]; then
      best="${SFT_RUN_DIR}/${preferred}"
      break
    fi
  done
  if [[ -z "${best}" ]]; then
    local latest
    latest="$(ls -1d "${SFT_RUN_DIR}"/lora_* 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${latest}" ]]; then
      best="${latest}"
    fi
  fi
  if [[ -z "${best}" ]]; then
    echo "no SFT LoRA checkpoint found under ${SFT_RUN_DIR}" >&2
    exit 1
  fi
  printf '%s\n' "${best}"
}

echo "pipeline start $(date -Is) sft_root=${SFT_RUN_ROOT} databc_name=${DATABC_NAME}" | tee -a "${PIPELINE_LOG}"
SKIP_SFT="${SKIP_SFT:-0}"
if [[ "${SKIP_SFT}" != "1" ]]; then
  bash "${RUN_DIR}/train_sft.sh" 2>&1 | tee -a "${SFT_LOG}"
else
  echo "SKIP_SFT=1, using existing LoRA under ${SFT_RUN_DIR}" | tee -a "${PIPELINE_LOG}"
fi
LORA_PATH="$(pick_sft_lora)"
echo "selected SFT LoRA ${LORA_PATH}" | tee -a "${PIPELINE_LOG}"
printf '%s\n' "${LORA_PATH}" > "${RUN_DIR}/selected_sft_lora.txt"
# Release the pipeline lock before DataBC. train_databc.sh re-acquires it so a
# parallel DataBC launch waits instead of sharing the GPU with SFT.
flock -u 9
bash "${RUN_DIR}/train_databc.sh" "${LORA_PATH}" 2>&1 | tee -a "${DATABC_LOG}"
echo "pipeline done $(date -Is)" | tee -a "${PIPELINE_LOG}"
