#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SFT_RUN_DIR="${REPO}/openvla/checkpoints/sft/steps_1000-no_aug"
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

echo "pipeline start $(date -Is)" | tee -a "${RUN_DIR}/pipeline.log"
bash "${RUN_DIR}/train_sft.sh" 2>&1 | tee -a "${RUN_DIR}/sft_train.log"
LORA_PATH="$(pick_sft_lora)"
echo "selected SFT LoRA ${LORA_PATH}" | tee -a "${RUN_DIR}/pipeline.log"
printf '%s\n' "${LORA_PATH}" > "${RUN_DIR}/selected_sft_lora.txt"
bash "${RUN_DIR}/train_databc.sh" "${LORA_PATH}" 2>&1 | tee -a "${RUN_DIR}/databc_train.log"
echo "pipeline done $(date -Is)" | tee -a "${RUN_DIR}/pipeline.log"
