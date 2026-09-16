#!/usr/bin/env bash
# Rebuild wiped sft_v2 + SFT LoRA, then resume DataBC from lora_000500.
# Checkpoints go to $RUN_DIR/ckpts (outside wandb/).
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
PREP_DIR="${RUN_DIR}/sft_v5_proprio"
DATASETS="${REPO}/datasets"
SFT_RUN_ROOT="checkpoints/sft/openvla_v2_scene14sep26_vision_llm"
# finetune.py writes under steps_${MAX_STEPS}-no_aug, not the original 1k-step dir.
SFT_MAX_STEPS="${SFT_MAX_STEPS:-500}"
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_${SFT_MAX_STEPS}-no_aug"
SFT_RUN_DIR_LEGACY="${REPO}/openvla/${SFT_RUN_ROOT}/steps_1000-no_aug"
SFT_COPY="${RUN_DIR}/sft_init/lora_000500"
LOG="${RUN_DIR}/restart_after_crash.log"

export PATH="${CONDA_BIN}:${PATH}"
export PYTHONUNBUFFERED=1

log() { echo "[restart] $* $(date -Is)" | tee -a "${LOG}"; }

log "start GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1) MiB"

if [[ ! -d "${PREP_DIR}" ]]; then
  log "missing proprio dumps ${PREP_DIR}"
  exit 1
fi
N_NPZ="$(find "${PREP_DIR}" -maxdepth 1 -name 'episode_*.npz' | wc -l)"
log "found ${N_NPZ} proprio npz in ${PREP_DIR}"

if [[ ! -d "${DATASETS}/sft_v2/1.0.0" ]]; then
  log "rebuild TFDS sft_v2"
  mkdir -p "${DATASETS}"
  (
    export CUDA_VISIBLE_DEVICES=""
    export RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}"
    export RLVLA_SFT_TOTAL_DEMOS="${N_NPZ}"
    export RLVLA_SFT_VAL_DEMOS=16
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    cd "${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset"
    tfds build --overwrite --data_dir "${DATASETS}"
  ) 2>&1 | tee -a "${LOG}"
fi
if [[ ! -d "${DATASETS}/sft_v2/1.0.0" ]]; then
  log "tfds did not produce ${DATASETS}/sft_v2/1.0.0"
  exit 1
fi
log "dataset ready ${DATASETS}/sft_v2/1.0.0"

NEED_SFT=1
pick_lora() {
  local cand
  for cand in "$@"; do
    if [[ -f "${cand}/adapter_model.safetensors" || -f "${cand}/adapter_model.bin" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

if LORA_PATH="$(pick_lora "${SFT_COPY}" "${SFT_RUN_DIR}/lora_000500" "${SFT_RUN_DIR_LEGACY}/lora_000500")"; then
  NEED_SFT=0
  log "reuse existing SFT LoRA ${LORA_PATH}"
fi

if [[ "${NEED_SFT}" == "1" ]]; then
  log "re-run SFT to step ${SFT_MAX_STEPS} (vision+LLM LoRA)"
  MAX_STEPS="${SFT_MAX_STEPS}" SAVE_STEPS="250,${SFT_MAX_STEPS}" \
    SFT_RUN_ROOT="${SFT_RUN_ROOT}" \
    bash "${RUN_DIR}/train_sft_v3.sh" 2>&1 | tee -a "${RUN_DIR}/sft_train_vision_llm.log" "${LOG}"
  LORA_PATH="${SFT_RUN_DIR}/lora_000500"
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  log "missing SFT LoRA ${LORA_PATH}"
  exit 1
fi
mkdir -p "${RUN_DIR}/sft_init"
if [[ "${LORA_PATH}" != "${SFT_COPY}" ]]; then
  rm -rf "${SFT_COPY}"
  cp -a "${LORA_PATH}" "${SFT_COPY}"
  log "copied SFT LoRA to ${SFT_COPY}"
fi
printf '%s\n' "${SFT_COPY}" > "${RUN_DIR}/selected_sft_lora_v4.txt"

log "start DataBC from ${SFT_COPY}"
export DATABC_CKPT_DIR="${RUN_DIR}/ckpts"
export DATABC_NAME="${DATABC_NAME:-DataBC_pick_red_cube_v3_scene14sep26_vision_llm_critic20_r2}"
export FREEZE_ACTOR_UPDATES="${FREEZE_ACTOR_UPDATES:-5}"
export ALG_VF_COEF="${ALG_VF_COEF:-0.5}"
export ALG_PPO_EPOCH="${ALG_PPO_EPOCH:-3}"
bash "${RUN_DIR}/train_databc_v3.sh" "${SFT_COPY}" 2>&1 | tee -a "${RUN_DIR}/databc_train_vision_llm_critic20_r2.log" "${LOG}"
log "DataBC finished"
