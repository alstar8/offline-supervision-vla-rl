#!/usr/bin/env bash
# End-to-end v4 pipeline: wait for pick_red_cube_100_v4 collection (scene
# airy_table_scene14sep26_left_image with calibrated lighting/hand materials),
# then bg-augment, convert to SFT npz, build the TFDS dataset, run SFT
# (LoRA on vision + LLM; fully trained projectors + LM head),
# then online DataBC in the same scene.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
SIM2REAL="${REPO}/sim2real"
RUN_DIR="${SIM2REAL}/runs/rl/pick_red_cube_sft_databc"
COLLECT_DIR="${SIM2REAL}/runs/manual/pick_red_cube_100_v4"
BG_DIR="${RUN_DIR}/sft_v4_episodes_bg"
PREP_DIR="${RUN_DIR}/sft_v4_proprio"
DATASETS="${REPO}/datasets"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"
LOG="${RUN_DIR}/pipeline_v4.log"
LOCK="${RUN_DIR}/pipeline_v4.lock"
SFT_RUN_ROOT="checkpoints/sft/openvla_v2_scene14sep26_vision_llm"
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_1000-no_aug"
VARIANTS="${VARIANTS:-10}"
TARGET_EPISODES=100

mkdir -p "${RUN_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "v4 pipeline already running" >&2
  exit 0
fi

log() { echo "[v4pipeline] $* $(date -Is)" | tee -a "${LOG}"; }

log "waiting for v4 collection (target=${TARGET_EPISODES})"
count_episodes() { find "${COLLECT_DIR}" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l; }
while true; do
  n="$(count_episodes)"
  if grep -q '^DONE' "${COLLECT_DIR}/collect.log" 2>/dev/null || [ "${n}" -ge "${TARGET_EPISODES}" ]; then
    break
  fi
  if grep -q '^ABORT' "${COLLECT_DIR}/collect.log" 2>/dev/null; then
    log "collection ABORTED with ${n} episodes"
    exit 1
  fi
  sleep 60
done
n="$(count_episodes)"
log "collection complete: ${n} episodes"

if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
log "stage 1: render_bg_variants (${VARIANTS} variants) -> ${BG_DIR}"
(cd "${SIM2REAL}" && \
  PYTHONPATH="${SIM2REAL}:${SIM2REAL}/openreal2sim/simulation/maniskill" \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "${PY}" "${RUN_DIR}/render_bg_variants.py" "${COLLECT_DIR}" "${BG_DIR}" \
    --variants "${VARIANTS}" \
    --key airy_table_scene14sep26_left_image \
    --config_path "${SIM2REAL}/config/config_debug.yaml" \
    --scene assets/scenes/airy_table_scene14sep26_left_image/simulation/scene.json) 2>&1 | tee -a "${LOG}"

log "stage 2: prepare_sft_v2_episodes -> ${PREP_DIR}"
"${PY}" "${RUN_DIR}/prepare_sft_v2_episodes.py" "${BG_DIR}" "${PREP_DIR}" 2>&1 | tee -a "${LOG}"
else
log "SKIP_DATA_PREP=1, using existing ${PREP_DIR}"
fi

if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
log "stage 3: tfds build sft_v2 -> ${DATASETS}"
(
  export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:${PATH}"
  export CUDA_VISIBLE_DEVICES=""
  export RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}"
  export RLVLA_SFT_TOTAL_DEMOS=$((TARGET_EPISODES * VARIANTS))
  export RLVLA_SFT_VAL_DEMOS=16
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export PYTHONUNBUFFERED=1
  cd "${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset"
  tfds build --overwrite --data_dir "${DATASETS}"
) 2>&1 | tee -a "${LOG}"
else
log "SKIP_DATA_PREP=1, using existing ${DATASETS}/sft_v2"
fi
if [[ ! -d "${DATASETS}/sft_v2/1.0.0" ]]; then
  log "tfds did not produce ${DATASETS}/sft_v2/1.0.0"
  exit 1
fi

log "stage 4: SFT (lora_target=vision_llm, train_projector, train_action_head)"
SFT_RUN_ROOT="${SFT_RUN_ROOT}" bash "${RUN_DIR}/train_sft_v3.sh" 2>&1 | tee -a "${RUN_DIR}/sft_train_vision_llm.log"

pick_sft_lora() {
  local preferred
  for preferred in lora_000500 lora_000750 lora_000250 lora_001000; do
    if [[ -d "${SFT_RUN_DIR}/${preferred}" && -f "${SFT_RUN_DIR}/${preferred}/adapter_config.json" ]]; then
      printf '%s\n' "${SFT_RUN_DIR}/${preferred}"
      return 0
    fi
  done
  ls -1d "${SFT_RUN_DIR}"/lora_* 2>/dev/null | sort | tail -n 1
}
LORA_PATH="$(pick_sft_lora)"
if [[ -z "${LORA_PATH}" ]]; then
  log "no SFT LoRA checkpoint found under ${SFT_RUN_DIR}"
  exit 1
fi
log "stage 5: DataBC online in new scene, init ${LORA_PATH}"
printf '%s\n' "${LORA_PATH}" > "${RUN_DIR}/selected_sft_lora_v4.txt"
bash "${RUN_DIR}/train_databc_v3.sh" "${LORA_PATH}" 2>&1 | tee -a "${RUN_DIR}/databc_train_vision_llm.log"

log "pipeline done"
