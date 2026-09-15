#!/usr/bin/env bash
# After 1000 unique successes: 10 bg variants -> 10k SFT trajs, XY-weighted SFT, RefKL from scratch on GPUs 0-2.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
SIM2REAL="${REPO}/sim2real"
DATABC="${SIM2REAL}/runs/rl/pick_red_cube_sft_databc"
REFKL="${SIM2REAL}/runs/rl/pick_red_cube_refkl"
COLLECT_DIR="${SIM2REAL}/runs/manual/pick_red_cube_100_v4"
BG_CACHE="${DATABC}/sft_v4_episodes_bg"
PREP_DIR="${DATABC}/sft_v5_proprio"
DATASETS="${REFKL}/datasets_v5"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"
LOG="${REFKL}/logs/pipeline_v5.log"
LOCK="${REFKL}/pipeline_v5.lock"
SFT_RUN_ROOT="checkpoints/sft/openvla_v2_scene14sep26_10k_xy"
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_16000-no_aug"
VARIANTS="${VARIANTS:-10}"
TARGET_UNIQUE="${TARGET_UNIQUE:-1000}"
TARGET_TRAJ=$((TARGET_UNIQUE * VARIANTS))

mkdir -p "${REFKL}/logs" "${REFKL}/pids" "${PREP_DIR}" "${DATASETS}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "v5 pipeline already running" >&2
  exit 0
fi

log() { echo "[v5pipeline] $* $(date -Is)" | tee -a "${LOG}"; }

count_unique() { find "${COLLECT_DIR}" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l; }
count_prep() { find "${PREP_DIR}" -maxdepth 1 -name 'episode_*_bg*.npz' 2>/dev/null | wc -l; }

log "stage 0: prepare existing 100 unique x 10 bg on CPU while collection continues"
(
  export RLVLA_MAX_STEPS_PER_CHUNK=4
  export PYTHONUNBUFFERED=1
  "${PY}" "${DATABC}/prepare_sft_v2_episodes.py" "${BG_CACHE}" "${PREP_DIR}"
) >> "${LOG}" 2>&1 &
PREP_EXISTING_PID=$!
echo "${PREP_EXISTING_PID}" > "${REFKL}/pids/prep_existing_v5.pid"

log "waiting for ${TARGET_UNIQUE} unique trajectories in ${COLLECT_DIR}"
while true; do
  n="$(count_unique)"
  if [ "${n}" -ge "${TARGET_UNIQUE}" ]; then
    break
  fi
  log "collection progress unique=${n}/${TARGET_UNIQUE} prep=$(count_prep)"
  sleep 120
done
n="$(count_unique)"
log "collection complete: ${n} unique"

log "waiting for existing-bg prepare pid=${PREP_EXISTING_PID}"
wait "${PREP_EXISTING_PID}"
log "existing-bg prepare done prep=$(count_prep)"

log "stage 1: render+prepare remaining uniques on GPUs 0-2 (stream, no extra bg keep)"
render_one() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH="${SIM2REAL}:${SIM2REAL}/openreal2sim/simulation/maniskill" \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  RLVLA_MAX_STEPS_PER_CHUNK=4 \
  PYTHONUNBUFFERED=1 \
  "${PY}" "${DATABC}/render_prepare_shard.py" "${COLLECT_DIR}" "${PREP_DIR}" \
    --bg_cache_dir "${BG_CACHE}" \
    --tmp_dir "${DATABC}/sft_v5_bg_tmp_gpu${gpu}" \
    --variants "${VARIANTS}" \
    --start "${start}" \
    --end "${end}" \
    --key airy_table_scene14sep26_left_image \
    --config_path "${SIM2REAL}/config/config_debug.yaml" \
    --scene assets/scenes/airy_table_scene14sep26_left_image/simulation/scene.json
}

# 1000 unique split across 3 GPUs.
render_one 0 0 334 >> "${LOG}" 2>&1 &
echo $! > "${REFKL}/pids/render_gpu0.pid"
render_one 1 334 667 >> "${LOG}" 2>&1 &
echo $! > "${REFKL}/pids/render_gpu1.pid"
render_one 2 667 1000 >> "${LOG}" 2>&1 &
echo $! > "${REFKL}/pids/render_gpu2.pid"
wait "$(cat "${REFKL}/pids/render_gpu0.pid")"
wait "$(cat "${REFKL}/pids/render_gpu1.pid")"
wait "$(cat "${REFKL}/pids/render_gpu2.pid")"
n_prep="$(count_prep)"
log "render+prepare complete prep=${n_prep} expected=${TARGET_TRAJ}"
if [ "${n_prep}" -lt "${TARGET_TRAJ}" ]; then
  log "ERROR: prepared ${n_prep} < ${TARGET_TRAJ}"
  exit 1
fi

if [[ -f "${PREP_DIR}/sft_episode_stats.json" ]]; then
  EPISODE_LEN="$(${PY} -c 'import json; print(json.load(open("'"${PREP_DIR}"'/sft_episode_stats.json"))["recommended_episode_len"])')"
else
  EPISODE_LEN=160
fi
log "recommended episode_len=${EPISODE_LEN}"

log "stage 2: tfds build sft_v2 -> ${DATASETS} (${TARGET_TRAJ} trajs, val=160)"
(
  export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:${PATH}"
  export CUDA_VISIBLE_DEVICES=""
  export RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}"
  export RLVLA_SFT_TOTAL_DEMOS="${TARGET_TRAJ}"
  export RLVLA_SFT_VAL_DEMOS=160
  export RLVLA_SFT_DISABLE_FILTER=1
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export PYTHONUNBUFFERED=1
  cd "${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset"
  tfds build --overwrite --data_dir "${DATASETS}"
) 2>&1 | tee -a "${LOG}"
if [[ ! -d "${DATASETS}/sft_v2/1.0.0" ]]; then
  log "tfds did not produce ${DATASETS}/sft_v2/1.0.0"
  exit 1
fi

log "stage 2b: drop prepared npz to free disk (TFDS is the training source)"
rm -rf "${PREP_DIR}"
rm -rf "${DATABC}/sft_v5_bg_tmp_gpu0" "${DATABC}/sft_v5_bg_tmp_gpu1" "${DATABC}/sft_v5_bg_tmp_gpu2"

log "stage 3: SFT 3-GPU XY-weighted, skip TF stretch"
SFT_RUN_ROOT="${SFT_RUN_ROOT}" \
MAX_STEPS="${MAX_STEPS:-16000}" \
SAVE_STEPS="${SAVE_STEPS:-2000,4000,8000,12000,16000}" \
EVAL_STEPS="${EVAL_STEPS:-1000}" \
NPROC=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
DATA_ROOT_DIR="${DATASETS}" \
SKIP_IMAGE_RESIZE=1 \
ACTION_DIM_LOSS_WEIGHTS="3:3:1.5:0:0:0:1" \
SHUFFLE_BUFFER_SIZE=8000 \
bash "${DATABC}/train_sft_v3.sh" 2>&1 | tee -a "${REFKL}/logs/sft_v5_10k.log"

LORA_PATH="$("${PY}" "${REFKL}/pick_sft_lora.py" "${SFT_RUN_DIR}")"
if [[ -z "${LORA_PATH}" || ! -d "${LORA_PATH}" ]]; then
  log "no SFT LoRA checkpoint found under ${SFT_RUN_DIR}"
  exit 1
fi
log "stage 4: RefKL from scratch on GPUs 0-2, teacher=${LORA_PATH} episode_len=${EPISODE_LEN}"
printf '%s\n' "${LORA_PATH}" > "${REFKL}/selected_sft_lora_v5.txt"

REFKL_GPUS=0,1,2 \
NUM_ENVS="${NUM_ENVS:-32}" \
BUFFER_INFERBATCH=8 \
SFT_LORA_PATH="${LORA_PATH}" \
REFKL_LOAD_PATH="${LORA_PATH}" \
REFKL_RESUME_EP=0 \
REFKL_JOB_DIR="${REFKL}/ddp_sft_v5_10k" \
NAME=RefKL_pick_red_cube_v5_10k \
MASTER_PORT=29691 \
LOG="${REFKL}/logs/ddp_sft_v5_10k.log" \
PID_FILE="${REFKL}/pids/ddp_v5_10k.pid" \
EPISODE_LEN="${EPISODE_LEN}" \
SFT_DATA_ROOT_DIR="${DATASETS}" \
KL_TO_REF_ENABLED=1 \
FREEZE_ACTOR_UPDATES=3 \
SFT_SKIP_IMAGE_RESIZE=1 \
SFT_ACTION_DIM_WEIGHTS="3:3:1.5:0:0:0:1" \
SFT_SHUFFLE_BUFFER_SIZE=8000 \
bash "${REFKL}/launch_gpus_v3.sh"

log "pipeline launched RefKL pid=$(cat "${REFKL}/pids/ddp_v5_10k.pid")"
log "pipeline done (RefKL running)"
