#!/usr/bin/env bash
# OpenVLA-V2 RefKL pipeline on airy_table_scene14sep26_left_image:
#   pick_red_cube demos x 10 360-bg variants
#   -> prepare SFT npz -> RLDS sft_v2
#   -> SFT LoRA (vision backbone + LLM; projector + lm_head fully trained)
#   -> RefKL DDP in the same scene
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
SIM2REAL="${REPO}/sim2real"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
REFKL_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SRC_DEMOS="${SRC_DEMOS:-${SIM2REAL}/runs/manual/pick_red_cube_100_v3}"
RENDER_DIR="${SFT_DIR}/sft_v3_episodes_bg"
PREP_DIR="${SFT_DIR}/sft_v3_proprio"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2_scene14sep26_refkl_vision}"
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_${MAX_STEPS:-1000}-no_aug"
RENDER_GPUS="${RENDER_GPUS:-0,1,2,4,5,6,7}"
SFT_GPU="${SFT_GPU:-0}"
N_VARIANTS="${N_VARIANTS:-10}"
TARGET_EPISODES="${TARGET_EPISODES:-100}"
SCENE_KEY="airy_table_scene14sep26_left_image"
SCENE_JSON="${SIM2REAL}/assets/scenes/${SCENE_KEY}/simulation/scene.json"
RUNTIME_CONFIG="${SIM2REAL}/config/config_debug.yaml"
LOG="${REFKL_DIR}/logs/v3_pipeline.log"
LOCK="${REFKL_DIR}/pipeline_v3.lock"

mkdir -p "${REFKL_DIR}/logs" "${REFKL_DIR}/pids" "${RENDER_DIR}" "${PREP_DIR}" "${REPO}/datasets"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "v3 RefKL pipeline already running (lock ${LOCK})" >&2
  exit 1
fi

log() { echo "[v3-refkl $(date -Is)] $*" | tee -a "${LOG}"; }

count_episodes() { find "${SRC_DEMOS}" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l; }

start_collection_if_needed() {
  local n
  n="$(count_episodes)"
  if (( n >= TARGET_EPISODES )); then
    log "collection already complete: ${n} episodes"
    return 0
  fi
  if grep -q '^DONE' "${SRC_DEMOS}/collect.log" 2>/dev/null; then
    log "collection marked DONE with ${n} episodes"
    return 0
  fi
  log "starting / resuming collection (${n}/${TARGET_EPISODES})"
  setsid bash "${SRC_DEMOS}/collect_until_100_v3.sh" >> "${SRC_DEMOS}/collect.log" 2>&1 &
  echo $! > "${REFKL_DIR}/pids/collect_v3.pid"
}

wait_for_collection() {
  local n
  log "waiting for v3 collection (target=${TARGET_EPISODES}) src=${SRC_DEMOS}"
  while true; do
    n="$(count_episodes)"
    if grep -q '^DONE' "${SRC_DEMOS}/collect.log" 2>/dev/null || [ "${n}" -ge "${TARGET_EPISODES}" ]; then
      break
    fi
    if grep -q '^ABORT' "${SRC_DEMOS}/collect.log" 2>/dev/null; then
      log "collection ABORTED with ${n} episodes"
      exit 1
    fi
    log "collection progress ${n}/${TARGET_EPISODES}"
    sleep 60
  done
  n="$(count_episodes)"
  log "collection complete: ${n} episodes"
  if (( n < 16 )); then
    echo "not enough episodes (${n}) to build train/val splits" >&2
    exit 1
  fi
}

wait_gpus_free() {
  local gpus="$1"
  local timeout_s="${2:-300}"
  local start elapsed used
  start="$(date +%s)"
  log "waiting for GPUs ${gpus} memory.used <= 2048 MiB"
  while true; do
    used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' -v g="${gpus}" '
      BEGIN { split(g, a, ","); for (i in a) keep[a[i]+0]=1 }
      { gsub(/ /, "", $1); gsub(/ /, "", $2); if ($1+0 in keep && $2+0 > max) max=$2+0 }
      END { print max+0 }
    ')"
    log "max used on ${gpus} = ${used} MiB"
    if (( used <= 2048 )); then
      return 0
    fi
    elapsed="$(( $(date +%s) - start ))"
    if (( elapsed >= timeout_s )); then
      log "GPUs still occupied after ${timeout_s}s (used=${used} MiB)"
      return 1
    fi
    sleep 10
  done
}

pick_sft_lora() {
  local best="" preferred latest
  for preferred in lora_000500 lora_000750 lora_000250 lora_001000; do
    if [[ -d "${SFT_RUN_DIR}/${preferred}" && -f "${SFT_RUN_DIR}/${preferred}/adapter_config.json" ]]; then
      best="${SFT_RUN_DIR}/${preferred}"
      break
    fi
  done
  if [[ -z "${best}" ]]; then
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

render_sharded() {
  local -a gpus
  IFS=',' read -ra gpus <<< "${RENDER_GPUS}"
  local nfiles nproc i start end gpu
  nfiles="$(ls -1 "${SRC_DEMOS}"/episode_*.npz | wc -l)"
  nproc="${#gpus[@]}"
  log "render ${nfiles} episodes x ${N_VARIANTS} variants on GPUs ${RENDER_GPUS} -> ${RENDER_DIR}"
  log "render config=${RUNTIME_CONFIG} key=${SCENE_KEY}"
  local -a pids=()
  for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    start=$(( i * nfiles / nproc ))
    end=$(( (i + 1) * nfiles / nproc ))
    if (( start >= end )); then
      continue
    fi
    log "render shard gpu=${gpu} episodes [${start}, ${end})"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PATH="${CONDA_BIN}:${PATH}"
      export PYTHONUNBUFFERED=1
      export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
      export PYTHONPATH="${REPO}/ManiSkill:${REPO}/SimplerEnv:${REPO}/real2sim:${REPO}/openvla:${REPO}/sim2real"
      cd "${SFT_DIR}"
      "${CONDA_BIN}/python" -u render_bg_variants.py \
        "${SRC_DEMOS}" "${RENDER_DIR}" \
        --variants "${N_VARIANTS}" --start "${start}" --end "${end}" --seed 0 \
        --config_path "${RUNTIME_CONFIG}" \
        --key "${SCENE_KEY}" \
        --scene "${SCENE_JSON}"
    ) >> "${REFKL_DIR}/logs/render_v3_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "${pids[-1]}" > "${REFKL_DIR}/pids/render_v3_gpu${gpu}.pid"
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      log "render pid ${pid} failed"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "one or more render shards failed; see ${REFKL_DIR}/logs/render_v3_gpu*.log" >&2
    exit 1
  fi
  local nout
  nout="$(ls -1 "${RENDER_DIR}"/episode_*_bg*.npz 2>/dev/null | wc -l)"
  log "render done: ${nout} npz"
  if (( nout < nfiles * N_VARIANTS )); then
    echo "expected $((nfiles * N_VARIANTS)) rendered npz, got ${nout}" >&2
    exit 1
  fi
}

build_tfds() {
  local n
  n="$(ls -1 "${PREP_DIR}"/episode_*.npz | wc -l)"
  log "tfds build sft_v2 from ${n} prepared episodes"
  (
    export PATH="${CONDA_BIN}:${PATH}"
    export CUDA_VISIBLE_DEVICES=""
    export RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}"
    export RLVLA_SFT_TOTAL_DEMOS="${n}"
    export RLVLA_SFT_VAL_DEMOS=16
    export PYTHONPATH="${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset${PYTHONPATH:+:$PYTHONPATH}"
    cd "${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset"
    "${CONDA_BIN}/tfds" build --overwrite --data_dir "${REPO}/datasets"
  ) >> "${LOG}" 2>&1
  if [[ ! -d "${REPO}/datasets/sft_v2/1.0.0" ]]; then
    echo "tfds did not produce ${REPO}/datasets/sft_v2/1.0.0" >&2
    exit 1
  fi
}

log "pipeline start src=${SRC_DEMOS} variants=${N_VARIANTS} scene=${SCENE_KEY}"
if [[ ! -f "${RUNTIME_CONFIG}" ]]; then
  echo "missing runtime config ${RUNTIME_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "missing scene json ${SCENE_JSON}" >&2
  exit 1
fi

if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  start_collection_if_needed
  wait_for_collection
else
  log "SKIP_COLLECT=1 have=$(count_episodes)"
fi

if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
  wait_gpus_free "${RENDER_GPUS}" 300
  render_sharded
else
  log "SKIP_RENDER=1"
fi

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  log "prepare_sft_v2_episodes ${RENDER_DIR} -> ${PREP_DIR}"
  "${CONDA_BIN}/python" -u "${SFT_DIR}/prepare_sft_v2_episodes.py" "${RENDER_DIR}" "${PREP_DIR}" >> "${LOG}" 2>&1
else
  log "SKIP_PREPARE=1"
fi

if [[ -f "${PREP_DIR}/sft_episode_stats.json" ]]; then
  EPISODE_LEN="${EPISODE_LEN:-$("${CONDA_BIN}/python" -c "import json; print(json.load(open('${PREP_DIR}/sft_episode_stats.json'))['recommended_episode_len'])")}"
  log "episode_len from stats: ${EPISODE_LEN}"
fi
export EPISODE_LEN="${EPISODE_LEN:-80}"

if [[ "${SKIP_TFDS:-0}" != "1" ]]; then
  build_tfds
else
  log "SKIP_TFDS=1"
fi

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  log "SFT OpenVLA-V2 (vision+LLM LoRA + projectors + lm_head) on GPU ${SFT_GPU}"
  wait_gpus_free "${SFT_GPU}" 120 || true
  CUDA_VISIBLE_DEVICES="${SFT_GPU}" \
    SFT_RUN_ROOT="${SFT_RUN_ROOT}" \
    DATA_ROOT_DIR="${SFT_DATA_ROOT_DIR:-${REFKL_DIR}/datasets}" \
    MAX_STEPS="${MAX_STEPS:-1000}" \
    BATCH_SIZE="${BATCH_SIZE:-8}" \
    bash "${SFT_DIR}/train_sft_v3.sh" >> "${REFKL_DIR}/logs/sft_v4_vision.log" 2>&1
else
  log "SKIP_SFT=1, using existing LoRA under ${SFT_RUN_DIR}"
fi

LORA_PATH="$(pick_sft_lora)"
log "selected SFT LoRA ${LORA_PATH}"
printf '%s\n' "${LORA_PATH}" > "${REFKL_DIR}/selected_sft_lora_v4_vision.txt"

if [[ "${SKIP_REFKL:-0}" != "1" ]]; then
  log "launch RefKL V4 vision"
  wait_gpus_free "${RENDER_GPUS}" 180
  SFT_LORA_PATH="${LORA_PATH}" \
    REFKL_LOAD_PATH="${LORA_PATH}" \
    EPISODE_LEN="${EPISODE_LEN}" \
    SFT_DATA_ROOT_DIR="${SFT_DATA_ROOT_DIR:-${REFKL_DIR}/datasets}" \
    NAME="${NAME:-RefKL_pick_red_cube_v4_vision}" \
    bash "${REFKL_DIR}/launch_gpus_v3.sh"
  log "RefKL V4 vision launched pid=$(cat "${REFKL_DIR}/pids/ddp_v4_vision.pid") log=${REFKL_DIR}/logs/ddp_sft_v4_vision.log"
else
  log "SKIP_REFKL=1"
fi

log "pipeline done $(date -Is)"
