#!/usr/bin/env bash
# OpenVLA-V2 RefKL pipeline on the new empty3 scene:
#   100 collected demos x 10 360-bg variants = 1k trajectories
#   -> prepare SFT npz -> RLDS sft_v2 -> SFT LoRA -> RefKL DDP
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
REFKL_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SRC_DEMOS="${REPO}/sim2real/runs/manual/pick_red_cube_100_v2"
RENDER_DIR="${SFT_DIR}/sft_v2_episodes_bg"
PREP_DIR="${SFT_DIR}/sft_v2_episodes"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2}"
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_${MAX_STEPS:-1000}-no_aug"
RENDER_GPUS="${RENDER_GPUS:-0,1,2,4,5,6,7}"
SFT_GPU="${SFT_GPU:-0}"
N_VARIANTS="${N_VARIANTS:-10}"
LOG="${REFKL_DIR}/logs/v2_pipeline.log"
LOCK="${REFKL_DIR}/pipeline_v2.lock"

mkdir -p "${REFKL_DIR}/logs" "${REFKL_DIR}/pids" "${RENDER_DIR}" "${PREP_DIR}" "${REPO}/datasets"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "v2 pipeline already running (lock ${LOCK})" >&2
  exit 1
fi

log() { echo "[v2-pipeline $(date -Is)] $*" | tee -a "${LOG}"; }

stop_v1_refkl() {
  local pid_file="${REFKL_DIR}/pids/ddp.pid"
  local pid=""
  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -d '[:space:]' < "${pid_file}")"
  fi
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    log "no live V1 RefKL pid"
    return 0
  fi
  log "stopping collapsed V1 RefKL pid=${pid}"
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  local waited=0
  while kill -0 "${pid}" 2>/dev/null && (( waited < 180 )); do
    sleep 5
    waited=$((waited + 5))
  done
  if kill -0 "${pid}" 2>/dev/null; then
    log "V1 still alive after 180s, sending KILL"
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    sleep 5
  fi
  log "V1 RefKL stopped"
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
  local nfiles nproc i start end gpu pid
  nfiles="$(ls -1 "${SRC_DEMOS}"/episode_*.npz | wc -l)"
  nproc="${#gpus[@]}"
  log "render ${nfiles} episodes x ${N_VARIANTS} variants on GPUs ${RENDER_GPUS} -> ${RENDER_DIR}"
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
        --variants "${N_VARIANTS}" --start "${start}" --end "${end}" --seed 0
    ) >> "${REFKL_DIR}/logs/render_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "${pids[-1]}" > "${REFKL_DIR}/pids/render_gpu${gpu}.pid"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      log "render pid ${pid} failed"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "one or more render shards failed; see ${REFKL_DIR}/logs/render_gpu*.log" >&2
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

log "pipeline start src=${SRC_DEMOS} variants=${N_VARIANTS}"
if [[ "${STOP_V1:-1}" == "1" ]]; then
  stop_v1_refkl
  wait_gpus_free "${RENDER_GPUS}" 300
fi

if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
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
  log "SFT OpenVLA-V2 on GPU ${SFT_GPU}"
  wait_gpus_free "${SFT_GPU}" 120 || true
  CUDA_VISIBLE_DEVICES="${SFT_GPU}" \
    SFT_RUN_ROOT="${SFT_RUN_ROOT}" \
    MAX_STEPS="${MAX_STEPS:-1000}" \
    BATCH_SIZE="${BATCH_SIZE:-16}" \
    bash "${SFT_DIR}/train_sft_v2.sh" >> "${REFKL_DIR}/logs/sft_v2.log" 2>&1
else
  log "SKIP_SFT=1, using existing LoRA under ${SFT_RUN_DIR}"
fi

LORA_PATH="$(pick_sft_lora)"
log "selected SFT LoRA ${LORA_PATH}"
printf '%s\n' "${LORA_PATH}" > "${REFKL_DIR}/selected_sft_lora_v2.txt"

if [[ "${SKIP_REFKL:-0}" != "1" ]]; then
  log "launch RefKL V2"
  wait_gpus_free "${RENDER_GPUS}" 180
  SFT_LORA_PATH="${LORA_PATH}" \
    REFKL_LOAD_PATH="${LORA_PATH}" \
    EPISODE_LEN="${EPISODE_LEN}" \
    bash "${REFKL_DIR}/launch_gpus_v2.sh"
  log "RefKL V2 launched pid=$(cat "${REFKL_DIR}/pids/ddp_v2.pid") log=${REFKL_DIR}/logs/ddp_sft_v2.log"
else
  log "SKIP_REFKL=1"
fi

log "pipeline done $(date -Is)"
