#!/usr/bin/env bash
# Fresh RefKL run after table_z fix + vision-encoder LoRA.
# Reuses v4 TFDS (private copy). Retrains SFT, then launches DDP RefKL.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
REFKL_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2_scene14sep26_refkl_vision}"
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_${MAX_STEPS:-1000}-no_aug"
SFT_DATA_ROOT_DIR="${SFT_DATA_ROOT_DIR:-${REFKL_DIR}/datasets}"
SFT_GPU="${SFT_GPU:-0}"
REFKL_GPUS="${REFKL_GPUS:-0,1,2,4,5,6,7}"
EPISODE_LEN="${EPISODE_LEN:-112}"
LOG="${REFKL_DIR}/logs/v4_vision_pipeline.log"
LOCK="${REFKL_DIR}/pipeline_v4_vision.lock"

mkdir -p "${REFKL_DIR}/logs" "${REFKL_DIR}/pids"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "v4 vision RefKL pipeline already running (lock ${LOCK})" >&2
  exit 1
fi

log() { echo "[v4-vision $(date -Is)] $*" | tee -a "${LOG}"; }

wait_gpus_free() {
  local gpus="$1"
  local timeout_s="${2:-600}"
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

stop_old_refkl() {
  local pidfile="${REFKL_DIR}/pids/ddp_v3.pid"
  local pid=""
  if [[ -f "${pidfile}" ]]; then
    pid="$(cat "${pidfile}")"
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    log "stopping old RefKL torchrun pid=${pid}"
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    local i
    for i in $(seq 1 90); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        log "old RefKL exited"
        return 0
      fi
      sleep 2
    done
    log "old RefKL still alive; sending KILL"
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    sleep 5
  else
    log "no live old RefKL pid"
  fi
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

if [[ ! -d "${SFT_DATA_ROOT_DIR}/sft_v2/1.0.0" ]]; then
  echo "missing private TFDS ${SFT_DATA_ROOT_DIR}/sft_v2/1.0.0" >&2
  exit 1
fi

log "pipeline start table_z_fix=1 lora_target=vision_llm sft_root=${SFT_RUN_ROOT} data=${SFT_DATA_ROOT_DIR} episode_len=${EPISODE_LEN}"
stop_old_refkl
wait_gpus_free "${REFKL_GPUS}" 600

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  log "SFT vision+LLM LoRA on GPU ${SFT_GPU}"
  CUDA_VISIBLE_DEVICES="${SFT_GPU}" \
    SFT_RUN_ROOT="${SFT_RUN_ROOT}" \
    DATA_ROOT_DIR="${SFT_DATA_ROOT_DIR}" \
    MAX_STEPS="${MAX_STEPS:-1000}" \
    BATCH_SIZE="${BATCH_SIZE:-8}" \
    GRAD_ACCUM="${GRAD_ACCUM:-2}" \
    bash "${SFT_DIR}/train_sft_v3.sh" >> "${REFKL_DIR}/logs/sft_v4_vision.log" 2>&1
else
  log "SKIP_SFT=1"
fi

LORA_PATH="$(pick_sft_lora)"
log "selected SFT LoRA ${LORA_PATH}"
printf '%s\n' "${LORA_PATH}" > "${REFKL_DIR}/selected_sft_lora_v4_vision.txt"

if [[ "${SKIP_REFKL:-0}" != "1" ]]; then
  wait_gpus_free "${REFKL_GPUS}" 180
  SFT_LORA_PATH="${LORA_PATH}" \
    REFKL_LOAD_PATH="${LORA_PATH}" \
    EPISODE_LEN="${EPISODE_LEN}" \
    SFT_DATA_ROOT_DIR="${SFT_DATA_ROOT_DIR}" \
    NAME="${NAME:-RefKL_pick_red_cube_v4_vision}" \
    bash "${REFKL_DIR}/launch_gpus_v3.sh"
  log "RefKL launched pid=$(cat "${REFKL_DIR}/pids/ddp_v4_vision.pid") log=${REFKL_DIR}/logs/ddp_sft_v4_vision.log"
else
  log "SKIP_REFKL=1"
fi

log "pipeline done $(date -Is)"
