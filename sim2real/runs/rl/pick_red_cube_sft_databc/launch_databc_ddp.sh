#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
TRAIN="${RUN_DIR}/train_databc_ddp.sh"
DEFAULT_CKPT="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc/wandb/offline-run-20260911_223551-1707yrec/glob/steps_0309"
DATABC_LOAD_PATH="${DATABC_LOAD_PATH:-${DEFAULT_CKPT}}"
DATABC_GPUS="${DATABC_GPUS:-0,1,2,4,5,6,7}"
NUM_ENVS="${NUM_ENVS:-64}"
LOG="${RUN_DIR}/logs/ddp.log"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/pids"
echo "${DATABC_LOAD_PATH}" > "${RUN_DIR}/selected_databc_ddp_lora.txt"

echo "[databc $(date -u +%H:%M:%S)] launch DDP gpus=${DATABC_GPUS} num_envs=${NUM_ENVS} load=${DATABC_LOAD_PATH}"
echo "===== start $(date -Is) ddp gpus=${DATABC_GPUS} num_envs=${NUM_ENVS} load=${DATABC_LOAD_PATH} =====" >> "${LOG}"
setsid env \
  DATABC_GPUS="${DATABC_GPUS}" NUM_ENVS="${NUM_ENVS}" \
  DATABC_LOAD_PATH="${DATABC_LOAD_PATH}" \
  STEPS_MAX="${STEPS_MAX:-2000000}" \
  DATABC_NAME="DataBC_pick_red_cube_8gpu" \
  "${TRAIN}" >> "${LOG}" 2>&1 &
echo $! > "${RUN_DIR}/pids/ddp.pid"
echo "[databc $(date -u +%H:%M:%S)] launched DDP pid=$(cat "${RUN_DIR}/pids/ddp.pid") log=${LOG}"
