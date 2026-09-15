#!/usr/bin/env bash
# Launch OpenVLA-V2 RefKL on airy_table_scene14sep26_left_image.
# Vision+LLM LoRA student/teacher. GPUs 0,1,2,4-7 (GPU 3 PhysX hangs).
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
TRAIN="${RUN_DIR}/train_refkl.sh"
SFT_LORA_PATH="${SFT_LORA_PATH:-${REPO}/openvla/checkpoints/sft/openvla_v2_scene14sep26_refkl_vision/steps_1000-no_aug/lora_000500}"
REFKL_LOAD_PATH="${REFKL_LOAD_PATH:-${SFT_LORA_PATH}}"
REFKL_GPUS="${REFKL_GPUS:-0,1,2,4,5,6,7}"
NUM_ENVS="${NUM_ENVS:-64}"
# Dual-camera V2 generate() OOMs at inferbatch=64 on 80GB; chunk like V1's 32-env budget.
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-8}"
LOG="${LOG:-${RUN_DIR}/logs/ddp_sft_v4_vision.log}"
PID_FILE="${PID_FILE:-${RUN_DIR}/pids/ddp_v4_vision.pid}"
MASTER_PORT="${MASTER_PORT:-29661}"
ALG_ENTROPY_COEF="${ALG_ENTROPY_COEF:-0.2}"
ALG_ENTROPY_TARGET="${ALG_ENTROPY_TARGET:-0.9}"
ALG_ENTROPY_OVERSHOOT_COEF="${ALG_ENTROPY_OVERSHOOT_COEF:-0.3}"
ALG_PPO_EPOCH="${ALG_PPO_EPOCH:-3}"
REFKL_RESUME_EP="${REFKL_RESUME_EP:-0}"
REFKL_JOB_DIR="${REFKL_JOB_DIR:-${RUN_DIR}/ddp_sft_v4_vision}"

echo "${SFT_LORA_PATH}" > "${RUN_DIR}/selected_sft_lora_v4_vision.txt"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/pids" "${REFKL_JOB_DIR}"

echo "[refkl-v4-vision $(date -u +%H:%M:%S)] launch DDP gpus=${REFKL_GPUS} num_envs=${NUM_ENVS} inferbatch=${BUFFER_INFERBATCH} student=${REFKL_LOAD_PATH} entropy_coef=${ALG_ENTROPY_COEF} entropy_target=${ALG_ENTROPY_TARGET} overshoot=${ALG_ENTROPY_OVERSHOOT_COEF} resume_ep=${REFKL_RESUME_EP}"
echo "===== start $(date -Is) ddp_sft_v4_vision gpus=${REFKL_GPUS} num_envs=${NUM_ENVS} inferbatch=${BUFFER_INFERBATCH} student=${REFKL_LOAD_PATH} =====" >> "${LOG}"
setsid env \
  REFKL_GPUS="${REFKL_GPUS}" NUM_ENVS="${NUM_ENVS}" \
  BUFFER_INFERBATCH="${BUFFER_INFERBATCH}" \
  SFT_LORA_PATH="${SFT_LORA_PATH}" \
  REFKL_LOAD_PATH="${REFKL_LOAD_PATH}" \
  REFKL_RESUME_EP="${REFKL_RESUME_EP}" \
  REFKL_JOB_DIR="${REFKL_JOB_DIR}" \
  VLA_MODEL_VARIANT=v2 \
  VLA_PROPRIO_DIM=7 \
  VLA_UNNORM_KEY=sft_v2 \
  SFT_DATASET_NAME=sft_v2 \
  KL_TO_REF_UNNORM_KEY=sft_v2 \
  EPISODE_LEN="${EPISODE_LEN:-112}" \
  SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}" \
  SFT_DATA_ROOT_DIR="${SFT_DATA_ROOT_DIR:-${RUN_DIR}/datasets}" \
  ALG_ENTROPY_COEF="${ALG_ENTROPY_COEF}" \
  ALG_ENTROPY_TARGET="${ALG_ENTROPY_TARGET}" \
  ALG_ENTROPY_OVERSHOOT_COEF="${ALG_ENTROPY_OVERSHOOT_COEF}" \
  ALG_PPO_EPOCH="${ALG_PPO_EPOCH}" \
  BC_TO_REF_COEF="${BC_TO_REF_COEF:-0.6}" \
  BC_TO_REF_HOLD_STEPS="${BC_TO_REF_HOLD_STEPS:-200000}" \
  BC_TO_REF_DECAY_STEPS="${BC_TO_REF_DECAY_STEPS:-600000}" \
  BC_TO_REF_MIN_COEF="${BC_TO_REF_MIN_COEF:-0.3}" \
  STEPS_MAX="${STEPS_MAX:-2000000}" \
  FREEZE_ACTOR_UPDATES="${FREEZE_ACTOR_UPDATES:-3}" \
  REWARD_REACH_COEF="${REWARD_REACH_COEF:-0.05}" \
  REWARD_REACH_CLIP="${REWARD_REACH_CLIP:-0.25}" \
  REWARD_YEET_WARMUP_STEPS="${REWARD_YEET_WARMUP_STEPS:-100000}" \
  REWARD_YEET_GRASP_ONLY="${REWARD_YEET_GRASP_ONLY:-1}" \
  REWARD_LIFT_COEF="${REWARD_LIFT_COEF:-0.5}" \
  REWARD_YEET_HEIGHT="${REWARD_YEET_HEIGHT:-0.15}" \
  REWARD_YEET_COEF="${REWARD_YEET_COEF:-2.0}" \
  SUCCESS_TERMINATE_STEPS="${SUCCESS_TERMINATE_STEPS:-5}" \
  STICKY_GRIPPER_STEPS="${STICKY_GRIPPER_STEPS:-0}" \
  MAX_EE_DELTA="${MAX_EE_DELTA:-0.05}" \
  NAME="${NAME:-RefKL_pick_red_cube_v4_vision}" \
  MASTER_PORT="${MASTER_PORT}" \
  KL_TO_REF_ENABLED="${KL_TO_REF_ENABLED:-1}" \
  KL_TO_REF_COEF="${KL_TO_REF_COEF:-0.008}" \
  SFT_SKIP_IMAGE_RESIZE="${SFT_SKIP_IMAGE_RESIZE:-0}" \
  SFT_ACTION_DIM_WEIGHTS="${SFT_ACTION_DIM_WEIGHTS:-}" \
  SFT_SHUFFLE_BUFFER_SIZE="${SFT_SHUFFLE_BUFFER_SIZE:-2000}" \
  "${TRAIN}" >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[refkl-v4-vision $(date -u +%H:%M:%S)] launched DDP pid=$(cat "${PID_FILE}") log=${LOG}"
