#!/usr/bin/env bash
# Launch RefKL (offline SFT BC + PPO + KL to frozen SFT).
# GPUs 0,1,2,4-7 (GPU 3 PhysX hangs). All ranks share LoRA gradients.
# Resume: REFKL_LOAD_PATH=<ckpt dir> REFKL_RESUME_EP=<next episode index>
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
TRAIN="${RUN_DIR}/train_refkl.sh"
SFT_LORA_PATH="${SFT_LORA_PATH:-${REPO}/openvla/checkpoints/sft/steps_1000-no_aug/lora_000500}"
REFKL_LOAD_PATH="${REFKL_LOAD_PATH:-${SFT_LORA_PATH}}"
REFKL_GPUS="${REFKL_GPUS:-0,1,2,4,5,6,7}"
NUM_ENVS="${NUM_ENVS:-64}"
LOG="${RUN_DIR}/logs/ddp_sft.log"
MASTER_PORT="${MASTER_PORT:-29591}"
ALG_ENTROPY_COEF="${ALG_ENTROPY_COEF:-0.08}"
ALG_ENTROPY_TARGET="${ALG_ENTROPY_TARGET:-0.9}"
REFKL_RESUME_EP="${REFKL_RESUME_EP:-0}"

echo "${REFKL_LOAD_PATH}" > "${RUN_DIR}/selected_sft_lora.txt"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/pids"

echo "[refkl $(date -u +%H:%M:%S)] launch DDP gpus=${REFKL_GPUS} num_envs=${NUM_ENVS} student=${REFKL_LOAD_PATH} entropy_coef=${ALG_ENTROPY_COEF} entropy_target=${ALG_ENTROPY_TARGET} resume_ep=${REFKL_RESUME_EP} sticky=${STICKY_GRIPPER_STEPS:-0} terminate=${SUCCESS_TERMINATE_STEPS:-5} freeze_actor=${FREEZE_ACTOR_UPDATES:-3} yeet_warmup=${REWARD_YEET_WARMUP_STEPS:-100000} reach=${REWARD_REACH_COEF:-0.05} steps_max=${STEPS_MAX:-2000000}"
echo "===== start $(date -Is) ddp_sft gpus=${REFKL_GPUS} num_envs=${NUM_ENVS} student=${REFKL_LOAD_PATH} entropy_coef=${ALG_ENTROPY_COEF} entropy_target=${ALG_ENTROPY_TARGET} resume_ep=${REFKL_RESUME_EP} sticky=${STICKY_GRIPPER_STEPS:-0} terminate=${SUCCESS_TERMINATE_STEPS:-5} freeze_actor=${FREEZE_ACTOR_UPDATES:-3} yeet_warmup=${REWARD_YEET_WARMUP_STEPS:-100000} reach=${REWARD_REACH_COEF:-0.05} steps_max=${STEPS_MAX:-2000000} =====" >> "${LOG}"
setsid env \
  REFKL_GPUS="${REFKL_GPUS}" NUM_ENVS="${NUM_ENVS}" \
  SFT_LORA_PATH="${SFT_LORA_PATH}" \
  REFKL_LOAD_PATH="${REFKL_LOAD_PATH}" \
  REFKL_RESUME_EP="${REFKL_RESUME_EP}" \
  ALG_ENTROPY_COEF="${ALG_ENTROPY_COEF}" \
  ALG_ENTROPY_TARGET="${ALG_ENTROPY_TARGET}" \
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
  NAME="RefKL_pick_red_cube_8gpu" \
  MASTER_PORT="${MASTER_PORT}" \
  "${TRAIN}" >> "${LOG}" 2>&1 &
echo $! > "${RUN_DIR}/pids/ddp.pid"
echo "[refkl $(date -u +%H:%M:%S)] launched DDP pid=$(cat "${RUN_DIR}/pids/ddp.pid") log=${LOG}"
