#!/usr/bin/env bash
# RefKL from scratch on 7 usable H100s (GPU 3 PhysX/Vulkan device-lost).
#
# Offline pretrain signal: load SFT LoRA + offline SFT dataset BC.
# Online tune: PPO + KL to a frozen copy of the same SFT teacher.
# BC/KL schedules are LOCAL env-steps (already how train_ms3_ppo_sft.py counts),
# so they stay equivalent to 1-GPU DataBC whether world_size is 7 or 8.
#
# 64 envs * 80 steps = 5120 local steps / update; steps_max=2e6 -> 390 updates.
# Hold 200k (~39 updates) then decay to min_coef=0.15 by 800k (~156 updates).
# The 7-GPU DataBC run slipped after BC decayed near 0; keep a BC floor.
# KL stays on for the full 2e6 local budget at paper coef 0.008.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_LORA_PATH="${SFT_LORA_PATH:-${REPO}/openvla/checkpoints/sft/steps_1000-no_aug/lora_000500}"
# Student LoRA: SFT for a fresh run, or a PPO checkpoint to resume.
LOAD_PATH="${REFKL_LOAD_PATH:-${SFT_LORA_PATH}}"
if [[ "${PPO_FROM_WARMUP:-0}" == "1" ]]; then
  LOAD_PATH=""
fi
NUM_ENVS="${NUM_ENVS:-64}"
SEED="${SEED:-0}"
NAME="${NAME:-RefKL_pick_red_cube_8gpu}"
VLA_MODEL_VARIANT="${VLA_MODEL_VARIANT:-v1}"
VLA_PROPRIO_DIM="${VLA_PROPRIO_DIM:-7}"
VLA_UNNORM_KEY="${VLA_UNNORM_KEY:-sft}"
SFT_DATASET_NAME="${SFT_DATASET_NAME:-sft}"
KL_UNNORM_KEY="${KL_TO_REF_UNNORM_KEY:-${VLA_UNNORM_KEY}}"
EPISODE_LEN="${EPISODE_LEN:-80}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
# GPU 3 PhysX/CUDA hangs during 64-env (and 32-env) init; skip unless overridden.
REFKL_GPUS="${REFKL_GPUS:-0,1,2,4,5,6,7}"
IFS=',' read -ra GPUS <<< "${REFKL_GPUS}"
NPROC="${#GPUS[@]}"
MASTER_PORT="${MASTER_PORT:-29561}"
# loss = policy + value + entropy_term.
# target>0: term = coef * (H - target). Positive coef pulls H down toward target.
# Default 0.08 toward 0.9 = the SFT policy's own entropy (H=0.91 at init), so the
# term is ~0 at start and only guards against drift, instead of pushing H up.
ENTROPY_COEF="${ALG_ENTROPY_COEF:-0.08}"
ENTROPY_TARGET="${ALG_ENTROPY_TARGET:-0.9}"
ENTROPY_OVERSHOOT="${ALG_ENTROPY_OVERSHOOT_COEF:-0.0}"
PPO_EPOCH="${ALG_PPO_EPOCH:-1}"
RESUME_EP="${REFKL_RESUME_EP:-0}"
# Dense lift while grasped, yeet penalty above 0.15 m, absorb after 5 consecutive
# successes (masks remaining OOD steps). Sticky-close is off so the policy can
# open/close at any step. Keep max_lift=0 so the +1 success bonus is not zeroed.
# Collapse fixes vs the k3n59ys6 run:
# - freeze_actor_updates=3: critic-only warmup before PPO touches the actor.
# - yeet gated on grasp and ramped 0->full over 100k local steps (~20 updates),
#   so the SFT lift-high habit is reshaped gradually instead of punished at step 0.
# - reach shaping (small -coef*gripper_obj_dist) keeps a dense approach signal so
#   the policy can recover if grasp rate dips.
YEET_GRASP_ONLY_FLAG=""
if [[ "${REWARD_YEET_GRASP_ONLY:-1}" == "1" ]]; then
  YEET_GRASP_ONLY_FLAG="--reward_yeet_grasp_only"
fi

KL_TO_REF_ENABLED="${KL_TO_REF_ENABLED:-1}"
KL_ARGS=()
if [[ "${KL_TO_REF_ENABLED}" == "1" ]]; then
  if [[ ! -d "${SFT_LORA_PATH}" ]]; then
    echo "missing SFT LoRA directory: ${SFT_LORA_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${SFT_LORA_PATH}/dataset_statistics.json" ]]; then
    echo "missing dataset_statistics.json in ${SFT_LORA_PATH}" >&2
    exit 1
  fi
  KL_ARGS+=(--kl_to_ref_enabled)
  KL_ARGS+=(--kl_to_ref_path="${SFT_LORA_PATH}")
  KL_ARGS+=(--kl_to_ref_unnorm_key="${KL_UNNORM_KEY}")
  KL_ARGS+=(--kl_to_ref_coef="${KL_TO_REF_COEF:-0.008}")
  KL_ARGS+=(--kl_to_ref_steps="${KL_TO_REF_STEPS:-2000000}")
fi
SFT_EXTRA_ARGS=()
if [[ "${SFT_SKIP_IMAGE_RESIZE:-0}" == "1" ]]; then
  SFT_EXTRA_ARGS+=(--sft_skip_image_resize)
fi
if [[ -n "${SFT_ACTION_DIM_WEIGHTS:-}" ]]; then
  SFT_EXTRA_ARGS+=(--sft_action_dim_weights="${SFT_ACTION_DIM_WEIGHTS}")
fi
LOAD_ARGS=()
if [[ -n "${LOAD_PATH}" ]]; then
  if [[ ! -d "${LOAD_PATH}" ]]; then
    echo "missing student checkpoint directory: ${LOAD_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${LOAD_PATH}/dataset_statistics.json" ]]; then
    echo "missing dataset_statistics.json in ${LOAD_PATH}" >&2
    exit 1
  fi
  LOAD_ARGS+=(--vla_load_path="${LOAD_PATH}")
fi
if [[ -n "${VLA_UNNORM_STATS_PATH:-}" ]]; then
  if [[ ! -f "${VLA_UNNORM_STATS_PATH}" ]]; then
    echo "missing unnorm stats file: ${VLA_UNNORM_STATS_PATH}" >&2
    exit 1
  fi
  SFT_EXTRA_ARGS+=(--vla_unnorm_stats_path="${VLA_UNNORM_STATS_PATH}")
elif [[ -z "${LOAD_PATH}" ]]; then
  echo "PPO-from-warmup requires VLA_UNNORM_STATS_PATH when REFKL_LOAD_PATH is empty" >&2
  exit 1
fi
BC_TO_REF_ENABLED="${BC_TO_REF_ENABLED:-1}"
BC_ARGS=()
if [[ "${BC_TO_REF_ENABLED}" == "1" ]]; then
  BC_ARGS+=(--bc_to_ref_enabled)
  BC_ARGS+=(--no_sft_image_aug)
  BC_ARGS+=(--sft_data_root_dir="${SFT_DATA_ROOT_DIR:-../datasets}")
  BC_ARGS+=(--sft_dataset_name="${SFT_DATASET_NAME}")
  BC_ARGS+=(--sft_batch_size="${SFT_BATCH_SIZE}")
  BC_ARGS+=(--sft_shuffle_buffer_size="${SFT_SHUFFLE_BUFFER_SIZE:-2000}")
  BC_ARGS+=(--bc_to_ref_coef="${BC_TO_REF_COEF:-0.6}")
  BC_ARGS+=(--bc_to_ref_hold_steps="${BC_TO_REF_HOLD_STEPS:-200000}")
  BC_ARGS+=(--bc_to_ref_decay_steps="${BC_TO_REF_DECAY_STEPS:-800000}")
  BC_ARGS+=(--bc_to_ref_min_coef="${BC_TO_REF_MIN_COEF:-0.15}")
fi

JOB_DIR="${REFKL_JOB_DIR:-${RUN_DIR}/ddp_sft}"
mkdir -p "${JOB_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/pids"

export PATH="${CONDA_BIN}:${PATH}"
export HF_HOME="/workspace-SR008.nfs2/users/staroverov/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="${JOB_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Keep RL on the same scene the SFT data was rendered from. release.v4 switched
# the env default to the metric copy (source rescaled by 1.0915); training the
# student there while its teacher and BC data come from the unscaled scene would
# put a 9% geometry mismatch between the two.
export OPENREAL2SIM_RL_SCENE_KEY="${OPENREAL2SIM_RL_SCENE_KEY:-airy_table_scene14sep26_left_image}"
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla"
export REFKL_GPUS
# train_ms3_ppo_sft.py reads DATABC_* for stagger / NCCL timeout.
export DATABC_LOAD_STAGGER_S="${REFKL_LOAD_STAGGER_S:-20}"
export DATABC_NCCL_TIMEOUT_MIN="${REFKL_NCCL_TIMEOUT_MIN:-180}"
export REFKL_LOAD_STAGGER_S="${DATABC_LOAD_STAGGER_S}"
export REFKL_NCCL_TIMEOUT_MIN="${DATABC_NCCL_TIMEOUT_MIN}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10800}"
unset TORCH_NCCL_BLOCKING_WAIT || true

cd "${REPO}/SimplerEnv"

echo "RefKL DDP start $(date -Is) gpus=${REFKL_GPUS} nproc=${NPROC} seed=${SEED} num_envs=${NUM_ENVS} variant=${VLA_MODEL_VARIANT} unnorm=${VLA_UNNORM_KEY} dataset=${SFT_DATASET_NAME} sft_data_root=${SFT_DATA_ROOT_DIR:-../datasets} student=${LOAD_PATH:-warmup} teacher=${SFT_LORA_PATH} kl_enabled=${KL_TO_REF_ENABLED} bc_enabled=${BC_TO_REF_ENABLED} entropy_coef=${ENTROPY_COEF} entropy_target=${ENTROPY_TARGET} entropy_overshoot=${ENTROPY_OVERSHOOT} resume_ep=${RESUME_EP}"

exec "${CONDA_BIN}/torchrun" \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  "${RUN_DIR}/ddp_entry.py" \
  --name="${NAME}" \
  --env_id=OpenReal2Sim-v0 \
  --vla_path=gen-robot/openvla-7b-rlvla-warmup \
  "${LOAD_ARGS[@]}" \
  --vla_unnorm_key="${VLA_UNNORM_KEY}" \
  --vla_model_variant="${VLA_MODEL_VARIANT}" \
  --vla_proprio_dim="${VLA_PROPRIO_DIM}" \
  --resume_episode_offset="${RESUME_EP}" \
  --seed="${SEED}" \
  --num_envs="${NUM_ENVS}" \
  --episode_len="${EPISODE_LEN}" \
  --store-rollouts-on-cpu \
  --use_wrist_camera \
  --buffer_inferbatch="${BUFFER_INFERBATCH:-${NUM_ENVS}}" \
  --vla_gradient_checkpointing \
  "${BC_ARGS[@]}" \
  "${KL_ARGS[@]}" \
  "${SFT_EXTRA_ARGS[@]}" \
  --alg_entropy_coef="${ENTROPY_COEF}" \
  --alg_entropy_target="${ENTROPY_TARGET}" \
  --alg_entropy_overshoot_coef="${ENTROPY_OVERSHOOT}" \
  --alg_ppo_epoch="${PPO_EPOCH}" \
  --alg_target_kl="${ALG_TARGET_KL:-0.0}" \
  --vla_lr="${VLA_LR:-1e-4}" \
  --freeze_actor_updates="${FREEZE_ACTOR_UPDATES:-3}" \
  --reward_max_lift_height=0 \
  --reward_reach_coef="${REWARD_REACH_COEF:-0.05}" \
  --reward_reach_clip="${REWARD_REACH_CLIP:-0.25}" \
  --reward_lift_coef="${REWARD_LIFT_COEF:-0.5}" \
  --reward_lift_height=0.05 \
  --reward_yeet_height="${REWARD_YEET_HEIGHT:-0.15}" \
  --reward_yeet_coef="${REWARD_YEET_COEF:-2.0}" \
  ${YEET_GRASP_ONLY_FLAG} \
  --reward_yeet_warmup_steps="${REWARD_YEET_WARMUP_STEPS:-100000}" \
  --success_terminate_steps="${SUCCESS_TERMINATE_STEPS:-5}" \
  --sticky_gripper_steps="${STICKY_GRIPPER_STEPS:-0}" \
  --max_ee_delta="${MAX_EE_DELTA:-0.05}" \
  --interval_eval=5 \
  --interval_save=5 \
  --stop_success_rate=0.9 \
  --stop_success_windows=2 \
  --steps_max="${STEPS_MAX:-2000000}" \
  "$@"
