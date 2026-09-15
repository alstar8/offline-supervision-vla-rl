#!/usr/bin/env bash
# 8-GPU DataBC: SFT student + offline dataset BC, local-step BC schedule.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
LOAD_PATH="${DATABC_LOAD_PATH:?set DATABC_LOAD_PATH to a LoRA checkpoint dir}"
DATABC_NAME="${DATABC_NAME:-DataBC_pick_red_cube_8gpu}"
DATABC_GPUS="${DATABC_GPUS:-0,1,2,4,5,6,7}"
NUM_ENVS="${NUM_ENVS:-64}"
IFS=',' read -ra GPUS <<< "${DATABC_GPUS}"
NPROC="${#GPUS[@]}"
MASTER_PORT="${MASTER_PORT:-29531}"

if [[ ! -d "${LOAD_PATH}" ]]; then
  echo "missing checkpoint directory: ${LOAD_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LOAD_PATH}/dataset_statistics.json" ]]; then
  echo "missing dataset_statistics.json in ${LOAD_PATH}" >&2
  exit 1
fi

JOB_DIR="${RUN_DIR}/ddp"
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
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla"
export DATABC_GPUS
export DATABC_LOAD_STAGGER_S="${DATABC_LOAD_STAGGER_S:-20}"
export DATABC_NCCL_TIMEOUT_MIN="${DATABC_NCCL_TIMEOUT_MIN:-180}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10800}"
unset TORCH_NCCL_BLOCKING_WAIT || true

cd "${REPO}/SimplerEnv"

echo "DataBC DDP start $(date -Is) gpus=${DATABC_GPUS} nproc=${NPROC} load=${LOAD_PATH}"

exec "${CONDA_BIN}/torchrun" \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  "${RUN_DIR}/ddp_entry.py" \
  --name="${DATABC_NAME}" \
  --env_id=OpenReal2Sim-v0 \
  --vla_path=gen-robot/openvla-7b-rlvla-warmup \
  --vla_load_path="${LOAD_PATH}" \
  --vla_unnorm_key=sft \
  --resume_episode_offset=0 \
  --seed=0 \
  --num_envs="${NUM_ENVS}" \
  --episode_len=80 \
  --store-rollouts-on-cpu \
  --use_wrist_camera \
  --buffer_inferbatch="${BUFFER_INFERBATCH:-${NUM_ENVS}}" \
  --bc_to_ref_enabled \
  --no_sft_image_aug \
  --sft_data_root_dir=../datasets \
  --sft_dataset_name=sft \
  --sft_batch_size=8 \
  --sft_shuffle_buffer_size=2000 \
  --bc_to_ref_coef=0.6 \
  --bc_to_ref_hold_steps=100000 \
  --bc_to_ref_decay_steps=300000 \
  --interval_eval=5 \
  --interval_save=5 \
  --stop_success_rate=0.9 \
  --stop_success_windows=2 \
  --steps_max="${STEPS_MAX:-2000000}" \
  "$@"
