#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SFT_LORA_PATH="${1:-}"

if [[ -z "${SFT_LORA_PATH}" ]]; then
  echo "usage: train_databc.sh /path/to/lora_XXXXXX" >&2
  exit 1
fi
if [[ ! -d "${SFT_LORA_PATH}" ]]; then
  echo "missing SFT LoRA directory: ${SFT_LORA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SFT_LORA_PATH}/dataset_statistics.json" ]]; then
  echo "missing dataset_statistics.json in ${SFT_LORA_PATH}" >&2
  exit 1
fi

export PATH="${CONDA_BIN}:${PATH}"
export HF_HOME="/workspace-SR008.nfs2/users/staroverov/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="${RUN_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Do not inherit PYTHONPATH: a leading sim2real/ entry shadows ManiSkill and drops RC5.
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla"

mkdir -p "${RUN_DIR}"
cd "${REPO}/SimplerEnv"

echo "DataBC start $(date -Is) vla_load_path=${SFT_LORA_PATH}"

"${CONDA_BIN}/python" simpler_env/train_ms3_ppo_sft.py \
  --name="DataBC_pick_red_cube" \
  --env_id=OpenReal2Sim-v0 \
  --vla_path=gen-robot/openvla-7b-rlvla-warmup \
  --vla_load_path="${SFT_LORA_PATH}" \
  --vla_unnorm_key=sft \
  --seed=0 \
  --num_envs=64 \
  --episode_len=80 \
  --store-rollouts-on-cpu \
  --use_wrist_camera \
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
  --stop_success_rate=1.0 \
  --stop_success_windows=2 \
  --steps_max=2000000

echo "DataBC done $(date -Is)"
