#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SFT_LORA_PATH="${1:-}"
DATABC_NAME="${DATABC_NAME:-DataBC_pick_red_cube_v2}"

if [[ -z "${SFT_LORA_PATH}" ]]; then
  echo "usage: train_databc_v2.sh /path/to/lora_XXXXXX_or_steps_NNNN" >&2
  exit 1
fi
if [[ ! -d "${SFT_LORA_PATH}" ]]; then
  echo "missing checkpoint directory: ${SFT_LORA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SFT_LORA_PATH}/dataset_statistics.json" ]]; then
  echo "missing dataset_statistics.json in ${SFT_LORA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SFT_LORA_PATH}/adapter_model.safetensors" && ! -f "${SFT_LORA_PATH}/adapter_model.bin" ]]; then
  echo "missing LoRA adapter weights in ${SFT_LORA_PATH}" >&2
  exit 1
fi

LOCK="${RUN_DIR}/pipeline.lock"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-2048}"
GPU_WAIT_SEC="${GPU_WAIT_SEC:-3600}"

wait_for_free_gpu() {
  local start used elapsed
  start="$(date +%s)"
  echo "waiting for GPU memory.used <= ${GPU_MAX_USED_MIB} MiB (timeout ${GPU_WAIT_SEC}s)"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "GPU memory.used=${used} MiB"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= GPU_MAX_USED_MIB )); then
      return 0
    fi
    elapsed="$(( $(date +%s) - start ))"
    if (( elapsed >= GPU_WAIT_SEC )); then
      echo "GPU still occupied after ${GPU_WAIT_SEC}s (used=${used} MiB)" >&2
      return 1
    fi
    sleep 15
  done
}

mkdir -p "${RUN_DIR}"
exec 9>"${LOCK}"
echo "acquiring ${LOCK}"
flock 9
wait_for_free_gpu

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

cd "${REPO}/SimplerEnv"

echo "DataBC_V2 start $(date -Is) name=${DATABC_NAME} vla_load_path=${SFT_LORA_PATH}"

"${CONDA_BIN}/python" simpler_env/train_ms3_ppo_sft.py \
  --name="${DATABC_NAME}" \
  --env_id=OpenReal2Sim-v0 \
  --vla_path=gen-robot/openvla-7b-rlvla-warmup \
  --vla_load_path="${SFT_LORA_PATH}" \
  --vla_unnorm_key=sft_v2 \
  --vla_model_variant=v2 \
  --vla_proprio_dim=7 \
  --seed=0 \
  --num_envs=64 \
  --episode_len="${EPISODE_LEN:-80}" \
  --store-rollouts-on-cpu \
  --use_wrist_camera \
  --bc_to_ref_enabled \
  --no_sft_image_aug \
  --sft_data_root_dir=../datasets \
  --sft_dataset_name=sft_v2 \
  --sft_batch_size=8 \
  --sft_shuffle_buffer_size=2000 \
  --bc_to_ref_coef="${BC_TO_REF_COEF:-0.6}" \
  --bc_to_ref_hold_steps="${BC_TO_REF_HOLD_STEPS:-200000}" \
  --bc_to_ref_decay_steps="${BC_TO_REF_DECAY_STEPS:-600000}" \
  --bc_to_ref_min_coef="${BC_TO_REF_MIN_COEF:-0.3}" \
  --alg_entropy_coef="${ALG_ENTROPY_COEF:-0.2}" \
  --alg_entropy_target="${ALG_ENTROPY_TARGET:-1.0}" \
  --vla_temperature="${VLA_TEMPERATURE:-1.0}" \
  --vla_temperature_final="${VLA_TEMPERATURE_FINAL:-0.6}" \
  --vla_temperature_anneal_steps="${VLA_TEMPERATURE_ANNEAL_STEPS:-200000}" \
  --eval_at_train_temperature \
  --reward_reach_coef="${REWARD_REACH_COEF:-0.3}" \
  --reward_reach_clip=0.4 \
  --reward_max_lift_height=0 \
  --reward_lift_coef="${REWARD_LIFT_COEF:-0.5}" \
  --reward_lift_height=0.05 \
  --reward_yeet_height="${REWARD_YEET_HEIGHT:-0.15}" \
  --reward_yeet_coef="${REWARD_YEET_COEF:-2.0}" \
  --success_terminate_steps="${SUCCESS_TERMINATE_STEPS:-5}" \
  --sticky_gripper_steps "${STICKY_GRIPPER_STEPS:-0}" \
  --max_ee_delta=0.05 \
  --interval_eval=5 \
  --interval_save=5 \
  --stop_success_rate=1.0 \
  --stop_success_windows=2 \
  --steps_max="${STEPS_MAX:-2000000}"

echo "DataBC_V2 done $(date -Is)"
