#!/usr/bin/env bash
# Deterministic OpenReal2Sim video eval for the best saved RefKL checkpoint.
# Default GPU is 3 (idle; training uses 0,1,2,4-7). GPU 3 PhysX CUDA hangs
# even at 1 env, so this script uses CPU physics + GPU render on GPU 3.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
CKPT_FILE="${RUN_DIR}/best_ckpt.txt"
CKPT="${CKPT:-$(grep -E '^/' "${CKPT_FILE}" | tail -n 1)}"
CUDA_ID="${CUDA_ID:-3}"
# GPU 3 PhysX CUDA hangs even at 1 env; CPU physics + GPU render is the workaround.
if [[ "${CUDA_ID}" == "3" ]]; then
  NUM_ENVS="${NUM_ENVS:-1}"
  export OPENREAL2SIM_SIM_BACKEND="${OPENREAL2SIM_SIM_BACKEND:-cpu}"
else
  NUM_ENVS="${NUM_ENVS:-8}"
  export OPENREAL2SIM_SIM_BACKEND="${OPENREAL2SIM_SIM_BACKEND:-gpu}"
fi
SEED="${SEED:-0}"
NAME="${NAME:-RefKL_eval_best_steps_0044}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/eval_sim_steps_0044}"

if [[ ! -d "${CKPT}" ]]; then
  echo "missing checkpoint directory: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}/adapter_model.safetensors" ]]; then
  echo "missing adapter_model.safetensors in ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}/dataset_statistics.json" ]]; then
  echo "missing dataset_statistics.json in ${CKPT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${RUN_DIR}/logs"

export PATH="${CONDA_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export HF_HOME="/workspace-SR008.nfs2/users/staroverov/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla"
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT || true
unset TORCH_NCCL_BLOCKING_WAIT || true

cd "${REPO}/SimplerEnv"

echo "RefKL sim eval start $(date -Is) gpu=${CUDA_ID} num_envs=${NUM_ENVS} ckpt=${CKPT} out=${OUT_DIR}"

exec "${CONDA_BIN}/python" -u simpler_env/train_ms3_ppo_sft.py \
  --name="${NAME}" \
  --env_id=OpenReal2Sim-v0 \
  --vla_path=gen-robot/openvla-7b-rlvla-warmup \
  --vla_load_path="${CKPT}" \
  --vla_unnorm_key=sft \
  --seed="${SEED}" \
  --num_envs="${NUM_ENVS}" \
  --episode_len=80 \
  --store-rollouts-on-cpu \
  --use_wrist_camera \
  --buffer_inferbatch="${NUM_ENVS}" \
  --max_ee_delta=0.05 \
  --success_terminate_steps=5 \
  --sticky_gripper_steps=0 \
  --reward_max_lift_height=0 \
  --reward_reach_coef=0.05 \
  --reward_reach_clip=0.25 \
  --reward_lift_coef=0.5 \
  --reward_lift_height=0.05 \
  --reward_yeet_height=0.15 \
  --reward_yeet_coef=2.0 \
  --reward_yeet_grasp_only \
  --no_wandb \
  --only_render \
  --render_info \
  "$@"
