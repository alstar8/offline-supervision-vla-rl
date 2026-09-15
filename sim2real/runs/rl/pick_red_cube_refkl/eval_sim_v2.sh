#!/usr/bin/env bash
# One OpenReal2Sim eval video for the best saved OpenVLA-V2 RefKL checkpoint.
# Peak saved V2 ckpt is steps_0004 (10.9% SR / 25% grasp). GPU 3 PhysX/Vulkan
# hangs, so default to GPU 7 (training already holds ~52GB; 1-env eval fits).
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
CKPT="${CKPT:-${RUN_DIR}/ddp_sft_v3/wandb/offline-run-20260915_003556-3wx7ygql/glob/steps_0004}"
CUDA_ID="${CUDA_ID:-7}"
NUM_ENVS="${NUM_ENVS:-1}"
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-${NUM_ENVS}}"
SEED="${SEED:-0}"
NAME="${NAME:-RefKL_eval_v2_best_steps_0004}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/eval_sim_v2_steps_0004}"
EPISODE_LEN="${EPISODE_LEN:-112}"

if [[ "${CUDA_ID}" == "3" ]]; then
  export OPENREAL2SIM_SIM_BACKEND="${OPENREAL2SIM_SIM_BACKEND:-cpu}"
else
  export OPENREAL2SIM_SIM_BACKEND="${OPENREAL2SIM_SIM_BACKEND:-gpu}"
fi

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

echo "RefKL V2 sim eval start $(date -Is) gpu=${CUDA_ID} backend=${OPENREAL2SIM_SIM_BACKEND} num_envs=${NUM_ENVS} inferbatch=${BUFFER_INFERBATCH} seed=${SEED} ckpt=${CKPT} out=${OUT_DIR}"

"${CONDA_BIN}/python" -u simpler_env/train_ms3_ppo_sft.py \
  --name="${NAME}" \
  --env_id=OpenReal2Sim-v0 \
  --vla_path=gen-robot/openvla-7b-rlvla-warmup \
  --vla_load_path="${CKPT}" \
  --vla_unnorm_key=sft_v2 \
  --vla_model_variant=v2 \
  --vla_proprio_dim=7 \
  --seed="${SEED}" \
  --num_envs="${NUM_ENVS}" \
  --episode_len="${EPISODE_LEN}" \
  --store-rollouts-on-cpu \
  --use_wrist_camera \
  --buffer_inferbatch="${BUFFER_INFERBATCH}" \
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

mapfile -t VIDEOS < <(find "${OUT_DIR}/wandb" -path '*glob/vis_0_train/*.mp4' -type f 2>/dev/null | sort)
if [[ "${#VIDEOS[@]}" -eq 0 ]]; then
  echo "eval finished but no vis_0_train mp4 under ${OUT_DIR}/wandb" >&2
  exit 1
fi
VID_DIR="${OUT_DIR}/videos"
mkdir -p "${VID_DIR}"
for VIDEO in "${VIDEOS[@]}"; do
  cp -f "${VIDEO}" "${VID_DIR}/$(basename "${VIDEO}")"
  echo "eval video: ${VID_DIR}/$(basename "${VIDEO}") (from ${VIDEO})"
done
cp -f "${VIDEOS[0]}" "${OUT_DIR}/eval_video.mp4"
