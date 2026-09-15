#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2}"

export PATH="${CONDA_BIN}:${PATH}"
export HF_HOME="/workspace-SR008.nfs2/users/staroverov/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="${RUN_DIR}"
export WANDB_PROJECT="offline-supervision-vla-rl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
# Use THIS repo's openvla (with OpenVLA_V2): the editable openvla install otherwise
# falls back to the stale rlvla_mod copy, which lacks the V2 classes.
export PYTHONPATH="${REPO}/openvla${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${RUN_DIR}" "${REPO}/openvla/checkpoints/sft"
cd "${REPO}/openvla"

echo "SFT_V2 start $(date -Is) max_steps=${MAX_STEPS} batch_size=${BATCH_SIZE} run_root=${SFT_RUN_ROOT}"

"${CONDA_BIN}/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "gen-robot/openvla-7b-rlvla-warmup" \
  --data_root_dir "../datasets" \
  --dataset_name "sft_v2" \
  --run_root_dir "${SFT_RUN_ROOT}" \
  --vla_model_variant "v2" \
  --proprio_dim 7 \
  --num_images_in_input 2 \
  --lora_rank 32 \
  --batch_size "${BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --eval_steps 250 \
  --save_steps "250,500,750,1000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --shuffle_buffer_size 2000 \
  --wandb_project "offline-supervision-vla-rl"

echo "SFT_V2 done $(date -Is)"
