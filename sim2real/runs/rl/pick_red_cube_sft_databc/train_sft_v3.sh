#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
MAX_STEPS="${MAX_STEPS:-1000}"
# V2 dual-cam + full projector/lm_head training OOMs at batch 16 on 80GB.
# Keep effective batch 16 via grad accumulation.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2_scene14sep26_vision_llm}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-../datasets}"
NPROC="${NPROC:-1}"
SAVE_STEPS="${SAVE_STEPS:-250,500,750,1000}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2000}"
SFT_EXTRA_ARGS=()
if [[ "${SKIP_IMAGE_RESIZE:-0}" == "1" ]]; then
  SFT_EXTRA_ARGS+=(--skip_image_resize True)
fi
if [[ -n "${ACTION_DIM_LOSS_WEIGHTS:-}" ]]; then
  SFT_EXTRA_ARGS+=(--action_dim_loss_weights "${ACTION_DIM_LOSS_WEIGHTS}")
fi

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

echo "SFT_V3 start $(date -Is) max_steps=${MAX_STEPS} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} nproc=${NPROC} run_root=${SFT_RUN_ROOT} data_root=${DATA_ROOT_DIR} lora_target=vision_llm skip_resize=${SKIP_IMAGE_RESIZE:-0} dim_weights=${ACTION_DIM_LOSS_WEIGHTS:-uniform}"
echo "SFT_V3 scheme: LoRA on vision backbone + LLM; projector + proprio_projector + lm_head fully trained"

"${CONDA_BIN}/torchrun" --standalone --nnodes 1 --nproc-per-node "${NPROC}" vla-scripts/finetune.py \
  --vla_path "gen-robot/openvla-7b-rlvla-warmup" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "sft_v2" \
  --run_root_dir "${SFT_RUN_ROOT}" \
  --vla_model_variant "v2" \
  --proprio_dim 7 \
  --num_images_in_input 2 \
  --lora_rank 32 \
  --lora_target vision_llm \
  --train_projector True \
  --train_action_head True \
  --batch_size "${BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --grad_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate 5e-4 \
  --image_aug False \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --wandb_project "offline-supervision-vla-rl" \
  "${SFT_EXTRA_ARGS[@]}"

echo "SFT_V3 done $(date -Is)"
