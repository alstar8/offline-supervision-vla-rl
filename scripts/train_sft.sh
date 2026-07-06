#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/openvla"

CUDA="${CUDA:-0,1,2,3}"
TASK_NAME="${TASK_NAME:-sft}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_AT_7500="${SAVE_AT_7500:-1}"

SAVE_STEPS="0,2500,5000,7500,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000"
if [[ "${SAVE_AT_7500}" -eq 0 ]]; then
  SAVE_STEPS="0,2500,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${CUDA}" \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "gen-robot/openvla-7b-rlvla-warmup" \
  --data_root_dir "../datasets" \
  --dataset_name "${TASK_NAME}" \
  --run_root_dir "checkpoints/${TASK_NAME}" \
  --lora_rank 32 \
  --batch_size 20 \
  --max_steps "${MAX_STEPS}" \
  --eval_steps 200 \
  --save_steps "${SAVE_STEPS}" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project "offline-supervision-vla-rl"
