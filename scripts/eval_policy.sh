#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT_PATH="${CKPT_PATH:-gen-robot/openvla-7b-rlvla-warmup}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
VLA_LOAD_PATH="${VLA_LOAD_PATH:-}"
CUDA_ID="${CUDA_ID:-0}"
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-64}"

OOD_ENVS=(
  "PutOnPlateInScene25VisionImage-v1"
  "PutOnPlateInScene25VisionTexture03-v1"
  "PutOnPlateInScene25VisionTexture05-v1"
  "PutOnPlateInScene25VisionWhole03-v1"
  "PutOnPlateInScene25VisionWhole05-v1"
  "PutOnPlateInScene25Carrot-v1"
  "PutOnPlateInScene25Plate-v1"
  "PutOnPlateInScene25Instruct-v1"
  "PutOnPlateInScene25MultiCarrot-v1"
  "PutOnPlateInScene25MultiPlate-v1"
  "PutOnPlateInScene25Position-v1"
  "PutOnPlateInScene25EEPose-v1"
  "PutOnPlateInScene25PositionChangeTo-v1"
)

usage() {
  cat <<'EOF'
Usage: ./scripts/eval_policy.sh [--seed SEED] [--ind-only] [--ood-only] [extra train_ms3_ppo.py args]

Environment variables:
  CKPT_PATH         HuggingFace model id or local checkpoint root
  UNNORM_KEY        Action unnormalization key (bridge_orig or sft)
  VLA_LOAD_PATH     Optional LoRA adapter path
  CUDA_ID           GPU id (default: 0)
  BUFFER_INFERBATCH Inference batch size (default: 64; use 16 on 40GB GPUs)
EOF
}

SEEDS=(0 1 2)
RUN_IND=1
RUN_OOD=1
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEEDS=("$2")
      shift 2
      ;;
    --seed=*)
      SEEDS=("${1#*=}")
      shift
      ;;
    --ind-only)
      RUN_OOD=0
      shift
      ;;
    --ood-only)
      RUN_IND=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${REPO_ROOT}/SimplerEnv"

run_eval() {
  local env_id="$1"
  local seed="$2"
  local extra_args=()
  if [[ -n "${VLA_LOAD_PATH}" ]]; then
    extra_args+=(--vla_load_path="${VLA_LOAD_PATH}")
  fi

  echo "Evaluating env=${env_id} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python simpler_env/train_ms3_ppo.py \
    --vla_path="${CKPT_PATH}" \
    --vla_unnorm_key="${UNNORM_KEY}" \
    --env_id="${env_id}" \
    --seed="${seed}" \
    --buffer_inferbatch="${BUFFER_INFERBATCH}" \
    --no_wandb \
    --only_render \
    "${extra_args[@]}" \
    "${FORWARD_ARGS[@]}"
}

for seed in "${SEEDS[@]}"; do
  if [[ "${RUN_IND}" -eq 1 ]]; then
    run_eval "PutOnPlateInScene25Main-v3" "${seed}"
  fi
  if [[ "${RUN_OOD}" -eq 1 ]]; then
    for env_id in "${OOD_ENVS[@]}"; do
      run_eval "${env_id}" "${seed}"
    done
  fi
done

echo "Evaluation complete. Aggregate metrics with: python SimplerEnv/scripts/calc_statistics.py"
