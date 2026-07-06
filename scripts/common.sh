#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIMPLER_ENV_DIR="${REPO_ROOT}/SimplerEnv"

ENV_ID="${ENV_ID:-PutOnPlateInScene25Main-v3}"
VLA_PATH="${VLA_PATH:-gen-robot/openvla-7b-rlvla-warmup}"
VLA_UNNORM_KEY="${VLA_UNNORM_KEY:-bridge_orig}"
SEED="${SEED:-0}"
SFT_LORA_PATH="${SFT_LORA_PATH:-../openvla/checkpoints/sft/steps_60000-no_aug/lora_007500}"

parse_seed_args() {
  SEED=0
  FORWARD_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seed)
        SEED="$2"
        shift 2
        ;;
      --seed=*)
        SEED="${1#*=}"
        shift
        ;;
      *)
        FORWARD_ARGS+=("$1")
        shift
        ;;
    esac
  done
}
