#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${ENV_NAME:-rlvla-guided}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flash-attn-wheel)
      FLASH_ATTN_WHEEL="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./install_env.sh [--flash-attn-wheel /path/to/flash_attn.whl]

Creates conda env `rlvla-guided` from environment.yml and installs local packages.
Optionally installs flash-attn from a prebuilt wheel (recommended on Linux + CUDA 12.1).
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found on PATH" >&2
  exit 1
fi

cd "${ROOT_DIR}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; activating and updating editable installs."
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
else
  conda env create -f "${ENV_FILE}" -n "${ENV_NAME}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

pip install --no-deps typeguard==4.5.1 tyro==1.0.5
pip install -e ./ManiSkill -e ./openvla -e ./real2sim -e ./SimplerEnv

if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
  pip install "${FLASH_ATTN_WHEEL}"
fi

echo "Done. Activate with: conda activate ${ENV_NAME}"
