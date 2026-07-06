#!/usr/bin/env bash
set -euo pipefail

# Upload the beta/IND/OOD table as a single W&B run using the existing CSV uploader.
# Required:
#   $1 = W&B project
# Optional:
#   $2 = W&B entity
#   $3 = run name (default: beta-ind-ood-table)
#   $4 = mode online|offline (default: online)

if [ $# -lt 1 ]; then
  echo "Usage: $0 <project> [entity] [run_name] [mode]" >&2
  exit 1
fi

PROJECT="$1"
ENTITY="${2:-}"
RUN_NAME="${3:-beta-ind-ood-table}"
MODE="${4:-online}"

ROOT_DIR="/workspace/rlvla_root/rlvla_mod"
CSV_PATH="${ROOT_DIR}/SimplerEnv/scripts/stats/beta_ind_ood_table.csv"
UPLOADER="${ROOT_DIR}/SimplerEnv/scripts/upload_wandb_csv_metrics.py"
PYTHON_BIN="/venv/rl/bin/python"

if [ ! -f "${CSV_PATH}" ]; then
  echo "Error: CSV not found: ${CSV_PATH}" >&2
  exit 1
fi

if [ ! -f "${UPLOADER}" ]; then
  echo "Error: uploader script not found: ${UPLOADER}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${UPLOADER}"
  --project "${PROJECT}"
  --csv "${CSV_PATH}"
  --single-run-name "${RUN_NAME}"
  --mode "${MODE}"
)

if [ -n "${ENTITY}" ]; then
  cmd+=(--entity "${ENTITY}")
fi

echo "Uploading ${CSV_PATH} to W&B..."
"${cmd[@]}"
