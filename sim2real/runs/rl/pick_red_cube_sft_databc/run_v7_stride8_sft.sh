#!/usr/bin/env bash
# v7 = v6 episodes re-chunked to stride 8. RUN THIS ON server8gpu_CLOUD.
#
# Rationale: every checkpoint so far stalls ~0.17-0.25 m short of the cube. Stride-4 labels
# average 0.010 m of translation per step, so covering the ~0.40 m approach needs ~40 steps
# of perfect heading before the gripper is even close. Stride-8 labels average 0.019 m
# (max clamped at 0.040 m), which halves the horizon: recommended_episode_len drops from
# 120-144 to 80.
#
# No re-render was needed. rechunk_stride8.py merges adjacent stride-4 chunks, which is
# exact because a chunk label is a *sum* of raw deltas with the observation taken from the
# chunk's first frame. See that script's docstring for the one lossy case.
#
# The TFDS build is CPU-only and is normally already done by the single-GPU host; this
# script rebuilds it only if it is missing.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
DATABC="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
REFKL="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"

PREP_DIR="${PREP_DIR:-${DATABC}/sft_v7_stride8_proprio}"
DATASETS="${DATASETS:-${REFKL}/datasets_v7}"
SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2_stride8}"

VARIANTS="${VARIANTS:-3}"
TARGET_UNIQUE="${TARGET_UNIQUE:-1000}"
TARGET_TRAJ=$((TARGET_UNIQUE * VARIANTS))
VAL_DEMOS=$((16 * VARIANTS))

SFT_GPUS="${SFT_GPUS:-0,1,2,3,4,5}"
SFT_NPROC="${SFT_NPROC:-6}"
MAX_STEPS="${MAX_STEPS:-8000}"
SAVE_STEPS="${SAVE_STEPS:-1000,2000,3000,4000,5000,6000,7000,8000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
# Same weights as v6: x/y dominate whether the gripper lands on the cube, and the rotation
# dims get a small nonzero weight so they are not untrained noise between position tokens.
DIM_WEIGHTS="${DIM_WEIGHTS:-3:3:1.5:0.2:0.2:0.2:1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-16000}"
LOG="${REFKL}/logs/pipeline_v7.log"

mkdir -p "${REFKL}/logs" "${REFKL}/pids" "${DATASETS}"
log() { echo "[v7] $* $(date -Is)" | tee -a "${LOG}"; }

n_prep="$(find "${PREP_DIR}" -maxdepth 1 -name 'episode_*_bg*.npz' | wc -l)"
log "prepared=${n_prep}/${TARGET_TRAJ} prep=${PREP_DIR}"
if [[ "${n_prep}" -lt "${TARGET_TRAJ}" ]]; then
  log "ERROR: re-chunk incomplete; run rechunk_stride8.py first"
  exit 1
fi

EPISODE_LEN=80
if [[ -f "${PREP_DIR}/sft_episode_stats.json" ]]; then
  EPISODE_LEN="$(${PY} -c 'import json;print(json.load(open("'"${PREP_DIR}"'/sft_episode_stats.json"))["recommended_episode_len"])')"
fi
log "recommended episode_len=${EPISODE_LEN} (use this for the RL sweep, not 120)"
printf '%s\n' "${EPISODE_LEN}" > "${REFKL}/episode_len_v7.txt"

# ---------------------------------------------------------------- tfds
if [[ ! -d "${DATASETS}/sft_v2/1.0.0" ]]; then
  log "tfds build -> ${DATASETS} (${TARGET_TRAJ} trajs, val=${VAL_DEMOS})"
  (
    export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:${PATH}"
    export CUDA_VISIBLE_DEVICES=""
    export RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}"
    export RLVLA_SFT_TOTAL_DEMOS="${TARGET_TRAJ}"
    export RLVLA_SFT_VAL_DEMOS="${VAL_DEMOS}"
    # No-op chunks were already dropped at re-chunk time.
    export RLVLA_SFT_DISABLE_FILTER=1
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export PYTHONUNBUFFERED=1
    cd "${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset"
    tfds build --overwrite --data_dir "${DATASETS}"
  ) 2>&1 | tee -a "${LOG}"
else
  log "tfds already built at ${DATASETS}/sft_v2/1.0.0"
fi

# ---------------------------------------------------------------- sft
if [[ "${SKIP_SFT:-0}" == "1" ]]; then
  log "SKIP_SFT=1: dataset is ready at ${DATASETS}, SFT left for server8gpu_CLOUD"
  exit 0
fi
log "SFT nproc=${SFT_NPROC} gpus=${SFT_GPUS} max_steps=${MAX_STEPS} run_root=${SFT_RUN_ROOT}"
SFT_RUN_ROOT="${SFT_RUN_ROOT}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
EVAL_STEPS="${EVAL_STEPS}" \
NPROC="${SFT_NPROC}" \
CUDA_VISIBLE_DEVICES="${SFT_GPUS}" \
DATA_ROOT_DIR="${DATASETS}" \
SKIP_IMAGE_RESIZE=1 \
ACTION_DIM_LOSS_WEIGHTS="${DIM_WEIGHTS}" \
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE}" \
  bash "${DATABC}/train_sft_v3.sh" 2>&1 | tee -a "${REFKL}/logs/sft_v7.log"

log "done. next: on the 1-GPU host run"
log "  STAGES=sweep_v6 SFT_V6_ROOT=${SFT_RUN_ROOT} SFT_MAX_STEPS=${MAX_STEPS} EPISODE_LEN=${EPISODE_LEN} ./run_queue.sh"
