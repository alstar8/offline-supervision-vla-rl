#!/usr/bin/env bash
# v6: re-render with fixed-stride action labels, then SFT for ~3 epochs.
#
# Why this differs from v5:
#   - 3 background variants instead of 10. The 10x only added visual diversity
#     (still 1000 unique demos) while making one epoch 10x more expensive, so v5
#     never completed a single epoch in its 16k steps.
#   - action labels come from chunk_fixed_stride: chunk length no longer depends
#     on future sign flips, and near-stationary chunks are dropped.
#   - rotation dims get a small nonzero loss weight. They are constant in this
#     data, but at weight 0 they were untrained noise sitting between the
#     position tokens and the gripper token in the autoregressive sequence.
#   - prepared npz are KEPT so the dataset can be re-chunked or TFDS rebuilt
#     without paying for another render.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
SIM2REAL="${REPO}/sim2real"
DATABC="${SIM2REAL}/runs/rl/pick_red_cube_sft_databc"
REFKL="${SIM2REAL}/runs/rl/pick_red_cube_refkl"
COLLECT_DIR="${SIM2REAL}/runs/manual/pick_red_cube_100_v4"
PREP_DIR="${PREP_DIR:-${DATABC}/sft_v6_proprio}"
DATASETS="${DATASETS:-${REFKL}/datasets_v6}"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"
LOG="${REFKL}/logs/pipeline_v6.log"
LOCK="${REFKL}/pipeline_v6.lock"

SFT_RUN_ROOT="${SFT_RUN_ROOT:-checkpoints/sft/openvla_v2_3var_fixedlabels}"
VARIANTS="${VARIANTS:-3}"
TARGET_UNIQUE="${TARGET_UNIQUE:-1000}"
TARGET_TRAJ=$((TARGET_UNIQUE * VARIANTS))
# 16 held-out unique trajectories, same ones as v5, so val stays comparable.
VAL_DEMOS=$((16 * VARIANTS))
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
RENDER_GPUS="${RENDER_GPUS:-0,1,2,4,5,6,7}"   # GPU 3 hangs on PhysX/Vulkan

SFT_GPUS="${SFT_GPUS:-0,1,2,3,4,5}"           # torch only, GPU 3 is fine here
SFT_NPROC="${SFT_NPROC:-6}"
MAX_STEPS="${MAX_STEPS:-8000}"
SAVE_STEPS="${SAVE_STEPS:-1000,2000,3000,4000,5000,6000,7000,8000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
DIM_WEIGHTS="${DIM_WEIGHTS:-3:3:1.5:0.2:0.2:0.2:1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-16000}"
WATCH_GPUS="${WATCH_GPUS:-6 7}"
WATCH_NENVS="${WATCH_NENVS:-8 8}"
SKIP_RENDER="${SKIP_RENDER:-0}"

mkdir -p "${REFKL}/logs" "${REFKL}/pids" "${PREP_DIR}" "${DATASETS}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "v6 pipeline already running" >&2
  exit 1
fi
echo $$ > "${REFKL}/pids/pipeline_v6.pid"

log() { echo "[v6pipeline] $* $(date -Is)" | tee -a "${LOG}"; }
count_prep() { find "${PREP_DIR}" -maxdepth 1 -name 'episode_*_bg*.npz' 2>/dev/null | wc -l; }

log "start unique=${TARGET_UNIQUE} variants=${VARIANTS} target=${TARGET_TRAJ} val=${VAL_DEMOS} prep=${PREP_DIR} datasets=${DATASETS}"

# ---------------------------------------------------------------- stage 1
if [[ "${SKIP_RENDER}" == "1" ]]; then
  log "stage 1: SKIPPED (prep=$(count_prep)/${TARGET_TRAJ})"
else
  IFS=',' read -r -a GPU_ARR <<< "${RENDER_GPUS}"
  N_GPU="${#GPU_ARR[@]}"
  STRIDE=$((N_GPU * WORKERS_PER_GPU))
  log "stage 1: render+prepare gpus=${RENDER_GPUS} workers_per_gpu=${WORKERS_PER_GPU} stride=${STRIDE} prep=$(count_prep)/${TARGET_TRAJ}"

  render_one() {
    local gpu="$1" offset="$2" worker="$3"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${SIM2REAL}:${SIM2REAL}/openreal2sim/simulation/maniskill" \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    RLVLA_MAX_STEPS_PER_CHUNK=4 \
    RLVLA_MIN_CHUNK_TRANSLATION=0.001 \
    PYTHONUNBUFFERED=1 \
    "${PY}" "${DATABC}/render_prepare_shard.py" "${COLLECT_DIR}" "${PREP_DIR}" \
      --tmp_dir "${DATABC}/sft_v6_bg_tmp_g${gpu}_w${worker}" \
      --variants "${VARIANTS}" \
      --start 0 \
      --end "${TARGET_UNIQUE}" \
      --offset "${offset}" \
      --stride "${STRIDE}" \
      --key airy_table_scene14sep26_left_image \
      --config_path "${SIM2REAL}/config/config_debug.yaml" \
      --scene assets/scenes/airy_table_scene14sep26_left_image/simulation/scene.json
  }

  offset=0
  pids=()
  for gpu in "${GPU_ARR[@]}"; do
    for worker in $(seq 0 $((WORKERS_PER_GPU - 1))); do
      wlog="${REFKL}/logs/render_v6_g${gpu}_w${worker}.log"
      log "launch render gpu=${gpu} worker=${worker} offset=${offset}/${STRIDE}"
      render_one "${gpu}" "${offset}" "${worker}" >> "${wlog}" 2>&1 &
      echo $! > "${REFKL}/pids/render_v6_g${gpu}_w${worker}.pid"
      pids+=("$!")
      offset=$((offset + 1))
    done
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      log "ERROR: render worker pid=${pid} failed"
      fail=1
    fi
  done
  [ "${fail}" -ne 0 ] && exit 1

  n_prep="$(count_prep)"
  log "render+prepare complete prep=${n_prep} expected=${TARGET_TRAJ}"
  if [ "${n_prep}" -lt "${TARGET_TRAJ}" ]; then
    log "ERROR: prepared ${n_prep} < ${TARGET_TRAJ}"
    exit 1
  fi
  rm -rf "${DATABC}"/sft_v6_bg_tmp_*
fi

EPISODE_LEN=144
if [[ -f "${PREP_DIR}/sft_episode_stats.json" ]]; then
  EPISODE_LEN="$(${PY} -c 'import json; print(json.load(open("'"${PREP_DIR}"'/sft_episode_stats.json"))["recommended_episode_len"])')"
fi
log "recommended episode_len=${EPISODE_LEN}"
printf '%s\n' "${EPISODE_LEN}" > "${REFKL}/episode_len_v6.txt"

# ---------------------------------------------------------------- stage 2
log "stage 2: tfds build sft_v2 -> ${DATASETS} (${TARGET_TRAJ} trajs, val=${VAL_DEMOS})"
(
  export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:${PATH}"
  export CUDA_VISIBLE_DEVICES=""
  export RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}"
  export RLVLA_SFT_TOTAL_DEMOS="${TARGET_TRAJ}"
  export RLVLA_SFT_VAL_DEMOS="${VAL_DEMOS}"
  # No-op chunks are already dropped at prepare time, so the builder's own
  # coarse 1 cm filter stays off.
  export RLVLA_SFT_DISABLE_FILTER=1
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export PYTHONUNBUFFERED=1
  cd "${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset"
  tfds build --overwrite --data_dir "${DATASETS}"
) 2>&1 | tee -a "${LOG}"
if [[ ! -d "${DATASETS}/sft_v2/1.0.0" ]]; then
  log "tfds did not produce ${DATASETS}/sft_v2/1.0.0"
  exit 1
fi

# ---------------------------------------------------------------- stage 3
log "stage 3: SFT nproc=${SFT_NPROC} gpus=${SFT_GPUS} max_steps=${MAX_STEPS} dim_weights=${DIM_WEIGHTS}"
SFT_LOG="${REFKL}/logs/sft_v6.log"
(
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
  nohup bash "${DATABC}/train_sft_v3.sh" >> "${SFT_LOG}" 2>&1 &
  echo $! > "${REFKL}/pids/sft_v6.pid"
)
sleep 5
log "SFT launched pid=$(cat "${REFKL}/pids/sft_v6.pid") log=${SFT_LOG}"

# ---------------------------------------------------------------- stage 4
SFT_RUN_DIR="${REPO}/openvla/${SFT_RUN_ROOT}/steps_${MAX_STEPS}-no_aug"
log "stage 4: SR watcher on gpus='${WATCH_GPUS}' watching ${SFT_RUN_DIR}"
(
  SFT_ROOT="${SFT_RUN_DIR}" \
  OUT_BASE="${REFKL}/sr_evals_v6" \
  APPEND_CSV="${REFKL}/sr_vs_steps_v6.csv" \
  GPUS="${WATCH_GPUS}" \
  NENVS="${WATCH_NENVS}" \
  SEEDS="10 11" \
  EPISODE_LEN="${EPISODE_LEN}" \
  POLL_SECONDS=300 \
  nohup bash "${REFKL}/watch_sr_evals.sh" >> "${REFKL}/logs/sr_watcher_v6.log" 2>&1 &
  echo $! > "${REFKL}/pids/sr_watcher_v6.pid"
)
sleep 2
log "SR watcher pid=$(cat "${REFKL}/pids/sr_watcher_v6.pid") csv=${REFKL}/sr_vs_steps_v6.csv"
log "pipeline done (SFT + SR watcher running)"
