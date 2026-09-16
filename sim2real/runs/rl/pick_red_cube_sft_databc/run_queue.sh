#!/usr/bin/env bash
# Serialized experiment queue for the SINGLE-GPU host (online / RL phase).
#
# Work is split across two hosts that share this NFS workspace:
#
#   server8gpu_CLOUD (7 GPUs)  owns the OFFLINE phase: v6 render+prepare into
#                              sft_v6_proprio/, the TFDS build, and SFT / RefKL.
#   this host (1 GPU)          owns the ONLINE phase: grasp sweeps and DataBC/PPO.
#
# The offline stages are kept here only so the chain is reproducible on a machine
# that owns them. They are REMOTE_STAGES and refuse to run by default: sft_v6_proprio
# is a shared directory and two hosts rendering into it duplicates work. Override
# with FORCE_REMOTE_STAGES=1 only if you know the 8-GPU host is idle.
#
# Every stage runs in the foreground under one lock so the single GPU is never
# oversubscribed.
#
# usage:  ./run_queue.sh                       # online stages only
#         STAGES="databc" ./run_queue.sh
#         ./run_queue.sh --list
set -uo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
SIM2REAL="${REPO}/sim2real"
DATABC="${SIM2REAL}/runs/rl/pick_red_cube_sft_databc"
REFKL="${SIM2REAL}/runs/rl/pick_red_cube_refkl"
COLLECT_DIR="${SIM2REAL}/runs/manual/pick_red_cube_100_v4"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"

PREP_DIR="${PREP_DIR:-${DATABC}/sft_v6_proprio}"
DATASETS_V6="${DATASETS_V6:-${REFKL}/datasets_v6}"
SFT_V6_ROOT="${SFT_V6_ROOT:-checkpoints/sft/openvla_v2_3var_fixedlabels}"
SFT_10K="${REPO}/openvla/checkpoints/sft/openvla_v2_scene14sep26_10k_xy/steps_16000-no_aug"

VARIANTS="${VARIANTS:-3}"
TARGET_UNIQUE="${TARGET_UNIQUE:-1000}"
TARGET_TRAJ=$((TARGET_UNIQUE * VARIANTS))
VAL_DEMOS=$((16 * VARIANTS))
RENDER_WORKERS="${RENDER_WORKERS:-10}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-8000}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-1000,2000,3000,4000,5000,6000,7000,8000}"
DIM_WEIGHTS="${DIM_WEIGHTS:-3:3:1.5:0.2:0.2:0.2:1}"
# 48x120 instead of 64x160: reach finishes by ~step 60 and grasp by ~step 80, so the
# extra horizon mostly bought physics divergence. Cutting transitions per update
# 10240 -> 5760 roughly doubles updates/day, and SR historically needs 200+ updates.
EPISODE_LEN="${EPISODE_LEN:-120}"
NUM_ENVS="${NUM_ENVS:-48}"
GRASP_GATE="${GRASP_GATE:-0.15}"

QUEUE_LOG="${DATABC}/queue.log"
LOCK="${DATABC}/queue.lock"
ONLINE_STAGES="sweep_v5 sweep_v6 databc"
REMOTE_STAGES="render_v6 tfds_v6 sft_v6"
ALL_STAGES="${ONLINE_STAGES} ${REMOTE_STAGES}"
STAGES="${STAGES:-${ONLINE_STAGES}}"

if [[ "${1:-}" == "--list" ]]; then
  echo "online (this host): ${ONLINE_STAGES}"
  echo "remote (8-GPU host): ${REMOTE_STAGES}"
  exit 0
fi

log() { echo "[queue] $* $(date -Is)" | tee -a "${QUEUE_LOG}"; }

mkdir -p "${DATABC}" "${PREP_DIR}" "${DATASETS_V6}" "${REFKL}/logs"
exec 9>"${LOCK}"
# A stage's children inherit fd 9, so the lock stays held for as long as an orphaned
# stage runs. QUEUE_WAIT_LOCK=1 blocks until it clears, which is how you enqueue work
# behind a sweep that is already in flight.
if [[ "${QUEUE_WAIT_LOCK:-0}" == "1" ]]; then
  echo "[queue] waiting for lock ${LOCK} ..."
  flock 9
elif ! flock -n 9; then
  echo "queue already running (lock ${LOCK})" >&2
  exit 1
fi
echo $$ > "${DATABC}/queue.pid"

export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:${PATH}"
export HF_HOME="/workspace-SR008.nfs2/users/staroverov/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

wait_for_free_gpu() {
  local limit="${1:-4096}" waited=0 timeout="${GPU_WAIT_SECONDS:-14400}"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= limit )) && return 0
    (( waited % 300 == 0 )) && log "waiting for GPU (used=${used} MiB > ${limit}, ${waited}s elapsed)"
    sleep 15; waited=$((waited + 15))
    (( waited > timeout )) && { log "ERROR: GPU still busy after ${timeout}s"; return 1; }
  done
}

# ------------------------------------------------------------------ stages

stage_sweep_v5() {
  # Clean baseline: rank the existing 10k (v5) checkpoints with the divergence
  # guards on, so grasp rate is not inflated by objects flung out of the scene.
  wait_for_free_gpu || return 1
  EPISODE_LEN="${EPISODE_LEN}" NUM_ENVS="${NUM_ENVS}" OUT_ROOT="${DATABC}/sweep_grasp_guarded" \
    bash "${DATABC}/sweep_sft_grasp.sh" \
      "${SFT_10K}/lora_008000" \
      "${SFT_10K}/lora_004000" \
      "${SFT_10K}/lora_002000"
}

stage_render_v6() {
  # Resumable: render_prepare_shard.py skips episodes whose variants already exist.
  local have
  have="$(find "${PREP_DIR}" -maxdepth 1 -name 'episode_*_bg*.npz' | wc -l)"
  log "render_v6: ${have}/${TARGET_TRAJ} prepared, ${RENDER_WORKERS} workers on GPU 0"
  if (( have >= TARGET_TRAJ )); then log "render_v6: already complete"; return 0; fi
  wait_for_free_gpu || return 1

  local pids=() w
  for (( w=0; w<RENDER_WORKERS; w++ )); do
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${SIM2REAL}:${SIM2REAL}/openreal2sim/simulation/maniskill" \
    RLVLA_MAX_STEPS_PER_CHUNK=4 \
    RLVLA_MIN_CHUNK_TRANSLATION=0.001 \
    "${PY}" "${DATABC}/render_prepare_shard.py" "${COLLECT_DIR}" "${PREP_DIR}" \
      --tmp_dir "${DATABC}/sft_v6_bg_tmp_q_w${w}" \
      --variants "${VARIANTS}" --start 0 --end "${TARGET_UNIQUE}" \
      --offset "${w}" --stride "${RENDER_WORKERS}" \
      --key airy_table_scene14sep26_left_image \
      --config_path "${SIM2REAL}/config/config_debug.yaml" \
      --scene assets/scenes/airy_table_scene14sep26_left_image/simulation/scene.json \
      >>"${REFKL}/logs/render_v6_queue_w${w}.log" 2>&1 &
    pids+=("$!")
  done

  local fail=0
  for p in "${pids[@]}"; do wait "${p}" || fail=1; done
  have="$(find "${PREP_DIR}" -maxdepth 1 -name 'episode_*_bg*.npz' | wc -l)"
  log "render_v6: finished with ${have}/${TARGET_TRAJ} (worker fail=${fail})"
  (( have >= TARGET_TRAJ )) || return 1
}

stage_tfds_v6() {
  # CPU only. Keeps the prepared npz so the set can be re-chunked later.
  RLVLA_SFT_SOURCE_DATA_DIR="${PREP_DIR}" \
  RLVLA_SFT_TOTAL_DEMOS="${TARGET_TRAJ}" \
  RLVLA_SFT_VAL_DEMOS="${VAL_DEMOS}" \
  RLVLA_SFT_DISABLE_FILTER=1 \
  CUDA_VISIBLE_DEVICES="" \
    bash -c "cd '${REPO}/openvla/rlds_dataset_builder/sft_v2_dataset' && tfds build --overwrite --data_dir '${DATASETS_V6}'"
  [[ -d "${DATASETS_V6}/sft_v2/1.0.0" ]] || return 1
}

stage_sft_v6() {
  wait_for_free_gpu || return 1
  SFT_RUN_ROOT="${SFT_V6_ROOT}" \
  MAX_STEPS="${SFT_MAX_STEPS}" \
  SAVE_STEPS="${SFT_SAVE_STEPS}" \
  EVAL_STEPS=500 \
  NPROC=1 \
  CUDA_VISIBLE_DEVICES=0 \
  DATA_ROOT_DIR="${DATASETS_V6}" \
  SKIP_IMAGE_RESIZE=1 \
  ACTION_DIM_LOSS_WEIGHTS="${DIM_WEIGHTS}" \
  SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-16000}" \
    bash "${DATABC}/train_sft_v3.sh"
}

stage_sweep_v6() {
  wait_for_free_gpu || return 1
  local root="${REPO}/openvla/${SFT_V6_ROOT}/steps_${SFT_MAX_STEPS}-no_aug"
  # Most-trained first, capped: each eval costs ~25 min on one GPU.
  local ckpts=()
  while IFS= read -r d; do ckpts+=("${d}"); done < <(
    find "${root}" -maxdepth 1 -name 'lora_*' -type d | sort -r | head -n "${SWEEP_V6_MAX:-4}")
  if (( ${#ckpts[@]} == 0 )); then log "sweep_v6: no checkpoints under ${root}"; return 1; fi
  log "sweep_v6: ${#ckpts[@]} checkpoint(s)"
  EPISODE_LEN="${EPISODE_LEN}" NUM_ENVS="${NUM_ENVS}" OUT_ROOT="${DATABC}/sweep_grasp_v6" \
    bash "${DATABC}/sweep_sft_grasp.sh" "${ckpts[@]}"
}

stage_databc() {
  # Only launch RL if some checkpoint actually grasps. Every historical run that
  # started below ~15% grasp has stalled at ~0% SR.
  #
  # Wait for the GPU before resolving the winner: an in-flight sweep holds the GPU
  # and has not yet written stats for its remaining checkpoints, so ranking first
  # would pick from a partial table.
  wait_for_free_gpu || return 1
  local best_dir="" sweep
  for sweep in "${DATABC}/sweep_grasp_v6" "${DATABC}/sweep_grasp_guarded"; do
    [[ -d "${sweep}" ]] || continue
    if "${PY}" "${DATABC}/summarize_grasp_sweep.py" "${sweep}" --gate "${GRASP_GATE}" >>"${QUEUE_LOG}" 2>&1; then
      best_dir="${sweep}"; break
    fi
    log "databc: gate ${GRASP_GATE} not met in ${sweep}"
  done
  if [[ -z "${best_dir}" ]]; then
    log "databc: SKIPPED, no sweep cleared the ${GRASP_GATE} grasp gate"
    return 0
  fi
  local best
  best="$("${PY}" "${DATABC}/summarize_grasp_sweep.py" "${best_dir}" --print-best)"
  [[ -d "${best}" ]] || { log "databc: bad checkpoint path '${best}'"; return 1; }
  log "databc: launching from ${best}"
  local name="${DATABC_NAME:-DataBC_v6_guarded_ep${EPISODE_LEN}}"
  # One log per run. A fixed filename made consecutive runs concatenate, so parsing the
  # metric history of the newer run silently returned the older run's episodes.
  local run_log="${DATABC}/databc_${name}_$(date +%Y%m%d_%H%M%S).log"
  log "databc: log ${run_log}"
  DATABC_NAME="${name}" \
  EPISODE_LEN="${EPISODE_LEN}" \
  NUM_ENVS="${NUM_ENVS}" \
    bash "${DATABC}/train_databc_v3.sh" "${best}" >"${run_log}" 2>&1
}

# ------------------------------------------------------------------ driver

# Stages that write the v6 dataset / train SFT. Skip them when the 8-GPU
# refkl pipeline is already doing that work on the shared NFS tree.
v6_remote_active() {
  local newest
  newest="$(find "${REFKL}/logs" -maxdepth 1 -name 'render_v6_g*_w*.log' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
  [[ -n "${newest}" ]] || return 1
  local age
  age="$(awk -v t="${newest}" 'BEGIN { printf "%d", systime() - t }')"
  (( age < 900 ))
}

log "queue start | stages: ${STAGES} | ${NUM_ENVS}x${EPISODE_LEN} gate=${GRASP_GATE}"
for stage in ${STAGES}; do
  if ! declare -F "stage_${stage}" >/dev/null; then
    log "unknown stage '${stage}' (valid: ${ALL_STAGES})"; exit 2
  fi
  if [[ " ${REMOTE_STAGES} " == *" ${stage} "* && "${FORCE_REMOTE_STAGES:-0}" != "1" ]]; then
    log "=== stage ${stage}: SKIPPED, owned by server8gpu_CLOUD (set FORCE_REMOTE_STAGES=1 to override)"
    continue
  fi
  case "${stage}" in
    render_v6|tfds_v6|sft_v6)
      if v6_remote_active; then
        log "=== stage ${stage}: SKIPPED (v6 render/SFT already running on 8-GPU host)"
        continue
      fi
      ;;
  esac
  log "=== stage ${stage}: start"
  start=$(date +%s)
  if "stage_${stage}"; then
    log "=== stage ${stage}: OK ($(( $(date +%s) - start ))s)"
  else
    rc=$?
    log "=== stage ${stage}: FAILED rc=${rc} ($(( $(date +%s) - start ))s) -- stopping queue"
    exit "${rc}"
  fi
done
log "queue done"
