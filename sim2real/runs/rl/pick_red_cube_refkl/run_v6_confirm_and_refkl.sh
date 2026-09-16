#!/usr/bin/env bash
# Stage 5-7 of v6: confirm the SFT checkpoints at 60 episodes, then start RefKL
# from whichever one actually wins in closed loop.
#
# Why a re-score at all: the watcher's points are 16 episodes each, so the
# standard error is ~10pp and the step-4000/5000/6000 readings (18.8/18.8/12.5)
# are mutually indistinguishable. Token accuracy cannot break the tie either --
# acc_core held 0.71-0.72 across the entire run while SR moved between 0% and
# 19%, which is the whole reason this pipeline exists. So the RL init is chosen
# from 60 paired episodes instead.
#
# Every checkpoint sees the identical 60 initial states (same GPUS/NENVS/SEEDS),
# making it a paired comparison rather than 8 independent samples.
#
#   nohup bash run_v6_confirm_and_refkl.sh > logs/confirm_refkl_v6.log 2>&1 &
set -uo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"

SFT_ROOT="${SFT_ROOT:-${REPO}/openvla/checkpoints/sft/openvla_v2_3var_fixedlabels/steps_8000-no_aug}"
FINAL_TAG="${FINAL_TAG:-lora_008000}"
DATASETS="${DATASETS:-${RUN_DIR}/datasets_v6}"

# 60 episodes over GPUs 0,1,2,4,5 only (3 hangs on PhysX/Vulkan; 6 and 7 are
# reserved for the online RL probe, which runs concurrently with this sweep).
CONFIRM_GPUS="${CONFIRM_GPUS:-0 1 2 4 5}"
CONFIRM_NENVS="${CONFIRM_NENVS:-12 12 12 12 12}"
# Disjoint from the watcher's seeds 10/11 so this is an independent estimate.
CONFIRM_SEEDS="${CONFIRM_SEEDS:-20 21 22 23 24}"
# Same 144 the 16-episode curve used, so both sets of points stay comparable.
CONFIRM_EPISODE_LEN="${CONFIRM_EPISODE_LEN:-144}"
CONFIRM_CSV="${CONFIRM_CSV:-${RUN_DIR}/sr_vs_steps_v6_x60.csv}"
CONFIRM_OUT_BASE="${CONFIRM_OUT_BASE:-${RUN_DIR}/sr_evals_v6_x60}"

# Demos reach first success at chunk 63 (median) / 108 (p99) under the v6
# 4-frame chunking, so 112 covers virtually every demo-paced episode. Going to
# 144 would buy nothing but cost updates: steps_max is a local env-step budget,
# so longer episodes mean fewer PPO updates for the same budget (279 -> 217).
# Consequence to remember when reading the RL curve: its starting SR will sit
# slightly below the 144-step confirmation number.
REFKL_EPISODE_LEN="${REFKL_EPISODE_LEN:-112}"
# Match the v6 SFT loss weights so the BC-to-ref term pulls toward the same
# objective the student was trained on.
REFKL_DIM_WEIGHTS="${REFKL_DIM_WEIGHTS:-3:3:1.5:0.2:0.2:0.2:1}"
# Off by default: committing a 2M-step run only makes sense once the online
# probe on GPUs 6/7 has shown the RL pipeline itself is sound. This script's job
# is to produce the 60-episode curve and name the init.
START_REFKL="${START_REFKL:-0}"

WATCHER_STOP="${RUN_DIR}/sr_evals_v6/STOP"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-180}"

mkdir -p "${CONFIRM_OUT_BASE}" "${RUN_DIR}/logs" "${RUN_DIR}/pids"

log() { echo "[v6confirm] $* $(date -Is)"; }

sft_running() { pgrep -f "vla-scripts/finetune.py" >/dev/null 2>&1; }

# ------------------------------------------------------------------ stage 5
log "waiting for SFT to finish (final=${FINAL_TAG}, max ${MAX_WAIT_MIN} min)"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while sft_running; do
  if [[ $(date +%s) -ge ${deadline} ]]; then
    log "WARN: SFT still running after ${MAX_WAIT_MIN} min; proceeding anyway"
    break
  fi
  sleep 60
done
log "SFT no longer running; checkpoints present: $(ls -d "${SFT_ROOT}"/lora_* 2>/dev/null | wc -l)"

# The watcher may already have been stopped by hand; only wait on it if it is
# actually still alive, otherwise this blocks for nothing waiting on a marker
# that will never be written.
touch "${WATCHER_STOP}"
if pgrep -f "[w]atch_sr_evals.sh" >/dev/null 2>&1; then
  log "watcher alive; stop file written, waiting for it to exit"
  deadline=$(( $(date +%s) + 20 * 60 ))
  while pgrep -f "[w]atch_sr_evals.sh" >/dev/null 2>&1; do
    if [[ $(date +%s) -ge ${deadline} ]]; then
      log "WARN: watcher still alive after 20 min; killing"
      pkill -f "[w]atch_sr_evals.sh" || true
      sleep 10
      break
    fi
    sleep 30
  done
else
  log "watcher already stopped"
fi
log "16-episode curve complete:"
cat "${RUN_DIR}/sr_vs_steps_v6.csv" 2>/dev/null

# ------------------------------------------------------------------ stage 6
n_conf=0
for n in ${CONFIRM_NENVS}; do n_conf=$((n_conf + n)); done
log "stage 6: re-scoring every checkpoint at ${n_conf} episodes, gpus='${CONFIRM_GPUS}'"

for ckpt in $(ls -d "${SFT_ROOT}"/lora_* 2>/dev/null | sort); do
  tag="$(basename "${ckpt}")"
  [[ -f "${ckpt}/adapter_model.safetensors" ]] || { log "skip ${tag} (no weights)"; continue; }
  [[ -f "${CONFIRM_OUT_BASE}/${tag}.done" ]] && { log "skip ${tag} (already scored)"; continue; }
  step="$(echo "${tag}" | sed -E 's/[^0-9]//g' | sed 's/^0*//')"
  log "=== confirm ${tag} (step ${step}) ==="
  CKPT="${ckpt}" \
  OUT_ROOT="${CONFIRM_OUT_BASE}/${tag}" \
  TAG="${tag}_x60" \
  STEP="${step}" \
  EPISODE_LEN="${CONFIRM_EPISODE_LEN}" \
  GPUS="${CONFIRM_GPUS}" \
  NENVS="${CONFIRM_NENVS}" \
  SEEDS="${CONFIRM_SEEDS}" \
  APPEND_CSV="${CONFIRM_CSV}" \
  bash "${RUN_DIR}/eval_sft_sr.sh"
  rc=$?
  echo "rc=${rc} at $(date -Is)" > "${CONFIRM_OUT_BASE}/${tag}.done"
  log "=== ${tag} confirmed rc=${rc} ==="
done

log "60-episode curve:"
cat "${CONFIRM_CSV}" 2>/dev/null

# ------------------------------------------------------------------ stage 7
BEST="$("${PY}" "${RUN_DIR}/pick_best_sr_lora.py" --csv "${CONFIRM_CSV}" \
        --min-episodes $((n_conf / 2)) --print-table)"
if [[ -z "${BEST}" || ! -d "${BEST}" ]]; then
  log "ERROR: could not pick a checkpoint from ${CONFIRM_CSV}"
  exit 1
fi
echo "${BEST}" > "${RUN_DIR}/selected_sft_lora_v6.txt"
log "selected init: ${BEST}"

if [[ "${START_REFKL}" != "1" ]]; then
  log "START_REFKL=0, stopping before RL"
  exit 0
fi

log "stage 7: launching RefKL from ${BEST} (episode_len=${REFKL_EPISODE_LEN})"
SFT_LORA_PATH="${BEST}" \
REFKL_LOAD_PATH="${BEST}" \
SFT_DATA_ROOT_DIR="${DATASETS}" \
SFT_SKIP_IMAGE_RESIZE=1 \
SFT_ACTION_DIM_WEIGHTS="${REFKL_DIM_WEIGHTS}" \
SFT_SHUFFLE_BUFFER_SIZE=4000 \
EPISODE_LEN="${REFKL_EPISODE_LEN}" \
REFKL_JOB_DIR="${RUN_DIR}/ddp_v6" \
LOG="${RUN_DIR}/logs/ddp_v6.log" \
PID_FILE="${RUN_DIR}/pids/ddp_v6.pid" \
MASTER_PORT=29761 \
NAME="RefKL_pick_red_cube_v6" \
bash "${RUN_DIR}/launch_gpus_v3.sh"

sleep 10
log "RefKL launched pid=$(cat "${RUN_DIR}/pids/ddp_v6.pid" 2>/dev/null) log=${RUN_DIR}/logs/ddp_v6.log"
log "done"
