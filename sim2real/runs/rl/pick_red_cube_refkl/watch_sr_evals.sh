#!/usr/bin/env bash
# Poll an SFT run directory and score every new lora_* checkpoint in closed loop.
#
# Teacher-forced token accuracy does not predict closed-loop success (it cannot
# see compounding error, and it scores the gripper token against a ground-truth
# prefix it never gets at inference). This watcher produces the only signal that
# tracks SR, appending one row per checkpoint to sr_vs_steps.csv.
#
#   SFT_ROOT=/path/to/steps_XXXX-no_aug GPUS="6 7" NENVS="8 8" \
#     nohup bash watch_sr_evals.sh > logs/sr_watcher.log 2>&1 &
set -uo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_ROOT="${SFT_ROOT:?set SFT_ROOT to the run dir holding lora_* checkpoints}"
OUT_BASE="${OUT_BASE:-${RUN_DIR}/sr_evals}"
APPEND_CSV="${APPEND_CSV:-${RUN_DIR}/sr_vs_steps.csv}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EPISODE_LEN="${EPISODE_LEN:-168}"
STOP_FILE="${STOP_FILE:-${OUT_BASE}/STOP}"

# Keep these off the training GPUs.
GPUS="${GPUS:-6 7}"
NENVS="${NENVS:-8 8}"
SEEDS="${SEEDS:-10 11}"

mkdir -p "${OUT_BASE}" "${RUN_DIR}/logs"

echo "SR watcher start $(date -Is) sft_root=${SFT_ROOT} gpus='${GPUS}' poll=${POLL_SECONDS}s csv=${APPEND_CSV}"

while true; do
  if [[ -f "${STOP_FILE}" ]]; then
    echo "stop file present, exiting $(date -Is)"
    exit 0
  fi

  for ckpt in $(ls -d "${SFT_ROOT}"/lora_* 2>/dev/null | sort); do
    tag="$(basename "${ckpt}")"
    marker="${OUT_BASE}/${tag}.done"
    [[ -f "${marker}" ]] && continue
    # A checkpoint dir appears before its weights finish writing.
    [[ -f "${ckpt}/adapter_model.safetensors" ]] || continue
    [[ -f "${ckpt}/dataset_statistics.json" ]] || continue

    step="$(echo "${tag}" | sed -E 's/[^0-9]//g' | sed 's/^0*//')"
    echo "=== scoring ${tag} (step ${step}) at $(date -Is) ==="
    CKPT="${ckpt}" \
    OUT_ROOT="${OUT_BASE}/${tag}" \
    TAG="${tag}" \
    STEP="${step}" \
    EPISODE_LEN="${EPISODE_LEN}" \
    GPUS="${GPUS}" \
    NENVS="${NENVS}" \
    SEEDS="${SEEDS}" \
    APPEND_CSV="${APPEND_CSV}" \
    bash "${RUN_DIR}/eval_sft_sr.sh"
    rc=$?
    # Mark done either way: a retry would re-run the same rollouts and the
    # summariser already records whatever episodes completed.
    echo "rc=${rc} at $(date -Is)" > "${marker}"
    echo "=== ${tag} done rc=${rc} ==="
    if [[ -f "${APPEND_CSV}" ]]; then
      echo "--- sr_vs_steps so far ---"
      cat "${APPEND_CSV}"
    fi
  done

  sleep "${POLL_SECONDS}"
done
