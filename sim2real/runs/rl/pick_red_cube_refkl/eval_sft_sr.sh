#!/usr/bin/env bash
# Closed-loop SR eval of one SFT LoRA checkpoint, sharded over a set of GPUs.
#
# Generalises eval_sft_v5_sr30.sh: takes any CKPT / OUT_ROOT / GPU set, and
# summarises from the per-worker stats.yaml instead of video filenames.
#
#   CKPT=/path/to/lora_008000 OUT_ROOT=/path/to/out GPUS="4 5 6 7" NENVS="8 8 8 6" \
#     bash eval_sft_sr.sh
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_ROOT="${SFT_ROOT:-${REPO}/openvla/checkpoints/sft/openvla_v2_scene14sep26_10k_xy/steps_16000-no_aug}"
CKPT="${CKPT:-$(ls -d "${SFT_ROOT}"/lora_* | sort | tail -n 1)}"
EVAL_SH="${RUN_DIR}/eval_sim_v2.sh"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"

STEP="${STEP:-$(basename "${CKPT}" | sed -E 's/[^0-9]//g' | sed 's/^0*//')}"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/eval_sr_$(basename "${CKPT}")}"
EPISODE_LEN="${EPISODE_LEN:-168}"
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-8}"
APPEND_CSV="${APPEND_CSV:-${RUN_DIR}/sr_vs_steps.csv}"
TAG="${TAG:-$(basename "${CKPT}")}"

# GPU 3 hangs on PhysX/Vulkan in this container, so it is excluded by default.
read -r -a GPUS <<< "${GPUS:-4 5 6 7}"
read -r -a NENVS <<< "${NENVS:-8 8 8 6}"
read -r -a SEEDS <<< "${SEEDS:-10 11 12 13}"
# Extra flags forwarded verbatim to train_ms3_ppo_sft.py, e.g.
# EXTRA_ARGS="--eval_at_train_temperature --vla_temperature 1.0"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

if [[ "${#GPUS[@]}" -ne "${#NENVS[@]}" || "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]]; then
  echo "GPUS/NENVS/SEEDS must have the same length" >&2
  exit 1
fi

# A 12-env V2 eval peaks near 22 GiB. Co-running one against a training job that
# still had 21 GiB free OOM-killed a 2h DataBC run on 2026-09-16: the trainer's own
# footprint grows during generate, so "free right now" is not enough on its own.
# Require the eval's peak plus margin for the other process before launching.
MIN_FREE_MIB="${MIN_FREE_MIB:-26000}"
if [[ "${SKIP_VRAM_CHECK:-0}" != "1" ]]; then
  for gpu in "${GPUS[@]}"; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | tr -d ' ')"
    if [[ -z "${free_mib}" ]]; then
      echo "cannot read free VRAM for GPU ${gpu}; set SKIP_VRAM_CHECK=1 to override" >&2
      exit 1
    fi
    if (( free_mib < MIN_FREE_MIB )); then
      echo "GPU ${gpu}: ${free_mib} MiB free < ${MIN_FREE_MIB} MiB required" >&2
      echo "another job is probably training here; launching now risks OOM-killing it" >&2
      echo "wait for the GPU, lower NENVS, or set SKIP_VRAM_CHECK=1 to override" >&2
      exit 1
    fi
  done
fi

mkdir -p "${OUT_ROOT}" "${RUN_DIR}/logs" "${RUN_DIR}/pids"

TOTAL=0
for n in "${NENVS[@]}"; do TOTAL=$((TOTAL + n)); done
echo "SR eval start $(date -Is) ckpt=${CKPT} step=${STEP} episodes=${TOTAL} episode_len=${EPISODE_LEN}" \
  | tee "${OUT_ROOT}/launch.log"

pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  n="${NENVS[$i]}"
  seed="${SEEDS[$i]}"
  out="${OUT_ROOT}/gpu${gpu}"
  log="${RUN_DIR}/logs/eval_sr_${TAG}_gpu${gpu}.log"
  echo "===== start $(date -Is) gpu=${gpu} num_envs=${n} seed=${seed} ckpt=${CKPT} =====" >> "${log}"
  CKPT="${CKPT}" \
  CUDA_ID="${gpu}" \
  NUM_ENVS="${n}" \
  BUFFER_INFERBATCH="${BUFFER_INFERBATCH}" \
  SEED="${seed}" \
  EPISODE_LEN="${EPISODE_LEN}" \
  NAME="SFT_sr_${TAG}_gpu${gpu}" \
  OUT_DIR="${out}" \
  bash "${EVAL_SH}" "${EXTRA[@]}" >> "${log}" 2>&1 &
  pid=$!
  echo "${pid}" > "${RUN_DIR}/pids/eval_sr_${TAG}_gpu${gpu}.pid"
  pids+=("${pid}")
  echo "launched gpu=${gpu} pid=${pid} n=${n} seed=${seed} log=${log}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "eval worker pid=${pid} failed"
    fail=1
  fi
done

"${PY}" "${RUN_DIR}/summarize_sr.py" \
  --out-root "${OUT_ROOT}" \
  --ckpt "${CKPT}" \
  --step "${STEP}" \
  --episode-len "${EPISODE_LEN}" \
  --append-csv "${APPEND_CSV}"

if [ "${fail}" -ne 0 ]; then
  echo "EVAL_SR_DONE_WITH_WORKER_ERRORS $(date -Is)"
  exit 1
fi
echo "EVAL_SR_DONE $(date -Is)"
