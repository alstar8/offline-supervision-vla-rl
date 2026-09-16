#!/usr/bin/env bash
# 30-episode OpenReal2Sim SR eval of the latest v5 SFT LoRA on GPUs 4-7.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_ROOT="${REPO}/openvla/checkpoints/sft/openvla_v2_scene14sep26_10k_xy/steps_16000-no_aug"
CKPT="${CKPT:-$(ls -d "${SFT_ROOT}"/lora_* | sort | tail -n 1)}"
EVAL_SH="${RUN_DIR}/eval_sim_v2.sh"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/eval_sft_v5_latest_x30}"
EPISODE_LEN="${EPISODE_LEN:-168}"
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-8}"

mkdir -p "${OUT_ROOT}" "${RUN_DIR}/logs" "${RUN_DIR}/pids"

echo "SFT SR eval start $(date -Is) ckpt=${CKPT} episode_len=${EPISODE_LEN} inferbatch=${BUFFER_INFERBATCH}" | tee "${OUT_ROOT}/launch.log"

# 8+8+8+6 = 30 episodes, disjoint seeds.
GPUS=(4 5 6 7)
NENVS=(8 8 8 6)
SEEDS=(10 11 12 13)

pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  n="${NENVS[$i]}"
  seed="${SEEDS[$i]}"
  out="${OUT_ROOT}/gpu${gpu}"
  log="${RUN_DIR}/logs/eval_sft_v5_x30_gpu${gpu}.log"
  echo "===== start $(date -Is) gpu=${gpu} num_envs=${n} seed=${seed} ckpt=${CKPT} =====" >> "${log}"
  CKPT="${CKPT}" \
  CUDA_ID="${gpu}" \
  NUM_ENVS="${n}" \
  BUFFER_INFERBATCH="${BUFFER_INFERBATCH}" \
  SEED="${seed}" \
  EPISODE_LEN="${EPISODE_LEN}" \
  NAME="SFT_v5_eval_lora_x30_gpu${gpu}" \
  OUT_DIR="${out}" \
  bash "${EVAL_SH}" >> "${log}" 2>&1 &
  echo $! > "${RUN_DIR}/pids/eval_sft_v5_x30_gpu${gpu}.pid"
  pids+=("$!")
  echo "launched gpu=${gpu} pid=${pids[-1]} n=${n} seed=${seed} log=${log}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "eval worker pid=${pid} failed"
    fail=1
  fi
done

PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"
"${PY}" - <<PY
import json
from pathlib import Path

root = Path("${OUT_ROOT}")
ckpt = "${CKPT}"
episode_len = int("${EPISODE_LEN}")
rows = []
videos = sorted(root.glob("gpu*/videos/video_*-s_*.mp4"))
if not videos:
    videos = sorted(root.glob("gpu*/wandb/*/glob/vis_0_train/video_*-s_*.mp4"))
for video in videos:
    name = video.stem
    last_s = int(name.rsplit("_", 1)[-1])
    gpu = int(str(video).split("/gpu")[1].split("/")[0])
    rows.append({"gpu": gpu, "env": name, "video": str(video), "last_success": last_s})
n = len(rows)
n_ok = int(sum(int(r["last_success"]) for r in rows))
summary = {
    "ckpt": ckpt,
    "n_episodes": n,
    "n_success": n_ok,
    "success_rate": (n_ok / n) if n else 0.0,
    "episode_len": episode_len,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("ckpt", "n_episodes", "n_success", "success_rate", "episode_len")}, indent=2))
PY

if [ "${fail}" -ne 0 ]; then
  exit 1
fi
echo "EVAL_SR_DONE $(date -Is)"
