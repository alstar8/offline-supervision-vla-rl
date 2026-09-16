#!/usr/bin/env bash
# One-GPU worker: collect proxy_ee_delta successes until DEST has TARGET unique npz.
set -u
set -o pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real"
OUT="${COLLECT_DIR:-$ROOT/runs/manual/pick_red_cube_100_v4}"
PATCH="$OUT/runtime_sim_patch.yaml"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"
TARGET="${TARGET:-1000}"
NUM_ENVS="${NUM_ENVS:-16}"
EPISODES_PER_ROUND="${EPISODES_PER_ROUND:-32}"
GPU="${GPU:?GPU env required}"
SEED_BASE="${SEED_BASE:?SEED_BASE env required}"
MAX_ROUNDS="${MAX_ROUNDS:-200}"
HARVEST="$ROOT/runs/manual/pick_red_cube_100/harvest_successes.py"
LOG="${WORKER_LOG:-$OUT/collect_1000_gpu${GPU}.log}"

export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/openreal2sim/simulation/maniskill${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/rounds_1000/gpu${GPU}" "$OUT"
exec 8>"$OUT/harvest.lock"
cd "$ROOT"

count() {
  find "$OUT" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l
}

n_have="$(count)"
round=0
echo "collect_worker gpu=${GPU} start $(date -Is) have=${n_have} target=${TARGET} seed_base=${SEED_BASE}" | tee -a "$LOG"

while [ "$n_have" -lt "$TARGET" ]; do
  if [ "$round" -ge "$MAX_ROUNDS" ]; then
    echo "ABORT gpu=${GPU} too many rounds have=${n_have}" | tee -a "$LOG"
    exit 1
  fi
  seed_start=$((SEED_BASE + round * EPISODES_PER_ROUND))
  round_dir="$OUT/rounds_1000/gpu${GPU}/round_$(printf '%03d' "$round")"
  mkdir -p "$round_dir"
  echo "ROUND gpu=${GPU} $round seed_start=${seed_start} n=${EPISODES_PER_ROUND} have=${n_have} $(date -Is)" | tee -a "$LOG"
  set +e
  "$PY" openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py \
    --run_mode collection \
    --motion_backend proxy_ee_delta \
    --config_path config/config_debug.yaml \
    --key airy_table_scene14sep26_left_image \
    --scene assets/scenes/airy_table_scene14sep26_left_image/simulation/scene.json \
    --task_type pick_up \
    --task_object_id orange_cube_ext \
    --headless \
    --sim_backend physx_cuda \
    --runtime_sim_patch "$PATCH" \
    --placement_seed_start "$seed_start" \
    --num_episodes "$EPISODES_PER_ROUND" \
    --num_envs "$NUM_ENVS" \
    --output_dir "$round_dir" \
    >>"$LOG" 2>&1
  rc=$?
  set +e
  echo "ROUND gpu=${GPU} $round collector_exit=${rc} $(date -Is)" | tee -a "$LOG"
  flock 8 "$PY" "$HARVEST" "$round_dir" "$OUT" "$TARGET" | tee -a "$LOG"
  rm -rf "$round_dir"
  n_have="$(count)"
  round=$((round + 1))
done

echo "DONE gpu=${GPU} have=${n_have} $(date -Is)" | tee -a "$LOG"
