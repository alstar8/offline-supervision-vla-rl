#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real"
OUT="$ROOT/runs/manual/pick_red_cube_100"
PATCH="$ROOT/runs/manual/pick_red_cube_collect100/runtime_sim_patch.yaml"
PY="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin/python"
TARGET=100
NUM_ENVS=16
EPISODES_PER_ROUND=32

export PATH="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/openreal2sim/simulation/maniskill${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
exec 9>"$OUT/collect.lock"
if ! flock -n 9; then
  echo "collect_until_100 already running" >&2
  exit 0
fi
cd "$ROOT"

count() {
  ls -1 "$OUT"/episode_*.npz 2>/dev/null | wc -l
}

n_have="$(count)"
seed_start=0
round=0
echo "collect_until_100 start $(date -Is) have=$n_have target=$TARGET" | tee -a "$OUT/collect.log"

while [ "$n_have" -lt "$TARGET" ]; do
  need=$((TARGET - n_have))
  n="$EPISODES_PER_ROUND"
  if [ "$need" -lt "$NUM_ENVS" ]; then
    n="$NUM_ENVS"
  fi
  round_dir="$OUT/rounds/round_$(printf '%02d' "$round")"
  mkdir -p "$round_dir"
  echo "ROUND $round seed_start=$seed_start n=$n have=$n_have $(date -Is)" | tee -a "$OUT/collect.log"
  set +e
  "$PY" openreal2sim/simulation/maniskill/scripts/run_rc5_unified.py \
    --run_mode collection \
    --motion_backend proxy_ee_delta \
    --config_path config/config_debug.yaml \
    --key airi_table_new_empty3_image \
    --scene assets/scenes/airi_table_new_empty3_image/simulation/scene.json \
    --task_type pick_up \
    --task_object_id orange_cube_ext \
    --headless \
    --sim_backend physx_cuda \
    --runtime_sim_patch "$PATCH" \
    --placement_seed_start "$seed_start" \
    --num_episodes "$n" \
    --num_envs "$NUM_ENVS" \
    --output_dir "$round_dir" \
    >>"$OUT/collect.log" 2>&1
  rc=$?
  set -e
  echo "ROUND $round collector_exit=$rc $(date -Is)" | tee -a "$OUT/collect.log"
  "$PY" "$OUT/harvest_successes.py" "$round_dir" "$OUT" "$TARGET" | tee -a "$OUT/collect.log"
  n_have="$(count)"
  seed_start=$((seed_start + n))
  round=$((round + 1))
  if [ "$round" -ge 12 ]; then
    echo "ABORT too many rounds have=$n_have" | tee -a "$OUT/collect.log"
    exit 1
  fi
done

echo "DONE $n_have $(date -Is)" | tee -a "$OUT/collect.log"
