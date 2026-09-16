#!/usr/bin/env bash
# Collect until 1000 unique successful trajectories using GPUs 0-2 in parallel.
set -euo pipefail

ROOT="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real"
OUT="$ROOT/runs/manual/pick_red_cube_100_v4"
WORKER="$OUT/collect_worker_1000.sh"
TARGET="${TARGET:-1000}"

mkdir -p "$OUT/pids" "$OUT"
exec 9>"$OUT/collect_1000.lock"
if ! flock -n 9; then
  echo "collect_until_1000 already running" >&2
  exit 0
fi

echo "collect_until_1000 start $(date -Is) have=$(find "$OUT" -maxdepth 1 -name 'episode_*.npz' | wc -l) target=${TARGET}" | tee -a "$OUT/collect_1000.log"

# Keep placement seeds far from the original 0-191 collection.
setsid env GPU=0 SEED_BASE=1000 TARGET="$TARGET" bash "$WORKER" >/dev/null 2>&1 &
echo $! > "$OUT/pids/collect_gpu0.pid"
setsid env GPU=1 SEED_BASE=101000 TARGET="$TARGET" bash "$WORKER" >/dev/null 2>&1 &
echo $! > "$OUT/pids/collect_gpu1.pid"
setsid env GPU=2 SEED_BASE=201000 TARGET="$TARGET" bash "$WORKER" >/dev/null 2>&1 &
echo $! > "$OUT/pids/collect_gpu2.pid"

echo "collect_until_1000 launched pids=$(cat "$OUT/pids/collect_gpu0.pid") $(cat "$OUT/pids/collect_gpu1.pid") $(cat "$OUT/pids/collect_gpu2.pid")" | tee -a "$OUT/collect_1000.log"
