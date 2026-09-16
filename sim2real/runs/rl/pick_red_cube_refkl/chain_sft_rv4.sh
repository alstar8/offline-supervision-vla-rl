#!/usr/bin/env bash
# Retrain SFT on the release.v4-retargeted labels once the pre-v4 re-score frees
# its GPUs. Same hyperparameters as the v6 run so the two are comparable; only
# the dataset (frame-retargeted) and the run root differ.
set -euo pipefail
REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
REFKL="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
DATABC="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"

while ! grep -q "RESCORE_ALL_DONE" "${REFKL}/logs/rescore120.log" 2>/dev/null; do sleep 60; done
echo "[chain] rescore finished, waiting for GPUs to drain $(date -Is)"
while [ "$(pgrep -cf '[t]rain_ms3_ppo_sft.py')" -gt 0 ]; do sleep 30; done
sleep 30
echo "[chain] launching SFT on retargeted data $(date -Is)"

SFT_RUN_ROOT="checkpoints/sft/openvla_v2_rv4frame" \
MAX_STEPS=8000 \
SAVE_STEPS="1000,2000,3000,4000,5000,6000,7000,8000" \
EVAL_STEPS=500 \
NPROC=6 \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" \
DATA_ROOT_DIR="${REFKL}/datasets_v6_rv4" \
SKIP_IMAGE_RESIZE=1 \
ACTION_DIM_LOSS_WEIGHTS="3:3:1.5:0.2:0.2:0.2:1" \
SHUFFLE_BUFFER_SIZE=16000 \
bash "${DATABC}/train_sft_v3.sh"
echo "[chain] SFT exited rc=$? $(date -Is)"
