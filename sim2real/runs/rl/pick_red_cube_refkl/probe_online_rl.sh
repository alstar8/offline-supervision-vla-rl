#!/usr/bin/env bash
# Online RL plumbing probe on GPUs 6/7: does PPO learn at all, and if not, which
# stage is wrong -- reward, advantages, gradient flow, or PPO step size?
#
# Deliberately minimal so that any result is attributable:
#   - BC-to-reference OFF. At the production bc_coef=0.6 the actor gradient is
#     dominated by supervised imitation, so a reward curve that moves would
#     prove nothing about PPO.
#   - KL-to-reference at 0.001 rather than 0.008: enough to keep the policy on
#     the action-token manifold, too weak to pin it to a checkpoint that only
#     scores ~19%.
#   - Init from the best v6 SFT checkpoint by closed-loop SR. PPO has to
#     bootstrap from real successes; from a 0% policy every rollout returns
#     nearly the same shaped reward and the run is uninformative.
#   - alg_ppo_epoch left at the production value of 3. The goal is to observe
#     whether that setting is itself harmful (now visible via clipfrac and
#     approx_kl), not to silently pre-fix it.
#
# Read logs/probe_online_rl.log in this order:
#   1. First 3 updates run actor=frozen (critic warmup). explained_var should
#      climb off zero. If it stays ~0, the value head or the reward signal is
#      broken and nothing downstream is worth interpreting.
#   2. actor grad_norm once actor=live: must be nonzero. Zero or n/a means the
#      LoRA parameters are not receiving gradient.
#   3. adv_std: if ~0 the reward does not discriminate between rollouts, so the
#      policy gradient carries no information regardless of PPO settings.
#   4. clipfrac / approx_kl: ~0 means the steps are too small to change
#      anything; a sustained clipfrac above ~0.3 means the step is too large and
#      clipping is the only thing bounding it.
#   5. rollout success across updates: the actual outcome.
#
# Note: periodic closed-loop eval is skipped under DDP (do_eval is gated on
# `not ddp`), so rollout env/success is the only success signal here.
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_RUN="${REPO}/openvla/checkpoints/sft/openvla_v2_3var_fixedlabels/steps_8000-no_aug"

# Best v6 checkpoint by 16-episode closed-loop SR (18.8% SR, 25% grasp).
INIT="${INIT:-${SFT_RUN}/lora_004000}"
GPUS="${GPUS:-6,7}"
# 32 envs/GPU instead of the production 64: halves rollout time per update so the
# probe yields ~16 data points in an afternoon instead of ~8.
NUM_ENVS="${NUM_ENVS:-32}"
# 112 covers the p99 of demo-paced first success (chunk 108) under v6 chunking.
EPISODE_LEN="${EPISODE_LEN:-112}"
# Local env-step budget: 32*112 = 3584 per update, so ~16 updates.
STEPS_MAX="${STEPS_MAX:-60000}"

KL_COEF="${KL_COEF:-0.001}"
BC_ENABLED="${BC_ENABLED:-0}"
PPO_EPOCH="${PPO_EPOCH:-3}"
FREEZE_ACTOR="${FREEZE_ACTOR:-3}"

if [[ ! -f "${INIT}/adapter_model.safetensors" ]]; then
  echo "init checkpoint has no weights: ${INIT}" >&2
  exit 1
fi
if [[ ! -f "${INIT}/dataset_statistics.json" ]]; then
  echo "init checkpoint has no dataset_statistics.json: ${INIT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/pids"

echo "[probe] init=${INIT}"
echo "[probe] gpus=${GPUS} num_envs=${NUM_ENVS} episode_len=${EPISODE_LEN} steps_max=${STEPS_MAX}"
echo "[probe] kl_coef=${KL_COEF} bc_enabled=${BC_ENABLED} ppo_epoch=${PPO_EPOCH} freeze_actor=${FREEZE_ACTOR}"
echo "[probe] updates ~ $((STEPS_MAX / (NUM_ENVS * EPISODE_LEN)))"

SFT_LORA_PATH="${INIT}" \
REFKL_LOAD_PATH="${INIT}" \
REFKL_GPUS="${GPUS}" \
NUM_ENVS="${NUM_ENVS}" \
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-8}" \
EPISODE_LEN="${EPISODE_LEN}" \
STEPS_MAX="${STEPS_MAX}" \
KL_TO_REF_ENABLED=1 \
KL_TO_REF_COEF="${KL_COEF}" \
BC_TO_REF_ENABLED="${BC_ENABLED}" \
SFT_SKIP_IMAGE_RESIZE=1 \
ALG_PPO_EPOCH="${PPO_EPOCH}" \
FREEZE_ACTOR_UPDATES="${FREEZE_ACTOR}" \
REFKL_JOB_DIR="${RUN_DIR}/ddp_v6_probe" \
LOG="${RUN_DIR}/logs/probe_online_rl.log" \
PID_FILE="${RUN_DIR}/pids/probe_online_rl.pid" \
MASTER_PORT="${MASTER_PORT:-29861}" \
NAME="${NAME:-PPO_probe_v6}" \
bash "${RUN_DIR}/launch_gpus_v3.sh"
