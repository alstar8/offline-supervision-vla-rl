#!/usr/bin/env bash
# Launch one online-RL arm with the PPO step size actually bounded.
#
# What the first probe established (logs/probe_online_rl.log):
#   - The online pipeline is correct. Rollout SR from the unmodified SFT policy
#     averaged 13.3% over four frozen-actor updates against 15.0% measured
#     offline on 60 episodes, so env, observations, action decoding and reward
#     all reproduce offline behaviour.
#   - Reward discriminates between rollouts (adv_std 0.61-0.70).
#   - The critic learns (value_loss 0.064 -> 0.0063) but explained_variance was
#     still negative (-0.49) when the actor was released after 3 updates.
#   - The first live actor update measured clipfrac=0.55, approx_kl=1.16,
#     ratio=1.51. That is a ~20x oversized PPO step.
#
# Fixes applied here, all four aimed at that single failure:
#   alg_target_kl=0.05  the epoch loop had NO approx_kl early stop at all, so
#                       ~22 optimizer steps per epoch ran with only the clip
#                       bounding them. This is the structural fix.
#   alg_ppo_epoch=1     the launcher was overriding the code default of 1 to 3,
#                       tripling off-policy reuse per rollout.
#   vla_lr=2e-5         1e-4 is high for PPO on a 7B LoRA; the gradient norm was
#                       7.93 against a clip threshold of 10.
#   freeze_actor=8      3 critic-only updates left explained_variance negative,
#                       so advantages were built on a critic worse than
#                       predicting the mean of returns.
#
# Arms:
#   ppo_fixed    BC off, KL 0.001. Isolates PPO: can reward alone improve on the
#                15% SFT policy once the step is bounded?
#   refkl_fixed  BC 0.6, KL 0.008. The production recipe plus the fix; this is
#                the configuration that would actually be used.
#
#   ARM=ppo_fixed GPUS=0,1,2 bash run_rl_arm.sh
set -euo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_refkl"
SFT_RUN="${REPO}/openvla/checkpoints/sft/openvla_v2_3var_fixedlabels/steps_8000-no_aug"

ARM="${ARM:?set ARM to ppo_fixed or refkl_fixed}"
GPUS="${GPUS:?set GPUS, e.g. 0,1,2}"
# Separate run identity from the hyperparameter set, so a short smoke variant of
# an arm can run alongside the real one without overwriting its logs.
TAG="${TAG:-${ARM}}"

# Best v6 checkpoint by 60-episode closed-loop SR (15.0% SR, 20.0% grasp).
INIT="${INIT:-${SFT_RUN}/lora_004000}"
DATASETS="${DATASETS:-${RUN_DIR}/datasets_v6}"

NUM_ENVS="${NUM_ENVS:-32}"
EPISODE_LEN="${EPISODE_LEN:-112}"
# Local env-step budget: 32*112 = 3584 per update, so ~25 updates
# (8 critic-only warmup + ~17 live-actor).
STEPS_MAX="${STEPS_MAX:-90000}"

# --- the fixes, shared by both arms ---
TARGET_KL="${TARGET_KL:-0.05}"
PPO_EPOCH="${PPO_EPOCH:-1}"
LR="${LR:-2e-5}"
FREEZE_ACTOR="${FREEZE_ACTOR:-8}"

case "${ARM}" in
  ppo_fixed)
    BC_ENABLED=0
    KL_COEF="${KL_COEF:-0.001}"
    ;;
  refkl_fixed)
    BC_ENABLED=1
    KL_COEF="${KL_COEF:-0.008}"
    ;;
  *)
    echo "unknown ARM '${ARM}' (expected ppo_fixed or refkl_fixed)" >&2
    exit 1
    ;;
esac

# Distinct rendezvous port per arm so two arms can run side by side.
case "${ARM}" in
  ppo_fixed)   PORT="${MASTER_PORT:-29871}" ;;
  refkl_fixed) PORT="${MASTER_PORT:-29881}" ;;
esac

for f in adapter_model.safetensors dataset_statistics.json; do
  if [[ ! -f "${INIT}/${f}" ]]; then
    echo "init checkpoint missing ${f}: ${INIT}" >&2
    exit 1
  fi
done

LOG="${RUN_DIR}/logs/rl_${TAG}.log"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/pids"

echo "[${TAG}] init=${INIT}"
echo "[${TAG}] gpus=${GPUS} num_envs=${NUM_ENVS} episode_len=${EPISODE_LEN} steps_max=${STEPS_MAX}"
echo "[${TAG}] target_kl=${TARGET_KL} ppo_epoch=${PPO_EPOCH} lr=${LR} freeze_actor=${FREEZE_ACTOR}"
echo "[${TAG}] bc_enabled=${BC_ENABLED} kl_coef=${KL_COEF}"
echo "[${TAG}] updates ~ $((STEPS_MAX / (NUM_ENVS * EPISODE_LEN))) log=${LOG}"

SFT_LORA_PATH="${INIT}" \
REFKL_LOAD_PATH="${INIT}" \
REFKL_GPUS="${GPUS}" \
NUM_ENVS="${NUM_ENVS}" \
BUFFER_INFERBATCH="${BUFFER_INFERBATCH:-8}" \
EPISODE_LEN="${EPISODE_LEN}" \
STEPS_MAX="${STEPS_MAX}" \
ALG_TARGET_KL="${TARGET_KL}" \
ALG_PPO_EPOCH="${PPO_EPOCH}" \
VLA_LR="${LR}" \
FREEZE_ACTOR_UPDATES="${FREEZE_ACTOR}" \
KL_TO_REF_ENABLED=1 \
KL_TO_REF_COEF="${KL_COEF}" \
BC_TO_REF_ENABLED="${BC_ENABLED}" \
SFT_DATA_ROOT_DIR="${DATASETS}" \
SFT_SKIP_IMAGE_RESIZE=1 \
SFT_ACTION_DIM_WEIGHTS="${SFT_ACTION_DIM_WEIGHTS:-3:3:1.5:0.2:0.2:0.2:1}" \
SFT_SHUFFLE_BUFFER_SIZE=4000 \
REFKL_JOB_DIR="${RUN_DIR}/ddp_${TAG}" \
LOG="${LOG}" \
PID_FILE="${RUN_DIR}/pids/rl_${TAG}.pid" \
MASTER_PORT="${PORT}" \
NAME="RL_${TAG}_v6" \
bash "${RUN_DIR}/launch_gpus_v3.sh"
