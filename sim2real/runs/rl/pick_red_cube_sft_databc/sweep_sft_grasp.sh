#!/usr/bin/env bash
# Eval-only sweep that ranks SFT checkpoints by simulator grasp rate.
#
# PPO here refines a policy that already grasps; it has never produced grasping
# from scratch. Gate DataBC launches on grasp rate from this sweep rather than on
# SFT token accuracy.
#
# usage: ./sweep_sft_grasp.sh [ckpt_dir ...]
#        EPISODE_LEN=160 NUM_ENVS=64 ./sweep_sft_grasp.sh
set -uo pipefail

REPO="/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl"
CONDA_BIN="/workspace-SR008.nfs2/users/staroverov/.conda/envs/rl/bin"
RUN_DIR="${REPO}/sim2real/runs/rl/pick_red_cube_sft_databc"
SFT_10K="${REPO}/openvla/checkpoints/sft/openvla_v2_scene14sep26_10k_xy/steps_16000-no_aug"

EPISODE_LEN="${EPISODE_LEN:-160}"
NUM_ENVS="${NUM_ENVS:-64}"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/sweep_grasp}"

if [[ $# -gt 0 ]]; then
  CKPTS=("$@")
else
  # Most-trained first so the go/no-go signal arrives before the whole sweep ends;
  # the old 500-step init is last purely as a baseline.
  CKPTS=(
    "${SFT_10K}/lora_008000"
    "${SFT_10K}/lora_004000"
    "${SFT_10K}/lora_002000"
    "${RUN_DIR}/sft_init/lora_000500"
  )
fi

export PATH="${CONDA_BIN}:${PATH}"
export HF_HOME="/workspace-SR008.nfs2/users/staroverov/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# rl_gym.py defaults to ..._left_image_metric (release.v4, scaled s=1.0915 about the camera
# centre). The SFT data is rendered on the unscaled left_image scene, so the metric scene is
# out of distribution: a run on it collapsed to 0.0% grasp / 0.0% SR for 7 straight updates
# against 25.0% / 10.4% here. Pin the source scene.
export OPENREAL2SIM_RL_SCENE_KEY="${OPENREAL2SIM_RL_SCENE_KEY:-airy_table_scene14sep26_left_image}"
# Do not inherit PYTHONPATH: a leading sim2real/ entry shadows ManiSkill and drops RC5.
export PYTHONPATH="${REPO}/SimplerEnv:${REPO}/ManiSkill:${REPO}/real2sim:${REPO}/openvla"

mkdir -p "${OUT_ROOT}"
cd "${REPO}/SimplerEnv"

echo "grasp sweep start $(date -Is) | episode_len=${EPISODE_LEN} num_envs=${NUM_ENVS} | ${#CKPTS[@]} checkpoint(s)"

for CKPT in "${CKPTS[@]}"; do
  if [[ ! -f "${CKPT}/dataset_statistics.json" ]]; then
    echo "SKIP ${CKPT} (no dataset_statistics.json)"
    continue
  fi
  # Tag as <parent>-<lora dir> so 10k/lora_002000 and sft_init/lora_000500 stay distinct.
  TAG="$(basename "$(dirname "${CKPT}")")-$(basename "${CKPT}")"
  OUT_DIR="${OUT_ROOT}/${TAG}"
  mkdir -p "${OUT_DIR}"

  export DATABC_CKPT_DIR="${OUT_DIR}"
  export WANDB_DIR="${OUT_DIR}"

  echo "--- ${TAG} -> ${OUT_DIR}"
  "${CONDA_BIN}/python" simpler_env/train_ms3_ppo_sft.py \
    --name="GraspSweep_${TAG}" \
    --env_id=OpenReal2Sim-v0 \
    --vla_path=gen-robot/openvla-7b-rlvla-warmup \
    --vla_load_path="${CKPT}" \
    --vla_unnorm_key=sft_v2 \
    --vla_model_variant=v2 \
    --vla_proprio_dim=7 \
    --seed=0 \
    --num_envs="${NUM_ENVS}" \
    --episode_len="${EPISODE_LEN}" \
    --store-rollouts-on-cpu \
    --use_wrist_camera \
    --only_render \
    --render_info \
    --vla_temperature=1.0 \
    --vla_temperature_anneal_steps=0 \
    --eval_at_train_temperature \
    --reward_reach_coef=0.3 \
    --reward_reach_clip=0.4 \
    --reward_max_lift_height=0 \
    --reward_lift_coef=0.5 \
    --reward_lift_height=0.05 \
    --reward_yeet_height=0.15 \
    --reward_yeet_coef=2.0 \
    --reward_yeet_clip="${REWARD_YEET_CLIP:-0.1}" \
    --escape_height="${ESCAPE_HEIGHT:-0.5}" \
    --escape_below="${ESCAPE_BELOW:-0.05}" \
    --escape_dist="${ESCAPE_DIST:-1.0}" \
    --reward_escape_penalty="${REWARD_ESCAPE_PENALTY:-1.0}" \
    --success_terminate_steps=5 \
    --sticky_gripper_steps 0 \
    --max_ee_delta=0.05 \
    >"${OUT_DIR}/render.log" 2>&1
  RC=$?
  if [[ ${RC} -ne 0 ]]; then
    echo "    FAILED rc=${RC}, see ${OUT_DIR}/render.log"
  else
    echo "    done"
  fi
done

echo
"${CONDA_BIN}/python" "${RUN_DIR}/summarize_grasp_sweep.py" "${OUT_ROOT}"
echo "grasp sweep done $(date -Is)"
