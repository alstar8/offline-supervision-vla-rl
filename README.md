# Offline Supervision for Reinforcement Learning in VLA Models

Code release for **Leveraging Offline Supervision for Efficient and Generalizable Reinforcement Learning in Large-Scale Vision-Language-Action Models**, accepted at the ICML 2026 workshop [Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning](https://icml.cc/).

**Project page:** [https://alstar8.github.io/offline-supervision-vla-rl/](https://alstar8.github.io/offline-supervision-vla-rl/)

This repository implements hybrid offline–online RL fine-tuning of OpenVLA LoRA adapters on the RL4VLA benchmark ([Liu et al., 2025](https://arxiv.org/abs/2505.19789)).

## Overview

We study whether offline supervision can be incorporated into PPO fine-tuning of vision–language–action (VLA) policies to improve **training efficiency** while preserving **out-of-distribution (OOD)** generalization.

| Method | Description | Entry point |
|--------|-------------|-------------|
| **SFT** | Supervised fine-tuning on offline demonstrations | `scripts/train_sft.sh` |
| **PPO** | Standard RL from OpenVLA-warmup | `scripts/train_ppo.sh` |
| **PPO SFT-init** | PPO initialized from the SFT LoRA | `scripts/train_ppo_sft_init.sh` |
| **RefKL** | PPO + KL penalty to a frozen SFT reference policy | `scripts/train_refkl.sh` |
| **DataBC** | PPO + behavior cloning on offline demonstrations | `scripts/train_databc.sh` |

Both guided variants use a curriculum on the auxiliary weight β: constant until 100k steps, linear decay to 0 by 300k, then pure PPO (see paper Section 4.4).

### Main result

RefKL at **1M environment steps** reaches comparable IND/OOD success to standard PPO at **2M steps**, using roughly half the online training budget.

## Repository layout

```
.
├── ManiSkill/          # Simulation environments and motion-planning data collection
├── SimplerEnv/         # RL training loop (PPO, RefKL, DataBC) and evaluation
├── openvla/            # OpenVLA SFT scripts and model code
├── real2sim/           # Shared utilities required by SimplerEnv (constants, presets)
├── scripts/            # Paper experiment launchers
├── environment.yml     # Conda environment (CUDA 12.1 / PyTorch 2.2)
└── install_env.sh      # One-shot environment setup
```

## Installation

### 1. Training environment

```bash
git clone https://github.com/alstar8/offline-supervision-vla-rl.git
cd offline-supervision-vla-rl

# Optional: install flash-attn from a prebuilt wheel
bash ./install_env.sh --flash-attn-wheel /path/to/flash_attn.whl

conda activate rlvla-guided
```

Manual install:

```bash
conda create -n rlvla-guided python=3.10 -y
conda activate rlvla-guided
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla && pip install -e . && cd ..
pip install -U tyro datasets==3.3.2
cd ManiSkill && pip install -e . && cd ..
cd real2sim && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..
```

### 2. RLDS dataset builder (for warmup / SFT datasets)

```bash
cd openvla/rlds_dataset_builder
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```

## Data preparation

Experiments follow the RL4VLA protocol on `PutOnPlateInScene25Main-v3`.

1. **Collect demonstrations** with the ManiSkill motion planner:

```bash
conda activate rlvla-guided
cd ManiSkill
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Main-v3" \
  --save_video --save_data --num_procs 16 --num_traj 16400 --seed=100
```

2. **Build the SFT RLDS dataset**:

```bash
conda activate rlds_env
cd openvla/rlds_dataset_builder/sft_dataset
# expects demos at ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/8200/data
tfds build --overwrite
mkdir -p ../../../datasets
mv -T ~/tensorflow_datasets/example_dataset ../../../datasets/sft
```

For warmup, use the HuggingFace checkpoint [`gen-robot/openvla-7b-rlvla-warmup`](https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup) or train your own warmup LoRA following the RL4VLA data-collection protocol.

## Training

All RL scripts assume:
- base model: `gen-robot/openvla-7b-rlvla-warmup`
- environment: `PutOnPlateInScene25Main-v3`
- SFT reference LoRA (2k demos, step 7500): `openvla/checkpoints/sft/steps_60000-no_aug/lora_007500`

### SFT (reference policy)

```bash
bash scripts/train_sft.sh
```

Paper setting uses an early-stopped checkpoint at **7.5k steps** on **2k demonstrations**.

### Baselines

```bash
# Standard PPO from warmup
bash scripts/train_ppo.sh --seed 0

# PPO initialized from SFT LoRA
bash scripts/train_ppo_sft_init.sh --seed 0
```

### Guided PPO (paper methods)

```bash
# RefKL — reference-policy KL regularization
bash scripts/train_refkl.sh --seed 0

# DataBC — offline behavior cloning auxiliary loss
bash scripts/train_databc.sh --seed 0
```

### β ablation (constant β, no curriculum)

```bash
bash scripts/train_refkl_constant_beta.sh --seed 0
```

Pass extra arguments through to the underlying trainer, e.g. `--steps_max=1000000`.

## Evaluation

Evaluate a checkpoint on IND (`PutOnPlateInScene25Main-v3`) and all 13 OOD environments with three seeds:

```bash
CKPT_PATH="gen-robot/openvla-7b-rlvla-warmup" \
UNNORM_KEY="bridge_orig" \
VLA_LOAD_PATH="../SimplerEnv/wandb/<run>/glob/steps_XXXXXX" \
bash scripts/eval_policy.sh
```

Aggregate success rates:

```bash
cd SimplerEnv/scripts
python calc_statistics.py
```

Evaluation protocol (from the paper):
- 64 IND episodes per checkpoint
- 960 OOD episodes total (64 × 15 OOD settings)
- seeds `{0, 1, 2}` for evaluation

## Mapping to the paper

| Paper term | Code |
|------------|------|
| RefKL | `train_ms3_ppo_bc_teacher.py` with `--bc_to_ref_enabled` |
| DataBC | `train_ms3_ppo_sft.py` with `--bc_to_ref_enabled` |
| β curriculum | `--bc_to_ref_coef`, `--bc_to_ref_hold_steps=100000`, `--bc_to_ref_decay_steps=300000` |
| PPO SFT-init | `train_ms3_ppo.py` with `--vla_load_path` pointing to SFT LoRA |
| RL4VLA benchmark | ManiSkill `PutOnPlateInScene25*` tasks via SimplerEnv |

## Citation

If you use this code, please cite our paper and the RL4VLA benchmark:

```bibtex
@misc{poyarkov2026offline,
  title={Leveraging Offline Supervision for Efficient and Generalizable Reinforcement Learning in Large-Scale Vision-Language-Action Models},
  author={Poyarkov, Dmitriy and Staroverov, Aleksei and Panov, Aleksandr I.},
  howpublished={ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning},
  year={2026}
}

@article{liu2025what,
  title={What Can {RL} Bring to {VLA} Generalization? An Empirical Study},
  author={Liu, Xiao and others},
  journal={arXiv preprint arXiv:2505.19789},
  year={2025}
}
```

## Acknowledgments

We thank the authors of [OpenVLA](https://github.com/openvla/openvla), [ManiSkill](https://github.com/haosulab/ManiSkill), and [SimplerEnv](https://github.com/simpler-env/SimplerEnv) for their open-source contributions.

## License

MIT License — see [LICENSE](LICENSE).
