#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from PIL import Image

from mani_skill.utils.visualization.misc import images_to_video


CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


EXAMPLES = [
    {
        "name": "01_ind_base_main_train",
        "env_id": "PutOnPlateInScene25Main-v3",
        "obj_set": "train",
        "caption": "IND base task (standard object/scene)",
    },
    {
        "name": "02_ood_visual_texture",
        "env_id": "PutOnPlateInScene25VisionTexture05-v1",
        "obj_set": "test",
        "caption": "OOD visual shift (texture/appearance)",
    },
    {
        "name": "03_ood_multi_receptacle",
        "env_id": "PutOnPlateInScene25MultiPlate-v1",
        "obj_set": "test",
        "caption": "OOD multiple receptacles (distractor receptacles)",
    },
]


def _extract_frame(obs: Any) -> np.ndarray:
    rgb = obs["sensor_data"]["3rd_view_camera"]["rgb"]
    # Expected shape: [B, H, W, 3]. We use first environment only.
    if hasattr(rgb, "detach"):
        frame = rgb[0].detach().cpu().numpy()
    else:
        frame = np.asarray(rgb[0])
    return frame.astype(np.uint8)


def _extract_instruction(env: Any) -> str:
    instruction = env.unwrapped.get_language_instruction()
    if isinstance(instruction, list):
        return str(instruction[0])
    return str(instruction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small IND/OOD visual showcase: one PNG and one short video "
            "per example from the first reset frame."
        )
    )
    parser.add_argument(
        "--out-dir",
        default="SimplerEnv/scripts/ood_showcase",
        help="Output directory for PNG/video/metadata.",
    )
    parser.add_argument(
        "--sim-backend",
        default="gpu",
        choices=["gpu", "cpu"],
        help="ManiSkill simulation backend.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for ex in EXAMPLES:
        env = gym.make(
            id=ex["env_id"],
            num_envs=1,
            obs_mode="rgb+segmentation",
            control_mode=CONTROL_MODE,
            sim_backend=args.sim_backend,
            sim_config={"sim_freq": 500, "control_freq": 5},
            max_episode_steps=80,
            sensor_configs={"shader_pack": "default"},
        )

        obs, info = env.reset(seed=[args.seed], options={"obj_set": ex["obj_set"]})
        frame = _extract_frame(obs)
        instruction = _extract_instruction(env)
        env.close()

        ex_dir = out_dir / ex["name"]
        ex_dir.mkdir(parents=True, exist_ok=True)

        png_path = ex_dir / "frame0.png"
        Image.fromarray(frame).save(png_path)

        clip_frames = [frame for _ in range(max(1, args.video_frames))]
        images_to_video(
            clip_frames,
            str(ex_dir),
            "first_frame_clip",
            fps=args.fps,
            verbose=False,
        )

        meta = {
            "name": ex["name"],
            "caption": ex["caption"],
            "env_id": ex["env_id"],
            "obj_set": ex["obj_set"],
            "instruction": instruction,
            "png": str(png_path),
            "video": str(ex_dir / "first_frame_clip.mp4"),
        }
        (ex_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        summary.append(meta)
        print(f"[saved] {ex['name']} -> {png_path}")

    summary_path = out_dir / "summary.yaml"
    summary_path.write_text(yaml.safe_dump({"examples": summary}, sort_keys=False))
    print(f"[done] Wrote showcase assets to: {out_dir}")
    print(f"[done] Summary: {summary_path}")


if __name__ == "__main__":
    main()
