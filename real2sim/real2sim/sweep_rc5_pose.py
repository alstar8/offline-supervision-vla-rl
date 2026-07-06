from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import tyro

from mani_skill.envs.sapien_env import BaseEnv
from real2sim.debug_paths import RC5_POSE_SWEEP_DIR
from real2sim.openreal2sim_validation import (
    PROBE_CARROT_POSE_P,
    PROBE_CARROT_POSE_Q,
    PROBE_PLATE_POSE_P,
    PROBE_PLATE_POSE_Q,
    ROBOT_BASE_POSE_P,
    ROBOT_BASE_POSE_Q,
    observation_camera_override,
)


def _to_uint8_image(frame) -> np.ndarray:
    if torch.is_tensor(frame):
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 3:
            if frame.dtype.is_floating_point:
                max_val = float(frame.max().item())
                if max_val <= 1.0 + 1e-6:
                    frame = frame * 255.0
            return frame.clamp(0, 255).to(torch.uint8).cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(arr.max()) if arr.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _extract_obs_frame(obs: dict) -> np.ndarray:
    sensor_data = obs.get("sensor_data", {})
    if "3rd_view_camera" not in sensor_data:
        available = sorted(sensor_data.keys())
        raise KeyError(
            "Expected observation camera '3rd_view_camera' was not found. "
            f"Available sensor_data keys: {available}"
        )
    return _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])


def _parse_csv_floats(raw: str) -> list[float]:
    vals = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(float(chunk))
    return vals


@dataclass
class Args:
    output_dir: str = str(RC5_POSE_SWEEP_DIR)
    image_prefix: str = "rc5_pose"
    sim_backend: str = "gpu"
    shader: str = "default"
    observation_camera_mode: str = "manual_best"
    seed: int = 0
    load_background: bool = True
    use_probe_objects: bool = True
    show_debug_markers: bool = False
    x_offsets: str = "-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30"
    y_offsets: str = "-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30"
    z_offsets: str = "0.00"


def _build_env(args: Args) -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": observation_camera_override(args.observation_camera_mode),
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids="rc5_aero_hand_right_openreal2sim_validation",
    )


def _default_rc5_qpos() -> np.ndarray:
    return np.array(
        [
            0.0,
            -0.10,
            0.72,
            0.0,
            -1.05,
            0.0,
            0.65,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _save_case(
    env: BaseEnv,
    args: Args,
    case_name: str,
    robot_base_pose_p: np.ndarray,
    robot_init_qpos: np.ndarray,
) -> None:
    obs, info = env.reset(
        seed=args.seed,
        options={
            "reconfigure": True,
            "load_background": args.load_background,
            "use_probe_objects": args.use_probe_objects,
            "show_debug_markers": args.show_debug_markers,
            "robot_far_away": False,
            "probe_carrot_pose_p": PROBE_CARROT_POSE_P,
            "probe_carrot_pose_q": PROBE_CARROT_POSE_Q,
            "probe_plate_pose_p": PROBE_PLATE_POSE_P,
            "probe_plate_pose_q": PROBE_PLATE_POSE_Q,
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": ROBOT_BASE_POSE_Q,
            "robot_init_qpos": robot_init_qpos.tolist(),
        },
    )
    frame = _extract_obs_frame(obs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.image_prefix}_{case_name}.png"
    meta_path = output_dir / f"{args.image_prefix}_{case_name}.txt"

    tcp_pose = obs["extra"]["tcp_pose"][0].cpu().numpy().tolist()
    info_summary = {
        k: (v[0].item() if torch.is_tensor(v) else v)
        for k, v in info.items()
    }

    iio.imwrite(image_path, frame)
    meta_path.write_text(
        "\n".join(
            [
                f"robot_base_pose_p={robot_base_pose_p.tolist()}",
                f"robot_base_pose_q={ROBOT_BASE_POSE_Q}",
                f"robot_init_qpos={robot_init_qpos.tolist()}",
                f"tcp_pose={tcp_pose}",
                f"info={info_summary}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved {image_path}")


def main(args: Args):
    base_pose = np.array(ROBOT_BASE_POSE_P, dtype=np.float32)
    robot_init_qpos = _default_rc5_qpos()
    x_offsets = _parse_csv_floats(args.x_offsets)
    y_offsets = _parse_csv_floats(args.y_offsets)
    z_offsets = _parse_csv_floats(args.z_offsets)

    env = _build_env(args)
    try:
        for dz in z_offsets:
            for dy in y_offsets:
                for dx in x_offsets:
                    robot_base_pose_p = base_pose + np.array([dx, dy, dz], dtype=np.float32)
                    case_name = f"dx_{dx:+.3f}_dy_{dy:+.3f}_dz_{dz:+.3f}".replace(".", "p")
                    _save_case(env, args, case_name, robot_base_pose_p, robot_init_qpos)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
