from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import tyro
from transforms3d.axangles import axangle2mat

from mani_skill.envs.sapien_env import BaseEnv
from real2sim.calibrate_rc5_pose import DEFAULT_LOAD_STATE_FILE, _load_state_file
from real2sim.debug_paths import GRID_ALIGNMENT_DIR
from real2sim.openreal2sim_validation import (
    AIRI_TABLE_EMPTY_ASSET_DIR,
    RC5AeroHandRightOpenReal2SimValidation,
    default_spawn_grid_state,
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


@dataclass
class Args:
    output_dir: str = str(GRID_ALIGNMENT_DIR)
    image_prefix: str = "openreal2sim_spawn_grid"
    scene_asset_dir: str = str(AIRI_TABLE_EMPTY_ASSET_DIR)
    sim_backend: str = "gpu"
    shader: str = "default"
    observation_camera_mode: str = "manual_best"
    load_background: bool = True
    use_probe_objects: bool = False
    show_debug_markers: bool = False
    robot_far_away: bool = False
    safe_robot_during_spawn: bool = False
    load_state_file: str = DEFAULT_LOAD_STATE_FILE
    seed: int = 0


def _build_env(args: Args) -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": observation_camera_override(
                args.observation_camera_mode,
                scene_asset_dir=args.scene_asset_dir,
            ),
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        scene_asset_dir=args.scene_asset_dir,
        robot_uids=RC5AeroHandRightOpenReal2SimValidation.uid,
    )


def _save_snapshot(
    env: BaseEnv,
    args: Args,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    yaw_deg: float,
    z_value: float,
    index: int,
) -> None:
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    obs, _ = env.reset(
        seed=args.seed,
        options={
            "reconfigure": True,
            "load_background": args.load_background,
            "use_probe_objects": args.use_probe_objects,
            "show_debug_markers": args.show_debug_markers,
            "robot_far_away": args.robot_far_away,
            "safe_robot_during_spawn": args.safe_robot_during_spawn,
            "show_spawn_grid": True,
            "spawn_grid_center_xy": center_xy.tolist(),
            "spawn_grid_size_xy": size_xy.tolist(),
            "spawn_grid_yaw_deg": float(yaw_deg),
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
        },
    )
    frame = _extract_obs_frame(obs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.image_prefix}_{index:03d}.png"
    latest_path = output_dir / f"{args.image_prefix}_latest.png"
    meta_path = output_dir / f"{args.image_prefix}_{index:03d}.txt"
    latest_meta_path = output_dir / f"{args.image_prefix}_latest.txt"

    x_half = float(size_xy[0] / 2.0)
    y_half = float(size_xy[1] / 2.0)
    rot = axangle2mat([0.0, 0.0, 1.0], np.deg2rad(float(yaw_deg)))
    base_corners = np.array(
        [
            [-x_half, -y_half, 0.0],
            [x_half, -y_half, 0.0],
            [x_half, y_half, 0.0],
            [-x_half, y_half, 0.0],
        ],
        dtype=np.float64,
    )
    corners = []
    for corner in base_corners:
        rotated = rot @ corner
        corners.append(
            [
                float(center_xy[0] + rotated[0]),
                float(center_xy[1] + rotated[1]),
                z_value,
            ]
        )

    iio.imwrite(image_path, frame)
    iio.imwrite(latest_path, frame)
    meta_text = "\n".join(
        [
            f"center_xy={center_xy.tolist()}",
            f"size_xy={size_xy.tolist()}",
            f"yaw_deg={yaw_deg}",
            f"z={z_value}",
            f"corners={corners}",
        ]
    )
    meta_path.write_text(meta_text, encoding="utf-8")
    latest_meta_path.write_text(meta_text, encoding="utf-8")
    print(f"Saved {image_path}")
    print(f"Latest image {latest_path}")
    print(f"Saved {meta_path}")
    print(f"Latest meta {latest_meta_path}")


def _print_help() -> None:
    print("Commands:")
    print("  move dx dy           move grid center in XY")
    print("  z delta              change Z by delta")
    print("  x delta              change X size by delta")
    print("  y delta              change Y size by delta")
    print("  size dx dy           change X and Y sizes by deltas")
    print("  center x y           set absolute XY center")
    print("  dims x y             set absolute X/Y sizes")
    print("  yaw deg              add yaw in degrees")
    print("  rotate deg           alias for yaw")
    print("  angle deg            set absolute yaw in degrees")
    print("  height z             set absolute Z")
    print("  show                 print current center/size/yaw/z")
    print("  reset                reset to default grid guess")
    print("  help                 show this message")
    print("  quit                 exit")


def main(args: Args):
    default_state = default_spawn_grid_state(args.scene_asset_dir)
    center_xy = np.array(default_state["center_xy"], dtype=np.float32)
    size_xy = np.array(default_state["size_xy"], dtype=np.float32)
    yaw_deg = float(default_state["yaw_deg"][0])
    z_value = float(default_state["z"][0])

    env = _build_env(args)
    try:
        save_idx = 0
        _save_snapshot(env, args, center_xy, size_xy, yaw_deg, z_value, save_idx)
        _print_help()

        while True:
            raw = input("grid> ").strip()
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()
            try:
                if cmd in {"quit", "exit", "q"}:
                    break
                if cmd == "help":
                    _print_help()
                    continue
                if cmd == "show":
                    print(f"center_xy={center_xy.tolist()}")
                    print(f"size_xy={size_xy.tolist()}")
                    print(f"yaw_deg={yaw_deg}")
                    print(f"z={z_value}")
                    continue
                if cmd == "reset":
                    default_state = default_spawn_grid_state(args.scene_asset_dir)
                    center_xy = np.array(default_state["center_xy"], dtype=np.float32)
                    size_xy = np.array(default_state["size_xy"], dtype=np.float32)
                    yaw_deg = float(default_state["yaw_deg"][0])
                elif cmd == "move":
                    center_xy += np.array([float(parts[1]), float(parts[2])], dtype=np.float32)
                elif cmd == "z":
                    z_value += float(parts[1])
                elif cmd == "x":
                    size_xy[0] = max(0.005, float(size_xy[0] + float(parts[1])))
                elif cmd == "y":
                    size_xy[1] = max(0.005, float(size_xy[1] + float(parts[1])))
                elif cmd == "size":
                    size_xy += np.array([float(parts[1]), float(parts[2])], dtype=np.float32)
                    size_xy = np.maximum(size_xy, 0.005)
                elif cmd == "center":
                    center_xy = np.array([float(parts[1]), float(parts[2])], dtype=np.float32)
                elif cmd == "dims":
                    size_xy = np.maximum(
                        np.array([float(parts[1]), float(parts[2])], dtype=np.float32),
                        0.005,
                    )
                elif cmd in {"yaw", "rotate"}:
                    yaw_deg += float(parts[1])
                elif cmd == "angle":
                    yaw_deg = float(parts[1])
                elif cmd == "height":
                    z_value = float(parts[1])
                else:
                    print("Unknown command. Type 'help'.")
                    continue
            except (IndexError, ValueError) as exc:
                print(f"Bad command: {exc}")
                continue

            print(f"center_xy={center_xy.tolist()}")
            print(f"size_xy={size_xy.tolist()}")
            print(f"yaw_deg={yaw_deg}")
            print(f"z={z_value}")
            save_idx += 1
            _save_snapshot(env, args, center_xy, size_xy, yaw_deg, z_value, save_idx)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
