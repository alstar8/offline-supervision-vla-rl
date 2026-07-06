from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import tyro

from mani_skill.envs.sapien_env import BaseEnv
from real2sim.openreal2sim_validation import (
    observation_camera_lookat_state,
    pose_from_eye_target_roll,
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


def _build_env(args, pose, fov: float) -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": {
                "pose": pose,
                "intrinsic": None,
                "fov": fov,
                "near": 0.01,
                "far": 10.0,
            },
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
    )


def _reset_env(
    args,
    env: BaseEnv | None,
    eye: np.ndarray,
    target: np.ndarray,
    fov: float,
    roll_deg: float,
) -> BaseEnv:
    pose = pose_from_eye_target_roll(eye=eye, target=target, roll_deg=roll_deg)
    if env is not None:
        env.close()
    env = _build_env(args, pose, fov)
    env.reset(
        seed=0,
        options={
            "reconfigure": True,
            "use_probe_objects": args.use_probe_objects,
            "load_background": args.load_background,
            "show_debug_markers": args.show_debug_markers,
            "robot_far_away": args.robot_far_away,
        },
    )
    return env


def _save_snapshot(
    env: BaseEnv,
    args,
    eye: np.ndarray,
    target: np.ndarray,
    fov: float,
    roll_deg: float,
    index: int,
) -> BaseEnv:
    env = _reset_env(args, env, eye, target, fov, roll_deg)
    obs = env.get_obs()
    frame = _extract_obs_frame(obs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.image_prefix}_{index:03d}.png"
    latest_path = output_dir / f"{args.image_prefix}_latest.png"
    meta_path = output_dir / f"{args.image_prefix}_{index:03d}.txt"
    iio.imwrite(image_path, frame)
    iio.imwrite(latest_path, frame)
    meta_path.write_text(
        "\n".join(
            [
                f"eye={eye.tolist()}",
                f"target={target.tolist()}",
                f"fov={float(fov)}",
                f"roll_deg={float(roll_deg)}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved {image_path}")
    print(f"Saved {meta_path}")
    return env


def _print_help() -> None:
    print("Commands:")
    print("  save                 save current view as PNG + TXT")
    print("  show                 print current eye/target/fov")
    print("  reset                reset to start preset")
    print("  target dx dy dz      move camera target in world coordinates")
    print("  yaw deg              orbit target around eye in world Z")
    print("  pitch deg            move target up/down around eye")
    print("  roll deg             roll around the current viewing direction")
    print("  fov delta            add delta to fov")
    print("  preset NAME          switch to a known preset name")
    print("  help                 show this message")
    print("  quit                 exit")


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    theta = np.deg2rad(degrees)
    return (
        vec * np.cos(theta)
        + np.cross(axis, vec) * np.sin(theta)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(theta))
    )


@dataclass
class Args:
    output_dir: str = "real2sim/calibration"
    image_prefix: str = "openreal2sim_view"
    sim_backend: str = "gpu"
    shader: str = "default"
    use_probe_objects: bool = False
    load_background: bool = True
    show_debug_markers: bool = False
    robot_far_away: bool = True
    start_mode: str = "manual_best"
    start_fov: float = 1.0
    auto_save: bool = False


def main(args: Args):
    state = observation_camera_lookat_state(args.start_mode)
    eye = np.array(state["eye"], dtype=np.float32)
    target = np.array(state["target"], dtype=np.float32)
    fov = float(args.start_fov if args.start_fov > 0 else state["fov"])
    roll_deg = float(state.get("roll_deg", 0.0))

    env = None
    try:
        env = _reset_env(args, env, eye, target, fov, roll_deg)
        save_idx = 0
        env = _save_snapshot(env, args, eye, target, fov, roll_deg, save_idx)
        _print_help()

        while True:
            raw = input("camera> ").strip()
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
                    print(f"eye={eye.tolist()}")
                    print(f"target={target.tolist()}")
                    print(f"fov={fov}")
                    print(f"roll_deg={roll_deg}")
                    continue
                if cmd == "reset":
                    state = observation_camera_lookat_state(args.start_mode)
                    eye = np.array(state["eye"], dtype=np.float32)
                    target = np.array(state["target"], dtype=np.float32)
                    fov = float(args.start_fov if args.start_fov > 0 else state["fov"])
                    roll_deg = float(state.get("roll_deg", 0.0))
                elif cmd == "preset":
                    state = observation_camera_lookat_state(parts[1])
                    eye = np.array(state["eye"], dtype=np.float32)
                    target = np.array(state["target"], dtype=np.float32)
                    fov = float(state["fov"])
                    roll_deg = float(state.get("roll_deg", 0.0))
                elif cmd == "save":
                    save_idx += 1
                    env = _save_snapshot(env, args, eye, target, fov, roll_deg, save_idx)
                    continue
                elif cmd == "target":
                    target += np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                elif cmd == "yaw":
                    offset = target - eye
                    offset = _rotate_vector(offset, np.array([0.0, 0.0, 1.0], dtype=np.float32), float(parts[1]))
                    target = eye + offset
                elif cmd == "pitch":
                    offset = target - eye
                    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    right = np.cross(offset, world_up)
                    if np.linalg.norm(right) < 1e-8:
                        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    offset = _rotate_vector(offset, right, float(parts[1]))
                    target = eye + offset
                elif cmd == "roll":
                    roll_deg += float(parts[1])
                elif cmd == "fov":
                    fov += float(parts[1])
                else:
                    print("Unknown command. Type 'help'.")
                    continue
            except (IndexError, ValueError) as exc:
                print(f"Bad command: {exc}")
                continue

            print(f"eye={eye.tolist()}")
            print(f"target={target.tolist()}")
            print(f"fov={fov}")
            print(f"roll_deg={roll_deg}")
            if args.auto_save:
                save_idx += 1
                env = _save_snapshot(env, args, eye, target, fov, roll_deg, save_idx)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
