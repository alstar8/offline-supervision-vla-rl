#!/usr/bin/env python3
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import termios
import tty

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import sapien
import torch
import tyro
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.envs.sapien_env import BaseEnv
from real2sim.debug_paths import RC5_POSE_LIVE_DIR, RC5_POSE_LIVE_LATEST_STATE_FILE
from real2sim.openreal2sim_validation import observation_camera_lookat_state, pose_from_eye_target_roll


DEFAULT_BASE_POSE_P = np.array(
    [-7.450580707946131e-10, -1.950000524520874, 0.12047362327575684],
    dtype=np.float32,
)
DEFAULT_BASE_POSE_Q = np.array(
    [0.9921479225158691, 0.0, 0.0, 0.12506994605064392],
    dtype=np.float32,
)
DEFAULT_ROBOT_INIT_QPOS = np.array(
    [
        2.3253769874572754,
        -5.344996929168701,
        2.0028696060180664,
        -1.2623703479766846,
        -1.5767884254455566,
        0.05147823691368103,
        0.6499999761581421,
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
DEFAULT_LOAD_STATE_FILE = str(RC5_POSE_LIVE_LATEST_STATE_FILE)
DEFAULT_CAMERA_PRESET = "default"


@dataclass
class Args:
    output_dir: str = str(RC5_POSE_LIVE_DIR)
    image_prefix: str = "openreal2sim_rc5_live"
    keep_history: bool = False
    sim_backend: str = "gpu"
    shader: str = "default"
    seed: int = 0
    load_background: bool = True
    start_camera_mode: str = "manual_best"
    base_step: float = 0.02
    z_step: float = 0.01
    yaw_step_deg: float = 3.0
    joint_step_deg: float = 3.0
    joint_big_step_deg: float = 8.0
    far_distance_scale: float = 1.35
    orbit_angle_deg: float = 50.0
    load_state_file: str = DEFAULT_LOAD_STATE_FILE


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
        raise KeyError(
            "Expected observation camera '3rd_view_camera' was not found. "
            f"Available keys: {sorted(sensor_data.keys())}"
        )
    return _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])


def _read_single_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _load_state_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot_base_pose_p = DEFAULT_BASE_POSE_P.copy()
    robot_base_pose_q = DEFAULT_BASE_POSE_Q.copy()
    robot_init_qpos = DEFAULT_ROBOT_INIT_QPOS.copy()
    if not path:
        return robot_base_pose_p, robot_base_pose_q, robot_init_qpos

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("robot_base_pose_p="):
            robot_base_pose_p = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
        elif line.startswith("robot_base_pose_q="):
            robot_base_pose_q = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
        elif line.startswith("robot_init_qpos="):
            robot_init_qpos = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
    return robot_base_pose_p, robot_base_pose_q, robot_init_qpos


def _build_env(args: Args, camera_state: dict[str, np.ndarray | float | str] | None = None) -> BaseEnv:
    if camera_state is None:
        camera_state = observation_camera_lookat_state(args.start_camera_mode)
    pose = pose_from_eye_target_roll(
        eye=np.array(camera_state["eye"], dtype=np.float32),
        target=np.array(camera_state["target"], dtype=np.float32),
        roll_deg=float(camera_state.get("roll_deg", 0.0)),
    )
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
                "fov": float(camera_state.get("fov", 1.0)),
                "near": 0.01,
                "far": 10.0,
            },
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids="rc5_aero_hand_right_openreal2sim_validation",
    )


def _reset_env_scene(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
) -> None:
    env.reset(
        seed=args.seed,
        options={
            "reconfigure": True,
            "load_background": args.load_background,
            "use_probe_objects": False,
            "show_debug_markers": False,
            "show_spawn_grid": False,
            "safe_robot_during_spawn": False,
            "robot_far_away": False,
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_qpos.tolist(),
        },
    )


def _rebuild_env_for_camera(
    env: BaseEnv,
    args: Args,
    camera_state: dict[str, np.ndarray | float | str],
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
) -> BaseEnv:
    env.close()
    new_env = _build_env(args, camera_state=camera_state)
    _reset_env_scene(new_env, args, robot_base_pose_p, robot_base_pose_q, robot_qpos)
    return new_env


def _sync_gpu(env: BaseEnv) -> None:
    if env.unwrapped.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()
        env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env.unwrapped.scene._gpu_fetch_all()


def _zero_robot_dynamics(env: BaseEnv) -> None:
    robot = env.unwrapped.agent.robot
    robot.set_qvel(torch.zeros_like(robot.get_qvel()))
    robot.set_qf(torch.zeros_like(robot.get_qf()))
    try:
        robot.set_root_linear_velocity(torch.zeros((1, 3), dtype=torch.float32, device=env.unwrapped.device))
        robot.set_root_angular_velocity(torch.zeros((1, 3), dtype=torch.float32, device=env.unwrapped.device))
    except Exception:
        pass


def _apply_robot_state(
    env: BaseEnv,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
) -> None:
    robot = env.unwrapped.agent.robot
    device = env.unwrapped.device
    robot.set_pose(
        sapien.Pose(
            p=robot_base_pose_p.astype(np.float64).tolist(),
            q=robot_base_pose_q.astype(np.float64).tolist(),
        )
    )
    robot.set_qpos(torch.as_tensor(robot_qpos[None, :], dtype=torch.float32, device=device))
    _zero_robot_dynamics(env)
    _sync_gpu(env)
    if hasattr(env.unwrapped.agent, "controller") and env.unwrapped.agent.controller is not None:
        try:
            env.unwrapped.agent.controller.reset()
        except Exception:
            pass


def _refresh_obs(env: BaseEnv) -> tuple[dict, dict]:
    info = env.get_info()
    obs = env.get_obs(info)
    return obs, info


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    theta = np.deg2rad(degrees)
    return (
        vec * np.cos(theta)
        + np.cross(axis, vec) * np.sin(theta)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(theta))
    )


def _camera_presets(args: Args) -> dict[str, dict[str, np.ndarray | float | str]]:
    state = observation_camera_lookat_state(args.start_camera_mode)
    default_eye = np.array(state["eye"], dtype=np.float32)
    default_target = np.array(state["target"], dtype=np.float32)
    default_roll_deg = float(state.get("roll_deg", 0.0))
    default_fov = float(state.get("fov", 1.0))
    offset = default_eye - default_target
    far_eye = default_target + offset * float(args.far_distance_scale)
    left_eye = default_target + _rotate_vector(
        offset, np.array([0.0, 0.0, 1.0], dtype=np.float32), args.orbit_angle_deg
    )
    right_eye = default_target + _rotate_vector(
        offset, np.array([0.0, 0.0, 1.0], dtype=np.float32), -args.orbit_angle_deg
    )
    return {
        "default": {
            "name": "default",
            "eye": default_eye,
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
        "far": {
            "name": "far",
            "eye": far_eye.astype(np.float32),
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
        "left": {
            "name": "left",
            "eye": left_eye.astype(np.float32),
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
        "right": {
            "name": "right",
            "eye": right_eye.astype(np.float32),
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
    }


def _apply_camera_state(env: BaseEnv, camera_state: dict[str, np.ndarray | float | str]) -> None:
    pose = pose_from_eye_target_roll(
        eye=np.asarray(camera_state["eye"], dtype=np.float32),
        target=np.asarray(camera_state["target"], dtype=np.float32),
        roll_deg=float(camera_state["roll_deg"]),
    )

    sensor = env.unwrapped._sensors["3rd_view_camera"]
    sensor.config.pose = sensor.config.pose.create(pose)
    sensor.camera.local_pose = pose
    sensor.camera._cached_local_pose = None
    sensor.camera._cached_model_matrix = None

    for camera_name in ("render_camera", "sanity_render_camera"):
        if camera_name not in env.unwrapped._human_render_cameras:
            continue
        render_camera = env.unwrapped._human_render_cameras[camera_name]
        render_camera.config.pose = render_camera.config.pose.create(pose)
        render_camera.camera.local_pose = pose
        render_camera.camera._cached_local_pose = None
        render_camera.camera._cached_model_matrix = None

    if env.unwrapped.viewer is not None:
        env.unwrapped.viewer.set_camera_pose(pose)

    env.unwrapped.scene.update_render(update_sensors=False, update_human_render_cameras=True)


def _save_snapshot(
    obs: dict,
    info: dict,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
    camera_state: dict[str, np.ndarray | float | str],
    index: int,
    source: str,
    *,
    force_history: bool = False,
) -> None:
    frame = _extract_obs_frame(obs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / f"{args.image_prefix}_latest.png"
    latest_meta_path = output_dir / f"{args.image_prefix}_latest.txt"
    image_path = latest_path
    meta_path = latest_meta_path

    save_history = args.keep_history or force_history
    if save_history:
        image_path = output_dir / f"{args.image_prefix}_{index:04d}.png"
        meta_path = output_dir / f"{args.image_prefix}_{index:04d}.txt"

    tcp_pose = obs["extra"]["tcp_pose"][0].detach().cpu().numpy().tolist()
    info_summary = {
        k: (v[0].item() if torch.is_tensor(v) else v)
        for k, v in info.items()
    }
    meta_text = "\n".join(
        [
            f"source={source}",
            f"robot_base_pose_p={robot_base_pose_p.tolist()}",
            f"robot_base_pose_q={robot_base_pose_q.tolist()}",
            f"robot_init_qpos={robot_qpos.tolist()}",
            f"camera_preset={camera_state['name']}",
            f"camera_eye={np.asarray(camera_state['eye']).tolist()}",
            f"camera_target={np.asarray(camera_state['target']).tolist()}",
            f"camera_roll_deg={float(camera_state['roll_deg'])}",
            f"camera_fov={float(camera_state['fov'])}",
            f"tcp_pose={tcp_pose}",
            f"info={info_summary}",
        ]
    )

    iio.imwrite(latest_path, frame)
    latest_meta_path.write_text(meta_text, encoding="utf-8")
    print(f"\nlatest image: {latest_path}")
    print(f"latest meta: {latest_meta_path}")
    if save_history:
        iio.imwrite(image_path, frame)
        meta_path.write_text(meta_text, encoding="utf-8")
        print(f"saved {image_path}")
        print(f"saved {meta_path}")


def _print_help(args: Args) -> None:
    print("Interactive RC5 pose calibration")
    print("Keys:")
    print("  w/s  base Y +/-")
    print("  a/d  base X -/+")
    print("  r/f  base Z +/-")
    print(f"  q/e  base yaw +/- ({args.yaw_step_deg:.1f} deg)")
    print("  1-6  select arm joint")
    print(f"  j/l  selected joint -/+ ({args.joint_step_deg:.1f} deg)")
    print(f"  u/o  selected joint -/+ big ({args.joint_big_step_deg:.1f} deg)")
    print("  c    camera default")
    print("  v    camera farther")
    print("  b    camera left orbit")
    print("  n    camera right orbit")
    print("  p    save numbered snapshot")
    print("  h    print current state")
    print("  x    exit")


def _print_state(
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
    camera_state: dict[str, np.ndarray | float | str],
    selected_joint: int,
) -> None:
    _, _, yaw = quat2euler(robot_base_pose_q, axes="sxyz")
    print()
    print(f"camera_preset={camera_state['name']}")
    print(f"robot_base_pose_p={robot_base_pose_p.tolist()}")
    print(f"robot_base_yaw_deg={np.rad2deg(yaw):.3f}")
    print(f"selected_joint={selected_joint + 1}")
    print(f"selected_joint_rad={float(robot_qpos[selected_joint]):.6f}")
    print(f"selected_joint_deg={float(np.rad2deg(robot_qpos[selected_joint])):.3f}")


def main(args: Args) -> int:
    robot_base_pose_p, robot_base_pose_q, robot_qpos = _load_state_file(args.load_state_file)
    camera_presets = _camera_presets(args)
    camera_state = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in camera_presets[DEFAULT_CAMERA_PRESET].items()
    }

    env = _build_env(args)
    try:
        _reset_env_scene(env, args, robot_base_pose_p, robot_base_pose_q, robot_qpos)
        _apply_robot_state(env, robot_base_pose_p, robot_base_pose_q, robot_qpos)
        obs, info = _refresh_obs(env)

        save_idx = 0
        selected_joint = 0
        _save_snapshot(
            obs,
            info,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_qpos,
            camera_state,
            save_idx,
            "startup",
        )
        _print_help(args)
        _print_state(robot_base_pose_p, robot_base_pose_q, robot_qpos, camera_state, selected_joint)

        while True:
            key = _read_single_key().lower()
            action_source = None
            changed = False
            force_history = False

            if key == "x":
                print("\nleaving RC5 live calibration")
                break
            if key in {"1", "2", "3", "4", "5", "6"}:
                selected_joint = int(key) - 1
                _print_state(robot_base_pose_p, robot_base_pose_q, robot_qpos, camera_state, selected_joint)
                continue
            if key == "h":
                _print_state(robot_base_pose_p, robot_base_pose_q, robot_qpos, camera_state, selected_joint)
                continue

            if key == "w":
                robot_base_pose_p[1] += args.base_step
                action_source = "base_y_plus"
                changed = True
            elif key == "s":
                robot_base_pose_p[1] -= args.base_step
                action_source = "base_y_minus"
                changed = True
            elif key == "a":
                robot_base_pose_p[0] -= args.base_step
                action_source = "base_x_minus"
                changed = True
            elif key == "d":
                robot_base_pose_p[0] += args.base_step
                action_source = "base_x_plus"
                changed = True
            elif key == "r":
                robot_base_pose_p[2] += args.z_step
                action_source = "base_z_plus"
                changed = True
            elif key == "f":
                robot_base_pose_p[2] -= args.z_step
                action_source = "base_z_minus"
                changed = True
            elif key == "q":
                roll, pitch, yaw = quat2euler(robot_base_pose_q, axes="sxyz")
                yaw += np.deg2rad(args.yaw_step_deg)
                robot_base_pose_q[:] = euler2quat(roll, pitch, yaw, axes="sxyz")
                action_source = "base_yaw_plus"
                changed = True
            elif key == "e":
                roll, pitch, yaw = quat2euler(robot_base_pose_q, axes="sxyz")
                yaw -= np.deg2rad(args.yaw_step_deg)
                robot_base_pose_q[:] = euler2quat(roll, pitch, yaw, axes="sxyz")
                action_source = "base_yaw_minus"
                changed = True
            elif key == "j":
                robot_qpos[selected_joint] -= np.deg2rad(args.joint_step_deg)
                action_source = "joint_minus"
                changed = True
            elif key == "l":
                robot_qpos[selected_joint] += np.deg2rad(args.joint_step_deg)
                action_source = "joint_plus"
                changed = True
            elif key == "u":
                robot_qpos[selected_joint] -= np.deg2rad(args.joint_big_step_deg)
                action_source = "joint_big_minus"
                changed = True
            elif key == "o":
                robot_qpos[selected_joint] += np.deg2rad(args.joint_big_step_deg)
                action_source = "joint_big_plus"
                changed = True
            elif key in {"c", "v", "b", "n"}:
                preset_name = {
                    "c": "default",
                    "v": "far",
                    "b": "left",
                    "n": "right",
                }[key]
                camera_state = {
                    name: (value.copy() if isinstance(value, np.ndarray) else value)
                    for name, value in camera_presets[preset_name].items()
                }
                action_source = f"camera_{preset_name}"
                changed = True
            elif key == "p":
                action_source = "manual_save"
                changed = True
                force_history = True
            else:
                continue

            if action_source.startswith("camera_"):
                env = _rebuild_env_for_camera(
                    env,
                    args,
                    camera_state,
                    robot_base_pose_p,
                    robot_base_pose_q,
                    robot_qpos,
                )
                _apply_robot_state(env, robot_base_pose_p, robot_base_pose_q, robot_qpos)
            elif action_source != "manual_save":
                _apply_robot_state(env, robot_base_pose_p, robot_base_pose_q, robot_qpos)

            obs, info = _refresh_obs(env)
            save_idx += 1
            _save_snapshot(
                obs,
                info,
                args,
                robot_base_pose_p,
                robot_base_pose_q,
                robot_qpos,
                camera_state,
                save_idx,
                action_source,
                force_history=force_history,
            )
            if changed:
                _print_state(
                    robot_base_pose_p,
                    robot_base_pose_q,
                    robot_qpos,
                    camera_state,
                    selected_joint,
                )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
