#!/usr/bin/env python3
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import termios
import tty
import xml.etree.ElementTree as ET

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import sapien
import torch
import tyro
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import mat2quat, quat2mat

from mani_skill.envs.sapien_env import BaseEnv
from real2sim.debug_paths import RC5_POSE_LIVE_LATEST_STATE_FILE, RC5_TCP_POSE_LIVE_DIR
from real2sim.openreal2sim_validation import observation_camera_lookat_state, pose_from_eye_target_roll


REPO_ROOT = Path(__file__).resolve().parents[2]
RC5_URDF_PATH = REPO_ROOT / "real2sim" / "assets" / "rc5" / "Robot _with_right_hand.urdf"
RIGHT_TCP_JOINT_NAME = "right_tcp_mount"
RIGHT_TCP_LINK_NAME = "right_tcp_link"
RIGHT_TCP_PARENT_LINK_NAME = "right_base_link"
DEFAULT_STATE_FILE = str(RC5_POSE_LIVE_LATEST_STATE_FILE)


@dataclass
class Args:
    output_dir: str = str(RC5_TCP_POSE_LIVE_DIR)
    image_prefix: str = "openreal2sim_rc5_tcp_live"
    keep_history: bool = False
    sim_backend: str = "gpu"
    shader: str = "default"
    seed: int = 0
    load_background: bool = True
    start_camera_mode: str = "manual_best"
    state_file: str = DEFAULT_STATE_FILE
    pos_step: float = 0.005
    pos_big_step: float = 0.02
    rot_step_deg: float = 3.0
    rot_big_step_deg: float = 10.0
    far_distance_scale: float = 1.35
    orbit_angle_deg: float = 50.0
    marker_radius: float = 0.04


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


def _load_robot_state(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_pose_p = None
    base_pose_q = None
    qpos = None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("robot_base_pose_p="):
            base_pose_p = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
        elif line.startswith("robot_base_pose_q="):
            base_pose_q = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
        elif line.startswith("robot_init_qpos="):
            qpos = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
    if base_pose_p is None or base_pose_q is None or qpos is None:
        raise ValueError(f"{path!r} does not contain robot_base_pose_p/q and robot_init_qpos")
    return base_pose_p, base_pose_q, qpos


def _load_tcp_mount_from_urdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != RIGHT_TCP_JOINT_NAME:
            continue
        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"Joint {RIGHT_TCP_JOINT_NAME!r} is missing <origin>")
        xyz = np.array([float(x) for x in origin.attrib.get("xyz", "0 0 0").split()], dtype=np.float32)
        rpy = np.array([float(x) for x in origin.attrib.get("rpy", "0 0 0").split()], dtype=np.float32)
        return xyz, rpy
    raise ValueError(f"Joint {RIGHT_TCP_JOINT_NAME!r} not found in {path}")


def _write_tcp_mount_to_urdf(path: Path, xyz: np.ndarray, rpy: np.ndarray) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != RIGHT_TCP_JOINT_NAME:
            continue
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", " ".join(f"{float(v):.9f}" for v in xyz))
        origin.set("rpy", " ".join(f"{float(v):.9f}" for v in rpy))
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return
    raise ValueError(f"Joint {RIGHT_TCP_JOINT_NAME!r} not found in {path}")


def _camera_presets(args: Args) -> dict[str, dict[str, np.ndarray | float | str]]:
    state = observation_camera_lookat_state(args.start_camera_mode)
    default_eye = np.array(state["eye"], dtype=np.float32)
    default_target = np.array(state["target"], dtype=np.float32)
    default_roll_deg = float(state.get("roll_deg", 0.0))
    default_fov = float(state.get("fov", 1.0))
    offset = default_eye - default_target

    def rotate_around_z(vec: np.ndarray, degrees: float) -> np.ndarray:
        theta = np.deg2rad(degrees)
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return rot @ vec

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
            "eye": (default_target + offset * float(args.far_distance_scale)).astype(np.float32),
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
        "left": {
            "name": "left",
            "eye": (default_target + rotate_around_z(offset, args.orbit_angle_deg)).astype(np.float32),
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
        "right": {
            "name": "right",
            "eye": (default_target + rotate_around_z(offset, -args.orbit_angle_deg)).astype(np.float32),
            "target": default_target,
            "roll_deg": default_roll_deg,
            "fov": default_fov,
        },
    }


def _build_env(args: Args, camera_state: dict[str, np.ndarray | float | str]) -> BaseEnv:
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
            "create_tcp_calibration_marker": True,
            "tcp_calibration_marker_radius": args.marker_radius,
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_qpos.tolist(),
        },
    )


def _sync_gpu(env: BaseEnv) -> None:
    if env.unwrapped.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()
        env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env.unwrapped.scene._gpu_fetch_all()


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
    robot.set_qvel(torch.zeros_like(robot.get_qvel()))
    robot.set_qf(torch.zeros_like(robot.get_qf()))
    try:
        robot.set_root_linear_velocity(torch.zeros((1, 3), dtype=torch.float32, device=device))
        robot.set_root_angular_velocity(torch.zeros((1, 3), dtype=torch.float32, device=device))
    except Exception:
        pass
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
    env.unwrapped.scene.update_render(update_sensors=False, update_human_render_cameras=True)


def _rebuild_env(
    env: BaseEnv,
    args: Args,
    camera_state: dict[str, np.ndarray | float | str],
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
) -> BaseEnv:
    env.close()
    env = _build_env(args, camera_state)
    _reset_env_scene(env, args, robot_base_pose_p, robot_base_pose_q, robot_qpos)
    _apply_robot_state(env, robot_base_pose_p, robot_base_pose_q, robot_qpos)
    return env


def _base_link_world_pose(env: BaseEnv) -> tuple[np.ndarray, np.ndarray]:
    link = env.unwrapped.agent.robot.links_map[RIGHT_TCP_PARENT_LINK_NAME]
    raw_pose = link.pose.raw_pose[0].detach().cpu().numpy().astype(np.float64)
    return raw_pose[:3], raw_pose[3:7]


def _compose_tcp_world_pose(
    env: BaseEnv,
    tcp_xyz: np.ndarray,
    tcp_rpy: np.ndarray,
) -> sapien.Pose:
    base_p, base_q = _base_link_world_pose(env)
    base_rot = quat2mat(base_q)
    local_rot = quat2mat(euler2quat(*tcp_rpy, axes="sxyz"))
    world_p = base_p + base_rot @ tcp_xyz.astype(np.float64)
    world_q = mat2quat(base_rot @ local_rot)
    return sapien.Pose(p=world_p, q=world_q)


def _update_tcp_marker(env: BaseEnv, marker, tcp_xyz: np.ndarray, tcp_rpy: np.ndarray) -> None:
    marker.set_pose(_compose_tcp_world_pose(env, tcp_xyz, tcp_rpy))


def _save_snapshot(
    obs: dict,
    info: dict,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_qpos: np.ndarray,
    camera_state: dict[str, np.ndarray | float | str],
    tcp_xyz: np.ndarray,
    tcp_rpy: np.ndarray,
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

    info_summary = {
        k: (v[0].item() if torch.is_tensor(v) else v)
        for k, v in info.items()
    }
    tcp_pose = obs["extra"]["tcp_pose"][0].detach().cpu().numpy().tolist()
    meta_text = "\n".join(
        [
            f"source={source}",
            f"robot_base_pose_p={robot_base_pose_p.tolist()}",
            f"robot_base_pose_q={robot_base_pose_q.tolist()}",
            f"robot_init_qpos={robot_qpos.tolist()}",
            f"camera_preset={camera_state['name']}",
            f"camera_eye={np.asarray(camera_state['eye']).tolist()}",
            f"camera_target={np.asarray(camera_state['target']).tolist()}",
            f"tcp_mount_xyz={tcp_xyz.tolist()}",
            f"tcp_mount_rpy_rad={tcp_rpy.tolist()}",
            f"tcp_mount_rpy_deg={np.rad2deg(tcp_rpy).tolist()}",
            f"sim_tcp_pose={tcp_pose}",
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
    print("Interactive RC5 TCP calibration")
    print("Keys:")
    print(f"  w/s  local Y +/- ({args.pos_step:.3f} m)")
    print(f"  a/d  local X -/+ ({args.pos_step:.3f} m)")
    print(f"  q/e  local Z +/- ({args.pos_step:.3f} m)")
    print(f"  r/f  local X -/+ big ({args.pos_big_step:.3f} m)")
    print(f"  t/g  local Y +/- big ({args.pos_big_step:.3f} m)")
    print(f"  y/h  local Z +/- big ({args.pos_big_step:.3f} m)")
    print(f"  i/k  pitch +/- ({args.rot_step_deg:.1f} deg)")
    print(f"  j/l  yaw +/- ({args.rot_step_deg:.1f} deg)")
    print(f"  u/o  roll +/- ({args.rot_step_deg:.1f} deg)")
    print(f"  m/,  pitch +/- big ({args.rot_big_step_deg:.1f} deg)")
    print(f"  n/.  yaw +/- big ({args.rot_big_step_deg:.1f} deg)")
    print(f"  7/9  roll +/- big ({args.rot_big_step_deg:.1f} deg)")
    print("  1/2/3/4  camera default/far/left/right")
    print("  p    save numbered snapshot")
    print("  z    reload current TCP from URDF")
    print("  enter  write current TCP into URDF and rebuild env so IK uses it")
    print("  5    print current TCP values")
    print("  x    exit")


def _print_state(tcp_xyz: np.ndarray, tcp_rpy: np.ndarray, camera_state: dict[str, np.ndarray | float | str]) -> None:
    print()
    print(f"camera_preset={camera_state['name']}")
    print(f"tcp_mount_xyz_m={tcp_xyz.tolist()}")
    print(f"tcp_mount_rpy_rad={tcp_rpy.tolist()}")
    print(f"tcp_mount_rpy_deg={np.rad2deg(tcp_rpy).tolist()}")


def main(args: Args) -> int:
    robot_base_pose_p, robot_base_pose_q, robot_qpos = _load_robot_state(args.state_file)
    tcp_xyz, tcp_rpy = _load_tcp_mount_from_urdf(RC5_URDF_PATH)
    camera_presets = _camera_presets(args)
    camera_state = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in camera_presets["default"].items()
    }

    env = _build_env(args, camera_state)
    try:
        _reset_env_scene(env, args, robot_base_pose_p, robot_base_pose_q, robot_qpos)
        _apply_robot_state(env, robot_base_pose_p, robot_base_pose_q, robot_qpos)
        marker = env.unwrapped.tcp_calibration_marker
        if marker is None:
            raise RuntimeError("TCP calibration marker was not created during environment reset.")
        _update_tcp_marker(env, marker, tcp_xyz, tcp_rpy)
        obs, info = _refresh_obs(env)

        save_idx = 0
        _save_snapshot(
            obs,
            info,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_qpos,
            camera_state,
            tcp_xyz,
            tcp_rpy,
            save_idx,
            "startup",
        )
        _print_help(args)
        _print_state(tcp_xyz, tcp_rpy, camera_state)

        while True:
            key = _read_single_key()
            source = None
            force_history = False
            camera_changed = False
            tcp_changed = False
            apply_to_urdf = False

            if key.lower() == "x":
                print("\nleaving RC5 TCP calibration")
                break
            if key == "5":
                _print_state(tcp_xyz, tcp_rpy, camera_state)
                continue
            if key == "\r" or key == "\n":
                apply_to_urdf = True
                source = "apply_tcp_to_urdf"
            elif key.lower() == "p":
                source = "manual_save"
                force_history = True
            elif key.lower() == "z":
                tcp_xyz, tcp_rpy = _load_tcp_mount_from_urdf(RC5_URDF_PATH)
                source = "reload_from_urdf"
                tcp_changed = True
            elif key.lower() == "w":
                tcp_xyz[1] += args.pos_step
                source = "y_plus"
                tcp_changed = True
            elif key.lower() == "s":
                tcp_xyz[1] -= args.pos_step
                source = "y_minus"
                tcp_changed = True
            elif key.lower() == "a":
                tcp_xyz[0] -= args.pos_step
                source = "x_minus"
                tcp_changed = True
            elif key.lower() == "d":
                tcp_xyz[0] += args.pos_step
                source = "x_plus"
                tcp_changed = True
            elif key.lower() == "q":
                tcp_xyz[2] += args.pos_step
                source = "z_plus"
                tcp_changed = True
            elif key.lower() == "e":
                tcp_xyz[2] -= args.pos_step
                source = "z_minus"
                tcp_changed = True
            elif key.lower() == "r":
                tcp_xyz[0] -= args.pos_big_step
                source = "x_big_minus"
                tcp_changed = True
            elif key.lower() == "f":
                tcp_xyz[0] += args.pos_big_step
                source = "x_big_plus"
                tcp_changed = True
            elif key.lower() == "t":
                tcp_xyz[1] -= args.pos_big_step
                source = "y_big_minus"
                tcp_changed = True
            elif key.lower() == "g":
                tcp_xyz[1] += args.pos_big_step
                source = "y_big_plus"
                tcp_changed = True
            elif key.lower() == "y":
                tcp_xyz[2] += args.pos_big_step
                source = "z_big_plus"
                tcp_changed = True
            elif key.lower() == "h":
                tcp_xyz[2] -= args.pos_big_step
                source = "z_big_minus"
                tcp_changed = True
            elif key.lower() == "i":
                tcp_rpy[1] += np.deg2rad(args.rot_step_deg)
                source = "pitch_plus"
                tcp_changed = True
            elif key.lower() == "k":
                tcp_rpy[1] -= np.deg2rad(args.rot_step_deg)
                source = "pitch_minus"
                tcp_changed = True
            elif key.lower() == "j":
                tcp_rpy[2] += np.deg2rad(args.rot_step_deg)
                source = "yaw_plus"
                tcp_changed = True
            elif key.lower() == "l":
                tcp_rpy[2] -= np.deg2rad(args.rot_step_deg)
                source = "yaw_minus"
                tcp_changed = True
            elif key.lower() == "u":
                tcp_rpy[0] += np.deg2rad(args.rot_step_deg)
                source = "roll_plus"
                tcp_changed = True
            elif key.lower() == "o":
                tcp_rpy[0] -= np.deg2rad(args.rot_step_deg)
                source = "roll_minus"
                tcp_changed = True
            elif key.lower() == "m":
                tcp_rpy[1] += np.deg2rad(args.rot_big_step_deg)
                source = "pitch_big_plus"
                tcp_changed = True
            elif key == ",":
                tcp_rpy[1] -= np.deg2rad(args.rot_big_step_deg)
                source = "pitch_big_minus"
                tcp_changed = True
            elif key.lower() == "n":
                tcp_rpy[2] += np.deg2rad(args.rot_big_step_deg)
                source = "yaw_big_plus"
                tcp_changed = True
            elif key == ".":
                tcp_rpy[2] -= np.deg2rad(args.rot_big_step_deg)
                source = "yaw_big_minus"
                tcp_changed = True
            elif key == "7":
                tcp_rpy[0] += np.deg2rad(args.rot_big_step_deg)
                source = "roll_big_plus"
                tcp_changed = True
            elif key == "9":
                tcp_rpy[0] -= np.deg2rad(args.rot_big_step_deg)
                source = "roll_big_minus"
                tcp_changed = True
            elif key in {"1", "2", "3", "4"}:
                preset_name = {"1": "default", "2": "far", "3": "left", "4": "right"}[key]
                camera_state = {
                    name: (value.copy() if isinstance(value, np.ndarray) else value)
                    for name, value in camera_presets[preset_name].items()
                }
                source = f"camera_{preset_name}"
                camera_changed = True
            else:
                continue

            if apply_to_urdf:
                _write_tcp_mount_to_urdf(RC5_URDF_PATH, tcp_xyz, tcp_rpy)
                env = _rebuild_env(env, args, camera_state, robot_base_pose_p, robot_base_pose_q, robot_qpos)
                marker = env.unwrapped.tcp_calibration_marker
                if marker is None:
                    raise RuntimeError("TCP calibration marker was not recreated after URDF apply.")
            elif camera_changed:
                env = _rebuild_env(env, args, camera_state, robot_base_pose_p, robot_base_pose_q, robot_qpos)
                marker = env.unwrapped.tcp_calibration_marker
                if marker is None:
                    raise RuntimeError("TCP calibration marker was not recreated after camera change.")
            elif tcp_changed:
                _apply_robot_state(env, robot_base_pose_p, robot_base_pose_q, robot_qpos)

            _update_tcp_marker(env, marker, tcp_xyz, tcp_rpy)
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
                tcp_xyz,
                tcp_rpy,
                save_idx,
                source,
                force_history=force_history,
            )
            _print_state(tcp_xyz, tcp_rpy, camera_state)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
