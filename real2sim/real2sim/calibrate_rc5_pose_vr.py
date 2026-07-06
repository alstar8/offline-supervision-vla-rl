import sys
import time
from dataclasses import dataclass
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import tyro
from transforms3d.euler import quat2euler
from transforms3d.quaternions import mat2quat, qinverse, qmult, quat2mat

from real2sim.calibrate_rc5_pose import (
    Args as BaseArgs,
    HAND_GRAB_STATE_FILE,
    HAND_OPEN_STATE_FILE,
    _apply_gripper_qpos_direct,
    _apply_grid_pair_override,
    _apply_sim_arm_target_qpos_direct,
    _build_env,
    _current_gripper_pinch_amount,
    _load_state_file,
    _maybe_render_live,
    _probe_carrot_grasped,
    _read_single_key,
    _remap_sim_delta,
    _reset_current_state,
    _sanitize_live_viewer_environment,
    _save_snapshot_from_obs,
    _sim_arm_controller,
    _state_file_gripper_qpos,
    _interpolate_gripper_qpos,
    _extract_obs_frame,
    observation_camera_lookat_state,
)


def _load_vr_tracker_backend(backend: str):
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg_root = repo_root / "quest3xr"
    local_pkg_root_str = str(local_pkg_root)
    if local_pkg_root_str not in sys.path:
        sys.path.insert(0, local_pkg_root_str)
    if backend == "playground":
        from quest3xr.playground_runtime import PlaygroundOpenXRTracker

        return PlaygroundOpenXRTracker
    if backend == "pyopenxr":
        from quest3xr.runtime import OpenXRControllerTracker

        return OpenXRControllerTracker
    raise RuntimeError(f"Unsupported VR tracker backend: {backend}")


def _load_quad_stream_writer():
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg_root = repo_root / "quest3xr"
    local_pkg_root_str = str(local_pkg_root)
    if local_pkg_root_str not in sys.path:
        sys.path.insert(0, local_pkg_root_str)
    from quest3xr.frame_stream import QuadFrameStreamWriter

    return QuadFrameStreamWriter


def _write_vr_stream_frame(stream_writer, obs) -> None:
    if stream_writer is None:
        return
    frame = _extract_obs_frame(obs)
    if bool(obs.get("_vr_stream_postprocess", False)):
        frame = _prepare_vr_stream_frame(
            frame,
            contrast=float(obs.get("_vr_stream_contrast", 1.22)),
            brightness=float(obs.get("_vr_stream_brightness", -10.0)),
            gamma=float(obs.get("_vr_stream_gamma", 0.88)),
            saturation=float(obs.get("_vr_stream_saturation", 1.22)),
            red_gain=float(obs.get("_vr_stream_red_gain", 1.08)),
            green_gain=float(obs.get("_vr_stream_green_gain", 0.97)),
            blue_gain=float(obs.get("_vr_stream_blue_gain", 1.00)),
        )
    stream_writer.write_rgb(frame)


def _get_vr_stream_source_frame(env, obs: dict, args: "Args") -> np.ndarray:
    if getattr(args, "vr_stream_use_human_render", False):
        frame = env.render_rgb_array(camera_name=args.vr_stream_camera_name)
        if frame is not None:
            if torch.is_tensor(frame):
                if frame.ndim == 4:
                    frame = frame[0]
                if frame.dtype.is_floating_point:
                    max_val = float(frame.max().detach().cpu().item()) if frame.numel() > 0 else 0.0
                    if max_val <= 1.0 + 1e-6:
                        frame = frame * 255.0
                return frame.detach().clamp(0, 255).to(torch.uint8).cpu().numpy()
            arr = np.asarray(frame)
            if arr.ndim == 4:
                arr = arr[0]
            if np.issubdtype(arr.dtype, np.floating):
                max_val = float(arr.max()) if arr.size > 0 else 0.0
                if max_val <= 1.0 + 1e-6:
                    arr = arr * 255.0
            return np.clip(arr, 0, 255).astype(np.uint8)
    return _extract_obs_frame(obs)


def _prepare_vr_stream_frame(
    frame: np.ndarray,
    *,
    contrast: float = 1.22,
    brightness: float = -10.0,
    gamma: float = 0.88,
    saturation: float = 1.22,
    red_gain: float = 1.08,
    green_gain: float = 0.97,
    blue_gain: float = 1.00,
) -> np.ndarray:
    image = np.ascontiguousarray(frame)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    enhanced = cv2.convertScaleAbs(image, alpha=float(contrast), beta=float(brightness))
    if abs(float(gamma) - 1.0) > 1e-3:
        lut = np.array(
            [np.clip(((i / 255.0) ** float(gamma)) * 255.0, 0.0, 255.0) for i in range(256)],
            dtype=np.uint8,
        )
        enhanced = cv2.LUT(enhanced, lut)
    if abs(float(saturation) - 1.0) > 1e-3:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation), 0.0, 255.0)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    gains = np.array([red_gain, green_gain, blue_gain], dtype=np.float32).reshape(1, 1, 3)
    enhanced = np.clip(enhanced.astype(np.float32) * gains, 0.0, 255.0).astype(np.uint8)
    return enhanced


def _obs_with_vr_stream_settings(obs: dict, args: "Args") -> dict:
    return {
        **obs,
        "_vr_stream_postprocess": bool(args.vr_stream_postprocess),
        "_vr_stream_contrast": float(args.vr_stream_contrast),
        "_vr_stream_brightness": float(args.vr_stream_brightness),
        "_vr_stream_gamma": float(args.vr_stream_gamma),
        "_vr_stream_saturation": float(args.vr_stream_saturation),
        "_vr_stream_red_gain": float(args.vr_stream_red_gain),
        "_vr_stream_green_gain": float(args.vr_stream_green_gain),
        "_vr_stream_blue_gain": float(args.vr_stream_blue_gain),
    }


def _extract_controller_pose_and_trigger(controller, use_aim_pose: bool):
    if hasattr(controller, "pose"):
        squeeze_value = float(getattr(controller, "squeeze_value", 0.0))
        return controller.pose, "legacy", float(controller.trigger_value), squeeze_value

    preferred_pose = controller.aim_pose if use_aim_pose else controller.grip_pose
    fallback_pose = controller.grip_pose if use_aim_pose else controller.aim_pose
    if preferred_pose.is_valid:
        pose = preferred_pose
        pose_source = "aim" if use_aim_pose else "grip"
    elif fallback_pose.is_valid:
        pose = fallback_pose
        pose_source = "grip" if use_aim_pose else "aim"
    else:
        pose = preferred_pose
        pose_source = "aim" if use_aim_pose else "grip"
    return (
        pose,
        pose_source,
        float(controller.trigger_value),
        float(getattr(controller, "squeeze_value", 0.0)),
    )


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _camera_planar_basis(camera_eye: np.ndarray, camera_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = np.asarray(camera_target, dtype=np.float32) - np.asarray(camera_eye, dtype=np.float32)
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        forward /= norm
    # Sim/world convention here is right-handed with +Z up, so camera-right is forward x up.
    # Using up x forward would produce camera-left and invert horizontal teleop.
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= right_norm
    return right, forward, up


def _xr_delta_to_sim_axes(delta_pos_world: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(delta_pos_world[0]),
            float(-delta_pos_world[2]),
            float(delta_pos_world[1]),
        ],
        dtype=np.float32,
    )


def _map_vr_translation_via_camera(delta_pos_world: np.ndarray, camera_right: np.ndarray, camera_forward: np.ndarray) -> list[float]:
    sim_delta = _xr_delta_to_sim_axes(delta_pos_world)
    local_delta = [
        float(np.dot(sim_delta, camera_right)),
        float(np.dot(sim_delta, camera_forward)),
        float(sim_delta[2]),
    ]
    mapped_delta_pos, _ = _remap_sim_delta(local_delta, [0.0, 0.0, 0.0])
    return mapped_delta_pos


def _xr_quat_to_sim_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    xr_rot = quat2mat(np.asarray(quat_wxyz, dtype=np.float64))
    basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    sim_rot = basis @ xr_rot @ basis.T
    return np.asarray(mat2quat(sim_rot), dtype=np.float64)


@dataclass
class Args(BaseArgs):
    live_viewer: bool = True
    live_viewer_force_cpu: bool = False
    human_render_shader: str = "minimal"
    viewer_shader: str = "minimal"
    use_environment_map: bool = False
    vr_hand: str = "right"
    vr_tracker_backend: str = "playground"
    vr_tracker_binary: str = ""
    vr_stream_path: str = "/tmp/quest3xr_stream.bin"
    vr_stream_to_headset: bool = True
    vr_stream_max_hz: float = 20.0
    vr_stream_use_human_render: bool = False
    vr_stream_camera_name: str = "render_camera"
    vr_stream_postprocess: bool = False
    vr_stream_contrast: float = 1.22
    vr_stream_brightness: float = -10.0
    vr_stream_gamma: float = 0.88
    vr_stream_saturation: float = 1.22
    vr_stream_red_gain: float = 1.08
    vr_stream_green_gain: float = 0.97
    vr_stream_blue_gain: float = 1.00
    vr_use_aim_pose: bool = True
    vr_translation_gain: float = 1.5
    vr_rotation_gain: float = 1.0
    vr_pos_deadband_m: float = 0.0005
    vr_rot_deadband_rad: float = 0.002
    vr_log_hz: float = 2.0
    vr_max_episode_steps: int = 100000
    vr_squeeze_teleop_threshold: float = 0.15
    vr_trigger_open: float = 0.05
    vr_trigger_close: float = 0.95
    vr_exit_key: str = "x"
    vr_recenter_key: str = "r"


def main(args: Args):
    if args.vr_hand not in {"left", "right"}:
        raise SystemExit("--vr-hand must be 'left' or 'right'")

    _sanitize_live_viewer_environment(args)
    TrackerClass = _load_vr_tracker_backend(args.vr_tracker_backend)
    QuadFrameStreamWriter = _load_quad_stream_writer()

    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
    _apply_grid_pair_override(args, robot_base_pose_p)
    print(f"Scene asset dir: {args.scene_asset_dir}")
    print(f"Loaded state file: {args.load_state_file or '<defaults>'}")
    print(
        f"VR hand={args.vr_hand} vr_use_aim_pose={args.vr_use_aim_pose} "
        f"vr_tracker_backend={args.vr_tracker_backend}"
    )
    print(f"Live viewer={args.live_viewer} live_viewer_force_cpu={args.live_viewer_force_cpu}")
    print("VR teleop controls:")
    print(f"  {args.vr_recenter_key} recenter controller deltas")
    print(f"  {args.vr_exit_key} exit VR teleop")
    print("  trigger controls pinch amount")
    print("  hold squeeze to enable EE teleop motion")

    camera_mode = args.observation_camera_mode
    cam_state = observation_camera_lookat_state(camera_mode, scene_asset_dir=args.scene_asset_dir)
    camera_eye = np.array(cam_state["eye"], dtype=np.float32)
    camera_target = np.array(cam_state["target"], dtype=np.float32)
    camera_roll_deg = float(cam_state.get("roll_deg", 0.0))
    camera_fov = float(cam_state.get("fov", 1.0))
    camera_right, camera_forward, _ = _camera_planar_basis(camera_eye, camera_target)

    env = _build_env(
        args,
        camera_mode=camera_mode,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_roll_deg=camera_roll_deg,
        camera_fov=camera_fov,
    )
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = int(args.vr_max_episode_steps)
    if hasattr(env.unwrapped, "_max_episode_steps"):
        env.unwrapped._max_episode_steps = int(args.vr_max_episode_steps)
    try:
        save_idx = 0
        obs, info = _reset_current_state(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_init_qpos,
            reconfigure=True,
        )
        _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)

        preset_open_gripper_qpos = _state_file_gripper_qpos(HAND_OPEN_STATE_FILE, env)
        preset_grab_gripper_qpos = _state_file_gripper_qpos(HAND_GRAB_STATE_FILE, env)
        pinch_amount = _current_gripper_pinch_amount(
            env,
            preset_open_qpos=preset_open_gripper_qpos,
            preset_closed_qpos=preset_grab_gripper_qpos,
        )

        last_log_time = 0.0
        prev_pos = None
        prev_quat_wxyz = None

        tracker_kwargs = {}
        if args.vr_tracker_backend == "playground":
            tracker_kwargs["binary_path"] = args.vr_tracker_binary or None
        else:
            tracker_kwargs["prefer_aim_pose"] = args.vr_use_aim_pose

        stream_writer = (
            QuadFrameStreamWriter(args.vr_stream_path, max_hz=args.vr_stream_max_hz)
            if args.vr_tracker_backend == "playground" and args.vr_stream_to_headset
            else None
        )
        if stream_writer is not None:
            _write_vr_stream_frame(
                stream_writer,
                {
                    **_obs_with_vr_stream_settings(obs, args),
                    "sensor_data": {"3rd_view_camera": {"rgb": _get_vr_stream_source_frame(env, obs, args)}},
                },
            )
            try:
                stream_size = os.path.getsize(args.vr_stream_path)
            except OSError:
                stream_size = -1
            print(f"VR stream initialized at {args.vr_stream_path} size={stream_size}")

        if args.vr_tracker_backend == "playground":
            tracker_kwargs["stream_path"] = args.vr_stream_path if args.vr_stream_to_headset else None

        with TrackerClass(**tracker_kwargs) as tracker:
            for vr_frame in tracker.frames():
                _maybe_render_live(env, args)

                key = _read_single_key(timeout_sec=0.0)
                if key is not None:
                    key = key.lower()
                    if key == args.vr_exit_key:
                        print("\nLeaving VR teleop")
                        break
                    if key == args.vr_recenter_key:
                        prev_pos = None
                        prev_quat_wxyz = None
                        print("\nRecentered VR controller delta origin")

                controller = vr_frame.controllers[args.vr_hand]
                controller_pose, pose_source, trigger_value, squeeze_value = _extract_controller_pose_and_trigger(
                    controller,
                    use_aim_pose=args.vr_use_aim_pose,
                )
                pinch_amount = float(np.clip(trigger_value, 0.0, 1.0))
                teleop_active = squeeze_value >= float(args.vr_squeeze_teleop_threshold)

                now = time.monotonic()
                if now - last_log_time >= 1.0 / max(args.vr_log_hz, 0.1):
                    last_log_time = now
                    grasped = _probe_carrot_grasped(env)
                    tcp_xyz = obs["extra"]["tcp_pose"][0].detach().cpu().numpy().tolist()[:3]
                    print(
                        f"vr_frame={vr_frame.frame_index} "
                        f"session={vr_frame.session_state} "
                        f"profile={controller.interaction_profile} "
                        f"pose_source={pose_source} "
                        f"pose_valid={int(controller_pose.is_valid)} "
                        f"trigger={pinch_amount:.3f} "
                        f"squeeze={squeeze_value:.3f} "
                        f"teleop_active={int(teleop_active)} "
                        f"grasped={grasped} "
                        f"tcp={tcp_xyz}"
                    ,
                        flush=True,
                    )

                if not controller_pose.is_valid:
                    prev_pos = None
                    prev_quat_wxyz = None
                    continue

                pos = np.asarray(controller_pose.position_xyz, dtype=np.float32)
                quat_wxyz = _quat_xyzw_to_wxyz(
                    np.asarray(controller_pose.orientation_xyzw, dtype=np.float64)
                )
                quat_sim_wxyz = _xr_quat_to_sim_wxyz(quat_wxyz)

                if prev_pos is None or prev_quat_wxyz is None or not teleop_active:
                    prev_pos = pos.copy()
                    prev_quat_wxyz = quat_sim_wxyz.copy()
                    if not teleop_active:
                        if args.teleop_apply_gripper_qpos_direct:
                            target_gripper_qpos = _interpolate_gripper_qpos(
                                preset_open_gripper_qpos,
                                preset_grab_gripper_qpos,
                                pinch_amount,
                            )
                            obs, info = _apply_gripper_qpos_direct(env, target_gripper_qpos)
                            _write_vr_stream_frame(
                                stream_writer,
                                {
                                    **_obs_with_vr_stream_settings(obs, args),
                                    "sensor_data": {"3rd_view_camera": {"rgb": _get_vr_stream_source_frame(env, obs, args)}},
                                },
                            )
                            _maybe_render_live(env, args)
                    continue

                delta_pos = (pos - prev_pos) * float(args.vr_translation_gain)
                delta_pos[np.abs(delta_pos) < float(args.vr_pos_deadband_m)] = 0.0

                delta_quat = qmult(quat_sim_wxyz, qinverse(prev_quat_wxyz))
                delta_rot = np.array(quat2euler(delta_quat, axes="sxyz"), dtype=np.float32)
                delta_rot *= float(args.vr_rotation_gain)
                delta_rot[np.abs(delta_rot) < float(args.vr_rot_deadband_rad)] = 0.0

                mapped_delta_pos = _map_vr_translation_via_camera(
                    delta_pos,
                    camera_right=camera_right,
                    camera_forward=camera_forward,
                )
                _, mapped_delta_rot = _remap_sim_delta(
                    [0.0, 0.0, 0.0],
                    delta_rot.tolist(),
                )
                gripper_cmd = 2.0 * pinch_amount - 1.0

                action = torch.tensor(
                    [[
                        mapped_delta_pos[0],
                        mapped_delta_pos[1],
                        mapped_delta_pos[2],
                        mapped_delta_rot[0],
                        mapped_delta_rot[1],
                        mapped_delta_rot[2],
                        gripper_cmd,
                    ]],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                obs, _, terminated, truncated, info = env.step(action)
                _maybe_render_live(env, args)

                if args.sim_apply_ik_qpos_direct:
                    arm_controller = _sim_arm_controller(env)
                    target_qpos = None if arm_controller is None else getattr(arm_controller, "_target_qpos", None)
                    obs, info = _apply_sim_arm_target_qpos_direct(env, target_qpos)
                    _maybe_render_live(env, args)

                if args.teleop_apply_gripper_qpos_direct:
                    target_gripper_qpos = _interpolate_gripper_qpos(
                        preset_open_gripper_qpos,
                        preset_grab_gripper_qpos,
                        pinch_amount,
                    )
                    obs, info = _apply_gripper_qpos_direct(env, target_gripper_qpos)
                    _maybe_render_live(env, args)

                _write_vr_stream_frame(
                    stream_writer,
                    {
                        **_obs_with_vr_stream_settings(obs, args),
                        "sensor_data": {"3rd_view_camera": {"rgb": _get_vr_stream_source_frame(env, obs, args)}},
                    },
                )

                prev_pos = pos.copy()
                prev_quat_wxyz = quat_sim_wxyz.copy()

                current_qpos = env.unwrapped.agent.robot.qpos[0].detach().cpu().numpy().astype(np.float32)
                robot_init_qpos[:] = current_qpos

                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    print("Episode ended, resetting environment state.")
                    obs, info = _reset_current_state(
                        env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        current_qpos,
                    )
                    _write_vr_stream_frame(
                        stream_writer,
                        {
                            **_obs_with_vr_stream_settings(obs, args),
                            "sensor_data": {"3rd_view_camera": {"rgb": _get_vr_stream_source_frame(env, obs, args)}},
                        },
                    )
                    prev_pos = None
                    prev_quat_wxyz = None
    finally:
        if 'stream_writer' in locals() and stream_writer is not None:
            stream_writer.close()
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
