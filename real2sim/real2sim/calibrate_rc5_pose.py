import ast
from dataclasses import dataclass
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
import tyro

from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.visualization import images_to_video
from real2sim.rc5_calibrated_presets import HAND_GRAB_POSE_FILE, HAND_OPEN_POSE_FILE
from real2sim.debug_paths import RC5_POSE_DIR, resolve_spawn_grid_state_path
from real2sim.openreal2sim_validation import (
    AIRI_CUBE_COLOR_NAMES,
    AIRI_TABLE_EMPTY_ASSET_DIR,
    DEFAULT_PLATE_MODEL_NAME,
    DEFAULT_SOURCE_MODEL_NAME,
    PROBE_CARROT_POSE_P,
    PROBE_CARROT_POSE_Q,
    PROBE_PLATE_POSE_P,
    PROBE_PLATE_POSE_Q,
    RC5AeroHandRightOpenReal2SimValidation,
    WRIST_CAMERA_NAME,
    default_spawn_grid_state,
    observation_camera_lookat_state,
    observation_camera_override,
    plate_xy_half_extent,
    plate_resting_center_z_offset,
    pose_from_eye_target_roll,
    scene_ground_point,
    source_xy_half_extent,
    source_resting_center_z_offset,
)
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

WRIST_INSET_TOP = 4
WRIST_INSET_LEFT = 4
WRIST_INSET_MARGIN = 4
WRIST_INSET_BORDER = 4
WRIST_INSET_HEIGHT = 224
WRIST_INSET_WIDTH = 168

DEFAULT_BASE_POSE_P = np.array([-0.3, -1.12, 0.1204], dtype=np.float32)
DEFAULT_BASE_POSE_Q = np.array([0.7071, 0.0, 0.0, -0.7071], dtype=np.float32)
DEFAULT_ROBOT_INIT_QPOS = np.array(
    [
        2.8,
        -1.1085608005523682,
        1.4299265146255493,
        -0.6574097871780396,
        -1.3829675912857056,
        0.18142791092395782,
        1.7452759742736816,
        1.103040631278418e-05,
        1.000440079224063e-05,
        8.387401067011524e-06,
        6.89399257680634e-06,
        0.00025434582494199276,
        1.2627153409994207e-05,
        1.0071582437376492e-05,
        7.455143531842623e-06,
        3.830670266324887e-06,
        0.00025852309772744775,
        1.4329766599985305e-06,
        1.0554273330853903e-06,
        6.426599838960101e-07,
        3.0619184965274826e-10,
        1.8202814544565626e-07,
    ],
    dtype=np.float32,
)
DEFAULT_OBJECT_POSITION = np.array([0.050495, -0.936646, 0.0], dtype=np.float32)
DEFAULT_LOAD_STATE_FILE = ""
HAND_OPEN_STATE_FILE = HAND_OPEN_POSE_FILE
HAND_GRAB_STATE_FILE = HAND_GRAB_POSE_FILE
SIM_DELTA_REMAP_RPY_DEG = np.array([0.0, 0.0, 90.0], dtype=np.float32)
HAND_CONTROL_SPECS = [
    ("thumb_abd", ["right_thumb_cmc_abd"], 0.0, 1.7453),
    ("thumb_cmc_flex", ["right_thumb_cmc_flex"], 0.0, 0.9599),
    ("thumb_mcp", ["right_thumb_mcp"], 0.0, 1.5708),
    ("thumb_ip", ["right_thumb_ip"], 0.0, 1.5708),
    ("index_mcp", ["right_index_mcp_flex"], 0.0, 1.5708),
    ("index_pip", ["right_index_pip"], 0.0, 1.5708),
    ("index_dip", ["right_index_dip"], 0.0, 1.5708),
    ("middle_mcp", ["right_middle_mcp_flex"], 0.0, 1.5708),
    ("middle_pip", ["right_middle_pip"], 0.0, 1.5708),
    ("middle_dip", ["right_middle_dip"], 0.0, 1.5708),
    ("ring_mcp", ["right_ring_mcp_flex"], 0.0, 1.5708),
    ("ring_pip", ["right_ring_pip"], 0.0, 1.5708),
    ("ring_dip", ["right_ring_dip"], 0.0, 1.5708),
    ("pinky_mcp", ["right_pinky_mcp_flex"], 0.0, 1.5708),
    ("pinky_pip", ["right_pinky_pip"], 0.0, 1.5708),
    ("pinky_dip", ["right_pinky_dip"], 0.0, 1.5708),
]
HAND_CONTROL_SELECT_KEYS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
]


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


def _resize_nearest_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[y_idx][:, x_idx]


def _compose_wrist_inset(
    scene_rgb: np.ndarray,
    wrist_rgb: np.ndarray | None,
    *,
    bottom_right: bool = False,
) -> np.ndarray:
    if wrist_rgb is None:
        return scene_rgb
    wrist_rgb = _resize_nearest_image(wrist_rgb, WRIST_INSET_HEIGHT, WRIST_INSET_WIDTH)
    border = WRIST_INSET_BORDER
    inset_h = WRIST_INSET_HEIGHT
    inset_w = WRIST_INSET_WIDTH
    if bottom_right:
        margin = WRIST_INSET_MARGIN
        top = scene_rgb.shape[0] - inset_h - 2 * border - margin
        left = scene_rgb.shape[1] - inset_w - 2 * border - margin
    else:
        top = WRIST_INSET_TOP
        left = WRIST_INSET_LEFT
    out = scene_rgb.copy()
    out[top:top + inset_h + 2 * border, left:left + inset_w + 2 * border, :] = 0
    out[top + border:top + border + inset_h, left + border:left + border + inset_w, :] = wrist_rgb
    return out


def _extract_obs_frame(obs: dict, *, wrist_inset_bottom_right: bool = False) -> np.ndarray:
    sensor_data = obs.get("sensor_data", {})
    if "3rd_view_camera" not in sensor_data:
        available = sorted(sensor_data.keys())
        raise KeyError(
            "Expected observation camera '3rd_view_camera' was not found. "
            f"Available sensor_data keys: {available}"
        )
    scene_rgb = _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])
    wrist_data = sensor_data.get(WRIST_CAMERA_NAME)
    wrist_rgb = None if wrist_data is None else _to_uint8_image(wrist_data["rgb"])
    return _compose_wrist_inset(scene_rgb, wrist_rgb, bottom_right=wrist_inset_bottom_right)


def _remap_sim_delta(delta_pos: list[float], delta_rot: list[float]) -> tuple[list[float], list[float]]:
    remap_quat = euler2quat(*np.deg2rad(SIM_DELTA_REMAP_RPY_DEG), axes="sxyz")
    remap_rot = quat2mat(remap_quat)
    mapped_pos = remap_rot @ np.asarray(delta_pos, dtype=np.float32)
    mapped_rot = remap_rot @ np.asarray(delta_rot, dtype=np.float32)
    return mapped_pos.tolist(), mapped_rot.tolist()


@dataclass
class Args:
    output_dir: str = str(RC5_POSE_DIR)
    image_prefix: str = "openreal2sim_rc5_pose"
    keep_history: bool = False
    teleop_record_video: bool = False
    teleop_video_dir: str = str(RC5_POSE_DIR / "teleop_videos")
    teleop_video_name: str = "calibrate_rc5_pose_teleop"
    teleop_video_fps: int = 10
    sim_backend: str = "gpu"
    render_backend_override: str = ""
    shader: str = "default"
    enable_shadow: bool = True
    human_render_shader: str = ""
    viewer_shader: str = ""
    scene_asset_dir: str = str(AIRI_TABLE_EMPTY_ASSET_DIR)
    observation_camera_mode: str = "manual_best"
    live_viewer: bool = False
    live_viewer_max_hz: float = 30.0
    live_viewer_force_cpu: bool = False
    live_viewer_width: int = 0
    live_viewer_height: int = 0
    live_viewer_fov: float = 0.0
    sim_freq_override: int = 0
    control_freq_override: int = 0
    use_environment_map: bool = True
    use_wrist_camera: bool = True
    seed: int = 0
    load_background: bool = True
    clean_scene: bool = True
    use_probe_objects: bool = False
    task_mode: str = "probe_on_plate"
    airi_cube_target_color: str = "red"
    probe_source_model_name: str = DEFAULT_SOURCE_MODEL_NAME
    probe_plate_model_name: str = DEFAULT_PLATE_MODEL_NAME
    show_debug_markers: bool = False
    robot_far_away: bool = False
    base_dx: float = 0.0
    base_dy: float = 0.0
    base_dz: float = 0.0
    load_state_file: str = DEFAULT_LOAD_STATE_FILE
    sim_apply_ik_qpos_direct: bool = False
    disable_self_collisions: bool = False
    teleop_apply_gripper_qpos_direct: bool = False
    teleop_idle_step_sec: float = 2.0
    teleop_translation_step: float = 0.02
    teleop_rotation_step: float = 0.12
    teleop_pinch_step: float = 0.1
    teleop_motion_substeps: int = 1
    teleop_contact_debug: bool = True
    teleop_contact_debug_force_threshold: float = 0.5
    teleop_contact_debug_log_file: str = str(RC5_POSE_DIR / "teleop_contact_debug.log")
    object_position_xyz: tuple[float, float, float] = (
        float(DEFAULT_OBJECT_POSITION[0]),
        float(DEFAULT_OBJECT_POSITION[1]),
        float(DEFAULT_OBJECT_POSITION[2]),
    )
    plate_position_xyz: tuple[float, float, float] = (
        float(DEFAULT_OBJECT_POSITION[0]),
        float(DEFAULT_OBJECT_POSITION[1] + 0.18),
        0.0,
    )
    spawn_grid_state_file: str = ""
    grid_pair_index: int = -1
    grid_steps_x: int = 6
    grid_steps_y: int = 6
    grid_min_pair_distance_xy: float = 0.10
    grid_min_robot_clearance_xy: float = 0.18


@register_agent(override=True)
class RC5AeroHandRightOpenReal2SimValidationNoSelfCollision(
    RC5AeroHandRightOpenReal2SimValidation
):
    uid = "rc5_aero_hand_right_openreal2sim_validation_no_self_collision"
    disable_self_collisions = True


def _robot_uid(args: Args) -> str:
    if args.disable_self_collisions:
        return RC5AeroHandRightOpenReal2SimValidationNoSelfCollision.uid
    return RC5AeroHandRightOpenReal2SimValidation.uid


def _build_env(
    args: Args,
    camera_mode: str,
    camera_eye: np.ndarray | None = None,
    camera_target: np.ndarray | None = None,
    camera_roll_deg: float | None = None,
    camera_fov: float | None = None,
) -> BaseEnv:
    sim_backend = args.sim_backend
    render_backend = args.render_backend_override or "gpu"
    if args.live_viewer:
        if args.live_viewer_force_cpu:
            render_backend = "cpu"
        if args.live_viewer_force_cpu and sim_backend != "cpu":
            print(
                f"Live viewer enabled: overriding sim_backend {sim_backend!r} -> 'cpu' "
                "to avoid SAPIEN CUDA render-sync issues."
            )
            sim_backend = "cpu"

    sim_config_override = {}
    if int(getattr(args, "sim_freq_override", 0)) > 0:
        sim_config_override["sim_freq"] = int(args.sim_freq_override)
    if int(getattr(args, "control_freq_override", 0)) > 0:
        sim_config_override["control_freq"] = int(args.control_freq_override)

    if camera_eye is not None and camera_target is not None:
        pose = pose_from_eye_target_roll(
            eye=camera_eye,
            target=camera_target,
            roll_deg=0.0 if camera_roll_deg is None else camera_roll_deg,
        )
        camera_cfg = {
            "pose": pose,
            "intrinsic": None,
            "fov": 1.0 if camera_fov is None else camera_fov,
            "near": 0.01,
            "far": 10.0,
        }
    else:
        camera_cfg = observation_camera_override(camera_mode, scene_asset_dir=args.scene_asset_dir)

    viewer_camera_configs = {"shader_pack": args.viewer_shader or args.human_render_shader or args.shader}
    if int(getattr(args, "live_viewer_width", 0)) > 0:
        viewer_camera_configs["width"] = int(args.live_viewer_width)
    if int(getattr(args, "live_viewer_height", 0)) > 0:
        viewer_camera_configs["height"] = int(args.live_viewer_height)
    if float(getattr(args, "live_viewer_fov", 0.0)) > 0.0:
        viewer_camera_configs["fov"] = float(args.live_viewer_fov)

    sensor_configs = {
        "shader_pack": args.shader,
        "3rd_view_camera": camera_cfg,
    }
    wrist_camera_pose = getattr(args, "wrist_camera_pose", None)
    if wrist_camera_pose is not None:
        sensor_configs[WRIST_CAMERA_NAME] = {
            "pose": wrist_camera_pose,
            "intrinsic": None,
            "fov": float(getattr(args, "wrist_camera_fov", 1.5707963267948966)),
            "near": float(getattr(args, "wrist_camera_near", 0.01)),
            "far": float(getattr(args, "wrist_camera_far", 2.0)),
        }

    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="human" if args.live_viewer else "rgb_array",
        enable_shadow=bool(args.enable_shadow),
        sensor_configs=sensor_configs,
        human_render_camera_configs={"shader_pack": args.human_render_shader or args.shader},
        viewer_camera_configs=viewer_camera_configs,
        sim_config=sim_config_override,
        num_envs=1,
        sim_backend=sim_backend,
        render_backend=render_backend,
        robot_uids=_robot_uid(args),
        scene_asset_dir=args.scene_asset_dir,
        use_wrist_camera=bool(args.use_wrist_camera),
    )


def _default_probe_object_pose_p(args: Args) -> np.ndarray:
    pose_p = np.array(args.object_position_xyz, dtype=np.float32)
    if abs(float(pose_p[2])) > 1e-8:
        return pose_p
    ground = scene_ground_point(args.scene_asset_dir)
    pose_p[2] = float(
        ground[2]
        + source_resting_center_z_offset(
            PROBE_CARROT_POSE_Q, model_name=args.probe_source_model_name
        )
    )
    return pose_p


def _default_probe_plate_pose_p(args: Args) -> np.ndarray:
    pose_p = np.array(args.plate_position_xyz, dtype=np.float32)
    if abs(float(pose_p[2])) > 1e-8:
        return pose_p
    ground = scene_ground_point(args.scene_asset_dir)
    pose_p[2] = float(
        ground[2]
        + plate_resting_center_z_offset(
            PROBE_PLATE_POSE_Q, model_name=args.probe_plate_model_name
        )
    )
    return pose_p


def _load_spawn_grid_state(path: str, scene_asset_dir: str) -> dict[str, np.ndarray]:
    if not path:
        return default_spawn_grid_state(scene_asset_dir)
    state: dict[str, object] = {}
    resolved_path = resolve_spawn_grid_state_path(path)
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key.strip()] = ast.literal_eval(value.strip())
    return {
        "center_xy": np.array(state["center_xy"], dtype=np.float32),
        "size_xy": np.array(state["size_xy"], dtype=np.float32),
        "yaw_deg": np.array([float(state.get("yaw_deg", 0.0))], dtype=np.float32),
        "z": np.array([float(state["z"])], dtype=np.float32),
    }


def _make_rotated_grid(center_xy: np.ndarray, size_xy: np.ndarray, yaw_deg: float, steps_x: int, steps_y: int) -> np.ndarray:
    xs = np.linspace(center_xy[0] - size_xy[0] / 2.0, center_xy[0] + size_xy[0] / 2.0, steps_x)
    ys = np.linspace(center_xy[1] - size_xy[1] / 2.0, center_xy[1] + size_xy[1] / 2.0, steps_y)
    yaw = np.deg2rad(float(yaw_deg))
    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    pts = []
    for x in xs:
        for y in ys:
            local = np.array([x - center_xy[0], y - center_xy[1]], dtype=np.float32)
            rotated = rot @ local
            pts.append([float(center_xy[0] + rotated[0]), float(center_xy[1] + rotated[1])])
    return np.array(pts, dtype=np.float32)


def _apply_grid_pair_override(args: Args, robot_base_pose_p: np.ndarray) -> None:
    if args.grid_pair_index < 0:
        return
    grid = _load_spawn_grid_state(args.spawn_grid_state_file, args.scene_asset_dir)
    center_xy = np.array(grid["center_xy"], dtype=np.float32)
    size_xy = np.array(grid["size_xy"], dtype=np.float32)
    yaw_deg = float(grid["yaw_deg"][0])
    footprint_margin = max(
        source_xy_half_extent(PROBE_CARROT_POSE_Q),
        plate_xy_half_extent(PROBE_PLATE_POSE_Q),
    )
    usable_size_xy = np.maximum(size_xy - 2.0 * footprint_margin, 0.01)
    grid_points = _make_rotated_grid(
        center_xy,
        usable_size_xy,
        yaw_deg=yaw_deg,
        steps_x=int(args.grid_steps_x),
        steps_y=int(args.grid_steps_y),
    )
    robot_base_xy = np.array(robot_base_pose_p[:2], dtype=np.float32)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for carrot_xy in grid_points:
        for plate_xy in grid_points:
            if np.allclose(carrot_xy, plate_xy):
                continue
            if np.linalg.norm(plate_xy - carrot_xy) <= float(args.grid_min_pair_distance_xy):
                continue
            if np.linalg.norm(carrot_xy - robot_base_xy) <= float(args.grid_min_robot_clearance_xy):
                continue
            if np.linalg.norm(plate_xy - robot_base_xy) <= float(args.grid_min_robot_clearance_xy):
                continue
            pairs.append((carrot_xy.copy(), plate_xy.copy()))
    if not pairs:
        raise ValueError("No valid grid pairs available with the current constraints.")
    if args.grid_pair_index >= len(pairs):
        raise ValueError(f"grid_pair_index={args.grid_pair_index} out of range for {len(pairs)} valid pairs")
    carrot_xy, plate_xy = pairs[args.grid_pair_index]
    args.object_position_xyz = (float(carrot_xy[0]), float(carrot_xy[1]), 0.0)
    args.plate_position_xyz = (float(plate_xy[0]), float(plate_xy[1]), 0.0)
    print(
        "Selected grid pair "
        f"{args.grid_pair_index}/{len(pairs) - 1}: "
        f"carrot_xy={carrot_xy.tolist()} plate_xy={plate_xy.tolist()} "
        f"grid_center={center_xy.tolist()} grid_size={size_xy.tolist()} "
        f"usable_size={usable_size_xy.tolist()} yaw_deg={yaw_deg:.2f}"
    )


def _scene_options(args: Args) -> dict:
    task_mode = str(getattr(args, "task_mode", "probe_on_plate")).strip().lower()
    use_airi_cubes = task_mode in {"airi_cube_pickup", "cube_pickup", "pick_cube"}
    use_probe_objects = args.use_probe_objects and not use_airi_cubes
    show_debug_markers = args.show_debug_markers
    if args.clean_scene:
        show_debug_markers = False
        # Keep "clean scene" from forcing out probe objects when they are
        # explicitly requested for teleop/calibration runs.
        if not args.use_probe_objects:
            use_probe_objects = False
    options = {
        "load_background": args.load_background,
        "use_probe_objects": use_probe_objects,
        "use_airi_cubes": use_airi_cubes,
        "airi_cube_target_color": str(getattr(args, "airi_cube_target_color", AIRI_CUBE_COLOR_NAMES[0])),
        "show_debug_markers": show_debug_markers,
        "robot_far_away": args.robot_far_away,
        "use_environment_map": bool(args.use_environment_map),
    }
    if use_probe_objects:
        options.update(
            {
                "probe_source_model_name": args.probe_source_model_name,
                "probe_plate_model_name": args.probe_plate_model_name,
                "probe_carrot_pose_p": _default_probe_object_pose_p(args).tolist(),
                "probe_carrot_pose_q": PROBE_CARROT_POSE_Q,
                "probe_plate_pose_p": _default_probe_plate_pose_p(args).tolist(),
                "probe_plate_pose_q": PROBE_PLATE_POSE_Q,
            }
        )
    if args.spawn_grid_state_file:
        grid = _load_spawn_grid_state(args.spawn_grid_state_file, args.scene_asset_dir)
        options["show_spawn_grid"] = True
        options["spawn_grid_center_xy"] = np.array(grid["center_xy"], dtype=np.float32).tolist()
        options["spawn_grid_size_xy"] = np.array(grid["size_xy"], dtype=np.float32).tolist()
        options["spawn_grid_yaw_deg"] = float(grid["yaw_deg"][0])
        options["spawn_grid_z"] = float(grid["z"][0])
    return options


def _maybe_render_live(env: BaseEnv, args: Args) -> None:
    if args.live_viewer:
        max_hz = float(getattr(args, "live_viewer_max_hz", 0.0))
        if max_hz > 0.0:
            now = time.monotonic()
            last_render = float(getattr(env.unwrapped, "_last_live_viewer_render_time", 0.0))
            min_period = 1.0 / max_hz
            if last_render > 0.0 and (now - last_render) < min_period:
                return
            env.unwrapped._last_live_viewer_render_time = now
        env.render_human()


def _apply_camera_state(
    env: BaseEnv,
    camera_eye: np.ndarray,
    camera_target: np.ndarray,
    camera_roll_deg: float,
    camera_fov: float,
) -> None:
    pose = pose_from_eye_target_roll(
        eye=np.asarray(camera_eye, dtype=np.float32),
        target=np.asarray(camera_target, dtype=np.float32),
        roll_deg=float(camera_roll_deg),
    )

    sensor = env.unwrapped._sensors["3rd_view_camera"]
    sensor.config.pose = sensor.config.pose.create(pose)
    sensor.config.fov = float(camera_fov)
    sensor.config.intrinsic = None
    sensor.camera.local_pose = pose
    sensor.camera.set_fovy(float(camera_fov), compute_x=True)
    sensor.camera._cached_local_pose = None
    sensor.camera._cached_model_matrix = None

    for camera_name in ("render_camera", "sanity_render_camera"):
        if camera_name not in env.unwrapped._human_render_cameras:
            continue
        render_camera = env.unwrapped._human_render_cameras[camera_name]
        render_camera.config.pose = render_camera.config.pose.create(pose)
        render_camera.config.fov = float(camera_fov)
        render_camera.camera.local_pose = pose
        render_camera.camera.set_fovy(float(camera_fov), compute_x=True)
        render_camera.camera._cached_local_pose = None
        render_camera.camera._cached_model_matrix = None

    if env.unwrapped.viewer is not None:
        env.unwrapped.viewer.set_camera_pose(pose)

    env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=True)


def _current_obs_info(env: BaseEnv, args: Args) -> tuple[dict, dict]:
    info = env.unwrapped.get_info()
    obs = env.unwrapped.get_obs(info)
    _maybe_render_live(env, args)
    return obs, info


def _sanitize_live_viewer_environment(args: Args) -> None:
    if not args.live_viewer:
        return

    stripped_vars = []
    for env_name in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env_value = os.environ.get(env_name)
        if env_value and "SteamVR" in env_value:
            stripped_vars.append((env_name, env_value))
            os.environ.pop(env_name, None)

    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_library_path:
        kept_entries = [entry for entry in ld_library_path.split(":") if "SteamVR" not in entry]
        if len(kept_entries) != len(ld_library_path.split(":")):
            stripped_vars.append(("LD_LIBRARY_PATH", ld_library_path))
            os.environ["LD_LIBRARY_PATH"] = ":".join(kept_entries)

    print("Live viewer diagnostics:")
    print(f"  DISPLAY={os.environ.get('DISPLAY', '')!r}")
    print(f"  XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE', '')!r}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"  QT_PLUGIN_PATH={os.environ.get('QT_PLUGIN_PATH', '')!r}")
    print(
        f"  QT_QPA_PLATFORM_PLUGIN_PATH="
        f"{os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', '')!r}"
    )
    if stripped_vars:
        print("  Stripped SteamVR-related GUI env vars:")
        for env_name, env_value in stripped_vars:
            print(f"    {env_name}={env_value!r}")
    else:
        print("  No SteamVR-specific Qt env vars were stripped.")


def _read_command_line_nonblocking(prompt: str, timeout_sec: float, *, show_prompt: bool) -> str | None:
    if show_prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    line = sys.stdin.readline()
    if line == "":
        return "quit"
    return line.strip()


def _save_snapshot(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    index: int,
) -> None:
    obs, info = env.reset(
        seed=args.seed,
        options={
            "reconfigure": True,
            **_scene_options(args),
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
        },
    )
    frame = _extract_obs_frame(obs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / f"{args.image_prefix}_latest.png"
    latest_meta_path = output_dir / f"{args.image_prefix}_latest.txt"
    image_path = latest_path
    meta_path = latest_meta_path
    if args.keep_history:
        image_path = output_dir / f"{args.image_prefix}_{index:03d}.png"
        meta_path = output_dir / f"{args.image_prefix}_{index:03d}.txt"

    tcp_pose = obs["extra"]["tcp_pose"][0].cpu().numpy().tolist()
    hand_joint_qpos = _hand_joint_qpos_summary(env)
    info_summary = {
        k: (v[0].item() if torch.is_tensor(v) else v)
        for k, v in info.items()
    }

    iio.imwrite(latest_path, frame)
    if args.keep_history:
        iio.imwrite(image_path, frame)
    meta_path.write_text(
        "\n".join(
            [
                f"robot_base_pose_p={robot_base_pose_p.tolist()}",
                f"robot_base_pose_q={robot_base_pose_q.tolist()}",
                f"robot_init_qpos={robot_init_qpos.tolist()}",
                f"hand_joint_qpos={hand_joint_qpos}",
                f"tcp_pose={tcp_pose}",
                f"info={info_summary}",
            ]
        ),
        encoding="utf-8",
    )
    if args.keep_history:
        latest_meta_path.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved {image_path}")
    print(f"Saved {meta_path}")


def _save_snapshot_from_obs(
    env: BaseEnv,
    obs: dict,
    info: dict,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    index: int,
) -> None:
    frame = _extract_obs_frame(obs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / f"{args.image_prefix}_latest.png"
    latest_meta_path = output_dir / f"{args.image_prefix}_latest.txt"
    image_path = latest_path
    meta_path = latest_meta_path
    if args.keep_history:
        image_path = output_dir / f"{args.image_prefix}_{index:03d}.png"
        meta_path = output_dir / f"{args.image_prefix}_{index:03d}.txt"

    tcp_pose = obs["extra"]["tcp_pose"][0].cpu().numpy().tolist()
    robot_qpos = obs["agent"]["qpos"][0].cpu().numpy().tolist()
    hand_joint_qpos = _hand_joint_qpos_summary(env)
    info_summary = {
        k: (v[0].item() if torch.is_tensor(v) else v)
        for k, v in info.items()
    }

    iio.imwrite(latest_path, frame)
    if args.keep_history:
        iio.imwrite(image_path, frame)
    meta_path.write_text(
        "\n".join(
            [
                f"robot_base_pose_p={robot_base_pose_p.tolist()}",
                f"robot_base_pose_q={robot_base_pose_q.tolist()}",
                f"robot_init_qpos={robot_qpos}",
                f"hand_joint_qpos={hand_joint_qpos}",
                f"tcp_pose={tcp_pose}",
                f"info={info_summary}",
            ]
        ),
        encoding="utf-8",
    )
    if args.keep_history:
        latest_meta_path.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Saved {image_path}")
    print(f"Saved {meta_path}")


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


def _named_joint_qpos(env: BaseEnv, joint_names: list[str]) -> dict[str, float]:
    robot = env.unwrapped.agent.robot
    qpos = robot.qpos[0].detach().cpu().numpy().astype(np.float32)
    named = {}
    for joint_name in joint_names:
        joint = robot.active_joints_map.get(joint_name)
        if joint is None:
            continue
        named[joint_name] = float(qpos[int(joint.active_index[0].item())])
    return named


def _hand_joint_qpos_summary(env: BaseEnv) -> dict[str, float]:
    agent = env.unwrapped.agent
    return _named_joint_qpos(env, list(getattr(agent, "gripper_joint_names", [])))


def _probe_carrot_grasped(env: BaseEnv) -> bool | None:
    unwrapped = env.unwrapped
    if not getattr(unwrapped, "use_probe_objects", False):
        if not getattr(unwrapped, "use_airi_cubes", False):
            return None
        target_color = getattr(unwrapped, "airi_cube_target_color", "")
        cube_actors = getattr(unwrapped, "airi_cube_actors", {})
        probe_carrot = cube_actors.get(target_color)
    else:
        probe_carrot = getattr(unwrapped, "probe_carrot", None)
    if probe_carrot is None:
        return None
    try:
        grasped = unwrapped.agent.is_grasping(probe_carrot)
        if torch.is_tensor(grasped):
            return bool(grasped[0].item())
        return bool(grasped)
    except Exception:
        return None


def _probe_carrot_contact_debug(env: BaseEnv, force_threshold: float = 0.5) -> str | None:
    unwrapped = env.unwrapped
    probe_carrot = getattr(unwrapped, "probe_carrot", None)
    if not getattr(unwrapped, "use_probe_objects", False) or probe_carrot is None:
        return None

    agent = unwrapped.agent
    scene = unwrapped.scene

    def _pair_norm(link) -> float:
        if link is None:
            return 0.0
        try:
            force = scene.get_pairwise_contact_forces(link, probe_carrot)
            if torch.is_tensor(force):
                return float(torch.linalg.norm(force[0]).item())
            arr = np.asarray(force)
            return float(np.linalg.norm(arr[0]))
        except Exception:
            return 0.0

    try:
        carrot_net_force = float(torch.linalg.norm(probe_carrot.get_net_contact_forces()[0]).item())
    except Exception:
        carrot_net_force = 0.0
    try:
        carrot_lin_vel = float(torch.linalg.norm(probe_carrot.linear_velocity[0]).item())
    except Exception:
        carrot_lin_vel = 0.0
    try:
        carrot_ang_vel = float(torch.linalg.norm(probe_carrot.angular_velocity[0]).item())
    except Exception:
        carrot_ang_vel = 0.0

    thumb_force = _pair_norm(getattr(agent, "thumb_tip_link", None))
    index_force = _pair_norm(getattr(agent, "index_tip_link", None))
    middle_force = _pair_norm(getattr(agent.robot, "links_map", {}).get("right_middle_tip_link"))
    ring_force = _pair_norm(getattr(agent.robot, "links_map", {}).get("right_ring_tip_link"))
    pinky_force = _pair_norm(getattr(agent.robot, "links_map", {}).get("right_pinky_tip_link"))
    max_force = max(
        carrot_net_force,
        thumb_force,
        index_force,
        middle_force,
        ring_force,
        pinky_force,
    )
    if max_force < force_threshold:
        return None

    return (
        "carrot["
        f"net={carrot_net_force:.2f}N "
        f"lv={carrot_lin_vel:.3f} av={carrot_ang_vel:.3f} "
        f"thumb={thumb_force:.2f} idx={index_force:.2f} mid={middle_force:.2f} "
        f"ring={ring_force:.2f} pinky={pinky_force:.2f}"
        "]"
    )


def _append_teleop_contact_log(path: str | Path, line: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _save_teleop_video(args: Args, frames: list[np.ndarray]) -> Path | None:
    if not args.teleop_record_video or not frames:
        return None

    output_dir = Path(args.teleop_video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{args.teleop_video_name}_*.mp4"))
    next_idx = len(existing)
    video_stem = f"{args.teleop_video_name}_{next_idx:03d}"
    images_to_video(
        frames,
        str(output_dir),
        video_stem,
        fps=max(1, int(args.teleop_video_fps)),
        verbose=False,
    )
    return output_dir / f"{video_stem}.mp4"


def _print_help() -> None:
    print("Commands:")
    print("  load PATH            load robot base pose and qpos from a state file")
    print("  move dx dy dz        move robot base in world coordinates")
    print("  x dx                 move robot base along X")
    print("  y dy                 move robot base along Y")
    print("  z dz                 move robot base along Z")
    print("  jogbase              enter WASD + Q/E yaw base-jog mode")
    print("  jogjoints            enter joint-jog mode for arm joints 1-6")
    print("  joghand              enter hand-jog mode for thumb + individual finger sliders")
    print("  joint idx delta      add delta to one qpos entry")
    print("  qpos i0 i1 ...       set full robot qpos vector")
    print("  cam MODE             switch camera preset")
    print("  cams                 list camera presets")
    print("  cameye dx dy dz      move camera eye")
    print("  camtarget dx dy dz   move camera target")
    print("  camroll deg          add camera roll")
    print("  camfov delta         add camera fov delta")
    print("  teleop               enter WASDQE ee-delta mode")
    print("  reset                reset to default base pose and qpos")
    print("  show                 print current base pose and qpos")
    print("  save                 save current snapshot again")
    print("  help                 show this message")
    print("  quit                 exit")


def _read_single_key(timeout_sec: float | None = None) -> str | None:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if timeout_sec is not None:
            readable, _, _ = select.select([sys.stdin], [], [], timeout_sec)
            if not readable:
                return None
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _sim_arm_controller(env: BaseEnv):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm")
    return controller


def _apply_sim_arm_target_qpos_direct(env: BaseEnv, target_qpos) -> tuple[dict, dict]:
    arm_controller = _sim_arm_controller(env)
    if arm_controller is None or target_qpos is None:
        info = env.unwrapped.get_info()
        return env.unwrapped.get_obs(info), info

    robot = env.unwrapped.agent.robot
    full_qpos = robot.get_qpos().clone()
    full_qvel = robot.get_qvel().clone()
    active_joint_indices = arm_controller.active_joint_indices.long()
    target_qpos = target_qpos.to(dtype=full_qpos.dtype, device=full_qpos.device)
    full_qpos[:, active_joint_indices] = target_qpos
    full_qvel[:, active_joint_indices] = 0.0
    robot.set_qpos(full_qpos)
    robot.set_qvel(full_qvel)
    if env.unwrapped.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()
        env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env.unwrapped.scene._gpu_fetch_all()
    info = env.unwrapped.get_info()
    obs = env.unwrapped.get_obs(info)
    return obs, info


def _gripper_active_indices(env: BaseEnv) -> list[int]:
    agent = env.unwrapped.agent
    robot = agent.robot
    return [
        int(robot.active_joints_map[joint_name].active_index[0].item())
        for joint_name in agent.gripper_joint_names
    ]


def _state_file_gripper_qpos(path: str | Path, env: BaseEnv) -> np.ndarray:
    _, _, robot_init_qpos = _load_state_file(str(path))
    return robot_init_qpos[_gripper_active_indices(env)].astype(np.float32)


def _apply_gripper_qpos_direct(env: BaseEnv, target_gripper_qpos: np.ndarray) -> tuple[dict, dict]:
    robot = env.unwrapped.agent.robot
    full_qpos = robot.get_qpos().clone()
    full_qvel = robot.get_qvel().clone()
    active_joint_indices = torch.as_tensor(
        _gripper_active_indices(env), dtype=torch.long, device=full_qpos.device
    )
    target_qpos = torch.as_tensor(
        target_gripper_qpos, dtype=full_qpos.dtype, device=full_qpos.device
    ).view(1, -1)
    full_qpos[:, active_joint_indices] = target_qpos
    full_qvel[:, active_joint_indices] = 0.0
    robot.set_qpos(full_qpos)
    robot.set_qvel(full_qvel)
    if env.unwrapped.gpu_sim_enabled:
        env.unwrapped.scene._gpu_apply_all()
        env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        env.unwrapped.scene._gpu_fetch_all()
    info = env.unwrapped.get_info()
    obs = env.unwrapped.get_obs(info)
    return obs, info


def _interpolate_gripper_qpos(
    open_gripper_qpos: np.ndarray,
    grab_gripper_qpos: np.ndarray,
    pinch_amount: float,
) -> np.ndarray:
    return open_gripper_qpos + pinch_amount * (grab_gripper_qpos - open_gripper_qpos)


def _reset_current_state(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    *,
    reconfigure: bool = False,
):
    obs, info = env.reset(
        seed=args.seed,
        options={
            "reconfigure": reconfigure,
            **_scene_options(args),
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
            "trajectory_instruction": str(getattr(args, "trajectory_instruction", "")).strip(),
        },
    )
    _maybe_render_live(env, args)
    return obs, info


def _rebuild_env_with_camera(
    env: BaseEnv,
    args: Args,
    camera_mode: str,
    camera_eye: np.ndarray,
    camera_target: np.ndarray,
    camera_roll_deg: float,
    camera_fov: float,
) -> BaseEnv:
    env.close()
    return _build_env(
        args,
        camera_mode=camera_mode,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_roll_deg=camera_roll_deg,
        camera_fov=camera_fov,
    )


def _teleop_mode(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    save_idx: int,
):
    step_size = args.teleop_translation_step
    rot_step = args.teleop_rotation_step
    pinch_step = args.teleop_pinch_step
    motion_substeps = max(1, int(args.teleop_motion_substeps))
    current_qpos = env.unwrapped.agent.robot.qpos[0].detach().cpu().numpy().astype(np.float32)
    robot_init_qpos[:] = current_qpos
    preset_open_gripper_qpos = _state_file_gripper_qpos(HAND_OPEN_STATE_FILE, env)
    preset_grab_gripper_qpos = _state_file_gripper_qpos(HAND_GRAB_STATE_FILE, env)
    pinch_amount = _current_gripper_pinch_amount(
        env,
        preset_open_qpos=preset_open_gripper_qpos,
        preset_closed_qpos=preset_grab_gripper_qpos,
    )
    target_gripper_qpos = _interpolate_gripper_qpos(
        preset_open_gripper_qpos, preset_grab_gripper_qpos, pinch_amount
    )

    print("Teleop: W/S=+/-Y, A/D=+/-X, Q/E=+/-Z")
    print("Teleop: I/K pitch, J/L yaw, U/O roll, [ open, ] close, X exit")
    print(f"Teleop: sim_apply_ik_qpos_direct={args.sim_apply_ik_qpos_direct}")
    print(
        f"Teleop: teleop_apply_gripper_qpos_direct="
        f"{args.teleop_apply_gripper_qpos_direct}"
    )
    print(f"Teleop: starting pinch_amount={pinch_amount:.2f}")
    print(f"Teleop: idle sim step every {args.teleop_idle_step_sec:.1f}s")
    print(
        f"Teleop: translation_step={step_size:.4f}m "
        f"rotation_step={rot_step:.3f}rad pinch_step={pinch_step:.2f} "
        f"motion_substeps={motion_substeps}"
    )
    if args.teleop_contact_debug:
        log_path = Path(args.teleop_contact_debug_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        print(f"Teleop: contact debug log -> {log_path}")

    info = env.unwrapped.get_info()
    obs = env.unwrapped.get_obs(info)
    recorded_frames: list[np.ndarray] = []
    if args.teleop_record_video:
        recorded_frames.append(_extract_obs_frame(obs))
        print(
            "Teleop: video recording enabled "
            f"(dir={args.teleop_video_dir}, fps={max(1, int(args.teleop_video_fps))})"
        )
    while True:
        key = _read_single_key(args.teleop_idle_step_sec)
        idle_step = key is None
        if key is None:
            key = "."
        key = key.lower()
        if key == "x":
            video_path = _save_teleop_video(args, recorded_frames)
            if video_path is not None:
                print(f"Saved teleop video: {video_path}")
            print("\nLeaving teleop mode")
            return save_idx

        delta_pos = [0.0, 0.0, 0.0]
        delta_rot = [0.0, 0.0, 0.0]

        if key == "w":
            delta_pos[1] += step_size
        elif key == "s":
            delta_pos[1] -= step_size
        elif key == "a":
            delta_pos[0] -= step_size
        elif key == "d":
            delta_pos[0] += step_size
        elif key == "q":
            delta_pos[2] += step_size
        elif key == "e":
            delta_pos[2] -= step_size
        elif key == "i":
            delta_rot[1] += rot_step
        elif key == "k":
            delta_rot[1] -= rot_step
        elif key == "j":
            delta_rot[2] += rot_step
        elif key == "l":
            delta_rot[2] -= rot_step
        elif key == "u":
            delta_rot[0] += rot_step
        elif key == "o":
            delta_rot[0] -= rot_step
        elif key == "[":
            pinch_amount = max(0.0, pinch_amount - pinch_step)
        elif key == "]":
            pinch_amount = min(1.0, pinch_amount + pinch_step)
        elif idle_step:
            pass
        else:
            continue
        target_gripper_qpos = _interpolate_gripper_qpos(
            preset_open_gripper_qpos, preset_grab_gripper_qpos, pinch_amount
        )

        mapped_delta_pos, mapped_delta_rot = _remap_sim_delta(delta_pos, delta_rot)
        gripper_cmd = 2.0 * pinch_amount - 1.0
        substep_delta_pos = [v / motion_substeps for v in mapped_delta_pos]
        substep_delta_rot = [v / motion_substeps for v in mapped_delta_rot]
        terminated = truncated = None
        for _ in range(motion_substeps):
            action = torch.tensor(
                [[substep_delta_pos[0], substep_delta_pos[1], substep_delta_pos[2], substep_delta_rot[0], substep_delta_rot[1], substep_delta_rot[2], gripper_cmd]],
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
                obs, info = _apply_gripper_qpos_direct(env, target_gripper_qpos)
                _maybe_render_live(env, args)
            if args.teleop_record_video:
                recorded_frames.append(_extract_obs_frame(obs))
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                break
        current_qpos = env.unwrapped.agent.robot.qpos[0].detach().cpu().numpy().astype(np.float32)
        robot_init_qpos[:] = current_qpos

        if bool(terminated[0].item()) or bool(truncated[0].item()):
            obs, info = _reset_current_state(env, args, robot_base_pose_p, robot_base_pose_q, current_qpos)
            if args.teleop_record_video:
                recorded_frames.append(_extract_obs_frame(obs))

        save_idx += 1
        grasped = _probe_carrot_grasped(env)
        carrot_debug = None
        if args.teleop_contact_debug:
            carrot_debug = _probe_carrot_contact_debug(
                env, force_threshold=args.teleop_contact_debug_force_threshold
            )
        status_suffix = "" if carrot_debug is None else f" {carrot_debug}"
        status_line = (
            f"key={key} pinch_amount={pinch_amount:.1f} gripper_cmd={gripper_cmd:.1f} grasped={grasped} "
            f"tcp={obs['extra']['tcp_pose'][0].detach().cpu().numpy().tolist()[:3]}"
            f"{status_suffix}"
        )
        print(f"\n{status_line}")
        if args.teleop_contact_debug and carrot_debug is not None:
            _append_teleop_contact_log(args.teleop_contact_debug_log_file, status_line)
        _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)

    return save_idx


def _base_jog_mode(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    save_idx: int,
):
    step_size = 0.05
    yaw_step = 0.12

    print("Base jog: W/S=+/-Y, A/D=+/-X, Q/E=+/-yaw, X exit")

    while True:
        key = _read_single_key().lower()
        if key == "x":
            print("\nLeaving base jog mode")
            return save_idx

        moved = False
        if key == "w":
            robot_base_pose_p[1] += step_size
            moved = True
        elif key == "s":
            robot_base_pose_p[1] -= step_size
            moved = True
        elif key == "a":
            robot_base_pose_p[0] -= step_size
            moved = True
        elif key == "d":
            robot_base_pose_p[0] += step_size
            moved = True
        elif key == "q":
            roll, pitch, yaw = quat2euler(robot_base_pose_q, axes="sxyz")
            yaw += yaw_step
            robot_base_pose_q[:] = euler2quat(roll, pitch, yaw, axes="sxyz")
            moved = True
        elif key == "e":
            roll, pitch, yaw = quat2euler(robot_base_pose_q, axes="sxyz")
            yaw -= yaw_step
            robot_base_pose_q[:] = euler2quat(roll, pitch, yaw, axes="sxyz")
            moved = True

        if not moved:
            continue

        obs, info = env.reset(
            seed=args.seed,
            options={
                "reconfigure": False,
                **_scene_options(args),
                "robot_base_pose_p": robot_base_pose_p.tolist(),
                "robot_base_pose_q": robot_base_pose_q.tolist(),
                "robot_init_qpos": robot_init_qpos.tolist(),
            },
        )
        _maybe_render_live(env, args)
        save_idx += 1
        _, _, yaw = quat2euler(robot_base_pose_q, axes="sxyz")
        grasped = _probe_carrot_grasped(env)
        print(
            f"\nkey={key} robot_base_pose_p={robot_base_pose_p.tolist()} "
            f"yaw_deg={np.rad2deg(yaw):.2f} "
            f"grasped={grasped} "
            f"tcp={obs['extra']['tcp_pose'][0].detach().cpu().numpy().tolist()[:3]}"
        )
        _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)


def _joint_jog_mode(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    save_idx: int,
):
    step_size = 0.08
    selected_joint = 0

    print("Joint jog: press 1-6 to select arm joint")
    print("Joint jog: J/L=-/+, U/O=-/+ big step, X exit")
    print(f"Joint jog: selected joint={selected_joint + 1} step={step_size:.3f} rad")

    while True:
        key = _read_single_key().lower()
        if key == "x":
            print("\nLeaving joint jog mode")
            return save_idx

        if key in {"1", "2", "3", "4", "5", "6"}:
            selected_joint = int(key) - 1
            print(
                f"\nselected joint={selected_joint + 1} "
                f"value={robot_init_qpos[selected_joint]:.4f} rad"
            )
            continue

        delta = 0.0
        if key == "j":
            delta = -step_size
        elif key == "l":
            delta = step_size
        elif key == "u":
            delta = -(2.5 * step_size)
        elif key == "o":
            delta = 2.5 * step_size
        else:
            continue

        robot_init_qpos[selected_joint] += delta
        obs, info = _reset_current_state(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_init_qpos,
        )
        save_idx += 1
        grasped = _probe_carrot_grasped(env)
        print(
            f"\nkey={key} joint={selected_joint + 1} "
            f"value={robot_init_qpos[selected_joint]:.4f} rad "
            f"grasped={grasped} "
            f"tcp={obs['extra']['tcp_pose'][0].detach().cpu().numpy().tolist()[:3]}"
        )
        _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)


def _resolve_hand_control_indices(env: BaseEnv):
    robot = env.unwrapped.agent.robot
    resolved = []
    for name, joint_names, lower, upper in HAND_CONTROL_SPECS:
        indices = []
        for joint_name in joint_names:
            joint = robot.active_joints_map[joint_name]
            indices.append(int(joint.active_index[0].item()))
        resolved.append((name, indices, lower, upper))
    return resolved


def _current_gripper_pinch_amount(
    env: BaseEnv,
    preset_open_qpos: np.ndarray | None = None,
    preset_closed_qpos: np.ndarray | None = None,
) -> float:
    agent = env.unwrapped.agent
    robot = agent.robot
    current_qpos = robot.qpos[0].detach().cpu().numpy().astype(np.float32)
    active_indices = _gripper_active_indices(env)
    gripper_qpos = current_qpos[active_indices]
    open_qpos = (
        np.asarray(preset_open_qpos, dtype=np.float32)
        if preset_open_qpos is not None
        else np.asarray(agent.gripper_open_qpos, dtype=np.float32)
    )
    closed_qpos = (
        np.asarray(preset_closed_qpos, dtype=np.float32)
        if preset_closed_qpos is not None
        else np.asarray(agent.gripper_closed_qpos, dtype=np.float32)
    )
    motion = closed_qpos - open_qpos
    moving_mask = np.abs(motion) > 1e-5
    if not np.any(moving_mask):
        return 0.0
    progress = (gripper_qpos[moving_mask] - open_qpos[moving_mask]) / motion[moving_mask]
    return float(np.clip(np.median(progress), 0.0, 1.0))


def _hand_control_value_text(
    robot_init_qpos: np.ndarray,
    resolved_specs,
    spec_idx: int,
) -> str:
    name, indices, lower, upper = resolved_specs[spec_idx]
    values = [float(robot_init_qpos[idx]) for idx in indices]
    values_text = ", ".join(f"{v:.4f}" for v in values)
    select_key = HAND_CONTROL_SELECT_KEYS[spec_idx]
    return (
        f"{select_key}:{name} values=[{values_text}] rad "
        f"limits=[{lower:.4f}, {upper:.4f}]"
    )


def _apply_hand_control_delta(
    robot_init_qpos: np.ndarray,
    resolved_specs,
    spec_idx: int,
    delta: float,
) -> None:
    _, indices, lower, upper = resolved_specs[spec_idx]
    for idx in indices:
        robot_init_qpos[idx] = np.clip(robot_init_qpos[idx] + delta, lower, upper)


def _hand_jog_mode(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    save_idx: int,
):
    step_size = 0.05
    selected_control = 0
    current_qpos = env.unwrapped.agent.robot.qpos[0].detach().cpu().numpy().astype(np.float32)
    robot_init_qpos[:] = current_qpos
    resolved_specs = _resolve_hand_control_indices(env)
    selection_map = {
        key: idx for idx, key in enumerate(HAND_CONTROL_SELECT_KEYS[: len(resolved_specs)])
    }

    print("Hand jog: press a selector key to choose a hand control slider")
    print("Hand jog: J/L=-/+, U/O=-/+ big step, P print all, X exit")
    print("Hand jog: starting from the robot's current live qpos")
    for spec_idx in range(len(resolved_specs)):
        print(_hand_control_value_text(robot_init_qpos, resolved_specs, spec_idx))
    print(f"Hand jog: selected {resolved_specs[selected_control][0]} step={step_size:.3f} rad")

    while True:
        key = _read_single_key().lower()
        if key == "x":
            print("\nLeaving hand jog mode")
            return save_idx

        if key in selection_map:
            selected_control = selection_map[key]
            print(f"\nselected {_hand_control_value_text(robot_init_qpos, resolved_specs, selected_control)}")
            continue

        if key == "p":
            print("")
            for spec_idx in range(len(resolved_specs)):
                print(_hand_control_value_text(robot_init_qpos, resolved_specs, spec_idx))
            continue

        delta = 0.0
        if key == "j":
            delta = -step_size
        elif key == "l":
            delta = step_size
        elif key == "u":
            delta = -(2.5 * step_size)
        elif key == "o":
            delta = 2.5 * step_size
        else:
            continue

        _apply_hand_control_delta(robot_init_qpos, resolved_specs, selected_control, delta)
        obs, info = _reset_current_state(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_init_qpos,
        )
        save_idx += 1
        grasped = _probe_carrot_grasped(env)
        print(
            f"\nkey={key} {_hand_control_value_text(robot_init_qpos, resolved_specs, selected_control)} "
            f"grasped={grasped} "
            f"tcp={obs['extra']['tcp_pose'][0].detach().cpu().numpy().tolist()[:3]}"
        )
        _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)


def main(args: Args):
    _sanitize_live_viewer_environment(args)
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
    _apply_grid_pair_override(args, robot_base_pose_p)
    print(f"Scene asset dir: {args.scene_asset_dir}")
    print(f"Loaded state file: {args.load_state_file or '<defaults>'}")
    print(f"Requested sim_backend={args.sim_backend}")
    print(f"Live viewer={args.live_viewer} live_viewer_force_cpu={args.live_viewer_force_cpu}")
    print(f"Loaded arm_qpos={robot_init_qpos[:6].tolist()}")
    print(f"Loaded hand_qpos={robot_init_qpos[6:].tolist()}")

    camera_mode = args.observation_camera_mode
    cam_state = observation_camera_lookat_state(camera_mode, scene_asset_dir=args.scene_asset_dir)
    camera_eye = np.array(cam_state["eye"], dtype=np.float32)
    camera_target = np.array(cam_state["target"], dtype=np.float32)
    camera_roll_deg = float(cam_state.get("roll_deg", 0.0))
    camera_fov = float(cam_state.get("fov", 1.0))

    env = _build_env(
        args,
        camera_mode=camera_mode,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_roll_deg=camera_roll_deg,
        camera_fov=camera_fov,
    )
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
        _print_help()
        prompt_shown = False

        while True:
            if args.live_viewer:
                raw = _read_command_line_nonblocking(
                    "rc5> ", timeout_sec=0.03, show_prompt=not prompt_shown
                )
                prompt_shown = True
                if raw is None:
                    _maybe_render_live(env, args)
                    continue
                print("")
                prompt_shown = False
            else:
                raw = input("rc5> ").strip()
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()
            rebuild_camera = False
            apply_camera_in_place = False
            try:
                if cmd in {"quit", "exit", "q"}:
                    break
                if cmd == "help":
                    _print_help()
                    continue
                if cmd == "load":
                    args.load_state_file = parts[1]
                    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
                    print(f"Loaded state file: {args.load_state_file}")
                    print(f"Loaded arm_qpos={robot_init_qpos[:6].tolist()}")
                    print(f"Loaded hand_qpos={robot_init_qpos[6:].tolist()}")
                    rebuild_camera = True
                if cmd == "show":
                    print(f"robot_base_pose_p={robot_base_pose_p.tolist()}")
                    print(f"robot_base_pose_q={robot_base_pose_q.tolist()}")
                    print(f"robot_init_qpos={robot_init_qpos.tolist()}")
                    print(f"camera_mode={camera_mode}")
                    print(f"camera_eye={camera_eye.tolist()}")
                    print(f"camera_target={camera_target.tolist()}")
                    print(f"camera_roll_deg={camera_roll_deg}")
                    print(f"camera_fov={camera_fov}")
                    continue
                if cmd == "cams":
                    print("manual_best")
                    print("position_ground_lookat")
                    print("position_center_lookat")
                    print("preset_ground")
                    print("preset_center")
                    print("preset_center_up")
                    print("preset_center_up_high")
                    print("preset_center_back")
                    print("preset_center_back_up")
                    print("preset_center_far_back_up")
                    continue
                if cmd == "reset":
                    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
                    camera_mode = args.observation_camera_mode
                    cam_state = observation_camera_lookat_state(
                        camera_mode, scene_asset_dir=args.scene_asset_dir
                    )
                    camera_eye = np.array(cam_state["eye"], dtype=np.float32)
                    camera_target = np.array(cam_state["target"], dtype=np.float32)
                    camera_roll_deg = float(cam_state.get("roll_deg", 0.0))
                    camera_fov = float(cam_state.get("fov", 1.0))
                    rebuild_camera = True
                elif cmd == "move":
                    robot_base_pose_p += np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float32,
                    )
                elif cmd == "x":
                    robot_base_pose_p[0] += float(parts[1])
                elif cmd == "y":
                    robot_base_pose_p[1] += float(parts[1])
                elif cmd == "z":
                    robot_base_pose_p[2] += float(parts[1])
                elif cmd == "joint":
                    idx = int(parts[1])
                    robot_init_qpos[idx] += float(parts[2])
                elif cmd == "qpos":
                    if len(parts) != len(robot_init_qpos) + 1:
                        raise ValueError(f"expected {len(robot_init_qpos)} qpos values")
                    robot_init_qpos = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                elif cmd == "cam":
                    camera_mode = parts[1]
                    cam_state = observation_camera_lookat_state(
                        camera_mode, scene_asset_dir=args.scene_asset_dir
                    )
                    camera_eye = np.array(cam_state["eye"], dtype=np.float32)
                    camera_target = np.array(cam_state["target"], dtype=np.float32)
                    camera_roll_deg = float(cam_state.get("roll_deg", 0.0))
                    camera_fov = float(cam_state.get("fov", 1.0))
                    apply_camera_in_place = True
                elif cmd == "cameye":
                    camera_eye += np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    camera_mode = "custom"
                    apply_camera_in_place = True
                elif cmd == "camtarget":
                    camera_target += np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                    camera_mode = "custom"
                    apply_camera_in_place = True
                elif cmd == "camroll":
                    camera_roll_deg += float(parts[1])
                    camera_mode = "custom"
                    apply_camera_in_place = True
                elif cmd == "camfov":
                    camera_fov += float(parts[1])
                    camera_mode = "custom"
                    apply_camera_in_place = True
                elif cmd == "teleop":
                    save_idx = _teleop_mode(env, args, robot_base_pose_p, robot_base_pose_q, robot_init_qpos, save_idx)
                    continue
                elif cmd == "jogbase":
                    save_idx = _base_jog_mode(env, args, robot_base_pose_p, robot_base_pose_q, robot_init_qpos, save_idx)
                    continue
                elif cmd == "jogjoints":
                    save_idx = _joint_jog_mode(
                        env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        robot_init_qpos,
                        save_idx,
                    )
                    continue
                elif cmd == "joghand":
                    save_idx = _hand_jog_mode(
                        env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        robot_init_qpos,
                        save_idx,
                    )
                    continue
                elif cmd == "save":
                    save_idx += 1
                    obs, info = _reset_current_state(env, args, robot_base_pose_p, robot_base_pose_q, robot_init_qpos)
                    _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
                    continue
                else:
                    print("Unknown command. Type 'help'.")
                    continue
            except (IndexError, ValueError) as exc:
                print(f"Bad command: {exc}")
                continue

            print(f"robot_base_pose_p={robot_base_pose_p.tolist()}")
            print(f"robot_init_qpos={robot_init_qpos.tolist()}")
            print(f"camera_mode={camera_mode}")
            print(f"camera_eye={camera_eye.tolist()}")
            print(f"camera_target={camera_target.tolist()}")
            save_idx += 1
            if rebuild_camera and not args.live_viewer:
                env = _rebuild_env_with_camera(
                    env,
                    args,
                    camera_mode=camera_mode,
                    camera_eye=camera_eye,
                    camera_target=camera_target,
                    camera_roll_deg=camera_roll_deg,
                    camera_fov=camera_fov,
                )
            if apply_camera_in_place:
                _apply_camera_state(env, camera_eye, camera_target, camera_roll_deg, camera_fov)
                obs, info = _current_obs_info(env, args)
            else:
                obs, info = _reset_current_state(
                    env,
                    args,
                    robot_base_pose_p,
                    robot_base_pose_q,
                    robot_init_qpos,
                    reconfigure=rebuild_camera,
                )
            _save_snapshot_from_obs(env, obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
