"""Collect scripted proxy-curve pickup demos for the AIRI cubes scene.

The third-party bundle does not include a standalone motion-planner generator,
but its YAML/NPZ artifacts describe a simple ``proxy_ee_delta`` pickup policy:
small end-effector deltas toward a calibrated pregrasp, descend, close, lift.
This script recreates that policy while saving through ManiSkill RecordEpisode
with the same environment settings used for AIRI cube demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time


def _disable_gui_runtime_side_effects() -> None:
    """Keep headless scripted collection from waking GUI runtime helpers."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["XR_RUNTIME_JSON"] = "/nonexistent/openxr_runtime.json"
    openvr_dir = Path(tempfile.gettempdir()) / "real2sim_no_openvr"
    openvr_dir.mkdir(parents=True, exist_ok=True)
    openvr_pathreg = openvr_dir / "openvrpaths.vrpath"
    if not openvr_pathreg.exists():
        openvr_pathreg.write_text(
            '{"config":[],"external_drivers":[],"jsonid":"vrpathreg","log":[],"runtime":[]}\n',
            encoding="utf-8",
        )
    os.environ["VR_PATHREG_OVERRIDE"] = str(openvr_pathreg)
    os.environ["VR_OVERRIDE"] = "/nonexistent/openvr_runtime"
    os.environ["VR_CONFIG_PATH"] = str(openvr_dir / "config")
    os.environ["VR_LOG_PATH"] = str(openvr_dir / "log")
    os.environ["DISABLE_VK_LAYER_VALVE_steam_overlay_1"] = "1"
    os.environ["DISABLE_VK_LAYER_VALVE_steam_fossilize_1"] = "1"
    disabled_layers = [
        "VK_LAYER_VALVE_steam_overlay_32",
        "VK_LAYER_VALVE_steam_overlay_64",
        "VK_LAYER_VALVE_steam_fossilize_32",
        "VK_LAYER_VALVE_steam_fossilize_64",
    ]
    existing_disabled = [
        item.strip()
        for item in os.environ.get("VK_LOADER_LAYERS_DISABLE", "").split(",")
        if item.strip()
    ]
    os.environ["VK_LOADER_LAYERS_DISABLE"] = ",".join(dict.fromkeys(existing_disabled + disabled_layers))
    gui_runtime_marker = "Steam" + "V" + "R"
    for env_name in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env_value = os.environ.get(env_name, "")
        if gui_runtime_marker in env_value:
            os.environ.pop(env_name, None)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if gui_runtime_marker in ld_library_path:
        kept_entries = [entry for entry in ld_library_path.split(":") if gui_runtime_marker not in entry]
        os.environ["LD_LIBRARY_PATH"] = ":".join(kept_entries)


_disable_gui_runtime_side_effects()

import gymnasium as gym
import numpy as np
import sapien
import torch
import tyro

from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers.record import RecordEpisode
from real2sim.calibrate_rc5_pose import (
    HAND_OPEN_STATE_FILE,
    _extract_obs_frame,
    _gripper_active_indices,
    _load_state_file,
    _robot_uid,
    _scene_options,
    _state_file_gripper_qpos,
    observation_camera_lookat_state,
)
from real2sim.debug_paths import HUMAN_DEMOS_DIR
from real2sim.openreal2sim_validation import (
    AIRI_CUBES_ASSET_DIR,
    AIRI_CUBES_ROBOT_BASE_POSE_Q,
    AIRI_CUBE_COLOR_NAMES,
    AIRI_CUBE_HALF_SIZE,
    AIRI_CUBE_PICKUP_SUCCESS_HEIGHT,
    DEFAULT_PLATE_MODEL_NAME,
    DEFAULT_REAL2SIM_CONTROL_FREQ,
    DEFAULT_REAL2SIM_SIM_FREQ,
    DEFAULT_SOURCE_MODEL_NAME,
    pose_from_eye_target_roll,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOME_FILE = REPO_ROOT / "real2sim" / "real2sim" / "presets" / "rc5_pose" / "airi_cubes_home.txt"
DEFAULT_THIRD_PARTY_CONFIG = (
    REPO_ROOT / "third_party_files" / "pack2" / "assets" / "selected_key_simulation.yaml"
)
MOTIONPLANNING_DEBUG_DIR = REPO_ROOT / "real2sim" / "debug" / "motionplanning_demos"

COLOR_TO_THIRD_PARTY_OBJECT = {
    "red": "orange_cube_ext",
    "blue": "blue_cube_ext",
    "green": "green_cube_ext",
    "yellow": "yellow_cube_ext",
    "white": "white_cube_ext",
}

FALLBACK_OBJECT_CALIBRATION = {
    "pregrasp_offset_xyz": [-0.039, -0.005, 0.0905],
    "descend_offset_xyz": [-0.039, -0.005, 0.0805],
    "target_quat": [-0.5161, -0.6785, 0.4517, 0.263],
}


@dataclass
class Args:
    output_dir: str = str(HUMAN_DEMOS_DIR)
    debug_dir: str = str(MOTIONPLANNING_DEBUG_DIR)
    session_name: str = ""
    resume_session_name: str = ""
    trajectory_name: str = "airi_cubes_curve_demos"

    scene_asset_dir: str = str(AIRI_CUBES_ASSET_DIR)
    load_state_file: str = str(DEFAULT_HOME_FILE)
    task_mode: str = "airi_cube_pickup"
    use_probe_objects: bool = False
    clean_scene: bool = True
    load_background: bool = True
    show_debug_markers: bool = False
    robot_far_away: bool = False
    use_wrist_camera: bool = os.environ.get("REAL2SIM_AIRI_CUBES_V3_USE_WRIST_IMAGE", "1") != "0"
    use_environment_map: bool = True
    enable_shadow: bool = True
    use_shadow_catcher: bool = False

    observation_camera_mode: str = "manual_best"
    shader: str = "default"
    human_render_shader: str = "default"
    sim_backend: str = "gpu"
    render_backend_override: str = "gpu"
    sim_freq_override: int = DEFAULT_REAL2SIM_SIM_FREQ
    control_freq_override: int = DEFAULT_REAL2SIM_CONTROL_FREQ
    seed: int = 0

    save_video: bool = True
    save_failed_videos: bool = True
    record_env_state: bool = True
    record_minimal_openvla_obs: bool = True
    action_debug_log: bool = True
    action_debug_log_file: str = "airi_curve_action_debug.jsonl"
    video_fps: int = 10
    demo_video_interval: int = 1
    max_demos: int = 25
    max_final_steps: int = 80
    success_stable_steps: int = 3
    airi_cube_target_colors: str = ",".join(AIRI_CUBE_COLOR_NAMES)
    randomize_airi_cube_target: bool = False
    randomize_airi_cube_positions: bool = True
    airi_cube_position_jitter_x: float = 0.012
    airi_cube_position_jitter_y: float = 0.008
    airi_cube_target_color: str = "red"

    probe_source_model_name: str = DEFAULT_SOURCE_MODEL_NAME
    probe_plate_model_name: str = DEFAULT_PLATE_MODEL_NAME
    spawn_grid_state_file: str = ""
    grid_pair_index: int = -1
    base_dx: float = 0.0
    base_dy: float = 0.0
    base_dz: float = 0.0
    disable_self_collisions: bool = False

    filter_small_actions_before_save: bool = False
    demo_step_translation_thresh_m: float = 0.005
    min_generated_translation_m: float = 0.005
    demo_step_rotation_thresh_rad: float = 0.03
    filter_control_same_pos_thresh: float = 0.0015
    filter_control_same_rot_thresh: float = 0.01
    filter_control_same_gripper_thresh: float = 0.02
    filter_image_change_keep_thresh: float = 0.015

    planner_config_file: str = str(DEFAULT_THIRD_PARTY_CONFIG)
    use_planner_object_calibrations: bool = True
    transform_planner_offsets_to_current_base: bool = True
    planner_offset_sweep_xyz_m: str = ""
    planner_offset_sweep_successes_per_candidate: int = 1
    planner_offset_sweep_discard_on_failure: bool = False
    planner_offset_sweep_allow_incomplete: bool = False
    planner_offset_sweep_stop_after_pass: bool = False
    max_demo_actions: int = 80
    max_saved_demo_actions: int = 80
    xy_step_m: float = 0.031
    z_step_m: float = 0.031
    action_translation_norm_m: float = 0.046
    action_step_jitter_frac: float = 0.04
    action_direction_noise_m: float = 0.0004
    pos_tol_m: float = 0.003
    approach_xy_hover_cube_heights: float = 1.5
    approach_curve_enabled: bool = True
    approach_curve_min_offset_m: float = 0.100
    approach_curve_max_offset_m: float = 0.220
    approach_curve_height_offset_m: float = 0.055
    approach_curve_waypoints: int = 48
    approach_curve_midpoint_min: float = 0.35
    approach_curve_midpoint_max: float = 0.70
    approach_curve_base_x_min_m: float = -0.180
    approach_curve_base_x_max_m: float = 0.160
    approach_curve_base_y_min_m: float = -0.850
    approach_curve_base_y_max_m: float = -0.220
    approach_curve_max_reverse_m: float = 0.012
    approach_top_clearance_m: float = 0.040
    approach_correction_enabled: bool = True
    approach_miss_probability: float = 0.50
    approach_correction_max_offset_m: float = 0.055
    joint_limit_slow_margin_rad: float = 0.040
    joint_limit_stop_margin_rad: float = 0.005
    tcp_bad_response_ratio: float = 0.25
    lift_curve_max_offset_m: float = 0.045
    final_hold_steps: int = 4
    recovery_enabled: bool = True
    recovery_max_attempts: int = 2
    min_descend_offset_z_m: float = 0.070
    max_stage_steps: int = 80
    stall_steps: int = 8
    hold_steps: int = 0
    grasp_hold_steps: int = 2
    lift_delta_z: float = max(0.16, float(AIRI_CUBE_PICKUP_SUCCESS_HEIGHT) + 0.04)
    use_third_party_lift_delta: bool = False
    random_seed: int = 0
    max_attempts: int = 0
    resume_from_existing: bool = False
    resume_saved_count: int = -1
    resume_attempt_count: int = -1
    parallel_workers: int = 1
    parallel_worker_index: int = -1
    parallel_worker_backend: str = "cpu"
    verbose: bool = False
    dry_run: bool = False


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _remaining_time_text(start_time: float, saved: int, max_demos: int) -> str:
    if saved <= 0:
        return "unknown"
    elapsed = time.monotonic() - float(start_time)
    remaining = max(0, int(max_demos) - int(saved))
    return _format_duration((elapsed / float(saved)) * float(remaining))


def _with_combined_camera_obs(obs: dict, *, minimal: bool = False, use_wrist_image: bool = True) -> dict:
    if not isinstance(obs, dict) or "sensor_data" not in obs:
        return obs
    try:
        sensor_data = obs.get("sensor_data", {})
        if use_wrist_image and "wrist_camera" in sensor_data:
            frame = _extract_obs_frame(obs, wrist_inset_bottom_right=True)
        else:
            frame = _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])
    except Exception:
        return obs
    sensor_data = dict(obs.get("sensor_data", {}))
    ref_rgb = sensor_data.get("3rd_view_camera", {}).get("rgb")
    if torch.is_tensor(ref_rgb):
        combined_rgb = torch.as_tensor(frame, dtype=torch.uint8, device=ref_rgb.device).unsqueeze(0)
    else:
        combined_rgb = frame[np.newaxis, ...]
    if minimal:
        out = {key: value for key, value in obs.items() if key != "sensor_data"}
        out["openvla_image"] = combined_rgb
        return out
    sensor_data["combined_camera"] = {"rgb": combined_rgb}
    out = dict(obs)
    out["sensor_data"] = sensor_data
    out["openvla_image"] = combined_rgb
    return out


class _CombinedCameraObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, *, minimal: bool = False, use_wrist_image: bool = True):
        super().__init__(env)
        self.minimal = bool(minimal)
        self.use_wrist_image = bool(use_wrist_image)

    def observation(self, obs):
        return _with_combined_camera_obs(
            obs,
            minimal=self.minimal,
            use_wrist_image=self.use_wrist_image,
        )


def _install_combined_video_capture(record_env) -> None:
    def capture_image(infos=None):
        return _extract_obs_frame(
            record_env.env.unwrapped.get_obs(),
            wrist_inset_bottom_right=True,
        )

    record_env.capture_image = capture_image


def _should_save_demo_video(saved_index: int, args: "Args") -> bool:
    interval = max(1, int(args.demo_video_interval))
    return int(saved_index) % interval == 0


def _count_saved_episodes(session_dir: Path, trajectory_name: str) -> int:
    total = 0
    for json_path in sorted(session_dir.glob(f"{trajectory_name}*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        episodes = payload.get("episodes", [])
        if isinstance(episodes, list):
            total += len(episodes)
    return total


def _next_trajectory_part_name(session_dir: Path, trajectory_name: str) -> str:
    for part_idx in range(1, 100000):
        candidate = f"{trajectory_name}_part{part_idx:04d}"
        if not (session_dir / f"{candidate}.h5").exists() and not (session_dir / f"{candidate}.json").exists():
            return candidate
    raise RuntimeError(f"Could not find free trajectory part name under {session_dir}")


def _infer_resume_counts(args: "Args", session_dir: Path) -> tuple[int, int]:
    saved = int(args.resume_saved_count)
    if saved < 0:
        saved = _count_saved_episodes(session_dir, args.trajectory_name)
    attempted = int(args.resume_attempt_count)
    if attempted < 0:
        attempted = saved
    return max(0, saved), max(0, attempted)


def _argv_with_option(argv: list[str], name: str, value: str | int | float | bool) -> list[str]:
    out: list[str] = []
    option = f"--{name}"
    skip_next = False
    i = 0
    while i < len(argv):
        item = argv[i]
        if skip_next:
            skip_next = False
            i += 1
            continue
        if item == option:
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                skip_next = True
            i += 1
            continue
        if item.startswith(f"{option}="):
            i += 1
            continue
        out.append(item)
        i += 1
    if isinstance(value, bool):
        out.append(option if value else f"--no-{name}")
    else:
        out.extend([option, str(value)])
    return out


def _run_parallel_workers(args: "Args") -> None:
    workers = max(1, int(args.parallel_workers))
    if workers <= 1 or int(args.parallel_worker_index) >= 0:
        return
    total = max(0, int(args.max_demos))
    if total <= 0:
        raise SystemExit("--parallel-workers requires --max-demos > 0")
    base_session = args.resume_session_name.strip() or args.session_name or time.strftime("%Y%m%d_%H%M%S")
    base_seed = int(args.random_seed if args.random_seed else args.seed)
    counts = [total // workers] * workers
    for i in range(total % workers):
        counts[i] += 1

    procs: list[tuple[int, subprocess.Popen]] = []
    print(f"Launching {workers} workers for {total} demos under session prefix {base_session}")
    for worker_idx, count in enumerate(counts):
        if count <= 0:
            continue
        worker_session = f"{base_session}_w{worker_idx:02d}"
        argv = list(sys.argv[1:])
        overrides = {
            "parallel-workers": 1,
            "parallel-worker-index": worker_idx,
            "max-demos": count,
            "max-attempts": count,
            "session-name": worker_session,
            "resume-session-name": "",
            "random-seed": base_seed + worker_idx + 1,
        }
        if str(args.parallel_worker_backend).strip().lower() == "cpu":
            overrides["sim-backend"] = "cpu"
            overrides["render-backend-override"] = "cpu"
        elif str(args.parallel_worker_backend).strip().lower() not in {"inherit", ""}:
            raise SystemExit(
                "--parallel-worker-backend must be 'cpu' or 'inherit' "
                f"(got {args.parallel_worker_backend!r})"
            )
        for key, value in overrides.items():
            argv = _argv_with_option(argv, key, value)
        cmd = [sys.executable, "-m", "real2sim.record_airi_cubes_curve_demos", *argv]
        print(f"  worker {worker_idx:02d}: demos={count} session={worker_session}")
        procs.append((worker_idx, subprocess.Popen(cmd)))

    failed: list[tuple[int, int]] = []
    for worker_idx, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((worker_idx, code))
    if failed:
        raise SystemExit(f"Parallel collection failed: {failed}")
    print(f"Parallel collection done: requested {total} demos across {len(procs)} workers.")
    raise SystemExit(0)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, sapien.Pose):
        return {"p": np.asarray(value.p).tolist(), "q": np.asarray(value.q).tolist()}
    return value


def _dataclass_replace(obj, **kwargs):
    values = {field.name: getattr(obj, field.name) for field in obj.__dataclass_fields__.values()}
    values.update(kwargs)
    return obj.__class__(**values)


def _info_bool(info: dict, key: str) -> bool:
    value = info.get(key, False)
    if torch.is_tensor(value):
        return bool(value[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def _update_last_session_debug_symlink(debug_dir: Path) -> None:
    link_path = debug_dir.parent / "last_session_debug"
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(debug_dir.resolve(), target_is_directory=True)
    except OSError as exc:
        print(f"Warning: failed to update {link_path}: {exc}")


def _reset_record_env(
    env,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    *,
    reconfigure: bool = False,
    episode_id: int | None = None,
):
    record_args = _dataclass_replace(
        args,
        spawn_grid_state_file="",
        grid_pair_index=-1,
        show_debug_markers=False,
    )
    return env.reset(
        seed=args.seed,
        options={
            "reconfigure": reconfigure,
            **_scene_options(record_args),
            "use_shadow_catcher": bool(args.use_shadow_catcher),
            "randomize_airi_cube_positions": bool(args.randomize_airi_cube_positions),
            "airi_cube_position_jitter_x": float(args.airi_cube_position_jitter_x),
            "airi_cube_position_jitter_y": float(args.airi_cube_position_jitter_y),
            **({} if episode_id is None else {"episode_id": int(episode_id)}),
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
            "trajectory_instruction": f"Pick up the {args.airi_cube_target_color} cube.",
            "real2sim_action_filter_applied": bool(args.filter_small_actions_before_save),
            "real2sim_action_filter": {
                "pos_thresh": float(args.demo_step_translation_thresh_m),
                "rot_thresh": float(args.demo_step_rotation_thresh_rad),
                "control_same_pos_thresh": float(args.filter_control_same_pos_thresh),
                "control_same_rot_thresh": float(args.filter_control_same_rot_thresh),
                "control_same_gripper_thresh": float(args.filter_control_same_gripper_thresh),
                "image_change_keep_thresh": float(args.filter_image_change_keep_thresh),
            },
        },
    )


def _with_open_gripper_qpos(env, robot_init_qpos: np.ndarray) -> np.ndarray:
    full_qpos = np.asarray(robot_init_qpos, dtype=np.float32).copy()
    gripper_indices = _gripper_active_indices(env)
    open_gripper_qpos = _state_file_gripper_qpos(HAND_OPEN_STATE_FILE, env)
    full_qpos[gripper_indices] = open_gripper_qpos
    return full_qpos


def _build_record_env(args: Args, session_dir: Path, trajectory_name: str):
    render_backend = args.render_backend_override or "cpu"
    sim_backend = args.sim_backend
    if sim_backend in {"gpu", "cuda", "physx_cuda"} and render_backend in {"cpu", "sapien_cpu"}:
        print(
            "GPU simulation requires a CUDA render backend for visual observations; "
            "using render_backend=gpu. Pass --sim-backend cpu --render-backend-override cpu "
            "to run fully on CPU.",
            flush=True,
        )
        render_backend = "gpu"
    record_args = _dataclass_replace(
        args,
        sim_backend=sim_backend,
        render_backend_override=render_backend,
        spawn_grid_state_file="",
        grid_pair_index=-1,
        show_debug_markers=False,
    )
    cam_state = observation_camera_lookat_state(
        record_args.observation_camera_mode,
        scene_asset_dir=record_args.scene_asset_dir,
    )
    camera_pose = pose_from_eye_target_roll(
        eye=np.array(cam_state["eye"], dtype=np.float32),
        target=np.array(cam_state["target"], dtype=np.float32),
        roll_deg=float(cam_state.get("roll_deg", 0.0)),
    )
    print(
        "Building ManiSkill env: "
        f"sim_backend={sim_backend} render_backend={render_backend} render_mode=rgb_array",
        flush=True,
    )
    env = gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        enable_shadow=bool(args.enable_shadow),
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": {
                "pose": camera_pose,
                "intrinsic": None,
                "fov": float(cam_state.get("fov", 1.0)),
                "near": 0.01,
                "far": 10.0,
            },
        },
        human_render_camera_configs={"shader_pack": args.human_render_shader or args.shader},
        sim_config={
            "sim_freq": int(args.sim_freq_override),
            "control_freq": int(args.control_freq_override),
        },
        num_envs=1,
        sim_backend=sim_backend,
        render_backend=render_backend,
        robot_uids=_robot_uid(args),
        scene_asset_dir=args.scene_asset_dir,
        use_wrist_camera=bool(args.use_wrist_camera),
    )
    if bool(args.use_wrist_camera) or bool(args.record_minimal_openvla_obs):
        env = _CombinedCameraObservationWrapper(
            env,
            minimal=bool(args.record_minimal_openvla_obs),
            use_wrist_image=bool(args.use_wrist_camera),
        )
    print("ManiSkill env created; wrapping RecordEpisode.", flush=True)
    record_env = RecordEpisode(
        env,
        output_dir=str(session_dir),
        save_trajectory=True,
        trajectory_name=trajectory_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="AIRI cubes scripted proxy end-effector delta pickup demonstrations",
        video_fps=args.video_fps,
        save_on_reset=False,
        record_env_state=bool(args.record_env_state),
        recording_camera_name="3rd_view_camera",
        avoid_overwriting_video=True,
        max_steps_per_video=max(args.max_final_steps + 5, 100),
        clean_on_close=False,
    )
    if bool(args.use_wrist_camera):
        _install_combined_video_capture(record_env)
    print("RecordEpisode wrapper created; resetting scene.", flush=True)
    record_env._json_data = _to_jsonable(record_env._json_data)
    record_env._json_data["replay_validation_sim_backend"] = record_args.sim_backend
    record_env._json_data["replay_validation_render_backend"] = record_args.render_backend_override
    dump_json(record_env._json_path, record_env._json_data, indent=2)

    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
    _reset_record_env(
        record_env,
        args,
        robot_base_pose_p,
        robot_base_pose_q,
        _with_open_gripper_qpos(record_env, robot_init_qpos),
        reconfigure=True,
        episode_id=0,
    )
    print("Initial scene reset completed.", flush=True)
    return record_env


def _parse_color_choices(spec: str) -> list[str]:
    raw = str(spec).strip().lower()
    if not raw or raw in {"all", "*"}:
        return list(AIRI_CUBE_COLOR_NAMES)
    colors = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(colors) - set(AIRI_CUBE_COLOR_NAMES))
    if unknown:
        raise SystemExit(f"Unknown AIRI cube colors {unknown}. Available: {list(AIRI_CUBE_COLOR_NAMES)}")
    return colors


def _load_proxy_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the third-party planner config.") from exc
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    sim = payload.get("simulation")
    if sim is None:
        key = payload.get("keys", [""])[0] if isinstance(payload.get("keys"), list) else ""
        sim = payload.get("local", {}).get(key, {}).get("simulation", {})
    return sim if isinstance(sim, dict) else {}


def _object_calibration(proxy_cfg: dict, color: str, *, use_planner_object_calibrations: bool) -> dict:
    calibrations = proxy_cfg.get("planner_object_calibrations", {})
    object_id = COLOR_TO_THIRD_PARTY_OBJECT[color]
    if (
        use_planner_object_calibrations
        and isinstance(calibrations, dict)
        and isinstance(calibrations.get(object_id), dict)
    ):
        merged = dict(FALLBACK_OBJECT_CALIBRATION)
        merged.update(calibrations[object_id])
        return merged
    return dict(FALLBACK_OBJECT_CALIBRATION)


def _tensor_row(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim >= 2:
        return arr[0]
    return arr


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _world_to_robot_base(env, world_xyz: np.ndarray) -> np.ndarray:
    robot_pose = env.unwrapped.agent.robot.pose
    base_p = _tensor_row(robot_pose.p).astype(np.float64)
    base_q = _tensor_row(robot_pose.q).astype(np.float64)
    base_rot = _quat_wxyz_to_mat(base_q)
    return base_rot.T @ (np.asarray(world_xyz, dtype=np.float64) - base_p)


def _reference_offset_to_current_base(offset_xyz: np.ndarray, current_base_q: np.ndarray) -> np.ndarray:
    """Keep planner offsets physically aligned when the robot base yaw changes."""
    reference_rot = _quat_wxyz_to_mat(AIRI_CUBES_ROBOT_BASE_POSE_Q)
    current_rot = _quat_wxyz_to_mat(np.asarray(current_base_q, dtype=np.float64))
    return current_rot.T @ reference_rot @ np.asarray(offset_xyz, dtype=np.float64)


def _parse_offset_sweep(spec: str) -> list[np.ndarray]:
    spec = str(spec or "").strip()
    if not spec:
        return []
    offsets: list[np.ndarray] = []
    for raw_item in spec.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) == 2:
            parts.append("0")
        if len(parts) != 3:
            raise SystemExit(
                "planner_offset_sweep_xyz_m entries must be 'dx,dy' or 'dx,dy,dz', "
                f"got {item!r}"
            )
        try:
            offsets.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64))
        except ValueError as exc:
            raise SystemExit(f"Invalid planner_offset_sweep_xyz_m entry: {item!r}") from exc
    return offsets


def _offset_sweep_candidate(args: Args, attempt_index: int) -> tuple[int, np.ndarray]:
    offsets = _parse_offset_sweep(args.planner_offset_sweep_xyz_m)
    if not offsets:
        return -1, np.zeros(3, dtype=np.float64)
    index = getattr(args, "_planner_offset_sweep_index", None)
    if index is None:
        index = max(0, int(attempt_index) - 1) % len(offsets)
    index = int(index)
    if index < 0 or index >= len(offsets):
        return -1, np.zeros(3, dtype=np.float64)
    return index, offsets[index].copy()


def _tcp_base_xyz(env) -> np.ndarray:
    pose = env.unwrapped.agent.ee_pose_at_robot_base.raw_pose
    return _tensor_row(pose)[:3].astype(np.float64)


def _cube_world_xyz(env, color: str) -> np.ndarray:
    actor = env.unwrapped.airi_cube_actors[color]
    return _tensor_row(actor.pose.p)[:3].astype(np.float64)


def _sample_xy_offset(rng: np.random.Generator, max_norm: float) -> np.ndarray:
    max_norm = max(0.0, float(max_norm))
    if max_norm <= 0.0:
        return np.zeros(2, dtype=np.float64)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(0.35, 1.0) * max_norm)
    return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float64)


def _bezier_point(control_points: list[np.ndarray], t: float) -> np.ndarray:
    points = [np.asarray(point, dtype=np.float64).copy() for point in control_points]
    t = float(np.clip(t, 0.0, 1.0))
    while len(points) > 1:
        points = [(1.0 - t) * points[i] + t * points[i + 1] for i in range(len(points) - 1)]
    return points[0]


def _path_workspace_violation(
    path: list[np.ndarray],
    args: Args,
    *,
    start: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> float:
    if not path:
        return 0.0
    points = np.asarray([np.asarray(point, dtype=np.float64)[:3] for point in path], dtype=np.float64)
    x_min = float(args.approach_curve_base_x_min_m)
    x_max = float(args.approach_curve_base_x_max_m)
    y_min = float(args.approach_curve_base_y_min_m)
    y_max = float(args.approach_curve_base_y_max_m)
    violation = np.maximum(0.0, x_min - points[:, 0]).sum()
    violation += np.maximum(0.0, points[:, 0] - x_max).sum()
    violation += np.maximum(0.0, y_min - points[:, 1]).sum()
    violation += np.maximum(0.0, points[:, 1] - y_max).sum()
    if start is not None and target is not None:
        start_arr = np.asarray(start, dtype=np.float64)[:3]
        target_arr = np.asarray(target, dtype=np.float64)[:3]
        chord_xy = target_arr[:2] - start_arr[:2]
        chord_norm = float(np.linalg.norm(chord_xy))
        if chord_norm > 1e-8:
            tangent = chord_xy / chord_norm
            progress = (points[:, :2] - start_arr[:2]) @ tangent
            max_reverse = max(0.0, float(args.approach_curve_max_reverse_m))
            violation += 8.0 * np.maximum(0.0, -progress - max_reverse).sum()
    return float(violation)


def _approach_bezier_waypoints(
    start_base: np.ndarray,
    target_base: np.ndarray,
    *,
    args: Args,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], dict]:
    start = np.asarray(start_base, dtype=np.float64)
    target = np.asarray(target_base, dtype=np.float64)
    chord = target[:2] - start[:2]
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= 1e-8:
        return [target.copy()], {
            "tangent_xy": np.array([1.0, 0.0], dtype=np.float64),
            "normal_xy": np.array([0.0, 1.0], dtype=np.float64),
            "bend": 0.0,
            "height": 0.0,
            "expressiveness": 0.0,
        }

    tangent = chord / chord_norm
    base_normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    signs = [float(rng.choice([-1.0, 1.0]))]
    signs.append(-signs[0])

    expressiveness = float(rng.beta(0.85, 0.85))
    min_offset = max(0.0, float(args.approach_curve_min_offset_m))
    max_offset = max(min_offset, float(args.approach_curve_max_offset_m))
    effective_min_offset = min(min_offset, chord_norm * 0.55)
    chord_scaled_offset = chord_norm * float(rng.uniform(0.48, 0.78))
    bend_mag = min(max_offset, max(effective_min_offset, chord_scaled_offset))
    bend_mag *= float(0.82 + 0.38 * expressiveness)
    bend_mag = min(max_offset, bend_mag)
    height = max(0.0, float(args.approach_curve_height_offset_m)) * float(0.45 + 0.75 * expressiveness)

    midpoint_min = float(np.clip(args.approach_curve_midpoint_min, 0.10, 0.80))
    midpoint_max = float(np.clip(args.approach_curve_midpoint_max, midpoint_min, 0.92))
    mid_t = float(rng.uniform(midpoint_min, midpoint_max))
    t2 = float(np.clip(mid_t + rng.uniform(0.18, 0.32), 0.52, 0.88))
    backtrack = min(chord_norm * float(rng.uniform(0.06, 0.20)) * (0.5 + expressiveness), 0.090)

    c1_normal_scale = float(rng.uniform(0.80, 1.15))
    c2_normal_scale = float(rng.uniform(0.95, 1.35))
    c2_height_scale = float(rng.uniform(0.25, 0.70))

    def _make_path(sign: float, bend_scale: float) -> tuple[list[np.ndarray], dict]:
        normal = base_normal * sign
        scaled_bend = bend_mag * float(bend_scale)
        control_1 = start.copy()
        control_1[:2] -= tangent * backtrack
        control_2 = start + t2 * (target - start)
        control_1[:2] += normal * scaled_bend * c1_normal_scale
        control_2[:2] += normal * scaled_bend * c2_normal_scale
        control_1[2] = max(float(control_1[2]), float(target[2])) + height
        control_2[2] = max(float(control_2[2]), float(target[2])) + height * c2_height_scale
        n_waypoints = max(2, int(args.approach_curve_waypoints))
        waypoints = [
            _bezier_point([start, control_1, control_2, target], float(i) / float(n_waypoints))
            for i in range(1, n_waypoints + 1)
        ]
        return waypoints, {
            "tangent_xy": tangent,
            "normal_xy": normal,
            "bend": float(scaled_bend),
            "requested_bend": float(bend_mag),
            "bend_scale": float(bend_scale),
            "height": float(height),
            "backtrack": float(backtrack),
            "expressiveness": float(expressiveness),
            "workspace_violation": _path_workspace_violation([start, *waypoints], args, start=start, target=target),
        }

    candidates = [_make_path(sign, bend_scale) for sign in signs for bend_scale in (1.0, 0.75, 0.50, 0.35)]
    waypoints, info = min(candidates, key=lambda candidate: float(candidate[1]["workspace_violation"]))
    return waypoints, info


def _approach_miss_target(
    hover_target: np.ndarray,
    curve_info: dict | None,
    *,
    args: Args,
    rng: np.random.Generator,
) -> np.ndarray:
    correction_max = max(0.0, float(args.approach_correction_max_offset_m))
    miss = np.asarray(hover_target, dtype=np.float64).copy()
    if correction_max <= 0.0:
        return miss

    if curve_info:
        tangent = np.asarray(curve_info.get("tangent_xy", [1.0, 0.0]), dtype=np.float64)
        normal = np.asarray(curve_info.get("normal_xy", [0.0, 1.0]), dtype=np.float64)
        expressiveness = float(curve_info.get("expressiveness", 0.5))
    else:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        tangent = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        expressiveness = 0.5

    tangent_norm = float(np.linalg.norm(tangent))
    normal_norm = float(np.linalg.norm(normal))
    if tangent_norm <= 1e-8 or normal_norm <= 1e-8:
        return miss
    tangent = tangent / tangent_norm
    normal = normal / normal_norm

    forward = correction_max * float(rng.uniform(-0.85, 1.15))
    if abs(forward) < 0.25 * correction_max:
        forward = float(np.sign(forward) if abs(forward) > 1e-8 else rng.choice([-1.0, 1.0])) * 0.25 * correction_max
    side = correction_max * float(rng.uniform(-0.85, 0.85)) * float(0.6 + 0.8 * expressiveness)
    if abs(side) < 0.20 * correction_max:
        side = float(np.sign(side) if abs(side) > 1e-8 else rng.choice([-1.0, 1.0])) * 0.20 * correction_max
    miss[:2] += tangent * forward + normal * side
    miss[2] = hover_target[2]
    return miss


def _lift_curve_waypoints(
    start_base: np.ndarray,
    target_base: np.ndarray,
    *,
    args: Args,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    start = np.asarray(start_base, dtype=np.float64)
    target = np.asarray(target_base, dtype=np.float64)
    offset_max = max(0.0, float(args.lift_curve_max_offset_m))
    if offset_max <= 0.0:
        return [target.copy()]

    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    side = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    bend = offset_max * float(rng.uniform(0.35, 1.0))
    control_1 = start + 0.35 * (target - start)
    control_2 = start + 0.75 * (target - start)
    control_1[:2] += side * bend
    control_2[:2] += side * bend * float(rng.uniform(0.35, 0.75))
    n_waypoints = max(8, int(args.approach_curve_waypoints) // 2)
    return [
        _bezier_point([start, control_1, control_2, target], float(i) / float(n_waypoints))
        for i in range(1, n_waypoints + 1)
    ]


def _hover_correction_waypoints(
    correction_start: np.ndarray,
    hover_target: np.ndarray,
    curve_info: dict | None,
    *,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    start = np.asarray(correction_start, dtype=np.float64)
    hover = np.asarray(hover_target, dtype=np.float64)
    if curve_info:
        normal = np.asarray(curve_info.get("normal_xy", [0.0, 1.0]), dtype=np.float64)
    else:
        normal = np.array([0.0, 1.0], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-8:
        normal = np.array([0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / normal_norm

    correction = hover - start
    if float(np.linalg.norm(correction[:2])) <= 1e-8:
        return [hover.copy()]

    side = np.zeros(3, dtype=np.float64)
    side[:2] = normal * float(rng.uniform(0.010, 0.026))
    control_1 = start + 0.35 * correction + side
    control_2 = start + 0.78 * correction - side * float(rng.uniform(0.25, 0.65))
    return [
        _bezier_point([start, control_1, control_2, hover], float(i) / 16.0)
        for i in range(1, 17)
    ]


def _make_stage_delta(
    delta: np.ndarray,
    *,
    args: Args,
    rng: np.random.Generator,
    allow_z: bool = True,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).copy()
    if not allow_z:
        delta[2] = 0.0

    norm = float(np.linalg.norm(delta))
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float64)

    base_step = max(1e-6, float(args.action_translation_norm_m))
    max_norm = base_step
    jitter = float(args.action_step_jitter_frac)
    step_scale = float(rng.uniform(max(0.1, 1.0 - jitter), 1.0 + jitter))
    step_norm = min(norm, base_step * step_scale, max_norm)

    direction = delta / norm
    if norm > float(args.pos_tol_m) * 3.0 and float(args.action_direction_noise_m) > 0.0:
        noise = rng.normal(0.0, float(args.action_direction_noise_m), size=3)
        if not allow_z:
            noise[2] = 0.0
        noisy_direction = direction + noise
        noisy_norm = float(np.linalg.norm(noisy_direction))
        if noisy_norm > 1e-8 and float(np.dot(noisy_direction, direction)) > 0.25:
            direction = noisy_direction / noisy_norm

    action_delta = direction * step_norm
    remaining_norm = float(np.linalg.norm(delta))
    if remaining_norm <= step_norm:
        action_delta = delta
    return action_delta


def _make_action(delta_xyz: np.ndarray, gripper: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.asarray(delta_xyz, dtype=np.float32)
    action[6] = float(gripper)
    return action


def _min_generated_translation(args: Args) -> float:
    return max(0.0, float(args.min_generated_translation_m))


def _is_upward_lift_escape(stage_name: str, delta: np.ndarray) -> bool:
    return "lift" in str(stage_name).lower() and float(np.asarray(delta, dtype=np.float64)[2]) > 0.0


def _path_delta(
    delta: np.ndarray,
    *,
    args: Args,
    rng: np.random.Generator,
    step_scale: float = 1.0,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float64)

    max_norm = max(1e-6, float(args.action_translation_norm_m)) * max(0.05, float(step_scale))
    base_step = min(max_norm, max(float(args.xy_step_m), float(args.z_step_m)) * max(0.05, float(step_scale)))
    jitter = max(0.0, float(args.action_step_jitter_frac))
    step_norm = min(norm, base_step * float(rng.uniform(max(0.2, 1.0 - jitter), 1.0 + jitter)), max_norm)
    return delta / norm * step_norm


def _step(env, action: np.ndarray):
    action_tensor = torch.as_tensor(action[None, :], dtype=torch.float32, device=env.unwrapped.device)
    obs, _, terminated, truncated, info = env.step(action_tensor)
    return obs, info, bool(_tensor_row(terminated)[0]), bool(_tensor_row(truncated)[0])


def _robot_joint_debug(env) -> dict:
    robot = env.unwrapped.agent.robot
    qpos = _tensor_row(robot.get_qpos()).astype(np.float64)
    qvel = _tensor_row(robot.get_qvel()).astype(np.float64)
    payload = {
        "qpos": qpos,
        "qvel": qvel,
    }
    try:
        qlimits = _tensor_row(robot.get_qlimits()).astype(np.float64)
        qlimits = qlimits[: len(qpos)]
        lower_margin = qpos - qlimits[:, 0]
        upper_margin = qlimits[:, 1] - qpos
        limit_margin = np.minimum(lower_margin, upper_margin)
        closest_idx = int(np.argmin(limit_margin))
        payload.update(
            {
                "limit_margin": limit_margin,
                "closest_limit_joint": closest_idx,
                "closest_limit_margin": float(limit_margin[closest_idx]),
            }
        )
    except Exception:
        pass
    return payload


def _closest_joint_limit_margin(env) -> float | None:
    robot = env.unwrapped.agent.robot
    try:
        qpos = _tensor_row(robot.get_qpos()).astype(np.float64)
        qlimits = _tensor_row(robot.get_qlimits()).astype(np.float64)
        arm_dof = min(6, len(qpos), len(qlimits))
        if arm_dof <= 0:
            return None
        arm_qpos = qpos[:arm_dof]
        arm_qlimits = qlimits[:arm_dof]
        lower_margin = arm_qpos - arm_qlimits[:, 0]
        upper_margin = arm_qlimits[:, 1] - arm_qpos
        return float(np.min(np.minimum(lower_margin, upper_margin)))
    except Exception:
        value = _robot_joint_debug(env).get("closest_limit_margin")
        if value is None:
            return None
        return float(value)


def _debug_joint_names(env) -> list[str]:
    try:
        return [str(joint.name) for joint in env.unwrapped.agent.robot.get_active_joints()]
    except Exception:
        return []


def _round_debug_value(value, digits: int = 4):
    if isinstance(value, dict):
        return {str(k): _round_debug_value(v, digits=digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_debug_value(v, digits=digits) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _round_debug_value(value.tolist(), digits=digits)
    if isinstance(value, np.generic):
        return _round_debug_value(value.item(), digits=digits)
    if torch.is_tensor(value):
        return _round_debug_value(value.detach().cpu().tolist(), digits=digits)
    if isinstance(value, float):
        return round(float(value), digits)
    return value


def _write_action_debug(debug_log, payload: dict) -> None:
    if debug_log is None:
        return
    debug_log.write(json.dumps(_round_debug_value(payload), separators=(",", ":")) + "\n")
    debug_log.flush()


def _log_action_step(
    env,
    action: np.ndarray,
    *,
    args: Args,
    actions: list[np.ndarray],
    stage_name: str,
    target_base_xyz: np.ndarray,
    step_idx: int,
    debug_log,
    color: str,
    attempt_index: int,
) -> tuple[dict, dict]:
    before_tcp = _tcp_base_xyz(env)
    before_joints = _robot_joint_debug(env)
    expected_delta = np.asarray(action[:3], dtype=np.float64)
    expected_after_tcp = before_tcp + expected_delta
    obs, info, terminated, truncated = _step(env, action)
    after_tcp = _tcp_base_xyz(env)
    after_joints = _robot_joint_debug(env)
    target = np.asarray(target_base_xyz, dtype=np.float64)
    _write_action_debug(
        debug_log,
        {
            "t": "step",
            "i": len(actions),
            "st": stage_name,
            "si": int(step_idx),
            "target": target,
            "tcp0": before_tcp,
            "cmd": expected_delta,
            "grip": float(action[6]),
            "exp": expected_after_tcp,
            "tcp1": after_tcp,
            "act": after_tcp - before_tcp,
            "err": after_tcp - expected_after_tcp,
            "dist": [
                float(np.linalg.norm(target - before_tcp)),
                float(np.linalg.norm(target - after_tcp)),
            ],
            "q0": before_joints.get("qpos"),
            "q1": after_joints.get("qpos"),
            "qv1": after_joints.get("qvel"),
            "lim1": after_joints.get("limit_margin"),
            "lim_min": after_joints.get("closest_limit_margin"),
            "lim_j": after_joints.get("closest_limit_joint"),
            "done": bool(terminated or truncated),
        },
    )
    return obs, info


def _move_to_base_target(
    env,
    target_base_xyz: np.ndarray,
    *,
    gripper: float,
    args: Args,
    actions: list[np.ndarray],
    stage_name: str,
    action_budget: int,
    debug_log,
    color: str,
    attempt_index: int,
    rng: np.random.Generator,
    allow_z: bool = True,
) -> tuple[dict, dict]:
    start = _tcp_base_xyz(env)
    if args.verbose:
        print(
            f"    stage={stage_name} start={np.array2string(start, precision=4, suppress_small=True)} "
            f"target={np.array2string(np.asarray(target_base_xyz), precision=4, suppress_small=True)} "
            f"gripper={gripper:+.1f}"
        )
    _write_action_debug(
        debug_log,
        {
            "t": "stage_start",
            "st": stage_name,
            "i": len(actions),
            "target": np.asarray(target_base_xyz, dtype=np.float64),
            "tcp": start,
            "q": _robot_joint_debug(env).get("qpos"),
        },
    )
    stage_start_actions = len(actions)
    obs = None
    info = {}
    stage_failed = False
    last_dist = None
    stalled = 0
    stage_steps = max(0, min(int(args.max_stage_steps), int(action_budget)))
    for step_idx in range(stage_steps):
        current = _tcp_base_xyz(env)
        delta = np.asarray(target_base_xyz, dtype=np.float64) - current
        min_translation = _min_generated_translation(args)
        if float(np.linalg.norm(delta)) <= max(float(args.pos_tol_m), min_translation):
            break
        dist = float(np.linalg.norm(delta))
        if last_dist is not None and dist >= last_dist - 1e-5:
            stalled += 1
        else:
            stalled = 0
        last_dist = dist
        if stalled >= int(args.stall_steps):
            if args.verbose:
                print(f"Stage {stage_name}: stopping after {stalled} stalled steps, remaining={dist:.4f} m")
            break
        limit_margin = _closest_joint_limit_margin(env)
        allow_close_target_attempt = dist <= max(0.020, 5.0 * float(args.pos_tol_m))
        allow_lift_escape = _is_upward_lift_escape(stage_name, delta)
        if (
            limit_margin is not None
            and limit_margin <= float(args.joint_limit_stop_margin_rad)
            and not allow_close_target_attempt
            and not allow_lift_escape
        ):
            _write_action_debug(
                debug_log,
                {
                    "t": "joint_limit_stop",
                    "st": stage_name,
                    "i": len(actions),
                    "si": int(step_idx),
                    "lim_min": float(limit_margin),
                    "tcp": current,
                    "target": np.asarray(target_base_xyz, dtype=np.float64),
                    "lift_escape": bool(allow_lift_escape),
                },
            )
            stage_failed = True
            break
        delta_cmd = _make_stage_delta(delta, args=args, rng=rng, allow_z=allow_z)
        unscaled_cmd = delta_cmd.copy()
        if (
            limit_margin is not None
            and limit_margin < float(args.joint_limit_slow_margin_rad)
            and not allow_lift_escape
        ):
            slow_span = max(1e-6, float(args.joint_limit_slow_margin_rad) - float(args.joint_limit_stop_margin_rad))
            slow_scale = float(np.clip((limit_margin - float(args.joint_limit_stop_margin_rad)) / slow_span, 0.15, 1.0))
            delta_cmd *= slow_scale
        if float(np.linalg.norm(delta_cmd)) < min_translation:
            unscaled_norm = float(np.linalg.norm(unscaled_cmd))
            if unscaled_norm >= min_translation:
                delta_cmd = unscaled_cmd / unscaled_norm * min_translation
            else:
                _write_action_debug(
                    debug_log,
                    {
                        "t": "small_motion_skip",
                        "st": stage_name,
                        "i": len(actions),
                        "si": int(step_idx),
                        "cmd_norm": float(np.linalg.norm(delta_cmd)),
                        "min_translation": float(min_translation),
                        "tcp": current,
                        "target": np.asarray(target_base_xyz, dtype=np.float64),
                    },
                )
                break
        action = _make_action(delta_cmd, gripper)
        actions.append(action)
        before = current
        obs, info = _log_action_step(
            env,
            action,
            args=args,
            actions=actions,
            stage_name=stage_name,
            target_base_xyz=target_base_xyz,
            step_idx=step_idx,
            debug_log=debug_log,
            color=color,
            attempt_index=attempt_index,
        )
        actual = _tcp_base_xyz(env) - before
        commanded = np.asarray(action[:3], dtype=np.float64)
        commanded_norm_sq = float(np.dot(commanded, commanded))
        if commanded_norm_sq > 1e-10:
            response_ratio = float(np.dot(actual, commanded) / commanded_norm_sq)
            if response_ratio < float(args.tcp_bad_response_ratio):
                _write_action_debug(
                    debug_log,
                    {
                        "t": "bad_tcp_response_stop",
                        "st": stage_name,
                        "i": len(actions),
                        "si": int(step_idx),
                        "ratio": response_ratio,
                        "cmd": commanded,
                        "act": actual,
                        "tcp": _tcp_base_xyz(env),
                    },
                )
                stage_failed = True
                break
    remaining_budget = max(0, int(action_budget) - (len(actions) - stage_start_actions))
    for hold_step in range(min(max(0, int(args.hold_steps)), remaining_budget)):
        action = _make_action(np.zeros(3, dtype=np.float32), gripper)
        actions.append(action)
        obs, info = _log_action_step(
            env,
            action,
            args=args,
            actions=actions,
            stage_name=f"{stage_name}_hold",
            target_base_xyz=target_base_xyz,
            step_idx=hold_step,
            debug_log=debug_log,
            color=color,
            attempt_index=attempt_index,
        )
    end = _tcp_base_xyz(env)
    if args.verbose:
        print(
            f"    stage={stage_name} end={np.array2string(end, precision=4, suppress_small=True)} "
            f"remaining={float(np.linalg.norm(np.asarray(target_base_xyz) - end)):.4f} "
            f"actions_total={len(actions)}"
        )
    _write_action_debug(
        debug_log,
        {
            "t": "stage_end",
            "st": stage_name,
            "i": len(actions),
            "target": np.asarray(target_base_xyz, dtype=np.float64),
            "tcp": end,
            "rem": float(np.linalg.norm(np.asarray(target_base_xyz) - end)),
            "failed": bool(stage_failed),
            "q": _robot_joint_debug(env).get("qpos"),
        },
    )
    if stage_failed:
        info = dict(info)
        info["_script_stage_failed"] = True
    return obs, info


def _follow_base_path(
    env,
    path_base_xyz: list[np.ndarray],
    *,
    gripper: float,
    args: Args,
    actions: list[np.ndarray],
    stage_name: str,
    action_budget: int,
    debug_log,
    color: str,
    attempt_index: int,
    rng: np.random.Generator,
    step_scale: float = 1.0,
    lookahead_m: float | None = None,
) -> tuple[dict, dict]:
    path = [np.asarray(point, dtype=np.float64).copy() for point in path_base_xyz]
    path = [point for point in path if point.shape[0] >= 3]
    if len(path) < 2:
        target = path[-1] if path else _tcp_base_xyz(env)
        return _move_to_base_target(
            env,
            target,
            gripper=gripper,
            args=args,
            actions=actions,
            stage_name=stage_name,
            action_budget=action_budget,
            debug_log=debug_log,
            color=color,
            attempt_index=attempt_index,
            rng=rng,
        )

    start = _tcp_base_xyz(env)
    if args.verbose:
        print(
            f"    stage={stage_name} continuous_path start={np.array2string(start, precision=4, suppress_small=True)} "
            f"target={np.array2string(path[-1], precision=4, suppress_small=True)} gripper={gripper:+.1f}"
        )
    _write_action_debug(
        debug_log,
        {
            "t": "path_start",
            "st": stage_name,
            "i": len(actions),
            "target": path[-1],
            "tcp": start,
            "points": len(path),
            "step_scale": float(step_scale),
            "lookahead_m": None if lookahead_m is None else float(lookahead_m),
            "q": _robot_joint_debug(env).get("qpos"),
        },
    )

    obs = None
    info = {}
    stage_failed = False
    stage_steps = max(0, min(int(args.max_stage_steps), int(action_budget)))
    target_idx = 1
    lookahead = max(float(args.action_translation_norm_m), float(args.xy_step_m), float(args.z_step_m)) * 1.35
    if lookahead_m is not None:
        lookahead = max(float(args.pos_tol_m), float(lookahead_m))
    closed_path = float(np.linalg.norm(path[-1] - path[0])) <= max(0.010, 3.0 * float(args.pos_tol_m))
    if closed_path and lookahead_m is None:
        lookahead = min(lookahead, max(0.010, float(args.xy_step_m) * 0.50))
    for step_idx in range(stage_steps):
        current = _tcp_base_xyz(env)
        final_delta = path[-1] - current
        min_translation = _min_generated_translation(args)
        if target_idx >= len(path) - 1 and float(np.linalg.norm(final_delta)) <= max(float(args.pos_tol_m), min_translation):
            break

        while target_idx < len(path) - 1 and float(np.linalg.norm(path[target_idx] - current)) < lookahead:
            target_idx += 1

        target = path[target_idx]
        delta = target - current
        if float(np.linalg.norm(delta)) <= float(args.pos_tol_m) and target_idx < len(path) - 1:
            target_idx += 1
            target = path[target_idx]
            delta = target - current

        limit_margin = _closest_joint_limit_margin(env)
        allow_close_target_attempt = float(np.linalg.norm(delta)) <= max(0.020, 5.0 * float(args.pos_tol_m))
        allow_lift_escape = _is_upward_lift_escape(stage_name, delta)
        if (
            limit_margin is not None
            and limit_margin <= float(args.joint_limit_stop_margin_rad)
            and not allow_close_target_attempt
            and not allow_lift_escape
        ):
            _write_action_debug(
                debug_log,
                {
                    "t": "joint_limit_stop",
                    "st": stage_name,
                    "i": len(actions),
                    "si": int(step_idx),
                    "lim_min": float(limit_margin),
                    "tcp": current,
                    "target": target,
                    "lift_escape": bool(allow_lift_escape),
                },
            )
            stage_failed = True
            break
        delta_cmd = _path_delta(delta, args=args, rng=rng, step_scale=step_scale)
        unscaled_cmd = delta_cmd.copy()
        if (
            limit_margin is not None
            and limit_margin < float(args.joint_limit_slow_margin_rad)
            and not allow_lift_escape
        ):
            slow_span = max(1e-6, float(args.joint_limit_slow_margin_rad) - float(args.joint_limit_stop_margin_rad))
            slow_scale = float(np.clip((limit_margin - float(args.joint_limit_stop_margin_rad)) / slow_span, 0.15, 1.0))
            delta_cmd *= slow_scale
        if float(np.linalg.norm(delta_cmd)) < min_translation:
            unscaled_norm = float(np.linalg.norm(unscaled_cmd))
            if unscaled_norm >= min_translation:
                delta_cmd = unscaled_cmd / unscaled_norm * min_translation
            else:
                _write_action_debug(
                    debug_log,
                    {
                        "t": "small_motion_skip",
                        "st": stage_name,
                        "i": len(actions),
                        "si": int(step_idx),
                        "cmd_norm": float(np.linalg.norm(delta_cmd)),
                        "min_translation": float(min_translation),
                        "tcp": current,
                        "target": target,
                    },
                )
                if target_idx < len(path) - 1:
                    target_idx += 1
                    continue
                break
        action = _make_action(delta_cmd, gripper)
        actions.append(action)
        before = current
        obs, info = _log_action_step(
            env,
            action,
            args=args,
            actions=actions,
            stage_name=stage_name,
            target_base_xyz=target,
            step_idx=step_idx,
            debug_log=debug_log,
            color=color,
            attempt_index=attempt_index,
        )
        actual = _tcp_base_xyz(env) - before
        commanded = np.asarray(action[:3], dtype=np.float64)
        commanded_norm_sq = float(np.dot(commanded, commanded))
        if commanded_norm_sq > 1e-10:
            response_ratio = float(np.dot(actual, commanded) / commanded_norm_sq)
            if response_ratio < float(args.tcp_bad_response_ratio):
                _write_action_debug(
                    debug_log,
                    {
                        "t": "bad_tcp_response_stop",
                        "st": stage_name,
                        "i": len(actions),
                        "si": int(step_idx),
                        "ratio": response_ratio,
                        "cmd": commanded,
                        "act": actual,
                        "tcp": _tcp_base_xyz(env),
                    },
                )
                stage_failed = True
                break

    end = _tcp_base_xyz(env)
    if args.verbose:
        print(
            f"    stage={stage_name} continuous_path end={np.array2string(end, precision=4, suppress_small=True)} "
            f"remaining={float(np.linalg.norm(path[-1] - end)):.4f} actions_total={len(actions)}"
        )
    _write_action_debug(
        debug_log,
        {
            "t": "path_end",
            "st": stage_name,
            "i": len(actions),
            "target": path[-1],
            "tcp": end,
            "rem": float(np.linalg.norm(path[-1] - end)),
            "failed": bool(stage_failed),
            "q": _robot_joint_debug(env).get("qpos"),
        },
    )
    if stage_failed:
        info = dict(info)
        info["_script_stage_failed"] = True
    return obs, info


def _collect_one(
    env,
    args: Args,
    color: str,
    robot_base_pose_p,
    robot_base_pose_q,
    robot_init_qpos,
    *,
    debug_log_path: Path | None = None,
    attempt_index: int = 0,
) -> tuple[bool, dict, np.ndarray]:
    args.airi_cube_target_color = color
    replay_robot_init_qpos = _with_open_gripper_qpos(env, robot_init_qpos)
    obs, info = _reset_record_env(
        env,
        args,
        robot_base_pose_p,
        robot_base_pose_q,
        replay_robot_init_qpos,
        reconfigure=False,
        episode_id=attempt_index,
    )
    actions: list[np.ndarray] = []
    rng = np.random.default_rng(int(args.random_seed) + int(attempt_index) * 10007)
    debug_log = None
    if args.action_debug_log and debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_log = debug_log_path.open("w", encoding="utf-8")
    try:
        proxy_cfg = _load_proxy_config(args.planner_config_file)
        calibration = _object_calibration(
            proxy_cfg,
            color,
            use_planner_object_calibrations=bool(args.use_planner_object_calibrations),
        )
        if args.use_third_party_lift_delta:
            args.lift_delta_z = float(proxy_cfg.get("planner_lift_delta_z", args.lift_delta_z))
        args.xy_step_m = max(float(args.xy_step_m), float(proxy_cfg.get("planner_proxy_xy_step_m", 0.0)))
        args.z_step_m = max(float(args.z_step_m), float(proxy_cfg.get("planner_proxy_z_step_m", 0.0)))
        args.action_translation_norm_m = max(1e-6, float(args.action_translation_norm_m))
        args.pos_tol_m = max(float(args.pos_tol_m), float(proxy_cfg.get("planner_proxy_pos_tol_m", 0.0)))
        args.max_stage_steps = min(int(args.max_stage_steps), int(args.max_demo_actions))
        args.stall_steps = min(int(args.stall_steps), int(args.max_stage_steps))
        args.hold_steps = max(0, int(args.hold_steps))
        args.grasp_hold_steps = max(1, min(int(args.grasp_hold_steps), int(args.max_demo_actions)))

        cube_xyz = _cube_world_xyz(env, color)
        cube_base = _world_to_robot_base(env, cube_xyz)
        raw_pregrasp_offset = np.asarray(calibration["pregrasp_offset_xyz"], dtype=np.float64)
        raw_descend_offset = np.asarray(calibration["descend_offset_xyz"], dtype=np.float64)
        if args.transform_planner_offsets_to_current_base:
            pregrasp_offset = _reference_offset_to_current_base(raw_pregrasp_offset, robot_base_pose_q)
            descend_offset = _reference_offset_to_current_base(raw_descend_offset, robot_base_pose_q)
        else:
            pregrasp_offset = raw_pregrasp_offset
            descend_offset = raw_descend_offset
        sweep_index, sweep_offset = _offset_sweep_candidate(args, attempt_index)
        pregrasp_offset = pregrasp_offset + sweep_offset
        descend_offset = descend_offset + sweep_offset
        pregrasp_base = cube_base + pregrasp_offset
        descend_base = cube_base + descend_offset
        cube_height = float(2.0 * np.asarray(AIRI_CUBE_HALF_SIZE, dtype=np.float64)[2])
        min_descend_z = float(cube_base[2] + max(0.0, float(args.min_descend_offset_z_m)))
        descend_base[2] = max(float(descend_base[2]), min_descend_z)
        approach_xy_hover_z = float(
            max(
                cube_base[2] + max(0.0, float(args.approach_xy_hover_cube_heights)) * cube_height,
                pregrasp_base[2] + max(0.0, float(args.approach_top_clearance_m)),
            )
        )
        _write_action_debug(
            debug_log,
            {
                "t": "meta",
                "attempt": int(attempt_index),
                "color": color,
                "object": COLOR_TO_THIRD_PARTY_OBJECT[color],
                "control": "arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos",
                "sim": args.sim_backend,
                "render": args.render_backend_override,
                "joint_names": _debug_joint_names(env),
                "cube_w": cube_xyz,
                "cube_b": cube_base,
                "raw_pre_offset_b": raw_pregrasp_offset,
                "raw_desc_offset_b": raw_descend_offset,
                "pre_offset_b": pregrasp_offset,
                "desc_offset_b": descend_offset,
                "offset_sweep_i": int(sweep_index),
                "offset_sweep_b": sweep_offset,
                "pre_b": pregrasp_base,
                "desc_b": descend_base,
                "tcp0": _tcp_base_xyz(env),
                "q0": _robot_joint_debug(env).get("qpos"),
                "params": {
                    "xy": float(args.xy_step_m),
                    "z": float(args.z_step_m),
                    "norm": float(args.action_translation_norm_m),
                    "jitter": float(args.action_step_jitter_frac),
                    "dir_noise": float(args.action_direction_noise_m),
                    "tol": float(args.pos_tol_m),
                    "min_generated_translation": float(args.min_generated_translation_m),
                    "max_actions": int(args.max_demo_actions),
                    "max_saved_actions": int(args.max_saved_demo_actions),
                    "planner_obj_cal": bool(args.use_planner_object_calibrations),
                    "offset_transform": bool(args.transform_planner_offsets_to_current_base),
                    "offset_reference_base_q": np.asarray(AIRI_CUBES_ROBOT_BASE_POSE_Q, dtype=np.float64),
                    "offset_current_base_q": np.asarray(robot_base_pose_q, dtype=np.float64),
                    "offset_sweep_spec": str(args.planner_offset_sweep_xyz_m),
                    "xy_hover_cube_heights": float(args.approach_xy_hover_cube_heights),
                    "xy_hover_z": float(approach_xy_hover_z),
                    "top_clearance": float(args.approach_top_clearance_m),
                    "min_desc_z": float(min_descend_z),
                    "curve": bool(args.approach_curve_enabled),
                    "curve_min_offset": float(args.approach_curve_min_offset_m),
                    "curve_max_offset": float(args.approach_curve_max_offset_m),
                    "curve_height_offset": float(args.approach_curve_height_offset_m),
                    "curve_waypoints": int(args.approach_curve_waypoints),
                    "curve_base_x_min": float(args.approach_curve_base_x_min_m),
                    "curve_base_x_max": float(args.approach_curve_base_x_max_m),
                    "curve_base_y_min": float(args.approach_curve_base_y_min_m),
                    "curve_base_y_max": float(args.approach_curve_base_y_max_m),
                    "curve_max_reverse": float(args.approach_curve_max_reverse_m),
                    "correction": bool(args.approach_correction_enabled),
                    "miss_probability": float(args.approach_miss_probability),
                    "correction_max_offset": float(args.approach_correction_max_offset_m),
                    "joint_limit_slow_margin": float(args.joint_limit_slow_margin_rad),
                    "joint_limit_stop_margin": float(args.joint_limit_stop_margin_rad),
                    "tcp_bad_response_ratio": float(args.tcp_bad_response_ratio),
                    "lift_curve_max_offset": float(args.lift_curve_max_offset_m),
                    "final_hold_steps": int(args.final_hold_steps),
                "recovery": bool(args.recovery_enabled),
                },
            },
        )
        if args.verbose:
            print(
                f"  object={COLOR_TO_THIRD_PARTY_OBJECT[color]} "
                f"cube_world={np.array2string(cube_xyz, precision=4, suppress_small=True)} "
                f"pregrasp_base={np.array2string(pregrasp_base, precision=4, suppress_small=True)} "
                f"descend_base={np.array2string(descend_base, precision=4, suppress_small=True)}"
            )

        # Keep the scripted route conservative, but avoid a single repeated straight-line trace.
        # The curve/correction waypoints stay near the safe hover/pregrasp region.
        pregrasp_xy_base = pregrasp_base.copy()
        pregrasp_xy_base[2] = approach_xy_hover_z
        max_actions = max(1, int(args.max_demo_actions))
        stage_budgets = {
            "approach_xy": min(46, max_actions),
            "approach_z": min(12, max_actions),
            "descend": min(10, max_actions),
            "lift": min(20, max_actions),
        }

        def _remaining_budget(stage_name: str) -> int:
            return max(0, min(stage_budgets[stage_name], max_actions - len(actions)))

        def _remaining_global() -> int:
            return max(0, max_actions - len(actions))

        def _stage_failed() -> bool:
            return isinstance(info, dict) and bool(info.get("_script_stage_failed"))

        def _near_target(target: np.ndarray, tolerance: float) -> bool:
            return float(np.linalg.norm(np.asarray(target, dtype=np.float64) - _tcp_base_xyz(env))) <= float(tolerance)

        def _approach_to_pregrasp(prefix: str, target_pregrasp: np.ndarray, target_descend: np.ndarray) -> bool:
            nonlocal obs, info
            hover_target = target_pregrasp.copy()
            hover_target[2] = approach_xy_hover_z
            start_safe = _tcp_base_xyz(env)
            if float(start_safe[2]) < approach_xy_hover_z - float(args.pos_tol_m) and _remaining_global() > 0:
                start_safe[2] = approach_xy_hover_z
                obs, info = _move_to_base_target(
                    env,
                    start_safe,
                    gripper=-1.0,
                    args=args,
                    actions=actions,
                    stage_name=f"{prefix}_precurve_lift",
                    action_budget=min(10, _remaining_budget("approach_z"), _remaining_global()),
                    debug_log=debug_log,
                    color=color,
                    attempt_index=attempt_index,
                    rng=rng,
                )
                if _stage_failed():
                    return False

            if bool(args.approach_curve_enabled) and _remaining_global() > 0:
                start_base = _tcp_base_xyz(env)
                _, hover_curve_info = _approach_bezier_waypoints(start_base, hover_target, args=args, rng=rng)
                use_miss_correction = bool(args.approach_correction_enabled) and (
                    float(rng.uniform(0.0, 1.0)) < float(np.clip(args.approach_miss_probability, 0.0, 1.0))
                )
                approach_target = (
                    _approach_miss_target(hover_target, hover_curve_info, args=args, rng=rng)
                    if use_miss_correction
                    else hover_target
                )
                waypoints, curve_info = _approach_bezier_waypoints(start_base, approach_target, args=args, rng=rng)
                curve_info["hover_target"] = hover_target
                curve_info["approach_target"] = approach_target
                curve_info["miss_correction"] = bool(use_miss_correction)
                _write_action_debug(
                    debug_log,
                    {
                        "t": "curve_plan",
                        "st": f"{prefix}_curve",
                        "i": len(actions),
                        "start": start_base,
                        "target": approach_target,
                        "final_hover_target": hover_target,
                        "miss_correction": bool(use_miss_correction),
                        "waypoints": waypoints,
                        "info": curve_info,
                    },
                )
                obs, info = _follow_base_path(
                    env,
                    [start_base, *waypoints],
                    gripper=-1.0,
                    args=args,
                    actions=actions,
                    stage_name=f"{prefix}_curve",
                    action_budget=min(_remaining_budget("approach_xy"), _remaining_global()),
                    debug_log=debug_log,
                    color=color,
                    attempt_index=attempt_index,
                    rng=rng,
                )
                if _stage_failed():
                    _write_action_debug(
                        debug_log,
                        {
                            "t": "approach_failed",
                            "st": f"{prefix}_curve",
                            "i": len(actions),
                            "tcp": _tcp_base_xyz(env),
                            "target": approach_target,
                            "rem": float(np.linalg.norm(approach_target - _tcp_base_xyz(env))),
                        },
                    )
                    return False
                curve_rem = float(np.linalg.norm(approach_target - _tcp_base_xyz(env)))
                if (
                    not use_miss_correction
                    and curve_rem > max(0.020, 4.0 * float(args.pos_tol_m))
                    and _remaining_global() > 0
                ):
                    obs, info = _move_to_base_target(
                        env,
                        hover_target,
                        gripper=-1.0,
                        args=args,
                        actions=actions,
                        stage_name=f"{prefix}_curve_settle_to_hover",
                        action_budget=min(10, _remaining_global()),
                        debug_log=debug_log,
                        color=color,
                        attempt_index=attempt_index,
                        rng=rng,
                    )
                    if _stage_failed():
                        return False
            else:
                curve_info = None
                obs, info = _move_to_base_target(
                    env,
                    hover_target,
                    gripper=-1.0,
                    args=args,
                    actions=actions,
                    stage_name=f"{prefix}_approach_xy",
                    action_budget=min(_remaining_budget("approach_xy"), _remaining_global()),
                    debug_log=debug_log,
                    color=color,
                    attempt_index=attempt_index,
                    rng=rng,
                )
                if _stage_failed():
                    return False

            if bool(args.approach_correction_enabled) and bool(curve_info.get("miss_correction", False)) and _remaining_global() > 0:
                correction_start = _tcp_base_xyz(env)
                correction_path = [
                    correction_start,
                    *_hover_correction_waypoints(correction_start, hover_target, curve_info, rng=rng),
                ]
                obs, info = _follow_base_path(
                    env,
                    correction_path,
                    gripper=-1.0,
                    args=args,
                    actions=actions,
                    stage_name=f"{prefix}_hover_overshoot_correct",
                    action_budget=min(28, _remaining_global()),
                    debug_log=debug_log,
                    color=color,
                    attempt_index=attempt_index,
                    rng=rng,
                    step_scale=0.45,
                    lookahead_m=max(0.010, float(args.xy_step_m) * 0.45),
                )
                if _stage_failed():
                    _write_action_debug(
                        debug_log,
                        {
                            "t": "approach_failed",
                            "st": f"{prefix}_hover_overshoot_correct",
                            "i": len(actions),
                            "tcp": _tcp_base_xyz(env),
                            "target": hover_target,
                            "rem": float(np.linalg.norm(hover_target - _tcp_base_xyz(env))),
                        },
                    )
                    return False
                correction_rem = float(np.linalg.norm(hover_target - _tcp_base_xyz(env)))
                if correction_rem > max(0.010, 3.0 * float(args.pos_tol_m)) and _remaining_global() > 0:
                    obs, info = _move_to_base_target(
                        env,
                        hover_target,
                        gripper=-1.0,
                        args=args,
                        actions=actions,
                        stage_name=f"{prefix}_hover_correct_final",
                        action_budget=min(8, _remaining_global()),
                        debug_log=debug_log,
                        color=color,
                        attempt_index=attempt_index,
                        rng=rng,
                    )
                    if _stage_failed():
                        return False

            if not _near_target(hover_target, max(0.030, 6.0 * float(args.pos_tol_m))):
                _write_action_debug(
                    debug_log,
                    {
                        "t": "approach_failed",
                        "st": f"{prefix}_hover",
                        "i": len(actions),
                        "tcp": _tcp_base_xyz(env),
                        "target": hover_target,
                        "rem": float(np.linalg.norm(hover_target - _tcp_base_xyz(env))),
                    },
                )
                return False

            pregrasp_z_target = target_pregrasp.copy()
            pregrasp_z_target[2] = target_pregrasp[2]
            obs, info = _move_to_base_target(
                env,
                pregrasp_z_target,
                gripper=-1.0,
                args=args,
                actions=actions,
                stage_name=f"{prefix}_approach_z",
                action_budget=min(_remaining_budget("approach_z"), _remaining_global()),
                debug_log=debug_log,
                color=color,
                attempt_index=attempt_index,
                rng=rng,
            )
            if _stage_failed() or not _near_target(target_pregrasp, max(0.020, 5.0 * float(args.pos_tol_m))):
                return False

            descend_z_target = target_descend.copy()
            descend_z_target[2] = target_descend[2]
            obs, info = _move_to_base_target(
                env,
                descend_z_target,
                gripper=-1.0,
                args=args,
                actions=actions,
                stage_name=f"{prefix}_descend",
                action_budget=min(_remaining_budget("descend"), _remaining_global()),
                debug_log=debug_log,
                color=color,
                attempt_index=attempt_index,
                rng=rng,
            )
            if _stage_failed() or not _near_target(target_descend, max(0.020, 5.0 * float(args.pos_tol_m))):
                return False
            return True

        def _grasp_and_lift(prefix: str):
            nonlocal obs, info
            for grasp_step in range(min(max(1, int(args.grasp_hold_steps)), _remaining_global())):
                action = _make_action(np.zeros(3, dtype=np.float32), gripper=1.0)
                actions.append(action)
                obs, info = _log_action_step(
                    env,
                    action,
                    args=args,
                    actions=actions,
                    stage_name=f"{prefix}_grasp_hold",
                    target_base_xyz=_tcp_base_xyz(env),
                    step_idx=grasp_step,
                    debug_log=debug_log,
                    color=color,
                    attempt_index=attempt_index,
                )
            lift_base = _tcp_base_xyz(env)
            lift_base[2] += float(args.lift_delta_z)
            lift_start = _tcp_base_xyz(env)
            lift_waypoints = _lift_curve_waypoints(lift_start, lift_base, args=args, rng=rng)
            _write_action_debug(
                debug_log,
                {
                    "t": "lift_curve_plan",
                    "st": f"{prefix}_lift",
                    "i": len(actions),
                    "start": lift_start,
                    "target": lift_base,
                    "waypoints": lift_waypoints,
                },
            )
            obs, info = _follow_base_path(
                env,
                [lift_start, *lift_waypoints],
                gripper=1.0,
                args=args,
                actions=actions,
                stage_name=f"{prefix}_lift",
                action_budget=min(_remaining_budget("lift"), _remaining_global()),
                debug_log=debug_log,
                color=color,
                attempt_index=attempt_index,
                rng=rng,
            )
            if float(np.linalg.norm(lift_base - _tcp_base_xyz(env))) <= max(0.015, 3.0 * float(args.pos_tol_m)):
                for hold_step in range(min(max(0, int(args.final_hold_steps)), _remaining_global())):
                    action = _make_action(np.zeros(3, dtype=np.float32), gripper=1.0)
                    actions.append(action)
                    obs, info = _log_action_step(
                        env,
                        action,
                        args=args,
                        actions=actions,
                        stage_name=f"{prefix}_final_hold",
                        target_base_xyz=_tcp_base_xyz(env),
                        step_idx=hold_step,
                        debug_log=debug_log,
                        color=color,
                        attempt_index=attempt_index,
                    )

        obs = None
        info = {}
        approach_ok = _approach_to_pregrasp("initial", pregrasp_base, descend_base)
        if approach_ok:
            _grasp_and_lift("initial")

        recovery_idx = 0
        while (
            bool(args.recovery_enabled)
            and recovery_idx < int(args.recovery_max_attempts)
            and _remaining_global() > max(4, int(args.grasp_hold_steps))
            and approach_ok
            and not _info_bool(info, "success")
            and not _info_bool(info, "is_src_obj_grasped")
        ):
            recovery_idx += 1
            rec_cube_xyz = _cube_world_xyz(env, color)
            rec_cube_base = _world_to_robot_base(env, rec_cube_xyz)
            rec_pregrasp_base = rec_cube_base + pregrasp_offset
            rec_descend_base = rec_cube_base + descend_offset
            rec_descend_base[2] = max(float(rec_descend_base[2]), min_descend_z)
            _write_action_debug(
                debug_log,
                {
                    "t": "recovery_start",
                    "idx": int(recovery_idx),
                    "i": len(actions),
                    "cube_w": rec_cube_xyz,
                    "cube_b": rec_cube_base,
                    "pre_b": rec_pregrasp_base,
                    "desc_b": rec_descend_base,
                    "info": info,
                },
            )
            approach_ok = _approach_to_pregrasp(f"recovery{recovery_idx}", rec_pregrasp_base, rec_descend_base)
            if approach_ok:
                _grasp_and_lift(f"recovery{recovery_idx}")
        success = _info_bool(info, "success")
        max_saved_actions = int(args.max_saved_demo_actions)
        if success and max_saved_actions > 0 and len(actions) > max_saved_actions:
            info = dict(info)
            info["_script_too_many_actions"] = {
                "actions": int(len(actions)),
                "max_saved_actions": int(max_saved_actions),
            }
            success = False
            _write_action_debug(
                debug_log,
                {
                    "t": "too_many_actions",
                    "actions": len(actions),
                    "max_saved_actions": int(max_saved_actions),
                },
            )
        _write_action_debug(
            debug_log,
            {
                "t": "end",
                "success": bool(success),
                "actions": len(actions),
                "tcp": _tcp_base_xyz(env),
                "q": _robot_joint_debug(env).get("qpos"),
                "lim": _robot_joint_debug(env).get("limit_margin"),
                "info": info,
            },
        )
        return success, info, np.asarray(actions, dtype=np.float32)
    finally:
        if debug_log is not None:
            debug_log.close()


def _write_session_meta(session_dir: Path, args: Args, colors: list[str]) -> None:
    meta_path = session_dir / "curve_motionplanning_session.json"
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "real2sim.record_airi_cubes_curve_demos",
        "task_mode": args.task_mode,
        "scene_asset_dir": args.scene_asset_dir,
        "load_state_file": args.load_state_file,
        "control_mode": "arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos",
        "obs_mode": "rgb+segmentation",
        "record_env_state": bool(args.record_env_state),
        "record_minimal_openvla_obs": bool(args.record_minimal_openvla_obs),
        "sim_freq_override": int(args.sim_freq_override),
        "control_freq_override": int(args.control_freq_override),
        "colors": colors,
        "planner_config_file": args.planner_config_file,
        "transform_planner_offsets_to_current_base": bool(args.transform_planner_offsets_to_current_base),
        "planner_offset_sweep_xyz_m": args.planner_offset_sweep_xyz_m,
        "planner_offset_sweep_successes_per_candidate": int(args.planner_offset_sweep_successes_per_candidate),
        "planner_offset_sweep_discard_on_failure": bool(args.planner_offset_sweep_discard_on_failure),
        "planner_offset_sweep_allow_incomplete": bool(args.planner_offset_sweep_allow_incomplete),
        "planner_offset_sweep_stop_after_pass": bool(args.planner_offset_sweep_stop_after_pass),
        "planner_backend": "proxy_ee_delta_recreated",
        "max_saved_demo_actions": int(args.max_saved_demo_actions),
        "xy_step_m": float(args.xy_step_m),
        "z_step_m": float(args.z_step_m),
        "action_translation_norm_m": float(args.action_translation_norm_m),
        "action_step_jitter_frac": float(args.action_step_jitter_frac),
        "action_direction_noise_m": float(args.action_direction_noise_m),
        "min_generated_translation_m": float(args.min_generated_translation_m),
        "lift_delta_z": float(args.lift_delta_z),
        "approach_curve_enabled": bool(args.approach_curve_enabled),
        "approach_curve_min_offset_m": float(args.approach_curve_min_offset_m),
        "approach_curve_max_offset_m": float(args.approach_curve_max_offset_m),
        "approach_curve_height_offset_m": float(args.approach_curve_height_offset_m),
        "approach_curve_waypoints": int(args.approach_curve_waypoints),
        "approach_curve_midpoint_min": float(args.approach_curve_midpoint_min),
        "approach_curve_midpoint_max": float(args.approach_curve_midpoint_max),
        "approach_curve_base_x_min_m": float(args.approach_curve_base_x_min_m),
        "approach_curve_base_x_max_m": float(args.approach_curve_base_x_max_m),
        "approach_curve_base_y_min_m": float(args.approach_curve_base_y_min_m),
        "approach_curve_base_y_max_m": float(args.approach_curve_base_y_max_m),
        "approach_curve_max_reverse_m": float(args.approach_curve_max_reverse_m),
        "approach_top_clearance_m": float(args.approach_top_clearance_m),
        "approach_correction_enabled": bool(args.approach_correction_enabled),
        "approach_miss_probability": float(args.approach_miss_probability),
        "approach_correction_max_offset_m": float(args.approach_correction_max_offset_m),
        "joint_limit_slow_margin_rad": float(args.joint_limit_slow_margin_rad),
        "joint_limit_stop_margin_rad": float(args.joint_limit_stop_margin_rad),
        "tcp_bad_response_ratio": float(args.tcp_bad_response_ratio),
        "lift_curve_max_offset_m": float(args.lift_curve_max_offset_m),
        "final_hold_steps": int(args.final_hold_steps),
        "recovery_enabled": bool(args.recovery_enabled),
        "recovery_max_attempts": int(args.recovery_max_attempts),
        "randomize_airi_cube_positions": bool(args.randomize_airi_cube_positions),
        "airi_cube_position_jitter_x": float(args.airi_cube_position_jitter_x),
        "airi_cube_position_jitter_y": float(args.airi_cube_position_jitter_y),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def main(args: Args) -> None:
    _run_parallel_workers(args)
    args.task_mode = "airi_cube_pickup"
    args.use_probe_objects = False
    args.trajectory_instruction = ""
    colors = _parse_color_choices(args.airi_cube_target_colors)
    if not colors:
        raise SystemExit("No AIRI cube target colors selected.")

    rng = np.random.default_rng(int(args.random_seed if args.random_seed else args.seed))
    session_name = args.resume_session_name.strip() or args.session_name or time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / session_name
    debug_dir = Path(args.debug_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    _update_last_session_debug_symlink(debug_dir)
    resume_saved = 0
    resume_attempted = 0
    trajectory_name = args.trajectory_name
    if args.resume_from_existing:
        resume_saved, resume_attempted = _infer_resume_counts(args, session_dir)
        trajectory_name = _next_trajectory_part_name(session_dir, args.trajectory_name)
        print(
            f"Resuming existing session: previous_saved={resume_saved} "
            f"previous_attempted={resume_attempted} new_trajectory={trajectory_name}"
        )
    _write_session_meta(session_dir, args, colors)
    print(f"Motionplanning session: {session_dir}")
    print(f"Debug dir: {debug_dir}")
    print(f"Targets: {colors} randomize={bool(args.randomize_airi_cube_target)}")
    print(
        "Recorder: "
        f"trajectory={args.trajectory_name} save_video={bool(args.save_video)} "
        f"video_fps={int(args.video_fps)} sim_freq={int(args.sim_freq_override)} "
        f"control_freq={int(args.control_freq_override)} "
        f"sim_backend={args.sim_backend} render_backend={args.render_backend_override or 'cpu'}"
    )
    if args.save_video:
        print(
            "Video recording enabled: "
            f"RecordEpisode will write videos under {session_dir} with prefix {args.trajectory_name}_<idx>_<color>; "
            f"demo_video_interval={max(1, int(args.demo_video_interval))}."
        )
    else:
        print("Video recording disabled.")

    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)

    if args.dry_run:
        print(f"session_dir={session_dir}")
        print(f"debug_dir={debug_dir}")
        print(f"colors={colors}")
        print(f"state_file={args.load_state_file}")
        print(f"planner_config_file={args.planner_config_file}")
        return

    env = _build_record_env(args, session_dir, trajectory_name)
    env._json_data["source_type"] = "motionplanning"
    env._json_data["source_desc"] = "AIRI cubes scripted proxy end-effector delta pickup demonstrations"
    env._json_data["planner_backend"] = "proxy_ee_delta_recreated"
    Path(env._json_path).write_text(json.dumps(env._json_data, indent=2), encoding="utf-8")
    saved = resume_saved
    attempted = resume_attempted
    collection_start_time = time.monotonic()
    sweep_offsets = _parse_offset_sweep(args.planner_offset_sweep_xyz_m)
    managed_sweep = bool(sweep_offsets) and (
        bool(args.planner_offset_sweep_discard_on_failure)
        or int(args.planner_offset_sweep_successes_per_candidate) > 1
    )
    sweep_index = 0
    sweep_candidate_successes = 0
    sweep_exhausted = False
    sweep_success_target = max(1, int(args.planner_offset_sweep_successes_per_candidate))
    try:
        max_attempts = int(args.max_attempts) if int(args.max_attempts) > 0 else max(1, int(args.max_demos))
        if args.resume_from_existing and int(args.max_attempts) <= 0:
            max_attempts += attempted
        while saved < int(args.max_demos) and attempted < max_attempts:
            if managed_sweep:
                if sweep_index >= len(sweep_offsets):
                    sweep_exhausted = True
                    print("Offset sweep exhausted all candidates.")
                    break
                setattr(args, "_planner_offset_sweep_index", sweep_index)
                print(
                    f"  sweep candidate {sweep_index + 1}/{len(sweep_offsets)} "
                    f"offset={sweep_offsets[sweep_index].tolist()} "
                    f"successes={sweep_candidate_successes}/{sweep_success_target}"
                )
            if args.randomize_airi_cube_target:
                color = str(rng.choice(colors))
            else:
                color = colors[attempted % len(colors)]
            attempted += 1
            attempt_start_time = time.monotonic()
            print(
                f"[attempt {attempted:04d}] target={color} saved={saved}/{args.max_demos} "
                f"eta={_remaining_time_text(collection_start_time, saved, int(args.max_demos))}"
            )
            debug_log_path = None
            if args.action_debug_log:
                debug_stem = Path(args.action_debug_log_file).stem
                debug_suffix = Path(args.action_debug_log_file).suffix or ".jsonl"
                debug_log_path = debug_dir / f"{debug_stem}_attempt_{attempted:04d}_{color}{debug_suffix}"
                if args.verbose:
                    print(f"  action debug log: {debug_log_path}")
            success, info, actions = _collect_one(
                env,
                args,
                color,
                robot_base_pose_p,
                robot_base_pose_q,
                robot_init_qpos,
                debug_log_path=debug_log_path,
                attempt_index=attempted,
            )
            if success:
                env.flush_trajectory(verbose=bool(args.verbose))
                save_demo_video = env.save_video and _should_save_demo_video(saved, args)
                if env.save_video:
                    if save_demo_video:
                        video_name = f"{args.trajectory_name}_{saved:04d}_{color}"
                        if args.verbose:
                            print(f"  flushing video name={video_name}")
                        env.flush_video(name=video_name, verbose=bool(args.verbose))
                    else:
                        env.flush_video(save=False)
                saved += 1
                print(
                    f"  success actions={len(actions)} saved={saved}/{args.max_demos} "
                    f"video={'saved' if save_demo_video else 'skipped'} "
                    f"attempt_time={_format_duration(time.monotonic() - attempt_start_time)} "
                    f"elapsed={_format_duration(time.monotonic() - collection_start_time)} "
                    f"eta={_remaining_time_text(collection_start_time, saved, int(args.max_demos))}"
                )
                if managed_sweep:
                    sweep_candidate_successes += 1
                    if sweep_candidate_successes >= sweep_success_target:
                        print(
                            f"  sweep candidate {sweep_index + 1}/{len(sweep_offsets)} passed "
                            f"{sweep_candidate_successes}/{sweep_success_target}."
                        )
                        if bool(args.planner_offset_sweep_stop_after_pass):
                            print("  stopping offset sweep after first passing candidate.")
                            sweep_exhausted = True
                            break
                        print("  moving on to next sweep candidate.")
                        sweep_index += 1
                        sweep_candidate_successes = 0
            else:
                env.flush_trajectory(save=False)
                if env.save_video:
                    if args.save_failed_videos:
                        video_name = f"{args.trajectory_name}_failed_attempt_{attempted:04d}_{color}"
                        if args.verbose:
                            print(f"  flushing failed-attempt video name={video_name}")
                        env.flush_video(name=video_name, verbose=bool(args.verbose))
                    else:
                        env.flush_video(save=False)
                status = (
                    f"  failed actions={len(actions)} saved={saved}/{args.max_demos} "
                    f"attempt_time={_format_duration(time.monotonic() - attempt_start_time)} "
                    f"elapsed={_format_duration(time.monotonic() - collection_start_time)} "
                    f"eta={_remaining_time_text(collection_start_time, saved, int(args.max_demos))}"
                )
                if args.verbose:
                    status += f" info_keys={sorted(info.keys())}"
                print(status)
                if managed_sweep and bool(args.planner_offset_sweep_discard_on_failure):
                    print(
                        f"  sweep candidate {sweep_index + 1}/{len(sweep_offsets)} failed after "
                        f"{sweep_candidate_successes}/{sweep_success_target} successes; discarding."
                    )
                    sweep_index += 1
                    sweep_candidate_successes = 0
    finally:
        env.close()
    print(f"Done: saved {saved} demos in {session_dir}")
    if saved < int(args.max_demos):
        if sweep_exhausted and bool(args.planner_offset_sweep_allow_incomplete):
            return
        raise SystemExit(f"Only saved {saved}/{args.max_demos} demos after {attempted} attempts.")


if __name__ == "__main__":
    main(tyro.cli(Args))
