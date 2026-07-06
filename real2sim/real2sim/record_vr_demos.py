import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import sapien
import torch
import tyro
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qinverse, qmult

from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers.record import RecordEpisode
from real2sim.calibrate_rc5_pose import (
    HAND_OPEN_STATE_FILE,
    _apply_camera_state,
    _apply_grid_pair_override,
    _build_env,
    _extract_obs_frame,
    _gripper_active_indices,
    _maybe_render_live,
    _probe_carrot_grasped,
    _read_single_key,
    _remap_sim_delta,
    _reset_current_state,
    _sanitize_live_viewer_environment,
    _scene_options,
    _sim_arm_controller,
    _state_file_gripper_qpos,
    _apply_sim_arm_target_qpos_direct,
    _load_spawn_grid_state,
    _make_rotated_grid,
    observation_camera_lookat_state,
)
from real2sim.openreal2sim_validation import (
    AIRI_CUBES_ACTION_TRANSLATION_CLAMP_M,
    AIRI_CUBES_ACTION_ROTATION_CLAMP_RAD,
    AIRI_CUBE_COLOR_NAMES,
    AIRI_CUBES_VR_ROTATION_GAIN,
    DEFAULT_PLATE_MODEL_NAME,
    DEFAULT_REAL2SIM_CONTROL_FREQ,
    DEFAULT_REAL2SIM_LANGUAGE_INSTRUCTION_TEMPLATE,
    DEFAULT_REAL2SIM_SIM_FREQ,
    DEFAULT_SOURCE_MODEL_NAME,
    MORE_PLATE_MODEL_DB,
    PROBE_CARROT_POSE_Q,
    PROBE_PLATE_POSE_Q,
    available_source_model_names,
    plate_xy_half_extent,
    resolve_language_instruction,
    source_xy_half_extent,
    source_model_names_for_obj_set,
)
from real2sim.calibrate_rc5_pose_vr import (
    Args as VRArgs,
    _camera_planar_basis,
    _extract_controller_pose_and_trigger,
    _load_quad_stream_writer,
    _load_vr_tracker_backend,
    _map_vr_translation_via_camera,
    _prepare_vr_stream_frame,
    _quat_xyzw_to_wxyz,
    _write_vr_stream_frame,
    _xr_quat_to_sim_wxyz,
)
from real2sim.debug_paths import HUMAN_DEMOS_DIR, VR_DEMOS_DEBUG_DIR

VR_GAIN_REFERENCE_SIM_FREQ = 200
VR_GAIN_REFERENCE_CONTROL_FREQ = 40


def _info_bool(info: dict, key: str) -> bool:
    value = info.get(key, False)
    if torch.is_tensor(value):
        return bool(value[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def _info_float(info: dict, key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    if torch.is_tensor(value):
        return float(value[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


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
        return {
            "p": np.asarray(value.p).tolist(),
            "q": np.asarray(value.q).tolist(),
        }
    return value


def _append_perf_log_line(handle, payload: dict[str, object]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(_to_jsonable(payload), separators=(",", ":")) + "\n")
    handle.flush()


def _round_nested(value, digits: int = 4):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _round_nested(value.tolist(), digits=digits)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, (int, bool, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _round_nested(v, digits=digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_nested(v, digits=digits) for v in value]
    return value


def _tensor_row(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim > 1:
        arr = arr[0]
    return arr


def _obs_tcp_pose(obs: dict):
    try:
        return _tensor_row(obs["extra"]["tcp_pose"])
    except Exception:
        return None


def _robot_qpos(env):
    try:
        return _tensor_row(env.unwrapped.agent.robot.get_qpos())
    except Exception:
        return None


def _arm_debug_state(env) -> dict[str, object]:
    arm_controller = _sim_arm_controller(env)
    if arm_controller is None:
        return {"present": False}
    cfg = getattr(arm_controller, "config", None)
    target_pose = getattr(arm_controller, "_target_pose", None)
    target_pose_raw = None if target_pose is None else getattr(target_pose, "raw_pose", None)
    active_joint_indices = getattr(arm_controller, "active_joint_indices", None)
    return {
        "present": True,
        "class": arm_controller.__class__.__name__,
        "frame": None if cfg is None else getattr(cfg, "frame", None),
        "use_delta": None if cfg is None else bool(getattr(cfg, "use_delta", False)),
        "use_target": None if cfg is None else bool(getattr(cfg, "use_target", False)),
        "normalize_action": None if cfg is None else bool(getattr(cfg, "normalize_action", False)),
        "active_joint_indices": _tensor_row(active_joint_indices),
        "start_qpos": _tensor_row(getattr(arm_controller, "_start_qpos", None)),
        "target_qpos": _tensor_row(getattr(arm_controller, "_target_qpos", None)),
        "target_pose": _tensor_row(target_pose_raw),
    }


def _write_control_debug_line(
    handle,
    *,
    digits: int,
    payload: dict[str, object],
) -> None:
    if handle is None:
        return
    handle.write(json.dumps(_round_nested(payload, digits=digits), separators=(",", ":")) + "\n")
    handle.flush()


def _scaled_vr_gain(base_gain: float, sim_freq: int, control_freq: int) -> float:
    del sim_freq
    return float(base_gain) * (float(control_freq) / float(VR_GAIN_REFERENCE_CONTROL_FREQ))


def _accumulate_timing(metrics: dict[str, object] | None, key: str, elapsed_ms: float) -> None:
    if metrics is None:
        return
    metrics[key] = float(metrics.get(key, 0.0)) + elapsed_ms
    count_key = f"{key}_count"
    metrics[count_key] = int(metrics.get(count_key, 0)) + 1


def _install_env_perf_hooks(env) -> None:
    unwrapped = env.unwrapped
    if bool(getattr(unwrapped, "_vr_perf_hooks_installed", False)):
        return

    unwrapped._vr_perf_current_step_metrics = None
    unwrapped._vr_perf_last_step_metrics = None
    unwrapped._vr_perf_step_index = 0

    def _wrap_call(target, attr_name: str, metric_name: str) -> None:
        original = getattr(target, attr_name)

        def wrapped(*args, **kwargs):
            metrics = getattr(unwrapped, "_vr_perf_current_step_metrics", None)
            if metrics is None:
                return original(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _accumulate_timing(metrics, metric_name, (time.perf_counter() - t0) * 1000.0)

        setattr(target, attr_name, wrapped)

    original_step = unwrapped.step

    def wrapped_step(action):
        step_index = int(getattr(unwrapped, "_vr_perf_step_index", 0))
        setattr(unwrapped, "_vr_perf_step_index", step_index + 1)
        metrics = {
            "step_index": step_index,
            "time_wall": time.time(),
        }
        setattr(unwrapped, "_vr_perf_current_step_metrics", metrics)
        t0 = time.perf_counter()
        try:
            return original_step(action)
        finally:
            metrics["step_total_ms"] = (time.perf_counter() - t0) * 1000.0
            setattr(unwrapped, "_vr_perf_last_step_metrics", metrics)
            setattr(unwrapped, "_vr_perf_current_step_metrics", None)

    setattr(unwrapped, "step", wrapped_step)
    _wrap_call(unwrapped, "_step_action", "base_step_action_ms")
    _wrap_call(unwrapped, "get_info", "get_info_ms")
    _wrap_call(unwrapped, "evaluate", "evaluate_ms")
    _wrap_call(unwrapped, "get_obs", "get_obs_ms")
    _wrap_call(unwrapped, "_get_obs_with_sensor_data", "get_obs_with_sensor_data_ms")
    _wrap_call(unwrapped, "_get_obs_sensor_data", "get_obs_sensor_data_ms")
    _wrap_call(unwrapped, "capture_sensor_data", "capture_sensor_data_ms")
    _wrap_call(unwrapped.scene, "step", "scene_step_ms")
    _wrap_call(unwrapped.scene, "update_render", "scene_update_render_ms")
    setattr(unwrapped, "_vr_perf_hooks_installed", True)


def _format_hud_lines(
    *,
    attempt_index: int,
    saved_count: int,
    steps: int,
    success_now: bool,
    stable_success: bool,
    grasped,
    trigger_value: float,
    squeeze_value: float,
    teleop_active: bool,
    carrot_plate_dist: float,
    tcp_to_carrot_dist: float,
    episode_frozen: bool,
    instruction_text: str = "",
    status_text: str = "",
) -> list[str]:
    instruction = f"Task: {instruction_text}" if instruction_text else ""
    controls = "squeeze=move  trigger=grip  A/X=save  B/Y=discard  stick=skip  r=recenter  x=exit"
    status = (
        f"attempt={attempt_index} saved={saved_count} steps={steps} "
        f"success={int(success_now)} stable={int(stable_success)} "
        f"grasped={grasped} frozen={int(episode_frozen)}"
    )
    metrics = (
        f"trigger={trigger_value:.2f} squeeze={squeeze_value:.2f} teleop={int(teleop_active)} "
        f"tcp_carrot={tcp_to_carrot_dist:.3f} carrot_plate={carrot_plate_dist:.3f}"
    )
    lines = [line for line in (instruction, controls, status, metrics) if line]
    if status_text:
        lines.append(status_text)
    return lines


def _draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = np.ascontiguousarray(frame.copy())
    h, w = image.shape[:2]
    pad = 10
    line_h = 26
    box_h = pad * 2 + line_h * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, min(h, box_h)), (0, 0, 0), thickness=-1)
    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0)
    y = pad + 18
    for line in lines:
        cv2.putText(
            image,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h
    return image


def _set_live_viewer_status(env, base_title: str, status_text: str = "") -> None:
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None:
        return
    title = base_title if not status_text else f"{base_title} | {status_text}"
    candidate_objects = [
        viewer,
        getattr(viewer, "window", None),
        getattr(getattr(viewer, "window", None), "window", None),
    ]
    candidate_methods = ("set_window_title", "set_title", "setTitle", "set_caption")
    for obj in candidate_objects:
        if obj is None:
            continue
        for method_name in candidate_methods:
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method(title)
                    return
                except Exception:
                    continue


@dataclass
class _TransientStatus:
    text: str = ""
    expires_at: float = 0.0


def _active_status_text(status: _TransientStatus) -> str:
    if not status.text:
        return ""
    if status.expires_at <= 0.0 or time.monotonic() <= status.expires_at:
        return status.text
    return ""


def _set_status(status: _TransientStatus, text: str, duration_sec: float) -> None:
    status.text = str(text)
    status.expires_at = time.monotonic() + max(0.0, float(duration_sec))


def _viewer_title_base(instruction_text: str) -> str:
    base = "OpenReal2Sim VR Demo Recorder"
    return base if not instruction_text else f"{base} | Task: {instruction_text}"


def _wrist_inset_bottom_right(args: VRArgs | None) -> bool:
    task_mode = str(getattr(args, "task_mode", "")).strip().lower()
    scene_asset_dir = str(getattr(args, "scene_asset_dir", "")).replace("\\", "/").lower()
    return task_mode in {"airi_cube_pickup", "cube_pickup", "pick_cube"} or "airi_cubes" in scene_asset_dir


def _write_hud_stream_frame(stream_writer, obs, hud_lines: list[str], args: VRArgs | None = None) -> None:
    if stream_writer is None:
        return
    frame = _extract_obs_frame(obs, wrist_inset_bottom_right=_wrist_inset_bottom_right(args))
    if bool(getattr(args, "vr_stream_postprocess", False)):
        frame = _prepare_vr_stream_frame(
            frame,
            contrast=float(getattr(args, "vr_stream_contrast", 1.22)),
            brightness=float(getattr(args, "vr_stream_brightness", -10.0)),
            gamma=float(getattr(args, "vr_stream_gamma", 0.88)),
            saturation=float(getattr(args, "vr_stream_saturation", 1.22)),
            red_gain=float(getattr(args, "vr_stream_red_gain", 1.08)),
            green_gain=float(getattr(args, "vr_stream_green_gain", 0.97)),
            blue_gain=float(getattr(args, "vr_stream_blue_gain", 1.00)),
        )
    stream_writer.write_rgb(_draw_hud(frame, hud_lines))


def _show_pc_status_window(
    window_name: str,
    obs,
    hud_lines: list[str],
    *,
    enabled: bool,
    args: VRArgs | None = None,
) -> None:
    if not enabled:
        return
    frame = _draw_hud(_extract_obs_frame(obs, wrist_inset_bottom_right=_wrist_inset_bottom_right(args)), hud_lines)
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame[:, :, ::-1])
        cv2.waitKey(1)
    except Exception:
        return


def _close_pc_status_window(window_name: str, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except Exception:
        return


def _stream_submit_due(now: float, last_submit_time: float, max_hz: float, *, force: bool = False) -> bool:
    if force or max_hz <= 0.0 or last_submit_time <= 0.0:
        return True
    return (now - last_submit_time) >= (1.0 / max_hz)


def _aggregate_action_chunk(actions: np.ndarray) -> np.ndarray:
    if len(actions) == 1:
        return actions[0].copy()
    out = np.zeros((7,), dtype=np.float32)
    out[:3] = actions[:, :3].sum(axis=0, dtype=np.float64)
    q_total = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for action in actions:
        q_step = euler2quat(
            float(action[3]),
            float(action[4]),
            float(action[5]),
            axes="sxyz",
        )
        q_total = qmult(q_step, q_total)
    out[3:6] = np.asarray(quat2euler(q_total, axes="sxyz"), dtype=np.float32)
    out[6] = float(actions[-1, 6])
    return out


def _viewer_eye_scaled_toward_target(
    eye: np.ndarray,
    target: np.ndarray,
    distance_scale: float,
) -> np.ndarray:
    scale = float(distance_scale)
    if scale <= 0.0:
        return np.array(eye, dtype=np.float32)
    return np.array(target, dtype=np.float32) + (
        np.array(eye, dtype=np.float32) - np.array(target, dtype=np.float32)
    ) * scale


def _apply_live_viewer_camera_override(
    env,
    args: "Args",
    camera_eye: np.ndarray,
    camera_target: np.ndarray,
    camera_roll_deg: float,
    camera_fov: float,
) -> None:
    if not bool(getattr(args, "live_viewer", False)):
        return
    viewer_eye = _viewer_eye_scaled_toward_target(
        camera_eye,
        camera_target,
        float(getattr(args, "live_viewer_distance_scale", 1.0)),
    )
    viewer_fov = float(getattr(args, "live_viewer_fov", 0.0))
    if viewer_fov <= 0.0:
        viewer_fov = float(camera_fov)
    _apply_camera_state(
        env,
        viewer_eye,
        np.array(camera_target, dtype=np.float32),
        float(camera_roll_deg),
        viewer_fov,
    )


def _allocate_segment_budgets(lengths: list[int], target_steps: int) -> list[int]:
    if sum(lengths) <= target_steps:
        return lengths.copy()
    alloc = [1] * len(lengths)
    remaining = target_steps - len(lengths)
    if remaining <= 0:
        return alloc
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    extra_capacity = np.maximum(lengths_arr - 1.0, 0.0)
    if float(extra_capacity.sum()) <= 0:
        return alloc
    ideal_extra = extra_capacity / extra_capacity.sum() * remaining
    floor_extra = np.floor(ideal_extra).astype(int)
    alloc = (np.asarray(alloc, dtype=int) + floor_extra).tolist()
    used = int(floor_extra.sum())
    leftovers = remaining - used
    if leftovers > 0:
        frac = ideal_extra - floor_extra
        order = np.argsort(-frac)
        for idx in order:
            if leftovers <= 0:
                break
            if alloc[int(idx)] < lengths[int(idx)]:
                alloc[int(idx)] += 1
                leftovers -= 1
    return alloc


def _compress_actions(actions: np.ndarray, target_steps: int, gripper_change_eps: float) -> np.ndarray:
    if len(actions) <= target_steps:
        return actions.copy()
    boundaries = [0]
    for idx in range(1, len(actions)):
        if abs(float(actions[idx, 6] - actions[idx - 1, 6])) >= gripper_change_eps:
            boundaries.append(idx)
    boundaries.append(len(actions))
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    if len(lengths) > target_steps:
        return actions.copy()
    budgets = _allocate_segment_budgets(lengths, target_steps)
    compressed: list[np.ndarray] = []
    for seg_idx, seg_len in enumerate(lengths):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]
        segment = actions[start:end]
        budget = min(max(1, budgets[seg_idx]), seg_len)
        if budget >= seg_len:
            compressed.extend(segment.copy())
            continue
        splits = np.linspace(0, seg_len, budget + 1)
        splits = np.round(splits).astype(int)
        splits[0] = 0
        splits[-1] = seg_len
        for chunk_idx in range(budget):
            chunk_start = int(splits[chunk_idx])
            chunk_end = int(splits[chunk_idx + 1])
            if chunk_end <= chunk_start:
                chunk_end = min(seg_len, chunk_start + 1)
            compressed.append(_aggregate_action_chunk(segment[chunk_start:chunk_end]))
    return np.asarray(compressed, dtype=np.float32)


def _downsampled_grayscale(image: np.ndarray, stride: int = 4) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    return gray[::stride, ::stride]


def _normalized_image_change(prev_image: np.ndarray, image: np.ndarray) -> float:
    prev_gray = _downsampled_grayscale(prev_image)
    gray = _downsampled_grayscale(image)
    return float(np.mean(np.abs(gray - prev_gray)) / 255.0)


def _small_action_image_filter_mask(
    actions: np.ndarray,
    images: np.ndarray,
    *,
    pos_thresh: float,
    rot_thresh: float,
    control_same_pos_thresh: float,
    control_same_rot_thresh: float,
    control_same_gripper_thresh: float,
    image_change_keep_thresh: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    images = np.asarray(images, dtype=np.uint8)
    valid_mask = np.zeros(actions.shape[0], dtype=bool)

    for idx, action in enumerate(actions):
        pos_movement = float(np.linalg.norm(action[:3]))
        rot_movement = float(np.linalg.norm(action[3:6]))
        gripper = float(action[6])

        is_valid = pos_movement > pos_thresh or rot_movement > rot_thresh
        if idx > 0 and actions[idx - 1][6] != gripper:
            is_valid = True

        if idx == 0:
            is_valid = True
        elif not is_valid:
            prev_action = actions[idx - 1]
            control_same = (
                float(np.linalg.norm(action[:3] - prev_action[:3])) <= control_same_pos_thresh
                and float(np.linalg.norm(action[3:6] - prev_action[3:6])) <= control_same_rot_thresh
                and abs(float(action[6] - prev_action[6])) <= control_same_gripper_thresh
            )
            if control_same:
                is_valid = _normalized_image_change(images[idx - 1], images[idx]) >= image_change_keep_thresh

        valid_mask[idx] = is_valid

    return valid_mask


def _filter_actions_by_replay_images(actions: np.ndarray, images: np.ndarray, args: "Args") -> tuple[np.ndarray, np.ndarray]:
    if not bool(args.filter_small_actions_before_save):
        return actions.copy(), np.ones(len(actions), dtype=bool)
    if len(actions) != len(images):
        raise ValueError(f"Expected one replay image per action, got {len(images)} images for {len(actions)} actions.")

    mask = _small_action_image_filter_mask(
        actions,
        images,
        pos_thresh=float(args.demo_step_translation_thresh_m),
        rot_thresh=float(args.demo_step_rotation_thresh_rad),
        control_same_pos_thresh=float(args.filter_control_same_pos_thresh),
        control_same_rot_thresh=float(args.filter_control_same_rot_thresh),
        control_same_gripper_thresh=float(args.filter_control_same_gripper_thresh),
        image_change_keep_thresh=float(args.filter_image_change_keep_thresh),
    )
    filtered = np.asarray(actions, dtype=np.float32)[mask]
    if len(filtered) == 0 and len(actions) > 0:
        mask[0] = True
        filtered = np.asarray(actions, dtype=np.float32)[mask]
    return filtered, mask


def _reset_record_env(
    env,
    args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    *,
    reconfigure: bool = False,
):
    record_args = dataclass_replace(
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
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
            "trajectory_instruction": _resolve_trajectory_instruction(args),
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


def _replay_and_optionally_save(
    env,
    args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    actions: np.ndarray,
    *,
    save: bool,
    video_name: str | None = None,
) -> tuple[bool, dict]:
    replay_robot_init_qpos = _with_open_gripper_qpos(env, robot_init_qpos)
    replay_reconfigure = (
        getattr(env, "_current_probe_source_model_name", None) != args.probe_source_model_name
        or getattr(env, "_current_probe_plate_model_name", None) != args.probe_plate_model_name
    )
    obs, info = _reset_record_env(
        env,
        args,
        robot_base_pose_p,
        robot_base_pose_q,
        replay_robot_init_qpos,
        reconfigure=replay_reconfigure,
    )
    env._current_probe_source_model_name = args.probe_source_model_name
    env._current_probe_plate_model_name = args.probe_plate_model_name
    final_info = info
    for action in actions:
        action_tensor = torch.tensor(
            action[np.newaxis, :],
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        obs, _, _, _, final_info = env.step(action_tensor)
    success = _info_bool(final_info, "success")
    if save and success:
        env.flush_trajectory(verbose=True)
        if env.save_video:
            env.flush_video(name=video_name, verbose=True)
    else:
        env.flush_trajectory(save=False)
        if env.save_video:
            env.flush_video(save=False)
    return success, final_info


def _replay_for_filter_images(
    env,
    args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, dict]:
    replay_robot_init_qpos = _with_open_gripper_qpos(env, robot_init_qpos)
    replay_reconfigure = (
        getattr(env, "_current_probe_source_model_name", None) != args.probe_source_model_name
        or getattr(env, "_current_probe_plate_model_name", None) != args.probe_plate_model_name
    )
    obs, info = _reset_record_env(
        env,
        args,
        robot_base_pose_p,
        robot_base_pose_q,
        replay_robot_init_qpos,
        reconfigure=replay_reconfigure,
    )
    env._current_probe_source_model_name = args.probe_source_model_name
    env._current_probe_plate_model_name = args.probe_plate_model_name
    final_info = info
    images: list[np.ndarray] = []
    for action in actions:
        images.append(_extract_obs_frame(obs, wrist_inset_bottom_right=_wrist_inset_bottom_right(args)))
        action_tensor = torch.tensor(
            action[np.newaxis, :],
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        obs, _, _, _, final_info = env.step(action_tensor)
    env.flush_trajectory(save=False)
    if env.save_video:
        env.flush_video(save=False)
    return np.asarray(images, dtype=np.uint8), final_info


def _build_record_env(args, session_dir: Path, trajectory_name: str):
    replay_sim_backend = args.sim_backend
    replay_render_backend = args.render_backend_override or "gpu"
    # Match the live demo backend override logic as closely as possible while
    # still disabling the interactive viewer for validation replay.
    if args.live_viewer and args.live_viewer_force_cpu:
        replay_render_backend = "cpu"
        if replay_sim_backend != "cpu":
            replay_sim_backend = "cpu"

    record_args = dataclass_replace(
        args,
        sim_backend=replay_sim_backend,
        render_backend_override=replay_render_backend,
        live_viewer=False,
        live_viewer_force_cpu=False,
        vr_stream_to_headset=False,
        vr_stream_max_hz=0.0,
        spawn_grid_state_file="",
        grid_pair_index=-1,
        show_debug_markers=False,
    )
    camera_mode = record_args.observation_camera_mode
    cam_state = observation_camera_lookat_state(camera_mode, scene_asset_dir=record_args.scene_asset_dir)
    env = _build_env(
        record_args,
        camera_mode=camera_mode,
        camera_eye=np.array(cam_state["eye"], dtype=np.float32),
        camera_target=np.array(cam_state["target"], dtype=np.float32),
        camera_roll_deg=float(cam_state.get("roll_deg", 0.0)),
        camera_fov=float(cam_state.get("fov", 1.0)),
    )
    record_env = RecordEpisode(
        env,
        output_dir=str(session_dir),
        save_trajectory=True,
        trajectory_name=trajectory_name,
        save_video=args.save_video,
        source_type="human",
        source_desc="Quest 3 VR teleoperation expert demonstrations",
        video_fps=args.teleop_video_fps,
        save_on_reset=False,
        recording_camera_name="3rd_view_camera",
        avoid_overwriting_video=True,
        max_steps_per_video=max(args.max_final_steps + 5, 100),
        clean_on_close=False,
    )
    record_env._json_data = _to_jsonable(record_env._json_data)
    record_env._json_data["replay_validation_sim_backend"] = record_args.sim_backend
    record_env._json_data["replay_validation_render_backend"] = record_args.render_backend_override
    dump_json(record_env._json_path, record_env._json_data, indent=2)
    from real2sim.calibrate_rc5_pose import _load_state_file
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
    _reset_record_env(
        record_env,
        args,
        robot_base_pose_p,
        robot_base_pose_q,
        _with_open_gripper_qpos(record_env, robot_init_qpos),
        reconfigure=True,
    )
    record_env._current_probe_source_model_name = args.probe_source_model_name
    record_env._current_probe_plate_model_name = args.probe_plate_model_name
    return record_env


def dataclass_replace(obj, **kwargs):
    values = {field.name: getattr(obj, field.name) for field in obj.__dataclass_fields__.values()}
    values.update(kwargs)
    return obj.__class__(**values)


def _parse_model_choices(spec: str, available: Sequence[str], *, label: str) -> list[str]:
    spec = spec.strip()
    if not spec:
        return []
    if spec.lower() in {"all", "*"}:
        return list(available)
    choices = []
    available_set = set(available)
    for raw_name in spec.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in available_set:
            raise SystemExit(
                f"Unknown {label} model {name!r}. Available examples: "
                + ", ".join(list(available)[:10])
            )
        choices.append(name)
    if not choices:
        raise SystemExit(f"No valid {label} models parsed from {spec!r}")
    return choices


def _is_cube_pickup_task(args: "Args") -> bool:
    return str(getattr(args, "task_mode", "probe_on_plate")).strip().lower() in {
        "airi_cube_pickup",
        "cube_pickup",
        "pick_cube",
    }


def _source_model_names_for_preset(preset: str) -> list[str]:
    normalized = preset.strip().lower()
    if not normalized:
        return []
    if normalized in {"all", "*"}:
        return available_source_model_names()
    if normalized in {"train", "test"}:
        return source_model_names_for_obj_set(normalized)
    raise SystemExit("--probe-source-model-preset must be one of: train, test, all")


def _parse_cube_color_choices(spec: str) -> list[str]:
    normalized = str(spec).strip().lower()
    if not normalized or normalized in {"all", "*"}:
        return list(AIRI_CUBE_COLOR_NAMES)
    colors = [part.strip().lower() for part in normalized.split(",") if part.strip()]
    unknown = sorted(set(colors) - set(AIRI_CUBE_COLOR_NAMES))
    if unknown:
        raise SystemExit(f"Unknown AIRI cube colors {unknown}. Available: {list(AIRI_CUBE_COLOR_NAMES)}")
    if not colors:
        raise SystemExit(f"No valid AIRI cube colors parsed from {spec!r}")
    return colors


def _resolve_trajectory_instruction(args: "Args") -> str:
    if _is_cube_pickup_task(args):
        return f"Pick up the {str(args.airi_cube_target_color).strip().lower()} cube."
    template = str(getattr(args, "trajectory_instruction", "")).strip()
    if not template:
        return ""
    model_name = str(getattr(args, "probe_source_model_name", "")).strip()
    plate_name = str(getattr(args, "probe_plate_model_name", "")).strip()
    try:
        return resolve_language_instruction(
            model_name,
            plate_model_name=plate_name,
            template=template,
        )
    except KeyError as exc:
        raise SystemExit(
            "--trajectory-instruction may only reference "
            "{object_name}, {model_id}, {model_name}, or {plate_model_name}; "
            f"got missing placeholder {exc.args[0]!r}"
        ) from exc


def _count_saved_episodes(session_dir: Path, trajectory_name_prefix: str) -> int:
    total = 0
    for json_path in sorted(session_dir.glob(f"{trajectory_name_prefix}*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        total += len(payload.get("episodes", []))
    return total


def _update_last_session_debug_symlink(debug_dir: Path) -> None:
    link_path = debug_dir.parent / "last_session_debug"
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(debug_dir.resolve(), target_is_directory=True)
    except OSError as exc:
        print(f"Warning: failed to update {link_path}: {exc}")


def _next_resume_shard_name(session_dir: Path, trajectory_name: str) -> str:
    base_h5 = session_dir / f"{trajectory_name}.h5"
    if not base_h5.exists():
        return trajectory_name
    shard_idx = 1
    while True:
        candidate = f"{trajectory_name}_resume_{shard_idx:03d}"
        if not (session_dir / f"{candidate}.h5").exists():
            return candidate
        shard_idx += 1


def _next_resume_debug_log_name(debug_dir: Path, perf_debug_file: str) -> str:
    base = Path(perf_debug_file)
    if not (debug_dir / base.name).exists():
        return base.name
    stem = base.stem
    suffix = base.suffix
    shard_idx = 1
    while True:
        candidate = f"{stem}_resume_{shard_idx:03d}{suffix}"
        if not (debug_dir / candidate).exists():
            return candidate
        shard_idx += 1


def _reconstruct_resume_setup(
    args: "Args",
    rng: np.random.Generator,
    robot_base_pose_p: np.ndarray,
    saved_count: int,
) -> dict[str, object]:
    setup = _choose_attempt_setup(args, rng, robot_base_pose_p)
    for _ in range(max(0, int(saved_count))):
        setup = _next_setup_after_success(args, rng, robot_base_pose_p, setup)
    return setup


def _choose_probe_models(args: "Args", rng: np.random.Generator) -> tuple[str, str]:
    if args.randomize_probe_source_model:
        source_name = str(rng.choice(args._probe_source_model_choices))
    else:
        source_name = args._probe_source_model_choices[0]
    if args.randomize_probe_plate_model:
        plate_name = str(rng.choice(args._probe_plate_model_choices))
    else:
        plate_name = args._probe_plate_model_choices[0]
    return source_name, plate_name


def _set_probe_models(args: "Args", source_name: str, plate_name: str) -> None:
    args.probe_source_model_name = source_name
    args.probe_plate_model_name = plate_name


def _valid_grid_pairs(
    args: "Args",
    robot_base_pose_p: np.ndarray,
    source_model_name: str,
    plate_model_name: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grid = _load_spawn_grid_state(args.spawn_grid_state_file, args.scene_asset_dir)
    center_xy = np.array(grid["center_xy"], dtype=np.float32)
    size_xy = np.array(grid["size_xy"], dtype=np.float32)
    yaw_deg = float(grid["yaw_deg"][0])
    footprint_margin = max(
        source_xy_half_extent(PROBE_CARROT_POSE_Q, model_name=source_model_name),
        plate_xy_half_extent(PROBE_PLATE_POSE_Q, model_name=plate_model_name),
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
    for source_xy in grid_points:
        for plate_xy in grid_points:
            if np.allclose(source_xy, plate_xy):
                continue
            if np.linalg.norm(plate_xy - source_xy) <= float(args.grid_min_pair_distance_xy):
                continue
            if np.linalg.norm(source_xy - robot_base_xy) <= float(args.grid_min_robot_clearance_xy):
                continue
            if np.linalg.norm(plate_xy - robot_base_xy) <= float(args.grid_min_robot_clearance_xy):
                continue
            pairs.append((source_xy.copy(), plate_xy.copy()))
    return pairs


def _choose_attempt_setup(
    args: "Args",
    rng: np.random.Generator,
    robot_base_pose_p: np.ndarray,
) -> dict[str, object]:
    if _is_cube_pickup_task(args):
        if args.randomize_airi_cube_target:
            target_color = str(rng.choice(args._airi_cube_target_choices))
        else:
            target_color = args._airi_cube_target_choices[0]
        return {
            "probe_source_model_name": target_color,
            "probe_plate_model_name": "none",
            "airi_cube_target_color": target_color,
            "object_position_xyz": tuple(float(x) for x in args.object_position_xyz),
            "plate_position_xyz": tuple(float(x) for x in args.plate_position_xyz),
            "grid_pair_index": int(args.grid_pair_index),
        }

    if args.spawn_grid_state_file:
        candidate_pairs: list[tuple[str, str, list[tuple[np.ndarray, np.ndarray]]]] = []
        for source_name in args._probe_source_model_choices:
            for plate_name in args._probe_plate_model_choices:
                pairs = _valid_grid_pairs(args, robot_base_pose_p, source_name, plate_name)
                if pairs:
                    candidate_pairs.append((source_name, plate_name, pairs))
        if not candidate_pairs:
            raise SystemExit(
                "No valid spawn-grid/object combinations available with the current constraints. "
                "Try fewer/larger-clearance-sensitive objects, a larger grid, or lower clearance thresholds."
            )

        if args.randomize_probe_source_model or args.randomize_probe_plate_model:
            source_name, plate_name, pairs = candidate_pairs[int(rng.integers(len(candidate_pairs)))]
        else:
            source_name = args._probe_source_model_choices[0]
            plate_name = args._probe_plate_model_choices[0]
            pairs = _valid_grid_pairs(args, robot_base_pose_p, source_name, plate_name)
            if not pairs:
                raise SystemExit(
                    f"Selected fixed model pair source={source_name!r}, plate={plate_name!r} "
                    "has no valid spawn-grid pairs under the current constraints."
                )

        grid_pair_index = args.grid_pair_index
        if args.randomize_grid_pair_per_attempt:
            grid_pair_index = int(rng.integers(len(pairs)))
        elif grid_pair_index < 0:
            grid_pair_index = 0
        if grid_pair_index >= len(pairs):
            raise SystemExit(
                f"grid_pair_index={grid_pair_index} out of range for {len(pairs)} valid pairs "
                f"for source={source_name!r}, plate={plate_name!r}"
            )
        source_xy, plate_xy = pairs[grid_pair_index]
        source_pos = np.array([float(source_xy[0]), float(source_xy[1]), 0.0], dtype=np.float32)
        plate_pos = np.array([float(plate_xy[0]), float(plate_xy[1]), 0.0], dtype=np.float32)
    else:
        source_name, plate_name = _choose_probe_models(args, rng)
        source_pos = np.array(args.object_position_xyz, dtype=np.float32)
        plate_pos = np.array(args.plate_position_xyz, dtype=np.float32)
        grid_pair_index = args.grid_pair_index

    return {
        "probe_source_model_name": source_name,
        "probe_plate_model_name": plate_name,
        "object_position_xyz": tuple(float(x) for x in source_pos.tolist()),
        "plate_position_xyz": tuple(float(x) for x in plate_pos.tolist()),
        "grid_pair_index": int(grid_pair_index),
    }


def _apply_attempt_setup(args: "Args", setup: dict[str, object]) -> None:
    args.probe_source_model_name = str(setup["probe_source_model_name"])
    args.probe_plate_model_name = str(setup["probe_plate_model_name"])
    if "airi_cube_target_color" in setup:
        args.airi_cube_target_color = str(setup["airi_cube_target_color"])
    args.object_position_xyz = tuple(setup["object_position_xyz"])
    args.plate_position_xyz = tuple(setup["plate_position_xyz"])
    args.grid_pair_index = int(setup["grid_pair_index"])


def _next_setup_after_success(
    args: "Args",
    rng: np.random.Generator,
    robot_base_pose_p: np.ndarray,
    current_setup: dict[str, object],
) -> dict[str, object]:
    if _is_cube_pickup_task(args):
        return _choose_attempt_setup(args, rng, robot_base_pose_p)

    if not args.spawn_grid_state_file:
        return _choose_attempt_setup(args, rng, robot_base_pose_p)

    if args.randomize_probe_source_model or args.randomize_probe_plate_model:
        return _choose_attempt_setup(args, rng, robot_base_pose_p)

    source_name = str(current_setup["probe_source_model_name"])
    plate_name = str(current_setup["probe_plate_model_name"])
    pairs = _valid_grid_pairs(args, robot_base_pose_p, source_name, plate_name)
    if not pairs:
        return _choose_attempt_setup(args, rng, robot_base_pose_p)

    if args.randomize_grid_pair_per_attempt:
        next_grid_pair_index = int(rng.integers(len(pairs)))
    else:
        current_grid_pair_index = int(current_setup.get("grid_pair_index", 0))
        next_grid_pair_index = (current_grid_pair_index + 1) % len(pairs)

    next_setup = {
        "probe_source_model_name": source_name,
        "probe_plate_model_name": plate_name,
        "object_position_xyz": (
            float(pairs[next_grid_pair_index][0][0]),
            float(pairs[next_grid_pair_index][0][1]),
            0.0,
        ),
        "plate_position_xyz": (
            float(pairs[next_grid_pair_index][1][0]),
            float(pairs[next_grid_pair_index][1][1]),
            0.0,
        ),
        "grid_pair_index": next_grid_pair_index,
    }
    return next_setup


@dataclass
class Args(VRArgs):
    vr_stream_max_hz: float = 45.0
    vr_translation_gain: float = 12.0
    vr_rotation_gain: float = 6.4
    vr_max_translation_step_m: float = 0.0
    vr_max_rotation_step_rad: float = 0.0
    vr_trigger_open: float = 0.30
    vr_trigger_close: float = 0.65
    vr_binary_grip: bool = True
    sim_freq_override: int = DEFAULT_REAL2SIM_SIM_FREQ
    control_freq_override: int = DEFAULT_REAL2SIM_CONTROL_FREQ
    output_dir: str = str(HUMAN_DEMOS_DIR)
    debug_dir: str = str(VR_DEMOS_DEBUG_DIR)
    vr_perf_debug: bool = True
    vr_perf_debug_file: str = "vr_loop_timing.jsonl"
    vr_perf_spike_threshold_ms: float = 150.0
    vr_control_debug: bool = False
    vr_control_debug_file: str = "vr_control_debug.jsonl"
    vr_control_debug_digits: int = 4
    resume_session_name: str = ""
    session_name: str = ""
    trajectory_name: str = "vr_human_demos"
    save_video: bool = True
    max_demos: int = 100
    max_final_steps: int = 80
    success_stable_steps: int = 3
    demo_step_translation_thresh_m: float = 0.005
    demo_step_rotation_thresh_rad: float = 0.03
    demo_step_gripper_thresh: float = 0.02
    demo_step_max_interval_sec: float = 1.0 / 60.0
    demo_idle_step_sec: float = 1.0 / 40.0
    demo_settle_window_sec: float = 1.0
    compression_gripper_change_eps: float = 0.20
    filter_small_actions_before_save: bool = True
    filter_control_same_pos_thresh: float = 0.0015
    filter_control_same_rot_thresh: float = 0.01
    filter_control_same_gripper_thresh: float = 0.02
    filter_image_change_keep_thresh: float = 0.015
    button_hand: str = "right"
    live_viewer_width: int = 1280
    live_viewer_height: int = 720
    live_viewer_fov: float = 0.70
    live_viewer_distance_scale: float = 0.55
    probe_source_model_preset: str = ""
    probe_source_model_names: str = DEFAULT_SOURCE_MODEL_NAME
    randomize_probe_source_model: bool = False
    probe_plate_model_names: str = DEFAULT_PLATE_MODEL_NAME
    randomize_probe_plate_model: bool = False
    randomize_grid_pair_per_attempt: bool = False
    task_mode: str = "probe_on_plate"
    airi_cube_target_colors: str = ",".join(AIRI_CUBE_COLOR_NAMES)
    randomize_airi_cube_target: bool = False
    trajectory_instruction: str = DEFAULT_REAL2SIM_LANGUAGE_INSTRUCTION_TEMPLATE
    pc_status_window: bool = False
    pc_status_window_name: str = "OpenReal2Sim Save Status"

    skip_count: int = 0


def main(args: Args):
    if args.button_hand not in {"left", "right"}:
        raise SystemExit("--button-hand must be 'left' or 'right'")
    if args.probe_source_model_preset.strip():
        args._probe_source_model_choices = _source_model_names_for_preset(args.probe_source_model_preset)
    else:
        args._probe_source_model_choices = _parse_model_choices(
            args.probe_source_model_names,
            available_source_model_names(),
            label="source",
        )
    args._probe_plate_model_choices = _parse_model_choices(
        args.probe_plate_model_names,
        sorted(MORE_PLATE_MODEL_DB.keys()),
        label="plate",
    )
    if args.randomize_probe_source_model and len(args._probe_source_model_choices) < 2:
        raise SystemExit("randomize_probe_source_model=True requires at least 2 source models")
    if args.randomize_probe_plate_model and len(args._probe_plate_model_choices) < 2:
        raise SystemExit("randomize_probe_plate_model=True requires at least 2 plate models")
    args._airi_cube_target_choices = _parse_cube_color_choices(args.airi_cube_target_colors)
    if args.randomize_airi_cube_target and len(args._airi_cube_target_choices) < 2:
        raise SystemExit("randomize_airi_cube_target=True requires at least 2 cube colors")
    if _is_cube_pickup_task(args):
        args.use_probe_objects = False
        args.trajectory_instruction = ""
        if float(args.vr_max_translation_step_m) <= 0.0:
            args.vr_max_translation_step_m = float(AIRI_CUBES_ACTION_TRANSLATION_CLAMP_M)
        if float(args.vr_max_rotation_step_rad) <= 0.0:
            args.vr_max_rotation_step_rad = float(AIRI_CUBES_ACTION_ROTATION_CLAMP_RAD)
        default_rotation_gain = type(args).__dataclass_fields__["vr_rotation_gain"].default
        if np.isclose(float(args.vr_rotation_gain), float(default_rotation_gain)):
            args.vr_rotation_gain = float(AIRI_CUBES_VR_ROTATION_GAIN)
    rng = np.random.default_rng(args.seed)
    _sanitize_live_viewer_environment(args)
    TrackerClass = _load_vr_tracker_backend(args.vr_tracker_backend)
    QuadFrameStreamWriter = _load_quad_stream_writer()

    resume_mode = bool(args.resume_session_name.strip())
    session_name = args.resume_session_name.strip() or args.session_name or time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / session_name
    debug_dir = Path(args.debug_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    _update_last_session_debug_symlink(debug_dir)

    from real2sim.calibrate_rc5_pose import _load_state_file
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
    initial_robot_init_qpos = robot_init_qpos.copy()
    saved_count = _count_saved_episodes(session_dir, args.trajectory_name) if resume_mode else 0
    
    print('RESUME MODE????', saved_count, args.skip_count)
    saved_count += args.skip_count
    if resume_mode:
        current_setup = _reconstruct_resume_setup(args, rng, robot_base_pose_p, saved_count)
        args.trajectory_name = _next_resume_shard_name(session_dir, args.trajectory_name)
        args.vr_perf_debug_file = _next_resume_debug_log_name(debug_dir, args.vr_perf_debug_file)
        args.vr_control_debug_file = _next_resume_debug_log_name(debug_dir, args.vr_control_debug_file)
    else:
        current_setup = _choose_attempt_setup(args, rng, robot_base_pose_p)
    _apply_attempt_setup(args, current_setup)
    initial_source_model_name = str(current_setup["probe_source_model_name"])
    initial_plate_model_name = str(current_setup["probe_plate_model_name"])

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
    if args.vr_perf_debug:
        _install_env_perf_hooks(env)
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = int(args.vr_max_episode_steps)
    if hasattr(env.unwrapped, "_max_episode_steps"):
        env.unwrapped._max_episode_steps = int(args.vr_max_episode_steps)
    record_env = _build_record_env(args, session_dir, args.trajectory_name)

    manifest_path = session_dir / "session_manifest.json"
    perf_log_path = debug_dir / args.vr_perf_debug_file
    manifest = {
        "session_name": session_name,
        "resume_session_name": args.resume_session_name,
        "resume_mode": resume_mode,
        "scene_asset_dir": args.scene_asset_dir,
        "load_state_file": args.load_state_file,
        "seed": args.seed,
        "grid_pair_index": args.grid_pair_index,
        "spawn_grid_state_file": args.spawn_grid_state_file,
        "max_final_steps": args.max_final_steps,
        "success_stable_steps": args.success_stable_steps,
        "probe_source_model_preset": args.probe_source_model_preset,
        "probe_source_model_names": args.probe_source_model_names,
        "randomize_probe_source_model": args.randomize_probe_source_model,
        "probe_plate_model_names": args.probe_plate_model_names,
        "randomize_probe_plate_model": args.randomize_probe_plate_model,
        "randomize_grid_pair_per_attempt": args.randomize_grid_pair_per_attempt,
        "trajectory_instruction_template": args.trajectory_instruction,
        "vr_perf_debug": args.vr_perf_debug,
        "vr_perf_debug_file": str(perf_log_path),
        "vr_control_debug": bool(args.vr_control_debug),
        "vr_control_debug_file": str(debug_dir / args.vr_control_debug_file),
        "vr_max_translation_step_m": float(args.vr_max_translation_step_m),
        "vr_max_rotation_step_rad": float(args.vr_max_rotation_step_rad),
        "vr_perf_spike_threshold_ms": args.vr_perf_spike_threshold_ms,
        "enable_shadow": bool(args.enable_shadow),
        "use_environment_map": bool(args.use_environment_map),
        "sim_freq_override": int(args.sim_freq_override),
        "control_freq_override": int(args.control_freq_override),
        "initial_probe_source_model_name": initial_source_model_name,
        "initial_probe_plate_model_name": initial_plate_model_name,
        "initial_object_position_xyz": list(args.object_position_xyz),
        "initial_plate_position_xyz": list(args.plate_position_xyz),
        "initial_grid_pair_index": args.grid_pair_index,
        "saved_count_at_start": saved_count,
        "trajectory_name_for_this_run": args.trajectory_name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    perf_log_handle = None
    control_debug_handle = None
    if args.vr_perf_debug:
        perf_log_path.write_text("", encoding="utf-8")
        perf_log_handle = perf_log_path.open("a", encoding="utf-8")
        _append_perf_log_line(
            perf_log_handle,
            {
                "event": "session_start",
                "time_wall": time.time(),
                "session_name": session_name,
                "resume_mode": resume_mode,
                "saved_count_at_start": saved_count,
                "perf_log_path": str(perf_log_path),
                "spawn_grid_state_file": args.spawn_grid_state_file,
                "grid_pair_index": args.grid_pair_index,
                "probe_source_model_preset": args.probe_source_model_preset,
                "live_viewer": bool(args.live_viewer),
                "vr_stream_to_headset": bool(args.vr_stream_to_headset),
                "vr_stream_max_hz": float(args.vr_stream_max_hz),
                "enable_shadow": bool(args.enable_shadow),
                "use_environment_map": bool(args.use_environment_map),
                "sim_freq_override": int(args.sim_freq_override),
                "control_freq_override": int(args.control_freq_override),
                "demo_step_translation_thresh_m": float(args.demo_step_translation_thresh_m),
                "demo_step_rotation_thresh_rad": float(args.demo_step_rotation_thresh_rad),
                "demo_step_gripper_thresh": float(args.demo_step_gripper_thresh),
                "demo_step_max_interval_sec": float(args.demo_step_max_interval_sec),
                "demo_idle_step_sec": float(args.demo_idle_step_sec),
            },
        )
        print(f"VR perf log: {perf_log_path}")
    if args.vr_control_debug:
        control_debug_path = debug_dir / args.vr_control_debug_file
        control_debug_path.write_text("", encoding="utf-8")
        control_debug_handle = control_debug_path.open("a", encoding="utf-8")
        _write_control_debug_line(
            control_debug_handle,
            digits=int(args.vr_control_debug_digits),
            payload={
                "event": "session_start",
                "time_wall": time.time(),
                "control_mode": "arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos",
                "sim_freq": int(args.sim_freq_override),
                "control_freq": int(args.control_freq_override),
                "vr_rotation_gain": float(args.vr_rotation_gain),
                "vr_max_translation_step_m": float(args.vr_max_translation_step_m),
                "vr_max_rotation_step_rad": float(args.vr_max_rotation_step_rad),
                "scene_asset_dir": args.scene_asset_dir,
                "task_mode": str(getattr(args, "task_mode", "")),
            },
        )
        print(f"VR control debug log: {control_debug_path}")

    attempt_index = 0
    current_source_model_name = initial_source_model_name
    current_plate_model_name = initial_plate_model_name
    current_instruction_text = _resolve_trajectory_instruction(args)
    viewer_title_base = _viewer_title_base(current_instruction_text)
    pc_status_window_name = str(args.pc_status_window_name)
    transient_status = _TransientStatus()
    obs, info = _reset_current_state(
        env,
        args,
        robot_base_pose_p,
        robot_base_pose_q,
        initial_robot_init_qpos.copy(),
        reconfigure=True,
    )
    _apply_live_viewer_camera_override(
        env,
        args,
        camera_eye,
        camera_target,
        camera_roll_deg,
        camera_fov,
    )

    stream_writer = (
        QuadFrameStreamWriter(args.vr_stream_path, max_hz=args.vr_stream_max_hz)
        if args.vr_tracker_backend == "playground" and args.vr_stream_to_headset
        else None
    )
    initial_hud_lines = _format_hud_lines(
        attempt_index=attempt_index,
        saved_count=saved_count,
        steps=0,
        success_now=False,
        stable_success=False,
        grasped=False,
        trigger_value=0.0,
        squeeze_value=0.0,
        teleop_active=False,
        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
        episode_frozen=False,
        instruction_text=current_instruction_text,
        status_text=_active_status_text(transient_status),
    )
    if stream_writer is not None:
        _write_hud_stream_frame(stream_writer, obs, initial_hud_lines, args)
        last_stream_submit_time = time.monotonic()
        print(f"VR stream initialized at {args.vr_stream_path} size={Path(args.vr_stream_path).stat().st_size if Path(args.vr_stream_path).exists() else -1}")
    _show_pc_status_window(pc_status_window_name, obs, initial_hud_lines, enabled=bool(args.pc_status_window), args=args)
    _set_live_viewer_status(env, viewer_title_base, _active_status_text(transient_status))

    tracker_kwargs = {}
    if args.vr_tracker_backend == "playground":
        tracker_kwargs["binary_path"] = args.vr_tracker_binary or None
        tracker_kwargs["stream_path"] = args.vr_stream_path if args.vr_stream_to_headset else None
    else:
        tracker_kwargs["prefer_aim_pose"] = args.vr_use_aim_pose

    print(f"Recording demos into {session_dir}")
    if resume_mode:
        print(
            f"Resuming session {session_name}: starting from saved_count={saved_count}, "
            f"writing new demos into {args.trajectory_name}.h5"
        )
    print("VR demo controls:")
    print("  squeeze hold: enable EE teleop motion")
    print("  trigger: gripper command")
    print("  A/X: save current demo if success is stable")
    print("  B/Y: discard current attempt and reset same initials")
    print("  thumbstick press: skip current setup and advance to the next one")
    print(f"  {args.vr_recenter_key}: recenter controller delta origin")
    print(f"  {args.vr_exit_key}: exit recorder")

    prev_pos = None
    prev_quat_wxyz = None
    anchor_pos = None
    anchor_quat_wxyz = None
    accumulated_delta_pos = np.zeros(3, dtype=np.float32)
    accumulated_delta_rot = np.zeros(3, dtype=np.float32)
    last_commit_time = time.monotonic()
    last_log_time = 0.0
    last_stream_submit_time = 0.0
    last_vr_frame_index = None
    last_vr_frame_time = None
    prev_primary_click = False
    prev_secondary_click = False
    prev_thumbstick_click = False
    prev_teleop_active = False
    last_gripper_cmd = -1.0
    gripper_closed = False
    success_streak = 0
    dense_actions: list[np.ndarray] = []
    pending_settle_until = time.monotonic()
    episode_frozen = False

    try:
        with TrackerClass(**tracker_kwargs) as tracker:
            for vr_frame in tracker.frames():
                loop_start = time.perf_counter()
                now_wall = time.time()
                render_live_ms = 0.0
                key_read_ms = 0.0
                controller_extract_ms = 0.0
                env_step_ms = 0.0
                ik_direct_ms = 0.0
                stream_submit_ms = 0.0
                success_eval_ms = 0.0
                step_breakdown = None
                spike_reasons: list[str] = []
                tracker_gap_s = 0.0 if last_vr_frame_time is None else (loop_start - last_vr_frame_time)
                tracker_frame_delta = (
                    None if last_vr_frame_index is None else int(vr_frame.frame_index - last_vr_frame_index)
                )
                last_vr_frame_time = loop_start
                last_vr_frame_index = int(vr_frame.frame_index)

                t0 = time.perf_counter()
                _maybe_render_live(env, args)
                _set_live_viewer_status(env, viewer_title_base, _active_status_text(transient_status))
                render_live_ms = (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                key = _read_single_key(timeout_sec=0.0)
                key_read_ms = (time.perf_counter() - t0) * 1000.0
                if key is not None:
                    key = key.lower()
                    if key == args.vr_exit_key:
                        _append_perf_log_line(
                            perf_log_handle,
                            {
                                "event": "exit_key",
                                "time_wall": now_wall,
                                "attempt_index": attempt_index,
                                "vr_frame_index": int(vr_frame.frame_index),
                            },
                        )
                        print("\nLeaving VR demo recorder")
                        break
                    if key == args.vr_recenter_key:
                        prev_pos = None
                        prev_quat_wxyz = None
                        anchor_pos = None
                        anchor_quat_wxyz = None
                        accumulated_delta_pos.fill(0.0)
                        accumulated_delta_rot.fill(0.0)
                        _append_perf_log_line(
                            perf_log_handle,
                            {
                                "event": "recenter_key",
                                "time_wall": now_wall,
                                "attempt_index": attempt_index,
                                "vr_frame_index": int(vr_frame.frame_index),
                            },
                        )
                        print("\nRecentered VR controller delta origin")

                controller = vr_frame.controllers[args.vr_hand]
                button_controller = vr_frame.controllers[args.button_hand]
                t0 = time.perf_counter()
                controller_pose, pose_source, trigger_value, squeeze_value = _extract_controller_pose_and_trigger(
                    controller,
                    use_aim_pose=args.vr_use_aim_pose,
                )
                controller_extract_ms = (time.perf_counter() - t0) * 1000.0
                teleop_active = squeeze_value >= float(args.vr_squeeze_teleop_threshold)
                if episode_frozen:
                    teleop_active = False
                teleop_rising_edge = teleop_active and not prev_teleop_active
                raw_grip_amount = float(np.clip(trigger_value, 0.0, 1.0))
                if bool(args.vr_binary_grip):
                    if raw_grip_amount >= float(args.vr_trigger_close):
                        gripper_closed = True
                    elif raw_grip_amount <= float(args.vr_trigger_open):
                        gripper_closed = False
                    gripper_cmd = 1.0 if gripper_closed else -1.0
                else:
                    gripper_cmd = 2.0 * raw_grip_amount - 1.0

                primary_click = bool(getattr(button_controller, "primary_click", False))
                secondary_click = bool(getattr(button_controller, "secondary_click", False))
                thumbstick_click = bool(getattr(button_controller, "thumbstick_click", False))
                save_edge = primary_click and not prev_primary_click
                discard_edge = secondary_click and not prev_secondary_click
                skip_edge = thumbstick_click and not prev_thumbstick_click
                prev_primary_click = primary_click
                prev_secondary_click = secondary_click
                prev_thumbstick_click = thumbstick_click

                if not controller_pose.is_valid:
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    _append_perf_log_line(
                        perf_log_handle,
                        {
                            "event": "frame",
                            "time_wall": now_wall,
                            "attempt_index": attempt_index,
                            "saved_count": saved_count,
                            "vr_frame_index": int(vr_frame.frame_index),
                            "tracker_frame_delta": tracker_frame_delta,
                            "tracker_gap_ms": tracker_gap_s * 1000.0,
                            "session_state": vr_frame.session_state,
                            "pose_valid": False,
                            "teleop_active": False,
                            "episode_frozen": bool(episode_frozen),
                            "steps": len(dense_actions),
                            "render_live_ms": render_live_ms,
                            "key_read_ms": key_read_ms,
                            "controller_extract_ms": controller_extract_ms,
                            "loop_ms": loop_ms,
                        },
                    )
                    prev_pos = None
                    prev_quat_wxyz = None
                    anchor_pos = None
                    anchor_quat_wxyz = None
                    prev_teleop_active = False
                    prev_primary_click = False
                    prev_secondary_click = False
                    prev_thumbstick_click = False
                    gripper_closed = False
                    last_gripper_cmd = -1.0
                    accumulated_delta_pos.fill(0.0)
                    accumulated_delta_rot.fill(0.0)
                    continue

                pos = np.asarray(controller_pose.position_xyz, dtype=np.float32)
                quat_wxyz = _quat_xyzw_to_wxyz(np.asarray(controller_pose.orientation_xyzw, dtype=np.float64))
                quat_sim_wxyz = _xr_quat_to_sim_wxyz(quat_wxyz)

                if prev_pos is None or prev_quat_wxyz is None:
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    _append_perf_log_line(
                        perf_log_handle,
                        {
                            "event": "frame",
                            "time_wall": now_wall,
                            "attempt_index": attempt_index,
                            "saved_count": saved_count,
                            "vr_frame_index": int(vr_frame.frame_index),
                            "tracker_frame_delta": tracker_frame_delta,
                            "tracker_gap_ms": tracker_gap_s * 1000.0,
                            "session_state": vr_frame.session_state,
                            "pose_source": pose_source,
                            "pose_valid": True,
                            "teleop_active": bool(teleop_active),
                            "episode_frozen": bool(episode_frozen),
                            "should_step": False,
                            "steps": len(dense_actions),
                            "render_live_ms": render_live_ms,
                            "key_read_ms": key_read_ms,
                            "controller_extract_ms": controller_extract_ms,
                            "loop_ms": loop_ms,
                            "spike_reasons": ["tracker_init"],
                        },
                    )
                    prev_pos = pos.copy()
                    prev_quat_wxyz = quat_sim_wxyz.copy()
                    anchor_pos = pos.copy()
                    anchor_quat_wxyz = quat_sim_wxyz.copy()
                    accumulated_delta_pos.fill(0.0)
                    accumulated_delta_rot.fill(0.0)
                    last_gripper_cmd = gripper_cmd
                    prev_teleop_active = teleop_active
                    continue

                if not teleop_active:
                    anchor_pos = pos.copy()
                    anchor_quat_wxyz = quat_sim_wxyz.copy()
                if teleop_rising_edge:
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    _append_perf_log_line(
                        perf_log_handle,
                        {
                            "event": "frame",
                            "time_wall": now_wall,
                            "attempt_index": attempt_index,
                            "saved_count": saved_count,
                            "vr_frame_index": int(vr_frame.frame_index),
                            "tracker_frame_delta": tracker_frame_delta,
                            "tracker_gap_ms": tracker_gap_s * 1000.0,
                            "session_state": vr_frame.session_state,
                            "pose_source": pose_source,
                            "pose_valid": True,
                            "teleop_active": bool(teleop_active),
                            "episode_frozen": bool(episode_frozen),
                            "should_step": False,
                            "steps": len(dense_actions),
                            "trigger_value": float(trigger_value),
                            "squeeze_value": float(squeeze_value),
                            "render_live_ms": render_live_ms,
                            "key_read_ms": key_read_ms,
                            "controller_extract_ms": controller_extract_ms,
                            "loop_ms": loop_ms,
                            "spike_reasons": ["teleop_rising_edge"],
                        },
                    )
                    anchor_pos = pos.copy()
                    anchor_quat_wxyz = quat_sim_wxyz.copy()
                    prev_pos = pos.copy()
                    prev_quat_wxyz = quat_sim_wxyz.copy()
                    accumulated_delta_pos.fill(0.0)
                    accumulated_delta_rot.fill(0.0)
                    prev_teleop_active = teleop_active
                    last_gripper_cmd = gripper_cmd
                    continue

                scaled_translation_gain = _scaled_vr_gain(
                    float(args.vr_translation_gain),
                    int(args.sim_freq_override),
                    int(args.control_freq_override),
                )
                scaled_rotation_gain = _scaled_vr_gain(
                    float(args.vr_rotation_gain),
                    int(args.sim_freq_override),
                    int(args.control_freq_override),
                )
                delta_pos = (pos - prev_pos) * scaled_translation_gain
                delta_quat = qmult(quat_sim_wxyz, qinverse(prev_quat_wxyz))
                delta_rot = np.array(quat2euler(delta_quat, axes="sxyz"), dtype=np.float32)
                delta_rot *= scaled_rotation_gain

                mapped_delta_pos = _map_vr_translation_via_camera(
                    delta_pos,
                    camera_right=camera_right,
                    camera_forward=camera_forward,
                )
                _, mapped_delta_rot = _remap_sim_delta(
                    [0.0, 0.0, 0.0],
                    delta_rot.tolist(),
                )
                if not teleop_active:
                    mapped_delta_pos = [0.0, 0.0, 0.0]
                    mapped_delta_rot = [0.0, 0.0, 0.0]
                else:
                    accumulated_delta_pos += np.asarray(mapped_delta_pos, dtype=np.float32)
                    accumulated_delta_rot += np.asarray(mapped_delta_rot, dtype=np.float32)

                translation_thresh = float(args.demo_step_translation_thresh_m)
                rotation_thresh = float(args.demo_step_rotation_thresh_rad)
                pos_mag = float(np.linalg.norm(accumulated_delta_pos))
                rot_mag = float(np.linalg.norm(accumulated_delta_rot))
                grip_delta = abs(gripper_cmd - last_gripper_cmd)
                now = time.monotonic()
                motion_step = teleop_active and (
                    pos_mag >= translation_thresh
                    or rot_mag >= rotation_thresh
                    or (now - last_commit_time >= float(args.demo_step_max_interval_sec) and (pos_mag > 1e-6 or rot_mag > 1e-6))
                )
                grip_step = grip_delta >= float(args.demo_step_gripper_thresh)

                should_step = False
                idle_settle_step = False
                if motion_step:
                    should_step = True
                if grip_step:
                    should_step = True
                if (not teleop_active) and (now - last_commit_time >= float(args.demo_idle_step_sec)) and now < pending_settle_until:
                    should_step = True
                    idle_settle_step = True

                if should_step:
                    unclamped_accumulated_delta_pos = accumulated_delta_pos.copy()
                    translation_clamp_scale = 1.0
                    max_translation_step = float(args.vr_max_translation_step_m)
                    if max_translation_step > 0.0:
                        accumulated_pos_mag = float(np.linalg.norm(accumulated_delta_pos))
                        if accumulated_pos_mag > max_translation_step:
                            translation_clamp_scale = max_translation_step / max(accumulated_pos_mag, 1e-8)
                            accumulated_delta_pos *= translation_clamp_scale
                            pos_mag = float(np.linalg.norm(accumulated_delta_pos))
                    unclamped_accumulated_delta_rot = accumulated_delta_rot.copy()
                    rotation_clamp_scale = 1.0
                    max_rotation_step = float(args.vr_max_rotation_step_rad)
                    if max_rotation_step > 0.0:
                        accumulated_rot_mag = float(np.linalg.norm(accumulated_delta_rot))
                        if accumulated_rot_mag > max_rotation_step:
                            rotation_clamp_scale = max_rotation_step / max(accumulated_rot_mag, 1e-8)
                            accumulated_delta_rot *= rotation_clamp_scale
                            rot_mag = float(np.linalg.norm(accumulated_delta_rot))
                    pre_step_tcp_pose = _obs_tcp_pose(obs)
                    pre_step_qpos = _robot_qpos(env)
                    pre_step_arm_state = _arm_debug_state(env)
                    action = np.array(
                        [
                            float(accumulated_delta_pos[0]),
                            float(accumulated_delta_pos[1]),
                            float(accumulated_delta_pos[2]),
                            float(accumulated_delta_rot[0]),
                            float(accumulated_delta_rot[1]),
                            float(accumulated_delta_rot[2]),
                            float(gripper_cmd),
                        ],
                        dtype=np.float32,
                    )
                    if idle_settle_step or (grip_step and not motion_step):
                        action[:6] = 0.0
                    action_tensor = torch.tensor(
                        action[np.newaxis, :],
                        dtype=torch.float32,
                        device=env.unwrapped.device,
                    )
                    t0 = time.perf_counter()
                    obs, _, terminated, truncated, info = env.step(action_tensor)
                    env_step_ms = (time.perf_counter() - t0) * 1000.0
                    post_step_tcp_pose = _obs_tcp_pose(obs)
                    post_step_qpos = _robot_qpos(env)
                    post_step_arm_state = _arm_debug_state(env)
                    step_breakdown = _to_jsonable(getattr(env.unwrapped, "_vr_perf_last_step_metrics", None))
                    post_direct_tcp_pose = None
                    post_direct_qpos = None
                    post_direct_arm_state = None
                    if args.sim_apply_ik_qpos_direct:
                        arm_controller = _sim_arm_controller(env)
                        target_qpos = None if arm_controller is None else getattr(arm_controller, "_target_qpos", None)
                        t0 = time.perf_counter()
                        obs, info = _apply_sim_arm_target_qpos_direct(env, target_qpos)
                        ik_direct_ms = (time.perf_counter() - t0) * 1000.0
                        post_direct_tcp_pose = _obs_tcp_pose(obs)
                        post_direct_qpos = _robot_qpos(env)
                        post_direct_arm_state = _arm_debug_state(env)
                    tcp_delta = None
                    if pre_step_tcp_pose is not None and _obs_tcp_pose(obs) is not None:
                        tcp_delta = np.asarray(_obs_tcp_pose(obs), dtype=np.float32) - np.asarray(
                            pre_step_tcp_pose, dtype=np.float32
                        )
                    qpos_delta = None
                    if pre_step_qpos is not None and _robot_qpos(env) is not None:
                        qpos_delta = np.asarray(_robot_qpos(env), dtype=np.float32) - np.asarray(
                            pre_step_qpos, dtype=np.float32
                        )
                    _write_control_debug_line(
                        control_debug_handle,
                        digits=int(args.vr_control_debug_digits),
                        payload={
                            "event": "control_step",
                            "time_wall": now_wall,
                            "attempt": attempt_index,
                            "saved": saved_count,
                            "vr_frame": int(vr_frame.frame_index),
                            "step": len(dense_actions),
                            "flags": {
                                "motion": bool(motion_step),
                                "grip": bool(grip_step),
                                "idle_settle": bool(idle_settle_step),
                                "teleop": bool(teleop_active),
                            },
                            "controller": {
                                "pose_source": pose_source,
                                "pos_xyz": pos,
                                "quat_wxyz": quat_sim_wxyz,
                                "trigger": float(trigger_value),
                                "squeeze": float(squeeze_value),
                                "raw_delta_pos": delta_pos,
                                "raw_delta_rot": delta_rot,
                                "mapped_delta_pos": mapped_delta_pos,
                                "mapped_delta_rot": mapped_delta_rot,
                                "accum_pos": accumulated_delta_pos,
                                "accum_pos_unclamped": unclamped_accumulated_delta_pos,
                                "translation_clamp_scale": translation_clamp_scale,
                                "accum_rot": accumulated_delta_rot,
                                "accum_rot_unclamped": unclamped_accumulated_delta_rot,
                                "rotation_clamp_scale": rotation_clamp_scale,
                                "pos_mag": pos_mag,
                                "rot_mag": rot_mag,
                            },
                            "action": action,
                            "tcp": {
                                "before": pre_step_tcp_pose,
                                "after_step": post_step_tcp_pose,
                                "after_direct": post_direct_tcp_pose,
                                "delta": tcp_delta,
                            },
                            "qpos": {
                                "before": pre_step_qpos,
                                "after_step": post_step_qpos,
                                "after_direct": post_direct_qpos,
                                "delta": qpos_delta,
                            },
                            "arm": {
                                "before": pre_step_arm_state,
                                "after_step": post_step_arm_state,
                                "after_direct": post_direct_arm_state,
                            },
                            "timing_ms": {
                                "env_step": env_step_ms,
                                "ik_direct": ik_direct_ms,
                            },
                            "truncated": bool(truncated[0].item()),
                            "terminated": bool(terminated[0].item()),
                        },
                    )
                    dense_actions.append(action.copy())
                    last_commit_time = now
                    if motion_step:
                        pending_settle_until = now + float(args.demo_settle_window_sec)
                        accumulated_delta_pos.fill(0.0)
                        accumulated_delta_rot.fill(0.0)
                    anchor_pos = pos.copy()
                    anchor_quat_wxyz = quat_sim_wxyz.copy()
                    last_gripper_cmd = gripper_cmd
                    if bool(truncated[0].item()):
                        episode_frozen = True
                        _set_status(transient_status, "ATTEMPT FROZEN: press A/X to save or B/Y to discard", 6.0)
                        print("Episode reached truncation/timeout. Attempt frozen; use A/X to save or B/Y to discard.")

                t0 = time.perf_counter()
                success_now = _info_bool(info, "success")
                success_streak = success_streak + 1 if success_now else 0
                stable_success = success_streak >= int(args.success_stable_steps)
                success_eval_ms = (time.perf_counter() - t0) * 1000.0

                if stream_writer is not None and _stream_submit_due(
                    now,
                    last_stream_submit_time,
                    float(args.vr_stream_max_hz),
                ):
                    t0 = time.perf_counter()
                    hud_lines = _format_hud_lines(
                        attempt_index=attempt_index,
                        saved_count=saved_count,
                        steps=len(dense_actions),
                        success_now=success_now,
                        stable_success=stable_success,
                        grasped=_probe_carrot_grasped(env),
                        trigger_value=trigger_value,
                        squeeze_value=squeeze_value,
                        teleop_active=teleop_active and not episode_frozen,
                        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
                        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
                        episode_frozen=episode_frozen,
                        instruction_text=current_instruction_text,
                        status_text=_active_status_text(transient_status),
                    )
                    _write_hud_stream_frame(stream_writer, obs, hud_lines, args)
                    _show_pc_status_window(pc_status_window_name, obs, hud_lines, enabled=bool(args.pc_status_window), args=args)
                    stream_submit_ms = (time.perf_counter() - t0) * 1000.0
                    last_stream_submit_time = now

                if now - last_log_time >= 1.0 / max(args.vr_log_hz, 0.1):
                    last_log_time = now
                    grasped = _probe_carrot_grasped(env)
                    tcp_xyz = obs["extra"]["tcp_pose"][0].detach().cpu().numpy().tolist()[:3]
                    print(
                        f"attempt={attempt_index} saved={saved_count} "
                        f"source_model={current_source_model_name} plate_model={current_plate_model_name} "
                        f"vr_frame={vr_frame.frame_index} session={vr_frame.session_state} "
                        f"profile={controller.interaction_profile} pose_source={pose_source} "
                        f"pose_valid={int(controller_pose.is_valid)} trigger={trigger_value:.3f} "
                        f"squeeze={squeeze_value:.3f} steps={len(dense_actions)} "
                        f"success={int(success_now)} stable_success={int(stable_success)} "
                        f"grasped={grasped} carrot_plate_dist={_info_float(info, 'carrot_to_plate_dist', -1.0):.3f} "
                        f"tcp={tcp_xyz}",
                        flush=True,
                    )

                if discard_edge:
                    _append_perf_log_line(
                        perf_log_handle,
                        {
                            "event": "discard_edge",
                            "time_wall": time.time(),
                            "attempt_index": attempt_index,
                            "saved_count": saved_count,
                            "vr_frame_index": int(vr_frame.frame_index),
                            "steps": len(dense_actions),
                        },
                    )
                    print(f"Discarding attempt {attempt_index} and resetting same initials.")
                    _set_status(transient_status, "ATTEMPT DISCARDED: resetting same initials", 3.0)
                    obs, info = _reset_current_state(
                        env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        initial_robot_init_qpos.copy(),
                        reconfigure=False,
                    )
                    _apply_live_viewer_camera_override(
                        env,
                        args,
                        camera_eye,
                        camera_target,
                        camera_roll_deg,
                        camera_fov,
                    )
                    reset_hud_lines = _format_hud_lines(
                        attempt_index=attempt_index + 1,
                        saved_count=saved_count,
                        steps=0,
                        success_now=False,
                        stable_success=False,
                        grasped=False,
                        trigger_value=0.0,
                        squeeze_value=0.0,
                        teleop_active=False,
                        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
                        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
                        episode_frozen=False,
                        instruction_text=current_instruction_text,
                        status_text=_active_status_text(transient_status),
                    )
                    if stream_writer is not None:
                        _write_hud_stream_frame(stream_writer, obs, reset_hud_lines, args)
                        last_stream_submit_time = time.monotonic()
                    _show_pc_status_window(pc_status_window_name, obs, reset_hud_lines, enabled=bool(args.pc_status_window), args=args)
                    dense_actions.clear()
                    success_streak = 0
                    attempt_index += 1
                    pending_settle_until = time.monotonic()
                    episode_frozen = False
                    prev_pos = None
                    prev_quat_wxyz = None
                    anchor_pos = None
                    anchor_quat_wxyz = None
                    prev_teleop_active = False
                    prev_primary_click = False
                    prev_secondary_click = False
                    prev_thumbstick_click = False
                    gripper_closed = False
                    last_gripper_cmd = -1.0
                    accumulated_delta_pos.fill(0.0)
                    accumulated_delta_rot.fill(0.0)
                    continue

                if skip_edge:
                    _append_perf_log_line(
                        perf_log_handle,
                        {
                            "event": "skip_edge",
                            "time_wall": time.time(),
                            "attempt_index": attempt_index,
                            "saved_count": saved_count,
                            "vr_frame_index": int(vr_frame.frame_index),
                            "steps": len(dense_actions),
                        },
                    )
                    next_setup = _next_setup_after_success(args, rng, robot_base_pose_p, current_setup)
                    next_source_model_name = str(next_setup["probe_source_model_name"])
                    next_plate_model_name = str(next_setup["probe_plate_model_name"])
                    print(
                        "Skipping current setup and advancing to "
                        f"grid_pair_index={int(next_setup['grid_pair_index'])} "
                        f"source_model={next_source_model_name} plate_model={next_plate_model_name}."
                    )
                    _set_status(transient_status, "SETUP SKIPPED: advancing to next setup", 3.0)
                    _apply_attempt_setup(args, next_setup)
                    obs, info = _reset_current_state(
                        env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        initial_robot_init_qpos.copy(),
                        reconfigure=(
                            next_source_model_name != current_source_model_name
                            or next_plate_model_name != current_plate_model_name
                        ),
                    )
                    _apply_live_viewer_camera_override(
                        env,
                        args,
                        camera_eye,
                        camera_target,
                        camera_roll_deg,
                        camera_fov,
                    )
                    current_source_model_name = next_source_model_name
                    current_plate_model_name = next_plate_model_name
                    current_setup = next_setup
                    current_instruction_text = _resolve_trajectory_instruction(args)
                    viewer_title_base = _viewer_title_base(current_instruction_text)
                    next_hud_lines = _format_hud_lines(
                        attempt_index=attempt_index + 1,
                        saved_count=saved_count,
                        steps=0,
                        success_now=False,
                        stable_success=False,
                        grasped=False,
                        trigger_value=0.0,
                        squeeze_value=0.0,
                        teleop_active=False,
                        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
                        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
                        episode_frozen=False,
                        instruction_text=current_instruction_text,
                        status_text=_active_status_text(transient_status),
                    )
                    if stream_writer is not None:
                        _write_hud_stream_frame(stream_writer, obs, next_hud_lines, args)
                        last_stream_submit_time = time.monotonic()
                    _show_pc_status_window(pc_status_window_name, obs, next_hud_lines, enabled=bool(args.pc_status_window), args=args)
                    dense_actions.clear()
                    success_streak = 0
                    attempt_index += 1
                    pending_settle_until = time.monotonic()
                    episode_frozen = False
                    prev_pos = None
                    prev_quat_wxyz = None
                    anchor_pos = None
                    anchor_quat_wxyz = None
                    prev_teleop_active = False
                    prev_primary_click = False
                    prev_secondary_click = False
                    prev_thumbstick_click = False
                    gripper_closed = False
                    last_gripper_cmd = -1.0
                    accumulated_delta_pos.fill(0.0)
                    accumulated_delta_rot.fill(0.0)
                    continue

                if save_edge:
                    _append_perf_log_line(
                        perf_log_handle,
                        {
                            "event": "save_edge",
                            "time_wall": time.time(),
                            "attempt_index": attempt_index,
                            "saved_count": saved_count,
                            "vr_frame_index": int(vr_frame.frame_index),
                            "steps": len(dense_actions),
                            "stable_success": bool(stable_success),
                        },
                    )
                    if not stable_success:
                        print("Save requested, but task success is not stable yet. Ignoring.")
                        _set_status(transient_status, "SAVE IGNORED: success not stable yet", 2.5)
                        continue
                    raw_actions = np.asarray(dense_actions, dtype=np.float32)
                    if len(raw_actions) == 0:
                        print("Save requested, but this attempt has no actions. Ignoring.")
                        _set_status(transient_status, "SAVE IGNORED: no actions recorded", 2.5)
                        continue
                    _set_status(transient_status, "SAVE REQUESTED: compressing actions...", 4.0)
                    _set_live_viewer_status(env, viewer_title_base, _active_status_text(transient_status))
                    _maybe_render_live(env, args)
                    save_hud_lines = _format_hud_lines(
                        attempt_index=attempt_index,
                        saved_count=saved_count,
                        steps=len(dense_actions),
                        success_now=success_now,
                        stable_success=stable_success,
                        grasped=_probe_carrot_grasped(env),
                        trigger_value=trigger_value,
                        squeeze_value=squeeze_value,
                        teleop_active=False,
                        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
                        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
                        episode_frozen=episode_frozen,
                        instruction_text=current_instruction_text,
                        status_text=_active_status_text(transient_status),
                    )
                    _show_pc_status_window(pc_status_window_name, obs, save_hud_lines, enabled=bool(args.pc_status_window), args=args)
                    final_actions = _compress_actions(
                        raw_actions,
                        target_steps=int(args.max_final_steps),
                        gripper_change_eps=float(args.compression_gripper_change_eps),
                    )
                    pre_filter_actions = final_actions
                    filter_mask = np.ones(len(pre_filter_actions), dtype=bool)
                    if bool(args.filter_small_actions_before_save):
                        print("Filtering compressed actions using replay image changes...")
                        _set_status(
                            transient_status,
                            f"FILTERING: replaying {len(pre_filter_actions)} compressed steps...",
                            8.0,
                        )
                        _set_live_viewer_status(env, viewer_title_base, _active_status_text(transient_status))
                        _maybe_render_live(env, args)
                        replay_images, _ = _replay_for_filter_images(
                            record_env,
                            args,
                            robot_base_pose_p,
                            robot_base_pose_q,
                            initial_robot_init_qpos.copy(),
                            pre_filter_actions,
                        )
                        final_actions, filter_mask = _filter_actions_by_replay_images(
                            pre_filter_actions,
                            replay_images,
                            args,
                        )
                        print(
                            f"Action filter kept {len(final_actions)}/{len(pre_filter_actions)} "
                            f"compressed steps before validation replay."
                        )
                    if len(final_actions) > int(args.max_final_steps):
                        print(
                            f"Compression failed: {len(raw_actions)} raw steps -> {len(final_actions)} final steps "
                            f"(limit {args.max_final_steps}). Redo the demo with fewer steps."
                        )
                        _set_status(
                            transient_status,
                            f"COMPRESSION FAILED: {len(raw_actions)} -> {len(final_actions)} steps (limit {int(args.max_final_steps)})",
                            6.0,
                        )
                        np.savez_compressed(
                            debug_dir / f"attempt_{attempt_index:04d}_compression_failed.npz",
                            raw_actions=raw_actions,
                            pre_filter_actions=pre_filter_actions,
                            final_actions=final_actions,
                            filter_mask=filter_mask,
                        )
                        continue
                    print("Saving demo: running compressed replay validation...")
                    _set_status(
                        transient_status,
                        f"REPLAY VALIDATION: {len(raw_actions)} -> {len(pre_filter_actions)} -> {len(final_actions)} steps, please wait...",
                        10.0,
                    )
                    _set_live_viewer_status(env, viewer_title_base, _active_status_text(transient_status))
                    _maybe_render_live(env, args)
                    replay_hud_lines = _format_hud_lines(
                        attempt_index=attempt_index,
                        saved_count=saved_count,
                        steps=len(dense_actions),
                        success_now=success_now,
                        stable_success=stable_success,
                        grasped=_probe_carrot_grasped(env),
                        trigger_value=trigger_value,
                        squeeze_value=squeeze_value,
                        teleop_active=False,
                        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
                        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
                        episode_frozen=episode_frozen,
                        instruction_text=current_instruction_text,
                        status_text=_active_status_text(transient_status),
                    )
                    _show_pc_status_window(pc_status_window_name, obs, replay_hud_lines, enabled=bool(args.pc_status_window), args=args)
                    success, replay_info = _replay_and_optionally_save(
                        record_env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        initial_robot_init_qpos.copy(),
                        final_actions,
                        save=True,
                        video_name=f"demo_{saved_count:04d}",
                    )
                    if not success:
                        print(
                            f"Replay validation failed for attempt {attempt_index}: "
                            f"{len(raw_actions)} raw -> {len(pre_filter_actions)} compressed -> "
                            f"{len(final_actions)} filtered steps. Demo not saved."
                        )
                        _set_status(
                            transient_status,
                            f"REPLAY FAILED: {len(raw_actions)} -> {len(pre_filter_actions)} -> {len(final_actions)} steps, demo not saved",
                            6.0,
                        )
                        np.savez_compressed(
                            debug_dir / f"attempt_{attempt_index:04d}_validation_failed.npz",
                            raw_actions=raw_actions,
                            pre_filter_actions=pre_filter_actions,
                            final_actions=final_actions,
                            filter_mask=filter_mask,
                            replay_success=np.array([success], dtype=bool),
                        )
                        continue
                    print(
                        f"Saved demo {saved_count} to {session_dir / (args.trajectory_name + '.h5')} "
                        f"({len(raw_actions)} raw -> {len(pre_filter_actions)} compressed -> "
                        f"{len(final_actions)} filtered steps, "
                        f"success={_info_bool(replay_info, 'success')}, "
                        f"source_model={current_source_model_name}, plate_model={current_plate_model_name})"
                    )
                    _set_status(
                        transient_status,
                        f"SAVED demo_{saved_count:04d}: {len(raw_actions)} -> {len(pre_filter_actions)} -> {len(final_actions)} steps",
                        6.0,
                    )
                    saved_count += 1
                    if saved_count >= int(args.max_demos):
                        print(f"Reached max_demos={args.max_demos}, exiting recorder.")
                        break
                    next_setup = _next_setup_after_success(args, rng, robot_base_pose_p, current_setup)
                    next_source_model_name = str(next_setup["probe_source_model_name"])
                    next_plate_model_name = str(next_setup["probe_plate_model_name"])
                    print(
                        "Demo saved. Resetting for the next attempt with "
                        f"grid_pair_index={int(next_setup['grid_pair_index'])} "
                        f"source_model={next_source_model_name} plate_model={next_plate_model_name}."
                    )
                    _apply_attempt_setup(args, next_setup)
                    obs, info = _reset_current_state(
                        env,
                        args,
                        robot_base_pose_p,
                        robot_base_pose_q,
                        initial_robot_init_qpos.copy(),
                        reconfigure=(
                            next_source_model_name != current_source_model_name
                            or next_plate_model_name != current_plate_model_name
                        ),
                    )
                    _apply_live_viewer_camera_override(
                        env,
                        args,
                        camera_eye,
                        camera_target,
                        camera_roll_deg,
                        camera_fov,
                    )
                    current_source_model_name = next_source_model_name
                    current_plate_model_name = next_plate_model_name
                    current_setup = next_setup
                    current_instruction_text = _resolve_trajectory_instruction(args)
                    viewer_title_base = _viewer_title_base(current_instruction_text)
                    next_hud_lines = _format_hud_lines(
                        attempt_index=attempt_index + 1,
                        saved_count=saved_count,
                        steps=0,
                        success_now=False,
                        stable_success=False,
                        grasped=False,
                        trigger_value=0.0,
                        squeeze_value=0.0,
                        teleop_active=False,
                        carrot_plate_dist=_info_float(info, "carrot_to_plate_dist", -1.0),
                        tcp_to_carrot_dist=_info_float(info, "tcp_to_carrot_dist", -1.0),
                        episode_frozen=False,
                        instruction_text=current_instruction_text,
                        status_text=_active_status_text(transient_status),
                    )
                    if stream_writer is not None:
                        _write_hud_stream_frame(stream_writer, obs, next_hud_lines, args)
                        last_stream_submit_time = time.monotonic()
                    _show_pc_status_window(pc_status_window_name, obs, next_hud_lines, enabled=bool(args.pc_status_window), args=args)
                    dense_actions.clear()
                    success_streak = 0
                    attempt_index += 1
                    pending_settle_until = time.monotonic()
                    episode_frozen = False
                    prev_pos = None
                    prev_quat_wxyz = None
                    anchor_pos = None
                    anchor_quat_wxyz = None
                    prev_teleop_active = False
                    prev_primary_click = False
                    prev_secondary_click = False
                    prev_thumbstick_click = False
                    gripper_closed = False
                    last_gripper_cmd = -1.0

                loop_ms = (time.perf_counter() - loop_start) * 1000.0
                if tracker_gap_s >= 0.25:
                    spike_reasons.append("tracker_gap")
                if env_step_ms >= float(args.vr_perf_spike_threshold_ms):
                    spike_reasons.append("env_step")
                if ik_direct_ms >= float(args.vr_perf_spike_threshold_ms):
                    spike_reasons.append("ik_direct")
                if render_live_ms >= float(args.vr_perf_spike_threshold_ms):
                    spike_reasons.append("render_live")
                if stream_submit_ms >= float(args.vr_perf_spike_threshold_ms):
                    spike_reasons.append("stream_submit")
                if loop_ms >= float(args.vr_perf_spike_threshold_ms):
                    spike_reasons.append("loop_total")
                _append_perf_log_line(
                    perf_log_handle,
                    {
                        "event": "frame",
                        "time_wall": now_wall,
                        "attempt_index": attempt_index,
                        "saved_count": saved_count,
                        "vr_frame_index": int(vr_frame.frame_index),
                        "tracker_frame_delta": tracker_frame_delta,
                        "tracker_gap_ms": tracker_gap_s * 1000.0,
                        "session_state": vr_frame.session_state,
                        "pose_source": pose_source,
                        "pose_valid": bool(controller_pose.is_valid),
                        "teleop_active": bool(teleop_active),
                        "episode_frozen": bool(episode_frozen),
                        "should_step": bool(should_step),
                        "success_now": bool(success_now),
                        "stable_success": bool(stable_success),
                        "steps": len(dense_actions),
                        "trigger_value": float(trigger_value),
                        "squeeze_value": float(squeeze_value),
                        "pos_mag": pos_mag,
                        "rot_mag": rot_mag,
                        "grip_delta": grip_delta,
                        "render_live_ms": render_live_ms,
                        "key_read_ms": key_read_ms,
                        "controller_extract_ms": controller_extract_ms,
                        "env_step_ms": env_step_ms,
                        "ik_direct_ms": ik_direct_ms,
                        "stream_submit_ms": stream_submit_ms,
                        "success_eval_ms": success_eval_ms,
                        "loop_ms": loop_ms,
                        "step_breakdown": step_breakdown,
                        "spike_reasons": spike_reasons,
                    },
                )
                prev_pos = pos.copy()
                prev_quat_wxyz = quat_sim_wxyz.copy()
                prev_teleop_active = teleop_active
    finally:
        _close_pc_status_window(pc_status_window_name, enabled=bool(args.pc_status_window))
        if perf_log_handle is not None:
            _append_perf_log_line(
                perf_log_handle,
                {
                    "event": "session_end",
                    "time_wall": time.time(),
                    "attempt_index": attempt_index,
                    "saved_count": saved_count,
                },
            )
            perf_log_handle.close()
        if control_debug_handle is not None:
            _write_control_debug_line(
                control_debug_handle,
                digits=int(args.vr_control_debug_digits),
                payload={
                    "event": "session_end",
                    "time_wall": time.time(),
                    "attempt_index": attempt_index,
                    "saved_count": saved_count,
                },
            )
            control_debug_handle.close()
        if stream_writer is not None:
            stream_writer.close()
        record_env.close()
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
