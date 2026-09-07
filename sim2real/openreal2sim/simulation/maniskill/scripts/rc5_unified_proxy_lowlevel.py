from __future__ import annotations

from contextlib import nullcontext
import time
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from mani_skill.utils.structs.pose import to_sapien_pose
from openreal2sim.simulation.maniskill.planner_core.grasp_state import (
    get_planner_grasp_state,
    save_planner_grasp_state,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_execution import (
    is_ee_delta_control_mode,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_logging import (
    is_debug_enabled,
    logger,
)
from transforms3d.quaternions import qinverse, qmult

_Y = "\033[33m"
_R = "\033[0m"
_RC5_BATCH_FIRM_CLOSE_JOINTS = (
    "right_thumb_cmc_flex",
    "right_thumb_mcp",
    "right_thumb_ip",
    "right_index_mcp_flex",
    "right_index_pip",
    "right_index_dip",
)


def get_debug_planner_config(env_like):
    env_unwrapped = env_like.unwrapped if hasattr(env_like, "unwrapped") else env_like
    return getattr(env_unwrapped, "_debug_planner_config", {}) or {}


def get_required_proxy_adaptive_step_config(planner_cfg):
    missing_keys = [
        key
        for key in (
            "planner_proxy_adaptive_steps_enabled",
            "planner_proxy_threshold_xy_m",
            "planner_proxy_threshold_z_m",
            "planner_proxy_pos_tol_m",
        )
        if key not in planner_cfg
    ]
    if missing_keys:
        raise RuntimeError(
            "Missing required proxy adaptive-step config keys: "
            + ", ".join(sorted(missing_keys))
        )
    return (
        bool(planner_cfg["planner_proxy_adaptive_steps_enabled"]),
        float(planner_cfg["planner_proxy_threshold_xy_m"]),
        float(planner_cfg["planner_proxy_threshold_z_m"]),
        float(planner_cfg["planner_proxy_pos_tol_m"]),
    )


def get_required_planner_waypoints_config(planner_cfg):
    if "planner_waypoints" not in planner_cfg:
        raise RuntimeError("Missing required planner waypoint config key: planner_waypoints")
    raw_block = planner_cfg["planner_waypoints"]
    if not isinstance(raw_block, Mapping):
        raise RuntimeError(
            "planner_waypoints must be a mapping with explicit 'enabled' and 'points' fields."
        )
    if "enabled" not in raw_block:
        raise RuntimeError("planner_waypoints is missing required field: enabled")
    if "points" not in raw_block:
        raise RuntimeError("planner_waypoints is missing required field: points")
    enabled = raw_block["enabled"]
    if not isinstance(enabled, bool):
        raise RuntimeError(
            f"planner_waypoints.enabled must be a bool, got {type(enabled).__name__}."
        )
    raw_points = raw_block["points"]
    if not isinstance(raw_points, list):
        raise RuntimeError("planner_waypoints.points must be a list.")
    normalized_points = []
    for idx, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, Mapping):
            raise RuntimeError(
                f"planner_waypoints.points[{idx}] must be a mapping with 'id' and 'position'."
            )
        if "id" not in raw_point:
            raise RuntimeError(f"planner_waypoints.points[{idx}] is missing required field: id")
        if "position" not in raw_point:
            raise RuntimeError(f"planner_waypoints.points[{idx}] is missing required field: position")
        point_id = str(raw_point["id"]).strip()
        if not point_id:
            raise RuntimeError(f"planner_waypoints.points[{idx}].id must be a non-empty string.")
        position = raw_point["position"]
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)):
            raise RuntimeError(
                f"planner_waypoints.points[{idx}].position must be a numeric sequence of length 3."
            )
        if len(position) != 3:
            raise RuntimeError(
                f"planner_waypoints.points[{idx}].position must have length 3."
            )
        position_np = np.asarray(position, dtype=np.float32).reshape(-1)
        if position_np.shape[0] != 3:
            raise RuntimeError(
                f"planner_waypoints.points[{idx}].position must have length 3."
            )
        if not np.all(np.isfinite(position_np)):
            raise RuntimeError(
                f"planner_waypoints.points[{idx}].position contains non-finite values."
            )
        normalized_points.append(
            {
                "id": point_id,
                "position": [float(x) for x in position_np.tolist()],
            }
        )
    return enabled, normalized_points


def get_required_pregrasp_joint_guard_config(planner_cfg):
    if "planner_proxy_pregrasp_joint_guard" not in planner_cfg:
        raise RuntimeError(
            "Missing required planner pregrasp joint-guard config key: planner_proxy_pregrasp_joint_guard"
        )
    raw_block = planner_cfg["planner_proxy_pregrasp_joint_guard"]
    if not isinstance(raw_block, Mapping):
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard must be a mapping with explicit "
            "'enabled', 'run_align_stage', 'joint_targets', 'tolerance_rad', 'mismatch_policy', and "
            "'align_solver_overrides' fields."
        )
    for field_name in (
        "enabled",
        "run_align_stage",
        "joint_targets",
        "tolerance_rad",
        "mismatch_policy",
        "align_solver_overrides",
    ):
        if field_name not in raw_block:
            raise RuntimeError(
                f"planner_proxy_pregrasp_joint_guard is missing required field: {field_name}"
            )
    enabled = raw_block["enabled"]
    if not isinstance(enabled, bool):
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.enabled must be a bool, "
            f"got {type(enabled).__name__}."
        )
    run_align_stage = raw_block["run_align_stage"]
    if not isinstance(run_align_stage, bool):
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.run_align_stage must be a bool, "
            f"got {type(run_align_stage).__name__}."
        )
    joint_targets_raw = raw_block["joint_targets"]
    if not isinstance(joint_targets_raw, Mapping):
        raise RuntimeError("planner_proxy_pregrasp_joint_guard.joint_targets must be a mapping.")
    normalized_targets = {}
    for raw_joint_name, raw_joint_value in joint_targets_raw.items():
        joint_name = str(raw_joint_name).strip()
        if not joint_name:
            raise RuntimeError(
                "planner_proxy_pregrasp_joint_guard.joint_targets contains an empty joint name."
            )
        try:
            joint_value = float(raw_joint_value)
        except (TypeError, ValueError):
            raise RuntimeError(
                "planner_proxy_pregrasp_joint_guard.joint_targets values must be finite numbers."
            ) from None
        if not np.isfinite(joint_value):
            raise RuntimeError(
                "planner_proxy_pregrasp_joint_guard.joint_targets values must be finite numbers."
            )
        normalized_targets[joint_name] = float(joint_value)
    tolerance_rad = float(raw_block["tolerance_rad"])
    if not np.isfinite(tolerance_rad) or tolerance_rad < 0.0:
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.tolerance_rad must be a finite non-negative number."
        )
    mismatch_policy = str(raw_block["mismatch_policy"]).strip().lower()
    if mismatch_policy not in {"fail_fast", "warn"}:
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.mismatch_policy must be 'fail_fast' or 'warn', "
            f"got {raw_block['mismatch_policy']!r}."
        )
    align_solver_overrides_raw = raw_block["align_solver_overrides"]
    if not isinstance(align_solver_overrides_raw, Mapping):
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.align_solver_overrides must be a mapping."
        )
    for field_name in ("posture_gain", "posture_gain_near_target", "posture_joint_weights"):
        if field_name not in align_solver_overrides_raw:
            raise RuntimeError(
                "planner_proxy_pregrasp_joint_guard.align_solver_overrides is missing required field: "
                f"{field_name}"
            )
    posture_gain = float(align_solver_overrides_raw["posture_gain"])
    if not np.isfinite(posture_gain) or posture_gain < 0.0:
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.align_solver_overrides.posture_gain must be a finite non-negative number."
        )
    posture_gain_near_target = float(align_solver_overrides_raw["posture_gain_near_target"])
    if not np.isfinite(posture_gain_near_target) or posture_gain_near_target < 0.0:
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.align_solver_overrides.posture_gain_near_target must be a finite non-negative number."
        )
    posture_joint_weights = np.asarray(
        align_solver_overrides_raw["posture_joint_weights"],
        dtype=np.float32,
    ).reshape(-1)
    if posture_joint_weights.shape[0] == 0:
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.align_solver_overrides.posture_joint_weights must be a non-empty numeric sequence."
        )
    if not np.all(np.isfinite(posture_joint_weights)):
        raise RuntimeError(
            "planner_proxy_pregrasp_joint_guard.align_solver_overrides.posture_joint_weights contains non-finite values."
        )
    return {
        "enabled": bool(enabled),
        "run_align_stage": bool(run_align_stage),
        "joint_targets": normalized_targets,
        "tolerance_rad": float(tolerance_rad),
        "mismatch_policy": mismatch_policy,
        "align_solver_overrides": {
            "posture_gain": float(posture_gain),
            "posture_gain_near_target": float(posture_gain_near_target),
            "posture_joint_weights": [float(item) for item in posture_joint_weights.tolist()],
        },
    }


def get_adaptive_proxy_xy_step(planner_cfg, *, xy_err: float, nominal_xy_step: float) -> float:
    adaptive_enabled, threshold_xy, _threshold_z, pos_tol = get_required_proxy_adaptive_step_config(planner_cfg)
    if adaptive_enabled and float(xy_err) <= float(threshold_xy):
        return float(min(float(nominal_xy_step), float(pos_tol)))
    return float(nominal_xy_step)


def get_adaptive_proxy_z_step_near_target(
    planner_cfg,
    *,
    tcp_z: float,
    descend_target_z: float,
    nominal_z_step: float,
) -> float:
    adaptive_enabled, _threshold_xy, threshold_z, pos_tol = get_required_proxy_adaptive_step_config(planner_cfg)
    if adaptive_enabled and float(tcp_z) <= float(descend_target_z + threshold_z):
        return float(min(float(nominal_z_step), float(pos_tol)))
    return float(nominal_z_step)


def _normalize_per_env_bool_list(value, *, num_envs: int):
    if value is None:
        return [False] * int(num_envs)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), int(num_envs), axis=0)
    else:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] == 1 and int(num_envs) > 1:
            arr = np.repeat(arr, int(num_envs), axis=0)
        if arr.shape[0] != int(num_envs):
            raise ValueError(
                f"Expected per-env bool payload with first dimension {int(num_envs)}, got shape={arr.shape}."
            )
        arr = arr[:, 0]
    return [bool(item) for item in arr.tolist()]


def _get_actor_position_rows(actor, *, num_envs: int):
    pose = getattr(actor, "pose", None)
    raw_pose = None if pose is None else getattr(pose, "raw_pose", None)
    if raw_pose is None:
        position = np.asarray(getattr(pose, "p", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(1, -1)[:, :3]
    else:
        if isinstance(raw_pose, torch.Tensor):
            raw_pose = raw_pose.detach().cpu().numpy()
        raw_pose = np.asarray(raw_pose, dtype=np.float32)
        if raw_pose.ndim == 1:
            raw_pose = raw_pose.reshape(1, -1)
        position = raw_pose[:, :3]
    if position.shape[0] == 1 and int(num_envs) > 1:
        position = np.repeat(position, int(num_envs), axis=0)
    if position.shape[0] != int(num_envs):
        raise ValueError(
            f"Expected per-env actor pose rows with first dimension {int(num_envs)}, got shape={position.shape}."
        )
    return np.asarray(position[:, :3], dtype=np.float32).copy()


def _get_robot_qpos_rows(robot, *, num_envs: int):
    qpos = robot.get_qpos()
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 1:
        qpos = qpos.reshape(1, -1)
    if qpos.shape[0] == 1 and int(num_envs) > 1:
        qpos = np.repeat(qpos, int(num_envs), axis=0)
    if qpos.shape[0] != int(num_envs):
        raise ValueError(
            f"Expected per-env robot qpos rows with first dimension {int(num_envs)}, got shape={qpos.shape}."
        )
    return np.asarray(qpos, dtype=np.float32).copy()


def _build_rc5_batch_firm_close_target_qpos(agent, base_close_qpos, *, alpha: float):
    alpha = float(alpha)
    if alpha <= 0.0:
        return None
    hand_joint_names = list(getattr(agent, "hand_joint_names", []) or [])
    if not hand_joint_names:
        return None
    robot = getattr(agent, "robot", None)
    if robot is None or not hasattr(robot, "get_active_joints"):
        return None
    joint_by_name = {joint.name: joint for joint in robot.get_active_joints()}
    hand_open_qpos = np.asarray(getattr(agent, "hand_open_qpos", []), dtype=np.float32).reshape(-1)
    target_qpos = np.asarray(base_close_qpos, dtype=np.float32).reshape(-1).copy()
    updated = False
    for joint_name in _RC5_BATCH_FIRM_CLOSE_JOINTS:
        if joint_name not in hand_joint_names:
            continue
        joint = joint_by_name.get(joint_name)
        if joint is None:
            continue
        hand_idx = hand_joint_names.index(joint_name)
        if hand_idx >= target_qpos.shape[0]:
            continue
        limits = joint.get_limits()
        lower = float(limits[0, 0].item() if hasattr(limits[0, 0], "item") else limits[0, 0])
        upper = float(limits[0, 1].item() if hasattr(limits[0, 1], "item") else limits[0, 1])
        open_q = float(hand_open_qpos[hand_idx]) if hand_idx < hand_open_qpos.shape[0] else float(target_qpos[hand_idx])
        close_q = float(target_qpos[hand_idx])
        closing_limit = upper if close_q >= open_q else lower
        firm_target = close_q + alpha * (closing_limit - close_q)
        target_qpos[hand_idx] = float(np.clip(firm_target, lower, upper))
        updated = True
    return target_qpos if updated else None


def _normalize_quat_np(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _select_first_env_pose(pose_like):
    raw_pose = getattr(pose_like, "raw_pose", None)
    if raw_pose is None:
        return pose_like
    if isinstance(raw_pose, torch.Tensor):
        if raw_pose.ndim <= 1:
            return pose_like
        from mani_skill.utils.structs.pose import Pose

        return Pose(raw_pose=raw_pose[0])
    raw_pose_np = np.asarray(raw_pose)
    if raw_pose_np.ndim <= 1:
        return pose_like
    from mani_skill.utils.structs.pose import Pose

    return Pose(raw_pose=torch.as_tensor(raw_pose_np[0], dtype=torch.float32))


def _to_sapien_pose_first_env(pose_like):
    return to_sapien_pose(_select_first_env_pose(pose_like))


def _pose_to_numpy_first_env(pose_like):
    pose = _to_sapien_pose_first_env(pose_like)
    return (
        np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3],
        np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4],
    )


def _pose_to_numpy_rows_local(pose_like, *, num_envs: int):
    if hasattr(pose_like, "p") and hasattr(pose_like, "q"):
        p_rows = pose_like.p
        q_rows = pose_like.q
        if isinstance(p_rows, torch.Tensor):
            p_rows = p_rows.detach().cpu().numpy()
        if isinstance(q_rows, torch.Tensor):
            q_rows = q_rows.detach().cpu().numpy()
        p_rows = np.asarray(p_rows, dtype=np.float32)
        q_rows = np.asarray(q_rows, dtype=np.float32)
        if p_rows.ndim == 1:
            p_rows = p_rows.reshape(1, -1)
        if q_rows.ndim == 1:
            q_rows = q_rows.reshape(1, -1)
    else:
        pose = to_sapien_pose(pose_like)
        p_rows = np.asarray(pose.p, dtype=np.float32)
        q_rows = np.asarray(pose.q, dtype=np.float32)
        if p_rows.ndim == 1:
            p_rows = p_rows.reshape(1, -1)
        if q_rows.ndim == 1:
            q_rows = q_rows.reshape(1, -1)
    if p_rows.shape[0] == 1 and int(num_envs) > 1:
        p_rows = np.repeat(p_rows, int(num_envs), axis=0)
    if q_rows.shape[0] == 1 and int(num_envs) > 1:
        q_rows = np.repeat(q_rows, int(num_envs), axis=0)
    if p_rows.shape[0] != int(num_envs) or q_rows.shape[0] != int(num_envs):
        raise ValueError(
            "Expected per-env pose rows to match num_envs: "
            f"p_shape={p_rows.shape} q_shape={q_rows.shape} num_envs={int(num_envs)}"
        )
    return (
        np.asarray(p_rows[:, :3], dtype=np.float32).copy(),
        np.asarray(q_rows[:, :4], dtype=np.float32).copy(),
    )


def compute_proxy_rotvec_step(current_q, target_q, max_step_rad: float):
    current_q = _normalize_quat_np(current_q)
    target_q = _normalize_quat_np(target_q)
    q_rel = _normalize_quat_np(qmult(target_q, qinverse(current_q)))
    if q_rel[0] < 0.0:
        q_rel = -q_rel
    w = float(np.clip(q_rel[0], -1.0, 1.0))
    angle = float(2.0 * np.arccos(w))
    if angle <= 1e-6:
        return np.zeros(3, dtype=np.float32), 0.0
    sin_half = float(np.sqrt(max(1.0 - w * w, 0.0)))
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64) if sin_half <= 1e-6 else np.asarray(q_rel[1:4], dtype=np.float64) / sin_half
    step_angle = min(float(max_step_rad), angle)
    return np.asarray(axis * step_angle, dtype=np.float32), angle


def _euler_rpy_deg_to_rotation_matrix(rpy_deg):
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _apply_teleop_delta_remap(delta_pos, delta_rpy, remap_rpy_deg):
    remap_rpy_deg = [0.0, 0.0, 0.0] if remap_rpy_deg is None else remap_rpy_deg
    rot = _euler_rpy_deg_to_rotation_matrix(remap_rpy_deg)
    delta_pos = np.asarray(delta_pos, dtype=np.float64)
    delta_rpy = np.asarray(delta_rpy, dtype=np.float64)
    if delta_pos.ndim == 1:
        remapped_pos = rot @ delta_pos
    else:
        remapped_pos = delta_pos @ rot.T
    if delta_rpy.ndim == 1:
        remapped_rpy = rot @ delta_rpy
    else:
        remapped_rpy = delta_rpy @ rot.T
    return remapped_pos.astype(np.float32), remapped_rpy.astype(np.float32)


def _map_teleop_gripper_signal_to_controller(signal_value, target_state):
    magnitude = abs(float(signal_value))
    if target_state == "open":
        return magnitude
    if target_state == "close":
        return -magnitude
    raise ValueError(f"Unsupported gripper target_state: {target_state}")


def _get_proxy_delta_remap_rpy_deg(env_unwrapped):
    planner_cfg = get_debug_planner_config(env_unwrapped)
    proxy_remap = planner_cfg.get("planner_proxy_delta_remap_rpy_deg")
    if proxy_remap is not None:
        return [float(x) for x in proxy_remap]
    agent_uid = str(getattr(env_unwrapped.agent, "uid", "") or "")
    if agent_uid.startswith("widowx") and agent_uid.endswith("_rl"):
        return [0.0, 0.0, -90.0]
    return [0.0, 0.0, 0.0]


def _rotation_matrix_from_pose(pose_like):
    if hasattr(pose_like, "to_transformation_matrix"):
        matrix = pose_like.to_transformation_matrix()
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim > 2:
            matrix = matrix.reshape((-1,) + matrix.shape[-2:])[0]
        return matrix[:3, :3]
    pose = _to_sapien_pose_first_env(pose_like)
    matrix = pose.to_transformation_matrix()
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim > 2:
        matrix = matrix.reshape((-1,) + matrix.shape[-2:])[0]
    return matrix[:3, :3]


def _transform_world_delta_to_robot_base(env_unwrapped, delta_pos, delta_rpy):
    robot = getattr(getattr(env_unwrapped, "agent", None), "robot", None)
    robot_pose = None if robot is None else getattr(robot, "pose", None)
    if robot_pose is None:
        raise RuntimeError("Proxy EE-delta backend requires env.agent.robot.pose to transform world-frame deltas.")
    world_from_base = _rotation_matrix_from_pose(robot_pose)
    base_from_world = world_from_base.T
    delta_pos = np.asarray(delta_pos, dtype=np.float64)
    delta_rpy = np.asarray(delta_rpy, dtype=np.float64)
    if delta_pos.ndim == 1:
        transformed_pos = base_from_world @ delta_pos.reshape(3)
    else:
        transformed_pos = delta_pos.reshape(-1, 3) @ base_from_world.T
    if delta_rpy.ndim == 1:
        transformed_rpy = base_from_world @ delta_rpy.reshape(3)
    else:
        transformed_rpy = delta_rpy.reshape(-1, 3) @ base_from_world.T
    return transformed_pos.astype(np.float32), transformed_rpy.astype(np.float32)


def _get_proxy_frame_mode(env_unwrapped) -> str:
    planner_cfg = get_debug_planner_config(env_unwrapped)
    return str(planner_cfg.get("planner_proxy_frame", "base_camera_plane") or "base_camera_plane")


def _refresh_viewer_during_proxy_step(env, viewer) -> None:
    if viewer is None:
        return
    try:
        env.render_human()
    except Exception:
        pass
    try:
        viewer.notify_render_update()
    except Exception:
        pass
    try:
        if getattr(viewer, "window", None) is not None:
            viewer.render()
    except Exception:
        pass


def viewer_key_pressed_once(env_unwrapped, viewer, key: str) -> bool:
    if viewer is None:
        return False
    window = getattr(viewer, "window", None)
    if window is None:
        return False
    key_states = getattr(env_unwrapped, "_debug_planner_last_key_states", None)
    if key_states is None:
        key_states = {}
        env_unwrapped._debug_planner_last_key_states = key_states
    try:
        is_pressed = bool(window.key_down(key))
    except Exception:
        is_pressed = False
    was_pressed = bool(key_states.get(key, False))
    key_states[key] = is_pressed
    return is_pressed and not was_pressed


def maybe_handle_proxy_viewer_video_hotkey(env, viewer) -> bool:
    if viewer is None:
        return False
    env_unwrapped = env.unwrapped
    if not viewer_key_pressed_once(env_unwrapped, viewer, "v"):
        return False
    save_video_buffer = getattr(env_unwrapped, "_debug_planner_save_video_buffer", None)
    if not callable(save_video_buffer):
        print(f"{_Y}[WARNING] [PlannerDebug] Viewer video save requested, but no save callback is configured.{_R}")
        return False
    try:
        return bool(save_video_buffer())
    except Exception as exc:
        print(
            f"{_Y}[WARNING] [PlannerDebug] Viewer video save failed and was ignored: "
            f"{type(exc).__name__}: {exc}{_R}"
        )
        return False


def wait_for_proxy_viewer_step_advance(env, viewer) -> bool:
    if viewer is None:
        return True
    env_unwrapped = env.unwrapped
    if not bool(getattr(env_unwrapped, "_debug_planner_step_by_step", False)):
        maybe_handle_proxy_viewer_video_hotkey(env, viewer)
        return True
    if not bool(getattr(env_unwrapped, "_debug_planner_viewer_step_gate_announced", False)):
        print("[PlannerDebug] Viewer step gate active. Press SPACE or 'n' to advance one proxy step.")
        env_unwrapped._debug_planner_viewer_step_gate_announced = True
    while viewer is not None and not getattr(viewer, "closed", False):
        _refresh_viewer_during_proxy_step(env, viewer)
        maybe_handle_proxy_viewer_video_hotkey(env, viewer)
        if viewer_key_pressed_once(env_unwrapped, viewer, " "):
            return True
        if viewer_key_pressed_once(env_unwrapped, viewer, "n"):
            return True
        time.sleep(0.01)
    return True


def _get_base_camera_pose_debug(env_unwrapped):
    sensor = getattr(getattr(env_unwrapped, "scene", None), "sensors", {}).get("base_camera")
    if sensor is not None:
        for candidate in (
            getattr(getattr(sensor, "camera", None), "get_pose", None),
            getattr(sensor, "get_pose", None),
        ):
            if callable(candidate):
                try:
                    return _to_sapien_pose_first_env(candidate())
                except Exception:
                    pass
        for candidate in (
            getattr(getattr(sensor, "camera", None), "pose", None),
            getattr(sensor, "pose", None),
        ):
            if candidate is not None:
                try:
                    return _to_sapien_pose_first_env(candidate)
                except Exception:
                    pass
    if hasattr(env_unwrapped, "_get_base_camera_pose_from_scene"):
        return _to_sapien_pose_first_env(env_unwrapped._get_base_camera_pose_from_scene())
    raise RuntimeError("Could not resolve base_camera pose for proxy frame alignment.")


def _get_base_camera_plane_axes(env_unwrapped):
    cache = getattr(env_unwrapped, "_planner_proxy_base_camera_axes", None)
    if cache is not None:
        return cache
    camera_pose = _get_base_camera_pose_debug(env_unwrapped)
    rotation = np.asarray(camera_pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
    camera_forward = np.asarray(rotation[:, 0], dtype=np.float64)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward_on_plane = camera_forward.copy()
    forward_on_plane[2] = 0.0
    forward_norm = float(np.linalg.norm(forward_on_plane))
    if forward_norm <= 1e-6:
        camera_right = np.asarray(-rotation[:, 1], dtype=np.float64)
        forward_on_plane = np.cross(world_up, camera_right)
        forward_on_plane[2] = 0.0
        forward_norm = float(np.linalg.norm(forward_on_plane))
    if forward_norm <= 1e-6:
        raise RuntimeError("base_camera forward direction is degenerate on the table plane.")
    forward_on_plane = forward_on_plane / forward_norm
    right_on_plane = np.cross(forward_on_plane, world_up)
    right_norm = float(np.linalg.norm(right_on_plane))
    if right_norm <= 1e-6:
        raise RuntimeError("base_camera right direction is degenerate on the table plane.")
    right_on_plane = right_on_plane / right_norm
    cache = (
        right_on_plane.astype(np.float32),
        forward_on_plane.astype(np.float32),
        world_up.astype(np.float32),
    )
    env_unwrapped._planner_proxy_base_camera_axes = cache
    print(
        "[PlannerDebug] Proxy base_camera plane axes: "
        f"right={np.array2string(cache[0], precision=4, suppress_small=True)} "
        f"forward={np.array2string(cache[1], precision=4, suppress_small=True)} "
        f"up={np.array2string(cache[2], precision=4, suppress_small=True)}"
    )
    return cache


def build_proxy_delta_pos(
    env_unwrapped,
    pos_err_vec,
    position_mask,
    *,
    max_xy_step: float,
    max_z_step: float,
    proxy_frame_mode: str,
):
    pos_err_vec = np.asarray(pos_err_vec, dtype=np.float32)
    single_env = pos_err_vec.ndim == 1
    pos_err_vec = pos_err_vec.reshape(1, 3) if single_env else pos_err_vec.reshape(-1, 3)
    position_mask = np.asarray(position_mask, dtype=bool).reshape(3)
    delta_pos = np.zeros_like(pos_err_vec, dtype=np.float32)
    planar_mask = bool(position_mask[0] or position_mask[1])
    if planar_mask and proxy_frame_mode == "base_camera_plane":
        right_axis, forward_axis, _up_axis = _get_base_camera_plane_axes(env_unwrapped)
        planar_world_delta = np.zeros_like(pos_err_vec, dtype=np.float32)
        if position_mask[0]:
            planar_world_delta[:, 0] = pos_err_vec[:, 0]
        if position_mask[1]:
            planar_world_delta[:, 1] = pos_err_vec[:, 1]
        delta_right = np.clip(planar_world_delta @ np.asarray(right_axis, dtype=np.float32), -max_xy_step, max_xy_step)
        delta_forward = np.clip(
            planar_world_delta @ np.asarray(forward_axis, dtype=np.float32),
            -max_xy_step,
            max_xy_step,
        )
        delta_pos += delta_right[:, None] * np.asarray(right_axis, dtype=np.float32).reshape(1, 3)
        delta_pos += delta_forward[:, None] * np.asarray(forward_axis, dtype=np.float32).reshape(1, 3)
    else:
        if position_mask[0]:
            delta_pos[:, 0] = np.clip(pos_err_vec[:, 0], -max_xy_step, max_xy_step)
        if position_mask[1]:
            delta_pos[:, 1] = np.clip(pos_err_vec[:, 1], -max_xy_step, max_xy_step)
    if position_mask[2]:
        delta_pos[:, 2] = np.clip(pos_err_vec[:, 2], -max_z_step, max_z_step)
    return delta_pos[0] if single_env else delta_pos


def _repeat_for_envs(action, num_envs):
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if num_envs <= 1:
        return action.astype(np.float32)
    if action.shape[0] == 1:
        action = np.repeat(action, num_envs, axis=0)
    return action.astype(np.float32)


def _step_env(env, action):
    from openreal2sim.simulation.maniskill.scripts import (
        rc5_unified_proxy_artifacts as unified_proxy_artifacts,
    )

    batched_action = _repeat_for_envs(action, env.unwrapped.num_envs)
    result = env.step(batched_action)
    unified_proxy_artifacts.record_dense_episode_step_if_enabled(env, batched_action, result)
    return result


def _get_runtime_gripper_controller(agent):
    controller = getattr(agent, "controller", None)
    controllers = getattr(controller, "controllers", {}) if controller is not None else {}
    return controllers.get("gripper")


def _get_runtime_gripper_target_qpos_rows(agent, *, num_envs: int):
    gripper_controller = _get_runtime_gripper_controller(agent)
    if gripper_controller is None:
        return None
    target_qpos = getattr(gripper_controller, "_target_qpos", None)
    if target_qpos is None:
        return None
    if hasattr(target_qpos, "detach") and callable(getattr(target_qpos, "detach", None)):
        target_qpos = target_qpos.detach().cpu().numpy()
    target_qpos = np.asarray(target_qpos, dtype=np.float32)
    if target_qpos.ndim == 1:
        target_qpos = target_qpos.reshape(1, -1)
    if target_qpos.shape[0] == 1 and int(num_envs) > 1:
        print(
            f"{_Y}[WARNING] [PlannerDebug] Runtime gripper target qpos payload has a single row; "
            f"repeating env0 target across {int(num_envs)} envs for diagnostics."
            f"{_R}"
        )
        target_qpos = np.repeat(target_qpos, int(num_envs), axis=0)
    if target_qpos.shape[0] != int(num_envs):
        print(
            f"{_Y}[WARNING] [PlannerDebug] Runtime gripper target qpos payload shape={target_qpos.shape} "
            f"does not match num_envs={int(num_envs)}; batched hand-target diagnostics are unavailable."
            f"{_R}"
        )
        return None
    return np.asarray(target_qpos, dtype=np.float32).copy()


def _get_batched_tcp_pose_rows(
    env_unwrapped,
    *,
    num_envs: int,
    get_debug_planner_ee_pose,
    get_debug_planner_ee_pose_rows,
    pose_to_numpy_rows,
):
    if callable(get_debug_planner_ee_pose_rows):
        return pose_to_numpy_rows(get_debug_planner_ee_pose_rows(env_unwrapped))
    print(
        f"{_Y}[WARNING] [PlannerDebug] Batched close did not receive get_debug_planner_ee_pose_rows; "
        "falling back to env0-first EE pose rows for batched TCP diagnostics."
        f"{_R}"
    )
    return pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))


def _runtime_gripper_uses_signal_only(agent) -> bool:
    gripper_controller = _get_runtime_gripper_controller(agent)
    if gripper_controller is None or not hasattr(gripper_controller, "config"):
        return False
    cfg = gripper_controller.config
    has_named_qpos = hasattr(cfg, "open_qpos") or hasattr(cfg, "close_qpos")
    has_bounds = hasattr(cfg, "lower") and hasattr(cfg, "upper")
    return (not has_named_qpos) and has_bounds


def _set_runtime_gripper_target_qpos(agent, target_state: str, target_qpos):
    gripper_controller = _get_runtime_gripper_controller(agent)
    if gripper_controller is None or not hasattr(gripper_controller, "config"):
        raise RuntimeError("Proxy EE-delta backend requires a runtime gripper controller with configurable presets.")
    attr_name = "open_qpos" if target_state == "open" else "close_qpos"
    if not hasattr(gripper_controller.config, attr_name):
        raise RuntimeError(f"Runtime gripper controller config has no attribute '{attr_name}'.")
    setattr(gripper_controller.config, attr_name, np.asarray(target_qpos, dtype=np.float32).tolist())


def _get_proxy_ee_latched_gripper_target(env_unwrapped):
    return getattr(env_unwrapped, "_proxy_ee_latched_gripper_target", None)


def _set_proxy_ee_latched_gripper_target(env_unwrapped, target_state):
    env_unwrapped._proxy_ee_latched_gripper_target = target_state


def relatch_runtime_ee_target_pose_to_current(env_unwrapped, *, reason: str):
    agent = getattr(env_unwrapped, "agent", None)
    controller = getattr(agent, "controller", None) if agent is not None else None
    controllers = getattr(controller, "controllers", {}) if controller is not None else {}
    arm_controller = controllers.get("arm")
    if arm_controller is None:
        raise RuntimeError("Cannot relatch EE target pose: runtime arm controller is unavailable.")
    current_pose = getattr(arm_controller, "ee_pose_at_base", None)
    if current_pose is None:
        raise RuntimeError("Cannot relatch EE target pose: ee_pose_at_base is unavailable on the arm controller.")
    pose_type = type(current_pose)
    if hasattr(pose_type, "create_from_pq"):
        relatched_pose = pose_type.create_from_pq(current_pose.p, current_pose.q)
    else:
        relatched_pose = current_pose
    writable_attrs = []
    for attr_name in ("_target_pose", "_target_pose_at_base"):
        if hasattr(arm_controller, attr_name):
            setattr(arm_controller, attr_name, relatched_pose)
            writable_attrs.append(attr_name)
    if not writable_attrs:
        raise RuntimeError("Cannot relatch EE target pose: controller exposes no writable target pose attribute.")
    pose_np = _pose_to_numpy_first_env(relatched_pose)
    print(
        f"[PlannerDebug] Relatched runtime EE target pose to current TCP ({reason}): "
        f"p={np.array2string(np.asarray(pose_np[0], dtype=np.float32), precision=4, suppress_small=True)} "
        f"q={np.array2string(np.asarray(pose_np[1], dtype=np.float32), precision=4, suppress_small=True)}"
    )


def _infer_proxy_ee_gripper_target(env_unwrapped) -> str:
    agent = env_unwrapped.agent
    grasp_state = get_planner_grasp_state(env_unwrapped)
    target_hand_qpos = None if grasp_state is None else grasp_state.target_hand_qpos
    if target_hand_qpos is not None and hasattr(agent, "hand_open_qpos") and hasattr(agent, "hand_close_qpos"):
        target_hand_qpos = np.asarray(target_hand_qpos, dtype=np.float32).reshape(-1)
        open_qpos = np.asarray(agent.hand_open_qpos, dtype=np.float32).reshape(-1)
        close_qpos = np.asarray(agent.hand_close_qpos, dtype=np.float32).reshape(-1)
        if open_qpos.shape == target_hand_qpos.shape and close_qpos.shape == target_hand_qpos.shape:
            if float(np.linalg.norm(target_hand_qpos - close_qpos)) <= float(np.linalg.norm(target_hand_qpos - open_qpos)):
                return "close"
    if grasp_state is not None and getattr(grasp_state, "grasp_flag", None):
        return "close"
    return "open"


def _get_proxy_ee_gripper_controller_signal(env_unwrapped, target_state: str) -> float:
    if target_state == "hold":
        latched = _get_proxy_ee_latched_gripper_target(env_unwrapped)
        if latched in {"open", "close"}:
            target_state = latched
        else:
            return 0.0
    planner_cfg = get_debug_planner_config(env_unwrapped)
    signal_value = planner_cfg.get("gripper_close_signal") if target_state == "close" else planner_cfg.get("gripper_open_signal")
    if signal_value is None:
        raise RuntimeError(
            "Proxy EE-delta backend requires gripper_open_signal/gripper_close_signal. "
            "Provide teleop_profile_config + teleop_profile or define them in the active config key."
        )
    return float(_map_teleop_gripper_signal_to_controller(signal_value, target_state))


def apply_proxy_ee_delta_action(
    env,
    raw_delta_pos,
    raw_delta_rpy,
    *,
    hold_steps: int,
    stage_label: str,
    gripper_target_state: str = None,
    gripper_signal_override=None,
    pose_to_numpy,
):
    env_unwrapped = env.unwrapped
    control_mode = getattr(env_unwrapped, "control_mode", None)
    if not is_ee_delta_control_mode(control_mode):
        raise RuntimeError(
            f"Proxy EE-delta backend requires an EE-delta control_mode, got '{control_mode}'."
        )
    action_dim = int(env.action_space.shape[-1])
    raw_delta_pos_arr = np.asarray(raw_delta_pos, dtype=np.float32)
    raw_delta_rpy_arr = np.asarray(raw_delta_rpy, dtype=np.float32)
    single_env = raw_delta_pos_arr.ndim == 1
    raw_delta_pos_arr = raw_delta_pos_arr.reshape(1, 3) if single_env else raw_delta_pos_arr.reshape(-1, 3)
    raw_delta_rpy_arr = raw_delta_rpy_arr.reshape(1, 3) if raw_delta_rpy_arr.ndim == 1 else raw_delta_rpy_arr.reshape(-1, 3)
    if raw_delta_rpy_arr.shape[0] == 1 and raw_delta_pos_arr.shape[0] > 1:
        raw_delta_rpy_arr = np.repeat(raw_delta_rpy_arr, raw_delta_pos_arr.shape[0], axis=0)
    base_delta_pos, base_delta_rpy = _transform_world_delta_to_robot_base(
        env_unwrapped,
        raw_delta_pos_arr,
        raw_delta_rpy_arr,
    )
    proxy_remapped_pos, proxy_remapped_rpy = _apply_teleop_delta_remap(
        base_delta_pos,
        base_delta_rpy,
        _get_proxy_delta_remap_rpy_deg(env_unwrapped),
    )
    if gripper_target_state is None:
        gripper_target_state = _infer_proxy_ee_gripper_target(env_unwrapped)
    action = np.zeros((proxy_remapped_pos.shape[0], action_dim), dtype=np.float32)
    action[:, :3] = proxy_remapped_pos
    action[:, 3:6] = proxy_remapped_rpy
    if action_dim >= 7:
        default_gripper_signal = _get_proxy_ee_gripper_controller_signal(env_unwrapped, gripper_target_state)
        gripper_signal = np.full((action.shape[0],), float(default_gripper_signal), dtype=np.float32)
        if gripper_signal_override is not None:
            override = np.asarray(gripper_signal_override, dtype=np.float32)
            if override.ndim == 0:
                override = np.repeat(override.reshape(1), action.shape[0], axis=0)
            else:
                override = override.reshape(-1)
            if override.shape[0] == 1 and action.shape[0] > 1:
                override = np.repeat(override, action.shape[0], axis=0)
            if override.shape[0] != action.shape[0]:
                raise ValueError(
                    f"gripper_signal_override must provide {action.shape[0]} values, got shape={override.shape}."
                )
            valid_override = ~np.isnan(override)
            gripper_signal[valid_override] = override[valid_override]
        action[:, 6] = gripper_signal
    agent = env_unwrapped.agent
    num_envs = int(getattr(env_unwrapped, "num_envs", action.shape[0]) or action.shape[0])
    debug_steps = is_debug_enabled()
    if debug_steps:
        current_tcp_p_rows, current_tcp_q_rows = _pose_to_numpy_rows_local(agent.tcp.pose, num_envs=num_envs)
        current_qpos_rows = _get_robot_qpos_rows(agent.robot, num_envs=num_envs)
    viewer = getattr(env_unwrapped, "_debug_planner_viewer", None)
    append_video = getattr(env_unwrapped, "_debug_planner_append_video_buffer_frame", None)
    for _ in range(max(int(hold_steps), 1)):
        wait_for_proxy_viewer_step_advance(env, viewer)
        _step_env(env, action)
        if callable(append_video):
            append_video()
        _refresh_viewer_during_proxy_step(env, viewer)
        maybe_handle_proxy_viewer_video_hotkey(env, viewer)
    if debug_steps:
        actual_tcp_p_rows, actual_tcp_q_rows = _pose_to_numpy_rows_local(agent.tcp.pose, num_envs=num_envs)
        actual_qpos_rows = _get_robot_qpos_rows(agent.robot, num_envs=num_envs)
        base_delta_pos_rows = np.asarray(base_delta_pos, dtype=np.float32).reshape(-1, 3)
        base_delta_rpy_rows = np.asarray(base_delta_rpy, dtype=np.float32).reshape(-1, 3)
        proxy_pos_rows = np.asarray(proxy_remapped_pos, dtype=np.float32).reshape(-1, 3)
        proxy_rpy_rows = np.asarray(proxy_remapped_rpy, dtype=np.float32).reshape(-1, 3)
        qpos_delta_rows = actual_qpos_rows - current_qpos_rows
        joint_order = [joint.name for joint in agent.robot.get_active_joints()]
        log_rows = min(
            int(num_envs),
            raw_delta_pos_arr.shape[0],
            raw_delta_rpy_arr.shape[0],
            base_delta_pos_rows.shape[0],
            base_delta_rpy_rows.shape[0],
            proxy_pos_rows.shape[0],
            proxy_rpy_rows.shape[0],
            current_tcp_p_rows.shape[0],
            actual_tcp_p_rows.shape[0],
            current_qpos_rows.shape[0],
            actual_qpos_rows.shape[0],
            qpos_delta_rows.shape[0],
            action.shape[0],
        )
        for env_id in range(log_rows):
            qpos_delta = qpos_delta_rows[env_id]
            max_joint_idx = int(np.argmax(np.abs(qpos_delta))) if qpos_delta.size > 0 else -1
            max_joint_name = (
                joint_order[max_joint_idx]
                if 0 <= max_joint_idx < len(joint_order)
                else "<none>"
            )
            logger.debug(
                "[{}] env_id={} proxy_raw_delta_pos={} proxy_raw_delta_rpy={} "
                "proxy_base_delta_pos={} proxy_base_delta_rpy={} "
                "proxy_action_delta_pos={} proxy_action_delta_rpy={} "
                "gripper_action={:.4f} tcp_pos_before={} tcp_pos_after={} "
                "tcp_quat_before={} tcp_quat_after={} "
                "qpos_before={} qpos_after={} qpos_delta={} "
                "max_abs_qpos_delta joint={} delta={:.4f}",
                stage_label,
                env_id,
                np.array2string(raw_delta_pos_arr[env_id], precision=4, suppress_small=True),
                np.array2string(raw_delta_rpy_arr[env_id], precision=4, suppress_small=True),
                np.array2string(base_delta_pos_rows[env_id], precision=4, suppress_small=True),
                np.array2string(base_delta_rpy_rows[env_id], precision=4, suppress_small=True),
                np.array2string(proxy_pos_rows[env_id], precision=4, suppress_small=True),
                np.array2string(proxy_rpy_rows[env_id], precision=4, suppress_small=True),
                float(action[env_id, 6]) if action.shape[1] >= 7 else float("nan"),
                np.array2string(current_tcp_p_rows[env_id], precision=4, suppress_small=True),
                np.array2string(actual_tcp_p_rows[env_id], precision=4, suppress_small=True),
                np.array2string(current_tcp_q_rows[env_id], precision=4, suppress_small=True),
                np.array2string(actual_tcp_q_rows[env_id], precision=4, suppress_small=True),
                np.array2string(current_qpos_rows[env_id], precision=4, suppress_small=True),
                np.array2string(actual_qpos_rows[env_id], precision=4, suppress_small=True),
                np.array2string(qpos_delta, precision=4, suppress_small=True),
                max_joint_name,
                float(qpos_delta[max_joint_idx]) if max_joint_idx >= 0 else float("nan"),
            )
    return True


def run_proxy_stationary_settle(
    env,
    *,
    settle_steps: int,
    stage_label: str,
    gripper_target_state: str = "hold",
    get_debug_planner_ee_pose_sapien,
    get_debug_planner_ee_pose=None,
    pose_to_numpy,
    pose_to_numpy_rows=None,
):
    env_unwrapped = env.unwrapped
    planner_cfg = get_debug_planner_config(env_unwrapped)
    settle_steps = max(int(settle_steps), 0)
    if settle_steps <= 0:
        return True
    use_batched_pose = callable(get_debug_planner_ee_pose) and callable(pose_to_numpy_rows) and int(getattr(env_unwrapped, "num_envs", 1) or 1) > 1
    if use_batched_pose:
        start_tcp_p_rows, _start_tcp_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))
        start_tcp_p = start_tcp_p_rows[0]
    else:
        start_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
        start_tcp_p = np.asarray(start_pose.p, dtype=np.float32).reshape(-1)[:3]
        start_tcp_p_rows = None
    max_xy_step = float(min(planner_cfg.get("planner_proxy_xy_step_m", 0.01), 0.002))
    max_z_step = float(min(planner_cfg.get("planner_proxy_z_step_m", 0.008), 0.002))
    proxy_frame_mode = _get_proxy_frame_mode(env_unwrapped)
    print(
        f"[PlannerDebug] Proxy stationary settle '{stage_label}' start: settle_steps={settle_steps} "
        f"start_tcp_p={np.array2string(start_tcp_p, precision=4, suppress_small=True)}"
    )
    for step_idx in range(1, settle_steps + 1):
        current_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
        if use_batched_pose:
            current_tcp_p_rows, _current_tcp_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))
            current_tcp_p = current_tcp_p_rows[0]
            pos_err_vec = start_tcp_p_rows - current_tcp_p_rows
        else:
            current_tcp_p = np.asarray(current_pose.p, dtype=np.float32).reshape(-1)[:3]
            pos_err_vec = start_tcp_p - current_tcp_p
        delta_pos = build_proxy_delta_pos(
            env_unwrapped,
            pos_err_vec,
            position_mask=(True, True, True),
            max_xy_step=max_xy_step,
            max_z_step=max_z_step,
            proxy_frame_mode=proxy_frame_mode,
        )
        apply_proxy_ee_delta_action(
            env,
            raw_delta_pos=delta_pos,
            raw_delta_rpy=np.zeros(3, dtype=np.float32),
            hold_steps=1,
            stage_label=f"{stage_label}:step{step_idx}",
            gripper_target_state=gripper_target_state,
            pose_to_numpy=pose_to_numpy,
        )
    final_tcp_p, _ = pose_to_numpy(get_debug_planner_ee_pose_sapien(env_unwrapped))
    settle_drift = np.asarray(final_tcp_p - start_tcp_p, dtype=np.float32)
    print(
        f"[PlannerDebug] Proxy stationary settle '{stage_label}' end: final_tcp_p="
        f"{np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
        f"settle_drift={np.array2string(settle_drift, precision=4, suppress_small=True)} "
        f"settle_drift_norm={float(np.linalg.norm(settle_drift)):.4f} m"
    )
    return True


def run_proxy_guarded_descend_to_object(
    env,
    initial_target_pose,
    *,
    initial_actor_p,
    bbox_np,
    stage_label: str,
    align_orientation: bool,
    get_debug_planner_ee_pose_sapien,
    get_debug_planner_ee_pose=None,
    get_debug_target_object,
    get_debug_actor_position_xyz,
    get_debug_actor_position_rows=None,
    pose_to_numpy_rows=None,
    log_debug_non_target_object_contacts=None,
):
    env_unwrapped = env.unwrapped
    planner_cfg = get_debug_planner_config(env_unwrapped)
    use_batched_pose = callable(get_debug_planner_ee_pose) and callable(pose_to_numpy_rows) and callable(get_debug_actor_position_rows) and int(getattr(env_unwrapped, "num_envs", 1) or 1) > 1
    if use_batched_pose:
        initial_target_p = np.asarray(initial_target_pose.p, dtype=np.float32).reshape(-1, 3)
        initial_target_q = np.asarray(initial_target_pose.q, dtype=np.float32).reshape(-1, 4)[0]
        initial_actor_p = np.asarray(initial_actor_p, dtype=np.float32).reshape(-1, 3)
    else:
        initial_target_pose = _to_sapien_pose_first_env(initial_target_pose)
        initial_target_p = np.asarray(initial_target_pose.p, dtype=np.float32).reshape(-1)[:3]
        initial_target_q = np.asarray(initial_target_pose.q, dtype=np.float32).reshape(-1)[:4]
        initial_actor_p = np.asarray(initial_actor_p, dtype=np.float32).reshape(-1)[:3]
    proxy_frame_mode = _get_proxy_frame_mode(env_unwrapped)
    max_xy_step = float(min(planner_cfg.get("planner_proxy_xy_step_m", 0.01), 0.004))
    max_z_step = float(planner_cfg.get("planner_proxy_z_step_m", 0.008))
    max_rot_step = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_step_deg", 6.0)))
    rot_tol = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_tol_deg", 8.0)))
    _adaptive_enabled, _threshold_xy, _threshold_z, pos_tol = get_required_proxy_adaptive_step_config(planner_cfg)
    hold_steps = int(planner_cfg.get("planner_proxy_hold_steps", 1) or 1)
    max_stage_steps = int(planner_cfg.get("planner_proxy_max_stage_steps", 100) or 100)
    stall_limit = int(planner_cfg.get("planner_proxy_stall_steps", 12) or 12)
    last_total_error = None
    stall_count = 0
    adaptive_xy_logged = False
    adaptive_z_logged = False
    print(
        f"[PlannerDebug] Guarded proxy descend '{stage_label}' start: "
        f"initial_actor_p={np.array2string(initial_actor_p, precision=4, suppress_small=True)} "
        f"initial_target_p={np.array2string(initial_target_p, precision=4, suppress_small=True)} "
        f"proxy_frame={proxy_frame_mode}"
    )
    for step_idx in range(1, max_stage_steps + 1):
        current_actor = get_debug_target_object(env_unwrapped)
        if callable(log_debug_non_target_object_contacts):
            log_debug_non_target_object_contacts(
                env_unwrapped,
                stage_label=f"{stage_label}:step{step_idx}",
            )
        if use_batched_pose:
            current_actor_p = np.asarray(get_debug_actor_position_rows(current_actor), dtype=np.float32).reshape(-1, 3)
            current_p_rows, current_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))
            current_q = current_q_rows[0]
            dynamic_target_p = initial_target_p + (current_actor_p - initial_actor_p)
            pos_err_vec = dynamic_target_p - current_p_rows
            xy_err = float(np.max(np.linalg.norm(pos_err_vec[:, :2], axis=1)))
            z_err = float(np.max(np.abs(pos_err_vec[:, 2])))
        else:
            current_actor_p = get_debug_actor_position_xyz(current_actor)
            current_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
            current_p_rows = None
            current_p = np.asarray(current_pose.p, dtype=np.float32).reshape(-1)[:3]
            current_q = np.asarray(current_pose.q, dtype=np.float32).reshape(-1)[:4]
            dynamic_target_p = initial_target_p + (current_actor_p - initial_actor_p)
            pos_err_vec = dynamic_target_p - current_p
            xy_err = float(np.linalg.norm(pos_err_vec[:2]))
            z_err = float(abs(pos_err_vec[2]))
        delta_rpy, rot_err = compute_proxy_rotvec_step(
            current_q,
            initial_target_q,
            max_rot_step,
        )
        orientation_reached = not align_orientation or rot_err <= rot_tol
        total_error = xy_err + z_err + (rot_err if align_orientation else 0.0)
        if xy_err <= pos_tol and z_err <= pos_tol and orientation_reached:
            if adaptive_xy_logged:
                print(
                    f"[PlannerDebug] Guarded proxy descend '{stage_label}' released adaptive XY step at convergence; "
                    f"restoring nominal_xy_step={max_xy_step:.4f} m for subsequent stages"
                )
            if adaptive_z_logged:
                print(
                    f"[PlannerDebug] Guarded proxy descend '{stage_label}' released adaptive Z step at convergence; "
                    f"restoring nominal_z_step={max_z_step:.4f} m for subsequent stages"
                )
            print(
                f"[PlannerDebug] Guarded proxy descend '{stage_label}' converged at step {step_idx - 1}: "
                f"xy_err={xy_err:.4f} m z_err={z_err:.4f} m rot_err={rot_err:.4f} rad "
                f"current_actor_p={np.array2string(np.asarray(current_actor_p, dtype=np.float32).reshape(-1, 3)[0], precision=4, suppress_small=True)}"
            )
            return True
        adaptive_xy_step = get_adaptive_proxy_xy_step(
            planner_cfg,
            xy_err=xy_err,
            nominal_xy_step=max_xy_step,
        )
        adaptive_z_step = get_adaptive_proxy_z_step_near_target(
            planner_cfg,
            tcp_z=float(np.asarray(current_p, dtype=np.float32).reshape(-1)[:3][2]),
            descend_target_z=float(np.asarray(dynamic_target_p, dtype=np.float32).reshape(-1)[:3][2]),
            nominal_z_step=max_z_step,
        )
        if adaptive_xy_step < max_xy_step and not adaptive_xy_logged:
            adaptive_xy_logged = True
            print(
                f"[PlannerDebug] Guarded proxy descend '{stage_label}' switched to adaptive XY step at step {step_idx}: "
                f"xy_err={xy_err:.4f} m threshold_xy={float(planner_cfg['planner_proxy_threshold_xy_m']):.4f} m "
                f"nominal_xy_step={max_xy_step:.4f} m adaptive_xy_step={adaptive_xy_step:.4f} m"
            )
        if adaptive_z_step < max_z_step and not adaptive_z_logged:
            adaptive_z_logged = True
            print(
                f"[PlannerDebug] Guarded proxy descend '{stage_label}' switched to adaptive Z step at step {step_idx}: "
                f"tcp_z={float(np.asarray(current_p, dtype=np.float32).reshape(-1)[:3][2]):.4f} "
                f"descend_target_z={float(np.asarray(dynamic_target_p, dtype=np.float32).reshape(-1)[:3][2]):.4f} "
                f"threshold_z={float(planner_cfg['planner_proxy_threshold_z_m']):.4f} m "
                f"nominal_z_step={max_z_step:.4f} m adaptive_z_step={adaptive_z_step:.4f} m"
            )
        delta_pos = build_proxy_delta_pos(
            env_unwrapped,
            pos_err_vec,
            position_mask=(True, True, True),
            max_xy_step=adaptive_xy_step,
            max_z_step=adaptive_z_step,
            proxy_frame_mode=proxy_frame_mode,
        )
        if not align_orientation or rot_err <= rot_tol:
            delta_rpy = np.zeros(3, dtype=np.float32)
        stall_count = stall_count + 1 if last_total_error is not None and total_error >= (last_total_error - 1e-4) else 0
        last_total_error = total_error
        if np.linalg.norm(delta_pos) <= 1e-6 and np.linalg.norm(delta_rpy) <= 1e-6:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Guarded proxy descend '{stage_label}' produced a near-zero step "
                f"before reaching tolerance; aborting.{_R}"
            )
            return False
        if stall_count >= stall_limit:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Guarded proxy descend '{stage_label}' stalled for {stall_count} iterations "
                f"(xy_err={xy_err:.4f} m, z_err={z_err:.4f} m).{_R}"
            )
            return False
        if is_debug_enabled():
            logger.debug(
                "[PlannerDebug] Guarded proxy descend '{}' step {}/{}: "
                "dynamic_target_p={} current_actor_p={} xy_err={:.4f} m z_err={:.4f} m "
                "rot_err={:.4f} rad delta_pos={} delta_rpy={}",
                stage_label,
                step_idx,
                max_stage_steps,
                np.array2string(dynamic_target_p, precision=4, suppress_small=True),
                np.array2string(current_actor_p, precision=4, suppress_small=True),
                xy_err,
                z_err,
                rot_err,
                np.array2string(delta_pos, precision=4, suppress_small=True),
                np.array2string(delta_rpy, precision=4, suppress_small=True),
            )
        apply_proxy_ee_delta_action(
            env,
            raw_delta_pos=delta_pos,
            raw_delta_rpy=delta_rpy,
            hold_steps=hold_steps,
            stage_label=f"{stage_label}:step{step_idx}",
            gripper_target_state="hold",
            pose_to_numpy=_pose_to_numpy_first_env,
        )
    return False


def run_proxy_close_gripper(
    env,
    *,
    close_steps: int,
    get_debug_planner_ee_pose,
    get_debug_planner_ee_pose_rows=None,
    pose_to_numpy,
    pose_to_numpy_rows=None,
    get_debug_target_object,
    get_debug_actor_position_xyz,
    log_debug_pre_close_snapshot,
    log_debug_post_close_retention,
    get_robot_hand_qpos_debug,
    get_robot_hand_range_debug,
    refresh_render_state,
    run_proxy_guarded_descend_to_object=None,
):
    env_unwrapped = env.unwrapped
    agent = env_unwrapped.agent
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    planner_cfg = get_debug_planner_config(env_unwrapped)
    preclose_settle_steps = int(planner_cfg.get("planner_proxy_preclose_settle_steps", 6) or 0)
    relatch_target_pose_between_stages = bool(
        planner_cfg.get("planner_proxy_relatch_ee_target_pose_between_stages", False)
    )
    # Batch execution must match singleton physics unless the runtime config
    # explicitly opts into batch-only close heuristics.
    default_postclose_settle_steps = 0
    postclose_settle_steps = int(
        planner_cfg.get("planner_proxy_postclose_settle_steps", default_postclose_settle_steps) or 0
    )
    default_retry_rounds = 0
    default_retry_steps = 0
    default_retry_settle_steps = 0
    default_reseat_rounds = 0
    default_reseat_close_steps = 0
    default_reseat_settle_steps = 0
    batch_close_retry_rounds = int(
        planner_cfg.get("planner_proxy_batch_close_retry_rounds", default_retry_rounds) or 0
    )
    batch_close_retry_steps = int(
        planner_cfg.get("planner_proxy_batch_close_retry_steps", default_retry_steps) or 0
    )
    batch_close_retry_settle_steps = int(
        planner_cfg.get("planner_proxy_batch_close_retry_settle_steps", default_retry_settle_steps) or 0
    )
    batch_postclose_reseat_rounds = int(
        planner_cfg.get("planner_proxy_batch_postclose_reseat_rounds", default_reseat_rounds) or 0
    )
    batch_postclose_reseat_close_steps = int(
        planner_cfg.get("planner_proxy_batch_postclose_reseat_close_steps", default_reseat_close_steps) or 0
    )
    batch_postclose_reseat_settle_steps = int(
        planner_cfg.get("planner_proxy_batch_postclose_reseat_settle_steps", default_reseat_settle_steps) or 0
    )
    batch_postclose_reseat_max_object_tcp_dist_raw = planner_cfg.get(
        "planner_proxy_batch_postclose_reseat_max_object_tcp_dist_m",
        0.12,
    )
    if batch_postclose_reseat_max_object_tcp_dist_raw is None:
        batch_postclose_reseat_max_object_tcp_dist_raw = 0.12
    batch_postclose_reseat_max_object_tcp_dist = float(batch_postclose_reseat_max_object_tcp_dist_raw)
    close_target_qpos = np.asarray(getattr(agent, "hand_close_qpos", []), dtype=np.float32).reshape(-1)
    if close_target_qpos.size == 0:
        raise RuntimeError("Proxy close requires a non-empty hand_close_qpos target.")
    if num_envs > 1:
        print(
            f"[PlannerDebug] Batched close effective config: "
            f"preclose_settle_steps={preclose_settle_steps} "
            f"postclose_settle_steps={postclose_settle_steps} "
            f"retry_rounds={batch_close_retry_rounds} retry_steps={batch_close_retry_steps} "
            f"retry_settle_steps={batch_close_retry_settle_steps} "
            f"reseat_rounds={batch_postclose_reseat_rounds} "
            f"reseat_close_steps={batch_postclose_reseat_close_steps} "
            f"reseat_settle_steps={batch_postclose_reseat_settle_steps} "
            f"reseat_max_object_tcp_dist_m={batch_postclose_reseat_max_object_tcp_dist:.4f}"
        )
        if (
            postclose_settle_steps > 0
            or batch_close_retry_rounds > 0
            or batch_close_retry_steps > 0
            or batch_postclose_reseat_rounds > 0
            or batch_postclose_reseat_close_steps > 0
            or batch_postclose_reseat_settle_steps > 0
        ):
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batch-only close heuristics are enabled via runtime config; "
                "this batch run will intentionally diverge from singleton parity in the close/retention path."
                f"{_R}"
            )
    signal_only_gripper = _runtime_gripper_uses_signal_only(agent)
    if signal_only_gripper:
        print(
            f"[PlannerDebug] Proxy close: agent uid='{getattr(agent, 'uid', 'unknown')}' "
            "uses a signal-only runtime gripper controller; skipping preset injection."
        )
    elif num_envs > 1:
        firm_close_alpha = float(planner_cfg.get("planner_proxy_batch_firm_close_alpha", 0.0) or 0.0)
        firm_close_target_qpos = _build_rc5_batch_firm_close_target_qpos(
            agent,
            close_target_qpos,
            alpha=firm_close_alpha,
        )
        if firm_close_target_qpos is not None:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batched RC5 firm close target enabled: alpha={firm_close_alpha:.2f} "
                f"base_range=[{float(np.min(close_target_qpos)):.4f}, {float(np.max(close_target_qpos)):.4f}] "
                f"firm_range=[{float(np.min(firm_close_target_qpos)):.4f}, {float(np.max(firm_close_target_qpos)):.4f}]"
                f"{_R}"
            )
            close_target_qpos = firm_close_target_qpos
    _set_proxy_ee_latched_gripper_target(env_unwrapped, None)
    target_object = get_debug_target_object(env_unwrapped)
    object_p_before_close = None if target_object is None else get_debug_actor_position_xyz(target_object)
    object_p_before_close_rows = None
    if (
        target_object is not None
        and num_envs > 1
        and callable(pose_to_numpy_rows)
    ):
        object_p_before_close_rows = _get_actor_position_rows(target_object, num_envs=num_envs)
    if target_object is not None:
        log_debug_pre_close_snapshot(env_unwrapped, target_object, int(close_steps))
    if relatch_target_pose_between_stages:
        relatch_runtime_ee_target_pose_to_current(
            env_unwrapped,
            reason="before_preclose_settle",
        )
    run_proxy_stationary_settle(
        env,
        settle_steps=preclose_settle_steps,
        stage_label="PreCloseSettle",
        gripper_target_state="hold",
        get_debug_planner_ee_pose_sapien=lambda base_env: _to_sapien_pose_first_env(get_debug_planner_ee_pose(base_env)),
        pose_to_numpy=pose_to_numpy,
    )
    if not signal_only_gripper:
        _set_runtime_gripper_target_qpos(agent, "close", close_target_qpos)
        _set_proxy_ee_latched_gripper_target(env_unwrapped, "close")
    else:
        _set_proxy_ee_latched_gripper_target(env_unwrapped, "close")
    for step_idx in range(1, max(int(close_steps), 1) + 1):
        apply_proxy_ee_delta_action(
            env,
            raw_delta_pos=np.zeros(3, dtype=np.float32),
            raw_delta_rpy=np.zeros(3, dtype=np.float32),
            hold_steps=1,
            stage_label=f"Close:step{step_idx}",
            gripper_target_state="hold",
            pose_to_numpy=pose_to_numpy,
        )
    if postclose_settle_steps > 0:
        if num_envs > 1 and not callable(pose_to_numpy_rows):
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batched close requested PostCloseSettle without pose_to_numpy_rows; "
                "per-env settle diagnostics will fall back to env0-first pose tracking."
                f"{_R}"
            )
        if not run_proxy_stationary_settle(
            env,
            settle_steps=postclose_settle_steps,
            stage_label="PostCloseSettle",
            gripper_target_state="close",
            get_debug_planner_ee_pose_sapien=lambda base_env: _to_sapien_pose_first_env(get_debug_planner_ee_pose(base_env)),
            get_debug_planner_ee_pose=(
                get_debug_planner_ee_pose_rows
                if callable(get_debug_planner_ee_pose_rows)
                else get_debug_planner_ee_pose
            ),
            pose_to_numpy=pose_to_numpy,
            pose_to_numpy_rows=pose_to_numpy_rows,
        ):
            raise RuntimeError("Proxy close post-settle failed before retention check.")
    if (
        num_envs > 1
        and target_object is not None
        and batch_close_retry_rounds > 0
        and batch_close_retry_steps > 0
    ):
        if not callable(pose_to_numpy_rows):
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batched close retries are enabled without pose_to_numpy_rows; "
                "retry diagnostics will be incomplete and env0-first where pose reads are required."
                f"{_R}"
            )
        for retry_idx in range(1, batch_close_retry_rounds + 1):
            grasp_flags = _normalize_per_env_bool_list(
                env_unwrapped.agent.is_grasping(target_object),
                num_envs=num_envs,
            )
            if all(grasp_flags):
                print(
                    f"[PlannerDebug] Batched close reinforcement skipped at retry {retry_idx}: "
                    f"grasp_flags={grasp_flags}"
                )
                break
            object_p_rows = _get_actor_position_rows(target_object, num_envs=num_envs)
            print(
                f"{_Y}[PlannerDebug] Batched close reinforcement retry {retry_idx}/{batch_close_retry_rounds}: "
                f"grasp_flags={grasp_flags} "
                f"object_z={np.array2string(object_p_rows[:, 2], precision=4, suppress_small=True)}{_R}"
            )
            for step_idx in range(1, batch_close_retry_steps + 1):
                apply_proxy_ee_delta_action(
                    env,
                    raw_delta_pos=np.zeros(3, dtype=np.float32),
                    raw_delta_rpy=np.zeros(3, dtype=np.float32),
                    hold_steps=1,
                    stage_label=f"CloseRetry{retry_idx}:step{step_idx}",
                    gripper_target_state="close",
                    pose_to_numpy=pose_to_numpy,
                )
            if batch_close_retry_settle_steps > 0:
                if not run_proxy_stationary_settle(
                    env,
                    settle_steps=batch_close_retry_settle_steps,
                    stage_label=f"CloseRetrySettle{retry_idx}",
                    gripper_target_state="close",
                    get_debug_planner_ee_pose_sapien=lambda base_env: _to_sapien_pose_first_env(get_debug_planner_ee_pose(base_env)),
                    get_debug_planner_ee_pose=(
                        get_debug_planner_ee_pose_rows
                        if callable(get_debug_planner_ee_pose_rows)
                        else get_debug_planner_ee_pose
                    ),
                    pose_to_numpy=pose_to_numpy,
                    pose_to_numpy_rows=pose_to_numpy_rows,
                ):
                    raise RuntimeError("Proxy close retry settle failed before retention check.")
        final_retry_flags = _normalize_per_env_bool_list(
            env_unwrapped.agent.is_grasping(target_object),
            num_envs=num_envs,
        )
        print(f"[PlannerDebug] Batched close final instantaneous grasp_flags={final_retry_flags}")
    if (
        num_envs > 1
        and target_object is not None
        and object_p_before_close_rows is not None
        and callable(pose_to_numpy_rows)
        and callable(run_proxy_guarded_descend_to_object)
        and batch_postclose_reseat_rounds > 0
        and batch_postclose_reseat_close_steps > 0
    ):
        if not callable(run_proxy_guarded_descend_to_object):
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batched post-close reseat is enabled but no guarded descend "
                "callback is available; reseat fallback is disabled for this run."
                f"{_R}"
            )
        last_task_pose = getattr(env_unwrapped, "_planner_last_task_pose", None)
        last_task_stage = str(getattr(env_unwrapped, "_planner_last_task_pose_stage", "") or "")
        if last_task_pose is not None and last_task_stage == "descend":
            for reseat_idx in range(1, batch_postclose_reseat_rounds + 1):
                current_tcp_p_rows, _current_tcp_q_rows = _get_batched_tcp_pose_rows(
                    env_unwrapped,
                    num_envs=num_envs,
                    get_debug_planner_ee_pose=get_debug_planner_ee_pose,
                    get_debug_planner_ee_pose_rows=get_debug_planner_ee_pose_rows,
                    pose_to_numpy_rows=pose_to_numpy_rows,
                )
                object_p_rows = _get_actor_position_rows(target_object, num_envs=num_envs)
                object_tcp_dist_rows = np.linalg.norm(object_p_rows - current_tcp_p_rows, axis=1)
                if float(np.max(object_tcp_dist_rows)) <= float(batch_postclose_reseat_max_object_tcp_dist):
                    print(
                        f"[PlannerDebug] Batched post-close reseat skipped at round {reseat_idx}: "
                        f"object_tcp_dist={np.array2string(object_tcp_dist_rows, precision=4, suppress_small=True)}"
                    )
                    break
                print(
                    f"{_Y}[PlannerDebug] Batched post-close reseat {reseat_idx}/{batch_postclose_reseat_rounds}: "
                    f"object_tcp_dist={np.array2string(object_tcp_dist_rows, precision=4, suppress_small=True)} "
                    f"threshold={batch_postclose_reseat_max_object_tcp_dist:.4f}{_R}"
                )
                if not run_proxy_guarded_descend_to_object(
                    env,
                    last_task_pose,
                    initial_actor_p=object_p_before_close_rows,
                    bbox_np=getattr(env_unwrapped, "_planner_last_task_bbox_np", None),
                    stage_label=f"PostCloseReseat{reseat_idx}",
                    align_orientation=False,
                ):
                    raise RuntimeError("Proxy close post-close reseat descend failed before lift.")
                for step_idx in range(1, batch_postclose_reseat_close_steps + 1):
                    apply_proxy_ee_delta_action(
                        env,
                        raw_delta_pos=np.zeros(3, dtype=np.float32),
                        raw_delta_rpy=np.zeros(3, dtype=np.float32),
                        hold_steps=1,
                        stage_label=f"PostCloseReseatClose{reseat_idx}:step{step_idx}",
                        gripper_target_state="close",
                        pose_to_numpy=pose_to_numpy,
                    )
                if batch_postclose_reseat_settle_steps > 0:
                    if not run_proxy_stationary_settle(
                        env,
                        settle_steps=batch_postclose_reseat_settle_steps,
                        stage_label=f"PostCloseReseatSettle{reseat_idx}",
                        gripper_target_state="close",
                        get_debug_planner_ee_pose_sapien=lambda base_env: _to_sapien_pose_first_env(get_debug_planner_ee_pose(base_env)),
                        get_debug_planner_ee_pose=(
                            get_debug_planner_ee_pose_rows
                            if callable(get_debug_planner_ee_pose_rows)
                            else get_debug_planner_ee_pose
                        ),
                        pose_to_numpy=pose_to_numpy,
                        pose_to_numpy_rows=pose_to_numpy_rows,
                    ):
                        raise RuntimeError("Proxy close post-close reseat settle failed before lift.")
            final_object_p_rows = _get_actor_position_rows(target_object, num_envs=num_envs)
            final_tcp_p_rows, _final_tcp_q_rows = _get_batched_tcp_pose_rows(
                env_unwrapped,
                num_envs=num_envs,
                get_debug_planner_ee_pose=get_debug_planner_ee_pose,
                get_debug_planner_ee_pose_rows=get_debug_planner_ee_pose_rows,
                pose_to_numpy_rows=pose_to_numpy_rows,
            )
            final_object_tcp_dist_rows = np.linalg.norm(final_object_p_rows - final_tcp_p_rows, axis=1)
            print(
                f"[PlannerDebug] Batched post-close reseat final object_tcp_dist="
                f"{np.array2string(final_object_tcp_dist_rows, precision=4, suppress_small=True)}"
            )
    elif num_envs > 1 and batch_postclose_reseat_rounds > 0 and not callable(run_proxy_guarded_descend_to_object):
        print(
            f"{_Y}[WARNING] [PlannerDebug] Batched post-close reseat is enabled but guarded descend "
            "callback is unavailable; reseat fallback is disabled for this run."
            f"{_R}"
        )
    elif num_envs > 1 and batch_postclose_reseat_rounds > 0 and not callable(pose_to_numpy_rows):
        print(
            f"{_Y}[WARNING] [PlannerDebug] Batched post-close reseat is enabled without pose_to_numpy_rows; "
            "per-env reseat diagnostics cannot be computed."
            f"{_R}"
        )
    final_tcp_p, _ = pose_to_numpy(get_debug_planner_ee_pose(env_unwrapped))
    grasp_flag_rows_for_state = None
    if num_envs > 1 and target_object is not None:
        if callable(pose_to_numpy_rows):
            current_tcp_p_rows, _current_tcp_q_rows = _get_batched_tcp_pose_rows(
                env_unwrapped,
                num_envs=num_envs,
                get_debug_planner_ee_pose=get_debug_planner_ee_pose,
                get_debug_planner_ee_pose_rows=get_debug_planner_ee_pose_rows,
                pose_to_numpy_rows=pose_to_numpy_rows,
            )
            object_p_rows = _get_actor_position_rows(target_object, num_envs=num_envs)
            grasp_flags_rows = _normalize_per_env_bool_list(
                env_unwrapped.agent.is_grasping(target_object),
                num_envs=num_envs,
            )
            grasp_flag_rows_for_state = np.asarray(grasp_flags_rows, dtype=bool)
            object_minus_tcp_rows = object_p_rows - current_tcp_p_rows
            object_tcp_dist_rows = np.linalg.norm(object_p_rows - current_tcp_p_rows, axis=1)
            object_delta_rows = None
            if object_p_before_close_rows is not None:
                object_delta_rows = object_p_rows - object_p_before_close_rows
            robot_qpos = env_unwrapped.agent.robot.get_qpos()
            if hasattr(robot_qpos, "detach") and callable(getattr(robot_qpos, "detach", None)):
                robot_qpos = robot_qpos.detach().cpu().numpy()
            robot_qpos = np.asarray(robot_qpos, dtype=np.float32)
            if robot_qpos.ndim == 1:
                robot_qpos = robot_qpos.reshape(1, -1)
            if robot_qpos.shape[0] == 1 and num_envs > 1:
                print(
                    f"{_Y}[WARNING] [PlannerDebug] Batched close hand qpos payload has a single row; "
                    f"repeating env0 hand state across {num_envs} envs for diagnostics."
                    f"{_R}"
                )
                robot_qpos = np.repeat(robot_qpos, num_envs, axis=0)
            hand_qpos_start = len(getattr(env_unwrapped.agent, "arm_joint_names", []))
            hand_qpos_rows = np.asarray(robot_qpos[:, hand_qpos_start:], dtype=np.float32)
            hand_min_rows = np.min(hand_qpos_rows, axis=1) if hand_qpos_rows.size > 0 else np.zeros((num_envs,), dtype=np.float32)
            hand_max_rows = np.max(hand_qpos_rows, axis=1) if hand_qpos_rows.size > 0 else np.zeros((num_envs,), dtype=np.float32)
            hand_target_qpos_rows = _get_runtime_gripper_target_qpos_rows(
                env_unwrapped.agent,
                num_envs=num_envs,
            )
            hand_target_err_rows = None
            if hand_target_qpos_rows is None:
                print(
                    f"{_Y}[WARNING] [PlannerDebug] Batched close could not read runtime gripper target qpos rows; "
                    "actual-vs-target hand diagnostics are unavailable."
                    f"{_R}"
                )
            elif hand_target_qpos_rows.shape == hand_qpos_rows.shape:
                hand_target_err_rows = np.linalg.norm(hand_qpos_rows - hand_target_qpos_rows, axis=1)
            else:
                print(
                    f"{_Y}[WARNING] [PlannerDebug] Batched close hand target shape={hand_target_qpos_rows.shape} "
                    f"does not match hand qpos shape={hand_qpos_rows.shape}; "
                    "actual-vs-target hand diagnostics are unavailable."
                    f"{_R}"
                )
            print(
                "[PlannerDebug] Batched close diagnostics: "
                f"grasp_flags={grasp_flags_rows} "
                f"tcp_p={np.array2string(current_tcp_p_rows, precision=4, suppress_small=True)} "
                f"object_p={np.array2string(object_p_rows, precision=4, suppress_small=True)} "
                f"object_minus_tcp={np.array2string(object_minus_tcp_rows, precision=4, suppress_small=True)} "
                f"object_delta={np.array2string(object_delta_rows, precision=4, suppress_small=True) if object_delta_rows is not None else '[n/a]'} "
                f"object_z={np.array2string(object_p_rows[:, 2], precision=4, suppress_small=True)} "
                f"object_tcp_dist={np.array2string(object_tcp_dist_rows, precision=4, suppress_small=True)} "
                f"hand_min={np.array2string(hand_min_rows, precision=4, suppress_small=True)} "
                f"hand_max={np.array2string(hand_max_rows, precision=4, suppress_small=True)} "
                f"hand_target_min={np.array2string(np.min(hand_target_qpos_rows, axis=1), precision=4, suppress_small=True) if hand_target_qpos_rows is not None and hand_target_qpos_rows.size > 0 else '[n/a]'} "
                f"hand_target_max={np.array2string(np.max(hand_target_qpos_rows, axis=1), precision=4, suppress_small=True) if hand_target_qpos_rows is not None and hand_target_qpos_rows.size > 0 else '[n/a]'} "
                f"hand_target_err={np.array2string(hand_target_err_rows, precision=4, suppress_small=True) if hand_target_err_rows is not None else '[n/a]'}"
            )
        else:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batched close finished without pose_to_numpy_rows; "
                "per-env final close diagnostics are unavailable."
                f"{_R}"
            )
    grasp_flag = None
    if target_object is not None:
        grasp_flag, _, _, hand_min, hand_max = log_debug_post_close_retention(
            env_unwrapped, target_object, object_p_before_close, int(close_steps)
        )
        final_hand_qpos = get_robot_hand_qpos_debug(env_unwrapped)
    else:
        final_hand_qpos = get_robot_hand_qpos_debug(env_unwrapped)
        hand_min, hand_max = get_robot_hand_range_debug(env_unwrapped)
    save_planner_grasp_state(
        env_unwrapped,
        target_hand_qpos=close_target_qpos,
        realized_hand_qpos=final_hand_qpos,
        grasp_flag=grasp_flag,
        grasp_flag_rows=grasp_flag_rows_for_state,
        object_id=str(getattr(env_unwrapped, "manip_object_id", None)) if target_object is not None else None,
        source_stage="close",
    )
    print(
        f"[PlannerDebug] Proxy close final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
        f"hand_range=[{hand_min:.4f}, {hand_max:.4f}]"
    )
    refresh_render_state(env)
    print("[PlannerDebug] Proxy close EXECUTE OK")
    return True


def maybe_seed_proxy_start_pose(env, *, reason: str, get_hybrid_jointspace_control_mode, temporary_agent_control_mode, get_robot_qpos):
    env_unwrapped = env.unwrapped
    control_mode = getattr(env_unwrapped, "control_mode", None)
    if not is_ee_delta_control_mode(control_mode):
        return False
    arm_names = list(getattr(env_unwrapped.agent, "arm_joint_names", []))
    if len(arm_names) == 0:
        return False
    current_qpos = get_robot_qpos(env)
    arm_qpos = np.asarray(current_qpos[: len(arm_names)], dtype=np.float32)
    if float(np.max(np.abs(arm_qpos))) > 0.05:
        return False
    planner_cfg = get_debug_planner_config(env_unwrapped)
    seed_arm_qpos = planner_cfg.get("planner_proxy_seed_arm_qpos", None)
    if seed_arm_qpos is None:
        if len(arm_names) == 6:
            seed_arm_qpos = [0.0, 0.4, -0.2, 0.0, 1.57, 0.0]
        else:
            return False
    seed_arm_qpos = np.asarray(seed_arm_qpos, dtype=np.float32).reshape(-1)[: len(arm_names)]
    jointspace_mode = get_hybrid_jointspace_control_mode(env_unwrapped)
    seed_steps = max(int(planner_cfg.get("planner_proxy_seed_steps", 15) or 15), 1)
    print(
        f"[PlannerDebug] Proxy seed move before {reason}: "
        f"arm_qpos={np.array2string(arm_qpos, precision=4, suppress_small=True)} "
        f"target_arm_qpos={np.array2string(seed_arm_qpos, precision=4, suppress_small=True)} "
        f"steps={seed_steps}"
    )
    context = temporary_agent_control_mode(env, jointspace_mode, reason=f"proxy seed move for {reason}") if jointspace_mode is not None else nullcontext()
    with context:
        for alpha in np.linspace(0.0, 1.0, seed_steps + 1, dtype=np.float32)[1:]:
            interp_qpos = current_qpos.copy()
            interp_qpos[: len(arm_names)] = (1.0 - alpha) * arm_qpos + alpha * seed_arm_qpos
            _step_env(env, interp_qpos.astype(np.float32))
    final_qpos = get_robot_qpos(env)
    print(
        f"[PlannerDebug] Proxy seed move finished: "
        f"final_arm_qpos={np.array2string(final_qpos[:len(arm_names)], precision=4, suppress_small=True)}"
    )
    return True
