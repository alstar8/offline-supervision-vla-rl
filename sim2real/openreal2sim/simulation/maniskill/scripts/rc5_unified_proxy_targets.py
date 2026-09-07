from __future__ import annotations

import numpy as np
import sapien
from mani_skill.utils.structs.pose import Pose

_Y = "\033[33m"
_R = "\033[0m"
RC5_CANONICAL_TARGET_FRAME = "right_tcp_link"
VALID_OBJECT_TARGET_ORIENTATION_MODES = ("fixed", "current_tcp")


def resolve_object_target_orientation_mode(object_profile, *, object_id: str) -> str:
    mode = object_profile.get("target_orientation_mode")
    if mode is None:
        mode = "fixed"
    mode = str(mode).strip().lower()
    if mode not in VALID_OBJECT_TARGET_ORIENTATION_MODES:
        raise RuntimeError(
            f"target_orientation_mode for '{object_id}' must be one of "
            f"{VALID_OBJECT_TARGET_ORIENTATION_MODES}, got {mode!r}."
        )
    return mode


def resolve_object_target_quaternion(object_profile, current_tcp_q, *, object_id: str):
    mode = resolve_object_target_orientation_mode(object_profile, object_id=object_id)

    if mode == "current_tcp":
        if "target_quat" in object_profile:
            raise RuntimeError(
                f"Object profile '{object_id}' with target_orientation_mode='current_tcp' "
                "must not define target_quat."
            )
        raw_quaternion = current_tcp_q
    else:
        if "target_quat" not in object_profile:
            raise RuntimeError(
                f"Object profile '{object_id}' with target_orientation_mode='fixed' "
                "requires target_quat."
            )
        raw_quaternion = object_profile["target_quat"]

    quaternion = np.asarray(raw_quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError(
            f"Resolved target quaternion for '{object_id}' must contain four finite values."
        )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise RuntimeError(f"Resolved target quaternion for '{object_id}' has zero norm.")
    return (quaternion / norm).astype(np.float32), mode


def _get_manip_actor_info(base_env, *, get_debug_target_object, pose_to_numpy, pose_to_numpy_rows=None):
    manip_actor = get_debug_target_object(base_env)
    manip_id = str(getattr(base_env, "manip_object_id", None))
    if callable(pose_to_numpy_rows):
        actor_p_rows, _actor_q_rows = pose_to_numpy_rows(manip_actor.pose)
        actor_p = actor_p_rows if actor_p_rows.shape[0] > 1 else actor_p_rows[0]
    else:
        actor_p, _ = pose_to_numpy(manip_actor.pose)
    bbox_world = base_env._get_actor_bbox_world(manip_id, manip_actor)
    if bbox_world is not None:
        if hasattr(bbox_world, "detach"):
            bbox_world = bbox_world.detach()
        if hasattr(bbox_world, "cpu"):
            bbox_world = bbox_world.cpu().numpy()
        bbox_np = np.asarray(bbox_world, dtype=np.float32).reshape(-1)[:3]
    else:
        bbox_np = np.array([0.06, 0.06, 0.06], dtype=np.float32)
    return manip_id, actor_p, bbox_np


def build_object_pregrasp_target(
    env,
    *,
    extra_clearance=0.10,
    radial_backoff_override=None,
    get_debug_target_object,
    get_object_specific_planner_profile,
    resolve_rc5_target_semantics,
    align_rc5_target_pose_to_active_move_group,
    pose_to_numpy,
    pose_to_numpy_rows=None,
    get_debug_planner_ee_pose,
    is_rc5_debug_planner_agent,
):
    del extra_clearance, radial_backoff_override
    base_env = env.unwrapped
    planner_cfg = getattr(base_env, "_debug_planner_config", {}) or {}
    manip_id, actor_p, bbox_np = _get_manip_actor_info(
        base_env,
        get_debug_target_object=get_debug_target_object,
        pose_to_numpy=pose_to_numpy,
        pose_to_numpy_rows=pose_to_numpy_rows,
    )
    object_profile = get_object_specific_planner_profile(planner_cfg, manip_id)
    current_tcp_p_rows = None
    current_tcp_q_rows = None
    if callable(pose_to_numpy_rows):
        current_tcp_p_rows, current_tcp_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(base_env))
    current_tcp_p, current_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(base_env))
    if object_profile is not None:
        profile_target_semantics = resolve_rc5_target_semantics(
            base_env,
            config_key="planner_rc5_object_profile_target_semantics",
            default_semantics=RC5_CANONICAL_TARGET_FRAME,
            label="object_profile:pregrasp",
        )
        pregrasp_offset = np.asarray(
            object_profile.get("pregrasp_offset_xyz", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(-1)[:3]
        target_q, target_orientation_mode = resolve_object_target_quaternion(
            object_profile,
            current_tcp_q,
            object_id=manip_id,
        )
        if isinstance(actor_p, np.ndarray) and actor_p.ndim == 2:
            target_p = actor_p.copy()
            target_p[:, :2] += pregrasp_offset[:2]
            if current_tcp_p_rows is None:
                current_tcp_p_rows = np.repeat(current_tcp_p.reshape(1, 3), target_p.shape[0], axis=0)
            target_p[:, 2] = current_tcp_p_rows[:, 2] + float(pregrasp_offset[2])
            target_q_rows = np.repeat(target_q.reshape(1, 4), target_p.shape[0], axis=0)
            target_pose = Pose.create_from_pq(p=target_p, q=target_q_rows)
        else:
            target_p = actor_p.copy()
            target_p[:2] += pregrasp_offset[:2]
            target_p[2] = float(current_tcp_p[2] + float(pregrasp_offset[2]))
            target_pose = sapien.Pose(p=target_p, q=target_q)
        if is_rc5_debug_planner_agent(getattr(base_env.agent, "uid", "")):
            align_rc5_target_pose_to_active_move_group(
                base_env,
                sapien.Pose(
                    p=np.asarray(target_pose.p, dtype=np.float32).reshape(-1, 3)[0],
                    q=np.asarray(target_pose.q, dtype=np.float32).reshape(-1, 4)[0],
                ),
                source_semantics=profile_target_semantics,
                label="object_profile:pregrasp",
            )
        print(
            f"[PlannerDebug] Object-calibrated pregrasp: manip_object_id={manip_id} "
            f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
            f"pregrasp_offset_xyz={np.array2string(pregrasp_offset, precision=4, suppress_small=True)} "
            f"pregrasp_target_z_from_current_tcp={current_tcp_p[2]:.4f} "
            f"target_quat={np.array2string(target_q, precision=4, suppress_small=True)} "
            f"target_orientation_mode={target_orientation_mode} "
            f"target_semantics={profile_target_semantics} "
            f"canonical_target_frame={RC5_CANONICAL_TARGET_FRAME}"
        )
        return manip_id, actor_p, bbox_np, target_pose

    raise RuntimeError(
        f"[PlannerDebug] Missing object-specific pregrasp profile for manip_object_id='{manip_id}'. "
        "Unified RC5 runtime requires an explicit object-specific pregrasp profile."
    )


def build_object_descend_target(
    env,
    *,
    extra_clearance=0.03,
    get_debug_target_object,
    get_object_specific_planner_profile,
    resolve_rc5_target_semantics,
    align_rc5_target_pose_to_active_move_group,
    pose_to_numpy,
    pose_to_numpy_rows=None,
    get_debug_planner_ee_pose,
    is_rc5_debug_planner_agent,
):
    base_env = env.unwrapped
    planner_cfg = getattr(base_env, "_debug_planner_config", {}) or {}
    manip_id, actor_p, bbox_np = _get_manip_actor_info(
        base_env,
        get_debug_target_object=get_debug_target_object,
        pose_to_numpy=pose_to_numpy,
        pose_to_numpy_rows=pose_to_numpy_rows,
    )
    object_profile = get_object_specific_planner_profile(planner_cfg, manip_id)
    current_tcp_p, current_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(base_env))
    if object_profile is not None:
        profile_target_semantics = resolve_rc5_target_semantics(
            base_env,
            config_key="planner_rc5_object_profile_target_semantics",
            default_semantics=RC5_CANONICAL_TARGET_FRAME,
            label="object_profile:descend",
        )
        descend_offset = np.asarray(
            object_profile.get("descend_offset_xyz", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(-1)[:3]
        target_q, target_orientation_mode = resolve_object_target_quaternion(
            object_profile,
            current_tcp_q,
            object_id=manip_id,
        )
        if isinstance(actor_p, np.ndarray) and actor_p.ndim == 2:
            target_p = actor_p.copy() + descend_offset.reshape(1, 3)
            target_q_rows = np.repeat(target_q.reshape(1, 4), target_p.shape[0], axis=0)
            target_pose = Pose.create_from_pq(p=target_p, q=target_q_rows)
        else:
            target_p = actor_p.copy() + descend_offset
            target_pose = sapien.Pose(p=target_p, q=target_q)
        if is_rc5_debug_planner_agent(getattr(base_env.agent, "uid", "")):
            align_rc5_target_pose_to_active_move_group(
                base_env,
                sapien.Pose(
                    p=np.asarray(target_pose.p, dtype=np.float32).reshape(-1, 3)[0],
                    q=np.asarray(target_pose.q, dtype=np.float32).reshape(-1, 4)[0],
                ),
                source_semantics=profile_target_semantics,
                label="object_profile:descend",
            )
        target_minus_actor = np.asarray(target_p, dtype=np.float32) - np.asarray(actor_p, dtype=np.float32)
        print(
            f"[PlannerDebug] Object-calibrated descend: manip_object_id={manip_id} "
            f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
            f"descend_offset_xyz={np.array2string(descend_offset, precision=4, suppress_small=True)} "
            f"target_minus_actor={np.array2string(target_minus_actor, precision=4, suppress_small=True)} "
            f"target_quat={np.array2string(target_q, precision=4, suppress_small=True)} "
            f"target_orientation_mode={target_orientation_mode} "
            f"target_semantics={profile_target_semantics} "
            f"canonical_target_frame={RC5_CANONICAL_TARGET_FRAME}"
        )
        return manip_id, actor_p, bbox_np, target_pose

    raise RuntimeError(
        f"[PlannerDebug] Missing object-specific descend profile for manip_object_id='{manip_id}'. "
        "Unified RC5 runtime requires an explicit object-specific descend profile."
    )
