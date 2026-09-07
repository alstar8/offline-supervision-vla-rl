from __future__ import annotations

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose, to_sapien_pose as _to_sapien_pose
from openreal2sim.simulation.maniskill.planner_core import (
    RUNTIME_EE_WORLD,
    TASK_WORLD,
    SemanticPose,
    build_vertical_lift_pose,
    get_planner_grasp_state,
    select_lift_reference_pose,
)
from transforms3d.quaternions import qinverse, qmult

RC5_DEBUG_MOVE_GROUPS = {
    "right_tcp_link",
}
RC5_TARGET_SEMANTICS = {
    "right_tcp_link",
}
RC5_CANONICAL_TARGET_FRAME = "right_tcp_link"
_Y = "\033[33m"
_R = "\033[0m"
RC5_DEBUG_FINGER_LINK_CHAINS = (
    ("thumb", ("right_thumb_proximal_link", "right_thumb_distal_link", "right_thumb_tip_link")),
    ("index", ("right_index_proximal_link", "right_index_middle_link", "right_index_distal_link", "right_index_tip_link")),
    ("middle", ("right_middle_proximal_link", "right_middle_middle_link", "right_middle_distal_link", "right_middle_tip_link")),
    ("ring", ("right_ring_proximal_link", "right_ring_middle_link", "right_ring_distal_link", "right_ring_tip_link")),
    ("pinky", ("right_pinky_proximal_link", "right_pinky_middle_link", "right_pinky_distal_link", "right_pinky_tip_link")),
)


def is_rc5_agent_uid(agent_uid) -> bool:
    return str(agent_uid).startswith("rc5_aero_hand_openr2s")


def to_numpy_1d(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.astype(np.float32).copy()


def pose_to_numpy(pose):
    p = to_numpy_1d(pose.p)[:3]
    q = to_numpy_1d(pose.q)[:4]
    return p, q


def to_numpy_rows(value):
    if hasattr(value, "detach") and callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32).copy()


def pose_to_numpy_rows(pose):
    p = to_numpy_rows(pose.p)[:, :3]
    q = to_numpy_rows(pose.q)[:, :4]
    return p, q


def _select_first_env_pose(pose):
    if hasattr(pose, "raw_pose"):
        raw_pose = pose.raw_pose
        if isinstance(raw_pose, torch.Tensor) and raw_pose.ndim > 1:
            from mani_skill.utils.structs.pose import Pose

            return Pose(raw_pose=raw_pose[0])
    return pose


def get_rc5_debug_link_pose(base_env, link_name: str):
    agent = getattr(base_env, "agent", None)
    if agent is None or not link_name:
        return None
    if link_name == "right_tcp_link" and hasattr(agent, "tcp"):
        return _to_sapien_pose(_select_first_env_pose(agent.tcp.pose))
    robot = getattr(agent, "robot", None)
    links_map = getattr(robot, "links_map", None)
    if links_map is not None and link_name in links_map:
        return _to_sapien_pose(_select_first_env_pose(links_map[link_name].pose))
    return None


def get_rc5_debug_link_pose_rows(base_env, link_name: str):
    agent = getattr(base_env, "agent", None)
    if agent is None or not link_name:
        return None
    if link_name == "right_tcp_link" and hasattr(agent, "tcp"):
        return agent.tcp.pose
    robot = getattr(agent, "robot", None)
    links_map = getattr(robot, "links_map", None)
    if links_map is not None and link_name in links_map:
        return links_map[link_name].pose
    return None


def resolve_rc5_debug_move_group(base_env):
    planner_cfg = getattr(base_env, "_debug_planner_config", {}) or {}
    raw_move_group = str(planner_cfg.get("planner_rc5_move_group", RC5_CANONICAL_TARGET_FRAME)).strip()
    if raw_move_group not in RC5_DEBUG_MOVE_GROUPS:
        raise ValueError(
            f"Unsupported planner_rc5_move_group='{raw_move_group}'. "
            f"Unified RC5 runtime supports only '{RC5_CANONICAL_TARGET_FRAME}'."
        )
    return raw_move_group


def resolve_rc5_target_semantics(base_env, config_key: str, default_semantics: str, label: str):
    planner_cfg = getattr(base_env, "_debug_planner_config", {}) or {}
    raw_semantics = str(planner_cfg.get(config_key, default_semantics)).strip()
    if raw_semantics not in RC5_TARGET_SEMANTICS:
        raise ValueError(
            f"[RC5_LINK_DEBUG][{label}] Unsupported {config_key}='{raw_semantics}'. "
            f"Unified RC5 runtime supports only semantics={sorted(RC5_TARGET_SEMANTICS)}."
        )
    return raw_semantics


def _convert_rc5_pose_between_link_semantics(base_env, pose, source_link: str, target_link: str, label: str):
    pose = _to_sapien_pose(pose)
    source_pose = get_rc5_debug_link_pose(base_env, source_link)
    target_pose = get_rc5_debug_link_pose(base_env, target_link)
    if source_pose is None or target_pose is None:
        raise RuntimeError(
            f"[RC5_LINK_DEBUG][{label}] Cannot convert {source_link}->{target_link}: "
            f"source_pose_available={source_pose is not None}, target_pose_available={target_pose is not None}."
        )

    source_to_target = source_pose.inv() * target_pose
    converted_pose = pose * source_to_target
    delta_p = np.asarray(source_to_target.p, dtype=np.float32).reshape(-1)[:3]
    delta_q = np.asarray(source_to_target.q, dtype=np.float32).reshape(-1)[:4]
    print(
        f"[RC5_LINK_DEBUG][{label}] applying frame conversion {source_link}->{target_link} "
        f"delta_p={np.array2string(delta_p, precision=4, suppress_small=True)} "
        f"delta_q={np.array2string(delta_q, precision=4, suppress_small=True)}"
    )
    return converted_pose


def align_rc5_target_pose_to_active_move_group(base_env, target_pose, source_semantics: str, label: str):
    agent_uid = getattr(getattr(base_env, "agent", None), "uid", "")
    target_pose = _to_sapien_pose(target_pose)
    if not is_rc5_agent_uid(agent_uid):
        return target_pose

    move_group = resolve_rc5_debug_move_group(base_env)
    print(
        f"[RC5_LINK_DEBUG][{label}] target_source_semantics={source_semantics} "
        f"active_move_group={move_group}"
    )
    if source_semantics == move_group:
        return target_pose
    raise ValueError(
        f"[RC5_LINK_DEBUG][{label}] target_source_semantics='{source_semantics}' does not match "
        f"active_move_group='{move_group}'. Unified RC5 runtime does not allow semantic conversion."
    )


def get_debug_planner_ee_pose(base_env):
    agent = base_env.agent
    agent_uid = getattr(agent, "uid", "")
    if is_rc5_agent_uid(agent_uid):
        move_group = resolve_rc5_debug_move_group(base_env)
        move_group_pose = get_rc5_debug_link_pose(base_env, move_group)
        if move_group_pose is not None:
            return move_group_pose
        raise RuntimeError(
            f"[RC5_LINK_DEBUG] Active debug move_group '{move_group}' pose is unavailable."
        )
    return agent.tcp.pose


def get_debug_planner_ee_pose_rows(base_env):
    agent = base_env.agent
    agent_uid = getattr(agent, "uid", "")
    if is_rc5_agent_uid(agent_uid):
        move_group = resolve_rc5_debug_move_group(base_env)
        move_group_pose = get_rc5_debug_link_pose_rows(base_env, move_group)
        if move_group_pose is not None:
            return move_group_pose
        raise RuntimeError(
            f"[RC5_LINK_DEBUG] Active debug move_group '{move_group}' pose is unavailable."
        )
    return agent.tcp.pose


def get_debug_planner_ee_pose_sapien(base_env):
    return _to_sapien_pose(_select_first_env_pose(get_debug_planner_ee_pose(base_env)))


def get_debug_planner_retention_pose(base_env):
    agent = base_env.agent
    agent_uid = getattr(agent, "uid", "")
    if is_rc5_agent_uid(agent_uid):
        canonical_pose = get_rc5_debug_link_pose(base_env, RC5_CANONICAL_TARGET_FRAME)
        if canonical_pose is not None:
            return canonical_pose
        raise RuntimeError(
            f"[RC5_LINK_DEBUG] Canonical retention pose '{RC5_CANONICAL_TARGET_FRAME}' is unavailable."
        )
    return get_debug_planner_ee_pose(base_env)


def get_debug_planner_retention_pose_rows(base_env):
    agent = base_env.agent
    agent_uid = getattr(agent, "uid", "")
    if is_rc5_agent_uid(agent_uid):
        canonical_pose = get_rc5_debug_link_pose_rows(base_env, RC5_CANONICAL_TARGET_FRAME)
        if canonical_pose is not None:
            return canonical_pose
        raise RuntimeError(
            f"[RC5_LINK_DEBUG] Canonical retention pose '{RC5_CANONICAL_TARGET_FRAME}' is unavailable."
        )
    return get_debug_planner_ee_pose_rows(base_env)


def get_debug_planner_retention_pose_sapien(base_env):
    return _to_sapien_pose(_select_first_env_pose(get_debug_planner_retention_pose(base_env)))


def set_debug_planner_last_task_pose(base_env, pose, stage_name: str):
    base_env._planner_last_task_pose = pose
    base_env._planner_last_task_pose_stage = str(stage_name)


def build_debug_lift_pose_from_policy(base_env, lift_delta_z: float):
    agent_uid = getattr(base_env.agent, "uid", "")
    runtime_pose_raw = get_debug_planner_ee_pose_rows(base_env)
    runtime_is_batched = hasattr(runtime_pose_raw, "raw_pose") and len(getattr(runtime_pose_raw, "shape", ())) > 1
    runtime_pose = runtime_pose_raw if runtime_is_batched else _to_sapien_pose(runtime_pose_raw)
    last_task_pose = getattr(base_env, "_planner_last_task_pose", None)
    last_task_stage = getattr(base_env, "_planner_last_task_pose_stage", None) or "grasp"
    if last_task_pose is None:
        raise RuntimeError(
            "Lift pose policy requires an explicit last task pose from a completed pregrasp/descend stage. "
            "Unified RC5 runtime requires an explicit task pose and will not reuse the current runtime pose."
        )
    task_semantic_pose = SemanticPose(
        pose_world=last_task_pose if (hasattr(last_task_pose, "raw_pose") and len(getattr(last_task_pose, "shape", ())) > 1) else _to_sapien_pose(last_task_pose),
        semantics=TASK_WORLD,
        source_stage=str(last_task_stage),
    )
    runtime_semantic_pose = SemanticPose(
        pose_world=runtime_pose,
        semantics=RUNTIME_EE_WORLD,
        source_stage="close",
        already_robot_adapted=True,
    )
    reference_pose = select_lift_reference_pose(
        agent_uid,
        task_pose=task_semantic_pose,
        runtime_pose=runtime_semantic_pose,
    )
    if runtime_is_batched:
        reference_p_rows, reference_q_rows = pose_to_numpy_rows(reference_pose.pose_world)
        lifted_p_rows = reference_p_rows.copy()
        lifted_p_rows[:, 2] += float(lift_delta_z)
        lift_semantic_pose = reference_pose.with_updates(
            pose_world=Pose.create_from_pq(p=lifted_p_rows, q=reference_q_rows),
            source_stage="lift",
        )
    else:
        lift_semantic_pose = build_vertical_lift_pose(reference_pose, lift_delta_z)
    return lift_semantic_pose.pose_world, reference_pose, task_semantic_pose, runtime_semantic_pose


def get_robot_qpos(env):
    return to_numpy_1d(env.unwrapped.agent.robot.get_qpos())


def to_scalar_bool(value):
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        if arr.size == 0:
            return False
        return bool(arr.reshape(-1)[0])
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def get_robot_hand_qpos_debug(env_unwrapped):
    robot_qpos = env_unwrapped.agent.robot.get_qpos()
    if hasattr(robot_qpos, "cpu"):
        robot_qpos = robot_qpos.cpu().numpy()
    robot_qpos = np.asarray(robot_qpos)
    if robot_qpos.ndim > 1:
        robot_qpos = robot_qpos[0]
    hand_qpos_start = len(getattr(env_unwrapped.agent, "arm_joint_names", []))
    return np.asarray(robot_qpos[hand_qpos_start:], dtype=np.float32).copy()


def get_robot_hand_qpos_rows_debug(env_unwrapped):
    robot_qpos = env_unwrapped.agent.robot.get_qpos()
    if hasattr(robot_qpos, "cpu"):
        robot_qpos = robot_qpos.cpu().numpy()
    robot_qpos = np.asarray(robot_qpos)
    if robot_qpos.ndim == 1:
        robot_qpos = robot_qpos.reshape(1, -1)
    hand_qpos_start = len(getattr(env_unwrapped.agent, "arm_joint_names", []))
    return np.asarray(robot_qpos[:, hand_qpos_start:], dtype=np.float32).copy()


def get_robot_hand_range_debug(env_unwrapped):
    hand_qpos = get_robot_hand_qpos_debug(env_unwrapped)
    if hand_qpos.size == 0:
        return 0.0, 0.0
    return float(np.min(hand_qpos)), float(np.max(hand_qpos))


def _get_latched_hand_qpos_debug(env_unwrapped):
    grasp_state = get_planner_grasp_state(env_unwrapped)
    if grasp_state is None or grasp_state.target_hand_qpos is None:
        return None
    return np.asarray(grasp_state.target_hand_qpos, dtype=np.float32).reshape(-1).copy()


def _get_rc5_debug_finger_chain_links(agent):
    robot = getattr(agent, "robot", None)
    links_map = getattr(robot, "links_map", None)
    if not isinstance(links_map, dict):
        return None
    finger_links = {}
    for finger_name, link_names in RC5_DEBUG_FINGER_LINK_CHAINS:
        chain_links = []
        for link_name in link_names:
            link = links_map.get(link_name)
            if link is not None:
                chain_links.append((link_name, link))
        if chain_links:
            finger_links[finger_name] = chain_links
    return finger_links or None


def _peak_and_flag_from_forces(forces, min_force: float):
    if isinstance(forces, torch.Tensor):
        norms = torch.linalg.norm(forces, axis=1).detach().cpu().numpy()
    else:
        force_arr = np.asarray(forces)
        if force_arr.size == 0:
            norms = np.zeros((0,), dtype=np.float32)
        else:
            if force_arr.ndim == 1:
                force_arr = force_arr.reshape(1, -1)
            norms = np.linalg.norm(force_arr, axis=1)
    norms = np.asarray(norms, dtype=np.float32).reshape(-1)
    peak = float(np.max(norms)) if norms.size > 0 else 0.0
    active = bool(np.any(norms >= float(min_force)))
    return peak, active


def _get_rc5_contact_diagnostics_debug(env_unwrapped, target_object, min_force: float = 0.5):
    agent = env_unwrapped.agent
    if not hasattr(agent, "scene"):
        return None
    finger_links = _get_rc5_debug_finger_chain_links(agent)
    if finger_links is None:
        return None

    diagnostics = {
        "min_force": float(min_force),
        "finger_chains": {},
    }
    for finger_name, chain_links in finger_links.items():
        link_entries = []
        for link_name, link in chain_links:
            peak, active = _peak_and_flag_from_forces(
                agent.scene.get_pairwise_contact_forces(link, target_object),
                min_force=min_force,
            )
            link_entries.append(
                {
                    "link_name": str(link_name),
                    "peak": peak,
                    "active": active,
                }
            )
        if not link_entries:
            continue
        tip_entry = next((entry for entry in link_entries if entry["link_name"].endswith("_tip_link")), link_entries[-1])
        diagnostics["finger_chains"][finger_name] = {
            "any_active": bool(any(entry["active"] for entry in link_entries)),
            "chain_peak": float(max(entry["peak"] for entry in link_entries)),
            "links": link_entries,
        }
        diagnostics[f"{finger_name}_peak"] = float(tip_entry["peak"])
        diagnostics[f"{finger_name}_active"] = bool(tip_entry["active"])
    return diagnostics if diagnostics["finger_chains"] else None


def _get_rc5_finger_pose_diagnostics_debug(env_unwrapped, target_object):
    agent = env_unwrapped.agent
    finger_links = _get_rc5_debug_finger_chain_links(agent)
    if finger_links is None:
        return None

    object_p = get_debug_actor_position_xyz(target_object)

    def _link_position_xyz(link):
        pose = getattr(link, "pose", None)
        if pose is None:
            return None
        link_p, _ = pose_to_numpy(pose)
        return np.asarray(link_p, dtype=np.float32).reshape(-1)[:3]

    diagnostics = {
        "object_p": np.asarray(object_p, dtype=np.float32).reshape(-1)[:3],
        "finger_chains": {},
    }
    for finger_name, chain_links in finger_links.items():
        link_entries = []
        for link_name, link in chain_links:
            link_p = _link_position_xyz(link)
            if link_p is None:
                continue
            object_dist = float(
                np.linalg.norm(
                    np.asarray(link_p, dtype=np.float32) - np.asarray(object_p, dtype=np.float32)
                )
            )
            link_entries.append(
                {
                    "link_name": str(link_name),
                    "position": np.asarray(link_p, dtype=np.float32).reshape(-1)[:3],
                    "object_dist": object_dist,
                }
            )
        if not link_entries:
            continue
        tip_entry = next((entry for entry in link_entries if entry["link_name"].endswith("_tip_link")), link_entries[-1])
        diagnostics["finger_chains"][finger_name] = {
            "nearest_object_dist": float(min(entry["object_dist"] for entry in link_entries)),
            "links": link_entries,
        }
        diagnostics[f"{finger_name}_p"] = np.asarray(tip_entry["position"], dtype=np.float32).reshape(-1)[:3]
        diagnostics[f"{finger_name}_object_dist"] = float(tip_entry["object_dist"])
    return diagnostics if diagnostics["finger_chains"] else None


def _format_rc5_contact_chain_debug_line(contact_diag, finger_name: str):
    finger_diag = (contact_diag or {}).get("finger_chains", {}).get(finger_name)
    if finger_diag is None:
        return None
    chain_entries = " ".join(
        f"{entry['link_name']}={entry['peak']:.3f}/{entry['active']}"
        for entry in finger_diag["links"]
    )
    return (
        f"[PlannerDebug] RC5 {finger_name}_chain_contact: "
        f"any_active={finger_diag['any_active']} "
        f"chain_peak={finger_diag['chain_peak']:.3f} "
        f"{chain_entries}"
    )


def _format_rc5_pose_chain_positions_debug_line(prefix: str, pose_diag, finger_name: str):
    finger_diag = (pose_diag or {}).get("finger_chains", {}).get(finger_name)
    if finger_diag is None:
        return None
    chain_entries = " ".join(
        f"{entry['link_name']}={np.array2string(entry['position'], precision=4, suppress_small=True)}"
        for entry in finger_diag["links"]
    )
    return (
        f"[PlannerDebug] {prefix}_{finger_name}_chain_positions: "
        f"object_p={np.array2string(pose_diag['object_p'], precision=4, suppress_small=True)} "
        f"{chain_entries}"
    )


def _format_rc5_pose_chain_distances_debug_line(prefix: str, pose_diag, finger_name: str):
    finger_diag = (pose_diag or {}).get("finger_chains", {}).get(finger_name)
    if finger_diag is None:
        return None
    chain_entries = " ".join(
        f"{entry['link_name']}={entry['object_dist']:.4f}"
        for entry in finger_diag["links"]
    )
    return (
        f"[PlannerDebug] {prefix}_{finger_name}_chain_object_dist: "
        f"nearest={finger_diag['nearest_object_dist']:.4f} "
        f"{chain_entries}"
    )


def _link_has_collision_shapes_debug(link) -> bool:
    raw_link_objs = getattr(link, "_objs", None)
    if not raw_link_objs:
        return False
    for raw_link in raw_link_objs:
        get_shapes = getattr(raw_link, "get_collision_shapes", None)
        shapes = get_shapes() if callable(get_shapes) else getattr(raw_link, "collision_shapes", [])
        if shapes:
            return True
    return False


def _get_rc5_non_target_contact_monitor_links(agent):
    robot = getattr(agent, "robot", None)
    if robot is None:
        return ()
    links = robot.get_links() if hasattr(robot, "get_links") else getattr(robot, "links", [])
    if not links:
        return ()
    preferred_tokens = (
        "right_",
        "hand",
        "palm",
        "prehand",
        "camera",
        "tcp",
    )
    selected = []
    for link in links:
        link_name = str(getattr(link, "name", "") or "")
        if not link_name:
            continue
        if not _link_has_collision_shapes_debug(link):
            continue
        if any(token in link_name for token in preferred_tokens):
            selected.append((link_name, link))
    if selected:
        return tuple(selected)
    fallback = []
    for link in links:
        link_name = str(getattr(link, "name", "") or "")
        if not link_name:
            continue
        if _link_has_collision_shapes_debug(link):
            fallback.append((link_name, link))
    return tuple(fallback)


def _classify_non_target_contact_link_debug(link_name: str) -> str:
    normalized = str(link_name or "").lower()
    if "prehand" in normalized or "camera" in normalized:
        return "PREHAND_CAMERA"
    if "palm" in normalized or "hand" in normalized:
        return "HAND_BODY"
    if "tcp" in normalized:
        return "TCP"
    if "thumb" in normalized or "index" in normalized or "middle" in normalized or "ring" in normalized or "pinky" in normalized:
        return "FINGER"
    return "ROBOT_LINK"


def _summarize_pairwise_contact_force_debug(forces, *, num_envs: int):
    if isinstance(forces, torch.Tensor):
        force_arr = forces.detach().cpu().numpy()
    else:
        force_arr = np.asarray(forces)
    force_arr = np.asarray(force_arr, dtype=np.float32)
    if force_arr.size == 0:
        return 0.0, ()
    if force_arr.shape[-1:] == (3,):
        norms = np.linalg.norm(force_arr, axis=-1)
    else:
        norms = np.abs(force_arr)
    norms = np.asarray(norms, dtype=np.float32)
    if norms.size == 0:
        return 0.0, ()
    global_peak = float(np.max(norms))
    env_count = int(max(num_envs, 1))
    if env_count <= 1:
        active_envs = (0,) if global_peak > 0.0 else ()
        return global_peak, active_envs
    if norms.ndim >= 1 and norms.shape[0] == env_count:
        per_env_peak = np.max(norms.reshape(env_count, -1), axis=1)
        active_envs = tuple(int(idx) for idx in np.flatnonzero(per_env_peak > 0.0).tolist())
        return global_peak, active_envs
    return global_peak, tuple(range(env_count))


def collect_debug_robot_object_contacts(
    env_unwrapped,
    *,
    min_force: float = 1e-8,
    include_target: bool = True,
):
    agent = getattr(env_unwrapped, "agent", None)
    scene = getattr(agent, "scene", None)
    if agent is None or scene is None:
        return []
    monitor_links = _get_rc5_non_target_contact_monitor_links(agent)
    if not monitor_links:
        return []
    object_actors = getattr(env_unwrapped, "object_actors", {}) or {}
    manip_object_id = str(getattr(env_unwrapped, "manip_object_id", None) or "")
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    records = []
    for object_id, actor in object_actors.items():
        object_id = str(object_id)
        if not include_target and object_id == manip_object_id:
            continue
        for link_name, link in monitor_links:
            try:
                forces = scene.get_pairwise_contact_forces(link, actor)
            except Exception:
                continue
            peak_force, active_envs = _summarize_pairwise_contact_force_debug(
                forces,
                num_envs=num_envs,
            )
            if peak_force < min_force:
                continue
            records.append(
                {
                    "kind": "robot_object",
                    "object_id": object_id,
                    "is_target": bool(object_id == manip_object_id),
                    "link": str(link_name),
                    "class": _classify_non_target_contact_link_debug(link_name),
                    "peak_force": float(peak_force),
                    "env_ids": [int(idx) for idx in active_envs],
                }
            )
    return records


def collect_debug_object_object_contacts(env_unwrapped, *, min_force: float = 1e-8):
    agent = getattr(env_unwrapped, "agent", None)
    scene = getattr(agent, "scene", None)
    if scene is None:
        scene = getattr(env_unwrapped, "scene", None)
    if scene is None:
        return []
    object_actors = getattr(env_unwrapped, "object_actors", {}) or {}
    sorted_items = sorted(object_actors.items(), key=lambda item: str(item[0]))
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    records = []
    for left_index, (left_id, left_actor) in enumerate(sorted_items):
        for right_id, right_actor in sorted_items[left_index + 1:]:
            try:
                forces = scene.get_pairwise_contact_forces(left_actor, right_actor)
            except Exception:
                continue
            peak_force, active_envs = _summarize_pairwise_contact_force_debug(
                forces,
                num_envs=num_envs,
            )
            if peak_force < min_force:
                continue
            records.append(
                {
                    "kind": "object_object",
                    "object_id_a": str(left_id),
                    "object_id_b": str(right_id),
                    "peak_force": float(peak_force),
                    "env_ids": [int(idx) for idx in active_envs],
                }
            )
    return records


def log_debug_non_target_object_contacts(env_unwrapped, *, stage_label: str, min_force: float = 1e-8):
    planner_cfg = getattr(env_unwrapped, "_debug_planner_config", {}) or {}
    if planner_cfg.get("planner_proxy_warn_non_target_contacts", True) is False:
        return
    configured_min_force = planner_cfg.get("planner_proxy_warn_non_target_contacts_min_force", min_force)
    if configured_min_force is None:
        return
    min_force = float(configured_min_force)
    contact_records = collect_debug_robot_object_contacts(
        env_unwrapped,
        min_force=min_force,
        include_target=False,
    )
    for record in contact_records:
        active_env_suffix = ""
        if record["env_ids"]:
            active_env_suffix = f" env_ids={record['env_ids']}"
        print(
            f"{_Y}[WARNING] [PlannerDebug] Non-target robot contact: "
            f"stage={stage_label} "
            f"class={record['class']} "
            f"link={record['link']} "
            f"object_id={record['object_id']} "
            f"peak_force={record['peak_force']:.8f} "
            f"min_force={min_force:.8f}"
            f"{active_env_suffix}{_R}"
        )


def get_debug_target_object(env_unwrapped):
    object_actors = getattr(env_unwrapped, "object_actors", {}) or {}
    manip_id = getattr(env_unwrapped, "manip_object_id", None)
    available_ids = [str(obj_id) for obj_id in object_actors.keys()]
    if manip_id is None:
        msg = (
            "[PlannerDebug] manip_object_id is not set; refusing to guess a target object. "
            f"Available object_actors={available_ids}"
        )
        print(f"{_Y}[WARNING] {msg}{_R}", flush=True)
        raise RuntimeError(msg)
    target = object_actors.get(str(manip_id)) or object_actors.get(manip_id)
    if target is None:
        msg = (
            f"[PlannerDebug] manip_object_id='{manip_id}' is not present in object_actors; "
            f"available={available_ids}"
        )
        print(f"{_Y}[WARNING] {msg}{_R}", flush=True)
        raise RuntimeError(msg)
    return target


def get_debug_actor_position_xyz(actor):
    pose = actor.pose
    if hasattr(pose, "raw_pose"):
        raw_pose = pose.raw_pose
        if isinstance(raw_pose, torch.Tensor) and raw_pose.ndim > 1:
            from mani_skill.utils.structs.pose import Pose

            pose = Pose(raw_pose=raw_pose[0])
    pose = _to_sapien_pose(pose)
    return np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]


def get_debug_actor_position_rows(actor):
    p_rows, _ = pose_to_numpy_rows(actor.pose)
    return p_rows


def log_debug_scene_object_layout(env_unwrapped, *, stage_label: str = "post_reset"):
    object_actors = getattr(env_unwrapped, "object_actors", {}) or {}
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    manip_object_id = str(getattr(env_unwrapped, "manip_object_id", None) or "")
    print(
        f"[PlannerDebug] SceneState begin: stage={stage_label} "
        f"num_envs={num_envs} manip_object_id={manip_object_id or '<unset>'} "
        f"object_count={len(object_actors)}"
    )
    if not object_actors:
        print(f"{_Y}[WARNING] [PlannerDebug] SceneState: no object_actors available{_R}")
        return
    sorted_items = sorted(object_actors.items(), key=lambda item: str(item[0]))
    for object_id, actor in sorted_items:
        object_id = str(object_id)
        try:
            p_rows, q_rows = pose_to_numpy_rows(actor.pose)
        except Exception as exc:
            print(
                f"{_Y}[WARNING] [PlannerDebug] SceneState: failed to read pose for object_id={object_id}: "
                f"{exc!r}{_R}"
            )
            continue
        row_count = int(min(len(p_rows), len(q_rows)))
        if row_count <= 0:
            print(
                f"{_Y}[WARNING] [PlannerDebug] SceneState: empty pose rows for object_id={object_id}{_R}"
            )
            continue
        for env_index in range(row_count):
            print(
                f"[PlannerDebug] SceneState env_index={env_index} "
                f"object_id={object_id} "
                f"actor_p={np.array2string(p_rows[env_index], precision=4, suppress_small=True)} "
                f"actor_q={np.array2string(q_rows[env_index], precision=4, suppress_small=True)}"
            )
    print(f"[PlannerDebug] SceneState end: stage={stage_label}")


def log_debug_pre_close_snapshot(env_unwrapped, target_object, close_steps: int):
    ee_p, ee_q = pose_to_numpy(get_debug_planner_ee_pose(env_unwrapped))
    object_p = get_debug_actor_position_xyz(target_object)
    hand_qpos = get_robot_hand_qpos_debug(env_unwrapped)
    hand_min, hand_max = get_robot_hand_range_debug(env_unwrapped)
    latched_hand_qpos = _get_latched_hand_qpos_debug(env_unwrapped)
    print(
        f"[PlannerDebug] before close_gripper: "
        f"tcp_p={np.array2string(ee_p, precision=4, suppress_small=True)} "
        f"tcp_q={np.array2string(ee_q, precision=4, suppress_small=True)} "
        f"object_p={np.array2string(object_p, precision=4, suppress_small=True)} "
        f"close_steps={close_steps} hand_range=[{hand_min:.4f}, {hand_max:.4f}]"
    )
    print(
        f"[PlannerDebug] preclose_hand_qpos="
        f"{np.array2string(hand_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    if latched_hand_qpos is not None:
        print(
            f"[PlannerDebug] preclose_latched_hand_qpos="
            f"{np.array2string(latched_hand_qpos, precision=4, suppress_small=True, max_line_width=200)}"
        )
    tip_pose_diag = _get_rc5_finger_pose_diagnostics_debug(env_unwrapped, target_object)
    if tip_pose_diag is not None:
        print(
            "[PlannerDebug] preclose_tip_positions: "
            f"object_p={np.array2string(tip_pose_diag['object_p'], precision=4, suppress_small=True)} "
            f"thumb_p={np.array2string(tip_pose_diag['thumb_p'], precision=4, suppress_small=True)} "
            f"index_p={np.array2string(tip_pose_diag['index_p'], precision=4, suppress_small=True)} "
            f"middle_p={np.array2string(tip_pose_diag['middle_p'], precision=4, suppress_small=True)}"
        )
        print(
            "[PlannerDebug] preclose_tip_object_dist: "
            f"thumb={tip_pose_diag['thumb_object_dist']:.4f} "
            f"index={tip_pose_diag['index_object_dist']:.4f} "
            f"middle={tip_pose_diag['middle_object_dist']:.4f} "
            f"ring={tip_pose_diag['ring_object_dist']:.4f} "
            f"pinky={tip_pose_diag['pinky_object_dist']:.4f}"
        )
        for finger_name, _chain in RC5_DEBUG_FINGER_LINK_CHAINS:
            positions_line = _format_rc5_pose_chain_positions_debug_line("preclose", tip_pose_diag, finger_name)
            if positions_line is not None:
                print(positions_line)
            distances_line = _format_rc5_pose_chain_distances_debug_line("preclose", tip_pose_diag, finger_name)
            if distances_line is not None:
                print(distances_line)


def log_debug_post_close_retention(env_unwrapped, target_object, object_p_before_close, close_steps: int):
    object_p_after_close = get_debug_actor_position_xyz(target_object)
    grasp_flag = to_scalar_bool(env_unwrapped.agent.is_grasping(target_object))
    object_delta = np.asarray(object_p_after_close - object_p_before_close, dtype=np.float32).reshape(-1)[:3]
    tcp_pose_after_close = get_debug_planner_ee_pose_sapien(env_unwrapped)
    tcp_p_after_close, _tcp_q_after_close = pose_to_numpy(tcp_pose_after_close)
    object_minus_tcp = np.asarray(object_p_after_close - tcp_p_after_close, dtype=np.float32).reshape(-1)[:3]
    object_tcp_dist = float(np.linalg.norm(object_minus_tcp))
    final_hand_qpos = get_robot_hand_qpos_debug(env_unwrapped)
    hand_min, hand_max = get_robot_hand_range_debug(env_unwrapped)
    latched_hand_qpos = _get_latched_hand_qpos_debug(env_unwrapped)
    contact_diag = _get_rc5_contact_diagnostics_debug(env_unwrapped, target_object, min_force=0.5)
    print(
        f"[PlannerDebug] after close_gripper: is_grasping={grasp_flag} "
        f"object_p={np.array2string(object_p_after_close, precision=4, suppress_small=True)} "
        f"object_delta={np.array2string(object_delta, precision=4, suppress_small=True)} "
        f"tcp_p={np.array2string(tcp_p_after_close, precision=4, suppress_small=True)} "
        f"object_minus_tcp={np.array2string(object_minus_tcp, precision=4, suppress_small=True)} "
        f"object_tcp_dist={object_tcp_dist:.4f} "
        f"close_steps={close_steps} hand_range=[{hand_min:.4f}, {hand_max:.4f}]"
    )
    print(
        f"[PlannerDebug] final_hand_qpos="
        f"{np.array2string(final_hand_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    if latched_hand_qpos is not None:
        print(
            f"[PlannerDebug] latched_hand_qpos="
            f"{np.array2string(latched_hand_qpos, precision=4, suppress_small=True, max_line_width=200)}"
        )
        print(
            f"[PlannerDebug] Latched planner hand target range="
            f"[{float(np.min(latched_hand_qpos)):.4f}, {float(np.max(latched_hand_qpos)):.4f}]"
        )
    if contact_diag is not None:
        print(
            "[PlannerDebug] RC5 contact retention (tip heuristic): "
            f"thumb={contact_diag['thumb_peak']:.3f}/{contact_diag['thumb_active']} "
            f"index={contact_diag['index_peak']:.3f}/{contact_diag['index_active']} "
            f"middle={contact_diag['middle_peak']:.3f}/{contact_diag['middle_active']} "
            f"ring={contact_diag['ring_peak']:.3f}/{contact_diag['ring_active']} "
            f"pinky={contact_diag['pinky_peak']:.3f}/{contact_diag['pinky_active']} "
            f"min_force={contact_diag['min_force']:.3f}"
        )
        for finger_name, _chain in RC5_DEBUG_FINGER_LINK_CHAINS:
            contact_line = _format_rc5_contact_chain_debug_line(contact_diag, finger_name)
            if contact_line is not None:
                print(contact_line)
    tip_pose_diag = _get_rc5_finger_pose_diagnostics_debug(env_unwrapped, target_object)
    if tip_pose_diag is not None:
        print(
            "[PlannerDebug] postclose_tip_positions: "
            f"object_p={np.array2string(tip_pose_diag['object_p'], precision=4, suppress_small=True)} "
            f"thumb_p={np.array2string(tip_pose_diag['thumb_p'], precision=4, suppress_small=True)} "
            f"index_p={np.array2string(tip_pose_diag['index_p'], precision=4, suppress_small=True)} "
            f"middle_p={np.array2string(tip_pose_diag['middle_p'], precision=4, suppress_small=True)}"
        )
        print(
            "[PlannerDebug] postclose_tip_object_dist: "
            f"thumb={tip_pose_diag['thumb_object_dist']:.4f} "
            f"index={tip_pose_diag['index_object_dist']:.4f} "
            f"middle={tip_pose_diag['middle_object_dist']:.4f} "
            f"ring={tip_pose_diag['ring_object_dist']:.4f} "
            f"pinky={tip_pose_diag['pinky_object_dist']:.4f}"
        )
        for finger_name, _chain in RC5_DEBUG_FINGER_LINK_CHAINS:
            positions_line = _format_rc5_pose_chain_positions_debug_line("postclose", tip_pose_diag, finger_name)
            if positions_line is not None:
                print(positions_line)
            distances_line = _format_rc5_pose_chain_distances_debug_line("postclose", tip_pose_diag, finger_name)
            if distances_line is not None:
                print(distances_line)
    return grasp_flag, object_p_after_close, object_delta, hand_min, hand_max


def refresh_render_state(env, viewer=None):
    if viewer is not None:
        try:
            env.render_human()
            return
        except Exception:
            pass


def get_object_specific_planner_profile(planner_cfg, manip_id):
    profiles = planner_cfg.get("planner_object_calibrations", None)
    if not isinstance(profiles, dict):
        return None
    profile = profiles.get(str(manip_id))
    return profile if isinstance(profile, dict) else None


def extract_planner_base_pose(env):
    robot_pose = env.unwrapped.agent.robot.pose
    if hasattr(robot_pose, "raw_pose"):
        raw_pose = robot_pose.raw_pose
        if isinstance(raw_pose, torch.Tensor) and raw_pose.ndim > 1 and raw_pose.shape[0] > 1:
            from mani_skill.utils.structs.pose import Pose

            return Pose(raw_pose=raw_pose[0])
    return robot_pose


def planner_visuals_supported(env) -> bool:
    device = str(getattr(env.unwrapped, "device", ""))
    return "cuda" not in device.lower()


def get_planner_recording_kwargs(env):
    base_env = env.unwrapped
    return dict(
        record_frames=getattr(base_env, "_planner_video_frames", None),
        normalize_frame_fn=getattr(base_env, "_planner_video_normalize_fn", None),
        record_frames_from=getattr(base_env, "_planner_video_record_from", "base_camera"),
    )


def configure_debug_planner_solver_runtime(solver):
    solver.auto_continue_time = 0.0
    return solver


def resolve_planner_debug_solver_class(agent_uid):
    from motion.planner_debug_variant import (
        PandaArmMotionPlanningSolver,
        RC5ArmMotionPlanningSolver,
        WidowXArmMotionPlanningSolver,
    )

    if agent_uid in ("widowx250s_openr2s", "widowx250s_bridgedataset_flat_table_openr2s"):
        return WidowXArmMotionPlanningSolver
    if is_rc5_agent_uid(agent_uid):
        return RC5ArmMotionPlanningSolver
    return PandaArmMotionPlanningSolver
