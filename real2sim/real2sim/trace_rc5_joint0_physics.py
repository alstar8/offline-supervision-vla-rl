from pathlib import Path

import numpy as np
import torch
import tyro
from dataclasses import dataclass

import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots.rc5.rc5 import RC5AeroHandRight
from real2sim.calibrate_rc5_pose import (
    Args as BaseArgs,
    DEFAULT_LOAD_STATE_FILE,
    _load_state_file,
    _reset_current_state,
    _save_snapshot_from_obs,
)
from real2sim.openreal2sim_validation import (
    observation_camera_lookat_state,
    observation_camera_override,
    pose_from_eye_target_roll,
)


CONTROL_MODE = "arm_pd_joint_pos_gripper_pd_joint_pos"


@dataclass
class Args(BaseArgs):
    image_prefix: str = "openreal2sim_rc5_joint0_trace"
    load_state_file: str = DEFAULT_LOAD_STATE_FILE
    control_mode_name: str = CONTROL_MODE
    joint_index: int = 0
    joint_delta_rad: float = 0.2
    secondary_joint_index: int = 2
    secondary_joint_delta_rad: float = 0.2
    trace_steps: int = 25
    trace_log_name: str = "openreal2sim_rc5_joint0_trace.log"
    compact_log: bool = True


def _tensor_to_numpy(data) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _format_array(values: np.ndarray, *, precision: int = 6) -> str:
    return np.array2string(
        np.asarray(values),
        precision=precision,
        separator=", ",
        suppress_small=False,
    )


def _format_pose(value) -> str:
    try:
        if hasattr(value, "raw_pose"):
            raw_pose = _tensor_to_numpy(value.raw_pose)[0].astype(np.float64)
            return _format_array(raw_pose)
        if hasattr(value, "p") and hasattr(value, "q"):
            p = np.asarray(value.p, dtype=np.float64).reshape(-1)
            q = np.asarray(value.q, dtype=np.float64).reshape(-1)
            return _format_array(np.concatenate([p, q]))
        if hasattr(value, "sp"):
            return _format_pose(value.sp)
    except Exception as exc:
        return f"<error:{exc}>"
    return str(value)


def _pose_to_numpy(value) -> np.ndarray | None:
    try:
        if hasattr(value, "raw_pose"):
            return _tensor_to_numpy(value.raw_pose)[0].astype(np.float64)
        if hasattr(value, "p") and hasattr(value, "q"):
            p = np.asarray(value.p, dtype=np.float64).reshape(-1)
            q = np.asarray(value.q, dtype=np.float64).reshape(-1)
            return np.concatenate([p, q])
        if hasattr(value, "sp"):
            return _pose_to_numpy(value.sp)
    except Exception:
        return None
    return None


def _format_scalar(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def _joint_tracking_error(target_qpos: np.ndarray | None, joint: int, qpos: np.ndarray) -> float | None:
    if target_qpos is None or joint >= len(target_qpos) or joint >= len(qpos):
        return None
    return float(target_qpos[joint] - qpos[joint])


def _joint_stuck(
    target_qpos: np.ndarray | None,
    joint: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    pos_diff_threshold: float = 1e-3,
    vel_threshold: float = 1e-4,
) -> bool | None:
    tracking_error = _joint_tracking_error(target_qpos, joint, qpos)
    if tracking_error is None or joint >= len(qvel):
        return None
    return abs(tracking_error) > pos_diff_threshold and abs(float(qvel[joint])) < vel_threshold


def _norm_or_none(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    return float(np.linalg.norm(values))


def _build_env(
    args: Args,
    camera_mode: str,
    camera_eye: np.ndarray | None = None,
    camera_target: np.ndarray | None = None,
    camera_roll_deg: float | None = None,
    camera_fov: float | None = None,
) -> BaseEnv:
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
        camera_cfg = observation_camera_override(camera_mode)

    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode=args.control_mode_name,
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": camera_cfg,
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids="rc5_aero_hand_right_openreal2sim_validation",
    )


def _sim_qpos(env: BaseEnv) -> np.ndarray:
    return _tensor_to_numpy(env.unwrapped.agent.robot.get_qpos())[0].astype(np.float64)


def _sim_qvel(env: BaseEnv) -> np.ndarray:
    return _tensor_to_numpy(env.unwrapped.agent.robot.get_qvel())[0].astype(np.float64)


def _sim_qf(env: BaseEnv) -> np.ndarray:
    return _tensor_to_numpy(env.unwrapped.agent.robot.get_qf())[0].astype(np.float64)


def _sim_tcp_pose(env: BaseEnv) -> np.ndarray:
    return _tensor_to_numpy(env.unwrapped.agent.ee_pose_at_robot_base.raw_pose)[0].astype(np.float64)


def _arm_controller(env: BaseEnv):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm")
    return controller


def _joint_axis_xyz(joint) -> str:
    axis = getattr(joint, "axis", None)
    if axis is None:
        return "unknown"
    try:
        return _format_array(_tensor_to_numpy(axis).reshape(-1))
    except Exception:
        return str(axis)


def _safe_attr(obj, name: str):
    try:
        value = getattr(obj, name)
        if callable(value):
            value = value()
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    except Exception as exc:
        return f"<error:{exc}>"


def _safe_pose(obj, name: str) -> str:
    try:
        value = getattr(obj, name)
        if callable(value):
            value = value()
        return _format_pose(value)
    except Exception as exc:
        return f"<error:{exc}>"


def _describe_joint(lines: list[str], prefix: str, joint) -> None:
    if joint is None:
        lines.append(f"{prefix}=None")
        return
    raw_joint = joint._objs[0]
    lines.extend(
        [
            f"{prefix}_name={joint.name}",
            f"{prefix}_axis={_joint_axis_xyz(joint)}",
            f"{prefix}_wrapper_type={type(joint).__name__}",
            f"{prefix}_raw_type={type(raw_joint).__name__}",
            f"{prefix}_wrapper_get_type={_safe_attr(joint, 'get_type')}",
            f"{prefix}_wrapper_drive_mode={_safe_attr(joint, 'get_drive_mode')}",
            f"{prefix}_wrapper_drive_target={_safe_attr(joint, 'get_drive_target')}",
            f"{prefix}_wrapper_stiffness={_safe_attr(joint, 'get_stiffness')}",
            f"{prefix}_wrapper_damping={_safe_attr(joint, 'get_damping')}",
            f"{prefix}_wrapper_force_limit={_safe_attr(joint, 'get_force_limit')}",
            f"{prefix}_wrapper_friction={_safe_attr(joint, 'get_friction')}",
            f"{prefix}_wrapper_limits={_safe_attr(joint, 'get_limits')}",
            f"{prefix}_wrapper_qpos={_safe_attr(joint, 'qpos')}",
            f"{prefix}_wrapper_qvel={_safe_attr(joint, 'qvel')}",
            f"{prefix}_wrapper_global_pose={_safe_pose(joint, 'get_global_pose')}",
            f"{prefix}_wrapper_pose_in_parent={_safe_pose(joint, 'get_pose_in_parent')}",
            f"{prefix}_wrapper_pose_in_child={_safe_pose(joint, 'get_pose_in_child')}",
            f"{prefix}_parent_link={joint.parent_link.name if joint.parent_link is not None else 'None'}",
            f"{prefix}_child_link={joint.child_link.name if joint.child_link is not None else 'None'}",
            f"{prefix}_parent_link_pose={_format_pose(joint.parent_link.pose) if joint.parent_link is not None else 'None'}",
            f"{prefix}_child_link_pose={_format_pose(joint.child_link.pose) if joint.child_link is not None else 'None'}",
            f"{prefix}_raw_joint_name={_safe_attr(raw_joint, 'name')}",
            f"{prefix}_raw_joint_type={_safe_attr(raw_joint, 'type')}",
            f"{prefix}_raw_joint_dof={_safe_attr(raw_joint, 'dof')}",
            f"{prefix}_raw_joint_drive_mode={_safe_attr(raw_joint, 'drive_mode')}",
            f"{prefix}_raw_joint_drive_target={_safe_attr(raw_joint, 'drive_target')}",
            f"{prefix}_raw_joint_stiffness={_safe_attr(raw_joint, 'stiffness')}",
            f"{prefix}_raw_joint_damping={_safe_attr(raw_joint, 'damping')}",
            f"{prefix}_raw_joint_force_limit={_safe_attr(raw_joint, 'force_limit')}",
            f"{prefix}_raw_joint_friction={_safe_attr(raw_joint, 'friction')}",
            f"{prefix}_raw_joint_limits={_safe_attr(raw_joint, 'limits')}",
            f"{prefix}_raw_joint_global_pose={_safe_pose(raw_joint, 'global_pose')}",
            f"{prefix}_raw_joint_pose_in_parent={_safe_pose(raw_joint, 'pose_in_parent')}",
            f"{prefix}_raw_joint_pose_in_child={_safe_pose(raw_joint, 'pose_in_child')}",
            f"{prefix}_raw_joint_parent_link={_safe_attr(_safe_attr(raw_joint, 'parent_link'), 'name')}",
            f"{prefix}_raw_joint_child_link={_safe_attr(_safe_attr(raw_joint, 'child_link'), 'name')}",
        ]
    )


def _describe_link(lines: list[str], prefix: str, link) -> None:
    if link is None:
        lines.append(f"{prefix}=None")
        return
    raw_link = link._objs[0]
    lines.extend(
        [
            f"{prefix}_name={link.name}",
            f"{prefix}_wrapper_type={type(link).__name__}",
            f"{prefix}_pose={_format_pose(link.pose)}",
            f"{prefix}_linear_velocity={_safe_attr(link, 'get_linear_velocity')}",
            f"{prefix}_angular_velocity={_safe_attr(link, 'get_angular_velocity')}",
            f"{prefix}_mass={_safe_attr(link, 'get_mass')}",
            f"{prefix}_cmass_local_pose={_safe_pose(link, 'cmass_local_pose')}",
            f"{prefix}_is_root={_safe_attr(link, 'is_root')}",
            f"{prefix}_joint_name={link.joint.name if getattr(link, 'joint', None) is not None else 'None'}",
            f"{prefix}_raw_name={_safe_attr(raw_link.entity, 'name')}",
            f"{prefix}_raw_pose={_safe_pose(raw_link, 'pose')}",
        ]
    )


def _joint_limits_for_arm(env: BaseEnv, arm_controller) -> np.ndarray:
    qlimits = _tensor_to_numpy(env.unwrapped.agent.robot.get_qlimits())[0]
    return qlimits[arm_controller.active_joint_indices.long().cpu().numpy()]


def _sim_passive_force(env: BaseEnv) -> np.ndarray | None:
    try:
        return np.asarray(env.unwrapped.agent.robot.compute_passive_force(), dtype=np.float64).reshape(-1)
    except Exception:
        return None


def _sim_contact_force(link) -> np.ndarray | None:
    if link is None:
        return None
    try:
        return _tensor_to_numpy(link.get_net_contact_forces())[0].astype(np.float64)
    except Exception:
        return None


def _sim_pairwise_contact_force(env: BaseEnv, link_a, link_b) -> np.ndarray | None:
    if link_a is None or link_b is None:
        return None
    try:
        return _tensor_to_numpy(env.unwrapped.scene.get_pairwise_contact_forces(link_a, link_b))[0].astype(
            np.float64
        )
    except Exception:
        return None


def _raw_link_bodies(link) -> list:
    if link is None:
        return []
    raw_link = link._objs[0]
    bodies = []
    for attr_name in ("_bodies", "bodies"):
        attr_value = getattr(raw_link, attr_name, None)
        if attr_value:
            bodies.extend(list(attr_value))
    if not bodies and hasattr(raw_link, "entity"):
        for component in getattr(raw_link.entity, "components", []):
            if hasattr(component, "get_collision_shapes"):
                bodies.append(component)
    return bodies


def _shape_type_name(shape) -> str:
    for attr_name in ("type", "geometry_type"):
        value = getattr(shape, attr_name, None)
        if value is not None:
            return str(value)
    geometry = getattr(shape, "geometry", None)
    if geometry is not None:
        return type(geometry).__name__
    return type(shape).__name__


def _shape_debug_line(shape, index: int) -> str:
    local_pose = None
    for attr_name in ("local_pose", "pose"):
        try:
            local_pose = getattr(shape, attr_name)
            break
        except Exception:
            continue
    geometry = getattr(shape, "geometry", None)
    geometry_desc = type(geometry).__name__ if geometry is not None else "unknown"
    return (
        f"shape[{index}](type={_shape_type_name(shape)}, geom={geometry_desc}, "
        f"local_pose={_format_pose(local_pose) if local_pose is not None else 'None'})"
    )


def _append_link_collision_shape_lines(lines: list[str], prefix: str, link) -> None:
    bodies = _raw_link_bodies(link)
    lines.append(f"{prefix}_body_count={len(bodies)}")
    for body_idx, body in enumerate(bodies):
        shapes = []
        try:
            shapes = list(body.get_collision_shapes())
        except Exception:
            shapes = []
        lines.append(f"{prefix}_body[{body_idx}]_shape_count={len(shapes)}")
        for shape_idx, shape in enumerate(shapes):
            lines.append(f"{prefix}_body[{body_idx}]_{_shape_debug_line(shape, shape_idx)}")


def _pairwise_cpu_contacts(env: BaseEnv, link_a, link_b) -> list[tuple[object, bool]]:
    if link_a is None or link_b is None or env.unwrapped.scene.gpu_sim_enabled:
        return []
    entity_a = link_a._objs[0].entity
    entity_b = link_b._objs[0].entity
    pair_contacts = []
    for contact in env.unwrapped.scene.get_contacts():
        if contact.bodies[0].entity == entity_a and contact.bodies[1].entity == entity_b:
            pair_contacts.append((contact, True))
        elif contact.bodies[0].entity == entity_b and contact.bodies[1].entity == entity_a:
            pair_contacts.append((contact, False))
    return pair_contacts


def _append_pairwise_contact_detail_lines(
    lines: list[str],
    *,
    label: str,
    env: BaseEnv,
    link_a,
    link_b,
    max_contacts: int = 4,
    max_points: int = 4,
) -> None:
    pair_contacts = _pairwise_cpu_contacts(env, link_a, link_b)
    lines.append(f"{label}_contact_count={len(pair_contacts)}")
    if not pair_contacts:
        return
    ranked_contacts = []
    for contact, a_is_body0 in pair_contacts:
        total_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        ranked_contacts.append((float(np.linalg.norm(total_impulse)), contact, a_is_body0, total_impulse))
    ranked_contacts.sort(key=lambda item: item[0], reverse=True)
    for contact_idx, (impulse_norm, contact, a_is_body0, total_impulse) in enumerate(
        ranked_contacts[:max_contacts]
    ):
        body_a = contact.bodies[0] if a_is_body0 else contact.bodies[1]
        body_b = contact.bodies[1] if a_is_body0 else contact.bodies[0]
        lines.extend(
            [
                f"{label}_contact[{contact_idx}]_body_a={getattr(body_a.entity, 'name', 'unknown')}",
                f"{label}_contact[{contact_idx}]_body_b={getattr(body_b.entity, 'name', 'unknown')}",
                f"{label}_contact[{contact_idx}]_point_count={len(contact.points)}",
                f"{label}_contact[{contact_idx}]_total_impulse={_format_array(total_impulse)}",
                f"{label}_contact[{contact_idx}]_total_force={_format_array(total_impulse / env.unwrapped.scene.timestep)}",
                f"{label}_contact[{contact_idx}]_impulse_norm={impulse_norm:.6f}",
            ]
        )
        for point_idx, point in enumerate(contact.points[:max_points]):
            position = getattr(point, "position", None)
            impulse = getattr(point, "impulse", None)
            normal = getattr(point, "normal", None)
            separation = getattr(point, "separation", None)
            lines.append(
                f"{label}_contact[{contact_idx}]_point[{point_idx}]="
                f"(position={_format_array(np.asarray(position, dtype=np.float64)) if position is not None else 'None'}, "
                f"impulse={_format_array(np.asarray(impulse, dtype=np.float64)) if impulse is not None else 'None'}, "
                f"normal={_format_array(np.asarray(normal, dtype=np.float64)) if normal is not None else 'None'}, "
                f"separation={separation})"
            )


def _append_joint_runtime_lines(
    lines: list[str],
    *,
    prefix: str,
    joint,
    joint_idx: int,
    target_qpos: np.ndarray | None,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qf: np.ndarray | None,
    passive_force: np.ndarray | None,
) -> None:
    if joint is None:
        lines.append(f"{prefix}=None")
        return
    lines.extend(
        [
            f"{prefix}_name={joint.name}",
            f"{prefix}_active_index={_safe_attr(joint, 'active_index')}",
            f"{prefix}_joint_index={_safe_attr(joint, 'index')}",
            f"{prefix}_drive_target={_safe_attr(joint, 'get_drive_target')}",
            f"{prefix}_qpos={_format_scalar(qpos[joint_idx] if joint_idx < len(qpos) else None)}",
            f"{prefix}_qvel={_format_scalar(qvel[joint_idx] if joint_idx < len(qvel) else None)}",
            f"{prefix}_tracking_error={_format_scalar(_joint_tracking_error(target_qpos, joint_idx, qpos))}",
            f"{prefix}_stuck={_joint_stuck(target_qpos, joint_idx, qpos, qvel)}",
            f"{prefix}_qf={_format_scalar(qf[joint_idx] if qf is not None and joint_idx < len(qf) else None)}",
            f"{prefix}_passive_force={_format_scalar(passive_force[joint_idx] if passive_force is not None and joint_idx < len(passive_force) else None)}",
        ]
    )


def _append_link_runtime_lines(
    lines: list[str],
    *,
    prefix: str,
    link,
    start_pose: np.ndarray | None,
) -> None:
    if link is None:
        lines.append(f"{prefix}=None")
        return
    pose = _pose_to_numpy(link.pose)
    pos_delta = None
    if pose is not None and start_pose is not None:
        pos_delta = pose[:3] - start_pose[:3]
    lines.extend(
        [
            f"{prefix}_name={link.name}",
            f"{prefix}_pose={_format_pose(link.pose)}",
            f"{prefix}_position_delta={_format_array(pos_delta) if pos_delta is not None else 'None'}",
            f"{prefix}_position_delta_norm={_format_scalar(np.linalg.norm(pos_delta) if pos_delta is not None else None)}",
            f"{prefix}_linear_velocity={_safe_attr(link, 'get_linear_velocity')}",
            f"{prefix}_angular_velocity={_safe_attr(link, 'get_angular_velocity')}",
            f"{prefix}_mass={_safe_attr(link, 'get_mass')}",
            f"{prefix}_cmass_local_pose={_safe_pose(link, 'cmass_local_pose')}",
            f"{prefix}_net_contact_force={_format_array(_sim_contact_force(link)) if _sim_contact_force(link) is not None else 'None'}",
        ]
    )


def _append_probe_snapshot_lines(
    lines: list[str],
    *,
    probe_joints: list[tuple[int, object]],
    probe_links: list[tuple[str, object]],
    target_qpos: np.ndarray | None,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qf: np.ndarray | None,
    passive_force: np.ndarray | None,
    start_link_poses: dict[str, np.ndarray | None],
) -> None:
    lines.append("probe_joint_states:")
    for joint_idx, joint in probe_joints:
        _append_joint_runtime_lines(
            lines,
            prefix=f"joint{joint_idx}",
            joint=joint,
            joint_idx=joint_idx,
            target_qpos=target_qpos,
            qpos=qpos,
            qvel=qvel,
            qf=qf,
            passive_force=passive_force,
        )
    lines.append("probe_link_states:")
    for link_name, link in probe_links:
        _append_link_runtime_lines(
            lines,
            prefix=link_name,
            link=link,
            start_pose=start_link_poses.get(link_name),
        )


def _append_compact_snapshot_line(
    lines: list[str],
    *,
    label: str,
    env: BaseEnv,
    probe_joints: list[tuple[int, object]],
    probe_links: list[tuple[str, object]],
    target_qpos: np.ndarray | None,
    qpos: np.ndarray,
    qvel: np.ndarray,
    start_link_poses: dict[str, np.ndarray | None],
) -> None:
    joint_chunks = []
    for joint_idx, joint in probe_joints:
        if joint is None:
            continue
        joint_chunks.append(
            f"j{joint_idx}(err={_format_scalar(_joint_tracking_error(target_qpos, joint_idx, qpos))},"
            f" vel={_format_scalar(qvel[joint_idx] if joint_idx < len(qvel) else None)},"
            f" stuck={_joint_stuck(target_qpos, joint_idx, qpos, qvel)})"
        )
    link_chunks = []
    for link_name, link in probe_links:
        pose = _pose_to_numpy(link.pose) if link is not None else None
        start_pose = start_link_poses.get(link_name)
        pos_delta_norm = None
        if pose is not None and start_pose is not None:
            pos_delta_norm = float(np.linalg.norm(pose[:3] - start_pose[:3]))
        contact_norm = _norm_or_none(_sim_contact_force(link))
        link_chunks.append(
            f"{link_name}(dp={_format_scalar(pos_delta_norm)}, cf={_format_scalar(contact_norm)})"
        )
    pair_chunks = []
    probe_link_map = {name: link for name, link in probe_links}
    for link_a_name, link_b_name in [
        ("body0", "body1"),
        ("body0", "body2"),
        ("body0", "body3"),
        ("body1", "body2"),
        ("body1", "body3"),
        ("body2", "body3"),
    ]:
        pair_force = _sim_pairwise_contact_force(env, probe_link_map.get(link_a_name), probe_link_map.get(link_b_name))
        pair_chunks.append(
            f"{link_a_name}<->{link_b_name}(cf={_format_scalar(_norm_or_none(pair_force))})"
        )
    lines.append(f"{label}: " + " ".join(joint_chunks + link_chunks + pair_chunks))


def _trace_joint(
    env: BaseEnv,
    args: Args,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    save_idx: int,
    *,
    joint_index: int,
    joint_delta_rad: float,
    section_name: str,
    append: bool,
) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / args.trace_log_name

    agent = env.unwrapped.agent
    arm_controller = _arm_controller(env)
    if arm_controller is None:
        raise RuntimeError("Arm controller is not available.")

    active_joints = agent.robot.get_active_joints()
    joint_index = int(joint_index)
    if joint_index < 0 or joint_index >= 6:
        raise ValueError("--joint-index must be in [0, 5]")
    traced_joint = active_joints[joint_index]
    wrapped_joint = traced_joint
    raw_joint = traced_joint._objs[0]
    probe_joint_indices = [0, 1, 2]
    probe_joints = [
        (idx, active_joints[idx] if idx < len(active_joints) else None)
        for idx in probe_joint_indices
    ]
    joint2 = probe_joints[2][1]
    probe_link_names = ["body0", "body1", "body2", "body3"]
    probe_links = [(name, agent.robot.links_map.get(name)) for name in probe_link_names]
    probe_links_map = {name: link for name, link in probe_links}
    body0 = probe_links_map.get("body0")
    body1 = probe_links_map.get("body1")
    body2 = probe_links_map.get("body2")

    qpos0 = _sim_qpos(env)
    qvel0 = _sim_qvel(env)
    qf0 = _sim_qf(env)
    passive_force0 = _sim_passive_force(env)
    tcp0 = _sim_tcp_pose(env)
    target_qpos = qpos0[:6].copy()
    target_qpos[joint_index] += float(joint_delta_rad)
    start_link_poses = {
        name: (_pose_to_numpy(link.pose) if link is not None else None)
        for name, link in probe_links
    }

    action = torch.tensor(
        [[
            float(target_qpos[0]),
            float(target_qpos[1]),
            float(target_qpos[2]),
            float(target_qpos[3]),
            float(target_qpos[4]),
            float(target_qpos[5]),
            -1.0,
        ]],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )

    lines = []
    if not append:
        lines.append("# rc5 joint physics trace")
    lines.extend([
        f"section={section_name}",
        f"control_mode={args.control_mode_name}",
        f"sim_backend={args.sim_backend}",
        f"compact_log={args.compact_log}",
        f"joint_index={joint_index}",
        f"joint_name={traced_joint.name}",
        f"articulation_raw_drive_target={_safe_attr(agent.robot._objs[0], 'get_drive_target')}",
        f"articulation_raw_qpos={_safe_attr(agent.robot._objs[0], 'get_qpos')}",
        f"articulation_raw_qvel={_safe_attr(agent.robot._objs[0], 'get_qvel')}",
        f"articulation_qf={_format_array(_sim_qf(env)[:6])}",
        f"articulation_passive_force={_format_array(_sim_passive_force(env)[:6]) if _sim_passive_force(env) is not None else 'None'}",
        f"joint0_qf={_sim_qf(env)[0]:.6f}",
        f"joint0_passive_force={(_sim_passive_force(env)[0] if _sim_passive_force(env) is not None else None)}",
        f"arm_active_joint_names={[j.name for j in arm_controller.joints]}",
        f"arm_active_joint_indices={arm_controller.active_joint_indices.tolist()}",
        f"arm_joint_limits={_format_array(_joint_limits_for_arm(env, arm_controller))}",
        f"start_qpos={_format_array(qpos0[:6])}",
        f"start_qvel={_format_array(qvel0[:6])}",
        f"target_qpos={_format_array(target_qpos)}",
        f"start_tcp_xyzquat={_format_array(tcp0)}",
        "body0_collision_shapes:",
    ])
    _append_link_collision_shape_lines(lines, "body0", body0)
    lines.extend([
        "body1_collision_shapes:",
    ])
    _append_link_collision_shape_lines(lines, "body1", body1)
    lines.extend([
        "body0_body1_start_contact_details:",
    ])
    _append_pairwise_contact_detail_lines(
        lines,
        label="body0_body1_start",
        env=env,
        link_a=body0,
        link_b=body1,
    )
    lines.extend([
        "after_set_action_and_step:",
    ])
    if args.compact_log:
        _append_compact_snapshot_line(
            lines,
            label="start_snapshot",
            env=env,
            probe_joints=probe_joints,
            probe_links=probe_links,
            target_qpos=target_qpos,
            qpos=qpos0[:6],
            qvel=qvel0[:6],
            start_link_poses=start_link_poses,
        )
    else:
        lines.extend(["", "[traced_joint]", "", "[joint2]", "", "[body2]", ""])
        traced_joint_marker = lines.index("[traced_joint]")
        joint2_marker = lines.index("[joint2]")
        body2_marker = lines.index("[body2]")
        traced_joint_lines: list[str] = []
        joint2_lines: list[str] = []
        body2_lines: list[str] = []
        _describe_joint(traced_joint_lines, "joint", wrapped_joint)
        _describe_joint(joint2_lines, "joint2", joint2)
        _describe_link(body2_lines, "body2", body2)
        lines[traced_joint_marker:traced_joint_marker + 1] = traced_joint_lines
        joint2_marker = joint2_marker - 1 + len(traced_joint_lines)
        lines[joint2_marker:joint2_marker + 1] = joint2_lines
        body2_marker = body2_marker - 2 + len(traced_joint_lines) + len(joint2_lines)
        lines[body2_marker:body2_marker + 1] = body2_lines
        _append_probe_snapshot_lines(
            lines,
            probe_joints=probe_joints,
            probe_links=probe_links,
            target_qpos=target_qpos,
            qpos=qpos0[:6],
            qvel=qvel0[:6],
            qf=qf0[:6],
            passive_force=passive_force0[:6] if passive_force0 is not None else None,
            start_link_poses=start_link_poses,
        )

    obs, _, terminated, truncated, info = env.step(action)
    if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
        raise RuntimeError("Simulation episode ended during initial joint trace step.")

    qpos1 = _sim_qpos(env)
    qvel1 = _sim_qvel(env)
    qf1 = _sim_qf(env)
    tcp1 = _sim_tcp_pose(env)
    passive_force1 = _sim_passive_force(env)
    controller_target_qpos = None
    if getattr(arm_controller, "_target_qpos", None) is not None:
        controller_target_qpos = _tensor_to_numpy(arm_controller._target_qpos)[0].astype(np.float64)

        lines.extend(
        [
            f"controller_target_qpos={_format_array(controller_target_qpos) if controller_target_qpos is not None else 'None'}",
            f"joint_wrapper_drive_target={_safe_attr(wrapped_joint, 'get_drive_target')}",
            f"raw_joint_drive_target={_safe_attr(raw_joint, 'drive_target')}",
            f"articulation_raw_drive_target={_safe_attr(agent.robot._objs[0], 'get_drive_target')}",
            f"joint_wrapper_qpos={_safe_attr(wrapped_joint, 'qpos')}",
            f"joint_wrapper_qvel={_safe_attr(wrapped_joint, 'qvel')}",
            f"qpos={_format_array(qpos1[:6])}",
            f"qvel={_format_array(qvel1[:6])}",
            f"qf={_format_array(qf1[:6])}",
            f"passive_force={_format_array(passive_force1[:6]) if passive_force1 is not None else 'None'}",
            f"joint0_qf={qf1[0]:.6f}",
            f"joint0_passive_force={(f'{passive_force1[0]:.6f}' if passive_force1 is not None else 'None')}",
            f"tcp_xyzquat={_format_array(tcp1)}",
            f"tracking_error={_format_array(target_qpos - qpos1[:6])}",
            "body0_body1_after_step_contact_details:",
        ]
    )
    _append_pairwise_contact_detail_lines(
        lines,
        label="body0_body1_after_step",
        env=env,
        link_a=body0,
        link_b=body1,
    )
    if args.compact_log:
        _append_compact_snapshot_line(
            lines,
            label="after_step_snapshot",
            env=env,
            probe_joints=probe_joints,
            probe_links=probe_links,
            target_qpos=controller_target_qpos if controller_target_qpos is not None else target_qpos,
            qpos=qpos1[:6],
            qvel=qvel1[:6],
            start_link_poses=start_link_poses,
        )
    else:
        _append_probe_snapshot_lines(
            lines,
            probe_joints=probe_joints,
            probe_links=probe_links,
            target_qpos=controller_target_qpos if controller_target_qpos is not None else target_qpos,
            qpos=qpos1[:6],
            qvel=qvel1[:6],
            qf=qf1[:6],
            passive_force=passive_force1[:6] if passive_force1 is not None else None,
            start_link_poses=start_link_poses,
        )
    lines.extend(["", "settle_steps:"])

    save_idx += 1
    _save_snapshot_from_obs(obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)

    for step_idx in range(args.trace_steps):
        obs, _, terminated, truncated, info = env.step(action)
        if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
            raise RuntimeError(f"Simulation episode ended during trace step {step_idx}.")
        qpos = _sim_qpos(env)
        qvel = _sim_qvel(env)
        qf = _sim_qf(env)
        tcp = _sim_tcp_pose(env)
        passive_force = _sim_passive_force(env)
        lines.extend(
            [
                f"step={step_idx + 1}",
                f"joint_wrapper_drive_target={_safe_attr(wrapped_joint, 'get_drive_target')}",
                f"raw_joint_drive_target={_safe_attr(raw_joint, 'drive_target')}",
                f"articulation_raw_drive_target={_safe_attr(agent.robot._objs[0], 'get_drive_target')}",
                f"joint_wrapper_qpos={_safe_attr(wrapped_joint, 'qpos')}",
                f"joint_wrapper_qvel={_safe_attr(wrapped_joint, 'qvel')}",
                f"qpos={_format_array(qpos[:6])}",
                f"qvel={_format_array(qvel[:6])}",
                f"qf={_format_array(qf[:6])}",
                f"passive_force={_format_array(passive_force[:6]) if passive_force is not None else 'None'}",
                f"joint0_qf={qf[0]:.6f}",
                f"joint0_passive_force={(f'{passive_force[0]:.6f}' if passive_force is not None else 'None')}",
                f"tcp_xyzquat={_format_array(tcp)}",
                f"tracking_error={_format_array(target_qpos - qpos[:6])}",
            ]
        )
        if args.compact_log:
            _append_compact_snapshot_line(
                lines,
                label=f"step_summary_{step_idx + 1}",
                env=env,
                probe_joints=probe_joints,
                probe_links=probe_links,
                target_qpos=controller_target_qpos if controller_target_qpos is not None else target_qpos,
                qpos=qpos[:6],
                qvel=qvel[:6],
                start_link_poses=start_link_poses,
            )
        else:
            _append_probe_snapshot_lines(
                lines,
                probe_joints=probe_joints,
                probe_links=probe_links,
                target_qpos=controller_target_qpos if controller_target_qpos is not None else target_qpos,
                qpos=qpos[:6],
                qvel=qvel[:6],
                qf=qf[:6],
                passive_force=passive_force[:6] if passive_force is not None else None,
                start_link_poses=start_link_poses,
            )
        lines.append("")

    if append and log_path.exists():
        existing = log_path.read_text(encoding="utf-8").rstrip()
        log_path.write_text(existing + "\n\n" + "\n".join(lines), encoding="utf-8")
    else:
        log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved trace log to {log_path}")
    return save_idx


def main(args: Args):
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)

    camera_mode = args.observation_camera_mode
    cam_state = observation_camera_lookat_state(camera_mode)
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
        _save_snapshot_from_obs(obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
        print(f"Control mode: {args.control_mode_name}")
        print(f"Tracing joint index {args.joint_index} with delta {args.joint_delta_rad} rad")
        _trace_joint(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            save_idx,
            joint_index=args.joint_index,
            joint_delta_rad=args.joint_delta_rad,
            section_name=f"joint_{args.joint_index}",
            append=False,
        )
        obs, info = _reset_current_state(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            robot_init_qpos,
            reconfigure=True,
        )
        _save_snapshot_from_obs(obs, info, args, robot_base_pose_p, robot_base_pose_q, save_idx)
        print(f"Tracing joint index {args.secondary_joint_index} with delta {args.secondary_joint_delta_rad} rad")
        _trace_joint(
            env,
            args,
            robot_base_pose_p,
            robot_base_pose_q,
            save_idx,
            joint_index=args.secondary_joint_index,
            joint_delta_rad=args.secondary_joint_delta_rad,
            section_name=f"joint_{args.secondary_joint_index}",
            append=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
