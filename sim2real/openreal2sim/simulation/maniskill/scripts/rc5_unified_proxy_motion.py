from __future__ import annotations

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose, to_sapien_pose
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_lowlevel import (
    compute_proxy_rotvec_step,
    get_adaptive_proxy_xy_step,
    get_required_pregrasp_joint_guard_config,
    get_required_planner_waypoints_config,
    relatch_runtime_ee_target_pose_to_current,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_targets import (
    resolve_object_target_orientation_mode,
)

_Y = "\033[33m"
_R = "\033[0m"
PLANNER_DEBUG_STAGE_MAX_POS_ERR_M = 0.1


def _maintain_object_target_orientation(planner_cfg, env_unwrapped) -> bool:
    profiles = planner_cfg.get("planner_object_calibrations")
    object_id = getattr(env_unwrapped, "manip_object_id", None)
    if not isinstance(profiles, dict) or object_id is None or str(object_id) not in profiles:
        return False
    object_profile = profiles[str(object_id)]
    if not isinstance(object_profile, dict):
        raise RuntimeError(f"planner_object_calibrations.{object_id} must be a mapping.")
    return (
        resolve_object_target_orientation_mode(object_profile, object_id=str(object_id))
        == "current_tcp"
    )


def _get_runtime_ee_arm_controller(env_unwrapped):
    agent = getattr(env_unwrapped, "agent", None)
    controller = getattr(agent, "controller", None)
    controllers = getattr(controller, "controllers", {}) if controller is not None else {}
    arm_controller = controllers.get("arm")
    if arm_controller is None:
        raise RuntimeError("Runtime arm controller is unavailable for pregrasp joint guard.")
    return arm_controller


def _get_robot_arm_qpos_rows(env_unwrapped):
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None or not hasattr(agent, "robot"):
        raise RuntimeError("Runtime agent.robot is unavailable for pregrasp joint guard.")
    arm_joint_names = list(getattr(agent, "arm_joint_names", []) or [])
    if not arm_joint_names:
        raise RuntimeError("Runtime agent exposes no arm_joint_names for pregrasp joint guard.")
    robot_qpos = agent.robot.get_qpos()
    if hasattr(robot_qpos, "detach"):
        robot_qpos = robot_qpos.detach().cpu().numpy()
    robot_qpos = np.asarray(robot_qpos, dtype=np.float32)
    if robot_qpos.ndim == 1:
        robot_qpos = robot_qpos.reshape(1, -1)
    return arm_joint_names, np.asarray(robot_qpos[:, : len(arm_joint_names)], dtype=np.float32).copy()


def _build_pregrasp_joint_target_rows(env_unwrapped, joint_targets: dict[str, float]):
    arm_joint_names, current_arm_qpos_rows = _get_robot_arm_qpos_rows(env_unwrapped)
    target_rows = current_arm_qpos_rows.copy()
    for joint_name, target_value in joint_targets.items():
        if joint_name not in arm_joint_names:
            raise RuntimeError(
                f"planner_proxy_pregrasp_joint_guard references unknown arm joint {joint_name!r}. "
                f"Available arm joints: {arm_joint_names}"
            )
        joint_idx = arm_joint_names.index(joint_name)
        target_rows[:, joint_idx] = float(target_value)
    return arm_joint_names, target_rows


def _resolve_pregrasp_joint_targets(planner_cfg, *, manip_id, global_joint_targets):
    profiles = planner_cfg.get("planner_object_calibrations", None)
    if isinstance(profiles, dict):
        object_profile = profiles.get(str(manip_id))
        if isinstance(object_profile, dict):
            object_joint_targets = object_profile.get("joint_targets", None)
            if isinstance(object_joint_targets, dict) and len(object_joint_targets) > 0:
                normalized_targets = {}
                for raw_joint_name, raw_joint_value in object_joint_targets.items():
                    joint_name = str(raw_joint_name).strip()
                    if not joint_name:
                        continue
                    try:
                        joint_value = float(raw_joint_value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(joint_value):
                        continue
                    normalized_targets[joint_name] = float(joint_value)
                if normalized_targets:
                    return normalized_targets, f"planner_object_calibrations.{str(manip_id)}.joint_targets"
    return dict(global_joint_targets), "planner_proxy_pregrasp_joint_guard.joint_targets"


def _pose_to_numpy_first_env(pose_like):
    pose_p = getattr(pose_like, "p")
    pose_q = getattr(pose_like, "q")
    if hasattr(pose_p, "detach"):
        pose_p = pose_p.detach().cpu().numpy()
    if hasattr(pose_q, "detach"):
        pose_q = pose_q.detach().cpu().numpy()
    pose_p = np.asarray(pose_p, dtype=np.float32).reshape(-1, 3)[0]
    pose_q = np.asarray(pose_q, dtype=np.float32).reshape(-1, 4)[0]
    return pose_p.copy(), pose_q.copy()


def _get_runtime_tcp_pose_np(env_unwrapped, *, get_debug_planner_ee_pose_sapien):
    current_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
    return _pose_to_numpy_first_env(current_pose)


def _log_pregrasp_tcp_target_xy(
    *,
    stage_label: str,
    phase_label: str,
    target_pose,
    tcp_p,
    reference_tcp_p=None,
):
    target_p, _target_q = _pose_to_numpy_first_env(target_pose)
    tcp_p = np.asarray(tcp_p, dtype=np.float32).reshape(3)
    target_xy_err_vec = np.asarray(tcp_p[:2] - target_p[:2], dtype=np.float32)
    target_xy_err = float(np.linalg.norm(target_xy_err_vec))
    target_z_err = float(tcp_p[2] - target_p[2])
    message = (
        f"[PlannerDebug] Proxy pregrasp tcp_xy_check '{stage_label}' {phase_label}: "
        f"tcp_xy={np.array2string(tcp_p[:2], precision=4, suppress_small=True)} "
        f"target_xy={np.array2string(target_p[:2], precision=4, suppress_small=True)} "
        f"target_xy_err_vec={np.array2string(target_xy_err_vec, precision=4, suppress_small=True)} "
        f"target_xy_err={target_xy_err:.4f} m "
        f"target_z_err={target_z_err:.4f} m"
    )
    if reference_tcp_p is not None:
        reference_tcp_p = np.asarray(reference_tcp_p, dtype=np.float32).reshape(3)
        tcp_xy_drift_vec = np.asarray(tcp_p[:2] - reference_tcp_p[:2], dtype=np.float32)
        tcp_xy_drift = float(np.linalg.norm(tcp_xy_drift_vec))
        tcp_z_drift = float(tcp_p[2] - reference_tcp_p[2])
        message += (
            f" align_xy_drift_vec={np.array2string(tcp_xy_drift_vec, precision=4, suppress_small=True)} "
            f"align_xy_drift={tcp_xy_drift:.4f} m "
            f"align_z_drift={tcp_z_drift:.4f} m"
        )
    print(message)


def _maybe_relatch_runtime_target_pose_for_settle(
    *,
    env_unwrapped,
    planner_cfg,
    target_pose,
    stage_label: str,
    relatch_enabled: bool,
    get_debug_planner_ee_pose_sapien,
):
    if not relatch_enabled:
        return
    _target_p, target_q = _pose_to_numpy_first_env(target_pose)
    _current_p, current_q = _get_runtime_tcp_pose_np(
        env_unwrapped,
        get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
    )
    max_rot_step = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_step_deg", 6.0)))
    rot_tol = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_tol_deg", 8.0)))
    _rot_delta, rot_err = compute_proxy_rotvec_step(current_q, target_q, max_rot_step)
    if rot_err > rot_tol:
        print(
            f"[PlannerDebug] Skipping runtime EE target pose relatch before {stage_label}: "
            f"rot_err={rot_err:.4f} rad exceeds rot_tol={rot_tol:.4f} rad"
        )
        return
    relatch_runtime_ee_target_pose_to_current(
        env_unwrapped,
        reason=f"{stage_label}:rot_err={rot_err:.4f}",
    )


def _run_proxy_pregrasp_joint_align_and_guard(
    env,
    *,
    target_pose,
    stage_label: str,
    planner_cfg,
    run_proxy_ee_delta_pose_stage,
    get_debug_planner_ee_pose_sapien,
):
    joint_guard_cfg = get_required_pregrasp_joint_guard_config(planner_cfg)
    if not joint_guard_cfg["enabled"]:
        return True
    run_align_stage = bool(joint_guard_cfg["run_align_stage"])
    env_unwrapped = env.unwrapped
    manip_id = str(getattr(env_unwrapped, "manip_object_id", None))
    arm_controller = _get_runtime_ee_arm_controller(env_unwrapped)
    resolved_joint_targets, joint_target_source = _resolve_pregrasp_joint_targets(
        planner_cfg,
        manip_id=manip_id,
        global_joint_targets=joint_guard_cfg["joint_targets"],
    )
    arm_joint_names, target_rows = _build_pregrasp_joint_target_rows(
        env_unwrapped,
        resolved_joint_targets,
    )
    tcp_p_before_guard, _tcp_q_before_guard = _get_runtime_tcp_pose_np(
        env_unwrapped,
        get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
    )
    _target_p_guard, target_q_guard = _pose_to_numpy_first_env(target_pose)
    max_rot_step = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_step_deg", 6.0)))
    rot_tol = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_tol_deg", 8.0)))
    _rot_delta_guard, rot_err_guard = compute_proxy_rotvec_step(
        _tcp_q_before_guard,
        target_q_guard,
        max_rot_step,
    )
    print(
        f"[PlannerDebug] Proxy pregrasp wrist guard '{stage_label}': "
        f"manip_object_id={manip_id} "
        f"joint_targets={resolved_joint_targets} "
        f"joint_target_source='{joint_target_source}' "
        f"tolerance_rad={joint_guard_cfg['tolerance_rad']:.4f} "
        f"rot_err={rot_err_guard:.4f} rad "
        f"rot_tol={rot_tol:.4f} rad "
        f"mismatch_policy='{joint_guard_cfg['mismatch_policy']}' "
        f"run_align_stage={run_align_stage}"
    )
    _log_pregrasp_tcp_target_xy(
        stage_label=stage_label,
        phase_label="before_guard",
        target_pose=target_pose,
        tcp_p=tcp_p_before_guard,
    )
    if run_align_stage:
        previous_preferred = getattr(arm_controller, "_preferred_arm_qpos", None)
        robot_qpos = env_unwrapped.agent.robot.get_qpos()
        if not hasattr(robot_qpos, "detach"):
            raise RuntimeError("Runtime robot qpos must be a torch tensor for pregrasp joint guard.")
        target_tensor = torch.as_tensor(
            target_rows,
            device=robot_qpos.device,
            dtype=robot_qpos.dtype,
        )
        previous_solver_config = dict(getattr(arm_controller.config, "delta_solver_config", {}) or {})
        align_solver_config = dict(previous_solver_config)
        align_solver_config.update(dict(joint_guard_cfg["align_solver_overrides"]))
        align_target_pose = Pose.create_from_pq(
            p=np.asarray(
                [
                    float(_target_p_guard[0]),
                    float(_target_p_guard[1]),
                    float(tcp_p_before_guard[2]),
                ],
                dtype=np.float32,
            ),
            q=np.asarray(target_q_guard, dtype=np.float32),
        )
        print(
            f"[PlannerDebug] Proxy pregrasp wrist align '{stage_label}': "
            f"align_target_p={np.array2string(np.asarray(align_target_pose.p, dtype=np.float32).reshape(-1)[:3], precision=4, suppress_small=True)} "
            f"align_target_q={np.array2string(np.asarray(align_target_pose.q, dtype=np.float32).reshape(-1)[:4], precision=4, suppress_small=True)} "
            f"align_solver_overrides={joint_guard_cfg['align_solver_overrides']}"
        )
        try:
            arm_controller._preferred_arm_qpos = target_tensor
            arm_controller.config.delta_solver_config = align_solver_config
            stage_ok = run_proxy_ee_delta_pose_stage(
                env,
                align_target_pose,
                stage_label=f"{stage_label}:wrist_pregrasp_align",
                position_mask=(True, True, True),
                align_orientation=True,
                gripper_target_state="hold",
            )
        finally:
            arm_controller.config.delta_solver_config = previous_solver_config
            arm_controller._preferred_arm_qpos = previous_preferred
        tcp_p_after_align, _tcp_q_after_align = _get_runtime_tcp_pose_np(
            env_unwrapped,
            get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
        )
        _log_pregrasp_tcp_target_xy(
            stage_label=stage_label,
            phase_label="after_align",
            target_pose=target_pose,
            tcp_p=tcp_p_after_align,
            reference_tcp_p=tcp_p_before_guard,
        )
        _rot_delta_guard, rot_err_guard = compute_proxy_rotvec_step(
            _tcp_q_after_align,
            target_q_guard,
            max_rot_step,
        )
        if not stage_ok:
            message = (
                f"Proxy pregrasp wrist align stage failed before descent "
                f"(policy={joint_guard_cfg['mismatch_policy']})."
            )
            if joint_guard_cfg["mismatch_policy"] == "fail_fast":
                print(f"{_Y}[WARNING] [PlannerDebug] {message}{_R}")
                return False
            print(
                f"{_Y}[WARNING] [PlannerDebug] {message} Continuing because mismatch_policy='warn'.{_R}"
            )
            return True
        tcp_p_before_guard = tcp_p_after_align
    else:
        _log_pregrasp_tcp_target_xy(
            stage_label=stage_label,
            phase_label="guard_only",
            target_pose=target_pose,
            tcp_p=tcp_p_before_guard,
        )

    if rot_err_guard > rot_tol:
        message = (
            f"Proxy pregrasp orientation gate '{stage_label}' failed: "
            f"rot_err={rot_err_guard:.4f} rad exceeds rot_tol={rot_tol:.4f} rad"
        )
        if joint_guard_cfg["mismatch_policy"] == "fail_fast":
            print(f"{_Y}[WARNING] [PlannerDebug] {message}{_R}")
            return False
        print(
            f"{_Y}[WARNING] [PlannerDebug] {message}. "
            "Continuing because mismatch_policy='warn'.{_R}"
        )

    _arm_joint_names_after, actual_rows = _get_robot_arm_qpos_rows(env_unwrapped)
    violations = []
    tolerance_rad = float(joint_guard_cfg["tolerance_rad"])
    for env_index in range(actual_rows.shape[0]):
        for joint_name, target_value in resolved_joint_targets.items():
            joint_idx = arm_joint_names.index(joint_name)
            actual_value = float(actual_rows[env_index, joint_idx])
            joint_error = float(abs(actual_value - float(target_value)))
            if joint_error > tolerance_rad:
                violations.append((env_index, joint_name, actual_value, float(target_value), joint_error))
    if not violations:
        summary = ", ".join(
            f"{joint_name}={float(actual_rows[0, arm_joint_names.index(joint_name)]):.4f}"
            for joint_name in resolved_joint_targets.keys()
        )
        print(
            f"[PlannerDebug] Proxy pregrasp wrist guard '{stage_label}' passed: "
            f"{summary} within tolerance_rad={tolerance_rad:.4f}"
        )
        return True
    details = "; ".join(
        f"env={env_index} {joint_name}: actual={actual_value:.4f} target={target_value:.4f} err={joint_error:.4f} rad"
        for env_index, joint_name, actual_value, target_value, joint_error in violations
    )
    if joint_guard_cfg["mismatch_policy"] == "fail_fast":
        print(
            f"{_Y}[WARNING] [PlannerDebug] Proxy pregrasp wrist guard '{stage_label}' failed: "
            f"{details}{_R}"
        )
        return False
    print(
        f"{_Y}[WARNING] [PlannerDebug] Proxy pregrasp wrist guard '{stage_label}' mismatch: "
        f"{details}{_R}"
    )
    return True


def _interpolate_pose_waypoints(start_pose, target_pose, waypoint_count: int):
    start_pose = to_sapien_pose(start_pose)
    if not hasattr(target_pose, "raw_pose"):
        target_pose = to_sapien_pose(target_pose)
    start_p = np.asarray(start_pose.p, dtype=np.float32).reshape(-1)[:3]
    target_p = np.asarray(target_pose.p, dtype=np.float32).reshape(-1)[:3]
    target_q = np.asarray(target_pose.q, dtype=np.float32).reshape(-1)[:4]
    waypoints = []
    for idx in range(int(waypoint_count)):
        alpha = float(idx + 1) / float(int(waypoint_count) + 1)
        waypoint_p = ((1.0 - alpha) * start_p + alpha * target_p).astype(np.float32)
        waypoints.append(
            Pose.create_from_pq(
                p=np.asarray(waypoint_p, dtype=np.float32),
                q=np.asarray(target_q, dtype=np.float32),
            )
        )
    return waypoints


def execute_real_planner_pose_with_backend(
    solver,
    target_pose,
    *,
    execute: bool,
    backend: str,
    method: str,
    planner_class_name: str,
    stage_label: str,
    is_proxy_ee_delta_backend,
    is_rc5_debug_planner_agent,
):
    if is_proxy_ee_delta_backend(backend):
        raise ValueError(
            "Proxy planner backend must be dispatched directly from the stage runner, "
            f"got stage_label='{stage_label}'."
        )

    if backend == "local_ik":
        if hasattr(solver, "move_to_pose_with_local_ik"):
            print(
                f"{_Y}[PlannerDebug] {stage_label} dispatch: method='{method}' -> local_ik "
                f"(hybrid planner flow){_R}"
            )
            return solver.move_to_pose_with_local_ik(target_pose, dry_run=not execute)
        print(
            f"{_Y}[PlannerDebug] {stage_label} dispatch: local_ik unavailable for "
            f"{planner_class_name} -> generic move_to_pose(){_R}"
        )
        return solver.move_to_pose(target_pose, dry_run=not execute)

    if backend == "planner":
        if method == "auto":
            if is_rc5_debug_planner_agent(getattr(solver.base_env.agent, "uid", "")):
                print(f"{_Y}[PlannerDebug] {stage_label} dispatch: auto -> RRTConnect (planner stability){_R}")
                return solver.move_to_pose_with_RRTConnect(target_pose, dry_run=not execute)
            print(f"{_Y}[PlannerDebug] {stage_label} dispatch: auto -> generic move_to_pose() (compatibility){_R}")
            return solver.move_to_pose(target_pose, dry_run=not execute)
        if method == "rrtconnect":
            return solver.move_to_pose_with_RRTConnect(target_pose, dry_run=not execute)
        if method == "screw":
            return solver.move_to_pose_with_screw(target_pose, dry_run=not execute)
        raise ValueError(f"Unsupported real planner / mplib planner {stage_label.lower()} method: {method}")

    raise ValueError(f"Unsupported real planner / mplib planner {stage_label.lower()} backend: {backend}")


def execute_planner_pose_with_backend(
    solver,
    target_pose,
    *,
    execute: bool,
    backend: str,
    method: str,
    planner_class_name: str,
    stage_label: str,
    is_proxy_ee_delta_backend,
    is_rc5_debug_planner_agent,
):
    return execute_real_planner_pose_with_backend(
        solver,
        target_pose,
        execute=execute,
        backend=backend,
        method=method,
        planner_class_name=planner_class_name,
        stage_label=stage_label,
        is_proxy_ee_delta_backend=is_proxy_ee_delta_backend,
        is_rc5_debug_planner_agent=is_rc5_debug_planner_agent,
    )


def run_linear_approach_waypoints(
    env,
    solver,
    target_pose,
    *,
    waypoint_count: int,
    waypoint_mode: str,
    execute: bool,
    backend: str,
    method: str,
    stage_label: str,
    get_debug_planner_ee_pose,
    pose_to_numpy,
    execute_planner_pose_with_backend,
):
    if int(waypoint_count) <= 0:
        return True
    waypoint_mode = str(waypoint_mode or "fixed").strip().lower()
    if waypoint_mode not in {"fixed", "current"}:
        raise ValueError(
            f"Unsupported {stage_label.lower()} waypoint_mode='{waypoint_mode}'. "
            "Unified RC5 runtime requires an explicit supported waypoint mode."
        )
    current_pose = to_sapien_pose(get_debug_planner_ee_pose(env.unwrapped))
    initial_pose = current_pose
    initial_p, initial_q = pose_to_numpy(initial_pose)
    target_p_full, target_q_full = pose_to_numpy(target_pose)
    waypoints = (
        _interpolate_pose_waypoints(current_pose, target_pose, int(waypoint_count))
        if waypoint_mode == "fixed"
        else None
    )
    print(
        f"[PlannerDebug] {stage_label} waypoint mode enabled: waypoint_count={int(waypoint_count)} "
        f"waypoint_mode='{waypoint_mode}' from current pose to final target"
    )
    print(
        f"[PlannerDebug][WAYPOINT_PLAN] stage={stage_label} mode='{waypoint_mode}' "
        f"start_p={np.array2string(initial_p, precision=4, suppress_small=True)} "
        f"start_q={np.array2string(initial_q, precision=4, suppress_small=True)} "
        f"final_target_p={np.array2string(target_p_full, precision=4, suppress_small=True)} "
        f"final_target_q={np.array2string(target_q_full, precision=4, suppress_small=True)} "
        f"count={int(waypoint_count)}"
    )
    total_waypoints = int(waypoint_count)
    for idx in range(1, total_waypoints + 1):
        if waypoint_mode == "fixed":
            waypoint_pose = waypoints[idx - 1]
            source_pose_for_waypoint = initial_pose
            remaining_waypoints = total_waypoints - idx + 1
        else:
            current_pose = to_sapien_pose(get_debug_planner_ee_pose(env.unwrapped))
            remaining_waypoints = total_waypoints - idx + 1
            source_pose_for_waypoint = current_pose
            waypoint_pose = _interpolate_pose_waypoints(current_pose, target_pose, remaining_waypoints)[0]
        source_p = np.asarray(source_pose_for_waypoint.p, dtype=np.float32).reshape(-1)[:3]
        source_q = np.asarray(source_pose_for_waypoint.q, dtype=np.float32).reshape(-1)[:4]
        waypoint_p = np.asarray(waypoint_pose.p, dtype=np.float32).reshape(-1)[:3]
        waypoint_q = np.asarray(waypoint_pose.q, dtype=np.float32).reshape(-1)[:4]
        print(
            f"[PlannerDebug][WAYPOINT_SELECT] stage={stage_label} mode='{waypoint_mode}' "
            f"selected={idx}/{total_waypoints} remaining_before_select={remaining_waypoints} "
            f"source_p={np.array2string(source_p, precision=4, suppress_small=True)} "
            f"source_q={np.array2string(source_q, precision=4, suppress_small=True)} "
            f"selected_target_p={np.array2string(waypoint_p, precision=4, suppress_small=True)} "
            f"selected_target_q={np.array2string(waypoint_q, precision=4, suppress_small=True)}"
        )
        print(
            f"[PlannerDebug] {stage_label} waypoint {idx}/{total_waypoints} "
            f"p={np.array2string(waypoint_p, precision=4, suppress_small=True)} "
            f"q={np.array2string(waypoint_q, precision=4, suppress_small=True)}"
        )
        result = execute_planner_pose_with_backend(
            solver,
            waypoint_pose,
            execute=execute,
            backend=backend,
            method=method,
            planner_class_name=type(solver).__name__,
            stage_label=f"{stage_label} waypoint {idx}/{total_waypoints}",
        )
        if result == -1:
            print(f"[PlannerDebug] {stage_label} waypoint {idx}/{total_waypoints} FAILED")
            return False
        if execute:
            final_tcp_p, _final_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - waypoint_p))
            max_accept_err = float(PLANNER_DEBUG_STAGE_MAX_POS_ERR_M)
            print(
                f"[PlannerDebug] {stage_label} waypoint {idx}/{total_waypoints} final tcp p="
                f"{np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(waypoint_p, precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] {stage_label} waypoint {idx}/{total_waypoints} "
                    f"pos_err={final_err:.4f} m exceeds acceptance threshold {max_accept_err:.4f} m; "
                    "treating stage as FAILED."
                )
                return False
    return True


def run_proxy_ee_delta_pose_stage(
    env,
    target_pose,
    *,
    stage_label: str,
    position_mask=(True, True, True),
    align_orientation: bool = True,
    gripper_target_state: str = "hold",
    max_z_step_override: float | None = None,
    pos_tol_override: float | None = None,
    get_debug_planner_config,
    get_debug_planner_ee_pose_sapien,
    get_debug_planner_ee_pose=None,
    pose_to_numpy_rows=None,
    compute_proxy_rotvec_step,
    build_proxy_delta_pos,
    apply_proxy_ee_delta_action,
    step_observer=None,
):
    def _positive_override(value, *, label: str):
        if value is None:
            return None
        normalized = float(value)
        if not np.isfinite(normalized) or normalized <= 0.0:
            raise RuntimeError(f"{label} must be a finite positive value, got {value!r}.")
        return normalized

    env_unwrapped = env.unwrapped
    planner_cfg = get_debug_planner_config(env_unwrapped)
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    use_batched_pose = callable(get_debug_planner_ee_pose) and callable(pose_to_numpy_rows) and num_envs > 1
    if num_envs > 1 and not use_batched_pose:
        print(
            f"{_Y}[WARNING] [PlannerDebug] Proxy stage '{stage_label}' is running batched "
            f"(num_envs={num_envs}) without batched pose helpers; falling back to env0-first "
            "pose tracking for this stage.{_R}"
        )
    if not hasattr(target_pose, "raw_pose"):
        target_pose = to_sapien_pose(target_pose)
    target_p_rows = np.asarray(target_pose.p, dtype=np.float32).reshape(-1, 3)
    target_q_rows = np.asarray(target_pose.q, dtype=np.float32).reshape(-1, 4)
    target_p = target_p_rows[0]
    target_q = target_q_rows[0]
    max_xy_step = float(planner_cfg.get("planner_proxy_xy_step_m", 0.01))
    max_z_step = _positive_override(max_z_step_override, label="max_z_step_override")
    if max_z_step is None:
        max_z_step = float(planner_cfg.get("planner_proxy_z_step_m", 0.008))
    max_rot_step = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_step_deg", 6.0)))
    hold_steps = int(planner_cfg.get("planner_proxy_hold_steps", 1) or 1)
    max_stage_steps = int(planner_cfg.get("planner_proxy_max_stage_steps", 100) or 100)
    pos_tol = _positive_override(pos_tol_override, label="pos_tol_override")
    if pos_tol is None:
        pos_tol = float(planner_cfg.get("planner_proxy_pos_tol_m", 0.01))
    rot_tol = float(np.deg2rad(planner_cfg.get("planner_proxy_rot_tol_deg", 8.0)))
    stall_limit = int(planner_cfg.get("planner_proxy_stall_steps", 12) or 12)
    position_mask = np.asarray(position_mask, dtype=bool).reshape(3)
    proxy_frame_mode = str(planner_cfg.get("planner_proxy_frame", "base_camera_plane") or "base_camera_plane")
    last_total_error = None
    stall_count = 0
    adaptive_xy_logged = False

    print(
        f"[PlannerDebug] Proxy stage '{stage_label}' start: "
        f"target_p={np.array2string(target_p, precision=4, suppress_small=True)} "
        f"target_q={np.array2string(target_q, precision=4, suppress_small=True)} "
        f"position_mask={position_mask.tolist()} align_orientation={align_orientation} "
        f"proxy_frame={proxy_frame_mode}"
    )

    for step_idx in range(1, max_stage_steps + 1):
        gripper_signal_override = None
        if use_batched_pose:
            current_p_rows, current_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))
            effective_target_p = current_p_rows.copy()
            target_p_effective = target_p_rows
            if target_p_effective.shape[0] == 1 and current_p_rows.shape[0] > 1:
                target_p_effective = np.repeat(target_p_effective, current_p_rows.shape[0], axis=0)
            effective_target_p[:, position_mask] = target_p_effective[:, position_mask]
            pos_err_vec = effective_target_p - current_p_rows
            per_env_pos_err = (
                np.linalg.norm(pos_err_vec[:, position_mask], axis=1)
                if np.any(position_mask)
                else np.zeros((current_p_rows.shape[0],), dtype=np.float32)
            )
            pos_err = float(np.max(per_env_pos_err)) if per_env_pos_err.size > 0 else 0.0
            current_q = current_q_rows[0]
        else:
            current_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
            current_p = np.asarray(current_pose.p, dtype=np.float32).reshape(-1)[:3]
            current_q = np.asarray(current_pose.q, dtype=np.float32).reshape(-1)[:4]
            effective_target_p = current_p.copy()
            effective_target_p[position_mask] = target_p[position_mask]
            pos_err_vec = effective_target_p - current_p
            pos_err = float(np.linalg.norm(pos_err_vec[position_mask])) if np.any(position_mask) else 0.0
        rot_delta, rot_err = compute_proxy_rotvec_step(current_q, target_q, max_rot_step)
        if pos_err <= pos_tol and (not align_orientation or rot_err <= rot_tol):
            if adaptive_xy_logged:
                print(
                    f"[PlannerDebug] Proxy stage '{stage_label}' released adaptive XY step at convergence; "
                    f"restoring nominal_xy_step={max_xy_step:.4f} m for subsequent stages"
                )
            print(
                f"[PlannerDebug] Proxy stage '{stage_label}' converged at step {step_idx - 1}: "
                f"pos_err={pos_err:.4f} m rot_err={rot_err:.4f} rad"
            )
            return True

        if bool(position_mask[0] or position_mask[1]):
            if use_batched_pose:
                xy_err = float(np.max(np.linalg.norm(pos_err_vec[:, :2], axis=1)))
            else:
                xy_err = float(np.linalg.norm(np.asarray(pos_err_vec, dtype=np.float32).reshape(-1)[:2]))
            adaptive_xy_step = get_adaptive_proxy_xy_step(
                planner_cfg,
                xy_err=xy_err,
                nominal_xy_step=max_xy_step,
            )
            if adaptive_xy_step < max_xy_step and not adaptive_xy_logged:
                adaptive_xy_logged = True
                print(
                    f"[PlannerDebug] Proxy stage '{stage_label}' switched to adaptive XY step at step {step_idx}: "
                    f"xy_err={xy_err:.4f} m threshold_xy={float(planner_cfg['planner_proxy_threshold_xy_m']):.4f} m "
                    f"nominal_xy_step={max_xy_step:.4f} m adaptive_xy_step={adaptive_xy_step:.4f} m"
                )
        else:
            adaptive_xy_step = max_xy_step

        delta_pos = build_proxy_delta_pos(
            env_unwrapped,
            pos_err_vec,
            position_mask,
            max_xy_step=adaptive_xy_step,
            max_z_step=max_z_step,
            proxy_frame_mode=proxy_frame_mode,
        )
        delta_rpy = rot_delta if align_orientation and rot_err > rot_tol else np.zeros(3, dtype=np.float32)
        if use_batched_pose:
            active_mask = np.asarray(per_env_pos_err > pos_tol, dtype=bool)
            if align_orientation and rot_err > rot_tol:
                active_mask[:] = True
            inactive_mask = ~active_mask
            if np.any(inactive_mask):
                inactive_indices = np.flatnonzero(inactive_mask).tolist()
                active_indices = np.flatnonzero(active_mask).tolist()
                print(
                    f"[PlannerDebug] Proxy stage '{stage_label}' batched partial convergence at step {step_idx}: "
                    f"active_envs={active_indices} inactive_envs={inactive_indices}. "
                    "Neutral gripper hold is applied to converged envs while the batch finishes the stage."
                )
                delta_pos = np.asarray(delta_pos, dtype=np.float32).copy()
                if delta_pos.ndim == 1:
                    delta_pos = np.repeat(delta_pos.reshape(1, -1), current_p_rows.shape[0], axis=0)
                delta_pos[inactive_mask] = 0.0
                if np.asarray(delta_rpy).ndim == 1:
                    delta_rpy = np.repeat(
                        np.asarray(delta_rpy, dtype=np.float32).reshape(1, -1),
                        current_p_rows.shape[0],
                        axis=0,
                    )
                else:
                    delta_rpy = np.asarray(delta_rpy, dtype=np.float32).copy()
                delta_rpy[inactive_mask] = 0.0
                gripper_signal_override = np.full((current_p_rows.shape[0],), np.nan, dtype=np.float32)
                gripper_signal_override[inactive_mask] = 0.0

        total_error = pos_err + (rot_err if align_orientation else 0.0)
        if last_total_error is not None and total_error >= (last_total_error - 1e-4):
            stall_count += 1
        else:
            stall_count = 0
        last_total_error = total_error

        if np.linalg.norm(delta_pos) <= 1e-6 and np.linalg.norm(delta_rpy) <= 1e-6:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Proxy stage '{stage_label}' produced a near-zero step "
                f"before reaching tolerance; aborting.{_R}"
            )
            return False
        if stall_count >= stall_limit:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Proxy stage '{stage_label}' stalled for "
                f"{stall_count} iterations (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad).{_R}"
            )
            return False

        print(
            f"[PlannerDebug] Proxy stage '{stage_label}' step {step_idx}/{max_stage_steps}: "
            f"pos_err={pos_err:.4f} m rot_err={rot_err:.4f} rad "
            f"delta_pos={np.array2string(delta_pos, precision=4, suppress_small=True)} "
            f"delta_rpy={np.array2string(delta_rpy, precision=4, suppress_small=True)}"
        )
        apply_proxy_ee_delta_action(
            env,
            raw_delta_pos=delta_pos,
            raw_delta_rpy=delta_rpy,
            hold_steps=hold_steps,
            stage_label=f"{stage_label}:step{step_idx}",
            gripper_target_state=gripper_target_state,
            gripper_signal_override=gripper_signal_override,
        )
        if callable(step_observer):
            step_observer(
                env_unwrapped,
                step_idx=step_idx,
                max_stage_steps=max_stage_steps,
                stage_label=stage_label,
            )

    final_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
    final_p = np.asarray(final_pose.p, dtype=np.float32).reshape(-1)[:3]
    final_err = float(np.linalg.norm((target_p - final_p)[position_mask])) if np.any(position_mask) else 0.0
    print(
        f"{_Y}[WARNING] [PlannerDebug] Proxy stage '{stage_label}' reached max_stage_steps={max_stage_steps} "
        f"with final_pos_err={final_err:.4f} m.{_R}"
    )
    return False


def run_proxy_full_approach_to_descend(
    env,
    target_pose,
    *,
    initial_actor_p,
    bbox_np,
    stage_label: str,
    safe_clearance_z: float,
    get_debug_planner_config,
    get_debug_planner_ee_pose_sapien,
    get_debug_planner_ee_pose=None,
    pose_to_numpy_rows=None,
    pose_to_numpy,
    run_proxy_ee_delta_pose_stage,
    run_proxy_stationary_settle,
    run_proxy_guarded_descend_to_object,
):
    env_unwrapped = env.unwrapped
    planner_cfg = get_debug_planner_config(env_unwrapped)
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    use_batched_pose = callable(get_debug_planner_ee_pose) and callable(pose_to_numpy_rows) and num_envs > 1
    if num_envs > 1 and not use_batched_pose:
        print(
            f"{_Y}[WARNING] [PlannerDebug] Proxy full approach '{stage_label}' is running batched "
            f"(num_envs={num_envs}) without batched pose helpers; falling back to env0-first "
            "rise/approach pose construction.{_R}"
        )
    if not hasattr(target_pose, "raw_pose"):
        target_pose = to_sapien_pose(target_pose)
    if use_batched_pose:
        current_p_rows, _current_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))
        current_p = current_p_rows[0]
    else:
        current_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
        current_p, _current_q = pose_to_numpy(current_pose)
        current_p_rows = None
    target_p_rows = np.asarray(target_pose.p, dtype=np.float32).reshape(-1, 3)
    target_p = target_p_rows[0]
    target_q = np.asarray(target_pose.q, dtype=np.float32).reshape(-1)[:4]
    target_q_rows = np.asarray(target_pose.q, dtype=np.float32).reshape(-1, 4)
    initial_actor_p_rows = np.asarray(initial_actor_p, dtype=np.float32).reshape(-1, 3)
    initial_actor_p = initial_actor_p_rows[0]
    bbox_np = np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3]
    predescent_settle_steps = int(planner_cfg.get("planner_proxy_predescent_settle_steps", 12) or 0)
    relatch_target_pose_between_stages = bool(
        planner_cfg.get("planner_proxy_relatch_ee_target_pose_between_stages", False)
    )
    waypoint_chain_enabled, waypoint_points = get_required_planner_waypoints_config(planner_cfg)
    maintain_target_orientation = _maintain_object_target_orientation(
        planner_cfg,
        env_unwrapped,
    )
    if use_batched_pose:
        object_top_z_rows = initial_actor_p_rows[:, 2] + 0.5 * bbox_np[2]
        safe_z_rows = np.maximum(current_p_rows[:, 2], object_top_z_rows + float(safe_clearance_z))
        if target_q_rows.shape[0] == 1 and current_p_rows.shape[0] > 1:
            target_q_rows = np.repeat(target_q_rows, current_p_rows.shape[0], axis=0)
        rise_pose = Pose.create_from_pq(
            p=np.stack([current_p_rows[:, 0], current_p_rows[:, 1], safe_z_rows], axis=1).astype(np.float32),
            q=target_q_rows,
        )
        approach_pose = Pose.create_from_pq(
            p=np.stack([target_p_rows[:, 0], target_p_rows[:, 1], safe_z_rows], axis=1).astype(np.float32),
            q=target_q_rows,
        )
        object_top_z = float(object_top_z_rows[0])
        safe_z = float(safe_z_rows[0])
    else:
        object_top_z = float(initial_actor_p[2] + 0.5 * bbox_np[2])
        safe_z = max(float(current_p[2]), float(object_top_z + safe_clearance_z))
        rise_pose = Pose.create_from_pq(
            p=np.asarray([current_p[0], current_p[1], safe_z], dtype=np.float32),
            q=np.asarray(target_q, dtype=np.float32),
        )
        approach_pose = Pose.create_from_pq(
            p=np.asarray([target_p[0], target_p[1], safe_z], dtype=np.float32),
            q=np.asarray(target_q, dtype=np.float32),
        )
    print(
        f"[PlannerDebug] Proxy full approach '{stage_label}': "
        f"safe_clearance_z={float(safe_clearance_z):.4f} "
        f"object_top_z={object_top_z:.4f} safe_z={safe_z:.4f}"
    )
    if waypoint_chain_enabled:
        print(
            f"[PlannerDebug] Proxy waypoint chain enabled for '{stage_label}': "
            f"count={len(waypoint_points)}"
        )
        for waypoint_idx, waypoint in enumerate(waypoint_points, start=1):
            waypoint_position = np.asarray(waypoint["position"], dtype=np.float32).reshape(3)
            print(
                f"[PlannerDebug] Proxy waypoint {waypoint_idx}/{len(waypoint_points)} "
                f"id='{waypoint['id']}' target_p={np.array2string(waypoint_position, precision=4, suppress_small=True)}"
            )
            if use_batched_pose:
                waypoint_p_rows = np.repeat(
                    waypoint_position.reshape(1, 3),
                    current_p_rows.shape[0],
                    axis=0,
                ).astype(np.float32)
                waypoint_pose = Pose.create_from_pq(
                    p=waypoint_p_rows,
                    q=target_q_rows,
                )
            else:
                waypoint_pose = Pose.create_from_pq(
                    p=waypoint_position.astype(np.float32),
                    q=np.asarray(target_q, dtype=np.float32),
                )
            if not run_proxy_ee_delta_pose_stage(
                env,
                waypoint_pose,
                stage_label=f"{stage_label}:waypoint_{waypoint_idx:03d}",
                position_mask=(True, True, True),
                align_orientation=maintain_target_orientation,
                gripper_target_state="hold",
            ):
                return False
    if not run_proxy_ee_delta_pose_stage(
        env,
        rise_pose,
        stage_label=f"{stage_label}:rise_to_safe_z",
        position_mask=(False, False, True),
        align_orientation=maintain_target_orientation,
        gripper_target_state="hold",
    ):
        return False
    if not run_proxy_ee_delta_pose_stage(
        env,
        approach_pose,
        stage_label=f"{stage_label}:move_xy_above_target",
        position_mask=(True, True, False),
        align_orientation=maintain_target_orientation,
        gripper_target_state="hold",
    ):
        return False
    if not _run_proxy_pregrasp_joint_align_and_guard(
        env,
        target_pose=approach_pose,
        stage_label=stage_label,
        planner_cfg=planner_cfg,
        run_proxy_ee_delta_pose_stage=run_proxy_ee_delta_pose_stage,
        get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
    ):
        return False
    _maybe_relatch_runtime_target_pose_for_settle(
        env_unwrapped=env_unwrapped,
        planner_cfg=planner_cfg,
        target_pose=approach_pose,
        stage_label=f"{stage_label}:after_move_xy_above_target",
        relatch_enabled=relatch_target_pose_between_stages,
        get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
    )
    if not run_proxy_stationary_settle(
        env,
        settle_steps=predescent_settle_steps,
        stage_label="PreDescentSettle",
        gripper_target_state="hold",
    ):
        return False
    return run_proxy_guarded_descend_to_object(
        env,
        target_pose,
        initial_actor_p=initial_actor_p_rows if use_batched_pose else initial_actor_p,
        bbox_np=bbox_np,
        stage_label=f"{stage_label}:descent",
        align_orientation=maintain_target_orientation,
    )


def run_proxy_full_approach_to_pregrasp(
    env,
    target_pose,
    *,
    initial_actor_p,
    bbox_np,
    stage_label: str,
    safe_clearance_z: float,
    get_debug_planner_config,
    get_debug_planner_ee_pose_sapien,
    get_debug_planner_ee_pose=None,
    pose_to_numpy_rows=None,
    pose_to_numpy,
    run_proxy_ee_delta_pose_stage,
    run_proxy_stationary_settle,
    refresh_render_state,
    set_debug_planner_last_task_pose,
):
    env_unwrapped = env.unwrapped
    planner_cfg = get_debug_planner_config(env_unwrapped)
    num_envs = int(getattr(env_unwrapped, "num_envs", 1) or 1)
    use_batched_pose = callable(get_debug_planner_ee_pose) and callable(pose_to_numpy_rows) and num_envs > 1
    if num_envs > 1 and not use_batched_pose:
        print(
            f"{_Y}[WARNING] [PlannerDebug] Proxy pregrasp approach '{stage_label}' is running batched "
            f"(num_envs={num_envs}) without batched pose helpers; falling back to env0-first "
            "rise/approach pose construction.{_R}"
        )
    target_pose = to_sapien_pose(target_pose)
    if use_batched_pose:
        current_p_rows, _current_q_rows = pose_to_numpy_rows(get_debug_planner_ee_pose(env_unwrapped))
        current_p = current_p_rows[0]
    else:
        current_pose = get_debug_planner_ee_pose_sapien(env_unwrapped)
        current_p, _current_q = pose_to_numpy(current_pose)
        current_p_rows = None
    target_p_rows = np.asarray(target_pose.p, dtype=np.float32).reshape(-1, 3)
    target_p = target_p_rows[0]
    target_q_rows = np.asarray(target_pose.q, dtype=np.float32).reshape(-1, 4)
    target_q = target_q_rows[0]
    initial_actor_p_rows = np.asarray(initial_actor_p, dtype=np.float32).reshape(-1, 3)
    initial_actor_p = initial_actor_p_rows[0]
    bbox_np = np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3]
    predescent_settle_steps = int(planner_cfg.get("planner_proxy_predescent_settle_steps", 12) or 0)
    relatch_target_pose_between_stages = bool(
        planner_cfg.get("planner_proxy_relatch_ee_target_pose_between_stages", False)
    )
    if use_batched_pose:
        object_top_z_rows = initial_actor_p_rows[:, 2] + 0.5 * bbox_np[2]
        safe_z_rows = np.maximum(current_p_rows[:, 2], object_top_z_rows + float(safe_clearance_z))
        if target_q_rows.shape[0] == 1 and current_p_rows.shape[0] > 1:
            target_q_rows = np.repeat(target_q_rows, current_p_rows.shape[0], axis=0)
        rise_pose = Pose.create_from_pq(
            p=np.stack([current_p_rows[:, 0], current_p_rows[:, 1], safe_z_rows], axis=1).astype(np.float32),
            q=target_q_rows,
        )
        approach_pose = Pose.create_from_pq(
            p=np.stack([target_p_rows[:, 0], target_p_rows[:, 1], safe_z_rows], axis=1).astype(np.float32),
            q=target_q_rows,
        )
        object_top_z = float(object_top_z_rows[0])
        safe_z = float(safe_z_rows[0])
    else:
        object_top_z = float(initial_actor_p[2] + 0.5 * bbox_np[2])
        safe_z = max(float(current_p[2]), float(object_top_z + safe_clearance_z))
        rise_pose = Pose.create_from_pq(
            p=np.asarray([current_p[0], current_p[1], safe_z], dtype=np.float32),
            q=np.asarray(target_q, dtype=np.float32),
        )
        approach_pose = Pose.create_from_pq(
            p=np.asarray([target_p[0], target_p[1], safe_z], dtype=np.float32),
            q=np.asarray(target_q, dtype=np.float32),
        )
    print(
        f"[PlannerDebug] Proxy pregrasp approach '{stage_label}': "
        f"safe_clearance_z={float(safe_clearance_z):.4f} "
        f"object_top_z={object_top_z:.4f} safe_z={safe_z:.4f}"
    )
    if not run_proxy_ee_delta_pose_stage(
        env,
        rise_pose,
        stage_label=f"{stage_label}:rise_to_safe_z",
        position_mask=(False, False, True),
        align_orientation=False,
        gripper_target_state="hold",
    ):
        return False
    if not run_proxy_ee_delta_pose_stage(
        env,
        approach_pose,
        stage_label=f"{stage_label}:move_xy_above_target",
        position_mask=(True, True, False),
        align_orientation=False,
        gripper_target_state="hold",
    ):
        return False
    if not _run_proxy_pregrasp_joint_align_and_guard(
        env,
        target_pose=approach_pose,
        stage_label=stage_label,
        planner_cfg=planner_cfg,
        run_proxy_ee_delta_pose_stage=run_proxy_ee_delta_pose_stage,
        get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
    ):
        return False
    _maybe_relatch_runtime_target_pose_for_settle(
        env_unwrapped=env_unwrapped,
        planner_cfg=planner_cfg,
        target_pose=approach_pose,
        stage_label=f"{stage_label}:after_move_xy_above_target",
        relatch_enabled=relatch_target_pose_between_stages,
        get_debug_planner_ee_pose_sapien=get_debug_planner_ee_pose_sapien,
    )
    if not run_proxy_stationary_settle(
        env,
        settle_steps=predescent_settle_steps,
        stage_label="PreDescentSettle",
        gripper_target_state="hold",
    ):
        return False
    refresh_render_state(env)
    set_debug_planner_last_task_pose(env_unwrapped, approach_pose, stage_name="pregrasp")
    print("[PlannerDebug] Proxy pregrasp approach EXECUTE OK")
    return True
