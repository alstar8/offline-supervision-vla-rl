from __future__ import annotations

import traceback

import numpy as np

PLANNER_DEBUG_STAGE_MAX_POS_ERR_M = 0.1
_Y = "\033[33m"
_R = "\033[0m"


def _to_numpy_1d_debug(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _collect_runtime_arm_controller_diagnostics(env_unwrapped):
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None:
        return {"available": False, "reason": "env_unwrapped.agent unavailable"}
    controller = getattr(agent, "controller", None)
    controllers = getattr(controller, "controllers", {}) if controller is not None else {}
    arm_controller = controllers.get("arm")
    if arm_controller is None:
        return {"available": False, "reason": "runtime arm controller unavailable"}
    diagnostics = {
        "available": True,
        "controller_class": type(arm_controller).__name__,
        "frame": getattr(getattr(arm_controller, "config", None), "frame", None),
        "use_target": getattr(getattr(arm_controller, "config", None), "use_target", None),
        "target_qpos": None,
        "private_target_qpos": None,
    }
    target_qpos = _to_numpy_1d_debug(getattr(arm_controller, "target_qpos", None))
    if target_qpos is not None:
        diagnostics["target_qpos"] = target_qpos
    private_target_qpos = _to_numpy_1d_debug(getattr(arm_controller, "_target_qpos", None))
    if private_target_qpos is not None:
        diagnostics["private_target_qpos"] = private_target_qpos
    return diagnostics


def _collect_runtime_rc5_contact_diagnostics(env_unwrapped, *, min_force: float = 0.5):
    object_actors = getattr(env_unwrapped, "object_actors", {}) or {}
    manip_id = getattr(env_unwrapped, "manip_object_id", None)
    available_ids = [str(obj_id) for obj_id in object_actors.keys()]
    if manip_id is None:
        return {
            "available": False,
            "reason": f"manip_object_id unavailable; available_object_ids={available_ids}",
        }
    target_object = object_actors.get(str(manip_id)) or object_actors.get(manip_id)
    if target_object is None:
        return {
            "available": False,
            "reason": f"manip_object_id={manip_id!r} missing in object_actors; available_object_ids={available_ids}",
        }

    agent = getattr(env_unwrapped, "agent", None)
    required_attrs = ("thumb_tip_link", "index_tip_link", "middle_tip_link", "scene")
    missing_attrs = [attr for attr in required_attrs if not hasattr(agent, attr)]
    if missing_attrs:
        return {
            "available": False,
            "reason": f"missing RC5 contact attrs: {missing_attrs}",
        }

    def _peak_and_flag(link):
        forces = agent.scene.get_pairwise_contact_forces(link, target_object)
        if hasattr(forces, "detach"):
            forces = forces.detach().cpu().numpy()
        forces = np.asarray(forces, dtype=np.float32)
        if forces.size == 0:
            norms = np.zeros((0,), dtype=np.float32)
        else:
            norms = np.linalg.norm(forces.reshape(-1, forces.shape[-1]), axis=1).astype(np.float32)
        peak = float(np.max(norms)) if norms.size > 0 else 0.0
        active = bool(np.any(norms >= float(min_force)))
        return peak, active

    thumb_peak, thumb_active = _peak_and_flag(agent.thumb_tip_link)
    index_peak, index_active = _peak_and_flag(agent.index_tip_link)
    middle_peak, middle_active = _peak_and_flag(agent.middle_tip_link)
    return {
        "available": True,
        "object_id": str(manip_id),
        "thumb_peak": thumb_peak,
        "thumb_active": thumb_active,
        "index_peak": index_peak,
        "index_active": index_active,
        "middle_peak": middle_peak,
        "middle_active": middle_active,
        "min_force": float(min_force),
    }


def _should_apply_rc5_branch_guard(solver) -> bool:
    if type(solver).__name__ != "RC5ArmMotionPlanningSolver":
        return False
    move_group_getter = getattr(solver, "_get_move_group", None)
    if not callable(move_group_getter):
        return False
    return str(move_group_getter()) == "right_tcp_link"


def _snapshot_solver_branch_debug(solver):
    branch_debug_getter = getattr(solver, "get_last_plan_branch_debug", None)
    if not callable(branch_debug_getter):
        return {}
    branch_debug = branch_debug_getter() or {}
    snapshot = {}
    for key, value in dict(branch_debug).items():
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        elif isinstance(value, dict):
            nested = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, np.ndarray):
                    nested[nested_key] = nested_value.copy()
                else:
                    nested[nested_key] = nested_value
            snapshot[key] = nested
        else:
            snapshot[key] = value
    return snapshot


def _planner_branch_guard_accepts(
    solver,
    planner_cfg,
    *,
    stage_label: str,
    method_name: str,
    branch_debug_override=None,
) -> bool:
    if not _should_apply_rc5_branch_guard(solver):
        return True
    branch_guard_enabled = bool(planner_cfg.get("planner_branch_guard_enabled", True))
    if not branch_guard_enabled:
        return True
    branch_debug = branch_debug_override
    if branch_debug is None:
        branch_debug = _snapshot_solver_branch_debug(solver)
    if not branch_debug:
        return True
    start_to_last = branch_debug.get("start_to_last") or {}
    suspicious_indices = np.asarray(start_to_last.get("suspicious_indices", []), dtype=np.int32).reshape(-1)
    raw_delta = np.asarray(start_to_last.get("raw_delta", []), dtype=np.float32).reshape(-1)
    compare_joint_names = list(start_to_last.get("compare_joint_names", []) or [])
    fail_on_wrap_jump = bool(planner_cfg.get("planner_branch_guard_fail_on_wrap_jump", True))
    large_delta_fail_rad = float(planner_cfg.get("planner_branch_guard_large_delta_rad", 2.8))
    large_delta_indices = np.where(np.abs(raw_delta) >= large_delta_fail_rad)[0]
    joint_limit_margin_fail_rad = float(
        planner_cfg.get("planner_branch_guard_joint_limit_margin_rad", 0.05) or 0.0
    )

    problems = []
    if fail_on_wrap_jump and suspicious_indices.size > 0:
        parts = []
        wrapped_delta = np.asarray(start_to_last.get("wrapped_delta", []), dtype=np.float32).reshape(-1)
        wrap_extra = np.asarray(start_to_last.get("wrap_extra", []), dtype=np.float32).reshape(-1)
        for idx in suspicious_indices[:8]:
            joint_name = compare_joint_names[idx] if idx < len(compare_joint_names) else f"joint{idx}"
            parts.append(
                f"{joint_name}: raw_delta={raw_delta[idx]:+.4f} wrapped_delta={wrapped_delta[idx]:+.4f} extra={wrap_extra[idx]:+.4f}"
            )
        problems.append("wrap_jump={" + "; ".join(parts) + "}")
    if large_delta_indices.size > 0:
        parts = []
        for idx in large_delta_indices[:8]:
            joint_name = compare_joint_names[idx] if idx < len(compare_joint_names) else f"joint{idx}"
            parts.append(f"{joint_name}: raw_delta={raw_delta[idx]:+.4f}")
        problems.append(
            f"large_delta>={large_delta_fail_rad:.4f}rad={{" + "; ".join(parts) + "}}"
        )
    last_qpos = branch_debug.get("last_qpos", None)
    if joint_limit_margin_fail_rad > 0.0 and last_qpos is not None:
        last_qpos = np.asarray(last_qpos, dtype=np.float32).reshape(-1)
        active_joints = list(getattr(getattr(solver, "robot", None), "get_active_joints", lambda: [])())
        near_limit_parts = []
        unavailable_limit_parts = []
        for idx, value in enumerate(last_qpos):
            if idx >= len(active_joints):
                break
            try:
                limits = active_joints[idx].get_limits()
            except Exception:
                joint_name = compare_joint_names[idx] if idx < len(compare_joint_names) else f"joint{idx}"
                unavailable_limit_parts.append(f"{joint_name}: get_limits_failed")
                continue
            if hasattr(limits, "cpu"):
                limits = limits.cpu().numpy()
            limits = np.asarray(limits, dtype=np.float32).reshape(-1)
            if limits.shape[0] < 2:
                joint_name = compare_joint_names[idx] if idx < len(compare_joint_names) else f"joint{idx}"
                unavailable_limit_parts.append(f"{joint_name}: malformed_limits={limits.tolist()}")
                continue
            lower, upper = float(limits[0]), float(limits[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                # Modern mplib may expose wraparound/continuous joints with non-finite limits.
                # These joints should not participate in finite-margin rejection.
                continue
            lower_margin = float(value) - lower
            upper_margin = upper - float(value)
            margin = min(lower_margin, upper_margin)
            if margin < joint_limit_margin_fail_rad:
                joint_name = compare_joint_names[idx] if idx < len(compare_joint_names) else f"joint{idx}"
                near_limit_parts.append(
                    f"{joint_name}: qpos={float(value):+.4f} lower_margin={lower_margin:.4f} upper_margin={upper_margin:.4f}"
                )
        if near_limit_parts:
            problems.append(
                f"joint_limit_margin<{joint_limit_margin_fail_rad:.4f}rad={{" + "; ".join(near_limit_parts[:8]) + "}}"
            )
        if unavailable_limit_parts:
            problems.append(
                "joint_limit_unavailable={" + "; ".join(unavailable_limit_parts[:8]) + "}"
            )
    if not problems:
        return True
    print(
        f"[WARNING] [PlannerDebug] {stage_label} branch guard rejected method={method_name}: "
        f"{' '.join(problems)}"
    )
    return False


def _collect_planner_branch_guard_metrics(solver, planner_cfg, *, target_pose, branch_debug_override=None):
    branch_debug = branch_debug_override
    if branch_debug is None:
        branch_debug = _snapshot_solver_branch_debug(solver)
    branch_debug = branch_debug or {}
    start_to_last = branch_debug.get("start_to_last") or {}
    suspicious_indices = np.asarray(start_to_last.get("suspicious_indices", []), dtype=np.int32).reshape(-1)
    raw_delta = np.asarray(start_to_last.get("raw_delta", []), dtype=np.float32).reshape(-1)
    large_delta_fail_rad = float(planner_cfg.get("planner_branch_guard_large_delta_rad", 2.8))
    large_delta_indices = np.where(np.abs(raw_delta) >= large_delta_fail_rad)[0]
    max_abs_raw_delta = float(np.max(np.abs(raw_delta))) if raw_delta.size > 0 else 0.0
    preview_fk_pos_err = float("inf")
    last_qpos = branch_debug.get("last_qpos", None)
    fk_fn = getattr(solver, "_compute_move_group_fk_position_error", None)
    prepare_pose_fn = getattr(solver, "_prepare_target_pose_for_solver", None)
    if last_qpos is not None and callable(fk_fn) and callable(prepare_pose_fn):
        try:
            last_qpos = np.asarray(last_qpos, dtype=np.float32).reshape(-1)
            qpos_for_fk = last_qpos
            current_qpos_getter = getattr(solver, "_get_current_qpos", None)
            if callable(current_qpos_getter):
                current_qpos = np.asarray(current_qpos_getter(), dtype=np.float32).reshape(-1)
                if current_qpos.shape[0] > last_qpos.shape[0]:
                    qpos_for_fk = current_qpos.copy()
                    qpos_for_fk[: last_qpos.shape[0]] = last_qpos
            preview_fk_pos_err = float(
                fk_fn(
                    qpos_for_fk,
                    prepare_pose_fn(target_pose),
                )
            )
        except Exception:
            preview_fk_pos_err = float("inf")
    return {
        "wrap_count": int(suspicious_indices.size),
        "large_delta_count": int(large_delta_indices.size),
        "max_abs_raw_delta": max_abs_raw_delta,
        "preview_fk_pos_err": preview_fk_pos_err,
    }


def _score_planner_pregrasp_preview_candidate(solver, planner_cfg, *, target_pose, branch_debug_override=None):
    metrics = _collect_planner_branch_guard_metrics(
        solver,
        planner_cfg,
        target_pose=target_pose,
        branch_debug_override=branch_debug_override,
    )
    preview_pos_tol = float(
        planner_cfg.get("planner_pregrasp_preview_max_pos_err_m", PLANNER_DEBUG_STAGE_MAX_POS_ERR_M)
    )
    acceptable = (
        metrics["wrap_count"] == 0
        and metrics["large_delta_count"] == 0
        and metrics["preview_fk_pos_err"] <= preview_pos_tol
    )
    score = (
        0 if acceptable else 1,
        metrics["wrap_count"],
        metrics["large_delta_count"],
        round(metrics["preview_fk_pos_err"], 6),
        round(metrics["max_abs_raw_delta"], 6),
    )
    return acceptable, score, metrics


def _collect_planner_execution_consistency_diagnostics(
    solver,
    *,
    target_pose,
    branch_debug_override=None,
):
    diagnostics = {
        "planned_last_arm_qpos": None,
        "actual_arm_qpos": None,
        "arm_tracking_err_norm": float("inf"),
        "arm_tracking_err_max": float("inf"),
        "planned_fk_pos_err": float("inf"),
        "actual_fk_pos_err": float("inf"),
        "follow_path_trace_tail": [],
        "arm_controller_diag": {"available": False, "reason": "not collected"},
        "contact_diag": {"available": False, "reason": "not collected"},
    }
    branch_debug = branch_debug_override
    if branch_debug is None:
        branch_debug = _snapshot_solver_branch_debug(solver)
    branch_debug = branch_debug or {}
    last_qpos = branch_debug.get("last_qpos", None)
    if last_qpos is None:
        return diagnostics

    planned_last_arm_qpos = np.asarray(last_qpos, dtype=np.float32).reshape(-1)
    diagnostics["planned_last_arm_qpos"] = planned_last_arm_qpos.copy()

    current_arm_qpos_getter = getattr(solver, "_get_current_arm_qpos", None)
    if callable(current_arm_qpos_getter):
        try:
            actual_arm_qpos = np.asarray(current_arm_qpos_getter(), dtype=np.float32).reshape(-1)
            diagnostics["actual_arm_qpos"] = actual_arm_qpos.copy()
            compare_len = min(planned_last_arm_qpos.shape[0], actual_arm_qpos.shape[0])
            if compare_len > 0:
                arm_delta = actual_arm_qpos[:compare_len] - planned_last_arm_qpos[:compare_len]
                diagnostics["arm_tracking_err_norm"] = float(np.linalg.norm(arm_delta))
                diagnostics["arm_tracking_err_max"] = float(np.max(np.abs(arm_delta)))
        except Exception:
            pass

    fk_fn = getattr(solver, "_compute_move_group_fk_position_error", None)
    prepare_pose_fn = getattr(solver, "_prepare_target_pose_for_solver", None)
    current_qpos_getter = getattr(solver, "_get_current_qpos", None)
    if callable(fk_fn) and callable(prepare_pose_fn):
        try:
            planner_target_pose = prepare_pose_fn(target_pose)
            qpos_for_planned_fk = planned_last_arm_qpos
            qpos_for_actual_fk = None
            if callable(current_qpos_getter):
                current_qpos = np.asarray(current_qpos_getter(), dtype=np.float32).reshape(-1)
                if current_qpos.shape[0] > planned_last_arm_qpos.shape[0]:
                    qpos_for_planned_fk = current_qpos.copy()
                    qpos_for_planned_fk[: planned_last_arm_qpos.shape[0]] = planned_last_arm_qpos
                    qpos_for_actual_fk = current_qpos.copy()
            diagnostics["planned_fk_pos_err"] = float(fk_fn(qpos_for_planned_fk, planner_target_pose))
            if qpos_for_actual_fk is not None:
                diagnostics["actual_fk_pos_err"] = float(fk_fn(qpos_for_actual_fk, planner_target_pose))
        except Exception:
            pass

    follow_path_trace = getattr(solver, "_last_follow_path_trace", None)
    if isinstance(follow_path_trace, list) and len(follow_path_trace) > 0:
        tail = []
        for item in follow_path_trace[-5:]:
            if not isinstance(item, dict):
                continue
            tail.append(
                {
                    "step_index": int(item.get("step_index", -1)),
                    "commanded_arm_qpos": np.asarray(
                        item.get("commanded_arm_qpos", []),
                        dtype=np.float32,
                    ).reshape(-1),
                    "actual_arm_qpos": np.asarray(
                        item.get("actual_arm_qpos", []),
                        dtype=np.float32,
                    ).reshape(-1),
                    "arm_tracking_err": np.asarray(
                        item.get("arm_tracking_err", []),
                        dtype=np.float32,
                    ).reshape(-1),
                }
            )
        diagnostics["follow_path_trace_tail"] = tail

    env_unwrapped = getattr(getattr(solver, "env", None), "unwrapped", None)
    if env_unwrapped is not None:
        diagnostics["arm_controller_diag"] = _collect_runtime_arm_controller_diagnostics(env_unwrapped)
        diagnostics["contact_diag"] = _collect_runtime_rc5_contact_diagnostics(env_unwrapped, min_force=0.5)
    return diagnostics


def _print_planner_execution_consistency_diagnostics(
    solver,
    *,
    stage_label: str,
    method_name: str,
    target_pose,
    branch_debug_override=None,
):
    diagnostics = _collect_planner_execution_consistency_diagnostics(
        solver,
        target_pose=target_pose,
        branch_debug_override=branch_debug_override,
    )
    planned_last_arm_qpos = diagnostics.get("planned_last_arm_qpos", None)
    actual_arm_qpos = diagnostics.get("actual_arm_qpos", None)
    planned_last_str = "unavailable"
    if planned_last_arm_qpos is not None:
        planned_last_str = np.array2string(planned_last_arm_qpos, precision=4, suppress_small=True)
    actual_arm_str = "unavailable"
    if actual_arm_qpos is not None:
        actual_arm_str = np.array2string(actual_arm_qpos, precision=4, suppress_small=True)
    print(
        f"[PlannerDebug] {stage_label} execution consistency method={method_name} "
        f"planned_last_arm_qpos={planned_last_str} "
        f"actual_arm_qpos={actual_arm_str} "
        f"arm_tracking_err_norm={diagnostics['arm_tracking_err_norm']:.4f} "
        f"arm_tracking_err_max={diagnostics['arm_tracking_err_max']:.4f} "
        f"planned_fk_pos_err={diagnostics['planned_fk_pos_err']:.4f} m "
        f"actual_fk_pos_err={diagnostics['actual_fk_pos_err']:.4f} m"
    )
    for trace_item in diagnostics.get("follow_path_trace_tail", []):
        commanded = np.array2string(
            np.asarray(trace_item["commanded_arm_qpos"], dtype=np.float32),
            precision=4,
            suppress_small=True,
        )
        actual = np.array2string(
            np.asarray(trace_item["actual_arm_qpos"], dtype=np.float32),
            precision=4,
            suppress_small=True,
        )
        err = np.array2string(
            np.asarray(trace_item["arm_tracking_err"], dtype=np.float32),
            precision=4,
            suppress_small=True,
        )
        print(
            f"[PlannerDebug] {stage_label} follow_path tail step={trace_item['step_index']} "
            f"commanded_arm_qpos={commanded} actual_arm_qpos={actual} arm_tracking_err={err}"
        )
    arm_controller_diag = diagnostics.get("arm_controller_diag", {}) or {}
    if arm_controller_diag.get("available", False):
        target_qpos = arm_controller_diag.get("target_qpos", None)
        private_target_qpos = arm_controller_diag.get("private_target_qpos", None)
        target_qpos_str = "unavailable"
        if target_qpos is not None:
            target_qpos_str = np.array2string(np.asarray(target_qpos, dtype=np.float32), precision=4, suppress_small=True)
        private_target_qpos_str = "unavailable"
        if private_target_qpos is not None:
            private_target_qpos_str = np.array2string(
                np.asarray(private_target_qpos, dtype=np.float32), precision=4, suppress_small=True
            )
        print(
            f"[PlannerDebug] {stage_label} arm_controller: "
            f"cls={arm_controller_diag.get('controller_class')} "
            f"frame={arm_controller_diag.get('frame')} "
            f"use_target={arm_controller_diag.get('use_target')} "
            f"target_qpos={target_qpos_str} "
            f"_target_qpos={private_target_qpos_str}"
        )
    else:
        print(
            f"[PlannerDebug] {stage_label} arm_controller: unavailable "
            f"reason={arm_controller_diag.get('reason', 'unknown')}"
        )
    contact_diag = diagnostics.get("contact_diag", {}) or {}
    if contact_diag.get("available", False):
        print(
            f"[PlannerDebug] {stage_label} contact_diag: "
            f"object_id={contact_diag.get('object_id')} "
            f"thumb_peak={contact_diag['thumb_peak']:.4f} active={contact_diag['thumb_active']} "
            f"index_peak={contact_diag['index_peak']:.4f} active={contact_diag['index_active']} "
            f"middle_peak={contact_diag['middle_peak']:.4f} active={contact_diag['middle_active']} "
            f"min_force={contact_diag['min_force']:.4f}"
        )
    else:
        print(
            f"[PlannerDebug] {stage_label} contact_diag: unavailable "
            f"reason={contact_diag.get('reason', 'unknown')}"
        )


def _execute_pregrasp_planner_method(solver, target_pose, method_name: str, *, dry_run: bool):
    if method_name == "local_ik":
        if not hasattr(solver, "move_to_pose_with_local_ik"):
            raise RuntimeError(
                "Pregrasp method 'local_ik' was requested, but this solver does not expose "
                "move_to_pose_with_local_ik()"
            )
        return solver.move_to_pose_with_local_ik(target_pose, dry_run=dry_run)
    if method_name == "rrtconnect":
        return solver.move_to_pose_with_RRTConnect(target_pose, dry_run=dry_run)
    if method_name == "screw":
        return solver.move_to_pose_with_screw(target_pose, dry_run=dry_run)
    raise ValueError(f"Unsupported planner object probe method candidate: {method_name}")


def _repeat_planner_result_waypoints(result, *, repeat_each_step: int):
    repeat_each_step = int(repeat_each_step)
    if repeat_each_step < 1:
        raise ValueError(
            f"planner pregrasp waypoint repeat must be >= 1, got {repeat_each_step}"
        )
    if repeat_each_step == 1:
        return result

    expanded = dict(result)
    positions = np.asarray(result["position"], dtype=np.float32)
    expanded["position"] = np.repeat(positions, repeat_each_step, axis=0)
    if "velocity" in result and result["velocity"] is not None:
        velocities = np.asarray(result["velocity"], dtype=np.float32)
        expanded["velocity"] = np.repeat(velocities, repeat_each_step, axis=0)
    return expanded


def run_planner_object_pregrasp_probe(
    env,
    *,
    method="auto",
    extra_clearance=0.10,
    execute=False,
    backend="planner",
    extract_planner_base_pose,
    resolve_planner_debug_solver_class,
    is_proxy_ee_delta_backend,
    is_proxy_then_planner_backend,
    is_ee_delta_control_mode,
    maybe_seed_proxy_start_pose,
    build_object_pregrasp_target,
    run_proxy_full_approach_to_pregrasp,
    planner_visuals_supported,
    configure_debug_planner_solver_runtime,
    get_planner_recording_kwargs,
    get_object_specific_planner_profile,
    refresh_render_state,
    pose_to_numpy,
    get_debug_planner_ee_pose,
    set_debug_planner_last_task_pose,
):
    agent_uid = getattr(env.unwrapped.agent, "uid", "unknown")
    control_mode = getattr(env.unwrapped, "control_mode", None)
    base_pose = extract_planner_base_pose(env)
    PlannerClass = resolve_planner_debug_solver_class(agent_uid)
    mode_label = "execute" if execute else "dry-run"
    print(
        f"[PlannerDebug] Starting object-aware pregrasp {mode_label} for agent_uid='{agent_uid}' "
        f"with solver_class={PlannerClass.__name__}, method={method}"
    )
    if is_proxy_ee_delta_backend(backend) or is_proxy_then_planner_backend(backend):
        if not is_ee_delta_control_mode(control_mode):
            print(
                f"[PlannerDebug] Proxy pregrasp requires an EE-delta control_mode, got '{control_mode}'."
            )
            return False
        if not execute:
            print("[PlannerDebug] Proxy pregrasp does not support dry-run mode.")
            return False
        try:
            maybe_seed_proxy_start_pose(env, reason="proxy pregrasp")
            planner_cfg = getattr(env.unwrapped, "_debug_planner_config", {}) or {}
            safe_clearance_z = float(planner_cfg.get("planner_proxy_safe_clearance_z", 0.10))
            manip_id, actor_p, bbox_np, target_pose = build_object_pregrasp_target(
                env, extra_clearance=extra_clearance
            )
            print(
                f"[PlannerDebug] Proxy pregrasp target manip_object_id={manip_id} "
                f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
                f"bbox_world={np.array2string(bbox_np, precision=4, suppress_small=True)}"
            )
            print(
                f"[PlannerDebug] Proxy pregrasp target pose p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"q={np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True)}"
            )
            return run_proxy_full_approach_to_pregrasp(
                env,
                target_pose,
                initial_actor_p=actor_p,
                bbox_np=bbox_np,
                stage_label="Pregrasp",
                safe_clearance_z=safe_clearance_z,
            )
        except Exception as exc:
            print(f"[PlannerDebug] Proxy pregrasp FAILED: {type(exc).__name__}: {exc}")
            print("[PlannerDebug] Traceback:")
            print(traceback.format_exc().rstrip())
            return False
    if agent_uid == "rc5_aero_hand_openr2s_rl" or is_ee_delta_control_mode(control_mode):
        print(
            f"[PlannerDebug] Skipping object probe: current control_mode='{control_mode}' is EE-delta/RL. "
            "The first object-aware planner probe is wired for the joint-space planner path "
            "(*_rl robot_uids + 'pd_joint_pos')."
        )
        return False
    try:
        enable_target_visual = planner_visuals_supported(env)
        if not enable_target_visual:
            print(
                "[PlannerDebug] Target marker visualization is disabled on CUDA PhysX. "
                "Planning/execution continues without a scene marker."
            )
        solver = configure_debug_planner_solver_runtime(
            PlannerClass(
                env,
                debug=True,
                vis=True,
                base_pose=base_pose,
                visualize_target_grasp_pose=enable_target_visual,
                print_env_info=False,
                **get_planner_recording_kwargs(env),
            )
        )
        planner_cfg = getattr(env.unwrapped, "_debug_planner_config", {}) or {}
        base_backoff = float(planner_cfg.get("planner_pregrasp_radial_backoff", 0.05))
        backoff_candidates = []
        for candidate in [base_backoff, base_backoff + 0.03, base_backoff + 0.06, max(base_backoff - 0.02, 0.0)]:
            if candidate not in backoff_candidates:
                backoff_candidates.append(candidate)

        accepted_candidate_found = False
        target_pose = None
        manip_id = None
        actor_p = None
        bbox_np = None
        candidate_records = []
        for candidate_idx, radial_backoff in enumerate(backoff_candidates, start=1):
            manip_id, actor_p, bbox_np, target_pose = build_object_pregrasp_target(
                env,
                extra_clearance=extra_clearance,
                radial_backoff_override=radial_backoff,
            )
            print(
                f"[PlannerDebug] Object probe manip_object_id={manip_id} "
                f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
                f"bbox_world={np.array2string(bbox_np, precision=4, suppress_small=True)}"
            )
            print(
                f"[PlannerDebug] Object probe candidate {candidate_idx}/{len(backoff_candidates)} "
                f"radial_backoff={radial_backoff:.4f}"
            )
            print(
                f"[PlannerDebug] Object probe target pose p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"q={np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True)}"
            )
            if enable_target_visual:
                refresh_render_state(env)
                refresh_render_state(env)
                print("[PlannerDebug] Target grasp marker updated in viewer (forced refresh).")

            object_profile = get_object_specific_planner_profile(planner_cfg, manip_id)
            if method == "auto":
                if object_profile is not None:
                    if not hasattr(solver, "move_to_pose_with_local_ik"):
                        raise RuntimeError(
                            "RC5 object-calibrated pregrasp requires local IK in auto mode, "
                            "but this solver does not expose move_to_pose_with_local_ik()"
                        )
                    print(f"{_Y}[PlannerDebug] Pregrasp dispatch: auto -> local_ik (object-calibrated RC5 target){_R}")
                    method_name = "local_ik"
                else:
                    print(f"{_Y}[PlannerDebug] Pregrasp dispatch: auto -> RRTConnect (planner stability){_R}")
                    method_name = "rrtconnect"
            elif method in {"rrtconnect", "screw", "local_ik"}:
                method_name = method
            else:
                raise ValueError(f"Unsupported planner object probe method: {method}")

            preview_result = _execute_pregrasp_planner_method(
                solver,
                target_pose,
                method_name,
                dry_run=True,
            )
            preview_ok = preview_result != -1
            preview_branch_ok = False
            preview_accepts = False
            preview_score = None
            preview_metrics = {}
            if preview_ok:
                preview_branch_debug = _snapshot_solver_branch_debug(solver)
                preview_branch_ok = _planner_branch_guard_accepts(
                    solver,
                    planner_cfg,
                    stage_label="PregraspPreview",
                    method_name=method_name,
                    branch_debug_override=preview_branch_debug,
                )
                preview_accepts, preview_score, preview_metrics = _score_planner_pregrasp_preview_candidate(
                    solver,
                    planner_cfg,
                    target_pose=target_pose,
                    branch_debug_override=preview_branch_debug,
                )
                preview_accepts = bool(preview_branch_ok and preview_accepts)
                if not preview_accepts:
                    preview_score = (
                        1,
                        int(preview_metrics.get("wrap_count", 0)),
                        int(preview_metrics.get("large_delta_count", 0)),
                        round(float(preview_metrics.get("preview_fk_pos_err", float("inf"))), 6),
                        round(float(preview_metrics.get("max_abs_raw_delta", float("inf"))), 6),
                    )
                print(
                    f"[PlannerDebug] Pregrasp preview candidate {candidate_idx}/{len(backoff_candidates)} "
                    f"method={method_name} "
                    f"preview_branch_ok={preview_branch_ok} "
                    f"preview_accepts={preview_accepts} "
                    f"preview_score={preview_score} "
                    f"wrap_count={preview_metrics['wrap_count']} "
                    f"large_delta_count={preview_metrics['large_delta_count']} "
                    f"preview_fk_pos_err={preview_metrics['preview_fk_pos_err']:.4f} m "
                    f"max_abs_raw_delta={preview_metrics['max_abs_raw_delta']:.4f} rad"
                )
                if not preview_branch_ok:
                    refresh_render_state(env)
            else:
                preview_branch_debug = None

            candidate_records.append(
                {
                    "candidate_idx": candidate_idx,
                    "radial_backoff": radial_backoff,
                    "method_name": method_name,
                    "target_pose": target_pose,
                    "manip_id": manip_id,
                    "actor_p": actor_p,
                    "bbox_np": bbox_np,
                    "preview_result": preview_result,
                    "preview_ok": preview_ok,
                    "preview_branch_ok": preview_branch_ok,
                    "preview_branch_debug": preview_branch_debug,
                    "preview_accepts": preview_accepts,
                    "preview_score": preview_score if preview_score is not None else (1, 999, 999, float("inf"), float("inf")),
                    "preview_metrics": preview_metrics,
                }
            )

        valid_records = [record for record in candidate_records if record["preview_ok"]]
        accepted_records = [record for record in valid_records if record["preview_accepts"]]
        selected_record = None
        if accepted_records:
            selected_record = min(accepted_records, key=lambda record: record["preview_score"])
            accepted_candidate_found = True
        elif valid_records:
            selected_record = min(valid_records, key=lambda record: record["preview_score"])

        if selected_record is None:
            print(f"[PlannerDebug] Object probe FAILED via method={method}")
            solver.close()
            return False

        if not accepted_candidate_found:
            print(
                f"[PlannerDebug] Object probe FAILED via method={method}: "
                f"no acceptable preview candidate was found. "
                f"best_preview_score={selected_record['preview_score']} "
                f"best_candidate_idx={selected_record['candidate_idx']} "
                f"radial_backoff={selected_record['radial_backoff']:.4f}"
            )
            solver.close()
            return False

        target_pose = selected_record["target_pose"]
        manip_id = selected_record["manip_id"]
        actor_p = selected_record["actor_p"]
        bbox_np = selected_record["bbox_np"]
        method_name = selected_record["method_name"]
        print(
            f"[PlannerDebug] Selected pregrasp candidate {selected_record['candidate_idx']}/{len(backoff_candidates)} "
            f"radial_backoff={selected_record['radial_backoff']:.4f} "
            f"preview_score={selected_record['preview_score']}"
        )

        result = selected_record["preview_result"]
        if execute:
            pregrasp_refine_steps = int(planner_cfg.get("planner_pregrasp_refine_steps", 0) or 0)
            pregrasp_waypoint_repeat = int(
                planner_cfg.get("planner_pregrasp_waypoint_repeat", 1) or 1
            )
            result = _repeat_planner_result_waypoints(
                result,
                repeat_each_step=pregrasp_waypoint_repeat,
            )
            print(
                f"[PlannerDebug] Reusing preview planner result for pregrasp execute "
                f"(candidate {selected_record['candidate_idx']}/{len(backoff_candidates)}, "
                f"method={method_name}, waypoint_repeat={pregrasp_waypoint_repeat}, "
                f"refine_steps={pregrasp_refine_steps})"
            )
            result = solver.follow_path(result, refine_steps=pregrasp_refine_steps)
            accepted = result != -1
            if accepted and not _planner_branch_guard_accepts(
                solver,
                planner_cfg,
                stage_label="Pregrasp",
                method_name=method_name,
                branch_debug_override=selected_record.get("preview_branch_debug"),
            ):
                refresh_render_state(env)
                accepted = False
            if not accepted:
                print(f"[PlannerDebug] Object probe FAILED via method={method}")
                solver.close()
                return False
            final_tcp_p, _final_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = PLANNER_DEBUG_STAGE_MAX_POS_ERR_M
            print(
                f"[PlannerDebug] Object probe final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m method={method_name}"
            )
            if final_err > max_accept_err:
                _print_planner_execution_consistency_diagnostics(
                    solver,
                    stage_label="Pregrasp",
                    method_name=method_name,
                    target_pose=target_pose,
                    branch_debug_override=selected_record.get("preview_branch_debug"),
                )
                print(
                    f"[WARNING] [PlannerDebug] Pregrasp final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m after method={method_name}; "
                    "treating stage as FAILED."
                )
                refresh_render_state(env)
                solver.close()
                return False

        if execute:
            refresh_render_state(env)
            set_debug_planner_last_task_pose(env.unwrapped, target_pose, stage_name="pregrasp")
            print("[PlannerDebug] Object probe EXECUTE OK")
            solver.close()
            return True

        status = result.get("status", "unknown")
        n_steps = int(result["position"].shape[0]) if "position" in result else -1
        print(f"[PlannerDebug] Object probe OK: status={status}, n_steps={n_steps}")
        solver.close()
        return True
    except Exception as exc:
        print(f"[PlannerDebug] Object probe FAILED: {type(exc).__name__}: {exc}")
        print("[PlannerDebug] Traceback:")
        print(traceback.format_exc().rstrip())
        return False


def run_planner_object_descend(
    env,
    *,
    method="rrtconnect",
    extra_clearance=0.03,
    execute=True,
    backend="local_ik",
    extract_planner_base_pose,
    resolve_planner_debug_solver_class,
    is_proxy_ee_delta_backend,
    is_ee_delta_control_mode,
    build_object_descend_target,
    run_proxy_ee_delta_pose_stage,
    planner_visuals_supported,
    configure_debug_planner_solver_runtime,
    get_planner_recording_kwargs,
    refresh_render_state,
    pose_to_numpy,
    get_debug_planner_ee_pose,
    run_linear_approach_waypoints,
    execute_planner_pose_with_backend=None,
    execute_real_planner_pose_with_backend=None,
    set_debug_planner_last_task_pose,
):
    planner_pose_executor = (
        execute_real_planner_pose_with_backend
        if execute_real_planner_pose_with_backend is not None
        else execute_planner_pose_with_backend
    )
    if planner_pose_executor is None:
        raise ValueError("Planner descend requires a real planner pose executor helper.")
    agent_uid = getattr(env.unwrapped.agent, "uid", "unknown")
    control_mode = getattr(env.unwrapped, "control_mode", None)
    base_pose = extract_planner_base_pose(env)
    PlannerClass = resolve_planner_debug_solver_class(agent_uid)
    mode_label = "execute" if execute else "dry-run"
    print(
        f"[PlannerDebug] Starting object-aware descend {mode_label} for agent_uid='{agent_uid}' "
        f"with solver_class={PlannerClass.__name__}, method={method}, backend={backend}"
    )
    if is_proxy_ee_delta_backend(backend):
        if not is_ee_delta_control_mode(control_mode):
            print(
                f"[PlannerDebug] Proxy descend requires an EE-delta control_mode, got '{control_mode}'."
            )
            return False
        if not execute:
            print("[PlannerDebug] Proxy descend does not support dry-run mode.")
            return False
        try:
            manip_id, actor_p, bbox_np, target_pose = build_object_descend_target(env, extra_clearance=extra_clearance)
            print(
                f"[PlannerDebug] Proxy descend target manip_object_id={manip_id} "
                f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
                f"bbox_world={np.array2string(bbox_np, precision=4, suppress_small=True)}"
            )
            print(
                f"[PlannerDebug] Proxy descend target pose "
                f"p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"q={np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True)}"
            )
            if not run_proxy_ee_delta_pose_stage(
                env,
                target_pose,
                stage_label="Descend",
                position_mask=(True, True, True),
                align_orientation=True,
            ):
                return False
            final_tcp_p, _ = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = PLANNER_DEBUG_STAGE_MAX_POS_ERR_M
            print(
                f"[PlannerDebug] Proxy descend final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] Proxy descend final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m; treating stage as FAILED."
                )
                refresh_render_state(env)
                return False
            set_debug_planner_last_task_pose(env.unwrapped, target_pose, stage_name="descend")
            env.unwrapped._planner_last_task_bbox_np = np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3].copy()
            refresh_render_state(env)
            print("[PlannerDebug] Proxy descend EXECUTE OK")
            return True
        except Exception as exc:
            print(f"[PlannerDebug] Proxy descend FAILED: {type(exc).__name__}: {exc}")
            print("[PlannerDebug] Traceback:")
            print(traceback.format_exc().rstrip())
            return False
    if is_ee_delta_control_mode(control_mode):
        print(
            f"[PlannerDebug] Skipping descend probe: current control_mode='{control_mode}' is EE-delta/RL. "
            "The descend planner probe is wired for the joint-space planner path "
            "(*_rl robot_uids + 'pd_joint_pos')."
        )
        return False
    try:
        enable_target_visual = planner_visuals_supported(env)
        if not enable_target_visual:
            print(
                "[PlannerDebug] Target marker visualization is disabled on CUDA PhysX. "
                "Planning/execution continues without a scene marker."
            )
        solver = configure_debug_planner_solver_runtime(
            PlannerClass(
                env,
                debug=True,
                vis=True,
                base_pose=base_pose,
                visualize_target_grasp_pose=enable_target_visual,
                print_env_info=False,
                **get_planner_recording_kwargs(env),
            )
        )
        manip_id, actor_p, bbox_np, target_pose = build_object_descend_target(env, extra_clearance=extra_clearance)
        print(
            f"[PlannerDebug] Descend target manip_object_id={manip_id} "
            f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
            f"bbox_world={np.array2string(bbox_np, precision=4, suppress_small=True)}"
        )
        print(
            f"[PlannerDebug] Descend target pose p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
            f"q={np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True)}"
        )
        if enable_target_visual:
            refresh_render_state(env)
            refresh_render_state(env)
            print("[PlannerDebug] Descend marker updated in viewer (forced refresh).")

        planner_cfg = getattr(env.unwrapped, "_debug_planner_config", {}) or {}
        approach_waypoints = max(int(planner_cfg.get("planner_approach_waypoints", 0) or 0), 0)
        approach_waypoint_mode = str(
            planner_cfg.get("planner_approach_waypoint_mode", "fixed") or "fixed"
        ).strip().lower()
        if approach_waypoints > 0:
            if not run_linear_approach_waypoints(
                env,
                solver,
                target_pose,
                waypoint_count=approach_waypoints,
                waypoint_mode=approach_waypoint_mode,
                execute=execute,
                backend=backend,
                method=method,
                stage_label="Descend",
            ):
                solver.close()
                return False

        result = planner_pose_executor(
            solver,
            target_pose,
            execute=execute,
            backend=backend,
            method=method,
            planner_class_name=PlannerClass.__name__,
            stage_label="Descend",
        )
        if result == -1:
            print(f"[PlannerDebug] Descend FAILED via method={method}")
            solver.close()
            return False
        if not _planner_branch_guard_accepts(
            solver,
            planner_cfg,
            stage_label="Descend",
            method_name=method,
        ):
            refresh_render_state(env)
            solver.close()
            return False

        if execute:
            final_tcp_p, _final_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = PLANNER_DEBUG_STAGE_MAX_POS_ERR_M
            print(
                f"[PlannerDebug] Descend final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] Descend final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m; treating stage as FAILED."
                )
                refresh_render_state(env)
                solver.close()
                return False
            set_debug_planner_last_task_pose(env.unwrapped, target_pose, stage_name="descend")
            env.unwrapped._planner_last_task_bbox_np = np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3].copy()
            refresh_render_state(env)
            print("[PlannerDebug] Descend EXECUTE OK")
            solver.close()
            return True

        status = result.get("status", "unknown")
        n_steps = int(result["position"].shape[0]) if "position" in result else -1
        print(f"[PlannerDebug] Descend OK: status={status}, n_steps={n_steps}")
        solver.close()
        return True
    except Exception as exc:
        print(f"[PlannerDebug] Descend FAILED: {type(exc).__name__}: {exc}")
        print("[PlannerDebug] Traceback:")
        print(traceback.format_exc().rstrip())
        return False


def run_planner_full_approach_to_descend(
    env,
    *,
    method="auto",
    extra_clearance=0.03,
    execute=True,
    backend="local_ik",
    extract_planner_base_pose,
    resolve_planner_debug_solver_class,
    is_proxy_ee_delta_backend,
    is_ee_delta_control_mode,
    build_object_descend_target,
    run_proxy_full_approach_to_descend,
    refresh_render_state,
    pose_to_numpy,
    get_debug_planner_ee_pose,
    set_debug_planner_last_task_pose,
    planner_visuals_supported,
    configure_debug_planner_solver_runtime,
    get_planner_recording_kwargs,
    run_linear_approach_waypoints,
    execute_planner_pose_with_backend=None,
    execute_real_planner_pose_with_backend=None,
):
    planner_pose_executor = (
        execute_real_planner_pose_with_backend
        if execute_real_planner_pose_with_backend is not None
        else execute_planner_pose_with_backend
    )
    if planner_pose_executor is None:
        raise ValueError("Full approach requires a real planner pose executor helper.")
    agent_uid = getattr(env.unwrapped.agent, "uid", "unknown")
    control_mode = getattr(env.unwrapped, "control_mode", None)
    base_pose = extract_planner_base_pose(env)
    PlannerClass = resolve_planner_debug_solver_class(agent_uid)
    mode_label = "execute" if execute else "dry-run"
    print(
        f"[PlannerDebug] Starting full approach {mode_label} for agent_uid='{agent_uid}' "
        f"with solver_class={PlannerClass.__name__}, method={method}, backend={backend}"
    )
    if is_proxy_ee_delta_backend(backend):
        if not is_ee_delta_control_mode(control_mode):
            print(
                f"[PlannerDebug] Proxy full approach requires an EE-delta control_mode, got '{control_mode}'."
            )
            return False
        if not execute:
            print("[PlannerDebug] Proxy full approach does not support dry-run mode.")
            return False
        try:
            planner_cfg = getattr(env.unwrapped, "_debug_planner_config", {}) or {}
            safe_clearance_z = float(planner_cfg.get("planner_proxy_safe_clearance_z", 0.10))
            manip_id, actor_p, bbox_np, target_pose = build_object_descend_target(env, extra_clearance=extra_clearance)
            print(
                f"[PlannerDebug] Proxy full approach target manip_object_id={manip_id} "
                f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
                f"bbox_world={np.array2string(bbox_np, precision=4, suppress_small=True)}"
            )
            print(
                f"[PlannerDebug] Proxy full approach final descend target pose "
                f"p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"q={np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True)}"
            )
            if not run_proxy_full_approach_to_descend(
                env,
                target_pose,
                initial_actor_p=actor_p,
                bbox_np=bbox_np,
                stage_label="FullApproach",
                safe_clearance_z=safe_clearance_z,
            ):
                return False
            final_tcp_p, _final_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = PLANNER_DEBUG_STAGE_MAX_POS_ERR_M
            print(
                f"[PlannerDebug] Proxy full approach final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] Proxy full approach final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m; treating stage as FAILED."
                )
                refresh_render_state(env)
                return False
            refresh_render_state(env)
            set_debug_planner_last_task_pose(env.unwrapped, target_pose, stage_name="descend")
            env.unwrapped._planner_last_task_bbox_np = np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3].copy()
            print("[PlannerDebug] Proxy full approach EXECUTE OK")
            return True
        except Exception as exc:
            print(f"[PlannerDebug] Proxy full approach FAILED: {type(exc).__name__}: {exc}")
            print("[PlannerDebug] Traceback:")
            print(traceback.format_exc().rstrip())
            return False
    if agent_uid == "rc5_aero_hand_openr2s_rl" or is_ee_delta_control_mode(control_mode):
        print(
            f"[PlannerDebug] Skipping full approach: current control_mode='{control_mode}' is EE-delta/RL. "
            "The full-approach planner probe is wired for the joint-space planner path "
            "(*_rl robot_uids + 'pd_joint_pos')."
        )
        return False
    try:
        enable_target_visual = planner_visuals_supported(env)
        if not enable_target_visual:
            print(
                "[PlannerDebug] Target marker visualization is disabled on CUDA PhysX. "
                "Planning/execution continues without a scene marker."
            )
        solver = configure_debug_planner_solver_runtime(
            PlannerClass(
                env,
                debug=True,
                vis=True,
                base_pose=base_pose,
                visualize_target_grasp_pose=enable_target_visual,
                print_env_info=False,
                **get_planner_recording_kwargs(env),
            )
        )
        planner_cfg = getattr(env.unwrapped, "_debug_planner_config", {}) or {}
        approach_waypoints = max(int(planner_cfg.get("planner_approach_waypoints", 0) or 0), 0)
        approach_waypoint_mode = str(
            planner_cfg.get("planner_approach_waypoint_mode", "fixed") or "fixed"
        ).strip().lower()
        if approach_waypoints <= 0:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Full approach requested without planner_approach_waypoints > 0; "
                "falling back to stage-split pregrasp/descend behavior would be required elsewhere.{_R}"
            )
            solver.close()
            return False

        manip_id, actor_p, bbox_np, target_pose = build_object_descend_target(env, extra_clearance=extra_clearance)
        print(
            f"[PlannerDebug] Full approach target manip_object_id={manip_id} "
            f"actor_p={np.array2string(actor_p, precision=4, suppress_small=True)} "
            f"bbox_world={np.array2string(bbox_np, precision=4, suppress_small=True)}"
        )
        print(
            f"[PlannerDebug] Full approach final descend target pose "
            f"p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
            f"q={np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True)}"
        )
        print(
            f"{_Y}[PlannerDebug] Full approach mode active: bypassing split pregrasp/descend stages "
            f"and using a single waypoint chain from current EE pose to the final grasp/descend target "
            f"with waypoint_count={approach_waypoints}, waypoint_mode='{approach_waypoint_mode}'.{_R}"
        )
        if not run_linear_approach_waypoints(
            env,
            solver,
            target_pose,
            waypoint_count=approach_waypoints,
            waypoint_mode=approach_waypoint_mode,
            execute=execute,
            backend=backend,
            method=method,
            stage_label="FullApproach",
        ):
            solver.close()
            return False

        result = planner_pose_executor(
            solver,
            target_pose,
            execute=execute,
            backend=backend,
            method=method,
            planner_class_name=PlannerClass.__name__,
            stage_label="FullApproach final target",
        )
        if result == -1:
            print(f"[PlannerDebug] Full approach FAILED at final target via method={method}")
            solver.close()
            return False

        if execute:
            final_tcp_p, _final_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = PLANNER_DEBUG_STAGE_MAX_POS_ERR_M
            print(
                f"[PlannerDebug] Full approach final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] Full approach final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m; treating stage as FAILED."
                )
                refresh_render_state(env)
                solver.close()
                return False
            refresh_render_state(env)
            set_debug_planner_last_task_pose(env.unwrapped, target_pose, stage_name="descend")
            env.unwrapped._planner_last_task_bbox_np = np.asarray(bbox_np, dtype=np.float32).reshape(-1)[:3].copy()
            print("[PlannerDebug] Full approach EXECUTE OK")
            solver.close()
            return True

        status = result.get("status", "unknown")
        n_steps = int(result["position"].shape[0]) if "position" in result else -1
        print(f"[PlannerDebug] Full approach OK: status={status}, n_steps={n_steps}")
        solver.close()
        return True
    except Exception as exc:
        print(f"[PlannerDebug] Full approach FAILED: {type(exc).__name__}: {exc}")
        print("[PlannerDebug] Traceback:")
        print(traceback.format_exc().rstrip())
        return False


def run_proxy_pregrasp_probe(*args, **kwargs):
    return run_planner_object_pregrasp_probe(*args, **kwargs)


def run_proxy_descend(*args, **kwargs):
    return run_planner_object_descend(*args, **kwargs)


def run_proxy_full_approach_to_descend(*args, **kwargs):
    return run_planner_full_approach_to_descend(*args, **kwargs)
