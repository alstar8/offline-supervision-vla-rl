from __future__ import annotations

import traceback

import numpy as np

from openreal2sim.simulation.maniskill.scripts.rc5_unified_logging import (
    is_debug_enabled,
    logger,
)

_Y = "\033[33m"
_R = "\033[0m"


def _to_numpy_rows(value):
    if value is None:
        return None
    if hasattr(value, "detach") and callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32).copy()


def _normalize_rows(value, *, num_envs: int, width: int, label: str):
    arr = _to_numpy_rows(value)
    if arr is None:
        return None
    if arr.shape[0] == 1 and int(num_envs) > 1:
        print(
            f"{_Y}[WARNING] [PlannerDebug] {label} is single-env in a batched lift; "
            f"repeating env0 across {int(num_envs)} envs for diagnostics.{_R}"
        )
        arr = np.repeat(arr, int(num_envs), axis=0)
    if arr.shape[0] != int(num_envs):
        raise ValueError(
            f"Expected {label} to have first dimension {int(num_envs)}, got shape={arr.shape}."
        )
    return np.asarray(arr[:, : int(width)], dtype=np.float32).copy()


def _normalize_bool_rows(value, *, num_envs: int, label: str, allow_none: bool = False):
    if value is None:
        return [None] * int(num_envs) if allow_none else [False] * int(num_envs)
    if hasattr(value, "detach") and callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 1 and arr.size == int(num_envs):
        return [bool(item) for item in arr.tolist()]
    arr = _normalize_rows(arr, num_envs=num_envs, width=1, label=label)
    return [bool(item) for item in arr[:, 0].tolist()]


def _extract_pose_position_rows(pose_value, *, num_envs: int, label: str):
    if pose_value is None:
        return None
    raw_pose = getattr(pose_value, "raw_pose", None)
    if raw_pose is not None:
        return _normalize_rows(raw_pose, num_envs=num_envs, width=3, label=label)
    pose_p = getattr(pose_value, "p", None)
    if pose_p is None:
        raise ValueError(f"{label} does not expose pose data for per-env diagnostics.")
    return _normalize_rows(pose_p, num_envs=num_envs, width=3, label=label)


def _get_runtime_gripper_target_qpos_rows(agent, *, num_envs: int):
    controller = getattr(agent, "controller", None)
    controllers = getattr(controller, "controllers", {}) if controller is not None else {}
    gripper_controller = controllers.get("gripper")
    if gripper_controller is None:
        return None
    target_qpos = getattr(gripper_controller, "_target_qpos", None)
    if target_qpos is None:
        return None
    return _normalize_rows(
        target_qpos,
        num_envs=num_envs,
        width=10_000,
        label="runtime_gripper_target_qpos",
    )


def _extract_lift_stage_overrides(base_env):
    planner_cfg = getattr(base_env, "_debug_planner_config", None)
    if planner_cfg is None:
        return {}
    if not isinstance(planner_cfg, dict):
        raise TypeError(
            "Expected base_env._debug_planner_config to be a dict, "
            f"got {type(planner_cfg).__name__}."
        )
    overrides = {}
    lift_z_step = planner_cfg.get("planner_proxy_lift_z_step_m")
    if lift_z_step is not None:
        overrides["max_z_step_override"] = float(lift_z_step)
    lift_pos_tol = planner_cfg.get("planner_proxy_lift_pos_tol_m")
    if lift_pos_tol is not None:
        overrides["pos_tol_override"] = float(lift_pos_tol)
    return overrides


def run_planner_lift(
    env,
    *,
    lift_delta_z=0.05,
    method="rrtconnect",
    execute=True,
    repeat=1,
    backend="local_ik",
    extract_planner_base_pose,
    resolve_planner_debug_solver_class,
    is_proxy_ee_delta_backend,
    is_ee_delta_control_mode,
    run_proxy_ee_delta_pose_stage,
    pose_to_numpy,
    get_debug_planner_ee_pose,
    get_debug_planner_ee_pose_rows=None,
    get_debug_planner_retention_pose=None,
    get_debug_planner_retention_pose_rows=None,
    get_debug_target_object,
    get_debug_actor_position_xyz,
    to_scalar_bool,
    get_robot_hand_qpos_debug,
    get_robot_hand_qpos_rows_debug=None,
    get_robot_qpos,
    configure_debug_planner_solver_runtime,
    get_planner_recording_kwargs,
    build_debug_lift_pose_from_policy,
    is_rc5_debug_planner_agent,
    evaluate_rc5_pick_lift_success,
    get_planner_grasp_state,
    save_planner_grasp_state,
    refresh_render_state,
    execute_planner_pose_with_backend=None,
    execute_real_planner_pose_with_backend=None,
):
    planner_pose_executor = (
        execute_real_planner_pose_with_backend
        if execute_real_planner_pose_with_backend is not None
        else execute_planner_pose_with_backend
    )
    num_envs = int(getattr(env.unwrapped, "num_envs", 1) or 1)
    if get_debug_planner_retention_pose is None:
        if num_envs > 1:
            print(
                f"{_Y}[WARNING] [PlannerDebug] Batched lift did not receive an explicit retention pose helper; "
                "falling back to the active EE pose for retention diagnostics."
                f"{_R}"
            )
        get_debug_planner_retention_pose = get_debug_planner_ee_pose
    agent_uid = getattr(env.unwrapped.agent, "uid", "unknown")
    control_mode = getattr(env.unwrapped, "control_mode", None)
    base_pose = extract_planner_base_pose(env)
    PlannerClass = resolve_planner_debug_solver_class(agent_uid)
    print(
        f"[PlannerDebug] Starting lift for agent_uid='{agent_uid}' "
        f"with solver_class={PlannerClass.__name__}, method={method}, backend={backend}, "
        f"lift_delta_z={float(lift_delta_z):.4f}, repeat={int(repeat)}"
    )
    if is_proxy_ee_delta_backend(backend):
        if not is_ee_delta_control_mode(control_mode):
            print(
                f"[PlannerDebug] Proxy lift requires an EE-delta control_mode, got '{control_mode}'."
            )
            return False
        try:
            lift_stage_overrides = _extract_lift_stage_overrides(env.unwrapped)
            target_object = get_debug_target_object(env.unwrapped)
            object_p_before_lift = None if target_object is None else get_debug_actor_position_xyz(target_object)
            object_p_before_lift_rows = None
            if target_object is not None and num_envs > 1:
                object_p_before_lift_rows = _extract_pose_position_rows(
                    getattr(target_object, "pose", None),
                    num_envs=num_envs,
                    label="target_object.pose before lift",
                )
            repeat = max(int(repeat), 1)
            target_pose = None
            lift_invariant_trace_steps = int(
                (
                    getattr(env.unwrapped, "_debug_planner_config", {}) or {}
                ).get("planner_proxy_lift_invariant_trace_steps", 5)
                or 0
            )
            def _log_lift_invariants_after_step(env_unwrapped, *, step_idx: int, max_stage_steps: int, stage_label: str):
                if int(step_idx) > int(lift_invariant_trace_steps):
                    return
                if target_object is None:
                    return
                tcp_p, _tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env_unwrapped))
                object_p = get_debug_actor_position_xyz(target_object)
                object_minus_tcp = np.asarray(object_p - tcp_p, dtype=np.float32).reshape(-1)[:3]
                object_tcp_dist = float(np.linalg.norm(object_minus_tcp))
                if num_envs > 1 and callable(get_debug_planner_ee_pose_rows):
                    tcp_p_rows = _extract_pose_position_rows(
                        get_debug_planner_ee_pose_rows(env_unwrapped),
                        num_envs=num_envs,
                        label=f"{stage_label}.tcp_pose_rows",
                    )
                    object_p_rows = _extract_pose_position_rows(
                        getattr(target_object, "pose", None),
                        num_envs=num_envs,
                        label=f"{stage_label}.target_object.pose",
                    )
                    object_minus_tcp_rows = object_p_rows - tcp_p_rows
                    object_tcp_dist_rows = np.linalg.norm(object_minus_tcp_rows, axis=1)
                    if is_debug_enabled():
                        logger.debug(
                            "[PlannerDebug] Lift invariant trace step {}/{}: object_minus_tcp={} object_tcp_dist={}",
                            step_idx,
                            max_stage_steps,
                            np.array2string(object_minus_tcp_rows, precision=4, suppress_small=True),
                            np.array2string(object_tcp_dist_rows, precision=4, suppress_small=True),
                        )
                    return
                if is_debug_enabled():
                    logger.debug(
                        "[PlannerDebug] Lift invariant trace step {}/{}: object_minus_tcp={} object_tcp_dist={:.4f}",
                        step_idx,
                        max_stage_steps,
                        np.array2string(object_minus_tcp, precision=4, suppress_small=True),
                        object_tcp_dist,
                    )
            for step_idx in range(repeat):
                target_pose, reference_pose, task_pose, runtime_pose = build_debug_lift_pose_from_policy(
                    env.unwrapped,
                    lift_delta_z=float(lift_delta_z),
                )
                current_tcp_p, _current_tcp_q = pose_to_numpy(runtime_pose.pose_world)
                if is_debug_enabled():
                    logger.debug(
                        "[PlannerDebug] Lift step {}/{} target pose p={} q={}",
                        step_idx + 1,
                        repeat,
                        np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True),
                        np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True),
                    )
                    logger.debug(
                        "[PlannerDebug] Lift step {}/{} target delta_z={:.4f} target_tcp_from_current={:.4f} m",
                        step_idx + 1,
                        repeat,
                        float(lift_delta_z),
                        float(np.linalg.norm(np.asarray(target_pose.p) - np.asarray(current_tcp_p))),
                    )
                    logger.debug(
                        "[PlannerDebug] Lift semantic policy: reference_semantics={} task_stage={}",
                        reference_pose.semantics,
                        task_pose.source_stage,
                    )
                if not run_proxy_ee_delta_pose_stage(
                    env,
                    target_pose,
                    stage_label=f"Lift:{step_idx + 1}/{repeat}",
                    position_mask=(False, False, True),
                    align_orientation=False,
                    step_observer=_log_lift_invariants_after_step,
                    **lift_stage_overrides,
                ):
                    print(f"[PlannerDebug] Proxy lift FAILED at step {step_idx + 1}/{repeat}")
                    return False

            final_tcp_p, _ = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            if num_envs > 1 and callable(get_debug_planner_ee_pose_rows):
                final_tcp_p_rows = _extract_pose_position_rows(
                    get_debug_planner_ee_pose_rows(env.unwrapped),
                    num_envs=num_envs,
                    label="ee_pose after lift",
                )
                target_p_rows = _normalize_rows(
                    getattr(target_pose, "p", target_pose),
                    num_envs=num_envs,
                    width=3,
                    label="lift_target_pose",
                )
                final_err_rows = np.linalg.norm(final_tcp_p_rows - target_p_rows, axis=1)
                final_err = float(np.max(final_err_rows))
            else:
                final_err_rows = None
                final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = 0.1
            print(
                f"[PlannerDebug] Lift final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err_rows is not None:
                print(
                    f"[PlannerDebug] Batched lift final pos_err_rows="
                    f"{np.array2string(final_err_rows, precision=4, suppress_small=True)}"
                )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] Lift final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m; treating stage as FAILED."
                )
                refresh_render_state(env)
                return False
            if target_object is not None and object_p_before_lift is not None:
                object_p_after_lift = get_debug_actor_position_xyz(target_object)
                lift_dz = float(object_p_after_lift[2] - object_p_before_lift[2])
                grasp_flag_after_lift = to_scalar_bool(env.unwrapped.agent.is_grasping(target_object))
                lift_success_threshold = max(0.02, 0.25 * float(lift_delta_z))
                grasp_state_before_lift = get_planner_grasp_state(env.unwrapped)
                grasp_flag_after_close = None
                if grasp_state_before_lift is not None:
                    grasp_flag_after_close = grasp_state_before_lift.grasp_flag
                retention_pose_p, _ = pose_to_numpy(get_debug_planner_retention_pose(env.unwrapped))
                object_minus_tcp_after_lift = (
                    np.asarray(object_p_after_lift, dtype=np.float32) - np.asarray(retention_pose_p, dtype=np.float32)
                )
                object_tcp_dist_after_lift = float(np.linalg.norm(object_minus_tcp_after_lift))
                success_decision = evaluate_rc5_pick_lift_success(
                    grasp_flag_after_lift=grasp_flag_after_lift,
                    lift_dz=lift_dz,
                    lift_success_threshold=lift_success_threshold,
                    grasp_flag_after_close=grasp_flag_after_close,
                    object_tcp_dist_after_lift=object_tcp_dist_after_lift,
                )
                task_success = success_decision.success
                print(
                    f"[PlannerDebug] after lift: is_grasping={grasp_flag_after_lift} "
                    f"is_grasping_after_close={grasp_flag_after_close} "
                    f"object_z={object_p_after_lift[2]:.4f} lift_dz={lift_dz:+.4f} "
                    f"retention_pose_p={np.array2string(retention_pose_p, precision=4, suppress_small=True)} "
                    f"object_minus_tcp={np.array2string(object_minus_tcp_after_lift, precision=4, suppress_small=True)} "
                    f"object_tcp_dist={object_tcp_dist_after_lift:.4f}"
                )
                print(
                    f"[PlannerDebug] task_success={task_success} "
                    f"(reason={success_decision.reason}, lift_dz_threshold={lift_success_threshold:.4f})"
                )
                if num_envs > 1 and object_p_before_lift_rows is not None:
                    object_p_after_lift_rows = _extract_pose_position_rows(
                        getattr(target_object, "pose", None),
                        num_envs=num_envs,
                        label="target_object.pose after lift",
                    )
                    grasp_after_lift_rows_payload = env.unwrapped.agent.is_grasping(target_object)
                    if hasattr(env.unwrapped, "evaluate") and callable(getattr(env.unwrapped, "evaluate", None)):
                        try:
                            evaluation_payload = env.unwrapped.evaluate()
                        except Exception:
                            evaluation_payload = None
                        if isinstance(evaluation_payload, dict) and "is_src_obj_grasped" in evaluation_payload:
                            grasp_after_lift_rows_payload = evaluation_payload.get("is_src_obj_grasped")
                    grasp_flag_after_lift_rows = _normalize_bool_rows(
                        grasp_after_lift_rows_payload,
                        num_envs=num_envs,
                        label="agent.is_grasping(target_object) after lift",
                    )
                    grasp_flag_after_close_rows = _normalize_bool_rows(
                        (
                            getattr(grasp_state_before_lift, "grasp_flag_rows", None)
                            if grasp_state_before_lift is not None
                            else grasp_flag_after_close
                        ),
                        num_envs=num_envs,
                        label=(
                            "grasp_flag_after_close"
                            if grasp_state_before_lift is None or getattr(grasp_state_before_lift, "grasp_flag_rows", None) is None
                            else "grasp_flag_rows_after_close"
                        ),
                        allow_none=True,
                    )
                    if callable(get_debug_planner_retention_pose_rows):
                        retention_pose_rows = _extract_pose_position_rows(
                            get_debug_planner_retention_pose_rows(env.unwrapped),
                            num_envs=num_envs,
                            label="retention_pose_rows",
                        )
                    else:
                        retention_pose_rows = _extract_pose_position_rows(
                            get_debug_planner_retention_pose(env.unwrapped),
                            num_envs=num_envs,
                            label="retention_pose",
                        )
                    lift_dz_rows = object_p_after_lift_rows[:, 2] - object_p_before_lift_rows[:, 2]
                    object_minus_tcp_rows = object_p_after_lift_rows - retention_pose_rows
                    object_tcp_dist_rows = np.linalg.norm(object_minus_tcp_rows, axis=1)
                    hand_qpos_rows = None
                    if callable(get_robot_hand_qpos_rows_debug):
                        hand_qpos_rows = _normalize_rows(
                            get_robot_hand_qpos_rows_debug(env.unwrapped),
                            num_envs=num_envs,
                            width=10_000,
                            label="robot_hand_qpos_rows",
                        )
                    hand_target_qpos_rows = _get_runtime_gripper_target_qpos_rows(
                        env.unwrapped.agent,
                        num_envs=num_envs,
                    )
                    hand_target_err_rows = None
                    if hand_target_qpos_rows is None:
                        print(
                            f"{_Y}[WARNING] [PlannerDebug] Batched lift could not read runtime gripper target qpos rows; "
                            "actual-vs-target hand diagnostics are unavailable."
                            f"{_R}"
                        )
                    elif hand_qpos_rows is not None and hand_qpos_rows.shape == hand_target_qpos_rows.shape:
                        hand_target_err_rows = np.linalg.norm(hand_qpos_rows - hand_target_qpos_rows, axis=1)
                    elif hand_qpos_rows is not None:
                        print(
                            f"{_Y}[WARNING] [PlannerDebug] Batched lift hand target shape={hand_target_qpos_rows.shape} "
                            f"does not match hand qpos shape={hand_qpos_rows.shape}; "
                            "actual-vs-target hand diagnostics are unavailable."
                            f"{_R}"
                        )
                    per_env_decisions = [
                        evaluate_rc5_pick_lift_success(
                            grasp_flag_after_lift=grasp_flag_after_lift_rows[env_index],
                            lift_dz=float(lift_dz_rows[env_index]),
                            lift_success_threshold=lift_success_threshold,
                            grasp_flag_after_close=grasp_flag_after_close_rows[env_index],
                            object_tcp_dist_after_lift=float(object_tcp_dist_rows[env_index]),
                        )
                        for env_index in range(num_envs)
                    ]
                    print(
                        "[PlannerDebug] Batched lift diagnostics: "
                        f"lift_dz={np.array2string(lift_dz_rows, precision=4, suppress_small=True)} "
                        f"grasp_after_close={grasp_flag_after_close_rows} "
                        f"grasp_after_lift={grasp_flag_after_lift_rows} "
                        f"object_z={np.array2string(object_p_after_lift_rows[:, 2], precision=4, suppress_small=True)} "
                        f"retention_z={np.array2string(retention_pose_rows[:, 2], precision=4, suppress_small=True)} "
                        f"object_minus_tcp={np.array2string(object_minus_tcp_rows, precision=4, suppress_small=True)} "
                        f"object_tcp_dist={np.array2string(object_tcp_dist_rows, precision=4, suppress_small=True)} "
                        f"hand_min={np.array2string(np.min(hand_qpos_rows, axis=1), precision=4, suppress_small=True) if hand_qpos_rows is not None and hand_qpos_rows.size > 0 else '[n/a]'} "
                        f"hand_max={np.array2string(np.max(hand_qpos_rows, axis=1), precision=4, suppress_small=True) if hand_qpos_rows is not None and hand_qpos_rows.size > 0 else '[n/a]'} "
                        f"hand_target_min={np.array2string(np.min(hand_target_qpos_rows, axis=1), precision=4, suppress_small=True) if hand_target_qpos_rows is not None and hand_target_qpos_rows.size > 0 else '[n/a]'} "
                        f"hand_target_max={np.array2string(np.max(hand_target_qpos_rows, axis=1), precision=4, suppress_small=True) if hand_target_qpos_rows is not None and hand_target_qpos_rows.size > 0 else '[n/a]'} "
                        f"hand_target_err={np.array2string(hand_target_err_rows, precision=4, suppress_small=True) if hand_target_err_rows is not None else '[n/a]'} "
                        f"success={[item.success for item in per_env_decisions]} "
                        f"reason={[item.reason for item in per_env_decisions]}"
                    )
                    if any(not item.success for item in per_env_decisions):
                        print(
                            f"{_Y}[WARNING] [PlannerDebug] Batched lift semantics diverged across envs; "
                            "stage-level lift control is still using scalar success gating, so final batch "
                            "semantic truth must be read from env.evaluate()/per-env runtime feedback."
                            f"{_R}"
                        )
                save_planner_grasp_state(
                    env.unwrapped,
                    realized_hand_qpos=get_robot_hand_qpos_debug(env.unwrapped),
                    grasp_flag=grasp_flag_after_lift,
                    grasp_flag_rows=(
                        np.asarray(grasp_flag_after_lift_rows, dtype=bool)
                        if num_envs > 1 and "grasp_flag_after_lift_rows" in locals()
                        else None
                    ),
                    object_id=str(getattr(env.unwrapped, "manip_object_id", None)),
                    source_stage="lift",
                )
                if num_envs > 1:
                    print(
                        f"{_Y}[WARNING] [PlannerDebug] save_planner_grasp_state() stores a scalar lift grasp state "
                        "in batched mode; use per-env runtime feedback for truthful batch diagnostics."
                        f"{_R}"
                    )
                if not task_success:
                    print(
                        "[WARNING] [PlannerDebug] Lift retention check failed even though arm motion succeeded; "
                        "treating lift stage as FAILED."
                    )
                    refresh_render_state(env)
                    return False
            refresh_render_state(env)
            print("[PlannerDebug] Lift EXECUTE OK")
            return True
        except Exception as exc:
            print(f"[PlannerDebug] Proxy lift FAILED: {type(exc).__name__}: {exc}")
            print("[PlannerDebug] Traceback:")
            print(traceback.format_exc().rstrip())
            return False
    if is_ee_delta_control_mode(control_mode):
        print(
            f"[PlannerDebug] Skipping lift: current control_mode='{control_mode}' is EE-delta/RL. "
            "The lift probe is wired for the joint-space planner path "
            "(*_rl robot_uids + 'pd_joint_pos')."
        )
        return False
    if planner_pose_executor is None:
        raise ValueError("Lift joint-space path requires a real planner pose executor helper.")
    try:
        current_tcp_p, current_tcp_q = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
        current_qpos = get_robot_qpos(env)
        target_object = get_debug_target_object(env.unwrapped)
        object_p_before_lift = get_debug_actor_position_xyz(target_object) if target_object is not None else None
        print(
            f"[PlannerDebug] Lift branch diagnostics: backend={backend}, execute={execute}, repeat={repeat}, "
            f"current_tcp_p={np.array2string(np.asarray(current_tcp_p), precision=4, suppress_small=True)} "
            f"current_tcp_q={np.array2string(np.asarray(current_tcp_q), precision=4, suppress_small=True)}"
        )
        print(
            f"[PlannerDebug] Lift branch diagnostics: current_qpos={np.array2string(np.asarray(current_qpos), precision=4, suppress_small=True)}"
        )
        solver = configure_debug_planner_solver_runtime(
            PlannerClass(
                env,
                debug=True,
                vis=True,
                base_pose=base_pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                **get_planner_recording_kwargs(env),
            )
        )
        grasp_state = get_planner_grasp_state(env.unwrapped)
        latched_hand_qpos = None if grasp_state is None else grasp_state.target_hand_qpos
        if latched_hand_qpos is not None:
            latched_hand_qpos = np.asarray(latched_hand_qpos, dtype=np.float32)
            print(
                f"[PlannerDebug] Lift using latched hand target range="
                f"[{float(np.min(latched_hand_qpos)):.4f}, {float(np.max(latched_hand_qpos)):.4f}]"
            )
        repeat = max(int(repeat), 1)
        result = None
        target_pose = None
        for step_idx in range(repeat):
            target_pose, reference_pose, task_pose, runtime_pose = build_debug_lift_pose_from_policy(
                env.unwrapped,
                lift_delta_z=float(lift_delta_z),
            )
            current_tcp_p, _current_tcp_q = pose_to_numpy(runtime_pose.pose_world)
            if is_debug_enabled():
                logger.debug(
                    "[PlannerDebug] Lift step {}/{} target pose p={} q={}",
                    step_idx + 1,
                    repeat,
                    np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True),
                    np.array2string(np.asarray(target_pose.q), precision=4, suppress_small=True),
                )
                logger.debug(
                    "[PlannerDebug] Lift step {}/{} target delta_z={:.4f} target_tcp_from_current={:.4f} m",
                    step_idx + 1,
                    repeat,
                    float(lift_delta_z),
                    float(np.linalg.norm(np.asarray(target_pose.p) - np.asarray(current_tcp_p))),
                )
                logger.debug(
                    "[PlannerDebug] Lift semantic policy: reference_semantics={} task_stage={}",
                    reference_pose.semantics,
                    task_pose.source_stage,
                )

            result = planner_pose_executor(
                solver,
                target_pose,
                execute=execute,
                backend=backend,
                method=method,
                planner_class_name=PlannerClass.__name__,
                stage_label="Lift",
            )

            if result == -1:
                print(f"[PlannerDebug] Lift FAILED via method={method} at step {step_idx + 1}/{repeat}")
                solver.close()
                return False

            if not execute:
                break

        if execute:
            final_tcp_p, _ = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
            final_err = float(np.linalg.norm(final_tcp_p - np.asarray(target_pose.p, dtype=np.float32)))
            max_accept_err = 0.1
            print(
                f"[PlannerDebug] Lift final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
                f"target_p={np.array2string(np.asarray(target_pose.p), precision=4, suppress_small=True)} "
                f"pos_err={final_err:.4f} m"
            )
            if final_err > max_accept_err:
                print(
                    f"[WARNING] [PlannerDebug] Lift final pos_err={final_err:.4f} m exceeds "
                    f"acceptance threshold {max_accept_err:.4f} m; treating stage as FAILED."
                )
                refresh_render_state(env)
                solver.close()
                return False
            if target_object is not None and object_p_before_lift is not None:
                object_p_after_lift = get_debug_actor_position_xyz(target_object)
                lift_dz = float(object_p_after_lift[2] - object_p_before_lift[2])
                grasp_flag_after_lift = to_scalar_bool(env.unwrapped.agent.is_grasping(target_object))
                lift_success_threshold = max(0.02, 0.25 * float(lift_delta_z))
                grasp_state_before_lift = get_planner_grasp_state(env.unwrapped)
                grasp_flag_after_close = None
                if grasp_state_before_lift is not None:
                    grasp_flag_after_close = grasp_state_before_lift.grasp_flag
                retention_pose_p, _ = pose_to_numpy(get_debug_planner_retention_pose(env.unwrapped))
                object_minus_tcp_after_lift = (
                    np.asarray(object_p_after_lift, dtype=np.float32) - np.asarray(retention_pose_p, dtype=np.float32)
                )
                object_tcp_dist_after_lift = float(np.linalg.norm(object_minus_tcp_after_lift))
                success_decision = evaluate_rc5_pick_lift_success(
                    grasp_flag_after_lift=grasp_flag_after_lift,
                    lift_dz=lift_dz,
                    lift_success_threshold=lift_success_threshold,
                    grasp_flag_after_close=grasp_flag_after_close,
                    object_tcp_dist_after_lift=object_tcp_dist_after_lift,
                )
                task_success = success_decision.success
                print(
                    f"[PlannerDebug] after lift: is_grasping={grasp_flag_after_lift} "
                    f"is_grasping_after_close={grasp_flag_after_close} "
                    f"object_z={object_p_after_lift[2]:.4f} lift_dz={lift_dz:+.4f} "
                    f"retention_pose_p={np.array2string(retention_pose_p, precision=4, suppress_small=True)} "
                    f"object_minus_tcp={np.array2string(object_minus_tcp_after_lift, precision=4, suppress_small=True)} "
                    f"object_tcp_dist={object_tcp_dist_after_lift:.4f}"
                )
                print(
                    f"[PlannerDebug] task_success={task_success} "
                    f"(reason={success_decision.reason}, lift_dz_threshold={lift_success_threshold:.4f})"
                )
                save_planner_grasp_state(
                    env.unwrapped,
                    realized_hand_qpos=get_robot_hand_qpos_debug(env.unwrapped),
                    grasp_flag=grasp_flag_after_lift,
                    object_id=str(getattr(env.unwrapped, "manip_object_id", None)),
                    source_stage="lift",
                )
                if not task_success:
                    print(
                        "[WARNING] [PlannerDebug] Lift retention check failed even though arm motion succeeded; "
                        "treating lift stage as FAILED."
                    )
                    refresh_render_state(env)
                    solver.close()
                    return False
            refresh_render_state(env)
            print("[PlannerDebug] Lift EXECUTE OK")
            solver.close()
            return True

        status = result.get("status", "unknown")
        n_steps = int(result["position"].shape[0]) if "position" in result else -1
        print(f"[PlannerDebug] Lift OK: status={status}, n_steps={n_steps}")
        solver.close()
        return True
    except Exception as exc:
        print(f"[PlannerDebug] Lift FAILED: {type(exc).__name__}: {exc}")
        print("[PlannerDebug] Traceback:")
        print(traceback.format_exc().rstrip())
        return False


def run_proxy_lift(*args, **kwargs):
    return run_planner_lift(*args, **kwargs)
