from __future__ import annotations

import traceback

import numpy as np


def run_planner_close_gripper(
    env,
    *,
    close_steps=20,
    backend="planner",
    extract_planner_base_pose,
    resolve_planner_debug_solver_class,
    is_proxy_ee_delta_backend,
    is_ee_delta_control_mode,
    run_proxy_close_gripper,
    configure_debug_planner_solver_runtime,
    get_planner_recording_kwargs,
    get_debug_target_object,
    get_debug_actor_position_xyz,
    log_debug_pre_close_snapshot,
    pose_to_numpy,
    get_debug_planner_ee_pose,
    log_debug_post_close_retention,
    get_robot_hand_qpos_debug,
    get_robot_hand_range_debug,
    save_planner_grasp_state,
    refresh_render_state,
):
    agent_uid = getattr(env.unwrapped.agent, "uid", "unknown")
    control_mode = getattr(env.unwrapped, "control_mode", None)
    base_pose = extract_planner_base_pose(env)
    PlannerClass = resolve_planner_debug_solver_class(agent_uid)
    print(
        f"[PlannerDebug] Starting in-place grasp close for agent_uid='{agent_uid}' "
        f"with solver_class={PlannerClass.__name__}, close_steps={close_steps}, backend={backend}"
    )
    if is_proxy_ee_delta_backend(backend):
        if not is_ee_delta_control_mode(control_mode):
            print(
                f"[PlannerDebug] Proxy close requires an EE-delta control_mode, got '{control_mode}'."
            )
            return False
        try:
            return run_proxy_close_gripper(env, close_steps=close_steps)
        except Exception as exc:
            print(f"[PlannerDebug] Proxy close FAILED: {type(exc).__name__}: {exc}")
            print("[PlannerDebug] Traceback:")
            print(traceback.format_exc().rstrip())
            return False
    if is_ee_delta_control_mode(control_mode):
        print(
            f"[PlannerDebug] Skipping close: current control_mode='{control_mode}' is EE-delta/RL. "
            "The in-place close probe is wired for the joint-space planner path "
            "(*_rl robot_uids + 'pd_joint_pos')."
        )
        return False
    try:
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
        target_object = get_debug_target_object(env.unwrapped)
        object_p_before_close = None
        if target_object is not None:
            object_p_before_close = get_debug_actor_position_xyz(target_object)
            log_debug_pre_close_snapshot(env.unwrapped, target_object, int(close_steps))
        solver.close_gripper(t=int(close_steps))
        final_tcp_p, _ = pose_to_numpy(get_debug_planner_ee_pose(env.unwrapped))
        grasp_flag = None
        if target_object is not None:
            grasp_flag, _, _, hand_min, hand_max = log_debug_post_close_retention(
                env.unwrapped,
                target_object,
                object_p_before_close,
                int(close_steps),
            )
            final_hand_qpos = get_robot_hand_qpos_debug(env.unwrapped)
        else:
            final_hand_qpos = get_robot_hand_qpos_debug(env.unwrapped)
            hand_min, hand_max = get_robot_hand_range_debug(env.unwrapped)
        save_planner_grasp_state(
            env.unwrapped,
            realized_hand_qpos=final_hand_qpos,
            grasp_flag=grasp_flag,
            object_id=str(getattr(env.unwrapped, "manip_object_id", None)) if target_object is not None else None,
            source_stage="close",
        )
        print(
            f"[PlannerDebug] Close final tcp p={np.array2string(final_tcp_p, precision=4, suppress_small=True)} "
            f"hand_range=[{hand_min:.4f}, {hand_max:.4f}]"
        )
        refresh_render_state(env)
        print("[PlannerDebug] Close EXECUTE OK")
        solver.close()
        return True
    except Exception as exc:
        print(f"[PlannerDebug] Close FAILED: {type(exc).__name__}: {exc}")
        print("[PlannerDebug] Traceback:")
        print(traceback.format_exc().rstrip())
        return False
