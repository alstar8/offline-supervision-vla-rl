from __future__ import annotations

from contextlib import nullcontext

from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_control import (
    get_default_debug_close_steps,
    get_default_planner_stage_method,
    get_hybrid_jointspace_control_mode,
    is_proxy_ee_delta_backend,
    is_proxy_then_planner_backend,
    resolve_macro_backend,
    resolve_proxy_only_macro_backend,
    temporary_agent_control_mode,
)

_Y = "\033[33m"
_R = "\033[0m"


def _resolve_stage_callback(primary, compatibility, *, callback_label: str):
    callback = primary or compatibility
    if callback is None:
        raise ValueError(
            f"Proxy planner macro contract requires callback '{callback_label}' to be provided."
        )
    return callback


def _run_pick_macro_common(
    env,
    config_overrides,
    *,
    default_backend: str,
    close_backend_when_not_full: str | None,
    log_prefix: str,
    hybrid_reason: str,
    use_full_log_prefix: str,
    pregrasp_abort_message: str,
    full_approach_abort_message: str,
    descend_abort_message: str,
    close_abort_message: str,
    lift_abort_message: str,
    success_message: str,
    run_proxy_pregrasp_probe,
    run_proxy_full_approach_to_descend,
    run_proxy_descend,
    run_proxy_close_gripper,
    run_proxy_lift,
    set_unified_macro_feedback,
):
    agent_uid = getattr(env.unwrapped.agent, "uid", "unknown")
    pregrasp_method = str(config_overrides.get("planner_pregrasp_method", "auto") or "auto").strip().lower()
    descend_method = get_default_planner_stage_method(agent_uid)
    lift_method = get_default_planner_stage_method(agent_uid)
    close_steps = get_default_debug_close_steps(agent_uid)
    approach_waypoints = max(int(config_overrides.get("planner_approach_waypoints", 0) or 0), 0)
    macro_backend = resolve_macro_backend(config_overrides, default_backend=default_backend)
    use_full_approach = is_proxy_ee_delta_backend(macro_backend) or approach_waypoints > 0
    hybrid_proxy_handoff = is_proxy_then_planner_backend(macro_backend)

    print(log_prefix)
    if hybrid_proxy_handoff:
        print(f"{_Y}{hybrid_reason}{_R}")
        if not run_proxy_pregrasp_probe(
            env,
            method=pregrasp_method,
            extra_clearance=0.10,
            execute=True,
            backend=macro_backend,
        ):
            set_unified_macro_feedback(semantic_task_success=False, failed_stage="pregrasp")
            print(pregrasp_abort_message)
            return False
        descend_backend = "planner"
        close_backend = "planner"
        lift_backend = "planner"
    elif use_full_approach:
        print(f"{_Y}{use_full_log_prefix}{_R}")
        if not run_proxy_full_approach_to_descend(
            env,
            method=descend_method,
            extra_clearance=0.03,
            execute=True,
            backend=macro_backend,
        ):
            set_unified_macro_feedback(semantic_task_success=False, failed_stage="full_approach")
            print(full_approach_abort_message)
            return False
        descend_backend = macro_backend
        close_backend = close_backend_when_not_full or macro_backend
        lift_backend = macro_backend
    else:
        if not run_proxy_pregrasp_probe(
            env,
            method=pregrasp_method,
            extra_clearance=0.10,
            execute=True,
        ):
            set_unified_macro_feedback(semantic_task_success=False, failed_stage="pregrasp")
            print(pregrasp_abort_message)
            return False
        descend_backend = macro_backend
        close_backend = close_backend_when_not_full or macro_backend
        lift_backend = macro_backend

    handoff_control_mode = None
    if hybrid_proxy_handoff:
        handoff_control_mode = get_hybrid_jointspace_control_mode(env.unwrapped)

    with (
        temporary_agent_control_mode(
            env,
            handoff_control_mode,
            reason=(
                "hybrid handoff to real planner descend/close/lift"
                if close_backend_when_not_full is None
                else "hybrid handoff to real planner full descend/close/lift"
            ),
        )
        if handoff_control_mode is not None
        else nullcontext()
    ):
        if not use_full_approach or hybrid_proxy_handoff:
            if not run_proxy_descend(
                env,
                method=descend_method,
                extra_clearance=0.03,
                execute=True,
                backend=descend_backend,
            ):
                set_unified_macro_feedback(semantic_task_success=False, failed_stage="descend")
                print(descend_abort_message)
                return False
        if not run_proxy_close_gripper(env, close_steps=close_steps, backend=close_backend):
            set_unified_macro_feedback(semantic_task_success=False, failed_stage="close")
            print(close_abort_message)
            return False
        if not run_proxy_lift(
            env,
            lift_delta_z=2.0 * float(config_overrides.get("planner_lift_delta_z", 0.05)),
            method=lift_method,
            execute=True,
            repeat=1,
            backend=lift_backend,
        ):
            set_unified_macro_feedback(semantic_task_success=False, failed_stage="lift")
            print(lift_abort_message)
            return False

    set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)
    print(success_message)
    return True


def run_planner_pick_macro(
    env,
    config_overrides,
    *,
    run_planner_object_pregrasp_probe,
    run_planner_full_approach_to_descend,
    run_planner_object_descend,
    run_planner_close_gripper,
    run_planner_lift,
    set_unified_macro_feedback,
):
    return _run_pick_macro_common(
        env,
        config_overrides,
        default_backend="local_ik",
        close_backend_when_not_full=None,
        log_prefix="[PlannerDebug] Starting pick macro: pregrasp -> descend -> close -> lift",
        hybrid_reason=(
            "[PlannerDebug] Hybrid macro mode enabled: proxy pregrasp handoff -> "
            "real planner / mplib planner descend/close/lift."
        ),
        use_full_log_prefix=(
            f"[PlannerDebug] Full-approach macro mode enabled: backend='"
            f"{resolve_macro_backend(config_overrides, default_backend='local_ik')}', "
            f"planner_approach_waypoints={max(int(config_overrides.get('planner_approach_waypoints', 0) or 0), 0)}. "
            "Split pregrasp/descend stages are bypassed for this macro."
        ),
        pregrasp_abort_message="[PlannerDebug] Pick macro aborted at pregrasp stage.",
        full_approach_abort_message="[PlannerDebug] Pick macro aborted at full-approach stage.",
        descend_abort_message="[PlannerDebug] Pick macro aborted at descend stage.",
        close_abort_message="[PlannerDebug] Pick macro aborted at close stage.",
        lift_abort_message="[PlannerDebug] Pick macro aborted at lift stage.",
        success_message="[PlannerDebug] Pick macro EXECUTE OK",
        run_proxy_pregrasp_probe=run_planner_object_pregrasp_probe,
        run_proxy_full_approach_to_descend=run_planner_full_approach_to_descend,
        run_proxy_descend=run_planner_object_descend,
        run_proxy_close_gripper=run_planner_close_gripper,
        run_proxy_lift=run_planner_lift,
        set_unified_macro_feedback=set_unified_macro_feedback,
    )


def run_proxy_pick_macro(
    env,
    config_overrides,
    *,
    run_proxy_pregrasp_probe=None,
    run_proxy_full_approach_to_descend=None,
    run_proxy_descend=None,
    run_proxy_close_gripper=None,
    run_proxy_lift=None,
    run_planner_object_pregrasp_probe=None,
    run_planner_full_approach_to_descend=None,
    run_planner_object_descend=None,
    run_planner_close_gripper=None,
    run_planner_lift=None,
    set_unified_macro_feedback,
):
    macro_backend = resolve_proxy_only_macro_backend(
        config_overrides,
        default_backend="proxy_ee_delta",
    )
    return _run_pick_macro_common(
        env,
        config_overrides,
        default_backend=macro_backend,
        close_backend_when_not_full=None,
        log_prefix="[PlannerDebug] Starting proxy pick macro: pregrasp -> descend -> close -> lift",
        hybrid_reason=(
            "[PlannerDebug] Proxy-only pick macro does not support hybrid handoff "
            "to real planner / mplib planner."
        ),
        use_full_log_prefix=(
            f"[PlannerDebug] Proxy-only macro mode enabled: backend='{macro_backend}', "
            f"planner_approach_waypoints={max(int(config_overrides.get('planner_approach_waypoints', 0) or 0), 0)}. "
            "Split pregrasp/descend stages are bypassed for this macro."
        ),
        pregrasp_abort_message="[PlannerDebug] Proxy pick macro aborted at pregrasp stage.",
        full_approach_abort_message="[PlannerDebug] Proxy pick macro aborted at full-approach stage.",
        descend_abort_message="[PlannerDebug] Proxy pick macro aborted at descend stage.",
        close_abort_message="[PlannerDebug] Proxy pick macro aborted at close stage.",
        lift_abort_message="[PlannerDebug] Proxy pick macro aborted at lift stage.",
        success_message="[PlannerDebug] Proxy pick macro EXECUTE OK",
        run_proxy_pregrasp_probe=_resolve_stage_callback(
            run_proxy_pregrasp_probe,
            run_planner_object_pregrasp_probe,
            callback_label="run_proxy_pregrasp_probe",
        ),
        run_proxy_full_approach_to_descend=_resolve_stage_callback(
            run_proxy_full_approach_to_descend,
            run_planner_full_approach_to_descend,
            callback_label="run_proxy_full_approach_to_descend",
        ),
        run_proxy_descend=_resolve_stage_callback(
            run_proxy_descend,
            run_planner_object_descend,
            callback_label="run_proxy_descend",
        ),
        run_proxy_close_gripper=_resolve_stage_callback(
            run_proxy_close_gripper,
            run_planner_close_gripper,
            callback_label="run_proxy_close_gripper",
        ),
        run_proxy_lift=_resolve_stage_callback(
            run_proxy_lift,
            run_planner_lift,
            callback_label="run_proxy_lift",
        ),
        set_unified_macro_feedback=set_unified_macro_feedback,
    )


def run_planner_pick_macro_full(
    env,
    config_overrides,
    *,
    run_planner_object_pregrasp_probe,
    run_planner_full_approach_to_descend,
    run_planner_object_descend,
    run_planner_close_gripper,
    run_planner_lift,
    set_unified_macro_feedback,
):
    return _run_pick_macro_common(
        env,
        config_overrides,
        default_backend="planner",
        close_backend_when_not_full="planner",
        log_prefix="[PlannerDebug] Starting FULL planner pick macro: pregrasp -> planner descend -> close -> planner lift",
        hybrid_reason=(
            "[PlannerDebug] Hybrid FULL macro mode enabled: proxy pregrasp handoff -> "
            "real planner / mplib planner descend/close/lift."
        ),
        use_full_log_prefix=(
            f"[PlannerDebug] Full-approach macro mode enabled: backend='"
            f"{resolve_macro_backend(config_overrides, default_backend='planner')}', "
            f"planner_approach_waypoints={max(int(config_overrides.get('planner_approach_waypoints', 0) or 0), 0)}. "
            "Split pregrasp/descend stages are bypassed for this macro."
        ),
        pregrasp_abort_message="[PlannerDebug] Full planner pick macro aborted at pregrasp stage.",
        full_approach_abort_message="[PlannerDebug] Full planner pick macro aborted at full-approach stage.",
        descend_abort_message="[PlannerDebug] Full planner pick macro aborted at descend stage.",
        close_abort_message="[PlannerDebug] Full planner pick macro aborted at close stage.",
        lift_abort_message="[PlannerDebug] Full planner pick macro aborted at lift stage.",
        success_message="[PlannerDebug] Full planner pick macro EXECUTE OK",
        run_proxy_pregrasp_probe=run_planner_object_pregrasp_probe,
        run_proxy_full_approach_to_descend=run_planner_full_approach_to_descend,
        run_proxy_descend=run_planner_object_descend,
        run_proxy_close_gripper=run_planner_close_gripper,
        run_proxy_lift=run_planner_lift,
        set_unified_macro_feedback=set_unified_macro_feedback,
    )
