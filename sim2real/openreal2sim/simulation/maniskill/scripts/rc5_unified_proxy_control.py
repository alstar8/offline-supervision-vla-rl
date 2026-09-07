from __future__ import annotations

from contextlib import contextmanager

from openreal2sim.simulation.maniskill.planner_core import default_close_steps_for_agent

PROXY_EE_DELTA_BACKEND = "proxy_ee_delta"
PROXY_THEN_PLANNER_BACKEND = "proxy_then_planner"
PLANNER_BACKENDS = ("planner", "local_ik")
SUPPORTED_MACRO_BACKENDS = PLANNER_BACKENDS + (
    PROXY_EE_DELTA_BACKEND,
    PROXY_THEN_PLANNER_BACKEND,
)
SUPPORTED_PROXY_ONLY_MACRO_BACKENDS = (PROXY_EE_DELTA_BACKEND,)


def is_proxy_ee_delta_backend(backend: str) -> bool:
    return str(backend or "").strip().lower() == PROXY_EE_DELTA_BACKEND


def is_proxy_then_planner_backend(backend: str) -> bool:
    return str(backend or "").strip().lower() == PROXY_THEN_PLANNER_BACKEND


def resolve_macro_backend(config_overrides, default_backend: str) -> str:
    backend = str(config_overrides.get("planner_backend", "") or "").strip().lower()
    if backend in SUPPORTED_MACRO_BACKENDS:
        return backend
    return default_backend


def resolve_proxy_only_macro_backend(config_overrides, default_backend: str = PROXY_EE_DELTA_BACKEND) -> str:
    resolved_backend = resolve_macro_backend(
        config_overrides,
        default_backend=str(default_backend or PROXY_EE_DELTA_BACKEND).strip().lower(),
    )
    if resolved_backend not in SUPPORTED_PROXY_ONLY_MACRO_BACKENDS:
        valid = ", ".join(SUPPORTED_PROXY_ONLY_MACRO_BACKENDS)
        raise ValueError(
            "Proxy-only macro backend contract does not allow "
            f"planner_backend='{resolved_backend}'. Expected one of: {valid}"
        )
    return resolved_backend


def get_hybrid_jointspace_control_mode(env_unwrapped):
    agent = getattr(env_unwrapped, "agent", None)
    supported = list(getattr(agent, "supported_control_modes", []) or [])
    if "pd_joint_pos" in supported:
        return "pd_joint_pos"
    if "pd_joint_pos_vel" in supported:
        return "pd_joint_pos_vel"
    raise RuntimeError(
        "Hybrid proxy->planner handoff requires a joint-space control mode, but the agent "
        f"supports only: {supported}"
    )


@contextmanager
def temporary_agent_control_mode(env, target_control_mode, reason: str):
    env_unwrapped = env.unwrapped
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None:
        raise RuntimeError("Cannot switch control mode: env.unwrapped.agent is missing")
    previous_control_mode = getattr(agent, "control_mode", None)
    if previous_control_mode == target_control_mode:
        print(
            f"[PlannerDebug] control_mode already '{target_control_mode}' for {reason}; "
            "reusing active controller."
        )
        yield previous_control_mode
        return
    print(
        f"[PlannerDebug] Switching control_mode for {reason}: "
        f"'{previous_control_mode}' -> '{target_control_mode}'"
    )
    agent.set_control_mode(target_control_mode)
    try:
        yield previous_control_mode
    finally:
        if previous_control_mode is not None and getattr(agent, "control_mode", None) != previous_control_mode:
            print(
                f"[PlannerDebug] Restoring control_mode after {reason}: "
                f"'{getattr(agent, 'control_mode', None)}' -> '{previous_control_mode}'"
            )
            agent.set_control_mode(previous_control_mode)


def is_rc5_debug_planner_agent(agent_uid):
    return str(agent_uid).startswith("rc5_aero_hand_openr2s")


def get_default_planner_stage_method(agent_uid):
    return "rrtconnect" if is_rc5_debug_planner_agent(agent_uid) else "auto"


def get_default_debug_close_steps(agent_uid):
    return default_close_steps_for_agent(str(agent_uid))
