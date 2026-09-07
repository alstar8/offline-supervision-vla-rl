from __future__ import annotations

from typing import Any, Sequence

PICK_UP_STAGE_SEQUENCE = (
    "move_to_pregrasp",
    "move_to_descend",
    "close_gripper",
    "lift_object",
    "retention_check",
)


def inject_unified_planner_backend_arg(
    argv: Sequence[str] | None,
    planner_backend_value: str,
) -> list[str]:
    forwarded = [] if argv is None else list(argv)

    existing_values: list[str] = []
    idx = 0
    while idx < len(forwarded):
        token = forwarded[idx]
        if token == "--planner_backend":
            if idx + 1 >= len(forwarded):
                raise ValueError("--planner_backend requires a value")
            existing_values.append(str(forwarded[idx + 1]))
            idx += 2
            continue
        idx += 1

    if existing_values:
        invalid_values = [value for value in existing_values if value != planner_backend_value]
        if invalid_values:
            raise ValueError(
                f"Unified backend entrypoint expected --planner_backend={planner_backend_value}, "
                f"but received: {invalid_values}"
            )
        return forwarded

    return forwarded + ["--planner_backend", planner_backend_value]


def inject_proxy_auto_pick_macro_one(
    forwarded_argv: Sequence[str] | None,
    task_plan: Any,
    *,
    scope: str,
) -> list[str]:
    forwarded = [] if forwarded_argv is None else list(forwarded_argv)
    if task_plan is None:
        return forwarded

    intent = getattr(task_plan, "intent", None)
    task_type = getattr(intent, "task_type", None)
    if task_type != "pick_up":
        return forwarded

    stages = list(getattr(task_plan, "stages", []) or [])
    stage_kinds = tuple(str(getattr(stage, "kind", "<unknown>")) for stage in stages)
    if stage_kinds != PICK_UP_STAGE_SEQUENCE:
        expected = ", ".join(PICK_UP_STAGE_SEQUENCE)
        actual = ", ".join(stage_kinds) or "<none>"
        raise ValueError(
            f"{scope} currently expects the exact stage sequence "
            f"[{expected}], but received [{actual}]"
        )

    existing_values: list[str] = []
    idx = 0
    while idx < len(forwarded):
        token = forwarded[idx]
        if token == "--auto_pick_macro":
            if idx + 1 >= len(forwarded):
                raise ValueError("--auto_pick_macro requires a value")
            existing_values.append(str(forwarded[idx + 1]))
            idx += 2
            continue
        idx += 1

    if existing_values:
        invalid_values = [value for value in existing_values if value != "1"]
        if invalid_values:
            raise ValueError(
                f"{scope} currently expects --auto_pick_macro=1, but received: {invalid_values}"
            )
        return forwarded

    return forwarded + ["--auto_pick_macro", "1"]


def resolve_unified_backend_request_argv(
    unified_request: Any,
    planner_backend_value: str,
) -> list[str]:
    if unified_request is None:
        return inject_unified_planner_backend_arg(None, planner_backend_value)

    task_plan = getattr(unified_request, "task_plan", None)
    forwarded = inject_proxy_auto_pick_macro_one(
        getattr(unified_request, "passthrough_argv", None),
        task_plan,
        scope="Unified proxy backend request",
    )
    return inject_unified_planner_backend_arg(forwarded, planner_backend_value)
