from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from openreal2sim.simulation.maniskill.scripts.rc5_unified_bootstrap import (
    emit_warning,
    extract_flag_values,
    has_flag,
    load_simulation_config_sections,
    pick_simulation_value,
    resolve_rc5_move_group,
)
from openreal2sim.simulation.maniskill.scripts.maniskill_num_envs_policy import (
    DEFAULT_RC5_SIM_BACKEND,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_argv import (
    PICK_UP_STAGE_SEQUENCE,
    inject_unified_planner_backend_arg,
    inject_proxy_auto_pick_macro_one,
)

PLANNER_BACKEND = "planner"
PROXY_BACKEND = "proxy_ee_delta"
HYBRID_BACKEND = "hybrid"
UNIFIED_BACKENDS = (PLANNER_BACKEND, PROXY_BACKEND, HYBRID_BACKEND)
DEFAULT_EE_DELTA_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
EE_DELTA_CONTROL_MODES = {
    "pd_ee_delta_pose",
    "pd_ee_target_delta_pose",
    DEFAULT_EE_DELTA_CONTROL_MODE,
}

UNIFIED_PLANNER_BACKEND_RRTCONNECT = "rrtconnect"
UNIFIED_PLANNER_BACKEND_LOCAL_IK = "local_ik"
UNIFIED_PLANNER_BACKEND_PROXY = "proxy"
_UNIFIED_PLANNER_BACKEND_CONFIG_KEYS = (
    "unified_planner_backend",
    "planner_backend",
)


def _normalize_unified_planner_backend_mode(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if not value:
        return UNIFIED_PLANNER_BACKEND_RRTCONNECT
    if value == "planner":
        return UNIFIED_PLANNER_BACKEND_RRTCONNECT
    if value == PROXY_BACKEND:
        return UNIFIED_PLANNER_BACKEND_PROXY
    if value in {
        UNIFIED_PLANNER_BACKEND_PROXY,
        UNIFIED_PLANNER_BACKEND_RRTCONNECT,
        UNIFIED_PLANNER_BACKEND_LOCAL_IK,
    }:
        return value
    valid = ", ".join(
        (
            UNIFIED_PLANNER_BACKEND_PROXY,
            UNIFIED_PLANNER_BACKEND_RRTCONNECT,
            UNIFIED_PLANNER_BACKEND_LOCAL_IK,
            PROXY_BACKEND,
            "planner",
        )
    )
    raise ValueError(
        "Invalid unified planner backend mode "
        f"'{raw_value}'. Expected one of: {valid}"
    )


def _resolve_unified_planner_backend_mode_from_request(request: "UnifiedBackendRequest") -> str:
    bootstrap = getattr(request, "bootstrap", None)
    if bootstrap is None:
        return UNIFIED_PLANNER_BACKEND_RRTCONNECT

    sections = load_simulation_config_sections(bootstrap.config_path, bootstrap.key)
    for key in _UNIFIED_PLANNER_BACKEND_CONFIG_KEYS:
        raw_value = pick_simulation_value(sections, key, None)
        if raw_value is None:
            continue
        normalized = str(raw_value).strip().lower()
        if key == "planner_backend" and normalized not in {
            "planner",
            PROXY_BACKEND,
            UNIFIED_PLANNER_BACKEND_PROXY,
            UNIFIED_PLANNER_BACKEND_RRTCONNECT,
            UNIFIED_PLANNER_BACKEND_LOCAL_IK,
        }:
            continue
        return _normalize_unified_planner_backend_mode(raw_value)
    return UNIFIED_PLANNER_BACKEND_RRTCONNECT


def _map_unified_planner_mode_to_legacy_backend(mode: str) -> str:
    if mode == UNIFIED_PLANNER_BACKEND_LOCAL_IK:
        return UNIFIED_PLANNER_BACKEND_LOCAL_IK
    return "planner"


def resolve_motion_backend_from_unified_planner_mode(mode: str) -> str:
    normalized = _normalize_unified_planner_backend_mode(mode)
    if normalized == UNIFIED_PLANNER_BACKEND_PROXY:
        return PROXY_BACKEND
    return PLANNER_BACKEND


def _normalize_planner_robot_uid(robot_uid: str | None) -> str | None:
    if robot_uid is None:
        return None
    normalized = str(robot_uid).strip()
    if not normalized:
        return None
    if normalized.endswith("_rl"):
        return normalized[:-3]
    return normalized


def _upsert_flag_value(argv: Sequence[str], flag: str, value: str | None) -> list[str]:
    updated: list[str] = []
    idx = 0
    tokens = list(argv)
    while idx < len(tokens):
        token = tokens[idx]
        if token == flag:
            idx += 2
            continue
        updated.append(token)
        idx += 1
    if value is not None:
        updated.extend([flag, str(value)])
    return updated


@dataclass(frozen=True)
class RC5AssetOverrideConfig:
    asset_dir: str | None = None
    urdf_filename: str | None = None
    planning_asset_dir: str | None = None
    planner_use_main_assets: bool | None = None


@dataclass(frozen=True)
class RC5PlannerRuntimeOverrideConfig:
    manip_object_id: str | None = None
    planner_pregrasp_method: str | None = None
    planner_pregrasp_refine_steps: int | None = None
    rc5_obb_target_semantics: str | None = None
    rc5_object_profile_target_semantics: str | None = None


def _resolve_rc5_asset_override_config_from_request(
    request: "UnifiedBackendRequest",
) -> RC5AssetOverrideConfig:
    bootstrap = getattr(request, "bootstrap", None)
    if bootstrap is None:
        return RC5AssetOverrideConfig()
    sections = load_simulation_config_sections(bootstrap.config_path, bootstrap.key)
    return RC5AssetOverrideConfig(
        asset_dir=pick_simulation_value(sections, "rc5_asset_dir", None),
        urdf_filename=pick_simulation_value(sections, "rc5_urdf_filename", None),
        planning_asset_dir=pick_simulation_value(sections, "rc5_planning_asset_dir", None),
        planner_use_main_assets=pick_simulation_value(sections, "rc5_planner_use_main_assets", None),
    )


def _resolve_rc5_planner_runtime_override_config_from_request(
    request: "UnifiedBackendRequest",
) -> RC5PlannerRuntimeOverrideConfig:
    bootstrap = getattr(request, "bootstrap", None)
    if bootstrap is None:
        return RC5PlannerRuntimeOverrideConfig()

    sections = load_simulation_config_sections(bootstrap.config_path, bootstrap.key)
    task_plan = getattr(request, "task_plan", None)
    intent = getattr(task_plan, "intent", None)
    intent_object_id = getattr(intent, "object_id", None)
    manip_object_id = intent_object_id or pick_simulation_value(sections, "manip_object_id", None)
    return RC5PlannerRuntimeOverrideConfig(
        manip_object_id=None if manip_object_id is None else str(manip_object_id),
        planner_pregrasp_method=pick_simulation_value(sections, "planner_pregrasp_method", None),
        planner_pregrasp_refine_steps=pick_simulation_value(
            sections,
            "planner_pregrasp_refine_steps",
            None,
        ),
        rc5_obb_target_semantics=pick_simulation_value(
            sections,
            "planner_rc5_obb_target_semantics",
            None,
        ),
        rc5_object_profile_target_semantics=pick_simulation_value(
            sections,
            "planner_rc5_object_profile_target_semantics",
            None,
        ),
    )


def parse_requested_num_envs_from_argv(passthrough_argv: Sequence[str]) -> int:
    values = extract_flag_values(passthrough_argv, "--num_envs")
    if not values:
        return 1
    if len(values) > 1:
        raise ValueError("Unified backend execution expects --num_envs to be specified at most once.")
    raw_value = str(values[0]).strip()
    if not raw_value:
        raise ValueError("--num_envs requires an integer value.")
    try:
        num_envs = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"--num_envs must be an integer, got '{raw_value}'.") from exc
    if num_envs <= 0:
        raise ValueError("--num_envs must be >= 1.")
    return num_envs


def validate_runtime_num_envs_supported(num_envs: int, *, scope: str) -> None:
    if int(num_envs) != 1:
        raise ValueError(
            f"{scope} currently supports only --num_envs=1 for actual execution. "
            f"Received --num_envs={int(num_envs)}. "
            "Batched ManiSkill envs are not yet safely supported because the current "
            "single-env execution contract does not preserve per-env poses, grasp flags, and success accounting."
        )


def validate_runtime_num_envs_from_argv(
    passthrough_argv: Sequence[str],
    *,
    scope: str,
) -> int:
    num_envs = parse_requested_num_envs_from_argv(passthrough_argv)
    validate_runtime_num_envs_supported(num_envs, scope=scope)
    return num_envs


@dataclass(frozen=True)
class BackendDispatchPlan:
    motion_backend: str
    module_name: str
    callable_name: str
    forwarded_argv: List[str]
    env_updates: Dict[str, str]


@dataclass(frozen=True)
class UnifiedBackendRequest:
    motion_backend: str
    passthrough_argv: List[str]
    bootstrap: Any
    task_plan: Any
    trace_seed: Any = None
    rc5_asset_dir: str | None = None
    rc5_move_group: str | None = None
    warning_scope: str = "RC5Unified"


@dataclass(frozen=True)
class UnifiedBackendResult:
    motion_backend: str
    exit_code: int
    dispatch_plan: BackendDispatchPlan
    trace_seed: Any = None
    trace_record: Any = None


@dataclass(frozen=True)
class NormalizedBackendExecutionFeedback:
    exit_code: int
    runtime_events: tuple["CanonicalTraceEventRecord", ...] = ()


@dataclass(frozen=True)
class CanonicalEpisodeTraceSeed:
    motion_backend: str
    config_key: str | None
    robot_uids: str | None
    task_type: str | None
    object_id: str | None
    destination_id: str | None
    prompt: str | None
    stage_names: tuple[str, ...]
    stage_kinds: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalStageTraceRecord:
    order_index: int
    name: str
    kind: str
    lifecycle: str = "submitted_to_backend"
    transitions: tuple[str, ...] = ("planned", "submitted_to_backend")
    backend_execution_outcome: str = "unknown"
    stage_outcome: str = "not_observed"


@dataclass(frozen=True)
class CanonicalTraceEventRecord:
    event_type: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class CanonicalPerEnvTraceRecord:
    env_index: int
    semantic_task_success: bool | None = None
    failed_stage: str | None = None
    execution_outcome: str = "unknown"
    is_src_obj_grasped: bool | None = None
    obj_height_above_table: float | None = None
    gripper_obj_dist: float | None = None
    gripper_goal_dist: float | None = None
    tcp_pose: tuple[float, ...] | None = None
    object_pose: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CanonicalBatchTraceRecord:
    requested_num_envs: int
    runtime_batch_size: int | None = None
    successful_env_count: int | None = None
    failed_env_indices: tuple[int, ...] = ()
    artifacts_recorded_per_env: bool | None = None
    runtime_batch_feedback_present: bool = False
    per_env_records: tuple[CanonicalPerEnvTraceRecord, ...] = ()


@dataclass(frozen=True)
class CanonicalEpisodeTraceRecord:
    seed: CanonicalEpisodeTraceSeed
    stage_records: tuple[CanonicalStageTraceRecord, ...]
    events: tuple[CanonicalTraceEventRecord, ...]
    execution_outcome: str
    exit_code: int
    dispatch_module: str
    dispatch_callable: str
    batch_trace: CanonicalBatchTraceRecord | None = None


@dataclass(frozen=True)
class DispatchBackend:
    motion_backend: str
    module_name: str
    callable_name: str = "main"
    supported_task_types: tuple[str, ...] = ("pick_up",)
    supported_stage_kinds: tuple[str, ...] = (
        "move_to_pregrasp",
        "move_to_descend",
        "close_gripper",
        "lift_object",
        "retention_check",
    )

    @property
    def requires_proxy_runtime_defaults(self) -> bool:
        return self.motion_backend in {PROXY_BACKEND, HYBRID_BACKEND}

    def validate_num_envs(self, passthrough_argv: Sequence[str]) -> None:
        validate_runtime_num_envs_from_argv(
            passthrough_argv,
            scope=f"Unified backend '{self.motion_backend}'",
        )

    def validate_request(self, request: UnifiedBackendRequest) -> None:
        self.validate_num_envs(request.passthrough_argv)

        task_plan = request.task_plan
        if task_plan is None:
            return

        intent = getattr(task_plan, "intent", None)
        task_type = getattr(intent, "task_type", None)
        if task_type is not None and task_type not in self.supported_task_types:
            valid = ", ".join(self.supported_task_types)
            raise ValueError(
                f"Backend '{self.motion_backend}' does not support task_type '{task_type}'. "
                f"Supported task types: {valid}"
            )

        stages = list(getattr(task_plan, "stages", []) or [])
        unsupported_stage_kinds = [
            getattr(stage, "kind", None)
            for stage in stages
            if getattr(stage, "kind", None) not in self.supported_stage_kinds
        ]
        if unsupported_stage_kinds:
            unique_kinds = ", ".join(sorted({kind for kind in unsupported_stage_kinds if kind is not None}))
            valid = ", ".join(self.supported_stage_kinds)
            raise ValueError(
                f"Backend '{self.motion_backend}' does not support stage kind(s): {unique_kinds}. "
                f"Supported stage kinds: {valid}"
            )

    def _apply_task_runtime_defaults(
        self,
        forwarded: Sequence[str],
        request: UnifiedBackendRequest,
    ) -> List[str]:
        return list(forwarded)

    def build_dispatch_plan(self, request: UnifiedBackendRequest) -> BackendDispatchPlan:
        self.validate_request(request)
        forwarded = inject_runtime_defaults(
            request.passthrough_argv,
            request.bootstrap,
            self.motion_backend,
            warning_scope=request.warning_scope,
        )
        forwarded = self._apply_task_runtime_defaults(forwarded, request)
        planner_backend_values = extract_flag_values(forwarded, "--planner_backend")
        if planner_backend_values:
            raise ValueError(
                "--planner_backend should not be passed to run_rc5_unified.py directly. "
                "Use --motion_backend planner|proxy_ee_delta|hybrid instead."
            )
        asset_overrides = _resolve_rc5_asset_override_config_from_request(request)
        env_updates = build_env_updates(
            rc5_asset_dir=request.rc5_asset_dir or asset_overrides.asset_dir,
            rc5_urdf_filename=asset_overrides.urdf_filename,
            rc5_planning_asset_dir=asset_overrides.planning_asset_dir,
            rc5_planner_use_main_assets=asset_overrides.planner_use_main_assets,
            rc5_move_group=request.rc5_move_group,
            warning_scope=request.warning_scope,
        )
        return BackendDispatchPlan(
            motion_backend=self.motion_backend,
            module_name=self.module_name,
            callable_name=self.callable_name,
            forwarded_argv=forwarded,
            env_updates=env_updates,
        )

    def execute(self, request: UnifiedBackendRequest) -> UnifiedBackendResult:
        plan = self.build_dispatch_plan(request)
        exit_code = execute_dispatch_plan(plan)
        trace_seed = ensure_trace_seed(request)
        trace_record = build_canonical_trace_record(
            trace_seed=trace_seed,
            dispatch_plan=plan,
            exit_code=exit_code,
        )
        return UnifiedBackendResult(
            motion_backend=self.motion_backend,
            exit_code=exit_code,
            dispatch_plan=plan,
            trace_seed=trace_seed,
            trace_record=trace_record,
        )


@dataclass(frozen=True)
class PlannerBackend(DispatchBackend):
    motion_backend: str = PLANNER_BACKEND
    module_name: str = "openreal2sim.simulation.maniskill.scripts.run_rc5_solver_debug_planner_LEGACY"
    callable_name: str = "main"

    def build_dispatch_plan(self, request: UnifiedBackendRequest) -> BackendDispatchPlan:
        self.validate_request(request)
        forwarded = inject_runtime_defaults(
            request.passthrough_argv,
            request.bootstrap,
            self.motion_backend,
            warning_scope=request.warning_scope,
        )

        planner_backend_values = extract_flag_values(forwarded, "--planner_backend")
        if planner_backend_values:
            raise ValueError(
                "--planner_backend should not be passed to run_rc5_unified.py directly. "
                "Set local.<key>.simulation.unified_planner_backend to 'local_ik' or 'rrtconnect' instead."
            )

        planner_mode = _resolve_unified_planner_backend_mode_from_request(request)
        if planner_mode == UNIFIED_PLANNER_BACKEND_PROXY:
            raise ValueError(
                "Unified planner backend resolved to 'proxy', but PlannerBackend was selected. "
                "Use motion_backend='proxy_ee_delta' or let run_rc5_unified.py auto-resolve it from config."
            )
        legacy_planner_backend = _map_unified_planner_mode_to_legacy_backend(planner_mode)
        forwarded = inject_unified_planner_backend_arg(forwarded, legacy_planner_backend)

        explicit_robot_uids = extract_flag_values(request.passthrough_argv, "--robot_uids")
        requested_robot_uid = (
            explicit_robot_uids[-1]
            if explicit_robot_uids
            else getattr(request.bootstrap, "robot_uids", None)
        )
        planner_robot_uid = _normalize_planner_robot_uid(requested_robot_uid)
        if planner_robot_uid is not None and planner_robot_uid != requested_robot_uid:
            emit_warning(
                request.warning_scope,
                "Planner backend normalized robot_uids from "
                f"'{requested_robot_uid}' to '{planner_robot_uid}' for joint-space execution.",
            )
        forwarded = _upsert_flag_value(forwarded, "--robot_uids", planner_robot_uid)

        explicit_control_mode = extract_flag_values(request.passthrough_argv, "--control_mode")
        requested_control_mode = (
            explicit_control_mode[-1]
            if explicit_control_mode
            else getattr(request.bootstrap, "control_mode", None)
        )
        planner_control_mode = (
            "pd_joint_pos"
            if requested_control_mode is None or is_ee_delta_control_mode(requested_control_mode)
            else str(requested_control_mode)
        )
        if planner_control_mode != requested_control_mode:
            emit_warning(
                request.warning_scope,
                "Planner backend normalized control_mode from "
                f"'{requested_control_mode}' to '{planner_control_mode}'.",
            )
        forwarded = _upsert_flag_value(forwarded, "--control_mode", planner_control_mode)

        planner_runtime_overrides = _resolve_rc5_planner_runtime_override_config_from_request(request)
        forwarded = _upsert_flag_value(
            forwarded,
            "--manip_object_id",
            planner_runtime_overrides.manip_object_id,
        )
        forwarded = _upsert_flag_value(
            forwarded,
            "--planner_pregrasp_method",
            None
            if planner_runtime_overrides.planner_pregrasp_method is None
            else str(planner_runtime_overrides.planner_pregrasp_method),
        )
        forwarded = _upsert_flag_value(
            forwarded,
            "--planner_pregrasp_refine_steps",
            None
            if planner_runtime_overrides.planner_pregrasp_refine_steps is None
            else str(int(planner_runtime_overrides.planner_pregrasp_refine_steps)),
        )
        forwarded = _upsert_flag_value(
            forwarded,
            "--rc5_obb_target_semantics",
            None
            if planner_runtime_overrides.rc5_obb_target_semantics is None
            else str(planner_runtime_overrides.rc5_obb_target_semantics),
        )
        forwarded = _upsert_flag_value(
            forwarded,
            "--rc5_object_profile_target_semantics",
            None
            if planner_runtime_overrides.rc5_object_profile_target_semantics is None
            else str(planner_runtime_overrides.rc5_object_profile_target_semantics),
        )

        emit_warning(
            request.warning_scope,
            "Planner backend dispatch selected unified_planner_backend="
            f"{planner_mode} (legacy --planner_backend={legacy_planner_backend}).",
        )

        asset_overrides = _resolve_rc5_asset_override_config_from_request(request)
        env_updates = build_env_updates(
            rc5_asset_dir=request.rc5_asset_dir or asset_overrides.asset_dir,
            rc5_urdf_filename=asset_overrides.urdf_filename,
            rc5_planning_asset_dir=asset_overrides.planning_asset_dir,
            rc5_planner_use_main_assets=asset_overrides.planner_use_main_assets,
            rc5_move_group=request.rc5_move_group,
            warning_scope=request.warning_scope,
        )
        return BackendDispatchPlan(
            motion_backend=self.motion_backend,
            module_name=self.module_name,
            callable_name=self.callable_name,
            forwarded_argv=forwarded,
            env_updates=env_updates,
        )

    def execute(self, request: UnifiedBackendRequest) -> UnifiedBackendResult:
        from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_runtime import (
            clear_unified_macro_feedback,
            consume_unified_macro_feedback,
        )

        plan = self.build_dispatch_plan(request)
        clear_unified_macro_feedback()
        exit_code = execute_dispatch_plan(plan)
        macro_feedback = consume_unified_macro_feedback()
        normalized_exit_code = int(exit_code)
        runtime_events: tuple[CanonicalTraceEventRecord, ...] = ()
        if isinstance(macro_feedback, Mapping):
            semantic_task_success = bool(macro_feedback.get("semantic_task_success"))
            failed_stage = macro_feedback.get("failed_stage")
            if normalized_exit_code == 0 and not semantic_task_success:
                normalized_exit_code = 1
            runtime_events = (
                CanonicalTraceEventRecord(
                    event_type="macro_finished",
                    payload={
                        "execution_outcome": "success" if semantic_task_success else "failed",
                        "semantic_task_success": semantic_task_success,
                        "failed_stage": failed_stage,
                    },
                ),
            )
        trace_seed = ensure_trace_seed(request)
        trace_record = build_canonical_trace_record(
            trace_seed=trace_seed,
            dispatch_plan=plan,
            exit_code=normalized_exit_code,
            runtime_events=runtime_events,
        )
        return UnifiedBackendResult(
            motion_backend=self.motion_backend,
            exit_code=normalized_exit_code,
            dispatch_plan=plan,
            trace_seed=trace_seed,
            trace_record=trace_record,
        )


@dataclass(frozen=True)
class ProxyEEDeltaBackend(DispatchBackend):
    motion_backend: str = PROXY_BACKEND
    module_name: str = "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_runtime"
    callable_name: str = "run_unified_proxy_backend_request"

    def validate_num_envs(self, passthrough_argv: Sequence[str]) -> None:
        num_envs = parse_requested_num_envs_from_argv(passthrough_argv)
        if int(num_envs) <= 0:
            raise ValueError("--num_envs must be >= 1.")

    def _apply_task_runtime_defaults(
        self,
        forwarded: Sequence[str],
        request: UnifiedBackendRequest,
    ) -> List[str]:
        return _inject_proxy_auto_pick_macro_from_task_plan(
            forwarded,
            request,
            warning_scope=request.warning_scope,
        )

    def execute(self, request: UnifiedBackendRequest) -> UnifiedBackendResult:
        plan = self.build_dispatch_plan(request)
        apply_env_updates(plan.env_updates)
        module = importlib.import_module(self.module_name)
        entrypoint = getattr(module, plan.callable_name)
        feedback = normalize_backend_execution_feedback(entrypoint(request))
        trace_seed = ensure_trace_seed(request)
        trace_record = build_canonical_trace_record(
            trace_seed=trace_seed,
            dispatch_plan=plan,
            exit_code=feedback.exit_code,
            runtime_events=feedback.runtime_events,
        )
        return UnifiedBackendResult(
            motion_backend=self.motion_backend,
            exit_code=feedback.exit_code,
            dispatch_plan=plan,
            trace_seed=trace_seed,
            trace_record=trace_record,
        )


@dataclass(frozen=True)
class HybridBackend(DispatchBackend):
    motion_backend: str = HYBRID_BACKEND
    module_name: str = "openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_runtime"
    callable_name: str = "run_unified_hybrid_backend_request"

    def _apply_task_runtime_defaults(
        self,
        forwarded: Sequence[str],
        request: UnifiedBackendRequest,
    ) -> List[str]:
        return _inject_proxy_auto_pick_macro_from_task_plan(
            forwarded,
            request,
            warning_scope=request.warning_scope,
        )

    def execute(self, request: UnifiedBackendRequest) -> UnifiedBackendResult:
        plan = self.build_dispatch_plan(request)
        apply_env_updates(plan.env_updates)
        module = importlib.import_module(self.module_name)
        entrypoint = getattr(module, plan.callable_name)
        feedback = normalize_backend_execution_feedback(entrypoint(request))
        trace_seed = ensure_trace_seed(request)
        trace_record = build_canonical_trace_record(
            trace_seed=trace_seed,
            dispatch_plan=plan,
            exit_code=feedback.exit_code,
            runtime_events=feedback.runtime_events,
        )
        return UnifiedBackendResult(
            motion_backend=self.motion_backend,
            exit_code=feedback.exit_code,
            dispatch_plan=plan,
            trace_seed=trace_seed,
            trace_record=trace_record,
        )

DISPATCH_BACKENDS = {
    PLANNER_BACKEND: PlannerBackend(),
    PROXY_BACKEND: ProxyEEDeltaBackend(),
    HYBRID_BACKEND: HybridBackend(),
}


def is_ee_delta_control_mode(control_mode: str | None) -> bool:
    return control_mode in EE_DELTA_CONTROL_MODES


def resolve_effective_control_mode(
    robot_uid: str | None,
    requested_control_mode: str | None,
    *,
    default_joint_mode: str = "pd_joint_pos",
) -> str:
    if requested_control_mode is not None:
        return requested_control_mode
    if robot_uid is not None and str(robot_uid).endswith("_rl"):
        return DEFAULT_EE_DELTA_CONTROL_MODE
    return default_joint_mode


def select_startup_stabilize_control_mode(
    *,
    supported_control_modes: Sequence[str],
    requested_control_mode: str | None,
) -> str | None:
    if not is_ee_delta_control_mode(requested_control_mode):
        return requested_control_mode
    supported = list(supported_control_modes or [])
    if "pd_joint_pos" in supported:
        return "pd_joint_pos"
    if "pd_joint_pos_vel" in supported:
        return "pd_joint_pos_vel"
    return None


def build_env_updates(
    *,
    rc5_asset_dir: str | None = None,
    rc5_urdf_filename: str | None = None,
    rc5_planning_asset_dir: str | None = None,
    rc5_planner_use_main_assets: bool | None = None,
    rc5_move_group: str | None = None,
    warning_scope: str = "RC5Unified",
) -> Dict[str, str]:
    env_updates: Dict[str, str] = {}
    if rc5_asset_dir:
        env_updates["RC5_AERO_HAND_ASSET_DIR"] = str(rc5_asset_dir)
    if rc5_urdf_filename:
        env_updates["RC5_AERO_HAND_URDF_FILENAME"] = str(rc5_urdf_filename)
    if rc5_planning_asset_dir:
        env_updates["OPENR2S_RC5_PLANNING_ASSET_DIR"] = str(rc5_planning_asset_dir)
    if rc5_planner_use_main_assets is not None:
        env_updates["RC5_DEBUG_PLANNER_USE_MAIN_ASSETS"] = (
            "1" if bool(rc5_planner_use_main_assets) else "0"
        )
    if rc5_move_group:
        env_updates["OPENR2S_RC5_MOVE_GROUP"] = resolve_rc5_move_group(
            rc5_move_group,
            env={},
            scope=warning_scope,
            warn_on_env=False,
            warn_on_default=False,
        )
    return env_updates


def build_canonical_trace_seed(
    *,
    motion_backend: str,
    bootstrap,
    task_plan,
) -> CanonicalEpisodeTraceSeed:
    intent = getattr(task_plan, "intent", None)
    stages = list(getattr(task_plan, "stages", []) or [])
    return CanonicalEpisodeTraceSeed(
        motion_backend=motion_backend,
        config_key=getattr(bootstrap, "key", None),
        robot_uids=getattr(bootstrap, "robot_uids", None),
        task_type=getattr(intent, "task_type", None),
        object_id=getattr(intent, "object_id", None),
        destination_id=getattr(intent, "destination_id", None),
        prompt=getattr(intent, "prompt", None),
        stage_names=tuple(str(getattr(stage, "name", "<unnamed>")) for stage in stages),
        stage_kinds=tuple(str(getattr(stage, "kind", "<unknown>")) for stage in stages),
    )


def _normalize_macro_failed_stage(
    failed_stage: str | None,
    *,
    trace_seed: CanonicalEpisodeTraceSeed,
) -> str | None:
    if failed_stage is None:
        return None
    normalized = str(failed_stage).strip().lower()
    if not normalized:
        return None
    if normalized in set(trace_seed.stage_names):
        return normalized
    if normalized == "full_approach":
        # Full-approach mode collapses pregrasp+descend into one macro step. The
        # safest unified interpretation is that we never successfully cleared the
        # first planned approach stage.
        return "pregrasp" if "pregrasp" in set(trace_seed.stage_names) else None
    return None


def _build_stage_records_from_runtime_events(
    *,
    trace_seed: CanonicalEpisodeTraceSeed,
    exit_code: int,
    runtime_events: Sequence[CanonicalTraceEventRecord] = (),
) -> tuple[CanonicalStageTraceRecord, ...]:
    macro_payload = extract_macro_finished_payload(runtime_events)
    if macro_payload is None:
        backend_execution_outcome = "success" if exit_code == 0 else "failed"
        return tuple(
            CanonicalStageTraceRecord(
                order_index=idx,
                name=name,
                kind=kind,
                lifecycle="submitted_to_backend",
                transitions=("planned", "submitted_to_backend"),
                backend_execution_outcome=backend_execution_outcome,
                stage_outcome="not_observed",
            )
            for idx, (name, kind) in enumerate(zip(trace_seed.stage_names, trace_seed.stage_kinds))
        )

    semantic_task_success = macro_payload.get("semantic_task_success")
    normalized_failed_stage = _normalize_macro_failed_stage(
        macro_payload.get("failed_stage"),
        trace_seed=trace_seed,
    )
    failure_index = (
        None
        if normalized_failed_stage is None
        else next(
            (
                idx
                for idx, name in enumerate(trace_seed.stage_names)
                if name == normalized_failed_stage
            ),
            None,
        )
    )

    stage_records: list[CanonicalStageTraceRecord] = []
    for idx, (name, kind) in enumerate(zip(trace_seed.stage_names, trace_seed.stage_kinds)):
        if semantic_task_success is True:
            stage_records.append(
                CanonicalStageTraceRecord(
                    order_index=idx,
                    name=name,
                    kind=kind,
                    lifecycle="completed",
                    transitions=("planned", "submitted_to_backend", "completed"),
                    backend_execution_outcome="success",
                    stage_outcome="success",
                )
            )
            continue

        if semantic_task_success is False and failure_index is not None:
            if idx < failure_index:
                stage_records.append(
                    CanonicalStageTraceRecord(
                        order_index=idx,
                        name=name,
                        kind=kind,
                        lifecycle="completed",
                        transitions=("planned", "submitted_to_backend", "completed"),
                        backend_execution_outcome="success",
                        stage_outcome="success",
                    )
                )
            elif idx == failure_index:
                stage_records.append(
                    CanonicalStageTraceRecord(
                        order_index=idx,
                        name=name,
                        kind=kind,
                        lifecycle="failed",
                        transitions=("planned", "submitted_to_backend", "failed"),
                        backend_execution_outcome="failed",
                        stage_outcome="failed",
                    )
                )
            else:
                stage_records.append(
                    CanonicalStageTraceRecord(
                        order_index=idx,
                        name=name,
                        kind=kind,
                        lifecycle="not_run",
                        transitions=("planned", "not_run"),
                        backend_execution_outcome="not_run",
                        stage_outcome="not_run",
                    )
                )
            continue

        backend_execution_outcome = "success" if exit_code == 0 else "failed"
        stage_records.append(
            CanonicalStageTraceRecord(
                order_index=idx,
                name=name,
                kind=kind,
                lifecycle="submitted_to_backend",
                transitions=("planned", "submitted_to_backend"),
                backend_execution_outcome=backend_execution_outcome,
                stage_outcome="unknown",
            )
        )

    return tuple(stage_records)


def build_canonical_trace_record(
    *,
    trace_seed: CanonicalEpisodeTraceSeed | None,
    dispatch_plan: BackendDispatchPlan,
    exit_code: int,
    runtime_events: Sequence[CanonicalTraceEventRecord] = (),
) -> CanonicalEpisodeTraceRecord | None:
    if trace_seed is None:
        return None
    stage_records = _build_stage_records_from_runtime_events(
        trace_seed=trace_seed,
        exit_code=exit_code,
        runtime_events=runtime_events,
    )
    events = build_canonical_trace_events(
        trace_seed=trace_seed,
        dispatch_plan=dispatch_plan,
        exit_code=exit_code,
    )
    events = tuple(events) + tuple(runtime_events)
    trace_execution_outcome = resolve_trace_execution_outcome(
        exit_code=exit_code,
        runtime_events=runtime_events,
    )
    batch_trace = build_canonical_batch_trace_record(
        dispatch_plan=dispatch_plan,
        runtime_events=runtime_events,
    )
    return CanonicalEpisodeTraceRecord(
        seed=trace_seed,
        stage_records=stage_records,
        events=events,
        execution_outcome=trace_execution_outcome,
        exit_code=exit_code,
        dispatch_module=dispatch_plan.module_name,
        dispatch_callable=dispatch_plan.callable_name,
        batch_trace=batch_trace,
    )


def build_canonical_trace_events(
    *,
    trace_seed: CanonicalEpisodeTraceSeed,
    dispatch_plan: BackendDispatchPlan,
    exit_code: int,
) -> tuple[CanonicalTraceEventRecord, ...]:
    env_keys = tuple(sorted(dispatch_plan.env_updates.keys()))
    return (
        CanonicalTraceEventRecord(
            event_type="dispatch_planned",
            payload={
                "motion_backend": trace_seed.motion_backend,
                "task_type": trace_seed.task_type,
                "stage_count": len(trace_seed.stage_kinds),
                "forwarded_argc": len(dispatch_plan.forwarded_argv),
                "dispatch_module": dispatch_plan.module_name,
                "dispatch_callable": dispatch_plan.callable_name,
            },
        ),
        CanonicalTraceEventRecord(
            event_type="dispatch_started",
            payload={
                "env_update_keys": env_keys,
                "env_update_count": len(env_keys),
            },
        ),
        CanonicalTraceEventRecord(
            event_type="dispatch_finished",
            payload={
                "exit_code": exit_code,
                "execution_outcome": "success" if exit_code == 0 else "failed",
            },
        ),
    )


def normalize_backend_execution_feedback(value: Any) -> NormalizedBackendExecutionFeedback:
    if value is None or isinstance(value, int):
        return NormalizedBackendExecutionFeedback(exit_code=_normalize_exit_code(value))

    if isinstance(value, dict):
        raw_events = list(value.get("runtime_events", []) or [])
        return NormalizedBackendExecutionFeedback(
            exit_code=_normalize_exit_code(value.get("exit_code", 0)),
            runtime_events=tuple(_coerce_trace_event_record(item) for item in raw_events),
        )

    raise TypeError(
        "Unified backend request hook must return None, int, or dict with optional "
        "'exit_code' and 'runtime_events'."
    )


def _coerce_per_env_trace_record(item: Mapping[str, Any]) -> CanonicalPerEnvTraceRecord:
    tcp_pose = item.get("tcp_pose")
    object_pose = item.get("object_pose")
    return CanonicalPerEnvTraceRecord(
        env_index=int(item.get("env_index", 0)),
        semantic_task_success=(
            None
            if item.get("semantic_task_success") is None
            else bool(item.get("semantic_task_success"))
        ),
        failed_stage=item.get("failed_stage"),
        execution_outcome=(
            "success"
            if item.get("semantic_task_success") is True
            else ("failed" if item.get("semantic_task_success") is False else "unknown")
        ),
        is_src_obj_grasped=(
            None
            if item.get("is_src_obj_grasped") is None
            else bool(item.get("is_src_obj_grasped"))
        ),
        obj_height_above_table=(
            None
            if item.get("obj_height_above_table") is None
            else float(item.get("obj_height_above_table"))
        ),
        gripper_obj_dist=(
            None
            if item.get("gripper_obj_dist") is None
            else float(item.get("gripper_obj_dist"))
        ),
        gripper_goal_dist=(
            None
            if item.get("gripper_goal_dist") is None
            else float(item.get("gripper_goal_dist"))
        ),
        tcp_pose=None if tcp_pose is None else tuple(float(v) for v in tcp_pose),
        object_pose=None if object_pose is None else tuple(float(v) for v in object_pose),
    )


def build_canonical_batch_trace_record(
    *,
    dispatch_plan: BackendDispatchPlan,
    runtime_events: Sequence[CanonicalTraceEventRecord] = (),
) -> CanonicalBatchTraceRecord | None:
    requested_num_envs = parse_requested_num_envs_from_argv(dispatch_plan.forwarded_argv)
    if int(requested_num_envs) <= 1:
        return None

    batch_payload = None
    for item in reversed(tuple(runtime_events)):
        if item.event_type == "batch_runtime_feedback" and isinstance(item.payload, Mapping):
            batch_payload = dict(item.payload)
            break

    if batch_payload is None:
        return CanonicalBatchTraceRecord(
            requested_num_envs=int(requested_num_envs),
            runtime_batch_feedback_present=False,
        )

    per_env_payloads = list(batch_payload.get("per_env_feedback", []) or [])
    return CanonicalBatchTraceRecord(
        requested_num_envs=int(requested_num_envs),
        runtime_batch_size=(
            None if batch_payload.get("batch_size") is None else int(batch_payload.get("batch_size"))
        ),
        successful_env_count=(
            None
            if batch_payload.get("successful_env_count") is None
            else int(batch_payload.get("successful_env_count"))
        ),
        failed_env_indices=tuple(int(item) for item in (batch_payload.get("failed_env_indices") or [])),
        artifacts_recorded_per_env=(
            None
            if batch_payload.get("artifacts_recorded_per_env") is None
            else bool(batch_payload.get("artifacts_recorded_per_env"))
        ),
        runtime_batch_feedback_present=True,
        per_env_records=tuple(
            _coerce_per_env_trace_record(item)
            for item in per_env_payloads
            if isinstance(item, Mapping)
        ),
    )


def extract_macro_finished_payload(
    runtime_events: Sequence[CanonicalTraceEventRecord] = (),
) -> Dict[str, Any] | None:
    for item in reversed(tuple(runtime_events)):
        if getattr(item, "event_type", None) != "macro_finished":
            continue
        payload = getattr(item, "payload", None)
        if isinstance(payload, dict):
            return dict(payload)
    return None


def resolve_trace_execution_outcome(
    *,
    exit_code: int,
    runtime_events: Sequence[CanonicalTraceEventRecord] = (),
) -> str:
    outcome = "success" if exit_code == 0 else "failed"
    payload = extract_macro_finished_payload(runtime_events)
    if payload is None:
        return outcome
    event_outcome = payload.get("execution_outcome")
    if isinstance(event_outcome, str) and event_outcome in {"success", "failed"}:
        outcome = event_outcome
    if payload.get("semantic_task_success") is False:
        return "failed"
    return outcome


def _coerce_trace_event_record(value: Any) -> CanonicalTraceEventRecord:
    if isinstance(value, CanonicalTraceEventRecord):
        return value
    if isinstance(value, dict):
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise TypeError("runtime event payload must be a dict")
        return CanonicalTraceEventRecord(
            event_type=str(value.get("event_type")),
            payload=dict(payload),
        )
    raise TypeError("runtime event must be CanonicalTraceEventRecord or dict")


def ensure_trace_seed(request: UnifiedBackendRequest):
    if getattr(request, "trace_seed", None) is not None:
        return request.trace_seed
    task_plan = getattr(request, "task_plan", None)
    bootstrap = getattr(request, "bootstrap", None)
    if task_plan is None or bootstrap is None:
        return None
    return build_canonical_trace_seed(
        motion_backend=request.motion_backend,
        bootstrap=bootstrap,
        task_plan=task_plan,
    )


def inject_runtime_defaults(
    forwarded_argv: Sequence[str],
    bootstrap,
    motion_backend: str,
    *,
    warning_scope: str = "RC5Unified",
) -> List[str]:
    forwarded = list(forwarded_argv)

    def inject(flag: str, value: Optional[str], reason: str) -> None:
        if value is None or has_flag(forwarded, flag):
            return
        emit_warning(
            warning_scope,
            f"Injecting {flag}={value} from validated config for key '{bootstrap.key}' ({reason}).",
        )
        forwarded.extend([flag, str(value)])

    inject("--key", bootstrap.key, "derived runtime key")
    inject("--robot_uids", bootstrap.robot_uids, "missing robot uid")
    inject("--sim_backend", DEFAULT_RC5_SIM_BACKEND, "default RC5 GPU physics backend")

    if motion_backend in {PROXY_BACKEND, HYBRID_BACKEND}:
        inject("--control_mode", bootstrap.control_mode, "missing proxy control mode")
        inject("--teleop_profile_config", bootstrap.teleop_profile_config, "missing teleop profile config")
        inject("--teleop_profile", bootstrap.teleop_profile, "missing teleop profile")

    return forwarded


def _inject_proxy_auto_pick_macro_from_task_plan(
    forwarded_argv: Sequence[str],
    request: UnifiedBackendRequest,
    *,
    warning_scope: str = "RC5Unified",
) -> List[str]:
    task_plan = request.task_plan
    forwarded = list(forwarded_argv)
    task_type = getattr(getattr(task_plan, "intent", None), "task_type", None)
    existing_values = extract_flag_values(forwarded, "--auto_pick_macro")
    forwarded = inject_proxy_auto_pick_macro_one(
        forwarded,
        task_plan,
        scope="Unified proxy/hybrid pick_up",
    )

    if task_type == "pick_up" and not existing_values and "--auto_pick_macro" in forwarded:
        emit_warning(
            warning_scope,
            "Injecting --auto_pick_macro=1 from unified task plan for pick_up execution.",
        )
    return forwarded


def build_backend_dispatch_plan(
    *,
    motion_backend: str,
    passthrough_argv: Sequence[str],
    bootstrap,
    task_plan=None,
    rc5_asset_dir: str | None = None,
    rc5_move_group: str | None = None,
    warning_scope: str = "RC5Unified",
) -> BackendDispatchPlan:
    backend = resolve_backend(motion_backend)
    return backend.build_dispatch_plan(
        UnifiedBackendRequest(
            motion_backend=motion_backend,
            passthrough_argv=list(passthrough_argv),
            bootstrap=bootstrap,
            task_plan=task_plan,
            trace_seed=build_canonical_trace_seed(
                motion_backend=motion_backend,
                bootstrap=bootstrap,
                task_plan=task_plan,
            )
            if task_plan is not None and bootstrap is not None
            else None,
            rc5_asset_dir=rc5_asset_dir,
            rc5_move_group=rc5_move_group,
            warning_scope=warning_scope,
        )
    )


def resolve_backend(motion_backend: str) -> DispatchBackend:
    if motion_backend not in DISPATCH_BACKENDS:
        valid = ", ".join(UNIFIED_BACKENDS)
        raise ValueError(f"Unsupported motion_backend '{motion_backend}'. Expected one of: {valid}")
    return DISPATCH_BACKENDS[motion_backend]


def execute_backend_request(request: UnifiedBackendRequest) -> UnifiedBackendResult:
    backend = resolve_backend(request.motion_backend)
    return backend.execute(request)


def _normalize_exit_code(value: Any) -> int:
    return 0 if value is None else int(value)


def apply_env_updates(env_updates: Dict[str, str]) -> None:
    for key, value in env_updates.items():
        os.environ[key] = value
        print(f"[RC5Unified] export {key}={value}")


def execute_dispatch_plan(plan: BackendDispatchPlan) -> int:
    apply_env_updates(plan.env_updates)
    module = importlib.import_module(plan.module_name)
    entrypoint = getattr(module, plan.callable_name)
    return _normalize_exit_code(entrypoint(plan.forwarded_argv))
