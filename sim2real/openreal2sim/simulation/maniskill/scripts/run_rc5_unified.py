from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
MANISKILL_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
# Host runs need `motion.*` (planner debug solver). Docker historically used PYTHONPATH=/app.
if str(MANISKILL_ROOT) not in sys.path:
    sys.path.append(str(MANISKILL_ROOT))

from openreal2sim.simulation.maniskill.scripts.rc5_unified_bootstrap import (
    detect_config_key,
    emit_warning,
    load_runner_config,
    load_simulation_config_sections,
    pick_simulation_value,
    resolve_rc5_move_group,
    resolve_unified_bootstrap_request,
    validate_rc5_bootstrap_for_backend,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_execution import (
    PLANNER_BACKEND,
    PROXY_BACKEND,
    UNIFIED_BACKENDS,
    UnifiedBackendResult,
    UnifiedBackendRequest,
    build_canonical_trace_seed,
    execute_backend_request,
    parse_requested_num_envs_from_argv,
    resolve_motion_backend_from_unified_planner_mode,
    resolve_backend,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_tasks import (
    SUPPORTED_TASK_TYPES,
    TASK_PICK_UP,
    build_task_plan,
    resolve_episode_intent,
    summarize_task_plan,
    validate_task_supported_for_runtime,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_trajectory import (
    serialize_trace_record,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    rewrite_rl4vla_raw_episode_embedded_runtime_request_json,
)
from openreal2sim.simulation.maniskill.scripts.maniskill_num_envs_policy import (
    DEFAULT_RC5_SIM_BACKEND,
)

_LEGACY_HELP = argparse.SUPPRESS
_VALID_RUN_MODES = ("episode", "collection")

_UNIFIED_ARG_TAKES_VALUE = {
    "--run_mode": True,
    "--motion_backend": True,
    "--rc5_asset_dir": True,
    "--rc5_move_group": True,
    "--task_type": True,
    "--task_destination_id": True,
    "--task_prompt": True,
    "--output_dir": True,
    "--collection_manifest": True,
    "--placement_manifest": True,
    "--placement_seed_start": True,
    "--num_episodes": True,
    "--seed_position_jitter_xy": True,
    "--dense_episode_image_width": True,
    "--dense_episode_image_height": True,
    "--runtime_sim_patch": True,
    "--exporter": True,
    "--embed_runtime_bundle_in_rl4vla_raw_npz": False,
    "--no-embed_runtime_bundle_in_rl4vla_raw_npz": False,
    "--disable_planner_proxy_adaptive_steps": False,
    "--save_debug_video": False,
    "--save_debug_gif": False,
    "--stop_on_failure": False,
}
class _TeeStream:
    def __init__(
        self,
        *streams,
        timestamp_fn: Callable[[], datetime] | None = None,
    ):
        self._streams = streams
        self._timestamp_fn = timestamp_fn or datetime.now
        self._at_line_start = True

    def _prefix(self) -> str:
        return self._timestamp_fn().strftime("[%Y-%m-%d %H:%M:%S] ")

    def write(self, data):
        if not data:
            return 0
        rendered_parts: list[str] = []
        idx = 0
        while idx < len(data):
            if self._at_line_start:
                rendered_parts.append(self._prefix())
                self._at_line_start = False
            newline_idx = data.find("\n", idx)
            if newline_idx == -1:
                rendered_parts.append(data[idx:])
                break
            rendered_parts.append(data[idx : newline_idx + 1])
            self._at_line_start = True
            idx = newline_idx + 1
        rendered = "".join(rendered_parts)
        for stream in self._streams:
            stream.write(rendered)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@dataclass(frozen=True)
class EpisodePlacementSpec:
    episode_id: str
    episode_index: int
    object_id: str
    task_semantic_name: str
    placement_source: str
    placement_seed: int | None
    placement: Dict[str, Any]


@dataclass(frozen=True)
class CollectionRequest:
    motion_backend: str
    task_type: str
    config_path: Path
    key: str
    scene_path: str
    output_dir: Path
    num_envs: int
    headless: bool
    task_object_id: str
    task_semantic_name: str
    dense_episode_image_width: int
    dense_episode_image_height: int
    save_debug_video: bool
    save_debug_gif: bool
    embed_runtime_bundle_in_rl4vla_raw_npz: bool
    runtime_sim_patch_path: Path | None
    exporter: str | None
    stop_on_failure: bool
    passthrough_argv: List[str]
    episodes: tuple[EpisodePlacementSpec, ...]


@dataclass(frozen=True)
class EpisodeRuntimeInput:
    episode_id: str
    episode_index: int
    config_path: Path
    request_path: Path
    runner_argv: tuple[str, ...]


@dataclass(frozen=True)
class PreparedExecutionBatch:
    batch_id: str
    compatibility_group_id: str
    env_count: int
    runtime_inputs: tuple[EpisodeRuntimeInput, ...]
    runner_argv: tuple[str, ...]
    batch_config_path: Path
    dense_batch_output_dir: Path | None = None
    rl4vla_raw_batch_output_dir: Path | None = None


@dataclass(frozen=True)
class EpisodeExecutionResult:
    episode_id: str
    episode_index: int
    request_path: Path
    exit_code: int
    runtime_exit_code: int | None
    execution_outcome: str
    success: bool
    semantic_task_success: bool | None = None
    failed_stage: str | None = None
    trace_record: Any = None
    result_path: Path | None = None
    export_manifest_path: Path | None = None
    dense_episode_artifact_path: Path | None = None
    rl4vla_raw_episode_artifact_path: Path | None = None
    debug_video_path: Path | None = None
    debug_video_gif_path: Path | None = None
    batch_id: str | None = None
    compatibility_group_id: str | None = None
    env_index: int | None = None


SEMANTIC_FAILURE_EXIT_CODE = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RC5 proxy planner prerelease runner. "
            "This thin entrypoint dispatches the validated proxy planner runtime and replay-compatible artifact flow."
        )
    )
    parser.add_argument(
        "--run_mode",
        default="episode",
        metavar="MODE",
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--motion_backend",
        required=False,
        default=None,
        metavar="BACKEND",
        help=(
            "Motion backend override. Prerelease public mode is 'proxy_ee_delta'. "
            "If omitted, local.<key>.simulation.unified_planner_backend is used."
        ),
    )
    parser.add_argument(
        "--rc5_asset_dir",
        default=None,
        help=(
            "Optional RC5 asset directory override. "
            "When provided, exported to RC5_AERO_HAND_ASSET_DIR before dispatch."
        ),
    )
    parser.add_argument(
        "--rc5_move_group",
        default=None,
        help=(
            "Optional RC5 move-group override. "
            "When provided, exported to OPENR2S_RC5_MOVE_GROUP before dispatch."
        ),
    )
    parser.add_argument(
        "--task_type",
        default="pick_up",
        metavar="TASK_TYPE",
        help=(
            "High-level task intent. Prerelease public mode is 'pick_up'."
        ),
    )
    parser.add_argument(
        "--task_object_id",
        default=None,
        help="Optional explicit task object id for the unified task layer.",
    )
    parser.add_argument(
        "--task_destination_id",
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--task_prompt",
        default=None,
        help="Optional natural-language task prompt stored in the unified task layer.",
    )
    parser.add_argument(
        "--disable_planner_proxy_adaptive_steps",
        action="store_true",
        help=(
            "Disable adaptive proxy step reduction for both XY approach and Z descent. "
            "Adaptive proxy stepping is enabled by default."
        ),
    )
    parser.add_argument(
        "--config_path",
        default="config/config_debug.yaml",
        help="Runner config used as the source of key/bootstrap/object placement defaults.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Explicit config key. If omitted, derived from --scene when possible.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene path forwarded to the runtime.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without a viewer.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Requested runtime batch size.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--collection_manifest",
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--placement_manifest",
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--placement_seed_start",
        type=int,
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--seed_position_jitter_xy",
        type=float,
        default=0.05,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--dense_episode_image_width",
        type=int,
        default=640,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--dense_episode_image_height",
        type=int,
        default=480,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--save_debug_video",
        action="store_true",
        default=False,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--save_debug_gif",
        action="store_true",
        default=False,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--embed_runtime_bundle_in_rl4vla_raw_npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Embed episode-specific runtime_config.yaml and runtime_request.json into "
            "rl4vla_raw_episode*.npz for experimenter-friendly self-contained proxy planner artifacts."
        ),
    )
    parser.add_argument(
        "--runtime_sim_patch",
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--exporter",
        default=None,
        help=_LEGACY_HELP,
    )
    parser.add_argument(
        "--stop_on_failure",
        action="store_true",
        default=False,
        help=_LEGACY_HELP,
    )
    return parser


def _validate_parser_surface_args(args: argparse.Namespace) -> None:
    if str(args.run_mode) not in _VALID_RUN_MODES:
        supported = ", ".join(_VALID_RUN_MODES)
        raise ValueError(f"Unsupported run_mode: {args.run_mode!r}. Supported values: {supported}.")
    if args.motion_backend is not None and str(args.motion_backend) not in UNIFIED_BACKENDS:
        supported = ", ".join(UNIFIED_BACKENDS)
        raise ValueError(
            f"Unsupported motion_backend: {args.motion_backend!r}. Supported values: {supported}."
        )
    if str(args.task_type) not in SUPPORTED_TASK_TYPES:
        supported = ", ".join(SUPPORTED_TASK_TYPES)
        raise ValueError(
            f"Unsupported task_type: {args.task_type!r}. Supported values: {supported}."
        )


def _strip_unified_args(argv: Sequence[str]) -> List[str]:
    forwarded: List[str] = []
    idx = 0
    argv = list(argv)
    while idx < len(argv):
        token = argv[idx]
        if token in _UNIFIED_ARG_TAKES_VALUE:
            if _UNIFIED_ARG_TAKES_VALUE[token]:
                idx += 2
            else:
                idx += 1
            continue
        forwarded.append(token)
        idx += 1
    return forwarded


def _append_singleton_passthrough_flags(
    source_argv: Sequence[str],
    passthrough_argv: Sequence[str],
) -> List[str]:
    augmented = list(passthrough_argv)
    if _has_flag(source_argv, "--embed_runtime_bundle_in_rl4vla_raw_npz"):
        augmented.append("--embed_runtime_bundle_in_rl4vla_raw_npz")
    if _has_flag(source_argv, "--no-embed_runtime_bundle_in_rl4vla_raw_npz"):
        augmented.append("--no-embed_runtime_bundle_in_rl4vla_raw_npz")
    return augmented


def _extract_last_flag_value(argv: Sequence[str], flag: str) -> str | None:
    values: list[str] = []
    argv = list(argv)
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token != flag:
            idx += 1
            continue
        if idx + 1 >= len(argv):
            raise ValueError(f"{flag} requires a value")
        values.append(str(argv[idx + 1]))
        idx += 2
    return None if not values else values[-1]


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return flag in list(argv)


def _sanitize_artifact_token(value: str | None, *, default: str) -> str:
    token = "".join(ch if str(ch).isalnum() or str(ch) in {"-", "_"} else "_" for ch in str(value or "").strip())
    token = token.strip("_")
    return token or default


def _resolve_auto_run_dir(argv: Sequence[str], args: argparse.Namespace) -> Path:
    key_token = _sanitize_artifact_token(_extract_last_flag_value(argv, "--key"), default="no_key")
    task_token = _sanitize_artifact_token(args.task_type, default="task")
    object_token = _sanitize_artifact_token(args.task_object_id, default="object")
    backend_token = _sanitize_artifact_token(args.motion_backend, default="backend")
    mode_token = "headless" if _has_flag(argv, "--headless") else "viewer"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "manual" / f"{backend_token}_{mode_token}_{key_token}_{task_token}_{object_token}_{timestamp}"
    return run_dir.resolve()


def _resolve_effective_motion_backend(
    args: argparse.Namespace,
    *,
    config_path: str | Path,
    key: str | None,
    warning_scope: str = "RC5Unified",
) -> str:
    requested_motion_backend = None if args.motion_backend is None else str(args.motion_backend)
    if not key:
        if requested_motion_backend is None:
            raise ValueError(
                "run_rc5_unified.py requires either --motion_backend or a resolvable config key "
                "with local.<key>.simulation.unified_planner_backend."
            )
        return requested_motion_backend

    sections = load_simulation_config_sections(config_path, key)
    unified_planner_backend = pick_simulation_value(sections, "unified_planner_backend", None)
    if unified_planner_backend is None:
        if requested_motion_backend is None:
            raise ValueError(
                f"Neither --motion_backend nor local.{key}.simulation.unified_planner_backend is set. "
                "Provide one of them."
            )
        return requested_motion_backend

    effective_motion_backend = resolve_motion_backend_from_unified_planner_mode(unified_planner_backend)
    if requested_motion_backend is None:
        emit_warning(
            warning_scope,
            "Using motion_backend resolved from unified_planner_backend="
            f"{unified_planner_backend} -> effective motion_backend={effective_motion_backend}.",
        )
    elif effective_motion_backend != requested_motion_backend:
        emit_warning(
            warning_scope,
            "Overriding --motion_backend="
            f"{requested_motion_backend} with unified_planner_backend={unified_planner_backend} "
            f"-> effective motion_backend={effective_motion_backend}.",
        )
    return effective_motion_backend


def _append_default_artifact_args(
    argv: Sequence[str],
    run_dir: Path,
    *,
    num_envs: int,
) -> List[str]:
    augmented = list(argv)
    if int(num_envs) == 1:
        if not _has_flag(augmented, "--rl4vla_raw_episode_output"):
            augmented.extend(["--rl4vla_raw_episode_output", str(run_dir / "rl4vla_raw_episode.npz")])
        should_save_gif = _has_flag(augmented, "--save_debug_gif") or not _has_flag(augmented, "--headless")
        if should_save_gif and not _has_flag(augmented, "--save_video_gif_path"):
            augmented.extend(
                [
                    "--save_video_gif_on_exit",
                    "--save_video_gif_path",
                    str(run_dir / "debug_video.gif"),
                ]
            )
    return augmented


def _resolve_effective_task_object_id(
    *,
    explicit_object_id: str | None,
    config_path: str | Path,
    key: str | None,
) -> str | None:
    if explicit_object_id is not None:
        return str(explicit_object_id)
    if not key:
        return None
    sections = load_simulation_config_sections(config_path, key)
    resolved = pick_simulation_value(sections, "manip_object_id", None)
    return None if resolved is None else str(resolved)


def _build_execution_summary_payload(
    *,
    effective_argv: Sequence[str],
    args: argparse.Namespace,
    result: UnifiedBackendResult | None,
    task_plan,
    run_dir: Path,
    artifact_paths: dict[str, str | None] | None = None,
    failure: str | None = None,
):
    plan = None if result is None else result.dispatch_plan
    trace_seed = None if result is None else result.trace_seed
    trace_record = None if result is None else result.trace_record
    batch_summary = _build_batch_execution_summary(
        effective_argv=effective_argv,
        result=result,
    )
    return {
        "motion_backend": (
            args.motion_backend
            if args.motion_backend is not None
            else (None if result is None else getattr(result, "motion_backend", None))
        ),
        "task_type": args.task_type,
        "task_object_id": args.task_object_id,
        "task_destination_id": args.task_destination_id,
        "task_prompt": args.task_prompt,
        "argv": list(effective_argv),
        "run_dir": str(run_dir),
        "result": {
            "exit_code": None if result is None else int(result.exit_code),
            "failure": failure,
        },
        "artifacts": dict(artifact_paths or {}),
        "dispatch_plan": None
        if plan is None
        else {
            "module_name": plan.module_name,
            "callable_name": plan.callable_name,
            "forwarded_argv": list(plan.forwarded_argv),
            "env_updates": dict(plan.env_updates),
        },
        "trace_seed": None if trace_seed is None else getattr(trace_seed, "__dict__", trace_seed),
        "trace_record": serialize_trace_record(trace_record),
        "batch": batch_summary,
        "task_plan_summary": summarize_task_plan(task_plan),
    }


def _write_execution_summary(
    *,
    summary_path: Path,
    effective_argv: Sequence[str],
    args: argparse.Namespace,
    result: UnifiedBackendResult | None,
    task_plan,
    run_dir: Path,
    artifact_paths: dict[str, str | None] | None = None,
    failure: str | None = None,
) -> Path:
    payload = _build_execution_summary_payload(
        effective_argv=effective_argv,
        args=args,
        result=result,
        task_plan=task_plan,
        run_dir=run_dir,
        artifact_paths=artifact_paths,
        failure=failure,
    )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _resolve_result_outcome_token(result: UnifiedBackendResult | None) -> str:
    if result is None:
        return "fail"
    trace_record = getattr(result, "trace_record", None)
    if isinstance(trace_record, Mapping):
        execution_outcome = trace_record.get("execution_outcome")
    else:
        execution_outcome = getattr(trace_record, "execution_outcome", None)
    if execution_outcome == "success" and int(getattr(result, "exit_code", 1)) == 0:
        return "success"
    return "fail"


def _extract_trace_events(trace_record) -> list:
    if trace_record is None:
        return []
    if isinstance(trace_record, dict):
        return list(trace_record.get("events", []) or [])
    return list(getattr(trace_record, "events", []) or [])


def _extract_batch_runtime_feedback_payload(trace_record) -> dict | None:
    for event in reversed(_extract_trace_events(trace_record)):
        if isinstance(event, dict):
            event_type = event.get("event_type")
            payload = event.get("payload", {})
        else:
            event_type = getattr(event, "event_type", None)
            payload = getattr(event, "payload", {})
        if event_type == "batch_runtime_feedback" and isinstance(payload, dict):
            return dict(payload)
    return None


def _build_batch_execution_summary(
    *,
    effective_argv: Sequence[str],
    result: UnifiedBackendResult | None,
) -> dict:
    requested_num_envs = parse_requested_num_envs_from_argv(_strip_unified_args(effective_argv))
    trace_record = None if result is None else getattr(result, "trace_record", None)
    batch_trace = None if trace_record is None else getattr(trace_record, "batch_trace", None)
    runtime_payload = _extract_batch_runtime_feedback_payload(trace_record)
    runtime_payload_present = runtime_payload is not None
    if batch_trace is not None:
        runtime_payload_present = bool(getattr(batch_trace, "runtime_batch_feedback_present", runtime_payload_present))
    return {
        "requested_num_envs": int(requested_num_envs),
        "batched_request": int(requested_num_envs) > 1,
        "runtime_batch_feedback_present": runtime_payload_present,
        "runtime_batch_size": (
            None
            if batch_trace is None
            else getattr(batch_trace, "runtime_batch_size", None)
        ),
        "successful_env_count": (
            None
            if batch_trace is None
            else getattr(batch_trace, "successful_env_count", None)
        ),
        "failed_env_indices": (
            []
            if batch_trace is None
            else list(getattr(batch_trace, "failed_env_indices", ()) or ())
        ),
        "artifacts_recorded_per_env": (
            None
            if batch_trace is None
            else getattr(batch_trace, "artifacts_recorded_per_env", None)
        ),
        "per_env_feedback_count": (
            0
            if batch_trace is None
            else len(list(getattr(batch_trace, "per_env_records", ()) or ()))
        ),
        "auto_dense_episode_enabled": int(requested_num_envs) == 1,
    }


def _rename_path_if_exists(path: Path, target: Path) -> Path | None:
    if not path.exists():
        return None
    path.rename(target)
    return target


def _finalize_artifact_path(path: Path | None, *, outcome_token: str) -> Path | None:
    if path is None or not path.exists():
        return None
    if path.stem.endswith("_success") or path.stem.endswith("_fail"):
        return path
    target = path.with_name(f"{path.stem}_{outcome_token}{path.suffix}")
    return _rename_path_if_exists(path, target)


def _rewrite_exact_string_values(value: Any, *, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_rewrite_exact_string_values(item, replacements=replacements) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _rewrite_exact_string_values(item, replacements=replacements)
            for key, item in value.items()
        }
    return value


def _resolve_singleton_requested_artifact_path(
    effective_argv: Sequence[str],
    *,
    flag: str,
    default_path: Path | None = None,
) -> Path | None:
    raw_value = _extract_last_flag_value(_strip_unified_args(effective_argv), flag)
    if raw_value is None:
        return default_path
    return Path(str(raw_value)).expanduser().resolve()


def _build_default_run_instruction(
    *,
    task_type: str,
    task_semantic_name: str,
    task_object_id: str | None = None,
) -> str:
    if task_type == TASK_PICK_UP:
        if str(task_object_id or "") == "orange_cube_ext":
            return "Pick red cube"
        return f"Pick up {task_semantic_name}."
    return f"{task_type}:{task_semantic_name}"


def _materialize_default_runtime_request(
    effective_argv: Sequence[str],
    *,
    run_dir: Path,
    bootstrap_request: UnifiedBootstrapRequest,
    args: argparse.Namespace,
) -> tuple[list[str], Path | None]:
    if not bool(getattr(args, "embed_runtime_bundle_in_rl4vla_raw_npz", False)):
        return list(effective_argv), None
    if _extract_last_flag_value(_strip_unified_args(effective_argv), "--runtime_request_path") is not None:
        request_path = Path(
            str(_extract_last_flag_value(_strip_unified_args(effective_argv), "--runtime_request_path"))
        ).expanduser().resolve()
        return list(effective_argv), request_path

    rl4vla_raw_path = _resolve_singleton_requested_artifact_path(
        effective_argv,
        flag="--rl4vla_raw_episode_output",
        default_path=None,
    )
    if rl4vla_raw_path is None:
        return list(effective_argv), None

    runtime_config_path = (run_dir / "runtime_config.yaml").resolve()
    runtime_config_path.write_text(
        Path(bootstrap_request.config_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    task_semantic_name = str(args.task_object_id)
    try:
        task_semantic_name = _resolve_task_semantic_name_from_object_cfg(
            _resolve_base_object_placement(
                bootstrap_request.config_path,
                bootstrap_request.key,
                str(args.task_object_id),
            ),
            object_id=str(args.task_object_id),
        )
    except Exception as exc:
        emit_warning(
            "RC5Unified",
            "Failed to resolve task_semantic_name for default runtime request materialization; "
            f"falling back to object id '{args.task_object_id}': {type(exc).__name__}: {exc}",
        )

    request_path = (run_dir / "runtime_request.json").resolve()
    payload = {
        "episode_id": run_dir.name,
        "episode_index": 0,
        "motion_backend": args.motion_backend,
        "task_type": args.task_type,
        "task_object_id": args.task_object_id,
        "task_semantic_name": task_semantic_name,
        "scene_path": bootstrap_request.scene_path,
        "key": bootstrap_request.key,
        "runtime_config_path": str(runtime_config_path),
        "dense_episode_artifact_path": (
            None
            if (dense_path := _resolve_singleton_requested_artifact_path(
                effective_argv,
                flag="--dense_episode_output",
                default_path=None,
            ))
            is None
            else str(dense_path)
        ),
        "rl4vla_raw_episode_artifact_path": str(rl4vla_raw_path),
        "dense_episode_instruction": _build_default_run_instruction(
            task_type=str(args.task_type),
            task_semantic_name=task_semantic_name,
            task_object_id=str(args.task_object_id) if args.task_object_id else None,
        ),
        "dense_episode_image_width": int(args.dense_episode_image_width),
        "dense_episode_image_height": int(args.dense_episode_image_height),
        "debug_video_path": (
            None
            if (debug_video_path := _resolve_singleton_requested_artifact_path(
                effective_argv,
                flag="--save_video_path",
                default_path=None,
            ))
            is None
            else str(debug_video_path)
        ),
        "debug_video_gif_path": (
            None
            if (debug_video_gif_path := _resolve_singleton_requested_artifact_path(
                effective_argv,
                flag="--save_video_gif_path",
                default_path=None,
            ))
            is None
            else str(debug_video_gif_path)
        ),
        "runner_argv": [*list(effective_argv), "--runtime_request_path", str(request_path)],
    }
    request_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return [*list(effective_argv), "--runtime_request_path", str(request_path)], request_path


def _finalize_default_runtime_request_artifacts(
    request_path: Path,
    *,
    requested_rl4vla_raw_episode_artifact_path: Path | None,
    finalized_rl4vla_raw_episode_artifact_path: Path | None,
    requested_debug_video_path: Path | None,
    finalized_debug_video_path: Path | None,
    requested_debug_video_gif_path: Path | None,
    finalized_debug_video_gif_path: Path | None,
) -> Dict[str, Any]:
    runtime_request = _load_runtime_request_payload(request_path)
    replacements: Dict[str, str] = {}
    if (
        requested_rl4vla_raw_episode_artifact_path is not None
        and finalized_rl4vla_raw_episode_artifact_path is not None
    ):
        replacements[str(requested_rl4vla_raw_episode_artifact_path)] = str(
            finalized_rl4vla_raw_episode_artifact_path
        )
    if requested_debug_video_path is not None and finalized_debug_video_path is not None:
        replacements[str(requested_debug_video_path)] = str(finalized_debug_video_path)
    if requested_debug_video_gif_path is not None and finalized_debug_video_gif_path is not None:
        replacements[str(requested_debug_video_gif_path)] = str(finalized_debug_video_gif_path)
    if replacements:
        runtime_request = _rewrite_exact_string_values(runtime_request, replacements=replacements)
    runtime_request["rl4vla_raw_episode_artifact_path"] = (
        None
        if finalized_rl4vla_raw_episode_artifact_path is None
        else str(finalized_rl4vla_raw_episode_artifact_path)
    )
    runtime_request["debug_video_path"] = (
        None if finalized_debug_video_path is None else str(finalized_debug_video_path)
    )
    runtime_request["debug_video_gif_path"] = (
        None if finalized_debug_video_gif_path is None else str(finalized_debug_video_gif_path)
    )
    request_path.write_text(
        json.dumps(runtime_request, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if finalized_rl4vla_raw_episode_artifact_path is not None:
        rewrite_rl4vla_raw_episode_embedded_runtime_request_json(
            finalized_rl4vla_raw_episode_artifact_path,
            runtime_request_json=json.dumps(runtime_request, indent=2, sort_keys=True),
        )
    return runtime_request


def _finalize_default_artifact_paths(
    run_dir: Path,
    outcome_token: str,
    *,
    effective_argv: Sequence[str],
) -> dict[str, str | None]:
    final_paths: dict[str, str | None] = {}
    dense_requested = _resolve_singleton_requested_artifact_path(
        effective_argv,
        flag="--dense_episode_output",
        default_path=None,
    )
    dense_final = _finalize_artifact_path(dense_requested, outcome_token=outcome_token)
    final_paths["dense_episode_path"] = None if dense_final is None else str(dense_final)
    rl4vla_raw_requested = _resolve_singleton_requested_artifact_path(
        effective_argv,
        flag="--rl4vla_raw_episode_output",
        default_path=None,
    )
    rl4vla_raw_final = _finalize_artifact_path(
        rl4vla_raw_requested,
        outcome_token=outcome_token,
    )
    final_paths["rl4vla_raw_episode_path"] = (
        None if rl4vla_raw_final is None else str(rl4vla_raw_final)
    )

    video_final = None
    for candidate in sorted(run_dir.glob("debug_video.*")):
        if candidate.suffix.lower() == ".gif":
            continue
        video_final = _finalize_artifact_path(candidate, outcome_token=outcome_token)
        if video_final is not None:
            break
    final_paths["debug_video_path"] = None if video_final is None else str(video_final)

    gif_final = _finalize_artifact_path(run_dir / "debug_video.gif", outcome_token=outcome_token)
    final_paths["debug_video_gif_path"] = None if gif_final is None else str(gif_final)
    final_paths["run_log_path"] = str(run_dir / "run.log")
    final_paths["execution_summary_path"] = str(run_dir / "execution_summary.json")
    return final_paths


def _load_manifest(path_str: str | Path) -> List[Dict[str, Any]]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Collection manifest does not exist: {path}")
    data = load_runner_config(path)
    if "episodes" not in data:
        raise ValueError(f"Collection manifest must define a top-level 'episodes' list: {path}")
    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Collection manifest 'episodes' must be a non-empty list: {path}")
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(episodes):
        if not isinstance(item, dict):
            raise ValueError(f"Collection manifest episode #{idx} must be a mapping: {path}")
        normalized.append(dict(item))
    return normalized


def _require_mapping(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _require_position_triplet(value: Any, *, label: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must define exactly 3 numeric values")
    return [float(item) for item in value]


def _require_orientation_quat(value: Any, *, label: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must define exactly 4 numeric values in wxyz order")
    return [float(item) for item in value]


def _resolve_task_semantic_name_from_object_cfg(
    object_cfg: Mapping[str, Any],
    *,
    object_id: str,
) -> str:
    semantic_name = object_cfg.get("task_semantic_name")
    if isinstance(semantic_name, str) and semantic_name.strip():
        return str(semantic_name).strip()
    display_name = object_cfg.get("name")
    if isinstance(display_name, str) and display_name.strip():
        emit_warning(
            "RC5Unified",
            "Missing task_semantic_name for object "
            f"{object_id}; falling back to object name '{str(display_name).strip().replace('_', ' ')}'.",
        )
        return str(display_name).strip().replace("_", " ")
    emit_warning(
        "RC5Unified",
        f"Missing task_semantic_name and name for object {object_id}; falling back to technical object id.",
    )
    return str(object_id)


def _resolve_collection_manifest_arg(args: argparse.Namespace) -> str | None:
    manifest = args.collection_manifest
    legacy_manifest = args.placement_manifest
    if manifest and legacy_manifest and manifest != legacy_manifest:
        raise ValueError("Use exactly one of --collection_manifest or --placement_manifest.")
    if manifest is None and legacy_manifest is not None:
        emit_warning(
            "RC5Unified",
            "--placement_manifest is deprecated; use --collection_manifest instead.",
        )
        manifest = legacy_manifest
    return manifest


def _resolve_base_object_placement(
    config_path: str | Path,
    key: str,
    object_id: str,
) -> Dict[str, Any]:
    sections = load_simulation_config_sections(config_path, key)
    object_placements = pick_simulation_value(sections, "object_placements", None)
    if not isinstance(object_placements, dict) or not object_placements:
        raise ValueError(
            f"Collection mode requires object_placements in config for key '{key}' to resolve base placements."
        )
    if object_id not in object_placements:
        available = ", ".join(sorted(object_placements.keys()))
        raise KeyError(
            f"Object placement for '{object_id}' not found under key '{key}'. Available objects: {available}"
        )
    base_cfg = _require_mapping(
        object_placements[object_id],
        label=f"object_placements.{object_id}",
    )
    base_cfg["position"] = _require_position_triplet(
        base_cfg.get("position"),
        label=f"object_placements.{object_id}.position",
    )
    base_cfg["orientation"] = _require_orientation_quat(
        base_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]),
        label=f"object_placements.{object_id}.orientation",
    )
    return base_cfg


def _build_seeded_placement(
    base_cfg: Mapping[str, Any],
    *,
    seed: int,
    jitter_xy: float,
) -> Dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(int(seed))
    base_position = _require_position_triplet(base_cfg.get("position"), label="base position")
    dx = float(rng.uniform(-jitter_xy, jitter_xy))
    dy = float(rng.uniform(-jitter_xy, jitter_xy))
    placement = dict(base_cfg)
    placement["position"] = [
        round(base_position[0] + dx, 6),
        round(base_position[1] + dy, 6),
        round(base_position[2], 6),
    ]
    placement["orientation"] = _require_orientation_quat(
        placement.get("orientation"),
        label="base orientation",
    )
    return placement


def _merge_manifest_episode_placement(
    base_cfg: Mapping[str, Any],
    episode_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    explicit = _require_mapping(
        episode_cfg.get("placement"),
        label="manifest episode placement",
    )
    merged = dict(base_cfg)
    merged.update(explicit)
    merged["position"] = _require_position_triplet(
        merged.get("position"),
        label="manifest episode placement.position",
    )
    merged["orientation"] = _require_orientation_quat(
        merged.get("orientation", [1.0, 0.0, 0.0, 0.0]),
        label="manifest episode placement.orientation",
    )
    return merged


def _resolve_episode_specs(
    args: argparse.Namespace,
    *,
    config_path: Path,
    key: str,
    task_object_id: str,
) -> tuple[EpisodePlacementSpec, ...]:
    manifest_path = _resolve_collection_manifest_arg(args)
    if manifest_path and args.placement_seed_start is not None:
        raise ValueError(
            "Choose exactly one placement source: --collection_manifest/--placement_manifest or "
            "--placement_seed_start/--num_episodes."
        )

    if manifest_path:
        manifest_episodes = _load_manifest(manifest_path)
        resolved: List[EpisodePlacementSpec] = []
        for idx, item in enumerate(manifest_episodes):
            object_id = str(item.get("object_id", task_object_id))
            base_object_cfg = _resolve_base_object_placement(config_path, key, object_id)
            resolved.append(
                EpisodePlacementSpec(
                    episode_id=f"episode_{idx:06d}",
                    episode_index=idx,
                    object_id=object_id,
                    task_semantic_name=_resolve_task_semantic_name_from_object_cfg(
                        base_object_cfg,
                        object_id=object_id,
                    ),
                    placement_source="manifest",
                    placement_seed=int(item["placement_seed"]) if "placement_seed" in item else None,
                    placement=_merge_manifest_episode_placement(base_object_cfg, item),
                )
            )
        return tuple(resolved)

    if args.placement_seed_start is None:
        raise ValueError(
            "Collection mode requires a placement source. Provide --collection_manifest or "
            "--placement_seed_start together with --num_episodes."
        )
    if args.num_episodes is None:
        raise ValueError("--num_episodes is required when --placement_seed_start is used.")
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be >= 1")
    if args.seed_position_jitter_xy < 0:
        raise ValueError("--seed_position_jitter_xy must be >= 0")

    base_object_cfg = _resolve_base_object_placement(config_path, key, task_object_id)
    episodes: List[EpisodePlacementSpec] = []
    for idx in range(int(args.num_episodes)):
        seed = int(args.placement_seed_start) + idx
        episodes.append(
            EpisodePlacementSpec(
                episode_id=f"episode_{idx:06d}",
                episode_index=idx,
                object_id=task_object_id,
                task_semantic_name=_resolve_task_semantic_name_from_object_cfg(
                    base_object_cfg,
                    object_id=task_object_id,
                ),
                placement_source="seeded_fixed",
                placement_seed=seed,
                placement=_build_seeded_placement(
                    base_object_cfg,
                    seed=seed,
                    jitter_xy=float(args.seed_position_jitter_xy),
                ),
            )
        )
    return tuple(episodes)


def _build_collection_bootstrap_passthrough(
    args: argparse.Namespace,
    passthrough_argv: Sequence[str],
) -> List[str]:
    argv = list(passthrough_argv)
    if not _has_flag(argv, "--sim_backend"):
        argv.extend(["--sim_backend", DEFAULT_RC5_SIM_BACKEND])
    argv.extend(["--config_path", str(args.config_path)])
    if args.scene is not None:
        argv.extend(["--scene", str(args.scene)])
    if args.key is not None:
        argv.extend(["--key", str(args.key)])
    return argv


def resolve_collection_request(argv: Optional[Sequence[str]] = None) -> CollectionRequest:
    parser = _build_parser()
    args, passthrough_argv = parser.parse_known_args(list(argv or ()))
    _validate_parser_surface_args(args)
    if args.run_mode != "collection":
        raise ValueError("resolve_collection_request requires --run_mode=collection.")
    bootstrap_argv = _build_collection_bootstrap_passthrough(args, passthrough_argv)
    bootstrap_request = resolve_unified_bootstrap_request(
        bootstrap_argv,
        default_config_path=args.config_path,
    )
    effective_motion_backend = _resolve_effective_motion_backend(
        args,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
    )
    if effective_motion_backend != PROXY_BACKEND:
        raise ValueError(
            "Collection mode currently supports only motion_backend='proxy_ee_delta'. "
            f"Received '{effective_motion_backend}'."
        )
    if args.task_type != TASK_PICK_UP:
        raise ValueError(
            "Collection mode currently supports only task_type='pick_up'. "
            f"Received '{args.task_type}'."
        )
    if not args.output_dir:
        raise ValueError("--output_dir is required when --run_mode=collection.")
    if int(_extract_last_flag_value(passthrough_argv, "--num_envs") or 1) > 1:
        raise ValueError("Collection mode passthrough argv must not override --num_envs; use the unified flag instead.")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be >= 1")
    if args.dense_episode_image_width <= 0 or args.dense_episode_image_height <= 0:
        raise ValueError("--dense_episode_image_width and --dense_episode_image_height must be >= 1")
    if args.headless and _has_flag(passthrough_argv, "--step_by_step"):
        raise ValueError("--headless is incompatible with --step_by_step in collection passthrough args.")
    if args.num_envs > 1 and (args.save_debug_video or args.save_debug_gif):
        raise ValueError(
            "Collection mode does not support shared debug-video artifacts for num_envs > 1. "
            "Use singleton collection or disable --save_debug_video/--save_debug_gif."
        )

    if bootstrap_request.scene_path is None:
        derived_key = detect_config_key(args.scene, args.key)
        raise ValueError(
            "Collection mode requires --scene and a resolvable config key. "
            f"Received scene={args.scene!r}, key={args.key!r}, derived_key={derived_key!r}."
        )
    validate_rc5_bootstrap_for_backend(bootstrap_request.bootstrap, effective_motion_backend)

    sections = load_simulation_config_sections(
        bootstrap_request.config_path,
        bootstrap_request.key,
    )
    task_object_id = args.task_object_id or pick_simulation_value(sections, "manip_object_id", None)
    if not task_object_id:
        raise ValueError(
            f"Collection mode requires --task_object_id or config manip_object_id for key '{bootstrap_request.key}'."
        )

    episodes = _resolve_episode_specs(
        args,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
        task_object_id=str(task_object_id),
    )

    return CollectionRequest(
        motion_backend=effective_motion_backend,
        task_type=args.task_type,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
        scene_path=str(bootstrap_request.scene_path),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        num_envs=int(args.num_envs),
        headless=bool(args.headless),
        task_object_id=str(task_object_id),
        task_semantic_name=_resolve_task_semantic_name_from_object_cfg(
            _resolve_base_object_placement(
                bootstrap_request.config_path,
                bootstrap_request.key,
                str(task_object_id),
            ),
            object_id=str(task_object_id),
        ),
        dense_episode_image_width=int(args.dense_episode_image_width),
        dense_episode_image_height=int(args.dense_episode_image_height),
        save_debug_video=bool(args.save_debug_video),
        save_debug_gif=bool(args.save_debug_gif),
        embed_runtime_bundle_in_rl4vla_raw_npz=bool(args.embed_runtime_bundle_in_rl4vla_raw_npz),
        runtime_sim_patch_path=(
            None if args.runtime_sim_patch is None else Path(args.runtime_sim_patch).expanduser().resolve()
        ),
        exporter=args.exporter,
        stop_on_failure=bool(args.stop_on_failure),
        passthrough_argv=list(passthrough_argv),
        episodes=episodes,
    )


def write_collection_plan(request: CollectionRequest) -> Path:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = request.output_dir / "collection_plan.json"
    payload = {
        "run_mode": "collection",
        "motion_backend": request.motion_backend,
        "task_type": request.task_type,
        "config_path": str(request.config_path),
        "key": request.key,
        "scene_path": request.scene_path,
        "output_dir": str(request.output_dir),
        "num_envs": request.num_envs,
        "headless": request.headless,
        "task_object_id": request.task_object_id,
        "task_semantic_name": request.task_semantic_name,
        "dense_episode_image_width": request.dense_episode_image_width,
        "dense_episode_image_height": request.dense_episode_image_height,
        "save_debug_video": request.save_debug_video,
        "save_debug_gif": request.save_debug_gif,
        "embed_runtime_bundle_in_rl4vla_raw_npz": request.embed_runtime_bundle_in_rl4vla_raw_npz,
        "runtime_sim_patch_path": (
            None if request.runtime_sim_patch_path is None else str(request.runtime_sim_patch_path)
        ),
        "stop_on_failure": request.stop_on_failure,
        "episode_count": len(request.episodes),
        "episodes": [asdict(episode) for episode in request.episodes],
    }
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return plan_path


def _load_runtime_sim_patch(path: Path) -> Dict[str, Any]:
    patch_path = Path(path).expanduser().resolve()
    if not patch_path.exists():
        raise FileNotFoundError(f"Runtime sim patch does not exist: {patch_path}")
    payload = load_runner_config(patch_path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Runtime sim patch must be a non-empty mapping: {patch_path}")
    return dict(payload)


def _deep_merge_mapping(base: Dict[str, Any], patch: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping):
            existing = base.get(key)
            if existing is None:
                base[key] = dict(value)
                continue
            if not isinstance(existing, dict):
                raise ValueError(f"{label}.{key} cannot merge mapping into non-mapping value")
            _deep_merge_mapping(existing, value, label=f"{label}.{key}")
            continue
        base[key] = value
    return base


def _apply_runtime_sim_patch(runtime_sim: Dict[str, Any], *, patch_path: Path) -> None:
    patch_payload = _load_runtime_sim_patch(patch_path)
    _deep_merge_mapping(runtime_sim, patch_payload, label="runtime_sim_patch")


def _clone_runtime_config_with_episode_placement(
    request: CollectionRequest,
    episode: EpisodePlacementSpec,
) -> Dict[str, Any]:
    raw_config = load_runner_config(request.config_path)
    local = raw_config.get("local")
    if not isinstance(local, dict) or request.key not in local:
        raise ValueError(
            f"Runtime config materialization requires local.{request.key} in {request.config_path}."
        )
    local_key_cfg = local[request.key]
    if not isinstance(local_key_cfg, dict):
        raise ValueError(f"local.{request.key} must be a mapping in {request.config_path}")
    simulation = local_key_cfg.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError(f"local.{request.key}.simulation must be a mapping in {request.config_path}")

    runtime_config = copy.deepcopy(raw_config)
    runtime_sim = runtime_config["local"][request.key]["simulation"]
    runtime_object_placements = dict(runtime_sim.get("object_placements") or {})
    runtime_object_placements[episode.object_id] = dict(episode.placement)
    runtime_sim["placement_mode"] = "fixed"
    runtime_sim["manip_object_id"] = episode.object_id
    runtime_sim["object_placements"] = runtime_object_placements
    if request.runtime_sim_patch_path is not None:
        _apply_runtime_sim_patch(runtime_sim, patch_path=request.runtime_sim_patch_path)
    return runtime_config


def _build_episode_instruction(request: CollectionRequest, *, task_semantic_name: str) -> str:
    if request.task_type == TASK_PICK_UP:
        if str(request.task_object_id) == "orange_cube_ext":
            return "Pick red cube"
        return f"Pick up {task_semantic_name}."
    return f"{request.task_type}:{task_semantic_name}"


def _build_collection_episode_runner_argv(
    request: CollectionRequest,
    episode: EpisodePlacementSpec,
    runtime_config_path: Path,
    runtime_request_path: Path,
    *,
    dense_episode_artifact_path: Path,
    rl4vla_raw_episode_artifact_path: Path,
    debug_video_path: Path | None,
    debug_video_gif_path: Path | None,
) -> tuple[str, ...]:
    passthrough = list(request.passthrough_argv)
    if request.headless and not _has_flag(passthrough, "--headless"):
        passthrough.append("--headless")
    if not _has_flag(passthrough, "--sim_backend"):
        passthrough.extend(["--sim_backend", DEFAULT_RC5_SIM_BACKEND])
    return (
        "--motion_backend",
        request.motion_backend,
        "--task_type",
        request.task_type,
        "--task_object_id",
        episode.object_id,
        "--dense_episode_output",
        str(dense_episode_artifact_path),
        "--rl4vla_raw_episode_output",
        str(rl4vla_raw_episode_artifact_path),
        "--dense_episode_instruction",
        _build_episode_instruction(request, task_semantic_name=episode.task_semantic_name),
        "--dense_episode_target_width",
        str(request.dense_episode_image_width),
        "--dense_episode_target_height",
        str(request.dense_episode_image_height),
        *(
            ("--embed_runtime_bundle_in_rl4vla_raw_npz",)
            if request.embed_runtime_bundle_in_rl4vla_raw_npz
            else ("--no-embed_runtime_bundle_in_rl4vla_raw_npz",)
        ),
        *tuple(str(item) for item in passthrough),
        "--config_path",
        str(runtime_config_path),
        "--key",
        request.key,
        "--scene",
        request.scene_path,
        "--num_envs",
        str(request.num_envs),
        *(() if debug_video_path is None else ("--save_video_on_exit", "--save_video_path", str(debug_video_path))),
        *(
            ()
            if debug_video_gif_path is None
            else ("--save_video_gif_on_exit", "--save_video_gif_path", str(debug_video_gif_path))
        ),
        "--runtime_request_path",
        str(runtime_request_path),
    )


def materialize_episode_runtime_inputs(
    request: CollectionRequest,
) -> tuple[EpisodeRuntimeInput, ...]:
    episodes_root = request.output_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    materialized: List[EpisodeRuntimeInput] = []
    for episode in request.episodes:
        episode_dir = episodes_root / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = _clone_runtime_config_with_episode_placement(request, episode)
        runtime_config_path = episode_dir / "runtime_config.yaml"
        runtime_config_path.write_text(
            yaml.safe_dump(runtime_config, sort_keys=False),
            encoding="utf-8",
        )

        dense_episode_artifact_path = episode_dir / "dense_episode.npz"
        rl4vla_raw_episode_artifact_path = episode_dir / "rl4vla_raw_episode.npz"
        debug_video_path = episode_dir / "debug_video.mkv" if request.save_debug_video else None
        debug_video_gif_path = episode_dir / "debug_video.gif" if request.save_debug_gif else None
        request_path = episode_dir / "runtime_request.json"
        runner_argv = _build_collection_episode_runner_argv(
            request,
            episode,
            runtime_config_path,
            request_path,
            dense_episode_artifact_path=dense_episode_artifact_path,
            rl4vla_raw_episode_artifact_path=rl4vla_raw_episode_artifact_path,
            debug_video_path=debug_video_path,
            debug_video_gif_path=debug_video_gif_path,
        )
        request_payload = {
            "episode_id": episode.episode_id,
            "episode_index": episode.episode_index,
            "motion_backend": request.motion_backend,
            "task_type": request.task_type,
            "task_object_id": episode.object_id,
            "task_semantic_name": episode.task_semantic_name,
            "scene_path": request.scene_path,
            "runtime_config_path": str(runtime_config_path),
            "dense_episode_artifact_path": str(dense_episode_artifact_path),
            "rl4vla_raw_episode_artifact_path": str(rl4vla_raw_episode_artifact_path),
            "dense_episode_instruction": _build_episode_instruction(
                request,
                task_semantic_name=episode.task_semantic_name,
            ),
            "dense_episode_image_width": request.dense_episode_image_width,
            "dense_episode_image_height": request.dense_episode_image_height,
            "debug_video_path": None if debug_video_path is None else str(debug_video_path),
            "debug_video_gif_path": None if debug_video_gif_path is None else str(debug_video_gif_path),
            "runtime_sim_patch_path": (
                None if request.runtime_sim_patch_path is None else str(request.runtime_sim_patch_path)
            ),
            "placement_source": episode.placement_source,
            "placement_seed": episode.placement_seed,
            "placement": dict(episode.placement),
            "runner_argv": list(runner_argv),
        }
        request_path.write_text(
            json.dumps(request_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        materialized.append(
            EpisodeRuntimeInput(
                episode_id=episode.episode_id,
                episode_index=episode.episode_index,
                config_path=runtime_config_path,
                request_path=request_path,
                runner_argv=runner_argv,
            )
        )
    return tuple(materialized)


def _load_runtime_request_payload(request_path: Path) -> Dict[str, Any]:
    path = Path(request_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Runtime request does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime request must be a JSON mapping: {path}")
    runner_argv = payload.get("runner_argv")
    if not isinstance(runner_argv, list) or not runner_argv:
        raise ValueError(f"Runtime request must define non-empty runner_argv: {path}")
    episode_id = payload.get("episode_id")
    episode_index = payload.get("episode_index")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(f"Runtime request must define a non-empty episode_id: {path}")
    if not isinstance(episode_index, int):
        raise ValueError(f"Runtime request must define integer episode_index: {path}")
    return payload


def _strip_flag_with_value(argv: Sequence[str], flag: str) -> List[str]:
    stripped: List[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        if values[index] == flag:
            index += 2
            continue
        stripped.append(str(values[index]))
        index += 1
    return stripped


def _resolve_runtime_config_simulation(
    runtime_config_payload: Mapping[str, Any],
    *,
    key: str,
) -> Dict[str, Any]:
    local_cfg = runtime_config_payload.get("local")
    if not isinstance(local_cfg, dict):
        raise ValueError("Runtime config must define top-level 'local' mapping for collection batching.")
    key_cfg = local_cfg.get(key)
    if not isinstance(key_cfg, dict):
        raise ValueError(f"Runtime config must define local.{key} mapping for collection batching.")
    simulation_cfg = key_cfg.get("simulation")
    if not isinstance(simulation_cfg, dict):
        raise ValueError(f"Runtime config must define local.{key}.simulation mapping for collection batching.")
    return simulation_cfg


def _compute_runtime_input_compatibility_group_id(
    request: CollectionRequest,
    runtime_input: EpisodeRuntimeInput,
) -> str:
    runtime_payload = _load_runtime_request_payload(runtime_input.request_path)
    task_object_id = str(runtime_payload.get("task_object_id", request.task_object_id))
    runtime_config_payload = yaml.safe_load(runtime_input.config_path.read_text(encoding="utf-8"))
    normalized_runtime_config = copy.deepcopy(runtime_config_payload)
    runtime_sim = _resolve_runtime_config_simulation(
        normalized_runtime_config,
        key=request.key,
    )
    object_placements = runtime_sim.get("object_placements")
    if isinstance(object_placements, dict):
        task_object_cfg = object_placements.get(task_object_id)
        if isinstance(task_object_cfg, dict):
            normalized_object_cfg = dict(task_object_cfg)
            if "position" in normalized_object_cfg:
                normalized_object_cfg["position"] = "__collection_batch_position__"
            if "orientation" in normalized_object_cfg:
                normalized_object_cfg["orientation"] = "__collection_batch_orientation__"
            normalized_object_cfg.pop("position_per_env", None)
            normalized_object_cfg.pop("orientation_per_env", None)
            normalized_object_cfg.pop("task_semantic_name", None)
            object_placements[task_object_id] = normalized_object_cfg
    fingerprint = {
        "motion_backend": request.motion_backend,
        "task_type": request.task_type,
        "task_object_id": task_object_id,
        "scene_path": request.scene_path,
        "headless": request.headless,
        "runtime_sim_patch_path": runtime_payload.get("runtime_sim_patch_path"),
        "runtime_config": normalized_runtime_config,
    }
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_batch_runner_argv(
    runtime_inputs: Sequence[EpisodeRuntimeInput],
    *,
    env_count: int,
    batch_config_path: Path,
    dense_batch_output_dir: Path | None,
    rl4vla_raw_batch_output_dir: Path | None,
) -> tuple[str, ...]:
    runner_argv = runtime_inputs[0].runner_argv
    batch_argv = _strip_flag_with_value(runner_argv, "--num_envs")
    batch_argv = _strip_flag_with_value(batch_argv, "--config_path")
    batch_argv.extend(["--num_envs", str(int(env_count))])
    if int(env_count) > 1:
        batch_argv = _strip_flag_with_value(batch_argv, "--dense_episode_output")
        batch_argv = _strip_flag_with_value(batch_argv, "--dense_episode_output_dir")
        batch_argv = _strip_flag_with_value(batch_argv, "--runtime_request_path")
        batch_argv = _strip_flag_with_value(batch_argv, "--runtime_config_path_per_env_json")
        batch_argv = _strip_flag_with_value(batch_argv, "--runtime_request_path_per_env_json")
        batch_argv = _strip_flag_with_value(batch_argv, "--rl4vla_raw_episode_output")
        batch_argv = _strip_flag_with_value(batch_argv, "--rl4vla_raw_episode_output_dir")
        batch_argv = _strip_flag_with_value(batch_argv, "--episode_instruction_per_env_json")
        if dense_batch_output_dir is not None:
            batch_argv.extend(["--dense_episode_output_dir", str(dense_batch_output_dir)])
        if rl4vla_raw_batch_output_dir is not None:
            batch_argv.extend(["--rl4vla_raw_episode_output_dir", str(rl4vla_raw_batch_output_dir)])
        per_env_instructions = [
            str(_load_runtime_request_payload(item.request_path)["dense_episode_instruction"])
            for item in runtime_inputs
        ]
        per_env_runtime_config_paths = [str(item.config_path) for item in runtime_inputs]
        per_env_runtime_request_paths = [str(item.request_path) for item in runtime_inputs]
        batch_argv.extend(["--episode_instruction_per_env_json", json.dumps(per_env_instructions)])
        batch_argv.extend(["--runtime_config_path_per_env_json", json.dumps(per_env_runtime_config_paths)])
        batch_argv.extend(["--runtime_request_path_per_env_json", json.dumps(per_env_runtime_request_paths)])
    batch_argv.extend(["--config_path", str(batch_config_path)])
    return tuple(str(item) for item in batch_argv)


def _materialize_batched_runtime_config(
    request: CollectionRequest,
    runtime_inputs: Sequence[EpisodeRuntimeInput],
    *,
    output_dir: Path,
) -> Path:
    if not runtime_inputs:
        raise ValueError("Collection batching requires at least one runtime input.")
    base_runtime_config = yaml.safe_load(runtime_inputs[0].config_path.read_text(encoding="utf-8"))
    runtime_sim = _resolve_runtime_config_simulation(base_runtime_config, key=request.key)
    object_placements = runtime_sim.get("object_placements")
    if not isinstance(object_placements, dict):
        raise ValueError("Collection batched runtime config requires simulation.object_placements mapping.")
    first_runtime_payload = _load_runtime_request_payload(runtime_inputs[0].request_path)
    batch_task_object_id = str(first_runtime_payload.get("task_object_id", request.task_object_id))
    task_object_cfg = object_placements.get(batch_task_object_id)
    if not isinstance(task_object_cfg, dict):
        raise ValueError(
            "Collection batched runtime config requires task object placement under "
            f"object_placements.{batch_task_object_id}."
        )

    per_env_positions: List[List[float]] = []
    per_env_orientations: List[List[float]] = []
    for runtime_input in runtime_inputs:
        runtime_payload = _load_runtime_request_payload(runtime_input.request_path)
        runtime_task_object_id = str(runtime_payload.get("task_object_id", batch_task_object_id))
        if runtime_task_object_id != batch_task_object_id:
            raise ValueError(
                "Collection batched runtime config requires homogeneous task_object_id within one batch. "
                f"Expected '{batch_task_object_id}', got '{runtime_task_object_id}' for {runtime_input.episode_id}."
            )
        placement_cfg = _require_mapping(
            runtime_payload.get("placement"),
            label=f"{runtime_input.episode_id}.placement",
        )
        per_env_positions.append(
            _require_position_triplet(
                placement_cfg.get("position"),
                label=f"{runtime_input.episode_id}.placement.position",
            )
        )
        per_env_orientations.append(
            _require_orientation_quat(
                placement_cfg.get("orientation", task_object_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])),
                label=f"{runtime_input.episode_id}.placement.orientation",
            )
        )

    batch_task_object_cfg = dict(task_object_cfg)
    batch_task_object_cfg["position"] = list(per_env_positions[0])
    batch_task_object_cfg["orientation"] = list(per_env_orientations[0])
    batch_task_object_cfg["position_per_env"] = per_env_positions
    batch_task_object_cfg["orientation_per_env"] = per_env_orientations
    object_placements[batch_task_object_id] = batch_task_object_cfg
    runtime_sim["manip_object_id"] = batch_task_object_id

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_config_path = output_dir / "runtime_config.yaml"
    batch_config_path.write_text(
        yaml.safe_dump(base_runtime_config, sort_keys=False),
        encoding="utf-8",
    )
    return batch_config_path


def plan_prepared_execution_batches(
    request: CollectionRequest,
    runtime_inputs: Sequence[EpisodeRuntimeInput],
) -> tuple[PreparedExecutionBatch, ...]:
    if not runtime_inputs:
        return ()
    batches_root = request.output_dir / "batches"
    planned: List[PreparedExecutionBatch] = []
    current_group_id: str | None = None
    current_inputs: List[EpisodeRuntimeInput] = []

    def _flush_current() -> None:
        nonlocal current_group_id, current_inputs
        if not current_inputs:
            return
        batch_index = len(planned)
        batch_id = f"batch_{batch_index:06d}"
        batch_dir = batches_root / batch_id
        dense_batch_output_dir = None
        rl4vla_raw_batch_output_dir = None
        batch_config_path = current_inputs[0].config_path
        if len(current_inputs) > 1:
            batch_config_path = _materialize_batched_runtime_config(
                request,
                current_inputs,
                output_dir=batch_dir,
            )
            dense_batch_output_dir = batch_dir / "dense_episodes"
            dense_batch_output_dir.mkdir(parents=True, exist_ok=True)
            rl4vla_raw_batch_output_dir = batch_dir / "rl4vla_raw_episodes"
            rl4vla_raw_batch_output_dir.mkdir(parents=True, exist_ok=True)
        planned.append(
            PreparedExecutionBatch(
                batch_id=batch_id,
                compatibility_group_id=str(current_group_id),
                env_count=len(current_inputs),
                runtime_inputs=tuple(current_inputs),
                runner_argv=_build_batch_runner_argv(
                    current_inputs,
                    env_count=len(current_inputs),
                    batch_config_path=batch_config_path,
                    dense_batch_output_dir=dense_batch_output_dir,
                    rl4vla_raw_batch_output_dir=rl4vla_raw_batch_output_dir,
                ),
                batch_config_path=batch_config_path,
                dense_batch_output_dir=dense_batch_output_dir,
                rl4vla_raw_batch_output_dir=rl4vla_raw_batch_output_dir,
            )
        )
        current_group_id = None
        current_inputs = []

    for runtime_input in runtime_inputs:
        compatibility_group_id = (
            _compute_runtime_input_compatibility_group_id(request, runtime_input)
            if int(request.num_envs) > 1
            else f"singleton:{runtime_input.episode_id}"
        )
        if (
            current_inputs
            and (
                compatibility_group_id != current_group_id
                or len(current_inputs) >= int(request.num_envs)
            )
        ):
            _flush_current()
        current_group_id = compatibility_group_id
        current_inputs.append(runtime_input)
    _flush_current()
    return tuple(planned)


def _extract_semantic_feedback(trace_record: Any) -> Dict[str, Any]:
    semantic_task_success = None
    failed_stage = None
    for event in reversed(_extract_trace_events(trace_record)):
        if isinstance(event, Mapping):
            event_type = event.get("event_type")
            payload = event.get("payload", {})
        else:
            event_type = getattr(event, "event_type", None)
            payload = getattr(event, "payload", {})
        if event_type != "macro_finished" or not isinstance(payload, Mapping):
            continue
        if isinstance(payload.get("semantic_task_success"), bool):
            semantic_task_success = bool(payload.get("semantic_task_success"))
        raw_failed_stage = payload.get("failed_stage")
        if isinstance(raw_failed_stage, str) and raw_failed_stage.strip():
            failed_stage = raw_failed_stage
        break
    return {
        "semantic_task_success": semantic_task_success,
        "failed_stage": failed_stage,
    }


def _normalize_runner_result(raw_result: Any) -> tuple[int, Any]:
    if raw_result is None or isinstance(raw_result, int):
        exit_code = 0 if raw_result is None else int(raw_result)
        return exit_code, None
    return int(getattr(raw_result, "exit_code")), getattr(raw_result, "trace_record", None)


def _run_logged_subprocess(argv: Sequence[str], *, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _TeeStream(sys.stdout, log_file)
        tee_stderr = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            return execute(list(argv))


def _extract_trace_batch_feedback(trace_record: Any) -> List[Dict[str, Any]]:
    batch_trace = None
    if isinstance(trace_record, Mapping):
        batch_trace = trace_record.get("batch_trace")
    elif trace_record is not None:
        batch_trace = getattr(trace_record, "batch_trace", None)
    if batch_trace is not None:
        raw_records = (
            batch_trace.get("per_env_records", [])
            if isinstance(batch_trace, Mapping)
            else getattr(batch_trace, "per_env_records", [])
        )
        normalized: List[Dict[str, Any]] = []
        for item in raw_records or []:
            normalized.append(dict(item) if isinstance(item, Mapping) else asdict(item))
        if normalized:
            return normalized

    for event in reversed(_extract_trace_events(trace_record)):
        if isinstance(event, Mapping):
            event_type = event.get("event_type")
            payload = event.get("payload", {})
        else:
            event_type = getattr(event, "event_type", None)
            payload = getattr(event, "payload", {})
        if event_type != "batch_runtime_feedback" or not isinstance(payload, Mapping):
            continue
        per_env_feedback = payload.get("per_env_feedback", [])
        return [dict(item) for item in per_env_feedback if isinstance(item, Mapping)]
    return []


def _resolve_dense_episode_artifact_path(
    payload: Mapping[str, Any],
    *,
    batch: PreparedExecutionBatch,
    env_index: int,
) -> Path | None:
    from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_artifacts import (
        resolve_batched_dense_episode_output_path,
    )

    raw_target = payload.get("dense_episode_artifact_path")
    target_path = None if raw_target is None else Path(str(raw_target)).expanduser().resolve()
    if batch.env_count <= 1 or batch.dense_batch_output_dir is None:
        return target_path
    source_path = resolve_batched_dense_episode_output_path(
        batch.dense_batch_output_dir,
        env_index=env_index,
    )
    if not source_path.exists():
        return None if target_path is None or not target_path.exists() else target_path
    if target_path is None:
        return source_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(target_path)
    try:
        if batch.dense_batch_output_dir.exists() and not any(batch.dense_batch_output_dir.iterdir()):
            batch.dense_batch_output_dir.rmdir()
    except OSError:
        pass
    return target_path


def _resolve_rl4vla_raw_episode_artifact_path(
    payload: Mapping[str, Any],
    *,
    batch: PreparedExecutionBatch,
    env_index: int,
) -> Path | None:
    from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_artifacts import (
        resolve_batched_rl4vla_raw_episode_output_path,
    )

    raw_target = payload.get("rl4vla_raw_episode_artifact_path")
    target_path = None if raw_target is None else Path(str(raw_target)).expanduser().resolve()
    if batch.env_count <= 1 or batch.rl4vla_raw_batch_output_dir is None:
        return target_path
    source_path = resolve_batched_rl4vla_raw_episode_output_path(
        batch.rl4vla_raw_batch_output_dir,
        env_index=env_index,
    )
    if not source_path.exists():
        return None if target_path is None or not target_path.exists() else target_path
    if target_path is None:
        return source_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(target_path)
    try:
        if batch.rl4vla_raw_batch_output_dir.exists() and not any(batch.rl4vla_raw_batch_output_dir.iterdir()):
            batch.rl4vla_raw_batch_output_dir.rmdir()
    except OSError:
        pass
    return target_path


def execute_prepared_execution_batches(
    batches: Sequence[PreparedExecutionBatch],
    *,
    stop_on_failure: bool,
    runner=None,
) -> tuple[EpisodeExecutionResult, ...]:
    if runner is None:
        runner = _run_logged_subprocess
    executed: List[EpisodeExecutionResult] = []
    for batch in batches:
        batch_log_path = batch.batch_config_path.parent / "run.log"
        runtime_exit_code, trace_record = _normalize_runner_result(
            runner(list(batch.runner_argv), log_path=batch_log_path)
        )
        per_env_feedback = _extract_trace_batch_feedback(trace_record)
        per_env_by_index = {
            int(item["env_index"]): item
            for item in per_env_feedback
            if isinstance(item.get("env_index"), int)
        }
        batch_semantic_feedback = _extract_semantic_feedback(trace_record)
        if batch.env_count > 1 and runtime_exit_code == 0 and len(per_env_by_index) != batch.env_count:
            raise ValueError(
                "Collection batched execution requires per-env runtime feedback for every env in the batch. "
                f"batch_id={batch.batch_id} env_count={batch.env_count} feedback_count={len(per_env_by_index)}"
            )
        batch_results: List[EpisodeExecutionResult] = []
        for env_index, runtime_input in enumerate(batch.runtime_inputs):
            payload = _load_runtime_request_payload(runtime_input.request_path)
            env_feedback = per_env_by_index.get(env_index)
            semantic_task_success = (
                batch_semantic_feedback.get("semantic_task_success")
                if env_feedback is None
                else env_feedback.get("semantic_task_success")
            )
            failed_stage = (
                batch_semantic_feedback.get("failed_stage")
                if env_feedback is None
                else env_feedback.get("failed_stage")
            )
            exit_code = int(runtime_exit_code)
            execution_outcome = "success" if exit_code == 0 else "failed"
            success = exit_code == 0
            if runtime_exit_code == 0 and semantic_task_success is False:
                exit_code = SEMANTIC_FAILURE_EXIT_CODE
                execution_outcome = "failed"
                success = False
            batch_results.append(
                EpisodeExecutionResult(
                    episode_id=payload["episode_id"],
                    episode_index=payload["episode_index"],
                    request_path=runtime_input.request_path,
                    exit_code=exit_code,
                    runtime_exit_code=int(runtime_exit_code),
                    execution_outcome=execution_outcome,
                    success=success,
                    semantic_task_success=semantic_task_success,
                    failed_stage=failed_stage,
                    trace_record=trace_record,
                    dense_episode_artifact_path=_resolve_dense_episode_artifact_path(
                        payload,
                        batch=batch,
                        env_index=env_index,
                    ),
                    rl4vla_raw_episode_artifact_path=_resolve_rl4vla_raw_episode_artifact_path(
                        payload,
                        batch=batch,
                        env_index=env_index,
                    ),
                    batch_id=batch.batch_id,
                    compatibility_group_id=batch.compatibility_group_id,
                    env_index=env_index,
                )
            )
        executed.extend(batch_results)
        if stop_on_failure and any(item.exit_code != 0 for item in batch_results):
            break
    return tuple(executed)


def execute_prepared_runtime_inputs(
    runtime_inputs: Sequence[EpisodeRuntimeInput],
    *,
    stop_on_failure: bool,
    runner=None,
) -> tuple[EpisodeExecutionResult, ...]:
    if runner is None:
        runner = _run_logged_subprocess
    executed: List[EpisodeExecutionResult] = []
    for runtime_input in runtime_inputs:
        payload = _load_runtime_request_payload(runtime_input.request_path)
        runner_argv = [str(item) for item in payload["runner_argv"]]
        episode_log_path = runtime_input.request_path.parent / "run.log"
        runtime_exit_code, trace_record = _normalize_runner_result(
            runner(runner_argv, log_path=episode_log_path)
        )
        semantic_feedback = _extract_semantic_feedback(trace_record)
        semantic_task_success = semantic_feedback.get("semantic_task_success")
        failed_stage = semantic_feedback.get("failed_stage")
        exit_code = int(runtime_exit_code)
        execution_outcome = "success" if exit_code == 0 else "failed"
        success = exit_code == 0
        if runtime_exit_code == 0 and semantic_task_success is False:
            exit_code = SEMANTIC_FAILURE_EXIT_CODE
            execution_outcome = "failed"
            success = False
        result = EpisodeExecutionResult(
            episode_id=payload["episode_id"],
            episode_index=payload["episode_index"],
            request_path=runtime_input.request_path,
            exit_code=exit_code,
            runtime_exit_code=int(runtime_exit_code),
            execution_outcome=execution_outcome,
            success=success,
            semantic_task_success=semantic_task_success,
            failed_stage=failed_stage,
            trace_record=trace_record,
            dense_episode_artifact_path=Path(str(payload["dense_episode_artifact_path"])).expanduser().resolve(),
            rl4vla_raw_episode_artifact_path=(
                None
                if payload.get("rl4vla_raw_episode_artifact_path") is None
                else Path(str(payload["rl4vla_raw_episode_artifact_path"])).expanduser().resolve()
            ),
        )
        executed.append(result)
        if stop_on_failure and exit_code != 0:
            break
    return tuple(executed)


def _finalize_episode_runtime_request_artifacts(
    item: EpisodeExecutionResult,
) -> tuple[EpisodeExecutionResult, Dict[str, Any]]:
    runtime_request = _load_runtime_request_payload(item.request_path)
    outcome_token = "success" if item.success else "fail"
    debug_video_raw = runtime_request.get("debug_video_path")
    debug_gif_raw = runtime_request.get("debug_video_gif_path")
    debug_video_final = _finalize_artifact_path(
        None if debug_video_raw is None else Path(str(debug_video_raw)).expanduser().resolve(),
        outcome_token=outcome_token,
    )
    debug_video_gif_final = _finalize_artifact_path(
        None if debug_gif_raw is None else Path(str(debug_gif_raw)).expanduser().resolve(),
        outcome_token=outcome_token,
    )
    finalized_raw_artifact_path = _finalize_artifact_path(
        item.rl4vla_raw_episode_artifact_path,
        outcome_token=outcome_token,
    )
    replacements: Dict[str, str] = {}
    if item.rl4vla_raw_episode_artifact_path is not None and finalized_raw_artifact_path is not None:
        replacements[str(item.rl4vla_raw_episode_artifact_path)] = str(finalized_raw_artifact_path)
    if isinstance(runtime_request.get("debug_video_path"), str) and debug_video_final is not None:
        replacements[str(runtime_request["debug_video_path"])] = str(debug_video_final)
    if isinstance(runtime_request.get("debug_video_gif_path"), str) and debug_video_gif_final is not None:
        replacements[str(runtime_request["debug_video_gif_path"])] = str(debug_video_gif_final)
    if replacements:
        runtime_request = _rewrite_exact_string_values(runtime_request, replacements=replacements)
    runtime_request["rl4vla_raw_episode_artifact_path"] = (
        None if finalized_raw_artifact_path is None else str(finalized_raw_artifact_path)
    )
    runtime_request["debug_video_path"] = None if debug_video_final is None else str(debug_video_final)
    runtime_request["debug_video_gif_path"] = (
        None if debug_video_gif_final is None else str(debug_video_gif_final)
    )
    item.request_path.write_text(
        json.dumps(runtime_request, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if finalized_raw_artifact_path is not None:
        rewrite_rl4vla_raw_episode_embedded_runtime_request_json(
            finalized_raw_artifact_path,
            runtime_request_json=json.dumps(runtime_request, indent=2, sort_keys=True),
        )
    return (
        EpisodeExecutionResult(
            episode_id=item.episode_id,
            episode_index=item.episode_index,
            request_path=item.request_path,
            exit_code=item.exit_code,
            runtime_exit_code=item.runtime_exit_code,
            execution_outcome=item.execution_outcome,
            success=item.success,
            semantic_task_success=item.semantic_task_success,
            failed_stage=item.failed_stage,
            trace_record=item.trace_record,
            result_path=item.result_path,
            export_manifest_path=item.export_manifest_path,
            dense_episode_artifact_path=_finalize_artifact_path(
                item.dense_episode_artifact_path,
                outcome_token=outcome_token,
            ),
            rl4vla_raw_episode_artifact_path=finalized_raw_artifact_path,
            debug_video_path=debug_video_final,
            debug_video_gif_path=debug_video_gif_final,
            batch_id=item.batch_id,
            compatibility_group_id=item.compatibility_group_id,
            env_index=item.env_index,
        ),
        runtime_request,
    )


def write_episode_result_artifacts(
    request: CollectionRequest,
    execution_results: Sequence[EpisodeExecutionResult],
) -> tuple[EpisodeExecutionResult, ...]:
    from openreal2sim.simulation.maniskill.scripts.rc5_unified_trajectory import (
        write_episode_artifact,
    )

    updated: List[EpisodeExecutionResult] = []
    for item in execution_results:
        finalized_item, runtime_request = _finalize_episode_runtime_request_artifacts(item)
        artifact_path = item.request_path.parent / "episode_result.json"
        write_episode_artifact(
            artifact_path=artifact_path,
            request=request,
            runtime_request=runtime_request,
            execution_result=finalized_item,
        )
        updated.append(
            EpisodeExecutionResult(
                episode_id=finalized_item.episode_id,
                episode_index=finalized_item.episode_index,
                request_path=finalized_item.request_path,
                exit_code=finalized_item.exit_code,
                runtime_exit_code=finalized_item.runtime_exit_code,
                execution_outcome=finalized_item.execution_outcome,
                success=finalized_item.success,
                semantic_task_success=finalized_item.semantic_task_success,
                failed_stage=finalized_item.failed_stage,
                trace_record=finalized_item.trace_record,
                result_path=artifact_path,
                export_manifest_path=finalized_item.export_manifest_path,
                dense_episode_artifact_path=finalized_item.dense_episode_artifact_path,
                rl4vla_raw_episode_artifact_path=finalized_item.rl4vla_raw_episode_artifact_path,
                debug_video_path=finalized_item.debug_video_path,
                debug_video_gif_path=finalized_item.debug_video_gif_path,
                batch_id=finalized_item.batch_id,
                compatibility_group_id=finalized_item.compatibility_group_id,
                env_index=finalized_item.env_index,
            )
        )
    return tuple(updated)


def write_episode_export_artifacts(
    execution_results: Sequence[EpisodeExecutionResult],
    *,
    exporter_name: str,
) -> tuple[EpisodeExecutionResult, ...]:
    from openreal2sim.simulation.maniskill.scripts.rc5_unified_exporters import (
        export_episode_artifact,
    )

    updated: List[EpisodeExecutionResult] = []
    for item in execution_results:
        if item.result_path is None:
            raise ValueError("Episode export requires result_path to be materialized before exporting.")
        export_dir = item.request_path.parent / "exports" / exporter_name
        export_manifest_path = export_episode_artifact(
            episode_artifact_path=item.result_path,
            export_dir=export_dir,
            exporter_name=exporter_name,
        )
        updated.append(
            EpisodeExecutionResult(
                episode_id=item.episode_id,
                episode_index=item.episode_index,
                request_path=item.request_path,
                exit_code=item.exit_code,
                runtime_exit_code=item.runtime_exit_code,
                execution_outcome=item.execution_outcome,
                success=item.success,
                semantic_task_success=item.semantic_task_success,
                failed_stage=item.failed_stage,
                trace_record=item.trace_record,
                result_path=item.result_path,
                export_manifest_path=export_manifest_path,
                dense_episode_artifact_path=item.dense_episode_artifact_path,
                rl4vla_raw_episode_artifact_path=item.rl4vla_raw_episode_artifact_path,
                debug_video_path=item.debug_video_path,
                debug_video_gif_path=item.debug_video_gif_path,
                batch_id=item.batch_id,
                compatibility_group_id=item.compatibility_group_id,
                env_index=item.env_index,
            )
        )
    return tuple(updated)


def write_collection_execution_summary(
    request: CollectionRequest,
    execution_results: Sequence[EpisodeExecutionResult],
    *,
    execution_batches: Sequence[PreparedExecutionBatch],
) -> Path:
    summary_path = request.output_dir / "execution_summary.json"
    completed = [
        {
            "episode_id": item.episode_id,
            "episode_index": item.episode_index,
            "request_path": str(item.request_path),
            "exit_code": item.exit_code,
            "runtime_exit_code": item.runtime_exit_code,
            "execution_outcome": item.execution_outcome,
            "success": item.success,
            "semantic_task_success": item.semantic_task_success,
            "failed_stage": item.failed_stage,
            "batch_id": item.batch_id,
            "compatibility_group_id": item.compatibility_group_id,
            "env_index": item.env_index,
            "result_path": None if item.result_path is None else str(item.result_path),
            "export_manifest_path": (
                None if item.export_manifest_path is None else str(item.export_manifest_path)
            ),
        }
        for item in execution_results
    ]
    fail_results = [item for item in execution_results if item.exit_code != 0]
    executed_batch_ids = {item.batch_id for item in execution_results if item.batch_id is not None}
    batched_execution_used = any(int(batch.env_count) > 1 for batch in execution_batches)
    payload = {
        "run_mode": "collection",
        "phase": (
            "homogeneous_batched_runtime_execution"
            if batched_execution_used
            else "sequential_runtime_execution"
        ),
        "motion_backend": request.motion_backend,
        "task_type": request.task_type,
        "requested_num_envs": request.num_envs,
        "episode_count_planned": len(request.episodes),
        "episode_count_executed": len(execution_results),
        "continue_after_failure": not request.stop_on_failure,
        "stopped_on_failure": bool(request.stop_on_failure and len(execution_results) < len(request.episodes)),
        "failure_count": len(fail_results),
        "failed_episode_ids": [item.episode_id for item in fail_results],
        "batching": {
            "requested_num_envs": request.num_envs,
            "batched_execution_used": batched_execution_used,
            "batch_count_planned": len(execution_batches),
            "batch_count_executed": len(executed_batch_ids),
            "max_planned_batch_size": (
                0 if not execution_batches else max(int(batch.env_count) for batch in execution_batches)
            ),
        },
        "execution_batches": [
            {
                "batch_id": batch.batch_id,
                "compatibility_group_id": batch.compatibility_group_id,
                "env_count": batch.env_count,
                "episode_ids": [item.episode_id for item in batch.runtime_inputs],
                "batch_config_path": str(batch.batch_config_path),
                "dense_batch_output_dir": (
                    None if batch.dense_batch_output_dir is None else str(batch.dense_batch_output_dir)
                ),
                "rl4vla_raw_batch_output_dir": (
                    None if batch.rl4vla_raw_batch_output_dir is None else str(batch.rl4vla_raw_batch_output_dir)
                ),
            }
            for batch in execution_batches
        ],
        "episodes": completed,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def run_collection(argv: Optional[Sequence[str]] = None) -> int:
    request = resolve_collection_request(argv)
    runtime_inputs = materialize_episode_runtime_inputs(request)
    execution_batches = plan_prepared_execution_batches(request, runtime_inputs)
    write_collection_plan(request)
    execution_results = execute_prepared_execution_batches(
        execution_batches,
        stop_on_failure=request.stop_on_failure,
    )
    execution_results = write_episode_result_artifacts(request, execution_results)
    if request.exporter is not None:
        execution_results = write_episode_export_artifacts(
            execution_results,
            exporter_name=request.exporter,
        )
    write_collection_execution_summary(
        request,
        execution_results,
        execution_batches=execution_batches,
    )
    fail_result = next((item for item in execution_results if item.exit_code != 0), None)
    return 0 if fail_result is None else int(fail_result.exit_code)


def _warn_if_env_override_is_used_without_cli(args: argparse.Namespace) -> None:
    if args.rc5_asset_dir is None and os.environ.get("RC5_AERO_HAND_ASSET_DIR"):
        emit_warning(
            "RC5Unified",
            f"Using RC5_AERO_HAND_ASSET_DIR from environment: {os.environ['RC5_AERO_HAND_ASSET_DIR']}",
        )
    if os.environ.get("RC5_AERO_HAND_URDF_FILENAME"):
        emit_warning(
            "RC5Unified",
            "Using RC5_AERO_HAND_URDF_FILENAME from environment: "
            f"{os.environ['RC5_AERO_HAND_URDF_FILENAME']}",
        )
    if os.environ.get("OPENR2S_RC5_PLANNING_ASSET_DIR"):
        emit_warning(
            "RC5Unified",
            "Using OPENR2S_RC5_PLANNING_ASSET_DIR from environment: "
            f"{os.environ['OPENR2S_RC5_PLANNING_ASSET_DIR']}",
        )
    if os.environ.get("RC5_DEBUG_PLANNER_USE_MAIN_ASSETS"):
        emit_warning(
            "RC5Unified",
            "Using RC5_DEBUG_PLANNER_USE_MAIN_ASSETS from environment: "
            f"{os.environ['RC5_DEBUG_PLANNER_USE_MAIN_ASSETS']}",
        )
    if args.rc5_move_group is None and os.environ.get("OPENR2S_RC5_MOVE_GROUP"):
        resolve_rc5_move_group(
            None,
            scope="RC5Unified",
            warn_on_env=True,
            warn_on_default=False,
        )


def execute(argv: Optional[Sequence[str]] = None):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    args, _unknown = parser.parse_known_args(list(argv))
    _validate_parser_surface_args(args)
    if args.run_mode != "episode":
        raise ValueError("execute() supports only run_mode='episode'; use main()/run_collection() for collection mode.")
    _warn_if_env_override_is_used_without_cli(args)
    passthrough_argv = _append_singleton_passthrough_flags(
        argv,
        _strip_unified_args(argv),
    )
    request = resolve_unified_bootstrap_request(passthrough_argv)
    bootstrap = request.bootstrap
    effective_motion_backend = _resolve_effective_motion_backend(
        args,
        config_path=request.config_path,
        key=request.key,
    )
    validate_rc5_bootstrap_for_backend(bootstrap, effective_motion_backend)
    effective_task_object_id = _resolve_effective_task_object_id(
        explicit_object_id=args.task_object_id,
        config_path=request.config_path,
        key=request.key,
    )
    intent = resolve_episode_intent(
        task_type=args.task_type,
        object_id=effective_task_object_id,
        destination_id=args.task_destination_id,
        prompt=args.task_prompt,
    )
    validate_task_supported_for_runtime(intent)
    task_plan = build_task_plan(intent)
    trace_seed = build_canonical_trace_seed(
        motion_backend=effective_motion_backend,
        bootstrap=bootstrap,
        task_plan=task_plan,
    )
    request = UnifiedBackendRequest(
        motion_backend=effective_motion_backend,
        passthrough_argv=list(passthrough_argv),
        bootstrap=bootstrap,
        task_plan=task_plan,
        trace_seed=trace_seed,
        rc5_asset_dir=args.rc5_asset_dir,
        rc5_move_group=args.rc5_move_group,
        warning_scope="RC5Unified",
    )
    result = execute_backend_request(request)
    if getattr(result, "trace_seed", None) is None:
        result = UnifiedBackendResult(
            motion_backend=result.motion_backend,
            exit_code=result.exit_code,
            dispatch_plan=result.dispatch_plan,
            trace_seed=trace_seed,
            trace_record=getattr(result, "trace_record", None),
        )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args, _unknown = parser.parse_known_args(effective_argv)
    _validate_parser_surface_args(args)
    if args.run_mode == "collection":
        return run_collection(effective_argv)
    passthrough_argv = _strip_unified_args(effective_argv)
    bootstrap_request = resolve_unified_bootstrap_request(passthrough_argv)
    args.motion_backend = _resolve_effective_motion_backend(
        args,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
    )
    args.task_object_id = _resolve_effective_task_object_id(
        explicit_object_id=args.task_object_id,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
    )
    requested_num_envs = parse_requested_num_envs_from_argv(_strip_unified_args(effective_argv))
    run_dir = _resolve_auto_run_dir(effective_argv, args)
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_argv = _append_default_artifact_args(
        effective_argv,
        run_dir,
        num_envs=requested_num_envs,
    )
    args, _unknown = parser.parse_known_args(effective_argv)
    args.motion_backend = _resolve_effective_motion_backend(
        args,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
    )
    args.task_object_id = _resolve_effective_task_object_id(
        explicit_object_id=args.task_object_id,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
    )
    task_plan = build_task_plan(
        resolve_episode_intent(
            task_type=args.task_type,
            object_id=args.task_object_id,
            destination_id=args.task_destination_id,
            prompt=args.task_prompt,
        )
    )
    effective_argv, runtime_request_path = _materialize_default_runtime_request(
        effective_argv,
        run_dir=run_dir,
        bootstrap_request=bootstrap_request,
        args=args,
    )
    log_path = run_dir / "run.log"
    summary_path = run_dir / "execution_summary.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _TeeStream(sys.stdout, log_file)
        tee_stderr = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"[RC5Unified] Auto artifact run directory: {run_dir}")
            result: UnifiedBackendResult | None = None
            artifact_paths: dict[str, str | None] | None = None
            try:
                result = execute(effective_argv)
                plan = result.dispatch_plan
                trace_record = result.trace_record
                trace_seed = result.trace_seed
                artifact_paths = _finalize_default_artifact_paths(
                    run_dir,
                    _resolve_result_outcome_token(result),
                    effective_argv=effective_argv,
                )
                if runtime_request_path is not None:
                    _finalize_default_runtime_request_artifacts(
                        runtime_request_path,
                        requested_rl4vla_raw_episode_artifact_path=_resolve_singleton_requested_artifact_path(
                            effective_argv,
                            flag="--rl4vla_raw_episode_output",
                            default_path=None,
                        ),
                        finalized_rl4vla_raw_episode_artifact_path=(
                            None
                            if artifact_paths.get("rl4vla_raw_episode_path") is None
                            else Path(str(artifact_paths["rl4vla_raw_episode_path"])).expanduser().resolve()
                        ),
                        requested_debug_video_path=_resolve_singleton_requested_artifact_path(
                            effective_argv,
                            flag="--save_video_path",
                            default_path=None,
                        ),
                        finalized_debug_video_path=(
                            None
                            if artifact_paths.get("debug_video_path") is None
                            else Path(str(artifact_paths["debug_video_path"])).expanduser().resolve()
                        ),
                        requested_debug_video_gif_path=_resolve_singleton_requested_artifact_path(
                            effective_argv,
                            flag="--save_video_gif_path",
                            default_path=None,
                        ),
                        finalized_debug_video_gif_path=(
                            None
                            if artifact_paths.get("debug_video_gif_path") is None
                            else Path(str(artifact_paths["debug_video_gif_path"])).expanduser().resolve()
                        ),
                    )
                if artifact_paths.get("dense_episode_path") is not None:
                    print(f"[RC5Unified] Dense artifact: {artifact_paths['dense_episode_path']}")
                if artifact_paths.get("rl4vla_raw_episode_path") is not None:
                    print(f"[RC5Unified] RL4VLA raw artifact: {artifact_paths['rl4vla_raw_episode_path']}")
                print(
                    f"[RC5Unified] motion_backend={args.motion_backend} "
                    f"key={getattr(trace_seed, 'config_key', None)} "
                    f"robot_uids={getattr(trace_seed, 'robot_uids', None)} "
                    f"{summarize_task_plan(task_plan)} "
                    f"trace_stage_count={len(getattr(trace_seed, 'stage_kinds', ()))} "
                    f"trace_event_count={len(getattr(trace_record, 'events', ()))} "
                    f"trace_outcome={getattr(trace_record, 'execution_outcome', '<none>')} "
                    f"target_module={plan.module_name} "
                    f"forwarded_argc={len(plan.forwarded_argv)}"
                )
                _write_execution_summary(
                    summary_path=summary_path,
                    effective_argv=effective_argv,
                    args=args,
                    result=result,
                    task_plan=task_plan,
                    run_dir=run_dir,
                    artifact_paths=artifact_paths,
                )
                print(f"[RC5Unified] Execution summary: {summary_path}")
                return result.exit_code
            except Exception as exc:
                traceback.print_exc()
                if artifact_paths is None:
                    artifact_paths = _finalize_default_artifact_paths(
                        run_dir,
                        "fail",
                        effective_argv=effective_argv,
                    )
                if runtime_request_path is not None:
                    _finalize_default_runtime_request_artifacts(
                        runtime_request_path,
                        requested_rl4vla_raw_episode_artifact_path=_resolve_singleton_requested_artifact_path(
                            effective_argv,
                            flag="--rl4vla_raw_episode_output",
                            default_path=None,
                        ),
                        finalized_rl4vla_raw_episode_artifact_path=(
                            None
                            if artifact_paths.get("rl4vla_raw_episode_path") is None
                            else Path(str(artifact_paths["rl4vla_raw_episode_path"])).expanduser().resolve()
                        ),
                        requested_debug_video_path=_resolve_singleton_requested_artifact_path(
                            effective_argv,
                            flag="--save_video_path",
                            default_path=None,
                        ),
                        finalized_debug_video_path=(
                            None
                            if artifact_paths.get("debug_video_path") is None
                            else Path(str(artifact_paths["debug_video_path"])).expanduser().resolve()
                        ),
                        requested_debug_video_gif_path=_resolve_singleton_requested_artifact_path(
                            effective_argv,
                            flag="--save_video_gif_path",
                            default_path=None,
                        ),
                        finalized_debug_video_gif_path=(
                            None
                            if artifact_paths.get("debug_video_gif_path") is None
                            else Path(str(artifact_paths["debug_video_gif_path"])).expanduser().resolve()
                        ),
                    )
                _write_execution_summary(
                    summary_path=summary_path,
                    effective_argv=effective_argv,
                    args=args,
                    result=result,
                    task_plan=task_plan,
                    run_dir=run_dir,
                    artifact_paths=artifact_paths,
                    failure=f"{type(exc).__name__}: {exc}",
                )
                print(f"[RC5Unified] Execution summary: {summary_path}")
                raise


if __name__ == "__main__":
    raise SystemExit(main())
