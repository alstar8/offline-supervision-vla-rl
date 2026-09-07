from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from openreal2sim.simulation.maniskill.scripts.rc5_unified_bootstrap import (
    detect_config_key,
    emit_warning,
    has_flag,
    load_runner_config,
    load_simulation_config_sections,
    pick_simulation_value,
    resolve_unified_bootstrap_request,
    validate_rc5_bootstrap_for_backend,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_exporters import (
    SUPPORTED_EXPORTERS,
    export_episode_artifact,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_execution import (
    PROXY_BACKEND,
    UNIFIED_BACKENDS,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_artifacts import (
    DEFAULT_VIDEO_FPS,
    flush_video_buffer_to_file,
    resolve_batched_debug_video_gif_output_path,
    resolve_batched_debug_video_output_path,
    resolve_batched_dense_episode_output_path,
    resolve_batched_object_pose_trace_output_path,
    resolve_batched_rl4vla_raw_episode_output_path,
    write_debug_video_gif_from_video,
)
from openreal2sim.simulation.maniskill.scripts.maniskill_num_envs_policy import (
    DEFAULT_RC5_SIM_BACKEND,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_tasks import (
    SUPPORTED_TASK_TYPES,
    TASK_PICK_UP,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    decode_dense_episode_images,
    load_dense_episode_artifact,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_trajectory import (
    write_episode_artifact,
)


def run_rc5_unified_execute(argv: Sequence[str]):
    from openreal2sim.simulation.maniskill.scripts.run_rc5_unified import (
        execute as run_rc5_unified_execute,
    )

    return run_rc5_unified_execute(argv)


@dataclass(frozen=True)
class EpisodePlacementSpec:
    episode_id: str
    episode_index: int
    object_id: str
    task_semantic_name: str
    placement_source: str
    placement_seed: int | None
    placement: Dict[str, Any]
    object_placements_override: Dict[str, Dict[str, Any]] | None = None


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
    save_dense_episode_artifact: bool
    save_debug_video: bool
    save_debug_gif: bool
    save_object_pose_trace: bool
    embed_runtime_bundle_in_rl4vla_raw_npz: bool
    rl4vla_success_output_dir: Path | None
    rl4vla_success_index_start: int
    runtime_sim_patch_path: Path | None
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
    debug_video_batch_output_dir: Path | None = None
    debug_video_gif_batch_output_dir: Path | None = None
    object_pose_trace_batch_output_dir: Path | None = None


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
    rl4vla_success_export_path: Path | None = None
    rl4vla_success_export_index: int | None = None
    debug_video_path: Path | None = None
    debug_video_gif_path: Path | None = None
    object_pose_trace_path: Path | None = None
    batch_id: str | None = None
    compatibility_group_id: str | None = None
    env_index: int | None = None


SEMANTIC_FAILURE_EXIT_CODE = 3


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
        primary = self._streams[0] if self._streams else None
        return bool(primary is not None and hasattr(primary, "isatty") and primary.isatty())

    @property
    def encoding(self):
        primary = self._streams[0] if self._streams else None
        return getattr(primary, "encoding", "utf-8")


def _resolve_episode_outcome_token(item: EpisodeExecutionResult) -> str:
    return "success" if item.success else "fail"


def _rewrite_exact_string_values(value: Any, *, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_rewrite_exact_string_values(item, replacements=replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_exact_string_values(item, replacements=replacements)
            for key, item in value.items()
        }
    return value


def _finalize_episode_sidecar_path(path_value: Any, *, outcome_token: str) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    artifact_path = Path(path_value).expanduser().resolve()
    if not artifact_path.exists():
        return None
    stem = artifact_path.stem
    if stem.endswith("_success") or stem.endswith("_fail"):
        return artifact_path
    target_path = artifact_path.with_name(f"{stem}_{outcome_token}{artifact_path.suffix}")
    artifact_path.rename(target_path)
    return target_path


def _build_final_rl4vla_raw_episode_filename(item: EpisodeExecutionResult) -> str:
    run_id = item.request_path.parents[2].name
    batch_token = item.batch_id if item.batch_id else f"batch_{int(item.episode_index):06d}"
    env_index = 0 if item.env_index is None else int(item.env_index)
    outcome_token = _resolve_episode_outcome_token(item)
    return (
        f"rl4vla_raw_episode__{run_id}__{batch_token}__env_{env_index:02d}__{outcome_token}.npz"
    )


def _build_rl4vla_success_export_filename(
    item: EpisodeExecutionResult,
    *,
    global_index: int,
) -> str:
    source_name = (
        item.rl4vla_raw_episode_artifact_path.name
        if item.rl4vla_raw_episode_artifact_path is not None
        else _build_final_rl4vla_raw_episode_filename(item)
    )
    return f"rl4vla_raw_episode__idx_{int(global_index):09d}__{source_name.removeprefix('rl4vla_raw_episode__')}"


def _finalize_rl4vla_raw_episode_path(item: EpisodeExecutionResult) -> Path | None:
    artifact_path = item.rl4vla_raw_episode_artifact_path
    if artifact_path is None:
        return None
    resolved_path = Path(artifact_path).expanduser().resolve()
    if not resolved_path.exists():
        return None
    target_name = _build_final_rl4vla_raw_episode_filename(item)
    if resolved_path.name == target_name:
        return resolved_path
    target_path = resolved_path.with_name(target_name)
    resolved_path.rename(target_path)
    return target_path


def _finalize_episode_runtime_request_artifacts(
    item: EpisodeExecutionResult,
) -> tuple[EpisodeExecutionResult, Dict[str, Any]]:
    runtime_request = _load_runtime_request_payload(item.request_path)
    outcome_token = _resolve_episode_outcome_token(item)
    rl4vla_raw_final = _finalize_rl4vla_raw_episode_path(item)
    debug_video_final = _finalize_episode_sidecar_path(
        runtime_request.get("debug_video_path"),
        outcome_token=outcome_token,
    )
    debug_video_gif_final = _finalize_episode_sidecar_path(
        runtime_request.get("debug_video_gif_path"),
        outcome_token=outcome_token,
    )
    replacements: Dict[str, str] = {}
    if item.rl4vla_raw_episode_artifact_path is not None and rl4vla_raw_final is not None:
        replacements[str(item.rl4vla_raw_episode_artifact_path)] = str(rl4vla_raw_final)
    if isinstance(runtime_request.get("debug_video_path"), str) and debug_video_final is not None:
        replacements[str(runtime_request["debug_video_path"])] = str(debug_video_final)
    if isinstance(runtime_request.get("debug_video_gif_path"), str) and debug_video_gif_final is not None:
        replacements[str(runtime_request["debug_video_gif_path"])] = str(debug_video_gif_final)
    if replacements:
        runtime_request = _rewrite_exact_string_values(runtime_request, replacements=replacements)
    runtime_request["rl4vla_raw_episode_artifact_path"] = (
        None if rl4vla_raw_final is None else str(rl4vla_raw_final)
    )
    runtime_request["debug_video_path"] = (
        None if debug_video_final is None else str(debug_video_final)
    )
    runtime_request["debug_video_gif_path"] = (
        None if debug_video_gif_final is None else str(debug_video_gif_final)
    )
    item.request_path.write_text(
        json.dumps(runtime_request, indent=2, sort_keys=True),
        encoding="utf-8",
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
            dense_episode_artifact_path=item.dense_episode_artifact_path,
            rl4vla_raw_episode_artifact_path=rl4vla_raw_final,
            rl4vla_success_export_path=item.rl4vla_success_export_path,
            rl4vla_success_export_index=item.rl4vla_success_export_index,
            debug_video_path=debug_video_final,
            debug_video_gif_path=debug_video_gif_final,
            object_pose_trace_path=item.object_pose_trace_path,
            batch_id=item.batch_id,
            compatibility_group_id=item.compatibility_group_id,
            env_index=item.env_index,
        ),
        runtime_request,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phased RC5 unified collector shell. "
            "This V0 entrypoint validates collector inputs, resolves placements, and writes "
            "a stable collection plan artifact. Runtime trajectory execution will be added in a later phase."
        )
    )
    parser.add_argument(
        "--motion_backend",
        default=PROXY_BACKEND,
        choices=UNIFIED_BACKENDS,
        help="Unified motion backend. Collector V0 currently supports only proxy_ee_delta.",
    )
    parser.add_argument(
        "--task_type",
        default=TASK_PICK_UP,
        choices=SUPPORTED_TASK_TYPES,
        help="Unified task type. Collector V0 currently supports only pick_up.",
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
        help="Scene path forwarded later to the unified runtime. Required for collection planning.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Collector output directory. V0 writes collection_plan.json here.",
    )
    parser.add_argument(
        "--task_object_id",
        default=None,
        help="Optional explicit task object id. Defaults to manip_object_id from config.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Planned batch size for later execution phases. Must be >= 1.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help=(
            "Run the prepared proxy runtime without a human viewer. "
            "Current headless support is intentionally narrow and intended only for the "
            "validated proxy_ee_delta pick_up collector recipe."
        ),
    )
    parser.add_argument(
        "--dense_episode_image_width",
        type=int,
        default=640,
        help=(
            "Stored dense episode canvas width. "
            "Frames keep base_camera aspect ratio, fit inside this canvas, and are padded to exact width x height."
        ),
    )
    parser.add_argument(
        "--dense_episode_image_height",
        type=int,
        default=480,
        help=(
            "Stored dense episode canvas height. "
            "Frames keep base_camera aspect ratio, fit inside this canvas, and are padded to exact width x height."
        ),
    )
    parser.add_argument(
        "--no_dense_episode_artifact",
        action="store_true",
        default=False,
        help=(
            "Do not persist dense_episode.npz. RL4VLA raw artifacts and debug GIF/video sidecars "
            "are still generated when their own outputs are enabled."
        ),
    )
    parser.add_argument(
        "--save_debug_video",
        action="store_true",
        default=False,
        help=(
            "Also save a full-resolution debug video sidecar per episode. "
            "This is separate from the resized dense RL4VLA-oriented image sequence."
        ),
    )
    parser.add_argument(
        "--save_debug_gif",
        action="store_true",
        default=False,
        help=(
            "Also save a GIF sidecar per episode derived from the buffered debug video."
        ),
    )
    parser.add_argument(
        "--embed_runtime_bundle_in_rl4vla_raw_npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Embed episode-specific runtime_config.yaml and runtime_request.json into "
            "rl4vla_raw_episode*.npz for self-contained proxy planner dataset artifacts."
        ),
    )
    parser.add_argument(
        "--save_object_pose_trace",
        action="store_true",
        default=False,
        help="Also save a per-episode JSONL trace with all scene object poses at every batched runtime step.",
    )
    parser.add_argument(
        "--rl4vla_success_output_dir",
        default=None,
        help=(
            "Optional flat output directory for success-only RL4VLA raw .npz files. "
            "Files are copied there after per-episode artifacts are finalized."
        ),
    )
    parser.add_argument(
        "--rl4vla_success_index_start",
        type=int,
        default=0,
        help=(
            "Global index assigned to the first successful episode copied to "
            "--rl4vla_success_output_dir. Use this to continue staged dataset runs."
        ),
    )
    parser.add_argument(
        "--placement_manifest",
        default=None,
        help="YAML/JSON manifest with explicit per-episode placements.",
    )
    parser.add_argument(
        "--placement_seed_start",
        type=int,
        default=None,
        help="Start seed for deterministic seed-based placement generation.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes for deterministic seed-based placement generation.",
    )
    parser.add_argument(
        "--seed_position_jitter_xy",
        type=float,
        default=0.05,
        help="Symmetric XY jitter radius used by deterministic seed-based placement generation.",
    )
    parser.add_argument(
        "--execute_prepared_requests",
        action="store_true",
        default=False,
        help="After preparing per-episode runtime inputs, run them sequentially through run_rc5_unified.py.",
    )
    parser.add_argument(
        "--exporter",
        default=None,
        choices=SUPPORTED_EXPORTERS,
        help="Optional exporter invoked per executed episode after episode_result.json is written.",
    )
    parser.add_argument(
        "--runtime_sim_patch",
        default=None,
        help=(
            "Optional YAML/JSON mapping merged into local.<key>.simulation during runtime "
            "config materialization. Use this for narrow collector-owned baseline recipes "
            "without modifying the source config."
        ),
    )
    return parser


def _build_bootstrap_passthrough(
    args: argparse.Namespace,
    passthrough_argv: Sequence[str],
) -> List[str]:
    argv = list(passthrough_argv)
    if not has_flag(argv, "--sim_backend"):
        argv.extend(["--sim_backend", DEFAULT_RC5_SIM_BACKEND])
    argv.extend(["--config_path", str(args.config_path)])
    if args.scene is not None:
        argv.extend(["--scene", str(args.scene)])
    if args.key is not None:
        argv.extend(["--key", str(args.key)])
    return argv


def _load_manifest(path_str: str | Path) -> List[Dict[str, Any]]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Placement manifest does not exist: {path}")
    data = load_runner_config(path)
    if "episodes" not in data:
        raise ValueError(f"Placement manifest must define a top-level 'episodes' list: {path}")
    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Placement manifest 'episodes' must be a non-empty list: {path}")
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(episodes):
        if not isinstance(item, dict):
            raise ValueError(f"Placement manifest episode #{idx} must be a mapping: {path}")
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
            "RC5Collection",
            "Missing task_semantic_name for object "
            f"{object_id}; falling back to object name '{str(display_name).strip().replace('_', ' ')}'.",
        )
        return str(display_name).strip().replace("_", " ")
    emit_warning(
        "RC5Collection",
        f"Missing task_semantic_name and name for object {object_id}; falling back to technical object id.",
    )
    return str(object_id)


@lru_cache(maxsize=32)
def _load_object_placements_for_config(config_path: str, key: str) -> Mapping[str, Any]:
    sections = load_simulation_config_sections(config_path, key)
    object_placements = pick_simulation_value(sections, "object_placements", None)
    if not isinstance(object_placements, dict) or not object_placements:
        raise ValueError(
            f"Collector requires object_placements in config for key '{key}' to resolve base placements."
        )
    return object_placements


def _resolve_base_object_placement(
    config_path: str | Path,
    key: str,
    object_id: str,
) -> Dict[str, Any]:
    config_path_str = str(Path(config_path).expanduser().resolve())
    object_placements = _load_object_placements_for_config(config_path_str, key)
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
    orientation = base_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])
    base_cfg["orientation"] = _require_orientation_quat(
        orientation,
        label=f"object_placements.{object_id}.orientation",
    )
    return base_cfg


def _build_seeded_placement(
    base_cfg: Mapping[str, Any],
    *,
    seed: int,
    jitter_xy: float,
) -> Dict[str, Any]:
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


def _resolve_manifest_object_placements_override(
    config_path: str | Path,
    key: str,
    episode_cfg: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]] | None:
    raw_override = episode_cfg.get("object_placements_override")
    if raw_override is None:
        return None
    override_mapping = _require_mapping(
        raw_override,
        label="manifest episode object_placements_override",
    )
    resolved: Dict[str, Dict[str, Any]] = {}
    for object_id, object_cfg in override_mapping.items():
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("manifest episode object_placements_override keys must be non-empty strings")
        base_cfg = _resolve_base_object_placement(config_path, key, object_id)
        explicit_cfg = _require_mapping(
            object_cfg,
            label=f"manifest episode object_placements_override.{object_id}",
        )
        merged = dict(base_cfg)
        merged.update(explicit_cfg)
        merged["position"] = _require_position_triplet(
            merged.get("position"),
            label=f"manifest episode object_placements_override.{object_id}.position",
        )
        merged["orientation"] = _require_orientation_quat(
            merged.get("orientation", [1.0, 0.0, 0.0, 0.0]),
            label=f"manifest episode object_placements_override.{object_id}.orientation",
        )
        resolved[object_id] = merged
    return resolved


def _resolve_episode_specs(
    args: argparse.Namespace,
    *,
    config_path: Path,
    key: str,
    task_object_id: str,
) -> tuple[EpisodePlacementSpec, ...]:
    if args.placement_manifest and args.placement_seed_start is not None:
        raise ValueError(
            "Choose exactly one placement source: --placement_manifest or "
            "--placement_seed_start/--num_episodes."
        )

    base_object_cfg = _resolve_base_object_placement(config_path, key, task_object_id)

    if args.placement_manifest:
        manifest_episodes = _load_manifest(args.placement_manifest)
        resolved = []
        for idx, item in enumerate(manifest_episodes):
            object_id = str(item.get("object_id", task_object_id))
            episode_base_object_cfg = _resolve_base_object_placement(config_path, key, object_id)
            object_placements_override = _resolve_manifest_object_placements_override(
                config_path,
                key,
                item,
            )
            placement = _merge_manifest_episode_placement(episode_base_object_cfg, item)
            if object_placements_override is not None:
                merged_target_override = dict(placement)
                merged_target_override.update(object_placements_override.get(object_id) or {})
                object_placements_override[object_id] = merged_target_override
            resolved.append(
                EpisodePlacementSpec(
                    episode_id=f"episode_{idx:06d}",
                    episode_index=idx,
                    object_id=object_id,
                    task_semantic_name=_resolve_task_semantic_name_from_object_cfg(
                        episode_base_object_cfg,
                        object_id=object_id,
                    ),
                    placement_source="manifest",
                    placement_seed=int(item["placement_seed"]) if "placement_seed" in item else None,
                    placement=placement,
                    object_placements_override=object_placements_override,
                )
            )
        return tuple(resolved)

    if args.placement_seed_start is None:
        raise ValueError(
            "Collector V0 requires a placement source. Provide --placement_manifest or "
            "--placement_seed_start together with --num_episodes."
        )
    if args.num_episodes is None:
        raise ValueError("--num_episodes is required when --placement_seed_start is used.")
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be >= 1")
    if args.seed_position_jitter_xy < 0:
        raise ValueError("--seed_position_jitter_xy must be >= 0")

    episodes = []
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
                object_placements_override=None,
            )
        )
    return tuple(episodes)


def resolve_collection_request(argv: Optional[Sequence[str]] = None) -> CollectionRequest:
    parser = _build_parser()
    args, passthrough_argv = parser.parse_known_args(list(argv or ()))

    if args.motion_backend != PROXY_BACKEND:
        raise ValueError(
            "rc5_unified_collection currently supports only motion_backend='proxy_ee_delta'. "
            f"Received '{args.motion_backend}'."
        )
    if args.task_type != TASK_PICK_UP:
        raise ValueError(
            "rc5_unified_collection currently supports only task_type='pick_up'. "
            f"Received '{args.task_type}'."
        )
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be >= 1")
    if args.dense_episode_image_width <= 0:
        raise ValueError("--dense_episode_image_width must be >= 1")
    if args.dense_episode_image_height <= 0:
        raise ValueError("--dense_episode_image_height must be >= 1")
    if args.headless and has_flag(passthrough_argv, "--step_by_step"):
        raise ValueError("--headless is incompatible with --step_by_step in collector passthrough args.")

    bootstrap_argv = _build_bootstrap_passthrough(args, passthrough_argv)
    bootstrap_request = resolve_unified_bootstrap_request(
        bootstrap_argv,
        default_config_path=args.config_path,
    )
    if bootstrap_request.scene_path is None:
        derived_key = detect_config_key(args.scene, args.key)
        raise ValueError(
            "Collector requires --scene and a resolvable config key. "
            f"Received scene={args.scene!r}, key={args.key!r}, derived_key={derived_key!r}."
        )
    validate_rc5_bootstrap_for_backend(bootstrap_request.bootstrap, args.motion_backend)

    sections = load_simulation_config_sections(
        bootstrap_request.config_path,
        bootstrap_request.key,
    )
    task_object_id = args.task_object_id or pick_simulation_value(
        sections,
        "manip_object_id",
        None,
    )
    if not task_object_id:
        raise ValueError(
            f"Collector requires --task_object_id or config manip_object_id for key '{bootstrap_request.key}'."
        )

    episodes = _resolve_episode_specs(
        args,
        config_path=bootstrap_request.config_path,
        key=bootstrap_request.key,
        task_object_id=str(task_object_id),
    )
    rl4vla_success_index_start = int(args.rl4vla_success_index_start)
    if rl4vla_success_index_start < 0:
        raise ValueError("--rl4vla_success_index_start must be >= 0")

    return CollectionRequest(
        motion_backend=args.motion_backend,
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
        save_dense_episode_artifact=not bool(args.no_dense_episode_artifact),
        save_debug_video=bool(args.save_debug_video),
        save_debug_gif=bool(args.save_debug_gif),
        save_object_pose_trace=bool(args.save_object_pose_trace),
        embed_runtime_bundle_in_rl4vla_raw_npz=bool(args.embed_runtime_bundle_in_rl4vla_raw_npz),
        rl4vla_success_output_dir=(
            None
            if args.rl4vla_success_output_dir is None
            else Path(args.rl4vla_success_output_dir).expanduser().resolve()
        ),
        rl4vla_success_index_start=rl4vla_success_index_start,
        runtime_sim_patch_path=(
            None if args.runtime_sim_patch is None else Path(args.runtime_sim_patch).expanduser().resolve()
        ),
        passthrough_argv=list(passthrough_argv),
        episodes=episodes,
    )


def write_collection_plan(request: CollectionRequest) -> Path:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = request.output_dir / "collection_plan.json"
    payload = {
        "phase": "runtime_inputs_prepared",
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
        "save_dense_episode_artifact": request.save_dense_episode_artifact,
        "save_debug_video": request.save_debug_video,
        "save_debug_gif": request.save_debug_gif,
        "save_object_pose_trace": request.save_object_pose_trace,
        "embed_runtime_bundle_in_rl4vla_raw_npz": request.embed_runtime_bundle_in_rl4vla_raw_npz,
        "runtime_sim_patch_path": (
            None if request.runtime_sim_patch_path is None else str(request.runtime_sim_patch_path)
        ),
        "episode_count": len(request.episodes),
        "episodes": [asdict(episode) for episode in request.episodes],
    }
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return plan_path


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
    if episode.object_placements_override is not None:
        for object_id, object_cfg in episode.object_placements_override.items():
            runtime_object_placements[object_id] = dict(object_cfg)
    else:
        runtime_object_placements[episode.object_id] = dict(episode.placement)
    runtime_sim["placement_mode"] = "fixed"
    runtime_sim["manip_object_id"] = episode.object_id
    runtime_sim["object_placements"] = runtime_object_placements
    if request.runtime_sim_patch_path is not None:
        _apply_runtime_sim_patch(
            runtime_sim,
            patch_path=request.runtime_sim_patch_path,
        )
    return runtime_config


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


def _build_episode_runner_argv(
    request: CollectionRequest,
    episode: EpisodePlacementSpec,
    runtime_config_path: Path,
    runtime_request_path: Path,
    *,
    dense_episode_artifact_path: Path | None,
    rl4vla_raw_episode_artifact_path: Path,
    debug_video_path: Path | None,
    debug_video_gif_path: Path | None,
) -> tuple[str, ...]:
    passthrough = list(request.passthrough_argv)
    if request.headless and not has_flag(passthrough, "--headless"):
        passthrough.append("--headless")
    if not has_flag(passthrough, "--sim_backend"):
        passthrough.extend(["--sim_backend", DEFAULT_RC5_SIM_BACKEND])
    artifact_args: List[str] = [
        "--rl4vla_raw_episode_output",
        str(rl4vla_raw_episode_artifact_path),
    ]
    if dense_episode_artifact_path is not None:
        artifact_args.extend(["--dense_episode_output", str(dense_episode_artifact_path)])

    return (
        "--motion_backend",
        request.motion_backend,
        "--task_type",
        request.task_type,
        "--task_object_id",
        episode.object_id,
        *tuple(artifact_args),
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


def _build_episode_object_pose_trace_path(episode_dir: Path) -> Path:
    return episode_dir / "object_pose_traces" / "scene_object_pose_trace.jsonl"


def _build_episode_instruction(request: CollectionRequest, *, task_semantic_name: str) -> str:
    if request.task_type == TASK_PICK_UP:
        return f"Pick up {task_semantic_name}."
    return f"{request.task_type}:{task_semantic_name}"


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

        dense_episode_artifact_path = (
            episode_dir / "dense_episode.npz" if request.save_dense_episode_artifact else None
        )
        rl4vla_raw_episode_artifact_path = episode_dir / "rl4vla_raw_episode.npz"
        debug_video_path = episode_dir / "debug_video.mkv" if request.save_debug_video else None
        debug_video_gif_path = episode_dir / "debug_video.gif" if request.save_debug_gif else None
        object_pose_trace_path = (
            _build_episode_object_pose_trace_path(episode_dir) if request.save_object_pose_trace else None
        )
        request_path = episode_dir / "runtime_request.json"
        runner_argv = _build_episode_runner_argv(
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
            "dense_episode_artifact_path": (
                None if dense_episode_artifact_path is None else str(dense_episode_artifact_path)
            ),
            "rl4vla_raw_episode_artifact_path": str(rl4vla_raw_episode_artifact_path),
            "dense_episode_instruction": _build_episode_instruction(
                request,
                task_semantic_name=episode.task_semantic_name,
            ),
            "dense_episode_image_width": request.dense_episode_image_width,
            "dense_episode_image_height": request.dense_episode_image_height,
            "debug_video_path": (None if debug_video_path is None else str(debug_video_path)),
            "debug_video_gif_path": (None if debug_video_gif_path is None else str(debug_video_gif_path)),
            "object_pose_trace_path": (
                None if object_pose_trace_path is None else str(object_pose_trace_path)
            ),
            "runtime_sim_patch_path": (
                None if request.runtime_sim_patch_path is None else str(request.runtime_sim_patch_path)
            ),
            "placement_source": episode.placement_source,
            "placement_seed": episode.placement_seed,
            "placement": dict(episode.placement),
            "object_placements_override": (
                None
                if episode.object_placements_override is None
                else copy.deepcopy(episode.object_placements_override)
            ),
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
        raise ValueError("Runtime config must define top-level 'local' mapping for collector batching.")
    key_cfg = local_cfg.get(key)
    if not isinstance(key_cfg, dict):
        raise ValueError(f"Runtime config must define local.{key} mapping for collector batching.")
    simulation_cfg = key_cfg.get("simulation")
    if not isinstance(simulation_cfg, dict):
        raise ValueError(f"Runtime config must define local.{key}.simulation mapping for collector batching.")
    return simulation_cfg


def _batching_artifacts_supported(request: CollectionRequest) -> bool:
    return True


def _normalize_object_placements_for_compatibility(
    object_placements: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    if not isinstance(object_placements, dict):
        return None
    normalized: Dict[str, Any] = {}
    for object_id in sorted(object_placements.keys()):
        object_cfg = object_placements.get(object_id)
        if not isinstance(object_cfg, dict):
            normalized[object_id] = object_cfg
            continue
        normalized_object_cfg = dict(object_cfg)
        if "position" in normalized_object_cfg:
            normalized_object_cfg["position"] = f"__collector_batch_position__:{object_id}"
        if "orientation" in normalized_object_cfg:
            normalized_object_cfg["orientation"] = f"__collector_batch_orientation__:{object_id}"
        normalized_object_cfg.pop("position_per_env", None)
        normalized_object_cfg.pop("orientation_per_env", None)
        normalized_object_cfg.pop("task_semantic_name", None)
        normalized[object_id] = normalized_object_cfg
    return normalized


def _build_hard_compatibility_fingerprint(
    request: CollectionRequest,
    *,
    runtime_payload: Mapping[str, Any],
    runtime_config_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized_runtime_config = copy.deepcopy(runtime_config_payload)
    runtime_sim = _resolve_runtime_config_simulation(
        normalized_runtime_config,
        key=request.key,
    )
    runtime_sim["object_placements"] = _normalize_object_placements_for_compatibility(
        runtime_sim.get("object_placements")
    )
    return {
        "motion_backend": request.motion_backend,
        "task_type": request.task_type,
        "task_object_id": str(runtime_payload.get("task_object_id", request.task_object_id)),
        "scene_path": request.scene_path,
        "headless": request.headless,
        "runtime_sim_patch_path": runtime_payload.get("runtime_sim_patch_path"),
        "runtime_config": normalized_runtime_config,
    }


def _compute_runtime_input_compatibility_group_id(
    request: CollectionRequest,
    runtime_input: EpisodeRuntimeInput,
) -> str:
    runtime_payload = _load_runtime_request_payload(runtime_input.request_path)
    runtime_config_payload = yaml.safe_load(runtime_input.config_path.read_text(encoding="utf-8"))
    fingerprint = _build_hard_compatibility_fingerprint(
        request,
        runtime_payload=runtime_payload,
        runtime_config_payload=runtime_config_payload,
    )
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
    debug_video_batch_output_dir: Path | None,
    debug_video_gif_batch_output_dir: Path | None,
    object_pose_trace_batch_output_dir: Path | None,
) -> tuple[str, ...]:
    runner_argv = runtime_inputs[0].runner_argv
    batch_argv = _strip_flag_with_value(runner_argv, "--num_envs")
    batch_argv = _strip_flag_with_value(batch_argv, "--config_path")
    batch_argv.extend(["--num_envs", str(int(env_count))])
    if int(env_count) > 1:
        batch_argv = _strip_flag_with_value(batch_argv, "--runtime_request_path")
        batch_argv = _strip_flag_with_value(batch_argv, "--runtime_config_path_per_env_json")
        batch_argv = _strip_flag_with_value(batch_argv, "--runtime_request_path_per_env_json")
        batch_argv = _strip_flag_with_value(batch_argv, "--dense_episode_output")
        batch_argv = _strip_flag_with_value(batch_argv, "--dense_episode_output_dir")
        batch_argv = _strip_flag_with_value(batch_argv, "--rl4vla_raw_episode_output")
        batch_argv = _strip_flag_with_value(batch_argv, "--rl4vla_raw_episode_output_dir")
        batch_argv = _strip_flag_with_value(batch_argv, "--episode_instruction_per_env_json")
        batch_argv = _strip_flag_with_value(batch_argv, "--save_video_path")
        batch_argv = _strip_flag_with_value(batch_argv, "--save_video_gif_path")
        batch_argv = _strip_flag_with_value(batch_argv, "--batched_save_video_output_dir")
        batch_argv = _strip_flag_with_value(batch_argv, "--batched_save_video_gif_output_dir")
        batch_argv = _strip_flag_with_value(batch_argv, "--batched_object_pose_trace_output_dir")
        batch_argv = [item for item in batch_argv if item != "--save_video_on_exit"]
        batch_argv = [item for item in batch_argv if item != "--save_video_gif_on_exit"]
        if dense_batch_output_dir is not None:
            batch_argv.extend(["--dense_episode_output_dir", str(dense_batch_output_dir)])
        if rl4vla_raw_batch_output_dir is not None:
            batch_argv.extend(["--rl4vla_raw_episode_output_dir", str(rl4vla_raw_batch_output_dir)])
        per_env_instructions = [
            str(_load_runtime_request_payload(item.request_path)["dense_episode_instruction"])
            for item in runtime_inputs
        ]
        batch_argv.extend(["--episode_instruction_per_env_json", json.dumps(per_env_instructions)])
        batch_argv.extend(
            [
                "--runtime_config_path_per_env_json",
                json.dumps([str(item.config_path) for item in runtime_inputs]),
                "--runtime_request_path_per_env_json",
                json.dumps([str(item.request_path) for item in runtime_inputs]),
            ]
        )
        if debug_video_batch_output_dir is not None:
            batch_argv.extend(["--batched_save_video_output_dir", str(debug_video_batch_output_dir)])
        if debug_video_gif_batch_output_dir is not None:
            batch_argv.extend(["--batched_save_video_gif_output_dir", str(debug_video_gif_batch_output_dir)])
        if object_pose_trace_batch_output_dir is not None:
            batch_argv.extend(["--batched_object_pose_trace_output_dir", str(object_pose_trace_batch_output_dir)])
    else:
        batch_argv.extend(["--runtime_request_path", str(runtime_inputs[0].request_path)])
    batch_argv.extend(["--config_path", str(batch_config_path)])
    return tuple(str(item) for item in batch_argv)


def _resolve_runtime_config_key(runtime_config_payload: Mapping[str, Any]) -> str | None:
    keys_value = runtime_config_payload.get("keys")
    if isinstance(keys_value, list) and keys_value:
        first_key = keys_value[0]
        if isinstance(first_key, str) and first_key.strip():
            return first_key
    local_cfg = runtime_config_payload.get("local")
    if isinstance(local_cfg, dict):
        for key in local_cfg.keys():
            if isinstance(key, str) and key.strip():
                return key
    return None


def _resolve_dense_episode_visible_frames(payload: Mapping[str, Any]) -> np.ndarray:
    frames = decode_dense_episode_images(payload)
    if len(frames) == 0:
        raise ValueError("dense episode artifact must contain at least one frame")

    original_hw = payload.get("image_original_hw")
    stored_hw = payload.get("image_stored_hw")
    if (
        not isinstance(original_hw, (list, tuple))
        or len(original_hw) != 2
        or not isinstance(stored_hw, (list, tuple))
        or len(stored_hw) != 2
    ):
        return frames

    original_h = int(original_hw[0])
    original_w = int(original_hw[1])
    stored_h = int(stored_hw[0])
    stored_w = int(stored_hw[1])
    if original_h <= 0 or original_w <= 0 or stored_h <= 0 or stored_w <= 0:
        return frames

    scale = min(float(stored_w) / float(original_w), float(stored_h) / float(original_h))
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))
    offset_x = max(0, (stored_w - resized_w) // 2)
    offset_y = max(0, (stored_h - resized_h) // 2)
    end_x = min(stored_w, offset_x + resized_w)
    end_y = min(stored_h, offset_y + resized_h)
    if end_x <= offset_x or end_y <= offset_y:
        return frames
    visible_frames = frames[:, offset_y:end_y, offset_x:end_x, :]
    if visible_frames.shape[1] == original_h and visible_frames.shape[2] == original_w:
        return visible_frames

    restored_frames = [
        np.asarray(
            Image.fromarray(frame, mode="RGB").resize(
                (original_w, original_h),
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        for frame in visible_frames
    ]
    return np.asarray(restored_frames, dtype=np.uint8)


def _resolve_batched_debug_video_fps(payload: Mapping[str, Any]) -> int:
    runtime_config_path_value = payload.get("runtime_config_path")
    if not isinstance(runtime_config_path_value, str) or not runtime_config_path_value.strip():
        return DEFAULT_VIDEO_FPS
    runtime_config_path = Path(runtime_config_path_value).expanduser().resolve()
    if not runtime_config_path.exists():
        return DEFAULT_VIDEO_FPS
    runtime_config_payload = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
    if not isinstance(runtime_config_payload, dict):
        return DEFAULT_VIDEO_FPS
    runtime_key = _resolve_runtime_config_key(runtime_config_payload)
    if runtime_key is None:
        return DEFAULT_VIDEO_FPS
    try:
        runtime_sim = _resolve_runtime_config_simulation(runtime_config_payload, key=runtime_key)
    except Exception:
        return DEFAULT_VIDEO_FPS
    video_fps = runtime_sim.get("video_fps", DEFAULT_VIDEO_FPS)
    try:
        resolved = int(video_fps)
    except Exception:
        return DEFAULT_VIDEO_FPS
    return resolved if resolved > 0 else DEFAULT_VIDEO_FPS


def _write_debug_video_from_dense_episode_artifact(
    dense_episode_artifact_path: Path,
    video_path: Path,
    *,
    video_fps: int,
) -> Path:
    payload = load_dense_episode_artifact(dense_episode_artifact_path)
    frames = _resolve_dense_episode_visible_frames(payload)
    if len(frames) == 0:
        raise ValueError(f"dense episode artifact must contain at least one frame: {dense_episode_artifact_path}")
    resolved_video_path = video_path.expanduser().resolve()
    ok = flush_video_buffer_to_file(
        frames,
        resolved_video_path,
        int(video_fps),
        video_codec="ffv1",
    )
    if not ok:
        raise RuntimeError(f"Failed to save debug video from dense episode artifact: {resolved_video_path}")
    return resolved_video_path


def _write_debug_gif_from_dense_episode_artifact(
    dense_episode_artifact_path: Path,
    gif_path: Path,
    *,
    video_fps: int,
) -> Path:
    temp_video_path = gif_path.expanduser().resolve().with_suffix(".tmp_debug_video.mkv")
    try:
        _write_debug_video_from_dense_episode_artifact(
            dense_episode_artifact_path,
            temp_video_path,
            video_fps=int(video_fps),
        )
        write_debug_video_gif_from_video(temp_video_path, gif_path)
    finally:
        if temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except Exception as exc:
                print(
                    "[WARNING] [VIDEO] Failed to remove temporary GIF source video: "
                    f"{temp_video_path} ({exc})"
                )
    return gif_path


def _write_batched_debug_sidecars_if_needed(
    payload: Mapping[str, Any],
    *,
    batch: PreparedExecutionBatch,
    env_index: int,
    dense_episode_artifact_path: Path | None,
) -> tuple[Path | None, Path | None]:
    raw_video_path = payload.get("debug_video_path")
    resolved_video_path = _resolve_batched_debug_sidecar_artifact_path(
        raw_video_path,
        batch_output_dir=batch.debug_video_batch_output_dir,
        env_index=env_index,
        resolver=resolve_batched_debug_video_output_path,
    )

    raw_gif_path = payload.get("debug_video_gif_path")
    resolved_gif_path = _resolve_batched_debug_sidecar_artifact_path(
        raw_gif_path,
        batch_output_dir=batch.debug_video_gif_batch_output_dir,
        env_index=env_index,
        resolver=resolve_batched_debug_video_gif_output_path,
    )

    if (
        (resolved_video_path is not None or resolved_gif_path is not None)
        or dense_episode_artifact_path is None
        or not dense_episode_artifact_path.exists()
    ):
        return resolved_video_path, resolved_gif_path

    video_fps = _resolve_batched_debug_video_fps(payload)

    if resolved_video_path is None and isinstance(raw_video_path, str) and raw_video_path.strip():
        resolved_video_path = _write_debug_video_from_dense_episode_artifact(
            dense_episode_artifact_path,
            Path(raw_video_path).expanduser().resolve(),
            video_fps=video_fps,
        )

    if resolved_gif_path is None and isinstance(raw_gif_path, str) and raw_gif_path.strip():
        resolved_gif_path = _write_debug_gif_from_dense_episode_artifact(
            dense_episode_artifact_path,
            Path(raw_gif_path).expanduser().resolve(),
            video_fps=video_fps,
        )
    return resolved_video_path, resolved_gif_path


def _materialize_batched_runtime_config(
    request: CollectionRequest,
    runtime_inputs: Sequence[EpisodeRuntimeInput],
    *,
    output_dir: Path,
) -> Path:
    if not runtime_inputs:
        raise ValueError("Collector batching requires at least one runtime input.")
    base_runtime_config = yaml.safe_load(runtime_inputs[0].config_path.read_text(encoding="utf-8"))
    runtime_sim = _resolve_runtime_config_simulation(base_runtime_config, key=request.key)
    object_placements = runtime_sim.get("object_placements")
    if not isinstance(object_placements, dict):
        raise ValueError("Collector batched runtime config requires simulation.object_placements mapping.")
    per_object_positions: Dict[str, List[List[float]]] = {}
    per_object_orientations: Dict[str, List[List[float]]] = {}
    touched_object_ids: set[str] = set()
    for runtime_input in runtime_inputs:
        runtime_payload = _load_runtime_request_payload(runtime_input.request_path)
        override_cfg = runtime_payload.get("object_placements_override")
        if override_cfg is not None:
            override_mapping = _require_mapping(
                override_cfg,
                label=f"{runtime_input.episode_id}.object_placements_override",
            )
            episode_object_ids = set(override_mapping.keys())
            for object_id, object_cfg in override_mapping.items():
                placement_cfg = _require_mapping(
                    object_cfg,
                    label=f"{runtime_input.episode_id}.object_placements_override.{object_id}",
                )
                base_cfg = _require_mapping(
                    object_placements.get(object_id),
                    label=f"object_placements.{object_id}",
                )
                per_object_positions.setdefault(object_id, []).append(
                    _require_position_triplet(
                        placement_cfg.get("position"),
                        label=f"{runtime_input.episode_id}.object_placements_override.{object_id}.position",
                    )
                )
                per_object_orientations.setdefault(object_id, []).append(
                    _require_orientation_quat(
                        placement_cfg.get("orientation", base_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])),
                        label=f"{runtime_input.episode_id}.object_placements_override.{object_id}.orientation",
                    )
                )
            if touched_object_ids and episode_object_ids != touched_object_ids:
                raise ValueError(
                    "Collector batched runtime config requires identical object_placements_override keys "
                    f"within a batch. batch_object_ids={sorted(touched_object_ids)} "
                    f"episode_object_ids={sorted(episode_object_ids)}"
                )
            touched_object_ids = episode_object_ids
            continue

        placement_cfg = _require_mapping(
            runtime_payload.get("placement"),
            label=f"{runtime_input.episode_id}.placement",
        )
        episode_task_object_id = str(runtime_payload.get("task_object_id", request.task_object_id))
        task_object_cfg = _require_mapping(
            object_placements.get(episode_task_object_id),
            label=f"object_placements.{episode_task_object_id}",
        )
        per_object_positions.setdefault(episode_task_object_id, []).append(
            _require_position_triplet(
                placement_cfg.get("position"),
                label=f"{runtime_input.episode_id}.placement.position",
            )
        )
        per_object_orientations.setdefault(episode_task_object_id, []).append(
            _require_orientation_quat(
                placement_cfg.get("orientation", task_object_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0])),
                label=f"{runtime_input.episode_id}.placement.orientation",
            )
        )
        touched_object_ids = {episode_task_object_id}

    for object_id in sorted(touched_object_ids):
        task_object_cfg = _require_mapping(
            object_placements.get(object_id),
            label=f"object_placements.{object_id}",
        )
        batch_task_object_cfg = dict(task_object_cfg)
        batch_task_object_cfg["position"] = list(per_object_positions[object_id][0])
        batch_task_object_cfg["orientation"] = list(per_object_orientations[object_id][0])
        batch_task_object_cfg["position_per_env"] = per_object_positions[object_id]
        batch_task_object_cfg["orientation_per_env"] = per_object_orientations[object_id]
        object_placements[object_id] = batch_task_object_cfg

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

    batching_enabled = int(request.num_envs) > 1 and _batching_artifacts_supported(request)
    batches_root = request.output_dir / "batches"
    planned: List[PreparedExecutionBatch] = []

    def _append_batch(
        compatibility_group_id: str,
        current_inputs: Sequence[EpisodeRuntimeInput],
    ) -> None:
        batch_index = len(planned)
        batch_id = f"batch_{batch_index:06d}"
        batch_dir = batches_root / batch_id
        dense_batch_output_dir = None
        rl4vla_raw_batch_output_dir = None
        debug_video_batch_output_dir = None
        debug_video_gif_batch_output_dir = None
        object_pose_trace_batch_output_dir = None
        batch_config_path = current_inputs[0].config_path
        if len(current_inputs) > 1:
            batch_config_path = _materialize_batched_runtime_config(
                request,
                current_inputs,
                output_dir=batch_dir,
            )
            if request.save_dense_episode_artifact:
                dense_batch_output_dir = batch_dir / "dense_episodes"
                dense_batch_output_dir.mkdir(parents=True, exist_ok=True)
            rl4vla_raw_batch_output_dir = batch_dir / "rl4vla_raw_episodes"
            rl4vla_raw_batch_output_dir.mkdir(parents=True, exist_ok=True)
            if request.save_debug_video:
                debug_video_batch_output_dir = batch_dir / "debug_videos"
                debug_video_batch_output_dir.mkdir(parents=True, exist_ok=True)
            if request.save_debug_gif:
                debug_video_gif_batch_output_dir = batch_dir / "debug_video_gifs"
                debug_video_gif_batch_output_dir.mkdir(parents=True, exist_ok=True)
            if request.save_object_pose_trace:
                object_pose_trace_batch_output_dir = batch_dir / "object_pose_traces"
                object_pose_trace_batch_output_dir.mkdir(parents=True, exist_ok=True)
        planned.append(
            PreparedExecutionBatch(
                batch_id=batch_id,
                compatibility_group_id=str(compatibility_group_id),
                env_count=len(current_inputs),
                runtime_inputs=tuple(current_inputs),
                runner_argv=_build_batch_runner_argv(
                    current_inputs,
                    env_count=len(current_inputs),
                    batch_config_path=batch_config_path,
                    dense_batch_output_dir=dense_batch_output_dir,
                    rl4vla_raw_batch_output_dir=rl4vla_raw_batch_output_dir,
                    debug_video_batch_output_dir=debug_video_batch_output_dir,
                    debug_video_gif_batch_output_dir=debug_video_gif_batch_output_dir,
                    object_pose_trace_batch_output_dir=object_pose_trace_batch_output_dir,
                ),
                batch_config_path=batch_config_path,
                dense_batch_output_dir=dense_batch_output_dir,
                rl4vla_raw_batch_output_dir=rl4vla_raw_batch_output_dir,
                debug_video_batch_output_dir=debug_video_batch_output_dir,
                debug_video_gif_batch_output_dir=debug_video_gif_batch_output_dir,
                object_pose_trace_batch_output_dir=object_pose_trace_batch_output_dir,
            )
        )

    if not batching_enabled:
        for runtime_input in runtime_inputs:
            _append_batch(f"singleton:{runtime_input.episode_id}", [runtime_input])
        return tuple(planned)

    grouped_inputs: Dict[str, List[EpisodeRuntimeInput]] = {}
    group_order: List[str] = []
    for runtime_input in runtime_inputs:
        compatibility_group_id = _compute_runtime_input_compatibility_group_id(request, runtime_input)
        if compatibility_group_id not in grouped_inputs:
            grouped_inputs[compatibility_group_id] = []
            group_order.append(compatibility_group_id)
        grouped_inputs[compatibility_group_id].append(runtime_input)

    batch_size = int(request.num_envs)
    for compatibility_group_id in group_order:
        group_runtime_inputs = grouped_inputs[compatibility_group_id]
        for start_index in range(0, len(group_runtime_inputs), batch_size):
            _append_batch(
                compatibility_group_id,
                group_runtime_inputs[start_index : start_index + batch_size],
            )
    return tuple(planned)


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
    return target_path


def _resolve_rl4vla_raw_episode_artifact_path(
    payload: Mapping[str, Any],
    *,
    batch: PreparedExecutionBatch,
    env_index: int,
) -> Path | None:
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
    return target_path


def _resolve_object_pose_trace_path(
    payload: Mapping[str, Any],
    *,
    batch: PreparedExecutionBatch,
    env_index: int,
) -> Path | None:
    raw_target = payload.get("object_pose_trace_path")
    target_path = None if raw_target is None else Path(str(raw_target)).expanduser().resolve()
    if batch.env_count <= 1 or batch.object_pose_trace_batch_output_dir is None:
        return target_path
    source_path = resolve_batched_object_pose_trace_output_path(
        batch.object_pose_trace_batch_output_dir,
        env_index=env_index,
    )
    if not source_path.exists():
        return None if target_path is None or not target_path.exists() else target_path
    if target_path is None:
        return source_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(target_path)
    return target_path


def _resolve_batched_debug_sidecar_artifact_path(
    raw_target: Any,
    *,
    batch_output_dir: Path | None,
    env_index: int,
    resolver,
) -> Path | None:
    target_path = None if raw_target is None else Path(str(raw_target)).expanduser().resolve()
    if batch_output_dir is None:
        return None if target_path is None or not target_path.exists() else target_path
    source_path = resolver(batch_output_dir, env_index=env_index)
    if not source_path.exists():
        return None if target_path is None or not target_path.exists() else target_path
    if target_path is None:
        return source_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(target_path)
    return target_path


def execute_prepared_execution_batches(
    batches: Sequence[PreparedExecutionBatch],
    *,
    runner=None,
) -> tuple[EpisodeExecutionResult, ...]:
    if runner is None:
        runner = run_rc5_unified_execute
    executed: List[EpisodeExecutionResult] = []
    for batch in batches:
        runtime_exit_code, trace_record = _normalize_runner_result(runner(list(batch.runner_argv)))
        per_env_feedback = _extract_trace_batch_feedback(trace_record)
        per_env_by_index = {
            int(item["env_index"]): item
            for item in per_env_feedback
            if isinstance(item.get("env_index"), int)
        }
        batch_semantic_feedback = _extract_semantic_feedback(trace_record)
        if batch.env_count > 1 and runtime_exit_code == 0 and len(per_env_by_index) != batch.env_count:
            raise ValueError(
                "Collector batched execution requires per-env runtime feedback for every env in the batch. "
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
            dense_episode_artifact_path = _resolve_dense_episode_artifact_path(
                payload,
                batch=batch,
                env_index=env_index,
            )
            rl4vla_raw_episode_artifact_path = _resolve_rl4vla_raw_episode_artifact_path(
                payload,
                batch=batch,
                env_index=env_index,
            )
            object_pose_trace_path = _resolve_object_pose_trace_path(
                payload,
                batch=batch,
                env_index=env_index,
            )
            if batch.env_count > 1:
                _write_batched_debug_sidecars_if_needed(
                    payload,
                    batch=batch,
                    env_index=env_index,
                    dense_episode_artifact_path=dense_episode_artifact_path,
                )
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
                    dense_episode_artifact_path=dense_episode_artifact_path,
                    rl4vla_raw_episode_artifact_path=rl4vla_raw_episode_artifact_path,
                    object_pose_trace_path=object_pose_trace_path,
                    batch_id=batch.batch_id,
                    compatibility_group_id=batch.compatibility_group_id,
                    env_index=env_index,
                )
            )
        executed.extend(batch_results)
    return tuple(executed)


def execute_prepared_runtime_inputs(
    runtime_inputs: Sequence[EpisodeRuntimeInput],
    *,
    runner=None,
) -> tuple[EpisodeExecutionResult, ...]:
    if runner is None:
        runner = run_rc5_unified_execute
    executed: List[EpisodeExecutionResult] = []
    for runtime_input in runtime_inputs:
        payload = _load_runtime_request_payload(runtime_input.request_path)
        runner_argv = [str(item) for item in payload["runner_argv"]]
        runtime_exit_code, trace_record = _normalize_runner_result(runner(runner_argv))
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
            dense_episode_artifact_path=(
                None
                if payload.get("dense_episode_artifact_path") is None
                else Path(str(payload["dense_episode_artifact_path"])).expanduser().resolve()
            ),
            rl4vla_raw_episode_artifact_path=(
                None
                if payload.get("rl4vla_raw_episode_artifact_path") is None
                else Path(str(payload["rl4vla_raw_episode_artifact_path"])).expanduser().resolve()
            ),
            object_pose_trace_path=(
                None
                if payload.get("object_pose_trace_path") is None
                else Path(str(payload["object_pose_trace_path"])).expanduser().resolve()
            ),
        )
        executed.append(result)
    return tuple(executed)


def _normalize_runner_result(raw_result: Any) -> tuple[int, Any]:
    if raw_result is None or isinstance(raw_result, int):
        exit_code = 0 if raw_result is None else int(raw_result)
        return exit_code, None
    return int(getattr(raw_result, "exit_code")), getattr(raw_result, "trace_record", None)


def _extract_trace_events(trace_record: Any) -> List[Any]:
    if trace_record is None:
        return []
    if isinstance(trace_record, Mapping):
        events = trace_record.get("events", [])
    else:
        events = getattr(trace_record, "events", [])
    return list(events or [])


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


def write_episode_result_artifacts(
    request: CollectionRequest,
    execution_results: Sequence[EpisodeExecutionResult],
) -> tuple[EpisodeExecutionResult, ...]:
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
                rl4vla_success_export_path=finalized_item.rl4vla_success_export_path,
                rl4vla_success_export_index=finalized_item.rl4vla_success_export_index,
                debug_video_path=finalized_item.debug_video_path,
                debug_video_gif_path=finalized_item.debug_video_gif_path,
                object_pose_trace_path=finalized_item.object_pose_trace_path,
                batch_id=finalized_item.batch_id,
                compatibility_group_id=finalized_item.compatibility_group_id,
                env_index=finalized_item.env_index,
            )
        )
    return tuple(updated)


def copy_rl4vla_success_artifacts_to_flat_dir(
    request: CollectionRequest,
    execution_results: Sequence[EpisodeExecutionResult],
) -> tuple[EpisodeExecutionResult, ...]:
    output_dir = request.rl4vla_success_output_dir
    if output_dir is None:
        return tuple(execution_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    updated: List[EpisodeExecutionResult] = []
    exported_count = 0
    for item in execution_results:
        export_path = item.rl4vla_success_export_path
        export_index = item.rl4vla_success_export_index
        if item.success and item.rl4vla_raw_episode_artifact_path is not None:
            source_path = Path(item.rl4vla_raw_episode_artifact_path).expanduser().resolve()
            if source_path.exists():
                export_index = int(request.rl4vla_success_index_start) + exported_count
                target_name = _build_rl4vla_success_export_filename(item, global_index=export_index)
                export_path = output_dir / target_name
                if export_path.exists():
                    raise FileExistsError(
                        "Collector cannot export RL4VLA success artifact because target already exists: "
                        f"{export_path}"
                    )
                shutil.copy2(source_path, export_path)
                exported_count += 1
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
                export_manifest_path=item.export_manifest_path,
                dense_episode_artifact_path=item.dense_episode_artifact_path,
                rl4vla_raw_episode_artifact_path=item.rl4vla_raw_episode_artifact_path,
                rl4vla_success_export_path=export_path,
                rl4vla_success_export_index=export_index,
                debug_video_path=item.debug_video_path,
                debug_video_gif_path=item.debug_video_gif_path,
                object_pose_trace_path=item.object_pose_trace_path,
                batch_id=item.batch_id,
                compatibility_group_id=item.compatibility_group_id,
                env_index=item.env_index,
            )
        )
    return tuple(updated)


def write_episode_export_artifacts(
    execution_results: Sequence[EpisodeExecutionResult],
    *,
    exporter_name: str,
) -> tuple[EpisodeExecutionResult, ...]:
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
                rl4vla_success_export_path=item.rl4vla_success_export_path,
                rl4vla_success_export_index=item.rl4vla_success_export_index,
                debug_video_path=item.debug_video_path,
                debug_video_gif_path=item.debug_video_gif_path,
                object_pose_trace_path=item.object_pose_trace_path,
                batch_id=item.batch_id,
                compatibility_group_id=item.compatibility_group_id,
                env_index=item.env_index,
            )
        )
    return tuple(updated)


def _rewrite_embedded_episode_paths(value: Any, *, old_dir: Path, new_dir: Path) -> Any:
    old_prefix = str(old_dir)
    new_prefix = str(new_dir)
    if isinstance(value, str):
        return new_prefix + value[len(old_prefix) :] if value.startswith(old_prefix) else value
    if isinstance(value, list):
        return [_rewrite_embedded_episode_paths(item, old_dir=old_dir, new_dir=new_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_embedded_episode_paths(item, old_dir=old_dir, new_dir=new_dir)
            for key, item in value.items()
        }
    return value


def _rewrite_json_file_embedded_episode_paths(path: Path, *, old_dir: Path, new_dir: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rewritten = _rewrite_embedded_episode_paths(payload, old_dir=old_dir, new_dir=new_dir)
    path.write_text(json.dumps(rewritten, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_episode_json_artifacts_under_directory(*, episode_dir: Path, old_dir: Path, new_dir: Path) -> None:
    for json_path in sorted(episode_dir.rglob("*.json")):
        _rewrite_json_file_embedded_episode_paths(json_path, old_dir=old_dir, new_dir=new_dir)


def _relocate_episode_output_directories(
    execution_results: Sequence[EpisodeExecutionResult],
) -> tuple[EpisodeExecutionResult, ...]:
    relocated: List[EpisodeExecutionResult] = []
    for item in execution_results:
        old_episode_dir = item.request_path.parent
        episodes_root = old_episode_dir.parent
        bucket_dir = episodes_root / ("success" if item.success else "fail")
        new_episode_dir = bucket_dir / old_episode_dir.name
        if old_episode_dir == new_episode_dir:
            relocated.append(item)
            continue
        bucket_dir.mkdir(parents=True, exist_ok=True)
        if new_episode_dir.exists():
            raise FileExistsError(
                "Collector cannot relocate episode output because target directory already exists: "
                f"{new_episode_dir}"
            )
        old_episode_dir.rename(new_episode_dir)
        try:
            old_episode_dir.symlink_to(new_episode_dir, target_is_directory=True)
        except Exception as exc:
            print(
                "[WARNING] [RC5Collection] Failed to create backward-compatible episode symlink: "
                f"{old_episode_dir} -> {new_episode_dir} ({exc})"
            )
        _rewrite_episode_json_artifacts_under_directory(
            episode_dir=new_episode_dir,
            old_dir=old_episode_dir,
            new_dir=new_episode_dir,
        )

        def _move_path(path_value: Path | None) -> Path | None:
            if path_value is None:
                return None
            raw = str(path_value)
            old_prefix = str(old_episode_dir)
            if raw.startswith(old_prefix):
                return Path(str(new_episode_dir) + raw[len(old_prefix) :])
            return path_value

        relocated.append(
            EpisodeExecutionResult(
                episode_id=item.episode_id,
                episode_index=item.episode_index,
                request_path=_move_path(item.request_path),
                exit_code=item.exit_code,
                runtime_exit_code=item.runtime_exit_code,
                execution_outcome=item.execution_outcome,
                success=item.success,
                semantic_task_success=item.semantic_task_success,
                failed_stage=item.failed_stage,
                trace_record=item.trace_record,
                result_path=_move_path(item.result_path),
                export_manifest_path=_move_path(item.export_manifest_path),
                dense_episode_artifact_path=_move_path(item.dense_episode_artifact_path),
                rl4vla_raw_episode_artifact_path=_move_path(item.rl4vla_raw_episode_artifact_path),
                rl4vla_success_export_path=item.rl4vla_success_export_path,
                rl4vla_success_export_index=item.rl4vla_success_export_index,
                debug_video_path=_move_path(item.debug_video_path),
                debug_video_gif_path=_move_path(item.debug_video_gif_path),
                object_pose_trace_path=_move_path(item.object_pose_trace_path),
                batch_id=item.batch_id,
                compatibility_group_id=item.compatibility_group_id,
                env_index=item.env_index,
            )
        )
    return tuple(relocated)


def write_execution_summary(
    request: CollectionRequest,
    execution_results: Sequence[EpisodeExecutionResult],
    *,
    execution_batches: Sequence[PreparedExecutionBatch] = (),
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
            "rl4vla_success_export_path": (
                None
                if item.rl4vla_success_export_path is None
                else str(item.rl4vla_success_export_path)
            ),
            "rl4vla_success_export_index": item.rl4vla_success_export_index,
        }
        for item in execution_results
    ]
    fail_result = next((item for item in execution_results if item.exit_code != 0), None)
    executed_batch_ids = {item.batch_id for item in execution_results if item.batch_id is not None}
    batched_execution_used = any(int(batch.env_count) > 1 for batch in execution_batches)
    stopped_on_failure = len(execution_results) < len(request.episodes)
    planned_env_capacity = len(execution_batches) * int(request.num_envs)
    compatibility_group_ids = {batch.compatibility_group_id for batch in execution_batches}
    exported_rl4vla_success_indices = [
        int(item.rl4vla_success_export_index)
        for item in execution_results
        if item.rl4vla_success_export_index is not None
    ]
    payload = {
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
        "stopped_on_failure": stopped_on_failure,
        "failed_episode_id": None if fail_result is None else fail_result.episode_id,
        "failed_exit_code": None if fail_result is None else fail_result.exit_code,
        "rl4vla_success_export": {
            "enabled": request.rl4vla_success_output_dir is not None,
            "output_dir": (
                None
                if request.rl4vla_success_output_dir is None
                else str(request.rl4vla_success_output_dir)
            ),
            "index_start": int(request.rl4vla_success_index_start),
            "exported_count": len(exported_rl4vla_success_indices),
            "next_index": int(request.rl4vla_success_index_start)
            + len(exported_rl4vla_success_indices),
            "index_min": (
                None if not exported_rl4vla_success_indices else min(exported_rl4vla_success_indices)
            ),
            "index_max": (
                None if not exported_rl4vla_success_indices else max(exported_rl4vla_success_indices)
            ),
        },
        "batching": {
            "requested_num_envs": request.num_envs,
            "batched_execution_used": batched_execution_used,
            "batch_count_planned": len(execution_batches),
            "batch_count_executed": len(executed_batch_ids),
            "compatibility_group_count_planned": len(compatibility_group_ids),
            "max_planned_batch_size": (
                0 if not execution_batches else max(int(batch.env_count) for batch in execution_batches)
            ),
            "planned_env_capacity": planned_env_capacity,
            "planned_env_fill_ratio": (
                0.0
                if planned_env_capacity <= 0
                else float(len(request.episodes)) / float(planned_env_capacity)
            ),
            "artifact_batching_supported": _batching_artifacts_supported(request),
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
            }
            for batch in execution_batches
        ],
        "episodes": completed,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _run_collection_main(args, request: CollectionRequest) -> int:
    runtime_inputs = materialize_episode_runtime_inputs(request)
    plan_path = write_collection_plan(request)
    execution_summary_path = None
    exit_code = 0
    if args.execute_prepared_requests:
        execution_batches = plan_prepared_execution_batches(request, runtime_inputs)
        execution_results = execute_prepared_execution_batches(execution_batches)
        execution_results = write_episode_result_artifacts(request, execution_results)
        execution_results = copy_rl4vla_success_artifacts_to_flat_dir(request, execution_results)
        if args.exporter is not None:
            execution_results = write_episode_export_artifacts(
                execution_results,
                exporter_name=str(args.exporter),
            )
        execution_results = _relocate_episode_output_directories(execution_results)
        execution_summary_path = write_execution_summary(
            request,
            execution_results,
            execution_batches=execution_batches,
        )
        fail_result = next((item for item in execution_results if item.exit_code != 0), None)
        exit_code = 0 if fail_result is None else int(fail_result.exit_code)
    print(
        f"[RC5Collection] phase={'runtime_execution' if args.execute_prepared_requests else 'runtime_inputs_prepared'} "
        f"motion_backend={request.motion_backend} "
        f"task_type={request.task_type} "
        f"key={request.key} "
        f"scene={request.scene_path} "
        f"num_envs={request.num_envs} "
        f"episodes={len(request.episodes)} "
        f"prepared_runtime_inputs={len(runtime_inputs)} "
        f"plan_path={plan_path} "
        f"execution_summary_path={execution_summary_path}"
    )
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, _unknown = parser.parse_known_args(list(argv or ()))
    if args.exporter is not None and not args.execute_prepared_requests:
        raise ValueError("--exporter requires --execute_prepared_requests because export depends on episode_result.json.")
    print(
        "[RC5Collection] phase=bootstrap_resolve_request "
        f"config_path={args.config_path} "
        f"placement_manifest={args.placement_manifest} "
        f"output_dir={args.output_dir} "
        f"num_envs={args.num_envs}",
        flush=True,
    )
    request = resolve_collection_request(argv)
    log_path = request.output_dir / "collector_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _TeeStream(sys.stdout, log_file)
        tee_stderr = _TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            return _run_collection_main(args, request)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
