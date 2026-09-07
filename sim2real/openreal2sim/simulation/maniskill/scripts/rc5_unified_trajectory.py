from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


EPISODE_ARTIFACT_SCHEMA_VERSION = "rc5_unified_episode_v0"


def serialize_trace_record(trace_record: Any) -> Dict[str, Any] | None:
    if trace_record is None:
        return None
    if is_dataclass(trace_record):
        return asdict(trace_record)
    if isinstance(trace_record, Mapping):
        return dict(trace_record)
    raise TypeError("trace_record must be a dataclass, mapping, or None")


def build_episode_artifact_payload(
    *,
    request: Any,
    runtime_request: Mapping[str, Any],
    execution_result: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": EPISODE_ARTIFACT_SCHEMA_VERSION,
        "episode_id": execution_result.episode_id,
        "episode_index": execution_result.episode_index,
        "motion_backend": request.motion_backend,
        "task": {
            "task_type": request.task_type,
            "object_id": request.task_object_id,
            "destination_id": None,
        },
        "config": {
            "config_path": str(request.config_path),
            "key": request.key,
            "scene_path": request.scene_path,
            "runtime_config_path": str(runtime_request["runtime_config_path"]),
            "runtime_request_path": str(execution_result.request_path),
            "dense_episode_artifact_path": runtime_request.get("dense_episode_artifact_path"),
            "rl4vla_raw_episode_artifact_path": runtime_request.get("rl4vla_raw_episode_artifact_path"),
            "debug_video_path": runtime_request.get("debug_video_path"),
            "debug_video_gif_path": runtime_request.get("debug_video_gif_path"),
            "runtime_sim_patch_path": runtime_request.get("runtime_sim_patch_path"),
        },
        "placement": {
            "source": runtime_request["placement_source"],
            "seed": runtime_request["placement_seed"],
            "object_id": request.task_object_id,
            "spec": dict(runtime_request["placement"]),
        },
        "batch": {
            "batch_id": execution_result.batch_id,
            "compatibility_group_id": execution_result.compatibility_group_id,
            "env_index": execution_result.env_index,
        },
        "result": {
            "exit_code": execution_result.exit_code,
            "runtime_exit_code": execution_result.runtime_exit_code,
            "execution_outcome": execution_result.execution_outcome,
            "success": execution_result.success,
            "semantic_task_success": execution_result.semantic_task_success,
            "failed_stage": execution_result.failed_stage,
        },
        "trace_record": serialize_trace_record(execution_result.trace_record),
    }


def write_episode_artifact(
    *,
    artifact_path: Path,
    request: Any,
    runtime_request: Mapping[str, Any],
    execution_result: Any,
) -> Path:
    payload = build_episode_artifact_payload(
        request=request,
        runtime_request=runtime_request,
        execution_result=execution_result,
    )
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path
