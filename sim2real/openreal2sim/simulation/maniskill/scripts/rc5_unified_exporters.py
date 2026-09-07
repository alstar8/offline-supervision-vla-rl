from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from openreal2sim.simulation.maniskill.scripts.rc5_unified_trajectory import (
    EPISODE_ARTIFACT_SCHEMA_VERSION,
)


RL4VLA_SFT_V0 = "rl4vla_sft_v0"
SUPPORTED_EXPORTERS = (RL4VLA_SFT_V0,)
RL4VLA_SFT_SCHEMA_VERSION = "rl4vla_sft_episode_v0"


def _require_mapping(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _require_non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return str(value)


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool")
    return bool(value)


def load_episode_artifact(path: str | Path) -> Dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Episode artifact does not exist: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return _require_mapping(payload, label="episode artifact")


def build_rl4vla_sft_v0_record(episode_artifact: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _require_mapping(episode_artifact, label="episode artifact")
    schema_version = _require_non_empty_str(payload.get("schema_version"), label="schema_version")
    if schema_version != EPISODE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported episode artifact schema_version '{schema_version}'. "
            f"Expected '{EPISODE_ARTIFACT_SCHEMA_VERSION}'."
        )

    task = _require_mapping(payload.get("task"), label="task")
    config = _require_mapping(payload.get("config"), label="config")
    placement = _require_mapping(payload.get("placement"), label="placement")
    result = _require_mapping(payload.get("result"), label="result")

    episode_id = _require_non_empty_str(payload.get("episode_id"), label="episode_id")
    motion_backend = _require_non_empty_str(payload.get("motion_backend"), label="motion_backend")
    task_type = _require_non_empty_str(task.get("task_type"), label="task.task_type")
    object_id = _require_non_empty_str(task.get("object_id"), label="task.object_id")
    config_key = _require_non_empty_str(config.get("key"), label="config.key")
    scene_path = _require_non_empty_str(config.get("scene_path"), label="config.scene_path")
    placement_source = _require_non_empty_str(placement.get("source"), label="placement.source")
    placement_spec = _require_mapping(placement.get("spec"), label="placement.spec")
    success = _require_bool(result.get("success"), label="result.success")
    execution_outcome = _require_non_empty_str(result.get("execution_outcome"), label="result.execution_outcome")

    instruction = f"{task_type}:{object_id}"
    if task_type == "pick_up":
        instruction = f"Pick up {object_id}."

    return {
        "schema_version": RL4VLA_SFT_SCHEMA_VERSION,
        "episode_id": episode_id,
        "instruction": instruction,
        "source": {
            "motion_backend": motion_backend,
            "config_key": config_key,
            "scene_path": scene_path,
        },
        "task": {
            "task_type": task_type,
            "object_id": object_id,
            "destination_id": task.get("destination_id"),
        },
        "placement": {
            "source": placement_source,
            "seed": placement.get("seed"),
            "spec": placement_spec,
        },
        "result": {
            "success": success,
            "execution_outcome": execution_outcome,
            "exit_code": result.get("exit_code"),
            "runtime_exit_code": result.get("runtime_exit_code"),
            "semantic_task_success": result.get("semantic_task_success"),
            "failed_stage": result.get("failed_stage"),
        },
        "trace_record": payload.get("trace_record"),
    }


def export_episode_artifact(
    *,
    episode_artifact_path: str | Path,
    export_dir: str | Path,
    exporter_name: str = RL4VLA_SFT_V0,
) -> Path:
    exporter = _require_non_empty_str(exporter_name, label="exporter_name")
    if exporter not in SUPPORTED_EXPORTERS:
        supported = ", ".join(SUPPORTED_EXPORTERS)
        raise ValueError(f"Unsupported exporter '{exporter}'. Supported exporters: {supported}")

    artifact = load_episode_artifact(episode_artifact_path)
    if exporter != RL4VLA_SFT_V0:
        raise AssertionError("unreachable exporter dispatch")
    record = build_rl4vla_sft_v0_record(artifact)

    output_dir = Path(export_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    record_path = output_dir / "episode_record.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = output_dir / "export_manifest.json"
    manifest_payload = {
        "exporter_name": RL4VLA_SFT_V0,
        "schema_version": RL4VLA_SFT_SCHEMA_VERSION,
        "source_episode_artifact": str(Path(episode_artifact_path).expanduser().resolve()),
        "episode_id": record["episode_id"],
        "artifact_count": 1,
        "artifacts": [
            {
                "artifact_type": "episode_record",
                "path": str(record_path),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
