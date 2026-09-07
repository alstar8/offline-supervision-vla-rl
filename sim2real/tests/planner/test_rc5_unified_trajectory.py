from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openreal2sim.simulation.maniskill.scripts.rc5_unified_trajectory import (
    EPISODE_ARTIFACT_SCHEMA_VERSION,
    build_episode_artifact_payload,
    write_episode_artifact,
)


@dataclass(frozen=True)
class _DummyRequest:
    motion_backend: str = "proxy_ee_delta"
    task_type: str = "pick_up"
    task_object_id: str = "orange_cube_ext"
    config_path: Path = Path("/tmp/config.yaml")
    key: str = "demo_key"
    scene_path: str = "scene.json"


@dataclass(frozen=True)
class _DummyExecutionResult:
    episode_id: str = "episode_000000"
    episode_index: int = 0
    request_path: Path = Path("/tmp/runtime_request.json")
    batch_id: str | None = None
    compatibility_group_id: str | None = None
    env_index: int | None = None
    exit_code: int = 0
    runtime_exit_code: int | None = None
    execution_outcome: str = "success"
    success: bool = True
    semantic_task_success: bool | None = None
    failed_stage: str | None = None
    trace_record: dict | None = None


def test_build_episode_artifact_payload_contains_minimal_schema():
    payload = build_episode_artifact_payload(
        request=_DummyRequest(),
        runtime_request={
            "runtime_config_path": "/tmp/runtime_config.yaml",
            "dense_episode_artifact_path": "/tmp/dense_episode.npz",
            "debug_video_path": "/tmp/debug_video.mkv",
            "debug_video_gif_path": "/tmp/debug_video.gif",
            "placement_source": "seeded_fixed",
            "placement_seed": 123,
            "placement": {"position": [0.1, -0.8, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
        },
        execution_result=_DummyExecutionResult(trace_record={"execution_outcome": "success"}),
    )

    assert payload["schema_version"] == EPISODE_ARTIFACT_SCHEMA_VERSION
    assert payload["task"]["task_type"] == "pick_up"
    assert payload["placement"]["source"] == "seeded_fixed"
    assert payload["result"]["success"] is True
    assert payload["trace_record"]["execution_outcome"] == "success"
    assert payload["config"]["dense_episode_artifact_path"] == "/tmp/dense_episode.npz"
    assert payload["config"]["debug_video_path"] == "/tmp/debug_video.mkv"
    assert payload["config"]["debug_video_gif_path"] == "/tmp/debug_video.gif"
    assert payload["config"]["runtime_sim_patch_path"] is None


def test_write_episode_artifact_round_trips_json(tmp_path):
    artifact_path = tmp_path / "episode_result.json"
    write_episode_artifact(
        artifact_path=artifact_path,
        request=_DummyRequest(),
        runtime_request={
            "runtime_config_path": "/tmp/runtime_config.yaml",
            "placement_source": "manifest",
            "placement_seed": None,
            "placement": {"position": [0.2, -0.7, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
        },
        execution_result=_DummyExecutionResult(
            exit_code=7,
            execution_outcome="failed",
            success=False,
            trace_record={"events": [{"event_type": "macro_finished"}]},
        ),
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["placement"]["source"] == "manifest"
    assert payload["result"]["exit_code"] == 7
    assert payload["result"]["execution_outcome"] == "failed"
    assert payload["result"]["success"] is False
    assert payload["trace_record"]["events"][0]["event_type"] == "macro_finished"


def test_build_episode_artifact_payload_persists_semantic_failure_fields():
    payload = build_episode_artifact_payload(
        request=_DummyRequest(),
        runtime_request={
            "runtime_config_path": "/tmp/runtime_config.yaml",
            "placement_source": "seeded_fixed",
            "placement_seed": 0,
            "placement": {"position": [0.1, -0.8, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
        },
        execution_result=_DummyExecutionResult(
            exit_code=3,
            runtime_exit_code=0,
            execution_outcome="failed",
            success=False,
            semantic_task_success=False,
            failed_stage="lift",
        ),
    )

    assert payload["result"]["exit_code"] == 3
    assert payload["result"]["runtime_exit_code"] == 0
    assert payload["result"]["semantic_task_success"] is False
    assert payload["result"]["failed_stage"] == "lift"


def test_build_episode_artifact_payload_persists_runtime_patch_path():
    payload = build_episode_artifact_payload(
        request=_DummyRequest(),
        runtime_request={
            "runtime_config_path": "/tmp/runtime_config.yaml",
            "runtime_sim_patch_path": "/tmp/runtime_patch.yaml",
            "placement_source": "seeded_fixed",
            "placement_seed": 0,
            "placement": {"position": [0.1, -0.8, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]},
        },
        execution_result=_DummyExecutionResult(),
    )

    assert payload["config"]["runtime_sim_patch_path"] == "/tmp/runtime_patch.yaml"
