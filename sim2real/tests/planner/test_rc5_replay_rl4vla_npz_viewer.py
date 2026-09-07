from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import openreal2sim.simulation.maniskill.scripts.rc5_replay_rl4vla_npz_viewer as uut
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    write_rl4vla_raw_episode_artifact,
)


def _make_args(npz_path: Path, *, config_path: str | None = None, scene: str | None = None):
    return SimpleNamespace(
        npz_path=str(npz_path),
        scene=scene,
        config_path=config_path,
        key=None,
        task_object_id=None,
        task_type=None,
        instruction=None,
        object_placements_json_path=None,
        control_mode=None,
        render_backend="gpu",
        sim_backend="physx_cuda",
        window_width=1920,
        window_height=1080,
        playback_fps=20.0,
        output_dir=None,
        video_basename=None,
        video_format=None,
        video_codec=None,
        video_fps=None,
        gif_fps=15,
        gif_scale_width=800,
        save_video=True,
        save_gif=True,
        dry_run=True,
        extract_embedded_bundle=False,
    )


def _write_embedded_npz(tmp_path: Path, *, scene_path: Path, runtime_config: dict, runtime_request: dict) -> Path:
    npz_path = tmp_path / "rl4vla_raw_episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=npz_path,
        instruction="Pick up orange cube.",
        images=[
            [[[0, 0, 0]]],
            [[[1, 1, 1]]],
        ],
        actions=[[0, 0, 0, 0, 0, 0, 0]],
        infos=[{"success": True}],
        result={"success": True, "semantic_task_success": True},
        source={"planner_backend": "proxy_ee_delta"},
        embedded_runtime_config_yaml=yaml.safe_dump(runtime_config, sort_keys=False),
        embedded_runtime_request_json=json.dumps(runtime_request, indent=2, sort_keys=True),
    )
    return npz_path


def test_load_replay_context_uses_embedded_runtime_bundle_without_config_path(tmp_path: Path):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    runtime_config = {
        "keys": ["demo_key"],
        "global": {"simulation": {"robot_uids": "rc5_aero_hand_openr2s_rl"}},
        "local": {
            "demo_key": {
                "simulation": {
                    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                    "manip_object_id": "orange_cube_ext",
                    "object_placements": {
                        "orange_cube_ext": {
                            "position": [-0.3, -0.7, 0.0],
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                }
            }
        },
    }
    runtime_request = {
        "episode_id": "episode_000000",
        "scene_path": str(scene_path),
        "runtime_config_path": str(tmp_path / "episode_dir" / "runtime_config.yaml"),
        "task_object_id": "orange_cube_ext",
        "task_type": "pick_up",
        "dense_episode_instruction": "Pick up orange cube.",
    }
    npz_path = _write_embedded_npz(
        tmp_path,
        scene_path=scene_path,
        runtime_config=runtime_config,
        runtime_request=runtime_request,
    )

    context = uut._load_replay_context(_make_args(npz_path))

    assert context.embedded_runtime_bundle_used is True
    assert context.runtime_request == runtime_request
    assert context.key == "demo_key"
    assert context.task_object_id == "orange_cube_ext"
    assert context.task_type == "pick_up"
    assert context.scene_path == scene_path.resolve()
    assert context.runtime_config is not None
    assert context.runtime_config["local"]["demo_key"]["simulation"]["manip_object_id"] == "orange_cube_ext"


def test_load_replay_context_prefers_explicit_config_path_over_embedded_bundle(tmp_path: Path):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    embedded_runtime_config = {
        "keys": ["embedded_key"],
        "global": {"simulation": {}},
        "local": {
            "embedded_key": {
                "simulation": {
                    "manip_object_id": "orange_cube_ext",
                    "object_placements": {
                        "orange_cube_ext": {
                            "position": [-0.3, -0.7, 0.0],
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                }
            }
        },
    }
    explicit_config = {
        "keys": ["explicit_key"],
        "global": {"simulation": {}},
        "local": {
            "explicit_key": {
                "simulation": {
                    "manip_object_id": "blue_cube_ext",
                    "object_placements": {
                        "blue_cube_ext": {
                            "position": [0.0, -0.7, 0.0],
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                }
            }
        },
        "replay_viewer": {
            "scene_path": str(scene_path),
            "task_object_id": "blue_cube_ext",
            "key": "explicit_key",
        },
    }
    runtime_request = {
        "episode_id": "episode_000000",
        "scene_path": str(scene_path),
        "runtime_config_path": str(tmp_path / "episode_dir" / "runtime_config.yaml"),
        "task_object_id": "orange_cube_ext",
        "task_type": "pick_up",
    }
    npz_path = _write_embedded_npz(
        tmp_path,
        scene_path=scene_path,
        runtime_config=embedded_runtime_config,
        runtime_request=runtime_request,
    )
    explicit_config_path = tmp_path / "explicit_runtime_config.yaml"
    explicit_config_path.write_text(yaml.safe_dump(explicit_config, sort_keys=False), encoding="utf-8")

    context = uut._load_replay_context(_make_args(npz_path, config_path=str(explicit_config_path)))

    assert context.embedded_runtime_bundle_used is False
    assert context.runtime_config_path == explicit_config_path.resolve()
    assert context.key == "explicit_key"
    assert context.task_object_id == "blue_cube_ext"


def test_build_env_kwargs_uses_embedded_runtime_config_object_placements(tmp_path: Path):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    runtime_config = {
        "keys": ["demo_key"],
        "global": {"simulation": {"robot_uids": "rc5_aero_hand_openr2s_rl"}},
        "local": {
            "demo_key": {
                "simulation": {
                    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                    "manip_object_id": "orange_cube_ext",
                    "object_placements": {
                        "orange_cube_ext": {
                            "position": [-0.31, -0.7, 0.0],
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                }
            }
        },
    }
    runtime_request = {
        "episode_id": "episode_000000",
        "scene_path": str(scene_path),
        "runtime_config_path": str(tmp_path / "episode_dir" / "runtime_config.yaml"),
        "task_object_id": "orange_cube_ext",
        "task_type": "pick_up",
    }
    npz_path = _write_embedded_npz(
        tmp_path,
        scene_path=scene_path,
        runtime_config=runtime_config,
        runtime_request=runtime_request,
    )

    context = uut._load_replay_context(_make_args(npz_path))
    env_kwargs, sim_cfg, _hand_pose_cfg_path, startup_cfg = uut._build_env_kwargs(context, _make_args(npz_path))

    assert env_kwargs["scene_json_path"] == str(scene_path.resolve())
    assert env_kwargs["manip_object_id"] == "orange_cube_ext"
    assert env_kwargs["object_placements"]["orange_cube_ext"]["position"] == [-0.31, -0.7, 0.0]
    assert sim_cfg["manip_object_id"] == "orange_cube_ext"
    assert startup_cfg["requested_control_mode"] == "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def test_extract_embedded_runtime_bundle_writes_sibling_directory(tmp_path: Path):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    runtime_config = {
        "keys": ["demo_key"],
        "global": {"simulation": {}},
        "local": {"demo_key": {"simulation": {"manip_object_id": "orange_cube_ext"}}},
    }
    runtime_request = {
        "episode_id": "episode_000000",
        "scene_path": str(scene_path),
        "runtime_config_path": str(tmp_path / "episode_dir" / "runtime_config.yaml"),
        "task_object_id": "orange_cube_ext",
        "task_type": "pick_up",
    }
    npz_path = _write_embedded_npz(
        tmp_path,
        scene_path=scene_path,
        runtime_config=runtime_config,
        runtime_request=runtime_request,
    )
    context = uut._load_replay_context(_make_args(npz_path))

    runtime_config_path, runtime_request_path = uut._extract_embedded_runtime_bundle(context)

    assert runtime_config_path == tmp_path / "rl4vla_raw_episode_success" / "runtime_config.yaml"
    assert runtime_request_path == tmp_path / "rl4vla_raw_episode_success" / "runtime_request.json"
    assert runtime_config_path.read_text(encoding="utf-8") == context.payload["embedded_runtime_config_yaml"]
    assert runtime_request_path.read_text(encoding="utf-8") == context.payload["embedded_runtime_request_json"]


def test_extract_embedded_runtime_bundle_fails_fast_when_bundle_missing(tmp_path: Path):
    npz_path = tmp_path / "rl4vla_raw_episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=npz_path,
        instruction="Pick up orange cube.",
        images=[
            [[[0, 0, 0]]],
            [[[1, 1, 1]]],
        ],
        actions=[[0, 0, 0, 0, 0, 0, 0]],
        infos=[{"success": True}],
        result={"success": True, "semantic_task_success": True},
        source={"planner_backend": "proxy_ee_delta"},
    )
    context = uut.ReplayContext(
        npz_path=npz_path,
        payload=uut._load_npz_payload(npz_path),
        runtime_config_path=None,
        runtime_config=None,
        runtime_request=None,
        scene_path=tmp_path / "scene.json",
        key="demo_key",
        task_object_id="orange_cube_ext",
        task_type="pick_up",
        instruction="Pick up orange cube.",
        object_placements=None,
        embedded_runtime_bundle_used=False,
    )

    with pytest.raises(RuntimeError, match="embedded_runtime_config_yaml is missing"):
        uut._extract_embedded_runtime_bundle(context)
