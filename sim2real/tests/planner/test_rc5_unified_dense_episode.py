from __future__ import annotations

import zipfile

import numpy as np
import pytest

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    DENSE_EPISODE_ARTIFACT_SCHEMA_VERSION,
    RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION,
    RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION_EMBEDDED_RUNTIME_BUNDLE,
    build_dense_episode_payload,
    build_rl4vla_raw_episode_payload,
    decode_dense_episode_images,
    decode_rl4vla_raw_episode_images,
    load_dense_episode_artifact,
    load_rl4vla_raw_episode_artifact,
    rewrite_rl4vla_raw_episode_embedded_runtime_request_json,
    write_dense_episode_artifact,
    write_rl4vla_raw_episode_artifact,
)


def test_build_dense_episode_payload_requires_dense_step_alignment():
    with pytest.raises(ValueError, match="one more frame than actions"):
        build_dense_episode_payload(
            instruction="Pick up orange_cube_ext.",
            images=[np.zeros((4, 4, 3), dtype=np.uint8)],
            actions=[np.zeros((7,), dtype=np.float32)],
            infos=[],
            result={"success": True},
            source={"planner_backend": "proxy_ee_delta"},
        )


def test_write_dense_episode_artifact_round_trips_npz(tmp_path):
    artifact_path = tmp_path / "dense_episode.npz"
    write_dense_episode_artifact(
        artifact_path=artifact_path,
        instruction="Pick up orange_cube_ext.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True, "reward": 1.0}],
        result={"success": True, "semantic_task_success": True},
        source={"planner_backend": "proxy_ee_delta", "macro_route": "pick_macro_1"},
    )

    payload = load_dense_episode_artifact(artifact_path)

    assert payload["schema_version"] == DENSE_EPISODE_ARTIFACT_SCHEMA_VERSION
    assert payload["instruction"] == "Pick up orange_cube_ext."
    assert payload["camera_name"] == "base_camera"
    assert payload["image_encoding"] == "jpeg"
    assert payload["image"].shape == (2,)
    assert decode_dense_episode_images(payload).shape == (2, 480, 640, 3)
    assert payload["action"].shape == (1, 7)
    assert payload["info"].shape == (1,)
    assert payload["info"][0]["success"] is True


def test_build_dense_episode_payload_resizes_to_rl4vla_canvas_with_letterbox():
    payload = build_dense_episode_payload(
        instruction="Pick up orange_cube_ext.",
        images=[
            np.zeros((1076, 1916, 3), dtype=np.uint8),
            np.ones((1076, 1916, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True}],
        result={"success": True},
        source={"planner_backend": "proxy_ee_delta"},
        image_target_width=640,
        image_target_height=480,
    )

    assert payload["image_encoding"] == "jpeg"
    assert payload["image_target_width"] == 640
    assert payload["image_target_height"] == 480
    assert payload["image_original_hw"] == [1076, 1916]
    assert payload["image_stored_hw"] == [480, 640]
    decoded = decode_dense_episode_images(payload)
    assert decoded.shape == (2, 480, 640, 3)


def test_build_rl4vla_raw_episode_payload_aligns_to_step_count():
    payload = build_rl4vla_raw_episode_payload(
        instruction="Pick up orange cube.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ],
        actions=[
            np.zeros((7,), dtype=np.float32),
            np.ones((7,), dtype=np.float32),
        ],
        infos=[{"success": False}, {"success": True}],
        result={"success": True},
        source={"planner_backend": "proxy_ee_delta"},
    )

    assert payload["schema_version"] == RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION
    assert payload["instruction"] == "Pick up orange cube."
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 2
    assert isinstance(payload["image"][0], np.ndarray)
    assert payload["image"][0].dtype == np.uint8
    assert payload["image"][0].ndim == 1
    assert decode_rl4vla_raw_episode_images(payload).shape == (2, 480, 640, 3)
    assert payload["action"].shape == (2, 7)
    assert len(payload["info"]) == 2


def test_write_rl4vla_raw_episode_artifact_round_trips_npz(tmp_path):
    artifact_path = tmp_path / "rl4vla_raw_episode.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="Pick up orange cube.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True, "reward": 1.0}],
        result={"success": True, "semantic_task_success": True},
        source={"planner_backend": "proxy_ee_delta", "macro_route": "pick_macro_1"},
    )

    payload = np.load(artifact_path, allow_pickle=True)["arr_0"].tolist()

    assert payload["schema_version"] == RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION
    assert payload["instruction"] == "Pick up orange cube."
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 1
    assert payload["image"][0].dtype == np.uint8
    assert payload["image"][0].ndim == 1
    assert decode_rl4vla_raw_episode_images(payload).shape == (1, 480, 640, 3)
    assert payload["action"].shape == (1, 7)
    assert payload["info"][0]["success"] is True


def test_write_rl4vla_raw_episode_artifact_can_embed_runtime_bundle(tmp_path):
    artifact_path = tmp_path / "rl4vla_raw_episode_embedded.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="Pick up orange cube.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True, "reward": 1.0}],
        result={"success": True, "semantic_task_success": True},
        source={"planner_backend": "proxy_ee_delta", "macro_route": "pick_macro_1"},
        embedded_runtime_config_yaml="local:\n  demo_key:\n    simulation:\n      manip_object_id: orange_cube_ext\n",
        embedded_runtime_request_json='{"episode_id":"episode_000000","task_object_id":"orange_cube_ext"}',
    )

    payload = load_rl4vla_raw_episode_artifact(artifact_path)

    assert payload["schema_version"] == RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION_EMBEDDED_RUNTIME_BUNDLE
    assert payload["embedded_runtime_config_yaml"].startswith("local:\n")
    assert '"episode_000000"' in payload["embedded_runtime_request_json"]


def test_rewrite_rl4vla_raw_episode_embedded_runtime_request_json_updates_embedded_copy(tmp_path):
    artifact_path = tmp_path / "rl4vla_raw_episode_embedded.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="Pick up orange cube.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True}],
        result={"success": True},
        source={"planner_backend": "proxy_ee_delta"},
        embedded_runtime_request_json='{"episode_id":"episode_000000","rl4vla_raw_episode_artifact_path":"old.npz"}',
    )

    rewrite_rl4vla_raw_episode_embedded_runtime_request_json(
        artifact_path,
        runtime_request_json='{"episode_id":"episode_000000","rl4vla_raw_episode_artifact_path":"new_success.npz"}',
    )

    payload = load_rl4vla_raw_episode_artifact(artifact_path)
    assert payload["embedded_runtime_request_json"].endswith('"new_success.npz"}')


def test_write_rl4vla_raw_episode_artifact_defaults_to_uncompressed_npz(tmp_path, monkeypatch):
    monkeypatch.delenv("RC5_RL4VLA_RAW_COMPRESS", raising=False)
    artifact_path = tmp_path / "rl4vla_raw_episode_uncompressed.npz"

    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="Pick up orange cube.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True}],
        result={"success": True},
        source={"planner_backend": "proxy_ee_delta"},
    )

    with zipfile.ZipFile(artifact_path, "r") as archive:
        members = archive.infolist()
        assert members
        assert all(item.compress_type == zipfile.ZIP_STORED for item in members)


def test_write_rl4vla_raw_episode_artifact_supports_env_for_compressed_npz(tmp_path, monkeypatch):
    monkeypatch.setenv("RC5_RL4VLA_RAW_COMPRESS", "1")
    artifact_path = tmp_path / "rl4vla_raw_episode_compressed.npz"

    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="Pick up orange cube.",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=[np.zeros((7,), dtype=np.float32)],
        infos=[{"success": True}],
        result={"success": True},
        source={"planner_backend": "proxy_ee_delta"},
    )

    with zipfile.ZipFile(artifact_path, "r") as archive:
        members = archive.infolist()
        assert members
        assert all(item.compress_type == zipfile.ZIP_DEFLATED for item in members)


def test_build_rl4vla_raw_episode_payload_accepts_preencoded_images():
    images = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
        np.full((4, 4, 3), 2, dtype=np.uint8),
    ]
    preencoded = [
        np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8).copy(),
        np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8).copy(),
    ]

    with pytest.raises(ValueError, match="exactly as many items as images"):
        build_rl4vla_raw_episode_payload(
            instruction="Pick up orange cube.",
            images=images,
            preencoded_images=preencoded,
            actions=[
                np.zeros((7,), dtype=np.float32),
                np.ones((7,), dtype=np.float32),
            ],
            infos=[{"success": False}, {"success": True}],
            result={"success": True},
            source={"planner_backend": "proxy_ee_delta"},
        )


def test_build_rl4vla_raw_episode_payload_stores_wrist_camera_images():
    payload = build_rl4vla_raw_episode_payload(
        instruction="Pick red cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ],
        wrist_images=[
            np.full((6, 8, 3), 9, dtype=np.uint8),
            np.full((6, 8, 3), 8, dtype=np.uint8),
            np.full((6, 8, 3), 7, dtype=np.uint8),
        ],
        actions=[
            np.zeros((7,), dtype=np.float32),
            np.ones((7,), dtype=np.float32),
        ],
        infos=[{"success": False}, {"success": True}],
        result={"semantic_task_success": True},
        source={"planner_backend": "proxy_ee_delta"},
    )

    assert payload["camera_names"] == ["base_camera", "wrist_camera"]
    assert isinstance(payload["image_wrist"], list)
    assert len(payload["image_wrist"]) == 2
    assert decode_rl4vla_raw_episode_images(payload).shape == (2, 480, 640, 3)
    assert decode_rl4vla_raw_episode_images(payload, key="image_wrist").shape == (2, 480, 640, 3)
