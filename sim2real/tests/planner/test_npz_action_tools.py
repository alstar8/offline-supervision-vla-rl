from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openreal2sim.simulation.maniskill.scripts.export_npz_actions_to_csv import (
    CSV_HEADER,
    default_output_csv_path,
    default_output_runtime_config_yaml_path,
    default_output_runtime_request_json_path,
    load_actions,
    write_csv,
    write_embedded_runtime_bundle,
)
from openreal2sim.simulation.maniskill.scripts.compress_npz_actions import (
    CompressionOptions,
    _compute_negative_dz_mergeable_mask,
    compress_actions_payload,
    default_output_gif_path,
    write_planner_style_gif_from_payload,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    write_rl4vla_raw_episode_artifact,
)
from openreal2sim.simulation.maniskill.scripts.summarize_npz_actions import (
    build_summary_lines,
)


def _write_demo_npz(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ],
        actions=[
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.2, 0.0, 0.0, 0.0, -1.0],
        ],
        infos=[{"success": False}, {"success": True}],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
        embedded_runtime_config_yaml="local:\n  demo: true\n",
        embedded_runtime_request_json=json.dumps(
            {"episode_id": "episode_000000", "task_object_id": "yellow_cube_ext"},
            indent=2,
            sort_keys=True,
        ),
    )
    return artifact_path


def test_summarize_npz_actions_reports_embedded_bundle(tmp_path):
    artifact_path = _write_demo_npz(tmp_path)
    payload, actions = load_actions(artifact_path)

    lines = build_summary_lines(artifact_path, payload, actions, eps=1e-9)

    assert any("steps=2" in line for line in lines)
    assert any("embedded_runtime_bundle: yaml=yes json=yes" in line for line in lines)
    assert any("first_nonzero_dz: step=1 value=-0.20000000298023224" in line for line in lines)
    assert any("gripper_hist=" in line for line in lines)


def test_export_npz_actions_to_csv_writes_csv_and_sidecars(tmp_path):
    artifact_path = _write_demo_npz(tmp_path)
    payload, actions = load_actions(artifact_path)

    output_csv = default_output_csv_path(artifact_path)
    output_yaml = default_output_runtime_config_yaml_path(artifact_path)
    output_json = default_output_runtime_request_json_path(artifact_path)

    write_csv(actions, output_csv)
    written_bundle = write_embedded_runtime_bundle(
        payload,
        output_runtime_config_yaml=output_yaml,
        output_runtime_request_json=output_json,
    )

    assert output_csv.exists()
    csv_lines = output_csv.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0] == ",".join(CSV_HEADER)
    assert csv_lines[1].startswith("0,0.10000000149011612,0.0,0.0,0.0,0.0,0.0,0.0")
    assert csv_lines[2].startswith("1,0.0,0.0,-0.20000000298023224,0.0,0.0,0.0,-1.0")

    assert written_bundle["runtime_config_yaml"] == output_yaml
    assert written_bundle["runtime_request_json"] == output_json
    assert output_yaml.read_text(encoding="utf-8") == "local:\n  demo: true\n"
    assert json.loads(output_json.read_text(encoding="utf-8"))["task_object_id"] == "yellow_cube_ext"


def test_export_npz_actions_to_csv_skips_missing_embedded_sidecars(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up blue cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        actions=np.zeros((1, 7), dtype=np.float32),
        infos=[{"success": True}],
        result={"success": True},
        source={"task_object_id": "blue_cube_ext"},
    )

    payload, _ = load_actions(artifact_path)
    written_bundle = write_embedded_runtime_bundle(
        payload,
        output_runtime_config_yaml=default_output_runtime_config_yaml_path(artifact_path),
        output_runtime_request_json=default_output_runtime_request_json_path(artifact_path),
    )

    assert written_bundle == {}


def test_compress_npz_actions_merges_safe_translation_only_chunks(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
            np.full((4, 4, 3), 3, dtype=np.uint8),
        ],
        actions=[
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.01, 0.0, 0.0, 0.0, 0.0, -1.0],
        ],
        infos=[{"step": 0}, {"step": 1}, {"step": 2}],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
    )

    payload = load_actions(artifact_path)[0]
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(max_steps_per_chunk=8, max_translation_norm=0.04),
    )

    compressed_actions = np.asarray(compressed_payload["action"], dtype=np.float32)
    assert summary == {"original_steps": 3, "compressed_steps": 2, "merged_steps": 1}
    assert compressed_actions.shape == (2, 7)
    assert np.allclose(compressed_actions[0], np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(compressed_actions[1], np.array([0.0, -0.01, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert len(compressed_payload["image"]) == 2
    assert compressed_payload["info"][0]["step"] == 1
    assert compressed_payload["info"][1]["step"] == 2


def test_compress_npz_actions_does_not_merge_rotation_sign_flip_or_gripper_steps(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
            np.full((4, 4, 3), 3, dtype=np.uint8),
            np.full((4, 4, 3), 4, dtype=np.uint8),
        ],
        actions=[
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ],
        infos=[{"step": 0}, {"step": 1}, {"step": 2}, {"step": 3}],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
    )

    payload = load_actions(artifact_path)[0]
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(max_steps_per_chunk=8, max_translation_norm=0.04),
    )

    compressed_actions = np.asarray(compressed_payload["action"], dtype=np.float32)
    assert summary == {"original_steps": 4, "compressed_steps": 4, "merged_steps": 0}
    assert np.allclose(compressed_actions, np.asarray(payload["action"], dtype=np.float32))


def test_compress_npz_actions_keeps_negative_dz_steps_unmerged_by_default(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ],
        actions=[
            [0.001, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.001, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
        ],
        infos=[{"step": 0}, {"step": 1}],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
    )

    payload = load_actions(artifact_path)[0]
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(max_steps_per_chunk=8, max_translation_norm=0.04),
    )

    compressed_actions = np.asarray(compressed_payload["action"], dtype=np.float32)
    assert summary == {"original_steps": 2, "compressed_steps": 2, "merged_steps": 0}
    assert np.allclose(compressed_actions, np.asarray(payload["action"], dtype=np.float32))


def test_compress_npz_actions_can_merge_negative_dz_steps_when_explicitly_allowed(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ],
        actions=[
            [0.001, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.001, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
        ],
        infos=[{"step": 0}, {"step": 1}],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
    )

    payload = load_actions(artifact_path)[0]
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(
            max_steps_per_chunk=8,
            max_translation_norm=0.04,
            allow_negative_dz_merge=True,
        ),
    )

    compressed_actions = np.asarray(compressed_payload["action"], dtype=np.float32)
    assert summary == {"original_steps": 2, "compressed_steps": 1, "merged_steps": 1}
    assert np.allclose(
        compressed_actions[0],
        np.array([0.002, 0.0, -0.004, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_negative_dz_keep_tail_ratio_preserves_last_part_of_descent():
    actions = np.asarray(
        [
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    mergeable = _compute_negative_dz_mergeable_mask(
        actions,
        options=CompressionOptions(
            allow_negative_dz_merge=True,
            negative_dz_keep_tail_ratio=0.2,
        ),
    )

    assert mergeable.tolist() == [True, True, True, True, False]


def test_compress_npz_actions_merges_only_early_part_of_descent_when_tail_ratio_is_set(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(6)],
        actions=[
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
        ],
        infos=[{"step": idx} for idx in range(5)],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
    )

    payload = load_actions(artifact_path)[0]
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(
            max_steps_per_chunk=8,
            max_translation_norm=0.04,
            allow_negative_dz_merge=True,
            negative_dz_keep_tail_ratio=0.2,
        ),
    )

    compressed_actions = np.asarray(compressed_payload["action"], dtype=np.float32)
    assert summary == {"original_steps": 5, "compressed_steps": 2, "merged_steps": 3}
    assert np.allclose(
        compressed_actions,
        np.asarray(
            [
                [0.0, 0.0, -0.008, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_compress_npz_actions_respects_max_merged_negative_dz_cap(tmp_path):
    artifact_path = tmp_path / "episode_success.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=artifact_path,
        instruction="pick up yellow cube",
        images=[np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(5)],
        actions=[
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.002, 0.0, 0.0, 0.0, 0.0],
        ],
        infos=[{"step": idx} for idx in range(4)],
        result={"success": True},
        source={"task_object_id": "yellow_cube_ext"},
    )

    payload = load_actions(artifact_path)[0]
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(
            max_steps_per_chunk=8,
            max_translation_norm=0.04,
            allow_negative_dz_merge=True,
            max_merged_negative_dz=0.004,
        ),
    )

    compressed_actions = np.asarray(compressed_payload["action"], dtype=np.float32)
    assert summary == {"original_steps": 4, "compressed_steps": 2, "merged_steps": 2}
    assert np.allclose(
        compressed_actions,
        np.asarray(
            [
                [0.0, 0.0, -0.004, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.004, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_default_output_gif_path_uses_input_npz_stem_with_compressed_suffix(tmp_path):
    input_npz = tmp_path / "episode_success.npz"
    assert default_output_gif_path(input_npz) == tmp_path / "episode_success.compressed.gif"


def test_write_planner_style_gif_from_payload_uses_embedded_images(tmp_path, monkeypatch):
    payload = {
        "image": [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8) * 255,
        ]
    }
    calls: dict[str, object] = {}

    def _fake_flush(video_frames, video_path, video_fps, video_codec=None, video_output_params=None):
        calls["flush"] = {
            "frame_count": len(video_frames),
            "video_path": Path(video_path),
            "video_fps": video_fps,
            "video_codec": video_codec,
            "video_output_params": video_output_params,
            "frame_shape": tuple(video_frames[0].shape),
        }
        Path(video_path).write_bytes(b"fake-video")
        return True

    def _fake_write_gif(video_path, gif_path):
        calls["gif"] = {
            "video_path": Path(video_path),
            "gif_path": Path(gif_path),
        }
        Path(gif_path).write_bytes(b"GIF89a")
        return Path(gif_path)

    monkeypatch.setattr(
        "openreal2sim.simulation.maniskill.scripts.compress_npz_actions.flush_video_buffer_to_file",
        _fake_flush,
    )
    monkeypatch.setattr(
        "openreal2sim.simulation.maniskill.scripts.compress_npz_actions.write_debug_video_gif_from_video",
        _fake_write_gif,
    )
    monkeypatch.setattr(
        "openreal2sim.simulation.maniskill.scripts.compress_npz_actions.resolve_effective_video_codec",
        lambda *_args, **_kwargs: ("mpeg4", None, "test"),
    )

    output_gif = tmp_path / "episode_success.compressed.gif"
    written_path = write_planner_style_gif_from_payload(output_gif, payload=payload)

    assert written_path == output_gif
    assert calls["flush"]["frame_count"] == 2
    assert calls["flush"]["frame_shape"] == (4, 4, 3)
    assert calls["flush"]["video_fps"] == 30
    assert calls["flush"]["video_codec"] == "mpeg4"
    assert calls["gif"]["gif_path"] == output_gif
    assert output_gif.exists()
    assert not output_gif.with_suffix(".tmp_debug_video.mkv").exists()
