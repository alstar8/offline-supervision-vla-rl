from __future__ import annotations

import numpy as np

from openreal2sim.simulation.maniskill.scripts import rc5_unified_proxy_artifacts as uut
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    decode_rl4vla_raw_episode_images,
    load_dense_episode_artifact,
)


def test_batched_dense_capture_persists_per_env_instructions_for_dense_and_rl4vla(tmp_path):
    dense_dir = tmp_path / "dense"
    raw_dir = tmp_path / "raw"
    uut.clear_unified_dense_episode_capture()
    try:
        uut.start_unified_dense_episode_capture(
            output_dir=dense_dir,
            rl4vla_raw_output_dir=raw_dir,
            instructions_per_env=[
                "Pick up orange cube.",
                "Pick up blue cube.",
            ],
            runtime_config_paths_per_env=[
                tmp_path / "episode_0_runtime_config.yaml",
                tmp_path / "episode_1_runtime_config.yaml",
            ],
            runtime_request_paths_per_env=[
                tmp_path / "episode_0_runtime_request.json",
                tmp_path / "episode_1_runtime_request.json",
            ],
            image_target_width=640,
            image_target_height=480,
            num_envs=2,
        )
        capture = uut._ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
        capture.runtime_config_paths_per_env[0].write_text("episode: 0\n", encoding="utf-8")
        capture.runtime_config_paths_per_env[1].write_text("episode: 1\n", encoding="utf-8")
        capture.runtime_request_paths_per_env[0].write_text('{"episode_id":"episode_000000"}', encoding="utf-8")
        capture.runtime_request_paths_per_env[1].write_text('{"episode_id":"episode_000001"}', encoding="utf-8")
        frame0 = np.zeros((4, 4, 3), dtype=np.uint8)
        frame1 = np.ones((4, 4, 3), dtype=np.uint8)
        capture.images_per_env[0].extend([frame0.copy(), frame1.copy()])
        capture.images_per_env[1].extend([frame1.copy(), frame0.copy()])
        capture.actions_per_env[0].append(np.zeros((7,), dtype=np.float32))
        capture.actions_per_env[1].append(np.ones((7,), dtype=np.float32))
        capture.infos_per_env[0].append({"success": True})
        capture.infos_per_env[1].append({"success": False})

        uut.finalize_batched_dense_episode_env_if_available(
            env_index=0,
            planner_backend_value="proxy_ee_delta",
            macro_route="pick_macro_1",
            exit_code=0,
            semantic_task_success=True,
            failed_stage=None,
        )
        uut.finalize_batched_dense_episode_env_if_available(
            env_index=1,
            planner_backend_value="proxy_ee_delta",
            macro_route="pick_macro_1",
            exit_code=3,
            semantic_task_success=False,
            failed_stage="lift",
        )

        uut.write_dense_episode_artifact_if_available(
            planner_backend_value="proxy_ee_delta",
            macro_route="pick_macro_1",
            exit_code=0,
            macro_feedback={"semantic_task_success": False, "failed_stage": "lift"},
        )

        dense_0 = load_dense_episode_artifact(dense_dir / "env_000000_dense_episode.npz")
        dense_1 = load_dense_episode_artifact(dense_dir / "env_000001_dense_episode.npz")
        raw_0 = np.load(raw_dir / "env_000000_rl4vla_raw_episode.npz", allow_pickle=True)["arr_0"].tolist()
        raw_1 = np.load(raw_dir / "env_000001_rl4vla_raw_episode.npz", allow_pickle=True)["arr_0"].tolist()

        assert dense_0["instruction"] == "Pick up orange cube."
        assert dense_1["instruction"] == "Pick up blue cube."
        assert raw_0["instruction"] == "Pick up orange cube."
        assert raw_1["instruction"] == "Pick up blue cube."
        assert raw_0["image"][0].dtype == np.uint8
        assert raw_0["image"][0].ndim == 1
        assert raw_1["image"][0].dtype == np.uint8
        assert raw_1["image"][0].ndim == 1
        assert raw_0["embedded_runtime_config_yaml"] == "episode: 0\n"
        assert raw_1["embedded_runtime_config_yaml"] == "episode: 1\n"
        assert raw_0["embedded_runtime_request_json"] == '{"episode_id":"episode_000000"}'
        assert raw_1["embedded_runtime_request_json"] == '{"episode_id":"episode_000001"}'
        assert decode_rl4vla_raw_episode_images(raw_0).shape == (1, 480, 640, 3)
        assert decode_rl4vla_raw_episode_images(raw_1).shape == (1, 480, 640, 3)
    finally:
        uut.clear_unified_dense_episode_capture()


def test_batched_dense_capture_can_share_video_frames_and_preencode_raw(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    shared_video_frames = [[], []]

    class _Env:
        class _Unwrapped:
            num_envs = 2

        unwrapped = _Unwrapped()

    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]

    monkeypatch.setattr(uut, "capture_base_camera_frames", lambda _env: frames)
    uut.clear_unified_dense_episode_capture()
    try:
        uut.start_unified_dense_episode_capture(
            rl4vla_raw_output_dir=raw_dir,
            instructions_per_env=[
                "Pick up orange cube.",
                "Pick up blue cube.",
            ],
            image_target_width=640,
            image_target_height=480,
            num_envs=2,
            shared_video_frame_targets_per_env=shared_video_frames,
        )
        capture = uut.get_active_unified_dense_episode_capture()
        assert capture is not None

        uut.append_dense_episode_initial_frame_if_enabled(_Env())
        assert capture.images_per_env[0][0] is frames[0]
        assert shared_video_frames[0][0] is frames[0]
        assert capture.rl4vla_preencoded_images_per_env is not None
        assert capture.rl4vla_preencoded_images_per_env[0][0].dtype == np.uint8
        assert capture.rl4vla_preencoded_images_per_env[0][0].ndim == 1

        uut.record_dense_episode_step_if_enabled(
            _Env(),
            np.zeros((2, 7), dtype=np.float32),
            (None, None, None, None, {"success": np.array([True, False])}),
        )
        assert len(capture.images_per_env[0]) == 2
        assert len(shared_video_frames[0]) == 2
        assert len(capture.rl4vla_preencoded_images_per_env[0]) == 2
    finally:
        uut.clear_unified_dense_episode_capture()
