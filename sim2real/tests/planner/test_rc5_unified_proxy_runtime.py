from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_runtime as uut


@pytest.fixture(autouse=True)
def _isolate_viewer_snapshot_dir(monkeypatch, tmp_path: Path):
    def _alloc(_env_unwrapped, _args, _config_overrides, *, video_format, video_codec):
        snapshot_dir = getattr(_env_unwrapped, "_debug_planner_viewer_video_dir", None)
        if snapshot_dir is None:
            snapshot_dir = (tmp_path / "viewer_video").resolve()
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            _env_unwrapped._debug_planner_viewer_video_dir = snapshot_dir
            _env_unwrapped._debug_planner_viewer_video_index = 0
        next_index = int(getattr(_env_unwrapped, "_debug_planner_viewer_video_index", 0)) + 1
        _env_unwrapped._debug_planner_viewer_video_index = next_index
        outcome_token = str(getattr(_env_unwrapped, "_debug_planner_episode_outcome_token", "fail"))
        ext = str(video_format).lstrip(".")
        return snapshot_dir / f"{next_index:03d}_{outcome_token}.{ext}"

    monkeypatch.setattr(uut, "_allocate_viewer_video_snapshot_path", _alloc)


def _make_request():
    return types.SimpleNamespace(
        task_plan=types.SimpleNamespace(
            intent=types.SimpleNamespace(task_type="pick_up"),
            stages=[
                types.SimpleNamespace(name="pregrasp", kind="move_to_pregrasp"),
                types.SimpleNamespace(name="descend", kind="move_to_descend"),
                types.SimpleNamespace(name="close", kind="close_gripper"),
                types.SimpleNamespace(name="lift", kind="lift_object"),
                types.SimpleNamespace(name="retention_check", kind="retention_check"),
            ],
        ),
        bootstrap=types.SimpleNamespace(key="demo_key", robot_uids="rc5_aero_hand_openr2s_rl"),
        trace_seed=types.SimpleNamespace(
            object_id="orange_cube_ext",
            config_key="demo_key",
            stage_kinds=(
                "move_to_pregrasp",
                "move_to_descend",
                "close_gripper",
                "lift_object",
                "retention_check",
            ),
        ),
    )


def test_unified_proxy_runtime_direct_path_avoids_legacy_main(monkeypatch, capsys):
    request = _make_request()
    monkeypatch.setattr(
        uut,
        "_resolve_unified_backend_request_argv",
        lambda _request, _backend: ["--headless", "--auto_pick_macro", "1", "--scene", "scene.json"],
    )
    monkeypatch.setattr(uut, "_write_dense_episode_artifact_if_available", lambda **_kwargs: None)

    def _fake_direct_executor(_argv):
        uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)
        return 0

    monkeypatch.setattr(uut, "_execute_pick_macro_direct", _fake_direct_executor)

    result = uut.run_unified_proxy_backend_request(request)

    assert result["exit_code"] == 0
    assert result["runtime_events"][-2]["event_type"] == "executor_path_selected"
    assert result["runtime_events"][-2]["payload"]["executor_path"] == "direct_unified"
    assert result["runtime_events"][-2]["payload"]["compatibility_reason"] is None
    assert result["runtime_events"][-2]["payload"]["skip_reason"] is None
    assert result["runtime_events"][-1]["event_type"] == "macro_finished"
    assert result["runtime_events"][-1]["payload"]["semantic_task_success"] is True
    captured = capsys.readouterr()
    assert "[WARNING] [RC5UnifiedProxyRuntime]" not in captured.out


def test_unified_proxy_runtime_non_headless_request_uses_direct_executor(monkeypatch):
    request = _make_request()
    observed = {"argv": None}
    monkeypatch.setattr(
        uut,
        "_resolve_unified_backend_request_argv",
        lambda _request, _backend: ["--scene", "scene.json"],
    )
    monkeypatch.setattr(uut, "_write_dense_episode_artifact_if_available", lambda **_kwargs: None)

    def _fake_direct_executor(argv):
        observed["argv"] = list(argv)
        uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)
        return 0

    monkeypatch.setattr(uut, "_execute_pick_macro_direct", _fake_direct_executor)

    result = uut.run_unified_proxy_backend_request(request)

    assert result["exit_code"] == 0
    assert observed["argv"] == ["--scene", "scene.json"]
    assert result["runtime_events"][-2]["payload"]["executor_path"] == "direct_unified"
    assert result["runtime_events"][-2]["payload"]["skip_reason"] is None


def test_unified_proxy_runtime_headless_auto_pick_fails_fast_without_direct_executor(monkeypatch):
    request = _make_request()
    monkeypatch.setattr(
        uut,
        "_resolve_unified_backend_request_argv",
        lambda _request, _backend: [
            "--headless",
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--robot_base_pose",
            "0",
            "0",
            "0",
        ],
    )
    monkeypatch.setattr(uut, "_execute_pick_macro_direct", lambda _argv: None)

    try:
        uut.run_unified_proxy_backend_request(request)
    except ValueError as exc:
        message = str(exc)
        assert "fail-fast" in message
        assert "direct unified proxy executor rejected the request" in message
        assert "custom multi-value pose/camera overrides" in message
        assert "Only the supported unified envelope is allowed." in message
    else:
        raise AssertionError("Expected unsupported headless auto-pick request to fail fast")


def test_unified_proxy_runtime_missing_macro_feedback_fails_fast(monkeypatch):
    request = _make_request()
    monkeypatch.setattr(
        uut,
        "_resolve_unified_backend_request_argv",
        lambda _request, _backend: ["--headless", "--auto_pick_macro", "1", "--scene", "scene.json"],
    )
    monkeypatch.setattr(uut, "_execute_pick_macro_direct", lambda _argv: 0)

    try:
        uut.run_unified_proxy_backend_request(request)
    except ValueError as exc:
        message = str(exc)
        assert "structured macro feedback is missing" in message
        assert "Only the supported unified envelope is allowed." in message
    else:
        raise AssertionError("Expected missing macro feedback to fail fast")


def test_unified_proxy_runtime_argv_entrypoints_fail_fast():
    for fn in (uut.run_unified_proxy_backend, uut.run_unified_hybrid_backend):
        try:
            fn(["--headless", "--auto_pick_macro", "1", "--scene", "scene.json"])
        except ValueError as exc:
            message = str(exc)
            assert "argv-based unified" in message
            assert "no longer allowed" in message
            assert "Only the supported unified envelope is allowed." in message
        else:
            raise AssertionError("Expected argv-based unified proxy entrypoint to fail fast")


def test_direct_executor_skip_allows_headless_batched_proxy_request():
    reason = uut._explain_direct_executor_skip(
        ["--headless", "--auto_pick_macro", "1", "--scene", "scene.json", "--num_envs", "2"]
    )

    assert reason == "direct executor eligibility check did not match the validated envelope"


def test_direct_executor_skip_rejects_batched_shared_dense_episode_artifact():
    reason = uut._explain_direct_executor_skip(
        [
            "--headless",
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--num_envs",
            "2",
            "--dense_episode_output",
            "/tmp/dense_episode.npz",
        ]
    )

    assert "single shared --dense_episode_output" in reason


def test_direct_executor_skip_allows_batched_dense_episode_output_dir():
    reason = uut._explain_direct_executor_skip(
        [
            "--headless",
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--num_envs",
            "2",
            "--dense_episode_output_dir",
            "/tmp/dense_batch",
        ]
    )

    assert reason == "direct executor eligibility check did not match the validated envelope"


def test_direct_executor_skip_rejects_batched_shared_rl4vla_raw_episode_artifact():
    reason = uut._explain_direct_executor_skip(
        [
            "--headless",
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--num_envs",
            "2",
            "--rl4vla_raw_episode_output",
            "/tmp/rl4vla_raw_episode.npz",
        ]
    )

    assert "single shared --rl4vla_raw_episode_output" in reason


def test_batched_per_env_artifacts_enabled_for_dense_output_dir():
    args = types.SimpleNamespace(
        num_envs=2,
        dense_episode_output_dir="/tmp/dense_batch",
        rl4vla_raw_episode_output_dir=None,
    )

    assert uut._batched_per_env_artifacts_enabled(args) is True


def test_batched_per_env_artifacts_enabled_for_rl4vla_raw_output_dir_only():
    args = types.SimpleNamespace(
        num_envs=2,
        dense_episode_output_dir=None,
        rl4vla_raw_episode_output_dir="/tmp/rl4vla_raw_batch",
    )

    assert uut._batched_per_env_artifacts_enabled(args) is True


def test_finalize_macro_feedback_for_batched_runtime_adds_per_env_feedback():
    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=2,
            manip_object_id="orange_cube_ext",
            agent=types.SimpleNamespace(
                is_grasping=lambda _obj: np.asarray([True, True]),
                tcp=types.SimpleNamespace(
                    pose=types.SimpleNamespace(
                        raw_pose=np.asarray(
                            [
                                [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                                [0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        )
                    )
                )
            ),
            object_actors={
                "orange_cube_ext": types.SimpleNamespace(
                    pose=types.SimpleNamespace(
                        raw_pose=np.asarray(
                            [
                                [0.0, -0.7, 0.1, 1.0, 0.0, 0.0, 0.0],
                                [0.1, -0.7, 0.1, 1.0, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        )
                    )
                )
            },
            evaluate=lambda: {
                "success": np.asarray([True, True]),
                "is_src_obj_grasped": np.asarray([True, True]),
                "obj_height_above_table": np.asarray([0.08, 0.09]),
                "gripper_obj_dist": np.asarray([0.03, 0.04]),
                "gripper_goal_dist": np.asarray([0.01, 0.02]),
            },
        )
    )

    feedback = uut._finalize_macro_feedback_for_runtime(
        fake_env,
        {"semantic_task_success": True, "failed_stage": None},
    )

    assert feedback["batch_size"] == 2
    assert feedback["successful_env_count"] == 2
    assert feedback["failed_env_indices"] == []
    assert feedback["artifacts_recorded_per_env"] is False
    assert [item["env_index"] for item in feedback["per_env_feedback"]] == [0, 1]
    assert all(item["semantic_task_success"] for item in feedback["per_env_feedback"])
    assert [item["instant_is_src_obj_grasped"] for item in feedback["per_env_feedback"]] == [True, True]


def test_finalize_macro_feedback_for_batched_runtime_aggregates_mixed_env_results():
    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=2,
            manip_object_id="orange_cube_ext",
            agent=types.SimpleNamespace(
                is_grasping=lambda _obj: np.asarray([True, False]),
                tcp=types.SimpleNamespace(
                    pose=types.SimpleNamespace(raw_pose=np.zeros((2, 7), dtype=np.float32))
                )
            ),
            object_actors={
                "orange_cube_ext": types.SimpleNamespace(
                    pose=types.SimpleNamespace(raw_pose=np.zeros((2, 7), dtype=np.float32))
                )
            },
            evaluate=lambda: {
                "success": np.asarray([True, False]),
                "is_src_obj_grasped": np.asarray([True, False]),
                "obj_height_above_table": np.asarray([0.08, 0.01]),
                "gripper_obj_dist": np.asarray([0.03, 0.15]),
                "gripper_goal_dist": np.asarray([0.01, 0.20]),
            },
        )
    )

    feedback = uut._finalize_macro_feedback_for_runtime(
        fake_env,
        {"semantic_task_success": True, "failed_stage": None},
    )

    assert feedback["semantic_task_success"] is False
    assert feedback["failed_stage"] is None
    assert feedback["batch_size"] == 2
    assert feedback["successful_env_count"] == 1
    assert feedback["failed_env_indices"] == [1]
    assert [item["env_index"] for item in feedback["per_env_feedback"]] == [0, 1]
    assert [item["semantic_task_success"] for item in feedback["per_env_feedback"]] == [True, False]
    assert [item["instant_is_src_obj_grasped"] for item in feedback["per_env_feedback"]] == [True, False]


def test_finalize_macro_feedback_for_batched_runtime_preserves_existing_per_env_failed_stage():
    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=2,
            manip_object_id="orange_cube_ext",
            agent=types.SimpleNamespace(
                is_grasping=lambda _obj: np.asarray([True, False]),
                tcp=types.SimpleNamespace(
                    pose=types.SimpleNamespace(raw_pose=np.zeros((2, 7), dtype=np.float32))
                ),
            ),
            object_actors={
                "orange_cube_ext": types.SimpleNamespace(
                    pose=types.SimpleNamespace(raw_pose=np.zeros((2, 7), dtype=np.float32))
                )
            },
            evaluate=lambda: {
                "success": np.asarray([True, False]),
                "is_src_obj_grasped": np.asarray([True, False]),
                "obj_height_above_table": np.asarray([0.08, 0.01]),
                "gripper_obj_dist": np.asarray([0.03, 0.15]),
                "gripper_goal_dist": np.asarray([0.01, 0.20]),
            },
        )
    )

    feedback = uut._finalize_macro_feedback_for_runtime(
        fake_env,
        {
            "semantic_task_success": False,
            "failed_stage": None,
            "per_env_feedback": [
                {"env_index": 0, "semantic_task_success": True, "failed_stage": None},
                {"env_index": 1, "semantic_task_success": False, "failed_stage": "full_approach"},
            ],
        },
    )

    assert feedback["semantic_task_success"] is False
    assert feedback["successful_env_count"] == 1
    assert feedback["failed_env_indices"] == [1]
    assert feedback["per_env_feedback"][0]["failed_stage"] is None
    assert feedback["per_env_feedback"][1]["failed_stage"] == "full_approach"


def test_build_direct_stage_callbacks_pass_batched_pose_helpers_to_proxy_pregrasp():
    observed = {}
    sentinel_rows_pose = object()
    sentinel_rows_numpy = object()

    def _capture_pregrasp(_env, _target_pose, **kwargs):
        observed.update(kwargs)
        return True

    def _invoke_pregrasp(_env, **kwargs):
        return kwargs["run_proxy_full_approach_to_pregrasp"](
            _env,
            object(),
            initial_actor_p=np.zeros((3, 3), dtype=np.float32),
            bbox_np=np.ones(3, dtype=np.float32),
            stage_label="pregrasp",
            safe_clearance_z=0.1,
        )

    callbacks = uut._build_direct_stage_callbacks(
        unified_control=types.SimpleNamespace(
            is_proxy_ee_delta_backend=lambda _backend: True,
            is_proxy_then_planner_backend=lambda _backend: False,
            is_rc5_debug_planner_agent=lambda _uid: False,
        ),
        unified_debug=types.SimpleNamespace(
            get_debug_planner_ee_pose_sapien=object(),
            get_debug_planner_ee_pose_rows=sentinel_rows_pose,
            pose_to_numpy_rows=sentinel_rows_numpy,
            pose_to_numpy=object(),
            refresh_render_state=object(),
            set_debug_planner_last_task_pose=object(),
            extract_planner_base_pose=object(),
            resolve_planner_debug_solver_class=object(),
            planner_visuals_supported=object(),
            configure_debug_planner_solver_runtime=object(),
            get_planner_recording_kwargs=object(),
            get_object_specific_planner_profile=object(),
            get_debug_planner_ee_pose=object(),
            get_debug_planner_retention_pose=object(),
            get_debug_planner_retention_pose_rows=object(),
            get_debug_target_object=object(),
            get_debug_actor_position_xyz=object(),
            get_robot_hand_qpos_debug=object(),
            get_robot_hand_qpos_rows_debug=object(),
            get_robot_qpos=object(),
            to_scalar_bool=object(),
        ),
        unified_motion=types.SimpleNamespace(
            execute_planner_pose_with_backend=lambda *_args, **_kwargs: None,
            run_linear_approach_waypoints=lambda *_args, **_kwargs: None,
            run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_pregrasp=_capture_pregrasp,
        ),
        unified_lowlevel=types.SimpleNamespace(
            get_debug_planner_config=object(),
            compute_proxy_rotvec_step=object(),
            build_proxy_delta_pos=object(),
            apply_proxy_ee_delta_action=lambda *_args, **_kwargs: None,
            run_proxy_stationary_settle=lambda *_args, **_kwargs: True,
            run_proxy_guarded_descend_to_object=lambda *_args, **_kwargs: True,
            run_proxy_close_gripper=lambda *_args, **_kwargs: True,
            is_ee_delta_control_mode=lambda _mode: True,
            maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        ),
        unified_targets=types.SimpleNamespace(
            build_object_pregrasp_target=lambda *_args, **_kwargs: None,
            build_object_descend_target=lambda *_args, **_kwargs: None,
        ),
        unified_stage_pose=types.SimpleNamespace(
            run_planner_object_pregrasp_probe=_invoke_pregrasp,
            run_planner_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_planner_object_descend=lambda *_args, **_kwargs: None,
        ),
        unified_stage_close=types.SimpleNamespace(
            run_planner_close_gripper=lambda *_args, **_kwargs: None,
        ),
        unified_stage_lift=types.SimpleNamespace(
            run_planner_lift=lambda *_args, **_kwargs: None,
        ),
    )

    callbacks["run_planner_object_pregrasp_probe"](
        types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=3)),
        backend="proxy_ee_delta",
        execute=True,
    )

    assert observed["get_debug_planner_ee_pose"] is sentinel_rows_pose
    assert observed["pose_to_numpy_rows"] is sentinel_rows_numpy


def test_build_direct_stage_callbacks_default_pregrasp_backend_is_proxy():
    observed = {}

    def _capture_pregrasp(_env, **kwargs):
        observed.update(kwargs)
        return True

    callbacks = uut._build_direct_stage_callbacks(
        unified_control=types.SimpleNamespace(
            is_proxy_ee_delta_backend=lambda _backend: True,
            is_proxy_then_planner_backend=lambda _backend: False,
            is_rc5_debug_planner_agent=lambda _uid: False,
        ),
        unified_debug=types.SimpleNamespace(
            get_debug_planner_ee_pose_sapien=object(),
            get_debug_planner_ee_pose_rows=object(),
            pose_to_numpy_rows=object(),
            pose_to_numpy=object(),
            refresh_render_state=object(),
            set_debug_planner_last_task_pose=object(),
            extract_planner_base_pose=object(),
            resolve_planner_debug_solver_class=object(),
            planner_visuals_supported=object(),
            configure_debug_planner_solver_runtime=object(),
            get_planner_recording_kwargs=object(),
            get_object_specific_planner_profile=object(),
            get_debug_planner_ee_pose=object(),
            get_debug_planner_retention_pose=object(),
            get_debug_planner_retention_pose_rows=object(),
            get_debug_target_object=object(),
            get_debug_actor_position_xyz=object(),
            get_robot_hand_qpos_debug=object(),
            get_robot_hand_qpos_rows_debug=object(),
            get_robot_qpos=object(),
            to_scalar_bool=object(),
        ),
        unified_motion=types.SimpleNamespace(
            execute_planner_pose_with_backend=lambda *_args, **_kwargs: None,
            run_linear_approach_waypoints=lambda *_args, **_kwargs: None,
            run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: None,
        ),
        unified_lowlevel=types.SimpleNamespace(
            get_debug_planner_config=object(),
            compute_proxy_rotvec_step=object(),
            build_proxy_delta_pos=object(),
            apply_proxy_ee_delta_action=lambda *_args, **_kwargs: None,
            run_proxy_stationary_settle=lambda *_args, **_kwargs: True,
            run_proxy_guarded_descend_to_object=lambda *_args, **_kwargs: True,
            run_proxy_close_gripper=lambda *_args, **_kwargs: True,
            is_ee_delta_control_mode=lambda _mode: True,
            maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        ),
        unified_targets=types.SimpleNamespace(
            build_object_pregrasp_target=lambda *_args, **_kwargs: None,
            build_object_descend_target=lambda *_args, **_kwargs: None,
        ),
        unified_stage_pose=types.SimpleNamespace(
            run_planner_object_pregrasp_probe=_capture_pregrasp,
            run_planner_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_planner_object_descend=lambda *_args, **_kwargs: None,
        ),
        unified_stage_close=types.SimpleNamespace(
            run_planner_close_gripper=lambda *_args, **_kwargs: None,
        ),
        unified_stage_lift=types.SimpleNamespace(
            run_planner_lift=lambda *_args, **_kwargs: None,
        ),
    )

    callbacks["run_planner_object_pregrasp_probe"](types.SimpleNamespace(), execute=True)

    assert observed["backend"] == "proxy_ee_delta"


def test_build_direct_stage_callbacks_exposes_proxy_first_aliases():
    callbacks = uut._build_direct_stage_callbacks(
        unified_control=types.SimpleNamespace(
            is_proxy_ee_delta_backend=lambda _backend: True,
            is_proxy_then_planner_backend=lambda _backend: False,
            is_rc5_debug_planner_agent=lambda _uid: False,
        ),
        unified_debug=types.SimpleNamespace(
            get_debug_planner_ee_pose_sapien=object(),
            get_debug_planner_ee_pose_rows=object(),
            pose_to_numpy_rows=object(),
            pose_to_numpy=object(),
            refresh_render_state=object(),
            set_debug_planner_last_task_pose=object(),
            extract_planner_base_pose=object(),
            resolve_planner_debug_solver_class=object(),
            planner_visuals_supported=object(),
            configure_debug_planner_solver_runtime=object(),
            get_planner_recording_kwargs=object(),
            get_object_specific_planner_profile=object(),
            get_debug_planner_ee_pose=object(),
            get_debug_planner_retention_pose=object(),
            get_debug_planner_retention_pose_rows=object(),
            get_debug_target_object=object(),
            get_debug_actor_position_xyz=object(),
            get_debug_actor_position_rows=object(),
            get_robot_hand_qpos_debug=object(),
            get_robot_hand_qpos_rows_debug=object(),
            get_robot_qpos=object(),
            to_scalar_bool=object(),
            build_debug_lift_pose_from_policy=object(),
            log_debug_pre_close_snapshot=object(),
            log_debug_post_close_retention=object(),
            log_debug_non_target_object_contacts=object(),
            get_robot_hand_range_debug=object(),
        ),
        unified_motion=types.SimpleNamespace(
            execute_real_planner_pose_with_backend=lambda *_args, **_kwargs: {"status": "ok"},
            execute_planner_pose_with_backend=lambda *_args, **_kwargs: {"status": "compat"},
            run_linear_approach_waypoints=lambda *_args, **_kwargs: True,
            run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: True,
            run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: True,
            run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: True,
        ),
        unified_lowlevel=types.SimpleNamespace(
            get_debug_planner_config=object(),
            compute_proxy_rotvec_step=object(),
            build_proxy_delta_pos=object(),
            apply_proxy_ee_delta_action=lambda *_args, **_kwargs: None,
            run_proxy_stationary_settle=lambda *_args, **_kwargs: True,
            run_proxy_guarded_descend_to_object=lambda *_args, **_kwargs: True,
            run_proxy_close_gripper=lambda *_args, **_kwargs: True,
            is_ee_delta_control_mode=lambda _mode: True,
            maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        ),
        unified_targets=types.SimpleNamespace(
            build_object_pregrasp_target=lambda *_args, **_kwargs: None,
            build_object_descend_target=lambda *_args, **_kwargs: None,
        ),
        unified_stage_pose=types.SimpleNamespace(
            run_planner_object_pregrasp_probe=lambda *_args, **_kwargs: True,
            run_planner_full_approach_to_descend=lambda *_args, **_kwargs: True,
            run_planner_object_descend=lambda *_args, **_kwargs: True,
        ),
        unified_stage_close=types.SimpleNamespace(
            run_planner_close_gripper=lambda *_args, **_kwargs: True,
        ),
        unified_stage_lift=types.SimpleNamespace(
            run_planner_lift=lambda *_args, **_kwargs: True,
        ),
    )

    assert callbacks["run_proxy_pregrasp_probe"] is callbacks["run_planner_object_pregrasp_probe"]
    assert callbacks["run_proxy_full_approach_to_descend"] is callbacks["run_planner_full_approach_to_descend"]
    assert callbacks["run_proxy_descend"] is callbacks["run_planner_object_descend"]
    assert callbacks["run_proxy_close_gripper"] is callbacks["run_planner_close_gripper"]
    assert callbacks["run_proxy_lift"] is callbacks["run_planner_lift"]
    assert callable(callbacks["execute_real_planner_pose_with_backend"])


def test_direct_executor_passes_proxy_first_callback_contract_to_proxy_macro(monkeypatch):
    observed = {}
    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_uid"),
            num_envs=1,
            control_mode="pd_ee_delta_pose",
            auto_table_z=None,
            manip_object_id="orange_cube_ext",
            _debug_planner_append_video_buffer_frame=None,
        ),
        close=lambda: None,
    )

    monkeypatch.setattr(
        uut,
        "_load_unified_proxy_setup",
        lambda: types.SimpleNamespace(
            inject_compat_shims=lambda: None,
            resolve_scene_path=lambda scene, *_args: scene,
            load_runner_config=lambda _args: {
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
                "control_mode": "pd_ee_delta_pose",
                "gripper_open_signal": 1.0,
            },
            apply_default_scene=lambda _args: None,
            apply_lighting_profile_overrides=lambda _args, _config_overrides: None,
            apply_teleop_profile_overrides=lambda _args, _config_overrides: None,
            validate_teleop_runtime_requirements=lambda _config_overrides: None,
            apply_hand_contact_profile_overrides=lambda _args, _config_overrides: None,
            initialize_sapien_renderer=lambda _renderer_kwargs: object(),
            make_env=lambda _args, _config_overrides, render_mode="none": fake_env,
            apply_hand_controller_profile_overrides=lambda _args, _config_overrides: None,
            apply_hand_pose_overrides=lambda _agent, _args, _config_overrides: None,
            ensure_hand_defaults=lambda _agent: None,
            log_hand_joint_state=lambda _agent, *, label: None,
            reset_and_prepare=lambda _env, _args, _control_mode, gripper_hold_signal=None: None,
        ),
    )
    monkeypatch.setattr(
        uut,
        "_load_unified_proxy_artifacts",
        lambda: types.SimpleNamespace(clear_unified_dense_episode_capture=lambda: None),
    )
    monkeypatch.setattr(
        uut,
        "_load_unified_proxy_macro",
        lambda: types.SimpleNamespace(
            run_proxy_pick_macro=lambda _env, _config_overrides, **kwargs: observed.update(kwargs)
            or uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None),
        ),
    )
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: types.SimpleNamespace())
    monkeypatch.setattr(
        uut,
        "_build_direct_stage_callbacks",
        lambda *_args, **_kwargs: {
            "run_proxy_pregrasp_probe": object(),
            "run_proxy_full_approach_to_descend": object(),
            "run_proxy_descend": object(),
            "run_proxy_close_gripper": object(),
            "run_proxy_lift": object(),
            "run_planner_object_pregrasp_probe": object(),
            "run_planner_full_approach_to_descend": object(),
            "run_planner_object_descend": object(),
            "run_planner_close_gripper": object(),
            "run_planner_lift": object(),
        },
    )
    monkeypatch.setattr(
        uut,
        "_build_direct_args_namespace",
        lambda _argv: types.SimpleNamespace(
            headless=True,
            step_by_step=False,
            num_envs=1,
            sim_backend="physx_cuda",
            scene="scene.json",
            config_path="config.yaml",
            key="demo_key",
            task_object_id=None,
            save_video_on_exit=False,
            save_video_gif_on_exit=False,
            video_output=None,
            save_video_path=None,
            save_video_gif_path=None,
            dense_episode_output=None,
            dense_episode_output_dir=None,
            dense_episode_instruction=None,
            episode_instruction_per_env_json=None,
            dense_episode_target_width=640,
            dense_episode_target_height=480,
            rl4vla_raw_episode_output=None,
            rl4vla_raw_episode_output_dir=None,
            auto_pick_macro="1",
        ),
    )
    monkeypatch.setattr(uut, "shared_reset_planner_hand_target_to_open", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "restore_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "save_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "_maybe_save_debug_side_artifacts", lambda *args, **kwargs: None)

    exit_code = uut._execute_pick_macro_direct(["--headless", "--auto_pick_macro", "1", "--scene", "scene.json"])

    assert exit_code == 0
    assert "run_proxy_pregrasp_probe" in observed
    assert "run_proxy_full_approach_to_descend" in observed
    assert "run_proxy_descend" in observed
    assert "run_proxy_close_gripper" in observed
    assert "run_proxy_lift" in observed
    assert "run_planner_object_pregrasp_probe" not in observed


def test_build_direct_stage_callbacks_pass_batched_helpers_to_lift():
    observed = {}
    sentinel_ee_rows = object()
    sentinel_retention_rows = object()
    sentinel_hand_rows = object()

    def _capture_lift(_env, **kwargs):
        observed.update(kwargs)
        return True

    callbacks = uut._build_direct_stage_callbacks(
        unified_control=types.SimpleNamespace(
            is_proxy_ee_delta_backend=lambda _backend: True,
            is_proxy_then_planner_backend=lambda _backend: False,
            is_rc5_debug_planner_agent=lambda _uid: False,
        ),
        unified_debug=types.SimpleNamespace(
            get_debug_planner_ee_pose_sapien=object(),
            get_debug_planner_ee_pose_rows=sentinel_ee_rows,
            pose_to_numpy_rows=object(),
            pose_to_numpy=object(),
            refresh_render_state=object(),
            set_debug_planner_last_task_pose=object(),
            extract_planner_base_pose=object(),
            resolve_planner_debug_solver_class=object(),
            planner_visuals_supported=object(),
            configure_debug_planner_solver_runtime=object(),
            get_planner_recording_kwargs=object(),
            get_object_specific_planner_profile=object(),
            get_debug_planner_ee_pose=object(),
            get_debug_planner_retention_pose=object(),
            get_debug_planner_retention_pose_rows=sentinel_retention_rows,
            get_debug_target_object=object(),
            get_debug_actor_position_xyz=object(),
            get_robot_hand_qpos_debug=object(),
            get_robot_hand_qpos_rows_debug=sentinel_hand_rows,
            get_robot_qpos=object(),
            to_scalar_bool=object(),
            build_debug_lift_pose_from_policy=object(),
        ),
        unified_motion=types.SimpleNamespace(
            execute_planner_pose_with_backend=lambda *_args, **_kwargs: None,
            run_linear_approach_waypoints=lambda *_args, **_kwargs: None,
            run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: None,
        ),
        unified_lowlevel=types.SimpleNamespace(
            get_debug_planner_config=object(),
            compute_proxy_rotvec_step=object(),
            build_proxy_delta_pos=object(),
            apply_proxy_ee_delta_action=lambda *_args, **_kwargs: None,
            run_proxy_stationary_settle=lambda *_args, **_kwargs: True,
            run_proxy_guarded_descend_to_object=lambda *_args, **_kwargs: True,
            run_proxy_close_gripper=lambda *_args, **_kwargs: True,
            is_ee_delta_control_mode=lambda _mode: True,
            maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        ),
        unified_targets=types.SimpleNamespace(
            build_object_pregrasp_target=lambda *_args, **_kwargs: None,
            build_object_descend_target=lambda *_args, **_kwargs: None,
        ),
        unified_stage_pose=types.SimpleNamespace(
            run_planner_object_pregrasp_probe=lambda *_args, **_kwargs: None,
            run_planner_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_planner_object_descend=lambda *_args, **_kwargs: None,
        ),
        unified_stage_close=types.SimpleNamespace(
            run_planner_close_gripper=lambda *_args, **_kwargs: None,
        ),
        unified_stage_lift=types.SimpleNamespace(
            run_planner_lift=_capture_lift,
        ),
    )

    callbacks["run_planner_lift"](
        types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=3)),
        backend="proxy_ee_delta",
        execute=True,
    )

    assert observed["get_debug_planner_ee_pose_rows"] is sentinel_ee_rows
    assert observed["get_debug_planner_retention_pose_rows"] is sentinel_retention_rows
    assert observed["get_robot_hand_qpos_rows_debug"] is sentinel_hand_rows


def test_build_direct_stage_callbacks_pass_batched_helpers_to_proxy_close():
    observed = {}
    sentinel_ee_rows = object()

    def _capture_lowlevel_close(_env, **kwargs):
        observed.update(kwargs)
        return True

    def _invoke_close(_env, **kwargs):
        return kwargs["run_proxy_close_gripper"](_env, close_steps=20)

    callbacks = uut._build_direct_stage_callbacks(
        unified_control=types.SimpleNamespace(
            is_proxy_ee_delta_backend=lambda _backend: True,
            is_proxy_then_planner_backend=lambda _backend: False,
            is_rc5_debug_planner_agent=lambda _uid: False,
        ),
        unified_debug=types.SimpleNamespace(
            get_debug_planner_ee_pose_sapien=object(),
            get_debug_planner_ee_pose_rows=sentinel_ee_rows,
            pose_to_numpy_rows=object(),
            pose_to_numpy=object(),
            refresh_render_state=object(),
            set_debug_planner_last_task_pose=object(),
            extract_planner_base_pose=object(),
            resolve_planner_debug_solver_class=object(),
            planner_visuals_supported=object(),
            configure_debug_planner_solver_runtime=object(),
            get_planner_recording_kwargs=object(),
            get_object_specific_planner_profile=object(),
            get_debug_planner_ee_pose=object(),
            get_debug_planner_retention_pose=object(),
            get_debug_planner_retention_pose_rows=object(),
            get_debug_target_object=object(),
            get_debug_actor_position_xyz=object(),
            get_robot_hand_qpos_debug=object(),
            get_robot_hand_qpos_rows_debug=object(),
            get_robot_qpos=object(),
            to_scalar_bool=object(),
            build_debug_lift_pose_from_policy=object(),
            log_debug_pre_close_snapshot=object(),
            log_debug_post_close_retention=object(),
            get_robot_hand_range_debug=object(),
        ),
        unified_motion=types.SimpleNamespace(
            execute_planner_pose_with_backend=lambda *_args, **_kwargs: None,
            run_linear_approach_waypoints=lambda *_args, **_kwargs: None,
            run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: None,
        ),
        unified_lowlevel=types.SimpleNamespace(
            get_debug_planner_config=object(),
            compute_proxy_rotvec_step=object(),
            build_proxy_delta_pos=object(),
            apply_proxy_ee_delta_action=lambda *_args, **_kwargs: None,
            run_proxy_stationary_settle=lambda *_args, **_kwargs: True,
            run_proxy_guarded_descend_to_object=lambda *_args, **_kwargs: True,
            run_proxy_close_gripper=_capture_lowlevel_close,
            is_ee_delta_control_mode=lambda _mode: True,
            maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        ),
        unified_targets=types.SimpleNamespace(
            build_object_pregrasp_target=lambda *_args, **_kwargs: None,
            build_object_descend_target=lambda *_args, **_kwargs: None,
        ),
        unified_stage_pose=types.SimpleNamespace(
            run_planner_object_pregrasp_probe=lambda *_args, **_kwargs: None,
            run_planner_full_approach_to_descend=lambda *_args, **_kwargs: None,
            run_planner_object_descend=lambda *_args, **_kwargs: None,
        ),
        unified_stage_close=types.SimpleNamespace(
            run_planner_close_gripper=_invoke_close,
        ),
        unified_stage_lift=types.SimpleNamespace(
            run_planner_lift=lambda *_args, **_kwargs: None,
        ),
    )

    callbacks["run_planner_close_gripper"](
        types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=3)),
        backend="proxy_ee_delta",
    )

    assert observed["get_debug_planner_ee_pose_rows"] is sentinel_ee_rows


def test_build_direct_args_namespace_captures_step_by_step():
    args = uut._build_direct_args_namespace(
        ["--auto_pick_macro", "1", "--scene", "scene.json", "--step_by_step"]
    )

    assert args is not None
    assert args.step_by_step is True


def test_build_direct_args_namespace_captures_task_object_id():
    args = uut._build_direct_args_namespace(
        ["--auto_pick_macro", "1", "--scene", "scene.json", "--task_object_id", "orange_cube_ext"]
    )

    assert args is not None
    assert args.task_object_id == "orange_cube_ext"
    assert args.manip_object_id == "orange_cube_ext"


def test_build_direct_args_namespace_captures_rl4vla_raw_episode_flags():
    args = uut._build_direct_args_namespace(
        [
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--runtime_request_path",
            "/tmp/runtime_request.json",
            "--runtime_config_path_per_env_json",
            "[\"/tmp/episode_0_runtime_config.yaml\", \"/tmp/episode_1_runtime_config.yaml\"]",
            "--runtime_request_path_per_env_json",
            "[\"/tmp/episode_0_runtime_request.json\", \"/tmp/episode_1_runtime_request.json\"]",
            "--rl4vla_raw_episode_output_dir",
            "/tmp/raw",
            "--episode_instruction_per_env_json",
            "[\"Pick up orange cube.\", \"Pick up blue cube.\"]",
        ]
    )

    assert args is not None
    assert args.runtime_request_path == "/tmp/runtime_request.json"
    assert args.runtime_config_path_per_env_json == (
        "[\"/tmp/episode_0_runtime_config.yaml\", \"/tmp/episode_1_runtime_config.yaml\"]"
    )
    assert args.runtime_request_path_per_env_json == (
        "[\"/tmp/episode_0_runtime_request.json\", \"/tmp/episode_1_runtime_request.json\"]"
    )
    assert args.rl4vla_raw_episode_output_dir == "/tmp/raw"
    assert args.episode_instruction_per_env_json == "[\"Pick up orange cube.\", \"Pick up blue cube.\"]"
    assert args.embed_runtime_bundle_in_rl4vla_raw_npz is True


def test_build_direct_args_namespace_can_disable_embedded_runtime_bundle():
    args = uut._build_direct_args_namespace(
        [
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--no-embed_runtime_bundle_in_rl4vla_raw_npz",
        ]
    )

    assert args is not None
    assert args.embed_runtime_bundle_in_rl4vla_raw_npz is False


def test_direct_executor_rejects_headless_step_by_step(monkeypatch):
    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: types.SimpleNamespace(resolve_scene_path=lambda scene, *_args: scene))
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: types.SimpleNamespace(clear_unified_dense_episode_capture=lambda: None))
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_build_direct_stage_callbacks", lambda **_kwargs: {})

    try:
        uut._execute_pick_macro_direct(
            ["--headless", "--step_by_step", "--auto_pick_macro", "1", "--scene", "scene.json"]
        )
    except ValueError as exc:
        message = str(exc)
        assert "fail-fast" in message
        assert "--step_by_step is supported only in viewer mode" in message
    else:
        raise AssertionError("Expected headless step-by-step request to fail fast")


def test_direct_executor_uses_unified_setup_layer(monkeypatch):
    calls = []
    observed = {}
    monkeypatch.setenv("RC5_DEBUG_LOG_SCENE_LAYOUT", "1")

    class _LegacyRuntime:
        DEFAULT_RC5_SIM_BACKEND = "physx_cuda"
        DEFAULT_VIDEO_FPS = 30
        DEFAULT_VIDEO_FORMAT = "mp4"
        AUTO_VIDEO_CODEC = "auto"
        _Y = "\x1b[33m"
        _R = "\x1b[0m"

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_agent"),
            num_envs=1,
            auto_table_z=None,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        ),
        close=lambda: calls.append("env_closed"),
    )

    class _UnifiedSetup:
        sapien = types.SimpleNamespace(Pose=lambda **kwargs: kwargs)

        @staticmethod
        def resolve_scene_path(scene, _config_path, _key):
            calls.append("resolve_scene_path")
            return scene

        @staticmethod
        def load_runner_config(_args):
            calls.append("load_runner_config")
            return {
                "renderer_kwargs": None,
                "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                "gripper_open_signal": 1.0,
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
            }

        @staticmethod
        def apply_lighting_profile_overrides(_args, _config_overrides):
            calls.append("apply_lighting_profile_overrides")

        @staticmethod
        def apply_teleop_profile_overrides(_args, _config_overrides):
            calls.append("apply_teleop_profile_overrides")

        @staticmethod
        def validate_teleop_runtime_requirements(_config_overrides):
            calls.append("validate_teleop_runtime_requirements")

        @staticmethod
        def apply_hand_contact_profile_overrides(_args, _config_overrides):
            calls.append("apply_hand_contact_profile_overrides")

        @staticmethod
        def apply_hand_controller_profile_overrides(_args, _config_overrides):
            calls.append("apply_hand_controller_profile_overrides")

        @staticmethod
        def initialize_sapien_renderer(_renderer_kwargs):
            calls.append("initialize_sapien_renderer")

        @staticmethod
        def make_env(_args, _config_overrides, render_mode="none"):
            calls.append(("make_env", render_mode))
            observed["manip_object_id"] = _config_overrides.get("manip_object_id")
            return fake_env

        @staticmethod
        def apply_hand_pose_overrides(_agent, _args, _config_overrides):
            calls.append("apply_hand_pose_overrides")

        @staticmethod
        def ensure_hand_defaults(_agent):
            calls.append("ensure_hand_defaults")

        @staticmethod
        def log_hand_joint_state(_agent, *, label):
            calls.append(("log_hand_joint_state", label))

        @staticmethod
        def reset_and_prepare(_env, _args, _control_mode, gripper_hold_signal=None):
            calls.append(("reset_and_prepare", gripper_hold_signal))
            return None

    class _UnifiedArtifacts:
        @staticmethod
        def clear_unified_dense_episode_capture():
            calls.append("clear_dense_capture")

    class _UnifiedMacro:
        @staticmethod
        def run_proxy_pick_macro(_env, _config_overrides, **_kwargs):
            calls.append("run_pick_macro")
            uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)

    class _UnifiedStagePose:
        run_planner_object_pregrasp_probe = staticmethod(lambda *_args, **_kwargs: True)
        run_planner_full_approach_to_descend = staticmethod(lambda *_args, **_kwargs: True)
        run_planner_object_descend = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedStageClose:
        run_planner_close_gripper = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedStageLift:
        run_planner_lift = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedControl:
        is_proxy_ee_delta_backend = staticmethod(lambda backend: backend == "proxy_ee_delta")
        is_proxy_then_planner_backend = staticmethod(lambda backend: backend == "proxy_then_planner")
        is_rc5_debug_planner_agent = staticmethod(lambda _uid: False)

    class _UnifiedDebug:
        extract_planner_base_pose = staticmethod(lambda _env: "base_pose")
        resolve_planner_debug_solver_class = staticmethod(lambda _uid: type("Planner", (), {}))
        planner_visuals_supported = staticmethod(lambda _env: False)
        configure_debug_planner_solver_runtime = staticmethod(lambda solver: solver)
        get_planner_recording_kwargs = staticmethod(lambda _env: {})
        pose_to_numpy = staticmethod(lambda _pose: ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        get_debug_planner_ee_pose = staticmethod(lambda _env: None)
        set_debug_planner_last_task_pose = staticmethod(lambda *_args, **_kwargs: None)
        get_debug_target_object = staticmethod(lambda _env: None)
        get_debug_actor_position_xyz = staticmethod(lambda _actor: [0.0, 0.0, 0.0])
        log_debug_pre_close_snapshot = staticmethod(lambda *_args, **_kwargs: None)
        log_debug_post_close_retention = staticmethod(lambda *_args, **_kwargs: (False, None, None, 0.0, 0.0))
        get_robot_hand_qpos_debug = staticmethod(lambda _env: [])
        get_robot_hand_range_debug = staticmethod(lambda _env: (0.0, 0.0))
        to_scalar_bool = staticmethod(lambda value: bool(value))
        get_robot_qpos = staticmethod(lambda _env: [])
        build_debug_lift_pose_from_policy = staticmethod(lambda *_args, **_kwargs: (None, None, None, None))
        refresh_render_state = staticmethod(lambda *_args, **_kwargs: None)
        get_object_specific_planner_profile = staticmethod(lambda *_args, **_kwargs: None)
        log_debug_scene_object_layout = staticmethod(
            lambda _env, stage_label: calls.append(("log_debug_scene_object_layout", stage_label))
        )

    class _UnifiedMotion:
        execute_planner_pose_with_backend = staticmethod(lambda *_args, **_kwargs: {"status": "ok"})
        run_linear_approach_waypoints = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_ee_delta_pose_stage = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_full_approach_to_descend = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_full_approach_to_pregrasp = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedLowlevel:
        is_ee_delta_control_mode = staticmethod(lambda _mode: True)
        get_debug_planner_config = staticmethod(lambda _env: {})
        compute_proxy_rotvec_step = staticmethod(lambda *_args, **_kwargs: ([0.0, 0.0, 0.0], 0.0))
        build_proxy_delta_pos = staticmethod(lambda *_args, **_kwargs: [0.0, 0.0, 0.0])
        apply_proxy_ee_delta_action = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_stationary_settle = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_guarded_descend_to_object = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_close_gripper = staticmethod(lambda *_args, **_kwargs: True)
        maybe_seed_proxy_start_pose = staticmethod(lambda *_args, **_kwargs: True)
        maybe_handle_proxy_viewer_video_hotkey = staticmethod(lambda _env, _viewer: _env.unwrapped._debug_planner_save_video_buffer())
        viewer_key_pressed_once = staticmethod(lambda _env_unwrapped, _viewer, key: key == "q")

    class _UnifiedTargets:
        build_object_pregrasp_target = staticmethod(lambda *_args, **_kwargs: ("obj", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], None))
        build_object_descend_target = staticmethod(lambda *_args, **_kwargs: ("obj", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], None))

    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: _UnifiedSetup)
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts)
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: _UnifiedMacro)
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: _UnifiedControl)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: _UnifiedStagePose)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: _UnifiedStageClose)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: _UnifiedStageLift)
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: _UnifiedDebug)
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: _UnifiedMotion)
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: _UnifiedLowlevel)
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: _UnifiedTargets)

    exit_code = uut._execute_pick_macro_direct(
        ["--headless", "--auto_pick_macro", "1", "--scene", "scene.json"]
    )

    assert exit_code == 0
    assert "resolve_scene_path" in calls
    assert "load_runner_config" in calls
    assert ("make_env", "none") in calls
    assert ("reset_and_prepare", 1.0) in calls
    assert ("log_debug_scene_object_layout", "post_reset_and_prepare") in calls
    assert "run_pick_macro" in calls
    assert "env_closed" in calls


def test_direct_executor_installs_headless_render_guard(monkeypatch):
    calls = []

    class _LegacyRuntime:
        DEFAULT_RC5_SIM_BACKEND = "physx_cuda"
        DEFAULT_VIDEO_FPS = 30
        DEFAULT_VIDEO_FORMAT = "mp4"
        AUTO_VIDEO_CODEC = "auto"
        _Y = "\x1b[33m"
        _R = "\x1b[0m"

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_agent"),
            num_envs=1,
            auto_table_z=None,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        ),
        render_human=lambda: calls.append("render_human"),
        close=lambda: calls.append("env_closed"),
    )

    class _UnifiedSetup:
        sapien = types.SimpleNamespace(Pose=lambda **kwargs: kwargs)

        @staticmethod
        def resolve_scene_path(scene, _config_path, _key):
            return scene

        @staticmethod
        def load_runner_config(_args):
            return {
                "renderer_kwargs": None,
                "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                "gripper_open_signal": 1.0,
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
            }

        @staticmethod
        def apply_lighting_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_teleop_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def validate_teleop_runtime_requirements(_config_overrides):
            return None

        @staticmethod
        def apply_hand_contact_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_hand_controller_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def initialize_sapien_renderer(_renderer_kwargs):
            return None

        @staticmethod
        def make_env(_args, _config_overrides, render_mode="none"):
            calls.append(("make_env", render_mode))
            return fake_env

        @staticmethod
        def apply_hand_pose_overrides(_agent, _args, _config_overrides):
            return None

        @staticmethod
        def ensure_hand_defaults(_agent):
            return None

        @staticmethod
        def log_hand_joint_state(_agent, *, label):
            calls.append(("log_hand_joint_state", label))

        @staticmethod
        def reset_and_prepare(_env, _args, _control_mode, gripper_hold_signal=None):
            return None

    class _UnifiedArtifacts:
        @staticmethod
        def clear_unified_dense_episode_capture():
            return None

    class _UnifiedMacro:
        @staticmethod
        def run_proxy_pick_macro(_env, _config_overrides, **_kwargs):
            uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)
            try:
                _env.render_human()
            except RuntimeError as exc:
                calls.append(str(exc))
            else:
                raise AssertionError("Expected headless render guard to reject render_human()")

    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: _UnifiedSetup())
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts())
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: _UnifiedMacro())
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: types.SimpleNamespace())
    monkeypatch.setattr(
        uut,
        "_build_direct_stage_callbacks",
        lambda *_args, **_kwargs: {
            "run_planner_object_pregrasp_probe": object(),
            "run_planner_full_approach_to_descend": object(),
            "run_planner_object_descend": object(),
            "run_planner_close_gripper": object(),
            "run_planner_lift": object(),
        },
    )
    monkeypatch.setattr(uut, "_build_direct_args_namespace", lambda _argv: types.SimpleNamespace(
        headless=True,
        step_by_step=False,
        num_envs=1,
        sim_backend="physx_cuda",
        scene="scene.json",
        config_path="config.yaml",
        key="demo_key",
        task_object_id=None,
        save_video_on_exit=False,
        save_video_gif_on_exit=False,
        video_output=None,
        save_video_path=None,
        save_video_gif_path=None,
        dense_episode_output=None,
        dense_episode_output_dir=None,
        dense_episode_instruction=None,
        episode_instruction_per_env_json=None,
        dense_episode_target_width=640,
        dense_episode_target_height=480,
        rl4vla_raw_episode_output=None,
        rl4vla_raw_episode_output_dir=None,
        auto_pick_macro="1",
    ))
    monkeypatch.setattr(uut, "shared_reset_planner_hand_target_to_open", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "restore_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "save_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "_maybe_save_debug_side_artifacts", lambda *args, **kwargs: None)

    result = uut._execute_pick_macro_direct(["--headless", "--auto_pick_macro", "1", "--scene", "scene.json"])

    assert result == 0
    assert ("make_env", "none") in calls
    assert fake_env.unwrapped._debug_planner_headless is True
    assert any("headless guard" in entry for entry in calls if isinstance(entry, str))
    assert "render_human" not in calls
    assert "env_closed" in calls


def test_direct_executor_overrides_manip_object_id_from_task_object_id(monkeypatch):
    observed = {}

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_agent"),
            num_envs=1,
            auto_table_z=None,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            manip_object_id="orange_cube_ext",
        ),
        close=lambda: None,
    )

    class _UnifiedSetup:
        sapien = types.SimpleNamespace(Pose=lambda **kwargs: kwargs)

        @staticmethod
        def resolve_scene_path(scene, _config_path, _key):
            return scene

        @staticmethod
        def load_runner_config(_args):
            return {
                "renderer_kwargs": None,
                "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                "gripper_open_signal": 1.0,
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
                "manip_object_id": "yellow_cube_ext",
            }

        @staticmethod
        def apply_lighting_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_teleop_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def validate_teleop_runtime_requirements(_config_overrides):
            return None

        @staticmethod
        def apply_hand_contact_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_hand_controller_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def initialize_sapien_renderer(_renderer_kwargs):
            return None

        @staticmethod
        def make_env(_args, _config_overrides, render_mode="none"):
            observed["manip_object_id"] = _config_overrides.get("manip_object_id")
            observed["render_mode"] = render_mode
            return fake_env

        @staticmethod
        def apply_hand_pose_overrides(_agent, _args, _config_overrides):
            return None

        @staticmethod
        def ensure_hand_defaults(_agent):
            return None

        @staticmethod
        def maybe_apply_auto_robot_base_pose_z(_env, _base_pose):
            return None

        @staticmethod
        def maybe_adjust_render_camera_from_env(_env, _config_overrides):
            return None

        @staticmethod
        def maybe_configure_debug_viewer(_env, _args):
            return None

        @staticmethod
        def log_hand_joint_state(_agent, *, label):
            return None

        @staticmethod
        def reset_and_prepare(_env, _args, _control_mode, gripper_hold_signal=None):
            return None

    class _UnifiedArtifacts:
        @staticmethod
        def clear_unified_dense_episode_capture():
            return None

        @staticmethod
        def maybe_start_unified_dense_episode_capture(*_args, **_kwargs):
            return None

        @staticmethod
        def maybe_finalize_unified_dense_episode_capture(*_args, **_kwargs):
            return None

    class _UnifiedMacro:
        @staticmethod
        def run_proxy_pick_macro(_env, _config_overrides, **_kwargs):
            uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)

    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: _UnifiedSetup())
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts())
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: _UnifiedMacro())
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: types.SimpleNamespace())
    monkeypatch.setattr(
        uut,
        "_build_direct_stage_callbacks",
        lambda *_args, **_kwargs: {
            "run_planner_object_pregrasp_probe": object(),
            "run_planner_full_approach_to_descend": object(),
            "run_planner_object_descend": object(),
            "run_planner_close_gripper": object(),
            "run_planner_lift": object(),
        },
    )
    monkeypatch.setattr(uut, "shared_reset_planner_hand_target_to_open", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "restore_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "save_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "_maybe_save_debug_side_artifacts", lambda *args, **kwargs: None)

    result = uut._execute_pick_macro_direct(
        [
            "--headless",
            "--auto_pick_macro",
            "1",
            "--scene",
            "scene.json",
            "--task_object_id",
            "orange_cube_ext",
        ]
    )

    assert result == 0
    assert observed["render_mode"] == "none"
    assert observed["manip_object_id"] == "orange_cube_ext"


def test_direct_executor_fails_fast_on_runtime_task_object_mismatch(monkeypatch):
    calls = []

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_agent"),
            num_envs=1,
            auto_table_z=None,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            manip_object_id="yellow_cube_ext",
        ),
        close=lambda: calls.append("env_closed"),
    )

    class _UnifiedSetup:
        sapien = types.SimpleNamespace(Pose=lambda **kwargs: kwargs)

        @staticmethod
        def resolve_scene_path(scene, _config_path, _key):
            return scene

        @staticmethod
        def load_runner_config(_args):
            return {
                "renderer_kwargs": None,
                "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                "gripper_open_signal": 1.0,
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
                "manip_object_id": "yellow_cube_ext",
            }

        @staticmethod
        def apply_lighting_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_teleop_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def validate_teleop_runtime_requirements(_config_overrides):
            return None

        @staticmethod
        def apply_hand_contact_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_hand_controller_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def initialize_sapien_renderer(_renderer_kwargs):
            return None

        @staticmethod
        def make_env(_args, _config_overrides, render_mode="none"):
            calls.append(("make_env", render_mode))
            return fake_env

    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: _UnifiedSetup())
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: types.SimpleNamespace(clear_unified_dense_episode_capture=lambda: None))
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: types.SimpleNamespace())
    monkeypatch.setattr(uut, "_build_direct_stage_callbacks", lambda *_args, **_kwargs: {})

    try:
        uut._execute_pick_macro_direct(
            [
                "--headless",
                "--auto_pick_macro",
                "1",
                "--scene",
                "scene.json",
                "--task_object_id",
                "orange_cube_ext",
            ]
        )
    except ValueError as exc:
        message = str(exc)
        assert "fail-fast" in message
        assert "task-object mismatch" in message
        assert "orange_cube_ext" in message
        assert "yellow_cube_ext" in message
    else:
        raise AssertionError("Expected runtime task-object mismatch to fail fast")

    assert ("make_env", "none") in calls
    assert "env_closed" in calls


def test_direct_executor_supports_viewer_mode_with_human_render_and_gpu_backend(monkeypatch):
    calls = []

    class _LegacyRuntime:
        DEFAULT_RC5_SIM_BACKEND = "physx_cuda"
        DEFAULT_VIDEO_FPS = 30
        DEFAULT_VIDEO_FORMAT = "mp4"
        AUTO_VIDEO_CODEC = "auto"
        _Y = "\x1b[33m"
        _R = "\x1b[0m"

    class _ViewerWindow:
        @staticmethod
        def key_down(_key):
            return True

    class _Viewer:
        def __init__(self):
            self.closed = False
            self.paused = True
            self.window = _ViewerWindow()

        def render(self):
            calls.append("viewer_render")

        def close(self):
            calls.append("viewer_close")
            self.closed = True

    viewer = _Viewer()
    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_agent"),
            num_envs=1,
            auto_table_z=None,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            scene=types.SimpleNamespace(update_render=lambda: None, sensors={}),
            capture_sensor_data=lambda: None,
        ),
        close=lambda: calls.append("env_closed"),
        render=lambda: viewer,
    )

    class _UnifiedSetup:
        sapien = types.SimpleNamespace(Pose=lambda **kwargs: kwargs)

        @staticmethod
        def resolve_scene_path(scene, _config_path, _key):
            return scene

        @staticmethod
        def load_runner_config(_args):
            return {
                "renderer_kwargs": None,
                "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                "gripper_open_signal": 1.0,
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
            }

        @staticmethod
        def apply_lighting_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_teleop_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def validate_teleop_runtime_requirements(_config_overrides):
            return None

        @staticmethod
        def apply_hand_contact_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def apply_hand_controller_profile_overrides(_args, _config_overrides):
            return None

        @staticmethod
        def initialize_sapien_renderer(_renderer_kwargs):
            return None

        @staticmethod
        def make_env(_args, _config_overrides, render_mode="none"):
            calls.append(("make_env", render_mode, _args.sim_backend))
            return fake_env

        @staticmethod
        def apply_hand_pose_overrides(_agent, _args, _config_overrides):
            return None

        @staticmethod
        def ensure_hand_defaults(_agent):
            return None

        @staticmethod
        def log_hand_joint_state(_agent, *, label):
            return None

        @staticmethod
        def reset_and_prepare(_env, _args, _control_mode, gripper_hold_signal=None):
            calls.append(("reset_and_prepare", gripper_hold_signal))
            return None

    class _UnifiedArtifacts:
        @staticmethod
        def clear_unified_dense_episode_capture():
            return None

        @staticmethod
        def resolve_effective_video_settings(_args, _config_overrides):
            return {
                "fps": 30,
                "format": "mp4",
                "requested_codec": "auto",
                "output_params": None,
            }

        @staticmethod
        def resolve_effective_video_codec(_video_format, _requested_codec):
            return ("mpeg4", None, "test codec")

        @staticmethod
        def capture_base_camera_frame(_env):
            calls.append("capture_base_camera_frame")
            return "frame"

        @staticmethod
        def normalize_video_frame(frame):
            return frame

        @staticmethod
        def resolve_output_video_path(video_path, *, video_format=None, video_codec=None):
            return Path(video_path)

        @staticmethod
        def flush_video_buffer_to_file(video_frames, video_path, *_args, **_kwargs):
            calls.append(("flush_video_buffer_to_file", list(video_frames), str(video_path)))
            return True

        @staticmethod
        def write_debug_video_gif_from_video(video_path, gif_path):
            calls.append(("write_debug_video_gif_from_video", str(video_path), str(gif_path)))
            return gif_path

    class _UnifiedMacro:
        @staticmethod
        def run_proxy_pick_macro(_env, _config_overrides, **_kwargs):
            calls.append("run_pick_macro")
            uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)

    class _UnifiedStagePose:
        run_planner_object_pregrasp_probe = staticmethod(lambda *_args, **_kwargs: True)
        run_planner_full_approach_to_descend = staticmethod(lambda *_args, **_kwargs: True)
        run_planner_object_descend = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedStageClose:
        run_planner_close_gripper = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedStageLift:
        run_planner_lift = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedControl:
        is_proxy_ee_delta_backend = staticmethod(lambda backend: backend == "proxy_ee_delta")
        is_proxy_then_planner_backend = staticmethod(lambda backend: backend == "proxy_then_planner")
        is_rc5_debug_planner_agent = staticmethod(lambda _uid: False)

    class _UnifiedDebug:
        extract_planner_base_pose = staticmethod(lambda _env: "base_pose")
        resolve_planner_debug_solver_class = staticmethod(lambda _uid: type("Planner", (), {}))
        planner_visuals_supported = staticmethod(lambda _env: False)
        configure_debug_planner_solver_runtime = staticmethod(lambda solver: solver)
        get_planner_recording_kwargs = staticmethod(lambda _env: {})
        pose_to_numpy = staticmethod(lambda _pose: ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        get_debug_planner_ee_pose = staticmethod(lambda _env: None)
        set_debug_planner_last_task_pose = staticmethod(lambda *_args, **_kwargs: None)
        get_debug_target_object = staticmethod(lambda _env: None)
        get_debug_actor_position_xyz = staticmethod(lambda _actor: [0.0, 0.0, 0.0])
        log_debug_pre_close_snapshot = staticmethod(lambda *_args, **_kwargs: None)
        log_debug_post_close_retention = staticmethod(lambda *_args, **_kwargs: (False, None, None, 0.0, 0.0))
        get_robot_hand_qpos_debug = staticmethod(lambda _env: [])
        get_robot_hand_range_debug = staticmethod(lambda _env: (0.0, 0.0))
        to_scalar_bool = staticmethod(lambda value: bool(value))
        get_robot_qpos = staticmethod(lambda _env: [])
        build_debug_lift_pose_from_policy = staticmethod(lambda *_args, **_kwargs: (None, None, None, None))
        refresh_render_state = staticmethod(lambda *_args, **_kwargs: None)
        get_object_specific_planner_profile = staticmethod(lambda *_args, **_kwargs: None)

    class _UnifiedMotion:
        execute_planner_pose_with_backend = staticmethod(lambda *_args, **_kwargs: {"status": "ok"})
        run_linear_approach_waypoints = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_ee_delta_pose_stage = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_full_approach_to_descend = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_full_approach_to_pregrasp = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedLowlevel:
        is_ee_delta_control_mode = staticmethod(lambda _mode: True)
        get_debug_planner_config = staticmethod(lambda _env: {})
        compute_proxy_rotvec_step = staticmethod(lambda *_args, **_kwargs: ([0.0, 0.0, 0.0], 0.0))
        build_proxy_delta_pos = staticmethod(lambda *_args, **_kwargs: [0.0, 0.0, 0.0])
        apply_proxy_ee_delta_action = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_stationary_settle = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_guarded_descend_to_object = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_close_gripper = staticmethod(lambda *_args, **_kwargs: True)
        maybe_seed_proxy_start_pose = staticmethod(lambda *_args, **_kwargs: True)
        maybe_handle_proxy_viewer_video_hotkey = staticmethod(
            lambda _env, _viewer: _env.unwrapped._debug_planner_save_video_buffer()
        )
        viewer_key_pressed_once = staticmethod(lambda _env_unwrapped, _viewer, key: key == "q")

    class _UnifiedTargets:
        build_object_pregrasp_target = staticmethod(lambda *_args, **_kwargs: ("obj", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], None))
        build_object_descend_target = staticmethod(lambda *_args, **_kwargs: ("obj", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], None))

    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: _UnifiedSetup)
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts)
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: _UnifiedMacro)
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: _UnifiedControl)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: _UnifiedStagePose)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: _UnifiedStageClose)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: _UnifiedStageLift)
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: _UnifiedDebug)
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: _UnifiedMotion)
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: _UnifiedLowlevel)
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: _UnifiedTargets)

    exit_code = uut._execute_pick_macro_direct(
        ["--auto_pick_macro", "1", "--scene", "scene.json", "--sim_backend", "physx_cuda"]
    )

    assert exit_code == 0
    assert ("make_env", "human", "physx_cuda") in calls
    assert ("reset_and_prepare", 1.0) in calls
    assert "capture_base_camera_frame" in calls
    assert "run_pick_macro" in calls
    assert any(
        isinstance(entry, tuple) and entry[0] == "flush_video_buffer_to_file"
        for entry in calls
    )
    assert "viewer_render" in calls
    assert "viewer_close" in calls
    assert "env_closed" in calls


def test_direct_executor_viewer_video_save_uses_container_compatible_suffix(monkeypatch):
    calls = []

    class _ViewerWindow:
        @staticmethod
        def key_down(_key):
            return True

    class _Viewer:
        def __init__(self):
            self.closed = False
            self.paused = True
            self.window = _ViewerWindow()

        def render(self):
            calls.append("viewer_render")

        def close(self):
            self.closed = True

    viewer = _Viewer()
    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_agent"),
            num_envs=1,
            auto_table_z=None,
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            scene=types.SimpleNamespace(update_render=lambda: None, sensors={}),
            capture_sensor_data=lambda: None,
        ),
        close=lambda: None,
        render=lambda: viewer,
    )

    class _UnifiedSetup:
        sapien = types.SimpleNamespace(Pose=lambda **kwargs: kwargs)
        resolve_scene_path = staticmethod(lambda scene, *_args: scene)
        load_runner_config = staticmethod(
            lambda _args: {
                "renderer_kwargs": None,
                "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                "gripper_open_signal": 1.0,
                "auto_placement": False,
                "robot_base_pose_z_auto": False,
                "robot_base_pose": None,
                "video_config": {"format": "mp4", "codec": "ffv1"},
            }
        )
        apply_lighting_profile_overrides = staticmethod(lambda *_args, **_kwargs: None)
        apply_teleop_profile_overrides = staticmethod(lambda *_args, **_kwargs: None)
        validate_teleop_runtime_requirements = staticmethod(lambda *_args, **_kwargs: None)
        apply_hand_contact_profile_overrides = staticmethod(lambda *_args, **_kwargs: None)
        apply_hand_controller_profile_overrides = staticmethod(lambda *_args, **_kwargs: None)
        initialize_sapien_renderer = staticmethod(lambda *_args, **_kwargs: None)
        make_env = staticmethod(lambda _args, _config_overrides, render_mode="none": fake_env)
        apply_hand_pose_overrides = staticmethod(lambda *_args, **_kwargs: None)
        ensure_hand_defaults = staticmethod(lambda *_args, **_kwargs: None)
        log_hand_joint_state = staticmethod(lambda *_args, **_kwargs: None)
        reset_and_prepare = staticmethod(lambda *_args, **_kwargs: None)

    class _UnifiedArtifacts:
        clear_unified_dense_episode_capture = staticmethod(lambda: None)
        resolve_effective_video_settings = staticmethod(
            lambda _args, _config_overrides: {
                "fps": 30,
                "format": "mp4",
                "requested_codec": "ffv1",
                "output_params": None,
            }
        )
        resolve_effective_video_codec = staticmethod(lambda _video_format, _requested_codec: ("ffv1", None, "explicit CLI override (ffv1)"))
        normalize_video_frame = staticmethod(lambda frame: frame)
        capture_base_camera_frame = staticmethod(lambda _env: "frame")

        @staticmethod
        def resolve_output_video_path(video_path, *, video_format=None, video_codec=None):
            assert video_format == "mp4"
            assert video_codec == "ffv1"
            return Path(video_path).with_suffix(".mkv")

        @staticmethod
        def flush_video_buffer_to_file(video_frames, video_path, *_args, **_kwargs):
            calls.append(("flush_video_buffer_to_file", str(video_path), list(video_frames)))
            return True

        @staticmethod
        def write_debug_video_gif_from_video(video_path, gif_path):
            calls.append(("write_debug_video_gif_from_video", str(video_path), str(gif_path)))
            return gif_path

    class _UnifiedMacro:
        @staticmethod
        def run_proxy_pick_macro(_env, _config_overrides, **_kwargs):
            uut.set_unified_macro_feedback(semantic_task_success=True, failed_stage=None)

    class _UnifiedStagePose:
        run_planner_object_pregrasp_probe = staticmethod(lambda *_args, **_kwargs: True)
        run_planner_full_approach_to_descend = staticmethod(lambda *_args, **_kwargs: True)
        run_planner_object_descend = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedStageClose:
        run_planner_close_gripper = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedStageLift:
        run_planner_lift = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedControl:
        is_proxy_ee_delta_backend = staticmethod(lambda backend: backend == "proxy_ee_delta")
        is_proxy_then_planner_backend = staticmethod(lambda backend: backend == "proxy_then_planner")
        is_rc5_debug_planner_agent = staticmethod(lambda _uid: False)

    class _UnifiedDebug:
        extract_planner_base_pose = staticmethod(lambda _env: "base_pose")
        resolve_planner_debug_solver_class = staticmethod(lambda _uid: type("Planner", (), {}))
        planner_visuals_supported = staticmethod(lambda _env: False)
        configure_debug_planner_solver_runtime = staticmethod(lambda solver: solver)
        get_planner_recording_kwargs = staticmethod(lambda _env: {})
        pose_to_numpy = staticmethod(lambda _pose: ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        get_debug_planner_ee_pose = staticmethod(lambda _env: None)
        set_debug_planner_last_task_pose = staticmethod(lambda *_args, **_kwargs: None)
        get_debug_target_object = staticmethod(lambda _env: None)
        get_debug_actor_position_xyz = staticmethod(lambda _actor: [0.0, 0.0, 0.0])
        log_debug_pre_close_snapshot = staticmethod(lambda *_args, **_kwargs: None)
        log_debug_post_close_retention = staticmethod(lambda *_args, **_kwargs: (False, None, None, 0.0, 0.0))
        get_robot_hand_qpos_debug = staticmethod(lambda _env: [])
        get_robot_hand_range_debug = staticmethod(lambda _env: (0.0, 0.0))
        to_scalar_bool = staticmethod(lambda value: bool(value))
        get_robot_qpos = staticmethod(lambda _env: [])
        build_debug_lift_pose_from_policy = staticmethod(lambda *_args, **_kwargs: (None, None, None, None))
        refresh_render_state = staticmethod(lambda *_args, **_kwargs: None)
        get_object_specific_planner_profile = staticmethod(lambda *_args, **_kwargs: None)

    class _UnifiedMotion:
        execute_planner_pose_with_backend = staticmethod(lambda *_args, **_kwargs: {"status": "ok"})
        run_linear_approach_waypoints = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_ee_delta_pose_stage = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_full_approach_to_descend = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_full_approach_to_pregrasp = staticmethod(lambda *_args, **_kwargs: True)

    class _UnifiedLowlevel:
        is_ee_delta_control_mode = staticmethod(lambda _mode: True)
        get_debug_planner_config = staticmethod(lambda _env: {})
        compute_proxy_rotvec_step = staticmethod(lambda *_args, **_kwargs: ([0.0, 0.0, 0.0], 0.0))
        build_proxy_delta_pos = staticmethod(lambda *_args, **_kwargs: [0.0, 0.0, 0.0])
        apply_proxy_ee_delta_action = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_stationary_settle = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_guarded_descend_to_object = staticmethod(lambda *_args, **_kwargs: True)
        run_proxy_close_gripper = staticmethod(lambda *_args, **_kwargs: True)
        maybe_seed_proxy_start_pose = staticmethod(lambda *_args, **_kwargs: True)
        maybe_handle_proxy_viewer_video_hotkey = staticmethod(lambda _env, _viewer: _env.unwrapped._debug_planner_save_video_buffer())
        viewer_key_pressed_once = staticmethod(lambda _env_unwrapped, _viewer, key: key == "q")

    class _UnifiedTargets:
        build_object_pregrasp_target = staticmethod(lambda *_args, **_kwargs: ("obj", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], None))
        build_object_descend_target = staticmethod(lambda *_args, **_kwargs: ("obj", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], None))

    monkeypatch.setattr(uut, "_load_unified_proxy_setup", lambda: _UnifiedSetup)
    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts)
    monkeypatch.setattr(uut, "_load_unified_proxy_macro", lambda: _UnifiedMacro)
    monkeypatch.setattr(uut, "_load_unified_proxy_control", lambda: _UnifiedControl)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_pose", lambda: _UnifiedStagePose)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_close", lambda: _UnifiedStageClose)
    monkeypatch.setattr(uut, "_load_unified_proxy_stage_lift", lambda: _UnifiedStageLift)
    monkeypatch.setattr(uut, "_load_unified_proxy_debug", lambda: _UnifiedDebug)
    monkeypatch.setattr(uut, "_load_unified_proxy_motion", lambda: _UnifiedMotion)
    monkeypatch.setattr(uut, "_load_unified_proxy_lowlevel", lambda: _UnifiedLowlevel)
    monkeypatch.setattr(uut, "_load_unified_proxy_targets", lambda: _UnifiedTargets)
    monkeypatch.setattr(uut, "shared_reset_planner_hand_target_to_open", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "restore_planner_grasp_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(uut, "save_planner_grasp_state", lambda *args, **kwargs: None)

    exit_code = uut._execute_pick_macro_direct(["--auto_pick_macro", "1", "--scene", "scene.json"])

    assert exit_code == 0
    assert any(
        entry[0] == "flush_video_buffer_to_file" and entry[1].endswith("_success.mkv")
        for entry in calls
        if isinstance(entry, tuple)
    )
    assert any(
        entry[0] == "write_debug_video_gif_from_video"
        and entry[1].endswith("_success.mkv")
        and entry[2].endswith("_success.gif")
        for entry in calls
        if isinstance(entry, tuple)
    )


def test_runtime_dense_artifact_writer_uses_unified_artifacts_layer(monkeypatch):
    observed = {}

    class _UnifiedArtifacts:
        @staticmethod
        def write_dense_episode_artifact_if_available(**kwargs):
            observed.update(kwargs)
            return "/tmp/fake_dense_episode.npz"

    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts)

    artifact_path = uut._write_dense_episode_artifact_if_available(
        planner_backend_value="proxy_ee_delta",
        macro_route="pick_macro_1",
        exit_code=0,
        macro_feedback={"semantic_task_success": True, "failed_stage": None},
    )

    assert artifact_path == "/tmp/fake_dense_episode.npz"
    assert observed["planner_backend_value"] == "proxy_ee_delta"
    assert observed["macro_route"] == "pick_macro_1"
    assert observed["exit_code"] == 0


def test_build_runtime_events_exposes_macro_backend_field():
    request = types.SimpleNamespace(
        task_plan=types.SimpleNamespace(
            intent=types.SimpleNamespace(task_type="pick_up"),
            stages=[types.SimpleNamespace(name="pregrasp", kind="move_to_pregrasp")],
        )
    )

    events = uut._build_runtime_events(
        request,
        planner_backend_value="proxy_ee_delta",
        macro_route="pick_macro_1",
    )

    assert events[0]["payload"]["macro_backend"] == "proxy_ee_delta"
    assert events[0]["payload"]["planner_backend"] == "proxy_ee_delta"
    assert events[1]["payload"]["macro_backend"] == "proxy_ee_delta"


def test_independent_batched_proxy_finalizes_per_env_artifacts_and_times_out_after_first_success():
    tcp_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [0.50, 0.50, 0.36],
        ],
        dtype=np.float32,
    )
    actor_rows = np.asarray(
        [
            [-0.30, -0.70, 0.28],
            [0.20, 0.20, 0.28],
        ],
        dtype=np.float32,
    )
    descend_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [0.20, 0.20, 0.36],
        ],
        dtype=np.float32,
    )
    lift_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [0.20, 0.20, 0.46],
        ],
        dtype=np.float32,
    )
    video_frames_per_env = [
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        [np.zeros((2, 2, 3), dtype=np.uint8)],
    ]
    saved_video_envs = []
    saved_gif_pairs = []
    finalized_dense_envs = []

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=2,
            agent=types.SimpleNamespace(uid="fake_uid"),
            evaluate=lambda: {"success": np.asarray([True, False], dtype=bool)},
        )
    )

    class _UnifiedArtifacts:
        @staticmethod
        def resolve_batched_debug_video_output_path(output_dir, *, env_index):
            return Path(output_dir) / f"env_{int(env_index):06d}.mkv"

        @staticmethod
        def resolve_batched_debug_video_gif_output_path(output_dir, *, env_index):
            return Path(output_dir) / f"env_{int(env_index):06d}.gif"

        @staticmethod
        def write_debug_video_gif_from_video(video_path, gif_path):
            saved_gif_pairs.append((Path(video_path), Path(gif_path)))
            return gif_path

        @staticmethod
        def finalize_batched_dense_episode_env_if_available(**kwargs):
            finalized_dense_envs.append(
                (
                    int(kwargs["env_index"]),
                    bool(kwargs["semantic_task_success"]),
                    kwargs["failed_stage"],
                    int(kwargs["exit_code"]),
                )
            )
            return Path("/tmp") / f"env_{int(kwargs['env_index']):06d}_dense_episode.npz"

    class _UnifiedControl:
        resolve_macro_backend = staticmethod(lambda _config_overrides, default_backend=None: "proxy_ee_delta")
        is_proxy_ee_delta_backend = staticmethod(lambda backend: backend == "proxy_ee_delta")
        get_default_debug_close_steps = staticmethod(lambda _uid: 1)

    class _UnifiedDebug:
        get_debug_target_object = staticmethod(lambda _env_unwrapped: object())
        get_debug_actor_position_rows = staticmethod(lambda _target_object: actor_rows.copy())
        get_debug_planner_ee_pose_rows = staticmethod(lambda _env_unwrapped: "tcp_pose")
        collect_debug_robot_object_contacts = staticmethod(lambda *_args, **_kwargs: [])
        collect_debug_object_object_contacts = staticmethod(lambda *_args, **_kwargs: [])

        @staticmethod
        def pose_to_numpy_rows(pose):
            if pose == "tcp_pose":
                quat_rows = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1))
                return tcp_rows.copy(), quat_rows
            if pose == "descend_pose":
                quat_rows = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1))
                return descend_rows.copy(), quat_rows
            if pose == "lift_pose":
                quat_rows = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1))
                return lift_rows.copy(), quat_rows
            raise AssertionError(f"Unexpected pose marker: {pose!r}")

        build_debug_lift_pose_from_policy = staticmethod(
            lambda *_args, **_kwargs: ("lift_pose", None, None, None)
        )
        log_debug_non_target_object_contacts = staticmethod(lambda *_args, **_kwargs: None)
        pose_to_numpy = staticmethod(lambda _pose: (np.zeros(3, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))

    class _UnifiedLowlevel:
        @staticmethod
        def get_debug_planner_config(_env_unwrapped):
                return {
                    "planner_proxy_safe_clearance_z": 0.05,
                    "planner_proxy_predescent_settle_steps": 0,
                    "planner_proxy_preclose_settle_steps": 0,
                    "planner_proxy_max_stage_steps": 1000,
                "planner_proxy_stall_steps": 1000,
                "planner_proxy_pos_tol_m": 0.01,
                "planner_proxy_xy_step_m": 0.01,
                "planner_proxy_z_step_m": 0.01,
                "planner_proxy_batch_env_step_timeout_scale": 2.0,
                "planner_proxy_batch_env_step_timeout_min_steps": 0,
                "planner_proxy_batch_runaway_abs_limit_m": 5.0,
                "planner_waypoints": {"enabled": False, "points": []},
            }

        get_required_planner_waypoints_config = staticmethod(lambda planner_cfg: (False, []))
        get_required_proxy_adaptive_step_config = staticmethod(lambda planner_cfg: (True, 0.01, 0.01, 0.01))
        get_adaptive_proxy_xy_step = staticmethod(
            lambda _planner_cfg, *, xy_err, nominal_xy_step: nominal_xy_step
        )
        get_adaptive_proxy_z_step_near_target = staticmethod(
            lambda _planner_cfg, *, tcp_z, descend_target_z, nominal_z_step: nominal_z_step
        )

        _get_proxy_ee_gripper_controller_signal = staticmethod(
            lambda _env_unwrapped, target_state: 1.0 if target_state == "open" else -1.0
        )
        build_proxy_delta_pos = staticmethod(lambda *_args, **_kwargs: np.asarray([0.01, 0.0, 0.0], dtype=np.float32))
        apply_proxy_ee_delta_action = staticmethod(lambda *_args, **_kwargs: True)

    def _build_object_descend_target(_env, *, extra_clearance=0.03):
        del extra_clearance
        return (
            "orange_cube_ext",
            actor_rows.copy(),
            np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
            "descend_pose",
        )

    def _save_video_buffer_for_env_to_path(env_index, video_path):
        saved_video_envs.append((int(env_index), Path(video_path)))
        return Path(video_path)

    args = types.SimpleNamespace(
        auto_pick_macro="1",
        num_envs=2,
        batched_save_video_output_dir="/tmp/debug_videos",
        batched_save_video_gif_output_dir="/tmp/debug_gifs",
        dense_episode_output_dir="/tmp/dense",
    )

    success = uut._execute_independent_batched_proxy_pick_macro(
        fake_env,
        args=args,
        config_overrides={"planner_lift_delta_z": 0.05},
        unified_artifacts=_UnifiedArtifacts,
        unified_control=_UnifiedControl,
        unified_debug=_UnifiedDebug,
        unified_lowlevel=_UnifiedLowlevel,
        build_object_descend_target=_build_object_descend_target,
        save_video_buffer_for_env_to_path=_save_video_buffer_for_env_to_path,
        video_frames_per_env=video_frames_per_env,
        finalized_video_env_indices=set(),
    )

    assert success is False
    macro_feedback = uut.peek_unified_macro_feedback()
    assert macro_feedback["successful_env_count"] == 1
    assert macro_feedback["failed_env_indices"] == [1]
    assert finalized_dense_envs[0] == (0, True, None, 0)
    assert finalized_dense_envs[1][0] == 1
    assert finalized_dense_envs[1][1] is False
    assert finalized_dense_envs[1][2] == "move_xy"
    assert finalized_dense_envs[1][3] == 1
    assert [item[0] for item in saved_video_envs] == [0, 1]
    assert [item[0] for item in finalized_dense_envs] == [0, 1]
    assert len(saved_gif_pairs) == 2
    assert video_frames_per_env == [[], []]


def test_independent_batched_proxy_fails_fast_on_runaway_pose():
    tcp_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [99.0, 0.0, 0.36],
        ],
        dtype=np.float32,
    )
    actor_rows = np.asarray(
        [
            [-0.30, -0.70, 0.28],
            [0.20, 0.20, 0.28],
        ],
        dtype=np.float32,
    )
    descend_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [0.20, 0.20, 0.36],
        ],
        dtype=np.float32,
    )
    finalized_dense_envs = []

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=2,
            agent=types.SimpleNamespace(uid="fake_uid"),
            evaluate=lambda: {"success": np.asarray([True, False], dtype=bool)},
        )
    )

    class _UnifiedArtifacts:
        finalize_batched_dense_episode_env_if_available = staticmethod(
            lambda **kwargs: finalized_dense_envs.append(
                (int(kwargs["env_index"]), bool(kwargs["semantic_task_success"]), kwargs["failed_stage"])
            )
        )

    class _UnifiedControl:
        resolve_macro_backend = staticmethod(lambda _config_overrides, default_backend=None: "proxy_ee_delta")
        is_proxy_ee_delta_backend = staticmethod(lambda backend: backend == "proxy_ee_delta")
        get_default_debug_close_steps = staticmethod(lambda _uid: 1)

    class _UnifiedDebug:
        get_debug_target_object = staticmethod(lambda _env_unwrapped: object())
        get_debug_actor_position_rows = staticmethod(lambda _target_object: actor_rows.copy())
        get_debug_planner_ee_pose_rows = staticmethod(lambda _env_unwrapped: "tcp_pose")
        collect_debug_robot_object_contacts = staticmethod(lambda *_args, **_kwargs: [])
        collect_debug_object_object_contacts = staticmethod(lambda *_args, **_kwargs: [])

        @staticmethod
        def pose_to_numpy_rows(pose):
            quat_rows = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1))
            if pose == "tcp_pose":
                return tcp_rows.copy(), quat_rows
            if pose == "descend_pose":
                return descend_rows.copy(), quat_rows
            if pose == "lift_pose":
                return descend_rows.copy(), quat_rows
            raise AssertionError(f"Unexpected pose marker: {pose!r}")

        build_debug_lift_pose_from_policy = staticmethod(
            lambda *_args, **_kwargs: ("lift_pose", None, None, None)
        )
        log_debug_non_target_object_contacts = staticmethod(lambda *_args, **_kwargs: None)
        pose_to_numpy = staticmethod(lambda _pose: (np.zeros(3, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))

    class _UnifiedLowlevel:
        @staticmethod
        def get_debug_planner_config(_env_unwrapped):
            return {
                "planner_proxy_safe_clearance_z": 0.05,
                "planner_proxy_predescent_settle_steps": 0,
                "planner_proxy_preclose_settle_steps": 0,
                "planner_proxy_max_stage_steps": 100,
                "planner_proxy_stall_steps": 1000,
                "planner_proxy_pos_tol_m": 0.01,
                "planner_proxy_xy_step_m": 0.01,
                "planner_proxy_z_step_m": 0.01,
                "planner_proxy_batch_runaway_abs_limit_m": 5.0,
                "planner_waypoints": {"enabled": False, "points": []},
            }

        get_required_planner_waypoints_config = staticmethod(lambda planner_cfg: (False, []))
        get_required_proxy_adaptive_step_config = staticmethod(lambda planner_cfg: (True, 0.01, 0.01, 0.01))

        _get_proxy_ee_gripper_controller_signal = staticmethod(
            lambda _env_unwrapped, target_state: 1.0 if target_state == "open" else -1.0
        )
        build_proxy_delta_pos = staticmethod(lambda *_args, **_kwargs: np.asarray([0.01, 0.0, 0.0], dtype=np.float32))
        apply_proxy_ee_delta_action = staticmethod(lambda *_args, **_kwargs: True)

    def _build_object_descend_target(_env, *, extra_clearance=0.03):
        del extra_clearance
        return (
            "orange_cube_ext",
            actor_rows.copy(),
            np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
            "descend_pose",
        )

    success = uut._execute_independent_batched_proxy_pick_macro(
        fake_env,
        args=types.SimpleNamespace(
            auto_pick_macro="1",
            num_envs=2,
            batched_save_video_output_dir=None,
            batched_save_video_gif_output_dir=None,
            dense_episode_output_dir="/tmp/dense",
        ),
        config_overrides={"planner_lift_delta_z": 0.05},
        unified_artifacts=_UnifiedArtifacts,
        unified_control=_UnifiedControl,
        unified_debug=_UnifiedDebug,
        unified_lowlevel=_UnifiedLowlevel,
        build_object_descend_target=_build_object_descend_target,
        save_video_buffer_for_env_to_path=lambda *_args, **_kwargs: None,
        video_frames_per_env=[[], []],
        finalized_video_env_indices=set(),
    )

    assert success is False
    macro_feedback = uut.peek_unified_macro_feedback()
    assert macro_feedback["failed_env_indices"] == [1]
    assert finalized_dense_envs[0] == (1, False, "rise")


def test_independent_batched_proxy_uses_hold_semantics_before_close():
    tcp_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [-0.30, -0.70, 0.36],
        ],
        dtype=np.float32,
    )
    actor_rows = np.asarray(
        [
            [-0.30, -0.70, 0.28],
            [-0.30, -0.70, 0.28],
        ],
        dtype=np.float32,
    )
    descend_rows = np.asarray(
        [
            [-0.30, -0.70, 0.36],
            [-0.30, -0.70, 0.36],
        ],
        dtype=np.float32,
    )
    observed = {"kwargs": None}

    class _StopAfterFirstAction(RuntimeError):
        pass

    fake_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=2,
            agent=types.SimpleNamespace(uid="fake_uid"),
            evaluate=lambda: {"success": np.asarray([False, False], dtype=bool)},
        )
    )

    class _UnifiedArtifacts:
        finalize_batched_dense_episode_env_if_available = staticmethod(lambda **_kwargs: None)

    class _UnifiedControl:
        resolve_macro_backend = staticmethod(lambda _config_overrides, default_backend=None: "proxy_ee_delta")
        is_proxy_ee_delta_backend = staticmethod(lambda backend: backend == "proxy_ee_delta")
        get_default_debug_close_steps = staticmethod(lambda _uid: 1)

    class _UnifiedDebug:
        get_debug_target_object = staticmethod(lambda _env_unwrapped: object())
        get_debug_actor_position_rows = staticmethod(lambda _target_object: actor_rows.copy())
        get_debug_planner_ee_pose_rows = staticmethod(lambda _env_unwrapped: "tcp_pose")
        collect_debug_robot_object_contacts = staticmethod(lambda *_args, **_kwargs: [])
        collect_debug_object_object_contacts = staticmethod(lambda *_args, **_kwargs: [])

        @staticmethod
        def pose_to_numpy_rows(pose):
            quat_rows = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1))
            if pose == "tcp_pose":
                return tcp_rows.copy(), quat_rows
            if pose == "descend_pose":
                return descend_rows.copy(), quat_rows
            if pose == "lift_pose":
                return descend_rows.copy(), quat_rows
            raise AssertionError(f"Unexpected pose marker: {pose!r}")

        build_debug_lift_pose_from_policy = staticmethod(
            lambda *_args, **_kwargs: ("lift_pose", None, None, None)
        )
        log_debug_non_target_object_contacts = staticmethod(lambda *_args, **_kwargs: None)
        pose_to_numpy = staticmethod(
            lambda _pose: (
                np.zeros(3, dtype=np.float32),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )

    class _UnifiedLowlevel:
        @staticmethod
        def get_debug_planner_config(_env_unwrapped):
            return {
                "planner_proxy_safe_clearance_z": 0.05,
                "planner_proxy_predescent_settle_steps": 0,
                "planner_proxy_preclose_settle_steps": 0,
                "planner_proxy_max_stage_steps": 100,
                "planner_proxy_stall_steps": 100,
                "planner_proxy_pos_tol_m": 0.01,
                "planner_proxy_xy_step_m": 0.01,
                "planner_proxy_z_step_m": 0.01,
                "planner_proxy_batch_runaway_abs_limit_m": 5.0,
                "planner_waypoints": {"enabled": False, "points": []},
            }

        get_required_planner_waypoints_config = staticmethod(lambda planner_cfg: (False, []))
        get_required_proxy_adaptive_step_config = staticmethod(lambda planner_cfg: (True, 0.01, 0.01, 0.01))

        _get_proxy_ee_gripper_controller_signal = staticmethod(
            lambda _env_unwrapped, target_state: 1.0 if target_state == "open" else -1.0
        )
        build_proxy_delta_pos = staticmethod(
            lambda *_args, **_kwargs: np.asarray([0.0, 0.0, 0.01], dtype=np.float32)
        )

        @staticmethod
        def apply_proxy_ee_delta_action(*_args, **kwargs):
            observed["kwargs"] = kwargs
            raise _StopAfterFirstAction

    def _build_object_descend_target(_env, *, extra_clearance=0.03):
        del extra_clearance
        return (
            "orange_cube_ext",
            actor_rows.copy(),
            np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
            "descend_pose",
        )

    with pytest.raises(_StopAfterFirstAction):
        uut._execute_independent_batched_proxy_pick_macro(
            fake_env,
            args=types.SimpleNamespace(
                auto_pick_macro="1",
                num_envs=2,
                batched_save_video_output_dir=None,
                batched_save_video_gif_output_dir=None,
                dense_episode_output_dir=None,
            ),
            config_overrides={"planner_lift_delta_z": 0.05},
            unified_artifacts=_UnifiedArtifacts,
            unified_control=_UnifiedControl,
            unified_debug=_UnifiedDebug,
            unified_lowlevel=_UnifiedLowlevel,
            build_object_descend_target=_build_object_descend_target,
            save_video_buffer_for_env_to_path=lambda *_args, **_kwargs: None,
            video_frames_per_env=[[], []],
            finalized_video_env_indices=set(),
        )

    assert observed["kwargs"] is not None
    assert observed["kwargs"]["gripper_target_state"] == "hold"
    override = np.asarray(observed["kwargs"]["gripper_signal_override"], dtype=np.float32)
    assert override.shape == (2,)
    assert np.isnan(override).all()


def test_batched_debug_side_artifacts_skip_already_finalized_envs(monkeypatch, tmp_path: Path):
    saved_video_envs = []
    saved_gif_envs = []

    class _UnifiedArtifacts:
        @staticmethod
        def resolve_batched_debug_video_output_path(output_dir, *, env_index):
            return Path(output_dir) / f"env_{int(env_index):06d}.mkv"

        @staticmethod
        def resolve_batched_debug_video_gif_output_path(output_dir, *, env_index):
            return Path(output_dir) / f"env_{int(env_index):06d}.gif"

        @staticmethod
        def write_debug_video_gif_from_video(video_path, gif_path):
            saved_gif_envs.append((Path(video_path).name, Path(gif_path).name))
            return gif_path

    monkeypatch.setattr(uut, "_load_unified_proxy_artifacts", lambda: _UnifiedArtifacts)

    def _save_video_buffer_for_env_to_path(env_index, video_path):
        saved_video_envs.append((int(env_index), Path(video_path).name))
        return Path(video_path)

    args = types.SimpleNamespace(
        batched_save_video_output_dir=str(tmp_path / "videos"),
        batched_save_video_gif_output_dir=str(tmp_path / "gifs"),
    )

    uut._maybe_save_batched_debug_side_artifacts(
        args,
        video_frames_per_env=[[object()], [object()]],
        save_video_buffer_for_env_to_path=_save_video_buffer_for_env_to_path,
        finalized_video_env_indices={0},
    )

    assert saved_video_envs == [(1, "env_000001.mkv")]
    assert saved_gif_envs == [("env_000001.mkv", "env_000001.gif")]
