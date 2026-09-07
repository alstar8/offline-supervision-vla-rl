from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import openreal2sim.simulation.maniskill.scripts.run_rc5_unified as uut
from openreal2sim.simulation.maniskill.scripts.rc5_unified_execution import (
    CanonicalBatchTraceRecord,
    CanonicalEpisodeTraceRecord,
    CanonicalEpisodeTraceSeed,
    CanonicalPerEnvTraceRecord,
    CanonicalStageTraceRecord,
    CanonicalTraceEventRecord,
    UnifiedBackendResult,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    load_rl4vla_raw_episode_artifact,
    write_rl4vla_raw_episode_artifact,
)

_ORIGINAL_RESOLVE_AUTO_RUN_DIR = uut._resolve_auto_run_dir


@pytest.fixture(autouse=True)
def _isolate_auto_run_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        uut,
        "_resolve_auto_run_dir",
        lambda _argv, _args: (tmp_path / "auto_run").resolve(),
    )


def _write_config(
    tmp_path: Path,
    *,
    teleop: bool = True,
    unified_planner_backend: str | None = None,
) -> Path:
    simulation = {
        "robot_uids": "rc5_aero_hand_openr2s_rl",
        "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        "hand_pose_config": "config/hand_pose_presets.yaml",
        "hand_contact_config": "config/hand_contact_profiles.yaml",
        "hand_contact_profile": "rubber_fingertips_v1",
        "hand_controller_config": "config/hand_controller_profiles.yaml",
        "hand_controller_profile": "stronger_grasp_v1",
        "manip_object_id": "orange_cube_ext",
        "object_placements": {
            "orange_cube_ext": {
                "task_semantic_name": "orange cube",
                "position": [-0.3, -0.7, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
            "white_cube_ext": {
                "task_semantic_name": "white cube",
                "position": [0.0, -0.7, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
        },
    }
    if teleop:
        simulation["teleop_profile_config"] = "config/teleop_profiles.yaml"
        simulation["teleop_profile"] = "rc5_teleop_v1"
    if unified_planner_backend is not None:
        simulation["unified_planner_backend"] = str(unified_planner_backend)

    data = {
        "keys": ["demo_key"],
        "global": {"simulation": {}},
        "local": {
            "demo_key": {
                "simulation": simulation,
            }
        },
    }
    path = tmp_path / "rc5_unified_test_config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _install_auto_run_dir(monkeypatch, tmp_path: Path) -> Path:
    run_dir = (tmp_path / "auto_run").resolve()
    monkeypatch.setattr(uut, "_resolve_auto_run_dir", lambda _argv, _args: run_dir)
    return run_dir


def test_resolve_auto_run_dir_includes_viewer_mode():
    args = uut._build_parser().parse_args(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--task_type",
            "pick_up",
            "--task_object_id",
            "orange_cube_ext",
        ]
    )

    run_dir = _ORIGINAL_RESOLVE_AUTO_RUN_DIR(["--key", "demo_key"], args)

    assert run_dir.name.startswith("proxy_ee_delta_viewer_demo_key_pick_up_orange_cube_ext_")


def test_resolve_auto_run_dir_includes_headless_mode():
    args = uut._build_parser().parse_args(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--task_type",
            "pick_up",
            "--task_object_id",
            "orange_cube_ext",
        ]
    )

    run_dir = _ORIGINAL_RESOLVE_AUTO_RUN_DIR(["--key", "demo_key", "--headless"], args)

    assert run_dir.name.startswith("proxy_ee_delta_headless_demo_key_pick_up_orange_cube_ext_")


def test_rc5_unified_parser_allows_motion_backend_to_be_resolved_from_config():
    parser = uut._build_parser()
    args = parser.parse_args([])

    assert args.motion_backend is None
    assert args.run_mode == "episode"


def test_rc5_unified_prerelease_help_hides_legacy_collection_surface():
    help_text = uut._build_parser().format_help()

    assert "--motion_backend" in help_text
    assert "--embed_runtime_bundle_in_rl4vla_raw_npz" in help_text
    assert "--run_mode" not in help_text
    assert "{episode,collection}" not in help_text
    assert "{planner,proxy_ee_delta,hybrid}" not in help_text
    assert "{pick_up,pick_and_place}" not in help_text
    assert "--task_destination_id" not in help_text
    assert "--collection_manifest" not in help_text
    assert "--placement_manifest" not in help_text
    assert "--num_episodes" not in help_text
    assert "--output_dir" not in help_text
    assert "--dense_episode_image_width" not in help_text
    assert "--dense_episode_image_height" not in help_text
    assert "--save_debug_video" not in help_text
    assert "--save_debug_gif" not in help_text
    assert "--runtime_sim_patch" not in help_text
    assert "--exporter" not in help_text
    assert "--stop_on_failure" not in help_text


def test_rc5_unified_rejects_invalid_public_surface_values():
    parser = uut._build_parser()
    args = parser.parse_args(
        [
            "--run_mode",
            "invalid_mode",
            "--motion_backend",
            "invalid_backend",
            "--task_type",
            "invalid_task",
        ]
    )

    with pytest.raises(ValueError, match="Unsupported run_mode"):
        uut._validate_parser_surface_args(args)


def test_rc5_unified_main_dispatches_proxy_backend_to_unified_proxy_runtime(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=True)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)
    observed = {}

    def fake_execute(request):
        observed["motion_backend"] = request.motion_backend
        observed["task_type"] = request.task_plan.intent.task_type
        observed["task_object_id"] = request.task_plan.intent.object_id
        observed["trace_task_type"] = request.trace_seed.task_type
        observed["trace_object_id"] = request.trace_seed.object_id
        observed["trace_stage_kinds"] = tuple(request.trace_seed.stage_kinds)
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        observed["module_name"] = plan.module_name
        observed["argv"] = list(plan.forwarded_argv)
        observed["asset_dir"] = plan.env_updates.get("RC5_AERO_HAND_ASSET_DIR")
        observed["move_group"] = plan.env_updates.get("OPENR2S_RC5_MOVE_GROUP")
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
        )

    monkeypatch.setattr(uut, "execute_backend_request", fake_execute)

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--task_type",
            "pick_up",
            "--task_object_id",
            "orange_cube_ext",
            "--rc5_asset_dir",
            "/tmp/rc5-assets",
            "--rc5_move_group",
            "right_tcp_link",
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
        ]
    )

    assert rc == 0
    assert observed["motion_backend"] == uut.PROXY_BACKEND
    assert observed["task_type"] == "pick_up"
    assert observed["task_object_id"] == "orange_cube_ext"
    assert observed["trace_task_type"] == "pick_up"
    assert observed["trace_object_id"] == "orange_cube_ext"
    assert observed["trace_stage_kinds"] == (
        "move_to_pregrasp",
        "move_to_descend",
        "close_gripper",
        "lift_object",
        "retention_check",
    )
    assert observed["module_name"].endswith("rc5_unified_proxy_runtime")
    assert observed["argv"] == [
        "--task_object_id",
        "orange_cube_ext",
        "--config_path",
        str(cfg_path),
        "--key",
        "demo_key",
        "--scene",
        "scene.json",
        "--rl4vla_raw_episode_output",
        str(run_dir / "rl4vla_raw_episode.npz"),
        "--save_video_gif_on_exit",
        "--save_video_gif_path",
        str(run_dir / "debug_video.gif"),
        "--runtime_request_path",
        str(run_dir / "runtime_request.json"),
        "--robot_uids",
        "rc5_aero_hand_openr2s_rl",
        "--sim_backend",
        "physx_cuda",
        "--control_mode",
        "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        "--teleop_profile_config",
        "config/teleop_profiles.yaml",
        "--teleop_profile",
        "rc5_teleop_v1",
        "--auto_pick_macro",
        "1",
    ]
    assert observed["asset_dir"] == "/tmp/rc5-assets"
    assert observed["move_group"] == "right_tcp_link"
    assert (run_dir / "run.log").exists()
    assert (run_dir / "execution_summary.json").exists()


def test_rc5_unified_main_supports_documented_prerelease_proxy_contract(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=True)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)
    observed = {}
    dense_path = (tmp_path / "manual_run" / "dense_episode.npz").resolve()
    raw_path = (tmp_path / "manual_run" / "rl4vla_raw_episode.npz").resolve()
    dense_path.parent.mkdir(parents=True, exist_ok=True)

    def fake_execute(request):
        observed["motion_backend"] = request.motion_backend
        observed["task_type"] = request.task_plan.intent.task_type
        observed["task_object_id"] = request.task_plan.intent.object_id
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        observed["argv"] = list(plan.forwarded_argv)
        request_path = Path(
            observed["argv"][observed["argv"].index("--runtime_request_path") + 1]
        ).expanduser().resolve()
        dense_path.write_bytes(b"dense")
        write_rl4vla_raw_episode_artifact(
            artifact_path=raw_path,
            instruction="Pick up orange cube.",
            images=[
                [[[0, 0, 0]]],
                [[[1, 1, 1]]],
            ],
            actions=[[0, 0, 0, 0, 0, 0, 0]],
            infos=[{"success": True}],
            result={"success": True},
            source={"planner_backend": "proxy_ee_delta"},
            embedded_runtime_config_yaml=cfg_path.read_text(encoding="utf-8"),
            embedded_runtime_request_json=request_path.read_text(encoding="utf-8"),
        )
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
            trace_record={"execution_outcome": "success", "events": []},
        )

    monkeypatch.setattr(uut, "execute_backend_request", fake_execute)

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
            "--task_type",
            "pick_up",
            "--dense_episode_output",
            str(dense_path),
            "--rl4vla_raw_episode_output",
            str(raw_path),
            "--embed_runtime_bundle_in_rl4vla_raw_npz",
        ]
    )

    assert rc == 0
    assert observed["motion_backend"] == uut.PROXY_BACKEND
    assert observed["task_type"] == "pick_up"
    assert observed["task_object_id"] == "orange_cube_ext"
    assert observed["argv"][:10] == [
        "--config_path",
        str(cfg_path),
        "--key",
        "demo_key",
        "--scene",
        "scene.json",
        "--dense_episode_output",
        str(dense_path),
        "--rl4vla_raw_episode_output",
        str(raw_path),
    ]
    assert "--embed_runtime_bundle_in_rl4vla_raw_npz" in observed["argv"]
    assert "--runtime_request_path" in observed["argv"]
    request_path = Path(
        observed["argv"][observed["argv"].index("--runtime_request_path") + 1]
    ).expanduser().resolve()
    assert request_path == (run_dir / "runtime_request.json").resolve()
    runtime_request = json.loads(request_path.read_text(encoding="utf-8"))
    runtime_config_path = (run_dir / "runtime_config.yaml").resolve()
    assert runtime_request["key"] == "demo_key"
    assert runtime_request["scene_path"] == "scene.json"
    assert runtime_request["task_object_id"] == "orange_cube_ext"
    assert runtime_request["runtime_config_path"] == str(runtime_config_path)
    assert runtime_config_path.read_text(encoding="utf-8") == cfg_path.read_text(encoding="utf-8")
    assert "--robot_uids" in observed["argv"]
    assert "rc5_aero_hand_openr2s_rl" in observed["argv"]
    assert "--sim_backend" in observed["argv"]
    assert "physx_cuda" in observed["argv"]
    assert "--control_mode" in observed["argv"]
    assert "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos" in observed["argv"]
    assert "--teleop_profile_config" in observed["argv"]
    assert "config/teleop_profiles.yaml" in observed["argv"]
    assert "--teleop_profile" in observed["argv"]
    assert "rc5_teleop_v1" in observed["argv"]
    assert observed["argv"][-2:] == ["--auto_pick_macro", "1"]
    summary_payload = json.loads((run_dir / "execution_summary.json").read_text(encoding="utf-8"))
    dense_final = dense_path.with_name("dense_episode_success.npz")
    raw_final = raw_path.with_name("rl4vla_raw_episode_success.npz")
    assert summary_payload["artifacts"]["dense_episode_path"] == str(dense_final)
    assert summary_payload["artifacts"]["rl4vla_raw_episode_path"] == str(raw_final)
    assert dense_final.exists()
    assert raw_final.exists()
    runtime_request = json.loads(request_path.read_text(encoding="utf-8"))
    assert runtime_request["rl4vla_raw_episode_artifact_path"] == str(raw_final)
    embedded_payload = load_rl4vla_raw_episode_artifact(raw_final)
    assert embedded_payload["embedded_runtime_request_json"] == json.dumps(
        runtime_request,
        indent=2,
        sort_keys=True,
    )
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert str(dense_final) in run_log
    assert str(raw_final) in run_log
    assert (run_dir / "run.log").exists()
    assert (run_dir / "execution_summary.json").exists()


def test_rc5_unified_main_default_proxy_run_materializes_embedded_runtime_request(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=True)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)
    observed = {}

    def fake_execute(request):
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        observed["argv"] = list(plan.forwarded_argv)
        request_path = Path(
            observed["argv"][observed["argv"].index("--runtime_request_path") + 1]
        ).expanduser().resolve()
        raw_path = (run_dir / "rl4vla_raw_episode.npz").resolve()
        write_rl4vla_raw_episode_artifact(
            artifact_path=raw_path,
            instruction="Pick up orange cube.",
            images=[
                [[[0, 0, 0]]],
                [[[1, 1, 1]]],
            ],
            actions=[[0, 0, 0, 0, 0, 0, 0]],
            infos=[{"success": True}],
            result={"success": True},
            source={"planner_backend": "proxy_ee_delta"},
            embedded_runtime_config_yaml=cfg_path.read_text(encoding="utf-8"),
            embedded_runtime_request_json=request_path.read_text(encoding="utf-8"),
        )
        (run_dir / "debug_video.gif").write_bytes(b"GIF89a")
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
            trace_record={"execution_outcome": "success", "events": []},
        )

    monkeypatch.setattr(uut, "execute_backend_request", fake_execute)

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
            "--task_type",
            "pick_up",
            "--embed_runtime_bundle_in_rl4vla_raw_npz",
        ]
    )

    assert rc == 0
    assert "--runtime_request_path" in observed["argv"]
    request_path = Path(
        observed["argv"][observed["argv"].index("--runtime_request_path") + 1]
    ).expanduser().resolve()
    assert request_path == (run_dir / "runtime_request.json").resolve()
    runtime_request = json.loads(request_path.read_text(encoding="utf-8"))
    runtime_config_path = (run_dir / "runtime_config.yaml").resolve()
    raw_final = (run_dir / "rl4vla_raw_episode_success.npz").resolve()
    assert runtime_request["key"] == "demo_key"
    assert runtime_request["scene_path"] == "scene.json"
    assert runtime_request["task_type"] == "pick_up"
    assert runtime_request["task_object_id"] == "orange_cube_ext"
    assert runtime_request["runtime_config_path"] == str(runtime_config_path)
    assert runtime_config_path.read_text(encoding="utf-8") == cfg_path.read_text(encoding="utf-8")
    assert runtime_request["rl4vla_raw_episode_artifact_path"] == str(raw_final)
    assert runtime_request["debug_video_gif_path"] == str((run_dir / "debug_video_success.gif").resolve())
    embedded_payload = load_rl4vla_raw_episode_artifact(raw_final)
    assert embedded_payload["embedded_runtime_request_json"] == json.dumps(
        runtime_request,
        indent=2,
        sort_keys=True,
    )


def test_append_default_artifact_args_skips_dense_episode_for_batched_runtime(tmp_path):
    run_dir = (tmp_path / "auto_run").resolve()
    argv = [
        "--motion_backend",
        uut.PROXY_BACKEND,
        "--scene",
        "scene.json",
        "--num_envs",
        "3",
    ]

    augmented = uut._append_default_artifact_args(
        argv,
        run_dir,
        num_envs=3,
    )

    assert augmented == argv


def test_collection_request_rejects_real_planner_backend_resolved_from_config(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True, unified_planner_backend="rrtconnect")

    with pytest.raises(ValueError, match="supports only motion_backend='proxy_ee_delta'"):
        uut.resolve_collection_request(
            [
                "--run_mode",
                "collection",
                "--config_path",
                str(cfg_path),
                "--key",
                "demo_key",
                "--scene",
                "assets/scenes/demo_key/simulation/scene.json",
                "--output_dir",
                str(tmp_path / "collection_out"),
                "--placement_seed_start",
                "0",
                "--num_episodes",
                "1",
            ]
        )


def test_collection_request_accepts_proxy_backend_resolved_from_config(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True, unified_planner_backend="proxy")

    request = uut.resolve_collection_request(
        [
            "--run_mode",
            "collection",
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "assets/scenes/demo_key/simulation/scene.json",
            "--output_dir",
            str(tmp_path / "collection_out"),
            "--placement_seed_start",
            "0",
            "--num_episodes",
            "1",
        ]
    )

    assert request.motion_backend == uut.PROXY_BACKEND
    assert request.task_object_id == "orange_cube_ext"
    assert request.embed_runtime_bundle_in_rl4vla_raw_npz is True
    assert len(request.episodes) == 1


def test_collection_request_can_disable_embedded_runtime_bundle(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True, unified_planner_backend="proxy")

    request = uut.resolve_collection_request(
        [
            "--run_mode",
            "collection",
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "assets/scenes/demo_key/simulation/scene.json",
            "--output_dir",
            str(tmp_path / "collection_out"),
            "--placement_seed_start",
            "0",
            "--num_episodes",
            "1",
            "--no-embed_runtime_bundle_in_rl4vla_raw_npz",
        ]
    )

    assert request.embed_runtime_bundle_in_rl4vla_raw_npz is False


def test_collection_runtime_inputs_propagate_disabled_embedded_runtime_bundle_flag(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True, unified_planner_backend="proxy")
    request = uut.resolve_collection_request(
        [
            "--run_mode",
            "collection",
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "assets/scenes/demo_key/simulation/scene.json",
            "--output_dir",
            str(tmp_path / "collection_out"),
            "--placement_seed_start",
            "0",
            "--num_episodes",
            "1",
            "--no-embed_runtime_bundle_in_rl4vla_raw_npz",
        ]
    )

    runtime_inputs = uut.materialize_episode_runtime_inputs(request)
    runtime_request = json.loads(Path(runtime_inputs[0].request_path).read_text(encoding="utf-8"))

    assert "--no-embed_runtime_bundle_in_rl4vla_raw_npz" in runtime_request["runner_argv"]
    assert "--embed_runtime_bundle_in_rl4vla_raw_npz" not in runtime_request["runner_argv"]


def test_rc5_unified_main_writes_batch_summary_for_batched_proxy_request(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=True)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)

    def fake_execute(request):
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        trace_seed = CanonicalEpisodeTraceSeed(
            motion_backend=request.motion_backend,
            config_key="demo_key",
            robot_uids="rc5_aero_hand_openr2s_rl",
            task_type="pick_up",
            object_id="orange_cube_ext",
            destination_id=None,
            prompt=None,
            stage_names=("pregrasp", "descend", "close", "lift", "retention_check"),
            stage_kinds=(
                "move_to_pregrasp",
                "move_to_descend",
                "close_gripper",
                "lift_object",
                "retention_check",
            ),
        )
        trace_record = CanonicalEpisodeTraceRecord(
            seed=trace_seed,
            stage_records=(
                CanonicalStageTraceRecord(order_index=0, name="pregrasp", kind="move_to_pregrasp"),
            ),
            events=(
                CanonicalTraceEventRecord(
                    event_type="batch_runtime_feedback",
                    payload={
                        "batch_size": 3,
                        "successful_env_count": 3,
                        "failed_env_indices": [],
                        "artifacts_recorded_per_env": False,
                        "per_env_feedback": [
                            {"env_index": 0, "semantic_task_success": True},
                            {"env_index": 1, "semantic_task_success": True},
                            {"env_index": 2, "semantic_task_success": True},
                        ],
                    },
                ),
                CanonicalTraceEventRecord(
                    event_type="macro_finished",
                    payload={
                        "execution_outcome": "success",
                        "semantic_task_success": True,
                        "batch_size": 3,
                        "successful_env_count": 3,
                        "failed_env_indices": [],
                    },
                ),
            ),
            execution_outcome="success",
            exit_code=0,
            dispatch_module=plan.module_name,
            dispatch_callable=plan.callable_name,
            batch_trace=CanonicalBatchTraceRecord(
                requested_num_envs=3,
                runtime_batch_size=3,
                successful_env_count=3,
                failed_env_indices=(),
                artifacts_recorded_per_env=False,
                runtime_batch_feedback_present=True,
                per_env_records=(
                    CanonicalPerEnvTraceRecord(env_index=0, semantic_task_success=True, execution_outcome="success"),
                    CanonicalPerEnvTraceRecord(env_index=1, semantic_task_success=True, execution_outcome="success"),
                    CanonicalPerEnvTraceRecord(env_index=2, semantic_task_success=True, execution_outcome="success"),
                ),
            ),
        )
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
            trace_seed=trace_seed,
            trace_record=trace_record,
        )

    monkeypatch.setattr(uut, "execute_backend_request", fake_execute)

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--task_type",
            "pick_up",
            "--task_object_id",
            "orange_cube_ext",
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
            "--headless",
            "--num_envs",
            "3",
        ]
    )

    assert rc == 0
    summary_payload = json.loads((run_dir / "execution_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["batch"] == {
        "requested_num_envs": 3,
        "batched_request": True,
        "runtime_batch_feedback_present": True,
        "runtime_batch_size": 3,
        "successful_env_count": 3,
        "failed_env_indices": [],
        "artifacts_recorded_per_env": False,
        "per_env_feedback_count": 3,
        "auto_dense_episode_enabled": False,
    }
    assert summary_payload["artifacts"]["dense_episode_path"] is None
    assert summary_payload["trace_record"]["batch_trace"]["requested_num_envs"] == 3
    assert summary_payload["trace_record"]["batch_trace"]["runtime_batch_size"] == 3
    assert len(summary_payload["trace_record"]["batch_trace"]["per_env_records"]) == 3


def test_rc5_unified_main_dispatches_planner_backend_to_legacy_runtime(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=False)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)
    observed = {}

    def fake_execute(request):
        observed["motion_backend"] = request.motion_backend
        observed["task_type"] = request.task_plan.intent.task_type
        observed["task_object_id"] = request.task_plan.intent.object_id
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        observed["module_name"] = plan.module_name
        observed["argv"] = list(plan.forwarded_argv)
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
        )

    monkeypatch.setattr(
        uut,
        "execute_backend_request",
        fake_execute,
    )

    rc = uut.main(
        [
            "--motion_backend",
            uut.PLANNER_BACKEND,
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
            "--vis",
        ]
    )

    assert rc == 0
    assert observed["motion_backend"] == uut.PLANNER_BACKEND
    assert observed["task_type"] == "pick_up"
    assert observed["task_object_id"] == "orange_cube_ext"
    assert observed["module_name"].endswith("run_rc5_solver_debug_planner_LEGACY")
    assert "--planner_backend" in observed["argv"]
    assert (run_dir / "execution_summary.json").exists()


def test_resolve_task_semantic_name_from_object_cfg_warns_on_name_fallback(capsys):
    resolved = uut._resolve_task_semantic_name_from_object_cfg(
        {"name": "orange_cube"},
        object_id="orange_cube_ext",
    )

    assert resolved == "orange cube"
    captured = capsys.readouterr()
    assert "[WARNING] [RC5Unified]" in captured.out
    assert "Missing task_semantic_name" in captured.out


def test_finalize_episode_runtime_request_artifacts_updates_embedded_rl4vla_request_copy(tmp_path):
    episode_dir = tmp_path / "episodes" / "episode_000000"
    episode_dir.mkdir(parents=True)
    runtime_config_path = episode_dir / "runtime_config.yaml"
    runtime_config_path.write_text("demo: true\n", encoding="utf-8")
    raw_path = episode_dir / "rl4vla_raw_episode.npz"
    write_rl4vla_raw_episode_artifact(
        artifact_path=raw_path,
        instruction="Pick up orange cube.",
        images=[
            [[ [0, 0, 0] ]],
            [[ [1, 1, 1] ]],
        ],
        actions=[[0, 0, 0, 0, 0, 0, 0]],
        infos=[{"success": True}],
        result={"success": True},
        source={"planner_backend": "proxy_ee_delta"},
        embedded_runtime_config_yaml="demo: true\n",
        embedded_runtime_request_json='{"episode_id":"episode_000000","rl4vla_raw_episode_artifact_path":"placeholder.npz"}',
    )
    request_path = episode_dir / "runtime_request.json"
    request_payload = {
        "episode_id": "episode_000000",
        "episode_index": 0,
        "runtime_config_path": str(runtime_config_path),
        "rl4vla_raw_episode_artifact_path": str(raw_path),
        "debug_video_path": None,
        "debug_video_gif_path": None,
        "runner_argv": [
            "--config_path",
            str(runtime_config_path),
            "--rl4vla_raw_episode_output",
            str(raw_path),
        ],
    }
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")

    finalized_item, runtime_request = uut._finalize_episode_runtime_request_artifacts(
        uut.EpisodeExecutionResult(
            episode_id="episode_000000",
            episode_index=0,
            request_path=request_path,
            exit_code=0,
            runtime_exit_code=0,
            execution_outcome="success",
            success=True,
            semantic_task_success=True,
            rl4vla_raw_episode_artifact_path=raw_path,
            batch_id="batch_000000",
            compatibility_group_id="group",
            env_index=0,
        )
    )

    assert finalized_item.rl4vla_raw_episode_artifact_path is not None
    assert runtime_request["rl4vla_raw_episode_artifact_path"] == str(finalized_item.rl4vla_raw_episode_artifact_path)
    assert runtime_request["runner_argv"] == [
        "--config_path",
        str(runtime_config_path),
        "--rl4vla_raw_episode_output",
        str(finalized_item.rl4vla_raw_episode_artifact_path),
    ]
    embedded_payload = load_rl4vla_raw_episode_artifact(finalized_item.rl4vla_raw_episode_artifact_path)
    assert embedded_payload["embedded_runtime_request_json"] == json.dumps(runtime_request, indent=2, sort_keys=True)


def test_resolve_task_semantic_name_from_object_cfg_warns_on_object_id_fallback(capsys):
    resolved = uut._resolve_task_semantic_name_from_object_cfg(
        {},
        object_id="orange_cube_ext",
    )

    assert resolved == "orange_cube_ext"
    captured = capsys.readouterr()
    assert "[WARNING] [RC5Unified]" in captured.out
    assert "falling back to technical object id" in captured.out


def test_rc5_unified_main_rejects_manual_planner_backend_override(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True)

    try:
        uut.main(
            [
                "--motion_backend",
                uut.PROXY_BACKEND,
                "--config_path",
                str(cfg_path),
                "--key",
                "demo_key",
                "--planner_backend",
                "planner",
            ]
        )
    except ValueError as exc:
        assert "--planner_backend should not be passed" in str(exc)
    else:
        raise AssertionError("Expected manual --planner_backend override to be rejected")


def test_rc5_unified_main_proxy_backend_fails_fast_without_teleop_profile(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=False)

    try:
        uut.main(
            [
                "--motion_backend",
                uut.PROXY_BACKEND,
                "--config_path",
                str(cfg_path),
                "--key",
                "demo_key",
                "--scene",
                "scene.json",
            ]
        )
    except ValueError as exc:
        assert "requires teleop_profile_config and teleop_profile" in str(exc)
    else:
        raise AssertionError("Expected proxy backend to fail fast without teleop profile")


def test_rc5_unified_main_logs_warning_for_env_fallbacks(tmp_path, monkeypatch, capsys):
    cfg_path = _write_config(tmp_path, teleop=True)
    _install_auto_run_dir(monkeypatch, tmp_path)

    monkeypatch.setattr(
        uut,
        "execute_backend_request",
        lambda request: UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=uut.resolve_backend(request.motion_backend).build_dispatch_plan(request),
        ),
    )
    monkeypatch.setenv("RC5_AERO_HAND_ASSET_DIR", "/env/assets")
    monkeypatch.setenv("OPENR2S_RC5_MOVE_GROUP", "right_tcp_link")

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "[WARNING] [RC5Unified] Using RC5_AERO_HAND_ASSET_DIR from environment" in captured.out
    assert "[WARNING] [RC5Unified] Using OPENR2S_RC5_MOVE_GROUP from environment" in captured.out
    assert "[WARNING] [RC5Unified] Injecting --robot_uids=rc5_aero_hand_openr2s_rl from validated config" in captured.out


def test_rc5_unified_main_rejects_invalid_rc5_move_group(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True)

    with pytest.raises(ValueError, match="Invalid RC5 move group 'bad_group'"):
        uut.main(
            [
                "--motion_backend",
                uut.PROXY_BACKEND,
                "--rc5_move_group",
                "bad_group",
                "--config_path",
                str(cfg_path),
                "--key",
                "demo_key",
                "--scene",
                "scene.json",
            ]
        )


def test_rc5_unified_main_headless_save_debug_gif_wires_singleton_gif_artifact(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=True)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)
    observed = {}

    def fake_execute(request):
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        observed["argv"] = list(plan.forwarded_argv)
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
        )

    monkeypatch.setattr(uut, "execute_backend_request", fake_execute)

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--config_path",
            str(cfg_path),
            "--key",
            "demo_key",
            "--scene",
            "scene.json",
            "--headless",
            "--save_debug_gif",
        ]
    )

    assert rc == 0
    assert "--save_video_gif_on_exit" in observed["argv"]
    assert observed["argv"][observed["argv"].index("--save_video_gif_path") + 1] == str(
        run_dir / "debug_video.gif"
    )


def test_rc5_unified_main_derives_key_from_scene_path_when_key_is_omitted(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path, teleop=True)
    run_dir = _install_auto_run_dir(monkeypatch, tmp_path)
    observed = {}

    def fake_execute(request):
        plan = uut.resolve_backend(request.motion_backend).build_dispatch_plan(request)
        observed["argv"] = list(plan.forwarded_argv)
        return UnifiedBackendResult(
            motion_backend=request.motion_backend,
            exit_code=0,
            dispatch_plan=plan,
        )

    monkeypatch.setattr(uut, "execute_backend_request", fake_execute)

    rc = uut.main(
        [
            "--motion_backend",
            uut.PROXY_BACKEND,
            "--config_path",
            str(cfg_path),
            "--scene",
            "/tmp/assets/scenes/demo_key/simulation/scene.json",
        ]
    )

    assert rc == 0
    assert observed["argv"] == [
        "--config_path",
        str(cfg_path),
        "--scene",
        "/tmp/assets/scenes/demo_key/simulation/scene.json",
        "--rl4vla_raw_episode_output",
        str(run_dir / "rl4vla_raw_episode.npz"),
        "--save_video_gif_on_exit",
        "--save_video_gif_path",
        str(run_dir / "debug_video.gif"),
        "--runtime_request_path",
        str(run_dir / "runtime_request.json"),
        "--key",
        "demo_key",
        "--robot_uids",
        "rc5_aero_hand_openr2s_rl",
        "--sim_backend",
        "physx_cuda",
        "--control_mode",
        "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        "--teleop_profile_config",
        "config/teleop_profiles.yaml",
        "--teleop_profile",
        "rc5_teleop_v1",
        "--auto_pick_macro",
        "1",
    ]


def test_rc5_unified_main_fails_fast_for_non_executable_task_type(tmp_path):
    cfg_path = _write_config(tmp_path, teleop=True)

    with pytest.raises(ValueError, match="is modeled in the unified task layer but is not yet executable"):
        uut.main(
            [
                "--motion_backend",
                uut.PROXY_BACKEND,
                "--task_type",
                "pick_and_place",
                "--task_object_id",
                "coke_can",
                "--task_destination_id",
                "yellow_plate",
                "--config_path",
                str(cfg_path),
                "--key",
                "demo_key",
                "--scene",
                "scene.json",
            ]
        )


def test_tee_stream_prefixes_each_log_line_with_datetime() -> None:
    stream = io.StringIO()
    tee = uut._TeeStream(
        stream,
        timestamp_fn=lambda: uut.datetime(2026, 5, 23, 12, 34, 56),
    )

    tee.write("first line")
    tee.write(" continued\nsecond line\n")

    assert (
        stream.getvalue()
        == "[2026-05-23 12:34:56] first line continued\n"
        "[2026-05-23 12:34:56] second line\n"
    )
