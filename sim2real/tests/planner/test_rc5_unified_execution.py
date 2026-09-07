from __future__ import annotations

import types

from openreal2sim.simulation.maniskill.scripts import rc5_unified_execution as uut


def test_planner_backend_forwards_pregrasp_refine_steps_from_runtime_overrides(monkeypatch):
    monkeypatch.setattr(uut, "inject_runtime_defaults", lambda argv, *_args, **_kwargs: list(argv))
    monkeypatch.setattr(uut, "emit_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        uut,
        "_resolve_unified_planner_backend_mode_from_request",
        lambda _request: uut.UNIFIED_PLANNER_BACKEND_RRTCONNECT,
    )
    monkeypatch.setattr(
        uut,
        "_resolve_rc5_planner_runtime_override_config_from_request",
        lambda _request: uut.RC5PlannerRuntimeOverrideConfig(
            manip_object_id="green_cube_ext",
            planner_pregrasp_method="rrtconnect",
            planner_pregrasp_refine_steps=24,
            rc5_obb_target_semantics="right_tcp_link",
            rc5_object_profile_target_semantics="right_tcp_link",
        ),
    )
    monkeypatch.setattr(
        uut,
        "_resolve_rc5_asset_override_config_from_request",
        lambda _request: uut.RC5AssetOverrideConfig(),
    )

    backend = uut.PlannerBackend()
    request = uut.UnifiedBackendRequest(
        motion_backend=uut.PLANNER_BACKEND,
        passthrough_argv=["--scene", "/tmp/scene.json", "--num_envs", "1"],
        bootstrap=types.SimpleNamespace(
            robot_uids="rc5_aero_hand_openr2s_rl",
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            config_path="/tmp/test_config.yaml",
            key="demo_key",
        ),
        task_plan=types.SimpleNamespace(intent=types.SimpleNamespace(object_id="green_cube_ext")),
    )

    plan = backend.build_dispatch_plan(request)

    refine_idx = plan.forwarded_argv.index("--planner_pregrasp_refine_steps")
    assert plan.forwarded_argv[refine_idx + 1] == "24"
