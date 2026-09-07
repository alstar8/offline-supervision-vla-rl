from __future__ import annotations

import types

import numpy as np

import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_stage_close as uut


def _make_env():
    return types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_uid"),
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            manip_object_id="orange_cube_ext",
        )
    )


def test_close_stage_proxy_backend_delegates_to_proxy_close():
    observed = {"called": False}

    result = uut.run_planner_close_gripper(
        _make_env(),
        close_steps=12,
        backend="proxy_ee_delta",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: type("Planner", (), {}),
        is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
        is_ee_delta_control_mode=lambda _mode: True,
        run_proxy_close_gripper=lambda _env, close_steps: observed.update(called=close_steps) or True,
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        get_debug_target_object=lambda _env: None,
        get_debug_actor_position_xyz=lambda _actor: None,
        log_debug_pre_close_snapshot=lambda *_args, **_kwargs: None,
        pose_to_numpy=lambda _pose: (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: None,
        log_debug_post_close_retention=lambda *_args, **_kwargs: (True, None, None, 0.0, 0.0),
        get_robot_hand_qpos_debug=lambda _env: [0.0],
        get_robot_hand_range_debug=lambda _env: (0.0, 0.0),
        save_planner_grasp_state=lambda **_kwargs: None,
        refresh_render_state=lambda _env: None,
    )

    assert result is True
    assert observed["called"] == 12


def test_close_stage_jointspace_path_saves_grasp_state():
    observed = {"saved": None, "solver_closed": False}

    class _Planner:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Solver:
        def close_gripper(self, t):
            observed["close_t"] = t

        def close(self):
            observed["solver_closed"] = True

    result = uut.run_planner_close_gripper(
        _make_env(),
        close_steps=9,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Planner,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        run_proxy_close_gripper=lambda _env, close_steps: False,
        configure_debug_planner_solver_runtime=lambda solver: _Solver(),
        get_planner_recording_kwargs=lambda _env: {},
        get_debug_target_object=lambda _env: None,
        get_debug_actor_position_xyz=lambda _actor: None,
        log_debug_pre_close_snapshot=lambda *_args, **_kwargs: None,
        pose_to_numpy=lambda _pose: (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: None,
        log_debug_post_close_retention=lambda *_args, **_kwargs: (True, None, None, 0.1, 0.2),
        get_robot_hand_qpos_debug=lambda _env: [0.1, 0.2],
        get_robot_hand_range_debug=lambda _env: (0.1, 0.2),
        save_planner_grasp_state=lambda *_args, **kwargs: observed.update(saved=kwargs),
        refresh_render_state=lambda _env: None,
    )

    assert result is True
    assert observed["close_t"] == 9
    assert observed["saved"]["source_stage"] == "close"
    assert observed["saved"]["object_id"] is None
    assert observed["solver_closed"] is True
