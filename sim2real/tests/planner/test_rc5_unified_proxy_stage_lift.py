from __future__ import annotations

import types

import numpy as np
import pytest

import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_stage_lift as uut


def _make_env():
    return types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(
                uid="fake_uid",
                is_grasping=lambda _obj: True,
            ),
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            manip_object_id="orange_cube_ext",
        )
    )


def test_lift_proxy_backend_delegates_to_proxy_pose_stage():
    observed = {"calls": 0, "saved": None, "stage_kwargs": []}
    target_pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.2], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    env = _make_env()
    env.unwrapped._debug_planner_config = {
        "planner_proxy_lift_z_step_m": 0.02,
        "planner_proxy_lift_pos_tol_m": 0.01,
    }

    result = uut.run_planner_lift(
        env,
        lift_delta_z=0.1,
        execute=True,
        repeat=2,
        backend="proxy_ee_delta",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: type("Planner", (), {}),
        is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
        is_ee_delta_control_mode=lambda _mode: True,
        run_proxy_ee_delta_pose_stage=lambda *_args, **kwargs: observed.update(
            calls=observed["calls"] + 1
        )
        or observed["stage_kwargs"].append(kwargs)
        or True,
        pose_to_numpy=lambda _pose: (np.array([0.0, 0.0, 0.2], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: target_pose,
        get_debug_target_object=lambda _env: object(),
        get_debug_actor_position_xyz=lambda _obj: np.array([0.0, 0.0, 0.15], dtype=np.float32),
        to_scalar_bool=lambda _value: True,
        get_robot_hand_qpos_debug=lambda _env: np.array([0.1, 0.2], dtype=np.float32),
        get_robot_qpos=lambda _env: np.zeros(8, dtype=np.float32),
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        build_debug_lift_pose_from_policy=lambda _env, lift_delta_z: (
            target_pose,
            types.SimpleNamespace(semantics="prehand"),
            types.SimpleNamespace(source_stage="close"),
            types.SimpleNamespace(pose_world=target_pose),
        ),
        is_rc5_debug_planner_agent=lambda _uid: False,
        evaluate_rc5_pick_lift_success=lambda **_kwargs: types.SimpleNamespace(success=True, reason="ok"),
        get_planner_grasp_state=lambda _env: types.SimpleNamespace(grasp_flag=True),
        save_planner_grasp_state=lambda *_args, **kwargs: observed.update(saved=kwargs),
        refresh_render_state=lambda _env: None,
    )

    assert result is True
    assert observed["calls"] == 2
    assert all(
        kwargs["max_z_step_override"] == pytest.approx(0.02)
        and kwargs["pos_tol_override"] == pytest.approx(0.01)
        for kwargs in observed["stage_kwargs"]
    )
    assert observed["saved"]["source_stage"] == "lift"


def test_proxy_lift_wrapper_delegates_to_planner_entry(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    monkeypatch.setattr(
        uut,
        "run_planner_lift",
        lambda *args, **kwargs: observed.update(args=args, kwargs=kwargs) or "ok",
    )

    result = uut.run_proxy_lift("env", backend="proxy_ee_delta")

    assert result == "ok"
    assert observed["args"] == ("env",)
    assert observed["kwargs"]["backend"] == "proxy_ee_delta"


def test_lift_jointspace_path_reports_failure_on_bad_retention():
    observed = {"solver_closed": False}
    target_pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.2], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    class _Planner:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Solver:
        def move_to_pose(self, _pose, dry_run=False):
            return {"status": "ok", "position": np.zeros((2, 7), dtype=np.float32)}

        def move_to_pose_with_RRTConnect(self, _pose, dry_run=False):
            return {"status": "ok", "position": np.zeros((2, 7), dtype=np.float32)}

        def close(self):
            observed["solver_closed"] = True

    result = uut.run_planner_lift(
        _make_env(),
        lift_delta_z=0.1,
        method="auto",
        execute=True,
        repeat=1,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Planner,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: True,
        pose_to_numpy=lambda _pose: (np.array([0.0, 0.0, 0.2], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: target_pose,
        get_debug_target_object=lambda _env: object(),
        get_debug_actor_position_xyz=lambda _obj: np.array([0.0, 0.0, 0.15], dtype=np.float32),
        to_scalar_bool=lambda _value: True,
        get_robot_hand_qpos_debug=lambda _env: np.array([0.1, 0.2], dtype=np.float32),
        get_robot_qpos=lambda _env: np.zeros(8, dtype=np.float32),
        configure_debug_planner_solver_runtime=lambda solver: _Solver(),
        get_planner_recording_kwargs=lambda _env: {},
        build_debug_lift_pose_from_policy=lambda _env, lift_delta_z: (
            target_pose,
            types.SimpleNamespace(semantics="prehand"),
            types.SimpleNamespace(source_stage="close"),
            types.SimpleNamespace(pose_world=target_pose),
        ),
        is_rc5_debug_planner_agent=lambda _uid: False,
        evaluate_rc5_pick_lift_success=lambda **_kwargs: types.SimpleNamespace(success=False, reason="drop"),
        get_planner_grasp_state=lambda _env: types.SimpleNamespace(grasp_flag=True, target_hand_qpos=np.array([0.1, 0.2], dtype=np.float32)),
        save_planner_grasp_state=lambda *_args, **_kwargs: None,
        refresh_render_state=lambda _env: None,
        execute_real_planner_pose_with_backend=lambda solver, pose, **kwargs: solver.move_to_pose(
            pose,
            dry_run=not kwargs["execute"],
        ),
    )

    assert result is False
    assert observed["solver_closed"] is True


def test_lift_jointspace_path_uses_real_planner_pose_executor():
    observed = {"executor_calls": [], "solver_closed": False}
    target_pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.2], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    class _Planner:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Solver:
        def close(self):
            observed["solver_closed"] = True

    def _executor(solver, pose, **kwargs):
        observed["executor_calls"].append(
            {
                "solver": solver,
                "pose": pose,
                "backend": kwargs["backend"],
                "method": kwargs["method"],
                "stage_label": kwargs["stage_label"],
                "execute": kwargs["execute"],
            }
        )
        return {"status": "ok", "position": np.zeros((2, 7), dtype=np.float32)}

    result = uut.run_planner_lift(
        _make_env(),
        lift_delta_z=0.1,
        method="local_ik",
        execute=False,
        repeat=1,
        backend="local_ik",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Planner,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: True,
        pose_to_numpy=lambda _pose: (
            np.array([0.0, 0.0, 0.2], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        get_debug_planner_ee_pose=lambda _env: target_pose,
        get_debug_target_object=lambda _env: None,
        get_debug_actor_position_xyz=lambda _obj: np.array([0.0, 0.0, 0.15], dtype=np.float32),
        to_scalar_bool=lambda _value: True,
        get_robot_hand_qpos_debug=lambda _env: np.array([0.1, 0.2], dtype=np.float32),
        get_robot_qpos=lambda _env: np.zeros(8, dtype=np.float32),
        configure_debug_planner_solver_runtime=lambda solver: _Solver(),
        get_planner_recording_kwargs=lambda _env: {},
        build_debug_lift_pose_from_policy=lambda _env, lift_delta_z: (
            target_pose,
            types.SimpleNamespace(semantics="prehand"),
            types.SimpleNamespace(source_stage="close"),
            types.SimpleNamespace(pose_world=target_pose),
        ),
        is_rc5_debug_planner_agent=lambda _uid: False,
        evaluate_rc5_pick_lift_success=lambda **_kwargs: types.SimpleNamespace(success=True, reason="ok"),
        get_planner_grasp_state=lambda _env: None,
        save_planner_grasp_state=lambda *_args, **_kwargs: None,
        refresh_render_state=lambda _env: None,
        execute_real_planner_pose_with_backend=_executor,
    )

    assert result is True
    assert len(observed["executor_calls"]) == 1
    assert observed["executor_calls"][0]["backend"] == "local_ik"
    assert observed["executor_calls"][0]["method"] == "local_ik"
    assert observed["executor_calls"][0]["stage_label"] == "Lift"
    assert observed["executor_calls"][0]["execute"] is False
    assert observed["solver_closed"] is True


def test_lift_proxy_backend_uses_canonical_retention_pose_not_active_move_group_pose():
    observed = {"distance": None}
    target_pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.2], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    object_pose_after_lift = np.array([0.0, 0.0, 0.15], dtype=np.float32)
    active_move_group_pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.2], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    canonical_retention_pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.16], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    def _eval_success(**kwargs):
        observed["distance"] = kwargs["object_tcp_dist_after_lift"]
        return types.SimpleNamespace(success=True, reason="ok")

    result = uut.run_planner_lift(
        _make_env(),
        lift_delta_z=0.1,
        execute=True,
        repeat=1,
        backend="proxy_ee_delta",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: type("Planner", (), {}),
        is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
        is_ee_delta_control_mode=lambda _mode: True,
        run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: True,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32).copy(),
            np.asarray(pose.q, dtype=np.float32).copy(),
        ),
        get_debug_planner_ee_pose=lambda _env: active_move_group_pose,
        get_debug_planner_retention_pose=lambda _env: canonical_retention_pose,
        get_debug_target_object=lambda _env: object(),
        get_debug_actor_position_xyz=lambda _obj: object_pose_after_lift.copy(),
        to_scalar_bool=lambda _value: True,
        get_robot_hand_qpos_debug=lambda _env: np.array([0.1, 0.2], dtype=np.float32),
        get_robot_qpos=lambda _env: np.zeros(8, dtype=np.float32),
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        build_debug_lift_pose_from_policy=lambda _env, lift_delta_z: (
            target_pose,
            types.SimpleNamespace(semantics="prehand"),
            types.SimpleNamespace(source_stage="close"),
            types.SimpleNamespace(pose_world=target_pose),
        ),
        is_rc5_debug_planner_agent=lambda _uid: False,
        evaluate_rc5_pick_lift_success=_eval_success,
        get_planner_grasp_state=lambda _env: types.SimpleNamespace(grasp_flag=True),
        save_planner_grasp_state=lambda *_args, **_kwargs: None,
        refresh_render_state=lambda _env: None,
    )

    assert result is True
    assert observed["distance"] == pytest.approx(0.01, abs=1e-6)


def test_get_runtime_gripper_target_qpos_rows_repeats_single_row_for_batch(capsys):
    agent = types.SimpleNamespace(
        controller=types.SimpleNamespace(
            controllers={
                "gripper": types.SimpleNamespace(
                    _target_qpos=np.asarray([[0.3, 0.5]], dtype=np.float32)
                )
            }
        )
    )

    rows = uut._get_runtime_gripper_target_qpos_rows(agent, num_envs=2)

    assert rows.shape == (2, 2)
    assert np.allclose(rows, np.asarray([[0.3, 0.5], [0.3, 0.5]], dtype=np.float32))
    captured = capsys.readouterr()
    assert "runtime_gripper_target_qpos is single-env in a batched lift" in captured.out


def test_lift_proxy_backend_batched_diagnostics_accept_scalar_close_grasp_state():
    observed = {"saved": None}
    batched_pose = types.SimpleNamespace(
        p=np.asarray(
            [
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.2],
            ],
            dtype=np.float32,
        ),
        q=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
    )
    target_object = types.SimpleNamespace(
        pose=types.SimpleNamespace(
            raw_pose=np.asarray(
                [
                    [0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=3,
            agent=types.SimpleNamespace(
                uid="rc5_aero_hand_openr2s_rl",
                is_grasping=lambda _obj: True,
            ),
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            manip_object_id="orange_cube_ext",
            evaluate=lambda: {"is_src_obj_grasped": np.asarray([True, True, True], dtype=bool)},
        )
    )

    result = uut.run_planner_lift(
        env,
        lift_delta_z=0.1,
        execute=True,
        repeat=1,
        backend="proxy_ee_delta",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: type("Planner", (), {}),
        is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
        is_ee_delta_control_mode=lambda _mode: True,
        run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: True,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32).reshape(-1, 3)[0].copy(),
            np.asarray(pose.q, dtype=np.float32).reshape(-1, 4)[0].copy(),
        ),
        get_debug_planner_ee_pose=lambda _env: batched_pose,
        get_debug_planner_retention_pose=lambda _env: batched_pose,
        get_debug_target_object=lambda _env: target_object,
        get_debug_actor_position_xyz=lambda _obj: np.array([0.0, 0.0, 0.15], dtype=np.float32),
        to_scalar_bool=lambda _value: True,
        get_robot_hand_qpos_debug=lambda _env: np.array([0.1, 0.2], dtype=np.float32),
        get_robot_qpos=lambda _env: np.zeros(8, dtype=np.float32),
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        build_debug_lift_pose_from_policy=lambda _env, lift_delta_z: (
            batched_pose,
            types.SimpleNamespace(semantics="prehand"),
            types.SimpleNamespace(source_stage="close"),
            types.SimpleNamespace(pose_world=batched_pose),
        ),
        is_rc5_debug_planner_agent=lambda _uid: False,
        evaluate_rc5_pick_lift_success=lambda **_kwargs: types.SimpleNamespace(success=True, reason="ok"),
        get_planner_grasp_state=lambda _env: types.SimpleNamespace(
            grasp_flag=True,
            grasp_flag_rows=np.asarray([True, True, True], dtype=bool),
        ),
        save_planner_grasp_state=lambda *_args, **kwargs: observed.update(saved=kwargs),
        refresh_render_state=lambda _env: None,
    )

    assert result is True
    assert observed["saved"]["source_stage"] == "lift"
