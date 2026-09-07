from __future__ import annotations

import types

import numpy as np
from mani_skill.utils.structs.pose import Pose

import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_motion as uut
import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_lowlevel as lowlevel_uut


def test_run_proxy_ee_delta_pose_stage_neutralizes_gripper_for_converged_envs():
    observed = {}
    pose_rows = types.SimpleNamespace(
        p=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        q=np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    target_pose = Pose.create_from_pq(
        p=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]], dtype=np.float32),
        q=np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=2))

    result = uut.run_proxy_ee_delta_pose_stage(
        env,
        target_pose,
        stage_label="Lift:1/1",
        position_mask=(False, False, True),
        align_orientation=False,
        gripper_target_state="hold",
        get_debug_planner_config=lambda _env: {
            "planner_proxy_max_stage_steps": 1,
            "planner_proxy_hold_steps": 1,
            "planner_proxy_pos_tol_m": 0.01,
        },
        get_debug_planner_ee_pose_sapien=lambda _env: pose_rows,
        get_debug_planner_ee_pose=lambda _env: pose_rows,
        pose_to_numpy_rows=lambda pose: (
            np.asarray(pose.p, dtype=np.float32),
            np.asarray(pose.q, dtype=np.float32),
        ),
        compute_proxy_rotvec_step=lambda *_args, **_kwargs: (np.zeros(3, dtype=np.float32), 0.0),
        build_proxy_delta_pos=lambda _env, pos_err_vec, *_args, **_kwargs: np.asarray(pos_err_vec, dtype=np.float32),
        apply_proxy_ee_delta_action=lambda _env, raw_delta_pos, raw_delta_rpy, **kwargs: observed.update(
            raw_delta_pos=np.asarray(raw_delta_pos, dtype=np.float32),
            raw_delta_rpy=np.asarray(raw_delta_rpy, dtype=np.float32),
            kwargs=kwargs,
        ),
    )

    assert result is False
    assert np.allclose(observed["raw_delta_pos"][0], np.zeros(3, dtype=np.float32))
    assert np.allclose(observed["raw_delta_pos"][1], np.array([0.0, 0.0, 0.1], dtype=np.float32))
    override = np.asarray(observed["kwargs"]["gripper_signal_override"], dtype=np.float32)
    assert float(override[0]) == 0.0
    assert np.isnan(override[1])


def test_run_proxy_ee_delta_pose_stage_uses_lift_step_and_tolerance_overrides():
    observed = {}
    current_pose = Pose.create_from_pq(
        p=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    target_pose = Pose.create_from_pq(
        p=np.asarray([0.0, 0.0, 0.1], dtype=np.float32),
        q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=1))

    result = uut.run_proxy_ee_delta_pose_stage(
        env,
        target_pose,
        stage_label="Lift:1/1",
        position_mask=(False, False, True),
        align_orientation=False,
        gripper_target_state="hold",
        max_z_step_override=0.02,
        pos_tol_override=0.01,
        get_debug_planner_config=lambda _env: {
            "planner_proxy_max_stage_steps": 1,
            "planner_proxy_hold_steps": 1,
            "planner_proxy_pos_tol_m": 0.001,
            "planner_proxy_xy_step_m": 0.004,
            "planner_proxy_z_step_m": 0.002,
            "planner_proxy_rot_step_deg": 6.0,
            "planner_proxy_rot_tol_deg": 8.0,
            "planner_proxy_stall_steps": 12,
            "planner_proxy_frame": "world",
        },
        get_debug_planner_ee_pose_sapien=lambda _env: current_pose,
        compute_proxy_rotvec_step=lambda *_args, **_kwargs: (
            np.zeros(3, dtype=np.float32),
            0.0,
        ),
        build_proxy_delta_pos=lambda _env, pos_err_vec, *_args, **kwargs: observed.update(
            max_z_step=kwargs["max_z_step"]
        )
        or np.asarray(
            [0.0, 0.0, np.clip(pos_err_vec[2], -kwargs["max_z_step"], kwargs["max_z_step"])],
            dtype=np.float32,
        ),
        apply_proxy_ee_delta_action=lambda _env, raw_delta_pos, **_kwargs: observed.update(
            raw_delta_pos=np.asarray(raw_delta_pos, dtype=np.float32)
        ),
    )

    assert result is False
    assert observed["max_z_step"] == 0.02
    assert np.isclose(observed["raw_delta_pos"][2], 0.02)


def test_apply_proxy_ee_delta_action_supports_per_env_gripper_signal_override():
    observed = {}

    class _Agent:
        tcp = types.SimpleNamespace(pose=types.SimpleNamespace())

        def __init__(self):
            self.robot = types.SimpleNamespace(
                pose=types.SimpleNamespace(
                    to_transformation_matrix=lambda: np.eye(4, dtype=np.float64)
                )
            )

    env_unwrapped = types.SimpleNamespace(
        num_envs=2,
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        agent=_Agent(),
    )
    env = types.SimpleNamespace(
        unwrapped=env_unwrapped,
        action_space=types.SimpleNamespace(shape=(7,)),
        step=lambda action: observed.update(action=np.asarray(action, dtype=np.float32)) or None,
    )

    lowlevel_uut.apply_proxy_ee_delta_action(
        env,
        raw_delta_pos=np.zeros((2, 3), dtype=np.float32),
        raw_delta_rpy=np.zeros((2, 3), dtype=np.float32),
        hold_steps=1,
        stage_label="Lift:1/1:step1",
        gripper_target_state="hold",
        gripper_signal_override=np.asarray([0.0, 1.0], dtype=np.float32),
        pose_to_numpy=lambda _pose: (
            np.zeros(3, dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
    )

    assert np.allclose(observed["action"][:, 6], np.asarray([0.0, 1.0], dtype=np.float32))


def test_guarded_descend_applies_rotation_when_orientation_is_required(monkeypatch):
    observed = {}
    current_pose = Pose.create_from_pq(
        p=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    target_pose = Pose.create_from_pq(
        p=np.asarray([0.0, 0.0, -0.01], dtype=np.float32),
        q=np.asarray([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            num_envs=1,
            _debug_planner_config={
                "planner_proxy_adaptive_steps_enabled": False,
                "planner_proxy_threshold_xy_m": 0.01,
                "planner_proxy_threshold_z_m": 0.01,
                "planner_proxy_pos_tol_m": 0.001,
                "planner_proxy_frame": "world",
                "planner_proxy_xy_step_m": 0.004,
                "planner_proxy_z_step_m": 0.002,
                "planner_proxy_rot_step_deg": 6.0,
                "planner_proxy_rot_tol_deg": 8.0,
                "planner_proxy_hold_steps": 1,
                "planner_proxy_max_stage_steps": 1,
                "planner_proxy_stall_steps": 12,
            },
        )
    )
    actor = object()
    monkeypatch.setattr(
        lowlevel_uut,
        "apply_proxy_ee_delta_action",
        lambda _env, raw_delta_pos, raw_delta_rpy, **_kwargs: observed.update(
            delta_pos=np.asarray(raw_delta_pos, dtype=np.float32),
            delta_rpy=np.asarray(raw_delta_rpy, dtype=np.float32),
        ),
    )

    result = lowlevel_uut.run_proxy_guarded_descend_to_object(
        env,
        target_pose,
        initial_actor_p=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        bbox_np=np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
        stage_label="FullApproach:descent",
        align_orientation=True,
        get_debug_planner_ee_pose_sapien=lambda _env: current_pose,
        get_debug_target_object=lambda _env: actor,
        get_debug_actor_position_xyz=lambda _actor: np.asarray(
            [0.0, 0.0, 0.0], dtype=np.float32
        ),
    )

    assert result is False
    assert np.linalg.norm(observed["delta_pos"]) > 0.0
    assert np.linalg.norm(observed["delta_rpy"]) > 0.0


def test_run_proxy_full_approach_to_descend_executes_waypoints_before_rise():
    stage_calls = []
    settle_calls = []

    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=1))
    target_pose = Pose.create_from_pq(
        p=np.asarray([0.5, 0.6, 0.7], dtype=np.float32),
        q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    result = uut.run_proxy_full_approach_to_descend(
        env,
        target_pose,
        initial_actor_p=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        bbox_np=np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
        stage_label="FullApproach",
        safe_clearance_z=0.10,
        get_debug_planner_config=lambda _env: {
            "planner_proxy_predescent_settle_steps": 2,
            "planner_proxy_pregrasp_joint_guard": {
                "enabled": False,
                "joint_targets": {},
                "tolerance_rad": 0.1,
                "mismatch_policy": "warn",
                "run_align_stage": False,
                "align_solver_overrides": {
                    "posture_gain": 0.0,
                    "posture_gain_near_target": 0.0,
                    "posture_joint_weights": [1.0],
                },
            },
            "planner_waypoints": {
                "enabled": True,
                "points": [
                    {"id": "wp_01", "position": [-0.3, -0.6, 0.42]},
                    {"id": "wp_02", "position": [-0.2, -0.5, 0.48]},
                ],
            },
        },
        get_debug_planner_ee_pose_sapien=lambda _env: Pose.create_from_pq(
            p=np.asarray([0.0, 0.0, 0.4], dtype=np.float32),
            q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        get_debug_planner_ee_pose=None,
        pose_to_numpy_rows=None,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3],
            np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4],
        ),
        run_proxy_ee_delta_pose_stage=lambda _env, pose, **kwargs: stage_calls.append(
            (
                kwargs["stage_label"],
                tuple(bool(x) for x in kwargs["position_mask"]),
                np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3].copy(),
            )
        ) or True,
        run_proxy_stationary_settle=lambda _env, **kwargs: settle_calls.append(kwargs["stage_label"]) or True,
        run_proxy_guarded_descend_to_object=lambda *_args, **_kwargs: True,
    )

    assert result is True
    assert [call[0] for call in stage_calls] == [
        "FullApproach:waypoint_001",
        "FullApproach:waypoint_002",
        "FullApproach:rise_to_safe_z",
        "FullApproach:move_xy_above_target",
    ]
    assert stage_calls[0][1] == (True, True, True)
    assert stage_calls[1][1] == (True, True, True)
    assert stage_calls[2][1] == (False, False, True)
    assert stage_calls[3][1] == (True, True, False)
    assert np.allclose(stage_calls[0][2], np.asarray([-0.3, -0.6, 0.42], dtype=np.float32))
    assert np.allclose(stage_calls[1][2], np.asarray([-0.2, -0.5, 0.48], dtype=np.float32))
    assert settle_calls == ["PreDescentSettle"]


def test_current_tcp_profile_maintains_orientation_through_full_approach():
    stage_calls = []
    descend_calls = []
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(num_envs=1, manip_object_id="spray_bottle_ext")
    )
    target_pose = Pose.create_from_pq(
        p=np.asarray([0.5, 0.6, 0.3], dtype=np.float32),
        q=np.asarray([0.7, 0.0, 0.0, 0.7], dtype=np.float32),
    )
    planner_cfg = {
        "planner_object_calibrations": {
            "spray_bottle_ext": {"target_orientation_mode": "current_tcp"}
        },
        "planner_proxy_predescent_settle_steps": 0,
        "planner_proxy_pregrasp_joint_guard": {
            "enabled": False,
            "joint_targets": {},
            "tolerance_rad": 0.1,
            "mismatch_policy": "warn",
            "run_align_stage": False,
            "align_solver_overrides": {
                "posture_gain": 0.0,
                "posture_gain_near_target": 0.0,
                "posture_joint_weights": [1.0],
            },
        },
        "planner_waypoints": {"enabled": False, "points": []},
    }

    result = uut.run_proxy_full_approach_to_descend(
        env,
        target_pose,
        initial_actor_p=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        bbox_np=np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
        stage_label="FullApproach",
        safe_clearance_z=0.10,
        get_debug_planner_config=lambda _env: planner_cfg,
        get_debug_planner_ee_pose_sapien=lambda _env: Pose.create_from_pq(
            p=np.asarray([0.0, 0.0, 0.4], dtype=np.float32),
            q=np.asarray([0.7, 0.0, 0.0, 0.7], dtype=np.float32),
        ),
        get_debug_planner_ee_pose=None,
        pose_to_numpy_rows=None,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3],
            np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4],
        ),
        run_proxy_ee_delta_pose_stage=lambda _env, _pose, **kwargs: stage_calls.append(
            (kwargs["stage_label"], kwargs["align_orientation"])
        )
        or True,
        run_proxy_stationary_settle=lambda *_args, **_kwargs: True,
        run_proxy_guarded_descend_to_object=lambda *_args, **kwargs: descend_calls.append(
            kwargs
        )
        or True,
    )

    assert result is True
    assert stage_calls == [
        ("FullApproach:rise_to_safe_z", True),
        ("FullApproach:move_xy_above_target", True),
    ]
    assert descend_calls[0]["align_orientation"] is True


def test_execute_real_planner_pose_with_backend_rejects_proxy_backend():
    try:
        uut.execute_real_planner_pose_with_backend(
            types.SimpleNamespace(),
            object(),
            execute=True,
            backend="proxy_ee_delta",
            method="auto",
            planner_class_name="Planner",
            stage_label="Descend",
            is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
            is_rc5_debug_planner_agent=lambda _uid: False,
        )
    except ValueError as exc:
        message = str(exc)
        assert "Proxy planner backend must be dispatched directly" in message
        assert "Descend" in message
    else:
        raise AssertionError("Expected real planner / mplib planner dispatcher to reject proxy backend")


def test_execute_real_planner_pose_with_backend_uses_local_ik_when_requested():
    observed = {}

    class _Solver:
        base_env = types.SimpleNamespace(agent=types.SimpleNamespace(uid="fake_uid"))

        def move_to_pose_with_local_ik(self, target_pose, dry_run=False):
            observed["target_pose"] = target_pose
            observed["dry_run"] = dry_run
            return {"status": "ok"}

    target_pose = object()
    result = uut.execute_real_planner_pose_with_backend(
        _Solver(),
        target_pose,
        execute=True,
        backend="local_ik",
        method="auto",
        planner_class_name="Planner",
        stage_label="Descend",
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_rc5_debug_planner_agent=lambda _uid: False,
    )

    assert result == {"status": "ok"}
    assert observed["target_pose"] is target_pose
    assert observed["dry_run"] is False


def test_execute_planner_pose_with_backend_alias_preserves_real_planner_dispatch():
    observed = {}

    class _Solver:
        base_env = types.SimpleNamespace(agent=types.SimpleNamespace(uid="fake_uid"))

        def move_to_pose(self, target_pose, dry_run=False):
            observed["target_pose"] = target_pose
            observed["dry_run"] = dry_run
            return {"status": "compat"}

    target_pose = object()
    result = uut.execute_planner_pose_with_backend(
        _Solver(),
        target_pose,
        execute=False,
        backend="planner",
        method="auto",
        planner_class_name="Planner",
        stage_label="Descend",
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_rc5_debug_planner_agent=lambda _uid: False,
    )

    assert result == {"status": "compat"}
    assert observed["target_pose"] is target_pose
    assert observed["dry_run"] is True
