from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "openreal2sim"
    / "simulation"
    / "maniskill"
    / "scripts"
    / "rc5_unified_proxy_stage_pose.py"
)
_SPEC = importlib.util.spec_from_file_location("test_rc5_unified_proxy_stage_pose_uut", _MODULE_PATH)
uut = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(uut)


def _make_env():
    return types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="fake_uid"),
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            _debug_planner_config={},
        )
    )


def _pose():
    return types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.2], dtype=np.float32),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_pregrasp_proxy_backend_delegates_to_proxy_full_approach():
    observed = {"called": False}

    result = uut.run_planner_object_pregrasp_probe(
        _make_env(),
        execute=True,
        backend="proxy_ee_delta",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: type("Planner", (), {}),
        is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
        is_proxy_then_planner_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: True,
        maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        build_object_pregrasp_target=lambda _env, extra_clearance, radial_backoff_override=None: (
            "obj",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            _pose(),
        ),
        run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: observed.update(called=True) or True,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        get_object_specific_planner_profile=lambda _cfg, _obj: None,
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda _pose: (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: _pose(),
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
    )

    assert result is True
    assert observed["called"] is True


def test_proxy_pregrasp_wrapper_delegates_to_planner_entry(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    monkeypatch.setattr(
        uut,
        "run_planner_object_pregrasp_probe",
        lambda *args, **kwargs: observed.update(args=args, kwargs=kwargs) or "ok",
    )

    result = uut.run_proxy_pregrasp_probe("env", backend="proxy_ee_delta")

    assert result == "ok"
    assert observed["args"] == ("env",)
    assert observed["kwargs"]["backend"] == "proxy_ee_delta"


def test_descend_proxy_backend_delegates_to_proxy_pose_stage():
    observed = {"called": False}

    result = uut.run_planner_object_descend(
        _make_env(),
        execute=True,
        backend="proxy_ee_delta",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: type("Planner", (), {}),
        is_proxy_ee_delta_backend=lambda backend: backend == "proxy_ee_delta",
        is_ee_delta_control_mode=lambda _mode: True,
        build_object_descend_target=lambda _env, extra_clearance: (
            "obj",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            _pose(),
        ),
        run_proxy_ee_delta_pose_stage=lambda *_args, **_kwargs: observed.update(called=True) or True,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda _pose: (np.array([0.0, 0.0, 0.2], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: _pose(),
        run_linear_approach_waypoints=lambda *_args, **_kwargs: True,
        execute_planner_pose_with_backend=lambda *_args, **_kwargs: {"status": "ok", "position": np.zeros((2, 7), dtype=np.float32)},
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
    )

    assert result is True
    assert observed["called"] is True


def test_proxy_descend_wrapper_delegates_to_planner_entry(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    monkeypatch.setattr(
        uut,
        "run_planner_object_descend",
        lambda *args, **kwargs: observed.update(args=args, kwargs=kwargs) or "ok",
    )

    result = uut.run_proxy_descend("env", backend="proxy_ee_delta")

    assert result == "ok"
    assert observed["args"] == ("env",)
    assert observed["kwargs"]["backend"] == "proxy_ee_delta"


def test_proxy_full_approach_wrapper_delegates_to_planner_entry(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    monkeypatch.setattr(
        uut,
        "run_planner_full_approach_to_descend",
        lambda *args, **kwargs: observed.update(args=args, kwargs=kwargs) or "ok",
    )

    result = uut.run_proxy_full_approach_to_descend("env", backend="proxy_ee_delta")

    assert result == "ok"
    assert observed["args"] == ("env",)
    assert observed["kwargs"]["backend"] == "proxy_ee_delta"


def test_full_approach_jointspace_requires_waypoints():
    env = _make_env()
    env.unwrapped._debug_planner_config = {"planner_approach_waypoints": 0}

    class _Planner:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Solver:
        def close(self):
            pass

    result = uut.run_planner_full_approach_to_descend(
        env,
        execute=True,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Planner,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        build_object_descend_target=lambda _env, extra_clearance: (
            "obj",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            _pose(),
        ),
        run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: True,
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda _pose: (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        get_debug_planner_ee_pose=lambda _env: _pose(),
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=lambda solver: _Solver(),
        get_planner_recording_kwargs=lambda _env: {},
        run_linear_approach_waypoints=lambda *_args, **_kwargs: True,
        execute_planner_pose_with_backend=lambda *_args, **_kwargs: {"status": "ok", "position": np.zeros((2, 7), dtype=np.float32)},
    )

    assert result is False


def test_pregrasp_execute_reuses_selected_preview_result_and_branch_debug(monkeypatch):
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="rc5_aero_hand_openr2s"),
            control_mode="pd_joint_pos",
            _debug_planner_config={},
        )
    )
    target_pose = _pose()
    holder = {}

    class _Solver:
        def __init__(self, *_args, **_kwargs):
            self.preview_calls = 0
            self.follow_calls = 0
            self.branch_debug = None
            self.followed_result = None

        def move_to_pose_with_RRTConnect(self, _target_pose, dry_run=False):
            assert dry_run is True
            self.preview_calls += 1
            self.branch_debug = {"candidate_idx": self.preview_calls}
            return {
                "status": "Success",
                "position": np.zeros((2, 7), dtype=np.float32),
                "candidate_idx": self.preview_calls,
            }

        def follow_path(self, result, refine_steps=0):
            assert refine_steps == 0
            self.follow_calls += 1
            self.followed_result = result
            return ("obs", 0.0, False, False, {})

        def get_last_plan_branch_debug(self):
            return self.branch_debug

        def close(self):
            return None

    def _configure_solver(solver):
        holder["solver"] = solver
        return solver

    def _score_candidate(solver, planner_cfg, *, target_pose, branch_debug_override=None):
        branch_debug = branch_debug_override or solver.get_last_plan_branch_debug()
        idx = branch_debug["candidate_idx"]
        acceptable = idx in {1, 2, 3}
        preview_err_by_idx = {
            1: 7e-06,
            2: 4e-06,
            3: 3e-06,
            4: 8e-06,
        }
        score = (
            0 if acceptable else 1,
            0 if acceptable else 2,
            0 if acceptable else 3,
            preview_err_by_idx[idx],
            0.0 if acceptable else 5.0,
        )
        metrics = {
            "wrap_count": 0 if acceptable else 2,
            "large_delta_count": 0 if acceptable else 3,
            "max_abs_raw_delta": 0.0 if acceptable else 5.0,
            "preview_fk_pos_err": preview_err_by_idx[idx],
        }
        return acceptable, score, metrics

    def _branch_guard_accepts(
        solver,
        planner_cfg,
        *,
        stage_label,
        method_name,
        branch_debug_override=None,
    ):
        branch_debug = branch_debug_override or solver.get_last_plan_branch_debug()
        idx = branch_debug["candidate_idx"]
        if stage_label == "PregraspPreview":
            return idx in {1, 2, 3}
        if stage_label == "Pregrasp":
            assert branch_debug_override is not None
            assert idx == 3
            return True
        return True

    monkeypatch.setattr(uut, "_score_planner_pregrasp_preview_candidate", _score_candidate)
    monkeypatch.setattr(uut, "_planner_branch_guard_accepts", _branch_guard_accepts)

    result = uut.run_planner_object_pregrasp_probe(
        env,
        method="rrtconnect",
        execute=True,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Solver,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_proxy_then_planner_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        build_object_pregrasp_target=lambda _env, extra_clearance, radial_backoff_override=None: (
            "green_cube_ext",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            target_pose,
        ),
        run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: True,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=_configure_solver,
        get_planner_recording_kwargs=lambda _env: {},
        get_object_specific_planner_profile=lambda _cfg, _obj: None,
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32),
            np.asarray(pose.q, dtype=np.float32),
        ),
        get_debug_planner_ee_pose=lambda _env: target_pose,
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
    )

    solver = holder["solver"]
    assert result is True
    assert solver.preview_calls == 4
    assert solver.follow_calls == 1
    assert solver.followed_result["candidate_idx"] == 3


def test_pregrasp_failure_logs_execution_consistency_diagnostics(monkeypatch, capsys):
    target_object = types.SimpleNamespace()
    scene = types.SimpleNamespace(
        get_pairwise_contact_forces=lambda link, actor: {
            "thumb": np.array([[0.6, 0.0, 0.0]], dtype=np.float32),
            "index": np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.8, 0.0, 0.0]], dtype=np.float32),
        }[link.name]
    )
    arm_controller = types.SimpleNamespace(
        config=types.SimpleNamespace(frame="base", use_target=True),
        target_qpos=np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32),
        _target_qpos=np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32),
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(
                uid="rc5_aero_hand_openr2s",
                controller=types.SimpleNamespace(controllers={"arm": arm_controller}),
                thumb_tip_link=types.SimpleNamespace(name="thumb"),
                index_tip_link=types.SimpleNamespace(name="index"),
                middle_tip_link=types.SimpleNamespace(name="middle"),
                scene=scene,
            ),
            control_mode="pd_joint_pos",
            _debug_planner_config={},
            manip_object_id="green_cube_ext",
            object_actors={"green_cube_ext": target_object},
        )
    )
    target_pose = _pose()

    class _Solver:
        def __init__(self, *_args, **_kwargs):
            self.env = env
            self.branch_debug = {
                "last_qpos": np.array([1.0, 0.5, -0.25, 0.0, 0.75, -1.0], dtype=np.float32),
                "start_to_last": {
                    "suspicious_indices": np.array([], dtype=np.int32),
                    "raw_delta": np.zeros(6, dtype=np.float32),
                },
            }
            self.current_full_qpos = np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.0, 9.0],
                dtype=np.float32,
            )

        def move_to_pose_with_RRTConnect(self, _target_pose, dry_run=False):
            assert dry_run is True
            return {
                "status": "Success",
                "position": np.zeros((2, 6), dtype=np.float32),
            }

        def follow_path(self, result, refine_steps=0):
            assert refine_steps == 0
            self._last_follow_path_trace = [
                {
                    "step_index": 11,
                    "commanded_arm_qpos": np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32),
                    "actual_arm_qpos": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
                    "arm_tracking_err": np.array([-0.9, -0.9, -0.9, -0.9, -0.9, -0.9], dtype=np.float32),
                }
            ]
            return ("obs", 0.0, False, False, {})

        def get_last_plan_branch_debug(self):
            return self.branch_debug

        def _get_current_arm_qpos(self):
            return self.current_full_qpos[:6]

        def _get_current_qpos(self):
            return self.current_full_qpos.copy()

        def _prepare_target_pose_for_solver(self, pose):
            return pose

        def _compute_move_group_fk_position_error(self, full_qpos, _planner_target_pose):
            full_qpos = np.asarray(full_qpos, dtype=np.float32).reshape(-1)
            if np.allclose(full_qpos[:6], self.branch_debug["last_qpos"]):
                return 0.0123
            if np.allclose(full_qpos[:6], self.current_full_qpos[:6]):
                return 0.4567
            raise AssertionError(f"unexpected qpos for FK error: {full_qpos}")

        def close(self):
            return None

    monkeypatch.setattr(
        uut,
        "_score_planner_pregrasp_preview_candidate",
        lambda *_args, **_kwargs: (
            True,
            (0, 0, 0, 0.0, 0.0),
            {
                "wrap_count": 0,
                "large_delta_count": 0,
                "max_abs_raw_delta": 0.0,
                "preview_fk_pos_err": 0.0,
            },
        ),
    )
    monkeypatch.setattr(uut, "_planner_branch_guard_accepts", lambda *_args, **_kwargs: True)

    result = uut.run_planner_object_pregrasp_probe(
        env,
        method="rrtconnect",
        execute=True,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Solver,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_proxy_then_planner_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        build_object_pregrasp_target=lambda _env, extra_clearance, radial_backoff_override=None: (
            "green_cube_ext",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            target_pose,
        ),
        run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: True,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=lambda solver: solver,
        get_planner_recording_kwargs=lambda _env: {},
        get_object_specific_planner_profile=lambda _cfg, _obj: None,
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda _pose: (
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        get_debug_planner_ee_pose=lambda _env: _pose(),
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
    )

    assert result is False
    captured = capsys.readouterr().out
    assert "Pregrasp execution consistency method=rrtconnect" in captured
    assert "planned_last_arm_qpos=[ 1." in captured
    assert "actual_arm_qpos=[0.1 0.2 0.3 0.4 0.5 0.6]" in captured
    assert "planned_fk_pos_err=0.0123 m" in captured
    assert "actual_fk_pos_err=0.4567 m" in captured
    assert "follow_path tail step=11" in captured
    assert "arm_tracking_err=[-0.9 -0.9 -0.9 -0.9 -0.9 -0.9]" in captured
    assert "Pregrasp arm_controller: cls=SimpleNamespace frame=base use_target=True" in captured
    assert "target_qpos=[1.  1.1 1.2 1.3 1.4 1.5]" in captured
    assert "Pregrasp contact_diag: object_id=green_cube_ext" in captured
    assert "thumb_peak=0.6000 active=True" in captured
    assert "index_peak=0.1000 active=False" in captured
    assert "middle_peak=0.8000 active=True" in captured


def test_collect_runtime_rc5_contact_diagnostics_accepts_torch_like_forces():
    class _FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array.copy()

    target_object = types.SimpleNamespace()
    env_unwrapped = types.SimpleNamespace(
        manip_object_id="green_cube_ext",
        object_actors={"green_cube_ext": target_object},
        agent=types.SimpleNamespace(
            thumb_tip_link=types.SimpleNamespace(name="thumb"),
            index_tip_link=types.SimpleNamespace(name="index"),
            middle_tip_link=types.SimpleNamespace(name="middle"),
            scene=types.SimpleNamespace(
                get_pairwise_contact_forces=lambda link, actor: {
                    "thumb": _FakeTensor([[0.6, 0.0, 0.0]]),
                    "index": _FakeTensor([[0.1, 0.0, 0.0]]),
                    "middle": _FakeTensor([[0.8, 0.0, 0.0]]),
                }[link.name]
            ),
        ),
    )

    diagnostics = uut._collect_runtime_rc5_contact_diagnostics(env_unwrapped, min_force=0.5)

    assert diagnostics["available"] is True
    assert diagnostics["object_id"] == "green_cube_ext"
    assert diagnostics["thumb_active"] is True
    assert diagnostics["index_active"] is False
    assert diagnostics["middle_active"] is True


def test_pregrasp_execute_honors_configured_refine_steps(monkeypatch):
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="rc5_aero_hand_openr2s"),
            control_mode="pd_joint_pos",
            _debug_planner_config={"planner_pregrasp_refine_steps": 7},
        )
    )
    target_pose = _pose()
    holder = {}

    class _Solver:
        def __init__(self, *_args, **_kwargs):
            self.branch_debug = {
                "last_qpos": np.zeros(6, dtype=np.float32),
                "start_to_last": {
                    "suspicious_indices": np.array([], dtype=np.int32),
                    "raw_delta": np.zeros(6, dtype=np.float32),
                },
            }
            self.follow_refine_steps = None

        def move_to_pose_with_RRTConnect(self, _target_pose, dry_run=False):
            assert dry_run is True
            return {"status": "Success", "position": np.zeros((2, 6), dtype=np.float32)}

        def follow_path(self, result, refine_steps=0):
            self.follow_refine_steps = int(refine_steps)
            return ("obs", 0.0, False, False, {})

        def get_last_plan_branch_debug(self):
            return self.branch_debug

        def close(self):
            return None

    monkeypatch.setattr(
        uut,
        "_score_planner_pregrasp_preview_candidate",
        lambda *_args, **_kwargs: (
            True,
            (0, 0, 0, 0.0, 0.0),
            {
                "wrap_count": 0,
                "large_delta_count": 0,
                "max_abs_raw_delta": 0.0,
                "preview_fk_pos_err": 0.0,
            },
        ),
    )
    monkeypatch.setattr(uut, "_planner_branch_guard_accepts", lambda *_args, **_kwargs: True)

    def _configure_solver(solver):
        holder["solver"] = solver
        return solver

    result = uut.run_planner_object_pregrasp_probe(
        env,
        method="rrtconnect",
        execute=True,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Solver,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_proxy_then_planner_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        build_object_pregrasp_target=lambda _env, extra_clearance, radial_backoff_override=None: (
            "green_cube_ext",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            target_pose,
        ),
        run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: True,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=_configure_solver,
        get_planner_recording_kwargs=lambda _env: {},
        get_object_specific_planner_profile=lambda _cfg, _obj: None,
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32),
            np.asarray(pose.q, dtype=np.float32),
        ),
        get_debug_planner_ee_pose=lambda _env: target_pose,
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
    )

    assert result is True
    assert holder["solver"].follow_refine_steps == 7


def test_repeat_planner_result_waypoints_repeats_positions_and_velocity():
    result = {
        "status": "Success",
        "position": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "velocity": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }

    expanded = uut._repeat_planner_result_waypoints(result, repeat_each_step=3)

    assert expanded["position"].shape == (6, 2)
    assert expanded["velocity"].shape == (6, 2)
    assert np.allclose(
        expanded["position"],
        np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(
        expanded["velocity"],
        np.array(
            [
                [0.1, 0.2],
                [0.1, 0.2],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.3, 0.4],
                [0.3, 0.4],
            ],
            dtype=np.float32,
        ),
    )


def test_pregrasp_execute_honors_configured_waypoint_repeat(monkeypatch):
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            agent=types.SimpleNamespace(uid="rc5_aero_hand_openr2s"),
            control_mode="pd_joint_pos",
            _debug_planner_config={"planner_pregrasp_waypoint_repeat": 3},
        )
    )
    target_pose = _pose()
    holder = {}

    class _Solver:
        def __init__(self, *_args, **_kwargs):
            self.branch_debug = {
                "last_qpos": np.zeros(6, dtype=np.float32),
                "start_to_last": {
                    "suspicious_indices": np.array([], dtype=np.int32),
                    "raw_delta": np.zeros(6, dtype=np.float32),
                },
            }
            self.follow_position_shape = None

        def move_to_pose_with_RRTConnect(self, _target_pose, dry_run=False):
            assert dry_run is True
            return {
                "status": "Success",
                "position": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }

        def follow_path(self, result, refine_steps=0):
            self.follow_position_shape = np.asarray(result["position"]).shape
            return ("obs", 0.0, False, False, {})

        def get_last_plan_branch_debug(self):
            return self.branch_debug

        def close(self):
            return None

    monkeypatch.setattr(
        uut,
        "_score_planner_pregrasp_preview_candidate",
        lambda *_args, **_kwargs: (
            True,
            (0, 0, 0, 0.0, 0.0),
            {
                "wrap_count": 0,
                "large_delta_count": 0,
                "max_abs_raw_delta": 0.0,
                "preview_fk_pos_err": 0.0,
            },
        ),
    )
    monkeypatch.setattr(uut, "_planner_branch_guard_accepts", lambda *_args, **_kwargs: True)

    def _configure_solver(solver):
        holder["solver"] = solver
        return solver

    result = uut.run_planner_object_pregrasp_probe(
        env,
        method="rrtconnect",
        execute=True,
        backend="planner",
        extract_planner_base_pose=lambda _env: "base_pose",
        resolve_planner_debug_solver_class=lambda _uid: _Solver,
        is_proxy_ee_delta_backend=lambda _backend: False,
        is_proxy_then_planner_backend=lambda _backend: False,
        is_ee_delta_control_mode=lambda _mode: False,
        maybe_seed_proxy_start_pose=lambda *_args, **_kwargs: True,
        build_object_pregrasp_target=lambda _env, extra_clearance, radial_backoff_override=None: (
            "green_cube_ext",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            target_pose,
        ),
        run_proxy_full_approach_to_pregrasp=lambda *_args, **_kwargs: True,
        planner_visuals_supported=lambda _env: False,
        configure_debug_planner_solver_runtime=_configure_solver,
        get_planner_recording_kwargs=lambda _env: {},
        get_object_specific_planner_profile=lambda _cfg, _obj: None,
        refresh_render_state=lambda _env: None,
        pose_to_numpy=lambda pose: (
            np.asarray(pose.p, dtype=np.float32),
            np.asarray(pose.q, dtype=np.float32),
        ),
        get_debug_planner_ee_pose=lambda _env: target_pose,
        set_debug_planner_last_task_pose=lambda *_args, **_kwargs: None,
    )

    assert result is True
    assert holder["solver"].follow_position_shape == (6, 2)


def test_branch_guard_rejects_terminal_qpos_too_close_to_joint_limit():
    class _Joint:
        def __init__(self, lower, upper):
            self._limits = np.array([[lower, upper]], dtype=np.float32)

        def get_limits(self):
            return self._limits

    class RC5ArmMotionPlanningSolver:
        def __init__(self):
            self.robot = types.SimpleNamespace(
                get_active_joints=lambda: [
                    _Joint(-3.14, 3.14),
                    _Joint(-3.14, 3.14),
                    _Joint(-3.14, 3.14),
                    _Joint(-3.14, 3.14),
                    _Joint(-3.14, 3.14),
                    _Joint(-3.14, 3.14),
                ]
            )
            self.env_agent = types.SimpleNamespace(
                arm_joint_names=["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
            )

        def _get_move_group(self):
            return "right_tcp_link"

        def get_last_plan_branch_debug(self):
            return {
                "last_qpos": np.array([-3.1293, 0.6766, -1.7396, -1.3143, -0.1444, 2.5791], dtype=np.float32),
                "start_to_last": {
                    "suspicious_indices": np.array([], dtype=np.int32),
                    "raw_delta": np.zeros(6, dtype=np.float32),
                    "compare_joint_names": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"],
                },
            }
    solver = RC5ArmMotionPlanningSolver()

    planner_cfg = {
        "planner_branch_guard_enabled": True,
        "planner_branch_guard_joint_limit_margin_rad": 0.05,
    }

    accepted = uut._planner_branch_guard_accepts(
        solver,
        planner_cfg,
        stage_label="PregraspPreview",
        method_name="rrtconnect",
    )

    assert accepted is False


def test_branch_guard_ignores_nonfinite_wraparound_joint_limits():
    class _Joint:
        def get_limits(self):
            return np.array([[np.NINF, np.PINF]], dtype=np.float32)

    class RC5ArmMotionPlanningSolver:
        def __init__(self):
            self.robot = types.SimpleNamespace(
                get_active_joints=lambda: [
                    _Joint(),
                    _Joint(),
                    _Joint(),
                    _Joint(),
                    _Joint(),
                    _Joint(),
                ]
            )
            self.env_agent = types.SimpleNamespace(
                arm_joint_names=["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
            )

        def _get_move_group(self):
            return "right_tcp_link"

        def get_last_plan_branch_debug(self):
            return {
                "last_qpos": np.array([-3.1293, 0.6766, -1.7396, -1.3143, -0.1444, 2.5791], dtype=np.float32),
                "start_to_last": {
                    "suspicious_indices": np.array([], dtype=np.int32),
                    "raw_delta": np.zeros(6, dtype=np.float32),
                    "compare_joint_names": ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"],
                },
            }

    solver = RC5ArmMotionPlanningSolver()
    planner_cfg = {
        "planner_branch_guard_enabled": True,
        "planner_branch_guard_joint_limit_margin_rad": 0.05,
    }

    accepted = uut._planner_branch_guard_accepts(
        solver,
        planner_cfg,
        stage_label="PregraspPreview",
        method_name="rrtconnect",
    )

    assert accepted is True
