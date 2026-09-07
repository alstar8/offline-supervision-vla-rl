from __future__ import annotations

import types

import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_macro as uut


def _make_env():
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(agent=types.SimpleNamespace(uid="fake_uid")))


def test_proxy_pick_macro_aborts_on_full_approach_failure():
    observed = {"feedback": None}

    result = uut.run_proxy_pick_macro(
        _make_env(),
        {},
        run_proxy_pregrasp_probe=lambda *_args, **_kwargs: True,
        run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: False,
        run_proxy_descend=lambda *_args, **_kwargs: True,
        run_proxy_close_gripper=lambda *_args, **_kwargs: True,
        run_proxy_lift=lambda *_args, **_kwargs: True,
        set_unified_macro_feedback=lambda **kwargs: observed.update(feedback=kwargs),
    )

    assert result is False
    assert observed["feedback"] == {
        "semantic_task_success": False,
        "failed_stage": "full_approach",
    }


def test_proxy_pick_macro_rejects_real_planner_backend():
    try:
        uut.run_proxy_pick_macro(
            _make_env(),
            {"planner_backend": "local_ik"},
            run_proxy_pregrasp_probe=lambda *_args, **_kwargs: True,
            run_proxy_full_approach_to_descend=lambda *_args, **_kwargs: True,
            run_proxy_descend=lambda *_args, **_kwargs: True,
            run_proxy_close_gripper=lambda *_args, **_kwargs: True,
            run_proxy_lift=lambda *_args, **_kwargs: True,
            set_unified_macro_feedback=lambda **_kwargs: None,
        )
    except ValueError as exc:
        message = str(exc)
        assert "Proxy-only macro backend contract" in message
        assert "planner_backend='local_ik'" in message
    else:
        raise AssertionError("Expected proxy macro to reject real planner / mplib planner backends")


def test_proxy_pick_macro_keeps_legacy_planner_callback_aliases_compatible():
    observed = {"feedback": None}

    result = uut.run_proxy_pick_macro(
        _make_env(),
        {},
        run_planner_object_pregrasp_probe=lambda *_args, **_kwargs: True,
        run_planner_full_approach_to_descend=lambda *_args, **_kwargs: True,
        run_planner_object_descend=lambda *_args, **_kwargs: True,
        run_planner_close_gripper=lambda *_args, **_kwargs: True,
        run_planner_lift=lambda *_args, **_kwargs: True,
        set_unified_macro_feedback=lambda **kwargs: observed.update(feedback=kwargs),
    )

    assert result is True
    assert observed["feedback"] == {
        "semantic_task_success": True,
        "failed_stage": None,
    }


def test_full_pick_macro_uses_planner_close_backend():
    observed = {"close_backend": None, "feedback": None}

    result = uut.run_planner_pick_macro_full(
        _make_env(),
        {},
        run_planner_object_pregrasp_probe=lambda *_args, **_kwargs: True,
        run_planner_full_approach_to_descend=lambda *_args, **_kwargs: True,
        run_planner_object_descend=lambda *_args, **_kwargs: True,
        run_planner_close_gripper=lambda _env, close_steps, backend: observed.update(close_backend=backend) or True,
        run_planner_lift=lambda *_args, **_kwargs: True,
        set_unified_macro_feedback=lambda **kwargs: observed.update(feedback=kwargs),
    )

    assert result is True
    assert observed["close_backend"] == "planner"
    assert observed["feedback"] == {
        "semantic_task_success": True,
        "failed_stage": None,
    }
