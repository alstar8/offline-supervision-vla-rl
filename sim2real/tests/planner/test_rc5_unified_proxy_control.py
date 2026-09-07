from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_uut():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "openreal2sim"
        / "simulation"
        / "maniskill"
        / "scripts"
        / "rc5_unified_proxy_control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_rc5_unified_proxy_control_uut",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    fake_openreal2sim = types.ModuleType("openreal2sim")
    fake_simulation = types.ModuleType("openreal2sim.simulation")
    fake_maniskill = types.ModuleType("openreal2sim.simulation.maniskill")
    fake_planner_core = types.ModuleType("openreal2sim.simulation.maniskill.planner_core")
    fake_planner_core.default_close_steps_for_agent = lambda _agent_uid: 3

    sys.modules.setdefault("openreal2sim", fake_openreal2sim)
    sys.modules.setdefault("openreal2sim.simulation", fake_simulation)
    sys.modules.setdefault("openreal2sim.simulation.maniskill", fake_maniskill)
    sys.modules["openreal2sim.simulation.maniskill.planner_core"] = fake_planner_core

    spec.loader.exec_module(module)
    return module


def test_resolve_proxy_only_macro_backend_defaults_to_proxy():
    uut = _load_uut()

    resolved = uut.resolve_proxy_only_macro_backend({}, default_backend="proxy_ee_delta")

    assert resolved == "proxy_ee_delta"


def test_resolve_proxy_only_macro_backend_rejects_real_planner_backend():
    uut = _load_uut()

    try:
        uut.resolve_proxy_only_macro_backend(
            {"planner_backend": "local_ik"},
            default_backend="proxy_ee_delta",
        )
    except ValueError as exc:
        message = str(exc)
        assert "Proxy-only macro backend contract" in message
        assert "planner_backend='local_ik'" in message
        assert "proxy_ee_delta" in message
    else:
        raise AssertionError("Expected real planner / mplib planner backend to be rejected")
