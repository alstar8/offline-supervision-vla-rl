from __future__ import annotations

import numpy as np
import pytest

from openreal2sim.simulation.maniskill.scripts import rc5_unified_proxy_targets as uut


def test_current_tcp_orientation_mode_uses_runtime_tcp_quaternion():
    current_q = np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float32)

    target_q, mode = uut.resolve_object_target_quaternion(
        {"target_orientation_mode": "current_tcp"},
        current_q,
        object_id="spray_bottle_ext",
    )

    assert mode == "current_tcp"
    assert target_q == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_current_tcp_orientation_mode_rejects_ambiguous_fixed_quaternion():
    with pytest.raises(RuntimeError, match="must not define target_quat"):
        uut.resolve_object_target_quaternion(
            {
                "target_orientation_mode": "current_tcp",
                "target_quat": [1.0, 0.0, 0.0, 0.0],
            },
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            object_id="spray_bottle_ext",
        )


def test_fixed_orientation_mode_requires_explicit_quaternion():
    with pytest.raises(RuntimeError, match="requires target_quat"):
        uut.resolve_object_target_quaternion(
            {"target_orientation_mode": "fixed"},
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            object_id="cube_ext",
        )


def test_orientation_mode_rejects_unknown_value():
    with pytest.raises(RuntimeError, match="must be one of"):
        uut.resolve_object_target_quaternion(
            {"target_orientation_mode": "guess"},
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            object_id="cube_ext",
        )
