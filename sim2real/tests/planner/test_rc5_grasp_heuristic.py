from __future__ import annotations

import torch

from openreal2sim.simulation.maniskill.rc5_grasp_heuristic import (
    rc5_finger_chain_contact_is_grasping,
    rc5_tip_contact_is_grasping,
)


def _forces(magnitude: float):
    return torch.tensor([[float(magnitude), 0.0, 0.0]], dtype=torch.float32)


def test_rc5_tip_contact_is_grasping_accepts_thumb_and_ring_contact():
    result = rc5_tip_contact_is_grasping(
        thumb_forces=_forces(0.8),
        finger_forces=[_forces(0.0), _forces(0.0), _forces(0.9), _forces(0.0)],
        min_force=0.5,
    )

    assert result.tolist() == [True]


def test_rc5_tip_contact_is_grasping_rejects_thumb_without_any_finger():
    result = rc5_tip_contact_is_grasping(
        thumb_forces=_forces(0.8),
        finger_forces=[_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.0)],
        min_force=0.5,
    )

    assert result.tolist() == [False]


def test_rc5_finger_chain_contact_is_grasping_accepts_thumb_distal_plus_middle_chain():
    result = rc5_finger_chain_contact_is_grasping(
        thumb_chain_forces=[_forces(0.0), _forces(0.9), _forces(0.0)],
        finger_chain_forces=[
            [_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.0)],
            [_forces(0.0), _forces(0.0), _forces(0.7), _forces(0.0)],
            [_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.0)],
            [_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.0)],
        ],
        min_force=0.5,
    )

    assert result.tolist() == [True]


def test_rc5_finger_chain_contact_is_grasping_rejects_without_thumb_chain_contact():
    result = rc5_finger_chain_contact_is_grasping(
        thumb_chain_forces=[_forces(0.0), _forces(0.0), _forces(0.0)],
        finger_chain_forces=[
            [_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.9)],
            [_forces(0.0), _forces(0.0), _forces(0.7), _forces(0.0)],
            [_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.0)],
            [_forces(0.0), _forces(0.0), _forces(0.0), _forces(0.0)],
        ],
        min_force=0.5,
    )

    assert result.tolist() == [False]
