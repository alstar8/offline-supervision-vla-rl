from __future__ import annotations

from typing import Iterable

import torch


def _contact_flag(forces, min_force: float):
    return torch.linalg.norm(forces, axis=1) >= float(min_force)


def _chain_contact_flag(link_forces: Iterable, min_force: float):
    chain_flag = None
    for forces in link_forces:
        link_flag = _contact_flag(forces, min_force)
        if chain_flag is None:
            chain_flag = torch.zeros_like(link_flag, dtype=torch.bool)
        chain_flag = torch.logical_or(chain_flag, link_flag)
    if chain_flag is None:
        raise ValueError("RC5 grasp heuristic requires at least one link force tensor per chain.")
    return chain_flag


def rc5_tip_contact_is_grasping(
    *,
    thumb_forces,
    finger_forces: Iterable,
    min_force: float = 0.5,
):
    thumb_flag = _contact_flag(thumb_forces, min_force)
    any_finger_flag = torch.zeros_like(thumb_flag, dtype=torch.bool)
    for forces in finger_forces:
        any_finger_flag = torch.logical_or(any_finger_flag, _contact_flag(forces, min_force))
    return torch.logical_and(thumb_flag, any_finger_flag)


def rc5_finger_chain_contact_is_grasping(
    *,
    thumb_chain_forces: Iterable,
    finger_chain_forces: Iterable[Iterable],
    min_force: float = 0.5,
):
    thumb_flag = _chain_contact_flag(thumb_chain_forces, min_force)
    any_finger_flag = torch.zeros_like(thumb_flag, dtype=torch.bool)
    for chain_forces in finger_chain_forces:
        any_finger_flag = torch.logical_or(
            any_finger_flag,
            _chain_contact_flag(chain_forces, min_force),
        )
    return torch.logical_and(thumb_flag, any_finger_flag)
