"""Shared close-gripper policy across planner entrypoints."""

from __future__ import annotations


def default_close_steps_for_agent(agent_uid: str) -> int:
    return 20 if str(agent_uid).startswith("rc5_aero_hand_openr2s") else 6
