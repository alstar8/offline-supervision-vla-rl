from types import SimpleNamespace

import numpy as np

from openreal2sim.simulation.maniskill.planner_core.grasp_state import (
    GraspState,
    get_planner_grasp_state,
    infer_gripper_state_from_hand_target,
    restore_planner_grasp_state,
    save_planner_grasp_state,
)


def test_restore_grasp_state_seeds_defaults():
    env = SimpleNamespace()

    state = restore_planner_grasp_state(
        env,
        default_target_hand_qpos=[0.037, 0.037],
        default_gripper_state="open",
        source_stage="init",
    )

    assert isinstance(state, GraspState)
    assert np.allclose(state.target_hand_qpos, [0.037, 0.037])
    assert state.gripper_state == "open"
    assert state.source_stage == "init"


def test_save_grasp_state_preserves_unspecified_fields():
    env = SimpleNamespace()

    save_planner_grasp_state(
        env,
        target_hand_qpos=[0.015, 0.015],
        gripper_state="closed",
        source_stage="close",
    )
    state = save_planner_grasp_state(
        env,
        realized_hand_qpos=[0.0259, 0.0326],
        grasp_flag=True,
        object_id="orange_cube_ext",
    )

    assert np.allclose(state.target_hand_qpos, [0.015, 0.015])
    assert np.allclose(state.realized_hand_qpos, [0.0259, 0.0326])
    assert state.gripper_state == "closed"
    assert state.grasp_flag is True
    assert state.grasp_flag_rows is None
    assert state.object_id == "orange_cube_ext"
    assert state.source_stage == "close"


def test_save_grasp_state_normalizes_per_env_grasp_flag_rows():
    env = SimpleNamespace()

    state = save_planner_grasp_state(
        env,
        grasp_flag=True,
        grasp_flag_rows=[True, False, True],
    )

    assert state.grasp_flag is True
    assert state.grasp_flag_rows.dtype == np.bool_
    assert state.grasp_flag_rows.tolist() == [True, False, True]


def test_get_grasp_state_returns_none_without_explicit_state():
    env = SimpleNamespace()

    state = get_planner_grasp_state(env)

    assert state is None


def test_infer_gripper_state_from_hand_target_chooses_nearest_reference():
    closed = infer_gripper_state_from_hand_target(
        [0.0155, 0.0152],
        open_hand_qpos=[0.037, 0.037],
        closed_hand_qpos=[0.015, 0.015],
        open_state="open",
        closed_state="closed",
    )
    opened = infer_gripper_state_from_hand_target(
        [0.0365, 0.0368],
        open_hand_qpos=[0.037, 0.037],
        closed_hand_qpos=[0.015, 0.015],
        open_state="open",
        closed_state="closed",
    )

    assert closed == "closed"
    assert opened == "open"
