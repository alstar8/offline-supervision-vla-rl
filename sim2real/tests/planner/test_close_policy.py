from openreal2sim.simulation.maniskill.planner_core.close_policy import (
    default_close_steps_for_agent,
)


def test_default_close_steps_for_rc5():
    assert default_close_steps_for_agent("rc5_aero_hand_openr2s") == 20
    assert default_close_steps_for_agent("rc5_aero_hand_openr2s_rl") == 20


def test_default_close_steps_for_widowx_and_generic_agents():
    assert default_close_steps_for_agent("widowx250s_bridgedataset_flat_table_openr2s") == 6
    assert default_close_steps_for_agent("panda") == 6
