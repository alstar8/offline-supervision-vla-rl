from __future__ import annotations

import textwrap

import pytest

import openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_setup as uut


class _Args:
    def __init__(self, **overrides):
        self.__dict__.update(overrides)

    def __getattr__(self, _name):
        return None


def _startup_profile_config(*, object_id="spray_bottle_ext"):
    top = [float(index) for index in range(22)]
    side = top.copy()
    side[5] = 1.570796
    return {
        "robot_init_qpos": top.copy(),
        "robot_init_qpos_profiles": {"top": top, "side": side},
        "robot_init_qpos_profile_by_object": {
            "plastic_cup_ext": "top",
            "spray_bottle_ext": "side",
        },
        "manip_object_id": object_id,
    }


def test_object_specific_startup_selects_named_profile():
    config = _startup_profile_config()

    uut._apply_object_specific_robot_init_qpos_profile(config, cli_override=False)

    assert config["robot_init_qpos"][5] == pytest.approx(1.570796)
    assert config["resolved_robot_init_qpos_profile"] == "side"


def test_object_specific_startup_requires_selected_object():
    config = _startup_profile_config(object_id=None)

    with pytest.raises(RuntimeError, match="manip_object_id is unset"):
        uut._apply_object_specific_robot_init_qpos_profile(config, cli_override=False)


def test_object_specific_startup_rejects_missing_object_mapping():
    config = _startup_profile_config(object_id="unknown_ext")

    with pytest.raises(RuntimeError, match="has no startup profile mapping"):
        uut._apply_object_specific_robot_init_qpos_profile(config, cli_override=False)


def test_object_specific_startup_rejects_unknown_profile():
    config = _startup_profile_config()
    config["robot_init_qpos_profile_by_object"]["spray_bottle_ext"] = "missing"

    with pytest.raises(RuntimeError, match="references unknown robot_init_qpos profile"):
        uut._apply_object_specific_robot_init_qpos_profile(config, cli_override=False)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ([0.0] * 21, "exactly 22 values"),
        ([0.0] * 21 + [float("nan")], "non-finite"),
        ([0.0] * 21 + ["invalid"], "numeric values"),
    ],
)
def test_object_specific_startup_validates_profile_values(profile, message):
    config = _startup_profile_config()
    config["robot_init_qpos_profiles"]["side"] = profile

    with pytest.raises((RuntimeError, ValueError), match=message):
        uut._apply_object_specific_robot_init_qpos_profile(config, cli_override=False)


def test_explicit_cli_qpos_overrides_object_startup_profile():
    config = _startup_profile_config()
    explicit_qpos = [-0.25] * 22
    config["robot_init_qpos"] = explicit_qpos.copy()

    uut._apply_object_specific_robot_init_qpos_profile(config, cli_override=True)

    assert config["robot_init_qpos"] == explicit_qpos
    assert config["resolved_robot_init_qpos_profile"] == "cli_override"


def test_load_runner_config_selects_object_startup_before_environment_creation(tmp_path):
    config_path = tmp_path / "config.yaml"
    top = [0.0] * 22
    side = top.copy()
    side[5] = 1.570796
    config_path.write_text(
        textwrap.dedent(
            f"""
            keys: [demo]
            global:
              simulation: {{}}
            local:
              demo:
                simulation:
                  manip_object_id: plastic_cup_ext
                  robot_init_qpos: {top}
                  robot_init_qpos_profiles:
                    top: {top}
                    side: {side}
                  robot_init_qpos_profile_by_object:
                    plastic_cup_ext: top
                    spray_bottle_ext: side
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = _Args(
        config_path=str(config_path),
        scene="assets/scenes/demo/simulation/scene.json",
        key="demo",
        task_object_id="spray_bottle_ext",
        no_auto_placement=False,
    )

    config = uut.load_runner_config(args)

    assert config["manip_object_id"] == "spray_bottle_ext"
    assert config["resolved_robot_init_qpos_profile"] == "side"
    assert config["robot_init_qpos"] == pytest.approx(side)
