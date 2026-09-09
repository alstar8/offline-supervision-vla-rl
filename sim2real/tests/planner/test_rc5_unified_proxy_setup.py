from __future__ import annotations

import json
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


def test_load_runner_config_picks_rl_env_flags_from_simulation_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            keys: [demo]
            global:
              simulation: {}
            local:
              demo:
                simulation:
                  manip_object_id: orange_cube_ext
                  task_description: Pick red cube
                  use_wrist_camera: true
                  use_360_background: true
                  pano_sphere_radius: 20
                  placement_mode: random
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    args = _Args(
        config_path=str(config_path),
        scene="assets/scenes/demo/simulation/scene.json",
        key="demo",
        no_auto_placement=False,
    )

    config = uut.load_runner_config(args)

    assert config["task_description"] == "Pick red cube"
    assert config["use_wrist_camera"] is True
    assert config["use_360_background"] is True
    assert config["pano_sphere_radius"] == 20
    assert config["placement_mode"] == "random"


def test_build_reset_options_from_runtime_request(tmp_path):
    request_path = tmp_path / "runtime_request.json"
    request_path.write_text(
        '{"episode_index": 7, "placement_seed": 42}',
        encoding="utf-8",
    )
    args = _Args(runtime_request_path=str(request_path))

    assert uut.build_reset_options_from_args(args) == {"episode_id": 42}


def test_build_reset_options_from_per_env_runtime_requests(tmp_path):
    paths = []
    for seed in (10, 11):
        path = tmp_path / f"runtime_request_{seed}.json"
        path.write_text(
            f'{{"episode_index": {seed}, "placement_seed": {seed}}}',
            encoding="utf-8",
        )
        paths.append(str(path))
    args = _Args(runtime_request_path_per_env_json=json.dumps(paths))

    assert uut.build_reset_options_from_args(args) == {"episode_id": [10, 11]}
