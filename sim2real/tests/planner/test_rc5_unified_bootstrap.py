from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from openreal2sim.simulation.maniskill.scripts.rc5_unified_bootstrap import (
    apply_lighting_profile_overrides,
    apply_hand_contact_profile_overrides,
    apply_hand_controller_profile_overrides,
    apply_teleop_profile_overrides,
    apply_hand_pose_config_to_agent,
    collect_cli_bootstrap_overrides,
    detect_config_key,
    extract_flag_values,
    extract_rc5_bootstrap_config,
    has_flag,
    load_hand_contact_config,
    load_hand_controller_config,
    load_lighting_profile_config,
    load_teleop_profile_config,
    load_simulation_config_sections,
    pick_simulation_value,
    reset_planner_hand_target_to_open,
    resolve_hand_contact_profile,
    resolve_hand_controller_profile,
    resolve_lighting_profile,
    resolve_rc5_move_group,
    resolve_teleop_profile,
    resolve_unified_bootstrap_request,
    validate_rc5_bootstrap_for_backend,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_config(
    tmp_path: Path,
    *,
    key: str = "demo_key",
    global_sim: dict | None = None,
    local_sim: dict | None = None,
    keys: list[str] | None = None,
) -> Path:
    data = {
        "keys": keys if keys is not None else [key],
        "global": {"simulation": global_sim or {}},
        "local": {
            key: {
                "simulation": local_sim or {},
            }
        },
    }
    path = tmp_path / "rc5_bootstrap_test.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_rc5_unified_bootstrap_reads_real_config_debug_key():
    cfg_path = _repo_root() / "config" / "config_debug.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    local_sim = cfg["local"]["airi_table_new_empty3_image"]["simulation"]

    bootstrap = extract_rc5_bootstrap_config(cfg_path, "airi_table_new_empty3_image")

    assert bootstrap.key == "airi_table_new_empty3_image"
    assert bootstrap.robot_uids in {"rc5_aero_hand_openr2s", "rc5_aero_hand_openr2s_rl"}
    assert bootstrap.hand_contact_profile == local_sim["hand_contact_profile"]
    assert bootstrap.hand_controller_profile == local_sim["hand_controller_profile"]


def test_real_config_rc5_object_calibrations_include_canonical_right_tcp_profile():
    cfg_path = _repo_root() / "config" / "config_debug.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    calibrations = data["local"]["airi_table_new_empty3_image"]["simulation"]["planner_object_calibrations"]

    expected = {
        "pregrasp_offset_xyz": [-0.0005, 0.0459, 0.0886],
        "descend_offset_xyz": [-0.0005, 0.0459, 0.0786],
        "target_quat": [-0.5161, -0.6785, 0.4517, 0.2630],
    }
    for object_id in (
        "green_cube_ext",
        "yellow_cube_ext",
        "blue_cube_ext",
        "white_cube_ext",
        "banana_ext",
    ):
        assert calibrations[object_id] == expected


def test_real_config_selects_side_startup_only_for_spray_bottle():
    cfg_path = _repo_root() / "config" / "config_debug.yaml"
    simulation = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["local"][
        "airi_table_new_empty3_image"
    ]["simulation"]

    profiles = simulation["robot_init_qpos_profiles"]
    top = profiles["top"]
    side = profiles["side"]
    mapping = simulation["robot_init_qpos_profile_by_object"]

    assert len(top) == len(side) == 22
    assert side[:5] == top[:5]
    assert top[5] == pytest.approx(-0.05)
    assert side[5] == pytest.approx(np.deg2rad(85.0), abs=1e-6)
    assert mapping["spray_bottle_ext"] == "side"
    assert mapping["plastic_cup_ext"] == "top"
    assert set(mapping) == set(simulation["planner_object_calibrations"])
    assert all(
        profile_name == "top"
        for object_id, profile_name in mapping.items()
        if object_id != "spray_bottle_ext"
    )

    bottle_profile = simulation["planner_object_calibrations"]["spray_bottle_ext"]
    assert bottle_profile["target_orientation_mode"] == "current_tcp"
    assert "target_quat" not in bottle_profile
    assert bottle_profile["pregrasp_offset_xyz"] == [-0.035, 0.07, 0.0886]
    assert bottle_profile["descend_offset_xyz"] == [-0.035, 0.07, -0.01]

    placements = simulation["object_placements"]
    assert placements["spray_bottle_ext"]["collision_mesh_path"].endswith("visual.glb.coacd.ply")
    assert placements["plastic_cup_ext"]["collision_mesh_path"].endswith("visual.glb.coacd.ply")


def test_rc5_unified_bootstrap_local_overrides_global_values(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        global_sim={
            "robot_uids": "global_robot",
            "hand_pose_config": "global_hand.yaml",
            "teleop_profile_config": "global_teleop.yaml",
            "teleop_profile": "global_profile",
        },
        local_sim={
            "robot_uids": "local_robot",
            "hand_pose_config": "local_hand.yaml",
        },
    )

    bootstrap = extract_rc5_bootstrap_config(cfg_path, "demo_key")

    assert bootstrap.robot_uids == "local_robot"
    assert bootstrap.hand_pose_config == "local_hand.yaml"
    assert bootstrap.teleop_profile_config == "global_teleop.yaml"
    assert bootstrap.teleop_profile == "global_profile"


def test_rc5_unified_bootstrap_cli_overrides_take_precedence(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        global_sim={"robot_uids": "global_robot"},
        local_sim={"robot_uids": "local_robot"},
    )

    bootstrap = extract_rc5_bootstrap_config(
        cfg_path,
        "demo_key",
        cli_overrides={"robot_uids": "cli_robot"},
    )

    assert bootstrap.robot_uids == "cli_robot"


def test_load_simulation_config_sections_reads_local_and_global_mappings(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        global_sim={"robot_uids": "global_robot", "planner_backend": "planner"},
        local_sim={"robot_uids": "local_robot"},
    )

    sections = load_simulation_config_sections(cfg_path, "demo_key")

    assert sections.key == "demo_key"
    assert sections.config_path == cfg_path.resolve()
    assert sections.global_sim["robot_uids"] == "global_robot"
    assert sections.local_sim["robot_uids"] == "local_robot"
    assert sections.raw_config["keys"] == ["demo_key"]


def test_pick_simulation_value_respects_cli_local_global_default_precedence(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        global_sim={"robot_uids": "global_robot", "planner_backend": "planner"},
        local_sim={"robot_uids": "local_robot"},
    )
    sections = load_simulation_config_sections(cfg_path, "demo_key")

    assert pick_simulation_value(sections, "robot_uids", "fallback_robot") == "local_robot"
    assert pick_simulation_value(sections, "planner_backend", "fallback_backend") == "planner"
    assert pick_simulation_value(sections, "missing_value", "fallback_value") == "fallback_value"
    assert (
        pick_simulation_value(
            sections,
            "robot_uids",
            "fallback_robot",
            cli_value="cli_robot",
        )
        == "cli_robot"
    )


def test_detect_config_key_prefers_explicit_key_and_falls_back_to_outputs_path():
    assert detect_config_key("/tmp/assets/scenes/demo_key/simulation/scene.json", "explicit_key") == "explicit_key"
    assert detect_config_key("/tmp/assets/scenes/demo_key/simulation/scene.json", None) == "demo_key"
    assert detect_config_key("/tmp/no_outputs/scene.json", None) is None
    assert detect_config_key(None, None) is None


def test_cli_flag_helpers_extract_and_collect_bootstrap_overrides():
    argv = [
        "--scene",
        "/tmp/assets/scenes/demo_key/simulation/scene.json",
        "--robot_uids",
        "robot_a",
        "--lighting_profile_config",
        "lighting.yaml",
        "--lighting_profile",
        "shadow_a",
        "--teleop_profile_config",
        "teleop.yaml",
        "--teleop_profile",
        "profile_a",
    ]

    assert extract_flag_values(argv, "--scene") == ["/tmp/assets/scenes/demo_key/simulation/scene.json"]
    assert extract_flag_values(argv, "--missing") == []
    assert has_flag(argv, "--robot_uids") is True
    assert has_flag(argv, "--missing") is False
    assert collect_cli_bootstrap_overrides(argv) == {
        "robot_uids": "robot_a",
        "lighting_profile_config": "lighting.yaml",
        "lighting_profile": "shadow_a",
        "teleop_profile_config": "teleop.yaml",
        "teleop_profile": "profile_a",
    }


def test_resolve_unified_bootstrap_request_reads_key_scene_and_cli_overrides(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        local_sim={
            "robot_uids": "robot_from_config",
            "control_mode": "control_from_config",
            "teleop_profile_config": "teleop_from_config.yaml",
            "teleop_profile": "teleop_from_config",
        },
    )
    request = resolve_unified_bootstrap_request(
        [
            "--config_path",
            str(cfg_path),
            "--scene",
            "/tmp/assets/scenes/demo_key/simulation/scene.json",
            "--robot_uids",
            "robot_from_cli",
        ]
    )

    assert request.config_path == cfg_path.resolve()
    assert request.scene_path == "/tmp/assets/scenes/demo_key/simulation/scene.json"
    assert request.key == "demo_key"
    assert request.cli_bootstrap_overrides["robot_uids"] == "robot_from_cli"
    assert request.bootstrap.robot_uids == "robot_from_cli"
    assert request.bootstrap.control_mode == "control_from_config"


def test_load_and_resolve_teleop_profile_from_real_config():
    cfg_path = _repo_root() / "config" / "teleop_profiles.yaml"

    resolved_cfg_path, cfg = load_teleop_profile_config(cfg_path)
    assert resolved_cfg_path == cfg_path.resolve()
    assert "profiles" in cfg

    profile = resolve_teleop_profile(cfg_path, "rc5_teleop_v1")
    assert profile.profile_name == "rc5_teleop_v1"
    assert profile.sim_delta_remap_rpy_deg == [0.0, 0.0, 90.0]
    assert profile.gripper_open_signal == -1.0
    assert profile.gripper_close_signal == 1.0


def test_apply_teleop_profile_overrides_updates_config_mapping(tmp_path):
    cfg_path = tmp_path / "teleop.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "demo_profile": {
                        "sim_delta_remap_rpy_deg": [1, 2, 3],
                        "gripper_open_signal": -0.5,
                        "gripper_close_signal": 0.75,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config_overrides = {
        "teleop_profile_config": str(cfg_path),
        "teleop_profile": "demo_profile",
    }

    resolved = apply_teleop_profile_overrides(config_overrides, scope="TeleopTest")

    assert resolved is not None
    assert config_overrides["teleop_profile_config"] == str(cfg_path.resolve())
    assert config_overrides["teleop_profile"] == "demo_profile"
    assert config_overrides["sim_delta_remap_rpy_deg"] == [1.0, 2.0, 3.0]
    assert config_overrides["gripper_open_signal"] == -0.5
    assert config_overrides["gripper_close_signal"] == 0.75


def test_load_and_apply_lighting_profile_overrides(tmp_path):
    cfg_path = tmp_path / "lighting.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "shadow_demo": {
                        "ambient_light": [0.1, 0.1, 0.1],
                        "directional_lights": [
                            {"direction": [0.3, 0.2, -1.0], "color": [1.0, 1.0, 1.0], "shadow": True}
                        ],
                        "point_lights": [
                            {"position": [0.0, -1.0, 1.5], "color": [0.2, 0.2, 0.2], "shadow": False}
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resolved_cfg_path, cfg = load_lighting_profile_config(cfg_path)
    assert resolved_cfg_path == cfg_path.resolve()
    assert "profiles" in cfg

    resolved = resolve_lighting_profile(cfg_path, "shadow_demo")
    assert resolved.profile_name == "shadow_demo"
    assert resolved.profile["ambient_light"] == [0.1, 0.1, 0.1]

    config_overrides = {
        "lighting_profile_config": str(cfg_path),
        "lighting_profile": "shadow_demo",
        "lighting_config": {"ambient_light": [0.9, 0.9, 0.9]},
    }
    applied = apply_lighting_profile_overrides(config_overrides, scope="LightingTest")

    assert applied is not None
    assert config_overrides["lighting_profile_config"] == str(cfg_path.resolve())
    assert config_overrides["lighting_profile"] == "shadow_demo"
    assert config_overrides["lighting_config"]["ambient_light"] == [0.1, 0.1, 0.1]


def test_apply_teleop_profile_overrides_rejects_unpaired_fields():
    with pytest.raises(ValueError, match="teleop_profile_config was provided but teleop_profile is missing"):
        apply_teleop_profile_overrides({"teleop_profile_config": "config/teleop_profiles.yaml"})


def test_load_and_resolve_hand_contact_profile_from_real_config():
    cfg_path = _repo_root() / "config" / "hand_contact_profiles.yaml"

    resolved_cfg_path, cfg = load_hand_contact_config(cfg_path)
    assert resolved_cfg_path == cfg_path.resolve()
    assert "profiles" in cfg

    profile = resolve_hand_contact_profile(cfg_path, "rubber_fingertips_v1")
    assert profile.profile_name == "rubber_fingertips_v1"
    assert "fingertip" in profile.material_names
    assert "right_thumb_tip_link" in profile.link_names


def test_rubber_fingertips_v2_covers_full_finger_chains_for_side_grasps():
    cfg_path = _repo_root() / "config" / "hand_contact_profiles.yaml"

    profile = resolve_hand_contact_profile(cfg_path, "rubber_fingertips_v2")

    expected_links = {
        "right_thumb_proximal_link",
        "right_thumb_distal_link",
        "right_thumb_tip_link",
    }
    for finger_name in ("index", "middle", "ring", "pinky"):
        expected_links.update(
            {
                f"right_{finger_name}_proximal_link",
                f"right_{finger_name}_middle_link",
                f"right_{finger_name}_distal_link",
                f"right_{finger_name}_tip_link",
            }
        )

    assert set(profile.link_names) == expected_links


def test_apply_hand_contact_profile_overrides_updates_config_mapping(tmp_path):
    cfg_path = tmp_path / "hand_contact.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "demo_contact": {
                        "materials": {"tip": {"static_friction": 1.0}},
                        "links": {"thumb": {"material": "tip"}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config_overrides = {
        "hand_contact_config": str(cfg_path),
        "hand_contact_profile": "demo_contact",
    }

    resolved = apply_hand_contact_profile_overrides(config_overrides, scope="HandContactTest")

    assert resolved is not None
    assert config_overrides["hand_contact_config"] == str(cfg_path.resolve())
    assert config_overrides["hand_contact_profile"] == "demo_contact"
    assert config_overrides["hand_contact"]["materials"]["tip"]["static_friction"] == 1.0


def test_load_and_resolve_hand_controller_profile_from_real_config():
    cfg_path = _repo_root() / "config" / "hand_controller_profiles.yaml"

    resolved_cfg_path, cfg = load_hand_controller_config(cfg_path)
    assert resolved_cfg_path == cfg_path.resolve()
    assert "profiles" in cfg

    profile = resolve_hand_controller_profile(cfg_path, "stronger_grasp_v1")
    assert profile.profile_name == "stronger_grasp_v1"
    assert profile.summary_fields["stiffness"] == 1000
    assert profile.summary_fields["force_limit"] == 100


def test_apply_hand_controller_profile_overrides_updates_config_mapping(tmp_path):
    cfg_path = tmp_path / "hand_controller.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "demo_controller": {
                        "stiffness": 11,
                        "damping": 22,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config_overrides = {
        "hand_controller_config": str(cfg_path),
        "hand_controller_profile": "demo_controller",
    }

    resolved = apply_hand_controller_profile_overrides(config_overrides, scope="HandControllerTest")

    assert resolved is not None
    assert config_overrides["hand_controller_config"] == str(cfg_path.resolve())
    assert config_overrides["hand_controller_profile"] == "demo_controller"
    assert config_overrides["hand_controller"]["stiffness"] == 11
    assert config_overrides["hand_controller"]["damping"] == 22


def test_resolve_rc5_move_group_prefers_explicit_value_without_warning(capsys):
    assert (
        resolve_rc5_move_group(
            "right_tcp_link",
            env={"OPENR2S_RC5_MOVE_GROUP": "right_tcp_link"},
            scope="Test",
        )
        == "right_tcp_link"
    )
    assert capsys.readouterr().out == ""


def test_resolve_rc5_move_group_warns_for_environment_and_default(capsys):
    assert (
        resolve_rc5_move_group(
            None,
            env={"OPENR2S_RC5_MOVE_GROUP": "right_tcp_link"},
            scope="Test",
        )
        == "right_tcp_link"
    )
    captured = capsys.readouterr()
    assert "[WARNING] [Test] Using OPENR2S_RC5_MOVE_GROUP from environment: right_tcp_link" in captured.out

    assert resolve_rc5_move_group(None, env={}, scope="Test") == "right_tcp_link"
    captured = capsys.readouterr()
    assert "[Test] Using canonical RC5 move group default: right_tcp_link" in captured.out


def test_resolve_rc5_move_group_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid RC5 move group 'bogus'"):
        resolve_rc5_move_group("bogus", env={}, scope="Test")


def test_rc5_unified_bootstrap_rejects_missing_key(tmp_path):
    cfg_path = _write_config(tmp_path, key="demo_key")

    with pytest.raises(KeyError):
        extract_rc5_bootstrap_config(cfg_path, "missing_key")


def test_rc5_unified_bootstrap_rejects_key_not_declared_in_keys_list(tmp_path):
    cfg_path = _write_config(tmp_path, key="demo_key", keys=["other_key"])

    with pytest.raises(ValueError, match="is not declared under top-level keys"):
        extract_rc5_bootstrap_config(cfg_path, "demo_key")


def test_rc5_unified_bootstrap_rejects_unpaired_hand_contact_fields(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        local_sim={
            "robot_uids": "rc5_aero_hand_openr2s",
            "hand_contact_config": "config/hand_contact_profiles.yaml",
        },
    )

    bootstrap = extract_rc5_bootstrap_config(cfg_path, "demo_key")

    with pytest.raises(ValueError, match="hand_contact_config and hand_contact_profile must be provided together"):
        validate_rc5_bootstrap_for_backend(bootstrap, "planner")


def test_rc5_unified_bootstrap_rejects_proxy_backend_without_teleop_profile(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        local_sim={
            "robot_uids": "rc5_aero_hand_openr2s_rl",
            "hand_pose_config": "config/hand_pose_presets.yaml",
            "hand_contact_config": "config/hand_contact_profiles.yaml",
            "hand_contact_profile": "rubber_fingertips_v1",
            "hand_controller_config": "config/hand_controller_profiles.yaml",
            "hand_controller_profile": "stronger_grasp_v1",
        },
    )

    bootstrap = extract_rc5_bootstrap_config(cfg_path, "demo_key")

    with pytest.raises(ValueError, match="requires teleop_profile_config and teleop_profile"):
        validate_rc5_bootstrap_for_backend(bootstrap, "proxy_ee_delta")


def test_rc5_unified_bootstrap_allows_planner_backend_without_teleop_profile(tmp_path):
    cfg_path = _write_config(
        tmp_path,
        local_sim={
            "robot_uids": "rc5_aero_hand_openr2s",
            "hand_pose_config": "config/hand_pose_presets.yaml",
            "hand_contact_config": "config/hand_contact_profiles.yaml",
            "hand_contact_profile": "rubber_fingertips_v1",
            "hand_controller_config": "config/hand_controller_profiles.yaml",
            "hand_controller_profile": "stronger_grasp_v1",
        },
    )

    bootstrap = extract_rc5_bootstrap_config(cfg_path, "demo_key")

    validate_rc5_bootstrap_for_backend(bootstrap, "planner")


class _DummyRobot:
    def __init__(self, qpos):
        self._qpos = np.asarray(qpos, dtype=np.float32)
        self.last_set_qpos = None

    def get_qpos(self):
        return self._qpos.copy()

    def set_qpos(self, qpos):
        self.last_set_qpos = np.asarray(qpos, dtype=np.float32).copy()
        self._qpos = self.last_set_qpos.copy()


class _DummyAgent:
    def __init__(self, qpos, *, arm_joint_names, hand_open_qpos=None):
        self.robot = _DummyRobot(qpos)
        self.arm_joint_names = list(arm_joint_names)
        if hand_open_qpos is not None:
            self.hand_open_qpos = np.asarray(hand_open_qpos, dtype=np.float32)


class _DummyGripperConfig:
    def __init__(self):
        self.open_qpos = None
        self.close_qpos = None


class _DummyGripperController:
    def __init__(self):
        self.config = _DummyGripperConfig()


class _DummyController:
    def __init__(self):
        self.controllers = {"gripper": _DummyGripperController()}


class _DummyEnv:
    def __init__(self, agent):
        self.agent = agent


def test_reset_planner_hand_target_to_open_updates_robot_qpos_and_callbacks():
    agent = _DummyAgent(
        qpos=[0.1, 0.2, 0.9, 0.8, 0.7],
        arm_joint_names=["joint0", "joint1"],
        hand_open_qpos=[0.3, 0.4, 0.5],
    )
    env = _DummyEnv(agent)
    restore_calls = []
    save_calls = []

    def _restore(*args, **kwargs):
        restore_calls.append((args, kwargs))

    def _save(*args, **kwargs):
        save_calls.append((args, kwargs))

    did_reset = reset_planner_hand_target_to_open(
        env,
        restore_planner_grasp_state_fn=_restore,
        save_planner_grasp_state_fn=_save,
        source_stage="reset",
    )

    assert did_reset is True
    assert len(restore_calls) == 1
    assert len(save_calls) == 1
    assert restore_calls[0][0] == (env,)
    assert np.allclose(restore_calls[0][1]["default_target_hand_qpos"], [0.3, 0.4, 0.5])
    assert restore_calls[0][1]["source_stage"] == "reset"
    assert save_calls[0][0] == (env,)
    assert np.allclose(save_calls[0][1]["target_hand_qpos"], [0.3, 0.4, 0.5])
    assert np.allclose(save_calls[0][1]["realized_hand_qpos"], [0.3, 0.4, 0.5])
    assert save_calls[0][1]["grasp_flag"] is False
    assert save_calls[0][1]["object_id"] is None
    assert save_calls[0][1]["source_stage"] == "reset"
    assert np.allclose(agent.robot.last_set_qpos, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_reset_planner_hand_target_to_open_returns_false_without_hand_open_qpos():
    agent = _DummyAgent(qpos=[0.1, 0.2], arm_joint_names=["joint0", "joint1"])
    env = _DummyEnv(agent)

    did_reset = reset_planner_hand_target_to_open(
        env,
        restore_planner_grasp_state_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("restore callback should not be called")
        ),
        save_planner_grasp_state_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("save callback should not be called")
        ),
        source_stage="reset",
    )

    assert did_reset is False


def test_apply_hand_pose_config_to_agent_loads_presets_and_updates_runtime_controller(tmp_path):
    cfg_path = tmp_path / "hand_pose.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "bindings": {"open": "teleop_open", "close": "teleop_close"},
                "poses": {
                    "teleop_open": {"joints": {"j0": 0.1, "j1": 0.2}},
                    "teleop_close": {"joints": {"j0": 0.8, "j1": 0.9}},
                },
            }
        ),
        encoding="utf-8",
    )
    agent = _DummyAgent(qpos=[0.0, 0.0], arm_joint_names=[])
    agent.hand_joint_names = ["j0", "j1"]
    agent.controller = _DummyController()

    applied = apply_hand_pose_config_to_agent(agent, cfg_path)

    assert applied is not None
    assert applied.config_path == cfg_path.resolve()
    assert applied.open_preset_name == "teleop_open"
    assert applied.close_preset_name == "teleop_close"
    assert np.allclose(agent.hand_open_qpos, [0.1, 0.2])
    assert np.allclose(agent.hand_close_qpos, [0.8, 0.9])
    assert np.allclose(agent.controller.controllers["gripper"].config.open_qpos, [0.1, 0.2])
    assert np.allclose(agent.controller.controllers["gripper"].config.close_qpos, [0.8, 0.9])
    assert np.allclose(applied.presets["teleop_open"], [0.1, 0.2])
    assert np.allclose(applied.presets["teleop_close"], [0.8, 0.9])


def test_apply_hand_pose_config_to_agent_supports_gripper_joint_name_fallback_and_overrides(tmp_path):
    cfg_path = tmp_path / "hand_pose.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "bindings": {
                    "open": "teleop_open",
                    "close": "teleop_close",
                    "full_open": "teleop_open",
                },
                "poses": {
                    "teleop_open": {"joints": {"g0": 0.0, "g1": 0.1}},
                    "teleop_close": {"joints": {"g0": 0.7, "g1": 0.8}},
                    "custom_close": {"joints": {"g0": 0.4, "g1": 0.5}},
                },
            }
        ),
        encoding="utf-8",
    )
    agent = _DummyAgent(qpos=[0.0, 0.0], arm_joint_names=[])
    agent.controller = _DummyController()

    with pytest.raises(RuntimeError, match="does not expose hand_joint_names"):
        apply_hand_pose_config_to_agent(
            agent,
            cfg_path,
            close_preset_name="custom_close",
            required_bindings=["open", "close", "full_open"],
        )
