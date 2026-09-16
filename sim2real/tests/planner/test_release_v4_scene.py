"""Portable scene import and release.v3 compatibility contracts."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml
from transforms3d.quaternions import quat2mat

from openreal2sim.simulation.maniskill.utils.scene_loader import load_scene_config

ROOT = Path(__file__).resolve().parents[2]
OLD = "airi_table_new_empty3_image"
NEW = "airy_table_scene14sep26_left_image"
METRIC = NEW + "_metric"
URDF = ROOT / "openreal2sim/simulation/maniskill/robot_assets/rc5_aero_hand/urdf_rc5_right_hand/Robot _with_right_hand_colored_visual_continuous.urdf"

# The imported NEW base yaw was rotated by -85.98 deg so that sim joint angles equal the
# real RC5 joints (hover check on the real robot, 2026-09-15). Joint0's axis is base Z,
# so this changes the base-frame labelling, not where anything is physically.
JOINT0_CONVENTION_DEG = 85.98


def _rz(deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def config():
    return yaml.safe_load((ROOT / "config/config_debug.yaml").read_text())


def test_old_scene_and_global_settings_unchanged():
    c = config()
    # The opt-in hand appearance is the only approved new global setting.
    c['global']['simulation'].pop('hand_visual_profile', None)
    payload = json.dumps({"global": c["global"], "old": c["local"][OLD]}, sort_keys=True)
    assert hashlib.sha256(payload.encode()).hexdigest() == "a8f809a908ff428781ffe26656b9e001f8ec6fd0c7beca69ae66c45b71ebc575"


@pytest.mark.parametrize('key', [OLD, NEW, METRIC])
def test_scenes_forward_named_hand_visual_profile(key, monkeypatch):
    from openreal2sim.simulation.maniskill.scripts import rc5_unified_proxy_setup as setup
    from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_runtime import _build_direct_args_namespace
    args = _build_direct_args_namespace(['--auto_pick_macro', '1', '--key', key, '--scene',
        str(ROOT / 'assets/scenes' / key / 'simulation/scene.json'),
        '--config_path', str(ROOT / 'config/config_debug.yaml')])
    cfg = setup.load_runner_config(args)
    expected = config()['hand_visual_profiles']['rc5_real_matte_v1']
    assert cfg['hand_visual_profile'] == expected
    assert len(expected['links']) == 23
    assert expected['links']['body6'] == expected['links']['right_base_link'] == 'plastic'
    assert not {f'body{i}' for i in range(6)} & expected['links'].keys()
    assert expected['parts']['prehand'][:3] == ['adapter_green'] * 3
    camera = expected['parts']['prehand'][3]
    assert camera['material'] == 'camera_metal'
    assert camera['expected_triangles'] == 6036
    assert camera['regions'] == [{'material': 'camera_panel', 'bounds_min': [-22, 46, 71.9],
                                  'bounds_max': [22, 89, 73], 'expected_triangles': 1710}]
    assert expected['materials']['camera_metal']['metallic'] == 0.8
    assert expected['materials']['camera_panel']['metallic'] == 0.0
    captured = {}
    def fake_env(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(setup, '_load_openreal2sim_env_class', lambda: fake_env)
    setup.make_env(args, cfg)
    assert captured['hand_visual_profile'] == expected


def test_new_scene_copies_old_object_selection():
    local = config()["local"]
    old, new = local[OLD]["simulation"], local[NEW]["simulation"]
    assert new["include_objects"] == old["include_objects"]
    assert new["manip_object_id"] == old["manip_object_id"]
    assert list(new["object_placements"]) == list(old["object_placements"])


def test_new_scene_preserves_old_robot_relative_object_layout():
    local = config()["local"]
    old, new = local[OLD]["simulation"], local[NEW]["simulation"]
    old_r = quat2mat(old["robot_base_pose"][3:])
    # Undo the joint0 convention change so the comparison uses the imported base frame.
    new_r = quat2mat(new["robot_base_pose"][3:]) @ _rz(JOINT0_CONVENTION_DEG)
    for oid, original in old["object_placements"].items():
        moved = new["object_placements"][oid]
        old_xy = np.array(original["position"][:2]) - old["robot_base_pose"][:2]
        new_xy = np.array(moved["position"][:2]) - new["robot_base_pose"][:2]
        np.testing.assert_allclose(new_r[:2, :2].T @ new_xy,
                                   old_r[:2, :2].T @ old_xy, atol=1e-9, rtol=0,
                                   err_msg=oid)
        np.testing.assert_allclose(new_r.T @ quat2mat(moved["orientation"]),
                                   old_r.T @ quat2mat(original["orientation"]),
                                   atol=1e-9, rtol=0, err_msg=oid)
        assert moved["position"][2] == original["position"][2]
        assert {k: v for k, v in moved.items() if k not in ("position", "orientation")} == {
            k: v for k, v in original.items() if k not in ("position", "orientation")}


@pytest.mark.parametrize("target", ["planner", "planner-headless"])
@pytest.mark.parametrize("key", [None, OLD, NEW])
def test_make_selects_matching_scene(key, target):
    args = ["make", "--no-print-directory", "-n", target]
    if key is not None:
        args.append("KEY=" + key)
    command = subprocess.check_output(args, cwd=ROOT, text=True)
    expected = key or NEW
    assert "--key " + expected + " " in command
    assert "--scene assets/scenes/" + expected + "/simulation/scene.json" in command
    assert "--motion_backend proxy_ee_delta" in command


def test_readme_and_make_help_document_both_scene_launches():
    help_text = subprocess.check_output(
        ["make", "--no-print-directory", "help"], cwd=ROOT, text=True)
    readme = (ROOT / "README.md").read_text()
    for key in config()["keys"]:
        for target in ("planner", "planner-headless"):
            example = f"make {target} KEY={key}"
            assert example in help_text
            assert example in readme


def test_make_explicit_scene_override_is_preserved():
    command = subprocess.check_output(["make", "-n", "planner", "SCENE=/tmp/custom/scene.json"], cwd=ROOT, text=True)
    assert "--scene /tmp/custom/scene.json" in command


@pytest.mark.parametrize("key", [OLD, NEW])
def test_scene_bundle_loads_and_all_declared_paths_exist(key):
    path = ROOT / "assets/scenes" / key / "simulation/scene.json"
    scene = load_scene_config(path)
    assert Path(scene.background_mesh_path).is_file()
    c = config()
    assert key in c["keys"]
    sim = c["global"]["simulation"] | c["local"][key]["simulation"]
    for oid in sim["include_objects"]:
        obj = sim["object_placements"][oid]
        for field in ("mesh_path", "collision_mesh_path"):
            assert (ROOT / obj[field].removeprefix("/app/")).is_file(), (oid, field)
    if key == NEW:
        def check_paths(value):
            if isinstance(value, dict):
                for v in value.values():
                    check_paths(v)
            elif isinstance(value, str) and value.startswith("/app/"):
                assert value.startswith("/app/assets/scenes/" + NEW + "/")
                assert (ROOT / value.removeprefix("/app/")).is_file(), value
        check_paths(json.loads(path.read_text()))


def test_new_scene_flat_surface_base_and_camera():
    c = config()
    sim = c["global"]["simulation"] | c["local"][NEW]["simulation"]
    assert sim["auto_placement"] is False
    assert sim["robot_base_pose_z_auto"] is False
    assert sim["placement_mode"] == "fixed"
    assert sim["object_spawn_clearance"] == .002
    np.testing.assert_allclose(sim["robot_base_pose"], [.49556433594, -.616155178655, .324685138162, .95269659488, 0, 0, -.303923013449], atol=1e-10, rtol=0)
    scene = load_scene_config(ROOT / "assets/scenes" / NEW / "simulation/scene.json")
    assert abs(scene.ground_plane_point[2] - sim["robot_base_pose"][2]) < 1e-9
    assert (scene.camera.width, scene.camera.height) == (1886, 1059)
    assert scene.camera.fx == 864.0193481445312
    assert sim["cameras"]["base_camera"]["use_scene_json"] is True


def test_new_scene_uses_supported_proxy_calibration_schema():
    sim = config()["local"][NEW]["simulation"]
    assert "planner_proxy_object_calibrations" not in sim
    assert "robot_init_qpos_by_object" not in sim
    cube = sim["planner_object_calibrations"]["blue_cube_ext"]
    np.testing.assert_allclose(cube["pregrasp_offset_xyz"], [-.003535468875, .045766368218, .0886])
    np.testing.assert_allclose(cube["target_quat"], [-.524541662652, -.693104403414, .429013234946, .24578440624])
    for profile in sim["planner_object_calibrations"].values():
        assert not any(k.startswith("hand_pregrasp") for k in profile)
    bottle = sim["planner_object_calibrations"]["spray_bottle_ext"]
    assert bottle["target_orientation_mode"] == "current_tcp"
    assert "target_quat" not in bottle
    assert bottle["pregrasp_offset_xyz"][:2] == bottle["descend_offset_xyz"][:2]
    assert sim["robot_init_qpos_profile_by_object"]["spray_bottle_ext"] == "side"
    assert len(sim["robot_init_qpos_profiles"]["side"]) == 22
    assert sim["robot_init_qpos_profiles"]["side"][5] == 1.570796


@pytest.mark.parametrize("filename,sha", [
    ("background_registered.glb", "9f50f3c584d5d38cf54e02498050bd35b4514964aa8def33df045d4c45600029"),
    ("background_registered_collision.glb", "94f0ece4fc4c8a747c50020edd57ae4e4e3b0a3bc7d6368ea7a639fca0ed8699"),
])
def test_imported_meshes_are_unmodified(filename, sha):
    path = ROOT / "assets/scenes" / NEW / "simulation" / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


@pytest.mark.parametrize("key", [NEW, METRIC])
def test_scene_joint0_convention_matches_real_robot(key):
    sim = config()["local"][key]["simulation"]
    q = sim["robot_base_pose"]
    yaw = np.degrees(2 * np.arctan2(q[6], q[3]))
    assert abs(yaw - (50.593235 - JOINT0_CONVENTION_DEG)) < 1e-3
    for name in ("top", "side"):
        prof = sim["robot_init_qpos_profiles"][name]
        assert abs(prof[0] - (0.2853 + np.radians(JOINT0_CONVENTION_DEG))) < 1e-6, name
        assert abs(prof[3] - (-3.15 + 2 * np.pi)) < 1e-6, name


def test_rc5_urdf_joint3_range_matches_real_robot():
    import re
    block = re.search(r'<joint name="joint3".*?</joint>', URDF.read_text(), re.S).group(0)
    limit = re.search(r'<limit[^>]*>', block).group(0)
    assert float(re.search(r'lower="([^"]+)"', limit).group(1)) == -6.283185
    assert float(re.search(r'upper="([^"]+)"', limit).group(1)) == 6.283185
