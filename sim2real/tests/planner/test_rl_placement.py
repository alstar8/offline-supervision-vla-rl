from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openreal2sim.simulation.maniskill.utils.rl_placement import (
    DEFAULT_ROBOT_BASE_XY,
    instruction_for_manip_object,
    sample_nonoverlapping_xy,
    xy_half_extent_from_bbox,
)
from openreal2sim.simulation.maniskill.utils.scene_loader import (
    SIM2REAL_REPO_ROOT,
    remap_object_placement_paths,
    resolve_app_runtime_path,
)


def test_resolve_app_runtime_path_keeps_existing_container_path(tmp_path: Path) -> None:
    container = tmp_path / "app" / "assets" / "object_bank" / "cube" / "visual.glb"
    container.parent.mkdir(parents=True)
    container.write_text("mesh", encoding="utf-8")
    resolved = resolve_app_runtime_path(str(container))
    assert resolved == container


def test_resolve_app_runtime_path_rebases_missing_app_path() -> None:
    missing = "/app/assets/object_bank/orange_cube_ext/visual.glb"
    resolved = resolve_app_runtime_path(missing)
    assert resolved == SIM2REAL_REPO_ROOT / "assets" / "object_bank" / "orange_cube_ext" / "visual.glb"


def test_remap_object_placement_paths_rewrites_mesh_keys() -> None:
    remapped = remap_object_placement_paths(
        {
            "orange_cube_ext": {
                "mesh_path": "/app/assets/object_bank/orange_cube_ext/visual.glb",
                "collision_mesh_path": "/app/assets/object_bank/orange_cube_ext/collision.coacd.ply",
                "position": [-0.3, -0.7, 0.0],
            }
        }
    )
    assert remapped["orange_cube_ext"]["mesh_path"] == str(
        SIM2REAL_REPO_ROOT / "assets" / "object_bank" / "orange_cube_ext" / "visual.glb"
    )
    assert remapped["orange_cube_ext"]["collision_mesh_path"] == str(
        SIM2REAL_REPO_ROOT / "assets" / "object_bank" / "orange_cube_ext" / "collision.coacd.ply"
    )
    assert remapped["orange_cube_ext"]["position"] == [-0.3, -0.7, 0.0]


def test_instruction_prefers_pick_red_cube_for_orange() -> None:
    assert (
        instruction_for_manip_object(
            task_description="Pick red cube",
            manip_object_id="orange_cube_ext",
            object_placements={"orange_cube_ext": {"task_semantic_name": "orange cube"}},
            scene_task_desc="Pick up the sheet.",
        )
        == "Pick red cube"
    )
    assert (
        instruction_for_manip_object(
            task_description=None,
            manip_object_id="orange_cube_ext",
            object_placements={"orange_cube_ext": {"task_semantic_name": "orange cube"}},
            scene_task_desc="Pick up the sheet.",
        )
        == "Pick red cube"
    )


def test_sample_nonoverlapping_xy_respects_gap_and_robot() -> None:
    rng = np.random.RandomState(0)
    occupied = [(np.array([-0.30, -0.70]), 0.03)]
    samples = []
    for _ in range(8):
        xy = sample_nonoverlapping_xy(
            rng,
            radius=0.03,
            bounds_min=[-0.38, -0.90],
            bounds_max=[0.05, -0.52],
            occupied=occupied,
            robot_base_xy=DEFAULT_ROBOT_BASE_XY,
            min_robot_clearance=0.28,
            pair_gap=0.015,
        )
        assert xy is not None
        for center, radius in occupied:
            assert float(np.linalg.norm(xy - center)) >= 0.03 + radius + 0.015 - 1e-9
        assert float(np.linalg.norm(xy - DEFAULT_ROBOT_BASE_XY)) >= 0.28 + 0.03 - 1e-9
        occupied.append((xy, 0.03))
        samples.append(xy)
    assert len(samples) == 8


def test_xy_half_extent_from_bbox() -> None:
    assert xy_half_extent_from_bbox((np.array([-0.02, -0.02, 0.0]), np.array([0.02, 0.02, 0.04]))) == 0.02


def test_rl_gym_kwargs_target_orange_cube_and_local_meshes() -> None:
    from openreal2sim.simulation.maniskill.rl_gym import build_openreal2sim_rl_gym_kwargs

    kwargs = build_openreal2sim_rl_gym_kwargs(initialize_renderer=False, use_wrist_camera=True)
    assert kwargs["manip_object_id"] == "orange_cube_ext"
    assert kwargs["task_description"] == "Pick red cube"
    assert kwargs["placement_mode"] == "random"
    assert kwargs["use_wrist_camera"] is True
    assert kwargs["use_360_background"] is True
    assert kwargs["robot_uids"] == "rc5_aero_hand_openr2s_rl"
    orange = kwargs["object_placements"]["orange_cube_ext"]
    assert not str(orange["mesh_path"]).startswith("/app/")
    assert Path(orange["mesh_path"]).name == "visual.glb"
    assert kwargs["random_placement"]["bounds_min"][0] == pytest.approx(-0.38)
    assert kwargs["cameras_config"]["base_camera"]["width"] == 640
    assert kwargs["cameras_config"]["base_camera"]["height"] == 480
