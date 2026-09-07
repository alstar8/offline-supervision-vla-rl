from __future__ import annotations

import json
from pathlib import Path

from openreal2sim.simulation.maniskill.utils.scene_loader import load_scene_config


def _write_scene(scene_json_path: Path, background_path: str, object_path: str) -> None:
    scene_json_path.parent.mkdir(parents=True, exist_ok=True)
    scene_json_path.write_text(
        json.dumps(
            {
                "background": {"registered": background_path},
                "camera": {
                    "width": 1,
                    "height": 1,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "camera_opencv_to_world": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "camera_position": [0.0, 0.0, 0.0],
                    "camera_heading_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "objects": {
                    "1": {
                        "oid": 1,
                        "name": "demo",
                        "optimized": object_path,
                        "object_center": [0.0, 0.0, 0.0],
                        "object_min": [0.0, 0.0, 0.0],
                        "object_max": [0.0, 0.0, 0.0],
                    }
                },
                "groundplane_in_sim": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
                "aabb": {"scene_min": [0.0, 0.0, 0.0], "scene_max": [1.0, 1.0, 1.0]},
            }
        ),
        encoding="utf-8",
    )


def test_load_scene_config_resolves_assets_scenes_layout_without_double_assets(tmp_path: Path) -> None:
    scene_json = tmp_path / "assets" / "scenes" / "demo_key" / "simulation" / "scene.json"
    _write_scene(
        scene_json,
        "/app/assets/scenes/demo_key/simulation/background_registered.glb",
        "/app/assets/scenes/demo_key/simulation/demo_object.glb",
    )

    scene_config = load_scene_config(scene_json)

    assert scene_config.background_mesh_path == str(
        tmp_path / "assets" / "scenes" / "demo_key" / "simulation" / "background_registered.glb"
    )
    assert scene_config.objects["1"].mesh_path == str(
        tmp_path / "assets" / "scenes" / "demo_key" / "simulation" / "demo_object.glb"
    )


def test_load_scene_config_preserves_legacy_outputs_layout(tmp_path: Path) -> None:
    scene_json = tmp_path / "outputs" / "demo_key" / "simulation" / "scene.json"
    _write_scene(
        scene_json,
        "/app/outputs/demo_key/simulation/background_registered.glb",
        "/app/outputs/demo_key/simulation/demo_object.glb",
    )

    scene_config = load_scene_config(scene_json)

    assert scene_config.background_mesh_path == str(
        tmp_path / "outputs" / "demo_key" / "simulation" / "background_registered.glb"
    )
    assert scene_config.objects["1"].mesh_path == str(
        tmp_path / "outputs" / "demo_key" / "simulation" / "demo_object.glb"
    )
