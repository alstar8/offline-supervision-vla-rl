# -*- coding: utf-8 -*-
"""Scene configuration loader for OpenReal2Sim outputs."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from mani_skill.utils.structs.pose import Pose
from transforms3d.quaternions import mat2quat


@dataclass
class CameraConfig:
    """Camera configuration from scene.json."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    extrinsic_matrix: list  # 4x4 matrix as nested list
    intrinsic_matrix: list  # 3x3 matrix as nested list
    position: list  # [x, y, z]
    orientation_wxyz: list  # [w, x, y, z]


@dataclass
class ObjectConfig:
    """Object configuration from scene.json."""

    oid: int
    name: str
    mesh_path: str
    center: list  # [x, y, z]
    bbox_min: list  # [x, y, z]
    bbox_max: list  # [x, y, z]
    collision_mesh_path: Optional[str] = None
    grasps: Optional[str] = None
    trajectory_path: Optional[str] = None


@dataclass
class SceneConfig:
    """Complete scene configuration."""

    background_mesh_path: str
    camera: CameraConfig
    objects: Dict[str, ObjectConfig]
    ground_plane_point: list  # [x, y, z]
    ground_plane_normal: list  # [x, y, z]
    scene_aabb_min: list
    scene_aabb_max: list
    manipulated_oid: Optional[str] = None  # id of the manipulation target from scene.json
    task_desc: Optional[str] = None        # natural-language task instruction from scene.json


def load_scene_config(scene_json_path: str | Path) -> SceneConfig:
    """
    Load scene configuration from scene.json file.

    Args:
        scene_json_path: Path to scene.json file

    Returns:
        SceneConfig object containing all scene information

    Raises:
        FileNotFoundError: If scene.json doesn't exist
        ValueError: If scene.json is malformed
    """
    scene_json_path = Path(scene_json_path)

    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")

    with open(scene_json_path, "r") as f:
        data = json.load(f)

    # Parse camera configuration
    cam_data = data.get("camera", {})
    intrinsic_matrix = np.array(
        [
            [cam_data["fx"], 0, cam_data["cx"]],
            [0, cam_data["fy"], cam_data["cy"]],
            [0, 0, 1],
        ]
    )
    camera = CameraConfig(
        width=int(cam_data["width"]),
        height=int(cam_data["height"]),
        fx=float(cam_data["fx"]),
        fy=float(cam_data["fy"]),
        cx=float(cam_data["cx"]),
        cy=float(cam_data["cy"]),
        extrinsic_matrix=cam_data["camera_opencv_to_world"],
        intrinsic_matrix=intrinsic_matrix.tolist(),
        position=cam_data["camera_position"],
        orientation_wxyz=cam_data["camera_heading_wxyz"],
    )

    # Parse objects
    objects = {}
    for obj_id, obj_data in data.get("objects", {}).items():
        # Use the optimized mesh if available, otherwise registered
        mesh_path = obj_data.get("optimized") or obj_data.get("registered")
        collision_mesh_path = obj_data.get("collision_mesh_path")

        # Resolve /app/... paths against the repo root for both supported layouts:
        #   - <repo>/outputs/<key>/simulation/scene.json
        #   - <repo>/assets/scenes/<key>/simulation/scene.json
        parents = scene_json_path.parents
        output_path = scene_json_path.parent.parent.parent.parent
        if len(parents) >= 4 and parents[2].name == "outputs":
            output_path = parents[3]
        elif len(parents) >= 5 and parents[2].name == "scenes" and parents[3].name == "assets":
            output_path = parents[4]

        if mesh_path and mesh_path.startswith("/app/"):
            mesh_path = mesh_path.replace("/app/", str(output_path) + "/")
        if collision_mesh_path and collision_mesh_path.startswith("/app/"):
            collision_mesh_path = collision_mesh_path.replace("/app/", str(output_path) + "/")

        grasp_path_raw = obj_data.get("grasps")
        grasp_path = grasp_path_raw.replace("/app/", str(output_path) + "/") if grasp_path_raw else ""

        # Resolve trajectory path
        trajectory_path = (
            obj_data.get("hybrid_trajs") or obj_data.get("simple_trajs") or ""
        )
        if trajectory_path:
            trajectory_path = trajectory_path.replace("/app/", str(output_path) + "/")

        objects[obj_id] = ObjectConfig(
            oid=obj_data["oid"],
            name=obj_data["name"],
            mesh_path=mesh_path,
            collision_mesh_path=collision_mesh_path,
            center=obj_data.get("object_center", [0, 0, 0]),
            bbox_min=obj_data.get("object_min", [0, 0, 0]),
            bbox_max=obj_data.get("object_max", [0, 0, 0]),
            grasps=grasp_path,
            trajectory_path=trajectory_path,
        )

    # Parse background
    bg_data = data.get("background", {})
    bg_path = bg_data.get("registered") or bg_data.get("original")
    if bg_path and bg_path.startswith("/app/"):
        bg_path = bg_path.replace("/app/", str(output_path) + "/")

    # Parse ground plane (use simulation frame)
    ground_data = data.get("groundplane_in_sim", {})

    # Parse AABB
    aabb_data = data.get("aabb", {})

    manipulated_oid = data.get("manipulated_oid")

    # task_desc is a list in scene.json; take first element as the instruction string
    task_desc_raw = data.get("task_desc")
    if isinstance(task_desc_raw, list) and task_desc_raw:
        task_desc = str(task_desc_raw[0])
    elif isinstance(task_desc_raw, str) and task_desc_raw:
        task_desc = task_desc_raw
    else:
        task_desc = None

    return SceneConfig(
        background_mesh_path=bg_path,
        camera=camera,
        objects=objects,
        ground_plane_point=ground_data.get("point", [0, 0, 0]),
        ground_plane_normal=ground_data.get("normal", [0, 0, 1]),
        scene_aabb_min=aabb_data.get("scene_min", [-1, -1, -1]),
        scene_aabb_max=aabb_data.get("scene_max", [1, 1, 1]),
        manipulated_oid=str(manipulated_oid) if manipulated_oid is not None else None,
        task_desc=task_desc,
    )


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path from scene.json, handling container paths.

    Args:
        path: Path string from scene.json
        base_dir: Base directory to resolve relative paths (default: cwd)

    Returns:
        Resolved Path object
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Keep container paths absolute. They are already canonical inside the runtime
    # container and must not be re-based against cwd, otherwise /app/assets/... can
    # incorrectly become /app/assets/assets/....
    if path.startswith("/app/"):
        return Path(path)

    resolved = Path(path)

    # If not absolute, make it relative to base_dir
    if not resolved.is_absolute():
        resolved = base_dir / resolved

    return resolved


def load_trajectory(filepath: str) -> list[Pose] | None:
    """Loads an object trajectory from a .npy file."""
    if not filepath or not filepath.endswith(".npy"):
        print(f"Error: Invalid trajectory file path provided: {filepath}")
        return None
    try:
        trajectory_data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Trajectory file not found at {filepath}")
        return None

    poses = []
    # The trajectory can be stored in different formats, so we handle both
    # a list of 4x4 matrices and a dictionary containing the poses.
    if isinstance(trajectory_data, dict):
        trajectory_data = trajectory_data["world_pose"]

    for matrix in trajectory_data:
        # position
        p = matrix[:3, 3]
        # rotation matrix to wxyz quaternion
        q_wxyz = mat2quat(matrix[:3, :3])
        poses.append(Pose.create_from_pq(p=p, q=q_wxyz))
    return poses


def get_base_camera_eye_target(
    scene_json_path: str | Path,
    look_distance: float = 2.0,
    background_init_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (eye, target) for look_at from base camera pose in scene.json.

    Uses the same logic as OpenReal2SimEnv._default_sensor_configs for base_camera.
    Useful for video recording with render camera matching the reconstruction view.

    Args:
        scene_json_path: Path to scene.json
        look_distance: Distance from eye to target for computing target point
        background_init_z: Z offset for camera (matches BACKGROUND_INIT_POS[2])

    Returns:
        tuple: (eye, target) as numpy arrays [3]
    """
    from .transform_utils import opencv_to_sapien_pose, qvec2rotmat

    scene_config = load_scene_config(scene_json_path)
    cam_config = scene_config.camera

    extrinsic_matrix = np.array(cam_config.extrinsic_matrix, dtype=np.float32)
    extrinsic_matrix[2, 3] += background_init_z
    camera_pose = opencv_to_sapien_pose(extrinsic_matrix)

    q_numpy = np.asarray(camera_pose.q).flatten()
    if hasattr(q_numpy, 'cpu'):
        q_numpy = q_numpy.cpu().numpy()
    p_numpy = np.asarray(camera_pose.p).flatten()
    if hasattr(p_numpy, 'cpu'):
        p_numpy = p_numpy.cpu().numpy()

    camera_rot_mat = qvec2rotmat(q_numpy)

    eye = p_numpy.astype(np.float64)
    forward = camera_rot_mat[:, 0]
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    target = eye + forward * look_distance

    print("[BASE_CAMERA] === get_base_camera_eye_target (video) ===")
    print(f"[BASE_CAMERA]   eye: [{eye[0]:.6f}, {eye[1]:.6f}, {eye[2]:.6f}]")
    print(f"[BASE_CAMERA]   target: [{target[0]:.6f}, {target[1]:.6f}, {target[2]:.6f}]")
    print(f"[BASE_CAMERA]   forward (R[:,0]): [{forward[0]:.6f}, {forward[1]:.6f}, {forward[2]:.6f}]")

    return eye, target


# Дефолтные параметры камер (используются при отсутствии конфига)
DEFAULT_CAMERAS_CONFIG = {
    "base_camera": {
        "use_scene_json": True,
    },
    "render_camera": {
        "use_base_camera": False,
        "eye": [0.8, 0.8, 0.6],
        "target": [0.0, 0.0, 0.2],
        "width": 512,
        "height": 512,
        "fov": 1.0471975511965976,  # np.pi / 3
    },
    "custom_cameras": [],
}


def load_cameras_config(
    cfg: dict,
    key: Optional[str] = None,
) -> dict:
    """
    Load and merge cameras config from config dict.
    local.<key>.simulation.cameras overrides global.simulation.cameras.

    Args:
        cfg: Full config dict (from yaml.safe_load)
        key: Scene key (e.g. scene8_video). If None, only global is used.

    Returns:
        Merged cameras config dict with keys: base_camera, render_camera, custom_cameras
    """
    import copy
    global_sim = cfg.get("global", {}).get("simulation", {})
    global_cameras = global_sim.get("cameras", {})
    result = copy.deepcopy(DEFAULT_CAMERAS_CONFIG)
    # Merge global cameras
    for cam_name in ("base_camera", "render_camera"):
        if cam_name in global_cameras and isinstance(global_cameras[cam_name], dict):
            for k, v in global_cameras[cam_name].items():
                result[cam_name][k] = v
    if "custom_cameras" in global_cameras and isinstance(global_cameras["custom_cameras"], list):
        result["custom_cameras"] = list(global_cameras["custom_cameras"])

    if key:
        local_sim = cfg.get("local", {}).get(key, {}).get("simulation", {})
        local_cameras = local_sim.get("cameras", {})
        for cam_name in ("base_camera", "render_camera"):
            if cam_name in local_cameras and isinstance(local_cameras[cam_name], dict):
                for k, v in local_cameras[cam_name].items():
                    result[cam_name][k] = v
        if "custom_cameras" in local_cameras and isinstance(local_cameras["custom_cameras"], list):
            result["custom_cameras"] = list(local_cameras["custom_cameras"])
    return result
