import ast
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import quat2mat
from transforms3d.quaternions import mat2quat

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.rc5.rc5 import RC5AeroHandRight
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    WidowX250SBridgeDatasetFlatTable,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from real2sim.rc5_calibrated_presets import (
    RC5_HAND_GRAB_QPOS,
    RC5_HAND_OPEN_QPOS,
    RC5_HOME_BASE_POSE_P,
    RC5_HOME_BASE_POSE_Q,
    RC5_HOME_INIT_QPOS,
)
from real2sim.debug_paths import resolve_spawn_grid_state_path


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENREAL2SIM_ASSET_DIR = REPO_ROOT / "real2sim" / "assets" / "openreal2sim"
AIRI_TABLE_EMPTY_ASSET_DIR = REPO_ROOT / "real2sim" / "assets" / "airi_table_empty" / "assets"
AIRI_CUBES_ASSET_DIR = REPO_ROOT / "real2sim" / "assets" / "airi_cubes" / "assets"
MANISKILL_REPO_ASSET_DIR = REPO_ROOT / "ManiSkill" / "mani_skill" / "assets"
REPLICACAD_HDR_ENV_MAP = (
    REPO_ROOT
    / "ManiSkill"
    / "mani_skill"
    / "utils"
    / "scene_builder"
    / "replicacad"
    / "autumn_field_puresky_4k.hdr"
)

PROBE_CARROT_VISUAL_PATH = (
    MANISKILL_REPO_ASSET_DIR / "carrot" / "more_carrot" / "001_carrot_simpler" / "textured.dae"
)
PROBE_CARROT_COLLISION_PATH = (
    MANISKILL_REPO_ASSET_DIR / "carrot" / "more_carrot" / "001_carrot_simpler" / "collision.obj"
)
PROBE_PLATE_VISUAL_PATH = (
    MANISKILL_REPO_ASSET_DIR / "carrot" / "more_plate" / "001_plate_simpler" / "textured.dae"
)
PROBE_PLATE_COLLISION_PATH = (
    MANISKILL_REPO_ASSET_DIR / "carrot" / "more_plate" / "001_plate_simpler" / "collision.obj"
)
MORE_CARROT_MODEL_DB_PATH = MANISKILL_REPO_ASSET_DIR / "carrot" / "more_carrot" / "model_db.json"
MORE_PLATE_MODEL_DB_PATH = MANISKILL_REPO_ASSET_DIR / "carrot" / "more_plate" / "model_db.json"

SENSOR_WIDTH = 640
SENSOR_HEIGHT = 480
WRIST_CAMERA_NAME = "wrist_camera"
WRIST_CAMERA_WIDTH = 168
WRIST_CAMERA_HEIGHT = 224
WRIST_CAMERA_FOV = 1.5000000000000002
WRIST_CAMERA_NEAR = 0.01
WRIST_CAMERA_FAR = 2.0
# Wrist camera calibration from real2sim/debug/wrist_camera_calibration/wrist_camera_latest.txt.
WRIST_CAMERA_MOUNT_LINK = "prehand_cam"
WRIST_CAMERA_LOCAL_P = [0.0, 0.06750000000000002, 0.060600000000000015]
WRIST_CAMERA_LOCAL_Q = [
    0.7071067811865476,
    -0.0,
    0.7071067811865475,
    0.0,
]
DEFAULT_USE_WRIST_CAMERA = True
AIRI_CUBES_VR_ROTATION_GAIN = 3.0
AIRI_CUBES_ACTION_TRANSLATION_CLAMP_M = 0.04
AIRI_CUBES_ACTION_ROTATION_CLAMP_RAD = 0.25


PROBE_CARROT_POSE_P = [-0.075, -2.145, 0.15047362327575684]
PROBE_CARROT_POSE_Q = [0.707, 0.0, 0.0, 0.707]
PROBE_PLATE_POSE_P = [-0.125, -1.962, 0.125]
PROBE_PLATE_POSE_Q = [1.0, 0.0, 0.0, 0.0]

CALIBRATED_SPAWN_GRID_CENTER_XY = np.array(
    [-0.005751613080501555, -2.0679838466644287], dtype=np.float32
)
CALIBRATED_SPAWN_GRID_SIZE_XY = np.array(
    [0.33000001311302185, 0.20000003278255463], dtype=np.float32
)
CALIBRATED_SPAWN_GRID_Z = 0.12047362327575684
OPENREAL2SIM_SPAWN_GRID_CENTER_XY = CALIBRATED_SPAWN_GRID_CENTER_XY.copy()
OPENREAL2SIM_SPAWN_GRID_SIZE_XY = CALIBRATED_SPAWN_GRID_SIZE_XY.copy()
OPENREAL2SIM_SPAWN_GRID_YAW_DEG = 0.0
# Initial guess for the AIRI table scene. This is intentionally only a starting
# rectangle for `calibrate_spawn_grid.py`; the actual 6x6 sampling scripts and
# env options consume whatever calibrated center/size you pass in.
AIRI_TABLE_EMPTY_SPAWN_GRID_CENTER_XY = np.array(
    [0.050495, -0.936646], dtype=np.float32
)
AIRI_TABLE_EMPTY_SPAWN_GRID_SIZE_XY = np.array(
    [0.33000001311302185, 0.20000003278255463], dtype=np.float32
)
AIRI_TABLE_EMPTY_SPAWN_GRID_YAW_DEG = 0.0
AIRI_TABLE_EMPTY_SHADOW_CATCHER_CENTER_XY = np.array(
    [0.10922135412693024, -0.9118247032165527], dtype=np.float32
)
AIRI_TABLE_EMPTY_SHADOW_CATCHER_SIZE_XY = np.array([1.33, 0.74], dtype=np.float32)
AIRI_TABLE_EMPTY_SHADOW_CATCHER_YAW_DEG = 42.8449821472168
AIRI_CUBES_SPAWN_GRID_CENTER_XY = np.array([-0.18, -0.73], dtype=np.float32)
AIRI_CUBES_SPAWN_GRID_SIZE_XY = np.array([0.36, 0.22], dtype=np.float32)
AIRI_CUBES_SPAWN_GRID_YAW_DEG = 0.0
AIRI_CUBES_ROBOT_BASE_POSE_P = np.array([0.45, -0.75, 0.1204], dtype=np.float32)
AIRI_CUBES_ROBOT_BASE_POSE_Q = np.array([0.7071, 0.0, 0.0, -0.7071], dtype=np.float32)
AIRI_CUBES_ROBOT_INIT_QPOS = np.array(
    [
        2.5769,
        -1.4724,
        -1.95,
        -3.15,
        -2.75,
        -0.05,
        1.1205,
        0.5045,
        0.3795,
        0.3793,
        0.0447,
        0.2129,
        0.6239,
        0.7487,
        0.7486,
        0.9985,
        0.2771,
        0.625,
        0.5,
        0.5,
        0.5,
        0.5,
    ],
    dtype=np.float32,
)
AIRI_CUBES_V3_ROBOT_BASE_POSE_P = np.array([0.45, -0.75, 0.1204], dtype=np.float32)
AIRI_CUBES_V3_ROBOT_BASE_POSE_Q = np.array([0.9177546, 0.0, 0.0, 0.3971479], dtype=np.float32)
AIRI_CUBES_V3_ROBOT_INIT_QPOS = np.array(
    [
        0.2853,
        -1.6320,
        -1.7735,
        -3.15,
        -2.75,
        -0.050,
        1.1204,
        0.5051,
        0.3802,
        0.3802,
        0.0452,
        0.2138,
        0.6271,
        0.7516,
        0.7517,
        1.0015,
        0.2797,
        0.6249,
        0.5000,
        0.5000,
        0.5001,
        0.5000,
    ],
    dtype=np.float32,
)
AIRI_CUBE_LOCAL_CENTER = np.array(
    [0.17080174386501312, -0.9838435649871826, 0.5726681351661682], dtype=np.float32
)
AIRI_CUBE_HALF_SIZE = np.array(
    [0.023353710770606995, 0.024188876152038574, 0.02238994836807251], dtype=np.float32
)
AIRI_CUBE_PICKUP_SUCCESS_HEIGHT = 0.12
AIRI_CUBE_SPAWN_EXTRA_Z = 0.004
AIRI_CUBE_CONFIGS = {
    "red": {
        "visual": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_optimized.glb",
        "collision": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_optimized.glb.coacd.ply",
        "xy": np.array([-0.30, -0.70], dtype=np.float32),
    },
    "blue": {
        "visual": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_cyan_optimized.glb",
        "collision": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_cyan_optimized.glb.coacd.ply",
        "xy": np.array([-0.15, -0.75], dtype=np.float32),
    },
    "green": {
        "visual": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_green_optimized.glb",
        "collision": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_green_optimized.glb.coacd.ply",
        "xy": np.array([0.00, -0.80], dtype=np.float32),
    },
    "yellow": {
        "visual": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_yellow_optimized.glb",
        "collision": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_yellow_optimized.glb.coacd.ply",
        "xy": np.array([-0.35, -0.80], dtype=np.float32),
    },
    "white": {
        "visual": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_white_optimized.glb",
        "collision": REPO_ROOT
        / "real2sim"
        / "assets"
        / "airi_cubes"
        / "objects"
        / "zed2i_6cubes_00039_image"
        / "simulation"
        / "6_orange_cube_white_optimized.glb.coacd.ply",
        "xy": np.array([0.00, -0.70], dtype=np.float32),
    },
}
AIRI_CUBE_COLOR_NAMES = tuple(AIRI_CUBE_CONFIGS.keys())
AIRI_CUBE_BASE_XY = np.stack(
    [np.asarray(cfg["xy"], dtype=np.float32) for cfg in AIRI_CUBE_CONFIGS.values()],
    axis=0,
)
AIRI_CUBE_SPAWN_RECT_MIN_XY = AIRI_CUBE_BASE_XY.min(axis=0)
AIRI_CUBE_SPAWN_RECT_MAX_XY = AIRI_CUBE_BASE_XY.max(axis=0)
AIRI_CUBE_SPAWN_GRID_COLS = 5
AIRI_CUBE_SPAWN_GRID_ROWS = 2
AIRI_CUBE_SPAWN_JITTER_XY = np.array([0.012, 0.008], dtype=np.float32)
PROBE_CARROT_Z_OFFSET = PROBE_CARROT_POSE_P[2] - CALIBRATED_SPAWN_GRID_Z
PROBE_PLATE_Z_OFFSET = PROBE_PLATE_POSE_P[2] - CALIBRATED_SPAWN_GRID_Z
SOURCE_SPAWN_EXTRA_Z = 0.08

# Keep the validation env aligned with the calibrated OpenReal2Sim preset used for training.
ROBOT_BASE_POSE_P = [
    float(RC5_HOME_BASE_POSE_P[0]),
    float(RC5_HOME_BASE_POSE_P[1]),
    float(RC5_HOME_BASE_POSE_P[2]),
]
ROBOT_BASE_POSE_Q = [
    float(RC5_HOME_BASE_POSE_Q[0]),
    float(RC5_HOME_BASE_POSE_Q[1]),
    float(RC5_HOME_BASE_POSE_Q[2]),
    float(RC5_HOME_BASE_POSE_Q[3]),
]
ROBOT_INIT_QPOS = RC5_HOME_INIT_QPOS.astype(np.float32).tolist()
SAFE_ROBOT_BASE_POSE_P = [
    ROBOT_BASE_POSE_P[0],
    ROBOT_BASE_POSE_P[1],
    float(CALIBRATED_SPAWN_GRID_Z + 1.0),
]
ROBOT_BLACK_RGBA = sapien_utils.hex2rgba("#2b2b2b")
ROBOT_ARM_METAL_RGBA = np.array([0.08, 0.08, 0.09, 1.0], dtype=np.float32)
ROBOT_HAND_LIGHT_RGBA = np.array([0.86, 0.87, 0.88, 1.0], dtype=np.float32)
ROBOT_PREHAND_GREEN_RGBA = np.array([0.47, 0.80, 0.28, 1.0], dtype=np.float32)


def _load_model_db(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


MORE_CARROT_MODEL_DB = _load_model_db(MORE_CARROT_MODEL_DB_PATH)
MORE_PLATE_MODEL_DB = _load_model_db(MORE_PLATE_MODEL_DB_PATH)
DEFAULT_SOURCE_MODEL_NAME = "001_carrot_simpler"
DEFAULT_PLATE_MODEL_NAME = "001_plate_simpler"
TRAIN_SOURCE_MODEL_COUNT = 16
TRAIN_SOURCE_MODEL_EXCLUDE = {
    "002_kitchen shovel_1",
    "015_golf ball_1",
}
DEFAULT_REAL2SIM_LANGUAGE_INSTRUCTION_TEMPLATE = "Put the {object_name} on the plate"
DEFAULT_REAL2SIM_SIM_FREQ = 50
DEFAULT_REAL2SIM_CONTROL_FREQ = 5
PROBE_SOURCE_DENSITY_SCALE = 0.2
DEFAULT_CUSTOM_SCENE_SPAWN_GRID_STATE_FILE = (
    REPO_ROOT / "real2sim" / "real2sim" / "presets" / "grid_alignment" / "openreal2sim_spawn_grid_latest.txt"
)
DEFAULT_AIRI_CUBES_SPAWN_GRID_STATE_FILE = (
    REPO_ROOT / "real2sim" / "real2sim" / "presets" / "grid_alignment" / "airi_cubes_spawn_grid_initial.txt"
)
DEFAULT_CUSTOM_SCENE_GRID_STEPS_X = 6
DEFAULT_CUSTOM_SCENE_GRID_STEPS_Y = 6
DEFAULT_CUSTOM_SCENE_MIN_PAIR_DISTANCE_XY = 0.07
DEFAULT_CUSTOM_SCENE_MIN_ROBOT_CLEARANCE_XY = 0.12


def available_source_model_names() -> list[str]:
    return sorted(MORE_CARROT_MODEL_DB.keys())


def source_model_names_for_obj_set(obj_set: str) -> list[str]:
    names = available_source_model_names()
    normalized = obj_set.strip().lower()
    if normalized == "all":
        return names
    if normalized == "train":
        train_names: list[str] = []
        for name in names:
            if name in TRAIN_SOURCE_MODEL_EXCLUDE:
                continue
            train_names.append(name)
            if len(train_names) >= min(TRAIN_SOURCE_MODEL_COUNT, len(names)):
                break
        return train_names
    if normalized == "test":
        train_names = set(source_model_names_for_obj_set("train"))
        return [name for name in names if name not in train_names]
    raise ValueError(f"Unknown obj_set: {obj_set}")


def source_object_name(model_name: str) -> str:
    entry = MORE_CARROT_MODEL_DB.get(model_name, {})
    if isinstance(entry, dict):
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    parts = [part for part in model_name.split("_") if part]
    if len(parts) >= 3 and parts[0].isdigit() and parts[-1].isdigit():
        object_name = " ".join(parts[1:-1]).strip()
        if object_name:
            return object_name
    return model_name


def resolve_language_instruction(
    model_name: str,
    *,
    plate_model_name: str = DEFAULT_PLATE_MODEL_NAME,
    template: str = DEFAULT_REAL2SIM_LANGUAGE_INSTRUCTION_TEMPLATE,
) -> str:
    return template.format(
        object_name=source_object_name(model_name),
        model_id=model_name,
        model_name=model_name,
        plate_model_name=plate_model_name,
    ).strip()


def _set_link_render_material(link, **kwargs) -> None:
    component = link.entity.find_component_by_type(sapien.render.RenderBodyComponent)
    if component is None:
        return
    for shape in component.render_shapes:
        if type(shape) == sapien.render.RenderShapeTriangleMesh:
            for part in shape.parts:
                # Imported meshes can carry texture maps that override scalar PBR knobs
                # like metallic / roughness, so clear them before applying overrides.
                for clear_tex in (
                    "set_base_color_texture",
                    "set_normal_texture",
                    "set_emission_texture",
                    "set_transmission_texture",
                    "set_metallic_texture",
                    "set_roughness_texture",
                ):
                    if hasattr(part.material, clear_tex):
                        getattr(part.material, clear_tex)(None)
                sapien_utils.set_render_material(part.material, **kwargs)


def _set_actor_render_material(actor, *, clear_textures: bool = False, **kwargs) -> None:
    entities = actor._objs if hasattr(actor, "_objs") else [actor]
    for entity in entities:
        component = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if component is None:
            continue
        for shape in component.render_shapes:
            if type(shape) == sapien.render.RenderShapeTriangleMesh:
                for part in shape.parts:
                    if clear_textures:
                        for clear_tex in (
                            "set_base_color_texture",
                            "set_normal_texture",
                            "set_emission_texture",
                            "set_transmission_texture",
                            "set_metallic_texture",
                            "set_roughness_texture",
                        ):
                            if hasattr(part.material, clear_tex):
                                getattr(part.material, clear_tex)(None)
                    sapien_utils.set_render_material(part.material, **kwargs)


def _apply_rc5_robot_materials(articulation) -> None:
    arm_link_names = {f"body{i}" for i in range(6)}
    hand_link_names = {
        "body6",
        "right_base_link",
        "right_t_link",
        "right_thumb_mcp_link",
        "right_thumb_proximal_link",
        "right_thumb_distal_link",
        "right_thumb_tip_link",
        "right_index_proximal_link",
        "right_index_middle_link",
        "right_index_distal_link",
        "right_index_tip_link",
        "right_middle_proximal_link",
        "right_middle_middle_link",
        "right_middle_distal_link",
        "right_middle_tip_link",
        "right_ring_proximal_link",
        "right_ring_middle_link",
        "right_ring_distal_link",
        "right_ring_tip_link",
        "right_pinky_proximal_link",
        "right_pinky_middle_link",
        "right_pinky_distal_link",
        "right_pinky_tip_link",
    }
    green_link_names = {"prehand"}

    for link in articulation.get_links():
        name = link.name
        if name in arm_link_names:
            _set_link_render_material(
                link,
                color=ROBOT_ARM_METAL_RGBA,
                emission=[0.0, 0.0, 0.0, 1.0],
                metallic=0.00,
                roughness=0.22,
                specular=0.95,
            )
        elif name in green_link_names:
            _set_link_render_material(
                link,
                color=ROBOT_PREHAND_GREEN_RGBA,
                emission=[0.0, 0.0, 0.0, 1.0],
                metallic=0.00,
                roughness=0.30,
                specular=0.65,
            )
        elif name in hand_link_names:
            _set_link_render_material(
                link,
                color=ROBOT_HAND_LIGHT_RGBA,
                emission=[0.0, 0.0, 0.0, 1.0],
                metallic=0.00,
                roughness=0.34,
                specular=0.60,
            )


def _model_paths(asset_group: str, model_name: str) -> tuple[Path, Path]:
    model_dir = MANISKILL_REPO_ASSET_DIR / "carrot" / asset_group / model_name
    visual_candidates = [model_dir / "textured.dae", model_dir / "textured.glb"]
    visual_path = next((p for p in visual_candidates if p.exists()), None)
    if visual_path is None:
        raise FileNotFoundError(f"No visual mesh found for {asset_group}/{model_name}")
    collision_path = model_dir / "collision.obj"
    if not collision_path.exists():
        raise FileNotFoundError(f"No collision mesh found for {asset_group}/{model_name}")
    return visual_path, collision_path


def source_model_paths(model_name: str) -> tuple[Path, Path]:
    return _model_paths("more_carrot", model_name)


def plate_model_paths(model_name: str = DEFAULT_PLATE_MODEL_NAME) -> tuple[Path, Path]:
    return _model_paths("more_plate", model_name)


def _bbox_corners(model_info: dict) -> np.ndarray:
    bbox = model_info["bbox"]
    bbox_min = np.array(bbox["min"], dtype=np.float32)
    bbox_max = np.array(bbox["max"], dtype=np.float32)
    xs = [bbox_min[0], bbox_max[0]]
    ys = [bbox_min[1], bbox_max[1]]
    zs = [bbox_min[2], bbox_max[2]]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
    return corners


def resting_center_z_offset_from_bbox(model_info: dict, quat_wxyz) -> float:
    corners = _bbox_corners(model_info)
    rot = quat2mat(np.array(quat_wxyz, dtype=np.float64))
    rotated = corners @ rot.T
    return float(-rotated[:, 2].min())


def source_resting_center_z_offset(quat_wxyz, model_name: str = DEFAULT_SOURCE_MODEL_NAME) -> float:
    return resting_center_z_offset_from_bbox(MORE_CARROT_MODEL_DB[model_name], quat_wxyz)


def plate_resting_center_z_offset(quat_wxyz, model_name: str = DEFAULT_PLATE_MODEL_NAME) -> float:
    return resting_center_z_offset_from_bbox(MORE_PLATE_MODEL_DB[model_name], quat_wxyz)


def centered_xy_half_extent_from_bbox(model_info: dict, quat_wxyz) -> float:
    corners = _bbox_corners(model_info)
    rot = quat2mat(np.array(quat_wxyz, dtype=np.float64))
    rotated = corners @ rot.T
    xy = np.abs(rotated[:, :2])
    return float(np.max(xy))


def source_xy_half_extent(quat_wxyz, model_name: str = DEFAULT_SOURCE_MODEL_NAME) -> float:
    return centered_xy_half_extent_from_bbox(MORE_CARROT_MODEL_DB[model_name], quat_wxyz)


def plate_xy_half_extent(quat_wxyz, model_name: str = DEFAULT_PLATE_MODEL_NAME) -> float:
    return centered_xy_half_extent_from_bbox(MORE_PLATE_MODEL_DB[model_name], quat_wxyz)


def resolve_scene_asset_dir(scene_asset_dir: str | Path | None = None) -> Path:
    if scene_asset_dir is None:
        return OPENREAL2SIM_ASSET_DIR
    path = Path(scene_asset_dir).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def scene_asset_paths(scene_asset_dir: str | Path | None = None) -> tuple[Path, Path, Path]:
    asset_dir = resolve_scene_asset_dir(scene_asset_dir)
    return (
        asset_dir / "scene.json",
        asset_dir / "background_registered.glb",
        asset_dir / "background_registered_collision.glb",
    )


def _load_scene_json(scene_asset_dir: str | Path | None = None) -> dict:
    scene_json_path, _, _ = scene_asset_paths(scene_asset_dir)
    with scene_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scaled_intrinsics(cam: dict) -> np.ndarray:
    sx = SENSOR_WIDTH / float(cam["width"])
    sy = SENSOR_HEIGHT / float(cam["height"])
    return np.array(
        [
            [float(cam["fx"]) * sx, 0.0, float(cam["cx"]) * sx],
            [0.0, float(cam["fy"]) * sy, float(cam["cy"]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _camera_pose_candidates(scene_asset_dir: str | Path | None = None) -> dict[str, sapien.Pose]:
    scene = _load_scene_json(scene_asset_dir)
    cam = scene["camera"]
    transform = np.array(cam["camera_opencv_to_world"], dtype=np.float64)
    rotation = transform[:3, :3]
    position = transform[:3, 3]

    cv_flip_yz = np.diag([1.0, -1.0, -1.0])
    cv_to_sapien_post = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cv_to_sapien_post_inv = cv_to_sapien_post.T

    candidates = {
        "raw": rotation,
        "post_cv_flip_yz": rotation @ cv_flip_yz,
        "post_cv_to_sapien": rotation @ cv_to_sapien_post,
        "post_cv_to_sapien_inv": rotation @ cv_to_sapien_post_inv,
        "pre_cv_to_sapien": cv_to_sapien_post @ rotation,
    }
    return {
        name: sapien.Pose(p=position, q=mat2quat(rot))
        for name, rot in candidates.items()
    }


def _camera_heading_pose(scene_asset_dir: str | Path | None = None) -> sapien.Pose:
    scene = _load_scene_json(scene_asset_dir)
    cam = scene["camera"]
    return sapien.Pose(
        p=np.array(cam["camera_position"], dtype=np.float64),
        q=np.array(cam["camera_heading_wxyz"], dtype=np.float64),
    )


def _scene_camera_pose(scene_asset_dir: str | Path | None = None) -> sapien.Pose:
    return _camera_pose_candidates(scene_asset_dir)["post_cv_to_sapien"]


def _camera_position(scene_asset_dir: str | Path | None = None) -> np.ndarray:
    scene = _load_scene_json(scene_asset_dir)
    return np.array(scene["camera"]["camera_position"], dtype=np.float32)


def scene_ground_point(scene_asset_dir: str | Path | None = None) -> np.ndarray:
    scene = _load_scene_json(scene_asset_dir)
    return np.array(scene["groundplane_in_sim"]["point"], dtype=np.float32)


def scene_camera_vertical_fov(scene_asset_dir: str | Path | None = None) -> float:
    scene = _load_scene_json(scene_asset_dir)
    cam = scene["camera"]
    return float(2.0 * np.arctan(float(cam["height"]) / (2.0 * float(cam["fy"]))))


def _scene_camera_lookat_state(scene_asset_dir: str | Path | None = None) -> dict[str, np.ndarray | float]:
    pose = _scene_camera_pose(scene_asset_dir)
    eye = np.array(pose.p, dtype=np.float32)
    rot = quat2mat(np.array(pose.q, dtype=np.float64))
    forward = rot[:, 0].astype(np.float32)
    target = eye + forward

    base_pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
    base_rot = quat2mat(base_pose.sp.q)
    rel_rot = base_rot.T @ rot
    roll_deg = float(np.rad2deg(np.arctan2(rel_rot[2, 1], rel_rot[1, 1])))

    return {
        "eye": eye,
        "target": target,
        "fov": scene_camera_vertical_fov(scene_asset_dir),
        "roll_deg": roll_deg,
    }


def pose_from_eye_target_roll(eye: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> sapien.Pose:
    base_pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
    if abs(roll_deg) < 1e-8:
        return base_pose
    rot = quat2mat(base_pose.sp.q)
    roll_local = axangle2mat([1.0, 0.0, 0.0], np.deg2rad(roll_deg))
    rolled_rot = rot @ roll_local
    return sapien.Pose(p=np.array(base_pose.sp.p), q=mat2quat(rolled_rot))


def observation_camera_lookat_state(
    mode: str,
    scene_asset_dir: str | Path | None = None,
) -> dict[str, np.ndarray | float]:
    if mode in {"manual_best", "raw"}:
        return _scene_camera_lookat_state(scene_asset_dir)

    debug_points = _scene_debug_points(scene_asset_dir)
    camera_pos = _camera_position(scene_asset_dir)
    center = debug_points["aabb_center"]
    ground = debug_points["ground"]
    eye_to_center = center - camera_pos
    eye_to_center_norm = np.linalg.norm(eye_to_center)
    if eye_to_center_norm < 1e-6:
        eye_to_center = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        eye_to_center_norm = 1.0
    eye_to_center = eye_to_center / eye_to_center_norm
    backward = -eye_to_center
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if mode == "position_ground_lookat":
        return {
            "eye": camera_pos,
            "target": ground,
            "fov": 1.0,
            "roll_deg": 0.0,
        }
    if mode == "position_center_lookat":
        return {
            "eye": camera_pos,
            "target": center,
            "fov": 1.0,
            "roll_deg": 0.0,
        }
    preset_targets = {
        "preset_ground": ground,
        "preset_center": center,
        "preset_center_up": center + np.array([0.0, 0.0, 0.20], dtype=np.float32),
        "preset_center_up_high": center + np.array([0.0, 0.0, 0.35], dtype=np.float32),
    }
    preset_eyes = {
        "preset_ground": camera_pos,
        "preset_center": camera_pos,
        "preset_center_up": camera_pos,
        "preset_center_up_high": camera_pos,
        "preset_center_back": camera_pos + 0.25 * backward,
        "preset_center_back_up": camera_pos + 0.25 * backward + 0.15 * up,
        "preset_center_far_back_up": camera_pos + 0.45 * backward + 0.25 * up,
    }
    if mode in preset_eyes:
        return {
            "eye": preset_eyes[mode],
            "target": preset_targets.get(mode, center),
            "fov": 1.0,
            "roll_deg": 0.0,
        }
    raise ValueError(f"Unknown look-at observation mode: {mode}")


def observation_camera_override(mode: str, scene_asset_dir: str | Path | None = None) -> dict:
    scene = _load_scene_json(scene_asset_dir)
    cam = scene["camera"]

    if mode == "raw":
        pose = _scene_camera_pose(scene_asset_dir)
        return {
            "pose": pose,
            "intrinsic": _scaled_intrinsics(cam),
            "fov": None,
            "near": 0.01,
            "far": 10.0,
        }
    lookat = observation_camera_lookat_state(mode, scene_asset_dir=scene_asset_dir)
    pose = pose_from_eye_target_roll(
        eye=lookat["eye"],
        target=lookat["target"],
        roll_deg=float(lookat.get("roll_deg", 0.0)),
    )
    return {
        "pose": pose,
        "intrinsic": None,
        "fov": float(lookat["fov"]),
        "near": 0.01,
        "far": 10.0,
    }


def _scene_debug_points(scene_asset_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    scene = _load_scene_json(scene_asset_dir)
    ground = np.array(scene["groundplane_in_sim"]["point"], dtype=np.float32)
    ground_normal = np.array(scene["groundplane_in_sim"]["normal"], dtype=np.float32)
    aabb_min = np.array(scene["aabb"]["scene_min"], dtype=np.float32)
    aabb_max = np.array(scene["aabb"]["scene_max"], dtype=np.float32)
    aabb_center = (aabb_min + aabb_max) / 2.0
    return {
        "ground": ground,
        "ground_normal": ground_normal,
        "aabb_center": aabb_center,
    }


def default_spawn_grid_state(scene_asset_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    pts = _scene_debug_points(scene_asset_dir)
    asset_dir = resolve_scene_asset_dir(scene_asset_dir)
    if asset_dir == AIRI_TABLE_EMPTY_ASSET_DIR.resolve():
        center_xy = AIRI_TABLE_EMPTY_SPAWN_GRID_CENTER_XY.copy()
        size_xy = AIRI_TABLE_EMPTY_SPAWN_GRID_SIZE_XY.copy()
        yaw_deg = AIRI_TABLE_EMPTY_SPAWN_GRID_YAW_DEG
    elif asset_dir == AIRI_CUBES_ASSET_DIR.resolve():
        center_xy = AIRI_CUBES_SPAWN_GRID_CENTER_XY.copy()
        size_xy = AIRI_CUBES_SPAWN_GRID_SIZE_XY.copy()
        yaw_deg = AIRI_CUBES_SPAWN_GRID_YAW_DEG
    else:
        center_xy = OPENREAL2SIM_SPAWN_GRID_CENTER_XY.copy()
        size_xy = OPENREAL2SIM_SPAWN_GRID_SIZE_XY.copy()
        yaw_deg = OPENREAL2SIM_SPAWN_GRID_YAW_DEG
    return {
        "center_xy": center_xy,
        "size_xy": size_xy,
        "yaw_deg": np.array([float(yaw_deg)], dtype=np.float32),
        "z": np.array([float(pts["ground"][2])], dtype=np.float32),
        "ground_normal": pts["ground_normal"].astype(np.float32),
    }


def default_shadow_catcher_state(scene_asset_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    asset_dir = resolve_scene_asset_dir(scene_asset_dir)
    if asset_dir == AIRI_TABLE_EMPTY_ASSET_DIR.resolve():
        center_xy = AIRI_TABLE_EMPTY_SHADOW_CATCHER_CENTER_XY.copy()
        size_xy = AIRI_TABLE_EMPTY_SHADOW_CATCHER_SIZE_XY.copy()
        yaw_deg = AIRI_TABLE_EMPTY_SHADOW_CATCHER_YAW_DEG
    else:
        grid = default_spawn_grid_state(scene_asset_dir)
        center_xy = np.array(grid["center_xy"], dtype=np.float32)
        size_xy = np.array(grid["size_xy"], dtype=np.float32) + np.array([0.07, 0.06], dtype=np.float32)
        yaw_deg = float(grid["yaw_deg"][0])
    return {
        "center_xy": np.array(center_xy, dtype=np.float32),
        "size_xy": np.array(size_xy, dtype=np.float32),
        "yaw_deg": np.array([float(yaw_deg)], dtype=np.float32),
    }


def airi_cube_grid_xy_positions(
    rng: np.random.Generator,
    *,
    randomize: bool = True,
    jitter_xy: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not randomize:
        return {
            color_name: np.asarray(cube_cfg["xy"], dtype=np.float32).copy()
            for color_name, cube_cfg in AIRI_CUBE_CONFIGS.items()
        }
    rect_min = AIRI_CUBE_SPAWN_RECT_MIN_XY.astype(np.float32)
    rect_max = AIRI_CUBE_SPAWN_RECT_MAX_XY.astype(np.float32)
    grid_x = np.linspace(rect_min[0], rect_max[0], AIRI_CUBE_SPAWN_GRID_COLS, dtype=np.float32)
    grid_y = np.linspace(rect_min[1], rect_max[1], AIRI_CUBE_SPAWN_GRID_ROWS, dtype=np.float32)
    grid_spacing = np.array(
        [
            0.0 if AIRI_CUBE_SPAWN_GRID_COLS <= 1 else float(grid_x[1] - grid_x[0]),
            0.0 if AIRI_CUBE_SPAWN_GRID_ROWS <= 1 else float(grid_y[1] - grid_y[0]),
        ],
        dtype=np.float32,
    )
    grid_ids = rng.choice(
        AIRI_CUBE_SPAWN_GRID_COLS * AIRI_CUBE_SPAWN_GRID_ROWS,
        size=len(AIRI_CUBE_COLOR_NAMES),
        replace=False,
    )
    jitter = AIRI_CUBE_SPAWN_JITTER_XY if jitter_xy is None else np.asarray(jitter_xy, dtype=np.float32)
    max_jitter = np.maximum(np.minimum(jitter, grid_spacing * 0.35), 0.0)
    positions: dict[str, np.ndarray] = {}
    for color_name, grid_id in zip(AIRI_CUBE_COLOR_NAMES, grid_ids):
        col = int(grid_id) % AIRI_CUBE_SPAWN_GRID_COLS
        row = int(grid_id) // AIRI_CUBE_SPAWN_GRID_COLS
        center = np.array([grid_x[col], grid_y[row]], dtype=np.float32)
        offset = rng.uniform(-max_jitter, max_jitter).astype(np.float32)
        positions[color_name] = np.clip(center + offset, rect_min, rect_max).astype(np.float32)
    return positions


def resolve_episode_ids(options: dict, num_envs: int, default: int = 0) -> np.ndarray:
    episode_id = options.get("episode_id", default)
    if torch.is_tensor(episode_id):
        if episode_id.numel() == 0:
            return np.full((num_envs,), int(default), dtype=np.int64)
        values = episode_id.detach().cpu().reshape(-1).to(torch.int64).numpy()
    elif isinstance(episode_id, np.ndarray):
        if episode_id.size == 0:
            return np.full((num_envs,), int(default), dtype=np.int64)
        values = episode_id.reshape(-1).astype(np.int64, copy=False)
    elif isinstance(episode_id, (list, tuple)):
        if len(episode_id) == 0:
            return np.full((num_envs,), int(default), dtype=np.int64)
        values = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    else:
        values = np.asarray([int(episode_id)], dtype=np.int64)
    if values.size == 1 and num_envs > 1:
        values = np.full((num_envs,), int(values[0]), dtype=np.int64)
    return values


def _load_spawn_grid_state(path: str | Path | None, scene_asset_dir: str | Path | None) -> dict[str, np.ndarray]:
    if path is None or str(path).strip() == "":
        return default_spawn_grid_state(scene_asset_dir)
    state: dict[str, object] = {}
    resolved_path = resolve_spawn_grid_state_path(path)
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key.strip()] = ast.literal_eval(value.strip())
    return {
        "center_xy": np.array(state["center_xy"], dtype=np.float32),
        "size_xy": np.array(state["size_xy"], dtype=np.float32),
        "yaw_deg": np.array([float(state.get("yaw_deg", 0.0))], dtype=np.float32),
        "z": np.array([float(state["z"])], dtype=np.float32),
    }


def _make_rotated_grid(
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    yaw_deg: float,
    steps_x: int,
    steps_y: int,
) -> np.ndarray:
    xs = np.linspace(center_xy[0] - size_xy[0] / 2.0, center_xy[0] + size_xy[0] / 2.0, steps_x)
    ys = np.linspace(center_xy[1] - size_xy[1] / 2.0, center_xy[1] + size_xy[1] / 2.0, steps_y)
    yaw = np.deg2rad(float(yaw_deg))
    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    points = []
    for x in xs:
        for y in ys:
            local = np.array([x - center_xy[0], y - center_xy[1]], dtype=np.float32)
            rotated = rot @ local
            points.append([float(center_xy[0] + rotated[0]), float(center_xy[1] + rotated[1])])
    return np.array(points, dtype=np.float32)


def _valid_spawn_pairs(
    *,
    scene_asset_dir: str | Path | None,
    spawn_grid_state_file: str | Path | None,
    robot_base_pose_p: np.ndarray,
    source_model_name: str,
    plate_model_name: str,
    steps_x: int = DEFAULT_CUSTOM_SCENE_GRID_STEPS_X,
    steps_y: int = DEFAULT_CUSTOM_SCENE_GRID_STEPS_Y,
    min_pair_distance_xy: float = DEFAULT_CUSTOM_SCENE_MIN_PAIR_DISTANCE_XY,
    min_robot_clearance_xy: float = DEFAULT_CUSTOM_SCENE_MIN_ROBOT_CLEARANCE_XY,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grid = _load_spawn_grid_state(spawn_grid_state_file, scene_asset_dir)
    center_xy = np.array(grid["center_xy"], dtype=np.float32)
    size_xy = np.array(grid["size_xy"], dtype=np.float32)
    yaw_deg = float(grid["yaw_deg"][0])
    footprint_margin = max(
        source_xy_half_extent(PROBE_CARROT_POSE_Q, model_name=source_model_name),
        plate_xy_half_extent(PROBE_PLATE_POSE_Q, model_name=plate_model_name),
    )
    usable_size_xy = np.maximum(size_xy - 2.0 * footprint_margin, 0.01)
    grid_points = _make_rotated_grid(center_xy, usable_size_xy, yaw_deg=yaw_deg, steps_x=steps_x, steps_y=steps_y)
    robot_base_xy = np.array(robot_base_pose_p[:2], dtype=np.float32)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for source_xy in grid_points:
        for plate_xy in grid_points:
            if np.allclose(source_xy, plate_xy):
                continue
            if np.linalg.norm(plate_xy - source_xy) <= float(min_pair_distance_xy):
                continue
            if np.linalg.norm(source_xy - robot_base_xy) <= float(min_robot_clearance_xy):
                continue
            if np.linalg.norm(plate_xy - robot_base_xy) <= float(min_robot_clearance_xy):
                continue
            pairs.append((source_xy.copy(), plate_xy.copy()))
    return pairs


@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SOpenReal2SimValidation(WidowX250SBridgeDatasetFlatTable):
    uid = "widowx250s_openreal2sim_validation"

    @property
    def _sensor_configs(self):
        # No robot-mounted observation camera here. The validation env injects a
        # static world camera through env-level sensor configs instead.
        return []


@register_agent()
class RC5AeroHandRightOpenReal2SimValidation(RC5AeroHandRight):
    uid = "rc5_aero_hand_right_openreal2sim_validation"
    # Softer arm dynamics for teleop so incidental fingertip contacts do not
    # transfer large impulses into lightweight objects.
    arm_stiffness = [900.0, 900.0, 720.0, 450.0, 320.0, 220.0]
    arm_damping = [240.0, 240.0, 185.0, 115.0, 78.0, 54.0]
    arm_force_limit = [180.0, 180.0, 140.0, 90.0, 60.0, 45.0]
    # Softer hand drives for teleop/contact tuning so small objects do not get
    # launched by rigid finger closure.
    gripper_stiffness = 1200.0
    gripper_damping = 140.0
    gripper_force_limit = 320.0
    # Match the teleop controller's open/grab hand shapes to the saved debug
    # hand poses so the controller-based grasp uses the intended posture.
    gripper_open_qpos = RC5_HAND_OPEN_QPOS.copy()
    gripper_closed_qpos = RC5_HAND_GRAB_QPOS.copy()

    @property
    def _sensor_configs(self):
        return []


def _rc5_wrist_camera_config(agent) -> CameraConfig | None:
    if agent is None or not hasattr(agent, "robot"):
        return None
    mount = agent.robot.links_map.get(WRIST_CAMERA_MOUNT_LINK)
    if mount is None:
        return None
    return CameraConfig(
        uid=WRIST_CAMERA_NAME,
        pose=sapien.Pose(p=WRIST_CAMERA_LOCAL_P, q=WRIST_CAMERA_LOCAL_Q),
        width=WRIST_CAMERA_WIDTH,
        height=WRIST_CAMERA_HEIGHT,
        fov=WRIST_CAMERA_FOV,
        near=WRIST_CAMERA_NEAR,
        far=WRIST_CAMERA_FAR,
        mount=mount,
    )


@register_env("OpenReal2SimValidation-v1", max_episode_steps=120)
class OpenReal2SimValidationEnv(BaseEnv):
    SUPPORTED_OBS_MODES = ["rgb+segmentation", "sensor_data"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(
        self,
        *args,
        robot_uids="widowx250s_openreal2sim_validation",
        scene_asset_dir: str | Path | None = None,
        **kwargs,
    ):
        self.use_probe_objects = True
        self.load_background = True
        self.show_debug_markers = False
        self.show_spawn_grid = False
        self.robot_far_away = False
        self.apply_robot_material_overrides = True
        self.safe_robot_during_spawn = True
        self.probe_carrot = None
        self.probe_carrots = {}
        self.probe_plate = None
        self.use_airi_cubes = False
        self.airi_cube_actors = {}
        self.airi_cube_target_color = AIRI_CUBE_COLOR_NAMES[0]
        self.airi_cube_target_colors_by_env = [self.airi_cube_target_color]
        self.airi_cube_initial_center_z = 0.0
        self.shadow_catcher = None
        self.debug_markers = []
        self.spawn_grid_markers = []
        self.tcp_calibration_marker = None
        self.probe_carrot_pose_p = np.array(PROBE_CARROT_POSE_P, dtype=np.float32)
        self.probe_carrot_pose_q = np.array(PROBE_CARROT_POSE_Q, dtype=np.float32)
        self.probe_plate_pose_p = np.array(PROBE_PLATE_POSE_P, dtype=np.float32)
        self.probe_plate_pose_q = np.array(PROBE_PLATE_POSE_Q, dtype=np.float32)
        self.probe_source_model_name = DEFAULT_SOURCE_MODEL_NAME
        self.probe_source_model_names = [DEFAULT_SOURCE_MODEL_NAME]
        self.probe_source_model_names_by_env = [DEFAULT_SOURCE_MODEL_NAME]
        self.probe_plate_model_name = DEFAULT_PLATE_MODEL_NAME
        self.language_instruction_template = DEFAULT_REAL2SIM_LANGUAGE_INSTRUCTION_TEMPLATE
        self.robot_base_pose_p = np.array(ROBOT_BASE_POSE_P, dtype=np.float32)
        self.robot_base_pose_q = np.array(ROBOT_BASE_POSE_Q, dtype=np.float32)
        self.robot_init_qpos = np.array(ROBOT_INIT_QPOS, dtype=np.float32)
        self.scene_asset_dir = resolve_scene_asset_dir(scene_asset_dir)
        self.use_wrist_camera = bool(kwargs.pop("use_wrist_camera", DEFAULT_USE_WRIST_CAMERA))
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    @property
    def _default_sensor_configs(self):
        cfg = observation_camera_override("manual_best", scene_asset_dir=self.scene_asset_dir)
        configs = [
            CameraConfig(
                uid="3rd_view_camera",
                pose=cfg["pose"],
                width=SENSOR_WIDTH,
                height=SENSOR_HEIGHT,
                fov=cfg.get("fov", None),
                intrinsic=cfg.get("intrinsic", None),
                near=cfg.get("near", 0.01),
                far=cfg.get("far", 10.0),
            )
        ]
        if self.use_wrist_camera:
            wrist_cfg = _rc5_wrist_camera_config(self.agent)
            if wrist_cfg is not None:
                configs.append(wrist_cfg)
        return configs

    @property
    def _default_human_render_camera_configs(self):
        debug_points = _scene_debug_points(self.scene_asset_dir)
        target = debug_points["ground"]
        camera_pos = _camera_position(self.scene_asset_dir)
        sanity_pose = sapien_utils.look_at(
            eye=(target + np.array([0.8, -0.8, 0.6], dtype=np.float32)).tolist(),
            target=target.tolist(),
        )
        manual_best = observation_camera_lookat_state(
            "manual_best", scene_asset_dir=self.scene_asset_dir
        )
        manual_best_pose = pose_from_eye_target_roll(
            eye=manual_best["eye"],
            target=manual_best["target"],
            roll_deg=float(manual_best.get("roll_deg", 0.0)),
        )
        return [
            CameraConfig(
                "render_camera",
                pose=manual_best_pose,
                width=SENSOR_WIDTH,
                height=SENSOR_HEIGHT,
                fov=1.0,
                near=0.01,
                far=10.0,
            ),
            CameraConfig(
                "sanity_render_camera",
                pose=sanity_pose,
                width=SENSOR_WIDTH,
                height=SENSOR_HEIGHT,
                fov=1.0,
                near=0.01,
                far=10.0,
            ),
        ]

    def _load_lighting(self, options: dict):
        ground = scene_ground_point(self.scene_asset_dir)
        light_focus = np.array(
            [float(ground[0]), float(ground[1]), float(ground[2] + 0.32)],
            dtype=np.float32,
        )
        robot_base = np.array(self.robot_base_pose_p, dtype=np.float32)
        shadow_light_position = np.array(
            [
                float(robot_base[0] + 0.06),
                float(robot_base[1] - 0.02),
                float(ground[2] + 2.15),
            ],
            dtype=np.float32,
        )

        # The HDR environment map adds a lot of flat fill light. Keep it optional
        # so the VR/headset path can use a drier, higher-contrast render.
        if bool(options.get("use_environment_map", True)) and REPLICACAD_HDR_ENV_MAP.exists():
            for sub_scene in self.scene.sub_scenes:
                sub_scene.set_environment_map(str(REPLICACAD_HDR_ENV_MAP))

        # Stay close to the standard ManiSkill bridge-task lighting, just slightly dimmer.
        # Use a focused overhead spotlight for the actual table shadow. In practice this
        # is more reliable than a point-light shadow for the current render path.
        self.scene.set_ambient_light([0.14, 0.14, 0.14])
        self.scene.add_spot_light(
            shadow_light_position.tolist(),
            (light_focus - shadow_light_position).tolist(),
            inner_fov=0.45,
            outer_fov=0.8,
            color=[3.1, 3.1, 3.1],
            shadow=True,
            shadow_near=0.1,
            shadow_far=3.0,
            shadow_map_size=4096,
        )
        self.scene.add_directional_light(
            [0, 0, -1],
            [1.15, 1.15, 1.15],
            shadow=False,
            position=(light_focus + np.array([0.0, 0.0, 1.0], dtype=np.float32)).tolist(),
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light(
            [-0.8, 0.25, -0.55],
            [0.28, 0.28, 0.28],
            shadow=False,
            position=(light_focus + np.array([0.4, -0.2, 0.8], dtype=np.float32)).tolist(),
        )
        self.scene.add_directional_light(
            [1, 1, -1],
            [0.28, 0.28, 0.28],
            shadow=False,
            position=(light_focus + np.array([-0.4, 0.1, 0.7], dtype=np.float32)).tolist(),
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=self.robot_base_pose_p.tolist(), q=self.robot_base_pose_q.tolist()),
        )

    def _build_mesh_actor(
        self,
        name: str,
        visual_path: Path,
        collision_path: Path,
        *,
        kinematic: bool,
        density: float = 1000.0,
    ):
        builder = self.scene.create_actor_builder()
        material = PhysxMaterial(static_friction=0.8, dynamic_friction=0.8, restitution=0.0)
        builder.add_multiple_convex_collisions_from_file(
            filename=str(collision_path),
            material=material,
            density=density,
        )
        builder.add_visual_from_file(filename=str(visual_path))
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        if kinematic:
            return builder.build_kinematic(name=name)
        return builder.build(name=name)

    def _build_centered_mesh_actor(
        self,
        name: str,
        visual_path: Path,
        collision_path: Path,
        *,
        local_center: np.ndarray,
        density: float = 250.0,
    ):
        builder = self.scene.create_actor_builder()
        material = PhysxMaterial(static_friction=0.9, dynamic_friction=0.8, restitution=0.0)
        local_pose = sapien.Pose(p=(-np.asarray(local_center, dtype=np.float32)).tolist())
        del collision_path
        builder.add_box_collision(
            half_size=AIRI_CUBE_HALF_SIZE.tolist(),
            material=material,
            density=density,
        )
        builder.add_visual_from_file(filename=str(visual_path), pose=local_pose)
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        return builder.build(name=name)

    @staticmethod
    def _normalize_model_name_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, np.ndarray):
            return [str(v) for v in value.reshape(-1).tolist()]
        if torch.is_tensor(value):
            return [str(v) for v in value.detach().cpu().reshape(-1).tolist()]
        return [str(v) for v in value]

    @staticmethod
    def _normalize_cube_color_list(value: object) -> list[str]:
        colors = OpenReal2SimValidationEnv._normalize_model_name_list(value)
        normalized = [color.strip().lower() for color in colors if color.strip()]
        unknown = sorted(set(normalized) - set(AIRI_CUBE_COLOR_NAMES))
        if unknown:
            raise ValueError(f"Unknown AIRI cube colors {unknown}. Available: {list(AIRI_CUBE_COLOR_NAMES)}")
        return normalized

    def _source_actor_name(self, model_name: str, index: int) -> str:
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in model_name)
        return f"probe_source_{index:02d}_{safe_name}"

    def _load_scene(self, options: dict):
        self.apply_robot_material_overrides = bool(
            options.get("apply_robot_material_overrides", self.apply_robot_material_overrides)
        )
        for i in range(self.num_envs):
            self.agent.robot._objs[i].set_name(f"robot_{i}")
            if self.apply_robot_material_overrides:
                sapien_utils.set_articulation_render_material(
                    self.agent.robot._objs[i],
                    specular=0.9,
                    roughness=0.3,
                )

        self.load_background = bool(options.get("load_background", True))
        self.show_debug_markers = bool(options.get("show_debug_markers", False))
        self.show_spawn_grid = bool(options.get("show_spawn_grid", False))
        self.robot_far_away = bool(options.get("robot_far_away", False))
        self.safe_robot_during_spawn = bool(options.get("safe_robot_during_spawn", True))
        self.use_shadow_catcher = bool(options.get("use_shadow_catcher", False))
        self.tcp_calibration_marker = None

        if self.load_background:
            _, background_visual_path, background_collision_path = scene_asset_paths(
                self.scene_asset_dir
            )
            builder = self.scene.create_actor_builder()
            builder.add_nonconvex_collision_from_file(str(background_collision_path))
            builder.add_visual_from_file(str(background_visual_path))
            builder.initial_pose = sapien.Pose()
            arena = builder.build_static(name="arena")
            # Keep the scanned texture, but make the surface much less glossy so
            # the spotlight shadow reads on the real table instead of requiring a
            # synthetic receiver patch.
            _set_actor_render_material(
                arena,
                metallic=0.0,
                roughness=1.0,
                specular=0.02,
                emission=[0.0, 0.0, 0.0, 1.0],
            )
            # Add a thin invisible support plane at the calibrated tabletop
            # height so finger contacts cannot sink through visual/collision
            # mismatches in the scanned background mesh.
            builder = self.scene.create_actor_builder()
            tabletop_half_thickness = 0.003
            builder.add_box_collision(half_size=[0.9, 0.9, tabletop_half_thickness])
            ground_point = scene_ground_point(self.scene_asset_dir)
            builder.initial_pose = sapien.Pose(
                p=[
                    float(ground_point[0]),
                    float(ground_point[1]),
                    float(ground_point[2] - tabletop_half_thickness),
                ]
            )
            builder.build_static(name="tabletop_collision_guard")
            if self.use_shadow_catcher:
                shadow_state = default_shadow_catcher_state(self.scene_asset_dir)
                center_xy = np.array(shadow_state["center_xy"], dtype=np.float32)
                size_xy = np.array(shadow_state["size_xy"], dtype=np.float32)
                shadow_half_size_xy = np.maximum(size_xy / 2.0, 0.09)
                shadow_yaw_deg = float(shadow_state["yaw_deg"][0])
                shadow_yaw_quat = mat2quat(axangle2mat([0.0, 0.0, 1.0], np.deg2rad(shadow_yaw_deg))).tolist()
                shadow_catcher_z = float(ground_point[2] + 0.00025)
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[
                        float(shadow_half_size_xy[0]),
                        float(shadow_half_size_xy[1]),
                        0.00025,
                    ],
                    material=sapien.render.RenderMaterial(
                        base_color=np.array([0.63, 0.70, 0.73, 1.0], dtype=np.float32),
                        metallic=0.0,
                        roughness=0.98,
                        specular=0.02,
                    ),
                )
                builder.initial_pose = sapien.Pose(
                    p=[
                        float(center_xy[0]),
                        float(center_xy[1]),
                        shadow_catcher_z,
                    ],
                    q=shadow_yaw_quat,
                )
                self.shadow_catcher = builder.build_static(name="tabletop_shadow_catcher")

        self.debug_markers = []
        if self.show_debug_markers:
            pts = _scene_debug_points(self.scene_asset_dir)
            ground = pts["ground"]
            center = pts["aabb_center"]
            marker_z = max(float(ground[2]) + 0.03, 0.03)
            self.debug_markers.append(
                actors.build_sphere(
                    self.scene,
                    radius=0.08,
                    color=np.array([1.0, 0.0, 0.0, 1.0]),
                    name="debug_ground_origin",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose([float(ground[0]), float(ground[1]), marker_z]),
                )
            )
            self.debug_markers.append(
                actors.build_cube(
                    self.scene,
                    half_size=0.08,
                    color=np.array([0.0, 1.0, 0.0, 1.0]),
                    name="debug_x_axis",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose([float(ground[0] + 0.15), float(ground[1]), marker_z]),
                )
            )
            self.debug_markers.append(
                actors.build_cube(
                    self.scene,
                    half_size=0.08,
                    color=np.array([0.0, 0.0, 1.0, 1.0]),
                    name="debug_y_axis",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose([float(ground[0]), float(ground[1] + 0.15), marker_z]),
                )
            )
            self.debug_markers.append(
                actors.build_cube(
                    self.scene,
                    half_size=0.08,
                    color=np.array([1.0, 1.0, 0.0, 1.0]),
                    name="debug_aabb_center",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(
                        [float(center[0]), float(center[1]), max(float(center[2]), marker_z + 0.08)]
                    ),
                )
            )

        self.spawn_grid_markers = []
        if self.show_spawn_grid:
            grid_state = default_spawn_grid_state(self.scene_asset_dir)
            center_xy = np.array(options.get("spawn_grid_center_xy", grid_state["center_xy"]), dtype=np.float32)
            size_xy = np.array(options.get("spawn_grid_size_xy", grid_state["size_xy"]), dtype=np.float32)
            yaw_deg = float(np.array(options.get("spawn_grid_yaw_deg", grid_state["yaw_deg"])).reshape(-1)[0])
            z = float(np.array(options.get("spawn_grid_z", grid_state["z"])).reshape(-1)[0] + 0.002)
            edge_thickness = 0.004
            edge_height = 0.003
            cx, cy = float(center_xy[0]), float(center_xy[1])
            sx = max(float(size_xy[0]), 0.005)
            sy = max(float(size_xy[1]), 0.005)
            hx, hy = sx / 2.0, sy / 2.0
            color = np.array([1.0, 0.15, 0.15, 1.0])
            yaw_rot = axangle2mat([0.0, 0.0, 1.0], np.deg2rad(yaw_deg))
            yaw_quat = mat2quat(yaw_rot)

            def _grid_pose(dx: float, dy: float) -> sapien.Pose:
                offset = yaw_rot @ np.array([dx, dy, 0.0], dtype=np.float64)
                return sapien.Pose([cx + float(offset[0]), cy + float(offset[1]), z], yaw_quat.tolist())

            self.spawn_grid_markers.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[hx, edge_thickness / 2, edge_height / 2],
                    color=color,
                    name="spawn_grid_top",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=_grid_pose(0.0, hy),
                )
            )
            self.spawn_grid_markers.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[hx, edge_thickness / 2, edge_height / 2],
                    color=color,
                    name="spawn_grid_bottom",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=_grid_pose(0.0, -hy),
                )
            )
            self.spawn_grid_markers.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[edge_thickness / 2, hy, edge_height / 2],
                    color=color,
                    name="spawn_grid_left",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=_grid_pose(-hx, 0.0),
                )
            )
            self.spawn_grid_markers.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[edge_thickness / 2, hy, edge_height / 2],
                    color=color,
                    name="spawn_grid_right",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=_grid_pose(hx, 0.0),
                )
            )

        if bool(options.get("create_tcp_calibration_marker", False)):
            marker_radius = float(options.get("tcp_calibration_marker_radius", 0.04))
            marker_color = np.array([1.0, 0.0, 0.0, 0.92], dtype=np.float32)
            self.tcp_calibration_marker = actors.build_sphere(
                self.scene,
                radius=marker_radius,
                color=marker_color,
                name="tcp_calibration_marker",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose([0.0, 0.0, -10.0]),
            )

        self.use_probe_objects = bool(options.get("use_probe_objects", True))
        self.use_airi_cubes = bool(options.get("use_airi_cubes", False))
        instruction_template = str(
            options.get("trajectory_instruction", options.get("language_instruction", self.language_instruction_template))
        ).strip()
        if instruction_template:
            self.language_instruction_template = instruction_template
        if self.use_probe_objects:
            source_model_names = self._normalize_model_name_list(options.get("probe_source_model_names"))
            if not source_model_names:
                source_model_names = [str(options.get("probe_source_model_name", self.probe_source_model_name))]
            # Preserve order while removing duplicates so deterministic source-index choices remain stable.
            source_model_names = list(dict.fromkeys(source_model_names))
            self.probe_source_model_names = source_model_names
            self.probe_source_model_name = source_model_names[0]
            self.probe_source_model_names_by_env = [self.probe_source_model_name] * self.num_envs
            self.probe_plate_model_name = str(
                options.get("probe_plate_model_name", self.probe_plate_model_name)
            )
            plate_visual_path, plate_collision_path = plate_model_paths(self.probe_plate_model_name)
            self.probe_carrots = {}
            for source_index, source_model_name in enumerate(source_model_names):
                source_visual_path, source_collision_path = source_model_paths(source_model_name)
                self.probe_carrots[source_model_name] = self._build_mesh_actor(
                    self._source_actor_name(source_model_name, source_index),
                    source_visual_path,
                    source_collision_path,
                    kinematic=False,
                    density=1000.0 * PROBE_SOURCE_DENSITY_SCALE,
                )
            self.probe_carrot = self.probe_carrots[self.probe_source_model_name]
            self.probe_plate = self._build_mesh_actor(
                "probe_plate",
                plate_visual_path,
                plate_collision_path,
                kinematic=True,
            )

        if self.use_airi_cubes:
            self.airi_cube_actors = {}
            for color_name, cube_cfg in AIRI_CUBE_CONFIGS.items():
                visual_path = Path(cube_cfg["visual"])
                collision_path = Path(cube_cfg["collision"])
                if not visual_path.exists():
                    raise FileNotFoundError(f"Missing AIRI cube visual mesh: {visual_path}")
                if not collision_path.exists():
                    raise FileNotFoundError(f"Missing AIRI cube collision mesh: {collision_path}")
                self.airi_cube_actors[color_name] = self._build_centered_mesh_actor(
                    f"airi_cube_{color_name}",
                    visual_path,
                    collision_path,
                    local_center=AIRI_CUBE_LOCAL_CENTER,
                    density=250.0,
                )

    def _settle(self, t: float = 0.5):
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            if "probe_carrot_pose_p" in options:
                self.probe_carrot_pose_p = np.array(options["probe_carrot_pose_p"], dtype=np.float32)
            if "probe_carrot_pose_q" in options:
                self.probe_carrot_pose_q = np.array(options["probe_carrot_pose_q"], dtype=np.float32)
            if "probe_source_model_names_by_env" in options:
                names_by_env = self._normalize_model_name_list(options["probe_source_model_names_by_env"])
                if len(names_by_env) == 1 and self.num_envs > 1:
                    names_by_env = names_by_env * self.num_envs
                if len(names_by_env) != self.num_envs:
                    raise ValueError(
                        "probe_source_model_names_by_env must have length num_envs "
                        f"({self.num_envs}), got {len(names_by_env)}."
                    )
                missing_names = sorted(set(names_by_env) - set(self.probe_carrots.keys()))
                if missing_names:
                    raise ValueError(f"Active source models were not loaded: {missing_names}")
                self.probe_source_model_names_by_env = names_by_env
                self.probe_source_model_name = names_by_env[0]
                self.probe_carrot = self.probe_carrots[self.probe_source_model_name]
            if "probe_plate_pose_p" in options:
                self.probe_plate_pose_p = np.array(options["probe_plate_pose_p"], dtype=np.float32)
            if "probe_plate_pose_q" in options:
                self.probe_plate_pose_q = np.array(options["probe_plate_pose_q"], dtype=np.float32)
            if "robot_base_pose_p" in options:
                self.robot_base_pose_p = np.array(options["robot_base_pose_p"], dtype=np.float32)
            if "robot_base_pose_q" in options:
                self.robot_base_pose_q = np.array(options["robot_base_pose_q"], dtype=np.float32)
            if "robot_init_qpos" in options:
                self.robot_init_qpos = np.array(options["robot_init_qpos"], dtype=np.float32)

            robot_pose = sapien.Pose(p=self.robot_base_pose_p.tolist(), q=self.robot_base_pose_q.tolist())
            if self.robot_far_away:
                robot_pose = sapien.Pose(p=[3.0, 3.0, 2.0], q=self.robot_base_pose_q.tolist())
            spawn_robot_pose = robot_pose
            if self.safe_robot_during_spawn and not self.robot_far_away:
                spawn_robot_pose = sapien.Pose(
                    p=[self.robot_base_pose_p[0], self.robot_base_pose_p[1], float(self.robot_base_pose_p[2] + 1.0)],
                    q=self.robot_base_pose_q.tolist(),
                )
            self.agent.robot.set_pose(spawn_robot_pose)
            self.agent.reset(init_qpos=self.robot_init_qpos.copy())

            if self.use_probe_objects:
                if self.probe_carrots:
                    source_pose_p = np.asarray(self.probe_carrot_pose_p, dtype=np.float32)
                    source_pose_q = np.asarray(self.probe_carrot_pose_q, dtype=np.float32)
                    if source_pose_p.ndim == 1:
                        source_pose_p = np.repeat(source_pose_p[None, :], self.num_envs, axis=0)
                    if source_pose_q.ndim == 1:
                        source_pose_q = np.repeat(source_pose_q[None, :], self.num_envs, axis=0)
                    inactive_pose_p = np.zeros((self.num_envs, 3), dtype=np.float32)
                    inactive_pose_p[:, 0] = 3.0
                    inactive_pose_p[:, 1] = 3.0
                    inactive_pose_p[:, 2] = 2.0
                    zero_velocity = torch.zeros((self.num_envs, 3), device=self.device)
                    active_names = np.asarray(self.probe_source_model_names_by_env, dtype=object)
                    for source_model_name, source_actor in self.probe_carrots.items():
                        pose_p = inactive_pose_p.copy()
                        active_mask = active_names == source_model_name
                        pose_p[active_mask] = source_pose_p[active_mask]
                        source_actor.set_pose(Pose.create_from_pq(p=pose_p, q=source_pose_q))
                        source_actor.set_linear_velocity(zero_velocity)
                        source_actor.set_angular_velocity(zero_velocity)
                else:
                    self.probe_carrot.set_pose(
                        Pose.create_from_pq(p=self.probe_carrot_pose_p, q=self.probe_carrot_pose_q)
                    )
                    self.probe_carrot.set_linear_velocity(torch.zeros(3, device=self.device))
                    self.probe_carrot.set_angular_velocity(torch.zeros(3, device=self.device))
                self.probe_plate.set_pose(
                    Pose.create_from_pq(p=self.probe_plate_pose_p, q=self.probe_plate_pose_q)
                )

            if self.use_airi_cubes:
                target_colors = self._normalize_cube_color_list(
                    options.get("airi_cube_target_colors_by_env", options.get("airi_cube_target_color", self.airi_cube_target_color))
                )
                if len(target_colors) == 1 and self.num_envs > 1:
                    target_colors = target_colors * self.num_envs
                if len(target_colors) != self.num_envs:
                    raise ValueError(
                        "airi_cube_target_colors_by_env must have length num_envs "
                        f"({self.num_envs}), got {len(target_colors)}."
                    )
                self.airi_cube_target_colors_by_env = target_colors
                self.airi_cube_target_color = target_colors[0]

                ground_z = float(scene_ground_point(self.scene_asset_dir)[2])
                cube_center_z = ground_z + float(AIRI_CUBE_HALF_SIZE[2] + AIRI_CUBE_SPAWN_EXTRA_Z)
                self.airi_cube_initial_center_z = cube_center_z
                episode_ids = resolve_episode_ids(options, self.num_envs, default=0)
                randomize_cube_positions = bool(options.get("randomize_airi_cube_positions", False))
                cube_position_seed = options.get("airi_cube_position_seed", None)
                if cube_position_seed is not None:
                    episode_ids = np.asarray(episode_ids, dtype=np.int64) + int(cube_position_seed)
                jitter_xy = np.array(
                    [
                        float(options.get("airi_cube_position_jitter_x", AIRI_CUBE_SPAWN_JITTER_XY[0])),
                        float(options.get("airi_cube_position_jitter_y", AIRI_CUBE_SPAWN_JITTER_XY[1])),
                    ],
                    dtype=np.float32,
                )
                cube_q = np.repeat(
                    np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    self.num_envs,
                    axis=0,
                )
                zero_velocity = torch.zeros((self.num_envs, 3), device=self.device)
                for color_name, cube_actor in self.airi_cube_actors.items():
                    cube_p = np.zeros((self.num_envs, 3), dtype=np.float32)
                    for env_idx in range(self.num_envs):
                        env_rng = np.random.default_rng(int(episode_ids[env_idx]))
                        xy_by_color = airi_cube_grid_xy_positions(
                            env_rng,
                            randomize=randomize_cube_positions,
                            jitter_xy=jitter_xy,
                        )
                        xy = xy_by_color[color_name]
                        cube_p[env_idx] = [float(xy[0]), float(xy[1]), cube_center_z]
                    cube_actor.set_pose(Pose.create_from_pq(p=cube_p, q=cube_q))
                    cube_actor.set_linear_velocity(zero_velocity)
                    cube_actor.set_angular_velocity(zero_velocity)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
            self._settle(0.5)
            if self.gpu_sim_enabled:
                self.scene._gpu_fetch_all()
            if self.use_probe_objects:
                if self.probe_carrots:
                    active_names = np.asarray(self.probe_source_model_names_by_env, dtype=object)
                    lin_vel = 0.0
                    ang_vel = 0.0
                    for source_model_name, source_actor in self.probe_carrots.items():
                        active_indices = np.flatnonzero(active_names == source_model_name)
                        if active_indices.size == 0:
                            continue
                        active_index_tensor = torch.as_tensor(active_indices, device=self.device, dtype=torch.long)
                        lin_vel = max(
                            lin_vel,
                            torch.linalg.norm(source_actor.linear_velocity[active_index_tensor], dim=1).max().item(),
                        )
                        ang_vel = max(
                            ang_vel,
                            torch.linalg.norm(source_actor.angular_velocity[active_index_tensor], dim=1).max().item(),
                        )
                else:
                    lin_vel = torch.linalg.norm(self.probe_carrot.linear_velocity).item()
                    ang_vel = torch.linalg.norm(self.probe_carrot.angular_velocity).item()
                if lin_vel > 1e-3 or ang_vel > 1e-2:
                    if self.gpu_sim_enabled:
                        self.scene._gpu_apply_all()
                    self._settle(6.0)
                    if self.gpu_sim_enabled:
                        self.scene._gpu_fetch_all()

            self.agent.robot.set_pose(robot_pose)
            self.agent.reset(init_qpos=self.robot_init_qpos.copy())
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

    def evaluate(self):
        info = {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }
        if self.use_airi_cubes:
            tcp_pos = self.agent.tcp.pose.p
            target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            active_colors = np.asarray(self.airi_cube_target_colors_by_env, dtype=object)
            for color_name, cube_actor in self.airi_cube_actors.items():
                active_indices = np.flatnonzero(active_colors == color_name)
                if active_indices.size == 0:
                    continue
                active_index_tensor = torch.as_tensor(active_indices, device=self.device, dtype=torch.long)
                target_pos[active_index_tensor] = cube_actor.pose.p[active_index_tensor]
                actor_grasped = self.agent.is_grasping(cube_actor)
                grasped[active_index_tensor] = actor_grasped[active_index_tensor]

            height_above_start = target_pos[:, 2] - float(self.airi_cube_initial_center_z)
            success = height_above_start >= float(AIRI_CUBE_PICKUP_SUCCESS_HEIGHT)
            info["tcp_to_carrot_dist"] = torch.linalg.norm(tcp_pos - target_pos, dim=1)
            info["carrot_to_plate_dist"] = torch.clamp(
                float(AIRI_CUBE_PICKUP_SUCCESS_HEIGHT) - height_above_start,
                min=0.0,
            )
            info["is_src_obj_grasped"] = grasped
            info["consecutive_grasp"] = grasped.clone()
            info["src_on_target"] = success
            info["target_cube_height_above_start"] = height_above_start
            info["success"] = success
            return info

        if self.use_probe_objects:
            tcp_pos = self.agent.tcp.pose.p
            plate_pos = self.probe_plate.pose.p
            if self.probe_carrots:
                active_names = np.asarray(self.probe_source_model_names_by_env, dtype=object)
                carrot_pos = torch.zeros_like(plate_pos)
                grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                contact_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                for source_model_name, source_actor in self.probe_carrots.items():
                    active_indices = np.flatnonzero(active_names == source_model_name)
                    if active_indices.size == 0:
                        continue
                    active_index_tensor = torch.as_tensor(active_indices, device=self.device, dtype=torch.long)
                    carrot_pos[active_index_tensor] = source_actor.pose.p[active_index_tensor]
                    actor_grasped = self.agent.is_grasping(source_actor)
                    grasped[active_index_tensor] = actor_grasped[active_index_tensor]
                    pair_forces = self.scene.get_pairwise_contact_forces(source_actor, self.probe_plate)
                    contact_force[active_index_tensor] = torch.linalg.norm(
                        pair_forces[active_index_tensor],
                        dim=1,
                    )
            else:
                carrot_pos = self.probe_carrot.pose.p
                grasped = self.agent.is_grasping(self.probe_carrot)
                contact_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                for idx in range(self.num_envs):
                    pair_force = self.scene.get_pairwise_contact_forces(self.probe_carrot, self.probe_plate)[idx]
                    contact_force[idx] = torch.linalg.norm(pair_force)
            info["tcp_to_carrot_dist"] = torch.linalg.norm(tcp_pos - carrot_pos, dim=1)
            info["carrot_to_plate_dist"] = torch.linalg.norm(carrot_pos - plate_pos, dim=1)
            info["is_src_obj_grasped"] = grasped

            plate_xy_limit = float(plate_xy_half_extent(PROBE_PLATE_POSE_Q)) + 0.01
            xy_dist = torch.linalg.norm(carrot_pos[:, :2] - plate_pos[:, :2], dim=1)
            xy_flag = xy_dist <= plate_xy_limit
            z_offset = carrot_pos[:, 2] - plate_pos[:, 2]
            z_flag = (z_offset > 0.0) & (z_offset <= 0.08)

            src_on_target = xy_flag & z_flag & (contact_force > 0.03)

            info["src_on_target"] = src_on_target
            info["plate_contact_force"] = contact_force
            info["success"] = src_on_target & (~grasped)
            # Keep compatibility with the Bridge-task wrappers used by SimplerEnv.
            info["consecutive_grasp"] = grasped.clone()
        return info

    def _get_obs_extra(self, info: Dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
        }
        if self.use_probe_objects:
            if self.probe_carrots:
                active_names = np.asarray(self.probe_source_model_names_by_env, dtype=object)
                probe_carrot_pose = torch.zeros_like(self.probe_plate.pose.raw_pose)
                for source_model_name, source_actor in self.probe_carrots.items():
                    active_indices = np.flatnonzero(active_names == source_model_name)
                    if active_indices.size == 0:
                        continue
                    active_index_tensor = torch.as_tensor(active_indices, device=self.device, dtype=torch.long)
                    probe_carrot_pose[active_index_tensor] = source_actor.pose.raw_pose[active_index_tensor]
                obs["probe_carrot_pose"] = probe_carrot_pose
            else:
                obs["probe_carrot_pose"] = self.probe_carrot.pose.raw_pose
            obs["probe_plate_pose"] = self.probe_plate.pose.raw_pose
        if self.use_airi_cubes:
            for color_name, cube_actor in self.airi_cube_actors.items():
                obs[f"airi_cube_{color_name}_pose"] = cube_actor.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)


@register_env("PutObjectOnPlateAiriTable-v1", max_episode_steps=80)
class PutObjectOnPlateAiriTable(OpenReal2SimValidationEnv):
    def __init__(
        self,
        *args,
        spawn_grid_state_file: str | Path | None = DEFAULT_CUSTOM_SCENE_SPAWN_GRID_STATE_FILE,
        **kwargs,
    ):
        self.spawn_grid_state_file = str(spawn_grid_state_file) if spawn_grid_state_file else ""
        super().__init__(
            *args,
            robot_uids="rc5_aero_hand_right_openreal2sim_validation",
            scene_asset_dir=AIRI_TABLE_EMPTY_ASSET_DIR,
            **kwargs,
        )

    def _source_names_for_obj_set(self, obj_set: str) -> list[str]:
        return source_model_names_for_obj_set(obj_set)

    def _resolve_episode_id(self, options: dict) -> int:
        episode_id = options.get("episode_id", 0)
        if torch.is_tensor(episode_id):
            if episode_id.numel() == 0:
                return 0
            return int(episode_id.reshape(-1)[0].item())
        if isinstance(episode_id, np.ndarray):
            if episode_id.size == 0:
                return 0
            return int(episode_id.reshape(-1)[0])
        if isinstance(episode_id, (list, tuple)):
            return int(episode_id[0]) if episode_id else 0
        return int(episode_id)

    def _sample_task_setup(self, options: dict) -> dict:
        source_names = self._source_names_for_obj_set(str(options.get("obj_set", "all")))
        plate_name = DEFAULT_PLATE_MODEL_NAME
        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        for source_name in source_names:
            for source_xy, plate_xy in _valid_spawn_pairs(
                scene_asset_dir=self.scene_asset_dir,
                spawn_grid_state_file=self.spawn_grid_state_file,
                robot_base_pose_p=np.asarray(self.robot_base_pose_p, dtype=np.float32),
                source_model_name=source_name,
                plate_model_name=plate_name,
            ):
                candidates.append((source_name, source_xy, plate_xy))
        if not candidates:
            raise RuntimeError("No valid custom-scene spawn pairs found for PutObjectOnPlateAiriTable-v1.")

        use_default_task = bool(options.get("use_default_task", False))
        idx = 0 if use_default_task else (self._resolve_episode_id(options) % len(candidates))
        source_name, source_xy, plate_xy = candidates[idx]
        return {
            "probe_source_model_name": source_name,
            "probe_plate_model_name": plate_name,
            "probe_carrot_pose_p": [
                float(source_xy[0]),
                float(source_xy[1]),
                float(_load_spawn_grid_state(self.spawn_grid_state_file, self.scene_asset_dir)["z"][0]
                      + source_resting_center_z_offset(PROBE_CARROT_POSE_Q, model_name=source_name)
                      + SOURCE_SPAWN_EXTRA_Z),
            ],
            "probe_carrot_pose_q": list(PROBE_CARROT_POSE_Q),
            "probe_plate_pose_p": [
                float(plate_xy[0]),
                float(plate_xy[1]),
                float(_load_spawn_grid_state(self.spawn_grid_state_file, self.scene_asset_dir)["z"][0]
                      + plate_resting_center_z_offset(PROBE_PLATE_POSE_Q, model_name=plate_name)),
            ],
            "probe_plate_pose_q": list(PROBE_PLATE_POSE_Q),
        }

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        options = dict(options)
        if "probe_source_model_name" not in options and "probe_carrot_pose_p" not in options:
            options.update(self._sample_task_setup(options))
        super()._initialize_episode(env_idx, options)

    def evaluate(self):
        info = super().evaluate()
        if "consecutive_grasp" not in info and "is_src_obj_grasped" in info:
            info["consecutive_grasp"] = info["is_src_obj_grasped"].clone()
        return info

    def get_language_instruction(self, **kwargs):
        source_names = list(getattr(self, "probe_source_model_names_by_env", []))
        if len(source_names) != self.num_envs:
            source_names = [self.probe_source_model_name] * self.num_envs
        return [
            resolve_language_instruction(
                source_name,
                plate_model_name=self.probe_plate_model_name,
                template=self.language_instruction_template,
            )
            for source_name in source_names
        ]


@register_env("PutObjectOnPlateAiriTableRecorder-v1", max_episode_steps=80)
class PutObjectOnPlateAiriTableRecorder(PutObjectOnPlateAiriTable):
    """Recorder-matching eval task that batches spawn pairs under one mesh setup.

    This env mirrors the non-resume sampling path used by `record_vr_demos.py`
    for the AIRI custom scene:
    - source model preset comes from obj_set=train|test|all
    - source model is randomized uniformly across valid source-model groups
    - grid pair is randomized uniformly within the chosen source-model group

    On GPU batched eval, the underlying probe-object env only instantiates one
    source mesh at a time, so one reset batch must share a single sampled source
    model. We still randomize spawn pairs independently across envs within that
    chosen source-model group so eval can run batched without loading multiple
    model copies.
    """

    def __init__(self, *args, **kwargs):
        self._current_probe_source_model_names = None
        self._current_probe_plate_model_name = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _resolve_episode_id_scalar(options: dict, default: int = 0) -> int:
        episode_id = options.get("episode_id", default)
        if torch.is_tensor(episode_id):
            if episode_id.numel() == 0:
                return int(default)
            return int(episode_id.reshape(-1)[0].item())
        if isinstance(episode_id, np.ndarray):
            if episode_id.size == 0:
                return int(default)
            return int(episode_id.reshape(-1)[0])
        if isinstance(episode_id, (list, tuple)):
            return int(episode_id[0]) if episode_id else int(default)
        return int(episode_id)

    def _resolve_episode_ids(self, options: dict, default: int = 0) -> np.ndarray:
        episode_id = options.get("episode_id", default)
        if torch.is_tensor(episode_id):
            if episode_id.numel() == 0:
                return np.full((self.num_envs,), int(default), dtype=np.int64)
            values = episode_id.detach().cpu().reshape(-1).to(torch.int64).numpy()
        elif isinstance(episode_id, np.ndarray):
            if episode_id.size == 0:
                return np.full((self.num_envs,), int(default), dtype=np.int64)
            values = episode_id.reshape(-1).astype(np.int64, copy=False)
        elif isinstance(episode_id, (list, tuple)):
            if len(episode_id) == 0:
                return np.full((self.num_envs,), int(default), dtype=np.int64)
            values = np.asarray(episode_id, dtype=np.int64).reshape(-1)
        else:
            values = np.asarray([int(episode_id)], dtype=np.int64)
        if values.size == 1 and self.num_envs > 1:
            values = np.full((self.num_envs,), int(values[0]), dtype=np.int64)
        return values

    def _resolve_grid_pair_indices(self, options: dict, default: int = 0) -> np.ndarray:
        grid_pair_index = options.get("grid_pair_index", default)
        if torch.is_tensor(grid_pair_index):
            if grid_pair_index.numel() == 0:
                return np.full((self.num_envs,), int(default), dtype=np.int64)
            values = grid_pair_index.detach().cpu().reshape(-1).to(torch.int64).numpy()
        elif isinstance(grid_pair_index, np.ndarray):
            if grid_pair_index.size == 0:
                return np.full((self.num_envs,), int(default), dtype=np.int64)
            values = grid_pair_index.reshape(-1).astype(np.int64, copy=False)
        elif isinstance(grid_pair_index, (list, tuple)):
            if len(grid_pair_index) == 0:
                return np.full((self.num_envs,), int(default), dtype=np.int64)
            values = np.asarray(grid_pair_index, dtype=np.int64).reshape(-1)
        else:
            values = np.asarray([int(grid_pair_index)], dtype=np.int64)
        if values.size == 1 and self.num_envs > 1:
            values = np.full((self.num_envs,), int(values[0]), dtype=np.int64)
        return values

    def _candidate_groups_for_obj_set(
        self,
        obj_set: str,
    ) -> list[tuple[str, str, list[tuple[np.ndarray, np.ndarray]]]]:
        plate_name = DEFAULT_PLATE_MODEL_NAME
        candidate_groups: list[tuple[str, str, list[tuple[np.ndarray, np.ndarray]]]] = []
        for source_name in self._source_names_for_obj_set(obj_set):
            pairs = _valid_spawn_pairs(
                scene_asset_dir=self.scene_asset_dir,
                spawn_grid_state_file=self.spawn_grid_state_file,
                robot_base_pose_p=np.asarray(self.robot_base_pose_p, dtype=np.float32),
                source_model_name=source_name,
                plate_model_name=plate_name,
            )
            if pairs:
                candidate_groups.append((source_name, plate_name, pairs))
        return candidate_groups

    def _sample_recorder_matching_setup(self, options: dict) -> dict:
        obj_set = str(options.get("obj_set", "train"))
        candidate_groups = self._candidate_groups_for_obj_set(obj_set)
        if not candidate_groups:
            raise RuntimeError(
                "No valid custom-scene spawn pairs found for PutObjectOnPlateAiriTableRecorder-v1."
            )

        use_default_task = bool(options.get("use_default_task", False))
        episode_ids = self._resolve_episode_ids(options, default=0)
        grid_pair_indices = self._resolve_grid_pair_indices(options, default=0)

        loaded_source_names = [source_name for source_name, _, _ in candidate_groups]
        source_names_by_env: list[str] = []
        source_positions = []
        plate_positions = []
        pair_indices = np.empty((self.num_envs,), dtype=np.int64)
        grid_state = _load_spawn_grid_state(self.spawn_grid_state_file, self.scene_asset_dir)
        grid_z = float(grid_state["z"][0])

        for idx, ep_id in enumerate(episode_ids):
            if use_default_task:
                source_name, plate_name, pairs = candidate_groups[0]
                pair_index = int(np.clip(grid_pair_indices[idx], 0, len(pairs) - 1))
            else:
                # Sample the source model independently per vectorized env. This
                # keeps batched PPO/eval visually representative instead of
                # forcing every env in a reset batch to share one mesh.
                env_rng = np.random.default_rng(int(ep_id))
                source_name, plate_name, pairs = candidate_groups[int(env_rng.integers(len(candidate_groups)))]
                pair_index = int(env_rng.integers(len(pairs)))

            pair_indices[idx] = pair_index
            source_names_by_env.append(source_name)
            source_xy, plate_xy = pairs[int(pair_index)]
            source_positions.append(
                [
                    float(source_xy[0]),
                    float(source_xy[1]),
                    float(
                        grid_z
                        + source_resting_center_z_offset(PROBE_CARROT_POSE_Q, model_name=source_name)
                        + SOURCE_SPAWN_EXTRA_Z
                    ),
                ]
            )
            plate_positions.append(
                [
                    float(plate_xy[0]),
                    float(plate_xy[1]),
                    float(grid_z + plate_resting_center_z_offset(PROBE_PLATE_POSE_Q, model_name=plate_name)),
                ]
            )
        return {
            "probe_source_model_name": source_names_by_env[0],
            "probe_source_model_names": loaded_source_names,
            "probe_source_model_names_by_env": source_names_by_env,
            "probe_plate_model_name": plate_name,
            "probe_carrot_pose_p": np.asarray(source_positions, dtype=np.float32),
            "probe_carrot_pose_q": np.repeat(
                np.asarray(PROBE_CARROT_POSE_Q, dtype=np.float32)[None, :], self.num_envs, axis=0
            ),
            "probe_plate_pose_p": np.asarray(plate_positions, dtype=np.float32),
            "probe_plate_pose_q": np.repeat(
                np.asarray(PROBE_PLATE_POSE_Q, dtype=np.float32)[None, :], self.num_envs, axis=0
            ),
            "grid_pair_index": pair_indices,
        }

    def reset(self, *, seed=None, options=None):
        next_options = dict(options or {})
        setup = self._sample_recorder_matching_setup(next_options)
        next_source_model_names = tuple(str(name) for name in setup["probe_source_model_names"])
        next_plate_model_name = str(setup["probe_plate_model_name"])
        needs_reconfigure = bool(next_options.get("reconfigure", False)) or (
            self._current_probe_source_model_names != next_source_model_names
            or self._current_probe_plate_model_name != next_plate_model_name
        )
        next_options.update(setup)
        next_options["reconfigure"] = needs_reconfigure

        obs, info = super().reset(seed=seed, options=next_options)
        self._current_probe_source_model_names = next_source_model_names
        self._current_probe_plate_model_name = next_plate_model_name
        return obs, info


@register_env("PutObjectOnPlateAiriCubes-v1", max_episode_steps=80)
class PutObjectOnPlateAiriCubes(OpenReal2SimValidationEnv):
    """AIRI cubes task: pick up the requested cube color."""

    def __init__(
        self,
        *args,
        spawn_grid_state_file: str | Path | None = DEFAULT_AIRI_CUBES_SPAWN_GRID_STATE_FILE,
        **kwargs,
    ):
        self.spawn_grid_state_file = str(spawn_grid_state_file) if spawn_grid_state_file else ""
        OpenReal2SimValidationEnv.__init__(
            self,
            *args,
            robot_uids="rc5_aero_hand_right_openreal2sim_validation",
            scene_asset_dir=AIRI_CUBES_ASSET_DIR,
            **kwargs,
        )

    @staticmethod
    def _resolve_episode_ids(options: dict, num_envs: int, default: int = 0) -> np.ndarray:
        episode_id = options.get("episode_id", default)
        if torch.is_tensor(episode_id):
            if episode_id.numel() == 0:
                return np.full((num_envs,), int(default), dtype=np.int64)
            values = episode_id.detach().cpu().reshape(-1).to(torch.int64).numpy()
        elif isinstance(episode_id, np.ndarray):
            if episode_id.size == 0:
                return np.full((num_envs,), int(default), dtype=np.int64)
            values = episode_id.reshape(-1).astype(np.int64, copy=False)
        elif isinstance(episode_id, (list, tuple)):
            if len(episode_id) == 0:
                return np.full((num_envs,), int(default), dtype=np.int64)
            values = np.asarray(episode_id, dtype=np.int64).reshape(-1)
        else:
            values = np.asarray([int(episode_id)], dtype=np.int64)
        if values.size == 1 and num_envs > 1:
            values = np.full((num_envs,), int(values[0]), dtype=np.int64)
        return values

    def _target_colors_for_reset(self, options: dict) -> list[str]:
        if "airi_cube_target_colors_by_env" in options:
            colors = self._normalize_cube_color_list(options["airi_cube_target_colors_by_env"])
            if len(colors) == 1 and self.num_envs > 1:
                colors = colors * self.num_envs
            return colors
        if "airi_cube_target_color" in options:
            colors = self._normalize_cube_color_list(options["airi_cube_target_color"])
            color = colors[0] if colors else AIRI_CUBE_COLOR_NAMES[0]
            return [color] * self.num_envs
        if bool(options.get("use_default_task", False)):
            return [AIRI_CUBE_COLOR_NAMES[0]] * self.num_envs
        colors = []
        episode_ids = self._resolve_episode_ids(options, self.num_envs, default=0)
        for ep_id in episode_ids:
            rng = np.random.default_rng(int(ep_id))
            colors.append(AIRI_CUBE_COLOR_NAMES[int(rng.integers(len(AIRI_CUBE_COLOR_NAMES)))])
        return colors

    def reset(self, *, seed=None, options=None):
        next_options = dict(options or {})
        next_options["use_probe_objects"] = False
        next_options["use_airi_cubes"] = True
        next_options.setdefault("randomize_airi_cube_positions", True)
        next_options["airi_cube_target_colors_by_env"] = self._target_colors_for_reset(next_options)
        if not self.airi_cube_actors:
            next_options["reconfigure"] = True
        return super().reset(seed=seed, options=next_options)

    def get_language_instruction(self, **kwargs):
        colors = list(getattr(self, "airi_cube_target_colors_by_env", []))
        if len(colors) != self.num_envs:
            colors = [self.airi_cube_target_color] * self.num_envs
        return [f"Pick up the {color} cube." for color in colors]


@register_env("PutObjectOnPlateAiriCubesRecorder-v1", max_episode_steps=80)
class PutObjectOnPlateAiriCubesRecorder(PutObjectOnPlateAiriCubes):
    """Recorder/PPO variant for the AIRI cubes camera scene."""


@register_env("PutObjectOnPlateAiriCubesV3-v1", max_episode_steps=80)
class PutObjectOnPlateAiriCubesV3(PutObjectOnPlateAiriCubes):
    """AIRI cubes task variant with the v3 robot base and initial pose."""


@register_env("PutObjectOnPlateAiriCubesV3Recorder-v1", max_episode_steps=80)
class PutObjectOnPlateAiriCubesV3Recorder(PutObjectOnPlateAiriCubesV3):
    """Recorder/PPO v3 variant for the AIRI cubes camera scene."""


@register_env("PickUpAiriCube-v1", max_episode_steps=80)
class PickUpAiriCube(PutObjectOnPlateAiriCubes):
    """Alias with task-accurate naming for AIRI cube pickup."""


@register_env("PickUpAiriCubeRecorder-v1", max_episode_steps=80)
class PickUpAiriCubeRecorder(PutObjectOnPlateAiriCubesRecorder):
    """Recorder/PPO alias with task-accurate naming for AIRI cube pickup."""


@register_env("PickUpAiriCubeV3-v1", max_episode_steps=80)
class PickUpAiriCubeV3(PutObjectOnPlateAiriCubesV3):
    """Task-accurate v3 alias for AIRI cube pickup."""


@register_env("PickUpAiriCubeV3Recorder-v1", max_episode_steps=80)
class PickUpAiriCubeV3Recorder(PutObjectOnPlateAiriCubesV3Recorder):
    """Recorder/PPO task-accurate v3 alias for AIRI cube pickup."""
