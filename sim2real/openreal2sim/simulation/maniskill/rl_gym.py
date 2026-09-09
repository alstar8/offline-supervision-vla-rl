"""Gym kwargs for PPO / SimplerEnv on the reconstructed AIRI table scene."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_setup import (
    apply_hand_contact_profile_overrides,
    apply_hand_controller_profile_overrides,
    apply_lighting_profile_overrides,
    initialize_sapien_renderer,
    load_runner_config,
    openreal2sim_env_kwargs_from_config,
)
from openreal2sim.simulation.maniskill.utils.rl_placement import (
    DEFAULT_MIN_ROBOT_CLEARANCE,
    DEFAULT_PAIR_GAP,
    DEFAULT_REACHABLE_BOUNDS_MAX_XY,
    DEFAULT_REACHABLE_BOUNDS_MIN_XY,
)
from openreal2sim.simulation.maniskill.utils.scene_loader import (
    DEFAULT_SCENE_JSON_PATH,
    SIM2REAL_REPO_ROOT,
)

PICK_RED_CUBE_INSTRUCTION = "Pick red cube"
PICK_RED_CUBE_OBJECT_ID = "orange_cube_ext"
RC5_RL_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
DEFAULT_CONFIG_PATH = SIM2REAL_REPO_ROOT / "config" / "config_debug.yaml"
DEFAULT_SCENE_KEY = "airi_table_new_empty3_image"
DEFAULT_RL_RANDOM_PLACEMENT = {
    "bounds_min": DEFAULT_REACHABLE_BOUNDS_MIN_XY.tolist(),
    "bounds_max": DEFAULT_REACHABLE_BOUNDS_MAX_XY.tolist(),
    "table_margin": 0.0,
    "min_robot_clearance": DEFAULT_MIN_ROBOT_CLEARANCE,
    "pair_gap": DEFAULT_PAIR_GAP,
    "max_attempts": 80,
}

_PROFILE_PATH_KEYS = (
    "hand_contact_config",
    "hand_controller_config",
    "lighting_profile_config",
    "teleop_profile_config",
    "hand_pose_config",
)


def _abs_sim2real_path(path) -> str:
    raw = Path(path)
    if raw.is_absolute():
        return str(raw)
    return str(SIM2REAL_REPO_ROOT / raw)


def _runner_args(
    *,
    scene: str,
    config_path: str,
    key: str,
    manip_object_id: str,
    use_wrist_camera: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        no_auto_placement=False,
        scene=scene,
        key=key,
        config_path=config_path,
        video_width=None,
        video_height=None,
        robot_base_pose=None,
        robot_init_qpos=None,
        manip_object_id=manip_object_id,
        task_object_id=None,
        robot_uids=None,
        approach_waypoints=None,
        approach_waypoint_mode=None,
        planner_backend=None,
        planner_proxy_frame=None,
        disable_planner_proxy_adaptive_steps=False,
        planner_proxy_safe_clearance_z=None,
        planner_proxy_predescent_settle_steps=None,
        planner_proxy_preclose_settle_steps=None,
        rc5_move_group=None,
        rc5_frame_conversion=None,
        rc5_obb_target_semantics=None,
        rc5_object_profile_target_semantics=None,
        hand_pose_config=None,
        lighting_profile_config=None,
        lighting_profile=None,
        hand_contact_config=None,
        hand_contact_profile=None,
        hand_controller_config=None,
        hand_controller_profile=None,
        teleop_profile_config=None,
        teleop_profile=None,
        control_mode=None,
        max_num_materials=None,
        max_num_textures=None,
        num_envs=1,
        render_backend="gpu",
        cam_width=640,
        cam_height=480,
        robot_init_qpos_noise=0.0,
        sim_backend="gpu",
        cam_eye=None,
        cam_target=None,
        window_width=512,
        window_height=512,
        use_wrist_camera=use_wrist_camera,
    )


def build_openreal2sim_rl_gym_kwargs(
    *,
    config_path: str | Path | None = None,
    scene_json_path: str | Path | None = None,
    key: str = DEFAULT_SCENE_KEY,
    manip_object_id: str = PICK_RED_CUBE_OBJECT_ID,
    task_description: str = PICK_RED_CUBE_INSTRUCTION,
    use_wrist_camera: bool = True,
    placement_mode: str = "random",
    initialize_renderer: bool = True,
) -> dict:
    """Return OpenReal2Sim-v0 kwargs for gym.make / SimplerEnv.

    Planner defaults in config_debug.yaml stay unchanged. This factory only
    overrides the RL task: orange cube, random reachable placements, wrist cam.
    """
    cfg_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = SIM2REAL_REPO_ROOT / cfg_path
    scene = str(scene_json_path) if scene_json_path is not None else str(DEFAULT_SCENE_JSON_PATH)
    args = _runner_args(
        scene=scene,
        config_path=str(cfg_path),
        key=key,
        manip_object_id=manip_object_id,
        use_wrist_camera=use_wrist_camera,
    )
    config_overrides = load_runner_config(args)
    for path_key in _PROFILE_PATH_KEYS:
        value = config_overrides.get(path_key)
        if value:
            config_overrides[path_key] = _abs_sim2real_path(value)
    apply_lighting_profile_overrides(args, config_overrides)
    apply_hand_contact_profile_overrides(args, config_overrides)
    apply_hand_controller_profile_overrides(args, config_overrides)
    if initialize_renderer:
        initialize_sapien_renderer(config_overrides.get("renderer_kwargs"))

    config_overrides["manip_object_id"] = manip_object_id
    config_overrides["task_description"] = task_description
    config_overrides["placement_mode"] = placement_mode
    config_overrides["use_wrist_camera"] = bool(use_wrist_camera)
    config_overrides["use_360_background"] = True
    random_placement = dict(config_overrides.get("random_placement") or {})
    random_placement = {**DEFAULT_RL_RANDOM_PLACEMENT, **random_placement}
    config_overrides["random_placement"] = random_placement

    env_kwargs = openreal2sim_env_kwargs_from_config(
        args,
        config_overrides,
        render_mode="rgb_array",
        obs_mode="rgb+segmentation",
    )
    env_kwargs.pop("num_envs", None)
    env_kwargs.pop("obs_mode", None)
    env_kwargs.pop("control_mode", None)
    env_kwargs.pop("sim_backend", None)
    env_kwargs.pop("render_mode", None)
    env_kwargs.pop("render_backend", None)
    env_kwargs.pop("viewer_camera_configs", None)
    env_kwargs["use_wrist_camera"] = bool(use_wrist_camera)
    env_kwargs["use_360_background"] = True
    env_kwargs["task_description"] = task_description
    env_kwargs["placement_mode"] = placement_mode
    env_kwargs["manip_object_id"] = manip_object_id
    cameras_config = dict(env_kwargs.get("cameras_config") or {})
    base_camera = dict(cameras_config.get("base_camera") or {})
    # Scene JSON is 1916x1076; PPO only consumes 3rd_view 640x480 + wrist.
    # Keep a matching downscaled base camera so parallel envs fit on one GPU.
    base_camera["width"] = 640
    base_camera["height"] = 480
    cameras_config["base_camera"] = base_camera
    env_kwargs["cameras_config"] = cameras_config
    return env_kwargs
