# -*- coding: utf-8 -*-
"""
OpenReal2Sim ManiSkill Environment.

Loads reconstructed scenes from the OpenReal2Sim pipeline and creates
an interactive robotic manipulation environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from ..utils.pano_sphere import (
    DEFAULT_360_PHOTOS_DIR,
    DEFAULT_PANO_SPHERE_RADIUS,
    build_pano_sphere_actor,
    list_360_photo_paths,
    load_pano_texture,
    resolve_equirect_texture,
)
from ..utils.rl_placement import (
    DEFAULT_MIN_ROBOT_CLEARANCE,
    DEFAULT_PAIR_GAP,
    DEFAULT_REACHABLE_BOUNDS_MAX_XY,
    DEFAULT_REACHABLE_BOUNDS_MIN_XY,
    instruction_for_manip_object,
    sample_nonoverlapping_xy,
    xy_half_extent_from_bbox,
)
from ..utils.scene_loader import (
    DEFAULT_CAMERAS_CONFIG,
    DEFAULT_SCENE_JSON_PATH,
    SceneConfig,
    get_base_camera_eye_target,
    load_scene_config,
    resolve_app_runtime_path,
    resolve_path,
)
from ..utils.transform_utils import opencv_to_sapien_pose, qvec2rotmat

WRIST_CAMERA_NAME = "wrist_camera"
WRIST_CAMERA_WIDTH = 168
WRIST_CAMERA_HEIGHT = 224
# Vertical FOV of the D405 colour stream as the policy sees it on hardware:
# the 640x480 frame is rotated by np.rot90, so the inset's vertical axis is
# the sensor's horizontal one, 2*atan(640/(2*fx)) with fx=391.8. Both the
# rotated frame and the 168x224 inset are 4:3, so this matches the horizontal
# FOV too (63.0 deg). Was 1.5, i.e. 85.9 x 69.9 deg against 78.5 x 63.0 real.
WRIST_CAMERA_FOV = 1.3697
WRIST_CAMERA_NEAR = 0.01
WRIST_CAMERA_FAR = 2.0
WRIST_CAMERA_FAR_PANO = 100.0
WRIST_CAMERA_MOUNT_LINKS = ("prehand", "prehand_cam")
# Pose of the real D405 in the `prehand` link, from eye-in-hand calibration
# against a ChArUco board (2026-09-11, 15 poses; leave-one-out std about
# [0.3, 0.5, 1.1] mm and 0.03 deg -- sim2real/real_replay/calibrate_wrist_handeye.py
# and handeye_to_sim.py). Against the previous CAD pose [0, 0.0675, 0.0606] the
# camera sits +8.85 mm along x: the D405 colour/depth origin is its left imager,
# ~9 mm off the mounting axis, not the housing centreline. The orientation is
# the camera the policy sees on hardware, i.e. after the np.rot90 that
# eval_openvla_real.py applies to the raw D405 frame.
WRIST_CAMERA_LOCAL_P = [0.00885, 0.06773, 0.06504]
WRIST_CAMERA_LOCAL_Q = [0.711969, -0.017879, 0.701977, 0.002846]
THIRD_VIEW_CAMERA_NAME = "3rd_view_camera"
THIRD_VIEW_WIDTH = 640
THIRD_VIEW_HEIGHT = 480


def _wrist_mount_link(agent):
    links = getattr(getattr(agent, "robot", None), "links_map", None) or {}
    for name in WRIST_CAMERA_MOUNT_LINKS:
        mount = links.get(name)
        if mount is not None:
            return name, mount
    return None, None


def _apply_runtime_joint_controller_tuning(
    *,
    agent,
    cfg: dict | None,
    joint_names: list[str],
    scope_label: str,
):
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        raise RuntimeError(f"{scope_label} must be a dict, got {type(cfg)}")
    if not joint_names:
        raise RuntimeError(f"{scope_label} was provided, but no target joint names are available.")

    def _broadcast_or_validate(value, expected_len, field_name):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(expected_len, float(arr), dtype=np.float64)
        flat = arr.reshape(-1)
        if len(flat) != expected_len:
            raise RuntimeError(
                f"{scope_label}.{field_name} expects either a scalar or {expected_len} values "
                f"for joints {joint_names}, got {len(flat)}"
            )
        return flat.astype(np.float64)

    field_values = {}
    for field_name in ("stiffness", "damping", "force_limit", "friction"):
        if field_name in cfg:
            field_values[field_name] = _broadcast_or_validate(cfg[field_name], len(joint_names), field_name)

    if not field_values:
        raise RuntimeError(
            f"{scope_label} must define at least one of: stiffness, damping, force_limit, friction"
        )

    controller_root = getattr(agent, "controller", None)
    if controller_root is None:
        raise RuntimeError(f"{scope_label} was provided, but agent.controller is not initialized.")
    controllers = getattr(controller_root, "controllers", None)
    controller_list = list(controllers.values()) if isinstance(controllers, dict) else [controller_root]

    applied_controllers = []
    for controller in controller_list:
        controller_joint_names = list(getattr(controller.config, "joint_names", []))
        if not controller_joint_names:
            continue
        joint_indices = [i for i, name in enumerate(controller_joint_names) if name in joint_names]
        if not joint_indices:
            continue

        for field_name, values in field_values.items():
            current = getattr(controller.config, field_name, None)
            if current is None:
                continue
            current_arr = np.asarray(current, dtype=np.float64)
            if current_arr.ndim == 0:
                current_arr = np.full(len(controller_joint_names), float(current_arr), dtype=np.float64)
            else:
                current_arr = current_arr.reshape(-1).astype(np.float64)
                if len(current_arr) != len(controller_joint_names):
                    raise RuntimeError(
                        f"Controller field {field_name} for {type(controller).__name__} expected "
                        f"{len(controller_joint_names)} values, got {len(current_arr)}"
                    )
            value_by_name = dict(zip(joint_names, values.tolist()))
            for idx in joint_indices:
                current_arr[idx] = value_by_name[controller_joint_names[idx]]
            setattr(controller.config, field_name, current_arr.tolist())

        controller.set_drive_property()
        applied_controllers.append(type(controller).__name__)

    if not applied_controllers:
        raise RuntimeError(
            f"{scope_label} was provided, but no runtime controller owning joints {joint_names} was found."
        )

    summary_parts = []
    for field_name, values in field_values.items():
        if np.allclose(values, values[0]):
            summary_parts.append(f"{field_name}={float(values[0]):.4f}")
        else:
            summary_parts.append(f"{field_name}=per_joint[{len(values)}]")
    print(
        f"[{scope_label}] Applied runtime joint drive tuning to "
        f"{', '.join(applied_controllers)}: " + ", ".join(summary_parts)
    )


@register_env("OpenReal2Sim-v0", max_episode_steps=200)
class OpenReal2SimEnv(BaseEnv):
    """
    OpenReal2Sim environment for ManiSkill.

    This environment loads a reconstructed scene from the OpenReal2Sim pipeline
    and provides a robotic manipulation environment with:
    - Reconstructed background geometry
    - Reconstructed object meshes
    - Camera configuration matching the original scene
    - Franka Panda robot for manipulation

    Args:
        scene_json_path: Path to scene.json file from OpenReal2Sim reconstruction
        robot_uids: Robot model to use (default: "panda")
        robot_init_qpos_noise: Noise level for initial robot configuration
        **kwargs: Additional arguments passed to BaseEnv
    """

    ROBOT_BASE_POS = [-0.6, 0.0, 0.3]
    ROBOT_BASE_QUAT = [1, 0, 0, 0]
    ROBOT_BASE_POSE = sapien.Pose(p=ROBOT_BASE_POS, q=ROBOT_BASE_QUAT)
    # SAPIEN uses wxyz quaternion format: identity = [1, 0, 0, 0]
    OBJECT_INIT_QUAT = [1, 0, 0, 0]
    OBJECT_INIT_POS = [0, 0, 0.0]
    BACKGROUND_INIT_POS = [0, 0, 0.0]
    BACKGROUND_INIT_QUAT = [1, 0, 0, 0]
    ROBOT_INIT_QPOS = [
        0.0,
        np.pi / 8,
        0,
        -np.pi * 5 / 8,
        0,
        np.pi * 3 / 4,
        np.pi / 4,
    ]
    ROBOT_INIT_QPOS_NOISE = 0.02
    ROBOT_INIT_QPOS_NOISE_2 = 0.04
    SUPPORTED_ROBOTS = [
        "panda",
        "widowx250s_openr2s",
        "widowx250s_bridgedataset_flat_table_openr2s",
        "widowx250s_openr2s_rl",
        "rc5_aero_hand_openr2s",
        "rc5_aero_hand_openr2s_rl",
    ]

    def __init__(
        self,
        *args,
        scene_json_path: str = None,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        render_width: int = None,
        render_height: int = None,
        scene_z_offset: float = 0.0,
        sim_ground_offset: float = None,
        render_camera_eye: list = None,
        render_camera_target: list = None,
        cameras_config: dict = None,
        lighting_config: dict = None,
        robot_base_pose: list = None,
        robot_init_qpos: list = None,
        object_material: dict = None,
        hand_contact_material: dict = None,
        hand_contact: dict = None,
        arm_controller: dict = None,
        hand_controller: dict = None,
        object_spawn_clearance: float = None,
        bg_collision_mode: str = "nonconvex",
        bg_use_decimated_collision_mesh: bool = False,
        bg_collision_mesh: str = "background_registered_collision.glb",
        bg_visual_mesh: str = None,
        obj_collision_mode: str = "coacd",
        physx_contact_offset: float = 0.005,
        physx_rest_offset: float = -0.001,
        placement_mode: str = "scene",
        object_placements: dict = None,
        random_placement: dict = None,
        auto_placement: bool = True,
        include_objects=None,
        exclude_objects=None,
        manip_object_id=None,
        target_object_id=None,
        task_description: str = None,
        settle_steps: int = 0,
        lift_height: float = None,
        finger_length: float = None,
        robot_base_pose_z_auto: bool = True,
        use_wrist_camera: bool = False,
        use_360_background: bool = False,
        pano_photos_dir: str = None,
        pano_sphere_radius: float = DEFAULT_PANO_SPHERE_RADIUS,
        **kwargs,
    ):
        """
        robot_base_pose: [x, y, z] или [x, y, z, qw, qx, qy, qz] — поза базы робота в мире.
        robot_init_qpos: [7 arm joints (rad), 2 gripper (m)] — начальная конфигурация суставов.
        cameras_config: конфиг камер из runtime config (cameras). Если None — используются DEFAULT_CAMERAS_CONFIG.
        render_camera_eye/target: переопределяют render_camera из cameras_config (CLI приоритет).
        object_material: dict с ключами static_friction, dynamic_friction, restitution.
        hand_contact_material: optional dict with static_friction, dynamic_friction,
            restitution applied to RC5 hand/finger collision shapes only.
        hand_contact: optional structured fingertip/hand contact config:
            {
              "materials": {
                "fingertip": {
                  "static_friction": 2.0,
                  "dynamic_friction": 1.5,
                  "restitution": 0.0,
                }
              },
              "links": {
                "right_thumb_tip_link": {
                  "material": "fingertip",
                  "patch_radius": 0.30,
                  "min_patch_radius": 0.15,
                }
              }
            }
            Also accepts source-compatible aliases "_materials" and "link".
        hand_controller: optional dict for tuning hand joint drive parameters, e.g.
            {
              "stiffness": 80,
              "damping": 16,
              "force_limit": 60,
              "friction": 0.0,
            }
            Values may be scalars or per-joint arrays for the active hand joints.
        arm_controller: optional dict for tuning arm joint drive parameters with the
            same schema as hand_controller. Values may be scalars or per-joint arrays
            for the active arm joints.
        object_spawn_clearance: зазор (м) между нижней гранью объекта и столом (default 0.001).
        bg_collision_mode: "nonconvex" | "coacd".
        bg_use_decimated_collision_mesh: if True, load bg_collision_mesh for collision geometry.
        bg_collision_mesh: filename (relative to scene dir) of pre-decimated collision mesh.
        bg_visual_mesh: filename (relative to scene dir) of decimated mesh for visual rendering.
            If None, uses the original background mesh. Use to speed up GPU rendering of large scenes.
        obj_collision_mode: "coacd" | "none" | "nonconvex".
        physx_contact_offset / physx_rest_offset: PhysX параметры. None = дефолт ManiSkill.
        placement_mode: "scene" | "fixed" | "random".
            "scene"  — позиции из scene.json + settle_steps + soft-reset (default).
            "fixed"  — позиции из object_placements dict (или scene.json для незаданных).
            "random" — случайные позиции на столе каждый эпизод.
        object_placements: dict {obj_id: {"position": [x,y,z], "orientation": [qx,qy,qz,qw]}}.
            Используется при placement_mode="fixed". Незаданные объекты берут позицию из scene.json.
        random_placement: dict с параметрами рандомизации (table_margin, yaw_range,
            min_object_distance, max_attempts, seed). Используется при placement_mode="random".
        auto_placement: if True, auto-compute object_spawn_clearance and sim_ground_offset
            via raycast on background mesh.  Explicit numeric values override auto.
        include_objects: list of object ids to load, or "manip_only".
            Mutually exclusive with exclude_objects (include wins if both set).
        exclude_objects: list of object ids to skip.
        manip_object_id: id of the manipulation target object used in evaluate().
            If None, defaults to the first object in object_actors.
        target_object_id: optional id of the support/goal object for put-on-target
            tasks. If set, evaluate() uses Bridge-style source-on-target success.
        settle_steps: number of physics steps to run after each reset to let objects
            settle on the table surface. Positions from scene.json may be slightly
            above or below the table due to reconstruction noise; settle_steps lets
            physics correct this. Default 0 (disabled) when placement_mode="fixed"
            with auto_placement=True (raycast provides correct Z). Use 5-30 for
            placement_mode="scene" where positions come directly from scene.json.
        robot_base_pose_z_auto: if True (default) and auto_placement=True, override
            robot_base_pose.z with auto_table_z (raycast-computed table surface height)
            after _load_scene(). Mirrors the post-env-creation step in the visualization
            script. Set to False to use the explicit Z from robot_base_pose as-is.
        use_wrist_camera: mount a RealSense-style wrist camera on ``prehand``.
        use_360_background: map Insta360 photos onto inverted spheres around the
            workspace. Planner path stays off; RL gym enables this by default.
        pano_photos_dir: directory of dual-fisheye / equirect 360 captures.
        pano_sphere_radius: inverted sphere radius in meters. Must exceed the
            camera far-plane distance only if cameras sit inside the sphere.
        """
        if scene_json_path is None:
            scene_json_path = DEFAULT_SCENE_JSON_PATH

        self.scene_json_path = Path(scene_json_path)
        self.use_wrist_camera = bool(use_wrist_camera)
        self.use_360_background = bool(use_360_background)
        self.pano_photos_dir = Path(pano_photos_dir) if pano_photos_dir else DEFAULT_360_PHOTOS_DIR
        self.pano_sphere_radius = float(pano_sphere_radius)
        self._pano_sphere_actor = None
        self._pano_sphere_material = None
        self._pano_textures = []
        self._pano_photo_paths = []
        self._pano_photo_idx = None
        self._pano_yaw = None
        self._pano_center = None
        self._pano_reset_counter = 0
        self._placement_reset_counter = 0
        self.scene_config: SceneConfig = load_scene_config(self.scene_json_path)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_z_offset = scene_z_offset

        self.auto_placement = auto_placement
        self._clearance_explicit = (object_spawn_clearance is not None)
        self._ground_offset_explicit = (sim_ground_offset is not None)
        self.sim_ground_offset = sim_ground_offset if sim_ground_offset is not None else 0.1
        self.auto_table_z: float | None = None
        self._auto_bg_mesh = None

        _default_mat = {"static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0}
        self.object_material = object_material if object_material is not None else _default_mat
        self.hand_contact_material = hand_contact_material
        self.hand_contact = hand_contact
        self.arm_controller = arm_controller
        self.hand_controller = hand_controller
        self.object_spawn_clearance = object_spawn_clearance if object_spawn_clearance is not None else 0.001
        self.bg_collision_mode = bg_collision_mode
        self.bg_use_decimated_collision_mesh = bg_use_decimated_collision_mesh
        self.bg_collision_mesh = bg_collision_mesh
        self.bg_visual_mesh = bg_visual_mesh
        self.obj_collision_mode = obj_collision_mode
        self.placement_mode = placement_mode
        self.object_placements = object_placements or {}
        self.random_placement = random_placement or {}
        self.include_objects = include_objects
        self.exclude_objects = exclude_objects
        # Normalize to str: object_actors keys are always strings; YAML/config may pass int.
        self.manip_object_id = str(manip_object_id) if manip_object_id is not None else None
        self.target_object_id = str(target_object_id) if target_object_id is not None else None
        self.task_description = task_description  # preset/config override for language instruction
        self.settle_steps = settle_steps
        self.lift_height = lift_height
        self.finger_length = finger_length
        self.robot_base_pose_z_auto = robot_base_pose_z_auto
        self._initial_object_poses: Dict[str, list] = {}
        self._initial_object_quats: Dict[str, list] = {}
        self._initial_object_poses_per_env: Dict[str, np.ndarray] = {}
        self._initial_object_quats_per_env: Dict[str, np.ndarray] = {}
        self._object_body_types: Dict[str, str] = {}
        self._object_names: Dict[str, str] = {}
        self.consecutive_grasp: torch.Tensor | None = None

        # cameras_config: из runtime config, merge local + global
        import copy
        self.cameras_config = copy.deepcopy(cameras_config) if cameras_config is not None else copy.deepcopy(DEFAULT_CAMERAS_CONFIG)
        self.lighting_config = copy.deepcopy(lighting_config) if lighting_config is not None else None

        # Store render resolution
        self.render_width = render_width if render_width is not None else 512
        self.render_height = render_height if render_height is not None else 512

        # render_camera_eye/target переопределяют cameras_config (CLI приоритет над config)
        if render_camera_eye is not None and render_camera_target is not None:
            self.custom_render_camera_pose = (np.array(render_camera_eye), np.array(render_camera_target))
            eye_l = list(self.custom_render_camera_pose[0])
            target_l = list(self.custom_render_camera_pose[1])
            print(
                f"[Camera] render_camera: eye={[round(x, 3) for x in eye_l]}, "
                f"target={[round(x, 3) for x in target_l]} (CLI override)"
            )
        else:
            self.custom_render_camera_pose = None
            rc = self.cameras_config.get("render_camera", {})
            if rc.get("use_base_camera"):
                eye, target = get_base_camera_eye_target(self.scene_json_path)
                print(f"[Camera] render_camera: from config (use_base_camera) eye={[round(x, 3) for x in eye]}, target={[round(x, 3) for x in target]}")
            else:
                eye = rc.get("eye", [0.8, 0.8, 0.6])
                target = rc.get("target", [0, 0, 0.2])
                print(f"[Camera] render_camera: from config eye={eye}, target={target}")

        self.ground = None
        self.background_actor = None
        self.object_actors: Dict[str, sapien.Entity] = {}

        if robot_base_pose is not None:
            arr = np.array(robot_base_pose, dtype=np.float64)
            if len(arr) >= 3:
                pos = arr[:3].tolist()
                quat = arr[3:7].tolist() if len(arr) >= 7 else [1.0, 0.0, 0.0, 0.0]
                self.robot_base_pose = sapien.Pose(p=pos, q=quat)
            else:
                self.robot_base_pose = None
        else:
            self.robot_base_pose = None

        self.robot_init_qpos_custom = None
        self.robot_uids = robot_uids
        if robot_init_qpos is not None:
            arr = np.array(robot_init_qpos, dtype=np.float64)
            # Panda: 9 values (7 arm + 2 gripper), WidowX: 8 values (6 arm + 2 gripper)
            if len(arr) >= 8:
                self.robot_init_qpos_custom = arr.tolist()

        if physx_contact_offset is not None or physx_rest_offset is not None:
            sim_config = dict(kwargs.get("sim_config") or {})
            scene_cfg = dict(sim_config.get("scene_config") or {})
            if physx_contact_offset is not None:
                scene_cfg["contact_offset"] = physx_contact_offset
            if physx_rest_offset is not None:
                scene_cfg["rest_offset"] = physx_rest_offset
            sim_config["scene_config"] = scene_cfg
            kwargs["sim_config"] = sim_config
            print(
                f"[PhysX] scene: contact_offset={physx_contact_offset}, "
                f"rest_offset={physx_rest_offset} "
                f"(ManiSkill default 0.02/0.0)"
            )
        elif "sim_config" not in kwargs:
            print(
                "[PhysX] scene: using ManiSkill defaults "
                "(contact_offset=0.02, rest_offset=0.0)"
            )

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        """Load the robot agent with proper initial pose."""
        pose = self.robot_base_pose if self.robot_base_pose is not None else self.ROBOT_BASE_POSE
        # Pass the full pose (position + orientation) as initial_agent_pose so it is applied
        # at build time. Calling set_pose() afterwards would attempt to write to CUDA rigid-body
        # buffers that are not yet initialised when num_envs > 1 (physx_cuda backend).
        super()._load_agent(options, pose)
        self._apply_arm_controller_tuning()
        self._apply_hand_controller_tuning()
        self._apply_hand_contact_material()

    def _apply_arm_controller_tuning(self):
        """Apply runtime arm joint drive tuning to whichever controller owns the arm joints."""
        if self.arm_controller is None:
            return

        arm_joint_names = list(getattr(self.agent, "arm_joint_names", []))
        if not arm_joint_names:
            raise RuntimeError(
                "arm_controller was provided, but agent has no arm_joint_names."
            )
        _apply_runtime_joint_controller_tuning(
            agent=self.agent,
            cfg=self.arm_controller,
            joint_names=arm_joint_names,
            scope_label="ArmController",
        )

    def _apply_hand_controller_tuning(self):
        """Apply runtime hand joint drive tuning to whichever controller owns the hand joints."""
        if self.hand_controller is None:
            return

        cfg = self.hand_controller
        if not isinstance(cfg, dict):
            raise RuntimeError(f"hand_controller must be a dict, got {type(cfg)}")

        hand_joint_names = list(getattr(self.agent, "hand_joint_names", []))
        if not hand_joint_names:
            fallback_joint_names = list(getattr(self.agent, "gripper_joint_names", []))
            if not fallback_joint_names:
                raise RuntimeError(
                    "hand_controller was provided, but agent has no hand_joint_names or gripper_joint_names."
                )
            self.agent.hand_joint_names = list(fallback_joint_names)
            hand_joint_names = list(fallback_joint_names)
            print(
                f"{_Y}[WARNING] [HandController] Agent uid='{getattr(self.agent, 'uid', 'unknown')}' "
                f"has no hand_joint_names; using gripper_joint_names={fallback_joint_names} "
                f"for runtime hand controller tuning.{_R}"
            )
        _apply_runtime_joint_controller_tuning(
            agent=self.agent,
            cfg=cfg,
            joint_names=hand_joint_names,
            scope_label="HandController",
        )

    def _apply_hand_contact_material(self):
        """Apply a custom physical material to finger collision shapes only."""
        if self.hand_contact is None and self.hand_contact_material is None:
            return

        robot = self.agent.robot
        if self.hand_contact is not None:
            materials_cfg = self.hand_contact.get("materials", self.hand_contact.get("_materials"))
            links_cfg = self.hand_contact.get("links", self.hand_contact.get("link"))
            if not isinstance(materials_cfg, dict) or not materials_cfg:
                raise RuntimeError(
                    "hand_contact requires a non-empty 'materials' (or '_materials') mapping."
                )
            if not isinstance(links_cfg, dict) or not links_cfg:
                raise RuntimeError(
                    "hand_contact requires a non-empty 'links' (or 'link') mapping."
                )

            material_by_name = {}
            for material_name, mat_cfg in materials_cfg.items():
                if not isinstance(mat_cfg, dict):
                    raise RuntimeError(f"hand_contact material '{material_name}' must be a dict, got {type(mat_cfg)}")
                material_by_name[material_name] = sapien.physx.PhysxMaterial(
                    static_friction=float(mat_cfg.get("static_friction", 0.5)),
                    dynamic_friction=float(mat_cfg.get("dynamic_friction", 0.5)),
                    restitution=float(mat_cfg.get("restitution", 0.0)),
                )

            applied_shapes = 0
            applied_descriptions = []
            for link_name, link_cfg in links_cfg.items():
                if not isinstance(link_cfg, dict):
                    raise RuntimeError(f"hand_contact link '{link_name}' must be a dict, got {type(link_cfg)}")
                material_name = link_cfg.get("material")
                if material_name not in material_by_name:
                    raise RuntimeError(
                        f"hand_contact link '{link_name}' references unknown material '{material_name}'. "
                        f"Known materials: {sorted(material_by_name.keys())}"
                    )
                link = robot.find_link_by_name(link_name)
                if link is None:
                    raise RuntimeError(f"hand_contact link '{link_name}' was not found on robot '{self.robot_uids}'.")
                raw_link_objs = getattr(link, "_objs", None)
                if not raw_link_objs:
                    raise RuntimeError(f"hand_contact link '{link_name}' has no underlying PhysX link objects.")
                all_shapes = []
                for raw_link in raw_link_objs:
                    shapes = raw_link.get_collision_shapes()
                    all_shapes.extend(list(shapes or []))
                if not all_shapes:
                    raise RuntimeError(f"hand_contact link '{link_name}' has no collision shapes to update.")
                patch_radius = link_cfg.get("patch_radius")
                min_patch_radius = link_cfg.get("min_patch_radius")
                for shape in all_shapes:
                    shape.set_physical_material(material_by_name[material_name])
                    if patch_radius is not None:
                        shape.set_patch_radius(float(patch_radius))
                    if min_patch_radius is not None:
                        shape.set_min_patch_radius(float(min_patch_radius))
                applied_shapes += len(all_shapes)
                applied_descriptions.append(
                    f"{link_name}:{material_name}"
                    + (f",patch_radius={float(patch_radius):.4f}" if patch_radius is not None else "")
                    + (f",min_patch_radius={float(min_patch_radius):.4f}" if min_patch_radius is not None else "")
                )

            print(
                "[Contact] Applied structured hand_contact to "
                f"{applied_shapes} collision shape(s) across {len(applied_descriptions)} link(s): "
                + "; ".join(applied_descriptions)
            )
            return

        mat_cfg = self.hand_contact_material
        phys_material = sapien.physx.PhysxMaterial(
            static_friction=float(mat_cfg.get("static_friction", 0.5)),
            dynamic_friction=float(mat_cfg.get("dynamic_friction", 0.5)),
            restitution=float(mat_cfg.get("restitution", 0.0)),
        )
        finger_prefixes = (
            "right_thumb",
            "right_index",
            "right_middle",
            "right_ring",
            "right_pinky",
        )
        links = robot.get_links() if hasattr(robot, "get_links") else getattr(robot, "links", [])
        applied_shapes = 0
        applied_links = []
        for link in links:
            link_name = getattr(link, "name", "")
            if not any(link_name.startswith(prefix) for prefix in finger_prefixes):
                continue
            raw_link_objs = getattr(link, "_objs", None)
            if not raw_link_objs:
                continue
            shape_count = 0
            for raw_link in raw_link_objs:
                get_shapes = getattr(raw_link, "get_collision_shapes", None)
                shapes = get_shapes() if callable(get_shapes) else getattr(raw_link, "collision_shapes", [])
                for shape in shapes:
                    shape.set_physical_material(phys_material)
                    shape_count += 1
            if shape_count > 0:
                applied_shapes += shape_count
                applied_links.append(link_name)

        if applied_shapes == 0:
            raise RuntimeError(
                "hand_contact_material was provided, but no RC5 finger collision shapes were found to update."
            )

        print(
            "[Contact] Applied legacy hand_contact_material to "
            f"{applied_shapes} collision shape(s) across {len(applied_links)} finger link(s): "
            f"static_friction={float(mat_cfg.get('static_friction', 0.5))}, "
            f"dynamic_friction={float(mat_cfg.get('dynamic_friction', 0.5))}, "
            f"restitution={float(mat_cfg.get('restitution', 0.0))}"
        )

    def _apply_robot_render_material(self):
        """Apply RL4VLA-style render material to robot (specular=0.9, roughness=0.3)."""
        try:
            robot = self.agent.robot
            objs = getattr(robot, "_objs", None)
            if objs is not None:
                for i in range(len(objs)):
                    sapien_utils.set_articulation_render_material(
                        objs[i], specular=0.9, roughness=0.3
                    )
            else:
                sapien_utils.set_articulation_render_material(
                    robot, specular=0.9, roughness=0.3
                )
        except Exception as e:
            print(f"[WARN] Could not set robot render material: {e}")

    def _compute_table_surface_z(self, bg_mesh_path: str) -> float | None:
        """Raycast downward from each object center onto the background mesh.

        Returns the median hit-point Z (the visual table surface height),
        or None if no hits were found.  Caches loaded mesh in _auto_bg_mesh
        for reuse by sim_ground_offset computation.
        """
        import trimesh

        _Y, _R = "\033[33m", "\033[0m"
        try:
            bg = trimesh.load(str(bg_mesh_path), force="mesh")
            if isinstance(bg, trimesh.Scene):
                bg = bg.dump(concatenate=True)
            self._auto_bg_mesh = bg
        except Exception as e:
            print(f"{_Y}[Auto] Failed to load background mesh for raycast: {e}{_R}")
            self._auto_bg_mesh = None
            return None

        hits: list[float] = []
        for obj_config in self.scene_config.objects.values():
            center = np.array(obj_config.center, dtype=float)
            origin = center.copy()
            origin[2] += 0.5  # start ray above the object
            origins = origin.reshape(1, 3)
            directions = np.array([[0.0, 0.0, -1.0]])
            locations, _, _ = bg.ray.intersects_location(origins, directions)
            if len(locations) > 0:
                hits.append(float(locations[:, 2].max()))

        if not hits:
            print(f"{_Y}[Auto] No raycast hits on background mesh — "
                  f"cannot determine table surface Z{_R}")
            return None

        table_z = float(np.median(hits))
        gpp_z = self.scene_config.ground_plane_point[2]
        print(f"[Auto] table_surface_z={table_z:.4f} "
              f"(raycast median from {len(hits)} object(s), "
              f"ground_plane_point_z={gpp_z:.4f}, "
              f"delta={table_z - gpp_z:.4f})")
        return table_z

    def _load_scene(self, options: dict):
        """Load all scene assets."""
        bg_path = resolve_path(self.scene_config.background_mesh_path)

        # --- Auto-placement: raycast to find true table surface ---
        _Y, _R = "\033[33m", "\033[0m"
        gpp_z = self.scene_config.ground_plane_point[2]

        if self.auto_placement:
            self.auto_table_z = self._compute_table_surface_z(bg_path)

        if self.auto_placement and self.auto_table_z is not None:
            if not self._clearance_explicit:
                _SPAWN_MARGIN = 0.005  # 5mm above collision surface
                auto_clearance = (self.auto_table_z - gpp_z) + _SPAWN_MARGIN
                self.object_spawn_clearance = auto_clearance
                print(f"[Auto] object_spawn_clearance={self.object_spawn_clearance:.4f} "
                      f"(table_surface({self.auto_table_z:.4f}) - ground_plane({gpp_z:.4f}) + margin({_SPAWN_MARGIN}))")
            else:
                print(f"[Auto] object_spawn_clearance={self.object_spawn_clearance} "
                      f"(explicit override)")

            if not self._ground_offset_explicit:
                if self._auto_bg_mesh is not None:
                    mesh_min_z = float(self._auto_bg_mesh.bounds[0][2])
                    self.sim_ground_offset = max(
                        0.1, self.auto_table_z - mesh_min_z + 0.05
                    )
                    print(f"[Auto] sim_ground_offset={self.sim_ground_offset:.4f} "
                          f"(table_surface - bg_mesh_min + 0.05)")
                else:
                    print(f"{_Y}[Auto] Cannot compute sim_ground_offset: "
                          f"background mesh not loaded{_R}")
            else:
                print(f"[Auto] sim_ground_offset={self.sim_ground_offset} "
                      f"(explicit override)")
        else:
            if self.auto_placement:
                print(f"{_Y}[Auto] auto_placement enabled but raycast failed — "
                      f"using manual defaults{_R}")

        self._auto_bg_mesh = None

        # Auto-set robot Z to table surface (mirrors visualization script post-env step).
        # Must happen before _load_background/_load_objects so the pose is ready for
        # _initialize_episode() which calls self.agent.robot.set_pose(self.robot_base_pose).
        if (self.robot_base_pose_z_auto
                and self.auto_placement
                and self.auto_table_z is not None
                and self.robot_base_pose is not None):
            old_z = float(self.robot_base_pose.p[2])
            pos = list(self.robot_base_pose.p)
            quat = list(self.robot_base_pose.q)
            # auto_table_z is in raw-mesh space; background actor is placed at
            # BACKGROUND_INIT_POS[2] + scene_z_offset, so actual table surface in
            # world frame = auto_table_z + scene_z_offset.
            robot_z = self.auto_table_z + self.scene_z_offset
            pos[2] = robot_z
            self.robot_base_pose = sapien.Pose(p=pos, q=quat)
            print(f"[Auto] robot_base_pose.z: {old_z:.4f} -> {robot_z:.4f} "
                  f"(table surface from raycast + scene_z_offset={self.scene_z_offset})")

        scene_ground_z = gpp_z + self.scene_z_offset
        sim_ground_z = scene_ground_z - self.sim_ground_offset
        self.ground = build_ground(
            self.scene,
            floor_width=100,
            altitude=sim_ground_z,
        )
        if self.use_360_background:
            self._hide_ground_visual()

        self._load_background()

        self._load_objects()

        if self.use_360_background:
            self._load_360_spheres()

        # RL4VLA-style robot render: specular=0.9, roughness=0.3 (matches Bridge dataset)
        self._apply_robot_render_material()

        self._setup_lighting()

    def _load_background(self):
        """Load the reconstructed background mesh.

        SAPIEN's GPU renderer does not apply glTF scene-graph transforms
        to the vertex buffer, so the background mesh (whose vertices are
        stored in camera coordinates with a camera-to-world transform in
        the scene graph) would appear at the wrong position.  We detect
        non-identity scene-graph transforms and bake them into the vertex
        data before loading.

        bg_collision_mode controls collision geometry:
        - "nonconvex": triangle mesh collision (exact match with visual surface,
          eliminates ~1cm gap caused by COACD convex hull approximation).
          Only valid for static actors (background is static).
        - "coacd": approximate convex decomposition (legacy behaviour, causes
          collision surface to protrude above visual surface due to
          reconstruction noise being encompassed by convex hulls).
        """
        bg_path = resolve_path(self.scene_config.background_mesh_path)
        if not bg_path.exists():
            raise FileNotFoundError(f"Background mesh not found: {bg_path}")

        _Y = "\033[33m"
        _R = "\033[0m"

        # --- Separate collision mesh (pre-decimated) ---
        bg_collision_path = str(bg_path)
        collision_mode = self.bg_collision_mode

        if self.bg_use_decimated_collision_mesh:
            coll_candidate = bg_path.parent / self.bg_collision_mesh
            if coll_candidate.exists():
                bg_collision_path = str(coll_candidate)
                collision_mode = "nonconvex"
                print(f"[Collision] background: using pre-decimated mesh {coll_candidate.name}")
            else:
                print(
                    f"{_Y}[WARNING] bg_use_decimated_collision_mesh=true but file "
                    f"'{self.bg_collision_mesh}' not found in {bg_path.parent}.\n"
                    f"  Create it with:  python decimate_mesh.py {bg_path} --target-faces 500000\n"
                    f"  Falling back to original mesh.{_R}"
                )

        builder = self.scene.create_actor_builder()

        # --- Visual mesh: use bg_visual_mesh override if specified, otherwise original ---
        if self.bg_visual_mesh:
            vis_candidate = bg_path.parent / self.bg_visual_mesh
            if vis_candidate.exists():
                bg_visual_path = str(vis_candidate)
                print(f"[Visual] background: {vis_candidate.name} (decimated, faster GPU rendering)")
            else:
                bg_visual_path = str(bg_path)
                print(f"{_Y}[WARNING] bg_visual_mesh='{self.bg_visual_mesh}' not found in "
                      f"{bg_path.parent}, falling back to original mesh.{_R}")
        else:
            bg_visual_path = str(bg_path)
            print(f"[Visual] background: {bg_path.name} (original)")
        builder.add_visual_from_file(bg_visual_path)

        _bv4_hint = (
            f"\n{_Y}  [HINT] The mesh is too large for PhysX BV4 tree construction.\n"
            f"  Run decimation to reduce face count:\n"
            f"    python decimate_mesh.py {bg_path} --target-faces 500000\n"
            f"  Then set in runtime config:\n"
            f"    bg_use_decimated_collision_mesh: true\n"
            f"    bg_collision_mesh: \"<output_filename>.glb\"{_R}"
        )

        if collision_mode == "nonconvex":
            try:
                builder.add_nonconvex_collision_from_file(bg_collision_path)
                print(f"[Collision] background: nonconvex collision loaded")
            except Exception as e:
                err = str(e).lower()
                if "bv4" in err or "triangle mesh" in err or "too many" in err:
                    print(
                        f"{_Y}[WARNING] background nonconvex collision FAILED — "
                        f"BV4 tree build error:{_R}\n  {e}{_bv4_hint}"
                    )
                else:
                    print(
                        f"{_Y}[Collision FALLBACK] background: nonconvex failed ({e}), "
                        f"falling back to COACD{_R}"
                    )
                builder.add_multiple_convex_collisions_from_file(
                    bg_collision_path, decomposition="coacd"
                )
        else:
            try:
                builder.add_multiple_convex_collisions_from_file(
                    bg_collision_path, decomposition="coacd"
                )
                print(f"[Collision] background: COACD collision loaded")
            except Exception as e:
                print(
                    f"{_Y}[Collision FALLBACK] background: COACD failed ({e}), "
                    f"falling back to single convex hull{_R}"
                )
                builder.add_multiple_convex_collisions_from_file(
                    bg_collision_path, decomposition="none"
                )

        bg_pos = [self.BACKGROUND_INIT_POS[0], self.BACKGROUND_INIT_POS[1],
                  self.BACKGROUND_INIT_POS[2] + self.scene_z_offset]
        builder.set_initial_pose(
            sapien.Pose(p=bg_pos, q=self.BACKGROUND_INIT_QUAT)
        )
        self.background_actor = builder.build_static(name="background")

    def _hide_ground_visual(self):
        """Keep the physics floor, hide the checkerboard so the 360 sphere shows."""
        ground = getattr(self, "ground", None)
        objs = getattr(ground, "_objs", None) or []
        for obj in objs:
            body = obj.find_component_by_type(sapien.render.RenderBodyComponent)
            if body is None:
                continue
            body.visibility = 0
            if hasattr(body, "disable"):
                body.disable()

    def _pano_sphere_center_xyz(self) -> np.ndarray:
        xy_min, xy_max = self._compute_table_bounds()
        z = float(self._get_object_support_z())
        return np.array(
            [
                0.5 * (float(xy_min[0]) + float(xy_max[0])),
                0.5 * (float(xy_min[1]) + float(xy_max[1])),
                z,
            ],
            dtype=np.float32,
        )

    def _load_360_spheres(self):
        """Build one inverted sphere and preload every 360 photo texture."""
        photos = list_360_photo_paths(self.pano_photos_dir)
        if not photos:
            print(
                f"{_Y}[360] no photos in {self.pano_photos_dir}; "
                f"wrist background will stay black{_R}"
            )
            self._pano_sphere_actor = None
            self._pano_textures = []
            return

        self._pano_center = self._pano_sphere_center_xyz()
        n_env = int(self.num_envs)
        self._pano_photo_idx = np.zeros(n_env, dtype=np.int32)
        self._pano_yaw = np.zeros(n_env, dtype=np.float32)
        textures = []
        paths = []
        print(
            f"[360] loading {len(photos)} photo(s) from {self.pano_photos_dir} "
            f"(radius={self.pano_sphere_radius:.1f}m, center={self._pano_center.tolist()})"
        )
        for photo in photos:
            texture_path = resolve_equirect_texture(photo)
            textures.append(load_pano_texture(texture_path))
            paths.append(photo)
        actor, material = build_pano_sphere_actor(
            self.scene,
            resolve_equirect_texture(photos[0]),
            name="pano_sphere",
            radius=self.pano_sphere_radius,
            initial_pose=sapien.Pose(p=self._pano_center.tolist(), q=[1.0, 0.0, 0.0, 0.0]),
        )
        self._pano_sphere_actor = actor
        self._pano_sphere_material = material
        self._pano_textures = textures
        self._pano_photo_paths = paths
        print(f"[360] inverted sphere ready ({len(textures)} equirect textures)")

    def _apply_pano_texture(self, texture):
        material = self._pano_sphere_material
        if material is None:
            return
        material.set_base_color_texture(texture)
        material.set_emission_texture(None)

    def _randomize_360_spheres(self, env_idx: torch.Tensor, options: dict):
        """Pick a random photo and yaw for each resetting env."""
        actor = self._pano_sphere_actor
        textures = self._pano_textures
        if actor is None or not textures:
            return
        env_indices = [int(i) for i in env_idx.detach().cpu().tolist()]
        n_env = int(self.num_envs)
        n_photos = len(textures)
        if self._pano_photo_idx is None or len(self._pano_photo_idx) != n_env:
            self._pano_photo_idx = np.zeros(n_env, dtype=np.int32)
            self._pano_yaw = np.zeros(n_env, dtype=np.float32)
        seed_offset = int((self.random_placement or {}).get("seed") or 0)
        self._pano_reset_counter = int(getattr(self, "_pano_reset_counter", 0)) + 1
        options = options or {}
        raw = options.get("episode_id")
        for env_i in env_indices:
            if raw is not None:
                arr = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
                arr = np.atleast_1d(arr).astype(np.int64).reshape(-1)
                if arr.size == 1:
                    episode_id = int(arr[0]) + int(env_i)
                else:
                    episode_id = int(arr[int(env_i) % arr.size])
            else:
                episode_id = int(self._pano_reset_counter) * 36007 + int(env_i)
            rng = np.random.RandomState((seed_offset + episode_id + 17) % (2**31 - 1))
            self._pano_photo_idx[env_i] = int(rng.randint(0, n_photos))
            self._pano_yaw[env_i] = float(rng.uniform(0.0, 2.0 * np.pi))

        center = (
            np.asarray(self._pano_center, dtype=np.float32)
            if self._pano_center is not None
            else self._pano_sphere_center_xyz()
        )
        positions = np.repeat(center.reshape(1, 3), n_env, axis=0)
        quats = np.zeros((n_env, 4), dtype=np.float32)
        for env_i in range(n_env):
            quats[env_i] = np.asarray(
                euler2quat(0.0, 0.0, float(self._pano_yaw[env_i])),
                dtype=np.float32,
            )
        actor.set_pose(Pose.create_from_pq(p=positions, q=quats))
        if getattr(self.scene, "gpu_sim_enabled", False):
            try:
                self.scene.px.gpu_apply_rigid_dynamic_data()
            except Exception:
                pass
        # One shared material: use the first resetting env's photo.
        photo_i = int(self._pano_photo_idx[env_indices[0]])
        self._apply_pano_texture(textures[photo_i])
        path = self._pano_photo_paths[photo_i] if self._pano_photo_paths else None
        print(
            f"[360] env {env_indices[0]}: photo={path.name if path is not None else photo_i} "
            f"yaw={float(self._pano_yaw[env_indices[0]]):.3f} rad"
        )

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        if not getattr(self, "use_360_background", False):
            return
        for name, sensor in getattr(self, "_sensors", {}).items():
            camera = getattr(sensor, "camera", None)
            if camera is None or not hasattr(camera, "far"):
                continue
            try:
                old_far = float(camera.far)
            except Exception:
                continue
            if old_far < WRIST_CAMERA_FAR_PANO:
                camera.far = float(WRIST_CAMERA_FAR_PANO)
                print(f"[360] {name} far {old_far:g} -> {WRIST_CAMERA_FAR_PANO:g}")

    def _get_object_support_z(self) -> float:
        """Return the Z of the surface objects should rest on."""
        if self.auto_table_z is not None:
            return float(self.auto_table_z + self.scene_z_offset)
        return float(self.scene_config.ground_plane_point[2] + self.scene_z_offset)

    def _get_spawn_clearance(self, placement_cfg=None) -> float:
        """Return per-object clearance override or the global default."""
        if placement_cfg is not None and "spawn_clearance" in placement_cfg:
            return float(placement_cfg["spawn_clearance"])
        return float(self.object_spawn_clearance)

    def _compute_spawn_pose(self, obj_id, world_center_xy, orientation_quat, bbox, spawn_clearance=None):
        """Compute actor origin so mesh center sits at *world_center_xy* on the table.

        With centering_pose, actor origin = mesh geometric center.  We compute
        Z so the lowest point of the (optionally rotated) centered bbox sits
        on the table, and XY = world_center_xy.

        Returns:
            (origin_p, quat): position list[3] and quat list[4].
        """
        bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
        half = 0.5 * (bbox_max - bbox_min)
        support_surface_z = self._get_object_support_z()
        if spawn_clearance is None:
            spawn_clearance = self.object_spawn_clearance

        q = np.asarray(orientation_quat, dtype=float)
        R = quat2mat(q)

        corners = np.array([
            [s0 * half[0], s1 * half[1], s2 * half[2]]
            for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)
        ])
        rotated = (R @ corners.T).T
        lowest_z = float(rotated[:, 2].min())

        origin_z = support_surface_z + float(spawn_clearance) - lowest_z
        origin_xy = np.array(world_center_xy, dtype=float)
        return [float(origin_xy[0]), float(origin_xy[1]), float(origin_z)], list(q)

    def _resolve_fixed_env_placements(self, placement_cfg, *, label: str):
        if not isinstance(placement_cfg, dict):
            return None
        positions = placement_cfg.get("position_per_env")
        if positions is None:
            return None
        if not isinstance(positions, (list, tuple)) or len(positions) != int(self.num_envs):
            raise ValueError(
                f"{label}.position_per_env must define exactly num_envs={self.num_envs} positions"
            )
        orientations = placement_cfg.get("orientation_per_env")
        if orientations is None:
            orientations = [placement_cfg.get("orientation", [1, 0, 0, 0]) for _ in range(int(self.num_envs))]
        if not isinstance(orientations, (list, tuple)) or len(orientations) != int(self.num_envs):
            raise ValueError(
                f"{label}.orientation_per_env must define exactly num_envs={self.num_envs} quaternions"
            )

        normalized_positions = []
        normalized_orientations = []
        for env_index, position in enumerate(positions):
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise ValueError(f"{label}.position_per_env[{env_index}] must define exactly 3 values")
            normalized_positions.append([float(value) for value in position])
        for env_index, orientation in enumerate(orientations):
            if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
                raise ValueError(f"{label}.orientation_per_env[{env_index}] must define exactly 4 values")
            normalized_orientations.append([float(value) for value in orientation])
        return normalized_positions, normalized_orientations

    def _compute_fixed_spawn_pose_bundle(self, obj_id, placement_cfg, bbox, *, label: str):
        per_env = self._resolve_fixed_env_placements(placement_cfg, label=label)
        if per_env is None:
            target_xy = np.array(placement_cfg["position"][:2], dtype=float)
            init_quat = placement_cfg.get("orientation", [1, 0, 0, 0])
            spawn_clearance = self._get_spawn_clearance(placement_cfg)
            if bbox is not None:
                init_pose_p, init_quat = self._compute_spawn_pose(
                    obj_id, target_xy, init_quat, bbox, spawn_clearance=spawn_clearance
                )
            else:
                init_pose_p = [float(target_xy[0]), float(target_xy[1]), self._get_object_support_z()]
            return init_pose_p, init_quat, None, None

        per_env_positions_raw, per_env_orientations_raw = per_env
        per_env_positions = []
        per_env_orientations = []
        for position, orientation in zip(per_env_positions_raw, per_env_orientations_raw):
            target_xy = np.array(position[:2], dtype=float)
            spawn_clearance = self._get_spawn_clearance(placement_cfg)
            if bbox is not None:
                env_pose_p, env_quat = self._compute_spawn_pose(
                    obj_id, target_xy, orientation, bbox, spawn_clearance=spawn_clearance
                )
            else:
                env_pose_p = [float(target_xy[0]), float(target_xy[1]), self._get_object_support_z()]
                env_quat = list(orientation)
            per_env_positions.append(env_pose_p)
            per_env_orientations.append(env_quat)

        return (
            list(per_env_positions[0]),
            list(per_env_orientations[0]),
            np.asarray(per_env_positions, dtype=np.float32),
            np.asarray(per_env_orientations, dtype=np.float32),
        )

    def _resolve_object_asset_paths(self, obj_config, placement_cfg=None) -> tuple[Path, Path]:
        placement_cfg = placement_cfg or {}
        mesh_path_override = placement_cfg.get("mesh_path")
        collision_mesh_override = placement_cfg.get("collision_mesh_path")

        if mesh_path_override:
            visual_path = resolve_path(str(mesh_path_override))
        else:
            visual_path = resolve_path(obj_config.mesh_path)

        collision_cfg_path = collision_mesh_override or getattr(obj_config, "collision_mesh_path", None)
        if collision_cfg_path:
            collision_path = resolve_path(str(collision_cfg_path))
        else:
            collision_path = visual_path

        return visual_path, collision_path

    def _compute_table_bounds(self):
        """Compute XY placement bounds from object effective centers in scene.json.

        Falls back to scene_aabb if explicit bounds are given in random_placement config.
        Caches result in self._table_bounds_xy = (xy_min, xy_max).
        """
        if hasattr(self, "_table_bounds_xy"):
            return self._table_bounds_xy

        cfg = self.random_placement or {}

        if "bounds_min" in cfg and "bounds_max" in cfg:
            bmin = np.asarray(cfg["bounds_min"][:2], dtype=float)
            bmax = np.asarray(cfg["bounds_max"][:2], dtype=float)
            print(f"[Placement] Table bounds from config: "
                  f"X=[{bmin[0]:.3f}, {bmax[0]:.3f}], Y=[{bmin[1]:.3f}, {bmax[1]:.3f}]")
            self._table_bounds_xy = (bmin, bmax)
            return self._table_bounds_xy

        if self.placement_mode == "random":
            bmin = np.asarray(DEFAULT_REACHABLE_BOUNDS_MIN_XY, dtype=float)
            bmax = np.asarray(DEFAULT_REACHABLE_BOUNDS_MAX_XY, dtype=float)
            print(
                "[Placement] Table bounds from reachable RC5 workspace: "
                f"X=[{bmin[0]:.3f}, {bmax[0]:.3f}], Y=[{bmin[1]:.3f}, {bmax[1]:.3f}]"
            )
            self._table_bounds_xy = (bmin, bmax)
            return self._table_bounds_xy

        centers_xy = []
        for obj_config in self.scene_config.objects.values():
            ec = obj_config.center
            centers_xy.append([ec[0], ec[1]])

        if not centers_xy:
            bmin = np.asarray(self.scene_config.scene_aabb_min[:2])
            bmax = np.asarray(self.scene_config.scene_aabb_max[:2])
        else:
            pts = np.array(centers_xy)
            xy_min = pts.min(axis=0)
            xy_max = pts.max(axis=0)
            padding = cfg.get("auto_bounds_padding", 0.05)
            bmin = xy_min - padding
            bmax = xy_max + padding

        print(f"[Placement] Table bounds (auto from object centers): "
              f"X=[{bmin[0]:.3f}, {bmax[0]:.3f}], Y=[{bmin[1]:.3f}, {bmax[1]:.3f}]")
        self._table_bounds_xy = (bmin, bmax)
        return self._table_bounds_xy

    def _robot_base_xy_np(self) -> np.ndarray:
        pose = self.robot_base_pose if self.robot_base_pose is not None else self.ROBOT_BASE_POSE
        p = np.asarray(pose.p, dtype=np.float64).reshape(-1)
        return p[:2]

    def _object_xy_radius(self, obj_id) -> float:
        return xy_half_extent_from_bbox(self.object_bbox_bounds.get(obj_id))

    def _is_movable_object(self, obj_id) -> bool:
        return self._object_body_types.get(obj_id, "dynamic") == "dynamic"

    def _ensure_per_env_pose_buffers(self, obj_id):
        n = int(self.num_envs)
        if obj_id not in self._initial_object_poses_per_env:
            p = np.asarray(self._initial_object_poses[obj_id], dtype=np.float32)
            q = np.asarray(
                self._initial_object_quats.get(obj_id, self.OBJECT_INIT_QUAT),
                dtype=np.float32,
            )
            self._initial_object_poses_per_env[obj_id] = np.tile(p.reshape(1, 3), (n, 1))
            self._initial_object_quats_per_env[obj_id] = np.tile(q.reshape(1, 4), (n, 1))

    def _episode_ids_for_reset(self, options, env_indices) -> dict[int, int]:
        options = options or {}
        ids: dict[int, int] = {}
        raw = options.get("episode_id")
        if raw is not None:
            arr = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
            arr = np.atleast_1d(arr).astype(np.int64).reshape(-1)
            for env_i in env_indices:
                if arr.size == 1:
                    ids[int(env_i)] = int(arr[0]) + int(env_i)
                elif int(env_i) < arr.size:
                    ids[int(env_i)] = int(arr[int(env_i)])
                else:
                    ids[int(env_i)] = int(arr[int(env_i) % arr.size])
            return ids
        self._placement_reset_counter += 1
        for env_i in env_indices:
            ids[int(env_i)] = int(self._placement_reset_counter) * 10007 + int(env_i)
        return ids

    def _random_place_object(self, obj_id, occupied, rng):
        """Pick a reachable XY that does not overlap occupied footprints.

        `occupied` is a list of `(xy, radius)` pairs and is updated in place.
        Returns ([x, y], quat_sapien) or None if placement failed.
        """
        cfg = self.random_placement or {}
        xy_min, xy_max = self._compute_table_bounds()
        margin = float(cfg.get("table_margin", 0.0))
        lo = np.asarray(xy_min, dtype=np.float64)[:2] + margin
        hi = np.asarray(xy_max, dtype=np.float64)[:2] - margin
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            lo = np.asarray(xy_min, dtype=np.float64)[:2]
            hi = np.asarray(xy_max, dtype=np.float64)[:2]

        radius = self._object_xy_radius(obj_id)
        xy = sample_nonoverlapping_xy(
            rng,
            radius=radius,
            bounds_min=lo,
            bounds_max=hi,
            occupied=occupied,
            robot_base_xy=self._robot_base_xy_np(),
            min_robot_clearance=float(cfg.get("min_robot_clearance", DEFAULT_MIN_ROBOT_CLEARANCE)),
            pair_gap=float(cfg.get("pair_gap", cfg.get("min_object_distance", DEFAULT_PAIR_GAP))),
            max_attempts=int(cfg.get("max_attempts", 80)),
        )
        if xy is None:
            print(
                f"[WARN] Random placement failed for object {obj_id} after "
                f"{cfg.get('max_attempts', 80)} attempts, keeping previous pose"
            )
            return None
        quat_sapien = list(self._initial_object_quats.get(obj_id, self.OBJECT_INIT_QUAT))
        occupied.append((xy, radius))
        return [float(xy[0]), float(xy[1])], quat_sapien

    def _randomize_episode_object_poses(self, env_idx: torch.Tensor, options: dict):
        env_indices = [int(i) for i in env_idx.detach().cpu().tolist()]
        episode_ids = self._episode_ids_for_reset(options, env_indices)
        movable = [oid for oid in self.object_actors if self._is_movable_object(oid)]
        fixtures = [oid for oid in self.object_actors if not self._is_movable_object(oid)]
        seed_offset = int(self.random_placement.get("seed") or 0)

        for env_i in env_indices:
            rng = np.random.RandomState((seed_offset + int(episode_ids[env_i]) + 7919) % (2**31 - 1))
            occupied: list[tuple[np.ndarray, float]] = []
            for oid in fixtures:
                p = np.asarray(self._initial_object_poses[oid], dtype=np.float64)
                occupied.append((p[:2], self._object_xy_radius(oid)))
            for oid in movable:
                bbox = self.object_bbox_bounds.get(oid)
                result = self._random_place_object(oid, occupied, rng)
                if result is None or bbox is None:
                    continue
                xy, quat = result
                pl = self.object_placements.get(str(oid), {}) or {}
                spawn_clearance = self._get_spawn_clearance(pl)
                p, q = self._compute_spawn_pose(oid, xy, quat, bbox, spawn_clearance=spawn_clearance)
                p[2] += float(pl.get("z_extra", 0.0))
                self._ensure_per_env_pose_buffers(oid)
                self._initial_object_poses_per_env[oid][env_i] = np.asarray(p, dtype=np.float32)
                self._initial_object_quats_per_env[oid][env_i] = np.asarray(q, dtype=np.float32)
                if env_i == env_indices[0]:
                    self._initial_object_poses[oid] = p
                    self._initial_object_quats[oid] = q

    def _load_objects(self):
        """Load reconstructed object meshes as dynamic actors.

        Meshes from the reconstruction pipeline have vertices in **world
        coordinates**.  We apply ``centering_pose = Pose(p=-mesh_center)`` as
        local_pose on both visual and collision shapes so that the actor origin
        coincides with the mesh geometric center.  This ensures that actor
        rotation acts around the mesh center (not around world origin) and that
        OBB / grasp calculations work correctly.

        The actor is then placed at the desired world position of the mesh
        center (= object_center from scene.json for scene mode, or random XY
        for random mode).
        """
        import trimesh

        _Y = "\033[33m"
        _R = "\033[0m"

        # --- Object filtering: include_objects vs exclude_objects ---
        skip_ids: set = set()
        allowed_ids: set | None = None
        excluded_ids: set = set()
        if self.include_objects is not None and self.exclude_objects is not None:
            print(f"{_Y}[WARNING] Both include_objects and exclude_objects set — "
                  f"using include_objects (higher priority){_R}")
            self.exclude_objects = None

        if self.include_objects is not None:
            if self.include_objects == "manip_only":
                # Special mode: load only the manipulation target object.
                # Resolves manip_object_id → scene_config.manipulated_oid → first object.
                if self.manip_object_id is not None:
                    manip_id = str(self.manip_object_id)
                elif self.scene_config.manipulated_oid is not None:
                    manip_id = str(self.scene_config.manipulated_oid)
                else:
                    manip_id = str(next(iter(self.scene_config.objects)))
                allowed_ids = {manip_id}
                print(f"[Filter] include_objects='manip_only' → manip_id={manip_id!r}")
            else:
                allowed_ids = {str(x) for x in self.include_objects}
            skip_ids = {oid for oid in self.scene_config.objects if oid not in allowed_ids}
            print(f"[Filter] keeping {len(allowed_ids)} object(s), "
                  f"skipping {len(skip_ids)}")
        elif self.exclude_objects is not None:
            excluded_ids = {str(x) for x in self.exclude_objects}
            skip_ids = set(excluded_ids)
            print(f"[Filter] exclude_objects={list(self.exclude_objects)} — "
                  f"skipping {len(skip_ids)} object(s)")

        scene_ground_z = self.scene_config.ground_plane_point[2] + self.scene_z_offset
        if not hasattr(self, "object_origin_offsets"):
            self.object_origin_offsets = {}
        if not hasattr(self, "object_bbox_bounds"):
            self.object_bbox_bounds = {}

        print(f"[Collision] Objects: requested decomposition={self.obj_collision_mode!r}")
        print(f"[Placement] mode={self.placement_mode!r}")

        for idx, (obj_id, obj_config) in enumerate(self.scene_config.objects.items()):
            if obj_id in skip_ids:
                print(f"[Filter] {obj_config.name} (id={obj_id}): skipped")
                continue

            pl_cfg = self.object_placements.get(str(obj_id), {})
            mesh_path_override = pl_cfg.get("mesh_path")
            obj_path, obj_collision_path = self._resolve_object_asset_paths(obj_config, pl_cfg)
            if mesh_path_override:
                print(f"[Placement] {obj_config.name}: mesh overridden → {obj_path.name}")
            if not obj_path.exists():
                print(f"[WARN] Skipping {obj_config.name}: mesh not found at {obj_path}")
                continue
            if not obj_collision_path.exists():
                print(f"[WARN] Skipping {obj_config.name}: collision mesh not found at {obj_collision_path}")
                continue
            if obj_collision_path != obj_path:
                print(f"[Collision] {obj_config.name}: using explicit collision mesh {obj_collision_path.name}")

            object_center = np.array(obj_config.center, dtype=float)
            visual_mesh = None
            collision_mesh = None
            visual_mesh_center_geom = np.zeros(3, dtype=float)
            collision_mesh_center_geom = np.zeros(3, dtype=float)

            try:
                visual_mesh = trimesh.load(str(obj_path), force="mesh")
                if isinstance(visual_mesh, trimesh.Scene):
                    visual_mesh = visual_mesh.dump(concatenate=True)
                if visual_mesh.vertices.shape[0] > 0:
                    visual_mesh_center_geom = 0.5 * (visual_mesh.bounds[0] + visual_mesh.bounds[1])
            except Exception as e:
                print(f"[WARN] Failed to load visual mesh for {obj_config.name}: {e}")

            try:
                collision_mesh = trimesh.load(str(obj_collision_path), force="mesh")
                if isinstance(collision_mesh, trimesh.Scene):
                    collision_mesh = collision_mesh.dump(concatenate=True)
                if collision_mesh.vertices.shape[0] > 0:
                    collision_mesh_center_geom = 0.5 * (collision_mesh.bounds[0] + collision_mesh.bounds[1])
                    centered_min = collision_mesh.bounds[0] - collision_mesh_center_geom
                    centered_max = collision_mesh.bounds[1] - collision_mesh_center_geom
                    self.object_origin_offsets[obj_id] = np.zeros(3, dtype=float)
                    self.object_bbox_bounds[obj_id] = (
                        np.array(centered_min, dtype=float),
                        np.array(centered_max, dtype=float),
                    )
                else:
                    self.object_origin_offsets[obj_id] = np.zeros(3, dtype=float)
                    bbox_min = np.array(obj_config.bbox_min, dtype=float)
                    bbox_max = np.array(obj_config.bbox_max, dtype=float)
                    self.object_bbox_bounds[obj_id] = (bbox_min, bbox_max)
            except Exception as e:
                print(f"[WARN] Failed to load collision mesh for {obj_config.name}: {e}")
                self.object_origin_offsets[obj_id] = np.zeros(3, dtype=float)
                bbox_min = np.array(obj_config.bbox_min, dtype=float)
                bbox_max = np.array(obj_config.bbox_max, dtype=float)
                self.object_bbox_bounds[obj_id] = (bbox_min, bbox_max)

            # Actor origin = mesh geometric center in world frame.
            bbox = self.object_bbox_bounds.get(obj_id)
            init_quat = list(self.OBJECT_INIT_QUAT)

            # Read scale / spawn pose from placement config whenever it is present.
            # Random mode still uses these as the initial pose, then jitters XY on reset.
            pl = self.object_placements.get(str(obj_id), {}) or {}
            obj_scale = float(pl.get("scale", 1.0))

            # centering_pose must account for scale: mesh verts are scaled first,
            # then pose is applied, so offset = -(mesh_center * scale)
            scaled_visual_center = visual_mesh_center_geom * obj_scale
            visual_centering_pose = sapien.Pose(p=-scaled_visual_center)
            scaled_collision_center = collision_mesh_center_geom * obj_scale
            collision_centering_pose = sapien.Pose(p=-scaled_collision_center)

            # Centered vertices (accounting for scale) for Z-correction.
            centered_collision_verts = None
            if collision_mesh is not None and collision_mesh.vertices.shape[0] > 0:
                centered_collision_verts = (collision_mesh.vertices - collision_mesh_center_geom) * obj_scale

            if obj_scale != 1.0 and bbox is not None:
                bbox = (bbox[0] * obj_scale, bbox[1] * obj_scale)
                self.object_bbox_bounds[obj_id] = bbox
                print(f"[Placement] {obj_config.name}: scale={obj_scale}")

            # Z: place mesh bottom on the support surface (table if known)
            bbox_min_z = float(bbox[0][2]) if bbox is not None else 0.0
            spawn_clearance = self._get_spawn_clearance(pl)
            support_surface_z = self._get_object_support_z()
            origin_z = support_surface_z + spawn_clearance - bbox_min_z

            if pl.get("position") is not None:
                (
                    init_pose_p,
                    init_quat,
                    per_env_init_pose_p,
                    per_env_init_quat,
                ) = self._compute_fixed_spawn_pose_bundle(
                    obj_id,
                    pl,
                    bbox,
                    label=f"object_placements.{obj_id}",
                )
                if per_env_init_pose_p is not None:
                    self._initial_object_poses_per_env[obj_id] = per_env_init_pose_p
                    self._initial_object_quats_per_env[obj_id] = per_env_init_quat

                # OBB-corner Z underestimates the actual mesh bottom → the
                # object "floats" by that error.  At scale=1 this error was
                # absorbed into the tuned clearance.  When scale changes the
                # error scales proportionally, shifting the object down.
                # Compensate so the mesh bottom stays at the same absolute Z.
                if centered_collision_verts is not None and obj_scale != 1.0 and bbox is not None:
                    R = quat2mat(np.asarray(init_quat, dtype=float))
                    exact_lowest = float((R @ centered_collision_verts.T).T[:, 2].min())
                    bmin, bmax = np.asarray(bbox[0]), np.asarray(bbox[1])
                    half = 0.5 * (bmax - bmin)
                    corners = np.array([
                        [s0 * half[0], s1 * half[1], s2 * half[2]]
                        for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)
                    ])
                    obb_lowest = float((R @ corners.T).T[:, 2].min())
                    delta_at_scale = exact_lowest - obb_lowest
                    delta_at_1 = delta_at_scale / obj_scale
                    z_correction = delta_at_1 - delta_at_scale
                    init_pose_p[2] += z_correction
                    print(f"[Placement] {obj_config.name}: "
                          f"Z correction +{z_correction:.4f}m "
                          f"(OBB→mesh error compensation for scale={obj_scale})")

                z_extra = float(pl.get("z_extra", 0.0))
                if z_extra != 0.0:
                    init_pose_p[2] += z_extra
                    if per_env_init_pose_p is not None:
                        per_env_init_pose_p[:, 2] += z_extra
                        self._initial_object_poses_per_env[obj_id] = per_env_init_pose_p
                    print(f"[Placement] {obj_config.name}: z_extra={z_extra:+.4f}m applied")

                print(f"[Placement] {obj_config.name}: fixed override")
            else:
                # scene & random: start at original mesh center XY
                init_pose_p = [float(visual_mesh_center_geom[0]),
                               float(visual_mesh_center_geom[1]), origin_z]

            # --- Build collision shape ---
            mat_cfg = self.object_material
            phys_material = sapien.physx.PhysxMaterial(
                static_friction=float(mat_cfg.get("static_friction", 0.5)),
                dynamic_friction=float(mat_cfg.get("dynamic_friction", 0.5)),
                restitution=float(mat_cfg.get("restitution", 0.0)),
            )

            _Y = "\033[33m"  # yellow
            _R = "\033[0m"   # reset

            decomposition = str(pl.get("collision_mode", self.obj_collision_mode))
            body_type = str(pl.get("body_type", "kinematic" if pl.get("kinematic", False) else "dynamic"))

            # Detect degenerate (near-flat) meshes whose COACD / convex hull
            # produces collision shapes too thin for reliable PhysX contact.
            # For such objects we use a box collision matching the OBB.
            _MIN_HULL_VOLUME = 1e-7  # m³ — below this mesh collision is unreliable
            use_box_collision = False
            hull_vol = 0.0
            if collision_mesh is not None:
                try:
                    hull_vol = float(collision_mesh.convex_hull.volume)
                except Exception:
                    hull_vol = 0.0
                obj_size = collision_mesh.bounds[1] - collision_mesh.bounds[0]
                print(f"[Collision] {obj_config.name}: size=[{obj_size[0]:.4f}, {obj_size[1]:.4f}, {obj_size[2]:.4f}], "
                      f"convex_hull_volume={hull_vol:.2e}")
                if hull_vol < _MIN_HULL_VOLUME:
                    use_box_collision = True
                    print(f"{_Y}[Collision] {obj_config.name}: convex hull volume {hull_vol:.2e} < {_MIN_HULL_VOLUME:.0e} — "
                          f"mesh is near-flat{_R}")
                    print(f"{_Y}  FALLBACK: using box collision (OBB) for reliable PhysX contact{_R}")
                elif not collision_mesh.is_watertight:
                    print(
                        f"{_Y}[Collision] {obj_config.name}: mesh not watertight "
                        f"(verts={collision_mesh.vertices.shape[0]}, euler={collision_mesh.euler_number}), "
                        f"trying COACD anyway (works on triangle soups){_R}"
                    )

            builder = self.scene.create_actor_builder()
            scale_vec = [obj_scale] * 3 if obj_scale != 1.0 else None
            builder.add_visual_from_file(
                str(obj_path), pose=visual_centering_pose,
                **({"scale": scale_vec} if scale_vec else {}))

            collision_ok = False

            if use_box_collision and bbox is not None:
                half_size = [
                    float(abs(bbox[1][0] - bbox[0][0])) / 2,
                    float(abs(bbox[1][1] - bbox[0][1])) / 2,
                    float(abs(bbox[1][2] - bbox[0][2])) / 2,
                ]
                # Ensure minimum half-size for PhysX stability
                half_size = [max(h, 0.001) for h in half_size]
                builder.add_box_collision(
                    half_size=half_size,
                    material=phys_material,
                    pose=collision_centering_pose,
                )
                collision_ok = True
                print(f"[Collision] {obj_config.name}: box collision half_size="
                      f"[{half_size[0]:.4f}, {half_size[1]:.4f}, {half_size[2]:.4f}]")

            if not collision_ok and decomposition == "nonconvex":
                try:
                    builder.add_nonconvex_collision_from_file(
                        str(obj_collision_path),
                        material=phys_material,
                        pose=collision_centering_pose,
                        **({"scale": scale_vec} if scale_vec else {}),
                    )
                    collision_ok = True
                    print(f"[Collision] {obj_config.name}: nonconvex collision loaded")
                except Exception as e:
                    print(
                        f"{_Y}[Collision FALLBACK] {obj_config.name}: nonconvex failed ({e}), "
                        f"falling back to {'COACD' if decomposition == 'coacd' else 'single convex hull'}{_R}"
                    )
            if not collision_ok and decomposition == "coacd":
                try:
                    builder.add_multiple_convex_collisions_from_file(
                        str(obj_collision_path),
                        decomposition="coacd",
                        material=phys_material,
                        pose=collision_centering_pose,
                        **({"scale": scale_vec} if scale_vec else {}),
                    )
                    collision_ok = True
                    print(f"[Collision] {obj_config.name}: COACD decomposition succeeded")
                except Exception as e:
                    print(
                        f"{_Y}[Collision FALLBACK] {obj_config.name}: COACD failed ({e}), "
                        f"falling back to single convex hull{_R}"
                    )
            if not collision_ok:
                builder.add_multiple_convex_collisions_from_file(
                    str(obj_collision_path),
                    decomposition="none",
                    material=phys_material,
                    pose=collision_centering_pose,
                    **({"scale": scale_vec} if scale_vec else {}),
                )
                print(
                    f"{_Y}[Collision FALLBACK] {obj_config.name}: using single convex hull{_R}"
                )

            builder.set_initial_pose(sapien.Pose(p=init_pose_p, q=init_quat))
            if body_type == "kinematic":
                actor = builder.build_kinematic(name=f"object_{obj_config.name}")
            elif body_type == "static":
                actor = builder.build_static(name=f"object_{obj_config.name}")
            else:
                actor = builder.build(name=f"object_{obj_config.name}")
            self.object_actors[obj_id] = actor
            self._initial_object_poses[obj_id] = init_pose_p
            self._initial_object_quats[obj_id] = init_quat
            self._object_body_types[obj_id] = body_type
            self._object_names[obj_id] = obj_config.name

            print(f"[Placement] {obj_config.name}: actor_p=[{init_pose_p[0]:.4f},{init_pose_p[1]:.4f},{init_pose_p[2]:.4f}] "
                  f"(mesh center at actor origin, body_type={body_type}, collision_mode={decomposition})")

        # --- External objects: entries in object_placements not found in scene_config.objects.
        # Requires mesh_path and position to be set explicitly in the placement config.
        scene_obj_ids = set(self.scene_config.objects.keys())
        for ext_id_raw, pl in self.object_placements.items():
            ext_id = str(ext_id_raw)
            if ext_id in scene_obj_ids:
                continue  # already handled in the main loop above
            if allowed_ids is not None and ext_id not in allowed_ids:
                print(f"[Filter] external object id={ext_id}: skipped (not in include_objects)")
                continue
            if ext_id in excluded_ids:
                print(f"[Filter] external object id={ext_id}: skipped (in exclude_objects)")
                continue
            if "mesh_path" not in pl:
                raise ValueError(
                    f"External object '{ext_id}' in object_placements must define 'mesh_path'"
                )
            if "position" not in pl:
                raise ValueError(
                    f"External object '{ext_id}' in object_placements must define 'position'"
                )

            ext_path = resolve_app_runtime_path(pl["mesh_path"])
            if not ext_path.exists():
                raise FileNotFoundError(
                    f"External object '{ext_id}': mesh not found at {ext_path}"
                )
            ext_collision_path = resolve_app_runtime_path(pl.get("collision_mesh_path", pl["mesh_path"]))
            if not ext_collision_path.exists():
                raise FileNotFoundError(
                    f"External object '{ext_id}': collision mesh not found at {ext_collision_path}"
                )

            ext_name = pl.get("name", ext_id)
            ext_scale = float(pl.get("scale", 1.0))
            ext_quat = pl.get("orientation", [1, 0, 0, 0])
            ext_collision_mode = str(pl.get("collision_mode", self.obj_collision_mode))
            ext_body_type = str(pl.get("body_type", "kinematic" if pl.get("kinematic", False) else "dynamic"))

            mesh = None
            mesh_center_geom = np.zeros(3, dtype=float)
            try:
                mesh = trimesh.load(str(ext_collision_path), force="mesh")
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                if mesh.vertices.shape[0] > 0:
                    mesh_center_geom = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
                    centered_min = mesh.bounds[0] - mesh_center_geom
                    centered_max = mesh.bounds[1] - mesh_center_geom
                    self.object_origin_offsets[ext_id] = np.zeros(3, dtype=float)
                    self.object_bbox_bounds[ext_id] = (
                        np.array(centered_min, dtype=float),
                        np.array(centered_max, dtype=float),
                    )
            except Exception as e:
                raise RuntimeError(
                    f"External object '{ext_id}': failed to load collision mesh at {ext_collision_path}: {e}"
                ) from e

            bbox = self.object_bbox_bounds.get(ext_id)
            scaled_center = mesh_center_geom * ext_scale
            centering_pose = sapien.Pose(p=-scaled_center)
            scale_vec = [ext_scale] * 3 if ext_scale != 1.0 else None

            if ext_scale != 1.0 and bbox is not None:
                bbox = (bbox[0] * ext_scale, bbox[1] * ext_scale)
                self.object_bbox_bounds[ext_id] = bbox

            (
                init_pose_p,
                ext_quat,
                per_env_init_pose_p,
                per_env_init_quat,
            ) = self._compute_fixed_spawn_pose_bundle(
                ext_id,
                pl,
                bbox,
                label=f"object_placements.{ext_id}",
            )
            if per_env_init_pose_p is not None:
                self._initial_object_poses_per_env[ext_id] = per_env_init_pose_p
                self._initial_object_quats_per_env[ext_id] = per_env_init_quat

            z_extra = float(pl.get("z_extra", 0.0))
            if z_extra != 0.0:
                init_pose_p[2] += z_extra
                if per_env_init_pose_p is not None:
                    per_env_init_pose_p[:, 2] += z_extra
                    self._initial_object_poses_per_env[ext_id] = per_env_init_pose_p

            mat_cfg = self.object_material
            phys_material = sapien.physx.PhysxMaterial(
                static_friction=float(mat_cfg.get("static_friction", 0.5)),
                dynamic_friction=float(mat_cfg.get("dynamic_friction", 0.5)),
                restitution=float(mat_cfg.get("restitution", 0.0)),
            )

            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(
                str(ext_path), pose=centering_pose,
                **({"scale": scale_vec} if scale_vec else {}))

            collision_path = ext_collision_path
            if collision_path != ext_path:
                print(
                    f"[Collision] {ext_name} (ext): using explicit collision mesh "
                    f"{collision_path.name}"
                )

            collision_ok = False
            if ext_collision_mode == "nonconvex":
                try:
                    builder.add_nonconvex_collision_from_file(
                        str(collision_path),
                        material=phys_material,
                        pose=centering_pose,
                        **({"scale": scale_vec} if scale_vec else {}),
                    )
                    collision_ok = True
                    print(f"[Collision] {ext_name} (ext): nonconvex collision loaded")
                except Exception as e:
                    print(
                        f"{_Y}[Collision FALLBACK] {ext_name} (ext): nonconvex failed ({e}), "
                        f"falling back to single convex hull{_R}"
                    )
            if not collision_ok and ext_collision_mode == "coacd":
                try:
                    builder.add_multiple_convex_collisions_from_file(
                        str(collision_path), decomposition="coacd",
                        material=phys_material, pose=centering_pose,
                        **({"scale": scale_vec} if scale_vec else {}),
                    )
                    collision_ok = True
                    print(f"[Collision] {ext_name} (ext): COACD decomposition succeeded")
                except Exception as e:
                    print(
                        f"{_Y}[Collision FALLBACK] {ext_name} (ext): COACD failed ({e}), "
                        f"falling back to convex hull{_R}"
                    )
            if not collision_ok:
                builder.add_multiple_convex_collisions_from_file(
                    str(collision_path), decomposition="none",
                    material=phys_material, pose=centering_pose,
                    **({"scale": scale_vec} if scale_vec else {}),
                )
                print(
                    f"{_Y}[Collision FALLBACK] {ext_name} (ext): using single convex hull{_R}"
                )

            builder.set_initial_pose(sapien.Pose(p=init_pose_p, q=ext_quat))
            if ext_body_type == "kinematic":
                actor = builder.build_kinematic(name=f"object_{ext_name}")
            elif ext_body_type == "static":
                actor = builder.build_static(name=f"object_{ext_name}")
            else:
                actor = builder.build(name=f"object_{ext_name}")
            self.object_actors[ext_id] = actor
            self._initial_object_poses[ext_id] = init_pose_p
            self._initial_object_quats[ext_id] = ext_quat
            self._object_body_types[ext_id] = ext_body_type
            self._object_names[ext_id] = ext_name
            print(f"[Placement] {ext_name} (ext, id={ext_id}): "
                  f"actor_p=[{init_pose_p[0]:.4f},{init_pose_p[1]:.4f},{init_pose_p[2]:.4f}] "
                  f"(body_type={ext_body_type}, collision_mode={ext_collision_mode})")

    def _setup_lighting(self):
        """Setup scene lighting."""
        cfg = self.lighting_config or {}
        directional_lights = cfg.get(
            "directional_lights",
            [{"direction": [0, 0, -1], "color": [1, 1, 1], "shadow": True}],
        )
        point_lights = cfg.get(
            "point_lights",
            [
                {"position": [2, 2, 2], "color": [1, 1, 1], "shadow": False},
                {"position": [-2, -2, 2], "color": [0.8, 0.8, 0.8], "shadow": False},
            ],
        )

        for light in directional_lights:
            self.scene.add_directional_light(
                direction=light.get("direction", [0, 0, -1]),
                color=light.get("color", [1, 1, 1]),
                shadow=bool(light.get("shadow", True)),
            )
        for light in point_lights:
            self.scene.add_point_light(
                position=light.get("position", [2, 2, 2]),
                color=light.get("color", [1, 1, 1]),
                shadow=bool(light.get("shadow", False)),
            )
        
        # Set background/clear color to match viewer (light gray instead of black)
        # This ensures consistent background color in both viewer and rgb_array mode
        try:
            # Try to set clear color through render system if available
            if hasattr(self.scene, 'render_system') and self.scene.render_system is not None:
                render_system = self.scene.render_system
                # Try different methods to set background color
                if hasattr(render_system, 'set_clear_color'):
                    # Set clear color to light gray (matching typical viewer background)
                    render_system.set_clear_color(cfg.get("clear_color", [0.5, 0.5, 0.5, 1.0]))
                elif hasattr(render_system, 'set_background_color'):
                    render_system.set_background_color(cfg.get("clear_color", [0.5, 0.5, 0.5, 1.0]))
                # Also try setting ambient light to improve background visibility
                if hasattr(render_system, 'set_ambient_light'):
                    render_system.set_ambient_light(cfg.get("ambient_light", [0.3, 0.3, 0.3]))
        except Exception as e:
            # If setting background color fails, continue without it
            # The background mesh should still be visible
            pass

    def reset_object_poses(self):
        """Reset objects after settle_steps.

        Behaviour depends on placement_mode:
        - "fixed" / "random": no-op — keep physics-settled state.
        - "scene": soft-reset — keep settled orientation, restore XY to
          initial position, recalculate Z so the lowest rotated corner
          sits on the table.  With centered meshes, actor origin = mesh
          With centering_pose, actor origin = mesh center; bbox is symmetric.
        """
        if self.placement_mode in ("fixed", "random"):
            return

        for obj_id, actor in self.object_actors.items():
            if obj_id not in self._initial_object_poses:
                continue
            if self._object_body_types.get(obj_id) == "static":
                continue

            init_p = self._initial_object_poses[obj_id]
            init_q = self._initial_object_quats.get(obj_id, list(self.OBJECT_INIT_QUAT))

            actor.set_pose(Pose.create_from_pq(p=init_p, q=init_q))
            actor.set_linear_velocity(np.zeros(3))
            actor.set_angular_velocity(np.zeros(3))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode poses and robot configuration."""
        with torch.device(self.device):
            b = len(env_idx)

            # Background pose is set once at build time via builder.set_initial_pose()
            # in _load_background_actor and never changes between episodes.
            # Calling set_pose() here on a static actor fails in GPU sim (physx_cuda).

            # --- Random placement: recompute positions each episode ---
            if self.placement_mode == "random":
                self._randomize_episode_object_poses(env_idx, options)

            if self.use_360_background:
                self._randomize_360_spheres(env_idx, options)

            for obj_id, actor in self.object_actors.items():
                if obj_id in self._initial_object_poses:
                    if self._object_body_types.get(obj_id) == "static":
                        continue
                    if obj_id in self._initial_object_poses_per_env:
                        per_env_positions = torch.as_tensor(
                            self._initial_object_poses_per_env[obj_id],
                            device=self.device,
                            dtype=torch.float32,
                        )[env_idx]
                        per_env_quats = torch.as_tensor(
                            self._initial_object_quats_per_env[obj_id],
                            device=self.device,
                            dtype=torch.float32,
                        )[env_idx]
                        actor.set_pose(Pose.create_from_pq(p=per_env_positions, q=per_env_quats))
                        actor.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                        actor.set_angular_velocity(torch.zeros((b, 3), device=self.device))
                        continue
                    q = self._initial_object_quats.get(obj_id, list(self.OBJECT_INIT_QUAT))
                    actor.set_pose(
                        Pose.create_from_pq(
                            p=self._initial_object_poses[obj_id],
                            q=q,
                        )
                    )
                    actor.set_linear_velocity(np.zeros(3))
                    actor.set_angular_velocity(np.zeros(3))

            # Use robot_uids to select/reset robot-specific defaults (agent.uid may not be set yet).
            uid = (self.robot_uids or "panda").lower()
            if self.robot_init_qpos_custom is not None:
                qpos = np.array(self.robot_init_qpos_custom, dtype=np.float64)
            else:
                if "widowx" in uid or "bridgedataset" in uid:
                    # WidowX250S: 6 arm + 2 gripper (home, gripper open)
                    qpos = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.037, 0.037]
                    )
                elif "rc5" in uid or "aero_hand" in uid:
                    # RC5 + Aero Hand: 6 arm + 16 hand joints (neutral arm, hand open)
                    qpos = np.zeros(22, dtype=np.float64)
                else:
                    # Panda: 7 arm + 2 gripper
                    qpos = np.array(
                        [
                            0.0,
                            np.pi / 8,
                            0,
                            -np.pi * 5 / 8,
                            0,
                            np.pi * 3 / 4,
                            np.pi / 4,
                            0.04,
                            0.04,
                        ]
                    )
            if self.robot_init_qpos_noise > 0:
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
                if qpos.shape[-1] == 9:
                    qpos[:, -2:] = 0.04
                elif qpos.shape[-1] == 8:
                    qpos[:, -2:] = 0.037
                elif qpos.shape[-1] == 22 and ("rc5" in uid or "aero_hand" in uid):
                    qpos[:, 6:] = 0.0
            pose = self.robot_base_pose if self.robot_base_pose is not None else self.ROBOT_BASE_POSE
            self.agent.robot.set_pose(pose)
            self.agent.reset(qpos)
            self._sync_agent_controller_targets_to_current_state()

            # Physics settling: let objects fall to stable positions on the table.
            # Needed when placement_mode="scene" (positions from reconstruction may be
            # slightly off). For placement_mode="fixed" with auto_placement=True the
            # raycast already gives a correct Z, so settle_steps=0 is fine.
            if self.settle_steps > 0:
                print(f"[Settle] running {self.settle_steps} physics steps "
                      f"(placement_mode={self.placement_mode!r}) ...")

                # DEBUG: capture frames during settling and save as video.
                # Saves to SETTLE_DEBUG_VIDEO env var path, or /tmp/settle_debug.mp4 by default.
                import os as _os
                _settle_video_path = _os.environ.get(
                    "SETTLE_DEBUG_VIDEO", "/workspace/runs/openreal2sim/settle_debug.mp4"
                )
                _settle_frames = []
                # Capture every N-th step to keep the video manageable.
                _capture_every = max(1, self.settle_steps // 200)

                _first_error = None
                for _si in range(self.settle_steps):
                    self.scene.step()
                    if _si % _capture_every == 0:
                        try:
                            self.scene.update_render()
                            self.capture_sensor_data()
                            for _sname, _sensor in self.scene.sensors.items():
                                _obs = _sensor.get_obs(rgb=True, depth=False,
                                                       position=False, segmentation=False)
                                if "rgb" in _obs:
                                    _rgb = _obs["rgb"]
                                    if hasattr(_rgb, 'cpu'):
                                        _rgb = _rgb.cpu().numpy()
                                    if _rgb.ndim == 4:
                                        _rgb = _rgb[0]
                                    if _rgb.dtype != np.uint8:
                                        _rgb = (_rgb * 255).clip(0, 255).astype(np.uint8)
                                    _settle_frames.append(_rgb[:, :, :3])
                                    break
                        except Exception as _e:
                            if _first_error is None:
                                _first_error = str(_e)

                if _first_error is not None and not _settle_frames:
                    print(f"[Settle] frame capture FAILED: {_first_error}")

                if _settle_frames:
                    try:
                        import cv2 as _cv2
                        _h, _w = _settle_frames[0].shape[:2]
                        _os.makedirs(_os.path.dirname(_settle_video_path), exist_ok=True)
                        _writer = _cv2.VideoWriter(
                            _settle_video_path,
                            _cv2.VideoWriter_fourcc(*"mp4v"),
                            30,
                            (_w, _h),
                        )
                        for _f in _settle_frames:
                            _writer.write(_cv2.cvtColor(_f, _cv2.COLOR_RGB2BGR))
                        _writer.release()
                        print(f"[Settle] debug video saved: {_settle_video_path} "
                              f"({len(_settle_frames)} frames)")
                    except Exception as _e:
                        print(f"[Settle] failed to save debug video: {_e}")
                else:
                    print(f"[Settle] WARNING: 0 frames captured, no debug video")

                print(f"[Settle] done")
                # After settling (scene mode): soft-reset — keep settled orientation,
                # restore XY to initial so objects don't drift laterally.
                self.reset_object_poses()

                # Re-apply robot pose/qpos after settle using the same order as before settle.
                # Setting the root pose first keeps the articulated hand/gripper frame stable
                # in the reset frame and avoids settle-time drift in the debug video.
                self.agent.robot.set_pose(pose)
                self.agent.reset(qpos)
                self._sync_agent_controller_targets_to_current_state()

            self.consecutive_grasp = torch.zeros(b, dtype=torch.int32, device=self.device)
            if self.target_object_id is not None:
                self.episode_stats = dict(
                    is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    gripper_obj_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    gripper_target_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    obj_target_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                )
            else:
                self.episode_stats = dict(
                    is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    gripper_obj_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    obj_goal_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    obj_height_above_table=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    gripper_goal_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                )

    def _sync_agent_controller_targets_to_current_state(self):
        """Sync controller internal targets and PhysX drive targets to the current robot qpos.

        This is needed before settle_steps: otherwise the physics step may still use
        stale drive targets from the previous rollout/reset and pull the robot away
        from the fixed reset pose.
        """
        controller = getattr(self.agent, "controller", None)
        if controller is None:
            return

        def _sync_one(ctrl):
            subcontrollers = getattr(ctrl, "controllers", None)
            if isinstance(subcontrollers, dict):
                for sub in subcontrollers.values():
                    _sync_one(sub)
                return

            try:
                ctrl.reset()
            except Exception as exc:
                print(f"[Settle] controller.reset sync skipped for {type(ctrl).__name__}: {exc}")
                return

            if not hasattr(ctrl, "set_drive_targets"):
                return

            try:
                targets = ctrl.qpos.clone()
                if targets.ndim == 1:
                    targets = targets.unsqueeze(0)
                ctrl.set_drive_targets(targets)
            except Exception as exc:
                print(f"[Settle] drive target sync skipped for {type(ctrl).__name__}: {exc}")

        _sync_one(controller)

    def _get_manip_actor(self):
        """Return the manipulation target actor.

        Uses self.manip_object_id if set, otherwise the first actor in object_actors.
        Raises ValueError if object_actors is empty.
        """
        if not self.object_actors:
            raise ValueError("No objects loaded in the scene.")
        if self.manip_object_id is not None:
            if self.manip_object_id not in self.object_actors:
                raise ValueError(
                    f"manip_object_id '{self.manip_object_id}' not found in "
                    f"object_actors: {list(self.object_actors.keys())}"
                )
            return self.object_actors[self.manip_object_id]
        return next(iter(self.object_actors.values()))

    def _get_target_actor(self):
        """Return the target/support actor for put-on-target tasks."""
        if self.target_object_id is None:
            return None
        if self.target_object_id not in self.object_actors:
            raise ValueError(
                f"target_object_id '{self.target_object_id}' not found in "
                f"object_actors: {list(self.object_actors.keys())}"
            )
        return self.object_actors[self.target_object_id]

    def _get_actor_bbox_world(self, obj_id: str, actor):
        """Approximate world-frame bbox extents for a batched actor."""
        bbox = getattr(self, "object_bbox_bounds", {}).get(str(obj_id))
        if bbox is None:
            return None

        bbox_min = np.asarray(bbox[0], dtype=float)
        bbox_max = np.asarray(bbox[1], dtype=float)
        half = 0.5 * (bbox_max - bbox_min)
        corners = np.array(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ],
            dtype=float,
        )

        quat = actor.pose.q
        if hasattr(quat, "detach"):
            quat_np = quat.detach().cpu().numpy()
        else:
            quat_np = np.asarray(quat)
        if quat_np.ndim == 1:
            quat_np = quat_np[None, :]

        sizes = []
        for q in quat_np:
            rot = quat2mat(np.asarray(q, dtype=float))
            rotated = (rot @ corners.T).T
            size = rotated.max(axis=0) - rotated.min(axis=0)
            sizes.append(size)

        return torch.as_tensor(np.asarray(sizes), dtype=torch.float32, device=self.device)

    def get_language_instruction(self) -> list:
        """Return natural-language task instruction for all envs.

        ManiSkill3 adapter calls this once per batch and expects a list of
        strings with one instruction per environment.
        Priority: task_description (preset/config override) → scene.json task_desc → generic fallback.
        """
        instruction = instruction_for_manip_object(
            task_description=self.task_description,
            manip_object_id=self.manip_object_id,
            object_placements=self.object_placements,
            scene_task_desc=self.scene_config.task_desc,
        )
        return [instruction] * self.num_envs

    def evaluate(self) -> dict:
        """Evaluate OpenReal2Sim task in Bridge-style reward/info format."""
        if self.consecutive_grasp is None:
            self.consecutive_grasp = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )

        if not hasattr(self, "episode_stats") or self.episode_stats is None:
            b = self.num_envs
            if self.target_object_id is not None:
                self.episode_stats = dict(
                    is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    gripper_obj_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    gripper_target_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    obj_target_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                )
            else:
                self.episode_stats = dict(
                    is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),
                    gripper_obj_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    obj_goal_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    obj_height_above_table=torch.zeros((b,), dtype=torch.float32, device=self.device),
                    gripper_goal_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
                )

        source_obj = self._get_manip_actor()
        is_src_obj_grasped = self.agent.is_grasping(source_obj)

        self.consecutive_grasp += is_src_obj_grasped.int()
        self.consecutive_grasp[~is_src_obj_grasped] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        source_p = source_obj.pose.p
        gripper_p = self.agent.tcp.pose.p

        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consecutive_grasp
        )

        if self.target_object_id is not None:
            target_obj = self._get_target_actor()
            target_p = target_obj.pose.p

            src_bbox_world = self._get_actor_bbox_world(
                self.manip_object_id or next(iter(self.object_actors)), source_obj
            )
            tgt_bbox_world = self._get_actor_bbox_world(self.target_object_id, target_obj)

            offset = source_p - target_p
            if src_bbox_world is not None and tgt_bbox_world is not None:
                tgt_obj_half_length_bbox = tgt_bbox_world / 2
                src_obj_half_length_bbox = src_bbox_world / 2
                xy_flag = (
                    torch.linalg.norm(offset[:, :2], dim=1)
                    <= torch.linalg.norm(tgt_obj_half_length_bbox[:, :2], dim=1) + 0.01
                )
                z_flag = (offset[:, 2] > 0) & (
                    offset[:, 2]
                    - tgt_obj_half_length_bbox[:, 2]
                    - src_obj_half_length_bbox[:, 2]
                    <= 0.05
                )
            else:
                xy_flag = torch.linalg.norm(offset[:, :2], dim=1) <= 0.08
                z_flag = (offset[:, 2] > 0) & (offset[:, 2] <= 0.08)

            src_on_target = xy_flag & z_flag
            contact_forces = self.scene.get_pairwise_contact_forces(source_obj, target_obj)
            net_forces = torch.linalg.norm(contact_forces, dim=1)
            src_on_target = src_on_target & (net_forces > 0.03)
            success = src_on_target

            self.episode_stats["src_on_target"] = src_on_target
            self.episode_stats["gripper_obj_dist"] = torch.linalg.norm(gripper_p - source_p, dim=1)
            self.episode_stats["gripper_target_dist"] = torch.linalg.norm(gripper_p - target_p, dim=1)
            self.episode_stats["obj_target_dist"] = torch.linalg.norm(source_p - target_p, dim=1)

            return dict(**self.episode_stats, success=success)

        table_z = (
            self.auto_table_z + self.scene_z_offset
            if self.auto_table_z is not None
            else 0.0
        )
        obj_height_above_table = source_p[:, 2] - table_z

        lift_height = self.lift_height if self.lift_height is not None else 0.05
        src_on_target = obj_height_above_table > lift_height
        success = is_src_obj_grasped & src_on_target

        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["gripper_obj_dist"] = torch.linalg.norm(gripper_p - source_p, dim=1)
        self.episode_stats["obj_goal_dist"] = torch.clamp(lift_height - obj_height_above_table, min=0.0)

        goal_p = source_p.clone()
        goal_p[:, 2] = table_z + lift_height
        self.episode_stats["obj_height_above_table"] = obj_height_above_table
        self.episode_stats["gripper_goal_dist"] = torch.linalg.norm(gripper_p - goal_p, dim=1)

        return dict(**self.episode_stats, success=success)

    def _get_obs_extra(self, info: Dict) -> Dict:
        """Get additional task-specific observations."""
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)

        if "state" in self.obs_mode:
            for obj_id, actor in self.object_actors.items():
                obj_name = self._object_names[obj_id]
                obs[f"obj_{obj_name}_pose"] = actor.pose.raw_pose

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward (placeholder)."""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs, action, info)

    def _get_base_camera_pose_from_scene(self) -> sapien.Pose:
        """Compute base_camera pose from scene.json (OpenCV → SAPIEN/ROS).

        Returns the direct Pose quaternion to preserve the original camera
        orientation exactly (including roll), matching the reconstruction image.
        SAPIEN camera convention: (forward, right, up) = (X, -Y, Z).
        """
        cam_config = self.scene_config.camera
        z_offset = np.array(self.BACKGROUND_INIT_POS)[2] + self.scene_z_offset
        extrinsic_matrix = np.array(cam_config.extrinsic_matrix, dtype=np.float32)
        extrinsic_matrix[2, 3] += z_offset
        camera_pose = opencv_to_sapien_pose(extrinsic_matrix)
        q_numpy = np.asarray(camera_pose.q).flatten()
        if hasattr(q_numpy, 'cpu'):
            q_numpy = q_numpy.cpu().numpy()
        p_numpy = np.asarray(camera_pose.p).flatten()
        if hasattr(p_numpy, 'cpu'):
            p_numpy = p_numpy.cpu().numpy()
        camera_rot_mat = qvec2rotmat(q_numpy)
        forward = camera_rot_mat[:, 0]
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        print("[BASE_CAMERA] === env base_camera pose (direct ROS) ===")
        print(f"[BASE_CAMERA]   position: [{p_numpy[0]:.6f}, {p_numpy[1]:.6f}, {p_numpy[2]:.6f}]")
        print(f"[BASE_CAMERA]   forward (R[:,0]): [{forward[0]:.6f}, {forward[1]:.6f}, {forward[2]:.6f}]")
        return sapien.Pose(p=p_numpy, q=q_numpy)

    def _uses_visual_obs(self) -> bool:
        mode = str(getattr(self, "obs_mode", "") or getattr(self, "_obs_mode", "") or "")
        return any(token in mode for token in ("rgb", "rgbd", "depth", "segmentation", "sensor_data"))

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        """
        Configure cameras from cameras_config.
        base_camera: use_scene_json or explicit eye/target.
        custom_cameras with type=sensor: additional sensor cameras.
        """
        cam_config = self.scene_config.camera
        configs = []
        bc = self.cameras_config.get("base_camera", {})

        if bc.get("use_scene_json", True):
            pose = self._get_base_camera_pose_from_scene()
            width = bc.get("width", cam_config.width)
            height = bc.get("height", cam_config.height)
        else:
            eye = bc.get("eye", [0.8, 0.8, 0.6])
            target = bc.get("target", [0, 0, 0.2])
            pose = sapien_utils.look_at(eye=eye, target=target)
            width = bc.get("width", cam_config.width)
            height = bc.get("height", cam_config.height)

        # If width/height was overridden but intrinsic was not explicitly provided,
        # scale the intrinsic matrix proportionally so FOV is preserved.
        # (Without scaling, SAPIEN renders a narrower FOV at lower resolution,
        #  which looks like a zoom-in rather than a scaled-down full scene.)
        if "intrinsic" in bc:
            intrinsic = np.array(bc["intrinsic"], dtype=float)
        else:
            intrinsic = np.array(cam_config.intrinsic_matrix, dtype=float)
            orig_w, orig_h = cam_config.width, cam_config.height
            if int(width) != orig_w or int(height) != orig_h:
                scale_x = int(width) / orig_w
                scale_y = int(height) / orig_h
                intrinsic = intrinsic.copy()
                intrinsic[0] *= scale_x   # fx, cx scaled by width ratio
                intrinsic[1] *= scale_y   # fy, cy scaled by height ratio
                print(f"[Camera] base_camera: intrinsic auto-scaled "
                      f"{orig_w}x{orig_h} → {int(width)}x{int(height)} "
                      f"(scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")

        uses_visual_obs = self._uses_visual_obs()
        if uses_visual_obs:
            # PPO / OpenVLA consume 3rd_view_camera. Skip a same-pose base_camera so
            # parallel GPU envs are not paying for a duplicate RGB+segmentation buffer.
            third_w, third_h = THIRD_VIEW_WIDTH, THIRD_VIEW_HEIGHT
            third_intrinsic = np.array(cam_config.intrinsic_matrix, dtype=float)
            orig_w, orig_h = cam_config.width, cam_config.height
            if int(third_w) != orig_w or int(third_h) != orig_h:
                third_intrinsic = third_intrinsic.copy()
                third_intrinsic[0] *= int(third_w) / orig_w
                third_intrinsic[1] *= int(third_h) / orig_h
            configs.append(CameraConfig(
                uid=THIRD_VIEW_CAMERA_NAME,
                pose=pose,
                width=int(third_w),
                height=int(third_h),
                intrinsic=third_intrinsic,
            ))
            print(
                f"[Camera] {THIRD_VIEW_CAMERA_NAME}: {third_w}x{third_h} "
                "(OpenVLA / PPO third-person view; base_camera omitted)"
            )
        else:
            configs.append(CameraConfig(
                uid="base_camera",
                pose=pose,
                width=int(width),
                height=int(height),
                intrinsic=intrinsic,
            ))

        if getattr(self, "use_wrist_camera", False):
            mount_name, mount = _wrist_mount_link(self.agent)
            if mount is None:
                print(
                    "[Camera] wrist_camera requested but neither "
                    f"{WRIST_CAMERA_MOUNT_LINKS} was found on the robot"
                )
            else:
                print(f"[Camera] wrist_camera mounted on RealSense link '{mount_name}'")
                configs.append(CameraConfig(
                    uid=WRIST_CAMERA_NAME,
                    pose=sapien.Pose(p=WRIST_CAMERA_LOCAL_P, q=WRIST_CAMERA_LOCAL_Q),
                    width=WRIST_CAMERA_WIDTH,
                    height=WRIST_CAMERA_HEIGHT,
                    fov=WRIST_CAMERA_FOV,
                    near=WRIST_CAMERA_NEAR,
                    far=(
                        WRIST_CAMERA_FAR_PANO
                        if getattr(self, "use_360_background", False)
                        else WRIST_CAMERA_FAR
                    ),
                    mount=mount,
                ))

        for cust in self.cameras_config.get("custom_cameras", []):
            if cust.get("type", "sensor") != "sensor":
                continue
            uid = cust.get("uid", "custom_camera")
            eye = cust.get("eye", [0, 0, 1])
            target = cust.get("target", [0, 0, 0])
            width = cust.get("width", 640)
            height = cust.get("height", 480)
            fov = cust.get("fov", np.pi / 3)
            pose = sapien_utils.look_at(eye=eye, target=target)
            configs.append(CameraConfig(
                uid=uid,
                pose=pose,
                width=int(width),
                height=int(height),
                fov=float(fov),
            ))
        return configs

    def set_render_camera_pose(self, eye: np.ndarray, target: np.ndarray):
        """
        Set custom camera pose for video recording.
        
        Args:
            eye: Camera position [x, y, z]
            target: Camera target point [x, y, z]
        """
        self.custom_render_camera_pose = (np.array(eye), np.array(target))

    def get_viewer_camera_pose(self, viewer=None):
        """
        Get current camera pose from viewer (if available).
        
        Strategy: Use fps_camera_controller for eye position (actual camera position),
        and arc_camera_controller.center for target (what user is looking at).
        This combination gives the most reliable result.
        
        Args:
            viewer: Optional viewer object. If None, will call self.render() to get it.
        
        Returns:
            tuple: (eye, target) if viewer camera is available, None otherwise
        """
        try:
            # Get viewer if not provided
            if viewer is None:
                viewer = self.render()
            if viewer is None:
                return None
            
            if not hasattr(viewer, 'control_window'):
                return None
            
            control_window = viewer.control_window
            if control_window is None:
                return None
            
            eye = None
            target = None
            
            # Get eye position from fps_camera_controller (actual camera position)
            try:
                if hasattr(control_window, 'fps_camera_controller'):
                    fps_ctrl = control_window.fps_camera_controller
                    if fps_ctrl is not None:
                        # Get eye position from xyz or pose
                        if hasattr(fps_ctrl, 'xyz'):
                            eye = np.array(fps_ctrl.xyz)
                        elif hasattr(fps_ctrl, 'pose') and hasattr(fps_ctrl.pose, 'p'):
                            eye = np.array(fps_ctrl.pose.p) if isinstance(fps_ctrl.pose.p, (list, tuple)) else fps_ctrl.pose.p
                            if hasattr(eye, 'cpu'):
                                eye = eye.cpu().numpy()
                        
                        if eye is not None:
                            eye = np.array(eye).flatten()[:3]
            except Exception as e:
                print(f"[WARN] Failed to get eye from fps_camera_controller: {e}")
            
            # Get target from arc_camera_controller.center (what user is looking at)
            try:
                if hasattr(control_window, 'arc_camera_controller'):
                    arc_ctrl = control_window.arc_camera_controller
                    if arc_ctrl is not None and hasattr(arc_ctrl, 'center'):
                        center = np.array(arc_ctrl.center)
                        target = center.flatten()[:3]
            except Exception as e:
                print(f"[WARN] Failed to get target from arc_camera_controller: {e}")
            
            # If we have both eye and target, return them
            if eye is not None and target is not None:
                return (eye, target)
            
            # Fallback: if we only have eye, try to compute target from forward direction
            if eye is not None:
                try:
                    if hasattr(control_window, 'fps_camera_controller'):
                        fps_ctrl = control_window.fps_camera_controller
                        if fps_ctrl is not None:
                            # Try to get forward direction
                            forward = None
                            
                            if hasattr(fps_ctrl, 'forward'):
                                forward = np.array(fps_ctrl.forward)
                                if hasattr(forward, 'cpu'):
                                    forward = forward.cpu().numpy()
                                forward = forward.flatten()[:3]
                            
                            if forward is None and hasattr(fps_ctrl, 'pose') and hasattr(fps_ctrl.pose, 'q'):
                                q = fps_ctrl.pose.q
                                if hasattr(q, 'cpu'):
                                    q = q.cpu().numpy()
                                if isinstance(q, (list, tuple)):
                                    q = np.array(q)
                                if len(q) == 4:
                                    q_wxyz = np.array([q[3], q[0], q[1], q[2]])
                                    R = quat2mat(q_wxyz)
                                    forward = -R[:, 2]  # Camera looks along -Z
                            
                            if forward is not None:
                                forward = forward / (np.linalg.norm(forward) + 1e-8)
                                # Look at a reasonable distance (use scene center as hint if available)
                                look_distance = 2.0
                                target = eye + forward * look_distance
                                return (eye, target)
                except Exception as e:
                    print(f"[WARN] Failed to compute target from forward: {e}")
            
        except Exception as e:
            print(f"[WARN] Could not get camera pose from viewer: {e}")
            import traceback
            traceback.print_exc()
        return None

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        """Configure camera for human viewing/recording from cameras_config."""
        if self.custom_render_camera_pose is not None:
            eye, target = self.custom_render_camera_pose
            pose = sapien_utils.look_at(eye=eye, target=target)
        else:
            rc = self.cameras_config.get("render_camera", {})
            if rc.get("use_base_camera"):
                eye, target = get_base_camera_eye_target(self.scene_json_path)
                pose = sapien_utils.look_at(eye=eye, target=target)
            else:
                eye = rc.get("eye", [0.8, 0.8, 0.6])
                target = rc.get("target", [0, 0, 0.2])
                pose = sapien_utils.look_at(eye=eye, target=target)
        width = self.cameras_config.get("render_camera", {}).get("width", self.render_width)
        height = self.cameras_config.get("render_camera", {}).get("height", self.render_height)
        fov = self.cameras_config.get("render_camera", {}).get("fov", np.pi / 3)
        return CameraConfig(
            uid="render_camera",
            pose=pose,
            width=int(width),
            height=int(height),
            fov=float(fov),
            near=0.01,
            far=100,
        )
_Y = "\033[33m"
_R = "\033[0m"
