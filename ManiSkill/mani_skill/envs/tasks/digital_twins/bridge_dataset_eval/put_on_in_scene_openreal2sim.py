from pathlib import Path

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, qmult, quat2mat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.rc5.rc5 import RC5AeroHandRight
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    WidowX250SBridgeDatasetFlatTable,
)
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene_multi import (
    CARROT_DATASET_DIR,
    PutOnPlateInScene25MainV3,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from real2sim.rc5_calibrated_presets import (
    RC5_HAND_GRAB_QPOS,
    RC5_HAND_OPEN_QPOS,
    RC5_HOME_BASE_POSE_P,
    RC5_HOME_BASE_POSE_Q,
    RC5_HOME_INIT_QPOS,
)


REPO_ROOT = Path(__file__).resolve().parents[6]
OPENREAL2SIM_ASSET_DIR = REPO_ROOT / "real2sim" / "assets" / "openreal2sim"
BACKGROUND_VISUAL_PATH = OPENREAL2SIM_ASSET_DIR / "background_registered.glb"

CALIBRATED_SPAWN_GRID_CENTER_XY = np.array(
    [-0.005751613080501555, -2.0679838466644287], dtype=np.float32
)
CALIBRATED_SPAWN_GRID_SIZE_XY = np.array(
    [0.33000001311302185, 0.20000003278255463], dtype=np.float32
)
CALIBRATED_SPAWN_GRID_Z = 0.12047362327575684
SOURCE_SPAWN_EXTRA_Z = 0.08

# Use a smaller training rectangle so the full 6x6 lattice stays inside the
# reachable region of the calibrated WidowX setup.
TRAIN_SPAWN_GRID_CENTER_XY = np.array([-0.06, -2.095], dtype=np.float32)
TRAIN_SPAWN_GRID_SIZE_XY = np.array([0.28, 0.13], dtype=np.float32)

PROBE_CARROT_POSE_Q = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float32)
PROBE_PLATE_POSE_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Keep PPO training aligned with the currently calibrated RC5 debug setup.
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
    CALIBRATED_SPAWN_GRID_Z + 1.0,
]

MANUAL_BEST_EYE = np.array(
    [-0.021688668057322502, -2.4112067222595215, 0.34091514348983765],
    dtype=np.float32,
)
MANUAL_BEST_TARGET = np.array(
    [0.03691490739583969, -0.19591856002807617, -0.6688752770423889],
    dtype=np.float32,
)
MANUAL_BEST_ROLL_DEG = 1.65
DEFAULT_CAMERA_FOV = 1.0
DEFAULT_CAMERA_NEAR = 0.01
DEFAULT_CAMERA_FAR = 10.0
ROBOT_BLACK_RGBA = sapien_utils.hex2rgba("#2b2b2b")
WRIST_CAMERA_NAME = "wrist_camera"
WRIST_CAMERA_WIDTH = 168
WRIST_CAMERA_HEIGHT = 224
WRIST_CAMERA_FOV = 1.5000000000000002
WRIST_CAMERA_NEAR = 0.01
WRIST_CAMERA_FAR = 2.0
WRIST_CAMERA_MOUNT_LINK = "prehand_cam"
WRIST_CAMERA_LOCAL_P = [0.0, 0.06750000000000002, 0.060600000000000015]
WRIST_CAMERA_LOCAL_Q = [
    0.7071067811865476,
    -0.0,
    0.7071067811865475,
    0.0,
]
DEFAULT_USE_WRIST_CAMERA = True

def _pose_from_eye_target_roll(eye: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> sapien.Pose:
    base_pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
    if abs(roll_deg) < 1e-8:
        return base_pose
    rot = quat2mat(base_pose.sp.q)
    roll_local = axangle2mat([1.0, 0.0, 0.0], np.deg2rad(roll_deg))
    return sapien.Pose(p=np.array(base_pose.sp.p), q=mat2quat(rot @ roll_local))


def _plane_collision_pose(z: float) -> sapien.Pose:
    return sapien.Pose(p=[0.0, 0.0, z], q=[0.7071068, 0.0, -0.7071068, 0.0])


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


@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SBridgeDatasetOpenReal2Sim(WidowX250SBridgeDatasetFlatTable):
    uid = "widowx250s_bridgedataset_openreal2sim"

    @property
    def _sensor_configs(self):
        # The task injects a fixed world camera via env-level sensor configs.
        return []


@register_agent()
class RC5AeroHandRightOpenReal2Sim(RC5AeroHandRight):
    uid = "rc5_aero_hand_right_openreal2sim"
    gripper_open_qpos = RC5_HAND_OPEN_QPOS.copy()
    gripper_closed_qpos = RC5_HAND_GRAB_QPOS.copy()

    @property
    def _sensor_configs(self):
        # The task injects a fixed world camera via env-level sensor configs.
        return []


@register_env("PutOnPlateInScene25OpenReal2Sim-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25OpenReal2Sim(PutOnPlateInScene25MainV3):
    def __init__(self, **kwargs):
        self._prep_init()
        self._generate_init_pose()
        self.use_wrist_camera = bool(kwargs.pop("use_wrist_camera", DEFAULT_USE_WRIST_CAMERA))
        self.initial_qpos = np.array(ROBOT_INIT_QPOS, dtype=np.float32)
        self.initial_robot_pos = sapien.Pose(p=ROBOT_BASE_POSE_P, q=ROBOT_BASE_POSE_Q)
        self.safe_robot_pos = sapien.Pose(p=SAFE_ROBOT_BASE_POSE_P, q=ROBOT_BASE_POSE_Q)
        self.extra_stats = dict()
        BaseEnv.__init__(self, robot_uids=RC5AeroHandRightOpenReal2Sim, **kwargs)

    def _prep_init(self):
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        if len(self.model_db_carrot) != 25:
            raise ValueError("Expected 25 carrot models for the Scene25 task.")

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}
        if len(self.model_db_plate) != 1:
            raise ValueError("Expected a single plate model for the Scene25 task.")

        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # The real2sim scene provides a fixed background directly in geometry,
        # so the RGB overlay path used by the original benchmark is disabled.
        self.overlay_images_numpy = []
        self.overlay_textures_numpy = []
        self.overlay_mix_numpy = []

    @property
    def _default_sensor_configs(self):
        pose = _pose_from_eye_target_roll(
            eye=MANUAL_BEST_EYE,
            target=MANUAL_BEST_TARGET,
            roll_deg=MANUAL_BEST_ROLL_DEG,
        )
        configs = [
            CameraConfig(
                uid="3rd_view_camera",
                pose=pose,
                width=640,
                height=480,
                fov=DEFAULT_CAMERA_FOV,
                near=DEFAULT_CAMERA_NEAR,
                far=DEFAULT_CAMERA_FAR,
            )
        ]
        if self.use_wrist_camera:
            wrist_cfg = _rc5_wrist_camera_config(self.agent)
            if wrist_cfg is not None:
                configs.append(wrist_cfg)
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = _pose_from_eye_target_roll(
            eye=MANUAL_BEST_EYE,
            target=MANUAL_BEST_TARGET,
            roll_deg=MANUAL_BEST_ROLL_DEG,
        )
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=640,
                height=480,
                fov=DEFAULT_CAMERA_FOV,
                near=DEFAULT_CAMERA_NEAR,
                far=DEFAULT_CAMERA_FAR,
            )
        ]

    def _generate_init_pose(self):
        x_min = TRAIN_SPAWN_GRID_CENTER_XY[0] - TRAIN_SPAWN_GRID_SIZE_XY[0] / 2.0
        x_max = TRAIN_SPAWN_GRID_CENTER_XY[0] + TRAIN_SPAWN_GRID_SIZE_XY[0] / 2.0
        y_min = TRAIN_SPAWN_GRID_CENTER_XY[1] - TRAIN_SPAWN_GRID_SIZE_XY[1] / 2.0
        y_max = TRAIN_SPAWN_GRID_CENTER_XY[1] + TRAIN_SPAWN_GRID_SIZE_XY[1] / 2.0

        grid_x = np.linspace(x_min, x_max, 6, dtype=np.float32)
        grid_y = np.linspace(y_min, y_max, 6, dtype=np.float32)
        grid_pos = np.array([[x, y] for x in grid_x for y in grid_y], dtype=np.float32)

        plate_rest_z = CALIBRATED_SPAWN_GRID_Z + self._plate_resting_center_z_offset(PROBE_PLATE_POSE_Q)
        carrot_spawn_z = CALIBRATED_SPAWN_GRID_Z + self._max_carrot_resting_center_z_offset() + SOURCE_SPAWN_EXTRA_Z

        xyz_configs = []
        for carrot_xy in grid_pos:
            for plate_xy in grid_pos:
                if np.allclose(carrot_xy, plate_xy):
                    continue
                if np.linalg.norm(plate_xy - carrot_xy) <= 0.10:
                    continue
                xyz_configs.append(
                    np.array(
                        [
                            [carrot_xy[0], carrot_xy[1], carrot_spawn_z],
                            [plate_xy[0], plate_xy[1], plate_rest_z],
                        ],
                        dtype=np.float32,
                    )
                )
        self.xyz_configs = np.stack(xyz_configs)

        quat_configs = []
        for yaw in [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]:
            yaw_quat = euler2quat(0.0, 0.0, yaw)
            carrot_quat = qmult(yaw_quat, PROBE_CARROT_POSE_Q)
            quat_configs.append(
                np.array([carrot_quat, PROBE_PLATE_POSE_Q], dtype=np.float32)
            )
        self.quat_configs = np.stack(quat_configs)

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(self, options, sapien.Pose(p=ROBOT_BASE_POSE_P, q=ROBOT_BASE_POSE_Q))

    def _load_scene(self, options: dict):
        use_default_task = options.get("use_default_task", False)
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i],
                color=ROBOT_BLACK_RGBA,
                specular=0.05,
                roughness=0.85,
            )

        ground_builder = self.scene.create_actor_builder()
        ground_builder.add_plane_collision(_plane_collision_pose(CALIBRATED_SPAWN_GRID_Z))
        ground_builder.initial_pose = sapien.Pose()
        ground_builder.build_static(name="ground")

        background_builder = self.scene.create_actor_builder()
        background_builder.add_visual_from_file(str(BACKGROUND_VISUAL_PATH))
        background_builder.initial_pose = sapien.Pose()
        background_builder.build_static(name="arena")

        self.model_bbox_sizes = {}
        self.objs_carrot = {}
        for idx, name in enumerate(self.model_db_carrot):
            model_path = CARROT_DATASET_DIR / "more_carrot" / name
            density = self.model_db_carrot[name].get("density", 1000)
            scale_list = self.model_db_carrot[name].get("scale", [1.0])
            bbox = self.model_db_carrot[name]["bbox"]
            scale = scale_list[0] if use_default_task else self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([1.0, 0.3 * idx, 1.0]))
            self.objs_carrot[name] = self._build_actor_helper(name, model_path, density, scale, pose)
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)

        self.objs_plate = {}
        for idx, name in enumerate(self.model_db_plate):
            model_path = CARROT_DATASET_DIR / "more_plate" / name
            density = self.model_db_plate[name].get("density", 1000)
            scale_list = self.model_db_plate[name].get("scale", [1.0])
            bbox = self.model_db_plate[name]["bbox"]
            scale = scale_list[0] if use_default_task else self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([2.0, 0.3 * idx, 1.0]))
            self.objs_plate[name] = self._build_actor_helper(name, model_path, density, scale, pose)
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)

    def _build_actor_helper(self, name: str, path: Path, density: float, scale: float, pose: Pose):
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=str(path / "collision.obj"),
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = path / "textured.obj"
        if not visual_file.exists():
            visual_file = path / "textured.dae"
            if not visual_file.exists():
                visual_file = path / "textured.glb"
        builder.add_visual_from_file(filename=str(visual_file), scale=[scale] * 3)
        builder.initial_pose = pose
        return builder.build(name=name)

    def _bbox_corners(self, model_info: dict) -> np.ndarray:
        bbox = model_info["bbox"]
        bbox_min = np.array(bbox["min"], dtype=np.float32)
        bbox_max = np.array(bbox["max"], dtype=np.float32)
        xs = [bbox_min[0], bbox_max[0]]
        ys = [bbox_min[1], bbox_max[1]]
        zs = [bbox_min[2], bbox_max[2]]
        return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)

    def _resting_center_z_offset_from_bbox(self, model_info: dict, quat_wxyz) -> float:
        corners = self._bbox_corners(model_info)
        rot = quat2mat(np.asarray(quat_wxyz, dtype=np.float64))
        rotated = corners @ rot.T
        return float(-rotated[:, 2].min())

    def _plate_resting_center_z_offset(self, quat_wxyz) -> float:
        plate_name = self.plate_names[0]
        return self._resting_center_z_offset_from_bbox(self.model_db_plate[plate_name], quat_wxyz)

    def _max_carrot_resting_center_z_offset(self) -> float:
        offsets = []
        for yaw in [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]:
            yaw_quat = euler2quat(0.0, 0.0, yaw)
            carrot_quat = qmult(yaw_quat, PROBE_CARROT_POSE_Q)
            for carrot_name in self.carrot_names:
                offsets.append(
                    self._resting_center_z_offset_from_bbox(
                        self.model_db_carrot[carrot_name], carrot_quat
                    )
                )
        return float(max(offsets))
