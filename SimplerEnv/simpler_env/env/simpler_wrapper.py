import gymnasium as gym
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from mani_skill.envs.sapien_env import BaseEnv
from real2sim.openreal2sim_validation import (
    AIRI_CUBES_ROBOT_BASE_POSE_P,
    AIRI_CUBES_ROBOT_BASE_POSE_Q,
    AIRI_CUBES_ROBOT_INIT_QPOS,
    AIRI_CUBES_V3_ROBOT_BASE_POSE_P,
    AIRI_CUBES_V3_ROBOT_BASE_POSE_Q,
    AIRI_CUBES_V3_ROBOT_INIT_QPOS,
    DEFAULT_REAL2SIM_CONTROL_FREQ,
    DEFAULT_REAL2SIM_SIM_FREQ,
    WRIST_CAMERA_NAME,
)


WRIST_INSET_TOP = 4
WRIST_INSET_LEFT = 4
WRIST_INSET_MARGIN = 4
WRIST_INSET_BORDER = 4
WRIST_INSET_HEIGHT = 224
WRIST_INSET_WIDTH = 168
REAL2SIM_RECORDER_ENV_IDS = {
    "PutObjectOnPlateAiriTableRecorder-v1",
    "PutObjectOnPlateAiriCubesRecorder-v1",
    "PickUpAiriCubeRecorder-v1",
    "PutObjectOnPlateAiriCubesV3Recorder-v1",
    "PickUpAiriCubeV3Recorder-v1",
}
AIRI_CUBES_ENV_IDS = {
    "PutObjectOnPlateAiriCubesRecorder-v1",
    "PickUpAiriCubeRecorder-v1",
    "PutObjectOnPlateAiriCubesV3-v1",
    "PutObjectOnPlateAiriCubesV3Recorder-v1",
    "PickUpAiriCubeV3-v1",
    "PickUpAiriCubeV3Recorder-v1",
}
AIRI_CUBES_V3_ENV_IDS = {
    "PutObjectOnPlateAiriCubesV3-v1",
    "PutObjectOnPlateAiriCubesV3Recorder-v1",
    "PickUpAiriCubeV3-v1",
    "PickUpAiriCubeV3Recorder-v1",
}


def _unnormalize_continuous_action(raw_actions: torch.Tensor, unnorm_state, action_scale: float = 1.0) -> torch.Tensor:
    normalized_actions = raw_actions.to(torch.float32)

    action_high = torch.as_tensor(unnorm_state["q99"], device=normalized_actions.device, dtype=torch.float32).reshape(1, -1)
    action_low = torch.as_tensor(unnorm_state["q01"], device=normalized_actions.device, dtype=torch.float32).reshape(1, -1)
    mask = unnorm_state.get("mask", np.ones_like(unnorm_state["q01"], dtype=bool))
    mask = torch.as_tensor(mask, device=normalized_actions.device, dtype=torch.bool).reshape(1, -1)

    raw_action = torch.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    world_vector = raw_action[:, :3] * action_scale
    rot_axangle = raw_action[:, 3:6]
    gripper = 2.0 * (raw_action[:, 6:7] > 0.5).to(torch.float32) - 1.0
    return torch.cat([world_vector, rot_axangle, gripper], dim=1)


def _to_debug_value(value, digits: int = 4):
    if isinstance(value, dict):
        return {str(k): _to_debug_value(v, digits=digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_debug_value(v, digits=digits) for v in value]
    if isinstance(value, np.ndarray):
        return _to_debug_value(value.tolist(), digits=digits)
    if isinstance(value, np.generic):
        return _to_debug_value(value.item(), digits=digits)
    if torch.is_tensor(value):
        return _to_debug_value(value.detach().cpu().tolist(), digits=digits)
    if isinstance(value, float):
        return round(float(value), digits)
    return value


def _tensor_row(value) -> np.ndarray:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr[0]


def _tcp_xyz(env) -> np.ndarray | None:
    try:
        return _tensor_row(env.unwrapped.agent.tcp.pose.p).astype(np.float64)
    except Exception:
        return None


def _joint_debug(env) -> dict:
    try:
        robot = env.unwrapped.agent.robot
        qpos = _tensor_row(robot.get_qpos()).astype(np.float64)
        qvel = _tensor_row(robot.get_qvel()).astype(np.float64)
        payload = {"qpos": qpos, "qvel": qvel}
        try:
            qlimits = _tensor_row(robot.get_qlimits()).astype(np.float64)[: len(qpos)]
            limit_margin = np.minimum(qpos - qlimits[:, 0], qlimits[:, 1] - qpos)
            closest_idx = int(np.argmin(limit_margin))
            payload.update(
                {
                    "limit_margin": limit_margin,
                    "closest_limit_joint": closest_idx,
                    "closest_limit_margin": float(limit_margin[closest_idx]),
                }
            )
        except Exception:
            pass
        return payload
    except Exception as exc:
        return {"error": str(exc)}


def _debug_joint_names(env) -> list[str]:
    try:
        return [str(joint.name) for joint in env.unwrapped.agent.robot.get_active_joints()]
    except Exception:
        return []


def _info_for_debug(info: dict, env_index: int = 0) -> dict:
    payload = {}
    for key, value in info.items():
        if key == "episode":
            continue
        try:
            if torch.is_tensor(value):
                payload[key] = value.reshape(value.shape[0], -1)[env_index].detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                payload[key] = np.asarray(value).reshape(value.shape[0], -1)[env_index].tolist()
            else:
                payload[key] = value
        except Exception:
            payload[key] = str(value)
    return payload


def _compose_wrist_inset(
    scene_rgb: torch.Tensor,
    wrist_rgb: torch.Tensor | None,
    *,
    bottom_right: bool = False,
) -> torch.Tensor:
    if wrist_rgb is None:
        return scene_rgb.to(torch.uint8)

    scene_rgb = scene_rgb.to(torch.uint8)
    wrist_rgb = wrist_rgb.to(torch.uint8)
    if wrist_rgb.shape[1] != WRIST_INSET_HEIGHT or wrist_rgb.shape[2] != WRIST_INSET_WIDTH:
        wrist_rgb = F.interpolate(
            wrist_rgb.permute(0, 3, 1, 2).to(torch.float32),
            size=(WRIST_INSET_HEIGHT, WRIST_INSET_WIDTH),
            mode="bilinear",
            align_corners=False,
        ).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

    border = WRIST_INSET_BORDER
    inset_h = WRIST_INSET_HEIGHT
    inset_w = WRIST_INSET_WIDTH
    if bottom_right:
        margin = WRIST_INSET_MARGIN
        top = scene_rgb.shape[1] - inset_h - 2 * border - margin
        left = scene_rgb.shape[2] - inset_w - 2 * border - margin
    else:
        top = WRIST_INSET_TOP
        left = WRIST_INSET_LEFT
    out = scene_rgb.clone()
    out[:, top:top + inset_h + 2 * border, left:left + inset_w + 2 * border, :] = 0
    out[:, top + border:top + border + inset_h, left + border:left + border + inset_w, :] = wrist_rgb
    return out


def _openvla_obs_image(obs: dict, *, wrist_inset_bottom_right: bool = False) -> torch.Tensor:
    sensor_data = obs["sensor_data"]
    scene_rgb = sensor_data["3rd_view_camera"]["rgb"]
    wrist_data = sensor_data.get(WRIST_CAMERA_NAME)
    wrist_rgb = None if wrist_data is None else wrist_data["rgb"]
    return _compose_wrist_inset(scene_rgb, wrist_rgb, bottom_right=wrist_inset_bottom_right)


class SimlerWrapper:
    def __init__(self, all_args, unnorm_state, extra_seed=0):
        self.args = all_args
        self.unnorm_state = unnorm_state
        self._real2sim_robot_state = None

        self.num_envs = self.args.num_envs
        robot_control_mode = "arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos"
        self._wrist_inset_bottom_right = bool(self.args.use_wrist_camera) and self.args.env_id in AIRI_CUBES_ENV_IDS
        self._eval_debug_file = None
        self._eval_debug_step = 0

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode="rgb+segmentation",
            control_mode=robot_control_mode,
            sim_backend="gpu",
            enable_shadow=True,
            sim_config={
                "sim_freq": DEFAULT_REAL2SIM_SIM_FREQ,
                "control_freq": DEFAULT_REAL2SIM_CONTROL_FREQ,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
            use_wrist_camera=bool(self.args.use_wrist_camera),
        )
        self.env: BaseEnv = gym.make(**env_config)
        self.env.reset(seed=[self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)])
        self._reset_counter = 0

        # variables
        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32)  # [B, 1]

        # constants
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self._setup_eval_debug()

    def _setup_eval_debug(self):
        debug_path = str(getattr(self.args, "eval_debug_jsonl", "") or "").strip()
        if not debug_path:
            return
        path = Path(debug_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._eval_debug_file = path.open("w", encoding="utf-8")
        self._write_eval_debug({"t": "meta", "env_id": self.args.env_id, "num_envs": int(self.num_envs), "joint_names": _debug_joint_names(self.env)})

    def _write_eval_debug(self, payload: dict):
        if self._eval_debug_file is None:
            return
        self._eval_debug_file.write(json.dumps(_to_debug_value(payload), separators=(",", ":")) + "\n")
        self._eval_debug_file.flush()

    def _real2sim_reset_options(self) -> dict:
        if self.args.env_id not in REAL2SIM_RECORDER_ENV_IDS:
            return {}
        if self.args.env_id in AIRI_CUBES_V3_ENV_IDS:
            robot_base_pose_p = AIRI_CUBES_V3_ROBOT_BASE_POSE_P
            robot_base_pose_q = AIRI_CUBES_V3_ROBOT_BASE_POSE_Q
            robot_init_qpos = AIRI_CUBES_V3_ROBOT_INIT_QPOS
        elif self.args.env_id in {"PutObjectOnPlateAiriCubesRecorder-v1", "PickUpAiriCubeRecorder-v1"}:
            robot_base_pose_p = AIRI_CUBES_ROBOT_BASE_POSE_P
            robot_base_pose_q = AIRI_CUBES_ROBOT_BASE_POSE_Q
            robot_init_qpos = AIRI_CUBES_ROBOT_INIT_QPOS
        else:
            if self._real2sim_robot_state is None:
                from real2sim.calibrate_rc5_pose import _load_state_file

                self._real2sim_robot_state = _load_state_file("")
            robot_base_pose_p, robot_base_pose_q, robot_init_qpos = self._real2sim_robot_state
        return {
            "load_background": True,
            "show_debug_markers": False,
            "robot_far_away": False,
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
            "trajectory_instruction": str(getattr(self.args, "real2sim_instruction_template", "")).strip(),
        }

    def render_frame(self, camera_name: str = "", obs_img: torch.Tensor | None = None) -> np.ndarray:
        if camera_name:
            frames = self.env.unwrapped.render_rgb_array(camera_name=camera_name)
            if torch.is_tensor(frames):
                frames = frames.detach().cpu().numpy()
            else:
                frames = np.asarray(frames)
            return frames.astype(np.uint8, copy=False)
        if obs_img is None:
            raise ValueError("obs_img is required when camera_name is empty.")
        return obs_img.detach().cpu().numpy()

    def get_reward(self, info):
        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(info["success"].device)  # [B, 1]

        reward += info["is_src_obj_grasped"].reshape(-1, 1) * 0.1
        reward += info["consecutive_grasp"].reshape(-1, 1) * 0.1
        reward += info["success"].reshape(-1, 1) * 1.0

        # diff
        reward_diff = reward - self.reward_old
        self.reward_old = reward

        return reward_diff

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        action_scale = 1.0

        # Extract predicted action tokens and translate into (normalized) continuous actions
        pact_token = raw_actions.cpu().numpy()  # [B, dim]
        dact = 32000 - pact_token  # [B, dim]
        dact = np.clip(dact - 1, a_min=0, a_max=254)  # [B, dim]
        normalized_actions = np.asarray([self.bin_centers[da] for da in dact])  # [B, dim]

        # Unnormalize actions
        action_norm_stats = self.unnorm_state
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))  # [dim]
        mask = np.asarray(mask).reshape(1, -1)  # [1, dim]
        action_high = np.array(action_norm_stats["q99"]).reshape(1, -1)  # [1, dim]
        action_low = np.array(action_norm_stats["q01"]).reshape(1, -1)  # [1, dim]
        raw_action_np = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        raw_action = {
            "world_vector": raw_action_np[:, :3],
            "rotation_delta": raw_action_np[:, 3:6],
            "open_gripper": raw_action_np[:, 6:7],  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * action_scale  # [B, 3]
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0  # [B, 1]

        # origin euler
        action["rot_axangle"] = raw_action["rotation_delta"]

        action = {k: torch.tensor(v) for k, v in action.items()}  # to float32 ?

        action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

        # to tpdv
        action = action.to(raw_actions.device)

        return action

    def reset(self, obj_set: str, same_init: bool = False):
        options = self._real2sim_reset_options()
        options["obj_set"] = obj_set
        if self.args.env_id in REAL2SIM_RECORDER_ENV_IDS:
            base_episode_id = self.args.seed * 1_000_000 + self._reset_counter * self.num_envs
            options["episode_id"] = (
                torch.arange(self.num_envs, device=self.env.device, dtype=torch.int64) + base_episode_id
            )
            self._reset_counter += 1
        if same_init:
            options["episode_id"] = torch.randint(1000000000, (1,)).expand(self.num_envs).to(self.env.device)  # [B]
        if self.args.use_default_task:
            options["use_default_task"] = True

        obs, info = self.env.reset(options=options)
        obs_image = _openvla_obs_image(obs, wrist_inset_bottom_right=self._wrist_inset_bottom_right)
        instruction = self.env.unwrapped.get_language_instruction()

        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(obs_image.device)  # [B, 1]
        self._eval_debug_step = 0
        self._write_eval_debug(
            {
                "t": "reset",
                "obj_set": obj_set,
                "instruction": instruction,
                "tcp": _tcp_xyz(self.env),
                "q": _joint_debug(self.env).get("qpos"),
                "info": _info_for_debug(info),
            }
        )

        return obs_image, instruction, info

    def step(self, raw_action):
        tcp0 = _tcp_xyz(self.env)
        joints0 = _joint_debug(self.env)
        action = self._process_action(raw_action)

        obs, _reward, _terminated, truncated, info = self.env.step(action)
        tcp1 = _tcp_xyz(self.env)
        joints1 = _joint_debug(self.env)
        obs_image = _openvla_obs_image(obs, wrist_inset_bottom_right=self._wrist_inset_bottom_right)
        truncated = truncated.reshape(-1, 1)  # [B, 1]

        # calculate reward
        reward = self.get_reward(info)

        # process episode info
        if truncated.any():
            info["episode"] = {}
            for k in ["is_src_obj_grasped", "consecutive_grasp", "success"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v

        self._write_eval_debug(
            {
                "t": "step",
                "i": int(self._eval_debug_step),
                "raw": raw_action,
                "cmd": action,
                "tcp0": tcp0,
                "tcp1": tcp1,
                "act": None if tcp0 is None or tcp1 is None else tcp1 - tcp0,
                "q0": joints0.get("qpos"),
                "q1": joints1.get("qpos"),
                "qv1": joints1.get("qvel"),
                "lim1": joints1.get("limit_margin"),
                "lim_min": joints1.get("closest_limit_margin"),
                "lim_j": joints1.get("closest_limit_joint"),
                "reward": reward,
                "done": truncated,
                "info": _info_for_debug(info),
            }
        )
        self._eval_debug_step += 1

        return obs_image, reward, truncated, info


class SimlerContinuousWrapper:
    def __init__(self, all_args, unnorm_state, extra_seed=0):
        self.args = all_args
        self.unnorm_state = unnorm_state
        self._real2sim_robot_state = None

        self.num_envs = self.args.num_envs
        robot_control_mode = "arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos"
        self._wrist_inset_bottom_right = bool(self.args.use_wrist_camera) and self.args.env_id in AIRI_CUBES_ENV_IDS
        self._eval_debug_file = None
        self._eval_debug_step = 0

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode="rgb+segmentation",
            control_mode=robot_control_mode,
            sim_backend="gpu",
            enable_shadow=True,
            sim_config={
                "sim_freq": DEFAULT_REAL2SIM_SIM_FREQ,
                "control_freq": DEFAULT_REAL2SIM_CONTROL_FREQ,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
            use_wrist_camera=bool(self.args.use_wrist_camera),
        )
        self.env: BaseEnv = gym.make(**env_config)
        self.env.reset(seed=[self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)])
        self._reset_counter = 0

        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32)
        self._setup_eval_debug()

    def _setup_eval_debug(self):
        debug_path = str(getattr(self.args, "eval_debug_jsonl", "") or "").strip()
        if not debug_path:
            return
        path = Path(debug_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._eval_debug_file = path.open("w", encoding="utf-8")
        self._write_eval_debug({"t": "meta", "env_id": self.args.env_id, "num_envs": int(self.num_envs), "joint_names": _debug_joint_names(self.env)})

    def _write_eval_debug(self, payload: dict):
        if self._eval_debug_file is None:
            return
        self._eval_debug_file.write(json.dumps(_to_debug_value(payload), separators=(",", ":")) + "\n")
        self._eval_debug_file.flush()

    def _real2sim_reset_options(self) -> dict:
        if self.args.env_id not in REAL2SIM_RECORDER_ENV_IDS:
            return {}
        if self.args.env_id in AIRI_CUBES_V3_ENV_IDS:
            robot_base_pose_p = AIRI_CUBES_V3_ROBOT_BASE_POSE_P
            robot_base_pose_q = AIRI_CUBES_V3_ROBOT_BASE_POSE_Q
            robot_init_qpos = AIRI_CUBES_V3_ROBOT_INIT_QPOS
        elif self.args.env_id in {"PutObjectOnPlateAiriCubesRecorder-v1", "PickUpAiriCubeRecorder-v1"}:
            robot_base_pose_p = AIRI_CUBES_ROBOT_BASE_POSE_P
            robot_base_pose_q = AIRI_CUBES_ROBOT_BASE_POSE_Q
            robot_init_qpos = AIRI_CUBES_ROBOT_INIT_QPOS
        else:
            if self._real2sim_robot_state is None:
                from real2sim.calibrate_rc5_pose import _load_state_file

                self._real2sim_robot_state = _load_state_file("")
            robot_base_pose_p, robot_base_pose_q, robot_init_qpos = self._real2sim_robot_state
        return {
            "load_background": True,
            "show_debug_markers": False,
            "robot_far_away": False,
            "robot_base_pose_p": robot_base_pose_p.tolist(),
            "robot_base_pose_q": robot_base_pose_q.tolist(),
            "robot_init_qpos": robot_init_qpos.tolist(),
            "trajectory_instruction": str(getattr(self.args, "real2sim_instruction_template", "")).strip(),
        }

    def render_frame(self, camera_name: str = "", obs_img: torch.Tensor | None = None) -> np.ndarray:
        if camera_name:
            frames = self.env.unwrapped.render_rgb_array(camera_name=camera_name)
            if torch.is_tensor(frames):
                frames = frames.detach().cpu().numpy()
            else:
                frames = np.asarray(frames)
            return frames.astype(np.uint8, copy=False)
        if obs_img is None:
            raise ValueError("obs_img is required when camera_name is empty.")
        return obs_img.detach().cpu().numpy()

    def get_reward(self, info):
        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(info["success"].device)
        reward += info["is_src_obj_grasped"].reshape(-1, 1) * 0.1
        reward += info["consecutive_grasp"].reshape(-1, 1) * 0.1
        reward += info["success"].reshape(-1, 1) * 1.0
        reward_diff = reward - self.reward_old
        self.reward_old = reward
        return reward_diff

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return _unnormalize_continuous_action(raw_actions, self.unnorm_state)

    def reset(self, obj_set: str, same_init: bool = False):
        options = self._real2sim_reset_options()
        options["obj_set"] = obj_set
        if self.args.env_id in REAL2SIM_RECORDER_ENV_IDS:
            base_episode_id = self.args.seed * 1_000_000 + self._reset_counter * self.num_envs
            options["episode_id"] = (
                torch.arange(self.num_envs, device=self.env.device, dtype=torch.int64) + base_episode_id
            )
            self._reset_counter += 1
        if same_init:
            options["episode_id"] = torch.randint(1000000000, (1,)).expand(self.num_envs).to(self.env.device)
        if self.args.use_default_task:
            options["use_default_task"] = True

        obs, info = self.env.reset(options=options)
        obs_image = _openvla_obs_image(obs, wrist_inset_bottom_right=self._wrist_inset_bottom_right)
        instruction = self.env.unwrapped.get_language_instruction()

        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(obs_image.device)
        self._eval_debug_step = 0
        self._write_eval_debug(
            {
                "t": "reset",
                "obj_set": obj_set,
                "instruction": instruction,
                "tcp": _tcp_xyz(self.env),
                "q": _joint_debug(self.env).get("qpos"),
                "info": _info_for_debug(info),
            }
        )
        return obs_image, instruction, info

    def step(self, raw_action):
        tcp0 = _tcp_xyz(self.env)
        joints0 = _joint_debug(self.env)
        action = self._process_action(raw_action)
        obs, _reward, _terminated, truncated, info = self.env.step(action)
        tcp1 = _tcp_xyz(self.env)
        joints1 = _joint_debug(self.env)
        obs_image = _openvla_obs_image(obs, wrist_inset_bottom_right=self._wrist_inset_bottom_right)
        truncated = truncated.reshape(-1, 1)

        reward = self.get_reward(info)

        if truncated.any():
            info["episode"] = {}
            for k in ["is_src_obj_grasped", "consecutive_grasp", "success"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v

        self._write_eval_debug(
            {
                "t": "step",
                "i": int(self._eval_debug_step),
                "raw": raw_action,
                "cmd": action,
                "tcp0": tcp0,
                "tcp1": tcp1,
                "act": None if tcp0 is None or tcp1 is None else tcp1 - tcp0,
                "q0": joints0.get("qpos"),
                "q1": joints1.get("qpos"),
                "qv1": joints1.get("qvel"),
                "lim1": joints1.get("limit_margin"),
                "lim_min": joints1.get("closest_limit_margin"),
                "lim_j": joints1.get("closest_limit_joint"),
                "reward": reward,
                "done": truncated,
                "info": _info_for_debug(info),
            }
        )
        self._eval_debug_step += 1

        return obs_image, reward, truncated, info
