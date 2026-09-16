import gymnasium as gym
import json
import os
import sys
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
OPENREAL2SIM_ENV_IDS = {
    "OpenReal2Sim-v0",
}
WIDOWX_DELTA_CONTROL_MODE = "arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos"
RC5_TARGET_DELTA_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
_SIM2REAL_ROOT = Path(__file__).resolve().parents[3] / "sim2real"


def _is_openreal2sim_env(env_id: str) -> bool:
    return env_id in OPENREAL2SIM_ENV_IDS


def _sim_backend() -> str:
    # GPU 3 PhysX CUDA hangs on this box; eval_sim.sh can set cpu.
    return os.environ.get("OPENREAL2SIM_SIM_BACKEND", "gpu")


def _obs_mode_for_env(env_id: str) -> str:
    # OpenVLA consumes RGB only. Segmentation textures double GPU camera buffers
    # and can fail with "cannot create buffer" after the 7B VLA is loaded.
    if _is_openreal2sim_env(env_id):
        return "rgb"
    return "rgb+segmentation"


def _control_mode_for_env(env_id: str) -> str:
    if _is_openreal2sim_env(env_id):
        return RC5_TARGET_DELTA_CONTROL_MODE
    return WIDOWX_DELTA_CONTROL_MODE


def _ensure_openreal2sim_importable() -> None:
    sim2real_root = str(_SIM2REAL_ROOT)
    # Append, do not insert at 0: sim2real vendors its own mani_skill package and
    # must not shadow the ManiSkill already imported by PPO / SimplerEnv.
    if sim2real_root not in sys.path:
        sys.path.append(sim2real_root)


def _openreal2sim_rl_gym_kwargs(*, use_wrist_camera: bool) -> dict:
    # Deferred import: only the reconstructed AIRI-table env needs sim2real.
    _ensure_openreal2sim_importable()
    import openreal2sim.simulation.maniskill  # noqa: F401
    from openreal2sim.simulation.maniskill.rl_gym import build_openreal2sim_rl_gym_kwargs

    return build_openreal2sim_rl_gym_kwargs(use_wrist_camera=use_wrist_camera)


def _as_float_col(value, device) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    return tensor.reshape(-1, 1)


# Inference-side counterpart of `retarget_frame` in the sft_v2 RLDS builder, for running a
# policy trained on one base convention against a scene defined in another (release.v4
# rotated the RC5 base yaw by -85.98 deg with a compensating +85.98 deg on joint0). Same
# sign convention as the builder: the yaw rotates model-frame xy deltas into the env frame,
# and the joint offsets are what the env adds on top of the training convention, so they are
# subtracted from proprio before the policy sees it. All default to off.
_ACTION_YAW_RAD = np.radians(float(os.environ.get("RLVLA_ACTION_YAW_DEG", "0") or 0.0))
_PROPRIO_J0_OFFSET_RAD = float(os.environ.get("RLVLA_PROPRIO_J0_OFFSET_RAD", "0") or 0.0)
_PROPRIO_J3_OFFSET_RAD = float(os.environ.get("RLVLA_PROPRIO_J3_OFFSET_RAD", "0") or 0.0)


def _clip_ee_delta(action: torch.Tensor, max_ee_delta: float) -> torch.Tensor:
    limit = float(max_ee_delta)
    if limit <= 0.0:
        return action
    return torch.cat([action[:, :3].clamp(-limit, limit), action[:, 3:]], dim=1)


def _effective_yeet_coef(args, step) -> float:
    coef = float(getattr(args, "reward_yeet_coef", 0.0) or 0.0)
    warmup = int(getattr(args, "reward_yeet_warmup_steps", 0) or 0)
    if coef > 0.0 and warmup > 0 and step is not None:
        coef = coef * min(1.0, max(0.0, float(step) / float(warmup)))
    return coef


def _shaped_grasp_reward(info, reward_old, args, step=None):
    device = info["success"].device
    grasped = info.get("instant_is_src_obj_grasped", info["is_src_obj_grasped"])
    consecutive = info.get("instant_consecutive_grasp", info["consecutive_grasp"])
    success = _as_float_col(info["success"], device)
    grasped = _as_float_col(grasped, device)
    consecutive = _as_float_col(consecutive, device)

    height_t = None
    height = info.get("obj_height_above_table")
    if height is not None:
        height_t = _as_float_col(height, device)

    max_lift = float(getattr(args, "reward_max_lift_height", 0.0) or 0.0)
    if height_t is not None and max_lift > 0.0:
        success = success * (height_t <= max_lift).to(dtype=torch.float32)

    reward_old = reward_old.to(device=device, dtype=torch.float32).reshape(-1, 1)
    reward = grasped * 0.1 + consecutive * 0.1 + success * 1.0

    lift_coef = float(getattr(args, "reward_lift_coef", 0.0) or 0.0)
    lift_ref = float(getattr(args, "reward_lift_height", 0.05) or 0.05)
    if lift_coef > 0.0 and height_t is not None:
        progress = (height_t / max(lift_ref, 1e-6)).clamp(0.0, 1.0)
        reward = reward + lift_coef * grasped * progress

    yeet_h = float(getattr(args, "reward_yeet_height", 0.0) or 0.0)
    yeet_coef = _effective_yeet_coef(args, step)
    if yeet_coef > 0.0 and yeet_h > 0.0 and height_t is not None:
        over = (height_t - yeet_h).clamp(min=0.0)
        # Unbounded, a diverged object at 8 m yields a -16 potential that swamps the
        # +1.0 success term and makes the critic unfittable.
        yeet_clip = float(getattr(args, "reward_yeet_clip", 0.0) or 0.0)
        if yeet_clip > 0.0:
            over = over.clamp(max=yeet_clip)
        if bool(getattr(args, "reward_yeet_grasp_only", False)):
            over = over * grasped
        reward = reward - yeet_coef * over

    reach_coef = float(getattr(args, "reward_reach_coef", 0.0) or 0.0)
    if reach_coef > 0.0 and "gripper_obj_dist" in info:
        reach_clip = float(getattr(args, "reward_reach_clip", 0.4) or 0.4)
        dist = _as_float_col(info["gripper_obj_dist"], device).clamp(max=reach_clip)
        reward = reward - reach_coef * dist

    reward_diff = reward - reward_old
    return reward_diff, reward


class GraspHoldAssist:
    """Force-close after first close, absorb after K consecutive successes, and
    absorb envs whose object has left the reachable workspace."""

    def __init__(self, args, num_envs: int):
        self.args = args
        self.num_envs = int(num_envs)
        self.sticky_steps = int(getattr(args, "sticky_gripper_steps", 0) or 0)
        self.terminate_steps = int(getattr(args, "success_terminate_steps", 0) or 0)
        self.escape_height = float(getattr(args, "escape_height", 0.0) or 0.0)
        self.escape_below = float(getattr(args, "escape_below", 0.0) or 0.0)
        self.escape_dist = float(getattr(args, "escape_dist", 0.0) or 0.0)
        self.escape_penalty = float(getattr(args, "reward_escape_penalty", 0.0) or 0.0)
        self.reward_scale = float(getattr(args, "reward_scale", 1.0) or 1.0)
        self._closed_once = None
        self._sticky_left = None
        self._absorbed = None
        self._streak = None
        self._latched_success = None
        self._escaped = None

    def reset(self, device):
        n = self.num_envs
        self._closed_once = torch.zeros(n, dtype=torch.bool, device=device)
        self._sticky_left = torch.zeros(n, dtype=torch.int32, device=device)
        self._absorbed = torch.zeros(n, dtype=torch.bool, device=device)
        self._streak = torch.zeros(n, dtype=torch.int32, device=device)
        self._latched_success = torch.zeros(n, dtype=torch.bool, device=device)
        self._escaped = torch.zeros(n, dtype=torch.bool, device=device)

    def before_physics(self, action: torch.Tensor) -> torch.Tensor:
        if self._absorbed is None:
            self.reset(action.device)
        action = action.clone()
        if self._absorbed.any():
            action[self._absorbed, :6] = 0.0
            action[self._absorbed, 6] = 0.0  # openness 0.0 = fully closed
        if self.sticky_steps == 0:
            return action
        closed = action[:, 6] < 0.5  # openness below halfway counts as a close command
        if self.sticky_steps < 0:
            self._closed_once = self._closed_once | closed
            if self._closed_once.any():
                action[self._closed_once, 6] = 0.0
            return action
        refresh = torch.full_like(self._sticky_left, int(self.sticky_steps))
        self._sticky_left = torch.where(closed, refresh, self._sticky_left)
        force = self._sticky_left > 0
        if force.any():
            action[force, 6] = 0.0
        self._sticky_left = torch.clamp(self._sticky_left - 1, min=0)
        return action

    def _detect_escape(self, info, device):
        """True where the object is no longer in a physically plausible place."""
        if self.escape_height <= 0.0 and self.escape_below <= 0.0 and self.escape_dist <= 0.0:
            return None
        escaped = None
        height = info.get("obj_height_above_table")
        if height is not None and (self.escape_height > 0.0 or self.escape_below > 0.0):
            h = _as_float_col(height, device).reshape(-1)
            if self.escape_height > 0.0:
                escaped = h > self.escape_height
            if self.escape_below > 0.0:
                fell = h < -self.escape_below
                escaped = fell if escaped is None else (escaped | fell)
        dist = info.get("gripper_obj_dist")
        if dist is not None and self.escape_dist > 0.0:
            flew = _as_float_col(dist, device).reshape(-1) > self.escape_dist
            escaped = flew if escaped is None else (escaped | flew)
        return escaped

    def after_physics(self, info, reward):
        if self._absorbed is None:
            extra = torch.zeros(reward.shape[0], 1, device=reward.device, dtype=torch.bool)
            return self._scale(reward), extra
        was_absorbed = self._absorbed

        # Detect escapes first so this step's success streak can ignore them. The env
        # reports success purely from object-vs-goal distance, so a cube that has fallen
        # through the table near the goal reports success while the arm is metres away —
        # observed in run8 at gripper_obj_dist 4.46 m, obj_height -0.017 m. Penalising the
        # reward is not enough: without this mask the streak still latches and SR reports
        # the hack as a real pick.
        escaped = self._detect_escape(info, reward.device)
        newly_escaped = None
        if escaped is not None:
            newly_escaped = escaped & ~self._absorbed
            self._escaped = self._escaped | escaped

        if self.terminate_steps > 0:
            success = info["success"].reshape(-1).to(device=reward.device)
            if success.dtype != torch.bool:
                success = success > 0.5
            if self._escaped is not None:
                success = success & ~self._escaped
            self._streak = torch.where(success, self._streak + 1, torch.zeros_like(self._streak))
            held = self._streak >= int(self.terminate_steps)
            self._absorbed = self._absorbed | held
            # Latch from the hold condition only: an escaped env is absorbed too, and
            # latching off _absorbed would report it as a success.
            self._latched_success = self._latched_success | held

        if escaped is not None:
            self._absorbed = self._absorbed | escaped

        reward = torch.where(was_absorbed.reshape(-1, 1), torch.zeros_like(reward), reward)
        # Charge escapes after the zeroing: a newly escaped env was not absorbed before,
        # so its reward survived above and the penalty lands exactly once.
        if newly_escaped is not None and self.escape_penalty > 0.0:
            reward = reward - self.escape_penalty * newly_escaped.reshape(-1, 1).to(reward.dtype)
        extra = self._absorbed.reshape(-1, 1)
        # Scale last so every term above (shaped potential, absorb zeroing, escape
        # penalty) keeps its relative weight; only the critic's target changes scale.
        return self._scale(reward), extra

    def _scale(self, reward):
        if self.reward_scale == 1.0:
            return reward
        return reward * self.reward_scale

    def escaped_flags(self):
        """Per-env escape flags, or None when escape detection is disabled.

        Reported as env/escaped so reward hacking is directly measurable: a run that
        grasps but does not succeed used to be diagnosable only by eyeballing mean
        gripper_obj_dist in the per-step dumps.
        """
        if self._escaped is None:
            return None
        if self.escape_height <= 0.0 and self.escape_below <= 0.0 and self.escape_dist <= 0.0:
            return None
        return [float(self._escaped[i].item()) for i in range(int(self._escaped.numel()))]

    def override_episode_success(self, values):
        if self.terminate_steps <= 0 or self._latched_success is None:
            return values
        out = []
        for i, value in enumerate(values):
            latched = bool(self._latched_success[i].item()) if i < int(self._latched_success.numel()) else False
            out.append(bool(value) or latched)
        return out


def _reward_shaping_log_line(args) -> str:
    return (
        "Reward shaping | "
        f"reach_coef={float(getattr(args, 'reward_reach_coef', 0.0) or 0.0)} | "
        f"reach_clip={float(getattr(args, 'reward_reach_clip', 0.4) or 0.4)} | "
        f"max_lift={float(getattr(args, 'reward_max_lift_height', 0.0) or 0.0)} | "
        f"lift_coef={float(getattr(args, 'reward_lift_coef', 0.0) or 0.0)} | "
        f"yeet_h={float(getattr(args, 'reward_yeet_height', 0.0) or 0.0)} | "
        f"yeet_coef={float(getattr(args, 'reward_yeet_coef', 0.0) or 0.0)} | "
        f"yeet_grasp_only={bool(getattr(args, 'reward_yeet_grasp_only', False))} | "
        f"yeet_warmup={int(getattr(args, 'reward_yeet_warmup_steps', 0) or 0)} | "
        f"yeet_clip={float(getattr(args, 'reward_yeet_clip', 0.0) or 0.0)} | "
        f"escape_h={float(getattr(args, 'escape_height', 0.0) or 0.0)} | "
        f"escape_below={float(getattr(args, 'escape_below', 0.0) or 0.0)} | "
        f"escape_dist={float(getattr(args, 'escape_dist', 0.0) or 0.0)} | "
        f"escape_penalty={float(getattr(args, 'reward_escape_penalty', 0.0) or 0.0)} | "
        f"reward_scale={float(getattr(args, 'reward_scale', 1.0) or 1.0)} | "
        f"max_ee_delta={float(getattr(args, 'max_ee_delta', 0.0) or 0.0)} | "
        f"sticky_gripper={int(getattr(args, 'sticky_gripper_steps', 0) or 0)} | "
        f"success_terminate={int(getattr(args, 'success_terminate_steps', 0) or 0)}"
    )


def _fill_episode_info(info, truncated, hold_assist: GraspHoldAssist):
    if not truncated.any():
        return
    info["episode"] = {}
    n = int(truncated.shape[0])
    for key in [
        "is_src_obj_grasped",
        "consecutive_grasp",
        "success",
        "instant_is_src_obj_grasped",
        "instant_consecutive_grasp",
    ]:
        if key not in info:
            continue
        values = [info[key][idx].item() for idx in range(n)]
        if key == "success":
            values = hold_assist.override_episode_success(values)
        info["episode"][key] = values

    escaped = hold_assist.escaped_flags()
    if escaped is not None:
        info["episode"]["escaped"] = escaped


def _quantize_gripper_openness(gripper: torch.Tensor) -> torch.Tensor:
    """Snap a continuous openness command to the discrete levels {0.0, 0.2, ..., 1.0}.

    Convention: 1.0 = fully open, 0.0 = fully closed (matches the env's
    RCLevelHandController and the Bridge open_gripper convention used in SFT data).
    """
    return (gripper.to(torch.float32) * 5.0).round().clamp(0.0, 5.0) / 5.0


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
    gripper = _quantize_gripper_openness(raw_action[:, 6:7])
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
    if "3rd_view_camera" in sensor_data and "rgb" in sensor_data["3rd_view_camera"]:
        scene_rgb = sensor_data["3rd_view_camera"]["rgb"]
    elif "base_camera" in sensor_data and "rgb" in sensor_data["base_camera"]:
        scene_rgb = sensor_data["base_camera"]["rgb"]
    else:
        available = list(sensor_data.keys()) if isinstance(sensor_data, dict) else type(sensor_data)
        raise KeyError(f"No scene RGB camera found in sensor_data keys={available}")
    wrist_data = sensor_data.get(WRIST_CAMERA_NAME)
    wrist_rgb = None if wrist_data is None else wrist_data["rgb"]
    return _compose_wrist_inset(scene_rgb, wrist_rgb, bottom_right=wrist_inset_bottom_right)


def _openvla_scene_wrist_images(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenVLA_V2: return (scene_rgb, wrist_rgb) as separate uint8 [B, H, W, 3] tensors."""
    sensor_data = obs["sensor_data"]
    if "3rd_view_camera" in sensor_data and "rgb" in sensor_data["3rd_view_camera"]:
        scene_rgb = sensor_data["3rd_view_camera"]["rgb"]
    elif "base_camera" in sensor_data and "rgb" in sensor_data["base_camera"]:
        scene_rgb = sensor_data["base_camera"]["rgb"]
    else:
        available = list(sensor_data.keys()) if isinstance(sensor_data, dict) else type(sensor_data)
        raise KeyError(f"No scene RGB camera found in sensor_data keys={available}")
    wrist_data = sensor_data.get(WRIST_CAMERA_NAME)
    if wrist_data is None or "rgb" not in wrist_data:
        raise KeyError(f"No wrist camera RGB found in sensor_data keys={list(sensor_data.keys())}")
    return scene_rgb.to(torch.uint8), wrist_data["rgb"].to(torch.uint8)


class SimlerWrapper:
    def __init__(self, all_args, unnorm_state, extra_seed=0):
        self.args = all_args
        self.unnorm_state = unnorm_state
        self._real2sim_robot_state = None
        self._vla_model_variant = str(getattr(self.args, "vla_model_variant", "v1"))

        self.num_envs = self.args.num_envs
        robot_control_mode = _control_mode_for_env(self.args.env_id)
        self._wrist_inset_bottom_right = bool(self.args.use_wrist_camera) and (
            self.args.env_id in AIRI_CUBES_ENV_IDS or _is_openreal2sim_env(self.args.env_id)
        )
        self._eval_debug_file = None
        self._eval_debug_step = 0

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode=_obs_mode_for_env(self.args.env_id),
            control_mode=robot_control_mode,
            sim_backend=_sim_backend(),
            enable_shadow=True,
            sim_config={
                "sim_freq": DEFAULT_REAL2SIM_SIM_FREQ,
                "control_freq": DEFAULT_REAL2SIM_CONTROL_FREQ,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
            use_wrist_camera=bool(self.args.use_wrist_camera),
        )
        if _is_openreal2sim_env(self.args.env_id):
            env_config.update(_openreal2sim_rl_gym_kwargs(use_wrist_camera=bool(self.args.use_wrist_camera)))
            env_config["obs_mode"] = _obs_mode_for_env(self.args.env_id)
        self.env: BaseEnv = gym.make(**env_config)
        self.env.reset(seed=[self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)])
        self._reset_counter = 0

        # variables
        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32)  # [B, 1]
        self.hold_assist = GraspHoldAssist(self.args, self.num_envs)
        self._shaping_step = 0

        # constants
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        print(_reward_shaping_log_line(self.args))
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

    def set_shaping_step(self, step: int):
        self._shaping_step = int(step)

    def _get_proprio_7d(self) -> torch.Tensor:
        """7D proprio for OpenVLA_V2: 6 arm joint positions + gripper-closure scalar in [0, 1].

        The closure scalar is the mean over the 16 Aero Hand joints of qpos / hand_close_qpos
        (hand_open_qpos is all zeros), clipped to [0, 1]; 0 = fully open, 1 = fully closed.
        """
        agent = self.env.unwrapped.agent
        qpos = agent.robot.get_qpos()
        if not torch.is_tensor(qpos):
            qpos = torch.as_tensor(qpos, device=self.env.device)
        qpos = qpos.to(dtype=torch.float32)
        n_arm = len(agent.arm_joint_names)
        close_qpos = torch.as_tensor(
            np.asarray(agent.hand_close_qpos, dtype=np.float32), device=qpos.device
        )
        arm_qpos = qpos[:, :n_arm]
        hand_closure = (qpos[:, n_arm:] / close_qpos).clamp(0.0, 1.0).mean(dim=1, keepdim=True)
        if _PROPRIO_J0_OFFSET_RAD or _PROPRIO_J3_OFFSET_RAD:
            arm_qpos = arm_qpos.clone()
            arm_qpos[:, 0] -= _PROPRIO_J0_OFFSET_RAD
            arm_qpos[:, 3] -= _PROPRIO_J3_OFFSET_RAD
        return torch.cat([arm_qpos, hand_closure], dim=1)

    def _form_obs(self, obs: dict):
        """Return the policy observation: composited image tensor (v1) or dict of separate inputs (v2)."""
        if self._vla_model_variant == "v2":
            scene_rgb, wrist_rgb = _openvla_scene_wrist_images(obs)
            return {
                "image": scene_rgb,
                "image_wrist": wrist_rgb,
                "proprio": self._get_proprio_7d(),
            }
        return _openvla_obs_image(obs, wrist_inset_bottom_right=self._wrist_inset_bottom_right)

    def get_reward(self, info):
        reward_diff, reward = _shaped_grasp_reward(info, self.reward_old, self.args, step=self._shaping_step)
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
        action["gripper"] = _quantize_gripper_openness(torch.as_tensor(raw_action["open_gripper"]))  # [B, 1] levels

        # origin euler
        action["rot_axangle"] = raw_action["rotation_delta"]

        action = {k: torch.tensor(v) for k, v in action.items()}  # to float32 ?

        action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

        # to tpdv
        action = action.to(raw_actions.device)
        if _ACTION_YAW_RAD:
            # Rotate before clipping: the env clips per axis in its own frame.
            cos_t, sin_t = np.cos(_ACTION_YAW_RAD), np.sin(_ACTION_YAW_RAD)
            x, y = action[:, 0].clone(), action[:, 1].clone()
            action[:, 0] = cos_t * x - sin_t * y
            action[:, 1] = sin_t * x + cos_t * y
        return _clip_ee_delta(action, float(getattr(self.args, "max_ee_delta", 0.0) or 0.0))

    def reset(self, obj_set: str, same_init: bool = False):
        options = self._real2sim_reset_options()
        options["obj_set"] = obj_set
        if self.args.env_id in REAL2SIM_RECORDER_ENV_IDS or _is_openreal2sim_env(self.args.env_id):
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
        obs_image = self._form_obs(obs)
        instruction = self.env.unwrapped.get_language_instruction()

        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(self.env.device)  # [B, 1]
        self.hold_assist.reset(self.env.device)
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
        action = self.hold_assist.before_physics(self._process_action(raw_action))

        obs, _reward, _terminated, truncated, info = self.env.step(action)
        tcp1 = _tcp_xyz(self.env)
        joints1 = _joint_debug(self.env)
        obs_image = self._form_obs(obs)
        time_limit = truncated.reshape(-1, 1)  # [B, 1]

        # calculate reward
        reward = self.get_reward(info)
        reward, absorbed = self.hold_assist.after_physics(info, reward)
        done = time_limit.to(dtype=torch.bool) | absorbed.to(dtype=torch.bool, device=time_limit.device)
        _fill_episode_info(info, time_limit, self.hold_assist)
        truncated = done.to(dtype=time_limit.dtype)

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
        robot_control_mode = _control_mode_for_env(self.args.env_id)
        self._wrist_inset_bottom_right = bool(self.args.use_wrist_camera) and (
            self.args.env_id in AIRI_CUBES_ENV_IDS or _is_openreal2sim_env(self.args.env_id)
        )
        self._eval_debug_file = None
        self._eval_debug_step = 0

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode=_obs_mode_for_env(self.args.env_id),
            control_mode=robot_control_mode,
            sim_backend=_sim_backend(),
            enable_shadow=True,
            sim_config={
                "sim_freq": DEFAULT_REAL2SIM_SIM_FREQ,
                "control_freq": DEFAULT_REAL2SIM_CONTROL_FREQ,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
            use_wrist_camera=bool(self.args.use_wrist_camera),
        )
        if _is_openreal2sim_env(self.args.env_id):
            env_config.update(_openreal2sim_rl_gym_kwargs(use_wrist_camera=bool(self.args.use_wrist_camera)))
            env_config["obs_mode"] = _obs_mode_for_env(self.args.env_id)
        self.env: BaseEnv = gym.make(**env_config)
        self.env.reset(seed=[self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)])
        self._reset_counter = 0

        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32)
        self.hold_assist = GraspHoldAssist(self.args, self.num_envs)
        self._shaping_step = 0
        print(_reward_shaping_log_line(self.args))
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

    def set_shaping_step(self, step: int):
        self._shaping_step = int(step)

    def get_reward(self, info):
        reward_diff, reward = _shaped_grasp_reward(info, self.reward_old, self.args, step=self._shaping_step)
        self.reward_old = reward
        return reward_diff

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        action = _unnormalize_continuous_action(raw_actions, self.unnorm_state)
        return _clip_ee_delta(action, float(getattr(self.args, "max_ee_delta", 0.0) or 0.0))

    def reset(self, obj_set: str, same_init: bool = False):
        options = self._real2sim_reset_options()
        options["obj_set"] = obj_set
        if self.args.env_id in REAL2SIM_RECORDER_ENV_IDS or _is_openreal2sim_env(self.args.env_id):
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
        self.hold_assist.reset(obs_image.device)
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
        action = self.hold_assist.before_physics(self._process_action(raw_action))
        obs, _reward, _terminated, truncated, info = self.env.step(action)
        tcp1 = _tcp_xyz(self.env)
        joints1 = _joint_debug(self.env)
        obs_image = _openvla_obs_image(obs, wrist_inset_bottom_right=self._wrist_inset_bottom_right)
        time_limit = truncated.reshape(-1, 1)

        reward = self.get_reward(info)
        reward, absorbed = self.hold_assist.after_physics(info, reward)
        done = time_limit.to(dtype=torch.bool) | absorbed.to(dtype=torch.bool, device=time_limit.device)
        _fill_episode_info(info, time_limit, self.hold_assist)
        truncated = done.to(dtype=time_limit.dtype)

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
