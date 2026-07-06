#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import shlex
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MANISKILL_ROOT = REPO_ROOT / "ManiSkill"
if LOCAL_MANISKILL_ROOT.is_dir():
    local_maniskill_root_str = str(LOCAL_MANISKILL_ROOT)
    if local_maniskill_root_str not in sys.path:
        sys.path.insert(0, local_maniskill_root_str)

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

import mani_skill
import mani_skill.envs.sapien_env as mani_skill_sapien_env
from mani_skill.envs.sapien_env import BaseEnv
from real2sim.debug_paths import RC5_POSE_DEBUG_STATE_FILE, RC5_SYNC_SHELL_DIR
from real2sim.openreal2sim_validation import (
    PROBE_CARROT_POSE_P,
    PROBE_CARROT_POSE_Q,
    PROBE_PLATE_POSE_P,
    PROBE_PLATE_POSE_Q,
    ROBOT_INIT_QPOS,
    observation_camera_override,
)
from real2sim.rc5_joint_convention import (
    RC5_REAL_TO_SIM_ZERO_OFFSET_DEG,
    real_to_sim_arm_joints,
)
from real2sim.real_robot.delta_pose_common import (
    compose_delta_pose,
    connect_and_prepare_robot,
)


DEFAULT_SIM_STATE_FILE = str(RC5_POSE_DEBUG_STATE_FILE)
DEFAULT_REAL_HOME_POSE_FILE = "real2sim/real_home_pose.txt"
DEFAULT_DEBUG_LOG_NAME = "sync_rc5_compare.log"


def _to_uint8_image(frame) -> np.ndarray:
    if torch.is_tensor(frame):
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 3:
            if frame.dtype.is_floating_point:
                max_val = float(frame.max().item())
                if max_val <= 1.0 + 1e-6:
                    frame = frame * 255.0
            return frame.clamp(0, 255).to(torch.uint8).cpu().numpy()

    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(arr.max()) if arr.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _extract_obs_frame(obs: dict) -> np.ndarray:
    sensor_data = obs.get("sensor_data", {})
    if "3rd_view_camera" not in sensor_data:
        raise KeyError(
            "Expected observation camera '3rd_view_camera' was not found. "
            f"Available keys: {sorted(sensor_data.keys())}"
        )
    return _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])


def _raw_pose_to_xyzrpy_rad(raw_pose: np.ndarray) -> np.ndarray:
    quat_wxyz = raw_pose[3:7]
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return np.concatenate([raw_pose[:3], rpy]).astype(np.float64)


def _tensor_to_numpy(data) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _format_pose_xyzrpy_rad(pose: np.ndarray) -> str:
    xyz = ", ".join(f"{value:.5f}" for value in pose[:3])
    rpy_deg = ", ".join(f"{value:.2f}" for value in np.rad2deg(pose[3:6]))
    return f"xyz[m]=[{xyz}] rpy[deg]=[{rpy_deg}]"


def _format_array(values: np.ndarray, *, precision: int = 6) -> str:
    return np.array2string(
        np.asarray(values),
        precision=precision,
        separator=", ",
        suppress_small=False,
    )


def _pose_raw_to_xyzquat(raw_pose) -> np.ndarray:
    return np.asarray(raw_pose, dtype=np.float64).reshape(-1)


def _pose_delta_base_xyzrpy_rad(
    before_xyzrpy_rad: np.ndarray,
    after_xyzrpy_rad: np.ndarray,
) -> np.ndarray:
    delta_xyz = after_xyzrpy_rad[:3] - before_xyzrpy_rad[:3]
    before_rot = Rotation.from_euler("xyz", before_xyzrpy_rad[3:6], degrees=False)
    after_rot = Rotation.from_euler("xyz", after_xyzrpy_rad[3:6], degrees=False)
    delta_rpy = (after_rot * before_rot.inv()).as_euler("xyz", degrees=False)
    return np.concatenate([delta_xyz, delta_rpy]).astype(np.float64)


def _pose_delta_local_xyzrpy_rad(
    before_xyzrpy_rad: np.ndarray,
    after_xyzrpy_rad: np.ndarray,
) -> np.ndarray:
    before_rot = Rotation.from_euler("xyz", before_xyzrpy_rad[3:6], degrees=False)
    delta_xyz_world = after_xyzrpy_rad[:3] - before_xyzrpy_rad[:3]
    delta_xyz_local = before_rot.inv().apply(delta_xyz_world)
    delta_rpy = _pose_delta_base_xyzrpy_rad(before_xyzrpy_rad, after_xyzrpy_rad)[3:6]
    return np.concatenate([delta_xyz_local, delta_rpy]).astype(np.float64)


def _wrap_angle_diff_rad(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return ((values + np.pi) % (2.0 * np.pi)) - np.pi


def _parse_rpy_deg_triplet(raw: str) -> np.ndarray:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"expected three comma-separated values like '0,0,0', got {raw!r}"
        )
    return np.array([float(part) for part in parts], dtype=np.float64)


def _parse_xyz_triplet(raw: str) -> np.ndarray:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"expected three comma-separated values like '0,0,0.1', got {raw!r}"
        )
    return np.array([float(part) for part in parts], dtype=np.float64)


def _read_single_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _build_env(args: argparse.Namespace) -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_delta_pose_align2",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": observation_camera_override(args.observation_camera_mode),
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        robot_uids="rc5_aero_hand_right_openreal2sim_validation",
    )


def _override_sim_ee_link(env: BaseEnv, ee_link_name: str | None) -> None:
    if not ee_link_name:
        return

    agent = env.unwrapped.agent
    if not hasattr(agent, "robot") or not hasattr(agent.robot, "links_map"):
        raise AttributeError("Sim agent does not expose robot link information.")
    if ee_link_name not in agent.robot.links_map:
        available = ", ".join(sorted(agent.robot.links_map.keys()))
        raise ValueError(
            f"Requested sim EE link {ee_link_name!r} was not found. "
            f"Available links: {available}"
        )

    agent.ee_link_name = ee_link_name
    agent.controllers = {}
    agent.supported_control_modes = list(agent._controller_configs.keys())
    agent.set_control_mode(agent.control_mode)
    agent._after_init()


def _load_sim_state_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_base_pose_p = np.array([0.0, -2.45, 0.12047362327575684], dtype=np.float32)
    sim_base_pose_q = np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
    sim_init_qpos = np.array(ROBOT_INIT_QPOS, dtype=np.float32)
    if not path:
        return sim_base_pose_p, sim_base_pose_q, sim_init_qpos

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("robot_base_pose_p="):
            sim_base_pose_p = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
        elif line.startswith("robot_base_pose_q="):
            sim_base_pose_q = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
        elif line.startswith("robot_init_qpos="):
            sim_init_qpos = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float32)
    # Keep the arm joints in a principal-angle range so the sim IK/controller
    # starts from a numerically stable configuration equivalent to the saved pose.
    sim_init_qpos[:6] = ((sim_init_qpos[:6] + np.pi) % (2.0 * np.pi)) - np.pi
    return sim_base_pose_p, sim_base_pose_q, sim_init_qpos


def _load_real_home_tcp_pose(path: str) -> np.ndarray:
    tcp_pose = np.array(
        [0.2240750751128322, 0.419037175472987, 0.18830342524489624, -3.1372092621750185, 0.004318070578632195, -0.8057179835923796],
        dtype=np.float64,
    )
    if not path:
        return tcp_pose

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("current_tcp_rad="):
            tcp_pose = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float64)
            break
    return tcp_pose


def _load_real_home_arm_joints_rad(path: str) -> np.ndarray | None:
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("current_joint_rad="):
            joints = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float64)
            if joints.shape[0] < 6:
                raise ValueError(
                    f"Expected at least 6 arm joints in current_joint_rad, got shape {joints.shape}"
                )
            return joints[:6]
    return None


@dataclass
class SnapshotRecord:
    image_path: Path
    latest_image_path: Path
    meta_path: Path


class SyncRC5Session:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_path = self.output_dir / args.debug_log_name
        self.sim_delta_remap_rpy_deg = _parse_rpy_deg_triplet(args.sim_delta_remap_rpy_deg)
        self.sim_delta_remap = Rotation.from_euler(
            "xyz",
            np.deg2rad(self.sim_delta_remap_rpy_deg),
            degrees=False,
        )
        self.sim_tcp_marker_offset_m = _parse_xyz_triplet(args.sim_tcp_marker_offset_m)

        self.sim_base_pose_p, self.sim_base_pose_q, self.sim_init_qpos = _load_sim_state_file(args.sim_state_file)
        self.real_home_tcp_rad = _load_real_home_tcp_pose(args.real_home_pose_file)
        self.real_home_joint_rad = _load_real_home_arm_joints_rad(args.real_home_pose_file)
        self.real_to_sim_zero_offset_deg = _parse_rpy_deg_triplet("0,0,0")
        self.snapshot_index = 0
        self.command_index = 0
        self.sim_non_arm_joint_targets: np.ndarray | None = None
        self.sim_non_arm_joints = []
        self.sim_non_arm_joint_indices: torch.Tensor | None = None
        self._orig_before_simulation_step = None
        self.sim_tcp_marker = None

        self.real_to_sim_zero_offset_rad = np.deg2rad(
            np.array(
                [float(v) for v in args.real_to_sim_zero_offset_deg.split(",")],
                dtype=np.float64,
            )
        )
        if self.real_to_sim_zero_offset_rad.shape != (6,):
            raise ValueError(
                "--real-to-sim-zero-offset-deg must contain 6 comma-separated values"
            )
        if args.sim_init_from_real_home_joints:
            if self.real_home_joint_rad is None:
                raise ValueError(
                    "Requested --sim-init-from-real-home-joints but "
                    f"{args.real_home_pose_file!r} does not contain current_joint_rad"
                )
            self.sim_init_qpos[:6] = real_to_sim_arm_joints(
                self.real_home_joint_rad,
                zero_offset_rad=self.real_to_sim_zero_offset_rad,
            ).astype(np.float32)

        self.env = _build_env(args)
        _override_sim_ee_link(self.env, args.sim_ee_link_name)
        self.obs: dict | None = None
        self.info: dict | None = None
        self.robot = connect_and_prepare_robot(args)

    def _sim_arm_controller(self):
        controller = self.env.unwrapped.agent.controller
        if hasattr(controller, "controllers"):
            return controller.controllers.get("arm")
        return controller

    def _apply_sim_arm_target_qpos_direct(self, target_qpos) -> None:
        arm_controller = self._sim_arm_controller()
        if arm_controller is None or target_qpos is None:
            return
        robot = self.env.unwrapped.agent.robot
        full_qpos = robot.get_qpos().clone()
        full_qvel = robot.get_qvel().clone()
        active_joint_indices = arm_controller.active_joint_indices.long()
        target_qpos = target_qpos.to(dtype=full_qpos.dtype, device=full_qpos.device)
        full_qpos[:, active_joint_indices] = target_qpos
        full_qvel[:, active_joint_indices] = 0.0
        robot.set_qpos(full_qpos)
        robot.set_qvel(full_qvel)
        if self.env.unwrapped.gpu_sim_enabled:
            self.env.unwrapped.scene._gpu_apply_all()
            self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            self.env.unwrapped.scene._gpu_fetch_all()
        self.info = self.env.get_info()
        self.obs = self.env.get_obs(self.info)
        self._update_sim_tcp_marker_pose()

    def _ensure_sim_tcp_marker(self) -> None:
        if self.sim_tcp_marker is not None:
            return
        builder = self.env.unwrapped.scene.create_actor_builder()
        builder.add_sphere_visual(
            pose=sapien.Pose(),
            radius=0.05,
            material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 0.9]),
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        self.sim_tcp_marker = builder.build_kinematic(name="sim_tcp_marker")
        self._update_sim_tcp_marker_pose()

    def _update_sim_tcp_marker_pose(self) -> None:
        if self.sim_tcp_marker is None:
            return
        tcp_link = None
        arm_controller = self._sim_arm_controller()
        if arm_controller is not None and getattr(arm_controller, "ee_link", None) is not None:
            tcp_link = arm_controller.ee_link
        else:
            agent = self.env.unwrapped.agent
            tcp_link = agent.robot.links_map.get(agent.ee_link_name, None)
            if tcp_link is None:
                tcp_link = getattr(agent, "tcp", None)
        if tcp_link is None:
            return
        raw_pose = _tensor_to_numpy(tcp_link.pose.raw_pose)[0].astype(np.float64)
        marker_position = raw_pose[:3]
        if np.linalg.norm(self.sim_tcp_marker_offset_m) > 0.0:
            marker_rotation = Rotation.from_quat(
                [raw_pose[4], raw_pose[5], raw_pose[6], raw_pose[3]]
            )
            marker_position = marker_position + marker_rotation.apply(
                self.sim_tcp_marker_offset_m
            )
        self.sim_tcp_marker.set_pose(
            sapien.Pose(p=marker_position.tolist(), q=raw_pose[3:7].tolist())
        )

    def close(self) -> None:
        try:
            self.robot.disconnect()
        finally:
            self.env.close()

    def initialize(self) -> None:
        self._reset_debug_log()
        if os.getenv("MANISKILL_TRACE_ACTIONS", "").strip():
            print(f"[SYNC_TRACE] mani_skill.__file__={Path(mani_skill.__file__).resolve()}")
            print(
                "[SYNC_TRACE] mani_skill.envs.sapien_env.__file__="
                f"{Path(mani_skill_sapien_env.__file__).resolve()}"
            )
        self._reset_sim(reconfigure=True)
        self._ensure_sim_tcp_marker()
        self._move_sim_to_home()
        self._install_sim_non_arm_hold_hook()
        record = self.save_snapshot("startup")
        print("Initialized sync session without moving the real robot.")
        print(f"Sim robot TCP at startup: {_format_pose_xyzrpy_rad(self._sim_tcp_pose_base_rad())}")
        print(f"Saved initial simulation snapshot to {record.image_path}")
        print(f"Per-command sync debug log: {self.debug_log_path}")
        print(f"Latest image: {record.latest_image_path}")

    def _reset_debug_log(self) -> None:
        lines = [
            "# sync_rc5_delta_pose_shell comparison log",
            f"sim_backend={self.args.sim_backend}",
            f"sim_ee_link_name={self.args.sim_ee_link_name}",
            f"sim_init_from_real_home_joints={self.args.sim_init_from_real_home_joints}",
            f"real_to_sim_zero_offset_deg={self.args.real_to_sim_zero_offset_deg}",
            f"translation_frame={self.args.translation_frame}",
            f"rotation_frame={self.args.rotation_frame}",
            f"sim_delta_remap_rpy_deg={self.sim_delta_remap_rpy_deg.tolist()}",
            f"sim_tcp_marker_offset_m={self.sim_tcp_marker_offset_m.tolist()}",
            f"sim_state_file={self.args.sim_state_file}",
            f"real_home_pose_file={self.args.real_home_pose_file}",
            f"sim_control_mode={self.env.unwrapped.agent.control_mode}",
            f"sim_apply_ik_qpos_direct={self.args.sim_apply_ik_qpos_direct}",
            "",
        ]
        arm_controller = self._sim_arm_controller()
        if arm_controller is not None:
            lines[-1:-1] = [
                f"sim_controller_frame={getattr(arm_controller.config, 'frame', None)}",
                f"sim_controller_use_delta={getattr(arm_controller.config, 'use_delta', None)}",
                f"sim_controller_use_target={getattr(arm_controller.config, 'use_target', None)}",
            ]
        self.debug_log_path.write_text("\n".join(lines), encoding="utf-8")

    def _remap_delta_for_sim(self, delta_xyzrpy_rad: np.ndarray) -> np.ndarray:
        remapped = np.asarray(delta_xyzrpy_rad, dtype=np.float64).copy()
        remapped[:3] = self.sim_delta_remap.apply(remapped[:3])
        remapped[3:6] = self.sim_delta_remap.apply(remapped[3:6])
        return remapped

    def _reset_sim(self, *, reconfigure: bool) -> None:
        self.obs, self.info = self.env.reset(
            seed=self.args.seed,
            options={
                "reconfigure": reconfigure,
                "load_background": self.args.load_background,
                "use_probe_objects": self.args.use_probe_objects,
                "show_debug_markers": self.args.show_debug_markers,
                "show_spawn_grid": self.args.show_spawn_grid,
                "robot_far_away": False,
                "probe_carrot_pose_p": PROBE_CARROT_POSE_P,
                "probe_carrot_pose_q": PROBE_CARROT_POSE_Q,
                "probe_plate_pose_p": PROBE_PLATE_POSE_P,
                "probe_plate_pose_q": PROBE_PLATE_POSE_Q,
                "robot_base_pose_p": self.sim_base_pose_p.tolist(),
                "robot_base_pose_q": self.sim_base_pose_q.tolist(),
                "robot_init_qpos": self.sim_init_qpos.tolist(),
            },
        )
        self._update_sim_tcp_marker_pose()

    def _sim_tcp_pose_base_rad(self) -> np.ndarray:
        raw_pose = _tensor_to_numpy(self.env.unwrapped.agent.ee_pose_at_robot_base.raw_pose)[0]
        return _raw_pose_to_xyzrpy_rad(raw_pose)

    def _sim_tcp_pose_world_raw(self) -> np.ndarray:
        assert self.obs is not None
        return _tensor_to_numpy(self.obs["extra"]["tcp_pose"])[0]

    def _real_tcp_pose_rad(self) -> np.ndarray:
        return np.asarray(
            self.robot.motion.linear.get_actual_position(orientation_units="rad"),
            dtype=np.float64,
        )

    def _real_joint_pos_rad(self) -> np.ndarray:
        return np.asarray(
            self.robot.motion.joint.get_actual_position(units="rad"),
            dtype=np.float64,
        )

    def _real_inverse_kinematics_rad(
        self,
        tcp_pose_rad: tuple[float, float, float, float, float, float],
    ) -> np.ndarray | None:
        try:
            result = self.robot.motion.kinematics.get_inverse(
                tcp_pose=tcp_pose_rad,
                orientation_units="rad",
            )
        except Exception:
            return None
        if result is None:
            return None
        return np.asarray(result, dtype=np.float64)

    def _sim_qpos(self) -> np.ndarray:
        if self.obs is not None:
            return _tensor_to_numpy(self.obs["agent"]["qpos"])[0].astype(np.float64)
        return _tensor_to_numpy(self.env.unwrapped.agent.robot.get_qpos())[0].astype(np.float64)

    def _append_command_debug_log(
        self,
        *,
        source: str,
        delta_xyzrpy_rad: np.ndarray,
        sim_delta_xyzrpy_rad: np.ndarray,
        real_before_tcp_rad: np.ndarray,
        real_target_tcp_rad: tuple[float, float, float, float, float, float],
        real_after_tcp_rad: np.ndarray,
        real_joint_before_rad: np.ndarray,
        real_joint_target_rad: np.ndarray | None,
        real_joint_after_rad: np.ndarray,
        sim_before_tcp_base_rad: np.ndarray,
        sim_after_tcp_base_rad: np.ndarray,
        sim_qpos_before_rad: np.ndarray,
        sim_qpos_after_rad: np.ndarray,
        sim_controller_prev_pose_raw: np.ndarray | None,
        sim_controller_expected_target_pose_raw: np.ndarray | None,
        sim_controller_recorded_target_pose_raw: np.ndarray | None,
        sim_controller_target_qpos_rad: np.ndarray | None,
    ) -> None:
        real_delta_base = _pose_delta_base_xyzrpy_rad(real_before_tcp_rad, real_after_tcp_rad)
        real_delta_local = _pose_delta_local_xyzrpy_rad(real_before_tcp_rad, real_after_tcp_rad)
        sim_delta_base = _pose_delta_base_xyzrpy_rad(sim_before_tcp_base_rad, sim_after_tcp_base_rad)
        sim_delta_local = _pose_delta_local_xyzrpy_rad(sim_before_tcp_base_rad, sim_after_tcp_base_rad)
        tcp_match_error = sim_after_tcp_base_rad - real_after_tcp_rad
        tcp_match_error[3:6] = _wrap_angle_diff_rad(tcp_match_error[3:6])
        joint_match_error = sim_qpos_after_rad[:6] - real_joint_after_rad[:6]
        joint_match_error = _wrap_angle_diff_rad(joint_match_error)
        real_joint_target_text = "None"
        if real_joint_target_rad is not None:
            real_joint_target_text = _format_array(real_joint_target_rad)
        sim_controller_target_qpos_text = "None"
        if sim_controller_target_qpos_rad is not None:
            sim_controller_target_qpos_text = _format_array(sim_controller_target_qpos_rad)
        sim_controller_prev_pose_text = "None"
        if sim_controller_prev_pose_raw is not None:
            sim_controller_prev_pose_text = _format_array(sim_controller_prev_pose_raw)
        sim_controller_expected_target_pose_text = "None"
        if sim_controller_expected_target_pose_raw is not None:
            sim_controller_expected_target_pose_text = _format_array(sim_controller_expected_target_pose_raw)
        sim_controller_recorded_target_pose_text = "None"
        if sim_controller_recorded_target_pose_raw is not None:
            sim_controller_recorded_target_pose_text = _format_array(sim_controller_recorded_target_pose_raw)

        lines = [
            f"=== command {self.command_index:04d} {source} ===",
            f"delta_command_xyzrpy_rad={_format_array(delta_xyzrpy_rad)}",
            f"sim_delta_command_xyzrpy_rad={_format_array(sim_delta_xyzrpy_rad)}",
            f"real_before_tcp_rad={_format_array(real_before_tcp_rad)}",
            f"real_target_tcp_rad={_format_array(np.asarray(real_target_tcp_rad, dtype=np.float64))}",
            f"real_after_tcp_rad={_format_array(real_after_tcp_rad)}",
            f"real_achieved_delta_base_xyzrpy_rad={_format_array(real_delta_base)}",
            f"real_achieved_delta_local_xyzrpy_rad={_format_array(real_delta_local)}",
            f"real_joint_before_rad={_format_array(real_joint_before_rad)}",
            f"real_joint_target_ik_rad={real_joint_target_text}",
            f"real_joint_after_rad={_format_array(real_joint_after_rad)}",
            f"sim_before_tcp_base_xyzrpy_rad={_format_array(sim_before_tcp_base_rad)}",
            f"sim_after_tcp_base_xyzrpy_rad={_format_array(sim_after_tcp_base_rad)}",
            f"sim_achieved_delta_base_xyzrpy_rad={_format_array(sim_delta_base)}",
            f"sim_achieved_delta_local_xyzrpy_rad={_format_array(sim_delta_local)}",
            f"sim_arm_qpos_before_rad={_format_array(sim_qpos_before_rad[:6])}",
            f"sim_arm_qpos_after_rad={_format_array(sim_qpos_after_rad[:6])}",
            f"tcp_match_error_sim_minus_real_xyzrpy_rad={_format_array(tcp_match_error)}",
            f"tcp_match_error_sim_minus_real_xyz_m={_format_array(tcp_match_error[:3])}",
            f"tcp_match_error_sim_minus_real_rpy_deg={_format_array(np.rad2deg(tcp_match_error[3:6]))}",
            f"joint_match_error_sim_minus_real_rad={_format_array(joint_match_error)}",
            f"joint_match_error_sim_minus_real_deg={_format_array(np.rad2deg(joint_match_error))}",
            f"sim_controller_prev_pose_base_xyzquat={sim_controller_prev_pose_text}",
            f"sim_controller_expected_target_pose_base_xyzquat={sim_controller_expected_target_pose_text}",
            f"sim_controller_recorded_target_pose_base_xyzquat={sim_controller_recorded_target_pose_text}",
            f"sim_controller_target_qpos_rad={sim_controller_target_qpos_text}",
            "",
        ]
        with self.debug_log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.command_index += 1

    def _move_real_to_saved_home(self) -> np.ndarray:
        self.robot.motion.linear.add_new_waypoint(
            tcp_pose=tuple(self.real_home_tcp_rad.tolist()),
            speed=self.args.speed,
            accel=self.args.accel,
            blend=0.0,
            orientation_units="rad",
        )
        self.robot.motion.mode.set("move")
        self.robot.motion.wait_waypoint_completion()
        return self._real_tcp_pose_rad()

    def _move_sim_to_home(self) -> None:
        init_qpos = torch.as_tensor(
            self.sim_init_qpos[None, :],
            dtype=torch.float32,
            device=self.env.unwrapped.device,
        )
        self.env.unwrapped.agent.reset(init_qpos=init_qpos)
        self.env.unwrapped.agent.controller.reset()
        if self.env.unwrapped.gpu_sim_enabled:
            self.env.unwrapped.scene._gpu_apply_all()
            self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            self.env.unwrapped.scene._gpu_fetch_all()
        self._configure_sim_non_arm_joint_holds()
        self.info = self.env.get_info()
        self.obs = self.env.get_obs(self.info)
        self._update_sim_tcp_marker_pose()

    def _configure_sim_non_arm_joint_holds(self) -> None:
        agent = self.env.unwrapped.agent
        active_joints = agent.robot.get_active_joints()
        non_arm_targets: list[float] = []
        non_arm_joints = []
        non_arm_joint_indices: list[int] = []

        for joint_index, joint in enumerate(active_joints):
            if joint.name in getattr(agent, "arm_joint_names", []):
                continue
            if joint_index >= len(self.sim_init_qpos):
                continue
            target = float(self.sim_init_qpos[joint_index])
            non_arm_targets.append(target)
            non_arm_joints.append(joint)
            non_arm_joint_indices.append(joint_index)
            joint.set_drive_properties(200.0, 40.0, force_limit=80.0, mode="force")

        if non_arm_targets:
            self.sim_non_arm_joint_targets = np.asarray(non_arm_targets, dtype=np.float32)[None, :]
            self.sim_non_arm_joints = non_arm_joints
            self.sim_non_arm_joint_indices = torch.as_tensor(
                non_arm_joint_indices,
                dtype=torch.int32,
                device=self.env.unwrapped.device,
            )
            self.env.unwrapped.agent.robot.set_joint_drive_targets(
                self.sim_non_arm_joint_targets,
                joints=self.sim_non_arm_joints,
                joint_indices=self.sim_non_arm_joint_indices,
            )
            self._lock_sim_non_arm_joints_state()
        else:
            self.sim_non_arm_joint_targets = None
            self.sim_non_arm_joints = []
            self.sim_non_arm_joint_indices = None

    def _lock_sim_non_arm_joints_state(self) -> None:
        if self.sim_non_arm_joint_targets is None or self.sim_non_arm_joint_indices is None:
            return

        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos().clone()
        qvel = robot.get_qvel().clone()
        qpos[:, self.sim_non_arm_joint_indices.long()] = torch.as_tensor(
            self.sim_non_arm_joint_targets,
            dtype=qpos.dtype,
            device=qpos.device,
        )
        qvel[:, self.sim_non_arm_joint_indices.long()] = 0.0
        robot.set_qpos(qpos)
        robot.set_qvel(qvel)
        if self.env.unwrapped.gpu_sim_enabled:
            self.env.unwrapped.scene._gpu_apply_all()
            self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            self.env.unwrapped.scene._gpu_fetch_all()
        self._update_sim_tcp_marker_pose()

    def _hold_sim_non_arm_joints(self) -> None:
        if self.sim_non_arm_joint_targets is None or self.sim_non_arm_joint_indices is None:
            return
        self.env.unwrapped.agent.robot.set_joint_drive_targets(
            self.sim_non_arm_joint_targets,
            joints=self.sim_non_arm_joints,
            joint_indices=self.sim_non_arm_joint_indices,
        )
        self._lock_sim_non_arm_joints_state()

    def _install_sim_non_arm_hold_hook(self) -> None:
        env = self.env.unwrapped
        if self._orig_before_simulation_step is not None:
            return

        self._orig_before_simulation_step = env._before_simulation_step

        def _hooked_before_simulation_step():
            self._hold_sim_non_arm_joints()
            return self._orig_before_simulation_step()

        env._before_simulation_step = _hooked_before_simulation_step

    def _step_sim_delta(
        self,
        delta_xyzrpy_rad: np.ndarray,
        *,
        settle_steps: int | None = None,
    ) -> dict:
        if settle_steps is None:
            settle_steps = self.args.sim_settle_steps
        print(f"[SYNC_TRACE] sim_delta_xyzrpy_rad={delta_xyzrpy_rad.tolist()}")

        action = torch.tensor(
            [[
                float(delta_xyzrpy_rad[0]),
                float(delta_xyzrpy_rad[1]),
                float(delta_xyzrpy_rad[2]),
                float(delta_xyzrpy_rad[3]),
                float(delta_xyzrpy_rad[4]),
                float(delta_xyzrpy_rad[5]),
            ]],
            dtype=torch.float32,
            device=self.env.unwrapped.device,
        )
        arm_controller = self._sim_arm_controller()
        prev_pose_raw = None
        expected_target_pose_raw = None
        if arm_controller is not None:
            prev_pose_raw = _pose_raw_to_xyzquat(arm_controller.ee_pose_at_base.raw_pose[0].detach().cpu().numpy())
            expected_target_pose = arm_controller.compute_target_pose(
                arm_controller.ee_pose_at_base,
                action,
            )
            expected_target_pose_raw = _pose_raw_to_xyzquat(
                expected_target_pose.raw_pose[0].detach().cpu().numpy()
            )
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        self._update_sim_tcp_marker_pose()
        if self.args.sim_apply_ik_qpos_direct and arm_controller is not None:
            target_qpos = getattr(arm_controller, "_target_qpos", None)
            self._apply_sim_arm_target_qpos_direct(target_qpos)
        self._hold_sim_non_arm_joints()
        if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
            raise RuntimeError(
                "Simulation episode ended during delta motion. "
                "Increase --max-episode-steps or restart the session."
            )

        if settle_steps <= 0:
            return

        for _ in range(settle_steps):
            self.obs, _, terminated, truncated, self.info = self.env.step(None)
            self._update_sim_tcp_marker_pose()
            if self.args.sim_apply_ik_qpos_direct and arm_controller is not None:
                self._apply_sim_arm_target_qpos_direct(getattr(arm_controller, "_target_qpos", None))
            self._hold_sim_non_arm_joints()
            if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
                raise RuntimeError(
                    "Simulation episode ended while settling after a delta motion. "
                    "Increase --max-episode-steps or restart the session."
                )
        recorded_target_pose_raw = None
        target_qpos_rad = None
        if arm_controller is not None:
            if getattr(arm_controller, "_target_pose", None) is not None:
                recorded_target_pose_raw = _pose_raw_to_xyzquat(
                    arm_controller._target_pose.raw_pose[0].detach().cpu().numpy()
                )
            if getattr(arm_controller, "_target_qpos", None) is not None:
                target_qpos_rad = _tensor_to_numpy(arm_controller._target_qpos)[0].astype(np.float64)
        return dict(
            prev_pose_raw=prev_pose_raw,
            expected_target_pose_raw=expected_target_pose_raw,
            recorded_target_pose_raw=recorded_target_pose_raw,
            target_qpos_rad=target_qpos_rad,
        )

    def move_both_home(self) -> None:
        real_pose = self._move_real_to_saved_home()
        self._move_sim_to_home()
        record = self.save_snapshot("sim_home")
        print(f"Real robot returned to saved home TCP: {real_pose}")
        print(f"Sim robot TCP at home: {_format_pose_xyzrpy_rad(self._sim_tcp_pose_base_rad())}")
        print(f"Latest image: {record.latest_image_path}")

    def execute_delta(self, delta_xyzrpy_rad: np.ndarray, *, source: str = "command") -> SnapshotRecord:
        real_pose_before = self._real_tcp_pose_rad()
        real_joint_before = self._real_joint_pos_rad()
        sim_pose_before = self._sim_tcp_pose_base_rad()
        sim_qpos_before = self._sim_qpos()
        sim_delta_xyzrpy_rad = self._remap_delta_for_sim(delta_xyzrpy_rad)
        delta_rotation = Rotation.from_euler("xyz", delta_xyzrpy_rad[3:6], degrees=False)
        target_real_pose = compose_delta_pose(
            current_tcp_pose_rad=real_pose_before,
            dxyz_m=delta_xyzrpy_rad[:3],
            delta_rotation=delta_rotation,
            translation_frame=self.args.translation_frame,
            rotation_frame=self.args.rotation_frame,
        )
        real_joint_target = self._real_inverse_kinematics_rad(target_real_pose)
        self.robot.motion.linear.add_new_waypoint(
            tcp_pose=target_real_pose,
            speed=self.args.speed,
            accel=self.args.accel,
            blend=0.0,
            orientation_units="rad",
        )
        self.robot.motion.mode.set("move")

        sim_controller_debug = self._step_sim_delta(sim_delta_xyzrpy_rad)
        self.robot.motion.wait_waypoint_completion()
        real_pose_after = self._real_tcp_pose_rad()
        real_joint_after = self._real_joint_pos_rad()
        sim_pose_after = self._sim_tcp_pose_base_rad()
        sim_qpos_after = self._sim_qpos()
        self._append_command_debug_log(
            source=source,
            delta_xyzrpy_rad=delta_xyzrpy_rad,
            sim_delta_xyzrpy_rad=sim_delta_xyzrpy_rad,
            real_before_tcp_rad=real_pose_before,
            real_target_tcp_rad=target_real_pose,
            real_after_tcp_rad=real_pose_after,
            real_joint_before_rad=real_joint_before,
            real_joint_target_rad=real_joint_target,
            real_joint_after_rad=real_joint_after,
            sim_before_tcp_base_rad=sim_pose_before,
            sim_after_tcp_base_rad=sim_pose_after,
            sim_qpos_before_rad=sim_qpos_before,
            sim_qpos_after_rad=sim_qpos_after,
            sim_controller_prev_pose_raw=sim_controller_debug.get("prev_pose_raw"),
            sim_controller_expected_target_pose_raw=sim_controller_debug.get("expected_target_pose_raw"),
            sim_controller_recorded_target_pose_raw=sim_controller_debug.get("recorded_target_pose_raw"),
            sim_controller_target_qpos_rad=sim_controller_debug.get("target_qpos_rad"),
        )
        return self.save_snapshot("delta")

    def save_snapshot(self, label: str) -> SnapshotRecord:
        assert self.obs is not None
        assert self.info is not None

        frame = _extract_obs_frame(self.obs)
        image_path = self.output_dir / f"{self.args.image_prefix}_{self.snapshot_index:04d}_{label}.png"
        latest_path = self.output_dir / f"{self.args.image_prefix}_latest.png"
        meta_path = self.output_dir / f"{self.args.image_prefix}_{self.snapshot_index:04d}_{label}.txt"

        real_pose = self._real_tcp_pose_rad().tolist()
        sim_pose_base = self._sim_tcp_pose_base_rad().tolist()
        sim_pose_world = self._sim_tcp_pose_world_raw().tolist()
        sim_qpos = _tensor_to_numpy(self.obs["agent"]["qpos"])[0].tolist()
        info_summary = {
            key: (_tensor_to_numpy(value)[0].tolist() if np.asarray(_tensor_to_numpy(value)[0]).ndim > 0 else _tensor_to_numpy(value)[0].item())
            for key, value in self.info.items()
        }

        iio.imwrite(image_path, frame)
        iio.imwrite(latest_path, frame)
        meta_path.write_text(
            "\n".join(
                [
                    f"snapshot_index={self.snapshot_index}",
                    f"label={label}",
                    f"sim_base_pose_p={self.sim_base_pose_p.tolist()}",
                    f"sim_base_pose_q={self.sim_base_pose_q.tolist()}",
                    f"real_tcp_pose_rad={real_pose}",
                    f"sim_tcp_pose_base_xyzrpy_rad={sim_pose_base}",
                    f"sim_tcp_pose_world_raw={sim_pose_world}",
                    f"sim_qpos={sim_qpos}",
                    f"info={info_summary}",
                ]
            ),
            encoding="utf-8",
        )

        self.snapshot_index += 1
        return SnapshotRecord(image_path=image_path, latest_image_path=latest_path, meta_path=meta_path)

    def print_status(self) -> None:
        print(f"Real TCP: {_format_pose_xyzrpy_rad(self._real_tcp_pose_rad())}")
        print(f"Sim  TCP: {_format_pose_xyzrpy_rad(self._sim_tcp_pose_base_rad())}")
        print(f"Sim base pose: {self.sim_base_pose_p.tolist()}")

    def teleop(self) -> None:
        pos_step = self.args.teleop_pos_step
        rot_step = np.deg2rad(self.args.teleop_rot_step_deg)

        if self.args.teleop_home_first:
            self.move_both_home()

        print("Teleop: W/S=+/-Y, A/D=+/-X, Q/E=+/-Z")
        print("Teleop: I/K pitch, J/L yaw, U/O roll, X exit")

        while True:
            key = _read_single_key().lower()
            if key == "x":
                print("\nLeaving teleop mode")
                return

            delta = np.zeros(6, dtype=np.float64)
            if key == "w":
                delta[1] += pos_step
            elif key == "s":
                delta[1] -= pos_step
            elif key == "a":
                delta[0] -= pos_step
            elif key == "d":
                delta[0] += pos_step
            elif key == "q":
                delta[2] += pos_step
            elif key == "e":
                delta[2] -= pos_step
            elif key == "i":
                delta[4] += rot_step
            elif key == "k":
                delta[4] -= rot_step
            elif key == "j":
                delta[5] += rot_step
            elif key == "l":
                delta[5] -= rot_step
            elif key == "u":
                delta[3] += rot_step
            elif key == "o":
                delta[3] -= rot_step
            else:
                continue

            record = self.execute_delta(delta, source=f"teleop:{key}")
            print(f"\nkey={key} delta={_format_pose_xyzrpy_rad(delta)}")
            print(f"Saved {record.image_path}")
            print(f"Latest image: {record.latest_image_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive shell that moves the real RC5 and the ManiSkill RC5 together "
            "with delta-pose commands, then saves a simulation image after each move."
        )
    )
    parser.add_argument("--robot-ip", default="10.10.10.10", help="Robot IPv4 address")
    parser.add_argument("--speed", type=float, default=0.3, help="Linear speed in m/s")
    parser.add_argument("--accel", type=float, default=0.15, help="Linear acceleration in m/s^2")
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.3,
        help="Global robot velocity scale in [0, 1]",
    )
    parser.add_argument(
        "--acceleration-scale",
        type=float,
        default=0.1,
        help="Global robot acceleration scale in [0, 1]",
    )
    parser.add_argument("--payload-mass", type=float, default=3.5, help="Payload mass in kg")
    parser.add_argument(
        "--translation-frame",
        choices=("base", "tcp"),
        default="base",
        help="Frame in which real-robot translational deltas are interpreted",
    )
    parser.add_argument(
        "--rotation-frame",
        choices=("base", "tcp"),
        default="base",
        help="Frame in which real-robot rotational deltas are interpreted",
    )
    parser.add_argument(
        "--real-home-pose-file",
        default=DEFAULT_REAL_HOME_POSE_FILE,
        help="Text file containing the manually confirmed real home pose; current_tcp_rad is used for homing",
    )
    parser.add_argument(
        "--sim-state-file",
        default=DEFAULT_SIM_STATE_FILE,
        help="Calibration snapshot txt file that defines the sim debug base pose, base quaternion, and qpos",
    )
    parser.add_argument("--output-dir", default=str(RC5_SYNC_SHELL_DIR), help="Directory for saved sim images")
    parser.add_argument("--image-prefix", default="sync_rc5", help="Prefix for saved sim images")
    parser.add_argument("--seed", type=int, default=0, help="Simulation seed")
    parser.add_argument(
        "--sim-backend",
        choices=("gpu", "cpu"),
        default="cpu",
        help="ManiSkill simulation backend; cpu is the safer default for single-env RC5 sync debugging",
    )
    parser.add_argument("--shader", default="default", help="Shader pack for sensors")
    parser.add_argument(
        "--observation-camera-mode",
        default="preset_center_far_back_up",
        help="Camera preset from openreal2sim_validation.observation_camera_override",
    )
    parser.add_argument("--max-episode-steps", type=int, default=100000, help="Large horizon to avoid frequent sim resets")
    parser.add_argument("--load-background", action="store_true", default=True, help="Load the OpenReal2Sim background mesh")
    parser.add_argument("--no-load-background", dest="load_background", action="store_false")
    parser.add_argument("--use-probe-objects", action="store_true", default=False, help="Spawn the probe carrot and plate")
    parser.add_argument("--no-probe-objects", dest="use_probe_objects", action="store_false")
    parser.add_argument("--show-debug-markers", action="store_true", default=False, help="Show debug markers in the scene")
    parser.add_argument("--show-spawn-grid", action="store_true", default=False, help="Show the calibrated spawn grid")
    parser.add_argument(
        "--sim-delta-remap-rpy-deg",
        default="0,0,90",
        help=(
            "Fixed XYZ Euler rotation in degrees applied to the requested delta before it is sent to sim. "
            "This remaps both translational and rotational delta axes, for example '0,0,90'."
        ),
    )
    parser.add_argument(
        "--sim-ee-link-name",
        default="prehand",
        help=(
            "Link name used by the ManiSkill RC5 EE pose controller and debug TCP pose. "
            "Useful for testing whether the control mismatch comes from the chosen sim EE frame."
        ),
    )
    parser.add_argument(
        "--sim-tcp-marker-offset-m",
        default="0,0,0",
        help=(
            "Local XYZ offset in meters for the red TCP marker relative to the active sim EE link. "
            "This affects only visualization, not control."
        ),
    )
    parser.add_argument(
        "--sim-init-from-real-home-joints",
        action="store_true",
        default=False,
        help=(
            "Initialize the first 6 sim arm joints from current_joint_rad in --real-home-pose-file, "
            "converted by the configured real-to-sim zero-angle offset."
        ),
    )
    parser.add_argument(
        "--real-to-sim-zero-offset-deg",
        default=",".join(f"{v:.6g}" for v in RC5_REAL_TO_SIM_ZERO_OFFSET_DEG.tolist()),
        help=(
            "Six comma-separated per-joint offsets in degrees for converting real arm joints into "
            "sim arm joints: q_sim ~= wrap(q_real + offset)."
        ),
    )
    parser.add_argument(
        "--debug-log-name",
        default=DEFAULT_DEBUG_LOG_NAME,
        help="Per-command sync debug log filename inside --output-dir; overwritten at startup",
    )
    parser.add_argument(
        "--trace-actions",
        action="store_true",
        default=False,
        help="Enable lower-level ManiSkill/controller tracing and print it with the SYNC_TRACE prefix",
    )
    parser.add_argument("--no-trace-actions", dest="trace_actions", action="store_false")
    parser.add_argument(
        "--sim-settle-steps",
        type=int,
        default=4,
        help="Number of zero-delta sim steps after each commanded move before saving an image",
    )
    parser.add_argument(
        "--sim-apply-ik-qpos-direct",
        action="store_true",
        default=False,
        help=(
            "Diagnostic mode: after sim IK computes a target arm qpos, write it directly into the "
            "sim articulation instead of relying on PD arm joint drives."
        ),
    )
    parser.add_argument(
        "--no-sim-apply-ik-qpos-direct",
        dest="sim_apply_ik_qpos_direct",
        action="store_false",
    )
    parser.add_argument(
        "--teleop-pos-step",
        type=float,
        default=0.002,
        help="Per-key translational TCP delta in meters for teleop mode",
    )
    parser.add_argument(
        "--teleop-rot-step-deg",
        type=float,
        default=5.0,
        help="Per-key rotational TCP delta in degrees for teleop mode",
    )
    parser.add_argument(
        "--teleop-home-first",
        action="store_true",
        default=True,
        help="Move both robots to their configured home states before entering teleop",
    )
    parser.add_argument("--no-teleop-home-first", dest="teleop_home_first", action="store_false")
    parser.add_argument(
        "--start-in-teleop",
        action="store_true",
        default=True,
        help="Enter key-driven teleop immediately after startup instead of waiting for the shell command",
    )
    parser.add_argument("--no-start-in-teleop", dest="start_in_teleop", action="store_false")
    return parser


def _normalize_argparse_argv(argv: list[str]) -> list[str]:
    """Allow comma-separated values that begin with '-' to be passed as:
    --arg -1,2,3
    instead of requiring:
    --arg=-1,2,3
    """
    options_requiring_value = {
        "--sim-delta-remap-rpy-deg",
        "--real-to-sim-zero-offset-deg",
        "--sim-tcp-marker-offset-m",
    }
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in options_requiring_value and i + 1 < len(argv):
            normalized.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def _print_help() -> None:
    print("Commands:")
    print("  move dx dy dz [droll dpitch dyaw]   delta pose in meters/radians")
    print("  move_deg dx dy dz droll dpitch dyaw delta pose in meters/degrees")
    print("  teleop                              enter key-driven TCP delta mode")
    print("  home                                move both robots to the saved real/sim home states")
    print("  save [label]                        save the current sim image")
    print("  show                                print current real/sim TCP poses")
    print("  help                                show this message")
    print("  quit                                exit")


def _parse_delta_command(parts: list[str], *, degrees: bool) -> np.ndarray:
    if len(parts) not in {4, 7}:
        raise ValueError("expected 3 translation values and optionally 3 rotation values")
    delta = np.zeros(6, dtype=np.float64)
    delta[:3] = [float(parts[1]), float(parts[2]), float(parts[3])]
    if len(parts) == 7:
        delta[3:6] = [float(parts[4]), float(parts[5]), float(parts[6])]
    if degrees:
        delta[3:6] = np.deg2rad(delta[3:6])
    return delta


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argparse_argv(sys.argv[1:]))
    if args.trace_actions:
        os.environ.setdefault("MANISKILL_TRACE_ACTIONS", "1")
        os.environ.setdefault("MANISKILL_TRACE_PREFIX", "SYNC_TRACE")
        os.environ.setdefault("MANISKILL_TRACE_MAX_ELEMS", "8")
        os.environ.setdefault(
            "MANISKILL_TRACE_TAGS",
            ",".join(
                [
                    "SapienEnv._step_action.input",
                    "BaseAgent.set_action",
                    "PDEEPoseController.set_action",
                    "Kinematics.compute_ik.output",
                    "SapienEnv._step_action.post_dynamics",
                ]
            ),
        )

    session = SyncRC5Session(args)
    try:
        try:
            session.initialize()
        except Exception as exc:
            print(f"Initialization failed: {exc}")
            return 1
        _print_help()

        if args.start_in_teleop:
            session.teleop()

        while True:
            try:
                raw = input("sync-rc5> ").strip()
            except EOFError:
                print("\nInput stream closed. Exiting sync shell.")
                break
            if not raw:
                continue
            parts = shlex.split(raw)
            cmd = parts[0].lower()

            try:
                if cmd in {"quit", "exit", "q"}:
                    break
                if cmd == "help":
                    _print_help()
                    continue
                if cmd == "show":
                    session.print_status()
                    continue
                if cmd == "teleop":
                    session.teleop()
                    continue
                if cmd == "home":
                    session.move_both_home()
                    continue
                if cmd == "save":
                    label = parts[1] if len(parts) > 1 else "manual"
                    record = session.save_snapshot(label)
                    print(f"Saved {record.image_path}")
                    print(f"Latest image: {record.latest_image_path}")
                    continue
                if cmd == "move":
                    delta = _parse_delta_command(parts, degrees=False)
                    record = session.execute_delta(delta, source="shell:move")
                    print(f"Applied delta: {_format_pose_xyzrpy_rad(delta)}")
                    print(f"Saved {record.image_path}")
                    print(f"Latest image: {record.latest_image_path}")
                    continue
                if cmd == "move_deg":
                    delta = _parse_delta_command(parts, degrees=True)
                    record = session.execute_delta(delta, source="shell:move_deg")
                    print(f"Applied delta: {_format_pose_xyzrpy_rad(delta)}")
                    print(f"Saved {record.image_path}")
                    print(f"Latest image: {record.latest_image_path}")
                    continue

                print("Unknown command. Type 'help'.")
            except (IndexError, ValueError) as exc:
                print(f"Bad command: {exc}")
            except RuntimeError as exc:
                print(str(exc))
                print("Use 'save' if you want the latest frame before restarting.")
                break
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
