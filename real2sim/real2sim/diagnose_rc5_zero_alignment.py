#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
import sys
from typing import Iterable

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from mani_skill.envs.sapien_env import BaseEnv
from real2sim.debug_paths import RC5_POSE_DEBUG_STATE_FILE, RC5_SYNC_COMPARE_LOG_FILE
from real2sim.openreal2sim_validation import (
    ROBOT_BASE_POSE_P,
    ROBOT_BASE_POSE_Q,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_PYTHON_API_ROOT = REPO_ROOT / "real2sim" / "vendor" / "python_api"
if VENDORED_PYTHON_API_ROOT.is_dir():
    vendored_python_api_root_str = str(VENDORED_PYTHON_API_ROOT)
    if vendored_python_api_root_str not in sys.path:
        sys.path.insert(0, vendored_python_api_root_str)
DEFAULT_SIM_STATE_FILE = str(RC5_POSE_DEBUG_STATE_FILE)
DEFAULT_REAL_HOME_FILE = "real2sim/real_home_pose.txt"
DEFAULT_COMPARE_LOG_FILE = str(RC5_SYNC_COMPARE_LOG_FILE)
URDF_LINK_NAMES = ("body6", "prehand", "right_base_link")


def _format_pose_xyzrpy(pose_xyzrpy: np.ndarray) -> str:
    xyz = ", ".join(f"{v:.6f}" for v in pose_xyzrpy[:3])
    rpy = ", ".join(f"{v:.3f}" for v in np.rad2deg(pose_xyzrpy[3:6]))
    return f"xyz[m]=[{xyz}] rpy[deg]=[{rpy}]"


def _pose_to_xyzrpy(pose) -> np.ndarray:
    quat = np.asarray(pose.q, dtype=np.float64)
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return np.concatenate([np.asarray(pose.p, dtype=np.float64), rpy])


def _array_from_literal(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=np.float64)


def _xyzquat_to_xyzrpy(raw_pose: np.ndarray) -> np.ndarray:
    raw_pose = np.asarray(raw_pose, dtype=np.float64).reshape(-1)
    quat_wxyz = raw_pose[3:7]
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    return np.concatenate([raw_pose[:3], rpy])


def _parse_joint_values(raw: str | None, *, degrees: bool) -> np.ndarray | None:
    if raw is None or not raw.strip():
        return None
    parts = [float(part.strip()) for part in raw.split(",")]
    arr = np.asarray(parts, dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError("expected exactly 6 comma-separated joint values")
    if degrees:
        arr = np.deg2rad(arr)
    return arr


def _read_key_from_file(path: str, key: str) -> np.ndarray | None:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith(f"{key}="):
            return _array_from_literal(line.split("=", 1)[1])
    return None


def _wrap_to_nearest(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(values, dtype=np.float64).copy()
    for i in range(len(wrapped)):
        diff = wrapped[i] - reference[i]
        wrapped[i] = reference[i] + ((diff + np.pi) % (2.0 * np.pi)) - np.pi
    return wrapped


def _dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _vendor_rc5_fk_flange(q_rad: np.ndarray) -> np.ndarray:
    from API.source.models.classes.data_classes.dh_models import DhParamsManager

    params = DhParamsManager.get("rc5")
    tf = np.eye(4, dtype=np.float64)
    for i in range(6):
        theta = float(params.theta[i]) + float(params.offset[i]) + float(q_rad[i])
        tf = tf @ _dh_transform(
            float(params.alpha[i]),
            float(params.a[i]),
            float(params.d[i]),
            theta,
        )
    return tf


def _matrix_to_xyzrpy(tf: np.ndarray) -> np.ndarray:
    rpy = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([tf[:3, 3], rpy])


def _build_env() -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="sensor_data",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
        robot_uids="rc5_aero_hand_right_openreal2sim_validation",
    )


def _compute_urdf_link_poses(env: BaseEnv, q_rad: np.ndarray) -> dict[str, np.ndarray]:
    agent = env.unwrapped.agent
    full_qpos = np.zeros((1, agent.robot.max_dof), dtype=np.float64)
    full_qpos[0, :6] = q_rad
    pin_model = agent.robot.create_pinocchio_model()
    pin_model.compute_forward_kinematics(full_qpos[0])

    poses: dict[str, np.ndarray] = {}
    for link_name in URDF_LINK_NAMES:
        link = agent.robot.links_map[link_name]
        pose = pin_model.get_link_pose(link.index)
        poses[link_name] = _pose_to_xyzrpy(pose)
    return poses


def _load_named_joint_sets(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    joint_sets: list[tuple[str, np.ndarray]] = []

    if args.joint_rad:
        joint_sets.append(("cli_joint_rad", _parse_joint_values(args.joint_rad, degrees=False)))
    if args.joint_deg:
        joint_sets.append(("cli_joint_deg", _parse_joint_values(args.joint_deg, degrees=True)))

    real_home_joint = _read_key_from_file(args.real_home_file, "current_joint_rad")
    if real_home_joint is not None and real_home_joint.shape[0] >= 6:
        joint_sets.append(("manual_real_home_file", real_home_joint[:6]))

    sim_state_joint = _read_key_from_file(args.sim_state_file, "robot_init_qpos")
    if sim_state_joint is not None and sim_state_joint.shape[0] >= 6:
        joint_sets.append(("sim_state_file_raw", sim_state_joint[:6]))
        joint_sets.append(("sim_state_file_wrapped", _wrap_to_nearest(np.zeros(6), sim_state_joint[:6])))

    if args.robot_ip:
        from API.rc_api import RobotApi

        robot = RobotApi(args.robot_ip, timeout=args.timeout, show_std_traceback=True)
        try:
            joint_sets.append(
                ("robot_current_joint", np.asarray(robot.motion.joint.get_actual_position(units="rad"), dtype=np.float64))
            )
            joint_sets.append(
                ("robot_controller_home", np.asarray(robot.motion.get_home_pose(units="rad"), dtype=np.float64))
            )
        finally:
            robot.disconnect()

    deduped: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, values in joint_sets:
        key = tuple(np.round(values, 9).tolist())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, values))
    return deduped


def _print_joint_deltas(joint_sets: list[tuple[str, np.ndarray]]) -> None:
    print("Joint-space comparisons (nearest wrapped delta in rad / deg):")
    for i, (name_a, q_a) in enumerate(joint_sets):
        for name_b, q_b in joint_sets[i + 1 :]:
            q_b_near = _wrap_to_nearest(q_a, q_b)
            delta = q_b_near - q_a
            delta_deg = np.rad2deg(delta)
            print(f"  {name_a} -> {name_b}")
            print(f"    delta_rad={np.array2string(delta, precision=6, separator=', ')}")
            print(f"    delta_deg={np.array2string(delta_deg, precision=3, separator=', ')}")


def _extract_bracket_value(block: str, key: str) -> np.ndarray | None:
    pattern = rf"{re.escape(key)}=\[(.*?)\]"
    match = re.search(pattern, block, flags=re.DOTALL)
    if match is None:
        return None
    payload = "[" + " ".join(match.group(1).split()) + "]"
    return np.fromstring(payload.strip("[]"), sep=",", dtype=np.float64)


def _load_compare_log_commands(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    sections = text.split("=== command ")
    commands: list[dict] = []
    for raw_section in sections[1:]:
        lines = raw_section.splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        block = raw_section
        commands.append(
            dict(
                header=header,
                sim_controller_target_qpos_rad=_extract_bracket_value(
                    block, "sim_controller_target_qpos_rad"
                ),
                sim_arm_qpos_after_rad=_extract_bracket_value(
                    block, "sim_arm_qpos_after_rad"
                ),
                sim_controller_expected_target_pose_base_xyzquat=_extract_bracket_value(
                    block, "sim_controller_expected_target_pose_base_xyzquat"
                ),
                sim_after_tcp_base_xyzrpy_rad=_extract_bracket_value(
                    block, "sim_after_tcp_base_xyzrpy_rad"
                ),
                real_target_tcp_rad=_extract_bracket_value(block, "real_target_tcp_rad"),
            )
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose RC5 joint-zero alignment by comparing vendor DH FK and "
            "the sim URDF FK for the same 6 joint values."
        )
    )
    parser.add_argument("--robot-ip", default="", help="Optional real robot IP for querying current/home joints")
    parser.add_argument("--timeout", type=int, default=5, help="Robot connection timeout in seconds")
    parser.add_argument("--joint-rad", default="", help="Optional explicit 6-joint vector in radians")
    parser.add_argument("--joint-deg", default="", help="Optional explicit 6-joint vector in degrees")
    parser.add_argument("--sim-state-file", default=DEFAULT_SIM_STATE_FILE, help="Calibration/state txt file to inspect")
    parser.add_argument("--real-home-file", default=DEFAULT_REAL_HOME_FILE, help="Saved manual real-home txt file")
    parser.add_argument(
        "--compare-log-file",
        default="",
        help="Optional sync compare log to inspect exact sim target qpos entries",
    )
    args = parser.parse_args()

    joint_sets = _load_named_joint_sets(args)
    if not joint_sets:
        raise SystemExit("No joint sets available. Provide --joint-rad/--joint-deg or valid repo files.")

    print(f"Reference robot base pose for env reset: p={ROBOT_BASE_POSE_P} q={ROBOT_BASE_POSE_Q}")
    _print_joint_deltas(joint_sets)
    print("")

    env = _build_env()
    try:
        env.reset(
            seed=0,
            options={
                "reconfigure": True,
                "load_background": False,
                "use_probe_objects": False,
                "show_debug_markers": False,
                "robot_far_away": False,
                "robot_base_pose_p": list(ROBOT_BASE_POSE_P),
                "robot_base_pose_q": list(ROBOT_BASE_POSE_Q),
                "robot_init_qpos": [0.0] * 22,
            },
        )

        for name, q_rad in joint_sets:
            print(f"=== {name} ===")
            print(f"joint_rad={np.array2string(q_rad, precision=6, separator=', ')}")
            print(f"joint_deg={np.array2string(np.rad2deg(q_rad), precision=3, separator=', ')}")

            vendor_tf = _vendor_rc5_fk_flange(q_rad)
            print(f"vendor_dh_flange: {_format_pose_xyzrpy(_matrix_to_xyzrpy(vendor_tf))}")

            urdf_poses = _compute_urdf_link_poses(env, q_rad)
            for link_name in URDF_LINK_NAMES:
                print(f"urdf_{link_name}: {_format_pose_xyzrpy(urdf_poses[link_name])}")
            print("")

        compare_log_file = args.compare_log_file.strip()
        if compare_log_file:
            print(f"=== compare_log_inspection: {compare_log_file} ===")
            commands = _load_compare_log_commands(compare_log_file)
            if not commands:
                print("No command blocks found in compare log.")
            for command in commands:
                print(f"-- {command['header']} --")
                target_qpos = command["sim_controller_target_qpos_rad"]
                if target_qpos is None or target_qpos.shape != (6,):
                    print("missing sim_controller_target_qpos_rad")
                    print("")
                    continue
                print(
                    f"sim_controller_target_qpos_rad={np.array2string(target_qpos, precision=6, separator=', ')}"
                )
                print(
                    f"sim_controller_target_qpos_deg={np.array2string(np.rad2deg(target_qpos), precision=3, separator=', ')}"
                )
                vendor_tf = _vendor_rc5_fk_flange(target_qpos)
                print(f"vendor_dh_flange: {_format_pose_xyzrpy(_matrix_to_xyzrpy(vendor_tf))}")
                urdf_poses = _compute_urdf_link_poses(env, target_qpos)
                for link_name in URDF_LINK_NAMES:
                    print(f"urdf_{link_name}: {_format_pose_xyzrpy(urdf_poses[link_name])}")
                raw_target_pose = command["sim_controller_expected_target_pose_base_xyzquat"]
                if raw_target_pose is not None and raw_target_pose.shape == (7,):
                    target_xyzrpy = _xyzquat_to_xyzrpy(raw_target_pose)
                    print(f"log_target_pose: {_format_pose_xyzrpy(target_xyzrpy)}")
                    for link_name in URDF_LINK_NAMES:
                        delta = urdf_poses[link_name][:3] - target_xyzrpy[:3]
                        print(
                            f"{link_name}_minus_log_target_xyz_m="
                            f"{np.array2string(delta, precision=6, separator=', ')}"
                        )
                achieved_pose = command["sim_after_tcp_base_xyzrpy_rad"]
                if achieved_pose is not None and achieved_pose.shape == (6,):
                    print(f"log_achieved_pose: {_format_pose_xyzrpy(achieved_pose)}")
                real_target_pose = command["real_target_tcp_rad"]
                if real_target_pose is not None and real_target_pose.shape == (6,):
                    print(f"real_target_pose: {_format_pose_xyzrpy(real_target_pose)}")
                print("")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
