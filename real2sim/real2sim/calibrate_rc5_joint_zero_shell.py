#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MANISKILL_ROOT = REPO_ROOT / "ManiSkill"
if LOCAL_MANISKILL_ROOT.is_dir():
    local_maniskill_root_str = str(LOCAL_MANISKILL_ROOT)
    if local_maniskill_root_str not in sys.path:
        sys.path.insert(0, local_maniskill_root_str)

import imageio.v3 as iio
import numpy as np
import torch

from real2sim.sync_rc5_delta_pose_shell import (
    DEFAULT_REAL_HOME_POSE_FILE,
    DEFAULT_SIM_STATE_FILE,
    SyncRC5Session,
    _extract_obs_frame,
    _format_array,
    _normalize_argparse_argv,
    _read_single_key,
)
from real2sim.debug_paths import RC5_JOINT_ZERO_SHELL_DIR


DEFAULT_OUTPUT_DIR = str(RC5_JOINT_ZERO_SHELL_DIR)
DEFAULT_IMAGE_PREFIX = "rc5_joint_zero"
DEFAULT_DEBUG_LOG_NAME = "rc5_joint_zero.log"


def _deg(rad: float) -> float:
    return float(np.rad2deg(rad))


def _joint_label(joint_index: int) -> str:
    return f"joint{joint_index}"


class JointZeroCalibrationSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sync = SyncRC5Session(args)
        if self.sync.real_home_joint_rad is None:
            raise ValueError(
                f"{args.real_home_pose_file!r} must contain current_joint_rad for joint-zero calibration."
            )

        self.joint_index = args.joint_index
        self.joint_name = _joint_label(args.joint_index)
        self.joint_step_rad = float(np.deg2rad(args.joint_step_deg))
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_path = self.output_dir / args.debug_log_name
        self.snapshot_index = 0

        self.real_home_joint_rad = self.sync.real_home_joint_rad.copy()
        self.sim_home_qpos = self.sync.sim_init_qpos.copy()
        self.real_anchor_joint_target = self.real_home_joint_rad.copy()
        self.real_anchor_joint_target[self.joint_index] = self._resolve_anchor_rad(
            explicit_both_deg=args.joint_anchor_deg,
            explicit_one_deg=args.real_joint_anchor_deg,
            default_rad=0.0,
        )
        self.sim_anchor_joint_target = self.sim_home_qpos.copy()
        self.sim_anchor_joint_target[self.joint_index] = self._resolve_anchor_rad(
            explicit_both_deg=args.joint_anchor_deg,
            explicit_one_deg=args.sim_joint_anchor_deg,
            default_rad=0.0,
        )

    @staticmethod
    def _resolve_anchor_rad(
        *,
        explicit_both_deg: float | None,
        explicit_one_deg: float | None,
        default_rad: float,
    ) -> float:
        if explicit_one_deg is not None:
            return float(np.deg2rad(explicit_one_deg))
        if explicit_both_deg is not None:
            return float(np.deg2rad(explicit_both_deg))
        return default_rad

    def close(self) -> None:
        self.sync.close()

    def initialize(self) -> None:
        self._reset_debug_log()
        self.sync._reset_sim(reconfigure=True)
        self.sync._install_sim_non_arm_hold_hook()
        self._move_real_joints(self.real_anchor_joint_target)
        self._apply_sim_qpos(self.sim_anchor_joint_target)
        record = self.save_snapshot("start")
        self._append_log("initialize")
        print(
            f"Initialized joint calibration for {self.joint_name}. "
            "Both robots were moved from the matched-home pose to the configured anchor."
        )
        print(f"Saved initial simulation snapshot to {record}")
        self.print_status()

    def _reset_debug_log(self) -> None:
        lines = [
            "# rc5 joint-zero calibration log",
            f"joint_index={self.joint_index}",
            f"joint_name={self.joint_name}",
            f"joint_step_deg={self.args.joint_step_deg}",
            f"real_home_pose_file={self.args.real_home_pose_file}",
            f"sim_state_file={self.args.sim_state_file}",
            f"sim_ee_link_name={self.args.sim_ee_link_name}",
            f"real_anchor_joint_target_rad={_format_array(self.real_anchor_joint_target)}",
            f"sim_anchor_joint_target_rad={_format_array(self.sim_anchor_joint_target[:6])}",
            "",
        ]
        self.debug_log_path.write_text("\n".join(lines), encoding="utf-8")

    def _move_real_joints(self, target_joint_rad: np.ndarray) -> None:
        self.sync.robot.motion.joint.add_new_waypoint(
            angle_pose=tuple(target_joint_rad.tolist()),
            speed=self.args.joint_speed,
            accel=self.args.joint_accel,
            blend=0.0,
            units="rad",
        )
        self.sync.robot.motion.mode.set("move")
        self.sync.robot.motion.wait_waypoint_completion()

    def _apply_sim_qpos(self, qpos: np.ndarray) -> None:
        robot = self.sync.env.unwrapped.agent.robot
        device = self.sync.env.unwrapped.device
        qpos_tensor = torch.as_tensor(qpos[None, :], dtype=torch.float32, device=device)
        qvel_tensor = torch.zeros_like(robot.get_qvel())
        qf_tensor = torch.zeros_like(robot.get_qf())
        robot.set_qpos(qpos_tensor)
        robot.set_qvel(qvel_tensor)
        robot.set_qf(qf_tensor)
        if self.sync.env.unwrapped.gpu_sim_enabled:
            self.sync.env.unwrapped.scene._gpu_apply_all()
            self.sync.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            self.sync.env.unwrapped.scene._gpu_fetch_all()
        self.sync._hold_sim_non_arm_joints()
        self.sync.info = self.sync.env.get_info()
        self.sync.obs = self.sync.env.get_obs(self.sync.info)

    def _current_real_joints(self) -> np.ndarray:
        return self.sync._real_joint_pos_rad()

    def _current_sim_qpos(self) -> np.ndarray:
        return self.sync._sim_qpos()

    def _append_log(self, source: str) -> None:
        real_joints = self._current_real_joints()
        sim_qpos = self._current_sim_qpos()
        lines = [
            f"=== {source} snapshot {self.snapshot_index:04d} ===",
            f"real_joint_rad={_format_array(real_joints)}",
            f"sim_arm_qpos_rad={_format_array(sim_qpos[:6])}",
            f"selected_joint_real_rad={real_joints[self.joint_index]:.9f}",
            f"selected_joint_real_deg={_deg(real_joints[self.joint_index]):.6f}",
            f"selected_joint_sim_rad={sim_qpos[self.joint_index]:.9f}",
            f"selected_joint_sim_deg={_deg(sim_qpos[self.joint_index]):.6f}",
            f"selected_joint_sim_minus_real_rad={sim_qpos[self.joint_index] - real_joints[self.joint_index]:.9f}",
            f"selected_joint_sim_minus_real_deg={_deg(sim_qpos[self.joint_index] - real_joints[self.joint_index]):.6f}",
            "",
        ]
        with self.debug_log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def save_snapshot(self, label: str) -> Path:
        assert self.sync.obs is not None
        frame = _extract_obs_frame(self.sync.obs)
        image_path = self.output_dir / f"{self.args.image_prefix}_{self.snapshot_index:04d}_{label}.png"
        latest_path = self.output_dir / f"{self.args.image_prefix}_latest.png"
        iio.imwrite(image_path, frame)
        iio.imwrite(latest_path, frame)
        self.snapshot_index += 1
        return image_path

    def print_status(self) -> None:
        real_joints = self._current_real_joints()
        sim_qpos = self._current_sim_qpos()
        real_value = float(real_joints[self.joint_index])
        sim_value = float(sim_qpos[self.joint_index])
        print(
            f"{self.joint_name}: "
            f"real={real_value:.6f} rad ({_deg(real_value):.3f} deg), "
            f"sim={sim_value:.6f} rad ({_deg(sim_value):.3f} deg)"
        )
        print(
            "Dialed offset from current sim anchor: "
            f"{sim_value:+.6f} rad ({_deg(sim_value):+.3f} deg)"
        )
        print(
            "Current sim-real delta on selected joint: "
            f"{sim_value - real_value:+.6f} rad ({_deg(sim_value - real_value):+.3f} deg)"
        )

    def reset_to_joint_anchor(self) -> None:
        self._move_real_joints(self.real_anchor_joint_target)
        self._apply_sim_qpos(self.sim_anchor_joint_target)
        self._append_log("reset_joint_anchor")
        self.print_status()

    def restore_matched_home(self) -> None:
        self._move_real_joints(self.real_home_joint_rad)
        self._apply_sim_qpos(self.sim_home_qpos)
        self._append_log("restore_home")
        self.print_status()

    def step_sim_joint(self, delta_rad: float) -> None:
        qpos = self._current_sim_qpos()
        qpos[self.joint_index] += delta_rad
        self._apply_sim_qpos(qpos)
        self._append_log("step_sim_joint")
        self.print_status()

    def step_real_joint(self, delta_rad: float) -> None:
        target = self._current_real_joints()
        target[self.joint_index] += delta_rad
        self._move_real_joints(target)
        self._append_log("step_real_joint")
        self.print_status()

    def teleop(self) -> None:
        print(f"Interactive calibration for {self.joint_name}")
        print("Keys:")
        print("  a/d  sim joint -/+ one step")
        print("  j/l  real joint -/+ one step")
        print("  r    reset both robots to the selected-joint anchor")
        print("  h    restore both robots to the matched-home pose")
        print("  s    save a simulation snapshot")
        print("  p    print current selected-joint values")
        print("  x    exit")

        while True:
            key = _read_single_key().lower()
            if key == "x":
                print("\nLeaving joint-zero calibration shell")
                return
            if key == "a":
                self.step_sim_joint(-self.joint_step_rad)
                continue
            if key == "d":
                self.step_sim_joint(self.joint_step_rad)
                continue
            if key == "j":
                self.step_real_joint(-self.joint_step_rad)
                continue
            if key == "l":
                self.step_real_joint(self.joint_step_rad)
                continue
            if key == "r":
                self.reset_to_joint_anchor()
                continue
            if key == "h":
                self.restore_matched_home()
                continue
            if key == "p":
                self.print_status()
                continue
            if key == "s":
                image_path = self.save_snapshot("manual")
                self._append_log("save_snapshot")
                print(f"\nSaved {image_path}")
                continue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive RC5 joint calibration shell. It starts from the same matched-home "
            "state as the sync shell, moves one selected joint to a chosen anchor on both the "
            "real robot and sim, then lets you nudge that joint while reading out the dialed offset."
        )
    )
    parser.add_argument("--robot-ip", default="10.10.10.10", help="Robot IPv4 address")
    parser.add_argument("--speed", type=float, default=0.3, help="Unused linear speed kept for SyncRC5Session compatibility")
    parser.add_argument("--accel", type=float, default=0.15, help="Unused linear accel kept for SyncRC5Session compatibility")
    parser.add_argument("--joint-speed", type=float, default=0.5, help="Real-robot joint speed in rad/s for calibration moves")
    parser.add_argument("--joint-accel", type=float, default=1.0, help="Real-robot joint accel in rad/s^2 for calibration moves")
    parser.add_argument("--velocity-scale", type=float, default=0.3, help="Global robot velocity scale in [0, 1]")
    parser.add_argument("--acceleration-scale", type=float, default=0.1, help="Global robot acceleration scale in [0, 1]")
    parser.add_argument("--payload-mass", type=float, default=3.5, help="Payload mass in kg")
    parser.add_argument("--translation-frame", choices=("base", "tcp"), default="base")
    parser.add_argument("--rotation-frame", choices=("base", "tcp"), default="base")
    parser.add_argument("--real-home-pose-file", default=DEFAULT_REAL_HOME_POSE_FILE)
    parser.add_argument("--sim-state-file", default=DEFAULT_SIM_STATE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="cpu")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--observation-camera-mode", default="preset_center_far_back_up")
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--load-background", action="store_true", default=True)
    parser.add_argument("--no-load-background", dest="load_background", action="store_false")
    parser.add_argument("--use-probe-objects", action="store_true", default=False)
    parser.add_argument("--no-probe-objects", dest="use_probe_objects", action="store_false")
    parser.add_argument("--show-debug-markers", action="store_true", default=False)
    parser.add_argument("--show-spawn-grid", action="store_true", default=False)
    parser.add_argument("--sim-delta-remap-rpy-deg", default="0,0,0")
    parser.add_argument("--sim-ee-link-name", default="prehand")
    parser.add_argument("--sim-init-from-real-home-joints", action="store_true", default=False)
    parser.add_argument("--real-to-sim-zero-offset-deg", default="0,0,0,0,0,0")
    parser.add_argument("--debug-log-name", default=DEFAULT_DEBUG_LOG_NAME)
    parser.add_argument("--trace-actions", action="store_true", default=False)
    parser.add_argument("--no-trace-actions", dest="trace_actions", action="store_false")
    parser.add_argument("--sim-settle-steps", type=int, default=0)
    parser.add_argument("--teleop-pos-step", type=float, default=0.002)
    parser.add_argument("--teleop-rot-step-deg", type=float, default=5.0)
    parser.add_argument("--teleop-home-first", action="store_true", default=False)
    parser.add_argument("--no-teleop-home-first", dest="teleop_home_first", action="store_false")
    parser.add_argument("--start-in-teleop", action="store_true", default=True)
    parser.add_argument("--no-start-in-teleop", dest="start_in_teleop", action="store_false")
    parser.add_argument("--joint-index", type=int, required=True, choices=range(6), help="Which RC5 arm joint to calibrate")
    parser.add_argument("--joint-step-deg", type=float, default=1.0, help="Interactive per-key joint increment in degrees")
    parser.add_argument(
        "--joint-anchor-deg",
        type=float,
        default=None,
        help=(
            "Common selected-joint anchor angle in degrees applied to both real and sim before "
            "interactive tuning. If omitted, the selected joint is anchored at 0 deg."
        ),
    )
    parser.add_argument(
        "--real-joint-anchor-deg",
        type=float,
        default=None,
        help="Selected-joint anchor angle in degrees for the real robot only.",
    )
    parser.add_argument(
        "--sim-joint-anchor-deg",
        type=float,
        default=None,
        help="Selected-joint anchor angle in degrees for the sim robot only.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argparse_argv(sys.argv[1:]))
    if args.trace_actions:
        os.environ.setdefault("MANISKILL_TRACE_ACTIONS", "1")
        os.environ.setdefault("MANISKILL_TRACE_PREFIX", "SYNC_TRACE")

    session = JointZeroCalibrationSession(args)
    try:
        session.initialize()
        if args.start_in_teleop:
            session.teleop()
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
