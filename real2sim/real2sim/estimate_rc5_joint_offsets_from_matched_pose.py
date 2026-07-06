#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
from real2sim.debug_paths import RC5_SYNC_STARTUP_HOME_FILE


DEFAULT_REAL_HOME_POSE_FILE = "real2sim/real_home_pose.txt"
DEFAULT_SIM_SNAPSHOT_FILE = str(RC5_SYNC_STARTUP_HOME_FILE)


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return ((values + np.pi) % (2.0 * np.pi)) - np.pi


def _load_real_joint_rad(path: str) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("current_joint_rad="):
            joints = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float64)
            if joints.shape[0] < 6:
                raise ValueError(f"Expected at least 6 joints in current_joint_rad, got {joints.shape}")
            return joints[:6]
    raise ValueError(f"Did not find current_joint_rad in {path}")


def _load_sim_joint_rad(path: str) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("sim_qpos="):
            qpos = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float64)
            if qpos.shape[0] < 6:
                raise ValueError(f"Expected at least 6 joints in sim_qpos, got {qpos.shape}")
            return qpos[:6]
        if line.startswith("robot_init_qpos="):
            qpos = np.array(ast.literal_eval(line.split("=", 1)[1]), dtype=np.float64)
            if qpos.shape[0] < 6:
                raise ValueError(f"Expected at least 6 joints in robot_init_qpos, got {qpos.shape}")
            return qpos[:6]
    raise ValueError(f"Did not find sim_qpos or robot_init_qpos in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-joint real-to-sim offsets from one visually matched real/sim arm pose. "
            "The result is a first-pass guess: offset ~= wrap(q_sim - q_real)."
        )
    )
    parser.add_argument("--real-home-pose-file", default=DEFAULT_REAL_HOME_POSE_FILE)
    parser.add_argument("--sim-snapshot-file", default=DEFAULT_SIM_SNAPSHOT_FILE)
    args = parser.parse_args()

    real_joint = _load_real_joint_rad(args.real_home_pose_file)
    sim_joint = _load_sim_joint_rad(args.sim_snapshot_file)
    offset = _wrap_to_pi(sim_joint - real_joint)

    print(f"real_home_pose_file={args.real_home_pose_file}")
    print(f"sim_snapshot_file={args.sim_snapshot_file}")
    print(f"real_joint_rad={tuple(real_joint.tolist())}")
    print(f"sim_joint_rad={tuple(sim_joint.tolist())}")
    print(f"estimated_real_to_sim_offset_rad={tuple(offset.tolist())}")
    print(f"estimated_real_to_sim_offset_deg={tuple(np.rad2deg(offset).tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
