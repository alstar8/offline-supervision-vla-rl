import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import torch

from mani_skill.utils.visualization.misc import images_to_video
from real2sim.calibrate_rc5_pose import (
    Args as CalibrateArgs,
    _apply_sim_arm_target_qpos_direct,
    _build_env,
    _extract_obs_frame,
    _load_state_file,
    _remap_sim_delta,
    _scene_options,
)
from real2sim.debug_paths import RC5_POSE_LIVE_LATEST_STATE_FILE, VIDEOS_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record one or more RC5 scripted delta-motion videos using the same "
            "simulation defaults as calibrate_rc5_pose.py."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(VIDEOS_DIR),
        help="Directory where mp4 files are written.",
    )
    parser.add_argument(
        "--video-prefix",
        default="openreal2sim_rc5_delta_sequence",
        help="Prefix used when naming output videos.",
    )
    parser.add_argument(
        "--sim-backend",
        nargs="+",
        choices=("cpu", "gpu"),
        default=["cpu"],
        help="One or more ManiSkill backends to render.",
    )
    parser.add_argument(
        "--sim-apply-ik-qpos-direct-mode",
        nargs="+",
        choices=("on", "off"),
        default=["on"],
        help="One or more IK-direct modes to render.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Video FPS.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Simulation seed passed through to the calibration environment.",
    )
    parser.add_argument(
        "--teleop-pos-step",
        type=float,
        default=0.02,
        help="Position delta in meters for one scripted translation step.",
    )
    parser.add_argument(
        "--teleop-rot-step-rad",
        type=float,
        default=0.12,
        help="Rotation delta in radians for one scripted rotation step.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Optional zero-delta steps inserted after each scripted motion segment.",
    )
    parser.add_argument(
        "--motion-substeps",
        type=int,
        default=4,
        help="How many smaller env steps to use for each scripted teleop-sized motion.",
    )
    parser.add_argument(
        "--gripper-transition-steps",
        type=int,
        default=4,
        help="Number of env steps used to visually settle a grab or release command.",
    )
    parser.add_argument(
        "--grab-wait-steps",
        type=int,
        default=8,
        help="Additional zero-delta closed-gripper steps after grabbing.",
    )
    parser.add_argument(
        "--release-wait-steps",
        type=int,
        default=4,
        help="Additional zero-delta open-gripper steps after releasing.",
    )
    parser.add_argument(
        "--observation-camera-mode",
        default="manual_best",
        help="Camera preset forwarded to calibrate_rc5_pose._build_env.",
    )
    parser.add_argument("--shader", default="default", help="Shader pack for sensors.")
    parser.add_argument(
        "--load-state-file",
        default=str(RC5_POSE_LIVE_LATEST_STATE_FILE),
        help="Calibration state file forwarded to the base calibration setup.",
    )
    parser.add_argument(
        "--use-probe-objects",
        action="store_true",
        default=True,
        help="Spawn probe carrot and plate, matching the requested calibrate invocation.",
    )
    parser.add_argument("--no-use-probe-objects", dest="use_probe_objects", action="store_false")
    parser.add_argument(
        "--clean-scene",
        action="store_true",
        default=False,
        help="Use a clean scene instead of the requested no-clean-scene default.",
    )
    parser.add_argument("--no-clean-scene", dest="clean_scene", action="store_false")
    return parser.parse_args()


def _base_calibrate_args(args: argparse.Namespace) -> CalibrateArgs:
    return CalibrateArgs(
        output_dir=args.output_dir,
        image_prefix=args.video_prefix,
        keep_history=False,
        sim_backend="cpu",
        shader=args.shader,
        observation_camera_mode=args.observation_camera_mode,
        seed=args.seed,
        load_background=True,
        clean_scene=args.clean_scene,
        use_probe_objects=args.use_probe_objects,
        show_debug_markers=False,
        robot_far_away=False,
        base_dx=0.0,
        base_dy=0.0,
        base_dz=0.0,
        load_state_file=args.load_state_file,
        sim_apply_ik_qpos_direct=True,
    )


def _scripted_segments(
    pos_step: float,
    rot_step_rad: float,
) -> list[tuple[str, int, np.ndarray, float | None]]:
    down = np.zeros(6, dtype=np.float32)
    down[2] = -pos_step

    left = np.zeros(6, dtype=np.float32)
    left[0] = -pos_step

    yaw_left = np.zeros(6, dtype=np.float32)
    yaw_left[5] = rot_step_rad

    yaw_right = np.zeros(6, dtype=np.float32)
    yaw_right[5] = -rot_step_rad

    return [
        ("down", 4, down, None),
        ("left", 5, left, None),
        ("yaw_left", 3, yaw_left, None),
        ("yaw_right", 6, yaw_right, None),
        ("yaw_left_return", 3, yaw_left, None),
        ("grab", 1, np.zeros(6, dtype=np.float32), 1.0),
        ("release", 1, np.zeros(6, dtype=np.float32), -1.0),
    ]


def _step_env(
    env,
    obs: dict,
    gripper_cmd: float,
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    *,
    apply_ik_qpos_direct: bool,
) -> tuple[dict, dict]:
    mapped_delta_pos, mapped_delta_rot = _remap_sim_delta(delta_pos.tolist(), delta_rot.tolist())
    action = torch.tensor(
        [[
            float(mapped_delta_pos[0]),
            float(mapped_delta_pos[1]),
            float(mapped_delta_pos[2]),
            float(mapped_delta_rot[0]),
            float(mapped_delta_rot[1]),
            float(mapped_delta_rot[2]),
            float(gripper_cmd),
        ]],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    obs, _, terminated, truncated, info = env.step(action)
    if apply_ik_qpos_direct:
        arm_controller = env.unwrapped.agent.controller
        if hasattr(arm_controller, "controllers"):
            arm_controller = arm_controller.controllers.get("arm")
        target_qpos = None if arm_controller is None else getattr(arm_controller, "_target_qpos", None)
        obs, info = _apply_sim_arm_target_qpos_direct(env, target_qpos)
    if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
        raise RuntimeError(
            "Simulation episode ended during scripted video capture. "
            "Increase the episode horizon in calibrate_rc5_pose.py if needed."
        )
    return obs, info


def _append_hold_frames(
    env,
    obs: dict,
    info: dict,
    frames: list[np.ndarray],
    *,
    gripper_cmd: float,
    hold_steps: int,
    apply_ik_qpos_direct: bool,
) -> tuple[dict, dict]:
    if hold_steps <= 0:
        return obs, info
    for _ in range(hold_steps):
        obs, info = _step_env(
            env,
            obs,
            gripper_cmd,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            apply_ik_qpos_direct=apply_ik_qpos_direct,
        )
        frames.append(_extract_obs_frame(obs))
    return obs, info


def _video_name(prefix: str, sim_backend: str, sim_apply_ik_qpos_direct: bool) -> str:
    ik_mode = "ik_direct_on" if sim_apply_ik_qpos_direct else "ik_direct_off"
    return f"{prefix}_{sim_backend}_{ik_mode}"


def _record_single_video(base_args: CalibrateArgs, cli_args: argparse.Namespace) -> Path:
    env = _build_env(base_args, camera_mode=base_args.observation_camera_mode)
    try:
        robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(base_args.load_state_file)
        robot_base_pose_p = robot_base_pose_p + np.array(
            [base_args.base_dx, base_args.base_dy, base_args.base_dz],
            dtype=np.float32,
        )
        obs, info = env.reset(
            seed=base_args.seed,
            options={
                "reconfigure": True,
                **_scene_options(base_args),
                "robot_base_pose_p": robot_base_pose_p.tolist(),
                "robot_base_pose_q": robot_base_pose_q.tolist(),
                "robot_init_qpos": robot_init_qpos.tolist(),
            },
        )

        frames = [_extract_obs_frame(obs)]
        gripper_cmd = -1.0
        for label, repeat_count, delta, gripper_override in _scripted_segments(
            pos_step=cli_args.teleop_pos_step,
            rot_step_rad=cli_args.teleop_rot_step_rad,
        ):
            if gripper_override is not None:
                gripper_cmd = gripper_override
            if label in {"grab", "release"}:
                hold_steps = cli_args.gripper_transition_steps
                if label == "grab":
                    hold_steps += cli_args.grab_wait_steps
                else:
                    hold_steps += cli_args.release_wait_steps
                obs, info = _append_hold_frames(
                    env,
                obs,
                info,
                frames,
                gripper_cmd=gripper_cmd,
                hold_steps=hold_steps,
                apply_ik_qpos_direct=base_args.sim_apply_ik_qpos_direct,
            )
                continue

            substeps = max(1, cli_args.motion_substeps)
            delta_per_step = delta / float(substeps)
            total_steps = repeat_count * substeps
            for _ in range(total_steps):
                obs, info = _step_env(
                    env,
                    obs,
                    gripper_cmd,
                    delta_per_step[:3],
                    delta_per_step[3:],
                    apply_ik_qpos_direct=base_args.sim_apply_ik_qpos_direct,
                )
                frames.append(_extract_obs_frame(obs))

            obs, info = _append_hold_frames(
                env,
                obs,
                info,
                frames,
                gripper_cmd=gripper_cmd,
                hold_steps=cli_args.settle_steps,
                apply_ik_qpos_direct=base_args.sim_apply_ik_qpos_direct,
            )

        output_dir = Path(cli_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = _video_name(
            cli_args.video_prefix,
            base_args.sim_backend,
            base_args.sim_apply_ik_qpos_direct,
        )
        images_to_video(frames, str(output_dir), video_name, fps=cli_args.fps, verbose=False)
        return output_dir / f"{video_name}.mp4"
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    base_args = _base_calibrate_args(args)
    saved_videos: list[Path] = []

    for sim_backend, ik_mode in product(args.sim_backend, args.sim_apply_ik_qpos_direct_mode):
        run_args = replace(
            base_args,
            sim_backend=sim_backend,
            sim_apply_ik_qpos_direct=(ik_mode == "on"),
        )
        video_path = _record_single_video(run_args, args)
        saved_videos.append(video_path)
        print(video_path.resolve())

    if saved_videos:
        print("Saved videos:")
        for path in saved_videos:
            print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
