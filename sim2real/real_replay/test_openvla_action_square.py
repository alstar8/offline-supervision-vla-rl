#!/usr/bin/env python3
"""Replay a scripted OpenVLA-format square in sim and/or on the real RC5.

The sequence is four 10 cm legs in the scene-camera plane, expressed as the
same 7D EE deltas OpenVLA emits (xyz meters, rpy radians, gripper):

    LEFT  → image left  (world -X, converted into the sim robot-base)
    UP    → image up    (world +Z)
    RIGHT → image right (world +X)
    DOWN  → image down  (world -Z)

Sim applies those deltas as training does. Real applies the eval remap
(default Rz(-90°)) and the same waypoint path as eval_openvla_real.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_openvla_real as real_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM2REAL_ROOT = REPO_ROOT / "sim2real"
DEFAULT_OUTPUT_ROOT = SIM2REAL_ROOT / "runs/rl/pick_red_cube_sft_databc/action_square"
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
FONT_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")

# Scene-camera left/up expressed in the sim robot-base (OpenVLA) frame.
# Robot base yaw in config_debug.yaml is +46.8° about world Z; 3rd_view / ZED
# image-right is world +X and image-up is world +Z.
CAMERA_LEFT_IN_SIM_BASE = np.array([-0.684541, 0.728974, 0.0], dtype=np.float64)
CAMERA_UP_IN_SIM_BASE = np.array([0.0, 0.0, 1.0], dtype=np.float64)
LEG_ORDER = ("LEFT", "UP", "RIGHT", "DOWN")
OPENVLA_GRIPPER_OPEN = 1.0


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("direction vector is zero")
    return vector / norm


def _leg_direction(name: str) -> np.ndarray:
    if name == "LEFT":
        return _unit(CAMERA_LEFT_IN_SIM_BASE)
    if name == "UP":
        return _unit(CAMERA_UP_IN_SIM_BASE)
    if name == "RIGHT":
        return -_unit(CAMERA_LEFT_IN_SIM_BASE)
    if name == "DOWN":
        return -_unit(CAMERA_UP_IN_SIM_BASE)
    raise ValueError(f"unknown leg {name!r}")


def _openvla_action(xyz: np.ndarray, gripper: float = OPENVLA_GRIPPER_OPEN) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.asarray(xyz, dtype=np.float32)
    action[6] = float(gripper)
    return action


def build_square_actions(
    *,
    leg_m: float,
    step_m: float,
    pause_steps: int,
    gripper: float = OPENVLA_GRIPPER_OPEN,
) -> list[dict]:
    if leg_m <= 0.0 or step_m <= 0.0:
        raise ValueError("leg_m and step_m must be positive")
    n_motion = max(1, int(round(float(leg_m) / float(step_m))))
    step_xyz = float(leg_m) / float(n_motion)
    hold = _openvla_action(np.zeros(3), gripper)
    records: list[dict] = []

    def _append(leg: str, action: np.ndarray, *, moving: bool, index: int, count: int) -> None:
        records.append(
            {
                "leg": leg,
                "moving": moving,
                "index_in_leg": index,
                "count_in_leg": count,
                "action": action.copy(),
            }
        )

    for _ in range(max(0, pause_steps)):
        _append("HOLD", hold, moving=False, index=0, count=pause_steps)
    for leg in LEG_ORDER:
        delta = _leg_direction(leg) * step_xyz
        move = _openvla_action(delta, gripper)
        for i in range(n_motion):
            _append(leg, move, moving=True, index=i + 1, count=n_motion)
        for i in range(max(0, pause_steps)):
            _append(leg, hold, moving=False, index=i + 1, count=pause_steps)
    return records


def _parse_remap(text: str) -> tuple[float, float, float]:
    parts = [float(part) for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"--action-remap-rpy-deg must be three numbers, got {text!r}")
    return parts[0], parts[1], parts[2]


def _load_font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def _overlay_frame(
    frame: np.ndarray,
    *,
    target: str,
    record: dict,
    applied: np.ndarray,
    tcp: list[float] | None,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _load_font(28)
    body_font = _load_font(18)
    action = np.asarray(record["action"], dtype=np.float32)
    phase = "MOVE" if record["moving"] else "PAUSE"
    title = f"{target}  {record['leg']}  {phase}  {record['index_in_leg']}/{record['count_in_leg']}"
    lines = [
        title,
        f"openvla dx,dy,dz  {action[0]:+.4f} {action[1]:+.4f} {action[2]:+.4f}",
        f"applied dx,dy,dz  {applied[0]:+.4f} {applied[1]:+.4f} {applied[2]:+.4f}",
    ]
    if tcp is not None and len(tcp) >= 3:
        lines.append(f"tcp xyz  {tcp[0]:+.4f} {tcp[1]:+.4f} {tcp[2]:+.4f}")
    padding = 8
    widths = [draw.textlength(line, font=title_font if i == 0 else body_font) for i, line in enumerate(lines)]
    box_w = int(max(widths) + 2 * padding)
    box_h = 28 + 20 * (len(lines) - 1) + 2 * padding
    draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=(0, 0, 0, 170))
    y = 12
    for i, line in enumerate(lines):
        font = title_font if i == 0 else body_font
        draw.text((16, y), line, font=font, fill=(255, 255, 80) if i == 0 else (255, 255, 255))
        y += 28 if i == 0 else 20
    return np.asarray(image, dtype=np.uint8)


def _write_sequence_json(path: Path, records: list[dict], extra: dict) -> None:
    payload = {
        **extra,
        "legs": list(LEG_ORDER),
        "camera_left_in_sim_base": CAMERA_LEFT_IN_SIM_BASE.tolist(),
        "camera_up_in_sim_base": CAMERA_UP_IN_SIM_BASE.tolist(),
        "steps": [
            {
                "step": index,
                "leg": record["leg"],
                "moving": record["moving"],
                "index_in_leg": record["index_in_leg"],
                "count_in_leg": record["count_in_leg"],
                "openvla_action": real_eval._action_to_dict(record["action"]),
            }
            for index, record in enumerate(records)
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _gym_action(openvla_action: np.ndarray) -> np.ndarray:
    action = np.asarray(openvla_action, dtype=np.float32).copy()
    action[6] = 1.0 if float(action[6]) > 0.5 else -1.0
    return action


def _to_hwc_uint8(value) -> np.ndarray:
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] > 3:
        array = array[..., :3]
    return np.clip(array, 0, 255).astype(np.uint8)


def _sim_scene_wrist(obs: dict) -> tuple[np.ndarray, np.ndarray | None]:
    sensors = obs["sensor_data"]
    if "3rd_view_camera" in sensors:
        scene = _to_hwc_uint8(sensors["3rd_view_camera"]["rgb"])
    elif "base_camera" in sensors:
        scene = _to_hwc_uint8(sensors["base_camera"]["rgb"])
    else:
        raise KeyError(f"no scene camera in {list(sensors.keys())}")
    wrist = None
    if "wrist_camera" in sensors and "rgb" in sensors["wrist_camera"]:
        wrist = _to_hwc_uint8(sensors["wrist_camera"]["rgb"])
        wrist = real_eval._resize_nearest_rgb(
            wrist, real_eval.WRIST_INSET_HEIGHT, real_eval.WRIST_INSET_WIDTH
        )
    return scene, wrist


def _compose_sim_frame(obs: dict) -> np.ndarray:
    scene, wrist = _sim_scene_wrist(obs)
    if wrist is None:
        return scene
    border = real_eval.WRIST_INSET_BORDER
    inset_h = real_eval.WRIST_INSET_HEIGHT
    inset_w = real_eval.WRIST_INSET_WIDTH
    margin = real_eval.WRIST_INSET_MARGIN
    top = scene.shape[0] - inset_h - 2 * border - margin
    left = scene.shape[1] - inset_w - 2 * border - margin
    out = scene.copy()
    out[top:top + inset_h + 2 * border, left:left + inset_w + 2 * border, :] = 0
    out[top + border:top + border + inset_h, left + border:left + border + inset_w, :] = wrist
    return out


def _sim_tcp_base_world(env) -> tuple[list[float], list[float]]:
    agent = env.unwrapped.agent
    world = agent.tcp.pose.p
    if hasattr(world, "detach"):
        world = world.detach().cpu().numpy()
    world_xyz = np.asarray(world, dtype=np.float64).reshape(-1)[:3]
    try:
        arm = agent.controller.controllers["arm"]
        base = arm.ee_pose_at_base.p
        if hasattr(base, "detach"):
            base = base.detach().cpu().numpy()
        base_xyz = np.asarray(base, dtype=np.float64).reshape(-1)[:3]
    except Exception:
        base_xyz = world_xyz
    return [float(v) for v in base_xyz], [float(v) for v in world_xyz]


def _prepare_sim_sys_path() -> None:
    preferred = [
        REPO_ROOT / "SimplerEnv",
        REPO_ROOT / "ManiSkill",
        REPO_ROOT / "real2sim",
        REPO_ROOT / "openvla",
    ]
    for path in reversed(preferred):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    sim2real = str(SIM2REAL_ROOT)
    if sim2real in sys.path:
        sys.path.remove(sim2real)
    sys.path.append(sim2real)


def _run_sim(args: argparse.Namespace, records: list[dict], output_dir: Path, remap_rpy_deg: tuple[float, float, float]) -> None:
    # Sapien/CUDA must not load when driving the real robot.
    _prepare_sim_sys_path()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    vulkan_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if vulkan_icd.is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", str(vulkan_icd))

    import gymnasium as gym
    import torch
    import openreal2sim.simulation.maniskill  # noqa: F401
    from openreal2sim.simulation.maniskill.rl_gym import build_openreal2sim_rl_gym_kwargs
    from real2sim.openreal2sim_validation import DEFAULT_REAL2SIM_CONTROL_FREQ, DEFAULT_REAL2SIM_SIM_FREQ

    env_kwargs = build_openreal2sim_rl_gym_kwargs(use_wrist_camera=not args.no_wrist)
    control_freq = int(round(float(args.hz))) if args.hz else DEFAULT_REAL2SIM_CONTROL_FREQ
    env = gym.make(
        "OpenReal2Sim-v0",
        **env_kwargs,
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        enable_shadow=True,
        sim_config={"sim_freq": DEFAULT_REAL2SIM_SIM_FREQ, "control_freq": control_freq},
        max_episode_steps=max(200, len(records) + 20),
        sensor_configs={"shader_pack": "default"},
    )
    obs, info = env.reset(seed=0)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frames: list[np.ndarray] = []
    start_base, start_world = _sim_tcp_base_world(env)
    print(f"  sim tcp_base={start_base}")
    print(f"  sim tcp_world={start_world}")

    try:
        for step, record in enumerate(records):
            if real_eval._STOP:
                break
            action = np.asarray(record["action"], dtype=np.float32)
            gym_action = _gym_action(action)
            action_t = torch.as_tensor(gym_action, device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
            obs, _reward, _terminated, _truncated, info = env.step(action_t)
            tcp_base, tcp_world = _sim_tcp_base_world(env)
            frame = _overlay_frame(
                _compose_sim_frame(obs),
                target="SIM",
                record=record,
                applied=action,
                tcp=tcp_world,
            )
            frames.append(frame)
            Image.fromarray(frame).save(frames_dir / f"step_{step:04d}.png")
            (frames_dir / f"step_{step:04d}.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "leg": record["leg"],
                        "moving": record["moving"],
                        "openvla_action": real_eval._action_to_dict(action),
                        "gym_action": real_eval._action_to_dict(gym_action),
                        "tcp_base": [round(v, 6) for v in tcp_base],
                        "tcp_world": [round(v, 6) for v in tcp_world],
                    },
                    indent=2,
                )
            )
            print(
                f"  step {step:04d} {record['leg']:<5} "
                f"openvla={[round(float(v), 4) for v in action[:3]]} "
                f"world={[round(v, 4) for v in tcp_world]}"
            )
        end_base, end_world = _sim_tcp_base_world(env)
        summary = {
            "status": "stopped" if real_eval._STOP else "complete",
            "target": "sim",
            "executed_steps": len(frames),
            "start_tcp_base": start_base,
            "start_tcp_world": start_world,
            "end_tcp_base": end_base,
            "end_tcp_world": end_world,
            "net_world_m": [round(end_world[i] - start_world[i], 6) for i in range(3)],
            "action_remap_rpy_deg": list(remap_rpy_deg),
        }
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  net world delta m={summary['net_world_m']}")
    finally:
        real_eval._save_mp4(frames, output_dir / "rollout.mp4", fps=max(1, int(round(float(args.hz)))))
        env.close()


def _send_abs_tcp_waypoint(robot, target: list[float], workspace: dict[str, tuple[float, float]]) -> tuple[list[float], bool]:
    clipped, changed = real_eval._clip_pose(list(target), workspace)
    tcp = tuple(clipped[:3] + clipped[3:6])
    try:
        if not robot.motion.is_point_reachable(tcp_pose=tcp, orientation_units="deg"):
            print(f"  [skip] unreachable xyz={[round(v, 4) for v in clipped[:3]]}")
            return clipped, changed
        robot.motion.linear.add_new_waypoint(
            tcp,
            speed=real_eval.WP_SPEED,
            accel=real_eval.WP_ACCEL,
            blend=real_eval.WP_BLEND,
            orientation_units="deg",
        )
        real_eval._start_move_if_needed(robot)
    except Exception as exc:
        if type(exc).__name__ != "AddWaypointError" and "waypoint" not in str(exc).lower():
            raise
        print(f"  [skip] waypoint rejected: {exc}")
    return clipped, changed


def _run_real(args: argparse.Namespace, records: list[dict], output_dir: Path, remap_rpy_deg: tuple[float, float, float]) -> None:
    workspace = {"x": real_eval.WORKSPACE_X, "y": real_eval.WORKSPACE_Y, "z": real_eval.WORKSPACE_Z}
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    robot = None
    hand = None
    zed_ctx = None
    rs_pipeline = None
    frames: list[np.ndarray] = []
    step = 0
    try:
        print("Connecting to RC5...")
        robot = real_eval._init_rc5(args.robot_ip)
        print(f"  RC5 OK tcp={robot.motion.linear.get_actual_position(orientation_units='deg')}")
        print("Connecting to AeroHand...")
        from aero_open_sdk.aero_hand import AeroHand

        hand = AeroHand()
        print("Starting ZED2...")
        zed_ctx = real_eval._init_zed2(args.scene_json)
        if not args.no_wrist:
            print("Starting RealSense D405...")
            rs_pipeline = real_eval._init_realsense()
        print("Warming up cameras...")
        for _ in range(20):
            real_eval._grab_openvla_frame(zed_ctx, rs_pipeline)
        if args.home_mode == "sim":
            print("Moving to recorded home pose...")
            real_eval._move_home(robot)
        elif args.home_mode == "current":
            print("Keeping current arm pose as episode start.")
        else:
            raise ValueError(f"unsupported home mode: {args.home_mode}")
        hand.set_joint_positions(real_eval.HAND_HOLD)
        if not args.no_confirm:
            if sys.stdin.isatty():
                input("\nClear the workspace, then press ENTER to start the square...")
            else:
                wait_s = 15
                print(
                    f"\nNon-interactive stdin: starting in {wait_s}s. "
                    "The arm will draw LEFT, UP, RIGHT, DOWN (10 cm each). Ctrl+C aborts."
                )
                time.sleep(wait_s)

        pose = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
        summary = {
            "status": "running",
            "target": "real",
            "action_remap_rpy_deg": list(remap_rpy_deg),
            "home_mode": args.home_mode,
            "baseline_tcp": [round(float(v), 6) for v in pose],
        }
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Running {len(records)} OpenVLA steps at {args.hz} Hz. Ctrl+C = E-stop.\n")
        print("  Real waypoints accumulate on a target pose, matching sim use_target=True.")
        step_interval = 1.0 / float(args.hz)
        target_pose = list(pose)
        prev_moving = False
        while step < len(records) and not real_eval._STOP:
            t0 = time.monotonic()
            record = records[step]
            action = np.asarray(record["action"], dtype=np.float32)
            remapped = real_eval._remap_sim_action_to_real(action, remap_rpy_deg)
            applied, safety = real_eval._limit_action_step(
                remapped,
                action_scale=1.0,
                max_translation_step_m=args.max_translation_step,
                max_rotation_step_rad=args.max_rotation_step,
            )
            if prev_moving and not record["moving"]:
                try:
                    robot.motion.wait_waypoint_completion(5)
                except Exception:
                    pass
            if record["moving"]:
                target_pose[0] += float(applied[0])
                target_pose[1] += float(applied[1])
                target_pose[2] += float(applied[2])
                target_pose[3] += float(np.degrees(applied[3]))
                target_pose[4] += float(np.degrees(applied[4]))
                target_pose[5] += float(np.degrees(applied[5]))
                target_pose, clipped = _send_abs_tcp_waypoint(robot, target_pose, workspace)
                if clipped:
                    print(f"  [clip] target xyz={[round(v, 4) for v in target_pose[:3]]}")
            prev_moving = bool(record["moving"])
            tcp = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
            image = real_eval._grab_openvla_frame(zed_ctx, rs_pipeline)
            frame = _overlay_frame(
                np.asarray(image.convert("RGB")),
                target="REAL",
                record=record,
                applied=applied,
                tcp=tcp,
            )
            frames.append(frame)
            Image.fromarray(frame).save(frames_dir / f"step_{step:04d}.png")
            (frames_dir / f"step_{step:04d}.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "leg": record["leg"],
                        "moving": record["moving"],
                        "current_tcp": [round(float(v), 6) for v in tcp],
                        "target_tcp": [round(float(v), 6) for v in target_pose],
                        "openvla_action": real_eval._action_to_dict(action),
                        "remapped_action": real_eval._action_to_dict(remapped),
                        "applied_action": real_eval._action_to_dict(applied),
                        "action_safety": safety,
                    },
                    indent=2,
                )
            )
            print(
                f"  step {step:04d} {record['leg']:<5} "
                f"openvla={[round(float(v), 4) for v in action[:3]]} "
                f"real={[round(float(v), 4) for v in remapped[:3]]} "
                f"tcp={[round(float(v), 4) for v in tcp[:3]]}"
            )
            step += 1
            sleep_t = step_interval - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
        if prev_moving:
            try:
                robot.motion.wait_waypoint_completion(5)
            except Exception:
                pass
        final_tcp = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
        summary.update(
            {
                "status": "stopped" if real_eval._STOP else "complete",
                "executed_steps": int(step),
                "final_tcp": [round(float(v), 6) for v in final_tcp],
                "net_tcp_m": [round(float(final_tcp[i]) - float(pose[i]), 6) for i in range(3)],
            }
        )
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  net real tcp delta m={summary['net_tcp_m']}")
    finally:
        real_eval._save_mp4(frames, output_dir / "rollout.mp4", fps=max(1, int(round(float(args.hz)))))
        print("\nHolding RC5, opening hand.")
        if robot is not None:
            try:
                robot.motion.mode.set("hold")
            except Exception:
                pass
        if hand is not None:
            try:
                hand.set_joint_positions(real_eval.HAND_OPEN)
                time.sleep(0.5)
                hand.close()
            except Exception:
                pass
        if rs_pipeline is not None:
            try:
                rs_pipeline.stop()
            except Exception:
                pass
        if zed_ctx is not None:
            try:
                zed_ctx[0].close()
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted OpenVLA-format left/up/right/down square")
    parser.add_argument("--target", choices=("sim", "real"), required=True)
    parser.add_argument("--leg-m", type=float, default=0.10, help="Length of each square side in meters")
    parser.add_argument("--step-m", type=float, default=0.01, help="OpenVLA translation per control step")
    parser.add_argument("--pause-steps", type=int, default=5, help="Zero-delta frames between legs")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--max-translation-step", type=float, default=0.018)
    parser.add_argument("--max-rotation-step", type=float, default=0.20)
    parser.add_argument(
        "--action-remap-rpy-deg",
        default=",".join(str(v) for v in real_eval.ACTION_REMAP_RPY_DEG),
        help="Used only on --target real. Default is eval_openvla_real Rz(-90).",
    )
    parser.add_argument("--robot-ip", default=real_eval.RC5_IP)
    parser.add_argument("--scene-json", type=Path, default=real_eval.DEFAULT_SCENE_JSON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-wrist", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--home-mode", choices=("sim", "current"), default="sim")
    parser.add_argument("--dry-run", action="store_true", help="Print the action list and exit")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    remap_rpy_deg = _parse_remap(args.action_remap_rpy_deg)
    records = build_square_actions(leg_m=args.leg_m, step_m=args.step_m, pause_steps=args.pause_steps)
    print(f"target={args.target}  legs={LEG_ORDER}  leg_m={args.leg_m}  step_m={args.step_m}  steps={len(records)}")
    print(f"camera_left_in_sim_base={CAMERA_LEFT_IN_SIM_BASE.tolist()}")
    print(f"action_remap_rpy_deg={list(remap_rpy_deg)} (real only)")
    sample = next(record for record in records if record["moving"] and record["leg"] == "LEFT")
    remapped_left = real_eval._remap_sim_action_to_real(sample["action"], remap_rpy_deg)
    print(f"LEFT openvla xyz={np.round(sample['action'][:3], 4).tolist()}")
    print(f"LEFT after real remap xyz={np.round(remapped_left[:3], 4).tolist()}")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / stamp / args.target
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sequence_json(
        output_dir / "sequence.json",
        records,
        {
            "target": args.target,
            "leg_m": float(args.leg_m),
            "step_m": float(args.step_m),
            "pause_steps": int(args.pause_steps),
            "hz": float(args.hz),
            "action_remap_rpy_deg": list(remap_rpy_deg),
        },
    )
    print(f"output={output_dir}")
    if args.dry_run:
        print("[DRY RUN] wrote sequence.json only")
        return
    if args.target == "sim":
        _run_sim(args, records, output_dir, remap_rpy_deg)
        return
    _run_real(args, records, output_dir, remap_rpy_deg)


if __name__ == "__main__":
    main()
