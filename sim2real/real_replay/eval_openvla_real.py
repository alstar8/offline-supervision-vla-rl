#!/usr/bin/env python3
"""Run an OpenVLA LoRA checkpoint closed-loop on the real RC5 + AeroHand.

Observation matches OpenReal2Sim PPO eval: ZED2 scene resized to 640x480 with a
RealSense D405 wrist inset in the bottom-right. Actions are 7D EE deltas
(xyz meters, rpy radians, gripper) unnormalized with the checkpoint stats.
Sim-base xyz/rpy are applied in the real RC5 base with no extra yaw
(matching ``run_checkpoint_real.py`` and NPZ replay). Gripper is left as
the model predicted it. Translation/rotation deltas accumulate on a target
TCP (sim ``use_target=True``), not on the live pose.

Usage:
    # Always dry-run first (loads the model, dummy frames, no hardware):
    python eval_openvla_real.py --dry-run

    # Real robot (Ctrl+C = hold + open hand):
    python eval_openvla_real.py --instruction "Pick red cube"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM2REAL_ROOT = REPO_ROOT / "sim2real"
DEFAULT_CHECKPOINT = (
    SIM2REAL_ROOT
    / "runs/rl/pick_red_cube_sft_databc/wandb/offline-run-20260909_171210-wm5o2078/glob/steps_0059"
)
DEFAULT_SCENE_JSON = (
    SIM2REAL_ROOT / "assets/scenes/airi_table_new_empty3_image/simulation/scene.json"
)
DEFAULT_OUTPUT_ROOT = SIM2REAL_ROOT / "runs/rl/pick_red_cube_sft_databc/real_eval"
DEFAULT_INSTRUCTION = "Pick red cube"
DEFAULT_UNNORM_KEY = "sft"
DEFAULT_BASE_MODEL = "gen-robot/openvla-7b-rlvla-warmup"
DEFAULT_RC5_PYTHON_API_CANDIDATES = (
    Path(os.environ["RC5_PYTHON_API_ROOT"]) if os.environ.get("RC5_PYTHON_API_ROOT") else None,
    Path("/home/admin/Desktop/RC5_Hand_OpenVLA/python_api"),
    Path("/home/aermakov/github/ros2_rc5_control_pregrasp/python_api"),
)

RC5_IP = "10.10.10.10"
HAND_SLOT_NAMES = ("thumb_abd", "thumb_flex", "thumb_mcp_ip", "index", "middle", "ring", "pinky")
HAND_SLOT_UPPER = (100.0, 55.0, 90.0, 90.0, 90.0, 90.0, 90.0)
HAND_OPEN = [70.0, 20.0, 15.0, 30.0, 30.0, 30.0, 30.0]
HAND_CLOSE = [100.0, 55.0, 30.0, 60.0, 60.0, 60.0, 60.0]
HAND_HOLD = [70.0, 3.5, 14.0, 37.4, 29.4, 30.2, 29.4]
for _name, _value, _hi in zip(HAND_SLOT_NAMES, HAND_HOLD, HAND_SLOT_UPPER):
    if not 0.0 <= _value <= _hi:
        raise ValueError(f"HAND_HOLD[{_name}]={_value} outside the joint limit 0..{_hi}")

WORKSPACE_X = (-0.60, 0.10)
WORKSPACE_Y = (0.15, 0.90)
WORKSPACE_Z = (0.08, 0.90)

# Measured on this RC5 after placing the arm in the real pick-red-cube home
# (2026-09-10 14:06). Joints are commanded; TCP is recorded for logs.
HOME_JOINTS_DEG = (106.345596, -93.485413, -101.003151, 179.99073, -157.6054, -2.867088)
HOME_TCP_M_DEG = (-0.160111, 0.396493, 0.32875, 95.567035, -10.529792, -96.423843)
HOME_JOINT_SPEED = 25.0
HOME_JOINT_ACCEL = 25.0

# Joint-1 +90° is only a qpos home convention. Cartesian VLA deltas are already
# in the robot-base frame used by the RC5 TCP API; the proven desktop eval and
# NPZ replay apply them with identity. Teleop's Rz(+90°) is real→sim for the
# human stick, not sim→real for the policy. Use --action-remap-rpy-deg if needed.
ACTION_REMAP_RPY_DEG = (0.0, 0.0, 0.0)

WP_SPEED = 0.10
WP_ACCEL = 0.10
WP_BLEND = 0.01

SCENE_W, SCENE_H = 640, 480
RS_W, RS_H = 640, 480
CAMERA_FPS = 30
WRIST_INSET_MARGIN = 4
WRIST_INSET_BORDER = 4
WRIST_INSET_HEIGHT = 224
WRIST_INSET_WIDTH = 168
ACTION_LABELS = ("dx", "dy", "dz", "drx", "dry", "drz", "grip")

_STOP = False


def _sigint_handler(sig, frame) -> None:
    global _STOP
    _STOP = True
    print("\n[E-STOP] Ctrl+C — stopping after the current step.")


signal.signal(signal.SIGINT, _sigint_handler)


def _resolve_python_api() -> Path:
    for candidate in DEFAULT_RC5_PYTHON_API_CANDIDATES:
        if candidate is not None and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "RC5 python_api not found. Set RC5_PYTHON_API_ROOT to the directory "
        "that contains the API package."
    )


def _load_norm_stats(checkpoint_dir: Path, key: str) -> dict:
    stats_path = checkpoint_dir / "dataset_statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing {stats_path}")
    with stats_path.open() as handle:
        all_stats = json.load(handle)
    if key not in all_stats:
        raise KeyError(f"unnorm key {key!r} not in {stats_path}. available={list(all_stats)}")
    return all_stats[key]["action"]


def _load_model(checkpoint_dir: Path, base_model_path: str):
    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    openvla_root = str(REPO_ROOT / "openvla")
    if openvla_root not in sys.path:
        sys.path.insert(0, openvla_root)
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    print("[model] loading base OpenVLA...")
    base = OpenVLAForActionPrediction.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    cfg_path = checkpoint_dir / "adapter_config.json"
    with cfg_path.open() as handle:
        cfg = json.load(handle)
    cfg["auto_mapping"] = None
    tmp = Path(tempfile.mkdtemp(prefix="openvla_lora_")) / "ckpt"
    tmp.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        src = checkpoint_dir / name
        if src.is_file() and name != "adapter_config.json":
            shutil.copy2(src, tmp / name)
    with (tmp / "adapter_config.json").open("w") as handle:
        json.dump(cfg, handle, indent=2)

    model = PeftModel.from_pretrained(base, str(tmp))
    model.eval()
    print(f"[model] LoRA loaded from {checkpoint_dir}")
    return processor, model


def _iter_wrapped_modules(model):
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        nxt = None
        if hasattr(current, "get_base_model"):
            try:
                nxt = current.get_base_model()
            except Exception:
                nxt = None
        if nxt is None:
            nxt = getattr(current, "model", None)
        current = None if nxt is current else nxt


def _inject_norm_stats(model, unnorm_key: str, stats: dict) -> None:
    payload = {"action": stats}
    for module in _iter_wrapped_modules(model):
        existing = getattr(module, "norm_stats", None)
        if existing is None:
            try:
                module.norm_stats = {unnorm_key: payload}
            except Exception:
                continue
        else:
            existing[unnorm_key] = payload


def _vla_prompt(instruction: str) -> str:
    text = " ".join(str(instruction).strip().split()).lower()
    return f"In: What action should the robot take to {text}?\nOut: "


def _predict_action(model, processor, image: Image.Image, instruction: str, unnorm_key: str, stats: dict) -> np.ndarray:
    import torch

    _inject_norm_stats(model, unnorm_key, stats)
    inputs = processor(_vla_prompt(instruction), image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
    # predict_action may append token 29871 to input_ids without padding attention_mask.
    generate_inputs = {key: value for key, value in inputs.items() if key != "attention_mask"}
    with torch.no_grad():
        action = model.predict_action(**generate_inputs, unnorm_key=unnorm_key, do_sample=False)
    if isinstance(action, np.ndarray):
        return action.astype(np.float32)
    return action.detach().cpu().float().numpy()


def _rpy_deg_to_rotation(rpy_deg: tuple[float, float, float] | list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _remap_sim_action_to_real(action: np.ndarray, remap_rpy_deg: tuple[float, float, float]) -> np.ndarray:
    remapped = np.asarray(action, dtype=np.float32).copy()
    rotation = _rpy_deg_to_rotation(remap_rpy_deg)
    remapped[:3] = (rotation @ remapped[:3]).astype(np.float32)
    remapped[3:6] = (rotation @ remapped[3:6]).astype(np.float32)
    remapped[6] = float(action[6])
    return remapped


def _limit_action_step(
    action: np.ndarray,
    *,
    action_scale: float,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> tuple[np.ndarray, dict]:
    applied = action.copy().astype(np.float32)
    applied[:6] *= float(action_scale)
    trans_norm = float(np.linalg.norm(applied[:3]))
    rot_norm = float(np.linalg.norm(applied[3:6]))
    trans_clip = 1.0
    rot_clip = 1.0
    if max_translation_step_m > 0.0 and trans_norm > max_translation_step_m:
        trans_clip = float(max_translation_step_m / max(trans_norm, 1e-8))
        applied[:3] *= trans_clip
    if max_rotation_step_rad > 0.0 and rot_norm > max_rotation_step_rad:
        rot_clip = float(max_rotation_step_rad / max(rot_norm, 1e-8))
        applied[3:6] *= rot_clip
    applied[6] = action[6]
    return applied, {
        "action_scale": float(action_scale),
        "raw_translation_norm_m": round(float(np.linalg.norm(action[:3])), 6),
        "applied_translation_norm_m": round(float(np.linalg.norm(applied[:3])), 6),
        "raw_rotation_norm_rad": round(float(np.linalg.norm(action[3:6])), 6),
        "applied_rotation_norm_rad": round(float(np.linalg.norm(applied[3:6])), 6),
        "translation_clip_scale": round(trans_clip, 6),
        "rotation_clip_scale": round(rot_clip, 6),
    }


def _clip_pose(pose: list[float], workspace: dict[str, tuple[float, float]]) -> tuple[list[float], bool]:
    clipped = list(pose)
    changed = False
    for index, key in enumerate(("x", "y", "z")):
        lo, hi = workspace[key]
        if clipped[index] < lo:
            clipped[index] = lo
            changed = True
        elif clipped[index] > hi:
            clipped[index] = hi
            changed = True
    return clipped, changed


def _gripper_closed(gripper: float) -> bool:
    # Match SimplerEnv: open_gripper > 0.5 stays open, otherwise close.
    return float(gripper) <= 0.5


def _action_to_dict(action: np.ndarray) -> dict[str, float]:
    return {name: float(action[i]) for i, name in enumerate(ACTION_LABELS)}


def _resize_nearest_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[y_idx][:, x_idx]


def _sim_scene_fov_radians(scene_json: Path) -> tuple[float, float]:
    with scene_json.open() as handle:
        cam = json.load(handle)["camera"]
    sx = SCENE_W / float(cam["width"])
    sy = SCENE_H / float(cam["height"])
    fx = float(cam["fx"]) * sx
    fy = float(cam["fy"]) * sy
    vfov = 2.0 * np.arctan(SCENE_H / (2.0 * fy))
    hfov = 2.0 * np.arctan(SCENE_W / (2.0 * fx))
    return hfov, vfov


def _crop_center_to_match_fov(
    rgb: np.ndarray,
    hfov_src: float,
    vfov_src: float,
    hfov_tgt: float,
    vfov_tgt: float,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale_h = np.tan(vfov_tgt / 2.0) / np.tan(vfov_src / 2.0) if vfov_tgt < vfov_src else 1.0
    scale_w = np.tan(hfov_tgt / 2.0) / np.tan(hfov_src / 2.0) if hfov_tgt < hfov_src else 1.0
    scale_h = float(np.clip(scale_h, 0.0, 1.0))
    scale_w = float(np.clip(scale_w, 0.0, 1.0))
    new_h = max(1, int(round(height * scale_h)))
    new_w = max(1, int(round(width * scale_w)))
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    return rgb[y0:y0 + new_h, x0:x0 + new_w]


def _prepare_wrist_inset(wrist_rgb: np.ndarray) -> np.ndarray:
    wrist_rgb = np.rot90(wrist_rgb, k=-1)
    return _resize_nearest_rgb(wrist_rgb, WRIST_INSET_HEIGHT, WRIST_INSET_WIDTH)


def _compose_wrist_inset(scene_rgb: np.ndarray, wrist_rgb: np.ndarray | None) -> np.ndarray:
    if wrist_rgb is None:
        return scene_rgb
    wrist_rgb = _prepare_wrist_inset(wrist_rgb)
    border = WRIST_INSET_BORDER
    inset_h = WRIST_INSET_HEIGHT
    inset_w = WRIST_INSET_WIDTH
    margin = WRIST_INSET_MARGIN
    top = scene_rgb.shape[0] - inset_h - 2 * border - margin
    left = scene_rgb.shape[1] - inset_w - 2 * border - margin
    out = scene_rgb.copy()
    out[top:top + inset_h + 2 * border, left:left + inset_w + 2 * border, :] = 0
    out[top + border:top + border + inset_h, left + border:left + border + inset_w, :] = wrist_rgb
    return out


def _init_zed2(scene_json: Path):
    import pyzed.sl as sl

    hfov_tgt, vfov_tgt = _sim_scene_fov_radians(scene_json)
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED2 open failed: {status}")
    left = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    hfov_src = np.radians(float(left.h_fov))
    vfov_src = np.radians(float(left.v_fov))
    print(
        f"  ZED2 FOV {np.degrees(hfov_src):.1f}x{np.degrees(vfov_src):.1f} deg -> "
        f"sim {np.degrees(hfov_tgt):.1f}x{np.degrees(vfov_tgt):.1f} deg"
    )
    return zed, sl.RuntimeParameters(), sl.Mat(), hfov_src, vfov_src, hfov_tgt, vfov_tgt


def _grab_zed_frame(zed_ctx) -> np.ndarray:
    import pyzed.sl as sl

    zed, runtime, image, hfov_src, vfov_src, hfov_tgt, vfov_tgt = zed_ctx
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED2 grab failed")
    zed.retrieve_image(image, sl.VIEW.LEFT)
    bgra = image.get_data()
    if bgra is None:
        raise RuntimeError("ZED2 returned an empty frame")
    rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
    rgb = _crop_center_to_match_fov(rgb, hfov_src, vfov_src, hfov_tgt, vfov_tgt)
    return _resize_nearest_rgb(rgb, SCENE_H, SCENE_W)


def _init_realsense():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, CAMERA_FPS)
    pipeline.start(cfg)
    return pipeline


def _grab_realsense_frame(pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    bgr = np.asanyarray(color.get_data())
    return bgr[:, :, ::-1]


def _grab_openvla_frame(zed_ctx, rs_pipeline) -> Image.Image:
    scene_rgb = _grab_zed_frame(zed_ctx)
    wrist_rgb = None if rs_pipeline is None else _grab_realsense_frame(rs_pipeline)
    return Image.fromarray(_compose_wrist_inset(scene_rgb, wrist_rgb))


def _init_rc5(robot_ip: str):
    api_root = _resolve_python_api()
    api_root_str = str(api_root)
    if api_root_str not in sys.path:
        sys.path.insert(0, api_root_str)
    from API.rc_api import RobotApi
    from API.source.models.classes.enum_classes.state_classes import (
        InComingControllerState as Ics,
        InComingSafetyStatus as Iss,
    )

    print(f"  python_api={api_root}")
    robot = RobotApi(robot_ip, show_std_traceback=True)
    if robot.safety_status.get() == Iss.fault.name or robot.controller_state.get() == Ics.failure.name:
        robot.controller_state.set("off")
    robot.controller_state.set("run", await_sec=120)
    return robot


def _start_move_if_needed(robot, await_sec: int = 5) -> bool:
    mode = robot.motion.mode.get()
    if mode == "move":
        return True
    if mode in ("hold", "pause"):
        return robot.motion.mode.set("move", await_sec=await_sec)
    return False


def _wrap_degrees_near(angle: float, reference: float) -> float:
    return float(angle + 360.0 * round((reference - angle) / 360.0))


def _home_joints_deg(reference_joints_deg: list[float] | None = None) -> list[float]:
    target = [float(v) for v in HOME_JOINTS_DEG]
    if reference_joints_deg is None:
        return target
    return [
        _wrap_degrees_near(angle, float(reference))
        for angle, reference in zip(target, reference_joints_deg)
    ]


def _get_robot_joints_deg(robot) -> list[float]:
    return [float(v) for v in robot.motion.joint.get_actual_position(units="deg")]


def _send_joint_waypoint(robot, target_joints_deg: list[float], *, context: str = "home") -> bool:
    label = f" ({context})" if context else ""
    try:
        robot.motion.joint.add_new_waypoint(
            angle_pose=tuple(float(v) for v in target_joints_deg),
            speed=HOME_JOINT_SPEED,
            accel=HOME_JOINT_ACCEL,
            blend=0.0,
            units="deg",
        )
        if not _start_move_if_needed(robot, await_sec=30):
            print(f"  [skip] could not enter MOVE{label}")
            return False
        robot.motion.wait_waypoint_completion(30)
        return True
    except Exception as exc:
        if type(exc).__name__ != "AddWaypointError" and "waypoint" not in str(exc).lower():
            raise
        print(f"  [skip] joint waypoint rejected{label}: {exc}")
        return False


def _move_home(robot) -> list[float]:
    current = _get_robot_joints_deg(robot)
    target = _home_joints_deg(current)
    print(f"  Moving to measured home joints deg={[round(v, 3) for v in target]}")
    if not _send_joint_waypoint(robot, target, context="home"):
        raise RuntimeError("Failed to reach the recorded home pose.")
    arrived = _get_robot_joints_deg(robot)
    tcp = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
    print(f"  home joints={[round(v, 3) for v in arrived]}")
    print(f"  home tcp={[round(float(v), 4) for v in tcp]}")
    return arrived


def _advance_target_pose(
    target: list[float],
    action: np.ndarray,
    workspace: dict[str, tuple[float, float]],
) -> tuple[list[float], bool]:
    # Match sim ee_align2 use_target=True: add the delta to the last commanded pose.
    next_pose = [
        float(target[0]) + float(action[0]),
        float(target[1]) + float(action[1]),
        float(target[2]) + float(action[2]),
        float(target[3]) + math.degrees(float(action[3])),
        float(target[4]) + math.degrees(float(action[4])),
        float(target[5]) + math.degrees(float(action[5])),
    ]
    return _clip_pose(next_pose, workspace)


def _send_target_waypoint(robot, target: list[float], workspace: dict[str, tuple[float, float]]) -> bool:
    clipped, changed = _clip_pose(list(target), workspace)
    if changed:
        print(f"  [clip] workspace clamp applied -> xyz={[round(v, 4) for v in clipped[:3]]}")
    tcp = tuple(clipped[:3] + clipped[3:6])
    try:
        if not robot.motion.is_point_reachable(tcp_pose=tcp, orientation_units="deg"):
            print(f"  [skip] unreachable xyz={[round(v, 4) for v in clipped[:3]]}")
            return False
        robot.motion.linear.add_new_waypoint(
            tcp, speed=WP_SPEED, accel=WP_ACCEL, blend=WP_BLEND, orientation_units="deg"
        )
        _start_move_if_needed(robot)
        return True
    except Exception as exc:
        if type(exc).__name__ != "AddWaypointError" and "waypoint" not in str(exc).lower():
            raise
        print(f"  [skip] waypoint rejected: {exc}")
        return False


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int = 5) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except Exception:
        print(f"  imageio not available; PNG frames are in {path.parent}")
        return
    writer = imageio.get_writer(str(path), fps=fps, quality=7)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  wrote {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an OpenVLA checkpoint on real RC5")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--max-translation-step", type=float, default=0.018)
    parser.add_argument("--max-rotation-step", type=float, default=0.20)
    parser.add_argument(
        "--action-remap-rpy-deg",
        default=",".join(str(v) for v in ACTION_REMAP_RPY_DEG),
        help="RPY degrees applied to sim xyz/rpy deltas before sending to the real RC5. "
        "Default is identity (0,0,0). Try 0,0,90 or 0,0,-90 only if XY is still rotated.",
    )
    parser.add_argument("--robot-ip", default=RC5_IP)
    parser.add_argument("--scene-json", type=Path, default=DEFAULT_SCENE_JSON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wrist", action="store_true", help="Skip RealSense D405 wrist inset")
    parser.add_argument("--no-confirm", action="store_true", help="Do not wait for ENTER before moving")
    parser.add_argument(
        "--home-mode",
        choices=("sim", "current"),
        default="sim",
        help="Episode start: recorded real home joints (default) or leave the arm where it is",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_dir = args.checkpoint.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")
    if not args.scene_json.is_file():
        raise FileNotFoundError(f"scene.json not found: {args.scene_json}")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / stamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    workspace = {"x": WORKSPACE_X, "y": WORKSPACE_Y, "z": WORKSPACE_Z}
    remap_rpy_deg = tuple(float(part) for part in str(args.action_remap_rpy_deg).split(","))
    if len(remap_rpy_deg) != 3:
        raise ValueError(f"--action-remap-rpy-deg must be three comma-separated numbers, got {args.action_remap_rpy_deg!r}")
    stats = _load_norm_stats(checkpoint_dir, args.unnorm_key)
    processor, model = _load_model(checkpoint_dir, args.base_model)
    print(f"instruction={args.instruction!r}  unnorm_key={args.unnorm_key}  hz={args.hz}")
    print(f"action_remap_rpy_deg={list(remap_rpy_deg)}")
    print(f"output={output_dir}")

    if args.dry_run:
        print("[DRY RUN] dummy frames, no hardware")
        print(f"  home_mode={args.home_mode} home_joints_deg={[round(v, 3) for v in _home_joints_deg()]}")
        dummy_frames = []
        for step in range(min(args.steps, 8)):
            scene = np.random.randint(0, 255, (SCENE_H, SCENE_W, 3), dtype=np.uint8)
            wrist = np.random.randint(0, 255, (RS_H, RS_W, 3), dtype=np.uint8)
            image = Image.fromarray(_compose_wrist_inset(scene, wrist))
            action = _predict_action(model, processor, image, args.instruction, args.unnorm_key, stats)
            remapped = _remap_sim_action_to_real(action, remap_rpy_deg)
            applied, safety = _limit_action_step(
                remapped,
                action_scale=args.action_scale,
                max_translation_step_m=args.max_translation_step,
                max_rotation_step_rad=args.max_rotation_step,
            )
            record = {
                "step": step,
                "prompt": _vla_prompt(args.instruction),
                "raw_model_action": _action_to_dict(action),
                "remapped_action": _action_to_dict(remapped),
                "applied_action": _action_to_dict(applied),
                "action_safety": safety,
                "gripper_closed": _gripper_closed(applied[6]),
            }
            image.save(frames_dir / f"step_{step:04d}.png")
            (frames_dir / f"step_{step:04d}.json").write_text(json.dumps(record, indent=2))
            dummy_frames.append(np.asarray(image))
            print(
                f"  step {step:04d} raw={[round(float(v), 5) for v in action]} "
                f"real={[round(float(v), 5) for v in remapped]}"
            )
        _save_mp4(dummy_frames, output_dir / "rollout.mp4", fps=max(1, int(args.hz)))
        return

    robot = None
    hand = None
    zed_ctx = None
    rs_pipeline = None
    frames: list[np.ndarray] = []
    step = 0
    grip_closed = False
    home_joints = None
    try:
        print("Connecting to RC5...")
        robot = _init_rc5(args.robot_ip)
        actual = robot.motion.linear.get_actual_position(orientation_units="deg")
        print(f"  RC5 OK tcp={actual}")

        print("Connecting to AeroHand...")
        from aero_open_sdk.aero_hand import AeroHand

        hand = AeroHand()
        print(f"  AeroHand OK actuations={hand.get_actuations()}")

        print("Starting ZED2...")
        zed_ctx = _init_zed2(args.scene_json)
        if not args.no_wrist:
            print("Starting RealSense D405...")
            rs_pipeline = _init_realsense()
            print("  RealSense OK")
        else:
            print("  wrist camera disabled")

        print("Warming up cameras...")
        for _ in range(20):
            _grab_openvla_frame(zed_ctx, rs_pipeline)

        home_joints = None
        if args.home_mode == "sim":
            print("Moving to recorded home pose...")
            home_joints = _move_home(robot)
        elif args.home_mode == "current":
            print("Keeping current arm pose as episode start.")
        else:
            raise ValueError(f"unsupported home mode: {args.home_mode}")

        hand.set_joint_positions(HAND_HOLD)
        if not args.no_confirm:
            if sys.stdin.isatty():
                input("\nScene ready? Place the cube, then press ENTER to start...")
            else:
                wait_s = 15
                print(
                    f"\nNon-interactive stdin: starting in {wait_s}s. "
                    "Place the cube now, or Ctrl+C to abort."
                )
                time.sleep(wait_s)

        pose = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
        summary = {
            "status": "running",
            "checkpoint": str(checkpoint_dir),
            "base_model": args.base_model,
            "unnorm_key": args.unnorm_key,
            "instruction": args.instruction,
            "prompt": _vla_prompt(args.instruction),
            "hz": float(args.hz),
            "action_scale": float(args.action_scale),
            "max_translation_step_m": float(args.max_translation_step),
            "max_rotation_step_rad": float(args.max_rotation_step),
            "action_remap_rpy_deg": [float(v) for v in remap_rpy_deg],
            "workspace": workspace,
            "home_mode": args.home_mode,
            "home_joints_target_deg": [round(v, 6) for v in HOME_JOINTS_DEG],
            "home_tcp_m_deg": [round(v, 6) for v in HOME_TCP_M_DEG],
            "home_joints_deg": None if home_joints is None else [round(float(v), 6) for v in home_joints],
            "baseline_tcp": [round(float(v), 6) for v in pose],
        }
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Running {args.steps} steps at {args.hz} Hz. Ctrl+C = E-stop.\n")
        print("  Waypoints accumulate on a target TCP (sim use_target=True).")
        step_interval = 1.0 / args.hz
        target_pose = list(pose)

        while step < args.steps and not _STOP:
            t0 = time.monotonic()
            image = _grab_openvla_frame(zed_ctx, rs_pipeline)
            action = _predict_action(model, processor, image, args.instruction, args.unnorm_key, stats)
            remapped = _remap_sim_action_to_real(action, remap_rpy_deg)
            applied, safety = _limit_action_step(
                remapped,
                action_scale=args.action_scale,
                max_translation_step_m=args.max_translation_step,
                max_rotation_step_rad=args.max_rotation_step,
            )
            target_pose, clipped = _advance_target_pose(target_pose, applied, workspace)
            record = {
                "step": step,
                "prompt": _vla_prompt(args.instruction),
                "current_tcp": [round(float(v), 6) for v in robot.motion.linear.get_actual_position(orientation_units="deg")],
                "target_tcp": [round(float(v), 6) for v in target_pose],
                "raw_model_action": _action_to_dict(action),
                "remapped_action": _action_to_dict(remapped),
                "applied_action": _action_to_dict(applied),
                "action_safety": safety,
                "workspace_clipped": clipped,
                "gripper_closed": _gripper_closed(applied[6]),
            }
            frame = np.asarray(image.convert("RGB"))
            frames.append(frame)
            Image.fromarray(frame).save(frames_dir / f"step_{step:04d}.png")
            (frames_dir / f"step_{step:04d}.json").write_text(json.dumps(record, indent=2))
            print(
                f"  step {step:04d} raw={[round(float(v), 5) for v in action]} "
                f"real={[round(float(v), 5) for v in remapped]} "
                f"applied={[round(float(v), 5) for v in applied]} "
                f"target={[round(float(v), 4) for v in target_pose[:3]]}"
            )
            _send_target_waypoint(robot, target_pose, workspace)
            new_closed = _gripper_closed(applied[6])
            if new_closed != grip_closed:
                hand.set_joint_positions(HAND_CLOSE if new_closed else HAND_HOLD)
                print(f"  step {step:04d}: gripper -> {'CLOSE' if new_closed else 'HOLD'}")
                grip_closed = new_closed
            step += 1
            sleep_t = step_interval - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        _save_mp4(frames, output_dir / "rollout.mp4", fps=max(1, int(args.hz)))
        summary_path = output_dir / "run_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            summary.update({
                "status": "stopped" if _STOP else "complete",
                "executed_steps": int(step),
            })
            summary_path.write_text(json.dumps(summary, indent=2))
        print("\nHolding RC5, opening hand.")
        if robot is not None:
            try:
                robot.motion.mode.set("hold")
            except Exception:
                pass
        if hand is not None:
            try:
                hand.set_joint_positions(HAND_OPEN)
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


if __name__ == "__main__":
    main()
