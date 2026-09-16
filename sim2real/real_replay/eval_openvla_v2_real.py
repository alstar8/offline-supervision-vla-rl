#!/usr/bin/env python3
"""Run an OpenVLA-V2 LoRA checkpoint closed-loop on the real RC5 + AeroHand.

OpenVLA-V2 (commit ed0f104) sees the scene and wrist cameras as separate images
plus a 7D proprio vector, where V1 saw one image with a wrist inset. Hardware
handling (RC5, AeroHand, ZED 2i, D405, homing, workspace, stall abort, logging)
comes from eval_openvla_real.py; the policy side mirrors the sim eval the
checkpoint was scored with:

  scene    ZED 2i left frame center-cropped to the sim scene camera FOV, 640x480
  wrist    D405 colour frame, np.rot90(k=-1), 224x168 (the sim wrist camera)
  proprio  6 arm joints in the training-scene convention [rad] + hand closure [0, 1]
  policy   SimplerEnv OpenVLAPolicy, parsed from the flags of
           runs/rl/pick_red_cube_refkl/eval_sim_v2.sh, greedy decoding
  action   SimlerWrapper._process_action: token bins -> sft_v2 unnormalization ->
           gripper openness snapped to {0, 0.2, ..., 1} -> max_ee_delta clip

Proprio convention. The checkpoint trained on the ed0f104 scene config, whose
`top` start profile has joint0 = 16.35 deg for a real home of 102.33 deg (hover
check, 2026-09-15): sim joint0 = real joint0 - 85.98 deg. Each joint is then
wrapped to within 180 deg of the checkpoint's proprio mean, which maps the real
joint3 = +179.5 deg to the -180.5 deg the data holds. Hand closure follows the
sim formula, mean(qpos / close_qpos) over the 16 hand joints, with qpos the
RCLevelHandController target for the commanded openness level; the open hand
gives 0.0228, the q01 of the sft_v2 closure stats.

Usage (rlvla_env; runs/rl/pick_red_cube_sft_databc/eval_real_v2.sh sets it up):
    eval_real_v2.sh --dry-run      # loads the model, dummy frames, no hardware
    eval_real_v2.sh                # real robot (Ctrl+C = hold + open hand)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Appended like simpler_wrapper does: sim2real/mani_skill must not shadow the repo ManiSkill (it has no rc5).
if str(REPO_ROOT / "sim2real") not in sys.path:
    sys.path.append(str(REPO_ROOT / "sim2real"))
import eval_openvla_real as v1  # noqa: E402  hardware helpers; installs the Ctrl+C handler

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "sim2real/runs/rl/pick_red_cube_sft_databc/wandb/offline-run-20260915_093914-9ehszx0r/glob/steps_0009"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "sim2real/runs/rl/pick_red_cube_sft_databc/real_eval_v2"
DEFAULT_UNNORM_KEY = "sft_v2"
DEFAULT_BASE_MODEL = "gen-robot/openvla-7b-rlvla-warmup"
SIM_JOINT0_OFFSET_DEG = 85.98
# eval_sim_v2.sh clips the decoded xyz delta to 5 cm; the real step limit below still applies.
MAX_EE_DELTA_M = 0.05
OPEN_HANDS = {"straight": v1.HAND_STRAIGHT, "hold": v1.HAND_HOLD}


def _build_policy(checkpoint: Path, base_model: str, unnorm_key: str, proprio_dim: int, max_ee_delta: float):
    import tyro
    from simpler_env import train_ms3_ppo_sft as trainer
    from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy

    flags = [
        "--env_id=OpenReal2Sim-v0",
        f"--vla_path={base_model}",
        f"--vla_load_path={checkpoint}",
        f"--vla_unnorm_key={unnorm_key}",
        "--vla_model_variant=v2",
        f"--vla_proprio_dim={proprio_dim}",
        "--num_envs=1",
        "--buffer_inferbatch=1",
        "--use_wrist_camera",
        f"--max_ee_delta={max_ee_delta}",
        "--no_wandb",
    ]
    args = tyro.cli(trainer.Args, args=flags)
    policy = OpenVLAPolicy(args, 0)
    policy.prep_rollout()
    return args, policy


def _build_decoder(args, policy):
    from simpler_env.env.simpler_wrapper import SimlerWrapper

    bins = np.linspace(-1, 1, 256)
    wrapper_state = SimpleNamespace(
        bin_centers=(bins[:-1] + bins[1:]) / 2.0,
        unnorm_state=policy.vla.base_model.norm_stats[args.vla_unnorm_key]["action"],
        args=args,
    )

    def decode(tokens) -> np.ndarray:
        return SimlerWrapper._process_action(wrapper_state, tokens).float().cpu().numpy()[0]

    return decode


def _build_closure():
    from openreal2sim.simulation.maniskill.agents.rc5_aero_hand_openr2s import RC5AeroHandOpenR2S as agent

    open_q = np.asarray(agent.canonical_hand_open_qpos, dtype=np.float64)
    close_q = np.asarray(agent.canonical_hand_close_qpos, dtype=np.float64)
    # _after_loading_articulation raises the closed thumb abduction to its URDF upper limit.
    close_q[0] = math.radians(100.0)

    def closure(openness: float) -> float:
        target = close_q + float(openness) * (open_q - close_q)
        return float(np.clip(target / close_q, 0.0, 1.0).mean())

    return closure


def _proprio(real_joints_deg, openness, closure, proprio_stats, joint0_offset_deg):
    mean_deg = np.degrees(np.asarray(proprio_stats["mean"][:6], dtype=np.float64))
    sim_deg = np.asarray(real_joints_deg, dtype=np.float64).copy()
    sim_deg[0] -= joint0_offset_deg
    sim_deg = mean_deg + (sim_deg - mean_deg + 180.0) % 360.0 - 180.0
    vector = np.concatenate([np.radians(sim_deg), [closure(openness)]]).astype(np.float32)
    return vector, sim_deg


def _hand_command(openness: float, open_hand) -> list[float]:
    close = np.asarray(v1.HAND_CLOSE, dtype=np.float64)
    opened = np.asarray(open_hand, dtype=np.float64)
    return [round(float(v), 3) for v in close + float(openness) * (opened - close)]


def _predict(policy, decode, scene_rgb, wrist_rgb, proprio, instruction):
    import torch

    obs = {
        "image": torch.from_numpy(np.ascontiguousarray(scene_rgb)[None]),
        "image_wrist": torch.from_numpy(np.ascontiguousarray(wrist_rgb)[None]),
        "proprio": torch.from_numpy(proprio[None]),
        "task_description": [instruction],
    }
    with torch.no_grad():
        _values, tokens, _logprobs = policy.get_action(obs, deterministic=True)
    return decode(tokens), [int(t) for t in tokens[0].tolist()]


def _grab_frames(zed_ctx, rs_pipeline):
    scene = v1._grab_zed_frame(zed_ctx)
    wrist_raw = v1._grab_realsense_frame(rs_pipeline)
    return scene, np.ascontiguousarray(v1._prepare_wrist_inset(wrist_raw)), wrist_raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--instruction", default=v1.DEFAULT_INSTRUCTION)
    parser.add_argument("--steps", type=int, default=112, help="Sim eval episode length for V2 (eval_sim_v2.sh)")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--max-translation-step", type=float, default=0.018)
    parser.add_argument("--max-rotation-step", type=float, default=0.20)
    parser.add_argument("--max-ee-delta", type=float, default=MAX_EE_DELTA_M)
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument(
        "--sim-joint0-offset-deg",
        type=float,
        default=SIM_JOINT0_OFFSET_DEG,
        help="sim joint0 = real joint0 - offset (training-scene convention)",
    )
    parser.add_argument(
        "--open-hand",
        choices=sorted(OPEN_HANDS),
        default="straight",
        help="Hand command at openness 1.0; the sim open preset is matched to the straight hand",
    )
    parser.add_argument("--robot-ip", default=v1.RC5_IP)
    parser.add_argument("--scene-json", type=Path, default=v1.DEFAULT_SCENE_JSON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-confirm", action="store_true", help="Do not wait for ENTER before moving")
    parser.add_argument("--home-mode", choices=("sim", "current"), default="sim")
    parser.add_argument("--workspace-z-min", type=float, default=v1.WORKSPACE_Z[0])
    parser.add_argument("--stall-abort-steps", type=int, default=v1.STALL_ABORT_STEPS)
    parser.add_argument("--action-remap-rpy-deg", default=",".join(str(v) for v in v1.ACTION_REMAP_RPY_DEG))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_dir = args.checkpoint.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")
    if not args.scene_json.is_file():
        raise FileNotFoundError(f"scene.json not found: {args.scene_json}")
    remap_rpy_deg = tuple(float(part) for part in str(args.action_remap_rpy_deg).split(","))
    if len(remap_rpy_deg) != 3:
        raise ValueError(f"--action-remap-rpy-deg needs three numbers, got {args.action_remap_rpy_deg!r}")

    output_dir = (args.output_dir or DEFAULT_OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")).resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    workspace = {"x": v1.WORKSPACE_X, "y": v1.WORKSPACE_Y, "z": (float(args.workspace_z_min), v1.WORKSPACE_Z[1])}
    open_hand = OPEN_HANDS[args.open_hand]

    policy_args, policy = _build_policy(
        checkpoint_dir, args.base_model, args.unnorm_key, args.proprio_dim, args.max_ee_delta
    )
    decode = _build_decoder(policy_args, policy)
    closure = _build_closure()
    proprio_stats = policy.vla.base_model.norm_stats[args.unnorm_key]["proprio"]
    prompt = v1._vla_prompt(args.instruction)
    print(f"instruction={args.instruction!r}  unnorm_key={args.unnorm_key}  hz={args.hz}  open_hand={args.open_hand}")
    print(f"closure(open)={closure(1.0):.4f}  closure(closed)={closure(0.0):.4f}  output={output_dir}")

    if args.dry_run:
        print("[DRY RUN] dummy frames, no hardware")
        home = v1._home_joints_deg()
        frames = []
        for step in range(min(args.steps, 8)):
            scene = np.random.randint(0, 255, (v1.SCENE_H, v1.SCENE_W, 3), dtype=np.uint8)
            wrist_raw = np.random.randint(0, 255, (v1.RS_H, v1.RS_W, 3), dtype=np.uint8)
            wrist = np.ascontiguousarray(v1._prepare_wrist_inset(wrist_raw))
            proprio, sim_deg = _proprio(home, 1.0, closure, proprio_stats, args.sim_joint0_offset_deg)
            action, tokens = _predict(policy, decode, scene, wrist, proprio, args.instruction)
            composite = v1._compose_wrist_inset(scene, wrist_raw)
            Image.fromarray(composite).save(frames_dir / f"step_{step:04d}.png")
            frames.append(composite)
            print(
                f"  step {step:04d} proprio_sim_deg={[round(float(v), 2) for v in sim_deg]} "
                f"closure={proprio[6]:.4f} action={[round(float(v), 5) for v in action]}"
            )
        v1._save_mp4(frames, output_dir / "rollout.mp4", fps=max(1, int(args.hz)))
        return

    robot = hand = zed_ctx = rs_pipeline = None
    frames: list[np.ndarray] = []
    step = 0
    openness = 1.0
    abort_reason = None
    failure = None
    try:
        print("Connecting to RC5...")
        robot = v1._init_rc5(args.robot_ip)
        print("Connecting to AeroHand...")
        from aero_open_sdk.aero_hand import AeroHand

        hand = AeroHand()
        print("Starting ZED2 and RealSense D405...")
        zed_ctx = v1._init_zed2(args.scene_json)
        rs_pipeline = v1._init_realsense()
        for _ in range(20):
            _grab_frames(zed_ctx, rs_pipeline)

        home_joints = home_tcp = None
        if args.home_mode == "sim":
            print("Moving to the scene home pose...")
            home_joints = v1._move_home(robot)
            home_tcp = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
        else:
            print("Keeping current arm pose as episode start.")

        hand.set_joint_positions(_hand_command(openness, open_hand))
        if not args.no_confirm:
            if sys.stdin.isatty():
                input("\nScene ready? Place the cube, then press ENTER to start...")
            else:
                print("\nNon-interactive stdin: starting in 15s. Place the cube now, or Ctrl+C to abort.")
                time.sleep(15)

        pose = list(robot.motion.linear.get_actual_position(orientation_units="deg"))
        summary = {
            "status": "running",
            "model_variant": "v2",
            "checkpoint": str(checkpoint_dir),
            "base_model": args.base_model,
            "unnorm_key": args.unnorm_key,
            "instruction": args.instruction,
            "prompt": prompt,
            "hz": float(args.hz),
            "decoding": "greedy",
            "max_ee_delta_m": float(args.max_ee_delta),
            "max_translation_step_m": float(args.max_translation_step),
            "max_rotation_step_rad": float(args.max_rotation_step),
            "action_remap_rpy_deg": list(remap_rpy_deg),
            "sim_joint0_offset_deg": float(args.sim_joint0_offset_deg),
            "open_hand": args.open_hand,
            "workspace": workspace,
            "home_mode": args.home_mode,
            "home_joints_target_deg": [round(v, 6) for v in v1.HOME_JOINTS_DEG],
            "home_joints_deg": None if home_joints is None else [round(float(v), 6) for v in home_joints],
            "home_tcp_m_deg": None if home_tcp is None else [round(float(v), 6) for v in home_tcp],
            "baseline_tcp": [round(float(v), 6) for v in pose],
        }
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Running {args.steps} steps at {args.hz} Hz. Ctrl+C = E-stop.\n")
        step_interval = 1.0 / args.hz
        target_pose = list(pose)
        prev_tcp = list(pose)
        stall_steps = 0

        while step < args.steps and not v1._STOP:
            t0 = time.monotonic()
            scene, wrist, wrist_raw = _grab_frames(zed_ctx, rs_pipeline)
            real_joints = v1._get_robot_joints_deg(robot)
            proprio, sim_deg = _proprio(real_joints, openness, closure, proprio_stats, args.sim_joint0_offset_deg)
            action, tokens = _predict(policy, decode, scene, wrist, proprio, args.instruction)
            remapped = v1._remap_sim_action_to_real(action, remap_rpy_deg)
            applied, safety = v1._limit_action_step(
                remapped,
                action_scale=args.action_scale,
                max_translation_step_m=args.max_translation_step,
                max_rotation_step_rad=args.max_rotation_step,
            )
            target_pose, clipped = v1._advance_target_pose(target_pose, applied, workspace)
            current_tcp = [float(v) for v in robot.motion.linear.get_actual_position(orientation_units="deg")]
            commanded_m = math.dist([0.0, 0.0, 0.0], [float(v) for v in applied[:3]])
            progress_m = math.dist(prev_tcp[:3], current_tcp[:3])
            at_limit = v1._axes_at_workspace_limit(current_tcp, workspace)
            holding_bound = clipped and bool(at_limit)
            blocked = commanded_m > v1.STALL_COMMAND_M and progress_m < v1.STALL_PROGRESS_M
            stall_steps = stall_steps + 1 if (holding_bound or blocked) else 0
            prev_tcp = current_tcp
            new_openness = float(applied[6])
            record = {
                "step": step,
                "current_tcp": [round(v, 6) for v in current_tcp],
                "target_tcp": [round(float(v), 6) for v in target_pose],
                "real_joints_deg": [round(float(v), 4) for v in real_joints],
                "proprio_sim_joints_deg": [round(float(v), 4) for v in sim_deg],
                "proprio_closure": round(float(proprio[6]), 4),
                "action_tokens": tokens,
                "raw_model_action": v1._action_to_dict(action),
                "remapped_action": v1._action_to_dict(remapped),
                "applied_action": v1._action_to_dict(applied),
                "action_safety": safety,
                "openness": new_openness,
                "workspace_clipped": clipped,
                "commanded_m": round(commanded_m, 6),
                "progress_m": round(progress_m, 6),
                "at_workspace_limit": at_limit,
                "stall_steps": int(stall_steps),
            }
            composite = v1._compose_wrist_inset(scene, wrist_raw)
            frames.append(composite)
            Image.fromarray(composite).save(frames_dir / f"step_{step:04d}.png")
            (frames_dir / f"step_{step:04d}.json").write_text(json.dumps(record, indent=2))
            print(
                f"  step {step:04d} action={[round(float(v), 5) for v in action]} "
                f"applied={[round(float(v), 5) for v in applied]} target={[round(float(v), 4) for v in target_pose[:3]]}"
            )
            if args.stall_abort_steps > 0 and stall_steps >= args.stall_abort_steps:
                cause = (
                    f"arm held against the {'/'.join(at_limit)} bound"
                    if holding_bound
                    else f"arm did not follow a {commanded_m * 1000:.1f} mm command (moved {progress_m * 1000:.1f} mm)"
                )
                abort_reason = f"{cause} for {stall_steps} consecutive steps at tcp={[round(v, 4) for v in current_tcp[:3]]}"
                print(f"  [abort] {abort_reason}")
                step += 1
                break
            v1._send_target_waypoint(robot, target_pose, workspace)
            if abs(new_openness - openness) > 1e-6:
                hand.set_joint_positions(_hand_command(new_openness, open_hand))
                print(f"  step {step:04d}: hand openness {openness:.1f} -> {new_openness:.1f}")
                openness = new_openness
            step += 1
            sleep_t = step_interval - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    except BaseException as exc:  # recorded for the summary, then re-raised
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        v1._save_mp4(frames, output_dir / "rollout.mp4", fps=max(1, int(args.hz)))
        summary_path = output_dir / "run_summary.json"
        if summary_path.is_file():
            if failure is not None:
                status = "failed"
            elif abort_reason is not None:
                status = "aborted"
            elif v1._STOP:
                status = "stopped"
            elif step >= int(args.steps):
                status = "complete"
            else:
                status = "incomplete"
            summary = json.loads(summary_path.read_text())
            summary.update({
                "status": status,
                "executed_steps": int(step),
                "requested_steps": int(args.steps),
                "abort_reason": abort_reason,
                "failure": failure,
            })
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"\nstatus={status} steps={step}/{int(args.steps)}")
        print("\nHolding RC5, opening hand.")
        if robot is not None:
            try:
                robot.motion.mode.set("hold")
            except Exception:
                pass
        if hand is not None:
            try:
                hand.set_joint_positions(v1.HAND_OPEN)
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
