from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from real2sim.calibrate_rc5_pose import (
    Args as CalibArgs,
    _apply_camera_state,
    _build_env,
    _current_obs_info,
    _load_state_file,
    _maybe_render_live,
    _read_command_line_nonblocking,
    _reset_current_state,
    _save_snapshot_from_obs,
    _sanitize_live_viewer_environment,
    observation_camera_lookat_state,
)
from real2sim.openreal2sim_validation import AIRI_CUBES_ASSET_DIR


DEFAULT_AIRI_CUBES_HOME = (
    Path(__file__).resolve().parent / "presets" / "rc5_pose" / "airi_cubes_home.txt"
)


@dataclass
class Args(CalibArgs):
    scene_asset_dir: str = str(AIRI_CUBES_ASSET_DIR)
    task_mode: str = "airi_cube_pickup"
    load_state_file: str = str(DEFAULT_AIRI_CUBES_HOME)
    output_state_file: str = str(DEFAULT_AIRI_CUBES_HOME)
    output_dir: str = str(Path("real2sim/debug/rc5_pose/airi_cubes_home_calib"))
    image_prefix: str = "airi_cubes_home"
    live_viewer: bool = True
    clean_scene: bool = True
    use_probe_objects: bool = False
    show_debug_markers: bool = False
    keep_history: bool = True
    joint_step_deg: float = 2.0
    joint_big_step_deg: float = 10.0


def _write_state_file(path: str | Path, base_p: np.ndarray, base_q: np.ndarray, qpos: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                f"robot_base_pose_p={base_p.astype(float).tolist()}",
                f"robot_base_pose_q={base_q.astype(float).tolist()}",
                f"robot_init_qpos={qpos.astype(float).tolist()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved state file: {out}")


def _tensor_row(value) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def _arm_limits(env) -> np.ndarray:
    qlimits = env.unwrapped.agent.robot.get_qlimits()
    return _tensor_row(qlimits)[:6]


def _print_state(env, obs: dict, qpos: np.ndarray, selected_joint: int) -> None:
    arm = np.asarray(qpos[:6], dtype=np.float32)
    limits = _arm_limits(env)
    margins = np.minimum(arm - limits[:, 0], limits[:, 1] - arm)
    tcp = _tensor_row(obs["extra"]["tcp_pose"])[:7]
    print("")
    print(f"selected joint{selected_joint} value={arm[selected_joint]:.4f} rad ({np.rad2deg(arm[selected_joint]):.1f} deg)")
    print("arm_qpos=" + np.array2string(arm, precision=4, suppress_small=True))
    print("limit_margins=" + np.array2string(margins, precision=4, suppress_small=True))
    print("tcp_pose=" + np.array2string(tcp, precision=4, suppress_small=True))


def _save_preview(env, obs: dict, info: dict, args: Args, base_p: np.ndarray, base_q: np.ndarray, save_idx: int) -> None:
    _save_snapshot_from_obs(env, obs, info, args, base_p, base_q, save_idx)
    frame_path = Path(args.output_dir) / f"{args.image_prefix}_latest.png"
    if frame_path.exists():
        print(f"Preview image: {frame_path}")


def _print_help(args: Args) -> None:
    print("")
    print("AIRI cubes home calibration commands:")
    print("  0..5          select arm joint index (zero-based, matching robot_init_qpos)")
    print("  - / +         jog selected joint by joint_step_deg")
    print("  -- / ++       jog selected joint by joint_big_step_deg")
    print("  joint I D     add D radians to arm joint index I")
    print("  show          print qpos, margins, and TCP")
    print("  save          write output_state_file and preview snapshot")
    print("  preview       save preview snapshot only")
    print("  reset         reload load_state_file")
    print("  load PATH     load another state file")
    print("  quit          exit")
    print(f"Small step: {args.joint_step_deg} deg, big step: {args.joint_big_step_deg} deg")
    print(f"Output state file: {args.output_state_file}")


def main(args: Args) -> None:
    _sanitize_live_viewer_environment(args)
    base_p, base_q, qpos = _load_state_file(args.load_state_file)
    base_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)

    cam_state = observation_camera_lookat_state(args.observation_camera_mode, scene_asset_dir=args.scene_asset_dir)
    camera_eye = np.array(cam_state["eye"], dtype=np.float32)
    camera_target = np.array(cam_state["target"], dtype=np.float32)
    camera_roll_deg = float(cam_state.get("roll_deg", 0.0))
    camera_fov = float(cam_state.get("fov", 1.0))

    env = _build_env(
        args,
        camera_mode=args.observation_camera_mode,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_roll_deg=camera_roll_deg,
        camera_fov=camera_fov,
    )
    try:
        selected_joint = 3
        save_idx = 0
        obs, info = _reset_current_state(env, args, base_p, base_q, qpos, reconfigure=True)
        _apply_camera_state(env, camera_eye, camera_target, camera_roll_deg, camera_fov)
        _save_preview(env, obs, info, args, base_p, base_q, save_idx)
        _print_help(args)
        _print_state(env, obs, qpos, selected_joint)

        prompt_shown = False
        while True:
            if args.live_viewer:
                raw = _read_command_line_nonblocking(
                    "airi-cubes-home> ",
                    timeout_sec=0.03,
                    show_prompt=not prompt_shown,
                )
                prompt_shown = True
                if raw is None:
                    _maybe_render_live(env, args)
                    continue
                print("")
                prompt_shown = False
            else:
                raw = input("airi-cubes-home> ").strip()
            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()
            reset_after_command = False
            try:
                if cmd in {"quit", "exit", "q"}:
                    break
                if cmd in {"help", "h"}:
                    _print_help(args)
                    continue
                if cmd in {"0", "1", "2", "3", "4", "5"}:
                    selected_joint = int(cmd)
                    _print_state(env, obs, qpos, selected_joint)
                    continue
                if cmd == "show":
                    _print_state(env, obs, qpos, selected_joint)
                    continue
                if cmd == "save":
                    _write_state_file(args.output_state_file, base_p, base_q, qpos)
                    save_idx += 1
                    _save_preview(env, obs, info, args, base_p, base_q, save_idx)
                    continue
                if cmd == "preview":
                    save_idx += 1
                    _save_preview(env, obs, info, args, base_p, base_q, save_idx)
                    continue
                if cmd == "reset":
                    base_p, base_q, qpos = _load_state_file(args.load_state_file)
                    base_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
                    reset_after_command = True
                elif cmd == "load":
                    args.load_state_file = parts[1]
                    base_p, base_q, qpos = _load_state_file(args.load_state_file)
                    base_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
                    reset_after_command = True
                elif cmd == "joint":
                    idx = int(parts[1])
                    if not 0 <= idx <= 5:
                        raise ValueError("joint index must be 0..5")
                    qpos[idx] += float(parts[2])
                    selected_joint = idx
                    reset_after_command = True
                elif cmd in {"-", "+" , "--", "++"}:
                    step = np.deg2rad(float(args.joint_step_deg))
                    if cmd in {"--", "++"}:
                        step = np.deg2rad(float(args.joint_big_step_deg))
                    if cmd.startswith("-"):
                        step = -step
                    qpos[selected_joint] += step
                    reset_after_command = True
                else:
                    print("Unknown command. Type help.")
                    continue
            except (IndexError, ValueError) as exc:
                print(f"Bad command: {exc}")
                continue

            if reset_after_command:
                obs, info = _reset_current_state(env, args, base_p, base_q, qpos)
                _apply_camera_state(env, camera_eye, camera_target, camera_roll_deg, camera_fov)
                save_idx += 1
                _save_preview(env, obs, info, args, base_p, base_q, save_idx)
                _print_state(env, obs, qpos, selected_joint)
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
