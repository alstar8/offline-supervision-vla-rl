from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import sapien
import tyro
from transforms3d.euler import euler2quat, quat2euler

from real2sim.calibrate_rc5_pose import (
    Args as BaseArgs,
    _apply_grid_pair_override,
    _build_env,
    _compose_wrist_inset,
    _load_state_file,
    _reset_current_state,
    _to_uint8_image,
)
from real2sim.debug_paths import WRIST_CAMERA_CALIBRATION_DIR
from real2sim.openreal2sim_validation import (
    WRIST_CAMERA_FOV,
    WRIST_CAMERA_LOCAL_P,
    WRIST_CAMERA_LOCAL_Q,
    WRIST_CAMERA_NAME,
    source_model_names_for_obj_set,
)


def _quat_to_rpy_deg(quat_wxyz) -> np.ndarray:
    return np.rad2deg(np.asarray(quat2euler(quat_wxyz, axes="sxyz"), dtype=np.float64))


def _pose_from_state(pos: np.ndarray, rpy_deg: np.ndarray) -> sapien.Pose:
    quat = euler2quat(*np.deg2rad(rpy_deg), axes="sxyz")
    return sapien.Pose(p=pos.tolist(), q=quat.tolist())


def _set_wrist_camera_pose(env, pos: np.ndarray, rpy_deg: np.ndarray, fov: float) -> None:
    sensor = env.unwrapped._sensors[WRIST_CAMERA_NAME]
    pose = _pose_from_state(pos, rpy_deg)
    sensor.config.pose = sensor.config.pose.create(pose)
    sensor.camera.set_local_pose(pose)
    sensor.camera.set_fovy(float(fov), compute_x=True)
    sensor.camera._cached_local_pose = None
    sensor.camera._cached_model_matrix = None
    sensor.camera._cached_extrinsic_matrix = None
    sensor.camera._cached_intrinsic_matrix = None
    sensor.config.fov = float(fov)
    env.unwrapped.scene.update_render(update_sensors=True, update_human_render_cameras=True)
    local_pose = sensor.camera.get_local_pose()
    if hasattr(local_pose, "raw_pose"):
        local_raw = local_pose.raw_pose[0].detach().cpu().numpy().tolist()
    else:
        local_raw = list(local_pose.p) + list(local_pose.q)
    print(f"Applied wrist local pose={local_raw} fov={float(fov)}")


def _build_calib_env(args, pos: np.ndarray, rpy_deg: np.ndarray, fov: float):
    args.wrist_camera_pose = _pose_from_state(pos, rpy_deg)
    args.wrist_camera_fov = float(fov)
    return _build_env(args, camera_mode=args.observation_camera_mode)


def _rebuild_calib_env(
    env,
    args,
    pos: np.ndarray,
    rpy_deg: np.ndarray,
    fov: float,
    robot_base_pose_p: np.ndarray,
    robot_base_pose_q: np.ndarray,
    robot_init_qpos: np.ndarray,
):
    if env is not None:
        env.close()
    env = _build_calib_env(args, pos, rpy_deg, fov)
    _reset_current_state(env, args, robot_base_pose_p, robot_base_pose_q, robot_init_qpos, reconfigure=True)
    return env


def _current_frames(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = env.unwrapped.get_obs(env.unwrapped.get_info())
    sensor_data = obs["sensor_data"]
    scene = _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])
    wrist = _to_uint8_image(sensor_data[WRIST_CAMERA_NAME]["rgb"])
    composite = _compose_wrist_inset(scene, wrist)
    return scene, wrist, composite


def _save_snapshot(env, args, pos: np.ndarray, rpy_deg: np.ndarray, fov: float, index: int) -> None:
    scene, wrist, composite = _current_frames(env)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.image_prefix}_{index:03d}"
    paths = {
        "scene": out_dir / f"{prefix}_scene.png",
        "wrist": out_dir / f"{prefix}_wrist.png",
        "openvla": out_dir / f"{prefix}_openvla.png",
    }
    latest_paths = {
        "scene": out_dir / f"{args.image_prefix}_latest_scene.png",
        "wrist": out_dir / f"{args.image_prefix}_latest_wrist.png",
        "openvla": out_dir / f"{args.image_prefix}_latest_openvla.png",
    }

    iio.imwrite(paths["scene"], scene)
    iio.imwrite(paths["wrist"], wrist)
    iio.imwrite(paths["openvla"], composite)
    for name, path in latest_paths.items():
        iio.imwrite(path, {"scene": scene, "wrist": wrist, "openvla": composite}[name])

    quat = euler2quat(*np.deg2rad(rpy_deg), axes="sxyz")
    meta = "\n".join(
        [
            f"WRIST_CAMERA_LOCAL_P = {pos.astype(float).tolist()}",
            f"WRIST_CAMERA_LOCAL_Q = {quat.astype(float).tolist()}",
            f"WRIST_CAMERA_FOV = {float(fov)}",
            f"rpy_deg = {rpy_deg.astype(float).tolist()}",
        ]
    )
    meta_path = out_dir / f"{prefix}.txt"
    latest_meta_path = out_dir / f"{args.image_prefix}_latest.txt"
    meta_path.write_text(meta + "\n", encoding="utf-8")
    latest_meta_path.write_text(meta + "\n", encoding="utf-8")
    print(f"Saved {paths['wrist']}")
    print(f"Saved {paths['openvla']}")
    print(meta)


def _print_help() -> None:
    print("Commands:")
    print("  save                  save scene/wrist/openvla PNGs + constants")
    print("  show                  print current local position, RPY, quat, fov")
    print("  p dx dy dz            translate local camera position in meters")
    print("  r droll dpitch dyaw   rotate local camera in degrees")
    print("  fov delta             add to vertical FOV in radians")
    print("  step value            set translation step used by x/y/z shortcuts")
    print("  deg value             set rotation step used by roll/pitch/yaw shortcuts")
    print("  x+/x-/y+/y-/z+/z-     translate by current step")
    print("  roll+/roll-           roll by current deg step")
    print("  pitch+/pitch-         pitch by current deg step")
    print("  yaw+/yaw-             yaw by current deg step")
    print("  flipx/flipy/flipz     add 180 deg about local axis")
    print("  reset                 reset to script defaults")
    print("  help                  show this message")
    print("  quit                  exit")


@dataclass
class Args(BaseArgs):
    output_dir: str = str(WRIST_CAMERA_CALIBRATION_DIR)
    image_prefix: str = "wrist_camera"
    use_probe_objects: bool = True
    use_wrist_camera: bool = True
    clean_scene: bool = True
    live_viewer: bool = False
    sim_backend: str = "gpu"
    shader: str = "default"
    start_pos: tuple[float, float, float] = tuple(float(v) for v in WRIST_CAMERA_LOCAL_P)
    start_rpy_deg: tuple[float, float, float] = tuple(float(v) for v in _quat_to_rpy_deg(WRIST_CAMERA_LOCAL_Q))
    start_fov: float = float(WRIST_CAMERA_FOV)
    pos_step: float = 0.01
    rot_step_deg: float = 10.0
    probe_source_model_preset: str = ""
    randomize_probe_source_model: bool = False


def main(args: Args) -> None:
    args.use_wrist_camera = True
    if args.probe_source_model_preset.strip():
        choices = source_model_names_for_obj_set(args.probe_source_model_preset)
        if args.randomize_probe_source_model:
            args.probe_source_model_name = choices[int(np.random.default_rng(args.seed).integers(len(choices)))]
        else:
            args.probe_source_model_name = choices[0]
        print(f"Using probe source model: {args.probe_source_model_name}")
    robot_base_pose_p, robot_base_pose_q, robot_init_qpos = _load_state_file(args.load_state_file)
    robot_base_pose_p += np.array([args.base_dx, args.base_dy, args.base_dz], dtype=np.float32)
    _apply_grid_pair_override(args, robot_base_pose_p)

    pos = np.array(args.start_pos, dtype=np.float64)
    rpy_deg = np.array(args.start_rpy_deg, dtype=np.float64)
    fov = float(args.start_fov)
    pos_step = float(args.pos_step)
    rot_step_deg = float(args.rot_step_deg)

    env = None
    try:
        env = _rebuild_calib_env(env, args, pos, rpy_deg, fov, robot_base_pose_p, robot_base_pose_q, robot_init_qpos)
        save_idx = 0
        _save_snapshot(env, args, pos, rpy_deg, fov, save_idx)
        _print_help()

        while True:
            raw = input("wristcam> ").strip()
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()
            try:
                if cmd in {"quit", "exit", "q"}:
                    break
                if cmd == "help":
                    _print_help()
                    continue
                if cmd == "show":
                    quat = euler2quat(*np.deg2rad(rpy_deg), axes="sxyz")
                    print(f"pos={pos.tolist()}")
                    print(f"rpy_deg={rpy_deg.tolist()}")
                    print(f"quat={quat.tolist()}")
                    print(f"fov={fov}")
                    continue
                if cmd == "save":
                    save_idx += 1
                    _save_snapshot(env, args, pos, rpy_deg, fov, save_idx)
                    continue
                if cmd == "reset":
                    pos = np.array(args.start_pos, dtype=np.float64)
                    rpy_deg = np.array(args.start_rpy_deg, dtype=np.float64)
                    fov = float(args.start_fov)
                elif cmd == "p":
                    pos += np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                elif cmd == "r":
                    rpy_deg += np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                elif cmd == "fov":
                    fov = max(0.1, fov + float(parts[1]))
                elif cmd == "step":
                    pos_step = float(parts[1])
                    print(f"pos_step={pos_step}")
                    continue
                elif cmd == "deg":
                    rot_step_deg = float(parts[1])
                    print(f"rot_step_deg={rot_step_deg}")
                    continue
                elif cmd in {"x+", "x-", "y+", "y-", "z+", "z-"}:
                    axis = {"x": 0, "y": 1, "z": 2}[cmd[0]]
                    pos[axis] += pos_step if cmd.endswith("+") else -pos_step
                elif cmd in {"roll+", "roll-", "pitch+", "pitch-", "yaw+", "yaw-"}:
                    axis = {"roll": 0, "pitch": 1, "yaw": 2}[cmd[:-1]]
                    rpy_deg[axis] += rot_step_deg if cmd.endswith("+") else -rot_step_deg
                elif cmd in {"flipx", "flipy", "flipz"}:
                    axis = {"flipx": 0, "flipy": 1, "flipz": 2}[cmd]
                    rpy_deg[axis] += 180.0
                else:
                    print("Unknown command. Type 'help'.")
                    continue

                env = _rebuild_calib_env(
                    env,
                    args,
                    pos,
                    rpy_deg,
                    fov,
                    robot_base_pose_p,
                    robot_base_pose_q,
                    robot_init_qpos,
                )
                save_idx += 1
                _save_snapshot(env, args, pos, rpy_deg, fov, save_idx)
            except Exception as exc:
                print(f"ERROR: {exc}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
