from dataclasses import dataclass
import ast
import csv
import json
from pathlib import Path
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import tyro
from transforms3d.axangles import axangle2mat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.visualization import images_to_video
from real2sim.debug_paths import resolve_spawn_grid_state_path
from real2sim.openreal2sim_validation import (
    AIRI_TABLE_EMPTY_ASSET_DIR,
    PROBE_CARROT_POSE_Q,
    PROBE_PLATE_POSE_Q,
    ROBOT_BASE_POSE_P,
    RC5AeroHandRightOpenReal2SimValidation,
    SOURCE_SPAWN_EXTRA_Z,
    available_source_model_names,
    default_spawn_grid_state,
    observation_camera_override,
    plate_xy_half_extent,
    plate_resting_center_z_offset,
    source_xy_half_extent,
    source_resting_center_z_offset,
)


def _load_spawn_grid_state(path: str, scene_asset_dir: str) -> dict[str, np.ndarray]:
    if not path:
        return default_spawn_grid_state(scene_asset_dir)
    state: dict[str, object] = {}
    resolved_path = resolve_spawn_grid_state_path(path)
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key.strip()] = ast.literal_eval(value.strip())
    return {
        "center_xy": np.array(state["center_xy"], dtype=np.float32),
        "size_xy": np.array(state["size_xy"], dtype=np.float32),
        "yaw_deg": np.array([float(state.get("yaw_deg", 0.0))], dtype=np.float32),
        "z": np.array([float(state["z"])], dtype=np.float32),
    }


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
        available = sorted(sensor_data.keys())
        raise KeyError(
            "Expected observation camera '3rd_view_camera' was not found. "
            f"Available sensor_data keys: {available}"
        )
    return _to_uint8_image(sensor_data["3rd_view_camera"]["rgb"])


def _build_env(args, obs_camera_mode: str) -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": observation_camera_override(
                obs_camera_mode,
                scene_asset_dir=args.scene_asset_dir,
            ),
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        scene_asset_dir=args.scene_asset_dir,
        robot_uids=RC5AeroHandRightOpenReal2SimValidation.uid,
    )


def _zero_action(env: BaseEnv) -> np.ndarray:
    return np.zeros_like(np.asarray(env.action_space.sample(), dtype=np.float32), dtype=np.float32)


def _sample_video_indices(count: int, max_videos: int) -> set[int]:
    if count <= max_videos:
        return set(range(count))
    raw = np.linspace(0, count - 1, max_videos)
    return set(int(round(v)) for v in raw.tolist())


def _make_grid(center_xy: np.ndarray, size_xy: np.ndarray, steps_x: int, steps_y: int, yaw_deg: float) -> np.ndarray:
    xs = np.linspace(center_xy[0] - size_xy[0] / 2.0, center_xy[0] + size_xy[0] / 2.0, steps_x)
    ys = np.linspace(center_xy[1] - size_xy[1] / 2.0, center_xy[1] + size_xy[1] / 2.0, steps_y)
    rot = axangle2mat([0.0, 0.0, 1.0], np.deg2rad(float(yaw_deg)))[:2, :2]
    pts = []
    for x in xs:
        for y in ys:
            local = np.array([float(x - center_xy[0]), float(y - center_xy[1])], dtype=np.float32)
            rotated = rot @ local
            pts.append([float(center_xy[0] + rotated[0]), float(center_xy[1] + rotated[1])])
    return np.array(pts, dtype=np.float32)


@dataclass
class Args:
    output_dir: str = "real2sim/spawn_stability"
    scene_asset_dir: str = str(AIRI_TABLE_EMPTY_ASSET_DIR)
    spawn_grid_state_file: str = ""
    video_dir_name: str = "videos"
    failure_dir_name: str = "failures"
    summary_json: str = "summary.json"
    summary_csv: str = "summary.csv"
    sim_backend: str = "gpu"
    shader: str = "default"
    observation_camera_mode: str = "manual_best"
    seed: int = 0
    steps_x: int = 6
    steps_y: int = 6
    settle_steps: int = 20
    min_pair_distance_xy: float = 0.10
    max_videos: int = 12
    load_background: bool = True
    show_debug_markers: bool = False
    show_spawn_grid: bool = True
    robot_far_away: bool = False
    safe_robot_during_spawn: bool = True
    min_robot_clearance_xy: float = 0.18
    progress_every: int = 25
    warn_drift_xy: float = 0.05
    warn_drift_z: float = 0.03
    warn_speed: float = 0.10
    max_failure_videos: int = 10
    priority_failure_carrot_ids: Tuple[int, ...] = (6,)
    num_random_source_models: int = 8


def main(args: Args):
    output_dir = Path(args.output_dir)
    video_dir = output_dir / args.video_dir_name
    failure_dir = output_dir / args.failure_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    grid = _load_spawn_grid_state(args.spawn_grid_state_file, args.scene_asset_dir)
    center_xy = np.array(grid["center_xy"], dtype=np.float32)
    size_xy = np.array(grid["size_xy"], dtype=np.float32)
    yaw_deg = float(grid["yaw_deg"][0])
    footprint_margin = max(
        source_xy_half_extent(PROBE_CARROT_POSE_Q),
        plate_xy_half_extent(PROBE_PLATE_POSE_Q),
    )
    usable_size_xy = np.maximum(size_xy - 2.0 * footprint_margin, 0.01)
    table_z = float(grid["z"][0])
    source_spawn_z = table_z + SOURCE_SPAWN_EXTRA_Z
    plate_rest_z = table_z + plate_resting_center_z_offset(PROBE_PLATE_POSE_Q)

    all_source_models = available_source_model_names()
    rng = np.random.default_rng(args.seed)
    if args.num_random_source_models >= len(all_source_models):
        sampled_source_models = all_source_models
    else:
        sampled_source_models = sorted(
            rng.choice(all_source_models, size=args.num_random_source_models, replace=False).tolist()
        )

    grid_points = _make_grid(center_xy, usable_size_xy, args.steps_x, args.steps_y, yaw_deg=yaw_deg)
    pairs = []
    robot_base_xy = np.array(ROBOT_BASE_POSE_P[:2], dtype=np.float32)
    for i, carrot_xy in enumerate(grid_points):
        for j, plate_xy in enumerate(grid_points):
            if i == j:
                continue
            if np.linalg.norm(plate_xy - carrot_xy) <= args.min_pair_distance_xy:
                continue
            if np.linalg.norm(carrot_xy - robot_base_xy) <= args.min_robot_clearance_xy:
                continue
            if np.linalg.norm(plate_xy - robot_base_xy) <= args.min_robot_clearance_xy:
                continue
            pairs.append((i, j, carrot_xy, plate_xy))

    total_pairs = len(pairs)
    print(
        "Spawn stability:"
        f" grid={args.steps_x}x{args.steps_y},"
        f" pairs={total_pairs},"
        f" settle_steps={args.settle_steps},"
        f" sampled_videos={min(args.max_videos, total_pairs)},"
        f" source_models={sampled_source_models}"
    )

    env = _build_env(args, args.observation_camera_mode)
    try:
        zero_action = _zero_action(env)
        video_indices = _sample_video_indices(len(pairs), args.max_videos)
        results = []
        warning_count = 0
        warning_examples = []
        saved_failure_videos = 0
        saved_priority_failure_videos = set()
        start_time = time.perf_counter()
        last_progress_time = start_time

        for pair_idx, (carrot_id, plate_id, carrot_xy, plate_xy) in enumerate(pairs):
            current_idx = pair_idx + 1
            source_model_name = sampled_source_models[pair_idx % len(sampled_source_models)]
            source_rest_z = table_z + source_resting_center_z_offset(
                PROBE_CARROT_POSE_Q, source_model_name
            )
            carrot_pose_p = [float(carrot_xy[0]), float(carrot_xy[1]), float(source_spawn_z)]
            plate_pose_p = [float(plate_xy[0]), float(plate_xy[1]), float(plate_rest_z)]
            obs, _ = env.reset(
                seed=args.seed,
                options={
                    "reconfigure": True,
                    "load_background": args.load_background,
                    "use_probe_objects": True,
                    "show_debug_markers": args.show_debug_markers,
                    "show_spawn_grid": args.show_spawn_grid,
                    "robot_far_away": args.robot_far_away,
                    "safe_robot_during_spawn": args.safe_robot_during_spawn,
                    "spawn_grid_center_xy": center_xy.tolist(),
                    "spawn_grid_size_xy": size_xy.tolist(),
                    "spawn_grid_yaw_deg": yaw_deg,
                    "probe_source_model_name": source_model_name,
                    "probe_carrot_pose_p": carrot_pose_p,
                    "probe_carrot_pose_q": PROBE_CARROT_POSE_Q,
                    "probe_plate_pose_p": plate_pose_p,
                    "probe_plate_pose_q": PROBE_PLATE_POSE_Q,
                },
            )

            carrot_start = env.unwrapped.probe_carrot.pose.p[0].detach().cpu().numpy().copy()
            plate_start = env.unwrapped.probe_plate.pose.p[0].detach().cpu().numpy().copy()

            capture_frames = pair_idx in video_indices or args.max_failure_videos > 0
            frames = []
            if capture_frames:
                frames.append(_extract_obs_frame(obs))

            for _ in range(args.settle_steps):
                obs, _, terminated, truncated, _ = env.step(zero_action)
                if capture_frames:
                    frames.append(_extract_obs_frame(obs))
                if bool(torch.as_tensor(terminated).any().item()) or bool(
                    torch.as_tensor(truncated).any().item()
                ):
                    break

            carrot_end = env.unwrapped.probe_carrot.pose.p[0].detach().cpu().numpy().copy()
            plate_end = env.unwrapped.probe_plate.pose.p[0].detach().cpu().numpy().copy()
            carrot_lin = env.unwrapped.probe_carrot.linear_velocity[0].detach().cpu().numpy().copy()
            plate_lin = env.unwrapped.probe_plate.linear_velocity[0].detach().cpu().numpy().copy()

            carrot_expected_rest = np.array([carrot_xy[0], carrot_xy[1], source_rest_z], dtype=np.float32)
            plate_expected_rest = np.array([plate_xy[0], plate_xy[1], plate_rest_z], dtype=np.float32)
            carrot_drift_xy = float(np.linalg.norm(carrot_end[:2] - carrot_expected_rest[:2]))
            plate_drift_xy = float(np.linalg.norm(plate_end[:2] - plate_expected_rest[:2]))
            carrot_drift_z = float(carrot_end[2] - carrot_expected_rest[2])
            plate_drift_z = float(plate_end[2] - plate_expected_rest[2])
            carrot_speed = float(np.linalg.norm(carrot_lin))
            plate_speed = float(np.linalg.norm(plate_lin))

            result = {
                "pair_idx": pair_idx,
                "carrot_grid_id": carrot_id,
                "plate_grid_id": plate_id,
                "source_model_name": source_model_name,
                "carrot_xy": carrot_xy.tolist(),
                "plate_xy": plate_xy.tolist(),
                "carrot_start": carrot_start.tolist(),
                "plate_start": plate_start.tolist(),
                "carrot_expected_rest": carrot_expected_rest.tolist(),
                "plate_expected_rest": plate_expected_rest.tolist(),
                "carrot_end": carrot_end.tolist(),
                "plate_end": plate_end.tolist(),
                "carrot_drift_xy": carrot_drift_xy,
                "plate_drift_xy": plate_drift_xy,
                "carrot_drift_z": carrot_drift_z,
                "plate_drift_z": plate_drift_z,
                "carrot_speed": carrot_speed,
                "plate_speed": plate_speed,
                "video_saved": pair_idx in video_indices,
            }
            results.append(result)

            warnings = []
            if carrot_drift_xy > args.warn_drift_xy:
                warnings.append(f"carrot_drift_xy={carrot_drift_xy:.4f}")
            if plate_drift_xy > args.warn_drift_xy:
                warnings.append(f"plate_drift_xy={plate_drift_xy:.4f}")
            if abs(carrot_drift_z) > args.warn_drift_z:
                warnings.append(f"carrot_drift_z={carrot_drift_z:.4f}")
            if abs(plate_drift_z) > args.warn_drift_z:
                warnings.append(f"plate_drift_z={plate_drift_z:.4f}")
            if carrot_speed > args.warn_speed:
                warnings.append(f"carrot_speed={carrot_speed:.4f}")
            if plate_speed > args.warn_speed:
                warnings.append(f"plate_speed={plate_speed:.4f}")
            if warnings:
                result["warnings"] = warnings
                warning_count += 1
                if len(warning_examples) < 20:
                    warning_examples.append(
                        {
                            "pair_idx": pair_idx,
                            "carrot_grid_id": carrot_id,
                            "plate_grid_id": plate_id,
                            "warnings": warnings,
                        }
                    )
                print(
                    f"WARNING [{current_idx}/{total_pairs}] "
                    f"c={carrot_id:02d} p={plate_id:02d} model={source_model_name} "
                    + " ".join(warnings)
                )
                failure_stem = f"failure_pair_{pair_idx:04d}_c{carrot_id:02d}_p{plate_id:02d}"
                save_failure_video = False
                if carrot_id in args.priority_failure_carrot_ids and (
                    carrot_id not in saved_priority_failure_videos
                ):
                    save_failure_video = True
                    saved_priority_failure_videos.add(carrot_id)
                elif saved_failure_videos < args.max_failure_videos:
                    save_failure_video = True
                if save_failure_video:
                    if not frames:
                        frames.append(_extract_obs_frame(obs))
                    images_to_video(frames, str(failure_dir), failure_stem, fps=10, verbose=False)
                    saved_failure_videos += 1
                edge_min_margin = float(
                    min(
                        carrot_xy[0] - (center_xy[0] - size_xy[0] / 2.0),
                        (center_xy[0] + size_xy[0] / 2.0) - carrot_xy[0],
                        carrot_xy[1] - (center_xy[1] - size_xy[1] / 2.0),
                        (center_xy[1] + size_xy[1] / 2.0) - carrot_xy[1],
                        plate_xy[0] - (center_xy[0] - size_xy[0] / 2.0),
                        (center_xy[0] + size_xy[0] / 2.0) - plate_xy[0],
                        plate_xy[1] - (center_xy[1] - size_xy[1] / 2.0),
                        (center_xy[1] + size_xy[1] / 2.0) - plate_xy[1],
                    )
                )
                failure_record = {
                    "pair_idx": pair_idx,
                    "carrot_grid_id": carrot_id,
                    "plate_grid_id": plate_id,
                    "source_model_name": source_model_name,
                    "warnings": warnings,
                    "grid_center_xy": center_xy.tolist(),
                    "grid_size_xy": size_xy.tolist(),
                    "table_z": table_z,
                    "source_rest_z": source_rest_z,
                    "plate_rest_z": plate_rest_z,
                    "carrot_spawn_pose_p": carrot_pose_p,
                    "plate_spawn_pose_p": plate_pose_p,
                    "carrot_spawn_pose_q": PROBE_CARROT_POSE_Q,
                    "plate_spawn_pose_q": PROBE_PLATE_POSE_Q,
                    "carrot_start": carrot_start.tolist(),
                    "plate_start": plate_start.tolist(),
                    "carrot_expected_rest": carrot_expected_rest.tolist(),
                    "plate_expected_rest": plate_expected_rest.tolist(),
                    "carrot_end": carrot_end.tolist(),
                    "plate_end": plate_end.tolist(),
                    "carrot_drift_xy": carrot_drift_xy,
                    "plate_drift_xy": plate_drift_xy,
                    "carrot_drift_z": carrot_drift_z,
                    "plate_drift_z": plate_drift_z,
                    "carrot_speed": carrot_speed,
                    "plate_speed": plate_speed,
                    "pair_distance_xy": float(np.linalg.norm(plate_xy - carrot_xy)),
                    "edge_min_margin_xy": edge_min_margin,
                    "failure_video_saved": save_failure_video,
                }
                (failure_dir / f"{failure_stem}.json").write_text(
                    json.dumps(failure_record, indent=2), encoding="utf-8"
                )

            if pair_idx in video_indices:
                video_name = f"spawn_pair_{pair_idx:04d}_c{carrot_id:02d}_p{plate_id:02d}"
                images_to_video(frames, str(video_dir), video_name, fps=10, verbose=False)

            should_report = (
                current_idx == 1
                or current_idx == total_pairs
                or current_idx % max(args.progress_every, 1) == 0
            )
            if should_report:
                now = time.perf_counter()
                elapsed = now - start_time
                recent = now - last_progress_time
                rate = current_idx / elapsed if elapsed > 0 else 0.0
                remaining = total_pairs - current_idx
                eta_sec = remaining / rate if rate > 0 else 0.0
                print(
                    f"[{current_idx}/{total_pairs}] "
                    f"c={carrot_id:02d} p={plate_id:02d} model={source_model_name} "
                    f"carrot_drift_xy={carrot_drift_xy:.4f} "
                    f"plate_drift_xy={plate_drift_xy:.4f} "
                    f"elapsed={elapsed:.1f}s "
                    f"recent={recent:.1f}s "
                    f"eta={eta_sec:.1f}s"
                )
                last_progress_time = now

        summary = {
            "grid_center_xy": center_xy.tolist(),
            "grid_size_xy": size_xy.tolist(),
            "usable_grid_size_xy": usable_size_xy.tolist(),
            "grid_yaw_deg": yaw_deg,
            "table_z": table_z,
            "source_spawn_z": source_spawn_z,
            "source_spawn_extra_z": SOURCE_SPAWN_EXTRA_Z,
            "plate_rest_z": plate_rest_z,
            "sampled_source_models": sampled_source_models,
            "steps_x": args.steps_x,
            "steps_y": args.steps_y,
            "pair_count": len(results),
            "min_pair_distance_xy": args.min_pair_distance_xy,
            "min_robot_clearance_xy": args.min_robot_clearance_xy,
            "settle_steps": args.settle_steps,
            "warn_drift_xy": args.warn_drift_xy,
            "warn_drift_z": args.warn_drift_z,
            "warn_speed": args.warn_speed,
            "max_carrot_drift_xy": max((r["carrot_drift_xy"] for r in results), default=0.0),
            "max_plate_drift_xy": max((r["plate_drift_xy"] for r in results), default=0.0),
            "max_carrot_speed": max((r["carrot_speed"] for r in results), default=0.0),
            "max_plate_speed": max((r["plate_speed"] for r in results), default=0.0),
            "warning_count": warning_count,
            "warning_examples": warning_examples,
            "max_failure_videos": args.max_failure_videos,
            "priority_failure_carrot_ids": list(args.priority_failure_carrot_ids),
            "saved_failure_videos": saved_failure_videos,
            "results": results,
        }
        (output_dir / args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

        with (output_dir / args.summary_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pair_idx",
                    "carrot_grid_id",
                    "plate_grid_id",
                    "carrot_drift_xy",
                    "plate_drift_xy",
                    "carrot_drift_z",
                    "plate_drift_z",
                    "carrot_speed",
                    "plate_speed",
                    "video_saved",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow({k: row[k] for k in writer.fieldnames})

        print(
            "Finished:"
            f" pairs={len(results)},"
            f" warnings={warning_count},"
            f" max_carrot_drift_xy={summary['max_carrot_drift_xy']:.4f},"
            f" max_plate_drift_xy={summary['max_plate_drift_xy']:.4f},"
            f" max_carrot_speed={summary['max_carrot_speed']:.4f},"
            f" max_plate_speed={summary['max_plate_speed']:.4f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
