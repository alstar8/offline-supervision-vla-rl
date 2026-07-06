import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.visualization import images_to_video
from real2sim.openreal2sim_validation import (
    PROBE_PLATE_Z_OFFSET,
    SOURCE_SPAWN_EXTRA_Z,
    observation_camera_override,
)


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


def _build_env(args) -> BaseEnv:
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs={
            "shader_pack": args.shader,
            "3rd_view_camera": observation_camera_override(args.observation_camera_mode),
        },
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
    )


def _zero_action(env: BaseEnv) -> np.ndarray:
    return np.zeros_like(np.asarray(env.action_space.sample(), dtype=np.float32), dtype=np.float32)


def _load_failure_records(failure_dir: Path) -> list[dict]:
    records = []
    for path in sorted(failure_dir.glob("failure_pair_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_source_json"] = str(path)
        records.append(data)
    return records


def _sample_records(records: list[dict], max_cases: int, priority_carrot_ids: list[int]) -> list[dict]:
    if len(records) <= max_cases:
        return records

    chosen = []
    used = set()
    for carrot_id in priority_carrot_ids:
        for idx, record in enumerate(records):
            if idx in used:
                continue
            if int(record["carrot_grid_id"]) == carrot_id:
                chosen.append(record)
                used.add(idx)
                break
        if len(chosen) >= max_cases:
            return chosen

    remaining_slots = max_cases - len(chosen)
    remaining_indices = [idx for idx in range(len(records)) if idx not in used]
    if remaining_slots <= 0:
        return chosen
    if len(remaining_indices) <= remaining_slots:
        chosen.extend(records[idx] for idx in remaining_indices)
        return chosen

    sample_positions = np.linspace(0, len(remaining_indices) - 1, remaining_slots)
    for pos in sample_positions:
        idx = remaining_indices[int(round(pos))]
        if idx not in used:
            chosen.append(records[idx])
            used.add(idx)
    return chosen[:max_cases]


def _select_records_by_pair_ids(records: list[dict], pair_ids: list[int]) -> list[dict]:
    if not pair_ids:
        return records
    pair_id_set = set(pair_ids)
    return [record for record in records if int(record["pair_idx"]) in pair_id_set]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-dir", default="real2sim/spawn_stability/failures")
    parser.add_argument("--output-dir", default="real2sim/spawn_stability/rerendered_failures")
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--observation-camera-mode", default="manual_best")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-ids", type=int, nargs="*", default=[])
    parser.add_argument("--priority-carrot-ids", type=int, nargs="*", default=[6])
    parser.add_argument("--use-current-spawn-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-debug-markers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-spawn-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-far-away", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(args):
    failure_dir = Path(args.failure_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_failure_records(failure_dir)
    records = _select_records_by_pair_ids(records, args.pair_ids)
    chosen = _sample_records(records, args.max_cases, args.priority_carrot_ids)
    print(
        f"Rerendering {len(chosen)} failure cases from {len(records)} records "
        f"into {output_dir}"
    )

    env = _build_env(args)
    try:
        zero_action = _zero_action(env)
        manifest = []
        for idx, record in enumerate(chosen, start=1):
            carrot_id = int(record["carrot_grid_id"])
            plate_id = int(record["plate_grid_id"])
            pair_idx = int(record["pair_idx"])
            carrot_spawn_pose_p = list(record["carrot_spawn_pose_p"])
            plate_spawn_pose_p = list(record["plate_spawn_pose_p"])
            if args.use_current_spawn_z:
                table_z = float(record["table_z"])
                carrot_spawn_pose_p[2] = table_z + float(SOURCE_SPAWN_EXTRA_Z)
                plate_spawn_pose_p[2] = table_z + float(PROBE_PLATE_Z_OFFSET)
            print(
                f"[{idx}/{len(chosen)}] pair={pair_idx:04d} "
                f"c={carrot_id:02d} p={plate_id:02d}"
            )
            obs, _ = env.reset(
                seed=args.seed,
                options={
                    "reconfigure": True,
                    "load_background": args.load_background,
                    "use_probe_objects": True,
                    "show_debug_markers": args.show_debug_markers,
                    "show_spawn_grid": args.show_spawn_grid,
                    "robot_far_away": args.robot_far_away,
                    "spawn_grid_center_xy": record["grid_center_xy"],
                    "spawn_grid_size_xy": record["grid_size_xy"],
                    "probe_source_model_name": record.get("source_model_name", "001_carrot_simpler"),
                    "probe_carrot_pose_p": carrot_spawn_pose_p,
                    "probe_carrot_pose_q": record["carrot_spawn_pose_q"],
                    "probe_plate_pose_p": plate_spawn_pose_p,
                    "probe_plate_pose_q": record["plate_spawn_pose_q"],
                },
            )

            frames = [_extract_obs_frame(obs)]
            for _ in range(args.settle_steps):
                obs, _, terminated, truncated, _ = env.step(zero_action)
                frames.append(_extract_obs_frame(obs))
                if bool(torch.as_tensor(terminated).any().item()) or bool(
                    torch.as_tensor(truncated).any().item()
                ):
                    break

            stem = f"rerender_pair_{pair_idx:04d}_c{carrot_id:02d}_p{plate_id:02d}"
            images_to_video(frames, str(output_dir), stem, fps=args.fps, verbose=False)
            manifest.append(
                {
                    "pair_idx": pair_idx,
                    "carrot_grid_id": carrot_id,
                    "plate_grid_id": plate_id,
                    "source_model_name": record.get("source_model_name", "001_carrot_simpler"),
                    "source_json": record["_source_json"],
                    "output_video": str(output_dir / f"{stem}.mp4"),
                    "use_current_spawn_z": args.use_current_spawn_z,
                    "carrot_spawn_pose_p": carrot_spawn_pose_p,
                    "plate_spawn_pose_p": plate_spawn_pose_p,
                    "warnings": record.get("warnings", []),
                }
            )

        (output_dir / "manifest.json").write_text(
            json.dumps({"cases": manifest}, indent=2), encoding="utf-8"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(build_parser().parse_args())
