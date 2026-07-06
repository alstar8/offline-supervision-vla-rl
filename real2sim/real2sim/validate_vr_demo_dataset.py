import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro

from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory import utils as trajectory_utils
from real2sim.calibrate_rc5_pose import _extract_obs_frame
from real2sim.openreal2sim_validation import observation_camera_override


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def _load_optional_json(path: str) -> dict:
    if not path.strip():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _deep_update(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_json_camera_poses(value):
    """Convert JSON-serialized sapien poses into ManiSkill camera override lists."""
    if isinstance(value, dict):
        if set(value.keys()) == {"p", "q"}:
            p = value["p"]
            q = value["q"]
            if isinstance(p, list) and isinstance(q, list):
                return [*p, *q]
        return {k: _normalize_json_camera_poses(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_json_camera_poses(v) for v in value]
    return value


@dataclass
class Args:
    traj_path: str = ""
    json_path: str = ""
    session_dir: str = ""
    trajectory_name: str = "vr_human_demos"
    episode_id: int = -1
    max_steps: int = 80
    settle_steps: int = 0
    replay_attempts: int = 1
    vis: bool = False
    save_video: bool = False
    save_obs_video: bool = False
    obs_video_fps: int = 10
    save_rerendered_trajectory: bool = False
    rerendered_trajectory_name: str = "wrist_rerendered"
    use_first_env_state: bool = True
    use_env_states: bool = False
    output_dir: str = ""
    rerender_reconfigure: bool = False
    auto_reconfigure_assets: bool = True
    env_kwargs_json: str = ""
    reset_options_json: str = ""
    render_backend_override: str = ""
    sim_backend_override: str = ""
    shader_override: str = ""
    observation_camera_mode_override: str = ""
    scene_asset_dir_override: str = ""
    load_background: Optional[bool] = None
    show_debug_markers: Optional[bool] = None
    show_spawn_grid: Optional[bool] = None
    robot_far_away: Optional[bool] = None
    use_environment_map: Optional[bool] = None
    apply_robot_material_overrides: Optional[bool] = None


def _scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.reshape(-1)[0]
        if value.dtype == torch.bool:
            return bool(value.item())
        return float(value.item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
        if value.dtype == np.bool_:
            return bool(value.item())
        return float(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _scalar(value[0])
    if isinstance(value, (bool, int, float)):
        return value
    return None


def _format_info_metrics(info: dict) -> str:
    keys = [
        "success",
        "src_on_target",
        "is_src_obj_grasped",
        "carrot_to_plate_dist",
        "tcp_to_carrot_dist",
        "plate_contact_force",
        "consecutive_grasp",
    ]
    parts = []
    for key in keys:
        if key not in info:
            continue
        value = _scalar(info[key])
        if isinstance(value, bool):
            parts.append(f"{key}={int(value)}")
        elif isinstance(value, (float, int)):
            parts.append(f"{key}={float(value):.5f}")
    return " ".join(parts)


def _shard_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    if "_resume_" not in stem:
        return (0, 0, stem)
    suffix = stem.rsplit("_resume_", 1)[1]
    try:
        idx = int(suffix)
    except ValueError:
        idx = 0
    return (1, idx, stem)


def _resolve_shards(args: Args) -> list[tuple[Path, Path]]:
    if args.session_dir.strip():
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")
        base_path = session_dir / f"{args.trajectory_name}.h5"
        h5_paths = []
        if base_path.exists():
            h5_paths.append(base_path)
        h5_paths.extend(session_dir.glob(f"{args.trajectory_name}_resume_*.h5"))
        h5_paths = sorted(set(h5_paths), key=_shard_sort_key)
        shards = []
        for h5_path in h5_paths:
            json_path = h5_path.with_suffix(".json")
            if json_path.exists():
                shards.append((h5_path, json_path))
        if not shards:
            raise FileNotFoundError(
                f"No {args.trajectory_name}*.h5/json shards found in {session_dir}"
            )
        return shards

    if not args.traj_path.strip():
        raise ValueError("Either --traj-path or --session-dir is required.")
    traj_path = Path(args.traj_path)
    json_path = Path(args.json_path) if args.json_path else traj_path.with_suffix(".json")
    return [(traj_path, json_path)]


def _override_env_kwargs(args: Args, env_kwargs: dict) -> dict:
    updated = _normalize_json_camera_poses(dict(env_kwargs))
    if args.render_backend_override.strip():
        updated["render_backend"] = args.render_backend_override.strip()
    if args.sim_backend_override.strip():
        updated["sim_backend"] = args.sim_backend_override.strip()
    if args.scene_asset_dir_override.strip():
        updated["scene_asset_dir"] = args.scene_asset_dir_override.strip()
    if args.shader_override.strip():
        sensor_configs = dict(updated.get("sensor_configs", {}))
        sensor_configs["shader_pack"] = args.shader_override.strip()
        if args.observation_camera_mode_override.strip():
            sensor_configs["3rd_view_camera"] = observation_camera_override(
                args.observation_camera_mode_override.strip(),
                scene_asset_dir=updated.get("scene_asset_dir", None),
            )
        updated["sensor_configs"] = sensor_configs
        updated["human_render_camera_configs"] = {
            **dict(updated.get("human_render_camera_configs", {})),
            "shader_pack": args.shader_override.strip(),
        }
    elif args.observation_camera_mode_override.strip():
        sensor_configs = dict(updated.get("sensor_configs", {}))
        sensor_configs["3rd_view_camera"] = observation_camera_override(
            args.observation_camera_mode_override.strip(),
            scene_asset_dir=updated.get("scene_asset_dir", None),
        )
        updated["sensor_configs"] = sensor_configs
    return _deep_update(updated, _load_optional_json(args.env_kwargs_json))


def _override_reset_kwargs(args: Args, reset_kwargs: dict) -> dict:
    updated = dict(reset_kwargs)
    options = dict(updated.get("options", {}) or {})
    visual_override_requested = False
    if args.load_background is not None:
        options["load_background"] = bool(args.load_background)
        visual_override_requested = True
    if args.show_debug_markers is not None:
        options["show_debug_markers"] = bool(args.show_debug_markers)
        visual_override_requested = True
    if args.show_spawn_grid is not None:
        options["show_spawn_grid"] = bool(args.show_spawn_grid)
        visual_override_requested = True
    if args.robot_far_away is not None:
        options["robot_far_away"] = bool(args.robot_far_away)
        visual_override_requested = True
    if args.use_environment_map is not None:
        options["use_environment_map"] = bool(args.use_environment_map)
        visual_override_requested = True
    if args.apply_robot_material_overrides is not None:
        options["apply_robot_material_overrides"] = bool(args.apply_robot_material_overrides)
        visual_override_requested = True
    if args.rerender_reconfigure or visual_override_requested or args.reset_options_json.strip():
        options["reconfigure"] = True
    updated["options"] = _deep_update(options, _load_optional_json(args.reset_options_json))
    return updated


def _is_airi_cubes_replay(env_info: dict, reset_kwargs: dict) -> bool:
    env_id = str(env_info.get("env_id", "")).strip()
    if env_id in {"PutObjectOnPlateAiriCubesRecorder-v1", "PickUpAiriCubeRecorder-v1"}:
        return True
    env_kwargs = env_info.get("env_kwargs", {}) if isinstance(env_info.get("env_kwargs", {}), dict) else {}
    options = reset_kwargs.get("options", {}) if isinstance(reset_kwargs.get("options", {}), dict) else {}
    task_mode = str(options.get("task_mode", "")).strip().lower()
    scene_asset_dir = str(env_kwargs.get("scene_asset_dir", options.get("scene_asset_dir", ""))).replace("\\", "/").lower()
    return task_mode in {"airi_cube_pickup", "cube_pickup", "pick_cube"} or "airi_cubes" in scene_asset_dir


def _asset_names_from_reset_kwargs(reset_kwargs: dict) -> tuple[str | None, str | None]:
    options = dict(reset_kwargs.get("options", {}) or {})
    source_name = options.get("probe_source_model_name", None)
    plate_name = options.get("probe_plate_model_name", None)
    return (
        str(source_name) if source_name is not None else None,
        str(plate_name) if plate_name is not None else None,
    )


def _set_reset_reconfigure(reset_kwargs: dict, reconfigure: bool) -> dict:
    updated = dict(reset_kwargs)
    options = dict(updated.get("options", {}) or {})
    options["reconfigure"] = bool(reconfigure)
    updated["options"] = options
    return updated


def _set_env_state(env, state: dict) -> None:
    target = getattr(env, "base_env", None)
    if target is None:
        target = env.unwrapped
    target.set_state_dict(state)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_rerendered_episode(
    *,
    h5_out: h5py.File,
    json_out: dict,
    episode_id: int,
    source_shard: Path,
    source_episode: dict,
    actions: np.ndarray,
    images: list[np.ndarray],
    success: list[bool],
    reset_kwargs: dict,
    successful_attempt: int,
) -> None:
    if len(images) != len(actions) + 1:
        raise ValueError(
            f"Expected {len(actions) + 1} rerendered images for episode {episode_id}, got {len(images)}."
        )
    if len(success) != len(actions):
        raise ValueError(
            f"Expected {len(actions)} success values for episode {episode_id}, got {len(success)}."
        )

    traj_id = f"traj_{episode_id}"
    if traj_id in h5_out:
        raise ValueError(f"Duplicate rerendered trajectory id {traj_id} in {h5_out.filename}.")
    group = h5_out.create_group(traj_id, track_order=True)
    obs_group = group.create_group("obs", track_order=True)
    obs_group.create_dataset(
        "openvla_image",
        data=np.asarray(images, dtype=np.uint8),
        dtype=np.uint8,
        compression="gzip",
        compression_opts=4,
    )
    group.create_dataset("actions", data=np.asarray(actions, dtype=np.float32), dtype=np.float32)
    group.create_dataset("success", data=np.asarray(success, dtype=bool), dtype=bool)
    group.create_dataset("terminated", data=np.zeros((len(actions),), dtype=bool), dtype=bool)
    group.create_dataset("truncated", data=np.zeros((len(actions),), dtype=bool), dtype=bool)
    group.create_dataset("rewards", data=np.zeros((len(actions),), dtype=np.float32), dtype=np.float32)

    json_out["episodes"].append(
        _to_jsonable(
            {
                "episode_id": int(episode_id),
                "source_shard": source_shard.name,
                "source_episode_id": int(source_episode["episode_id"]),
                "control_mode": source_episode.get("control_mode"),
                "elapsed_steps": int(len(actions)),
                "success": bool(success[-1]) if success else False,
                "successful_attempt": int(successful_attempt),
                "reset_kwargs": reset_kwargs,
            }
        )
    )


def main(args: Args):
    shards = _resolve_shards(args)
    total_episodes = 0
    total_success = 0
    output_dir = Path(args.output_dir) if args.output_dir else None
    validation_records = []

    for traj_path, json_path in shards:
        meta = _load_json(json_path)

        env_info = meta["env_info"]
        env_kwargs = _override_env_kwargs(args, dict(env_info["env_kwargs"]))
        env = gym.make(
            env_info["env_id"],
            **env_kwargs,
        )

        record_env = None
        shard_output_dir = output_dir if output_dir is not None else traj_path.parent / "validation_videos"
        if args.save_video:
            shard_output_dir.mkdir(parents=True, exist_ok=True)
            record_env = RecordEpisode(
                env,
                output_dir=str(shard_output_dir),
                save_trajectory=False,
                save_video=True,
                save_on_reset=False,
                recording_camera_name="3rd_view_camera",
            )
            env = record_env

        rerender_h5 = None
        rerender_meta = None
        rerender_json_path = None
        next_rerender_episode_id = 0
        if args.save_rerendered_trajectory:
            shard_output_dir.mkdir(parents=True, exist_ok=True)
            rerender_stem = f"{traj_path.stem}_{args.rerendered_trajectory_name}"
            rerender_h5_path = shard_output_dir / f"{rerender_stem}.h5"
            rerender_json_path = shard_output_dir / f"{rerender_stem}.json"
            rerender_h5 = h5py.File(rerender_h5_path, "w")
            rerender_meta = {
                "source_traj_path": str(traj_path),
                "source_json_path": str(json_path),
                "env_info": _to_jsonable(
                    {
                        "env_id": env_info["env_id"],
                        "env_kwargs": env_kwargs,
                    }
                ),
                "obs_format": {
                    "image_key": "obs/openvla_image",
                    "description": "Composed OpenVLA input image with wrist-camera inset.",
                    "image_count": "num_actions + 1",
                },
                "episodes": [],
            }

        episodes = meta["episodes"]
        if args.episode_id >= 0:
            episodes = [ep for ep in episodes if int(ep["episode_id"]) == int(args.episode_id)]

        print(f"replaying shard={traj_path.name} episodes={len(episodes)}")
        shard_success_count = 0
        current_source_model_name = None
        current_plate_model_name = None
        with h5py.File(traj_path, "r") as h5_file:
            for ep in episodes:
                ep_id = int(ep["episode_id"])
                traj = h5_file[f"traj_{ep_id}"]
                actions = np.asarray(traj["actions"], dtype=np.float32)
                env_states = (
                    trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
                    if "env_states" in traj and (args.use_first_env_state or args.use_env_states)
                    else None
                )
                base_reset_kwargs = _override_reset_kwargs(args, dict(ep["reset_kwargs"]))
                next_source_model_name, next_plate_model_name = _asset_names_from_reset_kwargs(base_reset_kwargs)

                final_info = {}
                success = False
                successful_attempt = -1
                successful_images = None
                successful_step_success = None
                successful_reset_kwargs = None
                attempts = int(args.replay_attempts)
                attempt_idx = 0
                while attempts <= 0 or attempt_idx < attempts:
                    is_airi_cubes_replay = _is_airi_cubes_replay(env_info, base_reset_kwargs)
                    needs_asset_reconfigure = (
                        bool(args.auto_reconfigure_assets)
                        and (
                            current_source_model_name != next_source_model_name
                            or current_plate_model_name != next_plate_model_name
                        )
                    )
                    reset_kwargs = _set_reset_reconfigure(
                        base_reset_kwargs,
                        bool(args.rerender_reconfigure)
                        or (attempt_idx == 0 and (needs_asset_reconfigure or is_airi_cubes_replay)),
                    )
                    obs, info = env.reset(**reset_kwargs)
                    if info.get("reconfigure", False):
                        current_source_model_name = next_source_model_name
                        current_plate_model_name = next_plate_model_name
                    if env_states:
                        _set_env_state(env, env_states[0])
                        obs = env.unwrapped.get_obs()
                        final_info = env.unwrapped.get_info()
                    else:
                        final_info = info
                    capture_rerender = bool(args.save_obs_video or args.save_rerendered_trajectory)
                    wrist_inset_bottom_right = is_airi_cubes_replay
                    obs_video_frames = [
                        _extract_obs_frame(obs, wrist_inset_bottom_right=wrist_inset_bottom_right)
                    ] if capture_rerender else []
                    step_success = []
                    replay_states = env_states[1:] if args.use_env_states and env_states else None
                    for step_idx, action in enumerate(actions):
                        obs, _, _, _, final_info = env.step(
                            torch.tensor(action[np.newaxis, :], dtype=torch.float32, device=env.unwrapped.device)
                        )
                        if replay_states is not None and step_idx < len(replay_states):
                            _set_env_state(env, replay_states[step_idx])
                            obs = env.unwrapped.get_obs()
                            final_info = env.unwrapped.get_info()
                        step_success.append(_as_bool(final_info.get("success", False)))
                        if capture_rerender:
                            obs_video_frames.append(
                                _extract_obs_frame(obs, wrist_inset_bottom_right=wrist_inset_bottom_right)
                            )
                        if args.vis:
                            env.unwrapped.render_human()
                    if int(args.settle_steps) > 0:
                        zero_action = torch.zeros(
                            (1, actions.shape[1]), dtype=torch.float32, device=env.unwrapped.device
                        )
                        for _ in range(int(args.settle_steps)):
                            obs, _, _, _, final_info = env.step(zero_action)
                            if args.save_obs_video:
                                obs_video_frames.append(
                                    _extract_obs_frame(obs, wrist_inset_bottom_right=wrist_inset_bottom_right)
                                )
                            if args.vis:
                                env.unwrapped.render_human()

                    attempt_success = _as_bool(final_info.get("success", False))
                    attempt_label = "until_success" if attempts <= 0 else str(attempts)
                    print(
                        f"episode={ep_id} attempt={attempt_idx + 1}/{attempt_label} "
                        f"success={int(attempt_success)} {_format_info_metrics(final_info)}"
                    )
                    if record_env is not None:
                        record_env.flush_video(
                            name=f"{traj_path.stem}_episode_{ep_id:04d}_attempt_{attempt_idx + 1:02d}"
                        )
                    if args.save_obs_video:
                        shard_output_dir.mkdir(parents=True, exist_ok=True)
                        images_to_video(
                            obs_video_frames,
                            str(shard_output_dir),
                            f"{traj_path.stem}_episode_{ep_id:04d}_attempt_{attempt_idx + 1:02d}_openvla_obs",
                            fps=int(args.obs_video_fps),
                            verbose=False,
                        )
                    if attempt_success:
                        success = True
                        successful_attempt = attempt_idx + 1
                        if args.save_rerendered_trajectory:
                            successful_images = list(obs_video_frames)
                            successful_step_success = list(step_success)
                            successful_reset_kwargs = reset_kwargs
                        break
                    attempt_idx += 1

                within_budget = len(actions) <= int(args.max_steps)
                status = "OK" if success and within_budget else "FAIL"
                print(
                    f"episode={ep_id} steps={len(actions)} within_budget={int(within_budget)} "
                    f"success={int(success)} successful_attempt={successful_attempt} status={status}"
                )
                if success and within_budget:
                    shard_success_count += 1
                    if args.save_rerendered_trajectory and rerender_h5 is not None and rerender_meta is not None:
                        _write_rerendered_episode(
                            h5_out=rerender_h5,
                            json_out=rerender_meta,
                            episode_id=next_rerender_episode_id,
                            source_shard=traj_path,
                            source_episode=ep,
                            actions=actions,
                            images=successful_images or [],
                            success=successful_step_success or [],
                            reset_kwargs=successful_reset_kwargs or base_reset_kwargs,
                            successful_attempt=successful_attempt,
                        )
                        next_rerender_episode_id += 1
                validation_records.append(
                    _to_jsonable(
                        {
                            "source_shard": traj_path.name,
                            "source_episode_id": ep_id,
                            "steps": int(len(actions)),
                            "within_budget": bool(within_budget),
                            "success": bool(success),
                            "successful_attempt": int(successful_attempt),
                            "status": status,
                            "rerendered_episode_id": int(next_rerender_episode_id - 1)
                            if success and within_budget and args.save_rerendered_trajectory
                            else None,
                            "final_metrics": {
                                key: _scalar(final_info[key])
                                for key in [
                                    "success",
                                    "src_on_target",
                                    "is_src_obj_grasped",
                                    "carrot_to_plate_dist",
                                    "tcp_to_carrot_dist",
                                    "plate_contact_force",
                                    "consecutive_grasp",
                                ]
                                if key in final_info
                            },
                        }
                    )
                )

        print(f"shard={traj_path.name} validated={len(episodes)} successful={shard_success_count}")
        total_episodes += len(episodes)
        total_success += shard_success_count
        if rerender_h5 is not None:
            rerender_h5.close()
        if rerender_meta is not None and rerender_json_path is not None:
            rerender_json_path.write_text(json.dumps(rerender_meta, indent=2), encoding="utf-8")
        env.close()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "validated": int(total_episodes),
            "successful": int(total_success),
            "failed": int(total_episodes - total_success),
            "records": validation_records,
            "failed_records": [record for record in validation_records if record["status"] != "OK"],
        }
        (output_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(f"validated={total_episodes} successful={total_success}")


if __name__ == "__main__":
    main(tyro.cli(Args))
