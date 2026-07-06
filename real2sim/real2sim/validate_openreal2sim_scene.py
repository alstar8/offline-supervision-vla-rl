from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.visualization import images_to_video
from real2sim.debug_paths import VIDEOS_DIR
from real2sim.openreal2sim_validation import (
    AIRI_TABLE_EMPTY_ASSET_DIR,
    RC5AeroHandRightOpenReal2SimValidation,
    _load_scene_json,
    default_spawn_grid_state,
    observation_camera_override,
)

@dataclass
class Args:
    output_dir: str = str(VIDEOS_DIR)
    scene_asset_dir: str = str(AIRI_TABLE_EMPTY_ASSET_DIR)
    video_name: str = "openreal2sim_validation_obs"
    sanity_video_name: str = "openreal2sim_validation_sanity"
    seed: int = 0
    num_steps: int = 90
    sim_backend: str = "gpu"
    use_probe_objects: bool = True
    load_background: bool = True
    show_debug_markers: bool = False
    show_spawn_grid: bool = True
    robot_far_away: bool = False
    observation_camera_mode: str = "manual_best"
    shader: str = "default"
    debug: bool = False
    move_robot: bool = True


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


def _extract_render_frame(env: BaseEnv, camera_name: str) -> np.ndarray:
    return _to_uint8_image(env.render_rgb_array(camera_name))


def _debug_target(scene_asset_dir: str) -> np.ndarray:
    scene = _load_scene_json(scene_asset_dir)
    return np.array(scene["groundplane_in_sim"]["point"], dtype=np.float32)


def _update_sanity_camera(env: BaseEnv, step_idx: int, total_steps: int, scene_asset_dir: str) -> None:
    cam = env.unwrapped._human_render_cameras["sanity_render_camera"]
    target = _debug_target(scene_asset_dir)
    phase = 2.0 * np.pi * (step_idx / max(total_steps, 1))
    eye = target + np.array(
        [0.75 * np.cos(phase), 0.75 * np.sin(phase), 0.55 + 0.15 * np.sin(2.0 * phase)],
        dtype=np.float32,
    )
    cam.local_pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist()).sp


def _build_env(args: Args, observation_camera_mode: str) -> BaseEnv:
    sensor_configs = {"shader_pack": args.shader}
    sensor_configs["3rd_view_camera"] = observation_camera_override(
        observation_camera_mode,
        scene_asset_dir=args.scene_asset_dir,
    )
    return gym.make(
        "OpenReal2SimValidation-v1",
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs=sensor_configs,
        human_render_camera_configs={"shader_pack": args.shader},
        num_envs=1,
        sim_backend=args.sim_backend,
        scene_asset_dir=args.scene_asset_dir,
        robot_uids=RC5AeroHandRightOpenReal2SimValidation.uid,
    )


def _scripted_actions(action_template: np.ndarray, num_steps: int, move_robot: bool) -> list[np.ndarray]:
    if not move_robot:
        zero_action = np.zeros_like(action_template, dtype=np.float32)
        return [zero_action for _ in range(num_steps)]

    actions: list[np.ndarray] = []
    downward = np.zeros_like(action_template, dtype=np.float32)
    downward[2] = -0.004
    downward[6] = 1.0
    actions = [downward.copy() for _ in range(num_steps)]
    return actions


def _append_frame(frames: list[np.ndarray], obs: dict) -> None:
    frames.append(_extract_obs_frame(obs))


def _collect_rollout(
    env: BaseEnv,
    args: Args,
    *,
    capture_sanity: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    obs, _ = env.reset(
        seed=args.seed,
        options={
            "reconfigure": True,
            "use_probe_objects": args.use_probe_objects,
            "load_background": args.load_background,
            "show_debug_markers": args.show_debug_markers,
            "show_spawn_grid": args.show_spawn_grid,
            "robot_far_away": args.robot_far_away,
            "spawn_grid_center_xy": default_spawn_grid_state(args.scene_asset_dir)["center_xy"].tolist(),
            "spawn_grid_size_xy": default_spawn_grid_state(args.scene_asset_dir)["size_xy"].tolist(),
        },
    )

    obs_frames = [_extract_obs_frame(obs)]
    sanity_frames: list[np.ndarray] = []
    if capture_sanity:
        _update_sanity_camera(env, step_idx=0, total_steps=args.num_steps, scene_asset_dir=args.scene_asset_dir)
        sanity_frames.append(_extract_render_frame(env, "sanity_render_camera"))

    action_template = np.asarray(env.action_space.sample(), dtype=np.float32)
    actions = _scripted_actions(
        action_template=action_template,
        num_steps=args.num_steps,
        move_robot=args.move_robot,
    )

    for step_idx, action in enumerate(actions):
        obs, _, terminated, truncated, _ = env.step(action)
        _append_frame(obs_frames, obs)
        if capture_sanity:
            _update_sanity_camera(
                env,
                step_idx=step_idx + 1,
                total_steps=len(actions),
                scene_asset_dir=args.scene_asset_dir,
            )
            sanity_frames.append(_extract_render_frame(env, "sanity_render_camera"))
        if bool(torch.as_tensor(terminated).any().item()) or bool(
            torch.as_tensor(truncated).any().item()
        ):
            print(f"Episode ended at step {step_idx}")
            break

    return obs_frames, sanity_frames


def main(args: Args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(args, args.observation_camera_mode)
    try:
        obs_frames, sanity_frames = _collect_rollout(env, args, capture_sanity=args.debug)
        print("obs_frame_shape:", obs_frames[0].shape)
        print("obs_frame_minmax:", int(obs_frames[0].min()), int(obs_frames[0].max()))
        print("obs_camera_pose:", env.unwrapped._sensors["3rd_view_camera"].camera.global_pose)
        if args.debug and sanity_frames:
            print("sanity_frame_shape:", sanity_frames[0].shape)
            print("sanity_frame_minmax:", int(sanity_frames[0].min()), int(sanity_frames[0].max()))
            print(
                "sanity_render_camera_pose:",
                env.unwrapped._human_render_cameras["sanity_render_camera"].camera.global_pose,
            )
        images_to_video(obs_frames, str(output_dir), args.video_name, fps=10, verbose=False)
        print(f"Saved video to {output_dir / (args.video_name + '.mp4')}")
        if args.debug and sanity_frames:
            images_to_video(sanity_frames, str(output_dir), args.sanity_video_name, fps=10, verbose=False)
            print(f"Saved video to {output_dir / (args.sanity_video_name + '.mp4')}")
    finally:
        env.close()



if __name__ == "__main__":
    main(tyro.cli(Args))
