#!/usr/bin/env python3
"""Re-render recorded RL4VLA episodes with N different 360 backgrounds.

Each source episode (schema rc5_rl4vla_raw_episode_v2 with per-step robot_qpos and
object_poses recorded at collection time) is replayed *kinematically*: per step we
set the robot qpos and object poses directly (no physics step), then render the
3rd_view_camera (640x480) and wrist_camera (168x224) — the exact sensors the RL
gym exposes. For each of --variants variants we apply a different (photo, yaw) to
the 360 background sphere, producing an augmented episode with identical
states/actions and new background pixels.

Also computes the 7D proprio (6 arm qpos + gripper closure scalar) per step from
the recorded qpos, matching SimlerWrapper._get_proprio_7d used at RL rollout time.

Usage:
  python render_bg_variants.py <src_dir> <dest_dir> [--variants 10] [--start 0] [--end -1] [--seed 0]
  python render_bg_variants.py <src_dir> <dest_dir> --validate   # reproduce recorded bg, diff vs recorded frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIM2REAL = Path("/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real")
if str(SIM2REAL) not in sys.path:
    sys.path.insert(0, str(SIM2REAL))

from openreal2sim.simulation.maniskill import agents as _agents_pkg  # noqa: E402,F401
from openreal2sim.simulation.maniskill import envs as _envs_pkg  # noqa: E402,F401

sys.modules.setdefault("agents", _agents_pkg)
sys.modules.setdefault("envs", _envs_pkg)

import sapien  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402
from transforms3d.euler import euler2quat  # noqa: E402

from openreal2sim.simulation.maniskill.scripts import rc5_replay_rl4vla_npz_viewer as V  # noqa: E402
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (  # noqa: E402
    _encode_frame_to_jpeg_uint8_buffer,
    decode_rl4vla_raw_episode_images,
    load_rl4vla_raw_episode_artifact,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_artifacts import (  # noqa: E402
    capture_named_camera_frames,
)
from openreal2sim.simulation.maniskill.utils.scene_loader import remap_object_placement_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_proprio_replay import (  # noqa: E402
    _prepare_startup_state,
    _runtime_config_path_for_episode,
    _viewer_args,
)

SCENE_CAM = "3rd_view_camera"
WRIST_CAM = "wrist_camera"


def _force_rl_camera_sizes(env_kwargs: dict) -> None:
    """Match OpenReal2Sim-v0 PPO cameras: 3rd_view 640x480 + wrist 224x168."""
    cameras_config = dict(env_kwargs.get("cameras_config") or {})
    base_camera = dict(cameras_config.get("base_camera") or {})
    base_camera["width"] = 640
    base_camera["height"] = 480
    cameras_config["base_camera"] = base_camera
    env_kwargs["cameras_config"] = cameras_config


def _build_render_env(
    npz_path: Path,
    *,
    config_path: Path | None = None,
    key: str | None = None,
    scene: str | None = None,
):
    """Single-env OpenReal2Sim with the RL gym's cameras + 360 background."""
    resolved_config = config_path if config_path is not None else _runtime_config_path_for_episode(npz_path)
    args = _viewer_args(npz_path, resolved_config, key=key, scene=scene)
    context = V._load_replay_context(args)
    env_kwargs, sim_cfg, hand_pose_cfg_path, startup_cfg = V._build_env_kwargs(context, args)
    env_kwargs["render_mode"] = None
    env_kwargs["obs_mode"] = "rgb"  # -> 3rd_view_camera (base_camera pose) + wrist_camera
    env_kwargs["placement_mode"] = "fixed"
    env_kwargs["use_wrist_camera"] = True
    env_kwargs["use_360_background"] = True
    _force_rl_camera_sizes(env_kwargs)
    if env_kwargs.get("object_placements") is not None:
        env_kwargs["object_placements"] = remap_object_placement_paths(env_kwargs["object_placements"])
    env = V.envs.OpenReal2SimEnv(**env_kwargs)
    _prepare_startup_state(env, hand_pose_cfg_path, startup_cfg)
    return env


def _set_pano(env, photo_idx: int, yaw: float) -> None:
    uw = env.unwrapped
    textures = uw._pano_textures
    if not textures:
        raise RuntimeError("360 background requested but no pano textures are loaded")
    photo_idx = int(photo_idx) % len(textures)
    uw._apply_pano_texture(textures[photo_idx])
    center = (
        np.asarray(uw._pano_center, dtype=np.float32)
        if uw._pano_center is not None
        else uw._pano_sphere_center_xyz()
    )
    quat = np.asarray(euler2quat(0.0, 0.0, float(yaw)), dtype=np.float32)
    pose = Pose.create_from_pq(p=center.reshape(1, 3), q=quat.reshape(1, 4))
    uw._pano_sphere_actor.set_pose(pose)
    if getattr(uw.scene, "gpu_sim_enabled", False):
        try:
            uw.scene.px.gpu_apply_rigid_dynamic_data()
        except Exception:
            pass


def _set_kinematic_state(env, qpos: np.ndarray, obj_poses: np.ndarray, obj_names: list[str]) -> None:
    uw = env.unwrapped
    device = uw.agent.robot.qpos.device
    uw.agent.robot.set_qpos(torch.as_tensor(qpos[None], dtype=torch.float32, device=device))
    object_actors = uw.object_actors
    for col, name in enumerate(obj_names):
        actor = object_actors.get(name)
        if actor is None:
            continue
        p = torch.as_tensor(obj_poses[col : col + 1, :3], dtype=torch.float32, device=device)
        q = torch.as_tensor(obj_poses[col : col + 1, 3:], dtype=torch.float32, device=device)
        actor.set_pose(Pose.create_from_pq(p=p, q=q))
    if getattr(uw.scene, "gpu_sim_enabled", False):
        uw.scene._gpu_apply_all()
        uw.scene.px.gpu_update_articulation_kinematics()
        uw.scene._gpu_fetch_all()


def _render_pair(env) -> tuple[np.ndarray, np.ndarray]:
    scene = capture_named_camera_frames(env, SCENE_CAM, required=True, capture=True)[0]
    wrist = capture_named_camera_frames(env, WRIST_CAM, required=True, capture=False)[0]
    return scene, wrist


def _proprio_from_qpos(qpos: np.ndarray, n_arm: int, close_qpos: np.ndarray) -> np.ndarray:
    arm = qpos[:n_arm]
    hand = qpos[n_arm : n_arm + close_qpos.size]
    closure = float(np.clip(hand / close_qpos, 0.0, 1.0).mean())
    return np.concatenate([arm, np.asarray([closure], dtype=np.float32)]).astype(np.float32)


def _variant_rng(episode_name: str, variant_idx: int, seed: int) -> np.random.RandomState:
    key = (seed * 1000003 + variant_idx * 7919 + sum(episode_name.encode("utf-8"))) % (2**31 - 1)
    return np.random.RandomState(key)


def render_episode_variants(
    npz_path: Path,
    dest_dir: Path,
    *,
    n_variants: int,
    seed: int,
    validate: bool,
    overwrite: bool = False,
    config_path: Path | None = None,
    key: str | None = None,
    scene: str | None = None,
) -> dict:
    payload = load_rl4vla_raw_episode_artifact(npz_path)
    qpos = np.asarray(payload.get("robot_qpos"), dtype=np.float32)
    obj_poses = np.asarray(payload.get("object_poses"), dtype=np.float32)
    obj_names = [str(n) for n in payload.get("object_names")]
    actions = np.asarray(payload.get("action"), dtype=np.float32)
    n_steps = actions.shape[0]
    if qpos.shape[0] < n_steps or obj_poses.shape[0] < n_steps:
        raise RuntimeError(
            f"{npz_path.name}: missing per-step state (qpos {qpos.shape}, poses {obj_poses.shape}, T={n_steps})"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    stats = {"episode": npz_path.name, "steps": n_steps, "variants": [], "validate_diffs": []}
    if not validate and not overwrite:
        existing = [dest_dir / f"{stem}_bg{k:02d}.npz" for k in range(n_variants)]
        if all(path.exists() for path in existing):
            stats["variants"] = [path.name for path in existing]
            print(f"  skip {npz_path.name}: all {n_variants} variants exist", flush=True)
            return stats

    env = _build_render_env(npz_path, config_path=config_path, key=key, scene=scene)
    try:
        agent = env.unwrapped.agent
        n_arm = len(agent.arm_joint_names)
        close_qpos = np.asarray(agent.hand_close_qpos, dtype=np.float32).reshape(-1)
        n_photos = len(env.unwrapped._pano_textures)

        rec_scene = rec_wrist = None
        if validate:
            rec_scene = decode_rl4vla_raw_episode_images(payload, key="image")
            rec_wrist = decode_rl4vla_raw_episode_images(payload, key="image_wrist")

        variant_ids = range(n_variants) if not validate else [0]
        for k in variant_ids:
            dest = dest_dir / f"{stem}_bg{k:02d}.npz"
            if (not validate) and dest.exists() and not overwrite:
                stats["variants"].append(dest.name)
                print(f"  skip existing {dest.name}", flush=True)
                continue
            if validate:
                photo_idx = int(payload.get("pano_photo_idx", 0))
                yaw = float(payload.get("pano_yaw", 0.0))
            else:
                rng = _variant_rng(stem, k, seed)
                photo_idx = int(rng.randint(0, n_photos))
                yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            _set_pano(env, photo_idx, yaw)

            scene_frames = []
            wrist_frames = []
            diffs = []
            for t in range(n_steps):
                _set_kinematic_state(env, qpos[t], obj_poses[t], obj_names)
                scene, wrist = _render_pair(env)
                if t == 0 and int(k) == 0:
                    if scene.shape != (480, 640, 3) or wrist.shape != (224, 168, 3):
                        raise RuntimeError(
                            f"{npz_path.name}: unexpected render shapes scene={scene.shape} wrist={wrist.shape}; "
                            "expected scene (480, 640, 3) and wrist_camera (224, 168, 3)."
                        )
                scene_frames.append(scene)
                wrist_frames.append(wrist)
                if validate and t < len(rec_scene):
                    diffs.append(float(np.abs(rec_scene[t].astype(np.int16) - scene.astype(np.int16)).mean()))

            proprio = np.stack(
                [_proprio_from_qpos(qpos[t], n_arm, close_qpos) for t in range(n_steps)]
            ).astype(np.float32)

            if validate:
                stats["validate_diffs"] = diffs
                arr = np.asarray(diffs)
                print(
                    f"  validate {npz_path.name}: scene |diff| mean={arr.mean():.2f} "
                    f"max={arr.max():.2f} p95={np.percentile(arr, 95):.2f}",
                    flush=True,
                )
                continue

            out_payload = {
                "schema_version": payload.get("schema_version"),
                "instruction": payload.get("instruction"),
                "image": [_encode_frame_to_jpeg_uint8_buffer(f) for f in scene_frames],
                "image_wrist": [_encode_frame_to_jpeg_uint8_buffer(f) for f in wrist_frames],
                "camera_names": [SCENE_CAM, WRIST_CAM],
                "action": actions,
                "proprio": proprio,
                "info": list(payload.get("info") or [])[:n_steps],
                "result": payload.get("result"),
                "source": {
                    **dict(payload.get("source") or {}),
                    "augmentation": "360_bg_variant",
                    "base_episode": npz_path.name,
                    "variant_idx": int(k),
                },
                "pano_photo_idx": int(photo_idx),
                "pano_yaw": float(yaw),
                "robot_qpos": qpos[:n_steps],
                "object_poses": obj_poses[:n_steps],
                "object_names": obj_names,
                "embedded_runtime_config_yaml": payload.get("embedded_runtime_config_yaml"),
                "embedded_runtime_request_json": payload.get("embedded_runtime_request_json"),
            }
            dest = dest_dir / f"{stem}_bg{k:02d}.npz"
            np_payload = np.array(out_payload, dtype=object)
            np.savez_compressed(dest, arr_0=np_payload)
            stats["variants"].append(dest.name)
            print(f"  [{k + 1}/{n_variants}] {dest.name} photo={photo_idx} yaw={yaw:.2f}", flush=True)
    finally:
        env.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dest_dir", type=Path)
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render even if the destination npz already exists.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Override per-episode runtime_config.yaml (use current config_debug.yaml for RL lighting).",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="airy_table_scene14sep26_left_image",
        help="Scene KEY in the runtime config (e.g. airy_table_scene14sep26_left_image).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional explicit scene.json path.",
    )
    args = parser.parse_args()

    files = sorted(args.src_dir.glob("episode_*.npz")) + sorted(args.src_dir.glob("rl4vla_raw_episode*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode npz in {args.src_dir}")
    end = len(files) if args.end < 0 else min(args.end, len(files))
    files = files[args.start:end]
    print(f"Rendering {len(files)} episodes x {args.variants} variants -> {args.dest_dir}", flush=True)

    all_stats = []
    for idx, src in enumerate(files):
        print(f"[{idx + 1}/{len(files)}] {src.name}", flush=True)
        stats = render_episode_variants(
            src,
            args.dest_dir,
            n_variants=args.variants,
            seed=args.seed,
            validate=args.validate,
            overwrite=args.overwrite,
            config_path=args.config_path,
            key=args.key,
            scene=args.scene,
        )
        all_stats.append(stats)

    summary = {
        "n_episodes": len(all_stats),
        "n_variants": 0 if args.validate else args.variants,
        "n_outputs": sum(len(s["variants"]) for s in all_stats),
    }
    if args.validate:
        diffs = np.concatenate([np.asarray(s["validate_diffs"]) for s in all_stats if s["validate_diffs"]])
        if diffs.size:
            summary["scene_diff_mean"] = float(diffs.mean())
            summary["scene_diff_p95"] = float(np.percentile(diffs, 95))
            summary["scene_diff_max"] = float(diffs.max())
    (args.dest_dir / "render_bg_variants_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
