#!/usr/bin/env python3
"""Replay raw RL4VLA episodes headless and dump per-step 7D proprio for OpenVLA_V2 SFT.

For each raw episode npz (schema rc5_rl4vla_raw_episode_v2 with embedded runtime
bundle), reconstruct the OpenReal2Sim env from the per-episode runtime_config.yaml
referenced by the embedded request, replay the recorded action sequence open-loop,
and record proprio[i] = [arm_qpos(6), gripper_closure(1)] captured BEFORE action[i]
(the same state that produced image[i] in the recording).

The closure scalar is mean(clip(hand_qpos / hand_close_qpos, 0, 1)) over the 16
Aero Hand joints, matching SimlerWrapper._get_proprio_7d used at RL rollout time.

Usage:
  python dump_proprio_replay.py <src_dir> <dest_dir> [--start 0] [--end 100]
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

# Import via the full package path so relative imports inside agents/envs resolve, then
# alias the top-level names the viewer module imports at load time.
from openreal2sim.simulation.maniskill import agents as _agents_pkg  # noqa: E402,F401
from openreal2sim.simulation.maniskill import envs as _envs_pkg  # noqa: E402,F401

sys.modules.setdefault("agents", _agents_pkg)
sys.modules.setdefault("envs", _envs_pkg)

from openreal2sim.simulation.maniskill.scripts import rc5_replay_rl4vla_npz_viewer as V  # noqa: E402
from openreal2sim.simulation.maniskill.utils.scene_loader import remap_object_placement_paths  # noqa: E402


def _viewer_args(
    npz_path: Path,
    config_path: Path,
    *,
    key: str | None = None,
    scene: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        npz_path=str(npz_path),
        scene=scene,
        config_path=str(config_path),
        key=key,
        task_object_id=None,
        task_type=None,
        instruction=None,
        object_placements_json_path=None,
        control_mode=None,
        render_backend="gpu",
        sim_backend="physx_cuda",
        window_width=512,
        window_height=512,
        playback_fps=20.0,
        output_dir=None,
        video_basename=None,
        video_format=None,
        video_codec=None,
        video_fps=None,
        gif_fps=15,
        gif_scale_width=800,
        save_video=False,
        save_gif=False,
        dry_run=False,
        extract_embedded_bundle=False,
    )


def _runtime_config_path_for_episode(npz_path: Path) -> Path:
    payload = V._load_npz_payload(npz_path)
    request = V._load_embedded_runtime_request_payload(payload, npz_path=npz_path)
    if request is None or not request.get("runtime_config_path"):
        raise RuntimeError(f"{npz_path.name}: embedded request has no runtime_config_path")
    config_path = Path(request["runtime_config_path"])
    if not config_path.exists():
        raise FileNotFoundError(f"{npz_path.name}: runtime_config not found: {config_path}")
    return config_path


def _prepare_startup_state(env, hand_pose_cfg_path, startup_cfg) -> None:
    """Mirror the viewer's startup: reset, optional hand pose, controller-held settle, sync."""
    env.reset(seed=0, options=dict(reconfigure=True))
    agent = env.unwrapped.agent
    if hand_pose_cfg_path is not None and hand_pose_cfg_path.exists():
        V._apply_hand_pose_config_to_agent(agent, hand_pose_cfg_path)
    requested_control_mode = startup_cfg.get("requested_control_mode")
    settle_steps = int(startup_cfg.get("settle_steps", 0) or 0)
    gripper_open_signal = startup_cfg.get("gripper_open_signal")
    supported_control_modes = list(getattr(agent, "supported_control_modes", []) or [])
    stabilize_control_mode = V._select_startup_stabilize_control_mode(
        supported_control_modes=supported_control_modes,
        requested_control_mode=requested_control_mode,
    )
    if stabilize_control_mode is None:
        if settle_steps > 0:
            raise RuntimeError(
                "Startup stabilization requires a supported joint-space control mode, "
                f"but requested control_mode='{requested_control_mode}' and supported={supported_control_modes}."
            )
    elif stabilize_control_mode != requested_control_mode:
        with V._temporary_agent_control_mode(env, stabilize_control_mode, reason="startup scene stabilization"):
            V._stabilize_env(env, settle_steps, stabilize_control_mode, gripper_hold_signal=gripper_open_signal)
    else:
        V._stabilize_env(env, settle_steps, requested_control_mode, gripper_hold_signal=gripper_open_signal)
    V._sync_controller_targets_to_current_state(env.unwrapped)


def _to_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def dump_episode_proprio(npz_path: Path, dest_dir: Path) -> dict:
    config_path = _runtime_config_path_for_episode(npz_path)
    args = _viewer_args(npz_path, config_path)
    context = V._load_replay_context(args)
    env_kwargs, sim_cfg, hand_pose_cfg_path, startup_cfg = V._build_env_kwargs(context, args)

    # Headless state-only replay; exact object poses from the per-episode config.
    env_kwargs["render_mode"] = None
    env_kwargs["obs_mode"] = "state"
    env_kwargs["placement_mode"] = "fixed"
    if env_kwargs.get("object_placements") is not None:
        env_kwargs["object_placements"] = remap_object_placement_paths(env_kwargs["object_placements"])

    actions = np.asarray(context.payload.get("action"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"{npz_path.name}: expected action array (T, 7), got {actions.shape}")

    env = V.envs.OpenReal2SimEnv(**env_kwargs)
    try:
        _prepare_startup_state(env, hand_pose_cfg_path, startup_cfg)

        agent = env.unwrapped.agent
        n_arm = len(agent.arm_joint_names)
        close_qpos = np.asarray(agent.hand_close_qpos, dtype=np.float32).reshape(-1)
        proprio = np.zeros((actions.shape[0], n_arm + 1), dtype=np.float32)

        last_info: dict = {}
        for i in range(actions.shape[0]):
            qpos = V._get_robot_qpos(env)
            arm_qpos = qpos[:n_arm]
            hand_qpos = qpos[n_arm : n_arm + close_qpos.size]
            closure = float(np.clip(hand_qpos / close_qpos, 0.0, 1.0).mean())
            proprio[i, :n_arm] = arm_qpos
            proprio[i, n_arm] = closure
            _, _, _, _, last_info = env.step(actions[i].astype(np.float32, copy=True))

        replay_success = _to_bool(last_info.get("success", False)) if last_info else False
    finally:
        env.close()

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / npz_path.name
    np.savez_compressed(
        dest,
        proprio=proprio,
        replay_success=np.asarray([replay_success]),
        hand_close_qpos=close_qpos,
    )
    return {
        "episode": npz_path.name,
        "steps": int(actions.shape[0]),
        "replay_success": replay_success,
        "proprio0": proprio[0].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dest_dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    files = sorted(args.src_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz in {args.src_dir}")
    end = len(files) if args.end < 0 else min(args.end, len(files))
    files = files[args.start : end]
    print(f"Dumping proprio for {len(files)} episodes [{args.start}:{end}] -> {args.dest_dir}", flush=True)

    results = []
    for idx, src in enumerate(files):
        rec = dump_episode_proprio(src, args.dest_dir)
        results.append(rec)
        print(
            f"[{idx + 1}/{len(files)}] {rec['episode']} steps={rec['steps']} "
            f"replay_success={rec['replay_success']}",
            flush=True,
        )

    n_success = sum(1 for r in results if r["replay_success"])
    init = np.asarray([r["proprio0"] for r in results], dtype=np.float32)
    summary = {
        "n_episodes": len(results),
        "replay_success_rate": n_success / max(1, len(results)),
        "proprio0_mean": init.mean(axis=0).tolist(),
        "proprio0_std": init.std(axis=0).tolist(),
    }
    (args.dest_dir / "proprio_dump_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
