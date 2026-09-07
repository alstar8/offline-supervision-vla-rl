#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone interactive replay viewer for RC5 RL4VLA raw episodes.

This script replays a recorded `rl4vla_raw_episode*.npz` action sequence inside
the OpenReal2Sim ManiSkill viewer using either explicit CLI inputs or a
same-name replay-config from a config bundle.

Important limitation: replay is action-only, not full-state. The startup scene
and RC5 hand pose can be reconstructed from config, but later grasp / lift
behavior can still diverge from the original run because the `.npz` does not
store a per-step simulator state trace.

Replay context can be provided in two ways:

1. fully explicit:
   - `--npz_path`
   - `--scene`
   - `--config_path`
   - `--key`
   - `--task_object_id`
2. via a replay-config bundle directory:
   - `--npz_path`
   - `--config_path <bundle_dir>`
   - the script resolves a same-name config from the bundle based on `.npz`

To replay the intended episode layout, either:

- pass the episode-specific `runtime_config.yaml` as `--config_path`, or
- pass a replay-config generated from that `runtime_config.yaml`, or
- pass a placement override via `--object_placements_json_path`

Viewer controls:

- `n`: start or resume full autoplay replay; if the replay already finished,
  reset the environment and start again from step 0
- `SPACE`: if autoplay is running, pause it; otherwise advance one action step
- `r`: reset to the beginning without starting autoplay
- `v`: save the currently buffered replay video/GIF immediately
- `q`: quit the viewer

Replay artifacts are written to `./tmp` by default and are only saved on the
explicit `v` hotkey or matching CLI flow. The script does not write artifacts
next to the source dataset `.npz`.

Example launch used for the `v8` flat success bundle:

```bash
docker exec -it rola-original-simulation bash -lc '
cd /app &&
python openreal2sim/simulation/maniskill/scripts/rc5_replay_rl4vla_npz_viewer.py \
  --npz_path /app/runs/dataset/rc5_v8_rl4vla_success_npz/rl4vla_raw_episode__idx_000000000__rc5_v8_clear_orange_full128_shard_000__batch_000000__env_00__success.npz \
  --config_path /app/runs/dataset/replay_npz_configs_v8_flat_success
'
```

The script intentionally does not import helper logic from sibling `scripts/*.py`
modules. It only uses standard libraries plus runtime dependencies such as
Gymnasium / ManiSkill / SAPIEN / imageio.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("DISPLAY", ":0")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = os.environ.get("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = os.environ.get("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ["__VK_LAYER_NV_optimus"] = os.environ.get("__VK_LAYER_NV_optimus", "NVIDIA_only")

_NVIDIA_LIB_PATHS = ["/usr/local/nvidia/lib", "/usr/local/nvidia/lib64"]
_PHYSX_LIB_DIR = "/root/.sapien/physx/105.1-physx-5.3.1.patch0"
_current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
for _nvidia_path in _NVIDIA_LIB_PATHS:
    if os.path.exists(_nvidia_path) and _nvidia_path not in _current_ld_path:
        _current_ld_path = f"{_nvidia_path}:{_current_ld_path}" if _current_ld_path else _nvidia_path
if os.path.exists(_PHYSX_LIB_DIR) and _PHYSX_LIB_DIR not in _current_ld_path:
    _current_ld_path = f"{_PHYSX_LIB_DIR}:{_current_ld_path}" if _current_ld_path else _PHYSX_LIB_DIR
if _current_ld_path != os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _current_ld_path

import gymnasium as gym
import imageio
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
MANISKILL_ROOT = REPO_ROOT / "openreal2sim" / "simulation" / "maniskill"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(MANISKILL_ROOT) not in sys.path:
    sys.path.append(str(MANISKILL_ROOT))

import agents  # noqa: F401 - registers agents
import envs  # noqa: F401 - registers OpenReal2Sim-v0


DEFAULT_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
EE_DELTA_CONTROL_MODES = {
    "pd_ee_delta_pose",
    "pd_ee_target_delta_pose",
    DEFAULT_CONTROL_MODE,
}
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_FORMAT = "mkv"
DEFAULT_VIDEO_CODEC = "ffv1"
DEFAULT_GIF_FPS = 15
DEFAULT_GIF_SCALE_WIDTH = 800
REQUIRED_HAND_POSE_BINDINGS = (
    "open",
    "close",
    "full_open",
    "pinch",
    "tripod",
    "thumb_abduction_full",
    "thumb_abduction_partial",
)
YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass
class ReplayContext:
    npz_path: Path
    payload: dict[str, Any]
    runtime_config_path: Path | None
    runtime_config: dict[str, Any] | None
    runtime_request: dict[str, Any] | None
    scene_path: Path
    key: str
    task_object_id: str
    task_type: str
    instruction: str
    object_placements: dict[str, Any] | None
    embedded_runtime_bundle_used: bool = False


def _warn(message: str) -> None:
    print(f"{YELLOW}[WARNING] {message}{RESET}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an RL4VLA raw episode .npz inside the OpenReal2Sim viewer."
    )
    parser.add_argument("--npz_path", required=True, help="Path to rl4vla_raw_episode*.npz")
    parser.add_argument("--scene", default=None, help="scene.json path for the replay environment.")
    parser.add_argument(
        "--config_path",
        default=None,
        help=(
            "Runtime/base config YAML used to construct the env, or a directory "
            "containing one replay-config YAML per npz basename. Optional when "
            "the rl4vla_raw_episode npz carries an embedded runtime bundle."
        ),
    )
    parser.add_argument("--key", default=None, help="Config key inside config_path.")
    parser.add_argument("--task_object_id", default=None, help="Manipulation target object id.")
    parser.add_argument(
        "--task_type",
        default=None,
        help="Task type for replay metadata. Defaults to embedded/runtime replay metadata or 'pick_up'.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional instruction override. Defaults to instruction stored in the npz payload.",
    )
    parser.add_argument(
        "--object_placements_json_path",
        default=None,
        help="Manual fallback JSON/YAML with object_placements override when no runtime_request is available.",
    )
    parser.add_argument(
        "--control_mode",
        default=None,
        help="Override control mode. Defaults to runtime config local.simulation.control_mode.",
    )
    parser.add_argument("--render_backend", default="gpu", help="Viewer render backend.")
    parser.add_argument("--sim_backend", default="physx_cuda", help="Simulation backend.")
    parser.add_argument("--window_width", type=int, default=1920, help="Viewer window width.")
    parser.add_argument("--window_height", type=int, default=1080, help="Viewer window height.")
    parser.add_argument("--playback_fps", type=float, default=20.0, help="Autoplay target FPS.")
    parser.add_argument("--output_dir", default=None, help="Directory for replay video artifacts.")
    parser.add_argument("--video_basename", default=None, help="Output basename without extension.")
    parser.add_argument("--video_format", default=None, help="Override saved video format (e.g. mkv, mp4).")
    parser.add_argument("--video_codec", default=None, help="Override saved video codec.")
    parser.add_argument("--video_fps", type=int, default=None, help="Override saved video FPS.")
    parser.add_argument("--gif_fps", type=int, default=DEFAULT_GIF_FPS, help="GIF FPS for the sidecar.")
    parser.add_argument(
        "--gif_scale_width",
        type=int,
        default=DEFAULT_GIF_SCALE_WIDTH,
        help="GIF width in pixels for ffmpeg scaling.",
    )
    parser.add_argument(
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable manual replay video saving via the 'v' hotkey.",
    )
    parser.add_argument(
        "--save_gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable manual replay GIF sidecar saving via the 'v' hotkey.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve replay context and print it without opening the viewer.",
    )
    parser.add_argument(
        "--extract_embedded_bundle",
        action="store_true",
        help=(
            "Extract embedded runtime_config.yaml and runtime_request.json from the npz "
            "into a sibling directory named after the npz stem. Works in both replay and dry-run modes."
        ),
    )
    return parser.parse_args()


def _repo_relative_or_self(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _normalize_path(raw_path: str | os.PathLike[str] | None, *, base_dirs: Iterable[Path] = ()) -> Path | None:
    if raw_path is None:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None

    candidates: list[Path] = []
    path_obj = Path(raw).expanduser()
    if path_obj.is_absolute():
        candidates.append(path_obj)
        if raw.startswith("/app/"):
            candidates.append(REPO_ROOT / Path(raw).relative_to("/app"))
    else:
        for base_dir in base_dirs:
            candidates.append((base_dir / path_obj).resolve())
        candidates.append((REPO_ROOT / path_obj).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve() if candidates else path_obj.resolve()


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in YAML file: {path}")
    return payload


def _load_yaml_text(text: str, *, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in embedded YAML payload: {label}")
    return payload


def _load_json_or_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _load_embedded_runtime_request_payload(payload: Mapping[str, Any], *, npz_path: Path) -> dict[str, Any] | None:
    raw = payload.get("embedded_runtime_request_json")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(
            f"Embedded runtime request payload in {npz_path} must be a non-empty JSON string."
        )
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError(
            f"Embedded runtime request payload in {npz_path} must decode to a JSON mapping."
        )
    return decoded


def _load_embedded_runtime_config_payload(payload: Mapping[str, Any], *, npz_path: Path) -> dict[str, Any] | None:
    raw = payload.get("embedded_runtime_config_yaml")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(
            f"Embedded runtime config payload in {npz_path} must be a non-empty YAML string."
        )
    return _load_yaml_text(raw, label=str(npz_path))


def _infer_runtime_config_key(
    runtime_config: Mapping[str, Any],
    *,
    embedded_runtime_request: Mapping[str, Any] | None = None,
) -> str:
    if embedded_runtime_request is not None:
        embedded_key = str(embedded_runtime_request.get("key") or "").strip()
        if embedded_key:
            return embedded_key
    keys_value = runtime_config.get("keys")
    if isinstance(keys_value, list):
        normalized_keys = [str(item).strip() for item in keys_value if str(item).strip()]
        if len(normalized_keys) == 1:
            return normalized_keys[0]
    local_cfg = runtime_config.get("local")
    if isinstance(local_cfg, dict):
        local_keys = [str(item).strip() for item in local_cfg.keys() if str(item).strip()]
        if len(local_keys) == 1:
            return local_keys[0]
    raise RuntimeError(
        "Replay key is missing. Pass --key explicitly, use replay_viewer.key, or provide "
        "an embedded/runtime config with exactly one key."
    )


def _resolve_runtime_config_path(config_path_arg: str, npz_path: Path) -> Path:
    config_candidate = _normalize_path(config_path_arg, base_dirs=[Path.cwd(), REPO_ROOT, npz_path.parent])
    if config_candidate is None or not config_candidate.exists():
        raise FileNotFoundError(f"--config_path not found: {config_path_arg}")

    if config_candidate.is_file():
        return config_candidate

    index_path = config_candidate / "replay_index.tsv"
    if index_path.exists():
        npz_relpath_candidates: list[str] = []
        try:
            npz_relpath_candidates.append(str(npz_path.resolve().relative_to(REPO_ROOT / "runs" / "dataset")))
        except Exception:
            pass
        npz_parts = npz_path.parts
        if "runs" in npz_parts and "dataset" in npz_parts:
            try:
                runs_index = npz_parts.index("runs")
                if npz_parts[runs_index + 1] == "dataset":
                    npz_relpath_candidates.append(str(Path(*npz_parts[runs_index + 2 :])))
            except Exception:
                pass
        if "/app/runs/dataset/" in npz_path.as_posix():
            npz_relpath_candidates.append(npz_path.as_posix().split("/app/runs/dataset/", 1)[1])

        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                relpath = str(row.get("npz_relpath") or "").strip()
                if relpath and relpath in npz_relpath_candidates:
                    config_filename = str(row.get("config_filename") or "").strip()
                    if config_filename:
                        candidate = config_candidate / config_filename
                        if candidate.exists():
                            return candidate.resolve()

    config_names = [
        f"{npz_path.name}.yaml",
        f"{npz_path.name}.yml",
        f"{npz_path.stem}.yaml",
        f"{npz_path.stem}.yml",
    ]
    for config_name in config_names:
        candidate = config_candidate / config_name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a same-name replay config inside bundle dir "
        f"{config_candidate} for npz {npz_path.name}"
    )


def _load_profile(config_path: Path | None, profile_name: str | None) -> dict[str, Any] | None:
    if config_path is None or profile_name is None:
        return None
    payload = _load_yaml(config_path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise RuntimeError(f"Profile config does not contain a 'profiles' mapping: {config_path}")
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise RuntimeError(
            f"Profile '{profile_name}' was not found in {config_path}"
        )
    return copy.deepcopy(profile)


def _load_named_profile(
    config_path: Path | None,
    profile_name: str | None,
    *,
    scope: str,
    summary_fields: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    profile = _load_profile(config_path, profile_name)
    if profile is None:
        return None
    summary = ", ".join(f"{field}={profile[field]}" for field in summary_fields if field in profile)
    if summary:
        print(f"[{scope}] Loaded profile '{profile_name}' from {config_path}: {summary}")
    else:
        print(f"[{scope}] Loaded profile '{profile_name}' from {config_path}")
    return profile


def _build_hand_qpos_from_pose(
    agent: Any,
    pose_cfg_path: Path,
    pose_name: str,
    pose_data: Mapping[str, Any],
) -> np.ndarray:
    joints = pose_data.get("joints")
    if not isinstance(joints, dict):
        raise ValueError(f"Hand pose '{pose_name}' in {pose_cfg_path} must define a 'joints' mapping")
    hand_names = list(getattr(agent, "hand_joint_names", []))
    missing = [name for name in hand_names if name not in joints]
    extra = [name for name in joints.keys() if name not in hand_names]
    if missing or extra:
        raise RuntimeError(
            f"Hand pose '{pose_name}' joint-name mismatch in {pose_cfg_path}\n"
            f"expected_hand_joint_names={hand_names}\n"
            f"missing={missing}\n"
            f"extra={extra}"
        )
    return np.asarray([float(joints[name]) for name in hand_names], dtype=np.float32)


def _apply_hand_pose_config_to_agent(
    agent: Any,
    hand_pose_config: Path | None,
    *,
    open_preset_name: str | None = None,
    close_preset_name: str | None = None,
    required_bindings: Sequence[str] = REQUIRED_HAND_POSE_BINDINGS,
) -> bool:
    if hand_pose_config is None:
        return False
    hand_names = list(getattr(agent, "hand_joint_names", []))
    if not hand_names:
        _warn(
            f"Agent uid='{getattr(agent, 'uid', 'unknown')}' does not expose hand_joint_names; "
            f"cannot apply hand pose config '{hand_pose_config}'."
        )
        return False
    cfg = _load_yaml(hand_pose_config)
    poses = cfg.get("poses")
    bindings = cfg.get("bindings", {})
    if not isinstance(poses, dict):
        raise ValueError(f"Hand pose config must define a 'poses' mapping: {hand_pose_config}")
    if not isinstance(bindings, dict):
        raise ValueError(f"Hand pose config bindings must be a mapping: {hand_pose_config}")
    missing_bindings = [name for name in required_bindings if name not in bindings]
    if missing_bindings:
        raise KeyError(f"Hand pose config {hand_pose_config} is missing required bindings: {missing_bindings}")
    resolved_open = open_preset_name or bindings.get("open")
    resolved_close = close_preset_name or bindings.get("close")
    if resolved_open not in poses or resolved_close not in poses:
        raise KeyError(
            f"Hand pose config {hand_pose_config} is missing open/close presets referenced by bindings: "
            f"open={resolved_open}, close={resolved_close}"
        )
    all_qpos_presets = {
        pose_name: _build_hand_qpos_from_pose(agent, hand_pose_config, pose_name, pose_data)
        for pose_name, pose_data in poses.items()
    }
    agent._debug_hand_pose_config_path = str(hand_pose_config)
    agent._debug_hand_pose_bindings = dict(bindings)
    agent._debug_hand_pose_presets = all_qpos_presets
    agent.hand_open_qpos = all_qpos_presets[str(resolved_open)].copy()
    agent.hand_close_qpos = all_qpos_presets[str(resolved_close)].copy()
    controller = getattr(agent, "controller", None)
    gripper_controller = getattr(controller, "controllers", {}).get("gripper") if controller is not None else None
    if gripper_controller is not None and hasattr(gripper_controller, "config"):
        if hasattr(gripper_controller.config, "open_qpos"):
            gripper_controller.config.open_qpos = agent.hand_open_qpos.tolist()
        if hasattr(gripper_controller.config, "close_qpos"):
            gripper_controller.config.close_qpos = agent.hand_close_qpos.tolist()
        print("[HandPose] Updated runtime gripper controller open/close presets.")
    print(
        f"[HandPose] Loaded hand pose presets from {hand_pose_config}: "
        f"open='{resolved_open}', close='{resolved_close}'"
    )
    print(f"[HandPose] bindings={bindings}")
    print(
        f"[HandPose] hand_open_qpos={np.array2string(agent.hand_open_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    print(
        f"[HandPose] hand_close_qpos={np.array2string(agent.hand_close_qpos, precision=4, suppress_small=True, max_line_width=200)}"
    )
    return True


def _sync_controller_targets_to_current_state(env_unwrapped: Any) -> None:
    sync_fn = getattr(env_unwrapped, "_sync_agent_controller_targets_to_current_state", None)
    if callable(sync_fn):
        sync_fn()
        return
    controller = getattr(getattr(env_unwrapped, "agent", None), "controller", None)
    if controller is None:
        return

    def _sync_one(ctrl: Any) -> None:
        subcontrollers = getattr(ctrl, "controllers", None)
        if isinstance(subcontrollers, dict):
            for sub in subcontrollers.values():
                _sync_one(sub)
            return
        try:
            ctrl.reset()
        except Exception:
            return
        if not hasattr(ctrl, "set_drive_targets") or not hasattr(ctrl, "qpos"):
            return
        try:
            targets = ctrl.qpos
            if hasattr(targets, "clone"):
                targets = targets.clone()
            if hasattr(targets, "ndim") and int(targets.ndim) == 1:
                targets = targets.unsqueeze(0)
            ctrl.set_drive_targets(targets)
        except Exception:
            return

    _sync_one(controller)


def _is_ee_delta_control_mode(control_mode: str | None) -> bool:
    return str(control_mode or "").strip() in EE_DELTA_CONTROL_MODES


def _select_startup_stabilize_control_mode(
    *,
    supported_control_modes: Sequence[str],
    requested_control_mode: str | None,
) -> str | None:
    if not _is_ee_delta_control_mode(requested_control_mode):
        return requested_control_mode
    supported = list(supported_control_modes or [])
    if "pd_joint_pos" in supported:
        return "pd_joint_pos"
    if "pd_joint_pos_vel" in supported:
        return "pd_joint_pos_vel"
    return None


def _map_teleop_gripper_signal_to_controller(signal_value: Any, target_state: str) -> float:
    magnitude = abs(float(signal_value))
    if target_state == "open":
        return magnitude
    if target_state == "close":
        return -magnitude
    raise ValueError(f"Unsupported gripper target_state: {target_state}")


def _get_robot_qpos(env: Any) -> np.ndarray:
    qpos = env.unwrapped.agent.robot.get_qpos()
    if hasattr(qpos, "detach") and callable(getattr(qpos, "detach", None)):
        qpos = qpos.detach().cpu().numpy()
    elif hasattr(qpos, "cpu") and callable(getattr(qpos, "cpu", None)):
        qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim > 1:
        qpos = qpos[0]
    return qpos.reshape(-1)


def _get_robot_joint_names(env: Any) -> list[str]:
    joints = env.unwrapped.agent.robot.get_active_joints()
    names: list[str] = []
    for joint in joints:
        name = getattr(joint, "name", None)
        names.append(str(name) if name is not None else "<unnamed>")
    return names


def _log_manual_step_state(env: Any, *, action: np.ndarray, step_index_before: int, step_count: int) -> None:
    qpos = _get_robot_qpos(env)
    joint_names = _get_robot_joint_names(env)
    print(
        "[ReplayStep] step={}/{} action={}".format(
            int(step_index_before + 1),
            int(step_count),
            np.array2string(np.asarray(action, dtype=np.float32).reshape(-1), precision=4, suppress_small=True, max_line_width=200),
        )
    )
    print(f"[ReplayStep] qpos_joint_order={joint_names}")
    print(
        "[ReplayStep] qpos={}".format(
            np.array2string(qpos, precision=4, suppress_small=True, max_line_width=200)
        )
    )


def _build_hold_action(env: Any, control_mode: str | None, gripper_hold_signal: float | None = None) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    action_dim = int(shape[-1]) if shape is not None and len(shape) > 0 else 0
    if _is_ee_delta_control_mode(control_mode):
        if action_dim <= 0:
            action_dim = 7
        action = np.zeros(action_dim, dtype=np.float32)
        if action_dim >= 7 and gripper_hold_signal is not None:
            action[6] = _map_teleop_gripper_signal_to_controller(gripper_hold_signal, "open")
        return action
    robot_qpos = _get_robot_qpos(env)
    if robot_qpos.size <= 0:
        raise RuntimeError("Viewer replay could not determine robot qpos for startup stabilization.")
    return robot_qpos.astype(np.float32, copy=True)


def _stabilize_env(
    env: Any,
    settle_steps: int,
    control_mode: str | None,
    gripper_hold_signal: float | None = None,
) -> None:
    settle_steps = int(settle_steps or 0)
    if settle_steps <= 0:
        return
    print(f"[Init] Stabilizing scene with {settle_steps} hold steps...")
    for _ in range(settle_steps):
        hold_action = _build_hold_action(env, control_mode, gripper_hold_signal=gripper_hold_signal)
        if hold_action.ndim == 1:
            hold_action = hold_action.reshape(1, -1)
        env.step(hold_action)


@contextmanager
def _temporary_agent_control_mode(env: Any, target_control_mode: str | None, reason: str):
    env_unwrapped = env.unwrapped
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None or target_control_mode is None:
        yield None
        return
    previous_control_mode = getattr(agent, "control_mode", None)
    if previous_control_mode == target_control_mode:
        print(
            f"[PlannerDebug] control_mode already '{target_control_mode}' for {reason}; "
            "reusing active controller."
        )
        yield previous_control_mode
        return
    print(
        f"[PlannerDebug] Switching control_mode for {reason}: "
        f"'{previous_control_mode}' -> '{target_control_mode}'"
    )
    agent.set_control_mode(target_control_mode)
    try:
        yield previous_control_mode
    finally:
        if previous_control_mode is not None and getattr(agent, "control_mode", None) != previous_control_mode:
            print(
                f"[PlannerDebug] Restoring control_mode after {reason}: "
                f"'{getattr(agent, 'control_mode', None)}' -> '{previous_control_mode}'"
            )
            agent.set_control_mode(previous_control_mode)


def _log_hand_joint_state(env_unwrapped: Any, *, label: str) -> None:
    agent = getattr(env_unwrapped, "agent", None)
    if agent is None:
        return
    hand_joint_names = list(getattr(agent, "hand_joint_names", []) or [])
    if not hand_joint_names:
        print(f"[HandPresetDebug] label={label} hand_joint_names unavailable")
        return
    robot_qpos = _get_robot_qpos(type("EnvProxy", (), {"unwrapped": env_unwrapped})())
    arm_dof = len(getattr(agent, "arm_joint_names", []) or [])
    hand_dof = len(hand_joint_names)
    robot_hand_qpos = robot_qpos[arm_dof:arm_dof + hand_dof]
    print(f"[HandPresetDebug] label={label} hand_joint_names={hand_joint_names}")
    print(
        "[HandPresetDebug] label={} robot_hand_qpos={}".format(
            label,
            np.array2string(robot_hand_qpos, precision=4, suppress_small=True, max_line_width=200),
        )
    )
    if hasattr(agent, "hand_open_qpos"):
        print(
            "[HandPresetDebug] label={} hand_open_qpos={}".format(
                label,
                np.array2string(np.asarray(agent.hand_open_qpos, dtype=np.float32).reshape(-1), precision=4, suppress_small=True, max_line_width=200),
            )
        )
    if hasattr(agent, "hand_close_qpos"):
        print(
            "[HandPresetDebug] label={} hand_close_qpos={}".format(
                label,
                np.array2string(np.asarray(agent.hand_close_qpos, dtype=np.float32).reshape(-1), precision=4, suppress_small=True, max_line_width=200),
            )
        )
    gripper_controller = getattr(getattr(agent, "controller", None), "controllers", {}).get("gripper")
    config = getattr(gripper_controller, "config", None) if gripper_controller is not None else None
    if config is not None and hasattr(config, "open_qpos"):
        print(
            "[HandPresetDebug] label={} controller_open_qpos={}".format(
                label,
                np.array2string(np.asarray(config.open_qpos, dtype=np.float32).reshape(-1), precision=4, suppress_small=True, max_line_width=200),
            )
        )
    if config is not None and hasattr(config, "close_qpos"):
        print(
            "[HandPresetDebug] label={} controller_close_qpos={}".format(
                label,
                np.array2string(np.asarray(config.close_qpos, dtype=np.float32).reshape(-1), precision=4, suppress_small=True, max_line_width=200),
            )
        )


def _load_npz_payload(npz_path: Path) -> dict[str, Any]:
    archive = np.load(npz_path, allow_pickle=True)
    if "arr_0" in archive.files:
        payload = archive["arr_0"].item()
    else:
        payload = {key: archive[key] for key in archive.files}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported replay payload format in {npz_path}")
    return payload


def _load_replay_context(args: argparse.Namespace) -> ReplayContext:
    npz_path = _normalize_path(args.npz_path, base_dirs=[Path.cwd(), REPO_ROOT])
    if npz_path is None or not npz_path.exists():
        raise FileNotFoundError(f"Replay npz not found: {args.npz_path}")

    payload = _load_npz_payload(npz_path)
    embedded_runtime_request = _load_embedded_runtime_request_payload(payload, npz_path=npz_path)
    embedded_runtime_config = _load_embedded_runtime_config_payload(payload, npz_path=npz_path)
    embedded_runtime_bundle_used = False
    runtime_config_path: Path | None
    runtime_config: dict[str, Any] | None
    if args.config_path:
        runtime_config_path = _resolve_runtime_config_path(args.config_path, npz_path)
        runtime_config = _load_yaml(runtime_config_path)
    elif embedded_runtime_config is not None:
        runtime_config = copy.deepcopy(embedded_runtime_config)
        runtime_config_path = _normalize_path(
            None if embedded_runtime_request is None else embedded_runtime_request.get("runtime_config_path"),
            base_dirs=[npz_path.parent, REPO_ROOT, Path.cwd()],
        )
        embedded_runtime_bundle_used = True
    else:
        raise FileNotFoundError(
            "Replay runtime config is missing. Pass --config_path explicitly or use "
            "an rl4vla_raw_episode npz with embedded_runtime_config_yaml."
        )
    replay_viewer_cfg = runtime_config.get("replay_viewer", {})
    if not isinstance(replay_viewer_cfg, dict):
        replay_viewer_cfg = {}

    scene_value = args.scene or replay_viewer_cfg.get("scene_path")
    if scene_value is None and embedded_runtime_request is not None:
        scene_value = embedded_runtime_request.get("scene_path")
    scene_path = _normalize_path(scene_value, base_dirs=[Path.cwd(), REPO_ROOT, npz_path.parent])
    if scene_path is None:
        raise FileNotFoundError(
            "Replay scene path is missing. Pass --scene explicitly or use a replay-config "
            "that contains replay_viewer.scene_path, or use an embedded runtime request bundle."
        )

    key = str(
        args.key
        or replay_viewer_cfg.get("key")
        or _infer_runtime_config_key(runtime_config, embedded_runtime_request=embedded_runtime_request)
    ).strip()
    if not key:
        raise RuntimeError("Replay key resolution produced an empty string.")

    global_sim = (
        runtime_config.get("global", {}).get("simulation", {})
        if isinstance(runtime_config.get("global"), dict)
        else {}
    )
    local_sim = (
        runtime_config.get("local", {}).get(key, {}).get("simulation", {})
        if isinstance(runtime_config.get("local"), dict)
        else {}
    )
    if not isinstance(global_sim, dict):
        global_sim = {}
    if not isinstance(local_sim, dict):
        local_sim = {}
    merged_sim = _deep_merge(global_sim, local_sim)

    task_object_id = str(
        args.task_object_id
        or replay_viewer_cfg.get("task_object_id")
        or (None if embedded_runtime_request is None else embedded_runtime_request.get("task_object_id"))
        or merged_sim.get("manip_object_id")
        or ""
    ).strip()
    if not task_object_id:
        raise RuntimeError(
            "Replay task_object_id is missing. Pass --task_object_id explicitly or use "
            "a replay-config that contains replay_viewer.task_object_id, or use an embedded runtime bundle."
        )

    task_type = str(
        args.task_type
        or replay_viewer_cfg.get("task_type")
        or (None if embedded_runtime_request is None else embedded_runtime_request.get("task_type"))
        or "pick_up"
    )
    instruction = str(
        args.instruction
        or replay_viewer_cfg.get("instruction")
        or (None if embedded_runtime_request is None else embedded_runtime_request.get("dense_episode_instruction"))
        or payload.get("instruction")
        or f"Pick up {task_object_id}."
    )
    object_placements = None
    if args.object_placements_json_path:
        placements_path = _normalize_path(
            args.object_placements_json_path,
            base_dirs=[Path.cwd(), REPO_ROOT, npz_path.parent],
        )
        if placements_path is None or not placements_path.exists():
            raise FileNotFoundError(f"--object_placements_json_path not found: {args.object_placements_json_path}")
        object_placements = _load_json_or_yaml(placements_path)

    if not scene_path.exists():
        raise FileNotFoundError(f"Resolved scene_path does not exist: {scene_path}")

    return ReplayContext(
        npz_path=npz_path,
        payload=payload,
        runtime_config_path=runtime_config_path,
        runtime_config=runtime_config,
        runtime_request=(None if embedded_runtime_request is None else copy.deepcopy(embedded_runtime_request)),
        scene_path=scene_path,
        key=str(key),
        task_object_id=str(task_object_id),
        task_type=str(task_type or "pick_up"),
        instruction=str(instruction or f"Pick up {task_object_id}."),
        object_placements=copy.deepcopy(object_placements) if isinstance(object_placements, dict) else object_placements,
        embedded_runtime_bundle_used=bool(embedded_runtime_bundle_used),
    )


def _build_simulation_config(context: ReplayContext) -> dict[str, Any]:
    config = context.runtime_config
    if config is None:
        raise RuntimeError("Runtime config is required to build replay env kwargs.")

    global_sim = (
        config.get("global", {}).get("simulation", {})
        if isinstance(config.get("global"), dict)
        else {}
    )
    local_sim = (
        config.get("local", {}).get(context.key, {}).get("simulation", {})
        if isinstance(config.get("local"), dict)
        else {}
    )
    if not isinstance(global_sim, dict):
        global_sim = {}
    if not isinstance(local_sim, dict):
        local_sim = {}
    return _deep_merge(global_sim, local_sim)


def _build_env_kwargs(
    context: ReplayContext,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path | None, dict[str, Any]]:
    sim_cfg = _build_simulation_config(context)
    config_dir = context.runtime_config_path.parent if context.runtime_config_path is not None else REPO_ROOT

    lighting_profile_cfg = _normalize_path(
        sim_cfg.get("lighting_profile_config"),
        base_dirs=[config_dir, REPO_ROOT],
    )
    hand_pose_cfg_path = _normalize_path(
        sim_cfg.get("hand_pose_config"),
        base_dirs=[config_dir, REPO_ROOT],
    )
    hand_contact_cfg_path = _normalize_path(
        sim_cfg.get("hand_contact_config"),
        base_dirs=[config_dir, REPO_ROOT],
    )
    hand_controller_cfg_path = _normalize_path(
        sim_cfg.get("hand_controller_config"),
        base_dirs=[config_dir, REPO_ROOT],
    )
    teleop_profile_cfg_path = _normalize_path(
        sim_cfg.get("teleop_profile_config"),
        base_dirs=[config_dir, REPO_ROOT],
    )

    lighting_config = None
    if isinstance(sim_cfg.get("lighting"), dict):
        lighting_config = copy.deepcopy(sim_cfg.get("lighting"))
    elif sim_cfg.get("lighting_profile") and lighting_profile_cfg is not None and lighting_profile_cfg.exists():
        lighting_config = _load_named_profile(
            lighting_profile_cfg,
            str(sim_cfg.get("lighting_profile")),
            scope="Lighting",
        )

    hand_contact = None
    if sim_cfg.get("hand_contact_profile") and hand_contact_cfg_path is not None and hand_contact_cfg_path.exists():
        hand_contact = _load_named_profile(
            hand_contact_cfg_path,
            str(sim_cfg.get("hand_contact_profile")),
            scope="HandContact",
        )

    hand_controller = None
    if (
        sim_cfg.get("hand_controller_profile")
        and hand_controller_cfg_path is not None
        and hand_controller_cfg_path.exists()
    ):
        hand_controller = _load_named_profile(
            hand_controller_cfg_path,
            str(sim_cfg.get("hand_controller_profile")),
            scope="HandController",
            summary_fields=("stiffness", "damping", "force_limit", "friction"),
        )

    teleop_profile = None
    if sim_cfg.get("teleop_profile") and teleop_profile_cfg_path is not None and teleop_profile_cfg_path.exists():
        teleop_profile = _load_named_profile(
            teleop_profile_cfg_path,
            str(sim_cfg.get("teleop_profile")),
            scope="Teleop",
        )

    object_placements = context.object_placements
    if object_placements is None:
        object_placements = copy.deepcopy(sim_cfg.get("object_placements"))

    viewer_camera_configs = {
        "viewer": {
            "width": int(args.window_width),
            "height": int(args.window_height),
        }
    }

    env_kwargs: dict[str, Any] = {
        "scene_json_path": str(context.scene_path),
        "num_envs": 1,
        "obs_mode": "state",
        "render_mode": "human",
        "render_backend": args.render_backend,
        "sim_backend": args.sim_backend,
        "viewer_camera_configs": viewer_camera_configs,
        # Match unified proxy runtime: RC5 replay must not inject reset-time hand noise.
        "robot_init_qpos_noise": 0.0,
        "robot_uids": sim_cfg.get("robot_uids", "panda"),
        "control_mode": args.control_mode or sim_cfg.get("control_mode", DEFAULT_CONTROL_MODE),
        "cameras_config": copy.deepcopy(sim_cfg.get("cameras")) if isinstance(sim_cfg.get("cameras"), dict) else None,
        "lighting_config": lighting_config,
        "robot_base_pose": sim_cfg.get("robot_base_pose"),
        "robot_init_qpos": sim_cfg.get("robot_init_qpos"),
        "object_material": sim_cfg.get("object_material"),
        "hand_contact": hand_contact,
        "hand_controller": hand_controller,
        "object_spawn_clearance": sim_cfg.get("object_spawn_clearance"),
        "bg_collision_mode": sim_cfg.get("bg_collision_mode", "nonconvex"),
        "bg_use_decimated_collision_mesh": bool(sim_cfg.get("bg_use_decimated_collision_mesh", False)),
        "bg_collision_mesh": sim_cfg.get("bg_collision_mesh", "background_registered_collision.glb"),
        "bg_visual_mesh": sim_cfg.get("bg_visual_mesh"),
        "obj_collision_mode": sim_cfg.get("obj_collision_mode", "coacd"),
        "physx_contact_offset": sim_cfg.get("physx_contact_offset"),
        "physx_rest_offset": sim_cfg.get("physx_rest_offset"),
        "placement_mode": sim_cfg.get("placement_mode", "fixed"),
        "object_placements": object_placements,
        "random_placement": sim_cfg.get("random_placement"),
        "auto_placement": bool(sim_cfg.get("auto_placement", True)),
        "include_objects": sim_cfg.get("include_objects"),
        "exclude_objects": sim_cfg.get("exclude_objects"),
        "manip_object_id": sim_cfg.get("manip_object_id", context.task_object_id),
        "target_object_id": sim_cfg.get("target_object_id"),
        "task_description": context.instruction,
        # Startup stabilization is reproduced externally to match the unified runtime.
        "settle_steps": 0,
        "lift_height": sim_cfg.get("lift_height"),
        "finger_length": sim_cfg.get("finger_length"),
        "sim_ground_offset": sim_cfg.get("sim_ground_offset"),
        "robot_base_pose_z_auto": bool(sim_cfg.get("robot_base_pose_z_auto", True)),
    }

    cleaned_env_kwargs = {key: value for key, value in env_kwargs.items() if value is not None}
    startup_cfg = {
        "requested_control_mode": cleaned_env_kwargs.get("control_mode"),
        "settle_steps": int(sim_cfg.get("settle_steps", 0) or 0),
        "gripper_open_signal": None if not isinstance(teleop_profile, dict) else teleop_profile.get("gripper_open_signal"),
    }
    return cleaned_env_kwargs, sim_cfg, hand_pose_cfg_path, startup_cfg


def _normalize_video_frame(frame: Any) -> np.ndarray:
    if frame is None:
        raise RuntimeError("Captured video frame is None.")
    if hasattr(frame, "detach") and callable(frame.detach):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"Expected HxWxC frame, got shape={frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return frame


def _decode_payload_video_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray) and frame.ndim == 1 and frame.dtype == np.uint8:
        decoded = imageio.imread(io.BytesIO(frame.tobytes()))
        return _normalize_video_frame(decoded)
    if isinstance(frame, (bytes, bytearray)):
        decoded = imageio.imread(io.BytesIO(bytes(frame)))
        return _normalize_video_frame(decoded)
    return _normalize_video_frame(frame)


def _capture_base_camera_frame(env: Any) -> np.ndarray:
    env.unwrapped.scene.update_render()
    env.unwrapped.capture_sensor_data()
    sensor = env.unwrapped.scene.sensors.get("base_camera")
    if sensor is None:
        raise RuntimeError("base_camera sensor is not available; cannot capture replay video frame.")
    obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    frame = obs.get("rgb", obs.get("Color"))
    return _normalize_video_frame(frame)


def _key_down_any(window: Any, *names: str) -> bool:
    for name in names:
        try:
            if hasattr(window, "key_press") and window.key_press(name):
                return True
        except Exception:
            pass
        try:
            if window.key_down(name):
                return True
        except Exception:
            continue
    return False


def _viewer_key_pressed_once(viewer: Any, key_states: dict[str, bool], key: str) -> bool:
    window = getattr(viewer, "window", None)
    if window is None:
        return False
    try:
        if len(key) == 1 and key.isalpha():
            is_pressed = _key_down_any(window, key.lower(), key.upper())
        else:
            is_pressed = _key_down_any(window, key)
    except Exception:
        is_pressed = False
    was_pressed = bool(key_states.get(key, False))
    key_states[key] = is_pressed
    return is_pressed and not was_pressed


def _refresh_viewer(env: Any, viewer: Any) -> None:
    if viewer is not None:
        try:
            env.render_human()
            return
        except Exception:
            pass
    try:
        env.unwrapped.scene.update_render()
    except Exception:
        pass
    try:
        viewer.notify_render_update()
    except Exception:
        pass
    try:
        if getattr(viewer, "window", None) is not None:
            viewer.render()
    except Exception:
        pass


def _resolve_replay_output_paths(
    context: ReplayContext,
    sim_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, int, str | None, list[str] | None]:
    base_output_dir = (
        _normalize_path(args.output_dir, base_dirs=[Path.cwd(), REPO_ROOT, context.npz_path.parent])
        if args.output_dir
        else (Path.cwd() / "tmp").resolve()
    )
    if base_output_dir is not None:
        base_output_dir.mkdir(parents=True, exist_ok=True)

    video_cfg = sim_cfg.get("video") if isinstance(sim_cfg.get("video"), dict) else {}
    default_video_fps = int(sim_cfg.get("video_fps") or video_cfg.get("fps") or DEFAULT_VIDEO_FPS)
    video_fps = int(args.video_fps or default_video_fps)
    video_format = str(args.video_format or video_cfg.get("format") or DEFAULT_VIDEO_FORMAT).lstrip(".")
    video_codec = args.video_codec if args.video_codec is not None else video_cfg.get("codec", DEFAULT_VIDEO_CODEC)
    output_params = video_cfg.get("output_params")
    if output_params is not None:
        output_params = [str(item) for item in output_params]

    basename = args.video_basename or f"{context.npz_path.stem}__replay"
    video_path = None
    gif_path = None
    if args.save_video and base_output_dir is not None:
        video_path = base_output_dir / f"{basename}.{video_format}"
    if args.save_gif and base_output_dir is not None:
        gif_path = base_output_dir / f"{basename}.gif"

    return video_path, gif_path, video_fps, video_codec, output_params


def _resolve_embedded_bundle_extract_dir(npz_path: Path) -> Path:
    return npz_path.with_suffix("")


def _extract_embedded_runtime_bundle(context: ReplayContext) -> tuple[Path, Path]:
    embedded_runtime_config_yaml = context.payload.get("embedded_runtime_config_yaml")
    embedded_runtime_request_json = context.payload.get("embedded_runtime_request_json")
    if not isinstance(embedded_runtime_config_yaml, str) or not embedded_runtime_config_yaml.strip():
        raise RuntimeError(
            "Cannot extract embedded runtime bundle because embedded_runtime_config_yaml is missing."
        )
    if not isinstance(embedded_runtime_request_json, str) or not embedded_runtime_request_json.strip():
        raise RuntimeError(
            "Cannot extract embedded runtime bundle because embedded_runtime_request_json is missing."
        )
    output_dir = _resolve_embedded_bundle_extract_dir(context.npz_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = output_dir / "runtime_config.yaml"
    runtime_request_path = output_dir / "runtime_request.json"
    runtime_config_path.write_text(str(embedded_runtime_config_yaml), encoding="utf-8")
    runtime_request_path.write_text(str(embedded_runtime_request_json), encoding="utf-8")
    print(f"[EmbeddedBundle] Extracted runtime_config.yaml -> {runtime_config_path}")
    print(f"[EmbeddedBundle] Extracted runtime_request.json -> {runtime_request_path}")
    return runtime_config_path, runtime_request_path


def _flush_video_buffer_to_file(
    video_frames: list[np.ndarray],
    video_path: Path,
    video_fps: int,
    video_codec: str | None,
    video_output_params: list[str] | None,
) -> Path | None:
    if len(video_frames) == 0:
        print("[ReplayVideo] Buffer is empty; skipping video save.")
        return None
    video_frames = _ensure_uniform_video_frame_sizes(video_frames)
    writer_kwargs: dict[str, Any] = {"fps": int(video_fps)}
    if video_path.suffix.lower() != ".gif":
        writer_kwargs["macro_block_size"] = None
        if video_codec:
            writer_kwargs["codec"] = str(video_codec)
        if video_output_params:
            writer_kwargs["output_params"] = list(video_output_params)
    print(f"[ReplayVideo] Saving {len(video_frames)} frames to {video_path}")
    writer = imageio.get_writer(str(video_path), **writer_kwargs)
    try:
        for frame in video_frames:
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"[ReplayVideo] Saved video: {video_path}")
    return video_path


def _resize_video_frame_nearest(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if frame.ndim != 3:
        raise RuntimeError(f"Expected HxWxC frame for resize, got shape={frame.shape}")
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    if src_h == target_h and src_w == target_w:
        return frame
    if target_h <= 0 or target_w <= 0:
        raise RuntimeError(f"Invalid resize target={target_hw}")
    y_idx = np.linspace(0, src_h - 1, num=target_h).round().astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, num=target_w).round().astype(np.int64)
    return frame[y_idx][:, x_idx]


def _ensure_uniform_video_frame_sizes(video_frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(video_frames) <= 1:
        return video_frames
    normalized_frames = [_normalize_video_frame(frame) for frame in video_frames]
    target_hw = (int(normalized_frames[0].shape[0]), int(normalized_frames[0].shape[1]))
    unique_shapes = sorted({tuple(int(dim) for dim in frame.shape) for frame in normalized_frames})
    if len(unique_shapes) == 1:
        return normalized_frames
    print(
        "[ReplayVideo] Mixed frame sizes detected in replay buffer; "
        f"resizing all frames to first-frame size HxW={target_hw[0]}x{target_hw[1]}. "
        f"Unique shapes={unique_shapes}"
    )
    return [_resize_video_frame_nearest(frame, target_hw) for frame in normalized_frames]


def _write_debug_video_gif_from_video(video_path: Path, gif_path: Path, *, fps: int, scale_width: int) -> Path:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg was not found in PATH; cannot save GIF.")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = (
        f"fps={int(fps)},scale={int(scale_width)}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=stats_mode=full[p];"
        "[s1][p]paletteuse=dither=sierra2_4a"
    )
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_chain,
        str(gif_path),
    ]
    print(f"[ReplayVideo] Saving GIF sidecar to {gif_path}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg GIF generation failed.\n"
            f"command={' '.join(command)}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    print(f"[ReplayVideo] Saved GIF: {gif_path}")
    return gif_path


def _print_context_summary(
    context: ReplayContext,
    env_kwargs: Mapping[str, Any],
    startup_cfg: Mapping[str, Any],
) -> None:
    payload = context.payload
    actions = np.asarray(payload.get("action"), dtype=np.float32)
    images = payload.get("image")
    infos = payload.get("info")
    print("=" * 80)
    print("RC5 RL4VLA Replay Viewer")
    print("=" * 80)
    print(f"npz_path: {_repo_relative_or_self(context.npz_path)}")
    print(f"scene_path: {_repo_relative_or_self(context.scene_path)}")
    print(f"runtime_config_path: {_repo_relative_or_self(context.runtime_config_path) if context.runtime_config_path else None}")
    print(f"embedded_runtime_bundle_used: {context.embedded_runtime_bundle_used}")
    print(f"key: {context.key}")
    print(f"task_object_id: {context.task_object_id}")
    print(f"task_type: {context.task_type}")
    print(f"instruction: {context.instruction}")
    print(f"action_count: {actions.shape[0]}")
    print(f"image_count: {len(images) if isinstance(images, list) else 'n/a'}")
    print(f"info_count: {len(infos) if isinstance(infos, list) else 'n/a'}")
    print(f"control_mode: {env_kwargs.get('control_mode')}")
    print(f"robot_uids: {env_kwargs.get('robot_uids')}")
    print(f"placement_mode: {env_kwargs.get('placement_mode')}")
    print(f"startup_settle_steps: {startup_cfg.get('settle_steps')}")
    print("=" * 80)


def _run_replay(context: ReplayContext, args: argparse.Namespace) -> int:
    env_kwargs, sim_cfg, hand_pose_cfg_path, startup_cfg = _build_env_kwargs(context, args)
    _print_context_summary(context, env_kwargs, startup_cfg)
    if args.extract_embedded_bundle:
        _extract_embedded_runtime_bundle(context)

    actions = np.asarray(context.payload.get("action"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(
            f"Expected action array with shape (T, 7), got shape={actions.shape}"
        )

    video_path, gif_path, video_fps, video_codec, video_output_params = _resolve_replay_output_paths(
        context,
        sim_cfg,
        args,
    )

    if args.dry_run:
        print("[DryRun] Resolved replay context successfully.")
        print(f"[DryRun] video_path={video_path}")
        print(f"[DryRun] gif_path={gif_path}")
        return 0

    env = envs.OpenReal2SimEnv(**env_kwargs)
    viewer = None
    video_frames: list[np.ndarray] = []
    key_states: dict[str, bool] = {}
    autoplay = False
    completed = False
    saved_once = False
    step_index = 0
    done_observed = False
    last_playback_ts = 0.0
    live_video_capture_warned = False

    def _prepare_startup_state() -> None:
        env.reset(seed=0, options=dict(reconfigure=True))
        agent = env.unwrapped.agent
        _log_hand_joint_state(env.unwrapped, label="after_env_reset")
        if hand_pose_cfg_path is not None and hand_pose_cfg_path.exists():
            _apply_hand_pose_config_to_agent(agent, hand_pose_cfg_path)
        _log_hand_joint_state(env.unwrapped, label="after_hand_pose_overrides")
        requested_control_mode = startup_cfg.get("requested_control_mode")
        settle_steps = int(startup_cfg.get("settle_steps", 0) or 0)
        gripper_open_signal = startup_cfg.get("gripper_open_signal")
        supported_control_modes = list(getattr(agent, "supported_control_modes", []) or [])
        stabilize_control_mode = _select_startup_stabilize_control_mode(
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
            with _temporary_agent_control_mode(env, stabilize_control_mode, reason="startup scene stabilization"):
                _stabilize_env(
                    env,
                    settle_steps,
                    stabilize_control_mode,
                    gripper_hold_signal=gripper_open_signal,
                )
        else:
            _stabilize_env(
                env,
                settle_steps,
                requested_control_mode,
                gripper_hold_signal=gripper_open_signal,
            )
        _sync_controller_targets_to_current_state(env.unwrapped)
        _log_hand_joint_state(env.unwrapped, label="after_reset_and_prepare")

    def _try_capture_frame() -> None:
        nonlocal live_video_capture_warned
        if not args.save_video and not args.save_gif:
            return
        try:
            video_frames.append(_capture_base_camera_frame(env))
        except Exception as exc:
            if not live_video_capture_warned:
                _warn(
                    "Failed to capture live base_camera replay frame; "
                    f"video/GIF saving may be unavailable. {type(exc).__name__}: {exc}"
                )
                live_video_capture_warned = True

    def _reset_replay() -> None:
        nonlocal step_index, autoplay, completed, done_observed, saved_once, video_frames, last_playback_ts, viewer
        _prepare_startup_state()
        viewer = env.render()
        if viewer is None:
            raise RuntimeError("Viewer re-initialization returned None after reset.")
        viewer.paused = False
        step_index = 0
        autoplay = False
        completed = False
        done_observed = False
        saved_once = False
        video_frames = []
        _try_capture_frame()
        last_playback_ts = time.time()
        print("[Replay] Reset to step 0. Press 'n' to autoplay or SPACE to advance one step.")

    def _save_outputs(force: bool = False) -> None:
        nonlocal saved_once
        if saved_once and not force:
            return
        if not args.save_video and not args.save_gif:
            return
        if len(video_frames) == 0:
            return
        resolved_video = None
        try:
            if args.save_video and video_path is not None:
                resolved_video = _flush_video_buffer_to_file(
                    video_frames,
                    video_path,
                    video_fps=video_fps,
                    video_codec=video_codec,
                    video_output_params=video_output_params,
                )
            if args.save_gif and gif_path is not None:
                source_video = resolved_video
                temp_video = None
                if source_video is None:
                    temp_base = video_path if video_path is not None else gif_path
                    temp_video = temp_base.with_suffix(".tmp_debug_video.mkv")
                    source_video = _flush_video_buffer_to_file(
                        video_frames,
                        temp_video,
                        video_fps=video_fps,
                        video_codec=video_codec,
                        video_output_params=video_output_params,
                    )
                if source_video is not None:
                    _write_debug_video_gif_from_video(
                        source_video,
                        gif_path,
                        fps=int(args.gif_fps),
                        scale_width=int(args.gif_scale_width),
                    )
                if temp_video is not None and temp_video.exists():
                    temp_video.unlink()
        finally:
            saved_once = True

    def _advance_one_step(*, log_step_state: bool = False) -> None:
        nonlocal step_index, autoplay, completed, done_observed
        if completed:
            print("[Replay] Trajectory already finished. Press 'n' to replay from the beginning or 'r' to reset.")
            return
        if step_index >= len(actions):
            completed = True
            autoplay = False
            print("[Replay] Trajectory finished.")
            return

        action = actions[step_index].astype(np.float32, copy=True)
        step_index_before = int(step_index)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        step_index += 1
        _refresh_viewer(env, viewer)
        _try_capture_frame()
        if log_step_state:
            _log_manual_step_state(
                env,
                action=action,
                step_index_before=step_index_before,
                step_count=len(actions),
            )
        if (terminated or truncated) and not done_observed:
            done_observed = True
            _warn(
                f"Environment reported terminated={bool(terminated)} truncated={bool(truncated)} at replay step {step_index}; continuing to the end of the recorded action sequence."
            )
        if step_index >= len(actions):
            completed = True
            autoplay = False
            print(f"[Replay] Trajectory finished at step {step_index}/{len(actions)}.")
        elif step_index % 25 == 0:
            print(f"[Replay] step {step_index}/{len(actions)}")

    try:
        _prepare_startup_state()
        viewer = env.render()
        if viewer is None:
            raise RuntimeError("Viewer initialization returned None.")
        viewer.paused = False
        _try_capture_frame()
        print("Viewer controls:")
        print("- n: autoplay full trajectory")
        print("- SPACE: pause autoplay / advance one step")
        print("- r: reset to step 0")
        print("- v: save current replay video/gif immediately")
        print("- q: quit")
        last_playback_ts = time.time()

        while viewer is not None and not getattr(viewer, "closed", True):
            _refresh_viewer(env, viewer)

            if _viewer_key_pressed_once(viewer, key_states, "q"):
                print("[Replay] Quit requested by viewer hotkey.")
                break
            if _viewer_key_pressed_once(viewer, key_states, "r"):
                _reset_replay()
            if _viewer_key_pressed_once(viewer, key_states, "v"):
                _save_outputs(force=True)
            if _viewer_key_pressed_once(viewer, key_states, "n"):
                if completed:
                    _reset_replay()
                autoplay = True
                last_playback_ts = time.time()
                print("[Replay] Autoplay started.")
            if _viewer_key_pressed_once(viewer, key_states, " "):
                if autoplay:
                    autoplay = False
                    print(f"[Replay] Autoplay paused at step {step_index}/{len(actions)}.")
                else:
                    _advance_one_step(log_step_state=True)

            if autoplay and not completed:
                now = time.time()
                min_dt = 1.0 / max(float(args.playback_fps), 1e-3)
                if now - last_playback_ts >= min_dt:
                    _advance_one_step()
                    last_playback_ts = now

            time.sleep(0.005)
    finally:
        if viewer is not None and not getattr(viewer, "closed", False):
            try:
                viewer.close()
            except Exception:
                pass
        env.close()

    return 0


def main() -> int:
    args = _parse_args()
    context = _load_replay_context(args)
    return _run_replay(context, args)


if __name__ == "__main__":
    raise SystemExit(main())
