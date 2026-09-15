#!/usr/bin/env python3
"""Convert augmented RL4VLA raw episodes into OpenVLA_V2 SFT npz.

V2 differences vs prepare_sft_episodes.py:
  - scene and wrist images are kept as SEPARATE channels (no wrist inset compositing)
  - per-step 7D proprio (6 arm qpos + gripper closure) is carried through
  - gripper action is quantized to the discrete openness levels {0.0, 0.2, ..., 1.0}
    (1 = fully open, 0 = fully closed) used by the RCLevelHandController

Input episodes are the output of render_bg_variants.py (or the raw recollected
episodes, which already carry image/image_wrist/action/proprio/info).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real")
sys.path.insert(0, str(REPO))

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (  # noqa: E402
    decode_rl4vla_raw_episode_images,
    load_rl4vla_raw_episode_artifact,
)

MAX_STEPS_PER_CHUNK = int(os.environ.get("RLVLA_MAX_STEPS_PER_CHUNK", "4"))
MAX_TRANSLATION_NORM = 0.04
TRANSLATION_EPS = 1e-6
GRIPPER_LEVELS = 5  # openness levels are round(o * 5) / 5 -> {0.0, 0.2, ..., 1.0}
EXPECTED_SCENE_HWC = (480, 640, 3)
EXPECTED_WRIST_HWC = (224, 168, 3)


def quantize_gripper_levels(actions: np.ndarray) -> np.ndarray:
    """Snap the gripper dim to the discrete openness levels {0.0, 0.2, ..., 1.0}."""
    out = np.asarray(actions, dtype=np.float32).copy()
    out[:, 6] = np.clip(np.round(out[:, 6] * GRIPPER_LEVELS) / GRIPPER_LEVELS, 0.0, 1.0)
    return out


def compress_same_gripper_translations(
    actions: np.ndarray,
    scene_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    proprio: np.ndarray,
    infos: list[dict],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, list[dict]]:
    n = int(actions.shape[0])
    if n == 0:
        return actions, scene_images, wrist_images, proprio, infos

    out_actions: list[np.ndarray] = []
    out_scene: list[np.ndarray] = []
    out_wrist: list[np.ndarray] = []
    out_proprio: list[np.ndarray] = []
    out_infos: list[dict] = []
    idx = 0
    while idx < n:
        chunk = np.asarray(actions[idx], dtype=np.float32).copy()
        start = idx
        idx += 1
        chunk_len = 1
        while idx < n:
            nxt = np.asarray(actions[idx], dtype=np.float32)
            if abs(float(nxt[6]) - float(chunk[6])) > TRANSLATION_EPS:
                break
            if chunk_len >= MAX_STEPS_PER_CHUNK:
                break
            merged = chunk.copy()
            merged[:3] = chunk[:3] + nxt[:3]
            if float(np.linalg.norm(merged[:3])) > MAX_TRANSLATION_NORM:
                break
            for axis in range(3):
                cur = float(chunk[axis])
                nxt_v = float(nxt[axis])
                if abs(cur) > TRANSLATION_EPS and abs(nxt_v) > TRANSLATION_EPS and np.sign(cur) != np.sign(nxt_v):
                    break
            else:
                chunk = merged
                idx += 1
                chunk_len += 1
                continue
            break
        out_actions.append(chunk)
        out_scene.append(scene_images[start])
        out_wrist.append(wrist_images[start])
        out_proprio.append(np.asarray(proprio[start], dtype=np.float32))
        last_info = dict(infos[idx - 1]) if isinstance(infos[idx - 1], dict) else {"success": False}
        last_info["success"] = bool(last_info.get("success", False))
        out_infos.append(last_info)
    return (
        np.stack(out_actions, axis=0).astype(np.float32),
        out_scene,
        out_wrist,
        np.stack(out_proprio, axis=0).astype(np.float32),
        out_infos,
    )


def _require_frame_shape(src_path: Path, name: str, frames, expected_hwc: tuple[int, int, int]) -> None:
    # decode_rl4vla_raw_episode_images returns a stacked ndarray; `if not frames`
    # is ambiguous for arrays with more than one element.
    n = 0 if frames is None else len(frames)
    if n == 0:
        raise ValueError(f"{src_path.name}: empty {name}")
    got = tuple(np.asarray(frames[0]).shape)
    if got != expected_hwc:
        raise ValueError(
            f"{src_path.name}: {name} shape {got} != {expected_hwc}. "
            "Raw collection dumps wrist as 640x480 JPEG; re-render with render_bg_variants.py "
            "so RL sees wrist_camera at 224x168 and 7D proprio."
        )


def convert_episode(src_path: Path) -> dict:
    payload = load_rl4vla_raw_episode_artifact(src_path)
    actions = quantize_gripper_levels(np.asarray(payload["action"], dtype=np.float32))
    scene_images = decode_rl4vla_raw_episode_images(payload, key="image")
    wrist_images = decode_rl4vla_raw_episode_images(payload, key="image_wrist")
    if "proprio" not in payload:
        raise KeyError(
            f"{src_path.name}: missing proprio. Raw V2 collection only stores robot_qpos; "
            "run render_bg_variants.py before prepare_sft_v2_episodes.py."
        )
    proprio = np.asarray(payload["proprio"], dtype=np.float32)
    if proprio.ndim != 2 or proprio.shape[0] != len(actions) or proprio.shape[1] != 7:
        raise ValueError(f"{src_path.name}: proprio shape {proprio.shape} != (T={len(actions)}, 7)")
    if len(scene_images) != len(actions) or len(wrist_images) != len(actions):
        raise ValueError(
            f"{src_path.name}: length mismatch "
            f"scene={len(scene_images)} wrist={len(wrist_images)} action={len(actions)}"
        )
    _require_frame_shape(src_path, "image", scene_images, EXPECTED_SCENE_HWC)
    _require_frame_shape(src_path, "image_wrist", wrist_images, EXPECTED_WRIST_HWC)
    infos = list(payload.get("info") or [])
    if len(infos) != len(actions):
        raise ValueError(f"{src_path.name}: info length {len(infos)} != action {len(actions)}")
    actions, scene_images, wrist_images, proprio, infos = compress_same_gripper_translations(
        actions, scene_images, wrist_images, proprio, infos
    )
    instruction = payload.get("instruction", "Pick red cube")
    if isinstance(instruction, np.ndarray):
        instruction = instruction.tolist()
        if isinstance(instruction, list):
            instruction = instruction[0]
    return {
        "instruction": str(instruction),
        "action": actions,
        "image": scene_images,
        "image_wrist": wrist_images,
        "proprio": proprio,
        "info": infos,
    }


def main() -> None:
    src_dir = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    dest_dir.mkdir(parents=True, exist_ok=True)
    overwrite = os.environ.get("RLVLA_PREPARE_OVERWRITE", "0") == "1"
    files = sorted(src_dir.glob("episode_*.npz")) + sorted(src_dir.glob("rl4vla_raw_episode*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode npz in {src_dir}")
    lengths = []
    for src in files:
        dest = dest_dir / src.name
        if dest.exists() and not overwrite:
            print(f"skip existing {dest.name}", flush=True)
            continue
        record = convert_episode(src)
        np.savez_compressed(dest, arr_0=np.array(record, dtype=object))
        lengths.append(int(record["action"].shape[0]))
        print(f"{src.name} -> {dest.name} steps={lengths[-1]}", flush=True)
    if not lengths:
        print(f"No new episodes converted in {src_dir}; dest already populated", flush=True)
        return
    lengths_arr = np.asarray(lengths, dtype=np.int32)
    stats = {
        "n_episodes": int(len(lengths_arr)),
        "steps_min": int(lengths_arr.min()),
        "steps_median": float(np.median(lengths_arr)),
        "steps_mean": float(np.mean(lengths_arr)),
        "steps_p95": float(np.percentile(lengths_arr, 95)),
        "steps_p99": float(np.percentile(lengths_arr, 99)),
        "steps_max": int(lengths_arr.max()),
        "recommended_episode_len": int(min(200, max(80, int(np.ceil(np.percentile(lengths_arr, 99) / 8.0) * 8) + 16))),
        "src_dir": str(src_dir),
        "dest_dir": str(dest_dir),
        "max_steps_per_chunk": MAX_STEPS_PER_CHUNK,
    }
    (dest_dir / "sft_episode_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
