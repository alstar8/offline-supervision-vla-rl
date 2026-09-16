#!/usr/bin/env python3
"""Convert augmented RL4VLA raw episodes into OpenVLA_V2 SFT npz.

V2 differences vs prepare_sft_episodes.py:
  - scene and wrist images are kept as SEPARATE channels (no wrist inset compositing)
  - per-step 7D proprio (6 arm qpos + gripper closure) is carried through
  - gripper action is quantized to the discrete openness levels {0.0, 0.2, ..., 1.0}
    (1 = fully open, 0 = fully closed) used by the RCLevelHandController
  - actions are chunked at a fixed stride and near-stationary chunks are dropped
    (see chunk_fixed_stride)

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
# Chunks translating less than this are dropped unless they carry a gripper
# transition. 1 mm is well under the 256-bin resolution of the action tokenizer
# (~0.15 mm on x, ~0.06 mm on z), so nothing informative is discarded.
MIN_CHUNK_TRANSLATION = float(os.environ.get("RLVLA_MIN_CHUNK_TRANSLATION", "0.001"))
TRANSLATION_EPS = 1e-6
GRIPPER_LEVELS = 5  # openness levels are round(o * 5) / 5 -> {0.0, 0.2, ..., 1.0}
EXPECTED_SCENE_HWC = (480, 640, 3)
EXPECTED_WRIST_HWC = (224, 168, 3)


def quantize_gripper_levels(actions: np.ndarray) -> np.ndarray:
    """Snap the gripper dim to the discrete openness levels {0.0, 0.2, ..., 1.0}."""
    out = np.asarray(actions, dtype=np.float32).copy()
    out[:, 6] = np.clip(np.round(out[:, 6] * GRIPPER_LEVELS) / GRIPPER_LEVELS, 0.0, 1.0)
    return out


def chunk_fixed_stride(
    actions: np.ndarray,
    scene_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    proprio: np.ndarray,
    infos: list[dict],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, list[dict]]:
    """Sum a fixed number of consecutive frames into one action label.

    Chunk length is a function of the timestep alone; it breaks only on a
    gripper-level change or the end of the episode. The earlier rule also cut a
    chunk short whenever a later frame reversed sign on any axis, which made the
    label depend on future frames -- 21% of chunks were truncated that way, so
    one observation could map to a 1-, 2-, 3- or 4-frame sum and the target was
    not a function of the observation.

    Chunks that barely translate are dropped unless they carry a gripper
    transition. Keeping them left a quarter of the x/y labels and a third of the
    z labels exactly zero, making the zero bin the single most likely token;
    greedy decoding then collapsed onto it at rollout time and the arm stalled.

    Observation (image/proprio) is taken from the first frame of the chunk and
    `info` from the last, so the label is the motion that follows the frame the
    policy sees.
    """
    n = int(actions.shape[0])
    if n == 0:
        return actions, scene_images, wrist_images, proprio, infos

    out_actions: list[np.ndarray] = []
    out_scene: list[np.ndarray] = []
    out_wrist: list[np.ndarray] = []
    out_proprio: list[np.ndarray] = []
    out_infos: list[dict] = []
    n_chunks = 0
    n_dropped = 0
    n_clamped = 0
    prev_gripper: float | None = None

    idx = 0
    while idx < n:
        start = idx
        chunk = np.asarray(actions[start], dtype=np.float32).copy()
        idx += 1
        while idx < n and (idx - start) < MAX_STEPS_PER_CHUNK:
            nxt = np.asarray(actions[idx], dtype=np.float32)
            if abs(float(nxt[6]) - float(chunk[6])) > TRANSLATION_EPS:
                break
            chunk[:3] = chunk[:3] + nxt[:3]
            idx += 1
        n_chunks += 1

        # Inert on this data (4 frames cap out around 0.024 m) but keeps the
        # bound a hard guarantee without making chunk length content-dependent.
        norm = float(np.linalg.norm(chunk[:3]))
        if norm > MAX_TRANSLATION_NORM:
            chunk[:3] *= MAX_TRANSLATION_NORM / norm
            norm = MAX_TRANSLATION_NORM
            n_clamped += 1

        gripper = float(chunk[6])
        gripper_event = prev_gripper is None or abs(gripper - prev_gripper) > TRANSLATION_EPS
        if norm < MIN_CHUNK_TRANSLATION and not gripper_event:
            n_dropped += 1
            continue

        out_actions.append(chunk)
        out_scene.append(scene_images[start])
        out_wrist.append(wrist_images[start])
        out_proprio.append(np.asarray(proprio[start], dtype=np.float32))
        last_info = dict(infos[idx - 1]) if isinstance(infos[idx - 1], dict) else {"success": False}
        last_info["success"] = bool(last_info.get("success", False))
        out_infos.append(last_info)
        prev_gripper = gripper

    print(
        f"  chunks={n_chunks} kept={len(out_actions)} dropped_noop={n_dropped} clamped={n_clamped}",
        flush=True,
    )
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
    actions, scene_images, wrist_images, proprio, infos = chunk_fixed_stride(
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
        "min_chunk_translation": MIN_CHUNK_TRANSLATION,
        "chunking": "fixed_stride",
    }
    (dest_dir / "sft_episode_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
