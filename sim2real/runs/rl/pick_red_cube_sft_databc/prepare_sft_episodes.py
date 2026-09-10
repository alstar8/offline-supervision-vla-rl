#!/usr/bin/env python3
"""Convert harvested RL4VLA raw episodes into SFT npz matching PPO observations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real")
sys.path.insert(0, str(REPO))

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (  # noqa: E402
    decode_rl4vla_raw_episode_images,
    load_rl4vla_raw_episode_artifact,
)

WRIST_INSET_MARGIN = 4
WRIST_INSET_BORDER = 4
WRIST_INSET_HEIGHT = 224
WRIST_INSET_WIDTH = 168
MAX_STEPS_PER_CHUNK = 8
MAX_TRANSLATION_NORM = 0.04
TRANSLATION_EPS = 1e-6


def _resize_hw(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    resized = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def compose_wrist_inset_bottom_right(scene_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    scene = np.asarray(scene_rgb, dtype=np.uint8).copy()
    wrist = _resize_hw(wrist_rgb, WRIST_INSET_HEIGHT, WRIST_INSET_WIDTH)
    border = WRIST_INSET_BORDER
    inset_h = WRIST_INSET_HEIGHT
    inset_w = WRIST_INSET_WIDTH
    margin = WRIST_INSET_MARGIN
    top = scene.shape[0] - inset_h - 2 * border - margin
    left = scene.shape[1] - inset_w - 2 * border - margin
    scene[top : top + inset_h + 2 * border, left : left + inset_w + 2 * border, :] = 0
    scene[top + border : top + border + inset_h, left + border : left + border + inset_w, :] = wrist
    return scene


def remap_gripper_to_bridge(actions: np.ndarray) -> np.ndarray:
    """Map RC5 controller signals to Bridge open_gripper in {0, 1}.

    Recorded values are -1 (close) and 0 (hold, hand already open). PPO binarizes
    unnormalized gripper at 0.5, so hold must become open (1.0).
    """
    out = np.asarray(actions, dtype=np.float32).copy()
    out[:, 6] = np.where(out[:, 6] < -0.5, 0.0, 1.0)
    return out


def compress_same_gripper_translations(
    actions: np.ndarray,
    images: list[np.ndarray],
    infos: list[dict],
) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    n = int(actions.shape[0])
    if n == 0:
        return actions, images, infos

    out_actions: list[np.ndarray] = []
    out_images: list[np.ndarray] = []
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
        out_images.append(images[start])
        last_info = dict(infos[idx - 1]) if isinstance(infos[idx - 1], dict) else {"success": False}
        last_info["success"] = bool(last_info.get("success", False))
        out_infos.append(last_info)
    return np.stack(out_actions, axis=0).astype(np.float32), out_images, out_infos


def convert_episode(src_path: Path) -> dict:
    payload = load_rl4vla_raw_episode_artifact(src_path)
    actions = remap_gripper_to_bridge(np.asarray(payload["action"], dtype=np.float32))
    scene_images = decode_rl4vla_raw_episode_images(payload, key="image")
    wrist_images = decode_rl4vla_raw_episode_images(payload, key="image_wrist")
    if len(scene_images) != len(actions) or len(wrist_images) != len(actions):
        raise ValueError(
            f"{src_path.name}: length mismatch "
            f"scene={len(scene_images)} wrist={len(wrist_images)} action={len(actions)}"
        )
    composed = [
        compose_wrist_inset_bottom_right(scene_images[i], wrist_images[i])
        for i in range(len(actions))
    ]
    infos = list(payload.get("info") or [])
    if len(infos) != len(actions):
        raise ValueError(f"{src_path.name}: info length {len(infos)} != action {len(actions)}")
    actions, composed, infos = compress_same_gripper_translations(actions, composed, infos)
    instruction = payload.get("instruction", "Pick red cube")
    if isinstance(instruction, np.ndarray):
        instruction = instruction.tolist()
        if isinstance(instruction, list):
            instruction = instruction[0]
    return {
        "instruction": str(instruction),
        "action": actions,
        "image": composed,
        "info": infos,
    }


def main() -> None:
    src_dir = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz in {src_dir}")
    lengths = []
    for src in files:
        record = convert_episode(src)
        dest = dest_dir / src.name
        np.savez_compressed(dest, arr_0=np.array(record, dtype=object))
        lengths.append(int(record["action"].shape[0]))
        print(f"{src.name} -> {dest.name} steps={lengths[-1]}", flush=True)
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
    }
    (dest_dir / "sft_episode_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
