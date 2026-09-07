#!/usr/bin/env python3
"""Conservatively compress RL4VLA raw episode delta-actions into fewer steps."""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from openreal2sim.simulation.maniskill.scripts.rc5_unified_proxy_artifacts import (
    DEFAULT_VIDEO_FPS,
    flush_video_buffer_to_file,
    resolve_effective_video_codec,
    write_debug_video_gif_from_video,
)
from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    _detect_npz_compressed,
    _write_npz_payload,
    load_rl4vla_raw_episode_artifact,
)


@dataclass(frozen=True)
class CompressionOptions:
    max_steps_per_chunk: int = 8
    max_translation_norm: float = 0.04
    translation_eps: float = 1e-6
    rotation_eps: float = 1e-6
    require_zero_gripper: bool = True
    allow_negative_dz_merge: bool = False
    negative_dz_keep_tail_ratio: float = 0.0
    max_merged_negative_dz: float | None = None


def default_output_npz_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}.compressed{npz_path.suffix}")


def default_output_gif_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}.compressed.gif")


def load_raw_payload(npz_path: Path) -> dict[str, Any]:
    payload = load_rl4vla_raw_episode_artifact(npz_path)
    actions = np.asarray(payload.get("action"), dtype=np.float32)
    images = payload.get("image")
    infos = payload.get("info")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action array with shape [N, 7], got {actions.shape}")
    if not isinstance(images, list) or len(images) != len(actions):
        raise ValueError("Expected step-aligned image list with length equal to action count")
    if not isinstance(infos, list) or len(infos) != len(actions):
        raise ValueError("Expected info list with length equal to action count")
    return payload


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


def _axis_sign(value: float, eps: float) -> int:
    if abs(value) <= eps:
        return 0
    return 1 if value > 0.0 else -1


def _is_rotation_free(action: np.ndarray, *, rotation_eps: float) -> bool:
    return bool(np.all(np.abs(action[3:6]) <= rotation_eps))


def _is_negative_dz_step(action: np.ndarray, *, translation_eps: float) -> bool:
    return float(action[2]) < -translation_eps


def _compute_negative_dz_mergeable_mask(
    actions: np.ndarray,
    *,
    options: CompressionOptions,
) -> np.ndarray:
    mergeable = np.zeros(len(actions), dtype=bool)
    if not options.allow_negative_dz_merge:
        return mergeable
    tail_ratio = float(options.negative_dz_keep_tail_ratio)
    if tail_ratio <= 0.0:
        for idx in range(len(actions)):
            if _is_negative_dz_step(actions[idx], translation_eps=options.translation_eps):
                mergeable[idx] = True
        return mergeable

    idx = 0
    while idx < len(actions):
        if not _is_negative_dz_step(actions[idx], translation_eps=options.translation_eps):
            idx += 1
            continue
        start = idx
        while idx < len(actions) and _is_negative_dz_step(actions[idx], translation_eps=options.translation_eps):
            idx += 1
        end = idx
        total_descent = float(np.sum(-actions[start:end, 2]))
        keep_tail_descent = tail_ratio * total_descent
        progressed_descent = 0.0
        for step_idx in range(start, end):
            remaining_before_step = total_descent - progressed_descent
            if remaining_before_step > keep_tail_descent + options.translation_eps:
                mergeable[step_idx] = True
            progressed_descent += float(-actions[step_idx, 2])
    return mergeable


def _can_merge_into_chunk(
    chunk_action: np.ndarray,
    next_action: np.ndarray,
    *,
    chunk_start: int,
    next_index: int,
    chunk_len: int,
    negative_dz_mergeable: np.ndarray,
    options: CompressionOptions,
) -> bool:
    if chunk_len >= options.max_steps_per_chunk:
        return False
    if options.require_zero_gripper and (
        abs(float(chunk_action[6])) > options.translation_eps or abs(float(next_action[6])) > options.translation_eps
    ):
        return False
    if not _is_rotation_free(chunk_action, rotation_eps=options.rotation_eps):
        return False
    if not _is_rotation_free(next_action, rotation_eps=options.rotation_eps):
        return False
    chunk_negative_dz = _is_negative_dz_step(chunk_action, translation_eps=options.translation_eps)
    next_negative_dz = _is_negative_dz_step(next_action, translation_eps=options.translation_eps)
    if not options.allow_negative_dz_merge and (chunk_negative_dz or next_negative_dz):
        return False
    if options.allow_negative_dz_merge and (chunk_negative_dz or next_negative_dz):
        if not negative_dz_mergeable[chunk_start] or not negative_dz_mergeable[next_index]:
            return False
    merged_translation = chunk_action[:3] + next_action[:3]
    if float(np.linalg.norm(merged_translation)) > options.max_translation_norm:
        return False
    if options.max_merged_negative_dz is not None and float(merged_translation[2]) < -options.translation_eps:
        if float(-merged_translation[2]) > float(options.max_merged_negative_dz) + options.translation_eps:
            return False
    for axis in range(3):
        current_sign = _axis_sign(float(chunk_action[axis]), options.translation_eps)
        next_sign = _axis_sign(float(next_action[axis]), options.translation_eps)
        if current_sign != 0 and next_sign != 0 and current_sign != next_sign:
            return False
    return True


def compress_actions_payload(
    payload: dict[str, Any],
    *,
    options: CompressionOptions,
) -> tuple[dict[str, Any], dict[str, int]]:
    actions = np.asarray(payload["action"], dtype=np.float32)
    images = list(payload["image"])
    infos = list(payload["info"])

    if len(actions) == 0:
        compressed_payload = dict(payload)
        return compressed_payload, {
            "original_steps": 0,
            "compressed_steps": 0,
            "merged_steps": 0,
        }

    compressed_actions: list[np.ndarray] = []
    compressed_images: list[Any] = []
    compressed_infos: list[dict[str, Any]] = []
    negative_dz_mergeable = _compute_negative_dz_mergeable_mask(actions, options=options)

    chunk_start = 0
    chunk_action = actions[0].copy()
    chunk_last_info = infos[0]
    chunk_len = 1

    for idx in range(1, len(actions)):
        next_action = actions[idx]
        if _can_merge_into_chunk(
            chunk_action,
            next_action,
            chunk_start=chunk_start,
            next_index=idx,
            chunk_len=chunk_len,
            negative_dz_mergeable=negative_dz_mergeable,
            options=options,
        ):
            chunk_action[:6] += next_action[:6]
            chunk_last_info = infos[idx]
            chunk_len += 1
            continue

        compressed_actions.append(chunk_action.copy())
        compressed_images.append(images[chunk_start])
        compressed_infos.append(dict(chunk_last_info))

        chunk_start = idx
        chunk_action = next_action.copy()
        chunk_last_info = infos[idx]
        chunk_len = 1

    compressed_actions.append(chunk_action.copy())
    compressed_images.append(images[chunk_start])
    compressed_infos.append(dict(chunk_last_info))

    compressed_payload = dict(payload)
    compressed_payload["action"] = np.asarray(compressed_actions, dtype=np.float32)
    compressed_payload["image"] = compressed_images
    compressed_payload["info"] = compressed_infos
    return compressed_payload, {
        "original_steps": int(len(actions)),
        "compressed_steps": int(len(compressed_actions)),
        "merged_steps": int(len(actions) - len(compressed_actions)),
    }


def write_compressed_payload(
    output_npz: Path,
    *,
    payload: dict[str, Any],
    compressed: bool,
) -> Path:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    return _write_npz_payload(output_npz, payload=payload, compressed=compressed)


def write_planner_style_gif_from_payload(
    output_gif: Path,
    *,
    payload: dict[str, Any],
) -> Path:
    frames_raw = payload.get("image")
    if not isinstance(frames_raw, list) or len(frames_raw) == 0:
        raise RuntimeError("Compressed payload does not contain step-aligned image frames for GIF export.")
    video_frames = [_decode_payload_video_frame(frame) for frame in frames_raw]
    output_gif = output_gif.expanduser().resolve()
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_gif.with_suffix(".tmp_debug_video.mkv")
    codec, _, _ = resolve_effective_video_codec("mkv", "auto")
    saved_video = flush_video_buffer_to_file(
        video_frames,
        temp_video,
        video_fps=DEFAULT_VIDEO_FPS,
        video_codec=codec,
        video_output_params=None,
    )
    if not saved_video:
        raise RuntimeError("Failed to flush temporary replay video while building GIF sidecar.")
    try:
        return write_debug_video_gif_from_video(temp_video, output_gif)
    finally:
        if temp_video.exists():
            temp_video.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively compress RL4VLA raw episode delta-actions by accumulating "
            "adjacent translation-only steps with zero gripper signal. By default, "
            "negative dz steps are kept unmerged to preserve contact-sensitive descents."
        )
    )
    parser.add_argument("--npz_path", required=True, help="Path to rl4vla_raw_episode*.npz")
    parser.add_argument(
        "--output_npz",
        default=None,
        help="Optional output NPZ path. Default: <npz_path>.compressed.npz",
    )
    parser.add_argument("--max_steps_per_chunk", type=int, default=8)
    parser.add_argument("--max_translation_norm", type=float, default=0.04)
    parser.add_argument("--translation_eps", type=float, default=1e-6)
    parser.add_argument("--rotation_eps", type=float, default=1e-6)
    parser.add_argument(
        "--allow_nonzero_gripper",
        action="store_true",
        help="Allow merging chunks whose gripper channel is non-zero.",
    )
    parser.add_argument(
        "--allow_negative_dz_merge",
        action="store_true",
        help="Allow merging steps whose dz is negative. Disabled by default to preserve descents.",
    )
    parser.add_argument(
        "--negative_dz_keep_tail_ratio",
        type=float,
        default=0.0,
        help=(
            "When negative dz merge is enabled, keep the final tail of each contiguous descent segment "
            "uncompressed. Example: 0.2 preserves the last 20%% of total descent depth."
        ),
    )
    parser.add_argument(
        "--max_merged_negative_dz",
        type=float,
        default=None,
        help=(
            "Optional cap on the magnitude of accumulated negative dz inside one merged chunk. "
            "Useful to keep early-descent merges conservative."
        ),
    )
    parser.add_argument(
        "--save_gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save a planner-style GIF sidecar from the compressed payload images.",
    )
    parser.add_argument(
        "--output_gif",
        default=None,
        help="Optional output GIF path. Default: <output_npz>.gif",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    output_npz = (
        Path(args.output_npz).resolve()
        if args.output_npz is not None
        else default_output_npz_path(npz_path)
    )
    output_gif = (
        Path(args.output_gif).resolve()
        if args.output_gif is not None
        else default_output_gif_path(npz_path)
    )
    if int(args.max_steps_per_chunk) <= 0:
        raise ValueError("max_steps_per_chunk must be a positive integer")
    if float(args.max_translation_norm) <= 0.0:
        raise ValueError("max_translation_norm must be > 0")
    if float(args.translation_eps) < 0.0 or float(args.rotation_eps) < 0.0:
        raise ValueError("translation_eps and rotation_eps must be >= 0")
    if not 0.0 <= float(args.negative_dz_keep_tail_ratio) <= 1.0:
        raise ValueError("negative_dz_keep_tail_ratio must be in [0, 1]")
    if args.max_merged_negative_dz is not None and float(args.max_merged_negative_dz) <= 0.0:
        raise ValueError("max_merged_negative_dz must be > 0 when provided")
    if not bool(args.allow_negative_dz_merge) and (
        float(args.negative_dz_keep_tail_ratio) > 0.0 or args.max_merged_negative_dz is not None
    ):
        raise ValueError(
            "negative_dz_keep_tail_ratio and max_merged_negative_dz require --allow_negative_dz_merge"
        )

    payload = load_raw_payload(npz_path)
    compressed_payload, summary = compress_actions_payload(
        payload,
        options=CompressionOptions(
            max_steps_per_chunk=int(args.max_steps_per_chunk),
            max_translation_norm=float(args.max_translation_norm),
            translation_eps=float(args.translation_eps),
            rotation_eps=float(args.rotation_eps),
            require_zero_gripper=not bool(args.allow_nonzero_gripper),
            allow_negative_dz_merge=bool(args.allow_negative_dz_merge),
            negative_dz_keep_tail_ratio=float(args.negative_dz_keep_tail_ratio),
            max_merged_negative_dz=(
                None if args.max_merged_negative_dz is None else float(args.max_merged_negative_dz)
            ),
        ),
    )
    write_compressed_payload(
        output_npz,
        payload=compressed_payload,
        compressed=_detect_npz_compressed(npz_path),
    )
    if bool(args.save_gif):
        write_planner_style_gif_from_payload(output_gif, payload=compressed_payload)

    print(
        "[compress-npz-actions] OK",
        f"input_npz={npz_path}",
        f"output_npz={output_npz}",
        f"output_gif={output_gif if bool(args.save_gif) else '<skipped>'}",
        f"original_steps={summary['original_steps']}",
        f"compressed_steps={summary['compressed_steps']}",
        f"merged_steps={summary['merged_steps']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
