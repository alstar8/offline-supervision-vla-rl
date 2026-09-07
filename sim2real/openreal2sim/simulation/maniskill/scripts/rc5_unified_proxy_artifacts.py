from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
import torch

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    _encode_frame_to_jpeg_uint8_buffer,
    _resize_frame_to_canvas,
    write_dense_episode_artifact,
    write_rl4vla_raw_episode_artifact,
)

AUTO_VIDEO_CODEC = "auto"
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_FORMAT = "mp4"

_ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE = None


@dataclass
class UnifiedDenseEpisodeCapture:
    output_path: Path | None
    output_dir: Path | None
    rl4vla_raw_output_path: Path | None
    rl4vla_raw_output_dir: Path | None
    instructions_per_env: list[str]
    runtime_config_paths_per_env: list[Path | None]
    runtime_request_paths_per_env: list[Path | None]
    camera_name: str
    image_target_width: int
    image_target_height: int
    num_envs: int
    images_per_env: list[list]
    rl4vla_preencoded_images_per_env: list[list] | None
    actions_per_env: list[list]
    infos_per_env: list[list]
    shared_video_frame_targets_per_env: list[list] | None
    finalized_env_indices: set[int]
    raw_writer_executor: ThreadPoolExecutor | None
    pending_raw_writer_futures: list[Future]


def normalize_video_frame(frame):
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"Expected video frame with 3 dimensions, got shape={frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    return frame


def capture_base_camera_frames(env):
    env.unwrapped.scene.update_render()
    env.unwrapped.capture_sensor_data()
    sensor = env.unwrapped.scene.sensors.get("base_camera")
    if sensor is None:
        raise RuntimeError("base_camera sensor is not available; cannot capture video frame.")
    obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    frame = obs.get("rgb", obs.get("Color"))
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        return [normalize_video_frame(frame[idx]) for idx in range(frame.shape[0])]
    return [normalize_video_frame(frame)]


def capture_base_camera_frame(env):
    frames = capture_base_camera_frames(env)
    if not frames:
        raise RuntimeError("base_camera sensor did not produce any frames.")
    return frames[0]


def resolve_batched_dense_episode_output_path(output_dir, *, env_index: int) -> Path:
    root = Path(output_dir).expanduser().resolve()
    return root / f"env_{int(env_index):06d}_dense_episode.npz"


def resolve_batched_rl4vla_raw_episode_output_path(output_dir, *, env_index: int) -> Path:
    root = Path(output_dir).expanduser().resolve()
    return root / f"env_{int(env_index):06d}_rl4vla_raw_episode.npz"


def resolve_batched_debug_video_output_path(output_dir, *, env_index: int) -> Path:
    root = Path(output_dir).expanduser().resolve()
    return root / f"env_{int(env_index):06d}_debug_video.mkv"


def resolve_batched_debug_video_gif_output_path(output_dir, *, env_index: int) -> Path:
    root = Path(output_dir).expanduser().resolve()
    return root / f"env_{int(env_index):06d}_debug_video.gif"


def resolve_batched_object_pose_trace_output_path(output_dir, *, env_index: int) -> Path:
    root = Path(output_dir).expanduser().resolve()
    return root / f"env_{int(env_index):06d}_scene_object_pose_trace.jsonl"


def _resolve_optional_runtime_paths_per_env(
    *,
    num_envs: int,
    singleton_path,
    per_env_paths,
    label: str,
) -> list[Path | None]:
    if per_env_paths is not None:
        values = list(per_env_paths)
        if len(values) != int(num_envs):
            raise ValueError(
                f"{label} must contain exactly {int(num_envs)} item(s); got {len(values)}."
            )
        return [None if item is None else Path(item).expanduser().resolve() for item in values]
    return [None if singleton_path is None else Path(singleton_path).expanduser().resolve() for _ in range(int(num_envs))]


def _read_optional_text_file(path: Path | None, *, label: str) -> str | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _list_ffmpeg_codec_names(ffmpeg_exe, flag):
    try:
        result = subprocess.run([ffmpeg_exe, flag], capture_output=True, text=True, check=False)
    except Exception:
        return set()
    names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


def resolve_effective_video_codec(video_format, requested_codec):
    if video_format.lower() == "gif":
        return None, None, "GIF output uses the native GIF encoder"

    if requested_codec and requested_codec.lower() != AUTO_VIDEO_CODEC:
        return requested_codec, None, f"explicit CLI override ({requested_codec})"

    imageio_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    available_encoders = _list_ffmpeg_codec_names(imageio_ffmpeg_exe, "-encoders")
    system_ffmpeg = shutil.which("ffmpeg")
    system_decoders = _list_ffmpeg_codec_names(system_ffmpeg, "-decoders") if system_ffmpeg else set()

    codec_candidates = []
    if "h264" in system_decoders and "libx264" in available_encoders:
        codec_candidates.append(("libx264", "system ffmpeg has native h264 decoder"))
    if "mpeg4" in system_decoders and "mpeg4" in available_encoders:
        codec_candidates.append(("mpeg4", "system ffmpeg has native mpeg4 decoder"))
    if "libx264" in available_encoders:
        codec_candidates.append(("libx264", "imageio_ffmpeg encoder availability"))
    if "mpeg4" in available_encoders:
        codec_candidates.append(("mpeg4", "imageio_ffmpeg encoder availability"))
    if "mjpeg" in available_encoders:
        codec_candidates.append(("mjpeg", "imageio_ffmpeg encoder availability"))

    if not codec_candidates:
        return None, imageio_ffmpeg_exe, "no preferred codec found; using imageio/ffmpeg default"

    codec, reason = codec_candidates[0]
    return codec, imageio_ffmpeg_exe, reason


def _normalize_video_output_params(output_params):
    if output_params is None:
        return None
    if not isinstance(output_params, (list, tuple)):
        raise ValueError(f"video.output_params must be a list/tuple, got {type(output_params).__name__}")
    return [str(x) for x in output_params]


def resolve_effective_video_settings(args, config_overrides):
    video_cfg = dict(config_overrides.get("video_config") or {})

    if args.video_fps != DEFAULT_VIDEO_FPS:
        effective_video_fps = int(args.video_fps)
    else:
        effective_video_fps = int(video_cfg.get("fps", DEFAULT_VIDEO_FPS))

    if args.video_format != DEFAULT_VIDEO_FORMAT:
        effective_video_format = str(args.video_format)
    else:
        effective_video_format = str(video_cfg.get("format", DEFAULT_VIDEO_FORMAT))

    if args.video_codec != AUTO_VIDEO_CODEC:
        requested_video_codec = args.video_codec
    else:
        requested_video_codec = str(video_cfg.get("codec", AUTO_VIDEO_CODEC))

    effective_video_output_params = _normalize_video_output_params(video_cfg.get("output_params"))

    return {
        "fps": effective_video_fps,
        "format": effective_video_format,
        "requested_codec": requested_video_codec,
        "output_params": effective_video_output_params,
    }


def resolve_output_video_path(video_path, *, video_format=None, video_codec=None):
    video_path = Path(video_path).expanduser().resolve()
    requested_format = None if video_format is None else str(video_format).strip().lower()
    requested_codec = None if video_codec is None else str(video_codec).strip().lower()
    suffix = video_path.suffix.lower()

    if requested_codec == "ffv1":
        if suffix == ".mp4":
            adjusted_path = video_path.with_suffix(".mkv")
            print(
                f"[WARNING] [VIDEO] Requested codec 'ffv1' is incompatible with '.mp4'; "
                f"saving viewer video as '{adjusted_path.name}' instead."
            )
            return adjusted_path
        if suffix == "":
            return video_path.with_suffix(".mkv")
    if suffix == "":
        if requested_format:
            return video_path.with_suffix(f".{requested_format}")
        return video_path.with_suffix(".mp4")
    return video_path


def flush_video_buffer_to_file(video_frames, video_path, video_fps, video_codec=None, video_output_params=None):
    if len(video_frames) == 0:
        print("[VIDEO] Buffer is empty; nothing to save.")
        return False
    video_path = resolve_output_video_path(video_path, video_codec=video_codec)
    print(f"[VIDEO] Saving {len(video_frames)} buffered frames to {video_path}")
    writer_kwargs = {"fps": video_fps}
    if video_path.suffix.lower() != ".gif":
        writer_kwargs["macro_block_size"] = None
        if video_codec:
            writer_kwargs["codec"] = video_codec
        if video_output_params:
            writer_kwargs["output_params"] = list(video_output_params)
    else:
        print("[VIDEO] GIF encoding can be noticeably slower during finalization, especially at large resolutions.")
    writer = imageio.get_writer(str(video_path), **writer_kwargs)
    try:
        for frame in video_frames:
            writer.append_data(frame)
        if video_path.suffix.lower() == ".gif":
            print("[VIDEO] Finalizing GIF...")
    finally:
        writer.close()
    print(f"[VIDEO] Saved: {video_path}")
    return True


def write_debug_video_gif_from_video(video_path, gif_path):
    source_path = Path(video_path).expanduser().resolve()
    output_path = Path(gif_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg is required for --save_video_gif_on_exit but was not found in PATH.")
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=sierra2_4a",
        str(output_path),
    ]
    print(f"[VIDEO] Saving GIF sidecar to {output_path}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg GIF generation failed for debug video sidecar.\n"
            f"command={' '.join(command)}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    print(f"[VIDEO] Saved GIF: {output_path}")
    return output_path


def clear_unified_dense_episode_capture():
    global _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    capture = _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    if capture is not None:
        _wait_for_pending_raw_writes(capture)
        _shutdown_capture_raw_writer(capture)
    _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE = None


def start_unified_dense_episode_capture(
    *,
    output_path=None,
    output_dir=None,
    instruction=None,
    instructions_per_env=None,
    runtime_config_path=None,
    runtime_request_path=None,
    runtime_config_paths_per_env=None,
    runtime_request_paths_per_env=None,
    rl4vla_raw_output_path=None,
    rl4vla_raw_output_dir=None,
    camera_name="base_camera",
    image_target_width=640,
    image_target_height=480,
    num_envs=1,
    shared_video_frame_targets_per_env=None,
):
    global _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    resolved_output_path = None if output_path is None else Path(output_path).expanduser().resolve()
    resolved_output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
    resolved_rl4vla_raw_output_path = (
        None if rl4vla_raw_output_path is None else Path(rl4vla_raw_output_path).expanduser().resolve()
    )
    resolved_rl4vla_raw_output_dir = (
        None if rl4vla_raw_output_dir is None else Path(rl4vla_raw_output_dir).expanduser().resolve()
    )
    if (
        resolved_output_path is None
        and resolved_output_dir is None
        and resolved_rl4vla_raw_output_path is None
        and resolved_rl4vla_raw_output_dir is None
    ):
        raise ValueError("Dense episode capture requires at least one artifact output path or directory.")
    normalized_num_envs = int(num_envs)
    if normalized_num_envs <= 0:
        raise ValueError("Dense episode capture requires num_envs >= 1.")
    if normalized_num_envs > 1 and (
        resolved_output_path is not None or resolved_rl4vla_raw_output_path is not None
    ):
        raise ValueError("Batched dense episode capture requires output_dir-style artifact targets.")
    if normalized_num_envs > 1 and (
        resolved_output_dir is None and resolved_rl4vla_raw_output_dir is None
    ):
        raise ValueError("Batched dense episode capture requires at least one output_dir-style artifact target.")
    normalized_shared_video_targets = None
    if shared_video_frame_targets_per_env is not None:
        if not isinstance(shared_video_frame_targets_per_env, (list, tuple)):
            raise ValueError("shared_video_frame_targets_per_env must be a list/tuple of per-env lists")
        normalized_shared_video_targets = list(shared_video_frame_targets_per_env)
        if len(normalized_shared_video_targets) != normalized_num_envs:
            raise ValueError(
                "shared_video_frame_targets_per_env length must match num_envs: "
                f"len={len(normalized_shared_video_targets)} num_envs={normalized_num_envs}"
            )
    if instructions_per_env is None:
        if instruction is None:
            raise ValueError("Dense episode capture requires instruction or instructions_per_env.")
        normalized_instructions_per_env = [str(instruction)] * normalized_num_envs
    else:
        if not isinstance(instructions_per_env, (list, tuple)):
            raise ValueError("instructions_per_env must be a list/tuple of strings")
        normalized_instructions_per_env = [str(item) for item in instructions_per_env]
        if len(normalized_instructions_per_env) != normalized_num_envs:
            raise ValueError(
                "instructions_per_env length must match num_envs: "
                f"len={len(normalized_instructions_per_env)} num_envs={normalized_num_envs}"
            )
    for env_index, item in enumerate(normalized_instructions_per_env):
        if not item.strip():
            raise ValueError(f"instructions_per_env[{env_index}] must be a non-empty string")
    normalized_runtime_config_paths_per_env = _resolve_optional_runtime_paths_per_env(
        num_envs=normalized_num_envs,
        singleton_path=runtime_config_path,
        per_env_paths=runtime_config_paths_per_env,
        label="runtime_config_paths_per_env",
    )
    normalized_runtime_request_paths_per_env = _resolve_optional_runtime_paths_per_env(
        num_envs=normalized_num_envs,
        singleton_path=runtime_request_path,
        per_env_paths=runtime_request_paths_per_env,
        label="runtime_request_paths_per_env",
    )
    raw_writer_executor = None
    if normalized_num_envs > 1 and resolved_rl4vla_raw_output_dir is not None:
        raw_writer_executor = ThreadPoolExecutor(
            max_workers=_resolve_raw_writer_worker_count(normalized_num_envs),
            thread_name_prefix="rc5-raw-writer",
        )
    _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE = UnifiedDenseEpisodeCapture(
        output_path=resolved_output_path,
        output_dir=resolved_output_dir,
        rl4vla_raw_output_path=resolved_rl4vla_raw_output_path,
        rl4vla_raw_output_dir=resolved_rl4vla_raw_output_dir,
        instructions_per_env=normalized_instructions_per_env,
        runtime_config_paths_per_env=normalized_runtime_config_paths_per_env,
        runtime_request_paths_per_env=normalized_runtime_request_paths_per_env,
        camera_name=str(camera_name),
        image_target_width=int(image_target_width),
        image_target_height=int(image_target_height),
        num_envs=normalized_num_envs,
        images_per_env=[[] for _ in range(normalized_num_envs)],
        rl4vla_preencoded_images_per_env=(
            None
            if resolved_rl4vla_raw_output_path is None and resolved_rl4vla_raw_output_dir is None
            else [[] for _ in range(normalized_num_envs)]
        ),
        actions_per_env=[[] for _ in range(normalized_num_envs)],
        infos_per_env=[[] for _ in range(normalized_num_envs)],
        shared_video_frame_targets_per_env=normalized_shared_video_targets,
        finalized_env_indices=set(),
        raw_writer_executor=raw_writer_executor,
        pending_raw_writer_futures=[],
    )


def get_active_unified_dense_episode_capture():
    return _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE


def _append_frame_to_dense_capture(capture: UnifiedDenseEpisodeCapture, *, env_index: int, frame) -> None:
    capture.images_per_env[env_index].append(frame)
    shared_targets = capture.shared_video_frame_targets_per_env
    if shared_targets is not None:
        target = shared_targets[env_index]
        if target is not capture.images_per_env[env_index]:
            target.append(frame)
    preencoded = capture.rl4vla_preencoded_images_per_env
    if preencoded is not None:
        preencoded[env_index].append(
            _encode_frame_to_jpeg_uint8_buffer(
                _resize_frame_to_canvas(
                    frame,
                    target_width=capture.image_target_width,
                    target_height=capture.image_target_height,
                )
            )
        )


def append_dense_episode_initial_frame_if_enabled(env):
    capture = _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    if capture is None:
        return
    frames = capture_base_camera_frames(env)
    if len(frames) != capture.num_envs:
        raise RuntimeError(
            "Dense episode capture expected one base_camera frame per env, "
            f"got {len(frames)} frames for num_envs={capture.num_envs}."
        )
    for env_index, frame in enumerate(frames):
        if env_index in capture.finalized_env_indices:
            continue
        _append_frame_to_dense_capture(capture, env_index=env_index, frame=frame)


def _normalize_dense_episode_info(info):
    if isinstance(info, dict):
        normalized = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            if arr.size == 0:
                normalized[key] = None
            else:
                normalized[key] = arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]
        return normalized
    raise RuntimeError("Dense episode recording requires env.step info to be a mapping.")


def _normalize_dense_episode_infos(info, *, num_envs: int):
    if not isinstance(info, dict):
        raise RuntimeError("Dense episode recording requires env.step info to be a mapping.")
    per_env = [dict() for _ in range(int(num_envs))]
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.size == 0:
            normalized_values = [None] * int(num_envs)
        elif arr.ndim == 0:
            scalar = arr.item() if hasattr(arr, "item") else arr
            normalized_values = [scalar] * int(num_envs)
        elif arr.shape[0] in {1, int(num_envs)}:
            rows = arr.reshape(arr.shape[0], -1)
            if rows.shape[0] == 1 and int(num_envs) > 1:
                rows = np.repeat(rows, int(num_envs), axis=0)
            normalized_values = []
            for row in rows:
                scalar = row.reshape(-1)[0]
                normalized_values.append(scalar.item() if hasattr(scalar, "item") else scalar)
        else:
            scalar = arr.reshape(-1)[0]
            normalized_values = [
                scalar.item() if hasattr(scalar, "item") else scalar
            ] * int(num_envs)
        for env_index in range(int(num_envs)):
            per_env[env_index][key] = normalized_values[env_index]
    return per_env


def record_dense_episode_step_if_enabled(env, batched_action, step_result):
    capture = _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    if capture is None:
        return
    num_envs = int(env.unwrapped.num_envs)
    if num_envs != capture.num_envs:
        raise RuntimeError(
            "Dense episode capture num_envs mismatch: "
            f"capture={capture.num_envs} env={num_envs}."
        )
    if not isinstance(step_result, tuple) or len(step_result) != 5:
        raise RuntimeError("Dense episode recording requires env.step to return a 5-tuple.")
    _obs, _reward, _terminated, _truncated, info = step_result
    action_arr = np.asarray(batched_action, dtype=np.float32)
    if action_arr.ndim == 1:
        action_arr = action_arr.reshape(1, -1)
    if action_arr.shape[0] == 1 and num_envs > 1:
        action_arr = np.repeat(action_arr, num_envs, axis=0)
    if action_arr.shape[0] != num_envs:
        raise RuntimeError(
            f"Dense episode recording expected batched action first dimension {num_envs}, got {action_arr.shape}."
        )
    per_env_infos = _normalize_dense_episode_infos(info, num_envs=num_envs)
    frames = capture_base_camera_frames(env)
    if len(frames) != num_envs:
        raise RuntimeError(
            "Dense episode capture expected one base_camera frame per env after step, "
            f"got {len(frames)} frames for num_envs={num_envs}."
        )
    for env_index in range(num_envs):
        if env_index in capture.finalized_env_indices:
            continue
        capture.actions_per_env[env_index].append(action_arr[env_index].copy())
        capture.infos_per_env[env_index].append(per_env_infos[env_index])
        _append_frame_to_dense_capture(capture, env_index=env_index, frame=frames[env_index])


def _write_batched_dense_episode_artifact_for_env(
    capture,
    *,
    env_index: int,
    planner_backend_value,
    macro_route,
    exit_code,
    semantic_task_success,
    failed_stage,
):
    if capture.output_dir is None:
        dense_artifact_path = None
    else:
        capture.output_dir.mkdir(parents=True, exist_ok=True)
        dense_artifact_path = resolve_batched_dense_episode_output_path(
            capture.output_dir,
            env_index=env_index,
        )
    if dense_artifact_path is not None:
        write_dense_episode_artifact(
            artifact_path=dense_artifact_path,
            instruction=capture.instructions_per_env[env_index],
            camera_name=capture.camera_name,
            images=capture.images_per_env[env_index],
            actions=capture.actions_per_env[env_index],
            infos=capture.infos_per_env[env_index],
            result={
                "exit_code": int(exit_code),
                "execution_outcome": "success" if int(exit_code) == 0 else "failed",
                "semantic_task_success": bool(semantic_task_success),
                "failed_stage": failed_stage,
                "step_count": len(capture.actions_per_env[env_index]),
                "frame_count": len(capture.images_per_env[env_index]),
                "env_index": int(env_index),
            },
            source={
                "planner_backend": planner_backend_value,
                "macro_route": macro_route,
                "camera_name": capture.camera_name,
                "env_index": int(env_index),
            },
            image_target_width=capture.image_target_width,
            image_target_height=capture.image_target_height,
        )
    if capture.rl4vla_raw_output_dir is not None:
        capture.rl4vla_raw_output_dir.mkdir(parents=True, exist_ok=True)
        preencoded_images = None
        if capture.rl4vla_preencoded_images_per_env is not None:
            candidate = capture.rl4vla_preencoded_images_per_env[env_index]
            if len(candidate) == len(capture.images_per_env[env_index]):
                preencoded_images = candidate
        raw_artifact_kwargs = {
            "artifact_path": resolve_batched_rl4vla_raw_episode_output_path(
                capture.rl4vla_raw_output_dir,
                env_index=env_index,
            ),
            "instruction": capture.instructions_per_env[env_index],
            "images": capture.images_per_env[env_index],
            "actions": capture.actions_per_env[env_index],
            "infos": capture.infos_per_env[env_index],
            "result": {
                "exit_code": int(exit_code),
                "execution_outcome": "success" if int(exit_code) == 0 else "failed",
                "semantic_task_success": bool(semantic_task_success),
                "failed_stage": failed_stage,
                "step_count": len(capture.actions_per_env[env_index]),
                "frame_count": len(capture.images_per_env[env_index]),
                "env_index": int(env_index),
            },
            "source": {
                "planner_backend": planner_backend_value,
                "macro_route": macro_route,
                "camera_name": capture.camera_name,
                "env_index": int(env_index),
            },
            "image_target_width": capture.image_target_width,
            "image_target_height": capture.image_target_height,
            "preencoded_images": preencoded_images,
            "embedded_runtime_config_yaml": _read_optional_text_file(
                capture.runtime_config_paths_per_env[env_index],
                label=f"runtime_config_paths_per_env[{env_index}]",
            ),
            "embedded_runtime_request_json": _read_optional_text_file(
                capture.runtime_request_paths_per_env[env_index],
                label=f"runtime_request_paths_per_env[{env_index}]",
            ),
        }
        if capture.raw_writer_executor is None:
            write_rl4vla_raw_episode_artifact(**raw_artifact_kwargs)
        else:
            capture.pending_raw_writer_futures.append(
                capture.raw_writer_executor.submit(write_rl4vla_raw_episode_artifact, **raw_artifact_kwargs)
            )


def finalize_batched_dense_episode_env_if_available(
    *,
    env_index: int,
    planner_backend_value,
    macro_route,
    exit_code,
    semantic_task_success,
    failed_stage,
):
    capture = _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    if capture is None or capture.num_envs <= 1:
        return None
    normalized_env_index = int(env_index)
    if normalized_env_index < 0 or normalized_env_index >= int(capture.num_envs):
        raise IndexError(
            f"env_index={normalized_env_index} is outside the active dense capture range 0..{int(capture.num_envs) - 1}"
        )
    if normalized_env_index in capture.finalized_env_indices:
        return (
            None
            if capture.output_dir is None
            else resolve_batched_dense_episode_output_path(
                capture.output_dir,
                env_index=normalized_env_index,
            )
        )
    _write_batched_dense_episode_artifact_for_env(
        capture,
        env_index=normalized_env_index,
        planner_backend_value=planner_backend_value,
        macro_route=macro_route,
        exit_code=exit_code,
        semantic_task_success=semantic_task_success,
        failed_stage=failed_stage,
    )
    capture.images_per_env[normalized_env_index] = []
    if capture.rl4vla_preencoded_images_per_env is not None:
        capture.rl4vla_preencoded_images_per_env[normalized_env_index] = []
    capture.actions_per_env[normalized_env_index] = []
    capture.infos_per_env[normalized_env_index] = []
    capture.finalized_env_indices.add(normalized_env_index)
    return (
        None
        if capture.output_dir is None
        else resolve_batched_dense_episode_output_path(
            capture.output_dir,
            env_index=normalized_env_index,
        )
    )


def _consume_unified_dense_episode_capture():
    global _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    capture = _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE
    _ACTIVE_UNIFIED_DENSE_EPISODE_CAPTURE = None
    return capture


def _resolve_raw_writer_worker_count(num_envs: int) -> int:
    configured = str(os.environ.get("RC5_RL4VLA_RAW_WRITE_WORKERS", "")).strip()
    if configured:
        try:
            resolved = int(configured)
        except ValueError:
            resolved = 0
        if resolved > 0:
            return resolved
    cpu_count = os.cpu_count() or 4
    return max(1, min(int(num_envs), int(cpu_count), 8))


def _wait_for_pending_raw_writes(capture: UnifiedDenseEpisodeCapture) -> None:
    pending = list(capture.pending_raw_writer_futures)
    capture.pending_raw_writer_futures.clear()
    for future in pending:
        future.result()


def _shutdown_capture_raw_writer(capture: UnifiedDenseEpisodeCapture) -> None:
    executor = capture.raw_writer_executor
    if executor is None:
        return
    executor.shutdown(wait=True)
    capture.raw_writer_executor = None


def write_dense_episode_artifact_if_available(
    *,
    planner_backend_value,
    macro_route,
    exit_code,
    macro_feedback,
):
    capture = _consume_unified_dense_episode_capture()
    if capture is None:
        return None

    semantic_task_success = None if macro_feedback is None else bool(macro_feedback.get("semantic_task_success"))
    failed_stage = None if macro_feedback is None else macro_feedback.get("failed_stage")
    if capture.num_envs == 1:
        if capture.output_path is not None:
            dense_output_path = write_dense_episode_artifact(
                artifact_path=capture.output_path,
                instruction=capture.instructions_per_env[0],
                camera_name=capture.camera_name,
                images=capture.images_per_env[0],
                actions=capture.actions_per_env[0],
                infos=capture.infos_per_env[0],
                result={
                    "exit_code": int(exit_code),
                    "execution_outcome": "success" if int(exit_code) == 0 else "failed",
                    "semantic_task_success": semantic_task_success,
                    "failed_stage": failed_stage,
                    "step_count": len(capture.actions_per_env[0]),
                    "frame_count": len(capture.images_per_env[0]),
                },
                source={
                    "planner_backend": planner_backend_value,
                    "macro_route": macro_route,
                    "camera_name": capture.camera_name,
                },
                image_target_width=capture.image_target_width,
                image_target_height=capture.image_target_height,
            )
        else:
            dense_output_path = None
        if capture.rl4vla_raw_output_path is not None:
            preencoded_images = None
            if capture.rl4vla_preencoded_images_per_env is not None:
                candidate = capture.rl4vla_preencoded_images_per_env[0]
                if len(candidate) == len(capture.images_per_env[0]):
                    preencoded_images = candidate
            write_rl4vla_raw_episode_artifact(
                artifact_path=capture.rl4vla_raw_output_path,
                instruction=capture.instructions_per_env[0],
                images=capture.images_per_env[0],
                actions=capture.actions_per_env[0],
                infos=capture.infos_per_env[0],
                result={
                    "exit_code": int(exit_code),
                    "execution_outcome": "success" if int(exit_code) == 0 else "failed",
                    "semantic_task_success": semantic_task_success,
                    "failed_stage": failed_stage,
                    "step_count": len(capture.actions_per_env[0]),
                    "frame_count": len(capture.images_per_env[0]),
                },
                source={
                    "planner_backend": planner_backend_value,
                    "macro_route": macro_route,
                    "camera_name": capture.camera_name,
                },
                image_target_width=capture.image_target_width,
                image_target_height=capture.image_target_height,
                preencoded_images=preencoded_images,
                embedded_runtime_config_yaml=_read_optional_text_file(
                    capture.runtime_config_paths_per_env[0],
                    label="runtime_config_paths_per_env[0]",
                ),
                embedded_runtime_request_json=_read_optional_text_file(
                    capture.runtime_request_paths_per_env[0],
                    label="runtime_request_paths_per_env[0]",
                ),
            )
        return dense_output_path

    if capture.output_dir is None:
        if capture.rl4vla_raw_output_dir is None:
            raise RuntimeError("Batched dense episode capture is missing artifact output directories.")
    else:
        capture.output_dir.mkdir(parents=True, exist_ok=True)
    if capture.rl4vla_raw_output_dir is not None:
        capture.rl4vla_raw_output_dir.mkdir(parents=True, exist_ok=True)
    for env_index in range(capture.num_envs):
        if env_index in capture.finalized_env_indices:
            continue
        _write_batched_dense_episode_artifact_for_env(
            capture,
            env_index=env_index,
            planner_backend_value=planner_backend_value,
            macro_route=macro_route,
            exit_code=exit_code,
            semantic_task_success=semantic_task_success,
            failed_stage=failed_stage,
        )
    _wait_for_pending_raw_writes(capture)
    _shutdown_capture_raw_writer(capture)
    return capture.output_dir
