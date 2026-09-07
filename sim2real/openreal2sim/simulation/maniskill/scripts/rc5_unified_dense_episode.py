from __future__ import annotations

import os
import zipfile
from pathlib import Path
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


DENSE_EPISODE_ARTIFACT_SCHEMA_VERSION = "rc5_dense_episode_v0"
RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION = "rc5_rl4vla_raw_episode_v1"
RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION_EMBEDDED_RUNTIME_BUNDLE = "rc5_rl4vla_raw_episode_v2"
DEFAULT_RL4VLA_IMAGE_WIDTH = 640
DEFAULT_RL4VLA_IMAGE_HEIGHT = 480
DEFAULT_RL4VLA_IMAGE_ENCODING = "jpeg"


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _write_npz_payload(
    output_path: Path,
    *,
    payload: Mapping[str, Any],
    compressed: bool,
) -> Path:
    np_payload = np.array(dict(payload), dtype=object)
    if compressed:
        np.savez_compressed(output_path, arr_0=np_payload)
    else:
        np.savez(output_path, arr_0=np_payload)
    return output_path


def _require_non_empty_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return str(value)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _require_positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _normalize_optional_text(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string when provided")
    return str(value)


def _detect_npz_compressed(path: Path) -> bool:
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        if not members:
            return False
        return any(item.compress_type != zipfile.ZIP_STORED for item in members)


def _resize_frame_to_canvas(frame: Any, *, target_width: int, target_height: int) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("each image frame must have shape [H, W, 3]")
    height, width = image.shape[:2]
    scale = min(float(target_width) / float(width), float(target_height) / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(image, mode="RGB")
    resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8)


def _encode_frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    pil_image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _encode_frame_to_jpeg_uint8_buffer(frame: np.ndarray) -> np.ndarray:
    return np.frombuffer(_encode_frame_to_jpeg_bytes(frame), dtype=np.uint8).copy()


def decode_dense_episode_images(payload: Mapping[str, Any]) -> np.ndarray:
    image_encoding = _require_non_empty_str(payload.get("image_encoding"), label="image_encoding")
    if image_encoding != DEFAULT_RL4VLA_IMAGE_ENCODING:
        raise ValueError(f"Unsupported dense episode image_encoding: {image_encoding!r}")
    encoded_images = payload.get("image")
    if not isinstance(encoded_images, np.ndarray):
        raise ValueError("dense episode image payload must be a numpy object array")
    decoded_images = []
    for item in encoded_images.tolist():
        if not isinstance(item, (bytes, bytearray)):
            raise ValueError("dense episode image payload items must be JPEG byte strings")
        with Image.open(BytesIO(bytes(item))) as pil_image:
            decoded_images.append(np.asarray(pil_image.convert("RGB"), dtype=np.uint8))
    return np.asarray(decoded_images, dtype=np.uint8)


def decode_rl4vla_raw_episode_images(payload: Mapping[str, Any]) -> np.ndarray:
    encoded_images = payload.get("image")
    if not isinstance(encoded_images, list):
        raise ValueError("rl4vla raw image payload must be a python list")
    decoded_images = []
    for idx, item in enumerate(encoded_images):
        if isinstance(item, np.ndarray):
            if item.dtype != np.uint8 or item.ndim != 1:
                raise ValueError(
                    f"rl4vla raw image payload item {idx} must be a 1D uint8 numpy array"
                )
            image_bytes = item.tobytes()
        elif isinstance(item, (bytes, bytearray)):
            image_bytes = bytes(item)
        else:
            raise ValueError(
                f"rl4vla raw image payload item {idx} must be JPEG bytes or a 1D uint8 numpy array"
            )
        with Image.open(BytesIO(image_bytes)) as pil_image:
            decoded_images.append(np.asarray(pil_image.convert("RGB"), dtype=np.uint8))
    return np.asarray(decoded_images, dtype=np.uint8)


def build_dense_episode_payload(
    *,
    instruction: str,
    images: Sequence[Any],
    actions: Sequence[Any],
    infos: Sequence[Any],
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    camera_name: str = "base_camera",
    image_target_width: int = DEFAULT_RL4VLA_IMAGE_WIDTH,
    image_target_height: int = DEFAULT_RL4VLA_IMAGE_HEIGHT,
) -> dict[str, Any]:
    resolved_instruction = _require_non_empty_str(instruction, label="instruction")
    resolved_camera_name = _require_non_empty_str(camera_name, label="camera_name")
    resolved_result = _require_mapping(result, label="result")
    resolved_source = _require_mapping(source, label="source")
    resolved_image_target_width = _require_positive_int(image_target_width, label="image_target_width")
    resolved_image_target_height = _require_positive_int(image_target_height, label="image_target_height")

    if not images:
        raise ValueError("images must contain at least one frame")
    if len(images) != len(actions) + 1:
        raise ValueError("images must contain exactly one more frame than actions")
    if len(infos) != len(actions):
        raise ValueError("infos must contain exactly as many items as actions")

    original_frame = np.asarray(images[0], dtype=np.uint8)
    if original_frame.ndim != 3 or original_frame.shape[-1] != 3:
        raise ValueError("images must contain RGB frames with shape [H, W, 3]")
    canvas_images = [
        _resize_frame_to_canvas(
            frame,
            target_width=resolved_image_target_width,
            target_height=resolved_image_target_height,
        )
        for frame in images
    ]
    image_jpeg_bytes = np.asarray(
        [_encode_frame_to_jpeg_bytes(frame) for frame in canvas_images],
        dtype=object,
    )

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2:
        raise ValueError("actions must form a dense float32 array with shape [T, D]")

    info_array = np.asarray(list(infos), dtype=object)

    return {
        "schema_version": DENSE_EPISODE_ARTIFACT_SCHEMA_VERSION,
        "instruction": resolved_instruction,
        "camera_name": resolved_camera_name,
        "image_encoding": DEFAULT_RL4VLA_IMAGE_ENCODING,
        "image_target_width": resolved_image_target_width,
        "image_target_height": resolved_image_target_height,
        "image_original_hw": [int(original_frame.shape[0]), int(original_frame.shape[1])],
        "image_stored_hw": [resolved_image_target_height, resolved_image_target_width],
        "image": image_jpeg_bytes,
        "action": action_array,
        "info": info_array,
        "result": resolved_result,
        "source": resolved_source,
    }


def write_dense_episode_artifact(
    *,
    artifact_path: str | Path,
    instruction: str,
    images: Sequence[Any],
    actions: Sequence[Any],
    infos: Sequence[Any],
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    camera_name: str = "base_camera",
    image_target_width: int = DEFAULT_RL4VLA_IMAGE_WIDTH,
    image_target_height: int = DEFAULT_RL4VLA_IMAGE_HEIGHT,
) -> Path:
    payload = build_dense_episode_payload(
        instruction=instruction,
        images=images,
        actions=actions,
        infos=infos,
        result=result,
        source=source,
        camera_name=camera_name,
        image_target_width=image_target_width,
        image_target_height=image_target_height,
    )
    output_path = Path(artifact_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_npz_payload(output_path, payload=payload, compressed=True)


def build_rl4vla_raw_episode_payload(
    *,
    instruction: str,
    images: Sequence[Any],
    actions: Sequence[Any],
    infos: Sequence[Any],
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    image_target_width: int = DEFAULT_RL4VLA_IMAGE_WIDTH,
    image_target_height: int = DEFAULT_RL4VLA_IMAGE_HEIGHT,
    preencoded_images: Sequence[Any] | None = None,
    embedded_runtime_config_yaml: str | None = None,
    embedded_runtime_request_json: str | None = None,
) -> dict[str, Any]:
    resolved_instruction = _require_non_empty_str(instruction, label="instruction")
    resolved_result = _require_mapping(result, label="result")
    resolved_source = _require_mapping(source, label="source")
    resolved_image_target_width = _require_positive_int(image_target_width, label="image_target_width")
    resolved_image_target_height = _require_positive_int(image_target_height, label="image_target_height")
    resolved_embedded_runtime_config_yaml = _normalize_optional_text(
        embedded_runtime_config_yaml,
        label="embedded_runtime_config_yaml",
    )
    resolved_embedded_runtime_request_json = _normalize_optional_text(
        embedded_runtime_request_json,
        label="embedded_runtime_request_json",
    )

    if not images:
        raise ValueError("images must contain at least one frame")
    if len(images) != len(actions) + 1:
        raise ValueError("images must contain exactly one more frame than actions")
    if len(infos) != len(actions):
        raise ValueError("infos must contain exactly as many items as actions")

    if preencoded_images is None:
        step_aligned_images = [
            _encode_frame_to_jpeg_uint8_buffer(
                _resize_frame_to_canvas(
                    frame,
                    target_width=resolved_image_target_width,
                    target_height=resolved_image_target_height,
                )
            )
            for frame in images[:-1]
        ]
    else:
        if len(preencoded_images) != len(images):
            raise ValueError("preencoded_images must contain exactly as many items as images")
        step_aligned_images = []
        for idx, item in enumerate(preencoded_images[:-1]):
            if isinstance(item, np.ndarray):
                encoded = np.asarray(item, dtype=np.uint8)
                if encoded.ndim != 1:
                    raise ValueError(
                        f"preencoded_images[{idx}] must be a 1D uint8 numpy array"
                    )
                step_aligned_images.append(encoded.copy())
            elif isinstance(item, (bytes, bytearray)):
                step_aligned_images.append(np.frombuffer(bytes(item), dtype=np.uint8).copy())
            else:
                raise ValueError(
                    f"preencoded_images[{idx}] must be JPEG bytes or a 1D uint8 numpy array"
                )

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[1] != 7:
        raise ValueError("actions must form a dense float32 array with shape [T, 7]")

    normalized_infos = []
    for idx, item in enumerate(infos):
        normalized_info = _require_mapping(item, label=f"info[{idx}]")
        normalized_infos.append(normalized_info)

    payload = {
        "schema_version": (
            RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION_EMBEDDED_RUNTIME_BUNDLE
            if (
                resolved_embedded_runtime_config_yaml is not None
                or resolved_embedded_runtime_request_json is not None
            )
            else RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION
        ),
        "instruction": resolved_instruction,
        "image": step_aligned_images,
        "action": action_array,
        "info": normalized_infos,
        "result": resolved_result,
        "source": resolved_source,
    }
    if resolved_embedded_runtime_config_yaml is not None:
        payload["embedded_runtime_config_yaml"] = resolved_embedded_runtime_config_yaml
    if resolved_embedded_runtime_request_json is not None:
        payload["embedded_runtime_request_json"] = resolved_embedded_runtime_request_json
    return payload


def write_rl4vla_raw_episode_artifact(
    *,
    artifact_path: str | Path,
    instruction: str,
    images: Sequence[Any],
    actions: Sequence[Any],
    infos: Sequence[Any],
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    image_target_width: int = DEFAULT_RL4VLA_IMAGE_WIDTH,
    image_target_height: int = DEFAULT_RL4VLA_IMAGE_HEIGHT,
    compress: bool | None = None,
    preencoded_images: Sequence[Any] | None = None,
    embedded_runtime_config_yaml: str | None = None,
    embedded_runtime_request_json: str | None = None,
) -> Path:
    payload = build_rl4vla_raw_episode_payload(
        instruction=instruction,
        images=images,
        actions=actions,
        infos=infos,
        result=result,
        source=source,
        image_target_width=image_target_width,
        image_target_height=image_target_height,
        preencoded_images=preencoded_images,
        embedded_runtime_config_yaml=embedded_runtime_config_yaml,
        embedded_runtime_request_json=embedded_runtime_request_json,
    )
    output_path = Path(artifact_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_compress = (
        _env_flag_enabled("RC5_RL4VLA_RAW_COMPRESS", default=False)
        if compress is None
        else bool(compress)
    )
    return _write_npz_payload(output_path, payload=payload, compressed=resolved_compress)


def load_rl4vla_raw_episode_artifact(artifact_path: str | Path) -> dict[str, Any]:
    input_path = Path(artifact_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"RL4VLA raw episode artifact does not exist: {input_path}")
    payload = np.load(input_path, allow_pickle=True)["arr_0"].tolist()
    if not isinstance(payload, dict):
        raise ValueError(f"RL4VLA raw episode artifact must contain a mapping payload: {input_path}")
    return payload


def rewrite_rl4vla_raw_episode_embedded_runtime_request_json(
    artifact_path: str | Path,
    *,
    runtime_request_json: str,
) -> Path:
    output_path = Path(artifact_path).expanduser().resolve()
    payload = load_rl4vla_raw_episode_artifact(output_path)
    if payload.get("schema_version") != RL4VLA_RAW_EPISODE_ARTIFACT_SCHEMA_VERSION_EMBEDDED_RUNTIME_BUNDLE:
        return output_path
    payload["embedded_runtime_request_json"] = _normalize_optional_text(
        runtime_request_json,
        label="runtime_request_json",
    )
    return _write_npz_payload(
        output_path,
        payload=payload,
        compressed=_detect_npz_compressed(output_path),
    )


def load_dense_episode_artifact(artifact_path: str | Path) -> dict[str, Any]:
    input_path = Path(artifact_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dense episode artifact does not exist: {input_path}")
    payload = np.load(input_path, allow_pickle=True)["arr_0"].tolist()
    if not isinstance(payload, dict):
        raise ValueError(f"Dense episode artifact must contain a mapping payload: {input_path}")
    return payload
