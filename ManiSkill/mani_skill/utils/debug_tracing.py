from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def trace_enabled() -> bool:
    return os.getenv("MANISKILL_TRACE_ACTIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def trace_prefix() -> str:
    prefix = os.getenv("MANISKILL_TRACE_PREFIX", "").strip()
    if prefix:
        return prefix
    return "MANISKILL_TRACE"


def trace_max_elems() -> int:
    raw = os.getenv("MANISKILL_TRACE_MAX_ELEMS", "").strip()
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, value)


def trace_tags() -> set[str] | None:
    raw = os.getenv("MANISKILL_TRACE_TAGS", "").strip()
    if not raw:
        return None
    tags = {tag.strip() for tag in raw.split(",") if tag.strip()}
    return tags or None


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return value


def summarize(value: Any, *, max_elems: int | None = None) -> str:
    if max_elems is None:
        max_elems = trace_max_elems()
    arr = _to_numpy(value)
    if arr.ndim > 1:
        arr = arr[0]
    flat = arr.reshape(-1)
    clipped = flat[:max_elems]
    summary = np.array2string(
        clipped,
        precision=5,
        suppress_small=False,
        separator=", ",
    )
    if flat.size > max_elems:
        summary = f"{summary} ... (len={flat.size})"
    return summary


def trace(tag: str, **fields: Any) -> None:
    if not trace_enabled():
        return
    allowed_tags = trace_tags()
    if allowed_tags is not None and tag not in allowed_tags:
        return
    parts = [f"{key}={summarize(value)}" for key, value in fields.items()]
    message = " | ".join(parts)
    print(f"[{trace_prefix()}] {tag}: {message}")
