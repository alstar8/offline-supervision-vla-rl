#!/usr/bin/env python3
"""Print a compact action summary for an RL4VLA raw episode `.npz`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    load_rl4vla_raw_episode_artifact,
)


CHANNEL_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def load_actions(npz_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = load_rl4vla_raw_episode_artifact(npz_path)
    actions = np.asarray(payload["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"Expected action array with shape [N, 7], got {actions.shape} from {npz_path}"
        )
    return payload, actions


def _first_nonzero(values: np.ndarray, eps: float) -> tuple[int | None, float | None]:
    nonzero = np.where(np.abs(values) > eps)[0]
    if len(nonzero) == 0:
        return None, None
    idx = int(nonzero[0])
    return idx, float(values[idx])


def build_summary_lines(npz_path: Path, payload: dict[str, Any], actions: np.ndarray, *, eps: float) -> list[str]:
    lines = [
        f"[summarize-npz-actions] npz={npz_path}",
        f"[summarize-npz-actions] schema_version={payload.get('schema_version')}",
        f"[summarize-npz-actions] steps={len(actions)}",
        (
            "[summarize-npz-actions] embedded_runtime_bundle: "
            f"yaml={'yes' if isinstance(payload.get('embedded_runtime_config_yaml'), str) and payload.get('embedded_runtime_config_yaml', '').strip() else 'no'} "
            f"json={'yes' if isinstance(payload.get('embedded_runtime_request_json'), str) and payload.get('embedded_runtime_request_json', '').strip() else 'no'}"
        ),
    ]
    for col_idx, name in enumerate(CHANNEL_NAMES):
        values = actions[:, col_idx]
        nonzero = int((np.abs(values) > eps).sum())
        lines.append(
            f"[summarize-npz-actions] {name}: "
            f"min={float(values.min()):+.9f} "
            f"max={float(values.max()):+.9f} "
            f"mean={float(values.mean()):+.9f} "
            f"nonzero={nonzero}"
        )
    dz_idx, dz_value = _first_nonzero(actions[:, 2], eps)
    lines.append(
        "[summarize-npz-actions] first_nonzero_dz: "
        f"step={dz_idx} value={dz_value}"
    )
    gripper_values, gripper_counts = np.unique(actions[:, 6], return_counts=True)
    gripper_summary = ", ".join(
        f"{float(v):+.3f}:{int(c)}" for v, c in zip(gripper_values.tolist(), gripper_counts.tolist())
    )
    lines.append(f"[summarize-npz-actions] gripper_hist={gripper_summary}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print action statistics for an RL4VLA raw episode `.npz`."
    )
    parser.add_argument("--npz_path", required=True, help="Path to rl4vla_raw_episode*.npz")
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Absolute threshold used to decide whether a value is non-zero.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    payload, actions = load_actions(npz_path)
    for line in build_summary_lines(npz_path, payload, actions, eps=args.eps):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
