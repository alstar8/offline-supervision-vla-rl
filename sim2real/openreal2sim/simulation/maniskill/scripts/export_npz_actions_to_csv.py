#!/usr/bin/env python3
"""Export RL4VLA raw episode actions and embedded runtime bundle from `.npz`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (
    load_rl4vla_raw_episode_artifact,
)


CSV_HEADER = ["step", "dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def default_output_csv_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".actions.csv")


def default_output_runtime_config_yaml_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".runtime_config.yaml")


def default_output_runtime_request_json_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".runtime_request.json")


def load_actions(npz_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = load_rl4vla_raw_episode_artifact(npz_path)
    actions = np.asarray(payload["action"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"Expected action array with shape [N, 7], got {actions.shape} from {npz_path}"
        )
    return payload, actions


def write_csv(actions: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for step_idx, row in enumerate(actions):
            writer.writerow([step_idx, *[float(x) for x in row]])


def write_embedded_runtime_bundle(
    payload: dict[str, Any],
    *,
    output_runtime_config_yaml: Path | None,
    output_runtime_request_json: Path | None,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    embedded_runtime_config_yaml = payload.get("embedded_runtime_config_yaml")
    if (
        output_runtime_config_yaml is not None
        and isinstance(embedded_runtime_config_yaml, str)
        and embedded_runtime_config_yaml.strip()
    ):
        output_runtime_config_yaml.parent.mkdir(parents=True, exist_ok=True)
        output_runtime_config_yaml.write_text(embedded_runtime_config_yaml, encoding="utf-8")
        written["runtime_config_yaml"] = output_runtime_config_yaml

    embedded_runtime_request_json = payload.get("embedded_runtime_request_json")
    if (
        output_runtime_request_json is not None
        and isinstance(embedded_runtime_request_json, str)
        and embedded_runtime_request_json.strip()
    ):
        output_runtime_request_json.parent.mkdir(parents=True, exist_ok=True)
        output_runtime_request_json.write_text(embedded_runtime_request_json, encoding="utf-8")
        written["runtime_request_json"] = output_runtime_request_json

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export `action` rows from an RL4VLA raw episode `.npz` to CSV and, "
            "when available, export embedded runtime_config.yaml/runtime_request.json sidecars."
        )
    )
    parser.add_argument("--npz_path", required=True, help="Path to rl4vla_raw_episode*.npz")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV path. Default: <npz_path>.actions.csv",
    )
    parser.add_argument(
        "--output_runtime_config_yaml",
        default=None,
        help=(
            "Optional output YAML path. Default when embedded YAML is present: "
            "<npz_path>.runtime_config.yaml"
        ),
    )
    parser.add_argument(
        "--output_runtime_request_json",
        default=None,
        help=(
            "Optional output JSON path. Default when embedded JSON is present: "
            "<npz_path>.runtime_request.json"
        ),
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv is not None
        else default_output_csv_path(npz_path)
    )
    output_runtime_config_yaml = (
        Path(args.output_runtime_config_yaml).resolve()
        if args.output_runtime_config_yaml is not None
        else default_output_runtime_config_yaml_path(npz_path)
    )
    output_runtime_request_json = (
        Path(args.output_runtime_request_json).resolve()
        if args.output_runtime_request_json is not None
        else default_output_runtime_request_json_path(npz_path)
    )

    payload, actions = load_actions(npz_path)
    write_csv(actions, output_csv)
    written_bundle = write_embedded_runtime_bundle(
        payload,
        output_runtime_config_yaml=output_runtime_config_yaml,
        output_runtime_request_json=output_runtime_request_json,
    )

    print(
        "[export-npz-actions-to-csv] OK",
        f"npz={npz_path}",
        f"steps={len(actions)}",
        f"output_csv={output_csv}",
        f"runtime_config_yaml={written_bundle.get('runtime_config_yaml')}",
        f"runtime_request_json={written_bundle.get('runtime_request_json')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
