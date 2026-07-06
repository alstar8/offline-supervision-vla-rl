from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List

import yaml


def _collect_success(entries: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for seed, metrics in entries.items():
        if isinstance(metrics, dict) and "success" in metrics:
            values.append(float(metrics["success"]))
    return values


def _avg(values: List[float]) -> float | None:
    return mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-env success for train/test from stats-*.yaml."
    )
    parser.add_argument(
        "stats_path",
        help="Path to stats-*.yaml (e.g., SimplerEnv/scripts/stats/stats-YYYYMMDD_HHMMSS.yaml)",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    data = yaml.safe_load(stats_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected top-level mapping in stats YAML.")

    for load_path, envs in data.items():
        print(f"\n=== {load_path} ===")
        overall_train: List[float] = []
        overall_test: List[float] = []

        for env_name, splits in envs.items():
            train_vals = _collect_success(splits.get("train", {}))
            test_vals = _collect_success(splits.get("test", {}))

            if train_vals:
                overall_train.extend(train_vals)
            if test_vals:
                overall_test.extend(test_vals)

            train_avg = _avg(train_vals)
            test_avg = _avg(test_vals)

            train_text = f"{train_avg:.4f}" if train_avg is not None else "n/a"
            test_text = f"{test_avg:.4f}" if test_avg is not None else "n/a"
            print(f"{env_name}: train={train_text} test={test_text}")

        if overall_train:
            print(f"OVERALL train mean: {mean(overall_train):.4f}")
        else:
            print("OVERALL train mean: n/a")

        if overall_test:
            print(f"OVERALL test mean: {mean(overall_test):.4f}")
        else:
            print("OVERALL test mean: n/a")


if __name__ == "__main__":
    main()
