#!/usr/bin/env python3
"""Pick the SFT LoRA with the best eval XY token accuracy."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _score(lora_dir: Path) -> float:
    metrics_path = lora_dir / "eval_metrics.json"
    step = 0
    try:
        step = int(lora_dir.name.split("_")[1])
    except (IndexError, ValueError):
        step = 0
    if not metrics_path.exists():
        return 1e-6 * step
    metrics = json.loads(metrics_path.read_text())
    x_acc = metrics.get("eval_action_accuracy/x")
    y_acc = metrics.get("eval_action_accuracy/y")
    if x_acc is None and y_acc is None:
        return 1e-6 * step
    if x_acc is None:
        return float(y_acc)
    if y_acc is None:
        return float(x_acc)
    return 0.5 * (float(x_acc) + float(y_acc))


def main() -> None:
    run_dir = Path(sys.argv[1])
    candidates = [path for path in sorted(run_dir.glob("lora_*")) if (path / "adapter_config.json").exists()]
    if not candidates:
        raise SystemExit(f"no lora_* checkpoints under {run_dir}")
    best = max(candidates, key=_score)
    print(best)


if __name__ == "__main__":
    main()
