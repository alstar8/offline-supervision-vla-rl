#!/usr/bin/env python3
"""Aggregate closed-loop SR from the per-worker stats.yaml files of an SR eval.

Each eval worker writes one `vis_0_train/stats.yaml` holding `last_info` for all
of its parallel envs. Success/grasp are read from there rather than from video
filenames, which do not encode the terminal flags reliably.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for stats_path in sorted(root.glob("gpu*/**/vis_0_train/stats.yaml")):
        gpu_part = [p for p in stats_path.parts if p.startswith("gpu")]
        gpu = int(gpu_part[0][3:]) if gpu_part else -1
        stats = yaml.safe_load(stats_path.read_text()) or {}
        last_info = stats.get("last_info") or {}
        for env_id in sorted(last_info, key=lambda k: int(k)):
            info = last_info[env_id] or {}
            rows.append({
                "gpu": gpu,
                "env": int(env_id),
                "success": bool(info.get("success", False)),
                "grasped": bool(info.get("is_src_obj_grasped", False)),
                "consecutive_grasp": bool(info.get("consecutive_grasp", False)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--episode-len", type=int, default=0)
    parser.add_argument("--append-csv", default="", help="Optional CSV to append one row to.")
    parser.add_argument("--step", type=int, default=-1, help="Training step of the checkpoint, for the CSV.")
    args = parser.parse_args()

    root = Path(args.out_root)
    rows = collect_rows(root)
    n = len(rows)
    n_success = sum(r["success"] for r in rows)
    n_grasped = sum(r["grasped"] for r in rows)
    summary = {
        "ckpt": args.ckpt,
        "step": args.step,
        "n_episodes": n,
        "n_success": n_success,
        "n_grasped": n_grasped,
        "success_rate": n_success / n if n else 0.0,
        "grasp_rate": n_grasped / n if n else 0.0,
        "episode_len": args.episode_len,
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.append_csv:
        csv_path = Path(args.append_csv)
        if not csv_path.exists():
            csv_path.write_text("step,n_episodes,n_success,success_rate,n_grasped,grasp_rate,ckpt\n")
        with csv_path.open("a") as fh:
            fh.write(
                f"{args.step},{n},{n_success},{summary['success_rate']:.4f},"
                f"{n_grasped},{summary['grasp_rate']:.4f},{args.ckpt}\n"
            )

    print(json.dumps({k: summary[k] for k in
                      ("ckpt", "step", "n_episodes", "n_success", "success_rate",
                       "n_grasped", "grasp_rate")}, indent=2))


if __name__ == "__main__":
    main()
