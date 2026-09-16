#!/usr/bin/env python
"""Re-chunk prepared stride-4 episodes into stride-8 without re-rendering.

Why this is exact rather than an approximation: chunk_fixed_stride builds a label by
*summing* consecutive raw xyz deltas, takes the observation from the chunk's FIRST frame
and `info` from its LAST, and breaks a chunk on any gripper-level change. Merging two
adjacent stride-4 chunks therefore reproduces what a stride-8 pass over the raw episode
would have produced: sum the two xyz sums, keep the earlier frame's observation, keep the
later chunk's info, and refuse to merge across a gripper change.

The one lossy case is a chunk that stride-4 dropped for translating under
MIN_CHUNK_TRANSLATION. Merging across that gap omits under 1 mm of motion, which is below
the action tokenizer's ~0.15 mm resolution on x.

The clamp matters here in a way it did not before: stride-4 sums cap out near 0.024 m so
MAX_TRANSLATION_NORM=0.04 was inert, while stride-8 sums reach ~0.045 m and do get
clamped. That is the point of the experiment -- roughly 1.7x the per-step reach speed.

usage: rechunk_stride8.py <src_prepared_dir> <dest_dir> [--workers N] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MAX_TRANSLATION_NORM = 0.04
MIN_CHUNK_TRANSLATION = float(os.environ.get("RLVLA_MIN_CHUNK_TRANSLATION", "0.001"))
TRANSLATION_EPS = 1e-6
TARGET_STRIDE_MULTIPLE = 2  # merge pairs: stride 4 -> stride 8


def merge_pairs(record: dict) -> tuple[dict, dict]:
    actions = np.asarray(record["action"], dtype=np.float32)
    scene = record["image"]
    wrist = record["image_wrist"]
    proprio = np.asarray(record["proprio"], dtype=np.float32)
    infos = list(record["info"])
    n = int(actions.shape[0])

    out_actions: list[np.ndarray] = []
    out_scene: list = []
    out_wrist: list = []
    out_proprio: list = []
    out_infos: list = []
    n_merged = n_single = n_dropped = n_clamped = 0
    prev_gripper: float | None = None

    idx = 0
    while idx < n:
        chunk = actions[idx].copy()
        last = idx
        # Only merge within a constant gripper level, mirroring the stride-4 break rule.
        for _ in range(TARGET_STRIDE_MULTIPLE - 1):
            nxt = last + 1
            if nxt >= n or abs(float(actions[nxt][6]) - float(chunk[6])) > TRANSLATION_EPS:
                break
            chunk[:3] = chunk[:3] + actions[nxt][:3]
            last = nxt
        if last > idx:
            n_merged += 1
        else:
            n_single += 1

        norm = float(np.linalg.norm(chunk[:3]))
        if norm > MAX_TRANSLATION_NORM:
            chunk[:3] *= MAX_TRANSLATION_NORM / norm
            norm = MAX_TRANSLATION_NORM
            n_clamped += 1

        gripper = float(chunk[6])
        gripper_event = prev_gripper is None or abs(gripper - prev_gripper) > TRANSLATION_EPS
        if norm < MIN_CHUNK_TRANSLATION and not gripper_event:
            n_dropped += 1
            idx = last + 1
            continue

        out_actions.append(chunk)
        out_scene.append(scene[idx])
        out_wrist.append(wrist[idx])
        out_proprio.append(proprio[idx])
        out_infos.append(infos[last])
        prev_gripper = gripper
        idx = last + 1

    merged = {
        "instruction": record["instruction"],
        "action": np.stack(out_actions, axis=0).astype(np.float32),
        "image": out_scene,
        "image_wrist": out_wrist,
        "proprio": np.stack(out_proprio, axis=0).astype(np.float32),
        "info": out_infos,
    }
    counts = {
        "steps_in": n,
        "steps_out": len(out_actions),
        "merged": n_merged,
        "single": n_single,
        "dropped": n_dropped,
        "clamped": n_clamped,
    }
    return merged, counts


def process_one(args: tuple[str, str, bool]) -> tuple[str, int, dict | None, str | None]:
    src_s, dest_s, overwrite = args
    src, dest = Path(src_s), Path(dest_s)
    try:
        if dest.exists() and not overwrite:
            with np.load(dest, allow_pickle=True) as d:
                steps = int(d["arr_0"].item()["action"].shape[0])
            return src.name, steps, None, "skip"
        with np.load(src, allow_pickle=True) as d:
            record = d["arr_0"].item()
        merged, counts = merge_pairs(record)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, arr_0=np.array(merged, dtype=object))
        os.replace(tmp, dest)
        return src.name, counts["steps_out"], counts, None
    except Exception as exc:  # noqa: BLE001
        return src.name, 0, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", type=Path)
    ap.add_argument("dest_dir", type=Path)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N episodes")
    ap.add_argument("--overwrite", action="store_true")
    opts = ap.parse_args()

    files = sorted(opts.src_dir.glob("episode_*_bg*.npz"))
    if opts.limit:
        files = files[: opts.limit]
    if not files:
        print(f"no prepared episodes in {opts.src_dir}")
        return 1
    opts.dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"re-chunking {len(files)} episodes with {opts.workers} workers", flush=True)

    lengths: list[int] = []
    totals = {"steps_in": 0, "steps_out": 0, "merged": 0, "single": 0, "dropped": 0, "clamped": 0}
    failures: list[str] = []
    done = 0
    payload = [(str(f), str(opts.dest_dir / f.name), opts.overwrite) for f in files]
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        futures = [pool.submit(process_one, p) for p in payload]
        for fut in as_completed(futures):
            name, steps, counts, err = fut.result()
            done += 1
            if err and err != "skip":
                failures.append(f"{name}: {err}")
            else:
                lengths.append(steps)
                if counts:
                    for k in totals:
                        totals[k] += counts[k]
            if done % 200 == 0 or done == len(files):
                print(f"  [{done}/{len(files)}] last={name} steps={steps}", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:10]:
            print(f"  {f}")
        return 1

    arr = np.asarray(lengths, dtype=np.int32)
    stats = {
        "n_episodes": int(arr.size),
        "steps_min": int(arr.min()),
        "steps_median": float(np.median(arr)),
        "steps_mean": float(np.mean(arr)),
        "steps_p95": float(np.percentile(arr, 95)),
        "steps_p99": float(np.percentile(arr, 99)),
        "steps_max": int(arr.max()),
        "recommended_episode_len": int(
            min(200, max(80, int(np.ceil(np.percentile(arr, 99) / 8.0) * 8) + 16))
        ),
        "src_dir": str(opts.src_dir),
        "dest_dir": str(opts.dest_dir),
        "max_steps_per_chunk": 8,
        "min_chunk_translation": MIN_CHUNK_TRANSLATION,
        "chunking": "fixed_stride_merged_from_stride4",
        "merge_counts": totals,
    }
    (opts.dest_dir / "sft_episode_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
