#!/usr/bin/env python3
"""Render 10 background variants per unique episode and prepare SFT npz immediately.

Existing files in --bg_cache_dir are reused. Newly rendered variants go to a temp
dir and are deleted after prepare so 10k trajectories fit on disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path("/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real/runs/rl/pick_red_cube_sft_databc")
sys.path.insert(0, str(RUN_DIR))

from prepare_sft_v2_episodes import convert_episode  # noqa: E402
from render_bg_variants import render_episode_variants  # noqa: E402


def _unique_files(src_dir: Path) -> list[Path]:
    files = sorted(src_dir.glob("episode_*.npz")) + sorted(src_dir.glob("rl4vla_raw_episode*.npz"))
    return [p for p in files if "_bg" not in p.name]


def _variant_paths(directory: Path, stem: str, n_variants: int) -> list[Path]:
    return [directory / f"{stem}_bg{k:02d}.npz" for k in range(n_variants)]


def _all_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def _prepare_one(src: Path, dest: Path) -> int:
    record = convert_episode(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, arr_0=np.array(record, dtype=object))
    return int(record["action"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("prep_dir", type=Path)
    parser.add_argument("--bg_cache_dir", type=Path, default=None)
    parser.add_argument("--tmp_dir", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_path", type=Path, default=None)
    parser.add_argument("--key", type=str, default="airy_table_scene14sep26_left_image")
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    files = _unique_files(args.src_dir)
    if not files:
        raise FileNotFoundError(f"No unique episode npz in {args.src_dir}")
    if args.stride < 1:
        raise ValueError(f"stride must be >= 1, got {args.stride}")
    if args.offset < 0:
        raise ValueError(f"offset must be >= 0, got {args.offset}")
    end = len(files) if args.end < 0 else min(args.end, len(files))
    files = files[args.start:end][args.offset::args.stride]
    args.prep_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    bg_cache = args.bg_cache_dir

    print(
        f"render_prepare shard start={args.start} end={end} offset={args.offset} "
        f"stride={args.stride} n={len(files)} variants={args.variants} prep={args.prep_dir}",
        flush=True,
    )
    lengths = []
    for idx, src in enumerate(files):
        stem = src.stem
        dests = _variant_paths(args.prep_dir, stem, args.variants)
        if _all_exist(dests):
            print(f"[{idx + 1}/{len(files)}] skip prepared {stem}", flush=True)
            continue
        cache_paths = _variant_paths(bg_cache, stem, args.variants) if bg_cache is not None else []
        if bg_cache is not None and _all_exist(cache_paths):
            print(f"[{idx + 1}/{len(files)}] prepare from cache {stem}", flush=True)
            for src_bg, dest in zip(cache_paths, dests):
                if dest.exists():
                    continue
                lengths.append(_prepare_one(src_bg, dest))
                print(f"  {src_bg.name} -> {dest.name} steps={lengths[-1]}", flush=True)
            continue

        episode_tmp = args.tmp_dir / stem
        if episode_tmp.exists():
            shutil.rmtree(episode_tmp)
        episode_tmp.mkdir(parents=True, exist_ok=True)
        print(f"[{idx + 1}/{len(files)}] render {stem}", flush=True)
        render_episode_variants(
            src,
            episode_tmp,
            n_variants=args.variants,
            seed=args.seed,
            validate=False,
            overwrite=True,
            config_path=args.config_path,
            key=args.key,
            scene=args.scene,
        )
        for dest, src_bg in zip(dests, _variant_paths(episode_tmp, stem, args.variants)):
            if not src_bg.exists():
                raise FileNotFoundError(f"missing rendered variant {src_bg}")
            lengths.append(_prepare_one(src_bg, dest))
            print(f"  {src_bg.name} -> {dest.name} steps={lengths[-1]}", flush=True)
        shutil.rmtree(episode_tmp, ignore_errors=True)

    dest_files = sorted(args.prep_dir.glob("episode_*_bg*.npz"))
    print(json.dumps({"shard_converted": len(lengths), "prep_total": len(dest_files)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
