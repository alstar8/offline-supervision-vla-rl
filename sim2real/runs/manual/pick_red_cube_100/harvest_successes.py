#!/usr/bin/env python3
"""Copy successful both-camera RL4VLA episodes into a flat folder."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace-SR008.nfs2/users/staroverov/B1K/offline-supervision-vla-rl/sim2real")
sys.path.insert(0, str(REPO / "openreal2sim/simulation/maniskill"))
sys.path.insert(0, str(REPO))

from openreal2sim.simulation.maniskill.scripts.rc5_unified_dense_episode import (  # noqa: E402
    load_rl4vla_raw_episode_artifact,
)


def _episode_dir_from_request(request_path: str | None) -> Path | None:
    if not request_path:
        return None
    path = Path(request_path)
    return path.parent if path.parent.name.startswith("episode_") else None


def _find_raw_npz(episode_dir: Path) -> Path | None:
    preferred = [
        episode_dir / "rl4vla_raw_episode_success.npz",
        episode_dir / "rl4vla_raw_episode.npz",
    ]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(episode_dir.glob("rl4vla_raw_episode*.npz"))
    success = [path for path in matches if "success" in path.name]
    if success:
        return success[0]
    return matches[0] if matches else None


def _is_valid_both_camera_success(npz_path: Path) -> tuple[bool, str]:
    try:
        payload = load_rl4vla_raw_episode_artifact(npz_path)
    except Exception as exc:
        return False, f"load_failed:{exc}"
    result = payload.get("result") or {}
    if result.get("semantic_task_success") is not True:
        return False, "not_semantic_success"
    image = payload.get("image")
    wrist = payload.get("image_wrist")
    action = np.asarray(payload.get("action"))
    if not isinstance(image, list) or not isinstance(wrist, list):
        return False, "missing_camera_images"
    if action.ndim != 2 or len(image) != int(action.shape[0]) or len(wrist) != int(action.shape[0]):
        return False, (
            f"length_mismatch image={len(image) if isinstance(image, list) else None} "
            f"wrist={len(wrist) if isinstance(wrist, list) else None} "
            f"action={action.shape}"
        )
    return True, "ok"


def harvest_round(*, round_dir: Path, dest_dir: Path, target: int) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("episode_*.npz"))
    count = len(existing)
    if count >= target:
        return count

    summary_path = round_dir / "execution_summary.json"
    if not summary_path.exists():
        print(f"[harvest] missing {summary_path}", flush=True)
        return count
    summary = json.loads(summary_path.read_text())
    copied = 0
    skipped = 0
    for item in summary.get("episodes") or []:
        if count >= target:
            break
        if item.get("semantic_task_success") is not True or item.get("success") is not True:
            skipped += 1
            continue
        episode_dir = _episode_dir_from_request(item.get("request_path"))
        if episode_dir is None:
            skipped += 1
            continue
        npz_path = _find_raw_npz(episode_dir)
        if npz_path is None:
            skipped += 1
            continue
        ok, reason = _is_valid_both_camera_success(npz_path)
        if not ok:
            print(f"[harvest] skip {npz_path.name}: {reason}", flush=True)
            skipped += 1
            continue
        dest = dest_dir / f"episode_{count:06d}.npz"
        shutil.copy2(npz_path, dest)
        count += 1
        copied += 1
        print(f"[harvest] {npz_path} -> {dest.name}", flush=True)
    print(
        f"[harvest] copied={copied} skipped={skipped} total={count}/{target}",
        flush=True,
    )
    manifest = {
        "count": count,
        "target": target,
        "dest_dir": str(dest_dir),
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return count


def main() -> None:
    round_dir = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    target = int(sys.argv[3])
    count = harvest_round(round_dir=round_dir, dest_dir=dest_dir, target=target)
    print(f"HAVE {count}", flush=True)


if __name__ == "__main__":
    main()
