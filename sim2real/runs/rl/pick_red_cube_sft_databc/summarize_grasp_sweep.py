#!/usr/bin/env python
"""Rank the grasp-rate sweep output produced by sweep_sft_grasp.sh.

Reports simulator grasp/success rates next to SFT token accuracy, plus how close
the gripper actually got to the cube. Reach depth is the diagnostic that matters
when grasp rate is near zero: a policy that stalls 24 cm away is not a PPO
problem, it is a reach problem.
"""
import argparse
import json
import re
from pathlib import Path

import yaml

STAT_KEYS = [
    "success",
    "is_src_obj_grasped",
    "consecutive_grasp",
    "instant_is_src_obj_grasped",
    "instant_consecutive_grasp",
]
DIST_RE = re.compile(r"'gripper_obj_dist': (-?[0-9.eE+]+)")
HEIGHT_RE = re.compile(r"'obj_height_above_table': (-?[0-9.eE+]+)")


def reach_stats(render_log: Path) -> tuple[float | None, float | None, float | None]:
    """Return (start_dist, min_dist, max_height) from the per-step info dicts."""
    if not render_log.exists():
        return None, None, None
    text = render_log.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
    dists = [float(m.group(1)) for m in DIST_RE.finditer(text)]
    heights = [float(m.group(1)) for m in HEIGHT_RE.finditer(text)]
    start = dists[0] if dists else None
    return start, (min(dists) if dists else None), (max(heights) if heights else None)


LOAD_PATH_RE = re.compile(r"^vla_load_path:\s*(.+?)\s*$", re.MULTILINE)


def checkpoint_path(out_dir: Path) -> str:
    """Pull vla_load_path out of the run config.

    Read line-wise on purpose: the dumped config contains a `!!python/tuple` tag, so
    yaml.safe_load raises and yaml.unsafe_load would construct arbitrary objects.
    """
    config = out_dir / "config.yaml"
    if not config.exists():
        return ""
    match = LOAD_PATH_RE.search(config.read_text())
    if not match:
        return ""
    return match.group(1).strip().strip("'\"")


def token_accuracy(out_dir: Path) -> dict:
    load_path = checkpoint_path(out_dir)
    metrics = Path(str(load_path)) / "eval_metrics.json"
    if not load_path or not metrics.exists():
        return {}
    try:
        return json.loads(metrics.read_text())
    except Exception:
        return {}


def collect(out_root: Path) -> list[dict]:
    rows = []
    for out_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        stats_file = out_dir / "vis_0_train" / "stats.yaml"
        row = {"tag": out_dir.name, "episode_len": None, "ckpt": checkpoint_path(out_dir)}
        if stats_file.exists():
            payload = yaml.safe_load(stats_file.read_text()) or {}
            stats = payload.get("stats", {}) or {}
            row["episode_len"] = payload.get("ep_len")
            for key in STAT_KEYS:
                row[key] = stats.get(key)
        start, min_dist, max_height = reach_stats(out_dir / "render.log")
        row["start_dist"] = start
        row["min_dist"] = min_dist
        row["max_height"] = max_height
        acc = token_accuracy(out_dir)
        row["acc_x"] = acc.get("eval_action_accuracy/x")
        row["acc_y"] = acc.get("eval_action_accuracy/y")
        rows.append(row)
    return rows


def fmt(value, width, spec=".3f", scale=1.0):
    if value is None:
        return f"{'-':>{width}}"
    return f"{value * scale:>{width}{spec}}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="?", default="sweep_grasp")
    ap.add_argument("--print-best", action="store_true",
                    help="print only the best checkpoint path (for scripting)")
    ap.add_argument("--gate", type=float, default=None,
                    help="exit 3 unless the best grasp rate reaches this fraction")
    opts = ap.parse_args()

    out_root = Path(opts.sweep_dir)
    if not out_root.is_dir():
        print(f"no such sweep dir: {out_root}")
        return 1
    rows = collect(out_root)
    if not rows:
        print(f"no sweep results under {out_root}")
        return 1

    # Grasp rate is the launch gate, so rank by it rather than by success.
    rows.sort(key=lambda r: (r.get("is_src_obj_grasped") or -1.0), reverse=True)

    if opts.print_best:
        print(rows[0].get("ckpt", ""))
        return 0
    if opts.gate is not None:
        grasp = rows[0].get("is_src_obj_grasped")
        if grasp is None or grasp < opts.gate:
            shown = "n/a" if grasp is None else f"{grasp * 100:.2f}%"
            print(f"GATE FAIL: best grasp {shown} < {opts.gate * 100:.2f}%")
            return 3
        print(f"GATE PASS: best grasp {grasp * 100:.2f}% >= {opts.gate * 100:.2f}% ({rows[0]['tag']})")
        return 0

    header = (
        f"{'checkpoint':<34}{'ep_len':>7}{'SR %':>8}{'grasp %':>9}{'consec %':>10}"
        f"{'reach m':>9}{'min d m':>9}{'maxh m':>8}{'acc_x':>7}{'acc_y':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['tag'][:33]:<34}"
            f"{(r['episode_len'] if r['episode_len'] is not None else '-'):>7}"
            f"{fmt(r.get('success'), 8, '.2f', 100.0)}"
            f"{fmt(r.get('is_src_obj_grasped'), 9, '.2f', 100.0)}"
            f"{fmt(r.get('consecutive_grasp'), 10, '.2f', 100.0)}"
            f"{fmt(r.get('start_dist'), 9)}"
            f"{fmt(r.get('min_dist'), 9)}"
            f"{fmt(r.get('max_height'), 8)}"
            f"{fmt(r.get('acc_x'), 7)}"
            f"{fmt(r.get('acc_y'), 7)}"
        )

    print()
    best = rows[0]
    grasp = best.get("is_src_obj_grasped")
    if grasp is None:
        print("No grasp rate recorded; check render.log files.")
    elif grasp >= 0.50:
        verdict = "matches the DDP/gripfix regime that reached 45-47% SR. Launch DataBC."
    elif grasp >= 0.15:
        verdict = "above the historical floor for PPO progress, but expect a slow climb."
    else:
        verdict = "below the ~15% floor: every run starting here has stalled. Improve SFT first."
    if grasp is not None:
        print(f"best: {best['tag']} at {grasp * 100:.2f}% grasp -- {verdict}")
        min_dist = best.get("min_dist")
        if min_dist is not None and min_dist > 0.05:
            print(
                f"warning: gripper never got closer than {min_dist:.3f} m "
                "(a grasp needs ~0.02-0.03 m), so this is a reach failure, not an RL one."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
