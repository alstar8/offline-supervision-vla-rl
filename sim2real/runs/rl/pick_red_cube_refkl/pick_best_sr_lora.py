#!/usr/bin/env python3
"""Pick the SFT LoRA with the best closed-loop success rate.

Supersedes pick_sft_lora.py, which ranked by teacher-forced x/y token accuracy.
On the v6 run that metric is useless as a selector: acc_core sat at 0.71-0.72
from step 4000 to 8000 while measured SR moved between 0% and 19%, so it cannot
distinguish the checkpoints it is being asked to rank.

Ties are broken by grasp rate, then by the *earlier* step: SR peaked mid-run and
fell off afterwards, so when two checkpoints score the same the less-trained one
is the safer RL init.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict] = []
    for row in rows:
        if not row.get("ckpt"):
            continue
        out.append({
            "step": int(row["step"]),
            "n_episodes": int(row["n_episodes"]),
            "success_rate": float(row["success_rate"]),
            "grasp_rate": float(row["grasp_rate"]),
            "ckpt": row["ckpt"],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="sr_vs_steps CSV written by summarize_sr.py")
    parser.add_argument("--min-episodes", type=int, default=0,
                        help="Ignore rows scored on fewer episodes than this.")
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()

    rows = [r for r in read_rows(Path(args.csv)) if r["n_episodes"] >= args.min_episodes]
    if not rows:
        raise SystemExit(f"no rows with >= {args.min_episodes} episodes in {args.csv}")

    # Later rows overwrite earlier ones for the same step (a re-score wins).
    by_step: dict[int, dict] = {}
    for row in rows:
        by_step[row["step"]] = row

    best = min(by_step.values(), key=lambda r: (-r["success_rate"], -r["grasp_rate"], r["step"]))

    if args.print_table:
        # stderr, so callers can capture the chosen path from stdout.
        for row in sorted(by_step.values(), key=lambda r: r["step"]):
            mark = " <- best" if row["step"] == best["step"] else ""
            print(f"  step {row['step']:>5}  n={row['n_episodes']:>3}  "
                  f"SR={row['success_rate']:.3f}  grasp={row['grasp_rate']:.3f}{mark}",
                  file=sys.stderr)

    print(best["ckpt"])


if __name__ == "__main__":
    main()
