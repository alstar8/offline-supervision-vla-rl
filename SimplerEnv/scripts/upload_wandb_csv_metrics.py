#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import wandb


STEP_CANDIDATES = ("Step", "_step", "step", "global_step")
SUCCESS_CANDIDATES = ("eval/success", "eval.success", "success")
OOD_SUCCESS_CANDIDATES = (
    "eval/ood_success",
    "eval/success_ood",
    "eval.ood_success",
    "eval.success_ood",
    "ood_success",
    "success_ood",
)


def _find_column(fieldnames: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    names = list(fieldnames)
    for c in candidates:
        if c in names:
            return c
    lowered = {n.lower(): n for n in names}
    for c in candidates:
        k = c.lower()
        if k in lowered:
            return lowered[k]

    # Support W&B CSV export headers that prefix metrics, e.g.
    # "onbc_eval - eval/success" or "myrun/eval/success".
    # Also ignore aggregate helper columns like "__MIN"/"__MAX".
    valid_names = [
        n for n in names
        if "__MIN" not in n.upper() and "__MAX" not in n.upper()
    ]
    lowered_valid = {n.lower(): n for n in valid_names}
    for c in candidates:
        ck = c.lower()
        for lk, orig in lowered_valid.items():
            if lk.endswith(ck) or f" - {ck}" in lk or f"/{ck}" in lk:
                return orig
    return None


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if math.isnan(x):
        return None
    return x


def _to_step(v: str | None) -> int | None:
    x = _to_float(v)
    if x is None:
        return None
    return int(round(x))


def _read_csv_points(csv_path: Path) -> dict[int, dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    points: dict[int, dict[str, float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header in {csv_path}")

        step_col = _find_column(reader.fieldnames, STEP_CANDIDATES)
        success_col = _find_column(reader.fieldnames, SUCCESS_CANDIDATES)
        ood_col = _find_column(reader.fieldnames, OOD_SUCCESS_CANDIDATES)

        if step_col is None:
            raise ValueError(
                f"Could not find step column in {csv_path}. Tried: {STEP_CANDIDATES}"
            )
        if success_col is None and ood_col is None:
            raise ValueError(
                f"Could not find eval metric columns in {csv_path}. "
                f"Tried success={SUCCESS_CANDIDATES}, ood={OOD_SUCCESS_CANDIDATES}"
            )

        for row in reader:
            step = _to_step(row.get(step_col))
            if step is None:
                continue
            payload: dict[str, float] = {}
            success = _to_float(row.get(success_col)) if success_col is not None else None
            ood = _to_float(row.get(ood_col)) if ood_col is not None else None
            if success is not None:
                payload["eval/success"] = success
            if ood is not None:
                payload["eval/success_ood"] = ood
            if not payload:
                continue
            points.setdefault(step, {}).update(payload)

    return points


def upload_csv(
    csv_path: Path,
    project: str,
    entity: str | None,
    mode: str,
    run_name_prefix: str,
) -> None:
    run_name = f"{run_name_prefix}-{csv_path.stem}"
    points = _read_csv_points(csv_path)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        config={
            "source_csv": str(csv_path),
        },
    )

    logged = 0
    for step in sorted(points.keys()):
        wandb.log(points[step], step=step)
        logged += 1

    print(f"[upload] {csv_path.name}: logged_rows={logged} run_url={run.url}")
    wandb.finish()


def upload_csvs_as_single_run(
    csv_paths: list[Path],
    project: str,
    entity: str | None,
    mode: str,
    run_name: str,
) -> None:
    merged: dict[int, dict[str, float]] = {}
    for p in csv_paths:
        points = _read_csv_points(p)
        for step, payload in points.items():
            merged.setdefault(step, {}).update(payload)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        config={"source_csvs": [str(p) for p in csv_paths]},
    )

    logged = 0
    for step in sorted(merged.keys()):
        wandb.log(merged[step], step=step)
        logged += 1
    print(f"[upload-single] csv_count={len(csv_paths)} logged_steps={logged} run_url={run.url}")
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload one or more W&B-exported CSV files back into W&B runs, "
            "mapping metrics to eval/success and eval/ood_success."
        )
    )
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to a CSV file. Pass this flag multiple times for multiple files.",
    )
    parser.add_argument("--project", help="Target W&B project", default="RLVLA")
    parser.add_argument("--entity", default=None, help="Target W&B entity (optional)")
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--run-name-prefix", default="csv-upload")
    parser.add_argument(
        "--single-run-name",
        default=None,
        help="If set, merge all --csv files into one W&B run with this run name.",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    if args.single_run_name:
        upload_csvs_as_single_run(
            csv_paths=csv_paths,
            project=args.project,
            entity=args.entity,
            mode=args.mode,
            run_name=args.single_run_name,
        )
    else:
        for p in csv_paths:
            upload_csv(
                csv_path=p,
                project=args.project,
                entity=args.entity,
                mode=args.mode,
                run_name_prefix=args.run_name_prefix,
            )


if __name__ == "__main__":
    main()
