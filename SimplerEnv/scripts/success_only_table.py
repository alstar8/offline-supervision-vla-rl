from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List

import yaml
from calc_and_aggregate_stats import build_stats


MAIN_ENV_ID = "PutOnPlateInScene25Main-v3"


@dataclass(frozen=True)
class RowSpec:
    label: str
    env_id: str
    split: str  # "train" or "test"


VISION_ROWS = [
    RowSpec("Table", "PutOnPlateInScene25VisionImage-v1", "test"),
    RowSpec("Texture-w", "PutOnPlateInScene25VisionTexture03-v1", "test"),
    RowSpec("Texture-s", "PutOnPlateInScene25VisionTexture05-v1", "test"),
    RowSpec("Noise-w", "PutOnPlateInScene25VisionWhole03-v1", "test"),
    RowSpec("Noise-s", "PutOnPlateInScene25VisionWhole05-v1", "test"),
]

LANG_ROWS = [
    RowSpec("Obj.", "PutOnPlateInScene25Carrot-v1", "test"),
    RowSpec("Recep.", "PutOnPlateInScene25Plate-v1", "test"),
    RowSpec("Instruct", "PutOnPlateInScene25Instruct-v1", "test"),
    RowSpec("M-Obj. (IND)", "PutOnPlateInScene25MultiCarrot-v1", "train"),
    RowSpec("M-Obj. (OOD)", "PutOnPlateInScene25MultiCarrot-v1", "test"),
    RowSpec("Distrb Recep.", "PutOnPlateInScene25MultiPlate-v1", "train"),
    RowSpec("M-Recep.", "PutOnPlateInScene25MultiPlate-v1", "test"),
]

ACTION_ROWS = [
    RowSpec("Obj. Pos.", "PutOnPlateInScene25Position-v1", "test"),
    RowSpec("Robot Pose", "PutOnPlateInScene25EEPose-v1", "test"),
    RowSpec("Obj. Rep.", "PutOnPlateInScene25PositionChangeTo-v1", "test"),
]

ALL_ROWS = VISION_ROWS + LANG_ROWS + ACTION_ROWS
RUN_ID_RE = re.compile(r"-([a-z0-9]{8})/glob/steps_", re.IGNORECASE)


def _collect_success(entries: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for _, metrics in entries.items():
        if isinstance(metrics, dict) and "success" in metrics:
            values.append(float(metrics["success"]))
    return values


def _row_values(env_stats: Dict[str, Any], spec: RowSpec) -> List[float]:
    if spec.env_id not in env_stats:
        return []
    split_stats = env_stats[spec.env_id]
    if not isinstance(split_stats, dict):
        return []
    return _collect_success(split_stats.get(spec.split, {}))


def _row_values_many(env_stats_list: List[Dict[str, Any]], spec: RowSpec) -> List[float]:
    out: List[float] = []
    for env_stats in env_stats_list:
        out.extend(_row_values(env_stats, spec))
    return out


def _ind_values_many(env_stats_list: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for env_stats in env_stats_list:
        out.extend(_collect_success(env_stats.get(MAIN_ENV_ID, {}).get("train", {})))
    return out


def _mean_std(values: List[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return mean(values), None
    return mean(values), stdev(values)


def _pooled_over_tasks(task_values: Iterable[List[float]]) -> tuple[float | None, float | None]:
    valid = [vals for vals in task_values if vals]
    if not valid:
        return None, None

    task_means = [mean(vals) for vals in valid]
    overall_mean = mean(task_means)

    n_samples = sum(len(vals) for vals in valid)
    dof = n_samples - len(valid)
    if dof <= 0:
        return overall_mean, None

    sq_dev_sum = 0.0
    for vals, mu in zip(valid, task_means):
        for v in vals:
            sq_dev_sum += (v - mu) ** 2
    pooled_std = sqrt(sq_dev_sum / dof)
    return overall_mean, pooled_std


def _fmt(mu: float | None, sigma: float | None) -> str:
    if mu is None:
        return "n/a"
    if sigma is None:
        return f"{mu:.3f}"
    return f"{mu:.3f} ({sigma:.3f})"


def _load_labels_config(root: Path) -> Dict[str, Dict[str, str]]:
    cfg_path = root / "scripts" / "stats" / "run-labels.yaml"
    if not cfg_path.exists():
        return {
            "exact": {},
            "prefix": {},
            "run_id": {},
            "group_exact": {},
            "group_prefix": {},
            "group_run_id": {},
        }
    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict):
        return {
            "exact": {},
            "prefix": {},
            "run_id": {},
            "group_exact": {},
            "group_prefix": {},
            "group_run_id": {},
        }
    out = {
        "exact": {},
        "prefix": {},
        "run_id": {},
        "group_exact": {},
        "group_prefix": {},
        "group_run_id": {},
    }
    for key in out.keys():
        val = raw.get(key, {})
        if isinstance(val, dict):
            out[key] = {str(k): str(v) for k, v in val.items()}
    return out


def _extract_run_id(load_path: str) -> str | None:
    m = RUN_ID_RE.search(load_path)
    if m is None:
        return None
    return m.group(1)


def _resolve_from_maps(
    load_path: str,
    run_id: str | None,
    exact_map: Dict[str, str],
    prefix_map: Dict[str, str],
    run_id_map: Dict[str, str],
) -> str | None:
    exact = exact_map
    if load_path in exact:
        return exact[load_path]

    prefix = prefix_map
    best_key = None
    for k in prefix.keys():
        if load_path.startswith(k) and (best_key is None or len(k) > len(best_key)):
            best_key = k
    if best_key is not None:
        return prefix[best_key]

    if run_id is not None:
        if run_id in run_id_map:
            return run_id_map[run_id]

    return None


def _resolve_label(load_path: str, labels_cfg: Dict[str, Dict[str, str]]) -> str | None:
    run_id = _extract_run_id(load_path)
    return _resolve_from_maps(
        load_path,
        run_id,
        labels_cfg.get("exact", {}),
        labels_cfg.get("prefix", {}),
        labels_cfg.get("run_id", {}),
    )


def _resolve_group(load_path: str, labels_cfg: Dict[str, Dict[str, str]]) -> str | None:
    run_id = _extract_run_id(load_path)
    return _resolve_from_maps(
        load_path,
        run_id,
        labels_cfg.get("group_exact", {}),
        labels_cfg.get("group_prefix", {}),
        labels_cfg.get("group_run_id", {}),
    )


def _build_table_lines(
    title: str,
    env_stats_list: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    lines.append(f"\n=== {title} ===")
    lines.append("| Row | Success |")
    lines.append("| --- | --- |")

    # Best match to current repo eval scripts: IND is main-task train split.
    ind_vals = _ind_values_many(env_stats_list)
    ind_mu, ind_std = _mean_std(ind_vals)
    lines.append(f"| IND | {_fmt(ind_mu, ind_std)} |")

    for spec in ALL_ROWS:
        vals = _row_values_many(env_stats_list, spec)
        mu, sigma = _mean_std(vals)
        lines.append(f"| {spec.label} | {_fmt(mu, sigma)} |")

    vision_mu, vision_std = _pooled_over_tasks([_row_values_many(env_stats_list, s) for s in VISION_ROWS])
    lang_mu, lang_std = _pooled_over_tasks([_row_values_many(env_stats_list, s) for s in LANG_ROWS])
    action_mu, action_std = _pooled_over_tasks([_row_values_many(env_stats_list, s) for s in ACTION_ROWS])
    ood_mu, ood_std = _pooled_over_tasks([_row_values_many(env_stats_list, s) for s in ALL_ROWS])

    lines.append("| --- | --- |")
    lines.append(f"| OOD-Vision (pooled) | {_fmt(vision_mu, vision_std)} |")
    lines.append(f"| OOD-Lang (pooled) | {_fmt(lang_mu, lang_std)} |")
    lines.append(f"| OOD-Action (pooled) | {_fmt(action_mu, action_std)} |")
    lines.append(f"| OOD-All (pooled) | {_fmt(ood_mu, ood_std)} |")
    return lines


def _build_table_lines_for_checkpoint(
    load_path: str,
    env_stats: Dict[str, Any],
    labels_cfg: Dict[str, Dict[str, str]],
) -> List[str]:
    label = _resolve_label(load_path, labels_cfg)
    if label is None:
        title = load_path
    else:
        title = f"{load_path} [{label}]"
    return _build_table_lines(title, [env_stats])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-mode",
        default="any",
        choices=["any", "greedy", "temperature", "missing"],
        help="Filter wandb offline runs by eval decode mode before building the table.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    stats = build_stats(root, eval_mode=args.eval_mode)
    labels_cfg = _load_labels_config(root)
    all_lines: List[str] = []
    entries: List[tuple[str, Dict[str, Any], str | None]] = []
    for load_path in sorted(stats.keys()):
        env_stats = stats[load_path]
        if isinstance(env_stats, dict):
            entries.append((load_path, env_stats, _resolve_group(load_path, labels_cfg)))

    # Ungrouped entries first.
    for load_path, env_stats, group_name in entries:
        if group_name is None:
            all_lines.extend(_build_table_lines_for_checkpoint(load_path, env_stats, labels_cfg))

    # Grouped entries: print members, then aggregated table(s).
    group_order: List[str] = []
    grouped_entries: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for load_path, env_stats, group_name in entries:
        if group_name is None:
            continue
        if group_name not in grouped_entries:
            grouped_entries[group_name] = []
            group_order.append(group_name)
        grouped_entries[group_name].append((load_path, env_stats))

    for group_name in group_order:
        members = grouped_entries[group_name]
        all_lines.append(f"\n## Group: {group_name}")
        for load_path, env_stats in members:
            all_lines.extend(_build_table_lines_for_checkpoint(load_path, env_stats, labels_cfg))

        # Aggregate per checkpoint id inside the group (e.g. steps_0199 across seeds/runs).
        by_ckpt: Dict[str, List[Dict[str, Any]]] = {}
        for load_path, env_stats in members:
            ckpt = load_path.split("/")[-1]
            by_ckpt.setdefault(ckpt, []).append(env_stats)

        for ckpt in sorted(by_ckpt.keys()):
            env_stats_list = by_ckpt[ckpt]
            title = f"GROUP AVG {group_name} / {ckpt} (n_runs={len(env_stats_list)})"
            all_lines.extend(_build_table_lines(title, env_stats_list))

    output_text = "\n".join(all_lines).lstrip("\n")
    print(output_text)

    out_dir = root / "scripts" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_ts = out_dir / f"success-table-{timestamp}.md"
    out_latest = out_dir / "success-table-latest.md"
    out_ts.write_text(output_text + "\n")
    out_latest.write_text(output_text + "\n")
    print(f"\nEval mode filter: {args.eval_mode}")
    print(f"\nSaved table to: {out_ts}")
    print(f"Updated latest: {out_latest}")
    print(f"Labels file: {root / 'scripts' / 'stats' / 'run-labels.yaml'}")


if __name__ == "__main__":
    main()
