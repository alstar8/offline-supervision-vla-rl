from __future__ import annotations

import argparse
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import yaml

MAIN_ENV_ID = "PutOnPlateInScene25Main-v3"


def _eval_mode_from_cfg(cfg: Dict[str, Any]) -> str:
    if "eval_deterministic" not in cfg:
        return "missing"
    return "greedy" if bool(cfg["eval_deterministic"]) else "temperature"


def build_stats(root: Path, eval_mode: str = "any") -> Dict[str, Any]:
    if eval_mode not in {"any", "greedy", "temperature", "missing"}:
        raise ValueError(f"Unsupported eval_mode={eval_mode}")
    stats: Dict[str, Any] = {}

    # Merge previously saved stats files.
    old_stats_path = root / "scripts" / "stats"
    old_stats_files = sorted(old_stats_path.glob("stats-*.yaml"), key=lambda x: x.name)
    for old_stat in old_stats_files:
        cfg = yaml.safe_load(old_stat.read_text())
        if not isinstance(cfg, dict):
            continue
        for load_path, envs in cfg.items():
            if load_path not in stats:
                stats[load_path] = {}
            for env_name, seeds in envs.items():
                if env_name not in stats[load_path]:
                    stats[load_path][env_name] = {}
                for seed, stat in seeds.items():
                    if seed not in stats[load_path][env_name]:
                        stats[load_path][env_name][seed] = {}
                    stats[load_path][env_name][seed].update(stat)

    # Merge offline wandb runs.
    wandb_path = root / "wandb"
    runs = wandb_path.glob("offline-run-*")
    for run in runs:
        cfg_path = run / "glob" / "config.yaml"
        if not cfg_path.exists():
            continue

        cfg = yaml.safe_load(cfg_path.read_text())
        if not isinstance(cfg, dict):
            continue
        if "vla_load_path" not in cfg or "env_id" not in cfg or "seed" not in cfg:
            continue
        run_eval_mode = _eval_mode_from_cfg(cfg)
        if eval_mode != "any" and run_eval_mode != eval_mode:
            continue

        load_path = "/".join(str(cfg["vla_load_path"]).split("/")[-3:])
        env_name = str(cfg["env_id"])
        seed = cfg["seed"]

        if load_path not in stats:
            stats[load_path] = {}
        if env_name not in stats[load_path]:
            stats[load_path][env_name] = {}

        train_vis_file = run / "glob" / "vis_0_train" / "stats.yaml"
        if train_vis_file.exists():
            train_stats = yaml.safe_load(train_vis_file.read_text())
            if isinstance(train_stats, dict) and "stats" in train_stats:
                if "train" not in stats[load_path][env_name]:
                    stats[load_path][env_name]["train"] = {}
                stats[load_path][env_name]["train"][seed] = train_stats["stats"]
                stats[load_path][env_name]["train"][seed]["path"] = str(run)
                stats[load_path][env_name]["train"][seed]["eval_mode"] = run_eval_mode

        test_vis_file = run / "glob" / "vis_0_test" / "stats.yaml"
        if test_vis_file.exists():
            test_stats = yaml.safe_load(test_vis_file.read_text())
            if isinstance(test_stats, dict) and "stats" in test_stats:
                if "test" not in stats[load_path][env_name]:
                    stats[load_path][env_name]["test"] = {}
                stats[load_path][env_name]["test"][seed] = test_stats["stats"]
                stats[load_path][env_name]["test"][seed]["path"] = str(run)
                stats[load_path][env_name]["test"][seed]["eval_mode"] = run_eval_mode

    return stats


def save_stats(root: Path, stats: Dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = root / "scripts" / "stats" / f"stats-{timestamp}.yaml"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)
    return save_path


def _collect_success(entries: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for _, metrics in entries.items():
        if isinstance(metrics, dict) and "success" in metrics:
            values.append(float(metrics["success"]))
    return values


def _avg(values: List[float]) -> float | None:
    return mean(values) if values else None


def _scatter(values: List[float]) -> float | None:
    if len(values) < 2:
        return None
    return stdev(values)


def _format_mean_scatter(values: List[float]) -> str:
    avg = _avg(values)
    if avg is None:
        return "n/a"
    scatter = _scatter(values)
    if scatter is None:
        return f"{avg * 100:.1f}%"
    return f"{avg * 100:.1f}% +/- {scatter * 100:.1f}%"


def _format_overall_from_tasks(task_values: List[List[float]]) -> str:
    # Treat each task as an individual quantity:
    # - overall mean is the average of per-task means
    # - overall scatter uses only within-task seed deviations
    valid_tasks = [vals for vals in task_values if vals]
    if not valid_tasks:
        return "n/a"

    task_means = [mean(vals) for vals in valid_tasks]
    overall_mean = mean(task_means)

    n_samples = sum(len(vals) for vals in valid_tasks)
    dof = n_samples - len(valid_tasks)
    if dof <= 0:
        return f"{overall_mean * 100:.1f}%"

    sq_dev_sum = 0.0
    for vals, task_mean in zip(valid_tasks, task_means):
        for v in vals:
            sq_dev_sum += (v - task_mean) ** 2

    overall_scatter = sqrt(sq_dev_sum / dof)
    return f"{overall_mean * 100:.1f}% +/- {overall_scatter * 100:.1f}%"


def print_aggregates(stats: Dict[str, Any]) -> None:
    for load_path, envs in stats.items():
        print(f"\n=== {load_path} ===")
        in_dist_task_values: List[List[float]] = []
        ood_task_values: List[List[float]] = []

        for env_name, splits in envs.items():
            if not isinstance(splits, dict):
                continue
            train_vals = _collect_success(splits.get("train", {}))
            test_vals = _collect_success(splits.get("test", {}))

            if env_name == MAIN_ENV_ID:
                env_vals = train_vals + test_vals
                if env_vals:
                    in_dist_task_values.append(env_vals)
                env_text = _format_mean_scatter(env_vals)
                print(f"IN_DIST {env_name}: mean={env_text}")
            else:
                # For OOD, ignore train split and only evaluate test split.
                env_vals = test_vals
                if env_vals:
                    ood_task_values.append(env_vals)
                env_text = _format_mean_scatter(env_vals)
                print(f"OOD {env_name}: mean={env_text}")

        in_dist_text = _format_overall_from_tasks(in_dist_task_values)
        ood_text = _format_overall_from_tasks(ood_task_values)
        print(f"IN_DIST overall ({MAIN_ENV_ID}): {in_dist_text}")
        print(f"OOD overall test-only (all envs except {MAIN_ENV_ID}): {ood_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-mode",
        default="any",
        choices=["any", "greedy", "temperature", "missing"],
        help="Filter wandb offline runs by eval decode mode. 'missing' matches runs without eval_deterministic.",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    stats = build_stats(root, eval_mode=args.eval_mode)
    save_path = save_stats(root, stats)
    print(f"Saved merged stats to: {save_path} (eval_mode={args.eval_mode})")
    print_aggregates(stats)


if __name__ == "__main__":
    main()
