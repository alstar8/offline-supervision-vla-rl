#!/usr/bin/env python3
from __future__ import annotations

import argparse

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Log constant eval/success and eval/success_ood values to W&B "
            "across a step range so they appear as horizontal threshold lines."
        )
    )
    parser.add_argument("--project", default="RLVLA")
    parser.add_argument("--name", default="eval-threshold-lines")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--success", type=float, required=True, help="Constant value for eval/success")
    parser.add_argument("--success-ood", type=float, required=True, help="Constant value for eval/success_ood")
    parser.add_argument("--step-start", type=int, default=0)
    parser.add_argument("--step-end", type=int, default=2_000_000)
    parser.add_argument("--step-interval", type=int, default=1_000)
    args = parser.parse_args()

    if args.step_start < 0 or args.step_end < args.step_start:
        raise ValueError("Invalid step range.")
    if args.step_interval <= 0:
        raise ValueError("step-interval must be > 0.")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        mode=args.mode,
        config={
            "threshold_success": args.success,
            "threshold_success_ood": args.success_ood,
            "step_start": args.step_start,
            "step_end": args.step_end,
            "step_interval": args.step_interval,
        },
    )

    for step in range(args.step_start, args.step_end + 1, args.step_interval):
        wandb.log(
            {
                "eval/success": args.success,
                "eval/success_ood": args.success_ood,
            },
            step=step,
        )

    # Ensure exact endpoint exists even if not hit by interval.
    if (args.step_end - args.step_start) % args.step_interval != 0:
        wandb.log(
            {
                "eval/success": args.success,
                "eval/success_ood": args.success_ood,
            },
            step=args.step_end,
        )

    print(f"Run URL: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
