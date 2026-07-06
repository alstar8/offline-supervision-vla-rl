#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import wandb


def load_rows(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    betas: list[float] = []
    ind_vals: list[float] = []
    ood_vals: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {csv_path}")

        required = {"beta", "eval/success", "eval/ood_success"}
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path}: {missing}. "
                "Expected columns: beta, eval/success, eval/ood_success"
            )

        for row in reader:
            betas.append(float(row["beta"]))
            ind_vals.append(float(row["eval/success"]))
            ood_vals.append(float(row["eval/ood_success"]))

    if not betas:
        raise ValueError(f"No data rows found in CSV: {csv_path}")

    return betas, ind_vals, ood_vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload beta-vs-success points to W&B as a Table and create "
            "beta-on-X custom plots for IND and OOD success."
        )
    )
    parser.add_argument("--project", default="RLVLA")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default="beta-ind-ood-table")
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument(
        "--csv",
        default="/workspace/rlvla_root/rlvla_mod/SimplerEnv/scripts/stats/beta_ind_ood_table.csv",
        help="CSV path with columns: beta, eval/success, eval/ood_success",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    betas, ind_vals, ood_vals = load_rows(csv_path)
    x_min = min(betas)
    x_max = max(betas)
    x_start = math.floor(x_min * 10.0) / 10.0
    x_end = math.ceil(x_max * 10.0) / 10.0
    x_ticks = [round(x_start + 0.1 * i, 2) for i in range(int(round((x_end - x_start) / 0.1)) + 1)]
    y_ticks = [round(0.3 + 0.1 * i, 2) for i in range(8)]

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        mode=args.mode,
        config={
            "source_csv": str(csv_path),
            "x_axis": "beta",
        },
    )

    table = wandb.Table(columns=["beta", "eval/success", "eval/ood_success"])
    for b, i, o in zip(betas, ind_vals, ood_vals):
        table.add_data(b, i, o)

    wandb.log({"beta_metrics_table": table})
    wandb.log(
        {
            "plots/ind_success_vs_beta": wandb.plot.line(
                table, "beta", "eval/success", title="IND Success vs Beta"
            ),
            "plots/ood_success_vs_beta": wandb.plot.line(
                table, "beta", "eval/ood_success", title="OOD Success vs Beta"
            ),
        }
    )

    # Native W&B styled chart via custom Vega spec:
    # - visible points
    # - thicker lines
    # - fixed y range [0.3, 1.0]
    long_table = wandb.Table(columns=["beta", "value", "series"])
    for b, i, o in zip(betas, ind_vals, ood_vals):
        long_table.add_data(b, i, "IND")
        long_table.add_data(b, o, "OOD")

    try:
        chart_name = f"beta-ind-ood-styled-{int(time.time())}"
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "title": {
                "text": "Success Rate",
                "anchor": "middle",
                "fontSize": 28,
                "fontWeight": 700,
                "color": "#2f3b52",
                "offset": 14,
            },
            "background": "#f4f5f7",
            "width": 760,
            "height": 420,
            "config": {
                "view": {"stroke": "#d8dbe1"},
                "axis": {
                    "labelFontSize": 15,
                    "titleFontSize": 16,
                    "labelFontWeight": "normal",
                    "titleFontWeight": "bold",
                    "labelColor": "#4b5563",
                    "titleColor": "#2f3b52",
                    "gridColor": "#e1e5eb",
                    "tickColor": "#c9cfd9",
                    "domainColor": "#c9cfd9",
                },
                "legend": {
                    "labelFontSize": 14,
                    "titleFontSize": 14,
                    "labelColor": "#4b5563",
                    "titleColor": "#4b5563",
                    "orient": "bottom",
                    "direction": "horizontal",
                    "symbolType": "circle",
                },
            },
            "data": {"name": "wandb"},
            "mark": {
                "type": "line",
                "strokeWidth": 3,
                "point": {"filled": True, "size": 180},
            },
            "encoding": {
                "x": {
                    "field": "${field:beta}",
                    "type": "quantitative",
                    "axis": {"title": "beta", "values": x_ticks},
                },
                "y": {
                    "field": "${field:value}",
                    "type": "quantitative",
                    "axis": {"title": "success", "values": y_ticks},
                    "scale": {"domain": [0.3, 1.0]},
                },
                "color": {
                    "field": "${field:series}",
                    "type": "nominal",
                    "scale": {"range": ["#4C78A8", "#F58518"]},
                    "legend": {"title": "metric"},
                },
                "tooltip": [
                    {"field": "${field:series}", "type": "nominal"},
                    {"field": "${field:beta}", "type": "quantitative"},
                    {"field": "${field:value}", "type": "quantitative"},
                ],
            },
        }

        api = wandb.Api()
        viewer = api.viewer
        chart_owner = args.entity or run.entity or getattr(viewer, "entity", None)
        if not chart_owner:
            raise RuntimeError(
                "Could not resolve chart owner entity for custom chart creation. "
                "Pass --entity explicitly."
            )
        chart_id = api.create_custom_chart(
            entity=chart_owner,
            name=chart_name,
            display_name="Beta Success Styled",
            spec_type="vega2",
            access="private",
            spec=vega_spec,
        )
        styled_chart = wandb.plot_table(
            vega_spec_name=chart_id,
            data_table=long_table,
            fields={"beta": "beta", "value": "value", "series": "series"},
        )
        wandb.log({"plots/beta_success_styled_native": styled_chart, "beta_metrics_long_table": long_table})
    except Exception as exc:
        print(f"Warning: native styled custom chart failed ({type(exc).__name__}: {exc}).")
        print("Falling back to built-in line plots only.")

    print(f"Run URL: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
