"""Plot one or more UR5 table-offset evaluation CSV files as heatmaps."""

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = (
    ("success", "Successful opening", "RdYlGn", 0.0, 1.0),
    ("steps", "Episode length [steps]", "magma", 0.0, None),
    ("final_angle_deg", "Final door angle [deg]", "viridis", 0.0, None),
    ("reward", "Episode reward", "coolwarm", None, None),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create position heatmaps from one or more CSV files produced by "
            "test_ur5_table_offsets.py."
        )
    )
    parser.add_argument("csv", type=Path, nargs="+", help="Evaluation CSV file(s)")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path (default: beside one CSV or in the current directory)",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--show", action="store_true", help="Also open the generated figure"
    )
    return parser.parse_args()


def parse_success(value):
    return 1.0 if str(value).strip().lower() in ("1", "true", "yes") else 0.0


def load_results(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no completed positions: {}".format(path))

    results = []
    for row in rows:
        final_angle_rad = row.get("final_door_angle_rad")
        if final_angle_rad in (None, ""):
            # Compatibility with CSV files made before final-angle logging.
            final_angle_rad = row["max_door_angle_rad"]
        results.append({
            "x": float(row["offset_x_cm"]),
            "y": float(row["offset_y_cm"]),
            "success": parse_success(row["success"]),
            "steps": float(row["steps"]),
            "final_angle_deg": float(np.degrees(float(final_angle_rad))),
            "reward": float(row["episode_reward"]),
        })

    first = rows[0]
    ft_mode = first.get("ft_mode", "unknown")
    label = {
        "normalized": "Model with normalized F/T",
        "none": "Model without F/T",
        "raw": "Model with raw F/T",
    }.get(ft_mode, "Model (F/T mode: {})".format(ft_mode))
    return {"path": path, "label": label, "results": results}


def to_grid(results, metric):
    x_values = sorted({row["x"] for row in results})
    y_values = sorted({row["y"] for row in results})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan)
    for row in results:
        grid[y_index[row["y"]], x_index[row["x"]]] = row[metric]
    return x_values, y_values, grid


def tick_labels(values):
    return ["{:g}".format(value) for value in values]


def default_output(csv_paths):
    if len(csv_paths) == 1:
        path = csv_paths[0]
        return path.with_name(path.stem + "_heatmaps.png")
    return Path.cwd() / "ur5_table_offset_comparison.png"


def main():
    args = parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [load_results(path.expanduser().resolve()) for path in args.csv]
    output = (args.output.expanduser().resolve() if args.output else
              default_output([item["path"] for item in datasets]).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_ranges = {}
    for metric, _, _, fixed_min, fixed_max in METRICS:
        values = [
            row[metric]
            for dataset in datasets
            for row in dataset["results"]
        ]
        metric_ranges[metric] = (
            fixed_min if fixed_min is not None else min(values),
            fixed_max if fixed_max is not None else max(values),
        )

    fig, axes = plt.subplots(
        len(datasets), len(METRICS),
        figsize=(18, 4.5 * len(datasets)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, dataset in enumerate(datasets):
        results = dataset["results"]
        success_rate = 100.0 * np.mean([row["success"] for row in results])
        mean_steps = np.mean([row["steps"] for row in results])
        mean_angle = np.mean([row["final_angle_deg"] for row in results])

        for column_index, (metric, title, cmap, _, _) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            x_values, y_values, grid = to_grid(results, metric)
            vmin, vmax = metric_ranges[metric]
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            image = axis.imshow(
                np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            axis.set_title(title)
            axis.set_xticks(range(len(x_values)), tick_labels(x_values), rotation=45)
            axis.set_yticks(range(len(y_values)), tick_labels(y_values))
            axis.set_xlabel("Cabinet offset X [cm]")
            if column_index == 0:
                axis.set_ylabel("Cabinet offset Y [cm]")
            fig.colorbar(image, ax=axis, shrink=0.82)

        axes[row_index, 0].text(
            0.0, 1.20,
            "{}\nSuccess: {:.1f}% | Avg. length: {:.1f} steps | "
            "Avg. final angle: {:.2f} deg".format(
                dataset["label"], success_rate, mean_steps, mean_angle
            ),
            transform=axes[row_index, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )

    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print("Saved heatmap figure:", output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
