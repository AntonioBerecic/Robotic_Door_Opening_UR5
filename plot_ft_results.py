"""Visualize UR5 wrist force/torque CSV logs produced by train.py."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "episode", "step", "fx_N", "fy_N", "fz_N",
    "tx_Nm", "ty_Nm", "tz_Nm",
    "force_magnitude_N", "torque_magnitude_Nm",
}


def load_ft_log(path):
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "CSV is missing required column(s): {}".format(
                ", ".join(sorted(missing))))
    if data.empty:
        raise ValueError("CSV contains no force/torque samples")
    return data.sort_values(["episode", "step"])


def mean_and_std_by_step(data, column):
    grouped = data.groupby("step")[column]
    return grouped.mean(), grouped.std().fillna(0.0)


def plot_episode_lines(ax, data, column, ylabel, color):
    episodes = sorted(data["episode"].unique())
    for episode in episodes:
        episode_data = data[data["episode"] == episode]
        ax.plot(
            episode_data["step"], episode_data[column],
            color=color, alpha=0.18, linewidth=0.9,
        )

    mean, std = mean_and_std_by_step(data, column)
    steps = mean.index.to_numpy()
    mean_values = mean.to_numpy()
    std_values = std.to_numpy()
    ax.fill_between(
        steps, mean_values - std_values, mean_values + std_values,
        color=color, alpha=0.16, label="mean ± std",
    )
    ax.plot(steps, mean_values, color=color, linewidth=2.2, label="mean")
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")


def create_figure(data, source_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        "UR5 wrist force/torque evaluation — {} episodes\n{}".format(
            data["episode"].nunique(), source_name),
        fontsize=14,
    )

    plot_episode_lines(
        axes[0, 0], data, "force_magnitude_N", "Force magnitude [N]", "tab:red")
    axes[0, 0].set_title("Force magnitude through each episode")

    plot_episode_lines(
        axes[0, 1], data, "torque_magnitude_Nm",
        "Torque magnitude [Nm]", "tab:purple")
    axes[0, 1].set_title("Torque magnitude through each episode")

    component_colors = {
        "fx_N": ("Fx", "tab:red"),
        "fy_N": ("Fy", "tab:green"),
        "fz_N": ("Fz", "tab:blue"),
    }
    for column, (label, color) in component_colors.items():
        mean, _ = mean_and_std_by_step(data, column)
        axes[1, 0].plot(
            mean.index, mean.values, label=label, color=color, linewidth=1.8)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1, 0].set_title("Mean force components across episodes")
    axes[1, 0].set_xlabel("Evaluation step")
    axes[1, 0].set_ylabel("Force [N]")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    episode_stats = data.groupby("episode")["force_magnitude_N"].agg(
        mean="mean", peak="max")
    x = np.arange(len(episode_stats))
    width = 0.38
    axes[1, 1].bar(
        x - width / 2, episode_stats["mean"], width,
        label="mean force", color="tab:orange")
    axes[1, 1].bar(
        x + width / 2, episode_stats["peak"], width,
        label="peak force", color="tab:red")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(episode_stats.index.astype(str))
    axes[1, 1].set_title("Force summary per episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Force magnitude [N]")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()

    return fig


def default_output_path(csv_path):
    stem, _ = os.path.splitext(csv_path)
    return stem + "_plot.png"


def main():
    parser = argparse.ArgumentParser(
        description="Plot UR5 force/torque evaluation data from --ft_log CSV")
    parser.add_argument("csv", help="CSV created by train.py --ft_log")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--dpi", type=int, default=160, help="Output image resolution")
    parser.add_argument(
        "--show", action="store_true", help="Open an interactive plot window")
    args = parser.parse_args()

    output = args.output or default_output_path(args.csv)
    data = load_ft_log(args.csv)
    figure = create_figure(data, os.path.basename(args.csv))

    output_parent = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_parent, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print("Saved F/T visualization:", os.path.abspath(output))
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
