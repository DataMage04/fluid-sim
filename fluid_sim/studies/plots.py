from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fluid_sim.studies.config import StudyPlotConfig


def plot_study_metric(
    rows: list[dict[str, object]],
    plot: StudyPlotConfig,
    output_directory: str | Path,
) -> Path:
    """Plot one scalar study result against another."""
    points: list[tuple[float, float, str]] = []

    for row in rows:
        x_value = row.get(plot.x)
        y_value = row.get(plot.y)
        if x_value is None or y_value is None:
            continue
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x_float) or not np.isfinite(y_float):
            continue
        if plot.log_x and x_float <= 0.0:
            continue
        if plot.log_y and y_float <= 0.0:
            continue
        points.append((x_float, y_float, str(row.get("label", ""))))

    if len(points) < 2:
        raise ValueError(
            f"Plot '{plot.filename}' requires at least two finite data points."
        )

    points.sort(key=lambda item: item[0])
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o")

    for x_value, y_value, label in points:
        if label:
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    if plot.log_x:
        ax.set_xscale("log")
    if plot.log_y:
        ax.set_yscale("log")

    ax.set_title(plot.title)
    ax.set_xlabel(plot.x_label)
    ax.set_ylabel(plot.y_label)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_directory) / plot.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
