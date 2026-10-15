from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fluid_sim.config import CaseConfig, load_case, save_case
from fluid_sim.io.output import TextReport, write_table_csv, write_to_output
from fluid_sim.studies.config import StudyConfig, load_study
from fluid_sim.studies.runner import build_study_case
from fluid_sim.studies.transient import (
    ENGINEERING_METRICS,
    TransientAnalysisConfig,
    _plot_history,
    _run_transient_case,
    summarize_history,
)


def obstacle_geometry_metrics(case: CaseConfig) -> tuple[float, float]:
    """Return obstacle height and blockage ratio for a single rectangle."""
    if len(case.obstacles) != 1:
        raise ValueError(
            "The blockage study requires exactly one rectangular obstacle."
        )

    obstacle = case.obstacles[0]
    height = obstacle.y_max - obstacle.y_min
    blockage_ratio = height / case.domain.height
    return float(height), float(blockage_ratio)


def _plot_averaged_vs_blockage(
    summaries: list[dict[str, object]],
    metric: str,
    output_path: Path,
    y_label: str,
) -> None:
    blockage = np.asarray(
        [float(row["blockage_ratio"]) for row in summaries],
        dtype=float,
    )
    means = np.asarray(
        [float(row[f"{metric}_mean"]) for row in summaries],
        dtype=float,
    )
    stds = np.asarray(
        [float(row[f"{metric}_std"]) for row in summaries],
        dtype=float,
    )

    order = np.argsort(blockage)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        blockage[order],
        means[order],
        yerr=stds[order],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Blockage ratio, H_obstacle / H_channel")
    ax.set_ylabel(y_label)
    ax.set_title(f"Time-Averaged {y_label} vs Blockage Ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _blockage_report(
    study: StudyConfig,
    config: TransientAnalysisConfig,
    summaries: list[dict[str, object]],
) -> str:
    report = TextReport(
        "CFD BLOCKAGE-RATIO STUDY REPORT",
        subtitle=f"Study: {study.name}",
    )

    report.add_section("Study design")
    report.add_fields(
        [
            ("Number of blockage cases", len(summaries)),
            ("Target Reynolds number", "40"),
            ("Sample interval", config.sample_interval, "steps"),
            ("Averaging start fraction", f"{config.averaging_start_fraction:.3f}"),
            ("Steadiness tolerance", f"{100.0 * config.steadiness_tolerance:.3f}", "%"),
        ]
    )
    report.add_lines(
        [
            "Obstacle height is varied while the obstacle remains vertically centred.",
            "Dynamic viscosity and characteristic length are adjusted together so",
            "the obstacle-based Reynolds number remains Re = 40 for every run.",
        ]
    )

    report.add_section("Time-averaged blockage study")
    report.add_table(
        (
            "Run",
            "Beta",
            "H_obs",
            "Re",
            "mu",
            "dP mean",
            "Cd,p mean",
            "Wake mean",
            "Reverse %",
            "u_min mean",
            "Umax mean",
            "Status",
        ),
        [
            (
                row["label"],
                f"{float(row['blockage_ratio']):.3f}",
                f"{float(row['obstacle_height_m']):.3f}",
                f"{float(row['reynolds_number']):.1f}",
                f"{float(row['dynamic_viscosity_Pa_s']):.5f}",
                f"{float(row['pressure_drop_mean']):.5f}",
                f"{float(row['pressure_drag_coefficient_mean']):.5f}",
                f"{float(row['wake_length_mean']):.5f}",
                f"{100.0 * float(row['reverse_flow_fraction_mean']):.3f}",
                f"{float(row['min_streamwise_velocity_mean']):.5f}",
                f"{float(row['max_speed_mean']):.5f}",
                row["steadiness_status"],
            )
            for row in summaries
        ],
    )

    report.add_section("Steadiness assessment")
    report.add_table(
        (
            "Run",
            "Beta",
            "Max drift (%)",
            "Max fluct. (%)",
            "Average samples",
            "Status",
        ),
        [
            (
                row["label"],
                f"{float(row['blockage_ratio']):.3f}",
                f"{100.0 * float(row['max_relative_drift']):.4f}",
                f"{100.0 * float(row['max_relative_fluctuation']):.4f}",
                row["average_sample_count"],
                row["steadiness_status"],
            )
            for row in summaries
        ],
    )

    return report.render()


def run_blockage_study(study_path: str | Path) -> list[dict[str, object]]:
    """Run the constant-Re blockage-ratio study with transient averaging."""
    study_path = Path(study_path).resolve()
    study = load_study(study_path)
    transient = TransientAnalysisConfig.from_study_file(study_path)

    base_case_path = Path(study.base_case)
    if not base_case_path.is_absolute():
        base_case_path = (study_path.parent / base_case_path).resolve()
    base_case = load_case(base_case_path)

    output_directory = Path(study.output_directory) / "transient"
    output_directory.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list[dict[str, float]]] = {}
    summaries: list[dict[str, object]] = []

    for run in study.runs:
        case = build_study_case(base_case, study, run, output_directory)
        case.output.save_plots = False
        case.output.save_animation = False
        case.output.show_final_plot = False

        obstacle_height, blockage_ratio = obstacle_geometry_metrics(case)

        run_directory = output_directory / "runs" / run.label
        run_directory.mkdir(parents=True, exist_ok=True)
        save_case(case, run_directory / "resolved_case.json")

        history, elapsed = _run_transient_case(case, transient)
        histories[run.label] = history
        write_table_csv(history, run_directory / "engineering_history.csv")

        summary = summarize_history(history, transient)
        summary["label"] = run.label
        summary["runtime_s"] = elapsed
        summary["dynamic_viscosity_Pa_s"] = case.fluid.dynamic_viscosity
        summary["obstacle_height_m"] = obstacle_height
        summary["blockage_ratio"] = blockage_ratio
        summaries.append(summary)

        write_table_csv(
            [summary],
            run_directory / "time_averaged_results.csv",
        )

    write_table_csv(
        summaries,
        output_directory / "time_averaged_summary.csv",
    )

    plot_specs = {
        "pressure_drop": (
            "Pressure drop (Pa)",
            "pressure_drop_history.png",
            "pressure_drop_vs_blockage.png",
        ),
        "pressure_drag_coefficient": (
            "Pressure-drag coefficient, Cd,p",
            "pressure_drag_history.png",
            "pressure_drag_vs_blockage.png",
        ),
        "wake_length": (
            "Wake length (m)",
            "wake_length_history.png",
            "wake_length_vs_blockage.png",
        ),
        "reverse_flow_fraction": (
            "Reverse-flow fraction",
            "reverse_flow_history.png",
            "reverse_flow_vs_blockage.png",
        ),
        "min_streamwise_velocity": (
            "Minimum streamwise velocity (m/s)",
            "reverse_velocity_history.png",
            "reverse_velocity_vs_blockage.png",
        ),
        "max_speed": (
            "Maximum speed (m/s)",
            "max_speed_history.png",
            "max_speed_vs_blockage.png",
        ),
    }

    for metric, (label, history_name, average_name) in plot_specs.items():
        _plot_history(
            histories,
            metric,
            output_directory / history_name,
            label,
        )
        _plot_averaged_vs_blockage(
            summaries,
            metric,
            output_directory / average_name,
            label,
        )

    write_to_output(
        _blockage_report(study, transient, summaries),
        output_directory / "blockage_report.txt",
    )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the constant-Re obstacle blockage-ratio study with "
            "time-history and time-averaged engineering metrics."
        )
    )
    parser.add_argument("study", help="Path to a blockage study JSON file.")
    args = parser.parse_args()
    run_blockage_study(args.study)


if __name__ == "__main__":
    main()
