from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fluid_sim.analysis.engineering import compute_engineering_results
from fluid_sim.config import CaseConfig, load_case, save_case
from fluid_sim.io.output import TextReport, write_table_csv, write_to_output
from fluid_sim.numerics.advection import advect_velocity
from fluid_sim.numerics.boundary_conditions import apply_boundary_conditions
from fluid_sim.numerics.diffusion import diffuse_velocity
from fluid_sim.numerics.solver import project_velocity, solve_pressure
from fluid_sim.simulation import build_grid, initialize_velocity
from fluid_sim.studies.config import StudyConfig, load_study
from fluid_sim.studies.runner import build_study_case


@dataclass(frozen=True)
class TransientAnalysisConfig:
    sample_interval: int = 10
    averaging_start_fraction: float = 0.6
    steadiness_tolerance: float = 0.01
    minimum_average_samples: int = 10

    def validate(self) -> None:
        if self.sample_interval <= 0:
            raise ValueError("Transient sample_interval must be positive.")
        if not 0.0 <= self.averaging_start_fraction < 1.0:
            raise ValueError(
                "Transient averaging_start_fraction must lie in [0, 1)."
            )
        if self.steadiness_tolerance <= 0.0:
            raise ValueError("Transient steadiness_tolerance must be positive.")
        if self.minimum_average_samples < 2:
            raise ValueError(
                "Transient minimum_average_samples must be at least 2."
            )

    @classmethod
    def from_study_file(cls, path: str | Path) -> "TransientAnalysisConfig":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        config = cls(**data.get("transient_analysis", {}))
        config.validate()
        return config


ENGINEERING_METRICS = (
    "pressure_drop",
    "pressure_drag_coefficient",
    "wake_length",
    "reverse_flow_fraction",
    "min_streamwise_velocity",
    "max_speed",
)

STEADINESS_METRICS = (
    "pressure_drop",
    "pressure_drag_coefficient",
)


def _engineering_sample(
    grid,
    case: CaseConfig,
    *,
    step: int,
) -> dict[str, float]:
    eng = compute_engineering_results(
        grid,
        density=case.fluid.density,
        reference_velocity=case.inlet.mean_velocity,
        upstream_x=case.engineering.upstream_x,
        downstream_x=case.engineering.downstream_x,
        sample_width=case.engineering.sample_width,
        reference_pressure=case.engineering.reference_pressure,
        analytical_pressure_gradient=None,
    )

    row = {
        "step": step,
        "time_s": step * case.time.dt,
        "reynolds_number": case.reynolds_number,
    }
    for metric in ENGINEERING_METRICS:
        value = getattr(eng, metric)
        row[metric] = float(value) if value is not None else np.nan
    return row


def _relative_mean_drift(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    split = max(1, len(values) // 2)
    first = values[:split]
    second = values[split:]
    if len(second) == 0:
        return float("nan")
    overall = float(np.mean(values))
    denominator = max(abs(overall), 1.0e-12)
    return abs(float(np.mean(second)) - float(np.mean(first))) / denominator


def summarize_history(
    history: list[dict[str, float]],
    config: TransientAnalysisConfig,
) -> dict[str, float | int | str | bool]:
    if not history:
        raise ValueError("Cannot summarize an empty engineering history.")

    final_time = float(history[-1]["time_s"])
    averaging_start_time = config.averaging_start_fraction * final_time
    window = [
        row for row in history
        if float(row["time_s"]) >= averaging_start_time
    ]

    if len(window) < config.minimum_average_samples:
        raise ValueError(
            "Transient averaging window contains too few samples: "
            f"{len(window)} < {config.minimum_average_samples}."
        )

    summary: dict[str, float | int | str | bool] = {
        "reynolds_number": float(history[-1]["reynolds_number"]),
        "sample_count": len(history),
        "average_sample_count": len(window),
        "averaging_start_time_s": averaging_start_time,
        "final_time_s": final_time,
    }

    for metric in ENGINEERING_METRICS:
        values = np.asarray([float(row[metric]) for row in window], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[f"{metric}_mean"] = np.nan
            summary[f"{metric}_std"] = np.nan
            summary[f"{metric}_final"] = float(history[-1][metric])
            continue

        summary[f"{metric}_mean"] = float(np.mean(finite))
        summary[f"{metric}_std"] = float(np.std(finite))
        summary[f"{metric}_final"] = float(history[-1][metric])

    max_drift = 0.0
    max_relative_fluctuation = 0.0
    for metric in STEADINESS_METRICS:
        values = np.asarray([float(row[metric]) for row in window], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        drift = _relative_mean_drift(values)
        mean = float(np.mean(values))
        relative_fluctuation = float(np.std(values)) / max(abs(mean), 1.0e-12)
        summary[f"{metric}_relative_drift"] = drift
        summary[f"{metric}_relative_fluctuation"] = relative_fluctuation
        max_drift = max(max_drift, drift)
        max_relative_fluctuation = max(
            max_relative_fluctuation,
            relative_fluctuation,
        )

    steady = (
        max_drift <= config.steadiness_tolerance
        and max_relative_fluctuation <= config.steadiness_tolerance
    )
    summary["max_relative_drift"] = max_drift
    summary["max_relative_fluctuation"] = max_relative_fluctuation
    summary["steady_by_window_test"] = steady
    summary["steadiness_status"] = "STEADY" if steady else "UNSTEADY/DEVELOPING"
    return summary


def _run_transient_case(
    case: CaseConfig,
    config: TransientAnalysisConfig,
) -> tuple[list[dict[str, float]], float]:
    if not case.engineering.enabled:
        raise ValueError("Transient engineering analysis requires engineering.enabled=true.")

    grid = build_grid(case)
    initialize_velocity(grid, case)

    rho = case.fluid.density
    nu = case.fluid.kinematic_viscosity
    dt = case.time.dt
    history: list[dict[str, float]] = []

    start = perf_counter()
    for zero_based_step in tqdm(
        range(case.time.steps),
        desc=f"Transient: {case.name}",
    ):
        apply_boundary_conditions(
            grid,
            inlet_velocity=case.inlet.mean_velocity,
            inlet_profile=case.inlet.profile,
        )
        advect_velocity(grid, dt)
        apply_boundary_conditions(
            grid,
            inlet_velocity=case.inlet.mean_velocity,
            inlet_profile=case.inlet.profile,
        )
        diffuse_velocity(grid, dt, nu)
        apply_boundary_conditions(
            grid,
            inlet_velocity=case.inlet.mean_velocity,
            inlet_profile=case.inlet.profile,
        )
        solve_pressure(grid, dt, rho=rho)
        project_velocity(grid, dt, rho=rho)

        step = zero_based_step + 1
        if step % config.sample_interval == 0 or step == case.time.steps:
            history.append(_engineering_sample(grid, case, step=step))

    return history, perf_counter() - start


def _plot_history(
    histories: dict[str, list[dict[str, float]]],
    metric: str,
    output_path: Path,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, history in histories.items():
        times = [row["time_s"] for row in history]
        values = [row[metric] for row in history]
        ax.plot(times, values, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} Time History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_averaged_metric(
    summaries: list[dict[str, object]],
    metric: str,
    output_path: Path,
    y_label: str,
) -> None:
    reynolds = np.asarray([float(row["reynolds_number"]) for row in summaries])
    means = np.asarray([float(row[f"{metric}_mean"]) for row in summaries])
    stds = np.asarray([float(row[f"{metric}_std"]) for row in summaries])

    order = np.argsort(reynolds)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        reynolds[order],
        means[order],
        yerr=stds[order],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Reynolds number, Re")
    ax.set_ylabel(y_label)
    ax.set_title(f"Time-Averaged {y_label} vs Reynolds Number")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _transient_report(
    study: StudyConfig,
    config: TransientAnalysisConfig,
    summaries: list[dict[str, object]],
) -> str:
    report = TextReport(
        "CFD TRANSIENT ENGINEERING STUDY REPORT",
        subtitle=f"Study: {study.name}",
    )
    report.add_section("Transient analysis setup")
    report.add_fields(
        [
            ("Sample interval", config.sample_interval, "steps"),
            ("Averaging start fraction", f"{config.averaging_start_fraction:.3f}"),
            ("Steadiness tolerance", f"{100.0 * config.steadiness_tolerance:.3f}", "%"),
            ("Minimum averaging samples", config.minimum_average_samples),
        ]
    )

    report.add_section("Time-averaged Reynolds study")
    report.add_table(
        (
            "Run",
            "Re",
            "dP mean",
            "Cd,p mean",
            "Wake mean",
            "Reverse %",
            "u_min mean",
            "Status",
        ),
        [
            (
                row["label"],
                f"{float(row['reynolds_number']):.0f}",
                f"{float(row['pressure_drop_mean']):.5f}",
                f"{float(row['pressure_drag_coefficient_mean']):.5f}",
                f"{float(row['wake_length_mean']):.5f}",
                f"{100.0 * float(row['reverse_flow_fraction_mean']):.3f}",
                f"{float(row['min_streamwise_velocity_mean']):.5f}",
                row["steadiness_status"],
            )
            for row in summaries
        ],
    )

    report.add_section("Steadiness assessment")
    report.add_table(
        (
            "Run",
            "Max drift (%)",
            "Max fluct. (%)",
            "Average samples",
            "Status",
        ),
        [
            (
                row["label"],
                f"{100.0 * float(row['max_relative_drift']):.4f}",
                f"{100.0 * float(row['max_relative_fluctuation']):.4f}",
                row["average_sample_count"],
                row["steadiness_status"],
            )
            for row in summaries
        ],
    )
    report.add_lines(
        [
            "The steadiness test compares the first and second halves of the",
            "averaging window and the RMS-like standard deviation of pressure drop",
            "and pressure-drag coefficient. A failed test means the reported window",
            "average should be treated as transient/developing rather than steady-state.",
        ]
    )
    return report.render()


def run_transient_study(study_path: str | Path) -> list[dict[str, object]]:
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
        summaries.append(summary)
        write_table_csv([summary], run_directory / "time_averaged_results.csv")

    write_table_csv(summaries, output_directory / "time_averaged_summary.csv")

    plot_specs = {
        "pressure_drop": ("Pressure drop (Pa)", "pressure_drop_history.png", "pressure_drop_time_averaged.png"),
        "pressure_drag_coefficient": ("Pressure-drag coefficient, Cd,p", "pressure_drag_history.png", "pressure_drag_time_averaged.png"),
        "wake_length": ("Wake length (m)", "wake_length_history.png", "wake_length_time_averaged.png"),
        "reverse_flow_fraction": ("Reverse-flow fraction", "reverse_flow_history.png", "reverse_flow_time_averaged.png"),
        "min_streamwise_velocity": ("Minimum streamwise velocity (m/s)", "reverse_velocity_history.png", "reverse_velocity_time_averaged.png"),
        "max_speed": ("Maximum speed (m/s)", "max_speed_history.png", "max_speed_time_averaged.png"),
    }
    for metric, (label, history_name, average_name) in plot_specs.items():
        _plot_history(
            histories,
            metric,
            output_directory / history_name,
            label,
        )
        _plot_averaged_metric(
            summaries,
            metric,
            output_directory / average_name,
            label,
        )

    write_to_output(
        _transient_report(study, transient, summaries),
        output_directory / "transient_report.txt",
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run time-history and time-averaged engineering analysis for a study."
    )
    parser.add_argument("study", help="Path to a study JSON file.")
    args = parser.parse_args()
    run_transient_study(args.study)


if __name__ == "__main__":
    main()
