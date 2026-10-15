from __future__ import annotations

import csv
from copy import deepcopy
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from tqdm import tqdm

from fluid_sim.analysis.engineering import (
    analytical_poiseuille_pressure_gradient,
    compute_engineering_results,
)
from fluid_sim.analysis.validation import (
    analytical_poiseuille_profile,
    numerical_velocity_profile,
    profile_error,
)
from fluid_sim.analysis.visualization import flow_rate_metrics, get_cell_center_velocity
from fluid_sim.config import CaseConfig, load_case
from fluid_sim.io.output import TextReport, write_table_csv, write_to_output
from fluid_sim.numerics.solver import divergence_metrics
from fluid_sim.simulation import run_case
from fluid_sim.studies.config import StudyConfig, StudyRunConfig, load_study
from fluid_sim.studies.plots import plot_study_metric


def _set_nested_value(data: Any, path: str, value: Any) -> None:
    """Set a dotted dict/list path such as 'domain.nx' or 'obstacles.0.y_max'."""
    tokens = path.split(".")
    target = data

    for token in tokens[:-1]:
        if isinstance(target, list):
            try:
                target = target[int(token)]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid list override path: {path}") from exc
        elif isinstance(target, dict):
            if token not in target:
                raise ValueError(f"Unknown override path: {path}")
            target = target[token]
        else:
            raise ValueError(f"Override path cannot descend through '{token}': {path}")

    final = tokens[-1]
    if isinstance(target, list):
        try:
            index = int(final)
        except ValueError as exc:
            raise ValueError(f"Invalid list override path: {path}") from exc
        if index < 0 or index >= len(target):
            raise ValueError(f"List override index out of range: {path}")
        target[index] = value
    elif isinstance(target, dict):
        if final not in target:
            raise ValueError(f"Unknown override path: {path}")
        target[final] = value
    else:
        raise ValueError(f"Override path cannot assign through '{final}': {path}")


def build_study_case(
    base_case: CaseConfig,
    study: StudyConfig,
    run: StudyRunConfig,
    output_directory: str | Path,
) -> CaseConfig:
    """Create one independent simulation case from a base case and overrides."""
    data = deepcopy(base_case.to_dict())

    for path, value in run.overrides.items():
        _set_nested_value(data, path, value)

    data["name"] = f"{study.name}_{run.label}"
    data["output"]["directory"] = str(Path(output_directory) / "runs" / run.label)
    data["output"]["save_plots"] = bool(study.keep_case_plots)
    data["output"]["save_animation"] = bool(study.keep_case_animation)
    data["output"]["show_final_plot"] = False

    return CaseConfig.from_dict(data)


def _case_metrics(
    grid,
    case: CaseConfig,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Collect scalar quantities used by parameter and convergence studies."""
    flow = flow_rate_metrics(grid)
    max_div, rms_div = divergence_metrics(grid)
    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)

    row: dict[str, object] = {
        "nx": grid.nx,
        "ny": grid.ny,
        "total_cells": grid.nx * grid.ny,
        "dx_m": grid.dx,
        "dy_m": grid.dy,
        "h_m": max(grid.dx, grid.dy),
        "reynolds_number": case.reynolds_number,
        "mean_inlet_velocity_m_s": case.inlet.mean_velocity,
        "dynamic_viscosity_Pa_s": case.fluid.dynamic_viscosity,
        "runtime_s": elapsed_seconds,
        "inlet_flow_rate_m2_s": flow["inlet"],
        "outlet_flow_rate_m2_s": flow["outlet"],
        "flow_rate_difference_m2_s": flow["difference"],
        "final_max_divergence_1_s": max_div,
        "final_rms_divergence_1_s": rms_div,
        "max_speed_m_s": float(np.nanmax(speed)),
        "mean_speed_m_s": float(np.nanmean(speed)),
    }

    if case.validation.enabled:
        y, numerical = numerical_velocity_profile(grid, case.validation.x_location)
        analytical = analytical_poiseuille_profile(
            y,
            grid.height,
            case.inlet.mean_velocity,
        )
        errors = profile_error(numerical, analytical)
        row.update(
            {
                "validation_rmse_m_s": float(errors["rmse"]),
                "validation_max_error_m_s": float(errors["max_error"]),
                "validation_relative_rmse": float(errors["relative_rmse"]),
                "validation_relative_rmse_percent": 100.0 * float(errors["relative_rmse"]),
            }
        )

    if case.engineering.enabled:
        analytical_gradient = None
        if case.inlet.profile == "poiseuille" and not case.obstacles:
            analytical_gradient = analytical_poiseuille_pressure_gradient(
                dynamic_viscosity=case.fluid.dynamic_viscosity,
                mean_velocity=case.inlet.mean_velocity,
                channel_height=grid.height,
            )

        engineering = compute_engineering_results(
            grid,
            density=case.fluid.density,
            reference_velocity=case.inlet.mean_velocity,
            upstream_x=case.engineering.upstream_x,
            downstream_x=case.engineering.downstream_x,
            sample_width=case.engineering.sample_width,
            reference_pressure=case.engineering.reference_pressure,
            analytical_pressure_gradient=analytical_gradient,
        )
        row.update(engineering.to_dict())
        if engineering.pressure_drop_relative_error is not None:
            row["pressure_drop_relative_error_percent"] = (
                100.0 * engineering.pressure_drop_relative_error
            )

    return row


def _observed_order(
    previous_error: object,
    current_error: object,
    previous_h: object,
    current_h: object,
) -> float | None:
    try:
        e1 = float(previous_error)
        e2 = float(current_error)
        h1 = float(previous_h)
        h2 = float(current_h)
    except (TypeError, ValueError):
        return None

    if e1 <= 0.0 or e2 <= 0.0 or h1 <= h2 or h2 <= 0.0:
        return None
    return log(e1 / e2) / log(h1 / h2)


def _add_grid_convergence_orders(rows: list[dict[str, object]]) -> None:
    """Add pairwise observed orders to rows supplied coarse-to-fine."""
    for row in rows:
        row["velocity_observed_order"] = None
        row["pressure_drop_observed_order"] = None

    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        current["velocity_observed_order"] = _observed_order(
            previous.get("validation_relative_rmse"),
            current.get("validation_relative_rmse"),
            previous.get("h_m"),
            current.get("h_m"),
        )
        current["pressure_drop_observed_order"] = _observed_order(
            previous.get("pressure_drop_relative_error"),
            current.get("pressure_drop_relative_error"),
            previous.get("h_m"),
            current.get("h_m"),
        )


def _three_grid_gci(
    rows: list[dict[str, object]],
    metric: str,
    *,
    safety_factor: float = 1.25,
) -> dict[str, object] | None:
    """Return three-grid Richardson/GCI metrics for one scalar solution quantity.

    The three finest grids are used and must form an approximately constant
    refinement ratio. Values must converge monotonically for Richardson
    extrapolation and GCI to be reported.
    """
    if len(rows) < 3:
        return None

    ordered = sorted(rows, key=lambda row: float(row["h_m"]))[:3]
    fine, medium, coarse = ordered

    try:
        phi1 = float(fine[metric])
        phi2 = float(medium[metric])
        phi3 = float(coarse[metric])
        h1 = float(fine["h_m"])
        h2 = float(medium["h_m"])
        h3 = float(coarse["h_m"])
    except (KeyError, TypeError, ValueError):
        return None

    r21 = h2 / h1
    r32 = h3 / h2
    if r21 <= 1.0 or r32 <= 1.0:
        return None
    if not np.isclose(r21, r32, rtol=0.05):
        return None

    eps21 = phi2 - phi1
    eps32 = phi3 - phi2
    if eps21 == 0.0 or eps32 == 0.0 or eps21 * eps32 <= 0.0:
        return None

    p = log(abs(eps32 / eps21)) / log(r21)
    denominator = r21**p - 1.0
    if p <= 0.0 or denominator <= 0.0 or phi1 == 0.0 or phi2 == 0.0:
        return None

    extrapolated = phi1 + (phi1 - phi2) / denominator
    apparent_error_fine = abs((phi1 - phi2) / phi1)
    apparent_error_medium = abs((phi2 - phi3) / phi2)
    gci_fine = safety_factor * apparent_error_fine / denominator
    gci_medium = safety_factor * apparent_error_medium / denominator
    asymptotic_ratio = (
        gci_medium / (gci_fine * r21**p)
        if gci_fine > 0.0
        else None
    )

    return {
        "metric": metric,
        "fine_grid": fine["label"],
        "medium_grid": medium["label"],
        "coarse_grid": coarse["label"],
        "refinement_ratio": r21,
        "observed_order": p,
        "fine_value": phi1,
        "richardson_extrapolated": extrapolated,
        "gci_fine_percent": 100.0 * gci_fine,
        "gci_medium_percent": 100.0 * gci_medium,
        "asymptotic_ratio": asymptotic_ratio,
    }


def _grid_convergence_assessment(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Calculate GCI/Richardson results for the main validation quantities."""
    assessments = []
    for metric in ("pressure_drop", "max_speed_m_s"):
        result = _three_grid_gci(rows, metric)
        if result is not None:
            assessments.append(result)
    return assessments


def _study_report(
    study: StudyConfig,
    rows: list[dict[str, object]],
    convergence: list[dict[str, object]] | None = None,
) -> str:
    report = TextReport(
        "CFD PARAMETER STUDY REPORT",
        subtitle=f"Study: {study.name} ({study.kind})",
    )
    report.add_section("Study setup")
    report.add_fields(
        [
            ("Base case", study.base_case),
            ("Number of runs", len(rows)),
            ("Output directory", study.output_directory),
            ("Case plots retained", "YES" if study.keep_case_plots else "NO"),
            ("Case animations retained", "YES" if study.keep_case_animation else "NO"),
        ]
    )

    report.add_section("Run summary")
    table_rows = []
    for row in rows:
        table_rows.append(
            (
                row["label"],
                f"{int(float(row['nx']))}x{int(float(row['ny']))}",
                f"{float(row['h_m']):.4e}",
                f"{float(row['runtime_s']):.2f}",
                (
                    f"{float(row['validation_relative_rmse_percent']):.4f}"
                    if row.get("validation_relative_rmse_percent") is not None
                    else "-"
                ),
                (
                    f"{float(row['pressure_drop_relative_error_percent']):.4f}"
                    if row.get("pressure_drop_relative_error_percent") is not None
                    else "-"
                ),
                (
                    f"{float(row['velocity_observed_order']):.3f}"
                    if row.get("velocity_observed_order") is not None
                    else "-"
                ),
            )
        )

    report.add_table(
        (
            "Run",
            "Grid",
            "h (m)",
            "Runtime (s)",
            "Vel. RMSE (%)",
            "dP error (%)",
            "Observed p",
        ),
        table_rows,
    )

    if convergence:
        report.add_section("Richardson extrapolation / Grid Convergence Index")
        report.add_table(
            (
                "Metric",
                "p",
                "Fine value",
                "Extrapolated",
                "GCI fine (%)",
                "Asymptotic ratio",
            ),
            [
                (
                    item["metric"],
                    f"{float(item['observed_order']):.3f}",
                    f"{float(item['fine_value']):.6g}",
                    f"{float(item['richardson_extrapolated']):.6g}",
                    f"{float(item['gci_fine_percent']):.4f}",
                    (
                        f"{float(item['asymptotic_ratio']):.4f}"
                        if item.get("asymptotic_ratio") is not None
                        else "-"
                    ),
                )
                for item in convergence
            ],
        )
        report.add_lines(
            [
                "An asymptotic ratio close to 1 indicates the three grids are in the",
                "asymptotic convergence range for the reported scalar quantity.",
            ]
        )

    report.add_section("Generated files")
    report.add_lines(["- summary.csv", "- study_report.txt"])
    if convergence:
        report.add_lines(["- convergence_summary.csv"])
    for plot in study.plots:
        report.add_lines([f"- {plot.filename}"])
    report.add_lines(["- runs/<label>/..."])
    return report.render()


def _coerce_csv_value(value: str) -> object:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_summary_csv(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def _write_study_outputs(
    study: StudyConfig,
    rows: list[dict[str, object]],
    output_directory: Path,
) -> None:
    convergence = None
    if study.kind == "grid_convergence":
        _add_grid_convergence_orders(rows)
        convergence = _grid_convergence_assessment(rows)

    write_table_csv(rows, output_directory / "summary.csv")

    if convergence:
        write_table_csv(convergence, output_directory / "convergence_summary.csv")

    for plot in study.plots:
        plot_study_metric(rows, plot, output_directory)

    write_to_output(
        _study_report(study, rows, convergence),
        output_directory / "study_report.txt",
    )


def run_study(study_path: str | Path) -> list[dict[str, object]]:
    """Execute all cases in a configured parameter study."""
    study_path = Path(study_path).resolve()
    study = load_study(study_path)

    base_case_path = Path(study.base_case)
    if not base_case_path.is_absolute():
        base_case_path = (study_path.parent / base_case_path).resolve()
    base_case = load_case(base_case_path)

    output_directory = Path(study.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    progress = tqdm(study.runs, desc=f"Study: {study.name}", unit="case")
    for run in progress:
        progress.set_postfix_str(run.label)
        case = build_study_case(base_case, study, run, output_directory)

        start = perf_counter()
        grid = run_case(case)
        elapsed = perf_counter() - start

        row = {"label": run.label}
        row.update(_case_metrics(grid, case, elapsed_seconds=elapsed))
        rows.append(row)

    _write_study_outputs(study, rows, output_directory)
    return rows


def reprocess_study(study_path: str | Path) -> list[dict[str, object]]:
    """Regenerate study analysis from an existing summary.csv without rerunning CFD."""
    study_path = Path(study_path).resolve()
    study = load_study(study_path)
    output_directory = Path(study.output_directory)
    summary_path = output_directory / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot reuse results because {summary_path} does not exist."
        )

    rows = _read_summary_csv(summary_path)
    _write_study_outputs(study, rows, output_directory)
    return rows
