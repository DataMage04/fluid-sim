from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from fluid_sim.analysis.engineering import (
    EngineeringResults,
    analytical_poiseuille_pressure_gradient,
    compute_engineering_results,
)
from fluid_sim.analysis.validation import (
    analytical_poiseuille_profile,
    numerical_velocity_profile,
    validate_poiseuille,
)
from fluid_sim.analysis.visualization import (
    animate_flow,
    capture_flow_state,
    flow_rate_metrics,
    get_cell_center_velocity,
    plot_pressure,
    plot_velocity_magnitude,
    plot_velocity_vectors,
    reconstruct_cell_centred_velocities,
)
from fluid_sim.config import CaseConfig, save_case
from fluid_sim.core.grid import MACGrid
from fluid_sim.io.output import TextReport, write_results_csv, write_to_output
from fluid_sim.numerics.advection import advect_velocity
from fluid_sim.numerics.boundary_conditions import apply_boundary_conditions
from fluid_sim.numerics.diffusion import diffuse_velocity, diffusion_timestep_limit
from fluid_sim.numerics.solver import divergence_metrics, project_velocity, solve_pressure


def build_grid(case: CaseConfig) -> MACGrid:
    """Create the MAC grid and apply configured solid geometry."""
    grid = MACGrid(
        nx=case.domain.nx,
        ny=case.domain.ny,
        width=case.domain.width,
        height=case.domain.height,
    )

    for obstacle in case.obstacles:
        grid.add_rectangle(
            x_min=obstacle.x_min,
            x_max=obstacle.x_max,
            y_min=obstacle.y_min,
            y_max=obstacle.y_max,
        )

    return grid


def initialize_velocity(grid: MACGrid, case: CaseConfig) -> None:
    """Initialise velocity using the configured inlet profile."""
    velocity = case.inlet.mean_velocity

    if case.inlet.profile == "poiseuille":
        y = (np.arange(grid.ny) + 0.5) * grid.dy
        eta = y / grid.height
        profile = 6.0 * velocity * eta * (1.0 - eta)
        grid.u[:, :] = profile[:, None]
    elif case.inlet.profile == "uniform":
        grid.u[:, :] = velocity
    else:
        raise ValueError(f"Unsupported inlet profile: {case.inlet.profile}")

    grid.v[:, :] = 0.0


def _output_directory(case: CaseConfig) -> Path:
    directory = Path(case.output.directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_case_for_run(case: CaseConfig) -> None:
    case.validate()

    if case.validation.enabled:
        if case.inlet.profile != "poiseuille":
            raise ValueError(
                "Poiseuille validation requires inlet.profile='poiseuille'."
            )
        if case.obstacles:
            raise ValueError(
                "Poiseuille validation requires a channel without obstacles."
            )

    if case.engineering.enabled and case.inlet.mean_velocity <= 0.0:
        raise ValueError(
            "Engineering coefficients require a positive reference inlet velocity."
        )


def _compute_case_engineering_results(
    grid: MACGrid,
    case: CaseConfig,
) -> EngineeringResults | None:
    if not case.engineering.enabled:
        return None

    analytical_gradient = None
    if case.inlet.profile == "poiseuille" and not case.obstacles:
        analytical_gradient = analytical_poiseuille_pressure_gradient(
            dynamic_viscosity=case.fluid.dynamic_viscosity,
            mean_velocity=case.inlet.mean_velocity,
            channel_height=grid.height,
        )

    return compute_engineering_results(
        grid,
        density=case.fluid.density,
        reference_velocity=case.inlet.mean_velocity,
        upstream_x=case.engineering.upstream_x,
        downstream_x=case.engineering.downstream_x,
        sample_width=case.engineering.sample_width,
        reference_pressure=case.engineering.reference_pressure,
        analytical_pressure_gradient=analytical_gradient,
    )


def _build_report(
    grid: MACGrid,
    case: CaseConfig,
    divergence_history: list[tuple[int, float, float]],
    elapsed_seconds: float,
    validation_errors: dict[str, float] | None,
    engineering_results: EngineeringResults | None,
) -> str:
    """Create the final human-readable engineering report."""
    rho = case.fluid.density
    mu = case.fluid.dynamic_viscosity
    nu = case.fluid.kinematic_viscosity
    dt = case.time.dt

    report = TextReport(
        "2D INCOMPRESSIBLE FLOW SIMULATION REPORT",
        subtitle=f"Case: {case.name}",
    )

    report.add_section("Run summary")
    report.add_fields(
        [
            ("Status", "COMPLETED"),
            ("Time steps completed", case.time.steps),
            ("Physical time simulated", f"{case.time.steps * dt:.6f}", "s"),
            ("Wall-clock runtime", f"{elapsed_seconds:.3f}", "s"),
            ("Output directory", case.output.directory),
        ]
    )

    fluid_cells = int(np.count_nonzero(~grid.solid))
    solid_cells = int(np.count_nonzero(grid.solid))
    total_cells = grid.nx * grid.ny
    report.add_section("Domain and grid")
    report.add_fields(
        [
            ("Domain width", f"{grid.width:.6g}", "m"),
            ("Domain height", f"{grid.height:.6g}", "m"),
            ("Grid cells (nx x ny)", f"{grid.nx} x {grid.ny}"),
            ("Cell spacing dx", f"{grid.dx:.6e}", "m"),
            ("Cell spacing dy", f"{grid.dy:.6e}", "m"),
            ("Fluid cells", fluid_cells),
            ("Solid cells", solid_cells),
            ("Solid-cell fraction", f"{100.0 * solid_cells / total_cells:.3f}", "%"),
        ]
    )

    report.add_section("Fluid and boundary conditions")
    report.add_fields(
        [
            ("Density", f"{rho:.6g}", "kg/m^3"),
            ("Dynamic viscosity", f"{mu:.6g}", "Pa.s"),
            ("Kinematic viscosity", f"{nu:.6g}", "m^2/s"),
            ("Inlet profile", case.inlet.profile),
            ("Mean inlet velocity", f"{case.inlet.mean_velocity:.6g}", "m/s"),
            ("Characteristic length", f"{case.solver.characteristic_length:.6g}", "m"),
            ("Reynolds number", f"{case.reynolds_number:.3f}" if mu > 0.0 else "inf"),
        ]
    )

    report.add_section("Geometry")
    if case.obstacles:
        report.add_table(
            ("Obstacle", "x_min", "x_max", "y_min", "y_max"),
            [
                (
                    index,
                    f"{obstacle.x_min:.4f}",
                    f"{obstacle.x_max:.4f}",
                    f"{obstacle.y_min:.4f}",
                    f"{obstacle.y_max:.4f}",
                )
                for index, obstacle in enumerate(case.obstacles, start=1)
            ],
        )
        report.add_lines(["Coordinates are in metres."])
    else:
        report.add_lines(["No internal solid obstacles."])

    dt_limit = diffusion_timestep_limit(grid, nu)
    diffusion_number_x = nu * dt / grid.dx**2
    diffusion_number_y = nu * dt / grid.dy**2
    combined_diffusion = diffusion_number_x + diffusion_number_y

    report.add_section("Time integration and stability")
    report.add_fields(
        [
            ("Time step dt", f"{dt:.6e}", "s"),
            ("Configured steps", case.time.steps),
            ("Diffusion dt limit", f"{dt_limit:.6e}", "s"),
            ("dt / diffusion limit", f"{dt / dt_limit:.6f}" if np.isfinite(dt_limit) else "0.000000"),
            ("Diffusion number x", f"{diffusion_number_x:.6f}"),
            ("Diffusion number y", f"{diffusion_number_y:.6f}"),
            ("Combined diffusion number", f"{combined_diffusion:.6f}"),
            ("Explicit diffusion stable", "YES" if dt <= dt_limit else "NO"),
        ]
    )

    flow = flow_rate_metrics(grid)
    report.add_section("Mass conservation")
    report.add_fields(
        [
            ("Inlet flow rate", f"{flow['inlet']:.9f}", "m^2/s"),
            ("Outlet flow rate", f"{flow['outlet']:.9f}", "m^2/s"),
            ("Flow-rate difference", f"{flow['difference']:.6e}", "m^2/s"),
            ("Relative flow imbalance", f"{100.0 * abs(flow['relative_difference']):.6e}", "%"),
        ]
    )

    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)
    fluid_speed = speed[~grid.solid]
    fluid_pressure = grid.p[~grid.solid]

    report.add_section("Final flow field")
    report.add_fields(
        [
            ("Maximum speed", f"{np.nanmax(fluid_speed):.6f}", "m/s"),
            ("Mean speed", f"{np.nanmean(fluid_speed):.6f}", "m/s"),
            ("Minimum pressure", f"{np.min(fluid_pressure):.6f}", "Pa"),
            ("Maximum pressure", f"{np.max(fluid_pressure):.6f}", "Pa"),
            ("Pressure range", f"{np.ptp(fluid_pressure):.6f}", "Pa"),
        ]
    )

    final_max_div, final_rms_div = divergence_metrics(grid)
    report.add_section("Pressure projection / continuity")
    report.add_fields(
        [
            ("Final max |divergence|", f"{final_max_div:.6e}", "1/s"),
            ("Final RMS divergence", f"{final_rms_div:.6e}", "1/s"),
        ]
    )
    report.add_table(
        ("Step", "RMS before", "RMS after", "Reduction"),
        [
            (
                step,
                f"{before:.3e}",
                f"{after:.3e}",
                f"{before / after:.3e}" if after > 0.0 else "inf",
            )
            for step, before, after in divergence_history
        ],
    )

    if engineering_results is not None:
        eng = engineering_results
        report.add_section("Engineering results")
        report.add_fields(
            [
                ("Upstream sample x", f"{eng.upstream_x:.6f}", "m"),
                ("Downstream sample x", f"{eng.downstream_x:.6f}", "m"),
                ("Pressure sample width", f"{eng.sample_width:.6f}", "m"),
                ("Mean upstream pressure", f"{eng.upstream_pressure:.6f}", "Pa"),
                ("Mean downstream pressure", f"{eng.downstream_pressure:.6f}", "Pa"),
                ("Pressure drop", f"{eng.pressure_drop:.6f}", "Pa"),
                ("Dynamic pressure", f"{eng.dynamic_pressure:.6f}", "Pa"),
                ("Upstream Cp", f"{eng.upstream_cp:.6f}"),
                ("Downstream Cp", f"{eng.downstream_cp:.6f}"),
                ("Pressure-drop coefficient", f"{eng.pressure_drop_coefficient:.6f}"),
                ("Maximum speed", f"{eng.max_speed:.6f}", "m/s"),
                ("Maximum-speed x", f"{eng.max_speed_x:.6f}", "m"),
                ("Maximum-speed y", f"{eng.max_speed_y:.6f}", "m"),
            ]
        )
        if eng.analytical_pressure_drop is not None:
            report.add_fields(
                [
                    ("Analytical -dp/dx", f"{eng.analytical_pressure_gradient:.6f}", "Pa/m"),
                    ("Analytical pressure drop", f"{eng.analytical_pressure_drop:.6f}", "Pa"),
                    ("Pressure-drop relative error", f"{100.0 * eng.pressure_drop_relative_error:.6f}", "%"),
                ]
            )
    else:
        report.add_section("Engineering results")
        report.add_lines(["Engineering post-processing disabled for this case."])

    if validation_errors is not None:
        y, u_numerical = numerical_velocity_profile(grid, case.validation.x_location)
        u_analytical = analytical_poiseuille_profile(
            y,
            grid.height,
            case.inlet.mean_velocity,
        )
        report.add_section("Poiseuille validation")
        report.add_fields(
            [
                ("Profile location", f"{case.validation.x_location:.6g}", "m"),
                ("Numerical max velocity", f"{np.max(u_numerical):.6f}", "m/s"),
                ("Analytical max velocity", f"{np.max(u_analytical):.6f}", "m/s"),
                ("RMSE", f"{validation_errors['rmse']:.6e}", "m/s"),
                ("Maximum error", f"{validation_errors['max_error']:.6e}", "m/s"),
                ("Relative RMSE", f"{100.0 * validation_errors['relative_rmse']:.6f}", "%"),
            ]
        )
    else:
        report.add_section("Validation")
        report.add_lines(["No analytical velocity-profile validation configured for this case."])

    report.add_section("Generated files")
    generated_files = ["output.txt", "resolved_case.json"]
    if engineering_results is not None:
        generated_files.append("results.csv")
    if case.output.save_plots:
        generated_files.extend(["velocity_magnitude.png", "velocity_vectors.png", "pressure.png"])
    if case.validation.enabled:
        generated_files.append("poiseuille_validation.png")
    if case.output.save_animation:
        generated_files.append("flow_animation.gif")
    report.add_lines([f"- {filename}" for filename in generated_files])

    return report.render()


def _csv_results(
    grid: MACGrid,
    case: CaseConfig,
    engineering_results: EngineeringResults,
) -> dict[str, object]:
    flow = flow_rate_metrics(grid)
    max_div, rms_div = divergence_metrics(grid)
    data: dict[str, object] = {
        "case": case.name,
        "nx": grid.nx,
        "ny": grid.ny,
        "density_kg_m3": case.fluid.density,
        "dynamic_viscosity_Pa_s": case.fluid.dynamic_viscosity,
        "mean_inlet_velocity_m_s": case.inlet.mean_velocity,
        "reynolds_number": case.reynolds_number,
        "inlet_flow_rate_m2_s": flow["inlet"],
        "outlet_flow_rate_m2_s": flow["outlet"],
        "flow_rate_difference_m2_s": flow["difference"],
        "final_max_divergence_1_s": max_div,
        "final_rms_divergence_1_s": rms_div,
    }
    data.update(engineering_results.to_dict())
    return data


def run_case(case: CaseConfig) -> MACGrid:
    """Run one configured CFD case and return the final grid."""
    _validate_case_for_run(case)

    output_directory = _output_directory(case)
    save_case(case, output_directory / "resolved_case.json")

    grid = build_grid(case)
    initialize_velocity(grid, case)

    rho = case.fluid.density
    nu = case.fluid.kinematic_viscosity
    dt = case.time.dt
    flow_states = []
    divergence_history: list[tuple[int, float, float]] = []

    print(f"Running case: {case.name}")
    print(
        f"Grid: {grid.nx} x {grid.ny}, "
        f"domain: {grid.width:g} x {grid.height:g} m"
    )

    start_time = perf_counter()

    for step in tqdm(range(case.time.steps), desc=f"Simulation: {case.name}"):
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

        _, rms_before = divergence_metrics(grid)
        solve_pressure(grid, dt, rho=rho)
        project_velocity(grid, dt, rho=rho)
        _, rms_after = divergence_metrics(grid)

        if case.output.save_animation and (
            step % case.output.frame_interval == 0
            or step == case.time.steps - 1
        ):
            flow_states.append(
                capture_flow_state(grid, time=(step + 1) * dt)
            )

        if (
            (step + 1) % case.output.diagnostics_interval == 0
            or step == case.time.steps - 1
        ):
            divergence_history.append((step + 1, rms_before, rms_after))

    elapsed_seconds = perf_counter() - start_time

    validation_errors = None
    if case.validation.enabled:
        validation_errors, _ = validate_poiseuille(
            grid,
            mean_velocity=case.inlet.mean_velocity,
            x_location=case.validation.x_location,
            filename=str(output_directory / "poiseuille_validation.png"),
        )

    engineering_results = _compute_case_engineering_results(grid, case)
    if engineering_results is not None:
        write_results_csv(
            _csv_results(grid, case, engineering_results),
            output_directory / "results.csv",
        )

    if case.output.save_plots:
        plot_velocity_magnitude(grid, output_directory / "velocity_magnitude.png")
        plot_velocity_vectors(grid, output_directory / "velocity_vectors.png")
        plot_pressure(grid, output_directory / "pressure.png")

    if case.output.save_animation:
        animate_flow(
            grid,
            flow_states,
            filename=output_directory / "flow_animation.gif",
            fps=case.output.animation_fps,
        )

    report = _build_report(
        grid=grid,
        case=case,
        divergence_history=divergence_history,
        elapsed_seconds=elapsed_seconds,
        validation_errors=validation_errors,
        engineering_results=engineering_results,
    )
    write_to_output(report, file_path=output_directory / "output.txt")

    if case.output.show_final_plot:
        reconstruct_cell_centred_velocities(grid)

    return grid
