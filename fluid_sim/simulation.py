from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

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
from fluid_sim.io.output import TextReport, write_to_output
from fluid_sim.numerics.advection import advect_velocity
from fluid_sim.numerics.boundary_conditions import apply_boundary_conditions
from fluid_sim.numerics.diffusion import (
    diffuse_velocity,
    diffusion_timestep_limit,
)
from fluid_sim.numerics.solver import (
    divergence_metrics,
    project_velocity,
    solve_pressure,
)


def build_grid(case: CaseConfig) -> MACGrid:
    """Create the MAC grid and apply configured solid geometry."""
    domain = case.domain
    grid = MACGrid(
        nx=domain.nx,
        ny=domain.ny,
        width=domain.width,
        height=domain.height,
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
        raise ValueError(
            f"Unsupported inlet profile: {case.inlet.profile}"
        )

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


def _build_report(
    grid: MACGrid,
    case: CaseConfig,
    divergence_history: list[tuple[int, float, float]],
    elapsed_seconds: float,
    validation_errors: dict[str, float] | None,
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

    report.add_section("Domain and grid")
    fluid_cells = int(np.count_nonzero(~grid.solid))
    solid_cells = int(np.count_nonzero(grid.solid))
    total_cells = grid.nx * grid.ny
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
            (
                "Reynolds number",
                f"{case.reynolds_number:.3f}" if mu > 0.0 else "inf",
            ),
        ]
    )

    report.add_section("Geometry")
    if case.obstacles:
        rows = []
        for index, obstacle in enumerate(case.obstacles, start=1):
            rows.append(
                (
                    index,
                    f"{obstacle.x_min:.4f}",
                    f"{obstacle.x_max:.4f}",
                    f"{obstacle.y_min:.4f}",
                    f"{obstacle.y_max:.4f}",
                )
            )
        report.add_table(
            ("Obstacle", "x_min", "x_max", "y_min", "y_max"),
            rows,
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
            (
                "dt / diffusion limit",
                f"{dt / dt_limit:.6f}" if np.isfinite(dt_limit) else "0.000000",
            ),
            ("Diffusion number x", f"{diffusion_number_x:.6f}"),
            ("Diffusion number y", f"{diffusion_number_y:.6f}"),
            ("Combined diffusion number", f"{combined_diffusion:.6f}"),
            (
                "Explicit diffusion stable",
                "YES" if dt <= dt_limit else "NO",
            ),
        ]
    )

    flow = flow_rate_metrics(grid)
    report.add_section("Mass conservation")
    report.add_fields(
        [
            ("Inlet flow rate", f"{flow['inlet']:.9f}", "m^2/s"),
            ("Outlet flow rate", f"{flow['outlet']:.9f}", "m^2/s"),
            ("Flow-rate difference", f"{flow['difference']:.6e}", "m^2/s"),
            (
                "Relative flow imbalance",
                f"{100.0 * abs(flow['relative_difference']):.6e}",
                "%",
            ),
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
            (
                "Pressure range",
                f"{np.ptp(fluid_pressure):.6f}",
                "Pa",
            ),
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

    if validation_errors is not None:
        y, u_numerical = numerical_velocity_profile(
            grid,
            case.validation.x_location,
        )
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
                (
                    "Relative RMSE",
                    f"{100.0 * validation_errors['relative_rmse']:.6f}",
                    "%",
                ),
            ]
        )
    else:
        report.add_section("Validation")
        report.add_lines(["No analytical validation configured for this case."])

    report.add_section("Generated files")
    generated_files = ["output.txt", "resolved_case.json"]
    if case.output.save_plots:
        generated_files.extend(
            [
                "velocity_magnitude.png",
                "velocity_vectors.png",
                "pressure.png",
            ]
        )
    if case.validation.enabled:
        generated_files.append("poiseuille_validation.png")
    if case.output.save_animation:
        generated_files.append("flow_animation.gif")

    report.add_lines([f"- {filename}" for filename in generated_files])
    return report.render()


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

    for step in tqdm(
        range(case.time.steps),
        desc=f"Simulation: {case.name}",
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

        _, rms_before = divergence_metrics(grid)
        solve_pressure(grid, dt, rho=rho)
        project_velocity(grid, dt, rho=rho)
        _, rms_after = divergence_metrics(grid)

        if case.output.save_animation and (
            step % case.output.frame_interval == 0
            or step == case.time.steps - 1
        ):
            flow_states.append(
                capture_flow_state(
                    grid,
                    time=(step + 1) * dt,
                )
            )

        if (
            (step + 1) % case.output.diagnostics_interval == 0
            or step == case.time.steps - 1
        ):
            divergence_history.append(
                (step + 1, rms_before, rms_after)
            )

    elapsed_seconds = perf_counter() - start_time

    validation_errors = None
    if case.validation.enabled:
        validation_errors, _ = validate_poiseuille(
            grid,
            mean_velocity=case.inlet.mean_velocity,
            x_location=case.validation.x_location,
            filename=str(
                output_directory / "poiseuille_validation.png"
            ),
        )

    if case.output.save_plots:
        plot_velocity_magnitude(
            grid,
            output_directory / "velocity_magnitude.png",
        )
        plot_velocity_vectors(
            grid,
            output_directory / "velocity_vectors.png",
        )
        plot_pressure(
            grid,
            output_directory / "pressure.png",
        )

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
    )
    write_to_output(
        report,
        file_path=output_directory / "output.txt",
    )

    if case.output.show_final_plot:
        reconstruct_cell_centred_velocities(grid)

    return grid
