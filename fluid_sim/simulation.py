from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from fluid_sim.analysis.validation import validate_poiseuille
from fluid_sim.analysis.visualization import (
    animate_flow,
    capture_flow_state,
    get_cell_center_velocity,
    measure_flow_rates,
    plot_pressure,
    plot_velocity_magnitude,
    plot_velocity_vectors,
    reconstuct_cell_centred_velocities,
)
from fluid_sim.config import CaseConfig
from fluid_sim.core.grid import MACGrid
from fluid_sim.io.output import write_to_output
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


def _visualization_filename(output_directory: Path, filename: str) -> str:
    """
    visualization.py currently writes relative to results/.
    Return the filename relative to that root.
    """
    try:
        relative = output_directory.relative_to("results")
    except ValueError as exc:
        raise ValueError(
            "Milestone 8 output.directory must be inside the results/ directory."
        ) from exc

    return str(relative / filename)


def _append_diagnostics(
    data: list[str],
    grid: MACGrid,
    case: CaseConfig,
) -> None:
    rho = case.fluid.density
    mu = case.fluid.dynamic_viscosity
    nu = case.fluid.kinematic_viscosity
    dt = case.time.dt

    data.extend(measure_flow_rates(grid))

    dt_limit = diffusion_timestep_limit(grid, nu)
    data.append(f"Diffusion dt limit:   {dt_limit:.6e} s")
    data.append(f"Current dt:            {dt:.6e} s")

    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)
    fluid_speed = speed[~grid.solid]

    data.append(
        f"Maximum fluid speed: {np.nanmax(fluid_speed):.6f} m/s"
    )
    data.append(
        f"Mean fluid speed: {np.nanmean(fluid_speed):.6f} m/s"
    )

    data.append(f"Density:              {rho:.6g} kg/m^3")
    data.append(f"Dynamic viscosity:    {mu:.6g} Pa.s")
    data.append(f"Kinematic viscosity:  {nu:.6g} m^2/s")

    if mu > 0.0:
        reynolds = case.reynolds_number
        data.append(f"Reynolds number:       {reynolds:.2f}")
    else:
        data.append("Reynolds number:       inf")

    diffusion_number_x = nu * dt / grid.dx**2
    diffusion_number_y = nu * dt / grid.dy**2

    data.append(f"Diffusion number x:    {diffusion_number_x:.4f}")
    data.append(f"Diffusion number y:    {diffusion_number_y:.4f}")
    data.append(
        "Combined diffusion #: "
        f"{diffusion_number_x + diffusion_number_y:.4f}"
    )


def run_case(case: CaseConfig) -> MACGrid:
    """Run one configured CFD case and return the final grid."""
    case.validate()

    output_directory = _output_directory(case)
    grid = build_grid(case)
    initialize_velocity(grid, case)

    rho = case.fluid.density
    nu = case.fluid.kinematic_viscosity
    dt = case.time.dt

    flow_states = []
    rms_on_steps: dict[str, str] = {}

    print(f"Running case: {case.name}")
    print(
        f"Grid: {grid.nx} x {grid.ny}, "
        f"domain: {grid.width:g} x {grid.height:g} m"
    )

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

        if (step + 1) % case.output.diagnostics_interval == 0:
            key = f"Step {step + 1}/{case.time.steps}"
            rms_on_steps[key] = (
                f"{rms_before:.3e} -> {rms_after:.3e}"
            )

    data = [
        f"Case: {case.name}",
        "RMS Divergence",
    ]
    for step_name, values in rms_on_steps.items():
        data.append(
            f"{step_name}: RMS divergence {values}"
        )

    _append_diagnostics(data, grid, case)

    if case.validation.enabled:
        if case.inlet.profile != "poiseuille":
            raise ValueError(
                "Poiseuille validation requires inlet.profile='poiseuille'."
            )
        if case.obstacles:
            raise ValueError(
                "Poiseuille validation requires a channel without obstacles."
            )

        errors, report = validate_poiseuille(
            grid,
            mean_velocity=case.inlet.mean_velocity,
            x_location=case.validation.x_location,
            filename=str(
                output_directory / "poiseuille_validation.png"
            ),
        )
        data.extend(report)
        data.append("=======Errors=======")
        for name, value in errors.items():
            data.append(f"{name}: {value}")

    if case.output.save_plots:
        velocity_magnitude = _visualization_filename(
            output_directory,
            "velocity_magnitude.png",
        )
        velocity_vectors = _visualization_filename(
            output_directory,
            "velocity_vectors.png",
        )
        pressure = _visualization_filename(
            output_directory,
            "pressure.png",
        )

        plot_velocity_magnitude(grid, velocity_magnitude)
        plot_velocity_vectors(grid, velocity_vectors)
        plot_pressure(grid, pressure)

    if case.output.save_animation:
        animate_flow(
            grid,
            flow_states,
            filename=_visualization_filename(
                output_directory,
                "flow_animation.gif",
            ),
            fps=case.output.animation_fps,
        )

    write_to_output(
        data,
        file_path=output_directory / "output.txt",
    )

    if case.output.show_final_plot:
        reconstuct_cell_centred_velocities(grid)

    return grid
