from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from fluid_sim.core.grid import MACGrid
from fluid_sim.numerics.advection import advect_velocity
from fluid_sim.analysis.visualization import (
    save_flow_plots,
    get_cell_center_velocity,
    measure_flow_rates,
    capture_flow_state,
    animate_flow,
    reconstuct_cell_centred_velocities,
)
from fluid_sim.numerics.boundary_conditions import apply_boundary_conditions
from fluid_sim.numerics.solver import (
    compute_divergence,
    solve_pressure,
    project_velocity,
    divergence_metrics,
)
from fluid_sim.numerics.diffusion import (
    diffuse_velocity,
    diffusion_timestep_limit,
)
from fluid_sim.analysis.validation import validate_poiseuille
from fluid_sim.io.output import write_to_output

data = []
def main():
    # --------------------------------------------
    # 1. Create the simulation domain
    # --------------------------------------------
    print("Defining grid")
    grid = MACGrid(
        nx=60,
        ny=30,
        width=2.0,
        height=1.0,
    )

    # Rectangular obstruction
    # grid.add_rectangle(
    #     x_min=0.8,
    #     x_max=1.0,
    #     y_min=0.3,
    #     y_max=0.7,
    # )

    # Simulation parameters
    rho = 1.0                 # kg/m^3
    mu = 0.01                 # Pa.s

    nu = mu / rho             # m^2/s
    frame_interval = 5
    flow_states = []
    dt = 0.005
    inlet_velocity = 1.0
    steps = 1000

    # Initialise the velocity field
    grid.u[:] = inlet_velocity
    grid.v[:] = 0.0

    y_u = (
        np.arange(grid.ny) + 0.5
    ) * grid.dy

    eta = y_u / grid.height

    profile = (
        6.0
        * inlet_velocity
        * eta
        * (1.0 - eta)
    )

    for i in range(grid.nx + 1):
        grid.u[:, i] = profile

    grid.v[:, :] = 0.0

    # --------------------------------------------
    # 2. Advance the velocity field
    # --------------------------------------------

    print("Starting simulation...")
    rms_on_steps = {}
    for step in tqdm(range(steps), desc="Simulation Progress"):

    # ---------------------------------------------------------
    # 1. Apply boundary conditions
    # ---------------------------------------------------------

        apply_boundary_conditions(
            grid,
            inlet_velocity=inlet_velocity
        )

        advect_velocity(grid, dt)

        apply_boundary_conditions(
            grid,
            inlet_velocity=inlet_velocity
        )

        diffuse_velocity(
            grid,
            dt,
            nu
        )

        apply_boundary_conditions(
            grid,
            inlet_velocity=inlet_velocity
        )

        max_before, rms_before = divergence_metrics(grid)

        solve_pressure(
            grid,
            dt,
            rho=rho
        )

        project_velocity(
            grid,
            dt,
            rho=rho
        )

        max_after, rms_after = divergence_metrics(grid)

    # Existing animation capture can remain here.

        # ---------------------------------------------------------
        # 8. Save animation frame
        # ---------------------------------------------------------

        if step % frame_interval == 0 or step == steps - 1:

            current_time = (step + 1) * dt

            state = capture_flow_state(
                grid,
                current_time
            )

            flow_states.append(state)

        # ---------------------------------------------------------
        # Print diagnostics occasionally
        # ---------------------------------------------------------

        if (step + 1) % 100 == 0:
            rms_on_steps[f"Step {step + 1}/{steps}"] = f"{rms_before:.3e} -> {rms_after:.3e}"

    data.append("RMS Divergence")
    for rms_step in rms_on_steps.keys():
        data.append(f"{rms_step}: RMS divergence {rms_on_steps[rms_step]}")

    # animate_flow(
    #     grid,
    #     flow_states,
    #     filename="flow_animation.gif",
    #     fps=20
    # )

    save_flow_plots(grid)

    flow_rates = measure_flow_rates(grid)
    for flow in flow_rates:
        data.append(flow)

    diffusion_dt_limit = diffusion_timestep_limit(grid,nu)

    data.append(
        f"Diffusion dt limit:   "
        f"{diffusion_dt_limit:.6e} s"
    )

    data.append(
        f"Current dt:            "
        f"{dt:.6e} s"
    )

    u_center, v_center = get_cell_center_velocity(grid)

    speed = np.sqrt(
        u_center**2
        + v_center**2
    )

    fluid_speed = speed[~grid.solid]

    data.append(
        f"Maximum fluid speed: "
        f"{np.nanmax(fluid_speed):.6f} m/s"
    )

    data.append(
        f"Mean fluid speed: "
        f"{np.nanmean(fluid_speed):.6f} m/s"
    )

    characteristic_length = 0.4

    reynolds_number = (
        rho
        * inlet_velocity
        * characteristic_length
        / mu
    )

    data.append(f"Density:              {rho:.6g} kg/m^3")
    data.append(f"Dynamic viscosity:    {mu:.6g} Pa.s")
    data.append(f"Kinematic viscosity:  {nu:.6g} m^2/s")
    data.append(f"Reynolds number:       {reynolds_number:.2f}")

    diffusion_number_x = nu * dt / grid.dx**2
    diffusion_number_y = nu * dt / grid.dy**2

    data.append(
        f"Diffusion number x:   "
        f"{diffusion_number_x:.4f}"
    )

    data.append(
        f"Diffusion number y:   "
        f"{diffusion_number_y:.4f}"
    )

    data.append(
        f"Combined diffusion #: "
        f"{diffusion_number_x + diffusion_number_y:.4f}"
    )
    errors, poiseuille = validate_poiseuille(
        grid,
        mean_velocity=1.0,
        x_location=1.5
    )

    for p in poiseuille:
        data.append(p)

    data.append("=======Errors=======")
    for error in errors.keys():
        data.append(f"{error}: {errors[error]}")

    write_to_output(data)


    # --------------------------------------------
    # 3. Reconstruct cell-centred velocities
    #    for visualisation only
    # --------------------------------------------
    reconstuct_cell_centred_velocities(grid)



if __name__ == "__main__":
    main()
