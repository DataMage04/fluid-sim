from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from grid import MACGrid
from advection import advect_velocity
from visualization import (
    save_flow_plots, 
    get_cell_center_velocity,
    measure_flow_rates,
    capture_flow_state,
    animate_flow,
)
from boundary_conditions import apply_boundary_conditions
from solver import (
    compute_divergence,
    solve_pressure,
    project_velocity,
    divergence_metrics,
)
from diffusion import (
    diffuse_velocity,
    diffusion_timestep_limit,
)
from validation import validate_poiseuille

def main():
    # --------------------------------------------
    # 1. Create the simulation domain
    # --------------------------------------------
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

    apply_boundary_conditions(grid, inlet_velocity)

    # --------------------------------------------
    # 2. Advance the velocity field
    # --------------------------------------------

    for step in tqdm(range(steps)):

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

            print(
                f"Step {step + 1}/{steps}: "
                f"RMS divergence "
                f"{rms_before:.3e} -> {rms_after:.3e}"
            )

    animate_flow(
        grid,
        flow_states,
        filename="flow_animation.gif",
        fps=20
    )

    save_flow_plots(grid)

    measure_flow_rates(grid)

    diffusion_dt_limit = diffusion_timestep_limit(grid,nu)

    print(
        f"Diffusion dt limit:   "
        f"{diffusion_dt_limit:.6e} s"
    )

    print(
        f"Current dt:            "
        f"{dt:.6e} s"
    )

    u_center, v_center = get_cell_center_velocity(grid)

    speed = np.sqrt(
        u_center**2
        + v_center**2
    )

    fluid_speed = speed[~grid.solid]

    print(
        f"Maximum fluid speed: "
        f"{np.nanmax(fluid_speed):.6f} m/s"
    )

    print(
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

    print(f"Density:              {rho:.6g} kg/m^3")
    print(f"Dynamic viscosity:    {mu:.6g} Pa.s")
    print(f"Kinematic viscosity:  {nu:.6g} m^2/s")
    print(f"Reynolds number:       {reynolds_number:.2f}")

    diffusion_number_x = nu * dt / grid.dx**2
    diffusion_number_y = nu * dt / grid.dy**2

    print(
        f"Diffusion number x:   "
        f"{diffusion_number_x:.4f}"
    )

    print(
        f"Diffusion number y:   "
        f"{diffusion_number_y:.4f}"
    )

    print(
        f"Combined diffusion #: "
        f"{diffusion_number_x + diffusion_number_y:.4f}"
    )
    errors = validate_poiseuille(
        grid,
        mean_velocity=1.0,
        x_location=1.5
    )
    print("Errors: ", errors)


    # --------------------------------------------
    # 3. Reconstruct cell-centred velocities
    #    for visualisation only
    # --------------------------------------------

    u_center = 0.5 * (
        grid.u[:, :-1] + grid.u[:, 1:]
    )

    v_center = 0.5 * (
        grid.v[:-1, :] + grid.v[1:, :]
    )

    speed = np.sqrt(
        u_center**2 + v_center**2
    )

    X, Y = grid.cell_centres()

    # Mask solid cells in the visualisation.
    speed_plot = np.ma.array(
        speed,
        mask=grid.solid,
    )

    # --------------------------------------------
    # 4. Plot velocity magnitude and vectors
    # --------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    contour = ax.pcolormesh(
        X,
        Y,
        speed_plot,
        shading="nearest",
        cmap="viridis",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Velocity magnitude (m/s)",
    )

    # Display the solid obstruction.
    ax.contourf(
        X,
        Y,
        grid.solid.astype(float),
        levels=[0.5, 1.5],
        colors=["dimgray"],
    )

    # Plot a subset of velocity vectors.
    stride = 3

    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        np.ma.array(
            u_center[::stride, ::stride],
            mask=grid.solid[::stride, ::stride],
        ),
        np.ma.array(
            v_center[::stride, ::stride],
            mask=grid.solid[::stride, ::stride],
        ),
        color="white",
        scale=20,
    )

    ax.set_title(
        "2D MAC Grid: Semi-Lagrangian Advection"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
