import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

def animate_flow(
    grid,
    states,
    filename="flow_animation.gif",
    fps=20
):
    """
    Create an animation containing:

    Left:
        velocity magnitude + velocity vectors

    Right:
        pressure field
    """

    if len(states) == 0:
        raise ValueError("No flow states were supplied.")

    # ---------------------------------------------------------
    # Determine fixed colour limits for the entire animation.
    # ---------------------------------------------------------

    max_speed = max(
        np.nanmax(state["speed"])
        for state in states
    )

    max_pressure = max(
        np.max(np.abs(state["pressure"][~grid.solid]))
        for state in states
    )

    if max_pressure == 0:
        max_pressure = 1.0

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    x, y = grid.cell_centres()

    # Plot fewer velocity arrows so the animation stays readable.
    stride = max(1, min(grid.nx, grid.ny) // 15)

    x_quiver = x[::stride, ::stride]
    y_quiver = y[::stride, ::stride]

    solid_quiver = grid.solid[::stride, ::stride]

    # Only draw arrows in fluid cells.
    fluid_quiver = ~solid_quiver

    xq = x_quiver[fluid_quiver]
    yq = y_quiver[fluid_quiver]

    # ---------------------------------------------------------
    # Initial frame
    # ---------------------------------------------------------

    first = states[0]

    speed0 = np.ma.masked_where(
        grid.solid,
        first["speed"]
    )

    pressure0 = np.ma.masked_where(
        grid.solid,
        first["pressure"]
    )

    u0 = first["u"][::stride, ::stride][fluid_quiver]
    v0 = first["v"][::stride, ::stride][fluid_quiver]

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, (ax_velocity, ax_pressure) = plt.subplots(
        1,
        2,
        figsize=(15, 5)
    )

    # ---------------------------------------------------------
    # Velocity magnitude
    # ---------------------------------------------------------

    velocity_image = ax_velocity.imshow(
        speed0,
        origin="lower",
        extent=[
            0,
            grid.width,
            0,
            grid.height
        ],
        vmin=0,
        vmax=max_speed,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal"
    )

    velocity_quiver = ax_velocity.quiver(
        xq,
        yq,
        u0,
        v0
    )

    fig.colorbar(
        velocity_image,
        ax=ax_velocity,
        label="Velocity magnitude (m/s)"
    )

    ax_velocity.set_xlabel("x (m)")
    ax_velocity.set_ylabel("y (m)")
    ax_velocity.set_xlim(0, grid.width)
    ax_velocity.set_ylim(0, grid.height)

    # Draw solid boundary.
    ax_velocity.contour(
        x,
        y,
        grid.solid.astype(float),
        levels=[0.5]
    )

    # ---------------------------------------------------------
    # Pressure
    # ---------------------------------------------------------

    pressure_image = ax_pressure.imshow(
        pressure0,
        origin="lower",
        extent=[
            0,
            grid.width,
            0,
            grid.height
        ],
        vmin=-max_pressure,
        vmax=max_pressure,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="equal"
    )

    fig.colorbar(
        pressure_image,
        ax=ax_pressure,
        label="Pressure"
    )

    ax_pressure.set_xlabel("x (m)")
    ax_pressure.set_ylabel("y (m)")
    ax_pressure.set_xlim(0, grid.width)
    ax_pressure.set_ylim(0, grid.height)

    ax_pressure.contour(
        x,
        y,
        grid.solid.astype(float),
        levels=[0.5]
    )

    # ---------------------------------------------------------
    # Animation update function
    # ---------------------------------------------------------

    def update(frame_number):

        state = states[frame_number]

        speed = np.ma.masked_where(
            grid.solid,
            state["speed"]
        )

        pressure = np.ma.masked_where(
            grid.solid,
            state["pressure"]
        )

        velocity_image.set_data(speed)
        pressure_image.set_data(pressure)

        u = state["u"][::stride, ::stride][fluid_quiver]
        v = state["v"][::stride, ::stride][fluid_quiver]

        velocity_quiver.set_UVC(u, v)

        time = state["time"]

        ax_velocity.set_title(
            f"Velocity Field — t = {time:.3f} s"
        )

        ax_pressure.set_title(
            f"Pressure Field — t = {time:.3f} s"
        )

        return (
            velocity_image,
            velocity_quiver,
            pressure_image
        )

    # ---------------------------------------------------------
    # Create animation
    # ---------------------------------------------------------
    animation = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=1000 / fps,
        blit=False
    )

    fig.tight_layout()

    writer = PillowWriter(fps=fps)

    animation.save(
        "results/"+filename,
        writer=writer,
        dpi=120
    )

    plt.close(fig)

def measure_flow_rates(grid) -> list:
    inlet_flow = np.sum(grid.u[:, 0]) * grid.dy
    outlet_flow = np.sum(grid.u[:, -1]) * grid.dy

    return [
        f"Inlet flow rate:  {inlet_flow:.6f}",
        f"Outlet flow rate: {outlet_flow:.6f}",
        f"Flow rate difference: {outlet_flow - inlet_flow:.6e}"
        ]

def get_cell_center_velocity(grid):
    """Estimate u and v at cell centres from MAC face velocities."""
    u_center = 0.5 * (grid.u[:, :-1] + grid.u[:, 1:])
    v_center = 0.5 * (grid.v[:-1, :] + grid.v[1:, :])

    # Hide velocity inside solid cells.
    u_center = u_center.copy()
    v_center = v_center.copy()

    u_center[grid.solid] = np.nan
    v_center[grid.solid] = np.nan

    return u_center, v_center


def plot_velocity_magnitude(grid, filename="velocity_magnitude.png"):
    """Plot the speed of the fluid at cell centres."""
    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)

    x, y = grid.cell_centres()

    fig, ax = plt.subplots(figsize=(12, 5))

    image = ax.pcolormesh(
        x, y, speed,
        shading="auto",
        cmap="viridis"
    )

    fig.colorbar(image, ax=ax, label="Velocity magnitude (m/s)")

    ax.set_title("Fluid Velocity Magnitude")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("results/"+filename, dpi=150)
    plt.close(fig)


def plot_velocity_vectors(grid, filename="velocity_vectors.png"):
    """Plot velocity vectors at cell centres."""
    u_center, v_center = get_cell_center_velocity(grid)
    x, y = grid.cell_centres()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw solid cells as a mask.
    solid = np.ma.masked_where(~grid.solid, grid.solid)

    ax.pcolormesh(
        x, y, solid,
        shading="auto",
        cmap="Greys",
        vmin=0,
        vmax=1
    )

    # Subsample vectors to avoid an overcrowded plot.
    stride = max(1, min(grid.nx, grid.ny) // 20)

    ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        u_center[::stride, ::stride],
        v_center[::stride, ::stride],
        scale=None
    )

    ax.set_title("Fluid Velocity Vectors")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("results/"+filename, dpi=150)
    plt.close(fig)


def plot_pressure(grid, filename="pressure.png"):
    """Plot pressure at cell centres."""
    x, y = grid.cell_centres()

    # Do not display pressure inside the solid obstacle.
    pressure = np.ma.masked_where(grid.solid, grid.p)

    fig, ax = plt.subplots(figsize=(12, 5))

    image = ax.pcolormesh(
        x, y, pressure,
        shading="auto",
        cmap="coolwarm"
    )

    fig.colorbar(image, ax=ax, label="Pressure (Pa)")

    ax.set_title("Fluid Pressure")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("results/"+filename, dpi=150)
    plt.close(fig)


def save_flow_plots(grid):
    """Save all Milestone 5 plots."""
    plot_velocity_magnitude(grid)
    plot_velocity_vectors(grid)
    plot_pressure(grid)

def capture_flow_state(grid, time):
    """
    Store one snapshot of the current flow field.
    """
    u_center, v_center = get_cell_center_velocity(grid)

    speed = np.sqrt(u_center**2 + v_center**2)

    return {
        "time": time,
        "u": u_center.copy(),
        "v": v_center.copy(),
        "speed": speed.copy(),
        "pressure": grid.p.copy(),
    }

def reconstuct_cell_centred_velocities(grid):
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