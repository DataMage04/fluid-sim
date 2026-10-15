from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm


class _ProgressPillowWriter(PillowWriter):
    """Pillow GIF writer with visible progress during final compression.

    Matplotlib renders all frames before Pillow performs the potentially slow
    final GIF compression/write step. Pillow does not expose byte-level
    progress callbacks, so an elapsed-time indicator is shown while that final
    blocking operation is running.
    """

    def finish(self):
        stop_event = Event()
        start_time = perf_counter()

        def show_finalizing_status():
            with tqdm(
                total=None,
                desc="Finalizing GIF",
                unit="s",
                bar_format="{desc}: {elapsed} elapsed",
                dynamic_ncols=True,
                leave=True,
            ) as status:
                while not stop_event.wait(0.5):
                    status.refresh()

        reporter = Thread(
            target=show_finalizing_status,
            name="gif-finalization-progress",
            daemon=True,
        )
        reporter.start()

        try:
            super().finish()
        finally:
            stop_event.set()
            reporter.join()

        elapsed = perf_counter() - start_time
        tqdm.write(
            f"GIF finalized in {elapsed:.1f} s: {self.outfile}"
        )


def _output_path(filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _draw_solid_outline(ax, x, y, solid) -> None:
    if np.any(solid):
        ax.contour(
            x,
            y,
            solid.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.0,
        )


def _symmetric_pressure_limit(pressure, solid) -> float:
    """Return a finite symmetric colour limit for the fluid pressure field."""
    fluid_pressure = np.asarray(pressure)[~solid]
    if fluid_pressure.size == 0:
        return 1.0

    limit = float(np.nanmax(np.abs(fluid_pressure)))
    if not np.isfinite(limit) or limit <= np.finfo(float).eps:
        return 1.0
    return limit


def animate_flow(
    grid,
    states,
    filename: str | Path = "results/flow_animation.gif",
    fps=20,
):
    """Create a velocity/pressure animation for stored flow states.

    Velocity uses a fixed colour scale so speed can be compared directly
    between frames. Pressure uses a symmetric scale that adapts to each frame.
    This prevents a large startup pressure transient from washing out the much
    smaller pressure variations once the flow settles.

    Progress is reported in two stages:

    1. Rendering the individual animation frames.
    2. Final Pillow GIF compression and file writing.

    Pillow does not expose a percentage for the second stage, so an elapsed
    timer is shown while finalization is in progress.
    """
    if len(states) == 0:
        raise ValueError("No flow states were supplied.")

    max_speed = max(np.nanmax(state["speed"]) for state in states)

    x, y = grid.cell_centres()
    stride = max(1, min(grid.nx, grid.ny) // 15)

    x_quiver = x[::stride, ::stride]
    y_quiver = y[::stride, ::stride]
    solid_quiver = grid.solid[::stride, ::stride]
    fluid_quiver = ~solid_quiver

    xq = x_quiver[fluid_quiver]
    yq = y_quiver[fluid_quiver]

    first = states[0]
    speed0 = np.ma.masked_where(grid.solid, first["speed"])
    pressure0 = np.ma.masked_where(grid.solid, first["pressure"])
    pressure_limit0 = _symmetric_pressure_limit(
        first["pressure"],
        grid.solid,
    )
    u0 = first["u"][::stride, ::stride][fluid_quiver]
    v0 = first["v"][::stride, ::stride][fluid_quiver]

    fig, (ax_velocity, ax_pressure) = plt.subplots(
        1,
        2,
        figsize=(15, 5),
    )

    velocity_image = ax_velocity.imshow(
        speed0,
        origin="lower",
        extent=[0, grid.width, 0, grid.height],
        vmin=0,
        vmax=max_speed,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    velocity_quiver = ax_velocity.quiver(xq, yq, u0, v0)
    fig.colorbar(
        velocity_image,
        ax=ax_velocity,
        label="Velocity magnitude (m/s)",
    )
    ax_velocity.set_xlabel("x (m)")
    ax_velocity.set_ylabel("y (m)")
    ax_velocity.set_xlim(0, grid.width)
    ax_velocity.set_ylim(0, grid.height)
    _draw_solid_outline(ax_velocity, x, y, grid.solid)

    pressure_image = ax_pressure.imshow(
        pressure0,
        origin="lower",
        extent=[0, grid.width, 0, grid.height],
        vmin=-pressure_limit0,
        vmax=pressure_limit0,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="equal",
    )
    pressure_colorbar = fig.colorbar(
        pressure_image,
        ax=ax_pressure,
        label="Pressure (Pa)",
    )
    ax_pressure.set_xlabel("x (m)")
    ax_pressure.set_ylabel("y (m)")
    ax_pressure.set_xlim(0, grid.width)
    ax_pressure.set_ylim(0, grid.height)
    _draw_solid_outline(ax_pressure, x, y, grid.solid)

    def update(frame_number):
        state = states[frame_number]
        speed = np.ma.masked_where(grid.solid, state["speed"])
        pressure = np.ma.masked_where(grid.solid, state["pressure"])

        velocity_image.set_data(speed)
        pressure_image.set_data(pressure)

        pressure_limit = _symmetric_pressure_limit(
            state["pressure"],
            grid.solid,
        )
        pressure_image.set_clim(-pressure_limit, pressure_limit)
        pressure_colorbar.update_normal(pressure_image)

        u = state["u"][::stride, ::stride][fluid_quiver]
        v = state["v"][::stride, ::stride][fluid_quiver]
        velocity_quiver.set_UVC(u, v)

        time = state["time"]
        ax_velocity.set_title(f"Velocity Field — t = {time:.3f} s")
        ax_pressure.set_title(f"Pressure Field — t = {time:.3f} s")

        return velocity_image, velocity_quiver, pressure_image

    animation = FuncAnimation(
        fig,
        update,
        frames=range(len(states)),
        interval=1000 / fps,
        blit=False,
    )

    fig.tight_layout()

    frame_progress = tqdm(
        total=len(states),
        desc="Rendering animation",
        unit="frame",
        dynamic_ncols=True,
    )

    def update_save_progress(frame_number, total_frames):
        completed = min(frame_number + 1, total_frames)
        if completed > frame_progress.n:
            frame_progress.update(completed - frame_progress.n)

        if completed >= total_frames and not frame_progress.disable:
            frame_progress.set_description("Frames rendered")
            frame_progress.close()

    try:
        animation.save(
            _output_path(filename),
            writer=_ProgressPillowWriter(fps=fps),
            dpi=120,
            progress_callback=update_save_progress,
        )
    finally:
        frame_progress.close()
        plt.close(fig)


def flow_rate_metrics(grid) -> dict[str, float]:
    """Return inlet/outlet volumetric flow rates for a 2D unit-depth channel."""
    inlet = float(np.sum(grid.u[:, 0]) * grid.dy)
    outlet = float(np.sum(grid.u[:, -1]) * grid.dy)
    difference = outlet - inlet
    relative_difference = (
        difference / inlet if abs(inlet) > np.finfo(float).eps else 0.0
    )
    return {
        "inlet": inlet,
        "outlet": outlet,
        "difference": difference,
        "relative_difference": relative_difference,
    }


def measure_flow_rates(grid) -> list[str]:
    """Return backwards-compatible formatted flow-rate diagnostics."""
    metrics = flow_rate_metrics(grid)
    return [
        f"Inlet flow rate:  {metrics['inlet']:.6f}",
        f"Outlet flow rate: {metrics['outlet']:.6f}",
        f"Flow rate difference: {metrics['difference']:.6e}",
    ]


def get_cell_center_velocity(grid):
    """Estimate velocity components at MAC cell centres."""
    u_center = 0.5 * (grid.u[:, :-1] + grid.u[:, 1:])
    v_center = 0.5 * (grid.v[:-1, :] + grid.v[1:, :])

    u_center = u_center.copy()
    v_center = v_center.copy()
    u_center[grid.solid] = np.nan
    v_center[grid.solid] = np.nan
    return u_center, v_center


def plot_velocity_magnitude(
    grid,
    filename: str | Path = "results/velocity_magnitude.png",
):
    """Save the fluid speed field at cell centres."""
    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)
    x, y = grid.cell_centres()

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.pcolormesh(
        x,
        y,
        speed,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Velocity magnitude (m/s)")
    ax.set_title("Fluid Velocity Magnitude")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    _draw_solid_outline(ax, x, y, grid.solid)

    fig.tight_layout()
    fig.savefig(_output_path(filename), dpi=150)
    plt.close(fig)


def plot_velocity_vectors(
    grid,
    filename: str | Path = "results/velocity_vectors.png",
):
    """Save cell-centred velocity vectors."""
    u_center, v_center = get_cell_center_velocity(grid)
    x, y = grid.cell_centres()

    fig, ax = plt.subplots(figsize=(12, 5))

    if np.any(grid.solid):
        solid = np.ma.masked_where(~grid.solid, grid.solid)
        ax.pcolormesh(
            x,
            y,
            solid,
            shading="auto",
            cmap="Greys",
            vmin=0,
            vmax=1,
        )

    stride = max(1, min(grid.nx, grid.ny) // 20)
    ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        u_center[::stride, ::stride],
        v_center[::stride, ::stride],
        scale=None,
    )

    ax.set_title("Fluid Velocity Vectors")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(_output_path(filename), dpi=150)
    plt.close(fig)


def plot_pressure(
    grid,
    filename: str | Path = "results/pressure.png",
):
    """Save the cell-centred pressure field."""
    x, y = grid.cell_centres()
    pressure = np.ma.masked_where(grid.solid, grid.p)

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.pcolormesh(
        x,
        y,
        pressure,
        shading="auto",
        cmap="coolwarm",
    )
    fig.colorbar(image, ax=ax, label="Pressure (Pa)")
    ax.set_title("Fluid Pressure")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    _draw_solid_outline(ax, x, y, grid.solid)

    fig.tight_layout()
    fig.savefig(_output_path(filename), dpi=150)
    plt.close(fig)


def save_flow_plots(grid, output_directory: str | Path = "results") -> None:
    """Save the standard velocity and pressure plots."""
    output_directory = Path(output_directory)
    plot_velocity_magnitude(grid, output_directory / "velocity_magnitude.png")
    plot_velocity_vectors(grid, output_directory / "velocity_vectors.png")
    plot_pressure(grid, output_directory / "pressure.png")


def capture_flow_state(grid, time):
    """Store one snapshot of the current flow field."""
    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)

    return {
        "time": time,
        "u": u_center.copy(),
        "v": v_center.copy(),
        "speed": speed.copy(),
        "pressure": grid.p.copy(),
    }


def reconstruct_cell_centred_velocities(grid) -> None:
    """Display a combined cell-centred velocity magnitude/vector plot."""
    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)
    x, y = grid.cell_centres()

    speed_plot = np.ma.array(speed, mask=grid.solid)

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.pcolormesh(
        x,
        y,
        speed_plot,
        shading="nearest",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Velocity magnitude (m/s)")

    if np.any(grid.solid):
        ax.contourf(
            x,
            y,
            grid.solid.astype(float),
            levels=[0.5, 1.5],
            colors=["dimgray"],
        )

    stride = 3
    ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
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

    ax.set_title("2D MAC Grid Velocity Field")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect("equal")

    fig.tight_layout()
    plt.show()


# Backwards compatibility for the original misspelled public helper.
def reconstuct_cell_centred_velocities(grid) -> None:
    reconstruct_cell_centred_velocities(grid)