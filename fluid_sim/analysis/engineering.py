from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fluid_sim.analysis.visualization import get_cell_center_velocity


@dataclass(frozen=True)
class EngineeringResults:
    """Scalar engineering quantities extracted from a completed flow field."""

    upstream_x: float
    downstream_x: float
    sample_width: float
    upstream_pressure: float
    downstream_pressure: float
    pressure_drop: float
    dynamic_pressure: float
    upstream_cp: float
    downstream_cp: float
    pressure_drop_coefficient: float
    max_speed: float
    max_speed_x: float
    max_speed_y: float
    min_streamwise_velocity: float
    min_streamwise_velocity_x: float
    min_streamwise_velocity_y: float
    reverse_flow_fraction: float
    pressure_drag_per_unit_depth: float | None = None
    pressure_lift_per_unit_depth: float | None = None
    pressure_drag_coefficient: float | None = None
    pressure_lift_coefficient: float | None = None
    wake_length: float | None = None
    wake_end_x: float | None = None
    analytical_pressure_gradient: float | None = None
    analytical_pressure_drop: float | None = None
    pressure_drop_relative_error: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _x_band_fluid_mask(grid, x_location: float, width: float) -> np.ndarray:
    """Return fluid cells whose centres lie inside a vertical sampling band."""
    if width <= 0.0:
        raise ValueError("Pressure sampling width must be positive.")
    if not 0.0 <= x_location <= grid.width:
        raise ValueError("Pressure sampling location must lie inside the domain.")

    x_centres = (np.arange(grid.nx) + 0.5) * grid.dx
    columns = np.abs(x_centres - x_location) <= 0.5 * width
    mask = (~grid.solid) & columns[None, :]

    if not np.any(mask):
        raise ValueError(
            "Pressure sampling band contains no fluid cell centres. "
            "Increase sample_width or move the sampling location."
        )

    return mask


def mean_pressure_in_x_band(grid, x_location: float, width: float) -> float:
    """Mean fluid pressure in a vertical band centred on x_location."""
    mask = _x_band_fluid_mask(grid, x_location, width)
    return float(np.mean(grid.p[mask]))


def pressure_drop(
    grid,
    upstream_x: float,
    downstream_x: float,
    sample_width: float,
) -> tuple[float, float, float]:
    """Return upstream pressure, downstream pressure, and p_upstream - p_downstream."""
    if upstream_x >= downstream_x:
        raise ValueError("upstream_x must be smaller than downstream_x.")

    upstream_pressure = mean_pressure_in_x_band(
        grid,
        upstream_x,
        sample_width,
    )
    downstream_pressure = mean_pressure_in_x_band(
        grid,
        downstream_x,
        sample_width,
    )

    return (
        upstream_pressure,
        downstream_pressure,
        upstream_pressure - downstream_pressure,
    )


def dynamic_pressure(density: float, reference_velocity: float) -> float:
    """Return q = 1/2 rho U^2."""
    if density <= 0.0:
        raise ValueError("Density must be positive.")
    if reference_velocity <= 0.0:
        raise ValueError("Reference velocity must be positive.")
    return 0.5 * density * reference_velocity**2


def pressure_coefficient(
    pressure: float,
    reference_pressure: float,
    density: float,
    reference_velocity: float,
) -> float:
    """Return Cp = (p - p_ref) / (1/2 rho U_ref^2)."""
    return (
        pressure - reference_pressure
    ) / dynamic_pressure(density, reference_velocity)


def streamwise_pressure_profile(
    grid,
    *,
    density: float,
    reference_velocity: float,
    reference_pressure: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, cross-section mean pressure, and mean-pressure Cp profiles."""
    q = dynamic_pressure(density, reference_velocity)
    x_centres = (np.arange(grid.nx) + 0.5) * grid.dx
    mean_pressure = np.full(grid.nx, np.nan, dtype=float)

    for i in range(grid.nx):
        fluid_rows = ~grid.solid[:, i]
        if np.any(fluid_rows):
            mean_pressure[i] = float(np.mean(grid.p[fluid_rows, i]))

    cp = (mean_pressure - reference_pressure) / q
    return x_centres, mean_pressure, cp


def maximum_velocity_location(grid) -> tuple[float, float, float]:
    """Return maximum cell-centred speed and its (x, y) coordinates."""
    u_center, v_center = get_cell_center_velocity(grid)
    speed = np.sqrt(u_center**2 + v_center**2)

    if np.all(np.isnan(speed)):
        raise ValueError("The grid contains no fluid cells.")

    flat_index = int(np.nanargmax(speed))
    j, i = np.unravel_index(flat_index, speed.shape)
    x, y = grid.cell_centres()

    return float(speed[j, i]), float(x[j, i]), float(y[j, i])


def reverse_flow_metrics(
    grid,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[float, float, float, float]:
    """Return minimum u, its location, and the fraction of fluid cells with reverse flow."""
    if tolerance < 0.0:
        raise ValueError("Reverse-flow tolerance cannot be negative.")

    u_center, _ = get_cell_center_velocity(grid)
    fluid_u = u_center[~grid.solid]
    if fluid_u.size == 0:
        raise ValueError("The grid contains no fluid cells.")

    flat_index = int(np.nanargmin(u_center))
    j, i = np.unravel_index(flat_index, u_center.shape)
    x, y = grid.cell_centres()
    reverse_fraction = float(np.count_nonzero(fluid_u < -tolerance) / fluid_u.size)

    return (
        float(u_center[j, i]),
        float(x[j, i]),
        float(y[j, i]),
        reverse_fraction,
    )


def pressure_force_on_solids(grid) -> tuple[float, float]:
    """Integrate pressure force on grid-aligned solid surfaces per unit depth.

    Returns (drag, lift) exerted by the fluid on all internal solid boundaries.
    Pressure is sampled from the fluid cell immediately adjacent to each
    solid-fluid interface.
    """
    if not np.any(grid.solid):
        return 0.0, 0.0

    drag = 0.0
    lift = 0.0

    solid_j, solid_i = np.nonzero(grid.solid)
    for j, i in zip(solid_j, solid_i):
        # Left face: outward solid normal points in -x, so pressure force is +x.
        if i > 0 and not grid.solid[j, i - 1]:
            drag += grid.p[j, i - 1] * grid.dy

        # Right face: outward solid normal points in +x, so pressure force is -x.
        if i < grid.nx - 1 and not grid.solid[j, i + 1]:
            drag -= grid.p[j, i + 1] * grid.dy

        # Bottom face: outward solid normal points in -y, so pressure force is +y.
        if j > 0 and not grid.solid[j - 1, i]:
            lift += grid.p[j - 1, i] * grid.dx

        # Top face: outward solid normal points in +y, so pressure force is -y.
        if j < grid.ny - 1 and not grid.solid[j + 1, i]:
            lift -= grid.p[j + 1, i] * grid.dx

    return float(drag), float(lift)


def wake_extent(
    grid,
    *,
    wake_start_x: float,
    velocity_tolerance: float = 1.0e-3,
) -> tuple[float, float]:
    """Return reverse-flow wake length and downstream end location.

    The wake is defined by fluid cells downstream of wake_start_x whose
    streamwise velocity is more negative than velocity_tolerance.
    """
    if not 0.0 <= wake_start_x <= grid.width:
        raise ValueError("wake_start_x must lie inside the domain.")
    if velocity_tolerance < 0.0:
        raise ValueError("velocity_tolerance cannot be negative.")

    u_center, _ = get_cell_center_velocity(grid)
    x, _ = grid.cell_centres()
    reverse = (
        (~grid.solid)
        & (x > wake_start_x)
        & (u_center < -velocity_tolerance)
    )

    if not np.any(reverse):
        return 0.0, float(wake_start_x)

    last_center = float(np.max(x[reverse]))
    wake_end_x = min(grid.width, last_center + 0.5 * grid.dx)
    return float(wake_end_x - wake_start_x), float(wake_end_x)


def analytical_poiseuille_pressure_gradient(
    dynamic_viscosity: float,
    mean_velocity: float,
    channel_height: float,
) -> float:
    """Magnitude of -dp/dx for fully developed plane Poiseuille flow."""
    if dynamic_viscosity < 0.0:
        raise ValueError("Dynamic viscosity cannot be negative.")
    if channel_height <= 0.0:
        raise ValueError("Channel height must be positive.")

    return (
        12.0
        * dynamic_viscosity
        * mean_velocity
        / channel_height**2
    )


def analytical_poiseuille_pressure_drop(
    dynamic_viscosity: float,
    mean_velocity: float,
    channel_height: float,
    distance: float,
) -> float:
    """Analytical pressure loss over distance for plane Poiseuille flow."""
    if distance < 0.0:
        raise ValueError("Pressure-drop distance cannot be negative.")

    return analytical_poiseuille_pressure_gradient(
        dynamic_viscosity,
        mean_velocity,
        channel_height,
    ) * distance


def compute_engineering_results(
    grid,
    *,
    density: float,
    reference_velocity: float,
    upstream_x: float,
    downstream_x: float,
    sample_width: float,
    reference_pressure: float = 0.0,
    analytical_pressure_gradient: float | None = None,
    reference_length: float | None = None,
    wake_start_x: float | None = None,
) -> EngineeringResults:
    """Calculate the standard Milestone 9 engineering result set."""
    upstream_pressure, downstream_pressure, delta_p = pressure_drop(
        grid,
        upstream_x=upstream_x,
        downstream_x=downstream_x,
        sample_width=sample_width,
    )

    q = dynamic_pressure(density, reference_velocity)
    max_speed, max_speed_x, max_speed_y = maximum_velocity_location(grid)
    min_u, min_u_x, min_u_y, reverse_fraction = reverse_flow_metrics(grid)

    pressure_drag = None
    pressure_lift = None
    pressure_cd = None
    pressure_cl = None
    wake_length = None
    wake_end_x = None

    if reference_length is not None:
        if reference_length <= 0.0:
            raise ValueError("reference_length must be positive.")
        pressure_drag, pressure_lift = pressure_force_on_solids(grid)
        force_scale = q * reference_length
        pressure_cd = pressure_drag / force_scale
        pressure_cl = pressure_lift / force_scale

    if wake_start_x is not None:
        wake_length, wake_end_x = wake_extent(
            grid,
            wake_start_x=wake_start_x,
            velocity_tolerance=max(1.0e-12, 1.0e-3 * reference_velocity),
        )

    analytical_drop = None
    relative_error = None
    if analytical_pressure_gradient is not None:
        distance = downstream_x - upstream_x
        analytical_drop = analytical_pressure_gradient * distance
        if analytical_drop != 0.0:
            relative_error = abs(delta_p - analytical_drop) / abs(analytical_drop)
        else:
            relative_error = 0.0 if delta_p == 0.0 else float("inf")

    return EngineeringResults(
        upstream_x=upstream_x,
        downstream_x=downstream_x,
        sample_width=sample_width,
        upstream_pressure=upstream_pressure,
        downstream_pressure=downstream_pressure,
        pressure_drop=delta_p,
        dynamic_pressure=q,
        upstream_cp=pressure_coefficient(
            upstream_pressure,
            reference_pressure,
            density,
            reference_velocity,
        ),
        downstream_cp=pressure_coefficient(
            downstream_pressure,
            reference_pressure,
            density,
            reference_velocity,
        ),
        pressure_drop_coefficient=delta_p / q,
        max_speed=max_speed,
        max_speed_x=max_speed_x,
        max_speed_y=max_speed_y,
        min_streamwise_velocity=min_u,
        min_streamwise_velocity_x=min_u_x,
        min_streamwise_velocity_y=min_u_y,
        reverse_flow_fraction=reverse_fraction,
        pressure_drag_per_unit_depth=pressure_drag,
        pressure_lift_per_unit_depth=pressure_lift,
        pressure_drag_coefficient=pressure_cd,
        pressure_lift_coefficient=pressure_cl,
        wake_length=wake_length,
        wake_end_x=wake_end_x,
        analytical_pressure_gradient=analytical_pressure_gradient,
        analytical_pressure_drop=analytical_drop,
        pressure_drop_relative_error=relative_error,
    )
