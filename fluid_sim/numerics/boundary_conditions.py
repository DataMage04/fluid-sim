import numpy as np


def apply_poiseuille_inlet(grid, mean_velocity):
    """
    Apply a fully developed parabolic velocity profile
    at the left inlet.
    """
    y = (np.arange(grid.ny) + 0.5) * grid.dy
    eta = y / grid.height

    inlet_profile = (
        6.0
        * mean_velocity
        * eta
        * (1.0 - eta)
    )

    grid.u[:, 0] = inlet_profile


def apply_uniform_inlet(grid, velocity):
    """Apply a uniform horizontal velocity at the left inlet."""
    grid.u[:, 0] = velocity


def apply_boundary_conditions(
    grid,
    inlet_velocity=1.0,
    inlet_profile="poiseuille",
):
    """Apply inlet, outlet, wall, and solid-cell conditions."""
    if inlet_profile == "poiseuille":
        apply_poiseuille_inlet(
            grid,
            mean_velocity=inlet_velocity,
        )
    elif inlet_profile == "uniform":
        apply_uniform_inlet(
            grid,
            velocity=inlet_velocity,
        )
    else:
        raise ValueError(
            "inlet_profile must be either 'poiseuille' or 'uniform'."
        )

    # Outlet: approximate zero horizontal velocity gradient.
    grid.u[:, -1] = grid.u[:, -2]

    # Top and bottom walls: no normal flow.
    grid.v[0, :] = 0.0
    grid.v[-1, :] = 0.0

    # Extrapolate vertical velocity near inlet and outlet.
    if grid.nx > 1:
        grid.v[:, 0] = grid.v[:, 1]
        grid.v[:, -1] = grid.v[:, -2]

    # u[j, i] lies between cells i-1 and i.
    u_blocked = np.zeros_like(grid.u, dtype=bool)
    u_blocked[:, 1:-1] = (
        grid.solid[:, :-1]
        | grid.solid[:, 1:]
    )

    # v[j, i] lies between cells j-1 and j.
    v_blocked = np.zeros_like(grid.v, dtype=bool)
    v_blocked[1:-1, :] = (
        grid.solid[:-1, :]
        | grid.solid[1:, :]
    )

    # Prevent normal flow across solid faces.
    grid.u[u_blocked] = 0.0
    grid.v[v_blocked] = 0.0

    # Solid cells do not participate in the pressure solve.
    grid.p[grid.solid] = 0.0
