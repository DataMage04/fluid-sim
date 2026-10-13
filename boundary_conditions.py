
import numpy as np


def apply_boundary_conditions(grid, inlet_velocity=1.0):
    """Apply channel, inlet, outlet, and solid-cell conditions."""

    # Inlet: prescribe horizontal velocity.
    grid.u[:, 0] = inlet_velocity

    # Outlet: approximate zero horizontal velocity gradient.
    grid.u[:, -1] = grid.u[:, -2]

    # Top and bottom walls:
    # vertical velocity is zero at the walls.
    grid.v[0, :] = 0.0
    grid.v[-1, :] = 0.0

    # Approximate no-slip tangential velocity at walls.
    grid.u[0, :] = 0.0
    grid.u[-1, :] = 0.0

    # Extrapolate vertical velocity near inlet and outlet.
    if grid.nx > 1:
        grid.v[:, 0] = grid.v[:, 1]
        grid.v[:, -1] = grid.v[:, -2]

    # Identify horizontal faces adjacent to solid cells.
    # u[j, i] lies between cells i-1 and i.
    u_blocked = np.zeros_like(grid.u, dtype=bool)

    u_blocked[:, 1:-1] = (
        grid.solid[:, :-1]
        | grid.solid[:, 1:]
    )

    # Identify vertical faces adjacent to solid cells.
    # v[j, i] lies between cells j-1 and j.
    v_blocked = np.zeros_like(grid.v, dtype=bool)

    v_blocked[1:-1, :] = (
        grid.solid[:-1, :]
        | grid.solid[1:, :]
    )

    # Prevent normal flow across solid faces.
    grid.u[u_blocked] = 0.0
    grid.v[v_blocked] = 0.0

    # Ensure solid cells do not retain pressure values
    # from previous iterations.
    grid.p[grid.solid] = 0.0
