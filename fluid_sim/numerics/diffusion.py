import numpy as np


def diffusion_timestep_limit(grid, nu):
    """
    Maximum stable timestep for explicit 2D diffusion.

    Stability condition:

        nu * dt * (1/dx^2 + 1/dy^2) <= 1/2
    """

    if nu <= 0.0:
        return np.inf

    return 0.5 / (
        nu * (
            1.0 / grid.dx**2
            + 1.0 / grid.dy**2
        )
    )


def diffuse_velocity(grid, dt, nu):
    """
    Apply explicit viscous diffusion to MAC-grid velocities.

    No-slip boundary treatment:

    - Normal velocity on a wall is zero.
    - Tangential velocity uses an antisymmetric ghost value:

          u_ghost = -u_fluid

      so the interpolated velocity at the wall is zero.

    The same idea is used at solid-obstacle surfaces.
    """

    if nu < 0.0:
        raise ValueError("Kinematic viscosity cannot be negative.")

    if nu == 0.0:
        return

    # ---------------------------------------------------------
    # Stability check
    # ---------------------------------------------------------

    dt_limit = diffusion_timestep_limit(grid, nu)

    if dt > dt_limit:
        raise ValueError(
            "Explicit diffusion timestep is unstable.\n"
            f"Current dt = {dt:.6e}\n"
            f"Maximum stable dt = {dt_limit:.6e}"
        )

    u_old = grid.u.copy()
    v_old = grid.v.copy()

    # ---------------------------------------------------------
    # Determine velocity faces blocked by solid cells
    # ---------------------------------------------------------

    u_blocked = np.zeros_like(grid.u, dtype=bool)

    u_blocked[:, 1:-1] = (
        grid.solid[:, :-1]
        | grid.solid[:, 1:]
    )

    v_blocked = np.zeros_like(grid.v, dtype=bool)

    v_blocked[1:-1, :] = (
        grid.solid[:-1, :]
        | grid.solid[1:, :]
    )

    dx2 = grid.dx**2
    dy2 = grid.dy**2

    # =========================================================
    # U VELOCITY
    # =========================================================
    #
    # u is stored on vertical faces.
    #
    # Do not diffuse inlet/outlet columns because those are
    # controlled by the inlet/outlet boundary conditions.
    #
    # We DO diffuse the top/bottom u rows because they are
    # half a cell away from the actual wall.
    # =========================================================

    uc = u_old[:, 1:-1]

    # ---------------------------------------------------------
    # x-direction neighbours
    #
    # u is normal to vertical surfaces.
    # Blocked neighbouring u faces therefore represent a
    # zero normal velocity at the solid surface.
    # ---------------------------------------------------------

    uw = np.where(
        u_blocked[:, :-2],
        0.0,
        u_old[:, :-2]
    )

    ue = np.where(
        u_blocked[:, 2:],
        0.0,
        u_old[:, 2:]
    )

    # ---------------------------------------------------------
    # y-direction neighbours
    #
    # u is tangential to horizontal walls.
    #
    # Use antisymmetric ghost values:
    #
    #     u_ghost = -u
    #
    # to obtain zero velocity exactly at the wall.
    # ---------------------------------------------------------

    us = np.empty_like(uc)
    un = np.empty_like(uc)

    # Bottom domain wall.
    us[0, :] = -uc[0, :]

    # Interior southern neighbours.
    us[1:, :] = np.where(
        u_blocked[:-1, 1:-1],
        -uc[1:, :],
        u_old[:-1, 1:-1]
    )

    # Top domain wall.
    un[-1, :] = -uc[-1, :]

    # Interior northern neighbours.
    un[:-1, :] = np.where(
        u_blocked[1:, 1:-1],
        -uc[:-1, :],
        u_old[1:, 1:-1]
    )

    laplacian_u = (
        (ue - 2.0 * uc + uw) / dx2
        +
        (un - 2.0 * uc + us) / dy2
    )

    u_candidate = (
        uc
        + nu * dt * laplacian_u
    )

    u_active = ~u_blocked[:, 1:-1]

    u_region = grid.u[:, 1:-1]

    u_region[u_active] = u_candidate[u_active]

    # =========================================================
    # V VELOCITY
    # =========================================================
    #
    # v is stored on horizontal faces.
    #
    # v[0, :] and v[-1, :] lie directly on the top/bottom
    # walls and remain zero.
    #
    # Columns 0 and -1 are left to your existing inlet/outlet
    # zero-gradient boundary condition.
    # =========================================================

    vc = v_old[1:-1, 1:-1]

    # ---------------------------------------------------------
    # x-direction neighbours
    #
    # v is tangential to vertical solid surfaces.
    # Therefore use antisymmetric ghost values there.
    # ---------------------------------------------------------

    vw = np.where(
        v_blocked[1:-1, :-2],
        -vc,
        v_old[1:-1, :-2]
    )

    ve = np.where(
        v_blocked[1:-1, 2:],
        -vc,
        v_old[1:-1, 2:]
    )

    # ---------------------------------------------------------
    # y-direction neighbours
    #
    # v is normal to horizontal surfaces.
    #
    # The wall-normal velocity is therefore zero.
    # ---------------------------------------------------------

    vs = np.where(
        v_blocked[:-2, 1:-1],
        0.0,
        v_old[:-2, 1:-1]
    )

    vn = np.where(
        v_blocked[2:, 1:-1],
        0.0,
        v_old[2:, 1:-1]
    )

    laplacian_v = (
        (ve - 2.0 * vc + vw) / dx2
        +
        (vn - 2.0 * vc + vs) / dy2
    )

    v_candidate = (
        vc
        + nu * dt * laplacian_v
    )

    v_active = ~v_blocked[1:-1, 1:-1]

    v_region = grid.v[1:-1, 1:-1]

    v_region[v_active] = v_candidate[v_active]

    # ---------------------------------------------------------
    # Solid-normal faces remain impermeable
    # ---------------------------------------------------------

    grid.u[u_blocked] = 0.0
    grid.v[v_blocked] = 0.0

    # Horizontal wall-normal velocity
    grid.v[0, :] = 0.0
    grid.v[-1, :] = 0.0

